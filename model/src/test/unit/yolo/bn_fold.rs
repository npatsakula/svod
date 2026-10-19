use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::{BatchNorm2d, Layer, Module, StateDict};
use test_case::test_case;

use crate::yolo::loader::fold_batchnorm;
use crate::yolo::{YOLO_BN_EPS, YoloConv};

fn ramp(n: usize, scale: f32, offset: f32) -> Vec<f32> {
    (0..n).map(|i| ((i * 7919 % 97) as f32 / 97.0 - 0.5) * scale + offset).collect()
}

pub(super) fn unfolded_state(cin: usize, cout: usize, k: usize) -> StateDict {
    let mut sd = StateDict::new();
    let t = |data: Vec<f32>, shape: &[isize]| Tensor::from_slice(data).try_reshape(shape.to_vec()).unwrap();
    sd.insert(
        "conv.weight".into(),
        t(ramp(cout * cin * k * k, 0.2, 0.0), &[cout as isize, cin as isize, k as isize, k as isize]),
    );
    sd.insert("bn.weight".into(), t(ramp(cout, 0.5, 1.0), &[cout as isize]));
    sd.insert("bn.bias".into(), t(ramp(cout, 0.3, 0.0), &[cout as isize]));
    sd.insert("bn.running_mean".into(), t(ramp(cout, 0.4, 0.0), &[cout as isize]));
    sd.insert("bn.running_var".into(), t(ramp(cout, 0.5, 1.0), &[cout as isize]));
    sd
}

/// The graph's conv, then the norm and the activation, over the NHWC map `x`:
/// what a block computes from an unfolded checkpoint.
fn conv_then_norm(sd: &StateDict, x: &Tensor, act: bool) -> Vec<f32> {
    let k = sd["conv.weight"].dim_const(2).unwrap();
    let p = (k / 2) as isize;
    let y = x.try_permute(&[0, 3, 1, 2]).unwrap().conv2d().weight(&sd["conv.weight"]).padding(&[(p, p), (p, p)]).call();
    let norm = BatchNorm2d::new(
        sd["bn.weight"].clone(),
        sd["bn.bias"].clone(),
        sd["bn.running_mean"].clone(),
        sd["bn.running_var"].clone(),
        YOLO_BN_EPS,
    );
    let y = norm.forward(&y.unwrap()).unwrap();
    let y = if act { y.silu().unwrap() } else { y };
    y.try_permute(&[0, 2, 3, 1]).unwrap().contiguous().to_vec::<f32>().unwrap()
}

/// Folding is value-preserving, whether the loader folds the checkpoint or the
/// block folds the norm it was handed: both reproduce conv + norm.
#[test_case(4, 8, 3, true; "3x3 with activation")]
#[test_case(16, 8, 1, false; "1x1 without")]
fn a_folded_conv_matches_conv_then_norm(cin: usize, cout: usize, k: usize, act: bool) {
    let sd = unfolded_state(cin, cout, k);
    let x = Tensor::from_slice(ramp(cin * 25, 2.0, 0.1)).try_reshape([1, 5, 5, cin as isize]).unwrap();
    let want = conv_then_norm(&sd, &x, act);
    for (path, state) in [("block", sd.clone()), ("loader", fold_batchnorm(&sd).unwrap())] {
        let mut conv = YoloConv::empty(cin, cout, k, 1, act);
        conv.load_state_dict(&state, "").unwrap();
        let got = conv.forward(&x).unwrap().to_vec::<f32>().unwrap();
        let max = want.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(max < 1e-5, "the {path} fold drifts by {max}");
    }
}

/// A conv that carries a bias is already normalized: its own state dict, folded
/// again, would apply the norm twice.
#[test]
fn a_biased_conv_is_not_folded_again() {
    let mut conv = YoloConv::empty(4, 8, 3, 1, true);
    conv.load_state_dict(&unfolded_state(4, 8, 3), "").unwrap();
    let sd = conv.state_dict("");
    let refolded = fold_batchnorm(&sd).unwrap();
    for key in ["conv.weight", "conv.bias"] {
        assert!(std::sync::Arc::ptr_eq(&sd[key].uop(), &refolded[key].uop()), "{key} is handed straight through");
    }
}

/// Only a `conv.weight` with a full `bn.*` beside it folds; anything else, such
/// as the head's biased final convs or the norm keys themselves, passes through.
#[test]
fn the_fold_leaves_other_keys_alone() {
    let mut sd = unfolded_state(2, 2, 1);
    sd.insert("head.2.weight".into(), Tensor::zeros(&[2, 2, 1, 1], DType::Float32));
    sd.insert("head.2.bias".into(), Tensor::zeros(&[2], DType::Float32));
    sd.insert("lone.conv.weight".into(), Tensor::zeros(&[2, 2, 1, 1], DType::Float32));
    let folded = fold_batchnorm(&sd).unwrap();
    assert_eq!(folded.len(), sd.len() + 1, "exactly one bias appears");
    assert!(folded.contains_key("conv.bias"));
    assert!(!folded.contains_key("lone.conv.bias"), "no norm, nothing to fold");
    for key in ["bn.weight", "bn.running_var", "head.2.weight", "lone.conv.weight"] {
        assert!(std::sync::Arc::ptr_eq(&sd[key].uop(), &folded[key].uop()), "{key} is handed straight through");
    }
}

/// The rounding boundary of a narrow block: downstream of the convolution's
/// reduce there is exactly **one** cast, the narrowing at the store. The fp32
/// accumulator reaches the norm and the activation unrounded.
///
/// Rounding it at the conv and letting `silu` widen again reads (1 widening,
/// 2 narrowings) instead — a round trip no rewrite may remove, because removing
/// it changes the result, and it measured +3.2% of the YOLO26x forward.
///
/// Counting only downstream of the reduce keeps the test blind to however the
/// caller happened to build its f16 operands.
#[test_case(true; "activated")]
fn a_narrow_block_rounds_once_after_the_reduce(act: bool) {
    use std::collections::HashSet;
    use std::sync::Arc;
    use svod_ir::{Op, UOp, ops};

    let sd: StateDict = fold_batchnorm(&unfolded_state(4, 8, 3))
        .unwrap()
        .into_iter()
        .map(|(k, t)| (k, t.cast(DType::Float16)))
        .collect();
    let mut conv = YoloConv::empty(4, 8, 3, 1, act);
    conv.load_state_dict(&sd, "").unwrap();

    let x = Tensor::from_slice(ramp(4 * 25, 2.0, 0.1)).try_reshape([1, 5, 5, 4]).unwrap().cast(DType::Float16);
    let y = conv.forward(&x).unwrap();
    assert_eq!(y.dtype(), DType::Float16, "the block leaves at the stream's width");

    let mut seen: HashSet<*const UOp> = HashSet::new();
    let (mut wide, mut narrow) = (0, 0);
    for node in y.uop().toposort() {
        let fed = node.op().sources().iter().any(|s| seen.contains(&Arc::as_ptr(s)));
        if !fed && !matches!(node.op(), Op::ReduceAxis(..) | Op::Reduce(..)) {
            continue;
        }
        seen.insert(Arc::as_ptr(&node));
        let Op::Cast(ops::Cast { src, dtype }) = node.op() else { continue };
        match (src.dtype() == DType::Float16, *dtype == DType::Float16) {
            (true, false) => wide += 1,
            (false, true) => narrow += 1,
            _ => {}
        }
    }
    assert_eq!((wide, narrow), (0, 1), "casts downstream of the reduce for act={act}");
}
