use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::{Module, StateDict};
use test_case::test_case;

use crate::yolo::YoloConv;
use crate::yolo::loader::fold_batchnorm;

fn ramp(n: usize, scale: f32, offset: f32) -> Vec<f32> {
    (0..n).map(|i| ((i * 7919 % 97) as f32 / 97.0 - 0.5) * scale + offset).collect()
}

fn unfolded_state(cin: usize, cout: usize, k: usize) -> StateDict {
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

/// Folding is value-preserving: the biased conv alone reproduces conv + norm.
#[test_case(4, 8, 3, true; "3x3 with activation")]
#[test_case(6, 6, 1, false; "1x1 without activation")]
fn a_folded_conv_matches_conv_then_norm(cin: usize, cout: usize, k: usize, act: bool) {
    let sd = unfolded_state(cin, cout, k);
    let mut plain = YoloConv::empty(cin, cout, k, 1, act);
    plain.load_state_dict(&sd, "").unwrap();
    let mut folded = YoloConv::empty(cin, cout, k, 1, act);
    folded.load_state_dict(&fold_batchnorm(&sd).unwrap(), "").unwrap();
    assert!(folded.conv.bias.is_some(), "the fold leaves a bias on the conv");
    assert!(plain.conv.bias.is_none());

    let x = Tensor::from_slice(ramp(cin * 25, 2.0, 0.1)).try_reshape([1, cin as isize, 5, 5]).unwrap();
    let want = plain.forward(&x).unwrap().to_vec::<f32>().unwrap();
    let got = folded.forward(&x).unwrap().to_vec::<f32>().unwrap();
    let max = want.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    assert!(max < 1e-5, "folded conv drifts by {max}");
}

/// A conv reading NCHW keeps its weight logically `[cout, cin, kh, kw]` while
/// storing it taps-major; one reading channels-last keeps the bytes as loaded.
/// Either way the forward is the same.
#[test_case(false; "nchw input, taps-major weight")]
#[test_case(true; "channels-last input, cin-major weight")]
fn the_weight_layout_follows_the_input_layout(channels_last_input: bool) {
    let sd: StateDict = unfolded_state(4, 8, 3).into_iter().map(|(k, t)| (k, t.cast(DType::Float16))).collect();
    let mut conv = YoloConv::empty(4, 8, 3, 1, true);
    if channels_last_input {
        conv = conv.channels_last_input();
    }
    conv.load_state_dict(&sd, "").unwrap();
    assert_eq!(conv.conv.weight.dims().unwrap(), vec![8, 4, 3, 3]);
    assert_eq!(
        conv.conv.weight.contiguous().cast(DType::Float32).to_vec::<f32>().unwrap(),
        sd["conv.weight"].contiguous().cast(DType::Float32).to_vec::<f32>().unwrap()
    );
    assert_eq!(
        std::sync::Arc::ptr_eq(&conv.conv.weight.uop(), &sd["conv.weight"].uop()),
        channels_last_input,
        "a taps-major weight is a view over its own buffer, a cin-major one the checkpoint's node"
    );
    let x = Tensor::from_slice(ramp(4 * 25, 2.0, 0.1)).try_reshape([1, 4, 5, 5]).unwrap().cast(DType::Float16);
    let mut plain = YoloConv::empty(4, 8, 3, 1, true).channels_last_input();
    plain.load_state_dict(&sd, "").unwrap();
    let want = plain.forward(&x).unwrap().cast(DType::Float32).to_vec::<f32>().unwrap();
    let got = conv.forward(&x).unwrap().cast(DType::Float32).to_vec::<f32>().unwrap();
    let max = want.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    assert!(max < 1e-3, "the layout changes the result by {max}");
}

/// At f32 there is no tensor core to lay out for, so the weight stays as loaded.
#[test]
fn an_f32_conv_keeps_the_checkpoint_layout() {
    let sd = unfolded_state(4, 8, 3);
    let mut conv = YoloConv::empty(4, 8, 3, 1, true);
    conv.load_state_dict(&sd, "").unwrap();
    assert!(std::sync::Arc::ptr_eq(&conv.conv.weight.uop(), &sd["conv.weight"].uop()));
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

/// The folded weight keeps the checkpoint's dtype; the fold itself runs in f32.
#[test]
fn the_fold_keeps_the_weight_dtype() {
    let sd: StateDict = unfolded_state(2, 4, 3).into_iter().map(|(k, t)| (k, t.cast(DType::Float16))).collect();
    let folded = fold_batchnorm(&sd).unwrap();
    assert_eq!(folded["conv.weight"].dtype(), DType::Float16);
    assert_eq!(folded["conv.bias"].dtype(), DType::Float16);
}

/// The tk convolution is asked for only where its tiling rule can be met on the
/// channel counts — the part knowable when the model is built. The bounds are
/// the tile lattice's own: `svod_tk::kernels::tiling` searches N edges from 32
/// and strips from one 16-wide fragment, so a narrower output or a thinner
/// input reaches no tile. A block that fails it keeps the graph path rather
/// than paying for a layout change that buys nothing.
#[test_case(192, 192, 3, true; "192 channels, 3x3")]
#[test_case(384, 128, 3, true; "128 output channels")]
#[test_case(96, 96, 3, true; "96 output channels tile the narrowest N edge")]
#[test_case(384, 96, 3, true; "96 outputs from 384 inputs")]
#[test_case(192, 192, 1, false; "a 1x1 stays on the graph")]
#[test_case(64, 48, 3, false; "48 output channels tile no N edge")]
#[test_case(8, 64, 3, false; "8 input channels do not fill a strip")]
fn the_tk_flag_follows_what_the_kernel_can_tile(cin: usize, cout: usize, k: usize, eligible: bool) {
    let conv = YoloConv::empty(cin, cout, k, 1, true);
    assert_eq!(conv.tk_eligible(), eligible);
    assert_eq!(conv.tk().tk, eligible, "the flag is set only where the kernel can serve");
}

/// A depthwise block never asks for it: the kernel has no grouped form.
#[test]
fn a_depthwise_block_stays_on_the_graph() {
    assert!(!YoloConv::empty_dw(192, 192, 3, 1, true).tk().tk);
}

/// The weight the kernel binds is the taps-major tensor, and the one the graph
/// path reads is the `[cout, cin, kh, kw]` view over the same buffer.
#[test]
fn a_tk_block_keeps_both_weight_views() {
    let sd: StateDict = unfolded_state(64, 64, 3).into_iter().map(|(k, t)| (k, t.cast(DType::Float16))).collect();
    let mut conv = YoloConv::empty(64, 64, 3, 1, true).tk();
    conv.load_state_dict(&sd, "").unwrap();
    let taps = conv.weight_taps.as_ref().expect("a tk block carries the taps-major weight");
    assert_eq!(taps.dims().unwrap(), vec![64, 3, 3, 64]);
    assert_eq!(conv.conv.weight.dims().unwrap(), vec![64, 64, 3, 3]);
    assert_eq!(
        taps.contiguous().cast(DType::Float32).to_vec::<f32>().unwrap(),
        sd["conv.weight"]
            .try_permute(&[0, 2, 3, 1])
            .unwrap()
            .contiguous()
            .cast(DType::Float32)
            .to_vec::<f32>()
            .unwrap()
    );
}
