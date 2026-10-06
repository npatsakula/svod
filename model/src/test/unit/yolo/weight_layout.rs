//! Where a YOLO conv's weight is stored, checked through the production load
//! path: `load_weights`, then the block's own `load_state_dict`.

use std::sync::Arc;

use svod_dtype::DType;
use svod_ir::{Op, UOp, ops};
use svod_tensor::Tensor;
use svod_tensor::nn::{Module, StateDict};
use test_case::test_case;

use super::bn_fold::unfolded_state;
use crate::yolo::loader::{fold_batchnorm, load_weights};
use crate::yolo::{YoloBottleneck, YoloConv};

fn values(t: &Tensor) -> Vec<f32> {
    t.contiguous().cast(DType::Float32).to_vec::<f32>().unwrap()
}

fn dims(uop: &Arc<UOp>) -> Vec<usize> {
    uop.shape().unwrap().unwrap().iter().map(|d| d.as_const().unwrap()).collect()
}

/// The loader keeps every tensor in the checkpoint's layout, folded, cast and
/// realized: how a conv weight is stored is the block's decision.
#[test_case(DType::Float32; "f32")]
#[test_case(DType::Float16; "f16")]
fn the_loader_keeps_the_checkpoint_layout(dtype: DType) {
    let sd = unfolded_state(3, 4, 3);
    let loaded = load_weights(&sd, &dtype).unwrap();
    for (key, tensor) in &loaded {
        assert!(tensor.uop().has_buffer_identity(), "{key} comes back realized, not as a view");
    }
    let folded = fold_batchnorm(&sd).unwrap();
    assert_eq!(loaded["conv.weight"].dims().unwrap(), vec![4, 3, 3, 3]);
    assert_eq!(values(&loaded["conv.weight"]), values(&folded["conv.weight"].cast(dtype)));
}

/// A block stores its weight `[cout, kh, kw, cin]` behind the `[cout, cin, kh,
/// kw]` view the graph reads, at every dtype and kernel size, and a tk block
/// binds that same buffer rather than a copy of it.
fn assert_stored_channels_innermost(conv: &YoloConv, loaded: &Tensor) {
    let view = conv.conv.weight.uop();
    let Op::Permute(ops::Permute { src, axes }) = view.op() else { panic!("a view over the stored weight") };
    assert_eq!(axes.as_slice(), [0, 3, 1, 2]);
    assert!(src.has_buffer_identity(), "the stored weight is a buffer");
    let [cout, cin, kh, kw] = loaded.dims().unwrap()[..] else { panic!("a 4-D weight") };
    assert_eq!(dims(src), [cout, kh, kw, cin]);
    assert_eq!(conv.conv.weight.dims().unwrap(), [cout, cin, kh, kw]);
    assert_eq!(conv.weight_taps.as_ref().map(|taps| Arc::ptr_eq(&taps.uop(), src)), conv.tk.then_some(true));
    assert_eq!(values(&conv.conv.weight), values(loaded));
}

#[test_case(DType::Float32, 4, 8, 3, false; "f32 3x3")]
#[test_case(DType::Float16, 4, 8, 3, false; "f16 3x3")]
#[test_case(DType::Float16, 6, 6, 1, false; "f16 1x1")]
#[test_case(DType::Float16, 64, 64, 3, true; "f16 tk 3x3")]
#[test_case(DType::Float32, 64, 64, 3, true; "f32 tk 3x3, which runs the graph")]
fn a_loaded_conv_stores_its_weight_channels_innermost(dtype: DType, cin: usize, cout: usize, k: usize, tk: bool) {
    let loaded = load_weights(&unfolded_state(cin, cout, k), &dtype).unwrap();
    let mut conv = YoloConv::empty(cin, cout, k, 1, true);
    if tk {
        conv = conv.tk();
        assert!(conv.tk, "the shape is one the kernel serves");
    }
    conv.load_state_dict(&loaded, "").unwrap();
    assert_stored_channels_innermost(&conv, &loaded["conv.weight"]);
}

/// A channels-last bottleneck's convs read channels-last activations and still
/// store their weights the same way: one layout decision for every block.
#[test_case(DType::Float32; "f32")]
#[test_case(DType::Float16; "f16")]
fn a_channels_last_bottleneck_stores_its_weights_channels_innermost(dtype: DType) {
    let mut sd = StateDict::new();
    for (prefix, (cin, cout)) in [("cv1.", (16, 8)), ("cv2.", (8, 16))] {
        sd.extend(unfolded_state(cin, cout, 3).into_iter().map(|(key, t)| (format!("{prefix}{key}"), t)));
    }
    let loaded = load_weights(&sd, &dtype).unwrap();
    let mut block = YoloBottleneck::empty(16, 16, true).channels_last();
    block.load_state_dict(&loaded, "").unwrap();
    assert_stored_channels_innermost(&block.cv1, &loaded["cv1.conv.weight"]);
    assert_stored_channels_innermost(&block.cv2, &loaded["cv2.conv.weight"]);
}
