//! Where a YOLO conv's weight is stored, checked through the production load
//! path: `load_weights`, then the block's own `load_state_dict`.

use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::{Module, StateDict};
use test_case::test_case;

use super::bn_fold::unfolded_state;
use crate::yolo::loader::{fold_batchnorm, load_weights};
use crate::yolo::{YoloBottleneck, YoloConv};

fn values(t: &Tensor) -> Vec<f32> {
    t.contiguous().cast(DType::Float32).to_vec::<f32>().unwrap()
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

/// A block stores its weight `[cout, kh, kw, cin]`, realized, the layout a
/// channels-last convolution reduces over, and hands the checkpoint's `[cout,
/// cin, kh, kw]` back in its state dict.
fn assert_stored_channels_innermost(conv: &YoloConv, loaded: &StateDict, prefix: &str) {
    let checkpoint = &loaded[&format!("{prefix}conv.weight")];
    let [cout, cin, kh, kw] = checkpoint.dims().unwrap()[..] else { panic!("a 4-D weight") };
    assert!(conv.weight.uop().has_buffer_identity(), "the stored weight is a buffer");
    assert!(conv.bias.uop().has_buffer_identity(), "the folded bias is a buffer");
    assert_eq!(conv.weight.dims().unwrap(), [cout, kh, kw, cin]);
    let written = conv.state_dict("");
    assert_eq!(written["conv.weight"].dims().unwrap(), [cout, cin, kh, kw]);
    assert_eq!(values(&written["conv.weight"]), values(checkpoint));
    assert_eq!(values(&conv.bias), values(&loaded[&format!("{prefix}conv.bias")]));
}

#[test_case(DType::Float32, 4, 8, 3; "f32 3x3")]
#[test_case(DType::Float16, 64, 64, 3; "f16 3x3")]
#[test_case(DType::Float16, 32, 48, 1; "f16 1x1")]
fn a_loaded_conv_stores_its_weight_channels_innermost(dtype: DType, cin: usize, cout: usize, k: usize) {
    let loaded = load_weights(&unfolded_state(cin, cout, k), &dtype).unwrap();
    let mut conv = YoloConv::empty(cin, cout, k, 1, true);
    conv.load_state_dict(&loaded, "").unwrap();
    assert_stored_channels_innermost(&conv, &loaded, "");
}

/// Every block of a bottleneck stores its weight the same way.
#[test]
fn a_bottleneck_stores_its_weights_channels_innermost() {
    let mut sd = StateDict::new();
    for (prefix, (cin, cout)) in [("cv1.", (16, 8)), ("cv2.", (8, 16))] {
        sd.extend(unfolded_state(cin, cout, 3).into_iter().map(|(key, t)| (format!("{prefix}{key}"), t)));
    }
    let loaded = load_weights(&sd, &DType::Float16).unwrap();
    let mut block = YoloBottleneck::empty(16, 16, true);
    block.load_state_dict(&loaded, "").unwrap();
    assert_stored_channels_innermost(&block.cv1, &loaded, "cv1.");
    assert_stored_channels_innermost(&block.cv2, &loaded, "cv2.");
}
