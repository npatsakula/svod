//! Which tile kernels a YOLO graph reaches, and the epilogue fusions that hold
//! whichever path a block takes.

use svod_dtype::{DType, DeviceSpec};
use svod_ir::{Op, ops};
use svod_tensor::Tensor;
use test_case::test_case;

use crate::yolo::{Yolo26Detect, YoloBottleneck, YoloConfig, YoloConv, YoloScale};

/// Whether `device` runs tk3 kernels at all; a shape can still take the graph.
pub(super) fn tk_device(device: &DeviceSpec) -> bool {
    svod_tk3::ops::supported(device)
}

/// The tk3 kernels in `t`'s graph whose name starts with `name`, counted before
/// it is realized.
pub(super) fn tk_calls(t: &Tensor, name: &str) -> usize {
    let named = |info: &svod_ir::CallInfo| info.name.as_deref().is_some_and(|n| n.starts_with(name));
    t.uop()
        .toposort()
        .iter()
        .filter(|node| matches!(node.op(), Op::Call(ops::Call { info, .. }) if named(info)))
        .count()
}

/// At f16 on a kernel device every family of layer reaches a tile kernel: the
/// dense 3×3s the convolution, the 1×1s the GEMM, the PSA blocks attention,
/// and the cls branch's f32 logits the convolution's wide output.
#[test_case(YoloScale::Nano; "n")]
#[test_case(YoloScale::XLarge; "x")]
#[ignore = "GPU: SVOD_DEVICE=CUDA:0, builds the f16 detect graph"]
fn every_layer_family_reaches_a_kernel(scale: YoloScale) {
    let model = Yolo26Detect::with_zero_weights(YoloConfig::new(scale, 80).with_compute_dtype(DType::Float16));
    let images = Tensor::zeros(&[1, 3, 640, 640], DType::Float32);
    if !tk_device(&images.device()) {
        return;
    }
    let out = model.forward(&images).expect("forward");
    let (convs, gemms, attention) = (tk_calls(&out, "conv"), tk_calls(&out, "gemm"), tk_calls(&out, "flash"));
    assert!(convs > 0 && gemms > 0 && attention > 0, "conv {convs}, gemm {gemms}, attention {attention}");
}

/// The residual reaches the block's output whichever path the conv takes: at
/// f32 there is no kernel, so this is the graph's epilogue, and it has to equal
/// the two convs plus the input exactly.
#[test]
fn the_residual_rides_the_second_conv() {
    let block = YoloBottleneck::empty(8, 8, true);
    let x = Tensor::randn(&[1, 6, 6, 8]).unwrap();
    let want = block.cv2.forward(&block.cv1.forward(&x).unwrap()).unwrap().try_add(&x).unwrap();
    let got = block.forward(&x).unwrap();
    assert_eq!(got.dtype(), DType::Float32);
    assert_eq!(got.to_vec::<f32>().unwrap(), want.to_vec::<f32>().unwrap());
}

/// A block is channels-last on both sides at every kernel size and stride,
/// padded by half its kernel.
#[test_case(1, 1, [6, 5]; "1x1")]
#[test_case(3, 1, [6, 5]; "3x3")]
#[test_case(3, 2, [3, 3]; "3x3 stride 2")]
fn a_block_maps_nhwc_to_nhwc(kernel: usize, stride: usize, hw: [usize; 2]) {
    let conv = YoloConv::empty(16, 24, kernel, stride, true);
    let y = conv.forward(&Tensor::randn(&[2, 6, 5, 16]).unwrap()).unwrap();
    assert_eq!(y.dims().unwrap(), vec![2, hw[0], hw[1], 24]);
}
