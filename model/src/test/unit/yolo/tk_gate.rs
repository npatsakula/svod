//! Where the tk convolution is asked for, per scale: the gate's reach as a test
//! rather than a census script, and the residual that rides its epilogue.

use svod_dtype::DType;
use svod_tensor::Tensor;
use test_case::test_case;

use crate::yolo::{C3k2, C3k2Inner, Yolo26Detect, YoloBottleneck, YoloConfig, YoloScale};

/// Whether every 3x3 body of the block's first inner unit runs the kernel.
fn bodies_on_tk(block: &C3k2) -> bool {
    match &block.m[0] {
        C3k2Inner::C3k(c3k) => c3k.m.iter().all(|b| b.nhwc),
        C3k2Inner::Bottleneck(b) | C3k2Inner::Attn(b, _) => b.cv1.tk && b.cv2.tk,
    }
}

/// Which convolutions ask for the kernel at each scale: the bodies of
/// backbone.2 never do (K = 288 at m/l, 48 channels at x), backbone.4's and
/// neck.16's do from m up — at x they are the 96-channel ones a `cout % 64`
/// bound turned away — and the box head's first 3x3 does wherever its width
/// reaches the lattice's 32-wide N edge (s and up).
#[test_case(YoloScale::Nano, (false, false, false, false); "n: a 16-wide head, 8- and 16-channel bodies")]
#[test_case(YoloScale::Small, (false, false, false, true); "s: the head's 32-wide reductions")]
#[test_case(YoloScale::Medium, (false, true, true, true); "m: 64-channel bodies")]
#[test_case(YoloScale::Large, (false, true, true, true); "l: as m")]
#[test_case(YoloScale::XLarge, (false, true, true, true); "x: 96-channel bodies and head")]
fn the_gate_reaches(scale: YoloScale, want: (bool, bool, bool, bool)) {
    let model = Yolo26Detect::with_zero_weights(YoloConfig::new(scale, 80));
    let got = (
        bodies_on_tk(&model.backbone.c3k2_2),
        bodies_on_tk(&model.backbone.c3k2_4),
        bodies_on_tk(&model.neck.c3k2_16),
        model.head.cv2.iter().all(|branch| branch.conv0.tk),
    );
    assert_eq!(got, want, "(backbone.2, backbone.4, neck.16, head conv0)");
}

/// The residual reaches the block's output whichever path the block takes: at
/// f32 there is no kernel, so this is the elementwise fallback, and it has to
/// equal the two convs plus the input exactly.
#[test]
fn the_residual_follows_the_block_off_the_kernel() {
    let block = YoloBottleneck::empty(8, 8, true);
    let x = Tensor::randn(&[1, 8, 6, 6]).unwrap();
    let want = block.cv2.forward(&block.cv1.forward(&x).unwrap()).unwrap().try_add(&x).unwrap();
    let got = block.forward(&x).unwrap();
    assert_eq!(got.dtype(), DType::Float32);
    assert_eq!(got.to_vec::<f32>().unwrap(), want.to_vec::<f32>().unwrap());
}
