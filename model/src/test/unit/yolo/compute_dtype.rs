use svod_dtype::DType;
use svod_tensor::Tensor;
use test_case::test_case;

use crate::yolo::{Yolo26Detect, YoloConfig, YoloScale};

/// The boundary contract, both halves at once: the backbone computes in the
/// requested dtype, and whatever it produces reaches the caller as f32.
///
/// The backbone assertion is the one that catches a half-migration — if weights
/// and activations disagree anywhere, the promotion is silent and the only
/// symptom is that nothing got faster.
#[test_case(DType::Float16 ; "f16")]
fn the_backbone_computes_at_the_compute_dtype_and_the_head_returns_f32(dtype: DType) {
    let cfg = YoloConfig::new(YoloScale::Nano, 80).with_compute_dtype(dtype.clone());
    let model = Yolo26Detect::with_zero_weights(cfg.clone());
    let images = Tensor::zeros(&[1, 3, 64, 64], DType::Float32);

    let cast = cfg.cast_input(&images).expect("input view");
    assert_eq!(cast.dtype(), dtype, "the input cast reaches the compute dtype");
    assert_eq!(cast.dims().unwrap(), vec![1, 64, 64, 3], "the model runs channels-last");

    let (l4, l6, l10) = model.backbone.forward(&cast).expect("backbone forward");
    for (name, feat) in [("l4", &l4), ("l6", &l6), ("l10", &l10)] {
        assert_eq!(feat.dtype(), dtype, "backbone {name} promoted back to f32");
    }
    let (p3, p4, p5) = model.neck.forward(&l4, &l6, &l10).expect("neck forward");
    for (name, feat) in [("p3", &p3), ("p4", &p4), ("p5", &p5)] {
        assert_eq!(feat.dtype(), dtype, "neck {name} promoted back to f32");
    }

    // The host reads predictions as f32 (`predictions_to_vec::<f32>` rejects
    // anything else), and the box branch must not round through f16 before the
    // `x stride` multiply.
    let out = model.forward(&images).expect("model forward");
    assert_eq!(out.dtype(), DType::Float32, "predictions must reach the caller as f32");
}

/// The head's branches stay at the compute dtype for as long as precision
/// allows and no longer: a box branch rounds only its first conv's output
/// through f16 (the second one accumulates and emits f32 for the decode), and
/// a cls branch rounds everything but its logits.
#[test_case(DType::Float16 ; "f16")]
fn the_head_branches_compute_narrow_and_emit_f32(dtype: DType) {
    let cfg = YoloConfig::new(YoloScale::Nano, 80).with_compute_dtype(dtype.clone());
    let model = Yolo26Detect::with_zero_weights(cfg);
    let feat = Tensor::zeros(&[1, 8, 8, 64], dtype.clone());

    let cv2 = &model.head.cv2[0];
    let x = cv2.conv0.forward(&feat).expect("box conv0");
    assert_eq!(x.dtype(), dtype, "the box branch's first conv stays narrow");
    let x = cv2.conv1.forward(&x).expect("box conv1");
    assert_eq!(x.dtype(), DType::Float32, "the box branch's second conv accumulates and emits f32");
    assert_eq!(cv2.forward(&feat).expect("box branch").dtype(), DType::Float32, "the distances are f32");

    let cv3 = &model.head.cv3[0];
    let x = cv3
        .conv1
        .forward(&cv3.dw1.forward(&cv3.conv0.forward(&cv3.dw0.forward(&feat).unwrap()).unwrap()).unwrap())
        .unwrap();
    assert_eq!(x.dtype(), dtype, "the cls branch stays narrow up to its logits");
    assert_eq!(cv3.forward(&feat).expect("cls branch").dtype(), DType::Float32, "the logits are f32");
}
