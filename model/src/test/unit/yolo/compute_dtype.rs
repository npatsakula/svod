use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::Module;
use test_case::test_case;

use crate::yolo::{Yolo26Detect, YoloConfig, YoloScale};
use svod_tensor::nn::Layer;

/// f32 unless asked otherwise: the knob must not change existing callers.
#[test]
fn the_default_compute_dtype_is_f32() {
    assert_eq!(YoloConfig::new(YoloScale::Nano, 80).compute_dtype, DType::Float32);
}

/// `with_zero_weights` mints f32 placeholders, so a non-f32 config has to move
/// them too — otherwise `least_upper_dtype(f16, f32) = f32` promotes the first
/// conv and the graph silently reverts with no error and no speedup.
#[test_case(DType::Float16 ; "f16")]
#[test_case(DType::BFloat16 ; "bf16")]
#[test_case(DType::Float32 ; "f32")]
fn placeholders_are_built_at_the_compute_dtype(dtype: DType) {
    let cfg = YoloConfig::new(YoloScale::Nano, 80).with_compute_dtype(dtype.clone());
    let model = Yolo26Detect::with_zero_weights(cfg);
    let sd = model.state_dict("");
    assert!(!sd.is_empty(), "the model has parameters to check");
    for (key, tensor) in &sd {
        if tensor.dtype().is_float() {
            assert_eq!(tensor.dtype(), dtype, "{key} kept its placeholder dtype");
        }
    }
}

/// The boundary contract, both halves at once: the backbone computes in the
/// requested dtype, and whatever it produces reaches the caller as f32.
///
/// The backbone assertion is the one that catches a half-migration — if weights
/// and activations disagree anywhere, the promotion is silent and the only
/// symptom is that nothing got faster.
#[test_case(DType::Float16 ; "f16")]
#[test_case(DType::BFloat16 ; "bf16")]
#[test_case(DType::Float32 ; "f32")]
fn the_backbone_computes_at_the_compute_dtype_and_the_head_returns_f32(dtype: DType) {
    let cfg = YoloConfig::new(YoloScale::Nano, 80).with_compute_dtype(dtype.clone());
    let model = Yolo26Detect::with_zero_weights(cfg.clone());
    let images = Tensor::zeros(&[1, 3, 64, 64], DType::Float32);

    let cast = cfg.cast_input(&images);
    assert_eq!(cast.dtype(), dtype, "the input cast reaches the compute dtype");

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
#[test_case(DType::BFloat16 ; "bf16")]
fn the_head_branches_compute_narrow_and_emit_f32(dtype: DType) {
    let cfg = YoloConfig::new(YoloScale::Nano, 80).with_compute_dtype(dtype.clone());
    let model = Yolo26Detect::with_zero_weights(cfg);
    let feat = Tensor::zeros(&[1, 64, 8, 8], dtype.clone());

    let cv2 = &model.head.cv2[0];
    let x = cv2.conv0.forward(&feat).expect("box conv0");
    assert_eq!(x.dtype(), dtype, "the box branch's first conv stays narrow");
    let x = cv2.conv1.forward(&x).expect("box conv1");
    assert_eq!(x.dtype(), DType::Float32, "the box branch's second conv accumulates and emits f32");
    assert_eq!(cv2.conv2.forward(&x).expect("box conv2").dtype(), DType::Float32, "the distances are f32");
    assert_eq!(cv2.forward(&feat).expect("box branch").dtype(), DType::Float32);

    let cv3 = &model.head.cv3[0];
    let x = cv3
        .conv1
        .forward(&cv3.dw1.forward(&cv3.conv0.forward(&cv3.dw0.forward(&feat).unwrap()).unwrap()).unwrap())
        .unwrap();
    assert_eq!(x.dtype(), dtype, "the cls branch stays narrow up to its logits");
    assert_eq!(cv3.conv2.forward(&x).expect("cls conv2").dtype(), DType::Float32, "the logits are f32");
}

/// Float parameters move; integer buffers do not — narrowing a count to f16
/// would be meaningless, and `num_batches_tracked` rides along in real
/// checkpoints.
#[test]
fn casting_weights_leaves_integer_buffers_alone() {
    let mut sd = svod_tensor::nn::StateDict::new();
    sd.insert("conv.weight".to_string(), Tensor::zeros(&[2, 2], DType::Float32));
    sd.insert("bn.num_batches_tracked".to_string(), Tensor::zeros(&[1], DType::Int64));

    let cast = crate::yolo::loader::cast_weights(&sd, &DType::Float16);

    assert_eq!(cast["conv.weight"].dtype(), DType::Float16);
    assert_eq!(cast["bn.num_batches_tracked"].dtype(), DType::Int64);
}

/// The f32 path must stay untouched: a dict that already matches comes back as
/// the very same graph nodes, not as a pile of no-op CASTs, and nothing gets
/// realized on the way through.
#[test]
fn loading_weights_at_their_own_dtype_is_a_passthrough() {
    let mut sd = svod_tensor::nn::StateDict::new();
    sd.insert("conv.weight".to_string(), Tensor::zeros(&[2, 2], DType::Float32));

    let loaded = crate::yolo::loader::load_weights(&sd, &DType::Float32).expect("passthrough");

    assert_eq!(loaded["conv.weight"].dtype(), DType::Float32);
    assert!(
        std::sync::Arc::ptr_eq(&sd["conv.weight"].uop(), &loaded["conv.weight"].uop()),
        "the original node is handed straight back"
    );
}
