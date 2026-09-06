//! Whisper attention internals: the padded flash-attention admission policy and
//! its numeric agreement with unpadded SDPA.

use crate::whisper::attention::padded_fa_sequence_len;
use svod_dtype::DType;
use svod_ir::ConstValue;
use svod_tensor::Tensor;

#[test]
fn padded_fa_policy_targets_long_untiled_self_attention() {
    assert_eq!(padded_fa_sequence_len(false, 1500, 1500, 1500), Some(1536));
    assert_eq!(padded_fa_sequence_len(false, 1536, 1536, 1536), None);
    assert_eq!(padded_fa_sequence_len(false, 127, 127, 127), None);
    assert_eq!(padded_fa_sequence_len(true, 1500, 1500, 1500), None);
    assert_eq!(padded_fa_sequence_len(false, 1500, 100, 100), None);
    assert_eq!(padded_fa_sequence_len(false, 1500, 1500, 1499), None);
    assert_eq!(padded_fa_sequence_len(false, 1409, 1409, 1409), None, "large padding overhead must be rejected");
}

fn encoder_shape_inputs() -> (Tensor, Tensor, Tensor) {
    let make = || {
        let mut tensor = Tensor::randn(&[1, 1500, 2, 64]).unwrap().cast(DType::Float16).unwrap();
        tensor.realize().unwrap();
        tensor
    };
    (make(), make(), make())
}

#[test]
#[ignore = "GPU: padded Whisper encoder flash-attention vs unpadded SDPA"]
fn padded_encoder_flash_attention_matches_unpadded_sdpa() {
    let (q, k, v) = encoder_shape_inputs();
    if !svod_tk::flash_attention_supported(&q.device()) {
        eprintln!("skipping: flash-attention is not supported on {:?}", q.device());
        return;
    }
    let padding = [(0, 0), (0, 36), (0, 0), (0, 0)];
    let (qp, kp, vp) = (q.try_pad(&padding).unwrap(), k.try_pad(&padding).unwrap(), v.try_pad(&padding).unwrap());
    let key_lens = Tensor::full(&[1], ConstValue::Int(1500), DType::Int32).unwrap().to(q.device());
    let mut got =
        svod_tk::flash_attention_with(&qp, &kp, &vp, svod_tk::FaOpts { causal: false, key_lens: Some(&key_lens) })
            .expect("padded flash-attention")
            .expect("flash-attention target passed the support gate")
            .try_shrink([(0, 1), (0, 1500), (0, 2), (0, 64)])
            .unwrap()
            .cast(DType::Float32)
            .unwrap();
    assert_eq!(got.shape().unwrap().iter().map(|d| d.as_const().unwrap()).collect::<Vec<_>>(), [1, 1500, 2, 64]);
    got.realize().unwrap();

    let perm = |t: &Tensor| t.cast(DType::Float32).unwrap().try_permute(&[0, 2, 1, 3]).unwrap();
    let mut reference = perm(&q)
        .scaled_dot_product_attention()
        .key(&perm(&k))
        .value(&perm(&v))
        .is_causal(false)
        .call()
        .unwrap()
        .try_permute(&[0, 2, 1, 3])
        .unwrap();
    reference.realize().unwrap();
    let got = got.as_vec::<f32>().unwrap();
    let expected = reference.as_vec::<f32>().unwrap();
    let max_abs = got.iter().zip(&expected).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(max_abs <= 2e-2, "padded encoder FA exceeds tolerance: {max_abs:e}");
}
