//! Tests for transformer building blocks: embedding, attention, rotary embeddings, rms_norm.

use crate::Tensor;
use ndarray::{Array2, array};
use svod_dtype::DType;
use test_case::test_case;

crate::codegen_tests! {
    // =========================================================================
    // RMS Norm tests
    // =========================================================================

    fn test_rms_norm_basic(config) {
        // rms_norm(x) = x * rsqrt(mean(x^2) + eps)
        let x = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0, 4.0]]);
        let mut result = x.rms_norm(-1, 1e-5).unwrap();
        result.realize_with(&config).unwrap();
        let view = result.array_view::<f32>().unwrap();
        assert_eq!(view.shape(), &[1, 4]);

        // Manual: mean([1,4,9,16]) = 7.5, rsqrt(7.5 + 1e-5) ≈ 0.36514837
        let rms_inv = 1.0 / (7.5f32 + 1e-5).sqrt();
        for i in 0..4 {
            let expected = (i + 1) as f32 * rms_inv;
            assert!((view[[0, i]] - expected).abs() < 1e-4, "rms_norm[{i}]: got {}, expected {}", view[[0, i]], expected);
        }
    }

    fn test_rms_norm_axis(config) {
        // (2, 3), normalize over last axis
        let x = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let mut result = x.rms_norm(-1, 1e-5).unwrap();
        result.realize_with(&config).unwrap();
        let view = result.array_view::<f32>().unwrap();
        assert_eq!(view.shape(), &[2, 3]);

        // Row 0: mean([1,4,9]) = 14/3, rsqrt(14/3 + 1e-5) ≈ 0.4629
        let rms0 = 1.0 / (14.0f32 / 3.0 + 1e-5).sqrt();
        assert!((view[[0, 0]] - 1.0 * rms0).abs() < 1e-4);
        assert!((view[[0, 1]] - 2.0 * rms0).abs() < 1e-4);

        // Row 1: mean([16,25,36]) = 77/3, rsqrt(77/3 + 1e-5)
        let rms1 = 1.0 / (77.0f32 / 3.0 + 1e-5).sqrt();
        assert!((view[[1, 0]] - 4.0 * rms1).abs() < 1e-4);
    }

    // =========================================================================
    // Embedding tests
    // =========================================================================

    fn test_embedding_basic(config) {
        // Weight: [3, 4] (3 vocab, 4 embed_dim)
        let weight_data: Vec<f32> = (0..12).map(|v| v as f32).collect();
        let weight = Tensor::from_ndarray(&Array2::from_shape_vec((3, 4), weight_data).unwrap());
        // Indices: [2, 0] -> should return rows 2 and 0
        let indices = Tensor::from_slice([2i32, 0]);
        let mut result = weight.embedding(&indices).unwrap();
        result.realize_with(&config).unwrap();
        let view = result.array_view::<f32>().unwrap();
        assert_eq!(view.shape(), &[2, 4]);
        // Row 0 = weight[2] = [8, 9, 10, 11]
        assert_eq!(view[[0, 0]], 8.0);
        assert_eq!(view[[0, 3]], 11.0);
        // Row 1 = weight[0] = [0, 1, 2, 3]
        assert_eq!(view[[1, 0]], 0.0);
        assert_eq!(view[[1, 3]], 3.0);
    }

    fn test_embedding_2d_indices(config) {
        // Weight: [4, 2] (4 vocab, 2 embed_dim)
        let weight = Tensor::from_ndarray(&array![[0.0f32, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]);
        // Indices: [2, 3] (batch=2, seq=3)
        let indices = Tensor::from_ndarray(&array![[0i32, 1, 2], [3, 2, 1]]);
        let mut result = weight.embedding(&indices).unwrap();
        result.realize_with(&config).unwrap();
        let view = result.array_view::<f32>().unwrap();
        assert_eq!(view.shape(), &[2, 3, 2]);
        // [0,0] = weight[0] = [0, 1]
        assert_eq!(view[[0, 0, 0]], 0.0);
        assert_eq!(view[[0, 0, 1]], 1.0);
        // [0,2] = weight[2] = [4, 5]
        assert_eq!(view[[0, 2, 0]], 4.0);
        // [1,0] = weight[3] = [6, 7]
        assert_eq!(view[[1, 0, 0]], 6.0);
        assert_eq!(view[[1, 0, 1]], 7.0);
    }

    // =========================================================================
    // Scaled Dot-Product Attention tests
    // =========================================================================

    fn test_sdpa_basic(config) {
        // Q, K, V: [1, 1, 2, 2] (batch=1, head=1, seq=2, dim=2)
        let q = Tensor::from_ndarray(&array![[[[1.0f32, 0.0], [0.0, 1.0]]]]);
        let k = q.clone();
        let v = Tensor::from_ndarray(&array![[[[1.0f32, 2.0], [3.0, 4.0]]]]);

        let mut result = q.scaled_dot_product_attention().key(&k).value(&v).call().unwrap();
        result.realize_with(&config).unwrap();
        let view = result.array_view::<f32>().unwrap();
        assert_eq!(view.shape(), &[1, 1, 2, 2]);
        // With identity-like Q=K, attention should weight both rows
    }

    fn test_sdpa_causal(config) {
        // Q, K, V: [1, 1, 3, 2] — verify causal masking zeros upper triangle
        let q = Tensor::from_ndarray(&array![[[[1.0f32, 0.0], [0.0, 1.0], [1.0, 1.0]]]]);
        let k = q.clone();
        let v = Tensor::from_ndarray(&array![[[[1.0f32, 0.0], [0.0, 1.0], [0.0, 0.0]]]]);

        let mut result = q.scaled_dot_product_attention().key(&k).value(&v).is_causal(true).call().unwrap();
        result.realize_with(&config).unwrap();
        let view = result.array_view::<f32>().unwrap();
        assert_eq!(view.shape(), &[1, 1, 3, 2]);
        // Position 0 can only attend to position 0 -> output[0] = V[0] = [1, 0]
        assert!((view[[0, 0, 0, 0]] - 1.0).abs() < 1e-4);
        assert!((view[[0, 0, 0, 1]] - 0.0).abs() < 1e-4);
    }

    fn test_sdpa_softcap(config) {
        // Verify softcap bounds the attention scores
        let q = Tensor::from_ndarray(&array![[[[10.0f32, 0.0], [0.0, 10.0]]]]);
        let k = q.clone();
        let v = Tensor::from_ndarray(&array![[[[1.0f32, 0.0], [0.0, 1.0]]]]);

        // With softcap, large scores get capped via tanh
        let mut result = q.scaled_dot_product_attention().key(&k).value(&v).softcap(1.0).call().unwrap();
        result.realize_with(&config).unwrap();
        // Should still produce valid output (no NaN/Inf)
        for val in result.as_vec::<f32>().unwrap() {
            assert!(val.is_finite(), "softcap produced non-finite value: {val}");
        }
    }

    fn test_sdpa_softcap_applies_before_mask(config) {
        // Softcap must cap the raw scaled scores. Applied after the causal mask
        // it squashes `dtype::min` to `-cap`, leaving the masked key ~27% of the
        // softmax weight (query 0 then reads 0.731 instead of V[0] = 1.0).
        let q = Tensor::from_ndarray(&array![[[[0.0f32], [0.0]]]]);
        let k = q.clone();
        let v = Tensor::from_ndarray(&array![[[[1.0f32], [100.0]]]]);

        let mut result = q
            .scaled_dot_product_attention()
            .key(&k)
            .value(&v)
            .is_causal(true)
            .softcap(1.0)
            .call()
            .unwrap();
        result.realize_with(&config).unwrap();
        let view = result.array_view::<f32>().unwrap();
        assert_eq!(view[[0, 0, 0, 0]], 1.0, "query 0 must see only V[0]");
        // Query 1 attends to both keys equally: (1 + 100) / 2.
        assert!((view[[0, 0, 1, 0]] - 50.5).abs() < 1e-3);
    }

    fn test_sdpa_float16_scores_stay_in_float32(config) {
        // 64 products of 100·100 sum to 640000, past float16's 65504: scores
        // formed in the input dtype overflow to inf and softmax gives NaN.
        let q_data: Vec<f32> = vec![100.0; 2 * 64];
        let mut k_data: Vec<f32> = vec![100.0; 64];
        k_data.extend(std::iter::repeat_n(99.0f32, 64));
        let q = Tensor::from_slice(q_data).try_reshape([1, 1, 2, 64]).unwrap().cast(DType::Float16).unwrap();
        let k = Tensor::from_slice(k_data).try_reshape([1, 1, 2, 64]).unwrap().cast(DType::Float16).unwrap();
        let v = Tensor::from_slice([1.0f32, 100.0]).try_reshape([1, 1, 2, 1]).unwrap().cast(DType::Float16).unwrap();

        let result = q.scaled_dot_product_attention().key(&k).value(&v).call().unwrap();
        assert_eq!(result.uop().dtype(), DType::Float16, "output must keep the query dtype");
        let mut result = result.cast(DType::Float32).unwrap();
        result.realize_with(&config).unwrap();
        // Key 0 wins by 800 in score, so every query reads V[0] = 1.0.
        for value in result.as_vec::<f32>().unwrap() {
            assert!(value.is_finite(), "float16 scores overflowed: {value}");
            assert!((value - 1.0).abs() < 1e-3, "expected 1.0, got {value}");
        }
    }

    fn test_sdpa_bool_mask_true_masks_out(config) {
        let q = Tensor::from_ndarray(&array![[[[1.0f32, 0.0]]]]);
        let k = Tensor::from_ndarray(&array![[[[1.0f32, 0.0], [0.0, 1.0]]]]);
        let v = Tensor::from_ndarray(&array![[[[10.0f32, 1.0], [1.0, 10.0]]]]);
        // True means masked, False means visible.
        let mask = Tensor::from_ndarray(&array![[[[true, false]]]]);

        let mut result = q
            .scaled_dot_product_attention()
            .key(&k)
            .value(&v)
            .maybe_attn_mask(Some(&mask))
            .call()
            .unwrap();
        result.realize_with(&config).unwrap();
        let view = result.array_view::<f32>().unwrap();
        assert_eq!(view.shape(), &[1, 1, 1, 2]);
        assert!((view[[0, 0, 0, 0]] - 1.0).abs() < 1e-4);
        assert!((view[[0, 0, 0, 1]] - 10.0).abs() < 1e-4);
    }

    fn test_sdpa_bool_mask_all_masked_row_finite(config) {
        let q = Tensor::from_ndarray(&array![[[[1.0f32, 0.0]]]]);
        let k = Tensor::from_ndarray(&array![[[[1.0f32, 0.0], [0.0, 1.0]]]]);
        let v = Tensor::from_ndarray(&array![[[[10.0f32, 1.0], [1.0, 10.0]]]]);
        let mask = Tensor::from_ndarray(&array![[[[true, true]]]]);

        let mut result = q
            .scaled_dot_product_attention()
            .key(&k)
            .value(&v)
            .maybe_attn_mask(Some(&mask))
            .call()
            .unwrap();
        result.realize_with(&config).unwrap();
        for v in result.as_vec::<f32>().unwrap() {
            assert!(v.is_finite(), "expected finite attention output, got {v}");
        }
    }

    fn test_sdpa_bool_mask_all_masked_with_causal_finite(config) {
        let q = Tensor::from_ndarray(&array![[[[1.0f32, 0.0], [0.0, 1.0]]]]);
        let k = q.clone();
        let v = Tensor::from_ndarray(&array![[[[10.0f32, 1.0], [1.0, 10.0]]]]);
        let mask = Tensor::from_ndarray(&array![[[[true, true], [true, true]]]]);

        let mut result = q
            .scaled_dot_product_attention()
            .key(&k)
            .value(&v)
            .is_causal(true)
            .maybe_attn_mask(Some(&mask))
            .call()
            .unwrap();
        result.realize_with(&config).unwrap();
        for v in result.as_vec::<f32>().unwrap() {
            assert!(v.is_finite(), "expected finite attention output with causal+mask, got {v}");
        }
    }

    fn test_sdpa_rejects_non_float_qkv(_config) {
        let qf = Tensor::from_ndarray(&array![[[[1.0f32, 0.0]]]]);
        let kf = Tensor::from_ndarray(&array![[[[1.0f32, 0.0], [0.0, 1.0]]]]);
        let vf = Tensor::from_ndarray(&array![[[[10.0f32, 1.0], [1.0, 10.0]]]]);

        let qi = Tensor::from_ndarray(&array![[[[1i32, 0]]]]);
        let ki = Tensor::from_ndarray(&array![[[[1i32, 0], [0, 1]]]]);
        let vi = Tensor::from_ndarray(&array![[[[10i32, 1], [1, 10]]]]);

        let err_q = match qi.scaled_dot_product_attention().key(&kf).value(&vf).call() {
            Ok(_) => panic!("expected query dtype error"),
            Err(err) => err,
        };
        assert!(matches!(err_q, crate::Error::FloatDTypeRequired { arg: "query", .. }));

        let err_k = match qf.scaled_dot_product_attention().key(&ki).value(&vf).call() {
            Ok(_) => panic!("expected key dtype error"),
            Err(err) => err,
        };
        assert!(matches!(err_k, crate::Error::FloatDTypeRequired { arg: "key", .. }));

        let err_v = match qf.scaled_dot_product_attention().key(&kf).value(&vi).call() {
            Ok(_) => panic!("expected value dtype error"),
            Err(err) => err,
        };
        assert!(matches!(err_v, crate::Error::FloatDTypeRequired { arg: "value", .. }));
    }

    fn test_sdpa_window_masks_far_keys(config) {
        // Seq len 4, head dim 1. Q=K=ones so raw scores are uniform; the only
        // thing distinguishing which keys are attended is the window band.
        // window=(0,0) → each query attends ONLY to itself. With V = [0,10,20,30]
        // the output equals the value at the query's own position.
        let q = Tensor::from_ndarray(&array![[[[1.0f32], [1.0], [1.0], [1.0]]]]); // [1,1,4,1]
        let k = q.clone();
        let v = Tensor::from_ndarray(&array![[[[0.0f32], [10.0], [20.0], [30.0]]]]);

        let mut result = q
            .scaled_dot_product_attention()
            .key(&k)
            .value(&v)
            .window((0usize, 0usize))
            .call()
            .unwrap();
        result.realize_with(&config).unwrap();
        let view = result.array_view::<f32>().unwrap();
        assert_eq!(view.shape(), &[1, 1, 4, 1]);
        // Self-only attention: output[q] = v[q].
        assert!((view[[0, 0, 0, 0]] - 0.0).abs() < 1e-4, "q0 leaked far key: {}", view[[0, 0, 0, 0]]);
        assert!((view[[0, 0, 1, 0]] - 10.0).abs() < 1e-4);
        assert!((view[[0, 0, 2, 0]] - 20.0).abs() < 1e-4);
        assert!((view[[0, 0, 3, 0]] - 30.0).abs() < 1e-4);
    }

    fn test_sdpa_window_band_attends_neighbors(config) {
        // window=(1,1): each query attends to itself and its immediate
        // neighbours. v = [0,10,20,30]; q=1 only sees keys 0,1,2 → mean of
        // (0,10,20)/3 = 10.0 (scores uniform). q=0 sees only keys 0,1 →
        // mean(0,10)/2 = 5.0.
        let q = Tensor::from_ndarray(&array![[[[1.0f32], [1.0], [1.0], [1.0]]]]);
        let k = q.clone();
        let v = Tensor::from_ndarray(&array![[[[0.0f32], [10.0], [20.0], [30.0]]]]);

        let mut result = q
            .scaled_dot_product_attention()
            .key(&k)
            .value(&v)
            .window((1usize, 1usize))
            .call()
            .unwrap();
        result.realize_with(&config).unwrap();
        let view = result.array_view::<f32>().unwrap();
        // q0: keys {0,1} → (0+10)/2 = 5
        assert!((view[[0, 0, 0, 0]] - 5.0).abs() < 1e-4, "q0: {}", view[[0, 0, 0, 0]]);
        // q1: keys {0,1,2} → (0+10+20)/3 = 10
        assert!((view[[0, 0, 1, 0]] - 10.0).abs() < 1e-4, "q1: {}", view[[0, 0, 1, 0]]);
        // q3: keys {2,3} → (20+30)/2 = 25
        assert!((view[[0, 0, 3, 0]] - 25.0).abs() < 1e-4, "q3: {}", view[[0, 0, 3, 0]]);
    }

    fn test_sdpa_window_intersects_bool_mask(config) {
        // window=(0,1) keeps keys {q, q+1}. A bool mask removes keys ≥2
        // everywhere. So q=0 keeps {0,1}∩{0,1}={0,1} → mean(0,10)=5; q=1 keeps
        // {1,2}∩{0,1}={1} → v[1]=10 (the window allowed key 2 but the mask
        // stripped it — this is the intersection under test).
        let q = Tensor::from_ndarray(&array![[[[1.0f32], [1.0], [1.0], [1.0]]]]);
        let k = q.clone();
        let v = Tensor::from_ndarray(&array![[[[0.0f32], [10.0], [20.0], [30.0]]]]);
        // True = masked out. Keys ≥2 masked everywhere; key 1 also masked for q0.
        let mask = Tensor::from_ndarray(&array![
            [[[false, true, true, true], [false, false, true, true], [false, false, true, true], [false, false, true, true]]]
        ]);

        let mut result = q
            .scaled_dot_product_attention()
            .key(&k)
            .value(&v)
            .window((0usize, 1usize))
            .maybe_attn_mask(Some(&mask))
            .call()
            .unwrap();
        result.realize_with(&config).unwrap();
        let view = result.array_view::<f32>().unwrap();
        // q0: window {0,1} ∩ mask-keep {0} = {0} → v[0] = 0.
        assert!((view[[0, 0, 0, 0]] - 0.0).abs() < 1e-4, "q0 intersect: {}", view[[0, 0, 0, 0]]);
        // q1: window {1,2} ∩ mask-keep {0,1} = {1} → v[1] = 10.
        assert!((view[[0, 0, 1, 0]] - 10.0).abs() < 1e-4, "q1 intersect: {}", view[[0, 0, 1, 0]]);
    }

    // =========================================================================
    // Rotary Embedding tests
    // =========================================================================

    fn test_rotary_emb_split(config) {
        // Non-interleaved: [1, 1, 4] -> split into [1, 1, 2] halves
        let x = Tensor::from_ndarray(&array![[[1.0f32, 2.0, 3.0, 4.0]]]);
        // cos = [1, 0], sin = [0, 1] (identity-like rotation)
        let cos = Tensor::from_ndarray(&array![[[1.0f32, 0.0]]]);
        let sin = Tensor::from_ndarray(&array![[[0.0f32, 0.0]]]);

        let mut result = x.apply_rotary_emb(&cos, &sin, false).unwrap();
        result.realize_with(&config).unwrap();
        let view = result.array_view::<f32>().unwrap();
        assert_eq!(view.shape(), &[1, 1, 4]);
        // With cos=[1,0], sin=[0,0]:
        // real = x1*cos - x2*sin = [1*1 - 3*0, 2*0 - 4*0] = [1, 0]
        // imag = x1*sin + x2*cos = [1*0 + 3*1, 2*0 + 4*0] = [3, 0]
        // Hmm, actually cos/sin broadcast element-wise to x1 and x2
        // x1 = [1, 2], x2 = [3, 4], cos = [1, 0], sin = [0, 0]
        // real = [1*1 - 3*0, 2*0 - 4*0] = [1, 0]
        // imag = [1*0 + 3*1, 2*0 + 4*0] = [3, 0]
        // cat = [1, 0, 3, 0]
        assert!((view[[0, 0, 0]] - 1.0).abs() < 1e-5);
        assert!((view[[0, 0, 1]] - 0.0).abs() < 1e-5);
        assert!((view[[0, 0, 2]] - 3.0).abs() < 1e-5);
        assert!((view[[0, 0, 3]] - 0.0).abs() < 1e-5);
    }

    fn test_rotary_emb_interleaved(config) {
        // Interleaved: [1, 1, 4] -> reshape [1,1,2,2] -> split -> squeeze
        let x = Tensor::from_ndarray(&array![[[1.0f32, 2.0, 3.0, 4.0]]]);
        // cos = [1, 1], sin = [0, 0] (identity rotation)
        let cos = Tensor::from_ndarray(&array![[[1.0f32, 1.0]]]);
        let sin = Tensor::from_ndarray(&array![[[0.0f32, 0.0]]]);

        let mut result = x.apply_rotary_emb(&cos, &sin, true).unwrap();
        result.realize_with(&config).unwrap();
        let view = result.array_view::<f32>().unwrap();
        assert_eq!(view.shape(), &[1, 1, 4]);
        // Interleaved: x1 = [1, 3] (even), x2 = [2, 4] (odd)
        // real = x1*cos - x2*sin = [1, 3]
        // imag = x1*sin + x2*cos = [2, 4]
        // stack on last dim -> [[1,2], [3,4]] -> flatten -> [1, 2, 3, 4]
        assert!((view[[0, 0, 0]] - 1.0).abs() < 1e-5);
        assert!((view[[0, 0, 1]] - 2.0).abs() < 1e-5);
        assert!((view[[0, 0, 2]] - 3.0).abs() < 1e-5);
        assert!((view[[0, 0, 3]] - 4.0).abs() < 1e-5);
    }

    fn test_rotary_emb_rotation(config) {
        // 90-degree rotation: cos=0, sin=1
        let x = Tensor::from_ndarray(&array![[[1.0f32, 0.0, 0.0, 1.0]]]);
        let cos = Tensor::from_ndarray(&array![[[0.0f32, 0.0]]]);
        let sin = Tensor::from_ndarray(&array![[[1.0f32, 1.0]]]);

        let mut result = x.apply_rotary_emb(&cos, &sin, false).unwrap();
        result.realize_with(&config).unwrap();
        let view = result.array_view::<f32>().unwrap();
        // x1 = [1, 0], x2 = [0, 1]
        // real = x1*cos - x2*sin = [0-0, 0-1] = [0, -1]
        // imag = x1*sin + x2*cos = [1+0, 0+0] = [1, 0]
        // cat = [0, -1, 1, 0]
        assert!((view[[0, 0, 0]] - 0.0).abs() < 1e-5);
        assert!((view[[0, 0, 1]] - (-1.0)).abs() < 1e-5);
        assert!((view[[0, 0, 2]] - 1.0).abs() < 1e-5);
        assert!((view[[0, 0, 3]] - 0.0).abs() < 1e-5);
    }
}

/// GigaAM-shaped fp16 SDPA inputs: realized `[B, T, H, d]` buffers (as the
/// model's projections leave them) with head-major views, plus per-batch key
/// lengths. `run` builds the attention on `device` and realizes it with `config`.
struct SdpaCase {
    b: usize,
    h: usize,
    t: usize,
    d: usize,
}

impl SdpaCase {
    fn sample(&self, seed: usize) -> ndarray::Array4<f32> {
        let (b, t, h, d) = (self.b, self.t, self.h, self.d);
        ndarray::Array4::from_shape_fn((b, t, h, d), |(i, j, k, l)| {
            let x = (i * 7919 + j * 104729 + k * 1301 + l * 7 + seed * 31) % 997;
            (x as f32 / 498.5) - 1.0
        })
    }

    fn key_lens(&self) -> Vec<i32> {
        (0..self.b).map(|i| (self.t - (i * self.t) / (2 * self.b)) as i32).collect()
    }

    /// Everything the expression builds (inputs, arange, consts) lands on `device`.
    /// The inputs realize under `inputs` (a plan meant for the attention kernels
    /// need not fit a cast), the expression under `config`. `build` receives
    /// head-major `q, k, v` and the `[B, 1, 1, 1]` key lengths.
    fn run(
        &self,
        device: svod_dtype::DeviceSpec,
        inputs: &crate::PrepareConfig,
        config: &crate::PrepareConfig,
        build: impl Fn(&Tensor, &Tensor, &Tensor, &Tensor) -> Tensor,
    ) -> Vec<f32> {
        svod_dtype::default_device::with_default_device(device, || {
            let realized = |mut x: Tensor| {
                x.realize_with(inputs).unwrap();
                x
            };
            let heads = |seed: usize| {
                realized(Tensor::from_ndarray(&self.sample(seed)).cast(DType::Float16).unwrap())
                    .try_permute(&[0, 2, 1, 3])
                    .unwrap()
            };
            let (q, k, v) = (heads(1), heads(2), heads(3));
            let lens = realized(Tensor::from_ndarray(&ndarray::Array1::from(self.key_lens())))
                .try_reshape([self.b, 1, 1, 1])
                .unwrap();
            let mut out = build(&q, &k, &v, &lens);
            out.realize_with(config).unwrap();
            out.as_vec::<f32>().unwrap()
        })
    }

    /// `[B, 1, 1, T]` bool key-padding mask: true (masked) where `arange(T) >= lens[b]`.
    fn key_mask(&self, lens: &Tensor) -> Tensor {
        let range = Tensor::arange(self.t as i64, None, None).unwrap().try_reshape([1usize, 1, 1, self.t]).unwrap();
        range.try_ge(lens).unwrap()
    }
}

/// Relative mismatch report: the count over `tol` and the worst offender.
fn assert_all_close(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len());
    let mut worst = (0usize, 0f32);
    let mut bad = 0usize;
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        let diff = (a - e).abs() / (1.0 + e.abs());
        if diff > worst.1 {
            worst = (i, diff);
        }
        if diff > tol {
            bad += 1;
        }
    }
    assert!(
        bad == 0,
        "{bad}/{} outputs differ by > {tol}; worst at {}: GPU={} CPU={}",
        actual.len(),
        worst.0,
        actual[worst.0],
        expected[worst.0]
    );
}

/// GigaAM-shaped masked fp16 SDPA on `CUDA:0` must match the CPU result. The
/// BEAM variant replays/searches every kernel's plan (minutes), so it is
/// opt-in; the heuristic variant is the everyday check.
#[test_case(false; "heuristic")]
#[test_case(true => ignore["BEAM search over five kernels takes minutes"]; "beam")]
fn test_sdpa_cuda_masked_f16_matches_cpu(beam: bool) {
    use crate::{CpuBackend, PrepareConfig};
    use svod_dtype::DeviceSpec;
    use svod_schedule::{BeamConfig, OptStrategy, OptimizerConfig};

    svod_schedule::testing::setup_test_tracing();
    let Some(config) = PrepareConfig::for_cuda_if_available() else {
        eprintln!("skipped: default device is not a CUDA GPU");
        return;
    };
    let case = SdpaCase { b: 8, h: 16, t: 1024, d: 64 };
    let build = |q: &Tensor, k: &Tensor, v: &Tensor, lens: &Tensor| {
        q.scaled_dot_product_attention()
            .key(k)
            .value(v)
            .is_causal(false)
            .attn_mask(&case.key_mask(lens))
            .call()
            .unwrap()
            .try_permute(&[0, 2, 1, 3])
            .unwrap()
            .cast(DType::Float32)
            .unwrap()
    };
    let cpu = PrepareConfig::for_cpu_backend(CpuBackend::Llvm);
    let expected = case.run(DeviceSpec::Cpu, &cpu, &cpu, build);
    let optimizer = if beam {
        OptimizerConfig::builder()
            .strategy(OptStrategy::Beam { width: 2 })
            .beam(BeamConfig::builder().beam_width(2).build())
            .build()
    } else {
        OptimizerConfig::builder().strategy(OptStrategy::Heuristic).build()
    };
    let cuda = PrepareConfig { optimizer, ..config };
    let actual = case.run(DeviceSpec::Cuda { device_id: 0 }, &cuda, &cuda, build);
    assert_all_close(&actual, &expected, 2e-2);
}

/// The BEAM plan GigaAM's masked `Q·Kᵀ` kernel picked on sm_86: a tensor-core
/// WARP axis plus three size-2 LOCALs, four local dims for CUDA's three. The
/// extra local must fold into `tid.y`/`tid.z`, never into the warp's `tid.x`:
/// with the warp in the high bits of `tid.x` every `mma.sync` lane read the
/// wrong fragment and the transcript came out empty. CUDA only; the CPU result
/// is the reference.
#[test]
fn test_sdpa_scores_cuda_tc_warp_with_three_locals_matches_cpu() {
    use crate::{CpuBackend, PrepareConfig};
    use svod_dtype::{DeviceSpec, ScalarDType};
    use svod_ir::{ConstValue, Opt, OptArg, OptOps};
    use svod_schedule::{OptStrategy, OptimizerConfig};

    svod_schedule::testing::setup_test_tracing();
    let Some(config) = PrepareConfig::for_cuda_if_available() else {
        eprintln!("skipped: default device is not a CUDA GPU");
        return;
    };
    if !crate::config::cuda_test_arch().is_some_and(|arch| arch.has_bf16_mma()) {
        eprintln!("skipped: no m16n8k16 tensor cores");
        return;
    }
    let case = SdpaCase { b: 8, h: 16, t: 256, d: 64 };
    let build = |q: &Tensor, k: &Tensor, _v: &Tensor, lens: &Tensor| {
        let keep = case.key_mask(lens).logical_not().unwrap();
        let kt = k.try_transpose(-1, -2).unwrap();
        let scores = q.matmul_with().other(&kt).dtype(DType::Float32).call().unwrap();
        let scores = scores.try_mul(&Tensor::const_(0.125f64, DType::Float32)).unwrap();
        scores.where_(&keep, &Tensor::const_(ConstValue::min(ScalarDType::Float32), DType::Float32)).unwrap()
    };
    let cpu = PrepareConfig::for_cpu_backend(CpuBackend::Llvm);
    let expected = case.run(DeviceSpec::Cpu, &cpu, &cpu, build);
    let swap = |axis: usize, other_axis: usize| Opt::new(OptOps::SWAP, Some(axis), OptArg::Swap { other_axis });
    let plan = vec![
        Opt::new(OptOps::TC, Some(0), OptArg::TensorCore { tc_select: -1, opt_level: 0, use_tc: 1 }),
        Opt::upcast(3, 4),
        Opt::upcast(2, 4),
        swap(2, 3),
        Opt::new(OptOps::UNROLL, Some(0), OptArg::Int(0)),
        Opt::upcast(2, 2),
        Opt::local(2, 2),
        Opt::local(1, 2),
        Opt::local(2, 2),
        swap(0, 1),
    ];
    let optimizer = OptimizerConfig::builder().strategy(OptStrategy::Heuristic).opts_to_apply(plan).build();
    let forced = PrepareConfig { optimizer, ..config.clone() };
    let actual = case.run(DeviceSpec::Cuda { device_id: 0 }, &config, &forced, build);
    assert_all_close(&actual, &expected, 1e-2);
}
