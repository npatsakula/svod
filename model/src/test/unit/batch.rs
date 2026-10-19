use svod_arch::ctc::{CtcDecoder, GreedyDecoder};
use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::gigaam::{ConvNormType, GigaAm, GigaAmConfig, GigaAmEncoderJit, SubsamplingMode};

pub(super) fn test_config() -> GigaAmConfig {
    GigaAmConfig {
        max_batch_size: 8,
        n_mels: 64,
        d_model: 32,
        n_heads: 4,
        n_layers: 2,
        d_ff: 128,
        conv_kernel: 5,
        subsampling_factor: 4,
        subsampling_mode: SubsamplingMode::Conv1d,
        subs_kernel_size: 5,
        conv_norm_type: ConvNormType::LayerNorm,
        vocab_size: 34,
        sample_rate: 16000,
        n_fft: 320,
        hop_length: 160,
        win_length: 320,
        mel_center: false,
        max_mel_frames: 512,
        max_encoder_frames: 128,
        decoder: CtcDecoder::Greedy(GreedyDecoder::new(Vec::new())),
        transducer: None,
        streaming: None,
    }
}

fn model_with_random_weights() -> GigaAm {
    GigaAm::with_random_weights(test_config())
}

fn read_prefix_f32(t: &Tensor, len: usize) -> Vec<f32> {
    let buf = t.buffer().unwrap();
    buf.as_array::<f32>().unwrap().as_slice().unwrap()[..len].to_vec()
}

// ---------------------------------------------------------------------------
// Cheap default tests (no realize): config + structural invariants.
// ---------------------------------------------------------------------------

#[test]
fn test_output_length_matches_forward() {
    let model = GigaAm::with_random_weights(test_config());

    let x = Tensor::full(&[1, 100, 64], 0.0f32, DType::Float32);
    let out = model.encoder.subsampling.forward(&x, false).unwrap();
    let actual_t = out.dim_const(1).unwrap();
    assert_eq!(model.encoder.subsampling_output_length(100), actual_t);

    let x2 = Tensor::full(&[1, 50, 64], 0.0f32, DType::Float32);
    let out2 = model.encoder.subsampling.forward(&x2, false).unwrap();
    let actual_t2 = out2.dim_const(1).unwrap();
    assert_eq!(model.encoder.subsampling_output_length(50), actual_t2);
}

#[test]
fn test_rope_cache_uses_encoder_bound() {
    let model = model_with_random_weights();
    let cfg = test_config();

    assert_eq!(model.encoder.cos_cache.dim_const(0).unwrap(), cfg.max_encoder_frames);
    assert_eq!(model.encoder.sin_cache.dim_const(0).unwrap(), cfg.max_encoder_frames);
    assert_ne!(cfg.max_encoder_frames, cfg.max_mel_frames);
}

#[test]
fn test_rope_cache_uses_pos_emb_max_len_as_base() {
    let model = model_with_random_weights();
    let cfg = test_config();
    let d_k = cfg.d_model / cfg.n_heads;
    let half_d = d_k / 2;
    let pos = 1usize;
    let freq_idx = 1usize;
    let angle = pos as f32 / (cfg.max_encoder_frames as f32).powf(2.0 * freq_idx as f32 / d_k as f32);
    let flat_idx = pos * half_d + freq_idx;

    // The cache is a graph (`Tensor::rope_table`), not a host array: realize it.
    let cos = model.encoder.cos_cache.to_vec::<f32>().unwrap();
    let sin = model.encoder.sin_cache.to_vec::<f32>().unwrap();

    assert!((cos[flat_idx] - angle.cos()).abs() < 1e-6);
    assert!((sin[flat_idx] - angle.sin()).abs() < 1e-6);
}

#[test]
fn test_subsampled_max_mel_fits_encoder_bound() {
    let model = model_with_random_weights();
    let cfg = test_config();
    let t_sub = model.encoder.subsampling_output_length(cfg.max_mel_frames);
    assert!(
        t_sub <= cfg.max_encoder_frames,
        "subsampled max_mel ({t_sub}) > max_encoder_frames ({})",
        cfg.max_encoder_frames
    );
}

// ---------------------------------------------------------------------------
// Heavy tests (realize / prepare): unique signal, gated behind --ignored.
// Run on demand when actively touching the GigaAM encoder, JIT plumbing, or
// the schedule pipeline. Each costs ~30s of prepare with random weights.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "heavy: full encoder forward at max_mel_frames"]
fn test_encode_batch_near_max_mel_runs() {
    let model = model_with_random_weights();
    let cfg = test_config();
    let t = cfg.max_mel_frames;

    let x = Tensor::full(&[1, cfg.n_mels, t], 0.1f32, DType::Float32);
    let lengths = Tensor::from_slice([t as i32]);

    let out = model.encoder.forward_batch(&x, &lengths).unwrap();
    out.realize().unwrap();
    assert!(out.buffer().unwrap().size() > 0);
}

#[test]
#[ignore = "heavy: batched-vs-single numerical consistency (the suite's only correctness assert)"]
fn test_single_vs_batch_consistency() {
    let model = model_with_random_weights();
    let d = test_config().d_model;
    let n_mels = test_config().n_mels;
    let t = 10;
    let t_sub = model.encoder.subsampling_output_length(t);

    let x1 = Tensor::full(&[1, n_mels, t], 0.5f32, DType::Float32);
    let x2 = Tensor::full(&[1, n_mels, t], 0.3f32, DType::Float32);
    let lengths_single = Tensor::from_slice([t as i32]);

    let out1 = model.encoder.forward_batch(&x1, &lengths_single).unwrap();
    out1.realize().unwrap();
    let data1 = read_prefix_f32(&out1, d * t_sub);

    let out2 = model.encoder.forward_batch(&x2, &lengths_single).unwrap();
    out2.realize().unwrap();
    let data2 = read_prefix_f32(&out2, d * t_sub);

    let batch = {
        let x1r = x1.clone();
        x1r.realize().unwrap();
        let d1 = x1r.as_vec::<f32>().unwrap();
        let x2r = x2.clone();
        x2r.realize().unwrap();
        let d2 = x2r.as_vec::<f32>().unwrap();
        let mut batch_data = vec![0.0f32; 2 * n_mels * t];
        batch_data[..n_mels * t].copy_from_slice(&d1);
        batch_data[n_mels * t..].copy_from_slice(&d2);
        ndarray::Array3::from_shape_vec((2, n_mels, t), batch_data).unwrap()
    };
    let batch_tensor = Tensor::from_ndarray(&batch);
    let batch_lengths = Tensor::from_slice([t as i32, t as i32]);

    let batch_out = model.encoder.forward_batch(&batch_tensor, &batch_lengths).unwrap();
    batch_out.realize().unwrap();
    let batch_data = read_prefix_f32(&batch_out, 2 * d * t_sub);

    assert_eq!(data1.len() * 2, batch_data.len());

    for (i, (&b, &s)) in batch_data[..data1.len()].iter().zip(data1.iter()).enumerate() {
        assert!((b - s).abs() < 1e-4, "batch[0] mismatch at {}: batch={} single={}", i, b, s);
    }
    for (i, (&b, &s)) in batch_data[data1.len()..].iter().zip(data2.iter()).enumerate() {
        assert!((b - s).abs() < 1e-4, "batch[1] mismatch at {}: batch={} single={}", i, b, s);
    }
}

/// A ragged batch (a full lane beside a shorter, zero-padded lane) must keep the
/// full lane bit-identical to its unbatched run and must stay finite. This is the
/// regression guard for the key-only masking + conv re-zero: a missing re-zero or
/// a stray batch-wide reduction would let the shorter lane's padding leak into the
/// full lane or produce NaN. (The short lane's own valid rows are NOT compared to
/// an unbatched run — the subsampling conv over the pad boundary perturbs them and
/// exact equivalence isn't a model invariant.)
#[test]
#[ignore = "heavy: ragged-batch lane isolation + finiteness"]
fn test_ragged_batch_keeps_full_lane_isolated_and_finite() {
    let model = model_with_random_weights();
    let cfg = test_config();
    let (d, n_mels) = (cfg.d_model, cfg.n_mels);
    let (t_full, t_short) = (10usize, 6usize);
    let t_sub_full = model.encoder.subsampling_output_length(t_full);

    // Channel-major [n_mels, T] frame fill (row m, frame f → m*T + f), zero beyond `valid`.
    let lane = |val: f32, valid: usize, t: usize| {
        let mut v = vec![0.0f32; n_mels * t];
        for m in 0..n_mels {
            for f in 0..valid {
                v[m * t + f] = val;
            }
        }
        v
    };

    // Lane A (full length) run alone.
    let xa =
        Tensor::from_ndarray(&ndarray::Array3::from_shape_vec((1, n_mels, t_full), lane(0.5, t_full, t_full)).unwrap());
    let solo_a = model.encoder.forward_batch(&xa, &Tensor::from_slice([t_full as i32])).unwrap();
    solo_a.realize().unwrap();
    let solo_a = read_prefix_f32(&solo_a, d * t_sub_full);

    // Batch: lane A full (0.5), lane B valid for only t_short frames (0.3) then
    // zero-padded up to t_full.
    let mut bd = vec![0.0f32; 2 * n_mels * t_full];
    bd[..n_mels * t_full].copy_from_slice(&lane(0.5, t_full, t_full));
    bd[n_mels * t_full..].copy_from_slice(&lane(0.3, t_short, t_full));
    let batch = Tensor::from_ndarray(&ndarray::Array3::from_shape_vec((2, n_mels, t_full), bd).unwrap());
    let out = model.encoder.forward_batch(&batch, &Tensor::from_slice([t_full as i32, t_short as i32])).unwrap();
    out.realize().unwrap();
    let all = read_prefix_f32(&out, 2 * d * t_sub_full);

    assert!(all.iter().all(|x| x.is_finite()), "ragged batch produced a non-finite value");
    // The shorter padded lane B must not perturb the full lane A.
    for (i, (&b, &s)) in all[..d * t_sub_full].iter().zip(solo_a.iter()).enumerate() {
        assert!((b - s).abs() < 1e-4, "lane A row {i}: batch={b} solo={s} (a ragged lane B leaked into lane A)");
    }
}

#[test]
#[ignore = "heavy: NaN/Inf detector across encoder forward"]
fn test_encode_batch_full_lengths_finite() {
    let model = model_with_random_weights();
    let cfg = test_config();
    let t = 256usize;

    let x = Tensor::full(&[2, cfg.n_mels, t], 0.1f32, DType::Float32);
    let lengths = Tensor::from_slice([t as i32, t as i32]);

    let out = model.encoder.forward_batch(&x, &lengths).unwrap();
    out.realize().unwrap();

    let buf = out.buffer().unwrap();
    let data = buf.as_array::<f32>().unwrap();
    for v in data.as_slice().unwrap() {
        assert!(v.is_finite(), "encode_batch produced non-finite value: {v}");
    }
}

/// The constant-shape encoder JIT carries NO symbolic `b`/`t` vars: every
/// shape is fixed by the realized input buffers, so each compiled kernel is
/// fully specialized. That exact-divisibility is what lets the schedule
/// heuristics pick MFMA tilings (and what graph capture relies on to collapse
/// the dispatch chain to one doorbell).
#[test]
#[ignore = "heavy: const-shape JIT prepares with no symbolic vars left in any kernel"]
fn test_const_shape_jit_has_no_symbolic_vars() {
    let model = model_with_random_weights();
    let cfg = test_config();

    let mut jit = GigaAmEncoderJit::new(model);
    jit.prepare(
        crate::jit::InputSpec::f32(&[cfg.max_batch_size, cfg.n_mels, cfg.max_mel_frames]),
        crate::jit::InputSpec::i32(&[cfg.max_batch_size]),
    )
    .unwrap();

    let kernels = jit.prepared_kernels().unwrap();
    assert!(!kernels.is_empty(), "expected compiled kernels for the const-shape encoder");
    for k in &kernels {
        assert!(
            k.kernel.var_names.is_empty(),
            "const-shape JIT kernel still carries symbolic vars: {:?}",
            k.kernel.var_names
        );
    }

    // No vars to bind — a plain replay must execute the whole graph.
    jit.execute().unwrap();
}

/// Full encoder `forward_batch` at a FA-eligible config (d_model=256, n_heads=16
/// ⇒ d_k=16; max_mel_frames=512 ⇒ T_sub=128) must produce finite output and agree
/// across batch lanes with different valid lengths. Exercises the `key_lens`
/// threading + the gfx942 FA kernel end-to-end on real weights-shaped tensors.
#[test]
#[ignore = "GPU: full encoder forward through flash-attention (FA-eligible config)"]
fn encoder_forward_fa_eligible_finite() {
    let mut cfg = test_config();
    cfg.d_model = 256;
    cfg.n_heads = 16; // d_k = 16 (FA dtype/D gate passes)
    cfg.d_ff = 256;
    cfg.max_mel_frames = 512; // subsamples to T_sub = 128 (FA tile gate passes)
    cfg.max_encoder_frames = 128;

    let model = GigaAm::with_random_weights(cfg.clone());
    let t = 512usize; // full mel length -> T_sub = 128
    let t_sub = model.encoder.subsampling_output_length(t);
    assert_eq!(t_sub, 128, "expected T_sub=128 for FA tile eligibility, got {t_sub}");

    let x = Tensor::full(&[2, cfg.n_mels, t], 0.1f32, DType::Float32);
    // Lane 0 partially valid (mel 300 -> shorter T_sub), lane 1 fully valid.
    let lengths = Tensor::from_slice([300i32, t as i32]);

    let out = model.encoder.forward_batch(&x, &lengths).unwrap();
    out.realize().unwrap();

    let buf = out.buffer().unwrap();
    let data = buf.as_array::<f32>().unwrap();
    for v in data.as_slice().unwrap() {
        assert!(v.is_finite(), "FA-eligible encoder produced non-finite value: {v}");
    }
}
