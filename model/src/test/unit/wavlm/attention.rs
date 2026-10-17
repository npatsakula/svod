use svod_dtype::DType;
use svod_tensor::nn::{Layer, Linear, Module};
use svod_tensor::{Tensor, s};
use test_case::test_case;

use crate::init::fan_in_uniform;
use crate::state::cast_all;
use crate::wavlm::{
    GatedRelPosAttention, WavLmConfig, compute_bucket_indices, compute_position_bias, wavlm_large_s80_md,
};

// ---------------------------------------------------------------------------
// Bucketing
// ---------------------------------------------------------------------------

/// Bidirectional bucketing with `num_buckets=8, max_distance=16`:
/// half-buckets = 4, max_exact = 2.
/// - rel=0: bucket 0
/// - rel=1: bucket 1+4 (positive side, exact)
/// - rel=-1: bucket 1 (negative side, exact)
/// - rel=2: bucket 2+4 = 6 (positive, just past max_exact: log-bucket)
/// - rel=3: bucket 2+4 = 6 (still in first log bucket: log(3/2)/log(8) * 2 = 0.585 → 0)
/// - rel=-7: bucket 3 (negative, max log-bucket clamp)
#[test]
fn bucket_indices_small_table() {
    // 4x4 grid: rel = k - q
    let buckets = compute_bucket_indices(4, 4, 8, 16);
    // Shape (4, 4) row-major: buckets[q * 4 + k] = bucket(rel = k - q)
    // Row q=0: rels 0, 1, 2, 3.
    assert_eq!(buckets[0], 0); // rel=0
    assert_eq!(buckets[1], 1 + 4); // rel=1 (positive, exact)
    // rel=2: just past max_exact=2. log(2/2)/log(8) * 2 = 0. → bucket = 2 + 0 = 2, then +4 (positive side).
    assert_eq!(buckets[2], 2 + 4);
    // rel=3: log(3/2)/log(8) * 2 ≈ 0.39 → floor 0; bucket = 2+0+4 = 6.
    assert_eq!(buckets[3], 2 + 4);

    // Row q=3: rels -3, -2, -1, 0.
    assert_eq!(buckets[12], 2); // rel=-3 → log-bucket=2, no sign offset
    assert_eq!(buckets[13], 2); // rel=-2 → log-bucket=2
    assert_eq!(buckets[14], 1); // rel=-1 → exact=1
    assert_eq!(buckets[15], 0); // rel=0
}

/// Bidirectional, `num_buckets=320, max_distance=800`: the production WavLM
/// parameters. Spot check: rel=0 is bucket 0; the largest log bucket is
/// `half - 1 = 159`; rel=799 should hit it for both positive and (after sign
/// offset) negative-sign reverse.
#[test]
fn bucket_indices_production_params_extremes() {
    let buckets = compute_bucket_indices(2, 800, 320, 800);
    assert_eq!(buckets[0], 0); // rel=0 (q=0,k=0)
    assert_eq!(buckets[1], 1 + 160); // rel=1
    assert_eq!(buckets[80], 80 + 160); // rel=80: first log bucket (max_exact=80) → bucket 80
    // rel=799 lands in the max log bucket = 159 (160 - 1), positive offset = 160.
    assert_eq!(buckets[799], 159 + 160);
    // Negative-side: q=1, k=0 gives rel=-1 → bucket 1, no offset.
    assert_eq!(buckets[800], 1);
}

// ---------------------------------------------------------------------------
// compute_position_bias
// ---------------------------------------------------------------------------

#[test]
fn position_bias_shape() {
    // rel_attn_embed: (num_buckets=320, total_num_heads=16)
    let rel_embed = Tensor::zeros(&[320, 16], DType::Float32);
    let bias = compute_position_bias(&rel_embed, 10, 10, 320, 800).unwrap();
    let shape = bias.dims().unwrap();
    assert_eq!(shape, vec![1, 16, 10, 10]);
}

// ---------------------------------------------------------------------------
// GatedRelPosAttention
// ---------------------------------------------------------------------------

/// Forward on a tiny shape with full heads (no pruning) preserves the
/// `(B, L, embed_dim)` shape contract.
#[test]
fn attention_forward_shape_full_heads() {
    // Synthetic config: 4 heads of dim 8 ⇒ embed_dim=32, all heads kept.
    let mut cfg = wavlm_large_s80_md();
    cfg.encoder_embed_dim = 32;
    cfg.encoder_head_dim = 8;
    cfg.encoder_total_num_heads = vec![4; cfg.encoder_num_layers];
    cfg.encoder_remaining_heads = (0..cfg.encoder_num_layers).map(|_| (0..4).collect()).collect();

    let attn = GatedRelPosAttention::empty(&cfg, 0);
    assert_eq!(attn.num_kept(), 4);

    let b = 1;
    let l = 8;
    let x = Tensor::zeros(&[b, l, cfg.encoder_embed_dim], DType::Float32);
    let rel_embed = Tensor::zeros(&[cfg.encoder_num_buckets, attn.total_num_heads], DType::Float32);
    let pb = compute_position_bias(&rel_embed, l, l, cfg.encoder_num_buckets, cfg.encoder_max_distance).unwrap();

    let out = attn.forward(&x, &pb).expect("symbolic forward");
    let shape = out.dims().unwrap();
    assert_eq!(shape, vec![b, l, cfg.encoder_embed_dim]);
}

/// Pruned-head case: kept heads = 3 out of 8. Q/K/V/O have the right shapes
/// and forward output is still `(B, L, embed_dim)`.
#[test]
fn attention_pruned_head_shapes() {
    let mut cfg = wavlm_large_s80_md();
    cfg.encoder_embed_dim = 64;
    cfg.encoder_head_dim = 8;
    cfg.encoder_total_num_heads = vec![8; cfg.encoder_num_layers];
    cfg.encoder_remaining_heads = (0..cfg.encoder_num_layers).map(|_| vec![2, 5, 7]).collect();

    let attn = GatedRelPosAttention::empty(&cfg, 0);
    assert_eq!(attn.num_kept(), 3);

    let q_shape = attn.q.weight.dims().unwrap();
    assert_eq!(q_shape, vec![3 * 8, 64], "q_weight should be (num_kept*head_dim, embed_dim)");
    let o_shape = attn.out.weight.dims().unwrap();
    assert_eq!(o_shape, vec![64, 3 * 8], "out_weight should be (embed_dim, num_kept*head_dim)");

    let gate_w: Vec<usize> = attn.gru_rel_pos_linear.weight.dims().unwrap();
    assert_eq!(gate_w, vec![8, 8], "gru_rel_pos_linear should be (8, head_dim)");
    let gate_c = attn.gru_rel_pos_const.dims().unwrap();
    assert_eq!(gate_c, vec![1, 8, 1, 1], "gru_rel_pos_const should be (1, total_num_heads, 1, 1) — NOT pruned");

    let l = 6;
    let x = Tensor::zeros(&[1, l, cfg.encoder_embed_dim], DType::Float32);
    let rel_embed = Tensor::zeros(&[cfg.encoder_num_buckets, attn.total_num_heads], DType::Float32);
    let pb = compute_position_bias(&rel_embed, l, l, cfg.encoder_num_buckets, cfg.encoder_max_distance).unwrap();
    let out = attn.forward(&x, &pb).expect("symbolic forward");
    let out_shape = out.dims().unwrap();
    assert_eq!(out_shape, vec![1, l, cfg.encoder_embed_dim]);
}

/// State-dict layout uses upstream Python names: `q_proj.weight/bias`, etc.,
/// plus the gating tensors. Round-trip preserves all keys.
#[test]
fn attention_state_dict_round_trip() {
    let cfg = sample_cfg();
    let attn = GatedRelPosAttention::empty(&cfg, 0);
    let sd = attn.state_dict("attn");

    for key in [
        "attn.q_proj.weight",
        "attn.q_proj.bias",
        "attn.k_proj.weight",
        "attn.k_proj.bias",
        "attn.v_proj.weight",
        "attn.v_proj.bias",
        "attn.out_proj.weight",
        "attn.out_proj.bias",
        "attn.gru_rel_pos_linear.weight",
        "attn.gru_rel_pos_linear.bias",
        "attn.gru_rel_pos_const",
    ] {
        assert!(sd.contains_key(key), "missing key: {key}");
    }

    let mut empty = GatedRelPosAttention::empty(&cfg, 0);
    empty.load_state_dict(&sd, "attn").expect("round-trip");
}

fn sample_cfg() -> WavLmConfig {
    let mut cfg = wavlm_large_s80_md();
    cfg.encoder_embed_dim = 32;
    cfg.encoder_head_dim = 8;
    cfg.encoder_total_num_heads = vec![4; cfg.encoder_num_layers];
    cfg.encoder_remaining_heads = (0..cfg.encoder_num_layers).map(|_| vec![0, 2]).collect();
    cfg
}

/// The `scaled_dot_product_attention` path must reproduce the formula upstream
/// spells out: gate the bucketed bias, keep the surviving heads, then
/// `softmax(scale·q @ kᵀ + bias) @ v`. Written out independently here so a
/// change to either side of the rewrite shows up.
#[test]
fn attention_matches_reference_formula() {
    let cfg = sample_cfg();
    let attn = GatedRelPosAttention::empty(&cfg, 0);
    let (l, embed) = (5usize, cfg.encoder_embed_dim);
    let x = ramp(&[1, l, embed]);
    let rel_embed = fan_in_uniform(&[cfg.encoder_num_buckets, attn.total_num_heads], 16, DType::Float32);
    let pb = compute_position_bias(&rel_embed, l, l, cfg.encoder_num_buckets, cfg.encoder_max_distance).unwrap();

    let got = attn.forward(&x, &pb).unwrap().to_vec::<f32>().unwrap();
    let want = reference_attention(&attn, &x, &pb, cfg.encoder_head_dim).to_vec::<f32>().unwrap();

    assert_eq!(got.len(), want.len());
    for (a, b) in got.iter().zip(&want) {
        assert!((a - b).abs() < 1e-5, "attention mismatch: {a} vs {b}");
    }
}

/// The 16-bit forward (tk3's flash attention with the gated bias on a GPU,
/// the graph elsewhere) against the f32 formula: pruned heads of width 64,
/// two batches, a length off every tile.
#[test_case(DType::Float16; "f16")]
#[test_case(DType::BFloat16; "bf16")]
fn half_attention_matches_reference_formula(dtype: DType) {
    let mut cfg = wavlm_large_s80_md();
    cfg.encoder_embed_dim = 256;
    cfg.encoder_head_dim = 64;
    cfg.encoder_total_num_heads = vec![4; cfg.encoder_num_layers];
    cfg.encoder_remaining_heads = (0..cfg.encoder_num_layers).map(|_| vec![0, 2, 3]).collect();
    let attn = GatedRelPosAttention::empty(&cfg, 0);
    let (l, embed) = (37usize, cfg.encoder_embed_dim);
    let x = ramp(&[2, l, embed]);
    let rel_embed = fan_in_uniform(&[cfg.encoder_num_buckets, attn.total_num_heads], 4, DType::Float32);
    let pb = compute_position_bias(&rel_embed, l, l, cfg.encoder_num_buckets, cfg.encoder_max_distance).unwrap();
    let want = reference_attention(&attn, &x, &pb, cfg.encoder_head_dim).to_vec::<f32>().unwrap();

    let mut half = attn.clone();
    half.load_state_dict(&cast_all(&attn.state_dict(""), dtype.clone()), "").unwrap();
    let got = half.forward(&x.cast(dtype.clone()), &pb.cast(dtype)).unwrap();
    let got = got.cast(DType::Float32).to_vec::<f32>().unwrap();
    assert_eq!(got.len(), want.len());
    let worst = got.iter().zip(&want).map(|(a, b)| (a - b).abs() / b.abs().max(1.0)).fold(0.0f32, f32::max);
    assert!(worst < 3e-2, "max relative error {worst}");
}

/// A deterministic non-constant input: `sin` over the flattened index.
fn ramp(shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
    let dims: Vec<isize> = shape.iter().map(|&d| d as isize).collect();
    Tensor::from_slice(&data).try_reshape(dims).unwrap()
}

/// Transcription of `WavLMSelfAttention.forward` + `SelfAttention.forward`
/// (components.py:429-486, 668-725) using only elementary tensor ops.
fn reference_attention(attn: &GatedRelPosAttention, x: &Tensor, pb: &Tensor, head_dim: usize) -> Tensor {
    let (h, nk) = (attn.total_num_heads as isize, attn.num_kept() as isize);
    let (b, l, hd) = (x.dim_const(0).unwrap() as isize, x.dim_const(1).unwrap() as isize, head_dim as isize);

    let query_layer = x.view([b, l, h, hd]).unwrap().try_permute(&[0, 2, 1, 3]).unwrap();
    let gate = attn
        .gru_rel_pos_linear
        .forward(&query_layer)
        .unwrap()
        .view([b, h, l, 2, 4])
        .unwrap()
        .sum_with()
        .axes(-1isize)
        .call()
        .unwrap()
        .sigmoid()
        .unwrap();
    let mut chunks = gate.chunk(2, -1).unwrap();
    let gate_b = chunks.pop().unwrap();
    let gate_a = chunks.pop().unwrap();
    let scale = gate_a
        .try_mul(gate_b.try_mul(&attn.gru_rel_pos_const).unwrap().try_sub(1.0).unwrap())
        .unwrap()
        .try_add(2.0)
        .unwrap();
    let mask = scale.try_mul(pb).unwrap().getitem(s![.., attn.remaining_heads.clone(), .., ..]).unwrap();

    let split = |lin: &Linear| lin.forward(x).unwrap().view([b, l, nk, hd]).unwrap();
    let q = split(&attn.q).try_transpose(2, 1).unwrap();
    let k = split(&attn.k).try_permute(&[0, 2, 3, 1]).unwrap();
    let v = split(&attn.v).try_transpose(2, 1).unwrap();

    let scaling = Tensor::const_((head_dim as f32).powf(-0.5), DType::Float32);
    let weights = q.try_mul(&scaling).unwrap().matmul(&k).unwrap().try_add(&mask).unwrap().softmax(-1).unwrap();
    let out = weights.matmul(&v).unwrap().try_transpose(2, 1).unwrap().view([b, l, nk * hd]).unwrap();
    attn.out.forward(&out).unwrap()
}
