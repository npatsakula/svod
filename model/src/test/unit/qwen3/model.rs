use svod_dtype::DType;
use svod_tensor::Tensor;
use test_case::test_case;

use svod_tensor::nn::{Module, StateDict};

use crate::qwen3::{Qwen3Config, Qwen3Embedding, Qwen3Model};

fn tiny_cfg() -> Qwen3Config {
    Qwen3Config {
        vocab_size: 100,
        hidden_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 32,
        intermediate_size: 128,
        max_position_embeddings: 64,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        attention_bias: false,
        tie_word_embeddings: true,
        pad_token_id: 0,
        dtype: DType::Float32,
    }
}

#[test]
fn forward_output_shape() {
    let model = Qwen3Model::empty(tiny_cfg());
    let ids = Tensor::from_slice([0i32, 1, 2, 3, 4, 5, 6, 7]).try_reshape([1isize, 8]).unwrap();

    let out = model.forward(&ids).unwrap();
    out.realize().unwrap();
    let s = out.dims().unwrap();
    assert_eq!(s[0], 1);
    assert_eq!(s[1], 8);
    assert_eq!(s[2], 64);

    let v = out.as_vec::<f32>().unwrap();
    assert!(v.iter().all(|x| x.is_finite()));
}

#[test]
fn embedding_output_shape() {
    let emb = Qwen3Embedding::empty(tiny_cfg());
    let ids = Tensor::from_slice([0i32, 1, 2, 3, 4, 5, 6, 7]).try_reshape([1isize, 8]).unwrap();
    let lengths = Tensor::from_slice([8i32]);

    let out = emb.encode(&ids, &lengths).unwrap();
    out.realize().unwrap();
    let s = out.dims().unwrap();
    assert_eq!(s[0], 1);
    assert_eq!(s[1], 64);

    let v = out.as_vec::<f32>().unwrap();
    assert!(v.iter().all(|x| x.is_finite()));
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-4, "unit norm, got {norm}");
}

fn realized(t: &Tensor) -> Vec<f32> {
    t.realize().unwrap();
    t.as_vec::<f32>().unwrap()
}

/// Right padding is invisible under the causal mask: the real positions of a
/// padded row equal the unpadded forward, and a row's embedding does not
/// depend on the other rows of its batch.
#[test]
fn right_padding_leaves_real_tokens_unchanged() {
    let cfg = tiny_cfg();
    let hidden = cfg.hidden_size;
    let emb = Qwen3Embedding::empty(cfg);
    let ids: Vec<i32> = (0..5).collect();
    let alone = realized(&emb.model.forward(&Tensor::from_slice(&ids).try_reshape([1isize, 5]).unwrap()).unwrap());

    let mut padded: Vec<i32> = ids.clone();
    padded.extend([7, 7, 7]);
    let mut other: Vec<i32> = (10..18).collect();
    other.reverse();
    let batch = Tensor::from_slice([padded, other].concat()).try_reshape([2isize, 8]).unwrap();
    let both = realized(&emb.model.forward(&batch).unwrap());
    let worst = alone.iter().zip(&both[..5 * hidden]).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(worst < 1e-4, "real positions drifted by {worst}");

    let lengths = Tensor::from_slice([5i32, 8]);
    let pooled = realized(&emb.encode(&batch, &lengths).unwrap());
    let alone_pooled = realized(
        &emb.encode(&Tensor::from_slice(&ids).try_reshape([1isize, 5]).unwrap(), &Tensor::from_slice([5i32])).unwrap(),
    );
    let worst = alone_pooled.iter().zip(&pooled[..hidden]).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(worst < 1e-4, "pooled row drifted by {worst}");
    // Pooling at the padded end instead reads a pad token: a different vector.
    let at_end = realized(&emb.encode(&batch, &Tensor::from_slice([8i32, 8])).unwrap());
    let moved = alone_pooled.iter().zip(&at_end[..hidden]).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(moved > 1e-3, "pooling the pad token must differ, moved {moved}");
}

/// Two sequences packed end to end in one row, each restarting its positions,
/// equal their separate forwards: the segment mask hides the first from the
/// second, and the per-token rope matches the per-position prefix.
#[test]
fn packed_rows_match_separate_forwards() {
    use crate::qwen3::Packing;
    let cfg = tiny_cfg();
    let hidden = cfg.hidden_size;
    let model = Qwen3Model::empty(cfg);
    let (a, b): (Vec<i32>, Vec<i32>) = ((1..6).collect(), (10..13).collect());
    let alone = |ids: &[i32]| {
        realized(&model.forward(&Tensor::from_slice(ids).try_reshape([1isize, ids.len() as isize]).unwrap()).unwrap())
    };
    let (want_a, want_b) = (alone(&a), alone(&b));

    // `[a | b | pad pad]`: the pads are their own one-token segments.
    let len = a.len() + b.len() + 2;
    let ids =
        Tensor::from_slice([a.clone(), b.clone(), vec![0, 0]].concat()).try_reshape([1isize, len as isize]).unwrap();
    let positions: Vec<i32> = (0..a.len() as i32).chain(0..b.len() as i32).chain([0, 0]).collect();
    let seg_start: Vec<i32> = vec![0; a.len()]
        .into_iter()
        .chain(vec![a.len() as i32; b.len()])
        .chain([len as i32 - 2, len as i32 - 1])
        .collect();
    let positions = Tensor::from_slice(positions).try_reshape([1isize, len as isize]).unwrap();
    let seg_start = Tensor::from_slice(seg_start).try_reshape([1isize, len as isize]).unwrap();
    let got = realized(&model.forward_packed(&ids, &Packing { positions: &positions, seg_start: &seg_start }).unwrap());

    let worst = |got: &[f32], want: &[f32]| got.iter().zip(want).map(|(g, w)| (g - w).abs()).fold(0.0f32, f32::max);
    assert!(worst(&got[..a.len() * hidden], &want_a) < 1e-4, "the first segment drifted");
    assert!(worst(&got[a.len() * hidden..(a.len() + b.len()) * hidden], &want_b) < 1e-4, "the second segment drifted");
    assert!(got.iter().all(|x| x.is_finite()), "the one-token pad segments must stay finite");
    // Without the segment mask the second sequence sees the first.
    let seg_start = Tensor::from_slice(vec![0i32; len]).try_reshape([1isize, len as isize]).unwrap();
    let leaked =
        realized(&model.forward_packed(&ids, &Packing { positions: &positions, seg_start: &seg_start }).unwrap());
    assert!(worst(&leaked[a.len() * hidden..(a.len() + b.len()) * hidden], &want_b) > 1e-3, "the mask must matter");
}

/// Every key `#[derive(Module)]` emits for a 2-layer tiny backbone, in the
/// published `Qwen3Model` naming. Drift here is a checkpoint-compatibility break.
fn expected_keys() -> Vec<String> {
    let mut keys = vec!["embed_tokens.weight".to_string(), "norm.weight".to_string()];
    for i in 0..2 {
        for k in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ] {
            keys.push(format!("layers.{i}.{k}"));
        }
    }
    keys.sort();
    keys
}

fn sorted_keys(sd: &StateDict) -> Vec<String> {
    let mut keys: Vec<String> = sd.keys().cloned().collect();
    keys.sort();
    keys
}

/// The emitted key set is exactly the published layout, at the root and nested
/// under a prefix (which must not grow a leading dot).
#[test]
fn state_dict_keys_match_published_layout() {
    let model = Qwen3Model::empty(tiny_cfg());
    let want = expected_keys();
    assert_eq!(sorted_keys(&model.state_dict("")), want);

    let want_nested: Vec<String> = want.iter().map(|k| format!("m.{k}")).collect();
    assert_eq!(sorted_keys(&model.state_dict("m")), want_nested);
}

/// The embedding wrapper is transparent: it emits the backbone's keys verbatim.
#[test]
fn embedding_state_dict_is_transparent() {
    let emb = Qwen3Embedding::empty(tiny_cfg());
    assert_eq!(sorted_keys(&emb.state_dict("")), expected_keys());
}

#[test]
fn state_dict_round_trip() {
    let model = Qwen3Model::empty(tiny_cfg());
    let sd = model.state_dict("");

    let mut model2 = Qwen3Model::empty(tiny_cfg());
    model2.load_state_dict(&sd, "").unwrap();
    assert_eq!(sorted_keys(&sd), sorted_keys(&model2.state_dict("")));
}

#[test]
fn key_count_matches_checkpoint_layout() {
    let cfg = qwen3_0_6b_structural();
    let model = Qwen3Model::empty(cfg);
    let sd = model.state_dict("");
    // embed + 28 × (input_ln, q/k/v/o, q/k_norm, post_ln, gate/up/down) + norm
    let expected = 1 + 28 * 11 + 1;
    assert_eq!(sd.len(), expected, "expected {expected} keys, got {}", sd.len());
}

fn qwen3_0_6b_structural() -> Qwen3Config {
    Qwen3Config {
        vocab_size: 151_669,
        hidden_size: 1024,
        num_hidden_layers: 28,
        num_attention_heads: 16,
        num_key_value_heads: 8,
        head_dim: 128,
        intermediate_size: 3072,
        max_position_embeddings: 32_768,
        rms_norm_eps: 1e-6,
        rope_theta: 1_000_000.0,
        attention_bias: false,
        tie_word_embeddings: true,
        pad_token_id: 151_643,
        dtype: DType::Float32,
    }
}

/// A config the tk3 kernels apply to on a supported device: a 128-wide hidden
/// state and head dim, and grouped query heads.
fn fusable_cfg() -> Qwen3Config {
    Qwen3Config {
        hidden_size: 128,
        head_dim: 128,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        max_position_embeddings: 128,
        dtype: crate::default_compute_dtype(),
        ..tiny_cfg()
    }
}

/// `Qwen3Model::forward` carries the residual stream unsummed between layers so
/// each add is absorbed by the norm that reads it; the eager
/// `Qwen3DecoderLayer::forward` materializes it instead. The two must agree —
/// exactly on the graph path, and to bf16 rounding where the fused
/// `add_rms_norm` replaces the lazy add plus its norm.
#[test]
fn residual_stream_matches_the_eager_layer_chain() {
    use svod_tensor::nn::Layer;

    let cfg = fusable_cfg();
    let (len, dtype) = (128usize, cfg.dtype.clone());
    let model = Qwen3Model::empty(cfg);
    let ids = Tensor::from_slice((0..len as i32).collect::<Vec<_>>()).try_reshape([1isize, len as isize]).unwrap();

    // The sequence already tiles the attention kernel, so `forward` pads nothing
    // and the eager chain sees exactly the same rope prefix.
    let rope = model.rope_prefix(len).unwrap();
    let mut h = model.embeddings.forward(&ids).unwrap();
    for layer in &model.layers {
        h = layer.forward(&h, &rope).unwrap();
        h = h.contiguous();
        h.realize().unwrap();
    }
    let want = realized(&model.norm.forward(&h).unwrap().cast(DType::Float32).contiguous());
    let got = realized(&model.forward(&ids).unwrap().cast(DType::Float32).contiguous());

    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0f32, |a, b| a.max(b.abs())).max(f32::MIN_POSITIVE);
    let err = got.iter().zip(&want).fold(0f32, |a, (g, w)| a.max((g - w).abs())) / scale;
    // f32 is exact; a 16-bit stream differs only by the rounding of a different
    // summation order in the row reduce — under two bf16 ulps.
    let tol = if dtype == DType::Float32 { 1e-6 } else { 8e-3 };
    assert!(err < tol, "residual stream diverges from the eager chain: relative error {err} (tol {tol})");
}

/// `forward` rounds a sequence up to the attention tile and narrows the
/// rope cache to that padded length, so the cache must cover it for every
/// admissible length — including a context that is not a whole number of tiles.
#[test_case(64; "below one tile")]
#[test_case(100; "not a tile multiple")]
#[test_case(128; "exactly one tile")]
#[test_case(200; "between tiles")]
fn rope_cache_covers_the_padded_context(max_positions: usize) {
    let model = Qwen3Model::empty(Qwen3Config { max_position_embeddings: max_positions, ..tiny_cfg() });
    let padded = max_positions.next_multiple_of(crate::qwen3::SEQUENCE_TILE);
    let (cos, sin) = model.rope_prefix(padded).expect("the cache must reach the padded context");
    assert_eq!(cos.dim_const(1).unwrap(), padded);
    assert_eq!(sin.dim_const(1).unwrap(), padded);
}

/// Rounding the cache up to the tile appends rows; the positions the model
/// already had must not move.
#[test]
fn rope_cache_rounding_leaves_positions_put() {
    let rows = |max_positions: usize| {
        let model = Qwen3Model::empty(Qwen3Config { max_position_embeddings: max_positions, ..tiny_cfg() });
        let (cos, sin) = model.rope_prefix(100).expect("100 positions");
        (realized(&cos.contiguous()), realized(&sin.contiguous()))
    };
    assert_eq!(rows(100), rows(128), "the rounded cache must agree with an exact one on their shared rows");
}

/// `lengths` names each row's real token count and the pooled token is the one
/// at `lengths - 1`. Out of `1..=L` there is no such token: the gather's one-hot
/// would match no position and the row would pool to zeros — a NaN once the
/// embedding normalizes it — so the index saturates into the row instead, the
/// same rule `Qwen3Embedder` applies when it embeds an empty row as one pad token.
#[test]
fn out_of_range_lengths_saturate_into_the_row() {
    let emb = Qwen3Embedding::empty(tiny_cfg());
    let ids = Tensor::from_slice([3i32, 1, 4, 1, 5, 9, 2, 6]).try_reshape([1isize, 8]).unwrap();
    let at = |len: i32| realized(&emb.encode(&ids, &Tensor::from_slice([len])).unwrap());

    let (first, last) = (at(1), at(8));
    assert!(first.iter().chain(&last).all(|x| x.is_finite()), "a well-formed length pools a real token");
    assert_ne!(first, last, "the tiny model must separate the first token from the last");
    assert_eq!(at(0), first, "a zero length pools the row's first token");
    assert_eq!(at(9), last, "a length past the row pools its last token");
}

/// The reranker pools through the same `last_token`, so a degenerate length
/// scores a real token rather than a row of zeros (a flat 0.5 after sigmoid).
#[test]
fn reranker_scores_a_real_token_at_a_zero_length() {
    use crate::qwen3::Qwen3Reranker;

    let mut reranker = Qwen3Reranker::empty(tiny_cfg());
    reranker.yes_loc = 5;
    let ids = Tensor::from_slice([3i32, 1, 4, 1, 5, 9, 2, 6]).try_reshape([1isize, 8]).unwrap();
    let at = |len: i32| realized(&reranker.forward(&ids, &Tensor::from_slice([len])).unwrap());

    assert_eq!(at(0), at(1), "a zero length scores the row's first token");
    assert!(at(0)[0] != 0.5, "a row of zeros would score exactly sigmoid(0)");
}

/// A sequence past the config's context is a caller bug named as one, not a
/// narrow that ran off the end of the rope cache.
#[test]
fn a_sequence_past_the_context_is_rejected() {
    let model = Qwen3Model::empty(Qwen3Config { max_position_embeddings: 4, ..tiny_cfg() });
    let ids = Tensor::from_slice([0i32; 5]).try_reshape([1isize, 5]).unwrap();
    let err = model.forward(&ids).expect_err("five tokens do not fit a four-position context");
    assert!(matches!(err, crate::qwen3::Error::ContextLength { seq_len: 5, max_position_embeddings: 4 }), "got {err}");
    // The whole context still runs, tile or no tile.
    model.forward(&Tensor::from_slice([0i32; 4]).try_reshape([1isize, 4]).unwrap()).expect("four tokens fit");
}

/// Norm weights kept at a different precision from the stream are the graph
/// path's business: the heads op rejects the mismatch, so the prologue must
/// take the graph and `forward` must still produce a hidden state.
#[test]
fn a_norm_weight_off_the_stream_dtype_falls_back_to_the_graph() {
    let mut model = Qwen3Model::empty(Qwen3Config { dtype: crate::default_compute_dtype(), ..fusable_cfg() });
    for layer in &mut model.layers {
        layer.input_layernorm.weight = layer.input_layernorm.weight.cast(DType::Float32);
        layer.post_attention_layernorm.weight = layer.post_attention_layernorm.weight.cast(DType::Float32);
        layer.attention.q_norm.weight = layer.attention.q_norm.weight.cast(DType::Float32);
    }
    let ids = Tensor::from_slice((0..128i32).collect::<Vec<_>>()).try_reshape([1isize, 128]).unwrap();
    let out =
        realized(&model.forward(&ids).expect("the graph path takes what the kernels will not").cast(DType::Float32));
    assert!(out.iter().all(|x| x.is_finite()));
}
