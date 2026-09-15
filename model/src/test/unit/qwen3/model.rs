use svod_dtype::DType;
use svod_tensor::Tensor;

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

/// A config the hand kernels apply to on a supported device: a 128-wide hidden
/// state and head dim (both a multiple of the 32-lane wave, halving to 64), and
/// head counts a wave grid divides.
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
