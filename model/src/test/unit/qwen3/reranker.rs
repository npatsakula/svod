use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::Module;

use crate::qwen3::{Qwen3Config, Qwen3Reranker};

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
    let mut reranker = Qwen3Reranker::empty(tiny_cfg());
    reranker.yes_loc = 5; // within tiny vocab_size=100
    let ids = Tensor::from_slice([0i32, 1, 2, 3, 4, 5, 6, 7]).try_reshape([1isize, 8]).unwrap();
    let lengths = Tensor::from_slice([8i32]);

    let out = reranker.forward(&ids, &lengths).unwrap();
    out.realize().unwrap();
    let s = out.dims().unwrap();
    // (B,) scalar score per row (after sigmoid)
    assert_eq!(s.len(), 1);
    assert_eq!(s[0], 1);

    let v = out.as_vec::<f32>().unwrap();
    assert!(v.iter().all(|x| x.is_finite()));
    // sigmoid output must be in (0, 1)
    assert!(v[0] > 0.0 && v[0] < 1.0);
}

#[test]
fn lm_head_tied_to_embeddings() {
    let reranker = Qwen3Reranker::empty(tiny_cfg());
    // The lm_head_weight should have the same shape as embed_tokens.weight
    let lm_shape = reranker.lm_head_weight.dims().unwrap();
    assert_eq!(lm_shape[0], 100); // vocab_size
    assert_eq!(lm_shape[1], 64); // hidden_size
}

/// The reranker nests the backbone under `model.` — the published
/// `Qwen3ForCausalLM` layout — and stores no LM head of its own (it is tied to
/// `embed_tokens.weight` and resolved on load).
#[test]
fn state_dict_nests_backbone_under_model_prefix() {
    let reranker = Qwen3Reranker::empty(tiny_cfg());
    let sd = reranker.state_dict("");
    assert!(sd.contains_key("model.embed_tokens.weight"));
    assert!(sd.contains_key("model.layers.0.self_attn.q_proj.weight"));
    assert!(sd.contains_key("model.norm.weight"));
    assert!(!sd.contains_key("lm_head.weight"), "the LM head is tied, never stored");

    // The tie is resolved on load: a fresh reranker picks the embedding table up.
    let mut reloaded = Qwen3Reranker::empty(tiny_cfg());
    reloaded.load_state_dict(&sd, "").unwrap();
    let want = sd["model.embed_tokens.weight"].clone();
    want.realize().unwrap();
    reloaded.lm_head_weight.realize().unwrap();
    assert_eq!(reloaded.lm_head_weight.as_vec::<f32>().unwrap(), want.as_vec::<f32>().unwrap());

    // A non-empty prefix nests every key under it, with no leading dot.
    let mut got: Vec<String> = reranker.state_dict("m").keys().cloned().collect();
    let mut want: Vec<String> = sd.keys().map(|k| format!("m.{k}")).collect();
    got.sort();
    want.sort();
    assert_eq!(got, want);
}
