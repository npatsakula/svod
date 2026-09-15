use svod_dtype::DType;

use crate::qwen3::{Qwen3Config, qwen3_embedding_0_6b};

#[test]
fn predefined_config_dims() {
    let cfg = qwen3_embedding_0_6b();
    assert_eq!(cfg.vocab_size, 151_669);
    assert_eq!(cfg.hidden_size, 1024);
    assert_eq!(cfg.num_hidden_layers, 28);
    assert_eq!(cfg.num_attention_heads, 16);
    assert_eq!(cfg.num_key_value_heads, 8);
    assert_eq!(cfg.head_dim, 128);
    assert_eq!(cfg.intermediate_size, 3072);
    assert_eq!(cfg.rope_theta, 1_000_000.0);
    assert_eq!(cfg.rms_norm_eps, 1e-6);
    assert!(!cfg.attention_bias);
    assert!(cfg.tie_word_embeddings);
}

#[test]
fn head_dim_not_derived() {
    let cfg = qwen3_embedding_0_6b();
    assert_ne!(cfg.head_dim, cfg.hidden_size / cfg.num_attention_heads);
    assert_eq!(cfg.num_attention_heads * cfg.head_dim, 2048);
}

#[test]
fn gqa_ratio() {
    let cfg = qwen3_embedding_0_6b();
    assert_eq!(cfg.num_kv_groups(), 2);
}

#[test]
fn json_parse() {
    let json = r#"{
        "model_type": "qwen3",
        "vocab_size": 100,
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "intermediate_size": 128,
        "max_position_embeddings": 512,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "attention_bias": false,
        "tie_word_embeddings": true,
        "pad_token_id": 0
    }"#;
    let cfg = Qwen3Config::from_json_str(json).unwrap();
    assert_eq!(cfg.vocab_size, 100);
    assert_eq!(cfg.hidden_size, 64);
    assert_eq!(cfg.head_dim, 32);
    assert_eq!(cfg.num_key_value_heads, 2);
    assert_eq!(cfg.rope_theta, 10000.0);
}

#[test]
fn merge_preserves_dtype() {
    let mut cfg = qwen3_embedding_0_6b();
    cfg.dtype = DType::Float32;

    let mut parsed = qwen3_embedding_0_6b();
    parsed.vocab_size = 999;
    cfg.merge_structural_from(&parsed);

    assert_eq!(cfg.vocab_size, 999);
    assert_eq!(cfg.dtype, DType::Float32);
}
