use crate::modernbert::ModernBertConfig;

fn test_cfg() -> ModernBertConfig {
    ModernBertConfig {
        hidden_size: 768,
        num_attention_heads: 12,
        local_attention: 128,
        global_attn_every_n_layers: 3,
        global_rope_theta: 160_000.0,
        local_rope_theta: 10_000.0,
        ..ModernBertConfig::default()
    }
}

#[test]
fn local_window_split_evenly() {
    let c = test_cfg();
    assert_eq!(c.local_window(), (64, 64));
}

#[test]
fn global_layer_pattern() {
    let c = test_cfg();
    assert!(c.is_global_layer(0));
    assert!(c.is_global_layer(3));
    assert!(c.is_global_layer(21));
    assert!(!c.is_global_layer(1));
    assert!(!c.is_global_layer(2));
    assert!(!c.is_global_layer(20));
}

#[test]
fn rope_theta_per_layer() {
    let c = test_cfg();
    assert_eq!(c.rope_theta(0), 160_000.0); // global
    assert_eq!(c.rope_theta(3), 160_000.0); // global
    assert_eq!(c.rope_theta(1), 10_000.0); // local
    assert_eq!(c.rope_theta(2), 10_000.0); // local
    assert_eq!(c.rope_theta(20), 10_000.0); // local (20 % 3 == 2)
}

/// Round-trip a synthetic `config.json` through `from_json_str` and confirm
/// the parsed values land on the right fields.
#[test]
fn parse_config_json() {
    let json = r#"{
        "model_type": "modernbert",
        "hidden_size": 1024,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "intermediate_size": 2624,
        "vocab_size": 50368,
        "global_rope_theta": 160000.0,
        "local_rope_theta": 10000.0,
        "local_attention": 128,
        "global_attn_every_n_layers": 3,
        "layer_norm_eps": 1e-5,
        "pad_token_id": 50283,
        "tie_word_embeddings": true,
        "decoder_bias": true
    }"#;
    let c = ModernBertConfig::from_json_str(json).expect("parse");
    assert_eq!(c.hidden_size, 1024);
    assert_eq!(c.num_hidden_layers, 28);
    assert_eq!(c.num_attention_heads, 16);
    assert_eq!(c.intermediate_size, 2624);
    assert_eq!(c.vocab_size, 50368);
    assert_eq!(c.local_attention, 128);
    assert_eq!(c.global_attn_every_n_layers, 3);
    assert_eq!(c.layer_norm_eps, 1e-5);
}

/// transformers ≥ 5 writes the rotary bases per layer type under
/// `rope_parameters`; a fine-tune saved by it has no `global_rope_theta` /
/// `local_rope_theta`, and a base left at the zero default turns every
/// rotation into NaN.
#[test]
fn parse_rope_parameters_per_layer_type() {
    let json = r#"{
        "rope_parameters": {
            "full_attention": { "rope_theta": 160000.0, "rope_type": "default" },
            "sliding_attention": { "rope_theta": 10000.0, "rope_type": "default" }
        }
    }"#;
    let c = ModernBertConfig::from_json_str(json).expect("parse");
    assert_eq!((c.global_rope_theta, c.local_rope_theta), (160_000.0, 10_000.0));
}

/// The published config.json publishes `norm_eps` (in addition to
/// `layer_norm_eps`); confirm either is accepted.
#[test]
fn parse_norm_eps_alias() {
    let json = r#"{ "norm_eps": 1e-5 }"#;
    let c = ModernBertConfig::from_json_str(json).expect("parse");
    assert!((c.layer_norm_eps - 1e-5).abs() < 1e-12);
}

/// `id2label` from `config.json` is parsed into a dense `Vec<String>`: sized to
/// `max(id)+1`, gaps filled with `"LABEL_{id}"`. `num_labels` follows the dense
/// length when `num_labels` is absent.
#[test]
fn parse_id2label_dense_with_gaps() {
    let json = r#"{
        "id2label": {"0": "O", "1": "B-PER", "2": "I-PER", "4": "B-LOC"}
    }"#;
    let c = ModernBertConfig::from_json_str(json).expect("parse");
    // max(id)+1 = 5; gap at index 3 filled with "LABEL_3".
    assert_eq!(c.id2label, vec!["O", "B-PER", "I-PER", "LABEL_3", "B-LOC"]);
    assert_eq!(c.num_labels, 5, "num_labels follows dense id2label length");
}

/// An absent `id2label` yields an empty vec and the base `num_labels` default.
#[test]
fn parse_absent_id2label_is_empty() {
    let c = ModernBertConfig::from_json_str("{}").expect("parse");
    assert!(c.id2label.is_empty());
    assert_eq!(c.num_labels, 2, "base default");
}
