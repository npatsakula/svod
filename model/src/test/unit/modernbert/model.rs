use svod_dtype::DType;
use svod_tensor::Tensor;

use svod_tensor::nn::{Module, StateDict};

use crate::modernbert::{ClassifierPooling, ModernBert, ModernBertConfig};

/// A tiny config for fast unit tests: 2 layers, hidden 32, 4 heads (head_dim 8),
/// vocab 64, intermediate 64, f32 compute. Global-every-3 still gives layer 0
/// global and layer 1 local.
pub(crate) fn tiny_cfg() -> ModernBertConfig {
    ModernBertConfig {
        vocab_size: 64,
        hidden_size: 32,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        intermediate_size: 64,
        max_position_embeddings: 128,
        layer_norm_eps: 1e-5,
        global_rope_theta: 10_000.0,
        local_rope_theta: 10_000.0,
        local_attention: 16,
        global_attn_every_n_layers: 3,
        pad_token_id: 0,
        tie_word_embeddings: true,
        decoder_bias: true,
        dtype: DType::Float32,
        max_batch_size: 1,
        num_labels: 3,
        classifier_pooling: ClassifierPooling::Cls,
        classifier_bias: false,
        norm_bias: false,
        id2label: vec![],
    }
}

/// `empty` builds the right structure: layer 0 has no attn_norm, layer 1 does.
#[test]
fn layer_zero_has_no_attn_norm() {
    let cfg = tiny_cfg();
    let m = ModernBert::empty(cfg);
    assert!(m.encoder.layers[0].attn_norm.is_none(), "layer 0 attn_norm must be None");
    assert!(m.encoder.layers[1].attn_norm.is_some(), "layer 1+ attn_norm must be Some");
}

/// `forward` produces a `(B, L, D)` tensor of the expected shape.
#[test]
fn forward_output_shape() {
    let cfg = tiny_cfg();
    let m = ModernBert::empty(cfg);
    // input_ids (B=2, L=5) int64.
    let ids = Tensor::from_slice((0..10i64).collect::<Vec<_>>()).try_reshape([2isize, 5]).unwrap();
    let out = m.forward(&ids, None).unwrap();
    out.realize().unwrap();
    let v = out.as_vec::<f32>().unwrap();
    // (2, 5, 32) = 320 elements, all finite.
    assert_eq!(v.len(), 2 * 5 * 32);
    for val in &v {
        assert!(val.is_finite(), "non-finite forward output: {val}");
    }
}

/// Every key `#[derive(Module)]` emits for a 2-layer tiny backbone, in the
/// published checkpoint's naming. Drift here is a checkpoint-compatibility
/// break.
fn expected_keys() -> Vec<String> {
    let mut keys = vec![
        "model.embeddings.tok_embeddings.weight".to_string(),
        "model.embeddings.norm.weight".to_string(),
        "model.final_norm.weight".to_string(),
        // Layer 0 is `skip_first_prenorm`: it emits no `attn_norm`.
        "model.layers.1.attn_norm.weight".to_string(),
    ];
    for i in 0..2 {
        for k in ["attn.Wqkv.weight", "attn.Wo.weight", "mlp_norm.weight", "mlp.Wi.weight", "mlp.Wo.weight"] {
            keys.push(format!("model.layers.{i}.{k}"));
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
    let m = ModernBert::empty(tiny_cfg());
    let want = expected_keys();
    assert_eq!(sorted_keys(&m.state_dict("")), want);

    let want_nested: Vec<String> = want.iter().map(|k| format!("m.{k}")).collect();
    assert_eq!(sorted_keys(&m.state_dict("m")), want_nested);
}

/// `state_dict` → `load_state_dict` round-trips: building from a model's own
/// state dict reproduces it (keys + shape). The key map matches the published
/// checkpoint layout.
#[test]
fn state_dict_round_trip() {
    let cfg = tiny_cfg();
    let m = ModernBert::empty(cfg);
    let sd: StateDict = m.state_dict("");

    // Spot-check the published key names are present.
    assert!(sd.contains_key("model.embeddings.tok_embeddings.weight"));
    assert!(sd.contains_key("model.embeddings.norm.weight"));
    assert!(sd.contains_key("model.final_norm.weight"));
    // Layer 0 has no attn_norm key; layer 1 does.
    assert!(!sd.contains_key("model.layers.0.attn_norm.weight"), "layer 0 must not emit attn_norm");
    assert!(sd.contains_key("model.layers.1.attn_norm.weight"));
    assert!(sd.contains_key("model.layers.0.attn.Wqkv.weight"));
    assert!(sd.contains_key("model.layers.0.attn.Wo.weight"));
    assert!(sd.contains_key("model.layers.0.mlp.Wi.weight"));
    assert!(sd.contains_key("model.layers.0.mlp.Wo.weight"));
    assert!(sd.contains_key("model.layers.0.mlp_norm.weight"));
    // No biases on any linear/norm (ModernBERT is bias-free).
    assert!(!sd.contains_key("model.layers.0.attn.Wqkv.bias"));
    assert!(!sd.contains_key("model.embeddings.norm.bias"));

    // Round-trip into a fresh model.
    let mut m2 = ModernBert::empty(tiny_cfg());
    m2.load_state_dict(&sd, "").expect("reload");
    assert_eq!(sorted_keys(&sd), sorted_keys(&m2.state_dict("")));
}

/// The fused QKV weight is `(3*D, D)`; the GLU `Wi` is `(2*I, D)`.
#[test]
fn published_weight_shapes() {
    let cfg = tiny_cfg();
    let d = cfg.hidden_size;
    let i = cfg.intermediate_size;
    let m = ModernBert::empty(cfg);
    let sd: StateDict = m.state_dict("");
    let qkv = &sd["model.layers.0.attn.Wqkv.weight"];
    let wi = &sd["model.layers.0.mlp.Wi.weight"];
    assert_eq!(qkv.dim_const(0).unwrap(), 3 * d);
    assert_eq!(qkv.dim_const(1).unwrap(), d);
    assert_eq!(wi.dim_const(0).unwrap(), 2 * i);
    assert_eq!(wi.dim_const(1).unwrap(), d);
}
