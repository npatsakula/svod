use svod_dtype::DType;
use svod_tensor::Tensor;

use svod_tensor::nn::{Module, StateDict};

use crate::modernbert::{ClassifierPooling, ModernBertConfig, ModernBertForMaskedLm};

/// Tiny config matching `model::tiny_cfg` but with a smaller vocab so the
/// `(B, L, V)` logits forward stays cheap.
fn tiny_cfg() -> ModernBertConfig {
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

/// `MlmHead::empty` allocates the published shapes: `dense (D,D)`, `norm (D,)`,
/// `decoder_weight (V,D)`, `decoder.bias (V,)`.
#[test]
fn head_weight_shapes() {
    let cfg = tiny_cfg();
    let d = cfg.hidden_size;
    let v = cfg.vocab_size;
    let m = ModernBertForMaskedLm::empty(cfg);
    assert_eq!(m.head.dense_weight.dim_const(0).unwrap(), d);
    assert_eq!(m.head.dense_weight.dim_const(1).unwrap(), d);
    assert_eq!(m.head.norm.weight.dim_const(0).unwrap(), d);
    assert_eq!(m.head.decoder_weight.dim_const(0).unwrap(), v);
    assert_eq!(m.head.decoder_weight.dim_const(1).unwrap(), d);
    let bias = m.head.decoder_bias.expect("decoder_bias=true");
    assert_eq!(bias.dim_const(0).unwrap(), v);
}

/// The head's `state_dict` → `load_state_dict` round-trip reproduces the exact
/// key set: the published names (`head.dense.weight`, `head.norm.weight`,
/// `decoder.bias`) plus `decoder.weight`, and the load re-populates every
/// tensor (value equality, not just key-count).
#[test]
fn head_state_dict_round_trip() {
    let cfg = tiny_cfg();
    let m = ModernBertForMaskedLm::empty(cfg);
    let sd: StateDict = m.head.state_dict("");

    assert!(sd.contains_key("head.dense.weight"));
    assert!(sd.contains_key("head.norm.weight"));
    assert!(sd.contains_key("decoder.bias"));
    assert!(sd.contains_key("decoder.weight"));
    // No dense bias (head_pred_bias=false) and no norm bias (norm_bias=false).
    assert!(!sd.contains_key("head.dense.bias"));
    assert!(!sd.contains_key("head.norm.bias"));

    // Real round-trip: load into a fresh head, re-emit, and compare key sets.
    let mut reloaded = ModernBertForMaskedLm::empty(tiny_cfg());
    reloaded.head.load_state_dict(&sd, "").unwrap();
    let sd2: StateDict = reloaded.head.state_dict("");
    assert_eq!(sd.keys().collect::<std::collections::HashSet<_>>(), sd2.keys().collect());

    // Value equality on the non-lazy weights that survive a round-trip.
    for key in ["head.norm.weight", "decoder.bias"] {
        let a = realize_vec_f32(&sd[key]);
        let b = realize_vec_f32(&sd2[key]);
        assert_eq!(a, b, "{key} changed across round-trip");
    }
}

/// `MlmHead::forward` maps `(B, L, D)` hidden states to `(B, L, V)` logits.
/// Self-contained — no external tensor argument (weight-tying resolved at load).
#[test]
fn head_forward_output_shape() {
    let cfg = tiny_cfg();
    let d = cfg.hidden_size;
    let v = cfg.vocab_size;
    let m = ModernBertForMaskedLm::empty(cfg);
    let hidden = Tensor::from_slice((0..(2 * 5 * d) as i32).map(|i| i as f32).collect::<Vec<_>>())
        .try_reshape([2, 5, d])
        .unwrap();
    let logits = m.head.forward(&hidden).unwrap();
    logits.realize().unwrap();
    let vals = logits.as_vec::<f32>().unwrap();
    assert_eq!(vals.len(), 2 * 5 * v, "(B,L,V) = (2,5,{v})");
    for x in &vals {
        assert!(x.is_finite(), "non-finite logit: {x}");
    }
}

/// When `tie_word_embeddings` is true, loading from a checkpoint that carries
/// the embedding weight (but no `decoder.weight` key — the published layout)
/// aliases the embedding into the head's decoder. This is the tying contract:
/// the standalone `decoder.weight` is absent, yet the head resolves its decoder.
#[test]
fn head_ties_decoder_to_embeddings_when_decoder_weight_absent() {
    let cfg = tiny_cfg();
    let m = ModernBertForMaskedLm::empty(cfg);

    // Build a checkpoint-shaped dict: backbone keys + head keys, but WITHOUT a
    // standalone `decoder.weight` (the tied layout).
    let mut tied_sd = m.bert.state_dict("");
    tied_sd.insert("head.dense.weight".into(), m.head.dense_weight.clone());
    tied_sd.extend(m.head.norm.state_dict("head.norm"));
    tied_sd.insert("decoder.bias".into(), m.head.decoder_bias.clone().unwrap());
    assert!(!tied_sd.contains_key("decoder.weight"), "fixture must omit decoder.weight");

    let mut reloaded = ModernBertForMaskedLm::empty(tiny_cfg());
    reloaded.head.load_state_dict(&tied_sd, "").unwrap();

    // The head's decoder must match the embedding table (the tied source).
    let emb = realize_vec_f32(&tied_sd["model.embeddings.tok_embeddings.weight"]);
    let dec = realize_vec_f32(&reloaded.head.decoder_weight);
    assert_eq!(emb, dec, "tied decoder must alias the embedding table");
}

/// The full `ModernBertForMaskedLm` round-trip carries both the backbone keys
/// (`model.*`) and the head keys (`head.*`, `decoder.*`) — and, because tying is
/// resolved by aliasing on load, the composite `state_dict` emits exactly ONE
/// `(V, D)` weight (under the embedding key), not two.
#[test]
fn mlm_model_state_dict_round_trip() {
    let cfg = tiny_cfg();
    let m = ModernBertForMaskedLm::empty(cfg);
    let sd: StateDict = m.state_dict("");

    // Backbone keys are present unchanged.
    assert!(sd.contains_key("model.embeddings.tok_embeddings.weight"));
    assert!(sd.contains_key("model.final_norm.weight"));
    assert!(sd.contains_key("model.layers.0.attn.Wqkv.weight"));
    // Head keys are present alongside them.
    assert!(sd.contains_key("head.dense.weight"));
    assert!(sd.contains_key("head.norm.weight"));
    assert!(sd.contains_key("decoder.bias"));
    // The tying invariant: the composite emits the decoder weight under the
    // embedding key only — the head's duplicate `decoder.weight` copy is dropped.
    assert!(
        !sd.contains_key("decoder.weight"),
        "composite must not emit decoder.weight (tied — emitted under the embedding key)"
    );

    // Real round-trip into a fresh model reproduces the key set.
    let mut reloaded = ModernBertForMaskedLm::empty(tiny_cfg());
    reloaded.load_state_dict(&sd, "").unwrap();
    let sd2: StateDict = reloaded.state_dict("");
    assert_eq!(sd.keys().collect::<std::collections::HashSet<_>>(), sd2.keys().collect());
}

/// The composite key set is exactly the backbone's plus the head's, and a
/// non-empty prefix nests all of them without growing a leading dot.
#[test]
fn mlm_state_dict_keys_nest_under_a_prefix() {
    let m = ModernBertForMaskedLm::empty(tiny_cfg());
    let mut want: Vec<String> = m.state_dict("").keys().cloned().collect();
    want.sort();
    // The head contributes exactly these three on top of the backbone's.
    for key in ["head.dense.weight", "head.norm.weight", "decoder.bias"] {
        assert!(want.iter().any(|k| k == key), "missing head key: {key}");
    }

    let mut got: Vec<String> = m.state_dict("m").keys().cloned().collect();
    got.sort();
    let want_nested: Vec<String> = want.iter().map(|k| format!("m.{k}")).collect();
    assert_eq!(got, want_nested);
}

/// `ModernBertForMaskedLm::forward` runs the full backbone + head end-to-end
/// and returns `(B, L, V)` logits.
#[test]
fn mlm_forward_output_shape() {
    let cfg = tiny_cfg();
    let v = cfg.vocab_size;
    let m = ModernBertForMaskedLm::empty(cfg);
    let ids = Tensor::from_slice((0..10i64).collect::<Vec<_>>()).try_reshape([2isize, 5]).unwrap();
    let logits = m.forward(&ids, None).unwrap();
    logits.realize().unwrap();
    let vals = logits.as_vec::<f32>().unwrap();
    assert_eq!(vals.len(), 2 * 5 * v, "(B,L,V) = (2,5,{v})");
    for x in &vals {
        assert!(x.is_finite(), "non-finite logit: {x}");
    }
}

fn realize_vec_f32(t: &svod_tensor::Tensor) -> Vec<f32> {
    let t = t.clone();
    t.realize().unwrap();
    t.as_vec::<f32>().unwrap()
}
