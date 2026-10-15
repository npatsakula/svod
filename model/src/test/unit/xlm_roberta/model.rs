use svod_dtype::DType;
use svod_tensor::nn::{Module, StateDict};

use crate::xlm_roberta::{XlmRobertaConfig, XlmRobertaModel};

/// A 2-layer, 32-wide backbone with 20 position rows.
pub fn tiny_cfg() -> XlmRobertaConfig {
    XlmRobertaConfig {
        vocab_size: 100,
        hidden_size: 32,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        intermediate_size: 64,
        max_position_embeddings: 20,
        type_vocab_size: 1,
        layer_norm_eps: 1e-5,
        pad_token_id: 1,
        num_labels: 2,
        id2label: Vec::new(),
        dtype: DType::Float32,
        max_batch_size: 1,
    }
}

#[test]
fn forward_output_shape() {
    let model = XlmRobertaModel::empty(tiny_cfg());
    let ids = svod_tensor::Tensor::from_slice([0i64, 10, 20, 2, 1]).try_reshape([1isize, 5]).unwrap();
    let out = model.forward(&ids, None).unwrap();
    out.realize().unwrap();
    let v = out.as_vec::<f32>().unwrap();
    assert_eq!(v.len(), 5 * 32);
    assert!(v.iter().all(|x| x.is_finite()));
}

/// Every key `#[derive(Module)]` emits for a 2-layer tiny backbone, in the
/// published `BAAI/bge-m3` naming. Drift here is a checkpoint-compatibility
/// break. Norm biases are absent from a freshly-built model and loaded only
/// when the checkpoint carries them.
pub fn expected_keys() -> Vec<String> {
    let mut keys: Vec<String> = ["word_embeddings", "position_embeddings", "token_type_embeddings"]
        .iter()
        .map(|t| format!("embeddings.{t}.weight"))
        .collect();
    keys.push("embeddings.LayerNorm.weight".to_string());
    for i in 0..2 {
        for k in [
            "attention.self.query.weight",
            "attention.self.query.bias",
            "attention.self.key.weight",
            "attention.self.key.bias",
            "attention.self.value.weight",
            "attention.self.value.bias",
            "attention.output.dense.weight",
            "attention.output.dense.bias",
            "attention.output.LayerNorm.weight",
            "intermediate.dense.weight",
            "intermediate.dense.bias",
            "output.dense.weight",
            "output.dense.bias",
            "output.LayerNorm.weight",
        ] {
            keys.push(format!("encoder.layer.{i}.{k}"));
        }
    }
    keys.sort();
    keys
}

pub fn sorted_keys(sd: &StateDict) -> Vec<String> {
    let mut keys: Vec<String> = sd.keys().cloned().collect();
    keys.sort();
    keys
}

/// The emitted key set is exactly the published layout, at the root and nested
/// under a prefix (which must not grow a leading dot).
#[test]
fn state_dict_keys_match_published_layout() {
    let model = XlmRobertaModel::empty(tiny_cfg());
    let want = expected_keys();
    assert_eq!(sorted_keys(&model.state_dict("")), want);

    let want_nested: Vec<String> = want.iter().map(|k| format!("m.{k}")).collect();
    assert_eq!(sorted_keys(&model.state_dict("m")), want_nested);
}

#[test]
fn state_dict_round_trip() {
    let model = XlmRobertaModel::empty(tiny_cfg());
    let sd = model.state_dict("");
    let mut model2 = XlmRobertaModel::empty(tiny_cfg());
    model2.load_state_dict(&sd, "").unwrap();
    assert_eq!(sorted_keys(&sd), sorted_keys(&model2.state_dict("")));
}
