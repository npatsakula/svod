use crate::xlm_roberta::{XlmRobertaConfig, xlm_roberta_large};
use svod_dtype::DType;

#[test]
fn large_dims() {
    let cfg = xlm_roberta_large();
    assert_eq!(cfg.vocab_size, 250002);
    assert_eq!(cfg.hidden_size, 1024);
    assert_eq!(cfg.num_hidden_layers, 24);
    assert_eq!(cfg.num_attention_heads, 16);
    assert_eq!(cfg.head_dim(), 64);
    assert_eq!(cfg.intermediate_size, 4096);
    assert_eq!(cfg.max_position_embeddings, 8194);
    assert_eq!(cfg.type_vocab_size, 1);
    assert_eq!(cfg.pad_token_id, 1);
    assert_eq!(cfg.layer_norm_eps, 1e-5);
}

#[test]
fn parse_config_json() {
    let json = r#"{
        "vocab_size": 250002, "hidden_size": 1024, "num_hidden_layers": 24,
        "num_attention_heads": 16, "intermediate_size": 4096,
        "max_position_embeddings": 8194, "type_vocab_size": 1,
        "layer_norm_eps": 1e-05, "pad_token_id": 1
    }"#;
    let cfg = XlmRobertaConfig::from_json_str(json).unwrap();
    assert_eq!(cfg.vocab_size, 250002);
    assert_eq!(cfg.pad_token_id, 1);
}

#[test]
fn parse_norm_eps_alias() {
    let cfg = XlmRobertaConfig::from_json_str(r#"{"norm_eps": 1e-5}"#).unwrap();
    assert!((cfg.layer_norm_eps - 1e-5).abs() < 1e-12);
}

#[test]
fn merge_preserves_dtype_and_takes_labels() {
    let mut caller = XlmRobertaConfig { dtype: DType::Float32, max_batch_size: 4, ..xlm_roberta_large() };
    caller.merge_structural_from(&XlmRobertaConfig::from_json_str(r#"{"id2label": {"0": "LABEL_0"}}"#).unwrap());
    assert_eq!(caller.dtype, DType::Float32);
    assert_eq!(caller.max_batch_size, 4);
    assert_eq!((caller.num_labels, caller.id2label), (1, vec!["LABEL_0".to_string()]));
}

/// `num_labels` as `config.json` sets it, else `len(id2label)`, else HF's 2.
#[test_case::test_case(r#"{"num_labels": 3, "id2label": {"0": "a"}}"#=> (3, 1); "explicit")]
#[test_case::test_case(r#"{"id2label": {"0": "LABEL_0"}}"# => (1, 1); "from id2label")]
#[test_case::test_case("{}" => (2, 0); "default")]
fn num_labels_resolution(json: &str) -> (usize, usize) {
    let cfg = XlmRobertaConfig::from_json_str(json).unwrap();
    (cfg.num_labels, cfg.id2label.len())
}

/// Positions start at `pad_token_id + 1`, so the table numbers that many fewer
/// tokens than it has rows.
#[test_case::test_case(514, 1 => 512; "xlm-roberta-base")]
#[test_case::test_case(8194, 1 => 8192; "bge-m3")]
fn max_sequence_length(max_position_embeddings: usize, pad_token_id: usize) -> usize {
    XlmRobertaConfig { max_position_embeddings, pad_token_id, ..xlm_roberta_large() }.max_sequence_length()
}

/// A SaT checkpoint's limited lookahead bands the attention, which the
/// backbone does not: such a config is refused, a null one is plain XLM-R.
#[test_case::test_case(r#"{"lookahead": 48}"# => false; "lookahead")]
#[test_case::test_case(r#"{"lookahead_split_layers": 2}"# => false; "causal split")]
#[test_case::test_case(r#"{"lookahead": null, "lookahead_split_layers": null}"# => true; "global")]
fn lookahead_is_refused(json: &str) -> bool {
    XlmRobertaConfig::from_json_str(json).is_ok()
}
