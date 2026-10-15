use svod_tensor::nn::Module;

use crate::bgem3::{BgeM3, BgeRerankerV2M3, EncodeOpts};
use crate::test::unit::xlm_roberta::model::{expected_keys, sorted_keys, tiny_cfg};

#[test]
fn bgem3_encode_dense_shape() {
    let model = BgeM3::empty(tiny_cfg());
    let ids = svod_tensor::Tensor::from_slice([0i64, 10, 20, 2]).try_reshape([1isize, 4]).unwrap();
    let mask = svod_tensor::Tensor::from_slice([1i64, 1, 1, 1]).try_reshape([1isize, 4]).unwrap();
    let dense = model.encode_dense(&ids, &mask).unwrap();
    dense.realize().unwrap();
    let s = dense.dims().unwrap();
    assert_eq!(s[0], 1);
    assert_eq!(s[1], 32);
}

#[test]
fn bgem3_encode_all_modalities() {
    let model = BgeM3::empty(tiny_cfg());
    let ids = svod_tensor::Tensor::from_slice([0i64, 10, 20, 2]).try_reshape([1isize, 4]).unwrap();
    let mask = svod_tensor::Tensor::from_slice([1i64, 1, 1, 1]).try_reshape([1isize, 4]).unwrap();
    let out = model.encode(&ids, &mask, EncodeOpts::all()).unwrap();
    assert!(out.dense_vecs.is_some());
    assert!(out.sparse_vecs.is_some());
    assert!(out.colbert_vecs.is_some());
}

#[test]
fn reranker_forward_shape() {
    let model = BgeRerankerV2M3::empty(tiny_cfg());
    let ids = svod_tensor::Tensor::from_slice([0i64, 10, 20, 2, 1]).try_reshape([1isize, 5]).unwrap();
    let mask = svod_tensor::Tensor::from_slice([1i64, 1, 1, 1, 0]).try_reshape([1isize, 5]).unwrap();
    let out = model.forward(&ids, Some(&mask)).unwrap();
    out.realize().unwrap();
    let s = out.dims().unwrap();
    assert_eq!(s[0], 1);
    assert_eq!(s[1], 1);
}

/// The reranker emits the published `BAAI/bge-reranker-v2-m3` layout: the
/// backbone under `roberta.`, the classification head under `classifier.`.
#[test]
fn reranker_state_dict_keys_match_published_layout() {
    let model = BgeRerankerV2M3::empty(tiny_cfg());
    let mut want: Vec<String> = expected_keys().iter().map(|k| format!("roberta.{k}")).collect();
    want.extend(
        ["dense.weight", "dense.bias", "out_proj.weight", "out_proj.bias"].iter().map(|k| format!("classifier.{k}")),
    );
    want.sort();
    assert_eq!(sorted_keys(&model.state_dict("")), want);

    let want_nested: Vec<String> = want.iter().map(|k| format!("m.{k}")).collect();
    assert_eq!(sorted_keys(&model.state_dict("m")), want_nested);
}

#[test]
fn reranker_state_dict_round_trip() {
    let model = BgeRerankerV2M3::empty(tiny_cfg());
    let sd = model.state_dict("");
    let mut model2 = BgeRerankerV2M3::empty(tiny_cfg());
    model2.load_state_dict(&sd, "").unwrap();
    assert_eq!(sorted_keys(&sd), sorted_keys(&model2.state_dict("")));
}
