use crate::bgem3::{BgeM3, EncodeOpts};
use crate::test::unit::xlm_roberta::model::tiny_cfg;

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
