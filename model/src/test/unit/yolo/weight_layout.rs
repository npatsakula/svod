use svod_dtype::DType;
use svod_ir::Op;
use svod_tensor::Tensor;
use svod_tensor::nn::StateDict;

use crate::yolo::loader::load_weights;

fn ramp(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i * 7919 % 97) as f32 / 97.0 - 0.5).collect()
}

/// A loaded conv weight keeps its `[cout, cin, kh, kw]` shape and values but is
/// stored channels-innermost: what the model holds is a permute of the realized
/// copy. Vectors and the norm parameters load as they are.
#[test]
fn conv_weights_load_channels_innermost() {
    let (cout, cin, k) = (4usize, 3usize, 3usize);
    let weight = Tensor::from_slice(ramp(cout * cin * k * k))
        .try_reshape([cout as isize, cin as isize, k as isize, k as isize])
        .unwrap();
    let mut sd = StateDict::new();
    sd.insert("head.weight".into(), weight.clone());
    sd.insert("head.bias".into(), Tensor::from_slice(ramp(cout)));
    let loaded = load_weights(&sd, &DType::Float32).unwrap();

    let w = &loaded["head.weight"];
    assert_eq!((0..4).map(|i| w.dim_const(i).unwrap()).collect::<Vec<_>>(), [cout, cin, k, k]);
    assert!(matches!(w.uop().op(), Op::Permute(..)), "the view over the channels-innermost copy");
    assert_eq!(w.to_vec::<f32>().unwrap(), ramp(cout * cin * k * k), "the values are untouched");

    let b = &loaded["head.bias"];
    assert!(!matches!(b.uop().op(), Op::Permute(..)), "a vector is stored as it comes");
}
