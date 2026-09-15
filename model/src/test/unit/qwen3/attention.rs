use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::Module;

use crate::qwen3::{Qwen3Attention, qwen3_embedding_0_6b};

fn tiny_attn() -> Qwen3Attention {
    Qwen3Attention::empty(64, 4, 2, 32, 1e-5, DType::Float32)
}

#[test]
fn gqa_projection_shapes() {
    let attn = tiny_attn();
    let sd = attn.state_dict("");
    let dims = |key: &str| sd[key].dims().unwrap();

    // q: (4*32, 64) = (128, 64), k/v: (2*32, 64) = (64, 64), o: (64, 128)
    assert_eq!(dims("q_proj.weight"), [128, 64]);
    assert_eq!(dims("k_proj.weight"), [64, 64]);
    assert_eq!(dims("v_proj.weight"), [64, 64]);
    assert_eq!(dims("o_proj.weight"), [64, 128]);
    assert_eq!(attn.qkv_weight.dims().unwrap(), [256, 64]);
}

#[test]
fn qk_norm_dims() {
    let attn = tiny_attn();
    let qn = attn.q_norm.weight.dims().unwrap();
    let kn = attn.k_norm.weight.dims().unwrap();
    assert_eq!(qn[0], 32);
    assert_eq!(kn[0], 32);
}

#[test]
fn forward_output_shape() {
    let attn = tiny_attn();

    let x = Tensor::from_slice([0.5f32; 512]).try_reshape([1isize, 8, 64]).unwrap();
    let (cos, sin) = Tensor::rope_table(10000.0, 8, 32, DType::Float32).unwrap();
    let rope = (cos.try_transpose(1, 2).unwrap(), sin.try_transpose(1, 2).unwrap());

    let out = attn.forward(&x, &rope).unwrap();
    out.realize().unwrap();
    let s = out.dims().unwrap();
    assert_eq!(s[0], 1);
    assert_eq!(s[1], 8);
    assert_eq!(s[2], 64);

    let v = out.as_vec::<f32>().unwrap();
    assert!(v.iter().all(|x| x.is_finite()));
}

#[test]
fn published_weight_shapes() {
    let cfg = qwen3_embedding_0_6b();
    let attn = Qwen3Attention::empty(
        cfg.hidden_size,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
        cfg.rms_norm_eps,
        DType::BFloat16,
    );
    let sd = attn.state_dict("");
    let dims = |key: &str| sd[key].dims().unwrap();
    assert_eq!(dims("q_proj.weight"), [16 * 128, 1024]);
    assert_eq!(dims("k_proj.weight"), [8 * 128, 1024]);
    assert_eq!(dims("v_proj.weight"), [8 * 128, 1024]);
    assert_eq!(dims("o_proj.weight"), [1024, 16 * 128]);
}
