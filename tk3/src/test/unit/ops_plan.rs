//! The op layer's decisions on the host: which plan each shape, dtype and
//! target gets, and which requests are errors rather than fallbacks.

use svod_dtype::{AmdArch, DType, DeviceSpec, GpuArch};
use svod_ir::SInt;
use svod_tensor::{Tensor, Variable};
use test_case::test_case;

use crate::atoms::{Target, sm86};
use crate::kernels::attention::FaCfg;
use crate::kernels::gemm::GemmCfg;
use crate::kernels::rows::NormCfg;
use crate::ops::shape::{self, Extent, Fallback, Plan, extent};
use crate::ops::{self as tk, Attn, Error, KeyMask, Linear};

const BF16: DType = DType::BFloat16;

fn ext(dims: &[usize]) -> Extent {
    Extent { dims: dims.to_vec(), var: false }
}

fn gemm_cfg(tile: [usize; 3], stages: usize, warps: [u32; 2]) -> GemmCfg {
    GemmCfg { tile, stages, warps, group_m: 8, unroll: false }
}

fn rdna3() -> Target {
    Target::for_arch(GpuArch::Amd(AmdArch::Gfx1100))
}

// ---- linear ------------------------------------------------------------------------

#[test_case(&[4096, 4096], 4096, false, Plan::Kernel(gemm_cfg([128, 128, 32], 3, [2, 4])); "large grid, deepest ring")]
#[test_case(&[4096, 4096], 4096, true, Plan::Kernel(gemm_cfg([128, 128, 32], 2, [2, 4])); "gated drops a stage for smem")]
#[test_case(&[8, 37, 512], 512, false, Plan::Kernel(GemmCfg { unroll: true, ..gemm_cfg([128, 64, 32], 2, [2, 2]) }); "medium grid")]
#[test_case(&[37, 64], 96, false, Plan::Kernel(gemm_cfg([64, 64, 32], 2, [2, 2])); "few rows")]
#[test_case(&[37, 48], 96, false, Plan::Kernel(gemm_cfg([64, 64, 16], 2, [2, 2])); "k a multiple of 16 only")]
#[test_case(&[37, 40], 96, false, Plan::Graph(Fallback::Config); "k off every bk")]
#[test_case(&[37, 64], 100, false, Plan::Graph(Fallback::Shape); "n not a multiple of 8")]
#[test_case(&[0, 64], 96, false, Plan::Graph(Fallback::Shape); "no rows")]
fn linear_plans(x: &[usize], n: usize, gated: bool, want: Plan<GemmCfg>) {
    assert_eq!(shape::linear(Some(&sm86()), &[BF16, BF16], Some(&ext(x)), n, gated), want);
}

#[test_case(None, &[BF16, BF16], true, Fallback::Target; "no target")]
#[test_case(Some(rdna3()), &[BF16, BF16], true, Fallback::Target; "no tables for the arch")]
#[test_case(Some(sm86()), &[DType::Float32, DType::Float32], true, Fallback::Dtype; "f32 keeps the graph")]
#[test_case(Some(sm86()), &[BF16, DType::Float32], true, Fallback::Dtype; "mixed operand types")]
#[test_case(Some(sm86()), &[DType::Int32, DType::Int32], true, Fallback::Dtype; "integers")]
#[test_case(Some(sm86()), &[BF16, BF16], false, Fallback::Symbolic; "symbolic past the batch")]
fn every_op_falls_back_alike(target: Option<Target>, dtypes: &[DType], static_dims: bool, why: Fallback) {
    let x = static_dims.then(|| ext(&[2, 128, 1024]));
    let kv = static_dims.then(|| ext(&[2, 128, 16, 64]));
    let q = static_dims.then(|| ext(&[2, 128, 16, 64]));
    let target = target.as_ref();
    assert_eq!(shape::linear(target, dtypes, x.as_ref(), 1024, false), Plan::Graph(why));
    assert_eq!(shape::attention(target, dtypes, q.as_ref(), kv.as_ref()), Plan::Graph(why));
    assert_eq!(shape::norm(target, dtypes, x.as_ref()), Plan::Graph(why));
}

/// A bound variable that is the reduced dim itself is no batch.
#[test]
fn a_variable_reduced_dim_falls_back() {
    let x = Extent { dims: vec![64], var: true };
    assert_eq!(shape::linear(Some(&sm86()), &[BF16; 2], Some(&x), 64, false), Plan::Graph(Fallback::Symbolic));
    let x = Extent { dims: vec![1024], var: true };
    assert_eq!(shape::norm(Some(&sm86()), &[BF16; 2], Some(&x)), Plan::Graph(Fallback::Symbolic));
}

#[test]
fn f16_takes_the_kernels() {
    let f16 = [DType::Float16, DType::Float16];
    assert!(matches!(shape::linear(Some(&sm86()), &f16, Some(&ext(&[64, 64])), 64, false), Plan::Kernel(_)));
}

// ---- attention ---------------------------------------------------------------------

#[test_case(64, Plan::Kernel(FaCfg { bq: 64, bkv: 64, stages: 2 }); "d 64")]
#[test_case(128, Plan::Kernel(FaCfg { bq: 64, bkv: 32, stages: 2 }); "d 128 narrows the key block")]
#[test_case(48, Plan::Graph(Fallback::Shape); "d 48")]
#[test_case(256, Plan::Graph(Fallback::Shape); "d 256")]
fn attention_plans(d: usize, want: Plan<FaCfg>) {
    let (q, kv) = (ext(&[2, 100, 8, d]), ext(&[2, 37, 2, d]));
    assert_eq!(shape::attention(Some(&sm86()), &[BF16; 3], Some(&q), Some(&kv)), want);
}

#[test]
fn attention_with_symbolic_keys_falls_back() {
    let q = ext(&[2, 100, 8, 64]);
    assert_eq!(shape::attention(Some(&sm86()), &[BF16; 3], Some(&q), None), Plan::Graph(Fallback::Symbolic));
}

// ---- norms -------------------------------------------------------------------------

#[test_case(1024, Plan::Kernel(NormCfg { br: 4 }); "1024")]
#[test_case(256, Plan::Kernel(NormCfg { br: 4 }); "256")]
#[test_case(768, Plan::Graph(Fallback::Shape); "not a power of two")]
#[test_case(128, Plan::Graph(Fallback::Shape); "narrower than a warp's loads")]
#[test_case(4096, Plan::Graph(Fallback::Shape); "wider than the measured range")]
fn norm_plans(d: usize, want: Plan<NormCfg>) {
    assert_eq!(shape::norm(Some(&sm86()), &[BF16; 2], Some(&ext(&[37, d]))), want);
}

// ---- extents -------------------------------------------------------------------------

/// Dims come back at capacity; only a bound leading variable is admitted.
#[test]
fn extents_take_a_bound_leading_variable_only() {
    let var = Variable::new("b", 1, 6);
    let bound = var.bind(2).unwrap();
    let t = Tensor::empty(&[6, 3, 5], BF16).try_shrink([Some((SInt::Const(0), bound.as_sint())), None, None]).unwrap();
    let (e, v) = extent(&t.shape().unwrap()).expect("a bound batch");
    assert_eq!(e, Extent { dims: vec![6, 3, 5], var: true });
    let v = v.unwrap();
    assert_eq!((v.name.as_str(), v.min, v.max, v.dim), ("b", 1, 6, bound.as_sint()));

    let (e, v) = extent(&Tensor::empty(&[6, 3], BF16).shape().unwrap()).unwrap();
    assert_eq!((e, v.is_none()), (Extent { dims: vec![6, 3], var: false }, true));

    let swapped = t.try_permute(&[1, 0, 2]).unwrap();
    assert!(extent(&swapped.shape().unwrap()).is_none(), "a symbolic non-leading dim");
}

#[test]
fn no_device_no_padding() {
    assert!(!tk::supported(&DeviceSpec::Cpu));
    assert_eq!(tk::preferred_len(&DeviceSpec::Cpu, &BF16, 100), 100);
}

// ---- errors ----------------------------------------------------------------------------

fn t(shape: &[usize], dtype: DType) -> Tensor {
    Tensor::empty(shape, dtype)
}

#[test]
fn semantic_mismatches_are_errors() {
    let x = t(&[4, 64], BF16);
    let err = |r: tk::Result<Tensor>| r.expect_err("an error");
    assert!(matches!(err(tk::linear(&x, &t(&[32, 48], BF16), Linear::default())), Error::Shape { operand: "w", .. }));
    assert!(matches!(
        err(tk::linear(&x, &t(&[32, 64], DType::Float16), Linear::default())),
        Error::Dtype { operand: "w", .. }
    ));
    let gated = Linear { gated: true, ..Linear::default() };
    assert!(matches!(err(tk::linear(&x, &t(&[33, 64], BF16), gated)), Error::Shape { operand: "w", .. }));
    let bias = t(&[32], BF16);
    let opts = Linear { bias: Some(&bias), gated: true, ..Linear::default() };
    assert!(matches!(err(tk::linear(&x, &t(&[64, 64], BF16), opts)), Error::Shape { operand: "bias", .. }));
    let residual = t(&[4, 64], BF16);
    let opts = Linear { residual: Some(&residual), ..Linear::default() };
    assert!(matches!(err(tk::linear(&x, &t(&[32, 64], BF16), opts)), Error::Shape { operand: "residual", .. }));

    let q = t(&[2, 10, 6, 64], BF16);
    let kv = t(&[2, 12, 4, 64], BF16);
    assert!(matches!(err(tk::attention(&q, &kv, &kv, Attn::default())), Error::Heads { heads: 6, kv_heads: 4, .. }));
    let kv2 = t(&[2, 12, 2, 64], BF16);
    let v = t(&[2, 11, 2, 64], BF16);
    assert!(matches!(err(tk::attention(&q, &kv2, &v, Attn::default())), Error::Shape { operand: "v", .. }));
    let short = t(&[2, 12, 2, 32], BF16);
    assert!(matches!(err(tk::attention(&q, &short, &short, Attn::default())), Error::Shape { operand: "k", .. }));
    let lens = t(&[2, 1], DType::Int32);
    let opts = Attn { keys: KeyMask::Lens(&lens), ..Attn::default() };
    assert!(matches!(err(tk::attention(&q, &kv2, &kv2, opts)), Error::Shape { operand: "key lens", .. }));

    let w = t(&[32], BF16);
    assert!(matches!(err(tk::layer_norm(&x, &w, None, 1e-5)), Error::Shape { operand: "w", .. }));
    let w = t(&[64], BF16);
    let r = t(&[4, 32], BF16);
    assert!(matches!(tk::add_rms_norm(&x, &r, &w, 1e-5).map(|_| ()), Err(Error::Shape { operand: "residual", .. })));
}
