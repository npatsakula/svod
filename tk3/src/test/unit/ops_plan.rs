//! The op layer's decisions on the host: which plan each shape, dtype and
//! target gets, and which requests are errors rather than fallbacks.

use svod_dtype::{AmdArch, DType, DeviceSpec, GpuArch};
use svod_ir::SInt;
use svod_tensor::{Tensor, Variable};
use test_case::test_case;

use crate::atoms::{Target, sm86};
use crate::build::BF16 as BF16T;
use crate::kernels::Batch;
use crate::kernels::attention::FaCfg;
use crate::kernels::gemm::{Epilogue, GemmCfg, GemmSpec, Scale, gemm};
use crate::kernels::rows::NormCfg;
use crate::ops::config;
use crate::ops::shape::{self, Extent, Fallback, Plan, extent};
use crate::ops::{self as tk, Act, Attn, Cache, Error, KeyMask, Linear, Qkv};

const BF16: DType = DType::BFloat16;

fn ext(dims: &[usize]) -> Extent {
    Extent { dims: dims.to_vec(), var: false }
}

fn gemm_cfg(tile: [usize; 3], stages: usize, warps: [u32; 2], unroll: bool) -> GemmCfg {
    GemmCfg { tile, stages, warps, group_m: 8, unroll }
}

/// The untuned pick of a plan: its first candidate.
fn first<C: Clone>(plan: Plan<C>) -> Result<C, Fallback> {
    match plan {
        Plan::Kernel(c) => Ok(c[0].clone()),
        Plan::Graph(why) => Err(why),
    }
}

fn rdna3() -> Target {
    Target::for_arch(GpuArch::Amd(AmdArch::Gfx1100))
}

// ---- linear ------------------------------------------------------------------------

const BIG: GemmCfg = GemmCfg { tile: [128, 128, 32], stages: 3, warps: [2, 4], group_m: 8, unroll: false };
const SMALL: GemmCfg = GemmCfg { tile: [64, 64, 32], stages: 3, warps: [2, 2], group_m: 8, unroll: true };

#[test_case(&[4096, 4096], 4096, false, Ok(BIG); "large grid, deepest ring")]
#[test_case(&[4096, 4096], 4096, true, Ok(BIG); "gated keeps the ring in 99 KB")]
#[test_case(&[8, 256, 512], 2048, false, Ok(BIG); "a static batch counts toward the grid")]
#[test_case(&[2048, 512], 1024, false, Ok(gemm_cfg([128, 64, 32], 2, [2, 2], true)); "half the big grid")]
#[test_case(&[8, 37, 512], 512, false, Ok(SMALL); "medium grid")]
#[test_case(&[37, 64], 96, false, Ok(SMALL); "few rows")]
#[test_case(&[37, 48], 96, false, Ok(gemm_cfg([64, 64, 16], 3, [2, 2], true)); "k a multiple of 16 only")]
#[test_case(&[37, 40], 96, false, Err(Fallback::Config); "k off every bk")]
#[test_case(&[37, 64], 100, false, Err(Fallback::Shape); "n not a multiple of 8")]
#[test_case(&[0, 64], 96, false, Err(Fallback::Shape); "no rows")]
fn linear_plans(x: &[usize], n: usize, gated: bool, want: Result<GemmCfg, Fallback>) {
    assert_eq!(first(shape::linear(Some(&sm86()), &[BF16, BF16], Some(&ext(x)), n, gated)), want);
}

fn gemm_list(x: &Extent, n: usize, gated: bool) -> Vec<GemmCfg> {
    match shape::linear(Some(&sm86()), &[BF16, BF16], Some(x), n, gated) {
        Plan::Kernel(list) => list,
        Plan::Graph(why) => panic!("graph: {why:?}"),
    }
}

/// The 4096³ list leads with the measured peak, then its pipeline variants,
/// then one entry per other tile family.
#[test]
fn large_gemm_candidates() {
    let want = [
        BIG,
        GemmCfg { stages: 2, ..BIG },
        GemmCfg { tile: [128, 128, 64], stages: 2, ..BIG },
        GemmCfg { unroll: true, ..BIG },
        GemmCfg { warps: [4, 2], ..BIG },
        gemm_cfg([128, 64, 32], 2, [2, 2], true),
        gemm_cfg([64, 128, 32], 2, [2, 2], false),
        SMALL,
    ];
    assert_eq!(gemm_list(&ext(&[4096, 4096]), 4096, false), want);
}

/// Nemotron's projections (704 rows under a batch variable of capacity 1):
/// 128-row tiles pad 64 rows and leave the 28 SMs with too few blocks, so
/// 64×64 and its variants come first.
#[test_case(512, 512, false; "qkv-sized out")]
#[test_case(1536, 512, false; "fused qkv")]
#[test_case(2048, 512, false; "ffn up")]
#[test_case(512, 2048, false; "ffn down")]
#[test_case(2048, 512, true; "gated up")]
fn small_m_gemm_candidates(n: usize, k: usize, gated: bool) {
    let x = Extent { dims: vec![1, 704, k], var: true };
    let list = gemm_list(&x, n, gated);
    assert!(list[..4].iter().all(|c| c.tile[..2] == [64, 64]), "{list:?}");
    assert_eq!(list[0], SMALL);
}

/// Every list is short, duplicate-free, within shared memory and divides `k`.
#[test_case(&[4096, 4096], 4096, false; "large")]
#[test_case(&[4096, 4096], 4096, true; "large gated")]
#[test_case(&[704, 2048], 512, false; "small m")]
#[test_case(&[300, 48], 200, false; "bk 16")]
#[test_case(&[1000, 96], 64, true; "bk 32 only, gated")]
fn gemm_candidates_are_valid(x: &[usize], n: usize, gated: bool) {
    let k = *x.last().unwrap();
    for smem in [sm86().smem_bytes, 48 << 10] {
        let target = Target { smem_bytes: smem, ..sm86() };
        let Plan::Kernel(list) = shape::linear(Some(&target), &[BF16; 2], Some(&ext(x)), n, gated) else {
            panic!("no kernel")
        };
        assert!((1..=8).contains(&list.len()), "{list:?}");
        for (i, c) in list.iter().enumerate() {
            assert!(k.is_multiple_of(c.tile[2]) && c.smem_bytes(gated) <= smem, "{c:?}");
            assert!(!list[..i].contains(c), "duplicate {c:?}");
        }
    }
}

/// A target capped at static shared memory drops a stage for the gated GEMM.
#[test]
fn gated_linear_drops_a_stage_under_a_static_cap() {
    let mut target = sm86();
    target.smem_bytes = 48 << 10;
    let want = Ok(GemmCfg { stages: 2, ..BIG });
    assert_eq!(first(shape::linear(Some(&target), &[BF16, BF16], Some(&ext(&[4096, 4096])), 4096, true)), want);
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

/// The tune store keys a GEMM by its epilogue's text: without a scale it
/// reads as it did before the field, so stored choices still apply; a scale
/// is its own key and its own program.
#[test]
fn an_unscaled_epilogue_keeps_its_tune_key() {
    let plain = Epilogue { bias: true, residual: true, ..Epilogue::default() };
    assert_eq!(format!("{plain:?}"), "Epilogue { bias: true, act: None, gated: false, residual: true }");
    let half = Epilogue { scale: Some(Scale::new(0.5)), ..plain };
    assert_eq!(format!("{half:?}"), "Epilogue { bias: true, act: None, gated: false, residual: true, scale: 0.5 }");
    let spec = |epilogue| GemmSpec { m: 64, n: 64, k: 64, batch: Batch::Static(1), epilogue, cfg: BIG };
    assert_ne!(gemm::<BF16T>(&spec(plain)), gemm::<BF16T>(&spec(half)));
}

/// The graph fallback (f32 here) applies the scale between the activation and
/// the residual: `scale·act(x·wᵀ + b) + r`.
#[test_case(Act::None, false; "plain")]
#[test_case(Act::Silu, false; "silu")]
#[test_case(Act::Gelu, true; "geglu")]
fn the_graph_scales_before_the_residual(act: Act, gated: bool) {
    let f = |shape: &[usize], seed: usize| {
        let len = shape.iter().product::<usize>();
        let data: Vec<f32> = (0..len).map(|i| ((i * 7 + seed * 13) % 17) as f32 / 8.0 - 1.0).collect();
        Tensor::from_slice(data).try_reshape(shape.iter().map(|&d| d as isize).collect::<Vec<_>>()).unwrap()
    };
    let (m, n, k) = (5, 6, 4);
    let rows = if gated { 2 * n } else { n };
    let (x, w, b, r) = (f(&[m, k], 1), f(&[rows, k], 2), f(&[rows], 3), f(&[m, n], 4));
    let opts = Linear { bias: Some(&b), act, gated, residual: Some(&r), scale: Some(-0.75) };
    let got = tk::linear(&x, &w, opts).unwrap().to_vec::<f32>().unwrap();
    let unscaled = Linear { residual: None, scale: None, ..opts };
    let y = tk::linear(&x, &w, unscaled).unwrap().to_vec::<f32>().unwrap();
    let r = r.to_vec::<f32>().unwrap();
    for (i, g) in got.iter().enumerate() {
        let want = -0.75 * y[i] + r[i];
        assert!((g - want).abs() <= 1e-5 * want.abs().max(1.0), "[{i}] = {g}, want {want}");
    }
}

// ---- attention ---------------------------------------------------------------------

#[test_case(64, Ok(FaCfg::new(64, 64, 2)); "d 64")]
#[test_case(128, Ok(FaCfg::new(64, 32, 2)); "d 128 keeps the half-width key block")]
#[test_case(48, Ok(FaCfg::new(64, 64, 2)); "d 48")]
#[test_case(32, Err(Fallback::Shape); "d 32")]
#[test_case(256, Err(Fallback::Shape); "d 256")]
fn attention_plans(d: usize, want: Result<FaCfg, Fallback>) {
    let (q, kv) = (ext(&[2, 100, 8, d]), ext(&[2, 37, 2, d]));
    assert_eq!(first(shape::attention(Some(&sm86()), &[BF16; 3], Some(&q), Some(&kv))), want);
    assert_eq!(first(shape::attention(Some(&sm86()), &[BF16; 4], Some(&q), Some(&kv))), want, "with a bias");
}

/// A bias takes the kernel in the stream type only.
#[test_case(BF16, true; "bf16")]
#[test_case(DType::Float16, false; "f16 under bf16")]
#[test_case(DType::Float32, false; "f32")]
fn biased_attention_plans(bias: DType, kernel: bool) {
    let (q, kv) = (ext(&[2, 100, 8, 64]), ext(&[2, 100, 8, 64]));
    let plan = shape::attention(Some(&sm86()), &[BF16, BF16, BF16, bias], Some(&q), Some(&kv));
    assert_eq!(matches!(plan, Plan::Kernel(_)), kernel, "{plan:?}");
    if !kernel {
        assert_eq!(plan, Plan::Graph(Fallback::Dtype));
    }
}

/// Both query-block widths and key-block widths, two- and three-deep rings,
/// within shared memory.
#[test_case(64, 99 << 10, 6; "d 64")]
#[test_case(128, 99 << 10, 6; "d 128")]
#[test_case(128, 48 << 10, 4; "d 128 under a static cap")]
fn attention_candidates(d: usize, smem: usize, count: usize) {
    let target = Target { smem_bytes: smem, ..sm86() };
    let (q, kv) = (ext(&[2, 100, 8, d]), ext(&[2, 37, 2, d]));
    let Plan::Kernel(list) = shape::attention(Some(&target), &[BF16; 3], Some(&q), Some(&kv)) else { panic!() };
    assert_eq!(list.len(), count, "{list:?}");
    assert!(list.iter().all(|c| c.smem_bytes(d) <= smem));
    for bq in [64, 128] {
        assert!(list.iter().any(|c| c.bq == bq));
    }
}

/// A decoder step's shapes get one-warp query tiles.
#[test_case(1, 64, 3; "one query, d 64")]
#[test_case(16, 128, 3; "sixteen queries, d 128")]
#[test_case(17, 64, 6; "seventeen queries keep the prefill list")]
fn decode_attention_candidates(t: usize, d: usize, count: usize) {
    let (q, kv) = (ext(&[4, t, 8, d]), ext(&[4, 1500, 8, d]));
    let Plan::Kernel(list) = shape::attention(Some(&sm86()), &[BF16; 3], Some(&q), Some(&kv)) else { panic!() };
    assert_eq!(list.len(), count, "{list:?}");
    assert!(list.iter().all(|c| (c.bq == 16) == (t <= 16)));
}

/// One and two always; around two blocks per SM otherwise, never past the
/// key blocks, and none without an SM count.
#[test_case(20, 24, &[1, 2, 3, 6]; "twenty tiles on 28 SMs")]
#[test_case(20, 2, &[1, 2]; "two key blocks")]
#[test_case(224, 24, &[1, 2]; "enough tiles already")]
#[test_case(1, 24, &[1, 2, 24]; "one tile: the blocks cap")]
fn split_candidates(tiles: usize, blocks: usize, want: &[usize]) {
    assert_eq!(config::split_candidates(&sm86(), tiles, blocks), want);
    let unknown = Target { sms: None, ..sm86() };
    assert_eq!(config::split_candidates(&unknown, tiles, blocks), [1]);
}

#[test]
fn attention_with_symbolic_keys_falls_back() {
    let q = ext(&[2, 100, 8, 64]);
    assert_eq!(shape::attention(Some(&sm86()), &[BF16; 3], Some(&q), None), Plan::Graph(Fallback::Symbolic));
}

// ---- heads -------------------------------------------------------------------------

#[test_case(64, Ok(NormCfg { br: 4 }); "d 64")]
#[test_case(128, Ok(NormCfg { br: 4 }); "d 128")]
#[test_case(48, Err(Fallback::Shape); "d 48")]
#[test_case(512, Err(Fallback::Shape); "d 512")]
fn heads_plan(d: usize, want: Result<NormCfg, Fallback>) {
    let x = ext(&[2, 100, 12 * d]);
    assert_eq!(first(shape::heads(Some(&sm86()), &[BF16], Some(&x), d)), want);
}

#[test]
fn heads_keep_the_graph_for_a_weight_off_the_stream_dtype() {
    let x = ext(&[2, 100, 12 * 64]);
    assert_eq!(shape::heads(Some(&sm86()), &[BF16, DType::Float16], Some(&x), 64), Plan::Graph(Fallback::Dtype));
}

#[test]
fn heads_need_three_dims() {
    let x = ext(&[200, 12 * 64]);
    assert_eq!(shape::heads(Some(&sm86()), &[BF16], Some(&x), 64), Plan::Graph(Fallback::Shape));
    assert_eq!(shape::heads(Some(&sm86()), &[BF16], None, 64), Plan::Graph(Fallback::Symbolic));
}

// ---- norms -------------------------------------------------------------------------

#[test_case(1024, Ok(vec![4, 8, 16]); "1024")]
#[test_case(256, Ok(vec![4, 8, 16]); "256")]
#[test_case(768, Err(Fallback::Shape); "not a power of two")]
#[test_case(128, Err(Fallback::Shape); "narrower than a warp's loads")]
#[test_case(4096, Err(Fallback::Shape); "wider than the measured range")]
fn norm_plans(d: usize, want: Result<Vec<usize>, Fallback>) {
    let want = want.map_or_else(Plan::Graph, |rows| Plan::Kernel(rows.into_iter().map(|br| NormCfg { br }).collect()));
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
fn no_device_no_kernels() {
    assert!(!tk::supported(&DeviceSpec::Cpu));
}

// ---- errors ----------------------------------------------------------------------------

fn t(shape: &[usize], dtype: DType) -> Tensor {
    Tensor::empty(shape, dtype)
}

#[test]
fn semantic_mismatches_are_errors() {
    let x = t(&[4, 64], BF16);
    let err = |r: tk::Result<Tensor>| r.expect_err("an error");
    let err3 = |r: tk::Result<(Tensor, Tensor, Tensor)>| r.expect_err("an error");
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
    let mask = t(&[2, 10], DType::Bool);
    let opts = Attn { keys: KeyMask::Bool(&mask), ..Attn::default() };
    assert!(matches!(err(tk::attention(&q, &kv2, &kv2, opts)), Error::Shape { operand: "key mask", .. }));
    let seg = t(&[2, 12], DType::Int32);
    let opts = Attn { seg_start: Some(&seg), ..Attn::default() };
    assert!(matches!(err(tk::attention(&q, &kv2, &kv2, opts)), Error::Shape { operand: "seg start", .. }));
    let cache = t(&[3, 12, 6, 64], BF16);
    let plain = Cache { head_start: 0, kv_heads: 2, row_map: None, appended: None };
    let opts = Attn { cache: Some(plain), ..Attn::default() };
    assert!(
        matches!(err(tk::attention(&q, &cache, &cache, opts)), Error::Shape { operand: "k", .. }),
        "rows need a map"
    );
    let map = t(&[2], DType::Int32);
    let opts = Attn { cache: Some(Cache { head_start: 5, row_map: Some(&map), ..plain }), ..Attn::default() };
    assert!(
        matches!(err(tk::attention(&q, &cache, &cache, opts)), Error::Shape { operand: "k", .. }),
        "heads past the row"
    );
    let app = t(&[2, 1, 2, 64], BF16);
    let opts =
        Attn { cache: Some(Cache { row_map: Some(&map), appended: Some((&app, &app)), ..plain }), ..Attn::default() };
    assert!(
        matches!(err(tk::attention(&q, &cache, &cache, opts)), Error::Shape { operand: "key lens", .. }),
        "appended needs lens"
    );
    let lens = t(&[2], DType::Int32);
    let wide = t(&[2, 1, 3, 64], BF16);
    let with_lens = Attn { keys: KeyMask::Lens(&lens), ..Attn::default() };
    let opts = Attn { cache: Some(Cache { row_map: Some(&map), appended: Some((&app, &wide)), ..plain }), ..with_lens };
    assert!(matches!(err(tk::attention(&q, &cache, &cache, opts)), Error::Shape { operand: "appended v", .. }));
    let bias = t(&[2, 6, 10, 13], BF16);
    let opts = Attn { cache: Some(Cache { row_map: Some(&map), appended: Some((&app, &app)), ..plain }), ..with_lens };
    assert!(tk::attention(&q, &cache, &cache, Attn { bias: Some(&bias), ..opts }).is_ok(), "the appended column");
    for dims in [[2, 6, 10, 12], [3, 6, 10, 13], [2, 2, 10, 13], [2, 6, 12, 13]] {
        let bias = t(&dims, BF16);
        let opts = Attn { bias: Some(&bias), ..opts };
        let got = tk::attention(&q, &cache, &cache, opts);
        assert!(matches!(err(got), Error::Shape { operand: "bias", .. }), "{dims:?}");
    }
    let bias = t(&[1, 6, 10, 12], BF16);
    assert!(tk::attention(&q, &kv2, &kv2, Attn { bias: Some(&bias), ..Attn::default() }).is_ok(), "shared");
    // Off the stream dtype is the graph's to handle, not an error.
    let bias = t(&[2, 6, 10, 12], DType::Float32);
    assert!(tk::attention(&q, &kv2, &kv2, Attn { bias: Some(&bias), ..Attn::default() }).is_ok(), "f32 bias");

    let qkv = t(&[2, 10, 8 * 64], BF16);
    let split = Qkv { heads: 4, kv_heads: 2, head_dim: 64, q_norm: None, k_norm: None, eps: 1e-6, rope: None };
    let bad = Qkv { kv_heads: 3, ..split };
    assert!(matches!(err3(tk::heads(&qkv, bad)), Error::Heads { heads: 4, kv_heads: 3, .. }));
    let bad = Qkv { heads: 8, ..split };
    assert!(matches!(err3(tk::heads(&qkv, bad)), Error::Shape { operand: "qkv", .. }));
    let w = t(&[32], BF16);
    let bad = Qkv { k_norm: Some(&w), ..split };
    assert!(matches!(err3(tk::heads(&qkv, bad)), Error::Shape { operand: "k norm", .. }));
    // Off the stream dtype is the graph's to handle, not an error.
    let w16 = t(&[64], DType::Float16);
    assert!(tk::heads(&qkv, Qkv { q_norm: Some(&w16), ..split }).is_ok());
    let cos = t(&[1, 10, 1, 64], BF16);
    let bad = Qkv { rope: Some((&cos, &cos)), ..split };
    assert!(matches!(err3(tk::heads(&qkv, bad)), Error::Shape { operand: "cos", .. }));
    let cos = t(&[1, 10, 1, 32], BF16);
    let sin = t(&[2, 10, 1, 32], BF16);
    let bad = Qkv { rope: Some((&cos, &sin)), ..split };
    assert!(matches!(err3(tk::heads(&qkv, bad)), Error::Shape { operand: "sin", .. }));

    let w = t(&[32], BF16);
    assert!(matches!(err(tk::layer_norm(&x, &w, None, 1e-5)), Error::Shape { operand: "w", .. }));
    let w = t(&[64], BF16);
    let r = t(&[4, 32], BF16);
    assert!(matches!(tk::add_rms_norm(&x, &r, &w, 1e-5).map(|_| ()), Err(Error::Shape { operand: "residual", .. })));
}
