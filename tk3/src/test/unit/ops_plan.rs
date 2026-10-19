//! The op layer's decisions on the host: which plan each shape, dtype and
//! target gets, and which requests are errors rather than fallbacks.

use svod_dtype::{AmdArch, DType, DeviceSpec, GpuArch, MetalFamily};
use svod_ir::SInt;
use svod_tensor::{Tensor, Variable};
use test_case::test_case;

use crate::atoms::{Target, sm86};
use crate::build::BF16 as BF16T;
use crate::kernels::Batch;
use crate::kernels::attention::FaCfg;
use crate::kernels::gemm::{Epilogue, GemmCfg, GemmSpec, Scale, gemm};
use crate::kernels::rows::NormCfg;
use crate::ops::config::{self, Planner};
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

/// A target with a matrix core and no tables.
fn apple() -> Target {
    Target::for_arch(GpuArch::Metal(MetalFamily::Apple(9)))
}

// ---- linear ------------------------------------------------------------------------

/// The model's lead for a big GEMM on sm_86: the 128×128 tile over 16 warps,
/// three 32-deep slots (100 KB holds them twice over).
const BIG: GemmCfg = GemmCfg { tile: [128, 128, 32], stages: 3, warps: [4, 4], group_m: 8, unroll: false };
/// The config measured fastest at 4096³ on the 3060 (25.4 TFLOP/s): the same
/// tile and ring over eight warps. It stays in the shortlist.
const BIG_MEASURED: GemmCfg = GemmCfg { tile: [128, 128, 32], stages: 3, warps: [2, 4], group_m: 8, unroll: false };

/// The untuned pick per shape on sm_86: a grid that fills the device takes
/// the 128×128 tile, 32 deep; fewer blocks per SM take deeper steps and, as
/// rows run out, shorter tiles; a reduction dim off 32 takes 16-deep steps.
#[test_case(&[4096, 4096], 4096, false, Ok(BIG); "large grid, deepest ring")]
#[test_case(&[4096, 4096], 4096, true, Ok(BIG); "gated keeps the ring in 99 KB")]
#[test_case(&[8, 256, 512], 2048, false, Ok(BIG); "a static batch counts toward the grid")]
#[test_case(&[2048, 512], 1024, false, Ok(gemm_cfg([128, 128, 64], 3, [4, 4], false)); "half the big grid, deeper steps")]
#[test_case(&[8, 37, 512], 512, false, Ok(gemm_cfg([64, 128, 64], 3, [4, 4], false)); "medium grid")]
#[test_case(&[37, 64], 96, false, Ok(gemm_cfg([32, 64, 64], 3, [2, 4], false)); "few rows")]
#[test_case(&[37, 48], 96, false, Ok(gemm_cfg([64, 64, 16], 3, [2, 2], false)); "k a multiple of 16 only")]
#[test_case(&[37, 40], 96, false, Err(Fallback::Config); "k off every bk")]
#[test_case(&[37, 64], 100, false, Err(Fallback::Shape); "n not a multiple of 8")]
#[test_case(&[0, 64], 96, false, Err(Fallback::Shape); "no rows")]
fn linear_plans(x: &[usize], n: usize, gated: bool, want: Result<GemmCfg, Fallback>) {
    assert_eq!(first(shape::linear(Some(&Planner::new(sm86())), &[BF16, BF16], Some(&ext(x)), n, gated)), want);
}

fn gemm_list(x: &Extent, n: usize, gated: bool) -> Vec<GemmCfg> {
    match shape::linear(Some(&Planner::new(sm86())), &[BF16, BF16], Some(x), n, gated) {
        Plan::Kernel(list) => list,
        Plan::Graph(why) => panic!("graph: {why:?}"),
    }
}

/// The 4096³ shortlist on sm_86: eight distinct tiles and warp grids, the
/// cheapest by the traffic model first, each with its one cheapest pipeline
/// (three stages, rolled: the deepest ring 100 KB holds, and `cp.async` slots
/// gain nothing from unrolling in the model), the measured peak among them.
#[test]
fn large_gemm_candidates() {
    let want = [
        BIG,
        gemm_cfg([128, 256, 32], 3, [4, 4], false),
        gemm_cfg([256, 128, 32], 3, [4, 4], false),
        BIG_MEASURED,
        gemm_cfg([128, 128, 32], 3, [4, 2], false),
        gemm_cfg([128, 256, 32], 3, [2, 4], false),
        gemm_cfg([256, 128, 32], 3, [4, 2], false),
        gemm_cfg([128, 256, 32], 3, [4, 2], false),
    ];
    assert_eq!(gemm_list(&ext(&[4096, 4096]), 4096, false), want);
}

/// Nemotron's projections (704 rows under a batch variable of capacity 1)
/// measured fastest on 64×64 tiles on the 3060, where 128-row tiles pad 64
/// rows and leave the 28 SMs with too few blocks: the shortlist keeps a
/// 64-row tile for the tune store to find, and spans eight tiles and grids.
#[test_case(512, 512, false; "qkv-sized out")]
#[test_case(1536, 512, false; "fused qkv")]
#[test_case(2048, 512, false; "ffn up")]
#[test_case(512, 2048, false; "ffn down")]
#[test_case(2048, 512, true; "gated up")]
fn small_m_gemm_candidates(n: usize, k: usize, gated: bool) {
    let x = Extent { dims: vec![1, 704, k], var: true };
    let list = gemm_list(&x, n, gated);
    assert_eq!(list.len(), 8);
    assert!(list.iter().any(|c| c.tile[0] == 64), "{list:?}");
    for (i, c) in list.iter().enumerate() {
        assert!(!list[..i].iter().any(|o| o.tile[..2] == c.tile[..2] && o.warps == c.warps), "{list:?}");
    }
}

/// A gated weight stages a B tile twice the config's width and holds two
/// accumulators: on gfx1201 the 128×128 tile over eight warps costs 216 of a
/// lane's 256 registers gated where its half-width form costs 128, and its
/// 64-deep form does not fit at all. The model knows it as a `2·bn`-wide
/// problem, so Qwen3's gate/up shortlist leads with 128×128 over 16 warps
/// and holds the half-width tiles (measured 329 / 329 / 338 µs against
/// 358 for the next and 431 for tk1); only the plain shortlist reaches a
/// 256-wide tile.
#[test]
fn a_gated_gemm_is_a_double_width_problem() {
    let target = Target { sms: Some(64), ..Target::for_arch(GpuArch::Amd(AmdArch::Gfx1201)) };
    let eight = gemm_cfg([128, 128, 32], 2, [2, 4], true);
    let half = gemm_cfg([128, 64, 32], 2, [2, 4], true);
    let deep = gemm_cfg([128, 128, 64], 2, [2, 4], true);
    assert_eq!((eight.registers(&target, 2), half.registers(&target, 2)), (Some(216), Some(128)));
    assert!(deep.registers(&target, 2).is_some_and(|r| r > 256), "{:?}", deep.registers(&target, 2));
    let planner = Planner::new(target.clone());
    let gated = planner.gemm_candidates(1, 4096, 3072, 1024, true);
    assert_eq!(gated[0], gemm_cfg([128, 128, 32], 2, [4, 4], true), "{gated:?}");
    assert!(gated.contains(&half) && gated.contains(&gemm_cfg([128, 64, 32], 2, [4, 2], true)), "{gated:?}");
    assert!(!gated.contains(&deep) && gated.iter().all(|c| c.tile[2] >= 32), "{gated:?}");
    let plain = planner.gemm_candidates(1, 4096, 3072, 1024, false);
    assert!(gated.iter().all(|c| c.tile[1] <= 128) && plain.iter().any(|c| c.tile[1] == 256), "{plain:?}");
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
        let Plan::Kernel(list) =
            shape::linear(Some(&Planner::new(target.clone())), &[BF16; 2], Some(&ext(x)), n, gated)
        else {
            panic!("no kernel")
        };
        assert!((1..=8).contains(&list.len()), "{list:?}");
        for (i, c) in list.iter().enumerate() {
            assert!(k.is_multiple_of(c.tile[2]) && c.smem_bytes(gated) <= smem, "{c:?}");
            assert!(!list[..i].contains(c), "duplicate {c:?}");
        }
    }
}

/// A target capped at static shared memory drops a stage for the gated GEMM:
/// its three-slot ring of a 384-wide fill is 72 KB.
#[test]
fn gated_linear_drops_a_stage_under_a_static_cap() {
    let mut target = sm86();
    target.smem_bytes = 48 << 10;
    let want = Ok(GemmCfg { stages: 2, ..BIG });
    assert_eq!(
        first(shape::linear(Some(&Planner::new(target.clone())), &[BF16, BF16], Some(&ext(&[4096, 4096])), 4096, true)),
        want
    );
}

#[test_case(None, &[BF16, BF16], true, Fallback::Target; "no target")]
#[test_case(Some(apple()), &[BF16, BF16], true, Fallback::Target; "no tables for the arch")]
#[test_case(Some(sm86()), &[DType::Float32, DType::Float32], true, Fallback::Dtype; "f32 keeps the graph")]
#[test_case(Some(sm86()), &[BF16, DType::Float32], true, Fallback::Dtype; "mixed operand types")]
#[test_case(Some(sm86()), &[DType::Int32, DType::Int32], true, Fallback::Dtype; "integers")]
#[test_case(Some(sm86()), &[BF16, BF16], false, Fallback::Symbolic; "symbolic past the batch")]
fn every_op_falls_back_alike(target: Option<Target>, dtypes: &[DType], static_dims: bool, why: Fallback) {
    let x = static_dims.then(|| ext(&[2, 128, 1024]));
    let kv = static_dims.then(|| ext(&[2, 128, 16, 64]));
    let q = static_dims.then(|| ext(&[2, 128, 16, 64]));
    let planner = target.map(Planner::new);
    let target = planner.as_ref();
    assert_eq!(shape::linear(target, dtypes, x.as_ref(), 1024, false), Plan::Graph(why));
    assert_eq!(shape::attention(target, dtypes, q.as_ref(), kv.as_ref(), false), Plan::Graph(why));
    assert_eq!(shape::norm(target, dtypes, x.as_ref()), Plan::Graph(why));
}

/// A bound variable that is the reduced dim itself is no batch.
#[test]
fn a_variable_reduced_dim_falls_back() {
    let x = Extent { dims: vec![64], var: true };
    assert_eq!(
        shape::linear(Some(&Planner::new(sm86())), &[BF16; 2], Some(&x), 64, false),
        Plan::Graph(Fallback::Symbolic)
    );
    let x = Extent { dims: vec![1024], var: true };
    assert_eq!(shape::norm(Some(&Planner::new(sm86())), &[BF16; 2], Some(&x)), Plan::Graph(Fallback::Symbolic));
}

#[test]
fn f16_takes_the_kernels() {
    let f16 = [DType::Float16, DType::Float16];
    assert!(matches!(
        shape::linear(Some(&Planner::new(sm86())), &f16, Some(&ext(&[64, 64])), 64, false),
        Plan::Kernel(_)
    ));
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

/// A plain global view prints as it did before row maps existed, so a
/// GEMM's program fingerprint, and with it the tune store, is unchanged; a
/// gathered view prints its map.
#[test]
fn an_unmapped_view_keeps_its_tune_key() {
    use crate::ir::{ParamId, Place, RowMap, ScalarId};
    let (param, offset, stride, bounds) =
        (ParamId(0), ScalarId(1), [ScalarId(2), ScalarId(3)], [None, Some(ScalarId(4))]);
    let plain = Place::Global { param, offset, stride, bounds, rows: None };
    let old = "Global { param: ParamId(0), offset: ScalarId(1), stride: [ScalarId(2), ScalarId(3)], bounds: [None, Some(ScalarId(4))] }";
    assert_eq!(format!("{plain:?}"), old);
    let rows = Some(RowMap { offset: ScalarId(5), valid: None });
    let gathered = Place::Global { param, offset, stride, bounds, rows };
    assert!(format!("{gathered:?}").ends_with("rows: RowMap { offset: ScalarId(5), valid: None } }"));
    let smem = Place::Smem { alloc: crate::ir::SmemId(1), offset };
    assert_eq!(format!("{smem:?} {:?}", Place::Reg), "Smem { alloc: SmemId(1), offset: ScalarId(1) } Reg");
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

/// The untuned pick per head size on sm_86 (a prefill of 100 queries over
/// 37 keys): 16-key blocks at `d = 128`, where a key block's fragments and
/// scores are what the register file is spent on, and the query block the
/// device's SMs fill; `d = 48` keeps the shapes whose K/V fills divide among
/// the block's threads (96-byte rows); `d = 32` fits 64-key blocks. A head
/// size past the register file keeps the graph.
#[test_case(64, Ok((64, 16)); "d 64")]
#[test_case(128, Ok((128, 16)); "d 128")]
#[test_case(48, Ok((64, 64)); "d 48")]
#[test_case(32, Ok((128, 64)); "d 32")]
#[test_case(256, Err(Fallback::Shape); "d 256")]
fn attention_plans(d: usize, want: Result<(usize, usize), Fallback>) {
    let (q, kv) = (ext(&[2, 100, 8, d]), ext(&[2, 37, 2, d]));
    let lead = |dtypes: &[DType]| {
        first(shape::attention(Some(&Planner::new(sm86())), dtypes, Some(&q), Some(&kv), false)).map(|c| (c.bq, c.bkv))
    };
    assert_eq!(lead(&[BF16; 3]), want);
    assert_eq!(lead(&[BF16; 4]), want, "with a bias");
}

/// A bias takes the kernel in the stream type only.
#[test_case(BF16, true; "bf16")]
#[test_case(DType::Float16, false; "f16 under bf16")]
#[test_case(DType::Float32, false; "f32")]
fn biased_attention_plans(bias: DType, kernel: bool) {
    let (q, kv) = (ext(&[2, 100, 8, 64]), ext(&[2, 100, 8, 64]));
    let plan = shape::attention(Some(&Planner::new(sm86())), &[BF16, BF16, BF16, bias], Some(&q), Some(&kv), false);
    assert_eq!(matches!(plan, Plan::Kernel(_)), kernel, "{plan:?}");
    if !kernel {
        assert_eq!(plan, Plan::Graph(Fallback::Dtype));
    }
}

/// The shortlist spans distinct query and key block sizes within shared
/// memory, one pipeline per tile, and keeps the configs measured fastest on
/// the 3060: `(64, 64)` at `d = 64` and `(64, 32)` at `d = 128` (64 KB per
/// block left one block per SM there). The score tile needs no scratch
/// there: the `mma.sync` accumulator is its A operand.
#[test_case(64, 99 << 10, (64, 64); "d 64")]
#[test_case(128, 99 << 10, (64, 32); "d 128")]
#[test_case(128, 48 << 10, (64, 32); "d 128 under a static cap")]
fn attention_candidates(d: usize, smem: usize, measured: (usize, usize)) {
    let target = Target { smem_bytes: smem, ..sm86() };
    let (q, kv) = (ext(&[2, 100, 8, d]), ext(&[2, 37, 2, d]));
    let Plan::Kernel(list) =
        shape::attention(Some(&Planner::new(target.clone())), &[BF16; 3], Some(&q), Some(&kv), false)
    else {
        panic!()
    };
    assert!((4..=8).contains(&list.len()), "{list:?}");
    assert!(list.iter().all(|c| c.smem_bytes(d) <= smem && c.fits(&target, d)), "{list:?}");
    assert!(list.iter().any(|c| (c.bq, c.bkv) == measured), "{measured:?} not in {list:?}");
    for (i, c) in list.iter().enumerate() {
        assert!(!list[..i].iter().any(|o| (o.bq, o.bkv) == (c.bq, c.bkv)), "one pipeline per tile: {list:?}");
        assert_eq!(c.scratch_bytes(&target), 0);
    }
}

/// A warp past the queries hides nothing and adds work: a decoder step's
/// lead has one warp per 16 queries, no more.
#[test_case(1, 64; "one query, d 64")]
#[test_case(16, 128; "sixteen queries, d 128")]
#[test_case(17, 64; "seventeen queries, two warps")]
#[test_case(100, 64; "a prefill fills a block")]
fn decode_attention_leads_with_the_queries_warps(t: usize, d: usize) {
    let (q, kv) = (ext(&[4, t, 8, d]), ext(&[4, 1500, 8, d]));
    let Plan::Kernel(list) = shape::attention(Some(&Planner::new(sm86())), &[BF16; 3], Some(&q), Some(&kv), false)
    else {
        panic!()
    };
    let lead = list[0];
    assert!(lead.bq <= t.div_ceil(16) * 16, "{lead:?} for {t} queries");
    if t <= 16 {
        assert_eq!(lead.bq, 16, "{list:?}");
    }
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
    assert_eq!(
        shape::attention(Some(&Planner::new(sm86())), &[BF16; 3], Some(&q), None, false),
        Plan::Graph(Fallback::Symbolic)
    );
}

// ---- heads -------------------------------------------------------------------------

#[test_case(64, Ok(NormCfg { br: 4 }); "d 64")]
#[test_case(128, Ok(NormCfg { br: 4 }); "d 128")]
#[test_case(48, Err(Fallback::Shape); "d 48")]
#[test_case(512, Err(Fallback::Shape); "d 512")]
fn heads_plan(d: usize, want: Result<NormCfg, Fallback>) {
    let x = ext(&[2, 100, 12 * d]);
    assert_eq!(first(shape::heads(Some(&Planner::new(sm86())), &[BF16], Some(&x), d)), want);
}

#[test]
fn heads_keep_the_graph_for_a_weight_off_the_stream_dtype() {
    let x = ext(&[2, 100, 12 * 64]);
    assert_eq!(
        shape::heads(Some(&Planner::new(sm86())), &[BF16, DType::Float16], Some(&x), 64),
        Plan::Graph(Fallback::Dtype)
    );
}

#[test]
fn heads_need_three_dims() {
    let x = ext(&[200, 12 * 64]);
    assert_eq!(shape::heads(Some(&Planner::new(sm86())), &[BF16], Some(&x), 64), Plan::Graph(Fallback::Shape));
    assert_eq!(shape::heads(Some(&Planner::new(sm86())), &[BF16], None, 64), Plan::Graph(Fallback::Symbolic));
}

// ---- norms -------------------------------------------------------------------------

#[test_case(1024, Ok(vec![4, 8, 16]); "1024")]
#[test_case(256, Ok(vec![4, 8, 16]); "256")]
#[test_case(768, Err(Fallback::Shape); "not a power of two")]
#[test_case(128, Err(Fallback::Shape); "narrower than a warp's loads")]
#[test_case(4096, Err(Fallback::Shape); "wider than the measured range")]
fn norm_plans(d: usize, want: Result<Vec<usize>, Fallback>) {
    let want = want.map_or_else(Plan::Graph, |rows| Plan::Kernel(rows.into_iter().map(|br| NormCfg { br }).collect()));
    assert_eq!(shape::norm(Some(&Planner::new(sm86())), &[BF16; 2], Some(&ext(&[37, d]))), want);
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

// ---- conv2d ------------------------------------------------------------------------

fn conv_geom(hw: usize, cin: usize, cout: usize, k: usize, s: usize) -> crate::kernels::conv::ConvGeom {
    crate::kernels::conv::ConvGeom {
        h: hw,
        w: hw,
        cin,
        cout,
        kernel: [k, k],
        stride: [s, s],
        pad: [k / 2, k / 2],
        dilation: [1, 1],
    }
}

/// The untuned pick's output width: a body of 96 or 48 channels takes a tile
/// of its width, not a power of two padded by a third.
fn conv_plan(x: &[usize], g: crate::kernels::conv::ConvGeom, groups: usize) -> Result<usize, Fallback> {
    let plan = shape::conv2d(Some(&Planner::new(sm86())), &[F16; 3], F16, None, Some(&ext(x)), &g, groups);
    first(plan).map(|c| c.gemm.tile[1])
}

const F16: DType = DType::Float16;

#[test_case(&[1, 80, 80, 384], conv_geom(80, 384, 96, 3, 1), 1, Ok(96); "head conv, 96 wide")]
#[test_case(&[1, 20, 20, 192], conv_geom(20, 192, 192, 3, 1), 1, Ok(64); "a starved shallow body")]
#[test_case(&[8, 80, 80, 384], conv_geom(80, 384, 384, 3, 2), 1, Ok(128); "a static batch folds into the rows")]
#[test_case(&[1, 40, 40, 192], conv_geom(40, 192, 192, 3, 1), 1, Ok(96); "a shallow body filling the device")]
#[test_case(&[1, 80, 80, 96], conv_geom(80, 96, 96, 3, 1), 1, Ok(96); "96 wide, shallow")]
#[test_case(&[1, 640, 640, 3], conv_geom(640, 3, 96, 3, 2), 1, Err(Fallback::Shape); "rgb stem")]
#[test_case(&[1, 160, 160, 48], conv_geom(160, 48, 48, 3, 1), 1, Ok(48); "48 wide on one warp")]
#[test_case(&[1, 20, 20, 64], conv_geom(20, 64, 60, 3, 1), 1, Err(Fallback::Shape); "cout off 8")]
#[test_case(&[1, 20, 20, 64], conv_geom(20, 64, 64, 3, 1), 64, Err(Fallback::Shape); "depthwise")]
#[test_case(&[0, 20, 20, 64], conv_geom(20, 64, 64, 3, 1), 1, Err(Fallback::Shape); "no images")]
#[test_case(&[1, 1, 1, 64], conv_geom(1, 64, 64, 5, 1).with_pad(0), 1, Err(Fallback::Shape); "empty output")]
fn conv_plans(x: &[usize], g: crate::kernels::conv::ConvGeom, groups: usize, want: Result<usize, Fallback>) {
    assert_eq!(conv_plan(x, g, groups), want);
}

#[test_case(&[F16, F16, F16], DType::Float32, Some(DType::Float32), true; "f32 out with an f32 residual")]
#[test_case(&[F16, F16, F16], DType::Float32, Some(F16), false; "residual off the output type")]
#[test_case(&[F16, F16, F16], DType::BFloat16, None, false; "another half output")]
#[test_case(&[DType::Float32; 3], DType::Float32, None, false; "f32 keeps the graph")]
#[test_case(&[F16, BF16, F16], F16, None, false; "mixed operands")]
fn conv_dtype_plans(dtypes: &[DType], out: DType, residual: Option<DType>, kernel: bool) {
    let g = conv_geom(80, 384, 96, 3, 1);
    let plan = shape::conv2d(Some(&Planner::new(sm86())), dtypes, out, residual, Some(&ext(&[1, 80, 80, 384])), &g, 1);
    assert_eq!(matches!(plan, Plan::Kernel(_)), kernel, "{plan:?}");
    if !kernel {
        assert_eq!(plan, Plan::Graph(Fallback::Dtype));
    }
}

/// A bound batch walks grid z: candidates for one image's rows, the batch
/// counted toward the grid.
#[test]
fn a_bound_conv_batch_plans_per_image() {
    let g = conv_geom(80, 384, 384, 3, 2);
    let x = Extent { dims: vec![8, 80, 80, 384], var: true };
    let plan = shape::conv2d(Some(&Planner::new(sm86())), &[F16; 3], F16, None, Some(&x), &g, 1);
    assert_eq!(plan, Plan::Kernel(Planner::new(sm86()).conv_candidates(8, 1600, &g)));
    let plan = shape::conv2d(Some(&Planner::new(apple())), &[F16; 3], F16, None, Some(&x), &g, 1);
    assert_eq!(plan, Plan::Graph(Fallback::Target));
    assert_eq!(
        shape::conv2d(Some(&Planner::new(sm86())), &[F16; 3], F16, None, None, &g, 1),
        Plan::Graph(Fallback::Symbolic)
    );
}

/// RDNA plans kernels from the lattice under its own facts: two-slot
/// register-staged rings only, unrolled over the slots, 16×16 WMMA-tileable
/// warp grids, attention within 64 KB of LDS with its score relayout.
#[test]
fn rdna_plans_register_staged_kernels() {
    let target = rdna3();
    let Plan::Kernel(gemms) =
        shape::linear(Some(&Planner::new(target.clone())), &[BF16, BF16], Some(&ext(&[4096, 4096])), 4096, false)
    else {
        panic!("a GEMM kernel")
    };
    assert_eq!(gemms[0], gemm_cfg([128, 256, 32], 2, [4, 4], true));
    assert!(gemms.iter().all(|c| c.stages == 2 && c.unroll && c.fits(&target)));
    let Plan::Kernel(small) =
        shape::linear(Some(&Planner::new(target.clone())), &[BF16, BF16], Some(&ext(&[704, 512])), 512, false)
    else {
        panic!("a GEMM kernel")
    };
    assert!(small.iter().any(|c| c.tile[..2] == [64, 64]), "a small grid offers the 64×64 tile: {small:?}");
    for (d, t) in [(64, 1500), (64, 1)] {
        let q = ext(&[1, t, 8, d]);
        let Plan::Kernel(fa) =
            shape::attention(Some(&Planner::new(target.clone())), &[BF16; 3], Some(&q), Some(&q), false)
        else {
            panic!("an attention kernel")
        };
        let within = |c: &FaCfg| c.smem_bytes(d) + c.scratch_bytes(&target) <= 64 << 10;
        assert!(fa.iter().all(|c| c.stages == 2 && c.unroll && within(c)), "{fa:?}");
        assert!(fa.iter().all(|c| c.scratch_bytes(&target) == c.bq * c.bkv * 4), "RDNA3 re-holds its scores");
    }
    // RDNA3's replicated operand fragments and the per-key value gather put
    // every `d = 128` config past the register file: the graph, until the
    // value tile is stored transposed.
    let q = ext(&[1, 1500, 8, 128]);
    let plan = shape::attention(Some(&Planner::new(target.clone())), &[BF16; 3], Some(&q), Some(&q), false);
    assert_eq!(plan, Plan::Graph(Fallback::Shape));
    assert!(matches!(
        shape::norm(Some(&Planner::new(target.clone())), &[BF16; 2], Some(&ext(&[37, 1024]))),
        Plan::Kernel(_)
    ));
}
