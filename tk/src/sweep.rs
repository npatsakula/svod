//! The kernels the host sweep lowers, shared by the tests and the
//! `tk_digest` tool: every family the op layer plans per target
//! ([`families`], [`group_families`]) and, for a byte-identical refactor
//! gate, the wider set of the planner's shortlists and the attention,
//! convolution, norm, heads and probe variants ([`digested`]), all under
//! the declared costs ([`costs`]) so a ranking is the same on any machine;
//! the targets of the sweep ([`targets`]); their LLVM text ([`render`]);
//! and digests of it compared across builds ([`digest`], [`diff`]).
//! Fixtures, not an API: hidden from the docs.

use std::sync::Arc;

use svod_codegen::Renderer;
use svod_codegen::llvm::LlvmTextRenderer;
use svod_dtype::{AmdArch, CudaArch, DType, DeviceSpec, GpuArch};
use svod_ir::{Op, UOp, ops};

use crate::atoms::{Core, Costs, Device, Target};
use crate::build::{BF16, F16};
use crate::ir::{Program, Var};
use crate::kernels::attention::{AttnMask, AttnSpec, Bias, Cache, Chunk, CombineSpec, FaCfg, attention, combine};
use crate::kernels::conv::{ConvCfg, ConvGeom, ConvSpec};
use crate::kernels::gemm::{Epilogue, GemmCfg, GemmSpec, Scale, gemm};
use crate::kernels::heads::{HeadsSpec, Rope, heads};
use crate::kernels::nearest::{
    Floors, Live, MergeSpec, Metric, NearestCfg, NearestSpec, SelectCfg, SelectSpec, nearest, nearest_merge,
    nearest_select,
};
use crate::kernels::rows::{Norm, NormCfg, NormSpec, norm};
use crate::kernels::{Act, Batch};
use crate::lower::{Lowering, lower};
use crate::ops::config::{self, Nearest, Planner};

/// Costs a test declares where it needs a ranking or an orientation: the
/// guessed sm_86-class fixture, the warpgroup step free.
pub(crate) fn costs() -> Costs {
    Costs { step_group: 0.0, ..crate::probe::GUESSED }
}

/// `target` under the declared [`costs`].
pub(crate) fn declared(target: Target) -> Device {
    Device { target, costs: costs() }
}

/// The planner of `target` under the declared [`costs`].
pub(crate) fn planner(target: Target) -> Planner {
    Planner::new(declared(target)).expect("a target with an SM count")
}

/// The front of a ranking: its first eight tiles, what a host test judges
/// a ranking by (the tuner's shortlist, which compiles on a device, runs
/// [`SHORTLIST`](crate::ops::config::SHORTLIST) deep).
pub(crate) fn first_eight<C: crate::ops::config::Tiled>(ranked: &[C]) -> Vec<C> {
    crate::ops::config::distinct(ranked).take(8).collect()
}

/// The convolution candidates of a ranking's [`first_eight`].
pub(crate) fn conv_first_eight(planner: &Planner, batches: usize, m: usize, g: &ConvGeom) -> Vec<ConvCfg> {
    planner.conv_splits(batches, m, g, &first_eight(&planner.conv_ranked(batches, m, g)))
}

/// One launch of a family: its name, program and lowering.
pub type Launch = (String, Program, Lowering);

/// The first candidate the op layer plans for each kernel family on `target`.
pub fn families(target: &Target) -> Vec<Launch> {
    let mut out: Vec<Launch> = vec![];
    let mut push = |name: &str, prog: Program, lowering: Lowering| out.push((name.to_string(), prog, lowering));
    let planner = planner(target.clone());

    // Shapes stay small for the interpreter; the tiles are those planned
    // for a large GEMM, so the biggest register tiles are lowered too.
    let gemm_spec = |m, n, k, epilogue: Epilogue| {
        let cfg = planner.gemm_ranked(1, 4096, 4096, k, epilogue.gated)[0];
        GemmSpec { m, n, k, batch: Batch::Static(1), epilogue, cfg }
    };
    let plain = gemm_spec(130, 192, 128, Epilogue::DEFAULT);
    for (i, cfg) in first_eight(&planner.gemm_ranked(1, 4096, 4096, 4096, false)).into_iter().enumerate() {
        let spec = GemmSpec { cfg, ..plain.clone() };
        push(&format!("gemm #{i} {:?}x{}", cfg.tile, cfg.stages), gemm::<BF16>(&spec), cfg.lowering(&planner.device));
    }
    // One slot holds a tile twice as deep in the same shared memory.
    let one_slot = GemmSpec { cfg: config::gemm_cfg([64, 64, 64], 1, [2, 2], false), ..plain.clone() };
    push("gemm one slot", gemm::<BF16>(&one_slot), one_slot.cfg.lowering(&planner.device));
    if target.crews >= 2 {
        let cfg = GemmCfg { crews: 2, ..config::gemm_cfg([64, 64, 64], 1, [2, 4], false) };
        let crews = GemmSpec { cfg, ..plain.clone() };
        push("gemm two crews", gemm::<BF16>(&crews), cfg.lowering(&planner.device));
    }
    let swiglu = gemm_spec(70, 64, 64, Epilogue { bias: true, act: Act::Silu, gated: true, ..Epilogue::DEFAULT });
    push("gemm swiglu", gemm::<F16>(&swiglu), swiglu.cfg.lowering(&planner.device));
    let half_step = Epilogue { residual: true, act: Act::Gelu, scale: Some(Scale::new(0.5)), ..Epilogue::DEFAULT };
    let half_step =
        GemmSpec { cfg: planner.gemm_ranked(1, 300, 512, 256, false)[0], ..gemm_spec(100, 96, 256, half_step) };
    push("gemm gelu scale residual", gemm::<BF16>(&half_step), half_step.cfg.lowering(&planner.device));

    // The planner's lead for the shape; a head size it plans no kernel for
    // has no family.
    let fa = |t, tk, d, mask: AttnMask, cache: Option<Cache>, batch| {
        let cfg = *planner.attention_ranked(8, t, tk, d, mask.causal, mask.bias.is_some()).first()?;
        let splits = if cache.is_some() { 2 } else { 1 };
        let cfg = crate::kernels::attention::FaCfg { splits, ..cfg };
        Some(AttnSpec { batch, t, tk, heads: 4, kv_heads: 2, d, mask, cache, scale: 0.125, cfg })
    };
    let causal = AttnMask { causal: true, ..AttnMask::default() };
    for d in [64, 128] {
        let Some(spec) = fa(100, 100, d, causal, None, Batch::Static(2)) else { continue };
        for (i, cfg) in first_eight(&planner.attention_ranked(8, 100, 100, d, true, false)).into_iter().enumerate() {
            let spec = AttnSpec { cfg, ..spec.clone() };
            let name = format!("attention d{d} causal #{i} {}x{}x{}", cfg.bq, cfg.bkv, cfg.stages);
            push(&name, attention::<BF16>(&spec), cfg.lowering(&planner.device));
        }
    }
    let masked = AttnMask { key_lens: true, key_mask: true, ..AttnMask::default() };
    let spec = fa(37, 75, 64, masked, None, Batch::Var { name: "b".into(), min: 1, max: 2 }).expect("d 64 plans");
    push("attention d64 key masks", attention::<F16>(&spec), spec.cfg.lowering(&planner.device));
    // The bias streams through its own ring slot; its 75 keys end inside a
    // key block, so the ring's fill meets the column bound.
    let biased = AttnMask { key_lens: true, bias: Some(Bias::PerBatch), ..AttnMask::default() };
    let spec = fa(37, 75, 64, biased, None, Batch::Static(2)).expect("d 64 plans with a bias");
    push("attention d64 bias", attention::<F16>(&spec), spec.cfg.lowering(&planner.device));
    let cache = Cache { rows: 6, heads_total: 6, row_map: true, appended: 1, app_lens: false };
    let lens = AttnMask { key_lens: true, ..AttnMask::default() };
    let spec = fa(1, 150, 64, lens, Some(cache), Batch::Var { name: "b".into(), min: 1, max: 3 }).expect("d 64 plans");
    push("attention d64 cache", attention::<F16>(&spec), spec.cfg.lowering(&planner.device));
    let cfg = crate::kernels::attention::FaCfg {
        key_warps: 4,
        splits: 2,
        ..crate::kernels::attention::FaCfg::new(16, 16, 2)
    };
    let spec = AttnSpec { cfg, ..spec };
    push("attention d64 cache key warps", attention::<F16>(&spec), spec.cfg.lowering(&planner.device));
    // Key warps re-hold the biased score tile through their own regions of
    // the relayout scratch where the core's accumulator cannot feed a product.
    let spec = AttnSpec { mask: AttnMask { bias: Some(Bias::PerBatch), ..lens }, ..spec };
    push("attention d64 cache key warps bias", attention::<F16>(&spec), spec.cfg.lowering(&planner.device));
    let merge = CombineSpec { batch: spec.batch.clone(), t: 1, heads: 4, d: 64, splits: 2, cfg: NormCfg { br: 4 } };
    push("attention combine", combine::<F16>(&merge), merge.cfg.lowering(&planner.device));

    for (name, kind, d, residual, bias) in
        [("layer norm", Norm::Layer, 1024, true, true), ("rms norm", Norm::Rms, 512, false, false)]
    {
        let cfg = config::norm_candidates(d)[0];
        let spec = NormSpec { norm: kind, rows: 37, d, batch: Batch::Static(1), eps: 1e-5, residual, bias, cfg };
        push(name, norm::<BF16>(&spec), cfg.lowering(&planner.device));
    }

    for d in [64, 128] {
        let cfg = config::heads_candidates(d)[0];
        let spec = HeadsSpec {
            batch: Batch::Static(2),
            t: 20,
            heads: 4,
            kv_heads: 2,
            d,
            q_norm: true,
            k_norm: true,
            eps: 1e-6,
            rope: Some(Rope { per_batch: false }),
            cfg,
        };
        push(&format!("heads d{d}"), heads::<BF16>(&spec), cfg.lowering(&planner.device));
    }

    // The tiles planned for a large self-join, over a ragged small one; a
    // row's least alone, and lists of its least scores.
    let both = Floors { row: true, col: true };
    for (name, d, metric, floors, groups, k) in [
        ("nearest squared l2", 64, Metric::SquaredL2, Floors::default(), false, 1),
        ("nearest l2 floors", 16, Metric::L2, both, false, 1),
        ("nearest groups", 32, Metric::ShiftedL2, Floors::default(), true, 1),
        ("nearest neg dot col floor", 64, Metric::NegDot, Floors { row: false, col: true }, false, 1),
        ("nearest streamed neg dot", 800, Metric::NegDot, Floors::default(), false, 1),
        ("nearest streamed l2 floors groups", 512, Metric::L2, both, true, 1),
        ("nearest 4 squared l2", 64, Metric::SquaredL2, Floors::default(), false, 4),
        ("nearest 16 l2 floors groups", 32, Metric::L2, both, true, 16),
        ("nearest 8 streamed neg dot", 768, Metric::NegDot, Floors::default(), false, 8),
        ("nearest 32 streamed shifted l2 groups", 512, Metric::ShiftedL2, Floors::default(), true, 32),
        ("nearest 64 queued squared l2", 64, Metric::SquaredL2, Floors::default(), false, 64),
        ("nearest 100 queued l2 floors groups", 32, Metric::L2, both, true, 100),
        ("nearest 256 queued streamed neg dot", 256, Metric::NegDot, Floors::default(), false, 256),
        ("nearest 48 registers l2 floors", 64, Metric::L2, both, false, 48),
    ] {
        let p = Nearest { n: 1 << 16, m: 1 << 16, d, metric, floors, groups, k };
        // A named way's first config, else the ranking's first.
        let way = |c: &&NearestCfg| match name {
            _ if name.contains("queued") => c.queue.is_some(),
            _ if name.contains("registers") => c.queue.is_none() && c.select.is_none(),
            _ => true,
        };
        let Some(&cfg) = planner.nearest_ranked(p).iter().find(way) else { continue };
        let spec = NearestSpec { n: 100, m: 150, d, metric, floors, groups, k, cfg, live: Live::default() };
        push(name, nearest::<BF16>(&spec), cfg.lowering(&planner.device));
    }
    // Split over the candidates (shares of one tile and of two), and a merge
    // of shares, of a least alone and of lists: in registers, the longest
    // there, and queued.
    for (k, queue, name) in [(1, None, ""), (8, None, " 8"), (64, None, " 64"), (256, Some(32), " 256 queued")] {
        let cfg = NearestCfg { splits: 2, queue, ..NearestCfg::streamed(32, 32, 32, 2) };
        let (metric, floors, live) = (Metric::L2, Floors::default(), Live::default());
        let spec = NearestSpec { n: 100, m: 90, d: 64, metric, floors, groups: true, k, cfg, live };
        push(&format!("nearest{name} split"), nearest::<BF16>(&spec), cfg.lowering(&planner.device));
        let merge = MergeSpec { n: 300, splits: 3, metric: Metric::L2, k, live: Live::default() };
        push(&format!("nearest{name} merge"), nearest_merge(&merge), merge.lowering(&planner.device));
    }

    // A score matrix, and its listing into empty lists of 100 and of 1024
    // (carried lists hold an order the scratch inputs would not).
    let select = SelectCfg { warps: 4, queue: 2 };
    let cfg = NearestCfg { select: Some(select), ..NearestCfg::new(64, 32, 2) };
    let (metric, live) = (Metric::L2, Live::default());
    let spec = NearestSpec { n: 100, m: 150, d: 64, metric, floors: both, groups: true, k: 100, cfg, live };
    push("nearest matrix", nearest::<BF16>(&spec), cfg.lowering(&planner.device));
    for (k, queue, name) in [(100, 2, "100"), (100, 4, "100 queue 4"), (1024, 8, "1024")] {
        let select = SelectCfg { queue, ..select };
        let b =
            SelectSpec { n: 100, cols: 150, k, metric: Metric::L2, first: true, cfg: select, live: Live::default() };
        push(&format!("nearest select {name}"), nearest_select(&b), select.lowering(&planner.device));
    }
    // Rows and lists bound at launch: a register list and a queued one, a
    // split and its merge, a score matrix's panel past its first row and its
    // listing into empty lists (carried lists hold an order the scratch
    // inputs would not).
    let var = |name: &str, max: i64| Var { name: name.into(), min: 1, max };
    let live = |from: usize, k: i64| Live { rows: Some((var("n", 100), from)), k: Some(var("k", k)) };
    for (name, d, k, queued) in [("nearest live 16", 64, 16, false), ("nearest live queued 256", 256, 256, true)] {
        let p = Nearest { n: 1 << 16, m: 1 << 16, d, metric: Metric::SquaredL2, floors: both, groups: true, k };
        let way = |c: &&NearestCfg| c.queue.is_some() == queued && c.select.is_none();
        let Some(&cfg) = planner.nearest_ranked(p).iter().find(way) else { continue };
        let (metric, floors, live) = (Metric::SquaredL2, both, live(0, k as i64));
        let spec = NearestSpec { n: 100, m: 150, d, metric, floors, groups: true, k, cfg, live };
        push(name, nearest::<F16>(&spec), cfg.lowering(&planner.device));
    }
    let cfg = NearestCfg { splits: 2, ..NearestCfg::streamed(32, 32, 32, 2) };
    let (metric, floors) = (Metric::NegDot, Floors::default());
    let spec = NearestSpec { n: 100, m: 90, d: 64, metric, floors, groups: false, k: 8, cfg, live: live(0, 8) };
    push("nearest live split", nearest::<BF16>(&spec), cfg.lowering(&planner.device));
    let merge = MergeSpec { n: 100, splits: 3, metric, k: 8, live: live(0, 8) };
    push("nearest live merge", nearest_merge(&merge), merge.lowering(&planner.device));
    let cfg = NearestCfg { select: Some(select), ..NearestCfg::new(64, 32, 2) };
    let (metric, live) = (Metric::L2, live(30, 100));
    let spec =
        NearestSpec { n: 64, m: 150, d: 64, metric, floors: both, groups: true, k: 100, cfg, live: live.clone() };
    push("nearest live matrix panel", nearest::<BF16>(&spec), cfg.lowering(&planner.device));
    let b = SelectSpec { n: 64, cols: 150, k: 100, metric, first: true, cfg: select, live };
    push("nearest live select", nearest_select(&b), select.lowering(&planner.device));

    let silu_bias = Epilogue { bias: true, act: Act::Silu, ..Epilogue::DEFAULT };
    for (name, g, images) in [
        ("conv 3x3", geom([12, 12], 64, 64, 3, 1, 1, 1), 2),
        ("conv 3x3 stride 2 split", geom([16, 16], 128, 64, 3, 2, 1, 1), 1),
    ] {
        let [ho, wo] = g.out_hw();
        let cfgs = conv_first_eight(&planner, 1, images * ho * wo, &g);
        let c = cfgs.iter().find(|c| (c.split > 1) == name.ends_with("split")).unwrap_or(&cfgs[0]);
        let spec = ConvSpec { batch: Batch::Static(images), geom: g, epilogue: silu_bias, cfg: c.gemm, split: c.split };
        for (i, (prog, lowering)) in spec.programs::<F16>(&planner.device).into_iter().enumerate() {
            push(&format!("{name} #{i}"), prog, lowering);
        }
    }
    out
}

/// An H100 SXM: sm_90 with 132 SMs.
pub fn hopper() -> Target {
    Target { sms: Some(132), ..Target::for_arch(GpuArch::Cuda(CudaArch { major: 9, minor: 0 })) }
}

/// GEMM tiles on Hopper's warpgroup core: one to four warpgroups down the
/// rows, one or two across, every swizzle span (16-, 32- and 64-deep rings
/// are 32-, 64- and 128-byte rows), a 96-wide tile of three `n32` and a
/// `k` of several trips. Shapes stay small for the interpreter.
pub fn group_families(target: &Target) -> Vec<Launch> {
    let planner = planner(target.clone());
    let tiles: [([usize; 3], [u32; 2], usize); 7] = [
        ([128, 128, 64], [8, 1], 3),
        ([128, 256, 64], [8, 1], 2),
        ([64, 128, 32], [4, 1], 3),
        ([64, 64, 16], [4, 1], 3),
        ([128, 96, 64], [8, 1], 2),
        ([256, 128, 64], [16, 1], 2),
        ([128, 256, 64], [8, 2], 2),
    ];
    tiles
        .into_iter()
        .map(|(tile, warps, stages)| {
            let cfg = GemmCfg { tile, stages, warps, group_m: 8, unroll: false, core: Core::Group, crews: 1 };
            assert!(cfg.fits(target), "{cfg:?} fits the warpgroup core");
            let (m, n, k) = (tile[0] + 2, tile[1] - 3, 3 * tile[2]);
            let spec = GemmSpec { m, n, k, batch: Batch::Static(1), epilogue: Epilogue::DEFAULT, cfg };
            (format!("group gemm {:?} {:?}", cfg.tile, cfg.warps), gemm::<BF16>(&spec), cfg.lowering(&planner.device))
        })
        .collect()
}

/// An AMD GPU with `cus` compute units and 64 KB of LDS per workgroup; a
/// CDNA3 one is an MI300X of eight dies.
pub fn amd(arch: AmdArch, cus: u32) -> Target {
    let xcds = if arch.is_cdna() { 8 } else { 1 };
    Target { sms: Some(cus), xcds, ..Target::for_arch(GpuArch::Amd(arch)) }
}

/// A square-kernel convolution of `cin` to `cout` channels over `hw`, its
/// stride, padding and dilation the same along both axes.
pub fn geom(hw: [usize; 2], cin: usize, cout: usize, k: usize, s: usize, p: usize, d: usize) -> ConvGeom {
    ConvGeom { h: hw[0], w: hw[1], cin, cout, kernel: [k, k], stride: [s, s], pad: [p, p], dilation: [d, d] }
}

/// `(cin, cout, k, [h, w], stride)` of the YOLO26x dense 3×3 classes at
/// 640² (the memo's census), padded `k / 2`.
pub const YOLO: [(usize, usize, usize, [usize; 2], usize); 9] = [
    (192, 192, 3, [40, 40], 1),
    (96, 96, 3, [80, 80], 1),
    (384, 384, 3, [160, 160], 2),
    (768, 768, 3, [80, 80], 2),
    (96, 192, 3, [320, 320], 2),
    (48, 48, 3, [160, 160], 1),
    (768, 768, 3, [40, 40], 2),
    (384, 96, 3, [80, 80], 1),
    (768, 96, 3, [20, 20], 1),
];

pub fn yolo_geom((cin, cout, k, hw, s): (usize, usize, usize, [usize; 2], usize)) -> ConvGeom {
    geom(hw, cin, cout, k, s, k / 2, 1)
}

pub(crate) fn device(target: &Target) -> DeviceSpec {
    match target.arch {
        GpuArch::Cuda(_) => DeviceSpec::Cuda { device_id: 0 },
        GpuArch::Amd(_) => DeviceSpec::Amd { device_id: 0 },
        GpuArch::Metal(_) => DeviceSpec::Metal { device_id: 0 },
    }
}

/// The lowered instruction list of `prog`.
pub fn linear(prog: Program, lowering: &Lowering) -> Arc<UOp> {
    let params =
        prog.params.iter().enumerate().map(|(i, p)| UOp::param(i, p.elems, DType::Scalar(p.dtype), None)).collect();
    let name = prog.name.clone();
    let lowered =
        lower(prog, lowering, params, device(&lowering.device.target)).unwrap_or_else(|e| panic!("{name}: {e}"));
    let Op::Program(ops::Program { linear: Some(linear), .. }) = lowered.program.op() else { panic!("a program") };
    linear.clone()
}

/// LLVM text of `prog` for its lowering's target.
pub fn render(prog: Program, lowering: &Lowering) -> String {
    let name = prog.name.clone();
    try_render(prog, lowering).unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// [`render`], a renderer's error returned.
pub fn try_render(prog: Program, lowering: &Lowering) -> Result<String, svod_codegen::Error> {
    let name = prog.name.clone();
    let renderer = match lowering.device.target.arch {
        GpuArch::Cuda(arch) => LlvmTextRenderer::nvptx(arch),
        GpuArch::Amd(arch) => LlvmTextRenderer::amd(arch),
        GpuArch::Metal(_) => unreachable!("Metal renders C"),
    };
    let linear = linear(prog, lowering);
    Ok(renderer.render(&linear, Some(&name))?.code)
}

/// The targets of the refactor sweep, by the names the gate files them
/// under: one of each CUDA generation and AMD family tk has kernels for.
pub fn targets() -> Vec<(&'static str, Target)> {
    let cuda = |major, minor, sms| Target::offline(GpuArch::Cuda(CudaArch { major, minor }), sms, 1);
    vec![
        ("sm_86", Target::card("rtx3060").expect("a listed card")),
        ("sm_89", cuda(8, 9, 76)),
        ("sm_90", hopper()),
        ("sm_100", cuda(10, 0, 148)),
        ("sm_120", cuda(12, 0, 84)),
        ("gfx1100", amd(AmdArch::Gfx1100, 96)),
        ("gfx1151", amd(AmdArch::Gfx1151, 40)),
        ("gfx1201", amd(AmdArch::Gfx1201, 64)),
        ("gfx942", amd(AmdArch::Gfx942, 304)),
        ("gfx950", amd(AmdArch::Gfx950, 256)),
    ]
}

/// Attention masks and caches beyond [`families`]': segments, windows,
/// chunks, key masks, biases, odd head sizes, and decode and prefill steps
/// over a cache with and without a row map and appended rows. Each is
/// `(name, t, tk, d, mask, cache, batch)`.
#[allow(clippy::type_complexity)]
fn attention_variants() -> Vec<(String, usize, usize, usize, AttnMask, Option<Cache>, Batch)> {
    let var = |max| Batch::Var { name: "b".into(), min: 1, max };
    let m = AttnMask::default();
    let lens = AttnMask { key_lens: true, ..m };
    let chunk = AttnMask { chunk: Some(Chunk { size: 32, left: Some(1), right: 0 }), ..lens };
    let mut cases = vec![
        ("seg".into(), 100, 100, 64, AttnMask { seg_start: true, ..m }, None, var(4)),
        ("seg_causal_lens".into(), 100, 100, 64, AttnMask { seg_start: true, causal: true, ..lens }, None, var(4)),
        ("seg_d128".into(), 100, 100, 128, AttnMask { seg_start: true, ..m }, None, var(4)),
        ("window_lens".into(), 100, 100, 64, AttnMask { window: Some((32, 0)), ..lens }, None, var(4)),
        ("chunk_lens".into(), 100, 100, 64, chunk, None, var(4)),
        ("keymask_lens_seg".into(), 37, 75, 64, AttnMask { key_mask: true, seg_start: true, ..lens }, None, var(2)),
        ("bias_lens".into(), 37, 75, 64, AttnMask { bias: Some(Bias::PerBatch), ..lens }, None, Batch::Static(2)),
        ("plain_d32".into(), 100, 100, 32, m, None, Batch::Static(1)),
        ("causal_d48".into(), 16, 100, 48, AttnMask { causal: true, ..m }, None, Batch::Static(1)),
    ];
    for row_map in [false, true] {
        for (appended, app_lens) in [(0, false), (4, true)] {
            let cache = Cache { rows: 6, heads_total: 6, row_map, appended, app_lens };
            let tag = format!("rm{}_app{appended}", row_map as u8);
            cases.push((format!("decode_{tag}"), 1, 150, 64, lens, Some(cache), var(3)));
            cases.push((format!("prefill_{tag}"), 8, 150, 64, lens, Some(cache), var(3)));
        }
    }
    cases
}

/// Every kernel the refactor gate renders on `target`: [`families`] (and
/// [`group_families`] on a warpgroup core), the planner's GEMM shortlists
/// on four shapes plain and gated, the first eight attention configs per
/// head size and query count, the [attention variants](attention_variants)
/// with their key splits (and warpgroup tiles on a warpgroup core), the
/// YOLO convolution candidates with their merges, every norm and heads
/// candidate, and the device probes' programs. Shapes stay small; the
/// tiles are those planned for the large shapes.
pub fn digested(target: &Target) -> Vec<Launch> {
    let planner = planner(target.clone());
    let device = planner.device.clone();
    let group = !target.group_mma.is_empty();
    let mut out = families(target);
    if group {
        out.extend(group_families(target));
    }
    for (m, n, k) in [(4096, 4096, 4096), (704, 512, 512), (1500, 1280, 5120), (37, 96, 48)] {
        for gated in [false, true] {
            let ranked = planner.gemm_ranked(1, m, n, k, gated);
            let deep = if m == 4096 { config::SHORTLIST } else { 8 };
            for cfg in config::distinct(&ranked).take(deep) {
                let epilogue = Epilogue { gated, ..Epilogue::DEFAULT };
                let spec = GemmSpec { m: 100, n: 96, k: cfg.tile[2] * 2, batch: Batch::Static(1), epilogue, cfg };
                out.push((format!("gemm {m}x{n}x{k} g{gated} {cfg:?}"), gemm::<BF16>(&spec), cfg.lowering(&device)));
            }
        }
    }
    for d in [32, 48, 64, 128] {
        for t in [1, 16, 100] {
            for cfg in first_eight(&planner.attention_ranked(2, t, 100, d, t > 1, false)) {
                let mask = AttnMask { causal: t > 1, ..AttnMask::default() };
                let spec = AttnSpec {
                    batch: Batch::Static(1),
                    t,
                    tk: 100,
                    heads: 2,
                    kv_heads: 1,
                    d,
                    mask,
                    cache: None,
                    scale: 0.125,
                    cfg,
                };
                out.push((format!("fa d{d} t{t} {cfg:?}"), attention::<BF16>(&spec), cfg.lowering(&device)));
            }
        }
    }
    for (name, t, tk, d, mask, cache, batch) in attention_variants() {
        let ranked = planner.attention_ranked(8, t, tk, d, mask.causal, mask.bias.is_some());
        let mut cfgs: Vec<FaCfg> = vec![];
        for cfg in first_eight(&ranked).into_iter().take(4) {
            for splits in if cache.is_some() { vec![1, 2] } else { vec![1] } {
                cfgs.push(FaCfg { splits, ..cfg });
            }
        }
        if cache.is_some() {
            cfgs.push(FaCfg { key_warps: 4, splits: 2, ..FaCfg::new(16, 16, 2) });
        }
        if group {
            for (bq, bkv) in [(64, 64), (128, 64), (64, 128)] {
                let cfg = FaCfg { core: Core::Group, ..FaCfg::new(bq, bkv, 2) };
                if cfg.fits(target, d, mask.bias.is_some()) && cache.is_none_or(|c| c.appended == 0) {
                    cfgs.push(cfg);
                }
            }
        }
        for cfg in cfgs {
            let spec =
                AttnSpec { batch: batch.clone(), t, tk, heads: 4, kv_heads: 2, d, mask, cache, scale: 0.125, cfg };
            out.push((format!("fa {name} {cfg:?}"), attention::<F16>(&spec), cfg.lowering(&device)));
        }
    }
    for class in YOLO {
        let g = yolo_geom(class);
        let [ho, wo] = g.out_hw();
        for c in conv_first_eight(&planner, 1, ho * wo, &g) {
            let spec =
                ConvSpec { batch: Batch::Static(1), geom: g, epilogue: Epilogue::DEFAULT, cfg: c.gemm, split: c.split };
            for (i, (prog, lowering)) in spec.programs::<F16>(&device).into_iter().enumerate() {
                out.push((format!("conv {g:?} {c:?} #{i}"), prog, lowering));
            }
        }
    }
    for d in [512, 1024, 2048] {
        for cfg in config::norm_candidates(d) {
            let spec = NormSpec {
                norm: Norm::Rms,
                rows: 37,
                d,
                batch: Batch::Static(1),
                eps: 1e-5,
                residual: true,
                bias: false,
                cfg,
            };
            out.push((format!("rms d{d} {cfg:?}"), norm::<BF16>(&spec), cfg.lowering(&device)));
        }
    }
    for d in [64, 128] {
        for cfg in config::heads_candidates(d) {
            let spec = HeadsSpec {
                batch: Batch::Static(2),
                t: 20,
                heads: 4,
                kv_heads: 2,
                d,
                q_norm: true,
                k_norm: true,
                eps: 1e-6,
                rope: Some(Rope { per_batch: false }),
                cfg,
            };
            out.push((format!("heads d{d} {cfg:?}"), heads::<BF16>(&spec), cfg.lowering(&device)));
        }
    }
    let probes = crate::probe::probes(target).expect("the probes build");
    let cached = crate::probe::cache_probes(target).expect("the cache probes build").into_iter().map(|(_, p)| p);
    for probe in probes.into_iter().chain(cached) {
        for (i, candidate) in probe.candidates.iter().enumerate() {
            for (j, (prog, lowering)) in candidate.iter().enumerate() {
                out.push((format!("{} {i} {j}", probe.op), prog.clone(), lowering.clone()));
            }
        }
    }
    unique(out)
}

/// `launches` with every repeated name numbered (`name #2`, …) after its first.
pub fn unique(launches: Vec<Launch>) -> Vec<Launch> {
    let mut seen = std::collections::HashMap::<String, usize>::new();
    launches
        .into_iter()
        .map(|(name, prog, lowering)| {
            let n = seen.entry(name.clone()).or_default();
            *n += 1;
            let name = if *n == 1 { name } else { format!("{name} #{n}") };
            (name, prog, lowering)
        })
        .collect()
}

/// The rankings the planner makes on `target` for the shapes [`digested`]
/// draws from and a few more: GEMMs plain and gated, attention per head
/// size, query and key count with and without a causal mask or a bias, the
/// YOLO convolutions and their splits, nearest self-joins resident and
/// streamed. A ranking change moves no kernel's text but what the tuner
/// chooses among.
pub fn rankings(target: &Target) -> String {
    let planner = planner(target.clone());
    let mut s = String::new();
    for (m, n, k) in [(4096, 4096, 4096), (704, 512, 512), (1500, 1280, 5120), (37, 96, 48), (128, 4096, 1024)] {
        for gated in [false, true] {
            s += &format!("gemm {m}x{n}x{k} {gated}: {:?}\n", &*planner.gemm_ranked(1, m, n, k, gated));
        }
    }
    for d in [32, 64, 128] {
        for (t, tk) in [(1, 150), (16, 100), (100, 100), (2048, 2048)] {
            for (causal, bias) in [(false, false), (true, false), (false, true)] {
                let ranked = planner.attention_ranked(8, t, tk, d, causal, bias);
                s += &format!("fa d{d} {t} {tk} {causal} {bias}: {:?}\n", &*ranked);
            }
        }
    }
    for class in YOLO {
        let g = yolo_geom(class);
        let [ho, wo] = g.out_hw();
        let ranked = planner.conv_ranked(1, ho * wo, &g);
        s += &format!("conv {g:?}: {:?}\n", &*ranked);
        s += &format!("splits: {:?}\n", planner.conv_splits(1, ho * wo, &g, &ranked[..ranked.len().min(8)]));
    }
    let self_join =
        |d, metric, k| Nearest { n: 1 << 16, m: 1 << 16, d, metric, floors: Floors::default(), groups: false, k };
    for d in [16, 64, 256, 512, 1024] {
        for metric in [Metric::SquaredL2, Metric::NegDot] {
            let p = self_join(d, metric, 1);
            let ranked = planner.nearest_ranked(p);
            s += &format!("nearest d{d} {metric:?}: {:?}\n", &*ranked);
            let split = planner.nearest_splits(&p, &first_eight(&ranked));
            s += &format!("splits: {split:?}\n");
        }
    }
    for (d, k) in [(128, 4), (256, 16), (768, 8), (768, 32)] {
        let p = self_join(d, Metric::SquaredL2, k);
        let ranked = planner.nearest_ranked(p);
        s += &format!("nearest d{d} k{k}: {:?}\n", &*ranked);
        s += &format!("splits: {:?}\n", planner.nearest_splits(&p, &first_eight(&ranked)));
    }
    s
}

/// The 64-bit FNV-1a digest of `text`: stable across builds, platforms and
/// toolchains, unlike `std`'s hasher.
pub fn digest(text: &[u8]) -> u64 {
    text.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3))
}

/// Two digest indexes (`name`, digest) of one target compared by name.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Diff {
    /// In both, digests differing.
    pub changed: Vec<String>,
    /// Only in the first.
    pub removed: Vec<String>,
    /// Only in the second.
    pub added: Vec<String>,
}

impl Diff {
    pub fn is_empty(&self) -> bool {
        self.changed.is_empty() && self.removed.is_empty() && self.added.is_empty()
    }
}

/// What differs from `a` to `b`, each list in `a`'s (then `b`'s) order.
pub fn diff(a: &[(String, u64)], b: &[(String, u64)]) -> Diff {
    let (in_a, in_b): (std::collections::HashMap<_, _>, std::collections::HashMap<_, _>) =
        (a.iter().map(|(n, d)| (n, d)).collect(), b.iter().map(|(n, d)| (n, d)).collect());
    Diff {
        changed: a.iter().filter(|(n, d)| in_b.get(n).is_some_and(|e| *e != d)).map(|(n, _)| n.clone()).collect(),
        removed: a.iter().filter(|(n, _)| !in_b.contains_key(n)).map(|(n, _)| n.clone()).collect(),
        added: b.iter().filter(|(n, _)| !in_a.contains_key(n)).map(|(n, _)| n.clone()).collect(),
    }
}
