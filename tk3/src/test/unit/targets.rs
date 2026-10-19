//! Every kernel family lowered for targets this machine cannot run: the
//! programs the op layer would launch, rendered to LLVM text and, where the
//! toolchain is installed, compiled to the target's code (PTX assembled by
//! `ptxas`, AMDGPU code objects by clang). A missing tool skips, never fails.

use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::Arc;

use proptest::prelude::*;
use svod_codegen::Renderer;
use svod_codegen::llvm::LlvmTextRenderer;
use svod_dtype::default_device::default_device;
use svod_dtype::{AmdArch, CudaArch, DType, DeviceSpec, GpuArch, ScalarDType};
use svod_ir::{Op, UOp, ops};
use svod_tensor::Tensor;
use test_case::test_case;

use crate::atoms::Target;
use crate::build::{BF16, F16};
use crate::interp::{self, round_to};
use crate::ir::{Axis, ParamKind, Program};
use crate::kernels::Act;
use crate::kernels::Batch;
use crate::kernels::attention::{AttnMask, AttnSpec, Bias, Cache, CombineSpec, Forms, attention, combine};
use crate::kernels::conv::ConvSpec;
use crate::kernels::gemm::{Core, Epilogue, GemmCfg, GemmSpec, Scale, gemm};
use crate::kernels::heads::{HeadsSpec, Rope, heads};
use crate::kernels::rows::{Norm, NormCfg, NormSpec, norm};
use crate::launch::graph_launch_all;
use crate::lower::{Lowering, lower};
use crate::ops::config;
use crate::schedule::{Prefetch, Schedule};

use super::conv::geom;
use super::{conv_first_eight, declared, first_eight, planner};

/// One launch of a family: its name, program and lowering.
pub(super) type Launch = (String, Program, Lowering);

/// The first candidate the op layer plans for each kernel family on `target`.
pub(super) fn families(target: &Target) -> Vec<Launch> {
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

fn device(target: &Target) -> DeviceSpec {
    match target.arch {
        GpuArch::Cuda(_) => DeviceSpec::Cuda { device_id: 0 },
        GpuArch::Amd(_) => DeviceSpec::Amd { device_id: 0 },
        GpuArch::Metal(_) => DeviceSpec::Metal { device_id: 0 },
    }
}

/// The lowered instruction list of `prog`.
pub(super) fn linear(prog: Program, lowering: &Lowering) -> Arc<UOp> {
    let params =
        prog.params.iter().enumerate().map(|(i, p)| UOp::param(i, p.elems, DType::Scalar(p.dtype), None)).collect();
    let name = prog.name.clone();
    let lowered =
        lower(prog, lowering, params, device(&lowering.device.target)).unwrap_or_else(|e| panic!("{name}: {e}"));
    let Op::Program(ops::Program { linear: Some(linear), .. }) = lowered.program.op() else { panic!("a program") };
    linear.clone()
}

/// Random parameters of `prog` rounded to their types; every i32 parameter
/// (key counts, masks, segment starts, row maps) is 1.
fn inputs(name: &str, prog: &Program) -> Vec<Vec<f64>> {
    let mut seed = name.len() as u64;
    let params = prog.params.iter().map(|p| match p.dtype {
        ScalarDType::Int32 => vec![1.0; p.elems],
        dtype => (0..p.elems)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                round_to(dtype, ((seed >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0)
            })
            .collect(),
    });
    params.collect()
}

/// `prog` lowered for its lowering's target, parameters declared flat.
fn lowered(prog: Program, lowering: &Lowering) -> crate::lower::Result<crate::lower::Lowered> {
    let params =
        prog.params.iter().enumerate().map(|(i, p)| UOp::param(i, p.elems, DType::Scalar(p.dtype), None)).collect();
    lower(prog, lowering, params, device(&lowering.device.target))
}

/// The tile program after `lowering` for interpretation.
fn lowered_tile(prog: Program, lowering: &Lowering) -> Program {
    lowered(prog, lowering).unwrap().tile
}

/// A lowering takes a program whose warps its grid takes all, or one whose
/// warps run on their own ([`Stmt::Warps`]) in whole grids; any other
/// pairing is refused before a layout is inferred.
#[test_case(4, [1, 1], false => matches Err(crate::lower::Error::Grid { warps: 4, grid: 1 }); "a norm's four warps on a lone grid")]
#[test_case(4, [2, 1], false => matches Err(crate::lower::Error::Grid { warps: 4, grid: 2 }); "a norm's four warps on half a grid")]
#[test_case(4, [4, 1], false => matches Ok(_); "a norm's four warps on their grid")]
#[test_case(4, [1, 1], true => matches Ok(_); "four key warps on a lone grid")]
#[test_case(4, [3, 1], true => matches Err(crate::lower::Error::Grid { warps: 4, grid: 3 }); "four key warps on a grid of three")]
fn a_grid_takes_every_warp_or_whole_grids_run_alone(
    warps: u32,
    grid: [u32; 2],
    key_warps: bool,
) -> crate::lower::Result<()> {
    let device = super::declared(crate::atoms::sm86());
    let lowering = Lowering::uniform(&device, false, crate::layouts::WarpGrid { rows: grid[0], cols: grid[1] });
    let batch = Batch::Static(1);
    let prog = if key_warps {
        let cfg = crate::kernels::attention::FaCfg {
            key_warps: warps as usize,
            ..crate::kernels::attention::FaCfg::new(16, 16, 2)
        };
        let mask = AttnMask::default();
        attention::<F16>(&AttnSpec {
            batch,
            t: 1,
            tk: 64,
            heads: 1,
            kv_heads: 1,
            d: 64,
            mask,
            cache: None,
            scale: 0.125,
            cfg,
        })
    } else {
        combine::<F16>(&CombineSpec { batch, t: 1, heads: 4, d: 64, splits: 2, cfg: NormCfg { br: warps as usize } })
    };
    lowered(prog, &lowering).map(drop)
}

/// LLVM text of `prog` for its lowering's target.
pub(super) fn render(prog: Program, lowering: &Lowering) -> String {
    let name = prog.name.clone();
    let renderer = match lowering.device.target.arch {
        GpuArch::Cuda(arch) => LlvmTextRenderer::nvptx(arch),
        GpuArch::Amd(arch) => LlvmTextRenderer::amd(arch),
        GpuArch::Metal(_) => unreachable!("Metal renders C"),
    };
    let linear = linear(prog, lowering);
    renderer.render(&linear, Some(&name)).unwrap_or_else(|e| panic!("{name}: {e}")).code
}

fn run(mut command: Command, input: &[u8], what: &str) -> Result<Vec<u8>, String> {
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("{what}: {e}"))?;
    child.stdin.take().expect("stdin").write_all(input).map_err(|e| format!("{what}: {e}"))?;
    let out = child.wait_with_output().map_err(|e| format!("{what}: {e}"))?;
    if out.status.success() {
        Ok(out.stdout)
    } else {
        Err(format!("{what} failed: {}", String::from_utf8_lossy(&out.stderr)))
    }
}

fn ptxas() -> Option<std::path::PathBuf> {
    let on_path = std::env::var_os("PATH")
        .into_iter()
        .flat_map(|p| std::env::split_paths(&p).collect::<Vec<_>>())
        .map(|dir| dir.join("ptxas"));
    let toolkits = ["/opt/cuda/bin/ptxas", "/usr/local/cuda/bin/ptxas"].map(std::path::PathBuf::from);
    on_path.chain(toolkits).find(|p| p.is_file())
}

pub(super) fn hopper() -> Target {
    Target { sms: Some(132), ..Target::for_arch(GpuArch::Cuda(CudaArch { major: 9, minor: 0 })) }
}

/// Every family lowers for Hopper, clang emits its PTX for `sm_90a` (the
/// warpgroup products need the arch-specific feature set, so every kernel
/// targets it) and `ptxas` assembles it.
#[test]
fn every_family_assembles_for_hopper() {
    let target = hopper();
    assert!(config::has_kernels(&target));
    assert_eq!(target.smem_bytes, 227 << 10);
    let Some(ptxas) = ptxas() else {
        for (_, prog, lowering) in families(&target) {
            render(prog, &lowering);
        }
        eprintln!("skipped: no ptxas; rendered only");
        return;
    };
    let GpuArch::Cuda(arch) = target.arch else { unreachable!() };
    for (name, prog, lowering) in families(&target) {
        let ir = render(prog, &lowering);
        let ptx = match svod_runtime::cuda::compile_ir_to_ptx(&ir, arch) {
            Ok(ptx) => ptx,
            Err(e) if e.to_string().contains("NVPTX target") => {
                eprintln!("skipped: clang has no NVPTX target");
                return;
            }
            Err(e) => panic!("{name}: {e}"),
        };
        let mut command = Command::new(&ptxas);
        command.args(["-arch=sm_90a", "-o", "/dev/null", "/dev/stdin"]);
        run(command, &ptx, &format!("ptxas sm_90a {name}")).unwrap_or_else(|e| panic!("{e}"));
        eprintln!("{name}: assembled for sm_90a");
    }
}

/// GEMM tiles on Hopper's warpgroup core: one to four warpgroups down the
/// rows, one or two across, every swizzle span (16-, 32- and 64-deep rings
/// are 32-, 64- and 128-byte rows), a 96-wide tile of three `n32` and a
/// `k` of several trips. Shapes stay small for the interpreter.
pub(super) fn group_families(target: &Target) -> Vec<Launch> {
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

/// Every warpgroup GEMM lowers for sm_90a and assembles: its main loop
/// holds `wgmma.mma_async` products over descriptors and no warp-level
/// product or fragment load, since both operands stay in shared memory.
#[test]
fn every_group_gemm_assembles_for_hopper() {
    let target = hopper();
    assert_eq!(target.group_mma.len(), 2);
    let GpuArch::Cuda(arch) = target.arch else { unreachable!() };
    let ptxas = ptxas();
    for (name, prog, lowering) in group_families(&target) {
        let ir = render(prog, &lowering);
        assert!(ir.contains("wgmma.mma_async.sync.aligned.m64n"), "{name}: no warpgroup product");
        assert!(!ir.contains("@llvm.nvvm.mma.") && !ir.contains("ldmatrix"), "{name}: a warp-level product");
        assert!(ir.contains("wgmma.fence") && ir.contains("fence.proxy.async"), "{name}: unfenced chain");
        // A ring of three slots or more leaves the chain in flight: the wait
        // is the template's intrinsic, not the chain's own.
        let in_flight =
            lowering.schedule == Schedule::Uniform { prefetch: Prefetch::CpAsync, unroll: false, async_products: true };
        assert_eq!(ir.contains("@llvm.nvvm.wgmma.wait_group"), in_flight, "{name}: the template's wait");
        assert_eq!(ir.contains("wgmma.wait_group.sync.aligned 0;"), !in_flight, "{name}: the chain's own wait");
        let ptx = match svod_runtime::cuda::compile_ir_to_ptx(&ir, arch) {
            Ok(ptx) => ptx,
            Err(e) if e.to_string().contains("NVPTX target") => {
                eprintln!("skipped: clang has no NVPTX target");
                return;
            }
            Err(e) => panic!("{name}: {e}"),
        };
        let text = String::from_utf8_lossy(&ptx);
        assert!(text.contains(".target sm_90a"), "{name}: not the arch-specific target");
        let Some(ptxas) = &ptxas else { continue };
        let mut command = Command::new(ptxas);
        command.args(["-arch=sm_90a", "-o", "/dev/null", "/dev/stdin"]);
        run(command, &ptx, &format!("ptxas sm_90a {name}")).unwrap_or_else(|e| panic!("{e}"));
        eprintln!("{name}: assembled for sm_90a");
    }
}

/// The lowered warpgroup GEMM keeps the interpreter's result: the shared
/// operands it leaves in place are read as the register tiles were.
#[test]
fn group_lowering_keeps_the_interpreted_result() {
    for (name, prog, lowering) in group_families(&hopper()) {
        let params = inputs(&name, &prog);
        let want = interp::run(&prog, params.clone(), &[]).unwrap();
        let tile = lowered_tile(prog.clone(), &lowering);
        assert!(tile.walk().all(|(_, s)| !matches!(s, crate::ir::Stmt::Copy { dst, .. } if tile.value(*dst).place == crate::ir::Place::Reg && tile.value(*dst).dtype == ScalarDType::BFloat16)), "{name}: an operand was loaded into registers");
        let got = interp::run(&tile, params, &[]).unwrap();
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(g.iter().zip(w).all(|(a, b)| a == b), "{name}: parameter {i} differs after lowering");
        }
    }
}

/// The interpreter gives the same result for a family's program before and
/// after its lowering for `target` (schedule expansion, operand loads,
/// barriers, relayouts): the bookkeeping the target's template adds keeps
/// the numerics.
#[test_case(hopper(); "sm_90")]
#[test_case(amd(AmdArch::Gfx1201, 64); "gfx1201")]
#[test_case(amd(AmdArch::Gfx1151, 40); "gfx1151")]
#[test_case(amd(AmdArch::Gfx1100, 96); "gfx1100")]
#[test_case(amd(AmdArch::Gfx942, 304); "gfx942")]
fn lowering_keeps_the_interpreted_result(target: Target) {
    for (name, prog, lowering) in families(&target) {
        let params = inputs(&name, &prog);
        let vars: Vec<(&str, i64)> = prog.vars.iter().map(|v| (v.name.as_str(), v.max)).collect();
        let want = interp::run(&prog, params.clone(), &vars).unwrap();
        let tile = lowered_tile(prog.clone(), &lowering);
        let got = interp::run(&tile, params, &vars).unwrap();
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            let same = g.iter().zip(w).all(|(a, b)| a == b || (a.is_nan() && b.is_nan()));
            assert!(same, "{name}: parameter {i} differs after lowering");
        }
    }
}

/// An AMD GPU with `cus` compute units and 64 KB of LDS per workgroup; a
/// CDNA3 one is an MI300X of eight dies.
pub(super) fn amd(arch: AmdArch, cus: u32) -> Target {
    let xcds = if arch.is_cdna() { 8 } else { 1 };
    Target { sms: Some(cus), xcds, ..Target::for_arch(GpuArch::Amd(arch)) }
}

proptest! {
    /// Over any dispatch and die count the die order is a permutation of the
    /// blocks on which each die's blocks (every `dies`th hardware position)
    /// form one contiguous run, the runs laid out die after die.
    #[test]
    fn the_die_order_lays_each_dies_blocks_end_to_end(total in 1u64..600, dies in 1u64..12) {
        let order: Vec<u64> = (0..total).map(|l| crate::lower::emit::die_order(l, total, dies)).collect();
        let mut sorted = order.clone();
        sorted.sort_unstable();
        prop_assert_eq!(sorted, (0..total).collect::<Vec<_>>());
        let mut next = 0;
        for die in 0..dies {
            let on_die: Vec<u64> = (die..total).step_by(dies as usize).map(|l| order[l as usize]).collect();
            prop_assert_eq!(on_die.clone(), (next..next + on_die.len() as u64).collect::<Vec<_>>());
            next += on_die.len() as u64;
        }
        prop_assert_eq!(next, total);
    }
}

/// A one-die target lowers the block index as the hardware's; a many-die
/// one derives every axis from the renumbered linear position, a longer
/// program.
#[test]
fn the_block_index_is_renumbered_only_over_several_dies() {
    let cfg = config::gemm_cfg([64, 64, 32], 2, [2, 2], false);
    let spec = GemmSpec { m: 192, n: 320, k: 64, batch: Batch::Static(3), epilogue: Epilogue::default(), cfg };
    let text = |xcds: u32| {
        let target = Target { xcds, ..amd(AmdArch::Gfx942, 304) };
        let device = crate::atoms::Device { target, costs: super::costs() };
        render(gemm::<BF16>(&spec), &cfg.lowering(&device))
    };
    let (one, eight) = (text(1), text(8));
    assert_ne!(one, eight);
    assert!(eight.len() > one.len());
    assert_eq!(one, text(1), "a one-die program is deterministic");
}

/// RDNA WMMA and CDNA MFMA with register-staged fills: every family lowers
/// for the target and, where clang has the AMDGPU backend, compiles to a
/// code object holding the kernel's descriptor.
#[test_case(amd(AmdArch::Gfx1201, 64); "gfx1201")]
#[test_case(amd(AmdArch::Gfx1151, 40); "gfx1151")]
#[test_case(amd(AmdArch::Gfx1100, 96); "gfx1100")]
#[test_case(amd(AmdArch::Gfx942, 304); "gfx942")]
fn every_family_compiles_for_amd(target: Target) {
    assert!(config::has_kernels(&target));
    let GpuArch::Amd(arch) = target.arch else { unreachable!() };
    for (name, prog, lowering) in families(&target) {
        let kernel = prog.name.clone();
        let ir = render(prog, &lowering);
        let object = match svod_runtime::amd::compile_ir_to_amd_object(&ir, arch) {
            Ok(object) => object,
            Err(e) if e.to_string().contains("AMDGPU target") || e.to_string().contains("too old") => {
                eprintln!("skipped: {e}");
                return;
            }
            Err(e) => panic!("{name}: {e}"),
        };
        assert_eq!(&object[..4], b"\x7fELF", "{name}: an ELF code object");
        let descriptor = format!("{kernel}.kd");
        assert!(object.windows(descriptor.len()).any(|w| w == descriptor.as_bytes()), "{name}: {descriptor}");
        eprintln!("{name}: compiled for {}", arch.mcpu());
    }
}

/// An `if`'s compare is listed with its scalar: a predicate first met inside
/// another branch still guards a later `if` on it outside that branch. With
/// the compare left to its first user it sat inside the first branch, and
/// clang refused the kernel ("Instruction does not dominate all uses"); seen
/// on gfx1201 with a warp guard nested in a crew guard.
#[test_case(amd(AmdArch::Gfx1201, 64); "gfx1201")]
#[test_case(amd(AmdArch::Gfx1151, 40); "gfx1151")]
fn a_predicate_met_inside_a_branch_guards_a_later_if(target: Target) {
    use crate::build::{I32, Kernel, Sc};
    use crate::ir::{Const, Shape};
    use crate::layouts::WarpGrid;
    let GpuArch::Amd(arch) = target.arch else { unreachable!() };
    let mut k = Kernel::new("nested_if");
    k.grid([Sc::from(4), Sc::from(1), Sc::from(1)]);
    k.warps(2);
    let tile = Shape::new(16, 16);
    let y = k.param::<I32>("y", ParamKind::Out, 4 * 2 * tile.elems());
    let slot = k.block(0) * 2 + k.warp();
    let dst = k.view(y, slot * tile.elems(), [tile.cols, 1], tile, [None, None]);
    let (outer, inner) = (k.block(0).lt(2), k.warp().eq(0));
    let ones = k.fill::<I32>(tile, Const::Int(1));
    let twos = k.fill::<I32>(tile, Const::Int(2));
    k.if_(outer, |k| k.if_(inner.clone(), |k| k.store(dst, ones), |_| {}), |_| {});
    k.if_(inner, |k| k.store(dst, twos), |_| {});
    let prog = k.finish();
    let device = crate::atoms::Device { target: target.clone(), costs: crate::atoms::Costs::default() };
    let lowering = Lowering::uniform(&device, false, WarpGrid { rows: 2, cols: 1 });
    let ir = render(prog, &lowering);
    match svod_runtime::amd::compile_ir_to_amd_object(&ir, arch) {
        Ok(object) => assert_eq!(&object[..4], b"\x7fELF"),
        Err(e) if e.to_string().contains("AMDGPU target") || e.to_string().contains("too old") => {
            eprintln!("skipped: {e}");
        }
        Err(e) => panic!("{e}"),
    }
}

/// Every access a lowering lists reads an allocation or a view of its
/// whole, never a crop at an offset: once a program's placeholders are
/// bound, an access of a crop is folded into the buffer at the summed
/// offset, and nothing lists the sum, so the LLVM renderer finds no name
/// for it. Key warps re-holding a biased score tile through their regions
/// of the relayout scratch hit this on gfx1151.
#[test_case(amd(AmdArch::Gfx1151, 40); "gfx1151")]
#[test_case(amd(AmdArch::Gfx1201, 64); "gfx1201")]
#[test_case(crate::atoms::sm86(); "sm86")]
fn accesses_address_allocations_not_crops(target: Target) {
    let zero = |u: &Arc<UOp>| matches!(u.op(), Op::Const(v) if v.0.try_int() == Some(0));
    for (name, prog, lowering) in families(&target) {
        let linear = linear(prog, &lowering);
        let Op::Linear(ops::Linear { ops: list }) = linear.op() else { panic!("{name}: a linear list") };
        for u in list.iter() {
            let buf = match u.op() {
                Op::Index(ops::Index { buffer, .. }) => buffer,
                Op::Shrink(ops::Shrink { src, .. }) => src,
                _ => continue,
            };
            if let Op::Shrink(ops::Shrink { offsets, .. }) = buf.op() {
                assert!(zero(offsets), "{name}: {} {} reads a crop at an offset", u.id, u.op().as_ref());
            }
        }
    }
}

/// Every output parameter of `prog` after launching it under `lowering` on
/// the default device, `None` for inputs.
fn on_device(prog: Program, lowering: &Lowering, params: &[Vec<f64>]) -> Vec<Option<Vec<f64>>> {
    let decl = prog.params.clone();
    let vars: Vec<(String, i64)> = prog.vars.iter().map(|v| (v.name.clone(), v.max)).collect();
    let vars: Vec<(&str, i64)> = vars.iter().map(|(n, v)| (n.as_str(), *v)).collect();
    let tensors: Vec<Tensor> = decl
        .iter()
        .zip(params)
        .map(|(p, v)| match p.dtype {
            ScalarDType::Int32 => Tensor::from_slice(v.iter().map(|&x| x as i32).collect::<Vec<_>>()),
            dtype => Tensor::from_slice(v.iter().map(|&x| x as f32).collect::<Vec<_>>()).cast(DType::Scalar(dtype)),
        })
        .collect();
    let refs: Vec<&Tensor> = tensors.iter().collect();
    let outs = graph_launch_all(prog, lowering, &refs).unwrap();
    decl.iter()
        .zip(outs)
        .map(|(p, t)| {
            (p.kind != ParamKind::In).then(|| {
                t.prepare().unwrap().execute_with_vars(&vars).unwrap();
                let mut bytes = vec![0u8; p.elems * p.dtype.bytes()];
                t.buffer().unwrap().copyout(&mut bytes).unwrap();
                let half = |b: &[u8]| u16::from_le_bytes([b[0], b[1]]);
                match p.dtype {
                    ScalarDType::Float32 => {
                        bytes.chunks(4).map(|b| f32::from_le_bytes(b.try_into().unwrap()) as f64).collect()
                    }
                    ScalarDType::BFloat16 => {
                        bytes.chunks(2).map(|b| f32::from_bits(u32::from(half(b)) << 16) as f64).collect()
                    }
                    ScalarDType::Float16 => bytes.chunks(2).map(|b| f16_bits(half(b))).collect(),
                    other => unreachable!("{other:?} output"),
                }
            })
        })
        .collect()
}

fn f16_bits(h: u16) -> f64 {
    let (sign, exp, frac) = (if h >> 15 == 1 { -1.0 } else { 1.0 }, (h >> 10) & 0x1f, f64::from(h & 0x3ff));
    sign * match exp {
        0 => frac * 2f64.powi(-24),
        31 if frac == 0.0 => f64::INFINITY,
        31 => f64::NAN,
        e => (1.0 + frac / 1024.0) * 2f64.powi(i32::from(e) - 15),
    }
}

/// Each family on the GPU against the interpreter: within two 16-bit ulps
/// of the output scale (accumulation order differs), NaN where it is NaN.
fn check_on_device(target: &Target) {
    for (name, prog, lowering) in families(target) {
        let params = inputs(&name, &prog);
        let vars: Vec<(&str, i64)> = prog.vars.iter().map(|v| (v.name.as_str(), v.max)).collect();
        let want = interp::run(&prog, params.clone(), &vars).unwrap();
        let got = on_device(prog, &lowering, &params);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            let Some(g) = g else { continue };
            let bad = g
                .iter()
                .zip(w)
                .position(|(g, w)| !(g.is_nan() && w.is_nan() || (g - w).abs() <= 2e-2 * w.abs().max(1.0)));
            let worst = g.iter().zip(w).map(|(g, w)| (g - w).abs()).fold(0.0, f64::max);
            eprintln!("{name}: output {i} max abs diff {worst:.3e}");
            assert!(bad.is_none(), "{name}: output {i}[{}] = {} vs {}", bad.unwrap(), g[bad.unwrap()], w[bad.unwrap()]);
        }
    }
}

/// The device's own target (any vendor with tables): every family's first
/// candidate computes what the interpreter computes. The remote gate for a
/// new GPU; skips without one.
#[test]
fn families_match_the_interpreter_on_the_device() {
    let device = default_device();
    let Some(target) = Target::for_device(&device).filter(config::has_kernels) else {
        eprintln!("skipped: no device with tk3 tables");
        return;
    };
    check_on_device(&target);
}

/// The device's occupancy facts come from the device where it reports them
/// (SIMDs per CU and waves per SIMD on AMD, the register file and waves per
/// SM on CUDA), and agree with the architecture's defaults the host tests
/// plan with: a lattice ranked on the host is the one the device runs.
#[test]
fn the_device_reports_the_architectures_occupancy() {
    let device = default_device();
    let Some(target) = Target::for_device(&device).filter(config::has_kernels) else {
        eprintln!("skipped: no device with tk3 tables");
        return;
    };
    let arch = Target::for_arch(target.arch);
    assert_eq!(target.occupancy, arch.occupancy, "{:?}", target.arch);
    assert!(target.sms.is_some_and(|n| n > 0));
}

/// The RDNA data path on an NVIDIA GPU: register-staged fills into the
/// chunk layout, vector shared-memory gathers instead of `ldmatrix`, and
/// two-slot rings, measured against the interpreter.
#[test]
fn register_staged_families_match_on_cuda() {
    let device = default_device();
    let Some(target) = matches!(device, DeviceSpec::Cuda { .. }).then(|| Target::for_device(&device)).flatten() else {
        eprintln!("skipped: no CUDA device");
        return;
    };
    check_on_device(&Target { cp_async: false, ldmatrix: false, ..target });
}

/// Every candidate the op layer may plan lowers (the tune store would
/// otherwise time a failure): the GEMM lists of a large, a small-M and a
/// ragged shape (the large one as deep as the shortlist compiles, on one
/// target of each fill path), the attention lists of every head size and
/// both query regimes, and the convolution lists of every YOLO class.
#[test_case(amd(AmdArch::Gfx1201, 64), true; "gfx1201")]
#[test_case(amd(AmdArch::Gfx1151, 40), false; "gfx1151")]
#[test_case(amd(AmdArch::Gfx942, 304), false; "gfx942")]
#[test_case(crate::atoms::sm86(), true; "sm86")]
#[test_case(hopper(), false; "hopper")]
fn every_candidate_lowers(target: Target, shortlist_deep: bool) {
    let planner = planner(target.clone());
    let mut gemms = vec![];
    for (m, n, k) in [(4096, 4096, 4096), (704, 512, 512), (1500, 1280, 5120), (37, 96, 48)] {
        for gated in [false, true] {
            let ranked = planner.gemm_ranked(1, m, n, k, gated);
            let deep = if m == 4096 && shortlist_deep { config::SHORTLIST } else { 8 };
            for cfg in config::distinct(&ranked).take(deep) {
                let epilogue = Epilogue { gated, ..Epilogue::DEFAULT };
                gemms.push(GemmSpec { m: 100, n: 96, k: cfg.tile[2] * 2, batch: Batch::Static(1), epilogue, cfg });
            }
        }
    }
    for spec in gemms {
        linear(gemm::<BF16>(&spec), &spec.cfg.lowering(&planner.device));
    }
    for d in [32, 48, 64, 128] {
        for t in [1, 16, 100] {
            for cfg in first_eight(&planner.attention_ranked(2, t, 100, d, t > 1, false)) {
                let spec = AttnSpec {
                    batch: Batch::Static(1),
                    t,
                    tk: 100,
                    heads: 2,
                    kv_heads: 1,
                    d,
                    mask: AttnMask { causal: t > 1, ..AttnMask::default() },
                    cache: None,
                    scale: 0.125,
                    cfg,
                };
                linear(attention::<BF16>(&spec), &cfg.lowering(&planner.device));
            }
        }
    }
    for class in super::conv::YOLO {
        let g = super::conv::yolo_geom(class);
        let [ho, wo] = g.out_hw();
        for c in conv_first_eight(&planner, 1, ho * wo, &g) {
            let spec =
                ConvSpec { batch: Batch::Static(1), geom: g, epilogue: Epilogue::DEFAULT, cfg: c.gemm, split: c.split };
            for (prog, lowering) in spec.programs::<F16>(&planner.device) {
                linear(prog, &lowering);
            }
        }
    }
}

/// A shared tile lies along the axis its readers' fragments run along: the
/// value tile of a register-staged attention kernel is stored column-major,
/// since the product's B fragments hold runs of keys, and gathers as 16-byte
/// loads; the key tile, whose fragments hold runs of head elements, and a
/// GEMM's operand tiles stay row-major. A `cp.async` target fills its tiles
/// in chunks, which cannot scatter, so there every tile stays row-major and
/// `ldmatrix.trans` does the transposing.
#[test_case(amd(AmdArch::Gfx1201, 64), Axis::Row; "gfx1201 stores values column-major")]
#[test_case(amd(AmdArch::Gfx942, 304), Axis::Row; "gfx942 stores values column-major")]
#[test_case(crate::atoms::sm86(), Axis::Col; "sm_86 keeps every tile row-major")]
fn shared_tiles_lie_along_their_fragments(target: Target, values: Axis) {
    let families = families(&target);
    let (_, prog, lowering) = families.iter().find(|(n, ..)| n.starts_with("attention d128")).unwrap().clone();
    let rings = |tile: Program| {
        let along = |name: &str| tile.smem.iter().find(|a| a.name == name).map(|a| a.along).unwrap();
        (along("k_s"), along("v_s"))
    };
    assert_eq!(rings(lowered_tile(prog, &lowering)), (Axis::Col, values));
    // The key warps' f32 merge views over the rings cast no vote.
    let (_, prog, lowering) = families.iter().find(|(n, ..)| n.starts_with("attention d64 cache key warps")).unwrap();
    assert_eq!(rings(lowered_tile(prog.clone(), lowering)), (Axis::Col, values));
    let (_, prog, lowering) = families.iter().find(|(n, ..)| n.starts_with("gemm #0")).unwrap().clone();
    assert!(lowered_tile(prog, &lowering).smem.iter().all(|a| a.along == Axis::Col));
}

/// Where the target loads 16×16 blocks transposed (RDNA4), the staged
/// registers of a column-major tile hold column runs straight from global
/// memory and store them as 16-byte vectors; elsewhere the staged registers
/// are walked down the rows.
#[test_case(amd(AmdArch::Gfx1201, 64), true; "gfx1201 loads values transposed")]
#[test_case(amd(AmdArch::Gfx942, 304), false; "gfx942 walks the rows")]
fn column_major_tiles_fill_by_transposing_loads(target: Target, transposing: bool) {
    assert_eq!(target.tr_load, transposing);
    let families = families(&target);
    let (name, prog, lowering) = families.iter().find(|(n, ..)| n.starts_with("attention d128")).unwrap().clone();
    let ir = render(prog, &lowering);
    let calls = ir.matches("call <8 x bfloat> @llvm.amdgcn.global.load.tr.b128.v8bf16(ptr addrspace(1) %").count();
    assert_eq!(calls > 0, transposing, "{name}: {calls} transposing loads");
    let (_, prog, lowering) = families.iter().find(|(n, ..)| n.starts_with("gemm #0")).unwrap().clone();
    assert!(!render(prog, &lowering).contains("global.load.tr"), "a row-major fill stays a chunk copy");
}

/// On AMD the softmax's `exp2` is the hardware `v_exp_f32` alone, with no
/// denormal range reduction around it; other targets keep `llvm.exp2`.
#[test_case(amd(AmdArch::Gfx1201, 64), true; "gfx1201")]
#[test_case(crate::atoms::sm86(), false; "sm_86")]
fn attention_exponentials_are_the_hardware_ones_on_amd(target: Target, raw: bool) {
    let (name, prog, lowering) =
        families(&target).into_iter().find(|(n, ..)| n.starts_with("attention d128")).expect("an attention family");
    let ir = render(prog, &lowering);
    assert_eq!(ir.contains("@llvm.amdgcn.exp2.f32("), raw, "{name}");
    assert_eq!(ir.contains("@llvm.exp2.f32("), !raw, "{name}");
}

/// The attention program carries the softmax forms its config declares: a
/// cast toward zero where bf16 probabilities are truncated (an f16 stream
/// rounds either way), a `Scale` where the rescale is skipped, and the row
/// sum after the P·V product exactly where `sum_after` puts it, whether the
/// probabilities are truncated or not, before the rescale otherwise.
#[test_case(crate::atoms::sm86(), false, (false, false, false), (false, false, false); "sm_86 plain")]
#[test_case(crate::atoms::sm86(), false, (false, true, false), (false, true, false); "sm_86 rescale skipped")]
#[test_case(crate::atoms::sm86(), false, (false, false, true), (false, false, true); "sm_86 rounded, sum after")]
#[test_case(amd(AmdArch::Gfx1201, 64), false, (true, true, true), (true, true, true); "gfx1201 truncated, sum after")]
#[test_case(amd(AmdArch::Gfx1201, 64), false, (true, true, false), (true, true, false); "gfx1201 truncated, sum before")]
#[test_case(amd(AmdArch::Gfx1201, 64), true, (true, true, false), (false, true, false); "gfx1201 f16 rounds")]
#[test_case(amd(AmdArch::Gfx1201, 64), true, (true, false, true), (false, false, true); "gfx1201 f16, sum after")]
fn attention_softmax_takes_the_declared_forms(
    target: Target,
    f16: bool,
    (truncate_p, skip_rescale, sum_after): (bool, bool, bool),
    (truncated, skipped, after): (bool, bool, bool),
) {
    use crate::ir::{BinaryOp, ReduceOp, Round, TileOp};
    let lead = planner(target.clone()).attention_ranked(8, 100, 100, 64, true, false)[0];
    let forms = Forms { truncate_p, skip_rescale, sum_after, ..lead.forms };
    let cfg = crate::kernels::attention::FaCfg { forms, ..lead };
    let mask = AttnMask { causal: true, ..AttnMask::default() };
    let spec = AttnSpec {
        batch: Batch::Static(2),
        t: 100,
        tk: 100,
        heads: 4,
        kv_heads: 2,
        d: 64,
        mask,
        cache: None,
        scale: 0.125,
        cfg,
    };
    let prog = if f16 { attention::<F16>(&spec) } else { attention::<BF16>(&spec) };
    let casts = tile_ops(&prog.body, &|op| matches!(op, TileOp::Cast { round: Round::Zero, .. }));
    let scales = tile_ops(&prog.body, &|op| matches!(op, TileOp::Binary { f: BinaryOp::Scale, .. }));
    assert_eq!((casts > 0, scales > 0), (truncated, skipped), "{casts} casts toward zero, {scales} scales");
    let ops = tile_ops_in_order(&prog.body);
    let products: Vec<usize> =
        ops.iter().enumerate().filter(|(_, op)| matches!(op, TileOp::Mma { .. })).map(|(i, _)| i).collect();
    let sum = ops.iter().position(|op| matches!(op, TileOp::Reduce { f: ReduceOp::Sum, .. })).expect("a row sum");
    // The first step's products: `q·kᵀ`, then `p·v`.
    assert!(products.len() >= 2);
    assert_eq!(sum > products[1], after, "sum at {sum}, products at {products:?}");
}

/// The tile ops of `block` in program order, recursively.
fn tile_ops_in_order(block: &crate::ir::Block) -> Vec<crate::ir::TileOp> {
    use crate::ir::Stmt;
    let mut out = vec![];
    for s in &block.0 {
        match s {
            Stmt::Let { op, .. } => out.push(op.clone()),
            Stmt::Copy { .. } | Stmt::Sync(_) | Stmt::Raw(_) => {}
            Stmt::Loop(l) => out.extend(tile_ops_in_order(&l.body)),
            Stmt::Pipeline(p) => {
                out.extend(tile_ops_in_order(&p.produce.body));
                out.extend(tile_ops_in_order(&p.consume.body));
            }
            Stmt::Warps { body } => out.extend(tile_ops_in_order(body)),
            Stmt::If { then, otherwise, .. } => {
                out.extend(tile_ops_in_order(then));
                out.extend(tile_ops_in_order(otherwise));
            }
        }
    }
    out
}

/// The tile ops of `block` (recursively) that `f` picks.
fn tile_ops(block: &crate::ir::Block, f: &dyn Fn(&crate::ir::TileOp) -> bool) -> usize {
    use crate::ir::Stmt;
    block
        .0
        .iter()
        .map(|s| match s {
            Stmt::Let { op, .. } => usize::from(f(op)),
            Stmt::Copy { .. } | Stmt::Sync(_) | Stmt::Raw(_) => 0,
            Stmt::Loop(l) => tile_ops(&l.body, f),
            Stmt::Pipeline(p) => tile_ops(&p.produce.body, f) + tile_ops(&p.consume.body, f),
            Stmt::Warps { body } => tile_ops(body, f),
            Stmt::If { then, otherwise, .. } => tile_ops(then, f) + tile_ops(otherwise, f),
        })
        .sum()
}

/// A bf16 tile summed into f32 lowers to the pair dot product where the
/// target has it (RDNA3 on, gfx950), and widens each element elsewhere.
#[test_case(amd(AmdArch::Gfx1201, 64), true; "gfx1201")]
#[test_case(amd(AmdArch::Gfx950, 256), true; "gfx950")]
#[test_case(amd(AmdArch::Gfx942, 304), false; "gfx942")]
#[test_case(crate::atoms::sm86(), false; "sm_86")]
fn a_bf16_sum_in_f32_takes_the_pair_dot_product_where_it_exists(target: Target, dot2: bool) {
    use crate::build::{F32, Kernel};
    use crate::ir::{ReduceOp, Shape};
    assert_eq!(target.dot2_bf16, dot2);
    let (rows, cols) = (4usize, 128usize);
    let mut k = Kernel::new("held_sum");
    k.warps(1);
    let x = k.param::<F32>("x", ParamKind::In, rows * cols);
    let y = k.param::<F32>("y", ParamKind::Out, rows);
    let xv = k.view(x, 0, [cols, 1], Shape::new(rows, cols), [None, None]);
    let xr = k.load(xv);
    let held = k.cast_toward_zero::<BF16>(xr);
    let sum = k.reduce_f32(held, Axis::Row, ReduceOp::Sum);
    let yv = k.view(y, 0, [1, 1], Shape::new(rows, 1), [None, None]);
    k.store(yv, sum);
    let lowering = Lowering {
        group: false,
        schedule: crate::schedule::Schedule::Uniform {
            prefetch: target.prefetch(),
            unroll: false,
            async_products: false,
        },
        device: declared(target),
        grid: crate::layouts::WarpGrid { rows: 1, cols: 1 },
        swizzle: false,
    };
    let ir = render(k.finish(), &lowering);
    assert_eq!(ir.contains("@llvm.amdgcn.fdot2.f32.bf16("), dot2, "{ir}");
    // A lane's odd member out is widened on every target.
    assert!(dot2 || ir.contains("fpext bfloat"), "{ir}");
}

/// A lane reads another lane's register on the vector unit on RDNA (DPP within
/// a row of 16, `permlanex16` across the rows), through the LDS crossbar on CDNA.
#[test_case(amd(AmdArch::Gfx1201, 64), true; "gfx1201")]
#[test_case(amd(AmdArch::Gfx1151, 40), true; "gfx1151")]
#[test_case(amd(AmdArch::Gfx942, 304), false; "gfx942")]
fn rdna_lane_moves_stay_on_the_vector_unit(target: Target, vector: bool) {
    let (name, prog, lowering) =
        families(&target).into_iter().find(|(n, ..)| n.starts_with("rms norm")).expect("a norm family");
    let ir = render(prog, &lowering);
    assert_eq!(ir.contains("@llvm.amdgcn.update.dpp.i32("), vector, "{name}");
    assert_eq!(ir.contains("@llvm.amdgcn.ds.bpermute("), !vector, "{name}");
}

/// `Sync::Fence` statements in `block`, recursively.
fn fences(block: &crate::ir::Block) -> usize {
    use crate::ir::{Stmt, Sync};
    block
        .0
        .iter()
        .map(|s| match s {
            Stmt::Sync(Sync::Fence) => 1,
            Stmt::Let { .. } | Stmt::Copy { .. } | Stmt::Sync(_) | Stmt::Raw(_) => 0,
            Stmt::Loop(l) => fences(&l.body),
            Stmt::Pipeline(p) => fences(&p.produce.body) + fences(&p.consume.body),
            Stmt::Warps { body } => fences(body),
            Stmt::If { then, otherwise, .. } => fences(then) + fences(otherwise),
        })
        .sum()
}

/// The two-slot register-staged template fences every trip between its
/// products and its commit on every target that runs it; the fence is an
/// instruction only where the scheduler would otherwise hoist the commit
/// (RDNA4), and the interpreter never sees it, so
/// `lowering_keeps_the_interpreted_result` covers its placement. One slot
/// needs none: a barrier holds the commit back.
#[test_case(AmdArch::Gfx1201, true; "gfx1201 fences")]
#[test_case(AmdArch::Gfx942, false; "gfx942 keeps order")]
#[test_case(AmdArch::Gfx1100, false; "gfx1100 keeps order")]
fn register_staged_trips_are_fenced(arch: AmdArch, emitted: bool) {
    let target = amd(arch, 64);
    assert_eq!(target.commit_fence, emitted);
    let device = declared(target);
    for (stages, want) in [(2, 1), (1, 0)] {
        let cfg = config::gemm_cfg([64, 64, 32], stages, [2, 2], stages == 2);
        let spec = GemmSpec { m: 130, n: 192, k: 128, batch: Batch::Static(1), epilogue: Epilogue::DEFAULT, cfg };
        let tile = lowered_tile(gemm::<BF16>(&spec), &cfg.lowering(&device));
        assert_eq!(fences(&tile.body), want, "{stages} slots: a fence per two-slot pipeline");
    }
}

/// `Relayout` statements of tiles in `block`, recursively; vectors are not
/// counted.
fn tile_relayouts(prog: &Program, block: &crate::ir::Block) -> usize {
    use crate::ir::{Stmt, TileOp};
    block
        .0
        .iter()
        .map(|s| match s {
            Stmt::Let { op: TileOp::Relayout { src }, .. } => {
                let shape = prog.value(*src).shape;
                usize::from(shape.rows > 1 && shape.cols > 1)
            }
            Stmt::Let { .. } | Stmt::Copy { .. } | Stmt::Sync(_) | Stmt::Raw(_) => 0,
            Stmt::Loop(l) => tile_relayouts(prog, &l.body),
            Stmt::Pipeline(p) => tile_relayouts(prog, &p.produce.body) + tile_relayouts(prog, &p.consume.body),
            Stmt::Warps { body } => tile_relayouts(prog, body),
            Stmt::If { then, otherwise, .. } => tile_relayouts(prog, then) + tile_relayouts(prog, otherwise),
        })
        .sum()
}

/// Where the accumulator is its operand's transpose (gfx12, CDNA) the
/// inference issues attention's products the way round that hands the scores
/// to P·V as they stand, so no attention kernel re-holds a score tile (the
/// decoder step's cache kernel still moves its softmax state, a 16-element
/// vector, between the cached keys and the appended one); where neither way
/// is free (RDNA3) the tile relayout remains, under the declared [`costs`] a
/// register exchange between the half-waves. The biased kernel is left out:
/// its bias tiles come from global memory in a layout of their own, and on
/// gfx942 the kernel re-holds three tiles however the products are issued.
#[test_case(amd(AmdArch::Gfx1201, 64), true; "gfx1201")]
#[test_case(amd(AmdArch::Gfx942, 304), true; "gfx942")]
#[test_case(amd(AmdArch::Gfx1100, 96), false; "gfx1100")]
fn attention_orientation_spares_the_relayout(target: Target, free: bool) {
    let kernels: Vec<_> = families(&target)
        .into_iter()
        .filter(|(name, ..)| name.starts_with("attention d") && !name.ends_with("bias"))
        .collect();
    assert!(!kernels.is_empty());
    for (name, prog, lowering) in kernels {
        let tile = lowered_tile(prog, &lowering);
        let n = tile_relayouts(&tile, &tile.body);
        assert_eq!(n == 0, free, "{name}: {n} tile relayouts");
    }
    // What the planner charges for the scratch agrees with the lowering.
    let atom = target.mma.first().unwrap();
    assert_eq!(atom.accumulator_feeds(target.wave), free);
}
