//! Every kernel family lowered for targets this machine cannot run: the
//! programs the op layer would launch, rendered to LLVM text and, where the
//! toolchain is installed, compiled to the target's code (PTX assembled by
//! `ptxas`, AMDGPU code objects by clang). A missing tool skips, never fails.

use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::Arc;

use proptest::prelude::*;
use svod_dtype::{AmdArch, DType, GpuArch, ScalarDType};
use svod_ir::{Op, UOp, ops};
use svod_tensor::Tensor;
use test_case::test_case;

use crate::atoms::{Core, Fill, Target};
use crate::build::{BF16, F16};
use crate::interp;
use crate::ir::{Axis, ParamKind, Program};
use crate::kernels::Batch;
use crate::kernels::attention::{AttnMask, AttnSpec, Bias, CombineSpec, Forms, attention, combine};
use crate::kernels::conv::ConvSpec;
use crate::kernels::gemm::{Epilogue, GemmSpec, gemm};
use crate::kernels::nearest::{Floors, Metric, NearestSpec, nearest};
use crate::kernels::rows::NormCfg;
use crate::launch::graph_launch_all;
use crate::lower::sync::insert_barriers;
use crate::lower::{Lowering, lower};
use crate::ops::config::{self, Nearest, Planner};
use crate::schedule::Schedule;

use super::{conv_first_eight, declared, first_eight, planner};
use crate::sweep::device;
pub(super) use crate::sweep::{Launch, amd, families, group_families, hopper, linear, render};

/// Random parameters of `prog` rounded to their types; every i32 parameter
/// (key counts, masks, segment starts, row maps) is 1, every i8 one an
/// integer of the full range.
fn inputs(name: &str, prog: &Program) -> Vec<Vec<f64>> {
    let mut seed = name.len() as u64;
    let params = prog.params.iter().map(|p| match p.dtype {
        ScalarDType::Int32 => vec![1.0; p.elems],
        ScalarDType::Int8 => super::rounded(ScalarDType::Int8, p.elems, &mut seed, |x| (127.0 * x).round()),
        dtype => super::rounded(dtype, p.elems, &mut seed, |x| x),
    });
    params.collect()
}

/// `prog` lowered for its lowering's target, parameters declared flat.
pub(super) fn lowered(prog: Program, lowering: &Lowering) -> crate::lower::Result<crate::lower::Lowered> {
    let params =
        prog.params.iter().enumerate().map(|(i, p)| UOp::param(i, p.elems, DType::Scalar(p.dtype), None)).collect();
    lower(prog, lowering, params, device(&lowering.device.target))
}

/// The tile program after `lowering` for interpretation.
fn lowered_tile(prog: Program, lowering: &Lowering) -> Program {
    lowered(prog, lowering).unwrap().tile
}

/// `check` on every launch at once, failing with every launch whose check
/// panics, so one failure does not hide another.
fn each(launches: Vec<Launch>, check: impl Fn(&str, Program, Lowering) + Sync) {
    let failed: Vec<String> = crate::tune::in_parallel(launches, |(name, prog, lowering)| {
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| check(&name, prog, lowering))).err()?;
        let why = panic.downcast_ref::<String>().map(String::as_str).or_else(|| panic.downcast_ref::<&str>().copied());
        Some(format!("{name}: {}", why.unwrap_or("a panic")))
    })
    .into_iter()
    .flatten()
    .collect();
    assert!(failed.is_empty(), "{} failed:\n{}", failed.len(), failed.join("\n"));
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

pub(super) fn run(mut command: Command, input: &[u8], what: &str) -> Result<Vec<u8>, String> {
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

/// Every warpgroup attention lowers for sm_90a and assembles: the scores
/// are a register-A chain over K-major key rows, the output's a register-A
/// chain over MN-major value rows (B transposed), and neither tile is ever
/// loaded into registers; a head of 128 spans two column panels.
#[test]
fn every_group_attention_assembles_for_hopper() {
    use crate::kernels::attention::FaCfg;
    let target = hopper();
    let GpuArch::Cuda(arch) = target.arch else { unreachable!() };
    let device = declared(target.clone());
    let ptxas = ptxas();
    let plain = AttnMask::default();
    let cases = [
        (64, 64, 64, plain),
        (128, 128, 64, AttnMask { causal: true, ..plain }),
        (128, 64, 128, plain),
        (64, 128, 128, AttnMask { bias: Some(Bias::PerBatch), key_lens: true, ..plain }),
    ];
    for (bq, bkv, d, mask) in cases {
        let cfg = FaCfg { core: Core::Group, ..FaCfg::new(bq, bkv, 2) };
        assert!(cfg.fits(&target, d, mask.bias.is_some()), "{cfg:?} fits the warpgroup core at d {d}");
        let spec = AttnSpec {
            batch: Batch::Static(2),
            t: 300,
            tk: 300,
            heads: 2,
            kv_heads: 2,
            d,
            mask,
            cache: None,
            scale: 0.125,
            cfg,
        };
        let name = format!("group attention bq {bq} bkv {bkv} d {d}");
        let ir = render(attention::<BF16>(&spec), &cfg.lowering(&device));
        assert!(ir.contains(&format!("wgmma.mma_async.sync.aligned.m64n{bkv}k16")), "{name}: no scores chain");
        assert!(ir.contains(&format!("wgmma.mma_async.sync.aligned.m64n{d}k16")), "{name}: no output chain");
        assert!(ir.contains(",p,1,1,0;") && ir.contains(",p,1,1,1;"), "{name}: a K-major and an MN-major B");
        assert!(!ir.contains("@llvm.nvvm.mma."), "{name}: a warp-level product");
        // Only the bias tile is gathered from shared memory into registers.
        assert!(mask.bias.is_some() || !ir.contains("ldmatrix"), "{name}: a key or value fragment load");
        let ptx = match svod_runtime::cuda::compile_ir_to_ptx(&ir, arch) {
            Ok(ptx) => ptx,
            Err(e) if e.to_string().contains("NVPTX target") => {
                eprintln!("skipped: clang has no NVPTX target");
                return;
            }
            Err(e) => panic!("{name}: {e}"),
        };
        let Some(ptxas) = &ptxas else { continue };
        let mut command = Command::new(ptxas);
        command.args(["-arch=sm_90a", "-o", "/dev/null", "/dev/stdin"]);
        run(command, &ptx, &format!("ptxas sm_90a {name}")).unwrap_or_else(|e| panic!("{e}"));
        eprintln!("{name}: assembled for sm_90a");
    }
}

/// What a target's families show beyond their numbers.
struct Facts {
    /// The axis a register-staged attention's value ring lies along (its key
    /// ring and every GEMM and convolution tile lie along the columns): along
    /// the rows where the product's B fragments hold runs of keys and the
    /// fill scatters through registers, row-major where a `cp.async` fill
    /// copies chunks, which cannot scatter (`ldmatrix.trans` transposes).
    values: Axis,
    /// The softmax's `exp2` is the hardware `v_exp_f32` alone, with no
    /// denormal range reduction around it, rather than `llvm.exp2`.
    raw_exp2: bool,
    /// A lane reads another lane's register on the vector unit (DPP within a
    /// row of 16, `permlanex16` across them) rather than through the LDS
    /// crossbar; `None` off AMD.
    dpp: Option<bool>,
    /// The accumulator is its operand's transpose, so the inference issues
    /// attention's products the way round that hands the scores to P·V as
    /// they stand: no tile relayout in an attention kernel without a bias
    /// (whose tiles come from global memory in a layout of their own; on
    /// gfx942 a biased kernel re-holds three tiles however the products are
    /// issued). Where neither way is free (RDNA3) the relayout remains, under
    /// the declared costs a register exchange between the half-waves. `None`
    /// off AMD.
    free_scores: Option<bool>,
}

const fn facts(values: Axis, raw_exp2: bool, dpp: Option<bool>, free_scores: Option<bool>) -> Facts {
    Facts { values, raw_exp2, dpp, free_scores }
}

/// Every family the op layer plans on `target` (and on Hopper every
/// warpgroup GEMM) lowers once and is checked whole, every failure of every
/// family collected before the test fails:
/// - the interpreter gives the same result before and after the lowering
///   (schedule expansion, operand loads, barriers, relayouts): the
///   bookkeeping the target's template adds keeps the numerics;
/// - the barrier pass is idempotent: run again on what it fenced, it adds
///   nothing;
/// - every access the lowering lists reads an allocation or a view of its
///   whole, never a crop at an offset (once a program's placeholders are
///   bound, an access of a crop is folded into the buffer at the summed
///   offset and nothing lists the sum, so the LLVM renderer finds no name
///   for it: key warps re-holding a biased score tile through their regions
///   of the relayout scratch hit this on gfx1151);
/// - the [`Facts`] of its shared tiles, exponentials, lane moves and score
///   relayouts hold; an attention's column-major value tile fills by
///   transposing loads exactly where the target has them (RDNA4), its staged
///   registers holding column runs straight from global memory, and a plain
///   GEMM's row-major fill stays a chunk copy;
/// - its LLVM text compiles where the toolchain is installed: clang to an
///   AMDGPU code object holding the kernel's descriptor, or clang to PTX
///   that `ptxas` assembles (Hopper's for `sm_90a`, the arch-specific
///   feature set its warpgroup products need). A missing tool skips.
///
/// A warpgroup GEMM's main loop holds `wgmma.mma_async` products over
/// descriptors, fenced, and no warp-level product or fragment load: both
/// operands stay in shared memory and no operand tile is loaded into
/// registers; a ring of three slots or more leaves the chain in flight, the
/// wait the template's intrinsic rather than the chain's own.
#[test_case(crate::atoms::sm86(), facts(Axis::Col, false, None, None); "sm_86")]
#[test_case(hopper(), facts(Axis::Col, false, None, None); "sm_90")]
#[test_case(amd(AmdArch::Gfx1201, 64), facts(Axis::Row, true, Some(true), Some(true)); "gfx1201")]
#[test_case(amd(AmdArch::Gfx1151, 40), facts(Axis::Row, true, Some(true), Some(false)); "gfx1151")]
#[test_case(amd(AmdArch::Gfx1100, 96), facts(Axis::Row, true, Some(true), Some(false)); "gfx1100")]
#[test_case(amd(AmdArch::Gfx942, 304), facts(Axis::Row, true, Some(false), Some(true)); "gfx942")]
fn every_family_lowers_keeps_its_numbers_and_compiles(target: Target, facts: Facts) {
    assert!(config::has_kernels(&target));
    if let Some(free) = facts.free_scores {
        let atom = target.mma.first().expect("a matrix core");
        assert_eq!(atom.accumulator_feeds(target.wave), free, "whether the scores feed the product as they stand");
    }
    let mut launches = families(&target);
    if target.arch == hopper().arch {
        assert_eq!((target.smem_bytes, target.group_mma.len()), (227 << 10, 2));
        launches.extend(group_families(&target));
    }
    let ptxas = ptxas();
    each(launches, |name, prog, lowering| {
        let mut failed: Vec<String> = vec![];
        let mut check = |ok: bool, what: &str| {
            if !ok {
                failed.push(what.to_string());
            }
        };
        let vars: Vec<(&str, i64)> = prog.vars.iter().map(|v| (v.name.as_str(), v.max)).collect();
        let params = inputs(name, &prog);
        let want = interp::run(&prog, params.clone(), &vars).unwrap();
        let low = lowered(prog.clone(), &lowering).unwrap();
        let tile = &low.tile;
        let got = interp::run(tile, params, &vars).unwrap();
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            let same = g.iter().zip(w).all(|(a, b)| a == b || (a.is_nan() && b.is_nan()));
            check(same, &format!("parameter {i} differs after lowering"));
        }

        let mut once = prog.clone();
        crate::lower::materialize_operands(&mut once, lowering.core);
        insert_barriers(&mut once).unwrap();
        let mut twice = once.clone();
        insert_barriers(&mut twice).unwrap();
        check(twice == once, "the barrier pass fences what it fenced again");

        let Op::Program(ops::Program { linear: Some(linear), .. }) = low.program.op() else { panic!("a program") };
        let Op::Linear(ops::Linear { ops: list }) = linear.op() else { panic!("a linear list") };
        let zero = |u: &Arc<UOp>| matches!(u.op(), Op::Const(v) if v.0.try_int() == Some(0));
        for u in list.iter() {
            let buf = match u.op() {
                Op::Index(ops::Index { buffer, .. }) => buffer,
                Op::Shrink(ops::Shrink { src, .. }) => src,
                _ => continue,
            };
            if let Op::Shrink(ops::Shrink { offsets, .. }) = buf.op() {
                check(zero(offsets), &format!("{} {} reads a crop at an offset", u.id, u.op().as_ref()));
            }
        }

        let along = |ring: &str| tile.smem.iter().find(|a| a.name == ring).map(|a| a.along);
        let attention = name.starts_with("attention d");
        let rings = (along("k_s"), along("v_s"));
        if attention {
            check(rings == (Some(Axis::Col), Some(facts.values)), &format!("k_s and v_s lie along {rings:?}"));
            if let (Some(free), false) = (facts.free_scores, name.ends_with("bias")) {
                let n = tile_relayouts(tile, &tile.body);
                check((n == 0) == free, &format!("{n} tile relayouts"));
            }
        } else if name.starts_with("gemm") || name.starts_with("group gemm") || name.starts_with("conv") {
            check(tile.smem.iter().all(|a| a.along == Axis::Col), "a GEMM tile stored column-major");
        }

        let ir = low.text(&target).unwrap();
        let tr_loads = ir.matches("@llvm.amdgcn.global.load.tr.b128.").count();
        let transposing = match rings.1 {
            Some(Axis::Row) => target.tr_load,
            _ => target.tr_load && !name.starts_with("gemm #"),
        };
        check(tr_loads == 0 || transposing, &format!("{tr_loads} transposing loads"));
        if attention {
            check((tr_loads > 0) == transposing, &format!("{tr_loads} transposing loads of the values"));
            let (raw, exp2) = (ir.contains("@llvm.amdgcn.exp2.f32("), ir.contains("@llvm.exp2.f32("));
            check((raw, exp2) == (facts.raw_exp2, !facts.raw_exp2), &format!("v_exp_f32 {raw}, llvm.exp2 {exp2}"));
        }
        if let (Some(dpp), true) = (facts.dpp, name.ends_with("norm")) {
            let moves = (ir.contains("@llvm.amdgcn.update.dpp.i32("), ir.contains("@llvm.amdgcn.ds.bpermute("));
            check(moves == (dpp, !dpp), &format!("DPP and ds_bpermute: {moves:?}"));
        }
        if name.starts_with("gemm int8") {
            let cores = ["mma.m16n8k32.row.col.satfinite.s8", "wmma.i32.16x16x16.iu8", "mfma.i32.16x16x32.i8"];
            check(cores.iter().any(|c| ir.contains(c)), "no 8-bit product");
            check(ir.contains("ldmatrix") == target.ldmatrix, "fragments not read by ldmatrix where it is");
        }
        if name.starts_with("group gemm") {
            let reg_operand = |s: &crate::ir::Stmt| matches!(s, crate::ir::Stmt::Copy { dst, .. } if tile.value(*dst).place == crate::ir::Place::Reg && tile.value(*dst).dtype == ScalarDType::BFloat16);
            check(!tile.walk().any(|(_, s)| reg_operand(s)), "an operand was loaded into registers");
            check(ir.contains("wgmma.mma_async.sync.aligned.m64n"), "no warpgroup product");
            check(!ir.contains("@llvm.nvvm.mma.") && !ir.contains("ldmatrix"), "a warp-level product");
            check(ir.contains("wgmma.fence") && ir.contains("fence.proxy.async"), "an unfenced chain");
            let in_flight =
                lowering.schedule == Schedule::Uniform { fill: Fill::CpAsync, unroll: false, async_products: true };
            check(ir.contains("@llvm.nvvm.wgmma.wait_group") == in_flight, "the template's wait");
            check(ir.contains("wgmma.wait_group.sync.aligned 0;") == !in_flight, "the chain's own wait");
        }

        match target.arch {
            GpuArch::Amd(arch) => match svod_runtime::amd::compile_ir_to_amd_object(&ir, arch) {
                Ok(object) => {
                    let descriptor = format!("{}.kd", prog.name);
                    check(object.starts_with(b"\x7fELF"), "an ELF code object");
                    check(
                        object.windows(descriptor.len()).any(|w| w == descriptor.as_bytes()),
                        &format!("no {descriptor}"),
                    );
                }
                Err(e) if e.to_string().contains("AMDGPU target") || e.to_string().contains("too old") => {
                    eprintln!("skipped: {e}")
                }
                Err(e) => check(false, &e.to_string()),
            },
            GpuArch::Cuda(arch) => match svod_runtime::cuda::compile_ir_to_ptx(&ir, arch) {
                Ok(ptx) => {
                    let sm = format!("sm_{}{}{}", arch.major, arch.minor, if arch.major >= 9 { "a" } else { "" });
                    check(String::from_utf8_lossy(&ptx).contains(&format!(".target {sm}")), &format!("not {sm}"));
                    if let Some(ptxas) = &ptxas {
                        let mut command = Command::new(ptxas);
                        command.args([&format!("-arch={sm}"), "-o", "/dev/null", "/dev/stdin"]);
                        if let Err(e) = run(command, &ptx, &format!("ptxas {sm}")) {
                            check(false, &e);
                        }
                    }
                }
                Err(e) if e.to_string().contains("NVPTX target") => eprintln!("skipped: clang has no NVPTX target"),
                Err(e) => check(false, &e.to_string()),
            },
            GpuArch::Metal(_) => unreachable!("Metal has no families"),
        }
        assert!(failed.is_empty(), "{}", failed.join("; "));
    });
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

/// What a lowering derives from the block index alone, at every hardware
/// position of its grid (in linear order, x fastest): the values of the
/// expressions over the block index and constants that the rest of the
/// kernel reads (addresses, guards), sorted.
fn block_derived(prog: Program, lowering: &Lowering) -> Vec<Vec<String>> {
    use std::collections::{HashMap, HashSet};
    use svod_ir::ConstValue;
    use svod_ir::uop::eval::{eval_binary_op, eval_ternary_op, eval_unary_op};
    let root = linear(prog, lowering);
    let nodes = root.toposort();
    let axis = |u: &Arc<UOp>| match u.op() {
        Op::Special(ops::Special { name, .. }) => name.strip_prefix("gidx").map(|a| a.parse::<usize>().unwrap()),
        _ => None,
    };
    // Children come first: a node is pure when it is a constant, a block
    // index, or arithmetic over pure nodes.
    let mut pure: HashMap<u64, bool> = HashMap::new();
    for u in &nodes {
        let arith = matches!(u.op(), Op::Unary(..) | Op::Binary(..) | Op::Ternary(..) | Op::Cast(..));
        let leaf = matches!(u.op(), Op::Const(_)) || axis(u).is_some();
        let p = leaf || (arith && u.op().sources().iter().all(|s| pure[&s.id]));
        pure.insert(u.id, p);
    }
    let read: HashSet<u64> = nodes
        .iter()
        .filter(|u| !pure[&u.id] && !Arc::ptr_eq(u, &root))
        .flat_map(|u| u.op().sources().into_iter().map(|s| s.id))
        .collect();
    let derived: Vec<&Arc<UOp>> = nodes
        .iter()
        .filter(|u| pure[&u.id] && read.contains(&u.id) && u.toposort().iter().any(|s| axis(s).is_some()))
        .collect();
    fn eval(u: &Arc<UOp>, at: &dyn Fn(&Arc<UOp>) -> Option<i64>, memo: &mut HashMap<u64, ConstValue>) -> ConstValue {
        if let Some(v) = memo.get(&u.id) {
            return *v;
        }
        let mut arg = |s: &Arc<UOp>| eval(s, at, memo);
        let v = match (at(u), u.op()) {
            (Some(index), _) => ConstValue::Int(index),
            (None, Op::Const(c)) => c.0,
            (None, Op::Unary(op, x)) => eval_unary_op(*op, arg(x)).expect("a defined unary"),
            (None, Op::Binary(op, x, y)) => eval_binary_op(*op, arg(x), arg(y)).expect("a defined binary"),
            (None, Op::Ternary(op, x, y, z)) => {
                let (x, y, z) = (arg(x), arg(y), arg(z));
                eval_ternary_op(*op, x, y, z).expect("a defined ternary")
            }
            (None, Op::Cast(ops::Cast { src, dtype })) => arg(src).cast(dtype).expect("a defined cast"),
            (None, other) => unreachable!("not block arithmetic: {}", other.as_ref()),
        };
        memo.insert(u.id, v);
        v
    }
    let mut ends = [1i64; 3];
    for u in nodes.iter().filter(|u| axis(u).is_some()) {
        let Op::Special(ops::Special { end, .. }) = u.op() else { unreachable!() };
        ends[axis(u).unwrap()] = eval(end, &|_| None, &mut HashMap::new()).try_int().expect("a constant grid");
    }
    (0..ends.iter().product::<i64>())
        .map(|l| {
            let hw = [l % ends[0], l / ends[0] % ends[1], l / (ends[0] * ends[1])];
            let at = |u: &Arc<UOp>| axis(u).map(|a| hw[a]);
            let mut memo = HashMap::new();
            let mut values: Vec<String> = derived.iter().map(|u| format!("{:?}", eval(u, &at, &mut memo))).collect();
            values.sort();
            values
        })
        .collect()
}

/// A one-die target lowers the block index as the hardware's; on eight
/// dies the hardware position `l` computes every block-derived value (each
/// address and guard) the one-die kernel computes at `die_order(l)`, so
/// the blocks a die runs are a contiguous run of the kernel's order.
#[test]
fn the_block_index_is_renumbered_only_over_several_dies() {
    let cfg = config::gemm_cfg([64, 64, 32], 2, [2, 2], false);
    let spec = GemmSpec { m: 192, n: 320, k: 64, batch: Batch::Static(3), epilogue: Epilogue::default(), cfg };
    let lowering = |xcds: u32| {
        let target = Target { xcds, ..amd(AmdArch::Gfx942, 304) };
        cfg.lowering(&crate::atoms::Device { target, costs: super::costs() })
    };
    let (one, eight) =
        (block_derived(gemm::<BF16>(&spec), &lowering(1)), block_derived(gemm::<BF16>(&spec), &lowering(8)));
    let total = one.len() as u64;
    assert_eq!(total, 3 * 5 * 3, "the grid");
    for l in 0..total {
        let block = crate::lower::emit::die_order(l, total, 8);
        assert_eq!(eight[l as usize], one[block as usize], "position {l} runs block {block}");
    }
    assert_ne!(eight, one, "the blocks are renumbered");
    assert!(one.iter().all(|v| !v.is_empty()), "the block index reaches the kernel");
    let text = || render(gemm::<BF16>(&spec), &lowering(1));
    assert_eq!(text(), text(), "a one-die program is deterministic");
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

/// Every output parameter of `prog` after launching it under `lowering` on
/// the default device, `None` for inputs.
pub(super) fn on_device(prog: Program, lowering: &Lowering, params: &[Vec<f64>]) -> Vec<Option<Vec<f64>>> {
    let vars: Vec<(String, i64)> = prog.vars.iter().map(|v| (v.name.clone(), v.max)).collect();
    let vars: Vec<(&str, i64)> = vars.iter().map(|(n, v)| (n.as_str(), *v)).collect();
    on_device_with(prog, lowering, params, &vars)
}

/// [`on_device`] with the program's variables bound to `vars`.
pub(super) fn on_device_with(
    prog: Program,
    lowering: &Lowering,
    params: &[Vec<f64>],
    vars: &[(&str, i64)],
) -> Vec<Option<Vec<f64>>> {
    launched(prog, lowering, params, vars, false)
}

/// [`on_device_with`], the program lowered for its target alone first and
/// launched as that body where `lowered`, as the tuner launches it.
fn launched(
    prog: Program,
    lowering: &Lowering,
    params: &[Vec<f64>],
    vars: &[(&str, i64)],
    lowered: bool,
) -> Vec<Option<Vec<f64>>> {
    let decl = prog.params.clone();
    let tensors: Vec<Tensor> = decl
        .iter()
        .zip(params)
        .map(|(p, v)| match p.dtype {
            ScalarDType::Int32 => Tensor::from_slice(v.iter().map(|&x| x as i32).collect::<Vec<_>>()),
            dtype => Tensor::from_slice(v.iter().map(|&x| x as f32).collect::<Vec<_>>()).cast(DType::Scalar(dtype)),
        })
        .collect();
    let refs: Vec<&Tensor> = tensors.iter().collect();
    let outs = match lowered {
        true => {
            let body = crate::lower::standalone(prog.clone(), lowering).unwrap().program;
            crate::launch::graph_launch_lowered(&prog, lowering, body, &refs).unwrap()
        }
        false => graph_launch_all(prog, lowering, &refs).unwrap(),
    };
    decl.iter()
        .zip(outs)
        .map(|(p, t)| {
            (p.kind != ParamKind::In).then(|| {
                t.prepare().unwrap().execute_with_vars(vars).unwrap();
                let mut bytes = vec![0u8; p.elems * p.dtype.bytes()];
                t.buffer().unwrap().copyout(&mut bytes).unwrap();
                super::decode(p.dtype, &bytes)
            })
        })
        .collect()
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
                .position(|(g, w)| !(g == w || g.is_nan() && w.is_nan() || (g - w).abs() <= 2e-2 * w.abs().max(1.0)));
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
    let Some(target) = super::device_target() else { return };
    check_on_device(&target);
}

/// Each family lowered for its target alone and launched as that body, as
/// the tuner launches a candidate it fingerprinted, writes bit for bit what
/// the same program lowered at its launch writes.
#[test]
fn a_program_lowered_alone_runs_as_one_lowered_at_launch() {
    let Some(target) = super::device_target() else { return };
    for (name, prog, lowering) in families(&target) {
        let params = inputs(&name, &prog);
        let vars: Vec<(String, i64)> = prog.vars.iter().map(|v| (v.name.clone(), v.max)).collect();
        let vars: Vec<(&str, i64)> = vars.iter().map(|(n, v)| (n.as_str(), *v)).collect();
        let bits = |outs: Vec<Option<Vec<f64>>>| -> Vec<Option<Vec<u64>>> {
            outs.into_iter().map(|o| o.map(|v| v.iter().map(|x| x.to_bits()).collect())).collect()
        };
        let at_launch = bits(launched(prog.clone(), &lowering, &params, &vars, false));
        assert_eq!(bits(launched(prog, &lowering, &params, &vars, true)), at_launch, "{name}");
    }
}

/// The device's occupancy facts come from the device where it reports them
/// (SIMDs per CU and waves per SIMD on AMD, the register file and waves per
/// SM on CUDA), and agree with the architecture's defaults the host tests
/// plan with: a lattice ranked on the host is the one the device runs.
#[test]
fn the_device_reports_the_architectures_occupancy() {
    let Some(target) = super::device_target() else { return };
    let arch = Target::for_arch(target.arch);
    assert_eq!(target.occupancy, arch.occupancy, "{:?}", target.arch);
    assert!(target.sms.is_some_and(|n| n > 0));
}

/// The RDNA data path on an NVIDIA GPU: register-staged fills into the
/// chunk layout, vector shared-memory gathers instead of `ldmatrix`, and
/// two-slot rings, measured against the interpreter (a device without
/// `cp.async` runs them in the family test).
#[test]
fn register_staged_families_match_on_cuda() {
    let Some(target) = super::cp_async_target() else { return };
    check_on_device(&Target { fill: Fill::Staged, ldmatrix: false, ..target });
}

/// The launches of every candidate the op layer may plan on the planner's
/// target, each with the shared memory its ranking admitted it with: the
/// GEMM lists of a large, a small-M and a ragged shape (the large one as
/// deep as the shortlist compiles where `shortlist_deep`), the attention
/// lists of every head size and both query regimes, the convolution lists
/// of every YOLO class (a split's merge after its kernel, no shared memory
/// of its own), and the nearest lists of a shallow, a middle and a deep
/// self-join under several metrics, floors and groups, and of the depths
/// that stream the rows (a last chunk overhanging 800), a row's least
/// alone and lists of 4 to 32.
fn ranked_launches(planner: &Planner, shortlist_deep: bool) -> Vec<(Launch, (usize, usize))> {
    let mut launches = vec![];
    for (m, n, k) in [(4096, 4096, 4096), (704, 512, 512), (1500, 1280, 5120), (37, 96, 48)] {
        for gated in [false, true] {
            let ranked = planner.gemm_ranked(1, m, n, k, gated);
            let deep = if m == 4096 && shortlist_deep { config::SHORTLIST } else { 8 };
            for cfg in config::distinct(&ranked).take(deep) {
                let epilogue = Epilogue { gated, ..Epilogue::DEFAULT };
                let spec = GemmSpec { m: 100, n: 96, k: cfg.tile[2] * 2, batch: Batch::Static(1), epilogue, cfg };
                let launch =
                    (format!("{m}x{n}x{k} g{gated} {cfg:?}"), gemm::<BF16>(&spec), cfg.lowering(&planner.device));
                launches.push((launch, (cfg.smem_bytes(gated, 2), 0)));
            }
        }
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
                let launch = (format!("d{d} t{t} {cfg:?}"), attention::<BF16>(&spec), cfg.lowering(&planner.device));
                let scratch = cfg.scratch_bytes(&planner.device.target);
                launches.push((launch, (cfg.smem_bytes(d, false) + scratch, scratch)));
            }
        }
    }
    for class in super::conv::YOLO {
        let g = super::conv::yolo_geom(class);
        let [ho, wo] = g.out_hw();
        for c in conv_first_eight(planner, 1, ho * wo, &g) {
            let spec =
                ConvSpec { batch: Batch::Static(1), geom: g, epilogue: Epilogue::DEFAULT, cfg: c.gemm, split: c.split };
            for (i, (prog, lowering)) in spec.programs::<F16>(&planner.device).into_iter().enumerate() {
                let smem = if i == 0 { c.gemm.smem_bytes(false, 2) } else { 0 };
                launches.push(((format!("{g:?} {c:?} #{i}"), prog, lowering), (smem, 0)));
            }
        }
    }
    let (none, both) = (Floors::default(), Floors { row: true, col: true });
    for (d, metric, floors, groups, k) in [
        (16, Metric::ShiftedL2, none, false, 1),
        (64, Metric::L2, both, true, 1),
        (128, Metric::SquaredL2, none, true, 1),
        (512, Metric::NegDot, none, false, 1),
        (800, Metric::SquaredL2, both, true, 1),
        (1024, Metric::ShiftedL2, none, false, 1),
        (64, Metric::SquaredL2, none, false, 4),
        (128, Metric::NegDot, none, true, 16),
        (768, Metric::SquaredL2, none, false, 8),
        (768, Metric::NegDot, none, false, 32),
    ] {
        let dtype = ScalarDType::Float16;
        let p = Nearest { n: 1 << 16, m: 1 << 16, d, dtype, metric, floors, groups, k };
        for cfg in first_eight(&planner.nearest_ranked(p)) {
            let spec = NearestSpec { n: 100, m: 150, d, metric, floors, groups, k, cfg, live: Default::default() };
            let name = format!("nearest d{d} {metric:?} {floors:?} {groups} k{k} {cfg:?}");
            let launch = (name, nearest::<BF16>(&spec), cfg.lowering(&planner.device));
            launches.push((launch, (cfg.smem_bytes(d, dtype), 0)));
        }
    }
    launches
}

/// The static shared memory a lowered program declares: its rings and any
/// scratch the lowering added (a relayout through shared memory).
fn shared_bytes(linear: &Arc<UOp>) -> usize {
    let local = |u: &Arc<UOp>| match u.op() {
        Op::Buffer(ops::Buffer { shape, arg }) if arg.addrspace == Some(svod_dtype::AddrSpace::Local) => Some(
            shape.vmax().try_int().expect("a committed size") as usize * u.dtype().scalar().expect("a type").bytes(),
        ),
        _ => None,
    };
    linear.toposort().iter().filter_map(local).sum()
}

/// Every candidate the op layer may plan ([`ranked_launches`]) lowers (the
/// tune store would otherwise time a failure), declaring the shared memory
/// its ranking admitted it with: the rings as the config's `smem_bytes`
/// counts them, so a `Rejected::Smem` is the kernel's own count; an
/// attention config on a core whose accumulator cannot feed a product is
/// admitted with the scratch its probabilities may cross besides
/// ([`FaCfg::scratch_bytes`], a bound), and declares it or not as the
/// inference decides.
#[test_case(amd(AmdArch::Gfx1100, 48), false; "gfx1100")]
#[test_case(amd(AmdArch::Gfx1201, 64), true; "gfx1201")]
#[test_case(amd(AmdArch::Gfx1151, 40), false; "gfx1151")]
#[test_case(amd(AmdArch::Gfx942, 304), false; "gfx942")]
#[test_case(crate::atoms::sm86(), true; "sm86")]
#[test_case(hopper(), false; "hopper")]
fn every_candidate_lowers(target: Target, shortlist_deep: bool) {
    let planner = planner(target.clone());
    let (launches, admitted): (Vec<Launch>, Vec<(usize, usize)>) =
        ranked_launches(&planner, shortlist_deep).into_iter().unzip();
    let admitted: std::collections::HashMap<String, (usize, usize)> =
        launches.iter().map(|(name, ..)| name.clone()).zip(admitted).collect();
    each(launches, |name, prog, lowering| {
        let declared = shared_bytes(&linear(prog, &lowering));
        let (smem, bound) = admitted[name];
        assert!(
            declared == smem || declared + bound == smem,
            "declares {declared} bytes of shared memory, admitted with {smem} ({bound} of them a bound)"
        );
    });
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
    block.walk().filter_map(|(_, s)| if let Stmt::Let { op, .. } = s { Some(op.clone()) } else { None }).collect()
}

/// The tile ops of `block` (recursively) that `f` picks.
fn tile_ops(block: &crate::ir::Block, f: &dyn Fn(&crate::ir::TileOp) -> bool) -> usize {
    block.walk().filter(|(_, s)| matches!(s, crate::ir::Stmt::Let { op, .. } if f(op))).count()
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
        core: Core::Warp,
        schedule: crate::schedule::Schedule::Uniform { fill: target.fill, unroll: false, async_products: false },
        device: declared(target),
        grid: crate::layouts::WarpGrid { rows: 1, cols: 1 },
    };
    let ir = render(k.finish(), &lowering);
    assert_eq!(ir.contains("@llvm.amdgcn.fdot2.f32.bf16("), dot2, "{ir}");
    // A lane's odd member out is widened on every target.
    assert!(dot2 || ir.contains("fpext bfloat"), "{ir}");
}

/// `Sync::Fence` statements in `block`, recursively.
fn fences(block: &crate::ir::Block) -> usize {
    use crate::ir::{Stmt, Sync};
    block.walk().filter(|(_, s)| matches!(s, Stmt::Sync(Sync::Fence))).count()
}

/// The register-staged template fences every trip between its loads and its
/// products, and over two slots again between its products and its commit,
/// on every target that runs it; the fences are instructions only where the
/// scheduler would otherwise move the loads or the commit past the products
/// (RDNA), and the interpreter never sees them, so
/// `every_family_lowers_keeps_its_numbers_and_compiles` covers their
/// placement. One slot's commit needs none: a barrier holds it back.
#[test_case(AmdArch::Gfx1201, true; "gfx1201 fences")]
#[test_case(AmdArch::Gfx1100, true; "gfx1100 fences")]
#[test_case(AmdArch::Gfx942, false; "gfx942 keeps order")]
fn register_staged_trips_are_fenced(arch: AmdArch, emitted: bool) {
    let target = amd(arch, 64);
    assert_eq!(target.sync.commit_fence, emitted);
    let device = declared(target);
    for (stages, want) in [(2, 2), (1, 1)] {
        let cfg = config::gemm_cfg([64, 64, 32], stages, [2, 2], stages == 2);
        let spec = GemmSpec { m: 130, n: 192, k: 128, batch: Batch::Static(1), epilogue: Epilogue::DEFAULT, cfg };
        let tile = lowered_tile(gemm::<BF16>(&spec), &cfg.lowering(&device));
        assert_eq!(
            fences(&tile.body),
            want,
            "{stages} slots: a fence after the loads, one more before a two-slot commit"
        );
    }
}

/// Whether some barrier in `asm` is reached with a global load in flight: one
/// issued since the last wait that left none outstanding (`s_wait_loadcnt
/// 0x0` and its merged forms on gfx12, `vmcnt(0)` before).
fn a_load_crosses_a_barrier(asm: &str) -> bool {
    let mut in_flight = false;
    for op in asm.lines().map(str::trim).filter(|l| !l.starts_with(';')) {
        if op.starts_with("global_load") {
            in_flight = true;
        } else if (op.starts_with("s_wait_loadcnt") && op.ends_with(" 0x0")) || op.contains("vmcnt(0)") {
            in_flight = false;
        } else if (op == "s_barrier" || op.starts_with("s_barrier_signal ")) && in_flight {
            return true;
        }
    }
    false
}

/// A tk barrier fences shared memory only, so the one-slot register-staged
/// trip's barrier between its products and its commit leaves the next step's
/// loads in flight. A plain workgroup release fence on RDNA would wait for
/// them first (`s_wait_loadcnt 0x0` on gfx12, `vmcnt(0)` on gfx11); CDNA's
/// fence never waited, and its code is unchanged.
#[test_case(AmdArch::Gfx1201, true; "gfx1201")]
#[test_case(AmdArch::Gfx1151, true; "gfx1151")]
#[test_case(AmdArch::Gfx1100, true; "gfx1100")]
#[test_case(AmdArch::Gfx942, false; "gfx942")]
#[test_case(AmdArch::Gfx950, false; "gfx950")]
fn prefetch_loads_stay_in_flight_across_a_barrier(arch: AmdArch, rdna: bool) {
    let device = declared(amd(arch, 64));
    let cfg = config::gemm_cfg([64, 64, 32], 1, [2, 2], false);
    let spec = GemmSpec { m: 256, n: 256, k: 512, batch: Batch::Static(1), epilogue: Epilogue::DEFAULT, cfg };
    let ir = render(gemm::<BF16>(&spec), &cfg.lowering(&device));
    let mmra = r#", !mmra !{!"amdgpu-synchronize-as", !"local"}"#;
    let fences = ir.matches("fence syncscope(\"workgroup\")").count();
    assert!(fences > 0 && ir.matches(mmra).count() == fences, "every barrier fence is LDS-only:\n{ir}");
    let asm = |ir: &str| match svod_runtime::amd::compile_ir_to_amd_asm(ir, arch) {
        Ok(asm) => Some(String::from_utf8(asm).unwrap()),
        Err(e) if e.to_string().contains("AMDGPU target") || e.to_string().contains("too old") => None,
        Err(e) => panic!("{e}"),
    };
    let (Some(local), Some(plain)) = (asm(&ir), asm(&ir.replace(mmra, ""))) else {
        eprintln!("skipped: clang has no AMDGPU target");
        return;
    };
    assert!(a_load_crosses_a_barrier(&local), "the next step's loads drain at the barrier:\n{local}");
    if rdna {
        assert!(!a_load_crosses_a_barrier(&plain), "a plain workgroup fence no longer drains:\n{plain}");
    } else {
        assert_eq!(local, plain, "the LDS-only fence changed CDNA code");
    }
}

/// `Relayout` statements of tiles in `block`, recursively; vectors are not
/// counted.
fn tile_relayouts(prog: &Program, block: &crate::ir::Block) -> usize {
    use crate::ir::{Stmt, TileOp};
    let tile = |src: &crate::ir::ValId| prog.value(*src).shape.rows > 1 && prog.value(*src).shape.cols > 1;
    block.walk().filter(|(_, s)| matches!(s, Stmt::Let { op: TileOp::Relayout { src }, .. } if tile(src))).count()
}

/// The probe's plain attention at head size `d` under `cfg` on `target`.
fn attention_at(target: &Target, cfg: crate::kernels::attention::FaCfg, d: usize) -> (Program, Lowering) {
    use crate::kernels::attention::{AttnMask, AttnSpec, attention};
    let spec = AttnSpec {
        batch: Batch::Static(4),
        t: 2048,
        tk: 2048,
        heads: 8,
        kv_heads: 8,
        d,
        mask: AttnMask::default(),
        cache: None,
        scale: 0.125,
        cfg,
    };
    let device = crate::atoms::Device { target: target.clone(), costs: crate::atoms::Costs::default() };
    (attention::<BF16>(&spec), cfg.lowering(&device))
}

/// Two atom rows per warp: the same tile over half the warps, each
/// reading every key and value fragment once for two atom rows of
/// products, at more registers a lane; it lowers and renders on both
/// vendors, and the planner admits it only where the warps divide the
/// query block.
#[test_case(crate::atoms::sm86(); "sm86")]
#[test_case(amd(AmdArch::Gfx1201, 64); "gfx1201")]
#[test_case(amd(AmdArch::Gfx942, 304); "gfx942")]
fn two_atom_rows_per_warp_lower(target: Target) {
    use crate::kernels::attention::{FaCfg, Forms, WARP_ROWS};
    let all = Forms { truncate_p: true, skip_rescale: true, raw_max: true, sum_after: true, bias_direct: false };
    // A wave of 64 needs key blocks twice as deep to give every thread of
    // the 16-row tile a chunk.
    let deep = if target.wave == 64 { 2 } else { 1 };
    for (d, bq, bkv) in [(64, 128, 32 * deep), (128, 128, 16 * deep), (64, 64, 64)] {
        let one = FaCfg { unroll: true, forms: all, ..FaCfg::new(bq, bkv, 2) };
        let two = FaCfg { warp_rows: 2 * WARP_ROWS, ..one };
        assert!(one.fits(&target, d, false) && two.fits(&target, d, false));
        assert_eq!(two.warps() * 2, one.warps());
        let (r1, r2) = (one.registers(&target, d, false).unwrap(), two.registers(&target, d, false).unwrap());
        assert!(r2 > r1 && r2 < 2 * r1, "{r1} -> {r2}: twice the accumulators, the queries and fills shared");
        let (prog, lowering) = attention_at(&target, two, d);
        assert!(!render(prog, &lowering).is_empty());
    }
    let odd = FaCfg { warp_rows: 2 * WARP_ROWS, ..FaCfg::new(16, 16, 2) };
    assert!(!odd.fits(&target, 64, false), "a 16-row block holds no 32-row warp");
    let keyed = FaCfg { warp_rows: 2 * WARP_ROWS, key_warps: 2, ..FaCfg::new(32, 16, 2) };
    assert!(!keyed.fits(&target, 64, false), "key warps share a block of one atom row warp");
}

/// Four warps of one 256-wide row each: every row's maximum, and its sum
/// instead where the maximum is positive, the sums taken under
/// [`Kernel::where_any`](crate::build::Kernel::where_any) (rows 1 and 3 of
/// [`voted_input`] pass, so warps 0 and 2 skip them).
fn voted_rows() -> (Program, crate::layouts::WarpGrid) {
    use crate::build::{CmpOp, F32, Kernel};
    use crate::ir::{ReduceOp, Shape};
    let (rows, d) = (4, 256);
    let mut k = Kernel::new("voted");
    let x = k.param::<F32>("x", ParamKind::In, rows * d);
    let y = k.param::<F32>("y", ParamKind::Out, rows);
    k.warps(rows as u32);
    let xv = k.view(x, 0, [d, 1], Shape::new(rows, d), [None, None]);
    let xs = k.load(xv);
    let most = k.reduce(xs, Axis::Row, ReduceOp::Max);
    let zero = k.zeros::<F32>(Shape::new(rows, 1));
    let positive = k.compare(zero, most, CmpOp::Lt);
    let [out] = k.where_any(positive, |k| [k.reduce(xs, Axis::Row, ReduceOp::Sum)], [most]);
    let yv = k.view(y, 0, [1, 1], Shape::new(rows, 1), [None, None]);
    k.store(yv, out);
    (k.finish(), crate::layouts::WarpGrid { rows: rows as u32, cols: 1 })
}

/// Small integers, every one of rows 0 and 2 negative, so their sums and
/// maxima are exact and only rows 1 and 3 take their sums.
fn voted_input() -> Vec<f64> {
    (0..4 * 256).map(|e| if (e / 256) % 2 == 0 { -1.0 - (e % 5) as f64 } else { (e % 7) as f64 - 3.0 }).collect()
}

/// What [`voted_rows`] computes from [`voted_input`]: rows 0 and 2 their
/// maxima, rows 1 and 3 their sums.
fn voted_want() -> Vec<f64> {
    let x = voted_input();
    (0..4)
        .map(|r| {
            let row = &x[r * 256..(r + 1) * 256];
            let most = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if most > 0.0 { row.iter().sum() } else { most }
        })
        .collect()
}

/// The interpreter takes a branch on a vote as the block's one warp: the
/// sums are found for every row and kept where the rows pass.
#[test]
fn a_branch_on_a_vote_keeps_the_otherwise_where_the_tile_is_clear() {
    let (prog, _) = voted_rows();
    let out = interp::run(&prog, vec![voted_input(), vec![0.0; 4]], &[]).unwrap();
    assert_eq!(out[1], voted_want());
}

/// A branch on a vote is a warp's vote on each target, behind a compiler
/// memory barrier; the kernel compiles where the toolchain is installed.
#[test_case(amd(AmdArch::Gfx1201, 64), "call i32 @llvm.amdgcn.ballot.i32(i1"; "gfx1201")]
#[test_case(amd(AmdArch::Gfx1151, 40), "call i32 @llvm.amdgcn.ballot.i32(i1"; "gfx1151")]
#[test_case(amd(AmdArch::Gfx942, 304), "call i64 @llvm.amdgcn.ballot.i64(i1"; "gfx942")]
#[test_case(crate::atoms::sm86(), "call i1 @llvm.nvvm.vote.any.sync(i32 -1, i1"; "sm_86")]
#[test_case(hopper(), "call i1 @llvm.nvvm.vote.any.sync(i32 -1, i1"; "sm_90")]
fn a_branch_on_a_vote_lowers_to_the_warps_vote(target: Target, vote: &str) {
    let (prog, grid) = voted_rows();
    let device = declared(target.clone());
    let ir = render(prog, &Lowering::uniform(&device, false, grid));
    let (pin, at) = (ir.find(r#"asm sideeffect "", "~{memory}"()"#), ir.find(vote));
    assert!(pin.zip(at).is_some_and(|(p, v)| p < v), "a pinned vote:\n{ir}");
    assert_eq!(ir.matches(vote).count(), 1, "one vote:\n{ir}");
    if let GpuArch::Amd(arch) = target.arch {
        match svod_runtime::amd::compile_ir_to_amd_object(&ir, arch) {
            Ok(object) => assert_eq!(&object[..4], b"\x7fELF"),
            Err(e) if e.to_string().contains("AMDGPU target") || e.to_string().contains("too old") => {
                eprintln!("skipped: {e}");
            }
            Err(e) => panic!("{e}"),
        }
    }
}

/// On the device the warps whose rows all fail skip the sums and keep their
/// maxima; the others take them: the interpreter's numbers exactly.
#[test]
fn a_branch_on_a_vote_matches_the_interpreter_on_the_device() {
    let Some(target) = super::device_target() else { return };
    let (prog, grid) = voted_rows();
    let device = declared(target);
    let got = on_device(prog, &Lowering::uniform(&device, false, grid), &[voted_input(), vec![0.0; 4]]);
    assert_eq!(got[1].as_deref(), Some(&voted_want()[..]));
}

/// The labels the conditional branches on a vote mask in `asm` jump to
/// (`s_cbranch_vccz`/`vccnz`, the vote's mask in VCC), and whether a global
/// load lies between each such label and the next barrier: the next step's
/// loads sunk past the branch, waited on before the commit.
fn loads_sunk_past_a_vote(asm: &str) -> Vec<(String, bool)> {
    let lines: Vec<&str> = asm.lines().map(str::trim).filter(|l| !l.starts_with(';')).collect();
    let targets = lines.iter().filter_map(|l| {
        let rest = l.strip_prefix("s_cbranch_vccz ").or_else(|| l.strip_prefix("s_cbranch_vccnz "))?;
        Some(rest.trim().to_string())
    });
    targets
        .map(|label| {
            let head = lines.iter().position(|l| l.starts_with(&format!("{label}:"))).expect("a branch target");
            let block = lines[head + 1..].iter().take_while(|l| !l.starts_with("s_barrier"));
            let sunk = block.clone().any(|l| l.starts_with("global_load"));
            (label, sunk)
        })
        .collect()
}

/// A resident nearest kernel's steady trip issues the next step's global
/// loads before its products and branches on a vote after them; the loads
/// are used only past the branch, and without the vote's memory barrier the
/// backend sinks them there, where the commit waits on them at once. They
/// stay at the head of the trip.
#[test_case(AmdArch::Gfx1201; "gfx1201")]
#[test_case(AmdArch::Gfx1151; "gfx1151")]
fn the_next_steps_loads_stay_above_a_vote(arch: AmdArch) {
    use crate::kernels::nearest::NearestCfg;
    let device = declared(amd(arch, 64));
    let cfg = NearestCfg { warp_rows: 32, ..NearestCfg::new(128, 16, 2) };
    let spec = NearestSpec {
        n: 32768,
        m: 32768,
        d: 256,
        metric: Metric::NegDot,
        floors: Floors::default(),
        groups: false,
        k: 1,
        cfg,
        live: Default::default(),
    };
    let ir = render(nearest::<BF16>(&spec), &cfg.lowering(&device));
    let asm = match svod_runtime::amd::compile_ir_to_amd_asm(&ir, arch) {
        Ok(asm) => String::from_utf8(asm).unwrap(),
        Err(e) if e.to_string().contains("AMDGPU target") || e.to_string().contains("too old") => {
            eprintln!("skipped: {e}");
            return;
        }
        Err(e) => panic!("{e}"),
    };
    let votes = loads_sunk_past_a_vote(&asm);
    assert!(!votes.is_empty(), "a branch on a vote:\n{asm}");
    assert!(votes.iter().all(|(_, sunk)| !sunk), "{votes:?}:\n{asm}");
}

/// Where the accumulator cannot feed the P·V product (RDNA3), whether the
/// probabilities cross shared memory is the whole program's and the
/// costs' call, not the config's: under the Strix Halo's store
/// ([`super::gfx1151_store_costs`]) the 32×128 block of head size 64
/// re-holds them through shared memory under a key mask (the card refused
/// it admitted at its 64 KB of rings alone: "local memory (73728) exceeds
/// limit (65536)") and exchanges them through lanes without one. The
/// admission's [`FaCfg::scratch_bytes`] bounds both.
#[test_case(AttnMask::default(), 0; "plain")]
#[test_case(AttnMask { key_mask: true, ..AttnMask::default() }, 32 * 128 * 2; "key mask")]
fn the_probabilities_cross_shared_memory_as_the_program_decides(mask: AttnMask, scratch: usize) {
    let target = amd(AmdArch::Gfx1151, 40);
    let device = crate::atoms::Device { target: target.clone(), costs: super::gfx1151_store_costs() };
    let cfg = crate::kernels::attention::FaCfg::new(32, 128, 2);
    let spec = AttnSpec {
        batch: Batch::Static(2),
        t: 100,
        tk: 100,
        heads: 4,
        kv_heads: 4,
        d: 64,
        mask,
        cache: None,
        scale: 0.125,
        cfg,
    };
    let declared = shared_bytes(&linear(attention::<F16>(&spec), &cfg.lowering(&device)));
    assert_eq!(declared, cfg.smem_bytes(64, false) + scratch);
    assert!(scratch <= cfg.scratch_bytes(&target));
}
