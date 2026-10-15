//! GPU-free graph-shape checks of the CUDA movement lowerings: the `ldmatrix.x4`
//! LOCAL→REG gather and the `cp.async` GLOBAL→LOCAL fill — which primitives are
//! emitted, in what count, and how the fetched words land in the fragment
//! registers. The AMD paths are pinned by the golden fingerprints.

use std::collections::HashMap;
use std::sync::Arc;

use svod_dtype::{CudaArch, DType, DeviceSpec, GpuArch};
use svod_ir::{ConstValue, Op, UOp};
use test_case::test_case;

use crate::tiles::{
    RT_16X16_MMA, RT_16X16_MMA_HALVES, RTBaseShape, ST_16X16_MMA, ST_16X32_MMA, ST_16X64_MMA, STBaseShape, TileLayout,
};
use crate::{ArchCaps, Kernel, MoveIdx};
use svod_ir::ops;

const SM_86: GpuArch = GpuArch::Cuda(CudaArch::from_compute_capability(8, 6));

fn customs<'a>(nodes: &'a [Arc<UOp>], needle: &str) -> Vec<&'a Arc<UOp>> {
    nodes.iter().filter(|u| matches!(u.op(), Op::Custom(ops::Custom { code, .. }) if code.contains(needle))).collect()
}

/// The constant register offset a STORE into a REG buffer targets.
fn reg_offset(store: &Arc<UOp>) -> i64 {
    let Op::Store(ops::Store { index, .. }) = store.op() else { panic!("STORE") };
    let Op::Index(ops::Index { indices, .. }) = index.op() else { panic!("INDEX") };
    match indices[0].op() {
        Op::Const(c) => match c.0 {
            ConstValue::Int(v) => v,
            other => panic!("{other:?}"),
        },
        other => panic!("register offset must be constant, got {other:?}"),
    }
}

/// Which fetched word (`extractvalue .., i`) a stored value comes from.
fn word_of(store: &Arc<UOp>) -> usize {
    let Op::Store(ops::Store { value, .. }) = store.op() else { panic!("STORE") };
    value
        .toposort()
        .iter()
        .find_map(|u| match u.op() {
            Op::Custom(ops::Custom { code, .. }) if code.starts_with("extractvalue") => {
                code.rsplit(", ").next().unwrap().trim().parse().ok()
            }
            _ => None,
        })
        .expect("stored value extracts an ldmatrix word")
}

/// A 32×32 bf16 `ST_16X16_MMA` tile gathered into an `mma.sync` fragment on sm_86 is
/// four `ldmatrix.x4` (plain when the layouts agree, `.trans` when they differ), and
/// result `i` of each lands in register pair `feed[i]`: the order the core reads
/// the fragment's pairs in.
/// The strip may be the fragment-wide [`ST_16X16_MMA`] or a full-row one, whose
/// 64- or 128-byte base tile holds several fragments of a row.
#[test_case(TileLayout::Row, RT_16X16_MMA, false, ST_16X16_MMA; "row gather")]
#[test_case(TileLayout::Col, RT_16X16_MMA, true, ST_16X16_MMA; "col gather is ldsm4t")]
#[test_case(TileLayout::Row, RT_16X16_MMA_HALVES, false, ST_16X16_MMA; "row gather in n-halves")]
#[test_case(TileLayout::Col, RT_16X16_MMA_HALVES, true, ST_16X16_MMA; "col gather in n-halves")]
#[test_case(TileLayout::Row, RT_16X16_MMA, false, ST_16X32_MMA; "row gather from 64-byte rows")]
#[test_case(TileLayout::Row, RT_16X16_MMA_HALVES, false, ST_16X32_MMA; "n-halves from 64-byte rows")]
#[test_case(TileLayout::Col, RT_16X16_MMA, true, ST_16X32_MMA; "col gather from 64-byte rows")]
fn ldmatrix_gather_shape(rt_layout: TileLayout, base: RTBaseShape, trans: bool, strip: STBaseShape) {
    let ker = Kernel::new("ldsm", [1, 1, 1], 32, vec![], ArchCaps::for_arch(SM_86));
    let warp = ker.warp();
    let st = ker.st((32, 32), DType::BFloat16, TileLayout::Row, strip);
    let rt = ker.rt((32, 32), DType::BFloat16, rt_layout, base);
    let rt = warp.load(rt, st, MoveIdx::default());
    let nodes = rt.uop().toposort();
    let intrinsic = format!("ldmatrix.sync.aligned.m8n8.x4{}.b16", if trans { ".trans" } else { "" });
    assert_eq!(customs(&nodes, &intrinsic).len(), 4, "one ldmatrix.x4 per 16×16 fragment");
    assert_eq!(customs(&nodes, "ldmatrix").len(), 4, "no other ldmatrix form");
    let stores: Vec<&Arc<UOp>> = nodes.iter().filter(|u| matches!(u.op(), Op::Store(..))).collect();
    assert_eq!(stores.len(), 4 * 8, "every fragment register is stored once");
    for store in stores {
        let reg = reg_offset(store) % 8;
        assert_eq!(base.feed[word_of(store)], reg as usize / 2, "register {reg} takes the word fed to its pair");
    }
    assert!(!nodes.iter().any(|u| matches!(u.op(), Op::Range(..))), "the gather is flat");
}

/// The register buffer an INDEX addresses, through the `AFTER`s that order it.
fn reg_root(index: &Arc<UOp>) -> Arc<UOp> {
    let Op::Index(ops::Index { buffer, .. }) = index.op() else { panic!("INDEX") };
    let mut buf = buffer.clone();
    while let Op::After(ops::After { passthrough, .. }) = buf.op() {
        buf = passthrough.clone();
    }
    buf
}

/// Every operand the `mma.sync` core reads in a GEMM step is one aligned run of
/// consecutive `ldmatrix.x4` results — all four for the A slot, an even-started
/// pair for each B-slot n-half — so `ptxas` binds it to the registers the load
/// wrote and moves nothing. Under tk's `Col` accumulator the A-position tile is
/// the core's B operand, whether B arrives `[N, K]` (`mma_abt`, gathered plain)
/// or `[K, N]` (`mma_ab`, gathered transposed).
#[test_case(TileLayout::Row; "b as n by k")]
#[test_case(TileLayout::Col; "b as k by n")]
fn mma_operands_are_consecutive_ldmatrix_results(b_layout: TileLayout) {
    let ker = Kernel::new("mma", [1, 1, 1], 32, vec![], ArchCaps::for_arch(SM_86));
    let warp = ker.warp();
    let bf = DType::BFloat16;
    let strip = || ker.st((32, 32), bf.clone(), TileLayout::Row, ST_16X16_MMA);
    let a = warp.load(ker.operand((32, 32), bf.clone(), TileLayout::Row), strip(), MoveIdx::default());
    let b = warp.load(ker.operand_b((32, 32), bf.clone(), b_layout), strip(), MoveIdx::default());
    // Unrolled, every register index is a constant the loads can be matched by.
    ker.set_unroll(true);
    let c = warp.zero(ker.acc((32, 32), TileLayout::Col));
    let c = if b_layout == TileLayout::Row { warp.mma_abt(c, &a, &b) } else { warp.mma_ab(c, &a, &b) };
    let nodes = c.uop().toposort();

    // (register buffer, offset) → (the `ldmatrix` call, its result index).
    let mut fetched: HashMap<(usize, i64), (usize, usize)> = HashMap::new();
    for store in nodes.iter().filter(|u| matches!(u.op(), Op::Store(..))) {
        let Op::Store(ops::Store { index, value, .. }) = store.op() else { unreachable!() };
        let call = value.toposort().into_iter().find_map(|u| match u.op() {
            Op::Custom(ops::Custom { code, deps }) if code.starts_with("extractvalue") => {
                Some(Arc::as_ptr(&deps[0]) as usize)
            }
            _ => None,
        });
        if let Some(call) = call {
            fetched.insert((Arc::as_ptr(&reg_root(index)) as usize, reg_offset(store)), (call, word_of(store)));
        }
    }
    let run = |operand: &Arc<UOp>| -> (usize, Vec<usize>) {
        let Op::Stack(ops::Stack { sources }) = operand.op() else { panic!("an mma operand is a STACK") };
        let at: Vec<(usize, usize)> = sources
            .iter()
            .map(|load| {
                let Op::Load(ops::Load { index, .. }) = load.op() else { panic!("an operand element is a LOAD") };
                let Op::Index(ops::Index { indices, .. }) = index.op() else { panic!("INDEX") };
                let Op::Const(off) = indices[0].op() else { panic!("unrolled register offset is constant") };
                let ConstValue::Int(off) = off.0 else { panic!("integer offset") };
                fetched[&(Arc::as_ptr(&reg_root(index)) as usize, off)]
            })
            .collect();
        assert!(at.iter().all(|(call, _)| *call == at[0].0), "one ldmatrix feeds the whole operand");
        let words: Vec<usize> = at
            .chunks(2)
            .map(|e| {
                assert_eq!(e[0], e[1], "a register's two elements are one result");
                e[0].1
            })
            .collect();
        (at[0].0, words)
    };
    let wmmas: Vec<&Arc<UOp>> = nodes.iter().filter(|u| matches!(u.op(), Op::Wmma(..))).collect();
    assert_eq!(wmmas.len(), 2 * 2 * 2 * 2, "2×2 fragments, 2 K steps, 2 n-halves each");
    for wmma in wmmas {
        let Op::Wmma(ops::Wmma { a, b, .. }) = wmma.op() else { unreachable!() };
        assert_eq!(run(a).1, [0, 1, 2, 3], "the A slot is one whole ldmatrix result run");
        let (_, half) = run(b);
        assert!(half == [0, 1] || half == [2, 3], "a B-slot n-half is an aligned result pair, got {half:?}");
    }
}

/// On AMD the same load stays the scalar gather (no CUDA intrinsic, looped).
#[test]
fn ldmatrix_gather_is_cuda_only() {
    let ker = Kernel::new("gather", [1, 1, 1], 64, vec![], ArchCaps::GFX942);
    let warp = ker.warp();
    let st = ker.st((16, 16), DType::BFloat16, TileLayout::Row, crate::tiles::ST_16X16);
    let rt = ker.rt((16, 16), DType::BFloat16, TileLayout::Row, crate::tiles::RT_16X16);
    let nodes = warp.load(rt, st, MoveIdx::default()).uop().toposort();
    assert!(customs(&nodes, "ldmatrix").is_empty());
    assert!(nodes.iter().any(|u| matches!(u.op(), Op::Range(..))));
}

/// The 128-bit fill of a `64×32` bf16 strip by a 4-warp group on sm_86 is `cp.async`:
/// `64·32·2 / (128·16) = 2` copies per lane, one commit, `wait_group 0`, and the
/// trailing barrier — no scalar LDS store.
#[test]
fn cp_async_fill_shape() {
    let n = 256usize;
    let bufs = vec![UOp::new_buffer(DeviceSpec::Cpu, n * n, DType::BFloat16)];
    let ker = Kernel::new("fill", [1, 1, 1], 128, bufs, ArchCaps::for_arch(SM_86));
    let g = ker.group(4);
    let src = ker.gl(&[1, 1, n, n], DType::BFloat16);
    let st = ker.st((64, 32), DType::BFloat16, TileLayout::Row, ST_16X16_MMA);
    assert!(g.cp_async_fill_applies(&st, &src));
    let filled = g.fill_local_vec(st, src, &[0.into(), 0.into(), 0.into(), 0.into()], 2);
    let nodes = filled.uop().toposort();
    assert_eq!(customs(&nodes, "cp.async.cg.shared.global.16(").len(), 2);
    assert_eq!(customs(&nodes, "cp.async.commit.group").len(), 1);
    assert_eq!(customs(&nodes, "cp.async.wait.group(i32 0)").len(), 1);
    assert_eq!(nodes.iter().filter(|u| matches!(u.op(), Op::Barrier(..))).count(), 1);
    assert!(!nodes.iter().any(|u| matches!(u.op(), Op::Store(..))), "no register-staged LDS store");
}

/// A full-row strip, one base tile across, fills the same way: `64·cols·2 /
/// (128·16)` copies per lane under one commit.
#[test_case(ST_16X32_MMA, 2; "64-byte rows")]
#[test_case(ST_16X64_MMA, 4; "128-byte rows")]
fn cp_async_fill_of_full_rows(strip: STBaseShape, copies: usize) {
    let n = 256usize;
    let bufs = vec![UOp::new_buffer(DeviceSpec::Cpu, n * n, DType::BFloat16)];
    let ker = Kernel::new("fill", [1, 1, 1], 128, bufs, ArchCaps::for_arch(SM_86));
    let g = ker.group(4);
    let src = ker.gl(&[1, 1, n, n], DType::BFloat16);
    let st = ker.st((64, strip.base.cols), DType::BFloat16, TileLayout::Row, strip);
    assert_eq!(st.shape()[st.shape().len() - 3], 1, "one base tile spans the row");
    assert!(g.cp_async_fill_applies(&st, &src));
    let nodes = g.cp_async_fill(&st, &src, &[0.into(), 0.into(), 0.into(), 0.into()], 2).toposort();
    assert_eq!(customs(&nodes, "cp.async.cg.shared.global.16(").len(), copies);
    assert_eq!(customs(&nodes, "cp.async.commit.group").len(), 1);
}

/// `cp.async` needs 16-byte lane runs with no element cast and a chunk-contiguous
/// swizzle; a strip that fails any of these keeps the register path.
#[test]
fn cp_async_fill_gates() {
    let n = 256usize;
    let bufs = vec![
        UOp::new_buffer(DeviceSpec::Cpu, n * n, DType::BFloat16),
        UOp::new_buffer(DeviceSpec::Cpu, n * n, DType::Float32),
    ];
    let ker = Kernel::new("gate", [1, 1, 1], 128, bufs, ArchCaps::for_arch(SM_86));
    let g = ker.group(4);
    let bf = ker.gl(&[1, 1, n, n], DType::BFloat16);
    let f32 = ker.gl(&[1, 1, n, n], DType::Float32);
    let mma = ker.st((64, 32), DType::BFloat16, TileLayout::Row, ST_16X16_MMA);
    assert!(!g.cp_async_fill_applies(&mma, &f32), "f32 → bf16 casts");
    let hk = ker.st((64, 32), DType::BFloat16, TileLayout::Row, crate::tiles::ST_16X16_SWIZZLED_W32);
    assert!(!g.cp_async_fill_applies(&hk, &bf), "the 8-byte-granular XOR splits chunks");
    let amd = Kernel::new("amd", [1, 1, 1], 128, vec![], ArchCaps::GFX942);
    let amd_st = amd.st((64, 32), DType::BFloat16, TileLayout::Row, ST_16X16_MMA);
    assert!(!amd.group(2).cp_async_fill_applies(&amd_st, &bf), "AMD has no cp.async");
}
