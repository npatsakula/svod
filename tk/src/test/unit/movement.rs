//! GPU-free graph-shape checks of the CUDA movement lowerings: the `ldmatrix.x4`
//! LOCAL→REG gather and the `cp.async` GLOBAL→LOCAL fill — which primitives are
//! emitted, in what count, and how the fetched words land in the fragment
//! registers. The AMD paths are pinned by the golden fingerprints.

use std::sync::Arc;

use svod_dtype::{CudaArch, DType, DeviceSpec, GpuArch};
use svod_ir::{ConstValue, Op, UOp};
use test_case::test_case;

use crate::tiles::{RT_16X16_MMA, ST_16X16_MMA, TileLayout};
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

/// A 32×32 bf16 `ST_16X16_MMA` tile gathered into an `RT_16X16_MMA` operand on sm_86 is
/// four `ldmatrix.x4` (plain when the layouts agree, `.trans` when they differ), and
/// register pair `p` of each fragment stores word `plan.words[p]`.
#[test_case(TileLayout::Row, false, [0, 1, 2, 3]; "row gather")]
#[test_case(TileLayout::Col, true, [0, 2, 1, 3]; "col gather is ldsm4t")]
fn ldmatrix_gather_shape(rt_layout: TileLayout, trans: bool, words: [usize; 4]) {
    let ker = Kernel::new("ldsm", [1, 1, 1], 32, vec![], ArchCaps::for_arch(SM_86));
    let warp = ker.warp();
    let st = ker.st((32, 32), DType::BFloat16, TileLayout::Row, ST_16X16_MMA);
    let rt = ker.rt((32, 32), DType::BFloat16, rt_layout, RT_16X16_MMA);
    let rt = warp.load(rt, st, MoveIdx::default());
    let nodes = rt.uop().toposort();
    let intrinsic = format!("ldmatrix.sync.aligned.m8n8.x4{}.b16", if trans { ".trans" } else { "" });
    assert_eq!(customs(&nodes, &intrinsic).len(), 4, "one ldmatrix.x4 per 16×16 fragment");
    assert_eq!(customs(&nodes, "ldmatrix").len(), 4, "no other ldmatrix form");
    let stores: Vec<&Arc<UOp>> = nodes.iter().filter(|u| matches!(u.op(), Op::Store(..))).collect();
    assert_eq!(stores.len(), 4 * 8, "every fragment register is stored once");
    for store in stores {
        let reg = reg_offset(store) % 8;
        assert_eq!(word_of(store), words[reg as usize / 2], "register {reg} takes word words[{}]", reg / 2);
    }
    assert!(!nodes.iter().any(|u| matches!(u.op(), Op::Range(..))), "the gather is flat");
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
