//! Tests for the position-aware elementwise primitives ([`Group::map_position`]
//! and [`Group::mask_where`]): arch-correct lane position computation, ternary
//! emission, and block-offset threading.

use std::sync::Arc;

use svod_dtype::{AmdArch, DType};
use svod_ir::{ConstValue, Op, UOp};

use crate::index::{Idx, cidx};
use crate::tiles::TileLayout;
use crate::{ArchCaps, Kernel};
use svod_ir::ops;

/// Whether the toposort contains a `BinaryOp::Mul` whose rhs is `Const(Int(val))`.
fn has_mul_by_const(topo: &[Arc<UOp>], val: i64) -> bool {
    topo.iter().any(|u| {
        let Op::Binary(svod_ir::BinaryOp::Mul, _, rhs) = u.op() else { return false };
        matches!(rhs.op(), Op::Const(c) if matches!(c.0, ConstValue::Int(v) if v == val))
    })
}

/// Whether the toposort contains a `BinaryOp::FloorMod` whose rhs is `Const(Int(val))`.
fn has_mod_by_const(topo: &[Arc<UOp>], val: i64) -> bool {
    topo.iter().any(|u| {
        let Op::Binary(svod_ir::BinaryOp::FloorMod, _, rhs) = u.op() else { return false };
        matches!(rhs.op(), Op::Const(c) if matches!(c.0, ConstValue::Int(v) if v == val))
    })
}

/// Build a kernel with a 16×16 Col accumulator tile, apply `map_position` that
/// adds the global `row` position (cast to f32) to each element, and return the
/// toposort of the resulting tile UOp.
fn map_position_topo(caps: ArchCaps, row_blk: Idx) -> Vec<Arc<UOp>> {
    let ker = Kernel::new("map_pos", [1, 1, 1], caps.wave_size as i64, vec![], caps);
    let warp = ker.warp();
    let tile = warp.zero(ker.acc((16, 16), TileLayout::Col));
    // Use both row and col so lane_row AND lane_col stay live in the toposort.
    let tile = warp.map_position(tile, row_blk, Idx::Const(0), |x, _idx, row, col| {
        x.add(&row.cast(DType::Float32)).add(&col.cast(DType::Float32))
    });
    tile.uop().toposort()
}

/// gfx942 (CDNA, wave64): `map_position` emits the contiguous-stride
/// `lane_rc` pattern — `row = (lane/16)*stride + inner` — so the graph carries
/// a `Mul` by the stride (4) and a `FloorMod` by 16 (the column), and NOT the
/// even/odd interleave `Mul` by 2.
#[test]
fn test_map_position_emits_gfx942_stride() {
    let topo = map_position_topo(ArchCaps::GFX942, Idx::Const(0));
    assert!(has_mul_by_const(&topo, 4), "gfx942: lane_rc stride (mul by 4)");
    assert!(has_mod_by_const(&topo, 16), "gfx942: lane_rc col (mod by 16)");
    assert!(!has_mul_by_const(&topo, 2), "gfx942: no RDNA interleave (mul by 2)");
}

/// gfx1151 (RDNA, wave32): `map_position` emits the even/odd interleave
/// `lane_rc` pattern — `row = 2*inner + lane/16` — so the graph carries a
/// `Mul` by 2 (the interleave factor) and NOT a `Mul` by 4 (the CDNA stride).
#[test]
fn test_map_position_emits_gfx1151_interleave() {
    let topo = map_position_topo(ArchCaps::for_amd(AmdArch::Gfx1151), Idx::Const(0));
    assert!(has_mul_by_const(&topo, 2), "gfx1151: interleave (mul by 2)");
    assert!(!has_mul_by_const(&topo, 4), "gfx1151: no CDNA stride (mul by 4)");
    assert!(has_mod_by_const(&topo, 16), "gfx1151: lane_rc col (mod by 16)");
}

/// `mask_where` emits exactly one `Op::Ternary` (the `where(pred, fill, x)`)
/// per element in the symbolic looped form.
#[test]
fn test_mask_where_emits_ternary() {
    let ker = Kernel::new("mask_where", [1, 1, 1], 64, vec![], ArchCaps::GFX942);
    let warp = ker.warp();
    let tile = warp.zero(ker.acc((16, 16), TileLayout::Col));
    let bound = cidx(8);
    let tile = warp.mask_where(tile, Idx::Const(0), Idx::Const(0), f64::NEG_INFINITY, move |row, _| row.ge(&bound));
    let topo = tile.uop().toposort();
    let ternaries = topo.iter().filter(|u| matches!(u.op(), Op::Ternary(..))).count();
    assert_eq!(ternaries, 1, "exactly one Ternary (the mask_where)");
}

/// A non-zero `row_blk` threads `row_blk * total_rows` into the per-element row
/// position; a zero `row_blk` does not reference the block index at all.
#[test]
fn test_map_position_block_offset() {
    let topo0 = map_position_topo(ArchCaps::GFX942, Idx::Const(0));
    assert!(
        !topo0.iter().any(|u| matches!(u.op(), Op::Special(ops::Special { name, .. }) if name.contains("gidx"))),
        "zero row_blk: no block index in the graph"
    );

    let ker = Kernel::new("map_pos_blk", [1, 1, 1], 64, vec![], ArchCaps::GFX942);
    let warp = ker.warp();
    let tile = warp.zero(ker.acc((16, 16), TileLayout::Col));
    let row_blk = Idx::Uop(ker.block_idx[0].clone());
    let tile = warp.map_position(tile, row_blk, Idx::Const(0), |x, _idx, row, col| {
        x.add(&row.cast(DType::Float32)).add(&col.cast(DType::Float32))
    });
    let topo = tile.uop().toposort();
    assert!(
        topo.iter().any(|u| matches!(u.op(), Op::Special(ops::Special { name, .. }) if name.contains("gidx0"))),
        "non-zero row_blk: block_idx[0] present"
    );
    assert!(has_mul_by_const(&topo, 16), "row_blk * total_rows (=16) product present");
}

/// Tinygrad TK anchors tile reads to every tracked enclosing range. Helper
/// ranges alone are insufficient: without this dependency a carried REG load
/// can be hoisted before the outer loop and observe only its initial value.
#[test]
fn tracked_loop_anchors_tile_reads_in_rolled_kernels() {
    let ker = Kernel::new("tracked_anchor", [1, 1, 1], 64, vec![], ArchCaps::GFX942);
    let warp = ker.warp();
    let tile = ker.acc((16, 16), TileLayout::Col);
    let lp = ker.loop_static(4);

    let anchored = warp.anchor(tile.uop());
    let Op::After(ops::After { passthrough, deps }) = anchored.op() else {
        panic!("tracked tile read must be wrapped in AFTER");
    };
    assert!(Arc::ptr_eq(passthrough, tile.uop()));
    assert!(deps.iter().any(|dep| Arc::ptr_eq(dep, lp.index())), "AFTER must depend on the enclosing tracked RANGE");
}
