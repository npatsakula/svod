//! AMD GPU LLVM IR text generation.
//!
//! Composed against [`cpu::render_uop`] as the base: AMD-specific ops are
//! intercepted here, everything else (ALU, INDEX, LOAD, STORE, CAST, RANGE)
//! falls through to the CPU emitter unchanged.

pub mod ops;
pub mod wmma;

pub use ops::render_uop;

use std::collections::HashSet;
use std::sync::Arc;

use svod_ir::{AddrSpace, Op, UOp, ops as uops};

/// The unroll hint a loop carries: the AMDGPU target's own default threshold,
/// which also caps the boosts its unroll preferences stack on it.
///
/// The optimizer has already decided what to unroll. The boost that fires on
/// its loops is the one per conditional branch on the loop's own index — every
/// gated load of a padded or concatenated operand — and it lifts a whole
/// reduce loop past the threshold. YOLO26-n's `neck.13.cv1` went from a 24-trip
/// loop at 96 VGPRs and 27 µs to 72 unrolled WMMAs, 843 spilled VGPRs and
/// 99 µs on gfx1201, and clang 20 then miscompiled the spill into NaN.
pub const LOOP_HINT: &str = "!{!\"amdgpu.loop.unroll.threshold\", i32 300}";

/// The ranges whose counter indexes a register array, which [`LOOP_HINT`]
/// leaves alone: LLVM has to unroll such a loop before SROA can keep the array
/// in registers, and its private-memory boost exists for exactly that. tk's
/// register tiles are indexed this way — capped, every tk convolution kept
/// 132 B of them in scratch — while the optimizer indexes its accumulators by
/// constant.
pub fn register_indexing_ranges(nodes: &[Arc<UOp>]) -> HashSet<u64> {
    nodes
        .iter()
        .filter_map(|node| match node.op() {
            Op::Index(uops::Index { buffer, indices, .. }) if buffer.addrspace() == Some(AddrSpace::Reg) => {
                Some(indices)
            }
            _ => None,
        })
        .flat_map(|indices| indices.iter().flat_map(|index| index.toposort()))
        .filter(|uop| matches!(uop.op(), Op::Range(..)))
        .map(|range| range.id)
        .collect()
}
