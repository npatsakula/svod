//! AMD GPU LLVM IR text generation.
//!
//! Composed against [`cpu::render_uop`](crate::llvm::cpu::render_uop) as the base: AMD-specific ops are
//! intercepted here, everything else (ALU, INDEX, LOAD, STORE, CAST, RANGE)
//! falls through to the CPU emitter unchanged.

pub mod ops;
pub mod tr;
pub mod wmma;

pub use ops::render_uop;

use std::collections::HashSet;
use std::sync::Arc;

use svod_ir::{AddrSpace, Op, UOp, ops as uops};

/// The unroll hint a loop carries: the AMDGPU target's own default threshold,
/// which also caps the boosts its unroll preferences stack on it, and the same
/// value as the loop's partial-unroll threshold, above the target's default.
///
/// The optimizer has already decided what to unroll. The boost that would fire
/// on its loops is the one per conditional branch on the loop's own index —
/// every gated load of a padded or concatenated operand — and it lifts a whole
/// reduce loop past the threshold into a spill. The deeper partial unroll is a
/// gain of its own, and the plans tuned under the hint rely on it.
pub const LOOP_HINT: &str = "!{!\"amdgpu.loop.unroll.threshold\", i32 300}";

/// The ranges whose counter indexes a register array, which [`LOOP_HINT`]
/// leaves alone: LLVM has to unroll such a loop before SROA can keep the array
/// in registers, and its private-memory boost exists for exactly that. tk's
/// register tiles are indexed this way — capped, they stay in scratch — while
/// the optimizer indexes its accumulators by constant.
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
