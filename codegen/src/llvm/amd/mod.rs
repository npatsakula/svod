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

/// The hint a loop over matrix-core products carries instead: it stays as
/// emitted. Such a loop is a tile pipeline's trip, whose author copies the
/// body per trip where that pays; LLVM's own partial unroll of it doubles
/// the live accumulators and staged operands at the copies' seam, which
/// the register allocator then moves between the copies and spills (a
/// 128×128×64 bf16 GEMM trip on gfx1201: 31 accumulator moves and 392
/// bytes of scratch a trip at 256 VGPRs; rolled, none at 253).
pub const ROLLED_HINT: &str = "!{!\"llvm.loop.unroll.disable\"}";

/// The ranges whose body issues a matrix-core product, every loop around
/// one: [`ROLLED_HINT`]'s.
pub fn product_ranges(nodes: &[Arc<UOp>]) -> HashSet<u64> {
    let (mut open, mut rolled) = (Vec::new(), HashSet::new());
    for node in nodes {
        match node.op() {
            Op::Range(..) => open.push(node.id),
            Op::End(uops::End { ranges, .. }) => open.retain(|id| ranges.iter().all(|r| r.id != *id)),
            Op::Wmma(..) => rolled.extend(open.iter().copied()),
            _ => {}
        }
    }
    rolled
}

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
