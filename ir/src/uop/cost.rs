//! Cheap, symbolic cost estimates over a kernel AST.
//!
//! These walk the UOp graph (no execution) to approximate a kernel's compute
//! work. Used by the BEAM `least_compute_ops` bloat filter and by the runtime
//! profiler's roofline (GFLOP/s) column.

use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::op::Op;
use crate::ops;
use crate::uop::UOp;

/// Symbolic estimate of compute ops in a kernel.
///
/// Each ALU/Ternary/Reduce/WMMA node contributes `prod(enclosing-RANGE sizes)`
/// flops, a WMMA counting the MACs its shape performs. Symbolic RANGE ends
/// resolve to the midpoint of their `vmin`/`vmax` bounds (matching the
/// `(vmax+vmin)/2` choice in the BEAM timing path), so dynamic-shape kernels
/// participate in the `least_compute_ops*1000` bloat filter.
///
/// Only arithmetic that produces *values* counts. Address arithmetic — the
/// operands of an INDEX, and the bounds of a RANGE or SPECIAL — is skipped: a
/// scheduler-built AST barely has any, but a hand-lowered kernel body (a tk
/// tile kernel) is mostly index math, and counting it reported a matmul at
/// tens of times the hardware's peak.
///
/// `None` when no reliable count exists: the count overflowed, or the kernel
/// is hand-lowered.
pub fn compute_ops_estimate(uop: &Arc<UOp>) -> Option<u64> {
    // A hand-lowered kernel does its own addressing, and the nesting an op sits
    // in is then no longer recoverable from what its operands depend on: a tile
    // kernel's loop variables reach the arithmetic only through addresses, so
    // the fold below both misses real nesting and inherits unrelated ranges. It
    // is honest to report no count rather than one that came out tens of times
    // off.
    if let Op::Sink(ops::Sink { info: Some(info), .. }) = uop.op()
        && info.is_hand_lowered()
    {
        return None;
    }
    let topo = uop.toposort();
    let pos: FxHashMap<u64, usize> = topo.iter().enumerate().map(|(i, node)| (node.id, i)).collect();

    // One bit per loop-bound node — RANGE for ordinary loops, SPECIAL for
    // hardware-provided indices — plus its iteration count.
    let mut range_sizes: Vec<u64> = Vec::new();
    let mut range_bit: FxHashMap<u64, usize> = FxHashMap::default();
    for node in &topo {
        if let Op::Range(ops::Range { end, .. }) | Op::Special(ops::Special { end, .. }) = node.op() {
            range_bit.insert(node.id, range_sizes.len());
            range_sizes.push(range_size_estimate(end));
        }
    }

    // Which nodes carry values rather than addresses. Seeded with the root and
    // pushed down value edges only, so an INDEX's `indices` and a RANGE's `end`
    // do not make their operands count as compute. A node feeding both a value
    // and an address stays counted, which is the conservative direction.
    let mut is_value = vec![false; topo.len()];
    if let Some(last) = is_value.last_mut() {
        *last = true;
    }
    for (i, node) in topo.iter().enumerate().rev() {
        if !is_value[i] {
            continue;
        }
        value_children(node, |child| is_value[pos[&child.id]] = true);
    }

    // Bottom-up fold over the toposort: `enclosing[node]` is the union of its
    // children's enclosing ranges plus itself. Structurally the same
    // information tinygrad tracks with its `mult_stack` discipline.
    let words = range_sizes.len().div_ceil(64);
    let mut enclosing = vec![0u64; topo.len() * words];
    let mut flops: u64 = 0;
    for (i, node) in topo.iter().enumerate() {
        let (done, rest) = enclosing.split_at_mut(i * words);
        let mask = &mut rest[..words];
        node.op().map_child(|child| {
            let j = pos[&child.id]; // children precede parents in a toposort
            for (w, cw) in mask.iter_mut().zip(&done[j * words..(j + 1) * words]) {
                *w |= cw;
            }
        });
        if let Some(&bit) = range_bit.get(&node.id) {
            mask[bit / 64] |= 1 << (bit % 64);
        }

        // Each ALU/Reduce/WMMA accumulates `prod(enclosing range sizes)`.
        let per_op = match node.op() {
            _ if !is_value[i] => 0,
            Op::Binary(..) | Op::Unary(..) | Op::Ternary(..) | Op::Reduce(..) => 1,
            // A WMMA is one instruction but `2*M*N*K` MACs of real work.
            Op::Wmma(ops::Wmma { metadata, .. }) => {
                let (n, m, k) = metadata.dims;
                2u64.saturating_mul(n as u64).saturating_mul(m as u64).saturating_mul(k as u64)
            }
            _ => 0,
        };
        if per_op != 0 {
            let mut weight: u64 = 1;
            for (w, word) in mask.iter().enumerate() {
                let mut bits = *word;
                while bits != 0 {
                    let bit = bits.trailing_zeros() as usize;
                    weight = weight.saturating_mul(range_sizes[w * 64 + bit]);
                    bits &= bits - 1;
                }
            }
            flops = flops.saturating_add(weight.saturating_mul(per_op));
        }
    }
    (flops != u64::MAX).then_some(flops)
}

/// The children of `op` that carry values. Everything an INDEX uses to address
/// a buffer, and the bound of a RANGE or SPECIAL, is address arithmetic.
fn value_children(node: &Arc<UOp>, mut f: impl FnMut(&Arc<UOp>)) {
    match node.op() {
        Op::Index(ops::Index { buffer, .. }) => f(buffer),
        Op::Range(..) | Op::Special(..) => {}
        op => op.map_child(|child| f(child)),
    }
}

/// Estimate a RANGE end's iteration count.
///
/// Concrete `Const(Int)` ends use the value directly; everything else falls
/// back to the midpoint of the `end` UOp's symbolic `vmin`/`vmax` bounds, so
/// dynamic-shape ranges still contribute a representative number of flops.
fn range_size_estimate(end: &Arc<UOp>) -> u64 {
    if let Op::Const(cv) = end.op()
        && let Some(v) = cv.0.try_int()
    {
        return (v.max(1)) as u64;
    }
    let vmin = end.vmin().try_int().unwrap_or(1);
    let vmax = end.vmax().try_int().unwrap_or(vmin);
    (((vmin + vmax) / 2).max(1)) as u64
}

#[cfg(test)]
#[path = "../test/unit/uop/cost.rs"]
mod tests;
