//! Deterministic build-time fingerprint of a kernel's UOp graph.
//!
//! The tk LLVM render is **non-deterministic** run-to-run (the global node id leaks
//! into `%reduce_*`/`%wmma_*` SSA names and into the linearizer's tie-break), so a
//! kernel cannot be regression-tested by its IR text. The *graph*, however, is fully
//! deterministic: every [`UOp`] carries a recursive structural `content_hash`
//! (op-variant + dtype + op-data + child hashes, excluding the global id — the very
//! hash hash-consing dedups by, [`svod_ir`] `xxh64` seed 0). The hash also mixes in
//! a node's origin, a process-local scope id, so the graph is fingerprinted with its
//! origins cleared. This module exposes it as a stable [`KernelFingerprint`] — the
//! right oracle for proving a behavior-preserving refactor, for golden-testing a
//! kernel builder, and for keying a tune line another process reads.

use std::collections::BTreeMap;
use std::sync::Arc;

use svod_ir::UOp;

/// A deterministic structural fingerprint of a kernel SINK's UOp graph. Equal
/// [`Self::digest`] ⇒ structurally identical graphs (same ops, dtypes, edges, tags).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelFingerprint {
    /// High 64 bits: the origin-free SINK's recursive `content_hash` (the whole
    /// graph's structure). Low 64 bits: an order-independent fold of every node's
    /// `tag` — the one datum `content_hash` omits. Stable across builds/machines.
    pub digest: u128,
    /// Per-op-variant node count (keyed by the op's discriminant), for informative
    /// golden-mismatch diffs.
    pub op_counts: BTreeMap<String, u32>,
    /// Total unique nodes reachable from the SINK.
    pub node_count: usize,
}

/// Fingerprint the UOp graph rooted at `sink` (typically a [`crate::Kernel::finish`]
/// SINK). Pure and deterministic: invariant to the global id counter, to the origin
/// scopes the graph was built under, and to the non-deterministic render stage.
pub fn kernel_fingerprint(sink: &Arc<UOp>) -> KernelFingerprint {
    let sink = sink.without_origins();
    let nodes = sink.toposort();
    let mut tag_fold: u64 = 0;
    let mut op_counts: BTreeMap<String, u32> = BTreeMap::new();
    for u in &nodes {
        let tag = match u.tag() {
            None => 0u64,
            Some(v) => v.iter().fold(0x9E37_79B9_7F4A_7C15u64, |a, &x| mix64(a ^ x as u64)),
        };
        // Order-independent fold (wrapping-add commutes); toposort yields each
        // unique node once, so this is a faithful multiset over node tags.
        tag_fold = tag_fold.wrapping_add(mix64(tag));
        *op_counts.entry(format!("{:?}", std::mem::discriminant(u.op()))).or_default() += 1;
    }
    let digest = ((sink.content_hash as u128) << 64) | tag_fold as u128;
    KernelFingerprint { digest, op_counts, node_count: nodes.len() }
}

/// SplitMix64 finalizer — a stable, dependency-free integer mix.
fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
