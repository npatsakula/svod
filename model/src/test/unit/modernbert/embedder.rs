//! Host-only tests of the batching every encoder head shares
//! ([`HeadPlans`](crate::encoder_head::HeadPlans)), through
//! [`ModernBertEmbedder`] over a tiny random-weight backbone: capacity
//! validation and the guards a batch meets before any plan compiles. None
//! compiles a plan — the numerics are the parity tests', the batched execution
//! the classifier's.

use svod_arch::pipelines::text::{EncoderHead, Encoding};

use crate::encoder_head::HeadError;
use crate::modernbert::{ModernBert, ModernBertEmbedder};
use crate::test::unit::encoder_head::encoding;
use crate::test::unit::modernbert::model::tiny_cfg;

const MAX_BATCH: usize = 4;
const MAX_SEQ: usize = 16;

/// An embedder over the tiny backbone. Construction compiles nothing: a plan
/// compiles on its shape's first batch, which these tests never reach.
fn embedder() -> ModernBertEmbedder {
    ModernBertEmbedder::new(ModernBert::empty(tiny_cfg()), MAX_BATCH, MAX_SEQ).expect("embedder")
}

/// A batch exceeding `max_batch` is rejected up front (CapacityExceeded), not
/// silently truncated or overflowed.
#[test]
fn batch_over_max_batch_is_rejected() {
    let encs = (1..=MAX_BATCH + 1).map(|i| encoding(&[i as u32], 0)).collect::<Vec<_>>();
    let refs: Vec<&Encoding> = encs.iter().collect();
    let err = embedder().run_batch(&refs, false).unwrap_err();
    assert!(matches!(err, HeadError::CapacityExceeded { .. }));
}

/// An encoding longer than `max_seq` is rejected (SequenceTooLong), not
/// silently truncated — the guard is reachable via the public chunk seam,
/// which lets a caller feed pre-built chunks that bypass the chunker's length
/// bound.
#[test]
fn sequence_over_max_seq_is_rejected() {
    let ids: Vec<u32> = (1..=MAX_SEQ as u32 + 1).collect();
    let err = embedder().run_batch(&[&encoding(&ids, 0)], false).unwrap_err();
    assert!(matches!(err, HeadError::SequenceTooLong { .. }));
}

/// An empty batch is a no-op returning no embeddings (mirrors the pipeline's
/// zero-chunk guard); profiled, it still returns the (empty) profile asked for.
#[test]
fn empty_batch_is_noop() {
    let mut emb = embedder();
    let (out, prof) = emb.run_batch(&[], false).expect("empty batch");
    assert!(out.is_empty());
    assert!(prof.is_none());

    let (out, prof) = emb.run_batch(&[], true).expect("empty batch profiled");
    assert!(out.is_empty());
    assert!(prof.is_some(), "profiled empty batch yields a (default) profile");
}

// ── capacity ─────────────────────────────────────────────────────────────────

/// `max_seq` past the checkpoint's positions, or a zero capacity, is refused at
/// construction rather than at the first oversized batch.
#[test_case::test_case(4, 129 => matches Err(HeadError::BeyondPositions { max_seq: 129, max_positions: 128, .. }); "past positions")]
#[test_case::test_case(0, 16 => matches Err(HeadError::EmptyCapacity { .. }); "zero batch")]
#[test_case::test_case(4, 0 => matches Err(HeadError::EmptyCapacity { .. }); "zero length")]
#[test_case::test_case(4, 128 => matches Ok((4, 128)); "all positions")]
fn embedder_capacity_is_validated(max_batch: usize, max_seq: usize) -> Result<(usize, usize), HeadError> {
    ModernBertEmbedder::new(ModernBert::empty(tiny_cfg()), max_batch, max_seq).map(|e| e.capacity())
}
