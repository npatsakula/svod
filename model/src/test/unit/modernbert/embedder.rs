//! Host-only tests of the batching every ModernBERT head shares
//! ([`HeadPlans`](crate::modernbert::head_jit::HeadPlans)), through
//! [`ModernBertEmbedder`] over a tiny random-weight backbone: capacity
//! validation, the guards a batch meets before any plan compiles, plan shapes
//! and input packing. None compiles a plan — the numerics are the parity
//! tests', the batched execution the classifier's.

use svod_arch::pipelines::text::{EncoderHead, Encoding};

use crate::modernbert::head_jit::plan_shape;
use crate::modernbert::packing::{pack_ids, pack_mask};
use crate::modernbert::{HeadError, ModernBert, ModernBertEmbedder};
use crate::test::unit::modernbert::encoding;
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

// ── input packing ───────────────────────────────────────────────────────────

/// Packing honors the per-row `len` stride, zero-fills pad positions and
/// unused rows, and truncates over-length inputs. The mask mirrors the ids'
/// real-token counts; only an unused row attends its first position, so a live
/// empty chunk stays fully masked and pools to the zero vector.
#[test]
fn pack_inputs_pad_stride_and_truncate_correctly() {
    let (rows, len) = (4, 4);
    // Stale contents from a previous batch must not survive the pack.
    let mut ids_v = vec![-1i64; rows * len];
    let mut mask_v = vec![-1i64; rows * len];

    // e1: 2 real tokens + 1 trailing pad (mask 0). Then a chunk with no token.
    // e2: 5 real tokens — one over len, so the last must be truncated. Row 3
    // left unused.
    let (e1, empty, e2) = (encoding(&[1, 2], 1), encoding(&[], 0), encoding(&[10, 20, 30, 40, 50], 0));
    let batch = [&e1, &empty, &e2];
    pack_ids(&mut ids_v, &batch, len);
    pack_mask(&mut mask_v, &batch, len);

    assert_eq!(ids_v, vec![1, 2, 0, 0, 0, 0, 0, 0, 10, 20, 30, 40, 0, 0, 0, 0]);
    assert_eq!(mask_v, vec![1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]);
}

// ── plan shapes and capacity ─────────────────────────────────────────────────

/// A batch runs at the smallest prepared shape that holds it: rows round up to
/// a power of two capped at `max_batch`, the length to the flash-attention tile.
#[test_case::test_case(3, 128, 8 => (4, 128); "rows round up, a full tile stays")]
#[test_case::test_case(5, 129, 6 => (6, 256); "rows cap at max_batch, length spills a tile")]
#[test_case::test_case(2, 0, 8 => (2, 128); "empty chunks still take a tile")]
fn plan_shape_rounds_rows_and_length(b: usize, longest: usize, max_batch: usize) -> (usize, usize) {
    plan_shape(b, longest, max_batch)
}

/// `max_seq` past the checkpoint's positions, or a zero capacity, is refused at
/// construction rather than at the first oversized batch.
#[test_case::test_case(4, 129 => matches Err(HeadError::BeyondPositions { max_seq: 129, max_positions: 128, .. }); "past positions")]
#[test_case::test_case(0, 16 => matches Err(HeadError::EmptyCapacity { .. }); "zero batch")]
#[test_case::test_case(4, 0 => matches Err(HeadError::EmptyCapacity { .. }); "zero length")]
#[test_case::test_case(4, 128 => matches Ok((4, 128)); "all positions")]
fn embedder_capacity_is_validated(max_batch: usize, max_seq: usize) -> Result<(usize, usize), HeadError> {
    ModernBertEmbedder::new(ModernBert::empty(tiny_cfg()), max_batch, max_seq).map(|e| e.capacity())
}
