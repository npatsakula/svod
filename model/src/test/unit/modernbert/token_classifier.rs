//! Host-only tests for [`ModernBertTokenClassifier`] over a tiny random-weight
//! model (f32, CPU). No real checkpoint, no HF Hub — the model is
//! `ModernBertTokenClassificationModel::empty` with `fan_in_uniform` random
//! weights, enough to exercise the JIT prepare/pack/execute/read path and the
//! fused mask + per-token head semantics.

use std::sync::{LazyLock, Mutex, MutexGuard};

use svod_arch::pipelines::text::{ClassifyTokens, EncoderHead, Encoding, TokenClassification};

use crate::modernbert::{ModernBertTokenClassificationModel, ModernBertTokenClassifier};
use crate::test::unit::modernbert::max_abs_delta;
use crate::test::unit::modernbert::model::tiny_cfg;

/// Canonical prepared sizes for the shared token-classifier fixture. Compiling
/// the 2-layer CPU JIT graph takes ~60s, so every test runs against a plan
/// prepared once at these sizes; metadata-only assertions that used other sizes
/// are normalized onto them.
const MAX_BATCH: usize = 4;
const MAX_SEQ: usize = 16;

/// One JIT-compiled token classifier shared by every test in the module.
/// Prepared once per process (`LazyLock` init), then borrowed under a `Mutex`:
/// the only shared mutable state is the plan's input/output buffers, which each
/// `classify_tokens_batch` fully overwrites before reading back the live `b` rows — so
/// sharing is contamination-free. Every assertion here is a weight-agnostic
/// invariant (shape, finiteness, single-vs-batch consistency, mask invariance),
/// so a single random instance serves them all.
static RECOGNIZER: LazyLock<Mutex<ModernBertTokenClassifier>> = LazyLock::new(|| {
    let model = ModernBertTokenClassificationModel::empty(&tiny_cfg());
    Mutex::new(ModernBertTokenClassifier::new(model, MAX_BATCH, MAX_SEQ).expect("prepare token-classifier JIT"))
});

/// Borrow the shared prepared token classifier. Recovers from poison so a
/// panicking test doesn't cascade failures into its siblings — the buffers are
/// rewritten each call, so the post-poison state is still safe to reuse.
fn token_classifier() -> MutexGuard<'static, ModernBertTokenClassifier> {
    RECOGNIZER.lock().unwrap_or_else(|p| p.into_inner())
}

/// A consistent encoding: `n` real token ids (1..) with mask all-ones, plus
/// optional trailing pad positions at mask 0.
fn encoding(real_ids: &[u32], n_pad: usize) -> Encoding {
    let mut ids = real_ids.to_vec();
    let mut mask = vec![1u32; real_ids.len()];
    ids.extend(std::iter::repeat_n(&0u32, n_pad).copied());
    mask.extend(std::iter::repeat_n(&0u32, n_pad));
    let l = ids.len();
    Encoding {
        input_ids: ids,
        attention_mask: mask,
        token_type_ids: vec![0; l],
        offsets: (0..l).map(|i| (i, i + 1)).collect(),
        special_tokens_mask: vec![0; l],
    }
}

// ── shape / contract ───────────────────────────────────────────────────────

/// `classify_tokens_batch` returns one token-classification per input, each with a
/// `(seq_len, num_labels)` logit grid (padding excluded from `seq_len`), and all
/// finite.
#[test]
#[ignore = "heavy: 2-layer ModernBERT JIT graph compile through the CPU backend"]
fn classify_tokens_batch_shapes_and_finite() {
    let mut rec = token_classifier();
    let nl = rec.num_labels();
    let e1 = encoding(&[1, 2, 3], 0);
    let e2 = encoding(&[4, 5], 1);
    let (out, prof) = rec.run_batch(&[&e1, &e2], false).expect("classify_tokens_batch");
    assert_eq!(out.len(), 2);
    // e1: 3 real tokens → 3*num_labels logits; e2: 2 real + 1 pad → 3*num_labels.
    assert_eq!(out[0].logits.len(), 3 * nl);
    assert_eq!(out[1].logits.len(), 3 * nl);
    for TokenClassification { logits, num_labels } in &out {
        assert_eq!(*num_labels, nl);
        assert!(logits.iter().all(|v| v.is_finite()), "non-finite logit");
    }
    assert!(prof.is_none(), "unprofiled run yields no profile");
}

/// `num_labels()` reports the config's value.
#[test]
#[ignore = "heavy: 2-layer ModernBERT JIT graph compile through the CPU backend"]
fn num_labels_reported() {
    let rec = token_classifier();
    assert_eq!(rec.num_labels(), tiny_cfg().num_labels);
}

/// `capacity()` reports the prepared sizes.
#[test]
#[ignore = "heavy: 2-layer ModernBERT JIT graph compile through the CPU backend"]
fn capacity_reported() {
    let rec = token_classifier();
    assert_eq!(rec.capacity(), (MAX_BATCH, MAX_SEQ));
}

/// The trait-default `classify_tokens` (batch-of-one) agrees exactly with
/// `classify_tokens_batch(&[e])[0]`.
#[test]
#[ignore = "heavy: 2-layer ModernBERT JIT graph compile through the CPU backend"]
fn classify_tokens_single_matches_batch_of_one() {
    let mut rec = token_classifier();
    let e = encoding(&[1, 2, 3], 0);
    let single = rec.run(&e, false).unwrap().0;
    let batch = rec.run_batch(&[&e], false).unwrap().0;
    assert_eq!(single.logits, batch.into_iter().next().unwrap().logits);
}

/// A multi-row batch yields the same per-token logits as individual calls
/// (within fp tolerance) — the symbolic batch dim doesn't cross-contaminate.
#[test]
#[ignore = "heavy: 2-layer ModernBERT JIT graph compile through the CPU backend"]
fn batch_rows_match_single_calls() {
    let mut rec = token_classifier();
    let nl = rec.num_labels();
    let inputs = [encoding(&[1, 2, 3], 0), encoding(&[4, 5], 1), encoding(&[6, 7, 8, 9], 0)];
    let refs: Vec<Vec<f32>> = inputs
        .iter()
        .map(|e| {
            let (mut s, _) = rec.run_batch(&[e], false).unwrap();
            s.pop().unwrap().logits
        })
        .collect();
    let (batch, _) = rec.run_batch(&inputs.iter().collect::<Vec<_>>(), false).unwrap();
    for (got, want) in batch.iter().zip(&refs) {
        assert_eq!(got.logits.len() / nl, want.len() / nl, "seq_len mismatch");
        assert!(max_abs_delta(&got.logits, want) < 1e-4, "row differs from single call");
    }
}

/// An empty batch is a cheap no-op (no profile unless requested).
#[test]
#[ignore = "heavy: 2-layer ModernBERT JIT graph compile through the CPU backend"]
fn empty_batch_returns_empty() {
    let mut rec = token_classifier();
    let (out, prof) = rec.run_batch(&[], false).expect("empty batch");
    assert!(out.is_empty());
    assert!(prof.is_none());
    let (out, prof) = rec.run_batch(&[], true).expect("empty batch profiled");
    assert!(out.is_empty());
    assert!(prof.is_some(), "profiled empty run still returns a default profile");
}

/// A batch larger than the prepared `max_batch` is rejected.
#[test]
#[ignore = "heavy: 2-layer ModernBERT JIT graph compile through the CPU backend"]
fn capacity_exceeded_errors() {
    let mut rec = token_classifier();
    // One more than the prepared MAX_BATCH.
    let e = encoding(&[1, 2, 3], 0);
    let batch = std::iter::repeat_n(&e, MAX_BATCH + 1).collect::<Vec<_>>();
    let err = rec.run_batch(&batch, false).unwrap_err();
    assert!(err.to_string().contains("exceeds"), "{err}");
}

/// A profiled run emits a `classify_tokens` GPU stage.
#[test]
#[ignore = "heavy: 2-layer ModernBERT JIT graph compile through the CPU backend"]
fn profile_returned_when_requested() {
    let mut rec = token_classifier();
    let e = encoding(&[1, 2, 3], 0);
    let (_, prof) = rec.run_batch(&[&e], true).expect("profiled run");
    let prof = prof.expect("profile collected");
    assert!(prof.stage("classify_tokens").is_some(), "missing 'classify_tokens' stage");
}

/// Adding masked pad positions must not change the per-token logits of the real
/// tokens (the load-bearing mask property): same content, with vs without
/// trailing pad, agrees within fp tolerance on the real-token rows.
#[test]
#[ignore = "heavy: 2-layer ModernBERT JIT graph compile through the CPU backend"]
fn padding_with_correct_mask_is_invariant() {
    let mut rec = token_classifier();
    let nl = rec.num_labels();
    let real = 3;

    let (out_no, _) = rec.run_batch(&[&encoding(&[1, 2, 3], 0)], false).unwrap();
    let (out_pad, _) = rec.run_batch(&[&encoding(&[1, 2, 3], 2)], false).unwrap();

    assert_eq!(out_no[0].logits.len(), real * nl);
    assert_eq!(out_pad[0].logits.len(), (real + 2) * nl, "pad positions keep their own logits");
    let content_no = &out_no[0].logits[..real * nl];
    let content_pad = &out_pad[0].logits[..real * nl];
    assert!(max_abs_delta(content_no, content_pad) < 1e-3, "padding leaked into real-token logits");
}
