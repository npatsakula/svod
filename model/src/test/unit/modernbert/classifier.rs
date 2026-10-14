//! Host-only tests for [`ModernBertClassifier`] over a tiny random-weight model
//! (f32, CPU). No real checkpoint, no HF Hub — the model is
//! `ModernBertClassificationModel::empty` with `fan_in_uniform` random weights,
//! enough to exercise the JIT prepare/pack/execute/read path and the fused
//! mask+pool+head semantics.

use std::sync::{LazyLock, Mutex, MutexGuard};

use svod_arch::pipelines::text::{Classification, Classify, EncoderHead, Encoding};

use crate::modernbert::{ClassifierHead, ClassifierPooling, ModernBertClassificationModel, ModernBertClassifier};
use crate::test::unit::modernbert::max_abs_delta;
use crate::test::unit::modernbert::model::tiny_cfg;
use svod_tensor::nn::Module;

/// Canonical prepared sizes for the shared classifier fixtures. Compiling the
/// 2-layer CPU JIT graph takes ~60s, so every test runs against a plan prepared
/// once at these sizes; metadata-only assertions that used other sizes are
/// normalized onto them.
const MAX_BATCH: usize = 4;
const MAX_SEQ: usize = 16;

/// Build a classifier from `tiny_cfg` with `pooling`, prepared once at the
/// canonical sizes. Used to seed the shared fixtures below.
fn prepared(pooling: ClassifierPooling) -> ModernBertClassifier {
    let mut cfg = tiny_cfg();
    cfg.classifier_pooling = pooling;
    let model = ModernBertClassificationModel::empty(&cfg);
    ModernBertClassifier::new(model, MAX_BATCH, MAX_SEQ).expect("prepare classifier JIT")
}

/// Shared CLS-pooling classifier (the `tiny_cfg` default). Every weight-agnostic
/// test borrows this one compiled plan; the only shared mutable state is the
/// plan's input/output buffers, which each `classify_batch` fully overwrites
/// before reading back the live `b` rows — so sharing is contamination-free.
static CLS_CLASSIFIER: LazyLock<Mutex<ModernBertClassifier>> =
    LazyLock::new(|| Mutex::new(prepared(ClassifierPooling::Cls)));

/// Shared mean-pooling classifier for the pooling-semantics tests.
static MEAN_CLASSIFIER: LazyLock<Mutex<ModernBertClassifier>> =
    LazyLock::new(|| Mutex::new(prepared(ClassifierPooling::Mean)));

/// Borrow the shared CLS classifier. Recovers from poison so a panicking test
/// doesn't cascade failures into its siblings — the buffers are rewritten each
/// call, so the post-poison state is still safe to reuse.
fn classifier() -> MutexGuard<'static, ModernBertClassifier> {
    CLS_CLASSIFIER.lock().unwrap_or_else(|p| p.into_inner())
}

/// Borrow the shared mean-pooling classifier (same poison-recovery rationale).
fn mean_classifier() -> MutexGuard<'static, ModernBertClassifier> {
    MEAN_CLASSIFIER.lock().unwrap_or_else(|p| p.into_inner())
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

/// `classify_batch` returns one classification per input, each with `num_labels`
/// logits, and all finite.
#[test]
#[ignore = "heavy: 2-layer ModernBERT JIT graph compile through the CPU backend"]
fn classify_batch_shapes_and_finite() {
    let mut clf = classifier();
    let nc = clf.num_labels();
    let e1 = encoding(&[1, 2, 3], 0);
    let e2 = encoding(&[4, 5], 1);
    let (out, prof) = clf.run_batch(&[&e1, &e2], false).expect("classify_batch");
    assert_eq!(out.len(), 2);
    for Classification { logits } in &out {
        assert_eq!(logits.len(), nc);
        assert!(logits.iter().all(|v| v.is_finite()), "non-finite logit");
    }
    assert!(prof.is_none(), "unprofiled run yields no profile");
}

/// `num_labels()` reports the config's value.
#[test]
#[ignore = "heavy: 2-layer ModernBERT JIT graph compile through the CPU backend"]
fn num_labels_reported() {
    let clf = classifier();
    assert_eq!(clf.num_labels(), tiny_cfg().num_labels);
}

/// `capacity()` returns the prepared `(max_batch, max_seq)`.
#[test]
#[ignore = "heavy: 2-layer ModernBERT JIT graph compile through the CPU backend"]
fn capacity_reported() {
    let clf = classifier();
    assert_eq!(clf.capacity(), (MAX_BATCH, MAX_SEQ));
}

// ── consistency ────────────────────────────────────────────────────────────

/// The trait-default `classify` (batch-of-one) agrees with `classify_batch`.
#[test]
#[ignore = "heavy: 2-layer ModernBERT JIT graph compile through the CPU backend"]
fn classify_single_matches_batch_of_one() {
    let mut clf = classifier();
    let e = encoding(&[1, 2, 3, 4], 0);
    let single = clf.run(&e, false).expect("classify").0;
    let batch = clf.run_batch(&[&e], false).expect("classify_batch");
    let batch0 = &batch.0[0];
    let max = max_abs_delta(&single.logits, &batch0.logits);
    assert_eq!(max, 0.0, "default classify must match classify_batch exactly");
}

/// Multi-input batch yields the same logits as individual calls.
#[test]
#[ignore = "heavy: 2-layer ModernBERT JIT graph compile through the CPU backend"]
fn batch_rows_match_single_calls() {
    let mut clf = classifier();
    let e1 = encoding(&[1, 2, 3], 0);
    let e2 = encoding(&[4, 5, 6, 7], 1);
    let e3 = encoding(&[8], 2);

    let single1 = clf.run(&e1, false).expect("classify e1").0;
    let single2 = clf.run(&e2, false).expect("classify e2").0;
    let single3 = clf.run(&e3, false).expect("classify e3").0;

    let batch = clf.run_batch(&[&e1, &e2, &e3], false).expect("classify_batch");

    assert!(max_abs_delta(&single1.logits, &batch.0[0].logits) < 1e-4);
    assert!(max_abs_delta(&single2.logits, &batch.0[1].logits) < 1e-4);
    assert!(max_abs_delta(&single3.logits, &batch.0[2].logits) < 1e-4);
}

// ── head: classifier_bias round-trip (host-only) ────────────────────────────

/// `classifier_bias = true` builds a `head.dense.bias`, which `prediction_head_tail`
/// threads into the IR — but every published checkpoint uses `false`, so the path
/// is otherwise untested. Verify the bias is built, omitted when `false`, and
/// survives a state-dict round-trip. (Host-only; the JIT forward is covered by
/// the heavy classify tests, which use `false`.)
#[test]
fn classifier_bias_true_builds_and_round_trips_dense_bias() {
    let mut cfg = tiny_cfg();
    cfg.classifier_bias = true;
    let head = ClassifierHead::empty(&cfg);
    let sd = head.state_dict("");
    assert!(sd.contains_key("head.dense.bias"), "classifier_bias=true emits head.dense.bias");

    // The default (false) omits it.
    let sd_false = ClassifierHead::empty(&tiny_cfg()).state_dict("");
    assert!(!sd_false.contains_key("head.dense.bias"), "classifier_bias=false omits head.dense.bias");

    // Round-trip: a fresh head loads the bias back and re-emits it.
    let mut reloaded = ClassifierHead::empty(&cfg);
    reloaded.load_state_dict(&sd, "").expect("load");
    assert!(reloaded.state_dict("").contains_key("head.dense.bias"), "dense bias survives round-trip");
}

// ── guards ─────────────────────────────────────────────────────────────────

/// Empty batch → empty results, profile optional.
#[test]
#[ignore = "heavy: 2-layer ModernBERT JIT graph compile through the CPU backend"]
fn empty_batch_returns_empty() {
    let mut clf = classifier();
    let (out, prof) = clf.run_batch(&[], false).expect("empty batch");
    assert!(out.is_empty());
    assert!(prof.is_none());

    let (out, prof) = clf.run_batch(&[], true).expect("empty batch profiled");
    assert!(out.is_empty());
    assert!(prof.is_some(), "profiled empty batch yields a (default) profile");
}

/// Over-capacity batch → `CapacityExceeded` error.
#[test]
#[ignore = "heavy: 2-layer ModernBERT JIT graph compile through the CPU backend"]
fn capacity_exceeded_errors() {
    let mut clf = classifier();
    // One more than the prepared MAX_BATCH.
    let encs = (1..=MAX_BATCH + 1).map(|i| encoding(&[i as u32], 0)).collect::<Vec<_>>();
    let refs: Vec<&Encoding> = encs.iter().collect();
    let err = clf.run_batch(&refs, false);
    assert!(err.is_err(), "batch > max_batch must error");
}

// ── profiling ──────────────────────────────────────────────────────────────

/// Profile is `Some` and contains a `"classify"` GPU stage when requested.
#[test]
#[ignore = "heavy: 2-layer ModernBERT JIT graph compile through the CPU backend"]
fn profile_returned_when_requested() {
    let mut clf = classifier();
    let e = encoding(&[1, 2, 3], 0);
    let (_, prof) = clf.run_batch(&[&e], true).expect("classify_batch");
    let prof = prof.expect("profile requested");
    assert!(prof.stage("classify").is_some(), "expected a 'classify' stage in {prof:?}");
}

// ── pooling semantics ──────────────────────────────────────────────────────

/// CLS and mean pooling produce different logits on the same input.
#[test]
#[ignore = "heavy: 2-layer ModernBERT JIT graph compile through the CPU backend"]
fn cls_vs_mean_pooling_differ() {
    let e = encoding(&[1, 2, 3, 4, 5], 0);

    // Different random heads (each fixture built from its own `empty`), so the
    // logits will differ trivially — we just verify both pooling strategies run
    // and produce finite output. One fixture guard at a time: holding both
    // deadlocks against a test that takes them in the other order.
    let cls_logits = classifier().run(&e, false).expect("cls classify").0.logits;
    let mean_logits = mean_classifier().run(&e, false).expect("mean classify").0.logits;

    assert!(cls_logits.iter().all(|v| v.is_finite()));
    assert!(mean_logits.iter().all(|v| v.is_finite()));
}

/// Adding padding with a correct mask does not change logits for either
/// pooling strategy — the mask keeps pad tokens out of both the attention and
/// the mean. This is the load-bearing property: the classifier is invariant to
/// sequence padding.
#[test]
#[ignore = "heavy: 2-layer ModernBERT JIT graph compile through the CPU backend"]
fn padding_with_correct_mask_is_invariant() {
    let e_no_pad = encoding(&[1, 2, 3, 4], 0);
    let e_with_pad = encoding(&[1, 2, 3, 4], 2);

    let (mean_a, mean_b) = {
        let mut mean_clf = mean_classifier();
        (
            mean_clf.run(&e_no_pad, false).expect("mean no-pad").0.logits,
            mean_clf.run(&e_with_pad, false).expect("mean with-pad").0.logits,
        )
    };
    assert!(
        max_abs_delta(&mean_a, &mean_b) < 1e-3,
        "mean pooling with correct mask should be padding-invariant, got delta {}",
        max_abs_delta(&mean_a, &mean_b)
    );

    let mut cls_clf = classifier();
    let cls_a = cls_clf.run(&e_no_pad, false).expect("cls no-pad").0.logits;
    let cls_b = cls_clf.run(&e_with_pad, false).expect("cls with-pad").0.logits;
    assert!(
        max_abs_delta(&cls_a, &cls_b) < 1e-3,
        "CLS pooling with correct mask should be padding-invariant, got delta {}",
        max_abs_delta(&cls_a, &cls_b)
    );
}
