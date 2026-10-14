//! Host-only tests for [`ModernBertClassifier`] over a tiny random-weight model
//! (f32, CPU). No real checkpoint, no HF Hub — the model is
//! `ModernBertClassificationModel::empty` with `fan_in_uniform` random weights,
//! enough to exercise the batching every head shares on a compiled plan and the
//! CLS pooling the parity checkpoints never use.

use std::sync::{LazyLock, Mutex, MutexGuard};

use svod_arch::pipelines::text::{EncoderHead, Encoding};
use svod_tensor::nn::Module;

use crate::modernbert::{
    ClassifierHead, ModernBertClassificationModel, ModernBertClassifier, ModernBertConfig,
    ModernBertTokenClassificationModel, ModernBertTokenClassifier,
};
use crate::test::unit::modernbert::model::tiny_cfg;
use crate::test::unit::modernbert::{encoding, max_abs_delta};

const MAX_BATCH: usize = 4;
/// Two attention tiles, so a batch can run at either plan length.
const MAX_SEQ: usize = 256;

/// `tiny_cfg` (CLS pooling) with the positions for [`MAX_SEQ`].
fn cfg() -> ModernBertConfig {
    ModernBertConfig { max_position_embeddings: MAX_SEQ, ..tiny_cfg() }
}

/// One CLS classifier shared by the heavy tests, so each batch shape compiles
/// once per process. Each run repacks its plan's inputs in full, so sharing
/// carries nothing from one test to the next.
static CLASSIFIER: LazyLock<Mutex<ModernBertClassifier>> = LazyLock::new(|| {
    let model = ModernBertClassificationModel::empty(&cfg());
    Mutex::new(ModernBertClassifier::new(model, MAX_BATCH, MAX_SEQ).expect("classifier"))
});

/// Borrow the shared classifier. Recovers from poison so a panicking test
/// doesn't cascade failures into its siblings.
fn classifier() -> MutexGuard<'static, ModernBertClassifier> {
    CLASSIFIER.lock().unwrap_or_else(|p| p.into_inner())
}

// ── batching ───────────────────────────────────────────────────────────────

/// Each row of a batch matches its chunk run alone. Three chunks run on a plan
/// of four rows, the last unused, and come back as three; one chunk has no
/// token, and neither it nor its neighbours go astray.
#[test]
#[ignore = "heavy: 2-layer ModernBERT JIT graph compile through the CPU backend"]
fn batch_rows_match_single_calls() {
    let mut clf = classifier();
    let chunks = [encoding(&[1, 2, 3], 0), encoding(&[4, 5, 6, 7], 1), encoding(&[], 0)];
    let refs: Vec<&Encoding> = chunks.iter().collect();
    let batch = clf.run_batch(&refs, false).expect("classify_batch").0;
    assert_eq!(batch.len(), chunks.len(), "the unused row is dropped");
    for (i, (row, chunk)) in batch.iter().zip(&chunks).enumerate() {
        let alone = clf.run(chunk, false).expect("classify").0;
        let delta = max_abs_delta(&row.logits, &alone.logits);
        assert!(delta < 1e-4, "batch row {i} differs from its single call: max |delta| = {delta}");
    }
}

/// A chunk's logits do not depend on the plan length its batch runs at. Alone,
/// the short chunk runs at one attention tile; next to a chunk past the tile,
/// at two, its extra positions masked padding. The long chunk alone runs at
/// two tiles on one row, the row count the short chunk's plan already has.
#[test]
#[ignore = "heavy: 2-layer ModernBERT JIT graph compile through the CPU backend"]
fn rows_match_across_plan_lengths() {
    let mut clf = classifier();
    let long_ids: Vec<u32> = (0..130).map(|i| 1 + i % 60).collect();
    let (short, long) = (encoding(&[1, 2, 3], 0), encoding(&long_ids, 0));
    let short_alone = clf.run(&short, false).expect("short alone").0;
    let long_alone = clf.run(&long, false).expect("long alone").0;
    let batch = clf.run_batch(&[&short, &long], false).expect("short with long").0;
    for (name, row, alone) in [("short", &batch[0], short_alone), ("long", &batch[1], long_alone)] {
        let delta = max_abs_delta(&row.logits, &alone.logits);
        assert!(delta < 1e-4, "{name} chunk differs across plan lengths: max |delta| = {delta}");
    }
}

/// A profiled run returns a `classify` GPU stage; an unprofiled run on the same
/// classifier returns none (per-call, no rebuild).
#[test]
#[ignore = "heavy: 2-layer ModernBERT JIT graph compile through the CPU backend"]
fn profile_returned_when_requested() {
    let mut clf = classifier();
    let e = encoding(&[1, 2, 3], 0);
    let prof = clf.run_batch(&[&e], true).expect("profiled").1.expect("profile requested");
    assert!(prof.stage("classify").is_some(), "expected a 'classify' stage in {prof:?}");
    assert!(clf.run_batch(&[&e], false).expect("unprofiled").1.is_none(), "unprofiled yields no profile");
}

// ── head ───────────────────────────────────────────────────────────────────

/// CLS pooling reads the first position: on the same weights, the sequence
/// head's logits are the token head's at position 0.
#[test]
#[ignore = "heavy: 2-layer ModernBERT JIT graph compile through the CPU backend"]
fn cls_pooling_reads_the_first_position() {
    let cfg = cfg();
    let model = ModernBertClassificationModel::empty(&cfg);
    let tokens = ModernBertTokenClassificationModel::from_state_dict(&model.state_dict(""), &cfg).expect("token model");
    let mut clf = ModernBertClassifier::new(model, 1, MAX_SEQ).expect("classifier");
    let mut tok = ModernBertTokenClassifier::new(tokens, 1, MAX_SEQ).expect("token classifier");

    let e = encoding(&[1, 2, 3, 4], 0);
    let cls = clf.run(&e, false).expect("classify").0.logits;
    let per_token = tok.run(&e, false).expect("classify tokens").0.logits;
    let delta = max_abs_delta(&cls, &per_token[..cls.len()]);
    assert!(delta < 1e-4, "CLS logits differ from the first token's: max |delta| = {delta}");
}

/// `classifier_bias = true` builds a `head.dense.bias` for the checkpoint to
/// fill, and `false` none — every published checkpoint uses `false`, so the
/// parity tests never reach the `true` layout.
#[test]
fn dense_bias_follows_classifier_bias() {
    let biased = ModernBertConfig { classifier_bias: true, ..tiny_cfg() };
    assert!(ClassifierHead::empty(&biased).state_dict("").contains_key("head.dense.bias"));
    assert!(!ClassifierHead::empty(&tiny_cfg()).state_dict("").contains_key("head.dense.bias"));
}
