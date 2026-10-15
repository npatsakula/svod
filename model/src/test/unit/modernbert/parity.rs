//! Parity against the PyTorch reference (`answerdotai/ModernBERT-base` and its
//! SST-2 / CoNLL-2003 fine-tunes). Heavy: each tier loads the real checkpoint
//! (fetched from HF Hub on first run) plus a golden produced by HuggingFace
//! `transformers`.
//!
//! # Goldens are not committed
//!
//! The `golden*.safetensors` fixtures are generated locally and are **not** on
//! HF Hub — only the model weights / configs are. Generate them before running
//! the `--ignored` tier (output lands in `data/modernbert/`, overridable via
//! `SVOD_MODERNBERT`):
//!
//! ```text
//! uv run model/scripts/generate_modernbert_golden.py            // backbone + MLM + embedder
//! uv run model/scripts/generate_modernbert_classifier_golden.py // SST-2 classifier
//! uv run model/scripts/generate_modernbert_token_golden.py      // CoNLL-2003 NER
//! ```
//!
//! Runs in **f32** (config dtype overridden) so it works on CPU backends
//! without GPU bf16 transcendentals. bf16 numerical parity is implied by the
//! framework's f32-accumulator guarantees in layernorm/matmul/attention.

use std::path::PathBuf;

use svod_arch::pipelines::text::{Classify, ClassifyTokens, EncoderHead, Encoding};

use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::encoder_head::HeadOptions;
use crate::modernbert::{
    ModernBert, ModernBertClassifier, ModernBertConfig, ModernBertEmbedder, ModernBertForMaskedLm,
    ModernBertTokenClassifier,
};
use crate::state::StateDict;
use crate::test::unit::encoder_head::{assert_tracks_pytorch_bf16, golden_encodings};
use crate::test::unit::{load_golden_vec, max_abs_delta, nan_max};

/// `answerdotai/ModernBERT-base`, at the revision the goldens were generated
/// from (each repo's pin is also its generate script's `--revision` default).
const BASE_REPO: (&str, &str) = ("answerdotai/ModernBERT-base", "8949b909ec900327062f0ebf497f51aef5e6f0c8");

/// [`encoder_head::local_or_hub`](crate::test::unit::encoder_head::local_or_hub)
/// over `SVOD_MODERNBERT` / `data/modernbert/`.
fn local_or_hub(repo: (&str, &str), name: &str) -> PathBuf {
    crate::test::unit::encoder_head::local_or_hub("SVOD_MODERNBERT", "modernbert", repo, name)
}

/// [`local_or_hub`] for `answerdotai/ModernBERT-base` (`golden.safetensors` is
/// the output of `model/scripts/generate_modernbert_golden.py`).
fn real_file(name: &str) -> PathBuf {
    local_or_hub(BASE_REPO, name)
}

/// Load the model + golden fixture once, returning (model, ids, mask, want).
/// `want` is `(T, D)` row-major (batch squeezed by the generator); `mask` is
/// the per-token attention mask (1 = real, 0 = pad).
fn load_fixture() -> (ModernBert, Tensor, Vec<i64>, Vec<f32>) {
    let weights = real_file("model.safetensors");
    let golden = crate::state::load_safetensors(&real_file("golden.safetensors")).expect("golden");

    let cfg_path = real_file("config.json");
    let mut cfg = ModernBertConfig::from_json(&cfg_path).expect("parse config.json");
    cfg.dtype = DType::Float32;

    let model = ModernBert::from_safetensors(&weights, cfg).expect("load weights");

    let input_ids: Vec<i64> = load_golden_vec(&golden, "input_ids");
    let want: Vec<f32> = load_golden_vec(&golden, "last_hidden_state");
    let mask: Vec<i64> = golden
        .get("attention_mask")
        .map(|_| load_golden_vec(&golden, "attention_mask"))
        .unwrap_or_else(|| vec![1; input_ids.len()]);
    let (b, l) = match golden.get("input_ids_shape") {
        Some(t) => {
            let t = t.clone();
            t.realize().unwrap();
            let s = t.as_vec::<i64>().unwrap();
            (s[0] as usize, s[1] as usize)
        }
        None => (1, input_ids.len()),
    };
    let ids = Tensor::from_slice(input_ids).try_reshape([b as isize, l as isize]).unwrap();

    (model, ids, mask, want)
}

/// Run the model with `mask` (bool, true=real) and return the flat (B, L, D) f32 output.
fn run_forward(model: &ModernBert, ids: &Tensor, mask: Option<&Tensor>) -> Vec<f32> {
    let out = model.forward(ids, mask).expect("forward");
    out.realize().expect("realize output");
    out.as_vec::<f32>().expect("output readout")
}

/// Max |delta| over the REAL-token positions only (where `mask == 1`), folded
/// across the hidden dim. Pad positions are excluded: transformers' pad outputs
/// are an artifact (the mask zeroes their attention) and a divergence there is
/// not a model-correctness signal.
fn real_token_max_delta(got: &[f32], want: &[f32], mask: &[i64], d: usize) -> f32 {
    got.chunks_exact(d)
        .zip(want.chunks_exact(d))
        .zip(mask.iter())
        .filter(|&(_, m)| *m == 1)
        .map(|((g, w), _)| max_abs_delta(g, w))
        .fold(0.0, nan_max)
}

/// `last_hidden_state` parity: our backbone (f32) vs the PyTorch reference.
/// Compares **real-token positions only** — the mask is load-bearing.
#[test]
#[ignore = "heavy: real ModernBERT-base weights + PyTorch golden (local or HF Hub download)"]
fn last_hidden_state_matches_pytorch() {
    let (model, ids, mask, want) = load_fixture();
    let d = want.len() / mask.len();

    let mask_t = Tensor::from_slice(mask.clone()).cast(DType::Bool).try_reshape([1isize, mask.len() as isize]).unwrap();
    let got = run_forward(&model, &ids, Some(&mask_t));

    let real_max = real_token_max_delta(&got, &want, &mask, d);
    eprintln!("real-token max |delta| = {real_max:.3e}");
    assert!(real_max < 1e-3, "real-token last_hidden_state drifted from PyTorch golden: max |delta| = {real_max}");
}

/// Control: the mask must be load-bearing. Running with the correct mask matches
/// the golden on real tokens (above); running with an all-ones mask (ignoring
/// padding) must DIVERGE on real tokens — the 20 pad tokens would otherwise
/// contaminate every real token's attention. If this test ever passes, the
/// golden is no longer exercising the mask.
#[test]
#[ignore = "heavy: real ModernBERT-base weights + PyTorch golden (local or HF Hub download)"]
fn ignoring_padding_diverges_from_golden() {
    let (model, ids, mask, want) = load_fixture();
    let d = want.len() / mask.len();

    // All-ones mask: attend to every position including padding.
    let all_ones = Tensor::from_slice(vec![1i64; mask.len()])
        .cast(DType::Bool)
        .try_reshape([1isize, mask.len() as isize])
        .unwrap();
    let got_unmasked = run_forward(&model, &ids, Some(&all_ones));
    let real_max_unmasked = real_token_max_delta(&got_unmasked, &want, &mask, d);

    // The golden was produced WITH the mask; ignoring padding must diverge on
    // real tokens by orders of magnitude more than the masked run (1e-3 bound).
    assert!(
        real_max_unmasked > 1e-2,
        "ignoring padding did NOT diverge from the golden on real tokens \
         (max |delta| = {real_max_unmasked:.3e}); the mask is not load-bearing in the golden"
    );
    eprintln!("unmasked real-token max |delta| = {real_max_unmasked:.3e} (diverges as expected)");
}

/// Load the MLM model (backbone + head) from the same weights + config as the
/// backbone fixture, plus the golden `mlm_logits`.
fn load_mlm_fixture() -> (ModernBertForMaskedLm, Tensor, Vec<i64>, Vec<f32>) {
    let weights = real_file("model.safetensors");
    let golden = crate::state::load_safetensors(&real_file("golden.safetensors")).expect("golden");

    let cfg_path = real_file("config.json");
    let mut cfg = ModernBertConfig::from_json(&cfg_path).expect("parse config.json");
    cfg.dtype = DType::Float32;

    let model = ModernBertForMaskedLm::from_safetensors(&weights, cfg).expect("load MLM weights");

    let input_ids: Vec<i64> = load_golden_vec(&golden, "input_ids");
    let want: Vec<f32> = load_golden_vec(&golden, "mlm_logits");
    let mask: Vec<i64> = golden
        .get("attention_mask")
        .map(|_| load_golden_vec(&golden, "attention_mask"))
        .unwrap_or_else(|| vec![1; input_ids.len()]);
    let (b, l) = match golden.get("input_ids_shape") {
        Some(t) => {
            let t = t.clone();
            t.realize().unwrap();
            let s = t.as_vec::<i64>().unwrap();
            (s[0] as usize, s[1] as usize)
        }
        None => (1, input_ids.len()),
    };
    let ids = Tensor::from_slice(input_ids).try_reshape([b as isize, l as isize]).unwrap();

    (model, ids, mask, want)
}

/// MLM-logits parity: our backbone + MLM head (f32) vs `AutoModelForMaskedLM`.
/// Compares **real-token positions only** (the mask is load-bearing), folding
/// across the vocab axis. The head reuses the f32-accumulator guarantees of
/// matmul/layernorm/GELU, so the same sub-1e-2 regime as the backbone applies.
#[test]
#[ignore = "heavy: real ModernBERT-base weights + PyTorch golden (local or HF Hub download)"]
fn mlm_logits_match_pytorch() {
    let (model, ids, mask, want) = load_mlm_fixture();
    let v = want.len() / mask.len();

    let mask_t = Tensor::from_slice(mask.clone()).cast(DType::Bool).try_reshape([1isize, mask.len() as isize]).unwrap();
    let got = model.forward(&ids, Some(&mask_t)).expect("MLM forward");
    got.realize().expect("realize logits");
    let got = got.as_vec::<f32>().expect("logits readout");

    let real_max = real_token_max_delta(&got, &want, &mask, v);
    eprintln!("MLM real-token max |delta| = {real_max:.3e}");
    assert!(real_max < 1e-2, "MLM logits drifted from PyTorch golden: real-token max |delta| = {real_max}");
}

/// Per-row real-token drift of `got` from the f32 golden `want`: each row's
/// mean |Δ| and the 95th percentile of its tokens' own mean |Δ|, over the
/// positions `mask` marks real.
fn row_drift(got: &[f32], want: &[f32], mask: &[i64], l: usize, d: usize) -> Vec<(f32, f32)> {
    let rows = mask.len() / l;
    (0..rows)
        .map(|row| {
            let mut tokens: Vec<f32> = (row * l..(row + 1) * l)
                .filter(|&t| mask[t] == 1)
                .map(|t| {
                    let deltas =
                        got[t * d..(t + 1) * d].iter().zip(&want[t * d..(t + 1) * d]).map(|(a, e)| (a - e).abs());
                    deltas.sum::<f32>() / d as f32
                })
                .collect();
            let mean = tokens.iter().sum::<f32>() / tokens.len() as f32;
            tokens.sort_by(f32::total_cmp);
            (mean, tokens[(tokens.len() - 1) * 95 / 100])
        })
        .collect()
}

/// The 16-bit backbone against the f32 golden, bounded by PyTorch's own bf16
/// drift on the same rows (`generate_modernbert_golden.py --long --long-length L`: a
/// full row and a 27-token one right-padded to `L`; 500 pads the attention to its
/// tile and leaves the GEMMs off theirs, 512 is on both). Four paths: the eager
/// forward and the JIT with its batch pinned (flash attention, the local layers
/// banded, tk's GEMMs where the rows tile), the JIT with its batch free (SDPA,
/// the generic GEMMs), and the full row alone with no mask (the attention masks
/// its own tile padding, if any). Each row's mean drift, and its 95th-percentile
/// token, stays within a quarter over the worst of PyTorch's bf16 paths on that
/// row — SDPA and eager on the CPU and on CUDA, batched and each row alone. They
/// differ by rounding alone, yet on the full row at 512 one lands at 1.61×
/// another's p95, so a single run is no yardstick; svod's own tile and kernel
/// choices spread as wide and reach PyTorch's worst, and a seventh draw from one
/// spread passes six draws' worst one time in seven — hence the quarter. Not the
/// single worst token either: a few tokens amplify bf16 rounding chaotically
/// (0.8 mean |Δ| on one of the 512), and which ones blow up differs by path. The
/// control — the same forward ignoring the padding — must break the short row's
/// bound, or the bound proves nothing.
#[test_case::test_case(500; "ragged")]
#[test_case::test_case(512; "tile aligned")]
#[ignore = "heavy: real ModernBERT-base weights + the long PyTorch golden, on a bf16 GPU"]
fn bf16_drift_tracks_pytorch_bf16(length: usize) {
    use crate::jit::InputSpec;
    use crate::modernbert::ModernBertJit;

    let dtype = crate::default_compute_dtype();
    if dtype != DType::BFloat16 {
        eprintln!("skip bf16_drift_tracks_pytorch_bf16: the default device computes in {dtype:?}");
        return;
    }
    let golden =
        crate::state::load_safetensors(&real_file(&format!("golden_long_{length}.safetensors"))).expect("golden_long");
    let (ids, mask): (Vec<i64>, Vec<i64>) =
        (load_golden_vec(&golden, "input_ids"), load_golden_vec(&golden, "attention_mask"));
    let want: Vec<f32> = load_golden_vec(&golden, "last_hidden_state");
    let (b, l) = (2usize, ids.len() / 2);
    let d = want.len() / ids.len();
    let mut pytorch: Vec<&String> = golden.keys().filter(|k| k.starts_with("last_hidden_state_bf16.")).collect();
    pytorch.sort();
    assert!(
        !pytorch.is_empty(),
        "golden_long_{length} predates PyTorch's bf16 paths: rerun generate_modernbert_golden.py --long"
    );

    let mut cfg = ModernBertConfig::from_json(&real_file("config.json")).expect("parse config.json");
    (cfg.dtype, cfg.max_batch_size) = (dtype, b);
    let model = ModernBert::from_safetensors(&real_file("model.safetensors"), cfg).expect("load weights");

    let ids_t = Tensor::from_slice(ids.clone()).try_reshape([b as isize, l as isize]).unwrap();
    let eager = |mask: &[i64]| -> Vec<f32> {
        let mask_t = Tensor::from_slice(mask).try_reshape([b as isize, l as isize]).unwrap();
        let out = model.forward(&ids_t, Some(&mask_t)).expect("forward").cast(DType::Float32);
        out.realize().expect("realize");
        out.as_vec::<f32>().expect("read")
    };
    let jit = |pinned: bool| -> Vec<f32> {
        let mut jit = ModernBertJit::new(model.clone());
        if pinned {
            jit = jit.with_b_fixed(b);
        }
        jit.prepare(InputSpec::i64(&[b, l]), InputSpec::i64(&[b, l])).expect("prepare");
        jit.input_ids_mut().unwrap().copyin(bytemuck::cast_slice(&ids)).unwrap();
        jit.attention_mask_mut().unwrap().copyin(bytemuck::cast_slice(&mask)).unwrap();
        jit.execute_bound(b as i64).expect("execute");
        // bf16 is the top half of an f32.
        let mut raw = vec![0u8; want.len() * 2];
        jit.hidden().expect("hidden").copyout_prefix(&mut raw).expect("read");
        raw.as_chunks::<2>().0.iter().map(|&h| f32::from_bits(u32::from(u16::from_le_bytes(h)) << 16)).collect()
    };

    let mut bound = vec![(0f32, 0f32); b];
    for key in pytorch {
        let drift = row_drift(&load_golden_vec(&golden, key), &want, &mask, l, d);
        eprintln!("pytorch {} per row (mean, p95 token) |Δ|: {drift:.3?}", &key["last_hidden_state_bf16.".len()..]);
        for (edge, row) in bound.iter_mut().zip(drift) {
            *edge = (edge.0.max(row.0), edge.1.max(row.1));
        }
    }
    eprintln!("bound, pytorch's worst per row (mean, p95 token) |Δ|: {bound:.3?}");
    let within = |drift: &[(f32, f32)]| drift.iter().zip(&bound).all(|(r, p)| r.0 <= 1.25 * p.0 && r.1 <= 1.25 * p.1);
    let mut paths: Vec<(&str, Vec<(f32, f32)>)> = Vec::new();
    for (name, got) in [("eager", eager(&mask)), ("jit pinned", jit(true)), ("jit free batch", jit(false))] {
        assert!(got.iter().all(|x| x.is_finite()), "{name}: non-finite output");
        paths.push((name, row_drift(&got, &want, &mask, l, d)));
    }
    // The full row alone and unmasked: the attention masks its own tile padding.
    let row = Tensor::from_slice(&ids[..l]).try_reshape([1, l as isize]).unwrap();
    let alone = model.forward(&row, None).expect("forward").cast(DType::Float32);
    alone.realize().expect("realize");
    let alone = row_drift(&alone.as_vec::<f32>().expect("read"), &want[..l * d], &mask[..l], l, d);
    paths.push(("eager, row 0 alone, no mask", [alone[0], bound[1]].to_vec()));
    for (name, drift) in &paths {
        eprintln!("svod {name} per row (mean, p95 token) |Δ|: {drift:.3?}");
    }
    for (name, drift) in &paths {
        assert!(within(drift), "svod {name} drifts past PyTorch's bf16: {drift:?} against {bound:?}");
    }
    let unmasked = row_drift(&eager(&vec![1; b * l]), &want, &mask, l, d);
    eprintln!("control, padding ignored, per row (mean, p95 token) |Δ|: {unmasked:.3?}");
    assert!(!within(&unmasked), "ignoring the padding stays within the bound: it cannot catch a mask bug");
}

/// The embedder at `dtype` over the golden's one row, and the golden.
fn load_embed_fixture(dtype: DType) -> (ModernBertEmbedder, Encoding, StateDict) {
    let golden = crate::state::load_safetensors(&real_file("golden.safetensors")).expect("golden");
    let enc = golden_encodings(&golden).pop().expect("one row");
    let opts = HeadOptions { max_seq: Some(enc.len()), dtype: Some(dtype), ..HeadOptions::default() };
    let embedder = ModernBertEmbedder::from_files(&real_file("config.json"), &real_file("model.safetensors"), &opts)
        .expect("embedder");
    (embedder, enc, golden)
}

/// Embedding-pipeline parity: Svod's fused backbone+masked-mean-pool+L2-norm JIT
/// (`ModernBertEmbedder`) vs the PyTorch reference (`expected_embedding` from the
/// golden, produced by `transformers` + the same pooling recipe). Validates the
/// full embed path — not just the backbone forward — against an independent
/// reference. The 1e-12 denominator/norm EPS Svod adds is negligible here.
#[test]
#[ignore = "heavy: real ModernBERT-base weights + PyTorch golden (local or HF Hub download)"]
fn embeddings_match_pytorch() {
    let (mut embedder, enc, golden) = load_embed_fixture(DType::Float32);
    let want: Vec<f32> = load_golden_vec(&golden, "expected_embedding");
    let (got, _prof) = embedder.run_batch(&[&enc], false).expect("embed");
    let got = &got[0].values;
    assert_eq!(got.len(), want.len(), "embedding dim mismatch");

    let max_delta = max_abs_delta(got, &want);
    eprintln!("embedding max |delta| = {max_delta:.3e}");
    assert!(max_delta < 1e-3, "embedding drifted from PyTorch golden: max |delta| = {max_delta}");
}

// ── classifier parity (SST-2 fine-tuned checkpoint) ─────────────────────────
//
// Tests the fused backbone + classification head (pool → dense → GELU → norm →
// classifier) against `AutoModelForSequenceClassification` from `transformers`.
// The SST-2 model uses mean pooling and `classifier_bias = false` — exercising
// the masked-mean path and the optional-bias code.

const CLASSIFIER_REPO: (&str, &str) =
    ("AnkitAI/Sensible-ModernBERT-Sentiment-Analysis", "df8a40e2ea5911d7c8d52cced4ad60048ae63490");

/// [`local_or_hub`] for the SST-2 fine-tuned repo.
fn classifier_file(name: &str) -> PathBuf {
    local_or_hub(CLASSIFIER_REPO, name)
}

/// A head sized to the golden's padded rows, computing in `dtype`.
fn fixture_options(encodings: &[Encoding], dtype: DType) -> HeadOptions {
    HeadOptions {
        max_batch: encodings.len(),
        max_seq: Some(encodings[0].len()),
        dtype: Some(dtype),
        ..HeadOptions::default()
    }
}

/// The classifier at `dtype` over the golden's rows, and the golden. The plan
/// matches the golden's padding (batch and length from its `input_ids`).
fn load_classifier_fixture(dtype: DType) -> (ModernBertClassifier, Vec<Encoding>, StateDict) {
    let golden = crate::state::load_safetensors(&classifier_file("golden_classifier.safetensors")).expect("golden");
    let encodings = golden_encodings(&golden);
    let opts = fixture_options(&encodings, dtype);
    let classifier =
        ModernBertClassifier::from_files(&classifier_file("config.json"), &classifier_file("model.safetensors"), &opts)
            .expect("build classifier");
    (classifier, encodings, golden)
}

/// Classification parity: Svod's fused backbone+head JIT (`ModernBertClassifier`)
/// vs the PyTorch reference (`expected_logits` from the golden, produced by
/// `transformers` `AutoModelForSequenceClassification`). The SST-2 model uses
/// mean pooling with `classifier_bias = false`.
#[test]
#[ignore = "heavy: real SST-2 ModernBERT-base classifier weights + PyTorch golden (local or HF Hub download)"]
fn classify_logits_match_pytorch() {
    let (mut classifier, encodings, golden) = load_classifier_fixture(DType::Float32);
    let want: Vec<f32> = load_golden_vec(&golden, "expected_logits");
    let num_labels = classifier.num_labels();

    let refs: Vec<&Encoding> = encodings.iter().collect();
    let (classifications, _prof) = classifier.run_batch(&refs, false).expect("classify");

    assert_eq!(classifications.len(), encodings.len(), "batch size mismatch");
    for (i, c) in classifications.iter().enumerate() {
        let expected = &want[i * num_labels..(i + 1) * num_labels];
        let max_delta = max_abs_delta(&c.logits, expected);
        eprintln!("[{i}] logits max |delta| = {max_delta:.3e}  got={:.4?}  want={:.4?}", c.logits, expected);
        assert!(max_delta < 1e-3, "classification logits [{i}] drifted from PyTorch golden: max |delta| = {max_delta}");
    }
}

/// Negative control: an all-ones mask (ignoring padding) must DIVERGE from the
/// golden — both attention and mean-pooling are contaminated by pad tokens. If
/// this passes, the golden isn't exercising the mask.
#[test]
#[ignore = "heavy: real SST-2 ModernBERT-base classifier weights + PyTorch golden (local or HF Hub download)"]
fn ignoring_padding_diverges_in_classification() {
    let (mut classifier, encodings, golden) = load_classifier_fixture(DType::Float32);
    let want: Vec<f32> = load_golden_vec(&golden, "expected_logits");
    let num_labels = classifier.num_labels();

    let unmasked: Vec<Encoding> = encodings
        .iter()
        .map(|e| {
            let mut e = e.clone();
            e.attention_mask.fill(1);
            e
        })
        .collect();

    let refs: Vec<&Encoding> = unmasked.iter().collect();
    let (classifications, _prof) = classifier.run_batch(&refs, false).expect("classify unmasked");

    for (i, c) in classifications.iter().enumerate() {
        let expected = &want[i * num_labels..(i + 1) * num_labels];
        let max_delta = max_abs_delta(&c.logits, expected);
        assert!(
            max_delta > 1e-2,
            "ignoring padding did NOT diverge from the golden [{i}] (max |delta| = {max_delta:.3e})"
        );
        eprintln!("[{i}] unmasked logits max |delta| = {max_delta:.3e} (diverges as expected)");
    }
}

// ─── token classification parity ───────────────────────────────────────────
//
// Tests the fused backbone + token head (`ClassifierHead::forward` over the full
// `(B, L, D)` state, no pooling) against `AutoModelForTokenClassification` from
// `transformers`. The CoNLL-2003 NER fine-tune has no pooling; `classifier_bias`
// is exercised by the always-present `classifier.bias`.

const TOKEN_REPO: (&str, &str) =
    ("sanketrai/modernbert-base-conll2003-english-ner", "6e6500c22245f5f17a2dd3a427577715625c3d9d");

/// [`local_or_hub`] for the CoNLL-2003 NER fine-tuned repo.
fn token_file(name: &str) -> PathBuf {
    local_or_hub(TOKEN_REPO, name)
}

/// The token classifier at `dtype` over the golden's rows, and the golden.
/// The plan matches the golden's padding (batch and length from its
/// `input_ids`).
fn load_token_fixture(dtype: DType) -> (ModernBertTokenClassifier, Vec<Encoding>, StateDict) {
    let golden = crate::state::load_safetensors(&token_file("golden_token.safetensors")).expect("golden");
    let encodings = golden_encodings(&golden);
    let opts = fixture_options(&encodings, dtype);
    let classifier =
        ModernBertTokenClassifier::from_files(&token_file("config.json"), &token_file("model.safetensors"), &opts)
            .expect("build token classifier");
    (classifier, encodings, golden)
}

/// Token-classification parity: Svod's fused backbone + per-token head JIT
/// (`ModernBertTokenClassifier`) vs the PyTorch reference (`expected_logits`
/// from the golden, produced by `transformers`
/// `AutoModelForTokenClassification`). Compares real-token positions only (mask
/// = 1); pad positions are a don't-care.
#[test]
#[ignore = "heavy: real CoNLL-2003 ModernBERT-base NER weights + PyTorch golden (local or HF Hub download)"]
fn token_logits_match_pytorch() {
    let (mut recognizer, encodings, golden) = load_token_fixture(DType::Float32);
    let want: Vec<f32> = load_golden_vec(&golden, "expected_logits");
    let (num_labels, seq_len) = (recognizer.num_labels(), encodings[0].len());

    let refs: Vec<&Encoding> = encodings.iter().collect();
    let (classifications, _prof) = recognizer.run_batch(&refs, false).expect("recognize");

    assert_eq!(classifications.len(), encodings.len(), "batch size mismatch");
    let mut worst = 0.0f32;
    for (i, c) in classifications.iter().enumerate() {
        assert_eq!((c.logits.len(), c.num_labels), (seq_len * num_labels, num_labels), "per-chunk grid shape");
        // Compare only real-token rows: pad rows are a don't-care and may drift.
        for t in 0..seq_len {
            if encodings[i].attention_mask[t] == 0 {
                continue;
            }
            let got = &c.logits[t * num_labels..(t + 1) * num_labels];
            let want_off = (i * seq_len + t) * num_labels;
            let expected = &want[want_off..want_off + num_labels];
            let max_delta = max_abs_delta(got, expected);
            worst = worst.max(max_delta);
            assert!(max_delta < 1e-3, "token logits [{i}][{t}] drifted from PyTorch golden: max |delta| = {max_delta}");
        }
    }
    eprintln!("token logits real-position max |delta| = {worst:.3e}");
}

// ─── bf16 heads ─────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
enum Head {
    Embed,
    Classify,
    ClassifyTokens,
}

/// The heads in bf16 — where they run on tk's flash attention, GEMMs and layer
/// norm, which the f32 parity tests above never reach — against the f32
/// goldens, bounded by PyTorch's own bf16 drift ([`assert_tracks_pytorch_bf16`]).
/// The token head is compared at real tokens only.
#[test_case::test_case(Head::Embed; "embed")]
#[test_case::test_case(Head::Classify; "classify")]
#[test_case::test_case(Head::ClassifyTokens; "classify tokens")]
#[ignore = "heavy: real ModernBERT checkpoints + PyTorch goldens, on a bf16 device"]
fn bf16_heads_track_pytorch_bf16(head: Head) {
    let dtype = crate::default_compute_dtype();
    if dtype != DType::BFloat16 {
        eprintln!("skip bf16_heads_track_pytorch_bf16: the default device computes in {dtype:?}");
        return;
    }
    let (got, golden, key, real) = match head {
        Head::Embed => {
            let (mut embedder, enc, golden) = load_embed_fixture(dtype);
            let got = embedder.run_batch(&[&enc], false).expect("embed").0.remove(0).values;
            (got, golden, "expected_embedding", None)
        }
        Head::Classify => {
            let (mut classifier, encodings, golden) = load_classifier_fixture(dtype);
            let refs: Vec<&Encoding> = encodings.iter().collect();
            let got = classifier.run_batch(&refs, false).expect("classify").0.into_iter().flat_map(|c| c.logits);
            (got.collect(), golden, "expected_logits", None)
        }
        Head::ClassifyTokens => {
            let (mut classifier, encodings, golden) = load_token_fixture(dtype);
            let refs: Vec<&Encoding> = encodings.iter().collect();
            let got = classifier.run_batch(&refs, false).expect("classify tokens").0.into_iter().flat_map(|c| c.logits);
            let mask: Vec<i64> = load_golden_vec(&golden, "attention_mask");
            (got.collect(), golden, "expected_logits", Some((mask, classifier.num_labels())))
        }
    };
    let real = real.as_ref().map(|(mask, width)| (mask.as_slice(), *width));
    assert_tracks_pytorch_bf16(&format!("{head:?}"), got, &golden, key, real);
}
