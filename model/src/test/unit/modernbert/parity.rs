//! Parity against the PyTorch reference (`answerdotai/ModernBERT-base`).
//! Heavy: loads the real checkpoint + a golden `last_hidden_state` produced by
//! HuggingFace `transformers` (`uv run scripts/convert_modernbert.py`).
//!
//! Runs in **f32** (config dtype overridden) so it works on CPU backends
//! without GPU bf16 transcendentals. bf16 numerical parity is implied by the
//! framework's f32-accumulator guarantees in layernorm/matmul/attention.

use std::path::{Path, PathBuf};

use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::modernbert::{ModernBert, ModernBertConfig, ModernBertForMaskedLm};
use crate::state::StateDict;

const HUB_REPO: &str = "answerdotai/ModernBERT-base";

/// Resolve `model.safetensors` / `golden.safetensors` for the real-checkpoint
/// tests: `SVOD_MODERNBERT` dir override → local `data/modernbert/` (output of
/// `scripts/convert_modernbert.py`) → HF Hub download.
fn real_file(name: &str) -> PathBuf {
    let dir = std::env::var_os("SVOD_MODERNBERT")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/modernbert"));
    let local = dir.join(name);
    if local.exists() {
        local
    } else {
        let repo = crate::hub::HubRepo::open(HUB_REPO, "main").expect("HF Hub API");
        repo.get(name).unwrap_or_else(|_| panic!("download {name} from HF Hub"))
    }
}

fn load_golden_vec<T: svod_dtype::ext::HasDType + Default + Clone>(sd: &StateDict, key: &str) -> Vec<T> {
    let t = sd.get(key).unwrap_or_else(|| panic!("golden key {key}")).clone();
    t.realize().expect("realize golden");
    t.as_vec::<T>().expect("golden readout")
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
        .flat_map(|((g, w), _)| g.iter().zip(w.iter()).map(|(a, e)| (a - e).abs()))
        .fold(0.0f32, f32::max)
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
/// drift on the same rows (`convert_modernbert.py --long --long-length L`: a
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
        "golden_long_{length} predates PyTorch's bf16 paths: rerun convert_modernbert.py --long"
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
