//! The batching every encoder head shares ([`crate::encoder_head`]): plan
//! shapes, input packing, the build options and the device a head runs on,
//! plus the fixtures and comparisons the heads' parity tests share.

use std::path::{Path, PathBuf};

use svod_arch::pipelines::text::{EncoderHead, Encoding};
use svod_dtype::default_device::{default_device, set_default_device, with_default_device};
use svod_dtype::{DType, DeviceSpec};

use crate::encoder_head::{HeadOptions, pack_ids, pack_mask, plan_shape};
use crate::modernbert::{ModernBertClassifier, ModernBertEmbedder, ModernBertTokenClassifier};
use crate::state::StateDict;
use crate::test::unit::xlm_roberta::model::tiny_cfg;
use crate::test::unit::{load_golden_vec, max_abs_delta, nan_max};
use crate::xlm_roberta::{XlmRobertaTokenClassificationModel, XlmRobertaTokenClassifier};

/// A chunk of `real_ids` (mask 1) followed by `n_pad` pad positions (id 0,
/// mask 0).
pub(super) fn encoding(real_ids: &[u32], n_pad: usize) -> Encoding {
    let mut ids = real_ids.to_vec();
    let mut mask = vec![1u32; real_ids.len()];
    ids.extend(std::iter::repeat_n(0u32, n_pad));
    mask.extend(std::iter::repeat_n(0u32, n_pad));
    let l = ids.len();
    Encoding {
        input_ids: ids,
        attention_mask: mask,
        token_type_ids: vec![0; l],
        offsets: (0..l).map(|i| (i, i + 1)).collect(),
        special_tokens_mask: vec![0; l],
    }
}

/// The golden's `input_ids` / `attention_mask` — `(B, L)`, or `(L,)` for one
/// row — as one encoding per row.
pub(super) fn golden_encodings(golden: &StateDict) -> Vec<Encoding> {
    let (ids, mask): (Vec<i64>, Vec<i64>) =
        (load_golden_vec(golden, "input_ids"), load_golden_vec(golden, "attention_mask"));
    let seq_len = golden["input_ids"].dim_const(-1).expect("concrete seq_len");
    ids.chunks_exact(seq_len)
        .zip(mask.chunks_exact(seq_len))
        .map(|(ids, mask)| Encoding {
            input_ids: ids.iter().map(|&x| x as u32).collect(),
            attention_mask: mask.iter().map(|&x| x as u32).collect(),
            token_type_ids: vec![0; seq_len],
            offsets: vec![(0, 0); seq_len],
            special_tokens_mask: vec![0; seq_len],
        })
        .collect()
}

/// Resolve a real-checkpoint test file of `(repo, revision)`: the `env`
/// directory override → local `data/<data_dir>/` → HF Hub download. Model
/// weights / configs live on HF Hub, but the `golden*.safetensors` fixtures are
/// generated locally (not committed, not published) — a missing golden says so
/// instead of a generic "download" message that's wrong for goldens.
pub(super) fn local_or_hub(env: &str, data_dir: &str, (repo, revision): (&str, &str), name: &str) -> PathBuf {
    let dir = std::env::var_os(env)
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../data").join(data_dir));
    let local = dir.join(name);
    if local.exists() {
        return local;
    }
    assert!(
        !name.starts_with("golden"),
        "{name} not found in {} and not on HF Hub — goldens are generated locally: run the generate script \
         the parity module names (or point {env} at its output dir)",
        dir.display()
    );
    let repo = crate::hub::HubRepo::open(repo, revision).expect("HF Hub API");
    repo.get(name).unwrap_or_else(|_| panic!("download {name} from HF Hub"))
}

/// The rows of a row-major `(tokens, width)` grid at the tokens `mask` marks
/// real: a per-token head's pad positions are a don't-care.
pub(super) fn real_rows(grid: &[f32], mask: &[i64], width: usize) -> Vec<f32> {
    assert_eq!(grid.len(), mask.len() * width, "grid and mask cover different tokens");
    grid.chunks_exact(width).zip(mask).filter(|&(_, &m)| m == 1).flat_map(|(row, _)| row.to_vec()).collect()
}

/// How far past PyTorch's worst bf16 path a bf16 head's drift may land. Its four
/// paths differ by rounding alone, yet ModernBERT's classifier logits sit 4.6x
/// apart, so four draws are still a thin yardstick for a fifth.
const BF16_HEAD_SLACK: f32 = 2.0;

/// A bf16 head's output `got` against the f32 golden `key`, bounded by
/// PyTorch's own bf16 drift on the same inputs (`<key>_bf16.{cpu,cuda}_{sdpa,eager}`,
/// written by the golden scripts) times [`BF16_HEAD_SLACK`]. The CUDA paths are
/// required: a GPU's matrix-core GEMMs and fused attention round apart from any
/// CPU path, so the CPU pair alone measures another rounding than the one under
/// test. `real`, the golden's `(attention_mask, labels)` for a per-token head,
/// keeps the real-token rows alone: pad positions are a don't-care.
pub(super) fn assert_tracks_pytorch_bf16(
    head: &str,
    got: Vec<f32>,
    golden: &StateDict,
    key: &str,
    real: Option<(&[i64], usize)>,
) {
    let real_rows = |grid: Vec<f32>| match real {
        Some((mask, width)) => real_rows(&grid, mask, width),
        None => grid,
    };
    let want = real_rows(load_golden_vec(golden, key));
    let drift = max_abs_delta(&real_rows(got), &want);
    let pytorch = golden
        .keys()
        .filter(|k| k.starts_with(&format!("{key}_bf16.")))
        .map(|k| (k.clone(), max_abs_delta(&real_rows(load_golden_vec(golden, k)), &want)))
        .collect::<Vec<_>>();
    assert!(
        pytorch.iter().any(|(k, _)| k.contains(".cuda_")),
        "the golden lacks PyTorch's CUDA bf16 paths: rerun its generate script on a CUDA machine"
    );
    let worst = pytorch.iter().map(|(_, d)| *d).fold(0.0, nan_max);
    eprintln!("{head}: svod bf16 max |Δ| {drift:.3e}; pytorch {pytorch:?}");
    assert!(drift <= BF16_HEAD_SLACK * worst, "{head} bf16 drifts past PyTorch's: {drift:.3e} against {worst:.3e}");
}

// ── input packing ───────────────────────────────────────────────────────────

/// Packing honors the per-row `len` stride, fills pad positions and unused rows
/// with the pad id, and truncates over-length inputs. A chunk's own ids, its
/// in-chunk padding included, are copied as they are. The mask mirrors the
/// ids' real-token counts; only an unused row attends its first position, so a
/// live empty chunk stays fully masked and pools to the zero vector.
#[test]
fn pack_inputs_pad_stride_and_truncate_correctly() {
    let (rows, len, pad) = (4, 4, 7);
    // Stale contents from a previous batch must not survive the pack.
    let mut ids_v = vec![-1i64; rows * len];
    let mut mask_v = vec![-1i64; rows * len];

    // e1: 2 real tokens + 1 in-chunk pad (id 0, mask 0). Then a chunk with no
    // token. e2: 5 real tokens — one over len, so the last must be truncated.
    // Row 3 left unused.
    let (e1, empty, e2) = (encoding(&[1, 2], 1), encoding(&[], 0), encoding(&[10, 20, 30, 40, 50], 0));
    let batch = [&e1, &empty, &e2];
    pack_ids(&mut ids_v, &batch, len, pad);
    pack_mask(&mut mask_v, &batch, len);

    assert_eq!(ids_v, vec![1, 2, 0, 7, 7, 7, 7, 7, 10, 20, 30, 40, 7, 7, 7, 7]);
    assert_eq!(mask_v, vec![1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]);
}

// ── plan shapes ─────────────────────────────────────────────────────────────

/// A batch runs at the smallest prepared shape that holds it: rows round up to
/// a power of two capped at `max_batch`, the length to the flash-attention tile.
#[test_case::test_case(3, 128, 8 => (4, 128); "rows round up, a full tile stays")]
#[test_case::test_case(5, 129, 6 => (6, 256); "rows cap at max_batch, length spills a tile")]
#[test_case::test_case(2, 0, 8 => (2, 128); "empty chunks still take a tile")]
fn plan_shape_rounds_rows_and_length(b: usize, longest: usize, max_batch: usize) -> (usize, usize) {
    plan_shape(b, longest, max_batch)
}

// ── options, threads and devices ────────────────────────────────────────────

/// Options left unset resolve on the building thread: its default device, and
/// that device's compute dtype. A set field wins.
#[test_case::test_case(None, None => (DeviceSpec::Cpu, DType::BFloat16); "the building thread's device")]
#[test_case::test_case(Some(DType::Float32), None => (DeviceSpec::Cpu, DType::Float32); "a set dtype")]
#[test_case::test_case(None, Some(DeviceSpec::Amd { device_id: 3 }) => (DeviceSpec::Amd { device_id: 3 }, DType::BFloat16); "a set device")]
fn head_options_resolve_on_the_building_thread(
    dtype: Option<DType>,
    device: Option<DeviceSpec>,
) -> (DeviceSpec, DType) {
    let opts = HeadOptions { dtype, device, ..HeadOptions::default() };
    with_default_device(DeviceSpec::Cpu, || opts.build(|dtype| (default_device(), dtype)))
}

/// Pins the threading contract a serving pipeline relies on: each head moves
/// to the worker thread that runs it.
#[test]
fn heads_are_send() {
    fn assert_send<T: Send>() {}
    assert_send::<ModernBertEmbedder>();
    assert_send::<ModernBertClassifier>();
    assert_send::<ModernBertTokenClassifier>();
    assert_send::<XlmRobertaTokenClassifier>();
}

/// A head built for the CPU compiles its first plan there even when the thread
/// that runs it defaults to a device that does not exist, whose allocations
/// would panic.
#[test]
fn a_head_compiles_for_the_device_it_was_built_on() {
    let mut classifier = with_default_device(DeviceSpec::Cpu, || {
        XlmRobertaTokenClassifier::new(XlmRobertaTokenClassificationModel::empty(tiny_cfg()), 1, 8)
    })
    .expect("classifier");
    let chunk = encoding(&[0, 10, 20, 2], 0);
    let run = std::thread::scope(|s| {
        s.spawn(|| {
            set_default_device(DeviceSpec::Amd { device_id: 99 });
            classifier.run(&chunk, false)
        })
        .join()
    });
    let logits = run.expect("the run stays on the CPU").expect("classify").0.logits;
    assert_eq!(logits.len(), chunk.len() * tiny_cfg().num_labels);
}
