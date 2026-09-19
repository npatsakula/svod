//! Golden parity test — runs YOLO26x against a PyTorch reference output.
//!
//! Weights resolve from `$SVOD_YOLO`, then `data/yolo/`, then the HuggingFace
//! repo below; only the golden needs `scripts/convert_yolo.py` (and PyTorch):
//!
//! ```text
//! # fetch the weights, generate the golden once, then run
//! python scripts/convert_yolo.py
//! cargo test -p svod-model --lib yolo::parity -- --ignored
//! ```

use std::path::PathBuf;

use svod_tensor::Tensor;

use crate::state::StateDict;
use crate::state::load_safetensors;
use crate::yolo::{Yolo26Detect, YoloConfig, YoloScale};

/// The published conversion of `Ultralytics/YOLO26`'s `yolo26x.pt`. Upstream is
/// AGPL-3.0, so this repo holds only the converted weights the loader reads.
const HUB_REPO: &str = "mexus/svod-yolo26x";

/// Boxes are decoded into pixels and reach ~656, so a deviation is only
/// meaningful against the image side. This is ~1e-4 of a 640 px coordinate;
/// against the x scale it leaves better than 7x headroom (observed 0.0067 px
/// on the box channels).
const BOX_TOL_PX: f32 = 0.05;

/// Class scores come out of the same sigmoid on both sides and agree exactly
/// at the x scale, so this only has to absorb f32 reassociation -- and is still
/// 1000x tighter than the score the reference run reports (0.966). A wrong
/// batch-norm epsilon moves a score by ~0.2, three orders of magnitude past it.
const SCORE_TOL: f32 = 1e-3;

/// Resolve a fixture from the local directories, falling back to the Hub.
///
/// Failing to find `model.safetensors` on the Hub is a real error, but the
/// golden is generated rather than published, so its absence is reported as
/// such instead of as a download failure.
fn resolve_file(name: &str, from_hub: bool) -> PathBuf {
    if let Ok(dir) = std::env::var("SVOD_YOLO") {
        let p = PathBuf::from(dir).join(name);
        if p.exists() {
            return p;
        }
    }
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../data/yolo").join(name);
    if p.exists() {
        return p;
    }
    if from_hub {
        let repo = crate::hub::HubRepo::open(HUB_REPO, "main").expect("HF Hub API");
        return repo.get(name).unwrap_or_else(|_| panic!("download {name} from {HUB_REPO}"));
    }
    panic!("missing {name}: run `python scripts/convert_yolo.py` first")
}

fn load_golden_vec<T: Clone + Default + svod_dtype::ext::HasDType>(sd: &StateDict, key: &str) -> Vec<T> {
    let t = sd.get(key).unwrap_or_else(|| panic!("missing golden key: {key}")).clone();
    t.realize().unwrap();
    t.as_vec::<T>().unwrap()
}

/// Decoded box coordinates live in pixel space, up to the image side; scores
/// are sigmoid outputs in [0, 1]. One absolute tolerance cannot serve both, so
/// split the deviation by channel: `4 + nc` channels of `anchors` each, boxes
/// first.
fn deltas_by_channel(got: &[f32], want: &[f32], channels: usize, anchors: usize) -> (f32, f32) {
    got.iter().zip(want).enumerate().fold((0.0f32, 0.0f32), |(boxes, scores), (i, (a, b))| {
        let d = (a - b).abs();
        if (i / anchors) % channels < 4 { (boxes.max(d), scores) } else { (boxes, scores.max(d)) }
    })
}

#[test]
#[ignore = "heavy: 236 MB YOLO26x weights + PyTorch golden (see scripts/convert_yolo.py)"]
fn detect_output_matches_pytorch() {
    let cfg = YoloConfig::new(YoloScale::XLarge, 80);
    let weights = resolve_file("model.safetensors", true);
    let golden_path = resolve_file("golden.safetensors", false);

    let model = Yolo26Detect::from_safetensors(&weights, cfg).expect("load model");

    let golden = load_safetensors(&golden_path).expect("load golden");
    let image_shape = load_golden_vec::<i64>(&golden, "images_shape");
    let images = Tensor::from_slice(load_golden_vec::<f32>(&golden, "images"))
        .try_reshape(image_shape.iter().map(|&d| d as isize).collect::<Vec<_>>())
        .unwrap();

    let out = model.forward(&images).expect("forward");
    out.realize().unwrap();

    let got = out.as_vec::<f32>().unwrap();
    let want = load_golden_vec::<f32>(&golden, "output");
    assert_eq!(got.len(), want.len(), "output length mismatch");

    let dims = out.dims().unwrap();
    let (channels, anchors) = (dims[1], dims[2]);
    let (box_delta, score_delta) = deltas_by_channel(&got, &want, channels, anchors);

    assert!(box_delta < BOX_TOL_PX, "max box |delta| = {box_delta:.6} px exceeds {BOX_TOL_PX}");
    assert!(score_delta < SCORE_TOL, "max score |delta| = {score_delta:.6} exceeds {SCORE_TOL:e}");
}
