//! Golden parity test — runs YOLO26x against a PyTorch reference output.
//!
//! Weights resolve from `$SVOD_YOLO`, then `data/yolo/`, then the HuggingFace
//! repo below; only the golden needs `scripts/convert_yolo.py` (and PyTorch):
//!
//! ```text
//! # fetch the weights, generate the golden once, then run
//! python scripts/convert_yolo.py
//! SVOD_DEVICE=CUDA:0 cargo test --release -p svod-model --lib yolo::parity -- --ignored
//! ```

use std::path::PathBuf;

use svod_dtype::DType;
use svod_tensor::Tensor;
use test_case::test_case;

use super::tk_gate::{tk_convs, tk_device};
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

/// Ultralytics' default detection threshold: a box is read only where the best
/// class score clears it, so that is where an f16 box has to be right. An anchor
/// scoring 5e-7 can drift by tens of pixels at f16 and no detection sees it.
const CONFIDENT: f32 = 0.25;

/// The f16 score band: twice what the x scale measures against the f32 golden
/// (8.7e-4), and still two hundred times under what a wrong batch-norm epsilon
/// moves a score by.
const F16_SCORE_TOL: f32 = 2e-3;

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

/// Decoded box coordinates live in pixel space, up to the image side; scores are
/// sigmoid outputs in [0, 1], so one absolute tolerance cannot serve both. Over
/// `[B, 4 + nc, A]` predictions this is the largest box deviation among anchors
/// whose golden best class score exceeds `floor`, the largest score deviation
/// over every anchor, and how many anchors' boxes were compared. A deviation a
/// NaN is part of counts as infinite.
fn deltas(got: &[f32], want: &[f32], channels: usize, anchors: usize, floor: f32) -> (f32, f32, usize) {
    let delta = |a: f32, b: f32| {
        let d = (a - b).abs();
        if d.is_nan() { f32::INFINITY } else { d }
    };
    let at =
        |v: &[f32], image: usize, channel: usize, anchor: usize| v[(image * channels + channel) * anchors + anchor];
    let (mut boxes, mut scores, mut compared) = (0f32, 0f32, 0usize);
    for image in 0..want.len() / (channels * anchors) {
        for anchor in 0..anchors {
            let best = (4..channels).map(|c| at(want, image, c, anchor)).fold(f32::NEG_INFINITY, f32::max);
            for c in 4..channels {
                scores = scores.max(delta(at(got, image, c, anchor), at(want, image, c, anchor)));
            }
            if best > floor {
                compared += 1;
                for c in 0..4 {
                    boxes = boxes.max(delta(at(got, image, c, anchor), at(want, image, c, anchor)));
                }
            }
        }
    }
    (boxes, scores, compared)
}

/// The f32 model against the golden over every anchor, and the f16 one over the
/// anchors a detection reads, with the tk convolution checked to have run where
/// the device has it.
#[test_case(DType::Float32, f32::NEG_INFINITY, SCORE_TOL; "f32, every anchor")]
#[test_case(DType::Float16, CONFIDENT, F16_SCORE_TOL; "f16, confident anchors")]
#[ignore = "heavy: 236 MB YOLO26x weights + PyTorch golden (see scripts/convert_yolo.py)"]
fn detect_output_matches_pytorch(dtype: DType, floor: f32, score_tol: f32) {
    let cfg = YoloConfig::new(YoloScale::XLarge, 80).with_compute_dtype(dtype.clone());
    let weights = resolve_file("model.safetensors", true);
    let golden_path = resolve_file("golden.safetensors", false);

    let model = Yolo26Detect::from_safetensors(&weights, cfg).expect("load model");

    let golden = load_safetensors(&golden_path).expect("load golden");
    let image_shape = load_golden_vec::<i64>(&golden, "images_shape");
    let images = Tensor::from_slice(load_golden_vec::<f32>(&golden, "images"))
        .try_reshape(image_shape.iter().map(|&d| d as isize).collect::<Vec<_>>())
        .unwrap();

    let out = model.forward(&images).expect("forward");
    if dtype != DType::Float32 && tk_device(&images.device()) {
        assert!(tk_convs(&out) > 0, "the tk convolution runs on this device");
    }
    let got = out.cast(DType::Float32).to_vec::<f32>().unwrap();
    let want = load_golden_vec::<f32>(&golden, "output");
    assert_eq!(got.len(), want.len(), "output length mismatch");

    let dims = out.dims().unwrap();
    let (channels, anchors) = (dims[1], dims[2]);
    let (box_delta, score_delta, compared) = deltas(&got, &want, channels, anchors, floor);
    println!("{dtype:?}: box {box_delta:.4} px over {compared} anchors, score {score_delta:.2e}");

    assert!(compared > 0, "no anchor clears {floor}");
    assert!(box_delta < BOX_TOL_PX, "max box |delta| = {box_delta:.6} px exceeds {BOX_TOL_PX}");
    assert!(score_delta < score_tol, "max score |delta| = {score_delta:.6} exceeds {score_tol:e}");
}
