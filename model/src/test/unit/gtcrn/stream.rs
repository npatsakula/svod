//! Streaming GTCRN tests.
//!
//! Default tier: the symbolic one-frame graph's shapes, the state-dict round
//! trip, and — the real coverage — frame-by-frame equality with the offline
//! model on random weights. Heavy tier: the PyTorch `StreamGTCRN` golden.

use std::path::PathBuf;
use std::sync::Arc;

use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::Module;

use crate::gtcrn::stream::{GtcrnStream, GtcrnStreamJit};
use crate::gtcrn::{Gtcrn, HUB_REPO, N_FREQ};

/// Depth-conv weights the stream model stores flipped and must emit unflipped.
const FLIPPED_KEYS: [&str; 3] = [
    "decoder.de_convs.0.depth_conv.weight",
    "decoder.de_convs.1.depth_conv.weight",
    "decoder.de_convs.2.depth_conv.weight",
];

fn zeros(shape: &[usize]) -> Tensor {
    Tensor::zeros(shape, DType::Float32)
}

/// The three cache families, all zero — a cold stream, built eagerly for the
/// tests that run the graph without the JIT.
fn cold_caches() -> ([Tensor; 6], [Tensor; 6], [Tensor; 2]) {
    (
        std::array::from_fn(|i| zeros(&GtcrnStream::CONV_CACHE[i])),
        std::array::from_fn(|_| zeros(&GtcrnStream::TRA_CACHE)),
        std::array::from_fn(|_| zeros(&GtcrnStream::INTER_CACHE)),
    )
}

fn refs<const N: usize>(t: &[Tensor; N]) -> [&Tensor; N] {
    std::array::from_fn(|i| &t[i])
}

/// Build the symbolic streaming graph on random weights with `T = 1` and verify
/// the output plus every recycled cache shape without realizing. Catches axis /
/// cache-slicing / permute bugs across the whole streaming graph in ms.
#[test]
fn stream_forward_shape() {
    let model = GtcrnStream::with_random_weights();
    let (conv, tra, inter) = cold_caches();
    let spec = zeros(&[1, N_FREQ, 1, 2]);

    let (enh, new_conv, new_tra, new_inter) =
        model.forward_stream(&spec, refs(&conv), refs(&tra), refs(&inter)).expect("forward_stream");

    assert_eq!(enh.dims().expect("concrete output shape"), vec![1, N_FREQ, 1, 2], "enhanced spec");

    // Each new cache must be exactly the shape the next call feeds back in:
    // (1,16,2,33), (1,16,4,33), (1,16,10,33) for the encoder's dilations 1,2,5
    // and the reverse for the decoder.
    for (i, c) in new_conv.iter().enumerate() {
        assert_eq!(c.dims().unwrap(), GtcrnStream::CONV_CACHE[i].to_vec(), "conv cache {i}");
    }
    for (i, c) in new_tra.iter().enumerate() {
        assert_eq!(c.dims().unwrap(), GtcrnStream::TRA_CACHE.to_vec(), "tra cache {i}");
    }
    for (i, c) in new_inter.iter().enumerate() {
        assert_eq!(c.dims().unwrap(), GtcrnStream::INTER_CACHE.to_vec(), "inter cache {i}");
    }
}

/// The stream model loads the *offline* checkpoint and must emit it back
/// unchanged: the `StreamConvTranspose2d` kernel flip is an involution applied
/// on both edges, so a reload of an emitted dict must not flip a second time.
/// A one-sided flip un-flips silently here and nowhere else.
#[test]
fn state_dict_keeps_the_checkpoint_spelling() {
    let checkpoint = Gtcrn::with_random_weights().state_dict("");
    let model = GtcrnStream::from_state_dict(&checkpoint).expect("load");
    let emitted = model.state_dict("");

    assert_eq!(emitted.len(), checkpoint.len(), "key count");
    for (key, want) in &checkpoint {
        let got = emitted.get(key).unwrap_or_else(|| panic!("key dropped: {key}"));
        if FLIPPED_KEYS.contains(&key.as_str()) {
            assert_eq!(got.to_vec::<f32>().unwrap(), want.to_vec::<f32>().unwrap(), "{key} came back flipped");
        } else {
            assert!(Arc::ptr_eq(&got.uop(), &want.uop()), "{key} was not taken from the state dict");
        }
    }

    // Idempotence: `from_state_dict(&m.state_dict(""))` is the identity.
    let reloaded = GtcrnStream::from_state_dict(&emitted).expect("reload").state_dict("");
    for key in FLIPPED_KEYS {
        let (a, b) = (emitted[key].to_vec::<f32>().unwrap(), reloaded[key].to_vec::<f32>().unwrap());
        assert_eq!(a, b, "{key} is not idempotent across a state-dict round trip");
    }
}

/// Frames of the `(1, F, T, 2)` layout: frame `i` is strided, not contiguous.
fn frame_of(flat: &[f32], frames: usize, i: usize) -> Vec<f32> {
    (0..N_FREQ).flat_map(|f| [flat[(f * frames + i) * 2], flat[(f * frames + i) * 2 + 1]]).collect()
}

/// Run `frames` frames through the streaming JIT and return each output frame.
fn run_stream(model: GtcrnStream, spec: &[f32], frames: usize) -> Vec<Vec<f32>> {
    let mut jit = GtcrnStreamJit::prepared(model).expect("prepare stream JIT");
    (0..frames)
        .map(|i| {
            jit.spec_host_write::<f32>().expect("spec guard").copy_from(&frame_of(spec, frames, i)).expect("spec");
            jit.execute().expect("execute");
            jit.enh_to_vec::<f32>().expect("enhanced frame")
        })
        .collect()
}

/// The offline network is strictly causal, so feeding the streaming model one
/// frame at a time must reproduce the offline forward on the same window
/// exactly. This is the assertion the shape test cannot make: it fails on a
/// wrong kernel flip, a dropped GRU hidden state, a lost dilation, a
/// non-causal pad, or a cache sliced from the wrong end.
#[test]
fn streaming_matches_offline() {
    const FRAMES: usize = 12;

    let checkpoint = Gtcrn::with_random_weights().state_dict("");
    let offline = Gtcrn::from_state_dict(&checkpoint).expect("offline model");
    let stream = GtcrnStream::from_state_dict(&checkpoint).expect("stream model");

    let spec = Tensor::uniform_with_dtype(&[1, N_FREQ, FRAMES, 2], -1.0, 1.0, DType::Float32).unwrap().contiguous();
    let spec_vec = spec.to_vec::<f32>().unwrap();
    let want = offline.forward(&spec).expect("offline forward").to_vec::<f32>().unwrap();

    let got = run_stream(stream, &spec_vec, FRAMES);

    let mut max_delta = 0.0f32;
    for (i, frame) in got.iter().enumerate() {
        let want_frame = frame_of(&want, FRAMES, i);
        assert_eq!(frame.len(), want_frame.len(), "frame {i} length");
        for (a, b) in frame.iter().zip(&want_frame) {
            assert!(a.is_finite(), "frame {i} produced {a}");
            max_delta = max_delta.max((a - b).abs());
        }
    }
    // Same weights, same arithmetic, different kernel fusion: the two paths
    // differ only by fp32 reassociation.
    assert!(max_delta < 1e-5, "streaming vs offline: max |delta| = {max_delta:e}");
}

fn resolve_file(name: &str) -> PathBuf {
    for dir in [std::env::var("SVOD_GTCRN").ok(), Some(format!("{}/../data/gtcrn", env!("CARGO_MANIFEST_DIR")))]
        .into_iter()
        .flatten()
    {
        let p = PathBuf::from(dir).join(name);
        if p.exists() {
            return p;
        }
    }
    crate::hub::HubRepo::open(HUB_REPO, "main")
        .and_then(|repo| repo.get(name))
        .unwrap_or_else(|e| panic!("download {name} from {HUB_REPO}: {e}"))
}

/// Run the streaming model frame-by-frame on the golden spec and compare to the
/// PyTorch `StreamGTCRN` output (`convert_gtcrn.py --stream-golden`), which was
/// captured from the same cold start over the same 24-frame crop.
///
/// ```text
/// SVOD_GTCRN=$PWD/data/gtcrn cargo test -p svod-model --lib gtcrn::stream -- --ignored
/// ```
#[test]
#[ignore = "heavy: real GTCRN weights + PyTorch stream golden (local or HF Hub download)"]
fn streaming_matches_pytorch_stream() {
    const FRAMES: usize = 24;

    let model = GtcrnStream::from_safetensors(&resolve_file("gtcrn.safetensors")).expect("load stream model");
    let golden = crate::state::load_safetensors(&resolve_file("golden_stream.safetensors")).expect("load golden");
    let get = |key: &str| {
        crate::state::get_tensor(&golden, key).unwrap_or_else(|_| panic!("missing golden key: {key}")).to_vec::<f32>()
    };

    let spec = get("spec_crop").unwrap();
    let want = get("stream_output_crop").unwrap();
    let got = run_stream(model, &spec, FRAMES);

    let (mut max_delta, mut peak) = (0.0f32, 0.0f32);
    for (i, frame) in got.iter().enumerate() {
        let want_frame = frame_of(&want, FRAMES, i);
        assert_eq!(frame.len(), want_frame.len(), "frame {i} length");
        for (a, b) in frame.iter().zip(&want_frame) {
            max_delta = max_delta.max((a - b).abs());
            peak = peak.max(b.abs());
        }
    }

    let rel = max_delta / peak.max(1e-6);
    assert!(rel < 1e-3, "streaming vs PyTorch: max |delta| = {max_delta:.6} (rel {rel:.2e}, peak {peak:.2})");
}
