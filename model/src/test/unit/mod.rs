mod audio;
mod batch;
mod bgem3;
mod bounds;
mod config;
mod diarizen;
mod encoder_head;
mod firered_vad;
mod firered_vad_stream;
mod gigaam;
mod gtcrn;
mod jit;
mod mel;
mod modernbert;
mod nemotron_diar;
mod origin;
mod qwen3;
mod remap;
mod resnet;
mod silero_vad;
mod simd;
mod state;
mod state_dict;
mod transcribe;
mod wavlm;
mod wespeaker;
mod whisper;
mod xlm_roberta;
mod yolo;

use crate::state::StateDict;

/// Largest `|a - b|`, NaN when any element is: `f32::max` drops a NaN, which
/// would read a broken output as a perfect match.
fn max_abs_delta(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "compared outputs differ in length");
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, nan_max)
}

/// `f32::max` that keeps a NaN where `f32::max` drops it.
fn nan_max(max: f32, d: f32) -> f32 {
    if d.is_nan() || d > max { d } else { max }
}

/// A golden's `key`, realized and read out flat.
fn load_golden_vec<T: svod_dtype::ext::HasDType + Default + Clone>(sd: &StateDict, key: &str) -> Vec<T> {
    let t = sd.get(key).unwrap_or_else(|| panic!("golden key {key}")).clone();
    t.realize().expect("realize golden");
    t.as_vec::<T>().expect("golden readout")
}
