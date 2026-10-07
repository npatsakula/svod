//! Parity against the HF transformers reference (fp32, CPU), produced by
//! `scripts/nemotron_diar_golden.py` over `audio_1.wav`: the whole 89 s clip
//! and its first 20 s, under the offline and the 1.04 s streaming profiles.
//! The long clip compresses the speaker cache in both profiles.
//!
//! ```text
//! uv run scripts/nemotron_diar_golden.py
//! cargo test -p svod-model --release --lib nemotron_diar::parity -- --ignored
//! ```
//!
//! Goldens resolve from `$SVOD_NEMOTRON_DIAR`, then `data/nemotron_diar`.

use std::path::PathBuf;

use svod_dtype::DType;
use test_case::test_case;

use super::{CONFIG_JSON, PROCESSOR_JSON};
use crate::audio::MelSpectrogram;
use crate::nemotron_diar::{Diarizer, NemotronDiar, NemotronDiarConfig, StreamingMode};
use crate::state::{StateDict, load_safetensors};

fn golden(clip: &str, profile: &str) -> StateDict {
    let name = format!("golden_{clip}_{profile}.safetensors");
    let path = [
        std::env::var("SVOD_NEMOTRON_DIAR").ok(),
        Some(format!("{}/../data/nemotron_diar", env!("CARGO_MANIFEST_DIR"))),
    ]
    .into_iter()
    .flatten()
    .map(|dir| PathBuf::from(dir).join(&name))
    .find(|p| p.exists())
    .unwrap_or_else(|| panic!("{name} not found: run scripts/nemotron_diar_golden.py or set SVOD_NEMOTRON_DIAR"));
    load_safetensors(&path).unwrap()
}

fn vec_f32(sd: &StateDict, key: &str) -> Vec<f32> {
    sd.get(key).unwrap_or_else(|| panic!("missing golden key {key}")).to_vec::<f32>().unwrap()
}

fn max_diff(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "length");
    got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max)
}

/// The front-end reproduces the HF feature extractor, masked tail included.
#[test]
#[ignore = "heavy: needs the generated goldens"]
fn mel_matches_reference() {
    let sd = golden("long", "offline");
    let audio = vec_f32(&sd, "audio");
    let want = vec_f32(&sd, "mel");
    let config = NemotronDiarConfig::from_json_strs(CONFIG_JSON, PROCESSOR_JSON).unwrap();
    let mel = MelSpectrogram::new(&config.mel_config());
    let frames = mel.num_frames(audio.len());
    assert_eq!(want.len(), 128 * frames);
    let mut framed = vec![0.0f32; mel.framed_len(audio.len())];
    mel.frame_into(&audio, &mut framed);
    let valid = (audio.len() / 160) as i32;
    let got = mel
        .forward_tensor(
            &svod_tensor::Tensor::from_slice(framed).try_unsqueeze(0).unwrap(),
            &svod_tensor::Tensor::from_slice(vec![valid]),
        )
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    let diff = max_diff(&got, &want);
    eprintln!("mel max abs diff: {diff:.3e}");
    assert!(diff < 2e-3, "mel drifted: {diff}");
}

/// The first mel frame whose reference output follows a compression that
/// ranked an exact tie at its cutoff: `torch.topk` breaks ties arbitrarily, so
/// from there on any reimplementation may keep an equally scored frame and the
/// outputs part ways.
fn first_tie_frame(sd: &StateDict, cache_len: usize) -> Option<usize> {
    let steps: Vec<i64> = sd["steps"].to_vec::<i64>().unwrap();
    (0..steps.len() / 6).find_map(|step| {
        let scores = sd.get(&format!("step{step}.compress_scores"))?.to_vec::<f32>().unwrap();
        // the flat ranking also holds one `+inf` silence slot per speaker
        let mut ranked: Vec<f32> = scores.into_iter().chain(std::iter::repeat_n(f32::INFINITY, 8)).collect();
        ranked.sort_by(|a, b| b.total_cmp(a));
        let tied = ranked.get(cache_len).is_some_and(|next| *next == ranked[cache_len - 1] && next.is_finite());
        tied.then(|| steps.get((step + 1) * 6 + 3).map_or(usize::MAX, |&frame| frame as usize))
    })
}

/// End to end: the per-frame speaker probabilities of the whole clip match
/// until a tied compression (see [`first_tie_frame`]); after it the speaker
/// decisions still agree.
#[test_case("short", "offline"; "short offline")]
#[test_case("long", "offline"; "long offline")]
#[test_case("short", "streaming"; "short streaming")]
#[test_case("long", "streaming"; "long streaming")]
#[ignore = "heavy: real weights and the generated goldens"]
fn probs_match_reference(clip: &str, profile: &str) {
    let sd = golden(clip, profile);
    let audio = vec_f32(&sd, "audio");
    let want = vec_f32(&sd, "probs");
    let model = NemotronDiar::from_hub(DType::Float32, 1).unwrap();
    let cache_len = model.config.cache.cache_len;
    let mut diarizer = match profile {
        "offline" => Diarizer::offline(model),
        _ => Diarizer::streaming(model, StreamingMode::LowLatency),
    }
    .unwrap();
    let got = diarizer.diarize(&audio, 16000).unwrap().probs;
    assert_eq!(got.len(), want.len(), "frames");

    let tie = first_tie_frame(&sd, cache_len).map_or(want.len(), |frame| (frame * 8).min(want.len()));
    let exact = max_diff(&got[..tie], &want[..tie]);
    let flips = got.iter().zip(&want).filter(|(a, b)| (**a > 0.5) != (**b > 0.5)).count();
    let scope = match tie / 8 == want.len() / 8 {
        true => "every frame".to_string(),
        false => format!("the {} frames before a tied compression", tie / 8),
    };
    eprintln!(
        "{clip}/{profile}: probs max abs diff {exact:.3e} over {scope}, {flips} of {} decisions flipped",
        want.len()
    );
    assert!(exact < 2e-3, "{clip}/{profile}: probs drifted by {exact}");
    assert!(flips * 1000 <= want.len(), "{clip}/{profile}: {flips} speaker decisions flipped");
}

/// The speaker-cache bookkeeping alone: the reference's own step-54
/// probabilities (the first compression of the long streaming run) through
/// [`SpeakerCache`] keep the reference's cache rows, up to frames whose scores
/// tie exactly (which `torch.topk` orders arbitrarily).
#[test]
#[ignore = "heavy: needs the generated goldens"]
fn cache_compression_matches_reference() {
    use svod_arch::diarization::{ContextRow, SpeakerCache};

    let sd = golden("long", "streaming");
    let config = NemotronDiarConfig::from_json_strs(CONFIG_JSON, PROCESSOR_JSON).unwrap();
    let profile = config.streaming_profile(StreamingMode::LowLatency);
    let lens: Vec<i64> = sd["step54.lens"].to_vec::<i64>().unwrap();
    let (cached, queued, chunk) = (lens[0] as usize, lens[1] as usize, lens[2] as usize);

    // Reach the step's state: `cached` uncompressed cache frames, `queued` in the FIFO.
    let mut cache = SpeakerCache::new(config.cache_config(&profile)).unwrap();
    cache.update(cached + queued, &vec![0.0; (cached + queued) * 8]).unwrap();
    assert_eq!((cache.cache_frames(), cache.fifo_frames()), (cached, queued));

    cache.update(chunk, &vec_f32(&sd, "step54.pooled_probs")).unwrap();
    let (hidden, embeds, silence) = (512, vec_f32(&sd, "step54.embeds"), vec_f32(&sd, "silence_embeds"));
    let (want, scores) = (vec_f32(&sd, "step54.spkcache"), vec_f32(&sd, "step54.compress_scores"));
    assert_eq!(cache.cache_frames() * hidden, want.len());
    // Kept rows as step frames (`None` = silence), ours and the reference's.
    let frame_of = |row: &[f32]| {
        (row != &silence[..]).then(|| {
            (0..embeds.len() / hidden)
                .find(|&i| &embeds[i * hidden..(i + 1) * hidden] == row)
                .expect("a cached row is a step input row")
        })
    };
    let ours: Vec<Option<usize>> = cache.layout()[..cache.cache_frames()]
        .iter()
        .map(|source| match *source {
            ContextRow::Step(i) => frame_of(&embeds[i * hidden..(i + 1) * hidden]),
            ContextRow::Silence => None,
        })
        .collect();
    let theirs: Vec<Option<usize>> = want.chunks(hidden).map(frame_of).collect();
    // What one side keeps and the other doesn't must pair up into exact ties.
    let without = |a: &[Option<usize>], b: &[Option<usize>]| {
        let mut rest = b.to_vec();
        let mut extra: Vec<Option<usize>> = a
            .iter()
            .filter(|f| match rest.iter().position(|g| g == *f) {
                Some(at) => {
                    rest.swap_remove(at);
                    false
                }
                None => true,
            })
            .copied()
            .collect();
        extra.sort();
        extra
    };
    let (only_ours, only_theirs) = (without(&ours, &theirs), without(&theirs, &ours));
    eprintln!("kept only by us {only_ours:?}, only by the reference {only_theirs:?}");
    assert_eq!(only_ours.len(), only_theirs.len());
    for (a, b) in only_ours.iter().zip(&only_theirs) {
        let (Some(a), Some(b)) = (a, b) else { panic!("a silence slot where the reference keeps a frame") };
        let tied = (0..8).any(|s| scores[a * 8 + s].is_finite() && scores[a * 8 + s] == scores[b * 8 + s]);
        assert!(tied, "frame {a} kept where the reference keeps {b}, with different scores");
    }
}

/// Half precision, offline (no tied compression): the speaker decisions of
/// the f32 reference survive bf16/f16 encoding. Run on a GPU
/// (`SVOD_DEVICE=CUDA:0`); flash attention and the hand GEMMs take this path.
#[test_case(DType::BFloat16; "bf16")]
#[test_case(DType::Float16; "f16")]
#[ignore = "heavy: real weights and the generated goldens"]
fn half_precision_keeps_decisions(dtype: DType) {
    let sd = golden("long", "offline");
    let audio = vec_f32(&sd, "audio");
    let want = vec_f32(&sd, "probs");
    let mut diarizer = Diarizer::offline(NemotronDiar::from_hub(dtype.clone(), 1).unwrap()).unwrap();
    let got = diarizer.diarize(&audio, 16000).unwrap().probs;
    let diff = max_diff(&got, &want);
    let flips = got.iter().zip(&want).filter(|(a, b)| (**a > 0.5) != (**b > 0.5)).count();
    eprintln!("{dtype:?}: probs max abs diff {diff:.3e}, {flips} of {} decisions flipped", want.len());
    assert!(flips * 1000 <= want.len(), "{dtype:?}: {flips} speaker decisions flipped");
}
