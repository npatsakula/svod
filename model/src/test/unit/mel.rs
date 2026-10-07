//! Tests: the graph mel front-end against a naive host DFT written here, so
//! the reference shares no transform code with the graph under test.

use std::f64::consts::TAU;

use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::{MelNorm, MelScale as FbScale};
use test_case::test_case;

use crate::audio::{FrameCursor, MelConfig, MelLog, MelScale, MelSpectrogram, PadMode};
use crate::whisper::{N_FRAMES, N_SAMPLES, WhisperMel};

fn gigaam_config() -> MelConfig {
    MelConfig {
        sample_rate: 16000,
        n_fft: 320,
        hop_length: 160,
        win_length: 320,
        n_mels: 64,
        center: true,
        mel_scale: MelScale::Htk,
        periodic: true,
        pad_mode: crate::audio::PadMode::Reflect,
        preemphasis: None,
        log: crate::audio::MelSpectrogram::LOG,
    }
}

fn whisper_config() -> MelConfig {
    MelConfig {
        sample_rate: 16000,
        n_fft: 400,
        hop_length: 160,
        win_length: 400,
        n_mels: 80,
        center: true,
        mel_scale: MelScale::Slaney,
        periodic: true,
        pad_mode: crate::audio::PadMode::Reflect,
        preemphasis: None,
        log: crate::audio::MelSpectrogram::LOG,
    }
}

/// Speech-like synthetic signal: a few tones with drifting amplitude plus
/// deterministic noise, so every mel band carries energy across the clip.
fn synthetic(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..len)
        .map(|i| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
            let t = i as f32 / 16000.0;
            let env = 0.5 + 0.5 * (2.0 * std::f32::consts::PI * 0.7 * t).sin();
            env * (0.4 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                + 0.3 * (2.0 * std::f32::consts::PI * 1375.0 * t).sin()
                + 0.2 * (2.0 * std::f32::consts::PI * 4310.0 * t).sin())
                + 0.05 * noise
        })
        .collect()
}

/// The first two seconds of the untracked `ru_clip_0.wav` from the repository
/// root, if it is there and is 16 kHz mono int16.
fn real_clip() -> Option<Vec<f32>> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../ru_clip_0.wav");
    let mut reader = hound::WavReader::open(path).ok()?;
    let spec = reader.spec();
    ((spec.channels, spec.sample_rate) == (1, 16000)).then_some(())?;
    reader.samples::<i16>().take(16000 * 2).map(|s| Some(s.ok()? as f32 / 32768.0)).collect()
}

fn max_abs_diff(a: &[f32], b: &[f64]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(&x, &y)| (f64::from(x) - y).abs()).fold(0.0f64, f64::max) as f32
}

// =========================================================================
// Host reference
// =========================================================================

/// `torch.nn.functional.pad(mode="reflect")`: mirror without repeating edges.
fn reflect_pad(x: &[f64], pad: usize) -> Vec<f64> {
    let n = x.len();
    let mut out = Vec::with_capacity(n + 2 * pad);
    out.extend((0..pad).rev().map(|j| x[j + 1]));
    out.extend_from_slice(x);
    out.extend((0..pad).map(|j| x[n - 2 - j]));
    out
}

/// Naive mel power `[n_mels, T]`: reflect padding under `center`, periodic
/// Hann, a windowed DFT per frame, `re² + im²`, the filterbank (the op's
/// table, in f64). Frames of zeros are skipped — they hold zero power — so a
/// short signal `pad_or_trim`med to 30 s stays cheap.
fn ref_mel_power(x: &[f32], config: &MelConfig) -> Vec<f64> {
    assert_eq!(config.win_length, config.n_fft, "the reference frames a full-length window");
    let (n_fft, hop, n_mels) = (config.n_fft, config.hop_length, config.n_mels);
    let n_bins = n_fft / 2 + 1;
    let (scale, norm) = match config.mel_scale {
        MelScale::Htk => (FbScale::Htk, None),
        MelScale::Slaney => (FbScale::Slaney, Some(MelNorm::Slaney)),
    };
    let f_max = config.sample_rate as f64 / 2.0;
    let fb: Vec<f64> =
        Tensor::mel_filterbank(config.sample_rate, n_fft, n_mels, 0.0, f_max, scale, norm, DType::Float32)
            .unwrap()
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(f64::from)
            .collect();
    let win: Vec<f64> = (0..n_fft).map(|k| 0.5 - 0.5 * (TAU * k as f64 / n_fft as f64).cos()).collect();

    let raw: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
    let sig = if config.center { reflect_pad(&raw, n_fft / 2) } else { raw };
    let frames = (sig.len() - n_fft) / hop + 1;
    let mut out = vec![0.0; n_mels * frames];
    for t in 0..frames {
        let frame = &sig[t * hop..t * hop + n_fft];
        if frame.iter().all(|&v| v == 0.0) {
            continue;
        }
        let power: Vec<f64> = (0..n_bins)
            .map(|k| {
                let (re, im) = (0..n_fft).fold((0.0, 0.0), |(re, im), n| {
                    let angle = TAU * ((k * n) % n_fft) as f64 / n_fft as f64;
                    let v = frame[n] * win[n];
                    (re + v * angle.cos(), im - v * angle.sin())
                });
                re * re + im * im
            })
            .collect();
        for m in 0..n_mels {
            out[m * frames + t] = (0..n_bins).map(|k| fb[m * n_bins + k] * power[k]).sum();
        }
    }
    out
}

/// GigaAM's `torch.log(mel.clamp(1e-9, 1e9))`, `[n_mels, num_frames]`.
fn ref_gigaam_log_mel(x: &[f32]) -> Vec<f64> {
    ref_mel_power(x, &gigaam_config()).into_iter().map(|v| v.clamp(1e-9, 1e9).ln()).collect()
}

/// `whisper.audio.log_mel_spectrogram` after `pad_or_trim`: the trailing
/// frame dropped, `log10(max(x, 1e-10))`, floored 8 below the maximum,
/// `(x + 4) / 4`; `[80, N_FRAMES]`.
fn ref_whisper_log_mel(x: &[f32]) -> Vec<f64> {
    let mut padded = vec![0.0f32; N_SAMPLES];
    WhisperMel::pad_or_trim_into(x, &mut padded);
    let power = ref_mel_power(&padded, &whisper_config());
    let frames = power.len() / 80;
    assert_eq!(frames, N_FRAMES + 1);
    let mut logged = Vec::with_capacity(80 * N_FRAMES);
    for m in 0..80 {
        logged.extend((0..N_FRAMES).map(|t| power[m * frames + t].max(1e-10).log10()));
    }
    let floor = logged.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - 8.0;
    logged.into_iter().map(|v| (v.max(floor) + 4.0) / 4.0).collect()
}

// =========================================================================
// Graph path against the reference
// =========================================================================

/// One graph batch of host-framed rows against the reference of each window
/// on its own; the graph's columns past a window's frame count must be zero,
/// as an encoder's mel input expects them.
fn assert_gigaam_parity(windows: &[&[f32]], label: &str) {
    let mel = MelSpectrogram::new(&gigaam_config());
    let max_frames = windows.iter().map(|w| mel.num_frames(w.len())).max().unwrap();
    let framed_len = (max_frames - 1) * 160 + 320;
    let mut framed = vec![0.0f32; windows.len() * framed_len];
    for (row, window) in framed.chunks_mut(framed_len).zip(windows) {
        mel.frame_into(window, row);
    }
    let frames: Vec<i32> = windows.iter().map(|w| mel.num_frames(w.len()) as i32).collect();
    let graph = mel
        .forward_tensor(
            &Tensor::from_slice(framed).try_reshape([windows.len() as isize, framed_len as isize]).unwrap(),
            &Tensor::from_slice(frames.clone()),
        )
        .unwrap();
    assert_eq!(graph.dims().unwrap(), vec![windows.len(), 64, max_frames]);
    let graph = graph.to_vec::<f32>().unwrap();

    let mut worst = 0.0f32;
    for (bi, window) in windows.iter().enumerate() {
        let valid = frames[bi] as usize;
        let want = ref_gigaam_log_mel(window);
        assert_eq!(want.len(), 64 * valid);
        let row = &graph[bi * 64 * max_frames..(bi + 1) * 64 * max_frames];
        for m in 0..64 {
            let got = &row[m * max_frames..m * max_frames + valid];
            worst = worst.max(max_abs_diff(got, &want[m * valid..(m + 1) * valid]));
            assert!(row[m * max_frames + valid..(m + 1) * max_frames].iter().all(|&v| v == 0.0), "unmasked tail");
        }
    }
    eprintln!("gigaam graph-vs-naive log-mel max abs diff ({label}): {worst:.3e}");
    assert!(worst <= 1e-3, "{label}: max abs diff {worst}");
}

fn assert_whisper_parity(windows: &[&[f32]], label: &str) {
    let mel = WhisperMel::new(80);
    let mut samples = vec![0.0f32; windows.len() * N_SAMPLES];
    for (row, window) in samples.chunks_mut(N_SAMPLES).zip(windows) {
        WhisperMel::pad_or_trim_into(window, row);
    }
    let graph = mel
        .forward_tensor(&Tensor::from_slice(samples).try_reshape([windows.len() as isize, N_SAMPLES as isize]).unwrap())
        .unwrap();
    assert_eq!(graph.dims().unwrap(), vec![windows.len(), 80, N_FRAMES]);
    let graph = graph.to_vec::<f32>().unwrap();
    let per_row = 80 * N_FRAMES;
    let worst = windows
        .iter()
        .enumerate()
        .map(|(bi, window)| max_abs_diff(&graph[bi * per_row..(bi + 1) * per_row], &ref_whisper_log_mel(window)))
        .fold(0.0, f32::max);
    eprintln!("whisper graph-vs-naive log-mel max abs diff ({label}): {worst:.3e}");
    assert!(worst <= 1e-3, "{label}: max abs diff {worst}");
}

#[test]
fn num_frames_follows_torch_stft() {
    let centered = MelSpectrogram::new(&whisper_config());
    assert_eq!((centered.num_frames(16000), centered.framed_len(16000)), (101, 16400));
    let snipped = MelSpectrogram::new(&MelConfig { center: false, ..gigaam_config() });
    assert_eq!((snipped.num_frames(16000), snipped.num_frames(319)), (99, 0));
}

/// Host framing (`framed_len`, `frame_into`) against the graph's own `center`
/// reflect padding, for an even and an odd `n_fft`: the frame counts agree
/// and the framed row transformed with `center = false` yields the graph's
/// centered log-mel.
#[test_case(400; "even n_fft")]
#[test_case(401; "odd n_fft")]
fn host_framing_matches_graph_centering(n_fft: usize) {
    let mel = MelSpectrogram::new(&MelConfig { n_fft, win_length: n_fft, ..whisper_config() });
    let signal = synthetic(16000, 5);
    let centered =
        mel.forward_power_tensor(&Tensor::from_slice(signal.clone())).unwrap().mel_log(MelSpectrogram::LOG).unwrap();
    let frames = centered.dim_const(-1).unwrap();
    assert_eq!(mel.num_frames(signal.len()), frames);
    assert_eq!(mel.framed_len(signal.len()), signal.len() + 2 * (n_fft / 2));

    let mut framed = vec![0.0f32; mel.framed_len(signal.len())];
    mel.frame_into(&signal, &mut framed);
    let framed = Tensor::from_slice(framed).try_unsqueeze(0).unwrap();
    let host = mel.forward_tensor(&framed, &Tensor::from_slice(vec![frames as i32])).unwrap();
    assert_eq!(host.dims().unwrap(), vec![1, 80, frames]);
    let (host, want) = (host.to_vec::<f32>().unwrap(), centered.to_vec::<f32>().unwrap());
    let worst = host.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(worst <= 1e-4, "host framing drifted from graph centering: {worst}");
}

#[test]
fn graph_mel_matches_naive_dft_on_synthetic_windows() {
    // Two VAD-style windows of different lengths plus a silent one share one
    // framed batch; silence must land on the clamp floor, `ln(1e-9)`.
    let (long, short) = (synthetic(16000 * 2 + 77, 1), synthetic(16000 + 5, 2));
    let silence = vec![0.0f32; 16000 / 2 + 3];
    assert_gigaam_parity(&[&long, &short, &silence], "synthetic");
}

#[test]
fn graph_whisper_mel_matches_naive_dft_on_synthetic_windows() {
    // Windows shorter than 30 s, which pad_or_trim zero-extends, and one
    // longer, which it cuts.
    let two_s = synthetic(16000 * 2, 3);
    let mut long = synthetic(16000 * 3 / 2, 4);
    long.resize(N_SAMPLES + 16000, 0.0);
    assert_whisper_parity(&[&two_s, &long], "synthetic");
}

#[test]
fn graph_mel_matches_naive_dft_on_real_clip() {
    let Some(clip) = real_clip() else {
        eprintln!("ru_clip_0.wav absent or not 16 kHz mono int16; skipping the real-clip parity check");
        return;
    };
    assert_gigaam_parity(&[&clip, &clip[16000 / 2..16000 + 321]], "ru_clip_0");
    assert_whisper_parity(&[&clip], "ru_clip_0");
}

// =========================================================================
// NeMo front-end and incremental framing
// =========================================================================

/// NeMo's `AudioToMelSpectrogramPreprocessor` as Nemotron-3-Diarization sets
/// it (`n_fft` 512 over a 400-sample symmetric Hann, zero `center` padding,
/// whole-signal pre-emphasis 0.97, 128 Slaney mels, `ln(x + 2^-24)`).
fn nemo_config() -> MelConfig {
    MelConfig {
        sample_rate: 16000,
        n_fft: 512,
        hop_length: 160,
        win_length: 400,
        n_mels: 128,
        center: true,
        mel_scale: MelScale::Slaney,
        periodic: false,
        pad_mode: PadMode::Zero,
        preemphasis: Some(0.97),
        log: MelLog::LnAdd { guard: 2f64.powi(-24) },
    }
}

/// Naive NeMo log-mel `[n_mels, L / hop + 1]`: pre-emphasis (first sample
/// kept), zero padding, the symmetric window centered in `n_fft`, a DFT per
/// frame, the Slaney filterbank, `ln(x + 2^-24)`.
fn ref_nemo_log_mel(x: &[f32]) -> Vec<f64> {
    let config = nemo_config();
    let (n_fft, hop, win_length, n_mels) = (config.n_fft, config.hop_length, config.win_length, config.n_mels);
    let n_bins = n_fft / 2 + 1;
    let fb: Vec<f64> = Tensor::mel_filterbank(
        16000,
        n_fft,
        n_mels,
        0.0,
        8000.0,
        FbScale::Slaney,
        Some(MelNorm::Slaney),
        DType::Float32,
    )
    .unwrap()
    .to_vec::<f32>()
    .unwrap()
    .into_iter()
    .map(f64::from)
    .collect();
    let left = (n_fft - win_length) / 2;
    let win: Vec<f64> = (0..n_fft)
        .map(|n| match n.checked_sub(left) {
            Some(k) if k < win_length => 0.5 - 0.5 * (TAU * k as f64 / (win_length - 1) as f64).cos(),
            _ => 0.0,
        })
        .collect();
    let emphasized: Vec<f64> =
        (0..x.len()).map(|n| f64::from(x[n]) - if n > 0 { 0.97 * f64::from(x[n - 1]) } else { 0.0 }).collect();
    let mut sig = vec![0.0; n_fft / 2];
    sig.extend(emphasized);
    sig.extend(std::iter::repeat_n(0.0, n_fft / 2));
    let frames = (sig.len() - n_fft) / hop + 1;
    let mut out = vec![0.0; n_mels * frames];
    for t in 0..frames {
        let frame = &sig[t * hop..t * hop + n_fft];
        let power: Vec<f64> = (0..n_bins)
            .map(|k| {
                let (re, im) = (0..n_fft).fold((0.0, 0.0), |(re, im), n| {
                    let angle = TAU * ((k * n) % n_fft) as f64 / n_fft as f64;
                    let v = frame[n] * win[n];
                    (re + v * angle.cos(), im - v * angle.sin())
                });
                re * re + im * im
            })
            .collect();
        for m in 0..n_mels {
            let energy: f64 = (0..n_bins).map(|k| fb[m * n_bins + k] * power[k]).sum();
            out[m * frames + t] = (energy + 2f64.powi(-24)).ln();
        }
    }
    out
}

/// Host framing and the graph's own `center` padding both reproduce the naive
/// NeMo front-end, pre-emphasis and zero padding included.
#[test]
fn nemo_front_end_matches_naive_reference() {
    let mel = MelSpectrogram::new(&nemo_config());
    let signal = synthetic(16000 / 2 + 37, 9);
    let frames = mel.num_frames(signal.len());
    assert_eq!(frames, signal.len() / 160 + 1);
    let want = ref_nemo_log_mel(&signal);

    let mut framed = vec![0.0f32; mel.framed_len(signal.len())];
    mel.frame_into(&signal, &mut framed);
    let host = mel
        .forward_tensor(&Tensor::from_slice(framed).try_unsqueeze(0).unwrap(), &Tensor::from_slice(vec![frames as i32]))
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    let graph = mel
        .forward_power_tensor(&Tensor::from_slice(signal.clone()))
        .unwrap()
        .mel_log(nemo_config().log)
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    for (label, got) in [("host framing", host), ("graph padding", graph)] {
        let worst = max_abs_diff(&got, &want);
        eprintln!("nemo {label} vs naive log-mel max abs diff: {worst:.3e}");
        assert!(worst <= 2e-3, "{label}: max abs diff {worst}");
    }
}

fn cursor_config(pad_mode: PadMode, preemphasis: Option<f32>, center: bool) -> MelConfig {
    MelConfig { pad_mode, preemphasis, center, n_fft: 64, win_length: 48, hop_length: 16, ..nemo_config() }
}

#[test]
#[should_panic(expected = "precedes the discarded prefix")]
fn staging_a_discarded_frame_panics() {
    let mel = MelSpectrogram::new(&cursor_config(PadMode::Zero, None, true));
    let mut cursor = mel.cursor();
    cursor.push(&synthetic(400, 3));
    cursor.discard(8);
    let mut out = vec![0.0f32; mel.frames_len(1)];
    cursor.stage(7, &mut out);
}

proptest::proptest! {
    /// A signal pushed in arbitrary pieces stages, for every frame run that
    /// has arrived, exactly the samples whole-signal framing stages — across
    /// discards of consumed frames.
    #[test]
    fn cursor_matches_whole_signal_framing(
        len in 70usize..600,
        cuts in proptest::collection::vec(1usize..90, 1..12),
        run in 1usize..6,
        zero_pad: bool,
        preemphasis: bool,
        center: bool,
    ) {
        let config = cursor_config(
            if zero_pad { PadMode::Zero } else { PadMode::Reflect },
            preemphasis.then_some(0.97),
            center,
        );
        let mel = MelSpectrogram::new(&config);
        let signal = synthetic(len, len as u32);
        let mut whole = vec![0.0f32; mel.framed_len(len)];
        mel.frame_into(&signal, &mut whole);

        let mut cursor = mel.cursor();
        let (mut fed, mut next) = (0, 0);
        let check = |cursor: &mut FrameCursor, next: &mut usize| -> Result<(), proptest::test_runner::TestCaseError> {
            while *next + run <= cursor.available() {
                let mut out = vec![f32::NAN; mel.frames_len(run)];
                cursor.stage(*next, &mut out);
                let from = *next * 16;
                proptest::prop_assert_eq!(&out[..], &whole[from..from + out.len()]);
                *next += 1;
                cursor.discard(*next);
            }
            Ok(())
        };
        for cut in cuts.iter().cycle() {
            if fed == len {
                break;
            }
            let to = (fed + cut).min(len);
            cursor.push(&signal[fed..to]);
            fed = to;
            check(&mut cursor, &mut next)?;
        }
        cursor.finish();
        proptest::prop_assert_eq!(cursor.available(), mel.num_frames(len));
        check(&mut cursor, &mut next)?;
    }
}
