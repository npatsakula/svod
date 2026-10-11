//! The streaming encoder on the CPU backend: a chunk-by-chunk run through
//! `forward_stream` reproduces the whole-sequence causal forward, however the
//! lanes join, and the look-ahead variant reproduces the forward truncated a
//! chunk after each emitted one.

use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::gigaam::encoder::{Encoder, StreamCaches, StreamCursors};
use crate::gigaam::{GigaAm, GigaAmConfig, StreamingConfig};

const CHUNK: usize = 4;
const MEL_CHUNK: usize = 4 * CHUNK;

fn config(streaming: StreamingConfig) -> GigaAmConfig {
    let mut cfg = super::super::batch::test_config();
    cfg.max_batch_size = 2;
    cfg.max_encoder_frames = 64;
    cfg.max_mel_frames = 256;
    cfg.streaming = Some(streaming);
    cfg
}

/// A deterministic mel signal of `frames` frames, `[n_mels, frames]` row-major.
fn signal(seed: usize, n_mels: usize, frames: usize) -> Vec<f32> {
    (0..n_mels * frames).map(|i| (((i + 13 * seed) * 7919) % 1000) as f32 / 500.0 - 1.0).collect()
}

/// `[lanes, n_mels, frames]` from per-lane `[n_mels, own frames]` signals, zero past a signal's end.
fn mel_tensor(lanes: &[&[f32]], n_mels: usize, frames: usize) -> Tensor {
    let mut data = Vec::with_capacity(lanes.len() * n_mels * frames);
    for lane in lanes {
        data.extend(window(lane, n_mels, 0, frames));
    }
    Tensor::from_slice(data).try_reshape([lanes.len() as isize, n_mels as isize, frames as isize]).expect("mel")
}

/// Mel frames `[first, first + count)` of a signal, zero past its end.
fn window(signal: &[f32], n_mels: usize, first: usize, count: usize) -> Vec<f32> {
    let frames = signal.len() / n_mels;
    let mut out = vec![0.0; n_mels * count];
    for mel in 0..n_mels {
        for t in 0..count {
            if first + t < frames {
                out[mel * count + t] = signal[mel * frames + first + t];
            }
        }
    }
    out
}

fn ints(values: &[i32]) -> Tensor {
    let t = Tensor::from_slice(values);
    t.realize().expect("realize");
    t
}

/// Encoder frames of a causal subsampling over `mel_frames` mel frames, none of none.
fn frames_of(mel_frames: usize) -> usize {
    mel_frames.div_ceil(2).div_ceil(2)
}

/// The whole-sequence forward's frames `[lane][frame][d]`.
fn offline(encoder: &Encoder, mel: &Tensor, lengths: &[i32]) -> Vec<Vec<Vec<f32>>> {
    let out = encoder.forward_batch(mel, &ints(lengths)).expect("forward").cast(DType::Float32);
    let (b, d, t) = (out.dim_const(0).unwrap(), out.dim_const(1).unwrap(), out.dim_const(2).unwrap());
    let flat = out.to_vec::<f32>().expect("read");
    (0..b).map(|l| (0..t).map(|f| (0..d).map(|c| flat[(l * d + c) * t + f]).collect()).collect()).collect()
}

/// Every cache a stream carries, zeroed: a stream's start.
struct Caches {
    subsampling: [Tensor; 2],
    conv: Tensor,
    keys: Tensor,
    values: Tensor,
    pending: Tensor,
}

impl Caches {
    fn zeros(cfg: &GigaAmConfig, lanes: usize, dtype: DType) -> Self {
        let streaming = cfg.streaming.expect("streaming");
        let (d, left) = (cfg.d_model, cfg.subs_kernel_size - 1);
        let cap = streaming.cache_frames(cfg.max_encoder_frames);
        let heads = cfg.n_layers * cfg.n_heads;
        Self {
            subsampling: [
                Tensor::zeros(&[lanes, cfg.n_mels, left], dtype.clone()),
                Tensor::zeros(&[lanes, d, left], dtype.clone()),
            ],
            conv: Tensor::zeros(&[lanes, cfg.n_layers, d, cfg.conv_kernel - 1], dtype.clone()),
            keys: Tensor::zeros(&[lanes, cap, heads, d / cfg.n_heads], dtype.clone()),
            values: Tensor::zeros(&[lanes, cap, heads, d / cfg.n_heads], dtype.clone()),
            pending: Tensor::zeros(&[lanes, streaming.chunk_size, d], dtype),
        }
    }

    fn borrow(&self) -> StreamCaches<'_> {
        StreamCaches {
            subsampling: [&self.subsampling[0], &self.subsampling[1]],
            conv: &self.conv,
            keys: &self.keys,
            values: &self.values,
            pending: &self.pending,
        }
    }
}

/// One step: the emitted frames `[lane][frame][d]` (all `chunk` of them) and
/// the caches realized for the next step.
fn step(encoder: &Encoder, mel: &Tensor, cursors: [&[i32]; 4], caches: Caches) -> (Vec<Vec<Vec<f32>>>, Caches) {
    let [valid, emitted, pending, flush] = cursors.map(ints);
    let cursors = StreamCursors { valid: &valid, emitted: &emitted, pending: &pending, flush: &flush };
    let out = encoder.forward_stream(mel, cursors, caches.borrow()).expect("step");
    let next = Caches {
        subsampling: out.subsampling,
        conv: out.conv,
        keys: out.keys,
        values: out.values,
        pending: out.pending,
    };
    let mut all = vec![&out.out, &next.conv, &next.keys, &next.values, &next.pending];
    all.extend(next.subsampling.iter());
    Tensor::realize_batch(all).expect("realize");
    let (b, chunk, d) = (out.out.dim_const(0).unwrap(), out.out.dim_const(1).unwrap(), out.out.dim_const(2).unwrap());
    let flat = out.out.cast(DType::Float32).to_vec::<f32>().expect("read");
    let frames = (0..b).map(|l| (0..chunk).map(|f| flat[(l * chunk + f) * d..][..d].to_vec()).collect()).collect();
    (frames, next)
}

fn assert_frames_match(what: &str, got: &[Vec<f32>], want: &[Vec<f32>]) {
    assert_eq!(got.len(), want.len(), "{what}: frame count");
    for (f, (g, w)) in got.iter().zip(want).enumerate() {
        let diff = g.iter().zip(w).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(g.iter().all(|x| x.is_finite()), "{what}: frame {f} is not finite");
        assert!(diff <= 2e-4, "{what}: frame {f} differs by {diff}");
    }
}

/// Lane 0 streams six chunks, the last one partial; lane 1 joins two steps
/// later with four chunks of its own signal. Each lane's chunks equal the
/// causal whole-sequence forward over its own signal.
#[test]
fn chunks_reproduce_the_causal_forward() {
    let streaming = StreamingConfig { chunk_size: CHUNK, left_context_chunks: Some(2), right_context_chunks: 0 };
    let cfg = config(streaming);
    let encoder = GigaAm::with_random_weights(cfg.clone()).encoder;
    let n_mels = cfg.n_mels;
    let (long, short) = (5 * MEL_CHUNK + 10, 4 * MEL_CHUNK);
    let signals = [signal(1, n_mels, long), signal(2, n_mels, short)];
    let mel = mel_tensor(&[&signals[0], &signals[1]], n_mels, 6 * MEL_CHUNK);
    let want = offline(&encoder, &mel, &[long as i32, short as i32]);

    let mut caches = Caches::zeros(&cfg, 2, DType::Float32);
    let mut emitted = [0usize; 2];
    for round in 0..6 {
        // Lane 0's chunk `round`; lane 1's chunk `round - 2`.
        let valid = |lane: usize, len: usize, first: usize| -> usize {
            if lane == 1 && round < 2 { 0 } else { len.saturating_sub(first).min(MEL_CHUNK) }
        };
        let firsts = [round * MEL_CHUNK, round.saturating_sub(2) * MEL_CHUNK];
        let valids = [valid(0, long, firsts[0]), valid(1, short, firsts[1])];
        let chunks =
            [window(&signals[0], n_mels, firsts[0], MEL_CHUNK), window(&signals[1], n_mels, firsts[1], MEL_CHUNK)];
        let mel = mel_tensor(&[&chunks[0], &chunks[1]], n_mels, MEL_CHUNK);
        let valid_i32 = valids.map(|v| v as i32);
        let emitted_i32 = emitted.map(|e| e as i32);
        let (frames, next) = step(&encoder, &mel, [&valid_i32, &emitted_i32, &[0, 0], &[0, 0]], caches);
        caches = next;
        for lane in 0..2 {
            let n = frames_of(valids[lane]);
            let got = &frames[lane][..n];
            let want = &want[lane][emitted[lane]..emitted[lane] + n];
            assert_frames_match(&format!("lane {lane}, round {round}"), got, want);
            emitted[lane] += n;
        }
    }
    assert_eq!(emitted, [frames_of(long), frames_of(short)]);
}

/// With one chunk of look-ahead and two chunks behind, chunk `c` comes out
/// once chunk `c + 1` is in and equals its frames of the hard-copy forward
/// over the whole signal; the flush emits the last chunk. Two lanes, the
/// second joining a round late with a shorter signal whose last chunk is
/// partial, and sitting out a round while holding a pending chunk.
#[test]
fn look_ahead_chunks_reproduce_the_hard_copy_forward() {
    let streaming = StreamingConfig { chunk_size: CHUNK, left_context_chunks: Some(2), right_context_chunks: 1 };
    let cfg = config(streaming);
    let encoder = GigaAm::with_random_weights(cfg.clone()).encoder;
    let n_mels = cfg.n_mels;
    let (long, short) = (5 * MEL_CHUNK, 3 * MEL_CHUNK + 6);
    let signals = [signal(3, n_mels, long), signal(4, n_mels, short)];
    let mel = mel_tensor(&[&signals[0], &signals[1]], n_mels, long);
    let want = offline(&encoder, &mel, &[long as i32, short as i32]);

    let mut caches = Caches::zeros(&cfg, 2, DType::Float32);
    let (mut emitted, mut pending) = ([0usize; 2], [0usize; 2]);
    let joins = [0, 1];
    let lengths = [long, short];
    let mut flushed = [false; 2];
    let mut fed_chunks = [0usize; 2];
    for round in 0..9 {
        let mut valid = [0i32; 2];
        let mut flush = [0i32; 2];
        let mut chunks = Vec::new();
        for lane in 0..2 {
            let first = fed_chunks[lane] * MEL_CHUNK;
            // Lane 1 sits round 3 out, its pending chunk waiting.
            let idle = lane == 1 && round == 3;
            let fed = round >= joins[lane] && first < lengths[lane] && !idle;
            valid[lane] = if fed { (lengths[lane] - first).min(MEL_CHUNK) as i32 } else { 0 };
            flush[lane] = i32::from(round >= joins[lane] && first >= lengths[lane] && !flushed[lane]);
            chunks.push(window(&signals[lane], n_mels, first, MEL_CHUNK));
            if fed {
                fed_chunks[lane] += 1;
            }
        }
        let mel = mel_tensor(&[&chunks[0], &chunks[1]], n_mels, MEL_CHUNK);
        let (emitted_i32, pending_i32) = (emitted.map(|e| e as i32), pending.map(|p| p as i32));
        let (frames, next) = step(&encoder, &mel, [&valid, &emitted_i32, &pending_i32, &flush], caches);
        caches = next;
        for lane in 0..2 {
            if pending[lane] > 0 && (valid[lane] > 0 || flush[lane] != 0) {
                let n = pending[lane];
                let want = &want[lane][emitted[lane]..emitted[lane] + n];
                assert_frames_match(&format!("lane {lane}, round {round}"), &frames[lane][..n], want);
                emitted[lane] += n;
                pending[lane] = frames_of(valid[lane] as usize);
            } else if valid[lane] > 0 {
                pending[lane] = frames_of(valid[lane] as usize);
            }
            if flush[lane] != 0 {
                flushed[lane] = true;
            }
        }
    }
    assert_eq!(emitted, [frames_of(long), frames_of(short)]);
    assert!(flushed.iter().all(|&f| f));
}

#[test]
fn the_offline_encoder_has_no_streaming_step() {
    let cfg = super::super::batch::test_config();
    let encoder = GigaAm::with_random_weights(cfg.clone()).encoder;
    let streaming = StreamingConfig { chunk_size: CHUNK, left_context_chunks: Some(1), right_context_chunks: 0 };
    let caches = Caches::zeros(&config(streaming), 1, DType::Float32);
    let zero = ints(&[0]);
    let cursors = StreamCursors { valid: &zero, emitted: &zero, pending: &zero, flush: &zero };
    let mel = Tensor::zeros(&[1, cfg.n_mels, MEL_CHUNK], DType::Float32);
    assert!(encoder.forward_stream(&mel, cursors, caches.borrow()).is_err());
}

// ---------------------------------------------------------------------------
// The streamer: lanes against the whole-sequence pipeline.
// ---------------------------------------------------------------------------

use svod_arch::ctc::{CtcDecoder, GreedyDecoder};
use svod_arch::rnnt::{RnntDecoder, RnntOpts, TokenEmission};
use test_case::test_case;

use crate::audio::{MelConfig, MelSpectrogram};
use crate::gigaam::rnnt::RnntBlockBackend;
use crate::gigaam::stream::GigaAmStreamer;
use crate::gigaam::{Head, TransducerConfig};

/// The test config with a head that emits: 33 single-character CTC tokens
/// (the first a space), or an eight-piece RN-T vocabulary.
fn head_config(rnnt: bool, streaming: StreamingConfig) -> GigaAmConfig {
    let mut cfg = config(streaming);
    if rnnt {
        let vocabulary: Vec<String> = (0..8).map(|i| format!("p{i}")).collect();
        cfg.transducer = Some(TransducerConfig {
            pred_hidden: 16,
            pred_rnn_layers: 1,
            joint_hidden: 16,
            num_classes: vocabulary.len() + 1,
            max_symbols_per_step: 3,
            vocabulary,
            sentencepiece: false,
        });
        cfg.vocab_size = 9;
    } else {
        let vocabulary: Vec<String> =
            std::iter::once(" ".to_string()).chain((0..32).map(|i| char::from(b'a' + i % 26).to_string())).collect();
        cfg.decoder = CtcDecoder::Greedy(GreedyDecoder::new(vocabulary));
    }
    cfg
}

fn mel_of(model: &GigaAm) -> MelSpectrogram {
    let cfg = &model.config;
    MelSpectrogram::new(&MelConfig {
        sample_rate: cfg.sample_rate,
        n_fft: cfg.n_fft,
        hop_length: cfg.hop_length,
        win_length: cfg.win_length,
        n_mels: cfg.n_mels,
        center: cfg.mel_center,
        mel_scale: crate::audio::MelScale::Htk,
        periodic: true,
        pad_mode: crate::audio::PadMode::Reflect,
        preemphasis: None,
        log: MelSpectrogram::LOG,
    })
}

fn waveform(seed: u32, samples: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2654435761).wrapping_add(12345);
    (0..samples)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        })
        .collect()
}

/// The whole-sequence pipeline over one waveform: mel, the streaming
/// encoder's forward, and the head's greedy decode.
fn offline_emissions(model: &GigaAm, wave: &[f32]) -> Vec<TokenEmission> {
    let mel = mel_of(model);
    let frames = mel.num_frames(wave.len());
    let mut framed = vec![0.0; mel.frames_len(frames)];
    mel.frame_into(wave, &mut framed);
    let framed = Tensor::from_slice(framed).try_reshape([1isize, -1]).unwrap();
    let features = mel.forward_tensor(&framed, &ints(&[frames as i32])).unwrap();
    let encoded = model.encoder.forward_batch(&features, &ints(&[frames as i32])).unwrap().cast(DType::Float32);
    let t = encoded.dim_const(2).unwrap();
    match &model.head {
        Head::Ctc(head) => {
            let labels = head.forward(&encoded).unwrap().argmax(-1isize).unwrap().to_vec::<i32>().unwrap();
            let labels: Vec<usize> = labels.iter().map(|&l| l as usize).collect();
            let decoder = GreedyDecoder::new(model.config.decoder.vocabulary().to_vec());
            let mut prev = decoder.blank_id();
            decoder
                .collapse(&labels, &mut prev)
                .into_iter()
                .map(|(token_id, frame)| TokenEmission { token_id, frame })
                .collect()
        }
        Head::Rnnt { runtime, .. } => {
            let rows = encoded.try_permute(&[0, 2, 1]).unwrap().contiguous().to_vec::<f32>().unwrap();
            let mut backend: RnntBlockBackend = RnntBlockBackend::from_model(model.clone(), 1, t).unwrap();
            backend.bind_batch(&[rows], &[t]).unwrap();
            let opts = RnntOpts { max_symbols_per_step: runtime.max_symbols_per_step };
            let decoder = RnntDecoder::new(runtime.vocabulary.clone(), opts);
            decoder.decode_batch_blocks(&[t], &mut backend).unwrap().swap_remove(0).1
        }
    }
}

/// A random model whose whole-sequence decode of `wave` emits something.
fn emitting_model(cfg: &GigaAmConfig, wave: &[f32]) -> (GigaAm, Vec<TokenEmission>) {
    (0..8)
        .find_map(|_| {
            let model = GigaAm::with_random_weights(cfg.clone());
            let emissions = offline_emissions(&model, wave);
            (!emissions.is_empty()).then_some((model, emissions))
        })
        .expect("eight random models emitted nothing")
}

/// Two streams through one streamer, the second joining three pushes late
/// with its own audio, both pushed in odd pieces: each ends with the tokens
/// and frames of the whole-sequence pipeline over its audio, the partial
/// results only ever growing.
#[test_case(false, 0; "ctc")]
#[test_case(false, 1; "ctc with look-ahead")]
#[test_case(true, 0; "rnnt")]
#[test_case(true, 1; "rnnt with look-ahead")]
fn streams_decode_like_the_whole_sequence(rnnt: bool, right_context_chunks: usize) {
    let streaming = StreamingConfig { chunk_size: CHUNK, left_context_chunks: Some(2), right_context_chunks };
    let cfg = head_config(rnnt, streaming);
    let waves = [waveform(1, 16000 * 2 + 777), waveform(2, 16000 + 1234)];
    let (model, want_a) = emitting_model(&cfg, &waves[0]);
    let want_b = offline_emissions(&model, &waves[1]);
    let mut streamer = GigaAmStreamer::new(model, 2).expect("streamer");
    assert_eq!(streamer.free_lanes(), 2);

    let a = streamer.open().unwrap();
    let mut b = None;
    let mut got = [Vec::new(), Vec::new()];
    let mut finals = [None, None];
    let (mut pos, push) = ([0usize; 2], [1234usize, 2000]);
    for round in 0..40 {
        if round == 3 {
            b = streamer.open();
            assert_eq!(streamer.free_lanes(), 0);
        }
        for (i, id) in [Some(a), b].into_iter().enumerate() {
            let Some(id) = id else { continue };
            if finals[i].is_some() {
                continue;
            }
            let end = (pos[i] + push[i]).min(waves[i].len());
            streamer.push(id, &waves[i][pos[i]..end]).unwrap();
            pos[i] = end;
            if end == waves[i].len() {
                streamer.finish(id).unwrap();
            }
        }
        for event in streamer.step().unwrap() {
            let i = usize::from(Some(event.id) != Some(a));
            got[i].extend(event.emissions.iter().cloned());
            if let Some(transcript) = event.transcript {
                finals[i] = Some(transcript);
            }
        }
        if finals.iter().all(Option::is_some) {
            break;
        }
    }
    assert_eq!(got[0], want_a, "stream a");
    assert_eq!(got[1], want_b, "stream b");
    assert_eq!(streamer.free_lanes(), 2, "both streams closed");
    let frames: Vec<usize> = got[0].iter().map(|e| e.frame).collect();
    assert!(frames.windows(2).all(|w| w[0] <= w[1]), "frames in order");
    assert!(streamer.transcript(a).is_err(), "a closed stream is gone");
    assert_eq!(streamer.latency_seconds(), 0.04 * (CHUNK * (1 + right_context_chunks)) as f32);
}

/// On a GPU the chunk step runs the tile kernels at the model's head size:
/// f16 lanes of head size 48 against the f16 whole-sequence forward.
#[test]
fn stream_step_runs_the_kernels() {
    let device = svod_dtype::default_device::default_device();
    if matches!(device, svod_dtype::DeviceSpec::Cpu) {
        eprintln!("skipped: no GPU");
        return;
    }
    let streaming = StreamingConfig { chunk_size: 8, left_context_chunks: Some(16), right_context_chunks: 0 };
    let mut cfg = config(streaming);
    cfg.d_model = 192;
    cfg.n_heads = 4;
    cfg.d_ff = 384;
    cfg.max_encoder_frames = 512;
    cfg.max_mel_frames = 2048;
    let sd = super::encoder::state_dict(&GigaAm::with_random_weights(cfg.clone()));
    let model = GigaAm::from_state_dict_with_encoder_dtype(&sd, cfg.clone(), None, DType::Float16).expect("load");
    let encoder = model.encoder;
    let n_mels = cfg.n_mels;
    let (chunk_mel, chunks) = (4 * 8, 6);
    let signals = [signal(5, n_mels, chunks * chunk_mel), signal(6, n_mels, 4 * chunk_mel + 20)];
    let mel = mel_tensor(&[&signals[0], &signals[1]], n_mels, chunks * chunk_mel);
    let want = offline(&encoder, &mel, &[(chunks * chunk_mel) as i32, (4 * chunk_mel + 20) as i32]);
    let mut caches = Caches::zeros(&cfg, 2, DType::Float16);
    let mut emitted = [0usize; 2];
    let mut worst = 0f32;
    for round in 0..chunks {
        let firsts = [round * chunk_mel; 2];
        let valids = [chunk_mel, (4 * chunk_mel + 20).saturating_sub(firsts[1]).min(chunk_mel)];
        let rows =
            [window(&signals[0], n_mels, firsts[0], chunk_mel), window(&signals[1], n_mels, firsts[1], chunk_mel)];
        let mel = mel_tensor(&[&rows[0], &rows[1]], n_mels, chunk_mel);
        let valid_i32 = valids.map(|v| v as i32);
        let emitted_i32 = emitted.map(|e| e as i32);
        let (frames, next) = step(&encoder, &mel, [&valid_i32, &emitted_i32, &[0, 0], &[0, 0]], caches);
        caches = next;
        for lane in 0..2 {
            let n = frames_of(valids[lane]);
            for (f, (g, w)) in frames[lane][..n].iter().zip(&want[lane][emitted[lane]..emitted[lane] + n]).enumerate() {
                let diff = g.iter().zip(w).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                assert!(g.iter().all(|x| x.is_finite()), "lane {lane} round {round} frame {f} not finite");
                worst = worst.max(diff);
            }
            emitted[lane] += n;
        }
    }
    eprintln!("f16 chunked vs whole-sequence: max |diff| {worst:.3e}");
    assert!(worst <= 3e-2, "max |diff| {worst}");
}
