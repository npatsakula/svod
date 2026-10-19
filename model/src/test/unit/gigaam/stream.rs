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
    fn zeros(cfg: &GigaAmConfig, lanes: usize) -> Self {
        let streaming = cfg.streaming.expect("streaming");
        let (d, left) = (cfg.d_model, cfg.subs_kernel_size - 1);
        let cap = streaming.cache_frames(cfg.max_encoder_frames);
        let heads = cfg.n_layers * cfg.n_heads;
        let f32 = DType::Float32;
        Self {
            subsampling: [
                Tensor::zeros(&[lanes, cfg.n_mels, left], f32.clone()),
                Tensor::zeros(&[lanes, d, left], f32.clone()),
            ],
            conv: Tensor::zeros(&[lanes, cfg.n_layers, d, cfg.conv_kernel - 1], f32.clone()),
            keys: Tensor::zeros(&[lanes, cap, heads, d / cfg.n_heads], f32.clone()),
            values: Tensor::zeros(&[lanes, cap, heads, d / cfg.n_heads], f32.clone()),
            pending: Tensor::zeros(&[lanes, streaming.chunk_size, d], f32),
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
    let flat = out.out.to_vec::<f32>().expect("read");
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

    let mut caches = Caches::zeros(&cfg, 2);
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
/// partial.
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

    let mut caches = Caches::zeros(&cfg, 2);
    let (mut emitted, mut pending) = ([0usize; 2], [0usize; 2]);
    let joins = [0, 1];
    let lengths = [long, short];
    let mut flushed = [false; 2];
    for round in 0..8 {
        let mut valid = [0i32; 2];
        let mut flush = [0i32; 2];
        let mut chunks = Vec::new();
        for lane in 0..2 {
            let first = (round - joins[lane].min(round)) * MEL_CHUNK;
            let fed = round >= joins[lane] && first < lengths[lane];
            valid[lane] = if fed { (lengths[lane] - first).min(MEL_CHUNK) as i32 } else { 0 };
            flush[lane] = i32::from(round >= joins[lane] && first >= lengths[lane] && !flushed[lane]);
            chunks.push(window(&signals[lane], n_mels, first, MEL_CHUNK));
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
    let caches = Caches::zeros(&config(streaming), 1);
    let zero = ints(&[0]);
    let cursors = StreamCursors { valid: &zero, emitted: &zero, pending: &zero, flush: &zero };
    let mel = Tensor::zeros(&[1, cfg.n_mels, MEL_CHUNK], DType::Float32);
    assert!(encoder.forward_stream(&mel, cursors, caches.borrow()).is_err());
}
