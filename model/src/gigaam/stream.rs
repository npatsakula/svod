//! Live streaming GigaAM: a [`GigaAmStreamer`] serves many concurrent audio
//! streams through one chunk-shaped encoder step and one decoder round per
//! chunk of 40 ms encoder frames.
//!
//! Every stream owns a lane. Audio pushed to a stream is framed as it
//! arrives; [`GigaAmStreamer::step`] then encodes every lane holding a whole
//! chunk (or a finished stream's last partial one) in one batched step
//! against the lanes' device-resident caches, decodes the emitted frames
//! (CTC greedy with the last label carried, or RN-T greedy with the predictor
//! state resident per lane) and returns what each stream emitted. The result
//! of a stream equals the whole-sequence forward of the streaming encoder
//! over its audio, whatever the pushes and whoever shares the step.

use snafu::{OptionExt, ResultExt, Snafu};
use svod_arch::ctc::GreedyDecoder;
use svod_arch::pipelines::audio::{Transcript, words_to_text};
use svod_arch::rnnt::{RnntDecoder, RnntOpts, TokenEmission};
use svod_dtype::DType;
use svod_macros::jit_wrapper;
use svod_tensor::PrepareConfig;

use crate::audio::{FrameCursor, MelJit, MelSpectrogram};
use crate::gigaam::encoder::{StreamCaches, StreamCursors};
use crate::gigaam::model::{GigaAm, Head};
use crate::gigaam::rnnt::RnntBlockBackend;
use crate::gigaam::transcribe::{ctc_frames_to_words, pack_lengths_buffer};
use crate::gigaam::{Error as ModelError, StreamingConfig};
use crate::jit::{InputSpec, JitError};

// One streaming step over every lane: the chunk's mel frames, the lanes'
// cursors and the key and value rings in, the small caches recycled in
// place, and out the emitted frames as CTC argmax labels `[S, chunk]` or the
// RN-T joint's projection `[S, chunk, joint_hidden]` plus the rows the host
// copies into the rings.
jit_wrapper! {
    GigaAmStreamJit(GigaAm) {
        inputs { mel: Tensor, valid: Tensor, emitted: Tensor, pending_len: Tensor, flush: Tensor, keys: Tensor, values: Tensor }
        state { subs0: Tensor, subs1: Tensor, conv: Tensor, pending: Tensor }
        outputs { frames, new_keys, new_values }

        build(mel, valid, emitted, pending_len, flush, keys, values, subs0, subs1, conv, pending) {
            let cursors = StreamCursors { valid, emitted, pending: pending_len, flush };
            let caches = StreamCaches { subsampling: [subs0, subs1], conv, keys, values, pending };
            let out: crate::gigaam::error::Result<_> = (|| {
                let step = model.encoder.forward_stream(mel, cursors, caches)?;
                let frames = step.out.cast(DType::Float32);
                let frames = match &model.head {
                    Head::Ctc(head) => head.forward(&frames.try_transpose(1, 2)?)?.argmax(-1isize)?,
                    Head::Rnnt { head, .. } => head.joint.project_encoder(&frames)?,
                };
                let [subs0, subs1] = step.subsampling;
                Ok((frames, step.keys, step.values, subs0, subs1, step.conv, step.pending))
            })();
            out
        }
    }
}

#[derive(Debug, Snafu)]
#[snafu(visibility(pub(crate)))]
pub enum StreamError {
    #[snafu(display("{source}"), context(false))]
    Jit {
        #[snafu(source(from(JitError, Box::new)))]
        source: Box<JitError>,
    },
    #[snafu(display("{source}"), context(false))]
    Model {
        #[snafu(source(from(ModelError, Box::new)))]
        source: Box<ModelError>,
    },
    #[snafu(display("{source}"), context(false))]
    Device {
        #[snafu(source(from(svod_device::error::Error, Box::new)))]
        source: Box<svod_device::error::Error>,
    },
    #[snafu(display("{source}"))]
    RnntDecode {
        #[snafu(source(from(svod_arch::rnnt::RnntDecodeError<JitError>, Box::new)))]
        source: Box<svod_arch::rnnt::RnntDecodeError<JitError>>,
    },
    #[snafu(display("the model is not a streaming one; enable streaming on it first"))]
    NotStreaming,
    #[snafu(display("stream {id:?} is not open"))]
    UnknownStream { id: StreamId },
}

pub type Result<T> = std::result::Result<T, StreamError>;

/// A stream open on a [`GigaAmStreamer`]: its lane, and a generation so a
/// closed stream's id is not mistaken for its lane's next occupant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StreamId {
    lane: u32,
    generation: u32,
}

/// What a [`GigaAmStreamer::step`] emitted for one stream.
#[derive(Clone, Debug, PartialEq)]
pub struct StreamEvent {
    pub id: StreamId,
    /// New tokens, frames absolute within the stream (40 ms each).
    pub emissions: Vec<TokenEmission>,
    /// The stream's last chunk is out: the id is closed and this is its
    /// whole transcript.
    pub transcript: Option<Transcript>,
}

/// One stream's host state.
struct Lane {
    generation: u32,
    cursor: FrameCursor,
    /// Mel frames fed to the encoder so far.
    fed: usize,
    /// Encoder frames emitted so far.
    emitted: usize,
    /// Look-ahead variant: encoder frames of the chunk awaiting its look-ahead.
    pending: usize,
    /// `finish` was called: the rest of the audio goes out as it is.
    finished: bool,
    /// CTC: the last argmax label.
    prev: usize,
    emissions: Vec<TokenEmission>,
}

impl Lane {
    /// Mel frames framed but not yet fed.
    fn waiting(&self) -> usize {
        self.cursor.available().saturating_sub(self.fed)
    }

    /// Nothing more will come out: the audio has ended and been fed, and no
    /// chunk awaits its look-ahead.
    fn spent(&self) -> bool {
        self.finished && self.waiting() == 0 && self.pending == 0
    }
}

/// A lane's part in a step.
#[derive(Clone, Copy, Default)]
struct Feed {
    /// Real mel frames in the chunk.
    valid: usize,
    /// Look-ahead variant: emit the pending chunk without a look-ahead.
    flush: bool,
    /// Encoder frames the step emits for the lane.
    out: usize,
}

/// One instance per streamer, so the variants' size gap costs nothing.
#[allow(clippy::large_enum_variant)]
enum StreamHead {
    Ctc { decoder: GreedyDecoder, labels: Vec<i32> },
    Rnnt { backend: RnntBlockBackend, decoder: RnntDecoder },
}

/// Concurrent GigaAM streams over one device-resident batch of lanes.
pub struct GigaAmStreamer {
    model: GigaAm,
    streaming: StreamingConfig,
    mel: MelSpectrogram,
    mel_jit: MelJit,
    framed: Vec<f32>,
    jit: GigaAmStreamJit,
    head: StreamHead,
    lanes: Vec<Option<Lane>>,
    generation: u32,
    /// Mel frames per chunk.
    chunk_mel: usize,
    rounds: u64,
    /// The rings: chunk-sized slots per lane, and the bytes of one.
    slots: usize,
    slot_bytes: usize,
    /// Bytes of one lane's row in each small cache, zeroed when a lane opens.
    lane_bytes: [usize; 3],
}

impl GigaAmStreamer {
    /// Prepare the step for `lanes` concurrent streams of a streaming model
    /// (see [`GigaAm::enable_streaming`]).
    pub fn new(model: GigaAm, lanes: usize) -> Result<Self> {
        let streaming = model.config.streaming.context(NotStreamingSnafu)?;
        let cfg = &model.config;
        let mel = model.mel_spectrogram();
        let chunk = streaming.chunk_size;
        let chunk_mel = chunk * cfg.subsampling_factor;
        let framed_len = mel.frames_len(chunk_mel);
        let device_local = PrepareConfig::device_local();
        let mut mel_jit = MelJit::new(mel.clone());
        mel_jit.prepare_with_config(
            InputSpec::f32(&[lanes, framed_len]).device_local(),
            InputSpec::i32(&[lanes]),
            &device_local,
        )?;

        let dtype = model.encoder.input_dtype();
        let bytes = dtype.bytes();
        let (d, h, layers, left) = (cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.subs_kernel_size - 1);
        let cap = streaming.cache_frames(cfg.max_encoder_frames);
        let cache = |shape: &[usize]| InputSpec::new(shape, dtype.clone()).device_local();
        let mut jit = GigaAmStreamJit::new(model.clone());
        jit.prepare_with_config(
            InputSpec::f32(&[lanes, cfg.n_mels, chunk_mel]).device_local(),
            InputSpec::i32(&[lanes]),
            InputSpec::i32(&[lanes]),
            InputSpec::i32(&[lanes]),
            InputSpec::i32(&[lanes]),
            cache(&[lanes, cap, layers * h, d / h]),
            cache(&[lanes, cap, layers * h, d / h]),
            cache(&[lanes, cfg.n_mels, left]),
            cache(&[lanes, d, left]),
            cache(&[lanes, layers, d, cfg.conv_kernel - 1]),
            cache(&[lanes, chunk, d]),
            &device_local,
        )?;
        let (slots, slot_bytes) = (cap / chunk, chunk * layers * d * bytes);
        let lane_bytes = [cfg.n_mels * left * bytes, d * left * bytes, layers * d * (cfg.conv_kernel - 1) * bytes];

        let head = match &model.head {
            Head::Ctc(_) => {
                let vocabulary = model.config.decoder.vocabulary().to_vec();
                StreamHead::Ctc { decoder: GreedyDecoder::new(vocabulary), labels: vec![0; lanes * chunk] }
            }
            Head::Rnnt { runtime, .. } => {
                let backend = RnntBlockBackend::from_model(model.clone(), lanes, chunk)?;
                let opts = RnntOpts { max_symbols_per_step: runtime.max_symbols_per_step };
                StreamHead::Rnnt { backend, decoder: RnntDecoder::new(runtime.vocabulary.clone(), opts) }
            }
        };
        Ok(Self {
            model,
            streaming,
            mel,
            mel_jit,
            framed: vec![0.0; lanes * framed_len],
            jit,
            head,
            lanes: (0..lanes).map(|_| None).collect(),
            generation: 0,
            chunk_mel,
            rounds: 0,
            slots,
            slot_bytes,
            lane_bytes,
        })
    }

    pub fn lanes(&self) -> usize {
        self.lanes.len()
    }

    /// Lanes no stream holds.
    pub fn free_lanes(&self) -> usize {
        self.lanes.iter().filter(|l| l.is_none()).count()
    }

    pub fn sample_rate(&self) -> usize {
        self.model.config.sample_rate
    }

    /// Batched steps run so far.
    pub fn rounds(&self) -> u64 {
        self.rounds
    }

    /// The sizes of every device buffer the streamer's plans own, in bytes:
    /// the caches, the step's intermediates and the decoder's state. Their
    /// sum is the streamer's device footprint besides the model's weights.
    pub fn device_buffers(&self) -> Result<Vec<usize>> {
        let mut sizes: Vec<usize> = self.jit.buffers()?.iter().map(|b| b.size()).collect();
        sizes.extend(self.mel_jit.buffers()?.iter().map(|b| b.size()));
        if let StreamHead::Rnnt { backend, .. } = &self.head {
            sizes.extend(backend.device_buffers()?);
        }
        Ok(sizes)
    }

    /// Seconds per encoder frame.
    pub fn frame_shift(&self) -> f32 {
        (self.model.config.hop_length * self.model.config.subsampling_factor) as f32 / self.sample_rate() as f32
    }

    /// Audio a frame waits for before it can come out: its chunk, and the
    /// look-ahead chunk when there is one.
    pub fn latency_seconds(&self) -> f32 {
        self.streaming.step_frames() as f32 * self.frame_shift()
    }

    /// Open a stream on a free lane; `None` when every lane is taken. The
    /// lane's convolution caches start as zeros, a stream's silence before
    /// its first frame; its ring slots are hidden until written.
    pub fn open(&mut self) -> Result<Option<StreamId>> {
        let Some(lane) = self.lanes.iter().position(Option::is_none) else { return Ok(None) };
        let zeros = vec![0u8; self.lane_bytes.iter().copied().max().unwrap_or(0)];
        self.jit.subs0_mut()?.copyin_at(lane * self.lane_bytes[0], &zeros[..self.lane_bytes[0]])?;
        self.jit.subs1_mut()?.copyin_at(lane * self.lane_bytes[1], &zeros[..self.lane_bytes[1]])?;
        self.jit.conv_mut()?.copyin_at(lane * self.lane_bytes[2], &zeros[..self.lane_bytes[2]])?;
        self.generation = self.generation.wrapping_add(1);
        let id = StreamId { lane: lane as u32, generation: self.generation };
        let blank = match &self.head {
            StreamHead::Ctc { decoder, .. } => decoder.blank_id(),
            StreamHead::Rnnt { decoder, .. } => decoder.blank_id(),
        };
        self.lanes[lane] = Some(Lane {
            generation: id.generation,
            cursor: self.mel.cursor(),
            fed: 0,
            emitted: 0,
            pending: 0,
            finished: false,
            prev: blank,
            emissions: Vec::new(),
        });
        Ok(Some(id))
    }

    fn lane_mut(&mut self, id: StreamId) -> Result<&mut Lane> {
        match self.lanes.get_mut(id.lane as usize) {
            Some(Some(lane)) if lane.generation == id.generation => Ok(lane),
            _ => Err(StreamError::UnknownStream { id }),
        }
    }

    fn lane(&self, id: StreamId) -> Result<&Lane> {
        match self.lanes.get(id.lane as usize) {
            Some(Some(lane)) if lane.generation == id.generation => Ok(lane),
            _ => Err(StreamError::UnknownStream { id }),
        }
    }

    /// Buffer mono samples at the model's rate; they are encoded by the next
    /// steps, a chunk at a time.
    pub fn push(&mut self, id: StreamId, samples: &[f32]) -> Result<()> {
        let lane = self.lane_mut(id)?;
        if !lane.finished {
            lane.cursor.push(samples);
        }
        Ok(())
    }

    /// End a stream's audio: the next steps emit the rest, then close it.
    pub fn finish(&mut self, id: StreamId) -> Result<()> {
        let lane = self.lane_mut(id)?;
        if !lane.finished {
            lane.finished = true;
            lane.cursor.finish();
        }
        Ok(())
    }

    /// Drop a stream without emitting the rest.
    pub fn close(&mut self, id: StreamId) -> Result<()> {
        self.lane_mut(id)?;
        self.lanes[id.lane as usize] = None;
        Ok(())
    }

    /// Tokens the stream has emitted so far.
    pub fn emissions(&self, id: StreamId) -> Result<&[TokenEmission]> {
        Ok(&self.lane(id)?.emissions)
    }

    /// The stream's transcript so far, with word times in seconds from its start.
    pub fn transcript(&self, id: StreamId) -> Result<Transcript> {
        Ok(self.assemble(&self.lane(id)?.emissions))
    }

    fn assemble(&self, emissions: &[TokenEmission]) -> Transcript {
        let shift = self.frame_shift();
        let words = match &self.head {
            StreamHead::Rnnt { decoder, .. } => decoder.frames_to_words(emissions, shift),
            StreamHead::Ctc { decoder, .. } => {
                let vocabulary = decoder.vocabulary();
                let text: String = emissions.iter().map(|e| vocabulary[e.token_id].as_str()).collect();
                let frames: Vec<usize> = emissions.iter().map(|e| e.frame).collect();
                ctc_frames_to_words(&text, &frames, shift)
            }
        };
        Transcript { text: words_to_text(&words), words, ..Default::default() }
    }

    /// What each lane feeds the next step, or nothing for every lane.
    fn feeds(&self) -> Option<Vec<Feed>> {
        let look_ahead = self.streaming.right_context_chunks > 0;
        let mut any = false;
        let feeds = self
            .lanes
            .iter()
            .map(|lane| {
                let Some(lane) = lane else { return Feed::default() };
                let waiting = lane.waiting();
                let (valid, flush) = if waiting >= self.chunk_mel {
                    (self.chunk_mel, false)
                } else if lane.finished && waiting > 0 {
                    (waiting, false)
                } else if lane.finished && look_ahead && lane.pending > 0 {
                    (0, true)
                } else {
                    return Feed::default();
                };
                any = true;
                let out = if look_ahead { lane.pending } else { self.model.encoder.subsampling_output_length(valid) };
                Feed { valid, flush, out }
            })
            .collect();
        any.then_some(feeds)
    }

    /// Run steps until no stream holds a whole chunk (or a finished stream's
    /// remainder); returns every emission, in lane order per step. A finished
    /// stream with nothing left to come out is closed, its last event carrying
    /// the transcript.
    pub fn step(&mut self) -> Result<Vec<StreamEvent>> {
        let mut events = Vec::new();
        loop {
            self.retire(&mut events);
            let Some(feeds) = self.feeds() else { break };
            self.run(&feeds, &mut events)?;
        }
        Ok(events)
    }

    /// Close every spent stream with its transcript.
    fn retire(&mut self, events: &mut Vec<StreamEvent>) {
        for i in 0..self.lanes.len() {
            if self.lanes[i].as_ref().is_some_and(Lane::spent) {
                let lane = self.lanes[i].take().expect("a spent lane");
                let id = StreamId { lane: i as u32, generation: lane.generation };
                let transcript = Some(self.assemble(&lane.emissions));
                events.push(StreamEvent { id, emissions: Vec::new(), transcript });
            }
        }
    }

    fn run(&mut self, feeds: &[Feed], events: &mut Vec<StreamEvent>) -> Result<()> {
        self.rounds += 1;
        let lanes = self.lanes.len();
        let framed_len = self.framed.len() / lanes;
        let look_ahead = self.streaming.right_context_chunks > 0;
        let (mut valid, mut emitted, mut pending, mut flush) =
            (vec![0; lanes], vec![0; lanes], vec![0; lanes], vec![0; lanes]);
        for (i, feed) in feeds.iter().enumerate() {
            let Some(lane) = &self.lanes[i] else { continue };
            if feed.valid > 0 {
                lane.cursor.stage(lane.fed, &mut self.framed[i * framed_len..(i + 1) * framed_len]);
            }
            valid[i] = feed.valid;
            emitted[i] = lane.emitted;
            pending[i] = lane.pending;
            flush[i] = usize::from(feed.flush);
        }
        self.mel_jit.framed_mut()?.copyin(bytemuck::cast_slice(&self.framed))?;
        pack_lengths_buffer(self.mel_jit.frames_view_mut::<i32>()?, &valid);
        self.mel_jit.execute()?;
        self.jit.mel_mut()?.copy_from(self.mel_jit.output()?)?;
        pack_lengths_buffer(self.jit.valid_view_mut::<i32>()?, &valid);
        pack_lengths_buffer(self.jit.emitted_view_mut::<i32>()?, &emitted);
        pack_lengths_buffer(self.jit.pending_len_view_mut::<i32>()?, &pending);
        pack_lengths_buffer(self.jit.flush_view_mut::<i32>()?, &flush);
        self.jit.execute()?;

        let outs: Vec<usize> = feeds.iter().map(|f| f.out).collect();
        let chunk = self.streaming.chunk_size;
        let shift_frames: Vec<usize> = (0..lanes).map(|i| self.lanes[i].as_ref().map_or(0, |l| l.emitted)).collect();
        let mut new: Vec<Vec<TokenEmission>> = vec![Vec::new(); lanes];
        match &mut self.head {
            StreamHead::Ctc { decoder, labels } => {
                self.jit.frames()?.copyout_prefix(bytemuck::cast_slice_mut(labels))?;
                for (i, lane) in self.lanes.iter_mut().enumerate() {
                    let Some(lane) = lane else { continue };
                    let run: Vec<usize> = labels[i * chunk..i * chunk + outs[i]].iter().map(|&l| l as usize).collect();
                    new[i] = decoder
                        .collapse(&run, &mut lane.prev)
                        .into_iter()
                        .map(|(token_id, at)| TokenEmission { token_id, frame: shift_frames[i] + at })
                        .collect();
                }
            }
            StreamHead::Rnnt { backend, decoder } => {
                // A lane decoding its first chunk starts the predictor afresh.
                let fresh: Vec<bool> = self.lanes.iter().map(|l| l.as_ref().is_some_and(|l| l.emitted == 0)).collect();
                backend.begin_chunk(&outs, &fresh)?;
                backend.bind_projected(self.jit.frames()?)?;
                let decoded = decoder.resume_batch_blocks(&outs, backend).context(RnntDecodeSnafu)?;
                for (i, (_, emissions)) in decoded.into_iter().enumerate() {
                    new[i] = emissions
                        .into_iter()
                        .map(|e| TokenEmission { token_id: e.token_id, frame: shift_frames[i] + e.frame })
                        .collect();
                }
            }
        }

        // The emitted rows take their lane's next ring slot, the one written
        // `cap` frames ago, as device copies queued after the step.
        for (i, feed) in feeds.iter().enumerate() {
            let Some(lane) = &self.lanes[i] else { continue };
            if feed.out == 0 {
                continue;
            }
            let slot = (lane.emitted / chunk) % self.slots;
            let (dst, src, len) = ((i * self.slots + slot) * self.slot_bytes, i * self.slot_bytes, self.slot_bytes);
            self.jit.copy_output_to_keys(1, dst, src, len)?;
            self.jit.copy_output_to_values(2, dst, src, len)?;
        }
        for (i, feed) in feeds.iter().enumerate() {
            let Some(lane) = &mut self.lanes[i] else { continue };
            if feed.valid == 0 && !feed.flush {
                continue;
            }
            lane.fed += feed.valid;
            lane.cursor.discard(lane.fed);
            lane.emitted += feed.out;
            lane.pending =
                if look_ahead && !feed.flush { self.model.encoder.subsampling_output_length(feed.valid) } else { 0 };
            lane.emissions.extend_from_slice(&new[i]);
            if !new[i].is_empty() {
                let id = StreamId { lane: i as u32, generation: lane.generation };
                events.push(StreamEvent { id, emissions: std::mem::take(&mut new[i]), transcript: None });
            }
        }
        Ok(())
    }
}
