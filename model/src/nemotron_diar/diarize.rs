//! Chunked inference driver: offline recordings and live streams run the same
//! loop, under a different [`Profile`].
//!
//! Every step encodes one chunk of a [`Session`] after its speaker cache and
//! FIFO, emits the chunk's probabilities and pushes the chunk into the cache.
//! Steps of several sessions batch together.
//!
//! Data movement per step: the chunk's audio goes up (one copy into a
//! device-local input), the probabilities come down (one copy, the step's only
//! sync). The cache's embeddings never leave the device: each session keeps
//! its last step input in a device buffer, copied on-device into the next
//! step's `previous` input, where the graph gathers the new context from it as
//! the host-side [`SpeakerCache`] layout says.
//!
//! One plan serves every step: it is compiled once, in [`Diarizer::new`], for
//! the largest step — `max_batch` sessions at the profile's full step
//! capacity. Smaller steps (a stream's first seconds, a short recording, fewer
//! ready sessions) run through it with their padding masked, so no length ever
//! recompiles it. The batch is the plan's bound variable: a step computes only
//! the sessions it holds.

use std::time::Instant;

use svod_arch::diarization::{Binarization, ContextRow, SpeakerCache, SpeakerSegment, speaker_segments};
use svod_device::{Buffer, BufferSpec};
use svod_dtype::DType;
use svod_runtime::{RunProfile, StageProfile};
use svod_tensor::PrepareConfig;

use crate::audio::FrameCursor;
use crate::jit::InputSpec;

use super::config::{Profile, StreamingMode};
use super::error::{FlushedSnafu, Result, SampleRateSnafu};
use super::jit::NemotronDiarStepJit;
use super::model::NemotronDiar;

/// Speaker probabilities of a recording, one row of `num_speakers` per mel
/// frame, speakers numbered by first arrival.
#[derive(Clone, Debug)]
pub struct Diarization {
    pub probs: Vec<f32>,
    pub num_speakers: usize,
    /// Seconds per row.
    pub frame_sec: f32,
}

impl Diarization {
    pub fn frames(&self) -> usize {
        self.probs.len() / self.num_speakers
    }

    pub fn segments(&self, binarization: &Binarization) -> Vec<SpeakerSegment> {
        speaker_segments(&self.probs, self.num_speakers, self.frame_sec, binarization)
    }
}

/// The state of one audio stream: its unprocessed audio, speaker cache, the
/// device copy of its last step input and the probabilities emitted so far.
pub struct Session {
    cursor: FrameCursor,
    cache: SpeakerCache,
    /// The last step input on the device, its first `previous_bytes` holding
    /// the rows `[0, context + chunk)` the next layout indexes.
    previous: Option<Buffer>,
    previous_bytes: usize,
    /// First encoder frame of the next chunk.
    next_frame: usize,
    probs: Vec<f32>,
}

impl Session {
    /// Append audio (mono, at the model's sample rate).
    pub fn push(&mut self, samples: &[f32]) -> Result<()> {
        snafu::ensure!(!self.cursor.is_finished(), FlushedSnafu);
        self.cursor.push(samples);
        Ok(())
    }

    /// End the audio: the remaining chunks become ready, the last ones shorter.
    ///
    /// The tail keeps the profile's chunking, as the reference's offline
    /// forward does (its streaming session API instead folds the whole
    /// remainder into one last chunk without look-ahead, so the last chunk of
    /// a live stream differs from that API by its look-ahead).
    pub fn finish(&mut self) {
        if !self.cursor.is_finished() {
            self.cursor.finish();
        }
    }

    /// Probabilities emitted so far, `[frames, num_speakers]`.
    pub fn probs(&self) -> &[f32] {
        &self.probs
    }

    /// Hand over the probabilities emitted since the last call.
    pub fn take_probs(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.probs)
    }
}

/// One session's share of a step, in encoder frames.
#[derive(Clone, Copy, Debug)]
struct StepPlan {
    /// Cached frames preceding the chunk.
    context: usize,
    /// Chunk frames emitted and pushed into the cache.
    chunk: usize,
    /// Chunk plus look-ahead frames encoded.
    input: usize,
    /// Of `input`, the frames holding audio (attended to).
    valid: usize,
    /// Valid mel frames of the staged audio.
    mel_valid: usize,
}

pub struct Diarizer {
    model: NemotronDiar,
    profile: Profile,
    /// The step plan, `max_batch` sessions of `capacity` encoder frames.
    jit: NemotronDiarStepJit,
    max_batch: usize,
    /// Encoder frames of the longest step: the rows of every step input.
    capacity: usize,
    /// Encoder frames one step stages: the chunk and its look-ahead.
    step_frames: usize,
    framed_len: usize,
    /// Host staging of the device-local inputs and the probabilities.
    framed: Vec<f32>,
    sources: Vec<i32>,
    probs: Vec<f32>,
    /// Per-step kernel profiles, while profiling.
    run_profile: Option<RunProfile>,
}

impl Diarizer {
    /// Whole recordings, under the checkpoint's offline chunking.
    pub fn offline(model: NemotronDiar) -> Result<Self> {
        let profile = model.config.offline;
        Self::new(model, profile)
    }

    /// Live streams with the model card's latency `mode`.
    pub fn streaming(model: NemotronDiar, mode: StreamingMode) -> Result<Self> {
        let profile = model.config.streaming_profile(mode);
        Self::new(model, profile)
    }

    /// Validate `profile` and compile the step plan for it.
    pub fn new(model: NemotronDiar, profile: Profile) -> Result<Self> {
        let config = &model.config;
        config.validate_profile(&profile)?;
        let step_frames = profile.chunk_len + profile.right_context;
        let framed_len = model.mel().frames_len(step_frames * config.subsampling_factor);
        let capacity = config.step_capacity(&profile);
        let (max_batch, hidden) = (config.max_batch, config.hidden_size);
        let mut jit = NemotronDiarStepJit::new(model.clone());
        jit.prepare_with_config(
            InputSpec::f32(&[max_batch, framed_len]).device_local(),
            InputSpec::i32(&[max_batch]),
            InputSpec::f32(&[max_batch, capacity, hidden]).device_local(),
            InputSpec::i32(&[max_batch, capacity]).device_local(),
            InputSpec::i32(&[max_batch]),
            InputSpec::i32(&[max_batch]),
            &PrepareConfig::device_local(),
        )?;
        Ok(Self {
            jit,
            max_batch,
            capacity,
            step_frames,
            framed_len,
            framed: Vec::new(),
            sources: Vec::new(),
            probs: Vec::new(),
            run_profile: None,
            model,
            profile,
        })
    }

    pub fn profile(&self) -> &Profile {
        &self.profile
    }

    pub fn num_speakers(&self) -> usize {
        self.model.config.num_speakers
    }

    /// Record the kernels of every step from now on, until
    /// [`take_profile`](Self::take_profile).
    pub fn start_profiling(&mut self) {
        self.run_profile = Some(RunProfile::default());
    }

    /// The steps recorded since [`start_profiling`](Self::start_profiling), one
    /// stage each; profiling stops.
    pub fn take_profile(&mut self) -> Option<RunProfile> {
        self.run_profile.take()
    }

    /// A new stream.
    pub fn session(&self) -> Session {
        Session {
            cursor: self.model.mel().cursor(),
            cache: SpeakerCache::new(self.model.config.cache_config(&self.profile))
                .expect("the profile was validated with the diarizer"),
            previous: None,
            previous_bytes: 0,
            next_frame: 0,
            probs: Vec::new(),
        }
    }

    /// Diarize one whole recording.
    pub fn diarize(&mut self, audio: &[f32], sample_rate: usize) -> Result<Diarization> {
        Ok(self.diarize_batch(&[audio], sample_rate)?.remove(0))
    }

    /// Diarize several recordings, their steps batched together.
    pub fn diarize_batch(&mut self, audios: &[&[f32]], sample_rate: usize) -> Result<Vec<Diarization>> {
        let expected = self.model.config.sample_rate;
        snafu::ensure!(sample_rate == expected, SampleRateSnafu { got: sample_rate, expected });
        let mut sessions: Vec<Session> = audios
            .iter()
            .map(|audio| {
                let mut session = self.session();
                session.cursor.push(audio);
                session.finish();
                session
            })
            .collect();
        self.run(&mut sessions.iter_mut().collect::<Vec<_>>())?;
        let (num_speakers, frame_sec) = (self.num_speakers(), self.model.config.frame_sec());
        Ok(sessions.into_iter().map(|s| Diarization { probs: s.probs, num_speakers, frame_sec }).collect())
    }

    /// Run every step whose audio has arrived, batching sessions together.
    /// Returns the number of steps run.
    pub fn run(&mut self, sessions: &mut [&mut Session]) -> Result<usize> {
        let mut steps = 0;
        loop {
            let ready: Vec<(usize, StepPlan)> =
                sessions.iter().enumerate().filter_map(|(i, s)| self.plan(s).map(|plan| (i, plan))).collect();
            if ready.is_empty() {
                return Ok(steps);
            }
            for group in ready.chunks(self.max_batch) {
                let plans: Vec<StepPlan> = group.iter().map(|(_, plan)| *plan).collect();
                let mut batch: Vec<&mut Session> = sessions
                    .iter_mut()
                    .enumerate()
                    .filter(|(i, _)| group.iter().any(|(j, _)| j == i))
                    .map(|(_, session)| &mut **session)
                    .collect();
                self.step(&mut batch, &plans)?;
                steps += 1;
            }
        }
    }

    /// The next step of `session`, if its audio has arrived.
    fn plan(&self, session: &Session) -> Option<StepPlan> {
        let factor = self.model.config.subsampling_factor;
        let (chunk, step_frames) = (self.profile.chunk_len, self.step_frames);
        let start = session.next_frame;
        let available = session.cursor.available();
        let context = session.cache.context_frames();
        if !session.cursor.is_finished() {
            return ((start + step_frames) * factor <= available).then_some(StepPlan {
                context,
                chunk,
                input: step_frames,
                valid: step_frames,
                mel_valid: step_frames * factor,
            });
        }
        // Finished: `available` counts every mel frame, the last of which only
        // the `center` padding reaches and which the reference zeroes.
        let total = available.div_ceil(factor);
        if start >= total {
            return None;
        }
        let mel_valid = available.saturating_sub(1).saturating_sub(start * factor);
        let input = step_frames.min(total - start);
        Some(StepPlan {
            context,
            chunk: chunk.min(total - start),
            input,
            valid: mel_valid.div_ceil(factor).min(input),
            mel_valid: mel_valid.min(input * factor),
        })
    }

    fn step(&mut self, sessions: &mut [&mut Session], plans: &[StepPlan]) -> Result<()> {
        let config = &self.model.config;
        let (hidden, factor, speakers) = (config.hidden_size, config.subsampling_factor, config.num_speakers);
        let (capacity, step_frames, framed_len) = (self.capacity, self.step_frames, self.framed_len);
        let batch = sessions.len();
        let (batch_cap, seq_cap) = (self.max_batch, capacity);
        // Rows of `[previous | chunk | silence | zero]`.
        let (chunk_row, silence_row, zero_row) = (capacity, capacity + step_frames, capacity + step_frames + 1);

        // Host staging: audio rows, and every step-input row's source.
        self.framed.resize(batch * framed_len, 0.0);
        self.sources.clear();
        self.sources.resize(batch_cap * seq_cap, zero_row as i32);
        for (b, (session, plan)) in sessions.iter().zip(plans).enumerate() {
            session.cursor.stage(session.next_frame * factor, &mut self.framed[b * framed_len..(b + 1) * framed_len]);
            let row = &mut self.sources[b * seq_cap..(b + 1) * seq_cap];
            for (slot, source) in row.iter_mut().zip(session.cache.layout()) {
                *slot = match *source {
                    ContextRow::Step(i) => i as i32,
                    ContextRow::Silence => silence_row as i32,
                };
            }
            for (j, slot) in row[plan.context..plan.context + plan.input].iter_mut().enumerate() {
                *slot = (chunk_row + j) as i32;
            }
        }

        let jit = &mut self.jit;
        jit.framed_mut()?.copyin_at(0, bytemuck::cast_slice(&self.framed))?;
        jit.sources_mut()?.copyin(bytemuck::cast_slice(&self.sources))?;
        let row_bytes = capacity * hidden * size_of::<f32>();
        for (b, session) in sessions.iter().enumerate() {
            if let Some(previous) = &session.previous {
                jit.previous_mut()?.copy_region_from(b * row_bytes, previous, 0, session.previous_bytes)?;
            }
        }
        // Padding rows encode one zero frame, attending to itself; a live row
        // without an attended frame (a recording shorter than one hop) does the
        // same, so every attention path sees at least one key.
        let lens = |view: ndarray::ArrayViewMutD<'_, i32>, of: &dyn Fn(&StepPlan) -> usize, pad: i32| {
            let mut view = view;
            let slots = view.as_slice_mut().expect("contiguous lengths");
            slots.fill(pad);
            for (slot, plan) in slots.iter_mut().zip(plans) {
                *slot = of(plan) as i32;
            }
        };
        lens(jit.mel_valid_view_mut::<i32>()?, &|p| p.mel_valid, 0);
        lens(jit.seq_lens_view_mut::<i32>()?, &|p| p.context + p.input, 1);
        lens(jit.key_lens_view_mut::<i32>()?, &|p| (p.context + p.valid).max(1), 1);
        match &mut self.run_profile {
            Some(profile) => {
                let started = Instant::now();
                let kernels = jit.execute_with_vars_profiled(&[("b", batch as i64)])?;
                profile.push(StageProfile::gpu(format!("step b{batch} s{seq_cap}"), started.elapsed(), kernels));
            }
            None => jit.execute_bound(batch as i64)?,
        }

        let row_probs = seq_cap * factor * speakers;
        self.probs.resize(batch * row_probs, 0.0);
        jit.probs()?.copyout_prefix(bytemuck::cast_slice_mut(&mut self.probs))?;

        let input = jit.input()?;
        for (b, (session, plan)) in sessions.iter_mut().zip(plans).enumerate() {
            let row = &self.probs[b * row_probs..(b + 1) * row_probs];
            // A recording ends at its last mel frame, inside the last stack.
            let start = session.next_frame * factor;
            let emitted = match session.cursor.is_finished() {
                true => (plan.chunk * factor).min(session.cursor.available() - start),
                false => plan.chunk * factor,
            };
            let first = plan.context * factor;
            session.probs.extend_from_slice(&row[first * speakers..(first + emitted) * speakers]);

            // Speaker probabilities per encoder frame, zero past the attended frames.
            let attended = plan.context + plan.valid;
            let mut pooled = vec![0.0f32; (plan.context + plan.input) * speakers];
            for (frame, out) in pooled.chunks_mut(speakers).enumerate().take(attended) {
                let window = &row[frame * factor * speakers..(frame + 1) * factor * speakers];
                for (speaker, value) in out.iter_mut().enumerate() {
                    *value = window.iter().skip(speaker).step_by(speakers).sum::<f32>() / factor as f32;
                }
            }
            session.cache.update(plan.chunk, &pooled)?;

            // The next layout indexes this step's context and chunk rows.
            let kept = (plan.context + plan.chunk) * hidden * size_of::<f32>();
            let previous = match &mut session.previous {
                Some(previous) => previous,
                slot @ None => slot.insert(Buffer::allocate(
                    input.allocator_arc(),
                    DType::Float32,
                    vec![capacity * hidden],
                    BufferSpec { cpu_access: false, ..BufferSpec::default() },
                )?),
            };
            let offset = b * seq_cap * hidden * size_of::<f32>();
            previous.copy_region_from(0, input, offset, kept)?;
            session.previous_bytes = kept;

            session.next_frame += plan.chunk;
            session.cursor.discard(session.next_frame * factor);
        }
        Ok(())
    }
}
