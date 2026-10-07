//! Nemotron-3-Diarization (`nvidia/Nemotron-3-Diarization`): streaming
//! Sortformer speaker diarization of up to 8 speakers, ordered by arrival.
//!
//! NeMo log-mel → stacks of 8 frames → 31-layer pre-LN RoPE Transformer →
//! sub-pixel upsampling back to 10 ms → per-speaker sigmoid. Audio is processed
//! in chunks, each attending to an Arrival-Order Speaker Cache and a FIFO of
//! past frames ([`svod_arch::diarization::SpeakerCache`]); offline recordings
//! and live streams differ only in their chunking [`Profile`].
//!
//! - `config` — [`NemotronDiarConfig`] from the HF `config.json` and
//!   `processor_config.json`, the [`Profile`]s.
//! - `model` — [`NemotronDiar`]: the front-end and the per-step classifier.
//! - `jit` — the step plan.
//! - `diarize` — [`Diarizer`] / [`Session`]: the chunked, batched driver.

mod config;
mod diarize;
mod error;
mod jit;
mod model;

pub use config::{CachePolicy, NemotronDiarConfig, Profile, StreamingMode};
pub use diarize::{Diarization, Diarizer, Session};
pub use error::{Error, Result};
pub use jit::NemotronDiarStepJit;
pub use model::{Attention, EncoderLayer, HUB_REPO, Mlp, NemotronDiar};
