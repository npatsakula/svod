//! GigaAM ASR model.
//!
//! One [`GigaAm`] struct wraps a shared Conformer encoder and a [`Head`] enum
//! that carries either a CTC projection (`Head::Ctc`) or an RN-T predictor +
//! joint pair (`Head::Rnnt`). The head variant is auto-detected from
//! `config.transducer.is_some()` at load time, so users construct the model
//! once and the head flows through the type system.
//!
//! Submodules:
//! - `model` — unified `GigaAm` + `Head` enum + `RnntRuntime` + all loaders.
//! - `jit` — shared `GigaAmEncoderJit` (encoder-only, fp32 output).
//! - `ctc` — `CTCHead` + `CtcHeadJit` (head-only; chains after the encoder JIT).
//! - `rnnt` — `RnntHead`, predictor / joint step JITs, `RnntStepBackend`.
//!
//! Shared infrastructure:
//! - `config` — `GigaAmConfig` JSON parsing.
//! - `encoder` — `Encoder` + Conformer building blocks (also owns the RoPE cache).
//! - `remap` — PyTorch state-dict key remapping.
//! - `error` — `Error` / `Result`.

mod config;
mod ctc;
mod encoder;
mod error;
mod jit;
mod model;
pub(crate) mod remap;
pub(crate) mod rnnt;
pub(crate) mod transcribe;

pub(crate) use config::subsampled_len;
pub use config::{ConvNormType, GigaAmConfig, SubsamplingMode, TransducerConfig};
pub use error::{Error, Result};
pub use jit::GigaAmEncoderJit;
pub use model::{GigaAm, Head, RnntRuntime};
pub use transcribe::{GigaAmTranscriber, TranscribeError, TranscribeOpts, Word};
