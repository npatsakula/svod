//! Audio preprocessing for ASR: mel features (`mel`) and encoder sizing
//! `bounds`.
//!
//! `bounds` operates on raw `&[f32]` PCM; `mel` runs in the graph, the
//! host only staging windows. Chunking itself lives in the arch
//! [`pipelines`](svod_arch::pipelines::audio) layer; [`EncoderBounds`] is the
//! model-config bridge a splitter consumes to derive its chunker config.

pub(crate) mod bounds;
pub(crate) mod mel;

pub(crate) use bounds::ChunkerKnobs;
pub use bounds::{AudioChunk, EncoderBounds};
pub use mel::{MelConfig, MelJit, MelScale, MelSpectrogram};
