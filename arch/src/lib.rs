//! Host-side building blocks for Svod: CTC and RNN-T decoders, VAD-driven
//! chunking, the long-form ASR pipeline ([`pipelines::audio`]) that chains a
//! splitter and a transcriber, the encoder text pipeline ([`pipelines::text`])
//! for embeddings, reranking and classification, and speaker-diarization state
//! and post-processing ([`diarization`]).

pub mod ctc;
pub mod diarization;
pub mod pipelines;
pub mod rnnt;
pub mod vad;

#[cfg(test)]
mod test;
