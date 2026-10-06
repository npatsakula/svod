//! Host-side speech building blocks for Svod: CTC and RNN-T decoders, VAD-driven
//! chunking, and the long-form ASR pipeline ([`pipelines::audio`]) that chains a
//! splitter and a transcriber.

pub mod ctc;
pub mod pipelines;
pub mod rnnt;
pub mod vad;

#[cfg(test)]
mod test;
