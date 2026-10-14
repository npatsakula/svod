//! Composable inference pipelines (host-side orchestration, GPU-free).
//!
//! [`audio`] is the long-form speech-to-text pipeline: VAD segmentation →
//! per-window transcription → core-crop → stitch, with the heavy machinery in
//! trait defaults so a model only implements its irreducible part.
//!
//! [`text`] is the encoder-head text pipeline: tokenize → chunk → model →
//! aggregate (embeddings / classification / token classification). It reuses
//! [`audio`]'s host-side, model-agnostic skeleton deliberately — a small
//! supertrait + trait defaults + host-side composer — adapted to text's
//! domain (sub-batching over byte offsets, span decoding). Future pipeline
//! families (e.g. speaker diarization) get sibling sub-modules here.

pub mod audio;
pub mod text;
