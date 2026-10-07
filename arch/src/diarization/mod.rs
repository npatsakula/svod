//! Host-side speaker diarization building blocks, shared by every frame-level
//! diarizer.
//!
//! - [`SpeakerCache`]: the Arrival-Order Speaker Cache and FIFO queue of
//!   streaming Sortformer models — which encoder frames each new chunk attends
//!   to, compressed per speaker when the cache overflows.
//! - [`Binarization`] / [`speaker_segments`]: per-frame speaker probabilities to
//!   [`SpeakerSegment`]s (hysteresis thresholds, padding, minimum durations),
//!   and [`write_rttm`].
//!
//! Both are model-agnostic: they read plain `&[f32]` probability rows.

mod cache;
mod segments;

pub use cache::{ContextRow, SpeakerCache, SpeakerCacheConfig};
pub use segments::{Binarization, SpeakerSegment, speaker_segments, write_rttm};

use snafu::Snafu;

#[derive(Debug, Snafu)]
#[snafu(visibility(pub(crate)))]
pub enum Error {
    #[snafu(display("speaker cache: {reason}"))]
    InvalidCacheConfig { reason: String },

    #[snafu(display("speaker cache update: {reason}"))]
    InvalidUpdate { reason: String },
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
