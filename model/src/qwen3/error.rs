//! Error type for the Qwen3 module.

use snafu::Snafu;

#[derive(Debug, Snafu)]
#[snafu(visibility(pub(crate)))]
pub enum Error {
    #[snafu(display("{source}"), context(false))]
    Tensor {
        #[snafu(source(from(svod_tensor::error::Error, Box::new)))]
        source: Box<svod_tensor::error::Error>,
    },

    #[snafu(display("{source}"), context(false))]
    Jit {
        #[snafu(source(from(crate::jit::JitError, Box::new)))]
        source: Box<crate::jit::JitError>,
    },

    #[snafu(display("{source}"), context(false))]
    Ops {
        #[snafu(source(from(svod_tk3::ops::Error, Box::new)))]
        source: Box<svod_tk3::ops::Error>,
    },

    #[snafu(display("hand kernel: {source}"))]
    Tk {
        #[snafu(source(from(svod_tk::LaunchError, Box::new)))]
        source: Box<svod_tk::LaunchError>,
    },

    #[snafu(display("state-dict op failed"), context(false))]
    State {
        #[snafu(source(from(crate::state::Error, Box::new)))]
        source: Box<crate::state::Error>,
    },

    #[snafu(display("HF Hub op failed"), context(false))]
    Hub { source: hf_hub::HFError },

    #[snafu(display("reading config failed: {message}"))]
    Config { message: String },

    #[snafu(display("a sequence of {seq_len} tokens exceeds the model's {max_position_embeddings}-position context"))]
    ContextLength { seq_len: usize, max_position_embeddings: usize },
}

pub type Result<T> = std::result::Result<T, Error>;
