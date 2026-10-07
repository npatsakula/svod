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
    Device {
        #[snafu(source(from(svod_device::error::Error, Box::new)))]
        source: Box<svod_device::error::Error>,
    },
    #[snafu(display("{source}"), context(false))]
    State { source: crate::state::Error },
    #[snafu(display("model not found: {name}"))]
    ModelNotFound { name: String },
    #[snafu(display("download failed: {source}"))]
    Download { source: std::io::Error },
    #[snafu(display("{source}"), context(false))]
    Io { source: std::io::Error },
    #[snafu(display("checkpoint error: {msg}"))]
    Checkpoint { msg: String },
    #[snafu(display("tokenizer error: {msg}"))]
    Tokenizer { msg: String },
    #[snafu(display("decode error: {msg}"))]
    Decode { msg: String },
}

pub type Result<T> = std::result::Result<T, Error>;
