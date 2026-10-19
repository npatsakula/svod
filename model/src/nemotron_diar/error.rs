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
    Ops {
        #[snafu(source(from(svod_tk3::ops::Error, Box::new)))]
        source: Box<svod_tk3::ops::Error>,
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
    #[snafu(display("state-dict op failed: {source}"), context(false))]
    State {
        #[snafu(source(from(crate::state::Error, Box::new)))]
        source: Box<crate::state::Error>,
    },
    #[snafu(display("HF Hub op failed: {source}"), context(false))]
    Hub { source: hf_hub::HFError },
    #[snafu(display("speaker cache: {source}"), context(false))]
    Cache { source: svod_arch::diarization::Error },
    #[snafu(display("config: {message}"))]
    Config { message: String },
    #[snafu(display("audio sample rate {got} Hz, the model expects {expected} Hz"))]
    SampleRate { got: usize, expected: usize },
    #[snafu(display("a flushed stream takes no more audio until it is reset"))]
    Flushed,
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
