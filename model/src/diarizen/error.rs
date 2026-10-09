use snafu::Snafu;

pub type Result<T> = std::result::Result<T, Error>;

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
    #[snafu(display("HF Hub op failed"), context(false))]
    Hub { source: hf_hub::HFError },
    #[snafu(display("pickle loader failed"))]
    Pickle {
        #[snafu(source(from(crate::wespeaker::pickle::Error, Box::new)))]
        source: Box<crate::wespeaker::pickle::Error>,
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
    #[snafu(display("waveform is {wav_sr} Hz, model expects {model_sr} Hz (resample first)"))]
    SampleRateMismatch { wav_sr: u32, model_sr: u32 },
}

impl From<crate::wavlm::Error> for Error {
    fn from(e: crate::wavlm::Error) -> Self {
        use crate::wavlm::Error as W;
        match e {
            W::Tensor { source } => Error::Tensor { source },
            W::Ops { source } => Error::Ops { source },
            W::Pickle { source } => Error::Pickle { source },
            W::Hub { source } => Error::Hub { source },
        }
    }
}
