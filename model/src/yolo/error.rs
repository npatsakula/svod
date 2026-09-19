use snafu::Snafu;

#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum Error {
    #[snafu(display("{source}"), context(false))]
    Tensor {
        #[snafu(source(from(svod_tensor::error::Error, Box::new)))]
        source: Box<svod_tensor::error::Error>,
    },
    #[snafu(display("{source}"), context(false))]
    State {
        #[snafu(source(from(crate::state::Error, Box::new)))]
        source: Box<crate::state::Error>,
    },
    #[snafu(display("hub error: {source}"), context(false))]
    Hub { source: hf_hub::HFError },
    #[snafu(display("invalid yolo config: {message}"))]
    Config { message: String },
    #[snafu(display("hand kernel: {source}"), context(false))]
    Tk {
        #[snafu(source(from(svod_tk::LaunchError, Box::new)))]
        source: Box<svod_tk::LaunchError>,
    },
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<crate::blocks::Error> for Error {
    fn from(e: crate::blocks::Error) -> Self {
        match e {
            crate::blocks::Error::Tensor { source } => Error::Tensor { source },
        }
    }
}
