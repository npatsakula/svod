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
    #[snafu(display("state-dict op failed"), context(false))]
    State {
        #[snafu(source(from(crate::state::Error, Box::new)))]
        source: Box<crate::state::Error>,
    },
    #[snafu(display("HF Hub op failed"), context(false))]
    Hub { source: hf_hub::HFError },
    #[snafu(display("reading config.json failed: {message}"))]
    Config { message: String },
}
