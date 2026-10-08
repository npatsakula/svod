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
    #[snafu(display("failed to read config: {source}"))]
    ConfigIo { source: std::io::Error },
    #[snafu(display("{source}"))]
    Config { source: serde_json::Error },
    #[snafu(display("invalid decoder config: {message}"))]
    DecoderConfig { message: String },
    #[snafu(display("checkpoint/config mismatch: {message}"))]
    CheckpointConfig { message: String },
    #[snafu(display("invalid encoder dtype: {dtype:?}; expected f16, bf16, or f32"))]
    EncoderDtype { dtype: svod_dtype::DType },
    #[snafu(display("hub error: {source}"), context(false))]
    Hub { source: hf_hub::HFError },
    #[snafu(display("{source}"), context(false))]
    Ops {
        #[snafu(source(from(svod_tk3::ops::Error, Box::new)))]
        source: Box<svod_tk3::ops::Error>,
    },
}

pub type Result<T> = std::result::Result<T, Error>;
