//! [`Yolo26Classify`] — YOLO v26 image classification.
//!
//! Backbone (no SPPF, layers 0–9) → Conv(1×1→1280) → GAP → Linear → softmax.
//! Forward returns `[B, nc]` class probabilities.

use svod_dtype::DType;
use svod_ir::SInt;
use svod_tensor::Tensor;
use svod_tensor::nn::{Layer, Linear, Module};

use crate::init::{Bias, linear};
use crate::state::StateDict;

use super::backbone::YoloBackboneCls;
use super::blocks::conv::YoloConv;
use super::config::YoloConfig;
use super::error::Result;

use super::loader;

const HIDDEN: usize = 1280;

/// Classification head: Conv(c4→1280, k1) → GAP → Linear(1280→nc).
///
/// State-dict keys: `conv.{conv,bn}.*`, `linear.weight`, `linear.bias`.
#[derive(Clone, Module)]
pub struct ClassifyHead {
    pub conv: YoloConv,
    pub linear: Linear,
}

impl ClassifyHead {
    pub fn empty(in_ch: usize, nc: usize) -> Self {
        Self {
            conv: YoloConv::empty(in_ch, HIDDEN, 1, 1, true),
            linear: linear(HIDDEN, nc, Bias::FanIn, DType::Float32),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = &super::head::in_head_dtype(x);
        let b = x.shape()?[0].clone();
        let x = self.conv.forward(x)?;
        // GAP: mean over H,W (axes 2,3)
        let x = x.mean_with().axes(vec![2isize, 3]).keepdim(true).call()?;
        // Flatten: [B, 1280, 1, 1] → [B, 1280]
        let x = x.try_reshape([b, SInt::from(HIDDEN)])?;
        // Linear → softmax
        Ok(self.linear.forward(&x)?.softmax(-1)?)
    }
}

/// YOLO v26 classification model.
///
/// Forward returns `[B, nc]` softmax probabilities.
#[derive(Clone, Module)]
pub struct Yolo26Classify {
    #[module(skip)]
    pub config: YoloConfig,
    #[module(key = "")]
    pub backbone: YoloBackboneCls,
    #[module(key = "10")]
    pub head: ClassifyHead,
}

impl Yolo26Classify {
    pub fn with_zero_weights(config: YoloConfig) -> Self {
        let scale = config.scale;
        let [_, _, _, _, c4] = super::backbone::scaled_channels(scale);
        let mut model = Self {
            config: config.clone(),
            backbone: YoloBackboneCls::empty(scale),
            head: ClassifyHead::empty(c4, config.nc),
        };
        loader::cast_placeholders(&mut model, &config.compute_dtype)
            .expect("a freshly built model round-trips its own state dict");
        model
    }

    pub fn from_hub(model_id: &str, config: YoloConfig) -> Result<Self> {
        Self::from_hub_with_revision(model_id, "main", config)
    }

    pub fn from_hub_with_revision(model_id: &str, revision: &str, config: YoloConfig) -> Result<Self> {
        let path = loader::download_safetensors(model_id, revision)?;
        Self::from_safetensors(&path, config)
    }

    pub fn from_safetensors(path: &std::path::Path, config: YoloConfig) -> Result<Self> {
        let sd = loader::prepare_state_dict(path)?;
        Self::from_state_dict(&sd, config)
    }

    pub fn from_state_dict(sd: &StateDict, config: YoloConfig) -> Result<Self> {
        let sd = loader::load_weights(sd, &config.compute_dtype)?;
        let mut model = Self::with_zero_weights(config);
        model.load_state_dict(&sd, "")?;
        Ok(model)
    }

    /// Run the full network. Returns `[B, nc]` softmax probabilities.
    pub fn forward(&self, images: &Tensor) -> Result<Tensor> {
        let images = &self.config.cast_input(images);
        let feat = self.backbone.forward(images)?;
        self.head.forward(&feat)
    }
}
