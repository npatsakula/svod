//! [`Yolo26Depth`] — YOLO v26 monocular depth estimation.
//!
//! Full backbone+neck + multi-scale fusion decoder. Projects each pyramid
//! level to `c_mid` channels, progressively upsamples+fuses from P5→P3,
//! then runs Conv→ConvTranspose2d→Conv→Conv2d to produce `[B, 1, H/4, W/4]`.

use svod_tensor::Tensor;
use svod_tensor::nn::{Conv2d, ConvTranspose2d, CoordinateTransformMode, Layer, Module, ResizeMode};

use crate::state::StateDict;

use super::backbone::YoloBackbone;
use super::blocks::conv::{YoloConv, conv2d_bias, deconv2d_2x};
use super::config::YoloConfig;
use super::error::Result;

use super::loader;
use super::neck::YoloNeck;

/// Bilinear 2× upsample with align_corners=True, as torch's
/// `interpolate(..., mode="bilinear", align_corners=True)`.
fn resize_bilinear_2x(x: &Tensor) -> Result<Tensor> {
    Ok(x.upsample_with()
        .scale(&[2, 2])
        .mode(ResizeMode::Linear)
        .coordinate_transformation_mode(CoordinateTransformMode::AlignCorners)
        .call()?)
}

/// Depth fusion decoder head.
///
/// State-dict keys:
/// - `proj.{i}.*` — 1×1 conv per level
/// - `refine.{i}.0.*`, `refine.{i}.1.*` — refinement blocks
/// - `head.0.*`, `head.1.*`, `head.2.*`, `head.3.*`
/// - `cal_a`, `cal_b` — log-affine calibration buffers, absent from
///   checkpoints that were never calibrated
#[derive(Clone, Module)]
pub struct DepthHead {
    pub proj: Vec<YoloConv>,
    pub refine: Vec<(YoloConv, YoloConv)>,
    #[module(key = "head.0")]
    pub head_conv0: YoloConv,
    #[module(key = "head.1")]
    pub head_deconv: ConvTranspose2d,
    #[module(key = "head.2")]
    pub head_conv1: YoloConv,
    #[module(key = "head.3")]
    pub head_conv2: Conv2d,
    #[module(optional)]
    pub cal_a: Option<Tensor>,
    #[module(optional)]
    pub cal_b: Option<Tensor>,
}

impl DepthHead {
    pub fn empty(ch: &[usize], c_mid: usize) -> Self {
        let proj: Vec<YoloConv> = ch.iter().map(|&c| YoloConv::empty(c, c_mid, 1, 1, true)).collect();
        let nl = ch.len();
        let refine: Vec<(YoloConv, YoloConv)> = (0..nl - 1)
            .map(|_| (YoloConv::empty(c_mid, c_mid, 3, 1, true), YoloConv::empty(c_mid, c_mid, 3, 1, true)))
            .collect();
        Self {
            proj,
            refine,
            head_conv0: YoloConv::empty(c_mid, c_mid / 2, 3, 1, true),
            head_deconv: deconv2d_2x(c_mid / 2, c_mid / 2, 2),
            head_conv1: YoloConv::empty(c_mid / 2, c_mid / 4, 3, 1, true),
            head_conv2: conv2d_bias(c_mid / 4, 1, 1, 1),
            cal_a: Some(Tensor::from_slice([1.0f32])),
            cal_b: Some(Tensor::from_slice([0.0f32])),
        }
    }

    pub fn forward(&self, feats: &[Tensor]) -> Result<Tensor> {
        let feats = &super::head::in_head_dtypes(feats);
        let nl = feats.len();
        let projected: Vec<Tensor> = (0..nl).map(|i| self.proj[i].forward(&feats[i])).collect::<Result<_>>()?;

        let mut out = projected[nl - 1].clone();
        for i in (0..nl - 1).rev() {
            out = resize_bilinear_2x(&out)?;
            out = out.try_add(&projected[i])?;
            out = self.refine[i].0.forward(&out)?;
            out = self.refine[i].1.forward(&out)?;
        }

        let out = self.head_conv0.forward(&out)?;
        let out = self.head_deconv.forward(&out)?;
        let out = self.head_conv1.forward(&out)?;
        let out = self.head_conv2.forward(&out)?;

        // Log-affine calibration of exp(clamp(out, -4, 5)):
        // `depth ** cal_a * exp(cal_b)`, the identity when uncalibrated.
        let depth = out.clamp().min(-4.0).max(5.0).call()?.try_exp()?;
        let depth = match &self.cal_a {
            Some(a) => depth.try_pow(a)?,
            None => depth,
        };
        match &self.cal_b {
            Some(b) => Ok(depth.try_mul(&b.try_exp()?)?),
            None => Ok(depth),
        }
    }
}

/// YOLO v26 depth model. Forward returns `[B, 1, H/4, W/4]` depth map.
#[derive(Clone, Module)]
pub struct Yolo26Depth {
    #[module(skip)]
    pub config: YoloConfig,
    #[module(key = "")]
    pub backbone: YoloBackbone,
    #[module(key = "")]
    pub neck: YoloNeck,
    #[module(key = "23")]
    pub head: DepthHead,
}

impl Yolo26Depth {
    pub fn with_zero_weights(config: YoloConfig) -> Self {
        let scale = config.scale;
        let [_, _, c2, c3, c4] = super::backbone::scaled_channels(scale);
        let mut model = Self {
            config: config.clone(),
            backbone: YoloBackbone::empty(scale),
            neck: YoloNeck::empty(scale),
            head: DepthHead::empty(&[c2, c3, c4], 256),
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

    pub fn forward(&self, images: &Tensor) -> Result<Tensor> {
        let images = &self.config.cast_input(images);
        let (l4, l6, l10) = self.backbone.forward(images)?;
        let (p3, p4, p5) = self.neck.forward(&l4, &l6, &l10)?;
        self.head.forward(&[p3, p4, p5])
    }
}
