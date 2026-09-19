//! [`Yolo26Detect`], [`Yolo26DetectP2`], [`Yolo26DetectP6`] — YOLO v26
//! object detection models (standard, P2, P6 topologies).
//!
//! All three share the same [`Detect`] head (differing only in the number of
//! detection scales) and differ in backbone/neck depth.

use svod_ir::SInt;
use svod_tensor::Tensor;
use svod_tensor::nn::Module;

use crate::state::{StateDict, scoped_index};

use super::backbone::{YoloBackbone, scaled_channels};
use super::config::{P2_STRIDES, P6_STRIDES, YoloConfig};
use super::error::Result;

use super::head::{BoxBranch, ClsBranch, dist2bbox, make_anchors};
use super::loader;

// ---------------------------------------------------------------------------
// Detect head
// ---------------------------------------------------------------------------

/// YOLO v26 detection head (end2end, reg_max=1). Loads only the one2one
/// branch weights (the one2many branch is training-only).
///
/// Forward returns a single tensor `[B, 4+nc, A]` — decoded boxes (xyxy in
/// pixel space) concatenated with sigmoid'd class scores.
#[derive(Clone, Module)]
pub struct Detect {
    #[module(key = "one2one_cv2")]
    pub cv2: Vec<BoxBranch>,
    #[module(key = "one2one_cv3")]
    pub cv3: Vec<ClsBranch>,
    pub nc: usize,
    pub reg_max: usize,
    pub strides: Vec<usize>,
}

impl Detect {
    pub fn empty(ch: &[usize], nc: usize, reg_max: usize) -> Self {
        Self::empty_with_strides(ch, nc, reg_max, &super::config::DETECT_STRIDES)
    }

    pub fn empty_with_strides(ch: &[usize], nc: usize, reg_max: usize, strides: &[usize]) -> Self {
        let hidden_box = (16usize).max(ch[0] / 4).max(reg_max * 4);
        let hidden_cls = ch[0].max(nc.min(100));
        Self {
            cv2: ch.iter().map(|&c| BoxBranch::empty(c, hidden_box, reg_max)).collect(),
            cv3: ch.iter().map(|&c| ClsBranch::empty(c, hidden_cls, nc)).collect(),
            nc,
            reg_max,
            strides: strides.to_vec(),
        }
    }

    /// Run box + cls heads on each feature map, decode boxes via dist2bbox,
    /// sigmoid scores, and cat into `[B, 4+nc, A]`. The branches run at the
    /// features' dtype and emit [`super::head::HEAD_DTYPE`], so the decode is
    /// full-width whatever the backbone computed in.
    pub fn forward(&self, feats: &[Tensor]) -> Result<Tensor> {
        let shape = feats[0].shape()?;
        let b = shape[0].clone();

        let mut boxes_list: Vec<Tensor> = Vec::with_capacity(feats.len());
        let mut scores_list: Vec<Tensor> = Vec::with_capacity(feats.len());
        let mut feat_sizes: Vec<(usize, usize)> = Vec::with_capacity(feats.len());

        for (i, feat) in feats.iter().enumerate() {
            let h = feat.dim_const(2)?;
            let w = feat.dim_const(3)?;
            let hw = h * w;
            feat_sizes.push((h, w));

            // Each branch ends in its own kernel: fused into the `cat` below, a
            // final conv would be recomputed over every scale's anchors.
            let box_out = scoped_index("one2one_cv2", i, || self.cv2[i].forward(feat))?.contiguous();
            let box_out = box_out.try_reshape([b.clone(), SInt::from(4 * self.reg_max), SInt::from(hw)])?;
            boxes_list.push(box_out);

            let cls_out = scoped_index("one2one_cv3", i, || self.cv3[i].forward(feat))?.contiguous();
            let cls_out = cls_out.try_reshape([b.clone(), SInt::from(self.nc), SInt::from(hw)])?;
            scores_list.push(cls_out);
        }

        let boxes_refs: Vec<&Tensor> = boxes_list.iter().collect();
        let scores_refs: Vec<&Tensor> = scores_list.iter().collect();

        let boxes = Tensor::cat(&boxes_refs, 2)?;
        let scores = Tensor::cat(&scores_refs, 2)?;

        let num_anchors: usize = feat_sizes.iter().map(|&(h, w)| h * w).sum();
        let (anchors, strides) = make_anchors(&feat_sizes, &self.strides)?;

        let dbox = dist2bbox(&boxes, &anchors, &strides, num_anchors)?;
        let scores = scores.sigmoid()?;

        Ok(Tensor::cat(&[&dbox, &scores], 1)?)
    }
}

// ---------------------------------------------------------------------------
// Yolo26Detect — standard P3/P4/P5
// ---------------------------------------------------------------------------

/// YOLO v26 object detection model (P3/8–P5/32).
///
/// Forward returns `[B, 4+nc, A]` (decoded boxes + scores).
#[derive(Clone, Module)]
pub struct Yolo26Detect {
    #[module(skip)]
    pub config: YoloConfig,
    #[module(key = "")]
    pub backbone: YoloBackbone,
    #[module(key = "")]
    pub neck: super::neck::YoloNeck,
    #[module(key = "23")]
    pub head: Detect,
}

impl Yolo26Detect {
    pub fn with_zero_weights(config: YoloConfig) -> Self {
        let scale = config.scale;
        let [_, _, c2, c3, c4] = scaled_channels(scale);
        let mut model = Self {
            config: config.clone(),
            backbone: YoloBackbone::empty(scale),
            neck: super::neck::YoloNeck::empty(scale),
            head: Detect::empty(&[c2, c3, c4], config.nc, config.reg_max),
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
        let (l4, l6, l10) = crate::state::scoped("backbone", || self.backbone.forward(images))?;
        let (p3, p4, p5) = crate::state::scoped("neck", || self.neck.forward(&l4, &l6, &l10))?;
        crate::state::scoped("head", || self.head.forward(&[p3, p4, p5]))
    }
}

// ---------------------------------------------------------------------------
// Yolo26DetectP2 — P2/P3/P4/P5
// ---------------------------------------------------------------------------

/// YOLO v26-P2 object detection model (P2/4–P5/32).
///
/// Uses the standard backbone (tapping P2/4) with an extended P2 neck,
/// producing four detection feature maps at strides 4, 8, 16, 32.
#[derive(Clone, Module)]
pub struct Yolo26DetectP2 {
    #[module(skip)]
    pub config: YoloConfig,
    #[module(key = "")]
    pub backbone: YoloBackbone,
    #[module(key = "")]
    pub neck: super::neck::YoloNeckP2,
    #[module(key = "29")]
    pub head: Detect,
}

impl Yolo26DetectP2 {
    pub fn with_zero_weights(config: YoloConfig) -> Self {
        let scale = config.scale;
        let sc = |yaml_c| crate::yolo::config::scale_channels(yaml_c, scale);
        let ch = [sc(128), sc(256), sc(512), sc(1024)];
        let mut model = Self {
            config: config.clone(),
            backbone: YoloBackbone::empty(scale),
            neck: super::neck::YoloNeckP2::empty(scale),
            head: Detect::empty_with_strides(&ch, config.nc, config.reg_max, &P2_STRIDES),
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
        let (l2, l4, l6, l10) = self.backbone.forward_with_p2(images)?;
        let (p2, p3, p4, p5) = self.neck.forward(&l2, &l4, &l6, &l10)?;
        self.head.forward(&[p2, p3, p4, p5])
    }
}

// ---------------------------------------------------------------------------
// Yolo26DetectP6 — P3/P4/P5/P6
// ---------------------------------------------------------------------------

/// YOLO v26-P6 object detection model (P3/8–P6/64).
///
/// Uses the P6 backbone (deeper, P5=768ch, P6=1024ch) with a P6 neck,
/// producing four detection feature maps at strides 8, 16, 32, 64.
#[derive(Clone, Module)]
pub struct Yolo26DetectP6 {
    #[module(skip)]
    pub config: YoloConfig,
    #[module(key = "")]
    pub backbone: super::backbone::YoloBackboneP6,
    #[module(key = "")]
    pub neck: super::neck::YoloNeckP6,
    #[module(key = "31")]
    pub head: Detect,
}

impl Yolo26DetectP6 {
    pub fn with_zero_weights(config: YoloConfig) -> Self {
        let scale = config.scale;
        let [_, _, c2, c3, c5, c6] = super::backbone::p6_scaled_channels(scale);
        let mut model = Self {
            config: config.clone(),
            backbone: super::backbone::YoloBackboneP6::empty(scale),
            neck: super::neck::YoloNeckP6::empty(scale),
            head: Detect::empty_with_strides(&[c2, c3, c5, c6], config.nc, config.reg_max, &P6_STRIDES),
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
        let (l4, l6, l8, l12) = self.backbone.forward(images)?;
        let (p3, p4, p5, p6) = self.neck.forward(&l4, &l6, &l8, &l12)?;
        self.head.forward(&[p3, p4, p5, p6])
    }
}
