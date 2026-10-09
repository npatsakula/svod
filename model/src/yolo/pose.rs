//! [`Yolo26Pose`] — YOLO v26 pose estimation.
//!
//! Detection head + keypoint prediction branches. Pose26 uses a shared
//! feature extractor (Conv→Conv) then separate keypoint Conv2d head.
//! At inference, sigma heads and RealNVP flow are not used.
//!
//! Forward returns `[B, 4+nc+nk, A]` — boxes + scores + decoded keypoints.

use svod_ir::SInt;
use svod_tensor::Tensor;
use svod_tensor::nn::{Conv2d, Layer, Module};

use crate::state::StateDict;

use super::backbone::YoloBackbone;
use super::blocks::conv::{YoloConv, conv2d_bias};
use super::config::DETECT_STRIDES;
use super::error::Result;

use super::head::{BoxBranch, ClsBranch, dist2bbox, make_anchors};
use super::loader;
use super::neck::YoloNeck;

/// Pose shared feature extractor: Conv(k3) → Conv(k3).
/// State-dict keys: `0.*`, `1.*`.
#[derive(Clone, Module)]
pub struct PoseFeatBranch {
    #[module(key = "0")]
    pub conv0: YoloConv,
    #[module(key = "1")]
    pub conv1: YoloConv,
}

impl PoseFeatBranch {
    pub fn empty(in_ch: usize, hidden: usize) -> Self {
        Self { conv0: YoloConv::empty(in_ch, hidden, 3, 1, true), conv1: YoloConv::empty(hidden, hidden, 3, 1, true) }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv0.forward(x)?;
        self.conv1.forward(&x)
    }
}

/// Pose26 head: Detect box+cls + shared pose features + keypoint prediction.
///
/// Forward returns `[B, 4+nc+nk, A]` — boxes + scores + decoded keypoints.
#[derive(Clone, Module)]
pub struct Pose26 {
    #[module(key = "one2one_cv2")]
    pub cv2: Vec<BoxBranch>,
    #[module(key = "one2one_cv3")]
    pub cv3: Vec<ClsBranch>,
    #[module(key = "one2one_cv4")]
    pub cv4: Vec<PoseFeatBranch>,
    #[module(key = "one2one_cv4_kpts")]
    pub cv4_kpts: Vec<Conv2d>,
    pub nc: usize,
    pub reg_max: usize,
    pub kpt_shape: (usize, usize),
    pub nk: usize,
}

impl Pose26 {
    pub fn empty(ch: &[usize], nc: usize, kpt_shape: (usize, usize), reg_max: usize) -> Self {
        let nk = kpt_shape.0 * kpt_shape.1;
        let hidden_box = (16usize).max(ch[0] / 4).max(reg_max * 4);
        let hidden_cls = ch[0].max(nc.min(100));
        // Pose26: c4 = max(ch[0]//4, num_kpts*(ndim+2))
        let c4 = (ch[0] / 4).max(kpt_shape.0 * (kpt_shape.1 + 2));
        Self {
            cv2: ch.iter().map(|&c| BoxBranch::empty(c, hidden_box, reg_max)).collect(),
            cv3: ch.iter().map(|&c| ClsBranch::empty(c, hidden_cls, nc)).collect(),
            cv4: ch.iter().map(|&c| PoseFeatBranch::empty(c, c4)).collect(),
            cv4_kpts: ch.iter().map(|_| conv2d_bias(c4, nk, 1, 1)).collect(),
            nc,
            reg_max,
            kpt_shape,
            nk,
        }
    }

    pub fn forward(&self, feats: &[Tensor]) -> Result<Tensor> {
        let feats = &super::head::in_head_dtypes(feats);
        let shape = feats[0].shape()?;
        let b = shape[0].clone();

        let mut boxes_list: Vec<Tensor> = Vec::with_capacity(feats.len());
        let mut scores_list: Vec<Tensor> = Vec::with_capacity(feats.len());
        let mut kpts_list: Vec<Tensor> = Vec::with_capacity(feats.len());
        let mut feat_sizes: Vec<(usize, usize)> = Vec::with_capacity(feats.len());

        for (i, feat) in feats.iter().enumerate() {
            let h = feat.dim_const(2)?;
            let w = feat.dim_const(3)?;
            feat_sizes.push((h, w));
            let hw = h * w;

            let box_out = self.cv2[i].forward(feat)?;
            boxes_list.push(box_out.try_reshape([b.clone(), SInt::from(4 * self.reg_max), SInt::from(hw)])?);

            let cls_out = self.cv3[i].forward(feat)?;
            scores_list.push(cls_out.try_reshape([b.clone(), SInt::from(self.nc), SInt::from(hw)])?);

            let features = self.cv4[i].forward(feat)?;
            let kpts = self.cv4_kpts[i].forward(&features)?;
            kpts_list.push(kpts.try_reshape([b.clone(), SInt::from(self.nk), SInt::from(hw)])?);
        }

        let boxes_refs: Vec<&Tensor> = boxes_list.iter().collect();
        let scores_refs: Vec<&Tensor> = scores_list.iter().collect();
        let kpts_refs: Vec<&Tensor> = kpts_list.iter().collect();

        let boxes = Tensor::cat(&boxes_refs, 2)?;
        let scores = Tensor::cat(&scores_refs, 2)?;
        let kpts = Tensor::cat(&kpts_refs, 2)?;

        let num_anchors: usize = feat_sizes.iter().map(|&(h, w)| h * w).sum();
        let (anchors, strides) = make_anchors(&feat_sizes, &DETECT_STRIDES)?;

        let dbox = dist2bbox(&boxes, &anchors, &strides, num_anchors)?;
        let scores = scores.sigmoid()?;

        // kpts_decode (Pose26 export path): y = kpts.view(bs, num_kpts, ndim, -1)
        // a = (y[:,:,:2] + anchors) * strides; vis = sigmoid(y[:,:,2:3])
        let decoded_kpts = kpts_decode(&kpts, &anchors, &strides, self.kpt_shape.0, self.kpt_shape.1, num_anchors)?;

        Ok(Tensor::cat(&[&dbox, &scores, &decoded_kpts], 1)?)
    }
}

/// Pose26 keypoint decode (export path):
/// `y = kpts.view(bs, num_kpts, ndim, A)`
/// `a = (y[:,:,:2] + anchors) * strides`
/// if ndim==3: `a = cat(a, y[:,:,2:3].sigmoid(), 2)`
/// `return a.view(bs, nk, A)`
fn kpts_decode(
    kpts: &Tensor,
    anchors: &Tensor,
    strides: &Tensor,
    num_kpts: usize,
    ndim: usize,
    num_anchors: usize,
) -> Result<Tensor> {
    let b = kpts.shape()?[0].clone();

    // [B, nk, A] → [B, num_kpts, ndim, A]
    let y = kpts.try_reshape([b.clone(), SInt::from(num_kpts), SInt::from(ndim), SInt::from(num_anchors)])?;

    // xy: [B, num_kpts, 2, A] = y[:,:,:2,:]
    let xy = y.slice_with().starts(&[0, 0, 0, 0]).ends(&[i64::MAX, i64::MAX, 2, i64::MAX]).call()?;

    // anchors [2,A] → [1,1,2,A], strides [1,A] → [1,1,1,A]
    let anchors_4d =
        anchors.try_reshape([SInt::from(1isize), SInt::from(1isize), SInt::from(2isize), SInt::from(num_anchors)])?;
    let strides_4d =
        strides.try_reshape([SInt::from(1isize), SInt::from(1isize), SInt::from(1isize), SInt::from(num_anchors)])?;

    let xy = xy.try_add(&anchors_4d)?;
    let xy = xy.try_mul(&strides_4d)?;

    if ndim == 3 {
        // vis: [B, num_kpts, 1, A] = y[:,:,2:3,:]
        let vis = y.slice_with().starts(&[0, 0, 2, 0]).ends(&[i64::MAX, i64::MAX, 3, i64::MAX]).call()?;
        let vis = vis.sigmoid()?;
        let cat = Tensor::cat(&[&xy, &vis], 2)?;
        Ok(cat.try_reshape([b, SInt::from(num_kpts * ndim), SInt::from(num_anchors)])?)
    } else {
        Ok(xy.try_reshape([b, SInt::from(num_kpts * ndim), SInt::from(num_anchors)])?)
    }
}

/// YOLO v26 pose estimation model.
///
/// Forward returns `[B, 4+nc+nk, A]` — boxes + scores + decoded keypoints.
#[derive(Clone, Module)]
pub struct Yolo26Pose {
    #[module(skip)]
    pub config: super::config::YoloConfig,
    #[module(key = "")]
    pub backbone: YoloBackbone,
    #[module(key = "")]
    pub neck: YoloNeck,
    #[module(key = "23")]
    pub head: Pose26,
}

impl Yolo26Pose {
    pub fn with_zero_weights(config: super::config::YoloConfig) -> Self {
        let scale = config.scale;
        let [_, _, c2, c3, c4] = super::backbone::scaled_channels(scale);
        let mut model = Self {
            config: config.clone(),
            backbone: YoloBackbone::empty(scale),
            neck: YoloNeck::empty(scale),
            head: Pose26::empty(&[c2, c3, c4], config.nc, (17, 3), config.reg_max),
        };
        loader::cast_placeholders(&mut model, &config.compute_dtype)
            .expect("a freshly built model round-trips its own state dict");
        model
    }

    pub fn from_hub(model_id: &str, config: super::config::YoloConfig) -> Result<Self> {
        Self::from_hub_with_revision(model_id, "main", config)
    }

    pub fn from_hub_with_revision(model_id: &str, revision: &str, config: super::config::YoloConfig) -> Result<Self> {
        let path = loader::download_safetensors(model_id, revision)?;
        Self::from_safetensors(&path, config)
    }

    pub fn from_safetensors(path: &std::path::Path, config: super::config::YoloConfig) -> Result<Self> {
        let sd = loader::prepare_state_dict(path)?;
        Self::from_state_dict(&sd, config)
    }

    pub fn from_state_dict(sd: &StateDict, config: super::config::YoloConfig) -> Result<Self> {
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
