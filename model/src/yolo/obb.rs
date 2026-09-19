//! [`Yolo26Obb`] — YOLO v26 oriented bounding box detection.
//!
//! Detection head + angle prediction branch per scale. Boxes decoded as
//! rotated boxes via `dist2rbox`. Returns `[B, 4+nc+1, A]`.

use svod_ir::SInt;
use svod_tensor::Tensor;
use svod_tensor::nn::{Conv2d, Layer, Module};

use crate::state::StateDict;

use super::backbone::YoloBackbone;
use super::blocks::conv::{YoloConv, conv2d_bias};
use super::config::DETECT_STRIDES;
use super::error::Result;

use super::head::{BoxBranch, ClsBranch, make_anchors};
use super::loader;
use super::neck::YoloNeck;

/// Angle branch: Conv(k3) → Conv(k3) → Conv2d(k1, bias). Outputs 1 channel.
///
/// State-dict keys: `0.*`, `1.*`, `2.weight`, `2.bias`.
#[derive(Clone, Module)]
pub struct AngleBranch {
    #[module(key = "0")]
    pub conv0: YoloConv,
    #[module(key = "1")]
    pub conv1: YoloConv,
    #[module(key = "2")]
    pub conv2: Conv2d,
}

impl AngleBranch {
    pub fn empty(in_ch: usize, ne: usize) -> Self {
        let hidden = (in_ch / 4).max(ne);
        Self {
            conv0: YoloConv::empty(in_ch, hidden, 3, 1, true),
            conv1: YoloConv::empty(hidden, hidden, 3, 1, true),
            conv2: conv2d_bias(hidden, ne, 1, 1),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv0.forward(x)?;
        let x = self.conv1.forward(&x)?;
        Ok(self.conv2.forward(&x)?)
    }
}

/// YOLO v26 OBB detection head (end2end, reg_max=1).
///
/// Reuses Detect's box+cls branches and adds an angle branch.
/// Forward returns `[B, 4+nc+ne, A]` — rotated boxes + scores + angles.
#[derive(Clone, Module)]
pub struct OBB26 {
    #[module(key = "one2one_cv2")]
    pub cv2: Vec<BoxBranch>,
    #[module(key = "one2one_cv3")]
    pub cv3: Vec<ClsBranch>,
    #[module(key = "one2one_cv4")]
    pub cv4: Vec<AngleBranch>,
    pub nc: usize,
    pub reg_max: usize,
    pub ne: usize,
}

impl OBB26 {
    pub fn empty(ch: &[usize], nc: usize, reg_max: usize, ne: usize) -> Self {
        let hidden_box = (16usize).max(ch[0] / 4).max(reg_max * 4);
        let hidden_cls = ch[0].max(nc.min(100));
        Self {
            cv2: ch.iter().map(|&c| BoxBranch::empty(c, hidden_box, reg_max)).collect(),
            cv3: ch.iter().map(|&c| ClsBranch::empty(c, hidden_cls, nc)).collect(),
            cv4: ch.iter().map(|&c| AngleBranch::empty(c, ne)).collect(),
            nc,
            reg_max,
            ne,
        }
    }

    pub fn forward(&self, feats: &[Tensor]) -> Result<Tensor> {
        let feats = &super::head::in_head_dtypes(feats);
        let shape = feats[0].shape()?;
        let b = shape[0].clone();

        let mut boxes_list: Vec<Tensor> = Vec::with_capacity(feats.len());
        let mut scores_list: Vec<Tensor> = Vec::with_capacity(feats.len());
        let mut angle_list: Vec<Tensor> = Vec::with_capacity(feats.len());
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

            let angle_out = self.cv4[i].forward(feat)?;
            angle_list.push(angle_out.try_reshape([b.clone(), SInt::from(self.ne), SInt::from(hw)])?);
        }

        let boxes_refs: Vec<&Tensor> = boxes_list.iter().collect();
        let scores_refs: Vec<&Tensor> = scores_list.iter().collect();
        let angle_refs: Vec<&Tensor> = angle_list.iter().collect();

        let boxes = Tensor::cat(&boxes_refs, 2)?;
        let scores = Tensor::cat(&scores_refs, 2)?;
        let angles = Tensor::cat(&angle_refs, 2)?;

        let num_anchors: usize = feat_sizes.iter().map(|&(h, w)| h * w).sum();
        let (anchors, strides) = make_anchors(&feat_sizes, &DETECT_STRIDES)?;

        // dist2rbox decode (consumes angles for box rotation)
        let dbox = dist2rbox(&boxes, &angles, &anchors, &strides, num_anchors)?;
        let scores = scores.sigmoid()?;

        // Cat: [B, 4+nc+ne, A] = dbox + scores + raw_angles
        Ok(Tensor::cat(&[&dbox, &scores, &angles], 1)?)
    }
}

/// Decode rotated boxes from distance + angle predictions.
/// `boxes [B,4,A]`, `angles [B,1,A]`, `anchors [2,A]`, `strides [1,A]` → `[B,4,A]` (xywh).
fn dist2rbox(
    boxes: &Tensor,
    angles: &Tensor,
    anchors: &Tensor,
    strides: &Tensor,
    num_anchors: usize,
) -> Result<Tensor> {
    let parts = boxes.split(&[2, 2], 1)?;
    let lt = &parts[0];
    let rb = &parts[1];

    let cos = angles.cos()?;
    let sin = angles.sin()?;

    // xf, yf = (rb - lt) / 2
    let diff = rb.try_sub(lt)?;
    let diff_halves = diff.split(&[1, 1], 1)?;
    let xf = diff_halves[0].try_mul(0.5)?;
    let yf = diff_halves[1].try_mul(0.5)?;

    // x = xf*cos - yf*sin, y = xf*sin + yf*cos
    let x = xf.try_mul(&cos)?.try_sub(&yf.try_mul(&sin)?)?;
    let y = xf.try_mul(&sin)?.try_add(&yf.try_mul(&cos)?)?;
    let xy = Tensor::cat(&[&x, &y], 1)?;

    // xy += anchors (broadcast [2,A] → [1,2,A])
    let anchors_3d = anchors.try_reshape([SInt::from(1isize), SInt::from(2isize), SInt::from(num_anchors as isize)])?;
    let xy = xy.try_add(&anchors_3d)?;

    // wh = lt + rb
    let wh = lt.try_add(rb)?;
    let bbox = Tensor::cat(&[&xy, &wh], 1)?;

    let strides_3d = strides.try_reshape([SInt::from(1isize), SInt::from(1isize), SInt::from(num_anchors as isize)])?;
    Ok(bbox.try_mul(&strides_3d)?)
}

/// YOLO v26 OBB model.
///
/// Forward returns `[B, 4+nc+1, A]` — rotated boxes + scores + angle.
#[derive(Clone, Module)]
pub struct Yolo26Obb {
    #[module(skip)]
    pub config: super::config::YoloConfig,
    #[module(key = "")]
    pub backbone: YoloBackbone,
    #[module(key = "")]
    pub neck: YoloNeck,
    #[module(key = "23")]
    pub head: OBB26,
}

impl Yolo26Obb {
    pub fn with_zero_weights(config: super::config::YoloConfig) -> Self {
        let scale = config.scale;
        let [_, _, c2, c3, c4] = super::backbone::scaled_channels(scale);
        let mut model = Self {
            config: config.clone(),
            backbone: YoloBackbone::empty(scale),
            neck: YoloNeck::empty(scale),
            head: OBB26::empty(&[c2, c3, c4], config.nc, config.reg_max, 1),
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
