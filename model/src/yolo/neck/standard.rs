//! YOLO v26 FPN+PAN neck (layers 11–22).
//!
//! Takes backbone skip features `(l4, l6, l10)` and produces three detection
//! feature maps `(P3, P4, P5)` at strides 8, 16, 32.

use svod_tensor::Tensor;
use svod_tensor::nn::{Module, ResizeMode};

use crate::state::scoped;
use crate::yolo::backbone::scaled_channels;
use crate::yolo::blocks::conv::YoloConv;
use crate::yolo::blocks::csp::C3k2;
use crate::yolo::config::{YoloScale, make_depth};
use crate::yolo::error::Result;

/// Full FPN+PAN neck (layers 11–22). Shared by detect, segment, obb, pose,
/// and depth models.
#[derive(Clone, Module)]
pub struct YoloNeck {
    #[module(key = "13")]
    pub c3k2_13: C3k2,
    #[module(key = "16")]
    pub c3k2_16: C3k2,
    #[module(key = "17")]
    pub conv17: YoloConv,
    #[module(key = "19")]
    pub c3k2_19: C3k2,
    #[module(key = "20")]
    pub conv20: YoloConv,
    #[module(key = "22")]
    pub c3k2_22: C3k2,
}

impl YoloNeck {
    pub fn empty(scale: YoloScale) -> Self {
        let d = |yaml_n| make_depth(yaml_n, scale);
        let [_, _, c2, c3, c4] = scaled_channels(scale);
        Self {
            // Layer 13: C3k2 [512, True] — input is cat(upsample(c4), c3) = c4+c3
            c3k2_13: C3k2::empty(c4 + c3, c3, d(2), true, 0.5, true, false),
            // Layer 16: C3k2 [256, True] — input is cat(upsample(c3), c3) = c3+c3
            c3k2_16: C3k2::empty(c3 + c3, c2, d(2), true, 0.5, true, false),
            // Layer 17: Conv [256, 3, 2]
            conv17: YoloConv::empty(c2, c2, 3, 2, true).tk(),
            // Layer 19: C3k2 [512, True] — input is cat(conv17_out, c3k2_13_out)
            c3k2_19: C3k2::empty(c2 + c3, c3, d(2), true, 0.5, true, false),
            // Layer 20: Conv [512, 3, 2]
            conv20: YoloConv::empty(c3, c3, 3, 2, true).tk(),
            // Layer 22: C3k2 [1024, True, 0.5, True] — attn=True
            c3k2_22: C3k2::empty(c3 + c4, c4, d(1), true, 0.5, true, true),
        }
    }

    /// Run the full FPN+PAN. Returns `(P3, P4, P5)` detection features.
    pub fn forward(&self, l4: &Tensor, l6: &Tensor, l10: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        // FPN top-down
        let up = l10.upsample(&[2, 2], ResizeMode::Nearest)?;
        let cat = Tensor::cat(&[&up, l6], 1)?;
        let l13 = scoped("13", || self.c3k2_13.forward(&cat))?;

        let up = l13.upsample(&[2, 2], ResizeMode::Nearest)?;
        let cat = Tensor::cat(&[&up, l4], 1)?;
        let l16 = scoped("16", || self.c3k2_16.forward(&cat))?;

        // PAN bottom-up. The downsampling convs feed only a `cat`, and fused
        // into it each would run over the whole concatenated channel range;
        // realized, each covers its own channels and the `cat` is a gather.
        let l17 = scoped("17", || self.conv17.forward(&l16))?.contiguous();
        let cat = Tensor::cat(&[&l17, &l13], 1)?;
        let l19 = scoped("19", || self.c3k2_19.forward(&cat))?;

        let l20 = scoped("20", || self.conv20.forward(&l19))?.contiguous();
        let cat = Tensor::cat(&[&l20, l10], 1)?;
        let l22 = scoped("22", || self.c3k2_22.forward(&cat))?;

        Ok((l16, l19, l22))
    }
}
