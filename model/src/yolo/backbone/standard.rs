//! Shared YOLO v26 backbone (layers 0–10) and classification backbone
//! (layers 0–9, no SPPF).
//!
//! The backbone produces three skip-connection feature maps (P3/8, P4/16,
//! P5/32) consumed by detection-family and depth neck/head. The classify
//! variant stops at the C2PSA layer (layer 9) and omits SPPF.

use svod_tensor::Tensor;
use svod_tensor::nn::Module;

use crate::state::scoped;
use crate::yolo::blocks::attention::C2PSA;
use crate::yolo::blocks::conv::YoloConv;
use crate::yolo::blocks::csp::C3k2;
use crate::yolo::blocks::sppf::Sppf;
use crate::yolo::config::{YoloScale, make_depth, scale_channels};
use crate::yolo::error::Result;

/// Backbone channels after scaling (used to construct neck/head).
pub fn scaled_channels(scale: YoloScale) -> [usize; 5] {
    let sc = |yaml_c| scale_channels(yaml_c, scale);
    [sc(64), sc(128), sc(256), sc(512), sc(1024)]
}

/// Full YOLO v26 backbone (layers 0–10): Conv→Conv→C3k2→Conv→C3k2→Conv→
/// C3k2→Conv→C3k2→SPPF→C2PSA.
///
/// The stride-2 downsamples run the tk convolution ([`YoloConv::tk`]); layer 0
/// does not, its three input channels being too few to fill a K strip.
///
/// Forward returns the three skip-connection outputs: `(l4, l6, l10)`.
#[derive(Clone, Module)]
pub struct YoloBackbone {
    #[module(key = "0")]
    pub conv0: YoloConv,
    #[module(key = "1")]
    pub conv1: YoloConv,
    #[module(key = "2")]
    pub c3k2_2: C3k2,
    #[module(key = "3")]
    pub conv3: YoloConv,
    #[module(key = "4")]
    pub c3k2_4: C3k2,
    #[module(key = "5")]
    pub conv5: YoloConv,
    #[module(key = "6")]
    pub c3k2_6: C3k2,
    #[module(key = "7")]
    pub conv7: YoloConv,
    #[module(key = "8")]
    pub c3k2_8: C3k2,
    #[module(key = "9")]
    pub sppf9: Sppf,
    #[module(key = "10")]
    pub c2psa10: C2PSA,
}

impl YoloBackbone {
    pub fn empty(scale: YoloScale) -> Self {
        let d = |yaml_n| make_depth(yaml_n, scale);
        let [c0, c1, c2, c3, c4] = scaled_channels(scale);
        Self {
            conv0: YoloConv::empty(3, c0, 3, 2, true),
            conv1: YoloConv::empty(c0, c1, 3, 2, true).tk(),
            c3k2_2: C3k2::empty(c1, c2, d(2), true, 0.25, scale.forces_c3k(), false),
            conv3: YoloConv::empty(c2, c2, 3, 2, true).tk(),
            c3k2_4: C3k2::empty(c2, c3, d(2), true, 0.25, scale.forces_c3k(), false),
            conv5: YoloConv::empty(c3, c3, 3, 2, true).tk(),
            c3k2_6: C3k2::empty(c3, c3, d(2), true, 0.5, true, false),
            conv7: YoloConv::empty(c3, c4, 3, 2, true).tk(),
            c3k2_8: C3k2::empty(c4, c4, d(2), true, 0.5, true, false),
            sppf9: Sppf::empty(c4, c4, 5, 3, true),
            c2psa10: C2PSA::empty(c4, c4, d(2), 0.5),
        }
    }

    /// Run backbone layers 0–10, returning skip features `(l4, l6, l10)`.
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (_, l4, l6, l10) = self.forward_with_p2(x)?;
        Ok((l4, l6, l10))
    }

    /// Run backbone layers 0–10, returning four skip features `(l2, l4, l6, l10)`.
    /// Used by the P2 variant neck which taps the P2/4 feature at layer 2.
    pub fn forward_with_p2(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let x = scoped("0", || self.conv0.forward(x))?;
        let x = scoped("1", || self.conv1.forward(&x))?;
        let l2 = scoped("2", || self.c3k2_2.forward(&x))?;
        let x = scoped("3", || self.conv3.forward(&l2))?;
        let l4 = scoped("4", || self.c3k2_4.forward(&x))?;
        let x = scoped("5", || self.conv5.forward(&l4))?;
        let l6 = scoped("6", || self.c3k2_6.forward(&x))?;
        let x = scoped("7", || self.conv7.forward(&l6))?;
        let x = scoped("8", || self.c3k2_8.forward(&x))?;
        let x = scoped("9", || self.sppf9.forward(&x))?;
        let l10 = scoped("10", || self.c2psa10.forward(&x))?;
        Ok((l2, l4, l6, l10))
    }
}

/// Classification backbone (layers 0–9): same as [`YoloBackbone`] but without
/// SPPF (layer 9 is C2PSA, not SPPF). The classify YAML puts C2PSA at index 9.
#[derive(Clone, Module)]
pub struct YoloBackboneCls {
    #[module(key = "0")]
    pub conv0: YoloConv,
    #[module(key = "1")]
    pub conv1: YoloConv,
    #[module(key = "2")]
    pub c3k2_2: C3k2,
    #[module(key = "3")]
    pub conv3: YoloConv,
    #[module(key = "4")]
    pub c3k2_4: C3k2,
    #[module(key = "5")]
    pub conv5: YoloConv,
    #[module(key = "6")]
    pub c3k2_6: C3k2,
    #[module(key = "7")]
    pub conv7: YoloConv,
    #[module(key = "8")]
    pub c3k2_8: C3k2,
    #[module(key = "9")]
    pub c2psa9: C2PSA,
}

impl YoloBackboneCls {
    pub fn empty(scale: YoloScale) -> Self {
        let d = |yaml_n| make_depth(yaml_n, scale);
        let [c0, c1, c2, c3, c4] = scaled_channels(scale);
        Self {
            conv0: YoloConv::empty(3, c0, 3, 2, true),
            conv1: YoloConv::empty(c0, c1, 3, 2, true).tk(),
            c3k2_2: C3k2::empty(c1, c2, d(2), true, 0.25, scale.forces_c3k(), false),
            conv3: YoloConv::empty(c2, c2, 3, 2, true).tk(),
            c3k2_4: C3k2::empty(c2, c3, d(2), true, 0.25, scale.forces_c3k(), false),
            conv5: YoloConv::empty(c3, c3, 3, 2, true).tk(),
            c3k2_6: C3k2::empty(c3, c3, d(2), true, 0.5, true, false),
            conv7: YoloConv::empty(c3, c4, 3, 2, true).tk(),
            c3k2_8: C3k2::empty(c4, c4, d(2), true, 0.5, true, false),
            c2psa9: C2PSA::empty(c4, c4, d(2), 0.5),
        }
    }

    /// Run layers 0–9, returning the final feature map.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = scoped("0", || self.conv0.forward(x))?;
        let x = scoped("1", || self.conv1.forward(&x))?;
        let x = scoped("2", || self.c3k2_2.forward(&x))?;
        let x = scoped("3", || self.conv3.forward(&x))?;
        let x = scoped("4", || self.c3k2_4.forward(&x))?;
        let x = scoped("5", || self.conv5.forward(&x))?;
        let x = scoped("6", || self.c3k2_6.forward(&x))?;
        let x = scoped("7", || self.conv7.forward(&x))?;
        let x = scoped("8", || self.c3k2_8.forward(&x))?;
        scoped("9", || self.c2psa9.forward(&x))
    }
}
