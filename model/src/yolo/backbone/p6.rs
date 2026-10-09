//! [`YoloBackboneP6`] — YOLO v26-P6 backbone (layers 0–12).
//!
//! Deeper than the standard backbone: adds a P6/64 stage (Conv→C3k2) after
//! P5/32, and P5 is narrowed to 768 channels. SPPF uses `shortcut=false`.

use svod_tensor::Tensor;
use svod_tensor::nn::Module;

use crate::state::scoped;
use crate::yolo::blocks::attention::C2PSA;
use crate::yolo::blocks::conv::YoloConv;
use crate::yolo::blocks::csp::C3k2;
use crate::yolo::blocks::sppf::Sppf;
use crate::yolo::config::{YoloScale, make_depth, scale_channels};
use crate::yolo::error::Result;

/// P6 backbone channels after scaling: `[c0, c1, c2, c3, c5, c6]`.
/// `c5 = scale_channels(768)` (P5), `c6 = scale_channels(1024)` (P6).
pub fn p6_scaled_channels(scale: YoloScale) -> [usize; 6] {
    let sc = |yaml_c| scale_channels(yaml_c, scale);
    [sc(64), sc(128), sc(256), sc(512), sc(768), sc(1024)]
}

/// Full YOLO v26-P6 backbone (layers 0–12).
///
/// Forward returns four skip-connection outputs: `(l4, l6, l8, l12)`
/// at strides 8, 16, 32, 64.
#[derive(Clone, Module)]
pub struct YoloBackboneP6 {
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
    pub conv9: YoloConv,
    #[module(key = "10")]
    pub c3k2_10: C3k2,
    #[module(key = "11")]
    pub sppf11: Sppf,
    #[module(key = "12")]
    pub c2psa12: C2PSA,
}

impl YoloBackboneP6 {
    pub fn empty(scale: YoloScale) -> Self {
        let d = |yaml_n| make_depth(yaml_n, scale);
        let [c0, c1, c2, c3, c5, c6] = p6_scaled_channels(scale);
        Self {
            conv0: YoloConv::empty(3, c0, 3, 2, true),
            conv1: YoloConv::empty(c0, c1, 3, 2, true),
            c3k2_2: C3k2::empty(c1, c2, d(2), true, 0.25, scale.forces_c3k(), false),
            conv3: YoloConv::empty(c2, c2, 3, 2, true),
            c3k2_4: C3k2::empty(c2, c3, d(2), true, 0.25, scale.forces_c3k(), false),
            conv5: YoloConv::empty(c3, c3, 3, 2, true),
            c3k2_6: C3k2::empty(c3, c3, d(2), true, 0.5, true, false),
            conv7: YoloConv::empty(c3, c5, 3, 2, true),
            c3k2_8: C3k2::empty(c5, c5, d(2), true, 0.5, true, false),
            conv9: YoloConv::empty(c5, c6, 3, 2, true),
            c3k2_10: C3k2::empty(c6, c6, d(2), true, 0.5, true, false),
            sppf11: Sppf::empty(c6, c6, 5, 3, false),
            c2psa12: C2PSA::empty(c6, c6, d(2), 0.5),
        }
    }

    /// Run backbone layers 0–12, returning `(l4, l6, l8, l12)`.
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let x = scoped("0", || self.conv0.forward(x))?;
        let x = scoped("1", || self.conv1.forward(&x))?;
        let x = scoped("2", || self.c3k2_2.forward(&x))?;
        let x = scoped("3", || self.conv3.forward(&x))?;
        let l4 = scoped("4", || self.c3k2_4.forward(&x))?;
        let x = scoped("5", || self.conv5.forward(&l4))?;
        let l6 = scoped("6", || self.c3k2_6.forward(&x))?;
        let x = scoped("7", || self.conv7.forward(&l6))?;
        let l8 = scoped("8", || self.c3k2_8.forward(&x))?;
        let x = scoped("9", || self.conv9.forward(&l8))?;
        let x = scoped("10", || self.c3k2_10.forward(&x))?;
        let x = scoped("11", || self.sppf11.forward(&x))?;
        let l12 = scoped("12", || self.c2psa12.forward(&x))?;
        Ok((l4, l6, l8, l12))
    }
}
