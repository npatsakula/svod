use svod_tensor::Tensor;
use svod_tensor::nn::Module;

use super::conv::YoloConv;
use crate::state::scoped;
use crate::yolo::error::Result;

/// Standard YOLO bottleneck: two Conv+BN+SiLU layers with optional residual,
/// which rides `cv2`'s epilogue.
///
/// State-dict keys: `cv1.{conv,bn}.*`, `cv2.{conv,bn}.*`.
#[derive(Clone, Module)]
pub struct YoloBottleneck {
    pub cv1: YoloConv,
    pub cv2: YoloConv,
    pub add: bool,
}

impl YoloBottleneck {
    /// Default: `k=(3,3)`, `e=0.5`.
    pub fn empty(in_ch: usize, out_ch: usize, shortcut: bool) -> Self {
        Self::empty_full(in_ch, out_ch, shortcut, 3, 3, 0.5)
    }

    /// Full control: separate kernel sizes for cv1/cv2 and expansion ratio.
    pub fn empty_full(in_ch: usize, out_ch: usize, shortcut: bool, k1: usize, k2: usize, e: f64) -> Self {
        let c_ = (out_ch as f64 * e) as usize;
        let add = shortcut && in_ch == out_ch;
        Self { cv1: YoloConv::empty(in_ch, c_, k1, 1, true), cv2: YoloConv::empty(c_, out_ch, k2, 1, true), add }
    }

    /// `x` is `[B, H, W, C]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = scoped("cv1", || self.cv1.forward(x))?;
        scoped("cv2", || if self.add { self.cv2.forward_residual(&h, x) } else { self.cv2.forward(&h) })
    }
}
