use svod_tensor::Tensor;
use svod_tensor::nn::Module;

use super::conv::{YoloConv, to_nchw, to_nhwc};
use crate::state::scoped;
use crate::yolo::error::Result;

/// Spatial Pyramid Pooling - Fast: 1×1 conv → `pools` chained MaxPool(k) →
/// cat → 1×1 conv over `[B, H, W, C]`. When `shortcut` is true and
/// `in_ch == out_ch`, a residual connection is added in `cv2`'s epilogue.
///
/// State-dict keys: `cv1.{conv,bn}.*`, `cv2.{conv,bn}.*`.
#[derive(Clone, Module)]
pub struct Sppf {
    pub cv1: YoloConv,
    pub cv2: YoloConv,
    pub add: bool,
    pub kernel: usize,
    pub pools: usize,
}

impl Sppf {
    pub fn empty(in_ch: usize, out_ch: usize, kernel: usize, pools: usize, shortcut: bool) -> Self {
        let c_hidden = in_ch / 2;
        Self {
            cv1: YoloConv::empty(in_ch, c_hidden, 1, 1, false),
            cv2: YoloConv::empty(c_hidden * (pools + 1), out_ch, 1, 1, true),
            add: shortcut && in_ch == out_ch,
            kernel,
            pools,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let p = (self.kernel / 2) as isize;
        let mut y = scoped("cv1", || self.cv1.forward(x))?;
        let mut ys = Vec::with_capacity(self.pools + 1);
        ys.push(y.clone());
        for _ in 0..self.pools {
            let pooled = to_nchw(&y)?
                .max_pool2d()
                .kernel_size(&[self.kernel, self.kernel])
                .stride(&[1, 1])
                .padding(&[(p, p), (p, p)])
                .call()?;
            y = to_nhwc(&pooled)?;
            ys.push(y.clone());
        }
        let cat = Tensor::cat(&ys.iter().collect::<Vec<_>>(), -1)?;
        scoped("cv2", || if self.add { self.cv2.forward_residual(&cat, x) } else { self.cv2.forward(&cat) })
    }
}
