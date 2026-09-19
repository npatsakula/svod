use svod_tensor::Tensor;
use svod_tensor::nn::Module;

use super::conv::{YoloConv, store_channels_last, tensor_core_dtype};
use crate::state::scoped;
use crate::yolo::error::Result;

/// Standard YOLO bottleneck: two Conv+BN+SiLU layers with optional residual.
///
/// State-dict keys: `cv1.{conv,bn}.*`, `cv2.{conv,bn}.*`.
#[derive(Clone, Module)]
pub struct YoloBottleneck {
    pub cv1: YoloConv,
    pub cv2: YoloConv,
    pub add: bool,
    /// The block's output is stored channels-last; see [`Self::channels_last`].
    #[module(skip)]
    pub channels_last: bool,
    /// The block takes and returns `[B, H, W, C]`; see [`Self::nhwc`].
    #[module(skip)]
    pub nhwc: bool,
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
        let cv1 = YoloConv::empty(in_ch, c_, k1, 1, true);
        Self { cv1, cv2: YoloConv::empty(c_, out_ch, k2, 1, true), add, channels_last: false, nhwc: false }
    }

    /// Store both `cv1`'s output and the block's channels-last: `cv1` feeds
    /// only `cv2`, a 3x3, and the block's output goes to the next block's 3x3
    /// or a 1x1. The residual add stays in `cv2`'s epilogue, ahead of the store.
    pub fn channels_last(mut self) -> Self {
        self.cv1 = self.cv1.channels_last().channels_last_input();
        self.cv2 = self.cv2.channels_last_input();
        self.channels_last = true;
        self
    }

    /// Both convs on the tk kernel, taking and returning NCHW: the first pays
    /// one permute of its input, the rest of the chain is already channels-last.
    pub fn tk(mut self) -> Self {
        if !self.tk_eligible() {
            return self.channels_last();
        }
        self.cv1 = self.cv1.tk().nhwc_out();
        self.cv2 = self.cv2.tk().nhwc_in();
        self
    }

    /// Both convs can run the tk kernel ([`YoloConv::tk_eligible`]). A chain
    /// whose middle falls back would pay for the layout and get nothing.
    pub fn tk_eligible(&self) -> bool {
        self.cv1.tk_eligible() && self.cv2.tk_eligible()
    }

    /// Take and return `[B, H, W, C]`, both convs on the tk kernel
    /// ([`YoloConv::nhwc`]). The residual add is elementwise, so it needs no
    /// layout of its own.
    pub fn nhwc(mut self) -> Self {
        // Where the kernel declines, the channels-last store still helps the
        // graph conv that runs instead, so fall back to that rather than to
        // plain NCHW.
        if !self.tk_eligible() {
            return self.channels_last();
        }
        self.cv1 = self.cv1.tk().nhwc_in().nhwc_out();
        self.cv2 = self.cv2.tk().nhwc_in().nhwc_out();
        self.nhwc = true;
        self
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = scoped("cv1", || self.cv1.forward(x))?;
        let out = scoped("cv2", || self.cv2.forward(&h))?;
        let out = if self.add { out.try_add(x)? } else { out };
        if self.channels_last && tensor_core_dtype(&out.dtype()) { store_channels_last(&out) } else { Ok(out) }
    }
}
