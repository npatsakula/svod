use svod_tensor::Tensor;
use svod_tensor::nn::Module;

use super::attention::PSABlock;
use super::bottleneck::YoloBottleneck;
use super::conv::YoloConv;
use crate::state::{scoped, scoped_index};
use crate::yolo::error::Result;

/// `cv1 → chunk(2) → chain → cat → cv2`, the forward shared by C2f and C3k2.
/// Each link of the chain feeds the next and contributes its own output to the
/// concatenation.
fn forward_chain<B>(
    cv1: &YoloConv,
    cv2: &YoloConv,
    chain: &[B],
    step: impl Fn(&B, &Tensor) -> Result<Tensor>,
    x: &Tensor,
) -> Result<Tensor> {
    let mut parts = scoped("cv1", || cv1.forward(x))?.chunk(2, 1)?;
    for (i, blk) in chain.iter().enumerate() {
        let next = scoped_index("m", i, || step(blk, parts.last().expect("cv1 output is chunked in two")))?;
        parts.push(next);
    }
    let cat = Tensor::cat(&parts.iter().collect::<Vec<_>>(), 1)?;
    scoped("cv2", || cv2.forward(&cat))
}

// ---------------------------------------------------------------------------
// C2f — the C2f/C3k2 parent: 1×1 conv → chunk(2) → chain → cat → 1×1 conv
// ---------------------------------------------------------------------------

/// Faster CSP Bottleneck: 1×1 conv splits into two halves; the second half
/// passes through a chain of bottlenecks; all outputs are concatenated and
/// merged by a final 1×1 conv.
///
/// State-dict keys: `cv1.*`, `cv2.*`, `m.{i}.*`.
#[derive(Clone, Module)]
pub struct C2f {
    pub cv1: YoloConv,
    pub cv2: YoloConv,
    pub m: Vec<YoloBottleneck>,
    pub c_hidden: usize,
}

impl C2f {
    pub fn empty(in_ch: usize, out_ch: usize, n: usize, shortcut: bool, e: f64) -> Self {
        let c_hidden = (out_ch as f64 * e) as usize;
        Self {
            cv1: YoloConv::empty(in_ch, 2 * c_hidden, 1, 1, true),
            cv2: YoloConv::empty((2 + n) * c_hidden, out_ch, 1, 1, true),
            m: (0..n).map(|_| YoloBottleneck::empty(c_hidden, c_hidden, shortcut)).collect(),
            c_hidden,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        forward_chain(&self.cv1, &self.cv2, &self.m, YoloBottleneck::forward, x)
    }
}

// ---------------------------------------------------------------------------
// C3k — CSP Bottleneck with 3 convs and configurable kernel size
// ---------------------------------------------------------------------------

/// CSP with three 1×1 convs and n inner bottlenecks with kernel `k`.
///
/// State-dict keys: `cv1.*`, `cv2.*`, `cv3.*`, `m.{i}.*`.
#[derive(Clone, Module)]
pub struct C3k {
    pub cv1: YoloConv,
    pub cv2: YoloConv,
    pub cv3: YoloConv,
    pub m: Vec<YoloBottleneck>,
}

impl C3k {
    pub fn empty(in_ch: usize, out_ch: usize, n: usize, shortcut: bool, e: f64, k: usize) -> Self {
        let c_ = (out_ch as f64 * e) as usize;
        // `cv1` hands the chain the `[B, H, W, C]` tensor itself, so its 3x3
        // convs run on the tk kernel with no layout change between them; the
        // chain's output reaches `cv3`'s `cat` as an NCHW view, free to a graph
        // op. `cv2` feeds only `cv3`.
        let m: Vec<YoloBottleneck> =
            (0..n).map(|_| YoloBottleneck::empty_full(c_, c_, shortcut, k, k, 1.0).nhwc()).collect();
        // The chain is `[B, H, W, C]` only if every link took it; one link on
        // the graph path would pay for the layout and get nothing back.
        let chained = m.iter().all(|b| b.nhwc);
        let cv1 = YoloConv::empty(in_ch, c_, 1, 1, true);
        Self {
            cv1: if chained { cv1.nhwc_out() } else { cv1.channels_last() },
            cv2: YoloConv::empty(in_ch, c_, 1, 1, true),
            cv3: YoloConv::empty(2 * c_, out_ch, 1, 1, true),
            m,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let left = scoped("cv1", || self.cv1.forward(x))?;
        let left =
            self.m.iter().enumerate().try_fold(left, |acc, (i, blk)| scoped_index("m", i, || blk.forward(&acc)))?;
        // The chain ran in `[B, H, W, C]`; the `cat` takes the NCHW view of it.
        let left = if self.cv1.nhwc_at(&x.dtype()) { left.try_permute(&[0, 3, 1, 2])? } else { left };
        let right = scoped("cv2", || self.cv2.forward(x))?;
        let cat = Tensor::cat(&[&left, &right], 1)?;
        scoped("cv3", || self.cv3.forward(&cat))
    }
}

// ---------------------------------------------------------------------------
// C3k2 — C2f parent with pluggable inner blocks
// ---------------------------------------------------------------------------

/// Inner block for C3k2: plain Bottleneck, C3k, or `Sequential(Bottleneck,
/// PSABlock)` when `attn=True` — the last one keyed `0.*` / `1.*`.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Module)]
pub enum C3k2Inner {
    Bottleneck(YoloBottleneck),
    C3k(C3k),
    Attn(YoloBottleneck, PSABlock),
}

impl C3k2Inner {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            C3k2Inner::Bottleneck(b) => b.forward(x),
            C3k2Inner::C3k(c) => c.forward(x),
            C3k2Inner::Attn(b, psa) => {
                let h = scoped("0", || b.forward(x))?;
                scoped("1", || psa.forward(&h))
            }
        }
    }
}

/// C3k2: C2f with pluggable inner blocks (Bottleneck, C3k, or attn).
///
/// State-dict keys: `cv1.*`, `cv2.*`, `m.{i}.*`.
#[derive(Clone, Module)]
pub struct C3k2 {
    pub cv1: YoloConv,
    pub cv2: YoloConv,
    pub m: Vec<C3k2Inner>,
    pub c_hidden: usize,
}

impl C3k2 {
    pub fn empty(in_ch: usize, out_ch: usize, n: usize, shortcut: bool, e: f64, c3k: bool, attn: bool) -> Self {
        let c_hidden = (out_ch as f64 * e) as usize;
        let num_heads = (c_hidden / 64).max(1);
        let inner = || {
            // A bottleneck reached from the `chunk`/`cat` chain keeps NCHW at
            // its edges — those are channel-axis ops — and runs the tk kernel
            // inside ([`YoloBottleneck::tk`]).
            if attn {
                C3k2Inner::Attn(
                    YoloBottleneck::empty(c_hidden, c_hidden, shortcut).tk(),
                    PSABlock::empty(c_hidden, num_heads),
                )
            } else if c3k {
                C3k2Inner::C3k(C3k::empty(c_hidden, c_hidden, 2, shortcut, 0.5, 3))
            } else {
                C3k2Inner::Bottleneck(YoloBottleneck::empty(c_hidden, c_hidden, shortcut).tk())
            }
        };
        Self {
            cv1: YoloConv::empty(in_ch, 2 * c_hidden, 1, 1, true),
            cv2: YoloConv::empty((2 + n) * c_hidden, out_ch, 1, 1, true),
            m: (0..n).map(|_| inner()).collect(),
            c_hidden,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        forward_chain(&self.cv1, &self.cv2, &self.m, C3k2Inner::forward, x)
    }
}
