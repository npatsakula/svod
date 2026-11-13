use svod_dtype::DType;
use svod_ir::SInt;
use svod_tensor::Tensor;
use svod_tensor::nn::Module;
use svod_tk::ops;

use super::attention::PSABlock;
use super::bottleneck::YoloBottleneck;
use super::conv::YoloConv;
use crate::state::{scoped, scoped_index};
use crate::yolo::error::Result;

/// `cv1 → chunk(2) → chain → cat → cv2`, the forward shared by C2f and C3k2.
/// Each link of the chain feeds the next and contributes its own output to the
/// concatenation. The parts live in one buffer, the concatenation: `cv1`
/// stores its two halves into the first channels, each link reads the part
/// before it in place and stores its own after it, and `cv2` reads the whole.
fn forward_chain<B>(
    cv1: &YoloConv,
    cv2: &YoloConv,
    chain: &[B],
    step: impl Fn(&B, &Tensor, (&Tensor, usize)) -> Result<Tensor>,
    x: &Tensor,
) -> Result<Tensor> {
    let c = cv1.weight.dim_const(0)? / 2;
    let cat = parts(x, (2 + chain.len()) * c, cv1.emitted(x.dtype()))?;
    let mut cat = scoped("cv1", || cv1.forward_into(x, None, (&cat, 0)))?;
    for (i, blk) in chain.iter().enumerate() {
        let last = channels(&cat, (1 + i) * c, c)?;
        cat = scoped_index("m", i, || step(blk, &last, (&cat, (2 + i) * c)))?;
    }
    scoped("cv2", || cv2.forward(&cat))
}

/// The buffer a block's parts are stored into: `x`'s pixels, `width` channels.
fn parts(x: &Tensor, width: usize, dtype: DType) -> Result<Tensor> {
    let shape = x.shape()?;
    let dims: Vec<SInt> = shape[..shape.len() - 1].iter().cloned().chain([SInt::Const(width)]).collect();
    Ok(ops::buffer(&dims, dtype)?)
}

/// Channels `at..at + width` of `t`.
fn channels(t: &Tensor, at: usize, width: usize) -> Result<Tensor> {
    let rank = t.shape()?.len();
    let range = |d: usize| (d + 1 == rank).then_some((at as isize, (at + width) as isize));
    Ok(t.try_shrink((0..rank).map(range))?)
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
        forward_chain(&self.cv1, &self.cv2, &self.m, YoloBottleneck::forward_into, x)
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
        Self {
            cv1: YoloConv::empty(in_ch, c_, 1, 1, true),
            cv2: YoloConv::empty(in_ch, c_, 1, 1, true),
            cv3: YoloConv::empty(2 * c_, out_ch, 1, 1, true),
            m: (0..n).map(|_| YoloBottleneck::empty_full(c_, c_, shortcut, k, k, 1.0)).collect(),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let cat = self.parts(x)?;
        scoped("cv3", || self.cv3.forward(&cat))
    }

    /// [`Self::forward`] stored into channels `at..` of `into`: `into` after the store.
    pub fn forward_into(&self, x: &Tensor, into: (&Tensor, usize)) -> Result<Tensor> {
        let cat = self.parts(x)?;
        scoped("cv3", || self.cv3.forward_into(&cat, None, into))
    }

    /// `cv3`'s input: the bottleneck chain's output beside `cv2`'s, the
    /// chain's last link and `cv2` storing into one buffer.
    fn parts(&self, x: &Tensor) -> Result<Tensor> {
        let c = self.cv1.weight.dim_const(0)?;
        let cat = parts(x, 2 * c, self.cv1.emitted(x.dtype()))?;
        let cat = match self.m.split_last() {
            None => scoped("cv1", || self.cv1.forward_into(x, None, (&cat, 0)))?,
            Some((last, chain)) => {
                let left = scoped("cv1", || self.cv1.forward(x))?;
                let step = |acc: Tensor, (i, blk): (usize, &YoloBottleneck)| scoped_index("m", i, || blk.forward(&acc));
                let left = chain.iter().enumerate().try_fold(left, step)?;
                scoped_index("m", chain.len(), || last.forward_into(&left, (&cat, 0)))?
            }
        };
        scoped("cv2", || self.cv2.forward_into(x, None, (&cat, c)))
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

    /// [`Self::forward`] stored into channels `at..` of `into`: `into` after the store.
    pub fn forward_into(&self, x: &Tensor, into: (&Tensor, usize)) -> Result<Tensor> {
        match self {
            C3k2Inner::Bottleneck(b) => b.forward_into(x, into),
            C3k2Inner::C3k(c) => c.forward_into(x, into),
            C3k2Inner::Attn(..) => Ok(ops::store_into(into.0, into.1, &self.forward(x)?)?),
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
            if attn {
                C3k2Inner::Attn(
                    YoloBottleneck::empty(c_hidden, c_hidden, shortcut),
                    PSABlock::empty(c_hidden, num_heads),
                )
            } else if c3k {
                C3k2Inner::C3k(C3k::empty(c_hidden, c_hidden, 2, shortcut, 0.5, 3))
            } else {
                C3k2Inner::Bottleneck(YoloBottleneck::empty(c_hidden, c_hidden, shortcut))
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
        forward_chain(&self.cv1, &self.cv2, &self.m, C3k2Inner::forward_into, x)
    }
}
