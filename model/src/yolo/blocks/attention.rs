use svod_ir::SInt;
use svod_tensor::Tensor;
use svod_tensor::nn::Module;
use svod_tk3::ops::{self, Attn};

use super::conv::YoloConv;
use crate::state::{scoped, scoped_index};
use crate::yolo::error::Result;

/// Multi-head attention with a depthwise-conv positional encoding.
///
/// Computes full N×N attention but with a reduced key dimension
/// (`key_dim = head_dim * attn_ratio`) to cut QK^T cost.
/// All projections are 1×1 convs (no activation).
///
/// State-dict keys: `qkv.{conv,bn}.*`, `proj.{conv,bn}.*`, `pe.{conv,bn}.*`.
#[derive(Clone, Module)]
pub struct Attention {
    pub qkv: YoloConv,
    pub proj: YoloConv,
    pub pe: YoloConv,
    pub num_heads: usize,
    pub key_dim: usize,
    pub head_dim: usize,
    pub scale: f32,
}

impl Attention {
    pub fn empty(dim: usize, num_heads: usize, attn_ratio: f64) -> Self {
        let head_dim = dim / num_heads;
        let key_dim = (head_dim as f64 * attn_ratio) as usize;
        let scale = (key_dim as f64).powf(-0.5) as f32;
        let qkv_out = dim + key_dim * num_heads * 2;

        Self {
            qkv: YoloConv::empty(dim, qkv_out, 1, 1, false),
            proj: YoloConv::empty(dim, dim, 1, 1, false),
            pe: YoloConv::empty_dw(dim, dim, 3, 1, false),
            num_heads,
            key_dim,
            head_dim,
            scale,
        }
    }

    /// `x` is `[B, H, W, C]`; so is the output.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_with(x, None)
    }

    /// `x + attention(x)`, the add in the output projection's epilogue.
    pub fn forward_residual(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_with(x, Some(x))
    }

    fn forward_with(&self, x: &Tensor, residual: Option<&Tensor>) -> Result<Tensor> {
        let b = x.dim(0)?;
        let (nh, kd, hd) = (self.num_heads, self.key_dim, self.head_dim);
        let (h, w) = (x.dim_const(1)?, x.dim_const(2)?);

        // QKV 1×1 conv, its channels split into heads of `q | k | v`:
        // [B, H, W, nh·(2kd+hd)] → [B, H·W, nh, 2kd+hd].
        let qkv = scoped("qkv", || self.qkv.forward(x))?;
        let qkv = qkv.try_reshape([b.clone(), SInt::from(h * w), SInt::from(nh), SInt::from(kd * 2 + hd)])?;
        let parts = qkv.split(&[kd, kd, hd], -1)?;
        let (q, k, v) = (&parts[0], &parts[1], &parts[2]);

        // The keys are narrower than the values: zeros past `kd` leave every
        // score as it is, so all three share one head width.
        let d = kd.max(hd);
        let widen = |t: &Tensor, from: usize| -> Result<Tensor> {
            Ok(if from == d { t.clone() } else { t.try_pad(&[(0, 0), (0, 0), (0, 0), (0, (d - from) as isize)])? })
        };
        let attn = Attn { scale: Some(self.scale), ..Default::default() };
        let out = ops::attention(&widen(q, kd)?, &widen(k, kd)?, &widen(v, hd)?, attn)?;
        let out = if d == hd { out } else { out.narrow(-1, 0usize, hd)? };

        // Positional encoding: a depthwise conv over the values as a map.
        let spatial = |t: &Tensor| t.try_reshape([b.clone(), SInt::from(h), SInt::from(w), SInt::from(nh * hd)]);
        let pe = scoped("pe", || self.pe.forward(&spatial(v)?))?;

        let out = spatial(&out)?.try_add(&pe)?;
        scoped("proj", || match residual {
            Some(r) => self.proj.forward_residual(&out, r),
            None => self.proj.forward(&out),
        })
    }
}

/// Position-Sensitive Attention block: attention + FFN, both with residual.
///
/// State-dict keys: `attn.*`, `ffn.0.{conv,bn}.*`, `ffn.1.{conv,bn}.*`.
#[derive(Clone, Module)]
pub struct PSABlock {
    pub attn: Attention,
    #[module(key = "ffn.0")]
    pub ffn0: YoloConv,
    #[module(key = "ffn.1")]
    pub ffn1: YoloConv,
}

impl PSABlock {
    pub fn empty(c: usize, num_heads: usize) -> Self {
        Self {
            attn: Attention::empty(c, num_heads, 0.5),
            ffn0: YoloConv::empty(c, c * 2, 1, 1, true),
            ffn1: YoloConv::empty(c * 2, c, 1, 1, false),
        }
    }

    /// `x` is `[B, H, W, C]`; both residuals ride the epilogues of the convs
    /// that end their branch.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = scoped("attn", || self.attn.forward_residual(x))?;
        let h = scoped("ffn.0", || self.ffn0.forward(&x))?;
        scoped("ffn.1", || self.ffn1.forward_residual(&h, &x))
    }
}

/// CSP with partial self-attention: split channels, run PSABlocks on half,
/// concat, project back.
///
/// State-dict keys: `cv1.{conv,bn}.*`, `cv2.{conv,bn}.*`, `m.{i}.*`.
#[derive(Clone, Module)]
pub struct C2PSA {
    pub cv1: YoloConv,
    pub cv2: YoloConv,
    pub m: Vec<PSABlock>,
    pub c_hidden: usize,
}

impl C2PSA {
    pub fn empty(in_ch: usize, out_ch: usize, n: usize, e: f64) -> Self {
        assert_eq!(in_ch, out_ch, "C2PSA requires c1 == c2");
        let c_hidden = (out_ch as f64 * e) as usize;
        let num_heads = (c_hidden / 64).max(1);
        Self {
            cv1: YoloConv::empty(in_ch, 2 * c_hidden, 1, 1, true),
            cv2: YoloConv::empty(2 * c_hidden, out_ch, 1, 1, true),
            m: (0..n).map(|_| PSABlock::empty(c_hidden, num_heads)).collect(),
            c_hidden,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = scoped("cv1", || self.cv1.forward(x))?;
        let a = y.narrow(-1, 0usize, self.c_hidden)?;
        let b = y.narrow(-1, self.c_hidden, self.c_hidden)?;
        let b = self.m.iter().enumerate().try_fold(b, |acc, (i, blk)| scoped_index("m", i, || blk.forward(&acc)))?;
        let cat = Tensor::cat(&[&a, &b], -1)?;
        scoped("cv2", || self.cv2.forward(&cat))
    }
}
