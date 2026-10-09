use svod_dtype::DType;
use svod_ir::SInt;
use svod_tensor::Tensor;
use svod_tensor::nn::{BatchNorm2d, Conv2d, ConvTranspose2d, Layer, Module, StateDict, get_tensor, prefixed};
use svod_tk3::ops::{self, Act};

use crate::blocks::batchnorm2d_with_eps;
use crate::init::fan_in_uniform;

use crate::yolo::error::Result;
use crate::yolo::loader::fold_norm;

/// Ultralytics' `initialize_weights` rewrites every BatchNorm's epsilon to
/// 1e-3, so YOLO checkpoints are not normalized with PyTorch's 1e-5 default.
pub const YOLO_BN_EPS: f64 = 1e-3;

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// A `kernel×kernel` convolution with a bias and `kernel / 2` padding, as the
/// Detect head's final 1×1 layers use it. State-dict keys: `weight`, `bias`.
/// Runs channels-last through [`pointwise`].
pub fn conv2d_bias(in_ch: usize, out_ch: usize, kernel: usize, stride: usize) -> Conv2d {
    let fan_in = in_ch * kernel * kernel;
    let bias = fan_in_uniform(&[out_ch], fan_in, DType::Float32);
    let p = (kernel / 2) as isize;
    Conv2d::new(fan_in_uniform(&[out_ch, in_ch, kernel, kernel], fan_in, DType::Float32), Some(bias))
        .with_stride((stride, stride))
        .with_padding(((p, p), (p, p)))
}

/// A biased transposed convolution that doubles the spatial resolution.
/// State-dict keys: `weight`, `bias`. Runs channels-last through [`deconv`].
pub fn deconv2d_2x(in_ch: usize, out_ch: usize, kernel: usize) -> ConvTranspose2d {
    let fan_in = in_ch * kernel * kernel;
    ConvTranspose2d::new(
        fan_in_uniform(&[in_ch, out_ch, kernel, kernel], fan_in, DType::Float32),
        Some(fan_in_uniform(&[out_ch], fan_in, DType::Float32)),
    )
    .with_stride((2, 2))
}

/// [`conv2d_bias`]'s 1×1 over `[B, H, W, C]`, emitting its `acc_dtype` when it
/// has one: the [`ops::conv2d`] its `[cout, 1, 1, cin]` weight is a reshape of.
pub fn pointwise(conv: &Conv2d, x: &Tensor) -> Result<Tensor> {
    let [cout, cin, 1, 1] = conv.weight.dims()?[..] else { panic!("a pointwise head conv") };
    let w = conv.weight.cast(x.dtype()).try_reshape([cout, 1, 1, cin])?;
    let bias = conv.bias.as_ref().map(|b| b.cast(x.dtype()));
    let opts = ops::Conv { bias: bias.as_ref(), out_dtype: conv.acc_dtype.clone(), ..Default::default() };
    Ok(ops::conv2d(x, &w, opts)?)
}

/// [`deconv2d_2x`] over `[B, H, W, C]`: the graph's transposed convolution on
/// the NCHW view.
pub fn deconv(conv: &ConvTranspose2d, x: &Tensor) -> Result<Tensor> {
    Ok(conv.forward(&to_nchw(x)?)?.try_permute(&[0, 2, 3, 1])?)
}

/// The `[B, C, H, W]` view of `[B, H, W, C]`.
pub(crate) fn to_nchw(x: &Tensor) -> Result<Tensor> {
    Ok(x.try_permute(&[0, 3, 1, 2])?)
}

/// The `[B, H, W, C]` view of `[B, C, H, W]`.
pub(crate) fn to_nhwc(x: &Tensor) -> Result<Tensor> {
    Ok(x.try_permute(&[0, 2, 3, 1])?)
}

/// Nearest-neighbour upsampling of `[B, H, W, C]` by `factor`: every pixel
/// repeated, as a view.
pub(crate) fn upsample_nearest(x: &Tensor, factor: usize) -> Result<Tensor> {
    let [b, h, w, c]: [SInt; 4] = x.shape()?.to_vec().try_into().expect("a [B, H, W, C] map");
    let (h, w) = (h.as_const().expect("a static height"), w.as_const().expect("a static width"));
    let spread = x.try_reshape([b.clone(), h.into(), 1.into(), w.into(), 1.into(), c.clone()])?;
    let spread = spread.try_expand([b.clone(), h.into(), factor.into(), w.into(), factor.into(), c.clone()])?;
    Ok(spread.try_reshape([b, (h * factor).into(), (w * factor).into(), c])?)
}

/// Conv2d(bias=False) + BatchNorm2d + SiLU, the universal YOLO building block,
/// over channels-last `[B, H, W, C]` activations. When `act` is `false` the
/// activation is skipped (used by SPPF.cv1, Attention projections, and
/// PSABlock FFN output conv).
///
/// State-dict keys: `conv.weight`, `bn.{weight,bias,running_mean,running_var}`,
/// and `conv.bias` once the norm is folded into the conv: a conv that carries a
/// bias is taken as already normalized, and one that does not has its norm
/// folded when it loads. The block stores `[cout, kh, kw, cin]`, the layout a
/// channels-last convolution reduces over, and runs [`ops::conv2d`] with the
/// bias, the activation and a residual in its epilogue.
#[derive(Clone)]
pub struct YoloConv {
    /// `[cout, kh, kw, cin / groups]`, the norm folded in.
    pub weight: Tensor,
    /// `[cout]`: the folded norm's shift.
    pub bias: Tensor,
    /// The checkpoint's norm, kept for the state dict; folded into `weight`
    /// and `bias`.
    pub bn: BatchNorm2d,
    pub stride: usize,
    pub groups: usize,
    pub act: bool,
    /// Accumulate and emit this dtype; see [`Self::with_acc_dtype`].
    pub acc_dtype: Option<DType>,
    /// Cast the input to this dtype first; see [`Self::with_io_dtype`].
    pub in_dtype: Option<DType>,
    /// Cast the output to this dtype last; see [`Self::with_io_dtype`].
    pub out_dtype: Option<DType>,
}

impl YoloConv {
    pub fn empty(in_ch: usize, out_ch: usize, kernel: usize, stride: usize, act: bool) -> Self {
        Self::grouped(in_ch, out_ch, kernel, stride, 1, act)
    }

    /// Depthwise variant: `groups = gcd(in_ch, out_ch)`.
    pub fn empty_dw(in_ch: usize, out_ch: usize, kernel: usize, stride: usize, act: bool) -> Self {
        Self::grouped(in_ch, out_ch, kernel, stride, gcd(in_ch, out_ch), act)
    }

    fn grouped(in_ch: usize, out_ch: usize, kernel: usize, stride: usize, groups: usize, act: bool) -> Self {
        let cin = in_ch / groups;
        let weight = fan_in_uniform(&[out_ch, cin, kernel, kernel], cin * kernel * kernel, DType::Float32);
        let bn = batchnorm2d_with_eps(out_ch, YOLO_BN_EPS);
        let (weight, bias) = fold_norm(&weight, &bn).expect("a placeholder norm folds");
        Self {
            weight: weight.try_permute(&[0, 2, 3, 1]).expect("a 4-D weight"),
            bias,
            bn,
            stride,
            groups,
            act,
            acc_dtype: None,
            in_dtype: None,
            out_dtype: None,
        }
    }

    /// Cast this block's input and/or output, which is how a stage is pinned to
    /// a dtype the rest of the model does not run at: every layer reads its
    /// width off the stream, so casting at a block's edge carries the whole
    /// stage with it.
    pub fn with_io_dtype(mut self, in_dtype: Option<DType>, out_dtype: Option<DType>) -> Self {
        (self.in_dtype, self.out_dtype) = (in_dtype, out_dtype);
        self
    }

    /// Accumulate the conv in `dtype` and emit it there, so half-width operands
    /// still leave the block at full width. (Otherwise the accumulator, the
    /// bias and the activation stay at f32 and round once, to the operands'
    /// dtype, at the store.)
    pub fn with_acc_dtype(mut self, dtype: DType) -> Self {
        self.acc_dtype = Some(dtype);
        self
    }

    /// `x` is `[B, H, W, C]`; so is the output.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_with(x, None)
    }

    /// [`Self::forward`] plus `residual`, added after the activation in the
    /// convolution's epilogue.
    pub fn forward_residual(&self, x: &Tensor, residual: &Tensor) -> Result<Tensor> {
        self.forward_with(x, Some(residual))
    }

    fn forward_with(&self, x: &Tensor, residual: Option<&Tensor>) -> Result<Tensor> {
        let cast = |t: &Tensor| self.in_dtype.clone().map_or_else(|| t.clone(), |dt| t.cast(dt));
        let x = cast(x);
        // The emitted dtype, which a residual added in the epilogue must have.
        let out = self.acc_dtype.clone().unwrap_or(x.dtype());
        let residual = residual.map(|r| r.cast(out.clone()));
        // A stage pinned wider than the weights runs them at its own width.
        let (w, bias) = (self.weight.cast(x.dtype()), self.bias.cast(x.dtype()));
        let pad = self.weight.dim_const(1)? / 2;
        let opts = ops::Conv {
            stride: [self.stride; 2],
            pad: [pad; 2],
            groups: self.groups,
            bias: Some(&bias),
            act: if self.act { Act::Silu } else { Act::None },
            residual: residual.as_ref(),
            out_dtype: self.acc_dtype.clone(),
            ..Default::default()
        };
        let y = ops::conv2d(&x, &w, opts)?;
        Ok(self.out_dtype.clone().map_or_else(|| y.clone(), |dt| y.cast(dt)))
    }
}

impl Module for YoloConv {
    fn write_state(&self, prefix: &str, out: &mut StateDict) {
        let conv = prefixed(prefix, "conv");
        let weight = self.weight.try_permute(&[0, 3, 1, 2]).expect("a 4-D weight");
        out.insert(prefixed(&conv, "weight"), weight);
        out.insert(prefixed(&conv, "bias"), self.bias.clone());
        self.bn.write_state(&prefixed(prefix, "bn"), out);
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> svod_tensor::error::Result<()> {
        let conv = prefixed(prefix, "conv");
        self.bn.load_state_dict(sd, &prefixed(prefix, "bn"))?;
        let weight = get_tensor(sd, &prefixed(&conv, "weight"))?;
        let (weight, bias) = match sd.get(&prefixed(&conv, "bias")) {
            Some(bias) => (weight, bias.clone()),
            None => fold_norm(&weight, &self.bn)?,
        };
        let weight = weight.try_permute(&[0, 2, 3, 1])?.contiguous();
        let bias = bias.contiguous();
        weight.realize()?;
        bias.realize()?;
        (self.weight, self.bias) = (weight, bias);
        Ok(())
    }
}
