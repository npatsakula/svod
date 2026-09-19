use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::{BatchNorm2d, Conv2d, ConvTranspose2d, Layer, Module, StateDict, prefixed};

use crate::blocks::{batchnorm2d_with_eps, conv2d, conv2d_grouped};
use crate::init::fan_in_uniform;

use crate::yolo::error::Result;

/// Ultralytics' `initialize_weights` rewrites every BatchNorm's epsilon to
/// 1e-3, so YOLO checkpoints are not normalized with PyTorch's 1e-5 default.
pub const YOLO_BN_EPS: f64 = 1e-3;

fn bn(channels: usize) -> BatchNorm2d {
    batchnorm2d_with_eps(channels, YOLO_BN_EPS)
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// A `kernel×kernel` convolution with a bias and `kernel / 2` padding, as the
/// Detect head's final 1×1 layers use it. State-dict keys: `weight`, `bias`.
pub fn conv2d_bias(in_ch: usize, out_ch: usize, kernel: usize, stride: usize) -> Conv2d {
    let fan_in = in_ch * kernel * kernel;
    let bias = fan_in_uniform(&[out_ch], fan_in, DType::Float32);
    let p = (kernel / 2) as isize;
    Conv2d::new(fan_in_uniform(&[out_ch, in_ch, kernel, kernel], fan_in, DType::Float32), Some(bias))
        .with_stride((stride, stride))
        .with_padding(((p, p), (p, p)))
}

/// A biased transposed convolution that doubles the spatial resolution.
/// State-dict keys: `weight`, `bias`.
pub fn deconv2d_2x(in_ch: usize, out_ch: usize, kernel: usize) -> ConvTranspose2d {
    let fan_in = in_ch * kernel * kernel;
    ConvTranspose2d::new(
        fan_in_uniform(&[in_ch, out_ch, kernel, kernel], fan_in, DType::Float32),
        Some(fan_in_uniform(&[out_ch], fan_in, DType::Float32)),
    )
    .with_stride((2, 2))
}

/// Conv2d(bias=False) + BatchNorm2d + SiLU — the universal YOLO building block.
/// When `act` is `false` the activation is skipped (used by SPPF.cv1,
/// Attention projections, and PSABlock FFN output conv).
///
/// State-dict keys: `conv.weight`, `bn.{weight,bias,running_mean,running_var}`.
/// A checkpoint load folds the norm into the conv ([`fold_batchnorm`]), and a
/// conv that carries a bias is taken as already normalized.
///
/// [`fold_batchnorm`]: crate::yolo::loader::fold_batchnorm
///
/// Layouts follow the tensor core, which reduces over the channels and wants a
/// fragment's K elements contiguous (at f32, which has no core on RDNA4, both
/// stay as loaded). A conv reading NCHW activations gets its
/// `k x k` weight stored taps-major, `[cout, kh, kw, cin]`, at load (under
/// BEAM a stride-2 3x3 runs 1.9-2.8x faster on gfx1201); one reading
/// channels-last activations keeps the checkpoint's `[cout, cin, kh, kw]`,
/// which measured faster there. [`Self::channels_last`] picks the output
/// layout, [`Self::channels_last_input`] declares the input's.
///
/// [`Self::nhwc`] goes further: the block takes and returns the `[B, H, W, C]`
/// tensor itself rather than an NCHW view of it, which is what lets
/// [`svod_tk::conv2d_nhwc`] bind the activation without a copy. A chain of such
/// blocks never changes layout; a graph consumer at the end takes the NCHW view
/// for free.
#[derive(Clone)]
pub struct YoloConv {
    pub conv: Conv2d,
    pub bn: BatchNorm2d,
    pub act: bool,
    /// Store the output channels-last; see [`Self::channels_last`].
    pub channels_last: bool,
    /// The input arrives channels-last; see [`Self::channels_last_input`].
    pub channels_last_input: bool,
    /// Run [`svod_tk::conv2d_nhwc`]; see [`Self::tk`].
    pub tk: bool,
    /// The input is `[B, H, W, C]`; see [`Self::nhwc_in`].
    pub nhwc_in: bool,
    /// The output is `[B, H, W, C]`; see [`Self::nhwc_out`].
    pub nhwc_out: bool,
    /// The `[cout, kh, kw, cin]` weight the tk kernel reads, filled at load for
    /// a block that takes and returns `[B, H, W, C]`. Shares its buffer with
    /// `conv.weight`.
    pub weight_taps: Option<Tensor>,
}

impl YoloConv {
    pub fn empty(in_ch: usize, out_ch: usize, kernel: usize, stride: usize, act: bool) -> Self {
        let conv = conv2d(out_ch, in_ch, kernel, stride, kernel / 2);
        Self::wrap(conv, out_ch, act)
    }

    /// Depthwise variant: `groups = gcd(in_ch, out_ch)`.
    pub fn empty_dw(in_ch: usize, out_ch: usize, kernel: usize, stride: usize, act: bool) -> Self {
        let groups = gcd(in_ch, out_ch);
        Self::wrap(conv2d_grouped(out_ch, in_ch, kernel, stride, kernel / 2, groups), out_ch, act)
    }

    fn wrap(conv: Conv2d, out_ch: usize, act: bool) -> Self {
        Self {
            conv,
            bn: bn(out_ch),
            act,
            channels_last: false,
            channels_last_input: false,
            tk: false,
            nhwc_in: false,
            nhwc_out: false,
            weight_taps: None,
        }
    }

    /// Run [`svod_tk::conv2d_nhwc`], which on gfx1201 under BEAM is 1.6-2.3x the
    /// best kernel the graph gets for a 3x3. The kernel reads and writes
    /// `[B, H, W, C]`: an NCHW input is permuted first, which costs a copy the
    /// win pays for several times over, while an NCHW output is only the view.
    /// Where the kernel declines (a dtype or shape it does not serve, another
    /// device) the block runs the graph conv, so this is a performance flag and
    /// never a correctness one.
    pub fn tk(mut self) -> Self {
        self.tk = self.tk_eligible();
        self
    }

    /// Whether [`svod_tk::conv2d_nhwc`] can serve this convolution on the
    /// channel counts alone — the part of its tiling rule that does not depend
    /// on the image: the output channels tile the widest N edge and the input
    /// channels fill a K strip. A block that fails this keeps the graph path
    /// rather than paying for a layout the kernel would decline anyway.
    pub fn tk_eligible(&self) -> bool {
        self.conv.groups == 1
            && self
                .conv
                .weight
                .dims()
                .is_ok_and(|d| d.len() == 4 && d[0].is_multiple_of(32) && d[1].is_multiple_of(16) && d[2] * d[3] > 1)
    }

    /// Emit `[B, H, W, C]` — the tensor itself, not an NCHW view of it, which is
    /// what lets the next block bind it without a copy. A graph consumer takes
    /// the view back for free.
    pub fn nhwc_out(mut self) -> Self {
        self.nhwc_out = true;
        self
    }

    /// Take `[B, H, W, C]`, so a [`Self::tk`] block needs no copy in.
    pub fn nhwc_in(mut self) -> Self {
        self.nhwc_in = true;
        self
    }

    /// Store the output channels-last, handing on the NCHW view every consumer
    /// expects. Under BEAM a 3x3 stride-1 conv reading it is 1.5-2.7x faster on
    /// gfx1201, a stride-2 conv 2-3x slower, so the producer chooses by what
    /// consumes it.
    pub fn channels_last(mut self) -> Self {
        self.channels_last = true;
        self
    }

    /// The input is stored channels-last, so the weight stays `cin`-major.
    pub fn channels_last_input(mut self) -> Self {
        self.channels_last_input = true;
        self
    }

    /// Whether the weight is stored `[cout, kh, kw, cin]`: the layout the tk
    /// kernel binds, and the one a conv reading NCHW activations wants from the
    /// graph. A conv reading channels-last keeps the checkpoint's, which
    /// measured faster there, and at f32 nothing moves.
    fn taps_major(&self) -> bool {
        (self.tk || !self.channels_last_input)
            && tensor_core_dtype(&self.conv.weight.dtype())
            && self.conv.weight.dims().is_ok_and(|d| d.len() == 4 && d[2] * d[3] > 1)
    }

    /// The tk kernel's operands, when this block can run it: a taps-major
    /// weight, the bias the norm was folded into, and no grouping.
    fn tk_operands(&self) -> Option<(&Tensor, &Tensor)> {
        let (w, bias) = (self.weight_taps.as_ref()?, self.conv.bias.as_ref()?);
        // The kernel takes the matrix core's operand dtypes and treats any other
        // as a caller bug, so an f32 model never asks.
        (self.tk && self.conv.groups == 1 && tensor_core_dtype(&w.dtype())).then_some((w, bias))
    }

    /// Accumulate the conv in `dtype` and keep the block's output there, so
    /// half-width operands still leave the norm and activation at full width.
    /// (Doing so for every block costs 5% of the forward for 0.01 px, so the
    /// default rounds the epilogue to the operand dtype.)
    pub fn with_acc_dtype(mut self, dtype: DType) -> Self {
        self.conv = self.conv.with_acc_dtype(dtype);
        self
    }

    /// Whether this block's edges really are `[B, H, W, C]` for a stream of
    /// `dtype`. Every layout here serves the matrix core, which only the
    /// half-width dtypes reach, so an f32 model keeps NCHW throughout — and
    /// because producer and consumer read the same stream dtype, they agree
    /// without being told.
    pub fn nhwc_at(&self, dtype: &DType) -> bool {
        self.nhwc_out && tensor_core_dtype(dtype)
    }

    /// `x` is `[B, H, W, C]` when [`Self::nhwc_in`] holds for its dtype, else
    /// `[B, C, H, W]`; the output follows [`Self::nhwc_at`].
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let core = tensor_core_dtype(&x.dtype());
        let (nhwc_in, nhwc_out) = (self.nhwc_in && core, self.nhwc_out && core);
        if core && let Some((w, bias)) = self.tk_operands() {
            let (stride, pad) = (self.conv.stride.0, self.conv.padding.0.0 as usize);
            let nhwc = if nhwc_in { x.clone() } else { x.try_permute(&[0, 2, 3, 1])? };
            if let Some(y) = svod_tk::conv2d_nhwc(&nhwc, w, bias, None, stride, pad, self.act)? {
                return Ok(if nhwc_out { y } else { y.try_permute(&[0, 3, 1, 2])? });
            }
        }
        let nchw = if nhwc_in { x.try_permute(&[0, 3, 1, 2])? } else { x.clone() };
        let y = self.conv.forward(&nchw)?;
        let y = if self.conv.bias.is_some() { y } else { self.bn.forward(&y)? };
        let y = if self.act { y.silu()? } else { y };
        if nhwc_out {
            return Ok(y.try_permute(&[0, 2, 3, 1])?.contiguous());
        }
        if self.channels_last && core { store_channels_last(&y) } else { Ok(y) }
    }
}

/// The layouts serve the tensor core, which the half-width dtypes reach; an
/// f32 model keeps the checkpoint's, which the scalar path reads faster.
pub(crate) fn tensor_core_dtype(dtype: &DType) -> bool {
    *dtype == DType::Float16 || *dtype == DType::BFloat16
}

impl Module for YoloConv {
    fn write_state(&self, prefix: &str, out: &mut StateDict) {
        self.conv.write_state(&prefixed(prefix, "conv"), out);
        self.bn.write_state(&prefixed(prefix, "bn"), out);
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> svod_tensor::error::Result<()> {
        self.conv.load_state_dict(sd, &prefixed(prefix, "conv"))?;
        self.bn.load_state_dict(sd, &prefixed(prefix, "bn"))?;
        if self.taps_major() {
            let taps_major = self.conv.weight.try_permute(&[0, 2, 3, 1])?.contiguous();
            taps_major.realize()?;
            // The NCHW form is kept a view: realizing it would copy the bytes
            // back cin-major, and the tk kernel binds the taps-major tensor.
            self.conv.weight = taps_major.try_permute(&[0, 3, 1, 2])?;
            self.weight_taps = self.tk.then_some(taps_major);
        } else {
            self.weight_taps = None;
        }
        Ok(())
    }
}

/// Realize an NCHW tensor channels-last and hand back the NCHW view of it.
/// Right after a conv the permute folds into the kernel's store.
pub(crate) fn store_channels_last(x: &Tensor) -> Result<Tensor> {
    Ok(x.try_permute(&[0, 2, 3, 1])?.contiguous().try_permute(&[0, 3, 1, 2])?)
}
