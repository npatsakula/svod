//! Parameterized layers with state-dict support.
//!
//! Every layer here owns its parameters plus the hyper-parameters its forward
//! needs, so it implements both [`Module`](crate::nn::Module) — state-dict keys
//! matching PyTorch's names — and [`Layer`], whose forward takes nothing but
//! the input. See [`Linear`](crate::nn::Linear) and
//! [`Conv1d`](crate::nn::Conv1d) for the same shape of type.
//!
//! `new` builds a layer from loaded tensors; `with_dims` builds one from
//! shapes, with a Kaiming-uniform weight for the convolutions and the identity
//! affine (ones / zeros) for the normalizations. Every initialized parameter
//! ends in [`contiguous`](Tensor::contiguous), so it materializes into its own
//! buffer instead of being fused into every consuming kernel.

use svod_dtype::DType;

use crate::Tensor;
use crate::nn::{Layer, Module};

type Result<T> = crate::Result<T>;

/// A Kaiming-uniform weight and, when asked for, a zero bias.
#[track_caller]
fn conv_params(weight_shape: &[usize], out_channels: usize, bias: bool, dtype: DType) -> (Tensor, Option<Tensor>) {
    let weight =
        Tensor::kaiming_uniform_with_dtype(weight_shape, 0.0, dtype.clone()).expect("non-empty shape").contiguous();
    (weight, bias.then(|| Tensor::zeros(&[out_channels], dtype).contiguous()))
}

/// 2D convolution over `[N, C, H, W]` inputs.
///
/// Weight shape: `[out_channels, in_channels / groups, kH, kW]`, optional bias
/// shape: `[out_channels]`. State-dict keys: `weight`, and `bias` when the
/// layer has one.
#[derive(Clone, Module)]
#[module(crate = "crate")]
pub struct Conv2d {
    pub weight: Tensor,
    #[module(optional)]
    pub bias: Option<Tensor>,
    pub stride: (usize, usize),
    pub padding: ((isize, isize), (isize, isize)),
    pub dilation: (usize, usize),
    pub groups: usize,
    /// Accumulate and emit the sum in this dtype instead of the operands' own;
    /// see [`Self::with_acc_dtype`].
    #[module(skip)]
    pub acc_dtype: Option<DType>,
}

impl Conv2d {
    /// Create a Conv2d from existing tensors, with unit stride and dilation,
    /// no padding and one group.
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias, stride: (1, 1), padding: ((0, 0), (0, 0)), dilation: (1, 1), groups: 1, acc_dtype: None }
    }

    #[track_caller]
    pub fn with_dims(
        in_channels: usize,
        out_channels: usize,
        kernel: (usize, usize),
        bias: bool,
        dtype: DType,
    ) -> Self {
        origin_call!("Conv2d::with_dims");
        let (weight, bias) = conv_params(&[out_channels, in_channels, kernel.0, kernel.1], out_channels, bias, dtype);
        Self { bias, ..Self::new(weight, None) }
    }

    pub fn with_stride(mut self, stride: (usize, usize)) -> Self {
        self.stride = stride;
        self
    }

    /// Per-axis `(before, after)` padding, outermost spatial axis first.
    /// Negative values crop, as in [`Tensor::conv2d`].
    pub fn with_padding(mut self, padding: ((isize, isize), (isize, isize))) -> Self {
        self.padding = padding;
        self
    }

    pub fn with_dilation(mut self, dilation: (usize, usize)) -> Self {
        self.dilation = dilation;
        self
    }

    pub fn with_groups(mut self, groups: usize) -> Self {
        self.groups = groups;
        self
    }

    /// Accumulate in `dtype` and hand the sum on at that width, so a half-width
    /// convolution can still emit full-width values (a box regression's pixel
    /// distances, say) without rounding them through its operand dtype.
    pub fn with_acc_dtype(mut self, dtype: DType) -> Self {
        self.acc_dtype = Some(dtype);
        self
    }
}

impl Layer for Conv2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.conv2d()
            .weight(&self.weight)
            .maybe_bias(self.bias.as_ref())
            .groups(self.groups)
            .stride(&[self.stride.0, self.stride.1])
            .dilation(&[self.dilation.0, self.dilation.1])
            .padding(&[self.padding.0, self.padding.1])
            .maybe_acc_dtype(self.acc_dtype.clone())
            .call()
    }
}

/// Transposed 2D convolution ("deconvolution") over `[N, C, H, W]` inputs.
///
/// Weight shape: `[in_channels, out_channels / groups, kH, kW]` — note the
/// leading axis is the *input* channel count, as in PyTorch. State-dict keys:
/// `weight`, and `bias` when the layer has one.
#[derive(Clone, Module)]
#[module(crate = "crate")]
pub struct ConvTranspose2d {
    pub weight: Tensor,
    #[module(optional)]
    pub bias: Option<Tensor>,
    pub stride: (usize, usize),
    pub padding: ((isize, isize), (isize, isize)),
    pub output_padding: (usize, usize),
    pub dilation: (usize, usize),
    pub groups: usize,
}

impl ConvTranspose2d {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self {
            weight,
            bias,
            stride: (1, 1),
            padding: ((0, 0), (0, 0)),
            output_padding: (0, 0),
            dilation: (1, 1),
            groups: 1,
        }
    }

    #[track_caller]
    pub fn with_dims(
        in_channels: usize,
        out_channels: usize,
        kernel: (usize, usize),
        bias: bool,
        dtype: DType,
    ) -> Self {
        origin_call!("ConvTranspose2d::with_dims");
        let (weight, bias) = conv_params(&[in_channels, out_channels, kernel.0, kernel.1], out_channels, bias, dtype);
        Self { bias, ..Self::new(weight, None) }
    }

    pub fn with_stride(mut self, stride: (usize, usize)) -> Self {
        self.stride = stride;
        self
    }

    pub fn with_padding(mut self, padding: ((isize, isize), (isize, isize))) -> Self {
        self.padding = padding;
        self
    }

    pub fn with_output_padding(mut self, output_padding: (usize, usize)) -> Self {
        self.output_padding = output_padding;
        self
    }

    pub fn with_dilation(mut self, dilation: (usize, usize)) -> Self {
        self.dilation = dilation;
        self
    }

    pub fn with_groups(mut self, groups: usize) -> Self {
        self.groups = groups;
        self
    }
}

impl Layer for ConvTranspose2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.conv_transpose2d()
            .weight(&self.weight)
            .maybe_bias(self.bias.as_ref())
            .groups(self.groups)
            .stride(&[self.stride.0, self.stride.1])
            .dilation(&[self.dilation.0, self.dilation.1])
            .padding(&[self.padding.0, self.padding.1])
            .output_padding(&[self.output_padding.0, self.output_padding.1])
            .call()
    }
}

/// Inference-time batch normalization over the channel axis of `[N, C, ...]`.
///
/// Normalizes with the stored running statistics, never with batch ones.
/// State-dict keys: `weight`, `bias`, `running_mean`, `running_var` — PyTorch's
/// names, so a `torch.nn.BatchNorm2d` dict loads unchanged (`num_batches_tracked`
/// carries no information here and is ignored).
#[derive(Clone, Module)]
#[module(crate = "crate")]
pub struct BatchNorm2d {
    pub weight: Tensor,
    pub bias: Tensor,
    pub running_mean: Tensor,
    pub running_var: Tensor,
    pub eps: f64,
}

impl BatchNorm2d {
    pub fn new(weight: Tensor, bias: Tensor, running_mean: Tensor, running_var: Tensor, eps: f64) -> Self {
        Self { weight, bias, running_mean, running_var, eps }
    }

    /// The identity: unit scale and variance, zero shift and mean.
    pub fn with_dims(channels: usize, eps: f64, dtype: DType) -> Self {
        let ones = || Tensor::ones(&[channels], dtype.clone()).contiguous();
        let zeros = || Tensor::zeros(&[channels], dtype.clone()).contiguous();
        Self::new(ones(), zeros(), zeros(), ones(), eps)
    }
}

impl Layer for BatchNorm2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.batchnorm()
            .scale(&self.weight)
            .bias(&self.bias)
            .mean(&self.running_mean)
            .var(&self.running_var)
            .eps(self.eps)
            .call()
    }
}

/// Layer normalization over the axes `[axis..ndim)`, with an affine epilogue.
///
/// State-dict keys: `weight`, and `bias` when the layer has one.
#[derive(Clone, Module)]
#[module(crate = "crate")]
pub struct LayerNorm {
    pub weight: Tensor,
    #[module(optional)]
    pub bias: Option<Tensor>,
    pub eps: f64,
    pub axis: isize,
}

impl LayerNorm {
    /// Normalize over the last axis with the given `eps`.
    pub fn new(weight: Tensor, bias: Option<Tensor>, eps: f64) -> Self {
        Self { weight, bias, eps, axis: -1 }
    }

    /// The identity affine: unit scale, zero shift.
    pub fn with_dims(size: usize, bias: bool, eps: f64, dtype: DType) -> Self {
        let weight = Tensor::ones(&[size], dtype.clone()).contiguous();
        Self::new(weight, bias.then(|| Tensor::zeros(&[size], dtype).contiguous()), eps)
    }

    /// Normalize over `[axis..ndim)` instead of over the last axis alone.
    pub fn with_axis(mut self, axis: isize) -> Self {
        self.axis = axis;
        self
    }
}

impl Layer for LayerNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.layernorm_with().axis(self.axis).eps(self.eps).weight(&self.weight).maybe_bias(self.bias.as_ref()).call()
    }
}

/// Root-mean-square normalization over the last axis, scaled by `weight`.
///
/// State-dict key: `weight`.
#[derive(Clone, Module)]
#[module(crate = "crate")]
pub struct RmsNorm {
    pub weight: Tensor,
    pub eps: f64,
}

impl RmsNorm {
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    /// The identity scale.
    pub fn with_dims(size: usize, eps: f64, dtype: DType) -> Self {
        Self::new(Tensor::ones(&[size], dtype).contiguous(), eps)
    }
}

impl Layer for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.rms_norm_with().eps(self.eps).weight(&self.weight).call()
    }
}

/// Embedding table lookup: `forward` takes the *indices*, not activations, and
/// returns `weight[indices]` with shape `[*indices.shape, embed_dim]`.
///
/// State-dict key: `weight`.
#[derive(Clone, Module)]
#[module(crate = "crate")]
pub struct Embedding {
    pub weight: Tensor,
}

impl Embedding {
    pub fn new(weight: Tensor) -> Self {
        Self { weight }
    }

    #[track_caller]
    pub fn with_dims(vocab_size: usize, embed_dim: usize, dtype: DType) -> Self {
        origin_call!("Embedding::with_dims");
        let weight = Tensor::kaiming_uniform_with_dtype(&[vocab_size, embed_dim], 0.0, dtype)
            .expect("non-empty shape")
            .contiguous();
        Self::new(weight)
    }
}

impl Layer for Embedding {
    fn forward(&self, indices: &Tensor) -> Result<Tensor> {
        self.weight.embedding(indices)
    }
}
