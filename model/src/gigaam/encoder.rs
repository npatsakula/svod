use svod_dtype::DType;
use svod_ir::SInt;
use svod_ir::origin::OriginScope;
use svod_tensor::Tensor;
use svod_tensor::nn::{Layer, LayerNorm, Linear, Module, StateDict, get_tensor, prefixed};

use crate::init::{Bias, fan_in_uniform, layer_norm, ones, zeros};
use crate::state::{scoped, scoped_index};

use super::{ConvNormType, GigaAmConfig, SubsamplingMode, subsampled_len};

/// PyTorch's `BatchNorm1d` default epsilon. The checkpoint carries no eps, and
/// the affine is folded into the depthwise conv at load, so this is the only
/// place it appears.
const BN_EPS: f64 = 1e-5;

/// RoPE cos/sin cache, `[max_encoder_frames, 1, 1, d_k/2]`. Upstream GigaAM
/// passes `pos_emb_max_len` as both cache length and RoPE base, and rotates a
/// position-major `[T, B, H, d_k]` tensor — hence the permute off
/// [`Tensor::rope_table`]'s `[1, 1, L, d_k/2]`. Realized here, once: left
/// lazy, every layer's RoPE kernel recomputes the pow/sin/cos of its slice.
fn build_rope_cache(config: &GigaAmConfig) -> svod_tensor::error::Result<(Tensor, Tensor)> {
    let d_k = config.d_model / config.n_heads;
    let (cos, sin) =
        Tensor::rope_table(config.max_encoder_frames as f64, config.max_encoder_frames, d_k, DType::Float32)?;
    let position_major = |t: Tensor| t.try_permute(&[2, 0, 1, 3]).expect("4-D rope table").contiguous();
    let (cos, sin) = (position_major(cos), position_major(sin));
    Tensor::realize_batch([&cos, &sin])?;
    Ok((cos, sin))
}

type Result<T> = super::Result<T>;

/// `nn::Linear` with an optional dynamic-quantization scale. The scale's
/// presence is a property of the *weight dtype*, not of the state dict, so the
/// pair is loaded together by the owner's `Module` impl rather than derived.
fn linear(x: &Tensor, weight: &Tensor, bias: &Tensor, weight_scale: Option<&Tensor>) -> Result<Tensor> {
    match weight_scale {
        Some(scale) => Ok(x.dynamic_quantized_linear().weight(weight).weight_scale(scale).bias(bias).call()?),
        None => Ok(x.linear().weight(weight).bias(bias).call()?),
    }
}

/// Every `Linear` in this encoder carries PyTorch's own fan-in uniform bias.
fn plain_linear(inp: usize, out: usize) -> Linear {
    crate::init::linear(inp, out, Bias::FanIn, DType::Float32)
}

/// The bias every `Linear` in this encoder carries.
fn bias_of(linear: &Linear) -> &Tensor {
    linear.bias.as_ref().expect("GigaAM linears are always biased")
}

// ---------------------------------------------------------------------------
// FeedForward
// ---------------------------------------------------------------------------

/// Conformer FFN: LayerNorm -> Linear(d->4d) -> SiLU -> Linear(4d->d).
///
/// Does NOT apply residual or 0.5 scaling — caller handles that.
#[derive(Clone, Module)]
pub struct FeedForward {
    pub norm: LayerNorm,
    pub linear1: Linear,
    pub linear2: Linear,
    #[module(key = "linear1.weight_scale", optional = "self.linear1.weight.dtype().is_signed()")]
    pub linear1_scale: Option<Tensor>,
    #[module(key = "linear2.weight_scale", optional = "self.linear2.weight.dtype().is_signed()")]
    pub linear2_scale: Option<Tensor>,
}

impl FeedForward {
    pub fn empty(config: &GigaAmConfig) -> Self {
        let (d, d_ff) = (config.d_model, config.d_ff);
        Self {
            norm: layer_norm(d, DType::Float32),
            linear1: plain_linear(d, d_ff),
            linear2: plain_linear(d_ff, d),
            linear1_scale: None,
            linear2_scale: None,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // The two-linear FFN lowers to GEMM1+silu → h → GEMM2 (two reduces force `h`
        // to realize between them), which the generic optimizer fuses + (with BEAM)
        // tunes as well as a hand kernel — so the FFN stays plain graph ops.
        let y = scoped("norm", || self.norm.forward(x))?;
        let y = linear(&y, &self.linear1.weight, bias_of(&self.linear1), self.linear1_scale.as_ref())?;
        let y = y.silu()?;
        linear(&y, &self.linear2.weight, bias_of(&self.linear2), self.linear2_scale.as_ref())
    }
}

// ---------------------------------------------------------------------------
// MultiHeadSelfAttention
// ---------------------------------------------------------------------------

/// Multi-head self-attention with rotary position embeddings.
///
/// Projections stay as bare `[out, in]` tensors rather than `nn::Linear`: the
/// checkpoint keys the pair `q_proj` / `q_bias`, not `q.weight` / `q.bias`.
#[derive(Clone, Module)]
pub struct MultiHeadSelfAttention {
    pub norm: LayerNorm,
    pub q_proj: Tensor,
    pub q_bias: Tensor,
    pub k_proj: Tensor,
    pub k_bias: Tensor,
    pub v_proj: Tensor,
    pub v_bias: Tensor,
    pub out_proj: Tensor,
    pub out_bias: Tensor,
    #[module(optional = "self.q_proj.dtype().is_signed()")]
    pub q_weight_scale: Option<Tensor>,
    #[module(optional = "self.k_proj.dtype().is_signed()")]
    pub k_weight_scale: Option<Tensor>,
    #[module(optional = "self.v_proj.dtype().is_signed()")]
    pub v_weight_scale: Option<Tensor>,
    #[module(optional = "self.out_proj.dtype().is_signed()")]
    pub out_weight_scale: Option<Tensor>,
    pub n_heads: usize,
    pub d_model: usize,
}

impl MultiHeadSelfAttention {
    pub fn empty(config: &GigaAmConfig) -> Self {
        let d = config.d_model;
        let proj = || fan_in_uniform(&[d, d], d, DType::Float32);
        let bias = || fan_in_uniform(&[d], d, DType::Float32);
        Self {
            norm: layer_norm(d, DType::Float32),
            q_proj: proj(),
            q_bias: bias(),
            k_proj: proj(),
            k_bias: bias(),
            v_proj: proj(),
            v_bias: bias(),
            out_proj: proj(),
            out_bias: bias(),
            q_weight_scale: None,
            k_weight_scale: None,
            v_weight_scale: None,
            out_weight_scale: None,
            n_heads: config.n_heads,
            d_model: d,
        }
    }

    /// `key_lens`, when present, is a realized `[B]` `i32` tensor of valid
    /// (unpadded) key positions per batch — keys at index `>= key_lens[b]` are
    /// masked (see [`crate::attention::attend`]).
    pub fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, key_lens: Option<&Tensor>) -> Result<Tensor> {
        let shape = x.shape()?;
        let b = shape[0].clone();
        let t = shape[1].clone();
        let d_model = self.d_model;
        let d_k = d_model / self.n_heads;
        let h = self.n_heads;

        let y = scoped("norm", || self.norm.forward(x))?;

        // RoPE expects [T, B, H, d_k] (PyTorch ordering). Rotate once, then
        // materialise back as [B, T, d_model] so the Q/K projections share
        // a single rotated buffer.
        let y_heads = y.try_transpose(0, 1)?.try_reshape([t.clone(), b.clone(), SInt::Const(h), SInt::Const(d_k)])?;
        // The table is shared by every layer, so its cast is built outside the
        // layer's origin scope and hash-conses across layers.
        let rope_dtype = y_heads.dtype();
        let (cos, sin) = {
            let _shared = OriginScope::suspend();
            (cos.cast(rope_dtype.clone()), sin.cast(rope_dtype))
        };
        let qk_input = y_heads
            .apply_rotary_emb(&cos, &sin, false)?
            .try_reshape([t.clone(), b.clone(), SInt::Const(d_model)])?
            .try_transpose(0, 1)?
            .contiguous();

        let q = linear(&qk_input, &self.q_proj, &self.q_bias, self.q_weight_scale.as_ref())?;
        let k = linear(&qk_input, &self.k_proj, &self.k_bias, self.k_weight_scale.as_ref())?;
        let v = linear(&y, &self.v_proj, &self.v_bias, self.v_weight_scale.as_ref())?;

        // Head-split into `[B, T, H, d_k]` — the layout `flash_attention_with`
        // consumes directly (seq second, head third, head_dim last). Not
        // `Tensor::split_heads`, which lands head-major `[B, H, T, d_k]`: the
        // hand kernel and the SDPA fallback both take/return `[B, T, H, d_k]`.
        let split = |p: Tensor| -> Result<Tensor> {
            Ok(p.try_reshape([b.clone(), t.clone(), SInt::Const(h), SInt::Const(d_k)])?)
        };
        let (q, k, v) = (split(q)?, split(k)?, split(v)?);

        let attn = crate::attention::attend(&q, &k, &v, false, key_lens)?;
        // Head-merge is a plain reshape here: the attention output is already
        // seq-major, so there is no transpose to undo.
        let out = attn.try_reshape([b, t, SInt::Const(d_model)])?;
        linear(&out, &self.out_proj, &self.out_bias, self.out_weight_scale.as_ref())
    }
}

// ---------------------------------------------------------------------------
// ConvModule
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub enum ConvNorm {
    LayerNorm(LayerNorm),
    BatchNorm {
        scale: Tensor,
        bias: Tensor,
        mean: Tensor,
        var: Tensor,
    },
    /// Inference-time BatchNorm fold: the affine collapsed into the depthwise
    /// conv at load (`w *= s`, `b = b*s + bias - mean*s` with
    /// `s = scale / sqrt(var + eps)`), so the norm op vanishes from the graph.
    Folded,
}

/// Conformer convolution module:
/// LayerNorm -> Conv1d(d,2d,k=1) -> GLU -> DepthwiseConv1d -> Norm -> SiLU -> Conv1d(d,d,k=1)
#[derive(Clone)]
pub struct ConvModule {
    pub norm: LayerNorm,
    pub pw1_weight: Tensor,
    pub pw1_bias: Tensor,
    pub dw_weight: Tensor,
    pub dw_bias: Tensor,
    pub conv_norm: ConvNorm,
    pub pw2_weight: Tensor,
    pub pw2_bias: Tensor,
    d_model: usize,
    conv_kernel: usize,
}

impl ConvModule {
    pub fn empty(config: &GigaAmConfig) -> Self {
        let (d, k) = (config.d_model, config.conv_kernel);
        let conv_norm = match &config.conv_norm_type {
            ConvNormType::LayerNorm => ConvNorm::LayerNorm(layer_norm(d, DType::Float32)),
            ConvNormType::BatchNorm => ConvNorm::BatchNorm {
                scale: ones(&[d], DType::Float32),
                bias: zeros(&[d], DType::Float32),
                mean: zeros(&[d], DType::Float32),
                var: ones(&[d], DType::Float32),
            },
        };
        Self {
            norm: layer_norm(d, DType::Float32),
            pw1_weight: fan_in_uniform(&[2 * d, d, 1], d, DType::Float32),
            pw1_bias: fan_in_uniform(&[2 * d], d, DType::Float32),
            dw_weight: fan_in_uniform(&[d, 1, k], k, DType::Float32),
            dw_bias: fan_in_uniform(&[d], k, DType::Float32),
            conv_norm,
            pw2_weight: fan_in_uniform(&[d, d, 1], d, DType::Float32),
            pw2_bias: fan_in_uniform(&[d], d, DType::Float32),
            d_model: d,
            conv_kernel: k,
        }
    }

    /// `pad_valid` (`[B, T]` bool, true = real frame) zeroes padded rows before
    /// the depthwise conv so they cannot leak into valid ones.
    pub fn forward(&self, x: &Tensor, pad_valid: Option<&Tensor>) -> Result<Tensor> {
        let activation_dtype = x.dtype();
        let y = scoped("norm", || self.norm.forward(x))?;

        let y = y.try_transpose(-1, -2)?;

        let mut y = y.conv1d().weight(&self.pw1_weight).bias(&self.pw1_bias).call()?.glu(1)?;

        if let Some(valid) = pad_valid {
            y = y.where_(&valid.try_unsqueeze(1)?, 0.0)?;
        }

        let pad = ((self.conv_kernel - 1) / 2) as isize;
        let y = y.conv1d().weight(&self.dw_weight).bias(&self.dw_bias).groups(self.d_model).padding(pad).call()?;

        let y = match &self.conv_norm {
            ConvNorm::LayerNorm(ln) => {
                // Materialized in the activation dtype: otherwise the norm's f32
                // cast fuses into the depthwise conv, which then stores f32.
                let y = y.try_transpose(-1, -2)?.contiguous();
                let y = scoped("conv_norm", || ln.forward(&y))?;
                y.try_transpose(-1, -2)?
            }
            ConvNorm::BatchNorm { scale, bias, mean, var } => {
                y.batchnorm().scale(scale).bias(bias).mean(mean).var(var).eps(BN_EPS).call()?
            }
            ConvNorm::Folded => y,
        };
        // BN params are stored fp32; broadcasting promotes the norm output. Re-cast
        // to the activation dtype so SiLU/pw2 stay in the right precision, matching
        // Python's BatchNorm1d dtype semantics. No-op when types match.
        let y = if y.dtype() != activation_dtype { y.cast(activation_dtype) } else { y };

        let y = y.silu()?.conv1d().weight(&self.pw2_weight).bias(&self.pw2_bias).call()?;

        Ok(y.try_transpose(-1, -2)?)
    }

    /// Fold BatchNorm into the depthwise conv: with `s = scale / sqrt(var + eps)`,
    /// `y_bn = conv(x)·s + (bias - mean·s)` — a per-channel scale on the conv
    /// weight rows plus a corrected bias. Removes the norm op from the graph.
    fn fold_batchnorm(
        &mut self,
        scale: &Tensor,
        bias: &Tensor,
        mean: &Tensor,
        var: &Tensor,
    ) -> svod_tensor::error::Result<()> {
        let s = scale.try_div(&var.try_add(BN_EPS)?.try_sqrt()?)?;
        self.dw_weight = self.dw_weight.try_mul(&s.try_reshape([self.d_model as isize, 1, 1])?)?;
        self.dw_bias = self.dw_bias.try_mul(&s)?.try_add(bias)?.try_sub(&mean.try_mul(&s)?)?;
        self.conv_norm = ConvNorm::Folded;
        Ok(())
    }
}

/// Hand-written: the load *changes the variant* (`BatchNorm` → `Folded`), which
/// no derive can express.
impl Module for ConvModule {
    fn write_state(&self, prefix: &str, out: &mut StateDict) {
        for (name, tensor) in [
            ("pw1_weight", &self.pw1_weight),
            ("pw1_bias", &self.pw1_bias),
            ("dw_weight", &self.dw_weight),
            ("dw_bias", &self.dw_bias),
            ("pw2_weight", &self.pw2_weight),
            ("pw2_bias", &self.pw2_bias),
        ] {
            out.insert(prefixed(prefix, name), tensor.clone());
        }
        self.norm.write_state(&prefixed(prefix, "norm"), out);
        match &self.conv_norm {
            ConvNorm::LayerNorm(ln) => ln.write_state(&prefixed(prefix, "conv_norm"), out),
            ConvNorm::BatchNorm { scale, bias, mean, var } => {
                for (name, t) in [("bn_scale", scale), ("bn_bias", bias), ("bn_mean", mean), ("bn_var", var)] {
                    out.insert(prefixed(prefix, name), t.clone());
                }
            }
            // Folded weights live in dw_weight/dw_bias; the BN params are gone.
            ConvNorm::Folded => {}
        }
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> svod_tensor::error::Result<()> {
        for (name, field) in [
            ("pw1_weight", &mut self.pw1_weight),
            ("pw1_bias", &mut self.pw1_bias),
            ("dw_weight", &mut self.dw_weight),
            ("dw_bias", &mut self.dw_bias),
            ("pw2_weight", &mut self.pw2_weight),
            ("pw2_bias", &mut self.pw2_bias),
        ] {
            *field = get_tensor(sd, &prefixed(prefix, name))?;
        }
        self.norm.load_state_dict(sd, &prefixed(prefix, "norm"))?;
        match &mut self.conv_norm {
            ConvNorm::LayerNorm(ln) => ln.load_state_dict(sd, &prefixed(prefix, "conv_norm"))?,
            // Folded round-trips (e.g. `cast_weights`) carry no BN keys — the
            // affine already lives in dw_weight/dw_bias.
            ConvNorm::Folded if !sd.contains_key(&prefixed(prefix, "bn_scale")) => {}
            ConvNorm::BatchNorm { .. } | ConvNorm::Folded => {
                let param = |name: &str| get_tensor(sd, &prefixed(prefix, name));
                let (scale, bias) = (param("bn_scale")?, param("bn_bias")?);
                let (mean, var) = (param("bn_mean")?, param("bn_var")?);
                self.fold_batchnorm(&scale, &bias, &mean, &var)?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// StridingSubsampling
// ---------------------------------------------------------------------------

/// Striding subsampling: 2x (Conv stride-2 + ReLU), optionally followed by Linear.
///
/// Supports two modes:
/// - **conv1d**: `Conv1d(n_mels→d, k, stride=2)` x2, no linear projection.
/// - **conv2d**: `Conv2d(1→d, 3x3, stride=2)` x2 + `Linear(d * n_mels/4, d)`.
///
/// Input: `[B, T, n_mels]` -> Output: `[B, T/4, d_model]`.
#[derive(Clone, Module)]
pub struct StridingSubsampling {
    pub conv1_weight: Tensor,
    pub conv1_bias: Tensor,
    pub conv2_weight: Tensor,
    pub conv2_bias: Tensor,
    #[module(optional = "self.is_conv2d()")]
    pub linear_weight: Option<Tensor>,
    #[module(optional = "self.is_conv2d()")]
    pub linear_bias: Option<Tensor>,
    n_mels: usize,
    d_model: usize,
    #[module(skip)]
    mode: SubsamplingMode,
    kernel_size: usize,
}

impl StridingSubsampling {
    pub fn empty(config: &GigaAmConfig) -> Self {
        let d = config.d_model;
        let k = config.subs_kernel_size;
        match &config.subsampling_mode {
            SubsamplingMode::Conv1d => {
                let fan_in1 = config.n_mels * k;
                let fan_in2 = d * k;
                Self {
                    conv1_weight: fan_in_uniform(&[d, config.n_mels, k], fan_in1, DType::Float32),
                    conv1_bias: fan_in_uniform(&[d], fan_in1, DType::Float32),
                    conv2_weight: fan_in_uniform(&[d, d, k], fan_in2, DType::Float32),
                    conv2_bias: fan_in_uniform(&[d], fan_in2, DType::Float32),
                    linear_weight: None,
                    linear_bias: None,
                    n_mels: config.n_mels,
                    d_model: d,
                    mode: SubsamplingMode::Conv1d,
                    kernel_size: k,
                }
            }
            SubsamplingMode::Conv2d => {
                let fan_in1 = 9;
                let fan_in2 = 9 * d;
                let linear_in = d * (config.n_mels / 4);
                Self {
                    conv1_weight: fan_in_uniform(&[d, 1, 3, 3], fan_in1, DType::Float32),
                    conv1_bias: fan_in_uniform(&[d], fan_in1, DType::Float32),
                    conv2_weight: fan_in_uniform(&[d, d, 3, 3], fan_in2, DType::Float32),
                    conv2_bias: fan_in_uniform(&[d], fan_in2, DType::Float32),
                    linear_weight: Some(fan_in_uniform(&[d, linear_in], linear_in, DType::Float32)),
                    linear_bias: Some(fan_in_uniform(&[d], linear_in, DType::Float32)),
                    n_mels: config.n_mels,
                    d_model: d,
                    mode: SubsamplingMode::Conv2d,
                    kernel_size: 3,
                }
            }
        }
    }

    /// Whether the linear projection exists — the `Module` derive's predicate
    /// for the two optional parameters.
    fn is_conv2d(&self) -> bool {
        matches!(self.mode, SubsamplingMode::Conv2d)
    }

    pub fn output_length(&self, input_length: usize) -> usize {
        subsampled_len(self.kernel_size, input_length)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match &self.mode {
            SubsamplingMode::Conv1d => self.forward_conv1d(x),
            SubsamplingMode::Conv2d => self.forward_conv2d(x),
        }
    }

    fn forward_conv1d(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.try_transpose(-1, -2)?;

        let pad = (self.kernel_size / 2) as isize;
        let stage = |x: &Tensor, weight: &Tensor, bias: &Tensor| -> Result<Tensor> {
            Ok(x.conv1d().weight(weight).bias(bias).stride(2).padding(pad).call()?.relu()?)
        };
        let x = stage(&x, &self.conv1_weight, &self.conv1_bias)?;
        let x = stage(&x, &self.conv2_weight, &self.conv2_bias)?;

        Ok(x.try_transpose(-1, -2)?)
    }

    fn forward_conv2d(&self, x: &Tensor) -> Result<Tensor> {
        let b = x.dim(0)?;

        let x = x.try_unsqueeze(1)?;

        let stage = |x: &Tensor, weight: &Tensor, bias: &Tensor| -> Result<Tensor> {
            Ok(x.conv2d().weight(weight).bias(bias).stride(&[2, 2]).padding(&[(1, 1), (1, 1)]).call()?.relu()?)
        };
        let x = stage(&x, &self.conv1_weight, &self.conv1_bias)?;
        let x = stage(&x, &self.conv2_weight, &self.conv2_bias)?;

        let x = x.try_permute(&[0, 2, 1, 3])?;
        let x = x.try_reshape([b, SInt::Infer, SInt::Const(self.d_model * self.n_mels / 4)])?;

        let lw = self.linear_weight.as_ref().expect("conv2d mode requires linear_weight");
        let lb = self.linear_bias.as_ref().expect("conv2d mode requires linear_bias");
        Ok(x.linear().weight(lw).bias(lb).call()?)
    }
}

// ---------------------------------------------------------------------------
// ConformerLayer
// ---------------------------------------------------------------------------

/// One Conformer layer (Macaron-style):
/// FFN1(x0.5) + MHSA + Conv + FFN2(x0.5) + LayerNorm
#[derive(Clone, Module)]
pub struct ConformerLayer {
    pub ffn1: FeedForward,
    pub mhsa: MultiHeadSelfAttention,
    pub conv: ConvModule,
    pub ffn2: FeedForward,
    pub final_norm: LayerNorm,
}

impl ConformerLayer {
    pub fn empty(config: &GigaAmConfig) -> Self {
        Self {
            ffn1: FeedForward::empty(config),
            mhsa: MultiHeadSelfAttention::empty(config),
            conv: ConvModule::empty(config),
            ffn2: FeedForward::empty(config),
            final_norm: layer_norm(config.d_model, DType::Float32),
        }
    }

    /// `key_lens` is the optional per-batch valid encoder-frame count (`[B]`
    /// `i32`) used as a key-only padding mask in MHSA (see
    /// [`MultiHeadSelfAttention::forward`]). `pad_valid` independently zeros
    /// padded rows in the conv module.
    pub fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        key_lens: Option<&Tensor>,
        pad_valid: Option<&Tensor>,
    ) -> Result<Tensor> {
        // FFN1 half-step
        let ffn1 = scoped("ffn1", || self.ffn1.forward(x))?;
        let x = x.try_add(&ffn1.try_mul(0.5)?)?;

        // MHSA
        let mhsa = scoped("mhsa", || self.mhsa.forward(&x, cos, sin, key_lens))?;
        let x = x.try_add(&mhsa)?;

        // Convolution
        let conv = scoped("conv", || self.conv.forward(&x, pad_valid))?;
        let x = x.try_add(&conv)?;

        // FFN2 half-step. Materialized in the activation dtype: otherwise the
        // final norm's f32 cast fuses into the FFN2 GEMM, which then stores f32.
        let ffn2 = scoped("ffn2", || self.ffn2.forward(&x))?;
        let x = x.try_add(&ffn2.try_mul(0.5)?)?.contiguous();

        // Final layer norm
        Ok(scoped("final_norm", || self.final_norm.forward(&x))?)
    }
}

// ---------------------------------------------------------------------------
// Encoder — audio preprocessor + Conformer backbone
// ---------------------------------------------------------------------------

/// Audio preprocessor + Conformer encoder. Shared by both heads of
/// [`crate::gigaam::GigaAm`] (`Head::Ctc` and `Head::Rnnt` layer different
/// heads on top of the same encoder). Encoder-only path: `forward` for
/// single-batch, `forward_batch` for batched JIT execution.
#[derive(Clone)]
pub struct Encoder {
    pub subsampling: StridingSubsampling,
    pub layers: Vec<ConformerLayer>,
    pub cos_cache: Tensor,
    pub sin_cache: Tensor,
    pub d_model: usize,
    pub n_heads: usize,
    pub max_encoder_frames: usize,
}

impl Encoder {
    pub fn with_random_weights(config: &GigaAmConfig) -> Self {
        let (cos_cache, sin_cache) =
            build_rope_cache(config).expect("validated config: even head dim, non-empty cache");
        let subsampling = StridingSubsampling::empty(config);
        let layers = (0..config.n_layers).map(|_| ConformerLayer::empty(config)).collect();
        Self {
            subsampling,
            layers,
            cos_cache,
            sin_cache,
            d_model: config.d_model,
            n_heads: config.n_heads,
            max_encoder_frames: config.max_encoder_frames,
        }
    }

    /// dtype the encoder operates in. Read off the first subsampling
    /// conv weight (the model's compute dtype is determined by the
    /// weights it was loaded with). Falls back to f32 when the weight
    /// isn't itself a float type — should never happen in practice but
    /// avoids producing an integer dtype here.
    pub fn input_dtype(&self) -> DType {
        let dtype = self.subsampling.conv1_weight.dtype();
        if dtype.is_float() { dtype } else { DType::Float32 }
    }

    /// Shrink the precomputed RoPE cache to `[t, 1, 1, d_k/2]` so both
    /// single-batch and batched encoder forwards consume the same shape.
    fn slice_rope(&self, t: SInt) -> Result<(Tensor, Tensor)> {
        let d_half = self.d_model / self.n_heads / 2;
        let shrink = [
            (SInt::Const(0), t),
            (SInt::Const(0), SInt::Const(1)),
            (SInt::Const(0), SInt::Const(1)),
            (SInt::Const(0), SInt::Const(d_half)),
        ];
        let cos = self.cos_cache.try_shrink(shrink.clone())?;
        let sin = self.sin_cache.try_shrink(shrink)?;
        Ok((cos, sin))
    }

    /// Encoder pass on a single mel batch with no padding mask.
    /// Input: tensor `[B, n_mels, T]`. Output: lazy tensor `[B, d_model, T/4]`.
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let x = mel.try_transpose(-1, -2)?;
        let x = x.cast(self.input_dtype());
        let x = scoped("subsampling", || self.subsampling.forward(&x))?;

        let (cos, sin) = self.slice_rope(x.dim(1)?)?;

        // Single-batch path realizes `T` at the actual subsampled frame count of
        // this mel (no bucketing / no padding), so every key position is valid:
        // `key_lens = None` lets attention see the whole sequence.
        let mut x = x;
        for (index, layer) in self.layers.iter().enumerate() {
            x = scoped_index("layers", index, || layer.forward(&x, &cos, &sin, None, None))?;
        }

        Ok(x.try_transpose(-1, -2)?)
    }

    /// Batched encoder path over the full, constant-shaped mel buffer.
    /// Input: `mel` `[B, n_mels, T_mel]`, `lengths` `[B]` valid lengths in mel frames.
    /// Output: `[B, d_model, T_sub]`.
    ///
    /// `B` and `T_mel` are read directly off `mel`'s shape — the JIT realizes the
    /// input buffers at the bucket's max shape (`max_batch_size × max_t_mel`), so
    /// every derived dim is `SInt::Const`. That exact divisibility is what lets the
    /// schedule heuristics fire MFMA tilings instead of the slow symbolic fallback.
    /// Inactive lanes (`lengths[b] == 0`) subsample to `lengths_sub == 0`, so
    /// `pad_valid` is all-false for them and the validity masks zero their output;
    /// the caller never reads those lanes.
    pub fn forward_batch(&self, mel: &Tensor, lengths: &Tensor) -> Result<Tensor> {
        // Two stride-2 stages: `len = (len + 1) / 2` each, in integer arithmetic.
        let mut lengths_sub = lengths.cast(DType::Int32);
        for _ in 0..2 {
            lengths_sub = lengths_sub.try_add(1i32)?.try_div(2i32)?;
        }

        let x = mel.try_transpose(-1, -2)?;
        let x = x.cast(self.input_dtype());
        let x = scoped("subsampling", || self.subsampling.forward(&x))?;

        let t_sub = x.dim(1)?;

        // `key_lens` = subsampled valid-frame counts as a realized `[B]` `i32`
        // tensor — attention's key-only padding mask. The same lengths drive the
        // conv module's per-row validity mask.
        let key_lens = lengths_sub.try_reshape([mel.dim(0)?])?;
        let pad_valid = Tensor::sequence_mask(&key_lens, x.dim_const(1)?)?;

        let (cos, sin) = self.slice_rope(t_sub)?;

        let mut x = x;
        for (index, layer) in self.layers.iter().enumerate() {
            x = scoped_index("layers", index, || layer.forward(&x, &cos, &sin, Some(&key_lens), Some(&pad_valid)))?;
        }

        Ok(x.try_transpose(-1, -2)?)
    }

    pub fn subsampling_output_length(&self, mel_frames: usize) -> usize {
        self.subsampling.output_length(mel_frames)
    }

    /// Construct an `Encoder` from an already-remapped state dict + config.
    /// Called from the unified [`crate::gigaam::GigaAm::from_state_dict`] loader.
    pub(crate) fn from_state_dict(sd: &StateDict, config: &GigaAmConfig) -> Result<Self> {
        let (cos_cache, sin_cache) = build_rope_cache(config)?;

        let mut subsampling = StridingSubsampling::empty(config);
        subsampling.load_state_dict(sd, "subsampling")?;

        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let mut layer = ConformerLayer::empty(config);
            layer.load_state_dict(sd, &format!("layers.{i}"))?;
            layers.push(layer);
        }

        Ok(Self {
            subsampling,
            layers,
            cos_cache,
            sin_cache,
            d_model: config.d_model,
            n_heads: config.n_heads,
            max_encoder_frames: config.max_encoder_frames,
        })
    }
}
