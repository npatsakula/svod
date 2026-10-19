use svod_dtype::DType;
use svod_ir::SInt;
use svod_ir::origin::OriginScope;
use svod_tensor::Tensor;
use svod_tensor::nn::{Layer, LayerNorm, Linear, Module, Pad1d, StateDict, get_tensor, prefixed};
use svod_tk::ops::{self, Act, Attn, Cache, Chunk, KeyMask};

use crate::init::{Bias, fan_in_uniform, layer_norm, ones, zeros};
use crate::state::{scoped, scoped_index};

use super::{ConvNormType, GigaAmConfig, StreamingConfig, SubsamplingMode, subsampled_len};

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

/// `scale·act(x·wᵀ + bias) + residual` as `opts` spells it (the bias is
/// `bias`), through the tile op layer unless the weight carries a
/// dynamic-quantization `weight_scale`. Its presence is a property of the
/// *weight dtype*, not of the state dict, so the pair is loaded together by
/// the owner's `Module` impl rather than derived.
fn linear(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    weight_scale: Option<&Tensor>,
    opts: ops::Linear,
) -> Result<Tensor> {
    let Some(scale) = weight_scale else {
        return Ok(ops::linear(x, weight, ops::Linear { bias: Some(bias), ..opts })?);
    };
    let y = x.dynamic_quantized_linear().weight(weight).weight_scale(scale).bias(bias).call()?;
    let y = match opts.act {
        Act::None => y,
        Act::Silu => y.silu()?,
        Act::Gelu => y.gelu_exact()?,
    };
    let y = match opts.scale {
        Some(s) => y.try_mul(s)?,
        None => y,
    };
    Ok(match opts.residual {
        Some(r) => r.try_add(&y)?,
        None => y,
    })
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

/// Conformer FFN half-step: `x + 0.5·Linear(4d->d)(SiLU(Linear(d->4d)(LayerNorm(x))))`,
/// the scale and residual in the second GEMM's epilogue.
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
        let y = scoped("norm", || self.norm.forward(x))?;
        let up = ops::Linear { act: Act::Silu, ..ops::Linear::default() };
        let y = linear(&y, &self.linear1.weight, bias_of(&self.linear1), self.linear1_scale.as_ref(), up)?;
        let down = ops::Linear { residual: Some(x), scale: Some(0.5), ..ops::Linear::default() };
        linear(&y, &self.linear2.weight, bias_of(&self.linear2), self.linear2_scale.as_ref(), down)
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

    /// The normed input's projections in the attention's sequence-major
    /// `[B, T, H, d_k]`, the queries and keys rotated by `cos`/`sin`
    /// (`[T, 1 | B, 1, d_k/2]`, position-major) before projection as upstream
    /// GigaAM does, the values not.
    fn qkv(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let shape = x.shape()?;
        let b = shape[0].clone();
        let t = shape[1].clone();
        let d_model = self.d_model;
        let d_k = d_model / self.n_heads;
        let h = self.n_heads;

        let y = scoped("norm", || self.norm.forward(x))?.contiguous();

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

        let q = linear(&qk_input, &self.q_proj, &self.q_bias, self.q_weight_scale.as_ref(), ops::Linear::default())?;
        let k = linear(&qk_input, &self.k_proj, &self.k_bias, self.k_weight_scale.as_ref(), ops::Linear::default())?;
        let v = linear(&y, &self.v_proj, &self.v_bias, self.v_weight_scale.as_ref(), ops::Linear::default())?;

        // Head-split into the attention's sequence-major `[B, T, H, d_k]`, not
        // `Tensor::split_heads`, which lands head-major `[B, H, T, d_k]`.
        let split = |p: Tensor| -> Result<Tensor> {
            Ok(p.try_reshape([b.clone(), t.clone(), SInt::Const(h), SInt::Const(d_k)])?)
        };
        Ok((split(q)?, split(k)?, split(v)?))
    }

    /// `x + out_proj(attn)`: the head merge is a plain reshape, the attention
    /// output being sequence-major already.
    fn project_out(&self, x: &Tensor, attn: &Tensor) -> Result<Tensor> {
        let shape = x.shape()?;
        let out = attn.try_reshape([shape[0].clone(), shape[1].clone(), SInt::Const(self.d_model)])?;
        let opts = ops::Linear { residual: Some(x), ..ops::Linear::default() };
        linear(&out, &self.out_proj, &self.out_bias, self.out_weight_scale.as_ref(), opts)
    }

    /// `x + mhsa(x)` under `masks`' key lengths, chunk mask and bias.
    pub fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, masks: &Masks) -> Result<Tensor> {
        let (q, k, v) = self.qkv(x, cos, sin)?;
        let keys = masks.key_lens.map_or(KeyMask::None, KeyMask::Lens);
        let opts = Attn { keys, chunk: masks.chunk, bias: masks.bias, ..Attn::default() };
        let attn = ops::attention(&q, &k, &v, opts)?;
        self.project_out(x, &attn)
    }

    /// `x + mhsa(x)` for one streaming step: the `n` new frames of `x`
    /// `[B, n, d]` attend to this layer's slice of the key and value caches
    /// and to the first `window.visible[b]` of their own keys. Returns the
    /// output and the new keys and values `[B, n, H, d_k]` for the caller to
    /// keep.
    pub fn forward_stream(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        window: &KvWindow,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (q, k, v) = self.qkv(x, cos, sin)?;
        let cache = Cache {
            head_start: window.head_start,
            kv_heads: self.n_heads,
            row_map: None,
            appended: Some((&k, &v)),
            app_lens: Some(window.visible),
        };
        let opts = Attn { keys: KeyMask::Lens(window.filled), cache: Some(cache), ..Attn::default() };
        let attn = ops::attention(&q, window.keys, window.values, opts)?;
        Ok((self.project_out(x, &attn)?, k, v))
    }
}

/// One layer's view of the streaming key and value caches `[B, cap, H_all, d_k]`:
/// its heads start at `head_start`, `filled[b]` cached keys are real, and
/// `visible[b]` of the step's own keys are.
#[derive(Clone, Copy)]
pub struct KvWindow<'a> {
    pub keys: &'a Tensor,
    pub values: &'a Tensor,
    pub head_start: usize,
    pub filled: &'a Tensor,
    pub visible: &'a Tensor,
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

    /// `masks.pad_valid` (`[B, T]` bool, true = real frame) zeroes padded
    /// rows before the depthwise conv so they cannot leak into valid ones;
    /// `masks.causal` pads the depthwise conv on the left only, as the
    /// streaming encoder does; `masks.hardcopy` lays the sequence out as the
    /// look-ahead variant's training forward does.
    pub fn forward(&self, x: &Tensor, masks: &Masks) -> Result<Tensor> {
        let mut y = self.gated(x)?;
        if let Some(valid) = masks.pad_valid {
            y = y.where_(&valid.try_unsqueeze(1)?, 0.0)?;
        }
        let left = self.conv_kernel - 1;
        let conv = |inp: &Tensor, pad: Pad1d| {
            inp.conv1d().weight(&self.dw_weight).bias(&self.dw_bias).groups(self.d_model).padding(pad).call()
        };
        let y = match masks.hardcopy {
            None => {
                let pad: Pad1d = if masks.causal { (left as isize, 0).into() } else { ((left / 2) as isize).into() };
                conv(&y, pad)?
            }
            // The main sequence gets the causal conv; copy `j`, a copy of
            // chunk `j + 1`, gets main chunk `j`'s tail as its left context,
            // as the look-ahead chunk does in a streaming step.
            Some((chunks, chunk)) => {
                let (b, d, nc) = (y.dim_const(0)?, self.d_model, chunks * chunk);
                let main = y.narrow(2, 0, nc)?.try_pad(&[(0, 0), (0, 0), (left as isize, 0)])?;
                let tails: Vec<i32> =
                    (1..=chunks).flat_map(|j| (j * chunk..j * chunk + left).map(|i| i as i32)).collect();
                let tails = main.index_select(2, &Tensor::from_slice(tails))?.try_reshape([b, d, chunks, left])?;
                let copies = y.narrow(2, nc, nc)?.try_reshape([b, d, chunks, chunk])?;
                let inp = Tensor::cat(&[&tails, &copies], 3)?.try_permute(&[0, 2, 1, 3])?;
                let out = conv(&inp.try_reshape([b * chunks, d, left + chunk])?, 0.into())?;
                let out =
                    out.try_reshape([b, chunks, d, chunk])?.try_permute(&[0, 2, 1, 3])?.try_reshape([b, d, nc])?;
                Tensor::cat(&[&conv(&main, 0.into())?, &out], 2)?
            }
        };
        self.finish(x.dtype(), y)
    }

    /// One streaming step: the depthwise conv's left context comes from
    /// `cache` `[B, d, k-1]`, the frames before `x` `[B, t, d]`. The new cache
    /// is the last `k-1` frames before the final `lookahead` ones, which are
    /// the look-ahead variant's next chunk and will be seen again.
    pub fn forward_stream(&self, x: &Tensor, cache: &Tensor, lookahead: usize) -> Result<(Tensor, Tensor)> {
        let y = self.gated(x)?;
        let inp = Tensor::cat(&[cache, &y], 2)?;
        let (left, total) = (self.conv_kernel - 1, inp.dim_const(2)?);
        let cache = inp.narrow(2, total - lookahead - left, left)?;
        let y = inp.conv1d().weight(&self.dw_weight).bias(&self.dw_bias).groups(self.d_model).padding(0).call()?;
        Ok((self.finish(x.dtype(), y)?, cache))
    }

    /// `GLU(pw1(norm(x)))`, channels-first `[B, d, T]`: what the depthwise
    /// conv reads and the streaming cache holds.
    fn gated(&self, x: &Tensor) -> Result<Tensor> {
        let y = scoped("norm", || self.norm.forward(x))?;
        let y = y.try_transpose(-1, -2)?;
        Ok(y.conv1d().weight(&self.pw1_weight).bias(&self.pw1_bias).call()?.glu(1)?)
    }

    /// The norm, activation and second pointwise conv after the depthwise
    /// conv, back to `[B, T, d]`.
    fn finish(&self, activation_dtype: DType, y: Tensor) -> Result<Tensor> {
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

    /// `causal` pads each conv1d stage on the left only (the streaming
    /// encoder); conv2d subsampling has no causal form.
    pub fn forward(&self, x: &Tensor, causal: bool) -> Result<Tensor> {
        match &self.mode {
            SubsamplingMode::Conv1d => self.forward_conv1d(x, causal),
            SubsamplingMode::Conv2d => self.forward_conv2d(x),
        }
    }

    fn forward_conv1d(&self, x: &Tensor, causal: bool) -> Result<Tensor> {
        let x = x.try_transpose(-1, -2)?;

        let left = (self.kernel_size - 1) as isize;
        let pad: Pad1d = if causal { (left, 0).into() } else { (left / 2).into() };
        let stage = |x: &Tensor, weight: &Tensor, bias: &Tensor| -> Result<Tensor> {
            Ok(x.conv1d().weight(weight).bias(bias).stride(2).padding(pad).call()?.relu()?)
        };
        let x = stage(&x, &self.conv1_weight, &self.conv1_bias)?;
        let x = stage(&x, &self.conv2_weight, &self.conv2_bias)?;

        Ok(x.try_transpose(-1, -2)?)
    }

    /// One streaming step of the causal conv1d stages over `x`
    /// `[B, n_mels, t]`: each stage reads its cache of the last `k-1` input
    /// frames (`[B, n_mels, k-1]`, `[B, d, k-1]`; zeros at a stream's start)
    /// instead of padding. Returns `[B, t/4, d]` and the new caches.
    pub fn forward_stream(&self, x: &Tensor, caches: [&Tensor; 2]) -> Result<(Tensor, [Tensor; 2])> {
        let left = self.kernel_size - 1;
        let stage = |x: &Tensor, cache: &Tensor, weight: &Tensor, bias: &Tensor| -> Result<(Tensor, Tensor)> {
            let inp = Tensor::cat(&[cache, x], 2)?;
            let total = inp.dim_const(2)?;
            let out = inp.conv1d().weight(weight).bias(bias).stride(2).padding(0).call()?.relu()?;
            Ok((out, inp.narrow(2, total - left, left)?))
        };
        let (x, first) = stage(x, caches[0], &self.conv1_weight, &self.conv1_bias)?;
        let (x, second) = stage(&x, caches[1], &self.conv2_weight, &self.conv2_bias)?;
        Ok((x.try_transpose(-1, -2)?, [first, second]))
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

    /// One layer over a whole sequence, the attention and convolution seeing
    /// what `masks` say.
    pub fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, masks: &Masks) -> Result<Tensor> {
        let x = scoped("ffn1", || self.ffn1.forward(x))?;
        // The residual is added in the out projection's epilogue.
        let x = scoped("mhsa", || self.mhsa.forward(&x, cos, sin, masks))?;
        let conv = scoped("conv", || self.conv.forward(&x, masks))?;
        self.finish(&x.try_add(&conv)?)
    }

    /// One streaming step (see [`MultiHeadSelfAttention::forward_stream`] and
    /// [`ConvModule::forward_stream`]): returns the output, the new keys and
    /// values and the new conv cache.
    pub fn forward_stream(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        window: &KvWindow,
        conv_cache: &Tensor,
        lookahead: usize,
    ) -> Result<(Tensor, [Tensor; 3])> {
        let x = scoped("ffn1", || self.ffn1.forward(x))?;
        let (x, k, v) = scoped("mhsa", || self.mhsa.forward_stream(&x, cos, sin, window))?;
        let (conv, conv_cache) = scoped("conv", || self.conv.forward_stream(&x, conv_cache, lookahead))?;
        Ok((self.finish(&x.try_add(&conv)?)?, [k, v, conv_cache]))
    }

    /// FFN2 half-step and the final norm. Materialized in the activation
    /// dtype: otherwise the final norm's f32 cast fuses into the FFN2 GEMM,
    /// which then stores f32.
    fn finish(&self, x: &Tensor) -> Result<Tensor> {
        let x = scoped("ffn2", || self.ffn2.forward(x))?.contiguous();
        Ok(scoped("final_norm", || self.final_norm.forward(&x))?.contiguous())
    }
}

/// What a layer's attention and convolution see of a whole sequence.
#[derive(Clone, Copy, Default)]
pub struct Masks<'a> {
    /// `[B]` i32 valid encoder frames per lane, a key-only mask in MHSA.
    pub key_lens: Option<&'a Tensor>,
    /// `[B, T]` bool, true at real frames: zeroes padded rows before the
    /// depthwise conv.
    pub pad_valid: Option<&'a Tensor>,
    /// The streaming encoder's block attention.
    pub chunk: Option<Chunk>,
    /// An additive attention bias, `[B, H, T, T]`: the look-ahead variant's mask.
    pub bias: Option<&'a Tensor>,
    /// The streaming encoder's left-only conv padding.
    pub causal: bool,
    /// The look-ahead variant's `[main | copies]` layout, as `(chunks, chunk)`.
    pub hardcopy: Option<(usize, usize)>,
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
    /// The streaming variant's settings; `None` is the offline encoder. The
    /// parameters are the same either way.
    pub streaming: Option<StreamingConfig>,
}

/// The caches a streaming encoder step reads, one row per lane.
#[derive(Clone, Copy)]
pub struct StreamCaches<'a> {
    /// The last `k-1` input frames of each subsampling stage:
    /// `[S, n_mels, k-1]` and `[S, d, k-1]`.
    pub subsampling: [&'a Tensor; 2],
    /// The last `k-1` gated frames before every layer's depthwise conv:
    /// `[S, L, d, k-1]`.
    pub conv: &'a Tensor,
    /// Every layer's keys and values of the frames emitted so far,
    /// `[S, cap, L·H, d_k]`: a ring of chunk-sized slots the caller fills
    /// in order from each step's new rows, so a lane's first
    /// `min(emitted, cap)` keys are the real ones. The step only reads it.
    pub keys: &'a Tensor,
    pub values: &'a Tensor,
    /// The look-ahead variant's subsampled chunk awaiting its look-ahead:
    /// `[S, chunk, d]`.
    pub pending: &'a Tensor,
}

/// What a lane carries into a streaming step besides its caches, `[S]` i32 each.
#[derive(Clone, Copy)]
pub struct StreamCursors<'a> {
    /// Real mel frames in the lane's chunk; 0 for a lane with no chunk this step.
    pub valid: &'a Tensor,
    /// Encoder frames the lane has emitted: the next frame's rotary position
    /// and the count of cached keys.
    pub emitted: &'a Tensor,
    /// Look-ahead variant: frames in the pending chunk, 0 for none.
    pub pending: &'a Tensor,
    /// Look-ahead variant: nonzero ends the stream, emitting the pending
    /// chunk without a look-ahead.
    pub flush: &'a Tensor,
}

/// A streaming step's results: the emitted frames, the next value of every
/// cache the step carries, and the rows for the caller's key and value rings.
pub struct StreamStep {
    /// `[S, chunk, d]`: the chunk's frames (the pending chunk's, with
    /// look-ahead); how many are real the caller knows from its cursors.
    pub out: Tensor,
    pub subsampling: [Tensor; 2],
    pub conv: Tensor,
    pub pending: Tensor,
    /// Every layer's keys and values of the emitted frames, `[S, chunk, L·H, d_k]`,
    /// for the slot `emitted / chunk` of a lane that emitted.
    pub keys: Tensor,
    pub values: Tensor,
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
            streaming: config.streaming,
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

    /// RoPE tables for `positions` `[S, n]` i32, position-major
    /// `[n, S, 1, d_k/2]`: computed from the frequencies, so a stream's
    /// absolute positions need no table bound, and bit-equal to the cache's
    /// rows ([`Tensor::rope_inv_freq`]).
    fn rope_at(&self, positions: &Tensor) -> Result<(Tensor, Tensor)> {
        let d_k = self.d_model / self.n_heads;
        let inv_freq = Tensor::rope_inv_freq(self.max_encoder_frames as f64, d_k)?;
        let angles = positions.cast(DType::Float32).try_unsqueeze(-1)?.try_mul(&inv_freq)?;
        let table = |t: Tensor| -> Result<Tensor> { Ok(t.try_unsqueeze(2)?.try_permute(&[1, 0, 2, 3])?) };
        Ok((table(angles.cos()?)?, table(angles.sin()?)?))
    }

    /// The whole-sequence masks of the streaming variant without look-ahead,
    /// or none for the offline encoder.
    fn streaming_masks(&self) -> Masks<'static> {
        let Some(streaming) = self.streaming else { return Masks::default() };
        let chunk = Chunk { size: streaming.chunk_size, left: streaming.left_context_chunks, right: 0 };
        Masks { chunk: Some(chunk), causal: true, ..Masks::default() }
    }

    /// The streaming settings with look-ahead, if that is what this encoder is.
    fn look_ahead(&self) -> Option<StreamingConfig> {
        self.streaming.filter(|s| s.right_context_chunks > 0)
    }

    /// Encoder pass on a single mel batch with no padding mask.
    /// Input: tensor `[B, n_mels, T]`. Output: lazy tensor `[B, d_model, T/4]`.
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        if self.look_ahead().is_some() {
            let (b, t) = (mel.dim_const(0)?, mel.dim_const(2)?);
            return self.forward_batch(mel, &Tensor::full(&[b], t as i32, DType::Int32));
        }
        let masks = self.streaming_masks();
        let x = mel.try_transpose(-1, -2)?;
        let x = x.cast(self.input_dtype());
        let x = scoped("subsampling", || self.subsampling.forward(&x, masks.causal))?;

        let (cos, sin) = self.slice_rope(x.dim(1)?)?;

        // Single-batch path realizes `T` at the actual subsampled frame count of
        // this mel (no bucketing / no padding), so every key position is valid:
        // `key_lens = None` lets attention see the whole sequence.
        let mut x = x;
        for (index, layer) in self.layers.iter().enumerate() {
            x = scoped_index("layers", index, || layer.forward(&x, &cos, &sin, &masks))?;
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
        if let Some(streaming) = self.look_ahead() {
            return self.forward_hardcopy(mel, lengths, streaming);
        }
        let masks = self.streaming_masks();
        let lengths_sub = Self::subsampled_lengths(lengths)?;

        let x = mel.try_transpose(-1, -2)?;
        let x = x.cast(self.input_dtype());
        let x = scoped("subsampling", || self.subsampling.forward(&x, masks.causal))?;

        let t_sub = x.dim(1)?;

        // `key_lens` = subsampled valid-frame counts as a realized `[B]` `i32`
        // tensor — attention's key-only padding mask. The same lengths drive the
        // conv module's per-row validity mask.
        let key_lens = lengths_sub.try_reshape([mel.dim(0)?])?;
        let pad_valid = Tensor::sequence_mask(&key_lens, x.dim_const(1)?)?;
        let masks = Masks { key_lens: Some(&key_lens), pad_valid: Some(&pad_valid), ..masks };

        let (cos, sin) = self.slice_rope(t_sub)?;

        let mut x = x;
        for (index, layer) in self.layers.iter().enumerate() {
            x = scoped_index("layers", index, || layer.forward(&x, &cos, &sin, &masks))?;
        }

        Ok(x.try_transpose(-1, -2)?)
    }

    /// The look-ahead variant's whole-sequence forward, as its training runs:
    /// the subsampled sequence, padded to whole chunks, is followed by a copy
    /// of every chunk's successor (`[main | copies]`, positions `t` and
    /// `t + chunk`). A query of chunk `j`, main or copy, sees the main keys of
    /// chunks `[j - left, j]` and the copy keys of chunk `j` only, so a copy
    /// never sees its own look-ahead and nothing later reaches chunk `j` at
    /// any depth: frame for frame what a streaming step computes over the
    /// cached, pending and look-ahead frames. The copies are dropped from the
    /// output. The mask is an additive `[B, H, 2·N·chunk, 2·N·chunk]` bias.
    fn forward_hardcopy(&self, mel: &Tensor, lengths: &Tensor, streaming: StreamingConfig) -> Result<Tensor> {
        let chunk = streaming.chunk_size;
        let (b, h, d) = (mel.dim_const(0)?, self.n_heads, self.d_model);
        let x = mel.try_transpose(-1, -2)?.cast(self.input_dtype());
        let x = scoped("subsampling", || self.subsampling.forward(&x, true))?;
        let t = x.dim_const(1)?;
        let (chunks, nc) = (t.div_ceil(chunk).max(1), t.div_ceil(chunk).max(1) * chunk);
        let x = x.try_pad(&[(0, 0), (0, (nc - t) as isize), (0, 0)])?;
        let copies = Tensor::cat(&[&x.narrow(1, chunk, nc - chunk)?, &Tensor::zeros(&[b, chunk, d], x.dtype())], 1)?;
        let aug = Tensor::cat(&[&x, &copies], 1)?;

        let pos = Tensor::arange(0, Some(nc as i64), None)?;
        let positions = Tensor::cat(&[&pos, &pos.try_add(chunk as i32)?], 0)?.try_reshape([1, 2 * nc])?;
        let (cos, sin) = self.rope_at(&positions)?;

        // Main frame `t` is real below the length, copy frame `t` as frame `t + chunk` is.
        let lengths = Self::subsampled_lengths(lengths)?.try_reshape([b as isize, 1])?;
        let key_valid = Tensor::cat(&[&pos.try_lt(&lengths)?, &pos.try_add(chunk as i32)?.try_lt(&lengths)?], 1)?;
        let chunk_of = Tensor::cat(&[&pos, &pos], 0)?.try_cdiv(chunk as i32)?;
        let (cq, ck) = (chunk_of.try_reshape([2 * nc, 1])?, chunk_of.try_reshape([1, 2 * nc])?);
        let is_copy = Tensor::arange(0, Some(2 * nc as i64), None)?.try_ge(nc as i32)?.try_reshape([1, 2 * nc])?;
        let mut main = ck.try_le(&cq)?;
        if let Some(left) = streaming.left_context_chunks {
            main = main.try_bitand(&ck.try_add(left as i32)?.try_ge(&cq)?)?;
        }
        let allowed = ck.try_eq(&cq)?.where_(&is_copy, &main)?.try_unsqueeze(0)?;
        // A row always sees itself, so no softmax row is ever empty.
        let own = Tensor::arange(0, Some(2 * nc as i64), None)?;
        let own = own.try_reshape([2 * nc, 1])?.try_eq(&own.try_reshape([1, 2 * nc])?)?;
        let allowed = allowed.try_bitand(&key_valid.try_unsqueeze(1)?)?.try_bitor(&own)?;
        let bias = Tensor::const_(0.0f32, DType::Float32)
            .where_(&allowed, f32::NEG_INFINITY)?
            .cast(aug.dtype())
            .try_unsqueeze(1)?
            .try_expand([b, h, 2 * nc, 2 * nc])?
            .contiguous();
        let masks = Masks {
            pad_valid: Some(&key_valid),
            bias: Some(&bias),
            causal: true,
            hardcopy: Some((chunks, chunk)),
            ..Masks::default()
        };

        let mut x = aug;
        for (index, layer) in self.layers.iter().enumerate() {
            x = scoped_index("layers", index, || layer.forward(&x, &cos, &sin, &masks))?;
        }
        Ok(x.narrow(1, 0, t)?.try_transpose(-1, -2)?)
    }

    /// Mel frame counts through the two stride-2 stages: `len = (len + 1) / 2`
    /// each, in integer arithmetic, the same under either padding.
    fn subsampled_lengths(lengths: &Tensor) -> Result<Tensor> {
        let mut lengths_sub = lengths.cast(DType::Int32);
        for _ in 0..2 {
            lengths_sub = lengths_sub.try_add(1i32)?.try_div(2i32)?;
        }
        Ok(lengths_sub)
    }

    /// One streaming step over `S` lanes: `mel` `[S, n_mels, 4·chunk]` is
    /// every lane's next chunk, `cursors.valid[s]` frames of it real (0 for a
    /// lane sitting this step out). Without look-ahead the chunk's frames come
    /// out; with it the pending chunk's do once a new chunk or the flush is
    /// in, the new chunk becoming pending. A lane's caches change only when it
    /// emits, so idle lanes are untouched whatever their cursors say.
    ///
    /// The frames beyond `valid` are zeros from the mel front end, harmless
    /// to the causal convolutions and hidden from the attention, so a final
    /// partial chunk runs in the same shape; with look-ahead a partial pending
    /// chunk can only be followed by a flush.
    pub fn forward_stream(&self, mel: &Tensor, cursors: StreamCursors, caches: StreamCaches) -> Result<StreamStep> {
        let streaming = self.streaming.ok_or_else(|| super::Error::Streaming {
            message: "the offline encoder has no streaming step; load it with a streaming config".into(),
        })?;
        let (chunk, lookahead) = (streaming.chunk_size, streaming.right_context_chunks * streaming.chunk_size);
        let (s, d, h) = (mel.dim_const(0)?, self.d_model, self.n_heads);
        let n_fresh = Self::subsampled_lengths(cursors.valid)?;
        let fresh = cursors.valid.try_gt(0i32)?;
        // A lane's tensors gated on a `[S]` predicate: every cache is lane-major.
        let gate = |cond: &Tensor, new: Tensor, old: &Tensor| -> Result<Tensor> {
            let rank = new.ndim()?;
            let mut shape = vec![SInt::Const(1); rank];
            shape[0] = SInt::Const(s);
            Ok(new.where_(&cond.try_reshape(shape)?, old)?)
        };

        let x = mel.cast(self.input_dtype());
        let (x0, subs) = scoped("subsampling", || self.subsampling.forward_stream(&x, caches.subsampling))?;
        let [subs0, subs1] = subs;
        let subsampling = [gate(&fresh, subs0, caches.subsampling[0])?, gate(&fresh, subs1, caches.subsampling[1])?];

        // The frames the layers run over, and how many of their keys are
        // real: the pending chunk's (none without look-ahead) and, short of a
        // flush, the fresh chunk's.
        let flushed = cursors.flush.try_ne(0i32)?;
        let visible = cursors.pending.try_add(&n_fresh.where_(&flushed.logical_not()?, 0i32)?)?;
        // A pending chunk goes out once its look-ahead or the flush arrives.
        let (x, emits) = if lookahead == 0 {
            (x0.clone(), fresh.clone())
        } else {
            let due = fresh.try_bitor(&flushed)?;
            (Tensor::cat(&[caches.pending, &x0], 1)?, cursors.pending.try_gt(0i32)?.try_bitand(&due)?)
        };
        // A lane with nothing to see keeps a finite output.
        let visible = visible.maximum(1i32)?;
        let steps = x.dim_const(1)?;
        let positions = cursors.emitted.try_unsqueeze(1)?.try_add(&Tensor::arange(0, Some(steps as i64), None)?)?;
        let (cos, sin) = self.rope_at(&positions)?;
        let cap = caches.keys.dim_const(1)?;
        let filled = cursors.emitted.minimum(cap as i32)?;

        let mut x = x;
        let (mut keys, mut values, mut convs) = (Vec::new(), Vec::new(), Vec::new());
        for (index, layer) in self.layers.iter().enumerate() {
            let conv_cache = caches.conv.narrow(1, index, 1)?.try_squeeze(Some(1))?;
            let window = KvWindow {
                keys: caches.keys,
                values: caches.values,
                head_start: index * h,
                filled: &filled,
                visible: &visible,
            };
            let (out, [k, v, conv]) = scoped_index("layers", index, || {
                layer.forward_stream(&x, &cos, &sin, &window, &conv_cache, lookahead)
            })?;
            x = out;
            keys.push(k.narrow(1, 0, chunk)?);
            values.push(v.narrow(1, 0, chunk)?);
            convs.push(conv.try_unsqueeze(1)?);
        }
        // The rings are the caller's to write: a graph update of a cache
        // would copy the whole of it aside first, a device copy of the
        // chunk's rows into the lane's slot does not.
        let rows = |new: &[Tensor]| -> Result<Tensor> { Ok(Tensor::cat(&new.iter().collect::<Vec<_>>(), 2)?) };
        let (keys, values) = (rows(&keys)?, rows(&values)?);
        let convs: Vec<&Tensor> = convs.iter().collect();
        let conv = gate(&emits, Tensor::cat(&convs, 1)?, caches.conv)?;
        let pending = gate(&fresh, x0, caches.pending)?;
        let out = x.narrow(1, 0, chunk)?.try_reshape([s as isize, chunk as isize, d as isize])?;
        Ok(StreamStep { out, subsampling, conv, pending, keys, values })
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
            streaming: config.streaming,
        })
    }
}
