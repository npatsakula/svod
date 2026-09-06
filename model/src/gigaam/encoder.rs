use ndarray::Array4;
use snafu::ResultExt;
use svod_dtype::{DType, ScalarDType};
use svod_ir::SInt;
use svod_ir::origin::OriginScope;
use svod_tensor::Tensor;

use crate::init::{fan_in_uniform, ones, zeros};
use crate::state::{
    self, HasStateDict, StateDict, TensorSnafu as StateTensorSnafu, get_tensor, prefixed, scoped, scoped_index,
};
use crate::{load_state_field, state_field};

use super::error::{StateSnafu, TensorSnafu, TkSnafu};
use super::{ConvNormType, GigaAmConfig, SubsamplingMode};

/// Precompute RoPE cos/sin cache tensors. Returns `(cos, sin)` each of shape
/// `[max_encoder_frames, 1, 1, d_k/2]` where `d_k = d_model / n_heads`. Upstream
/// GigaAM passes `pos_emb_max_len` as both cache length and RoPE base.
fn build_rope_cache(config: &GigaAmConfig) -> (Tensor, Tensor) {
    let d_k = config.d_model / config.n_heads;
    let half_d = d_k / 2;
    let max_len = config.max_encoder_frames;
    let base = config.max_encoder_frames as f32;

    let inv_freq: Vec<f32> = (0..half_d).map(|i| 1.0 / base.powf(2.0 * i as f32 / d_k as f32)).collect();

    let mut cos_arr = Array4::<f32>::zeros((max_len, 1, 1, half_d));
    let mut sin_arr = Array4::<f32>::zeros((max_len, 1, 1, half_d));

    for pos in 0..max_len {
        for i in 0..half_d {
            let angle = pos as f32 * inv_freq[i];
            cos_arr[[pos, 0, 0, i]] = angle.cos();
            sin_arr[[pos, 0, 0, i]] = angle.sin();
        }
    }

    (Tensor::from_ndarray(&cos_arr), Tensor::from_ndarray(&sin_arr))
}

type Result<T> = super::Result<T>;

#[derive(Clone)]
pub struct DynamicQuantization {
    pub weight_scale: Tensor,
}

fn load_dynamic_quantization(
    sd: &StateDict,
    weight: &Tensor,
    weight_scale_key: &str,
) -> std::result::Result<Option<DynamicQuantization>, state::Error> {
    if !weight.uop().dtype().is_signed() {
        return Ok(None);
    }
    Ok(Some(DynamicQuantization { weight_scale: get_tensor(sd, weight_scale_key)? }))
}

fn linear(x: &Tensor, weight: &Tensor, bias: &Tensor, quantization: Option<&DynamicQuantization>) -> Result<Tensor> {
    match quantization {
        Some(quantization) => x
            .dynamic_quantized_linear()
            .weight(weight)
            .weight_scale(&quantization.weight_scale)
            .bias(bias)
            .call()
            .context(TensorSnafu),
        None => x.linear().weight(weight).bias(bias).call().context(TensorSnafu),
    }
}

// ---------------------------------------------------------------------------
// LayerNormWeights
// ---------------------------------------------------------------------------

/// Affine layer normalization: `layernorm(x) * weight + bias`.
#[derive(Clone)]
pub struct LayerNormWeights {
    pub weight: Tensor,
    pub bias: Tensor,
    pub eps: f64,
}

impl LayerNormWeights {
    pub fn empty(size: usize) -> Self {
        Self { weight: ones(&[size], DType::Float32), bias: zeros(&[size], DType::Float32), eps: 1e-5 }
    }

    pub fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let normed = x.layernorm(-1, self.eps).context(TensorSnafu)?;
        normed.try_mul(&self.weight).context(TensorSnafu)?.try_add(&self.bias).context(TensorSnafu)
    }
}

impl HasStateDict for LayerNormWeights {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        state_field!(sd, prefix, self, [weight, bias]);
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), crate::state::Error> {
        load_state_field!(self, sd, prefix, [weight, bias]);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// FeedForward
// ---------------------------------------------------------------------------

/// Conformer FFN: LayerNorm -> Linear(d->4d) -> SiLU -> Linear(4d->d).
///
/// Does NOT apply residual or 0.5 scaling — caller handles that.
#[derive(Clone)]
pub struct FeedForward {
    pub norm: LayerNormWeights,
    pub linear1_weight: Tensor,
    pub linear1_bias: Tensor,
    pub linear2_weight: Tensor,
    pub linear2_bias: Tensor,
    pub linear1_quantization: Option<DynamicQuantization>,
    pub linear2_quantization: Option<DynamicQuantization>,
}

impl FeedForward {
    pub fn empty(config: &GigaAmConfig) -> Self {
        let (d, d_ff) = (config.d_model, config.d_ff);
        Self {
            norm: LayerNormWeights::empty(d),
            linear1_weight: fan_in_uniform(&[d_ff, d], d, DType::Float32),
            linear1_bias: fan_in_uniform(&[d_ff], d, DType::Float32),
            linear2_weight: fan_in_uniform(&[d, d_ff], d_ff, DType::Float32),
            linear2_bias: fan_in_uniform(&[d], d_ff, DType::Float32),
            linear1_quantization: None,
            linear2_quantization: None,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // The two-linear FFN lowers to GEMM1+silu → h → GEMM2 (two reduces force `h`
        // to realize between them), which the generic optimizer fuses + (with BEAM)
        // tunes as well as a hand kernel — so the FFN stays plain graph ops.
        let y = scoped("norm", || self.norm.apply(x))?;
        let y = linear(&y, &self.linear1_weight, &self.linear1_bias, self.linear1_quantization.as_ref())?;
        let y = y.silu().context(TensorSnafu)?;
        linear(&y, &self.linear2_weight, &self.linear2_bias, self.linear2_quantization.as_ref())
    }
}

impl HasStateDict for FeedForward {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = self.norm.state_dict(&prefixed(prefix, "norm"));
        sd.insert(prefixed(prefix, "linear1.weight"), self.linear1_weight.clone());
        sd.insert(prefixed(prefix, "linear1.bias"), self.linear1_bias.clone());
        sd.insert(prefixed(prefix, "linear2.weight"), self.linear2_weight.clone());
        sd.insert(prefixed(prefix, "linear2.bias"), self.linear2_bias.clone());
        if let Some(quantization) = &self.linear1_quantization {
            sd.insert(prefixed(prefix, "linear1.weight_scale"), quantization.weight_scale.clone());
        }
        if let Some(quantization) = &self.linear2_quantization {
            sd.insert(prefixed(prefix, "linear2.weight_scale"), quantization.weight_scale.clone());
        }
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), crate::state::Error> {
        self.norm.load_state_dict(sd, &prefixed(prefix, "norm"))?;
        self.linear1_weight = get_tensor(sd, &prefixed(prefix, "linear1.weight"))?;
        self.linear1_bias = get_tensor(sd, &prefixed(prefix, "linear1.bias"))?;
        self.linear2_weight = get_tensor(sd, &prefixed(prefix, "linear2.weight"))?;
        self.linear2_bias = get_tensor(sd, &prefixed(prefix, "linear2.bias"))?;
        self.linear1_quantization =
            load_dynamic_quantization(sd, &self.linear1_weight, &prefixed(prefix, "linear1.weight_scale"))?;
        self.linear2_quantization =
            load_dynamic_quantization(sd, &self.linear2_weight, &prefixed(prefix, "linear2.weight_scale"))?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MultiHeadSelfAttention
// ---------------------------------------------------------------------------

/// Multi-head self-attention with rotary position embeddings.
#[derive(Clone)]
pub struct MultiHeadSelfAttention {
    pub norm: LayerNormWeights,
    pub q_proj: Tensor,
    pub q_bias: Tensor,
    pub k_proj: Tensor,
    pub k_bias: Tensor,
    pub v_proj: Tensor,
    pub v_bias: Tensor,
    pub out_proj: Tensor,
    pub out_bias: Tensor,
    pub q_quantization: Option<DynamicQuantization>,
    pub k_quantization: Option<DynamicQuantization>,
    pub v_quantization: Option<DynamicQuantization>,
    pub out_quantization: Option<DynamicQuantization>,
    pub n_heads: usize,
    pub d_model: usize,
}

impl MultiHeadSelfAttention {
    pub fn empty(config: &GigaAmConfig) -> Self {
        let d = config.d_model;
        Self {
            norm: LayerNormWeights::empty(d),
            q_proj: fan_in_uniform(&[d, d], d, DType::Float32),
            q_bias: fan_in_uniform(&[d], d, DType::Float32),
            k_proj: fan_in_uniform(&[d, d], d, DType::Float32),
            k_bias: fan_in_uniform(&[d], d, DType::Float32),
            v_proj: fan_in_uniform(&[d, d], d, DType::Float32),
            v_bias: fan_in_uniform(&[d], d, DType::Float32),
            out_proj: fan_in_uniform(&[d, d], d, DType::Float32),
            out_bias: fan_in_uniform(&[d], d, DType::Float32),
            q_quantization: None,
            k_quantization: None,
            v_quantization: None,
            out_quantization: None,
            n_heads: config.n_heads,
            d_model: d,
        }
    }

    /// `key_lens`, when present, is a realized `[B]` `i32` tensor of valid
    /// (unpadded) key positions per batch — keys at index `>= key_lens[b]` are
    /// masked. Passed to [`svod_tk::flash_attention_with`] as a key-only padding
    /// mask; when the hand kernel doesn't apply it returns `None` and [`sdpa_attention`]
    /// runs the same masked attention, so the result is correct on any device.
    pub fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, key_lens: Option<&Tensor>) -> Result<Tensor> {
        let shape = x.shape().context(TensorSnafu)?;
        let b = shape[0].clone();
        let t = shape[1].clone();
        let d_model = self.d_model;
        let d_k = d_model / self.n_heads;
        let h = self.n_heads;

        let y = scoped("norm", || self.norm.apply(x))?;

        // RoPE expects [T, B, H, d_k] (PyTorch ordering). Rotate once, then
        // materialise back as [B, T, d_model] so the Q/K projections share
        // a single rotated buffer.
        let y_heads = y
            .try_transpose(0, 1)
            .context(TensorSnafu)?
            .try_reshape([t.clone(), b.clone(), SInt::Const(h), SInt::Const(d_k)])
            .context(TensorSnafu)?;
        // The table is shared by every layer, so its cast is built outside the
        // layer's origin scope and hash-conses across layers.
        let rope_dtype = y_heads.uop().dtype();
        let (cos, sin) = {
            let _shared = OriginScope::suspend();
            (cos.cast(rope_dtype.clone()).context(TensorSnafu)?, sin.cast(rope_dtype).context(TensorSnafu)?)
        };
        let qk_input = y_heads
            .apply_rotary_emb(&cos, &sin, false)
            .context(TensorSnafu)?
            .try_reshape([t.clone(), b.clone(), SInt::Const(d_model)])
            .context(TensorSnafu)?
            .try_transpose(0, 1)
            .context(TensorSnafu)?
            .contiguous();

        let q = linear(&qk_input, &self.q_proj, &self.q_bias, self.q_quantization.as_ref())?;
        let k = linear(&qk_input, &self.k_proj, &self.k_bias, self.k_quantization.as_ref())?;
        let v = linear(&y, &self.v_proj, &self.v_bias, self.v_quantization.as_ref())?;

        // Head-split into `[B, T, H, d_k]` — the layout `flash_attention_with`
        // consumes directly (seq second, head third, head_dim last). No
        // `transpose(1,2)` to `[B, H, T, d_k]`: the hand kernel and the SDPA
        // fallback both take/return `[B, T, H, d_k]`.
        let q = split_heads(&q, b.clone(), t.clone(), h, d_k)?;
        let k = split_heads(&k, b.clone(), t.clone(), h, d_k)?;
        let v = split_heads(&v, b.clone(), t.clone(), h, d_k)?;

        // The hand FA kernel when it applies (a supported GPU + tiling shape), else this model's
        // own SDPA — tk no longer falls back silently; the policy lives here.
        let attn = if matches!(q.uop().dtype().base(), ScalarDType::Float16 | ScalarDType::BFloat16) {
            match svod_tk::flash_attention_with(&q, &k, &v, svod_tk::FaOpts { causal: false, key_lens })
                .context(TkSnafu)?
            {
                Some(out) => out,
                None => sdpa_attention(&q, &k, &v, key_lens)?,
            }
        } else {
            sdpa_attention(&q, &k, &v, key_lens)?
        };
        let out = merge_heads(&attn, b, t, d_model)?;
        linear(&out, &self.out_proj, &self.out_bias, self.out_quantization.as_ref())
    }
}

/// SDPA fallback for when `svod_tk::flash_attention_with` returns `None` (non-AMD
/// device or a non-tiling sequence length). Mirrors the kernel's contract: input
/// and output stay `[B, T, H, d_k]`, attention is non-causal, and `key_lens` masks
/// padded KEY positions only (`kv_pos ≥ key_lens[b]`). Permutes to the
/// `[B, H, T, d_k]` SDPA wants and back.
fn sdpa_attention(q: &Tensor, k: &Tensor, v: &Tensor, key_lens: Option<&Tensor>) -> Result<Tensor> {
    let perm = |t: &Tensor| t.try_permute(&[0, 2, 1, 3]).context(TensorSnafu);
    let (qp, kp, vp) = (perm(q)?, perm(k)?, perm(v)?);
    let mask = match key_lens {
        Some(lens) => {
            let qs = q.shape().expect("q shape");
            let dim = |i: usize| qs[i].as_const().expect("concrete dim");
            let (b, n) = (dim(0), dim(1));
            // [B, 1, 1, N] bool key mask: true (masked) where arange(N) ≥ key_lens[b].
            // A property of `key_lens`, shared by every layer: built outside the
            // layer's origin scope so the layers share one mask.
            let _shared = OriginScope::suspend();
            let range = Tensor::arange(n as i64, None, None)
                .context(TensorSnafu)?
                .try_reshape([1usize, 1, 1, n])
                .context(TensorSnafu)?;
            let lens = lens.cast(DType::Int32).context(TensorSnafu)?.try_reshape([b, 1, 1, 1]).context(TensorSnafu)?;
            Some(range.try_ge(&lens).context(TensorSnafu)?)
        }
        None => None,
    };
    let out = qp
        .scaled_dot_product_attention()
        .key(&kp)
        .value(&vp)
        .is_causal(false)
        .maybe_attn_mask(mask.as_ref())
        .call()
        .context(TensorSnafu)?;
    out.try_permute(&[0, 2, 1, 3]).context(TensorSnafu)
}

/// `[B, T, H*d_k] → [B, T, H, d_k]`. Leaves the seq-major layout that
/// `flash_attention_with` consumes (no `transpose(1,2)` to head-major).
fn split_heads(x: &Tensor, b: SInt, t: SInt, h: usize, d_k: usize) -> Result<Tensor> {
    x.try_reshape([b, t, SInt::Const(h), SInt::Const(d_k)]).context(TensorSnafu)
}

/// `[B, T, H, d_k] → [B, T, d_model]`. The attention output is already
/// seq-major, so this is a plain head-merge reshape (no transpose).
fn merge_heads(x: &Tensor, b: SInt, t: SInt, d_model: usize) -> Result<Tensor> {
    x.try_reshape([b, t, SInt::Const(d_model)]).context(TensorSnafu)
}

impl HasStateDict for MultiHeadSelfAttention {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = self.norm.state_dict(&prefixed(prefix, "norm"));
        state_field!(sd, prefix, self, [q_proj, q_bias, k_proj, k_bias, v_proj, v_bias, out_proj, out_bias]);
        for (name, quantization) in [
            ("q", &self.q_quantization),
            ("k", &self.k_quantization),
            ("v", &self.v_quantization),
            ("out", &self.out_quantization),
        ] {
            if let Some(quantization) = quantization {
                sd.insert(prefixed(prefix, &format!("{name}_weight_scale")), quantization.weight_scale.clone());
            }
        }
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), crate::state::Error> {
        self.norm.load_state_dict(sd, &prefixed(prefix, "norm"))?;
        load_state_field!(self, sd, prefix, [q_proj, q_bias, k_proj, k_bias, v_proj, v_bias, out_proj, out_bias]);
        self.q_quantization = load_dynamic_quantization(sd, &self.q_proj, &prefixed(prefix, "q_weight_scale"))?;
        self.k_quantization = load_dynamic_quantization(sd, &self.k_proj, &prefixed(prefix, "k_weight_scale"))?;
        self.v_quantization = load_dynamic_quantization(sd, &self.v_proj, &prefixed(prefix, "v_weight_scale"))?;
        self.out_quantization = load_dynamic_quantization(sd, &self.out_proj, &prefixed(prefix, "out_weight_scale"))?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ConvModule
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub enum ConvNorm {
    LayerNorm(LayerNormWeights),
    BatchNorm {
        scale: Tensor,
        bias: Tensor,
        mean: Tensor,
        invstd: Tensor,
    },
    /// Inference-time BatchNorm fold: the affine collapsed into the depthwise
    /// conv at load (`w *= s`, `b = b*s + bias - mean*s` with
    /// `s = scale*invstd`), so the norm op vanishes from the graph.
    Folded,
}

/// Conformer convolution module:
/// LayerNorm -> Conv1d(d,2d,k=1) -> GLU -> DepthwiseConv1d -> Norm -> SiLU -> Conv1d(d,d,k=1)
#[derive(Clone)]
pub struct ConvModule {
    pub norm: LayerNormWeights,
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
            ConvNormType::LayerNorm => ConvNorm::LayerNorm(LayerNormWeights::empty(d)),
            ConvNormType::BatchNorm => ConvNorm::BatchNorm {
                scale: ones(&[d], DType::Float32),
                bias: zeros(&[d], DType::Float32),
                mean: zeros(&[d], DType::Float32),
                invstd: ones(&[d], DType::Float32),
            },
        };
        Self {
            norm: LayerNormWeights::empty(d),
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

    pub fn forward(&self, x: &Tensor, pad_mask: Option<&Tensor>) -> Result<Tensor> {
        let activation_dtype = x.uop().dtype();
        let y = scoped("norm", || self.norm.apply(x))?;

        let y = y.try_transpose(-1, -2).context(TensorSnafu)?;

        let y = y.conv2d().weight(&self.pw1_weight).bias(&self.pw1_bias).call().context(TensorSnafu)?;

        let mut y = y.glu(1).context(TensorSnafu)?;

        if let Some(mask) = pad_mask {
            let valid = mask.logical_not().context(TensorSnafu)?;
            let valid = valid.try_unsqueeze(1).context(TensorSnafu)?;
            let zeros = y.zero().context(TensorSnafu)?;
            y = y.where_(&valid, &zeros).context(TensorSnafu)?;
        }

        let pad = ((self.conv_kernel - 1) / 2) as isize;
        let y = y
            .conv2d()
            .weight(&self.dw_weight)
            .bias(&self.dw_bias)
            .groups(self.d_model)
            .padding(&[(pad, pad)])
            .call()
            .context(TensorSnafu)?;

        let y = match &self.conv_norm {
            ConvNorm::LayerNorm(ln) => {
                let y = y.try_transpose(-1, -2).context(TensorSnafu)?;
                let y = scoped("conv_norm", || ln.apply(&y))?;
                y.try_transpose(-1, -2).context(TensorSnafu)?
            }
            ConvNorm::BatchNorm { scale, bias, mean, invstd } => {
                y.batchnorm().scale(scale).bias(bias).mean(mean).invstd(invstd).call().context(TensorSnafu)?
            }
            ConvNorm::Folded => y,
        };
        // BN/LN params (scale, bias, mean, invstd) are stored fp32; broadcasting promotes
        // the norm output. Re-cast to the activation dtype so SiLU/pw2 stay in the right
        // precision, matching Python's BatchNorm1d dtype semantics. No-op when types match.
        let y = if y.uop().dtype() != activation_dtype { y.cast(activation_dtype).context(TensorSnafu)? } else { y };

        let y = y.silu().context(TensorSnafu)?;

        let y = y.conv2d().weight(&self.pw2_weight).bias(&self.pw2_bias).call().context(TensorSnafu)?;

        y.try_transpose(-1, -2).context(TensorSnafu)
    }
}

impl HasStateDict for ConvModule {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = self.norm.state_dict(&prefixed(prefix, "norm"));
        state_field!(sd, prefix, self, [pw1_weight, pw1_bias, dw_weight, dw_bias, pw2_weight, pw2_bias]);
        match &self.conv_norm {
            ConvNorm::LayerNorm(ln) => sd.extend(ln.state_dict(&prefixed(prefix, "conv_norm"))),
            ConvNorm::BatchNorm { scale, bias, mean, invstd } => {
                for (name, t) in [("bn_scale", scale), ("bn_bias", bias), ("bn_mean", mean), ("bn_invstd", invstd)] {
                    sd.insert(prefixed(prefix, name), t.clone());
                }
            }
            // Folded weights live in dw_weight/dw_bias; the BN params are gone.
            ConvNorm::Folded => {}
        }
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), crate::state::Error> {
        self.norm.load_state_dict(sd, &prefixed(prefix, "norm"))?;
        load_state_field!(self, sd, prefix, [pw1_weight, pw1_bias, dw_weight, dw_bias, pw2_weight, pw2_bias]);
        match &mut self.conv_norm {
            ConvNorm::LayerNorm(ln) => ln.load_state_dict(sd, &prefixed(prefix, "conv_norm"))?,
            // Folded round-trips (e.g. `cast_weights`) carry no BN keys — the
            // affine already lives in dw_weight/dw_bias.
            ConvNorm::Folded if !sd.contains_key(&prefixed(prefix, "bn_scale")) => {}
            ConvNorm::BatchNorm { .. } | ConvNorm::Folded => {
                // Fold BN into the depthwise conv at load: with s = scale*invstd,
                // y_bn = conv(x)*s + (bias - mean*s) — per-channel scale folds
                // into the conv weight rows + a corrected bias. Removes the
                // norm op from the encoder graph.
                let scale = get_tensor(sd, &prefixed(prefix, "bn_scale"))?;
                let bias = get_tensor(sd, &prefixed(prefix, "bn_bias"))?;
                let mean = get_tensor(sd, &prefixed(prefix, "bn_mean"))?;
                let invstd = get_tensor(sd, &prefixed(prefix, "bn_invstd"))?;
                let fold = || -> std::result::Result<(Tensor, Tensor), state::Error> {
                    let d = self.d_model as isize;
                    let s = scale.try_mul(&invstd).context(StateTensorSnafu)?;
                    let w = self
                        .dw_weight
                        .try_mul(&s.try_reshape([d, 1, 1]).context(StateTensorSnafu)?)
                        .context(StateTensorSnafu)?;
                    let scaled_mean = mean.try_mul(&s).context(StateTensorSnafu)?;
                    let b = self
                        .dw_bias
                        .try_mul(&s)
                        .context(StateTensorSnafu)?
                        .try_add(&bias)
                        .context(StateTensorSnafu)?
                        .try_sub(&scaled_mean)
                        .context(StateTensorSnafu)?;
                    Ok((w, b))
                };
                let (w, b) = fold()?;
                self.dw_weight = w;
                self.dw_bias = b;
                self.conv_norm = ConvNorm::Folded;
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
#[derive(Clone)]
pub struct StridingSubsampling {
    pub conv1_weight: Tensor,
    pub conv1_bias: Tensor,
    pub conv2_weight: Tensor,
    pub conv2_bias: Tensor,
    pub linear_weight: Option<Tensor>,
    pub linear_bias: Option<Tensor>,
    n_mels: usize,
    d_model: usize,
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

    pub fn output_length(&self, input_length: usize) -> usize {
        let pad = (self.kernel_size - 1) / 2;
        let mut len = input_length;
        for _ in 0..2 {
            len = (len + 2 * pad - self.kernel_size) / 2 + 1;
        }
        len
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match &self.mode {
            SubsamplingMode::Conv1d => self.forward_conv1d(x),
            SubsamplingMode::Conv2d => self.forward_conv2d(x),
        }
    }

    fn forward_conv1d(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.try_transpose(-1, -2).context(TensorSnafu)?;

        let pad = (self.kernel_size / 2) as isize;
        let x = x
            .conv2d()
            .weight(&self.conv1_weight)
            .bias(&self.conv1_bias)
            .stride(&[2])
            .padding(&[(pad, pad)])
            .call()
            .context(TensorSnafu)?;
        let x = x.relu().context(TensorSnafu)?;

        let x = x
            .conv2d()
            .weight(&self.conv2_weight)
            .bias(&self.conv2_bias)
            .stride(&[2])
            .padding(&[(pad, pad)])
            .call()
            .context(TensorSnafu)?;
        let x = x.relu().context(TensorSnafu)?;

        x.try_transpose(-1, -2).context(TensorSnafu)
    }

    fn forward_conv2d(&self, x: &Tensor) -> Result<Tensor> {
        let shape = x.shape().context(TensorSnafu)?;
        let b = shape[0].clone();

        let x = x.try_unsqueeze(1).context(TensorSnafu)?;

        let x = x
            .conv2d()
            .weight(&self.conv1_weight)
            .bias(&self.conv1_bias)
            .stride(&[2, 2])
            .padding(&[(1, 1), (1, 1)])
            .call()
            .context(TensorSnafu)?;
        let x = x.relu().context(TensorSnafu)?;

        let x = x
            .conv2d()
            .weight(&self.conv2_weight)
            .bias(&self.conv2_bias)
            .stride(&[2, 2])
            .padding(&[(1, 1), (1, 1)])
            .call()
            .context(TensorSnafu)?;
        let x = x.relu().context(TensorSnafu)?;

        let x = x.try_permute(&[0, 2, 1, 3]).context(TensorSnafu)?;
        let x = x.try_reshape([b, SInt::Infer, SInt::Const(self.d_model * self.n_mels / 4)]).context(TensorSnafu)?;

        let lw = self.linear_weight.as_ref().expect("conv2d mode requires linear_weight");
        let lb = self.linear_bias.as_ref().expect("conv2d mode requires linear_bias");
        x.linear().weight(lw).bias(lb).call().context(TensorSnafu)
    }
}

impl HasStateDict for StridingSubsampling {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        state_field!(sd, prefix, self, [conv1_weight, conv1_bias, conv2_weight, conv2_bias]);
        if let (Some(lw), Some(lb)) = (&self.linear_weight, &self.linear_bias) {
            sd.insert(prefixed(prefix, "linear_weight"), lw.clone());
            sd.insert(prefixed(prefix, "linear_bias"), lb.clone());
        }
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), crate::state::Error> {
        load_state_field!(self, sd, prefix, [conv1_weight, conv1_bias, conv2_weight, conv2_bias]);
        if matches!(self.mode, SubsamplingMode::Conv2d) {
            self.linear_weight = Some(get_tensor(sd, &prefixed(prefix, "linear_weight"))?);
            self.linear_bias = Some(get_tensor(sd, &prefixed(prefix, "linear_bias"))?);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ConformerLayer
// ---------------------------------------------------------------------------

/// One Conformer layer (Macaron-style):
/// FFN1(x0.5) + MHSA + Conv + FFN2(x0.5) + LayerNorm
#[derive(Clone)]
pub struct ConformerLayer {
    pub ffn1: FeedForward,
    pub mhsa: MultiHeadSelfAttention,
    pub conv: ConvModule,
    pub ffn2: FeedForward,
    pub final_norm: LayerNormWeights,
}

impl ConformerLayer {
    pub fn empty(config: &GigaAmConfig) -> Self {
        Self {
            ffn1: FeedForward::empty(config),
            mhsa: MultiHeadSelfAttention::empty(config),
            conv: ConvModule::empty(config),
            ffn2: FeedForward::empty(config),
            final_norm: LayerNormWeights::empty(config.d_model),
        }
    }

    /// `key_lens` is the optional per-batch valid encoder-frame count (`[B]`
    /// `i32`) used as a key-only padding mask in MHSA (see
    /// [`MultiHeadSelfAttention::forward`]). `pad_mask` independently zeros
    /// padded rows in the conv module.
    pub fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        key_lens: Option<&Tensor>,
        pad_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let half = Tensor::from_const(0.5f64).cast(x.uop().dtype()).context(TensorSnafu)?;

        // FFN1 half-step
        let ffn1 = scoped("ffn1", || self.ffn1.forward(x))?;
        let x = x.try_add(&ffn1.try_mul(&half).context(TensorSnafu)?).context(TensorSnafu)?;

        // MHSA
        let mhsa = scoped("mhsa", || self.mhsa.forward(&x, cos, sin, key_lens))?;
        let x = x.try_add(&mhsa).context(TensorSnafu)?;

        // Convolution
        let conv = scoped("conv", || self.conv.forward(&x, pad_mask))?;
        let x = x.try_add(&conv).context(TensorSnafu)?;

        // FFN2 half-step
        let ffn2 = scoped("ffn2", || self.ffn2.forward(&x))?;
        let x = x.try_add(&ffn2.try_mul(&half).context(TensorSnafu)?).context(TensorSnafu)?;

        // Final layer norm
        scoped("final_norm", || self.final_norm.apply(&x))
    }
}

impl HasStateDict for ConformerLayer {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.extend(self.ffn1.state_dict(&prefixed(prefix, "ffn1")));
        sd.extend(self.mhsa.state_dict(&prefixed(prefix, "mhsa")));
        sd.extend(self.conv.state_dict(&prefixed(prefix, "conv")));
        sd.extend(self.ffn2.state_dict(&prefixed(prefix, "ffn2")));
        sd.extend(self.final_norm.state_dict(&prefixed(prefix, "final_norm")));
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), crate::state::Error> {
        self.ffn1.load_state_dict(sd, &prefixed(prefix, "ffn1"))?;
        self.mhsa.load_state_dict(sd, &prefixed(prefix, "mhsa"))?;
        self.conv.load_state_dict(sd, &prefixed(prefix, "conv"))?;
        self.ffn2.load_state_dict(sd, &prefixed(prefix, "ffn2"))?;
        self.final_norm.load_state_dict(sd, &prefixed(prefix, "final_norm"))?;
        Ok(())
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
        let (cos_cache, sin_cache) = build_rope_cache(config);
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
        let dtype = self.subsampling.conv1_weight.uop().dtype();
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
        let cos = self.cos_cache.try_shrink(shrink.clone()).context(TensorSnafu)?;
        let sin = self.sin_cache.try_shrink(shrink).context(TensorSnafu)?;
        Ok((cos, sin))
    }

    /// Encoder pass on a single mel batch with no padding mask.
    /// Input: tensor `[B, n_mels, T]`. Output: lazy tensor `[B, d_model, T/4]`.
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let x = mel.try_transpose(-1, -2).context(TensorSnafu)?;
        let x = x.cast(self.input_dtype()).context(TensorSnafu)?;
        let x = scoped("subsampling", || self.subsampling.forward(&x))?;

        let shape = x.shape().context(TensorSnafu)?;
        let seq_len = shape[1].clone();
        let (cos, sin) = self.slice_rope(seq_len)?;

        // Single-batch path realizes `T` at the actual subsampled frame count of
        // this mel (no bucketing / no padding), so every key position is valid:
        // `key_lens = None` lets attention see the whole sequence.
        let mut x = x;
        for (index, layer) in self.layers.iter().enumerate() {
            x = scoped_index("layers", index, || layer.forward(&x, &cos, &sin, None, None))?;
        }

        x.try_transpose(-1, -2).context(TensorSnafu)
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
        let mel_shape = mel.shape().context(TensorSnafu)?;
        let b = mel_shape[0].clone();

        let lengths = lengths.cast(DType::Int32).context(TensorSnafu)?;

        let two_t = Tensor::const_(2i32, DType::Int32);
        let one_t = Tensor::const_(1i32, DType::Int32);

        let mut lengths_sub = lengths;
        for _ in 0..2 {
            lengths_sub = lengths_sub.try_add(&one_t).context(TensorSnafu)?.try_div(&two_t).context(TensorSnafu)?;
        }

        let x = mel.try_transpose(-1, -2).context(TensorSnafu)?;
        let x = x.cast(self.input_dtype()).context(TensorSnafu)?;
        let x = scoped("subsampling", || self.subsampling.forward(&x))?;

        let shape = x.shape().context(TensorSnafu)?;
        let t_sub = shape[1].clone();

        // `key_lens` = subsampled valid-frame counts as a realized `[B]` `i32`
        // tensor — attention's key-only padding mask. Keep `lengths_sub` in the
        // same concrete dtype for the `pad_valid` comparison (the conv `pad_mask`).
        let key_lens =
            lengths_sub.cast(DType::Int32).context(TensorSnafu)?.try_reshape([b.clone()]).context(TensorSnafu)?;

        let range = Tensor::arange(self.max_encoder_frames as i64, None, None).context(TensorSnafu)?;
        let range = range.cast(DType::Int32).context(TensorSnafu)?;
        let range = range.try_shrink([(SInt::Const(0), t_sub.clone())]).context(TensorSnafu)?;
        let range = range.try_reshape([SInt::Const(1), t_sub.clone()]).context(TensorSnafu)?;
        let lens = lengths_sub;
        let lens = lens.try_reshape([b.clone(), SInt::Const(1)]).context(TensorSnafu)?;
        let pad_valid = range.try_lt(&lens).context(TensorSnafu)?;

        let pad_mask = pad_valid.logical_not().context(TensorSnafu)?;

        let (cos, sin) = self.slice_rope(t_sub)?;

        let mut x = x;
        for (index, layer) in self.layers.iter().enumerate() {
            x = scoped_index("layers", index, || layer.forward(&x, &cos, &sin, Some(&key_lens), Some(&pad_mask)))?;
        }

        x.try_transpose(-1, -2).context(TensorSnafu)
    }

    pub fn subsampling_output_length(&self, mel_frames: usize) -> usize {
        self.subsampling.output_length(mel_frames)
    }

    /// Construct an `Encoder` from an already-remapped state dict + config.
    /// Called from the unified [`crate::gigaam::GigaAm::from_state_dict`] loader.
    pub(crate) fn from_state_dict(sd: &StateDict, config: &GigaAmConfig) -> Result<Self> {
        let (cos_cache, sin_cache) = build_rope_cache(config);

        let mut subsampling = StridingSubsampling::empty(config);
        subsampling.load_state_dict(sd, "subsampling").context(StateSnafu)?;

        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let mut layer = ConformerLayer::empty(config);
            layer.load_state_dict(sd, &format!("layers.{i}")).context(StateSnafu)?;
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
