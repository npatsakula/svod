//! Qwen3 pre-norm decoder layer:
//! ```text
//! h = x + attn(input_layernorm(x))
//! y = h + mlp(post_attention_layernorm(h))
//! ```
//!
//! Each residual add feeds exactly one norm, so the stream travels as an
//! unsummed [`Residual`] until something writes it. Left to the graph the add is
//! lazy and every consumer — the norm's reduce, its apply, and the next layer —
//! recomputes it. Two places can absorb it, in order of preference:
//!
//! 1. The projection that **produced** the addend: `o_proj` and `down_proj` fold
//!    it into their GEMM's epilogue ([`svod_tk::Epilogue::Add`]), so the summed
//!    stream is the GEMM's own store and the norm reading it is the two-pass
//!    [`svod_tk::rms_norm`].
//! 2. Failing that, the norm that consumes it: [`svod_tk::add_rms_norm`] writes
//!    the summed stream and its norm together in one pass over memory.

use snafu::ResultExt;
use svod_dtype::ScalarDType;
use svod_tensor::Tensor;
use svod_tensor::nn::{Layer, Module, RmsNorm};

use super::attention::Qwen3Attention;
use super::error::{Result, TkSnafu};
use super::feed_forward::Qwen3MLP;
use super::linear::Projected;

/// The residual stream between sublayers: logically `stream + pending`, held
/// unsummed until a norm consumes both in one pass.
#[derive(Clone)]
pub(crate) struct Residual {
    stream: Tensor,
    pending: Option<Tensor>,
}

impl From<Tensor> for Residual {
    fn from(stream: Tensor) -> Self {
        Self { stream, pending: None }
    }
}

/// Whether the hand kernels can take this activation: 16-bit at a concrete shape.
fn fusable(x: &Tensor) -> bool {
    matches!(x.dtype().base(), ScalarDType::BFloat16 | ScalarDType::Float16)
        && x.shape().is_ok_and(|s| s.iter().all(|d| d.as_const().is_some()))
}

impl Residual {
    /// The stream after a projection wrote into it: [`Projected::Summed`] came
    /// out of its GEMM already added, so the stream is that output and nothing
    /// is pending; [`Projected::Pending`] leaves the add to the next norm.
    pub(crate) fn advance(stream: Tensor, projected: Projected) -> Self {
        match projected {
            Projected::Summed(h) => Self { stream: h, pending: None },
            Projected::Pending(y) => Self { stream, pending: Some(y) },
        }
    }

    /// The stream as one tensor.
    pub(crate) fn join(self) -> Result<Tensor> {
        Ok(match self.pending {
            Some(p) => self.stream.try_add(&p)?,
            None => self.stream,
        })
    }

    /// `(stream, rms_norm(stream))`, the pending add folded in — one kernel when
    /// it applies, else the lazy add plus `RmsNorm::forward`.
    pub(crate) fn norm(self, norm: &RmsNorm) -> Result<(Tensor, Tensor)> {
        let Residual { stream, pending } = self;
        if fusable(&stream) {
            let fused = match &pending {
                Some(p) => svod_tk::add_rms_norm(p, &stream, &norm.weight, norm.eps).context(TkSnafu)?,
                None => {
                    svod_tk::rms_norm(&stream, &norm.weight, norm.eps).context(TkSnafu)?.map(|y| (stream.clone(), y))
                }
            };
            if let Some(out) = fused {
                return Ok(out);
            }
        }
        let x = Residual { stream, pending }.join()?;
        let y = norm.forward(&x)?;
        Ok((x, y))
    }
}

#[derive(Clone, Module)]
pub struct Qwen3DecoderLayer {
    pub input_layernorm: RmsNorm,
    #[module(key = "self_attn")]
    pub attention: Qwen3Attention,
    pub post_attention_layernorm: RmsNorm,
    pub mlp: Qwen3MLP,
}

impl Qwen3DecoderLayer {
    pub fn empty(config: &super::Qwen3Config) -> Self {
        let dtype = config.dtype.clone();
        Self {
            input_layernorm: RmsNorm::with_dims(config.hidden_size, config.rms_norm_eps, dtype.clone()),
            attention: Qwen3Attention::empty(
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                config.rms_norm_eps,
                dtype.clone(),
            ),
            post_attention_layernorm: RmsNorm::with_dims(config.hidden_size, config.rms_norm_eps, dtype.clone()),
            mlp: Qwen3MLP::empty(config.hidden_size, config.intermediate_size, dtype),
        }
    }

    pub fn forward(&self, x: &Tensor, rope: &(Tensor, Tensor)) -> Result<Tensor> {
        self.forward_residual(Residual::from(x.clone()), rope)?.join()
    }

    /// The layer over the unsummed stream: the incoming add is consumed by the
    /// input norm, and the outgoing mlp add is left pending for the next layer.
    pub(crate) fn forward_residual(&self, x: Residual, rope: &(Tensor, Tensor)) -> Result<Residual> {
        let (x, x_norm) = x.norm(&self.input_layernorm)?;
        let attn = self.attention.forward_into(&x_norm, rope, Some(&x))?;
        let (h, h_norm) = Residual::advance(x, attn).norm(&self.post_attention_layernorm)?;
        let mlp = self.mlp.forward_into(&h_norm, Some(&h))?;
        Ok(Residual::advance(h, mlp))
    }
}
