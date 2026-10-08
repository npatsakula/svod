//! Audio encoder: Conv1d frontend + sinusoidal positional embeddings + transformer blocks.

use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::{Conv1d, Layer, LayerNorm, Linear, Module};
use svod_tk3::ops::{self, Act};

use crate::init::{Bias, conv1d, layer_norm, linear};
use crate::state::{scoped, scoped_index};

use super::attention::MultiHeadAttention;
use super::blocks::{project, sinusoids};
use super::config::ModelDimensions;
use super::error::Result;

fn norm(layer: &LayerNorm, x: &Tensor) -> Result<Tensor> {
    Ok(ops::layer_norm(x, &layer.weight, layer.bias.as_ref(), layer.eps)?)
}

/// Encoder transformer block: self-attention + MLP, pre-norm.
#[derive(Clone, Module)]
pub struct EncoderBlock {
    pub attn: MultiHeadAttention,
    pub attn_ln: LayerNorm,
    #[module(key = "mlp.0")]
    pub mlp0: Linear,
    #[module(key = "mlp.2")]
    pub mlp2: Linear,
    pub mlp_ln: LayerNorm,
    pub n_state: usize,
}

impl EncoderBlock {
    pub fn empty(n_state: usize, n_head: usize) -> Self {
        Self::empty_dtype(n_state, n_head, DType::Float32)
    }

    pub fn empty_dtype(n_state: usize, n_head: usize, dtype: DType) -> Self {
        let mlp = n_state * 4;
        Self {
            attn: MultiHeadAttention::empty_dtype(n_state, n_head, dtype.clone()),
            attn_ln: layer_norm(n_state, dtype.clone()),
            mlp0: linear(n_state, mlp, Bias::FanIn, dtype.clone()),
            mlp2: linear(mlp, n_state, Bias::FanIn, dtype.clone()),
            mlp_ln: layer_norm(n_state, dtype),
            n_state,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = scoped("attn_ln", || norm(&self.attn_ln, x))?;
        // The residual adds stay out of the GEMM epilogue: with the norm on
        // the graph path, a launch whose residual operand the norm also reads
        // fails kernel-graph verification.
        let x = x.try_add(&scoped("attn", || self.attn.encode(&h))?)?;
        let h = scoped("mlp_ln", || norm(&self.mlp_ln, &x))?;
        let h = project(&self.mlp0, &h, Act::Gelu)?;
        Ok(x.try_add(&project(&self.mlp2, &h, Act::None)?)?)
    }
}

/// Whisper audio encoder: Conv1d × 2 + sinusoidal pos-emb + N × EncoderBlock + LayerNorm.
#[derive(Clone, Module)]
pub struct AudioEncoder {
    pub conv1: Conv1d,
    pub conv2: Conv1d,
    pub positional_embedding: Tensor,
    pub blocks: Vec<EncoderBlock>,
    pub ln_post: LayerNorm,
    pub n_state: usize,
    pub n_head: usize,
}

impl AudioEncoder {
    pub fn empty(dims: &ModelDimensions) -> Self {
        let n_state = dims.n_audio_state;
        let dtype = dims.dtype.clone();
        Self {
            conv1: conv1d(dims.n_mels, n_state, 3, Bias::FanIn, dtype.clone()).with_padding((1, 1)),
            conv2: conv1d(n_state, n_state, 3, Bias::FanIn, dtype.clone()).with_stride(2).with_padding((1, 1)),
            positional_embedding: sinusoids(dims.n_audio_ctx, n_state, 10_000.0).expect("sinusoidal embedding"),
            blocks: (0..dims.n_audio_layer)
                .map(|_| EncoderBlock::empty_dtype(n_state, dims.n_audio_head, dtype.clone()))
                .collect(),
            ln_post: layer_norm(n_state, dtype),
            n_state,
            n_head: dims.n_audio_head,
        }
    }

    /// Forward: mel `[B, n_mels, T]` → encoder features `[B, T/2, D]`.
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        // Cast input to the compute dtype (weights are dims.dtype; the host
        // feeds fp32 mel). Matches `model.py:48` weight.to(x.dtype) from the
        // other direction — we cast x to the weight dtype so the graph is uniform.
        let dtype = self.conv1.weight.dtype();
        let mel = mel.cast(dtype.clone());
        let x = scoped("conv1", || self.conv1.forward(&mel))?.gelu_exact()?;
        let x = scoped("conv2", || self.conv2.forward(&x))?.gelu_exact()?;

        // [B, D, T/2] → [B, T/2, D]
        let x = x.try_permute(&[0, 2, 1])?;

        // Add the positional embedding [n_audio_ctx, D]. The 1500 frames stay
        // unpadded: the attention kernel masks its ragged last tile, and
        // padding to 1536 measured 12% slower on large-v3.
        let mut x = x.try_add(&self.positional_embedding)?.cast(dtype.clone());
        for (index, block) in self.blocks.iter().enumerate() {
            x = scoped_index("blocks", index, || block.forward(&x))?;
        }

        // The features feed the decoder's cross projection, which runs in the
        // compute dtype, so they stay in it: the final norm keeps its checkpoint
        // precision and would otherwise widen the largest encoder output.
        Ok(scoped("ln_post", || norm(&self.ln_post, &x))?.cast(dtype))
    }
}
