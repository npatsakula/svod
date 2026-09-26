//! BGE-M3 ColBERT multi-vector embedding head.
//!
//! Per-token `Linear(D, colbert_dim, bias)` applied to hidden states excluding
//! the CLS token, with padding masked to zero. Optional L2 normalization.
//!
//! Weights are loaded from `colbert_linear.pt` (PyTorch pickle with bare
//! `weight` / `bias` keys).

use svod_dtype::DType;
use svod_tensor::nn::Module;
use svod_tensor::{Tensor, s};

use crate::init::{fan_in_uniform, zeros};
use crate::state::{self, StateDict};
use crate::xlm_roberta::error::Result;

#[derive(Clone, Module)]
pub struct ColbertHead {
    pub weight: Tensor,
    pub bias: Tensor,
    pub colbert_dim: usize,
    pub normalize: bool,
}

impl ColbertHead {
    pub fn empty(hidden_size: usize, colbert_dim: usize, dtype: DType) -> Self {
        Self {
            weight: fan_in_uniform(&[colbert_dim, hidden_size], hidden_size, dtype.clone()),
            bias: zeros(&[colbert_dim], dtype),
            colbert_dim,
            normalize: true,
        }
    }

    /// Load from a `colbert_linear.pt` checkpoint (bare `weight`/`bias` keys).
    pub fn from_pytorch_bin(path: &std::path::Path, colbert_dim: usize, dtype: DType) -> Result<Self> {
        let sd = crate::wespeaker::pickle::load_flat_pytorch_bin(path, "")
            .map_err(|e| crate::xlm_roberta::Error::Pickle { source: Box::new(e) })?;
        Self::from_state_dict(&sd, "", colbert_dim, dtype)
    }

    /// Build from a preloaded state dict, casting to `dtype`.
    pub fn from_state_dict(sd: &StateDict, prefix: &str, colbert_dim: usize, dtype: DType) -> Result<Self> {
        let sd = state::cast_all(sd, dtype.clone())?;
        let mut head = Self::empty(0, colbert_dim, dtype);
        head.load_state_dict(&sd, prefix)?;
        Ok(head)
    }

    /// Forward. `hidden`: `(B, L, D)`, `attention_mask`: optional `(B, L)` bool
    /// where `true` = real token. Returns `(B, L-1, colbert_dim)` — the CLS
    /// token (position 0) is dropped.
    pub fn forward(&self, hidden: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let hidden_no_cls = hidden.getitem(s![.., 1.., ..])?;

        let vecs = hidden_no_cls.linear().weight(&self.weight).bias(&self.bias).call()?;

        let vecs = match attention_mask {
            Some(m) => {
                let m_no_cls = m.getitem(s![.., 1..])?;
                let m_3d = m_no_cls.cast(vecs.dtype()).try_unsqueeze(-1)?;
                vecs.try_mul(&m_3d)?
            }
            None => vecs,
        };

        if self.normalize { Ok(vecs.lp_normalize(-1, 2)?) } else { Ok(vecs) }
    }
}
