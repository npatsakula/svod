//! Qwen3 gated feed-forward (SwiGLU): `down(silu(gate(x)) * up(x))`, no biases.
//!
//! The checkpoint stores `gate_proj` and `up_proj` separately; they read the
//! same input, so they are stacked gate over up into one `[2I, H]` weight at
//! load, and the GEMM that reads it writes `silu(gate)·up` from its epilogue
//! where the tile kernel runs ([`svod_tk3::ops::Linear::gated`]). The state
//! dict keeps the published two-key layout.

use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::{Module, StateDict, get_tensor, prefixed};
use svod_tk3::ops::{self, Act};

use crate::init::fan_in_uniform;

use super::error::Result;

#[derive(Clone)]
pub struct Qwen3MLP {
    pub intermediate_size: usize,
    /// `gate_proj.weight` over `up_proj.weight`, `[2I, H]`.
    pub gate_up_weight: Tensor,
    pub down_weight: Tensor,
}

impl Qwen3MLP {
    pub fn empty(hidden_size: usize, intermediate_size: usize, dtype: DType) -> Self {
        let gate_up_weight = fan_in_uniform(&[2 * intermediate_size, hidden_size], hidden_size, dtype.clone());
        let down_weight = fan_in_uniform(&[hidden_size, intermediate_size], intermediate_size, dtype);
        Self { intermediate_size, gate_up_weight, down_weight }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_into(x, None)
    }

    /// The MLP over `x`, `residual` added in the `down_proj` GEMM's epilogue.
    pub(crate) fn forward_into(&self, x: &Tensor, residual: Option<&Tensor>) -> Result<Tensor> {
        let act =
            ops::linear(x, &self.gate_up_weight, ops::Linear { act: Act::Silu, gated: true, ..Default::default() })?;
        Ok(ops::linear(&act, &self.down_weight, ops::Linear { residual, ..Default::default() })?)
    }
}

impl Module for Qwen3MLP {
    fn write_state(&self, prefix: &str, out: &mut StateDict) {
        let i = self.intermediate_size;
        let half = |which: usize| self.gate_up_weight.narrow(0, which * i, i).expect("[2I, H] weight");
        out.insert(prefixed(prefix, "gate_proj.weight"), half(0));
        out.insert(prefixed(prefix, "up_proj.weight"), half(1));
        out.insert(prefixed(prefix, "down_proj.weight"), self.down_weight.clone());
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> svod_tensor::error::Result<()> {
        let gate = get_tensor(sd, &prefixed(prefix, "gate_proj.weight"))?;
        let up = get_tensor(sd, &prefixed(prefix, "up_proj.weight"))?;
        // One buffer: a lazy `cat` would be re-read part by part inside the
        // GEMM's K loop (2x the weight loads, half the throughput).
        self.gate_up_weight = Tensor::cat(&[&gate, &up], 0)?.contiguous();
        self.gate_up_weight.realize()?;
        self.down_weight = get_tensor(sd, &prefixed(prefix, "down_proj.weight"))?;
        Ok(())
    }
}
