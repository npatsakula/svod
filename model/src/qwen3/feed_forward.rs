//! Qwen3 gated feed-forward (SwiGLU): `down(silu(gate(x)) * up(x))`, no biases.
//!
//! The checkpoint stores `gate_proj` and `up_proj` separately; they read the
//! same input, so they are stacked into one `[2I, H]` weight at load and the
//! GEMM that reads it folds `silu(gate)·up` into its own epilogue, writing
//! `[M, I]` and never forming the `[M, 2I]` intermediate
//! ([`svod_tk::Epilogue::SwiGlu`]). That epilogue pairs each gate column with
//! its up column **inside one wave's accumulator**, so the stacked rows are
//! interleaved in blocks of the weight's device's [`svod_tk::swiglu_pair_width`]
//! — `[g0.., u0.., g1.., u1.., …]` — at load time; on a device without the hand
//! GEMM they stay plainly stacked. The state dict keeps the published two-key
//! layout.

use svod_dtype::DType;
use svod_ir::SInt;
use svod_tensor::Tensor;
use svod_tensor::nn::{Module, StateDict, get_tensor, prefixed};

use crate::init::fan_in_uniform;

use super::error::Result;
use super::linear::{Projected, linear, linear_add};

#[derive(Clone)]
pub struct Qwen3MLP {
    pub intermediate_size: usize,
    /// `gate_proj.weight` over `up_proj.weight`, `[2I, H]` — in alternating
    /// `pair`-row gate/up blocks when [`Qwen3MLP::pair`] is set, plainly stacked
    /// otherwise.
    pub gate_up_weight: Tensor,
    pub down_weight: Tensor,
    /// The gate/up row-block width `gate_up_weight` is interleaved in; `None`
    /// when the rows are plainly stacked (no fused epilogue on this device).
    pair: Option<usize>,
}

impl Qwen3MLP {
    pub fn empty(hidden_size: usize, intermediate_size: usize, dtype: DType) -> Self {
        let gate_up_weight = fan_in_uniform(&[2 * intermediate_size, hidden_size], hidden_size, dtype.clone());
        let down_weight = fan_in_uniform(&[hidden_size, intermediate_size], intermediate_size, dtype);
        Self { intermediate_size, gate_up_weight, down_weight, pair: None }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.forward_into(x, None)?.into_tensor())
    }

    /// The MLP over `x`, with `residual` folded into the `down_proj` GEMM's
    /// epilogue when that kernel takes it (see [`linear_add`]).
    pub(crate) fn forward_into(&self, x: &Tensor, residual: Option<&Tensor>) -> Result<Projected> {
        linear_add(&self.activation(x)?, &self.down_weight, residual)
    }

    /// `silu(gate(x)) * up(x)` — one GEMM whose epilogue writes it, or the fused
    /// GEMM plus the split / silu / multiply graph where that epilogue declines.
    fn activation(&self, x: &Tensor) -> Result<Tensor> {
        if let Some(pair) = self.pair
            && let Some(act) = super::linear::linear_swiglu(x, &self.gate_up_weight, pair)?
        {
            return Ok(act);
        }
        let (gate, up) = self.split_gate_up(&linear(x, &self.gate_up_weight)?)?;
        Ok(gate.silu()?.try_mul(&up)?)
    }

    /// The `[.., 2I]` fused GEMM output split into `(gate, up)`, honoring the
    /// row interleave `pair` describes.
    fn split_gate_up(&self, gate_up: &Tensor) -> Result<(Tensor, Tensor)> {
        let i = self.intermediate_size;
        let Some(pair) = self.pair else {
            let halves = gate_up.split(&[i, i], -1)?;
            return Ok((halves[0].clone(), halves[1].clone()));
        };
        // `[.., 2I]` → `[.., I/pair, 2, pair]`: axis −2 is the gate/up selector.
        let lead: Vec<SInt> = gate_up.shape()?[..gate_up.shape()?.len() - 1].to_vec();
        let blocked: Vec<SInt> =
            lead.iter().cloned().chain([SInt::Const(i / pair), SInt::Const(2), SInt::Const(pair)]).collect();
        let flat: Vec<SInt> = lead.into_iter().chain([SInt::Const(i)]).collect();
        let blocked = gate_up.try_reshape(blocked)?;
        let half = |which: usize| -> Result<Tensor> {
            Ok(blocked.narrow(-2, which, 1)?.contiguous().try_reshape(flat.clone())?)
        };
        Ok((half(0)?, half(1)?))
    }

    /// `gate_proj.weight` (`which = 0`) or `up_proj.weight` (`which = 1`) read
    /// back out of `gate_up_weight` — the inverse of the load-time interleave.
    fn published_half(&self, which: usize) -> Tensor {
        let i = self.intermediate_size;
        let Some(pair) = self.pair else {
            return self.gate_up_weight.narrow(0, which * i, i).expect("[2I, H] weight");
        };
        let h = self.gate_up_weight.dim_const(1).expect("[2I, H] weight");
        let dims = |v: [usize; 4]| v.map(|d| d as isize).to_vec();
        self.gate_up_weight
            .try_reshape(dims([i / pair, 2, pair, h]))
            .and_then(|t| t.narrow(1, which, 1))
            .map(|t| t.contiguous())
            .and_then(|t| t.try_reshape(vec![i as isize, h as isize]))
            .expect("un-interleave the gate/up rows")
    }
}

/// `gate` over `up` as one `[2I, H]` buffer (a lazy `cat` would be re-read part
/// by part inside the GEMM's K loop: 2x the weight loads, half the throughput),
/// in alternating `pair`-row gate/up blocks when `pair` is set.
pub(crate) fn pair_rows(gate: &Tensor, up: &Tensor, pair: Option<usize>) -> svod_tensor::error::Result<Tensor> {
    let (i, h) = (gate.dim_const(0)?, gate.dim_const(1)?);
    let stacked = Tensor::cat(&[gate, up], 0)?;
    let Some(pair) = pair else { return Ok(stacked.contiguous()) };
    stacked
        .try_reshape([2, (i / pair) as isize, pair as isize, h as isize])?
        .try_permute(&[1, 0, 2, 3])?
        .contiguous()
        .try_reshape([(2 * i) as isize, h as isize])
}

impl Module for Qwen3MLP {
    fn write_state(&self, prefix: &str, out: &mut StateDict) {
        out.insert(prefixed(prefix, "gate_proj.weight"), self.published_half(0));
        out.insert(prefixed(prefix, "up_proj.weight"), self.published_half(1));
        out.insert(prefixed(prefix, "down_proj.weight"), self.down_weight.clone());
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> svod_tensor::error::Result<()> {
        let gate = get_tensor(sd, &prefixed(prefix, "gate_proj.weight"))?;
        let up = get_tensor(sd, &prefixed(prefix, "up_proj.weight"))?;
        let i = self.intermediate_size;
        // The epilogue pairs a gate column with its up column inside one wave's
        // accumulator, so their rows must be interleaved in blocks of `pair` —
        // the width the GEMM tiles of the weight's device read.
        self.pair = svod_tk::swiglu_pair_width(&gate.device()).filter(|p| i.is_multiple_of(*p));
        self.gate_up_weight = pair_rows(&gate, &up, self.pair)?;
        self.gate_up_weight.realize()?;
        self.down_weight = get_tensor(sd, &prefixed(prefix, "down_proj.weight"))?;
        Ok(())
    }
}
