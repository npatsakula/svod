//! `x @ w^T` for the projections: the hand GEMM when it applies, else the
//! generic `Tensor::linear`.

use snafu::ResultExt;
use svod_dtype::ScalarDType;
use svod_tensor::Tensor;

use super::error::{Result, TkSnafu};

/// Whether the hand GEMM can take these operands: concrete 16-bit, matching
/// dtypes. The tile grid is the kernel's own call (`Ok(None)`).
fn fusable(x: &Tensor, w: &Tensor) -> bool {
    matches!(x.dtype().base(), ScalarDType::BFloat16 | ScalarDType::Float16)
        && x.dtype() == w.dtype()
        && x.shape().is_ok_and(|s| s.iter().all(|d| d.as_const().is_some()))
}

/// `x` `[B, L, K]` · `w` `[N, K]`ᵀ → `[B, L, N]`. The tk kernel takes concrete
/// 16-bit operands and returns `None` off its tile grid or arch; every other
/// case is the generic GEMM.
///
/// The generic operand is materialized: left lazy, its producer fuses into
/// the GEMM's loads and is recomputed per output tile (the norm-and-scale
/// measured 1.5x slower that way, the SwiGLU 5x). The kernel launch does the
/// same for its inputs, and binds an already realized one without a copy.
pub(crate) fn linear(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    if fusable(x, w)
        && let Some(y) = svod_tk::gemm_nt(x, w).context(TkSnafu)?
    {
        return Ok(y);
    }
    Ok(x.contiguous().linear().weight(w).call()?)
}

/// `y = silu(gate)·up` off the fused `[2I, K]` gate/up weight, computed in the
/// GEMM's epilogue so the `[M, 2I]` intermediate is never written — or `None`
/// where the hand kernel declines (an unsupported device, a shape off its tile
/// grid, or a tile that does not read `pair`-row gate/up blocks).
pub(crate) fn linear_swiglu(x: &Tensor, w: &Tensor, pair: usize) -> Result<Option<Tensor>> {
    if !fusable(x, w) {
        return Ok(None);
    }
    svod_tk::gemm_nt_with_epilogue(x, w, svod_tk::Epilogue::SwiGlu { pair }).context(TkSnafu)
}

/// A projection's output, with the residual add either already folded into the
/// GEMM that produced it or still owed to whatever consumes the stream next.
pub(crate) enum Projected {
    /// `x·wᵀ + residual`, summed inside the GEMM's epilogue.
    Summed(Tensor),
    /// `x·wᵀ` alone — the add is still pending.
    Pending(Tensor),
}

impl Projected {
    /// The tensor either way — the product where the add is still pending, the
    /// sum where the epilogue took it.
    pub(crate) fn into_tensor(self) -> Tensor {
        match self {
            Projected::Summed(t) | Projected::Pending(t) => t,
        }
    }
}

/// `x·wᵀ (+ residual)`: `residual` rides the GEMM's epilogue when the hand
/// kernel takes it, so the sum is written once and the norm that reads it is the
/// two-pass `rms_norm` rather than the four-pass `add_rms_norm`. With no
/// residual, or where the kernel declines, this is [`linear`] with the add left
/// [`Projected::Pending`].
pub(crate) fn linear_add(x: &Tensor, w: &Tensor, residual: Option<&Tensor>) -> Result<Projected> {
    if let Some(residual) = residual
        && fusable(x, w)
        && let Some(y) = svod_tk::gemm_nt_with_epilogue(x, w, svod_tk::Epilogue::Add(residual)).context(TkSnafu)?
    {
        return Ok(Projected::Summed(y));
    }
    Ok(Projected::Pending(linear(x, w)?))
}
