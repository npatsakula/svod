//! `x·wᵀ (+ residual)` for the projections: tk's hand GEMM where it applies, the
//! generic `Tensor::linear` elsewhere.

use snafu::ResultExt;
use svod_dtype::DType;
use svod_ir::SInt;
use svod_tensor::Tensor;

use super::error::{Result, TkSnafu};

/// `x` `[.., K]` · `w` `[N, K]`ᵀ, plus `residual` `[.., N]` when given: tk's
/// kernel ([`tk_linear`]) with the residual folded into its store, else the
/// generic GEMM with the add after it.
///
/// `x` is materialized either way — the kernel launch copies a lazy operand into
/// its own buffer, the generic path realizes it here: left lazy, its producer
/// fuses into the GEMM's K loop and is recomputed once per output feature (the
/// GEGLU's erf made the MLP's `Wo` 154 µs instead of 74 at 1×512).
pub(crate) fn linear(x: &Tensor, w: &Tensor, residual: Option<&Tensor>) -> Result<Tensor> {
    if let Some(y) = tk_linear(x, w, residual.map_or(svod_tk::Epilogue::Plain, svod_tk::Epilogue::Add))? {
        return Ok(y);
    }
    let y = x.contiguous().linear().weight(w).call()?;
    Ok(match residual {
        Some(residual) => y.try_add(residual)?,
        None => y,
    })
}

/// `x` `[.., K]` · `w` `[N, K]`ᵀ with `epilogue` folded into the store, on
/// tk's hand GEMM, back in `x`'s leading dims (a JIT-pinned batch included).
/// `None` where it declines: operands that are not 16-bit, dims neither static
/// nor pinned by the JIT, or a shape off its tile grid, arch or epilogue.
pub(crate) fn tk_linear(x: &Tensor, w: &Tensor, epilogue: svod_tk::Epilogue<&Tensor>) -> Result<Option<Tensor>> {
    let shape = x.shape()?.to_vec();
    let sixteen_bit = [DType::BFloat16, DType::Float16].contains(&x.dtype()) && x.dtype() == w.dtype();
    if !sixteen_bit || shape.iter().any(|d| svod_tk::launch::pinned_dim(d).is_none()) {
        return Ok(None);
    }
    let Some(y) = svod_tk::gemm_nt_with_epilogue(x, w, epilogue).context(TkSnafu)? else { return Ok(None) };
    let dims: Vec<SInt> = shape[..shape.len() - 1].iter().cloned().chain([w.dim(0)?]).collect();
    Ok(Some(y.try_reshape(dims)?))
}
