//! The backbone's LayerNorms: tk's one-pass row kernel where it applies, the
//! graph's `LayerNorm` (a mean, a variance and an apply kernel) elsewhere.

use snafu::ResultExt;
use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::{Layer, LayerNorm};

use super::error::{Result, TkSnafu};

/// `norm(x)` over the last axis. The tk kernel takes a 16-bit `x` whose dims
/// are static or pinned by the JIT, with the parameters in its dtype; it
/// declines a row it cannot hold in registers, and every other case is the
/// graph's norm.
pub(crate) fn layer_norm(x: &Tensor, norm: &LayerNorm) -> Result<Tensor> {
    let shape = x.shape()?.to_vec();
    let dtype = x.dtype();
    let fits = norm.axis == -1
        && shape.len() >= 2
        && [DType::BFloat16, DType::Float16].contains(&dtype)
        && norm.weight.dtype() == dtype
        && norm.bias.as_ref().is_none_or(|b| b.dtype() == dtype)
        && shape.iter().all(|d| svod_tk::launch::pinned_dim(d).is_some());
    if fits && let Some(y) = svod_tk::layer_norm(x, &norm.weight, norm.bias.as_ref(), norm.eps).context(TkSnafu)? {
        // Back to `x`'s own dims, a JIT-pinned batch included.
        return Ok(y.try_reshape(shape)?);
    }
    Ok(norm.forward(x)?)
}
