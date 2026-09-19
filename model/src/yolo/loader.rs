//! Shared loader helpers for YOLO models.

use std::path::Path;

use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::state::{self, StateDict};

use super::blocks::conv::YOLO_BN_EPS;
use super::error::Result;

/// Download `model.safetensors` from HuggingFace Hub.
pub fn download_safetensors(model_id: &str, revision: &str) -> Result<std::path::PathBuf> {
    let repo = crate::hub::HubRepo::open(model_id, revision)?;
    Ok(repo.get("model.safetensors")?)
}

/// Load a checkpoint and strip the `model.` prefix, returning a clean state
/// dict. The layers read PyTorch's own keys, so nothing else is renamed.
pub fn prepare_state_dict(path: &Path) -> Result<StateDict> {
    Ok(strip_model_prefix(&state::load_safetensors(path)?))
}

/// Cast the float parameters of `sd` to `dtype`; integer buffers (`num_batches_tracked`
/// and friends) keep theirs, since narrowing those would be meaningless.
///
/// Lazy: every cast is just a CAST node over the source buffer. Anything that
/// will run more than once wants [`load_weights`] instead.
pub fn cast_weights(sd: &StateDict, dtype: &DType) -> StateDict {
    sd.iter()
        .map(|(key, tensor)| {
            let cast = if tensor.dtype().is_float() && tensor.dtype() != *dtype {
                tensor.cast(dtype.clone())
            } else {
                tensor.clone()
            };
            (key.clone(), cast)
        })
        .collect()
}

/// Fold every `YoloConv`'s batch norm into its convolution: `conv.weight`
/// scales by `gamma / sqrt(var + eps)` per output channel and a `conv.bias` of
/// `beta - mean * scale` appears beside it, which [`YoloConv`] takes as the
/// sign that the norm is already applied. The conv kernel then reads two
/// buffers instead of six and its epilogue is a bias and the activation, as
/// Ultralytics' `fuse()` leaves it. Folded in f32, before any narrowing.
///
/// [`YoloConv`]: super::blocks::conv::YoloConv
pub fn fold_batchnorm(sd: &StateDict) -> Result<StateDict> {
    let mut out = sd.clone();
    for (key, weight) in sd {
        let Some(prefix) = key.strip_suffix("conv.weight") else { continue };
        let bn = |name: &str| sd.get(&format!("{prefix}bn.{name}")).map(|t| t.cast(DType::Float32));
        let (Some(gamma), Some(beta), Some(mean), Some(var)) =
            (bn("weight"), bn("bias"), bn("running_mean"), bn("running_var"))
        else {
            continue;
        };
        let scale = var.try_add(Tensor::const_(YOLO_BN_EPS, DType::Float32))?.try_rsqrt()?.try_mul(&gamma)?;
        let bias = beta.try_sub(&mean.try_mul(&scale)?)?;
        let cout = weight.dim_const(0)?;
        let scale = scale.try_reshape(vec![cout as isize, 1, 1, 1])?;
        let folded = weight.cast(DType::Float32).try_mul(&scale)?.cast(weight.dtype());
        out.insert(key.clone(), folded);
        out.insert(format!("{prefix}conv.bias"), bias.cast(weight.dtype()));
    }
    Ok(out)
}

/// [`cast_weights`], materialised — the checkpoint path.
///
/// Left lazy, every conv would re-read the checkpoint's bytes and convert per
/// element on each pass: nominally f16, while still paying f32 bandwidth.
/// Realizing here makes it a one-off conversion at load, at a transient peak of
/// both copies until the originals drop (the weight cache holds them by `Weak`).
///
/// Narrowing to f16 is bit-exact for an Ultralytics checkpoint: the trainer
/// saves fp16 and the converters widen it on the way in, so the f32 on disk
/// carries no more information than the f16 it came from.
pub fn load_weights(sd: &StateDict, dtype: &DType) -> Result<StateDict> {
    let cast = cast_weights(&fold_batchnorm(sd)?, dtype);
    for tensor in cast.values() {
        tensor.realize()?;
    }
    Ok(cast)
}

/// Bring a freshly built model's placeholder weights to `dtype`.
///
/// `with_zero_weights` mints f32 placeholders. That is invisible for a
/// checkpoint load, where every one is replaced, but not for zero-weight runs
/// and graph-shape tests: f32 weights under an f16 compute dtype promote the
/// first conv straight back to f32 (`least_upper_dtype(f16, f32)` is f32) and
/// silently revert the whole graph. Stays lazy — nothing here is worth
/// materialising.
pub fn cast_placeholders<M: svod_tensor::nn::Module>(model: &mut M, dtype: &DType) -> Result<()> {
    if *dtype == DType::Float32 {
        return Ok(());
    }
    let sd = cast_weights(&model.state_dict(""), dtype);
    model.load_state_dict(&sd, "")?;
    Ok(())
}

/// Strip the `model.` prefix from all keys if present (Ultralytics wraps
/// everything in `self.model`).
pub fn strip_model_prefix(sd: &StateDict) -> StateDict {
    if sd.keys().any(|k| k.starts_with("model.")) {
        sd.iter()
            .map(|(k, v)| {
                let k2 = k.strip_prefix("model.").unwrap_or(k);
                (k2.to_string(), v.clone())
            })
            .collect()
    } else {
        sd.clone()
    }
}
