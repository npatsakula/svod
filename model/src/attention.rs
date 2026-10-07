//! Multi-head attention dispatch shared by the encoder ports.
//!
//! Heads are seq-major, `[B, T, H, D]` — the layout the hand flash-attention
//! kernel consumes. That kernel runs when it applies; [SDPA] runs otherwise,
//! with the same `key_lens` key-only padding mask, so the result is correct on
//! any device.
//!
//! The kernel's mma operands are 16-bit, so it only runs on activations that
//! already are: casting fp32 down to reach it silently trades about three
//! decimal digits for the speedup (Whisper's fp32 encoder drifted from 1e-3 to
//! 1.8 against its golden), so fp32 keeps SDPA.
//!
//! [SDPA]: svod_tensor::Tensor::scaled_dot_product_attention

use svod_dtype::ScalarDType;
use svod_ir::origin::OriginScope;
use svod_tensor::Tensor;
use svod_tensor::error::Result;

/// Attention over `[B, T, H, D]` queries, keys and values, returning
/// `[B, T, H, D]`. `key_lens`, a realized `[B]` `i32` tensor, masks the keys at
/// and past each row's length.
pub(crate) fn attend(q: &Tensor, k: &Tensor, v: &Tensor, causal: bool, key_lens: Option<&Tensor>) -> Result<Tensor> {
    let sixteen_bit = matches!(q.dtype().base(), ScalarDType::Float16 | ScalarDType::BFloat16);
    if sixteen_bit && q.dim_const(-1)?.is_multiple_of(16) {
        let opts = svod_tk::FaOpts { causal, key_lens, ..Default::default() };
        if let Some(out) = svod_tk::flash_attention_with(q, k, v, opts).map_err(tk_launch_error)? {
            return Ok(out);
        }
    }
    sdpa(q, k, v, causal, key_lens)
}

/// The sequence multiple the flash-attention kernel tiles, when it can run
/// `dtype` activations on `device` at all: a caller choosing padded lengths
/// pads to it there, and only there.
pub(crate) fn flash_attention_tile(device: &svod_dtype::DeviceSpec, dtype: &svod_dtype::DType) -> Option<usize> {
    let sixteen_bit = matches!(dtype.base(), ScalarDType::Float16 | ScalarDType::BFloat16);
    (sixteen_bit && svod_tk::flash_attention_supported(device)).then_some(svod_tk::FLASH_ATTENTION_SEQUENCE_MULTIPLE)
}

/// [`attend`]'s fallback: SDPA wants head-major `[B, H, T, D]`.
fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor, causal: bool, key_lens: Option<&Tensor>) -> Result<Tensor> {
    let head_major = |t: &Tensor| t.try_permute(&[0, 2, 1, 3]);
    // `[B, T]` key validity is a property of `key_lens`, shared by every layer:
    // built outside the layer's origin scope so the layers share one mask.
    let valid = match key_lens {
        Some(lens) => {
            let _shared = OriginScope::suspend();
            Some(Tensor::sequence_mask(lens, k.dim_const(1)?)?)
        }
        None => None,
    };
    head_major(q)?
        .scaled_dot_product_attention()
        .key(&head_major(k)?)
        .value(&head_major(v)?)
        .is_causal(causal)
        .maybe_key_padding_mask(valid.as_ref())
        .call()?
        .try_permute(&[0, 2, 1, 3])
}

/// Bridge a `svod-tk` launch error into the tensor error domain. tk's launch
/// `Err` means a structurally invalid request (a caller bug — fallback-worthy
/// conditions come back as `Ok(None)` instead), so it surfaces as an IR
/// construction failure.
pub(crate) fn tk_launch_error(e: impl std::fmt::Display) -> svod_tensor::error::Error {
    svod_tensor::error::ErrorKind::IrConstruction { details: e.to_string() }.into()
}
