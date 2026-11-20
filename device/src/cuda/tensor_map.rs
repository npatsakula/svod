//! Tensor-map PARAMs encoded with `cuTensorMapEncodeTiled` wherever a
//! kernarg blob is packed. A descriptor embeds its buffer's address, so every
//! bind re-encodes it: one host-side driver call, no device work, and nothing
//! cached to go stale when a graph replay rebinds the buffer.

use svod_ir::{TensorMap, TensorMapSwizzle};

use super::sys::{Api, CUtensorMap, tensor_map as raw};
use crate::Result;
use crate::hcq::ClikeKernargLayout;

/// Pack `buffers`/`vals` into `blob` through `layout`, encoding its tensor
/// maps; the caller has entered the device's context.
pub(crate) fn pack(
    api: &Api,
    layout: &ClikeKernargLayout,
    blob: &mut [u8],
    buffers: &[u64],
    vals: &[i64],
) -> Result<usize> {
    layout.pack_encoded(blob, buffers, vals, &mut |map, address| encode(api, map, address))
}

/// The descriptor of `map` over the buffer at `address`.
pub(crate) fn encode(api: &Api, map: &TensorMap, address: u64) -> Result<[u8; TensorMap::BYTES]> {
    let mut descriptor = CUtensorMap([0; 16]);
    let strides = [map.row_bytes()];
    let element_strides = [1u32; 2];
    // SAFETY: a 128-byte-aligned descriptor and rank-2 arrays that outlive
    // the call; the driver only reads `address`, which it encodes.
    unsafe {
        (api.tensor_map_encode_tiled)(
            &mut descriptor,
            data_type(map),
            2,
            address as *mut std::ffi::c_void,
            map.extent.as_ptr(),
            strides.as_ptr(),
            map.tile.as_ptr(),
            element_strides.as_ptr(),
            raw::INTERLEAVE_NONE,
            swizzle(map.swizzle),
            raw::L2_PROMOTION_NONE,
            raw::FLOAT_OOB_FILL_NONE,
        )
    }
    .check("cuTensorMapEncodeTiled")?;
    Ok(std::array::from_fn(|byte| descriptor.0[byte / 8].to_ne_bytes()[byte % 8]))
}

fn data_type(map: &TensorMap) -> i32 {
    use svod_dtype::ScalarDType::*;
    match map.dtype {
        UInt8 => raw::DATA_TYPE_UINT8,
        UInt16 => raw::DATA_TYPE_UINT16,
        UInt32 => raw::DATA_TYPE_UINT32,
        Int32 => raw::DATA_TYPE_INT32,
        UInt64 => raw::DATA_TYPE_UINT64,
        Int64 => raw::DATA_TYPE_INT64,
        Float16 => raw::DATA_TYPE_FLOAT16,
        Float32 => raw::DATA_TYPE_FLOAT32,
        Float64 => raw::DATA_TYPE_FLOAT64,
        BFloat16 => raw::DATA_TYPE_BFLOAT16,
        dtype => unreachable!("ABI validation admits no tensor map of {dtype:?}"),
    }
}

fn swizzle(swizzle: TensorMapSwizzle) -> i32 {
    match swizzle {
        TensorMapSwizzle::None => raw::SWIZZLE_NONE,
        TensorMapSwizzle::B32 => raw::SWIZZLE_32B,
        TensorMapSwizzle::B64 => raw::SWIZZLE_64B,
        TensorMapSwizzle::B128 => raw::SWIZZLE_128B,
    }
}
