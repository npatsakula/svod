//! Static template of a Hopper tensor map (the TMA descriptor) over a storage
//! PARAM. A kernel receives the 128-byte descriptor by value; the backend
//! encodes it from this template and the buffer's address at every bind.

use snafu::ensure;
use svod_dtype::ScalarDType;

use crate::Result;
use crate::error::{
    TensorMapDTypeSnafu, TensorMapExtentSnafu, TensorMapRowStrideSnafu, TensorMapSwizzleSpanSnafu,
    TensorMapTileRowSnafu, TensorMapTileSnafu,
};

/// Shared-memory bank swizzle of a tensor map's box.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum TensorMapSwizzle {
    None,
    B32,
    B64,
    B128,
}

impl TensorMapSwizzle {
    /// Bytes a swizzled box row may span; `None` when unswizzled.
    pub const fn span(self) -> Option<u64> {
        match self {
            Self::None => None,
            Self::B32 => Some(32),
            Self::B64 => Some(64),
            Self::B128 => Some(128),
        }
    }
}

/// A rank-2 tiled tensor map over the storage PARAM in slot `buffer`, which
/// holds `extent[1]` rows of `extent[0]` contiguous `dtype` elements; one TMA
/// copy moves a box of `tile = [cols, rows]` elements. Interleave, L2
/// promotion and out-of-bounds fill are off (zeros), element strides are 1.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct TensorMap {
    pub buffer: usize,
    pub dtype: ScalarDType,
    pub extent: [u64; 2],
    pub tile: [u32; 2],
    pub swizzle: TensorMapSwizzle,
}

impl TensorMap {
    /// Size of the descriptor a kernel receives by value.
    pub const BYTES: usize = 128;
    /// Alignment of the descriptor in the kernel's parameter space.
    pub const ALIGN: usize = 64;

    pub fn new(
        buffer: usize,
        dtype: ScalarDType,
        extent: [u64; 2],
        tile: [u32; 2],
        swizzle: TensorMapSwizzle,
    ) -> Result<Self> {
        let map = Self { buffer, dtype, extent, tile, swizzle };
        map.validate()?;
        Ok(map)
    }

    /// The driver's encoding rules for this template, checked statically.
    pub fn validate(&self) -> Result<()> {
        let element = self.element_bytes()?;
        ensure!(
            self.extent.iter().all(|&dim| (1..=1 << 32).contains(&dim)),
            TensorMapExtentSnafu { extent: self.extent }
        );
        let row = self.row_bytes();
        // Below 2^40 too, as the driver requires: 2^32 elements of at most 8 bytes.
        ensure!(row.is_multiple_of(16), TensorMapRowStrideSnafu { bytes: row });
        ensure!(self.tile.iter().all(|&dim| (1..=256).contains(&dim)), TensorMapTileSnafu { tile: self.tile });
        let tile_row = u64::from(self.tile[0]) * element;
        ensure!(tile_row.is_multiple_of(16), TensorMapTileRowSnafu { bytes: tile_row });
        if let Some(span) = self.swizzle.span() {
            ensure!(tile_row <= span, TensorMapSwizzleSpanSnafu { bytes: tile_row, span });
        }
        Ok(())
    }

    /// Bytes of one element, for the dtypes TMA has a data type for.
    pub fn element_bytes(&self) -> Result<u64> {
        use ScalarDType::*;
        match self.dtype {
            UInt8 | UInt16 | UInt32 | Int32 | UInt64 | Int64 | Float16 | BFloat16 | Float32 | Float64 => {
                Ok(self.dtype.bytes() as u64)
            }
            dtype => TensorMapDTypeSnafu { dtype }.fail(),
        }
    }

    /// The row stride in bytes: rows are dense.
    pub fn row_bytes(&self) -> u64 {
        self.extent[0] * self.dtype.bytes() as u64
    }
}
