//! Shared-tile (ST) swizzles.
//!
//! A swizzle remaps the `(row, col)` of an element within a base tile to avoid
//! LDS bank conflicts. The MVP path uses [`Swizzle::Identity`] (`ST_16X16`); the
//! XOR variants port the HipKittens scheme (`st.cuh:88-97`): the element's byte
//! offset within the tile is XORed with `((addr % repeat) >> 7) << 3`, a
//! bijection applied identically on every LDS store and load, so it never
//! changes the numeric result — it only re-lays-out the banks.

use std::sync::Arc;

use svod_dtype::ScalarDType;
use svod_ir::UOp;

use crate::index::cidx;

/// The five predefined ST base-tile swizzles (see tinygrad `tiles.py`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Swizzle {
    /// No remapping (`ST_16X16`).
    Identity,
    /// `ST_16X16_SWIZZLED` (bf16 XOR).
    Sw16x16,
    /// `ST_32X32` (bf16 double XOR).
    Sw32x32,
    /// `ST_16X32` (bf16 XOR).
    Sw16x32,
    /// `ST_32X16` (bf16 XOR).
    Sw32x16,
    /// `ST_16X16_MMA`: the 16-byte chunk index XORed with row bit 2
    /// (`col ^= (16/itemsize)·((row/4)%2)`). Conflict-free for the `mma.sync`
    /// fragment gather (per register a warp touches 8 rows × one 16-byte chunk:
    /// unswizzled rows `r` and `r+4` share banks in a 32-byte row), for
    /// `ldmatrix` (the same 8-row × 16-byte phase) and for 16-byte `cp.async`
    /// writers; 16-byte groups stay contiguous.
    Sw16x16Mma,
}

/// HipKittens `swizzle_bytes` (`st.cuh:74-86`): the bank-conflict-avoiding XOR
/// period in bytes, selected from the tile's underlying width (in 16-col
/// fragments) and element size. `cols` is the (per-fragment) column count.
///
/// # Panics
/// `itemsize` must be 1/2/4 bytes — i.e. a swizzled shared (`ST`) tile's dtype
/// must be a 1/2/4-byte type (bf16/f16/f32 in practice). An 8-byte element
/// (f64/i64) panics; this is a kernel-authoring precondition, the USE-face
/// kernels only allocate bf16/f32 LDS tiles.
fn swizzle_bytes(cols: usize, itemsize: i64) -> i64 {
    let uw = cols / 16; // underlying width in 16-col tiles
    match itemsize {
        1 | 2 => {
            if uw.is_multiple_of(4) {
                128
            } else if uw.is_multiple_of(2) {
                64
            } else {
                32
            }
        }
        4 => {
            if uw.is_multiple_of(2) {
                128
            } else {
                64
            }
        }
        other => panic!("swizzle: unsupported itemsize {other} (bf16/f32 only)"),
    }
}

impl Swizzle {
    /// The XOR swizzle period in bytes for a `cols`-wide / `itemsize`-byte base
    /// tile (`None` for [`Swizzle::Identity`], which has no period). Used by the
    /// vectorized fill; asserts 16-byte group alignment.
    pub(crate) fn period_bytes(&self, cols: usize, itemsize: i64) -> Option<i64> {
        match self {
            Swizzle::Identity => None,
            Swizzle::Sw16x16Mma => Some(256), // the row-bit-2 pattern repeats every 8 rows of 32 bytes
            _ => Some(swizzle_bytes(cols, itemsize)),
        }
    }

    /// Whether every aligned 16-byte column chunk of a row stays one contiguous,
    /// 16-byte-aligned run under the swizzle — the `ldmatrix` row / `cp.async`
    /// copy unit. The HipKittens XOR variants permute at 8-byte granularity
    /// (`st.cuh:96` `<< 3`), so only the identity and the chunk-granular
    /// [`Swizzle::Sw16x16Mma`] qualify.
    pub fn keeps_16b_chunks(&self) -> bool {
        matches!(self, Swizzle::Identity | Swizzle::Sw16x16Mma)
    }

    /// Map `(row, col)` within a base tile of `cols` columns to swizzled
    /// `(row, col)`. `scalar` is the element type (the XOR variants depend on
    /// `itemsize`). The mapping is a bijection on `[0,rows)×[0,cols)`, so a write
    /// and a later read at the same logical `(row, col)` hit the same slot.
    ///
    /// # Panics
    /// For a non-[`Swizzle::Identity`] variant, panics if the scalar itemsize is
    /// not 1, 2, or 4 bytes (only bf16/f16/f32 LDS tiles are swizzled).
    pub fn swizzle_rc(&self, row: Arc<UOp>, col: Arc<UOp>, cols: usize, scalar: ScalarDType) -> (Arc<UOp>, Arc<UOp>) {
        match self {
            Swizzle::Identity => (row, col),
            Swizzle::Sw16x16Mma => {
                let chunk = 16 / scalar.bytes() as i64;
                (row.clone(), col.xor(&row.shr(&cidx(2)).mod_(&cidx(2)).mul(&cidx(chunk))))
            }
            Swizzle::Sw16x16 | Swizzle::Sw32x32 | Swizzle::Sw16x32 | Swizzle::Sw32x16 => {
                let cols_i = cols as i64;
                let itemsize = scalar.bytes() as i64;
                let repeat = swizzle_bytes(cols, itemsize) << 4; // st.cuh:87
                // Row-major element offset within the fragment, then its byte
                // address; XOR the byte address per st.cuh:96-97 and divide back.
                let e = row.mul(&cidx(cols_i)).add(&col);
                let byte = e.mul(&cidx(itemsize));
                let sw_bytes = byte.mod_(&cidx(repeat)).shr(&cidx(7)).shl(&cidx(3));
                let e2 = e.xor(&sw_bytes.floor_div(&cidx(itemsize)));
                (e2.floor_div(&cidx(cols_i)), e2.mod_(&cidx(cols_i)))
            }
        }
    }
}
