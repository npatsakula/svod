//! Tile shape descriptors and layouts.
//!
//! These are the pure, data-only building blocks shared by every tile kind. A
//! [`BaseShape`] is one WMMA-sized fragment (e.g. 16×16); a full tile is a grid
//! of base shapes. The concrete tile wrappers (GL/ST/RT/RV) that bind a buffer
//! and a [`crate::Kernel`] live alongside the builder.
//!
//! `elements_per_thread` is carried **explicitly** per shape rather than derived
//! `num_elements / WARP_THREADS`, because it is a function of the matrix-core
//! fragment layout, which differs by arch: CDNA wave64 16×16 = 4/lane; RDNA
//! wave32 = 8/lane for the accumulator and **16/lane for the (replicated) WMMA
//! inputs** (256/32 × the 0-15≡16-31 wave-half replication). The `_W32_*`
//! constants below are the RDNA (gfx11) shapes; the unsuffixed ones are gfx942.

pub use crate::layout::LaneMap;
use crate::swizzle::Swizzle;

/// Register-tile element layout within a warp.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TileLayout {
    Row,
    Col,
}

/// Register-vector layout.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VecLayout {
    Ortho,
}

/// A WMMA-sized base fragment, carrying its per-lane element count (`ept`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BaseShape {
    pub rows: usize,
    pub cols: usize,
    /// Elements each lane holds for one base fragment — arch/layout-specific (see
    /// the module docs), NOT always `num_elements / wave_size` (RDNA inputs are
    /// replicated, so `ept > num_elements / wave_size`).
    pub ept: usize,
}

impl BaseShape {
    pub const fn num_elements(&self) -> usize {
        self.rows * self.cols
    }
    /// Elements each thread (lane) holds for one base fragment.
    pub const fn elements_per_thread(&self) -> usize {
        self.ept
    }
}

/// Shared-tile base fragment: a [`BaseShape`] plus its LDS [`Swizzle`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct STBaseShape {
    pub base: BaseShape,
    pub swizzle: Swizzle,
}

/// Register-tile base fragment: a [`BaseShape`] plus its per-lane [`LaneMap`]
/// (which element of the fragment each lane's register `j` holds).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RTBaseShape {
    pub base: BaseShape,
    pub map: LaneMap,
}

impl RTBaseShape {
    pub const fn elements_per_thread(&self) -> usize {
        self.base.elements_per_thread()
    }
}

// ── gfx942 (CDNA3, wave64) base shapes — ept = num_elements / 64 ──────────────

// Predefined shared-tile base shapes.
pub const ST_16X16: STBaseShape =
    STBaseShape { base: BaseShape { rows: 16, cols: 16, ept: 4 }, swizzle: Swizzle::Identity };
pub const ST_16X16_SWIZZLED: STBaseShape =
    STBaseShape { base: BaseShape { rows: 16, cols: 16, ept: 4 }, swizzle: Swizzle::Sw16x16 };
pub const ST_32X32: STBaseShape =
    STBaseShape { base: BaseShape { rows: 32, cols: 32, ept: 16 }, swizzle: Swizzle::Sw32x32 };
pub const ST_16X32: STBaseShape =
    STBaseShape { base: BaseShape { rows: 16, cols: 32, ept: 8 }, swizzle: Swizzle::Sw16x32 };
pub const ST_32X16: STBaseShape =
    STBaseShape { base: BaseShape { rows: 32, cols: 16, ept: 8 }, swizzle: Swizzle::Sw32x16 };

// Predefined register-tile base shapes.
pub const RT_16X16: RTBaseShape =
    RTBaseShape { base: BaseShape { rows: 16, cols: 16, ept: 4 }, map: LaneMap::Strided { stride: 4 } };
pub const RT_32X32: RTBaseShape =
    RTBaseShape { base: BaseShape { rows: 32, cols: 32, ept: 16 }, map: LaneMap::Strided { stride: 4 } };
pub const RT_16X32: RTBaseShape =
    RTBaseShape { base: BaseShape { rows: 16, cols: 32, ept: 8 }, map: LaneMap::Strided { stride: 8 } };
pub const RT_32X16: RTBaseShape =
    RTBaseShape { base: BaseShape { rows: 32, cols: 16, ept: 8 }, map: LaneMap::Strided { stride: 8 } };

// ── RDNA (gfx11, wave32) base shapes — for the gfx1151 WMMA matmul ────────────
//
// Accumulator: ept = 256/32 = 8, [`LaneMap::Interleaved`] (the RDNA3 WMMA f32
// even/odd row map; NOT the gfx12/CK contiguous layout). Inputs: ept = 16
// (replicated across wave-halves), stride = 0 ⇒ lane = M/N, the 16 elements = the
// K run, identical for lanes L and L+16.

/// LDS strip fragment for the wave32 matmul (`ept = 256/32 = 8`).
pub const ST_16X16_SWIZZLED_W32: STBaseShape =
    STBaseShape { base: BaseShape { rows: 16, cols: 16, ept: 8 }, swizzle: Swizzle::Sw16x16 };
/// wave32 WMMA f32 accumulator fragment: even/odd row interleave.
pub const RT_16X16_W32_ACC: RTBaseShape =
    RTBaseShape { base: BaseShape { rows: 16, cols: 16, ept: 8 }, map: LaneMap::Interleaved };
/// wave32 WMMA input fragment: 16 K/lane, replicated across the two wave-halves.
pub const RT_16X16_W32_IN: RTBaseShape =
    RTBaseShape { base: BaseShape { rows: 16, cols: 16, ept: 16 }, map: LaneMap::Strided { stride: 0 } };
/// wave32 WMMA f32 accumulator, **transposed** for an N-major memory store
/// ([`LaneMap::InterleavedT`]). Used for the FA output
/// tile (`o_reg_t`, `O[q,d]`) — the transpose of the `[d,q]` PV accumulator
/// ([`RT_16X16_W32_ACC`]). gfx942 reaches the same transposed store through the
/// plain stride map, so this is RDNA-only.
pub const RT_16X16_W32_ACC_T: RTBaseShape =
    RTBaseShape { base: BaseShape { rows: 16, cols: 16, ept: 8 }, map: LaneMap::InterleavedT };

// ── CUDA sm_80+ (warp32, `mma.sync.m16n8k16`) base shapes ─────────────────────
//
// A 16×16 register tile is two m16n8 halves along the register axis
// ([`LaneMap::MmaSync`], ThunderKittens `rt_base`): 8 elements/lane for f16/bf16
// inputs AND the f32 accumulator, so every fragment role shares one shape and an
// accumulator is directly reusable as an A operand (as on CDNA). The LDS strip
// fills 256/32 = 8 elements/lane and is XOR-swizzled for the quad-strided gather.

/// LDS strip fragment for the warp32 `mma.sync` kernels (`ept = 256/32 = 8`),
/// swizzled conflict-free for the m16n8k16 gather ([`Swizzle::Sw16x16Mma`]).
pub const ST_16X16_MMA: STBaseShape =
    STBaseShape { base: BaseShape { rows: 16, cols: 16, ept: 8 }, swizzle: Swizzle::Sw16x16Mma };
/// warp32 `mma.sync` fragment: A operand, B operand (read transposed) and f32
/// accumulator alike.
pub const RT_16X16_MMA: RTBaseShape =
    RTBaseShape { base: BaseShape { rows: 16, cols: 16, ept: 8 }, map: LaneMap::MmaSync };
