//! Tile shape descriptors and layouts.
//!
//! These are the pure, data-only building blocks shared by every tile kind. A
//! [`BaseShape`] is one WMMA-sized fragment (e.g. 16×16); a full tile is a grid
//! of base shapes. The concrete tile wrappers (GL/ST/RT/RV) that bind a buffer
//! and a [`crate::Kernel`] live alongside the builder.
//!
//! `elements_per_thread` is carried **explicitly** per shape rather than derived
//! `num_elements / WARP_THREADS`, because it is a function of the matrix-core
//! fragment layout, which differs by arch: CDNA wave64 16×16 = 4/lane; gfx11
//! wave32 = 8/lane for the accumulator and **16/lane for the (replicated) WMMA
//! inputs** (256/32 × the 0-15≡16-31 wave-half replication); gfx12 wave32 drops
//! that replication — 8/lane for every role. The `_W32_*` constants below are the
//! gfx11 shapes, [`RT_16X16_GFX12`] the gfx12 one; the unsuffixed ones are gfx942.

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
    /// the module docs), NOT always `num_elements / wave_size` (gfx11 inputs are
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

/// Register-tile base fragment: a [`BaseShape`], its per-lane [`LaneMap`]
/// (which element of the fragment each lane's register `j` holds), and the order
/// the matrix core reads the lane's 32-bit register pairs in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RTBaseShape {
    pub base: BaseShape,
    pub map: LaneMap,
    /// The fragment's register pairs (elements `2p, 2p+1`) in the order the
    /// matrix core reads them. A gather that fetches four pairs in one
    /// instruction (`ldmatrix.x4`) returns them into four consecutive registers
    /// in this order, so every operand the core takes is already an aligned run
    /// of adjacent registers — `ptxas` otherwise moves each run into place.
    pub feed: [usize; 4],
}

/// [`RTBaseShape::feed`] for a fragment the core reads whole, pair by pair.
pub const PAIRS_IN_ORDER: [usize; 4] = [0, 1, 2, 3];

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
pub const RT_16X16: RTBaseShape = RTBaseShape {
    base: BaseShape { rows: 16, cols: 16, ept: 4 },
    map: LaneMap::Strided { stride: 4 },
    feed: PAIRS_IN_ORDER,
};
pub const RT_32X32: RTBaseShape = RTBaseShape {
    base: BaseShape { rows: 32, cols: 32, ept: 16 },
    map: LaneMap::Strided { stride: 4 },
    feed: PAIRS_IN_ORDER,
};
pub const RT_16X32: RTBaseShape = RTBaseShape {
    base: BaseShape { rows: 16, cols: 32, ept: 8 },
    map: LaneMap::Strided { stride: 8 },
    feed: PAIRS_IN_ORDER,
};
pub const RT_32X16: RTBaseShape = RTBaseShape {
    base: BaseShape { rows: 32, cols: 16, ept: 8 },
    map: LaneMap::Strided { stride: 8 },
    feed: PAIRS_IN_ORDER,
};

// ── RDNA3 (gfx11, wave32) base shapes — for the gfx1151 WMMA matmul ───────────
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
    RTBaseShape { base: BaseShape { rows: 16, cols: 16, ept: 8 }, map: LaneMap::Interleaved, feed: PAIRS_IN_ORDER };
/// wave32 WMMA input fragment: 16 K/lane, replicated across the two wave-halves.
pub const RT_16X16_W32_IN: RTBaseShape = RTBaseShape {
    base: BaseShape { rows: 16, cols: 16, ept: 16 },
    map: LaneMap::Strided { stride: 0 },
    feed: PAIRS_IN_ORDER,
};
/// wave32 WMMA f32 accumulator, **transposed** for an N-major memory store
/// ([`LaneMap::InterleavedT`]). Used for the FA output
/// tile (`o_reg_t`, `O[q,d]`) — the transpose of the `[d,q]` PV accumulator
/// ([`RT_16X16_W32_ACC`]). gfx942 and gfx12 reach the same transposed store through
/// the plain stride map, so this is gfx11-only.
pub const RT_16X16_W32_ACC_T: RTBaseShape =
    RTBaseShape { base: BaseShape { rows: 16, cols: 16, ept: 8 }, map: LaneMap::InterleavedT, feed: PAIRS_IN_ORDER };

/// gfx12 (RDNA4, wave32) WMMA fragment — every `FragRole` on the arch. gfx12
/// drops RDNA3's wave-half replication: 8 elements/lane for the inputs AND the
/// f32 accumulator, under CDNA's strided map at stride 8. An operand reads
/// `row = L%16, col = 8·(L/16)+j`; a `Col` accumulator and the N-major
/// `AccumulatorT` store read its transpose. Hardware-verified on gfx1201.
pub const RT_16X16_GFX12: RTBaseShape = RTBaseShape {
    base: BaseShape { rows: 16, cols: 16, ept: 8 },
    map: LaneMap::Strided { stride: 8 },
    feed: PAIRS_IN_ORDER,
};

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
/// The warp32 strip whose base tile spans a whole 64-byte row (32 16-bit
/// columns): a row's four 16-byte `cp.async` chunks land side by side instead of
/// in two base tiles 512 bytes apart, and the chunk swizzle keeps the
/// `ldmatrix` phase conflict-free. For a strip read only by whole rows — a
/// [`crate::ST::subtile`] along its rows — see [`crate::ArchCaps::shared_rows`].
pub const ST_16X32_MMA: STBaseShape =
    STBaseShape { base: BaseShape { rows: 16, cols: 32, ept: 8 }, swizzle: Swizzle::Sw16x16Mma };
/// [`ST_16X32_MMA`] for 128-byte rows (64 16-bit columns).
pub const ST_16X64_MMA: STBaseShape =
    STBaseShape { base: BaseShape { rows: 16, cols: 64, ept: 8 }, swizzle: Swizzle::Sw16x16Mma };
/// warp32 `mma.sync` fragment: the B-position operand and the f32 accumulator.
/// Under tk's `Col` accumulator the B-position tile is the core's A operand,
/// read as one run of all four pairs.
pub const RT_16X16_MMA: RTBaseShape =
    RTBaseShape { base: BaseShape { rows: 16, cols: 16, ept: 8 }, map: LaneMap::MmaSync, feed: PAIRS_IN_ORDER };
/// [`RT_16X16_MMA`] in the A position. A `Col` accumulator makes the product
/// `Cᵀ += Bᵀ·Aᵀ` (see `MmaPlan::resolve`), so the A-position tile is the core's
/// **B** operand, read one n-half at a time: pairs `{0, 2}`, then `{1, 3}`.
pub const RT_16X16_MMA_HALVES: RTBaseShape = RTBaseShape { feed: [0, 2, 1, 3], ..RT_16X16_MMA };

// ── Apple (Metal, SIMD-group 32, `simdgroup_matrix<T, 8, 8>`) base shapes ─────
//
// Apple's only matrix-core shape is 8×8×8 over a 32-lane SIMD group, so a lane
// holds 64/32 = 2 elements — a quarter of the 16×16 fragment every other arch
// carries, which simply makes a 16×16 logical tile a 2×2 grid of fragments. The
// map ([`LaneMap::SimdgroupMatrix`]) is shared by the B operand and the f32
// accumulator, so an MMA result feeds straight back as a B input with no LDS
// relayout; only the A operand reads it flipped (see [`crate::arch::FragRole`]).
// The LDS strip is unswizzled: at 8 columns of 16-bit data a row is
// 16 bytes, one bank-conflict-free chunk already, and the 16-byte XOR swizzles
// assume `chunk < cols`.

/// LDS strip fragment for the Metal kernels (`ept = 64/32 = 2`), unswizzled.
pub const ST_8X8: STBaseShape =
    STBaseShape { base: BaseShape { rows: 8, cols: 8, ept: 2 }, swizzle: Swizzle::Identity };
/// The **A operand** fragment on Apple: [`RT_8X8_SIMD`] read transposed, so the
/// `Row`-declared A tile reaches the core already transposed — the half of
/// `Cᵀ = Bᵀ·Aᵀ` the `Col` declaration does not supply on its own.
pub const RT_8X8_SIMD_T: RTBaseShape =
    RTBaseShape { base: BaseShape { rows: 8, cols: 8, ept: 2 }, map: LaneMap::SimdgroupMatrixT, feed: PAIRS_IN_ORDER };
/// Apple `simdgroup_matrix` fragment: `half`/`bfloat` B operand and `float`
/// accumulator alike (both 2 elements per lane under one map).
pub const RT_8X8_SIMD: RTBaseShape =
    RTBaseShape { base: BaseShape { rows: 8, cols: 8, ept: 2 }, map: LaneMap::SimdgroupMatrix, feed: PAIRS_IN_ORDER };
