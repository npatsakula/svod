//! Vendor fragment layouts, each equal to the tk1 `LaneMap::rc` closed form it replaces
//! (proved exhaustively in `test/unit/layout.rs`).

use super::Dim::{Col, Lane, Reg, Row};
use super::Layout;

const LANE_ROWS16: [[u32; 2]; 4] = [[1, 0], [2, 0], [4, 0], [8, 0]];

/// NVIDIA `mma.sync.m16n8k16` 16×16 register tile (tk `rt_base`, `LaneMap::MmaSync`),
/// `g = L>>2, t = L&3`: `row = g + 8·((j/2)%2), col = 2t + j%2 + 8·(j/4)`. Registers
/// `0..4` / `4..8` are the two [`mma_sync_c`] n-halves; the whole tile is the PTX A
/// fragment `a0..a7`; read transposed it holds the two [`mma_sync_b`] fragments
/// `{0,1,4,5}` / `{2,3,6,7}`.
pub fn mma_sync_16x16() -> Layout {
    mma_sync_c().product(&Layout::identity(Reg, 2, Col))
}

/// PTX `mma.m16n8k16` A fragment (16×16, M×K): the [`mma_sync_16x16`] tile itself.
pub fn mma_sync_a() -> Layout {
    mma_sync_16x16()
}

/// PTX `mma.m16n8k16` B fragment (16×8, K×N): `k = 2t + b%2 + 8·(b/2), n = g`.
pub fn mma_sync_b() -> Layout {
    Layout::from_bases(
        [(Row, 16), (Col, 8)],
        &[(Reg, &[[1, 0], [8, 0]]), (Lane, &[[2, 0], [4, 0], [0, 1], [0, 2], [0, 4]])],
    )
}

/// PTX `mma.m16n8k16` C/D fragment (16×8, M×N): `row = g + 8·(c/2), col = 2t + c%2`.
pub fn mma_sync_c() -> Layout {
    Layout::from_bases(
        [(Row, 16), (Col, 8)],
        &[(Reg, &[[0, 1], [8, 0]]), (Lane, &[[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]])],
    )
}

/// The registers one `ldmatrix.sync.aligned.m8n8.x4[.trans].b16` fills: each lane gets
/// a pair `(L/4, 2(L%4) + e)` of every 8×8 matrix (transposed under `.trans`), and
/// register pair `p` is matrix `p` of the [`ldmatrix_x4_rows`] order (TL, BL, TR, BR)
/// — plain, or with the middle two swapped under `.trans` (ThunderKittens `ldsm4t`).
/// Plain equals [`mma_sync_16x16`], `.trans` its transpose.
pub fn ldmatrix_x4(trans: bool) -> Layout {
    let m8 = Layout::from_bases(
        [(Row, 8), (Col, 8)],
        &[(Reg, &[[0, 1]]), (Lane, &[[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]])],
    );
    let words: &[[u32; 2]] = if trans { &[[0, 1], [1, 0]] } else { &[[1, 0], [0, 1]] };
    let m8 = if trans { m8.transpose() } else { m8 };
    m8.product(&Layout::from_bases([(Row, 2), (Col, 2)], &[(Reg, words)]))
}

/// The `ldmatrix.x4` addressing: lane `L` supplies the 16-byte row `L % 16` at column
/// `8·(L/16)`, so lanes `8m..8m+8` address matrix `m` = TL, BL, TR, BR.
pub fn ldmatrix_x4_rows() -> Layout {
    Layout::from_bases([(Row, 16), (Col, 16)], &[(Lane, &[[1, 0], [2, 0], [4, 0], [8, 0], [0, 8]])])
}

/// The registers one wave32 `global_load_tr_b128` fills: each group of eight
/// lanes reads eight 16-byte rows and receives them transposed, lane `k` of the
/// group taking element `k` of every row (`ldmatrix.trans` per group). With the
/// four 8×8 blocks of a 16×16 tile addressed in [`global_tr_rows`] order, lane
/// `L` holds column `L % 16` at rows `8·(L/16) + j`: the RDNA4 WMMA B fragment,
/// [`strided`]`(8, 32)` read transposed.
pub fn global_tr_b128() -> Layout {
    strided(8, 32).transpose()
}

/// Its addressing: lane `L` supplies the 16-byte row `L % 8 + 8·(L/16)` at
/// column `8·((L/8) % 2)`, so lane groups `0..8`, `8..16`, `16..24`, `24..32`
/// read the TL, TR, BL, BR blocks.
pub fn global_tr_rows() -> Layout {
    Layout::from_bases([(Row, 16), (Col, 16)], &[(Lane, &[[1, 0], [2, 0], [4, 0], [0, 8], [8, 0]])])
}

/// tk `LaneMap::Strided { stride }` on a 16×16 tile over `lanes` lanes: `row = L % 16,
/// col = (L / 16)·stride + j`, `j < stride` (`j < 16` at `stride = 0`, where every
/// lane-group holds the same K run and the high lane bits are free).
pub fn strided(stride: u32, lanes: u32) -> Layout {
    let regs = if stride == 0 { 16 } else { stride };
    assert!(stride == 0 || lanes / 16 * stride == 16, "strided({stride}, {lanes}) does not cover 16x16");
    let reg: Vec<[u32; 2]> = (0..regs.trailing_zeros()).map(|i| [0, 1 << i]).collect();
    let lane: Vec<[u32; 2]> =
        LANE_ROWS16.into_iter().chain((0..(lanes / 16).trailing_zeros()).map(|i| [0, stride << i])).collect();
    Layout::from_bases([(Row, 16), (Col, 16)], &[(Reg, &reg), (Lane, &lane)])
}

/// CDNA `v_mfma_f32_16x16x16_bf16` A operand (M×K): `row = L % 16, k = 4·(L/16) + j`.
/// B (K×N) and the C/D accumulator (M×N: `row = 4·(L/16) + j, col = L % 16`) are its
/// [`Layout::transpose`].
pub fn mfma_16x16x16() -> Layout {
    strided(4, 64)
}

/// CDNA `v_mfma_f32_32x32x8_bf16` A operand (32×8, M×K): `row = L % 32, k = 4·(L/32) + j`;
/// B (K×N) is its transpose.
pub fn mfma_32x32x8_a() -> Layout {
    Layout::from_bases(
        [(Row, 32), (Col, 8)],
        &[(Reg, &[[0, 1], [0, 2]]), (Lane, &[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]])],
    )
}

/// CDNA `v_mfma_f32_32x32x8_bf16` C/D accumulator (32×32): 16 f32 per lane in four
/// groups of four rows, `row = 8·(j/4) + 4·(L/32) + j%4, col = L % 32` (CK
/// `xdlops_gemm.hpp`: rows unmerge as `groups × input blocks × group size`).
pub fn mfma_32x32x8_c() -> Layout {
    Layout::from_bases(
        [(Row, 32), (Col, 32)],
        &[(Reg, &[[1, 0], [2, 0], [8, 0], [16, 0]]), (Lane, &[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [4, 0]])],
    )
}

/// RDNA3 WMMA f32 accumulator (`LaneMap::Interleaved`): `row = 2j + L/16, col = L % 16`.
pub fn wmma_gfx11_acc() -> Layout {
    Layout::from_bases(
        [(Row, 16), (Col, 16)],
        &[(Reg, &[[2, 0], [4, 0], [8, 0]]), (Lane, &[[0, 1], [0, 2], [0, 4], [0, 8], [1, 0]])],
    )
}

/// RDNA3 WMMA input: the replicated `strided(0, 32)`, lanes `L` and `L + 16` identical.
pub fn wmma_gfx11_input() -> Layout {
    strided(0, 32)
}

/// RDNA4 WMMA operand: `strided(8, 32)`; the accumulator is its transpose.
pub fn wmma_gfx12() -> Layout {
    strided(8, 32)
}

/// Apple `simdgroup_matrix<T, 8, 8>` (`LaneMap::SimdgroupMatrix`, measured on Apple9):
/// `row = 4·((L/16)%2) + (L/2)%4, col = 4·((L/8)%2) + 2·(L%2) + j`. B operand and
/// accumulator; the A operand is its transpose.
pub fn simdgroup_8x8() -> Layout {
    Layout::from_bases([(Row, 8), (Col, 8)], &[(Reg, &[[0, 1]]), (Lane, &[[0, 2], [1, 0], [2, 0], [0, 4], [4, 0]])])
}
