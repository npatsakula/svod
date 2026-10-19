//! Tile configs per target and size class, from the measured picks on sm_86.
//! Each op has one `choose_*` seam, where a tune store will plug in.

use svod_dtype::GpuArch;

use crate::atoms::Target;
use crate::kernels::attention::FaCfg;
use crate::kernels::gemm::GemmCfg;
use crate::kernels::rows::NormCfg;

/// Targets with measured tables: `mma.sync` + `cp.async` + `ldmatrix` (sm_80+).
pub fn has_tables(target: &Target) -> bool {
    matches!(target.arch, GpuArch::Cuda(c) if c.major >= 8)
}

const fn gemm_cfg(tile: [usize; 3], stages: usize, warps: [u32; 2]) -> GemmCfg {
    GemmCfg { tile, stages, warps, group_m: 8, unroll: false }
}

/// Largest first: 4096³ peaks at the first, the narrower tiles keep small
/// grids busy, and `bk = 16` serves a reduction dim that is no multiple of 32.
const GEMM_LADDER: [GemmCfg; 5] = [
    gemm_cfg([128, 128, 32], 3, [2, 4]),
    gemm_cfg([128, 128, 32], 2, [2, 4]),
    GemmCfg { unroll: true, ..gemm_cfg([128, 64, 32], 2, [2, 2]) },
    gemm_cfg([64, 64, 32], 2, [2, 2]),
    gemm_cfg([64, 64, 16], 2, [2, 2]),
];

/// `m` rows in all (every batch), `n` output columns, reduction dim `k`.
pub fn choose_gemm(target: &Target, m: usize, n: usize, k: usize, gated: bool) -> Option<GemmCfg> {
    let start = if m <= 64 {
        3
    } else if m.div_ceil(128) * n.div_ceil(128) >= 64 {
        0
    } else {
        2
    };
    GEMM_LADDER[start..]
        .iter()
        .find(|c| k > 0 && k.is_multiple_of(c.tile[2]) && c.smem_bytes(gated) <= target.smem_bytes)
        .copied()
}

/// One warp per 16 query rows; `d = 128` takes half-width key blocks to stay
/// within static shared memory.
pub fn choose_attention(target: &Target, d: usize) -> Option<FaCfg> {
    let cfg = match d {
        64 => FaCfg { bq: 64, bkv: 64, stages: 2 },
        128 => FaCfg { bq: 64, bkv: 32, stages: 2 },
        _ => return None,
    };
    (cfg.smem_bytes(d) <= target.smem_bytes).then_some(cfg)
}

/// One warp per row of a power-of-two width a warp's vector loads span.
pub fn choose_norm(_target: &Target, d: usize) -> Option<NormCfg> {
    (d.is_power_of_two() && (256..=2048).contains(&d)).then_some(NormCfg { br: 4 })
}
