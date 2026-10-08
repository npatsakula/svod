//! Tile config candidates per target and shape, most promising first: the
//! first is what runs untuned, the tune store measures the rest.

use svod_dtype::GpuArch;

use crate::atoms::Target;
use crate::kernels::attention::FaCfg;
use crate::kernels::gemm::GemmCfg;
use crate::kernels::rows::NormCfg;

/// Targets with measured tables: `mma.sync` + `cp.async` + `ldmatrix` (sm_80+).
pub fn has_tables(target: &Target) -> bool {
    matches!(target.arch, GpuArch::Cuda(c) if c.major >= 8)
}

const fn gemm_cfg(tile: [usize; 3], stages: usize, warps: [u32; 2], unroll: bool) -> GemmCfg {
    GemmCfg { tile, stages, warps, group_m: 8, unroll }
}

/// Output tile families, largest first, each with its measured best
/// pipeline on sm_86 (4096³ peaks at the first; 64×64 wins M = 704).
const GEMM_FAMILIES: [GemmCfg; 4] = [
    gemm_cfg([128, 128, 32], 3, [2, 4], false),
    gemm_cfg([128, 64, 32], 2, [2, 2], true),
    gemm_cfg([64, 128, 32], 2, [2, 2], false),
    gemm_cfg([64, 64, 32], 3, [2, 2], true),
];

/// Blocks per SM a family's grid needs before its larger tile pays off:
/// at M = 704 the 128-row tiles lose to 64×64 with 7 blocks per SM.
const BLOCKS_PER_SM: usize = 8;
/// The SM count assumed when the device reports none.
const DEFAULT_SMS: u32 = 32;

/// At most eight GEMM configs for `batches` GEMMs of `m` rows, `n` columns
/// and reduction dim `k`. The lead family is the largest whose grid fills
/// every SM `BLOCKS_PER_SM` times with at most 1/16 of its output tiles'
/// area past `m × n` (else 64×64); it comes with its pipeline variants, the
/// other families follow with their own pipelines.
pub fn gemm_candidates(target: &Target, batches: usize, m: usize, n: usize, k: usize, gated: bool) -> Vec<GemmCfg> {
    let sms = target.sms.unwrap_or(DEFAULT_SMS) as usize;
    let fills = |c: &GemmCfg| {
        let [bm, bn, _] = c.tile;
        let (gm, gn) = (m.div_ceil(bm), n.div_ceil(bn));
        let padded = 16 * (gm * bm * gn * bn - m * n) <= gm * bm * gn * bn;
        padded && batches * gm * gn >= BLOCKS_PER_SM * sms
    };
    let lead = GEMM_FAMILIES.iter().position(fills).unwrap_or(GEMM_FAMILIES.len() - 1);
    let base = GEMM_FAMILIES[lead];
    let [bm, bn, _] = base.tile;
    let warps = match base.warps {
        [2, 4] => [4, 2],
        [2, 2] if bm > bn => [4, 2],
        [2, 2] if bm < bn => [2, 4],
        w => w,
    };
    let variants = [
        base,
        GemmCfg { stages: if base.stages == 2 { 3 } else { 2 }, ..base },
        GemmCfg { tile: [bm, bn, 64], stages: 2, ..base },
        GemmCfg { unroll: !base.unroll, ..base },
        GemmCfg { warps, ..base },
    ];
    let others = GEMM_FAMILIES.iter().enumerate().filter(|(i, _)| *i != lead).map(|(_, c)| *c);
    let mut out: Vec<GemmCfg> = vec![];
    for mut c in variants.into_iter().chain(others) {
        // A reduction dim off `bk` halves it, down to one 16-deep mma step.
        while !k.is_multiple_of(c.tile[2]) && c.tile[2] > 16 {
            c.tile[2] /= 2;
        }
        if k > 0 && k.is_multiple_of(c.tile[2]) && c.smem_bytes(gated) <= target.smem_bytes && !out.contains(&c) {
            out.push(c);
        }
    }
    out
}

/// One warp per 16 query rows. `d = 128` leads with half-width key blocks:
/// 64-wide ones measured 18.2 TFLOP/s against 22.2 on sm_86, since 64 KB
/// per block leaves one block per SM. `d = 48` keeps the shapes whose K/V
/// fills divide among the block's threads (96-byte rows, 16-byte chunks).
pub fn attention_candidates(target: &Target, d: usize) -> Vec<FaCfg> {
    let fa = |bq, bkv, stages| FaCfg { bq, bkv, stages };
    let list = match d {
        48 => vec![fa(64, 64, 2), fa(64, 64, 3)],
        64 => vec![fa(64, 64, 2), fa(64, 64, 3), fa(128, 64, 2), fa(64, 32, 2), fa(128, 32, 2), fa(64, 32, 3)],
        128 => vec![fa(64, 32, 2), fa(64, 32, 3), fa(128, 32, 2), fa(64, 64, 2), fa(128, 64, 2), fa(128, 32, 3)],
        _ => return vec![],
    };
    list.into_iter().filter(|c| c.smem_bytes(d) <= target.smem_bytes).collect()
}

/// One warp per row of a power-of-two width a warp's vector loads span.
pub fn norm_candidates(_target: &Target, d: usize) -> Vec<NormCfg> {
    if !(d.is_power_of_two() && (256..=2048).contains(&d)) {
        return vec![];
    }
    [4, 8, 16].map(|br| NormCfg { br }).to_vec()
}
