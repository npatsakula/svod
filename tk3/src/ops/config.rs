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

/// Implicit-GEMM convolution tiles, largest first: the GEMM families, then
/// output widths for 96- and 48-channel convolutions (a power-of-two tile
/// would pad them by a third or re-read A per narrow block). A 48-wide B
/// tile splits into whole chunks per thread only at `bk = 64` on four warps
/// or `bk = 32` on two.
const CONV_FAMILIES: [GemmCfg; 9] = [
    gemm_cfg([128, 128, 32], 3, [2, 4], false),
    gemm_cfg([128, 96, 32], 2, [2, 2], true),
    gemm_cfg([128, 64, 32], 2, [2, 2], true),
    gemm_cfg([64, 128, 32], 2, [2, 2], false),
    gemm_cfg([64, 96, 32], 3, [2, 2], true),
    gemm_cfg([128, 48, 64], 2, [4, 1], true),
    gemm_cfg([64, 64, 32], 3, [2, 2], true),
    gemm_cfg([32, 96, 32], 3, [2, 2], true),
    gemm_cfg([64, 48, 32], 3, [2, 1], true),
];

/// Blocks per SM a convolution's lead tile must reach: its grids are a few
/// hundred blocks, where the GEMM's eight would leave only the smallest tile.
const CONV_BLOCKS_PER_SM: usize = 2;

/// Whether `cfg` lowers: the warp grid tiles the output with 16×8 atoms and
/// both operand tiles split into whole 16-byte chunks per thread.
pub fn conv_cfg_fits(cfg: &GemmCfg) -> bool {
    let ([bm, bn, bk], [wr, wc]) = (cfg.tile, cfg.warps.map(|w| w as usize));
    let threads = 32 * wr * wc;
    bm.is_multiple_of(16 * wr)
        && bn.is_multiple_of(8 * wc)
        && bk.is_multiple_of(16)
        && (bm * bk / 8).is_multiple_of(threads)
        && (bn * bk / 8).is_multiple_of(threads)
}

/// Convolution configs for `batches` grid batches of `m` output pixels,
/// `n = cout`, `k = kh·kw·cin`. A family is eligible when `bk` (halved down
/// to 16 until it divides `cin`, so a step stays in one tap) lowers and its
/// tiles pad `m × n` by at most 1/16. The lead is the largest eligible
/// family whose grid gives every SM [`CONV_BLOCKS_PER_SM`] blocks, else the
/// eligible one with the most blocks; its pipeline variants follow, then
/// every other eligible family.
pub fn conv_candidates(target: &Target, batches: usize, m: usize, n: usize, k: usize, cin: usize) -> Vec<GemmCfg> {
    if m == 0 || n == 0 || k == 0 {
        return vec![];
    }
    let sms = target.sms.unwrap_or(DEFAULT_SMS) as usize;
    let fit = |mut c: GemmCfg| {
        while !cin.is_multiple_of(c.tile[2]) && c.tile[2] > 16 {
            c.tile[2] /= 2;
        }
        let [bm, bn, bk] = c.tile;
        let (gm, gn) = (m.div_ceil(bm), n.div_ceil(bn));
        let padded = 16 * (gm * bm * gn * bn - m * n) <= gm * bm * gn * bn;
        (padded && cin.is_multiple_of(bk) && conv_cfg_fits(&c) && c.smem_bytes(false) <= target.smem_bytes).then_some(c)
    };
    let eligible: Vec<GemmCfg> = CONV_FAMILIES.into_iter().filter_map(fit).collect();
    let blocks = |c: &GemmCfg| batches * m.div_ceil(c.tile[0]) * n.div_ceil(c.tile[1]);
    let Some(&base) = eligible
        .iter()
        .find(|c| blocks(c) >= CONV_BLOCKS_PER_SM * sms)
        .or_else(|| eligible.iter().max_by_key(|c| blocks(c)))
    else {
        return vec![];
    };
    let variants = [
        GemmCfg { stages: if base.stages == 2 { 3 } else { 2 }, ..base },
        GemmCfg { tile: [base.tile[0], base.tile[1], 64], stages: 2, ..base },
        GemmCfg { unroll: !base.unroll, ..base },
    ];
    let mut out = vec![base];
    for c in variants.into_iter().filter_map(fit).chain(eligible) {
        if !out.contains(&c) {
            out.push(c);
        }
    }
    out
}

/// One warp per 16 query rows. `d = 128` leads with half-width key blocks:
/// 64-wide ones measured 18.2 TFLOP/s against 22.2 on sm_86, since 64 KB
/// per block leaves one block per SM. `d = 48` keeps the shapes whose K/V
/// fills divide among the block's threads (96-byte rows, 16-byte chunks).
/// A decoder step (`t ≤ 16`) is bandwidth-bound: one-warp blocks, which
/// leave room for several per SM.
pub fn attention_candidates(target: &Target, d: usize, t: usize) -> Vec<FaCfg> {
    let fa = FaCfg::new;
    let list = match (d, t <= 16) {
        (48 | 64 | 128, true) => vec![fa(16, 64, 2), fa(16, 64, 3), fa(16, 32, 2)],
        (48, false) => vec![fa(64, 64, 2), fa(64, 64, 3)],
        (64, false) => vec![fa(64, 64, 2), fa(64, 64, 3), fa(128, 64, 2), fa(64, 32, 2), fa(128, 32, 2), fa(64, 32, 3)],
        (128, false) => {
            vec![fa(64, 32, 2), fa(64, 32, 3), fa(128, 32, 2), fa(64, 64, 2), fa(128, 64, 2), fa(128, 32, 3)]
        }
        _ => return vec![],
    };
    list.into_iter().filter(|c| c.smem_bytes(d) <= target.smem_bytes).collect()
}

/// Key splits worth measuring for `tiles` independent query tiles over
/// `blocks` key blocks: one, two, the counts around what keeps two blocks
/// per SM busy, and every block its own split when that is within reach
/// (the merge costs a launch, so never past the blocks). A target without
/// an SM count keeps one.
pub fn split_candidates(target: &Target, tiles: usize, blocks: usize) -> Vec<usize> {
    let Some(sms) = target.sms else { return vec![1] };
    let want = (2 * sms as usize).div_ceil(tiles.max(1));
    let mut out: Vec<usize> = [1, 2, want / 2, want, 2 * want, blocks]
        .into_iter()
        .filter(|&s| (1..=blocks.min(2 * want)).contains(&s))
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// One warp per head row, the head split in halves a warp spreads over.
pub fn heads_candidates(_target: &Target, d: usize) -> Vec<NormCfg> {
    if !(d.is_power_of_two() && (16..=256).contains(&d)) {
        return vec![];
    }
    [4, 8, 16].map(|br| NormCfg { br }).to_vec()
}

/// One warp per row of a power-of-two width a warp's vector loads span.
pub fn norm_candidates(_target: &Target, d: usize) -> Vec<NormCfg> {
    if !(d.is_power_of_two() && (256..=2048).contains(&d)) {
        return vec![];
    }
    [4, 8, 16].map(|br| NormCfg { br }).to_vec()
}
