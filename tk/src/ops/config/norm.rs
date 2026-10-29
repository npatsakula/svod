//! Row kernel configs: the rows a block of one-warp rows spans, for the
//! norms and the heads merge.

use crate::kernels::rows::NormCfg;

/// One warp per head row, the head split in halves a warp spreads over.
pub(crate) fn heads_candidates(d: usize) -> Vec<NormCfg> {
    if !(d.is_power_of_two() && (16..=256).contains(&d)) {
        return vec![];
    }
    ROW_WARPS.map(|br| NormCfg { br }).to_vec()
}

/// One warp per row, a multiple of 256 wide: a wave32 warp's 16-byte loads span
/// 256 columns, so on every target one warp holds a row in whole vectors, and
/// 2048 is 64 elements a lane.
pub fn norm_candidates(d: usize) -> Vec<NormCfg> {
    if !(d.is_multiple_of(256) && (256..=2048).contains(&d)) {
        return vec![];
    }
    ROW_WARPS.map(|br| NormCfg { br }).to_vec()
}

/// Search bound: the rows (one warp each) of a row kernel's block.
const ROW_WARPS: [usize; 3] = [4, 8, 16];
