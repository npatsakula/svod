//! Convolution configs over the GEMM lattice: which tiles a convolution
//! body takes, and the split counts a starved grid's reduction breaks into.

use std::sync::Arc;

use crate::atoms::Fill;
use crate::kernels::conv::{ConvCfg, ConvGeom};
use crate::kernels::gemm::GemmCfg;

use super::cost::Planner;
use super::gemm::GemmProblem;

/// Search bound: half the blocks per SM a single-batch convolution's grid
/// must reach before its reduction is left whole ([`Planner::conv_splits`]).
const CONV_BLOCKS_PER_SM: usize = 2;
/// Model: the share of a convolution tile's area that may pad past `m × n`;
/// a split is sought among tiles within half of it as well.
const CONV_PADDING: f64 = 1.0 / 8.0;
/// Search bound: the counts a starved convolution's reduction splits into.
const CONV_SPLITS: [usize; 5] = [2, 3, 4, 6, 9];
/// Search bound: the most blocks per SM a split convolution grid reaches.
const SPLIT_BLOCKS_PER_SM: usize = 16;

impl Planner {
    /// Unsplit convolution configs for `batches` grid batches of `m` output
    /// pixels of `g` (`n = cout`, `k = kh·kw·cin`), the cheapest lattice
    /// configs whose step stays within one tap (`bk` divides `cin`) and
    /// whose tiles pad `m × n` by at most 1/8. On the cp.async path a step
    /// is at least two matrix-core depths, else the conv takes the graph.
    pub(crate) fn conv_ranked(&self, batches: usize, m: usize, g: &ConvGeom) -> Arc<[ConvCfg]> {
        let (n, k, cin) = (g.cout, g.k(), g.cin);
        if batches * m * n * k == 0 {
            return Arc::new([]);
        }
        let p = GemmProblem { batches, m, n, k, halves: 1 };
        let target = &self.device.target;
        let depth = match (target.fill, target.geometry()) {
            (Fill::CpAsync, Some(atom)) => 2 * atom.k as usize,
            _ => 0,
        };
        let eligible =
            |c: &&GemmCfg| cin.is_multiple_of(c.tile[2]) && c.tile[2] >= depth && p.padding(c) <= CONV_PADDING;
        self.ranked(p).iter().filter(eligible).map(|&gemm| ConvCfg { gemm, split: 1 }).collect()
    }

    /// The convolution candidates of `admitted` (the shortlist of
    /// [`Self::conv_ranked`]): each unsplit and, on a starved grid (fewer
    /// than twice [`CONV_BLOCKS_PER_SM`] blocks per SM, a single static
    /// batch), the lead and the admitted configs with the most blocks at
    /// 1/16 and at 1/8 padding also split over the reduction, in counts
    /// dividing the steps that keep at most 16 blocks per SM.
    pub(crate) fn conv_splits(&self, batches: usize, m: usize, g: &ConvGeom, admitted: &[ConvCfg]) -> Vec<ConvCfg> {
        let Some(&ConvCfg { gemm: base, .. }) = admitted.first() else { return vec![] };
        let (p, k) = (GemmProblem { batches, m, n: g.cout, k: g.k(), halves: 1 }, g.k());
        let mut out = admitted.to_vec();
        let sms = self.sms();
        let blocks = |c: &GemmCfg| p.blocks(c);
        let widest = |tight: bool| {
            let fits = |c: &&GemmCfg| !tight || p.padding(c) <= CONV_PADDING / 2.0;
            admitted.iter().map(|c| &c.gemm).filter(fits).max_by_key(|c| blocks(c)).copied().unwrap_or(base)
        };
        if batches == 1 && blocks(&base) < 2 * CONV_BLOCKS_PER_SM * sms {
            for gemm in [base, widest(true), widest(false)] {
                let trips = k / gemm.tile[2];
                for split in CONV_SPLITS {
                    let c = ConvCfg { gemm, split };
                    let within = blocks(&gemm) * split <= SPLIT_BLOCKS_PER_SM * sms;
                    if trips.is_multiple_of(split) && within && !out.contains(&c) {
                        out.push(c);
                    }
                }
            }
        }
        out
    }
}
