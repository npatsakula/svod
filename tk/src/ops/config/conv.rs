//! Convolution configs over the GEMM lattice: which tiles a convolution
//! body takes, the split counts a starved grid's reduction breaks into, and
//! the convolution's own claim — the GEMM's with its fill's working set the
//! image's, not the expanded matrix's. The norms and the heads merge
//! (`norm_candidates`, `heads_candidates`) rank nothing: their kernels are
//! one warp a row and the tuner times the candidate list as given.

use std::sync::Arc;

use crate::atoms::{Fill, Target};
use crate::kernels::conv::{ConvCfg, ConvGeom};
use crate::kernels::gemm::GemmCfg;

use super::cost::ELEM;
use super::cost::{Claim, Family, Planner, cached};
use super::gemm::{GemmProblem, gemm_lattice};
use super::search::{Rejected, sorted, spread};

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

/// A convolution as the GEMM body runs it: `batches` grid batches of `m`
/// output pixels of `g` (`n = cout`, `k = kh·kw·cin`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ConvProblem {
    pub gemm: GemmProblem,
    pub geom: ConvGeom,
}

impl ConvProblem {
    /// The GEMM body's claim with the convolution's own fill: each input
    /// pixel is read by every tap of every output pixel within a stride of
    /// it, so the stream's distinct bytes are the image's and the weight's —
    /// the working set the cache ladder prices the gathers' re-reads at —
    /// while a trip still fills the expanded tile's.
    pub(crate) fn claim(&self, planner: &Planner, cfg: &GemmCfg) -> Result<Claim, Rejected> {
        let mut claim = self.gemm.claim(planner, cfg)?;
        let ConvGeom { h, w, cin, cout, .. } = self.geom;
        claim.fill.unique = (self.gemm.batches * (h * w * cin + cout * self.geom.k()) * ELEM) as f64;
        Ok(claim)
    }

    /// The gemm lattice's configs that run this convolution on the planner's
    /// target, cheapest first through the convolution's own claim: a step
    /// that stays within one tap (`bk` divides `cin`) and tiles that pad
    /// `m × n` by at most 1/8. On the cp.async path a step is at least two
    /// matrix-core depths, else the conv takes the graph.
    fn ranked(&self, planner: &Planner) -> Vec<ConvCfg> {
        let depth = match (planner.device.target.fill, planner.device.target.geometry()) {
            (Fill::CpAsync, Some(atom)) => 2 * atom.k as usize,
            _ => 0,
        };
        let eligible = |c: &GemmCfg| {
            self.geom.cin.is_multiple_of(c.tile[2]) && c.tile[2] >= depth && self.gemm.padding(c) <= CONV_PADDING
        };
        let lattice = gemm_lattice(&planner.device.target);
        let count = lattice.len();
        let admitted: Vec<(f64, u32, GemmCfg)> = lattice
            .into_iter()
            .filter(|c| eligible(c))
            .filter_map(|c| self.claim(planner, &c).ok().map(|claim| (planner.price(&claim), claim.regs, c)))
            .collect();
        spread(sorted(self, admitted, count).into_iter().map(|(cost, _, c)| (cost, c)).collect(), |c| {
            [c.tile[0], c.tile[1]]
        })
        .into_iter()
        .map(|gemm| ConvCfg { gemm, split: 1 })
        .collect()
    }
}

impl Family for ConvProblem {
    type Cfg = GemmCfg;

    /// The gemm lattice's configs a convolution body can run: a step that
    /// stays within one tap and tiles that pad `m × n` by at most an
    /// eighth; on the cp.async path a step at least two matrix-core depths,
    /// else the conv takes the graph.
    fn lattice(&self, target: &Target) -> Vec<GemmCfg> {
        let depth = match (target.fill, target.geometry()) {
            (Fill::CpAsync, Some(atom)) => 2 * atom.k as usize,
            _ => 0,
        };
        gemm_lattice(target)
            .into_iter()
            .filter(|c| {
                self.geom.cin.is_multiple_of(c.tile[2]) && c.tile[2] >= depth && self.gemm.padding(c) <= CONV_PADDING
            })
            .collect()
    }

    fn claim(&self, planner: &Planner, cfg: &GemmCfg) -> Result<Claim, Rejected> {
        ConvProblem::claim(self, planner, cfg)
    }
}

impl Planner {
    /// Unsplit convolution configs for `batches` grid batches of `m` output
    /// pixels of `g`, [`ConvProblem::ranked`] cached per problem.
    pub(crate) fn conv_ranked(&self, batches: usize, m: usize, g: &ConvGeom) -> Arc<[ConvCfg]> {
        if batches * m * g.cout * g.k() == 0 {
            return Arc::new([]);
        }
        let p = ConvProblem { gemm: GemmProblem { batches, m, n: g.cout, k: g.k(), halves: 1, rope: None }, geom: *g };
        cached(&self.conv, p, || p.ranked(self))
    }

    /// The convolution candidates of `admitted` (the shortlist of
    /// [`Self::conv_ranked`]): each unsplit and, on a starved grid (fewer
    /// than twice [`CONV_BLOCKS_PER_SM`] blocks per SM, a single static
    /// batch), the lead and the admitted configs with the most blocks at
    /// 1/16 and at 1/8 padding also split over the reduction, in counts
    /// dividing the steps that keep at most 16 blocks per SM.
    pub(crate) fn conv_splits(&self, batches: usize, m: usize, g: &ConvGeom, admitted: &[ConvCfg]) -> Vec<ConvCfg> {
        let Some(&ConvCfg { gemm: base, .. }) = admitted.first() else { return vec![] };
        let (p, k) = (GemmProblem { batches, m, n: g.cout, k: g.k(), halves: 1, rope: None }, g.k());
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
