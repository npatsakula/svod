//! Attention configs: the query and key block lattice with its key warps
//! and softmax forms, what the lowering lays them into, and what a
//! problem's blocks cost on a planner.

use std::sync::Arc;

use crate::atoms::{Core, Fill, GroupMmaAtom, Target};
use crate::ir::{Axis, Shape};
use crate::kernels::attention::{FaCfg, WARP_ROWS};
use crate::layouts::{Swizzle, WarpGrid, chunked, fills_in_chunks};

use super::cost::{Claim, ELEM, Kind, Ops, PAIR_OPS, Planner, STEP, Spread, Steps, Stream, cached};
use super::search::{Rejected, sorted};

impl FaCfg {
    /// Whether the config lowers on `target` for head size `d`: the atom's M
    /// is the kernel's [`WARP_ROWS`], a warp owns whole atom rows of queries
    /// and the block whole warps, both products tile the atoms, the key
    /// warps share the block as the kernel shares it
    /// ([`FaCfg::shares_keys`]), and the K and V fills (and the bias
    /// tile's under a bias) go in whole 16-byte chunks per thread
    /// ([`fills_in_chunks`]). On the warpgroup core a group of four row
    /// warps of one atom row each owns 64 query rows, both products' column
    /// blocks tile the instruction, a value row spans whole column panels
    /// (one swizzle span, or 128-byte panels) and no key warps split the
    /// block.
    pub fn fits(&self, target: &Target, d: usize, bias: bool) -> bool {
        let Some(atom) = target.geometry() else { return false };
        let (bq, bkv, rows) = (self.bq, self.bkv, self.warp_rows);
        // Key warps fill their own tiles alone.
        let threads = self.row_warps() as usize * target.wave as usize;
        let core = match self.core {
            Core::Warp => true,
            Core::Group => target.group_mma.first().is_some_and(|g| {
                let groups = self.row_warps() / GroupMmaAtom::WARPS;
                self.key_warps == 1
                    && rows == WARP_ROWS
                    && bq == GroupMmaAtom::M as usize * groups as usize
                    && groups.is_power_of_two()
                    && g.n_for(bkv as u32).is_some()
                    && g.n_for(d as u32).is_some()
                    && (Swizzle::spans(d, ELEM) || (d * ELEM).is_multiple_of(Swizzle::B128.span()))
            }),
        };
        core && atom.m as usize == WARP_ROWS
            && rows.is_multiple_of(WARP_ROWS)
            && rows > 0
            && bq.is_multiple_of(rows)
            && bq > 0
            && self.shares_keys().is_ok()
            && [bkv, d].iter().all(|&x| x.is_multiple_of(atom.n as usize) && x.is_multiple_of(atom.k as usize))
            && fills_in_chunks(Shape::new(bkv, d), ELEM, threads)
            && (!self.bias_ring(bias) || fills_in_chunks(Shape::new(bq, bkv), ELEM, threads))
            && target.fill.runs(self.stages)
    }

    /// The registers a lane holds for head size `d`, from the layouts the
    /// lowering assigns: the query fragments and the f32 output, live across
    /// the key loop; the key fragments, the f32 scores, the probabilities and
    /// the value fragments, all live together while a block's scores are
    /// turned into probabilities and multiplied in; and the staged K and V
    /// chunks where the fill goes through registers (the value tile is
    /// stored column-major where it is staged, so its fragments gather as
    /// 16-byte runs like the keys'), with the bias tile's under a bias.
    /// Addressing and what the compiler keeps beside the tiles are not
    /// counted: the compiled kernel's own count is what the tuner reads.
    /// On the warpgroup core the key and value tiles stay in shared memory
    /// and a chain's descriptors (two registers a `k16` step) stand in for
    /// their fragments.
    pub(crate) fn registers(&self, target: &Target, d: usize, bias: bool) -> Option<u32> {
        let grid = WarpGrid { rows: self.row_warps(), cols: 1 };
        let (qk, pv) = (
            target.issue(self.core, grid, self.bq, self.bkv, d)?,
            target.issue(self.core, grid, self.bq, d, self.bkv)?,
        );
        let half = |regs: u32| regs.div_ceil(2);
        let fill = if target.fill == Fill::CpAsync {
            0
        } else {
            let chunks = |shape| chunked(shape, 2, self.row_warps(), target.wave, Axis::Col).map(|t| half(t.regs()));
            let bias = if self.bias_ring(bias) { chunks(Shape::new(self.bq, self.bkv))? } else { 0 };
            2 * chunks(Shape::new(self.bkv, d))? + bias
        };
        let (q, o) = (half(qk.a.regs()), pv.c.regs());
        let (k, s, p, v) = match self.core {
            Core::Warp => (half(qk.b.regs()), qk.c.regs(), half(pv.a.regs()), half(pv.b.regs())),
            Core::Group => (2 * qk.a.reps[1], qk.c.regs(), half(pv.a.regs()), 2 * pv.a.reps[1]),
        };
        Some(q + o + k + s + p + v + fill)
    }

    /// Of [`Self::registers`], the f32 output accumulator a lane carries
    /// across the key loop: what a matrix core with a file of its own
    /// (CDNA's AGPRs) may hold beside the rest.
    pub(crate) fn accumulators(&self, target: &Target, d: usize) -> Option<u32> {
        let grid = WarpGrid { rows: self.row_warps(), cols: 1 };
        Some(target.issue(Core::Warp, grid, self.bq, d, self.bkv)?.c.regs())
    }
}

/// Search bound: an attention block's queries.
pub(super) const QUERY_BLOCKS: [usize; 5] = [16, 32, 64, 128, 256];
/// Search bound: an attention block's keys a trip.
const KEY_BLOCKS: [usize; 4] = [16, 32, 64, 128];
/// Search bound: an attention ring's depths, the deepest first; a
/// register-staged fill runs the last alone.
pub(super) const FA_STAGES: [usize; 2] = [3, 2];
/// Search bound: the warps a one-row-warp block's keys may split over.
const KEY_WARPS: [usize; 3] = [1, 2, 4];

/// Search bound: the fewest queries a block offers two atom rows per warp for.
const TWO_ROWS_FROM: usize = 64;

/// Model: trips a key block on a causal query block's diagonal counts for: the
/// block itself, its mask, and the rows above the diagonal every warp still
/// runs. A count of the work, not a fit.
pub(crate) const DIAGONAL: usize = 3;

/// An attention problem: `tiles` independent query rows of `t` queries over
/// `tk` keys at head size `d` (`tiles` = batch × heads), `causal` when a
/// query sees the keys up to its own position, `bias` when an additive bias
/// streams beside the keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AttentionProblem {
    pub tiles: usize,
    pub t: usize,
    pub tk: usize,
    pub d: usize,
    pub causal: bool,
    pub bias: bool,
}

impl AttentionProblem {
    fn blocks(&self, cfg: &FaCfg) -> usize {
        self.tiles * self.t.div_ceil(cfg.bq)
    }

    /// What `cfg` costs here, as [`GemmProblem::admit`]: a key block's trip pays
    /// both products' fragment reads, fills (priced by where their bytes
    /// come from, every tile's keys and values the distinct ones; a bias
    /// tile is filled and read once per trip like them, every one of its
    /// bytes distinct) and steps, the softmax of every score and its
    /// overhead. A causal query block scores the keys its queries see on
    /// average plus its diagonal ([`Self::causal_trips`]). Key warps each take a share of the
    /// trips, every trip of theirs the block's whole work at once. The query
    /// blocks in flight at once share their tiles' keys and values whatever
    /// the cache holds of the whole; a wave's independent accumulators are
    /// the narrower product's, over every atom row the warp owns. A warp
    /// owning more rows reads each fragment for all of them: the fragment
    /// reads are per wave, so they divide by the rows per warp.
    pub fn admit(&self, planner: &Planner, cfg: &FaCfg) -> Result<(f64, u32), Rejected> {
        let claim = self.claim(planner, cfg)?;
        Ok((planner.price(&claim), claim.regs))
    }

    /// The config's [claim](super::cost::Claim): its checks run first — a
    /// config that lowers and fits the target's shared memory and register
    /// files — then its resources. A warpgroup product reads no key or
    /// value fragments and holds one chain per product.
    pub(crate) fn claim(&self, planner: &Planner, cfg: &FaCfg) -> Result<Claim, Rejected> {
        let target = &planner.device.target;
        let (bq, bkv, d, kw) = (cfg.bq, cfg.bkv, self.d, cfg.key_warps);
        let smem = cfg.smem_bytes(d, self.bias);
        if !cfg.fits(target, d, self.bias) {
            return Err(Rejected::Lowers);
        }
        if smem > target.smem_bytes {
            return Err(Rejected::Smem { need: smem, have: target.smem_bytes });
        }
        let regs = cfg.registers(target, d, self.bias).ok_or(Rejected::Layout)?;
        // Where the accumulators have a file of their own, the output's are
        // priced against it and the rest against the vector registers.
        if target.occupancy.acc_file > 0 {
            let acc = cfg.accumulators(target, d).ok_or(Rejected::Layout)?;
            if !target.occupancy.holds(acc, regs - acc) {
                return Err(Rejected::Registers {
                    need: regs,
                    have: target.occupancy.acc_file + target.occupancy.registers,
                });
            }
        }
        let waves = cfg.warps() as usize;
        // A warpgroup product reads no key or value fragments, holds one
        // chain per product, and is past `ptxas` beyond the block's share of
        // the register file.
        let (atoms, kv_reads) = match cfg.core {
            Core::Warp => {
                let atom = target.geometry().ok_or(Rejected::Lowers)?;
                let atoms = (cfg.warp_rows / atom.m as usize) * (bkv.min(d) / atom.n as usize);
                (atoms, 2 * waves * bkv * d * ELEM)
            }
            Core::Group if regs <= target.lane_budget(waves as u32) => (1, 0),
            Core::Group => return Err(Rejected::Registers { need: regs, have: target.lane_budget(waves as u32) }),
        };
        let bias = usize::from(self.bias) * kw * bq * bkv * ELEM;
        let fragments = kv_reads + bias;
        let fill = kw * 2 * bkv * d * ELEM + bias;
        let steps = kw * 2 * bq * bkv * d / STEP;
        let trips = if self.causal { self.causal_trips(bq, bkv) } else { self.tk.div_ceil(bkv) };
        let trips = trips.div_ceil(kw);
        let (blocks, stages) = (self.blocks(cfg), cfg.stages);
        let unique = (self.tiles * self.tk * (d * 2 + usize::from(self.bias) * self.t) * ELEM) as f64;
        let flight = planner.sms().min(blocks).max(1);
        let shared = 1.0 - flight.div_ceil(self.t.div_ceil(bq)) as f64 / flight as f64;
        Ok(Claim {
            blocks,
            smem,
            regs,
            waves,
            atoms,
            steps: match cfg.core {
                Core::Warp => Steps::F16(steps as f64),
                Core::Group => Steps::Group(steps as f64),
            },
            trips,
            reads: fragments as f64,
            ops: Ops::of(Kind::Pair, (kw * bq * bkv) as f64 * PAIR_OPS),
            fill: Stream { bytes: fill as f64, unique, shared },
            stores: 0.0,
            ring: stages,
            unroll: cfg.unroll,
            waits: 0.0,
            sections: [Spread::default(); 2],
        })
    }

    /// The key-block trips of a causal query block of `bq` rows over key
    /// blocks of `bkv`, on average over the blocks: the mask is aligned at
    /// the first query and key (query `i` sees keys `0..=i`), so a query
    /// sees `min(i + 1, tk)` keys and the `t` queries `seen · (2t − seen) / 2`
    /// on average with `seen = min(t, tk)`; plus the block's diagonal, a key
    /// block per `bkv` of its rows, each [`DIAGONAL`] trips.
    pub(crate) fn causal_trips(&self, bq: usize, bkv: usize) -> usize {
        let (t, seen) = (self.t, self.t.min(self.tk));
        (seen * (2 * t - seen)).div_ceil(2 * t * bkv) + DIAGONAL * bq.div_ceil(bkv)
    }

    /// Every config the ranking spans for this problem on `target`: every
    /// pipeline of every query and key block size at one atom row per warp,
    /// on the warp-level core and, where the target has it, the warpgroup
    /// core, in the plain softmax forms.
    pub fn lattice(&self, target: &Target) -> Vec<FaCfg> {
        let stages: &[usize] = if target.fill == Fill::CpAsync { &FA_STAGES } else { &FA_STAGES[1..] };
        let cores: &[Core] = if target.group_mma.is_empty() { &[Core::Warp] } else { &[Core::Warp, Core::Group] };
        let mut out = vec![];
        for (core, bq) in cores.iter().flat_map(|&c| QUERY_BLOCKS.map(|bq| (c, bq))) {
            // The warp-level core's blocks stop at eight row warps.
            if core == Core::Warp && bq > 128 {
                continue;
            }
            // A row warp of queries may share its block with key warps.
            let key_warps: &[usize] = if bq == WARP_ROWS && self.t <= WARP_ROWS { &KEY_WARPS } else { &[1] };
            for bkv in KEY_BLOCKS {
                for &stages in stages {
                    for unroll in [false, true] {
                        for &key_warps in key_warps {
                            out.push(FaCfg { unroll, key_warps, core, ..FaCfg::new(bq, bkv, stages) });
                        }
                    }
                }
            }
        }
        out
    }

    /// The [`lattice`](Self::lattice)'s configs that run this problem on the planner's target,
    /// cheapest first, then by fewer registers, every pipeline of every query
    /// and key block size at one atom row per warp, on the warp-level core
    /// and, where the target has it, the warpgroup core, in the plain softmax
    /// forms (the tuner sets the others); the shortlist keeps a tile's first
    /// clean one. After them, the same configs at two atom rows per warp
    /// where that can pay ([`Self::two_rows_pay`]), in their twins' order: a
    /// variant of a tile (its rows per warp are not part of it), so the
    /// shortlist holds the one-row representatives and the twins stage times
    /// the variant beside them for the contender tiles.
    pub(crate) fn ranked(&self, planner: &Planner) -> Vec<FaCfg> {
        let lattice = self.lattice(&planner.device.target);
        let count = lattice.len();
        let admitted: Vec<(f64, u32, FaCfg)> = lattice
            .into_iter()
            .filter_map(|c| self.admit(planner, &c).ok().map(|(cost, regs)| (cost, regs, c)))
            .collect();
        let one: Vec<FaCfg> = sorted(self, admitted, count).into_iter().map(|(_, _, c)| c).collect();
        let two = one
            .iter()
            .map(|&c| (c, FaCfg { warp_rows: 2 * WARP_ROWS, ..c }))
            .filter(|&(c, wide)| self.two_rows_pay(planner, &c, &wide))
            .map(|(_, wide)| wide)
            .collect::<Vec<_>>();
        [one, two].concat()
    }

    /// Whether `wide`, `one` at two atom rows per warp, can pay: the
    /// warp-level core (a warpgroup's rows are the instruction's), a block of
    /// at least [`TWO_ROWS_FROM`] queries (fewer leave half the warps a
    /// block holds, too few to hide a chain), one key warp, admitted, and as
    /// many blocks resident as `one` keeps (the wider tiles' registers must
    /// not cost the occupancy the shared fragments buy back).
    fn two_rows_pay(&self, planner: &Planner, one: &FaCfg, wide: &FaCfg) -> bool {
        let target = &planner.device.target;
        let resident = |cfg: &FaCfg| -> Option<usize> {
            let smem = cfg.smem_bytes(self.d, self.bias);
            Some(planner.resident(smem, cfg.registers(target, self.d, self.bias)?, cfg.warps() as usize))
        };
        one.core == Core::Warp
            && one.bq >= TWO_ROWS_FROM
            && one.key_warps == 1
            && self.admit(planner, wide).is_ok()
            && resident(wide).zip(resident(one)).is_some_and(|(w, o)| w >= o)
    }
}

impl Planner {
    /// Every attention config for `tiles` rows of `t` queries over `tk`
    /// keys at head size `d` ([`AttentionProblem`]), the cheapest first.
    pub fn attention_ranked(
        &self,
        tiles: usize,
        t: usize,
        tk: usize,
        d: usize,
        causal: bool,
        bias: bool,
    ) -> Arc<[FaCfg]> {
        if tiles * t * tk * d == 0 {
            return Arc::new([]);
        }
        let p = AttentionProblem { tiles, t, tk, d, causal, bias };
        cached(&self.attention, p, || p.ranked(self))
    }

    /// Key splits worth measuring for `tiles` independent query tiles over
    /// `blocks` key blocks: one, two, the counts around what keeps two blocks
    /// per SM busy, and every block its own split when that is within reach
    /// (the merge costs a launch, so never past the blocks).
    pub fn splits(&self, tiles: usize, blocks: usize) -> Vec<usize> {
        let want = (2 * self.sms()).div_ceil(tiles.max(1));
        let mut out: Vec<usize> = [1, 2, want / 2, want, 2 * want, blocks]
            .into_iter()
            .filter(|&s| (1..=blocks.min(2 * want)).contains(&s))
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }
}
