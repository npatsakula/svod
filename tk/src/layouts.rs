//! Layout inference: every register tile gets the layout the atoms around it
//! need, free values adopt their consumer's, and a value two consumers want
//! differently is re-held through an inserted [`TileOp::Relayout`].

use std::collections::{HashMap, HashSet};

use snafu::{OptionExt, Snafu, ensure};

use crate::atoms::{Core, Costs, Device, GroupMmaAtom, LaneMove, MmaAtom, Target};
use crate::ir::*;
use crate::layout::Dim::{Col, Lane, Reg, Row, Warp};
use crate::layout::{self as frag, Layout};

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("no {dtype_in:?} → {dtype_out:?} matrix core on {arch:?}"))]
    NoMatrixCore { arch: svod_dtype::GpuArch, dtype_in: svod_dtype::ScalarDType, dtype_out: svod_dtype::ScalarDType },
    #[snafu(display("a {rows}×{cols} tile does not tile a {wr}×{wc} warp grid of {m}×{n} atoms"))]
    NotTileable { rows: usize, cols: usize, wr: u32, wc: u32, m: u32, n: u32 },
    #[snafu(display("the reduction dim {k} is not a multiple of the atom's {atom_k}"))]
    ReductionNotTileable { k: usize, atom_k: u32 },
    #[snafu(display("loop-carried value {value:?} changes layout inside the loop"))]
    CarriedLayoutChanges { value: ValId },
    #[snafu(display("value {value:?} ({rows}×{cols}) has no layout and no natural one"))]
    Undetermined { value: ValId, rows: usize, cols: usize },
    #[snafu(display("a warpgroup product takes {what}"))]
    GroupOperand { what: String },
    #[snafu(display("the layouts reach no fixed point in {passes} passes"))]
    NotConverged { passes: usize },
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Warps along the rows and columns of a block tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WarpGrid {
    pub rows: u32,
    pub cols: u32,
}

impl WarpGrid {
    pub fn count(self) -> u32 {
        self.rows * self.cols
    }

    /// Warp index → warp sub-tile coordinate: the low bits walk columns.
    pub fn layout(self) -> Layout {
        let lc = self.cols.trailing_zeros();
        let lr = self.rows.trailing_zeros();
        let bases: Vec<[u32; 2]> = (0..lc).map(|i| [0, 1 << i]).chain((0..lr).map(|i| [1 << i, 0])).collect();
        Layout::from_bases([(Row, self.rows), (Col, self.cols)], &[(Warp, &bases)])
    }

    /// The grid as warpgroups of four consecutive warps down the rows: the
    /// two low warp bits walk a group's rows, the bits above walk the
    /// groups, columns first. `rows` is a multiple of four.
    pub fn group_layout(self) -> Layout {
        assert!(self.rows.is_multiple_of(GroupMmaAtom::WARPS), "a {}×{} grid of warpgroups", self.rows, self.cols);
        let lc = self.cols.trailing_zeros();
        let lr = (self.rows / GroupMmaAtom::WARPS).trailing_zeros();
        let bases: Vec<[u32; 2]> = [[1, 0], [2, 0]]
            .into_iter()
            .chain((0..lc).map(|i| [0, 1 << i]))
            .chain((0..lr).map(|i| [GroupMmaAtom::WARPS << i, 0]))
            .collect();
        Layout::from_bases([(Row, self.rows), (Col, self.cols)], &[(Warp, &bases)])
    }
}

/// How a block holds a register tile: a fragment over `(Reg, Lane)`, repeated
/// `reps` times (row-major, by register) inside each warp's sub-tile, and the
/// warps' sub-tile coordinates (free warp bits replicate the sub-tile).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileLayout {
    pub frag: Layout,
    pub reps: [u32; 2],
    pub warps: Layout,
}

impl TileLayout {
    pub fn frag_shape(&self) -> [u32; 2] {
        [self.frag.out_size(Row), self.frag.out_size(Col)]
    }

    pub fn sub_shape(&self) -> [u32; 2] {
        let [r, c] = self.frag_shape();
        [r * self.reps[0], c * self.reps[1]]
    }

    pub fn shape(&self) -> Shape {
        let [r, c] = self.sub_shape();
        Shape::new((r * self.warps.out_size(Row)) as usize, (c * self.warps.out_size(Col)) as usize)
    }

    pub fn frag_regs(&self) -> u32 {
        self.frag.in_size(Reg)
    }

    pub fn regs(&self) -> u32 {
        self.frag_regs() * self.reps[0] * self.reps[1]
    }

    /// The `(row, col)` register `reg` of lane `lane` in warp `warp` holds.
    pub fn coord(&self, warp: u32, lane: u32, reg: u32) -> (u32, u32) {
        let regs = self.frag_regs();
        let rep = reg / regs;
        let (rr, rc) = (rep / self.reps[1], rep % self.reps[1]);
        let (fr, fc) = self.frag.apply(&[(Lane, lane), (Reg, reg % regs)]);
        let (wr, wc) = self.warps.apply(&[(Warp, warp)]);
        let [fr_n, fc_n] = self.frag_shape();
        let [sr, sc] = self.sub_shape();
        (wr * sr + rr * fr_n + fr, wc * sc + rc * fc_n + fc)
    }

    /// Register runs `(start, width)` of elements consecutive along `along`
    /// (columns of one row, or rows of one column) in every lane, aligned to
    /// their width and at most `max_width` long.
    pub fn runs(&self, lanes: u32, max_width: u32, along: Axis) -> Vec<(u32, u32)> {
        let pos = |(r, c): (u32, u32)| if along == Axis::Col { c } else { r };
        let step = |(r, c): (u32, u32), e: u32| if along == Axis::Col { (r, c + e) } else { (r + e, c) };
        let mut out = vec![];
        let mut j = 0;
        while j < self.regs() {
            let mut w = max_width.min(self.regs() - j);
            while w > 1 {
                let ok = (0..lanes).all(|lane| {
                    let at = self.coord(0, lane, j);
                    pos(at) % w == 0 && (1..w).all(|e| self.coord(0, lane, j + e) == step(at, e))
                });
                if ok {
                    break;
                }
                w /= 2;
            }
            out.push((j, w));
            j += w;
        }
        out
    }

    /// The register permutation under which this layout is `produced` (what
    /// an instruction leaves a tile in), if it is one lane for lane.
    pub fn as_permutation_of(&self, produced: &Self, warps: u32, lanes: u32) -> Option<Vec<u32>> {
        match produced.held(self, warps, lanes) {
            Held::InLane(perm) => Some(perm),
            Held::AcrossLanes { .. } | Held::AcrossWarps => None,
        }
    }

    pub fn transposed(&self) -> Self {
        Self { frag: self.frag.transpose(), reps: [self.reps[1], self.reps[0]], warps: self.warps.transpose() }
    }

    /// The layout of a `[rows, 1]` reduction of this tile: registers and
    /// lanes that only told columns apart drop out or replicate.
    pub fn row_vector(&self) -> Self {
        Self { frag: keep_axis(&self.frag, Row), reps: [self.reps[0], 1], warps: self.warps.sublayout(&[Warp], &[Row]) }
    }

    /// A `[rows, cols]` tile whose [`Self::row_vector`] is `vector` (a
    /// `[rows, 1]` layout): a lane holds runs of a row's consecutive columns
    /// in its registers, and the lanes that hold the same rows of `vector`
    /// hold its successive runs, so a row's columns are held once rather
    /// than in every lane sharing it, and only a run's first column has its
    /// predecessor in another lane. The columns' power-of-two part spreads
    /// so; the registers repeat the rest.
    pub fn spread_row(vector: &Self, cols: u32) -> Self {
        let (frag, rows) = (&vector.frag, vector.frag.out_size(Row));
        let row_of = |dim, bit| frag.basis(dim, bit).first().map_or(0, |o| o.1);
        let free = (0..frag.in_bits(Lane)).filter(|&b| row_of(Lane, b) == 0).count() as u32;
        let part = cols.trailing_zeros();
        let (spread, run) = (free.min(part), part - free.min(part));
        let regs: Vec<[u32; 2]> =
            (0..frag.in_bits(Reg)).map(|b| [row_of(Reg, b), 0]).chain((0..run).map(|b| [0, 1 << b])).collect();
        let mut taken = 0;
        let lanes: Vec<[u32; 2]> = (0..frag.in_bits(Lane))
            .map(|b| match row_of(Lane, b) {
                0 if taken < spread => {
                    taken += 1;
                    [0, 1 << (run + taken - 1)]
                }
                r => [r, 0],
            })
            .collect();
        let frag = Layout::from_bases([(Row, rows), (Col, 1 << part)], &[(Reg, &regs), (Lane, &lanes)]);
        Self { frag, reps: [vector.reps[0], cols >> part], warps: vector.warps.clone() }
    }

    /// The moves of a [`TileOp::Shift`] along `axis`: per register, the
    /// lane-xor moves bringing each lane the element one step back, as an
    /// [`Relayout::Exchange`]'s, and the lanes whose element is the first
    /// (the fill); `None` where an element's predecessor is in another warp
    /// or the moves differ between warps.
    pub fn shift(&self, axis: Axis, warps: u32, lanes: u32) -> Option<Vec<(Vec<Pick>, u64)>> {
        let back = |(r, c): (u32, u32)| match axis {
            Axis::Col => c.checked_sub(1).map(|c| (r, c)),
            Axis::Row => r.checked_sub(1).map(|r| (r, c)),
        };
        // The (mask, register) moves that bring lane `l` of warp `w` element `j`'s predecessor.
        let moves = |w: u32, l: u32, j: u32| -> Option<Vec<(u32, u32)>> {
            let want = back(self.coord(w, l, j))?;
            let found = (0..lanes).flat_map(|from| (0..self.regs()).map(move |reg| (from, reg)));
            Some(
                found
                    .filter(|&(from, reg)| self.coord(w, from, reg) == want)
                    .map(|(from, reg)| (from ^ l, reg))
                    .collect(),
            )
        };
        (0..self.regs())
            .map(|j| {
                let fill = (0..lanes).filter(|&l| moves(0, l, j).is_none()).fold(0u64, |a, l| a | 1 << l);
                let mut open: u64 = (u64::MAX >> (64 - lanes)) & !fill;
                let mut picks: Vec<Pick> = vec![];
                while open != 0 {
                    let lanes_of = |mv: (u32, u32)| {
                        (0..lanes)
                            .filter(|&m| open >> m & 1 == 1 && moves(0, m, j).is_some_and(|v| v.contains(&mv)))
                            .fold(0u64, |a, m| a | 1 << m)
                    };
                    // The same lane first, then the move serving the most lanes still open.
                    let (mask, reg) = (0..lanes)
                        .filter(|l| open >> l & 1 == 1)
                        .flat_map(|l| moves(0, l, j).unwrap_or_default())
                        .max_by_key(|&mv| (lanes_of(mv).count_ones(), mv.0 == 0))?;
                    let covered = lanes_of((mask, reg));
                    open &= !covered;
                    picks.push(Pick { mask, reg, lanes: covered });
                }
                let every_warp = (1..warps).all(|w| {
                    (0..lanes).all(|l| match moves(w, l, j) {
                        None => fill >> l & 1 == 1,
                        Some(v) => picks.iter().any(|p| p.lanes >> l & 1 == 1 && v.contains(&(p.mask, p.reg))),
                    })
                });
                every_warp.then_some((picks, fill))
            })
            .collect()
    }

    pub fn col_vector(&self) -> Self {
        Self { frag: keep_axis(&self.frag, Col), reps: [1, self.reps[1]], warps: self.warps.sublayout(&[Warp], &[Col]) }
    }

    /// Lane xor-masks whose butterfly folds every row (registers that share a
    /// row fold first, in-lane).
    pub fn row_fold_masks(&self) -> smallvec::SmallVec<[u32; 6]> {
        crate::layout::lanes_sharing_row(&self.frag)
    }

    pub fn col_fold_masks(&self) -> smallvec::SmallVec<[u32; 6]> {
        crate::layout::lanes_sharing_row(&self.frag.transpose())
    }

    /// The fold masks of a reduction along `axis`: [`Self::row_fold_masks`]
    /// over rows, [`Self::col_fold_masks`] over columns.
    pub fn fold_masks(&self, axis: Axis) -> smallvec::SmallVec<[u32; 6]> {
        match axis {
            Axis::Row => self.row_fold_masks(),
            Axis::Col => self.col_fold_masks(),
        }
    }

    /// The lane moves each lane issues reducing `self` along `axis` into
    /// `dst`: a butterfly over the [`Self::fold_masks`] per register of
    /// `dst`, each mask moved as `how` moves it.
    pub fn fold_moves(&self, dst: &Self, axis: Axis, how: LaneMove) -> u32 {
        dst.regs() * self.fold_masks(axis).iter().map(|&m| how.issues(m)).sum::<u32>()
    }

    /// How `self` becomes `dst`, for `warps` warps of `lanes` lanes: lanes
    /// trade registers by lane moves while that is cheaper per element than
    /// a round trip through shared memory.
    pub fn relayout(&self, dst: &Self, warps: u32, lanes: u32, costs: &Costs) -> Relayout {
        match self.held(dst, warps, lanes) {
            Held::InLane(perm) if is_identity(&perm) => Relayout::Identity,
            Held::InLane(perm) => Relayout::RegPermute(perm),
            Held::AcrossLanes(holders) => match self.exchange(dst, &holders, warps, lanes).map(Relayout::Exchange) {
                Some(exchange) if exchange.cost_per_element(costs) < Relayout::ViaSmem.cost_per_element(costs) => {
                    exchange
                }
                _ => Relayout::ViaSmem,
            },
            Held::AcrossWarps => Relayout::ViaSmem,
        }
    }

    /// Where every register of `dst` is held in `self`, preferring the same
    /// lane, then the same warp.
    fn held(&self, dst: &Self, warps: u32, lanes: u32) -> Held {
        assert_eq!(self.shape(), dst.shape(), "relayout between different tiles");
        if self == dst {
            return Held::InLane((0..dst.regs()).collect());
        }
        let Shape { rows, cols } = self.shape();
        let mut holders: Vec<Vec<(u32, u32, u32)>> = vec![vec![]; rows * cols];
        for w in 0..warps {
            for l in 0..lanes {
                for j in 0..self.regs() {
                    let (r, c) = self.coord(w, l, j);
                    holders[r as usize * cols + c as usize].push((w, l, j));
                }
            }
        }
        let (mut cross_lane, mut cross_warp) = (false, false);
        let mut lane_perm: Option<Vec<u32>> = None;
        let mut uniform = true;
        for w in 0..warps {
            for l in 0..lanes {
                let mut perm = Vec::with_capacity(dst.regs() as usize);
                for j in 0..dst.regs() {
                    let (r, c) = dst.coord(w, l, j);
                    let held = &holders[r as usize * cols + c as usize];
                    let pick = held
                        .iter()
                        .find(|h| (h.0, h.1) == (w, l))
                        .or_else(|| held.iter().find(|h| h.0 == w))
                        .or_else(|| held.first())
                        .expect("the source holds every element");
                    cross_lane |= pick.1 != l;
                    cross_warp |= pick.0 != w;
                    perm.push(pick.2);
                }
                uniform &= lane_perm.get_or_insert_with(|| perm.clone()) == &perm;
            }
        }
        if cross_warp {
            Held::AcrossWarps
        } else if cross_lane || !uniform {
            Held::AcrossLanes(holders)
        } else {
            Held::InLane(lane_perm.unwrap_or_default())
        }
    }

    /// Every destination register as lane-xor moves of one source register
    /// each, the same in every warp: lanes trading registers with fixed
    /// partners. `holders` lists who holds each element.
    fn exchange(&self, dst: &Self, holders: &[Vec<(u32, u32, u32)>], warps: u32, lanes: u32) -> Option<Vec<Vec<Pick>>> {
        let cols = self.shape().cols;
        // The (mask, source register) moves that bring lane `l` of warp `w` its element `j`.
        let moves = |w: u32, l: u32, j: u32| -> Vec<(u32, u32)> {
            let (r, c) = dst.coord(w, l, j);
            holders[r as usize * cols + c as usize].iter().filter(|h| h.0 == w).map(|h| (h.1 ^ l, h.2)).collect()
        };
        (0..dst.regs())
            .map(|j| {
                // Greedy cover: the move serving the most lanes still open, until none are.
                let mut picks: Vec<Pick> = vec![];
                let mut open: u64 = u64::MAX >> (64 - lanes);
                while open != 0 {
                    let lanes_of = |mv: (u32, u32)| {
                        (0..lanes)
                            .filter(|&m| open >> m & 1 == 1 && moves(0, m, j).contains(&mv))
                            .fold(0u64, |a, m| a | 1 << m)
                    };
                    let (mask, reg) = (0..lanes)
                        .filter(|l| open >> l & 1 == 1)
                        .flat_map(|l| moves(0, l, j))
                        .max_by_key(|&mv| lanes_of(mv).count_ones())?;
                    let covered = lanes_of((mask, reg));
                    open &= !covered;
                    picks.push(Pick { mask, reg, lanes: covered });
                }
                let every_warp = (1..warps).all(|w| {
                    picks.iter().all(|p| {
                        (0..lanes).filter(|l| p.lanes >> l & 1 == 1).all(|l| moves(w, l, j).contains(&(p.mask, p.reg)))
                    })
                });
                every_warp.then_some(picks)
            })
            .collect()
    }
}

/// Where a destination layout's registers are held in a source layout.
enum Held {
    /// In the same lane of every warp, destination register `j` at source
    /// register `perm[j]`.
    InLane(Vec<u32>),
    /// In other lanes of the same warp: who holds each element.
    AcrossLanes(Vec<Vec<(u32, u32, u32)>>),
    AcrossWarps,
}

/// One lane-xor move of a [`Relayout::Exchange`]: the lanes in `lanes` take
/// source register `reg` of lane `lane ^ mask`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pick {
    pub mask: u32,
    pub reg: u32,
    pub lanes: u64,
}

/// What re-holding a tile costs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Relayout {
    Identity,
    /// Destination register `j` is source register `perm[j]` of the same lane.
    RegPermute(Vec<u32>),
    /// Per destination register, the lane-xor moves that fill it, selected by lane.
    Exchange(Vec<Vec<Pick>>),
    /// A round trip through shared memory: across warps, or across lanes
    /// where no exchange is cheaper.
    ViaSmem,
}

impl Relayout {
    /// What moving one element this way costs, for the inference to weigh
    /// one assignment against another: its [`Self::work`] in its primitive's
    /// cost.
    pub fn cost_per_element(&self, costs: &Costs) -> f64 {
        let unit = match self {
            Relayout::Exchange(_) => costs.lane_move,
            Relayout::ViaSmem => costs.smem_trip,
            _ => 0.0,
        };
        self.work() * unit
    }

    /// The primitives the emitter issues per element: none within a lane
    /// (the compiler renames an in-lane permutation), one round trip through
    /// shared memory across lanes or warps. An exchange moves each distinct
    /// `(mask, reg)` of a non-zero mask once, shared by every destination
    /// register that picks it, and selects among a register's picks, a
    /// select counting as a move; its work is that over its registers.
    pub fn work(&self) -> f64 {
        match self {
            Relayout::Identity | Relayout::RegPermute(_) => 0.0,
            Relayout::Exchange(picks) => {
                let moves: HashSet<(u32, u32)> =
                    picks.iter().flatten().filter(|p| p.mask != 0).map(|p| (p.mask, p.reg)).collect();
                let selects: usize = picks.iter().map(|p| p.len().saturating_sub(1)).sum();
                (moves.len() + selects) as f64 / picks.len().max(1) as f64
            }
            Relayout::ViaSmem => 1.0,
        }
    }
}

/// Whether register `j` stays register `j`.
fn is_identity(perm: &[u32]) -> bool {
    perm.iter().enumerate().all(|(j, &p)| j as u32 == p)
}

/// Keep the input bits that tell `axis` apart; the rest of the registers are
/// dropped (they would be copies) and the rest of the lanes replicate.
fn keep_axis(frag: &Layout, axis: crate::layout::Dim) -> Layout {
    let only = frag.sublayout(&[Reg, Lane], &[axis]);
    let size = only.out_size(axis);
    let reg: Vec<[u32; 1]> = (0..only.in_bits(Reg))
        .filter_map(|b| {
            let v = only.basis(Reg, b).first().map_or(0, |o| o.1);
            (v != 0).then_some([v])
        })
        .collect();
    let lane: Vec<[u32; 1]> =
        (0..only.in_bits(Lane)).map(|b| [only.basis(Lane, b).first().map_or(0, |o| o.1)]).collect();
    Layout::from_bases([(axis, size)], &[(Reg, &reg), (Lane, &lane)])
}

/// A product as the matrix core issues it: the layout each operand is held
/// in and whether the slots are exchanged, from [`MmaAtom::issue`], so the
/// layouts and the slot order agree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Issue {
    pub a: TileLayout,
    pub b: TileLayout,
    pub c: TileLayout,
    /// The core computes `Cᵀ = Bᵀ·Aᵀ`: `b` feeds its A slot and `a` its B
    /// slot.
    pub swapped: bool,
}

impl MmaAtom {
    /// Whether an accumulator feeds the next product without moving: the
    /// accumulator of an `m × k` product as its A operand directly, or that
    /// of a `k × n` product as the B operand of the swapped product (both
    /// products issued swapped, the accumulator holds the transpose).
    pub fn accumulator_feeds(&self, lanes: u32) -> bool {
        let (m, n, k) = (self.m as usize, self.n as usize, self.k as usize);
        let one = WarpGrid { rows: 1, cols: 1 };
        let feeds = |rows, cols, operand: fn(&Issue) -> &TileLayout| {
            self.issue(Orient::Direct, one, rows, cols, k)
                .is_ok_and(|p| operand(&p).as_permutation_of(&p.c, 1, lanes).is_some_and(|perm| is_identity(&perm)))
        };
        feeds(m, k, |p| &p.a) || (self.swappable() && feeds(k, n, |p| &p.b))
    }

    /// The `[m, n] += [m, k] · [k, n]` product over `grid`, issued `orient`
    /// way round. Swapped, our A feeds the slot expecting `atom.b` (as Aᵀ)
    /// and our B the one expecting `atom.a`, so each is held in the other's
    /// transpose and the accumulator in its own; the warp grid and repeat
    /// counts are the direct form's either way, so no warp's work moves.
    pub fn issue(&self, orient: Orient, grid: WarpGrid, m: usize, n: usize, k: usize) -> Result<Issue> {
        assert!(orient == Orient::Direct || self.swappable(), "a {}×{} atom cannot issue swapped", self.m, self.n);
        let (wr, wc) = (grid.rows, grid.cols);
        ensure!(
            m.is_multiple_of((wr * self.m) as usize) && n.is_multiple_of((wc * self.n) as usize),
            NotTileableSnafu { rows: m, cols: n, wr, wc, m: self.m, n: self.n }
        );
        ensure!(k.is_multiple_of(self.k as usize), ReductionNotTileableSnafu { k, atom_k: self.k });
        let (rm, rn, rk) =
            ((m / (wr * self.m) as usize) as u32, (n / (wc * self.n) as usize) as u32, (k / self.k as usize) as u32);
        let warps = grid.layout();
        let (fa, fb, fc, swapped) = match orient {
            Orient::Direct => (self.a.clone(), self.b.clone(), self.c.clone(), false),
            Orient::Swapped => (self.b.transpose(), self.a.transpose(), self.c.transpose(), true),
        };
        Ok(Issue {
            a: TileLayout { frag: fa, reps: [rm, rk], warps: warps.sublayout(&[Warp], &[Row]) },
            b: TileLayout { frag: fb, reps: [rk, rn], warps: warps.sublayout(&[Warp], &[Col]) },
            c: TileLayout { frag: fc, reps: [rm, rn], warps },
            swapped,
        })
    }
}

impl GroupMmaAtom {
    /// The `[m, n] += [m, k] · [k, n]` product over `grid` as warpgroups
    /// ([`WarpGrid::group_layout`]): each group holds 64 rows of the
    /// accumulator and `n / cols` columns as [`Self::n_for`] instructions
    /// repeated along the columns; a register `A` holds a group's rows as
    /// [`frag::wgmma_a`] repeated along `k`. The `b` layout is the shape's
    /// placeholder: `B` is read through a descriptor and never held.
    pub fn issue(&self, grid: WarpGrid, m: usize, n: usize, k: usize) -> Result<Issue> {
        let (wr, wc) = (grid.rows, grid.cols);
        let groups = wr / Self::WARPS;
        let n_atom = (n as u32).checked_div(wc).and_then(|cols| self.n_for(cols));
        ensure!(
            wr.is_multiple_of(Self::WARPS) && m == (Self::M * groups) as usize && n_atom.is_some(),
            NotTileableSnafu { rows: m, cols: n, wr, wc, m: Self::M, n: self.n_max }
        );
        ensure!(k.is_multiple_of(Self::K as usize), ReductionNotTileableSnafu { k, atom_k: Self::K });
        let n_atom = n_atom.expect("checked");
        let (rn, rk) = (n as u32 / wc / n_atom, k as u32 / Self::K);
        let warps = grid.group_layout();
        Ok(Issue {
            a: TileLayout { frag: self.a(), reps: [1, rk], warps: warps.sublayout(&[Warp], &[Row]) },
            b: TileLayout {
                frag: frag::mma_sync_b(),
                reps: [rk, n as u32 / wc / 8],
                warps: warps.sublayout(&[Warp], &[Col]),
            },
            c: TileLayout { frag: self.c(n_atom), reps: [1, rn], warps },
            swapped: false,
        })
    }
}

/// The XOR swizzle of a row-major shared row: its 16-byte chunks exchanged
/// against bits of the row index, so the rows a fragment read meets fall in
/// distinct banks (CuTe's `Swizzle<B, 4, 3>` in the modes a warpgroup
/// descriptor names). Decided here alone: the emitter's shared addresses,
/// the descriptors and the planner's admissible warpgroup rows read it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Swizzle {
    /// A row other than a power-of-two count of chunks, two at least: an
    /// XOR would leave the row.
    None,
    /// 32-byte rows: chunk `^= (row >> 2) & 1`.
    B32,
    /// 64-byte rows: chunk `^= (row >> 1) & 3`.
    B64,
    /// 128-byte spans: chunk `^= row & 7`, on every span of a longer row.
    B128,
}

impl Swizzle {
    /// The swizzle of a row of `cols` elements of `bytes` each.
    pub fn of_row(cols: usize, bytes: usize) -> Self {
        let chunk = 16 / bytes;
        match cols / chunk {
            chunks if !cols.is_multiple_of(chunk) || !chunks.is_power_of_two() => Swizzle::None,
            0 | 1 => Swizzle::None,
            2 => Swizzle::B32,
            4 => Swizzle::B64,
            _ => Swizzle::B128,
        }
    }

    /// The bits of a row's chunk index the XOR flips.
    pub fn bits(self) -> u32 {
        match self {
            Swizzle::None => 0,
            Swizzle::B32 => 1,
            Swizzle::B64 => 2,
            Swizzle::B128 => 3,
        }
    }

    /// The bytes one swizzle pattern spans: 32, 64 or 128 (16 unswizzled).
    pub fn span(self) -> usize {
        16 << self.bits()
    }

    /// Whether a row of `cols` elements of `bytes` each is one whole
    /// swizzle span, as a warpgroup descriptor reads a row.
    pub fn spans(cols: usize, bytes: usize) -> bool {
        let swizzle = Self::of_row(cols, bytes);
        swizzle != Swizzle::None && swizzle.span() == cols * bytes
    }
}

/// A layout for a tile nothing constrains: each lane holds a short row
/// vector, lanes walk the columns then the rows, warps split the rows. The
/// vector is as wide as a wave can spread over one row, up to 8 elements,
/// so a narrow row still occupies a whole warp rather than replicating. A
/// width that is not a power of two spreads its power-of-two part and
/// repeats across the rest (48 columns: 16 three times).
pub fn natural(shape: Shape, warps: u32, lanes: u32) -> Option<TileLayout> {
    let (rows, cols) = (shape.rows as u32, shape.cols as u32);
    if !rows.is_power_of_two() || cols == 0 {
        return None;
    }
    let spread = 1u32 << cols.trailing_zeros();
    let v = (spread / lanes).clamp(1, 8);
    let lane_cols = (spread / v).min(lanes);
    let lane_rows = (lanes / lane_cols).min(rows);
    let warp_rows = (rows / lane_rows).min(warps);
    let frag = Layout::identity(Reg, v, Col)
        .product(&Layout::identity(Lane, lane_cols, Col))
        .product(&Layout::identity(Lane, lane_rows, Row))
        .product(&Layout::zeros(Lane, lanes / (lane_cols * lane_rows)));
    let warps_layout = Layout::identity(Warp, warp_rows, Row).product(&Layout::zeros(Warp, warps / warp_rows));
    let reps = [rows / (lane_rows * warp_rows), cols / (v * lane_cols)];
    Some(TileLayout { frag, reps, warps: warps_layout })
}

/// The layout of a register-staged fill: each lane holds a 16-byte run of
/// a row (the `cp.async` chunk), consecutive lanes walk the runs along
/// `walk` first (`Col`: along the row, as a row-major tile is written;
/// `Row`: down the rows, so lanes writing a column-major tile store
/// consecutive elements), then warps, and registers repeat the pattern down
/// the rows and across the columns. Dims only need a power-of-two part as
/// wide as the lanes and warps use, so 48- and 96-wide tiles repeat by
/// three; warps past the tile replicate.
pub fn chunked(shape: Shape, elem_bytes: usize, warps: u32, lanes: u32, walk: Axis) -> Option<TileLayout> {
    let pow2 = |n: u32| 1u32 << n.trailing_zeros();
    let (rows, cols) = (shape.rows as u32, shape.cols as u32);
    if rows == 0 || cols == 0 {
        return None;
    }
    let v = (16 / elem_bytes as u32).min(pow2(cols));
    let (lane_rows, lane_cols) = match walk {
        Axis::Col => {
            let lane_cols = pow2(cols / v).min(lanes);
            ((lanes / lane_cols).min(pow2(rows)), lane_cols)
        }
        Axis::Row => {
            let lane_rows = pow2(rows).min(lanes);
            (lane_rows, (lanes / lane_rows).min(pow2(cols / v)))
        }
    };
    let (first, second) = match walk {
        Axis::Col => (Layout::identity(Lane, lane_cols, Col), Layout::identity(Lane, lane_rows, Row)),
        Axis::Row => (Layout::identity(Lane, lane_rows, Row), Layout::identity(Lane, lane_cols, Col)),
    };
    // Warps tile the rows the lanes left, then the columns, so a tile a wave
    // covers in rows is divided among the warps instead of written by each.
    let warp_rows = pow2(rows / lane_rows).min(warps);
    let warp_cols = pow2(cols / (v * lane_cols)).min(warps / warp_rows);
    let frag = Layout::identity(Reg, v, Col)
        .product(&first)
        .product(&second)
        .product(&Layout::zeros(Lane, lanes / (lane_cols * lane_rows)));
    let warps_layout = Layout::identity(Warp, warp_rows, Row)
        .product(&Layout::identity(Warp, warp_cols, Col))
        .product(&Layout::zeros(Warp, warps / (warp_rows * warp_cols)));
    let reps = [rows / (lane_rows * warp_rows), cols / (v * lane_cols * warp_cols)];
    Some(TileLayout { frag, reps, warps: warps_layout })
}

/// The layout transposing 16×16 loads ([`frag::global_tr_b128`]) leave a
/// `shape` tile of 16-bit elements in: every lane holds 16-byte column runs,
/// and the blocks are divided among the warps, rows first. Wave32 only, and
/// `None` unless 16 divides both sides.
pub fn transposing(shape: Shape, warps: u32, lanes: u32) -> Option<TileLayout> {
    let pow2 = |n: u32| 1u32 << n.trailing_zeros();
    let (rows, cols) = (shape.rows as u32, shape.cols as u32);
    if lanes != 32 || rows == 0 || cols == 0 || rows % 16 != 0 || cols % 16 != 0 {
        return None;
    }
    let warp_rows = pow2(rows / 16).min(warps);
    let warp_cols = pow2(cols / 16).min(warps / warp_rows);
    let warps_layout = Layout::identity(Warp, warp_rows, Row)
        .product(&Layout::identity(Warp, warp_cols, Col))
        .product(&Layout::zeros(Warp, warps / (warp_rows * warp_cols)));
    let reps = [rows / (16 * warp_rows), cols / (16 * warp_cols)];
    Some(TileLayout { frag: frag::global_tr_b128(), reps, warps: warps_layout })
}

pub(crate) struct Infer<'a> {
    prog: &'a Program,
    target: &'a Target,
    grid: WarpGrid,
    /// The matrix instruction products issue on.
    core: Core,
    lay: Vec<Option<TileLayout>>,
    changed: bool,
    /// Operands whose current layout differs from the one a consumer needs.
    conflicts: HashMap<(usize, ValId), TileLayout>,
    /// Position of the statement being visited, for conflict keys.
    pos: usize,
}

/// A program whose register values all have a layout and whose products are
/// all oriented: what [`infer`] makes and the emitter consumes.
pub struct Laid {
    pub prog: Program,
    pub layouts: Vec<Option<TileLayout>>,
}

/// Assign a layout to every register value of `prog` and an [`Orient`] to
/// every product, inserting relayouts where a value is consumed under two
/// layouts.
///
/// A product may be issued either way round at no cost, and which way it goes
/// decides whether the next product takes its result as it stands and how
/// many lanes a reduction of it folds across. The assignment of the least
/// relayout traffic, scattered loads and fold moves wins — every assignment
/// is tried while the free products are few, one flip at a time beyond that —
/// so an accumulator that feeds another product reaches it in the layout that
/// product wants, which on a target whose accumulator is its operand's
/// transpose is free.
/// Products the author oriented stay as pinned.
pub fn infer(mut prog: Program, device: &Device, grid: WarpGrid, core: Core) -> Result<Laid> {
    orient_products(&mut prog, device, grid, core)?;
    let mut it = Infer::new(&prog, &device.target, grid, core);
    it.solve()?;
    let (mut layouts, conflicts) = (it.lay, it.conflicts);
    if !conflicts.is_empty() {
        let body = std::mem::take(&mut prog.body);
        let body = Rewrite { prog: &mut prog, lay: &mut layouts, conflicts, pos: 0 }.insert(body);
        prog.body = body;
    }
    Ok(Laid { prog, layouts })
}

/// Search bound: the propagation passes a register value may take to reach
/// the fixed point before the inference gives up.
const PASSES_PER_VALUE: usize = 4;

/// Search bound: up to this many free products every orientation is tried; past it each is
/// flipped on its own and a flip that lowers the cost is kept.
const EXHAUSTIVE_UP_TO: usize = 6;

/// Whether lanes reading global view `from` into layout `lay` scatter: each
/// lane's run along the view's unit-stride axis is under 16 bytes and the next
/// lane does not continue it, so a wave's request touches a line per lane.
fn scatters(prog: &Program, lay: &TileLayout, from: &Value, lanes: u32) -> bool {
    let Place::Global { stride, .. } = from.place else { return false };
    let unit = |s: ScalarId| matches!(prog.scalar(s), Scalar::Const(1));
    let along = if unit(stride[1]) {
        Axis::Col
    } else if unit(stride[0]) {
        Axis::Row
    } else {
        return false;
    };
    let bytes = from.dtype.bytes() as u32;
    let run = lay.runs(lanes, (16 / bytes).max(1), along).first().map_or(1, |r| r.1);
    let ((r0, c0), next) = (lay.coord(0, 0, 0), lay.coord(0, 1, 0));
    let continued = if along == Axis::Col { next == (r0, c0 + run) } else { next == (r0 + run, c0) };
    run * bytes < 16 && !continued
}

/// Write an orientation into every product of `prog`: the pinned ones stay,
/// the rest take the assignment of least relayout, scatter and fold cost.
fn orient_products(prog: &mut Program, device: &Device, grid: WarpGrid, core: Core) -> Result<()> {
    let Device { target, costs } = device;
    // A product listed twice (a peeled loop trip) is one decision; a
    // warpgroup product is never square, so it issues one way.
    let mut free: Vec<ValId> = vec![];
    for (_, s) in prog.walk() {
        if let Stmt::Let { dst, op: TileOp::Mma { a, orient: None, .. } } = s {
            let (va, vd) = (prog.value(*a), prog.value(*dst));
            if core == Core::Warp
                && target.mma(va.dtype, vd.dtype).is_some_and(MmaAtom::swappable)
                && !free.contains(dst)
            {
                free.push(*dst);
            }
        }
    }
    let cost_of = |choice: &HashMap<ValId, Orient>| -> Result<f64> {
        let mut trial = prog.clone();
        set_orient(&mut trial.body, choice);
        let mut it = Infer::new(&trial, target, grid, core);
        it.solve()?;
        let relayouts: f64 = it
            .conflicts
            .iter()
            .map(|((_, v), want)| match &it.lay[v.index()] {
                Some(have) => {
                    let elems = trial.value(*v).shape.elems() as f64;
                    have.relayout(want, grid.count(), target.wave, costs).cost_per_element(costs) * elems
                }
                None => 0.0,
            })
            .sum();
        // A register tile loaded from global memory in a layout whose lanes
        // scatter costs a memory line per lane.
        let scattered: f64 = trial
            .walk()
            .filter_map(|(_, s)| match s {
                Stmt::Copy { dst, src, .. } if trial.value(*dst).tier() == Tier::Reg => {
                    let (lay, from) = (it.lay[dst.index()].as_ref()?, trial.value(*src));
                    scatters(&trial, lay, from, target.wave).then(|| from.shape.elems() as f64)
                }
                _ => None,
            })
            .sum();
        // A reduction folds across the lanes that share its rows (or
        // columns): what the emitter issues per lane, over every lane.
        let folds: f64 = trial
            .walk()
            .filter_map(|(_, s)| match s {
                Stmt::Let { dst, op: TileOp::Reduce { src, axis, .. } } => {
                    let (ls, ld) = (it.lay[src.index()].as_ref()?, it.lay[dst.index()].as_ref()?);
                    let moves = ls.fold_moves(ld, *axis, target.lane_move) * target.wave * grid.count();
                    Some(moves as f64)
                }
                _ => None,
            })
            .sum();
        Ok(relayouts + costs.scatter * scattered + costs.lane_move * folds)
    };
    let mut best: HashMap<ValId, Orient> = HashMap::new();
    let mut best_cost = cost_of(&best)?;
    let swapped = |bits: u32| -> HashMap<ValId, Orient> {
        free.iter().enumerate().filter(|(i, _)| bits >> i & 1 == 1).map(|(_, &v)| (v, Orient::Swapped)).collect()
    };
    if free.len() <= EXHAUSTIVE_UP_TO {
        for bits in 1..1u32 << free.len() {
            let choice = swapped(bits);
            if let Ok(cost) = cost_of(&choice)
                && cost < best_cost
            {
                (best, best_cost) = (choice, cost);
            }
        }
    } else {
        for &v in &free {
            let mut choice = best.clone();
            choice.insert(v, Orient::Swapped);
            if let Ok(cost) = cost_of(&choice)
                && cost < best_cost
            {
                (best, best_cost) = (choice, cost);
            }
        }
    }
    set_orient(&mut prog.body, &best);
    Ok(())
}

/// Orient every free product of `block`: as `choice` says, else direct.
fn set_orient(block: &mut Block, choice: &HashMap<ValId, Orient>) {
    for stmt in &mut block.0 {
        match stmt {
            Stmt::Let { dst, op: TileOp::Mma { orient: orient @ None, .. } } => {
                *orient = Some(choice.get(dst).copied().unwrap_or(Orient::Direct));
            }
            nested => nested.blocks_mut().for_each(|b| set_orient(b, choice)),
        }
    }
}

impl<'a> Infer<'a> {
    pub(crate) fn new(prog: &'a Program, target: &'a Target, grid: WarpGrid, core: Core) -> Self {
        let mut lay = vec![None; prog.values.len()];
        for (v, l) in &prog.pins {
            lay[v.index()] = Some(l.clone());
        }
        Infer { prog, target, grid, core, lay, changed: true, conflicts: HashMap::new(), pos: 0 }
    }

    /// Run the propagation to its fixed point, seeding what nothing constrains.
    fn solve(&mut self) -> Result<()> {
        self.solve_within(PASSES_PER_VALUE * (self.lay.len() + 1))
    }

    /// [`Self::solve`] within `passes` passes after the first.
    pub(crate) fn solve_within(&mut self, passes: usize) -> Result<()> {
        let prog = self.prog;
        let (warps, lanes) = (self.grid.count(), self.target.wave);
        for _ in 0..=passes {
            if !self.changed {
                // At a fixed point, a value nothing constrains takes the natural
                // layout and the passes go on, so what consumes it (a vector it
                // broadcasts over, say) is demanded from it rather than defaulted
                // apart from it. Tiles seed before vectors: a vector's layout is
                // its tile's to decide.
                // A tile a laid vector spreads into first: its loop's
                // initial value and its next take its layout from it.
                let spread = self.open_tiles().find_map(|v| Some((v, self.spread(v)?)));
                let Some(v) = spread.as_ref().map(|s| s.0).or_else(|| self.unconstrained()) else { return Ok(()) };
                let shape = prog.values[v.index()].shape;
                let l = spread.map(|s| s.1).or_else(|| natural(shape, warps, lanes)).context(UndeterminedSnafu {
                    value: v,
                    rows: shape.rows,
                    cols: shape.cols,
                })?;
                self.set(v, l);
            }
            self.changed = false;
            self.conflicts.clear();
            self.pos = 0;
            self.pass(&prog.body)?;
        }
        NotConvergedSnafu { passes }.fail()
    }

    /// For a tile `v` nothing constrains that a laid row vector broadcasts
    /// over (or a column vector, transposed), the tile it [spreads
    /// into](TileLayout::spread_row): a list of a row's values held beside
    /// that row's.
    fn spread(&self, v: ValId) -> Option<TileLayout> {
        let cols = self.shape(v).cols as u32;
        self.prog.walk().find_map(|(_, stmt)| {
            let Stmt::Let { dst, op } = stmt else { return None };
            let pairs = match *op {
                TileOp::Binary { a, b, .. } | TileOp::Where { a, b, .. } | TileOp::MulAdd { a, b, .. } => [a, b],
                _ => return None,
            };
            let (tile, vector, axis) = self.broadcast(pairs[0], pairs[1])?;
            let vector = self.lay[vector.index()].as_ref()?;
            ((tile == v || *dst == v) && self.lay[dst.index()].is_none()).then(|| match axis {
                Axis::Row => TileLayout::spread_row(vector, cols),
                Axis::Col => TileLayout::spread_row(&vector.transposed(), self.shape(v).rows as u32).transposed(),
            })
        })
    }

    /// The register tiles (not vectors) without a layout.
    fn open_tiles(&self) -> impl Iterator<Item = ValId> + '_ {
        self.prog.values.iter().enumerate().filter_map(|(i, v)| {
            let open = v.place == Place::Reg && self.lay[i].is_none() && v.shape.rows != 1 && v.shape.cols != 1;
            open.then_some(ValId(i as u32))
        })
    }

    /// The first register value without a layout, tiles before vectors.
    fn unconstrained(&self) -> Option<ValId> {
        let open = |vector: bool| {
            self.prog.values.iter().enumerate().find(|(i, v)| {
                v.place == Place::Reg && self.lay[*i].is_none() && (v.shape.rows == 1 || v.shape.cols == 1) == vector
            })
        };
        open(false).or_else(|| open(true)).map(|(i, _)| ValId(i as u32))
    }

    fn set(&mut self, v: ValId, l: TileLayout) {
        if self.lay[v.index()].as_ref() != Some(&l) {
            self.lay[v.index()] = Some(l);
            self.changed = true;
        }
    }

    /// `v` must be held as `want` at the current statement.
    fn demand(&mut self, v: ValId, want: TileLayout) {
        match &self.lay[v.index()] {
            None => self.set(v, want),
            Some(have) if *have != want => {
                self.conflicts.insert((self.pos, v), want);
            }
            Some(_) => {}
        }
    }

    /// Hold `a` and `b` alike: an unlaid one takes the other's layout, and
    /// where both are laid differently `b` is re-held as `a` before the
    /// statement. An op unifies its result with each operand in that order,
    /// so the operands reach it in the result's layout (a conflict is only
    /// ever resolved on a statement's operand).
    fn unify(&mut self, a: ValId, b: ValId) {
        match (self.lay[a.index()].clone(), self.lay[b.index()].clone()) {
            (Some(l), None) => self.set(b, l),
            (None, Some(l)) => self.set(a, l),
            (Some(x), Some(y)) if x != y => {
                self.conflicts.insert((self.pos, b), x);
            }
            _ => {}
        }
    }

    fn carried(&mut self, carried: &[Carried]) -> Result<()> {
        for c in carried {
            for (x, y) in [(c.init, c.phi), (c.next, c.phi)] {
                match (self.lay[x.index()].clone(), self.lay[y.index()].clone()) {
                    (Some(l), None) => self.set(y, l),
                    (None, Some(l)) => self.set(x, l),
                    (Some(a), Some(b)) => ensure!(a == b, CarriedLayoutChangesSnafu { value: c.phi }),
                    _ => {}
                }
            }
        }
        Ok(())
    }

    fn pass(&mut self, block: &Block) -> Result<()> {
        for stmt in &block.0 {
            self.pos += 1;
            match stmt {
                Stmt::Let { dst, op } => self.let_(*dst, op)?,
                Stmt::Loop(Loop { carried, .. }) | Stmt::Pipeline(Pipeline { carried, .. }) => self.carried(carried)?,
                Stmt::Copy { dst, src, mode: CopyMode::Staged, scope } if self.lay[dst.index()].is_none() => {
                    let (d, s) = (self.prog.value(*dst), self.prog.value(*src));
                    if d.place == Place::Reg && s.tier() == Tier::Global {
                        let warps = if *scope == Scope::Warp { 1 } else { self.prog.warps };
                        if let Some(l) = chunked(d.shape, d.dtype.bytes(), warps, self.target.wave, Axis::Col) {
                            self.set(*dst, l);
                        }
                    }
                }
                _ => {}
            }
            for b in stmt.blocks() {
                self.pass(b)?;
            }
        }
        Ok(())
    }

    fn shape(&self, v: ValId) -> Shape {
        self.prog.value(v).shape
    }

    /// The vector/tile pairing of a broadcasting binary op: `(tile, vector, axis)`.
    fn broadcast(&self, a: ValId, b: ValId) -> Option<(ValId, ValId, Axis)> {
        let (sa, sb) = (self.shape(a), self.shape(b));
        if sa == sb {
            None
        } else if sb.cols == 1 && sb.rows == sa.rows {
            Some((a, b, Axis::Row))
        } else if sb.rows == 1 && sb.cols == sa.cols {
            Some((a, b, Axis::Col))
        } else if sa.cols == 1 {
            Some((b, a, Axis::Row))
        } else {
            Some((b, a, Axis::Col))
        }
    }

    fn vector_of(&self, tile: ValId, axis: Axis) -> Option<TileLayout> {
        self.lay[tile.index()].as_ref().map(|l| match axis {
            Axis::Row => l.row_vector(),
            Axis::Col => l.col_vector(),
        })
    }

    fn elementwise(&mut self, dst: ValId, a: ValId, b: ValId) {
        match self.broadcast(a, b) {
            None => {
                self.unify(dst, a);
                self.unify(dst, b);
            }
            Some((tile, vector, axis)) => {
                self.unify(dst, tile);
                if let Some(want) = self.vector_of(tile, axis) {
                    self.demand(vector, want);
                }
            }
        }
    }

    fn let_(&mut self, dst: ValId, op: &TileOp) -> Result<()> {
        match *op {
            TileOp::Fill(_) | TileOp::Splat(_) | TileOp::Coord(_) | TileOp::Relayout { .. } => {}
            TileOp::Unary { src, .. } | TileOp::Cast { src, .. } | TileOp::Move { src } | TileOp::Shift { src, .. } => {
                self.unify(dst, src)
            }
            TileOp::Binary { a, b, .. } => self.elementwise(dst, a, b),
            TileOp::Where { pred, a, b } => {
                self.elementwise(dst, a, b);
                self.unify(dst, pred);
            }
            TileOp::MulAdd { a, b, c } => {
                self.elementwise(dst, a, b);
                self.unify(dst, c);
            }
            TileOp::Reduce { src, axis, .. } => {
                if let Some(want) = self.vector_of(src, axis) {
                    self.demand(dst, want);
                }
            }
            // The result is held under its operand's transposed layout; one
            // laid otherwise has the operand re-held to match it.
            TileOp::Transpose { src } => match (self.lay[src.index()].clone(), self.lay[dst.index()].clone()) {
                (_, Some(l)) => self.demand(src, l.transposed()),
                (Some(l), None) => self.set(dst, l.transposed()),
                (None, None) => {}
            },
            TileOp::Mma { acc, a, b, a_t, .. } if self.core == Core::Group => {
                let (va, vc) = (self.prog.value(a).clone(), self.prog.value(dst).clone());
                let atom = self
                    .target
                    .group_mma(va.dtype, vc.dtype)
                    .context(NoMatrixCoreSnafu { arch: self.target.arch, dtype_in: va.dtype, dtype_out: vc.dtype })?
                    .clone();
                ensure!(
                    self.prog.value(b).tier() == Tier::Smem && !a_t,
                    GroupOperandSnafu { what: "a shared B operand and an untransposed A" }
                );
                let k = va.shape.cols;
                let Issue { a: la, c: lc, .. } = atom.issue(self.grid, vc.shape.rows, vc.shape.cols, k)?;
                self.demand(acc, lc.clone());
                self.demand(dst, lc);
                if self.prog.value(a).place == Place::Reg {
                    self.demand(a, la);
                }
            }
            TileOp::Mma { acc, a, b, a_t, b_t, orient } => {
                let (va, vc) = (self.prog.value(a).clone(), self.prog.value(dst).clone());
                let atom = self
                    .target
                    .mma(va.dtype, vc.dtype)
                    .context(NoMatrixCoreSnafu { arch: self.target.arch, dtype_in: va.dtype, dtype_out: vc.dtype })?
                    .clone();
                let k = if a_t { va.shape.rows } else { va.shape.cols };
                let orient = orient.expect("products are oriented before their layouts are solved");
                let Issue { a: la, b: lb, c: lc, .. } =
                    atom.issue(orient, self.grid, vc.shape.rows, vc.shape.cols, k)?;
                self.demand(acc, lc.clone());
                self.demand(dst, lc);
                if self.prog.value(a).place == Place::Reg {
                    self.demand(a, if a_t { la.transposed() } else { la });
                }
                if self.prog.value(b).place == Place::Reg {
                    self.demand(b, if b_t { lb.transposed() } else { lb });
                }
            }
        }
        Ok(())
    }
}

/// The rewrite that follows a solved inference: a fresh register value for
/// each conflicting operand, re-held just before its consumer.
struct Rewrite<'a> {
    prog: &'a mut Program,
    lay: &'a mut Vec<Option<TileLayout>>,
    conflicts: HashMap<(usize, ValId), TileLayout>,
    pos: usize,
}

impl Rewrite<'_> {
    /// Re-hold every conflicting operand just before its consumer.
    fn insert(&mut self, block: Block) -> Block {
        let mut out = Vec::with_capacity(block.0.len());
        for mut stmt in block.0 {
            self.pos += 1;
            let pos = self.pos;
            let mut rewrite = |it: &mut Self, v: &mut ValId| {
                if let Some(want) = it.conflicts.remove(&(pos, *v)) {
                    let Value { dtype, shape, .. } = it.prog.value(*v).clone();
                    it.prog.values.push(Value { dtype, shape, place: Place::Reg });
                    let fresh = ValId(it.prog.values.len() as u32 - 1);
                    it.lay.push(Some(want));
                    out.push(Stmt::Let { dst: fresh, op: TileOp::Relayout { src: *v } });
                    *v = fresh;
                }
            };
            match &mut stmt {
                Stmt::Let { op, .. } => match op {
                    TileOp::Unary { src, .. }
                    | TileOp::Cast { src, .. }
                    | TileOp::Transpose { src }
                    | TileOp::Shift { src, .. }
                    | TileOp::Relayout { src }
                    | TileOp::Move { src } => rewrite(self, src),
                    TileOp::Binary { a, b, .. } => {
                        rewrite(self, a);
                        rewrite(self, b);
                    }
                    TileOp::Where { pred, a, b } => {
                        rewrite(self, pred);
                        rewrite(self, a);
                        rewrite(self, b);
                    }
                    TileOp::MulAdd { a, b, c } => {
                        rewrite(self, a);
                        rewrite(self, b);
                        rewrite(self, c);
                    }
                    TileOp::Mma { acc, a, b, .. } => {
                        rewrite(self, acc);
                        rewrite(self, a);
                        rewrite(self, b);
                    }
                    TileOp::Reduce { src, .. } => rewrite(self, src),
                    TileOp::Fill(_) | TileOp::Splat(_) | TileOp::Coord(_) => {}
                },
                nested => {
                    for b in nested.blocks_mut() {
                        *b = self.insert(std::mem::take(b));
                    }
                }
            }
            out.push(stmt);
        }
        Block(out)
    }
}
