//! Layout inference: every register tile gets the layout the atoms around it
//! need, free values adopt their consumer's, and a value two consumers want
//! differently is re-held through an inserted [`TileOp::Relayout`].

use std::collections::HashMap;

use snafu::{OptionExt, Snafu, ensure};

use crate::atoms::{MmaAtom, Target};
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
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Warps along the rows and columns of a block tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WarpGrid {
    pub rows: u32,
    pub cols: u32,
}

impl WarpGrid {
    /// Warp index → warp sub-tile coordinate: the low bits walk columns.
    pub fn layout(self) -> Layout {
        let lc = self.cols.trailing_zeros();
        let lr = self.rows.trailing_zeros();
        let bases: Vec<[u32; 2]> = (0..lc).map(|i| [0, 1 << i]).chain((0..lr).map(|i| [1 << i, 0])).collect();
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
        match produced.relayout(self, warps, lanes) {
            Relayout::Identity => Some((0..self.regs()).collect()),
            Relayout::RegPermute(perm) => Some(perm),
            Relayout::LaneShuffle(_) | Relayout::ViaSmem => None,
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

    /// How `self` becomes `dst`, for `warps` warps of `lanes` lanes.
    pub fn relayout(&self, dst: &Self, warps: u32, lanes: u32) -> Relayout {
        assert_eq!(self.shape(), dst.shape(), "relayout between different tiles");
        if self == dst {
            return Relayout::Identity;
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
        let mut sources = Vec::with_capacity((warps * lanes * dst.regs()) as usize);
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
                    sources.push((pick.0, pick.1, pick.2));
                }
                uniform &= lane_perm.get_or_insert_with(|| perm.clone()) == &perm;
            }
        }
        let perm = lane_perm.unwrap_or_default();
        if cross_warp {
            Relayout::ViaSmem
        } else if cross_lane || !uniform {
            Relayout::LaneShuffle(sources)
        } else if perm.iter().enumerate().all(|(j, &p)| j as u32 == p) {
            Relayout::Identity
        } else {
            Relayout::RegPermute(perm)
        }
    }
}

/// What re-holding a tile costs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Relayout {
    Identity,
    /// Destination register `j` is source register `perm[j]` of the same lane.
    RegPermute(Vec<u32>),
    /// Per `(warp, lane, dst reg)` in row-major order: the `(warp, lane, reg)` it reads.
    LaneShuffle(Vec<(u32, u32, u32)>),
    ViaSmem,
}

impl Relayout {
    /// What moving one element this way is worth avoiding, for the inference
    /// to weigh one assignment against another. The classes are ordered, not
    /// calibrated, and a lane shuffle is priced as a round trip because that
    /// is what the emitter lowers one into today.
    pub fn cost_per_element(&self) -> u32 {
        match self {
            Relayout::Identity => 0,
            Relayout::RegPermute(_) => 1,
            Relayout::LaneShuffle(_) | Relayout::ViaSmem => 16,
        }
    }
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
    /// The `[m, n] += [m, k] · [k, n]` product over `grid`, issued `orient`
    /// way round. Swapped, our A feeds the slot expecting `atom.b` (as Aᵀ)
    /// and our B the one expecting `atom.a`, so each is held in the other's
    /// transpose and the accumulator in its own; the warp grid and repeat
    /// counts are the direct form's either way, so no warp's work moves.
    /// Whether an accumulator feeds the next product without moving: the
    /// accumulator of an `m × k` product as its A operand directly, or that
    /// of a `k × n` product as the B operand of the swapped product (both
    /// products issued swapped, the accumulator holds the transpose).
    pub fn accumulator_feeds(&self, lanes: u32) -> bool {
        let (m, n, k) = (self.m as usize, self.n as usize, self.k as usize);
        let one = WarpGrid { rows: 1, cols: 1 };
        let feeds = |rows, cols, operand: fn(&Issue) -> &TileLayout| {
            self.issue(Orient::Direct, one, rows, cols, k)
                .is_ok_and(|p| matches!(p.c.relayout(operand(&p), 1, lanes), Relayout::Identity))
        };
        feeds(m, k, |p| &p.a) || (self.swappable() && feeds(k, n, |p| &p.b))
    }

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

/// A layout for a tile nothing constrains: each lane holds a short row
/// vector, lanes walk the columns then the rows, warps split the rows. The
/// vector is as wide as a wave can spread over one row, up to 8 elements,
/// so a narrow row still occupies a whole warp rather than replicating.
pub fn natural(shape: Shape, warps: u32, lanes: u32) -> Option<TileLayout> {
    let (rows, cols) = (shape.rows as u32, shape.cols as u32);
    if !rows.is_power_of_two() || !cols.is_power_of_two() {
        return None;
    }
    let v = (cols / lanes).clamp(1, 8);
    let lane_cols = (cols / v).min(lanes);
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

struct Infer<'a> {
    prog: &'a Program,
    target: &'a Target,
    grid: WarpGrid,
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
/// decides whether the next product takes its result as it stands. The
/// assignment that forces the least relayout traffic wins — every assignment
/// is tried while the free products are few, one flip at a time beyond that —
/// so an accumulator that feeds another product reaches it in the layout that
/// product wants, which on a target whose accumulator is its operand's
/// transpose is free (what tk1's `acc_reusable_as_input` flag used to say).
/// Products the author oriented stay as pinned.
pub fn infer(mut prog: Program, target: &Target, grid: WarpGrid) -> Result<Laid> {
    orient_products(&mut prog, target, grid)?;
    let mut it = Infer::new(&prog, target, grid);
    it.solve()?;
    let (mut layouts, conflicts) = (it.lay, it.conflicts);
    if !conflicts.is_empty() {
        let body = std::mem::take(&mut prog.body);
        let body = Rewrite { prog: &mut prog, lay: &mut layouts, conflicts, pos: 0 }.insert(body);
        prog.body = body;
    }
    Ok(Laid { prog, layouts })
}

/// Up to this many free products every orientation is tried; past it each is
/// flipped on its own and a flip that lowers the cost is kept.
const EXHAUSTIVE_UP_TO: usize = 6;

/// Write an orientation into every product of `prog`: the pinned ones stay,
/// the rest take the assignment of least relayout cost.
fn orient_products(prog: &mut Program, target: &Target, grid: WarpGrid) -> Result<()> {
    // A product listed twice (a peeled loop trip) is one decision.
    let mut free: Vec<ValId> = vec![];
    for (_, s) in prog.walk() {
        if let Stmt::Let { dst, op: TileOp::Mma { a, orient: None, .. } } = s {
            let (va, vd) = (prog.value(*a), prog.value(*dst));
            if target.mma(va.dtype, vd.dtype).is_some_and(MmaAtom::swappable) && !free.contains(dst) {
                free.push(*dst);
            }
        }
    }
    let cost_of = |choice: &HashMap<ValId, Orient>| -> Result<u32> {
        let mut trial = prog.clone();
        set_orient(&mut trial.body, choice);
        let mut it = Infer::new(&trial, target, grid);
        it.solve()?;
        Ok(it
            .conflicts
            .iter()
            .map(|((_, v), want)| match &it.lay[v.index()] {
                Some(have) => {
                    let shape = trial.value(*v).shape;
                    have.relayout(want, trial.warps, target.wave).cost_per_element() * (shape.rows * shape.cols) as u32
                }
                None => 0,
            })
            .sum())
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
            Stmt::Let { .. } | Stmt::Copy { .. } | Stmt::Sync(_) | Stmt::Raw(_) => {}
            Stmt::Loop(l) => set_orient(&mut l.body, choice),
            Stmt::Pipeline(p) => {
                set_orient(&mut p.produce.body, choice);
                set_orient(&mut p.consume.body, choice);
            }
            Stmt::Role { body, .. } => set_orient(body, choice),
            Stmt::If { then, otherwise, .. } => {
                set_orient(then, choice);
                set_orient(otherwise, choice);
            }
        }
    }
}

impl<'a> Infer<'a> {
    fn new(prog: &'a Program, target: &'a Target, grid: WarpGrid) -> Self {
        let n = prog.values.len();
        Infer { prog, target, grid, lay: vec![None; n], changed: true, conflicts: HashMap::new(), pos: 0 }
    }

    /// Run the propagation to its fixed point, seeding what nothing constrains.
    fn solve(&mut self) -> Result<()> {
        let prog = self.prog;
        let n = self.lay.len();
        let (warps, lanes) = (prog.warps, self.target.wave);
        for _ in 0..4 * n + 4 {
            if !self.changed {
                // At a fixed point, a value nothing constrains takes the natural
                // layout and the passes go on, so what consumes it (a vector it
                // broadcasts over, say) is demanded from it rather than defaulted
                // apart from it. Tiles seed before vectors: a vector's layout is
                // its tile's to decide.
                let Some(v) = self.unconstrained() else { break };
                let shape = prog.values[v.index()].shape;
                let l = natural(shape, warps, lanes).context(UndeterminedSnafu {
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
        Ok(())
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

    /// `a` and `b` are held alike; a disagreement is resolved on `b`'s side.
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
                Stmt::Loop(l) => {
                    self.carried(&l.carried)?;
                    self.pass(&l.body)?;
                }
                Stmt::Pipeline(p) => {
                    self.carried(&p.carried)?;
                    self.pass(&p.produce.body)?;
                    self.pass(&p.consume.body)?;
                }
                Stmt::Role { body, .. } => self.pass(body)?,
                Stmt::If { then, otherwise, .. } => {
                    self.pass(then)?;
                    self.pass(otherwise)?;
                }
                Stmt::Copy { dst, src, mode: CopyMode::Staged } if self.lay[dst.index()].is_none() => {
                    let (d, s) = (self.prog.value(*dst), self.prog.value(*src));
                    if d.place == Place::Reg && s.tier() == Tier::Global {
                        let (warps, lanes) = (self.prog.warps, self.target.wave);
                        if let Some(l) = chunked(d.shape, d.dtype.bytes(), warps, lanes, Axis::Col) {
                            self.set(*dst, l);
                        }
                    }
                }
                Stmt::Copy { .. } | Stmt::Sync(_) | Stmt::Raw(_) => {}
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
                self.unify(a, b);
                self.unify(a, dst);
                self.unify(dst, a);
            }
            Some((tile, vector, axis)) => {
                self.unify(tile, dst);
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
            TileOp::Unary { src, .. } | TileOp::Cast { src, .. } | TileOp::Move { src } => {
                self.unify(src, dst);
                self.unify(dst, src);
            }
            TileOp::Binary { a, b, .. } => self.elementwise(dst, a, b),
            TileOp::Where { pred, a, b } => {
                self.elementwise(dst, a, b);
                self.unify(dst, pred);
            }
            TileOp::Reduce { src, axis, .. } => {
                if let Some(want) = self.vector_of(src, axis) {
                    self.demand(dst, want);
                }
            }
            TileOp::Transpose { src } => {
                if let Some(l) = self.lay[src.index()].clone() {
                    self.demand(dst, l.transposed());
                } else if let Some(l) = self.lay[dst.index()].clone() {
                    self.set(src, l.transposed());
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
                    TileOp::Mma { acc, a, b, .. } => {
                        rewrite(self, acc);
                        rewrite(self, a);
                        rewrite(self, b);
                    }
                    TileOp::Reduce { src, .. } => rewrite(self, src),
                    TileOp::Fill(_) | TileOp::Splat(_) | TileOp::Coord(_) => {}
                },
                Stmt::Loop(l) => l.body = self.insert(std::mem::take(&mut l.body)),
                Stmt::Pipeline(p) => {
                    p.produce.body = self.insert(std::mem::take(&mut p.produce.body));
                    p.consume.body = self.insert(std::mem::take(&mut p.consume.body));
                }
                Stmt::Role { body, .. } => *body = self.insert(std::mem::take(body)),
                Stmt::If { then, otherwise, .. } => {
                    *then = self.insert(std::mem::take(then));
                    *otherwise = self.insert(std::mem::take(otherwise));
                }
                Stmt::Copy { .. } | Stmt::Sync(_) | Stmt::Raw(_) => {}
            }
            out.push(stmt);
        }
        Block(out)
    }
}
