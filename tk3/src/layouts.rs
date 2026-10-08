//! Layout inference: every register tile gets the layout the atoms around it
//! need, free values adopt their consumer's, and a value two consumers want
//! differently is re-held through an inserted [`TileOp::Relayout`].

use std::collections::HashMap;

use snafu::{OptionExt, Snafu, ensure};

use crate::atoms::{MmaAtom, Target};
use crate::ir::*;
use crate::layout::Dim::{Col, Lane, Reg, Row, Warp};
use crate::layout::Layout;

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

/// The operand layouts of an `[m, n] += [m, k] · [k, n]` product on `atom`
/// over `grid`.
pub fn mma_layouts(atom: &MmaAtom, grid: WarpGrid, m: usize, n: usize, k: usize) -> Result<[TileLayout; 3]> {
    let (wr, wc) = (grid.rows, grid.cols);
    ensure!(
        m % (wr * atom.m) as usize == 0 && n % (wc * atom.n) as usize == 0,
        NotTileableSnafu { rows: m, cols: n, wr, wc, m: atom.m, n: atom.n }
    );
    ensure!(k % atom.k as usize == 0, ReductionNotTileableSnafu { k, atom_k: atom.k });
    let (rm, rn, rk) =
        ((m / (wr * atom.m) as usize) as u32, (n / (wc * atom.n) as usize) as u32, (k / atom.k as usize) as u32);
    let warps = grid.layout();
    Ok([
        TileLayout { frag: atom.a.clone(), reps: [rm, rk], warps: warps.sublayout(&[Warp], &[Row]) },
        TileLayout { frag: atom.b.clone(), reps: [rk, rn], warps: warps.sublayout(&[Warp], &[Col]) },
        TileLayout { frag: atom.c.clone(), reps: [rm, rn], warps },
    ])
}

/// A layout for a tile nothing constrains: each lane holds a short row
/// vector, lanes walk the columns then the rows, warps split the rows.
pub fn natural(shape: Shape, warps: u32, lanes: u32) -> Option<TileLayout> {
    let (rows, cols) = (shape.rows as u32, shape.cols as u32);
    if !rows.is_power_of_two() || !cols.is_power_of_two() {
        return None;
    }
    let v = cols.min(8);
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

struct Infer<'a> {
    prog: &'a mut Program,
    target: &'a Target,
    grid: WarpGrid,
    lay: Vec<Option<TileLayout>>,
    changed: bool,
    /// Operands whose current layout differs from the one a consumer needs.
    conflicts: HashMap<(usize, ValId), TileLayout>,
    /// Position of the statement being visited, for conflict keys.
    pos: usize,
}

/// Assign a layout to every register value of `prog`, inserting relayouts
/// where a value is consumed under two layouts. Returns the layouts by value.
pub fn infer(prog: &mut Program, target: &Target, grid: WarpGrid) -> Result<Vec<Option<TileLayout>>> {
    let n = prog.values.len();
    let mut it = Infer { prog, target, grid, lay: vec![None; n], changed: true, conflicts: HashMap::new(), pos: 0 };
    for _ in 0..32 {
        if !it.changed {
            break;
        }
        it.changed = false;
        it.conflicts.clear();
        it.pos = 0;
        let body = std::mem::take(&mut it.prog.body);
        it.pass(&body)?;
        it.prog.body = body;
    }
    if !it.conflicts.is_empty() {
        it.pos = 0;
        let body = std::mem::take(&mut it.prog.body);
        let body = it.insert(body);
        it.prog.body = body;
    }
    let (warps, lanes) = (it.prog.warps, it.target.wave);
    for (i, v) in it.prog.values.iter().enumerate() {
        if v.place == Place::Reg && it.lay[i].is_none() {
            let shape = v.shape;
            it.lay[i] = Some(natural(shape, warps, lanes).context(UndeterminedSnafu {
                value: ValId(i as u32),
                rows: shape.rows,
                cols: shape.cols,
            })?);
        }
    }
    Ok(it.lay)
}

impl Infer<'_> {
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
            TileOp::Fill(_) | TileOp::Coord(_) | TileOp::Relayout { .. } => {}
            TileOp::Unary { src, .. } | TileOp::Cast { src, .. } => {
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
            TileOp::Mma { acc, a, b, a_t, b_t } => {
                let (va, vc) = (self.prog.value(a).clone(), self.prog.value(dst).clone());
                let atom = self
                    .target
                    .mma(va.dtype, vc.dtype)
                    .context(NoMatrixCoreSnafu { arch: self.target.arch, dtype_in: va.dtype, dtype_out: vc.dtype })?
                    .clone();
                let k = if a_t { va.shape.rows } else { va.shape.cols };
                let [la, lb, lc] = mma_layouts(&atom, self.grid, vc.shape.rows, vc.shape.cols, k)?;
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
                    | TileOp::Relayout { src } => rewrite(self, src),
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
                    TileOp::Fill(_) | TileOp::Coord(_) => {}
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
