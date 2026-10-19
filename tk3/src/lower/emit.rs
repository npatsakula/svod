//! Statement-by-statement emission into a pre-linearized instruction list.
//! Register tiles are per-thread register buffers addressed by register
//! index; every statement's instructions are listed in program order, so no
//! toposort decides anything.

use std::collections::HashMap;
use std::sync::Arc;

use smallvec::smallvec;
use svod_codegen::llvm::amd::tr::global_load_tr_b128;
use svod_codegen::llvm::nvptx::ops::shfl_bfly;
use svod_codegen::llvm::nvptx::smem::{
    cp_async_16, cp_async_16_zfill, cp_async_commit, cp_async_wait, ld_global_cg_16, ldmatrix,
};
use svod_codegen::llvm::nvptx::wgmma;
use svod_dtype::{AddrSpace, DType, DeviceSpec, GpuArch, ScalarDType};
use svod_ir::{AxisId, AxisType, ConstValue, KernelInfo, Op, UOp, WmmaMetadata, ops};

use super::{DivergentBarrierSnafu, Lowering, ProgramSnafu, Result, UnsupportedSnafu};
use crate::atoms::LaneMove;
use crate::index::*;
use crate::ir::*;
use crate::layout::Dim::{Col, Lane, Reg, Row, Warp};
use crate::layout::{self as frag, Layout};
use crate::layouts::{Laid, Relayout, TileLayout};
use crate::schedule::Prefetch;
use snafu::{OptionExt, ResultExt};

pub fn emit(laid: &Laid, low: &Lowering, params: Vec<Arc<UOp>>, device: DeviceSpec) -> Result<Arc<UOp>> {
    let (prog, lay) = (&laid.prog, laid.layouts.as_slice());
    let mut e = Emit::new(prog, lay, low, params);
    e.prologue()?;
    e.block(&prog.body)?;
    if std::env::var_os("TK3_DUMP_LIST").is_some() {
        for (i, u) in e.list.iter().enumerate() {
            let srcs: Vec<u64> = u.op().sources().iter().map(|s| s.id).collect();
            eprintln!("[{i:4}] id={} {:?} {:?} <- {srcs:?}", u.id, u.op().as_ref(), u.dtype());
        }
    }
    let info = KernelInfo { name: Some(prog.name.clone()), ..Default::default() };
    UOp::linear_program(info, e.list, device).context(ProgramSnafu)
}

struct Emit<'a> {
    prog: &'a Program,
    lay: &'a [Option<TileLayout>],
    low: &'a Lowering,
    params: Vec<Arc<UOp>>,
    vars: HashMap<String, Arc<UOp>>,
    list: Vec<Arc<UOp>>,
    /// Insertion point of values computed at each nesting level: the
    /// kernel's, then every open loop's body and branch of an `If`.
    headers: Vec<usize>,
    /// The levels of the open branches, innermost last.
    guards: Vec<usize>,
    /// Open branches on the warp of the block's scope: some warps skip them.
    divergent: usize,
    /// The level every hoisted value sits at.
    levels: HashMap<u64, usize>,
    scalars: HashMap<ScalarId, (usize, Arc<UOp>)>,
    /// Scalars over [`Scalar::Row`], memoized per row value they were taken at.
    row_scalars: HashMap<(ScalarId, u64), (usize, Arc<UOp>)>,
    /// Whether each scalar reads [`Scalar::Row`].
    row_dep: Vec<bool>,
    inductions: HashMap<ScalarId, (usize, Arc<UOp>)>,
    /// Register buffer of every register value (carried values share one).
    regs: HashMap<ValId, Arc<UOp>>,
    alias: HashMap<ValId, ValId>,
    /// The carried values of every open loop, innermost last.
    carried: Vec<Vec<Carried>>,
    /// Per open block: carried values defined in it, moved at its end.
    pending: Vec<Vec<Carried>>,
    smem: Vec<Arc<UOp>>,
    /// A shared allocation's buffer read under another type, per pair.
    typed: HashMap<(usize, ScalarDType), Arc<UOp>>,
    scratch: HashMap<ScalarDType, (Arc<UOp>, usize)>,
    lane: Arc<UOp>,
    warp: Arc<UOp>,
    tid: Arc<UOp>,
    /// The block's index along each grid axis, once taken ([`Self::block_index`]).
    blocks: [Option<Arc<UOp>>; 3],
    /// Whose threads the statement being emitted belongs to: the block's, or
    /// one warp's own inside a [`Stmt::Warps`].
    scope: Scope,
    /// The block being emitted holds a [`Sync::WaitProducts`]: its warpgroup
    /// products issue without waiting, that statement is their wait.
    async_products: bool,
    range_id: usize,
    reg_slot: usize,
    local_slot: usize,
    tag: u64,
    last: Arc<UOp>,
}

/// A register-side view of a tiled value for addressing shared/global memory.
struct View {
    buf: Arc<UOp>,
    /// Where the program holds the tile; `None` for the relayout scratch,
    /// no allocation of the program: a row-major tile at its start.
    place: Option<Place>,
    shape: Shape,
    dtype: ScalarDType,
    /// Added to a shared view's offset: a warp's region of the relayout
    /// scratch inside a [`Stmt::Warps`].
    region: Option<Arc<UOp>>,
}

/// A register's coordinate in its tile: the part the lane and warp decide,
/// and the constant the register adds to each dimension.
struct Coord {
    row: Arc<UOp>,
    col: Arc<UOp>,
    dr: i64,
    dc: i64,
}

impl Coord {
    fn plain(row: Arc<UOp>, col: Arc<UOp>) -> Self {
        Self { row, col, dr: 0, dc: 0 }
    }

    /// The coordinate with its constant added.
    fn full(&self) -> (Arc<UOp>, Arc<UOp>) {
        let shift = |v: &Arc<UOp>, d: i64| if d == 0 { v.clone() } else { add(v, &c32(d)) };
        (shift(&self.row, self.dr), shift(&self.col, self.dc))
    }
}

impl<'a> Emit<'a> {
    fn new(prog: &'a Program, lay: &'a [Option<TileLayout>], low: &'a Lowering, params: Vec<Arc<UOp>>) -> Self {
        let threads = prog.warps * low.device.target.wave;
        let tid = UOp::special_dtype(c32(threads as i64), "lidx0".to_string(), DType::Int32);
        let wave = c32(low.device.target.wave as i64);
        let lane = tid.try_cmod(&wave).expect("lane");
        let warp = tid.try_cdiv(&wave).expect("warp");
        let mut alias = HashMap::new();
        for (_, stmt) in prog.walk() {
            let carried = match stmt {
                Stmt::Loop(l) => &l.carried,
                Stmt::Pipeline(p) => &p.carried,
                _ => continue,
            };
            // `init` starts in place; `next` is moved into `phi` at the end of
            // every iteration (a body may read `phi` after computing `next`).
            for c in carried {
                alias.insert(c.init, c.phi);
            }
        }
        let mut row_dep = vec![None; prog.scalars.len()];
        for id in 0..prog.scalars.len() {
            reads_row(prog, ScalarId(id as u32), &mut row_dep);
        }
        Self {
            prog,
            lay,
            low,
            params,
            vars: HashMap::new(),
            list: vec![],
            headers: vec![0],
            guards: vec![],
            divergent: 0,
            levels: HashMap::new(),
            scalars: HashMap::new(),
            row_scalars: HashMap::new(),
            row_dep: row_dep.into_iter().map(|d| d.expect("every scalar visited")).collect(),
            inductions: HashMap::new(),
            regs: HashMap::new(),
            alias,
            carried: vec![],
            pending: vec![],
            smem: vec![],
            typed: HashMap::new(),
            scratch: HashMap::new(),
            last: tid.clone(),
            lane,
            warp,
            tid,
            blocks: [None, None, None],
            scope: Scope::Block,
            async_products: false,
            range_id: 0,
            reg_slot: 0,
            local_slot: 0,
            tag: 0,
        }
    }

    // ---- the list -------------------------------------------------------

    fn push(&mut self, u: Arc<UOp>) -> Arc<UOp> {
        self.list.push(u.clone());
        self.levels.insert(u.id, self.level());
        self.last = u.clone();
        u
    }

    /// The nesting level a value belongs to: the deepest level among the
    /// listed values it reads (leaves and constants are level 0). Unlisted
    /// pure sources are listed first, at their own level.
    fn level_of(&mut self, u: &Arc<UOp>) -> usize {
        if let Some(&l) = self.levels.get(&u.id) {
            return l;
        }
        match u.op() {
            Op::Const(..) | Op::Param(..) | Op::Buffer(..) | Op::DefineVar(..) | Op::VConst(..) => 0,
            _ => {
                let mut level = 0;
                for src in u.op().sources() {
                    level = level.max(self.level_of(&src));
                }
                level
            }
        }
    }

    /// List a pure value at the header of the level its inputs require, so
    /// every region that can reach those inputs can reach it too. Listed
    /// values stay where they are.
    fn hoist(&mut self, u: Arc<UOp>) -> Arc<UOp> {
        self.hoist_at(u, 0)
    }

    /// [`Self::hoist`] at level `floor` or deeper.
    fn hoist_at(&mut self, u: Arc<UOp>, floor: usize) -> Arc<UOp> {
        if self.levels.contains_key(&u.id) {
            return u;
        }
        let mut level = floor;
        for src in u.op().sources() {
            let pure = !matches!(
                src.op(),
                Op::Const(..) | Op::Param(..) | Op::Buffer(..) | Op::DefineVar(..) | Op::VConst(..)
            ) && !self.levels.contains_key(&src.id);
            if pure {
                self.hoist(src.clone());
            }
            level = level.max(self.level_of(&src));
        }
        let at = self.headers[level];
        self.list.insert(at, u.clone());
        for h in &mut self.headers[level..] {
            *h += 1;
        }
        self.levels.insert(u.id, level);
        u
    }

    fn level(&self) -> usize {
        self.headers.len() - 1
    }

    fn next_tag(&mut self) -> u64 {
        self.tag += 1;
        self.tag
    }

    fn prologue(&mut self) -> Result<()> {
        let n = self.params.len();
        for (i, v) in self.prog.vars.iter().enumerate() {
            let p = UOp::scalar_param(n + i, Some(v.name.clone()), DType::Int32, v.min, v.max);
            let p = self.hoist(p);
            self.vars.insert(v.name.clone(), p);
        }
        self.hoist(self.tid.clone());
        self.hoist(self.lane.clone());
        self.hoist(self.warp.clone());
        for (i, alloc) in self.prog.smem.iter().enumerate() {
            let buf = UOp::buffer(i, alloc.elems, DType::Scalar(alloc.dtype), AddrSpace::Local, None);
            let buf = self.hoist(buf);
            self.smem.push(buf);
        }
        self.local_slot = self.prog.smem.len();
        Ok(())
    }

    // ---- scalars ----------------------------------------------------------

    fn scalar(&mut self, id: ScalarId) -> (usize, Arc<UOp>) {
        self.scalar_at(id, None)
    }

    /// `id` with [`Scalar::Row`] bound to `row`. A row-dependent value is
    /// memoized per row value, so every chunk of a fill gets its own and
    /// whatever only depends on the row and the block is listed once, in the
    /// prologue.
    fn scalar_at(&mut self, id: ScalarId, row: Option<&Arc<UOp>>) -> (usize, Arc<UOp>) {
        let row = row.filter(|_| self.row_dep[id.index()]);
        let found = match row {
            Some(r) => self.row_scalars.get(&(id, r.id)),
            None => self.scalars.get(&id),
        };
        if let Some(found) = found {
            return found.clone();
        }
        let (level, u) = match self.prog.scalar(id) {
            Scalar::Row => {
                let row = row.expect("a row map is evaluated at a row").clone();
                return (self.level_of(&row), row);
            }
            // Constants have no position: an unlisted node lands before its first user.
            Scalar::Const(v) => {
                let u = c32(*v);
                self.scalars.insert(id, (0, u.clone()));
                return (0, u);
            }
            Scalar::Var(name) => (0, self.vars[name].clone()),
            Scalar::Special(Special::Block(axis)) => (0, self.block_index(*axis as usize)),
            Scalar::Special(Special::Warp) => (0, self.warp.clone()),
            Scalar::Induction => return self.inductions[&id].clone(),
            Scalar::Bin(op, a, b) => {
                let ((la, a), (lb, b)) = (self.scalar_at(*a, row), self.scalar_at(*b, row));
                let u = match op {
                    BinOp::Add => a.try_add(&b),
                    BinOp::Sub => a.try_sub(&b),
                    BinOp::Mul => a.try_mul(&b),
                    BinOp::Div => a.try_cdiv(&b),
                    BinOp::Rem => a.try_cmod(&b),
                    BinOp::Min => minimum(&a, &b),
                    BinOp::Max => a.try_max(&b),
                    BinOp::Lt => a.try_cmplt(&b).map(|c| c.cast(DType::Int32)),
                    BinOp::Le => a.try_cmple(&b).map(|c| c.cast(DType::Int32)),
                    BinOp::Eq => a.try_cmpeq(&b).map(|c| c.cast(DType::Int32)),
                    BinOp::And => a.try_and_op(&b),
                    BinOp::Or => a.try_or_op(&b),
                };
                (la.max(lb), u.expect("i32 scalar"))
            }
            // The parameter is an input, so no store moves under its read; a
            // read whose index may leave the parameter stays under the
            // branches it is met in, where the index is in range.
            Scalar::Load { param, index } => {
                let (level, index) = self.scalar_at(*index, row);
                let buf = self.params[param.index()].clone();
                let idx = self.access(&buf, &index, 1);
                let tag = self.next_tag();
                let elems = self.prog.params[param.index()].elems as i64;
                let (lo, hi) = (index.vmin().try_int(), index.vmax().try_int());
                let inside = lo.is_some_and(|lo| lo >= 0) && hi.is_some_and(|hi| hi < elems);
                let level = if inside { level } else { level.max(self.guards.last().copied().unwrap_or(0)) };
                let load = self.hoist_at(load_at(&idx, tag), level);
                (level, load.cast(DType::Int32))
            }
        };
        let u = self.hoist(u);
        match row {
            Some(r) => self.row_scalars.insert((id, r.id), (level, u.clone())),
            None => self.scalars.insert(id, (level, u.clone())),
        };
        (level, u)
    }

    fn sc(&mut self, id: ScalarId) -> Arc<UOp> {
        self.scalar(id).1
    }

    /// The block's index along grid `axis`: the hardware's, or on a target
    /// of several dies the index of the block that the hardware's linear
    /// position lands on under [`die_order`], so the blocks a kernel places
    /// next to each other (sharing a panel, a head's keys) share a die's L2.
    /// Every axis is derived at once and kept, listed in the prologue.
    fn block_index(&mut self, axis: usize) -> Arc<UOp> {
        if let Some(b) = &self.blocks[axis] {
            return b.clone();
        }
        let ends: Vec<Arc<UOp>> = (0..3).map(|a| self.scalar(self.prog.grid[a]).1).collect();
        let hw: Vec<Arc<UOp>> =
            (0..3).map(|a| UOp::special_dtype(ends[a].clone(), format!("gidx{a}"), DType::Int32)).collect();
        let dies = self.low.device.target.xcds as i64;
        let blocks = if dies > 1 {
            let linear = add(&hw[0], &mul(&ends[0], &add(&hw[1], &mul(&ends[1], &hw[2]))));
            let total = mul(&ends[0], &mul(&ends[1], &ends[2]));
            let (base, tall) = (total.try_cdiv(&c32(dies)).expect("base"), total.try_cmod(&c32(dies)).expect("tall"));
            let (die, local) = (linear.try_cmod(&c32(dies)).expect("die"), linear.try_cdiv(&c32(dies)).expect("local"));
            // `min(die, tall)`: the dies before `tall` hold a block more.
            let extra = add(&die, &tall).try_sub(&die.try_max(&tall).expect("max")).expect("min");
            let ordered = self.hoist(add(&add(&mul(&die, &base), &extra), &local));
            let per_plane = mul(&ends[0], &ends[1]);
            [
                ordered.try_cmod(&ends[0]).expect("x"),
                ordered.try_cdiv(&ends[0]).expect("row").try_cmod(&ends[1]).expect("y"),
                ordered.try_cdiv(&per_plane).expect("z"),
            ]
        } else {
            [hw[0].clone(), hw[1].clone(), hw[2].clone()]
        };
        for (a, b) in blocks.into_iter().enumerate() {
            self.blocks[a] = Some(self.hoist(b));
        }
        self.blocks[axis].clone().expect("just taken")
    }

    /// The value of a scalar that is arithmetic over constants.
    fn constant(&self, id: ScalarId) -> Option<i64> {
        match self.prog.scalar(id) {
            Scalar::Const(v) => Some(*v),
            Scalar::Bin(op, a, b) => {
                let (a, b) = (self.constant(*a)?, self.constant(*b)?);
                match op {
                    BinOp::Add => Some(a + b),
                    BinOp::Sub => Some(a - b),
                    BinOp::Mul => Some(a * b),
                    BinOp::Min => Some(a.min(b)),
                    BinOp::Max => Some(a.max(b)),
                    BinOp::Lt => Some(i64::from(a < b)),
                    BinOp::Le => Some(i64::from(a <= b)),
                    BinOp::Eq => Some(i64::from(a == b)),
                    BinOp::Div | BinOp::Rem | BinOp::And | BinOp::Or => None,
                }
            }
            _ => None,
        }
    }

    // ---- memory accesses ------------------------------------------------------

    /// The access node of a `w`-wide element run at `off`, listed where its
    /// offset is, so every copy of a loop body reaches it.
    fn access(&mut self, buf: &Arc<UOp>, off: &Arc<UOp>, w: u32) -> Arc<UOp> {
        self.hoist(access(buf, off, w as usize))
    }

    fn mem_load(&mut self, buf: &Arc<UOp>, off: &Arc<UOp>, w: u32, gate: Option<&Arc<UOp>>) -> Arc<UOp> {
        let tag = self.next_tag();
        let idx = self.access(buf, off, w);
        let u = match gate {
            Some(gate) => {
                let zero = self.hoist_zero(load_zero(&idx));
                load_gated_at(&idx, gate, &zero, tag)
            }
            None => load_at(&idx, tag),
        };
        self.push(u)
    }

    fn mem_store(&mut self, buf: &Arc<UOp>, off: &Arc<UOp>, vals: Vec<Arc<UOp>>) {
        let idx = self.access(buf, off, vals.len() as u32);
        self.push(store_at(&idx, vals));
    }

    // ---- values -------------------------------------------------------------

    fn layout(&self, v: ValId) -> &TileLayout {
        self.lay[v.index()].as_ref().expect("every register value has a layout")
    }

    fn value(&self, v: ValId) -> &Value {
        self.prog.value(v)
    }

    fn root(&self, v: ValId) -> ValId {
        let mut r = v;
        while let Some(&next) = self.alias.get(&r) {
            if next == r {
                break;
            }
            r = next;
        }
        r
    }

    fn reg_buf(&mut self, v: ValId) -> Arc<UOp> {
        let root = self.root(v);
        if let Some(buf) = self.regs.get(&root) {
            return buf.clone();
        }
        let regs = self.layout(root).regs() as usize;
        let dtype = DType::Scalar(self.value(root).dtype);
        let buf = UOp::buffer(self.reg_slot, regs, dtype, AddrSpace::Reg, None);
        self.reg_slot += 1;
        let buf = self.hoist(buf);
        self.regs.insert(root, buf.clone());
        buf
    }

    /// Registers are scalar accesses (the coalescer never widens them).
    fn reg_load(&mut self, v: ValId, j: u32) -> Arc<UOp> {
        let buf = self.reg_buf(v);
        let idx = self.access(&buf, &c32(j as i64), 1);
        let tag = self.next_tag();
        self.push(load_at(&idx, tag))
    }

    fn reg_loads(&mut self, v: ValId, j: u32, w: u32) -> Vec<Arc<UOp>> {
        (0..w).map(|e| self.reg_load(v, j + e)).collect()
    }

    /// Registers `j..j + w` of `v` as one `w`-wide value (one for `w` = 1):
    /// a fragment read whole, so a loop carries it as one vector and the
    /// backend keeps it in the core's register file (gfx942's AGPRs) instead
    /// of moving each element through a vector register every trip.
    fn reg_load_vec(&mut self, v: ValId, j: u32, w: u32) -> Arc<UOp> {
        if w == 1 {
            return self.reg_load(v, j);
        }
        let buf = self.reg_buf(v);
        let idx = self.access(&buf, &c32(j as i64), w);
        let tag = self.next_tag();
        self.push(load_at(&idx, tag))
    }

    /// The `w`-wide `val` stored whole to registers `j..j + w` of `v`.
    fn reg_store_vec(&mut self, v: ValId, j: u32, w: u32, val: Arc<UOp>) {
        if w == 1 {
            return self.reg_store(v, j, vec![val]);
        }
        let buf = self.reg_buf(v);
        let idx = self.access(&buf, &c32(j as i64), w);
        self.push(idx.store(val));
    }

    fn reg_store(&mut self, v: ValId, j: u32, vals: Vec<Arc<UOp>>) {
        let buf = self.reg_buf(v);
        for (e, val) in vals.into_iter().enumerate() {
            let idx = self.access(&buf, &c32(j as i64 + e as i64), 1);
            self.push(store_at(&idx, vec![val]));
        }
    }

    fn view(&mut self, v: ValId) -> View {
        let val = self.value(v).clone();
        let buf = match &val.place {
            Place::Global { param, .. } => self.params[param.index()].clone(),
            Place::Smem { alloc, .. } => self.smem_as(alloc.index(), val.dtype),
            Place::Reg => unreachable!("a memory view"),
        };
        View { buf, place: Some(val.place), shape: val.shape, dtype: val.dtype, region: None }
    }

    /// Allocation `alloc`'s buffer read under `dtype`: its own where they
    /// agree, else the whole of it as a view typed `dtype` (an address runs
    /// over the view's type), made once per pair. A view's offset counts its
    /// own elements, so nothing is scaled.
    fn smem_as(&mut self, alloc: usize, dtype: ScalarDType) -> Arc<UOp> {
        let a = &self.prog.smem[alloc];
        if a.dtype == dtype {
            return self.smem[alloc].clone();
        }
        if let Some(buf) = self.typed.get(&(alloc, dtype)) {
            return buf.clone();
        }
        let elems = (a.elems * a.dtype.bytes() / dtype.bytes()) as i64;
        let shrink = ops::Shrink { src: self.smem[alloc].clone(), offsets: c32(0), sizes: c32(elems) };
        let buf = self.hoist(UOp::new(Op::Shrink(shrink), DType::Scalar(dtype)));
        self.typed.insert((alloc, dtype), buf.clone());
        buf
    }

    /// Element offset of a register's coordinate in a memory view, with the
    /// swizzle a shared view takes, and its bounds gate if any. The lane's
    /// part of the offset is one value listed where its inputs are (the
    /// prologue, for a tile). In a plain global view the view's offset is
    /// added to it and the register's constant last, so a compiler keeps the
    /// first two as a running pointer and the constant as an immediate; a
    /// gathered row and a swizzled shared row need the full coordinate.
    fn address(&mut self, view: &View, c: &Coord) -> (Arc<UOp>, Option<Arc<UOp>>) {
        match &view.place {
            Some(Place::Global { offset, stride, rows: Some(map), .. }) => {
                let (row, col) = c.full();
                let ((_, offset), (_, s1)) = (self.scalar(*offset), self.scalar(stride[1]));
                let (_, start) = self.scalar_at(map.offset, Some(&row));
                // The view's offset moves per step; the row's start and the
                // column are listed where their inputs are.
                let off = add(&offset, &add(&start, &mul(&col, &s1)));
                let gate = self.inside(view, c);
                (self.hoist(off), gate.map(|g| self.hoist(g)))
            }
            Some(Place::Global { offset, stride, .. }) => {
                let ((_, offset), (_, s0), (_, s1)) =
                    (self.scalar(*offset), self.scalar(stride[0]), self.scalar(stride[1]));
                let lane = self.hoist(add(&mul(&c.row, &s0), &mul(&c.col, &s1)));
                let mut off = add(&offset, &lane);
                let delta =
                    [(c.dr, &s0), (c.dc, &s1)].into_iter().filter(|(d, _)| *d != 0).map(|(d, s)| mul(&c32(d), s));
                if let Some(delta) = delta.reduce(|a, b| add(&a, &b)) {
                    off = add(&off, &delta);
                }
                let gate = self.inside(view, c);
                (self.hoist(off), gate.map(|g| self.hoist(g)))
            }
            Some(Place::Smem { .. }) | None => {
                let (mut row, mut col) = c.full();
                let mut cols = view.shape.cols as i64;
                // A column-major tile is the row-major tile of its transpose.
                if self.along(view).0 == Axis::Row {
                    (row, col) = (col, row);
                    cols = view.shape.rows as i64;
                }
                let offset = match &view.place {
                    Some(Place::Smem { offset, .. }) => self.sc(*offset),
                    _ => c32(0),
                };
                // The relayout scratch is addressed through the warp's region.
                let offset = match &view.region {
                    Some(region) => add(&offset, region),
                    None => offset,
                };
                let bytes = view.dtype.bytes();
                // A row past the widest swizzle span is kept as column panels.
                let panel = (self.along(view).0 == Axis::Col)
                    .then(|| self.panel_cols(view.shape.cols, bytes))
                    .flatten()
                    .map(|pc| pc as i64);
                let (panel_off, cols) = match panel {
                    Some(pc) => {
                        let bits = pc.trailing_zeros();
                        let at = mul(&shr(&col, bits), &c32(view.shape.rows as i64 * pc));
                        col = and(&col, &c32(pc - 1));
                        (Some(at), pc)
                    }
                    None => (None, cols),
                };
                let chunk = 16 / bytes as i64;
                let cpr = cols / chunk;
                // The XOR must stay within the row: a power-of-two chunk count.
                let col = if self.low.swizzle && cols % chunk == 0 && cpr >= 2 && (cpr as u64).is_power_of_two() {
                    let bits = chunk.trailing_zeros();
                    let (c, within) = (shr(&col, bits), and(&col, &c32(chunk - 1)));
                    let sw = if cpr >= 8 {
                        and(&row, &c32(7))
                    } else {
                        and(&shr(&row, (8 / cpr).trailing_zeros()), &c32(cpr - 1))
                    };
                    add(&mul(&xor(&c, &sw), &c32(chunk)), &within)
                } else {
                    col
                };
                let lane = add(&mul(&row, &c32(cols)), &col);
                let lane = self.hoist(panel_off.map_or(lane.clone(), |at| add(&at, &lane)));
                (self.hoist(add(&offset, &lane)), None)
            }
            Some(Place::Reg) => unreachable!(),
        }
    }

    /// Whether coordinate `c` of a global view lies within its bounds and,
    /// gathered, inside its row map; `None` where the view has neither.
    fn inside(&mut self, view: &View, c: &Coord) -> Option<Arc<UOp>> {
        let Some(Place::Global { bounds, rows, .. }) = &view.place else { return None };
        let (row, col) = c.full();
        let in_rows = match rows {
            Some(map) => map.valid.map(|v| {
                let (_, v) = self.scalar_at(v, Some(&row));
                v.try_cmpne(&c32(0)).expect("row gate")
            }),
            None => bounds[0].map(|b| row.try_cmplt(&self.sc(b)).expect("bound compare")),
        };
        let in_cols = bounds[1].map(|b| col.try_cmplt(&self.sc(b)).expect("bound compare"));
        in_rows.into_iter().chain(in_cols).reduce(|a, b| a.try_and_op(&b).expect("and"))
    }

    /// The coordinate register `j` of this thread holds in layout `l`:
    /// its lane- and warp-dependent part, induction-free and listed in the
    /// prologue, and the constant the register adds to it.
    fn coord(&mut self, l: &TileLayout, j: u32) -> Coord {
        let regs = l.frag_regs();
        let rep = j / regs;
        let (rr, rc) = (rep / l.reps[1], rep % l.reps[1]);
        let (fr, fc) = l.frag.apply(&[(Reg, j % regs)]);
        let [fr_n, fc_n] = l.frag_shape();
        let [sr, sc] = l.sub_shape();
        // What the bits of `src` move in each dimension, and the bits they reach.
        let bit_terms = |src: &Arc<UOp>, layout: &Layout, dim| {
            let (mut r, mut c): (Option<Arc<UOp>>, Option<Arc<UOp>>) = (None, None);
            let (mut mr, mut mc) = (0, 0);
            let mix = |acc: &mut Option<Arc<UOp>>, bit: &Arc<UOp>, v: u32| {
                if v != 0 {
                    let term = mul(bit, &c32(v as i64));
                    *acc = Some(acc.as_ref().map_or(term.clone(), |a| xor(a, &term)));
                }
            };
            for b in 0..layout.in_bits(dim) {
                let out = layout.basis(dim, b);
                let get = |d| out.iter().find(|o| o.0 == d).map_or(0, |o| o.1);
                let (vr, vc) = (get(Row), get(Col));
                if vr == 0 && vc == 0 {
                    continue;
                }
                let bit = and(&shr(src, b), &c32(1));
                mix(&mut r, &bit, vr);
                mix(&mut c, &bit, vc);
                (mr, mc) = (mr | vr, mc | vc);
            }
            (r.unwrap_or_else(|| c32(0)), c.unwrap_or_else(|| c32(0)), mr, mc)
        };
        let (lr, lc, mr, mc) = bit_terms(&self.lane, &l.frag, Lane);
        let (wr, wc, ..) = bit_terms(&self.warp, &l.warps, Warp);
        // The register's own bits add to the lane's when they share none (the
        // XOR is then a sum), and the constant stays apart; else it is mixed in.
        let (fr, fc) = (fr as i64, fc as i64);
        let ((lr, dr), (lc, dc)) = if fr & mr as i64 == 0 && fc & mc as i64 == 0 {
            ((lr, fr), (lc, fc))
        } else {
            ((xor(&lr, &c32(fr)), 0), (xor(&lc, &c32(fc)), 0))
        };
        let row = add(&mul(&wr, &c32(sr as i64)), &lr);
        let col = add(&mul(&wc, &c32(sc as i64)), &lc);
        Coord { row: self.hoist(row), col: self.hoist(col), dr: dr + (rr * fr_n) as i64, dc: dc + (rc * fc_n) as i64 }
    }

    /// The axis a view stores consecutive elements along, and whether a run
    /// along it may be one vector access: a global view is contiguous along
    /// the axis of its unit stride (neither: every element on its own), a
    /// shared tile along its allocation's axis.
    fn along(&self, view: &View) -> (Axis, bool) {
        match &view.place {
            Some(Place::Global { stride, .. }) => {
                let unit = |s: ScalarId| matches!(self.prog.scalar(s), Scalar::Const(1));
                if unit(stride[1]) { (Axis::Col, true) } else { (Axis::Row, unit(stride[0])) }
            }
            Some(Place::Smem { alloc, .. }) => (self.prog.smem[alloc.index()].along, true),
            None => (Axis::Col, true),
            Some(Place::Reg) => unreachable!(),
        }
    }

    /// The width in elements of the column panels a shared row of `cols`
    /// elements is stored as: under the warpgroup core a row longer than the
    /// widest swizzle span (128 bytes) is kept as panels of that span, each
    /// a row-major region of its own, so a descriptor's canonical layout
    /// reaches every row; `None` keeps the row whole.
    fn panel_cols(&self, cols: usize, bytes: usize) -> Option<usize> {
        let row = cols * bytes;
        (self.low.group && self.low.swizzle && row > wgmma::SPAN && row.is_multiple_of(wgmma::SPAN))
            .then_some(wgmma::SPAN / bytes)
    }

    /// The register runs of `l` a view is accessed by: 16-byte vectors along
    /// the view's axis where it is contiguous, else single elements.
    fn runs(&self, l: &TileLayout, view: &View) -> Vec<(u32, u32)> {
        let (along, contiguous) = self.along(view);
        let width = if contiguous { 16 / view.dtype.bytes() as u32 } else { 1 };
        l.runs(self.low.device.target.wave, width, along)
    }

    // ---- statements -------------------------------------------------------

    fn block(&mut self, block: &Block) -> Result<()> {
        self.pending.push(vec![]);
        let waited = block.0.iter().any(|s| matches!(s, Stmt::Sync(Sync::WaitProducts { .. })));
        let outer = std::mem::replace(&mut self.async_products, waited);
        for stmt in &block.0 {
            self.stmt(stmt)?;
        }
        self.async_products = outer;
        // A carried tile moves a fragment at a time, as the products read
        // it, so the backend carries vectors and never unpacks them.
        for c in self.pending.pop().expect("the block just opened") {
            let (regs, w) = (self.layout(c.phi).regs(), self.layout(c.phi).frag_regs().max(1));
            for j in (0..regs).step_by(w as usize) {
                let x = self.reg_load_vec(c.next, j, w);
                self.reg_store_vec(c.phi, j, w, x);
            }
        }
        Ok(())
    }

    fn stmt(&mut self, stmt: &Stmt) -> Result<()> {
        match stmt {
            Stmt::Let { dst, op } => {
                self.let_(*dst, op)?;
                self.advance(*dst);
                Ok(())
            }
            Stmt::Copy { dst, src, mode, scope } => {
                self.copy(*dst, *src, *mode, *scope)?;
                self.advance(*dst);
                Ok(())
            }
            Stmt::Loop(l) => self.loop_(l),
            Stmt::Pipeline(_) => UnsupportedSnafu { what: "an unexpanded pipeline" }.fail(),
            Stmt::Warps { body } => {
                let outer = std::mem::replace(&mut self.scope, Scope::Warp);
                let out = self.block(body);
                self.scope = outer;
                out
            }
            Stmt::If { pred, then, otherwise } => {
                let divergent = usize::from(self.scope == Scope::Block && self.prog.varies_by_warp(*pred));
                let pred = self.sc(*pred);
                // Listed like any scalar: a compare left to its first user
                // would land inside whichever branch meets it first and not
                // reach a later `if` on the same predicate outside it.
                let cond = self.hoist(pred.try_cmpne(&c32(0)).expect("predicate"));
                let not = self.hoist(cond.not());
                self.divergent += divergent;
                for (branch, cond) in [(then, cond), (otherwise, not)] {
                    if branch.0.is_empty() {
                        continue;
                    }
                    let tag = self.next_tag();
                    let if_ = self.push(UOp::if_(cond, smallvec![]).rtag(Some(smallvec![tag as usize])));
                    self.headers.push(self.list.len());
                    self.guards.push(self.level());
                    self.block(branch)?;
                    let level = self.guards.pop().expect("the branch just opened");
                    self.headers.pop();
                    // What was listed under the branch is out of reach past it.
                    self.scalars.retain(|_, (l, _)| *l < level);
                    self.row_scalars.retain(|_, (l, _)| *l < level);
                    self.push(UOp::endif(if_));
                }
                self.divergent -= divergent;
                Ok(())
            }
            Stmt::Sync(sync) => self.sync(*sync),
            Stmt::Raw(_) => UnsupportedSnafu { what: "raw statements" }.fail(),
        }
    }

    /// A rolled loop, or `unroll` copies of the body per iteration followed by
    /// a rolled remainder (skipped when a constant extent divides evenly).
    fn loop_(&mut self, l: &Loop) -> Result<()> {
        let (start, extent) = (self.sc(l.start), self.sc(l.extent));
        let u = l.unroll.max(1) as i64;
        let constant = self.constant(l.extent);
        if u == 1 {
            return self.loop_rolled(l, extent, |_, range| add(&start, &range));
        }
        let main = self.hoist(extent.try_cdiv(&c32(u)).expect("trips"));
        let rem = self.hoist(extent.try_cmod(&c32(u)).expect("remainder"));
        if constant.is_none_or(|e| e / u > 0) {
            self.loop_unrolled(l, main.clone(), u)?;
        }
        if constant.is_none_or(|e| e % u != 0) {
            let base = self.hoist(add(&start, &mul(&main, &c32(u))));
            self.loop_rolled(l, rem, move |_, range| add(&base, &range))?;
        }
        Ok(())
    }

    fn open_range(&mut self, end: Arc<UOp>) -> Arc<UOp> {
        let range = UOp::range_axis_dtype(end, AxisId::Renumbered(self.range_id), AxisType::Loop, DType::Int32);
        self.range_id += 1;
        let range = self.push(range);
        self.headers.push(self.list.len());
        // The induction variable belongs to the level it opens.
        self.levels.insert(range.id, self.level());
        range
    }

    fn close_range(&mut self, range: Arc<UOp>) {
        self.headers.pop();
        let end = self.last.clone().end(smallvec![range]);
        self.push(end);
    }

    /// Bind the induction variable for one copy of a body: memoized scalars
    /// of this level or deeper were computed for the previous binding.
    fn bind(&mut self, iv: ScalarId, level: usize, value: Arc<UOp>) {
        let value = self.hoist(value);
        self.levels.insert(value.id, level);
        self.inductions.insert(iv, (level, value));
        self.scalars.retain(|_, (l, _)| *l < level);
        self.row_scalars.retain(|_, (l, _)| *l < level);
    }

    /// One iteration's body; a carried value moves into its `phi` right where
    /// it is defined (see [`Self::advance`]).
    fn body(&mut self, l: &Loop) -> Result<()> {
        self.carried.push(l.carried.clone());
        self.block(&l.body)?;
        self.carried.pop();
        Ok(())
    }

    /// After the statement defining `v`: if `v` is the `next` of an open loop,
    /// schedule its move into that loop's `phi` for the end of the block that
    /// defines it (so the body may still read `phi`, and a guarded body moves
    /// only on the iterations that run the guard).
    fn advance(&mut self, v: ValId) {
        let moves: Vec<Carried> = self
            .carried
            .iter()
            .flatten()
            .filter(|c| c.next == v && self.root(c.next) != self.root(c.phi))
            .copied()
            .collect();
        self.pending.last_mut().expect("an open block").extend(moves);
    }

    fn loop_rolled(&mut self, l: &Loop, end: Arc<UOp>, iv: impl Fn(&mut Self, Arc<UOp>) -> Arc<UOp>) -> Result<()> {
        let range = self.open_range(end);
        let level = self.level();
        let value = iv(self, range.clone());
        self.bind(l.iv, level, value);
        self.body(l)?;
        self.close_range(range);
        Ok(())
    }

    fn loop_unrolled(&mut self, l: &Loop, trips: Arc<UOp>, u: i64) -> Result<()> {
        let start = self.sc(l.start);
        let range = self.open_range(trips);
        let level = self.level();
        let base = add(&start, &mul(&range, &c32(u)));
        for copy in 0..u {
            self.bind(l.iv, level, add(&base, &c32(copy)));
            self.body(l)?;
        }
        self.close_range(range);
        Ok(())
    }

    fn sync(&mut self, sync: Sync) -> Result<()> {
        let cuda = matches!(self.low.device.target.arch, GpuArch::Cuda(_));
        let u = match sync {
            Sync::Barrier { scope: Scope::Block } => {
                let tag = self.next_tag();
                self.last.clone().barrier(smallvec![]).rtag(Some(smallvec![tag as usize]))
            }
            // A warp's own barrier: CUDA's lanes schedule independently and
            // its shared accesses need `bar.warp.sync`; an AMD wave issues
            // its LDS accesses in order and waits on their counters.
            Sync::Barrier { scope: Scope::Warp } if cuda => UOp::custom(
                smallvec![self.last.clone()],
                "declare void @llvm.nvvm.bar.warp.sync(i32)\ncall void @llvm.nvvm.bar.warp.sync(i32 -1)".to_string(),
                DType::Void,
            ),
            Sync::Barrier { scope: Scope::Warp } => return Ok(()),
            Sync::CommitAsync if cuda && self.low.device.target.cp_async => cp_async_commit(smallvec![]),
            Sync::WaitAsync { pending } if cuda && self.low.device.target.cp_async => {
                cp_async_wait(pending, smallvec![])
            }
            // Without `cp.async` the copies were synchronous: nothing to wait for.
            Sync::CommitAsync | Sync::WaitAsync { .. } => return Ok(()),
            Sync::Fence if self.low.device.target.commit_fence => UOp::custom(
                smallvec![self.last.clone()],
                "declare void @llvm.amdgcn.sched.barrier(i32)\ncall void @llvm.amdgcn.sched.barrier(i32 0)".to_string(),
                DType::Void,
            ),
            Sync::Fence => return Ok(()),
            Sync::WaitProducts { pending } if cuda && self.low.group => wgmma::wait(pending),
            Sync::WaitProducts { .. } => return Ok(()),
            // A side effect in list order, with no value to depend on: it may
            // follow a barrier.
            Sync::Prio { level } if matches!(self.low.device.target.arch, GpuArch::Amd(_)) => UOp::custom(
                smallvec![],
                format!("declare void @llvm.amdgcn.s.setprio(i16)\ncall void @llvm.amdgcn.s.setprio(i16 {level})"),
                DType::Void,
            ),
            Sync::Prio { .. } => return Ok(()),
        };
        self.push(u);
        Ok(())
    }

    // ---- copies ---------------------------------------------------------------

    fn copy(&mut self, dst: ValId, src: ValId, mode: CopyMode, scope: Scope) -> Result<()> {
        match (self.value(dst).tier(), self.value(src).tier()) {
            (Tier::Smem, Tier::Global) => self.fill(dst, src, mode, scope),
            (Tier::Reg, Tier::Smem) => self.gather(dst, src),
            (Tier::Reg, Tier::Global) => self.load_regs(dst, src, mode),
            (Tier::Smem | Tier::Global, Tier::Reg) => self.store_regs(dst, src),
            (Tier::Reg, Tier::Reg) => self.relayout(dst, src),
            (d, s) => UnsupportedSnafu { what: format!("a {s:?} → {d:?} copy") }.fail(),
        }
    }

    /// The warps a register tile is spread over: the lowering's grid, which
    /// key warps sharing a block's tiles are not part of.
    fn grid_warps(&self) -> u32 {
        self.low.grid.count()
    }

    /// Global → shared, every thread of the copy's scope (the block's, or
    /// one warp's own) moving 16-byte chunks.
    fn fill(&mut self, dst: ValId, src: ValId, mode: CopyMode, scope: Scope) -> Result<()> {
        let (d, s) = (self.view(dst), self.view(src));
        let chunk = 16 / d.dtype.bytes() as i64;
        let cols = d.shape.cols as i64;
        let (warps, tid) = match scope {
            Scope::Block => (self.prog.warps, self.tid.clone()),
            Scope::Warp => (1, self.lane.clone()),
        };
        let threads = (warps * self.low.device.target.wave) as i64;
        let chunks = d.shape.rows as i64 * cols / chunk;
        let cuda = matches!(self.low.device.target.arch, GpuArch::Cuda(_));
        snafu::ensure!(
            cols % chunk == 0 && chunks % threads == 0,
            UnsupportedSnafu { what: format!("a {}×{} fill by {threads} threads", d.shape.rows, cols) }
        );
        snafu::ensure!(self.along(&d).0 == Axis::Col, UnsupportedSnafu { what: "a chunk fill of a column-major tile" });
        let Some(Place::Global { stride, bounds, rows, .. }) = &s.place else { unreachable!() };
        let gathered = rows.is_some();
        snafu::ensure!(
            matches!(self.prog.scalar(stride[1]), Scalar::Const(1)),
            UnsupportedSnafu { what: "a fill from a column-strided view" }
        );
        let bounds = *bounds;
        let cpr = cols / chunk;
        for t in 0..chunks / threads {
            let c = add(&tid, &c32(t * threads));
            let (row, cc) = (c.try_cdiv(&c32(cpr)).expect("row"), c.try_cmod(&c32(cpr)).expect("chunk"));
            let (row, col) = (self.hoist(row), self.hoist(mul(&cc, &c32(chunk))));
            // Rows past a bound re-read the last valid one (the first, under an
            // empty bound); the consumer masks.
            let src_row = match bounds[0] {
                Some(b) => {
                    let bound = self.sc(b);
                    let last = bound.try_sub(&c32(1)).expect("bound").try_max(&c32(0)).expect("floor");
                    let over = row.try_cmplt(&last).expect("cmp");
                    self.hoist(UOp::try_where(over, row.clone(), last).expect("clamp"))
                }
                None => row.clone(),
            };
            let (src_off, gate) = self.address(&s, &Coord::plain(src_row, col.clone()));
            // A plain view's fill clamps its rows (above) and, under a column
            // bound, lands a chunk starting past it as zeros, as a gathered
            // row outside its map does, from element 0 of the operand.
            let gate = gate.filter(|_| gathered || bounds[1].is_some());
            let src_off = match &gate {
                Some(g) => self.hoist(UOp::try_where(g.clone(), src_off, c32(0)).expect("safe offset")),
                None => src_off,
            };
            let (dst_off, _) = self.address(&d, &Coord::plain(row, col));
            if cuda && mode == CopyMode::Async && self.low.device.target.cp_async {
                let (dst, src) = (self.access(&d.buf, &dst_off, 1), self.access(&s.buf, &src_off, 1));
                match gate {
                    Some(g) => {
                        let bytes = self.hoist(UOp::try_where(g, c32(16), c32(0)).expect("source size"));
                        self.push(cp_async_16_zfill(&dst, &src, &bytes))
                    }
                    None => self.push(cp_async_16(&dst, &src)),
                };
            } else {
                let v = self.mem_load(&s.buf, &src_off, chunk as u32, None);
                let v = match gate {
                    Some(g) => self.zeroed(&g, v),
                    None => v,
                };
                let vals = (0..chunk as usize).map(|e| elem(&v, e, chunk as usize)).collect();
                self.mem_store(&d.buf, &dst_off, vals);
            }
        }
        Ok(())
    }

    /// Shared → registers under the destination's layout: `ldmatrix.x4` per
    /// 16×16 block when the layout is a register permutation of what it
    /// produces, else vector loads per register run.
    fn gather(&mut self, dst: ValId, src: ValId) -> Result<()> {
        let l = self.layout(dst).clone();
        let s = self.view(src);
        let [sub_r, sub_c] = l.sub_shape();
        // `ldmatrix` reads 16-byte memory rows: a row-major tile only.
        let rows = self.along(&s).0 == Axis::Col;
        if self.low.device.target.ldmatrix && rows && s.dtype.bytes() == 2 && sub_r % 16 == 0 && sub_c % 16 == 0 {
            for trans in [false, true] {
                let produced = TileLayout {
                    frag: frag::ldmatrix_x4(trans),
                    reps: [sub_r / 16, sub_c / 16],
                    warps: l.warps.clone(),
                };
                if let Some(perm) = l.as_permutation_of(&produced, self.grid_warps(), self.low.device.target.wave) {
                    return self.ldmatrix_gather(dst, &s, &l, trans, &perm);
                }
            }
        }
        for (j, w) in self.runs(&l, &s) {
            let c = self.coord(&l, j);
            let (off, _) = self.address(&s, &c);
            let v = self.mem_load(&s.buf, &off, w, None);
            let vals = (0..w as usize).map(|e| elem(&v, e, w as usize)).collect();
            self.reg_store(dst, j, vals);
        }
        Ok(())
    }

    fn ldmatrix_gather(&mut self, dst: ValId, s: &View, l: &TileLayout, trans: bool, perm: &[u32]) -> Result<()> {
        let [sub_r, sub_c] = l.sub_shape();
        let (blocks_r, blocks_c) = (sub_r / 16, sub_c / 16);
        let pair = DType::Scalar(s.dtype).vec(2).expect("a pair");
        let lane16 = self.hoist(and(&self.lane, &c32(15)));
        let lane_hi = self.hoist(mul(&shr(&self.lane, 4), &c32(8)));
        // Where this warp's sub-tile starts.
        let (wr, wc) =
            self.coord(&TileLayout { frag: Layout::zeros(Lane, 1), reps: [1, 1], warps: l.warps.clone() }, 0).full();
        let (wr, wc) = (self.hoist(mul(&wr, &c32(sub_r as i64))), self.hoist(mul(&wc, &c32(sub_c as i64))));
        for br in 0..blocks_r {
            for bc in 0..blocks_c {
                let row = self.hoist(add(&add(&wr, &c32((br * 16) as i64)), &lane16));
                let col = self.hoist(add(&add(&wc, &c32((bc * 16) as i64)), &lane_hi));
                let (off, _) = self.address(s, &Coord::plain(row, col));
                let idx = self.access(&s.buf, &off, 1);
                let words = ldmatrix(&idx, 4, trans, pair.clone());
                let block = br * blocks_c + bc;
                for (jd, &js) in perm.iter().enumerate() {
                    let jd = jd as u32;
                    if js / 8 != block {
                        continue;
                    }
                    let e = js % 8;
                    // Register pair `p` holds matrix `p` of the TL, BL, TR, BR address
                    // order, with the middle two swapped under `.trans`.
                    let word = if trans { [0, 2, 1, 3][(e / 2) as usize] } else { (e / 2) as usize };
                    let v = elem(&words[word], (e % 2) as usize, 2);
                    self.reg_store(dst, jd, vec![v]);
                }
            }
        }
        Ok(())
    }

    /// Global → registers: past a plain bound the load is gated; a gathered
    /// row outside its map reads element 0 and one select zeroes the run.
    /// Global → registers. The register half of a fill ([`CopyMode::Staged`])
    /// reads as the kernel's fills read: where those are `cp.async.cg`
    /// copies, its 16-byte runs load at L2 only, bypassing L1 as the copies
    /// do; under register staging, as plain loads.
    fn load_regs(&mut self, dst: ValId, src: ValId, mode: CopyMode) -> Result<()> {
        let l = self.layout(dst).clone();
        let s = self.view(src);
        if let Some(perm) = self.transposing(&l, &s) {
            return self.tr_load_regs(dst, &s, &l, &perm);
        }
        let gathered = matches!(s.place, Some(Place::Global { rows: Some(_), .. }));
        let cuda = matches!(self.low.device.target.arch, GpuArch::Cuda(_));
        let l2_only = mode == CopyMode::Staged && cuda && self.low.schedule.prefetch() == Prefetch::CpAsync;
        for (j, w) in self.runs(&l, &s) {
            let c = self.coord(&l, j);
            let (off, gate) = self.address(&s, &c);
            if l2_only && w as usize * s.dtype.bytes() == 16 {
                let vals = self.load_l2_only(&s, off, gate, w as usize);
                self.reg_store(dst, j, vals);
                continue;
            }
            let v = match gate {
                Some(g) if gathered => {
                    let safe = self.hoist(UOp::try_where(g.clone(), off, c32(0)).expect("safe offset"));
                    let v = self.mem_load(&s.buf, &safe, w, None);
                    self.zeroed(&g, v)
                }
                gate => self.mem_load(&s.buf, &off, w, gate.as_ref()),
            };
            let vals = (0..w as usize).map(|e| elem(&v, e, w as usize)).collect();
            self.reg_store(dst, j, vals);
        }
        Ok(())
    }

    /// A 16-byte run of `w` elements of `s` at `off` loaded at L2 only (as a
    /// `cp.async.cg` fill reads), a run past `gate` reading element 0 and
    /// landing as zeros.
    fn load_l2_only(&mut self, s: &View, off: Arc<UOp>, gate: Option<Arc<UOp>>, w: usize) -> Vec<Arc<UOp>> {
        let safe = match &gate {
            Some(g) => self.hoist(UOp::try_where(g.clone(), off, c32(0)).expect("safe offset")),
            None => off,
        };
        let src = self.access(&s.buf, &safe, 1);
        let words = ld_global_cg_16(&src);
        let per_word = w / words.len();
        let vals: Vec<Arc<UOp>> = (0..w)
            .map(|e| {
                let word = &words[e / per_word];
                match per_word {
                    1 if word.dtype() == s.dtype.into() => word.clone(),
                    1 => word.bitcast(s.dtype.into()),
                    n => elem(&word.bitcast(s.dtype.vec(n)), e % n, n),
                }
            })
            .collect();
        match gate {
            Some(g) => vals
                .into_iter()
                .map(|v| UOp::try_where(g.clone(), v.clone(), v.const_like(0)).expect("zeroed"))
                .collect(),
            None => vals,
        }
    }

    /// The register permutation under which `l` is what transposing 16×16
    /// loads leave a row-major 16-bit `view` in, where the target has them.
    fn transposing(&self, l: &TileLayout, view: &View) -> Option<Vec<u32>> {
        let [sub_r, sub_c] = l.sub_shape();
        let global = matches!(view.place, Some(Place::Global { .. })) && self.along(view) == (Axis::Col, true);
        if !self.low.device.target.tr_load || !global || view.dtype.bytes() != 2 || sub_r % 16 != 0 || sub_c % 16 != 0 {
            return None;
        }
        let produced =
            TileLayout { frag: frag::global_tr_b128(), reps: [sub_r / 16, sub_c / 16], warps: l.warps.clone() };
        l.as_permutation_of(&produced, self.grid_warps(), self.low.device.target.wave)
    }

    /// Global → registers by transposing loads, one per 16×16 block, its
    /// lanes addressing the block's rows in [`frag::global_tr_rows`] order and
    /// receiving column runs. A lane whose row is past the view's bounds, or
    /// a gathered row outside its map, addresses the operand's first
    /// elements instead; every element received from outside reads zero,
    /// judged at its own coordinate, since a lane receives other lanes' rows.
    fn tr_load_regs(&mut self, dst: ValId, s: &View, l: &TileLayout, perm: &[u32]) -> Result<()> {
        let [sub_r, sub_c] = l.sub_shape();
        let rows = TileLayout { frag: frag::global_tr_rows(), reps: [sub_r / 16, sub_c / 16], warps: l.warps.clone() };
        for block in 0..rows.regs() {
            let c = self.coord(&rows, block);
            let (off, gate) = self.address(s, &c);
            let bounded = gate.is_some();
            let off = match gate {
                Some(g) => self.hoist(UOp::try_where(g, off, c32(0)).expect("safe offset")),
                None => off,
            };
            let idx = self.access(&s.buf, &off, 1);
            let elems = global_load_tr_b128(&idx, s.dtype);
            for (jd, &js) in perm.iter().enumerate() {
                if js / 8 != block {
                    continue;
                }
                let mut e = elems[(js % 8) as usize].clone();
                if bounded {
                    let held = self.coord(l, jd as u32);
                    let inside = self.inside(s, &held).expect("a bounded view");
                    e = UOp::try_where(self.hoist(inside), e.clone(), e.const_like(0)).expect("zeroed");
                }
                self.reg_store(dst, jd as u32, vec![e]);
            }
        }
        Ok(())
    }

    fn store_regs(&mut self, dst: ValId, src: ValId) -> Result<()> {
        let l = self.layout(src).clone();
        let d = self.view(dst);
        for (j, w) in self.runs(&l, &d) {
            let c = self.coord(&l, j);
            let (off, gate) = self.address(&d, &c);
            let vals = self.reg_loads(src, j, w);
            match gate {
                Some(gate) => {
                    let tag = self.next_tag();
                    let if_ = self.push(UOp::if_(gate, smallvec![]).rtag(Some(smallvec![tag as usize])));
                    self.mem_store(&d.buf, &off, vals);
                    self.push(UOp::endif(if_));
                }
                None => self.mem_store(&d.buf, &off, vals),
            }
        }
        Ok(())
    }

    fn relayout(&mut self, dst: ValId, src: ValId) -> Result<()> {
        let (ls, ld) = (self.layout(src).clone(), self.layout(dst).clone());
        match ls.relayout(&ld, self.grid_warps(), self.low.device.target.wave, &self.low.device.costs) {
            Relayout::Identity => {
                let buf = self.reg_buf(src);
                let root = self.root(dst);
                self.regs.insert(root, buf);
                Ok(())
            }
            Relayout::RegPermute(perm) => {
                for (jd, js) in perm.into_iter().enumerate() {
                    let v = self.reg_load(src, js);
                    self.reg_store(dst, jd as u32, vec![v]);
                }
                Ok(())
            }
            Relayout::Exchange(picks) => {
                // A (mask, register) move is made once and shared by every
                // destination register that selects it.
                let mut moved: HashMap<(u32, u32), Arc<UOp>> = HashMap::new();
                for (jd, picks) in picks.into_iter().enumerate() {
                    let mut out: Option<Arc<UOp>> = None;
                    for p in picks.into_iter().rev() {
                        let v = moved
                            .entry((p.mask, p.reg))
                            .or_insert_with(|| {
                                let x = self.reg_load(src, p.reg);
                                if p.mask == 0 { x } else { self.shuffle_xor(&x, p.mask) }
                            })
                            .clone();
                        out = Some(match out {
                            None => v,
                            Some(rest) => {
                                let mine = self.hoist(self.lane_in(p.lanes));
                                UOp::try_where(mine, v, rest).expect("exchange select")
                            }
                        });
                    }
                    self.reg_store(dst, jd as u32, vec![out.expect("a move per register")]);
                }
                Ok(())
            }
            Relayout::ViaSmem => self.relayout_via_smem(dst, src),
        }
    }

    /// Whether this lane is one of `lanes`: a 32-bit word on a wave of 32,
    /// a 64-bit one on a wave64.
    fn lane_in(&self, lanes: u64) -> Arc<UOp> {
        let (dtype, word, lane) = match self.low.device.target.wave {
            ..=32 => (DType::Int32, lanes as u32 as i32 as i64, self.lane.clone()),
            _ => (DType::Int64, lanes as i64, self.lane.cast(DType::Int64)),
        };
        let c = |v: i64| UOp::const_(dtype.clone(), ConstValue::Int(v));
        let bit = and(&c(word).try_shr_op(&lane).expect("lane bit"), &c(1));
        bit.try_cmpne(&c(0)).expect("lane predicate")
    }

    /// A tile re-held through a scratch of shared memory; inside a
    /// [`Stmt::Warps`] every warp takes its own region of it.
    fn relayout_via_smem(&mut self, dst: ValId, src: ValId) -> Result<()> {
        let Value { dtype, shape, .. } = self.value(src).clone();
        let elems = shape.elems();
        let regions = if self.scope == Scope::Warp { self.prog.warps as usize } else { 1 };
        let scratch = match self.scratch.get(&dtype) {
            Some((buf, size)) if *size >= elems * regions => buf.clone(),
            _ => {
                let buf = UOp::buffer(self.local_slot, elems * regions, DType::Scalar(dtype), AddrSpace::Local, None);
                self.local_slot += 1;
                let buf = self.hoist(buf);
                self.scratch.insert(dtype, (buf.clone(), elems * regions));
                buf
            }
        };
        // A warp's region starts at its index's multiple of the tile: an
        // offset of every access, never a crop of the scratch, since an
        // access of a crop is folded into the buffer at the summed offset
        // once the program's placeholders are bound, and the sum is listed
        // nowhere.
        let region = (regions > 1).then(|| self.hoist(mul(&self.warp, &c32(elems as i64))));
        let view = View { buf: scratch, place: None, shape, dtype, region };
        snafu::ensure!(
            self.scope == Scope::Warp || self.divergent == 0,
            DivergentBarrierSnafu { what: "a relayout through shared memory" }
        );
        self.sync(Sync::Barrier { scope: self.scope })?;
        self.store_view(&view, src)?;
        self.sync(Sync::Barrier { scope: self.scope })?;
        self.gather_view(dst, &view)
    }

    fn store_view(&mut self, d: &View, src: ValId) -> Result<()> {
        let l = self.layout(src).clone();
        for (j, w) in self.runs(&l, d) {
            let c = self.coord(&l, j);
            let (off, _) = self.address(d, &c);
            let vals = self.reg_loads(src, j, w);
            self.mem_store(&d.buf, &off, vals);
        }
        Ok(())
    }

    fn gather_view(&mut self, dst: ValId, s: &View) -> Result<()> {
        let l = self.layout(dst).clone();
        for (j, w) in self.runs(&l, s) {
            let c = self.coord(&l, j);
            let (off, _) = self.address(s, &c);
            let v = self.mem_load(&s.buf, &off, w, None);
            let vals = (0..w as usize).map(|e| elem(&v, e, w as usize)).collect();
            self.reg_store(dst, j, vals);
        }
        Ok(())
    }

    // ---- tile ops -----------------------------------------------------------------

    fn let_(&mut self, dst: ValId, op: &TileOp) -> Result<()> {
        let dtype = DType::Scalar(self.value(dst).dtype);
        let regs = self.layout(dst).regs();
        match *op {
            TileOp::Fill(c) => {
                let v = match c {
                    Const::Int(i) => UOp::const_(dtype, ConstValue::Int(i)),
                    Const::Float(f) => UOp::const_(dtype, ConstValue::Float(f)),
                };
                for j in 0..regs {
                    self.reg_store(dst, j, vec![v.clone()]);
                }
            }
            TileOp::Splat(id) => {
                let v = self.sc(id).cast(dtype);
                for j in 0..regs {
                    self.reg_store(dst, j, vec![v.clone()]);
                }
            }
            TileOp::Coord(axis) => {
                let l = self.layout(dst).clone();
                for j in 0..regs {
                    let (row, col) = self.coord(&l, j).full();
                    let v = self.hoist(if axis == Axis::Row { row } else { col });
                    self.reg_store(dst, j, vec![v]);
                }
            }
            TileOp::Unary { src, f } => {
                for j in 0..regs {
                    let x = self.reg_load(src, j);
                    let y = match f {
                        UnaryOp::Neg => x.neg(),
                        UnaryOp::Exp2 => self.exp2(&x),
                        UnaryOp::Log2 => x.try_log2().expect("log2"),
                        UnaryOp::Recip => UOp::try_reciprocal(&x).expect("recip"),
                        UnaryOp::Sqrt => x.try_sqrt().expect("sqrt"),
                        UnaryOp::Rsqrt => x.try_rsqrt().expect("rsqrt"),
                        UnaryOp::Abs => x.abs(),
                        UnaryOp::Not => x.not(),
                    };
                    self.reg_store(dst, j, vec![y]);
                }
            }
            TileOp::Binary { a, b, f: BinaryOp::Scale } => self.scale(dst, a, b),
            TileOp::Binary { a, b, f } => {
                let (ja, jb) = (self.operand_regs(dst, a), self.operand_regs(dst, b));
                for j in 0..regs {
                    let (x, y) = (self.reg_load(a, ja[j as usize]), self.reg_load(b, jb[j as usize]));
                    let z = match f {
                        BinaryOp::Add => x.try_add(&y),
                        BinaryOp::Sub => x.try_sub(&y),
                        BinaryOp::Mul | BinaryOp::Scale => x.try_mul(&y),
                        BinaryOp::Div if x.dtype().is_float() => x.try_div(&y),
                        BinaryOp::Div => x.try_cdiv(&y),
                        BinaryOp::Max => x.try_max(&y),
                        BinaryOp::Min => minimum(&x, &y),
                        BinaryOp::Lt => x.try_cmplt(&y),
                        BinaryOp::Le => x.try_cmple(&y),
                        BinaryOp::Eq => x.try_cmpeq(&y),
                        BinaryOp::Ne => x.try_cmpne(&y),
                        BinaryOp::And => x.try_and_op(&y),
                        BinaryOp::Or => x.try_or_op(&y),
                    }
                    .expect("elementwise");
                    self.reg_store(dst, j, vec![z]);
                }
            }
            TileOp::Cast { src, to, round } => {
                for j in 0..regs {
                    let x = self.reg_load(src, j);
                    let y = match round {
                        Round::Nearest => x.cast(DType::Scalar(to)),
                        // The upper half of the f32's bits.
                        Round::Zero => shr(&x.bitcast(DType::Int32), 16).cast(DType::Int16).bitcast(DType::Scalar(to)),
                    };
                    self.reg_store(dst, j, vec![y]);
                }
            }
            TileOp::Where { pred, a, b } => {
                let (jp, ja, jb) = (self.operand_regs(dst, pred), self.operand_regs(dst, a), self.operand_regs(dst, b));
                for j in 0..regs {
                    let p = self.reg_load(pred, jp[j as usize]);
                    let (x, y) = (self.reg_load(a, ja[j as usize]), self.reg_load(b, jb[j as usize]));
                    self.reg_store(dst, j, vec![UOp::try_where(p, x, y).expect("where")]);
                }
            }
            TileOp::MulAdd { a, b, c } => {
                let (ja, jb, jc) = (self.operand_regs(dst, a), self.operand_regs(dst, b), self.operand_regs(dst, c));
                for j in 0..regs {
                    let (x, y) = (self.reg_load(a, ja[j as usize]), self.reg_load(b, jb[j as usize]));
                    let z = self.reg_load(c, jc[j as usize]);
                    self.reg_store(dst, j, vec![UOp::try_mulacc(x, y, z).expect("mul-add")]);
                }
            }
            TileOp::Mma { .. } if self.low.group => self.group_mma(dst, op)?,
            TileOp::Mma { .. } => self.mma(dst, op)?,
            TileOp::Reduce { src, axis, f } => self.reduce(dst, src, axis, f)?,
            TileOp::Transpose { .. } => return UnsupportedSnafu { what: "register transposes" }.fail(),
            TileOp::Relayout { src } => self.relayout(dst, src)?,
            TileOp::Move { src } => {
                for j in 0..regs {
                    let x = self.reg_load(src, j);
                    self.reg_store(dst, j, vec![x]);
                }
            }
        }
        Ok(())
    }

    /// For every register of `dst`, the register of `operand` holding the
    /// same element (or, for a broadcast vector, the element of its row/column).
    fn operand_regs(&self, dst: ValId, operand: ValId) -> Vec<u32> {
        let (ld, lo) = (self.layout(dst), self.layout(operand));
        let os = self.value(operand).shape;
        let ds = self.value(dst).shape;
        if os == ds {
            assert_eq!(ld, lo, "the inference holds elementwise operands in the result's layout");
            return (0..ld.regs()).collect();
        }
        let key = |r: u32, c: u32| if os.cols == 1 { (r, 0) } else { (0, c) };
        let lanes = self.low.device.target.wave;
        (0..ld.regs())
            .map(|j| {
                let (r, c) = ld.coord(0, 0, j);
                let want = key(r, c);
                let found = (0..lo.regs()).find(|&jo| key(lo.coord(0, 0, jo).0, lo.coord(0, 0, jo).1) == want);
                let jo = found.expect("the vector holds every row/column of the tile");
                debug_assert!((1..lanes).all(|lane| {
                    let (r, c) = ld.coord(0, lane, j);
                    key(lo.coord(0, lane, jo).0, lo.coord(0, lane, jo).1) == key(r, c)
                }));
                jo
            })
            .collect()
    }

    fn mma(&mut self, dst: ValId, op: &TileOp) -> Result<()> {
        let TileOp::Mma { acc, a, b, a_t, b_t, orient } = *op else { unreachable!("a product") };
        let orient = orient.expect("a laid program orients every product");
        let (va, vd) = (self.value(a).clone(), self.value(dst).clone());
        let atom = self.low.device.target.mma(va.dtype, vd.dtype).expect("inference found the core").clone();
        let k = if a_t { va.shape.rows } else { va.shape.cols };
        // The product as laid, from the one function the inference used: the
        // slot each operand feeds and the registers each fragment holds.
        let issue =
            atom.issue(orient, self.low.grid, vd.shape.rows, vd.shape.cols, k).expect("inference laid this product");
        let [rm, rn] = issue.c.reps;
        let rk = (k / atom.k as usize) as u32;
        let (ra, rb, rc) = (issue.a.frag_regs(), issue.b.frag_regs(), issue.c.frag_regs());
        let swapped = issue.swapped;
        // Fragment block index of each operand for a (m, n, k) step.
        let block_a = |i: u32, kk: u32| if a_t { kk * rm + i } else { i * rk + kk };
        let block_b = |j: u32, kk: u32| if b_t { j * rk + kk } else { kk * rn + j };
        for i in 0..rm {
            for j in 0..rn {
                let cblock = i * rn + j;
                let mut c = self.reg_load_vec(acc, cblock * rc, rc);
                for kk in 0..rk {
                    let xa = UOp::stack(self.reg_loads(a, block_a(i, kk) * ra, ra).into_iter().collect());
                    let xb = UOp::stack(self.reg_loads(b, block_b(j, kk) * rb, rb).into_iter().collect());
                    let (x, y) = if swapped { (xb, xa) } else { (xa, xb) };
                    c = UOp::wmma(x, y, c, atom.meta.clone());
                }
                self.reg_store_vec(dst, cblock * rc, rc, c);
            }
        }
        Ok(())
    }

    /// A product on the warpgroup atom: per instruction column block, one
    /// `Wmma` over the whole K chain, its shared operands as the descriptors
    /// of this warp's group rows (and the block's columns of `B`) at every
    /// `k16` step, `A` as its register fragments when it is a register tile.
    /// `B` is K-major (`b_t`: its rows run along N) or MN-major (its rows run
    /// along K, the instruction transposing them). The descriptor's swizzle
    /// is the lowering's own (the hardware applies it to the absolute
    /// address, so the ring's slots start on 1024-byte boundaries) over the
    /// operand's rows, or over its column panels where a row is longer than
    /// the widest span ([`Self::panel_cols`]). The chain is fenced and
    /// committed inside the one statement (`codegen::llvm::nvptx::wgmma`) and
    /// waited for there too, unless the block holds the
    /// [`Sync::WaitProducts`] that waits for it later.
    fn group_mma(&mut self, dst: ValId, op: &TileOp) -> Result<()> {
        let TileOp::Mma { acc, a, b, a_t, b_t, .. } = *op else { unreachable!("a product") };
        let (va, vb, vd) = (self.value(a).clone(), self.value(b).clone(), self.value(dst).clone());
        let atom = self.low.device.target.group_mma(va.dtype, vd.dtype).expect("inference found the core").clone();
        snafu::ensure!(
            !a_t && vb.tier() == Tier::Smem && self.low.swizzle,
            UnsupportedSnafu { what: "a warpgroup product over other than an untransposed A and swizzled shared B" }
        );
        let grid = self.low.grid;
        let k = va.shape.cols;
        let issue = atom.issue(grid, vd.shape.rows, vd.shape.cols, k).expect("inference laid this product");
        let n_atom = atom.n_for(vd.shape.cols as u32 / grid.cols).expect("inference laid this product") as usize;
        let (rn, rc, rk) = (issue.c.reps[1] as usize, issue.c.frag_regs() as usize, k / wgmma::K);
        // This warp's group and the group's place in the grid, columns first.
        let group = self.hoist(shr(&self.warp, 2));
        let cols = c32(grid.cols as i64);
        let group_row = self.hoist(group.try_cdiv(&cols).expect("group row"));
        let group_col = self.hoist(group.try_cmod(&cols).expect("group column"));
        // The `k16` steps' descriptors of a shared operand from `at`: its
        // first row when K-major (rows along M or N), else its first column
        // (rows along K). Panel `p` of `pc` columns starts `p·rows·pc` in.
        let describe = |this: &mut Self, v: ValId, at: Arc<UOp>, k_major: bool| -> Result<Vec<Arc<UOp>>> {
            let view = this.view(v);
            snafu::ensure!(
                this.along(&view).0 == Axis::Col,
                UnsupportedSnafu { what: "a column-major shared operand" }
            );
            let (rows, e) = (view.shape.rows, view.dtype.bytes());
            let pc = this.panel_cols(view.shape.cols, e).unwrap_or(view.shape.cols);
            let swizzle = wgmma::Swizzle::for_row_bytes(pc * e)
                .context(UnsupportedSnafu { what: format!("a warpgroup operand over {}-byte shared rows", pc * e) })?;
            snafu::ensure!(
                k_major || n_atom.is_multiple_of(pc),
                UnsupportedSnafu { what: format!("an MN-major n{n_atom} block within a {pc}-wide panel") }
            );
            // The leading offset steps between an MN-major operand's panels; a
            // K-major one never reads it and declares one unit, as ever.
            let leading = if k_major { 16 } else { (rows * pc * e) as u32 };
            let stride = (8 * pc * e) as u32;
            let Some(Place::Smem { offset, .. }) = view.place else { unreachable!("a shared operand") };
            let (_, offset) = this.scalar(offset);
            let offset = match &view.region {
                Some(region) => add(&offset, region),
                None => offset,
            };
            let base = if k_major {
                mul(&at, &c32(pc as i64))
            } else {
                mul(&shr(&at, pc.trailing_zeros()), &c32((rows * pc) as i64))
            };
            let base = this.hoist(add(&offset, &base));
            Ok((0..rk)
                .map(|kk| {
                    let kk = kk * wgmma::K;
                    let step = if k_major { kk / pc * rows * pc + kk % pc } else { kk * pc };
                    let at = add(&base, &c32(step as i64));
                    let ptr = this.access(&view.buf, &at, 1);
                    let address = this.hoist(wgmma::shared_address(&ptr));
                    this.hoist(wgmma::descriptor(&address, leading, stride, swizzle))
                })
                .collect())
        };
        // One descriptor is the operand itself, not a one-wide vector.
        let gather =
            |ops: Vec<Arc<UOp>>| if ops.len() == 1 { ops[0].clone() } else { UOp::stack(ops.into_iter().collect()) };
        let from_registers = va.tier() == Tier::Reg;
        let a_op = if from_registers {
            let regs = issue.a.frag_regs() as usize;
            UOp::stack((0..rk).flat_map(|kk| self.reg_loads(a, (kk * regs) as u32, regs as u32)).collect())
        } else {
            let rows = self.hoist(mul(&group_row, &c32(wgmma::M as i64)));
            gather(describe(self, a, rows, true)?)
        };
        let meta = WmmaMetadata {
            name: wgmma::name(from_registers, !b_t, !self.async_products),
            dims: (n_atom, wgmma::M, k),
            dtype_in: DType::Scalar(va.dtype),
            dtype_out: DType::Scalar(vd.dtype),
            device: atom.device,
            threads: wgmma::THREADS,
            upcast_axes: None,
            reduce_axes: vec![],
        };
        for j in 0..rn {
            let at = add(&mul(&group_col, &c32((n_atom * rn) as i64)), &c32((j * n_atom) as i64));
            let at = self.hoist(at);
            let b_op = gather(describe(self, b, at, b_t)?);
            let c = UOp::stack(self.reg_loads(acc, (j * rc) as u32, rc as u32).into_iter().collect());
            let d = UOp::wmma(a_op.clone(), b_op, c, meta.clone());
            let vals = (0..rc).map(|p| elem(&d, p, rc)).collect();
            self.reg_store(dst, (j * rc) as u32, vals);
        }
        Ok(())
    }

    /// The fold runs in the destination's type: a 16-bit source is widened
    /// element by element, except a bf16 sum on a target with `v_dot2_f32_bf16`,
    /// which adds a lane's pairs as held with one instruction each.
    fn reduce(&mut self, dst: ValId, src: ValId, axis: Axis, f: ReduceOp) -> Result<()> {
        let (ls, ld) = (self.layout(src).clone(), self.layout(dst).clone());
        let (from, to) = (self.value(src).dtype, self.value(dst).dtype);
        let dot2 =
            from != to && from == ScalarDType::BFloat16 && f == ReduceOp::Sum && self.low.device.target.dot2_bf16;
        let (fold_masks, across) = match axis {
            Axis::Row => (ls.row_fold_masks(), ls.warps.out_size(Col)),
            Axis::Col => (ls.col_fold_masks(), ls.warps.out_size(Row)),
        };
        snafu::ensure!(across == 1, UnsupportedSnafu { what: "a reduction across warps" });
        let key = |l: &TileLayout, lane: u32, j: u32| {
            let (r, c) = l.coord(0, lane, j);
            if axis == Axis::Row { r } else { c }
        };
        let combine = |x: &Arc<UOp>, y: &Arc<UOp>| {
            match f {
                ReduceOp::Sum => x.try_add(y),
                ReduceOp::Max => x.try_max(y),
                ReduceOp::Min => minimum(x, y),
            }
            .expect("reduce op")
        };
        for jd in 0..ld.regs() {
            let want = key(&ld, 0, jd);
            let members: Vec<u32> = (0..ls.regs()).filter(|&j| key(&ls, 0, j) == want).collect();
            debug_assert!(!members.is_empty());
            let (mut acc, rest) = if dot2 {
                let mut acc = UOp::const_(DType::Float32, ConstValue::Float(0.0));
                let (pairs, rest) = members.as_chunks::<2>();
                for &[a, b] in pairs {
                    let (x, y) = (self.reg_load(src, a), self.reg_load(src, b));
                    acc = self.dot2_bf16(UOp::stack(smallvec![x, y]), acc);
                }
                (acc, rest.to_vec())
            } else {
                let x = self.reg_load(src, members[0]);
                (if from != to { x.cast(DType::Scalar(to)) } else { x }, members[1..].to_vec())
            };
            for &j in &rest {
                let x = self.reg_load(src, j);
                let x = if from != to { x.cast(DType::Scalar(to)) } else { x };
                acc = combine(&acc, &x);
            }
            for &mask in &fold_masks {
                let partner = self.shuffle_xor(&acc, mask);
                acc = combine(&acc, &partner);
            }
            self.reg_store(dst, jd, vec![acc]);
        }
        Ok(())
    }

    /// `dst = a · b` ([`BinaryOp::Scale`]): `a` as it is, then the products
    /// under a branch on whether any factor this thread holds differs from 1,
    /// which a wave whose threads all hold ones jumps over.
    fn scale(&mut self, dst: ValId, a: ValId, b: ValId) {
        let (ja, jb) = (self.operand_regs(dst, a), self.operand_regs(dst, b));
        let regs = ja.len();
        for (j, &src) in ja.iter().enumerate() {
            let x = self.reg_load(a, src);
            self.reg_store(dst, j as u32, vec![x]);
        }
        let mut held = jb.clone();
        held.sort_unstable();
        held.dedup();
        let one = |f: &Arc<UOp>| UOp::const_(f.dtype(), ConstValue::Float(1.0));
        let pred = held
            .into_iter()
            .map(|r| {
                let f = self.reg_load(b, r);
                f.try_cmpne(&one(&f)).expect("factor test")
            })
            .reduce(|p, q| p.try_or_op(&q).expect("or"))
            .expect("a factor");
        let tag = self.next_tag();
        let if_ = self.push(UOp::if_(pred, smallvec![]).rtag(Some(smallvec![tag as usize])));
        for j in 0..regs {
            let (x, y) = (self.reg_load(a, ja[j]), self.reg_load(b, jb[j]));
            self.reg_store(dst, j as u32, vec![x.try_mul(&y).expect("product")]);
        }
        self.push(UOp::endif(if_));
    }

    /// `v` where `gate` holds, else zeros: one select over the whole run (a
    /// multiply would turn an Inf read in its place into NaN).
    fn zeroed(&mut self, gate: &Arc<UOp>, v: Arc<UOp>) -> Arc<UOp> {
        let zero = self.hoist_zero(v.vconst_like(0));
        UOp::try_where(gate.clone(), v, zero).expect("zero fill")
    }

    /// A zero is one hash-consed node per type: a vector of them (rendered as
    /// an instruction chain) is listed in the prologue rather than at its
    /// first use, which another branch would not reach; a scalar is a literal.
    fn hoist_zero(&mut self, zero: Arc<UOp>) -> Arc<UOp> {
        if matches!(zero.op(), Op::Const(..) | Op::VConst(..)) { zero } else { self.hoist(zero) }
    }

    /// `2^x`. Where the target has [`crate::atoms::Target::hardware_exp2`], an f32
    /// one is the hardware instruction alone (`v_exp_f32`), without the
    /// range reduction LLVM wraps `llvm.exp2` in to keep results below 2^-126
    /// denormal: those flush to zero, which no kernel tells apart.
    fn exp2(&mut self, x: &Arc<UOp>) -> Arc<UOp> {
        if self.low.device.target.hardware_exp2 && x.dtype() == DType::Float32 {
            UOp::custom(
                smallvec![x.clone()],
                "declare float @llvm.amdgcn.exp2.f32(float)\ncall float @llvm.amdgcn.exp2.f32(float {0})".to_string(),
                DType::Float32,
            )
        } else {
            x.try_exp2().expect("exp2")
        }
    }

    /// `acc + pair·(1, 1)` for a bf16 `pair`, accumulated in f32.
    fn dot2_bf16(&mut self, pair: Arc<UOp>, acc: Arc<UOp>) -> Arc<UOp> {
        let call = "declare float @llvm.amdgcn.fdot2.f32.bf16(<2 x bfloat>, <2 x bfloat>, float, i1)\n\
                    call float @llvm.amdgcn.fdot2.f32.bf16(<2 x bfloat> {0}, <2 x bfloat> <bfloat 0xR3F80, bfloat 0xR3F80>, float {1}, i1 false)";
        self.push(UOp::custom(smallvec![pair, acc], call.to_string(), DType::Float32))
    }

    fn shuffle_xor(&mut self, value: &Arc<UOp>, mask: u32) -> Arc<UOp> {
        match self.low.device.target.lane_move {
            LaneMove::Shuffle => shfl_bfly(value, &c32(mask as i64)),
            LaneMove::SimdShuffle => svod_codegen::c::metal::simd_shuffle_xor(value, &c32(mask as i64)),
            LaneMove::Dpp if mask < 32 => {
                let (mut word, back) = as_word(value);
                if mask & 15 != 0 {
                    // `row_xmask:mask`, every row and bank written, so no lane keeps
                    // the old value and it need not be set.
                    word = UOp::custom(
                        smallvec![word],
                        format!(
                            "declare i32 @llvm.amdgcn.update.dpp.i32(i32, i32, i32, i32, i32, i1)\n\
                             call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 {{0}}, i32 {}, i32 15, i32 15, i1 false)",
                            DPP_ROW_XMASK | (mask & 15)
                        ),
                        DType::Int32,
                    );
                }
                if mask & 16 != 0 {
                    // Each lane takes the same lane of the other row of 16.
                    word = UOp::custom(
                        smallvec![word],
                        format!(
                            "declare i32 @llvm.amdgcn.permlanex16.i32(i32, i32, i32, i32, i1, i1)\n\
                             call i32 @llvm.amdgcn.permlanex16.i32(i32 {{0}}, i32 {{0}}, i32 {}, i32 {}, i1 false, i1 false)",
                            0x76543210, 0xFEDCBA98_u32 as i32
                        ),
                        DType::Int32,
                    );
                }
                back(word)
            }
            LaneMove::Permute | LaneMove::Dpp => {
                let (word, back) = as_word(value);
                // Pure and lane-only: listed in the prologue, so every branch reaches it.
                let addr = self.hoist(mul(&xor(&self.lane, &c32(mask as i64)), &c32(4)));
                back(UOp::custom(
                    smallvec![addr, word],
                    "declare i32 @llvm.amdgcn.ds.bpermute(i32, i32)\ncall i32 @llvm.amdgcn.ds.bpermute(i32 {0}, i32 {1})"
                        .to_string(),
                    DType::Int32,
                ))
            }
        }
    }
}

/// The block that hardware position `linear` of `total` runs on a target of
/// `dies` dies, each taking every `dies`th block of a dispatch: the dies'
/// shares laid end to end (the first `total % dies` dies hold a block more),
/// so a die runs a contiguous range of the kernel's block order.
pub fn die_order(linear: u64, total: u64, dies: u64) -> u64 {
    let (base, tall) = (total / dies, total % dies);
    let (die, local) = (linear % dies, linear / dies);
    die * base + die.min(tall) + local
}

/// `min(x, y)`, which the IR has no node for: a float's is `-max(-x, -y)`,
/// exact at infinities and NaN-ignoring as `max` is; an integer's is
/// `x + y - max(x, y)`, exact short of overflow.
pub(crate) fn minimum(x: &Arc<UOp>, y: &Arc<UOp>) -> svod_ir::Result<Arc<UOp>> {
    if x.dtype().is_float() {
        return Ok(x.neg().try_max(&y.neg())?.neg());
    }
    x.try_add(y)?.try_sub(&x.try_max(y)?)
}

/// DPP's `row_xmask` control: OR the lane xor-mask within a row of 16.
const DPP_ROW_XMASK: u32 = 0x160;

/// `value` as the 32-bit word a lane move carries, and the way back: a
/// 16-bit value travels zero-extended. A lane move carries one 2- or 4-byte
/// scalar, nothing else.
fn as_word(value: &Arc<UOp>) -> (Arc<UOp>, impl Fn(Arc<UOp>) -> Arc<UOp>) {
    let dtype = value.dtype();
    let bytes = dtype.scalar().map(|s| s.bytes());
    assert!(matches!(bytes, Some(2 | 4)), "a lane move carries a 2- or 4-byte scalar, not {dtype:?}");
    let wide = bytes == Some(4);
    let int = dtype == DType::Int32;
    let word = if int {
        value.clone()
    } else if wide {
        value.bitcast(DType::Int32)
    } else {
        value.bitcast(DType::Scalar(ScalarDType::UInt16)).cast(DType::Int32)
    };
    let back = move |w: Arc<UOp>| {
        if int {
            w
        } else if wide {
            w.bitcast(dtype.clone())
        } else {
            w.cast(DType::Scalar(ScalarDType::UInt16)).bitcast(dtype.clone())
        }
    };
    (word, back)
}

/// Whether scalar `id` reads [`Scalar::Row`], memoized in `memo`.
fn reads_row(prog: &Program, id: ScalarId, memo: &mut Vec<Option<bool>>) -> bool {
    if let Some(known) = memo[id.index()] {
        return known;
    }
    let dep = match prog.scalar(id) {
        Scalar::Row => true,
        Scalar::Bin(_, a, b) => reads_row(prog, *a, memo) | reads_row(prog, *b, memo),
        Scalar::Load { index, .. } => reads_row(prog, *index, memo),
        Scalar::Const(_) | Scalar::Var(_) | Scalar::Special(_) | Scalar::Induction => false,
    };
    memo[id.index()] = Some(dep);
    dep
}
