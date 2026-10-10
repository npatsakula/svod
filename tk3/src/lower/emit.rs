//! Statement-by-statement emission into a pre-linearized instruction list.
//! Register tiles are per-thread register buffers addressed by register
//! index; every statement's instructions are listed in program order, so no
//! toposort decides anything.

use std::collections::HashMap;
use std::sync::Arc;

use smallvec::smallvec;
use svod_codegen::llvm::nvptx::ops::shfl_bfly;
use svod_codegen::llvm::nvptx::smem::{cp_async_16, cp_async_16_zfill, cp_async_commit, cp_async_wait, ldmatrix};
use svod_dtype::{AddrSpace, DType, DeviceSpec, GpuArch, ScalarDType};
use svod_ir::{AxisId, AxisType, ConstValue, KernelInfo, Op, UOp};

use super::{Lowering, ProgramSnafu, Result, UnsupportedSnafu};
use crate::index::*;
use crate::ir::*;
use crate::layout::Dim::{Col, Lane, Reg, Row, Warp};
use crate::layout::{self as frag, Layout};
use crate::layouts::{Laid, Relayout, TileLayout};
use snafu::ResultExt;

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
    /// Insertion point of values computed at each loop nesting level.
    headers: Vec<usize>,
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
    scratch: HashMap<ScalarDType, (Arc<UOp>, usize)>,
    lane: Arc<UOp>,
    warp: Arc<UOp>,
    tid: Arc<UOp>,
    range_id: usize,
    reg_slot: usize,
    local_slot: usize,
    tag: u64,
    last: Arc<UOp>,
}

/// A register-side view of a tiled value for addressing shared/global memory.
struct View {
    buf: Arc<UOp>,
    place: Place,
    shape: Shape,
    dtype: ScalarDType,
}

impl<'a> Emit<'a> {
    fn new(prog: &'a Program, lay: &'a [Option<TileLayout>], low: &'a Lowering, params: Vec<Arc<UOp>>) -> Self {
        let threads = prog.warps * low.target.wave;
        let tid = UOp::special_dtype(c32(threads as i64), "lidx0".to_string(), DType::Int32);
        let wave = c32(low.target.wave as i64);
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
            scratch: HashMap::new(),
            last: tid.clone(),
            lane,
            warp,
            tid,
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
        if self.levels.contains_key(&u.id) {
            return u;
        }
        let mut level = 0;
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
            Scalar::Special(Special::Block(axis)) => {
                let end = self.scalar(self.prog.grid[*axis as usize]).1;
                (0, UOp::special_dtype(end, format!("gidx{axis}"), DType::Int32))
            }
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
                    BinOp::Min => a.try_max(&b).map(|m| a.try_add(&b).unwrap().try_sub(&m).unwrap()),
                    BinOp::Max => a.try_max(&b),
                    BinOp::Lt => a.try_cmplt(&b).map(|c| c.cast(DType::Int32)),
                    BinOp::Le => a.try_cmple(&b).map(|c| c.cast(DType::Int32)),
                    BinOp::Eq => a.try_cmpeq(&b).map(|c| c.cast(DType::Int32)),
                    BinOp::And => a.try_and_op(&b),
                    BinOp::Or => a.try_or_op(&b),
                };
                (la.max(lb), u.expect("i32 scalar"))
            }
            Scalar::Load { param, index } => {
                let (level, index) = self.scalar_at(*index, row);
                let buf = self.params[param.index()].clone();
                let idx = self.access(&buf, &index, 1);
                let tag = self.next_tag();
                let load = self.hoist(load_at(&idx, tag));
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
            Place::Smem { alloc, .. } => self.smem[alloc.index()].clone(),
            Place::Reg => unreachable!("a memory view"),
        };
        View { buf, place: val.place, shape: val.shape, dtype: val.dtype }
    }

    /// Element offset of `(row, col)` in a memory view, with the swizzle a
    /// shared view takes, and its bounds gate if any. Both are listed at the
    /// nesting level of their inputs, so a loop body computes them once and
    /// an epilogue after the loop may still use them.
    fn address(&mut self, view: &View, row: &Arc<UOp>, col: &Arc<UOp>) -> (Arc<UOp>, Option<Arc<UOp>>) {
        match &view.place {
            Place::Global { offset, stride, bounds, rows: Some(map), .. } => {
                let ((_, offset), (_, s1)) = (self.scalar(*offset), self.scalar(stride[1]));
                let (_, start) = self.scalar_at(map.offset, Some(row));
                // The view's offset moves per step; the row's start and the
                // column are listed where their inputs are.
                let off = add(&offset, &add(&start, &mul(col, &s1)));
                let valid = map.valid.map(|v| {
                    let (_, v) = self.scalar_at(v, Some(row));
                    v.try_cmpne(&c32(0)).expect("row gate")
                });
                let in_cols = bounds[1].map(|b| col.try_cmplt(&self.sc(b)).expect("bound compare"));
                let gate = valid.into_iter().chain(in_cols).reduce(|a, b| a.try_and_op(&b).expect("and"));
                (self.hoist(off), gate.map(|g| self.hoist(g)))
            }
            Place::Global { offset, stride, bounds, .. } => {
                let ((l0, offset), (l1, s0), (l2, s1)) =
                    (self.scalar(*offset), self.scalar(stride[0]), self.scalar(stride[1]));
                let mut level = l0.max(l1).max(l2);
                let off = add(&add(&offset, &mul(row, &s0)), &mul(col, &s1));
                let gate = bounds
                    .iter()
                    .zip([row, col])
                    .filter_map(|(b, i)| b.map(|b| (self.scalar(b), i)))
                    .map(|((l, bound), i)| {
                        level = level.max(l);
                        i.try_cmplt(&bound).expect("bound compare")
                    })
                    .reduce(|a, b| a.try_and_op(&b).expect("and"));
                (self.hoist(off), gate.map(|g| self.hoist(g)))
            }
            Place::Smem { offset, .. } => {
                let offset = self.sc(*offset);
                let cols = view.shape.cols as i64;
                let chunk = 16 / view.dtype.bytes() as i64;
                let cpr = cols / chunk;
                // The XOR must stay within the row: a power-of-two chunk count.
                let col = if self.low.swizzle && cols % chunk == 0 && cpr >= 2 && (cpr as u64).is_power_of_two() {
                    let bits = chunk.trailing_zeros();
                    let (c, within) = (shr(col, bits), and(col, &c32(chunk - 1)));
                    let sw = if cpr >= 8 {
                        and(row, &c32(7))
                    } else {
                        and(&shr(row, (8 / cpr).trailing_zeros()), &c32(cpr - 1))
                    };
                    add(&mul(&xor(&c, &sw), &c32(chunk)), &within)
                } else {
                    col.clone()
                };
                let off = add(&add(&offset, &mul(row, &c32(cols))), &col);
                (self.hoist(off), None)
            }
            Place::Reg => unreachable!(),
        }
    }

    /// Integer coordinate `(row, col)` register `j` of this thread holds;
    /// induction-free, so listed in the prologue.
    fn coord(&mut self, l: &TileLayout, j: u32) -> (Arc<UOp>, Arc<UOp>) {
        let regs = l.frag_regs();
        let rep = j / regs;
        let (rr, rc) = (rep / l.reps[1], rep % l.reps[1]);
        let (fr, fc) = l.frag.apply(&[(Reg, j % regs)]);
        let [fr_n, fc_n] = l.frag_shape();
        let [sr, sc] = l.sub_shape();
        let bit_terms = |src: &Arc<UOp>, layout: &Layout, dim, init: (u32, u32)| {
            let (mut r, mut c) = (c32(init.0 as i64), c32(init.1 as i64));
            for b in 0..layout.in_bits(dim) {
                let (vr, vc) = {
                    let out = layout.basis(dim, b);
                    let get = |d| out.iter().find(|o| o.0 == d).map_or(0, |o| o.1);
                    (get(Row), get(Col))
                };
                if vr == 0 && vc == 0 {
                    continue;
                }
                let bit = and(&shr(src, b), &c32(1));
                if vr != 0 {
                    r = xor(&r, &mul(&bit, &c32(vr as i64)));
                }
                if vc != 0 {
                    c = xor(&c, &mul(&bit, &c32(vc as i64)));
                }
            }
            (r, c)
        };
        let (fr, fc) = bit_terms(&self.lane, &l.frag, Lane, (fr, fc));
        let (wr, wc) = bit_terms(&self.warp, &l.warps, Warp, (0, 0));
        let row = add(&add(&mul(&wr, &c32(sr as i64)), &c32((rr * fr_n) as i64)), &fr);
        let col = add(&add(&mul(&wc, &c32(sc as i64)), &c32((rc * fc_n) as i64)), &fc);
        (self.hoist(row), self.hoist(col))
    }

    /// Register runs `(start, width)` whose elements are consecutive columns
    /// of one row in every lane, aligned to their width.
    fn runs(l: &TileLayout, lanes: u32, max_width: u32) -> Vec<(u32, u32)> {
        let mut out = vec![];
        let mut j = 0;
        while j < l.regs() {
            let mut w = max_width.min(l.regs() - j);
            while w > 1 {
                let ok = (0..lanes).all(|lane| {
                    let (r0, c0) = l.coord(0, lane, j);
                    c0 % w == 0 && (1..w).all(|e| l.coord(0, lane, j + e) == (r0, c0 + e))
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

    // ---- statements -------------------------------------------------------

    fn block(&mut self, block: &Block) -> Result<()> {
        self.pending.push(vec![]);
        for stmt in &block.0 {
            self.stmt(stmt)?;
        }
        for c in self.pending.pop().expect("the block just opened") {
            for j in 0..self.layout(c.phi).regs() {
                let x = self.reg_load(c.next, j);
                self.reg_store(c.phi, j, vec![x]);
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
            Stmt::Copy { dst, src, mode } => {
                self.copy(*dst, *src, *mode)?;
                self.advance(*dst);
                Ok(())
            }
            Stmt::Loop(l) => self.loop_(l),
            Stmt::Pipeline(_) => UnsupportedSnafu { what: "an unexpanded pipeline" }.fail(),
            Stmt::Role { .. } => UnsupportedSnafu { what: "warp roles" }.fail(),
            Stmt::If { pred, then, otherwise } => {
                let pred = self.sc(*pred);
                let cond = pred.try_cmpne(&c32(0)).expect("predicate");
                for (branch, cond) in [(then, cond.clone()), (otherwise, cond.not())] {
                    if branch.0.is_empty() {
                        continue;
                    }
                    let tag = self.next_tag();
                    let if_ = self.push(UOp::if_(cond, smallvec![]).rtag(Some(smallvec![tag as usize])));
                    self.block(branch)?;
                    self.push(UOp::endif(if_));
                }
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
        let cuda = matches!(self.low.target.arch, GpuArch::Cuda(_));
        let u = match sync {
            Sync::Barrier { role: None } => {
                let tag = self.next_tag();
                self.last.clone().barrier(smallvec![]).rtag(Some(smallvec![tag as usize]))
            }
            Sync::Barrier { role: Some(_) } => return UnsupportedSnafu { what: "role barriers" }.fail(),
            Sync::CommitAsync if cuda => cp_async_commit(smallvec![]),
            Sync::WaitAsync { pending } if cuda => cp_async_wait(pending, smallvec![]),
            Sync::CommitAsync | Sync::WaitAsync { .. } => {
                return UnsupportedSnafu { what: "async copies off CUDA" }.fail();
            }
            Sync::Fence if self.low.target.commit_fence => UOp::custom(
                smallvec![self.last.clone()],
                "declare void @llvm.amdgcn.sched.barrier(i32)\ncall void @llvm.amdgcn.sched.barrier(i32 0)".to_string(),
                DType::Void,
            ),
            Sync::Fence => return Ok(()),
        };
        self.push(u);
        Ok(())
    }

    // ---- copies ---------------------------------------------------------------

    fn copy(&mut self, dst: ValId, src: ValId, mode: CopyMode) -> Result<()> {
        match (self.value(dst).tier(), self.value(src).tier()) {
            (Tier::Smem, Tier::Global) => self.fill(dst, src, mode),
            (Tier::Reg, Tier::Smem) => self.gather(dst, src),
            (Tier::Reg, Tier::Global) => self.load_regs(dst, src),
            (Tier::Smem | Tier::Global, Tier::Reg) => self.store_regs(dst, src),
            (Tier::Reg, Tier::Reg) => self.relayout(dst, src),
            (d, s) => UnsupportedSnafu { what: format!("a {s:?} → {d:?} copy") }.fail(),
        }
    }

    /// Global → shared, every thread moving 16-byte chunks.
    fn fill(&mut self, dst: ValId, src: ValId, mode: CopyMode) -> Result<()> {
        let (d, s) = (self.view(dst), self.view(src));
        let chunk = 16 / d.dtype.bytes() as i64;
        let cols = d.shape.cols as i64;
        let threads = (self.prog.warps * self.low.target.wave) as i64;
        let chunks = d.shape.rows as i64 * cols / chunk;
        let cuda = matches!(self.low.target.arch, GpuArch::Cuda(_));
        snafu::ensure!(
            cols % chunk == 0 && chunks % threads == 0,
            UnsupportedSnafu { what: format!("a {}×{} fill by {threads} threads", d.shape.rows, cols) }
        );
        let Place::Global { stride, bounds, rows, .. } = &s.place else { unreachable!() };
        let gathered = rows.is_some();
        snafu::ensure!(
            matches!(self.prog.scalar(stride[1]), Scalar::Const(1)),
            UnsupportedSnafu { what: "a fill from a column-strided view" }
        );
        let bounds = *bounds;
        let cpr = cols / chunk;
        for t in 0..chunks / threads {
            let c = add(&self.tid, &c32(t * threads));
            let (row, cc) = (c.try_cdiv(&c32(cpr)).expect("row"), c.try_cmod(&c32(cpr)).expect("chunk"));
            let (row, col) = (self.hoist(row), self.hoist(mul(&cc, &c32(chunk))));
            // Rows past a bound re-read the last valid one; the consumer masks.
            let src_row = match bounds[0] {
                Some(b) => {
                    let bound = self.sc(b);
                    let last = bound.try_sub(&c32(1)).expect("bound");
                    let over = row.try_cmplt(&last).expect("cmp");
                    self.hoist(UOp::try_where(over, row.clone(), last).expect("clamp"))
                }
                None => row.clone(),
            };
            let (src_off, gate) = self.address(&s, &src_row, &col);
            // A plain view's fill clamps (above); a gathered row outside its
            // map lands as zeros, from element 0 of the operand.
            let gate = gate.filter(|_| gathered);
            let src_off = match &gate {
                Some(g) => self.hoist(UOp::try_where(g.clone(), src_off, c32(0)).expect("safe offset")),
                None => src_off,
            };
            let (dst_off, _) = self.address(&d, &row, &col);
            if cuda && mode == CopyMode::Async && self.low.target.cp_async {
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
        if self.low.target.ldmatrix && s.dtype.bytes() == 2 && sub_r % 16 == 0 && sub_c % 16 == 0 {
            for trans in [false, true] {
                let produced = TileLayout {
                    frag: frag::ldmatrix_x4(trans),
                    reps: [sub_r / 16, sub_c / 16],
                    warps: l.warps.clone(),
                };
                let perm = match produced.relayout(&l, self.prog.warps, self.low.target.wave) {
                    Relayout::Identity => (0..l.regs()).collect::<Vec<_>>(),
                    Relayout::RegPermute(p) => p,
                    _ => continue,
                };
                return self.ldmatrix_gather(dst, &s, &l, trans, &perm);
            }
        }
        for (j, w) in Self::runs(&l, self.low.target.wave, 16 / s.dtype.bytes() as u32) {
            let (row, col) = self.coord(&l, j);
            let (off, _) = self.address(&s, &row, &col);
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
            self.coord(&TileLayout { frag: Layout::zeros(Lane, 1), reps: [1, 1], warps: l.warps.clone() }, 0);
        let (wr, wc) = (self.hoist(mul(&wr, &c32(sub_r as i64))), self.hoist(mul(&wc, &c32(sub_c as i64))));
        for br in 0..blocks_r {
            for bc in 0..blocks_c {
                let row = self.hoist(add(&add(&wr, &c32((br * 16) as i64)), &lane16));
                let col = self.hoist(add(&add(&wc, &c32((bc * 16) as i64)), &lane_hi));
                let (off, _) = self.address(s, &row, &col);
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
    fn load_regs(&mut self, dst: ValId, src: ValId) -> Result<()> {
        let l = self.layout(dst).clone();
        let s = self.view(src);
        let gathered = matches!(s.place, Place::Global { rows: Some(_), .. });
        for (j, w) in Self::runs(&l, self.low.target.wave, 16 / s.dtype.bytes() as u32) {
            let (row, col) = self.coord(&l, j);
            let (off, gate) = self.address(&s, &row, &col);
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

    fn store_regs(&mut self, dst: ValId, src: ValId) -> Result<()> {
        let l = self.layout(src).clone();
        let d = self.view(dst);
        for (j, w) in Self::runs(&l, self.low.target.wave, 16 / d.dtype.bytes() as u32) {
            let (row, col) = self.coord(&l, j);
            let (off, gate) = self.address(&d, &row, &col);
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
        match ls.relayout(&ld, self.prog.warps, self.low.target.wave) {
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
            Relayout::LaneShuffle(_) | Relayout::ViaSmem => self.relayout_via_smem(dst, src),
        }
    }

    fn relayout_via_smem(&mut self, dst: ValId, src: ValId) -> Result<()> {
        let Value { dtype, shape, .. } = self.value(src).clone();
        let elems = shape.elems();
        let scratch = match self.scratch.get(&dtype) {
            Some((buf, size)) if *size >= elems => buf.clone(),
            _ => {
                let buf = UOp::buffer(self.local_slot, elems, DType::Scalar(dtype), AddrSpace::Local, None);
                self.local_slot += 1;
                let buf = self.hoist(buf);
                self.scratch.insert(dtype, (buf.clone(), elems));
                buf
            }
        };
        let view =
            View { buf: scratch, place: Place::Smem { alloc: SmemId(u32::MAX), offset: self.zero() }, shape, dtype };
        self.sync(Sync::Barrier { role: None })?;
        self.store_view(&view, src)?;
        self.sync(Sync::Barrier { role: None })?;
        self.gather_view(dst, &view)
    }

    fn zero(&mut self) -> ScalarId {
        if let Some(id) = self.prog.scalars.iter().position(|s| *s == Scalar::Const(0)) {
            return ScalarId(id as u32);
        }
        unreachable!("every program has a zero constant")
    }

    fn store_view(&mut self, d: &View, src: ValId) -> Result<()> {
        let l = self.layout(src).clone();
        for (j, w) in Self::runs(&l, self.low.target.wave, 16 / d.dtype.bytes() as u32) {
            let (row, col) = self.coord(&l, j);
            let (off, _) = self.address(d, &row, &col);
            let vals = self.reg_loads(src, j, w);
            self.mem_store(&d.buf, &off, vals);
        }
        Ok(())
    }

    fn gather_view(&mut self, dst: ValId, s: &View) -> Result<()> {
        let l = self.layout(dst).clone();
        for (j, w) in Self::runs(&l, self.low.target.wave, 16 / s.dtype.bytes() as u32) {
            let (row, col) = self.coord(&l, j);
            let (off, _) = self.address(s, &row, &col);
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
                    let (row, col) = self.coord(&l, j);
                    self.reg_store(dst, j, vec![if axis == Axis::Row { row } else { col }]);
                }
            }
            TileOp::Unary { src, f } => {
                for j in 0..regs {
                    let x = self.reg_load(src, j);
                    let y = match f {
                        UnaryOp::Neg => x.neg(),
                        UnaryOp::Exp2 => x.try_exp2().expect("exp2"),
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
            TileOp::Binary { a, b, f } => {
                let (ja, jb) = (self.operand_regs(dst, a), self.operand_regs(dst, b));
                for j in 0..regs {
                    let (x, y) = (self.reg_load(a, ja[j as usize]), self.reg_load(b, jb[j as usize]));
                    let z = match f {
                        BinaryOp::Add => x.try_add(&y),
                        BinaryOp::Sub => x.try_sub(&y),
                        BinaryOp::Mul => x.try_mul(&y),
                        BinaryOp::Div if x.dtype().is_float() => x.try_div(&y),
                        BinaryOp::Div => x.try_cdiv(&y),
                        BinaryOp::Max => x.try_max(&y),
                        BinaryOp::Min => x.try_max(&y).map(|m| x.try_add(&y).unwrap().try_sub(&m).unwrap()),
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
            TileOp::Cast { src, to } => {
                for j in 0..regs {
                    let x = self.reg_load(src, j);
                    self.reg_store(dst, j, vec![x.cast(DType::Scalar(to))]);
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
            debug_assert_eq!(ld, lo, "elementwise operands share a layout");
            return (0..ld.regs()).collect();
        }
        let key = |r: u32, c: u32| if os.cols == 1 { (r, 0) } else { (0, c) };
        let lanes = self.low.target.wave;
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
        let atom = self.low.target.mma(va.dtype, vd.dtype).expect("inference found the core").clone();
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
                let mut c = UOp::stack(self.reg_loads(acc, cblock * rc, rc).into_iter().collect());
                for kk in 0..rk {
                    let xa = UOp::stack(self.reg_loads(a, block_a(i, kk) * ra, ra).into_iter().collect());
                    let xb = UOp::stack(self.reg_loads(b, block_b(j, kk) * rb, rb).into_iter().collect());
                    let (x, y) = if swapped { (xb, xa) } else { (xa, xb) };
                    c = UOp::wmma(x, y, c, atom.meta.clone());
                }
                let vals = (0..rc as usize).map(|p| elem(&c, p, rc as usize)).collect();
                self.reg_store(dst, cblock * rc, vals);
            }
        }
        Ok(())
    }

    fn reduce(&mut self, dst: ValId, src: ValId, axis: Axis, f: ReduceOp) -> Result<()> {
        let (ls, ld) = (self.layout(src).clone(), self.layout(dst).clone());
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
                ReduceOp::Min => x.try_max(y).map(|m| x.try_add(y).unwrap().try_sub(&m).unwrap()),
            }
            .expect("reduce op")
        };
        for jd in 0..ld.regs() {
            let want = key(&ld, 0, jd);
            let members: Vec<u32> = (0..ls.regs()).filter(|&j| key(&ls, 0, j) == want).collect();
            debug_assert!(!members.is_empty());
            let mut acc = self.reg_load(src, members[0]);
            for &j in &members[1..] {
                let x = self.reg_load(src, j);
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

    fn shuffle_xor(&mut self, value: &Arc<UOp>, mask: u32) -> Arc<UOp> {
        match self.low.target.arch {
            GpuArch::Cuda(_) => shfl_bfly(value, &c32(mask as i64)),
            GpuArch::Metal(_) => svod_codegen::c::metal::simd_shuffle_xor(value, &c32(mask as i64)),
            GpuArch::Amd(_) => {
                let is_f32 = value.dtype() == DType::Float32;
                let data = if is_f32 { value.bitcast(DType::Int32) } else { value.clone() };
                // Pure and lane-only: listed in the prologue, so every branch reaches it.
                let addr = self.hoist(mul(&xor(&self.lane, &c32(mask as i64)), &c32(4)));
                let sh = UOp::custom(
                    smallvec![addr, data],
                    "declare i32 @llvm.amdgcn.ds.bpermute(i32, i32)\ncall i32 @llvm.amdgcn.ds.bpermute(i32 {0}, i32 {1})"
                        .to_string(),
                    DType::Int32,
                );
                if is_f32 { sh.bitcast(DType::Float32) } else { sh }
            }
        }
    }
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
