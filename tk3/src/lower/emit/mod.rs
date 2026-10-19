//! Statement-by-statement emission into a pre-linearized instruction list.
//! Register tiles are per-thread register buffers addressed by register
//! index; every statement's instructions are listed in program order, so no
//! toposort decides anything.

use std::collections::HashMap;
use std::sync::Arc;

use smallvec::smallvec;
use svod_codegen::llvm::nvptx::smem::{cp_async_commit, cp_async_wait};
use svod_codegen::llvm::nvptx::wgmma;
use svod_dtype::{AddrSpace, DType, DeviceSpec, GpuArch, ScalarDType};
use svod_ir::uop::cached_property::CachedProperty;
use svod_ir::uop::properties::SoundVminVmaxProperty;
use svod_ir::{AxisId, AxisType, KernelInfo, Op, UOp};

use crate::index::*;
use crate::ir::*;
use crate::layouts::{Laid, TileLayout};
use crate::lower::{Lowering, ProgramSnafu, Result, UnsupportedSnafu};
use snafu::ResultExt;

mod address;
mod amdgpu;
mod compute;
mod copy;
mod nvptx;

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
            // branches it is met in, where the index is in range. Only a sound
            // range proves it inside: the plain one has no rule for a
            // truncating quotient or remainder.
            Scalar::Load { param, index } => {
                let (level, index) = self.scalar_at(*index, row);
                let buf = self.params[param.index()].clone();
                let idx = self.access(&buf, &index, 1);
                let tag = self.next_tag();
                let elems = self.prog.params[param.index()].elems as i64;
                let inside = SoundVminVmaxProperty::get(&index)
                    .as_ref()
                    .and_then(|(lo, hi)| Some((lo.try_int()?, hi.try_int()?)))
                    .is_some_and(|(lo, hi)| lo >= 0 && hi < elems);
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
            Sync::Barrier { scope: Scope::Warp } if cuda => nvptx::warp_barrier(&self.last),
            Sync::Barrier { scope: Scope::Warp } => return Ok(()),
            Sync::CommitAsync if cuda && self.low.device.target.cp_async => cp_async_commit(smallvec![]),
            Sync::WaitAsync { pending } if cuda && self.low.device.target.cp_async => {
                cp_async_wait(pending, smallvec![])
            }
            // Without `cp.async` the copies were synchronous: nothing to wait for.
            Sync::CommitAsync | Sync::WaitAsync { .. } => return Ok(()),
            Sync::Fence if self.low.device.target.commit_fence => amdgpu::sched_barrier(&self.last),
            Sync::Fence => return Ok(()),
            Sync::WaitProducts { pending } if cuda && self.low.group => wgmma::wait(pending),
            Sync::WaitProducts { .. } => return Ok(()),
            Sync::Prio { level } if matches!(self.low.device.target.arch, GpuArch::Amd(_)) => amdgpu::setprio(level),
            Sync::Prio { .. } => return Ok(()),
        };
        self.push(u);
        Ok(())
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

/// `min(x, y)`, which the IR has no node for: `-max(-x, -y)`. A float's is
/// exact at infinities and NaN-ignoring as `max` is. An integer's range is
/// exact, `[min(x₀, y₀), min(x₁, y₁)]`, where `x + y - max(x, y)` loses the
/// correlation of its terms and widens an index past its parameter; its
/// negation wraps, so a side of `i32::MIN` yields the other side. Only a
/// scalar read from memory can hold `i32::MIN`, a length no valid input has:
/// every other operand is a size, a coordinate or a variable in its range.
pub(crate) fn minimum(x: &Arc<UOp>, y: &Arc<UOp>) -> svod_ir::Result<Arc<UOp>> {
    Ok(x.neg().try_max(&y.neg())?.neg())
}

/// Whether scalar `id` is a multiple of `m` whatever its inputs.
fn multiple_of(prog: &Program, id: ScalarId, m: i64) -> bool {
    match prog.scalar(id) {
        Scalar::Const(v) => v % m == 0,
        Scalar::Bin(BinOp::Add | BinOp::Sub | BinOp::Min | BinOp::Max, a, b) => {
            multiple_of(prog, *a, m) && multiple_of(prog, *b, m)
        }
        Scalar::Bin(BinOp::Mul, a, b) => multiple_of(prog, *a, m) || multiple_of(prog, *b, m),
        _ => false,
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
