//! The recording builder: every call appends a statement to the open block
//! and hands back a tier-typed handle. Tiles carry their tier and element type
//! in the Rust type; shapes are checked when the statement is recorded.
//! Scalars are ordinary expressions with operators, interned when consumed.

use std::marker::PhantomData;
use std::ops::Range;
use std::rc::Rc;

use svod_dtype::ScalarDType;

use crate::ir::*;

pub trait Elem: Copy {
    const DTYPE: ScalarDType;
}
macro_rules! elem {
    ($($name:ident = $dtype:ident),*) => {$(
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub struct $name;
        impl Elem for $name {
            const DTYPE: ScalarDType = ScalarDType::$dtype;
        }
    )*};
}
elem!(BF16 = BFloat16, F16 = Float16, F32 = Float32, I32 = Int32, Bool = Bool);

pub trait TierMark: Copy {
    const TIER: Tier;
}
macro_rules! tier {
    ($($name:ident),*) => {$(
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub struct $name;
        impl TierMark for $name {
            const TIER: Tier = Tier::$name;
        }
    )*};
}
tier!(Global, Smem, Reg);

/// A tile handle; `P` is its tier marker and `T` its element type.
pub struct Tile<P, T>(pub ValId, PhantomData<(P, T)>);

impl<P, T> Clone for Tile<P, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<P, T> Copy for Tile<P, T> {}
impl<P, T> std::fmt::Debug for Tile<P, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tile({})", self.0.0)
    }
}
impl<P, T> From<Tile<P, T>> for ValId {
    fn from(t: Tile<P, T>) -> Self {
        t.0
    }
}
impl<P, T> Tile<P, T> {
    fn new(id: ValId) -> Self {
        Self(id, PhantomData)
    }
}

pub type Gmem<T> = Tile<Global, T>;
pub type Shared<T> = Tile<Smem, T>;
pub type Regs<T> = Tile<Reg, T>;

/// A scalar expression: built with ordinary operators, interned into the
/// program's arena by the builder call that consumes it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Sc {
    Id(ScalarId),
    Const(i64),
    Bin(BinOp, Rc<Sc>, Rc<Sc>),
}

impl Sc {
    fn bin(self, op: BinOp, rhs: impl Into<Sc>) -> Sc {
        Sc::Bin(op, Rc::new(self), Rc::new(rhs.into()))
    }
    pub fn min(self, rhs: impl Into<Sc>) -> Sc {
        self.bin(BinOp::Min, rhs)
    }
    pub fn max(self, rhs: impl Into<Sc>) -> Sc {
        self.bin(BinOp::Max, rhs)
    }
    pub fn lt(self, rhs: impl Into<Sc>) -> Sc {
        self.bin(BinOp::Lt, rhs)
    }
    pub fn le(self, rhs: impl Into<Sc>) -> Sc {
        self.bin(BinOp::Le, rhs)
    }
    pub fn eq(self, rhs: impl Into<Sc>) -> Sc {
        self.bin(BinOp::Eq, rhs)
    }
    pub fn and(self, rhs: impl Into<Sc>) -> Sc {
        self.bin(BinOp::And, rhs)
    }
    pub fn or(self, rhs: impl Into<Sc>) -> Sc {
        self.bin(BinOp::Or, rhs)
    }
}

macro_rules! sc_from {
    ($($t:ty),*) => {$(impl From<$t> for Sc {
        fn from(v: $t) -> Sc {
            Sc::Const(v as i64)
        }
    })*};
}
sc_from!(i64, i32, usize, u32);

macro_rules! sc_ops {
    ($($trait:ident $method:ident $op:ident),*) => {$(
        impl<R: Into<Sc>> std::ops::$trait<R> for Sc {
            type Output = Sc;
            fn $method(self, rhs: R) -> Sc {
                self.bin(BinOp::$op, rhs)
            }
        }
    )*};
}
sc_ops!(Add add Add, Sub sub Sub, Mul mul Mul, Div div Div, Rem rem Rem);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParamRef<T>(pub ParamId, PhantomData<T>);

pub struct Kernel {
    prog: Program,
    blocks: Vec<Block>,
}

impl Kernel {
    pub fn new(name: impl Into<String>) -> Self {
        let mut k = Self {
            prog: Program {
                name: name.into(),
                params: vec![],
                vars: vec![],
                grid: [ScalarId(0); 3],
                warps: 4,
                roles: vec![],
                smem: vec![],
                scalars: vec![],
                values: vec![],
                body: Block::default(),
            },
            blocks: vec![Block::default()],
        };
        let one = k.push_scalar(Scalar::Const(1));
        k.prog.grid = [one; 3];
        k
    }

    // ---- program-level declarations -------------------------------------

    pub fn param<T: Elem>(&mut self, name: impl Into<String>, kind: ParamKind, elems: usize) -> ParamRef<T> {
        self.prog.params.push(Param { name: name.into(), dtype: T::DTYPE, kind, elems });
        ParamRef(ParamId(self.prog.params.len() as u32 - 1), PhantomData)
    }

    /// A symbolic variable in `min..=max` the launch binds by name.
    pub fn var(&mut self, name: impl Into<String>, min: i64, max: i64) -> Sc {
        let name = name.into();
        if !self.prog.vars.iter().any(|v| v.name == name) {
            self.prog.vars.push(Var { name: name.clone(), min, max });
        }
        self.scalar(Scalar::Var(name))
    }

    pub fn grid(&mut self, dims: [Sc; 3]) {
        self.prog.grid = dims.map(|s| self.intern(s));
    }

    pub fn warps(&mut self, warps: u32) {
        self.prog.warps = warps;
    }

    pub fn role(&mut self, name: impl Into<String>, warps: Range<u32>, regs: Option<u32>) -> RoleId {
        self.prog.roles.push(Role { name: name.into(), warps, regs });
        RoleId(self.prog.roles.len() as u32 - 1)
    }

    pub fn smem<T: Elem>(&mut self, name: impl Into<String>, elems: usize) -> SmemId {
        self.prog.smem.push(SmemAlloc { name: name.into(), dtype: T::DTYPE, elems, along: Axis::Col });
        SmemId(self.prog.smem.len() as u32 - 1)
    }

    // ---- scalars ---------------------------------------------------------

    fn push_scalar(&mut self, s: Scalar) -> ScalarId {
        self.prog.scalars.push(s);
        ScalarId(self.prog.scalars.len() as u32 - 1)
    }

    fn scalar(&mut self, s: Scalar) -> Sc {
        Sc::Id(self.push_scalar(s))
    }

    /// Intern an expression into the arena.
    pub fn intern(&mut self, sc: impl Into<Sc>) -> ScalarId {
        match sc.into() {
            Sc::Id(id) => id,
            Sc::Const(v) => self.push_scalar(Scalar::Const(v)),
            Sc::Bin(op, a, b) => {
                let (a, b) = (self.intern((*a).clone()), self.intern((*b).clone()));
                self.push_scalar(Scalar::Bin(op, a, b))
            }
        }
    }

    pub fn block(&mut self, axis: u8) -> Sc {
        self.scalar(Scalar::Special(Special::Block(axis)))
    }

    pub fn warp(&mut self) -> Sc {
        self.scalar(Scalar::Special(Special::Warp))
    }

    /// `param[index]` of an integer parameter.
    pub fn load_scalar<T: Elem>(&mut self, param: ParamRef<T>, index: impl Into<Sc>) -> Sc {
        let index = self.intern(index);
        self.scalar(Scalar::Load { param: param.0, index })
    }

    // ---- views -----------------------------------------------------------

    fn value(&mut self, dtype: ScalarDType, shape: Shape, place: Place) -> ValId {
        self.prog.values.push(Value { dtype, shape, place });
        ValId(self.prog.values.len() as u32 - 1)
    }

    pub fn shape(&self, v: impl Into<ValId>) -> Shape {
        self.prog.value(v.into()).shape
    }

    /// A `shape` window at `offset` elements into `param`, rows `stride.0`
    /// apart and columns `stride.1` apart; `bounds` are its valid row/column
    /// counts: writes past them are dropped, register loads read zero, and a
    /// fill into shared memory clamps (see [`Place::Global`]).
    pub fn view<T: Elem>(
        &mut self,
        param: ParamRef<T>,
        offset: impl Into<Sc>,
        stride: [impl Into<Sc>; 2],
        shape: Shape,
        bounds: [Option<Sc>; 2],
    ) -> Gmem<T> {
        let offset = self.intern(offset);
        let stride = stride.map(|s| self.intern(s));
        let bounds = bounds.map(|b| b.map(|s| self.intern(s)));
        Tile::new(self.value(T::DTYPE, shape, Place::Global { param: param.0, offset, stride, bounds, rows: None }))
    }

    /// A `shape` window of contiguous columns whose row `r` starts at
    /// `offset + rows[Row := r]` (see [`Self::row`]) and reads zero wherever
    /// `valid[Row := r]` is zero, on every path; `col_bound` is a plain column
    /// bound.
    pub fn gather<T: Elem>(
        &mut self,
        param: ParamRef<T>,
        offset: impl Into<Sc>,
        rows: Sc,
        valid: Option<Sc>,
        shape: Shape,
        col_bound: Option<Sc>,
    ) -> Gmem<T> {
        let offset = self.intern(offset);
        let rows = RowMap { offset: self.intern(rows), valid: valid.map(|v| self.intern(v)) };
        let stride = [self.intern(0), self.intern(1)];
        let bounds = [None, col_bound.map(|b| self.intern(b))];
        Tile::new(self.value(
            T::DTYPE,
            shape,
            Place::Global { param: param.0, offset, stride, bounds, rows: Some(rows) },
        ))
    }

    /// The row coordinate a [`Self::gather`] map is evaluated at.
    pub fn row(&mut self) -> Sc {
        self.scalar(Scalar::Row)
    }

    /// The same window moved by `rows`/`cols` elements, keeping its bounds
    /// relative to the new origin. A gathered view moves along its columns only.
    pub fn at<T: Elem>(&mut self, view: Gmem<T>, rows: impl Into<Sc>, cols: impl Into<Sc>) -> Gmem<T> {
        let (rows, cols) = (rows.into(), cols.into());
        let Value { dtype, shape, place } = self.prog.value(view.0).clone();
        let Place::Global { param, offset, stride, bounds, rows: map } = place else { unreachable!("global tier") };
        assert!(map.is_none() || rows == Sc::Const(0), "a gathered view moves along its columns only");
        let moved = Sc::Id(offset) + rows.clone() * Sc::Id(stride[0]) + cols.clone() * Sc::Id(stride[1]);
        let offset = self.intern(moved);
        let bounds = [(bounds[0], rows), (bounds[1], cols)].map(|(b, by)| b.map(|b| self.intern(Sc::Id(b) - by)));
        Tile::new(self.value(dtype, shape, Place::Global { param, offset, stride, bounds, rows: map }))
    }

    pub fn smem_view<T: Elem>(&mut self, alloc: SmemId, offset: impl Into<Sc>, shape: Shape) -> Shared<T> {
        assert_eq!(self.prog.smem[alloc.index()].dtype, T::DTYPE, "shared allocation dtype");
        let offset = self.intern(offset);
        Tile::new(self.value(T::DTYPE, shape, Place::Smem { alloc, offset }))
    }

    /// Slot `slot` of a ring of `shape` tiles in `alloc`.
    pub fn smem_slot<T: Elem>(&mut self, alloc: SmemId, slot: impl Into<Sc>, shape: Shape) -> Shared<T> {
        self.smem_view(alloc, slot.into() * shape.elems(), shape)
    }

    // ---- tile ops --------------------------------------------------------

    fn push(&mut self, stmt: Stmt) {
        self.blocks.last_mut().expect("an open block").0.push(stmt);
    }

    fn let_<T: Elem>(&mut self, shape: Shape, op: TileOp) -> Regs<T> {
        let dst = self.value(T::DTYPE, shape, Place::Reg);
        self.push(Stmt::Let { dst, op });
        Tile::new(dst)
    }

    pub fn fill<T: Elem>(&mut self, shape: Shape, value: Const) -> Regs<T> {
        self.let_(shape, TileOp::Fill(value))
    }

    pub fn zeros<T: Elem>(&mut self, shape: Shape) -> Regs<T> {
        self.fill(shape, Const::Float(0.0))
    }

    pub fn splat<T: Elem>(&mut self, shape: Shape, value: impl Into<Sc>) -> Regs<T> {
        let value = self.intern(value);
        self.let_(shape, TileOp::Splat(value))
    }

    pub fn coord(&mut self, shape: Shape, axis: Axis) -> Regs<I32> {
        self.let_(shape, TileOp::Coord(axis))
    }

    pub fn unary<T: Elem>(&mut self, src: Regs<T>, f: UnaryOp) -> Regs<T> {
        let shape = self.shape(src);
        self.let_(shape, TileOp::Unary { src: src.0, f })
    }

    /// Elementwise `a f b`; `b` may be a row or column vector of `a`.
    pub fn binary<T: Elem>(&mut self, a: Regs<T>, b: Regs<T>, f: BinaryOp) -> Regs<T> {
        let shape = self.broadcast_shape(a.0, b.0);
        self.let_(shape, TileOp::Binary { a: a.0, b: b.0, f })
    }

    pub fn compare<T: Elem>(&mut self, a: Regs<T>, b: Regs<T>, f: BinaryOp) -> Regs<Bool> {
        assert!(matches!(f, BinaryOp::Lt | BinaryOp::Le | BinaryOp::Eq | BinaryOp::Ne), "a comparison");
        let shape = self.broadcast_shape(a.0, b.0);
        self.let_(shape, TileOp::Binary { a: a.0, b: b.0, f })
    }

    fn broadcast_shape(&self, a: ValId, b: ValId) -> Shape {
        let (sa, sb) = (self.shape(a), self.shape(b));
        let fits = |big: Shape, small: Shape| {
            small == big || (small.rows == big.rows && small.cols == 1) || (small.cols == big.cols && small.rows == 1)
        };
        if fits(sa, sb) {
            sa
        } else if fits(sb, sa) {
            sb
        } else {
            panic!("shapes {sa:?} and {sb:?} do not broadcast")
        }
    }

    pub fn cast<T: Elem, U: Elem>(&mut self, src: Regs<T>) -> Regs<U> {
        let shape = self.shape(src);
        self.let_(shape, TileOp::Cast { src: src.0, to: U::DTYPE })
    }

    /// `acc + A·B` where `A` is `[m, k]` (or `[k, m]` when `a_t`) and `B` is
    /// `[k, n]` (or `[n, k]` when `b_t`); operands may live in registers or
    /// shared memory.
    pub fn mma<T: Elem, PA: TierMark, PB: TierMark>(
        &mut self,
        acc: Regs<F32>,
        a: Tile<PA, T>,
        a_t: bool,
        b: Tile<PB, T>,
        b_t: bool,
    ) -> Regs<F32> {
        self.mma_oriented(acc, a, a_t, b, b_t, None)
    }

    /// [`Self::mma`] pinned to issue `orient` way round; `None` leaves the
    /// choice to the layout inference.
    pub fn mma_oriented<T: Elem, PA: TierMark, PB: TierMark>(
        &mut self,
        acc: Regs<F32>,
        a: Tile<PA, T>,
        a_t: bool,
        b: Tile<PB, T>,
        b_t: bool,
        orient: Option<Orient>,
    ) -> Regs<F32> {
        let (sacc, sa, sb) = (self.shape(acc), self.shape(a), self.shape(b));
        let (m, ka) = if a_t { (sa.cols, sa.rows) } else { (sa.rows, sa.cols) };
        let (kb, n) = if b_t { (sb.cols, sb.rows) } else { (sb.rows, sb.cols) };
        assert_eq!(ka, kb, "mma reduction dims");
        assert_eq!(sacc, Shape::new(m, n), "mma accumulator shape");
        self.let_(sacc, TileOp::Mma { acc: acc.0, a: a.0, b: b.0, a_t, b_t, orient })
    }

    pub fn reduce<T: Elem>(&mut self, src: Regs<T>, axis: Axis, f: ReduceOp) -> Regs<T> {
        let s = self.shape(src);
        let shape = match axis {
            Axis::Row => Shape::new(s.rows, 1),
            Axis::Col => Shape::new(1, s.cols),
        };
        self.let_(shape, TileOp::Reduce { src: src.0, axis, f })
    }

    pub fn where_<T: Elem>(&mut self, pred: Regs<Bool>, a: Regs<T>, b: Regs<T>) -> Regs<T> {
        let shape = self.broadcast_shape(a.0, b.0);
        assert_eq!(self.shape(pred), shape, "predicate shape");
        self.let_(shape, TileOp::Where { pred: pred.0, a: a.0, b: b.0 })
    }

    pub fn transpose<T: Elem>(&mut self, src: Regs<T>) -> Regs<T> {
        let shape = self.shape(src).transposed();
        self.let_(shape, TileOp::Transpose { src: src.0 })
    }

    // ---- movement --------------------------------------------------------

    fn copy(&mut self, dst: ValId, src: ValId, mode: CopyMode) {
        let (d, s) = (self.prog.value(dst), self.prog.value(src));
        assert_eq!(d.shape, s.shape, "copy shapes");
        assert_eq!(d.dtype, s.dtype, "copy dtypes");
        self.push(Stmt::Copy { dst, src, mode });
    }

    /// Global → shared; `Async` lets the lowering overlap it.
    pub fn stage<T: Elem>(&mut self, dst: Shared<T>, src: Gmem<T>, mode: CopyMode) {
        self.copy(dst.0, src.0, mode);
    }

    /// Shared or global → a fresh register tile.
    pub fn load<T: Elem, P: TierMark>(&mut self, src: Tile<P, T>) -> Regs<T> {
        let Value { dtype, shape, .. } = self.prog.value(src.0).clone();
        let dst = self.value(dtype, shape, Place::Reg);
        self.copy(dst, src.0, CopyMode::Sync);
        Tile::new(dst)
    }

    /// Registers → shared or global.
    pub fn store<T: Elem, P: TierMark>(&mut self, dst: Tile<P, T>, src: Regs<T>) {
        self.copy(dst.0, src.0, CopyMode::Sync);
    }

    // ---- control ---------------------------------------------------------

    fn carried<const N: usize>(&mut self, init: [ValId; N]) -> [ValId; N] {
        init.map(|i| {
            let Value { dtype, shape, .. } = self.prog.value(i).clone();
            self.value(dtype, shape, Place::Reg)
        })
    }

    fn with_block(&mut self, f: impl FnOnce(&mut Self)) -> Block {
        self.blocks.push(Block::default());
        f(self);
        self.blocks.pop().expect("the block just opened")
    }

    /// `for iv in 0..extent`, threading `init` through `body` as register
    /// tiles; returns the values after the last iteration.
    pub fn loop_<T: Elem, const N: usize>(
        &mut self,
        extent: impl Into<Sc>,
        init: [Regs<T>; N],
        body: impl FnOnce(&mut Self, Sc, [Regs<T>; N]) -> [Regs<T>; N],
    ) -> [Regs<T>; N] {
        let (start, extent) = (self.intern(0), self.intern(extent));
        let iv = self.push_scalar(Scalar::Induction);
        let phi = self.carried(init.map(ValId::from));
        let mut next = [ValId(0); N];
        let block = self.with_block(|k| next = body(k, Sc::Id(iv), phi.map(Tile::new)).map(ValId::from));
        let carried = (0..N).map(|i| Carried { init: init[i].0, phi: phi[i], next: next[i] }).collect();
        self.push(Stmt::Loop(Loop { iv, start, extent, carried, body: block, unroll: 1 }));
        phi.map(Tile::new)
    }

    /// A producer/consumer loop over `extent` steps with `stages` slots.
    /// `produce(k, step, slot)` stages step `step` into slot `slot`;
    /// `consume(k, step, slot, carried)` computes from it.
    pub fn pipeline<T: Elem, const N: usize>(
        &mut self,
        extent: impl Into<Sc>,
        stages: usize,
        init: [Regs<T>; N],
        produce: impl FnOnce(&mut Self, Sc, Sc),
        consume: impl FnOnce(&mut Self, Sc, Sc, [Regs<T>; N]) -> [Regs<T>; N],
    ) -> [Regs<T>; N] {
        assert!(stages >= 1, "at least one stage");
        let extent = self.intern(extent);
        let [pstep, pslot, cstep, cslot] = [(); 4].map(|()| self.push_scalar(Scalar::Induction));
        let phi = self.carried(init.map(ValId::from));
        let body = self.with_block(|k| produce(k, Sc::Id(pstep), Sc::Id(pslot)));
        let produce = Stage { step: pstep, slot: pslot, body };
        let mut next = [ValId(0); N];
        let body =
            self.with_block(|k| next = consume(k, Sc::Id(cstep), Sc::Id(cslot), phi.map(Tile::new)).map(ValId::from));
        let consume = Stage { step: cstep, slot: cslot, body };
        let carried = (0..N).map(|i| Carried { init: init[i].0, phi: phi[i], next: next[i] }).collect();
        self.push(Stmt::Pipeline(Pipeline { extent, stages, carried, produce, consume }));
        phi.map(Tile::new)
    }

    pub fn role_block(&mut self, role: RoleId, body: impl FnOnce(&mut Self)) {
        let body = self.with_block(body);
        self.push(Stmt::Role { role, body });
    }

    pub fn if_(&mut self, pred: impl Into<Sc>, then: impl FnOnce(&mut Self), otherwise: impl FnOnce(&mut Self)) {
        let pred = self.intern(pred);
        let then = self.with_block(then);
        let otherwise = self.with_block(otherwise);
        self.push(Stmt::If { pred, then, otherwise });
    }

    /// `if pred { then } else { otherwise }` producing `N` register tiles:
    /// each branch computes its values and the merged tiles are returned.
    pub fn select_if<T: Elem, const N: usize>(
        &mut self,
        pred: impl Into<Sc>,
        then: impl FnOnce(&mut Self) -> [Regs<T>; N],
        otherwise: impl FnOnce(&mut Self) -> [Regs<T>; N],
    ) -> [Regs<T>; N] {
        let pred = self.intern(pred);
        let mut merged: Option<[ValId; N]> = None;
        type Branch<'a, T, const N: usize> = Box<dyn FnOnce(&mut Kernel) -> [Regs<T>; N] + 'a>;
        let mut branch = |k: &mut Self, f: Branch<'_, T, N>| {
            k.with_block(|k| {
                let vals = f(k);
                let out = *merged.get_or_insert_with(|| {
                    vals.map(|v| {
                        let Value { dtype, shape, .. } = k.prog.value(v.0).clone();
                        k.value(dtype, shape, Place::Reg)
                    })
                });
                for (dst, v) in out.into_iter().zip(vals) {
                    assert_eq!(k.shape(dst), k.shape(v), "both branches produce the same shapes");
                    k.push(Stmt::Let { dst, op: TileOp::Move { src: v.0 } });
                }
            })
        };
        let then = branch(self, Box::new(then));
        let otherwise = branch(self, Box::new(otherwise));
        self.push(Stmt::If { pred, then, otherwise });
        merged.expect("both branches ran").map(Tile::new)
    }

    pub fn raw(&mut self, raw: Raw) {
        self.push(Stmt::Raw(raw));
    }

    pub fn finish(mut self) -> Program {
        assert_eq!(self.blocks.len(), 1, "every block closed");
        self.prog.body = self.blocks.pop().expect("the root block");
        self.prog
    }
}
