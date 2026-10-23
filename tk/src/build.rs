//! The recording builder: every call appends a statement to the open block
//! and hands back a tier-typed handle. Tiles carry their tier and element type
//! in the Rust type; shapes are checked when the statement is recorded.
//! Scalars are ordinary expressions with operators, interned when consumed.

use std::marker::PhantomData;
use std::rc::Rc;

use svod_dtype::ScalarDType;

use crate::ir::*;
pub use crate::layouts::WarpGrid;

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
elem!(BF16 = BFloat16, F16 = Float16, I8 = Int8, F32 = Float32, I32 = Int32, Bool = Bool);

/// An element a matrix core multiplies, and the element it accumulates into:
/// f32 for 16-bit floats, i32 for 8-bit integers (exact, as their products
/// and sums are).
pub trait MmaIn: Elem {
    type Acc: Elem;

    /// An accumulator's values in f32: as held, or converted from i32.
    fn widened(k: &mut Kernel, acc: Regs<Self::Acc>) -> Regs<F32>;
}

/// A 16-bit float operand, accumulated in f32.
pub trait Half: MmaIn<Acc = F32> {}

macro_rules! half {
    ($($name:ident),*) => {$(
        impl MmaIn for $name {
            type Acc = F32;

            fn widened(_: &mut Kernel, acc: Regs<F32>) -> Regs<F32> {
                acc
            }
        }
        impl Half for $name {}
    )*};
}
half!(BF16, F16);

impl MmaIn for I8 {
    type Acc = I32;

    fn widened(k: &mut Kernel, acc: Regs<I32>) -> Regs<F32> {
        k.cast::<I32, F32>(acc)
    }
}

pub trait TierMark: Copy {}
macro_rules! tier {
    ($($name:ident),*) => {$(
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub struct $name;
        impl TierMark for $name {}
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

/// A shared allocation of `T` elements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SmemRef<T>(pub SmemId, PhantomData<T>);

/// An elementwise comparison ([`Kernel::compare`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CmpOp {
    Lt,
    Le,
    Eq,
    Ne,
}

impl From<CmpOp> for BinaryOp {
    fn from(op: CmpOp) -> Self {
        match op {
            CmpOp::Lt => BinaryOp::Lt,
            CmpOp::Le => BinaryOp::Le,
            CmpOp::Eq => BinaryOp::Eq,
            CmpOp::Ne => BinaryOp::Ne,
        }
    }
}

/// How a [`Kernel::pipeline`] runs its ring: `stages` slots, the loop
/// copied once per slot or not, over one crew or two ([`Pipeline`]); a
/// count of slots is a rolled ring of one crew.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ring {
    pub stages: usize,
    pub unroll: bool,
    pub crews: u32,
}

impl From<usize> for Ring {
    fn from(stages: usize) -> Self {
        Self { stages, unroll: false, crews: 1 }
    }
}

pub struct Kernel {
    prog: Program,
    blocks: Vec<Block>,
    /// Whose lanes the statements being recorded belong to.
    scope: Scope,
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
                warp_grid: WarpGrid { rows: 4, cols: 1 },
                smem: vec![],
                scalars: vec![],
                values: vec![],
                body: Block::default(),
                pins: vec![],
            },
            blocks: vec![Block::default()],
            scope: Scope::Block,
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

    /// A symbolic variable in `min..=max` the launch binds by name; asked
    /// for again, the same variable over the same range.
    pub fn var(&mut self, name: impl Into<String>, min: i64, max: i64) -> Sc {
        let name = name.into();
        match self.prog.vars.iter().find(|v| v.name == name) {
            Some(v) => assert!(v.min == min && v.max == max, "variable {name} is in {}..={}", v.min, v.max),
            None => self.prog.vars.push(Var { name: name.clone(), min, max }),
        }
        self.scalar(Scalar::Var(name))
    }

    pub fn grid(&mut self, dims: [Sc; 3]) {
        self.prog.grid = dims.map(|s| self.intern(s));
    }

    /// The block's warps and how a register tile spreads over them; a count
    /// is a grid of one column.
    pub fn warps(&mut self, grid: impl Into<WarpGrid>) {
        let grid = grid.into();
        self.prog.warp_grid = grid;
        self.prog.warps = grid.count();
    }

    /// `warps` warps a block, whole copies of the grid: the ones past it run
    /// only inside [`Self::each_warp`].
    pub fn lone_warps(&mut self, warps: u32) {
        assert!(warps.is_multiple_of(self.prog.warp_grid.count()), "whole copies of the warp grid");
        self.prog.warps = warps;
    }

    pub fn smem<T: Elem>(&mut self, name: impl Into<String>, elems: usize) -> SmemRef<T> {
        self.prog.smem.push(SmemAlloc { name: name.into(), dtype: T::DTYPE, elems, along: Axis::Col });
        SmemRef(SmemId(self.prog.smem.len() as u32 - 1), PhantomData)
    }

    // ---- scalars ---------------------------------------------------------

    fn push_scalar(&mut self, s: Scalar) -> ScalarId {
        self.prog.scalars.push(s);
        ScalarId(self.prog.scalars.len() as u32 - 1)
    }

    fn scalar(&mut self, s: Scalar) -> Sc {
        Sc::Id(self.push_scalar(s))
    }

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

    /// `param[index]` of an integer input: no store of the kernel changes it,
    /// so the lowering reads it once, wherever its index is known.
    pub fn load_scalar<T: Elem>(&mut self, param: ParamRef<T>, index: impl Into<Sc>) -> Sc {
        assert_eq!(self.prog.params[param.0.index()].kind, ParamKind::In, "a scalar load reads an input");
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

    /// The first `shape` rows and columns of `view`.
    pub fn narrow<T: Elem>(&mut self, view: Gmem<T>, shape: Shape) -> Gmem<T> {
        let Value { dtype, shape: whole, place } = self.prog.value(view.0).clone();
        assert!(shape.rows <= whole.rows && shape.cols <= whole.cols, "a window within the view");
        Tile::new(self.value(dtype, shape, place))
    }

    pub fn smem_view<T: Elem>(&mut self, alloc: SmemRef<T>, offset: impl Into<Sc>, shape: Shape) -> Shared<T> {
        self.smem_view_as(alloc, offset, shape)
    }

    /// `shape` of `T` at `offset` (in `T` elements) of `alloc`, read under
    /// `T` whatever the allocation's type: scratch over a region whose tiles
    /// are dead, as a key warp's merge state reuses its ring. The allocation
    /// holds a whole number of `T`s; a `T` of the allocation's width or wider
    /// keeps every element of the view aligned.
    pub fn smem_view_as<T: Elem>(
        &mut self,
        alloc: SmemRef<impl Elem>,
        offset: impl Into<Sc>,
        shape: Shape,
    ) -> Shared<T> {
        let alloc = alloc.0;
        let a = &self.prog.smem[alloc.index()];
        let (held, bytes) = (a.elems * a.dtype.bytes(), T::DTYPE.bytes());
        assert!(held.is_multiple_of(bytes) && bytes >= a.dtype.bytes(), "{} read as {:?}", a.name, T::DTYPE);
        let offset = self.intern(offset);
        Tile::new(self.value(T::DTYPE, shape, Place::Smem { alloc, offset }))
    }

    /// Slot `slot` of a ring of `shape` tiles in `alloc`.
    pub fn smem_slot<T: Elem>(&mut self, alloc: SmemRef<T>, slot: impl Into<Sc>, shape: Shape) -> Shared<T> {
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

    /// `a · factor` (see [`BinaryOp::Scale`]).
    pub fn scale<T: Elem>(&mut self, a: Regs<T>, factor: Regs<T>) -> Regs<T> {
        self.binary(a, factor, BinaryOp::Scale)
    }

    pub fn compare<T: Elem>(&mut self, a: Regs<T>, b: Regs<T>, f: CmpOp) -> Regs<Bool> {
        let shape = self.broadcast_shape(a.0, b.0);
        self.let_(shape, TileOp::Binary { a: a.0, b: b.0, f: f.into() })
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
        self.let_(shape, TileOp::Cast { src: src.0, to: U::DTYPE, round: Round::Nearest })
    }

    /// [`Self::cast`] rounding toward zero: f32 to bf16 only.
    pub fn cast_toward_zero<U: Elem>(&mut self, src: Regs<F32>) -> Regs<U> {
        assert_eq!(U::DTYPE, ScalarDType::BFloat16, "a truncating cast is f32 to bf16");
        let shape = self.shape(src);
        self.let_(shape, TileOp::Cast { src: src.0, to: U::DTYPE, round: Round::Zero })
    }

    /// `acc + A·B` where `A` is `[m, k]` (or `[k, m]` when `a_t`) and `B` is
    /// `[k, n]` (or `[n, k]` when `b_t`); operands may live in registers or
    /// shared memory, the accumulator in `T`'s type ([`MmaIn::Acc`]).
    pub fn mma<T: MmaIn, PA: TierMark, PB: TierMark>(
        &mut self,
        acc: Regs<T::Acc>,
        a: Tile<PA, T>,
        a_t: bool,
        b: Tile<PB, T>,
        b_t: bool,
    ) -> Regs<T::Acc> {
        self.mma_oriented(acc, a, a_t, b, b_t, None)
    }

    /// [`Self::mma`] pinned to issue `orient` way round; `None` leaves the
    /// choice to the layout inference.
    pub fn mma_oriented<T: MmaIn, PA: TierMark, PB: TierMark>(
        &mut self,
        acc: Regs<T::Acc>,
        a: Tile<PA, T>,
        a_t: bool,
        b: Tile<PB, T>,
        b_t: bool,
        orient: Option<Orient>,
    ) -> Regs<T::Acc> {
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

    /// A reduction of a 16-bit tile folded in f32: the values as held, summed
    /// without the rounding a 16-bit fold would add.
    pub fn reduce_f32<T: Elem>(&mut self, src: Regs<T>, axis: Axis, f: ReduceOp) -> Regs<F32> {
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

    /// `a·b + c`, rounded once ([`TileOp::MulAdd`]).
    pub fn mul_add<T: Elem>(&mut self, a: Regs<T>, b: Regs<T>, c: Regs<T>) -> Regs<T> {
        let shape = self.broadcast_shape(a.0, b.0);
        assert_eq!(self.shape(c), shape, "addend shape");
        self.let_(shape, TileOp::MulAdd { a: a.0, b: b.0, c: c.0 })
    }

    pub fn transpose<T: Elem>(&mut self, src: Regs<T>) -> Regs<T> {
        let shape = self.shape(src).transposed();
        self.let_(shape, TileOp::Transpose { src: src.0 })
    }

    /// `src` one step on along `axis`, `fill` first ([`TileOp::Shift`]).
    pub fn shift<T: Elem>(&mut self, src: Regs<T>, axis: Axis, fill: Const) -> Regs<T> {
        let shape = self.shape(src);
        self.let_(shape, TileOp::Shift { src: src.0, axis, fill })
    }

    /// `src` with every element exchanged across the bits of `mask` along
    /// `axis` ([`TileOp::Flip`]).
    pub fn flip<T: Elem>(&mut self, src: Regs<T>, axis: Axis, mask: u32) -> Regs<T> {
        let shape = self.shape(src);
        let len = if axis == Axis::Col { shape.cols } else { shape.rows };
        let span = (mask as usize + 1).next_power_of_two();
        assert!(mask > 0 && len.is_multiple_of(span), "a flip across {mask:#b} of {len}");
        self.let_(shape, TileOp::Flip { src: src.0, axis, mask })
    }

    /// The `j`th of the columns each lane holds of `src`'s rows, `w` a row
    /// ([`TileOp::Deal`]): over `j < cols / w`, every column of a row once.
    pub fn deal<T: Elem>(&mut self, src: Regs<T>, w: usize, j: usize) -> Regs<T> {
        self.dealt(src, w, j, false)
    }

    /// The columns of `src` that [`Self::deal`] of the same `w` and `j` takes.
    pub fn deal_col<T: Elem>(&mut self, src: Regs<T>, w: usize, j: usize) -> Regs<I32> {
        self.dealt(src, w, j, true)
    }

    fn dealt<T: Elem, U: Elem>(&mut self, src: Regs<T>, w: usize, j: usize, col: bool) -> Regs<U> {
        let shape = self.shape(src);
        assert!(w > 0 && shape.cols.is_multiple_of(w) && j < shape.cols / w, "deal {j} of {shape:?} by {w}");
        self.let_(Shape::new(shape.rows, w), TileOp::Deal { src: src.0, j: j as u32, col })
    }

    /// Hold `v` under `layout` (see [`Program::pins`]).
    pub fn pin<T: Elem>(&mut self, v: Regs<T>, layout: crate::layouts::TileLayout) {
        assert_eq!(layout.shape(), self.shape(v), "a pin covers its tile");
        self.prog.pins.push((v.0, layout));
    }

    // ---- movement --------------------------------------------------------

    fn copy(&mut self, dst: ValId, src: ValId, mode: CopyMode) {
        let (d, s) = (self.prog.value(dst), self.prog.value(src));
        assert_eq!(d.shape, s.shape, "copy shapes");
        assert_eq!(d.dtype, s.dtype, "copy dtypes");
        if let Place::Global { param, .. } = d.place {
            assert_ne!(self.prog.params[param.index()].kind, ParamKind::In, "a store into an input");
        }
        self.push(Stmt::Copy { dst, src, mode, scope: self.scope });
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

    /// Global → registers as the register half of the target's fill loads:
    /// the same instruction class and cache policy as a fill of `src`
    /// ([`CopyMode::Staged`]), where [`Self::load`] is a plain load.
    pub fn load_as_fill<T: Elem>(&mut self, src: Gmem<T>) -> Regs<T> {
        let Value { dtype, shape, .. } = self.prog.value(src.0).clone();
        let dst = self.value(dtype, shape, Place::Reg);
        self.copy(dst, src.0, CopyMode::Staged);
        Tile::new(dst)
    }

    /// Registers → shared or global.
    pub fn store<T: Elem, P: TierMark>(&mut self, dst: Tile<P, T>, src: Regs<T>) {
        self.copy(dst.0, src.0, CopyMode::Sync);
    }

    // ---- control ---------------------------------------------------------

    fn carried<T>(&mut self, init: &[Regs<T>]) -> Vec<ValId> {
        init.iter()
            .map(|i| {
                let Value { dtype, shape, .. } = self.prog.value(i.0).clone();
                self.value(dtype, shape, Place::Reg)
            })
            .collect()
    }

    /// A carried value whose next is defined outside its body (another
    /// carried value, as a ring rotating its slots, or a value from before
    /// the loop) takes a copy of it made where the body starts: the moves
    /// into the carried values then read what the iteration began with,
    /// as the interpreter's simultaneous assignment does, and a next the
    /// body never defines still moves.
    fn snapshot_carried<T: Elem>(&mut self, body: &mut Block, phi: &[ValId], next: &mut [Regs<T>]) {
        let mut defined = std::collections::HashSet::new();
        for (_, s) in body.walk() {
            match s {
                Stmt::Let { dst, .. } | Stmt::Copy { dst, .. } => {
                    defined.insert(*dst);
                }
                Stmt::If { results, .. } => defined.extend(results.iter().map(|&(dst, ..)| dst)),
                _ => {}
            }
        }
        let mut copies = vec![];
        for (n, &p) in next.iter_mut().zip(phi) {
            if n.0 == p || defined.contains(&n.0) {
                continue;
            }
            let Value { dtype, shape, .. } = self.prog.value(n.0).clone();
            let dst = self.value(dtype, shape, Place::Reg);
            copies.push(Stmt::Let { dst, op: TileOp::Move { src: n.0 } });
            *n = Tile::new(dst);
        }
        body.0.splice(0..0, copies);
    }

    /// Carried values may share a `next` (each takes it), but no two of a
    /// loop's `next`s are one value: every one past the first reads a copy
    /// made at the body's end, where the moves into the `phi`s read, so
    /// their order decides nothing ([`check_carried`]).
    fn deshare_carried<T: Elem>(&mut self, body: &mut Block, next: &mut [Regs<T>]) {
        let mut seen = std::collections::HashSet::new();
        for n in next.iter_mut() {
            if seen.insert(n.0) {
                continue;
            }
            let Value { dtype, shape, .. } = self.prog.value(n.0).clone();
            let dst = self.value(dtype, shape, Place::Reg);
            body.0.push(Stmt::Let { dst, op: TileOp::Move { src: n.0 } });
            *n = Tile::new(dst);
        }
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
        let out = self.loop_vec(extent, init.to_vec(), |k, iv, phi| body(k, iv, tiles(phi)).to_vec());
        tiles(out)
    }

    /// [`Self::loop_`] over as many tiles as `init` holds.
    pub fn loop_vec<T: Elem>(
        &mut self,
        extent: impl Into<Sc>,
        init: Vec<Regs<T>>,
        body: impl FnOnce(&mut Self, Sc, Vec<Regs<T>>) -> Vec<Regs<T>>,
    ) -> Vec<Regs<T>> {
        let (start, extent) = (self.intern(0), self.intern(extent));
        let iv = self.push_scalar(Scalar::Induction);
        let phi = self.carried(&init);
        let mut next = vec![];
        let mut block = self.with_block(|k| next = body(k, Sc::Id(iv), phi.iter().map(|&v| Tile::new(v)).collect()));
        self.snapshot_carried(&mut block, &phi, &mut next);
        self.deshare_carried(&mut block, &mut next);
        let carried = carry(&init, &phi, &next);
        self.push(Stmt::Loop(Loop { iv, start, extent, carried, body: block, unroll: 1 }));
        phi.into_iter().map(Tile::new).collect()
    }

    /// A producer/consumer loop over `extent` steps with `stages` slots.
    /// `produce(k, step, slot)` stages step `step` into slot `slot`;
    /// `consume(k, step, slot, carried)` computes from it.
    pub fn pipeline<T: Elem, const N: usize>(
        &mut self,
        extent: impl Into<Sc>,
        ring: impl Into<Ring>,
        init: [Regs<T>; N],
        produce: impl FnOnce(&mut Self, Sc, Sc),
        consume: impl FnOnce(&mut Self, Sc, Sc, [Regs<T>; N]) -> [Regs<T>; N],
    ) -> [Regs<T>; N] {
        let consume = |k: &mut Self, step, slot, phi: Vec<Regs<T>>| consume(k, step, slot, tiles(phi)).to_vec();
        tiles(self.pipeline_vec(extent, ring, init.to_vec(), produce, consume))
    }

    /// [`Self::pipeline`] over as many tiles as `init` holds.
    pub fn pipeline_vec<T: Elem>(
        &mut self,
        extent: impl Into<Sc>,
        ring: impl Into<Ring>,
        init: Vec<Regs<T>>,
        produce: impl FnOnce(&mut Self, Sc, Sc),
        consume: impl FnOnce(&mut Self, Sc, Sc, Vec<Regs<T>>) -> Vec<Regs<T>>,
    ) -> Vec<Regs<T>> {
        let Ring { stages, unroll, crews } = ring.into();
        assert!(stages >= 1, "at least one stage");
        assert!(crews == 1 || (crews == 2 && stages == 1), "two crews run over one slot");
        let extent = self.intern(extent);
        let [pstep, pslot, cstep, cslot] = [(); 4].map(|()| self.push_scalar(Scalar::Induction));
        let phi = self.carried(&init);
        let body = self.with_block(|k| produce(k, Sc::Id(pstep), Sc::Id(pslot)));
        let produce = Stage { step: pstep, slot: pslot, body };
        let mut next = vec![];
        let mut body = self.with_block(|k| {
            next = consume(k, Sc::Id(cstep), Sc::Id(cslot), phi.iter().map(|&v| Tile::new(v)).collect())
        });
        self.snapshot_carried(&mut body, &phi, &mut next);
        self.deshare_carried(&mut body, &mut next);
        let consume = Stage { step: cstep, slot: cslot, body };
        let carried = carry(&init, &phi, &next);
        self.push(Stmt::Pipeline(Pipeline { extent, stages, unroll, crews, carried, produce, consume }));
        phi.into_iter().map(Tile::new).collect()
    }

    /// Every warp runs `body` on its own ([`Stmt::Warps`]): [`warp`](Self::warp)
    /// is its index there, and what it leaves in shared memory is visible to
    /// the block afterwards.
    pub fn each_warp(&mut self, body: impl FnOnce(&mut Self)) {
        let outer = std::mem::replace(&mut self.scope, Scope::Warp);
        let body = self.with_block(body);
        self.scope = outer;
        self.push(Stmt::Warps { body });
    }

    pub fn if_(&mut self, pred: impl Into<Sc>, then: impl FnOnce(&mut Self), otherwise: impl FnOnce(&mut Self)) {
        let pred = self.intern(pred);
        let then = self.with_block(then);
        let otherwise = self.with_block(otherwise);
        self.push(Stmt::If { pred, then, otherwise, results: vec![] });
    }

    /// `if pred { then } else { otherwise }` producing `N` register tiles:
    /// each branch computes its values and the merged tiles are returned.
    pub fn select_if<T: Elem, const N: usize>(
        &mut self,
        pred: impl Into<Sc>,
        then: impl FnOnce(&mut Self) -> [Regs<T>; N],
        otherwise: impl FnOnce(&mut Self) -> [Regs<T>; N],
    ) -> [Regs<T>; N] {
        tiles(self.select_if_vec(pred, |k| then(k).to_vec(), |k| otherwise(k).to_vec()))
    }

    /// [`Self::select_if`] over as many tiles as the branches produce.
    pub fn select_if_vec<T: Elem>(
        &mut self,
        pred: impl Into<Sc>,
        then: impl FnOnce(&mut Self) -> Vec<Regs<T>>,
        otherwise: impl FnOnce(&mut Self) -> Vec<Regs<T>>,
    ) -> Vec<Regs<T>> {
        let pred = self.intern(pred);
        let mut tv = vec![];
        let then = self.with_block(|k| tv = then(k).into_iter().map(|v| v.0).collect());
        let mut ov = vec![];
        let otherwise = self.with_block(|k| ov = otherwise(k).into_iter().map(|v| v.0).collect());
        assert_eq!(tv.len(), ov.len(), "both branches produce as many tiles");
        // One merge per tile, on the `if` itself: each branch's value stays
        // its own, and the merged tile is defined once, past the statement.
        let results = tv
            .into_iter()
            .zip(ov)
            .map(|(t, o)| {
                let Value { dtype, shape, .. } = self.prog.value(t).clone();
                let dst = self.value(dtype, shape, Place::Reg);
                assert_eq!(self.shape(dst), self.shape(o), "both branches produce the same shapes");
                (dst, t, Some(o))
            })
            .collect::<Vec<_>>();
        self.push(Stmt::If { pred, then, otherwise, results: results.clone() });
        results.into_iter().map(|(dst, ..)| Tile::new(dst)).collect()
    }

    /// `where_(pred, then(k), otherwise)` for each pair of tiles, `then` run
    /// only by the warps holding some set element of `pred` ([`Scalar::Any`]):
    /// work most warps skip where `pred` is mostly clear. `then` must hold
    /// no block barrier.
    pub fn where_any<T: Elem, const N: usize>(
        &mut self,
        pred: Regs<Bool>,
        then: impl FnOnce(&mut Self) -> [Regs<T>; N],
        otherwise: [Regs<T>; N],
    ) -> [Regs<T>; N] {
        let any = self.scalar(Scalar::Any(pred.0));
        self.select_if(
            any,
            |k| {
                let mut taken = then(k).into_iter().zip(otherwise);
                [(); N].map(|()| {
                    let (t, o) = taken.next().expect("a tile per output");
                    k.where_(pred, t, o)
                })
            },
            |_| otherwise,
        )
    }

    /// `then(k)`, run only by the warps holding some set element of `pred`
    /// ([`Scalar::Any`]), else `otherwise`; unlike [`Self::where_any`] nothing
    /// selects between them, so `then` must itself leave the program's results
    /// what `otherwise` leads to wherever `pred` is clear (a tile it changes
    /// there may only feed work that changes nothing there). `then` must hold
    /// no block barrier.
    ///
    /// The results hold `otherwise` before the branch (one move each), which
    /// `then`'s replace as the branch's merge: no path past it leaves them
    /// unset, so a loop around it carries none of them from its last trip.
    pub fn if_any<T: Elem>(
        &mut self,
        pred: Regs<Bool>,
        then: impl FnOnce(&mut Self) -> Vec<Regs<T>>,
        otherwise: Vec<Regs<T>>,
    ) -> Vec<Regs<T>> {
        let any = self.scalar(Scalar::Any(pred.0));
        let held: Vec<ValId> = otherwise
            .iter()
            .map(|v| {
                let Value { dtype, shape, .. } = self.prog.value(v.0).clone();
                let dst = self.value(dtype, shape, Place::Reg);
                self.push(Stmt::Let { dst, op: TileOp::Move { src: v.0 } });
                dst
            })
            .collect();
        let mut vals = vec![];
        let then = self.with_block(|k| vals = then(k).into_iter().map(|v| v.0).collect());
        assert_eq!(vals.len(), held.len(), "both branches produce as many tiles");
        let results = held
            .clone()
            .into_iter()
            .zip(vals)
            .map(|(dst, t)| {
                assert_eq!(self.shape(dst), self.shape(t), "both branches produce the same shapes");
                (dst, t, None)
            })
            .collect();
        let pred = self.intern(any);
        self.push(Stmt::If { pred, then, otherwise: Block::default(), results });
        held.into_iter().map(Tile::new).collect()
    }

    /// A barrier across the block's warps, and a fence of its shared memory;
    /// block-wide even inside [`Self::each_warp`].
    pub fn barrier(&mut self) {
        self.push(Stmt::Sync(Sync::Barrier { scope: Scope::Block }));
    }

    pub fn finish(mut self) -> Program {
        assert_eq!(self.blocks.len(), 1, "every block closed");
        self.prog.body = self.blocks.pop().expect("the root block");
        check_carried(&self.prog.body);
        self.prog
    }
}

/// Every loop's and pipeline's `next`s are statements of its own body (an
/// `if`'s merge among them, never a branch of one) and pairwise distinct:
/// the moves into the `phi`s all run at that body's end, which is where the
/// interpreter's simultaneous assignment leaves them, so the device and the
/// interpreter agree wherever this holds ([`Kernel::loop_`],
/// [`Kernel::pipeline`]).
fn check_carried(block: &Block) {
    for stmt in &block.0 {
        let (body, carried) = match stmt {
            Stmt::Loop(l) => (&l.body, &l.carried),
            Stmt::Pipeline(p) => (&p.consume.body, &p.carried),
            _ => {
                stmt.blocks().for_each(check_carried);
                continue;
            }
        };
        let mut defined = std::collections::HashSet::new();
        for s in &body.0 {
            match s {
                Stmt::Let { dst, .. } | Stmt::Copy { dst, .. } => {
                    defined.insert(*dst);
                }
                Stmt::If { results, .. } => defined.extend(results.iter().map(|&(dst, ..)| dst)),
                _ => {}
            }
        }
        let mut nexts = std::collections::HashSet::new();
        for c in carried {
            if c.next == c.phi {
                continue;
            }
            assert!(defined.contains(&c.next), "a carried next is a statement of its loop's body");
            assert!(nexts.insert(c.next), "a loop's nexts are distinct values");
        }
        stmt.blocks().for_each(check_carried);
    }
}

/// `N` tiles of a list as long.
fn tiles<T, const N: usize>(list: Vec<Regs<T>>) -> [Regs<T>; N] {
    list.try_into().unwrap_or_else(|l: Vec<_>| panic!("{} tiles where {N} are carried", l.len()))
}

/// Each `init`'s carried register: its `phi` and its `next`.
fn carry<T>(init: &[Regs<T>], phi: &[ValId], next: &[Regs<T>]) -> Vec<Carried> {
    assert_eq!(next.len(), init.len(), "a body returns a tile per carried one");
    (0..init.len()).map(|i| Carried { init: init[i].0, phi: phi[i], next: next[i].0 }).collect()
}
