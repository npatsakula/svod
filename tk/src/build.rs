//! The recording builder: every call appends a statement to the open block
//! and hands back a tier-typed handle. Tiles carry their tier and element type
//! in the Rust type; shapes are checked when the statement is recorded.
//! Scalars are ordinary expressions with operators, interned when consumed.

use std::marker::PhantomData;
use std::ops::Deref;
use std::rc::Rc;

use svod_dtype::ScalarDType;

use crate::ir::*;
pub use crate::layouts::WarpGrid;

/// The brand of a kernel's handles: invariant in `'k`, which only
/// [`Kernel::record`] names, so a handle of one kernel fits no other.
type Brand<'k> = fn(&'k ()) -> &'k ();

/// An element type: a marker, so a tile of it is `'static` but for its brand.
pub trait Elem: Copy + 'static {
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
    fn widened<'k>(k: &mut Kernel<'k>, acc: Acc<'k, Self::Acc>) -> Regs<'k, F32>;
}

/// A 16-bit float operand, accumulated in f32.
pub trait Half: MmaIn<Acc = F32> {}

macro_rules! half {
    ($($name:ident),*) => {$(
        impl MmaIn for $name {
            type Acc = F32;

            fn widened<'k>(_: &mut Kernel<'k>, acc: Acc<'k, F32>) -> Regs<'k, F32> {
                *acc
            }
        }
        impl Half for $name {}
    )*};
}
half!(BF16, F16);

impl MmaIn for I8 {
    type Acc = I32;

    fn widened<'k>(k: &mut Kernel<'k>, acc: Acc<'k, I32>) -> Regs<'k, F32> {
        k.cast::<I32, F32>(*acc)
    }
}

/// How a kernel touches a parameter: reads it, writes it, or both.
pub trait Access: Copy {
    const KIND: ParamKind;
}
macro_rules! access {
    ($($name:ident),*) => {$(
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub struct $name;
        impl Access for $name {
            const KIND: ParamKind = ParamKind::$name;
        }
    )*};
}
access!(In, Out, InOut);

pub trait TierMark: Copy {}
/// A tier a register tile is loaded from.
pub trait Readable: TierMark {}
/// A tier a register tile is stored to.
pub trait Writable: TierMark {}
/// A window into a parameter, which the author moves ([`Kernel::at`]).
pub trait Window: TierMark {}

/// A view of a parameter the kernel accesses as `A`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Global<A: Access>(PhantomData<A>);
impl<A: Access> TierMark for Global<A> {}
impl<A: Access> Window for Global<A> {}
impl Readable for Global<In> {}
impl Readable for Global<InOut> {}
impl Writable for Global<Out> {}
impl Writable for Global<InOut> {}

macro_rules! tier {
    ($($name:ident),*) => {$(
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub struct $name;
        impl TierMark for $name {}
    )*};
}
tier!(Smem, Reg, Period);
impl Readable for Reg {}
impl Readable for Smem {}
impl Writable for Smem {}
impl Readable for Period {}
impl Window for Period {}

/// A ring slot as a consume reads it, branded by its stage: read, never
/// written, and never kept past the stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Held<'r>(PhantomData<Brand<'r>>);
/// A ring slot as a produce fills it, branded by its stage: a fill lands
/// in it and registers store to it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Filled<'r>(PhantomData<Brand<'r>>);
impl TierMark for Held<'_> {}
impl Readable for Held<'_> {}
impl TierMark for Filled<'_> {}
impl Readable for Filled<'_> {}
impl Writable for Filled<'_> {}
/// A tier in shared memory, which a fill lands in ([`Kernel::stage`]).
pub trait SharedTier: Writable {}
impl SharedTier for Smem {}
impl SharedTier for Filled<'_> {}

/// A tile handle of kernel `'k`; `P` is its tier marker and `T` its
/// element type.
pub struct Tile<'k, P, T>(pub ValId, PhantomData<(P, T, Brand<'k>)>);

impl<P, T> Clone for Tile<'_, P, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<P, T> Copy for Tile<'_, P, T> {}
impl<P, T> std::fmt::Debug for Tile<'_, P, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tile({})", self.0.0)
    }
}
impl<P, T> From<Tile<'_, P, T>> for ValId {
    fn from(t: Tile<'_, P, T>) -> Self {
        t.0
    }
}
impl<P, T> Tile<'_, P, T> {
    fn new(id: ValId) -> Self {
        Self(id, PhantomData)
    }
}

pub type Gmem<'k, T, A = In> = Tile<'k, Global<A>, T>;
pub type Shared<'k, T> = Tile<'k, Smem, T>;
pub type Regs<'k, T> = Tile<'k, Reg, T>;
/// A global view whose columns repeat ([`Kernel::periodic`]): read into
/// registers only.
pub type Periodic<'k, T> = Tile<'k, Period, T>;
/// A ring slot as a pipeline's consume of stage `'r` sees it.
pub type Slotted<'k, 'r, T> = Tile<'k, Held<'r>, T>;

/// A register tile a loop carries or a product accumulates into: owned, not
/// copied, so what consumes it ([`Kernel::mma`], the next a loop body
/// returns) retires it, and a stale one is a compile error where a register
/// tile would be read or carried on by mistake. `*acc` is the tile it holds
/// now, for the operands of other ops; any register tile becomes one with
/// `into`. The `_vec` forms of the loops carry plain register tiles.
pub struct Acc<'k, T>(pub Regs<'k, T>);

impl<'k, T> From<Regs<'k, T>> for Acc<'k, T> {
    fn from(v: Regs<'k, T>) -> Self {
        Acc(v)
    }
}
impl<'k, T> Deref for Acc<'k, T> {
    type Target = Regs<'k, T>;
    fn deref(&self) -> &Regs<'k, T> {
        &self.0
    }
}
impl<T> std::fmt::Debug for Acc<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Acc({})", self.0.0.0)
    }
}

/// A scalar expression: built with ordinary operators, interned into the
/// program's arena by the builder call that consumes it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Sc<'k> {
    Id(ScalarId, PhantomData<Brand<'k>>),
    Const(i64),
    Bin(BinOp, Rc<Sc<'k>>, Rc<Sc<'k>>),
}

impl<'k> Sc<'k> {
    pub(crate) fn id(id: ScalarId) -> Self {
        Sc::Id(id, PhantomData)
    }
    fn bin(self, op: BinOp, rhs: impl Into<Sc<'k>>) -> Sc<'k> {
        Sc::Bin(op, Rc::new(self), Rc::new(rhs.into()))
    }
    pub fn min(self, rhs: impl Into<Sc<'k>>) -> Sc<'k> {
        self.bin(BinOp::Min, rhs)
    }
    pub fn max(self, rhs: impl Into<Sc<'k>>) -> Sc<'k> {
        self.bin(BinOp::Max, rhs)
    }
    pub fn lt(self, rhs: impl Into<Sc<'k>>) -> Sc<'k> {
        self.bin(BinOp::Lt, rhs)
    }
    pub fn le(self, rhs: impl Into<Sc<'k>>) -> Sc<'k> {
        self.bin(BinOp::Le, rhs)
    }
    pub fn eq(self, rhs: impl Into<Sc<'k>>) -> Sc<'k> {
        self.bin(BinOp::Eq, rhs)
    }
    pub fn and(self, rhs: impl Into<Sc<'k>>) -> Sc<'k> {
        self.bin(BinOp::And, rhs)
    }
    pub fn or(self, rhs: impl Into<Sc<'k>>) -> Sc<'k> {
        self.bin(BinOp::Or, rhs)
    }
}

macro_rules! sc_from {
    ($($t:ty),*) => {$(impl From<$t> for Sc<'_> {
        fn from(v: $t) -> Self {
            Sc::Const(v as i64)
        }
    })*};
}
sc_from!(i64, i32, usize, u32);

macro_rules! sc_ops {
    ($($trait:ident $method:ident $op:ident),*) => {$(
        impl<'k, R: Into<Sc<'k>>> std::ops::$trait<R> for Sc<'k> {
            type Output = Sc<'k>;
            fn $method(self, rhs: R) -> Sc<'k> {
                self.bin(BinOp::$op, rhs)
            }
        }
    )*};
}
sc_ops!(Add add Add, Sub sub Sub, Mul mul Mul, Div div Div, Rem rem Rem);

/// A parameter of `T` elements the kernel accesses as `A`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParamRef<'k, T, A: Access>(pub ParamId, PhantomData<(T, A, Brand<'k>)>);

/// A shared allocation of `T` elements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SmemRef<'k, T>(pub SmemId, PhantomData<(T, Brand<'k>)>);

/// A shared allocation of slots of `slot` elements that only a pipeline's
/// stages reach ([`Kernel::slot`]): its produce fills a slot, its consume
/// reads one. Anything else over it is [`Kernel::ring_scratch`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ring<'k, T> {
    alloc: SmemRef<'k, T>,
    slot: usize,
}

/// The slot a pipeline stage holds, branded by the stage so it cannot leave
/// it: a produce's [`Filling`] or a consume's [`Reading`].
pub trait Hold<'k>: Clone {
    /// What a view of the slot is: writable while filled, read-only while
    /// read, branded by the stage either way.
    type Tier: TierMark;
    fn index(&self) -> Sc<'k>;
}

/// The slot a produce of stage `'r` fills.
#[derive(Clone, Debug)]
pub struct Filling<'k, 'r>(Sc<'k>, PhantomData<Brand<'r>>);
/// The slot a consume of stage `'r` reads.
#[derive(Clone, Debug)]
pub struct Reading<'k, 'r>(Sc<'k>, PhantomData<Brand<'r>>);

macro_rules! hold {
    ($($name:ident => $tier:ident),*) => {$(
        impl<'k, 'r> Hold<'k> for $name<'k, 'r> {
            type Tier = $tier<'r>;
            fn index(&self) -> Sc<'k> {
                self.0.clone()
            }
        }
        impl<'k> $name<'k, '_> {
            fn new(slot: ScalarId) -> Self {
                Self(Sc::id(slot), PhantomData)
            }
            /// The stage's slot past the first `first`, where the ring holds a
            /// run of slots per warp.
            pub fn past(self, first: impl Into<Sc<'k>>) -> Self {
                Self(first.into() + self.0, PhantomData)
            }
        }
    )*};
}
hold!(Filling => Filled, Reading => Held);

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

/// The recording builder of one kernel, branded `'k`: its handles fit it
/// alone ([`Brand`]).
pub struct Kernel<'k> {
    prog: Program,
    blocks: Vec<Block>,
    /// Whose lanes the statements being recorded belong to.
    scope: Scope,
    brand: PhantomData<Brand<'k>>,
}

impl<'k> Kernel<'k> {
    /// Record a kernel: `f` builds it on a kernel of a brand no other has,
    /// so its handles and scalars are its own.
    ///
    /// ```compile_fail
    /// use svod_tk::build::*;
    /// use svod_tk::ir::{BinaryOp, Shape};
    /// let a = Kernel::record_with("a", |k| k.zeros::<F32>(Shape::new(4, 4))).1;
    /// Kernel::record("b", |k| {
    ///     k.binary(a, a, BinaryOp::Add); // a tile of another kernel
    /// });
    /// ```
    ///
    /// ```compile_fail
    /// use svod_tk::build::*;
    /// let n = Kernel::record_with("a", |k| k.var("n", 1, 4)).1;
    /// Kernel::record("b", |k| {
    ///     k.grid([n, Sc::from(1), Sc::from(1)]); // a scalar of another kernel
    /// });
    /// ```
    pub fn record(name: impl Into<String>, f: impl for<'a> FnOnce(&mut Kernel<'a>)) -> Program {
        Self::record_with(name, f).0
    }

    /// [`Self::record`], with what `f` returns beside the program.
    pub fn record_with<R>(name: impl Into<String>, f: impl for<'a> FnOnce(&mut Kernel<'a>) -> R) -> (Program, R) {
        let mut k = Kernel::new(name);
        let r = f(&mut k);
        (k.finish(), r)
    }

    fn new(name: impl Into<String>) -> Self {
        let mut k = Self {
            prog: Program {
                name: name.into(),
                params: vec![],
                vars: vec![],
                index: IndexWidth::I32,
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
            brand: PhantomData,
        };
        let one = k.push_scalar(Scalar::Const(1));
        k.prog.grid = [one; 3];
        k
    }

    // ---- program-level declarations -------------------------------------

    fn param<T: Elem, A: Access>(&mut self, name: impl Into<String>, elems: usize) -> ParamRef<'k, T, A> {
        self.prog.params.push(Param { name: name.into(), dtype: T::DTYPE, kind: A::KIND, elems });
        ParamRef(ParamId(self.prog.params.len() as u32 - 1), PhantomData)
    }

    /// A parameter of `elems` elements the kernel reads.
    pub fn input<T: Elem>(&mut self, name: impl Into<String>, elems: usize) -> ParamRef<'k, T, In> {
        self.param(name, elems)
    }

    /// A parameter of `elems` elements the kernel writes and never reads.
    pub fn output<T: Elem>(&mut self, name: impl Into<String>, elems: usize) -> ParamRef<'k, T, Out> {
        self.param(name, elems)
    }

    /// A parameter of `elems` elements the kernel writes and reads back.
    /// Nothing orders a load after a store to it but each lane's own program
    /// order: no fence is emitted, so a read-back is correct only where the
    /// lanes that read an element are the lanes that stored it, which is the
    /// case for the same tile shape under the same layout.
    pub fn inout<T: Elem>(&mut self, name: impl Into<String>, elems: usize) -> ParamRef<'k, T, InOut> {
        self.param(name, elems)
    }

    /// A symbolic variable in `min..=max` the launch binds by name, counting
    /// in `i32` as every range does; asked for again, the same variable over
    /// the same range.
    pub fn var(&mut self, name: impl Into<String>, min: i64, max: i64) -> Sc<'k> {
        let name = name.into();
        assert!(max <= i64::from(i32::MAX), "variable {name} counts to {max}, past i32");
        match self.prog.vars.iter().find(|v| v.name == name) {
            Some(v) => assert!(v.min == min && v.max == max, "variable {name} is in {}..={}", v.min, v.max),
            None => self.prog.vars.push(Var { name: name.clone(), min, max }),
        }
        self.scalar(Scalar::Var(name))
    }

    pub fn grid(&mut self, dims: [Sc<'k>; 3]) {
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

    pub fn smem<T: Elem>(&mut self, name: impl Into<String>, elems: usize) -> SmemRef<'k, T> {
        self.prog.smem.push(SmemAlloc { name: name.into(), dtype: T::DTYPE, elems, halves: None, along: Axis::Col });
        SmemRef(SmemId(self.prog.smem.len() as u32 - 1), PhantomData)
    }

    // ---- scalars ---------------------------------------------------------

    fn push_scalar(&mut self, s: Scalar) -> ScalarId {
        self.prog.scalars.push(s);
        ScalarId(self.prog.scalars.len() as u32 - 1)
    }

    fn scalar(&mut self, s: Scalar) -> Sc<'k> {
        Sc::id(self.push_scalar(s))
    }

    pub fn intern(&mut self, sc: impl Into<Sc<'k>>) -> ScalarId {
        match sc.into() {
            Sc::Id(id, _) => id,
            Sc::Const(v) => self.push_scalar(Scalar::Const(v)),
            Sc::Bin(op, a, b) => {
                let (a, b) = (self.intern((*a).clone()), self.intern((*b).clone()));
                self.push_scalar(Scalar::Bin(op, a, b))
            }
        }
    }

    pub fn block(&mut self, axis: u8) -> Sc<'k> {
        self.scalar(Scalar::Special(Special::Block(axis)))
    }

    pub fn warp(&mut self) -> Sc<'k> {
        self.scalar(Scalar::Special(Special::Warp))
    }

    /// `param[index]` of an integer input: no store of the kernel changes it,
    /// so the lowering reads it once, wherever its index is known.
    ///
    /// ```compile_fail
    /// use svod_tk::build::*;
    /// Kernel::record("read", |k| {
    ///     let p = k.inout::<I32>("p", 4);
    ///     k.load_scalar(p, 0); // a parameter the kernel writes is not `In`
    /// });
    /// ```
    pub fn load_scalar<T: Elem>(&mut self, param: ParamRef<'k, T, In>, index: impl Into<Sc<'k>>) -> Sc<'k> {
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
    pub fn view<T: Elem, A: Access>(
        &mut self,
        param: ParamRef<'k, T, A>,
        offset: impl Into<Sc<'k>>,
        stride: [impl Into<Sc<'k>>; 2],
        shape: Shape,
        bounds: [Option<Sc<'k>>; 2],
    ) -> Gmem<'k, T, A> {
        let offset = self.intern(offset);
        let stride = stride.map(|s| self.intern(s));
        let bounds = bounds.map(|b| b.map(|s| self.intern(s)));
        let place = Place::Global { param: param.0, offset, stride, bounds, rows: None, period: None };
        Tile::new(self.value(T::DTYPE, shape, place))
    }

    /// A `shape` window of contiguous columns whose row `r` starts at
    /// `offset + rows[Row := r]` (see [`Self::row`]) and reads zero wherever
    /// `valid[Row := r]` is zero, on every path; `col_bound` is a plain column
    /// bound.
    pub fn gather<T: Elem, A: Access>(
        &mut self,
        param: ParamRef<'k, T, A>,
        offset: impl Into<Sc<'k>>,
        rows: Sc<'k>,
        valid: Option<Sc<'k>>,
        shape: Shape,
        col_bound: Option<Sc<'k>>,
    ) -> Gmem<'k, T, A> {
        let offset = self.intern(offset);
        let rows = RowMap { offset: self.intern(rows), valid: valid.map(|v| self.intern(v)) };
        let stride = [self.intern(0), self.intern(1)];
        let bounds = [None, col_bound.map(|b| self.intern(b))];
        Tile::new(self.value(
            T::DTYPE,
            shape,
            Place::Global { param: param.0, offset, stride, bounds, rows: Some(rows), period: None },
        ))
    }

    /// `view` with its columns repeating every `period`: column `c` reads
    /// column `c % period`, a table every block of `period` columns shares.
    /// Its contiguous columns hold the period, and it is read into
    /// registers only.
    ///
    /// ```compile_fail
    /// use svod_tk::build::*;
    /// use svod_tk::ir::{CopyMode, Shape};
    /// Kernel::record("table", |k| {
    ///     let p = k.input::<F32>("p", 16);
    ///     let v = k.view(p, 0, [4, 1], Shape::new(1, 16), [None, None]);
    ///     let table = k.periodic(v, 4);
    ///     let s = k.smem::<F32>("s", 16);
    ///     let t = k.smem_view(s, 0, Shape::new(1, 16));
    ///     k.stage(t, table, CopyMode::Sync); // a periodic view is not staged
    /// });
    /// ```
    pub fn periodic<T: Elem, A: Access>(&mut self, view: Gmem<'k, T, A>, period: usize) -> Periodic<'k, T>
    where
        Global<A>: Readable,
    {
        let Value { dtype, shape, place } = self.prog.value(view.0).clone();
        let Place::Global { param, offset, stride, bounds, rows, period: None } = place else {
            unreachable!("a global view of one period")
        };
        let unit = matches!(self.prog.scalar(stride[1]), Scalar::Const(1));
        assert!(period > 0 && unit && bounds[1].is_none(), "a periodic view has unbounded contiguous columns");
        let place = Place::Global { param, offset, stride, bounds, rows, period: Some(period) };
        Tile::new(self.value(dtype, shape, place))
    }

    /// The row coordinate a [`Self::gather`] map is evaluated at.
    pub fn row(&mut self) -> Sc<'k> {
        self.scalar(Scalar::Row)
    }

    /// The same window moved by `rows`/`cols` elements, keeping its bounds
    /// relative to the new origin. A gathered view moves along its columns only.
    pub fn at<T: Elem, P: Window>(
        &mut self,
        view: Tile<'k, P, T>,
        rows: impl Into<Sc<'k>>,
        cols: impl Into<Sc<'k>>,
    ) -> Tile<'k, P, T> {
        let (rows, cols) = (rows.into(), cols.into());
        let Value { dtype, shape, place } = self.prog.value(view.0).clone();
        let Place::Global { param, offset, stride, bounds, rows: map, period } = place else {
            unreachable!("global tier")
        };
        assert!(map.is_none() || rows == Sc::Const(0), "a gathered view moves along its columns only");
        assert!(period.is_none() || cols == Sc::Const(0), "a periodic view moves along its rows only");
        let moved = Sc::id(offset) + rows.clone() * Sc::id(stride[0]) + cols.clone() * Sc::id(stride[1]);
        let offset = self.intern(moved);
        let bounds = [(bounds[0], rows), (bounds[1], cols)].map(|(b, by)| b.map(|b| self.intern(Sc::id(b) - by)));
        Tile::new(self.value(dtype, shape, Place::Global { param, offset, stride, bounds, rows: map, period }))
    }

    /// The first `shape` rows and columns of `view`.
    pub fn narrow<T: Elem, P: Window>(&mut self, view: Tile<'k, P, T>, shape: Shape) -> Tile<'k, P, T> {
        let Value { dtype, shape: whole, place } = self.prog.value(view.0).clone();
        assert!(shape.rows <= whole.rows && shape.cols <= whole.cols, "a window within the view");
        Tile::new(self.value(dtype, shape, place))
    }

    pub fn smem_view<T: Elem>(
        &mut self,
        alloc: SmemRef<'k, T>,
        offset: impl Into<Sc<'k>>,
        shape: Shape,
    ) -> Shared<'k, T> {
        self.view_as(alloc, offset, shape)
    }

    /// `shape` of `T` at `offset` (in `T` elements) of `alloc`, read under
    /// `T` whatever the allocation's type. The allocation holds a whole number
    /// of `T`s; a `T` of the allocation's width or wider keeps every element
    /// of the view aligned.
    fn view_as<T: Elem, P>(
        &mut self,
        alloc: SmemRef<'k, impl Elem>,
        offset: impl Into<Sc<'k>>,
        shape: Shape,
    ) -> Tile<'k, P, T> {
        let alloc = alloc.0;
        let a = &self.prog.smem[alloc.index()];
        let (held, bytes) = (a.elems * a.dtype.bytes(), T::DTYPE.bytes());
        assert!(held.is_multiple_of(bytes) && bytes >= a.dtype.bytes(), "{} read as {:?}", a.name, T::DTYPE);
        let offset = self.intern(offset);
        Tile::new(self.value(T::DTYPE, shape, Place::Smem { alloc, offset }))
    }

    /// Slot `slot` of a ring of `shape` tiles in `alloc`.
    pub fn smem_slot<T: Elem>(
        &mut self,
        alloc: SmemRef<'k, T>,
        slot: impl Into<Sc<'k>>,
        shape: Shape,
    ) -> Shared<'k, T> {
        self.smem_view(alloc, slot.into() * shape.elems(), shape)
    }

    /// A ring of `slots` slots of `slot` elements, for a pipeline's stages.
    pub fn ring<T: Elem>(&mut self, name: impl Into<String>, slot: usize, slots: usize) -> Ring<'k, T> {
        Ring { alloc: self.smem(name, slots * slot), slot }
    }

    /// The stage's slot of `ring` as one `shape` tile: written by a produce,
    /// read by a consume.
    ///
    /// ```compile_fail
    /// use svod_tk::build::*;
    /// use svod_tk::ir::{CopyMode, Shape};
    /// Kernel::record("ring", |k| {
    ///     let p = k.input::<F32>("p", 16);
    ///     let v = k.view(p, 0, [4, 1], Shape::new(4, 4), [None, None]);
    ///     let ring = k.ring::<F32>("ring", 16, 2);
    ///     let zero = k.zeros::<F32>(Shape::new(4, 4));
    ///     k.pipeline(1, 2, [zero], |k, _, slot| {
    ///         let t = k.slot(ring, &slot, Shape::new(4, 4));
    ///         k.stage(t, v, CopyMode::Async);
    ///     }, |k, _, slot, [acc]| {
    ///         let t = k.slot(ring, &slot, Shape::new(4, 4));
    ///         k.store(t, acc); // a consume reads its slot, never writes it
    ///         [acc]
    ///     });
    /// });
    /// ```
    ///
    /// ```compile_fail
    /// use svod_tk::build::*;
    /// use svod_tk::ir::Shape;
    /// Kernel::record("ring", |k| {
    ///     let ring = k.ring::<F32>("ring", 16, 2);
    ///     let zero = k.zeros::<F32>(Shape::new(4, 4));
    ///     let mut kept = None;
    ///     k.pipeline(1, 2, [zero], |_, _, slot| kept = Some(slot), |_, _, _, c| c);
    ///     k.slot(ring, &kept.unwrap(), Shape::new(4, 4)); // a slot stays in its stage
    /// });
    /// ```
    ///
    /// ```compile_fail
    /// use svod_tk::build::*;
    /// use svod_tk::ir::Shape;
    /// Kernel::record("ring", |k| {
    ///     let ring = k.ring::<F32>("ring", 16, 2);
    ///     let zero = k.zeros::<F32>(Shape::new(4, 4));
    ///     let mut kept = None;
    ///     k.pipeline(1, 2, [zero], |_, _, _| {}, |k, _, slot, c| {
    ///         kept = Some(k.slot(ring, &slot, Shape::new(4, 4)));
    ///         c
    ///     });
    ///     k.load(kept.unwrap()); // a slot's tile stays in its stage too
    /// });
    /// ```
    pub fn slot<T: Elem, H: Hold<'k>>(&mut self, ring: Ring<'k, T>, slot: &H, shape: Shape) -> Tile<'k, H::Tier, T> {
        assert_eq!(shape.elems(), ring.slot, "a whole slot");
        self.view_as(ring.alloc, slot.index() * shape.elems(), shape)
    }

    /// Part `part` of the stage's slot of `ring`, the slot cut into `shape`
    /// tiles.
    pub fn slot_part<T: Elem, H: Hold<'k>>(
        &mut self,
        ring: Ring<'k, T>,
        slot: &H,
        part: usize,
        shape: Shape,
    ) -> Tile<'k, H::Tier, T> {
        assert!((part + 1) * shape.elems() <= ring.slot, "a part within the slot");
        self.view_as(ring.alloc, slot.index() * ring.slot + part * shape.elems(), shape)
    }

    /// `shape` of `T` at `offset` (in `T` elements) of `ring`, whatever the
    /// ring's element type, as scratch shared by the block.
    ///
    /// # Safety
    ///
    /// No pipeline stage reaches the slots the view covers while it is used:
    /// their pipeline has ended (as a key warp's merge state reuses its own
    /// run of slots) or not begun. The lowering fences the view against the
    /// block's other accesses; it cannot tell a live slot from a dead one.
    pub unsafe fn ring_scratch<T: Elem>(
        &mut self,
        ring: Ring<'k, impl Elem>,
        offset: impl Into<Sc<'k>>,
        shape: Shape,
    ) -> Shared<'k, T> {
        self.view_as(ring.alloc, offset, shape)
    }

    // ---- tile ops --------------------------------------------------------

    fn push(&mut self, stmt: Stmt) {
        self.blocks.last_mut().expect("an open block").0.push(stmt);
    }

    fn let_<T: Elem>(&mut self, shape: Shape, op: TileOp) -> Regs<'k, T> {
        let dst = self.value(T::DTYPE, shape, Place::Reg);
        self.push(Stmt::Let { dst, op });
        Tile::new(dst)
    }

    pub fn fill<T: Elem>(&mut self, shape: Shape, value: Const) -> Regs<'k, T> {
        self.let_(shape, TileOp::Fill(value))
    }

    pub fn zeros<T: Elem>(&mut self, shape: Shape) -> Regs<'k, T> {
        self.fill(shape, Const::Float(0.0))
    }

    pub fn splat<T: Elem>(&mut self, shape: Shape, value: impl Into<Sc<'k>>) -> Regs<'k, T> {
        let value = self.intern(value);
        self.let_(shape, TileOp::Splat(value))
    }

    pub fn coord(&mut self, shape: Shape, axis: Axis) -> Regs<'k, I32> {
        self.let_(shape, TileOp::Coord(axis))
    }

    pub fn unary<T: Elem>(&mut self, src: Regs<'k, T>, f: UnaryOp) -> Regs<'k, T> {
        let shape = self.shape(src);
        self.let_(shape, TileOp::Unary { src: src.0, f })
    }

    /// Elementwise `a f b`; `b` may be a row or column vector of `a`.
    pub fn binary<T: Elem>(&mut self, a: Regs<'k, T>, b: Regs<'k, T>, f: BinaryOp) -> Regs<'k, T> {
        let shape = self.broadcast_shape(a.0, b.0);
        self.let_(shape, TileOp::Binary { a: a.0, b: b.0, f })
    }

    /// `a · factor` (see [`BinaryOp::Scale`]).
    pub fn scale<T: Elem>(&mut self, a: Regs<'k, T>, factor: Regs<'k, T>) -> Regs<'k, T> {
        self.binary(a, factor, BinaryOp::Scale)
    }

    pub fn compare<T: Elem>(&mut self, a: Regs<'k, T>, b: Regs<'k, T>, f: CmpOp) -> Regs<'k, Bool> {
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

    pub fn cast<T: Elem, U: Elem>(&mut self, src: Regs<'k, T>) -> Regs<'k, U> {
        let shape = self.shape(src);
        self.let_(shape, TileOp::Cast { src: src.0, to: U::DTYPE, round: Round::Nearest })
    }

    /// [`Self::cast`] rounding toward zero: f32 to bf16 only.
    pub fn cast_toward_zero<U: Elem>(&mut self, src: Regs<'k, F32>) -> Regs<'k, U> {
        assert_eq!(U::DTYPE, ScalarDType::BFloat16, "a truncating cast is f32 to bf16");
        let shape = self.shape(src);
        self.let_(shape, TileOp::Cast { src: src.0, to: U::DTYPE, round: Round::Zero })
    }

    /// `acc + A·B` where `A` is `[m, k]` (or `[k, m]` when `a_t`) and `B` is
    /// `[k, n]` (or `[n, k]` when `b_t`); operands may live in registers or
    /// shared memory, the accumulator in `T`'s type ([`MmaIn::Acc`]). The
    /// product consumes `acc` and is the accumulator from then on.
    ///
    /// ```compile_fail
    /// use svod_tk::build::*;
    /// use svod_tk::ir::{BinaryOp, Shape};
    /// Kernel::record("mma", |k| {
    ///     let p = k.input::<BF16>("p", 256);
    ///     let x = k.view(p, 0, [16, 1], Shape::new(16, 16), [None, None]);
    ///     let x = k.load(x);
    ///     let acc: Acc<'_, F32> = k.zeros::<F32>(Shape::new(16, 16)).into();
    ///     let next = k.mma(acc, x, false, x, true);
    ///     k.binary(*acc, *next, BinaryOp::Add); // `acc` is consumed: `next` holds the sum
    /// });
    /// ```
    pub fn mma<T: MmaIn, PA: Readable, PB: Readable>(
        &mut self,
        acc: impl Into<Acc<'k, T::Acc>>,
        a: Tile<'k, PA, T>,
        a_t: bool,
        b: Tile<'k, PB, T>,
        b_t: bool,
    ) -> Acc<'k, T::Acc> {
        self.mma_oriented(acc, a, a_t, b, b_t, None)
    }

    /// [`Self::mma`] pinned to issue `orient` way round; `None` leaves the
    /// choice to the layout inference.
    pub fn mma_oriented<T: MmaIn, PA: Readable, PB: Readable>(
        &mut self,
        acc: impl Into<Acc<'k, T::Acc>>,
        a: Tile<'k, PA, T>,
        a_t: bool,
        b: Tile<'k, PB, T>,
        b_t: bool,
        orient: Option<Orient>,
    ) -> Acc<'k, T::Acc> {
        let acc = acc.into();
        let (sacc, sa, sb) = (self.shape(*acc), self.shape(a), self.shape(b));
        let (m, ka) = if a_t { (sa.cols, sa.rows) } else { (sa.rows, sa.cols) };
        let (kb, n) = if b_t { (sb.cols, sb.rows) } else { (sb.rows, sb.cols) };
        assert_eq!(ka, kb, "mma reduction dims");
        assert_eq!(sacc, Shape::new(m, n), "mma accumulator shape");
        Acc(self.let_(sacc, TileOp::Mma { acc: acc.0.0, a: a.0, b: b.0, a_t, b_t, orient }))
    }

    pub fn reduce<T: Elem>(&mut self, src: Regs<'k, T>, axis: Axis, f: ReduceOp) -> Regs<'k, T> {
        let s = self.shape(src);
        let shape = match axis {
            Axis::Row => Shape::new(s.rows, 1),
            Axis::Col => Shape::new(1, s.cols),
        };
        self.let_(shape, TileOp::Reduce { src: src.0, axis, f })
    }

    /// A reduction of a 16-bit tile folded in f32: the values as held, summed
    /// without the rounding a 16-bit fold would add.
    pub fn reduce_f32<T: Elem>(&mut self, src: Regs<'k, T>, axis: Axis, f: ReduceOp) -> Regs<'k, F32> {
        let s = self.shape(src);
        let shape = match axis {
            Axis::Row => Shape::new(s.rows, 1),
            Axis::Col => Shape::new(1, s.cols),
        };
        self.let_(shape, TileOp::Reduce { src: src.0, axis, f })
    }

    pub fn where_<T: Elem>(&mut self, pred: Regs<'k, Bool>, a: Regs<'k, T>, b: Regs<'k, T>) -> Regs<'k, T> {
        let shape = self.broadcast_shape(a.0, b.0);
        assert_eq!(self.shape(pred), shape, "predicate shape");
        self.let_(shape, TileOp::Where { pred: pred.0, a: a.0, b: b.0 })
    }

    /// `a·b + c`, rounded once ([`TileOp::MulAdd`]).
    pub fn mul_add<T: Elem>(&mut self, a: Regs<'k, T>, b: Regs<'k, T>, c: Regs<'k, T>) -> Regs<'k, T> {
        let shape = self.broadcast_shape(a.0, b.0);
        assert_eq!(self.shape(c), shape, "addend shape");
        self.let_(shape, TileOp::MulAdd { a: a.0, b: b.0, c: c.0 })
    }

    pub fn transpose<T: Elem>(&mut self, src: Regs<'k, T>) -> Regs<'k, T> {
        let shape = self.shape(src).transposed();
        self.let_(shape, TileOp::Transpose { src: src.0 })
    }

    /// `src` one step on along `axis`, `fill` first ([`TileOp::Shift`]).
    pub fn shift<T: Elem>(&mut self, src: Regs<'k, T>, axis: Axis, fill: Const) -> Regs<'k, T> {
        let shape = self.shape(src);
        self.let_(shape, TileOp::Shift { src: src.0, axis, fill })
    }

    /// `src` with every element exchanged across the bits of `mask` along
    /// `axis` ([`TileOp::Flip`]).
    pub fn flip<T: Elem>(&mut self, src: Regs<'k, T>, axis: Axis, mask: u32) -> Regs<'k, T> {
        let shape = self.shape(src);
        let len = if axis == Axis::Col { shape.cols } else { shape.rows };
        let span = (mask as usize + 1).next_power_of_two();
        assert!(mask > 0 && len.is_multiple_of(span), "a flip across {mask:#b} of {len}");
        self.let_(shape, TileOp::Flip { src: src.0, axis, mask })
    }

    /// The `j`th of the columns each lane holds of `src`'s rows, `w` a row
    /// ([`TileOp::Deal`]): over `j < cols / w`, every column of a row once.
    pub fn deal<T: Elem>(&mut self, src: Regs<'k, T>, w: usize, j: usize) -> Regs<'k, T> {
        self.dealt(src, w, j, false)
    }

    /// The columns of `src` that [`Self::deal`] of the same `w` and `j` takes.
    pub fn deal_col<T: Elem>(&mut self, src: Regs<'k, T>, w: usize, j: usize) -> Regs<'k, I32> {
        self.dealt(src, w, j, true)
    }

    fn dealt<T: Elem, U: Elem>(&mut self, src: Regs<'k, T>, w: usize, j: usize, col: bool) -> Regs<'k, U> {
        let shape = self.shape(src);
        assert!(w > 0 && shape.cols.is_multiple_of(w) && j < shape.cols / w, "deal {j} of {shape:?} by {w}");
        self.let_(Shape::new(shape.rows, w), TileOp::Deal { src: src.0, j: j as u32, col, order: None })
    }

    /// Hold `v` under `layout` (see [`Program::pins`]).
    pub fn pin<T: Elem>(&mut self, v: Regs<'k, T>, layout: crate::layouts::TileLayout) {
        assert_eq!(layout.shape(), self.shape(v), "a pin covers its tile");
        self.prog.pins.push((v.0, layout));
    }

    // ---- movement --------------------------------------------------------

    fn copy(&mut self, dst: ValId, src: ValId, mode: CopyMode) {
        assert_eq!(self.prog.value(dst).shape, self.prog.value(src).shape, "copy shapes");
        self.push(Stmt::Copy { dst, src, mode, scope: self.scope });
    }

    /// Global → shared; `Async` lets the lowering overlap it.
    ///
    /// ```compile_fail
    /// use svod_tk::build::*;
    /// use svod_tk::ir::{CopyMode, Shape};
    /// Kernel::record("stage", |k| {
    ///     let p = k.output::<F32>("p", 16);
    ///     let v = k.view(p, 0, [4, 1], Shape::new(4, 4), [None, None]);
    ///     let s = k.smem::<F32>("s", 16);
    ///     let t = k.smem_view(s, 0, Shape::new(4, 4));
    ///     k.stage(t, v, CopyMode::Async); // an output is not `Readable`
    /// });
    /// ```
    pub fn stage<T: Elem, P: SharedTier, A: Access>(&mut self, dst: Tile<'k, P, T>, src: Gmem<'k, T, A>, mode: CopyMode)
    where
        Global<A>: Readable,
    {
        self.copy(dst.0, src.0, mode);
    }

    /// Shared or global → a fresh register tile.
    ///
    /// ```compile_fail
    /// use svod_tk::build::*;
    /// use svod_tk::ir::Shape;
    /// Kernel::record("load", |k| {
    ///     let p = k.output::<F32>("p", 16);
    ///     let v = k.view(p, 0, [4, 1], Shape::new(4, 4), [None, None]);
    ///     k.load(v); // an output is not `Readable`
    /// });
    /// ```
    pub fn load<T: Elem, P: Readable>(&mut self, src: Tile<'k, P, T>) -> Regs<'k, T> {
        let Value { dtype, shape, .. } = self.prog.value(src.0).clone();
        let dst = self.value(dtype, shape, Place::Reg);
        self.copy(dst, src.0, CopyMode::Sync);
        Tile::new(dst)
    }

    /// Global → registers as the register half of the target's fill loads:
    /// the same instruction class and cache policy as a fill of `src`
    /// ([`CopyMode::Staged`]), where [`Self::load`] is a plain load.
    pub fn load_as_fill<T: Elem, A: Access>(&mut self, src: Gmem<'k, T, A>) -> Regs<'k, T>
    where
        Global<A>: Readable,
    {
        let Value { dtype, shape, .. } = self.prog.value(src.0).clone();
        let dst = self.value(dtype, shape, Place::Reg);
        self.copy(dst, src.0, CopyMode::Staged);
        Tile::new(dst)
    }

    /// Registers → shared or global.
    ///
    /// ```compile_fail
    /// use svod_tk::build::*;
    /// use svod_tk::ir::Shape;
    /// Kernel::record("store", |k| {
    ///     let p = k.input::<I32>("p", 4);
    ///     let v = k.splat::<I32>(Shape::new(1, 4), 1);
    ///     let out = k.view(p, 0, [4, 1], Shape::new(1, 4), [None, None]);
    ///     k.store(out, v); // an input is not `Writable`
    /// });
    /// ```
    pub fn store<T: Elem, P: Writable>(&mut self, dst: Tile<'k, P, T>, src: Regs<'k, T>) {
        self.copy(dst.0, src.0, CopyMode::Sync);
    }

    // ---- control ---------------------------------------------------------

    fn carried<T>(&mut self, init: &[Regs<'k, T>]) -> Vec<ValId> {
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
    fn snapshot_carried<T: Elem>(&mut self, body: &mut Block, phi: &[ValId], next: &mut [Regs<'k, T>]) {
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
    fn deshare_carried<T: Elem>(&mut self, body: &mut Block, next: &mut [Regs<'k, T>]) {
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

    /// `for iv in 0..extent`, threading `init` through `body` as owned
    /// accumulators ([`Acc`]): the body returns each one's next, and the
    /// values after the last iteration come back.
    ///
    /// ```compile_fail
    /// use svod_tk::build::*;
    /// use svod_tk::ir::Shape;
    /// Kernel::record("stale", |k| {
    ///     let p = k.input::<BF16>("p", 256);
    ///     let x = k.view(p, 0, [16, 1], Shape::new(16, 16), [None, None]);
    ///     let x = k.load(x);
    ///     let zero = k.zeros::<F32>(Shape::new(16, 16));
    ///     k.loop_(4, [zero], |k, _, [acc]| {
    ///         let _sum = k.mma(acc, x, false, x, true);
    ///         [acc] // the accumulator before the product, which `_sum` retired
    ///     });
    /// });
    /// ```
    pub fn loop_<T: Elem, I: Into<Acc<'k, T>>, const N: usize>(
        &mut self,
        extent: impl Into<Sc<'k>>,
        init: [I; N],
        body: impl FnOnce(&mut Self, Sc<'k>, [Acc<'k, T>; N]) -> [Acc<'k, T>; N],
    ) -> [Acc<'k, T>; N] {
        let init = init.map(|i| i.into().0).to_vec();
        let out = self.loop_vec(extent, init, |k, iv, phi| body(k, iv, tiles(phi).map(Acc)).map(|a| a.0).to_vec());
        tiles(out).map(Acc)
    }

    /// [`Self::loop_`] over as many tiles as `init` holds, carried as plain
    /// register tiles.
    pub fn loop_vec<T: Elem>(
        &mut self,
        extent: impl Into<Sc<'k>>,
        init: Vec<Regs<'k, T>>,
        body: impl FnOnce(&mut Self, Sc<'k>, Vec<Regs<'k, T>>) -> Vec<Regs<'k, T>>,
    ) -> Vec<Regs<'k, T>> {
        let (start, extent) = (self.intern(0), self.intern(extent));
        let iv = self.push_scalar(Scalar::Induction);
        let phi = self.carried(&init);
        let mut next = vec![];
        let mut block = self.with_block(|k| next = body(k, Sc::id(iv), phi.iter().map(|&v| Tile::new(v)).collect()));
        self.snapshot_carried(&mut block, &phi, &mut next);
        self.deshare_carried(&mut block, &mut next);
        let carried = carry(&init, &phi, &next);
        self.push(Stmt::Loop(Loop { iv, start, extent, carried, body: block, unroll: 1 }));
        phi.into_iter().map(Tile::new).collect()
    }

    /// A producer/consumer loop over `extent` steps with a ring of slots run
    /// as `schedule` says ([`Schedule`]; a count of slots is a rolled ring).
    /// `produce(k, step, slot)` stages step `step` into the slot it fills
    /// ([`Kernel::slot`]); `consume(k, step, slot, carried)` computes from the
    /// slot it reads. A slot is its stage's alone.
    pub fn pipeline<T: Elem, I: Into<Acc<'k, T>>, const N: usize>(
        &mut self,
        extent: impl Into<Sc<'k>>,
        schedule: impl Into<Schedule>,
        init: [I; N],
        produce: impl for<'r> FnOnce(&mut Self, Sc<'k>, Filling<'k, 'r>),
        consume: impl for<'r> FnOnce(&mut Self, Sc<'k>, Reading<'k, 'r>, [Acc<'k, T>; N]) -> [Acc<'k, T>; N],
    ) -> [Acc<'k, T>; N] {
        let init = init.map(|i| i.into().0).to_vec();
        let consume = |k: &mut Self, step, slot: Reading<'k, '_>, phi: Vec<Regs<'k, T>>| {
            consume(k, step, slot, tiles(phi).map(Acc)).map(|a| a.0).to_vec()
        };
        tiles(self.pipeline_vec(extent, schedule, init, produce, consume)).map(Acc)
    }

    /// [`Self::pipeline`] over as many tiles as `init` holds, carried as
    /// plain register tiles.
    pub fn pipeline_vec<T: Elem>(
        &mut self,
        extent: impl Into<Sc<'k>>,
        schedule: impl Into<Schedule>,
        init: Vec<Regs<'k, T>>,
        produce: impl for<'r> FnOnce(&mut Self, Sc<'k>, Filling<'k, 'r>),
        consume: impl for<'r> FnOnce(&mut Self, Sc<'k>, Reading<'k, 'r>, Vec<Regs<'k, T>>) -> Vec<Regs<'k, T>>,
    ) -> Vec<Regs<'k, T>> {
        let schedule = schedule.into();
        let extent = self.intern(extent);
        let [pstep, pslot, cstep, cslot] = [(); 4].map(|()| self.push_scalar(Scalar::Induction));
        let phi = self.carried(&init);
        let body = self.with_block(|k| produce(k, Sc::id(pstep), Filling::new(pslot)));
        let produce = Stage { step: pstep, slot: pslot, body };
        let mut next = vec![];
        let mut body = self.with_block(|k| {
            next = consume(k, Sc::id(cstep), Reading::new(cslot), phi.iter().map(|&v| Tile::new(v)).collect())
        });
        self.snapshot_carried(&mut body, &phi, &mut next);
        self.deshare_carried(&mut body, &mut next);
        let consume = Stage { step: cstep, slot: cslot, body };
        let carried = carry(&init, &phi, &next);
        self.push(Stmt::Pipeline(Pipeline { extent, schedule, carried, produce, consume }));
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

    pub fn if_(&mut self, pred: impl Into<Sc<'k>>, then: impl FnOnce(&mut Self), otherwise: impl FnOnce(&mut Self)) {
        let pred = self.intern(pred);
        let then = self.with_block(then);
        let otherwise = self.with_block(otherwise);
        self.push(Stmt::If { pred, then, otherwise, results: vec![] });
    }

    /// `if pred { then } else { otherwise }` producing `N` register tiles:
    /// each branch computes its values and the merged tiles are returned.
    pub fn select_if<T: Elem, const N: usize>(
        &mut self,
        pred: impl Into<Sc<'k>>,
        then: impl FnOnce(&mut Self) -> [Regs<'k, T>; N],
        otherwise: impl FnOnce(&mut Self) -> [Regs<'k, T>; N],
    ) -> [Regs<'k, T>; N] {
        tiles(self.select_if_vec(pred, |k| then(k).to_vec(), |k| otherwise(k).to_vec()))
    }

    /// [`Self::select_if`] over as many tiles as the branches produce.
    pub fn select_if_vec<T: Elem>(
        &mut self,
        pred: impl Into<Sc<'k>>,
        then: impl FnOnce(&mut Self) -> Vec<Regs<'k, T>>,
        otherwise: impl FnOnce(&mut Self) -> Vec<Regs<'k, T>>,
    ) -> Vec<Regs<'k, T>> {
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
        pred: Regs<'k, Bool>,
        then: impl FnOnce(&mut Self) -> [Regs<'k, T>; N],
        otherwise: [Regs<'k, T>; N],
    ) -> [Regs<'k, T>; N] {
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
        pred: Regs<'k, Bool>,
        then: impl FnOnce(&mut Self) -> Vec<Regs<'k, T>>,
        otherwise: Vec<Regs<'k, T>>,
    ) -> Vec<Regs<'k, T>> {
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

    fn finish(mut self) -> Program {
        assert_eq!(self.blocks.len(), 1, "every block closed");
        self.prog.body = self.blocks.pop().expect("the root block");
        self.prog.index = IndexWidth::of(&self.prog.params);
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
fn tiles<'k, T, const N: usize>(list: Vec<Regs<'k, T>>) -> [Regs<'k, T>; N] {
    list.try_into().unwrap_or_else(|l: Vec<_>| panic!("{} tiles where {N} are carried", l.len()))
}

/// Each `init`'s carried register: its `phi` and its `next`.
fn carry<T>(init: &[Regs<'_, T>], phi: &[ValId], next: &[Regs<'_, T>]) -> Vec<Carried> {
    assert_eq!(next.len(), init.len(), "a body returns a tile per carried one");
    (0..init.len()).map(|i| Carried { init: init[i].0, phi: phi[i], next: next[i].0 }).collect()
}
