//! The tile program: statements in program order over block-collective tile
//! values. Nothing here is a DAG; identity is position, and the order the
//! author wrote is the order the kernel runs in.
//!
//! Layouts are not part of a value: the lowering assigns them from the atoms
//! that consume each value and keeps them in a side table, starting from the
//! few an author pins ([`Program::pins`]).

use svod_dtype::ScalarDType;

macro_rules! id {
    ($($name:ident),*) => {$(
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(pub u32);
        impl $name {
            pub fn index(self) -> usize {
                self.0 as usize
            }
        }
    )*};
}
id!(ValId, ScalarId, ParamId, SmemId);

/// Where a tile value lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tier {
    Global,
    Smem,
    Reg,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Shape {
    pub rows: usize,
    pub cols: usize,
}

impl Shape {
    pub const fn new(rows: usize, cols: usize) -> Self {
        Self { rows, cols }
    }
    pub fn elems(self) -> usize {
        self.rows * self.cols
    }
    pub fn transposed(self) -> Self {
        Self { rows: self.cols, cols: self.rows }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Axis {
    Row,
    Col,
}

/// A tile value: its element type, logical shape, and where it lives.
#[derive(Clone, Debug, PartialEq)]
pub struct Value {
    pub dtype: ScalarDType,
    pub shape: Shape,
    pub place: Place,
}

impl Value {
    pub fn tier(&self) -> Tier {
        match self.place {
            Place::Reg => Tier::Reg,
            Place::Smem { .. } => Tier::Smem,
            Place::Global { .. } => Tier::Global,
        }
    }
}

/// A view into an allocation. Global and shared views are windows the author
/// moves with scalar expressions; registers are the value itself.
#[derive(Clone, Debug, PartialEq)]
pub enum Place {
    Reg,
    /// `alloc[offset + r·cols + c]` in a row-major allocation,
    /// `alloc[offset + c·rows + r]` in a column-major one ([`SmemAlloc::along`]),
    /// before the swizzle the lowering applies; `offset` in elements.
    Smem {
        alloc: SmemId,
        offset: ScalarId,
    },
    /// `param[offset + r·stride.0 + c·stride.1]`, in elements; `bounds` are
    /// the valid row/col counts of the window (`None` = fully in bounds).
    /// Stores past them are dropped and register loads past them read zero,
    /// but a fill into shared memory re-reads the last valid row past the
    /// row bound (what it stages there is only ever consumed into rows the
    /// store drops) and zeroes its 16-byte chunks that start past the column
    /// bound, so the rows' stride must cover the chunk the bound falls in.
    ///
    /// With `rows`, row `r` starts at `offset + rows.offset[Row := r]`
    /// instead (`stride.0` is unused) and a row outside `rows.valid` reads
    /// zero on every path, fills included: a gather whose padding is data.
    ///
    /// With `period`, column `c` reads column `c % period`: a table every
    /// `period` columns of the window repeat. It is read into registers
    /// only, and has no column bound.
    Global {
        param: ParamId,
        offset: ScalarId,
        stride: [ScalarId; 2],
        bounds: [Option<ScalarId>; 2],
        rows: Option<RowMap>,
        period: Option<usize>,
    },
}

/// A per-row offset and validity, expressions over [`Scalar::Row`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RowMap {
    pub offset: ScalarId,
    /// Nonzero where the row is in bounds; `None`: every row is.
    pub valid: Option<ScalarId>,
}

/// Scalar expressions live in an arena and are shared by index.
#[derive(Clone, Debug, PartialEq)]
pub enum Scalar {
    Const(i64),
    /// A bound symbolic variable of the launch (e.g. the live batch).
    Var(String),
    Special(Special),
    /// A loop induction variable or pipeline step; defined by its statement.
    Induction,
    Bin(BinOp, ScalarId, ScalarId),
    /// `param[index]` of an integer parameter (lengths, offsets, row maps).
    Load {
        param: ParamId,
        index: ScalarId,
    },
    /// The row coordinate inside the view whose [`RowMap`] reads it.
    Row,
    /// Whether some element of a boolean register tile is set among those
    /// the warp evaluating it holds: one for the whole block in the
    /// interpreter, whose block is one thread. Only a branch reads it, and
    /// only one whose results agree with the other branch's wherever the
    /// tile is clear ([`crate::build::Kernel::where_any`] selects so,
    /// [`crate::build::Kernel::if_any`] leaves it to its author), so how the
    /// lowering spreads the tile over warps cannot change them.
    Any(ValId),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Special {
    /// Block index along grid axis 0..3.
    Block(u8),
    /// Warp index within the block.
    Warp,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    /// Truncating toward zero, as C's.
    Div,
    /// The remainder of [`BinOp::Div`]: the numerator's sign.
    Rem,
    Min,
    Max,
    Lt,
    Le,
    Eq,
    And,
    Or,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Const {
    Int(i64),
    Float(f64),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Neg,
    Exp2,
    Log2,
    Recip,
    Sqrt,
    Rsqrt,
    Abs,
    Not,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    /// `a · b`, exactly as [`BinaryOp::Mul`]; the lowering skips the
    /// multiplies of a thread whose factors in `b` are all exactly 1, so an
    /// update that rarely applies (rescaling an accumulator) costs a compare
    /// where it does not.
    Scale,
    Div,
    Max,
    Min,
    Lt,
    Le,
    Eq,
    Ne,
    And,
    Or,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    Sum,
    Max,
    Min,
}

/// Pure tile operations. Elementwise operands must share a shape, or one of
/// them is a vector (`[rows, 1]` or `[1, cols]`) broadcast along the other axis.
#[derive(Clone, Debug, PartialEq)]
pub enum TileOp {
    Fill(Const),
    /// A scalar in every element.
    Splat(ScalarId),
    /// The row or column index of every element, as `i32`.
    Coord(Axis),
    Unary {
        src: ValId,
        f: UnaryOp,
    },
    Binary {
        a: ValId,
        b: ValId,
        f: BinaryOp,
    },
    Cast {
        src: ValId,
        to: ScalarDType,
        round: Round,
    },
    /// `acc + A·B`, with `A` read transposed when `a_t` (an `[k, m]` tile) and
    /// `B` read transposed when `b_t` (an `[n, k]` tile). `orient` is which
    /// way round the core issues it: `None` until the layout inference
    /// chooses, `Some` once laid or where the author pinned it.
    Mma {
        acc: ValId,
        a: ValId,
        b: ValId,
        a_t: bool,
        b_t: bool,
        orient: Option<Orient>,
    },
    /// Reduce along `axis`: `Row` folds every row to one column (`[rows, 1]`),
    /// `Col` folds every column to one row (`[1, cols]`). The fold runs in
    /// the destination's type, so a 16-bit source may be summed in f32.
    Reduce {
        src: ValId,
        axis: Axis,
        f: ReduceOp,
    },
    Where {
        pred: ValId,
        a: ValId,
        b: ValId,
    },
    /// `a·b + c` rounded once: one fused instruction whatever the layout
    /// around it, where a separate multiply and add leave the backend free
    /// to fuse either multiply feeding the add.
    MulAdd {
        a: ValId,
        b: ValId,
        c: ValId,
    },
    /// Logical transpose of a register tile.
    Transpose {
        src: ValId,
    },
    /// Every element one step on along `axis` (`out[r][c] = src[r][c − 1]`
    /// along `Col`), the first row or column `fill`.
    Shift {
        src: ValId,
        axis: Axis,
        fill: Const,
    },
    /// Every element exchanged with the one whose coordinate along `axis`
    /// differs by the bits of `mask` (`out[r][c] = src[r][c ^ mask]` along
    /// `Col`): a butterfly, a step of a sorting or merging network. The
    /// axis is a multiple of the power of two past `mask` long.
    Flip {
        src: ValId,
        axis: Axis,
        mask: u32,
    },
    /// The `j`th of the columns each lane holds of its rows: `src` is
    /// `[rows, w·n]`, the result `[rows, w]` (`w` its columns, `j < n`), each
    /// element one of the same row of `src`, or with `col` that element's
    /// column (`i32`). Over `j < n` a row's results take each of its columns
    /// once, in the order the lowering holds them: column `c` of the result
    /// is held by the lanes holding `src`'s columns it takes, so dealing is
    /// a renaming of registers wherever the layouts allow, else the
    /// interpreter's order, column `c·n + j`. What a program computes from
    /// a deal must not depend on that order but through `col`.
    Deal {
        src: ValId,
        j: u32,
        col: bool,
        /// The source column each element takes, per row of the result, as
        /// the layouts the inference assigned hold them
        /// ([`crate::layouts::TileLayout::dealt`]): the device's order,
        /// recorded once laid so the interpreter replays it. `None` until
        /// then (and in a program never lowered): the fallback order,
        /// column `c·n + j`.
        order: Option<Vec<Vec<u32>>>,
    },
    /// The same elements held under the layout its consumer needs; inserted
    /// by the layout inference, never by an author (who pins layouts instead).
    Relayout {
        src: ValId,
    },
    /// A copy into a value that mutually exclusive branches each define once
    /// (the merge of an `if`), or that is defined before an `if` and again
    /// in its branch, which replaces it ([`crate::build::Kernel::if_any`]).
    Move {
        src: ValId,
    },
}

impl TileOp {
    /// The register values the op reads.
    pub fn srcs(&self) -> impl Iterator<Item = ValId> {
        let [a, b, c] = match *self {
            TileOp::Fill(_) | TileOp::Splat(_) | TileOp::Coord(_) => [None; 3],
            TileOp::Unary { src, .. }
            | TileOp::Cast { src, .. }
            | TileOp::Reduce { src, .. }
            | TileOp::Transpose { src }
            | TileOp::Shift { src, .. }
            | TileOp::Flip { src, .. }
            | TileOp::Deal { src, .. }
            | TileOp::Relayout { src }
            | TileOp::Move { src } => [Some(src), None, None],
            TileOp::Binary { a, b, .. } => [Some(a), Some(b), None],
            TileOp::Mma { acc, a, b, .. } => [Some(acc), Some(a), Some(b)],
            TileOp::Where { pred, a, b } => [Some(pred), Some(a), Some(b)],
            TileOp::MulAdd { a, b, c } => [Some(a), Some(b), Some(c)],
        };
        [a, b, c].into_iter().flatten()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CopyMode {
    /// Complete before the next statement; the sync pass fences it.
    Sync,
    /// May complete later; the schedule template waits and fences it.
    Async,
    /// One half of a register-staged copy the schedule template placed and
    /// fences itself. Its global → register half loads as the kernel's fills
    /// read (at L2 only where they are `cp.async.cg` copies, plain loads
    /// where they are register-staged), so a register load in this mode
    /// moves bytes exactly as a fill would.
    Staged,
}

/// A loop-carried register: `phi` holds `init` on entry, is assigned `next`
/// at the end of every iteration, and holds the last `next` after the loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Carried {
    pub init: ValId,
    pub phi: ValId,
    pub next: ValId,
}

/// `iv` runs over `start..start + extent` (no trip at all below an extent
/// of one); several loops may share an `iv` (and the scalars defined from
/// it) over consecutive ranges.
#[derive(Clone, Debug, PartialEq)]
pub struct Loop {
    pub iv: ScalarId,
    pub start: ScalarId,
    pub extent: ScalarId,
    pub carried: Vec<Carried>,
    pub body: Block,
    /// Copies of the body per emitted iteration (1 = rolled); the schedule
    /// template sets it so ring-slot arithmetic folds to constants.
    pub unroll: u32,
}

/// One side of a [`Pipeline`]: a body parameterized by the step it handles
/// and the ring slot that step uses.
#[derive(Clone, Debug, PartialEq)]
pub struct Stage {
    pub step: ScalarId,
    pub slot: ScalarId,
    pub body: Block,
}

/// A producer/consumer loop over `extent` steps with a ring of `stages`
/// shared-memory slots. `produce` fills its slot for its step; `consume`
/// reads its slot for its step; the schedule template decides how far ahead
/// production runs and which warps do what.
#[derive(Clone, Debug, PartialEq)]
pub struct Pipeline {
    pub extent: ScalarId,
    pub stages: usize,
    /// The loop copied once per ring slot, so slot addressing folds to
    /// constants.
    pub unroll: bool,
    /// 1: every warp loads and computes; 2: the block's warps as two crews
    /// a cluster apart over one slot ([`crate::schedule`]'s ping-pong).
    pub crews: u32,
    pub carried: Vec<Carried>,
    pub produce: Stage,
    pub consume: Stage,
}

/// Whose threads a barrier gathers and a copy's chunks are spread over: the
/// block's, or (inside [`Stmt::Warps`]) one warp's own.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Scope {
    Block,
    Warp,
}

/// Synchronization the schedule templates and the sync pass emit; an author
/// writes only a block barrier ([`crate::build::Kernel::barrier`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Sync {
    /// Every thread of the `scope` arrives before any continues;
    /// shared-memory writes before it are visible after it. A warp's own
    /// barrier orders its lanes' shared accesses and nothing else.
    Barrier { scope: Scope },
    /// Close the group of this thread's async copies issued since the last
    /// commit.
    CommitAsync,
    /// Block until at most `pending` of this thread's committed groups are in
    /// flight.
    WaitAsync { pending: u32 },
    /// No instruction moves across this point: a pipeline trip's loads stay
    /// before its products and its commit after them, where a machine
    /// scheduler would otherwise move either past them. Nothing to a target
    /// that keeps program order.
    Fence,
    /// Block until at most `pending` of this thread's warpgroup product
    /// chains are in flight (`wgmma.wait_group`): the products of the block
    /// holding this statement issue asynchronously and this is their wait.
    /// Nothing to a target without such chains.
    WaitProducts { pending: u32 },
    /// The wave's issue priority from here on (`s_setprio`): raised over a
    /// crew's products so the SIMD's other wave, loading, yields its issue
    /// slots to them. Nothing where waves have no priority.
    Prio { level: u8 },
}

#[derive(Clone, Debug, PartialEq)]
pub enum Stmt {
    Let {
        dst: ValId,
        op: TileOp,
    },
    Copy {
        dst: ValId,
        src: ValId,
        mode: CopyMode,
        /// Whose lanes move it: the block's, or one warp's own inside a
        /// [`Stmt::Warps`].
        scope: Scope,
    },
    Loop(Loop),
    Pipeline(Pipeline),
    /// Every warp of the block runs `body` on its own: `Special::Warp` is
    /// its index, its copies move whole tiles with its lanes alone, and its
    /// barriers gather only itself. What the warps leave in shared memory is
    /// the block's open effect afterwards: the sync pass fences it before the
    /// block next touches it.
    Warps {
        body: Block,
    },
    If {
        pred: ScalarId,
        then: Block,
        otherwise: Block,
        /// The branch's merge: past the statement each `dst` holds its
        /// branch's source (`None`: what `dst` held before it, which `then`
        /// replaces). Both sources keep their identity — every value stays
        /// defined once, by the branch that computes it or before — and the
        /// moves into `dst` are the branch's last, so a `dst` a loop carries
        /// moves into its `phi` at the end of the body holding the `if`,
        /// never inside the branch ([`crate::build::Kernel::select_if_vec`],
        /// [`crate::build::Kernel::if_any`]).
        results: Vec<(ValId, ValId, Option<ValId>)>,
    },
    Sync(Sync),
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Block(pub Vec<Stmt>);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ParamKind {
    In,
    Out,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    pub name: String,
    pub dtype: ScalarDType,
    pub kind: ParamKind,
    /// Capacity in elements (symbolic dims sized at their maximum).
    pub elems: usize,
}

/// A shared allocation. Its tiles are stored with consecutive elements
/// along `along`: `Col` is row-major, `Row` column-major. The lowering sets
/// it to the axis the tiles' consumers read fragments along
/// (`lower::orient_shared`); an author never does.
#[derive(Clone, Debug, PartialEq)]
pub struct SmemAlloc {
    pub name: String,
    pub dtype: ScalarDType,
    pub elems: usize,
    pub along: Axis,
}

/// A bound symbolic variable of the launch, with the range it may take.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Var {
    pub name: String,
    pub min: i64,
    pub max: i64,
}

/// How a cast to a narrower float rounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Round {
    /// To the nearest value, ties to even.
    Nearest,
    /// Toward zero: f32 → bf16 keeps the upper half of the bits, one
    /// instruction per pair where the target has no bf16 convert. A NaN whose
    /// payload lies in the lower half becomes an infinity of its sign.
    Zero,
}

/// Which way round a product is issued. `C = A·B` and its transpose
/// `Cᵀ = Bᵀ·Aᵀ` are the same instruction with the operand slots exchanged,
/// and a transposed layout is a relabelling of `Row` and `Col`, so the
/// choice costs nothing and only decides which of the core's layouts each
/// value is held in. The layout inference chooses; an author may pin.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Orient {
    Direct,
    Swapped,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Program {
    pub name: String,
    pub params: Vec<Param>,
    pub vars: Vec<Var>,
    pub grid: [ScalarId; 3],
    pub warps: u32,
    /// How the warps spread a block-wide register tile: every warp, or a
    /// whole grid of them where the others run only inside
    /// [`Stmt::Warps`].
    pub warp_grid: crate::layouts::WarpGrid,
    pub smem: Vec<SmemAlloc>,
    pub scalars: Vec<Scalar>,
    pub values: Vec<Value>,
    pub body: Block,
    /// Register values the author holds under a fixed layout; a consumer
    /// wanting another re-holds them (a probe of what that costs, say).
    pub pins: Vec<(ValId, crate::layouts::TileLayout)>,
}

impl Program {
    pub fn value(&self, id: ValId) -> &Value {
        &self.values[id.index()]
    }
    pub fn scalar(&self, id: ScalarId) -> &Scalar {
        &self.scalars[id.index()]
    }
    /// Whether scalar `id` reads [`Special::Warp`]: the warps of a block may
    /// see it differ.
    pub fn varies_by_warp(&self, id: ScalarId) -> bool {
        match self.scalar(id) {
            Scalar::Special(Special::Warp) | Scalar::Any(_) => true,
            Scalar::Bin(_, a, b) => self.varies_by_warp(*a) || self.varies_by_warp(*b),
            Scalar::Load { index, .. } => self.varies_by_warp(*index),
            _ => false,
        }
    }
    /// Every statement, outermost first, with its nesting depth.
    pub fn walk(&self) -> impl Iterator<Item = (usize, &Stmt)> {
        self.body.walk()
    }

    /// The values scalar `id` may take, `lo..=hi`, from its inputs' ranges
    /// (a variable's, a block index's grid, the warps, and an induction's as
    /// `induction` bounds it where it is bound); `None` where an input is
    /// unbounded (a value read from memory, a row coordinate) or a bound
    /// leaves `i64`. Each operation is bounded on its own: a truncating
    /// quotient over a divisor of one sign, a remainder by the numerator's
    /// sign and the divisor's magnitude, a comparison `0..=1`.
    pub fn bounds(&self, id: ScalarId, induction: &impl Fn(ScalarId) -> Option<(i64, i64)>) -> Option<(i64, i64)> {
        match self.scalar(id) {
            Scalar::Const(v) => Some((*v, *v)),
            Scalar::Var(name) => self.vars.iter().find(|v| v.name == *name).map(|v| (v.min, v.max)),
            Scalar::Special(Special::Block(axis)) => {
                Some((0, self.bounds(self.grid[*axis as usize], induction)?.1 - 1))
            }
            Scalar::Special(Special::Warp) => Some((0, i64::from(self.warps) - 1)),
            Scalar::Any(_) => Some((0, 1)),
            Scalar::Induction => induction(id),
            Scalar::Bin(op, a, b) => {
                let ((a0, a1), (b0, b1)) = (self.bounds(*a, induction)?, self.bounds(*b, induction)?);
                let corners = |f: fn(i64, i64) -> Option<i64>| -> Option<(i64, i64)> {
                    let v = [f(a0, b0)?, f(a0, b1)?, f(a1, b0)?, f(a1, b1)?];
                    Some((v.into_iter().min()?, v.into_iter().max()?))
                };
                // `x − [c < x]` (a count less one where it passes `c`: a
                // register-staged ring's steady trips) stays at `x` up to
                // `c` and a step under it past, never a step under `x`'s
                // own low.
                let less_one_past = |b: ScalarId| match self.scalar(b) {
                    Scalar::Bin(BinOp::Lt, c, x) if *x == *a => match self.scalar(*c) {
                        Scalar::Const(c) => Some(*c),
                        _ => None,
                    },
                    _ => None,
                };
                match op {
                    BinOp::Add => Some((a0.checked_add(b0)?, a1.checked_add(b1)?)),
                    BinOp::Sub => match less_one_past(*b) {
                        Some(c) => Some((if a0 > c { a0 - 1 } else { a0 }, if a1 > c { a1 - 1 } else { a1 })),
                        None => Some((a0.checked_sub(b1)?, a1.checked_sub(b0)?)),
                    },
                    BinOp::Mul => corners(i64::checked_mul),
                    BinOp::Div if b0 > 0 || b1 < 0 => corners(i64::checked_div),
                    BinOp::Rem if b0 > 0 || b1 < 0 => {
                        let most = b0.unsigned_abs().max(b1.unsigned_abs()) as i64 - 1;
                        Some((if a0 < 0 { a0.max(-most) } else { 0 }, if a1 > 0 { a1.min(most) } else { 0 }))
                    }
                    BinOp::Div | BinOp::Rem => None,
                    BinOp::Min => Some((a0.min(b0), a1.min(b1))),
                    BinOp::Max => Some((a0.max(b0), a1.max(b1))),
                    BinOp::Lt | BinOp::Le | BinOp::Eq => Some((0, 1)),
                    BinOp::And | BinOp::Or if a0 >= 0 && b0 >= 0 && a1 <= 1 && b1 <= 1 => Some((0, 1)),
                    BinOp::And | BinOp::Or => None,
                }
            }
            Scalar::Load { .. } | Scalar::Row => None,
        }
    }
}

impl Block {
    /// Every statement, each before the ones it nests, with its depth below
    /// this block.
    pub fn walk(&self) -> impl Iterator<Item = (usize, &Stmt)> {
        fn visit<'a>(block: &'a Block, depth: usize, out: &mut Vec<(usize, &'a Stmt)>) {
            for stmt in &block.0 {
                out.push((depth, stmt));
                stmt.blocks().for_each(|b| visit(b, depth + 1, out));
            }
        }
        let mut out = Vec::new();
        visit(self, 0, &mut out);
        out.into_iter()
    }
}

impl Stmt {
    /// The blocks the statement nests, in program order: a loop's or the
    /// warps' body, a pipeline's produce then consume, a branch's then and
    /// otherwise.
    pub fn blocks(&self) -> impl Iterator<Item = &Block> {
        let (first, second) = match self {
            Stmt::Loop(l) => (Some(&l.body), None),
            Stmt::Pipeline(p) => (Some(&p.produce.body), Some(&p.consume.body)),
            Stmt::Warps { body } => (Some(body), None),
            Stmt::If { then, otherwise, .. } => (Some(then), Some(otherwise)),
            Stmt::Let { .. } | Stmt::Copy { .. } | Stmt::Sync(_) => (None, None),
        };
        first.into_iter().chain(second)
    }

    /// [`Self::blocks`], mutably.
    pub fn blocks_mut(&mut self) -> impl Iterator<Item = &mut Block> {
        let (first, second) = match self {
            Stmt::Loop(l) => (Some(&mut l.body), None),
            Stmt::Pipeline(p) => (Some(&mut p.produce.body), Some(&mut p.consume.body)),
            Stmt::Warps { body } => (Some(body), None),
            Stmt::If { then, otherwise, .. } => (Some(then), Some(otherwise)),
            Stmt::Let { .. } | Stmt::Copy { .. } | Stmt::Sync(_) => (None, None),
        };
        first.into_iter().chain(second)
    }
}
