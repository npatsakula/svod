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
    /// `alloc[offset + r·cols + c]`, `offset` in elements.
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
    Global {
        param: ParamId,
        offset: ScalarId,
        stride: [ScalarId; 2],
        bounds: [Option<ScalarId>; 2],
        rows: Option<RowMap>,
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
    Div,
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
    /// The same elements held under the layout its consumer needs; inserted
    /// by the layout inference, never by an author (who pins layouts instead).
    Relayout {
        src: ValId,
    },
    /// A copy into a value that mutually exclusive branches each define once
    /// (the merge of an `if`).
    Move {
        src: ValId,
    },
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

/// `iv` runs over `start..start + extent`; several loops may share an `iv`
/// (and the scalars defined from it) over consecutive ranges.
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
    pub carried: Vec<Carried>,
    pub produce: Stage,
    pub consume: Stage,
}

/// Vendor text with declared operands, results and effects.
#[derive(Clone, Debug, PartialEq)]
pub struct Raw {
    pub code: String,
    pub operands: Vec<ValId>,
    pub results: Vec<ValId>,
    pub reads: Vec<ValId>,
    pub writes: Vec<ValId>,
}

/// Whose threads a barrier gathers and a copy's chunks are spread over: the
/// block's, or (inside [`Stmt::Warps`]) one warp's own.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Scope {
    Block,
    Warp,
}

/// Synchronization the schedule templates and the sync pass emit; authors
/// never write these.
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
    /// No instruction moves across this point: a pipeline trip's commit stays
    /// after its products, where a machine scheduler would otherwise hoist
    /// it above them. Nothing to a target that keeps program order.
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
    /// fenced for the block at the statement's end.
    Warps {
        body: Block,
    },
    If {
        pred: ScalarId,
        then: Block,
        otherwise: Block,
    },
    Sync(Sync),
    Raw(Raw),
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Block(pub Vec<Stmt>);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ParamKind {
    In,
    Out,
    InOut,
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
#[derive(Clone, Debug, PartialEq, Eq)]
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
    pub fn param(&self, id: ParamId) -> &Param {
        &self.params[id.index()]
    }
    /// Every statement, outermost first, with its nesting depth.
    pub fn walk(&self) -> impl Iterator<Item = (usize, &Stmt)> {
        let mut out = Vec::new();
        self.body.walk(0, &mut out);
        out.into_iter()
    }
}

impl Block {
    fn walk<'a>(&'a self, depth: usize, out: &mut Vec<(usize, &'a Stmt)>) {
        for stmt in &self.0 {
            out.push((depth, stmt));
            match stmt {
                Stmt::Loop(l) => l.body.walk(depth + 1, out),
                Stmt::Pipeline(p) => {
                    p.produce.body.walk(depth + 1, out);
                    p.consume.body.walk(depth + 1, out);
                }
                Stmt::Warps { body } => body.walk(depth + 1, out),
                Stmt::If { then, otherwise, .. } => {
                    then.walk(depth + 1, out);
                    otherwise.walk(depth + 1, out);
                }
                Stmt::Let { .. } | Stmt::Copy { .. } | Stmt::Sync(_) | Stmt::Raw(_) => {}
            }
        }
    }
}
