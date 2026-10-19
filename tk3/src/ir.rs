//! The tile program: statements in program order over block-collective tile
//! values. Nothing here is a DAG; identity is position, and the order the
//! author wrote is the order the kernel runs in.
//!
//! Layouts are not part of a value: the lowering assigns them from the atoms
//! that consume each value and keeps them in a side table.

use std::ops::Range;

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
id!(ValId, ScalarId, ParamId, SmemId, RoleId);

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
    /// the valid row/col counts of the window (`None` = fully in bounds), so
    /// loads past them are gated and stores past them dropped.
    Global {
        param: ParamId,
        offset: ScalarId,
        stride: [ScalarId; 2],
        bounds: [Option<ScalarId>; 2],
    },
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
    },
    /// `acc + A·B`, with `A` read transposed when `a_t` (an `[k, m]` tile) and
    /// `B` read transposed when `b_t` (an `[n, k]` tile).
    Mma {
        acc: ValId,
        a: ValId,
        b: ValId,
        a_t: bool,
        b_t: bool,
    },
    /// Reduce along `axis`: `Row` folds every row to one column (`[rows, 1]`),
    /// `Col` folds every column to one row (`[1, cols]`).
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
    /// Logical transpose of a register tile.
    Transpose {
        src: ValId,
    },
    /// The same elements held under the layout its consumer needs; inserted
    /// by the layout inference, never by an author.
    Relayout {
        src: ValId,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CopyMode {
    /// Complete before the next statement.
    Sync,
    /// May complete later; the lowering waits before the first read of `dst`.
    Async,
}

/// A loop-carried register: `phi` holds `init` on entry, is assigned `next`
/// at the end of every iteration, and holds the last `next` after the loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Carried {
    pub init: ValId,
    pub phi: ValId,
    pub next: ValId,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Loop {
    pub iv: ScalarId,
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

/// Synchronization the schedule templates and the sync pass emit; authors
/// never write these.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Sync {
    /// Every thread of the block (or of `role`'s warps) arrives before any
    /// continues; shared-memory writes before it are visible after it.
    Barrier { role: Option<RoleId> },
    /// Close the group of this thread's async copies issued since the last
    /// commit.
    CommitAsync,
    /// Block until at most `pending` of this thread's committed groups are in
    /// flight.
    WaitAsync { pending: u32 },
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
    },
    Loop(Loop),
    Pipeline(Pipeline),
    /// Run only by the warps of `role`.
    Role {
        role: RoleId,
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

#[derive(Clone, Debug, PartialEq)]
pub struct SmemAlloc {
    pub name: String,
    pub dtype: ScalarDType,
    pub elems: usize,
}

/// A warp-specialized role: which warps run it and, when the target supports
/// it, how many registers each of its threads may use.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Role {
    pub name: String,
    pub warps: Range<u32>,
    pub regs: Option<u32>,
}

/// A bound symbolic variable of the launch, with the range it may take.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Var {
    pub name: String,
    pub min: i64,
    pub max: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Program {
    pub name: String,
    pub params: Vec<Param>,
    pub vars: Vec<Var>,
    pub grid: [ScalarId; 3],
    pub warps: u32,
    pub roles: Vec<Role>,
    pub smem: Vec<SmemAlloc>,
    pub scalars: Vec<Scalar>,
    pub values: Vec<Value>,
    pub body: Block,
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
                Stmt::Role { body, .. } => body.walk(depth + 1, out),
                Stmt::If { then, otherwise, .. } => {
                    then.walk(depth + 1, out);
                    otherwise.walk(depth + 1, out);
                }
                Stmt::Let { .. } | Stmt::Copy { .. } | Stmt::Sync(_) | Stmt::Raw(_) => {}
            }
        }
    }
}
