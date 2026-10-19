//! Operation enum and implementation.
//!
//! The [`Op`] enum defines all possible operations in the IR, from basic arithmetic
//! to complex control flow and memory operations.

use std::sync::Arc;

use smallvec::SmallVec;

use crate::types::*;
use crate::uop::UOp;
use svod_dtype::DType;
use svod_dtype::DeviceSpec;

/// Operation type with typed operands.
///
/// Each operation encodes its operand structure directly in the enum variant.
/// This provides compile-time verification of operand count and types.
///
/// Design choices:
/// - Fixed-arity ops grouped by arity: Unary, Binary, Ternary
/// - Special ops with extra data remain separate: Cast (dtype), MSelect (device_index)
/// - Variable-arity ops use SmallVec: `Index { indices: SmallVec<[Arc<UOp>; 4]> }`
/// - SmallVec avoids heap allocation for common cases (≤4 children)
/// - Late gating is carried by LOAD/STORE following Tinygrad's model
///
/// `Hash`/`PartialEq` are derived: children hash by their content hash and compare
/// by interned identity, so the derives give both the structural content hash and
/// the exact equality the intern table needs.
#[svod_macros::op_enum]
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
#[derive(strum::AsRefStr)]
#[pattern(grouped = [Unary, Binary, Ternary])]
pub enum Op {
    // Nullary operations
    Const(ConstValueHash),
    Unique(usize),
    LUnique(usize),
    Noop,

    // Graph organization operations (2 variants)
    Sink {
        sources: SmallVec<[Arc<UOp>; 4]>,
        info: Option<Box<crate::types::KernelInfo>>,
    },
    Group {
        sources: SmallVec<[Arc<UOp>; 4]>,
    },

    // Grouped operations (3 variants)
    Unary(UnaryOp, Arc<UOp>),
    Binary(BinaryOp, Arc<UOp>, Arc<UOp>),
    Ternary(TernaryOp, Arc<UOp>, Arc<UOp>, Arc<UOp>),

    // Type operations (2 variants)
    Cast {
        src: Arc<UOp>,
        dtype: DType,
    },
    BitCast {
        src: Arc<UOp>,
        dtype: DType,
    },

    // Special operations (2 variants)
    MSelect {
        buffer: Arc<UOp>,
        device_index: usize,
    },
    Special {
        end: Arc<UOp>,
        name: String,
    },

    // Buffer operations (high-level, 8 variants)
    /// Normalized buffer parameter — positional reference to an input/output buffer.
    /// Created by pre-schedule normalization (BUFFER→PARAM) to erase buffer identity,
    /// enabling structural deduplication of identical computations on different buffers.
    /// Matches Tinygrad's Ops.PARAM (engine/schedule.py:125).
    Param {
        shape: Arc<UOp>,
        arg: Box<ParamArg>,
    },
    Buffer {
        shape: Arc<UOp>,
        arg: Box<ParamArg>,
    },
    /// Contiguous typed slice of a buffer. The offset is in source elements.
    /// Concrete allocation aliasing is owned by `svod_device::Buffer`.
    Slice {
        buffer: Arc<UOp>,
        offset: Arc<UOp>,
        size: usize,
    },
    Stage {
        compute: Arc<UOp>,
        ranges: SmallVec<[Arc<UOp>; 4]>,
        opts: Box<BufferizeOpts>,
    },
    Index {
        buffer: Arc<UOp>,
        indices: SmallVec<[Arc<UOp>; 4]>,
    },
    GetAddr {
        src: Arc<UOp>,
        device: DeviceSpec,
    },
    Copy {
        src: Arc<UOp>,
        device: DeviceSpec,
    },
    MStack {
        buffers: SmallVec<[Arc<UOp>; 4]>,
    },

    // Movement/Reshape operations (7 variants)
    Reshape {
        src: Arc<UOp>,
        new_shape: Arc<UOp>,
    },
    Permute {
        src: Arc<UOp>,
        axes: Vec<usize>,
    },
    Expand {
        src: Arc<UOp>,
        new_shape: Arc<UOp>,
    },
    Pad {
        src: Arc<UOp>,
        begin_pads: Arc<UOp>,
        end_pads: Arc<UOp>,
    },
    Shrink {
        src: Arc<UOp>,
        offsets: Arc<UOp>,
        sizes: Arc<UOp>,
    },
    Flip {
        src: Arc<UOp>,
        axes: Vec<bool>,
    },
    Multi {
        src: Arc<UOp>,
        axis: usize,
    },

    // Reduction operations (3 variants)
    ReduceAxis {
        src: Arc<UOp>,
        reduce_op: ReduceOp,
        axes: Vec<usize>,
    },
    Reduce {
        src: Arc<UOp>,
        ranges: SmallVec<[Arc<UOp>; 4]>,
        reduce_op: ReduceOp,
        num_axes: usize,
    },
    AllReduce {
        src: Arc<UOp>,
        device: DeviceSpec,
        reduce_op: ReduceOp,
    },

    // Control flow operations (5 variants)
    If {
        condition: Arc<UOp>,
        body: SmallVec<[Arc<UOp>; 4]>,
    },
    EndIf {
        if_op: Arc<UOp>,
    },
    Range {
        end: Arc<UOp>,
        axis_id: AxisId,
        axis_type: AxisType,
        deps: SmallVec<[Arc<UOp>; 2]>,
    },
    End {
        computation: Arc<UOp>,
        ranges: SmallVec<[Arc<UOp>; 4]>,
    },
    Barrier {
        src: Arc<UOp>,
        deps: SmallVec<[Arc<UOp>; 4]>,
        fence: BarrierFence,
    },

    // Shaped values and late vector operations
    Stack {
        sources: SmallVec<[Arc<UOp>; 4]>,
    },
    VConst {
        values: Vec<ConstValue>,
    },

    // Symbolic/Define operations (3 variants)
    DefineVar {
        name: String,
        min_val: i64,
        max_val: i64,
    },
    Bind {
        var: Arc<UOp>,
        value: Arc<UOp>,
    },

    // Advanced operations (11 variants)
    Wmma {
        a: Arc<UOp>,
        b: Arc<UOp>,
        c: Arc<UOp>,
        metadata: Box<WmmaMetadata>,
    },
    Call {
        body: Arc<UOp>,
        args: SmallVec<[Arc<UOp>; 4]>,
        info: Box<CallInfo>,
    },
    Function {
        body: Arc<UOp>,
        args: SmallVec<[Arc<UOp>; 4]>,
        info: Box<CallInfo>,
    },
    /// Heterogeneous tuple of values; dtype is always Void.
    /// Mirrors tinygrad `Ops.TUPLE` — wraps multi-value FUNCTION bodies so FUNCTION dtype stays void.
    Tuple {
        src: SmallVec<[Arc<UOp>; 4]>,
    },
    /// Extract element `index` from a TUPLE (or a FUNCTION whose body is a TUPLE).
    /// dtype matches the inner element's dtype.
    /// Mirrors tinygrad `Ops.GETTUPLE`.
    GetTuple {
        src: Arc<UOp>,
        index: usize,
    },
    Program {
        sink: Arc<UOp>,
        info: Box<ProgramInfo>,
        linear: Option<Arc<UOp>>,
        source: Option<Arc<UOp>>,
        binary: Option<Arc<UOp>>,
    },
    Linear {
        ops: SmallVec<[Arc<UOp>; 8]>,
    },
    Source {
        code: String,
        identity: Option<Box<SourceStageIdentity>>,
    },
    ProgramBinary {
        bytes: Vec<u8>,
        identity: Option<Box<BinaryStageIdentity>>,
    },
    Detach {
        src: Arc<UOp>,
    },
    Contiguous {
        src: Arc<UOp>,
        /// Optimization hints (Tinygrad: CONTIGUOUS.arg)
        opts: Vec<crate::types::ContiguousHint>,
    },
    ContiguousBackward {
        src: Arc<UOp>,
    },
    After {
        passthrough: Arc<UOp>,
        deps: SmallVec<[Arc<UOp>; 4]>,
    },
    Precast {
        src: Arc<UOp>,
    },
    Custom {
        deps: SmallVec<[Arc<UOp>; 4]>,
        code: String,
    },
    CustomFunction {
        kind: CustomFunctionKind,
        attrs: SmallVec<[Arc<UOp>; 4]>,
    },
    CustomI {
        deps: SmallVec<[Arc<UOp>; 4]>,
        code: String,
    },

    // Memory operations (low-level, after kernel splitting, 2 variants)
    /// Load from buffer at index.
    ///
    /// Gated loads carry both `alt` and `gate`; ungated loads carry neither.
    Load {
        index: Arc<UOp>,
        alt: Option<Arc<UOp>>,
        gate: Option<Arc<UOp>>,
    },
    Store {
        index: Arc<UOp>,
        value: Arc<UOp>,
        gate: Option<Arc<UOp>>,
    },
    /// Target instruction selected by an ISA renderer. Kept at the enum tail so
    /// adding ISA support does not perturb existing operation discriminants.
    Ins {
        sources: SmallVec<[Arc<UOp>; 4]>,
        arg: InsArg,
    },
}

impl Op {
    /// Get all child UOps as a Vec of references.
    ///
    /// This is the convenient API for traversing the graph.
    /// Allocates a Vec but is simple to use.
    pub fn children(&self) -> SmallVec<[&Arc<UOp>; 4]> {
        match self {
            // Nullary operations
            Self::Const(_)
            | Self::Unique(_)
            | Self::LUnique(_)
            | Self::Noop
            | Self::VConst(..)
            | Self::DefineVar(..)
            | Self::Source(..)
            | Self::ProgramBinary(..) => SmallVec::new(),

            Self::Param(ops::Param { shape, .. }) => SmallVec::from_slice(&[shape]),

            // Graph organization operations
            Self::Sink(ops::Sink { sources, .. }) | Self::Group(ops::Group { sources }) => sources.iter().collect(),

            // Grouped operations
            Self::Unary(_, x) => SmallVec::from_slice(&[x]),
            Self::Binary(_, a, b) => SmallVec::from_slice(&[a, b]),
            Self::Ternary(_, a, b, c) => SmallVec::from_slice(&[a, b, c]),

            // Type operations
            Self::Cast(ops::Cast { src, .. }) | Self::BitCast(ops::BitCast { src, .. }) => SmallVec::from_slice(&[src]),

            // Special operations
            Self::MSelect(ops::MSelect { buffer, .. }) => SmallVec::from_slice(&[buffer]),
            Self::Special(ops::Special { end, .. }) => SmallVec::from_slice(&[end]),

            // Buffer operations
            Self::Buffer(ops::Buffer { shape, .. }) => SmallVec::from_slice(&[shape]),
            Self::Slice(ops::Slice { buffer, offset, .. }) => SmallVec::from_slice(&[buffer, offset]),
            Self::Stage(ops::Stage { compute, ranges, .. }) => {
                let mut children = SmallVec::from_slice(&[compute]);
                children.extend(ranges.iter());
                children
            }
            Self::Index(ops::Index { buffer, indices }) => {
                let mut children = SmallVec::from_slice(&[buffer]);
                children.extend(indices.iter());
                children
            }
            Self::GetAddr(ops::GetAddr { src, .. }) => SmallVec::from_slice(&[src]),
            Self::Copy(ops::Copy { src, .. }) => SmallVec::from_slice(&[src]),
            Self::MStack(ops::MStack { buffers }) => buffers.iter().collect(),

            // Movement operations
            Self::Reshape(ops::Reshape { src, new_shape }) => SmallVec::from_slice(&[src, new_shape]),
            Self::Permute(ops::Permute { src, .. })
            | Self::Flip(ops::Flip { src, .. })
            | Self::Multi(ops::Multi { src, .. }) => SmallVec::from_slice(&[src]),
            Self::Expand(ops::Expand { src, new_shape }) => SmallVec::from_slice(&[src, new_shape]),
            Self::Pad(ops::Pad { src, begin_pads, end_pads }) => SmallVec::from_slice(&[src, begin_pads, end_pads]),
            Self::Shrink(ops::Shrink { src, offsets, sizes }) => SmallVec::from_slice(&[src, offsets, sizes]),

            // Reduction operations
            Self::ReduceAxis(ops::ReduceAxis { src, .. }) => SmallVec::from_slice(&[src]),
            Self::Reduce(ops::Reduce { src, ranges, .. }) => {
                let mut children = SmallVec::from_slice(&[src]);
                children.extend(ranges.iter());
                children
            }
            Self::AllReduce(ops::AllReduce { src, .. }) => SmallVec::from_slice(&[src]),

            // Control flow operations
            Self::If(ops::If { condition, body }) => {
                let mut children = SmallVec::from_slice(&[condition]);
                children.extend(body.iter());
                children
            }
            Self::EndIf(ops::EndIf { if_op }) => SmallVec::from_slice(&[if_op]),
            Self::Range(ops::Range { end, deps, .. }) => {
                let mut children = SmallVec::from_slice(&[end]);
                children.extend(deps.iter());
                children
            }
            Self::End(ops::End { computation, ranges }) => {
                let mut children = SmallVec::from_slice(&[computation]);
                children.extend(ranges.iter());
                children
            }
            Self::Barrier(ops::Barrier { src, deps, .. }) => {
                let mut children = SmallVec::from_slice(&[src]);
                children.extend(deps.iter());
                children
            }

            // Shaped value and late vector operations
            Self::Stack(ops::Stack { sources }) => sources.iter().collect(),

            // Symbolic/Define operations
            Self::Bind(ops::Bind { var, value }) => SmallVec::from_slice(&[var, value]),

            // Advanced operations
            Self::Wmma(ops::Wmma { a, b, c, .. }) => SmallVec::from_slice(&[a, b, c]),
            Self::Detach(ops::Detach { src })
            | Self::Contiguous(ops::Contiguous { src, .. })
            | Self::ContiguousBackward(ops::ContiguousBackward { src })
            | Self::Precast(ops::Precast { src }) => SmallVec::from_slice(&[src]),
            Self::Call(ops::Call { body, args, .. }) | Self::Function(ops::Function { body, args, .. }) => {
                let mut children = SmallVec::from_slice(&[body]);
                children.extend(args.iter());
                children
            }
            Self::Tuple(ops::Tuple { src }) => src.iter().collect(),
            Self::GetTuple(ops::GetTuple { src, .. }) => SmallVec::from_slice(&[src]),
            Self::Program(ops::Program { sink, linear, source, binary, .. }) => {
                let mut children = SmallVec::from_slice(&[sink]);
                children.extend(linear.iter());
                children.extend(source.iter());
                children.extend(binary.iter());
                children
            }
            Self::Linear(ops::Linear { ops }) => ops.iter().collect(),
            Self::Ins(ops::Ins { sources, .. }) => sources.iter().collect(),
            Self::After(ops::After { passthrough, deps }) => {
                let mut children = SmallVec::from_slice(&[passthrough]);
                children.extend(deps.iter());
                children
            }
            Self::Custom(ops::Custom { deps, .. }) | Self::CustomI(ops::CustomI { deps, .. }) => deps.iter().collect(),
            Self::CustomFunction(ops::CustomFunction { attrs, .. }) => attrs.iter().collect(),

            // Memory operations
            Self::Load(ops::Load { index, alt, gate }) => {
                let mut children = SmallVec::from_slice(&[index]);
                children.extend(alt);
                children.extend(gate);
                children
            }
            Self::Store(ops::Store { index, value, gate }) => {
                let mut children = SmallVec::from_slice(&[index, value]);
                children.extend(gate);
                children
            }
        }
    }

    /// Get all child UOps as a Vec of owned Rcs (cloned).
    ///
    /// Similar to `children()` but returns owned Rcs instead of references.
    /// Useful when you need to reconstruct nodes or store sources.
    pub fn sources(&self) -> SmallVec<[Arc<UOp>; 4]> {
        self.children().iter().map(|rc| (*rc).clone()).collect()
    }

    /// Apply a function to each child UOp.
    pub fn map_child<F>(&self, mut f: F)
    where
        F: FnMut(&Arc<UOp>),
    {
        for child in self.children() {
            f(child);
        }
    }

    /// Check if this operation is a movement operation.
    ///
    /// Movement operations transform tensor shapes without changing data values:
    /// - RESHAPE: Change shape with same number of elements
    /// - PERMUTE: Transpose/reorder axes
    /// - EXPAND: Broadcast to larger shape
    /// - PAD: Add padding around tensor
    /// - SHRINK: Extract sub-region
    /// - FLIP: Reverse along axes
    ///
    /// Note: MULTI is not considered a pure movement op as it has different semantics.
    pub fn is_movement(&self) -> bool {
        matches!(
            self,
            Self::Reshape(..)
                | Self::Permute(..)
                | Self::Expand(..)
                | Self::Pad(..)
                | Self::Shrink(..)
                | Self::Flip(..)
        )
    }

    /// Get the source index where ranges start being "ended" by this operation.
    ///
    /// Based on Tinygrad's `range_start` dict (ops.py:28).
    /// Returns `Some(index)` if this operation ends ranges, `None` otherwise.
    ///
    /// Operations that end ranges:
    /// - STAGE: ranges start at index 1 (compute is 0, ranges are 1+)
    /// - REDUCE: ranges start at index 1 (src is 0, ranges are 1+)
    /// - WMMA: ranges start at index 3 (a=0, b=1, c=2)
    /// - END: ranges start at index 1 (computation=0, ranges=1+)
    /// - CALL/FUNCTION: ranges start at index 1 (body=0, args=1+)
    ///
    /// These operations mark range boundaries in the computation graph.
    /// Any RANGE operations in sources at or after the returned index
    /// are considered "ended" and removed from scope.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use svod_ir::Op;
    ///
    /// // STAGE ends ranges starting at source index 1
    /// let stage_op = Op::Stage(ops::Stage { /* ... */ });
    /// assert_eq!(stage_op.range_ending_src_index(), Some(1));
    ///
    /// // Regular arithmetic operations don't end ranges
    /// let binary_op = Op::Binary(/* ... */);
    /// assert_eq!(binary_op.range_ending_src_index(), None);
    /// ```
    pub fn range_ending_src_index(&self) -> Option<usize> {
        // Source layout for range-ending ops:
        // - STAGE: compute=0, ranges=1+
        // - REDUCE: src=0, ranges=1+
        // - WMMA: a=0, b=1, c=2, (ranges start at 3)
        // - END: computation=0, ranges=1+
        // - CALL/FUNCTION: body=0, args=1+
        match self {
            Self::Stage(..) => Some(1),
            Self::Reduce(..) => Some(1),
            Self::Wmma(..) => Some(3),
            Self::End(..) => Some(1),
            Self::Call(..) | Self::Function(..) => Some(1),
            _ => None,
        }
    }

    /// Check if this operation participates in shaped range expansion.
    ///
    /// The expander handles:
    /// - ALU ops (Unary, Binary, Ternary)
    /// - Type ops (Cast, BitCast)
    /// - Shaped values (Stack, Index)
    /// - Tensor core ops (Wmma)
    /// - Memory ops (Load, Store, Index)
    /// - Buffer ops (Stage)
    /// - Control flow (Reduce, End, After)
    ///
    /// These operations propagate expanded lanes through the computation graph.
    pub fn is_expandable(&self) -> bool {
        matches!(
            self,
            // ALU operations
            Self::Unary(..) | Self::Binary(..) | Self::Ternary(..) |
            // Type operations
            Self::Cast(..) | Self::BitCast(..) |
            Self::Stack(..) |
            // Tensor core
            Self::Wmma(..) |
            // Memory operations
            Self::Load(..) | Self::Store(..) |
            Self::Index(..) |
            // Buffer operations
            Self::Stage(..) |
            // Control flow (range-ending ops)
            Self::Reduce(..) | Self::End(..) | Self::After(..)
        )
    }

    /// Get the "ended ranges" for this operation.
    ///
    /// These are the RANGE operations (and operations containing ranges)
    /// that should be removed from scope after this operation.
    ///
    /// Based on Tinygrad's `ended_ranges` property (ops.py:296-299).
    ///
    /// # Returns
    ///
    /// A SmallVec of references to child UOps that represent ended ranges.
    /// For operations that don't end ranges, returns an empty SmallVec.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use svod_ir::{Op, UOp};
    ///
    /// // END operation ends its range arguments
    /// let range = UOp::range(/* ... */);
    /// let computation = UOp::const_(/* ... */);
    /// let end_op = computation.end(vec![range.clone()]);
    ///
    /// // ended_ranges() returns the ranges that are closed
    /// let ended = end_op.op().ended_ranges();
    /// assert_eq!(ended.len(), 1);
    /// ```
    pub fn ended_ranges(&self) -> SmallVec<[&Arc<UOp>; 4]> {
        if let Some(start_idx) = self.range_ending_src_index() {
            let children = self.children();
            children.into_iter().skip(start_idx).collect()
        } else if let Self::After(ops::After { deps, .. }) = self {
            // Tinygrad (ops.py:312): flatten([x.ended_ranges for x in self.src[1:]])
            // AFTER propagates ended ranges from its dependency chain.
            let mut result = SmallVec::new();
            for dep in deps {
                result.extend(dep.op().ended_ranges());
            }
            result
        } else if matches!(self, Self::Copy(..) | Self::Slice(..)) {
            // Tinygrad (ops.py:314): return self.src[0].ranges
            // COPY/SLICE ends all ranges from the source.
            // We return the source itself (not individual ranges) — the
            // InScopeRangesProperty handles the non-RANGE branch by looking
            // up the ended node's in_scope_ranges and removing them all.
            let children = self.children();
            if children.is_empty() { SmallVec::new() } else { SmallVec::from_elem(children[0], 1) }
        } else {
            SmallVec::new()
        }
    }
}

/// Number of `u64` words needed to hold one bit per [`pattern_derived::OpKey`].
const OP_MASK_WORDS: usize = pattern_derived::OP_KEY_COUNT.div_ceil(64);

/// Set of operation kinds, one bit per [`pattern_derived::OpKey`].
///
/// Backs the pattern matcher's early reject: every UOp caches the mask of its direct
/// children's op kinds ([`UOp::src_ops`]), and every compiled pattern entry carries the
/// mask of op kinds its fixed-position sources demand. An entry whose mask is not a
/// subset of the node's mask cannot match, so its closure is never called.
///
/// Tinygrad equivalent: `UPat.early_reject` / `UOp._src_ops` (uop/ops.py:1352, 1480).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct OpMask([u64; OP_MASK_WORDS]);

impl OpMask {
    /// Mask with no op kinds set — never rejects.
    pub const EMPTY: Self = Self([0; OP_MASK_WORDS]);

    /// Every op kind.
    pub const ALL: Self = Self::of_range(0, pattern_derived::OP_KEY_COUNT);

    /// Mask holding exactly `key`.
    #[inline]
    pub const fn of_key(key: &pattern_derived::OpKey) -> Self {
        Self::of_range(key.index(), key.index() + 1)
    }

    /// Mask holding the dense key indices `start..end` — a grouped variant's whole kind
    /// via its `OP_KEY_BASE_*`/`OP_KEY_END_*` bounds.
    pub const fn of_range(start: usize, end: usize) -> Self {
        let mut words = [0; OP_MASK_WORDS];
        let mut index = start;
        while index < end {
            words[index / 64] |= 1 << (index % 64);
            index += 1;
        }
        Self(words)
    }

    /// Mask holding exactly the kind of `op`.
    #[inline]
    pub fn of_op(op: &Op) -> Self {
        Self::of_key(&pattern_derived::OpKey::from_op(op))
    }

    /// Union of two masks.
    #[inline]
    pub const fn union(self, other: Self) -> Self {
        let mut words = self.0;
        let mut word = 0;
        while word < OP_MASK_WORDS {
            words[word] |= other.0[word];
            word += 1;
        }
        Self(words)
    }

    /// Whether `key` is in the mask.
    #[inline]
    pub const fn has(&self, key: &pattern_derived::OpKey) -> bool {
        let index = key.index();
        self.0[index / 64] & (1 << (index % 64)) != 0
    }

    /// Whether every kind in `self` is also in `other` (Tinygrad: `issubset`).
    #[inline]
    pub fn is_subset_of(self, other: Self) -> bool {
        self.0.iter().zip(other.0).all(|(mine, theirs)| mine & !theirs == 0)
    }

    /// Whether no kind is set.
    #[inline]
    pub fn is_empty(self) -> bool {
        self.0.iter().all(|word| *word == 0)
    }

    /// Number of kinds set.
    pub fn count(self) -> usize {
        self.0.iter().map(|word| word.count_ones() as usize).sum()
    }
}

impl FromIterator<pattern_derived::OpKey> for OpMask {
    fn from_iter<I: IntoIterator<Item = pattern_derived::OpKey>>(keys: I) -> Self {
        keys.into_iter().fold(Self::EMPTY, |mask, key| mask.union(Self::of_key(&key)))
    }
}
