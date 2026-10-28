//! Core UOp struct and fundamental operations.
//!
//! This module contains the [`UOp`] struct definition and its core methods
//! for accessing operation data, dtype, shape, and graph traversal.

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use bon::bon;
use smallvec::SmallVec;

use crate::op::Op;
use crate::ops;
use crate::origin::OriginId;
use crate::pattern::{Matcher, RewriteResult};
use crate::shape;
use crate::types::ConstValue;
use svod_dtype::DType;

/// Matcher for `UOp::substitute` — looks up each node in a substitution map.
struct SubstituteMatcher<'a>(&'a HashMap<UOpKey, Arc<UOp>>);

impl Matcher<()> for SubstituteMatcher<'_> {
    fn rewrite(&self, uop: &Arc<UOp>, _ctx: &mut ()) -> RewriteResult {
        match self.0.get(&UOpKey(uop.clone())) {
            Some(replacement) if !Arc::ptr_eq(uop, replacement) => RewriteResult::Rewritten(replacement.clone()),
            _ => RewriteResult::NoMatch,
        }
    }
}

/// Matcher for `UOp::substitute_gated` — substitution with range-scope gating.
///
/// - If a node is in the substitution map, replace it.
/// - If a node's ranges don't overlap with substitution keys, gate (skip subtree).
struct SubstituteGatedMatcher<'a> {
    map: &'a HashMap<UOpKey, Arc<UOp>>,
    range_ids: &'a HashSet<u64>,
}

impl Matcher<()> for SubstituteGatedMatcher<'_> {
    fn rewrite(&self, uop: &Arc<UOp>, _ctx: &mut ()) -> RewriteResult {
        // Direct substitution lookup
        if let Some(replacement) = self.map.get(&UOpKey(uop.clone()))
            && !Arc::ptr_eq(uop, replacement)
        {
            return RewriteResult::Rewritten(replacement.clone());
        }
        // Gate: skip subtrees whose ranges don't overlap with substitution keys.
        if !uop.in_scope_ranges().iter().any(|id| self.range_ids.contains(id)) {
            return RewriteResult::Gate(uop.clone());
        }
        RewriteResult::NoMatch
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TraversalMode {
    Full,
    PreserveCalls,
}

/// Borrowed children — `Op::sources()` would clone every `Arc` just to read it.
fn traversal_sources(node: &Arc<UOp>, mode: TraversalMode) -> SmallVec<[&Arc<UOp>; 4]> {
    if mode == TraversalMode::Full {
        return node.op().children();
    }

    match node.op() {
        Op::Call(ops::Call { args, .. }) | Op::Function(ops::Function { args, .. }) => args.iter().collect(),
        // PROGRAM is opaque when preserving call bodies.
        Op::Program(..) => SmallVec::new(),
        _ => node.op().children(),
    }
}

/// Sorted, deduplicated RANGE ids; inline for the common handful of loops.
pub type RangeIds = SmallVec<[u64; 4]>;

/// Wrapper for `Arc<UOp>` that implements Hash and Eq based on stable ID.
///
/// This allows using `Arc<UOp>` as HashMap keys without implementing
/// Hash/Eq on UOp itself (which would be problematic due to OnceCell fields).
///
/// Note: While UOp contains OnceCell fields, Hash/Eq are based solely on the
/// immutable `id` field, making this safe to use as a HashMap key.
#[derive(Clone)]
pub struct UOpKey(pub Arc<UOp>);

// Custom Debug impl to show only the UOp ID, avoiding recursive printing
impl std::fmt::Debug for UOpKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "UOpKey(id={})", self.0.id)
    }
}

impl PartialEq for UOpKey {
    fn eq(&self, other: &Self) -> bool {
        self.0.id == other.0.id
    }
}

impl Eq for UOpKey {}

impl Hash for UOpKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.id.hash(state);
    }
}

/// Micro-operation node in the computation graph.
///
/// UOps form a DAG where operations reference their inputs through the Op enum.
/// Hash consing ensures that structurally identical UOps share the same allocation.
///
/// Shape inference is lazy and cached - computed on first access via `shape()` method.
///
/// Note: Debug uses derive_more with `#[debug(skip)]` on cache fields to prevent
/// stack overflow from recursive `Arc<UOp>` references in caches.
#[derive(derive_more::Debug)]
pub struct UOp {
    /// Unique stable ID for this UOp instance.
    /// Used for identity-based caching instead of fragile raw pointers.
    pub id: u64,
    pub(crate) op: Op,
    pub(crate) dtype: DType,
    /// Cached shape - computed lazily on first access.
    /// OnceLock provides thread-safe lazy initialization.
    #[debug(skip)]
    pub(crate) shape_cache: std::sync::OnceLock<Result<Option<shape::Shape>, Box<crate::error::Error>>>,
    /// Cached list of RANGE operations in this UOp's graph.
    /// Computed lazily via toposort to collect all RANGE ops.
    #[debug(skip)]
    // NOTE: never contains self (see `RangesProperty` — a cached self-`Arc`
    // is a refcount cycle that leaks the node).
    pub(crate) ranges_cache: std::sync::OnceLock<Vec<Arc<UOp>>>,
    /// Cached set of RANGE operation *ids* that are in scope at this UOp.
    /// Unlike ranges_cache which contains ALL ranges in the graph,
    /// this contains only the ranges that are currently "active" (not yet
    /// ended). Ids rather than `Arc`s: a RANGE's own cache entry would
    /// otherwise be a refcount cycle (permanent leak).
    #[debug(skip)]
    pub(crate) in_scope_ranges_cache: std::sync::OnceLock<RangeIds>,
    /// Cached vmin/vmax range analysis values.
    /// Computed lazily via range propagation through the computation graph.
    /// Returns (vmin, vmax) as ConstValue types.
    #[debug(skip)]
    pub(crate) vmin_vmax_cache: std::sync::OnceLock<(ConstValue, ConstValue)>,
    /// Sound vmin/vmax: `None` for ops where range analysis is unsound (LOAD, Pow, etc.).
    /// Used by patterns that must not act on unsound bounds (e.g., vmin_vmax_collapse).
    #[debug(skip)]
    pub(crate) sound_vmin_vmax_cache: std::sync::OnceLock<Option<(ConstValue, ConstValue)>>,
    /// Whether this node or any of its sources is an INDEX op.
    /// Cached O(1) lookup used by `simplify_valid` to skip And chains inside INDEX trees.
    #[debug(skip)]
    pub(crate) has_index_in_sources_cache: std::sync::OnceLock<bool>,
    /// Whether this node or any node in its backward slice has a weak-float dtype.
    /// Cached O(1) lookup used by the value-sensitive symbolic guard.
    #[debug(skip)]
    pub(crate) has_weak_float_cache: std::sync::OnceLock<bool>,
    /// Cached device specification carried by this node's backward slice.
    /// Without the memo the recursive walk is exponential on shared (diamond) DAGs.
    #[debug(skip)]
    pub(crate) device_spec_cache: std::sync::OnceLock<Option<svod_dtype::DeviceSpec>>,
    /// Cached storage address space carried by this value. Same exponential-blowup
    /// reason as `device_spec_cache`.
    #[debug(skip)]
    pub(crate) addrspace_cache: std::sync::OnceLock<Option<svod_dtype::AddrSpace>>,
    /// Structural content hash — deterministic regardless of allocation order.
    /// Computed at creation time: hash(op_discriminant, dtype, op_data, children_content_hashes).
    /// O(1) per node since children are already created with their content_hash set.
    /// Used for schedule-level caching where UOp IDs are not stable across runs.
    pub content_hash: u64,
    /// Set of op kinds among this node's direct children, computed at creation time.
    ///
    /// Drives the pattern matcher's early reject: a compiled pattern whose fixed-position
    /// sources demand an op kind absent here cannot match, so its closure is skipped.
    /// Tinygrad equivalent: `UOp._src_ops` (uop/ops.py:1480), memoised there instead.
    #[debug(skip)]
    pub(crate) src_ops: crate::op::OpMask,
    /// Tag for tracking tensor identity through the rangeify pipeline.
    ///
    /// Tags are sequences of integer indices that track which original tensor
    /// UOps map to which final kernel outputs. They participate in hash consing
    /// — different tag = different UOp.
    ///
    /// Values:
    /// - `None` — untagged (default)
    /// - `Some([])` — empty tag (e.g., RANGE ops)
    /// - `Some([i])` — single index (assigned by add_tags)
    /// - `Some([i, j, ...])` — merged indices (from buffer folding)
    pub tag: Option<SmallVec<[usize; 2]>>,
    /// Scope this node was built under (see [`crate::origin`]).
    ///
    /// Unlike `tag` it participates in `content_hash`, so a cache keyed on the
    /// content hash can never serve one scope's plan to another. `None` while
    /// capture is off, which makes hashes identical to an origin-free build.
    pub(crate) origin: Option<OriginId>,
    /// Optional metadata attached to this UOp.
    ///
    /// Metadata is NOT part of hash consing - attaching metadata creates a new UOp
    /// instance with a different ID. This is used for kernel info (name, opts) after
    /// optimization is complete.
    ///
    /// Uses `Arc<dyn Any>` to allow attaching any metadata type without
    /// circular dependencies (e.g., schedule::KernelInfo).
    #[debug(skip)]
    pub(crate) metadata: Option<std::sync::Arc<dyn std::any::Any + Send + Sync>>,
}

/// Hashes by the precomputed structural content hash, so a derived `Hash` on a
/// parent `Op` is O(children) and deterministic across runs.
impl Hash for UOp {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.content_hash);
    }
}

/// Interned identity: structurally equal nodes are the same allocation, so
/// comparing ids is exact and O(1).
impl PartialEq for UOp {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for UOp {}

impl UOp {
    /// Get the operation.
    pub fn op(&self) -> &Op {
        &self.op
    }

    /// Set of op kinds among the direct children (Tinygrad: `UOp._src_ops`).
    pub fn src_ops(&self) -> crate::op::OpMask {
        self.src_ops
    }

    /// Get the data type.
    pub fn dtype(&self) -> DType {
        self.dtype.clone()
    }

    /// Get the tag.
    pub fn tag(&self) -> &Option<SmallVec<[usize; 2]>> {
        &self.tag
    }

    /// Create a new UOp with the given tag. Returns self unchanged if tag is already equal.
    pub fn rtag(self: &Arc<Self>, tag: Option<SmallVec<[usize; 2]>>) -> Arc<Self> {
        if self.tag == tag {
            return self.clone();
        }
        Self::new_with_origin(self.op.clone(), self.dtype.clone(), tag, self.origin)
    }

    /// Create a new UOp with the given tag set.
    pub fn with_tag(self: &Arc<Self>, tag: SmallVec<[usize; 2]>) -> Arc<Self> {
        self.rtag(Some(tag))
    }

    /// Get the origin scope this node was built under.
    #[inline]
    pub fn origin(&self) -> Option<OriginId> {
        self.origin
    }

    /// Re-intern this node under `origin`. Returns self unchanged when equal.
    pub fn rorigin(self: &Arc<Self>, origin: Option<OriginId>) -> Arc<Self> {
        if self.origin == origin {
            return self.clone();
        }
        Self::new_with_origin(self.op.clone(), self.dtype.clone(), self.tag.clone(), origin)
    }

    /// What a kernel rooted here is charged to, in one pass: the nearest attributed
    /// node walking root-first (the toposort is children-first, so it is consumed in
    /// reverse), and every origin the body carries.
    pub fn kernel_attribution(self: &Arc<Self>) -> (Option<OriginId>, crate::origin::OriginSet) {
        let body = self.toposort();
        (body.iter().rev().find_map(|node| node.origin()), body.iter().filter_map(|node| node.origin()).collect())
    }

    /// Rebuild this tree with every origin cleared.
    ///
    /// A kernel body keys the optimizer, BEAM, the compiled-program and the object
    /// cache, so attribution rides the callable instead: two dispatches of the same
    /// computation from different scopes must still share one compiled program.
    /// Nodes whose sources are unchanged and that carry no origin are returned
    /// as-is, so an already origin-free tree hash-conses back to itself.
    pub fn without_origins(self: &Arc<Self>) -> Arc<Self> {
        let mut rebuilt: HashMap<u64, Arc<Self>> = HashMap::new();
        for node in self.toposort() {
            let children = node.op().children();
            let sources: Vec<Arc<Self>> = children.iter().map(|child| rebuilt[&child.id].clone()).collect();
            let moved = children.iter().zip(&sources).any(|(old, new)| !Arc::ptr_eq(old, new));
            let stripped = if moved { node.with_sources(sources) } else { node.clone() };
            rebuilt.insert(node.id, stripped.rorigin(None));
        }
        rebuilt.remove(&self.id).expect("toposort ends at the root")
    }

    /// Check if this UOp has a concrete buffer identity in the graph.
    ///
    /// Returns true for buffer-like identities or RESHAPE/MULTI chains leading to them.
    /// These are already contiguous by definition, so wrapping in CONTIGUOUS is a no-op.
    pub fn has_buffer_identity(&self) -> bool {
        match &self.op {
            Op::Reshape(ops::Reshape { src, .. }) | Op::Multi(ops::Multi { src, .. }) => src.has_buffer_identity(),
            Op::Buffer(..) | Op::Slice(..) | Op::Param(..) => true,
            Op::GetTuple(ops::GetTuple { src, index }) => match src.op() {
                Op::Tuple(ops::Tuple { src: elements }) => {
                    elements.get(*index).is_some_and(|t| t.has_buffer_identity())
                }
                _ => false,
            },
            _ => false,
        }
    }

    /// Get address dtype components from a Ptr dtype or PARAM metadata.
    ///
    /// Returns `(base, addrspace, size)` for address-bearing values, None otherwise.
    /// This simplifies pattern matching on pointer types.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use svod_ir::UOp;
    /// # use svod_dtype::{DType, AddrSpace, DeviceSpec};
    /// let buffer = UOp::new_buffer(DeviceSpec::Cpu, 10, DType::Float32);
    /// if let Some((base, addrspace, size)) = buffer.ptrdtype() {
    ///     assert_eq!(*base, DType::Float32);
    ///     assert_eq!(addrspace, AddrSpace::Global);
    /// }
    /// ```
    pub fn ptrdtype(&self) -> Option<(&DType, svod_dtype::AddrSpace, Option<usize>)> {
        match (&self.dtype, self.op()) {
            (DType::Ptr { base, addrspace, size, .. }, _) => Some((base.as_ref(), *addrspace, *size)),
            (_, Op::Param(ops::Param { arg, .. }) | Op::Buffer(ops::Buffer { arg, .. })) => {
                Some((&arg.dtype, arg.addrspace.unwrap_or(svod_dtype::AddrSpace::Global), None))
            }
            _ => None,
        }
    }

    /// Storage address space carried by this value.
    ///
    /// This is the structured-storage equivalent of Tinygrad's `UOp.addrspace`:
    /// PARAM/BUFFER read it from `ParamArg`, exact address-preserving wrappers
    /// project source zero, and elementwise/shaped containers preserve only a
    /// common non-ALU address space. `None` covers ALU values and operations
    /// without address semantics.
    pub fn addrspace(self: &Arc<Self>) -> Option<svod_dtype::AddrSpace> {
        use crate::uop::cached_property::CachedProperty;
        *crate::uop::properties::AddrSpaceProperty::get(self)
    }

    pub(crate) fn compute_addrspace(self: &Arc<Self>) -> Option<svod_dtype::AddrSpace> {
        let common = |sources: SmallVec<[Arc<UOp>; 4]>| {
            let mut address_spaces = sources.iter().filter_map(|source| source.addrspace());
            let first = address_spaces.next()?;
            address_spaces.all(|address_space| address_space == first).then_some(first)
        };

        match self.op() {
            Op::Param(ops::Param { arg, .. }) | Op::Buffer(ops::Buffer { arg, .. }) => arg.addrspace,
            Op::Index(ops::Index { buffer, .. }) | Op::MSelect(ops::MSelect { buffer, .. }) => buffer.addrspace(),
            Op::Cast(ops::Cast { src, .. })
            | Op::After(ops::After { passthrough: src, .. })
            | Op::Reduce(ops::Reduce { src, .. })
            | Op::End(ops::End { computation: src, .. })
            | Op::Reshape(ops::Reshape { src, .. })
            | Op::Permute(ops::Permute { src, .. })
            | Op::Expand(ops::Expand { src, .. })
            | Op::Pad(ops::Pad { src, .. })
            | Op::Shrink(ops::Shrink { src, .. })
            | Op::Flip(ops::Flip { src, .. })
            | Op::Multi(ops::Multi { src, .. }) => src.addrspace(),
            Op::Store(ops::Store { index, .. }) => index.addrspace(),
            Op::MStack(ops::MStack { buffers }) => buffers.first().and_then(|buffer| buffer.addrspace()),
            Op::Unary(..)
            | Op::Binary(..)
            | Op::Ternary(..)
            | Op::BitCast(..)
            | Op::Stack(..)
            | Op::Wmma(..)
            | Op::Group(..) => common(self.op().sources()),
            _ => None,
        }
    }

    /// Create a copy of this UOp with a different dtype.
    ///
    /// If the dtype is unchanged, returns self (clone of Arc).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use std::sync::Arc;
    /// # use svod_ir::UOp;
    /// # use svod_dtype::DType;
    /// let int_const = UOp::const_(DType::Int32, svod_ir::ConstValue::Int(5));
    /// let float_const = int_const.with_dtype(DType::Float32);
    /// assert_eq!(float_const.dtype(), DType::Float32);
    /// ```
    pub fn with_dtype(self: &Arc<Self>, dtype: DType) -> Arc<Self> {
        if self.dtype == dtype {
            return self.clone();
        }
        Self::new_with_origin(self.op.clone(), dtype, None, self.origin)
    }

    /// Walk through AFTER nodes to get the passthrough value.
    ///
    /// Recursively unwraps AFTER nodes to find the underlying value.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Given: AFTER(AFTER(value, [dep1]), [dep2])
    /// // Returns: value
    /// let inner = wrapped.unwrap_after();
    /// ```
    pub fn unwrap_after(self: &Arc<Self>) -> Arc<Self> {
        match self.op() {
            Op::After(ops::After { passthrough, .. }) => passthrough.unwrap_after(),
            _ => self.clone(),
        }
    }

    /// Walk through CAST nodes to get the inner value.
    ///
    /// Recursively unwraps CAST nodes to find the underlying value.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Given: CAST(CAST(value, dtype1), dtype2)
    /// // Returns: value
    /// let inner = casted.unwrap_cast();
    /// ```
    pub fn unwrap_cast(self: &Arc<Self>) -> Arc<Self> {
        match self.op() {
            Op::Cast(ops::Cast { src, .. }) => src.unwrap_cast(),
            _ => self.clone(),
        }
    }

    /// Get the buffer from a STORE operation (via its INDEX child).
    ///
    /// STORE operations reference the buffer indirectly through an INDEX node.
    /// This helper extracts the buffer from `STORE.index.buffer`.
    ///
    /// Returns `None` if:
    /// - This is not a STORE operation
    /// - The STORE's index is not an INDEX operation
    pub fn store_buffer(&self) -> Option<&Arc<UOp>> {
        match self.op() {
            Op::Store(ops::Store { index, .. }) => match index.op() {
                Op::Index(ops::Index { buffer, .. }) => Some(buffer),
                _ => None,
            },
            _ => None,
        }
    }

    /// Store a value at this INDEX node.
    ///
    /// Convenience method for `self.store(value)`.
    ///
    /// # Panics
    ///
    /// Debug-asserts that self is an INDEX operation.
    pub fn store_value(self: &Arc<Self>, value: Arc<Self>) -> Arc<Self> {
        debug_assert!(matches!(self.op(), Op::Index(..)), "store_value requires INDEX");
        self.store(value)
    }

    /// Alias for `with_sources()`.
    ///
    /// Creates a new UOp with the same operation type and dtype, but with
    /// the provided sources replacing the original ones.
    pub fn with_src(self: &Arc<Self>, new_srcs: Vec<Arc<Self>>) -> Arc<Self> {
        self.with_sources(new_srcs)
    }

    /// Get the shape of this UOp.
    ///
    /// Shape is computed lazily on first access and cached.
    /// Returns Ok(None) if shape cannot be determined (e.g., for control flow ops).
    /// Returns Err if there is a shape mismatch error.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use svod_ir::{UOp, ConstValue};
    /// # use svod_dtype::DType;
    /// let scalar = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    /// assert_eq!(scalar.shape().unwrap().as_ref().map(|s| s.len()), Some(0)); // Scalar has empty shape
    /// ```
    pub fn shape(self: &Arc<Self>) -> crate::Result<Option<&shape::Shape>> {
        use crate::uop::cached_property::CachedProperty;
        use crate::uop::properties::ShapeProperty;
        match ShapeProperty::get(self) {
            Ok(opt) => Ok(opt.as_ref()),
            Err(e) => Err((**e).clone()),
        }
    }

    /// Get the minimum possible value of this UOp.
    ///
    /// Returns the minimum value based on range analysis.
    /// Computed lazily on first access and cached.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use svod_ir::{UOp, ConstValue};
    /// # use svod_dtype::DType;
    /// let five = UOp::const_(DType::Int32, ConstValue::Int(5));
    /// assert_eq!(five.vmin(), &ConstValue::Int(5));
    /// ```
    pub fn vmin(self: &Arc<Self>) -> &ConstValue {
        use crate::uop::cached_property::CachedProperty;
        use crate::uop::properties::VminVmaxProperty;
        &VminVmaxProperty::get(self).0
    }

    /// Get the maximum possible value of this UOp.
    ///
    /// Returns the maximum value based on range analysis.
    /// Computed lazily on first access and cached.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use svod_ir::{UOp, ConstValue};
    /// # use svod_dtype::DType;
    /// let five = UOp::const_(DType::Int32, ConstValue::Int(5));
    /// assert_eq!(five.vmax(), &ConstValue::Int(5));
    /// ```
    pub fn vmax(self: &Arc<Self>) -> &ConstValue {
        use crate::uop::cached_property::CachedProperty;
        use crate::uop::properties::VminVmaxProperty;
        &VminVmaxProperty::get(self).1
    }

    /// Extract device specification from this UOp graph.
    ///
    /// Traverses the graph to find storage or transfer device metadata.
    /// - Otherwise: searches children recursively
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use svod_ir::UOp;
    /// # use svod_dtype::{DType, DeviceSpec};
    /// let buffer = UOp::new_buffer(DeviceSpec::Cpu, 10, DType::Float32);
    /// assert_eq!(buffer.device_spec(), Some(DeviceSpec::Cpu));
    /// ```
    pub fn device_spec(self: &Arc<Self>) -> Option<svod_dtype::DeviceSpec> {
        use crate::uop::cached_property::CachedProperty;
        crate::uop::properties::DeviceSpecProperty::get(self).clone()
    }

    pub(crate) fn compute_device_spec(self: &Arc<Self>) -> Option<svod_dtype::DeviceSpec> {
        match self.op() {
            Op::Buffer(ops::Buffer { arg, .. }) | Op::Param(ops::Param { arg, .. }) => arg.device.clone(),
            Op::Copy(ops::Copy { device, .. }) | Op::AllReduce(ops::AllReduce { device, .. }) => Some(device.clone()),
            // Children are memoised, so this is a single level, not a re-walk.
            _ => self.op().children().iter().find_map(|child| child.device_spec()),
        }
    }

    /// Concrete element count encoded by a PARAM/BUFFER's single shape source.
    pub fn buffer_size(self: &Arc<Self>) -> Option<usize> {
        self.shape().ok().flatten()?.iter().try_fold(1usize, |size, dim| match dim {
            crate::SInt::Const(value) => size.checked_mul(*value),
            crate::SInt::Symbolic(_) | crate::SInt::Infer => None,
        })
    }

    /// Get the base UOp by walking through movement operations.
    ///
    /// Movement operations (RESHAPE, PERMUTE, EXPAND, etc.) are views that don't
    /// change the underlying data. This method recursively walks through these
    /// operations to find the actual buffer or computation that owns the data.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use svod_ir::{UOp, SInt, shape::Shape};
    /// # use svod_dtype::DType;
    /// # use svod_dtype::DeviceSpec;
    /// let buffer = UOp::new_buffer(DeviceSpec::Cpu, 10, DType::Float32);
    /// let shape = Shape::from_iter([SInt::Const(2), SInt::Const(5)]);
    /// let reshaped = buffer.try_reshape(&shape).unwrap();
    ///
    /// // base() walks through RESHAPE to get the original BUFFER
    /// assert!(std::sync::Arc::ptr_eq(&reshaped.base(), &buffer));
    /// ```
    pub fn base(self: &Arc<Self>) -> Arc<Self> {
        match &self.op {
            // Movement operations - recursively get base of source
            Op::Reshape(ops::Reshape { src, .. })
            | Op::Permute(ops::Permute { src, .. })
            | Op::Expand(ops::Expand { src, .. })
            | Op::Pad(ops::Pad { src, .. })
            | Op::Shrink(ops::Shrink { src, .. })
            | Op::Flip(ops::Flip { src, .. })
            | Op::Multi(ops::Multi { src, .. }) => src.base(),
            // All other operations are their own base
            _ => self.clone(),
        }
    }

    /// Get the underlying buffer UOp, walking through AFTER/MSELECT/MSTACK chains.
    ///
    /// Recursively unwraps AFTER chains to find the actual buffer.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use svod_ir::UOp;
    ///
    /// // AFTER wrapping a buffer
    /// let buffer = UOp::new_buffer(...);
    /// let after = buffer.after(deps);
    ///
    /// // buf_uop() walks through AFTER to get the underlying buffer
    /// assert!(Arc::ptr_eq(&after.buf_uop(), &buffer));
    /// ```
    pub fn buf_uop(self: &Arc<Self>) -> Arc<Self> {
        match self.op() {
            Op::Buffer(..) => self.clone(),
            Op::MSelect(ops::MSelect { buffer, .. }) => buffer.buf_uop(),
            Op::MStack(ops::MStack { buffers }) if !buffers.is_empty() => buffers[0].buf_uop(),
            Op::After(ops::After { passthrough, .. }) => passthrough.buf_uop(),
            Op::Call(ops::Call { body, .. }) | Op::Function(ops::Function { body, .. }) => body.buf_uop(),
            _ => {
                // For other ops, check if base is AFTER
                let base = self.base();
                if matches!(base.op(), Op::After(..)) { base.buf_uop() } else { self.clone() }
            }
        }
    }

    /// Topological sort of the computation graph.
    ///
    /// Returns nodes in an order where all dependencies come before their dependents.
    pub fn toposort(self: &Arc<Self>) -> Vec<Arc<Self>> {
        let mut visited = visited_set(FULL_GRAPH_HINT);
        let mut result = Vec::new();
        let mut stack = vec![(self.clone(), false)];

        while let Some((node, processed)) = stack.pop() {
            let ptr = Arc::as_ptr(&node);

            if visited.contains(&ptr) {
                continue;
            }

            if processed {
                visited.insert(ptr);
                result.push(node);
            } else {
                stack.push((node.clone(), true));

                // Use for_each_child for zero-allocation traversal
                let mut children = Vec::new();
                node.op.map_child(|child| {
                    if !visited.contains(&Arc::as_ptr(child)) {
                        children.push(child.clone());
                    }
                });

                // Push in reverse order for proper traversal
                for child in children.into_iter().rev() {
                    stack.push((child, false));
                }
            }
        }

        result
    }

    /// Topological sort with gate function (filtered toposort).
    ///
    /// Only traverses nodes for which `gate(node)` returns true.
    /// Nodes for which gate returns false are excluded from the
    /// traversal entirely (along with their ancestors).
    ///
    /// This is a key optimization for cached property computation,
    /// allowing us to skip nodes that already have a property cached.
    ///
    /// # Performance
    ///
    /// For a graph with 10,000 nodes where 9,900 already have a cached property:
    /// - **Full toposort**: 10,000 nodes visited
    /// - **Filtered toposort**: 100 nodes visited
    /// - **Speedup**: 100x
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Only process nodes that don't have shape cached
    /// let uncached = uop.toposort_filtered(|node| {
    ///     node.shape_cache.get().is_none()
    /// });
    /// ```
    pub fn toposort_filtered<F>(self: &Arc<Self>, gate: F) -> Vec<Arc<Self>>
    where
        F: Fn(&Arc<UOp>) -> bool,
    {
        let mut visited = visited_set(LOCAL_HINT);
        let mut result = Vec::new();
        let mut stack = vec![(self.clone(), false)];

        while let Some((node, processed)) = stack.pop() {
            let ptr = Arc::as_ptr(&node);

            if visited.contains(&ptr) {
                continue;
            }

            if processed {
                visited.insert(ptr);
                result.push(node);
            } else {
                // Key optimization: only traverse nodes that pass the gate
                if gate(&node) {
                    stack.push((node.clone(), true));

                    let mut children = Vec::new();
                    node.op.map_child(|child| {
                        if !visited.contains(&Arc::as_ptr(child)) {
                            children.push(child.clone());
                        }
                    });

                    // Push in reverse order for proper traversal
                    for child in children.into_iter().rev() {
                        stack.push((child, false));
                    }
                }
            }
        }

        result
    }

    /// Topological sort with optional CALL/FUNCTION/PROGRAM boundary traversal.
    ///
    /// When `include_call_bodies` is false, traversal does not descend into
    /// CALL/FUNCTION bodies or PROGRAM internals. Call/function arguments and
    /// program device are
    /// still traversed.
    pub fn toposort_call_aware(self: &Arc<Self>, include_call_bodies: bool) -> Vec<Arc<Self>> {
        let mode = if include_call_bodies { TraversalMode::Full } else { TraversalMode::PreserveCalls };

        let mut visited = visited_set(FULL_GRAPH_HINT);
        let mut result = Vec::new();
        let mut stack = vec![(self.clone(), false)];

        while let Some((node, processed)) = stack.pop() {
            let ptr = Arc::as_ptr(&node);

            if visited.contains(&ptr) {
                continue;
            }

            if processed {
                visited.insert(ptr);
                result.push(node);
            } else {
                stack.push((node.clone(), true));
                for child in traversal_sources(&node, mode).into_iter().rev() {
                    if !visited.contains(&Arc::as_ptr(child)) {
                        stack.push((child.clone(), false));
                    }
                }
            }
        }

        result
    }

    /// Filtered topological sort with optional CALL/FUNCTION/PROGRAM boundary traversal.
    pub fn toposort_filtered_call_aware<F>(self: &Arc<Self>, gate: F, include_call_bodies: bool) -> Vec<Arc<Self>>
    where
        F: Fn(&Arc<UOp>) -> bool,
    {
        let mode = if include_call_bodies { TraversalMode::Full } else { TraversalMode::PreserveCalls };

        let mut visited = visited_set(LOCAL_HINT);
        let mut result = Vec::new();
        let mut stack = vec![(self.clone(), false)];

        while let Some((node, processed)) = stack.pop() {
            let ptr = Arc::as_ptr(&node);

            if visited.contains(&ptr) {
                continue;
            }

            if processed {
                visited.insert(ptr);
                result.push(node);
            } else if gate(&node) {
                stack.push((node.clone(), true));
                for child in traversal_sources(&node, mode).into_iter().rev() {
                    if !visited.contains(&Arc::as_ptr(child)) {
                        stack.push((child.clone(), false));
                    }
                }
            }
        }

        result
    }

    /// Check if any node in the backward slice satisfies a predicate.
    ///
    /// Early-exit DFS — returns `true` as soon as a matching node is found,
    /// without building the full toposort Vec. Use this instead of
    /// `toposort().iter().any(pred)` when you only need an existential check.
    pub fn any_in_subtree<F>(self: &Arc<Self>, pred: F) -> bool
    where
        F: Fn(&Arc<UOp>) -> bool,
    {
        let mut visited = visited_set(FULL_GRAPH_HINT);
        let mut stack = vec![self.clone()];
        while let Some(node) = stack.pop() {
            if !visited.insert(Arc::as_ptr(&node)) {
                continue;
            }
            if pred(&node) {
                return true;
            }
            node.op.map_child(|child| {
                if !visited.contains(&Arc::as_ptr(child)) {
                    stack.push(child.clone());
                }
            });
        }
        false
    }

    /// Collect all nodes in the backward slice that match a predicate.
    ///
    /// DFS collecting matches — cheaper than `toposort().iter().filter(pred).collect()`
    /// when you don't need topological ordering.
    pub fn collect_in_subtree<F>(self: &Arc<Self>, pred: F) -> Vec<Arc<UOp>>
    where
        F: Fn(&Arc<UOp>) -> bool,
    {
        let mut visited = visited_set(FULL_GRAPH_HINT);
        let mut stack = vec![self.clone()];
        let mut result = Vec::new();
        while let Some(node) = stack.pop() {
            if !visited.insert(Arc::as_ptr(&node)) {
                continue;
            }
            if pred(&node) {
                result.push(node.clone());
            }
            node.op.map_child(|child| {
                if !visited.contains(&Arc::as_ptr(child)) {
                    stack.push(child.clone());
                }
            });
        }
        result
    }

    /// Count unique nodes in the DAG rooted at this UOp.
    ///
    /// Much cheaper than `toposort().len()` — no result Vec, no ordering.
    /// Uses pointer-based visited set for O(1) identity checks.
    pub fn node_count(self: &Arc<Self>) -> usize {
        let mut visited = visited_set(FULL_GRAPH_HINT);
        let mut stack = vec![self.clone()];
        while let Some(node) = stack.pop() {
            if !visited.insert(Arc::as_ptr(&node)) {
                continue;
            }
            node.op.map_child(|child| {
                if !visited.contains(&Arc::as_ptr(child)) {
                    stack.push(child.clone());
                }
            });
        }
        visited.len()
    }

    /// O(1) cached check: does this node or any of its sources contain an INDEX op?
    ///
    /// Computed lazily and cached. Each node checks itself and its direct sources'
    /// cached values, so the total cost across the graph is O(N).
    pub fn has_index_in_sources(self: &Arc<Self>) -> bool {
        *self.has_index_in_sources_cache.get_or_init(|| {
            if matches!(self.op, Op::Index(..)) {
                return true;
            }
            let mut result = false;
            self.op.map_child(|child| {
                if child.has_index_in_sources() {
                    result = true;
                }
            });
            result
        })
    }

    /// Render this UOp and its sources as a compact ASCII tree.
    ///
    /// Shared nodes (appearing multiple times due to hash-consing) are shown
    /// as back-references: `[id] → (see above)`
    ///
    /// # Example Output
    ///
    /// ```text
    /// [42] STORE : Void
    /// ├── [10] PARAM(0) : Ptr<Float32> shape=[4]
    /// ├── [35] INDEX : Ptr<Float32> shape=[4]
    /// │   ├── [10] → (see above)
    /// │   └── [30] RANGE(0, Reduce) : Index
    /// │       └── [5] CONST(Int(4)) : Index
    /// └── [40] REDUCE(Add) : Float32 shape=[]
    ///     └── [35] → (see above)
    /// ```
    pub fn tree(self: &Arc<Self>) -> String {
        crate::uop::tree::render_tree_compact(self)
    }

    /// Render this UOp and its sources as a full ASCII tree.
    ///
    /// Shared nodes are expanded every time they appear (verbose but complete).
    /// Use this when you need to see the full subtree at every occurrence.
    pub fn tree_full(self: &Arc<Self>) -> String {
        crate::uop::tree::render_tree_full(self)
    }

    /// Get all RANGE operations in this UOp's computation graph (self first
    /// when self is a RANGE, matching Tinygrad's `{self} | self._ranges`).
    ///
    /// Backed by a per-node cache that deliberately excludes self (a cached
    /// self-`Arc` would be a refcount cycle), so the RANGE case chains self
    /// lazily and the method returns an owned `Vec`.
    pub fn ranges(self: &Arc<Self>) -> Vec<Arc<Self>> {
        use crate::uop::cached_property::CachedProperty;
        use crate::uop::properties::RangesProperty;
        let cached = RangesProperty::get(self);
        if matches!(self.op, Op::Range(..)) {
            std::iter::once(self.clone()).chain(cached.iter().cloned()).collect()
        } else {
            cached.clone()
        }
    }

    /// Get the RANGE operations that are in scope at this UOp.
    ///
    /// Returns only the ranges that are currently "active" (not yet ended).
    /// This is computed by:
    /// 1. Merging ranges from all source operations
    /// 2. Removing ranges that are ended by this operation
    /// 3. Adding self if this is a RANGE operation
    ///
    /// # Returns
    ///
    /// A HashSet of RANGE UOp *ids* in scope at this point in the graph.
    /// Ids, not `Arc`s: a RANGE storing a self-`Arc` in its own cache would
    /// be a refcount cycle (permanent leak). The result is cached.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use svod_ir::{UOp, AxisType};
    ///
    /// // A simple computation inside a range
    /// let range = UOp::range(end, 0, AxisType::Loop);
    /// let value = UOp::const_(...);
    /// let end_op = value.end(vec![range.clone()]);
    ///
    /// // Value has range in scope
    /// assert!(value.in_scope_ranges().contains(&range.id));
    ///
    /// // After END, range is no longer in scope
    /// assert!(!end_op.in_scope_ranges().contains(&range.id));
    /// ```
    pub fn in_scope_ranges(self: &Arc<Self>) -> &[u64] {
        use crate::uop::cached_property::CachedProperty;
        use crate::uop::properties::InScopeRangesProperty;
        InScopeRangesProperty::get(self)
    }

    /// Build a consumer map for this UOp's computation graph.
    ///
    /// Returns a HashMap where each UOp maps to the list of UOps that consume it.
    /// Useful for reverse traversal and dependency analysis.
    pub fn get_consumer_map(self: &Arc<Self>) -> HashMap<UOpKey, Vec<Arc<Self>>> {
        self.get_consumer_map_call_aware(true)
    }

    /// Build a consumer map with optional CALL/FUNCTION/PROGRAM boundary traversal.
    ///
    /// When `include_call_bodies` is false, traversal does not descend into
    /// CALL/FUNCTION bodies or PROGRAM internals. Call/function arguments and
    /// program device are still traversed.
    pub fn get_consumer_map_call_aware(self: &Arc<Self>, include_call_bodies: bool) -> HashMap<UOpKey, Vec<Arc<Self>>> {
        let mut consumer_map: HashMap<UOpKey, Vec<Arc<Self>>> = HashMap::new();
        let mode = if include_call_bodies { TraversalMode::Full } else { TraversalMode::PreserveCalls };

        for node in self.toposort_call_aware(include_call_bodies) {
            for child in traversal_sources(&node, mode) {
                consumer_map.entry(UOpKey(child.clone())).or_default().push(node.clone());
            }
        }

        consumer_map
    }

    /// Reverse topological sort of the computation graph.
    ///
    /// Returns nodes in bottom-up order (leaves first, root last).
    /// Requires a consumer map to traverse from leaves to roots.
    pub fn reverse_toposort(self: &Arc<Self>, consumer_map: &HashMap<UOpKey, Vec<Arc<Self>>>) -> Vec<Arc<Self>> {
        let mut visited = HashMap::new(); // Use HashMap to track visited by ID
        let mut result = Vec::new();
        let mut stack = vec![(self.clone(), false)];

        while let Some((node, processed)) = stack.pop() {
            if visited.contains_key(&node.id) {
                continue;
            }

            if processed {
                visited.insert(node.id, ());
                result.push(node);
            } else {
                stack.push((node.clone(), true));

                // Visit consumers (nodes that depend on this node)
                if let Some(consumers) = consumer_map.get(&UOpKey(node.clone())) {
                    for consumer in consumers {
                        if !visited.contains_key(&consumer.id) {
                            stack.push((consumer.clone(), false));
                        }
                    }
                }
            }
        }

        result
    }

    /// Replace UOps in the computation graph according to a substitution map.
    ///
    /// Delegates to `graph_rewrite_bottom_up` with a wildcard pattern that looks up
    /// each node in the map. The rewrite engine provides O(n) memoization via its
    /// result cache and an explicit work-stack (no Rust recursion, so deep graphs
    /// do not exhaust the thread stack).
    pub fn substitute(self: &Arc<Self>, map: &HashMap<UOpKey, Arc<Self>>) -> Arc<Self> {
        if map.is_empty() {
            return self.clone();
        }
        let matcher = SubstituteMatcher(map);
        crate::rewrite::graph_rewrite_bottom_up(&matcher, self.clone(), &mut ())
    }

    /// Replace UOps using walk semantics — single-pass, no re-traversal into
    /// rewritten subtrees.
    ///
    /// Use when a replacement may contain the original key (e.g.
    /// `Buffer → After(Buffer, [Store(...)])` for view-assign). The default
    /// [`Self::substitute`] would re-traverse replacements and loop or wrap
    /// the key multiple times.
    pub fn substitute_walk(self: &Arc<Self>, map: &HashMap<UOpKey, Arc<Self>>) -> Arc<Self> {
        if map.is_empty() {
            return self.clone();
        }
        let matcher = SubstituteMatcher(map);
        crate::rewrite::graph_rewrite_walk(&matcher, self.clone(), &mut ())
    }

    /// Single-pass substitution that also preserves opaque callable bodies.
    pub fn substitute_walk_preserve_calls(self: &Arc<Self>, map: &HashMap<UOpKey, Arc<Self>>) -> Arc<Self> {
        if map.is_empty() {
            return self.clone();
        }
        let matcher = SubstituteMatcher(map);
        crate::rewrite::graph_rewrite_walk_preserve_calls(&matcher, self.clone(), &mut ())
    }

    /// Replace UOps while preserving CALL/FUNCTION/PROGRAM body boundaries.
    ///
    /// Direct substitutions still apply to CALL/FUNCTION/PROGRAM nodes themselves.
    /// Traversal skips CALL/FUNCTION bodies and PROGRAM internals by default,
    /// while still rewriting CALL/FUNCTION arguments.
    pub fn substitute_preserve_calls(self: &Arc<Self>, map: &HashMap<UOpKey, Arc<Self>>) -> Arc<Self> {
        if map.is_empty() {
            return self.clone();
        }
        let matcher = SubstituteMatcher(map);
        crate::rewrite::graph_rewrite_bottom_up_preserve_calls(&matcher, self.clone(), &mut ())
    }

    /// Replace UOps with range-gated substitution.
    ///
    /// Like `substitute`, but skips subtrees whose `in_scope_ranges()` don't contain
    /// any of the substitution keys. Prevents substituting ranges in subexpressions
    /// that don't reference them.
    pub fn substitute_gated(self: &Arc<Self>, map: &HashMap<UOpKey, Arc<Self>>) -> Arc<Self> {
        if map.is_empty() {
            return self.clone();
        }
        let range_ids: HashSet<u64> = map.keys().map(|key| key.0.id).collect();
        let matcher = SubstituteGatedMatcher { map, range_ids: &range_ids };
        crate::rewrite::graph_rewrite_bottom_up(&matcher, self.clone(), &mut ())
    }

    /// Range-gated substitute that also preserves CALL/FUNCTION/PROGRAM boundaries.
    pub fn substitute_gated_preserve_calls(self: &Arc<Self>, map: &HashMap<UOpKey, Arc<Self>>) -> Arc<Self> {
        if map.is_empty() {
            return self.clone();
        }
        let range_ids: HashSet<u64> = map.keys().map(|key| key.0.id).collect();
        let matcher = SubstituteGatedMatcher { map, range_ids: &range_ids };
        crate::rewrite::graph_rewrite_bottom_up_preserve_calls(&matcher, self.clone(), &mut ())
    }

    /// Reconstruct this UOp with new sources.
    ///
    /// Creates a new UOp with the same operation type and dtype, but with the provided
    /// sources replacing the original ones. Hash consing ensures that if an identical
    /// UOp already exists, it will be reused.
    ///
    /// This is used by the graph rewrite engine when sources have been rewritten.
    ///
    /// # Panics
    ///
    /// Panics if the number of sources doesn't match the operation's arity.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Original: a + b
    /// let add = UOp::add(a.clone(), b.clone());
    ///
    /// // Rewrite sources: a' + b'
    /// let new_add = add.with_sources(vec![a_prime, b_prime]);
    /// ```
    pub fn with_sources(self: &Arc<Self>, new_srcs: Vec<Arc<Self>>) -> Arc<Self> {
        use smallvec::SmallVec;

        // Helper to get nth source
        let src = |n: usize| new_srcs[n].clone();

        let new_op = match &self.op {
            // Nullary operations - no sources
            Op::Const(_)
            | Op::Unique(_)
            | Op::LUnique(_)
            | Op::Noop
            | Op::VConst(..)
            | Op::DefineVar(..)
            | Op::Source(..)
            | Op::ProgramBinary(..) => {
                assert_eq!(new_srcs.len(), 0, "Nullary op should have no sources");
                return self.clone(); // No sources to replace
            }

            // Unary operations
            Op::Unary(op_type, _) => {
                assert_eq!(new_srcs.len(), 1);
                Op::Unary(*op_type, src(0))
            }

            // Binary operations
            Op::Binary(op_type, _, _) => {
                assert_eq!(new_srcs.len(), 2);
                Op::Binary(*op_type, src(0), src(1))
            }

            // Ternary operations
            Op::Ternary(op_type, _, _, _) => {
                assert_eq!(new_srcs.len(), 3);
                Op::Ternary(*op_type, src(0), src(1), src(2))
            }

            // Type operations
            Op::Cast(ops::Cast { dtype, .. }) => {
                assert_eq!(new_srcs.len(), 1);
                Op::Cast(ops::Cast { src: src(0), dtype: dtype.clone() })
            }
            Op::BitCast(ops::BitCast { dtype, .. }) => {
                assert_eq!(new_srcs.len(), 1);
                Op::BitCast(ops::BitCast { src: src(0), dtype: dtype.clone() })
            }
            Op::GetAddr(ops::GetAddr { device, .. }) => {
                assert_eq!(new_srcs.len(), 1);
                Op::GetAddr(ops::GetAddr { src: src(0), device: device.clone() })
            }

            // Special operations
            Op::MSelect(ops::MSelect { device_index, .. }) => {
                assert_eq!(new_srcs.len(), 1);
                Op::MSelect(ops::MSelect { buffer: src(0), device_index: *device_index })
            }
            Op::Special(ops::Special { name, .. }) => {
                assert_eq!(new_srcs.len(), 1);
                Op::Special(ops::Special { end: src(0), name: name.clone() })
            }

            // Buffer operations
            Op::Buffer(ops::Buffer { arg, .. }) => {
                assert_eq!(new_srcs.len(), 1);
                Op::Buffer(ops::Buffer { shape: src(0), arg: arg.clone() })
            }
            Op::Param(ops::Param { arg, .. }) => {
                assert_eq!(new_srcs.len(), 1);
                Op::Param(ops::Param { shape: src(0), arg: arg.clone() })
            }
            Op::Slice(ops::Slice { size, .. }) => {
                assert_eq!(new_srcs.len(), 2);
                Op::Slice(ops::Slice { buffer: src(0), offset: src(1), size: *size })
            }
            Op::Stage(ops::Stage { opts, .. }) => {
                assert!(!new_srcs.is_empty());
                Op::Stage(ops::Stage {
                    compute: src(0),
                    ranges: new_srcs[1..].iter().cloned().collect(),
                    opts: opts.clone(),
                })
            }
            Op::Index(..) => {
                assert!(!new_srcs.is_empty());
                let buffer = src(0);
                let indices: SmallVec<[Arc<Self>; 4]> = new_srcs[1..].iter().cloned().collect();
                Op::Index(ops::Index { buffer, indices })
            }
            Op::Copy(ops::Copy { device, .. }) => {
                assert_eq!(new_srcs.len(), 1);
                Op::Copy(ops::Copy { src: src(0), device: device.clone() })
            }
            Op::MStack(..) => Op::MStack(ops::MStack { buffers: new_srcs.iter().cloned().collect() }),

            // Movement operations
            Op::Reshape(..) => {
                assert_eq!(new_srcs.len(), 2);
                Op::Reshape(ops::Reshape { src: src(0), new_shape: src(1) })
            }
            Op::Permute(ops::Permute { axes, .. }) => {
                assert_eq!(new_srcs.len(), 1);
                Op::Permute(ops::Permute { src: src(0), axes: axes.clone() })
            }
            Op::Expand(..) => {
                assert_eq!(new_srcs.len(), 2);
                Op::Expand(ops::Expand { src: src(0), new_shape: src(1) })
            }
            Op::Pad(..) => {
                assert_eq!(new_srcs.len(), 3);
                Op::Pad(ops::Pad { src: src(0), begin_pads: src(1), end_pads: src(2) })
            }
            Op::Shrink(..) => {
                assert_eq!(new_srcs.len(), 3);
                Op::Shrink(ops::Shrink { src: src(0), offsets: src(1), sizes: src(2) })
            }
            Op::Flip(ops::Flip { axes, .. }) => {
                assert_eq!(new_srcs.len(), 1);
                Op::Flip(ops::Flip { src: src(0), axes: axes.clone() })
            }
            Op::Multi(ops::Multi { axis, .. }) => {
                assert_eq!(new_srcs.len(), 1);
                Op::Multi(ops::Multi { src: src(0), axis: *axis })
            }

            // Reduction operations
            Op::ReduceAxis(ops::ReduceAxis { reduce_op, axes, .. }) => {
                assert_eq!(new_srcs.len(), 1);
                Op::ReduceAxis(ops::ReduceAxis { src: src(0), reduce_op: *reduce_op, axes: axes.clone() })
            }
            Op::Reduce(ops::Reduce { reduce_op, num_axes, .. }) => {
                assert!(!new_srcs.is_empty());
                Op::Reduce(ops::Reduce {
                    src: src(0),
                    ranges: new_srcs[1..].iter().cloned().collect(),
                    reduce_op: *reduce_op,
                    num_axes: *num_axes,
                })
            }
            Op::AllReduce(ops::AllReduce { device, reduce_op, .. }) => {
                assert_eq!(new_srcs.len(), 1);
                Op::AllReduce(ops::AllReduce { src: src(0), device: device.clone(), reduce_op: *reduce_op })
            }

            // Control flow operations
            Op::If(..) => {
                assert!(!new_srcs.is_empty());
                Op::If(ops::If { condition: src(0), body: new_srcs[1..].iter().cloned().collect() })
            }
            Op::EndIf(..) => {
                assert_eq!(new_srcs.len(), 1);
                Op::EndIf(ops::EndIf { if_op: src(0) })
            }
            Op::Range(ops::Range { axis_id, axis_type, .. }) => {
                assert!(!new_srcs.is_empty());
                Op::Range(ops::Range {
                    end: src(0),
                    axis_id: axis_id.clone(),
                    axis_type: *axis_type,
                    deps: new_srcs[1..].iter().cloned().collect(),
                })
            }
            Op::End(..) => {
                assert!(!new_srcs.is_empty());
                Op::End(ops::End { computation: src(0), ranges: new_srcs[1..].iter().cloned().collect() })
            }
            Op::Barrier(ops::Barrier { fence, .. }) => {
                assert!(!new_srcs.is_empty());
                Op::Barrier(ops::Barrier { src: src(0), deps: new_srcs[1..].iter().cloned().collect(), fence: *fence })
            }

            Op::Stack(..) => {
                return Self::stack(new_srcs.iter().cloned().collect());
            }
            Op::Shuffle(ops::Shuffle { mask, .. }) => {
                assert_eq!(new_srcs.len(), 1);
                Op::Shuffle(ops::Shuffle { src: src(0), mask: mask.clone() })
            }

            // Symbolic/Define operations
            Op::Bind(..) => {
                assert_eq!(new_srcs.len(), 2);
                Op::Bind(ops::Bind { var: src(0), value: src(1) })
            }

            // Advanced operations
            Op::Wmma(ops::Wmma { metadata, .. }) => {
                assert_eq!(new_srcs.len(), 3);
                Op::Wmma(ops::Wmma { a: src(0), b: src(1), c: src(2), metadata: metadata.clone() })
            }
            Op::Call(ops::Call { info, .. }) => {
                assert!(!new_srcs.is_empty(), "Call requires at least body source");
                Op::Call(ops::Call { body: src(0), args: new_srcs[1..].iter().cloned().collect(), info: info.clone() })
            }
            Op::Function(ops::Function { info, .. }) => {
                assert!(!new_srcs.is_empty(), "Function requires at least body source");
                Op::Function(ops::Function {
                    body: src(0),
                    args: new_srcs[1..].iter().cloned().collect(),
                    info: info.clone(),
                })
            }
            Op::Program(ops::Program { info, linear, source, binary, .. }) => {
                assert!(!new_srcs.is_empty(), "Program requires a sink source");
                let mut idx = 0usize;
                let sink = src(idx);
                idx += 1;

                let linear_new = if linear.is_some() {
                    let value = src(idx);
                    idx += 1;
                    Some(value)
                } else {
                    None
                };
                let source_new = if source.is_some() {
                    let value = src(idx);
                    idx += 1;
                    Some(value)
                } else {
                    None
                };
                let binary_new = if binary.is_some() {
                    let value = src(idx);
                    idx += 1;
                    Some(value)
                } else {
                    None
                };

                assert_eq!(idx, new_srcs.len(), "Program source count mismatch");
                Op::Program(ops::Program {
                    sink,
                    info: info.clone(),
                    linear: linear_new,
                    source: source_new,
                    binary: binary_new,
                })
            }
            Op::Linear(..) => Op::Linear(ops::Linear { ops: new_srcs.iter().cloned().collect() }),
            Op::Ins(ops::Ins { arg, .. }) => {
                Op::Ins(ops::Ins { sources: new_srcs.iter().cloned().collect(), arg: arg.clone() })
            }
            Op::Tuple(..) => Op::Tuple(ops::Tuple { src: new_srcs.iter().cloned().collect() }),
            Op::GetTuple(ops::GetTuple { index, .. }) => {
                assert_eq!(new_srcs.len(), 1);
                Op::GetTuple(ops::GetTuple { src: src(0), index: *index })
            }
            Op::Detach(..) => {
                assert_eq!(new_srcs.len(), 1);
                Op::Detach(ops::Detach { src: src(0) })
            }
            Op::Contiguous(ops::Contiguous { opts, .. }) => {
                assert_eq!(new_srcs.len(), 1);
                Op::Contiguous(ops::Contiguous { src: src(0), opts: opts.clone() })
            }
            Op::ContiguousBackward(..) => {
                assert_eq!(new_srcs.len(), 1);
                Op::ContiguousBackward(ops::ContiguousBackward { src: src(0) })
            }
            Op::After(..) => {
                assert!(!new_srcs.is_empty());
                let passthrough = src(0);
                // AFTER passthrough must not be control flow.
                debug_assert!(
                    !matches!(passthrough.op(), Op::Range(..) | Op::End(..)),
                    "reconstruct_sources: AFTER passthrough is {:?} (id={}), expected non-control-flow",
                    passthrough.op(),
                    passthrough.id
                );
                Op::After(ops::After { passthrough, deps: new_srcs[1..].iter().cloned().collect() })
            }
            Op::Precast(..) => {
                assert_eq!(new_srcs.len(), 1);
                Op::Precast(ops::Precast { src: src(0) })
            }
            Op::Custom(ops::Custom { code, .. }) => {
                Op::Custom(ops::Custom { deps: new_srcs.iter().cloned().collect(), code: code.clone() })
            }
            Op::CustomFunction(ops::CustomFunction { kind, .. }) => Op::CustomFunction(ops::CustomFunction {
                kind: kind.clone(),
                attrs: new_srcs.iter().cloned().collect(),
            }),
            Op::CustomI(ops::CustomI { code, .. }) => {
                Op::CustomI(ops::CustomI { deps: new_srcs.iter().cloned().collect(), code: code.clone() })
            }

            // Memory operations
            Op::Load(ops::Load { alt, gate, .. }) => {
                assert_eq!(alt.is_some(), gate.is_some(), "LOAD requires either index only or index, alt, and gate");
                let expected = 1 + usize::from(alt.is_some()) + usize::from(gate.is_some());
                assert_eq!(new_srcs.len(), expected);
                let mut next = 1;
                let new_alt = alt.as_ref().map(|_| {
                    let value = src(next);
                    next += 1;
                    value
                });
                let new_gate = gate.as_ref().map(|_| src(next));
                Op::Load(ops::Load { index: src(0), alt: new_alt, gate: new_gate })
            }
            Op::Store(ops::Store { gate, .. }) => {
                assert_eq!(new_srcs.len(), 2 + usize::from(gate.is_some()));
                Op::Store(ops::Store { index: src(0), value: src(1), gate: gate.as_ref().map(|_| src(2)) })
            }

            // Graph organization
            Op::Sink(ops::Sink { info, .. }) => {
                Op::Sink(ops::Sink { sources: new_srcs.iter().cloned().collect(), info: info.clone() })
            }
            Op::Group(..) => Op::Group(ops::Group { sources: new_srcs.iter().cloned().collect() }),
        };

        // ALU dtype and shape are independent: shaped STACK sources still have
        // scalar element dtypes. Always derive rebuilt ALU dtype so a legacy
        // vector result cannot survive after its lanes move into source shape.
        let source_dtypes_unchanged =
            self.op.children().iter().zip(new_op.children()).all(|(old, new)| old.dtype() == new.dtype());
        let dtype = if matches!(new_op, Op::Unary(..) | Op::Binary(..) | Op::Ternary(..)) {
            crate::dtype_from_op(&new_op).unwrap_or_else(|| self.dtype.clone())
        } else if source_dtypes_unchanged {
            self.dtype.clone()
        } else {
            crate::dtype_from_op(&new_op).unwrap_or_else(|| self.dtype.clone())
        };
        Self::new_with_origin(new_op, dtype, self.tag.clone(), self.origin)
    }
}

#[bon]
impl UOp {
    /// Create a modified copy with optional field overrides.
    ///
    /// Enables concise pattern implementations by allowing selective field modification.
    /// Returns `self.clone()` if nothing changed (optimization for hash consing).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let new_load = load.replace().dtype(new_dtype).src(new_sources).call();
    /// let dtype_only = load.replace().dtype(new_dtype).call();
    /// ```
    #[builder]
    pub fn replace(self: &Arc<Self>, dtype: Option<DType>, src: Option<Vec<Arc<Self>>>) -> Arc<Self> {
        let new_dtype = dtype.unwrap_or_else(|| self.dtype());
        let new_sources = src.unwrap_or_else(|| self.op().sources().to_vec());

        // Short-circuit if nothing changed
        let old_sources = self.op().sources();
        let sources_unchanged = new_sources.len() == old_sources.len()
            && new_sources.iter().zip(old_sources.iter()).all(|(a, b)| Arc::ptr_eq(a, b));

        if new_dtype == self.dtype() && sources_unchanged {
            return self.clone();
        }

        self.with_sources(new_sources).with_dtype(new_dtype)
    }
}

/// Pre-sizing hint for traversals that walk a whole kernel graph.
const FULL_GRAPH_HINT: usize = 256;

/// Pre-sizing hint for traversals that typically stop after a handful of nodes
/// (cached-property cold paths walk ~2 nodes on average).
const LOCAL_HINT: usize = 8;

/// Visited set for the pointer-identity graph traversals below.
///
/// `Arc::as_ptr` values are already well distributed, so FxHash is both faster
/// and sufficient; `capacity` is the caller's expected node count, which avoids
/// both rehashing on full-graph walks and 2 KiB of dead table on local ones.
fn visited_set(capacity: usize) -> rustc_hash::FxHashSet<*const UOp> {
    rustc_hash::FxHashSet::with_capacity_and_hasher(capacity, Default::default())
}

impl UOp {
    /// Allocate a node with a fresh id; interning is the caller's job.
    pub(crate) fn fresh(
        op: Op,
        dtype: DType,
        tag: Option<SmallVec<[usize; 2]>>,
        origin: Option<OriginId>,
        content_hash: u64,
        src_ops: crate::op::OpMask,
        metadata: Option<Arc<dyn std::any::Any + Send + Sync>>,
    ) -> Self {
        Self {
            id: crate::uop::hash_consing::next_uop_id(),
            op,
            dtype,
            content_hash,
            src_ops,
            tag,
            origin,
            metadata,
            shape_cache: std::sync::OnceLock::new(),
            ranges_cache: std::sync::OnceLock::new(),
            in_scope_ranges_cache: std::sync::OnceLock::new(),
            vmin_vmax_cache: std::sync::OnceLock::new(),
            sound_vmin_vmax_cache: std::sync::OnceLock::new(),
            has_index_in_sources_cache: std::sync::OnceLock::new(),
            has_weak_float_cache: std::sync::OnceLock::new(),
            device_spec_cache: std::sync::OnceLock::new(),
            addrspace_cache: std::sync::OnceLock::new(),
        }
    }
}

/// Trait for converting scalar values into UOps.
///
/// This allows operator overloading to work with mixed scalar/UOp operands.
/// For example: `uop + 5.0` or `5.0 + uop`.
pub trait IntoUOp {
    fn into_uop(self, dtype: DType) -> Arc<UOp>;
}

impl IntoUOp for ConstValue {
    fn into_uop(self, dtype: DType) -> Arc<UOp> {
        UOp::const_(dtype, self)
    }
}

impl IntoUOp for f32 {
    fn into_uop(self, dtype: DType) -> Arc<UOp> {
        UOp::const_(dtype, ConstValue::Float(self as f64))
    }
}

impl IntoUOp for f64 {
    fn into_uop(self, dtype: DType) -> Arc<UOp> {
        UOp::const_(dtype, ConstValue::Float(self))
    }
}

impl IntoUOp for i32 {
    fn into_uop(self, dtype: DType) -> Arc<UOp> {
        UOp::const_(dtype, ConstValue::Int(self as i64))
    }
}

impl IntoUOp for i64 {
    fn into_uop(self, dtype: DType) -> Arc<UOp> {
        UOp::const_(dtype, ConstValue::Int(self))
    }
}

impl IntoUOp for u32 {
    fn into_uop(self, dtype: DType) -> Arc<UOp> {
        UOp::const_(dtype, ConstValue::UInt(self as u64))
    }
}

impl IntoUOp for u64 {
    fn into_uop(self, dtype: DType) -> Arc<UOp> {
        UOp::const_(dtype, ConstValue::UInt(self))
    }
}

impl IntoUOp for bool {
    fn into_uop(self, dtype: DType) -> Arc<UOp> {
        UOp::const_(dtype, ConstValue::Bool(self))
    }
}
