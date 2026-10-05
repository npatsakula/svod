//! Global tensor registry for atomic graph substitution.
//!
//! This module implements Tinygrad's `all_tensors` pattern using papaya's lock-free HashMap.
//! When rangeify transforms a UOp (e.g., NEG → STAGE(NEG)), the `becomes_map` must be
//! applied to ALL tensors that reference it - not just the one being realized.
//!
//! Without this, diamond patterns (like argmin's NEG feeding both MAX and EQ) fail because
//! different consumers see different versions of the same producer.
//!
//! # Thread Safety
//!
//! All operations are lock-free and thread-safe. Uses papaya's epoch-based reclamation
//! for concurrent access and parking_lot::RwLock for interior UOp mutation.
//!
//! # Memory Management (Tinygrad-aligned)
//!
//! Tensors are stored as `Weak<TensorEntry>` in the registry. When all strong references
//! (held by `Tensor` structs) are dropped, the entry becomes eligible for cleanup.
//! Dead weak refs are cleaned lazily on access or via `gc_dead_refs()`.
//!
//! This matches Tinygrad's `weakref.WeakKeyDictionary` pattern - no manual cleanup required.
//!
//! # Buffer Storage
//!
//! Buffers are stored in a separate map (`BUFFERS`) indexed by UOp ID.
//! This is a lookup index for `collect_input_buffers()` during schedule creation.
//! - Key is UOp ID (unique per buffer via the `ParamArg.slot` unique counter)
//! - Value is `Arc<Buffer>` (strong ref — kept alive while the BUFFER UOp lives)
//! - Entries expire automatically when the BUFFER UOp is dropped, via
//!   `svod_ir::uop::set_uop_drop_hook` (installed on first map access)
//!
//! Unlike Tinygrad (which stores buffers inline in a `WeakKeyDictionary` keyed
//! on UOps), Svod uses a separate id-keyed index because UOps are immutable
//! and hash-consed; the drop hook gives the same lifetime semantics.
//!
//! TensorEntry also caches the buffer for direct access via tensor.buffer().

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, Weak};

use papaya::HashMap as PapayaMap;
use parking_lot::RwLock;
use svod_device::Buffer;
use svod_ir::ops;
use svod_ir::uop::SubtreeMemo;
use svod_ir::{Op, UOp, UOpKey};

/// Atomic counter for unique tensor IDs.
static TENSOR_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

fn next_tensor_id() -> u64 {
    TENSOR_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Entry in the global tensor registry.
///
/// Uses RwLock for interior mutability of the UOp during global substitution.
/// The RwLock allows concurrent reads (typical tensor operations) with exclusive
/// writes only during `apply_map_to_tensors`.
///
/// Buffer can be set at construction (input tensors) or later (realized tensors).
/// Uses OnceLock for thread-safe one-time initialization.
pub struct TensorEntry {
    /// Unique tensor ID (stable across UOp updates).
    pub id: u64,
    /// The computation graph (mutable for global substitution).
    pub uop: RwLock<Arc<UOp>>,
    /// The materialized buffer (can be set once via OnceLock).
    buffer: OnceLock<Arc<Buffer>>,
}

impl TensorEntry {
    /// Get the buffer if materialized.
    pub fn buffer(&self) -> Option<&Arc<Buffer>> {
        self.buffer.get()
    }

    /// Set the buffer (can only be called once, subsequent calls are no-ops).
    /// Returns true if buffer was set, false if already set.
    pub fn set_buffer(&self, buffer: Arc<Buffer>) -> bool {
        self.buffer.set(buffer).is_ok()
    }
}

// Global tensor registry using lock-free concurrent HashMap.
//
// Design: Stores Weak<TensorEntry> for automatic memory management (Tinygrad-aligned).
// - Tensor structs hold Arc<TensorEntry> (strong refs)
// - Registry holds Weak<TensorEntry> (weak refs)
// - When Tensor is dropped, TensorEntry can be cleaned up
// - Dead weak refs cleaned lazily on access or via gc_dead_refs()
static TENSORS: OnceLock<PapayaMap<u64, Weak<TensorEntry>>> = OnceLock::new();

// Direct buffer storage: UOp ID → Arc<Buffer>.
//
// Lookup index for collect_input_buffers() during schedule creation.
// Buffer UOp IDs are unique (Op::Unique monotonic counter), so entries never
// collide across tests. Stale entries cleaned via gc_dead_refs().
static BUFFERS: OnceLock<PapayaMap<u64, Arc<Buffer>>> = OnceLock::new();

fn tensors() -> &'static PapayaMap<u64, Weak<TensorEntry>> {
    TENSORS.get_or_init(PapayaMap::new)
}

fn buffers() -> &'static PapayaMap<u64, Arc<Buffer>> {
    BUFFERS.get_or_init(|| {
        // Expire entries automatically when their BUFFER UOp dies: uop ids are
        // per-allocation and never reused, so once the node is dropped no
        // live graph can look the id up again — the entry is unreachable
        // garbage from that point. Lock-free removal; safe from Drop context.
        svod_ir::uop::set_uop_drop_hook(|uop_id| {
            if let Some(map) = BUFFERS.get() {
                let guard = map.guard();
                map.remove(&uop_id, &guard);
            }
        });
        PapayaMap::new()
    })
}

/// Register a new tensor without buffer (for lazy computation graphs).
///
/// Thread-safe: each call creates a unique tensor ID.
/// The registry stores a weak reference; the caller holds the strong reference.
///
/// # Arguments
///
/// * `uop` - The tensor's computation graph
///
/// # Returns
///
/// Arc to the registered TensorEntry (caller owns the strong reference)
pub fn register_tensor(uop: Arc<UOp>) -> Arc<TensorEntry> {
    let id = next_tensor_id();
    let entry = Arc::new(TensorEntry { id, uop: RwLock::new(uop), buffer: OnceLock::new() });

    // Store weak ref in registry - entry stays alive as long as caller holds Arc
    let guard = tensors().guard();
    tensors().insert(id, Arc::downgrade(&entry), &guard);

    entry
}

/// Register a new tensor with buffer (for input tensors and realized tensors).
///
/// Stores buffer in both:
/// 1. BUFFERS map (indexed by UOp ID) - for schedule buffer lookups
/// 2. TensorEntry.buffer - for direct tensor access
///
/// The registry stores a weak reference; the caller holds the strong reference.
///
/// # Arguments
///
/// * `uop` - The tensor's computation graph
/// * `buffer` - The materialized buffer
/// * `buffer_uop_id` - The UOp ID to index under (for lookups)
///
/// # Returns
///
/// Arc to the registered TensorEntry (caller owns the strong reference)
pub fn register_tensor_with_buffer(uop: Arc<UOp>, buffer: Arc<Buffer>, buffer_uop_id: u64) -> Arc<TensorEntry> {
    let id = next_tensor_id();
    let entry = Arc::new(TensorEntry { id, uop: RwLock::new(uop), buffer: OnceLock::from(buffer.clone()) });

    // Store weak ref in tensor registry
    let guard = tensors().guard();
    tensors().insert(id, Arc::downgrade(&entry), &guard);

    // Store buffer indexed by UOp ID (for collect_input_buffers lookups)
    let buf_guard = buffers().guard();
    buffers().insert(buffer_uop_id, buffer, &buf_guard);

    entry
}

/// Get buffer by UOp ID (cloned).
///
/// Direct lookup from BUFFERS map.
/// Used by collect_input_buffers() during schedule creation.
pub fn get_buffer(uop_id: u64) -> Option<Buffer> {
    let guard = buffers().guard();
    buffers().get(&uop_id, &guard).map(|arc_buf| (**arc_buf).clone())
}

/// Get buffer by UOp ID as Arc (shared reference, no clone).
///
/// Used by ensure_buffer() to attach a buffer without cloning.
pub fn get_buffer_arc(uop_id: u64) -> Option<Arc<Buffer>> {
    let guard = buffers().guard();
    buffers().get(&uop_id, &guard).cloned()
}

/// Remove buffer entry from the BUFFERS map.
///
/// Called during cleanup to eagerly remove stale entries.
pub fn remove_buffer(uop_id: u64) {
    let buf_guard = buffers().guard();
    buffers().remove(&uop_id, &buf_guard);
}

/// Get count of buffers in the registry (for testing/diagnostics).
pub fn buffer_count() -> usize {
    buffers().len()
}

/// Register a buffer for an existing tensor.
///
/// Used by realize() to associate output buffers with tensors for schedule lookups.
/// Stores buffer in both BUFFERS map and TensorEntry.
///
/// # Arguments
///
/// * `uop_id` - The UOp ID to index under (for lookups)
/// * `tensor_id` - The tensor ID that owns this buffer
/// * `buffer` - The materialized buffer
pub fn register_buffer(uop_id: u64, tensor_id: u64, buffer: Arc<Buffer>) {
    // Store buffer indexed by UOp ID (for collect_input_buffers lookups)
    let buf_guard = buffers().guard();
    buffers().insert(uop_id, buffer.clone(), &buf_guard);

    // Also set buffer on the TensorEntry for direct tensor access. A lost
    // `set_buffer` for the SAME buffer is a benign re-registration; a lost set
    // for a DIFFERENT buffer means two realizes disagreed about this tensor's
    // storage — an identity bug upstream that must never pass silently.
    if let Some(entry) = get_tensor(tensor_id)
        && !entry.set_buffer(buffer.clone())
        && let Some(existing) = entry.buffer()
        && existing.id() != buffer.id()
    {
        tracing::error!(
            tensor_id,
            existing = existing.id().0,
            incoming = buffer.id().0,
            "register_buffer: tensor already holds a different buffer — lost set indicates an identity bug"
        );
    }
}

/// Register a buffer by UOp ID only (no TensorEntry association).
///
/// Used for pending assign side-realization where the buffer belongs
/// to the computation graph, not a specific tensor.
pub fn register_buffer_by_uop_id(uop_id: u64, buffer: Arc<Buffer>) {
    let guard = buffers().guard();
    buffers().insert(uop_id, buffer, &guard);
}

/// Get a tensor entry by ID.
///
/// Thread-safe read operation. Returns None if tensor was dropped.
pub fn get_tensor(id: u64) -> Option<Arc<TensorEntry>> {
    let guard = tensors().guard();
    tensors().get(&id, &guard)?.upgrade()
}

/// Remove dead weak references from the tensor registry.
///
/// Tensors: removes entries whose `Weak<TensorEntry>` can no longer be
/// upgraded. Buffer entries need no sweep — they expire automatically via the
/// UOp drop hook installed in `buffers`.
pub fn gc_dead_refs() {
    let map = tensors();
    let guard = map.guard();
    let to_remove: Vec<u64> = map.iter(&guard).filter(|(_, weak)| weak.upgrade().is_none()).map(|(k, _)| *k).collect();
    for id in to_remove {
        map.remove(&id, &guard);
    }
}

/// Legacy alias for gc_dead_refs (for compatibility).
///
/// With weak references, tensors are automatically cleaned up when no longer
/// referenced. This function now just cleans up dead weak refs in the registry.
#[deprecated(note = "Tensor registry now uses weak refs - cleanup is automatic. Use gc_dead_refs() to clean registry.")]
pub fn gc_unused_tensors() {
    gc_dead_refs();
}

/// Apply a transformation map to ALL live tensors globally.
///
/// This is Svod's equivalent of Tinygrad's `_apply_map_to_tensors`.
/// When rangeify creates a becomes_map (old UOp → new UOp), this function
/// ensures ALL tensors see the same transformed versions.
///
/// # Arguments
///
/// * `becomes_map` - Mapping from original UOps to their transformed versions
///
/// # Thread Safety
///
/// This function acquires write locks on affected tensors during the update phase.
/// Other tensors can still be read/written concurrently.
pub fn apply_map_to_tensors(becomes_map: &HashMap<UOpKey, Arc<UOp>>) {
    apply_map_to_tensors_inner(becomes_map, false, None);
}

/// [`apply_map_to_tensors`] for the realize-final `{old → realized BUFFER}`
/// broadcast. Device-scoped: the rewrite pulls a concrete device into the
/// receiver's graph, so it only folds tensors already anchored to the SAME
/// device. Device-less (pure) receivers keep their graphs and recompute on
/// whatever device their own realize resolves — value identity must never
/// move a tensor onto another device (an `amd` test variant realizing a
/// constant must not turn the concurrent `clang` variant's plan into an AMD
/// plan).
pub fn apply_map_to_tensors_realized(becomes_map: &HashMap<UOpKey, Arc<UOp>>) {
    let device = becomes_map.values().find_map(|new| new.device_spec());
    apply_map_to_tensors_inner(becomes_map, false, device);
}

/// Walk variant: replacements are NOT re-traversed.
///
/// Use when a replacement may contain the original key, such as the
/// view-assign case `Buffer → After(Buffer, [Store(...)])`.
pub fn apply_map_to_tensors_walk(becomes_map: &HashMap<UOpKey, Arc<UOp>>) {
    apply_map_to_tensors_inner(becomes_map, true, None);
}

fn apply_map_to_tensors_inner(
    becomes_map: &HashMap<UOpKey, Arc<UOp>>,
    walk: bool,
    same_device: Option<svod_dtype::DeviceSpec>,
) {
    if becomes_map.is_empty() {
        return;
    }

    let map = tensors();
    let guard = map.guard();

    // Phase 1: Find affected tensors (read-only scan, skip dead weak refs).
    //
    // One memo shared across every live tensor: tensor graphs overlap heavily,
    // so a per-tensor walk would re-answer "does this subtree touch a replaced
    // node" for the same shared nodes on every realize (the dominant
    // multi-model prepare cost).
    let replaced: HashSet<u64> = becomes_map.keys().map(|key| key.0.id).collect();
    let mut touches_replaced = SubtreeMemo::new(|node: &Arc<UOp>| replaced.contains(&node.id));
    let affected: Vec<Arc<TensorEntry>> = map
        .iter(&guard)
        .filter_map(|(_, weak)| {
            let entry = weak.upgrade()?; // Skip dead entries
            let is_affected = {
                let uop = entry.uop.read();
                // Device scope (realize-final broadcasts): only fold tensors
                // anchored to the realized buffer's device.
                if let Some(device) = same_device.as_ref()
                    && uop.device_spec().as_ref() != Some(device)
                {
                    return None;
                }
                touches_replaced.contains(&uop)
            }; // uop lock dropped here
            if is_affected { Some(entry) } else { None }
        })
        .collect();

    if affected.is_empty() {
        return;
    }

    // Phase 2: Create SINK of affected tensor UOps
    let sources: Vec<Arc<UOp>> = affected.iter().map(|e| e.uop.read().clone()).collect();
    let sink = UOp::sink(sources.clone());

    // Phase 3: Atomic substitution across all affected UOps
    let new_sink = if walk { sink.substitute_walk(becomes_map) } else { sink.substitute(becomes_map) };

    // Phase 4: Update each tensor's UOp (acquires write locks). An entry may
    // have been concurrently finalized (`set_uop` from another realize)
    // between the Phase-2 snapshot and here; a blind store would lose that
    // update — the historical cross-plan input-aliasing bug. Under the write
    // lock the entry cannot move, so on a detected change re-apply the
    // substitution to the CURRENT value instead of storing the stale batch
    // result. Re-applying is idempotent: realize maps are `old → replacement`
    // where the replacement no longer contains the old key, so an
    // already-rewritten value comes back unchanged and is left alone.
    if let Op::Sink(ops::Sink { sources: new_sources, .. }) = new_sink.op() {
        for (entry, (old, new)) in affected.iter().zip(sources.iter().zip(new_sources.iter())) {
            if Arc::ptr_eq(old, new) {
                continue;
            }
            let mut slot = entry.uop.write();
            if Arc::ptr_eq(&slot, old) {
                *slot = new.clone();
                continue;
            }
            tracing::debug!(
                tensor_id = entry.id,
                "apply_map: entry changed concurrently; re-substituting its current value"
            );
            let current = slot.clone();
            let updated = if walk { current.substitute_walk(becomes_map) } else { current.substitute(becomes_map) };
            if !Arc::ptr_eq(&updated, &current) {
                *slot = updated;
            }
        }
    }
}

#[cfg(test)]
#[path = "test/unit/tensor_registry.rs"]
mod tests;
