//! Process-global cache of immutable weight storages, keyed by checkpoint
//! provenance. Loading the same file into N model instances uploads each
//! weight ONCE and shares the device storage; the storage is sealed
//! immutable so no handle can corrupt a peer model. `Weak` values: the last
//! model's teardown frees the VRAM, and dead entries are pruned on the next
//! lookup. Concurrent loads of one file dedup through singleflight — one
//! thread allocates and uploads, the rest reuse.
//!
//! Sharing is gated strictly on PROVENANCE + immutability, never on
//! structural graph equality — value-identical mutable tensors (JIT inputs)
//! must never share storage.

use std::sync::{Arc, OnceLock, Weak};

use papaya::HashMap;
use svod_device::Buffer;
use svod_dtype::{DType, DeviceSpec};

/// Provenance identity of one weight tensor in a checkpoint file.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct WeightKey {
    /// Canonicalized checkpoint file path.
    pub path: std::path::PathBuf,
    /// Tensor name within the file.
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub device: DeviceSpec,
}

static WEIGHTS: OnceLock<HashMap<WeightKey, Weak<Buffer>>> = OnceLock::new();

fn weights() -> &'static HashMap<WeightKey, Weak<Buffer>> {
    WEIGHTS.get_or_init(HashMap::new)
}

fn weight_flight() -> &'static crate::singleflight::Singleflight<WeightKey> {
    static FLIGHT: OnceLock<crate::singleflight::Singleflight<WeightKey>> = OnceLock::new();
    FLIGHT.get_or_init(crate::singleflight::Singleflight::new)
}

/// Get-or-create the shared immutable storage for `key`, uploading `bytes`
/// on first use. The returned `Arc` is what keeps the storage alive — the
/// cache itself holds only a `Weak`.
pub(crate) fn shared_weight_buffer(key: WeightKey, bytes: &[u8]) -> Arc<Buffer> {
    let map = weights();
    let lookup = || {
        let guard = map.guard();
        match map.get(&key, &guard) {
            Some(weak) => match weak.upgrade() {
                Some(buffer) => Some(buffer),
                None => {
                    // Last owner died; prune so the map stays bounded by the
                    // number of LIVE weights. Conditional remove: between our
                    // dead read and this removal, a concurrent winner may have
                    // inserted a FRESH entry under the same key — a blind
                    // `remove(&key)` would delete it and silently defeat
                    // sharing for the next loader.
                    use papaya::{Compute, Operation};
                    let _: Compute<_, _, ()> = map.compute(
                        key.clone(),
                        |entry| match entry {
                            Some((_, weak)) if weak.upgrade().is_none() => Operation::Remove,
                            _ => Operation::Abort(()),
                        },
                        &guard,
                    );
                    None
                }
            },
            None => None,
        }
    };
    weight_flight()
        .run::<_, std::convert::Infallible>(key.clone(), lookup, || {
            let allocator = svod_device::registry::registry().get(&key.device).unwrap_or_else(|e| {
                panic!("Failed to get allocator for {:?}: {e}", key.device);
            });
            // `nolru`: a checkpoint weight lives as long as its model and its
            // size rarely matches another allocation, so pooling it on free
            // would only hold device memory (all of a model's f32 weights once
            // they are cast to a narrower dtype) until an allocation fails.
            let spec = svod_device::BufferSpec { nolru: true, ..Default::default() };
            let mut buffer = Buffer::new(allocator, key.dtype.clone(), key.shape.clone(), spec);
            buffer.copyin(bytes).expect("weight upload");
            buffer.mark_immutable();
            let arc = Arc::new(buffer);
            let guard = map.guard();
            map.insert(key.clone(), Arc::downgrade(&arc), &guard);
            Ok(arc)
        })
        .unwrap_or_else(|infallible| match infallible {})
}

#[cfg(test)]
#[path = "test/unit/weight_cache.rs"]
mod tests;
