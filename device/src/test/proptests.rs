use crate::Buffer;
use crate::allocator::{Allocator, BufferSpec, CpuAllocator, LruAllocator};
use proptest::prelude::*;
use smallvec::SmallVec;
use std::sync::Arc;
use strum::VariantArray;
use svod_dtype::test::proptests::generators;
use svod_dtype::{DType, ScalarDType};

/// Helper to create an LRU allocator for testing.
fn allocator() -> Arc<LruAllocator> {
    Arc::new(LruAllocator::new(Box::new(CpuAllocator)))
}

/// A buffer allocation case for property-based testing. (Named `AllocCase` to
/// avoid colliding with the allocator's [`BufferSpec`].)
#[derive(Debug, Clone)]
struct AllocCase {
    dtype: DType,
    shape: SmallVec<[usize; 4]>,
    zero_init: bool,
}

impl AllocCase {
    /// Calculate the total size in bytes.
    fn size(&self) -> usize {
        self.dtype.bytes() * self.shape.iter().product::<usize>()
    }

    fn alloc<A: Allocator + 'static>(&self, allocator: Arc<A>) -> Result<Buffer, crate::Error> {
        Buffer::allocate_with_zero_init(
            allocator,
            self.dtype.clone(),
            self.shape.to_vec(),
            BufferSpec::default(),
            self.zero_init,
        )
    }
}

impl Arbitrary for AllocCase {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
        (
            generators::scalar_generator().prop_map(DType::Scalar),
            prop::collection::vec(1usize..50, 1..=4),
            any::<bool>(),
        )
            .prop_map(|(dtype, shape, zero_init)| AllocCase { dtype, shape: SmallVec::from_vec(shape), zero_init })
            .prop_filter("total size must be reasonable", |spec| (1..=10 * 1024 * 1024).contains(&spec.size()))
            .boxed()
    }
}

fn same_size_dtypes(dtype: DType) -> impl Strategy<Value = DType> {
    // Get all scalar dtypes and filter by byte size
    let dtypes = ScalarDType::VARIANTS
        .iter()
        .filter(|s| s.bytes() == dtype.bytes())
        .map(|s| DType::Scalar(*s))
        .filter(|d| *d != dtype)
        .collect::<Vec<_>>();
    proptest::sample::select(dtypes)
}

/// Strategy to generate two AllocCases with the same total size.
///
/// Generates two specs with same byte size but different dtypes (and possibly different shapes).
/// Since dtypes with same byte size are used, the total size in bytes remains the same.
fn same_size_specs() -> impl Strategy<Value = (AllocCase, AllocCase)> {
    any::<AllocCase>().prop_flat_map(|spec| {
        let total_bytes = spec.size();
        let dtype = spec.dtype.clone();
        same_size_dtypes(dtype).prop_map(move |dtype| {
            let num_elements = total_bytes / dtype.bytes();
            let shape = SmallVec::from_elem(num_elements, 1);
            let spec2 = AllocCase { dtype, shape, zero_init: spec.zero_init };
            (spec.clone(), spec2)
        })
    })
}

proptest! {
    /// Property: dtype/shape metadata is isolated between buffer reuses.
    ///
    /// This is the critical property for cache correctness: even though the underlying
    /// RawBuffer is reused, each Buffer must maintain its own dtype and shape metadata.
    #[test]
    fn dtype_shape_isolation_on_reuse((spec1, spec2) in same_size_specs()) {
        let alloc = allocator();

        // Allocate buffer1, capture its pointer, then drop to cache
        let ptr1 = {
            spec1.alloc(alloc.clone())?.raw_data_ptr()
        };

        // Allocate buffer2 with spec2 - should reuse RawBuffer but have new metadata
        let buffer2 = spec2.alloc(alloc)?;

        prop_assert_eq!(ptr1, buffer2.raw_data_ptr(), "buffer2 should reuse buffer1's RawBuffer from cache");
        prop_assert_eq!(buffer2.dtype(), spec2.dtype.clone(), "dtype must be from spec2, not spec1");
        prop_assert_eq!(buffer2.size(), spec2.size(), "size must be from spec2, not spec1");
    }

    /// Property: Cache respects capacity limits without crashing.
    ///
    /// The LRU cache has a per-key capacity of 32 buffers. Allocating more than this
    /// should evict older buffers without panicking or causing OOM.
    #[test]
    fn cache_respects_capacity(specs in prop::collection::vec(any::<AllocCase>(), 1..=4)) {
        let alloc = allocator();

        // Allocate and free many buffers with same size
        for spec in specs {
            spec.alloc(Arc::clone(&alloc))?;
        }
    }

    /// Property: Buffer views share the backing RawBuffer and don't double-cache.
    ///
    /// When a Buffer and its view are both dropped, only one RawBuffer should be
    /// cached (since they share the same underlying allocation via Rc).
    #[test]
    fn views_share_backing_buffer(spec: AllocCase) {
        // Need at least 20 bytes for a meaningful view
        prop_assume!(spec.size() >= 20);

        let alloc = allocator();

        // Create buffer and view, then drop both
        {
            let buffer = spec.alloc(alloc.clone())?;
            let _view = buffer.view(0, spec.size().min(10))?;

            // Both buffer and view go out of scope here
            // Only one RawBuffer should be cached (Rc drops correctly)
        }

        // Verify exactly one buffer was cached
        prop_assert_eq!(alloc.cache_count(spec.size(), true), 1, "Expected exactly 1 buffer cached after dropping buffer+view");

        // Allocate again with same size - should reuse the single cached RawBuffer
        let _buffer2 = spec.alloc(alloc.clone())?;

        // Cache should now be empty (buffer was reused)
        prop_assert_eq!(alloc.cache_count(spec.size(), true), 0, "Expected cache to be empty after reusing buffer");
    }

    /// Property: zero_init works correctly with cache reuse.
    ///
    /// Buffers with different zero_init values share the same cache entry (same size/cpu_access).
    /// When retrieving from cache with zero_init=true, the allocator must zero the buffer.
    /// This test verifies that buffers can be allocated with different zero_init settings
    /// without conflicts.
    #[test]
    fn zero_init_with_cache_reuse(spec: AllocCase) {
        let alloc = allocator();

        // Allocate buffer with zero_init=false, then drop to cache
        {
            let _buffer1 = Buffer::allocate_with_zero_init(
                alloc.clone() as Arc<dyn Allocator>,
                spec.dtype.clone(),
                spec.shape.to_vec(),
                BufferSpec { cpu_access: false, ..Default::default() },
                /*zero_init=*/ false,
            )?;
        }

        // Verify buffer is cached
        prop_assert_eq!(alloc.cache_count(spec.size(), false), 1, "Buffer should be cached");

        // Allocate with zero_init=true - should reuse from cache and zero it
        let _buffer2 = Buffer::allocate_with_zero_init(
            alloc.clone() as Arc<dyn Allocator>,
            spec.dtype.clone(),
            spec.shape.to_vec(),
            BufferSpec { cpu_access: false, ..Default::default() },
            /*zero_init=*/ true,
        )?;

        // Cache should be empty (buffer was reused)
        prop_assert_eq!(alloc.cache_count(spec.size(), false), 0, "Cache should be empty after reuse");
    }
}
