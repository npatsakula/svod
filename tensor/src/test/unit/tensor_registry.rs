use super::*;
use svod_dtype::DType;
use svod_ir::ConstValue;

#[test]
fn test_register_and_get() {
    crate::test::helpers::test_setup();

    let uop = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let entry = register_tensor(uop.clone());

    let retrieved = get_tensor(entry.id).expect("Should find tensor");
    assert_eq!(retrieved.id, entry.id);
    assert!(Arc::ptr_eq(&*retrieved.uop.read(), &uop));
}

#[test]
fn test_apply_map_updates_tensors() {
    crate::test::helpers::test_setup();

    // Create two tensors sharing a common UOp
    let shared = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let t1_uop = shared.neg();
    let t2_uop = shared.neg(); // Same as t1_uop due to hash consing

    let t1 = register_tensor(t1_uop.clone());
    let t2 = register_tensor(t2_uop.clone());

    // Create a replacement for the shared const
    let replacement = UOp::const_(DType::Float32, ConstValue::Float(2.0));

    let mut becomes_map = HashMap::new();
    becomes_map.insert(UOpKey(shared.clone()), replacement.clone());

    // Apply the map
    apply_map_to_tensors(&becomes_map);

    // Both tensors should now reference the replacement
    let t1_new = t1.uop.read();
    let t2_new = t2.uop.read();

    // The root NEG should now have the replacement as its source
    assert!(!Arc::ptr_eq(&*t1_new, &t1_uop), "t1 should be updated");
    assert!(!Arc::ptr_eq(&*t2_new, &t2_uop), "t2 should be updated");
}

/// Regression: two threads realizing hash-cons-identical `zeros` graphs must
/// end up with COHERENT identities — the buffer a tensor's `buffer()` returns
/// must be the buffer its `uop()` graph resolves to through the registry.
/// Before the Phase-4 CAS in `apply_map_to_tensors_inner`, one realize's
/// broadcast could clobber the other tensor's just-finalized entry, leaving
/// `buffer()` and `uop()` pointing at DIFFERENT buffers (the model-JIT
/// parallel-test flake). Convergence of both tensors onto ONE shared buffer
/// is legitimate (value-identical pure graphs dedup); incoherence is not.
#[test]
fn concurrent_zeros_realizes_keep_coherent_identities() {
    crate::test::helpers::test_setup();
    let cfg = crate::PrepareConfig::default();
    // Warm the schedule cache so both threads race through apply_map together.
    {
        let warm = crate::Tensor::zeros(&[3], DType::Float32);
        warm.realize_with(&cfg).unwrap();
    }
    for _ in 0..100 {
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let handles = [(); 2].map(|()| {
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                let cfg = crate::PrepareConfig::default();
                let t = crate::Tensor::zeros(&[3], DType::Float32);
                barrier.wait();
                t.realize_with(&cfg).unwrap();
                let local = t.buffer().expect("realized tensor has a buffer").id();
                let via_uop = crate::tensor_registry::get_buffer(t.uop().base().id)
                    .expect("realized graph resolves through the registry")
                    .id();
                assert_eq!(local, via_uop, "buffer()/uop() identity channels must agree");
                let mut bytes = [1u8; 12];
                t.buffer().unwrap().copyout(&mut bytes).unwrap();
                assert_eq!(bytes, [0u8; 12], "zeros must read back as zeros");
            })
        });
        for handle in handles {
            handle.join().unwrap();
        }
    }
}

/// Buffer registry entries must expire automatically when their BUFFER UOp is
/// dropped (the `set_uop_drop_hook` path) — no manual release required.
#[test]
fn buffer_entry_expires_when_uop_drops() {
    crate::test::helpers::test_setup();
    let uop = UOp::new_buffer(svod_dtype::DeviceSpec::Cpu, 4, DType::Float32);
    let id = uop.id;
    let buffer = svod_device::Buffer::new(
        svod_device::registry::cpu().expect("cpu allocator"),
        DType::Float32,
        vec![4],
        Default::default(),
    );
    register_buffer_by_uop_id(id, Arc::new(buffer));
    assert!(get_buffer(id).is_some());

    drop(uop);
    assert!(get_buffer(id).is_none(), "entry must expire with its UOp");
}

/// Re-registering a live id (an in-place assign re-inserts its own storage)
/// must never hide the buffer from a concurrent lookup: only removal may
/// empty a slot that a reader can still hold.
#[test]
fn re_registering_a_live_id_never_hides_its_buffer() {
    let uop = UOp::new_buffer(svod_dtype::DeviceSpec::Cpu, 4, DType::Float32);
    let allocator = svod_device::registry::cpu().expect("cpu allocator");
    let buffer = Arc::new(svod_device::Buffer::new(allocator, DType::Float32, vec![4], Default::default()));
    register_buffer_by_uop_id(uop.id, buffer.clone());
    let done = std::sync::atomic::AtomicBool::new(false);
    std::thread::scope(|scope| {
        let readers: Vec<_> = (0..3)
            .map(|_| {
                scope.spawn(|| {
                    let mut misses = 0;
                    while !done.load(Ordering::Relaxed) {
                        misses += usize::from(get_buffer_arc(uop.id).is_none());
                    }
                    misses
                })
            })
            .collect();
        for _ in 0..100_000 {
            register_buffer_by_uop_id(uop.id, buffer.clone());
        }
        done.store(true, Ordering::Relaxed);
        let misses: usize = readers.into_iter().map(|reader| reader.join().unwrap()).sum();
        assert_eq!(misses, 0, "lookups of a live id found no buffer");
    });
}

/// The realize-final broadcast is device-scoped: a device-less (pure)
/// receiver sharing the realized graph must NOT be folded onto the realized
/// device buffer — an `amd` variant realizing a shared constant must never
/// turn a concurrent CPU variant's plan into an AMD plan. The pure receiver
/// keeps its graph and realizes independently.
#[test]
fn realized_broadcast_skips_deviceless_receivers() {
    crate::test::helpers::test_setup();
    let a = crate::Tensor::zeros(&[3], DType::Float32).contiguous();
    let b = crate::Tensor::zeros(&[3], DType::Float32).contiguous();
    let shared = b.uop();

    a.realize().unwrap();
    assert!(a.uop().device_spec().is_some(), "realized tensor is device-anchored");
    assert!(
        Arc::ptr_eq(&b.uop(), &shared),
        "pure receiver must keep its graph instead of adopting the realized device buffer"
    );

    // ...and still realizes correctly on its own.
    let b = b;
    b.realize().unwrap();
    let local = b.buffer().expect("own buffer").id();
    let via_uop = crate::tensor_registry::get_buffer(b.uop().base().id).expect("registry entry").id();
    assert_eq!(local, via_uop);
}
