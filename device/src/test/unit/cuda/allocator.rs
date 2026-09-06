use std::sync::Arc;

use svod_dtype::DType;
use test_case::test_case;

use super::cuda_alloc_or_skip;
use crate::Buffer;
use crate::allocator::{Allocator, BufferSpec, RawBuffer};
use crate::cuda::device::STAGING_BYTES;
use crate::cuda::{CudaAllocator, CudaMemory};

fn pattern(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i * 7 % 251) as u8).collect()
}

fn device_local() -> BufferSpec {
    BufferSpec { cpu_access: false, ..BufferSpec::default() }
}

#[test]
fn alloc_copyin_copyout_roundtrip_with_offsets() {
    let Some(alloc) = cuda_alloc_or_skip() else { return };
    let buffer = alloc._alloc(4096, &device_local(), false).expect("alloc");
    assert!(
        matches!(buffer, RawBuffer::Cuda { size: 4096, memory: CudaMemory::Device, host_ptr: None, .. }),
        "{buffer:?}"
    );
    assert!(!buffer.cpu_accessible());
    let data = pattern(4096);
    alloc._copyin(&buffer, 0, &data).expect("copyin");
    let mut back = vec![0u8; 4096];
    alloc._copyout(&mut back, &buffer, 0).expect("copyout");
    assert_eq!(back, data);
    let mut tail = vec![0u8; 16];
    alloc._copyout(&mut tail, &buffer, 4080).expect("copyout tail");
    assert_eq!(tail, &data[4080..]);
    alloc._copyin(&buffer, 100, &[0xAB; 8]).unwrap();
    alloc._copyout(&mut back, &buffer, 0).unwrap();
    assert_eq!(&back[100..108], &[0xAB; 8]);
    assert_eq!(&back[..100], &data[..100]);
    assert_eq!(&back[108..], &data[108..]);
    alloc._free(buffer, &device_local());
}

/// Transfers up to the bounce size are one synchronous `cuMemcpy`, larger
/// ones stage through the pinned bounce buffer in chunks; both must land at
/// the requested offset and leave the bytes around it alone, for device and
/// managed memory alike.
#[test_case(0, 0; "empty")]
#[test_case(1, 3; "one byte at an offset")]
#[test_case(4096, 8; "page")]
#[test_case(STAGING_BYTES - 1, 1; "just below the bounce size")]
#[test_case(STAGING_BYTES, 0; "exactly the bounce size")]
#[test_case(STAGING_BYTES, 64; "bounce size at an offset")]
#[test_case(STAGING_BYTES + 1, 0; "just above the bounce size")]
#[test_case(2 * STAGING_BYTES + 12345, 4097; "several chunks at an offset")]
fn copies_round_trip_across_the_bounce_threshold(len: usize, offset: usize) {
    let Some(alloc) = cuda_alloc_or_skip() else { return };
    let data = pattern(len);
    let total = offset + len + 16;
    for spec in [device_local(), BufferSpec::default()] {
        let buffer = alloc._alloc(total, &spec, true).unwrap();
        alloc._copyin(&buffer, offset, &data).unwrap();
        let mut back = vec![0xFFu8; total];
        alloc._copyout(&mut back, &buffer, 0).unwrap();
        assert!(back[..offset].iter().chain(&back[offset + len..]).all(|byte| *byte == 0), "bytes around the range");
        assert!(back[offset..offset + len] == data[..], "written range");
        let mut window = vec![0u8; len];
        alloc._copyout(&mut window, &buffer, offset).unwrap();
        assert!(window == data, "offset read");
        alloc._free(buffer, &spec);
    }
}

/// Larger than the pinned bounce buffer, so copies are chunked.
#[test]
fn large_copies_stage_in_chunks() {
    let Some(alloc) = cuda_alloc_or_skip() else { return };
    const LEN: usize = (4 << 20) * 3 + 12345;
    let data = pattern(LEN);
    let buffer = alloc._alloc(LEN, &device_local(), false).unwrap();
    alloc._copyin(&buffer, 0, &data).unwrap();
    let mut back = vec![0u8; LEN];
    alloc._copyout(&mut back, &buffer, 0).unwrap();
    assert!(back == data);
    alloc._free(buffer, &device_local());
}

#[test_case(BufferSpec { cpu_access: false, ..BufferSpec::default() }, true, CudaMemory::Device; "device local")]
#[test_case(BufferSpec { cpu_access: false, ..BufferSpec::default() }, false, CudaMemory::Device; "device local without managed")]
#[test_case(BufferSpec::default(), true, CudaMemory::Managed; "cpu access with coherent managed memory")]
#[test_case(BufferSpec::default(), false, CudaMemory::Pinned; "cpu access without it is pinned")]
#[test_case(BufferSpec { host: true, ..BufferSpec::default() }, true, CudaMemory::Pinned; "host memory is pinned")]
fn memory_kind_honours_cpu_access_without_managed_memory(spec: BufferSpec, managed: bool, expected: CudaMemory) {
    assert_eq!(CudaAllocator::memory_kind(&spec, managed), expected);
}

#[test_case(BufferSpec { cpu_access: false, ..BufferSpec::default() }, CudaMemory::Device, false; "device local")]
#[test_case(BufferSpec::default(), CudaMemory::Managed, true; "host visible is managed")]
#[test_case(BufferSpec { host: true, ..BufferSpec::default() }, CudaMemory::Pinned, true; "host memory is pinned")]
fn buffer_spec_selects_the_memory_kind(spec: BufferSpec, expected: CudaMemory, host_visible: bool) {
    let Some(alloc) = cuda_alloc_or_skip() else { return };
    if expected == CudaMemory::Managed && !alloc.dev.limits().managed_memory {
        eprintln!("skipping: device has no coherent managed memory");
        return;
    }
    let buffer = alloc._alloc(1024, &spec, true).unwrap();
    let RawBuffer::Cuda { memory, host_ptr, device_ptr, .. } = &buffer else { unreachable!() };
    assert_eq!((*memory, host_ptr.is_some(), buffer.cpu_accessible()), (expected, host_visible, host_visible));
    assert_ne!(*device_ptr, 0);
    let data = pattern(1024);
    alloc._copyin(&buffer, 0, &data).unwrap();
    let mut back = vec![0u8; 1024];
    alloc._copyout(&mut back, &buffer, 0).unwrap();
    assert_eq!(back, data);
    if let Some(host) = host_ptr {
        // The host side of the same allocation sees the device copy.
        alloc.dev.synchronize().unwrap();
        let host = unsafe { std::slice::from_raw_parts(host.as_ptr(), 1024) };
        assert_eq!(host, &data[..]);
    }
    alloc._free(buffer, &spec);
}

#[test]
fn zero_initializes_fresh_and_recycled_allocations() {
    let Some(alloc) = cuda_alloc_or_skip() else { return };
    let alloc: Arc<dyn Allocator> = Arc::new(crate::allocator::LruAllocator::new(Box::new((*alloc).clone())));
    let spec = device_local();
    let mut first = Buffer::new_with_zero_init(alloc.clone(), DType::UInt8, vec![256], spec, true);
    first.ensure_allocated().unwrap();
    let mut back = vec![0xFFu8; 256];
    first.copyout(&mut back).unwrap();
    assert!(back.iter().all(|byte| *byte == 0));
    first.copyin(&pattern(256)).unwrap();
    let recycled_base = first.raw_data_ptr();
    drop(first);
    // The LRU hands the same allocation back; it must be re-zeroed on device.
    let second = Buffer::new_with_zero_init(alloc, DType::UInt8, vec![256], spec, true);
    second.ensure_allocated().unwrap();
    assert_eq!(second.raw_data_ptr(), recycled_base, "expected LRU reuse");
    second.copyout(&mut back).unwrap();
    assert!(back.iter().all(|byte| *byte == 0));
}

#[test]
fn transfer_copies_between_and_within_buffers() {
    let Some(alloc) = cuda_alloc_or_skip() else { return };
    let spec = device_local();
    let src = alloc._alloc(64, &spec, false).unwrap();
    let dst = alloc._alloc(64, &spec, true).unwrap();
    let data = pattern(64);
    alloc._copyin(&src, 0, &data).unwrap();
    alloc._transfer(&dst, 8, &src, 0, 32).unwrap();
    let mut back = vec![0u8; 64];
    alloc._copyout(&mut back, &dst, 0).unwrap();
    assert_eq!(&back[8..40], &data[..32]);
    assert!(back[..8].iter().chain(&back[40..]).all(|byte| *byte == 0));
    // Overlapping ranges (memory planning) keep memmove semantics.
    alloc._transfer(&src, 4, &src, 0, 32).unwrap();
    alloc._copyout(&mut back, &src, 0).unwrap();
    assert_eq!(&back[4..36], &data[..32]);
    alloc._transfer(&src, 0, &src, 4, 32).unwrap();
    alloc._copyout(&mut back, &src, 0).unwrap();
    assert_eq!(&back[..32], &data[..32]);
    // A self-copy is a no-op, not a fault.
    alloc._transfer(&src, 0, &src, 0, 64).unwrap();
    let cpu = crate::allocator::CpuAllocator._alloc(64, &BufferSpec::default(), true).unwrap();
    assert!(matches!(alloc._transfer(&cpu, 0, &src, 0, 8), Err(crate::Error::Unsupported { .. })));
    alloc._free(src, &spec);
    alloc._free(dst, &spec);
}

/// Managed buffers expose typed host views (after a device drain); device-only
/// ones report `NotCpuAccessible` instead of faulting.
#[test]
fn buffer_views_expose_typed_host_slices() {
    let Some(alloc) = cuda_alloc_or_skip() else { return };
    let managed = alloc.dev.limits().managed_memory;
    let alloc: Arc<dyn Allocator> = Arc::new((*alloc).clone());
    let values: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let mut buffer = Buffer::new(alloc.clone(), DType::Float32, vec![8], BufferSpec::default());
    buffer.copyin(super::f32_bytes(&values)).unwrap();
    if managed {
        assert_eq!(buffer.as_slice::<f32>().unwrap(), &values[..]);
        let view = buffer.view(8, 8).unwrap();
        assert_eq!(view.as_slice::<f32>().unwrap(), &values[2..4]);
        assert_eq!(unsafe { view.as_raw_ptr() }, unsafe { buffer.as_raw_ptr().add(8) });
        buffer.as_host_bytes_mut().unwrap()[..4].copy_from_slice(&1.5f32.to_le_bytes());
        let mut back = vec![0u8; 32];
        buffer.copyout(&mut back).unwrap();
        assert_eq!(f32::from_le_bytes(back[..4].try_into().unwrap()), 1.5);
    }
    let mut local = Buffer::new(alloc, DType::Float32, vec![8], device_local());
    local.copyin(super::f32_bytes(&values)).unwrap();
    assert!(matches!(local.as_slice::<f32>(), Err(crate::Error::NotCpuAccessible)));
    assert!(matches!(local.as_host_bytes(), Err(crate::Error::NotCpuAccessible)));
    let mut back = vec![0u8; 32];
    local.copyout(&mut back).unwrap();
    assert_eq!(back, super::f32_bytes(&values));
}

#[test]
fn registry_wraps_the_allocator_in_the_lru() {
    let Some(alloc) = cuda_alloc_or_skip() else { return };
    let registered = crate::registry::registry().get(&svod_dtype::DeviceSpec::Cuda { device_id: 0 }).unwrap();
    assert_eq!(registered.name(), "CUDA");
    assert_eq!(registered.device_spec(), svod_dtype::DeviceSpec::Cuda { device_id: 0 });
    assert!(registered.supports_device_local());
    assert_eq!(crate::registry::resolve_cuda_arch(0).unwrap(), alloc.dev.arch());
    let error = crate::registry::resolve_cuda_arch(1 << 20).expect_err("out of range");
    assert!(matches!(error, crate::Error::NoCudaGpu { .. }), "{error:?}");
}

/// Host read/write latency of the sizes a model's host readbacks use; run
/// with `--ignored --nocapture` to print per-transfer timings.
#[test]
#[ignore = "benchmark: prints CUDA host transfer latency"]
fn host_transfer_latency_benchmark() {
    let Some(alloc) = cuda_alloc_or_skip() else { return };
    let managed = alloc.dev.limits().managed_memory;
    for (label, spec) in [("device", device_local()), ("managed", BufferSpec::default())] {
        if label == "managed" && !managed {
            continue;
        }
        for len in [4096usize, 200 << 10, 4 << 20, 12 << 20] {
            let buffer = alloc._alloc(len, &spec, false).unwrap();
            let data = pattern(len);
            let mut back = vec![0u8; len];
            let rounds = if len > (1 << 20) { 50 } else { 1000 };
            let time = |op: &mut dyn FnMut()| {
                let start = std::time::Instant::now();
                for _ in 0..rounds {
                    op();
                }
                start.elapsed().as_secs_f64() * 1e6 / rounds as f64
            };
            let copyin = time(&mut || alloc._copyin(&buffer, 0, &data).unwrap());
            let copyout = time(&mut || alloc._copyout(&mut back, &buffer, 0).unwrap());
            assert_eq!(back, data);
            let other = alloc._alloc(len, &spec, false).unwrap();
            let transfer = time(&mut || alloc._transfer(&other, 0, &buffer, 0, len).unwrap());
            alloc._copyout(&mut back, &other, 0).unwrap();
            assert_eq!(back, data);
            println!(
                "{label:>8} {len:>9} B: copyin {copyin:8.1} us  copyout {copyout:8.1} us  transfer {transfer:8.1} us"
            );
            alloc._free(buffer, &spec);
            alloc._free(other, &spec);
        }
    }
}
