use crate::{Buffer, BufferSpec, CpuAllocator};
use std::sync::Arc;
use svod_dtype::DType;

#[test]
fn test_lazy_allocation() {
    let allocator = Arc::new(CpuAllocator);
    let buffer = Buffer::new(allocator, DType::Float32, vec![10], BufferSpec::default());

    assert!(!buffer.is_allocated());
    buffer.ensure_allocated().unwrap();
    assert!(buffer.is_allocated());
}

#[test]
fn test_buffer_alias() {
    let allocator = Arc::new(CpuAllocator);
    let buffer = Buffer::allocate(allocator, DType::Float32, vec![10], BufferSpec::default()).unwrap();

    let view = buffer.view(4, 16).unwrap();
    assert_eq!(view.offset(), 4);
    assert_eq!(view.size(), 16);
}

#[test]
fn test_view_has_distinct_handle_id() {
    // Each Buffer value (including views) carries its own handle id.
    // Disjoint views of one allocation must compare as different handles
    // so the parallel-hazard model can treat them as independent.
    let allocator = Arc::new(CpuAllocator);
    let buffer = Buffer::allocate(allocator, DType::Float32, vec![16], BufferSpec::default()).unwrap();
    let view_a = buffer.view(0, 16).unwrap();
    let view_b = buffer.view(16, 16).unwrap();

    assert_ne!(buffer.id(), view_a.id(), "view must have a fresh handle id distinct from its base");
    assert_ne!(view_a.id(), view_b.id(), "two distinct views must have distinct handle ids");
}

#[test]
fn test_view_shares_storage_id() {
    // Storage identity must be shared between a base and its views; this is
    // what alias detection in the memory planner relies on.
    let allocator = Arc::new(CpuAllocator);
    let buffer = Buffer::allocate(allocator, DType::Float32, vec![16], BufferSpec::default()).unwrap();
    let view = buffer.view(8, 16).unwrap();

    assert_eq!(buffer.storage_id(), view.storage_id(), "view must share its base's storage id");
}

#[test]
fn test_independent_buffers_have_distinct_storage_ids() {
    let allocator = Arc::new(CpuAllocator);
    let a = Buffer::allocate(allocator.clone(), DType::Float32, vec![8], BufferSpec::default()).unwrap();
    let b = Buffer::allocate(allocator, DType::Float32, vec![8], BufferSpec::default()).unwrap();

    assert_ne!(a.storage_id(), b.storage_id(), "independent allocations must have distinct storage ids");
    assert_ne!(a.id(), b.id());
}

#[test]
fn test_invalid_view() {
    let allocator = Arc::new(CpuAllocator);
    let buffer = Buffer::allocate(allocator, DType::Float32, vec![10], BufferSpec::default()).unwrap();

    // Try to create a view that exceeds buffer size
    let result = buffer.view(36, 16);
    assert!(result.is_err());
}

fn byte_buffer(allocator: Arc<CpuAllocator>, len: usize) -> Buffer {
    Buffer::allocate(allocator, DType::UInt8, vec![len], BufferSpec::default()).unwrap()
}

#[test]
fn test_copyin_at_writes_region_and_checks_bounds() {
    let mut buffer = byte_buffer(Arc::new(CpuAllocator), 8);
    buffer.copyin(&[0; 8]).unwrap();

    buffer.copyin_at(3, &[1, 2, 3]).unwrap();
    let mut actual = [0; 8];
    buffer.copyout(&mut actual).unwrap();
    assert_eq!(actual, [0, 0, 0, 1, 2, 3, 0, 0]);

    assert!(buffer.copyin_at(7, &[1, 2]).is_err());
    assert!(buffer.copyin_at(usize::MAX, &[1]).is_err());
}

#[test]
fn test_copy_region_from_copies_partial_regions_and_checks_both_bounds() {
    let allocator = Arc::new(CpuAllocator);
    let mut src = byte_buffer(allocator.clone(), 8);
    let mut dst = byte_buffer(allocator, 8);
    src.copyin(&[0, 1, 2, 3, 4, 5, 6, 7]).unwrap();
    dst.copyin(&[9; 8]).unwrap();

    dst.copy_region_from(2, &src, 3, 3).unwrap();
    let mut actual = [0; 8];
    dst.copyout(&mut actual).unwrap();
    assert_eq!(actual, [9, 9, 3, 4, 5, 9, 9, 9]);

    assert!(dst.copy_region_from(6, &src, 0, 3).is_err());
    assert!(dst.copy_region_from(0, &src, 7, 2).is_err());
    assert!(dst.copy_region_from(usize::MAX, &src, 0, 1).is_err());
}

#[test]
fn test_copy_within_allows_non_overlapping_regions_and_rejects_overlap() {
    let mut buffer = byte_buffer(Arc::new(CpuAllocator), 8);
    buffer.copyin(&[0, 1, 2, 3, 4, 5, 6, 7]).unwrap();

    buffer.copy_within(4, 0, 4).unwrap();
    let mut actual = [0; 8];
    buffer.copyout(&mut actual).unwrap();
    assert_eq!(actual, [0, 1, 2, 3, 0, 1, 2, 3]);

    assert!(buffer.copy_within(2, 0, 4).is_err());
    assert!(buffer.copy_within(7, 0, 2).is_err());
    assert!(buffer.copy_within(0, usize::MAX, 1).is_err());
}

#[test]
fn test_fork_views_preserves_geometry_and_contents() {
    let allocator = Arc::new(CpuAllocator);
    let base = Buffer::allocate(allocator, DType::Float32, vec![8], BufferSpec::default()).unwrap();
    let mut head = base.view(0, 16).unwrap();
    let mut tail = base.view(16, 16).unwrap();
    head.copyin(&[1u8; 16]).unwrap();
    tail.copyin(&[2u8; 16]).unwrap();

    // Snapshot fork: ONE fresh storage, every view re-minted at its offset
    // with the original bytes.
    let forked = Buffer::fork_views(&[&head, &tail], true).unwrap();
    assert_eq!(forked.len(), 2);
    assert_eq!(forked[0].storage_id(), forked[1].storage_id(), "views must land on one storage");
    assert_ne!(forked[0].storage_id(), base.storage_id());
    assert_eq!((forked[0].offset(), forked[1].offset()), (0, 16));
    let mut bytes = [0u8; 16];
    forked[1].copyout(&mut bytes).unwrap();
    assert_eq!(bytes, [2u8; 16]);

    // A bare fork shares nothing: writes to it never reach the original.
    let mut bare = Buffer::fork_views(&[&head], false).unwrap();
    bare[0].copyin(&[7u8; 16]).unwrap();
    head.copyout(&mut bytes).unwrap();
    assert_eq!(bytes, [1u8; 16]);
}

#[test]
fn test_fork_views_rejects_mixed_storages() {
    let allocator = Arc::new(CpuAllocator);
    let left = Buffer::allocate(allocator.clone(), DType::Float32, vec![4], BufferSpec::default()).unwrap();
    let right = Buffer::allocate(allocator, DType::Float32, vec![4], BufferSpec::default()).unwrap();
    assert!(Buffer::fork_views(&[&left, &right], false).is_err());
}

#[test]
fn test_mark_immutable_blocks_host_writes_but_not_forks() {
    let allocator = Arc::new(CpuAllocator);
    let mut buffer = Buffer::allocate(allocator, DType::Float32, vec![4], BufferSpec::default()).unwrap();
    buffer.copyin(&[7u8; 16]).unwrap();
    buffer.mark_immutable();

    assert!(buffer.copyin(&[0u8; 16]).is_err());
    assert!(buffer.copy_within(0, 8, 8).is_err());
    let mut out = [0u8; 16];
    buffer.copyout(&mut out).unwrap();
    assert_eq!(out, [7u8; 16], "reads must stay legal after sealing");

    // Forking mints fresh, MUTABLE storage — the way to get a private copy.
    let mut fork = Buffer::fork_views(&[&buffer], true).unwrap().remove(0);
    assert!(!fork.is_immutable());
    fork.copyin(&[1u8; 16]).unwrap();
}
