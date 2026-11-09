use std::sync::Arc;

use proptest::prelude::*;
use svod_dtype::DType;
use test_case::test_case;

use crate::error::Error;
use crate::{Buffer, BufferSpec, CpuAllocator};

fn buffer(dtype: DType, len: usize) -> Buffer {
    Buffer::allocate(Arc::new(CpuAllocator), dtype, vec![len], BufferSpec::default()).unwrap()
}

fn i32s(buffer: &Buffer) -> Vec<i32> {
    buffer.as_slice::<i32>().unwrap().to_vec()
}

#[test]
fn copy_from_fill_and_copy_at_store_in_place() {
    let mut buf = buffer(DType::Int32, 4);
    buf.host_write::<i32>().unwrap().copy_from(&[1, 2, 3, 4]).unwrap();
    assert_eq!(i32s(&buf), [1, 2, 3, 4]);
    buf.host_write::<i32>().unwrap().copy_at(1, &[20, 30]).unwrap();
    assert_eq!(i32s(&buf), [1, 20, 30, 4]);
    buf.host_write::<i32>().unwrap().fill(9);
    assert_eq!(i32s(&buf), [9; 4]);
}

#[test_case(3; "short")]
#[test_case(5; "long")]
#[test_case(0; "empty")]
fn copy_from_needs_the_exact_length(len: usize) {
    let mut buf = buffer(DType::Int32, 4);
    let err = buf.host_write::<i32>().unwrap().copy_from(&vec![1; len]).unwrap_err();
    assert!(matches!(err, Error::SizeMismatch { expected: 4, actual } if actual == len), "{err}");
    assert_eq!(i32s(&buf), [0; 4], "nothing is stored");
}

#[test_case(0, 5; "too long")]
#[test_case(3, 2; "past the end")]
#[test_case(usize::MAX, 1; "overflowing offset")]
fn copy_at_stays_in_bounds(offset: usize, len: usize) {
    let mut buf = buffer(DType::Int32, 4);
    let err = buf.host_write::<i32>().unwrap().copy_at(offset, &vec![1; len]).unwrap_err();
    assert!(matches!(err, Error::SizeMismatch { expected: 4, .. }), "{err}");
    assert_eq!(i32s(&buf), [0; 4]);
}

#[test]
fn copy_at_may_end_at_the_end() {
    let mut buf = buffer(DType::Int32, 4);
    buf.host_write::<i32>().unwrap().copy_at(4, &[]).unwrap();
    buf.host_write::<i32>().unwrap().copy_at(2, &[7, 8]).unwrap();
    assert_eq!(i32s(&buf), [0, 0, 7, 8]);
}

#[test]
fn rows_checks_every_row_before_any_store() {
    let mut buf = buffer(DType::Int32, 4);
    let err = buf.host_write::<i32>().unwrap().rows(&[(0, 5), (4, 6)]).unwrap_err();
    assert!(matches!(err, Error::HostWriteRow { row: 4, len: 4 }), "{err}");
    assert_eq!(i32s(&buf), [0; 4], "the in-bounds row is not stored either");
    buf.host_write::<i32>().unwrap().rows(&[(3, 5), (1, 6), (3, 7)]).unwrap();
    assert_eq!(i32s(&buf), [0, 6, 0, 7], "later rows win");
}

#[test]
fn the_typed_guard_checks_the_dtype() {
    let mut buf = buffer(DType::Float32, 2);
    let err = buf.host_write::<i32>().unwrap_err();
    assert!(matches!(err, Error::TypeMismatch { .. }), "{err}");
    let mut bytes = buf.host_write_bytes().unwrap();
    assert_eq!(bytes.len(), 8, "the byte guard spans the whole view");
    bytes.copy_at(4, &2.5f32.to_le_bytes()).unwrap();
    drop(bytes);
    assert_eq!(buf.as_slice::<f32>().unwrap(), [0.0, 2.5]);
}

#[test]
fn a_guard_excludes_every_other_handle_and_view_of_its_storage() {
    let mut buf = buffer(DType::Int32, 4);
    let mut clone = buf.clone();
    let mut view = buf.view(8, 8).unwrap();
    let guard = buf.host_write::<i32>().unwrap();
    for refused in [
        clone.host_write::<i32>().map(drop),
        view.host_write_bytes().map(drop),
        clone.copyin(&[0; 16]),
        clone.as_host_bytes().map(drop),
        view.as_slice::<i32>().map(drop),
        clone.copyout(&mut [0; 16]),
        clone.ensure_no_host_write("launch"),
    ] {
        assert!(matches!(refused, Err(Error::HostWriteLive { .. })), "{refused:?}");
    }
    assert!(clone.is_host_writing() && view.is_host_writing());
    drop(guard);
    assert!(!clone.is_host_writing());
    clone.host_write::<i32>().unwrap().fill(1);
    view.host_write_bytes().unwrap().fill(0);
    assert_eq!(i32s(&buf), [1, 1, 0, 0]);
}

#[test]
fn another_storage_is_not_excluded() {
    let mut a = buffer(DType::Int32, 1);
    let mut b = buffer(DType::Int32, 1);
    let _guard = a.host_write::<i32>().unwrap();
    b.host_write::<i32>().unwrap().fill(3);
    assert_eq!(i32s(&b), [3]);
}

#[test]
fn the_closure_form_releases_the_guard_on_success_and_error() {
    let mut buf = buffer(DType::Int32, 2);
    let len = buf.with_host_write::<i32, _>(|w| w.copy_from(&[4, 5]).map(|()| w.len())).unwrap();
    assert_eq!(len, 2);
    assert!(!buf.is_host_writing());
    let err = buf.with_host_write::<i32, _>(|w| w.copy_from(&[1])).unwrap_err();
    assert!(matches!(err, Error::SizeMismatch { .. }), "{err}");
    assert!(!buf.is_host_writing(), "an error releases the guard");
    assert_eq!(i32s(&buf), [4, 5]);
}

#[test]
fn a_failed_acquire_leaves_the_storage_free() {
    let mut sealed = buffer(DType::Int32, 1);
    sealed.mark_immutable();
    assert!(matches!(sealed.host_write::<i32>(), Err(Error::ImmutableBuffer { .. })));
    let mut misaligned = buffer(DType::Int32, 2).view(1, 4).unwrap();
    let err = misaligned.host_write::<i32>().unwrap_err();
    assert!(matches!(err, Error::MisalignedHostWrite { offset: 1, align: 4 }), "{err}");
    assert!(!misaligned.is_host_writing(), "the claim is released");
    misaligned.host_write_bytes().unwrap().fill(1);
}

proptest! {
    /// `rows` stores exactly what a scatter into a host vector would, or
    /// nothing at all when any row is out of bounds.
    #[test]
    fn rows_match_a_host_scatter(
        len in 1usize..32,
        rows in prop::collection::vec((0usize..40, any::<i32>()), 0..16),
    ) {
        let mut buf = buffer(DType::Int32, len);
        let result = buf.host_write::<i32>().unwrap().rows(&rows);
        let mut expected = vec![0; len];
        if rows.iter().all(|&(row, _)| row < len) {
            prop_assert!(result.is_ok());
            for &(row, value) in &rows {
                expected[row] = value;
            }
        } else {
            prop_assert!(
                matches!(result, Err(Error::HostWriteRow { .. })),
                "an out-of-bounds row must fail"
            );
        }
        prop_assert_eq!(i32s(&buf), expected);
    }
}
