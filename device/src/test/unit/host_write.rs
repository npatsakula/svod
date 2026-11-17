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

/// `copy_at(offset, src)` splices `src` in when it ends within the buffer and
/// `copy_from(src)` replaces everything when it is exactly as long; either
/// fails with a size mismatch and stores nothing otherwise. `fill` sets every
/// element.
fn copies_match_a_host_splice(init: &[i32], offset: usize, src: &[i32]) {
    let len = init.len();
    let mut buf = buffer(DType::Int32, len);
    buf.host_write::<i32>().unwrap().copy_from(init).unwrap();
    let mut expected = init.to_vec();
    assert_eq!(i32s(&buf), expected);

    let result = buf.host_write::<i32>().unwrap().copy_at(offset, src);
    match offset.checked_add(src.len()).filter(|&end| end <= len) {
        Some(end) => {
            result.unwrap();
            expected[offset..end].copy_from_slice(src);
        }
        None => assert!(
            matches!(result, Err(Error::SizeMismatch { expected, actual })
                if expected == len && actual == offset.saturating_add(src.len())),
            "{result:?}"
        ),
    }
    assert_eq!(i32s(&buf), expected, "copy_at({offset}, {src:?})");

    let result = buf.host_write::<i32>().unwrap().copy_from(src);
    if src.len() == len {
        result.unwrap();
        expected = src.to_vec();
    } else {
        assert!(
            matches!(result, Err(Error::SizeMismatch { expected, actual }) if expected == len && actual == src.len()),
            "{result:?}"
        );
    }
    assert_eq!(i32s(&buf), expected, "copy_from({src:?})");

    buf.host_write::<i32>().unwrap().fill(9);
    assert_eq!(i32s(&buf), vec![9; len]);
}

#[test_case(1, 2; "inside")]
#[test_case(2, 2; "ending at the end")]
#[test_case(4, 0; "empty at the end")]
#[test_case(0, 3; "short")]
#[test_case(0, 5; "too long")]
#[test_case(0, 0; "empty")]
#[test_case(3, 2; "past the end")]
#[test_case(usize::MAX, 1; "overflowing offset")]
fn copies_at_the_boundaries(offset: usize, len: usize) {
    copies_match_a_host_splice(&[1, 2, 3, 4], offset, &(10..).take(len).collect::<Vec<_>>());
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
    #[test]
    fn copies_match_a_host_splice_anywhere(
        init in prop::collection::vec(any::<i32>(), 1..8),
        offset in prop_oneof![9 => 0usize..10, 1 => Just(usize::MAX)],
        src in prop::collection::vec(any::<i32>(), 0..10),
    ) {
        copies_match_a_host_splice(&init, offset, &src);
    }

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
