//! [`DType::math_dtype`]: the width a pointwise chain is *evaluated* in.

use test_case::test_case;

use crate::{DType, ScalarDType};

#[test_case(ScalarDType::Float16, true)]
#[test_case(ScalarDType::BFloat16, true)]
#[test_case(ScalarDType::FP8E4M3, true)]
#[test_case(ScalarDType::FP8E5M2FNUZ, true)]
#[test_case(ScalarDType::Float32, false)]
#[test_case(ScalarDType::Float64, false)]
#[test_case(ScalarDType::WeakFloat, false)]
#[test_case(ScalarDType::Int16, false)]
#[test_case(ScalarDType::Bool, false)]
fn narrow_floats_are_the_ones_below_fp32(scalar: ScalarDType, narrow: bool) {
    assert_eq!(scalar.is_narrow_float(), narrow);
    assert_eq!(DType::Scalar(scalar).is_narrow_float(), narrow);
}

/// A narrow float evaluates in fp32; everything else evaluates in itself, so the
/// promotion is inert wherever there is nothing to gain.
#[test_case(DType::Float16, DType::Float32)]
#[test_case(DType::BFloat16, DType::Float32)]
#[test_case(DType::FP8E4M3, DType::Float32)]
#[test_case(DType::Float32, DType::Float32)]
#[test_case(DType::Float64, DType::Float64)]
#[test_case(DType::WeakFloat, DType::WeakFloat)]
#[test_case(DType::Int32, DType::Int32)]
#[test_case(DType::Bool, DType::Bool)]
fn math_dtype_widens_only_the_narrow_floats(dtype: DType, want: DType) {
    assert_eq!(dtype.math_dtype(), want);
}

/// Widening keeps the lane structure: a widened chain over a vector stays a
/// vector, so the cast back can pair with the cast in.
#[test]
fn math_dtype_preserves_the_vector_count() {
    let f16x4 = DType::Float16.vec(4).expect("f16 vectorizes");
    assert!(f16x4.is_narrow_float());
    assert_eq!(f16x4.math_dtype(), DType::Float32.vec(4).expect("f32 vectorizes"));

    let f32x4 = DType::Float32.vec(4).expect("f32 vectorizes");
    assert_eq!(f32x4.math_dtype(), f32x4);
}

/// Pointers are not values and must survive untouched — `with_base` would flatten
/// one into a scalar.
#[test]
fn math_dtype_leaves_a_pointer_alone() {
    let ptr = DType::Float16.ptr(Some(16), crate::AddrSpace::Global).expect("f16 pointer");
    assert_eq!(ptr.math_dtype(), ptr);
}
