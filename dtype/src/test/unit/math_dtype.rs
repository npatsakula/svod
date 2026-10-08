//! [`DType::math_dtype`]: the width a pointwise chain is *evaluated* in.

use test_case::test_case;

use crate::DType;

/// A narrow float evaluates in fp32; everything else evaluates in itself, so the
/// promotion is inert wherever there is nothing to gain.
#[test_case(DType::Float16, DType::Float32)]
#[test_case(DType::BFloat16, DType::Float32)]
#[test_case(DType::FP8E4M3, DType::Float32)]
#[test_case(DType::Float32, DType::Float32)]
#[test_case(DType::Float64, DType::Float64)]
#[test_case(DType::WeakFloat, DType::WeakFloat)]
#[test_case(DType::Int32, DType::Int32)]
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
