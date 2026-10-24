//! Arithmetic operation tests.
//!
//! Tests all arithmetic operations including basic ops, type promotion, and error handling.

use std::f32::consts::PI;

use svod_dtype::DType;

use crate::{BinaryOp, ConstValue, Op, UOp, error::Error, uop::eval::eval_binary_op}; // ConstValue kept for Void, Float16, i8, u8

// =========================================================================
// Basic Arithmetic Operations
// =========================================================================

#[test]
fn test_add_same_type() {
    assert_eq!(UOp::native_const(5i32).try_add(&UOp::native_const(3i32)).unwrap().dtype(), DType::Int32);
}

#[test]
fn weak_sources_remain_uncast_until_lowering() {
    for result in [
        UOp::const_(DType::WeakInt, ConstValue::Int(7)).try_add(&UOp::native_const(2i32)).unwrap(),
        UOp::const_(DType::WeakFloat, ConstValue::Float(-0.0)).try_add(&UOp::native_const(1.0f32)).unwrap(),
    ] {
        let Op::Binary(BinaryOp::Add, lhs, rhs) = result.op() else { panic!("expected ADD") };
        assert!(matches!(lhs.op(), Op::Const(_)), "weak lhs must remain direct: {lhs:?}");
        assert!(matches!(rhs.op(), Op::Const(_)), "strong rhs must remain a constant: {rhs:?}");
        assert!(lhs.dtype().is_weak());
        assert!(!rhs.dtype().is_weak());
        assert_eq!(result.dtype(), rhs.dtype());
        assert!(!result.toposort().iter().any(|node| matches!(node.op(), Op::Cast(..))));
    }
}

#[test]
fn test_sub_same_type() {
    assert_eq!(UOp::native_const(10.0f32).try_sub(&UOp::native_const(3.0f32)).unwrap().dtype(), DType::Float32);
}

#[test]
fn test_mul_same_type() {
    assert_eq!(UOp::native_const(4i32).try_mul(&UOp::native_const(5i32)).unwrap().dtype(), DType::Int32);
}

#[test]
fn test_mod_same_type() {
    assert_eq!(UOp::native_const(10i32).try_mod(&UOp::native_const(3i32)).unwrap().dtype(), DType::Int32);
}

#[test]
fn test_idiv_same_type() {
    assert_eq!(UOp::native_const(10i32).try_div(&UOp::native_const(3i32)).unwrap().dtype(), DType::Int32);
}

#[test]
fn test_integer_division_operation_split() {
    let a = UOp::native_const(-9i32);
    let b = UOp::native_const(5i32);
    assert!(matches!(a.try_div(&b).unwrap().op(), Op::Binary(BinaryOp::FloorDiv, ..)));
    assert!(matches!(a.try_mod(&b).unwrap().op(), Op::Binary(BinaryOp::FloorMod, ..)));
    assert!(matches!(a.try_cdiv(&b).unwrap().op(), Op::Binary(BinaryOp::CDiv, ..)));
    assert!(matches!(a.try_cmod(&b).unwrap().op(), Op::Binary(BinaryOp::CMod, ..)));
}

#[test]
fn test_signed_floor_divmod_semantics() {
    for a in -12i64..=12 {
        for b in -6i64..=6 {
            if b == 0 {
                continue;
            }
            let floor_div = match eval_binary_op(BinaryOp::FloorDiv, ConstValue::Int(a), ConstValue::Int(b)) {
                Some(ConstValue::Int(v)) => v,
                other => panic!("unexpected floor div result: {other:?}"),
            };
            let floor_mod = match eval_binary_op(BinaryOp::FloorMod, ConstValue::Int(a), ConstValue::Int(b)) {
                Some(ConstValue::Int(v)) => v,
                other => panic!("unexpected floor mod result: {other:?}"),
            };
            let cdiv = match eval_binary_op(BinaryOp::CDiv, ConstValue::Int(a), ConstValue::Int(b)) {
                Some(ConstValue::Int(v)) => v,
                other => panic!("unexpected C div result: {other:?}"),
            };
            let cmod = match eval_binary_op(BinaryOp::CMod, ConstValue::Int(a), ConstValue::Int(b)) {
                Some(ConstValue::Int(v)) => v,
                other => panic!("unexpected C mod result: {other:?}"),
            };

            assert_eq!(a, floor_div * b + floor_mod);
            assert!(floor_mod == 0 || (floor_mod < 0) == (b < 0));
            assert_eq!(cdiv, a / b);
            assert_eq!(cmod, a % b);
        }
    }
}

/// The integer bf16 narrowing replaces exactly the f32 → bf16 cast: no cast to
/// bf16 survives and the payload arrives by bitcast, while the other float
/// casts (f32 → f16, bf16 → f32) stay for the backend to select.
#[test]
fn bf16_integer_casts_rewrite_only_f32_to_bf16() {
    let patterns = crate::decompositions::bf16_integer_cast_patterns();
    let casts_to = |root: &std::sync::Arc<UOp>, dt: &DType| {
        root.toposort().iter().filter(|u| matches!(u.op(), Op::Cast(_)) && u.dtype() == *dt).count()
    };
    let narrow = crate::decompositions::decompose_with(&UOp::native_const(0.25f32).cast(DType::BFloat16), &patterns);
    assert_eq!(narrow.dtype(), DType::BFloat16);
    assert!(matches!(narrow.op(), Op::BitCast(_)), "{narrow:?}");
    assert_eq!(casts_to(&narrow, &DType::BFloat16), 0);
    for (from, to) in [(DType::Float32, DType::Float16), (DType::BFloat16, DType::Float32)] {
        let root = UOp::native_const(0.25f32).cast(from).cast(to.clone());
        let kept = crate::decompositions::decompose_with(&root, &patterns);
        assert_eq!(casts_to(&kept, &to), 1, "{to:?} cast must stay");
    }
}

#[test]
fn test_floor_divmod_decompose_to_c_ops() {
    let a = UOp::define_var("a".to_string(), -20, 20);
    let b = UOp::define_var("b".to_string(), 1, 7);
    for root in [a.try_div(&b).unwrap(), a.try_mod(&b).unwrap()] {
        let lowered =
            crate::decompositions::decompose_with(&root, &crate::decompositions::divmod_decomposition_patterns());
        assert!(
            !lowered
                .toposort()
                .iter()
                .any(|u| { matches!(u.op(), Op::Binary(BinaryOp::FloorDiv | BinaryOp::FloorMod, ..)) })
        );
        assert!(lowered.toposort().iter().any(|u| matches!(u.op(), Op::Binary(BinaryOp::CDiv | BinaryOp::CMod, ..))));
    }
}

#[test]
fn test_fdiv_same_type() {
    assert_eq!(UOp::native_const(10.0f32).try_div(&UOp::native_const(3.0f32)).unwrap().dtype(), DType::Float32);
}

#[test]
fn test_max_same_type() {
    assert_eq!(UOp::native_const(10i32).try_max(&UOp::native_const(20i32)).unwrap().dtype(), DType::Int32);
}

#[test]
fn test_pow_same_type() {
    assert_eq!(UOp::native_const(2.0f32).try_pow(&UOp::native_const(3.0f32)).unwrap().dtype(), DType::Float32);
}

// =========================================================================
// Unary Operations
// =========================================================================

#[test]
fn test_neg_int() {
    let result = UOp::native_const(5i32).neg();
    assert_eq!(result.dtype(), DType::Int32);
}

#[test]
fn test_neg_float() {
    let result = UOp::native_const(PI).neg();
    assert_eq!(result.dtype(), DType::Float32);
}

// =========================================================================
// Type Promotion Tests
// =========================================================================

#[test]
fn test_add_type_promotion_int_to_float() {
    let int_val = UOp::native_const(5i32);
    let float_val = UOp::native_const(PI);

    let result = int_val.try_add(&float_val).unwrap();
    // Int32 should promote to Float32
    assert_eq!(result.dtype(), DType::Float32);
}

#[test]
fn test_mul_type_promotion_smaller_to_larger() {
    let small = UOp::const_(DType::Int8, ConstValue::Int(5));
    let large = UOp::native_const(10i32);

    let result = small.try_mul(&large).unwrap();
    // Int8 should promote to Int32
    assert_eq!(result.dtype(), DType::Int32);
}

#[test]
fn test_sub_type_promotion_uint_to_int() {
    let uint_val = UOp::const_(DType::UInt8, ConstValue::UInt(5));
    let int_val = UOp::native_const(10i32);

    let result = uint_val.try_sub(&int_val).unwrap();
    // UInt8 should promote to Int32
    assert_eq!(result.dtype(), DType::Int32);
}

// =========================================================================
// Division by Zero Tests
// =========================================================================

#[test]
fn test_idiv_by_zero() {
    let numerator = UOp::native_const(10i32);
    let zero = UOp::native_const(0i32);

    let result = numerator.try_div(&zero);
    assert!(matches!(result, Err(Error::DivisionByZero)));
}

#[test]
fn test_fdiv_by_zero() {
    let numerator = UOp::native_const(10.0f32);
    let zero = UOp::native_const(0.0f32);

    let result = numerator.try_div(&zero);
    assert!(matches!(result, Err(Error::DivisionByZero)));
}

#[test]
fn test_mod_by_zero() {
    let numerator = UOp::native_const(10i32);
    let zero = UOp::native_const(0i32);

    let result = numerator.try_mod(&zero);
    assert!(matches!(result, Err(Error::DivisionByZero)));
}

// =========================================================================
// Void Type Error Tests
// =========================================================================

#[test]
fn test_add_void_type() {
    let void_val = UOp::const_(DType::Void, ConstValue::Int(0));
    let int_val = UOp::const_(DType::Int32, ConstValue::Int(5));

    let result = void_val.try_add(&int_val);
    assert!(matches!(result, Err(Error::VoidTypeInOp)));
}

#[test]
fn test_mul_void_type() {
    let void_val = UOp::const_(DType::Void, ConstValue::Int(0));
    let float_val = UOp::const_(DType::Float32, ConstValue::Float(PI as f64));

    let result = void_val.try_mul(&float_val);
    assert!(matches!(result, Err(Error::VoidTypeInOp)));
}

// =========================================================================
// Mixed Type Tests
// =========================================================================

#[test]
fn test_add_bool_and_int() {
    let bool_val = UOp::native_const(true);
    let int_val = UOp::native_const(5i32);

    let result = bool_val.try_add(&int_val).unwrap();
    // Bool should promote to Int32
    assert_eq!(result.dtype(), DType::Int32);
}

#[test]
fn test_mul_different_float_types() {
    let f16 = UOp::const_(DType::Float16, ConstValue::Float(2.0));
    let f32 = UOp::const_(DType::Float32, ConstValue::Float(3.0));

    let result = f16.try_mul(&f32).unwrap();
    // Float16 should promote to Float32
    assert_eq!(result.dtype(), DType::Float32);
}

// =========================================================================
// Operation Chaining Tests
// =========================================================================

#[test]
fn test_chained_operations() {
    let a = UOp::native_const(10i32);
    let b = UOp::native_const(5i32);
    let c = UOp::native_const(2i32);

    // (a + b) * c
    let sum = a.try_add(&b).unwrap();
    let product = sum.try_mul(&c).unwrap();
    assert_eq!(product.dtype(), DType::Int32);
}

#[test]
fn test_chained_with_promotion() {
    let int_val = UOp::native_const(10i32);
    let float_val = UOp::native_const(2.5f32);

    // int + float -> Float32
    let sum = int_val.try_add(&float_val).unwrap();
    assert_eq!(sum.dtype(), DType::Float32);

    // Float32 * Int32 -> Float32
    let product = sum.try_mul(&int_val).unwrap();
    assert_eq!(product.dtype(), DType::Float32);
}
