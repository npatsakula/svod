//! Multi-dtype `Tensor::rand` tests — range/finite invariants plus
//! statistical sanity for f16, bf16, f64.

use svod_dtype::{DType, DeviceSpec};

use crate::Tensor;
use crate::rand::manual_seed;

use super::RAND_TEST_LOCK;

const N: usize = 4096;
const TOL: f64 = 0.05;

fn realize_f32_via_cast(t: Tensor, config: &crate::PrepareConfig) -> Vec<f32> {
    let casted = t.cast(DType::Float32);
    casted.realize_with(config).expect("realize");
    casted.as_vec::<f32>().expect("read")
}

fn realize_f64(t: &mut Tensor, config: &crate::PrepareConfig) -> Vec<f64> {
    t.realize_with(config).expect("realize");
    t.as_vec::<f64>().expect("read")
}

fn mean_stddev_f32(v: &[f32]) -> (f64, f64) {
    let n = v.len() as f64;
    let mean = v.iter().map(|&x| x as f64).sum::<f64>() / n;
    let var = v.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n;
    (mean, var.sqrt())
}

fn mean_stddev_f64(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    (mean, var.sqrt())
}

crate::codegen_tests! {
    fn rand_f16_in_unit_interval_and_finite(config) {
        let _g = RAND_TEST_LOCK.lock();
        manual_seed(0xC0DE);
        let t = Tensor::rand_with(&[N], DType::Float16, DeviceSpec::Cpu).unwrap();
        let v = realize_f32_via_cast(t, &config);
        assert_eq!(v.len(), N);
        for (i, &x) in v.iter().enumerate() {
            assert!(x.is_finite(), "rand_f16[{i}] = {x} is non-finite");
            assert!((0.0..1.0).contains(&x), "rand_f16[{i}] = {x} outside [0, 1)");
        }
        let (m, s) = mean_stddev_f32(&v);
        assert!((m - 0.5).abs() < TOL, "rand_f16 mean={m}, expected≈0.5");
        let expected_stddev = 1.0 / 12.0_f64.sqrt();
        assert!((s - expected_stddev).abs() < TOL, "rand_f16 stddev={s}, expected≈{expected_stddev}");
    }

    fn rand_bf16_in_unit_interval_and_finite(config) {
        let _g = RAND_TEST_LOCK.lock();
        manual_seed(0xBEAD);
        let t = Tensor::rand_with(&[N], DType::BFloat16, DeviceSpec::Cpu).unwrap();
        let v = realize_f32_via_cast(t, &config);
        assert_eq!(v.len(), N);
        for (i, &x) in v.iter().enumerate() {
            assert!(x.is_finite(), "rand_bf16[{i}] = {x} is non-finite");
            assert!((0.0..1.0).contains(&x), "rand_bf16[{i}] = {x} outside [0, 1)");
        }
        let (m, s) = mean_stddev_f32(&v);
        assert!((m - 0.5).abs() < TOL, "rand_bf16 mean={m}, expected≈0.5");
        let expected_stddev = 1.0 / 12.0_f64.sqrt();
        assert!((s - expected_stddev).abs() < TOL, "rand_bf16 stddev={s}, expected≈{expected_stddev}");
    }

    fn rand_f64_in_unit_interval_and_finite(config) {
        let _g = RAND_TEST_LOCK.lock();
        manual_seed(0xF064);
        let mut t = Tensor::rand_with(&[N], DType::Float64, DeviceSpec::Cpu).unwrap();
        let v = realize_f64(&mut t, &config);
        assert_eq!(v.len(), N);
        for (i, &x) in v.iter().enumerate() {
            assert!(x.is_finite(), "rand_f64[{i}] = {x} is non-finite");
            assert!((0.0..1.0).contains(&x), "rand_f64[{i}] = {x} outside [0, 1)");
        }
        let (m, s) = mean_stddev_f64(&v);
        assert!((m - 0.5).abs() < TOL, "rand_f64 mean={m}, expected≈0.5");
        let expected_stddev = 1.0 / 12.0_f64.sqrt();
        assert!((s - expected_stddev).abs() < TOL, "rand_f64 stddev={s}, expected≈{expected_stddev}");
    }

    fn rand_f16_never_yields_one(config) {
        let _g = RAND_TEST_LOCK.lock();
        manual_seed(7);
        // f16 has 10 mantissa bits → 1024 distinct mantissa positions, so at
        // N >= 2048 the lower bound (0.0) is statistically guaranteed and the
        // upper bound (1.0) is structurally impossible (mantissa-fill construction
        // produces float ∈ [1.0, 2.0), shift gives [0.0, 1.0)).
        let t = Tensor::rand_with(&[N], DType::Float16, DeviceSpec::Cpu).unwrap();
        let v = realize_f32_via_cast(t, &config);
        assert!(v.iter().all(|&x| x < 1.0), "rand_f16 produced a value >= 1.0");
    }

    fn uniform_with_dtype_produces_target_dtype(config) {
        let _g = RAND_TEST_LOCK.lock();
        manual_seed(0xF16D);
        let t = Tensor::uniform_with_dtype(&[N], -2.0, 5.0, DType::Float16).unwrap();
        assert_eq!(t.uop().dtype(), DType::Float16);
        let v = realize_f32_via_cast(t, &config);
        for (i, &x) in v.iter().enumerate() {
            assert!(x.is_finite(), "uniform_with_dtype f16[{i}] = {x} non-finite");
            // f16's coarse precision means we tolerate near-boundary values; widen
            // the strict (-2.0, 5.0) bounds slightly to absorb f16 rounding.
            assert!((-2.01..5.01).contains(&x), "uniform_with_dtype f16[{i}] = {x} outside [-2, 5)");
        }
    }

    fn kaiming_uniform_with_dtype_produces_f16(config) {
        let _g = RAND_TEST_LOCK.lock();
        manual_seed(0xAA);
        let t = Tensor::kaiming_uniform_with_dtype(&[32, 16, 3, 3], 0.0, DType::Float16).unwrap();
        assert_eq!(t.uop().dtype(), DType::Float16);
        let _ = realize_f32_via_cast(t, &config); // realize check
    }

    fn rand_handles_odd_numel_f16(config) {
        // Odd numel × 2-byte itemsize → ceildiv(odd*2, 4) = (odd+1)/2 u32 words →
        // bitcast produces (odd+1) u16s → bits_to_rand trims to numel. Verifies
        // the truncation path doesn't blow up.
        let _g = RAND_TEST_LOCK.lock();
        manual_seed(13);
        let t = Tensor::rand_with(&[5], DType::Float16, DeviceSpec::Cpu).unwrap();
        let v = realize_f32_via_cast(t, &config);
        assert_eq!(v.len(), 5);
        for &x in &v {
            assert!(x.is_finite() && (0.0..1.0).contains(&x));
        }
    }
}

// Validation-only test — no realize, no codegen variants needed.
#[test]
fn rand_rejects_integer_dtype() {
    let _g = RAND_TEST_LOCK.lock();
    let err = Tensor::rand_with(&[4], DType::Int32, DeviceSpec::Cpu);
    assert!(err.is_err(), "rand on Int32 should error (callers should use randint)");
}

/// A draw past `i32::MAX` words still counts them all, however many a graph
/// holds before anything runs.
#[test]
fn a_draw_past_i32_words_keeps_its_shape() {
    let n = 1usize << 33;
    let t = Tensor::rand_with(&[n], DType::Float32, DeviceSpec::Cpu).expect("a lazy draw");
    assert_eq!(svod_ir::shape::to_vec_usize(&t.shape().unwrap()).unwrap(), [n]);
}
