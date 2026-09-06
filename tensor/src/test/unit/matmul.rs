use crate::*;
use ndarray::{Array2, Array4, array};
use svod_dtype::DType;
use svod_ir::ops;
use svod_schedule::{
    BeamConfig, HeuristicsConfig, OptStrategy, OptimizerConfig, TcOptLevel, TcSelect, testing::setup_test_tracing,
};
use test_case::test_case;

fn prep_config(optimizer: OptimizerConfig) -> PrepareConfig {
    optimizer.into()
}
fn env_config() -> PrepareConfig {
    PrepareConfig::from_env()
}

/// Helper to compare svod result against ndarray reference with tolerance.
fn assert_matmul_close(actual: &[f32], expected: &Array2<f32>, tol: f32) {
    let expected_flat: Vec<f32> = expected.iter().copied().collect();
    assert_eq!(actual.len(), expected_flat.len(), "Length mismatch: {} != {}", actual.len(), expected_flat.len());

    for (i, (a, e)) in actual.iter().zip(expected_flat.iter()).enumerate() {
        assert!((a - e).abs() < tol, "Mismatch at index {}: svod={} vs ndarray={} (diff: {})", i, a, e, (a - e).abs());
    }
}

/// Helper to run validated square matmul test for a given size.
fn run_validated_square_matmul(size: usize, tol: f32) {
    // Use prime modulos to create varied but reproducible data
    let a_data: Vec<f32> = (0..size * size).map(|x| ((x % 31) as f32) * 0.05 - 0.8).collect();
    let b_data: Vec<f32> = (0..size * size).map(|x| ((x % 37) as f32) * 0.04 - 0.7).collect();

    let a_nd = Array2::from_shape_vec((size, size), a_data).unwrap();
    let b_nd = Array2::from_shape_vec((size, size), b_data).unwrap();
    let a = Tensor::from_ndarray(&a_nd);
    let b = Tensor::from_ndarray(&b_nd);

    let config = env_config();
    let mut c = a.matmul(&b).unwrap();
    c.realize_with(&config).unwrap();

    let expected = a_nd.dot(&b_nd);

    assert_matmul_close(&c.as_vec::<f32>().unwrap(), &expected, tol);
}

/// Helper to run validated non-square matmul test.
fn run_validated_matmul(m: usize, k: usize, n: usize, tol: f32) {
    let a_data: Vec<f32> = (0..m * k).map(|x| ((x % 41) as f32) * 0.04 - 0.8).collect();
    let b_data: Vec<f32> = (0..k * n).map(|x| ((x % 43) as f32) * 0.035 - 0.7).collect();

    let a_nd = Array2::from_shape_vec((m, k), a_data).unwrap();
    let b_nd = Array2::from_shape_vec((k, n), b_data).unwrap();
    let a = Tensor::from_ndarray(&a_nd);
    let b = Tensor::from_ndarray(&b_nd);

    let config = env_config();
    let mut c = a.matmul(&b).unwrap();
    c.realize_with(&config).unwrap();

    let c_shape = c.shape().unwrap();
    assert_eq!(c_shape[0].as_const().unwrap(), m, "Output shape mismatch");
    assert_eq!(c_shape[1].as_const().unwrap(), n, "Output shape mismatch");

    let expected = a_nd.dot(&b_nd);

    assert_matmul_close(&c.as_vec::<f32>().unwrap(), &expected, tol);
}

// =========================================================================
// Validated matmul tests (codegen required)
// =========================================================================

crate::codegen_tests! {
    fn test_matmul_validated_2x2(config) {
        // Simple 2x2 matmul with known values
        let a_nd = Array2::from_shape_vec((2, 2), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let b_nd = Array2::from_shape_vec((2, 2), vec![5.0f32, 6.0, 7.0, 8.0]).unwrap();

        // Compute with svod
        let a = Tensor::from_ndarray(&a_nd);
        let b = Tensor::from_ndarray(&b_nd);
        let mut c = a.matmul(&b).unwrap();
    c.realize_with(&config).unwrap();

        // Compute reference with ndarray
        let expected = a_nd.dot(&b_nd);

        // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
        assert_matmul_close(&c.as_vec::<f32>().unwrap(), &expected, 1e-5);
    }

    fn test_matmul_int8_returns_narrow_dtype(config) {
        // int8·int8 must return int8 (the promoted operand dtype), not the widened
        // int32 sum accumulator. [[1,2],[3,4]]·[[5,6],[7,8]] = [[19,22],[43,50]] (fit i8).
        let a = Tensor::from_ndarray(&Array2::from_shape_vec((2, 2), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap())
            .cast(DType::Int8)
            .unwrap();
        let b = Tensor::from_ndarray(&Array2::from_shape_vec((2, 2), vec![5.0f32, 6.0, 7.0, 8.0]).unwrap())
            .cast(DType::Int8)
            .unwrap();
        let mut c = a.matmul(&b).unwrap();
        assert_eq!(c.uop().dtype(), DType::Int8, "int8 matmul must return int8, not the int32 accumulator");
        c.realize_with(&config).unwrap();
        assert_eq!(c.as_vec::<i8>().unwrap(), vec![19i8, 22, 43, 50]);
    }

    fn test_matmul_validated_3x3(config) {
        // 3x3 matmul with sequential values
        let a_data: Vec<f32> = (1..=9).map(|x| x as f32).collect();
        let b_data: Vec<f32> = (10..=18).map(|x| x as f32).collect();

        let a_nd = Array2::from_shape_vec((3, 3), a_data).unwrap();
        let b_nd = Array2::from_shape_vec((3, 3), b_data).unwrap();
        let a = Tensor::from_ndarray(&a_nd);
        let b = Tensor::from_ndarray(&b_nd);
        let mut c = a.matmul(&b).unwrap();
    c.realize_with(&config).unwrap();

        let expected = a_nd.dot(&b_nd);

        assert_matmul_close(&c.as_vec::<f32>().unwrap(), &expected, 1e-4);
    }

    fn test_matmul_validated_2x3_3x4(config) {
        // [2, 3] @ [3, 4] -> [2, 4]
        let a_data: Vec<f32> = (1..=6).map(|x| x as f32).collect();
        let b_data: Vec<f32> = (1..=12).map(|x| x as f32).collect();

        let a_nd = Array2::from_shape_vec((2, 3), a_data).unwrap();
        let b_nd = Array2::from_shape_vec((3, 4), b_data).unwrap();
        let a = Tensor::from_ndarray(&a_nd);
        let b = Tensor::from_ndarray(&b_nd);
        let mut c = a.matmul(&b).unwrap();
    c.realize_with(&config).unwrap();

        let expected = a_nd.dot(&b_nd);

        assert_matmul_close(&c.as_vec::<f32>().unwrap(), &expected, 1e-4);
    }

    fn test_matmul_validated_tall_wide(config) {
        // [4, 2] @ [2, 5] -> [4, 5]
        let a_data: Vec<f32> = (1..=8).map(|x| x as f32 * 0.5).collect();
        let b_data: Vec<f32> = (1..=10).map(|x| x as f32 * 0.3).collect();

        let a_nd = Array2::from_shape_vec((4, 2), a_data).unwrap();
        let b_nd = Array2::from_shape_vec((2, 5), b_data).unwrap();
        let a = Tensor::from_ndarray(&a_nd);
        let b = Tensor::from_ndarray(&b_nd);
        let mut c = a.matmul(&b).unwrap();
    c.realize_with(&config).unwrap();

        let expected = a_nd.dot(&b_nd);

        assert_matmul_close(&c.as_vec::<f32>().unwrap(), &expected, 1e-5);
    }

    fn test_matmul_validated_16x16(config) {
        // Larger matrix to test vectorization paths
        const SIZE: usize = 16;
        let a_data: Vec<f32> = (0..SIZE * SIZE).map(|x| (x as f32) * 0.1).collect();
        let b_data: Vec<f32> = (0..SIZE * SIZE).map(|x| (x as f32) * 0.05 + 1.0).collect();

        let a_nd = Array2::from_shape_vec((SIZE, SIZE), a_data).unwrap();
        let b_nd = Array2::from_shape_vec((SIZE, SIZE), b_data).unwrap();
        let a = Tensor::from_ndarray(&a_nd);
        let b = Tensor::from_ndarray(&b_nd);
        let mut c = a.matmul(&b).unwrap();
    c.realize_with(&config).unwrap();

        let expected = a_nd.dot(&b_nd);

        assert_matmul_close(&c.as_vec::<f32>().unwrap(), &expected, 1e-3);
    }

    fn test_matmul_validated_32x32(config) {
        // Test with 32x32 to exercise more optimization paths
        const SIZE: usize = 32;
        let a_data: Vec<f32> = (0..SIZE * SIZE).map(|x| ((x % 17) as f32) * 0.1 - 0.8).collect();
        let b_data: Vec<f32> = (0..SIZE * SIZE).map(|x| ((x % 13) as f32) * 0.15 - 0.5).collect();

        let a_nd = Array2::from_shape_vec((SIZE, SIZE), a_data).unwrap();
        let b_nd = Array2::from_shape_vec((SIZE, SIZE), b_data).unwrap();
        let a = Tensor::from_ndarray(&a_nd);
        let b = Tensor::from_ndarray(&b_nd);
        let mut c = a.matmul(&b).unwrap();
    c.realize_with(&config).unwrap();

        let expected = a_nd.dot(&b_nd);

        assert_matmul_close(&c.as_vec::<f32>().unwrap(), &expected, 1e-2);
    }

    fn test_dot_product_validated(config) {
        // 1D @ 1D dot product
        let a_data = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b_data = [2.0f32, 3.0, 4.0, 5.0, 6.0];

        let a = Tensor::from_slice(a_data);
        let b = Tensor::from_slice(b_data);
        let mut c = a.dot(&b).unwrap();
        c.realize_with(&config).unwrap();

        // Expected: 1*2 + 2*3 + 3*4 + 4*5 + 5*6 = 2 + 6 + 12 + 20 + 30 = 70
        let expected: f32 = a_data.iter().zip(b_data.iter()).map(|(a, b)| a * b).sum();

        assert_eq!(c.shape().unwrap().len(), 0, "Dot product should be scalar");
        let result = c.as_vec::<f32>().unwrap();
        assert!((result[0] - expected).abs() < 1e-5, "Expected {}, got {}", expected, result[0]);
    }

    fn test_vector_matrix_validated(config) {
        // [4] @ [4, 3] -> [3]
        let v_data = [1.0f32, 2.0, 3.0, 4.0];
        let m_data: Vec<f32> = (1..=12).map(|x| x as f32).collect();

        let v = Tensor::from_slice(v_data);
        let m_nd = Array2::from_shape_vec((4, 3), m_data).unwrap();
        let m = Tensor::from_ndarray(&m_nd);
        let mut c = v.dot(&m).unwrap();
        c.realize_with(&config).unwrap();

        // ndarray: need to treat vector as [1, 4] @ [4, 3] -> [1, 3], then squeeze
        let v_nd = ndarray::Array1::from_vec(v_data.to_vec());
        let expected = v_nd.dot(&m_nd);

        assert_eq!(c.shape().unwrap()[0].as_const().unwrap(), 3);
        let svod_result = c.as_vec::<f32>().unwrap();
        for (i, (a, e)) in svod_result.iter().zip(expected.iter()).enumerate() {
            assert!((a - e).abs() < 1e-5, "Mismatch at index {}: {} != {}", i, a, e);
        }
    }

    fn test_matrix_vector_validated(config) {
        // [3, 4] @ [4] -> [3]
        let m_data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let v_data = [1.0f32, 2.0, 3.0, 4.0];

        let m_nd = Array2::from_shape_vec((3, 4), m_data).unwrap();
        let m = Tensor::from_ndarray(&m_nd);
        let v = Tensor::from_slice(v_data);
        let mut c = m.dot(&v).unwrap();
        c.realize_with(&config).unwrap();

        let v_nd = ndarray::Array1::from_vec(v_data.to_vec());
        let expected = m_nd.dot(&v_nd);

        assert_eq!(c.shape().unwrap()[0].as_const().unwrap(), 3);
        let svod_result = c.as_vec::<f32>().unwrap();
        for (i, (a, e)) in svod_result.iter().zip(expected.iter()).enumerate() {
            assert!((a - e).abs() < 1e-5, "Mismatch at index {}: {} != {}", i, a, e);
        }
    }

    fn test_matmul_identity_validated(config) {
        // A @ I = A
        let a_data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let identity_data = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];

        let a_nd = Array2::from_shape_vec((4, 4), a_data.clone()).unwrap();
        let i_nd = Array2::from_shape_vec((4, 4), identity_data).unwrap();
        let a = Tensor::from_ndarray(&a_nd);
        let i = Tensor::from_ndarray(&i_nd);
        let mut c = a.matmul(&i).unwrap();
        c.realize_with(&config).unwrap();
        let svod_result = c.as_vec::<f32>().unwrap();

        // Result should equal original A
        for (i, (actual, expected)) in svod_result.iter().zip(a_data.iter()).enumerate() {
            assert!((actual - expected).abs() < 1e-5, "Mismatch at index {}: {} != {}", i, actual, expected);
        }
    }

    fn test_matmul_negative_values_validated(config) {
        // Test with negative values to ensure sign handling
        let a_nd = Array2::from_shape_vec((2, 3), vec![-1.0f32, 2.0, -3.0, 4.0, -5.0, 6.0]).unwrap();
        let b_nd = Array2::from_shape_vec((3, 2), vec![1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0]).unwrap();

        let a = Tensor::from_ndarray(&a_nd);
        let b = Tensor::from_ndarray(&b_nd).try_transpose(0, 1).unwrap();
        let b = b.try_transpose(0, 1).unwrap(); // Back to [3, 2] but contiguous
        let mut c = a.matmul(&b).unwrap();
    c.realize_with(&config).unwrap();

        let expected = a_nd.dot(&b_nd);

        assert_matmul_close(&c.as_vec::<f32>().unwrap(), &expected, 1e-5);
    }
}

// ========== Shape Tests ==========

/// `dot` output shape per operand-rank combination. Values are validated by the
/// `*_validated` cases above; this table pins the lazy, pre-realize shape.
#[test_case(&[2, 2], &[2, 2], &[2, 2]; "2d square")]
#[test_case(&[2, 3], &[3, 4], &[2, 4]; "2d non-square")]
#[test_case(&[3], &[3], &[]; "1d dot product is scalar")]
#[test_case(&[3], &[3, 4], &[4]; "vector times matrix")]
#[test_case(&[2, 3], &[3], &[2]; "matrix times vector")]
#[test_case(&[2, 3, 4], &[2, 4, 5], &[2, 3, 5]; "batched")]
fn test_dot_output_shape(a: &[usize], b: &[usize], expected: &[usize]) {
    let a = Tensor::zeros(a, DType::Float32).unwrap();
    let b = Tensor::zeros(b, DType::Float32).unwrap();
    let shape = a.dot(&b).unwrap().shape().unwrap().iter().map(|d| d.as_const().unwrap()).collect::<Vec<_>>();
    assert_eq!(shape, expected);
}

/// A 1D weight is an elementwise multiply, not a matmul.
#[test_case(&[1, 3], &[2, 3], true, &[1, 2]; "with bias")]
#[test_case(&[1, 3], &[2, 3], false, &[1, 2]; "without bias")]
#[test_case(&[4, 3], &[2, 3], false, &[4, 2]; "batched")]
#[test_case(&[3], &[3], false, &[3]; "1d weight")]
fn test_linear_output_shape(input: &[usize], weight: &[usize], bias: bool, expected: &[usize]) {
    let input = Tensor::zeros(input, DType::Float32).unwrap();
    let weight = Tensor::zeros(weight, DType::Float32).unwrap();
    let result = match bias {
        true => {
            let out = weight.shape().unwrap()[0].as_const().unwrap();
            let bias = Tensor::zeros(&[out], DType::Float32).unwrap();
            input.linear().weight(&weight).bias(&bias).call().unwrap()
        }
        false => input.linear().weight(&weight).call().unwrap(),
    };
    let shape = result.shape().unwrap().iter().map(|d| d.as_const().unwrap()).collect::<Vec<_>>();
    assert_eq!(shape, expected);
}

// ========== Edge Cases ==========

#[test]
fn test_matmul_error_0d() {
    let scalar = Tensor::from_ndarray(&ndarray::Array0::<f32>::from_elem((), 1.0));
    let vector = Tensor::from_slice([1.0f32, 2.0, 3.0]);

    // 0D tensors not supported
    assert!(scalar.dot(&vector).is_err());
    assert!(vector.dot(&scalar).is_err());
}

#[test]
fn test_matmul_error_shape_mismatch() {
    // [2, 3] @ [4, 5] - inner dimensions don't match
    let a = Tensor::from_ndarray(&Array2::<f32>::ones((2, 3)));
    let b = Tensor::from_ndarray(&Array2::<f32>::ones((4, 5)));

    let result = a.dot(&b);
    assert!(result.is_err());
}

// ========== Dtype Tests ==========

#[test]
fn test_matmul_dtype_promotion() {
    let a = Tensor::from_ndarray(&array![[1i32, 2], [3, 4]]);
    let b = Tensor::from_ndarray(&array![[5.0f32, 6.0], [7.0, 8.0]]);

    let c = a.dot(&b).unwrap();
    // Result should be promoted to float32
    assert_eq!(c.uop().dtype(), DType::Float32);
}

#[test]
fn test_matmul_explicit_dtype() {
    let a = Tensor::from_ndarray(&array![[1.0f32, 2.0], [3.0, 4.0]]);
    let b = Tensor::from_ndarray(&array![[5.0f32, 6.0], [7.0, 8.0]]);

    // Use float64 accumulation
    let c = a.matmul_with().other(&b).dtype(DType::Float64).call().unwrap();
    assert_eq!(c.uop().dtype(), DType::Float64);
}

crate::codegen_tests! {
    /// A wider integer accumulator must not wrap the products: `127 * 2` is
    /// 254, not -2, on every backend (LLVM emits `mul i8` unless the operands
    /// are widened first; C hid the wrap behind integer promotion).
    fn test_matmul_int8_products_do_not_wrap(config) {
        let a = Tensor::from_ndarray(&array![[127i8, -64, 32, 0]]);
        let b = Tensor::from_ndarray(&array![[2i8], [-1], [0], [1]]);
        let mut c = a.matmul_with().other(&b).dtype(DType::Int32).call().unwrap();
        assert_eq!(c.uop().dtype(), DType::Int32);
        c.realize_with(&config).unwrap();
        assert_eq!(c.as_vec::<i32>().unwrap(), vec![318]);
    }

    /// Narrow integer arithmetic wraps at its own width on every backend before
    /// a widening cast sees it: `neg` is a product with a wrapped `-1`, and C
    /// evaluates `signed char` arithmetic at `int` width unless told otherwise.
    fn test_narrow_int_arithmetic_wraps_before_widening(config) {
        let realized = |t: Tensor| {
            let mut t = t.cast(DType::Int32).unwrap();
            t.realize_with(&config).unwrap();
            t.as_vec::<i32>().unwrap()
        };
        let u8s = Tensor::from_slice([1u8, 2, 200]);
        assert_eq!(realized(u8s.try_neg().unwrap()), vec![255, 254, 56]);
        assert_eq!(realized(u8s.try_sub(&Tensor::from_slice([2u8, 1, 100])).unwrap()), vec![255, 1, 100]);
        assert_eq!(realized(u8s.lshift(&Tensor::from_slice([4u8, 4, 4])).unwrap()), vec![16, 32, 128]);
        let i8s = Tensor::from_slice([100i8, -128, 127]);
        assert_eq!(realized(i8s.try_add(&i8s).unwrap()), vec![-56, 0, -2]);
    }

    /// The same widening reaches `conv`: an int8 1x1 convolution with an int32
    /// accumulator forms int32 products.
    fn test_conv_int8_products_do_not_wrap(config) {
        let x = Tensor::from_ndarray(&Array4::from_shape_fn((1, 4, 1, 1), |(_, c, _, _)| [127i8, -64, 32, 0][c]));
        let w = Tensor::from_ndarray(&Array4::from_shape_fn((1, 4, 1, 1), |(_, c, _, _)| [2i8, -1, 0, 1][c]));
        let mut y = x.conv2d().weight(&w).acc_dtype(DType::Int32).call().unwrap();
        y.realize_with(&config).unwrap();
        assert_eq!(y.as_vec::<i32>().unwrap(), vec![318]);
    }
}

crate::codegen_tests! {
    /// A tensor-core-shaped int8 GEMM (every axis a multiple of 16) with
    /// products that overflow int8 and int16: every backend must match the
    /// int32 reference, and an RDNA3 GPU must get there through the integer WMMA.
    fn test_matmul_int8_tensor_core_shapes_match_reference(config) {
        let a_nd = Array2::from_shape_fn((16, 16), |(m, k)| ((m * 16 + k) * 37 + 11) as i32 % 256 - 128);
        let b_nd = Array2::from_shape_fn((16, 16), |(k, n)| ((k * 16 + n) * 91 + 5) as i32 % 256 - 128);
        let a = Tensor::from_ndarray(&a_nd.mapv(|v| v as i8));
        let b = Tensor::from_ndarray(&b_nd.mapv(|v| v as i8));
        let build = || a.matmul_with().other(&b).dtype(DType::Int32).call().unwrap();

        let plan = build().prepare_with(&config).unwrap();
        let rdna3 = match a.device() {
            DeviceSpec::Amd { device_id } => svod_device::registry::resolve_amd_arch_from_topology(device_id)
                .is_ok_and(|arch| !arch.is_cdna() && !arch.is_rdna4()),
            _ => false,
        };
        if rdna3 {
            assert!(
                plan.kernels().any(|kernel| kernel.code.contains("llvm.amdgcn.wmma.i32.16x16x16.iu8")),
                "rdna3 must lower the widened int8 GEMM through the integer WMMA"
            );
        }

        let mut c = build();
        c.realize_with(&config).unwrap();
        assert_eq!(c.as_vec::<i32>().unwrap(), a_nd.dot(&b_nd).iter().copied().collect::<Vec<i32>>());
    }
}

// =========================================================================
// AMD tensor-core lowering. Compile-only: every test below stops at LLVM text
// (plus an ELF check when an amdgpu target is installed) and never opens a GPU.
// =========================================================================

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use svod_dtype::{AmdArch, DeviceSpec};
use svod_ir::{BinaryOp, ConstValue, Op, RendererDevice, UOp};
use svod_schedule::{OptimizerRenderer, optimize_kernel_with_config};

/// The single kernel AST the scheduler produces for `a·b` accumulating in `out`.
fn matmul_kernel_ast(a_shape: &[usize], b_shape: &[usize], in_dtype: DType, out: DType) -> Arc<UOp> {
    fused_matmul_kernel_ast(a_shape, b_shape, in_dtype, out, |a| a)
}

/// Like [`matmul_kernel_ast`], with `producer` applied to `a` before the matmul.
fn fused_matmul_kernel_ast(
    a_shape: &[usize],
    b_shape: &[usize],
    in_dtype: DType,
    out: DType,
    producer: impl Fn(Tensor) -> Tensor,
) -> Arc<UOp> {
    let a = producer(Tensor::empty(a_shape, in_dtype.clone()));
    let b = Tensor::empty(b_shape, in_dtype);
    let c = a.matmul_with().other(&b).dtype(out).call().expect("tensor matmul");
    let rangeified = svod_schedule::rangeify_with_map(UOp::sink(vec![c.uop().contiguous()])).expect("rangeify matmul");
    let (kernel_graph, _) = svod_schedule::try_get_kernel_graph(rangeified.sink).expect("split kernels");
    let pre = crate::schedule::create_pre_schedule(kernel_graph).expect("prepare tensor schedule");
    assert_eq!(pre.items.len(), 1, "matmul must remain one tensor kernel");
    pre.items[0].ast.clone()
}

/// `decompose_with` carries the renderer's dtype decompositor — FP8 emulation
/// needs it, native tensor-core paths must run without it.
fn amd_optimizer(arch: AmdArch, decompose_with: Option<&svod_codegen::llvm::LlvmTextRenderer>) -> OptimizerRenderer {
    OptimizerRenderer::for_amd_arch(arch).with_rewrite_capabilities(
        svod_ir::RendererOps::all(),
        decompose_with.and_then(svod_codegen::traits::Renderer::decompositor),
        None,
    )
}

fn pinned_tc_config(tc_index: usize, tc_opt: TcOptLevel) -> OptimizerConfig {
    let heuristics =
        HeuristicsConfig::builder().tc_opt(tc_opt).tc_select(TcSelect::Index(tc_index)).matvec_enabled(false).build();
    OptimizerConfig::builder().strategy(OptStrategy::Heuristic).heuristics(heuristics).build()
}

fn find_wmma<'a>(nodes: impl IntoIterator<Item = &'a Arc<UOp>>) -> &'a Arc<UOp> {
    nodes.into_iter().find(|u| matches!(u.op(), Op::Wmma(..))).expect("lowered matmul must emit a tensor-core op")
}

/// PROGRAM → LINEAR → LLVM text for `arch`, asserting the object assembles when
/// an amdgpu target is installed.
fn render_amd(optimized: Arc<UOp>, arch: AmdArch, name: &str) -> (Arc<UOp>, svod_codegen::RenderedKernel) {
    let program = svod_codegen::program_pipeline::program_from_sink(optimized, DeviceSpec::Amd { device_id: 0 })
        .expect("final target graph");
    let linearized = svod_codegen::program_pipeline::do_linearize(&program).expect("linearize");
    let linear = linearized.toposort().into_iter().find(|u| matches!(u.op(), Op::Linear(..))).expect("LINEAR stage");
    let renderer = svod_codegen::llvm::LlvmTextRenderer::amd(arch);
    let rendered = svod_codegen::traits::Renderer::render(&renderer, &linear, Some(name)).expect("render LLVM text");
    if svod_runtime::amd::compile::has_amdgpu_target() {
        let object =
            svod_runtime::amd::compile::compile_ir_to_amd_object(&rendered.code, arch).expect("assemble amdgpu object");
        assert_eq!(&object[..4], b"\x7fELF");
    }
    (linear, rendered)
}

/// RDNA4 follows Tinygrad 8c8b43de's tensor-core table: FP8 storage is emulated
/// as bytes, arithmetic is widened to f16, and the matmul uses the f16→f32
/// gfx12 WMMA.
#[test]
fn test_matmul_fp8_gfx1201_decomposes_to_f16_wmma_compile_only() {
    use svod_dtype::ScalarDType;

    for dtype in [DType::FP8E4M3, DType::FP8E5M2, DType::FP8E4M3FNUZ, DType::FP8E5M2FNUZ] {
        let ast = matmul_kernel_ast(&[16, 16], &[16, 16], dtype.clone(), DType::Float32);
        let renderer = svod_codegen::llvm::LlvmTextRenderer::amd(AmdArch::Gfx1201);
        let optimized = optimize_kernel_with_config(
            ast,
            &amd_optimizer(AmdArch::Gfx1201, Some(&renderer)),
            &pinned_tc_config(0, TcOptLevel::Strict),
        )
        .expect("gfx1201 FP8 decomposition and TC optimization");

        let nodes = optimized.toposort();
        let Op::Wmma(ops::Wmma { metadata: wmma, .. }) = find_wmma(&nodes).op() else { unreachable!() };
        assert_eq!((wmma.dtype_in.clone(), wmma.dtype_out.clone()), (DType::Float16, DType::Float32));
        assert_eq!((wmma.device, wmma.threads), (RendererDevice::AmdRdna4, 32));
        assert!(
            !nodes.iter().any(|u| u.dtype().base() == dtype.base()),
            "{dtype:?} arithmetic must be fully decomposed"
        );
        assert!(
            nodes
                .iter()
                .any(|u| matches!(u.op(), Op::Param(ops::Param { arg, .. }) if arg.dtype.base() == ScalarDType::UInt8)),
            "{dtype:?} storage must remain byte-addressed"
        );

        let (_, rendered) = render_amd(optimized, AmdArch::Gfx1201, "matmul_fp8_gfx1201");
        assert!(
            rendered.code.contains("llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16"),
            "{dtype:?} must select gfx12 f16 WMMA"
        );
        assert!(!rendered.code.contains("16x16x16.fp8"), "{dtype:?} must not claim native FP8 WMMA");
        assert!(!rendered.code.contains("16x16x16.bf8"), "{dtype:?} must not alias E5M2 to native BF8 WMMA");
    }
}

/// Native AMD tensor-core selection, compile-only: the pinned tensor core is
/// chosen, its WMMA metadata matches, and the rendered text names the intrinsic.
/// `int8 · int8 → int32` widens the operands before the product, so the matcher
/// must look through those casts; gfx950 keeps OCP FP8 operands for its scaled
/// K=128 MFMA and selects the format per dtype.
#[test_case(AmdArch::Gfx1151, DType::Int8, DType::Int32, (16, 16, 16), "llvm.amdgcn.wmma.i32.16x16x16.iu8", None; "gfx1151 int8 wmma")]
#[test_case(AmdArch::Gfx950, DType::FP8E4M3, DType::Float32, (16, 16, 128), "llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4", Some("i32 0, i32 0"); "gfx950 e4m3 scaled mfma")]
#[test_case(AmdArch::Gfx950, DType::FP8E5M2, DType::Float32, (16, 16, 128), "llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4", Some("i32 1, i32 1"); "gfx950 e5m2 scaled mfma")]
fn native_amd_tensor_core_compile_only(
    arch: AmdArch,
    in_dtype: DType,
    out_dtype: DType,
    dims: (usize, usize, usize),
    intrinsic: &str,
    format_selectors: Option<&str>,
) {
    let ast = matmul_kernel_ast(&[16, dims.2], &[dims.2, 16], in_dtype.clone(), out_dtype.clone());
    let optimizer = amd_optimizer(arch, None);
    let tc_index = optimizer
        .tensor_cores
        .iter()
        .position(|tc| tc.dtype_in == in_dtype && tc.dtype_out == out_dtype && tc.dims == dims)
        .expect("native tensor core");
    let optimized = optimize_kernel_with_config(ast, &optimizer, &pinned_tc_config(tc_index, TcOptLevel::Strict))
        .expect("native TC optimization");

    let nodes = optimized.toposort();
    let Op::Wmma(ops::Wmma { metadata, .. }) = find_wmma(&nodes).op() else { unreachable!() };
    assert_eq!((metadata.dims, metadata.dtype_in.clone(), metadata.dtype_out.clone()), (dims, in_dtype, out_dtype));

    let (_, rendered) = render_amd(optimized, arch, "native_tc");
    assert!(rendered.code.contains(intrinsic), "must select {intrinsic}");
    if let Some(format_selectors) = format_selectors {
        assert!(rendered.code.contains(format_selectors), "wrong scaled format selectors");
    }
}

/// A fused elementwise producer (`relu(a) @ b`) keeps the WMMA legal; the
/// hand-coded path must select it as tinygrad does.
#[test]
fn test_matmul_fused_producer_gfx1151_uses_wmma_compile_only() {
    let ast = fused_matmul_kernel_ast(&[16, 16], &[16, 16], DType::Float16, DType::Float32, |a| a.relu().unwrap());
    let optimizer = amd_optimizer(AmdArch::Gfx1151, None);
    let heuristics = HeuristicsConfig::builder().matvec_enabled(false).build();
    let config = OptimizerConfig::builder().strategy(OptStrategy::Heuristic).heuristics(heuristics).build();
    let optimized = optimize_kernel_with_config(ast, &optimizer, &config).expect("heuristic optimization");
    let nodes = optimized.toposort();
    let Op::Wmma(ops::Wmma { metadata, .. }) = find_wmma(&nodes).op() else { unreachable!() };
    assert_eq!((metadata.dtype_in.clone(), metadata.dtype_out.clone()), (DType::Float16, DType::Float32));
}

fn eval_lane(u: &Arc<UOp>, lane: i64) -> i64 {
    match u.op() {
        Op::Const(value) => value.0.try_int().expect("integer constant"),
        Op::Special(ops::Special { name, .. }) if name == "lidx0" => lane,
        Op::Cast(ops::Cast { src, .. }) => eval_lane(src, lane),
        Op::Binary(op, a, b) => {
            let (a, b) = (eval_lane(a, lane), eval_lane(b, lane));
            match op {
                BinaryOp::Add => a + b,
                BinaryOp::Mul => a * b,
                BinaryOp::And => a & b,
                BinaryOp::Shl => a << b,
                BinaryOp::Shr => a >> b,
                _ => panic!("unexpected lane-index operation {op:?}"),
            }
        }
        Op::Ternary(svod_ir::TernaryOp::MulAcc, a, b, c) => {
            eval_lane(a, lane) * eval_lane(b, lane) + eval_lane(c, lane)
        }
        _ => panic!("unexpected lane-index node {}", u.op().as_ref()),
    }
}

fn eval_gate(u: &Arc<UOp>, lane: i64) -> bool {
    match u.op() {
        Op::Binary(BinaryOp::Lt, lhs, rhs) => eval_lane(lhs, lane) < eval_lane(rhs, lane),
        _ => panic!("unexpected validity gate {}", u.op().as_ref()),
    }
}

fn memory_param_slot(index: &Arc<UOp>) -> Option<usize> {
    let buffer = match index.op() {
        Op::Index(ops::Index { buffer, .. }) => buffer,
        Op::Shrink(ops::Shrink { src, .. }) => src,
        _ => return None,
    };
    match buffer.op() {
        Op::Param(ops::Param { arg, .. }) => Some(arg.slot),
        _ => None,
    }
}

fn is_zero_stack(u: &Arc<UOp>, lanes: usize) -> bool {
    let is_zero = |u: &Arc<UOp>| matches!(u.op(), Op::Const(value) if value.0 == ConstValue::Float(0.0));
    matches!(u.op(), Op::Stack(ops::Stack { sources }) if sources.len() == lanes && sources.iter().all(is_zero))
}

fn is_lidx_lt(gate: &Arc<UOp>, bound: i64) -> bool {
    matches!(gate.op(), Op::Binary(BinaryOp::Lt, lhs, rhs)
        if rhs.vmin().try_int() == Some(bound)
            && rhs.vmax().try_int() == Some(bound)
            && lhs.toposort().iter().any(|u| matches!(u.op(), Op::Special(ops::Special { end, name })
                if name == "lidx0" && end.vmax().try_int() == Some(32))))
}

struct AFragment {
    load: Arc<UOp>,
    index: Arc<UOp>,
    offsets: Arc<UOp>,
    gate: Arc<UOp>,
    alt: Arc<UOp>,
}

/// The memory fragments of a padded WMMA kernel, collected from either the
/// optimized graph or the linearized op list — both must describe the same
/// bytes, so `assert_padded_5x16` runs against each.
struct Fragments {
    a: Vec<AFragment>,
    /// Address expression of each scalar B load.
    b: Vec<Arc<UOp>>,
    c: Vec<CStore>,
}

/// One C store: (address, gate in force, WMMA result lane).
type CStore = (Arc<UOp>, Option<Arc<UOp>>, Arc<UOp>);

impl Fragments {
    /// `store_gate` maps a store and its own gate to the gate actually in force:
    /// the store's own before linearization, the enclosing IF condition after.
    fn collect<'a>(
        ops: impl IntoIterator<Item = &'a Arc<UOp>>,
        wmma: &Arc<UOp>,
        store_gate: impl Fn(&Arc<UOp>, &Option<Arc<UOp>>) -> Option<Arc<UOp>>,
    ) -> Self {
        let mut fragments = Fragments { a: Vec::new(), b: Vec::new(), c: Vec::new() };
        for u in ops {
            if let Op::Load(ops::Load { index, alt, gate }) = u.op() {
                match memory_param_slot(index) {
                    Some(1) => {
                        let Op::Shrink(ops::Shrink { offsets, sizes, .. }) = index.op() else {
                            panic!("every A load must use a shaped SHRINK address: {}", u.tree())
                        };
                        let shape = u
                            .shape()
                            .unwrap()
                            .unwrap()
                            .iter()
                            .filter_map(|extent| extent.as_const())
                            .collect::<Vec<_>>();
                        assert_eq!(shape, [4]);
                        assert_eq!(eval_lane(sizes, 0), 4, "each pinned A access must contain four shaped lanes");
                        let alt = alt.as_ref().expect("padded A loads require a shaped zero alternative");
                        let gate = gate.as_ref().expect("padded A loads require a validity gate");
                        assert!(
                            is_zero_stack(alt, 4),
                            "every invalid padded A lane must contribute zero: {}",
                            alt.tree()
                        );
                        assert!(is_lidx_lt(gate, 5), "A loads must be guarded by row < M: {}", gate.tree());
                        fragments.a.push(AFragment {
                            load: u.clone(),
                            index: index.clone(),
                            offsets: offsets.clone(),
                            gate: gate.clone(),
                            alt: alt.clone(),
                        });
                    }
                    Some(2) => {
                        let Op::Index(ops::Index { indices, .. }) = index.op() else {
                            panic!("every B load must use a scalar INDEX address: {}", u.tree())
                        };
                        assert_eq!(indices.len(), 1, "B loads must have one address expression");
                        assert!(alt.is_none() && gate.is_none(), "unpadded B fragment loads must remain ungated");
                        assert!(u.shape().unwrap().unwrap().is_empty(), "B loads must remain scalar");
                        fragments.b.push(indices[0].clone());
                    }
                    slot => panic!("unexpected or malformed load (slot {slot:?}): {}", u.tree()),
                }
            }
            if let Op::Store(ops::Store { index, value, gate }) = u.op() {
                assert_eq!(memory_param_slot(index), Some(0), "only C stores are permitted: {}", u.tree());
                let Op::Index(ops::Index { indices, .. }) = index.op() else {
                    panic!("C stores must use scalar INDEX addresses")
                };
                assert_eq!(indices.len(), 1);
                let Op::Index(ops::Index { buffer, indices: value_indices }) = value.op() else {
                    panic!("C store value must index the WMMA result: {}", value.tree())
                };
                assert!(Arc::ptr_eq(buffer, wmma), "C store must consume this graph's WMMA accumulator");
                assert_eq!(value_indices.len(), 1);
                fragments.c.push((indices[0].clone(), store_gate(u, gate), value_indices[0].clone()));
            }
        }
        fragments
    }

    /// A 5x16 operand padded into a 16x16 tile: A[0..80] and C[0..80] are real,
    /// A[80..256] and C[80..96] are the padded tails the gates must disable.
    fn assert_padded_5x16(&self, stage: &str) {
        assert_eq!(self.a.len(), 4, "{stage}: padded WMMA A fragment must contain four shaped loads");
        let (mut loaded_a, mut padded_a) = (BTreeSet::new(), BTreeSet::new());
        for AFragment { offsets, gate, .. } in &self.a {
            for lane in 0..32 {
                for shaped_lane in 0..4 {
                    let index = eval_lane(offsets, lane) + shaped_lane;
                    assert!((0..256).contains(&index), "{stage}: raw padded A index escaped the 16x16 tile");
                    if eval_gate(gate, lane) {
                        loaded_a.insert(index);
                    } else {
                        padded_a.insert(index);
                    }
                }
            }
        }
        assert_eq!(loaded_a, (0..80).collect(), "{stage}: enabled A loads must cover exactly the real allocation");
        assert_eq!(padded_a, (80..256).collect(), "{stage}: zero-fill lanes must cover exactly the padded A tail");

        assert_eq!(self.b.len(), 16, "{stage}: WMMA B fragment must retain its 16 scalar column loads");
        let mut loaded_b = BTreeMap::new();
        for index in &self.b {
            for lane in 0..32 {
                *loaded_b.entry(eval_lane(index, lane)).or_insert(0usize) += 1;
            }
        }
        assert_eq!(loaded_b.keys().copied().collect::<BTreeSet<_>>(), (0..256).collect());
        assert!(
            loaded_b.values().all(|count| *count == 2),
            "{stage}: wave32 duplicates each B value across its two WMMA lane halves"
        );

        assert_eq!(self.c.len(), 3, "{stage}: WMMA must cover C with three per-lane output fragments");
        assert_eq!(self.c.iter().filter(|(_, gate, _)| gate.is_some()).count(), 1);
        let (mut stored_c, mut padded_c, mut result_lanes) = (BTreeSet::new(), BTreeSet::new(), BTreeSet::new());
        for (index, gate, value_index) in &self.c {
            result_lanes.insert(eval_lane(value_index, 0));
            if let Some(gate) = gate {
                assert!(is_lidx_lt(gate, 16), "{stage}: the partial C fragment must be guarded by lane<16");
            }
            for lane in 0..32 {
                let index = eval_lane(index, lane);
                if gate.as_ref().is_none_or(|gate| eval_gate(gate, lane)) {
                    assert!((0..80).contains(&index), "{stage}: enabled C store escaped the 5x16 allocation");
                    assert!(stored_c.insert(index), "{stage}: C index {index} must be stored exactly once");
                } else {
                    assert!((80..96).contains(&index), "{stage}: only the final C fragment tail may be disabled");
                    padded_c.insert(index);
                }
            }
        }
        assert_eq!(stored_c, (0..80).collect(), "{stage}: stores must cover exactly C[0..80]");
        assert_eq!(padded_c, (80..96).collect(), "{stage}: the partial C fragment must gate exactly C[80..96]");
        assert_eq!(result_lanes, [0, 1, 2].into_iter().collect(), "{stage}: C stores must consume WMMA lanes 0,1,2");
    }
}

/// Compile-only regression for tinygrad 8c8b43de's padded tensor-core path:
/// M=5 padded into gfx1151's 16-row WMMA tile. The address coverage is asserted
/// both before and after linearization, which must lift the partial C store's
/// gate into IF/ENDIF without moving a byte.
#[test]
fn test_matmul_m5_gfx1151_padded_wmma_compile_only() {
    let ast = matmul_kernel_ast(&[5, 16], &[16, 16], DType::Float16, DType::Float32);
    let optimized = optimize_kernel_with_config(
        ast,
        &amd_optimizer(AmdArch::Gfx1151, None),
        &pinned_tc_config(0, TcOptLevel::Padded),
    )
    .expect("gfx1151 padded tensor-core optimization");

    let nodes = optimized.toposort();
    assert!(
        nodes.iter().all(|u| !matches!(u.op(), Op::Reduce(..))),
        "pre-coalescing M=5 WMMA must not retain an operand-side range as a residual REDUCE",
    );
    let params = nodes
        .iter()
        .filter_map(|u| match u.op() {
            Op::Param(ops::Param { shape, arg }) => Some((arg.slot, u.dtype(), shape.vmax().try_int())),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        params,
        vec![(0, DType::Float32, Some(80)), (1, DType::Float16, Some(80)), (2, DType::Float16, Some(256))],
        "kernel ABI must be C[5x16], A[5x16], B[16x16]",
    );

    let wmma_node = find_wmma(&nodes);
    let Op::Wmma(ops::Wmma { metadata: wmma, c: accumulator, .. }) = wmma_node.op() else { unreachable!() };
    assert_eq!(wmma.dims, (16, 16, 16));
    assert_eq!((wmma.dtype_in.clone(), wmma.dtype_out.clone()), (DType::Float16, DType::Float32));
    assert_eq!((wmma.device, wmma.threads), (RendererDevice::AmdRdna3, 32));
    assert!(wmma.upcast_axes.is_none(), "expander must consume WMMA axis metadata");
    assert_eq!(
        accumulator.shape().unwrap().unwrap().last().and_then(|extent| extent.as_const()),
        Some(8),
        "gfx1151 must retain its eight-register hardware accumulator fragment",
    );
    Fragments::collect(&nodes, wmma_node, |_, gate| gate.clone()).assert_padded_5x16("optimized");

    let (linear, rendered) = render_amd(optimized, AmdArch::Gfx1151, "matmul_m5_gfx1151");
    let Op::Linear(ops::Linear { ops }) = linear.op() else { unreachable!() };

    let positions = |wanted: fn(&Op) -> bool| {
        ops.iter().enumerate().filter(|(_, u)| wanted(u.op())).map(|(position, _)| position).collect::<Vec<_>>()
    };
    let ifs = positions(|op| matches!(op, Op::If(..)));
    let endifs = positions(|op| matches!(op, Op::EndIf(..)));
    assert_eq!((ifs.len(), endifs.len()), (1, 1), "LINEAR must contain exactly the partial C store IF/ENDIF");
    let (if_position, if_node) = (ifs[0], &ops[ifs[0]]);
    let Op::If(ops::If { condition: if_condition, body: if_body }) = if_node.op() else { unreachable!() };
    let Op::EndIf(ops::EndIf { if_op: endif_owner }) = ops[endifs[0]].op() else { unreachable!() };
    assert!(is_lidx_lt(if_condition, 16), "partial C store IF must be lane<16");
    assert!(Arc::ptr_eq(endif_owner, if_node), "ENDIF must reference the partial-store IF by source identity");
    assert_eq!(if_body.len(), 1, "partial-store IF must own exactly one address dependency");
    assert_eq!(endifs[0], if_position + 2, "ENDIF must immediately follow the partial C store");
    let guarded_store = &ops[if_position + 1];
    let Op::Store(ops::Store { index: guarded_address, value: guarded_value, .. }) = guarded_store.op() else {
        panic!("the partial C store must immediately follow its IF")
    };
    assert!(Arc::ptr_eq(&if_body[0], guarded_address), "IF body must own the partial store address by identity");

    let linear_fragments = Fragments::collect(ops, find_wmma(ops), |store, gate| {
        assert!(gate.is_none(), "LINEAR cleanup must move C gates to IF/ENDIF");
        Arc::ptr_eq(store, guarded_store).then(|| if_condition.clone())
    });
    linear_fragments.assert_padded_5x16("linearized");

    assert!(rendered.code.contains("llvm.amdgcn.wmma.f32.16x16x16.f16"), "must select gfx11 f16 WMMA");
    assert!(!rendered.code.contains("mfma"), "gfx1151 must not select CDNA MFMA");
    let rendered_op =
        |id| rendered.operations.iter().find(|operation| operation.uop_id == id).expect("rendered UOp metadata");
    let result_of = |id| rendered_op(id).result.clone().expect("rendered result name");
    for AFragment { load, index, gate, alt, .. } in &linear_fragments.a {
        let (load_render, index_render) = (rendered_op(load.id), rendered_op(index.id));
        let (address_name, gate_name, alt_name) = (result_of(index.id), result_of(gate.id), result_of(alt.id));
        assert_eq!(index_render.lines.len(), 1, "each A address must render one GEP");
        assert!(
            index_render.lines[0].contains("getelementptr") && index_render.lines[0].contains("half"),
            "A address metadata must own its half-vector GEP: {:?}",
            index_render.lines
        );
        assert!(load_render.lines.iter().any(|line| line.contains("br i1") && line.contains(&gate_name)));
        assert!(load_render.lines.iter().any(|line| line.contains("load <4 x half>") && line.contains(&address_name)));
        assert!(load_render.lines.iter().any(|line| line.contains("phi <4 x half>") && line.contains(&alt_name)));
        assert_eq!(
            load_render.source_ids,
            vec![index.id, alt.id, gate.id],
            "rendered A load sources must preserve ownership"
        );
    }

    let (address_name, value_name) = (result_of(guarded_address.id), result_of(guarded_value.id));
    let if_render = rendered_op(if_node.id);
    assert_eq!(if_render.source_ids, vec![if_condition.id, guarded_address.id]);
    assert!(if_render.lines.iter().any(|line| line.contains("br i1") && line.contains(&result_of(if_condition.id))));
    assert!(rendered_op(guarded_address.id).lines.iter().any(|line| line.contains("getelementptr")));
    assert!(
        rendered_op(guarded_store.id)
            .lines
            .iter()
            .any(|line| { line.contains("store float") && line.contains(&address_name) && line.contains(&value_name) })
    );
    assert_eq!(rendered_op(guarded_store.id).source_ids, vec![guarded_address.id, guarded_value.id]);
    let endif_render = rendered_op(ops[endifs[0]].id);
    assert_eq!(endif_render.source_ids, vec![if_node.id]);
    assert!(endif_render.lines.iter().any(|line| line.contains("br label") && line.contains("if_end_")));
}

/// Hardware acceptance for the compile-only regression above. This test exits
/// before dispatch unless the selected device is exactly AMD:0 on gfx1151.
///
/// Run once after a clean boot:
/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tensor test_matmul_m5_gfx1151_padded_wmma_amd -- --ignored --nocapture --test-threads=1`.
#[test]
#[ignore = "requires AMD:0 with gfx1151; dispatches a real padded WMMA kernel"]
fn test_matmul_m5_gfx1151_padded_wmma_amd() {
    use svod_dtype::{AmdArch, DeviceSpec};

    setup_test_tracing();
    let device = DeviceSpec::Amd { device_id: 0 };
    assert_eq!(svod_device::registry::resolve_amd_arch_from_topology(0).expect("AMD:0 topology"), AmdArch::Gfx1151);

    let a_data = (0..5 * 16).map(|i| (i as f32 % 11.0 - 5.0) * 0.125).collect::<Vec<_>>();
    let b_data = (0..16 * 16).map(|i| (i as f32 % 13.0 - 6.0) * 0.0625).collect::<Vec<_>>();
    let expected = (0..5)
        .flat_map(|m| {
            let a_data = &a_data;
            let b_data = &b_data;
            (0..16).map(move |n| (0..16).map(|k| a_data[m * 16 + k] * b_data[k * 16 + n]).sum::<f32>())
        })
        .collect::<Vec<_>>();

    let a = Tensor::from_slice(&a_data).try_reshape([5, 16]).unwrap().cast(DType::Float16).unwrap();
    let b = Tensor::from_slice(&b_data).try_reshape([16, 16]).unwrap().cast(DType::Float16).unwrap();
    assert_eq!(a.device(), device, "set SVOD_DEVICE=AMD:0; refusing to dispatch elsewhere");
    assert_eq!(b.device(), device, "set SVOD_DEVICE=AMD:0; refusing to dispatch elsewhere");

    let heuristics = HeuristicsConfig::builder()
        .tc_opt(TcOptLevel::Padded)
        .tc_select(TcSelect::Index(0))
        .matvec_enabled(false)
        .build();
    let optimizer = OptimizerConfig::builder().strategy(OptStrategy::Heuristic).heuristics(heuristics).build();
    let config = prep_config(optimizer);
    let mut c = a.matmul_with().other(&b).dtype(DType::Float32).call().expect("tensor matmul");
    assert_eq!(c.device(), device);

    let plan = c.prepare_with(&config).expect("prepare padded WMMA on AMD:0");
    assert!(
        plan.kernels().any(|kernel| kernel.code.contains("llvm.amdgcn.wmma.f32.16x16x16.f16")),
        "prepared plan must contain gfx11 f16-to-f32 WMMA before execution"
    );

    plan.execute().expect("dispatch padded WMMA");
    let output = plan.output_buffer().expect("matmul output buffer");
    output.synchronize().expect("synchronize padded WMMA immediately after dispatch");
    let mut actual = vec![0.0f32; 5 * 16];
    output
        .copyout(unsafe {
            std::slice::from_raw_parts_mut(actual.as_mut_ptr().cast::<u8>(), actual.len() * std::mem::size_of::<f32>())
        })
        .expect("copy padded WMMA output to host");

    for (i, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 2e-2,
            "mismatch at output {i}: GPU={actual}, CPU={expected}, diff={}",
            (actual - expected).abs()
        );
    }
}

/// Ampere+ tensor cores end to end on `CUDA:0`: an `in_dtype` matmul
/// accumulating in `out_dtype` must lower through `mma.sync` (asserted on the
/// rendered IR of the prepared plan) and match the f32 reference. The inputs
/// are small integers, exact in f16, bf16 and int8, so only accumulation order
/// separates the float GPU sums from the reference; the int8 `m16n8k32`
/// (`satfinite.s8`, the quantized-linear path) accumulates in int32 and must
/// match exactly.
#[test_case(DType::Float16, DType::Float32, 0, "m16n8k16.row.col.f32.f32"; "f16")]
#[test_case(DType::BFloat16, DType::Float32, 1, "m16n8k16.row.col.bf16"; "bf16")]
#[test_case(DType::Int8, DType::Int32, 5, "m16n8k32.row.col.satfinite.s8"; "int8")]
fn test_matmul_cuda_tensor_core_matches_reference(in_dtype: DType, out_dtype: DType, tc_index: usize, intrinsic: &str) {
    setup_test_tracing();
    let Some(config) = PrepareConfig::for_cuda_if_available() else {
        eprintln!("skipped: default device is not a CUDA GPU");
        return;
    };
    let arch = crate::config::cuda_test_arch().expect("CUDA:0 is open");
    if !arch.has_bf16_mma() {
        eprintln!("skipped: {arch} has no m16n8k16 tensor cores");
        return;
    }
    let size = 64;
    let a_nd = Array2::from_shape_fn((size, size), |(m, k)| ((m * 7 + k) % 7) as f32 - 3.0);
    let b_nd = Array2::from_shape_fn((size, size), |(k, n)| ((k * 5 + n) % 5) as f32 - 2.0);
    let a = Tensor::from_ndarray(&a_nd).cast(in_dtype.clone()).unwrap();
    let b = Tensor::from_ndarray(&b_nd).cast(in_dtype).unwrap();
    let heuristics = HeuristicsConfig::builder().tc_select(TcSelect::Index(tc_index)).matvec_enabled(false).build();
    let optimizer = OptimizerConfig::builder().strategy(OptStrategy::Heuristic).heuristics(heuristics).build();
    let config = PrepareConfig { optimizer, ..config };

    let build = || a.matmul_with().other(&b).dtype(out_dtype.clone()).call().unwrap();
    let plan = build().prepare_with(&config).unwrap();
    let mma = format!("@llvm.nvvm.mma.{intrinsic}(");
    assert!(
        plan.kernels().any(|kernel| kernel.code.contains(&mma)),
        "the prepared plan must carry {mma}:\n{}",
        plan.kernels().map(|kernel| kernel.code.as_str()).collect::<Vec<_>>().join("\n")
    );
    let mut c = build();
    assert_eq!(c.uop().dtype(), out_dtype);
    c.realize_with(&config).unwrap();
    let (actual, tolerance) = if out_dtype.is_float() {
        (c.as_vec::<f32>().unwrap(), 1e-3)
    } else {
        (c.as_vec::<i32>().unwrap().into_iter().map(|value| value as f32).collect(), 0.5)
    };
    assert_matmul_close(&actual, &a_nd.dot(&b_nd), tolerance);
}

#[test]
fn test_beam_search_matmul() {
    // Test beam search optimization for matmul - reproduces float vector index bug
    let size = 512; // Original size that triggered the bug
    let a = Tensor::from_ndarray(
        &Array2::from_shape_vec((size, size), (0..size * size).map(|i| (i as f32) * 0.01).collect()).unwrap(),
    );
    let b = Tensor::from_ndarray(
        &Array2::from_shape_vec((size, size), (0..size * size).map(|i| (i as f32) * 0.01).collect()).unwrap(),
    );
    let mut c = a.matmul(&b).unwrap();

    // Use width=2 for reasonable test time. Disable disk cache to avoid stale results
    // from previous runs affecting correctness (beam cache is keyed by AST hash, but
    // the post-optimization pipeline may have changed).
    let beam_config = prep_config(
        OptimizerConfig::builder()
            .strategy(OptStrategy::Beam { width: 2 })
            .beam(BeamConfig::builder().disable_cache(true).build())
            .build(),
    );

    c.prepare_with(&beam_config).expect("beam search prepare should succeed");
}

/// All-ones operands: every output element is exactly `size`, so a mis-tiled
/// vectorization or upcast shows up as a wrong sum rather than a wrong shape.
#[test_case(64; "64x64")]
#[test_case(512; "512x512")]
fn test_matmul_ones_is_exactly_the_inner_dimension(size: usize) {
    let a = Tensor::from_ndarray(&Array2::<f32>::ones((size, size)));
    let b = Tensor::from_ndarray(&Array2::<f32>::ones((size, size)));
    let mut c = a.matmul(&b).unwrap();
    // from_env() so SVOD_OUTPUT_UPCAST and friends still steer the kernel.
    c.realize_with(&env_config()).unwrap();

    let result = c.as_vec::<f32>().unwrap();
    assert_eq!(result.len(), size * size);
    assert!(result.iter().all(|value| (value - size as f32).abs() < 0.01), "got {}", result[0]);
}

/// gfx942 (CDNA3) MFMA tensor-core matmul: end-to-end proof + numerical
/// validation, parameterized over the low-precision input dtype. A
/// `in_dtype·in_dtype` matmul accumulating in f32 matches a cdna3 tensor core,
/// so BEAM must lower it to `intrinsic`. Inputs are small integers (−3..3 /
/// −2..2, all exact in bf16/f16/fp8e4m3), so the MFMA result — accumulated
/// across the residual K-tile loop and fanned out across the M/N output tiles —
/// must equal the f32 reference exactly. Guards both the reduce-loop lowering
/// and the per-tile expansion. `tol` stays small because the inputs round-trip
/// losslessly and the accumulation is in f32.
fn validate_mfma_square(size: usize, in_dtype: DType, intrinsic: &str, tol: f32) {
    let a_data: Vec<f32> = (0..size * size).map(|x| ((x % 7) as f32) - 3.0).collect();
    let b_data: Vec<f32> = (0..size * size).map(|x| ((x % 5) as f32) - 2.0).collect();
    let a_nd = Array2::from_shape_vec((size, size), a_data).unwrap();
    let b_nd = Array2::from_shape_vec((size, size), b_data).unwrap();

    let beam = prep_config(
        OptimizerConfig::builder()
            .strategy(OptStrategy::Beam { width: 2 })
            .beam(BeamConfig::builder().disable_cache(true).build())
            .build(),
    );
    let build = || {
        let a = Tensor::from_ndarray(&a_nd).cast(in_dtype.clone()).unwrap();
        let b = Tensor::from_ndarray(&b_nd).cast(in_dtype.clone()).unwrap();
        a.matmul_with().other(&b).dtype(DType::Float32).call().unwrap()
    };

    // 1) The selected kernel must actually use the expected MFMA (not a fallback).
    let mut probe = build();
    let plan = probe.prepare_with(&beam).expect("prepare should succeed");
    let saw_mfma = plan.prepared_kernels().iter().any(|k| k.kernel.code.contains(intrinsic));
    assert!(saw_mfma, "BEAM did not select {intrinsic} for a {in_dtype:?} {size}x{size} matmul on gfx942");

    // 2) The MFMA result must match the f32 reference (exact for integer inputs).
    let mut c = build();
    c.realize_with(&beam).unwrap();
    let expected = a_nd.dot(&b_nd);
    assert_matmul_close(&c.as_vec::<f32>().unwrap(), &expected, tol);
}

/// Hardware-gated: `SVOD_DEVICE=AMD:0 cargo test -p svod-tensor test_matmul_bf16_mfma_validated -- --ignored --nocapture`.
#[test]
#[ignore]
fn test_matmul_bf16_mfma_validated() {
    validate_mfma_square(512, DType::BFloat16, "llvm.amdgcn.mfma.f32.16x16x16bf16.1k", 1.0);
}

/// gfx942 f16 16×16×16 MFMA (the `f16` plain form).
#[test]
#[ignore]
fn test_matmul_f16_mfma_validated() {
    validate_mfma_square(512, DType::Float16, "llvm.amdgcn.mfma.f32.16x16x16f16", 1.0);
}

/// gfx942 fp8 (e4m3) 16×16×32 MFMA. The cdna3 fp8 tensor core is K=32, so this
/// also exercises the K=32 reduce-tile lowering and i64 operand packing. Uses a
/// smaller matrix: the fp8 path compiles many BEAM candidates (each with the
/// fp8-conversion prelude), so 512² BEAM is ~8min; 128² keeps it tractable while
/// still tiling into the 16×16×32 core.
#[test]
#[ignore]
fn test_matmul_fp8_mfma_validated() {
    validate_mfma_square(128, DType::FP8E4M3, "llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8", 1.0);
}

// ========== Validated Matmul Tests (64x64 with env_config) ==========

#[test]
fn test_matmul_validated_64x64() {
    // 64x64 test with varied data
    const SIZE: usize = 64;
    let a_data: Vec<f32> = (0..SIZE * SIZE).map(|x| ((x as f32) * 0.01).sin()).collect();
    let b_data: Vec<f32> = (0..SIZE * SIZE).map(|x| ((x as f32) * 0.02).cos()).collect();

    let a_nd = Array2::from_shape_vec((SIZE, SIZE), a_data).unwrap();
    let b_nd = Array2::from_shape_vec((SIZE, SIZE), b_data).unwrap();
    let a = Tensor::from_ndarray(&a_nd);
    let b = Tensor::from_ndarray(&b_nd);

    let config = env_config();
    let mut c = a.matmul(&b).unwrap();
    c.realize_with(&config).unwrap();

    let expected = a_nd.dot(&b_nd);

    // Larger tolerance for accumulated floating point error
    assert_matmul_close(&c.as_vec::<f32>().unwrap(), &expected, 1e-1);
}

// ========== Large Dimension Validated Tests ==========

// Square matrix tests with increasing sizes
#[test_case(128, 0.5; "128x128")]
#[test_case(256, 1.0; "256x256")]
#[test_case(500, 1.5; "500x500 non-power-of-2")]
#[test_case(512, 2.0; "512x512")]
#[test_case(1024, 3.0; "1024x1024")]
fn test_matmul_validated_square(size: usize, tol: f32) {
    setup_test_tracing();
    run_validated_square_matmul(size, tol);
}

// Non-square matrix tests
#[test_case(512, 256, 384, 2.0; "512x256 @ 256x384")]
#[test_case(1024, 64, 128, 1.0; "1024x64 @ 64x128 tall-skinny")]
#[test_case(64, 512, 64, 1.5; "64x512 @ 512x64 wide")]
#[test_case(256, 1024, 256, 2.5; "256x1024 @ 1024x256 large-K")]
#[test_case(2, 384, 51865, 0.05; "2x384 @ 384x51865 whisper logits")]
#[test_case(1, 512, 10007, 0.05; "1x512 @ 512x10007 prime vocabulary")]
#[test_case(2, 64, 10007, 0.01; "2x64 @ 64x10007 prime vocabulary")]
fn test_matmul_validated_non_square(m: usize, k: usize, n: usize, tol: f32) {
    run_validated_matmul(m, k, n, tol);
}

/// Whisper's decoder logits `[rows, 384] x [384, 51865]`: the vocabulary axis
/// (5·11·23·41) has no standard block divisor, so the heuristic pads it to a
/// multiple of 32 and masks the tail. The launch must fill at least a warp per
/// block and the padded lanes must not leak into the result.
#[test_case(2, DType::Float32, 1e-3; "two rows f32")]
#[test_case(2, DType::Float16, 0.1; "two rows f16")]
#[test_case(1, DType::Float32, 1e-3; "one row f32 matvec")]
#[test_case(1, DType::Float16, 0.1; "one row f16 matvec")]
fn test_matmul_cuda_padded_vocabulary_axis(rows: usize, dtype: DType, tol: f32) {
    let Some(config) = PrepareConfig::for_cuda_if_available() else {
        eprintln!("skipped: default device is not a CUDA GPU");
        return;
    };
    let (k, n) = (384, 51865);
    // Zero-mean multiples of 1/32: exact in f16, so the f32 reference sees the same inputs.
    let a_nd = Array2::from_shape_fn((rows, k), |(m, i)| (((m * 17 + i) % 23) as f32 - 11.0) / 32.0);
    let b_nd = Array2::from_shape_fn((k, n), |(i, j)| (((i * 13 + j) % 29) as f32 - 14.0) / 32.0);
    let mut a = Tensor::from_ndarray(&a_nd).cast(dtype.clone()).unwrap();
    let mut b = Tensor::from_ndarray(&b_nd).cast(dtype).unwrap();
    a.realize().unwrap();
    b.realize().unwrap();

    let build = || a.matmul(&b).unwrap();
    let plan = build().prepare_with(&config).unwrap();
    let reduce_kernels: Vec<_> = plan.kernels().filter(|kernel| kernel.entry_point.starts_with("r_")).collect();
    assert_eq!(reduce_kernels.len(), 1, "one matmul kernel expected");
    let local = reduce_kernels[0].local_size.as_ref().expect("the matmul launches with a block size");
    let threads: i64 = local.iter().map(|dim| dim.vmax().try_int().expect("constant block dim")).product();
    assert!(threads >= 32, "{}: {threads} threads per block", reduce_kernels[0].entry_point);

    let mut c = build();
    c.realize_with(&config).unwrap();
    let mut c = c.cast(DType::Float32).unwrap();
    c.realize().unwrap();
    assert_matmul_close(&c.as_vec::<f32>().unwrap(), &a_nd.dot(&b_nd), tol);
}
