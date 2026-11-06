//! `Tensor::rand` — uniform `[0, 1)` float draws backed by `BinaryOp::Threefry`.
//!
//! Pipeline:
//!
//! 1. `state::next_counter(device, num)` returns `(seed, counter)`, two `[2]`
//!    u32 BUFFER Tensors — the counter advances atomically per call. Both are
//!    kernel inputs rather than graph constants, so same-shape draws share one
//!    program and the output never const-folds.
//! 2. Derive a per-call `new_key` via one THREEFRY pass over `(c_low, c_high)`
//!    against the seed.
//! 3. Build `counts0 = arange(num/2)`, `counts1 = counts0 + num/2`, then run
//!    the bulk THREEFRY pass to produce `num` uint32 random words.
//! 4. `bits_to_rand`: shift-right by `(bitsize - mantissa_bits)`, OR with the
//!    bit pattern of `1.0`, bitcast back to the target float dtype, then
//!    subtract 1.0 to land in `[0, 1)`.
//!
//! Supports `Float16`, `BFloat16`, `Float32`, `Float64` on `DeviceSpec::Cpu`.
//! Multi-device support is a straightforward extension of the same pipeline.

use snafu::ResultExt;
use svod_dtype::{DType, DeviceSpec, ScalarDType};
use svod_ir::{ConstValue, UOp, shape::Shape, shape::to_vec_usize};

use crate::{ErrorKind, Result, Tensor, UOpSnafu};

use super::state;

impl Tensor {
    /// Uniform `[0, 1)` random tensor with float32 dtype on the active default
    /// device (CPU unless overridden via `set_default_device` or `SVOD_DEVICE`).
    ///
    /// THREEFRY-backed; deterministic for a fixed seed (set via
    /// [`crate::rand::manual_seed`]).
    #[track_caller]
    pub fn rand(shape: &[usize]) -> Result<Tensor> {
        origin_call!("rand");
        Self::rand_with(shape, DType::Float32, svod_dtype::default_device::default_device())
    }

    /// Variant of [`Tensor::rand`] with explicit dtype and device.
    ///
    /// Supported dtypes: `Float16`, `BFloat16`, `Float32`, `Float64`. Integer
    /// dtypes are not supported here — use `Tensor::randint` instead.
    #[track_caller]
    pub fn rand_with(shape: &[usize], dtype: DType, device: DeviceSpec) -> Result<Tensor> {
        origin_call!("rand");
        let scalar = dtype.scalar().ok_or_else(|| ErrorKind::SymbolicShapeUnsupported {
            operation: format!("Tensor::rand: non-scalar dtype {dtype:?}"),
        })?;
        if !scalar.is_float() {
            return Err(ErrorKind::SymbolicShapeUnsupported {
                operation: format!(
                    "Tensor::rand: float dtype required, got {scalar:?}; use Tensor::randint for integers"
                ),
            }
            .into());
        }
        let numel: usize = shape.iter().product();
        if numel == 0 {
            return Ok(Tensor::zeros(shape, dtype));
        }
        // Number of uint32 words needed to cover `numel * itemsize` bytes.
        let num_words = (numel * scalar.bytes()).div_ceil(4) as u64;
        let (seed, counter) = state::next_counter(&device, num_words);
        let bits = random_bits(&seed, &counter, num_words as usize)?;
        bits_to_rand(&bits, shape, dtype)
    }
}

/// Produce `num` uint32 random words from the `[lo, hi]` u32 `counter`
/// BUFFER. Single-chunk (no outer loop): `num` is bounded by `usize`, which is
/// more than enough for any realistic tensor shape.
fn random_bits(seed: &Tensor, counter: &Tensor, num: usize) -> Result<Tensor> {
    let u32_dt = DType::Scalar(ScalarDType::UInt32);

    let c_low = counter.try_shrink([(0usize, 1usize)])?;
    let c_high = counter.try_shrink([(1usize, 2usize)])?;

    // Step 1: per-call key derivation. THREEFRY(seed, [c_low, c_high]) → `[2]` u32.
    let new_key = threefry_random_bits(seed, &c_low, &c_high)?;

    // Step 2: build counts0 = arange(half), counts1 = counts0 + half.
    // `arange_with_dtype` only fast-paths on `ConstValue::Int`, so count in
    // i64 (a draw may pass i32::MAX words) and cast to u32.
    let half = num.div_ceil(2);
    let int64 = DType::Scalar(ScalarDType::Int64);
    let counts0 = Tensor::arange_with_dtype()
        .start(UOp::const_(int64.clone(), ConstValue::Int(0)))
        .stop(UOp::const_(int64.clone(), ConstValue::Int(half as i64)))
        .dtype(int64)
        .call()?
        .cast(u32_dt.clone());
    let half_t = Tensor::full(&[half], half as u32, u32_dt);
    let counts1 = counts0.try_add(&half_t)?;

    // Step 3: bulk THREEFRY pass. Returns `[2 * half]` u32; truncate to `num`.
    let bits_full = threefry_random_bits(&new_key, &counts0, &counts1)?;
    bits_full.try_shrink([(0usize, num)])
}

/// Pack `(counts1, counts0)` into u64, run THREEFRY against a broadcast u64
/// key, then split the result back into `(low_u32, high_u32)` and concat. The
/// result shape is `[2 * counts0.len()]` u32.
///
/// Pub(crate) so the test module can pin its output against JAX's
/// `jax.extend.random.threefry_2x32` reference values.
pub(crate) fn threefry_random_bits(key: &Tensor, counts0: &Tensor, counts1: &Tensor) -> Result<Tensor> {
    let u32_dt = DType::Scalar(ScalarDType::UInt32);
    let u64_dt = DType::Scalar(ScalarDType::UInt64);
    let counts_shape: Shape = counts0.shape()?;

    let shift_32 = Tensor::full(&to_vec_usize(&counts_shape).context(UOpSnafu)?, 32u32, u64_dt.clone());

    // x = (counts1 << 32) | counts0  (u64)
    let c0_u64 = counts0.cast(u64_dt.clone());
    let c1_u64 = counts1.cast(u64_dt.clone());
    let c1_shifted = c1_u64.try_shl(&shift_32)?;
    let x = c1_shifted.try_bitor(&c0_u64)?;

    // key_packed = (key[1] << 32) | key[0], then broadcast from [1] to counts_shape.
    let k_shift_32 = Tensor::full(&[1], 32u32, u64_dt.clone());
    let k0 = key.try_shrink([(0usize, 1usize)])?.cast(u64_dt.clone());
    let k1 = key.try_shrink([(1usize, 2usize)])?.cast(u64_dt.clone());
    let key_packed = k1.try_shl(&k_shift_32)?.try_bitor(&k0)?;
    let key_broadcast = key_packed.broadcast_to(&counts_shape)?;

    // THREEFRY at the UOp level (no Tensor-level wrapper for it yet — the
    // op is otherwise only used inside the rangeify decomp).
    let result_uop = UOp::threefry(x.uop().clone(), key_broadcast.uop().clone()).context(UOpSnafu)?;
    let result = Tensor::from_lazy(result_uop);

    // Split each u64 result into two u32s, concat: `[low_0, …, low_{N-1}, high_0, …, high_{N-1}]`.
    // Narrowing casts truncate, so no mask is needed; this is the exact inverse
    // of the pack above, which lets `symbolic_simple` cancel both against the
    // THREEFRY decomposition instead of emitting 64-bit ALU.
    let lo = result.cast(u32_dt.clone());
    let hi = result.try_shr(&shift_32)?.cast(u32_dt);

    Tensor::cat(&[&lo, &hi], 0)
}

/// Convert raw u32 random bits → float in `[0, 1)` via mantissa-fill.
///
/// `bits` has shape `[num_u32_words]`. Output has shape `shape` and dtype `dtype`.
///
/// For f32 this is straightforward (u32 → u32 = identity bitcast → shift+OR →
/// bitcast to f32). For f16/bf16 (2 bytes) we bitcast u32 → u16 which doubles
/// the element count; for f64 (8 bytes) we bitcast u32 → u64 which halves it.
/// The size-changing bitcast is handled by [`Tensor::bitcast`].
fn bits_to_rand(bits: &Tensor, shape: &[usize], dtype: DType) -> Result<Tensor> {
    let scalar = dtype.scalar().expect("scalar dtype validated by rand_with");
    let (_, nmant) = scalar.finfo().expect("rand float dtype validated by rand_with");
    let uint_dt = DType::Scalar(scalar.float_to_uint().expect("rand float dtype validated by rand_with"));
    let total_bits = (scalar.bytes() * 8) as u32;
    let shift = total_bits - nmant;

    // Bitcast u32 bits → matching uint of the target float dtype. May change
    // element count when `scalar.bytes() != 4`.
    let uint_bits = bits.bitcast(uint_dt.clone())?;
    let bits_shape_concrete = to_vec_usize(&uint_bits.shape()?).context(UOpSnafu)?;

    let shift_t = Tensor::full(&bits_shape_concrete, ConstValue::UInt(shift as u64), uint_dt.clone());
    let shifted = uint_bits.try_shr(&shift_t)?;

    let one_bits = ConstValue::UInt(one_bits_for(scalar));
    let one_bits_t = Tensor::full(&bits_shape_concrete, one_bits, uint_dt);
    let or_ed = shifted.try_bitor(&one_bits_t)?;

    let in_one_two = or_ed.bitcast(dtype.clone())?;
    let one_f = Tensor::full(&bits_shape_concrete, ConstValue::Float(1.0), dtype);
    let in_unit = in_one_two.try_sub(&one_f)?;

    // Bits-to-floats may produce more elements than needed (e.g. odd-numel f16
    // doubles to even count). Truncate to `numel` then reshape.
    let numel: usize = shape.iter().product();
    let trimmed = in_unit.try_shrink([(0usize, numel)])?;
    let isize_shape: Vec<isize> = shape.iter().map(|&d| d as isize).collect();
    trimmed.try_reshape(&isize_shape)
}

/// Bit pattern of `1.0` in the given float dtype, widened to `u64` for the
/// (uniform) `ConstValue::UInt` constructor. Values are well-known and
/// verifiable via `half::f16::from_f32(1.0).to_bits()` etc.
fn one_bits_for(s: ScalarDType) -> u64 {
    match s {
        ScalarDType::Float16 => 0x3C00,
        ScalarDType::BFloat16 => 0x3F80,
        ScalarDType::Float32 => 0x3F80_0000,
        ScalarDType::Float64 => 0x3FF0_0000_0000_0000,
        _ => panic!("one_bits_for: non-float dtype {s:?}"),
    }
}
