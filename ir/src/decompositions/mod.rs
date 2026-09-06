//! UOp decomposition framework.
//!
//! This module provides conditional decomposition of complex operations into
//! simpler primitives that all backends can handle. Backends that don't support
//! certain transcendental operations can use the pattern-based decompositor
//! to transform them into equivalent primitive operations.
//!
//! # Architecture
//!
//! 1. **Backend provides decomposition patterns** via `Renderer::decompositor()`
//! 2. **Decomposition pass** uses `graph_rewrite_bottom_up` to apply patterns
//! 3. **Each pattern** transforms one op into a subtree of primitive ops
//!
//! # Example
//!
//! ```ignore
//! // In tensor realization, before rendering:
//! if let Some(decompositor) = renderer.decompositor() {
//!     let ast = decompose_with(&kernel.ast, &decompositor);
//! }
//! let rendered = renderer.render(&ast)?;
//! ```

pub mod helpers;
pub mod transcendentals;

use std::sync::Arc;

use crate::pattern::TypedPatternMatcher;
use crate::rewrite::graph_rewrite_bottom_up;
use crate::uop::UOp;
use svod_macros::patterns;

use transcendentals::{xcos, xexp, xexp2, xlog, xlog2, xpow, xsin, xtan};

fn same_truncating_bucket(a: &Arc<UOp>, b: &Arc<UOp>) -> bool {
    let (Some(amin), Some(amax), Some(bmin), Some(bmax)) =
        (a.vmin().try_int(), a.vmax().try_int(), b.vmin().try_int(), b.vmax().try_int())
    else {
        return false;
    };
    (amin >= 0 && bmin > 0) || (amax <= 0 && bmax < 0)
}

fn floor_correction(a: &Arc<UOp>, b: &Arc<UOp>, remainder: &Arc<UOp>) -> Arc<UOp> {
    let zero = a.const_like(0);
    remainder.ne(&zero).and_(&a.lt(&zero).ne(&b.lt(&zero)))
}

/// Lower floor division/modulo to the C-style truncating operations exposed by
/// code-generation targets. This is Tinygrad's `floordiv_to_idiv` and
/// `floormod_to_mod` decomposition.
pub fn divmod_decomposition_patterns() -> TypedPatternMatcher<()> {
    patterns! {
        FloorDiv(a, b) => {
            let q = a.cdiv(b);
            if same_truncating_bucket(a, b) {
                Some(q)
            } else {
                let r = a.cmod(b);
                Some(q.sub(&floor_correction(a, b, &r).cast(a.dtype())))
            }
        },
        FloorMod(a, b) => {
            let r = a.cmod(b);
            if same_truncating_bucket(a, b) {
                Some(r)
            } else {
                let correction = floor_correction(a, b, &r);
                Some(r.add(&UOp::try_where(correction, b.clone(), b.const_like(0)).expect("floor modulo correction")))
            }
        },
    }
}

/// Pinned Tinygrad `get_transcendental_patterns`: decompose target
/// transcendentals only when the renderer lacks them, unless force mode is on.
pub fn get_transcendental_patterns(supported: &crate::RendererOps, force: bool) -> TypedPatternMatcher<()> {
    use crate::{DType, UnaryOp};
    use svod_dtype::ScalarDType::{Float16, Float32, Float64};

    fn approximation_dtype(dtype: &DType) -> bool {
        matches!(dtype.base(), Float16 | Float32 | Float64)
    }
    fn other_float(dtype: &DType) -> bool {
        dtype.is_float() && !approximation_dtype(dtype)
    }

    let mut pm = TypedPatternMatcher::default();
    if force || !supported.supports_unary(UnaryOp::Exp2) {
        pm = pm
            + patterns! {
                Exp2(src) if approximation_dtype(&src.dtype()) => xexp2(src),
                node @ Exp2(src) if other_float(&src.dtype())
                    => src.cast(DType::Float32).try_exp2().expect("float32 exp2").cast(node.dtype()),
            };
    }
    if force || !supported.supports_unary(UnaryOp::Log2) {
        pm = pm
            + patterns! {
                Log2(src) if approximation_dtype(&src.dtype()) => xlog2(src),
                node @ Log2(src) if other_float(&src.dtype())
                    => src.cast(DType::Float32).try_log2().expect("float32 log2").cast(node.dtype()),
            };
    }
    if force || !supported.supports_unary(UnaryOp::Sin) {
        pm = pm
            + patterns! {
                Sin(src) if approximation_dtype(&src.dtype()) => xsin(src),
                node @ Sin(src) if other_float(&src.dtype())
                    => src.cast(DType::Float32).try_sin().expect("float32 sin").cast(node.dtype()),
            };
    }
    if force || !supported.supports_unary(UnaryOp::Sqrt) {
        pm = pm + patterns! { Sqrt(src) => xpow(src, &src.const_like(0.5)), };
    }
    pm
}

/// f32 → bf16 round-to-nearest-even done in the integer domain, emitting no
/// `fptrunc`. amdgcn (LLVM 18) cannot select the vectorized bf16 truncstore that
/// `-O3` forms by fusing `fptrunc float to bfloat` + `store bfloat`; routing the
/// bits through integers and a final `bitcast i16 → bfloat` keeps `fptrunc` away
/// from the store. Port of Tinygrad's `cast_float_to_bf16` (`renderer/cstyle.py`),
/// bit-exact with the native conversion and vector-count-preserving.
fn cast_float_to_bf16(x: &Arc<UOp>) -> Arc<UOp> {
    use crate::DType;
    use svod_dtype::ScalarDType;

    let n = x.dtype().vcount();
    let vec = |s: ScalarDType| DType::Scalar(s).vec(n).expect("scalar dtype is vectorizable");

    // The XLA/Tinygrad round-half-to-even encoding. The two branches don't split
    // cleanly along finite/NaN lines (most NaN and Inf take the `rnd` branch); the
    // whole expression is opaque on purpose and is verified bit-exact, so the
    // bindings below are named after their arithmetic, not a semantic gloss.
    let u = x.bitcast(vec(ScalarDType::UInt32));
    // rnd = u + ((u >> 16) & 1) + 0x7fff.
    let lsb = u.try_shr_op(&u.const_like(16)).and_then(|s| s.try_and_op(&u.const_like(1))).expect("bf16: rne lsb");
    let rnd = u.try_add(&lsb).and_then(|r| r.try_add(&u.const_like(0x7fff))).expect("bf16: rne bias");
    // alt = (u & 0xffff) != 0 ? (u | 0x10000) : u.
    let low_nz =
        u.try_and_op(&u.const_like(0xffff)).and_then(|lo| lo.try_cmpne(&u.const_like(0))).expect("bf16: low16 != 0");
    let or_bit = u.try_or_op(&u.const_like(0x10000)).expect("bf16: or 0x10000");
    let alt = UOp::try_where(low_nz, or_bit, u.clone()).expect("bf16: alt select");
    // bits = ((0 - u) & 0x7f800000) != 0 ? rnd : alt.
    let exp_nz = u
        .neg()
        .try_and_op(&u.const_like(0x7f80_0000))
        .and_then(|e| e.try_cmpne(&u.const_like(0)))
        .expect("bf16: exponent test");
    let bits = UOp::try_where(exp_nz, rnd, alt).expect("bf16: rnd/alt select");
    // High 16 bits are the bf16 payload: truncate to u16, reinterpret as bf16.
    bits.try_shr_op(&bits.const_like(16))
        .expect("bf16: extract high half")
        .cast(vec(ScalarDType::UInt16))
        .bitcast(vec(ScalarDType::BFloat16))
}

/// Decomposition patterns for the AMD backend.
///
/// This supplements the renderer-conditioned target matcher for Morok-only
/// Exp/Log/Cos/Tan/Pow operations. Exp2/Log2/Sin/Sqrt and renderer-supported
/// Erf are deliberately
/// absent so there is exactly one approximation-selection path.
///
/// Every pattern is guarded to `f16`/`f32`/`f64` (tinygrad's
/// `TRANSCENDENTAL_DTYPES`): the polynomials are only defined for those, and
/// integer `Pow` (ONNX `test_pow_types_*`) / `bf16` / `fp8` must keep their
/// native lowering.
pub fn amd_decomposition_patterns() -> TypedPatternMatcher<()> {
    use crate::DType;
    fn transc(d: &DType) -> bool {
        use svod_dtype::ScalarDType::{Float16, Float32, Float64};
        matches!(d.base(), Float16 | Float32 | Float64)
    }
    patterns! {
        Exp(src)  if transc(&src.dtype()) => xexp(src),
        Log(src)  if transc(&src.dtype()) => xlog(src),
        Cos(src)  if transc(&src.dtype()) => xcos(src),
        Tan(src)  if transc(&src.dtype()) => xtan(src),

        // Binary pow: x^y = exp2(y * log2(x))
        Pow(base, exp) if transc(&base.dtype()) => xpow(base, exp),

        // bf16/fp8/int fall back to f32 then cast back (tinygrad's cast arm).
        // Int `Pow` would otherwise hit `@llvm.pow.f64`, which amdgcn can't
        // select; bf16/fp8 transcendentals have no native intrinsic either.
        Exp(src)  => xexp(&src.cast(DType::Float32)).cast(src.dtype()),
        Log(src)  => xlog(&src.cast(DType::Float32)).cast(src.dtype()),
        Cos(src)  => xcos(&src.cast(DType::Float32)).cast(src.dtype()),
        Tan(src)  => xtan(&src.cast(DType::Float32)).cast(src.dtype()),
        Pow(base, exp) => xpow(&base.cast(DType::Float32), &exp.cast(DType::Float32)).cast(base.dtype()),

        // f32 → bf16: integer round (see `cast_float_to_bf16`) instead of the
        // `fptrunc` whose vectorized truncstore amdgcn can't select. The result
        // is a BitCast, never a matching Cast, so the rewrite can't recurse.
        node @ Cast { src, .. }
            if node.dtype().base() == svod_dtype::ScalarDType::BFloat16
                && src.dtype().base() == svod_dtype::ScalarDType::Float32
            => cast_float_to_bf16(src),
    }
}

/// Decomposition patterns for the NVPTX backend.
///
/// The AMD set (transcendentals over native `exp2`/`log2`, integer-domain bf16
/// rounding) plus the f64 `Exp2`/`Log2` expansions: NVPTX lowers
/// `@llvm.exp2` for f16/f32 only and `Log2` rides the f32-only
/// `lg2.approx.f32`, so double precision takes the polynomial path.
pub fn nvptx_decomposition_patterns() -> TypedPatternMatcher<()> {
    fn f64(d: &crate::DType) -> bool {
        d.base() == svod_dtype::ScalarDType::Float64
    }
    amd_decomposition_patterns()
        + patterns! {
            Exp2(src) if f64(&src.dtype()) => xexp2(src),
            Log2(src) if f64(&src.dtype()) => xlog2(src),
        }
}

/// Apply decomposition to a UOp graph using the provided pattern matcher.
///
/// Uses `graph_rewrite_bottom_up` to traverse the graph and apply decomposition
/// patterns. This ensures children are processed before parents, which is
/// important for recursive decomposition (e.g., when a decomposition result
/// contains more operations that need decomposition).
///
/// # Arguments
///
/// * `root` - The root UOp of the graph to decompose
/// * `matcher` - The pattern matcher containing decomposition rules
///
/// # Returns
///
/// A new UOp graph with matched operations replaced by their decompositions.
///
/// # Example
///
/// ```ignore
/// let matcher = all_decomposition_patterns();
/// let decomposed = decompose_with(&kernel.ast, &matcher);
/// ```
pub fn decompose_with(root: &Arc<UOp>, matcher: &TypedPatternMatcher<()>) -> Arc<UOp> {
    graph_rewrite_bottom_up(matcher, root.clone(), &mut ())
}
