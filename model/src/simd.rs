//! Lane-wise transcendentals `fearless_simd` does not provide, shared by the
//! host-side scans (Whisper's logits row, Silero's LSTM recurrence).

use fearless_simd::{Simd, prelude::*};

/// `exp(x)` for `EXP_FLOOR <= x <= 0`, lane-wise.
///
/// Cephes' `expf` reduction: `x = n*ln2 + r` with `ln2` split so `n * C1` is
/// exact, a degree-5 minimax polynomial on `r`, and `2^n` assembled directly
/// into the exponent field. Within one ulp of `f32::exp` across the range --
/// this is the algorithm libm itself uses, not a fast approximation.
// The coefficients are Cephes' published `expf` constants; they are written at
// their documented precision so they stay recognizable against the reference,
// and every one of them rounds to the same f32 either way.
#[allow(clippy::excessive_precision)]
#[inline(always)]
pub(crate) fn exp_nonpos<S: Simd>(x: S::f32s) -> S::f32s {
    const C1: f32 = 0.693_359_375;
    const C2: f32 = -2.121_944_4e-4;

    let n = (x * std::f32::consts::LOG2_E).round_ties_even();
    let r = n.mul_add(-C2, n.mul_add(-C1, x));
    let p = r.mul_add(1.987_569_1e-4, 1.398_199_9e-3);
    let p = p.mul_add(r, 8.333_452e-3);
    let p = p.mul_add(r, 4.166_579_6e-2);
    let p = p.mul_add(r, 1.666_666_6e-1);
    let p = p.mul_add(r, 5.000_000_1e-1);
    let p = (p * r).mul_add(r, r) + 1.0;

    let bits: S::i32s = n.to_int();
    p * ((bits + 127) << 23u32).bitcast::<S::f32s>()
}

/// Below this [`exp_nonpos`]'s exponent assembly leaves the normal range; the
/// result there is below `f32::MIN_POSITIVE` and rounds to nothing that matters.
const EXP_FLOOR: f32 = -87.0;

/// `1 / (1 + exp(-x))`, lane-wise, through [`exp_nonpos`] of `-|x|` so it never
/// overflows.
#[inline(always)]
pub(crate) fn sigmoid<S: Simd>(simd: S, x: S::f32s) -> S::f32s {
    let e = exp_nonpos::<S>((-x.abs()).max(S::f32s::splat(simd, EXP_FLOOR)));
    let s = S::f32s::splat(simd, 1.0) / (e + 1.0);
    x.simd_ge(S::f32s::splat(simd, 0.0)).select(s, e * s)
}

/// `tanh(x)` lane-wise, as `(1 - e) / (1 + e)` with `e = exp(-2|x|)`.
#[inline(always)]
pub(crate) fn tanh<S: Simd>(simd: S, x: S::f32s) -> S::f32s {
    let e = exp_nonpos::<S>((x.abs() * -2.0).max(S::f32s::splat(simd, EXP_FLOOR)));
    ((-e + 1.0) / (e + 1.0)).copysign(x)
}
