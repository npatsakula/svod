//! Vector scans over a logits row.
//!
//! Whisper reads one ~51k-entry logits row to the host per beam row per decode
//! step, and every consumer walks the whole row: the log-softmax normalizer, the
//! bounded top-k, the sampler. Scalar `f32::exp` is a libm call LLVM will not
//! vectorize, which made the normalizer alone 130 us per row -- ~0.8 s of a
//! 27 s transcription at beam 5.
//!
//! `fearless_simd` picks the ISA at run time (AVX-512 / AVX2 / SSE on x86, NEON
//! on aarch64) behind a safe API, so one implementation covers every target
//! svod builds for. The scalar versions stay: they are the short-input path, the
//! guard for the degenerate cases below, and the oracle the tests compare to.

use fearless_simd::{Level, Simd, dispatch, prelude::*};

use crate::simd::exp_nonpos;

/// Widest top-`k` the lane network handles before the scalar scan is cheaper.
const TOP_K_MAX: usize = 8;

/// Below this the dispatch preamble (~50 ns) costs more than the vectors save.
const SIMD_FLOOR: usize = 64;

#[inline(always)]
fn max_inner<S: Simd>(simd: S, arr: &[f32]) -> f32 {
    let lanes = S::f32s::LEN;
    let mut acc = S::f32s::splat(simd, f32::NEG_INFINITY);
    let mut pass = arr.chunks_exact(lanes);
    for chunk in &mut pass {
        acc = acc.max(S::f32s::from_slice(simd, chunk));
    }
    acc.as_slice().iter().chain(pass.remainder()).fold(f32::NEG_INFINITY, |a, &b| a.max(b))
}

/// `exp((x - max) * scale)` into `out`, returning the sum. `-inf` entries are
/// floored as in `logsumexp_inner`, so a suppressed token keeps a weight of
/// about `1.6e-38` instead of zero: unreachable by the sampler at f32 spacing.
#[inline(always)]
fn scaled_exp_inner<S: Simd>(simd: S, arr: &[f32], scale: f32, out: &mut [f32]) -> f32 {
    let lanes = S::f32s::LEN;
    let max_val = max_inner(simd, arr);
    if max_val == f32::NEG_INFINITY {
        out.fill(0.0);
        return 0.0;
    }
    let (mv, floor, k) = (S::f32s::splat(simd, max_val), S::f32s::splat(simd, -87.0), S::f32s::splat(simd, scale));
    let mut sum = S::f32s::splat(simd, 0.0);
    let mut src = arr.chunks_exact(lanes);
    let mut dst = out.chunks_exact_mut(lanes);
    for (chunk, slot) in (&mut src).zip(&mut dst) {
        let e = exp_nonpos::<S>(((S::f32s::from_slice(simd, chunk) - mv) * k).max(floor));
        slot.copy_from_slice(e.as_slice());
        sum += e;
    }
    let mut tail = 0.0;
    for (&value, slot) in src.remainder().iter().zip(dst.into_remainder()) {
        *slot = ((value - max_val) * scale).max(-87.0).exp();
        tail += *slot;
    }
    sum.as_slice().iter().sum::<f32>() + tail
}

#[inline(always)]
fn logsumexp_inner<S: Simd>(simd: S, arr: &[f32]) -> f32 {
    let lanes = S::f32s::LEN;
    let max_val = max_inner(simd, arr);
    if max_val == f32::NEG_INFINITY {
        return f32::NEG_INFINITY;
    }

    let mv = S::f32s::splat(simd, max_val);
    // exp(-87) is about 1.6e-38, next to the smallest normal f32, so clamping
    // there keeps `-inf` (suppressed tokens) out of the polynomial's range while
    // contributing nothing measurable to a sum whose largest term is exp(0) = 1.
    let floor = S::f32s::splat(simd, -87.0);
    let (mut s0, mut s1) = (S::f32s::splat(simd, 0.0), S::f32s::splat(simd, 0.0));
    let mut pass = arr.chunks_exact(lanes * 2);
    for chunk in &mut pass {
        let a = S::f32s::from_slice(simd, &chunk[..lanes]);
        let b = S::f32s::from_slice(simd, &chunk[lanes..]);
        s0 += exp_nonpos::<S>((a - mv).max(floor));
        s1 += exp_nonpos::<S>((b - mv).max(floor));
    }
    let tail: f32 = pass.remainder().iter().map(|&l| (l - max_val).exp()).sum();
    ((s0 + s1).as_slice().iter().sum::<f32>() + tail).ln() + max_val
}

#[inline(always)]
fn top_k_inner<S: Simd>(simd: S, logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let lanes = S::f32s::LEN;
    // Each lane keeps its own top-`k` through an insertion network. Their union
    // contains the global top-`k`: an element outside its own lane's top-`k`
    // already has `k` larger elements in that lane alone.
    let mut ladder = [S::f32s::splat(simd, f32::NEG_INFINITY); TOP_K_MAX];
    let mut pass = logits.chunks_exact(lanes);
    for chunk in &mut pass {
        let mut v = S::f32s::from_slice(simd, chunk);
        for rung in ladder.iter_mut().take(k) {
            let (hi, lo) = (rung.max(v), rung.min(v));
            *rung = hi;
            v = lo;
        }
    }
    let mut survivors: Vec<f32> = ladder[..k].iter().flat_map(|r| r.as_slice().iter().copied()).collect();
    survivors.extend_from_slice(pass.remainder());
    survivors.sort_unstable_by(|a, b| b.total_cmp(a));
    let kth = survivors[k - 1];

    // `>= kth` admits the top `k` plus any duplicates of the k-th value.
    let cut = S::f32s::splat(simd, kth);
    let mut top: Vec<(usize, f32)> = Vec::with_capacity(k + 4);
    let mut pass = logits.chunks_exact(lanes);
    let mut base = 0;
    for chunk in &mut pass {
        let mut hits = S::f32s::from_slice(simd, chunk).simd_ge(cut).to_bitmask();
        while hits != 0 {
            let lane = hits.trailing_zeros() as usize;
            hits &= hits - 1;
            top.push((base + lane, chunk[lane]));
        }
        base += lanes;
    }
    top.extend(pass.remainder().iter().copied().enumerate().filter(|&(_, v)| v >= kth).map(|(i, v)| (base + i, v)));
    top.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    top.truncate(k);
    top
}

/// `log(sum(exp(arr)))`, computed in one max pass and one exponentiated sum.
pub(crate) fn logsumexp(arr: &[f32]) -> f32 {
    if arr.len() < SIMD_FLOOR {
        return logsumexp_scalar(arr);
    }
    dispatch!(Level::new(), simd => logsumexp_inner(simd, arr))
}

/// `exp((x - max(x)) / temperature)` for every entry, and their sum: the
/// sampling distribution before normalization.
pub(crate) fn scaled_exp(arr: &[f32], temperature: f32) -> (Vec<f32>, f32) {
    let mut out = vec![0f32; arr.len()];
    let scale = temperature.recip();
    let sum = if arr.len() < SIMD_FLOOR {
        scaled_exp_scalar(arr, scale, &mut out)
    } else {
        dispatch!(Level::new(), simd => scaled_exp_inner(simd, arr, scale, &mut out))
    };
    (out, sum)
}

/// The reference the vector path is checked against, and the short-input path.
pub(crate) fn scaled_exp_scalar(arr: &[f32], scale: f32, out: &mut [f32]) -> f32 {
    let max_val = arr.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    if max_val == f32::NEG_INFINITY {
        out.fill(0.0);
        return 0.0;
    }
    arr.iter().zip(out).fold(0.0, |sum, (&value, slot)| {
        *slot = ((value - max_val) * scale).exp();
        sum + *slot
    })
}

/// The reference the vector path is checked against, and the short-input path.
pub(crate) fn logsumexp_scalar(arr: &[f32]) -> f32 {
    let max_val = arr.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    if max_val == f32::NEG_INFINITY {
        return f32::NEG_INFINITY;
    }
    (arr.iter().map(|&l| (l - max_val).exp()).sum::<f32>()).ln() + max_val
}

/// The `k` highest-scoring `(token, logprob)` pairs, ordered by descending
/// logprob and then by ascending token id.
///
/// Ranking on the raw logit is equivalent to ranking on the logprob: the
/// log-softmax normalizer is one constant per row, so it shifts every score
/// alike and cannot reorder them. That lets the scan hold `k` entries instead
/// of materializing — and sorting — a logprob for all ~51k tokens.
pub(crate) fn top_k_logprobs(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let k = k.min(logits.len());
    if k == 0 {
        return Vec::new();
    }
    let logsum = logsumexp(logits);
    if k > TOP_K_MAX || logits.len() < 4 * k || logits.len() < SIMD_FLOOR {
        return top_k_scalar(logits, k, logsum);
    }
    let picked = dispatch!(Level::new(), simd => top_k_inner(simd, logits, k));
    // A `-inf` k-th place means the row is mostly suppressed, so the `>= kth`
    // cut would sweep in every masked token; the scalar scan is exact there.
    if picked.len() < k || picked[k - 1].1 == f32::NEG_INFINITY {
        return top_k_scalar(logits, k, logsum);
    }
    picked.into_iter().map(|(token, logit)| (token, logit - logsum)).collect()
}

/// The bounded insertion scan: the guard path above, and the tests' oracle.
pub(crate) fn top_k_scalar(logits: &[f32], k: usize, logsum: f32) -> Vec<(usize, f32)> {
    // `a` outranks `b` on the higher logit, and on the lower token id in a tie.
    let outranks = |a: (usize, f32), b: (usize, f32)| a.1.total_cmp(&b.1).then_with(|| b.0.cmp(&a.0)).is_gt();
    let mut top: Vec<(usize, f32)> = Vec::with_capacity(k + 1);
    for candidate in logits.iter().copied().enumerate() {
        if top.len() == k && !outranks(candidate, top[k - 1]) {
            continue;
        }
        let at = top.partition_point(|&held| outranks(held, candidate));
        top.insert(at, candidate);
        top.truncate(k);
    }
    top.into_iter().map(|(token, logit)| (token, logit - logsum)).collect()
}
