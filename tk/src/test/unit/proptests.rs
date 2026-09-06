//! Property-based correctness tests for the tk kernels — [`matmul`] and
//! [`flash_attention_with`] vs independent references across proptest-generated
//! shapes, with a per-case [`manual_seed`] so a reported failure reproduces.
//!
//! All are `#[ignore]` + device-gated: they dispatch real AMD kernels. Run with
//! `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib proptests -- --ignored`. The
//! tolerances scale with the reduction dim but are intentionally loose — they catch
//! gross errors (wrong layout, off-by-tile, NaN), not last-ULP drift. The
//! structured-`Err` (malformed request) and `None`
//! (non-tiling length) paths are asserted directly rather than through proptest.

use proptest::prelude::*;
use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::rand::manual_seed;
use svod_tensor::testing::allclose_f32;

use crate::kernels::fa::{FA_SUPPORTED_ARCHS, FaOpts, flash_attention_with};
use crate::kernels::matmul::{MATMUL_SUPPORTED_ARCHS, matmul};

use super::device_supported;

/// A realized random tensor of `dtype` on the env-selected device.
fn randn_dt(shape: &[usize], dtype: DType) -> Tensor {
    let mut t = Tensor::randn(shape).expect("randn").cast(dtype).expect("cast");
    t.realize().expect("realize");
    t
}

/// Realize, cast to f32, and read as a host `Vec<f32>` for comparison.
fn to_f32_vec(t: Tensor) -> Vec<f32> {
    let mut f = t.cast(DType::Float32).expect("→f32");
    f.realize().expect("realize f32");
    f.as_vec::<f32>().expect("read f32")
}

// ── matmul ───────────────────────────────────────────────────────────────────

/// bf16→f32 tile matmul vs the f32 GEMM of the SAME bf16-rounded operands (isolating
/// the WMMA accumulation error from input rounding). `n` is a multiple of the gfx1151
/// block (64). `... proptests::prop_matmul_vs_reference_amd -- --ignored`.
#[test]
#[ignore]
fn prop_matmul_vs_reference_amd() {
    if !device_supported(MATMUL_SUPPORTED_ARCHS) {
        eprintln!("skip prop_matmul_vs_reference_amd: no supported AMD GPU / toolchain");
        return;
    }
    proptest!(ProptestConfig::with_cases(24), |(seed in any::<u64>(), nk in 1usize..=8)| {
        let n = nk * 64;
        manual_seed(seed);
        let a = randn_dt(&[n, n], DType::BFloat16);
        let b = randn_dt(&[n, n], DType::BFloat16);

        let Some(got_t) = matmul(&a, &b).expect("matmul build") else {
            return Ok(()); // device ineligible (the guard should prevent this)
        };
        let got = to_f32_vec(got_t);

        let af = a.cast(DType::Float32).expect("a→f32");
        let bf = b.cast(DType::Float32).expect("b→f32");
        let exp = to_f32_vec(af.matmul(&bf).expect("ref matmul"));

        // abs err ~ √n·bf16_ulp and |e| ~ √n, so the atol floor grows with √n while
        // rtol stays ~constant.
        let (atol, rtol) = (0.02 * (n as f32).sqrt(), 2e-2);
        let r = allclose_f32(&got, &exp, atol, rtol);
        prop_assert!(r.ok, "matmul n={n}: {}", r.message);
    });
}

/// A non-square request is a structural error (`NotSquare`), never `None`/`Some`.
#[test]
#[ignore]
fn matmul_non_square_is_err_amd() {
    if !device_supported(MATMUL_SUPPORTED_ARCHS) {
        eprintln!("skip matmul_non_square_is_err_amd: no supported AMD GPU / toolchain");
        return;
    }
    let a = randn_dt(&[64, 128], DType::BFloat16);
    let b = randn_dt(&[64, 64], DType::BFloat16);
    let result = matmul(&a, &b);
    assert!(matches!(result, Err(crate::launch::Error::NotSquare { .. })), "non-square must be a NotSquare error");
}

// ── flash-attention ──────────────────────────────────────────────────────────

/// FA vs an independent f32 SDPA over the same operands, across batch / heads /
/// length / head-dim / causal / dtype. `N` is a multiple of 128 (always tiles), so
/// every generated case exercises the kernel rather than the `None` fallback.
/// `... proptests::prop_fa_vs_sdpa_amd -- --ignored`.
#[test]
#[ignore]
fn prop_fa_vs_sdpa_amd() {
    if !device_supported(FA_SUPPORTED_ARCHS) {
        eprintln!("skip prop_fa_vs_sdpa_amd: no supported AMD GPU / toolchain");
        return;
    }
    proptest!(ProptestConfig::with_cases(24), |(
        seed in any::<u64>(),
        bsz in 1usize..=2,
        nblk in 1usize..=3,
        h in prop_oneof![Just(1usize), Just(2), Just(4), Just(8)],
        dblk in 1usize..=4,
        causal in any::<bool>(),
        use_f16 in any::<bool>(),
        masked in any::<bool>(),
        // Per-lane valid-key fraction of 128; the `128` boundary (no masking) is
        // weighted in. Lens are floored to ≥ 1 below: `key_lens == 0` is the
        // inactive-lane case `flash_attention_with` clamps to ≥ 1 (it can't match an
        // all-masked SDPA row), covered separately by `test_fa_key_lens_zero_is_finite_amd`.
        frac0 in prop_oneof![Just(1i64), Just(128), 1i64..=128],
        frac1 in prop_oneof![Just(1i64), Just(128), 1i64..=128],
    )| {
        let (n, d) = (nblk * 128, dblk * 16);
        let dtype = if use_f16 { DType::Float16 } else { DType::BFloat16 };
        manual_seed(seed);
        let q = randn_dt(&[bsz, n, h, d], dtype.clone());
        let k = randn_dt(&[bsz, n, h, d], dtype.clone());
        let v = randn_dt(&[bsz, n, h, d], dtype.clone());

        // Key-padding mask only on the non-causal path (matches production: the
        // GigaAM encoder masks key positions, never both). Per-lane lens in [1,n].
        let key_lens_vec: Option<Vec<i32>> = (masked && !causal).then(|| {
            let mk = |frac: i64| ((frac * n as i64) / 128).clamp(1, n as i64) as i32;
            (0..bsz).map(|i| if i == 0 { mk(frac0) } else { mk(frac1) }).collect()
        });
        let lens_t = key_lens_vec.as_ref().map(|kl| {
            let mut t = Tensor::from_slice(kl.as_slice());
            t.realize().expect("realize key_lens");
            t
        });

        let Some(got_t) = flash_attention_with(&q, &k, &v, FaOpts { causal, key_lens: lens_t.as_ref() })
            .expect("fa build")
        else {
            return Ok(()); // shapes chosen to tile; the guard prevents an ineligible device
        };
        let got = to_f32_vec(got_t);

        // Independent reference: f32 SDPA in [B,H,N,D] layout, permuted back to [B,N,H,D].
        // For the masked case, the same [B,1,1,N] `arange(N) >= lens[b]` key mask
        // (true = masked) the kernel applies — an all-masked lane is a zero row on
        // both sides (SDPA zero-fills, the kernel's denominator clamp yields 0).
        let perm = |t: &Tensor| t.cast(DType::Float32).expect("→f32").try_permute(&[0, 2, 1, 3]).expect("perm");
        let (qp, kp, vp) = (perm(&q), perm(&k), perm(&v));
        let sdpa = qp.scaled_dot_product_attention().key(&kp).value(&vp);
        let refb = if let Some(kl) = &key_lens_vec {
            let range = Tensor::arange(n as i64, None, None).expect("arange").try_reshape([1usize, 1, 1, n]).expect("reshape");
            let lref = Tensor::from_slice(kl.as_slice()).try_reshape([bsz, 1, 1, 1]).expect("reshape lens");
            let mask = range.try_ge(&lref).expect("ge mask");
            sdpa.is_causal(false).attn_mask(&mask).call().expect("sdpa masked")
        } else {
            sdpa.is_causal(causal).call().expect("sdpa")
        };
        let exp = to_f32_vec(refb.try_permute(&[0, 2, 1, 3]).expect("perm back"));

        // QKᵀ reduces over D; softmax-weighted PV ~O(1). Scale the atol floor with √D.
        let (atol, rtol) = (0.02 * (d as f32).sqrt(), 3e-2);
        let r = allclose_f32(&got, &exp, atol, rtol);
        prop_assert!(r.ok, "fa b={bsz} n={n} h={h} d={d} causal={causal} f16={use_f16} key_lens={key_lens_vec:?}: {}", r.message);
    });
}

/// A malformed FA request (head dim not a multiple of 16) is a structural error.
#[test]
#[ignore]
fn fa_bad_head_dim_is_err_amd() {
    if !device_supported(FA_SUPPORTED_ARCHS) {
        eprintln!("skip fa_bad_head_dim_is_err_amd: no supported AMD GPU / toolchain");
        return;
    }
    let q = randn_dt(&[1, 128, 4, 24], DType::BFloat16); // D=24 not a multiple of 16
    let result = flash_attention_with(&q, &q, &q, FaOpts::default());
    assert!(matches!(result, Err(crate::launch::Error::DimMultiple { .. })), "D%16!=0 must be a DimMultiple error");
}

/// A non-tiling sequence length is the fallback trigger: `Ok(None)`, not an error.
#[test]
#[ignore]
fn fa_non_tiling_n_is_none_amd() {
    if !device_supported(FA_SUPPORTED_ARCHS) {
        eprintln!("skip fa_non_tiling_n_is_none_amd: no supported AMD GPU / toolchain");
        return;
    }
    let q = randn_dt(&[1, 100, 4, 64], DType::BFloat16); // N=100 doesn't tile (not %128)
    let out = flash_attention_with(&q, &q, &q, FaOpts::default()).expect("no error");
    assert!(out.is_none(), "non-tiling N should yield None, got Some");
}
