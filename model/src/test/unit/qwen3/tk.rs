//! Tests for the Qwen3 tk fusion ([`crate::qwen3::tk`]): the GPU-free
//! applicability predicate, and the hardware-gated comparison of
//! `qkv_norm_rope` against the split / norm / rope graph it replaces.
//!
//! `SVOD_DEVICE=CUDA cargo test -p svod-model --lib qwen3::tk -- --ignored --nocapture`.

use proptest::prelude::*;
use svod_dtype::DType;
use svod_ir::SInt;
use svod_tensor::Tensor;
use svod_tensor::nn::{Layer, RmsNorm};
use svod_tk::NORM_SUPPORTED_ARCHS;
use test_case::test_case;

use crate::qwen3::tk::{Heads, qkv_norm_rope, select_qkv_cfg};

/// The wave width the tables below are written against (CUDA / RDNA).
const W32: usize = 32;

const EPS: f64 = 1e-6;

/// Whether the env-selected device is one the prologue is built for, with its
/// LLVM backend present — the self-skip gate for the `#[ignore]`d HW tests.
fn device_supported() -> bool {
    let spec = Tensor::empty(&[1], DType::Float32).device();
    svod_tk::target::check_target(&spec, NORM_SUPPORTED_ARCHS).is_ok()
}

// ── Applicability (GPU-free) ─────────────────────────────────────────────────

/// The prologue needs a wave count dividing both head counts, and a head dim
/// whose halves each divide into the wave.
#[test_case(16, 8, 128, Some(8); "qwen3 0.6b")]
#[test_case(32, 8, 128, Some(8); "a wider query grid")]
#[test_case(12, 6, 128, Some(2); "head counts that only share two")]
#[test_case(16, 8, 64, Some(8); "a 64 head dim still fits, with scalar accesses")]
#[test_case(16, 8, 32, None; "a 32 head dim halves below the wave")]
#[test_case(16, 8, 127, None; "an odd head dim declines")]
#[test_case(5, 5, 128, Some(1); "prime head counts fall to one wave")]
fn select_qkv_cfg_applicability(h: usize, h_kv: usize, dh: usize, warps: Option<usize>) {
    let heads = Heads { h, h_kv, dh };
    assert_eq!(select_qkv_cfg(heads, W32).map(|c| c.warps), warps, "select_qkv_cfg({h}, {h_kv}, {dh})");
}

proptest! {
    /// The prologue's wave count divides both head counts, so every slot's role
    /// is a build-time constant.
    #[test]
    fn prop_selected_qkv_cfg_splits_the_roles(h in 1usize..40, h_kv in 1usize..40, dh in 1usize..5) {
        let heads = Heads { h, h_kv, dh: dh * 64 };
        let Some(cfg) = select_qkv_cfg(heads, W32) else { return Ok(()) };
        prop_assert!(h.is_multiple_of(cfg.warps));
        prop_assert!(h_kv.is_multiple_of(cfg.warps));
        prop_assert!(cfg.warps * W32 <= 1024);
    }
}

// ── Hardware-gated correctness (CUDA sm_80+) ─────────────────────────────────

/// Realize, cast to f32, and read as a host `Vec<f32>`.
fn to_f32_vec(t: &Tensor) -> Vec<f32> {
    let f = t.cast(DType::Float32).contiguous();
    f.realize().expect("realize f32");
    f.as_vec::<f32>().expect("read f32")
}

/// A realized pseudo-random operand of `dtype`, deterministic in `seed` so the
/// kernel and the reference see identical roundings.
fn operand(shape: &[usize], dtype: DType, seed: f32) -> Tensor {
    let n: usize = shape.iter().product();
    let v: Vec<f32> = (0..n).map(|i| ((i as f32 + 1.0) * seed).sin() * 0.5).collect();
    let dims: Vec<isize> = shape.iter().map(|&d| d as isize).collect();
    let t = Tensor::from_slice(v).try_reshape(dims).expect("reshape").cast(dtype).contiguous();
    t.realize().expect("realize operand");
    t
}

/// Largest elementwise difference relative to the reference's own magnitude.
fn rel_err(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "length mismatch");
    let scale = want.iter().fold(0f32, |a, b| a.max(b.abs())).max(f32::MIN_POSITIVE);
    got.iter().zip(want).fold(0f32, |a, (g, w)| a.max((g - w).abs())) / scale
}

/// The kernel and the graph run the same ops in the same dtypes; only the head
/// reduce's summation order differs (a wave butterfly against the scheduler's
/// tree), so the result can move by the bf16 rounding of a different order: two
/// ulps, `2 · 2⁻⁸ ≈ 7.8e-3` of the output's magnitude.
const BF16_REL_TOL: f32 = 8e-3;

/// The graph the prologue replaces: split the fused row, view the heads, norm
/// q and k over `dh`, rotate, and leave v as the plain head view.
fn reference_prologue(
    qkv: &Tensor,
    wq: &Tensor,
    wk: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    heads: Heads,
) -> (Tensor, Tensor, Tensor) {
    let (b, l) = (qkv.dim(0).expect("b"), qkv.dim(1).expect("l"));
    let kv = heads.h_kv * heads.dh;
    let parts = qkv.split(&[heads.h * heads.dh, kv, kv], -1).expect("split");
    let view = |p: &Tensor, h: usize| {
        p.try_reshape([b.clone(), l.clone(), SInt::Const(h), SInt::Const(heads.dh)]).expect("head view")
    };
    let norm_rope = |p: &Tensor, h: usize, w: &Tensor| {
        RmsNorm::new(w.clone(), EPS)
            .forward(&view(p, h))
            .expect("reference head norm")
            .apply_rotary_emb(cos, sin, false)
            .expect("reference rope")
    };
    (norm_rope(&parts[0], heads.h, wq), norm_rope(&parts[1], heads.h_kv, wk), view(&parts[2], heads.h_kv).contiguous())
}

/// `qkv_norm_rope` against that graph, at the model's geometry and at a
/// single-row batch (where the rope position folds to the block index).
#[test_case(8, 512, Heads { h: 16, h_kv: 8, dh: 128 }; "qwen3 0.6b at 8x512")]
#[test_case(1, 128, Heads { h: 16, h_kv: 8, dh: 128 }; "a batch-1 prefill")]
#[test_case(2, 256, Heads { h: 8, h_kv: 8, dh: 128 }; "multi-head attention")]
#[test_case(4, 128, Heads { h: 12, h_kv: 6, dh: 256 }; "a two-wave block with a 256 head dim")]
#[ignore]
fn qkv_norm_rope_matches_the_graph_gpu(b: usize, l: usize, heads: Heads) {
    if !device_supported() {
        eprintln!("skip qkv_norm_rope_matches_the_graph_gpu: no CUDA sm_80+ device / toolchain");
        return;
    }
    let half = heads.dh / 2;
    let qkv = operand(&[b, l, heads.h * heads.dh + 2 * heads.h_kv * heads.dh], DType::BFloat16, 0.31);
    let wq = operand(&[heads.dh], DType::BFloat16, 0.17);
    let wk = operand(&[heads.dh], DType::BFloat16, 0.23);
    // The model's sequence-major rope cache: `[1, L, 1, dh/2]`.
    let cos = operand(&[1, l, 1, half], DType::BFloat16, 0.11);
    let sin = operand(&[1, l, 1, half], DType::BFloat16, 0.07);

    let (q, k, v) = qkv_norm_rope(&qkv, &wq, &wk, &cos, &sin, EPS, heads).expect("build").expect("the kernel applies");
    assert_eq!(q.dims().expect("dims"), [b, l, heads.h, heads.dh]);
    assert_eq!(k.dims().expect("dims"), [b, l, heads.h_kv, heads.dh]);
    assert_eq!(v.dims().expect("dims"), [b, l, heads.h_kv, heads.dh]);

    let (wq_r, wk_r, wv_r) = reference_prologue(&qkv, &wq, &wk, &cos, &sin, heads);
    // v is a copy of the fused row's tail: bit for bit.
    assert_eq!(to_f32_vec(&v), to_f32_vec(&wv_r), "v must be an exact copy");
    for (name, got, want) in [("q", &q, &wq_r), ("k", &k, &wk_r)] {
        let err = rel_err(&to_f32_vec(got), &to_f32_vec(want));
        println!("qkv_norm_rope {b}x{l} {name}: relative error {err:e}");
        assert!(err < BF16_REL_TOL, "{name} at {b}x{l}: relative error {err} exceeds {BF16_REL_TOL}");
    }
}

/// A geometry no block shape covers declines (`Ok(None)`) so the caller keeps
/// its graph; a malformed request is a structured `Err`. Both need a supported
/// device: on anything else `launch_custom` declines before it looks at the
/// request at all.
#[test]
#[ignore]
fn qkv_norm_rope_outcomes_gpu() {
    if !device_supported() {
        eprintln!("skip qkv_norm_rope_outcomes_gpu: no CUDA sm_80+ device / toolchain");
        return;
    }
    // A 32-wide head dim halves to 16, below the wave, so the prologue declines
    // rather than erroring.
    let heads = Heads { h: 16, h_kv: 8, dh: 32 };
    let small = operand(&[2, 128, 32 * 32], DType::BFloat16, 0.31);
    let w32 = operand(&[32], DType::BFloat16, 0.17);
    let (cos, sin) = (operand(&[128, 16], DType::BFloat16, 0.11), operand(&[128, 16], DType::BFloat16, 0.07));
    assert!(
        qkv_norm_rope(&small, &w32, &w32, &cos, &sin, EPS, heads).expect("builds").is_none(),
        "a head dim whose halves do not divide the wave must decline"
    );

    let heads = Heads { h: 16, h_kv: 8, dh: 128 };
    let qkv = operand(&[2, 128, 32 * 128], DType::BFloat16, 0.31);
    let w128 = operand(&[128], DType::BFloat16, 0.17);
    let (cos, sin) = (operand(&[128, 64], DType::BFloat16, 0.11), operand(&[128, 64], DType::BFloat16, 0.07));
    let e = qkv_norm_rope(&qkv, &w128, &w32, &cos, &sin, EPS, heads).expect_err("a [32] k weight is a caller bug");
    assert!(matches!(e, svod_tk::LaunchError::OperandShape { operand: "k_weight", .. }), "got {e:?}");

    let short = operand(&[64, 64], DType::BFloat16, 0.11);
    let e = qkv_norm_rope(&qkv, &w128, &w128, &short, &sin, EPS, heads).expect_err("a short cos table is a caller bug");
    assert!(matches!(e, svod_tk::LaunchError::OperandShape { operand: "cos", .. }), "got {e:?}");
}
