//! Tests for the NT linear-layer GEMM ([`crate::kernels::gemm`]): the GPU-free
//! applicability predicate ([`GemmPolicy::cfg`] / [`select_cfg`]) and the config
//! invariants every tile must satisfy, plus the hardware-gated comparison of
//! `gemm_nt` against the generic `Tensor::linear` on the transformer shapes.
//!
//! `SVOD_DEVICE={CUDA,AMD}:0 cargo test -p svod-tk --lib gemm -- --ignored --nocapture`.

use proptest::prelude::*;
use svod_dtype::{DType, GpuArch};
use svod_tensor::Tensor;
use test_case::test_case;

use crate::kernels::gemm::{
    CUDA_TILES, Epilogue, GEMM_NT_SUPPORTED_ARCHS, GemmCfg, GemmPolicy, NT_64X64, NT_128X64, NT_SPLIT_K, RDNA_TILES,
    gemm_nt, gemm_nt_with, gemm_nt_with_epilogue, select_cfg, swiglu_pair_width,
};

use super::device_supported;

/// Every tile the CUDA [`GemmPolicy`] can return. A shape is servable exactly
/// when one of these tiles it, so the predicate tests are written against the
/// same list the policy searches rather than a restatement of its divisibility
/// rules.
const TABLE: [GemmCfg; 2] = CUDA_TILES;

/// The accumulator fragment width of every arch the GEMM is built for
/// (`mma.sync`'s and gfx11 WMMA's 16×16).
const FRAG_COLS: usize = 16;

const SM86: GpuArch = GpuArch::Cuda(svod_dtype::CudaArch::from_compute_capability(8, 6));
/// An RDNA part: the table is keyed by the family, not the part.
const RDNA: GpuArch = GpuArch::Amd(svod_dtype::AmdArch::Gfx1151);

/// The static shared-memory a CUDA block may take without the opt-in dynamic
/// allocation (48 KiB).
const SHARED_MAX: usize = 48 << 10;

// ── Applicability + config invariants (GPU-free) ─────────────────────────────

/// The linear-layer shapes the kernel is tuned for, plus the odd ones a caller
/// might hand it: `M`/`N` multiples of 64 with `K` a multiple of the 32-wide strip
/// are served, everything else declines so the caller can pad.
#[test_case(4096, 1024, 6144, true; "gate_up")]
#[test_case(1024, 1024, 6144, true; "gate_up small M")]
#[test_case(1024, 1024, 4096, true; "fused qkv")]
#[test_case(1024, 1024, 2048, true; "q only")]
#[test_case(4096, 3072, 1024, true; "down")]
#[test_case(1024, 3072, 1024, true; "down small M")]
#[test_case(3072, 1280, 5120, true; "whisper ffn")]
#[test_case(128, 1024, 6144, true; "batch-1 prefill")]
#[test_case(128, 1024, 1024, true; "batch-1 narrow")]
#[test_case(256, 192, 128, true; "odd K")]
#[test_case(64, 32, 64, false; "K shorter than the pipeline")]
#[test_case(1024, 1024, 100, false; "N not a multiple of 64")]
#[test_case(100, 1024, 1024, false; "M not a multiple of 64")]
#[test_case(1024, 48, 1024, false; "K not a multiple of the strip")]
fn select_cfg_applicability(m: usize, k: usize, n: usize, served: bool) {
    assert_eq!(select_cfg(m, k, n).is_some(), served, "select_cfg({m}, {k}, {n})");
}

/// The narrow-N / short-M crossover: the default 128×64 tile once its grid covers
/// the device, the finer 64×64 tile when it would not.
#[test_case(4096, 6144, NT_128X64; "large grid keeps the default tile")]
#[test_case(1024, 1024, NT_128X64; "128 blocks still fill 28 SMs")]
#[test_case(128, 6144, NT_64X64; "batch-1 prefill takes the finer tile")]
#[test_case(128, 1024, NT_64X64; "a 16-block grid takes the finer tile")]
fn select_cfg_crossover(m: usize, n: usize, want: GemmCfg) {
    assert_eq!(select_cfg(m, 1024, n), Some(want), "select_cfg({m}, 1024, {n})");
}

/// A bigger device keeps the default tile on a grid a small one gives up on: the
/// crossover reads the SM count, it is not a hardcoded shape threshold.
#[test]
fn select_cfg_crossover_follows_the_sm_count() {
    let (m, k, n) = (128, 1024, 6144); // 96 blocks of the 128×64 tile
    let cuda = GemmPolicy::for_arch(SM86);
    assert_eq!(cuda.compute_units, 28);
    assert_eq!(cuda.cfg(m, k, n), Some(NT_64X64));
    assert_eq!(GemmPolicy { compute_units: 8, ..cuda }.cfg(m, k, n), Some(NT_128X64));
}

/// The RDNA table: the same shape rules (`M`/`N` by 64, `K` by the strip)
/// served by its tiles (`0` wide, `1` the deep-strip fine tile, `2` the short-K
/// fine tile), with the crossover against the family's 40 CUs; a family nobody
/// measured declines every shape.
#[test_case(4096, 1024, 6144, Some(0); "gate_up keeps the wide tile")]
#[test_case(1024, 1024, 6144, Some(0); "gate_up small M")]
#[test_case(4096, 3072, 1024, Some(0); "512 blocks take the wide tile")]
#[test_case(1024, 1024, 2048, Some(1); "256 blocks stay on the fine tile")]
#[test_case(128, 1024, 6144, Some(1); "batch-1 prefill takes the fine tile")]
#[test_case(64, 128, 192, Some(1); "short grid takes the deep strip")]
#[test_case(64, 64, 192, Some(2); "a K too short for the deep strip")]
#[test_case(128, 96, 6144, Some(2); "K of three strips")]
#[test_case(100, 1024, 1024, None; "M not a multiple of 64")]
#[test_case(1024, 48, 1024, None; "K not a multiple of the strip")]
fn rdna_policy_applicability(m: usize, k: usize, n: usize, want: Option<usize>) {
    let policy = GemmPolicy::for_arch(RDNA);
    assert_eq!(policy.compute_units, 40);
    assert_eq!(policy.cfg(m, k, n), want.map(|i| RDNA_TILES[i]), "rdna cfg({m}, {k}, {n})");
    let cdna = GemmPolicy::for_arch(GpuArch::Amd(svod_dtype::AmdArch::Gfx942));
    assert_eq!(cdna.cfg(m, k, n), None, "a family nobody measured declines");
    assert_eq!(cdna.swiglu_pair_width(), None);
}

/// A rank-1 operand is a structured `Err`, not a panic — the shape preconditions
/// resolve before any device dispatch, so this runs GPU-free.
#[test]
fn gemm_nt_low_rank_operand_is_operand_rank_err() {
    let ok = Tensor::randn(&[128, 128]).expect("randn");
    let v = Tensor::randn(&[128]).expect("randn");
    let e = gemm_nt(&v, &ok).expect_err("rank-1 x must error, not panic");
    assert!(matches!(e, crate::launch::Error::OperandRank { operand: "x", .. }), "got {e:?}");
    let e = gemm_nt(&ok, &v).expect_err("rank-1 w must error, not panic");
    assert!(matches!(e, crate::launch::Error::OperandRank { operand: "w", .. }), "got {e:?}");
    let x3 = Tensor::randn(&[2, 64, 128]).expect("randn");
    let e = gemm_nt(&ok, &x3).expect_err("rank-3 w must error, not panic");
    assert!(matches!(e, crate::launch::Error::OperandRank { operand: "w", .. }), "got {e:?}");
}

proptest! {
    /// `select_cfg` serves a shape exactly when some tile in the table tiles it,
    /// and the tile it returns is one that does.
    #[test]
    fn prop_select_cfg_agrees_with_the_table(m in 1usize..40, k in 1usize..40, n in 1usize..40) {
        let (m, k, n) = (m * 32, k * 32, n * 32); // dense around the tile boundaries
        let any = TABLE.iter().any(|cfg| cfg.tiles(m, k, n));
        let got = select_cfg(m, k, n);
        prop_assert_eq!(got.is_some(), any, "{}x{}x{}", m, k, n);
        if let Some(cfg) = got {
            prop_assert!(cfg.tiles(m, k, n));
        }
    }

    /// Whatever the shape, a returned tile is launchable: a legal CUDA block, a
    /// static shared-memory budget, and a wave grid its block edges divide into.
    #[test]
    fn prop_selected_cfg_is_launchable(m in 1usize..64, k in 1usize..64, n in 1usize..64) {
        let (m, k, n) = (m * 64, k * 32, n * 64);
        let Some(cfg) = select_cfg(m, k, n) else { return Ok(()) };
        prop_assert!(cfg.threads(32) <= 1024, "block {} threads", cfg.threads(32));
        prop_assert!(cfg.shared_bytes(2) <= SHARED_MAX, "{} shared bytes", cfg.shared_bytes(2));
        prop_assert_eq!(cfg.reg_m() * cfg.blocks_m(), cfg.block_m);
        prop_assert_eq!(cfg.reg_n() * cfg.blocks_n(), cfg.block_n);
        prop_assert_eq!(cfg.reg_m() % 16, 0);
        prop_assert_eq!(cfg.reg_n() % 16, 0);
        // The launch grid covers the whole C tile exactly once per K-slab.
        let grid = cfg.grid_dims(m, n);
        prop_assert_eq!((grid[0] * grid[1] * grid[2]) as usize, cfg.blocks(m, n));
    }
}

// ── Epilogue applicability (GPU-free) ────────────────────────────────────────

/// Every tile a policy can pick reads the same gate/up row arrangement, so a
/// weight permuted once at load is servable whatever `M` turns out to be — the
/// invariant [`GemmPolicy::swiglu_pair_width`] exists to state, on every arch
/// table.
#[test_case(SM86, &CUDA_TILES; "cuda")]
#[test_case(RDNA, &RDNA_TILES; "rdna")]
fn swiglu_pair_width_is_common_to_every_tile(arch: GpuArch, table: &[GemmCfg]) {
    let policy = GemmPolicy::for_arch(arch);
    assert_eq!(policy.tiles, table);
    let pair = policy.swiglu_pair_width().expect("the tiles agree on a pair width");
    assert_eq!(pair, table[0].reg_n() / 2);
    for cfg in table {
        assert_eq!(cfg.reg_n() / 2, pair, "{cfg:?} reads a different gate/up block width");
        assert!(cfg.carries(Epilogue::SwiGlu { pair }, Some(FRAG_COLS)));
    }
}

/// The device-level [`swiglu_pair_width`] is the resolved arch's, and `None`
/// where no arch resolves (the host), so a model on the CPU keeps its rows
/// plainly stacked.
#[test]
fn swiglu_pair_width_is_none_off_the_gpu() {
    assert_eq!(swiglu_pair_width(&svod_dtype::DeviceSpec::Cpu), None);
}

/// `Epilogue` is the shape contract too: SwiGLU halves the output columns, the
/// other two keep them, and `kind` drops the operand without changing either.
#[test]
fn epilogue_out_cols_and_kind() {
    let t = Tensor::randn(&[8, 8]).expect("randn");
    assert_eq!(Epilogue::<&Tensor>::Plain.out_cols(6144), 6144);
    assert_eq!(Epilogue::Add(&t).out_cols(6144), 6144);
    assert_eq!(Epilogue::<&Tensor>::SwiGlu { pair: 16 }.out_cols(6144), 3072);
    assert_eq!(Epilogue::Add(&t).kind(), Epilogue::Add(()));
    assert_eq!(Epilogue::<&Tensor>::SwiGlu { pair: 16 }.kind(), Epilogue::SwiGlu { pair: 16 });
}

/// A tile carries an epilogue only when its store can: split-K writes f32
/// partials, so neither fused form rides it; SwiGLU further needs `reg_n/2` to be
/// the caller's `pair` and a whole number of fragments.
#[test_case(NT_128X64, Epilogue::Plain, true; "plain rides any tile")]
#[test_case(NT_SPLIT_K, Epilogue::Plain, true; "plain rides split-K")]
#[test_case(NT_128X64, Epilogue::Add(()), true; "add rides the default tile")]
#[test_case(NT_64X64, Epilogue::Add(()), true; "add rides the finer tile")]
#[test_case(NT_SPLIT_K, Epilogue::Add(()), false; "add declines split-K")]
#[test_case(NT_128X64, Epilogue::SwiGlu { pair: 16 }, true; "swiglu at the tiles' own pair")]
#[test_case(NT_64X64, Epilogue::SwiGlu { pair: 16 }, true; "swiglu on the finer tile")]
#[test_case(NT_128X64, Epilogue::SwiGlu { pair: 32 }, false; "swiglu declines a wider pair")]
#[test_case(NT_128X64, Epilogue::SwiGlu { pair: 8 }, false; "swiglu declines a narrower pair")]
#[test_case(NT_SPLIT_K, Epilogue::SwiGlu { pair: 16 }, false; "swiglu declines split-K")]
fn cfg_carries_epilogue(cfg: GemmCfg, epi: Epilogue<()>, carries: bool) {
    assert_eq!(cfg.carries(epi, Some(FRAG_COLS)), carries, "{cfg:?} carrying {epi:?}");
}

/// A `pair` the fragment width does not divide would split the gate block inside
/// a fragment, where the two halves are not separable in registers.
#[test]
fn swiglu_declines_a_pair_off_the_fragment_grid() {
    let odd = GemmCfg { block_n: 48, warps_n: 2, ..NT_128X64 }; // reg_n = 24, pair = 12
    assert!(!odd.carries(Epilogue::SwiGlu { pair: 12 }, Some(FRAG_COLS)));
    assert!(!NT_128X64.carries(Epilogue::SwiGlu { pair: 16 }, None), "no matrix core, no SwiGLU");
}

// ── Hardware-gated correctness (CUDA sm_80+, RDNA) ───────────────────────────

/// Realize, cast to f32, and read as a host `Vec<f32>`.
fn to_f32_vec(t: &Tensor) -> Vec<f32> {
    let f = t.cast(DType::Float32).contiguous();
    f.realize().expect("realize f32");
    f.as_vec::<f32>().expect("read f32")
}

/// A realized pseudo-random `[rows, cols]` operand of `dtype`, deterministic in
/// `seed` so the kernel and the reference see identical roundings.
fn operand(rows: usize, cols: usize, dtype: DType, seed: f32) -> Tensor {
    let v: Vec<f32> = (0..rows * cols).map(|i| ((i as f32 + 1.0) * seed).sin() * 0.5).collect();
    let t = Tensor::from_slice(v)
        .try_reshape(vec![rows as isize, cols as isize])
        .expect("reshape")
        .cast(dtype)
        .contiguous();
    t.realize().expect("realize operand");
    t
}

/// Largest elementwise difference **relative to the reference's own magnitude** —
/// the scale a bf16 output's rounding is measured against.
fn rel_err(got: &[f32], want: &[f32]) -> f32 {
    let scale = want.iter().fold(0f32, |a, b| a.max(b.abs())).max(f32::MIN_POSITIVE);
    got.iter().zip(want).fold(0f32, |a, (g, w)| a.max((g - w).abs())) / scale
}

/// Both `gemm_nt` and the generic `linear` accumulate in f32 and round the result
/// to bf16, so they may differ by the rounding of a different summation order:
/// up to two bf16 ulps, `2 · 2⁻⁸ ≈ 7.8e-3` of the output's magnitude. The measured
/// error on these shapes is ≤ 3.1e-3 (one ulp); the bound leaves one ulp of slack.
const BF16_REL_TOL: f32 = 8e-3;

/// The SwiGLU epilogue's band. `silu(gate)·up` carries the GEMM's two-ulp
/// summation difference through a smooth activation (`|silu'| < 1.1`) and a
/// bf16 multiply, and the reference's own `[M, 2I]` intermediate is rounded to
/// bf16 exactly where the epilogue rounds its accumulators — so the error stays
/// the product's, not a new one. Measured ≤ 4.5e-3 on the shapes below.
const SWIGLU_REL_TOL: f32 = 1.2e-2;

/// `gemm_nt` against the generic `Tensor::linear` over the same bf16 operands, on
/// every tuned linear-layer shape plus the odd ones (a batch-1 M, an N that only
/// tiles by 64, a K that is neither a power of two nor a multiple of 64).
#[test_case(4096, 1024, 6144; "gate_up")]
#[test_case(1024, 1024, 6144; "gate_up small M")]
#[test_case(1024, 1024, 4096; "fused qkv")]
#[test_case(1024, 1024, 2048; "q only")]
#[test_case(4096, 3072, 1024; "down")]
#[test_case(1024, 3072, 1024; "down small M")]
#[test_case(3072, 1280, 5120; "whisper ffn")]
#[test_case(128, 1024, 1024; "batch-1 narrow")]
#[test_case(256, 192, 128; "odd K")]
#[test_case(64, 64, 64 * 3; "single-tile M")]
#[ignore]
fn gemm_nt_matches_linear_gpu(m: usize, k: usize, n: usize) {
    if !device_supported(GEMM_NT_SUPPORTED_ARCHS) {
        eprintln!("skip gemm_nt_matches_linear_gpu: no supported device / toolchain");
        return;
    }
    let (x, w) = (operand(m, k, DType::BFloat16, 0.31), operand(n, k, DType::BFloat16, 0.17));
    let y = gemm_nt(&x, &w).expect("gemm_nt build").expect("the kernel applies to a tiling shape");
    let want = to_f32_vec(&x.linear().weight(&w).call().expect("reference linear"));
    let err = rel_err(&to_f32_vec(&y), &want);
    println!("gemm_nt {m}x{k}x{n}: relative error {err:e}");
    assert!(err < BF16_REL_TOL, "{m}x{k}x{n}: relative error {err} exceeds the bf16 tolerance {BF16_REL_TOL}");
}

/// A `[B, L, K]` activation is `B·L` rows: the output is `[B, L, N]` and equals
/// the rank-2 kernel on the flattened rows.
#[test]
#[ignore]
fn gemm_nt_rank3_rows_match_rank2_gpu() {
    if !device_supported(GEMM_NT_SUPPORTED_ARCHS) {
        eprintln!("skip gemm_nt_rank3_rows_match_rank2_gpu: no supported device / toolchain");
        return;
    }
    let (b, l, k, n) = (4, 256, 192, 128);
    let (x, w) = (operand(b * l, k, DType::BFloat16, 0.31), operand(n, k, DType::BFloat16, 0.17));
    let x3 = x.try_reshape([b as isize, l as isize, k as isize]).expect("reshape");
    let y3 = gemm_nt(&x3, &w).expect("gemm_nt build").expect("the kernel applies");
    assert_eq!(y3.dims().expect("dims"), [b, l, n]);
    let y2 = gemm_nt(&x, &w).expect("gemm_nt build").expect("the kernel applies");
    assert_eq!(to_f32_vec(&y3), to_f32_vec(&y2));
}

/// f16 operands take the same path (both are K=16 matrix-core input dtypes) and
/// come back as f16.
#[test]
#[ignore]
fn gemm_nt_f16_matches_linear_gpu() {
    if !device_supported(GEMM_NT_SUPPORTED_ARCHS) {
        eprintln!("skip gemm_nt_f16_matches_linear_gpu: no supported device / toolchain");
        return;
    }
    let (m, k, n) = (256usize, 1024usize, 1024usize);
    let (x, w) = (operand(m, k, DType::Float16, 0.31), operand(n, k, DType::Float16, 0.17));
    let y = gemm_nt(&x, &w).expect("gemm_nt build").expect("the kernel applies");
    assert_eq!(y.uop().dtype(), DType::Float16, "the output keeps the operand dtype");
    let want = to_f32_vec(&x.linear().weight(&w).call().expect("reference linear"));
    let err = rel_err(&to_f32_vec(&y), &want);
    println!("gemm_nt f16 {m}x{k}x{n}: relative error {err:e}");
    // f16 carries 11 mantissa bits, so its two-ulp band is 8× tighter than bf16's.
    assert!(err < BF16_REL_TOL / 8.0, "f16: relative error {err}");
}

/// The split-K path ([`NT_SPLIT_K`]) — `split_k` f32 partial slabs plus the
/// reduction pass — computes the same result as the single-slab kernel. It is a
/// measured performance loss on this card (see [`NT_SPLIT_K`]) and so is never
/// selected, but the core supports it and must stay correct.
#[test]
#[ignore]
fn gemm_nt_split_k_matches_linear_gpu() {
    if !device_supported(GEMM_NT_SUPPORTED_ARCHS) {
        eprintln!("skip gemm_nt_split_k_matches_linear_gpu: no supported device / toolchain");
        return;
    }
    for (m, k, n) in [(128usize, 1024usize, 6144usize), (256, 512, 1024)] {
        let (x, w) = (operand(m, k, DType::BFloat16, 0.31), operand(n, k, DType::BFloat16, 0.17));
        let split = |_, _, _| Some(NT_SPLIT_K);
        let y = gemm_nt_with(&x, &w, split).expect("split-K build").expect("the split-K tile applies");
        let want = to_f32_vec(&x.linear().weight(&w).call().expect("reference linear"));
        let err = rel_err(&to_f32_vec(&y), &want);
        println!("gemm_nt split-K {m}x{k}x{n}: relative error {err:e}");
        assert!(err < BF16_REL_TOL, "split-K {m}x{k}x{n}: relative error {err}");
    }
}

/// `gemm_nt_with_epilogue(Add)` against the graph's `linear` + `try_add` over the
/// same bf16 operands, on the projections the epilogue is for (`o_proj`,
/// `down_proj` at the Qwen3-Embedding shapes) plus an odd one.
#[test_case(4096, 1024, 1024; "o_proj")]
#[test_case(4096, 3072, 1024; "down_proj")]
#[test_case(128, 1024, 1024; "batch-1 o_proj")]
#[test_case(256, 192, 128; "odd K")]
#[ignore]
fn gemm_nt_add_matches_graph_gpu(m: usize, k: usize, n: usize) {
    if !device_supported(GEMM_NT_SUPPORTED_ARCHS) {
        eprintln!("skip gemm_nt_add_matches_graph_gpu: no supported device / toolchain");
        return;
    }
    let (x, w) = (operand(m, k, DType::BFloat16, 0.31), operand(n, k, DType::BFloat16, 0.17));
    let res = operand(m, n, DType::BFloat16, 0.53);
    let y = gemm_nt_with_epilogue(&x, &w, Epilogue::Add(&res)).expect("build").expect("the kernel applies");
    assert_eq!(y.dims().expect("dims"), [m, n]);
    let want = x.linear().weight(&w).call().expect("reference linear").try_add(&res).expect("reference add");
    let err = rel_err(&to_f32_vec(&y), &to_f32_vec(&want));
    println!("gemm_nt+add {m}x{k}x{n}: relative error {err:e}");
    assert!(err < BF16_REL_TOL, "{m}x{k}x{n}: relative error {err} exceeds {BF16_REL_TOL}");
}

/// The `[2I, K]` gate/up weight rearranged into the alternating `pair`-row blocks
/// [`Epilogue::SwiGlu`] reads: `[g0.., u0.., g1.., u1.., …]`, so a wave's N tile
/// holds a gate block beside its matching up block.
fn pair_rows(w: &Tensor, pair: usize) -> Tensor {
    let d = w.dims().expect("dims");
    let (blocks, k) = (d[0] / (2 * pair), d[1]);
    let dim = |v: [usize; 4]| v.map(|d| d as isize).to_vec();
    let t = w
        .try_reshape(dim([2, blocks, pair, k]))
        .expect("split the halves")
        .try_permute(&[1, 0, 2, 3])
        .expect("interleave the blocks")
        .contiguous()
        .try_reshape(vec![d[0] as isize, k as isize])
        .expect("flatten");
    t.realize().expect("realize the paired weight");
    t
}

/// `gemm_nt_with_epilogue(SwiGlu)` against the graph it replaces — the fused
/// gate/up GEMM, the split, `silu` and the multiply — on the Qwen3-Embedding MLP
/// shapes plus an odd one. The reference reads the **un-permuted** weight, so the
/// test also pins the row arrangement and the output column mapping.
#[test_case(4096, 1024, 6144; "gate_up")]
#[test_case(1024, 1024, 6144; "gate_up small M")]
#[test_case(128, 1024, 6144; "batch-1 prefill")]
#[test_case(256, 192, 128; "odd K")]
#[ignore]
fn gemm_nt_swiglu_matches_graph_gpu(m: usize, k: usize, n: usize) {
    if !device_supported(GEMM_NT_SUPPORTED_ARCHS) {
        eprintln!("skip gemm_nt_swiglu_matches_graph_gpu: no supported device / toolchain");
        return;
    }
    let (x, w) = (operand(m, k, DType::BFloat16, 0.31), operand(n, k, DType::BFloat16, 0.17));
    let pair = swiglu_pair_width(&x.device()).expect("a common pair width");
    let y = gemm_nt_with_epilogue(&x, &pair_rows(&w, pair), Epilogue::SwiGlu { pair })
        .expect("build")
        .expect("the kernel applies");
    assert_eq!(y.dims().expect("dims"), [m, n / 2], "SwiGLU writes half the columns");

    let gate_up = x.linear().weight(&w).call().expect("reference linear");
    let halves = gate_up.split(&[n / 2, n / 2], -1).expect("split");
    let want = halves[0].silu().expect("silu").try_mul(&halves[1]).expect("gate·up");
    let err = rel_err(&to_f32_vec(&y), &to_f32_vec(&want));
    println!("gemm_nt+swiglu {m}x{k}x{n}: relative error {err:e}");
    assert!(err < SWIGLU_REL_TOL, "{m}x{k}x{n}: relative error {err} exceeds {SWIGLU_REL_TOL}");
}

/// The epilogues decline (`Ok(None)`) where the tile cannot carry them and error
/// on a malformed operand, the same split [`gemm_nt`] draws.
#[test]
#[ignore]
fn gemm_nt_epilogue_outcomes_gpu() {
    if !device_supported(GEMM_NT_SUPPORTED_ARCHS) {
        eprintln!("skip gemm_nt_epilogue_outcomes_gpu: no supported device / toolchain");
        return;
    }
    let (m, k, n) = (128usize, 1024usize, 1024usize);
    let (x, w) = (operand(m, k, DType::BFloat16, 0.31), operand(n, k, DType::BFloat16, 0.17));

    let wrong = operand(m, n / 2, DType::BFloat16, 0.53);
    let e = gemm_nt_with_epilogue(&x, &w, Epilogue::Add(&wrong)).expect_err("a mis-shaped residual is a caller bug");
    assert!(matches!(e, crate::launch::Error::OperandShape { operand: "residual", .. }), "got {e:?}");

    let f32res = operand(m, n, DType::Float32, 0.53);
    let e = gemm_nt_with_epilogue(&x, &w, Epilogue::Add(&f32res)).expect_err("an f32 residual is a caller bug");
    assert!(matches!(e, crate::launch::Error::Dtype { kernel: "gemm-nt", .. }), "got {e:?}");

    let pair = swiglu_pair_width(&x.device()).expect("a common pair width");
    let e = gemm_nt_with_epilogue(&x, &w, Epilogue::SwiGlu { pair: 0 }).expect_err("a zero pair is a caller bug");
    assert!(matches!(e, crate::launch::Error::DimMultiple { kernel: "gemm-nt", .. }), "got {e:?}");

    // A pair no tile reads is a fallback trigger, not an error.
    let off = gemm_nt_with_epilogue(&x, &w, Epilogue::SwiGlu { pair: 2 * pair }).expect("builds");
    assert!(off.is_none(), "a pair the tiles do not read must decline");

    // Split-K writes f32 partials, so no fused epilogue rides it.
    let ragged = operand(100, k, DType::BFloat16, 0.31);
    let res = operand(100, n, DType::BFloat16, 0.53);
    assert!(gemm_nt_with_epilogue(&ragged, &w, Epilogue::Add(&res)).expect("builds").is_none(), "M % 64 != 0");
}

/// A shape no tile covers declines (`Ok(None)`) so the caller can pad or fall
/// back; a malformed request (a wrong dtype, a K the two operands disagree on) is
/// a structured `Err`. Both need a supported device: on anything else
/// `launch_custom` declines before it looks at the request at all.
#[test]
#[ignore]
fn gemm_nt_outcomes_gpu() {
    if !device_supported(GEMM_NT_SUPPORTED_ARCHS) {
        eprintln!("skip gemm_nt_outcomes_gpu: no supported device / toolchain");
        return;
    }
    let ragged = operand(100, 1024, DType::BFloat16, 0.31);
    let w = operand(1024, 1024, DType::BFloat16, 0.17);
    assert!(gemm_nt(&ragged, &w).expect("ragged M builds").is_none(), "M % 64 != 0 must decline");

    let x = operand(128, 1024, DType::BFloat16, 0.31);
    let f32x = operand(128, 1024, DType::Float32, 0.31);
    let e = gemm_nt(&f32x, &w).expect_err("an f32 operand is a caller bug");
    assert!(matches!(e, crate::launch::Error::Dtype { kernel: "gemm-nt", .. }), "got {e:?}");

    let short = operand(1024, 512, DType::BFloat16, 0.17);
    let e = gemm_nt(&x, &short).expect_err("a K mismatch is a caller bug");
    assert!(matches!(e, crate::launch::Error::OperandShape { operand: "w", .. }), "got {e:?}");
}

// ── Render pins (host, no GPU) ───────────────────────────────────────────────

/// The staged pipeline on gfx1151 rendered through the launch path's pipeline
/// (post-optimization, linearize, render): the K loop pays exactly one
/// workgroup barrier per strip — the commit of both operands is one fenced
/// store — and the prologue one, so the implicit-barrier pass adds none. Also
/// pins the vector LDS path: no 16-bit LDS access survives.
#[test]
fn staged_gemm_gfx1151_fences_each_strip_once() {
    use std::sync::Arc;

    use svod_dtype::{AmdArch, DeviceSpec};
    use svod_ir::UOp;

    use crate::kernels::gemm::build_gemm_nt;

    let (m, k, n) = (128usize, 128usize, 64usize);
    let cfg = RDNA_TILES[0];
    let caps = crate::ArchCaps::for_amd(AmdArch::Gfx1151);
    let buffers: Vec<Arc<UOp>> =
        [m * n, m * k, n * k].into_iter().map(|size| UOp::new_buffer(DeviceSpec::Cpu, size, DType::BFloat16)).collect();
    let ker = crate::Kernel::new("gemm_nt", cfg.grid_dims(m, n), cfg.threads(caps.wave_size), buffers, caps);
    build_gemm_nt(&ker, (m, k, n), cfg, DType::BFloat16, DType::BFloat16, Epilogue::Plain);
    let sink = ker.finish(cfg.acc_m);

    let renderer = svod_codegen::llvm::LlvmTextRenderer::amd(AmdArch::Gfx1151);
    let opt = svod_schedule::OptimizerRenderer::for_amd_arch(AmdArch::Gfx1151).with_rewrite_capabilities(
        svod_ir::RendererOps::all(),
        svod_codegen::traits::Renderer::decompositor(&renderer),
        None,
    );
    let optimized = svod_schedule::apply_post_optimization_with_renderer(sink, &opt).expect("post optimization");
    let program =
        svod_codegen::program_pipeline::program_from_sink(optimized, DeviceSpec::Cpu).expect("final target graph");
    let linearized = svod_codegen::program_pipeline::do_linearize(&program).expect("do_linearize");
    let linear =
        linearized.toposort().into_iter().find(|u| matches!(u.op(), svod_ir::Op::Linear(..))).expect("LINEAR present");
    let code = svod_codegen::traits::Renderer::render(&renderer, &linear, Some("gemm_nt")).expect("render").code;

    let barriers = code.lines().filter(|l| l.contains("llvm.amdgcn.s.barrier()") && !l.contains("declare")).count();
    assert_eq!(barriers, 2, "one fence for the prologue and one per K strip:\n{code}");
    assert!(code.contains("wmma.f32.16x16x16"), "the wave32 WMMA path");
    let narrow = code.lines().filter(|l| l.contains("addrspace(3)") && l.contains(" bfloat,")).count();
    assert_eq!(narrow, 0, "LDS is read and written in vector groups, never one element at a time");
}
