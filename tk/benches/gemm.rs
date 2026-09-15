//! Criterion GPU-device-time bench for `svod_tk::gemm_nt` — the hand NT GEMM
//! (`y[M, N] = x[M, K] · w[N, K]ᵀ`, bf16 in and out) — against svod's generic
//! `Tensor::linear` on the transformer linear-layer shapes. See [`common`] for
//! device-time stamping and self-skip.
//!
//! Run: `SVOD_DEVICE=CUDA:0 cargo bench -p svod-tk --bench gemm`

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

mod common;
use common::{bench_kernel, bench_plan, randn_bf16, requirements_met};

/// The linear-layer shapes `(M, K, N)` the kernel is tuned for: Qwen3's gate/up,
/// fused QKV and down projections at two batch sizes, and Whisper's FFN.
const SHAPES: &[(usize, usize, usize)] = &[
    (4096, 1024, 6144),
    (1024, 1024, 6144),
    (1024, 1024, 4096),
    (1024, 1024, 2048),
    (4096, 3072, 1024),
    (1024, 3072, 1024),
    (3072, 1280, 5120),
    (128, 1024, 6144),
];

fn bench_gemm_nt(c: &mut Criterion) {
    if !requirements_met(svod_tk::kernels::gemm::GEMM_NT_SUPPORTED_ARCHS) {
        eprintln!("svod-tk gemm bench: skipped (no supported GPU / toolchain)");
        return;
    }
    let mut group = c.benchmark_group("gemm_nt");
    for &(m, k, n) in SHAPES {
        let id = format!("{m}x{k}x{n}");
        group.throughput(Throughput::Elements(2 * (m * k * n) as u64)); // 2·M·N·K
        let x = randn_bf16(&[m, k]);
        let w = randn_bf16(&[n, k]);

        let y = svod_tk::gemm_nt(&x, &w).expect("gemm_nt build").expect("the kernel applies to a tiling shape");
        let plan = y.prepare().expect("prepare gemm_nt");
        group.bench_with_input(BenchmarkId::new("tk", &id), &id, |b, _| bench_kernel(b, &plan, "gemm_nt"));

        // Reference: the generic optimizer's tensor-core GEMM behind `Tensor::linear`.
        let reference = x.linear().weight(&w).call().expect("reference linear").contiguous();
        let ref_plan = reference.prepare().expect("prepare reference");
        group.bench_with_input(BenchmarkId::new("generic", &id), &id, |b, _| bench_plan(b, &ref_plan));
    }
    group.finish();

    // The fused epilogues on the shapes that use them: SwiGLU on the gate/up
    // projection (`[M, 2I]` never written), the residual add on the down
    // projection — each against the plain GEMM plus the graph's elementwise pass.
    let mut group = c.benchmark_group("gemm_nt_epilogue");
    for &(m, k, n) in &[(4096usize, 1024usize, 6144usize), (1024, 1024, 6144), (128, 1024, 6144)] {
        let id = format!("{m}x{k}x{n}");
        group.throughput(Throughput::Elements(2 * (m * k * n) as u64));
        let (x, w) = (randn_bf16(&[m, k]), randn_bf16(&[n, k]));
        let pair = svod_tk::swiglu_pair_width().expect("a common pair width");
        let fused = svod_tk::gemm_nt_with_epilogue(&x, &w, svod_tk::Epilogue::SwiGlu { pair })
            .expect("gemm_nt build")
            .expect("the epilogue applies");
        let plan = fused.prepare().expect("prepare swiglu");
        group.bench_with_input(BenchmarkId::new("swiglu", &id), &id, |b, _| bench_kernel(b, &plan, "gemm_nt"));
        let y = svod_tk::gemm_nt(&x, &w).expect("gemm_nt build").expect("applies");
        let halves = y.split(&[n / 2, n / 2], -1).expect("split");
        let act = halves[0].silu().expect("silu").try_mul(&halves[1]).expect("mul").contiguous();
        let ref_plan = act.prepare().expect("prepare plain + silu");
        group.bench_with_input(BenchmarkId::new("plain+silu", &id), &id, |b, _| bench_plan(b, &ref_plan));
    }
    for &(m, k, n) in &[(4096usize, 3072usize, 1024usize), (1024, 3072, 1024), (128, 3072, 1024)] {
        let id = format!("{m}x{k}x{n}");
        group.throughput(Throughput::Elements(2 * (m * k * n) as u64));
        let (x, w, r) = (randn_bf16(&[m, k]), randn_bf16(&[n, k]), randn_bf16(&[m, n]));
        let fused = svod_tk::gemm_nt_with_epilogue(&x, &w, svod_tk::Epilogue::Add(&r))
            .expect("gemm_nt build")
            .expect("the epilogue applies");
        let plan = fused.prepare().expect("prepare add");
        group.bench_with_input(BenchmarkId::new("add", &id), &id, |b, _| bench_kernel(b, &plan, "gemm_nt"));
        let y = svod_tk::gemm_nt(&x, &w).expect("gemm_nt build").expect("applies");
        let sum = y.try_add(&r).expect("add").contiguous();
        let ref_plan = sum.prepare().expect("prepare plain + add");
        group.bench_with_input(BenchmarkId::new("plain+add", &id), &id, |b, _| bench_plan(b, &ref_plan));
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(common::bench_profiler());
    targets = bench_gemm_nt
}
criterion_main!(benches);
