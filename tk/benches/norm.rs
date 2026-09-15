//! Criterion GPU-device-time bench for `svod_tk::rms_norm` / `add_rms_norm` —
//! the one-pass row norms — vs the graph's `RmsNorm` (a reduce kernel plus a
//! broadcast apply) on transformer row shapes. See [`common`] for device-time
//! stamping and self-skip.
//!
//! Run: `SVOD_DEVICE={CUDA,AMD}:0 cargo bench -p svod-tk --bench norm`

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use svod_tensor::nn::{Layer, RmsNorm};

mod common;
use common::{bench_kernel, bench_plan, randn_bf16, requirements_met};

const EPS: f64 = 1e-6;
/// `(rows, D)`: Qwen3's hidden rows at three batch sizes.
const SHAPES: &[(usize, usize)] = &[(4096, 1024), (1024, 1024), (128, 1024)];

fn bench_norm(c: &mut Criterion) {
    if !requirements_met(svod_tk::NORM_SUPPORTED_ARCHS) {
        eprintln!("svod-tk norm bench: skipped (no supported GPU / toolchain)");
        return;
    }
    let mut group = c.benchmark_group("rms_norm");
    for &(rows, d) in SHAPES {
        let id = format!("{rows}x{d}");
        let (x, residual, weight) = (randn_bf16(&[rows, d]), randn_bf16(&[rows, d]), randn_bf16(&[d]));
        let norm = RmsNorm { weight: weight.clone(), eps: EPS };

        // `y = rms_norm(x)`: one pass vs the graph's reduce + apply.
        group.throughput(Throughput::Bytes((2 * rows * d * 2) as u64));
        let y = svod_tk::rms_norm(&x, &weight, EPS).expect("rms_norm build").expect("the kernel applies");
        let plan = y.prepare().expect("prepare rms_norm");
        group.bench_with_input(BenchmarkId::new("tk", &id), &id, |b, _| bench_kernel(b, &plan, "rms_norm"));
        let reference = norm.forward(&x).expect("graph norm").contiguous();
        let ref_plan = reference.prepare().expect("prepare graph norm");
        group.bench_with_input(BenchmarkId::new("graph", &id), &id, |b, _| bench_plan(b, &ref_plan));

        // `(h, y) = (x + r, rms_norm(x + r))`: the residual stream and its norm in one pass.
        group.throughput(Throughput::Bytes((4 * rows * d * 2) as u64));
        let (h, y) = svod_tk::add_rms_norm(&x, &residual, &weight, EPS).expect("add_rms_norm build").expect("applies");
        let plan = svod_tensor::Tensor::prepare_batch([&h, &y]).expect("prepare add_rms_norm");
        group.bench_with_input(BenchmarkId::new("tk_add", &id), &id, |b, _| bench_kernel(b, &plan, "add_rms_norm"));
        let h = x.try_add(&residual).expect("residual add").contiguous();
        let y = norm.forward(&h).expect("graph norm").contiguous();
        let ref_plan = svod_tensor::Tensor::prepare_batch([&h, &y]).expect("prepare graph add + norm");
        group.bench_with_input(BenchmarkId::new("graph_add", &id), &id, |b, _| bench_plan(b, &ref_plan));
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(common::bench_profiler());
    targets = bench_norm
}
criterion_main!(benches);
