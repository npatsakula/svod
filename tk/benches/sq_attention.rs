//! Whisper single-query attention device-time comparison.
//!
//! Run: `SVOD_DEVICE=AMD:0` (or `CUDA:0`) `cargo bench -p svod-tk --bench sq_attention`.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use svod_dtype::DType;
use svod_tensor::Tensor;

mod common;
use common::{bench_plan, randn_bf16, requirements_met};

fn randn_f32(shape: &[usize]) -> Tensor {
    let mut t = randn_bf16(shape).cast(DType::Float32).expect("f32");
    t.realize().expect("realize");
    t
}

fn bench_sq_attention(c: &mut Criterion) {
    if !requirements_met(svod_tk::kernels::sq_attention::SQ_ATTENTION_SUPPORTED_ARCHS) {
        eprintln!("svod-tk single-query attention bench: skipped (unsupported target/toolchain)");
        return;
    }
    let (b, h, d) = (5usize, 20usize, 64usize);
    let mut group = c.benchmark_group("whisper_sq_attention");
    for &(mode, n) in &[("self_short", 449usize), ("self_full", 449usize), ("cross", 1500usize)] {
        group.throughput(Throughput::Elements((4 * b * h * n * d) as u64));
        let q = randn_f32(&[b, 1, h, d]);
        let k = randn_f32(&[b, n, h, d]);
        let v = randn_f32(&[b, n, h, d]);
        let lens_values = if mode == "self_short" { [8i32, 9, 7, 8, 9] } else { [444i32, 445, 446, 447, 448] };
        let lens = mode.starts_with("self").then(|| {
            let mut t = Tensor::from_slice(lens_values);
            t.realize().expect("realize lens");
            t
        });

        let splits: &[usize] = if mode == "cross" { &[1, 2, 4, 5, 10] } else { &[1] };
        for &split in splits {
            let opts =
                svod_tk::SqAttentionOpts { key_lens: lens.as_ref(), include_last: mode.starts_with("self"), split };
            let mut tk = svod_tk::single_query_attention(&q, &k, &v, opts).expect("sq attention").expect("supported");
            let tk_plan = tk.prepare().expect("prepare tk");
            group.bench_with_input(BenchmarkId::new(format!("tk/{mode}/split_{split}"), n), &n, |bencher, _| {
                bench_plan(bencher, &tk_plan)
            });
        }

        let perm = |t: &Tensor| t.try_permute(&[0, 2, 1, 3]).expect("permute");
        let (qp, kp, vp) = (perm(&q), perm(&k), perm(&v));
        let mask = mode.starts_with("self").then(|| {
            let values: Vec<bool> =
                lens_values.iter().flat_map(|&len| (0..n).map(move |i| i >= len as usize && i + 1 != n)).collect();
            Tensor::from_slice(values.as_slice()).try_reshape([b, 1, 1, n]).expect("self mask")
        });
        let sdpa = qp.scaled_dot_product_attention().key(&kp).value(&vp).is_causal(false);
        let mut sdpa = match &mask {
            Some(mask) => sdpa.attn_mask(mask).call().expect("masked sdpa"),
            None => sdpa.call().expect("sdpa"),
        };
        let sdpa_plan = sdpa.prepare().expect("prepare sdpa");
        group.bench_with_input(BenchmarkId::new(format!("sdpa/{mode}"), n), &n, |bencher, _| {
            bench_plan(bencher, &sdpa_plan)
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(common::bench_profiler());
    targets = bench_sq_attention
}
criterion_main!(benches);
