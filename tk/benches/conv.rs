//! Criterion GPU-device-time bench for `svod_tk::conv2d_nhwc` — the implicit-GEMM
//! convolution (`act(x ⊛ w + bias)`, channels-last, bf16/f16 in and out) — against
//! svod's generic `Tensor::conv2d` on the YOLO26-x layer shapes. See [`common`]
//! for device-time stamping and self-skip.
//!
//! The kernel reads the NT GEMM's tile table, so what it can run is bounded by
//! that table: `cin` a multiple of the strip depth and `cout` of the tile's N
//! edge. The 96-channel row below is in the list precisely because no CUDA tile
//! serves it today — it is ~40% of a YOLO26 frame and it never reaches here.
//!
//! Run: `SVOD_DEVICE={CUDA,AMD}:0 cargo bench -p svod-tk --bench conv`

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use svod_dtype::DType;
use svod_tensor::Tensor;

mod common;
use common::{bench_kernel, bench_plan, requirements_met};

/// `(cin, cout, side, stride, label)` — one 3x3 `pad = 1` convolution each, at
/// batch 1. The first four are the shapes commit 7f6e8134 measured on gfx1201,
/// so the numbers here sit beside that message's; `side` is the *input* side.
const SHAPES: &[(usize, usize, usize, usize, &str)] = &[
    (768, 768, 80, 2, "768-768-s2-80"),
    (768, 768, 20, 1, "768-768-s1-20"),
    (384, 384, 160, 2, "384-384-s2-160"),
    (192, 192, 40, 1, "192-192-s1-40"),
    // The C3k bottleneck body and the box head's first conv: `cout = 96` tiles
    // no CUDA `block_n` (64) nor any RDNA4 one, so `conv2d_nhwc` declines it and
    // the graph kernel runs instead. Benched to size what a narrower tile buys.
    (96, 96, 80, 1, "96-96-s1-80"),
    (384, 96, 80, 1, "384-96-s1-80"),
];

/// A realized random tensor on the env-selected device, at the bench dtype.
fn randn(shape: &[usize], dtype: DType) -> Tensor {
    let t = Tensor::randn(shape).expect("randn").cast(dtype);
    t.realize().expect("realize");
    t
}

fn bench_conv2d_nhwc(c: &mut Criterion) {
    if !requirements_met(svod_tk::CONV_SUPPORTED_ARCHS) {
        eprintln!("svod-tk conv bench: skipped (no supported GPU / toolchain)");
        return;
    }
    // f16 is what the YOLO path computes in; the matrix core needs it either way.
    let dtype = DType::Float16;
    let mut group = c.benchmark_group("conv2d_nhwc");
    for &(cin, cout, side, stride, label) in SHAPES {
        let (kh, kw, pad) = (3usize, 3usize, 1usize);
        let out = (side + 2 * pad - kh) / stride + 1;
        group.throughput(Throughput::Elements(2 * (out * out * cout * kh * kw * cin) as u64));

        // Channels-last activation and taps-major weight: what the kernel binds.
        let x = randn(&[1, side, side, cin], dtype.clone());
        let w = randn(&[cout, kh, kw, cin], dtype.clone());
        let bias = randn(&[cout], dtype.clone());

        match svod_tk::conv2d_nhwc(&x, &w, &bias, None, stride, pad, true) {
            Ok(Some(y)) => {
                let plan = y.prepare().expect("prepare conv2d_nhwc");
                group.bench_with_input(BenchmarkId::new("tk", label), &label, |b, _| {
                    bench_kernel(b, &plan, "conv2d_nhwc")
                });
            }
            // No tile of the device's table serves this shape — the model keeps
            // its graph conv. Recording nothing says so louder than a zero would.
            Ok(None) => eprintln!("conv2d_nhwc: no tile serves {label}; only the generic row is recorded"),
            Err(err) => panic!("conv2d_nhwc build {label}: {err}"),
        }

        // Reference: the optimizer's own conv, NCHW as a model would hold it.
        let xn = randn(&[1, cin, side, side], dtype.clone());
        let wn = randn(&[cout, cin, kh, kw], dtype.clone());
        let reference = xn
            .conv2d()
            .weight(&wn)
            .stride(&[stride, stride])
            .padding(&[(pad as isize, pad as isize), (pad as isize, pad as isize)])
            .call()
            .expect("reference conv2d")
            .contiguous();
        let ref_plan = reference.prepare().expect("prepare reference");
        group.bench_with_input(BenchmarkId::new("generic", label), &label, |b, _| bench_plan(b, &ref_plan));
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(common::bench_profiler());
    targets = bench_conv2d_nhwc
}
criterion_main!(benches);
