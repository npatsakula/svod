//! End to end on the attached GPU: the lowered program computes what the
//! interpreter computes. Skips unless `SVOD_DEVICE` names a CUDA device.

use svod_dtype::{DType, DeviceSpec, ScalarDType, default_device::default_device};
use svod_tensor::Tensor;
use test_case::test_case;

use crate::atoms::Target;
use crate::interp::{round_to, run};
use crate::launch::graph_launch;
use crate::layouts::WarpGrid;
use crate::lower::Lowering;
use crate::schedule::{Prefetch, Schedule};

fn cuda_target() -> Option<Target> {
    let spec = default_device();
    matches!(spec, DeviceSpec::Cuda { .. }).then(|| Target::for_device(&spec)).flatten()
}

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
}

/// The pipelined GEMM over `cp.async` + `mma.sync` + `ldmatrix` matches the
/// interpreter's bf16/f32 result on every tile of the grid.
#[test_case(128, 128, 64, 128, 128, 32, 2, 2, 4; "one block, two stages")]
#[test_case(256, 128, 256, 128, 64, 32, 3, 4, 2; "four blocks, three stages")]
#[test_case(64, 64, 128, 64, 64, 64, 2, 2, 2; "64-wide k")]
#[allow(clippy::too_many_arguments)]
fn gemm_matches_the_interpreter(
    m: usize,
    n: usize,
    k: usize,
    bm: usize,
    bn: usize,
    bk: usize,
    stages: usize,
    wr: u32,
    wc: u32,
) {
    let Some(target) = cuda_target() else {
        eprintln!("skipped: no CUDA device");
        return;
    };
    let mut prog = super::programs::gemm_nt(m, n, k, bm, bn, bk, stages);
    prog.warps = wr * wc;
    let lowering = Lowering {
        target,
        schedule: Schedule::Uniform { prefetch: Prefetch::CpAsync },
        grid: WarpGrid { rows: wr, cols: wc },
        swizzle: true,
    };

    let mut seed = 11;
    let a: Vec<f32> = (0..m * k).map(|_| round_to(ScalarDType::BFloat16, lcg(&mut seed) as f64) as f32).collect();
    let b: Vec<f32> = (0..n * k).map(|_| round_to(ScalarDType::BFloat16, lcg(&mut seed) as f64) as f32).collect();
    let want = run(
        &prog,
        vec![a.iter().map(|&x| x as f64).collect(), b.iter().map(|&x| x as f64).collect(), vec![0.0; m * n]],
        &[("b", 1)],
    )
    .unwrap();

    let a_t = Tensor::from_slice(&a).cast(DType::BFloat16);
    let b_t = Tensor::from_slice(&b).cast(DType::BFloat16);
    let c_t = Tensor::empty(&[m * n], DType::BFloat16);
    let out = graph_launch(prog, &lowering, &[&a_t, &b_t, &c_t]).unwrap();
    let got: Vec<f32> = out.cast(DType::Float32).to_vec::<f32>().unwrap();
    let mut worst = 0.0f64;
    for (i, (g, w)) in got.iter().zip(&want[2]).enumerate() {
        let diff = (*g as f64 - w).abs();
        worst = worst.max(diff);
        assert!(diff <= 1.6e-2 * w.abs().max(1.0), "c[{}, {}] = {g}, interpreter {w}", i / n, i % n);
    }
    eprintln!("max abs diff {worst:.3e}");
}
