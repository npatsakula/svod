mod atoms;
mod attention;
mod build;
mod conv;
mod device;
mod gather;
mod heads;
mod interp;
mod layout;
mod layouts;
mod ops;
mod ops_plan;
mod parts;
mod rows;
mod schedule;
mod sync;
mod targets;
mod tune;

use crate::atoms::Target;
use crate::build::BF16;
use crate::ir::Program;
use crate::kernels::Batch;
use crate::kernels::gemm::{Epilogue, GemmCfg, GemmSpec, gemm};

/// Every kernel's device time in seconds for one profiled run of `plan`, or
/// `None` when any kernel lost its stamps: a probe that dropped it would read
/// the run as faster than it was.
fn kernel_secs(plan: &svod_runtime::ExecutionPlan) -> Option<Vec<f64>> {
    let profile = plan.execute_profiled().unwrap();
    profile.iter().map(|k| Some(k.gpu_end_ns?.checked_sub(k.gpu_start_ns?)? as f64 * 1e-9)).collect()
}

/// The default device's target when tk3 has tables for it, of any vendor.
fn device_target() -> Option<Target> {
    let target = Target::for_device(&svod_dtype::default_device::default_device());
    target.filter(crate::ops::config::has_kernels)
}

/// A plain bf16 `c = a·bᵀ` program on a 2×4 warp grid.
fn gemm_nt(m: usize, n: usize, k: usize, bm: usize, bn: usize, bk: usize, stages: usize) -> Program {
    gemm_nt_batched(m, n, k, [bm, bn, bk], stages, Batch::Static(1))
}

fn gemm_nt_batched(m: usize, n: usize, k: usize, tile: [usize; 3], stages: usize, batch: Batch) -> Program {
    let cfg = GemmCfg { tile, stages, warps: [2, 4], group_m: 0, unroll: true };
    gemm::<BF16>(&GemmSpec { m, n, k, batch, epilogue: Epilogue::default(), cfg })
}
