mod atoms;
mod attention;
mod build;
mod device;
mod interp;
mod layout;
mod layouts;
mod ops;
mod ops_plan;
mod parts;
mod rows;
mod schedule;

use crate::build::BF16;
use crate::ir::Program;
use crate::kernels::Batch;
use crate::kernels::gemm::{Epilogue, GemmCfg, GemmSpec, gemm};

/// A plain bf16 `c = a·bᵀ` program on a 2×4 warp grid.
fn gemm_nt(m: usize, n: usize, k: usize, bm: usize, bn: usize, bk: usize, stages: usize) -> Program {
    gemm_nt_batched(m, n, k, [bm, bn, bk], stages, Batch::Static(1))
}

fn gemm_nt_batched(m: usize, n: usize, k: usize, tile: [usize; 3], stages: usize, batch: Batch) -> Program {
    let cfg = GemmCfg { tile, stages, warps: [2, 4], group_m: 0, unroll: true };
    gemm::<BF16>(&GemmSpec { m, n, k, batch, epilogue: Epilogue::default(), cfg })
}
