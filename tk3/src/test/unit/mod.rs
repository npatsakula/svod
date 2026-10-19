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
mod probe;
mod rows;
mod schedule;
mod sync;
mod targets;
mod tune;

use std::sync::Arc;

use crate::atoms::{Costs, Device, Target};
use crate::build::BF16;
use crate::ir::Program;
use crate::kernels::Batch;
use crate::kernels::conv::{ConvCfg, ConvGeom};
use crate::kernels::gemm::{Epilogue, GemmCfg, GemmSpec, gemm};
use crate::ops::config::Planner;

/// The default device's target when tk3 has tables for it, of any vendor.
fn device_target() -> Option<Target> {
    traced();
    let target = Target::for_device(&svod_dtype::default_device::default_device());
    target.filter(crate::ops::config::has_kernels)
}

/// The default device's planner as the op layer holds it: its costs probed
/// on the device (read from the tune store after the first run).
fn device_planner() -> Option<Arc<Planner>> {
    traced();
    crate::ops::planner(&svod_dtype::default_device::default_device())
}

/// The crate's tracing under `RUST_LOG` on stderr, for the probes: the
/// tuner's picks and round times at `svod_tk3=debug`.
fn traced() {
    use tracing_subscriber::EnvFilter;
    let _ = tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .without_time()
        .try_init();
}

/// What the tuner makes of `ranked` on the device, as the op layer tunes:
/// the candidates `expand` makes of the admitted shortlist (compiled, the
/// spilling ones dropped while clean ones remain) and its pick among them,
/// none where the planner offers no kernel (the op layer takes the graph).
pub(crate) fn tuned_on_device<C: crate::ops::config::Tiled + PartialEq + std::fmt::Debug>(
    planner: &Planner,
    (op, dtype, shape): (&'static str, svod_dtype::DType, &[usize]),
    ranked: &[C],
    expand: impl Fn(&[C]) -> Vec<C>,
    build: impl Fn(C) -> crate::tune::Candidate,
) -> (Vec<C>, Option<C>) {
    let mut tuner = crate::ops::Tuner::new(&planner.device, dtype, shape, "a device test", build);
    let candidates = expand(&tuner.admitted(op, ranked));
    let pick = (!candidates.is_empty()).then(|| {
        let ordered = tuner.ordered(op, &candidates);
        tuner.pipelined(op, &ordered, ranked)
    });
    (candidates, pick)
}

/// The front of a ranking: its first eight tiles, what a host test judges
/// a ranking by (the tuner's shortlist, which compiles on a device, runs
/// [`SHORTLIST`](crate::ops::config::SHORTLIST) deep).
pub(crate) fn first_eight<C: crate::ops::config::Tiled>(ranked: &[C]) -> Vec<C> {
    crate::ops::config::distinct(ranked).take(8).collect()
}

/// The convolution candidates of a ranking's [`first_eight`].
pub(crate) fn conv_first_eight(planner: &Planner, batches: usize, m: usize, g: &ConvGeom) -> Vec<ConvCfg> {
    planner.conv_splits(batches, m, g, &first_eight(&planner.conv_ranked(batches, m, g)))
}

/// Costs a test declares where it needs a ranking or an orientation: a
/// guessed sm_86-class fixture.
pub(crate) fn costs() -> Costs {
    Costs {
        smem_byte: 0.005,
        global_byte: 0.02,
        scatter: 8.0,
        step: 5.0,
        // A guess: three waves a SIMD hide the chain.
        step_lone: 15.0,
        score: 0.5,
        trip: 150.0,
        trip_rolled: 170.0,
        lane_move: 1.0,
        smem_trip: 8.0,
        latency: 600.0,
        l2_byte: 0.0,
        l2_bytes: 0.0,
    }
}

/// gfx1151's costs (Strix Halo, 40 CUs) as `probe::costs` measured them on
/// the device on 2026-10-14.
pub(crate) fn gfx1151_costs() -> Costs {
    Costs {
        smem_byte: 0.00328,
        global_byte: 0.1877,
        scatter: 3.05,
        step: 6.25,
        // Unmeasured there: equal to `step`, so a lone wave hides nothing.
        step_lone: 6.25,
        score: 0.0773,
        trip: 11.67,
        trip_rolled: 13.85,
        lane_move: 0.0316,
        smem_trip: 0.0841,
        latency: 577.9,
        l2_byte: 0.0,
        l2_bytes: 0.0,
    }
}

/// [`costs`] on a part whose lane moves cost more than a round trip through
/// shared memory: every exchange becomes the round trip.
pub(crate) fn lane_moves_dear() -> Costs {
    Costs { lane_move: 16.0, ..costs() }
}

/// A plain bf16 `c = a·bᵀ` program on a 2×4 warp grid.
fn gemm_nt(m: usize, n: usize, k: usize, bm: usize, bn: usize, bk: usize, stages: usize) -> Program {
    gemm_nt_batched(m, n, k, [bm, bn, bk], stages, Batch::Static(1))
}

fn gemm_nt_batched(m: usize, n: usize, k: usize, tile: [usize; 3], stages: usize, batch: Batch) -> Program {
    let cfg = GemmCfg { tile, stages, warps: [2, 4], group_m: 0, unroll: true };
    gemm::<BF16>(&GemmSpec { m, n, k, batch, epilogue: Epilogue::default(), cfg })
}

/// `target` under the declared [`costs`].
pub(crate) fn declared(target: Target) -> Device {
    Device { target, costs: costs() }
}

/// The planner of `target` under the declared [`costs`].
pub(crate) fn planner(target: Target) -> Planner {
    Planner::new(declared(target))
}
