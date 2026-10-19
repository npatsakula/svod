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

use std::sync::{Arc, LazyLock};

use svod_dtype::DeviceSpec;
use svod_dtype::default_device::default_device;

use crate::atoms::{Costs, Device, Target};
use crate::build::BF16;
use crate::ir::Program;
use crate::kernels::Batch;
use crate::kernels::conv::{ConvCfg, ConvGeom};
use crate::kernels::gemm::{Core, Epilogue, GemmCfg, GemmSpec, gemm};
use crate::ops::config::{Planner, has_kernels};

/// The default device's target when tk3 has kernels for it, of any vendor:
/// the gate of every device test. It skips (`None`) only where there is
/// nothing to run on: `SVOD_DEVICE=CPU`, a GPU tk3 has no kernels for, or no
/// `SVOD_DEVICE` and no GPU tk3 supports attached. A supported GPU the suite
/// is not pointed at, or an `SVOD_DEVICE` that does not open, fails the
/// test, since a suite whose device tests all skip passes and proves nothing.
fn device_target() -> Option<Target> {
    traced();
    let device = default_device();
    let Some(target) = Target::for_device(&device) else {
        assert_eq!(device, DeviceSpec::Cpu, "the default device does not open");
        if std::env::var_os("SVOD_DEVICE").is_some() {
            eprintln!("skipped: SVOD_DEVICE selects the CPU");
            return None;
        }
        if let Some((select, target)) = attached() {
            panic!(
                "tk3 has kernels for the attached {} but SVOD_DEVICE is unset: run with SVOD_DEVICE={select} \
                 (SVOD_DEVICE=CPU runs the host tests alone)",
                target.arch.target_name()
            );
        }
        eprintln!("skipped: no GPU tk3 has kernels for");
        return None;
    };
    if !has_kernels(&target) {
        eprintln!("skipped: tk3 has no kernels for {}", target.arch.target_name());
        return None;
    }
    Some(target)
}

/// The first GPU tk3 has kernels for on this machine and the `SVOD_DEVICE`
/// that selects it, found without selecting it: AMD ones from the KFD
/// topology in sysfs, a CUDA one once the driver counts devices (`cuInit`
/// creates no context; reading the architecture opens device 0's).
fn attached() -> Option<(String, Target)> {
    static ATTACHED: LazyLock<Option<(String, Target)>> = LazyLock::new(|| {
        let amd = (0..svod_device::amd::topology::enumerate().len()).map(|device_id| DeviceSpec::Amd { device_id });
        let cuda = svod_device::cuda::has_devices().then_some(DeviceSpec::Cuda { device_id: 0 });
        amd.chain(cuda).find_map(|spec| {
            let target = Target::for_device(&spec).filter(has_kernels)?;
            let select = match spec {
                DeviceSpec::Amd { device_id: 0 } => "AMD".to_string(),
                DeviceSpec::Amd { device_id } => format!("AMD:{device_id}"),
                _ => "CUDA".to_string(),
            };
            Some((select, target))
        })
    });
    ATTACHED.clone()
}

/// The default device's planner as the op layer holds it, behind
/// [`device_target`]'s gate: its costs probed on the device (read from the
/// tune store after the first run).
fn device_planner() -> Option<Arc<Planner>> {
    device_target()?;
    Some(crate::ops::planner(&default_device()).expect("the device's cost probes measure"))
}

/// [`device_target`] when the device is a CUDA one.
fn cuda_target() -> Option<Target> {
    let target = device_target()?;
    let cuda = matches!(target.arch, svod_dtype::GpuArch::Cuda(_));
    if !cuda {
        eprintln!("skipped: not a CUDA device");
    }
    cuda.then_some(target)
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
    let candidates = expand(&tuner.admitted(op, &crate::ops::config::representatives(ranked)));
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
        step_group: 0.0,
        // A guess: three waves a SIMD hide the chain.
        step_lone: 15.0,
        score: 0.5,
        trip: 150.0,
        trip_rolled: 170.0,
        // A guess: the barrier pair costs about what the two-slot trip does again.
        trip_one: 300.0,
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
        step_group: 0.0,
        // Unmeasured there: equal to `step`, so a lone wave hides nothing.
        step_lone: 6.25,
        score: 0.0773,
        trip: 11.67,
        trip_rolled: 13.85,
        // Unmeasured there: a guess at the barrier pair's share.
        trip_one: 20.0,
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

/// A plain bf16 `c = a·bᵀ` program for two crews: a 2×4 warp grid over one slot.
fn gemm_nt_crews(m: usize, n: usize, k: usize, tile: [usize; 3]) -> Program {
    let cfg = GemmCfg { tile, stages: 1, warps: [2, 4], group_m: 0, unroll: false, core: Core::Warp, crews: 2 };
    gemm::<BF16>(&GemmSpec { m, n, k, batch: Batch::Static(1), epilogue: Epilogue::default(), cfg })
}

fn gemm_nt_batched(m: usize, n: usize, k: usize, tile: [usize; 3], stages: usize, batch: Batch) -> Program {
    let cfg = GemmCfg { core: Core::Warp, tile, stages, warps: [2, 4], group_m: 0, unroll: true, crews: 1 };
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
