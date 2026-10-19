mod atoms;
mod attention;
mod build;
mod conv;
mod device;
mod emit;
mod gather;
mod heads;
mod interp;
mod isa;
mod layout;
mod layouts;
mod offline;
mod ops;
mod ops_plan;
mod parts;
mod probe;
mod replay;
mod rows;
mod schedule;
mod sweep;
mod sync;
mod targets;
mod tune;

use std::sync::{Arc, LazyLock};

use svod_dtype::default_device::default_device;
use svod_dtype::{DeviceSpec, ScalarDType};
use test_case::test_case;

use crate::atoms::{Core, Costs, Fill, Target};
use crate::build::BF16;
use crate::interp::round_to;
use crate::ir::Program;
use crate::kernels::Batch;
use crate::kernels::gemm::{Epilogue, GemmCfg, GemmSpec, gemm};
use crate::ops::config::{Planner, has_kernels};
pub(crate) use crate::sweep::{conv_first_eight, costs, declared, first_eight, planner};

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

/// [`device_target`] when the device fills shared memory with `cp.async`,
/// for the tests that build those lowerings or rings deeper than two slots.
fn cp_async_target() -> Option<Target> {
    let target = device_target()?;
    if target.fill == Fill::Staged {
        eprintln!("skipped: no cp.async");
    }
    (target.fill == Fill::CpAsync).then_some(target)
}

/// [`device_target`] when the device has a warpgroup matrix core.
fn group_target() -> Option<Target> {
    let target = device_target()?;
    let group = !target.group_mma.is_empty();
    if !group {
        eprintln!("skipped: no warpgroup matrix core");
    }
    group.then_some(target)
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
    let store = crate::tune::TuneStore::global();
    let mut tuner = crate::ops::Tuner::new(store, &planner.device, dtype, shape, crate::ops::DEVICE_TEST, build);
    let candidates = expand(&tuner.admitted(op, &crate::ops::config::representatives(ranked)));
    let pick = (!candidates.is_empty()).then(|| {
        let ordered = tuner.ordered(op, &candidates);
        tuner.pipelined(op, &ordered, ranked)
    });
    (candidates, pick)
}

/// Each candidate's best device time in ns, its programs' summed, on
/// scratch buffers at capacity: compiled, then timed.
fn measure(candidates: impl IntoIterator<Item = crate::tune::Candidate>) -> Vec<Option<u64>> {
    let pool = crate::tune::ScratchPool::default();
    crate::tune::time(&mut crate::tune::compile(candidates.into_iter().collect(), &pool))
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

/// The next value of a 64-bit LCG stepping `seed`, uniform in `[-1, 1)`.
fn lcg(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0
}

/// `n` values `f(x)` of the [`lcg`] draws `x`, each rounded to `dtype`.
fn rounded(dtype: ScalarDType, n: usize, seed: &mut u64, f: impl Fn(f64) -> f64) -> Vec<f64> {
    (0..n).map(|_| round_to(dtype, f(lcg(seed)))).collect()
}

/// Little-endian `dtype` elements read back from a device buffer.
fn decode(dtype: ScalarDType, bytes: &[u8]) -> Vec<f64> {
    let half = |b: &[u8]| u16::from_le_bytes([b[0], b[1]]);
    match dtype {
        ScalarDType::Float32 => bytes.chunks(4).map(|b| f64::from(f32::from_le_bytes(b.try_into().unwrap()))).collect(),
        ScalarDType::BFloat16 => bytes.chunks(2).map(|b| bf16(half(b))).collect(),
        ScalarDType::Float16 => bytes.chunks(2).map(|b| half::f16::from_bits(half(b)).to_f64()).collect(),
        other => unreachable!("{other:?} output"),
    }
}

/// The bf16 of `bits`.
fn bf16(bits: u16) -> f64 {
    half::bf16::from_bits(bits).to_f64()
}

/// [`decode`] reads a 16-bit output as IEEE does: a NaN (all-ones exponent,
/// a payload) stays NaN rather than reading as the infinity a check would
/// take for an overflow, and infinities, subnormals and the largest finite
/// value are exact.
#[test_case(ScalarDType::Float16, 0x7e01, None; "f16 NaN")]
#[test_case(ScalarDType::Float16, 0xfc00, Some(f64::NEG_INFINITY); "f16 negative infinity")]
#[test_case(ScalarDType::Float16, 0x0001, Some(2f64.powi(-24)); "f16 least subnormal")]
#[test_case(ScalarDType::Float16, 0x7bff, Some(65504.0); "f16 largest finite")]
#[test_case(ScalarDType::BFloat16, 0xffc1, None; "bf16 NaN")]
#[test_case(ScalarDType::BFloat16, 0x7f80, Some(f64::INFINITY); "bf16 infinity")]
#[test_case(ScalarDType::BFloat16, 0xbf80, Some(-1.0); "bf16 minus one")]
fn a_16_bit_output_decodes_as_ieee(dtype: ScalarDType, bits: u16, want: Option<f64>) {
    let got = decode(dtype, &bits.to_le_bytes())[0];
    assert!(want.map_or(got.is_nan(), |w| got == w), "{bits:#06x}: {got}");
}

/// `|got - want| ≤ tol·max(|want|, 1)` everywhere: an error relative to the
/// value, or absolute below one. Returns the largest absolute difference.
fn assert_rel_close(what: &str, got: &[impl Copy + Into<f64>], want: &[impl Copy + Into<f64>], tol: f64) -> f64 {
    assert_eq!(got.len(), want.len(), "{what}: element count");
    let mut worst = 0.0f64;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let (g, w) = (g.into(), w.into());
        let diff = (g - w).abs();
        worst = worst.max(diff);
        assert!(diff <= tol * w.abs().max(1.0), "{what}[{i}] = {g}, want {w}");
    }
    worst
}

/// `|got - want| ≤ tol` everywhere, whatever the magnitude.
fn assert_abs_close(what: &str, got: &[f64], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len(), "{what}: element count");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= tol, "{what}[{i}] = {g}, want {w}");
    }
}
