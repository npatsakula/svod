mod atoms;
mod attention;
mod build;
mod conv;
mod cost_model;
mod device;
mod emit;
mod facts;
mod gather;
mod hazards;
mod heads;
mod interp;
mod isa;
mod layout;
mod layouts;
mod nearest;
mod offline;
mod ops;
mod ops_plan;
mod oracle;
mod parts;
mod probe;
mod rank;
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
use crate::kernels::nearest::{Floors, Metric};
use crate::ops::config::{self, Planner, has_kernels};
pub(crate) use crate::sweep::{conv_first_eight, costs, declared, first_eight, planner};

/// The default device's target when tk has kernels for it, of any vendor:
/// the gate of every device test. It skips (`None`) only where there is
/// nothing to run on: `SVOD_DEVICE=CPU`, a GPU tk has no kernels for, or no
/// `SVOD_DEVICE` and no GPU tk supports attached. A supported GPU the suite
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
                "tk has kernels for the attached {} but SVOD_DEVICE is unset: run with SVOD_DEVICE={select} \
                 (SVOD_DEVICE=CPU runs the host tests alone)",
                target.arch.target_name()
            );
        }
        eprintln!("skipped: no GPU tk has kernels for");
        return None;
    };
    if !has_kernels(&target) {
        eprintln!("skipped: tk has no kernels for {}", target.arch.target_name());
        return None;
    }
    Some(target)
}

/// The first GPU tk has kernels for on this machine and the `SVOD_DEVICE`
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
/// tuner's picks and round times at `svod_tk=debug`.
fn traced() {
    use tracing_subscriber::EnvFilter;
    let _ = tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .without_time()
        .try_init();
}

/// What the tuner makes of `ranked` on the device, as the op layer tunes
/// (keyed by `salt`, what else the programs vary with):
/// the candidates `expand` makes of the admitted shortlist (compiled, the
/// spilling ones dropped while clean ones remain) and its pick among them,
/// none where the planner offers no kernel (the op layer takes the graph).
pub(crate) fn tuned_on_device<C: crate::ops::config::Tiled + PartialEq + std::fmt::Debug>(
    planner: &Planner,
    (op, dtype, shape, salt): (&'static str, svod_dtype::DType, &[usize], impl std::fmt::Debug),
    ranked: &[C],
    expand: impl Fn(&[C]) -> Vec<C>,
    build: impl Fn(C) -> crate::tune::Candidate,
) -> (Vec<C>, Option<C>) {
    let store = crate::tune::TuneStore::global();
    let mut tuner = crate::ops::Tuner::new(store, &planner.device, dtype, shape, salt, build);
    let candidates = expand(&tuner.admitted(op, &crate::ops::config::representatives(ranked)));
    let pick = (!candidates.is_empty()).then(|| {
        let ordered = tuner.ordered(op, &candidates, crate::tune::Pass::Screen);
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

/// The 8-bit step `probe::costs` measured on each fixture's device on
/// 2026-10-22, after the fixture's other fields; the bf16 `step` of the
/// same run: gfx1151 7.77 (1.22x), sm_86 8.10 (4.0x), gfx1201 2.73 (2.0x).
const STEP_I8_GFX1151: f64 = 6.3627;
const STEP_I8_SM86: f64 = 2.0166;
const STEP_I8_GFX1201: f64 = 1.3564;

/// gfx1151's costs (Strix Halo, 40 CUs) as `probe::costs` measured them on
/// the device on 2026-10-14.
pub(crate) fn gfx1151_costs() -> Costs {
    Costs {
        smem_byte: 0.00328,
        global_byte: 0.1877,
        scatter: 3.05,
        step: 6.25,
        step_group: 0.0,
        step_i8: STEP_I8_GFX1151,
        // Unmeasured there: equal to `step`, so a lone wave hides nothing.
        step_lone: 6.25,
        step_lone_i8: 0.0,
        score: 0.0773,
        trip: 11.67,
        trip_rolled: 13.85,
        // Unmeasured there: a guess at the barrier pair's share.
        trip_one: 20.0,
        // Unmeasured there: a lone block's trip costs what the full SM's does.
        trip_lone: 0.0,
        trip_one_lone: 0.0,
        lane_move: 0.0316,
        smem_trip: 0.0841,
        latency: 577.9,
        // A register-staged slot loads a trip ahead as two do: not probed.
        latency_one: 0.0,
        l2_byte: 0.0,
        l2_bytes: 0.0,
        ladder: [(0, 0.0); crate::atoms::LADDER],
    }
}

/// gfx1151's costs as the Strix Halo's tune store held them on 2026-10-23
/// (`probe::stored`), every field measured: under them the inference
/// re-holds a 32×128 attention probability tile through shared memory
/// under a key mask, where [`gfx1151_costs`] has it exchanged.
pub(crate) fn gfx1151_store_costs() -> Costs {
    // The cache ladder the same store measured (fs per byte, 2026-10-23):
    // its knee runs 8–32 MB, the Infinity Cache answering most of a 32 MB
    // working set.
    let rungs = [
        (1 << 20, 11593.0),
        (2 << 20, 13374.0),
        (4 << 20, 22319.0),
        (8 << 20, 33120.0),
        (16 << 20, 36746.0),
        (32 << 20, 39297.0),
        (64 << 20, 143145.0),
        (128 << 20, 167180.0),
    ]
    .map(|(f, fs)| (f, fs * 1e-6));
    let mut costs = Costs {
        smem_byte: 0.00356,
        global_byte: 0.187412,
        scatter: 2.415543,
        step: 7.611237,
        step_group: 0.0,
        step_i8: 6.152153,
        step_lone: 12.841797,
        step_lone_i8: 0.0,
        score: 0.079565,
        trip: 1.905518,
        trip_rolled: 1.893921,
        trip_one: 1.983032,
        trip_lone: 51.728516,
        trip_one_lone: 59.355469,
        lane_move: 0.034459,
        smem_trip: 0.086365,
        latency: 598.285924,
        latency_one: 0.0,
        l2_byte: 0.0,
        l2_bytes: 0.0,
        ladder: [(0, 0.0); crate::atoms::LADDER],
    };
    set_ladder(&mut costs, &rungs);
    costs
}

/// sm_86's costs (RTX 3060, 28 SMs) as `probe::costs` measured them on the
/// device on 2026-10-19; the ladder rungs are the same device's store
/// (2026-10-25), its 3 MB L2 answering past none of 16 MB.
pub(crate) fn sm86_costs() -> Costs {
    let rungs = [(1 << 20, 15824.0), (2 << 20, 27536.0), (4 << 20, 29535.0), (8 << 20, 41680.0), (16 << 20, 61173.0)]
        .map(|(f, fs)| (f, fs * 1e-6));
    let mut costs = Costs {
        smem_byte: 0.003447,
        global_byte: 0.08274,
        scatter: 2.825,
        step: 8.196,
        step_group: 0.0,
        step_i8: STEP_I8_SM86,
        step_lone: 33.25,
        step_lone_i8: 0.0,
        score: 0.07915,
        trip: 1.81,
        trip_rolled: 0.0,
        trip_one: 3.692,
        trip_lone: 126.75,
        trip_one_lone: 149.33,
        lane_move: 0.09804,
        smem_trip: 0.2449,
        latency: 311.79,
        latency_one: 324.77,
        l2_byte: 0.0,
        l2_bytes: 0.0,
        ladder: [(0, 0.0); crate::atoms::LADDER],
    };
    set_ladder(&mut costs, &rungs);
    costs
}

/// gfx1201's costs (RX 9070 XT, 64 CUs) as `probe::costs` measured them on
/// the device on 2026-10-19; the ladder rungs are the same device's store
/// (2026-10-25), its 8 MB L2 and 64 MB Infinity Cache answering past half
/// of 128 MB.
pub(crate) fn gfx1201_costs() -> Costs {
    let rungs = [
        (1 << 20, 5704.0),
        (2 << 20, 5729.0),
        (4 << 20, 5831.0),
        (8 << 20, 5863.0),
        (16 << 20, 6596.0),
        (32 << 20, 10334.0),
        (64 << 20, 20732.0),
        (128 << 20, 58746.0),
        (256 << 20, 99716.0),
    ]
    .map(|(f, fs)| (f, fs * 1e-6));
    let mut costs = Costs {
        smem_byte: 0.00325,
        global_byte: 0.10823,
        scatter: 1.8443,
        step: 2.7104,
        step_group: 0.0,
        step_i8: STEP_I8_GFX1201,
        step_lone: 5.5957,
        step_lone_i8: 0.0,
        score: 0.059305,
        trip: 7.3026,
        trip_rolled: 7.1622,
        trip_one: 7.3544,
        trip_lone: 53.623,
        trip_one_lone: 57.119,
        lane_move: 0.024552,
        smem_trip: 0.078096,
        latency: 243.15,
        latency_one: 0.0,
        l2_byte: 0.0,
        l2_bytes: 0.0,
        ladder: [(0, 0.0); crate::atoms::LADDER],
    };
    set_ladder(&mut costs, &rungs);
    costs
}

/// A cost set's cache views from its ladder rungs, as measuring them would
/// have left them.
fn set_ladder(costs: &mut Costs, rungs: &[(usize, f64)]) {
    costs.ladder[..rungs.len()].copy_from_slice(rungs);
    (costs.l2_byte, costs.l2_bytes) = crate::probe::held(rungs, costs.global_byte);
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
    let cfg = GemmCfg { tile, stages: 1, warps: [2, 4], unroll: false, core: Core::Warp, crews: 2 };
    gemm::<BF16>(&GemmSpec { m, n, k, batch: Batch::Static(1), epilogue: Epilogue::default(), cfg })
}

fn gemm_nt_batched(m: usize, n: usize, k: usize, tile: [usize; 3], stages: usize, batch: Batch) -> Program {
    let cfg = GemmCfg { core: Core::Warp, tile, stages, warps: [2, 4], unroll: true, crews: 1 };
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
        ScalarDType::Int32 => bytes.chunks(4).map(|b| f64::from(i32::from_le_bytes(b.try_into().unwrap()))).collect(),
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

/// The planner's problem of `n` f16 rows against `m` candidates at depth
/// `d`, keeping `k` scores a row, with neither floors nor groups.
fn self_join(n: usize, m: usize, d: usize, metric: Metric, k: usize) -> config::NearestProblem {
    let (dtype, floors) = (ScalarDType::Float16, Floors::default());
    config::NearestProblem { n, m, d, dtype, metric, floors, groups: false, k }
}

/// A fresh directory under the system temp dir, removed when dropped (a
/// test's panic included).
pub(super) struct Scratch(std::path::PathBuf);

impl Scratch {
    pub(super) fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("svod-tk-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch dir");
        Self(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

impl std::ops::Deref for Scratch {
    type Target = std::path::Path;
    fn deref(&self) -> &std::path::Path {
        &self.0
    }
}

impl AsRef<std::path::Path> for Scratch {
    fn as_ref(&self) -> &std::path::Path {
        &self.0
    }
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
