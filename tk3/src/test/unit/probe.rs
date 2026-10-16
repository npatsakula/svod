//! The probes that measure a device's [`Costs`]: every pair builds and lowers
//! for targets this machine may not have, divides by the units its programs
//! do, and issues the primitive it claims; on the attached GPU the measured
//! costs are finite and ordered as the hardware is.

use std::collections::HashSet;
use std::sync::Arc;

use svod_dtype::{AmdArch, DType, DeviceSpec, GpuArch};
use svod_ir::UOp;
use test_case::test_case;

use super::targets::{amd, render};
use crate::atoms::{Costs, Target, sm86};
use crate::ir::*;
use crate::layouts::Relayout;
use crate::lower::{Lowered, Lowering, lower};
use crate::probe::{self, Probe, STREAM};

fn gfx1151() -> Target {
    amd(AmdArch::Gfx1151, 40)
}

fn lowered(prog: &Program, lowering: &Lowering) -> Lowered {
    let params: Vec<Arc<UOp>> =
        prog.params.iter().enumerate().map(|(i, p)| UOp::param(i, p.elems, DType::Scalar(p.dtype), None)).collect();
    let device = match lowering.device.target.arch {
        GpuArch::Cuda(_) => DeviceSpec::Cuda { device_id: 0 },
        _ => DeviceSpec::Amd { device_id: 0 },
    };
    lower(prog.clone(), lowering, params, device).unwrap_or_else(|e| panic!("{}: {e}", prog.name))
}

fn probes(target: &Target) -> Vec<Probe> {
    probe::probes(target).expect("a target with an SM count")
}

fn find<'a>(probes: &'a [Probe], field: &str) -> &'a Probe {
    probes.iter().find(|p| p.op == format!("probe:{field}")).expect("a probe of that field")
}

/// `(baseline, probe)` programs of a pair.
fn pair(p: &Probe) -> (&Program, &Program) {
    assert_eq!(p.candidates.len(), 2, "{}: a baseline and a probe", p.op);
    (&p.candidates[0][0].0, &p.candidates[1][0].0)
}

fn constant(prog: &Program, id: ScalarId) -> usize {
    match prog.scalar(id) {
        Scalar::Const(v) => *v as usize,
        other => panic!("{other:?} is not a constant"),
    }
}

/// The program's blocks and the extent of its loop or pipeline (0 without one).
fn work(prog: &Program) -> (usize, usize) {
    let blocks = prog.grid.iter().map(|&g| constant(prog, g)).product();
    let extent = prog.body.0.iter().find_map(|s| match s {
        Stmt::Loop(l) => Some(constant(prog, l.extent)),
        Stmt::Pipeline(p) => Some(constant(prog, p.extent)),
        _ => None,
    });
    (blocks, extent.unwrap_or(0))
}

/// The shape and bytes of every copy into registers from `tier` in `prog`'s loop.
fn loads_from(prog: &Program, tier: Tier) -> Vec<(Shape, usize)> {
    prog.walk()
        .filter(|(depth, _)| *depth > 0)
        .filter_map(|(_, s)| match s {
            Stmt::Copy { dst, src, .. } if prog.value(*dst).tier() == Tier::Reg && prog.value(*src).tier() == tier => {
                let v = prog.value(*src);
                Some((v.shape, v.shape.elems() * v.dtype.bytes()))
            }
            _ => None,
        })
        .collect()
}

/// The product of every probe's loop: the units each divides by, counted
/// from its programs.
fn counted(target: &Target, probes: &[Probe], field: &str) -> f64 {
    let (base, probe) = pair(find(probes, field));
    let ((blocks, extent), (_, base_extent)) = (work(probe), work(base));
    let warps = probe.warps as usize;
    let carried = || {
        let Some(Stmt::Loop(l)) = probe.body.0.iter().find(|s| matches!(s, Stmt::Loop(_))) else { panic!("a loop") };
        probe.value(l.carried[0].phi).shape.elems()
    };
    match field {
        "step" => {
            let mma = probe.walk().find_map(|(_, s)| match s {
                Stmt::Let { dst, op: TileOp::Mma { a, .. } } => {
                    Some(probe.value(*dst).shape.elems() * probe.value(*a).shape.cols)
                }
                _ => None,
            });
            (blocks * extent * mma.expect("a product") / 4096) as f64
        }
        "smem_byte" | "global_byte" => {
            let tier = if field == "smem_byte" { Tier::Smem } else { Tier::Global };
            (blocks * extent * loads_from(probe, tier).iter().map(|l| l.1).sum::<usize>()) as f64
        }
        "score" => (blocks * extent * carried()) as f64,
        // One wave's chain: every wave runs its own alongside the others.
        "scatter" => (extent * loads_from(probe, Tier::Global)[0].0.elems() / warps) as f64,
        // The work of every relayout of a trip, as the inference prices it.
        "smem_trip" | "lane_move" => {
            let moved: f64 = relayouts(target, find(probes, field), 1).iter().map(Relayout::work).sum();
            extent as f64 * moved * (carried() / warps) as f64
        }
        // Every trip of the pipeline, which the baseline loads in a plain loop.
        "trip" | "trip_rolled" => {
            assert_eq!(base_extent, extent, "{field}: as many trips");
            (blocks * extent) as f64
        }
        // Each wave waits its own trips out, alongside the others.
        "latency" => (extent - base_extent) as f64,
        other => unreachable!("{other}"),
    }
}

/// A cache probe's region holds a prime number of tiles that nearly fills
/// its footprint, so on every target's occupancy the sweep of its blocks
/// reaches every tile and the blocks an SM runs together, dealt any stride
/// apart (gfx1201's 64 CUs dealt in turn would put the 512 blocks of a
/// 64-tile region eight to a tile), read distinct ones.
#[test_case(sm86(), 28; "sm86")]
#[test_case(gfx1151(), 20; "gfx1151")]
#[test_case(amd(AmdArch::Gfx1201, 64), 64; "gfx1201")]
#[test_case(amd(AmdArch::Gfx942, 304), 304; "gfx942")]
fn a_cache_probe_sweeps_every_tile(target: Target, sms: u32) {
    let o = target.occupancy;
    let blocks = (sms * o.simds * o.waves / 4) as usize;
    let gcd = |mut a: usize, mut b: usize| {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    };
    for (footprint, _) in crate::probe::cache_probes(&target).unwrap() {
        let region = crate::probe::region_tiles(footprint);
        let tiles = footprint / (32 * 128 * 4);
        assert!(region <= tiles && region * 4 > tiles * 3, "{region} of {tiles} tiles");
        assert!((2..region).all(|d| !region.is_multiple_of(d)), "{region} tiles: prime");
        assert_eq!(gcd(blocks, region), 1, "{blocks} blocks over {region} tiles");
        assert!((1..=16).all(|r| !(sms as usize * r).is_multiple_of(region)), "an SM's blocks share no tile");
    }
}

/// Every probe pair lowers and renders for the target, and is keyed by
/// the field it measures; so do the cache probes, smallest footprint first.
#[test_case(sm86(); "sm86")]
#[test_case(gfx1151(); "gfx1151")]
#[test_case(amd(AmdArch::Gfx942, 304); "gfx942")]
fn every_probe_renders(target: Target) {
    let probes = probes(&target);
    let ops: Vec<String> = Costs::default().fields().map(|(f, _)| format!("probe:{f}")).into();
    assert_eq!(probes.iter().map(|p| p.op).collect::<Vec<_>>(), ops);
    let cached = crate::probe::cache_probes(&target).unwrap();
    assert!(cached.windows(2).all(|w| w[0].0 < w[1].0), "footprints grow");
    for p in probes.iter().chain(cached.iter().map(|(_, p)| p)) {
        for (prog, lowering) in p.candidates.iter().flatten() {
            let decides = ["probe:lane_move", "probe:smem_trip"].contains(&p.op);
            assert!(decides || lowering.device.costs == Costs::default(), "{}: only a relayout's kind is costed", p.op);
            assert!(!render(prog.clone(), lowering).is_empty(), "{}", p.op);
        }
    }
}

/// What each value divides by is what its programs do: a throughput
/// probe's blocks, a relayout probe's one wave, times its loop's extent
/// times the primitive per trip (the extra trips over the baseline's, for a
/// pipeline; every trip, for the rolled ring).
#[test_case(sm86(); "sm86")]
#[test_case(gfx1151(); "gfx1151")]
fn units_count_the_primitive(target: Target) {
    let probes = probes(&target);
    for p in &probes {
        let field = p.op.strip_prefix("probe:").expect("a probe op");
        let (_, probe) = pair(p);
        assert_eq!(p.units, counted(&target, &probes, field), "{}", p.op);
        if ["scatter", "lane_move", "smem_trip"].contains(&field) {
            let (blocks, warps) = (work(probe).0, probe.warps);
            let simds = target.occupancy.simds;
            assert_eq!((blocks as u32, warps), (target.sms.unwrap(), simds), "{}: a wave per SIMD", p.op);
        }
    }
    assert_eq!(find(&probes, "global_byte").units, STREAM as f64, "the stream is read once");
    let [trip, rolled] = ["trip", "trip_rolled"].map(|f| find(&probes, f));
    let unroll = |p: &Probe| -> Vec<bool> {
        p.candidates
            .iter()
            .map(|c| match c[0].1.schedule {
                crate::schedule::Schedule::Uniform { unroll, .. } => unroll,
            })
            .collect()
    };
    assert_eq!((unroll(trip), unroll(rolled)), (vec![true; 2], vec![false; 2]), "the ring unrolled, then rolled");
    assert_eq!(pair(trip), pair(rolled), "the same programs");
    let (_, latency) = pair(find(&probes, "latency"));
    assert_eq!((work(latency).0, latency.warps), (target.sms.unwrap() as usize, 1), "a lone wave per SM");
    let tile = loads_from(latency, Tier::Smem)[0].1;
    assert!(work(latency).0 * work(latency).1 * tile <= STREAM, "no line read twice in a run");
    let (_, step) = pair(find(&probes, "step"));
    let o = target.occupancy;
    let waves = work(step).0 as u32 * step.warps;
    assert!(waves >= 4 * target.sms.unwrap() * o.simds * o.waves, "the device is full four times");
}

/// The scatter pair reads the same rows, the probe through views whose
/// every element is a line of its own.
#[test]
fn scatter_walks_the_baseline_bytes_column_major() {
    let probes = probes(&gfx1151());
    let (base, probe) = pair(find(&probes, "scatter"));
    let stride = |prog: &Program| {
        prog.walk()
            .find_map(|(_, s)| match s {
                Stmt::Copy { src, .. } => match prog.value(*src).place {
                    Place::Global { stride, .. } => Some(stride.map(|s| constant(prog, s))),
                    _ => None,
                },
                _ => None,
            })
            .expect("a global load")
    };
    assert_eq!(loads_from(base, Tier::Global), loads_from(probe, Tier::Global));
    assert_eq!((stride(base), stride(probe)), ([512, 1], [1, 64]), "columns 128 bytes apart");
}

/// The rendered body of a program's one loop.
fn in_loop(ir: &str) -> &str {
    let start = ir.find("\nloop_body_").expect("a loop");
    let end = ir.rfind("\nloop_footer_").expect("a loop");
    &ir[start..end]
}

/// A probe issues the instruction it prices in its loop, and its baseline
/// nowhere. A shared tile is read through the ring's pointer, `%local0`.
#[test_case(sm86(), "step", "@llvm.nvvm.mma."; "sm86 mma.sync")]
#[test_case(gfx1151(), "step", "@llvm.amdgcn.wmma."; "gfx1151 wmma")]
#[test_case(sm86(), "smem_byte", "ptr %local0"; "sm86 shared load")]
#[test_case(gfx1151(), "smem_byte", "ptr %local0"; "gfx1151 shared load")]
#[test_case(sm86(), "lane_move", "@llvm.nvvm.shfl.sync.bfly"; "sm86 shuffle")]
#[test_case(gfx1151(), "lane_move", "@llvm.amdgcn.permlanex16"; "gfx1151 across rows")]
#[test_case(gfx1151(), "lane_move", "@llvm.amdgcn.update.dpp"; "gfx1151 within rows")]
#[test_case(amd(AmdArch::Gfx942, 304), "lane_move", "@llvm.amdgcn.ds.bpermute"; "gfx942 permute")]
#[test_case(gfx1151(), "smem_trip", "@llvm.amdgcn.s.barrier"; "gfx1151 fenced round trip")]
#[test_case(sm86(), "smem_trip", "@llvm.nvvm.barrier0"; "sm86 fenced round trip")]
#[test_case(sm86(), "score", "@llvm.exp2.f32"; "sm86 exp2")]
#[test_case(gfx1151(), "score", "@llvm.amdgcn.exp2.f32"; "gfx1151 exp2")]
fn probes_issue_their_primitive(target: Target, field: &str, needle: &str) {
    let probes = probes(&target);
    let p = find(&probes, field);
    let [base, with] = [0, 1].map(|i| {
        let (prog, lowering) = &p.candidates[i][0];
        render(prog.clone(), lowering)
    });
    assert!(in_loop(&with).contains(needle), "{field}: no {needle} in the loop");
    assert!(!base.contains(needle), "{field}: the baseline has {needle}");
}

/// The shared-memory probe's loop reads the ring and nothing fences it: the
/// one barrier lands the ring's writes before the loop.
#[test_case(sm86(), "@llvm.nvvm.barrier0"; "sm86")]
#[test_case(gfx1151(), "@llvm.amdgcn.s.barrier"; "gfx1151")]
fn shared_reads_are_unfenced(target: Target, barrier: &str) {
    let probes = probes(&target);
    let (prog, lowering) = &find(&probes, "smem_byte").candidates[1][0];
    let ir = render(prog.clone(), lowering);
    assert!(ir.contains(barrier) && !in_loop(&ir).contains(barrier), "a barrier before the loop only");
}

/// The relayouts a re-holding probe's programs make once laid: none in the
/// baseline, and the probe's per loop trip.
fn relayouts(target: &Target, p: &Probe, i: usize) -> Vec<Relayout> {
    let (prog, lowering) = &p.candidates[i][0];
    let Lowered { tile, layouts, .. } = lowered(prog, lowering);
    let lay = |v: ValId| layouts[v.index()].clone().expect("laid");
    tile.walk()
        .filter_map(|(_, s)| match s {
            Stmt::Let { dst, op: TileOp::Relayout { src } } => {
                Some(lay(*src).relayout(&lay(*dst), tile.warps, target.wave, &lowering.device.costs))
            }
            _ => None,
        })
        .collect()
}

/// The pinned sums are re-held there and back each iteration the way their
/// cost names: by half-wave exchanges of two picks per register (the lane
/// itself and the lane 16 away, then the neighbouring lane), or through
/// shared memory; the baseline holds every value naturally.
#[test_case(sm86(); "sm86")]
#[test_case(gfx1151(); "gfx1151")]
#[test_case(amd(AmdArch::Gfx942, 304); "wave64")]
fn reholding_probes_relayout_as_named(target: Target) {
    let probes = probes(&target);
    for (field, kind, masks) in [("lane_move", "Exchange", vec![16, 16, 1, 1]), ("smem_trip", "ViaSmem", vec![])] {
        let p = find(&probes, field);
        assert!(relayouts(&target, p, 0).is_empty(), "{field}");
        let moved = relayouts(&target, p, 1);
        assert_eq!(moved.len(), masks.len().max(2), "{field}");
        for (i, r) in moved.iter().enumerate() {
            assert!(format!("{r:?}").starts_with(kind), "{field}: {r:?}");
            if let Relayout::Exchange(picks) = r {
                let found: HashSet<u32> = picks.iter().flatten().map(|p| p.mask).collect();
                assert!(picks.iter().all(|p| p.len() == 2), "{field}: {picks:?}");
                assert_eq!(found, HashSet::from([0, masks[i]]), "{field}");
            }
        }
    }
}

/// A signal is strong when the probe's excess over its baseline is at least
/// half the probe's own time.
#[test_case(0, 1, true; "an empty baseline")]
#[test_case(50, 100, true; "half")]
#[test_case(51, 100, false; "under half")]
#[test_case(100, 100, false; "no signal")]
fn a_signal_is_half_the_probe(base: u64, with: u64, strong: bool) {
    assert_eq!(probe::strong(base, with), strong);
}

/// Without an SM count no throughput can be scaled to one SM: an error
/// naming the cause, before anything runs.
#[test]
fn costs_need_the_sm_count() {
    let target = Target { sms: None, ..gfx1151() };
    let err = probe::costs(&target).unwrap_err();
    assert!(matches!(err, probe::Error::NoSms { .. }), "{err}");
    assert!(matches!(probe::probes(&target), Err(probe::Error::NoSms { .. })));
}

/// On the attached GPU every cost measures, finite and non-negative, and
/// ordered as the hardware is: a lone trip's wait outlasts a matrix step, a
/// rolled ring's trip about an unrolled one's or more, and a round trip
/// through shared memory costs something. The retries absorb a pair timed alongside another
/// test's kernels.
#[test]
fn device_costs_are_physical() {
    let Some(target) = super::device_target() else {
        eprintln!("skipped: no device with kernels");
        return;
    };
    let c = probe::costs(&target).unwrap();
    eprintln!("{} ({:?} SMs): {c:#?}", target.arch.target_name(), target.sms);
    let mut fields = c;
    for (field, v) in fields.fields() {
        assert!(v.is_finite() && *v >= 0.0, "{field}: {c:?}");
    }
    assert!(c.latency > c.step, "{c:?}");
    assert!(c.trip_rolled >= 0.9 * c.trip, "{c:?}");
    assert!(c.smem_trip > 0.0, "{c:?}");
}

/// Every probe's signal on the attached GPU, timed once: at least half the
/// probe's own time, else the probe times its scaffolding more than its
/// primitive. Another kernel sharing the device breaks this, so it runs
/// alone.
#[test]
#[ignore = "perf probe: times every pair alone, run with --ignored under the GPU lock"]
fn probe_signals_are_strong() {
    let Some(target) = super::device_target() else {
        eprintln!("skipped: no device with kernels");
        return;
    };
    for p in &probes(&target) {
        let [Some(base), Some(with)] = crate::tune::measure(p.candidates.clone())[..] else {
            panic!("{}: not timed", p.op)
        };
        let signal = (with as f64 - base as f64) / with as f64;
        eprintln!("{}: baseline {base} ns, probe {with} ns, signal {signal:.3} of the probe", p.op);
        assert!(probe::strong(base, with), "{}: the signal is {signal:.3} of the probe's time", p.op);
    }
}
