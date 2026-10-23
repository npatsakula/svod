//! The probes that measure a device's [`Costs`]: every pair builds and lowers
//! for targets this machine may not have, divides by the units its programs
//! do, and issues the primitive it claims; on the attached GPU the measured
//! costs are finite and ordered as the hardware is.

use std::collections::HashSet;

use svod_dtype::{AmdArch, GpuArch};
use test_case::test_case;

use super::targets::{amd, hopper, lowered, render, run};
use crate::atoms::{Costs, Fill, Target, sm86};
use crate::ir::*;
use crate::layouts::Relayout;
use crate::lower::Lowered;
use crate::probe::{self, Probe, STREAM};

fn gfx1151() -> Target {
    amd(AmdArch::Gfx1151, 40)
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

/// The `m × n × k` of the product `prog` accumulates.
fn mma(prog: &Program) -> usize {
    prog.walk()
        .find_map(|(_, s)| match s {
            Stmt::Let { dst, op: TileOp::Mma { a, .. } } => {
                Some(prog.value(*dst).shape.elems() * prog.value(*a).shape.cols)
            }
            _ => None,
        })
        .expect("a product")
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
        "step" | "step_group" | "step_i8" => (blocks * extent * mma(probe) / 4096) as f64,
        "smem_byte" | "global_byte" => {
            let tier = if field == "smem_byte" { Tier::Smem } else { Tier::Global };
            (blocks * extent * loads_from(probe, tier).iter().map(|l| l.1).sum::<usize>()) as f64
        }
        "score" => (blocks * extent * carried()) as f64,
        // One wave's chain: every wave runs its own alongside the others.
        "scatter" => (extent * loads_from(probe, Tier::Global)[0].0.elems() / warps) as f64,
        "step_lone" => (extent * mma(probe) / 4096 / warps) as f64,
        // The work of every relayout of a trip, as the inference prices it.
        "smem_trip" | "lane_move" => {
            let moved: f64 = relayouts(target, find(probes, field), 1).iter().map(Relayout::work).sum();
            extent as f64 * moved * (carried() / warps) as f64
        }
        // Every trip of the pipeline, which the baseline loads in a plain
        // loop, a group of loads per iteration.
        "trip" | "trip_rolled" | "trip_one" | "trip_lone" | "trip_one_lone" => {
            let (base, _) = pair(find(probes, field));
            let group = loads_from(base, Tier::Global).len();
            assert_eq!(base_extent * group, extent, "{field}: as many trips");
            (blocks * extent) as f64
        }
        // Each wave waits its own trips out, alongside the others.
        "latency" | "latency_one" => (extent - base_extent) as f64,
        other => unreachable!("{other}"),
    }
}

/// A probe that fails leaves its field at zero and names itself; every
/// other field is what its probe measured, and a primitive the target
/// lacks (sm_86 has no warpgroup core, and prices no rolled ring apart
/// under `cp.async` fills) costs zero without being unmeasured.
#[test]
fn an_unmeasured_field_leaves_its_default() {
    let target = sm86();
    let measured = crate::probe::assemble(&target, probes(&target), |p| match p.op {
        "probe:step" => Err(crate::probe::Error::Unmeasured { probe: p.op, last: "no stamps".into() }),
        "probe:step_group" | "probe:trip_rolled" => unreachable!("an absent primitive is not measured"),
        _ => Ok(1.5),
    });
    assert_eq!(measured.unmeasured, ["step"]);
    let mut costs = measured.costs;
    for (name, value) in costs.fields() {
        assert_eq!(*value, if ["step", "step_group", "trip_rolled"].contains(&name) { 0.0 } else { 1.5 }, "{name}");
    }
}

/// The ladder's reading: the cache's byte is the smallest footprint's, never
/// dearer than DRAM's, and its bytes the most any footprint shows it
/// answering, the share of the footprint's reads at the cache's rate rather
/// than DRAM's times the footprint; a footprint slower than DRAM holds
/// nothing.
#[test_case(&[(1 << 20, 10.0), (2 << 20, 10.0), (4 << 20, 60.0), (8 << 20, 95.0)], 100.0, (10.0, 2.0 * (1 << 20) as f64); "a knee at two")]
#[test_case(&[(1 << 20, 10.0), (2 << 20, 11.0), (4 << 20, 40.0), (8 << 20, 95.0)], 100.0, (10.0, 4.0 * (1 << 20) as f64 * (60.0 / 90.0)); "a rung two thirds answered shows that share")]
#[test_case(&[(1 << 20, 10.0), (2 << 20, 10.0), (4 << 20, 55.0), (8 << 20, 77.5), (16 << 20, 88.75)], 100.0, (10.0, 2.0 * (1 << 20) as f64); "every rung past the cache shows the same bytes")]
#[test_case(&[(1 << 20, 10.0), (2 << 20, 10.0), (4 << 20, 30.0), (8 << 20, 31.0), (16 << 20, 90.0)], 100.0, (10.0, 8.0 * (1 << 20) as f64 * (69.0 / 90.0)); "a second level counts for the share it answers")]
#[test_case(&[(1 << 20, 120.0), (2 << 20, 130.0)], 100.0, (100.0, 0.0); "slower than DRAM holds nothing")]
fn the_ladder_reads_the_cache(rates: &[(usize, f64)], global_byte: f64, want: (f64, f64)) {
    assert_eq!(crate::probe::held(rates, global_byte), want);
}

/// The ladder is climbed smallest footprint first and stops once two
/// footprints in a row read nearer DRAM than the cache, so a device with a
/// small cache measures a few rungs and one whose cache holds every
/// footprint measures them all; a rung that cannot be measured (a rate of
/// 0 here) fails the ladder. A list of rates repeats its last.
#[test_case(&[10.0, 11.0, 60.0, 95.0, 100.0], 100.0, 4, Some((10.0, 2.0 * (1 << 20) as f64 * (89.0 / 90.0))); "a small cache stops the ladder at four")]
#[test_case(&[10.0, 12.0, 30.0, 31.0, 90.0, 95.0], 100.0, 6, Some((10.0, 8.0 * (1 << 20) as f64 * (69.0 / 90.0))); "a second level is climbed through")]
#[test_case(&[120.0, 130.0], 100.0, 2, Some((100.0, 0.0)); "slower than DRAM stops at two")]
#[test_case(&[10.0], 100.0, 9, Some((10.0, (STREAM / 2) as f64)); "a cache holding half the stream climbs every rung")]
#[test_case(&[10.0, 11.0, 0.0], 100.0, 3, None; "a rung that fails")]
fn the_ladder_stops_past_the_cache(rates: &[f64], global_byte: f64, want_calls: usize, want: Option<(f64, f64)>) {
    let probes = crate::probe::cache_probes(&sm86()).unwrap();
    assert_eq!(probes.last().map(|p| p.0), Some(STREAM / 2), "the ladder reaches half the stream");
    let mut calls = 0;
    let got = crate::probe::climbed(probes, global_byte, |p| {
        let rate = rates.get(calls).or(rates.last()).copied().unwrap();
        calls += 1;
        (rate > 0.0).then_some(rate).ok_or(crate::probe::Error::Unmeasured { probe: p.op, last: "lost".into() })
    });
    assert_eq!(got.ok(), want);
    assert_eq!(calls, want_calls);
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
#[test_case(hopper(); "sm90")]
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
/// pipeline; every trip, for the rolled ring). A lone trip probe runs a
/// block an SM.
#[test_case(sm86(); "sm86")]
#[test_case(hopper(); "sm90")]
#[test_case(gfx1151(); "gfx1151")]
fn units_count_the_primitive(target: Target) {
    let probes = probes(&target);
    for p in &probes {
        let field = p.op.strip_prefix("probe:").expect("a probe op");
        if p.candidates.is_empty() {
            let absent = match field {
                "step_group" => target.group_mma.is_empty(),
                "step_i8" => target.geometry_of(svod_dtype::ScalarDType::Int8).is_none(),
                "trip_rolled" => target.fill == Fill::CpAsync,
                "latency_one" => target.fill == Fill::Staged,
                _ => false,
            };
            assert!(absent, "{}: an absent primitive", p.op);
            continue;
        }
        let (_, probe) = pair(p);
        assert_eq!(p.units, counted(&target, &probes, field), "{}", p.op);
        if field.ends_with("_lone") && field.starts_with("trip") {
            assert_eq!(work(probe).0 as u32, target.sms.unwrap(), "{}: a block an SM", p.op);
        }
        if ["scatter", "step_lone", "lane_move", "smem_trip"].contains(&field) {
            let (blocks, warps) = (work(probe).0, probe.warps);
            let simds = target.occupancy.simds;
            assert_eq!((blocks as u32, warps), (target.sms.unwrap(), simds), "{}: a wave per SIMD", p.op);
        }
    }
    assert_eq!(find(&probes, "global_byte").units, STREAM as f64, "the stream is read once");
    let [trip, rolled, one] = ["trip", "trip_rolled", "trip_one"].map(|f| find(&probes, f));
    // Each candidate's ring: unrolled or not, `None` without one.
    let unroll = |p: &Probe| -> Vec<Option<bool>> {
        let ring = |c: &crate::tune::Candidate| {
            c[0].0.walk().find_map(|(_, s)| match s {
                Stmt::Pipeline(p) => {
                    assert_eq!(p.crews, 1, "a probe runs every warp alike");
                    Some(p.unroll)
                }
                _ => None,
            })
        };
        p.candidates.iter().map(ring).collect()
    };
    assert_eq!(unroll(trip), [None, Some(true)], "the ring unrolled");
    // A rolled ring is priced apart only where the fill goes through registers.
    let rolled = (target.fill == Fill::Staged).then_some(rolled);
    if let Some(rolled) = rolled {
        assert_eq!(unroll(rolled), [None, Some(false)], "then rolled");
        fn roll(block: &mut Block) {
            for s in &mut block.0 {
                if let Stmt::Pipeline(p) = s {
                    p.unroll = false;
                }
                s.blocks_mut().for_each(roll);
            }
        }
        let (base, mut probe) = (pair(trip).0.clone(), pair(trip).1.clone());
        roll(&mut probe.body);
        assert_eq!((&base, &probe), pair(rolled), "the same programs");
    }
    // The one-slot trip: the ring one slot deep, rolled, against a baseline
    // reading its fragments from a ring as deep.
    assert_eq!(unroll(one), [None, Some(false)], "a one-slot ring has nothing to unroll");
    let ring = |p: &Program| p.smem.iter().map(|a| a.elems).collect::<Vec<_>>();
    for p in [Some(trip), rolled, Some(one)].into_iter().flatten() {
        let (base, probe) = pair(p);
        assert_eq!(ring(base), ring(probe), "{}: the baseline reads a ring as deep as the probe's", p.op);
    }
    let depth = |p: &Program| {
        p.walk().find_map(|(_, s)| if let Stmt::Pipeline(p) = s { Some(p.stages) } else { None }).expect("a ring")
    };
    assert_eq!(
        (depth(pair(one).1), depth(pair(trip).1)),
        (1, target.fill.stages(2)),
        "one slot against the target's two"
    );
    let one = (target.fill == Fill::CpAsync).then_some(("latency_one", 1));
    for (field, stages) in [("latency", target.fill.stages(2))].into_iter().chain(one) {
        let (_, latency) = pair(find(&probes, field));
        assert_eq!((work(latency).0, latency.warps), (target.sms.unwrap() as usize, 1), "{field}: a lone wave per SM");
        let tile = loads_from(latency, Tier::Smem)[0].1;
        assert!(work(latency).0 * work(latency).1 * tile <= STREAM, "{field}: no line read twice in a run");
        assert_eq!(depth(latency), stages, "{field}: its ring's depth");
    }
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
#[test_case(hopper(), "step_group", "wgmma.mma_async.sync.aligned.m64n256k16"; "sm90 wgmma")]
#[test_case(sm86(), "step_i8", "@llvm.nvvm.mma.m16n8k32.row.col.satfinite.s8"; "sm86 s8 mma.sync")]
#[test_case(gfx1151(), "step_i8", "@llvm.amdgcn.wmma.i32.16x16x16.iu8"; "gfx1151 iu8 wmma")]
#[test_case(amd(AmdArch::Gfx942, 304), "step_i8", "@llvm.amdgcn.mfma.i32.16x16x32.i8"; "gfx942 i8 mfma")]
#[test_case(sm86(), "step_lone", "@llvm.nvvm.mma."; "sm86 lone mma.sync")]
#[test_case(gfx1151(), "step_lone", "@llvm.amdgcn.wmma."; "gfx1151 lone wmma")]
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
    let Lowered { tile, layouts, .. } =
        lowered(prog.clone(), lowering).unwrap_or_else(|e| panic!("{}: {e}", prog.name));
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

/// A probe settles on two timings that agree and keeps the lesser; timings
/// apart (one slowed by a kernel sharing the device) take a third, and the
/// least of them stands.
#[test_case(&[100, 105, 90] => (Some(100), 2); "two agree")]
#[test_case(&[100, 130, 120] => (Some(100), 3); "none agree, the least")]
#[test_case(&[130, 100, 103] => (Some(100), 3); "a slow first")]
#[test_case(&[] => (None, 0); "nothing timed")]
fn a_probe_keeps_the_least_of_agreeing_timings(timings: &[u64]) -> (Option<u64>, usize) {
    let taken = std::cell::Cell::new(0);
    let least = probe::least(timings.iter().copied().inspect(|_| taken.set(taken.get() + 1)));
    (least, taken.get())
}

/// On the attached GPU every cost measures (none is left at zero), finite
/// and non-negative, and ordered as the hardware is: a shared byte cheaper
/// than a global one, a cached one no dearer than DRAM's, a lone trip's wait
/// (through a one-slot `cp.async` ring too) outlasting a matrix step, a
/// rolled ring's trip about an unrolled two-slot one's or more, a one-slot
/// ring's within a factor of two of it either way (its two barriers a trip
/// cost more, its half footprint runs twice the blocks that hide them:
/// gfx1201 measures 5.9 ns to the two-slot ring's 7.5 once a trip's loads
/// are fenced ahead of its products, 7.2 to 7.2 before), a lone block's
/// trip no cheaper than one with the SM full, and a round trip through
/// shared memory costing something. The retries absorb a pair timed
/// alongside another test's kernels.
#[test]
fn device_costs_are_physical() {
    let Some(target) = super::device_target() else { return };
    let probe::Measured { costs: c, unmeasured } = probe::costs(&target).unwrap();
    eprintln!("{} ({:?} SMs): {c:#?}", target.arch.target_name(), target.sms);
    assert!(unmeasured.is_empty(), "unmeasured: {unmeasured:?}");
    let mut fields = c;
    for (field, v) in fields.fields() {
        assert!(v.is_finite() && *v >= 0.0, "{field}: {c:?}");
    }
    assert!(c.smem_byte > 0.0 && c.smem_byte < c.global_byte, "{c:?}");
    assert!(c.l2_byte > 0.0 && c.l2_byte <= c.global_byte, "{c:?}");
    assert!(c.latency > c.step, "{c:?}");
    assert!(target.fill != Fill::CpAsync || c.latency_one > c.step, "{c:?}");
    assert!(c.trip_lone >= c.trip && c.trip_one_lone >= c.trip_one, "{c:?}");
    assert!(c.step_lone >= c.step, "{c:?}");
    let int8 = target.geometry_of(svod_dtype::ScalarDType::Int8).is_some();
    assert_eq!(c.step_i8 > 0.0, int8, "{c:?}");
    assert!(c.smem_trip > 0.0, "{c:?}");
    // A ring the planner prices apart from the unrolled two-slot one does no
    // less than it, within what the trip pairs resolve: each pair's
    // run-to-run spread, per unit, timed again here. A one-slot ring is priced
    // on every target; a rolled one only where fills are register-staged
    // (`Costs::trip`): on `cp.async` targets the two rings differ by less
    // than one process's measurement differs from the next (sm_86 over four
    // fresh stores: trip 1.74–2.38 ns, rolled 1.86–2.48, each within 0.1 of
    // its own re-timing), so no probe measures the rolled one there.
    let probes = probes(&target);
    let spread = |field: &str| {
        let p = find(&probes, field);
        let mut built = crate::tune::compile(p.candidates.clone(), &crate::tune::ScratchPool::default());
        let r = probe::resolved(&probe::resolve_pairs(&mut built)).unwrap_or_else(|| panic!("{field}: not resolved"));
        r.spread * p.scale / p.units
    };
    let (trip, one) = (spread("trip"), spread("trip_one"));
    eprintln!("spreads per unit: trip {trip:.3}, trip_one {one:.3}");
    if target.fill == Fill::Staged {
        let rolled = spread("trip_rolled");
        eprintln!("trip_rolled {rolled:.3}");
        assert!(c.trip_rolled >= 0.9 * c.trip - (trip + rolled), "{c:?}");
    }
    assert!(c.trip_one >= 0.5 * c.trip - (trip + one) && c.trip_one <= 2.0 * c.trip + (trip + one), "{c:?}");
}

/// Every probe's signal on the attached GPU: a dominant one at least half
/// the probe's own time, else the probe times its scaffolding more than its
/// primitive; a resolved one clear of its pair's run-to-run spread. Another
/// kernel sharing the device breaks this, so it runs alone; a primitive the
/// target lacks has no pair.
#[test]
#[ignore = "perf probe: times every pair alone, run with --ignored under the GPU lock"]
fn probe_signals_are_strong() {
    let Some(target) = super::device_target() else { return };
    for p in probes(&target).iter().filter(|p| !p.candidates.is_empty()) {
        let [Some(base), Some(with)] = super::measure(p.candidates.clone())[..] else { panic!("{}: not timed", p.op) };
        let signal = (with as f64 - base as f64) / with as f64;
        eprintln!("{}: baseline {base} ns, probe {with} ns, signal {signal:.3} of the probe", p.op);
        match p.signal {
            probe::Signal::Dominant => {
                assert!(probe::strong(base, with), "{}: the signal is {signal:.3} of the probe's time", p.op)
            }
            probe::Signal::Resolved => {
                let mut built = crate::tune::compile(p.candidates.clone(), &crate::tune::ScratchPool::default());
                let pairs = probe::resolve_pairs(&mut built);
                let resolved = probe::resolved(&pairs);
                assert!(
                    resolved.is_some_and(|r| r.measurable() && r.clear()),
                    "{}: {pairs:?} resolve to {resolved:?}",
                    p.op
                );
            }
        }
    }
}

/// A trip pair moves the same bytes along the same path: the baseline loads
/// each chunk as the fill loads it (the register half of a fill), from a
/// region per SM, and reads a fragment back from shared memory as the ring's
/// consume does; on a target filling through `cp.async.cg` that load stays
/// out of L1 as the copy does (and no rolled ring is probed).
#[test_case(sm86(), true; "sm86")]
#[test_case(hopper(), true; "sm90")]
#[test_case(gfx1151(), false; "gfx1151")]
fn trip_baselines_load_as_the_fill_loads(target: Target, l2_only: bool) {
    let probes = probes(&target);
    for field in ["trip", "trip_rolled", "trip_one"] {
        let p = find(&probes, field);
        if field == "trip_rolled" && l2_only {
            assert!(p.candidates.is_empty(), "no rolled ring under cp.async");
            continue;
        }
        assert_eq!(p.signal, probe::Signal::Resolved, "{field}: a trip rides on the bytes it moves");
        let (base, probe) = pair(p);
        let loads: Vec<CopyMode> = base
            .walk()
            .filter_map(|(_, s)| match s {
                Stmt::Copy { src, dst, mode, .. }
                    if base.value(*src).tier() == Tier::Global && base.value(*dst).tier() == Tier::Reg =>
                {
                    Some(*mode)
                }
                _ => None,
            })
            .collect();
        let depth = |p: &Program| p.smem.iter().map(|a| a.elems).sum::<usize>() / (4 * target.wave / 8 * 32) as usize;
        assert_eq!(
            loads,
            vec![CopyMode::Staged; depth(probe)],
            "{field}: a load per slot, as the fill's register half"
        );
        let src = |prog: &Program| prog.params.iter().find(|q| q.name == "src").expect("a source").elems;
        let tile = (4 * target.wave / 8 * 32) as usize;
        assert_eq!(src(base), target.sms.unwrap() as usize * 16 * tile, "{field}: sixteen tiles per SM");
        assert_eq!(src(base), src(probe), "{field}: one source");
        let (_, lowering) = &p.candidates[0][0];
        let code = render(base.clone(), lowering);
        assert_eq!(code.contains("ld.global.cg.v4.u32"), l2_only, "{field}: the baseline's load policy");
    }
}

/// Every trip pair, as clang -O3 compiles it for RDNA, keeps a global load
/// inside its loop: no ring depth folds a trip's chunk or slot index to a
/// constant the loop's loads could be hoisted on. With the slot
/// `order % stages` and the chunk `step % 16`, the one-slot baseline's
/// sixteen loads all left the loop on gfx1201 and gfx1151.
#[test_case(amd(AmdArch::Gfx1201, 64); "gfx1201")]
#[test_case(amd(AmdArch::Gfx1151, 40); "gfx1151")]
fn trip_pairs_load_inside_their_loops(target: Target) {
    if !svod_runtime::amd::has_amdgpu_target() {
        eprintln!("skipped: clang has no AMDGPU target");
        return;
    }
    let GpuArch::Amd(arch) = target.arch else { unreachable!() };
    let probes = probes(&target);
    for field in ["trip", "trip_rolled", "trip_one"] {
        let p = find(&probes, field);
        let (_, lowering) = &p.candidates[0][0];
        let (base, probe) = pair(p);
        for (side, prog) in [("baseline", base), ("probe", probe)] {
            let mut clang = std::process::Command::new("clang");
            clang.args(["-x", "ir", "-O3", "-S", "--target=amdgcn-amd-amdhsa", "-nogpulib", "-", "-o", "-"]);
            clang.arg(format!("-mcpu={}", arch.mcpu()));
            let ir = render(prog.clone(), lowering);
            let isa = String::from_utf8(run(clang, ir.as_bytes(), "clang").unwrap()).unwrap();
            let (_, looped) = isa.split_once("\n.LBB0_").unwrap_or_else(|| panic!("{field} {side}: no loop"));
            assert!(looped.contains("global_load"), "{field} {side}: every global load left the loop");
        }
    }
}

/// A pair's timings resolve to the median difference, the spread of the
/// differences with their extremes trimmed and the value (the difference
/// where it clears the spread, else the spread bounding it), and are
/// measurable where the baseline takes at least half the probe's time: a
/// baseline under half its probe moved less than the bytes (its loads left
/// the loop on gfx1201 and gfx1151), and its difference would price them.
#[test_case(&[(100, 160), (102, 161), (101, 159)], Some((59.0, 0.0, 59.0, true)); "clear of the spread, a baseline that moves the bytes")]
#[test_case(&[(80, 160), (81, 161), (79, 159)], Some((80.0, 0.0, 80.0, true)); "a baseline half the probe")]
#[test_case(&[(29, 2400), (30, 2410), (29, 2395)], Some((2371.0, 0.0, 2371.0, false)); "a baseline whose loads left the loop")]
#[test_case(&[(100, 103), (110, 104), (104, 112), (100, 106), (103, 105)], Some((3.0, 4.0, 4.0, true)); "inside the spread, bounded by it")]
#[test_case(&[(150, 160), (100, 140), (101, 141), (102, 139), (99, 140)], Some((40.0, 3.0, 40.0, true)); "a slow first run trimmed")]
#[test_case(&[(200, 100), (201, 101), (202, 102)], Some((-100.0, 0.0, 0.0, true)); "faster than its baseline, bounded by the spread")]
#[test_case(&[(100, 160)], None; "one pair has no spread")]
#[test_case(&[(100, 160), (102, 161)], None; "two pairs leave none trimmed")]
#[test_case(&[(3000, 3230), (3300, 3525), (3100, 3328), (3400, 3633), (3200, 3420)], Some((228.0, 5.0, 228.0, true)); "a clock drifting between pairs")]
#[test_case(&[], None; "no pair")]
fn a_resolved_difference_clears_its_spread(pairs: &[(u64, u64)], want: Option<(f64, f64, f64, bool)>) {
    let got = probe::resolved(pairs).map(|r| (r.difference, r.spread, r.value(), r.measurable()));
    assert_eq!(got, want);
}
