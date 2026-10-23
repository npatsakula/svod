//! Planning and emitting for cards this machine has no device for: their
//! targets from the card table, their costs from a tune store copied from
//! them.

use super::Scratch;

use svod_dtype::DType;
use test_case::test_case;

use crate::atoms::{CARDS, Core, Target};
use crate::ops::config::{self, Planner};
use crate::ops::shape::{self, Extent, Plan};
use crate::probe::{self, Measured};
use crate::tune::{self, TuneStore};

/// A card's target is the architecture's with the card's SMs and dies, and
/// its shared memory per block is what the device reports.
#[test_case("rtx3060", 28, 1, 99 << 10; "sm_86")]
#[test_case("h100-pcie", 114, 1, 227 << 10; "sm_90 pcie")]
#[test_case("h100-sxm", 132, 1, 227 << 10; "sm_90 sxm")]
#[test_case("rtx5070", 48, 1, 99 << 10; "sm_120 5070")]
#[test_case("rtx5090", 170, 1, 99 << 10; "sm_120 5090")]
#[test_case("mi300x", 304, 8, 64 << 10; "gfx942")]
#[test_case("mi355x", 256, 8, 160 << 10; "gfx950")]
#[test_case("rx9070xt", 64, 1, 64 << 10; "gfx1201")]
#[test_case("strix-halo", 40, 1, 64 << 10; "gfx1151")]
fn a_card_reports_its_facts(name: &str, sms: u32, xcds: u32, smem: usize) {
    let target = Target::card(name).expect("a listed card");
    assert_eq!((target.sms, target.xcds, target.smem_bytes), (Some(sms), xcds, smem));
    assert!(config::has_kernels(&target), "{name} has kernels");
}

#[test]
fn every_card_is_named_once() {
    let mut names: Vec<&str> = CARDS.iter().map(|c| c.0).collect();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), CARDS.len());
    assert_eq!(Target::card("no such card").map(|t| t.arch), None);
}

/// The probe values an H100 PCIe stored on 2026-10-18, in femtoseconds per unit.
const H100: [(&str, u64); 21] = [
    ("smem_byte", 3580),
    ("global_byte", 60957),
    ("scatter", 2453247),
    ("step", 1869110),
    ("step_group", 1476177),
    ("step_lone", 7820313),
    ("score", 60163),
    ("trip", 5351440),
    ("trip_rolled", 5756836),
    ("trip_one", 6038452),
    ("lane_move", 104174),
    ("smem_trip", 153481),
    ("latency", 445814702),
    ("cached_1m", 9685),
    ("cached_2m", 13517),
    ("cached_4m", 15850),
    ("cached_8m", 16980),
    ("cached_16m", 17057),
    ("cached_32m", 18391),
    ("cached_64m", 28771),
    ("cached_128m", 43500),
];

/// A probe's store line for `device` under another compiler and probe
/// programs (`programs`) than this build's.
fn line(device: &str, op: &str, programs: &str, fs: u64) -> String {
    format!("value|probe:{op}|{device}|Int32||00000000000000c0|00000000000000aa|{programs} {fs}")
}

/// A scratch directory holding `lines` as the store file of `target`'s device.
fn store(name: &str, target: &Target, lines: &[String]) -> Scratch {
    let dir = Scratch::new(&format!("offline-{name}"));
    let file = TuneStore::at(Some(dir.to_path_buf())).file(&tune::device(target)).expect("a rooted store");
    std::fs::write(file, format!("format 4\n{}\n", lines.join("\n"))).expect("a store file");
    dir
}

/// Every field the store holds reads from it under another machine's toolchain and
/// format, the cache from its ladder; lines of other devices and ops are
/// passed over, and the store is left as it was.
#[test]
fn costs_read_from_a_copied_store() {
    let target = Target::card("h100-pcie").expect("a listed card");
    let mut lines: Vec<String> = H100.iter().map(|&(op, fs)| line("sm_90-114sm", op, "0000000000000001", fs)).collect();
    lines.push(line("sm_90-132sm", "step", "0000000000000001", 1));
    lines.push(
        "choice|gemm|sm_90-114sm|BFloat16|1x4096x4096x4096|0000000000000003|00000000000000aa|0000000000000004 0 100"
            .into(),
    );
    let dir = store("copied", &target, &lines);
    let before: Vec<_> = std::fs::read_dir(&dir).unwrap().map(|e| std::fs::read(e.unwrap().path()).unwrap()).collect();
    let Measured { costs, unmeasured } = probe::stored(&dir, &target).expect("the device's file");
    // Probes added since the store was copied are unmeasured.
    assert_eq!(unmeasured, ["step_i8", "trip_lone", "trip_one_lone", "latency_one"]);
    // A rolled ring is not probed where fills are `cp.async`: its line is passed over.
    let want = |op: &str| match op {
        "step_i8" | "trip_rolled" | "trip_lone" | "trip_one_lone" | "latency_one" => 0.0,
        _ => H100.iter().find(|f| f.0 == op).expect("a probe").1 as f64 * 1e-6,
    };
    let mut costs = costs;
    for (name, value) in costs.fields() {
        assert_eq!(*value, want(name), "{name}");
    }
    assert_eq!(costs.l2_byte, want("cached_1m"));
    // H100's 50 MB L2, as the ladder past its knee shows it.
    assert!((40e6..60e6).contains(&costs.l2_bytes), "{}", costs.l2_bytes);
    let after: Vec<_> = std::fs::read_dir(&dir).unwrap().map(|e| std::fs::read(e.unwrap().path()).unwrap()).collect();
    assert_eq!(before, after, "the store is read, never written");
    assert_eq!(probe::stored(&dir, &Target::card("h100-sxm").unwrap()), None, "no file for 132 SMs");
}

/// Of a probe's several lines the one this build's programs measured
/// counts; several of none of them leave the field unmeasured, as does a
/// missing one, and a ladder with a missing rung leaves the cache so.
#[test]
fn a_probe_of_several_lines_takes_this_builds() {
    let target = Target::card("h100-pcie").expect("a listed card");
    let step = probe::probes(&target).unwrap().into_iter().find(|p| p.op == "probe:step").unwrap();
    let ours = format!("{:016x}", tune::fingerprint(&step.candidates));
    let mut lines: Vec<String> = H100
        .iter()
        .filter(|f| !matches!(f.0, "step" | "trip" | "latency" | "cached_2m"))
        .map(|&(op, fs)| line("sm_90-114sm", op, "0000000000000001", fs))
        .collect();
    lines.extend([
        line("sm_90-114sm", "step", "0000000000000001", 9_000_000),
        line("sm_90-114sm", "step", &ours, 2_000_000),
        line("sm_90-114sm", "trip", "0000000000000001", 1),
        line("sm_90-114sm", "trip", "0000000000000002", 2),
    ]);
    let dir = store("several", &target, &lines);
    let Measured { costs, unmeasured } = probe::stored(&dir, &target).expect("the device's file");
    assert_eq!(costs.step, 2.0);
    assert_eq!((costs.trip, costs.latency, costs.l2_bytes), (0.0, 0.0, 0.0));
    assert_eq!(unmeasured, ["step_i8", "trip", "trip_lone", "trip_one_lone", "latency", "latency_one", "l2_bytes"]);
}

/// A planner needs no device: a card's target under its stored costs
/// plans the op layer's GEMM and attention on a machine whose op layer
/// has no planner for that card, its warpgroup core leading where the
/// card measured it faster.
#[test]
fn a_planner_plans_for_a_card_without_it() {
    let target = Target::card("h100-pcie").expect("a listed card");
    let lines: Vec<String> = H100.iter().map(|&(op, fs)| line("sm_90-114sm", op, "0000000000000001", fs)).collect();
    let dir = store("planner", &target, &lines);
    let planner = Planner::measured(target, probe::stored(&dir, &Target::card("h100-pcie").unwrap()).unwrap())
        .expect("an H100 has SMs");
    let bf16 = [DType::BFloat16; 2];
    let x = Extent { dims: vec![4096, 4096], var: false };
    let Plan::Kernel(gemm) = shape::linear(Some(&planner), &bf16, Some(&x), 4096, false) else { panic!("no GEMM") };
    assert_eq!(gemm[0].core, Core::Group, "{:?}", gemm[0]);
    let q = Extent { dims: vec![8, 1024, 16, 128], var: false };
    let Plan::Kernel(fa) = shape::attention(Some(&planner), &[DType::BFloat16; 3], Some(&q), Some(&q), true, false)
    else {
        panic!("no attention")
    };
    assert!(!fa.is_empty());
}
