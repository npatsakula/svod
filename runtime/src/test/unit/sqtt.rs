use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;

use proptest::prelude::*;
use svod_device::profile::{CodeImage, SeTrace, SqttTrace};
use svod_device::sqtt::asm::Counter;
use svod_device::sqtt::hidden::Hiding;
use svod_device::sqtt::stitch::objdump_lines;
use svod_dtype::AmdArch;
use test_case::test_case;

use svod_device::amd::occupancy::Limiter;

use super::verdict::{Buckets, load_bytes};
use super::{
    Analysis, Approx, BucketDelta, Capture, Caution, Class, Diff, Evidence, Finding, FindingKind, LoopRow, Partition,
    Pstate, Spread, Unit, WaitOn, align,
};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../device/src/test/fixtures/sqtt")
}

fn arch(disassembly: &str) -> AmdArch {
    let target = disassembly.lines().find_map(|l| l.trim().strip_prefix(".amdgcn_target \"")).unwrap();
    AmdArch::parse(target.trim_end_matches('"').rsplit('-').next().unwrap()).unwrap()
}

/// A device fixture as a [`Capture`]: AMD's `amd/<name>` (one SE) or tinygrad's `<arch>/<name>`.
fn capture(fixture: &str) -> Capture {
    let dir = fixtures().join(fixture);
    let read = |name: &str| std::fs::read(dir.join(name)).unwrap();
    let text = |name: &str| String::from_utf8(read(name)).unwrap();
    let (elf, disassembly, load_base, per_se) = if fixture.starts_with("amd/") {
        let base = u64::from_str_radix(text("base.txt").trim().trim_start_matches("0x"), 16).unwrap();
        let se = SeTrace { se: 0, itrace: true, bytes: read("se0.att"), truncated: false };
        (read("code_object.out"), text("code_object.s"), base, vec![se])
    } else {
        let golden: serde_json::Value = serde_json::from_slice(&read("golden.json")).unwrap();
        let ses = golden["events"].as_array().unwrap().iter().map(|e| SeTrace {
            se: e["se"].as_u64().unwrap() as u32,
            itrace: e["itrace"].as_bool().unwrap(),
            bytes: read(&format!("trace{}.bin", e["index"])),
            truncated: false,
        });
        (read("kernel.co"), text("kernel.s"), golden["base"].as_u64().unwrap(), ses.collect())
    };
    let code = Some(CodeImage { elf, load_base, entry_pc: 0 });
    let trace = SqttTrace { arch: arch(&disassembly), per_se, code, mclk_mhz: None };
    Capture {
        trace,
        disassembly,
        kernel_ns: None,
        traced_ns: None,
        labels: BTreeMap::from([("fixture".into(), fixture.into())]),
        resources: None,
        workgroup: None,
        grid: None,
        pstate: None,
    }
}

/// The fixture saved and loaded back, as an agent replays it.
fn replayed(fixture: &str) -> Analysis {
    let dir = tempfile::tempdir().unwrap();
    capture(fixture).save(dir.path()).unwrap();
    Analysis::load(dir.path()).unwrap()
}

/// Every exposed bucket within its raw one.
fn exposed_within(raw: &Partition, exposed: &Partition) -> bool {
    fn within<K: PartialEq>(raw: &[(K, i64)], exposed: &[(K, i64)]) -> bool {
        exposed.iter().all(|(k, e)| raw.iter().any(|(r, c)| r == k && (0..=*c).contains(e)))
    }
    within(&raw.issue, &exposed.issue)
        && within(&raw.stall, &exposed.stall)
        && within(&raw.wait, &exposed.wait)
        && within(&raw.idle, &exposed.idle)
        && (0..=raw.barrier).contains(&exposed.barrier)
}

#[test_case("amd/navi4_setpc")]
#[test_case("amd/navi4_sdata")]
#[test_case("amd/navi4_fifo")]
#[test_case("amd/navi4_barriernop")]
#[test_case("amd/strix_barrier")]
#[test_case("gfx1100/gemm")]
#[test_case("gfx1100/plus")]
#[test_case("gfx1200/gemm")]
#[test_case("gfx1200/plus")]
fn summarizes(fixture: &str) {
    let a = replayed(fixture);
    assert_eq!(a, capture(fixture).analyze().unwrap(), "save and load keep the analysis");
    assert!(a.waves > 0 && a.stitched == a.records && a.records > 0, "{fixture}: {} of {}", a.stitched, a.records);
    let sum = |f: fn(&super::Row) -> i64| a.pcs.iter().map(f).sum::<i64>();
    assert_eq!(a.classes.iter().map(|c| c.latency).sum::<i64>(), sum(|r| r.latency));
    assert_eq!(a.classes.iter().map(|c| c.stall).sum::<i64>(), sum(|r| r.stall));
    assert_eq!(a.classes.iter().map(|c| c.hits).sum::<u64>(), a.pcs.iter().map(|r| r.hits).sum::<u64>());
    assert!(a.pcs.iter().all(|r| r.hits > 0 && !r.asm.is_empty() && r.stall <= r.latency), "{fixture}");
    assert!(a.ses.iter().any(|s| s.itrace && s.span > 0 && s.occupancy_peak > 0));
    let span = a.ses.iter().filter(|s| s.itrace).map(|s| s.span).max().unwrap() as u64;
    let timed = Capture { kernel_ns: Some(span), ..capture(fixture) }.analyze().unwrap();
    assert_eq!((a.sclk_mhz, timed.sclk_mhz), (None, Some(1000)), "a cycle a ns is 1 GHz");
    assert!(a.lifetime.is_some_and(|l| l.min > 0 && l.min <= l.median && l.median <= l.max));
    assert_ne!(a.kernel, "?");
    let json = a.to_json();
    assert!(!json.contains('\n'));
    assert_eq!(serde_json::from_str::<Analysis>(&json).unwrap(), a);
    let text = a.text(5);
    assert!(text.starts_with(&format!("== sqtt {} ", a.kernel)) && text.contains("== hottest 5 by cycles"));
    assert!(!text.contains('\x1b') && !text.contains("!!"), "{text}");
    assert!(a.summary().starts_with(&a.kernel));
    assert_eq!(a.waits.waves + a.waits.skipped, a.waves);
    assert!(text.contains("== waits: "));
    assert_eq!(
        a.traced.iter().map(|t| (t.waves, t.complete)).fold((0, 0), |s, t| (s.0 + t.0, s.1 + t.1)),
        (a.waves, a.complete)
    );
    for findings in [&a.verdict.bound, &a.verdict.findings] {
        assert!(findings.windows(2).all(|w| w[0].share >= w[1].share), "largest share first");
    }
    assert!(text.lines().nth(1).unwrap().starts_with("== verdict"), "the verdict leads: {text}");
    if let Some(l) = &a.main_loop {
        let (p, e) = (&l.partition, &l.exposed);
        assert_eq!(p.buckets(), p.cycles, "{fixture}: every loop cycle in one bucket");
        assert_eq!(e.buckets(), e.cycles, "{fixture}: the exposed cycles add up");
        assert!(e.cycles <= p.cycles && exposed_within(p, e), "{fixture}: exposed within raw");
        let (mut sum, mut exposed) = (Partition::default(), Partition::default());
        l.ses.iter().for_each(|s| {
            sum.add(&s.partition);
            exposed.add(&s.exposed);
        });
        assert_eq!((&sum, &exposed), (p, e), "the SEs' partitions add up");
        let edges = a.pcs.iter().find(|r| r.pc == l.back_edge).unwrap().hits;
        assert_eq!(
            (l.trips.total, l.trips.labelled, l.trips.per_iteration),
            (edges, false, 1),
            "unlabelled: per iteration"
        );
    }
}

/// The waits report ELF PCs, as the rest of the analysis: v3's select pulls the ring fill's first wait
/// up to 0x74c, on the oldest of its six loads, waited within the trip that issued it.
#[test]
fn waits_report_elf_pcs() {
    let a = Analysis::load(&fixtures().join("svod/rdna4_gemm_v3")).unwrap();
    let site = a.waits.sites.iter().find(|s| s.pc == 0x74c).unwrap();
    assert_eq!(site.bindings, [(0x6fc, site.instances)]);
    let load = a.waits.bindings.iter().find(|b| b.pc == 0x6fc).unwrap();
    assert_eq!(load.distance, [(0, load.count)]);
    let row = |pc: u64| a.pcs.iter().find(|r| r.pc == pc).unwrap();
    assert_eq!((row(0x74c).class, row(0x6fc).class), (Class::WaitVmem, Class::VmemLoad));
    assert!(a.text(4).lines().any(|l| l.trim_start().starts_with("0xa64") && l.ends_with("global_load_b128 v[144:147], v[232:233], off")));
}

/// An empty kernel's waves issue only `s_endpgm`, which sends no token.
#[test_case("gfx1100/empty")]
#[test_case("gfx1200/empty")]
fn empty_kernel_has_no_records(fixture: &str) {
    let a = replayed(fixture);
    assert!(a.waves > 0 && a.records == 0 && a.pcs.is_empty() && a.classes.is_empty() && a.main_loop.is_none());
    assert!(a.text(5).contains("records    0 stitched of 0"));
}

/// tinygrad's GEMM unrolls its reduction: no backward branch, so no main loop.
#[test_case("gfx1100/gemm")]
#[test_case("gfx1200/gemm")]
fn unrolled_has_no_main_loop(fixture: &str) {
    let a = replayed(fixture);
    assert_eq!(a.main_loop, None);
    assert_eq!(a.cycles(&[Class::Matrix]).0, 0, "tinygrad's examples run no matrix op");
    assert!(a.text(5).contains("== main loop: none"));
}

/// `vectorops` loops back from `s_cbranch_scc0` at 0x2590 to 0x1afc; no matrix op, so the hottest loop.
#[test]
fn finds_the_main_loop() {
    let a = replayed("amd/navi4_sdata");
    let l = a.main_loop.clone().unwrap();
    assert_eq!((l.head, l.back_edge, l.matrix), (0x1afc, 0x2590, false));
    let (trips, cycles) = (l.iterations.unwrap(), l.cycles.unwrap());
    assert!(trips.min > 1, "{trips:?}");
    let edge_hits = a.pcs.iter().find(|r| r.pc == l.back_edge).unwrap().hits as usize;
    assert_eq!(cycles.n, edge_hits - trips.n, "one gap fewer than back edges a wave");
    assert!(cycles.min > 0 && cycles.p90 <= cycles.max);
    assert!(a.loop_rows().all(|r| (l.head..=l.back_edge).contains(&r.pc)) && a.loop_rows().count() > 1);
}

/// The text an agent reads, fixed: `SVOD_SQTT_BLESS=1` rewrites the expected file after a deliberate change.
#[test_case("amd/navi4_sdata", "sqtt_navi4_sdata.txt")]
#[test_case("svod/rdna4_gemm_v1", "sqtt_rdna4_gemm_v1.txt")]
#[test_case("svod/rdna4_gemm_1500", "sqtt_rdna4_gemm_1500.txt")]
fn text_is_stable(fixture: &str, golden: &str) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/test/fixtures").join(golden);
    let text = match fixture.strip_prefix("svod/") {
        Some(name) => Analysis::load(&fixtures().join("svod").join(name)).unwrap().text(8),
        None => replayed(fixture).text(8),
    };
    if std::env::var_os("SVOD_SQTT_BLESS").is_some() {
        std::fs::write(&path, &text).unwrap();
    }
    assert_eq!(text, std::fs::read_to_string(&path).unwrap());
}

/// Tier 5 renders one paragraph a traced kernel under the stage's report.
#[test]
fn profile_report_lists_traces() {
    let ok = crate::SqttKernel {
        index: 0,
        kernel: "_Z9vectoropsv".into(),
        dir: "sqtt/000__Z9vectoropsv".into(),
        analysis: Ok(replayed("amd/navi4_sdata")),
    };
    let failed = crate::SqttKernel {
        index: 1,
        kernel: "k".into(),
        dir: "sqtt/001_k".into(),
        analysis: Err(super::Error::NoCode.into()),
    };
    let mut stage = crate::StageProfile::host("profile", std::time::Duration::ZERO);
    stage.sqtt = vec![ok, failed];
    let report = crate::RunProfile { stages: vec![stage], origin_depth: None }.render_report();
    let lines: Vec<&str> = report.lines().collect();
    assert!(lines[1].starts_with("thread traces (SQTT"), "{report}");
    assert!(lines[2].starts_with("  #0 sqtt/000__Z9vectoropsv: _Z9vectoropsv gfx1200: 8 traced waves; time valu 51%"));
    assert!(lines[2].ends_with("main loop 16 iterations/wave at 7960 cycles (median)."), "{report}");
    assert!(lines[3].starts_with("  #1 k: SQTT: the trace carries no code image"), "{report}");
}

#[test]
fn flags_truncation() {
    let mut c = capture("amd/strix_barrier");
    c.trace.per_se[0].truncated = true;
    let a = c.analyze().unwrap();
    assert!(a.text(3).lines().nth(1).unwrap().starts_with("!! TRUNCATED se0"));
    assert!(a.summary().ends_with("TRUNCATED."));
}

#[test]
fn refuses_a_trace_without_code_or_itrace() {
    let mut c = capture("amd/strix_barrier");
    c.trace.per_se[0].itrace = false;
    assert!(matches!(c.analyze(), Err(super::Error::NoItrace)));
    c.trace.code = None;
    assert!(matches!(c.analyze(), Err(super::Error::NoCode)));
}

/// Where this machine has `llvm-objdump`, a loaded directory without `kernel.s` disassembles to the
/// fixture's text.
#[test]
fn disassembles_without_kernel_s() {
    let dir = tempfile::tempdir().unwrap();
    let c = capture("gfx1200/gemm");
    c.save(dir.path()).unwrap();
    std::fs::remove_file(dir.path().join("kernel.s")).unwrap();
    match Capture::load(dir.path()) {
        Ok(loaded) => assert_eq!(objdump_lines(&loaded.disassembly), objdump_lines(&c.disassembly)),
        Err(super::Error::Objdump { .. }) => eprintln!("no llvm-objdump: skipped"),
        Err(e) => panic!("{e}"),
    }
}

#[test_case("v_wmma_f32_16x16x16_bf16 v[0:7], v[8:15], v[16:23], v[0:7]", Class::Matrix)]
#[test_case("v_mfma_f32_32x32x8_f16 a[0:15], v[0:1], v[2:3], a[0:15]", Class::Matrix)]
#[test_case("v_add_f32_e32 v0, v1, v2", Class::Valu)]
#[test_case("ds_load_b128 v[0:3], v4 offset:16", Class::Lds)]
#[test_case("global_load_b128 v[0:3], v4, s[0:1]", Class::VmemLoad)]
#[test_case("buffer_store_b32 v0, off, s[0:3], 0", Class::VmemStore)]
#[test_case("s_load_b128 s[4:7], s[0:1], null", Class::Smem)]
#[test_case("s_cbranch_scc1 65530 // 0000000017B8: BFA2FFFA", Class::Branch)]
#[test_case("s_waitcnt vmcnt(0) lgkmcnt(0)", Class::WaitVmem)]
#[test_case("s_waitcnt lgkmcnt(0)", Class::WaitLgkm)]
#[test_case("s_wait_loadcnt_dscnt 0x0", Class::WaitVmem)]
#[test_case("s_wait_dscnt 0x0", Class::WaitLgkm)]
#[test_case("s_waitcnt_depctr 0xfffe", Class::Immediate)]
#[test_case("s_delay_alu instid0(VALU_DEP_1)", Class::Immediate)]
#[test_case("s_waitcnt_vscnt null, 0x0", Class::WaitVmem)]
#[test_case("s_wait_kmcnt 0x0", Class::WaitLgkm)]
#[test_case("s_wait_expcnt 0x0", Class::Immediate)]
#[test_case("s_wait_idle", Class::Immediate)]
#[test_case("s_wait_alu depctr_va_sdst(0)", Class::Immediate)]
#[test_case("image_bvh_intersect_ray v[0:3], v[0:10], s[0:3]", Class::VmemLoad)]
#[test_case("global_atomic_add_u32 v0, v1, v2, s[0:1] glc", Class::VmemStore)]
#[test_case("s_memrealtime s[0:1]", Class::Smem)]
#[test_case("s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)", Class::Message)]
#[test_case("s_barrier_wait 0xffff", Class::Barrier)]
#[test_case("s_endpgm", Class::Message)]
#[test_case("s_mov_b32 s0, 0", Class::Salu)]
#[test_case("", Class::Other)]
fn classifies(asm: &str, class: Class) {
    assert_eq!(Class::of(asm), class);
}

proptest! {
    #[test]
    fn spread_is_ordered(samples in prop::collection::vec(any::<i32>(), 0..64)) {
        let samples: Vec<i64> = samples.into_iter().map(i64::from).collect();
        match Spread::of(samples.clone()) {
            None => prop_assert!(samples.is_empty()),
            Some(s) => {
                prop_assert_eq!(s.n, samples.len());
                prop_assert!(s.min <= s.median && s.median <= s.p90 && s.p90 <= s.max);
                prop_assert_eq!((s.min, s.max), (*samples.iter().min().unwrap(), *samples.iter().max().unwrap()));
            }
        }
    }

    /// Any bytes after a real header analyze without panicking, and the class totals still add up.
    #[test]
    fn garbage_streams_keep_the_invariants(tail in prop::collection::vec(any::<u8>(), 0..2048)) {
        let mut c = capture("gfx1200/gemm");
        for se in &mut c.trace.per_se {
            se.bytes.truncate(64);
            se.bytes.extend_from_slice(&tail);
        }
        if let Ok(a) = c.analyze() {
            let total: i64 = a.pcs.iter().map(|r| r.latency).sum();
            prop_assert_eq!(a.classes.iter().map(|c| c.latency).sum::<i64>(), total);
            prop_assert!(a.stitched <= a.records);
            let _ = a.text(5);
        }
    }
}

/// One of svod's own captures with what a capture now records beside the trace: its kernel's
/// resources (read off `kernel.co`), the 8-wave workgroup of its `[4, 2]` warps and its grid.
fn svod_capture(name: &str, grid: [u32; 3]) -> Capture {
    let mut capture = Capture::load(&fixtures().join("svod").join(name)).unwrap();
    assert_eq!((capture.resources, capture.workgroup, capture.grid), (None, None, None), "older captures load");
    assert_eq!(capture.trace.mclk_mhz, None);
    let elf = &capture.trace.code.as_ref().unwrap().elf;
    let major = capture.trace.arch.gfx_major();
    capture.resources = Some(svod_device::amd::program::kernel_resources(elf, "gemm", major).unwrap());
    (capture.workgroup, capture.grid) = (Some([256, 1, 1]), Some(grid));
    capture
}

#[test]
fn capture_keeps_resources_launch_and_mclk() {
    let mut capture = svod_capture("rdna4_gemm_v1", [10, 6, 1]);
    capture.trace.mclk_mhz = Some(1800);
    let dir = tempfile::tempdir().unwrap();
    capture.save(dir.path()).unwrap();
    let back = Capture::load(dir.path()).unwrap();
    assert_eq!((back.resources, back.workgroup, back.grid), (capture.resources, capture.workgroup, capture.grid));
    assert_eq!(back.trace.mclk_mhz, Some(1800));
    let a = back.analyze().unwrap();
    assert_eq!(a, capture.analyze().unwrap());
    assert!(a.text(1).lines().any(|l| l.starts_with("kernel ") && l.contains("MHz, mclk 1800 MHz, pstate")));
}

/// The kernel's limit beside the waves the trace saw. v1 and v3 (256 VGPRs, 48 KiB LDS) and the
/// Strix GEMM (208 VGPRs) fit one 8-wave workgroup a CU, 4 waves a SIMD, and the traced SIMDs hold
/// that through the window. Strix's peak passes it: its waves free their VGPRs
/// (`MSG_DEALLOC_VGPRS`) before they end, so the next workgroup starts beside their tails.
#[test_case("rdna4_gemm_v1", [10, 6, 1], &[Limiter::Vgpr, Limiter::Lds], 4)]
#[test_case("rdna4_gemm_v3", [10, 6, 1], &[Limiter::Vgpr, Limiter::Lds], 4)]
#[test_case("strix_smoke", [10, 12, 1], &[Limiter::Vgpr], 8)]
fn occupancy_beside_the_limit(name: &str, grid: [u32; 3], by: &[Limiter], peak: u32) {
    let a = svod_capture(name, grid).analyze().unwrap();
    let occ = &a.occupancy;
    let limit = occ.limit.clone().expect("resources and workgroup known");
    assert_eq!((limit.waves_per_simd, limit.workgroups_per_cu, limit.by.as_slice()), (4, 1, by));
    assert_eq!(occ.ses.len(), a.ses.len(), "every SE's residency");
    for se in &occ.ses {
        let r = &se.residency;
        let s = a.ses.iter().find(|s| s.se == se.se).unwrap();
        assert!(r.window.0 < r.window.1 && r.mean() <= r.peak as f64 && r.peak as usize <= s.occupancy_peak);
        let window = (r.window.1 - r.window.0) as f64 / s.span as f64;
        assert!(window > 0.9, "{name} se{}: the robust peak's window spans the dispatch, {window:.2}", se.se);
        let simds: i64 = r.simds.iter().map(|s| s.wave_cycles).sum();
        assert_eq!(simds, r.wave_cycles, "the SIMDs' waves add up to the SE's");
        assert_eq!(se.busy.is_some(), se.traced.is_some());
        assert!(se.itrace || se.traced.is_none());
        if let Some(b) = se.busy {
            let parts = [b.matrix, b.valu, b.vmem, b.scalar];
            assert!(b.any <= b.cycles && parts.iter().all(|&c| (0..=b.any).contains(&c)), "{b:?}");
        }
        if let Some(t) = se.traced {
            assert!((t.mean() - 4.0).abs() < 0.05 && t.peak == peak, "{name} se{}: {t:?}", se.se);
        }
    }
    assert!(occ.ses.iter().any(|se| se.traced.is_some()));
    let text = a.text(4);
    assert!(
        text.contains("== occupancy") && text.contains("limit      4 waves/SIMD, 1 workgroups/CU, by vgpr"),
        "{text}"
    );
    assert_eq!(occ.untimed_matrix, a.arch.starts_with("gfx11"), "gfx11 traces WMMA as 1-cycle VALU tokens");
}

/// AMD's hidden latency folded into the rows: hidden stall and issue stay within the row's own.
/// Hidden idle is measured from the previous record's own end, not the latest end so far as `idle`
/// is, so it may pass `idle` where records overlap.
#[test]
fn rows_carry_hidden_latency() {
    let a = Analysis::load(&fixtures().join("svod/rdna4_gemm_v1")).unwrap();
    assert_eq!(a.occupancy.limit, None, "no resources recorded");
    assert!(a.text(4).contains("limit      - "));
    for r in &a.pcs {
        assert!(r.hidden_idle >= 0 && (0..=r.stall).contains(&r.hidden_stall), "{r:?}");
        assert!((0..=r.latency - r.stall).contains(&r.hidden_issue), "{r:?}");
    }
    assert!(a.pcs.iter().any(|r| r.hidden_stall > 0 && r.hidden_issue > 0));
}

fn verdict_of(name: &str) -> Analysis {
    Analysis::load(&fixtures().join("svod").join(name)).unwrap()
}

/// The VALU and SALU cycles a trip in the main loop, their stall included.
fn alu_per_trip(a: &Analysis) -> f64 {
    let l = a.main_loop.as_ref().unwrap();
    let alu = [Class::Valu, Class::Salu];
    (l.partition.issued(&alu) + l.partition.stalled(&alu)) as f64 / l.trips.total as f64
}

fn find(a: &Analysis, kind: FindingKind) -> Option<&Finding> {
    a.verdict.bound.iter().chain(&a.verdict.findings).find(|f| f.kind == kind)
}

fn kinds(findings: &[Finding]) -> Vec<FindingKind> {
    findings.iter().map(|f| f.kind).collect()
}

/// The exposed share of the loop's vector-memory waits and the stall before its loads issued.
fn load_share(a: &Analysis) -> f64 {
    let l = a.main_loop.as_ref().unwrap();
    let vmem = |on: WaitOn| on.counter == Counter::Load;
    let e = &l.exposed;
    (e.waited(vmem) + e.stalled(&[Class::VmemLoad])) as f64 / l.partition.cycles as f64
}

/// svod's ring GEMM, its 1500- and 1536-row twins and the Strix GEMM all run 160 K-trips of 32,
/// unrolled by 2. v1, v3 and Strix pinned the clock at the standard pstate and ran over 1.2x the
/// untraced time; the 1500 and 1536 captures ran unpinned at 1.04x.
#[test_case("rdna4_gemm_v1", Some(Pstate::Standard))]
#[test_case("rdna4_gemm_v3", Some(Pstate::Standard))]
#[test_case("rdna4_gemm_1500", Some(Pstate::Unpinned))]
#[test_case("rdna4_gemm_1536", Some(Pstate::Unpinned))]
#[test_case("strix_smoke", Some(Pstate::Standard))]
fn verdict_reads_the_capture(name: &str, pstate: Option<Pstate>) {
    let a = verdict_of(name);
    let t = a.main_loop.as_ref().unwrap().trips;
    assert_eq!((t.per_iteration, t.labelled), (2, true), "160 trips over 79 iterations and the head's");
    assert_eq!(a.pstate, pstate);
    let perturbed = pstate == Some(Pstate::Standard);
    let all = a.verdict.bound.iter().chain(&a.verdict.findings);
    assert!(all.clone().all(|f| f.approx.contains(&Approx::Perturbed) == perturbed), "{:?}", a.verdict);
    assert!(a.verdict.findings.iter().all(|f| f.share <= f.raw + 1e-12), "exposed within raw: {:?}", a.verdict);
    let l = a.main_loop.as_ref().unwrap();
    let signals: Vec<u64> = a.pcs.iter().filter(|r| r.asm.starts_with("s_barrier_signal")).map(|r| r.pc).collect();
    assert!(!l.barriers.is_empty());
    for s in &l.barriers {
        assert!(!signals.contains(&s.pc), "{name}: a signal is no wait: {s:?}");
        assert!(s.last <= s.hits && s.causes.iter().map(|c| c.1).sum::<usize>() <= s.last, "{s:?}");
        assert!(s.last > 0 && s.last * 2 < s.hits, "{name}: some traced wave arrives last, most don't: {s:?}");
    }
    let d = a.verdict.demand.clone().unwrap();
    assert!(d.bytes_per_trip > 0 && d.gbs > 0.0 && d.bandwidth_gbs.is_none(), "{d:?}");
}

/// v1 waits on its A/B fill after the trip's matrix work: its loads' latency is covered, and with
/// what other waves' pipes hid off, no loop finding passes a tenth of the loop. The barriers stay as
/// a sink, the matrix pipe is ~74% busy and the SIMDs idle ~16% of the window.
#[test]
fn v1_runs_near_matrix_throughput() {
    let a = verdict_of("rdna4_gemm_v1");
    let c = a.verdict.ceiling.clone().unwrap();
    assert!((0.70..0.80).contains(&c.matrix), "{c:?}");
    assert_eq!(kinds(&a.verdict.bound), [FindingKind::MatrixBound, FindingKind::OccupancyLimited]);
    let occupancy = find(&a, FindingKind::OccupancyLimited).unwrap();
    assert!((0.10..0.20).contains(&occupancy.share), "{occupancy:?}");
    assert!(a.verdict.findings.iter().all(|f| f.share < 0.10), "{:?}", a.verdict.findings);
    let barrier = find(&a, FindingKind::Barrier).unwrap();
    assert!(barrier.raw > 0.20 && barrier.share < 0.10, "a sink: {barrier:?}");
    assert!(load_share(&a) < 0.10 && find(&a, FindingKind::LoadExposed).is_none());
    let l = a.main_loop.as_ref().unwrap();
    for b in l.bindings.iter().filter(|b| b.pc == 0x9a0 || b.pc == 0x96c) {
        let latency = b.latency.unwrap().median;
        assert!(b.distance == [(0, b.count)] && b.cover.median * 10 > latency * 8, "mostly covered: {b:?}");
    }
}

/// v3's select pulls the ring fill's waits up to just after issue (cover ~nothing of a ~1000-cycle
/// latency): its exposed loads lead, and the knob names the wait and the select right after it.
#[test]
fn v3_waits_right_behind_its_load() {
    let (v1, v3) = (verdict_of("rdna4_gemm_v1"), verdict_of("rdna4_gemm_v3"));
    let top = &v3.verdict.findings[0];
    assert_eq!(top.kind, FindingKind::LoadExposed, "{:?}", v3.verdict.findings);
    assert!(top.share > 2.0 * load_share(&v1) && top.share < top.raw, "{} vs {}", top.share, load_share(&v1));
    let Evidence::Waits { loads, .. } = &top.evidence else { panic!("{top:?}") };
    let load = &loads[0];
    assert_eq!((load.pc, load.wait, load.stalled, load.waits), (0xa64, 0xab0, 316, 316));
    assert!(load.short_cover() && load.consumer.starts_with("v_cndmask_b32"), "{load:?}");
    assert!(top.knob.contains("0xab0 waits for 0xa64, then `v_cndmask_b32_e64 v147, 0, v147, s1`"), "{}", top.knob);
    assert!(!v3.verdict.findings[1..].iter().any(|f| f.share >= 0.10), "{:?}", v3.verdict.findings);
    let added = alu_per_trip(&v3) - alu_per_trip(&v1);
    assert!((260.0..300.0).contains(&added), "v3's guards: {added:.1} VALU+SALU cycles a trip");
}

/// The 1500-row GEMM guards its staged fill's rows: 8 exec branches in the loop, ~11 VALU ops a
/// trip of them. Its 1536-row control has no guard and keeps the matrix pipe ~90% busy.
#[test]
fn ragged_rows_are_loop_overhead() {
    let (ragged, control) = (verdict_of("rdna4_gemm_1500"), verdict_of("rdna4_gemm_1536"));
    let f = find(&ragged, FindingKind::LoopOverhead).unwrap();
    let Evidence::Overhead { divergent, valu_hits, .. } = &f.evidence else { panic!("{f:?}") };
    let execz: Vec<u64> = divergent.iter().filter(|d| d.1.starts_with("s_cbranch_execz")).map(|d| d.0).collect();
    assert_eq!((divergent.len(), execz), (8, vec![0x5f4, 0x624, 0x7e4, 0x810]), "{divergent:?}");
    assert!(*valu_hits > 5.0, "{valu_hits}");
    assert!(ragged.text(4).contains("   0x810 s_cbranch_execz 65290"));
    assert_eq!(kinds(&control.verdict.bound), [FindingKind::MatrixBound]);
    assert!((0.85..0.95).contains(&control.verdict.bound[0].share), "{:?}", control.verdict.bound);
    assert!(find(&control, FindingKind::LoopOverhead).is_none(), "{:?}", control.verdict.findings);
    assert!(control.verdict.findings.iter().all(|f| f.share < 0.10), "{:?}", control.verdict.findings);
}

/// gfx11 traces matrix ops as 1-cycle tokens: no ceiling or pipe-share class, raw shares flagged, and
/// the LDS ops' pre-issue stall (an untimed matrix op's cycles) is no LDS finding.
#[test]
fn strix_claims_nothing_of_its_matrix_pipe() {
    let a = verdict_of("strix_smoke");
    assert!(a.occupancy.untimed_matrix && a.verdict.ceiling.is_none());
    let bound = [FindingKind::MatrixBound, FindingKind::IssueBound];
    assert!(!a.verdict.bound.iter().any(|f| bound.contains(&f.kind)), "{:?}", a.verdict.bound);
    assert!(a.verdict.findings.iter().all(|f| f.approx.contains(&Approx::UntimedMatrix) && f.share == f.raw));
    let l = a.main_loop.as_ref().unwrap();
    assert!(l.partition.stalled(&[Class::Lds]) as f64 > 0.10 * l.partition.cycles as f64, "the stall is there");
    assert!(find(&a, FindingKind::LdsExposed).is_none(), "{:?}", a.verdict.findings);
    let overhead = find(&a, FindingKind::LoopOverhead).unwrap();
    let Evidence::Overhead { divergent, .. } = &overhead.evidence else { panic!("{overhead:?}") };
    assert!(divergent.iter().any(|(pc, asm)| *pc == 0x920 && asm.starts_with("s_cbranch_execz")), "{divergent:?}");
    assert!(a.text(4).contains("note       gfx11 traces matrix ops as 1-cycle tokens"));
}

#[test_case("global_load_b128 v[0:3], v4, s[0:1]", Some(16))]
#[test_case("global_load_b32 v0, v1, s[0:1]", Some(4))]
#[test_case("buffer_load_dwordx4 v[0:3], v4, s[0:3], 0 offen", Some(16))]
#[test_case("buffer_load_dword v0, v4, s[0:3], 0 offen", Some(4))]
#[test_case("global_load_u16 v0, v1, s[0:1]", Some(2))]
#[test_case("global_load_d16_hi_b16 v0, v1, s[0:1]", Some(2))]
#[test_case("global_load_i8 v0, v1, s[0:1]", Some(1))]
#[test_case("global_load_tr_b64 v[0:1], v2, s[0:1]", Some(8))]
#[test_case("s_nop 0", None)]
fn load_widths(asm: &str, bytes: Option<u64>) {
    assert_eq!(load_bytes(asm), bytes);
}

fn record(time: i64, duration: i64, stall: i64, pc: u64) -> svod_device::sqtt::wave::Instruction {
    svod_device::sqtt::wave::Instruction { time, category: svod_device::sqtt::Category::Valu, duration, stall, pc }
}

fn wave(instructions: Vec<svod_device::sqtt::wave::Instruction>) -> svod_device::sqtt::wave::Wave {
    svod_device::sqtt::wave::Wave {
        unit: svod_device::sqtt::Unit { sa: 0, wgp: 0, simd: 0, wave: 0 },
        me: 0,
        pipe: 0,
        begin: 0,
        end: 0,
        complete: true,
        contexts: 0,
        exclude_barrier_wait: false,
        instructions,
        pcs: vec![],
    }
}

const LOOP: [(u64, &str); 6] = [
    (0x0, "v_mov_b32 v0, 0"),
    (0x8, "ds_store_b128 v0, v[4:7]"),
    (0x10, "v_add_f32 v0, v1, v2"),
    (0x14, "s_wait_loadcnt_dscnt 0x0"),
    (0x18, "s_barrier_wait 0xffff"),
    (0x20, "s_cbranch_scc0 65530"),
];

fn partition(waves: &[svod_device::sqtt::wave::Wave], bound: &[HashMap<usize, (Counter, u64)>]) -> Partition {
    let hiding = Hiding::new(waves, &[], &HashSet::new());
    Buckets::new(&LOOP).partition(waves, bound, 0x10..=0x20, &hiding).raw
}

/// A loop at 0x10..=0x20 of a VALU op, a wait, a barrier: overlapping records count once, a stall
/// apart from its issue, gaps as idle on the record after them; records outside the loop don't
/// count, and a wave that leaves and re-enters the loop counts its runs inside alone. The wait
/// counts on the counter its binding was queued on, a store's apart.
#[test]
fn partition_counts_each_cycle_once() {
    let records = vec![
        record(0, 5, 0, 0x0),
        record(10, 6, 2, 0x10),
        record(14, 4, 4, 0x14),
        record(20, 10, 0, 0x18),
        record(25, 1, 0, 0x20),
        record(40, 3, 0, 0x8),
        record(100, 4, 0, 0x10),
        record(106, 4, 0, 0x14),
    ];
    let expect = |wait: WaitOn| Partition {
        waves: 1,
        cycles: 30,
        issue: vec![(Class::Valu, 8)],
        stall: vec![(Class::Valu, 2)],
        wait: vec![(wait, 6)],
        barrier: 10,
        idle: vec![(Class::WaitVmem, 2), (Class::Barrier, 2)],
    };
    let unbound = WaitOn { counter: Counter::Load, store: false };
    assert_eq!(partition(&[wave(records.clone())], &[]), expect(unbound), "unbound: its first counter");
    let flush = HashMap::from([(2, (Counter::Ds, 0x8)), (7, (Counter::Ds, 0x8))]);
    let p = partition(&[wave(records)], &[flush]);
    assert_eq!(p, expect(WaitOn { counter: Counter::Ds, store: true }), "a ds_store flush");
}

/// Another wave's VALU issue over a wave's wait and gap hides them: the exposed partition takes the
/// hidden cycles off, the raw keeps them.
#[test]
fn exposed_takes_off_what_other_waves_hid() {
    let wait = svod_device::sqtt::wave::Instruction {
        category: svod_device::sqtt::Category::Immed,
        ..record(20, 10, 9, 0x14)
    };
    let waiting = wave(vec![record(10, 2, 0, 0x10), wait, record(40, 1, 0, 0x20)]);
    let busy = wave((0..60).map(|t| record(t, 1, 0, 0x0)).collect());
    let waves = [waiting, busy];
    let hiding = Hiding::new(&waves, &[], &HashSet::new());
    let pass = Buckets::new(&LOOP).partition(&waves[..1], &[], 0x10..=0x20, &hiding);
    let (raw, exposed) = (pass.raw, pass.exposed);
    assert_eq!((raw.cycles, raw.waited(|_| true), raw.idled(&[])), (31, 10, 18));
    assert_eq!((exposed.waited(|_| true), exposed.idled(&[]), exposed.cycles), (0, 0, exposed.buckets()));
    assert!(exposed_within(&raw, &exposed));
}

/// A gfx12 barrier: the wave arrives at its signal and waits at the wait. A short wait is the last
/// arriver's, and the record that held the wave longest before its signal is why.
#[test]
fn barrier_names_the_last_arrivers_delay() {
    let lines = [
        (0x10, "v_add_f32 v0, v1, v2"),
        (0x14, "s_wait_dscnt 0x0"),
        (0x18, "s_barrier_signal -1"),
        (0x1c, "s_mov_b32 s0, 0"),
        (0x20, "s_barrier_wait 0xffff"),
        (0x24, "s_cbranch_scc0 65530"),
    ];
    let trip = |t: i64, wait: i64, barrier: i64| {
        vec![
            record(t, 1, 0, 0x10),
            record(t + 1, wait, wait - 1, 0x14),
            record(t + 1 + wait, 1, 0, 0x18),
            record(t + 30 + wait, 1, 0, 0x1c),
            record(t + 31 + wait, barrier, 0, 0x20),
            record(t + 31 + wait + barrier, 1, 0, 0x24),
        ]
    };
    let records = [trip(0, 50, 2), trip(200, 3, 300)].concat();
    let waves = [wave(records)];
    let hiding = Hiding::new(&waves, &[], &HashSet::new());
    let pass = Buckets::new(&lines).partition(&waves, &[], 0x10..=0x24, &hiding);
    let sites: Vec<_> = pass.barriers.into_values().collect();
    assert_eq!(sites, [super::BarrierSite { pc: 0x20, hits: 2, last: 1, causes: vec![(0x14, 1)] }]);
}

proptest! {
    /// Whatever the records, the buckets add up to the cycles of each wave's runs in the loop, from
    /// a run's first record to the latest end of its records, and every exposed bucket is within
    /// its raw one.
    #[test]
    fn partition_adds_up(
        waves in prop::collection::vec(
            prop::collection::vec((-20i64..200, 0i64..40, 0i64..40, 0usize..5), 1..40),
            1..4,
        ),
    ) {
        let lines = [
            (0x0, "v_mov_b32 v0, 0"),
            (0x10, "ds_load_b128 v[0:3], v4"),
            (0x14, "s_waitcnt vmcnt(0) lgkmcnt(0)"),
            (0x18, "s_barrier"),
            (0x20, "s_cbranch_scc0 65530"),
        ];
        let pcs = [0x0, 0x10, 0x14, 0x18, 0x20];
        let waves: Vec<_> = waves
            .iter()
            .map(|w| wave(w.iter().map(|&(t, d, s, pc)| record(t, d, s.min(d), pcs[pc])).collect()))
            .collect();
        let hiding = Hiding::new(&waves, &[], &HashSet::new());
        let pass = Buckets::new(&lines).partition(&waves, &[], 0x10..=0x20, &hiding);
        let (p, e) = (pass.raw, pass.exposed);
        prop_assert_eq!(p.buckets(), p.cycles);
        prop_assert_eq!(e.buckets(), e.cycles);
        prop_assert!(exposed_within(&p, &e));
        let expected: i64 = waves
            .iter()
            .flat_map(|w| w.instructions.chunk_by(|a, b| (a.pc != 0) == (b.pc != 0)).filter(|r| r[0].pc != 0))
            .map(|run| run.iter().map(|r| r.time + r.duration).max().unwrap().max(run[0].time) - run[0].time)
            .sum();
        prop_assert_eq!(p.cycles, expected);
    }
}

/// The class findings past their thresholds, from v1 with its pipes or residency moved: a busy matrix
/// pipe is matrix-bound on gfx12 alone, a busy SIMD mostly not on matrix work is issue-bound, and
/// idle SIMD cycles are occupancy-limited, its knob by what the capture says of the limit.
#[test]
fn class_findings_fire_past_their_thresholds() {
    use FindingKind::{IssueBound, MatrixBound, OccupancyLimited};
    let bound = |a: &Analysis| kinds(&super::verdict::rank(a).bound);
    let mut a = verdict_of("rdna4_gemm_v1");
    let busy = |a: &mut Analysis, matrix: i64, any: i64| {
        for b in a.occupancy.ses.iter_mut().filter_map(|s| s.busy.as_mut()) {
            (b.matrix, b.any) = (b.cycles * matrix / 100, b.cycles * any / 100);
        }
    };
    busy(&mut a, 69, 95);
    assert!(bound(&a).is_empty(), "under every threshold");
    busy(&mut a, 71, 95);
    assert_eq!(bound(&a), [MatrixBound]);
    a.occupancy.untimed_matrix = true;
    assert!(!bound(&a).contains(&MatrixBound), "gfx11 cannot tell");
    a.occupancy.untimed_matrix = false;
    busy(&mut a, 30, 95);
    assert_eq!(bound(&a), [IssueBound]);
    busy(&mut a, 30, 80);
    let f = super::verdict::rank(&a).bound.into_iter().find(|f| f.kind == OccupancyLimited).unwrap();
    let Evidence::Occupancy { observed, need, idle, limit: None } = f.evidence else { panic!("{f:?}") };
    assert!((f.share - 0.2).abs() < 0.01 && idle == Some(f.share) && (need - observed / 0.8).abs() < 0.01, "{f:?}");
    assert!(f.knob.contains("no resources"), "no limit known: {}", f.knob);
    let mut capped = svod_capture("rdna4_gemm_v1", [10, 6, 1]).analyze().unwrap();
    busy(&mut capped, 30, 80);
    let f = super::verdict::rank(&capped).bound.into_iter().find(|f| f.kind == OccupancyLimited).unwrap();
    assert!(f.knob.contains("resources cap the waves"), "4 of 4 waves: {}", f.knob);
}

fn compared(a: &str, b: &str) -> Diff {
    Diff::load(&fixtures().join("svod").join(a), &fixtures().join("svod").join(b)).unwrap()
}

fn bucket<'a>(d: &'a Diff, name: &str) -> &'a BucketDelta {
    d.buckets.iter().find(|b| b.bucket == name).unwrap_or_else(|| panic!("{name}: {:?}", d.buckets))
}

fn finding_delta(d: &Diff, kind: FindingKind) -> [Option<f64>; 2] {
    d.findings.iter().find(|f| f.kind == kind).map_or([None; 2], |f| f.share)
}

/// The same GEMM at the same pinned clock: v3's zero-select on loaded data pulls the ring fill's
/// wait up behind its load (+~740 load-wait cycles a trip, ~600 of them exposed) and inserts 34
/// `v_cndmask` rows (17 ops a trip, ~250 cycles a trip with their stall); load_exposed appears.
#[test]
fn compare_v1_v3_shows_the_select() {
    let d = compared("rdna4_gemm_v1", "rdna4_gemm_v3");
    assert_eq!(d.unit, Unit::Trip);
    assert!(d.cautions.is_empty(), "both standard, ratios 1.60x and 1.74x: {:?}", d.cautions);
    assert_eq!((d.a.kernel_ns, d.b.kernel_ns), (Some(175920), Some(203880)));
    let wait = bucket(&d, "wait load");
    assert!((280.0..300.0).contains(&wait.raw[0]) && (1020.0..1045.0).contains(&wait.raw[1]), "{wait:?}");
    assert!((720.0..760.0).contains(&wait.delta_raw()) && (580.0..620.0).contains(&wait.delta_exposed()), "{wait:?}");
    assert_eq!(d.buckets[0].bucket, "wait load", "the largest exposed delta leads");
    assert_eq!(d.buckets.last().unwrap().bucket, "total");
    let alu = ["issue valu", "stall valu", "issue salu"].map(|n| bucket(&d, n).delta_raw()).iter().sum::<f64>();
    assert!((260.0..300.0).contains(&alu), "v3's guards: {alu:.1} VALU+SALU cycles a trip");
    let cnd: Vec<&LoopRow> = d.rows.inserted.iter().filter(|r| r.asm.starts_with("v_cndmask_b32")).collect();
    let (ops, cycles) = cnd.iter().fold((0.0, 0.0), |(h, c), r| (h + r.hits, c + r.cycles));
    assert_eq!(cnd.len(), 34);
    assert!((ops - 17.0f64).abs() < 1e-9 && (250.0..300.0).contains(&cycles), "{ops} ops, {cycles:.1} cycles");
    let valu: f64 = d.rows.inserted.iter().filter(|r| r.class == Class::Valu).map(|r| r.hits).sum();
    assert!((30.0..40.0).contains(&valu), "v1's 5 VALU ops a trip grow to 40: {valu} inserted");
    assert!(!d.rows.matched.iter().any(|[a, _]| a.asm.starts_with("v_cndmask")), "v1's loop has no select");
    assert!(d.rows.matched.iter().all(|[a, b]| a.asm.split(' ').next() == b.asm.split(' ').next()));
    assert!(matches!(finding_delta(&d, FindingKind::LoadExposed), [None, Some(s)] if s > 0.10));
    assert!(matches!(finding_delta(&d, FindingKind::MatrixBound), [Some(_), None]));
    let busy = d.a.matrix_busy.zip(d.b.matrix_busy).unwrap();
    assert!(busy.1 < busy.0, "{busy:?}");
}

/// The 1500-row twin guards its ragged rows: exec-mask branches inserted in the loop, loop_overhead
/// appears. Both ran unpinned at ~1.04x: no caution.
#[test]
fn compare_1536_1500_shows_the_guards() {
    let d = compared("rdna4_gemm_1536", "rdna4_gemm_1500");
    assert!(d.cautions.is_empty(), "{:?}", d.cautions);
    let inserted = |m: &str| d.rows.inserted.iter().filter(|r| r.asm.starts_with(m)).count();
    assert_eq!((inserted("s_and_saveexec_b32"), inserted("s_cbranch_execz")), (4, 4));
    let removed = |m: &str| d.rows.removed.iter().filter(|r| r.asm.starts_with(m)).count();
    assert_eq!(removed("s_and_saveexec") + removed("s_cbranch_exec"), 0);
    assert!(matches!(finding_delta(&d, FindingKind::LoopOverhead), [None, Some(_)]));
    assert!(bucket(&d, "stall valu").delta_raw() > 200.0, "{:?}", bucket(&d, "stall valu"));
    let a = d.a.limit.as_ref().unwrap();
    assert_eq!((a.waves_per_simd, &a.by), (8, &d.b.limit.as_ref().unwrap().by));
}

/// A capture against itself: nothing inserted or removed, every delta zero.
#[test]
fn compare_with_itself_is_empty() {
    let d = compared("rdna4_gemm_v1", "rdna4_gemm_v1");
    assert!(d.rows.inserted.is_empty() && d.rows.removed.is_empty() && d.cautions.is_empty());
    assert!(d.rows.matched.iter().all(|[a, b]| a == b));
    assert!(d.buckets.iter().all(|b| b.delta_raw() == 0.0 && b.delta_exposed() == 0.0));
    assert!(d.findings.iter().all(|f| f.share[0] == f.share[1]));
}

/// Different clocks and perturbation, and trips over different K: each named in the header.
#[test]
fn compare_cautions() {
    let (v1, control) = (verdict_of("rdna4_gemm_v1"), verdict_of("rdna4_gemm_1536"));
    let d = v1.compare(&control);
    assert!(d.cautions.contains(&Caution::Pstate { a: Some(Pstate::Standard), b: Some(Pstate::Unpinned) }));
    assert!(d.cautions.iter().any(|c| matches!(c, Caution::Slowdown { .. })), "{:?}", d.cautions);
    let mut wide = control.clone();
    wide.labels.insert("model.trips".into(), "80".into());
    let d = control.compare(&wide);
    assert_eq!(d.cautions, [Caution::TripsDiffer { a: 32.0, b: 64.0 }]);
    assert!(d.text(4).contains("Δ/k exp"));
    let mut unlabelled = control.clone();
    unlabelled.main_loop.as_mut().unwrap().trips.labelled = false;
    let d = control.compare(&unlabelled);
    assert_eq!(d.unit, Unit::Iteration);
    assert_eq!(d.a.units * 2, control.main_loop.as_ref().unwrap().trips.total);
    let text = d.text(4);
    assert!(text.contains("per loop iteration") && text.contains("an iteration of each"), "{text}");
}

/// What an agent reads, fixed: `SVOD_SQTT_BLESS=1` rewrites it after a deliberate change.
#[test]
fn compare_text_is_stable() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/test/fixtures/sqtt_compare_v1_v3.txt");
    let text = compared("rdna4_gemm_v1", "rdna4_gemm_v3").text(6);
    if std::env::var_os("SVOD_SQTT_BLESS").is_some() {
        std::fs::write(&path, &text).unwrap();
    }
    assert_eq!(text, std::fs::read_to_string(&path).unwrap());
    let json: serde_json::Value = serde_json::from_str(&compared("rdna4_gemm_v1", "rdna4_gemm_v3").to_json()).unwrap();
    for key in ["a", "b", "unit", "cautions", "buckets", "findings", "rows"] {
        assert!(json.get(key).is_some(), "{key}");
    }
}

#[test_case(&[], &[], &[])]
#[test_case(&[1, 2, 3], &[1, 2, 3], &[(Some(0), Some(0)), (Some(1), Some(1)), (Some(2), Some(2))])]
#[test_case(&[1, 3], &[1, 2, 3], &[(Some(0), Some(0)), (None, Some(1)), (Some(1), Some(2))])]
#[test_case(&[1, 2, 3], &[1, 3], &[(Some(0), Some(0)), (Some(1), None), (Some(2), Some(1))])]
#[test_case(&[1, 2], &[3], &[(Some(0), None), (Some(1), None), (None, Some(0))])]
fn aligns(a: &[u8], b: &[u8], pairs: &[(Option<usize>, Option<usize>)]) {
    assert_eq!(align(a, b), pairs);
}

proptest! {
    /// An alignment walks both sequences once, in order, and pairs only equal items, as many as a
    /// longest common subsequence holds.
    #[test]
    fn alignment_is_a_longest_common_subsequence(
        a in prop::collection::vec(0u8..4, 0..40),
        b in prop::collection::vec(0u8..4, 0..40),
    ) {
        let pairs = align(&a, &b);
        let left: Vec<usize> = pairs.iter().filter_map(|p| p.0).collect();
        let right: Vec<usize> = pairs.iter().filter_map(|p| p.1).collect();
        prop_assert_eq!(left, (0..a.len()).collect::<Vec<_>>());
        prop_assert_eq!(right, (0..b.len()).collect::<Vec<_>>());
        let matched: Vec<(usize, usize)> = pairs.iter().filter_map(|&(i, j)| i.zip(j)).collect();
        prop_assert!(matched.iter().all(|&(i, j)| a[i] == b[j]));
        // The common subsequence's length by the textbook recurrence, forward.
        let mut lcs = vec![vec![0usize; b.len() + 1]; a.len() + 1];
        for i in 0..a.len() {
            for j in 0..b.len() {
                lcs[i + 1][j + 1] = if a[i] == b[j] { lcs[i][j] + 1 } else { lcs[i][j + 1].max(lcs[i + 1][j]) };
            }
        }
        prop_assert_eq!(matched.len(), lcs[a.len()][b.len()]);
    }
}
