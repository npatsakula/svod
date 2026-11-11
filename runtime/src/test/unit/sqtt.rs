use std::collections::BTreeMap;
use std::path::PathBuf;

use proptest::prelude::*;
use svod_device::profile::{CodeImage, SeTrace, SqttTrace};
use svod_device::sqtt::stitch::objdump_lines;
use svod_dtype::AmdArch;
use test_case::test_case;

use svod_device::amd::occupancy::Limiter;

use super::verdict::Buckets;
use super::{Analysis, Approx, Capture, Class, Evidence, FindingKind, Partition, Pstate, Spread};

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
    let trace = SqttTrace { arch: arch(&disassembly), per_se, code };
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
    assert!(a.verdict.windows(2).all(|w| w[0].share >= w[1].share), "largest share first");
    assert!(text.lines().nth(1).unwrap().starts_with("== verdict"), "the verdict leads: {text}");
    if let Some(l) = &a.main_loop {
        let p = &l.partition;
        assert_eq!(p.buckets(), p.cycles, "{fixture}: every loop cycle in one bucket");
        let mut sum = Partition::default();
        l.ses.iter().for_each(|s| sum.add(&s.partition));
        assert_eq!(&sum, p, "the SEs' partitions add up");
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
    let elf = &capture.trace.code.as_ref().unwrap().elf;
    let major = capture.trace.arch.gfx_major();
    capture.resources = Some(svod_device::amd::program::kernel_resources(elf, "gemm", major).unwrap());
    (capture.workgroup, capture.grid) = (Some([256, 1, 1]), Some(grid));
    capture
}

#[test]
fn capture_keeps_resources_and_launch() {
    let capture = svod_capture("rdna4_gemm_v1", [10, 6, 1]);
    let dir = tempfile::tempdir().unwrap();
    capture.save(dir.path()).unwrap();
    let back = Capture::load(dir.path()).unwrap();
    assert_eq!((back.resources, back.workgroup, back.grid), (capture.resources, capture.workgroup, capture.grid));
    assert_eq!(back.analyze().unwrap(), capture.analyze().unwrap());
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

fn loads(a: &Analysis, kind: FindingKind) -> (f64, Vec<super::Load>) {
    let f = a.verdict.iter().find(|f| f.kind == kind).unwrap_or_else(|| panic!("{kind:?} in {:?}", a.verdict));
    let Evidence::Waits { loads, .. } = &f.evidence else { panic!("{f:?}") };
    (f.share, loads.clone())
}

/// svod's ring GEMM (160 K-trips of 32, unrolled by 2): v1 waits on its A/B fill after the trip's
/// matrix work, so the loads' latency is mostly covered and the barriers lead; v3's select pulls
/// the waits up to just after issue (cover ~nothing of a ~1000-cycle latency), so its exposed loads
/// lead and its guard arithmetic adds ~280 VALU cycles a trip. All three captures pinned the clock
/// at the standard pstate and ran over 1.2x the untraced time.
#[test]
fn verdict_names_the_ring_gemm_limiters() {
    let (v1, v3, strix) = (verdict_of("rdna4_gemm_v1"), verdict_of("rdna4_gemm_v3"), verdict_of("strix_smoke"));
    for a in [&v1, &v3, &strix] {
        let t = a.main_loop.as_ref().unwrap().trips;
        assert_eq!((t.per_iteration, t.labelled), (2, true), "160 trips over 79 iterations and the head's");
        assert_eq!(a.pstate, Some(Pstate::Standard));
        assert!(a.verdict.iter().all(|f| f.approx.contains(&Approx::Perturbed)), "{:?}", a.verdict);
    }
    assert!(matches!(v1.verdict[0].kind, FindingKind::Barrier | FindingKind::LoadExposed), "{:?}", v1.verdict);
    let (v1_share, v1_loads) = loads(&v1, FindingKind::LoadExposed);
    let pcs: Vec<u64> = v1_loads.iter().map(|l| l.pc).collect();
    assert_eq!(pcs[..2], [0x9a0, 0x96c], "the A/B fill's last loads");
    for l in &v1_loads {
        let latency = l.latency.unwrap().median;
        assert!(l.distance == [(0, l.waits)] && l.cover.median * 10 > latency * 8, "mostly covered: {l:?}");
    }
    let (v3_share, v3_loads) = loads(&v3, FindingKind::LoadExposed);
    assert_eq!(v3.verdict[0].kind, FindingKind::LoadExposed);
    assert!(v3_share > 2.0 * v1_share, "{v3_share} vs {v1_share}");
    let top = &v3_loads[0];
    assert_eq!((top.pc, top.stalled, top.waits), (0xa64, 316, 316));
    assert!(top.cover.median * 10 < top.latency.unwrap().median, "waited right after issue: {top:?}");
    let added = alu_per_trip(&v3) - alu_per_trip(&v1);
    assert!((260.0..300.0).contains(&added), "v3's guards: {added:.1} VALU+SALU cycles a trip");
    assert!(v3.verdict.iter().any(|f| f.kind == FindingKind::LoopOverhead));
    assert!(!v1.verdict.iter().any(|f| f.kind == FindingKind::LoopOverhead), "{:?}", v1.verdict);
    let bound = [FindingKind::MatrixBound, FindingKind::IssueBound];
    assert!(strix.occupancy.untimed_matrix && !strix.verdict.iter().any(|f| bound.contains(&f.kind)));
    let overhead = strix.verdict.iter().find(|f| f.kind == FindingKind::LoopOverhead).unwrap();
    let Evidence::Overhead { divergent, .. } = &overhead.evidence else { panic!("{overhead:?}") };
    assert!(divergent.iter().any(|(pc, asm)| *pc == 0x920 && asm.starts_with("s_cbranch_execz")), "{divergent:?}");
    assert!(strix.text(4).contains("note       gfx11 traces matrix ops as 1-cycle tokens"));
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

/// A loop at 0x10..=0x20 of a VALU op, a wait, a barrier: overlapping records count once, a stall
/// apart from its issue, gaps as idle; records before the loop and after its last record don't count.
#[test]
fn partition_counts_each_cycle_once() {
    let lines = [
        (0x0, "v_mov_b32 v0, 0"),
        (0x10, "v_add_f32 v0, v1, v2"),
        (0x14, "s_wait_loadcnt 0x0"),
        (0x18, "s_barrier_wait 0xffff"),
        (0x20, "s_cbranch_scc0 65530"),
    ];
    let records = vec![
        record(0, 5, 0, 0x0),
        record(10, 6, 2, 0x10),
        record(14, 4, 4, 0x14),
        record(20, 10, 0, 0x18),
        record(25, 1, 0, 0x20),
        record(40, 3, 0, 0x0),
    ];
    let p = Buckets::new(&lines).partition(&[wave(records)], 0x10..=0x20);
    let expect = Partition {
        waves: 1,
        cycles: 20,
        issue: vec![(Class::Valu, 4)],
        stall: vec![(Class::Valu, 2)],
        wait: vec![(svod_device::sqtt::asm::Counter::Load, 2)],
        barrier: 10,
        idle: 2,
    };
    assert_eq!(p, expect);
}

proptest! {
    /// Whatever the records, the buckets add up to the cycles from each wave's first loop record to
    /// the latest end of its records through its last.
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
        let p = Buckets::new(&lines).partition(&waves, 0x10..=0x20);
        prop_assert_eq!(p.buckets(), p.cycles);
        let expected: i64 = waves
            .iter()
            .filter_map(|w| {
                let inside = |i: &svod_device::sqtt::wave::Instruction| i.pc != 0;
                let (first, last) = (w.instructions.iter().position(inside)?, w.instructions.iter().rposition(inside)?);
                let records = &w.instructions[first..=last];
                Some(records.iter().map(|r| r.time + r.duration).max().unwrap().max(records[0].time) - records[0].time)
            })
            .sum();
        prop_assert_eq!(p.cycles, expected);
    }
}

/// The findings no fixture reaches, from v1 with its pipes or residency moved past their thresholds:
/// a busy matrix pipe is matrix-bound on gfx12 alone, a busy SIMD mostly not on matrix work is
/// issue-bound, and one wave a SIMD cannot hide the loop's stall.
#[test]
fn busy_and_occupancy_findings_fire_past_their_thresholds() {
    use FindingKind::{IssueBound, MatrixBound, OccupancyLimited};
    let kinds = |a: &Analysis| super::verdict::rank(a).iter().map(|f| f.kind).collect::<Vec<_>>();
    let mut a = verdict_of("rdna4_gemm_v1");
    assert!(!kinds(&a).iter().any(|k| [MatrixBound, IssueBound, OccupancyLimited].contains(k)), "{:?}", a.verdict);
    let busy = |a: &mut Analysis, matrix: i64, any: i64| {
        for b in a.occupancy.ses.iter_mut().filter_map(|s| s.busy.as_mut()) {
            (b.matrix, b.any) = (b.cycles * matrix / 100, b.cycles * any / 100);
        }
    };
    busy(&mut a, 90, 95);
    assert_eq!(kinds(&a)[0], MatrixBound);
    a.occupancy.untimed_matrix = true;
    assert!(!kinds(&a).contains(&MatrixBound), "gfx11 cannot tell");
    a.occupancy.untimed_matrix = false;
    busy(&mut a, 30, 95);
    assert!(kinds(&a).contains(&IssueBound) && !kinds(&a).contains(&MatrixBound));
    for t in a.occupancy.ses.iter_mut().filter_map(|s| s.traced.as_mut()) {
        t.wave_cycles = t.cycles;
    }
    let f = super::verdict::rank(&a).into_iter().find(|f| f.kind == OccupancyLimited).unwrap();
    let Evidence::Occupancy { observed, need, .. } = f.evidence else { panic!("{f:?}") };
    assert!(observed == 1.0 && need > 1.5 && f.share > 0.0 && f.approx == [Approx::Perturbed], "{f:?}");
    assert!(f.knob.contains("more workgroups"), "no limit known: {}", f.knob);
}
