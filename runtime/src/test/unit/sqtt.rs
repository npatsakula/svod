use std::collections::BTreeMap;
use std::path::PathBuf;

use proptest::prelude::*;
use svod_device::profile::{CodeImage, SeTrace, SqttTrace};
use svod_device::sqtt::stitch::objdump_lines;
use svod_dtype::AmdArch;
use test_case::test_case;

use super::{Analysis, Capture, Class, Spread};

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
#[test]
fn text_is_stable() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/test/fixtures/sqtt_navi4_sdata.txt");
    let text = replayed("amd/navi4_sdata").text(8);
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
