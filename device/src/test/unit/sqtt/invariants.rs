//! Invariants every stitched fixture keeps: a record's category agrees with the instruction at its PC,
//! and VALU issue cycles fit the time they were issued in.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde_json::Value;
use test_case::test_case;

use crate::sqtt::stitch::{Stitcher, objdump_lines};
use crate::sqtt::wave::{Instruction, Trace};
use crate::sqtt::{Category, Layout, decode, wave};

/// One fixture's disassembly `(pc, line)` and the stitched traces of its instruction-traced SEs.
struct Stitched {
    code: BTreeMap<u64, String>,
    traces: Vec<Trace>,
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/test/fixtures/sqtt")
}

fn read(dir: &std::path::Path, file: &str) -> Vec<u8> {
    std::fs::read(dir.join(file)).unwrap_or_else(|e| panic!("{}: {e}", dir.join(file).display()))
}

/// `name`: `amd/<case>` (AMD's controls), `gfx1100|gfx1200/<example>` (tinygrad's) or `svod/<capture>`.
fn stitched(name: &str) -> Stitched {
    let dir = root().join(name);
    let text = |file: &str| String::from_utf8(read(&dir, file)).unwrap();
    let json = |file: &str| serde_json::from_str::<Value>(&text(file)).unwrap();
    let (base, dump, streams): (u64, String, Vec<Vec<u8>>) = match name.split_once('/').unwrap().0 {
        "amd" => {
            let base = u64::from_str_radix(text("base.txt").trim().trim_start_matches("0x"), 16).unwrap();
            (base, text("code_object.s"), vec![read(&dir, "se0.att")])
        }
        "svod" => {
            let meta = json("meta.json");
            let ses = meta["ses"].as_array().unwrap().iter().filter(|s| s["itrace"].as_bool().unwrap());
            let streams = ses.map(|s| read(&dir, &format!("se{}.att", s["se"]))).collect();
            (meta["load_base"].as_u64().unwrap(), text("kernel.s"), streams)
        }
        _ => {
            let golden = json("golden.json");
            let events = golden["events"].as_array().unwrap().iter().filter(|e| e["itrace"].as_bool().unwrap());
            let streams = events.map(|e| read(&dir, &format!("trace{}.bin", e["index"]))).collect();
            (golden["base"].as_u64().unwrap(), text("kernel.s"), streams)
        }
    };
    let code: Vec<(u64, &str)> = objdump_lines(&dump).into_iter().map(|(pc, line)| (pc + base, line)).collect();
    let traces = streams
        .iter()
        .map(|bytes| {
            let mut trace = wave::analyze(decode(bytes));
            let layout = if trace.header.version == 4 { Layout::Rdna4 } else { Layout::Rdna3 };
            let mut stitcher = Stitcher::new(&code, layout);
            trace.waves.iter_mut().for_each(|w| _ = stitcher.stitch(w));
            trace
        })
        .collect();
    let code = code.into_iter().map(|(pc, line)| (pc, line.split("//").next().unwrap().trim().to_string())).collect();
    Stitched { code, traces }
}

/// The categories a record attributed to an instruction may carry, by mnemonic: the stitcher's
/// trivial matches, with `Msg` for barriers and messages, and both branch outcomes.
fn categories(mnemonic: &str) -> &'static [Category] {
    use Category::*;
    const NO_TOKEN: [&str; 4] = ["s_delay_alu", "s_wait_alu", "s_wait_dep", "s_set_vgpr"];
    const IMMED: [&str; 13] = [
        "s_wait",
        "s_nop",
        "s_barrier",
        "s_sendmsg",
        "s_endpgm",
        "s_sleep",
        "s_clause",
        "s_setprio",
        "s_icache_inv",
        "s_trap",
        "s_set_inst",
        "s_inst_prefetch",
        "s_ttrace",
    ];
    const SMEM: [&str; 7] =
        ["s_load", "s_buffer_load", "s_prefetch", "s_dcache", "s_store", "s_atomic", "s_memrealtime"];
    const VMEM: [&str; 5] = ["global_", "buffer_", "scratch_", "tbuffer_", "image_"];
    let starts = |prefixes: &[&str]| prefixes.iter().any(|p| mnemonic.starts_with(p));
    if mnemonic.contains("branch") {
        &[Jump, Next]
    } else if mnemonic.starts_with("v_") {
        &[Valu]
    } else if starts(&NO_TOKEN) {
        &[]
    } else if starts(&IMMED) {
        &[Immed, Msg]
    } else if starts(&SMEM) {
        &[Smem]
    } else if starts(&VMEM) {
        &[Vmem]
    } else if mnemonic.starts_with("flat_") {
        &[Vmem, Flat]
    } else if mnemonic.starts_with("ds_") {
        &[Lds]
    } else if mnemonic.starts_with("s_") {
        &[Salu]
    } else {
        &[]
    }
}

/// `(record category, mnemonic) -> count` of stitched records the instruction at their PC disagrees with.
fn disagreements(s: &Stitched) -> BTreeMap<(Category, String), usize> {
    let mut out = BTreeMap::new();
    for inst in s.traces.iter().flat_map(|t| &t.waves).flat_map(|w| &w.instructions).filter(|i| i.pc != 0) {
        let mnemonic = s.code.get(&inst.pc).map_or("?", |l| l.split_whitespace().next().unwrap_or("?"));
        if inst.category != Category::Trap && !categories(mnemonic).contains(&inst.category) {
            *out.entry((inst.category, mnemonic.to_string())).or_default() += 1;
        }
    }
    out
}

/// Total length of the union of half-open intervals.
fn union(mut spans: Vec<(i64, i64)>) -> i64 {
    spans.sort_unstable();
    let (mut total, mut reach) = (0, i64::MIN);
    for (start, end) in spans {
        let start = start.max(reach);
        if end > start {
            total += end - start;
            reach = end;
        }
    }
    total
}

/// A VALU record's issue cycles: its duration after the stall.
fn issue(i: &Instruction) -> Option<(i64, i64)> {
    (i.category == Category::Valu).then_some((i.time + i.stall, i.time + i.duration))
}

/// AMD's stitcher lets an `IMMED` record match a memory line it would otherwise skip: every hit
/// `navi4_sdata` attributes to its `scratch_*` ops is such a record, as in AMD's `stats.csv`
/// ([`super::amd`] checks it per PC).
#[test_case("amd/navi4_setpc", 0)]
#[test_case("amd/navi4_sdata", 48)]
#[test_case("amd/navi4_fifo", 0)]
#[test_case("amd/navi4_barriernop", 0)]
#[test_case("amd/strix_barrier", 0)]
#[test_case("gfx1100/empty", 0)]
#[test_case("gfx1100/gemm", 0)]
#[test_case("gfx1100/plus", 0)]
#[test_case("gfx1200/empty", 0)]
#[test_case("gfx1200/gemm", 0)]
#[test_case("gfx1200/plus", 0)]
#[test_case("svod/rdna4_gemm_v1", 0)]
#[test_case("svod/rdna4_gemm_v3", 0)]
#[test_case("svod/strix_smoke", 0)]
fn records_agree_with_their_instruction(name: &str, expected: usize) {
    let s = stitched(name);
    let records = s.traces.iter().flat_map(|t| &t.waves).flat_map(|w| &w.instructions);
    assert!(records.clone().all(|i| i.pc != 0), "{name}: a record went unstitched");
    let found = disagreements(&s);
    assert_eq!(found.values().sum::<usize>(), expected, "{name}: {found:?}");
}

/// The union of a wave's VALU (and WMMA) issue cycles fits its lifetime, and their union over the
/// traced SIMD's waves fits the trace's. Their sum need not: a WMMA's table cycles overlap the records
/// issued after it.
#[test_case("amd/navi4_setpc")]
#[test_case("amd/navi4_sdata")]
#[test_case("amd/navi4_fifo")]
#[test_case("amd/navi4_barriernop")]
#[test_case("amd/strix_barrier")]
#[test_case("gfx1100/gemm")]
#[test_case("gfx1100/plus")]
#[test_case("gfx1200/gemm")]
#[test_case("gfx1200/plus")]
#[test_case("svod/rdna4_gemm_v1")]
#[test_case("svod/rdna4_gemm_v3")]
#[test_case("svod/strix_smoke")]
fn valu_issue_fits_its_span(name: &str) {
    let s = stitched(name);
    for (se, trace) in s.traces.iter().enumerate() {
        let issues = |w: &wave::Wave| w.instructions.iter().filter_map(issue).collect::<Vec<_>>();
        for w in &trace.waves {
            let (busy, span) = (union(issues(w)), w.end - w.begin);
            assert!(busy <= span, "{name} se{se} {:?}: VALU busy {busy} over a span of {span}", w.unit);
        }
        let begin = trace.waves.iter().map(|w| w.begin).min().unwrap_or(0);
        let end = trace.waves.iter().map(|w| w.end).max().unwrap_or(0);
        let busy = union(trace.waves.iter().flat_map(issues).collect());
        assert!(busy <= end - begin, "{name} se{se}: VALU busy {busy} over a span of {}", end - begin);
    }
}
