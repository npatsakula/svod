//! Invariants every stitched fixture keeps: a record's category agrees with the instruction at its PC,
//! and VALU issue cycles fit the time they were issued in.

use std::collections::BTreeMap;

use test_case::test_case;

use super::fixture::{Stitched, stitched};
use crate::sqtt::wave::Instruction;
use crate::sqtt::{Category, wave};

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
        let mnemonic = s.text(inst.pc).map_or("?", |l| l.split_whitespace().next().unwrap_or("?"));
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
