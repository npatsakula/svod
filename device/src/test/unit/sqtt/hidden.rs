use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::PathBuf;

use proptest::prelude::*;
use test_case::test_case;

use super::fixture::stitched;
use crate::sqtt::hidden::{self, Hidden};
use crate::sqtt::wave::{Instruction, OtherSimd, Wave};
use crate::sqtt::{Category, Unit};

/// AMD's `hidden_latency.py` on the fixture's SE, `{scope: {ELF pc: [idle, stall, issue]}}`: the
/// decoder library's records through the Python package (`analyze_hidden_latency`).
fn golden(file: &str) -> BTreeMap<String, BTreeMap<u64, Hidden>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/test/fixtures/sqtt/hidden").join(file);
    let doc: BTreeMap<String, BTreeMap<String, [i64; 3]>> =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let pc = |hex: &str| u64::from_str_radix(hex.trim_start_matches("0x"), 16).unwrap();
    let hidden = |[idle, stall, issue]: [i64; 3]| Hidden { idle, stall, issue };
    doc.into_iter().map(|(scope, pcs)| (scope, pcs.into_iter().map(|(k, v)| (pc(&k), hidden(v))).collect())).collect()
}

#[test_case("amd/navi4_setpc", 0)]
#[test_case("amd/navi4_sdata", 0)]
#[test_case("amd/navi4_fifo", 0)]
#[test_case("amd/navi4_barriernop", 0)]
#[test_case("amd/strix_barrier", 0)]
#[test_case("svod/rdna4_gemm_v1", 0)]
#[test_case("svod/rdna4_gemm_v1", 1)]
#[test_case("svod/rdna4_gemm_v3", 0)]
#[test_case("svod/rdna4_gemm_v3", 1)]
#[test_case("svod/rdna4_gemm_1500", 0)]
#[test_case("svod/rdna4_gemm_1500", 1)]
#[test_case("svod/rdna4_gemm_1536", 0)]
#[test_case("svod/rdna4_gemm_1536", 1)]
#[test_case("svod/strix_smoke", 0)]
#[test_case("svod/strix_smoke", 1)]
fn matches_amd(name: &str, se: u32) {
    let s = stitched(name);
    let trace = &s.traces[s.ses.iter().position(|&x| x == se).unwrap()];
    let ours = hidden::analyze([(se, trace)], &hidden::matrix_pcs(&s.lines()), s.base);
    let ours: BTreeMap<String, BTreeMap<u64, Hidden>> =
        ours.by_scope.into_iter().map(|((se, simd), pcs)| (format!("se{se}_simd{simd}"), pcs)).collect();
    let expected = golden(&format!("{}_se{se}.json", name.replace('/', "_")));
    assert_eq!(ours.keys().collect::<Vec<_>>(), expected.keys().collect::<Vec<_>>(), "{name}: scopes");
    for (scope, want) in &expected {
        let got = &ours[scope];
        let pcs: BTreeSet<&u64> = want.keys().chain(got.keys()).collect();
        let diffs: Vec<String> = pcs
            .into_iter()
            .filter(|pc| want.get(pc) != got.get(pc))
            .map(|pc| format!("{pc:#x} {:?}: amd {:?} ours {:?}", s.text(pc + s.base), want.get(pc), got.get(pc)))
            .collect();
        assert!(diffs.is_empty(), "{name} {scope}: {} of {} differ:\n{}", diffs.len(), want.len(), diffs.join("\n"));
    }
}

/// The matrix PCs are the `v_wmma` lines: AMD's `v_mfma`/`v_smfma`/`v_wmma`/`v_swmma` prefixes.
#[test]
fn matrix_pcs_are_the_wmma_lines() {
    let s = stitched("svod/rdna4_gemm_v1");
    let matrix = hidden::matrix_pcs(&s.lines());
    let wmma: BTreeSet<u64> =
        s.lines().iter().filter(|(_, l)| l.trim_start().starts_with("v_wmma")).map(|l| l.0).collect();
    assert!(!wmma.is_empty());
    assert_eq!(matrix.into_iter().collect::<BTreeSet<_>>(), wmma);
}

fn category() -> impl Strategy<Value = Category> {
    prop::sample::select(vec![
        Category::Valu,
        Category::Salu,
        Category::Smem,
        Category::Vmem,
        Category::Lds,
        Category::Immed,
    ])
}

proptest! {
    /// Whatever the pipes, a record's hidden cycles are each within the record's own: its gap since
    /// the wave's previous record, its stall and its issue.
    #[test]
    fn hidden_is_bounded_by_the_record(
        waves in prop::collection::vec(prop::collection::vec((0i64..8, 0i64..6, 1i64..12, category(), any::<bool>()), 1..30), 1..4),
    ) {
        let matrix: HashSet<u64> = (0..1000).filter(|pc| pc % 3 == 0).map(|pc| 4 * pc).collect();
        let mut pc = 0;
        let waves: Vec<Wave> = waves
            .iter()
            .map(|records| {
                let mut time = 0;
                let instructions = records
                    .iter()
                    .map(|&(gap, stall, cycles, category, wmma)| {
                        time += gap;
                        pc += if wmma { 12 - pc % 12 } else { 4 + 4 * u64::from(pc % 12 == 8) };
                        let inst = Instruction { time, category, duration: stall + cycles, stall, pc };
                        time += stall + cycles;
                        inst
                    })
                    .collect();
                Wave { unit: Unit::default(), me: 0, pipe: 0, begin: 0, end: time, complete: true, contexts: 0,
                    exclude_barrier_wait: false, instructions, pcs: vec![] }
            })
            .collect();
        let other = [OtherSimd { time: 3, category: Category::Vmem, cycles: 5 }];
        for wave in &waves {
            let alone = hidden::scope([wave], &other, &matrix);
            let mut last = wave.begin;
            for inst in &wave.instructions {
                let h = alone[&inst.pc];
                prop_assert!((0..=(inst.time - last).max(0)).contains(&h.idle), "{h:?} {inst:?}");
                prop_assert!((0..=inst.stall).contains(&h.stall), "{h:?} {inst:?}");
                prop_assert!((0..=inst.duration - inst.stall).contains(&h.issue), "{h:?} {inst:?}");
                last = inst.time + inst.duration;
            }
        }
        let all = hidden::scope(&waves, &other, &matrix);
        prop_assert_eq!(all.keys().collect::<Vec<_>>(), waves.iter().flat_map(|w| &w.instructions).map(|i| &i.pc).collect::<BTreeSet<_>>().into_iter().collect::<Vec<_>>());
    }
}
