use std::collections::BTreeMap;
use std::path::PathBuf;

use test_case::test_case;

use super::fixture::stitched;
use crate::sqtt::report::{Report, Stats};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/test/fixtures/sqtt/amd").join(name)
}

/// Splits a CSV row, keeping commas inside quotes.
fn csv_row(row: &str) -> Vec<String> {
    let (mut fields, mut field, mut quoted) = (vec![], String::new(), false);
    for c in row.chars() {
        match c {
            '"' => quoted = !quoted,
            ',' if !quoted => fields.push(std::mem::take(&mut field)),
            c => field.push(c),
        }
    }
    fields.push(field);
    fields
}

#[test_case("navi4_setpc")]
#[test_case("navi4_sdata")]
#[test_case("navi4_fifo")]
#[test_case("navi4_barriernop")]
#[test_case("strix_barrier")]
fn matches_amd(name: &str) {
    let s = stitched(&format!("amd/{name}"));
    let ours: BTreeMap<u64, Stats> =
        Report::new(&s.traces[0].waves).pcs.into_iter().map(|(pc, stats)| (pc - s.base, stats)).collect();
    let mut expected = BTreeMap::new();
    let text = |vaddr: u64| s.text(vaddr + s.base);
    let stats = std::fs::read_to_string(fixture(name).join("stats.csv")).unwrap();
    for row in stats.lines().skip(1) {
        let f = csv_row(row);
        if f[2].starts_with(';') || f[2].starts_with("label") {
            continue;
        }
        let n = |i: usize| f[i].parse::<i64>().unwrap();
        assert_eq!(f[0], "1");
        let vaddr = f[1].parse::<u64>().unwrap();
        let mnemonic = |l: &str| l.split_whitespace().next().unwrap_or_default().to_string();
        assert_eq!(mnemonic(f[2].as_str()), mnemonic(text(vaddr).unwrap()), "{name}: {vaddr:#x}");
        if n(3) != 0 || n(4) != 0 || n(5) != 0 || n(6) != 0 {
            expected.insert(vaddr, Stats { hits: n(3) as u64, latency: n(4), stall: n(5), idle: n(6) });
        }
    }
    assert!(!expected.is_empty());
    let mut diffs = vec![];
    for pc in expected.keys().chain(ours.keys()).collect::<std::collections::BTreeSet<_>>() {
        let (e, o) = (expected.get(pc), ours.get(pc));
        if e != o {
            diffs.push(format!("{pc:#x} {:?}: expected {e:?} ours {o:?}", text(*pc)));
        }
    }
    assert!(diffs.is_empty(), "{name}: {} of {} rows differ:\n{}", diffs.len(), expected.len(), diffs.join("\n"));
}
