use std::collections::BTreeMap;
use std::path::PathBuf;

use test_case::test_case;

use crate::sqtt::report::{Report, Stats};
use crate::sqtt::stitch::{Stitcher, objdump_lines};
use crate::sqtt::{Layout, decode, wave};

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
    let dir = fixture(name);
    let read = |file: &str| std::fs::read_to_string(dir.join(file)).unwrap();
    let base = u64::from_str_radix(read("base.txt").trim().trim_start_matches("0x"), 16).unwrap();
    let dump = read("code_object.s");
    let code: Vec<(u64, &str)> = objdump_lines(&dump).into_iter().map(|(pc, line)| (pc + base, line)).collect();
    let mut trace = wave::analyze(decode(&std::fs::read(dir.join("se0.att")).unwrap()));
    let layout = if trace.header.version == 4 { Layout::Rdna4 } else { Layout::Rdna3 };
    let mut stitcher = Stitcher::new(&code, layout);
    for wave in &mut trace.waves {
        let records = wave.instructions.len();
        assert_eq!(stitcher.stitch(wave), records, "{name}: a record went unstitched");
    }
    let ours: BTreeMap<u64, Stats> =
        Report::new(&trace.waves).pcs.into_iter().map(|(pc, stats)| (pc - base, stats)).collect();
    let mut expected = BTreeMap::new();
    let text: BTreeMap<u64, &str> = code.iter().map(|&(pc, line)| (pc - base, line)).collect();
    for row in read("stats.csv").lines().skip(1) {
        let f = csv_row(row);
        if f[2].starts_with(';') || f[2].starts_with("label") {
            continue;
        }
        let n = |i: usize| f[i].parse::<i64>().unwrap();
        assert_eq!(f[0], "1");
        let vaddr = f[1].parse::<u64>().unwrap();
        let mnemonic = |s: &str| s.split_whitespace().next().unwrap_or_default().to_string();
        assert_eq!(mnemonic(f[2].as_str()), mnemonic(text[&vaddr]), "{name}: {vaddr:#x}");
        if n(3) != 0 || n(4) != 0 || n(5) != 0 || n(6) != 0 {
            expected.insert(vaddr, Stats { hits: n(3) as u64, latency: n(4), stall: n(5), idle: n(6) });
        }
    }
    assert!(!expected.is_empty());
    let mut diffs = vec![];
    for pc in expected.keys().chain(ours.keys()).collect::<std::collections::BTreeSet<_>>() {
        let (e, o) = (expected.get(pc), ours.get(pc));
        if e != o {
            diffs.push(format!("{pc:#x} {:?}: expected {e:?} ours {o:?}", text.get(pc)));
        }
    }
    assert!(diffs.is_empty(), "{name}: {} of {} rows differ:\n{}", diffs.len(), expected.len(), diffs.join("\n"));
}
