use proptest::prelude::*;
use svod_dtype::AmdArch;

use super::targets::{amd, each, hopper, inputs, lowered, on_device};
use crate::atoms::{Fill, sm86};
use crate::interp;
use crate::lower::Error;
use crate::sweep::{families, group_families};
use crate::trace::{BARE, COUNT, Report, rows, traced};

/// The words of a trace of `cycles[line][warp]` (the tail first, the bare
/// stamps last) at `hits` stamps a word, over `blocks` blocks of `ns` each.
fn words(cycles: &[Vec<i64>], hits: i64, blocks: i64, ns: i64) -> Vec<i64> {
    let (bare, body) = cycles.split_last().unwrap();
    let word = |c: &i64, hits: i64| c * blocks + COUNT * hits * blocks;
    let row = |row: &Vec<i64>, hits: i64| row.iter().map(|c| word(c, hits)).collect::<Vec<_>>();
    let head = row(&body[0], 1);
    let timed = vec![ns * blocks + COUNT * blocks; bare.len()];
    [head, body[1..].iter().flat_map(|r| row(r, hits)).collect(), timed, row(bare, BARE as i64)].concat()
}

/// A block of two warps: its nanoseconds split as its cycles are, a stamp's
/// cost taken from the bare stamps and off every row.
#[test]
fn a_report_splits_the_block_as_the_cycles_and_nets_the_stamps() {
    // 1000 cycles a warp over 500 ns: half a nanosecond a cycle.
    let cycles = [vec![40, 40], vec![300, 100], vec![500, 700], vec![160, 160]];
    let report = Report::decode("load\n  … mma [steady]\n", 2, &words(&cycles, 4, 3, 500)).unwrap();
    assert_eq!((report.blocks, report.block_ns, report.tail_ns, report.stamp_ns), (3, 500.0, 20.0, 20.0));
    let [load, mma] = &report.rows[..] else { panic!("two rows: {report:?}") };
    assert_eq!((load.ns, load.least, load.most, load.hits), (100.0, 50.0, 150.0, 4.0));
    assert_eq!(
        (mma.ns, mma.text.as_str(), report.net(mma), report.net(load)),
        (300.0, "  … mma [steady]", 220.0, 20.0)
    );
    assert_eq!(Report::decode("load\n  … mma [steady]\n", 2, &vec![0; rows(2) * 2]), None, "no block sampled");
}

proptest! {
    /// A printed report reads back to the digits it prints, and against
    /// itself every statement stands still, the `n`th of a text beside the
    /// `n`th.
    #[test]
    fn a_printed_report_reads_back(
        cycles in prop::collection::vec(prop::collection::vec(1i64..1_000_000, 3), 3..9),
        (hits, blocks) in (1i64..64, 1i64..500),
    ) {
        let lines = cycles.len() - 2;
        let listing: String = (0..lines).map(|i| format!("{}copy {} [steady]\n", "  ".repeat(i % 3), i % 2)).collect();
        let report = Report::decode(&listing, 3, &words(&cycles, hits, blocks, 1 << 20)).unwrap();
        let read = Report::parse(&report.to_string()).unwrap();
        prop_assert_eq!((read.blocks, read.rows.len()), (report.blocks, report.rows.len()));
        prop_assert!((read.block_ns - report.block_ns).abs() <= 0.5 && (read.stamp_ns - report.stamp_ns).abs() <= 0.05);
        for (read, row) in read.rows.iter().zip(&report.rows) {
            let near = |a: f64, b: f64, by: f64| (a - b).abs() <= by;
            prop_assert_eq!(&read.text, &row.text);
            prop_assert!(near(read.ns, row.ns, 0.5) && near(read.least, row.least, 0.5) && near(read.most, row.most, 0.5));
            prop_assert!(near(read.hits, row.hits, 0.05));
        }
        let same = report.compare(&report);
        prop_assert_eq!(same.lines().count(), lines + 1);
        prop_assert!(same.lines().skip(1).all(|l| l.split('|').nth(3).unwrap().trim() == "+0"), "{}", same);
    }
}

/// Every NVIDIA family lowers traced, a stamp to each copy, sync and waiting
/// product and none without the flag; AMD has its thread trace.
#[test]
fn every_nvidia_family_lowers_traced() {
    let group = group_families(&hopper());
    each(families(&sm86()).into_iter().chain(families(&hopper())).chain(group).collect(), |name, prog, lowering| {
        let bare = lowered(prog.clone(), &lowering).unwrap();
        let ir = |l: &crate::lower::Lowered| l.text(&lowering.device.target).unwrap();
        assert!(!ir(&bare).contains("clock64"), "{name}");
        let t = traced(prog, &lowering).unwrap();
        assert_eq!(t.listing.lines().count(), bare.tile.walk().count(), "{name}");
        let stamped = ir(&lowered(t.prog, &t.lowering).unwrap());
        let own = t.listing.lines().filter(|l| !l.trim_start().starts_with('…')).count();
        // Open, close and the bare stamps beside the statements'.
        let reads = stamped.matches("call i64 @llvm.nvvm.read.ptx.sreg.clock64()").count();
        assert!(reads >= own.min(1) + 2 + BARE, "{name}: {reads} stamps over {own} statements");
    });
    let target = amd(AmdArch::Gfx1201, 64);
    let (_, prog, lowering) = families(&target).swap_remove(0);
    let mut t = traced(prog, &lowering).unwrap();
    t.lowering.device.target.fill = Fill::Staged;
    assert!(matches!(lowered(t.prog, &t.lowering), Err(Error::Unsupported { .. })));
}

/// On the device a traced family computes what the interpreter does, and
/// its trace accounts for the block: every sampled warp counted the same
/// blocks, the rows less their stamps within the block's time.
#[test]
fn traced_families_match_the_interpreter_and_account_for_their_blocks() {
    let Some(target) = super::device_target() else { return };
    if !matches!(target.arch, svod_dtype::GpuArch::Cuda(_)) {
        return eprintln!("skipped: the cycle trace is NVIDIA's");
    }
    let group = if target.group_mma.is_empty() { vec![] } else { group_families(&target) };
    for (name, prog, lowering) in families(&target).into_iter().chain(group) {
        let t = traced(prog.clone(), &lowering).unwrap();
        let mut params = inputs(&name, &prog);
        params.push(vec![0.0; t.prog.params.last().unwrap().elems]);
        let vars: Vec<(&str, i64)> = prog.vars.iter().map(|v| (v.name.as_str(), v.max)).collect();
        let want = interp::run(&prog, params[..params.len() - 1].to_vec(), &vars).unwrap();
        let mut got = on_device(t.prog.clone(), &t.lowering, &params);
        let words: Vec<i64> = got.pop().flatten().expect("the trace").iter().map(|&w| w as i64).collect();
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            let Some(g) = g else { continue };
            let close =
                |(g, w): (&f64, &f64)| g == w || g.is_nan() && w.is_nan() || (g - w).abs() <= 2e-2 * w.abs().max(1.0);
            assert!(g.iter().zip(w).all(close), "{name}: output {i} differs traced");
        }
        let warps = t.prog.warps as usize;
        assert!(words[..warps].iter().all(|w| w / COUNT == words[0] / COUNT), "{name}: warps sampled apart");
        // A grid whose every index sum misses the sample holds no report.
        let Some(report) = Report::decode(&t.listing, warps, &words) else { continue };
        let net: f64 = report.rows.iter().map(|r| report.net(r)).sum();
        assert!(net > 0.0 && net <= report.block_ns, "{name}: rows {net} ns of a {} ns block", report.block_ns);
    }
}
