use rayon::prelude::*;

/// The compile-probe oracle agrees with the hand tables on every target the
/// sweep digests — the phase-B gate: the tables still drive, and on a target
/// whose toolchain answers here, a disagreement is a bug in one of the two.
/// Skipped per vendor where its toolchain is absent (no ptxas, no amdgcn
/// clang); the assembles are subprocesses, so they fan out.
#[test]
fn oracle_verdicts_match_the_tables() {
    let (cuda, amd) = (svod_runtime::cuda::has_ptxas(), svod_runtime::amd::has_amdgpu_target());
    if !cuda && !amd {
        eprintln!("skipped: neither ptxas nor an amdgcn clang is installed");
        return;
    }
    let rows: Vec<(String, &'static str, bool, Option<bool>)> = crate::sweep::targets()
        .into_par_iter()
        .flat_map_iter(|(name, target)| {
            // A `None` verdict is the vendor's toolchain being absent — that
            // vendor's targets are skipped, not held against the tables.
            crate::oracle::probed(&target)
                .into_iter()
                .filter(|(_, _, verdict)| verdict.is_some())
                .map(|(cap, table, verdict)| (name.to_string(), cap, table, verdict))
                .collect::<Vec<_>>()
        })
        .collect();
    assert!(!rows.is_empty(), "a toolchain is installed but no target asked anything");
    let mut mismatches = vec![];
    for (target, cap, table, verdict) in &rows {
        eprintln!("{target:8} {cap:10} table={table} oracle={verdict:?}");
        if verdict.is_some_and(|v| v != *table) {
            mismatches.push(format!("{target}'s {cap}: the tables say {table}, the toolchain {verdict:?}"));
        }
    }
    assert!(mismatches.is_empty(), "the oracle disagrees with the tables:\n{}", mismatches.join("\n"));
}
