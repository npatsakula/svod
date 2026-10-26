use rayon::prelude::*;

use crate::atoms::{Fill, Target};
use crate::sweep::targets;
use svod_dtype::ScalarDType;

/// The compile-probe oracle agrees with the hand tables on every target the
/// sweep digests, and the verdicts gate exactly: on a target whose toolchain
/// answers here, a disagreement is a bug in one of the two, and the flip the
/// planner applies ([`crate::oracle::gate`]) is the identity on it. Skipped
/// per vendor where its toolchain is absent (no ptxas, no amdgcn clang); the
/// assembles are subprocesses, so they fan out.
#[test]
fn oracle_verdicts_match_the_tables_and_gate() {
    let (cuda, amd) = (svod_runtime::cuda::has_ptxas(), svod_runtime::amd::has_amdgpu_target());
    if !cuda && !amd {
        eprintln!("skipped: neither ptxas nor an amdgcn clang is installed");
        return;
    }
    let rows: Vec<(String, &'static str, bool, Option<bool>)> = targets()
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

/// The flip the oracle's verdicts make: a deny gates its capability off
/// however the tables answer, a confirm gates a boolean on, `None` leaves
/// the tables' answer, and the verdicts of a toolchain that agrees with the
/// tables change nothing — for every target the sweep digests, every
/// capability it asks of it.
#[test]
fn oracle_verdicts_flip_the_target_gates() {
    for (name, target) in targets() {
        let asks = crate::oracle::tables(&target);
        assert!(!asks.is_empty(), "{name} asks nothing");
        let all = |verdict| asks.iter().map(|&(cap, _)| (cap, verdict)).collect::<Vec<_>>();
        // No toolchain: the tables stand.
        assert_eq!(crate::oracle::flipped(&target, &all(None)), target, "{name} moved with no oracle");
        // A toolchain that agrees: the identity.
        let agreed: Vec<_> = asks.clone().into_iter().map(|(cap, v)| (cap, Some(v))).collect();
        assert_eq!(crate::oracle::flipped(&target, &agreed), target, "{name} moved under agreement");
        // Every deny holds, whatever the table said. The vendors ask
        // different questions (CUDA gates its warpgroup core, AMD its WMMA);
        // a capability a vendor does not ask of stands as its table has it.
        let asks_of = |cap: &str| asks.iter().any(|&(c, _)| c == cap);
        let denied = crate::oracle::flipped(&target, &all(Some(false)));
        let off = |t: &Target| {
            (!asks_of("wgmma") || t.group_mma.is_empty())
                && (!asks_of("ldmatrix") || !t.ldmatrix)
                && (!asks_of("tr_load") || !t.tr_load)
                && (!asks_of("dot2_bf16") || !t.dot2_bf16)
                && (!asks_of("cp_async") || t.fill != Fill::CpAsync)
                && (!asks_of("wmma16") || t.geometry().is_none())
                && (!asks_of("mma_i8") || t.geometry_of(ScalarDType::Int8).is_none())
        };
        assert!(off(&denied), "{name} kept a denied capability: {denied:?}");
        // A confirm flips a boolean on; an atom the tables did not build
        // stays unbuilt (the mismatch is logged, not conjured).
        let confirmed = crate::oracle::flipped(&target, &all(Some(true)));
        assert!(confirmed.ldmatrix || !asks_of("ldmatrix"));
        assert!(confirmed.tr_load || !asks_of("tr_load"));
        assert!(confirmed.dot2_bf16 || !asks_of("dot2_bf16"));
        assert!(confirmed.fill == Fill::CpAsync || !asks_of("cp_async"), "{name} did not take a confirmed cp.async");
    }
}
