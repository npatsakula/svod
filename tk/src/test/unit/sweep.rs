//! The refactor sweep's own logic: its digest, the comparison of two runs,
//! the numbering of repeated names, and that its kernel set holds every
//! family under unique names and renders whole on every target it lists.

use test_case::test_case;

use crate::ops::config::has_kernels;
use crate::sweep::{Diff, diff, digest, digested, families, targets, try_render, unique};

/// FNV-1a 64's published vectors: a digest stable across builds.
#[test_case(b"" => 0xcbf2_9ce4_8422_2325; "empty")]
#[test_case(b"a" => 0xaf63_dc4c_8601_ec8c; "a")]
#[test_case(b"foobar" => 0x8594_4171_f739_67e8; "foobar")]
fn the_digest_is_fnv_1a(text: &[u8]) -> u64 {
    digest(text)
}

/// Two runs compared by name: a changed digest, a name only in the first
/// and one only in the second, each listed in its run's order.
#[test]
fn two_runs_differ_by_name() {
    let run = |list: &[(&str, u64)]| list.iter().map(|&(n, d)| (n.to_string(), d)).collect::<Vec<_>>();
    let a = run(&[("gemm", 1), ("fa", 2), ("norm", 3), ("conv", 4)]);
    let b = run(&[("conv", 4), ("gemm", 9), ("heads", 5), ("fa", 2)]);
    let names = |list: &[&str]| list.iter().map(|n| n.to_string()).collect::<Vec<_>>();
    let want = Diff { changed: names(&["gemm"]), removed: names(&["norm"]), added: names(&["heads"]) };
    assert_eq!(diff(&a, &b), want);
    assert!(diff(&a, &a).is_empty());
}

/// A repeated name is numbered from its second launch on.
#[test]
fn repeated_names_are_numbered() {
    let target = crate::atoms::sm86();
    let launch = families(&target).swap_remove(0);
    let named = unique(vec![launch.clone(), launch.clone(), launch]);
    let names: Vec<&str> = named.iter().map(|l| l.0.as_str()).collect();
    let first = names[0];
    assert_eq!(names, [first.to_string(), format!("{first} #2"), format!("{first} #3")]);
}

/// The sweep lists ten targets tk has kernels for, and on each its kernel
/// set holds every family under names that tell its launches apart.
#[test]
fn every_target_digests_its_families_under_unique_names() {
    let targets = targets();
    assert_eq!(targets.len(), 10);
    for (name, target) in targets {
        assert!(has_kernels(&target), "{name}");
        let set: Vec<String> = digested(&target).into_iter().map(|l| l.0).collect();
        let distinct: std::collections::HashSet<&String> = set.iter().collect();
        assert_eq!(distinct.len(), set.len(), "{name}: a repeated name");
        for (family, _, _) in families(&target) {
            assert!(set.contains(&family), "{name}: {family} is not digested");
        }
    }
}

/// Every kernel the sweep digests renders on its target: a config the
/// sweep admits but its lowering refuses — the warpgroup attention tiles
/// over rows appended to a cache, which only gmem can feed — fails here,
/// not as a FAILED line the digest gate prints.
#[test]
fn every_target_renders_what_it_digests() {
    std::thread::scope(|s| {
        for (name, target) in targets() {
            s.spawn(move || {
                for (kname, prog, lowering) in digested(&target) {
                    if let Err(e) = try_render(prog, &lowering) {
                        panic!("{name} {kname}: {e}");
                    }
                }
            });
        }
    });
}
