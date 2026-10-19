//! Planning and emitting for cards this machine has no device for: their
//! targets from the card table.

use test_case::test_case;

use crate::atoms::{CARDS, Target};
use crate::ops::config;

/// A card's target is the architecture's with the card's SMs and dies, and
/// its shared memory per block is what the device reports.
#[test_case("rtx3060", 28, 1, 99 << 10; "sm_86")]
#[test_case("h100-pcie", 114, 1, 227 << 10; "sm_90 pcie")]
#[test_case("h100-sxm", 132, 1, 227 << 10; "sm_90 sxm")]
#[test_case("rtx5070", 48, 1, 99 << 10; "sm_120 5070")]
#[test_case("rtx5090", 170, 1, 99 << 10; "sm_120 5090")]
#[test_case("mi300x", 304, 8, 64 << 10; "gfx942")]
#[test_case("mi355x", 256, 8, 160 << 10; "gfx950")]
#[test_case("rx9070xt", 64, 1, 64 << 10; "gfx1201")]
#[test_case("strix-halo", 40, 1, 64 << 10; "gfx1151")]
fn a_card_reports_its_facts(name: &str, sms: u32, xcds: u32, smem: usize) {
    let target = Target::card(name).expect("a listed card");
    assert_eq!((target.sms, target.xcds, target.smem_bytes), (Some(sms), xcds, smem));
    assert!(config::has_kernels(&target), "{name} has kernels");
}

#[test]
fn every_card_is_named_once() {
    let mut names: Vec<&str> = CARDS.iter().map(|c| c.0).collect();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), CARDS.len());
    assert_eq!(Target::card("no such card").map(|t| t.arch), None);
}
