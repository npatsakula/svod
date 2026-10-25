use svod_dtype::GpuArch;

use crate::atoms::Target;
use crate::ops::config::Facts;

/// The device's layer-0 facts equal the hand tables where they overlap: the
/// tables still drive the planner (phase B captures only), so a disagreement
/// is a finding about one of the two — reported loudly here, not absorbed.
#[test]
fn device_facts_match_the_hand_tables() {
    let device = svod_dtype::default_device::default_device();
    let Some(target) = super::device_target() else { return };
    let Some(facts) = Facts::for_device(&device) else {
        panic!("{} opens a planner but reports no facts", target.arch.target_name());
    };
    eprintln!("{facts:#?}");
    assert_eq!(facts.arch, target.arch, "the facts and the target name one architecture");
    assert_eq!(facts.sms, target.sms.expect("a device reports its SMs"), "the SM count the planner scales by");
    assert_eq!(facts.wave, target.wave, "the wave size the layouts tile over");
    let GpuArch::Cuda(c) = facts.arch else {
        // The AMD overlaps are the KFD topology's own (lds, XCCs); the tables
        // they replace land with the AMD facts commit.
        assert_eq!(facts.smem_per_block_optin, Some(target.smem_bytes as u64), "the LDS budget");
        assert_eq!(facts.xcds, target.xcds, "the dies a dispatch spreads over");
        return;
    };
    // The hand tables of `atoms::cuda` and `dtype`'s per-cc smem table.
    assert_eq!(facts.smem_per_block_optin, Some(target.smem_bytes as u64), "the admission smem budget");
    assert_eq!(
        facts.smem_per_block_optin,
        Some(c.max_shared_per_block_optin() as u64),
        "dtype's per-cc smem table for {c}"
    );
    let reserved = facts.smem_reserved.expect("a driver reserved-smem answer");
    assert!(reserved > 0 && reserved < facts.smem_per_block_optin.unwrap(), "the reservation out of the opt-in");
    let o = target.occupancy;
    assert_eq!(
        facts.registers_per_sm,
        Some(o.file * o.simds * target.wave),
        "the register file the lane budget divides"
    );
    assert_eq!(
        facts.threads_per_sm,
        Some(o.waves * o.simds * target.wave),
        "the resident threads the occupancy's waves model"
    );
}

/// An offline target's facts are the tables' own, marked as such: nothing
/// pretends to have queried a card this machine has not.
#[test]
fn offline_facts_say_what_the_tables_know() {
    let target = Target::card("h100-sxm").expect("a listed card");
    let facts = Facts::of(&target);
    assert!(facts.table, "an offline target's facts are its tables'");
    assert_eq!(facts.smem_per_block_optin, Some(227 << 10), "the sm_90 smem table");
    assert_eq!(facts.l2_bytes, None, "no query answers an offline L2");
}
