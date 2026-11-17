//! The classifier over each rule the fixtures' disassemblies reach, and every wait form.

use proptest::prelude::*;
use test_case::test_case;

use crate::sqtt::asm::{Access, Counter, Kind, Memory, Space, classify};

fn mem(space: Space, access: Access) -> Kind {
    Kind::Memory(Memory { space, access })
}

/// A fixture line for each rule the fixtures' `.s` files reach that [`other_mnemonics`] does not:
/// the prefix families, the `s_*`/`v_*` fallbacks once, and the near misses (`s_getpc`, `s_barrier_wait`).
#[test_case("ds_load_b128 v[1:4], v0", mem(Space::Ds, Access::Load))]
#[test_case("ds_store_b32 v5, v4", mem(Space::Ds, Access::Store))]
#[test_case("global_inv scope:SCOPE_SE", mem(Space::Global, Access::Invalidate))]
#[test_case("global_store_b32 v1, v0, s[0:1]", mem(Space::Global, Access::Store))]
#[test_case("s_barrier", Kind::Barrier)]
#[test_case("s_barrier_wait 0xffff", Kind::Barrier)]
#[test_case("s_branch 624", Kind::Branch)]
#[test_case("s_cbranch_execz 21", Kind::Branch)]
#[test_case("s_clause 0x1", Kind::Immediate)]
#[test_case("s_code_end", Kind::Salu)]
#[test_case("scratch_load_b32 v1, off, s32", mem(Space::Scratch, Access::Load))]
#[test_case("s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)", Kind::NoToken)]
#[test_case("s_endpgm", Kind::Message)]
#[test_case("s_getpc_b64 s[2:3]", Kind::Salu)]
#[test_case("s_load_b32 s0, s[0:1], 0xc", mem(Space::Smem, Access::Load))]
#[test_case("s_mov_b32 exec_lo, s0", Kind::Salu)]
#[test_case("s_nop 1", Kind::Immediate)]
#[test_case("s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)", Kind::SendMsg)]
#[test_case("s_setpc_b64 s[30:31]", Kind::Branch)]
#[test_case("s_sleep 2", Kind::Immediate)]
#[test_case("s_swappc_b64 s[30:31], s[2:3]", Kind::Branch)]
#[test_case("s_ttracedata", Kind::Message)]
#[test_case("s_wait_alu 0xfffe", Kind::NoToken)]
#[test_case("v_add_f32_e32 v0, v3, v0", Kind::Valu)]
#[test_case("v_wmma_f32_16x16x16_bf16 v[121:128], v[180:183], v[184:187], v[121:128]", Kind::Matrix)]
fn fixture_mnemonics(line: &str, kind: Kind) {
    let comment = format!("{line:<56}// 000000000BD0: BFA1FEB6");
    for line in [line, &comment] {
        let inst = classify(line);
        assert_eq!((inst.mnemonic, inst.kind), (line.split_whitespace().next().unwrap(), kind));
    }
}

#[test_case("buffer_load_b32 v0, off, s[0:3], 0", mem(Space::Buffer, Access::Load))]
#[test_case("tbuffer_store_format_x v0, off, s[0:3], 0", mem(Space::Buffer, Access::Store))]
#[test_case("buffer_gl0_inv", mem(Space::Buffer, Access::Invalidate))]
#[test_case("global_wbinv scope:SCOPE_SYS", mem(Space::Global, Access::Writeback))]
#[test_case("global_load_lds_dwordx4 v[0:1], off", mem(Space::Global, Access::Load))]
#[test_case("global_atomic_add_u32 v0, v1, s[0:1]", mem(Space::Global, Access::Atomic { returns: false }))]
#[test_case("global_atomic_add_u32 v0, v1, v2, s[0:1] glc", mem(Space::Global, Access::Atomic { returns: true }))]
#[test_case("global_atomic_add_u32 v0, v1, v2, s[0:1] th:TH_ATOMIC_RETURN", mem(Space::Global, Access::Atomic { returns: true }))]
#[test_case("global_prefetch_b8 v[0:1], off", mem(Space::Global, Access::Prefetch))]
#[test_case("flat_load_b32 v0, v[0:1]", mem(Space::Flat, Access::Load))]
#[test_case("flat_store_b32 v[0:1], v2", mem(Space::Flat, Access::Store))]
#[test_case("image_sample v[0:3], v0, s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_1D", mem(Space::Image, Access::Sample))]
#[test_case("image_gather4 v[0:3], v[0:1], s[0:7], s[8:11] dmask:0x1", mem(Space::Image, Access::Sample))]
#[test_case("image_msaa_load v[0:3], v[0:2], s[0:7] dmask:0x1", mem(Space::Image, Access::Sample))]
#[test_case("image_bvh_intersect_ray v[0:3], v[0:10], s[0:3]", mem(Space::Image, Access::Bvh))]
#[test_case("image_load v[0:3], v0, s[0:7] dmask:0xf", mem(Space::Image, Access::Load))]
#[test_case("ds_add_rtn_u32 v0, v1, v2", mem(Space::Ds, Access::Atomic { returns: true }))]
#[test_case("ds_bpermute_b32 v0, v1, v2", mem(Space::Ds, Access::Other))]
#[test_case("lds_param_load v0, attr0.x", mem(Space::LdsDirect, Access::Load))]
#[test_case("s_buffer_load_b32 s0, s[0:3], 0x0", mem(Space::Smem, Access::Load))]
#[test_case("s_store_dword s0, s[0:1], 0x0", mem(Space::Smem, Access::Store))]
#[test_case("s_atomic_add s0, s[0:1], 0x0 glc", mem(Space::Smem, Access::Atomic { returns: true }))]
#[test_case("s_dcache_inv", mem(Space::Smem, Access::Invalidate))]
#[test_case("s_memrealtime s[0:1]", mem(Space::Smem, Access::Load))]
#[test_case("s_prefetch_data s[0:1], 0x0, null, 0", mem(Space::Smem, Access::Prefetch))]
#[test_case("v_mfma_f32_32x32x8_f16 a[0:15], v[0:1], v[2:3], a[0:15]", Kind::Matrix)]
#[test_case("v_swmmac_f32_16x16x32_f16 v[0:7], v[8:11], v[12:19], v20", Kind::Matrix)]
#[test_case("v_dual_mov_b32 v121, 0 :: v_dual_and_b32 v4, 2, v2", Kind::Valu)]
#[test_case("s_call_b64 s[30:31], 16", Kind::Branch)]
#[test_case("s_rfe_b64 s[0:1]", Kind::Branch)]
#[test_case("s_waitcnt_depctr 0xfffe", Kind::NoToken)]
#[test_case("s_set_vgpr_msb 0x1", Kind::NoToken)]
#[test_case("s_wait_event 0x2", Kind::Immediate)]
#[test_case("s_wait_xcnt 0x0", Kind::Immediate)]
#[test_case("s_waitcnt vmcnt(bad)", Kind::Immediate)]
#[test_case("s_sendmsg_rtn_b32 s0, sendmsg(MSG_RTN_GET_DOORBELL)", Kind::SendMsg)]
#[test_case("s_trap 2", Kind::Message)]
#[test_case("s_setprio 3", Kind::Message)]
#[test_case("s_icache_inv", Kind::Message)]
#[test_case("s_set_inst_prefetch_distance 0x1", Kind::Immediate)]
#[test_case("exp mrt0 v0, v0, v0, v0 done", Kind::Other)]
#[test_case("", Kind::Other)]
fn other_mnemonics(line: &str, kind: Kind) {
    assert_eq!(classify(line).kind, kind);
}

/// Every wait form the fixtures hold, then gfx11's other forms and gfx12's combined ones.
#[test_case("s_waitcnt lgkmcnt(0)", &[(Counter::Lgkm, 0)])]
#[test_case("s_waitcnt lgkmcnt(16)", &[(Counter::Lgkm, 16)])]
#[test_case("s_waitcnt vmcnt(2)", &[(Counter::Load, 2)])]
#[test_case("s_wait_loadcnt 0x5", &[(Counter::Load, 5)])]
#[test_case("s_wait_storecnt 0x0", &[(Counter::Store, 0)])]
#[test_case("s_wait_dscnt 0xe", &[(Counter::Ds, 14)])]
#[test_case("s_wait_kmcnt 0x0", &[(Counter::Km, 0)])]
#[test_case("s_wait_samplecnt 0x0", &[(Counter::Sample, 0)])]
#[test_case("s_wait_bvhcnt 0x0", &[(Counter::Bvh, 0)])]
#[test_case("s_wait_expcnt 0x0", &[(Counter::Exp, 0)])]
#[test_case("s_wait_loadcnt_dscnt 0x0", &[(Counter::Load, 0), (Counter::Ds, 0)])]
#[test_case("s_wait_loadcnt_dscnt 0x305", &[(Counter::Load, 3), (Counter::Ds, 5)])]
#[test_case("s_wait_storecnt_dscnt 0x102", &[(Counter::Store, 1), (Counter::Ds, 2)])]
#[test_case("s_waitcnt vmcnt(0) lgkmcnt(0)", &[(Counter::Load, 0), (Counter::Lgkm, 0)])]
#[test_case("s_waitcnt vmcnt(63) expcnt(7) lgkmcnt(63)", &[(Counter::Load, 63), (Counter::Lgkm, 63), (Counter::Exp, 7)])]
#[test_case("s_waitcnt vmcnt(1) & lgkmcnt(2)", &[(Counter::Load, 1), (Counter::Lgkm, 2)])]
#[test_case("s_waitcnt vmcnt_sat(70)", &[(Counter::Load, 63)])]
#[test_case("s_waitcnt 0x0", &[(Counter::Load, 0), (Counter::Lgkm, 0), (Counter::Exp, 0)])]
#[test_case("s_waitcnt 0x1c17", &[(Counter::Load, 7), (Counter::Lgkm, 1), (Counter::Exp, 7)])]
#[test_case("s_waitcnt_vscnt null, 0x0", &[(Counter::Store, 0)])]
#[test_case("s_waitcnt_vmcnt null, 0x3", &[(Counter::Load, 3)])]
#[test_case("s_waitcnt_lgkmcnt null, 1", &[(Counter::Lgkm, 1)])]
#[test_case("s_waitcnt_expcnt null, 0x0", &[(Counter::Exp, 0)])]
fn wait_forms(line: &str, counts: &[(Counter, u8)]) {
    let Kind::Wait(wait) = classify(line).kind else { panic!("{line}: {:?}", classify(line)) };
    let mut expected = counts.to_vec();
    expected.sort();
    assert_eq!(wait.counts().collect::<Vec<_>>(), expected);
    assert!(!wait.idle);
}

#[test]
fn wait_idle_waits_on_every_counter() {
    let Kind::Wait(wait) = classify("s_wait_idle").kind else { panic!() };
    assert!(wait.idle && wait.counts().all(|(_, n)| n == 0) && wait.counts().count() == Counter::ALL.len());
}

proptest! {
    #[test]
    fn never_panics(line in ".*", prefix in prop::sample::select(vec!["", "s_waitcnt ", "s_wait_", "s_waitcnt_", "global_", "ds_", "s_load"])) {
        let line = format!("{prefix}{line}");
        let inst = classify(&line);
        prop_assert!(line.contains(inst.mnemonic));
    }
}
