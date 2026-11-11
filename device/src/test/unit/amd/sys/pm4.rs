use crate::amd::sys::pm4::*;

#[test]
fn packet3_header_layout() {
    // PACKET3(RELEASE_MEM=0x49, 6) = (3 << 30) | (0x49 << 8) | (6 << 16)
    let hdr = packet3(PACKET3_RELEASE_MEM, 6);
    assert_eq!(hdr, (3 << 30) | (0x49 << 8) | (6 << 16));
}

#[test]
fn release_mem_packet_shape() {
    for is_gfx9 in [false, true] {
        let pkt = release_mem(0x1234_5678_0000_4000, 42, true, is_gfx9);
        assert_eq!(pkt.len(), 8);
        assert_eq!(pkt[3], 0x0000_4000); // addr lo
        assert_eq!(pkt[4], 0x1234_5678); // addr hi
        assert_eq!(pkt[5], 42); // value
        // memsel_dw must have DATA_SEL=1 in bits 29-31 (value 1 << 29)
        assert_eq!(pkt[2] & (0b111 << 29), 1 << 29);
        // INT_SEL=2 in bits 24-26
        assert_eq!((pkt[2] >> 24) & 0b111, 2);
    }
}

#[test]
fn release_mem_cache_flush_is_arch_specific() {
    // gfx9 uses the TC action bits; gfx10+ uses the GCR bitfield. The event
    // type/index (low bits of DW1) are shared.
    let event = release_mem_event_type(CACHE_FLUSH_AND_INV_TS_EVENT) | release_mem_event_index(EVENT_INDEX_END_OF_PIPE);
    let gfx9 = release_mem(0x4000, 1, true, true);
    let gfx10 = release_mem(0x4000, 1, true, false);
    assert_eq!(gfx9[1], event | EOP_CACHE_FLUSH_GFX9);
    assert_eq!(gfx10[1], event | RELEASE_MEM_CACHE_FLUSH_ALL);
    // gfx9 omits DST_SEL (memory implicit); gfx10+ sets it (DST_SEL_MEMORY=0,
    // so both happen to be equal here, but the data/int sel must match).
    assert_eq!(
        gfx9[2],
        release_mem_data_sel(DATA_SEL_SEND_32_BIT_LOW) | release_mem_int_sel(INT_SEL_INTERRUPT_AFTER_WRITE)
    );
}

#[test]
fn release_mem_order_writes_nothing_and_raises_no_interrupt() {
    let event = release_mem_event_type(CACHE_FLUSH_AND_INV_TS_EVENT) | release_mem_event_index(EVENT_INDEX_END_OF_PIPE);
    for is_gfx9 in [false, true] {
        let pkt = release_mem_order(is_gfx9);
        assert_eq!((pkt[0], pkt[1]), (packet3(PACKET3_RELEASE_MEM, 6), event));
        assert_eq!(pkt[2], 0, "no data, no interrupt, memory destination");
        assert_eq!(&pkt[3..], &[0; 5]);
    }
}

#[test]
fn release_mem_timestamp_is_a_bare_gpu_clock_probe() {
    // The timestamp probe must carry the GPU-clock data_sel, no interrupt, and —
    // despite the `CACHE_FLUSH_AND_INV_TS_EVENT` event NAME — NO cache action.
    let event = release_mem_event_type(CACHE_FLUSH_AND_INV_TS_EVENT) | release_mem_event_index(EVENT_INDEX_END_OF_PIPE);
    for is_gfx9 in [false, true] {
        let pkt = release_mem_timestamp(0x1234_5678_0000_4000, is_gfx9);
        assert_eq!(pkt.len(), 8);
        assert_eq!(pkt[0], packet3(PACKET3_RELEASE_MEM, 6));
        // DW1: exactly the EOP timestamp event, with no cache-flush bits OR'd in
        // (cf. `release_mem_cache_flush_is_arch_specific`, which DOES set them).
        assert_eq!(pkt[1], event);
        assert_eq!(pkt[1] & RELEASE_MEM_CACHE_FLUSH_ALL, 0);
        assert_eq!(pkt[1] & EOP_CACHE_FLUSH_GFX9, 0);
        // DW2: DATA_SEL = GPU clock (64-bit), and NO INT_SEL (no waiter to wake).
        assert_eq!((pkt[2] >> 29) & 0b111, DATA_SEL_SEND_GPU_CLOCK);
        assert_eq!((pkt[2] >> 24) & 0b111, 0, "timestamp probe must not request an interrupt");
        assert_eq!(pkt[3], 0x0000_4000); // addr lo
        assert_eq!(pkt[4], 0x1234_5678); // addr hi
        assert_eq!((pkt[5], pkt[6]), (0, 0), "value dwords ignored for gpu-clock data_sel");
    }
    // DST_SEL_MEMORY exists only on gfx10+ (it is 0, so invisible; assert the
    // gfx9 form simply omits it and otherwise matches).
    let gfx10 = release_mem_timestamp(0x4000, false);
    let gfx9 = release_mem_timestamp(0x4000, true);
    assert_eq!(gfx10[2], release_mem_data_sel(DATA_SEL_SEND_GPU_CLOCK) | release_mem_dst_sel(DST_SEL_MEMORY));
    assert_eq!(gfx9[2], release_mem_data_sel(DATA_SEL_SEND_GPU_CLOCK));
}

#[test]
fn acquire_mem_gfx9_shape() {
    let pkt = acquire_mem_gfx9();
    assert_eq!(pkt.len(), 7);
    assert_eq!(pkt[0], packet3(PACKET3_ACQUIRE_MEM, 5));
    assert_eq!(pkt[2], 0xFFFF_FFFF); // coher size lo (full VA)
    assert_eq!(pkt[3], 0xFFFF_FFFF); // coher size hi
    assert_eq!(pkt[6], 0x0000_000A); // poll interval
}

#[test]
fn acquire_mem_gfx9_narrow_skips_l2() {
    let full = acquire_mem_gfx9()[1];
    let narrow = acquire_mem_gfx9_narrow()[1];
    // Narrow keeps the per-CU caches (I-cache, K-cache, vector TCL1)…
    assert_ne!(narrow & COHER_SH_ICACHE_ACTION_ENA, 0);
    assert_ne!(narrow & COHER_SH_KCACHE_ACTION_ENA, 0);
    assert_ne!(narrow & COHER_TCL1_ACTION_ENA, 0);
    // …but drops the L2 (TC) invalidate + write-back that the EOP release does.
    assert_eq!(narrow & COHER_TC_ACTION_ENA, 0);
    assert_eq!(narrow & COHER_TC_WB_ACTION_ENA, 0);
    assert_ne!(full & COHER_TC_ACTION_ENA, 0, "full acquire must still invalidate L2");
}

#[test]
fn pred_exec_shape() {
    let pkt = pred_exec(0b1, 8);
    assert_eq!(pkt[0], packet3(PACKET3_PRED_EXEC, 0));
    assert_eq!(pkt[1] >> 24, 0b1); // xcc_mask
    assert_eq!(pkt[1] & 0x3FFF, 8); // exec dword count
}

#[test]
fn set_sh_reg_header_layout() {
    // Two values at COMPUTE_PGM_LO (= PGM_LO + PGM_HI in one packet).
    let pkt = set_sh_reg(COMPUTE_PGM_LO, &[0x1234_5678, 0x9abc_def0]);
    assert_eq!(pkt.len(), 4);
    // header: count = 2 (number of data dwords excluding header AND reg_offset)
    assert_eq!(pkt[0], packet3(PACKET3_SET_SH_REG, 2));
    // reg_offset takes the low 16 bits
    assert_eq!(pkt[1], COMPUTE_PGM_LO);
    assert_eq!(pkt[2], 0x1234_5678);
    assert_eq!(pkt[3], 0x9abc_def0);
}

#[test]
fn dispatch_direct_layout() {
    let di =
        DISPATCH_INITIATOR_FORCE_START_AT_000 | DISPATCH_INITIATOR_COMPUTE_SHADER_EN | DISPATCH_INITIATOR_CS_W32_EN;
    let pkt = dispatch_direct([4, 8, 16], di);
    // header + grid_x + grid_y + grid_z + dispatch_initiator = 5 dwords
    assert_eq!(pkt.len(), 5);
    assert_eq!(pkt[0], packet3(PACKET3_DISPATCH_DIRECT, 3));
    assert_eq!(pkt[1], 4);
    assert_eq!(pkt[2], 8);
    assert_eq!(pkt[3], 16);
    assert_eq!(pkt[4], di);
}

#[test]
fn event_write_partial_flush_shape() {
    let pkt = event_write(CS_PARTIAL_FLUSH, EVENT_INDEX_PARTIAL_FLUSH);
    // header + (event_type | (event_index << 8))
    assert_eq!(pkt[0], packet3(PACKET3_EVENT_WRITE, 0));
    assert_eq!(pkt[1], CS_PARTIAL_FLUSH | (EVENT_INDEX_PARTIAL_FLUSH << 8));
}
