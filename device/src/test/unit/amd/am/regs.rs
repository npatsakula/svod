use crate::amd::am::regs::*;

#[test]
fn selects_gfx1151_gc_module() {
    // Strix Halo reports GC 11.5.1 → the gc_11_5_0 table.
    let gc = select("gc", (11, 5, 1)).expect("gc module for gfx1151");
    assert!(!gc.is_empty());
    // RDNA4 reports GC 12.0.1 → the gc_12_0_0 table; a major with no vendored
    // module has no match.
    assert!(select("gc", (12, 0, 1)).is_some_and(|gc| find(gc, "regSQ_THREAD_TRACE_BUF0_BASE_HI").is_some()));
    assert!(select("gc", (10, 3, 0)).is_none());
}

#[test]
fn selects_gfx942_modules() {
    // MI300X discovery: GC 9.4.3, MMHUB 1.8.0, OSSSYS 4.4.2, SDMA 4.4.2,
    // NBIO 7.9.0, HDP 4.4.2, MP0 13.0.6 (resolved to 13.0.0), MP1 11.0.x.
    let gc = select("gc", (9, 4, 3)).expect("gc 9.4.3");
    assert!(select("gc", (9, 4, 2)).is_none(), "9.4.2 < oldest vendored 9.4.3");
    let hqd = find(gc, "regCP_HQD_ACTIVE").expect("CP_HQD_ACTIVE");
    assert_eq!(hqd.field("active").map(|f| (f.lo, f.hi)), Some((0, 0)));
    for (prefix, ver) in [
        ("mmhub", (1, 8, 0)),
        ("osssys", (4, 4, 2)),
        ("sdma", (4, 4, 2)),
        ("nbio", (7, 9, 0)),
        ("hdp", (4, 4, 2)),
        ("mp", (13, 0, 6)),
        ("mp", (11, 0, 2)),
    ] {
        assert!(select(prefix, ver).is_some(), "missing {prefix} {ver:?}");
    }
    // Aqua Vanjaram essentials resolve with expected fields.
    let inv = find(gc, "regVM_INVALIDATE_ENG17_REQ").or(find(gc, "regGCVM_INVALIDATE_ENG17_REQ"));
    assert!(inv.is_some(), "ENG17 invalidate register present");
    let sdma = select("sdma", (4, 4, 2)).unwrap();
    assert!(find(sdma, "regSDMA_GFX_RB_CNTL").is_some());
}

#[test]
fn resolves_known_register_and_fields() {
    let gc = select("gc", (11, 5, 1)).unwrap();
    let grbm = find(gc, "regGRBM_STATUS").expect("regGRBM_STATUS present");
    assert_eq!(grbm.offset, 3492);
    assert_eq!(grbm.segment, 0);
    let gui = grbm.field("gui_active").expect("gui_active field");
    assert_eq!((gui.lo, gui.hi), (31, 31));
}

#[test]
fn encode_decode_round_trip() {
    let gc = select("gc", (11, 5, 1)).unwrap();
    let grbm = find(gc, "regGRBM_STATUS").unwrap();
    let word = grbm.encode(&[("gui_active", 1), ("cp_busy", 1)]);
    assert_eq!(word, (1u64 << 31) | (1u64 << 29));
    assert_eq!(grbm.get(word, "gui_active"), 1);
    assert_eq!(grbm.get(word, "cp_busy"), 1);
    assert_eq!(grbm.get(word, "cb_busy"), 0);
}

#[test]
fn multi_bit_field_masks_correctly() {
    let gc = select("gc", (11, 5, 1)).unwrap();
    // regGRBM_STATUS.me0pipe0_cmdfifo_avail is a 4-bit field [0,3].
    let grbm = find(gc, "regGRBM_STATUS").unwrap();
    let word = grbm.encode(&[("me0pipe0_cmdfifo_avail", 0xF)]);
    assert_eq!(grbm.get(word, "me0pipe0_cmdfifo_avail"), 0xF);
    // High bits of an over-wide value don't bleed past the field on decode.
    assert_eq!(grbm.get(0xFF, "me0pipe0_cmdfifo_avail"), 0xF);
}
