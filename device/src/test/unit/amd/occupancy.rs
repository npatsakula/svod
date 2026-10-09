//! Unit tests for static kernel-resource + occupancy decoding.

use crate::amd::occupancy::{decode_resources, vgpr_limited_occupancy};

#[test]
fn decodes_vgpr_sgpr_segments_wave32() {
    // GRANULATED_WORKITEM_VGPR_COUNT = 24 → (24+1)*8 = 200 VGPRs (confirmed on
    // gfx1151). GRANULATED_WAVEFRONT_SGPR_COUNT = 1 → (1+1)*16 = 32 SGPRs.
    let rsrc1 = 24u32 | (1 << 6);
    let r = decode_resources(rsrc1, 4608, 0, 32, 11);
    assert_eq!(r.vgprs, Some(200));
    assert_eq!(r.sgprs, Some(32));
    assert_eq!(r.lds_bytes, 4608);
    assert_eq!(r.scratch_bytes, Some(0));
    assert_eq!(r.wave_size, 32);
}

/// Wave32 occupancy steps read off `llc -mcpu=gfx1151|gfx1201` (LLVM 20): the
/// 24-VGPR allocation granule of `FeatureGFX11FullVGPRs` puts the waves/SIMD
/// breaks at 96, 120 and 144 VGPRs. A 16-VGPR granule would give 16/13/12 here,
/// a 32-VGPR one 16/12/12.
#[test_case::test_case(96 => 16; "the last count that saturates the SIMD")]
#[test_case::test_case(97 => 12; "one past 96 rounds up to 120")]
#[test_case::test_case(121 => 10; "one past 120 rounds up to 144")]
fn wave32_vgpr_occupancy_matches_llc(vgprs: u32) -> u32 {
    let waves = |major| {
        let occ = vgpr_limited_occupancy(vgprs, 32, major).expect("gfx{major} geometry known");
        (occ * 16.0).round() as u32
    };
    let (gfx11, gfx12) = (waves(11), waves(12));
    assert_eq!(gfx11, gfx12, "gfx11 and gfx12 share the wave32 register file");
    gfx12
}

#[test]
fn vgpr_limited_occupancy_gfx11() {
    // 200 VGPRs on gfx11 wave32: round_up(200,24)=216, floor(1536/216)=7 waves,
    // 7/16 = 0.4375.
    let r = decode_resources(24, 0, 0, 32, 11);
    let occ = r.occupancy.expect("gfx11 occupancy known");
    assert!((occ - 7.0 / 16.0).abs() < 1e-6, "occ={occ}");

    // A tiny kernel (few VGPRs) saturates max waves → 100%.
    let small = decode_resources(0, 0, 0, 32, 11); // (0+1)*8 = 8 VGPRs
    assert_eq!(small.occupancy, Some(1.0));
}

#[test]
fn occupancy_unknown_arch_is_none() {
    // CDNA (gfx9) geometry not modeled → occupancy None, but counts still decode.
    let r = decode_resources(24, 0, 0, 64, 9);
    assert!(r.occupancy.is_none());
    assert_eq!(r.vgprs, Some((24 + 1) * 4)); // wave64 block = 4
}
