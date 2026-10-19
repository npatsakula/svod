//! Unit tests for static kernel-resource + occupancy decoding.

use crate::amd::occupancy::{decode_resources, vgpr_limited_occupancy};

#[test]
fn decodes_vgpr_sgpr_segments_wave32() {
    // GRANULATED_WORKITEM_VGPR_COUNT = 24 → (24+1)*8 = 200 VGPRs (confirmed on
    // gfx1151). GRANULATED_WAVEFRONT_SGPR_COUNT = 1 → (1+1)*16 = 32 SGPRs.
    let rsrc1 = 24u32 | (1 << 6);
    let r = decode_resources(rsrc1, 0, 4608, 0, 32, 11);
    assert_eq!(r.vgprs, Some(200));
    assert_eq!(r.agprs, None);
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
    let r = decode_resources(24, 0, 0, 0, 32, 11);
    let occ = r.occupancy.expect("gfx11 occupancy known");
    assert!((occ - 7.0 / 16.0).abs() < 1e-6, "occ={occ}");

    // A tiny kernel (few VGPRs) saturates max waves → 100%.
    let small = decode_resources(0, 0, 0, 0, 32, 11); // (0+1)*8 = 8 VGPRs
    assert_eq!(small.occupancy, Some(1.0));
}

#[test]
fn occupancy_unknown_arch_is_none() {
    // CDNA (gfx9) geometry not modeled → occupancy None, but counts still decode.
    let r = decode_resources(24, 49, 0, 0, 64, 9);
    assert!(r.occupancy.is_none());
    assert_eq!(r.vgprs, Some((24 + 1) * 8)); // gfx9's unified file: blocks of 8
    assert_eq!(r.agprs, Some(0)); // the accumulator offset (49+1)*4 = 200 sits at the end
}

/// gfx9's unified file splits at the rsrc3 accumulator offset: a kernel of
/// 256 vector and 256 accumulator registers encodes field 63 in both (read
/// off clang's `.amdhsa_next_free_vgpr 512` / `.amdhsa_accum_offset 256` on
/// gfx942); one using no accumulators sets the offset to its vector count.
#[test_case::test_case(63, 63 => (256, 256); "half and half")]
#[test_case::test_case(7, 15 => (64, 0); "no accumulators")]
#[test_case::test_case(13, 12 => (52, 60); "52 vector registers, 60 accumulators")]
fn gfx9_splits_the_unified_file_at_the_accumulator_offset(rsrc1: u32, rsrc3: u32) -> (u32, u32) {
    let r = decode_resources(rsrc1, rsrc3, 0, 0, 64, 9);
    (r.vgprs.unwrap(), r.agprs.unwrap())
}
