//! Unit tests for static kernel-resource + occupancy decoding.

use svod_dtype::AmdArch;

use crate::amd::occupancy::{Cu, Limiter, decode_resources, limits, vgpr_limited_occupancy};

#[test]
fn decodes_vgpr_sgpr_segments_wave32() {
    // GRANULATED_WORKITEM_VGPR_COUNT = 24 → (24+1)*8 = 200 VGPRs (confirmed on
    // gfx1151). GRANULATED_WAVEFRONT_SGPR_COUNT = 1 → (1+1)*16 = 32 SGPRs.
    let rsrc1 = 24u32 | (1 << 6);
    let r = decode_resources(rsrc1, 0, 4608, 0, 32, AmdArch::Gfx1151);
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
    let waves = |arch| (vgpr_limited_occupancy(vgprs, 32, arch).expect("RDNA geometry known") * 16.0).round() as u32;
    let (gfx11, gfx12) = (waves(AmdArch::Gfx1151), waves(AmdArch::Gfx1201));
    assert_eq!(gfx11, gfx12, "gfx1151 and gfx1201 share the wave32 register file");
    gfx12
}

/// gfx1102's 1024-VGPR file in granules of 16 holds fewer waves than the 1536 of its siblings, and a
/// wave64 kernel has no modelled occupancy.
#[test_case::test_case(AmdArch::Gfx1102, 32 => Some(9); "97 VGPRs take 112 of 1024")]
#[test_case::test_case(AmdArch::Gfx1100, 32 => Some(12); "97 VGPRs take 120 of 1536")]
#[test_case::test_case(AmdArch::Gfx1100, 64 => None; "wave64")]
#[test_case::test_case(AmdArch::Gfx942, 32 => None; "CDNA")]
fn vgpr_occupancy_follows_the_arch(arch: AmdArch, wave_size: u32) -> Option<u32> {
    vgpr_limited_occupancy(97, wave_size, arch).map(|occ| (occ * 16.0).round() as u32)
}

#[test]
fn vgpr_limited_occupancy_gfx11() {
    // 200 VGPRs on gfx11 wave32: round_up(200,24)=216, floor(1536/216)=7 waves,
    // 7/16 = 0.4375.
    let r = decode_resources(24, 0, 0, 0, 32, AmdArch::Gfx1151);
    let occ = r.occupancy.expect("gfx11 occupancy known");
    assert!((occ - 7.0 / 16.0).abs() < 1e-6, "occ={occ}");

    // A tiny kernel (few VGPRs) saturates max waves → 100%.
    let small = decode_resources(0, 0, 0, 0, 32, AmdArch::Gfx1151); // (0+1)*8 = 8 VGPRs
    assert_eq!(small.occupancy, Some(1.0));
}

#[test]
fn occupancy_unknown_arch_is_none() {
    // CDNA (gfx9) geometry not modeled → occupancy None, but counts still decode.
    let r = decode_resources(24, 49, 0, 0, 64, AmdArch::Gfx942);
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
    let r = decode_resources(rsrc1, rsrc3, 0, 0, 64, AmdArch::Gfx942);
    (r.vgprs.unwrap(), r.agprs.unwrap())
}

fn resources(vgprs: Option<u32>, lds_bytes: u32, wave_size: u32) -> crate::profile::KernelResources {
    crate::profile::KernelResources {
        vgprs,
        agprs: None,
        sgprs: Some(32),
        lds_bytes,
        scratch_bytes: Some(0),
        wave_size,
        occupancy: None,
    }
}

/// A CU's workgroups at once and their waves a SIMD, by the first resource that runs out.
#[test_case::test_case(AmdArch::Gfx1201, 128, 256, 0 => Some((2, 8, vec![Limiter::Vgpr])); "128 VGPRs take 144: ten waves a SIMD")]
#[test_case::test_case(AmdArch::Gfx1201, 96, 256, 40000 => Some((1, 4, vec![Limiter::Lds])); "40000 LDS bytes take 40448")]
#[test_case::test_case(AmdArch::Gfx1201, 96, 256, 32768 => Some((2, 8, vec![Limiter::Lds])); "two halves of the CU's LDS")]
#[test_case::test_case(AmdArch::Gfx1201, 64, 64, 1024 => Some((16, 16, vec![Limiter::Workgroups, Limiter::Waves])); "barriers and slots alike")]
#[test_case::test_case(AmdArch::Gfx1151, 32, 32, 0 => Some((32, 16, vec![Limiter::Waves])); "a lone wave takes no barrier")]
#[test_case::test_case(AmdArch::Gfx1102, 128, 256, 0 => Some((2, 8, vec![Limiter::Vgpr])); "gfx1102's 1024-VGPR file")]
#[test_case::test_case(AmdArch::Gfx1100, 256, 512, 0 => Some((0, 0, vec![Limiter::Vgpr])); "sixteen waves of 264 do not fit")]
#[test_case::test_case(AmdArch::Gfx1201, 256, 256, 49152 => Some((1, 4, vec![Limiter::Vgpr, Limiter::Lds])); "v1: registers and LDS alike")]
#[test_case::test_case(AmdArch::Gfx1201, 200, 96, 0 => Some((4, 6, vec![Limiter::Vgpr])); "three-wave groups: seven a SIMD, six used")]
fn cu_limits(arch: AmdArch, vgprs: u32, threads: u32, lds: u32) -> Option<(u32, u32, Vec<Limiter>)> {
    let cu = Cu::rdna(arch).expect("RDNA");
    limits(&resources(Some(vgprs), lds, 32), threads, &cu).map(|l| (l.workgroups_per_cu, l.waves_per_simd, l.by))
}

#[test]
fn cu_limits_need_wave32_vgprs_and_threads() {
    let cu = Cu::rdna(AmdArch::Gfx1201).unwrap();
    assert_eq!(limits(&resources(None, 0, 32), 256, &cu), None);
    assert_eq!(limits(&resources(Some(64), 0, 64), 256, &cu), None);
    assert_eq!(limits(&resources(Some(64), 0, 32), 0, &cu), None);
    assert_eq!(Cu::rdna(AmdArch::Gfx942), None);
}

proptest::proptest! {
    /// The occupancy a descriptor reports is the CU's: a lone wave's workgroups fill a SIMD as far as
    /// its registers allow.
    #[test]
    fn vgpr_occupancy_is_the_cus(
        arch in proptest::sample::select(&[AmdArch::Gfx1100, AmdArch::Gfx1102, AmdArch::Gfx1151, AmdArch::Gfx1201][..]),
        vgprs in 1u32..=512,
    ) {
        let occupancy = vgpr_limited_occupancy(vgprs, 32, arch).unwrap();
        let l = limits(&resources(Some(vgprs), 0, 32), 32, &Cu::rdna(arch).unwrap()).unwrap();
        proptest::prop_assert_eq!((occupancy * 16.0).round() as u32, l.waves_per_simd);
    }

    /// The CU never holds more than any one resource allows, and holds as many as the limiter allows.
    #[test]
    fn cu_limits_respect_every_resource(vgprs in 1u32..=256, threads in 1u32..=1024, lds in 0u32..=65536) {
        let cu = Cu::rdna(AmdArch::Gfx1201).unwrap();
        let l = limits(&resources(Some(vgprs), lds, 32), threads, &cu).unwrap();
        let waves = threads.div_ceil(32);
        let alloc = vgprs.next_multiple_of(24);
        proptest::prop_assert!(l.workgroups_per_cu * waves <= cu.simds * cu.waves_per_simd);
        proptest::prop_assert!(l.workgroups_per_cu * lds.next_multiple_of(512) <= cu.lds_bytes);
        proptest::prop_assert!(l.waves_per_simd * alloc <= cu.vgprs);
        proptest::prop_assert!(waves == 1 || l.workgroups_per_cu <= cu.barriers);
        proptest::prop_assert!(l.waves_per_simd <= cu.waves_per_simd);
    }
}
