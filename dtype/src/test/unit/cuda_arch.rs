use proptest::prelude::*;
use test_case::test_case;

use super::*;

#[test_case("sm_86", 8, 6)]
#[test_case("sm_70", 7, 0)]
#[test_case("sm_89", 8, 9)]
#[test_case("sm_100", 10, 0; "three digit blackwell")]
#[test_case("SM_120", 12, 0; "uppercase prefix")]
fn parse_yields_compute_capability(label: &str, major: u8, minor: u8) {
    let arch = CudaArch::from_compute_capability(major, minor);
    assert_eq!(label.parse::<CudaArch>(), Ok(arch));
    assert_eq!(arch.sm(), major as u32 * 10 + minor as u32);
    assert_eq!(arch.to_string(), label.to_ascii_lowercase());
}

#[test_case(""; "empty")]
#[test_case("sm_"; "no digits")]
#[test_case("sm86"; "missing underscore")]
#[test_case("sm_9x"; "non digit")]
#[test_case("sm_90a"; "feature suffix")]
#[test_case("sm_-86"; "sign")]
#[test_case("gfx1100"; "amd arch")]
#[test_case("sm_99999"; "major overflows u8")]
fn parse_rejects_malformed_labels(label: &str) {
    assert_eq!(label.parse::<CudaArch>(), Err(ParseCudaArchError(label.to_string())));
}

#[test_case(6, 1, false, false, false; "pascal")]
#[test_case(7, 0, true, false, false; "volta")]
#[test_case(7, 5, true, false, false; "turing")]
#[test_case(8, 0, true, true, false; "ampere a100")]
#[test_case(8, 6, true, true, false; "ampere ga10x")]
#[test_case(8, 9, true, true, true; "ada")]
#[test_case(9, 0, true, true, true; "hopper")]
#[test_case(10, 0, true, true, true; "blackwell")]
fn capability_thresholds(major: u8, minor: u8, tensor_cores: bool, bf16_mma: bool, fp8: bool) {
    let arch = CudaArch::from_compute_capability(major, minor);
    assert_eq!(arch.has_tensor_cores(), tensor_cores);
    assert_eq!(arch.has_bf16_mma(), bf16_mma);
    assert_eq!(arch.has_fp8(), fp8);
    assert_eq!(arch.wave_size(), 32);
}

#[test]
fn orders_by_major_then_minor() {
    let sm = CudaArch::from_compute_capability;
    assert!(sm(7, 5) < sm(8, 0));
    assert!(sm(8, 6) < sm(8, 9));
    assert!(sm(9, 0) < sm(10, 0));
}

proptest! {
    #[test]
    fn display_parse_round_trip(major in 0u8..=25, minor in 0u8..=9) {
        let arch = CudaArch::from_compute_capability(major, minor);
        let label = arch.to_string();
        prop_assert!(label.starts_with("sm_"));
        prop_assert_eq!(label.parse::<CudaArch>(), Ok(arch));
        prop_assert_eq!(arch.sm(), major as u32 * 10 + minor as u32);
    }

    #[test]
    fn capability_predicates_are_monotone(a in (0u8..=25, 0u8..=9), b in (0u8..=25, 0u8..=9)) {
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        let lo = CudaArch::from_compute_capability(lo.0, lo.1);
        let hi = CudaArch::from_compute_capability(hi.0, hi.1);
        prop_assert!(lo <= hi);
        prop_assert!(lo.has_tensor_cores() <= hi.has_tensor_cores());
        prop_assert!(lo.has_bf16_mma() <= hi.has_bf16_mma());
        prop_assert!(lo.has_fp8() <= hi.has_fp8());
    }
}
