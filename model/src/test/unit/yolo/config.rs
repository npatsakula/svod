use svod_tensor::nn::{Module, get_tensor};
use test_case::test_case;

use crate::yolo::{Yolo26Detect, YoloConfig, YoloScale, make_depth, make_divisible, scale_channels};

#[test]
fn make_divisible_rounds_up_to_multiple_of_8() {
    assert_eq!(make_divisible(16, 8), 16);
    assert_eq!(make_divisible(17, 8), 24);
    assert_eq!(make_divisible(64, 8), 64);
}

#[test]
fn nano_scale_channels() {
    let s = YoloScale::Nano;
    assert_eq!(scale_channels(64, s), 16);
    assert_eq!(scale_channels(128, s), 32);
    assert_eq!(scale_channels(256, s), 64);
    assert_eq!(scale_channels(512, s), 128);
    assert_eq!(scale_channels(1024, s), 256);
}

#[test]
fn depth_halves_repeats_for_nano() {
    let s = YoloScale::Nano;
    assert_eq!(make_depth(2, s), 1);
    assert_eq!(make_depth(1, s), 1);
}

#[test]
fn depth_preserved_for_large() {
    let s = YoloScale::Large;
    assert_eq!(make_depth(2, s), 2);
    assert_eq!(make_depth(1, s), 1);
}

#[test_case(YoloScale::Nano, false)]
#[test_case(YoloScale::Small, false)]
#[test_case(YoloScale::Medium, true)]
#[test_case(YoloScale::Large, true)]
#[test_case(YoloScale::XLarge, true)]
fn c3k_is_forced_for_medium_and_up(scale: YoloScale, expected: bool) {
    assert_eq!(scale.forces_c3k(), expected);
}

/// Layers 2 and 4 carry `c3k=False` in the YAML, but Ultralytics overrides it
/// for M/L/X. The two shapes differ observably: a plain bottleneck is a pair of
/// 3x3 convs over `c_hidden`, whereas `C3k` splits with 1x1 convs. Getting this
/// wrong loads real checkpoints without complaint — `load_state_dict` replaces
/// tensors without checking shapes — and only fails later in graph
/// construction, so pin the state-dict shapes directly.
#[test_case(YoloScale::Nano, [8, 16, 3, 3], [16, 8, 3, 3])]
#[test_case(YoloScale::XLarge, [48, 96, 1, 1], [48, 96, 1, 1])]
fn shallow_c3k2_blocks_follow_the_scale(scale: YoloScale, cv1: [usize; 4], cv2: [usize; 4]) {
    let sd = Yolo26Detect::with_zero_weights(YoloConfig::new(scale, 80)).state_dict("");
    let dims = |key: &str| get_tensor(&sd, key).expect("state-dict key").dims().expect("dims");

    assert_eq!(dims("2.m.0.cv1.conv.weight"), cv1);
    assert_eq!(dims("2.m.0.cv2.conv.weight"), cv2);
}
