//! `Yolo26Detect::force_stage_dtype` / `YoloConv::with_io_dtype` — running one
//! stage at a dtype the rest of the model does not use.

use svod_dtype::DType;
use svod_tensor::Tensor;
use test_case::test_case;

use crate::yolo::{C3k2Inner, Yolo26Detect, YoloConfig, YoloConv, YoloScale};

const NARROW: DType = DType::Float16;
const WIDE: DType = DType::Float32;

fn model() -> Yolo26Detect {
    Yolo26Detect::with_zero_weights(YoloConfig::new(YoloScale::Nano, 80).with_compute_dtype(NARROW))
}

/// The pair a pin is made of, so a test can say which edges moved.
fn io(conv: &YoloConv) -> (Option<DType>, Option<DType>) {
    (conv.in_dtype.clone(), conv.out_dtype.clone())
}

/// The hook is opt-in, and everything the `--stage-f32` A/B concluded rests on
/// that: an unpinned model must build the graph it built before, or the cached
/// BEAM plans and tk tunings stop applying to it.
#[test]
fn a_fresh_model_pins_no_stage() {
    let m = model();
    let (b, n) = (&m.backbone, &m.neck);
    for (name, conv) in [
        ("backbone.0", &b.conv0),
        ("backbone.1", &b.conv1),
        ("backbone.2.cv1", &b.c3k2_2.cv1),
        ("backbone.2.cv2", &b.c3k2_2.cv2),
        ("backbone.9.cv1", &b.sppf9.cv1),
        ("backbone.9.cv2", &b.sppf9.cv2),
        ("backbone.10.cv1", &b.c2psa10.cv1),
        ("backbone.10.cv2", &b.c2psa10.cv2),
        ("backbone.10.m.0.attn.qkv", &b.c2psa10.m[0].attn.qkv),
        ("neck.17", &n.conv17),
        ("neck.20", &n.conv20),
        ("neck.22.cv1", &n.c3k2_22.cv1),
        ("neck.22.cv2", &n.c3k2_22.cv2),
    ] {
        assert_eq!(io(conv), (None, None), "{name} starts unpinned");
    }
}

/// The hook itself: a block takes its width from the stream it is handed, so
/// widening the input carries the block and the exit cast hands the stream
/// back. This is the whole mechanism a stage pin is built out of.
#[test]
fn with_io_dtype_casts_the_edges_it_is_given() {
    let conv = model().backbone.conv0;
    let x = Tensor::zeros(&[1, 3, 64, 64], NARROW);

    assert_eq!(conv.forward(&x).expect("unpinned").dtype(), NARROW, "unpinned, the stream keeps its width");

    let widened = conv.clone().with_io_dtype(Some(WIDE), None);
    assert_eq!(widened.forward(&x).expect("widened").dtype(), WIDE, "a widened input carries the block");

    let island = conv.with_io_dtype(Some(WIDE), Some(NARROW));
    assert_eq!(island.forward(&x).expect("island").dtype(), NARROW, "the exit cast restores the stream");
}

/// A stage that spans several blocks is pinned at its two ends and nowhere
/// else — the blocks between them need no marking because they read the
/// stream, and the neighbouring stages must not be touched at all.
#[test]
fn pinning_a_multi_block_stage_casts_only_its_two_edges() {
    let mut m = model();
    assert!(m.force_stage_dtype("backbone.9", WIDE));

    let b = &m.backbone;
    assert_eq!(io(&b.sppf9.cv1), (Some(WIDE), None), "the entry takes the new dtype in");
    assert_eq!(io(&b.sppf9.cv2), (None, Some(NARROW)), "the exit hands the compute dtype back");
    assert_eq!(io(&b.c3k2_8.cv2), (None, None), "the stage before is untouched");
    assert_eq!(io(&b.c2psa10.cv1), (None, None), "the stage after is untouched");
}

/// A stage that is one block carries both casts itself.
#[test]
fn pinning_a_single_block_stage_casts_both_ends_of_it() {
    let mut m = model();
    assert!(m.force_stage_dtype("neck.17", WIDE));

    assert_eq!(io(&m.neck.conv17), (Some(WIDE), Some(NARROW)));
    assert_eq!(io(&m.neck.c3k2_16.cv2), (None, None), "the stage before is untouched");
    assert_eq!(io(&m.neck.c3k2_19.cv1), (None, None), "the stage after is untouched");
}

/// The attention sub-stages walk a `Vec`, so a pin has to reach every block in
/// the stack rather than the first one — a partial pin would read as a clean
/// bisect step and quietly measure a mixture.
#[test]
fn pinning_the_backbone_attention_reaches_every_block_of_the_stack() {
    let mut m = model();
    assert!(m.force_stage_dtype("backbone.10.attn", WIDE));

    let blocks = &m.backbone.c2psa10.m;
    assert!(!blocks.is_empty(), "nano's C2PSA has a PSA block to pin");
    for (i, blk) in blocks.iter().enumerate() {
        assert_eq!(io(&blk.attn.qkv), (Some(WIDE), None), "block {i} qkv takes the new dtype in");
        assert_eq!(io(&blk.attn.proj), (None, Some(NARROW)), "block {i} proj hands it back");
    }
    assert_eq!(io(&m.backbone.c2psa10.cv1), (None, None), "the C2PSA around it is untouched");
}

/// Layer 22 hides its `PSABlock` inside a `C3k2Inner::Attn`, so this pin also
/// exercises the enum match.
#[test]
fn pinning_the_neck_attention_reaches_inside_the_c3k2() {
    let mut m = model();
    assert!(m.force_stage_dtype("neck.22.attn", WIDE));

    let pinned = m
        .neck
        .c3k2_22
        .m
        .iter()
        .filter_map(|inner| match inner {
            C3k2Inner::Attn(_, psa) => Some(psa),
            _ => None,
        })
        .inspect(|psa| {
            assert_eq!(io(&psa.attn.qkv), (Some(WIDE), None));
            assert_eq!(io(&psa.attn.proj), (None, Some(NARROW)));
        })
        .count();
    assert!(pinned > 0, "layer 22 is built with attn = true, so there is one to pin");
    assert_eq!(io(&m.neck.c3k2_22.cv1), (None, None), "the C3k2 around it is untouched");
}

/// A name that does not resolve must be refused, not silently ignored: the
/// flag would otherwise report a bisect step it never ran.
#[test]
fn an_unknown_stage_is_refused_and_changes_nothing() {
    let mut m = model();
    for name in ["backbone.11", "neck.14", "head.23", "backbone.10.ffn", ""] {
        assert!(!m.force_stage_dtype(name, WIDE), "{name:?} is not a stage");
    }
    assert_eq!(io(&m.backbone.conv0), (None, None));
    assert_eq!(io(&m.backbone.c2psa10.m[0].attn.qkv), (None, None));
    assert_eq!(io(&m.neck.c3k2_22.cv2), (None, None));
}

/// Every name the flag accepts has to run end to end. The casts sit on the
/// NCHW module edges on purpose — inside a chain that keeps `[B, H, W, C]` the
/// producer and consumer agree on layout only because they read the same
/// stream dtype — so a pin landing anywhere else would mismatch the layout
/// here rather than in a profiling run.
#[test_case("backbone.0" ; "stem")]
#[test_case("backbone.1" ; "a tk downsample")]
#[test_case("backbone.2" ; "a C3k2")]
#[test_case("backbone.9" ; "SPPF")]
#[test_case("backbone.10" ; "C2PSA")]
#[test_case("backbone.10.attn" ; "C2PSA attention")]
#[test_case("neck.13" ; "FPN")]
#[test_case("neck.17" ; "a PAN downsample")]
#[test_case("neck.22" ; "the P5 C3k2")]
#[test_case("neck.22.attn" ; "the P5 attention")]
fn a_pinned_model_still_runs_and_returns_f32(stage: &str) {
    let mut m = model();
    assert!(m.force_stage_dtype(stage, WIDE), "{stage} resolves");

    let out = m.forward(&Tensor::zeros(&[1, 3, 64, 64], DType::Float32)).expect("pinned model forward");
    assert_eq!(out.dtype(), DType::Float32, "predictions still reach the caller as f32");
}

/// The stream leaves the pinned stage at the compute dtype, so the rest of the
/// backbone is unaffected by the island.
#[test_case("backbone.9" ; "SPPF")]
#[test_case("backbone.10" ; "C2PSA")]
fn a_pinned_stage_hands_the_stream_back_narrow(stage: &str) {
    let mut m = model();
    assert!(m.force_stage_dtype(stage, WIDE));

    let (l4, l6, l10) = m.backbone.forward(&Tensor::zeros(&[1, 3, 64, 64], NARROW)).expect("backbone forward");
    for (name, feat) in [("l4", &l4), ("l6", &l6), ("l10", &l10)] {
        assert_eq!(feat.dtype(), NARROW, "{name} leaves the backbone at the compute dtype");
    }
}
