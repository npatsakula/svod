//! Module scopes: the origin path of a node built during `forward` is the
//! state-dict prefix of the module that built it.
//!
//! Cheap — no `.realize()`, no JIT. Both models are built with random weights.

use std::collections::BTreeSet;

use svod_dtype::DType;
use svod_ir::origin::{self, OriginFrame};
use svod_tensor::Tensor;

use crate::gigaam::GigaAm;
use crate::state::StateDict;
use crate::whisper::{ModelDimensions, Whisper, WhisperSize};
use crate::yolo::{Yolo26Detect, YoloConfig, YoloScale};
use svod_tensor::nn::Module as _;

use super::batch::test_config;

/// Every module path reachable from the tensor's cone, including the partial
/// paths of the enclosing scopes. Call frames are dropped: they are the
/// file:line layer beneath the module path, not part of it.
fn module_paths(tensor: &Tensor) -> BTreeSet<String> {
    let mut paths = BTreeSet::new();
    let leaves: BTreeSet<_> = tensor.uop().toposort().iter().filter_map(|node| node.origin()).collect();
    for leaf in leaves {
        let mut path = String::new();
        for frame in origin::chain(leaf).into_iter().filter_map(origin::get) {
            if let OriginFrame::Module { name } = frame.frame {
                if !path.is_empty() {
                    path.push('.');
                }
                path.push_str(&name);
                paths.insert(path.clone());
            }
        }
    }
    paths
}

/// Every scope opened during a forward names a module that owns weights, and
/// every weight is built under the scope that owns it.
fn assert_paths_are_state_dict_prefixes(paths: &BTreeSet<String>, keys: &BTreeSet<String>) {
    assert!(!paths.is_empty(), "the forward opened no module scope");
    for path in paths {
        assert!(
            keys.iter().any(|key| key.starts_with(&format!("{path}."))),
            "scope {path} owns no state-dict key; scopes are {paths:?}"
        );
    }
    for key in keys {
        assert!(
            paths.iter().any(|path| key.starts_with(&format!("{path}."))),
            "weight {key} is built outside any module scope; scopes are {paths:?}"
        );
    }
}

fn keys(state_dict: &StateDict) -> BTreeSet<String> {
    state_dict.keys().cloned().collect()
}

/// The encoder's own state dict, composed the way `Encoder::from_state_dict`
/// keys it (the encoder itself does not implement `Module`).
fn encoder_state_dict(model: &GigaAm) -> StateDict {
    let mut sd = model.encoder.subsampling.state_dict("subsampling");
    for (index, layer) in model.encoder.layers.iter().enumerate() {
        sd.extend(layer.state_dict(&format!("layers.{index}")));
    }
    sd
}

#[test]
fn gigaam_module_scopes_match_the_state_dict() {
    let _capture = origin::capture_for_thread(true);
    let config = test_config();
    let model = GigaAm::with_random_weights(config.clone());
    let mel = Tensor::zeros(&[1, config.n_mels, 64], DType::Float32);

    let out = model.encoder.forward(&mel).expect("encoder forward");
    let paths = module_paths(&out);

    assert_paths_are_state_dict_prefixes(&paths, &keys(&encoder_state_dict(&model)));
    // The composite scopes are there, not just the leaves.
    for expected in ["subsampling", "layers.0", "layers.0.ffn1", "layers.0.mhsa.norm", "layers.1.final_norm"] {
        assert!(paths.contains(expected), "missing {expected} in {paths:?}");
    }
}

#[test]
fn gigaam_layers_get_distinct_origins() {
    let _capture = origin::capture_for_thread(true);
    let config = test_config();
    let model = GigaAm::with_random_weights(config.clone());
    let mel = Tensor::zeros(&[1, config.n_mels, 64], DType::Float32);

    let out = model.encoder.forward(&mel).expect("encoder forward");
    let paths = module_paths(&out);

    // Structurally identical layers must not collapse onto one origin.
    let per_layer: Vec<BTreeSet<&String>> = (0..config.n_layers)
        .map(|index| paths.iter().filter(|path| path.starts_with(&format!("layers.{index}"))).collect())
        .collect();
    assert!(per_layer.iter().all(|layer| !layer.is_empty()), "a layer opened no scope: {paths:?}");
    assert_eq!(per_layer[0].len(), per_layer[config.n_layers - 1].len(), "layers disagree on their scope shape");
}

#[test]
fn gigaam_capture_is_off_by_default() {
    let _capture = origin::capture_for_thread(false);
    let config = test_config();
    let model = GigaAm::with_random_weights(config.clone());
    let mel = Tensor::zeros(&[1, config.n_mels, 64], DType::Float32);

    let out = model.encoder.forward(&mel).expect("encoder forward");
    assert!(module_paths(&out).is_empty());
}

#[test]
fn whisper_encoder_module_scopes_match_the_state_dict() {
    let _capture = origin::capture_for_thread(true);
    let dims = ModelDimensions::for_size(WhisperSize::Tiny);
    let model = Whisper::empty(dims.clone());
    let mel = Tensor::zeros(&[1, dims.n_mels, 3000], DType::Float32);

    let out = model.encode(&mel).expect("whisper encode");
    let paths = module_paths(&out);

    // Whisper loads at the empty root prefix, so its encoder keys are
    // `encoder.blocks.0.attn.query.weight` and the scopes must mirror that.
    let encoder_keys: BTreeSet<String> =
        keys(&model.state_dict("")).into_iter().filter(|key| key.starts_with("encoder.")).collect();
    assert_paths_are_state_dict_prefixes(&paths, &encoder_keys);
    for expected in ["encoder", "encoder.conv1", "encoder.blocks.0.attn", "encoder.blocks.0.attn.query"] {
        assert!(paths.contains(expected), "missing {expected} in {paths:?}");
    }
}

/// YOLO nests every layer under a stage frame (`backbone`, `neck`, `head`)
/// that owns no weights of its own. Beneath it the path is the state-dict
/// prefix: `backbone.2.m.0.cv1` for `2.m.0.cv1.conv.weight`, and the head,
/// which loads at `23`, is `head.one2one_cv2.0.0` for `23.one2one_cv2.0.0.*`.
#[test]
fn yolo_module_scopes_match_the_state_dict() {
    let _capture = origin::capture_for_thread(true);
    let model = Yolo26Detect::with_zero_weights(YoloConfig::new(YoloScale::Nano, 80));
    let images = Tensor::zeros(&[1, 3, 64, 64], DType::Float32);

    let out = model.forward(&images).expect("yolo forward");
    let paths: BTreeSet<String> = module_paths(&out)
        .into_iter()
        .filter_map(|path| match path.split_once('.') {
            Some(("backbone" | "neck", layer)) => Some(layer.to_owned()),
            Some(("head", branch)) => Some(format!("23.{branch}")),
            Some((stage, _)) => panic!("{path} sits under an unknown stage frame {stage}"),
            None => None,
        })
        .collect();

    assert_paths_are_state_dict_prefixes(&paths, &keys(&model.state_dict("")));
    for expected in
        ["0", "2.m.0.cv1", "9.cv2", "10.m.0.attn.qkv", "10.m.0.ffn.1", "13", "23.one2one_cv2.0.0", "23.one2one_cv3.2.2"]
    {
        assert!(paths.contains(expected), "missing {expected} in {paths:?}");
    }
}
