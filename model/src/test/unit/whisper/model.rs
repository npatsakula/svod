//! Forward shape + state-dict round-trip tests for Whisper model.

use std::collections::BTreeSet;
use std::mem::size_of;

use svod_dtype::{DType, ScalarDType};
use svod_ir::{AxisType, ConstValue, Op};
use svod_macros::jit_wrapper;
use svod_tensor::Tensor;
use test_case::test_case;

use crate::jit::InputSpec;
use crate::whisper::blocks::linear_forward;
use crate::whisper::{
    DecodeOptions, DecodeResult, DecodeStrategy, ModelDimensions, Whisper, WhisperAlignmentJit, WhisperAlignmentModel,
    WhisperDecoderStepJit, WhisperPlan, WhisperPrefillJit, WhisperSize,
};
use svod_ir::ops;
use svod_tensor::nn::{Layer, Linear, Module};

// The cross projection has no wrapper of its own any more -- prefill owns it --
// but the shape of the graph it emits is still worth pinning, so the tests
// compile one.
jit_wrapper! {
    CrossKvProbeJit(Whisper) {
        audio_features: Tensor,

        outputs { cross_k, cross_v }

        build(audio_features) {
            model.project_cross_kv(audio_features)
        }
    }
}

fn make_dims() -> ModelDimensions {
    ModelDimensions::for_size(WhisperSize::Tiny)
}

#[test]
fn encoder_forward_shape() {
    let dims = make_dims();
    let model = Whisper::empty(dims.clone());

    // mel: [1, n_mels, 3000]
    let mel = Tensor::zeros(&[1, dims.n_mels, 3000], DType::Float32);

    let out = model.encode(&mel).unwrap();
    let shape = out.shape().unwrap();
    assert_eq!(shape.len(), 3);
    assert_eq!(shape[0].as_const(), Some(1));
    // conv stride 2: 3000/2 = 1500
    assert_eq!(shape[1].as_const(), Some(1500));
    assert_eq!(shape[2].as_const(), Some(dims.n_audio_state));
    // Encoder features leave in the activation dtype; prefill consumes them there.
    assert_eq!(out.dtype(), dims.dtype);
}

#[test]
fn decoder_forward_shape() {
    let dims = make_dims();
    let model = Whisper::empty(dims.clone());

    // mel → encoder → features
    let mel = Tensor::zeros(&[1, dims.n_mels, 3000], DType::Float32);
    let features = model.encode(&mel).unwrap();

    // tokens: [1, 4]
    let tokens = Tensor::from_slice([50363i32, 50364, 50359, 50363]).try_reshape([1usize, 4]).unwrap();

    let logits = model.decode(&tokens, &features, 0).unwrap();
    let shape = logits.shape().unwrap();
    assert_eq!(shape.len(), 3);
    assert_eq!(shape[0].as_const(), Some(1));
    assert_eq!(shape[1].as_const(), Some(4));
    assert_eq!(shape[2].as_const(), Some(dims.n_vocab));
}

fn small_decoder_dims() -> ModelDimensions {
    ModelDimensions {
        n_mels: 4,
        n_audio_ctx: 5,
        n_audio_state: 8,
        n_audio_head: 2,
        n_audio_layer: 1,
        n_vocab: 16,
        n_text_ctx: 8,
        n_text_state: 8,
        n_text_head: 2,
        n_text_layer: 2,
        dtype: DType::Float32,
    }
}

/// Both K/V caches follow the activation dtype, because the projections that
/// fill them already produce it. FP8 is the one exception: attention cannot
/// read it, so the cache widens to f16.
#[test_case(DType::Float32, DType::Float32; "f32 passes through")]
#[test_case(DType::Float16, DType::Float16; "f16 passes through")]
#[test_case(DType::BFloat16, DType::BFloat16; "bf16 passes through")]
#[test_case(DType::FP8E4M3, DType::Float16; "fp8 e4m3 widens")]
#[test_case(DType::FP8E4M3FNUZ, DType::Float16; "fp8 e4m3fnuz widens")]
#[test_case(DType::FP8E5M2, DType::Float16; "fp8 e5m2 widens")]
#[test_case(DType::FP8E5M2FNUZ, DType::Float16; "fp8 e5m2fnuz widens")]
fn cache_dtype_follows_the_activation_dtype_except_fp8(activation: DType, expected: DType) {
    let dims = ModelDimensions { dtype: activation, ..small_decoder_dims() };
    assert_eq!(dims.cache_dtype(), expected);
}

fn reference_cross_kv_projection(model: &Whisper, audio: &Tensor) -> (Tensor, Tensor) {
    let mut keys = Vec::with_capacity(model.decoder.blocks.len());
    let mut values = Vec::with_capacity(model.decoder.blocks.len());
    for block in &model.decoder.blocks {
        let heads = block.cross_attn.n_head;
        let key = linear_forward(&block.cross_attn.key, audio).unwrap();
        let value = linear_forward(&block.cross_attn.value, audio).unwrap();
        keys.push(key.split_heads(heads).unwrap().try_permute(&[0, 2, 1, 3]).unwrap());
        values.push(value.split_heads(heads).unwrap().try_permute(&[0, 2, 1, 3]).unwrap());
    }
    let keys = Tensor::cat(&keys.iter().collect::<Vec<_>>(), 2).unwrap().cast(DType::Float32);
    let values = Tensor::cat(&values.iter().collect::<Vec<_>>(), 2).unwrap().cast(DType::Float32);
    (keys, values)
}

/// Read a cache tensor as f32, and read a reference rounded through the cache's
/// own dtype -- the only fair comparison once the cache is stored narrow.
fn as_f32(t: &Tensor) -> Vec<f32> {
    t.cast(DType::Float32).to_vec::<f32>().unwrap()
}

fn stored_as_f32(t: &Tensor, cache_dtype: DType) -> Vec<f32> {
    as_f32(&t.cast(cache_dtype))
}

#[test]
fn materialized_cross_kv_matches_reference_projection() {
    let mut dims = small_decoder_dims();
    dims.n_text_layer = 3;
    let seed = Whisper::empty(dims.clone());
    let model = Whisper::from_state_dict(&seed.state_dict(""), dims.clone()).unwrap();
    let audio_values: Vec<f32> =
        (0..2 * dims.n_audio_ctx * dims.n_text_state).map(|index| (index as f32 - 31.0) * 0.017).collect();
    let audio = Tensor::from_slice(audio_values).try_reshape([2usize, dims.n_audio_ctx, dims.n_text_state]).unwrap();

    let (expected_k, expected_v) = reference_cross_kv_projection(&model, &audio);
    let (actual_k, actual_v) = model.project_cross_kv(&audio).unwrap();
    Tensor::realize_batch([&expected_k, &expected_v, &actual_k, &actual_v]).unwrap();

    let expected_shape = [2, dims.n_audio_ctx, dims.n_text_layer * dims.n_text_head, 4];
    for (expected, actual) in [(&expected_k, &actual_k), (&expected_v, &actual_v)] {
        assert_eq!(actual.dtype(), dims.cache_dtype());
        assert_eq!(actual.dims().unwrap(), expected_shape);
        // The cache may store narrower than the reference computes, so round the
        // reference through it: this keeps the check on the projection rather
        // than on the storage's own quantization.
        let expected = stored_as_f32(expected, dims.cache_dtype());
        let actual = as_f32(actual);
        let max_delta = expected.iter().zip(&actual).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        // One ULP of the storage dtype at these magnitudes. Rounding is a step
        // function, so two f32 values a hair apart can land either side of a
        // boundary; anything past a ULP is a real projection error, not storage.
        assert!(max_delta < 1e-3, "materialized cross projection drifted by {max_delta}");
    }
}

#[test]
fn low_precision_cross_projection_runs_in_the_activation_dtype() {
    let _structural = svod_ir::origin::capture_for_thread(false);
    let mut dims = small_decoder_dims();
    dims.dtype = DType::Float16;
    let model = Whisper::empty(dims.clone());
    let audio_values: Vec<f32> =
        (0..dims.n_audio_ctx * dims.n_text_state).map(|index| (index as f32 - 13.0) * 0.037).collect();
    let audio = Tensor::from_slice(audio_values).try_reshape([1usize, dims.n_audio_ctx, dims.n_text_state]).unwrap();
    let native_audio = audio.cast(DType::Float16);

    let (expected_k, expected_v) = reference_cross_kv_projection(&model, &native_audio);
    let (legacy_k, legacy_v) = reference_cross_kv_projection(&model, &audio);
    let (actual_k, actual_v) = model.project_cross_kv(&audio).unwrap();
    Tensor::realize_batch([&expected_k, &expected_v, &legacy_k, &legacy_v, &actual_k, &actual_v]).unwrap();

    for ((expected, legacy), actual) in
        [(&expected_k, &legacy_k), (&expected_v, &legacy_v)].into_iter().zip([&actual_k, &actual_v])
    {
        assert_eq!(actual.dtype(), dims.cache_dtype());
        let (expected, legacy) =
            (stored_as_f32(expected, dims.cache_dtype()), stored_as_f32(legacy, dims.cache_dtype()));
        let actual = as_f32(actual);
        assert_eq!(actual, expected, "projection must run in the model activation dtype before cache storage");
        assert_ne!(legacy, expected, "fixture must detect projection inherited from the f32 encoder output");
    }
}

/// OpenAI keeps embeddings and LayerNorm affine parameters at checkpoint
/// precision; only projections take the compute dtype. Observable at FP8,
/// where a coerced token embedding would quantize the vocabulary table.
#[test_case(DType::Float16)]
#[test_case(DType::FP8E4M3)]
fn low_precision_load_preserves_openai_fp32_parameters(compute: DType) {
    let source_dims = small_decoder_dims();
    let source = Whisper::empty(source_dims.clone()).state_dict("");
    let mut low_dims = source_dims;
    low_dims.dtype = compute.clone();
    let model = Whisper::from_state_dict(&source, low_dims).unwrap();

    assert_eq!(model.encoder.conv1.weight.dtype(), compute);
    assert_eq!(model.encoder.blocks[0].attn.query.weight.dtype(), compute);
    assert_eq!(model.encoder.positional_embedding.dtype(), DType::Float32);
    assert_eq!(model.encoder.blocks[0].attn_ln.weight.dtype(), DType::Float32);
    assert_eq!(model.decoder.token_embedding.dtype(), DType::Float32);
    assert_eq!(model.decoder.positional_embedding.dtype(), DType::Float32);
    assert_eq!(model.decoder.blocks[0].cross_attn_ln.bias.as_ref().unwrap().dtype(), DType::Float32);
    assert_eq!(model.decoder.ln.weight.dtype(), DType::Float32);
}

/// A quantized linear keeps its fp8 weight and per-output-channel scale as
/// loaded, and its forward applies the scale to the f32 accumulator: the same
/// result as multiplying the scale into a dequantized weight, with the kernel
/// reading one byte per weight.
#[test]
fn quantized_weight_stays_fp8_and_scales_the_accumulator() {
    let dims = small_decoder_dims();
    let mut sd = Whisper::empty(dims.clone()).state_dict("");
    let key = "decoder.blocks.0.mlp.0.weight";
    let (out, inp) = (dims.n_text_state * 4, dims.n_text_state);
    let quantized: Vec<u8> = (0..out * inp)
        .map(|index| svod_dtype::cast::float_to_fp8(((index % 13) as f64 - 6.0) * 0.5, ScalarDType::FP8E4M3).unwrap())
        .collect();
    let scale: Vec<f32> = (0..out).map(|row| 0.5 + row as f32 * 0.125).collect();
    sd.insert(key.into(), Tensor::from_raw_bytes(&quantized, &[out, inp], DType::FP8E4M3).unwrap());
    sd.insert(format!("{key}.weight_scale"), Tensor::from_slice(scale.clone()).try_reshape([out, 1]).unwrap());

    let model = Whisper::from_state_dict(&sd, dims).unwrap();
    let layer = &model.decoder.blocks[0].mlp0;
    assert_eq!(layer.weight.dtype(), DType::FP8E4M3, "the weight is read as stored");
    assert_eq!(layer.weight_scale.as_ref().map(|scale| scale.dims().unwrap()), Some(vec![out, 1]));

    let x = Tensor::from_slice((0..2 * inp).map(|value| (value % 7) as f32 * 0.25 - 0.5).collect::<Vec<_>>())
        .try_reshape([2, inp])
        .unwrap()
        .cast(DType::Float16);
    let actual = linear_forward(layer, &x).unwrap().cast(DType::Float32).to_vec::<f32>().unwrap();
    let dequantized: Vec<f32> = quantized
        .iter()
        .enumerate()
        .map(|(index, &byte)| {
            svod_dtype::cast::fp8_to_float(byte, ScalarDType::FP8E4M3).unwrap() as f32 * scale[index / inp]
        })
        .collect();
    let reference = Tensor::from_slice(dequantized).try_reshape([out, inp]).unwrap().cast(DType::Float16);
    let expected = linear_forward(&Linear::new(reference, layer.bias.clone()), &x)
        .unwrap()
        .cast(DType::Float32)
        .to_vec::<f32>()
        .unwrap();
    for (a, e) in actual.iter().zip(&expected) {
        assert!((a - e).abs() <= 1e-2 * e.abs().max(1.0), "fp8 linear {a} vs dequantized {e}");
    }
}

/// Only `Linear` has a field for a `weight_scale`. A checkpoint that quantizes
/// anything else -- a convolution, the token embedding -- must have the scale
/// folded into its weight, or the module would load raw fp8 codes with the
/// scale silently dropped. Quantized linears in the same checkpoint still stay
/// narrow with their scale beside them.
#[test_case("encoder.conv1.weight"; "conv1d weight")]
fn scale_on_a_non_linear_weight_folds_into_the_weight(key: &str) {
    let mut dims = small_decoder_dims();
    dims.dtype = DType::Float16;
    let mut sd = Whisper::empty(dims.clone()).state_dict("");
    let shape = sd[key].dims().unwrap();
    let (weight, scale, dequantized) = quantize_rows(&shape);
    sd.insert(key.into(), weight);
    sd.insert(format!("{key}.weight_scale"), scale);

    // A genuinely quantized linear in the same checkpoint, which must stay narrow.
    let linear_key = "decoder.blocks.0.mlp.0.weight";
    let (weight, scale, _) = quantize_rows(&[dims.n_text_state * 4, dims.n_text_state]);
    sd.insert(linear_key.into(), weight);
    sd.insert(format!("{linear_key}.weight_scale"), scale);

    let model = Whisper::from_state_dict(&sd, dims.clone()).unwrap();
    let loaded = match key {
        "encoder.conv1.weight" => &model.encoder.conv1.weight,
        other => unreachable!("unhandled fixture key {other}"),
    };
    assert_eq!(loaded.dtype(), dims.dtype, "a weight no module can scale must widen to the compute dtype");
    assert_eq!(loaded.dims().unwrap(), shape, "folding preserves the weight's shape");
    let actual = loaded.cast(DType::Float32).to_vec::<f32>().unwrap();
    for (index, (a, e)) in actual.iter().zip(&dequantized).enumerate() {
        assert!((a - e).abs() <= 1e-2 * e.abs().max(1.0), "{key}[{index}] folded {a} vs dequantized {e}");
    }

    let quantized = &model.decoder.blocks[0].mlp0;
    assert_eq!(quantized.weight.dtype(), DType::FP8E4M3, "a linear weight still reaches the kernel as stored");
    assert!(quantized.weight_scale.is_some(), "and keeps its scale to apply to the accumulator");
}

/// An fp8 weight of `shape`, its `[out, 1]` per-output-channel scale, and the
/// row-major dequantized product the two stand for.
fn quantize_rows(shape: &[usize]) -> (Tensor, Tensor, Vec<f32>) {
    let (out, inner) = (shape[0], shape[1..].iter().product::<usize>());
    let codes: Vec<u8> = (0..out * inner)
        .map(|index| svod_dtype::cast::float_to_fp8(((index % 13) as f64 - 6.0) * 0.5, ScalarDType::FP8E4M3).unwrap())
        .collect();
    let scale: Vec<f32> = (0..out).map(|row| 0.5 + row as f32 * 0.125).collect();
    let dequantized = codes
        .iter()
        .enumerate()
        .map(|(index, &byte)| {
            svod_dtype::cast::fp8_to_float(byte, ScalarDType::FP8E4M3).unwrap() as f32 * scale[index / inner]
        })
        .collect();
    let weight = Tensor::from_raw_bytes(&codes, shape, DType::FP8E4M3).unwrap();
    let scale = Tensor::from_slice(scale).try_reshape([out, 1]).unwrap();
    (weight, scale, dequantized)
}

/// Prefill owns the cross projection now, so the cache dtype is a property of
/// what it hands back rather than of what it is handed. Storage must be the
/// declared cache dtype, must not re-round what `project_cross_kv` produced,
/// and must not narrow the compute that reads it back in the same pass.
#[test]
fn low_precision_prefill_does_not_inherit_cache_storage_dtype() {
    let source_dims = small_decoder_dims();
    let source = Whisper::empty(source_dims.clone()).state_dict("");
    let mut dims = source_dims;
    dims.dtype = DType::Float16;
    let model = Whisper::from_state_dict(&source, dims.clone()).unwrap();
    let audio = Tensor::from_slice(
        (0..dims.n_audio_ctx * dims.n_text_state).map(|index| (index as f32 - 9.0) * 0.031).collect::<Vec<_>>(),
    )
    .try_reshape([1usize, dims.n_audio_ctx, dims.n_text_state])
    .unwrap();
    let tokens = Tensor::from_slice([1i32, 2, 3]).try_reshape([1usize, 3]).unwrap();

    let (projected_k, projected_v) = model.project_cross_kv(&audio).unwrap();
    let (logits, self_k, self_v, cross_k, cross_v) = model.decode_prefill(&tokens, &audio, 0).unwrap();
    // The un-stored path: the decoder never round-trips anything through the cache.
    let direct = model.decode(&tokens, &audio, 0).unwrap();
    Tensor::realize_batch([&projected_k, &projected_v, &logits, &self_k, &self_v, &cross_k, &cross_v, &direct])
        .unwrap();

    for cache in [&self_k, &self_v, &cross_k, &cross_v] {
        assert_eq!(cache.dtype(), dims.cache_dtype());
    }
    for (prefilled, projected) in [(&cross_k, &projected_k), (&cross_v, &projected_v)] {
        assert_eq!(as_f32(prefilled), as_f32(projected), "prefill must hand back the projection it stored");
    }

    let (direct, prefilled) = (direct.as_vec::<f32>().unwrap(), logits.as_vec::<f32>().unwrap());
    let max_delta = direct.iter().zip(&prefilled).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(max_delta < 2e-2, "cache storage changed low-precision prefill compute by {max_delta}");
}

#[test]
fn prepared_cross_kv_materializes_each_projection_before_packing() {
    let mut dims = small_decoder_dims();
    dims.n_text_layer = 3;
    let seed = Whisper::empty(dims.clone());
    let model = Whisper::from_state_dict(&seed.state_dict(""), dims.clone()).unwrap();
    let mut jit = CrossKvProbeJit::new(model);
    jit.prepare(InputSpec::f32(&[1, dims.n_audio_ctx, dims.n_text_state])).unwrap();

    let kernels = jit.prepared_kernels().unwrap();
    let reduction_counts: Vec<_> = kernels
        .iter()
        .map(|kernel| {
            kernel
                .ast
                .toposort()
                .into_iter()
                .filter(|uop| matches!(uop.op(), Op::Range(ops::Range { axis_type: AxisType::Reduce, .. })))
                .count()
        })
        .collect();
    assert!(
        reduction_counts.iter().all(|&count| count <= 1),
        "no dispatch may contain repeated independent reductions: {reduction_counts:?}"
    );
    assert_eq!(
        reduction_counts.iter().filter(|&&count| count == 1).count(),
        2 * dims.n_text_layer,
        "each layer must have independent key and value projection kernels: {reduction_counts:?}"
    );
    assert!(
        reduction_counts.iter().filter(|&&count| count == 0).count() >= 2,
        "key and value packing must remain reduction-free: {reduction_counts:?}"
    );
}

#[test]
fn projected_cross_kv_and_prefill_shapes_are_concrete() {
    let dims = small_decoder_dims();
    let model = Whisper::empty(dims.clone());
    let audio = Tensor::zeros(&[1, dims.n_audio_ctx, dims.n_text_state], DType::Float32);
    let tokens = Tensor::from_slice([1i32, 2, 3]).try_reshape([1usize, 3]).unwrap();

    let expected_cross = [1, dims.n_audio_ctx, dims.n_text_layer * dims.n_text_head, 4];
    let (projected_k, projected_v) = model.project_cross_kv(&audio).unwrap();
    for cache in [&projected_k, &projected_v] {
        assert_eq!(cache.dims().unwrap(), expected_cross);
    }

    let (logits, self_k, self_v, cross_k, cross_v) = model.decode_prefill(&tokens, &audio, 0).unwrap();
    assert_eq!(logits.dims().unwrap(), [1, 3, dims.n_vocab]);
    let expected_self = [1, 3, dims.n_text_layer * dims.n_text_head, 4];
    for cache in [&self_k, &self_v] {
        assert_eq!(cache.dims().unwrap(), expected_self);
    }
    for cache in [&cross_k, &cross_v] {
        assert_eq!(cache.dims().unwrap(), expected_cross);
    }
}

#[test]
fn prepared_cross_kv_prefill_matches_direct_decoder() {
    let dims = small_decoder_dims();
    let model = Whisper::empty(dims.clone());
    let audio_values: Vec<f32> = (0..dims.n_audio_ctx * dims.n_text_state).map(|i| i as f32 * 0.01).collect();
    let audio = Tensor::from_slice(audio_values).try_reshape([1usize, dims.n_audio_ctx, dims.n_text_state]).unwrap();
    let tokens = Tensor::from_slice([1i32, 2, 3]).try_reshape([1usize, 3]).unwrap();

    let direct = model.decode(&tokens, &audio, 0).unwrap();
    let prepared = model.decode_prefill(&tokens, &audio, 0).unwrap().0;
    Tensor::realize_batch([&direct, &prepared]).unwrap();
    let direct = direct.as_vec::<f32>().unwrap();
    let prepared = prepared.as_vec::<f32>().unwrap();
    let max_delta = direct.iter().zip(&prepared).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(max_delta < 1e-5, "prepared cross-cache logits drifted by {max_delta}");
}

#[test]
fn cached_steps_match_teacher_forced_full_prefix() {
    let dims = small_decoder_dims();
    let model = Whisper::empty(dims.clone());
    let d_head = dims.n_text_state / dims.n_text_head;
    let layer_heads = dims.n_text_layer * dims.n_text_head;
    let cache_elements = dims.n_text_ctx * layer_heads * d_head;
    let audio_values: Vec<f32> =
        (0..dims.n_audio_ctx * dims.n_text_state).map(|index| (index as f32 - 17.0) * 0.021).collect();
    let audio = Tensor::from_slice(audio_values).try_reshape([1usize, dims.n_audio_ctx, dims.n_text_state]).unwrap();
    let mut prefix = vec![1i32, 7, 3];
    let prefix_tensor = Tensor::from_slice(&prefix).try_reshape([1usize, prefix.len()]).unwrap();
    let (_, prefill_k, prefill_v, cross_k, cross_v) = model.decode_prefill(&prefix_tensor, &audio, 0).unwrap();
    Tensor::realize_batch([&prefill_k, &prefill_v, &cross_k, &cross_v]).unwrap();

    // The host-side cache mirror is f32, which is the cache dtype of these dims.
    assert_eq!(dims.cache_dtype(), DType::Float32);
    assert_eq!(cross_k.dtype(), dims.cache_dtype());
    assert_eq!(prefill_k.dtype(), dims.cache_dtype());

    let mut cache_k = vec![0.0f32; cache_elements];
    let mut cache_v = vec![0.0f32; cache_elements];
    let prefill_elements = prefix.len() * layer_heads * d_head;
    cache_k[..prefill_elements].copy_from_slice(&prefill_k.as_vec::<f32>().unwrap());
    cache_v[..prefill_elements].copy_from_slice(&prefill_v.as_vec::<f32>().unwrap());
    assert_eq!(cache_k.len() * size_of::<f32>(), dims.n_text_ctx * layer_heads * d_head * 4);

    for next_token in [5i32, 11] {
        let pos = prefix.len();
        let token = Tensor::from_slice([next_token]).try_reshape([1usize, 1]).unwrap();
        let self_k = Tensor::from_slice(&cache_k).try_reshape([1usize, dims.n_text_ctx, layer_heads, d_head]).unwrap();
        let self_v = Tensor::from_slice(&cache_v).try_reshape([1usize, dims.n_text_ctx, layer_heads, d_head]).unwrap();
        // Each row's cached-key count is also its position, so the graph gathers
        // the positional embedding from it -- no host-side `pos_emb` input.
        let key_lens = Tensor::from_slice([pos as i32]);
        let cross_map = Tensor::from_slice([0i32]);

        let (step_logits, new_k, new_v) =
            model.decode_step(&token, &self_k, &self_v, &cross_k, &cross_v, &key_lens, &cross_map).unwrap();
        prefix.push(next_token);
        let full_tokens = Tensor::from_slice(&prefix).try_reshape([1usize, prefix.len()]).unwrap();
        let teacher = model.decode_prefill(&full_tokens, &audio, 0).unwrap().0;
        Tensor::realize_batch([&step_logits, &new_k, &new_v, &teacher]).unwrap();

        let step = step_logits.as_vec::<f32>().unwrap();
        let teacher = teacher.as_vec::<f32>().unwrap();
        let teacher_last = &teacher[(prefix.len() - 1) * dims.n_vocab..prefix.len() * dims.n_vocab];
        let max_delta = step.iter().zip(teacher_last).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_delta < 1e-5, "cached step at position {pos} drifted from teacher forcing by {max_delta}");

        let cache_offset = pos * layer_heads * d_head;
        let cache_end = cache_offset + layer_heads * d_head;
        cache_k[cache_offset..cache_end].copy_from_slice(&new_k.as_vec::<f32>().unwrap());
        cache_v[cache_offset..cache_end].copy_from_slice(&new_v.as_vec::<f32>().unwrap());
    }
}

/// `detect_language` reads logits row 0 of the prefill graph, which is
/// conditioned on SOT alone. A one-token prompt and a full-length one must
/// therefore agree on that row, down to the ranking of the language tokens.
#[test]
fn one_token_language_logits_match_full_context_sot_logits() {
    let dims = small_decoder_dims();
    let model = Whisper::empty(dims.clone());
    let audio_values: Vec<f32> = (0..dims.n_audio_ctx * dims.n_text_state).map(|i| i as f32 * 0.013).collect();
    let audio = Tensor::from_slice(audio_values).try_reshape([1usize, dims.n_audio_ctx, dims.n_text_state]).unwrap();
    let mut padded_tokens = vec![0i32; dims.n_text_ctx];
    padded_tokens[0] = 1;
    let full_tokens = Tensor::from_slice(padded_tokens).try_reshape([1usize, dims.n_text_ctx]).unwrap();
    let one_token = Tensor::from_slice([1i32]).try_reshape([1usize, 1]).unwrap();

    let full = model.decode_prefill(&full_tokens, &audio, 0).unwrap().0;
    let one = model.decode_prefill(&one_token, &audio, 0).unwrap().0;
    Tensor::realize_batch([&full, &one]).unwrap();
    let full = full.as_vec::<f32>().unwrap();
    let one = one.as_vec::<f32>().unwrap();
    let max_delta = full[..dims.n_vocab].iter().zip(&one).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(max_delta < 1e-5, "one-token SOT logits drifted by {max_delta}");

    let language_tokens = [2usize, 5, 9, 12];
    let rank = |logits: &[f32]| {
        let mut ranked = language_tokens.map(|token| (token, logits[token]));
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        ranked
    };
    assert_eq!(rank(&full[..dims.n_vocab]).map(|(token, _)| token), rank(&one).map(|(token, _)| token));
}

#[test]
fn prepared_language_detector_reads_the_sot_row_of_the_prefill_graph() {
    const WHISPER_TEXT_CONTEXT: i64 = 448;

    let dims = small_decoder_dims();
    let model = Whisper::empty(dims.clone());
    let audio = Tensor::zeros(&[1, dims.n_audio_ctx, dims.n_text_state], DType::Float32);
    // `detect_language` fills the whole prompt with SOT; only row 0 is read, and
    // it must not depend on what follows.
    let prompt_len = 3usize;
    let sot_prompt = model
        .decode_prefill(&Tensor::from_slice([1i32; 3]).try_reshape([1usize, prompt_len]).unwrap(), &audio, 0)
        .unwrap()
        .0;
    let mixed_prompt = model
        .decode_prefill(&Tensor::from_slice([1i32, 4, 9]).try_reshape([1usize, prompt_len]).unwrap(), &audio, 0)
        .unwrap()
        .0;
    Tensor::realize_batch([&sot_prompt, &mixed_prompt]).unwrap();
    assert_eq!(sot_prompt.dims().unwrap(), [1, prompt_len, dims.n_vocab]);
    let (sot_prompt, mixed_prompt) = (sot_prompt.as_vec::<f32>().unwrap(), mixed_prompt.as_vec::<f32>().unwrap());
    assert_eq!(sot_prompt[..dims.n_vocab], mixed_prompt[..dims.n_vocab], "row 0 must depend on SOT alone");

    let mut detector = WhisperPrefillJit::new(model);
    detector
        .prepare(
            InputSpec::i32(&[1, prompt_len]),
            InputSpec::new(&[1, dims.n_audio_ctx, dims.n_text_state], dims.dtype.clone()).device_local(),
        )
        .unwrap();
    assert_eq!(detector.tokens_mut().unwrap().size(), prompt_len * size_of::<i32>());
    assert_eq!(detector.logits().unwrap().size(), prompt_len * dims.n_vocab * size_of::<f32>());
    assert!(detector.prepared_kernels().unwrap().iter().all(|kernel| {
        kernel.ast.toposort().into_iter().all(|uop| {
            !matches!(uop.op(), Op::Const(value) if matches!(value.0, ConstValue::Int(WHISPER_TEXT_CONTEXT) | ConstValue::UInt(448)))
        })
    }));
}

/// Element count of one packed cache of `positions` rows.
fn cache_shape(dims: &ModelDimensions, rows: usize, positions: usize) -> [usize; 4] {
    [rows, positions, dims.n_text_layer * dims.n_text_head, dims.n_text_state / dims.n_text_head]
}

#[test]
#[ignore = "heavy: prepares and runs the prefill and decoder-step graphs through the CPU backend"]
fn prefill_cross_kv_seeds_the_decoder_step_device_locally() {
    let dims = small_decoder_dims();
    let model = Whisper::empty(dims.clone());
    let cache_bytes = |shape: [usize; 4]| shape.iter().product::<usize>() * dims.cache_dtype().bytes();
    let cache = |shape: [usize; 4]| InputSpec::new(&shape, dims.cache_dtype()).device_local();

    let mut prefill = WhisperPrefillJit::new(model.clone());
    prefill
        .prepare(
            InputSpec::i32(&[1, 3]),
            InputSpec::new(&[1, dims.n_audio_ctx, dims.n_text_state], dims.dtype.clone()).device_local(),
        )
        .unwrap();
    prefill.tokens_mut().unwrap().copyin(bytemuck::cast_slice(&[1i32, 2, 3])).unwrap();
    prefill.execute().unwrap();

    assert_eq!(prefill.logits().unwrap().size(), 3 * dims.n_vocab * size_of::<f32>());
    for (buffer, shape) in [
        (prefill.self_k().unwrap(), cache_shape(&dims, 1, 3)),
        (prefill.self_v().unwrap(), cache_shape(&dims, 1, 3)),
        (prefill.cross_k().unwrap(), cache_shape(&dims, 1, dims.n_audio_ctx)),
        (prefill.cross_v().unwrap(), cache_shape(&dims, 1, dims.n_audio_ctx)),
    ] {
        assert_eq!(buffer.size(), cache_bytes(shape));
    }

    // The cross caches prefill produced feed the step graph without a host round trip.
    let mut step = WhisperDecoderStepJit::new(model);
    step.prepare(
        InputSpec::i32(&[1, 1]),
        cache(cache_shape(&dims, 1, dims.n_text_ctx)),
        cache(cache_shape(&dims, 1, dims.n_text_ctx)),
        cache(cache_shape(&dims, 1, dims.n_audio_ctx)),
        cache(cache_shape(&dims, 1, dims.n_audio_ctx)),
        InputSpec::i32(&[1]),
        InputSpec::i32(&[1]),
    )
    .unwrap();
    let cross_k = prefill.cross_k().unwrap();
    step.cross_k_mut().unwrap().copy_region_from(0, cross_k, 0, cross_k.size()).unwrap();
    let cross_v = prefill.cross_v().unwrap();
    step.cross_v_mut().unwrap().copy_region_from(0, cross_v, 0, cross_v.size()).unwrap();
    step.token_mut().unwrap().copyin(bytemuck::cast_slice(&[5i32])).unwrap();
    step.execute().unwrap();
    assert_eq!(step.logits().unwrap().size(), dims.n_vocab * size_of::<f32>());
}

#[test]
fn alignment_forward_exports_only_selected_heads() {
    let dims = make_dims();
    let model = Whisper::empty(dims.clone());
    let features = Tensor::zeros(&[2, 8, dims.n_text_state], DType::Float32);
    let tokens = Tensor::from_slice([50363i32, 50364, 50359, 50257, 50363, 50364, 50359, 50257])
        .try_reshape([2usize, 4])
        .unwrap();
    let heads = &[(2, 2), (3, 0)];

    let (cross_k, cross_v) = model.project_cross_kv(&features).unwrap();
    let qk = model.align_with_cross_kv(&tokens, &cross_k, &cross_v, heads).unwrap();
    let shape = qk.shape().unwrap();
    assert_eq!(shape[0].as_const(), Some(2));
    assert_eq!(shape[1].as_const(), Some(heads.len()));
    assert_eq!(shape[2].as_const(), Some(4));
    assert_eq!(shape[3].as_const(), Some(8));
}

#[test]
fn alignment_compute_does_not_inherit_cache_storage_dtype() {
    let mut dims = small_decoder_dims();
    dims.dtype = DType::Float16;
    let model = Whisper::empty(dims.clone());
    let features = Tensor::from_slice(
        (0..dims.n_audio_ctx * dims.n_text_state).map(|index| (index as f32 - 11.0) * 0.029).collect::<Vec<_>>(),
    )
    .try_reshape([1usize, dims.n_audio_ctx, dims.n_text_state])
    .unwrap();
    let tokens = Tensor::from_slice([1i32, 2, 3, 4]).try_reshape([1usize, 4]).unwrap();
    let heads = [(0, 0), (1, 1)];
    let (cross_k, cross_v) = model.project_cross_kv(&features).unwrap();
    assert_eq!(cross_k.dtype(), dims.cache_dtype());

    // Storage width must not reach the compute dtype, so a cache widened to f32
    // has to give exactly what the natively-stored one gives. Narrowing instead
    // would be a no-op now that the cache is already the narrow type.
    let wide_k = cross_k.cast(DType::Float32);
    let wide_v = cross_v.cast(DType::Float32);
    let actual = model.align_with_cross_kv(&tokens, &wide_k, &wide_v, &heads).unwrap();
    let expected = model.align_with_cross_kv(&tokens, &cross_k, &cross_v, &heads).unwrap();
    Tensor::realize_batch([&actual, &expected]).unwrap();

    let actual = actual.as_vec::<f32>().unwrap();
    let expected = expected.as_vec::<f32>().unwrap();
    let max_delta = actual.iter().zip(&expected).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(max_delta < 1e-5, "FP32 cache storage changed low-precision alignment compute by {max_delta}");
}

fn eager_audio_feature_alignment_reference(
    model: &Whisper,
    tokens: &Tensor,
    features: &Tensor,
    alignment_heads: &[(usize, usize)],
) -> Tensor {
    let seq_len = tokens.dim_const(1).unwrap();
    let tok_emb = model.decoder.token_embedding.embedding(tokens).unwrap();
    let pos_emb = model.decoder.positional_embedding.try_shrink([Some((0isize, seq_len as isize)), None]).unwrap();
    let mut x = tok_emb.try_add(&pos_emb).unwrap().cast(features.dtype());
    let mask = Tensor::causal_mask(seq_len, x.dtype()).unwrap();
    let mut selected: Vec<Option<Tensor>> = (0..alignment_heads.len()).map(|_| None).collect();

    for (layer, block) in model.decoder.blocks.iter().enumerate() {
        let heads = block.cross_attn.n_head;
        let h = block.attn_ln.forward(&x).unwrap();
        x = x.try_add(block.attn.forward(&h, None, Some(&mask)).unwrap()).unwrap();

        let h = block.cross_attn_ln.forward(&x).unwrap();
        x = x.try_add(block.cross_attn.forward(&h, Some(features), None).unwrap()).unwrap();
        let q = linear_forward(&block.cross_attn.query, &h).unwrap().split_heads(heads).unwrap();
        let k = linear_forward(&block.cross_attn.key, features).unwrap().split_heads(heads).unwrap();
        for (index, &(_, head)) in alignment_heads.iter().enumerate().filter(|&(_, &(l, _))| l == layer) {
            let q = q.narrow(1, head, 1usize).unwrap();
            let k = k.narrow(1, head, 1usize).unwrap();
            let scores = q.matmul(&k.try_transpose(-1, -2).unwrap()).unwrap();
            let scale = ((model.decoder.n_state / model.decoder.n_head) as f64).sqrt().recip();
            selected[index] = Some(scores.try_mul(scale).unwrap());
        }

        let h = block.mlp_ln.forward(&x).unwrap();
        let h = linear_forward(&block.mlp0, &h).unwrap().gelu_exact().unwrap();
        let h = linear_forward(&block.mlp2, &h).unwrap();
        x = x.try_add(&h).unwrap();
    }

    let selected: Vec<_> = selected.into_iter().map(Option::unwrap).collect();
    Tensor::cat(&selected.iter().collect::<Vec<_>>(), 1).unwrap().cast(DType::Float32)
}

#[test]
fn cached_cross_alignment_matches_audio_feature_reference() {
    let dims = small_decoder_dims();
    let model = Whisper::empty(dims.clone());
    let values: Vec<f32> =
        (0..2 * dims.n_audio_ctx * dims.n_text_state).map(|index| (index as f32 - 17.0) * 0.013).collect();
    let features = Tensor::from_slice(values).try_reshape([2usize, dims.n_audio_ctx, dims.n_text_state]).unwrap();
    let tokens = Tensor::from_slice([1i32, 2, 3, 4, 4, 3, 2, 1]).try_reshape([2usize, 4]).unwrap();
    let heads = [(1, 1), (0, 0)];

    let reference = eager_audio_feature_alignment_reference(&model, &tokens, &features, &heads);
    let (cross_k, cross_v) = model.project_cross_kv(&features).unwrap();
    let cached = model.align_with_cross_kv(&tokens, &cross_k, &cross_v, &heads).unwrap();
    Tensor::realize_batch([&reference, &cached]).unwrap();
    let reference = reference.as_vec::<f32>().unwrap();
    let cached = cached.as_vec::<f32>().unwrap();
    let max_delta = reference.iter().zip(&cached).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(max_delta < 1e-5, "cached-cross alignment drifted by {max_delta}");
}

#[test]
#[ignore = "heavy: prepares the prefill and alignment graphs through the CPU backend"]
fn recognition_cross_kv_seeds_prefill_and_alignment_device_locally() {
    let dims = small_decoder_dims();
    let model = Whisper::empty(dims.clone());
    let cross_spec = || InputSpec::new(&cache_shape(&dims, 1, dims.n_audio_ctx), dims.cache_dtype()).device_local();

    let mut prefill = WhisperPrefillJit::new(model.clone());
    prefill
        .prepare(
            InputSpec::i32(&[1, 3]),
            InputSpec::new(&[1, dims.n_audio_ctx, dims.n_text_state], dims.dtype.clone()).device_local(),
        )
        .unwrap();
    let alignment_model = WhisperAlignmentModel::new(model, vec![(0, 0), (1, 1)]);
    let mut alignment = WhisperAlignmentJit::new(alignment_model);
    alignment.prepare(cross_spec(), cross_spec(), InputSpec::i32(&[1, 3])).unwrap();

    prefill.tokens_mut().unwrap().copyin(bytemuck::cast_slice(&[1i32, 2, 3])).unwrap();
    prefill.execute().unwrap();

    // Seed the aligner from prefill's own device-local cross caches.
    let cross_k = prefill.cross_k().unwrap();
    alignment.cross_k_mut().unwrap().copy_region_from(0, cross_k, 0, cross_k.size()).unwrap();
    let cross_v = prefill.cross_v().unwrap();
    alignment.cross_v_mut().unwrap().copy_region_from(0, cross_v, 0, cross_v.size()).unwrap();
    alignment.tokens_mut().unwrap().copyin(bytemuck::cast_slice(&[1i32, 2, 3])).unwrap();
    alignment.execute().unwrap();

    assert_eq!(prefill.logits().unwrap().size(), 3 * dims.n_vocab * size_of::<f32>());
    assert_eq!(alignment.output().unwrap().size(), 2 * 3 * dims.n_audio_ctx * size_of::<f32>());
}

#[test]
fn state_dict_round_trip() {
    let dims = make_dims();
    let model = Whisper::empty(dims.clone());

    let sd = model.state_dict("");

    // Verify some key names
    assert!(sd.contains_key("encoder.conv1.weight"));
    assert!(sd.contains_key("encoder.conv1.bias"));
    assert!(sd.contains_key("encoder.conv2.weight"));
    assert!(sd.contains_key("encoder.positional_embedding"));
    assert!(sd.contains_key("encoder.blocks.0.attn.query.weight"));
    assert!(sd.contains_key("encoder.blocks.0.attn.query.bias"));
    assert!(sd.contains_key("encoder.blocks.0.attn.key.weight"));
    assert!(!sd.contains_key("encoder.blocks.0.attn.key.bias"));
    assert!(sd.contains_key("encoder.blocks.0.attn.value.weight"));
    assert!(sd.contains_key("encoder.blocks.0.attn.value.bias"));
    assert!(sd.contains_key("encoder.blocks.0.attn.out.weight"));
    assert!(sd.contains_key("encoder.blocks.0.attn.out.bias"));
    assert!(sd.contains_key("encoder.blocks.0.attn_ln.weight"));
    assert!(sd.contains_key("encoder.blocks.0.mlp.0.weight"));
    assert!(sd.contains_key("encoder.blocks.0.mlp.2.weight"));
    assert!(sd.contains_key("encoder.blocks.0.mlp_ln.weight"));
    assert!(sd.contains_key("encoder.ln_post.weight"));
    assert!(sd.contains_key("decoder.token_embedding.weight"));
    assert!(sd.contains_key("decoder.positional_embedding"));
    assert!(sd.contains_key("decoder.blocks.0.attn.query.weight"));
    assert!(sd.contains_key("decoder.blocks.0.cross_attn.query.weight"));
    assert!(sd.contains_key("decoder.blocks.0.cross_attn.key.weight"));
    assert!(sd.contains_key("decoder.blocks.0.mlp.0.weight"));
    assert!(sd.contains_key("decoder.ln.weight"));

    // Reload into a fresh model
    let mut model2 = Whisper::empty(dims);
    model2.load_state_dict(&sd, "").unwrap();
}

/// The key set `#[derive(Module)]` emits, at both the root and a nested
/// prefix. Whisper loads at the empty prefix, so a stray leading dot would
/// silently break every checkpoint.
fn expected_state_dict_keys(dims: &ModelDimensions, prefix: &str) -> BTreeSet<String> {
    let join = |a: &str, b: &str| if a.is_empty() { b.to_string() } else { format!("{a}.{b}") };
    let mut keys = BTreeSet::new();
    let mut push = |key: String| {
        keys.insert(join(prefix, &key));
    };
    for (tower, layers) in [("encoder", dims.n_audio_layer), ("decoder", dims.n_text_layer)] {
        push(format!("{tower}.positional_embedding"));
        for index in 0..layers {
            let block = format!("{tower}.blocks.{index}");
            let attentions: &[&str] = if tower == "encoder" { &["attn"] } else { &["attn", "cross_attn"] };
            for attn in attentions {
                for projection in ["query", "value", "out"] {
                    push(format!("{block}.{attn}.{projection}.weight"));
                    push(format!("{block}.{attn}.{projection}.bias"));
                }
                push(format!("{block}.{attn}.key.weight"));
                push(format!("{block}.{attn}_ln.weight"));
                push(format!("{block}.{attn}_ln.bias"));
            }
            for mlp in ["mlp.0", "mlp.2"] {
                push(format!("{block}.{mlp}.weight"));
                push(format!("{block}.{mlp}.bias"));
            }
            push(format!("{block}.mlp_ln.weight"));
            push(format!("{block}.mlp_ln.bias"));
        }
    }
    for conv in ["encoder.conv1", "encoder.conv2"] {
        push(format!("{conv}.weight"));
        push(format!("{conv}.bias"));
    }
    for norm in ["encoder.ln_post", "decoder.ln"] {
        push(format!("{norm}.weight"));
        push(format!("{norm}.bias"));
    }
    push("decoder.token_embedding.weight".into());
    keys
}

#[test_case(""; "root prefix")]
#[test_case("m"; "nested prefix")]
fn derived_state_dict_keys_match_the_hand_written_impls(prefix: &str) {
    let dims = small_decoder_dims();
    let model = Whisper::empty(dims.clone());
    let keys: BTreeSet<String> = model.state_dict(prefix).into_keys().collect();
    assert_eq!(keys, expected_state_dict_keys(&dims, prefix));
}

#[test]
fn dims_table() {
    let tiny = ModelDimensions::for_size(WhisperSize::Tiny);
    assert_eq!(tiny.n_audio_state, 384);
    assert_eq!(tiny.n_audio_head, 6);
    assert_eq!(tiny.n_audio_layer, 4);
    assert!(tiny.is_multilingual()); // "tiny" (non-.en) is multilingual

    let tiny_en = ModelDimensions::for_size(WhisperSize::TinyEn);
    assert!(!tiny_en.is_multilingual());

    let base = ModelDimensions::for_size(WhisperSize::Base);
    assert_eq!(base.n_audio_state, 512);
    assert_eq!(base.n_audio_head, 8);

    let large_v3 = ModelDimensions::for_size(WhisperSize::LargeV3);
    assert_eq!(large_v3.n_audio_state, 1280);
    assert_eq!(large_v3.n_mels, 128);
    assert_eq!(large_v3.n_vocab, 51866);

    // Turbo is large-v3's encoder with a four-layer decoder distilled onto it.
    let turbo = ModelDimensions::for_size(WhisperSize::Turbo);
    assert_eq!(turbo.n_audio_layer, 32);
    assert_eq!(turbo.n_text_layer, 4);
    assert_eq!(turbo.n_audio_state, 1280);
}

#[test]
fn alignment_heads_nonempty() {
    for size in [
        WhisperSize::Tiny,
        WhisperSize::Base,
        WhisperSize::Small,
        WhisperSize::Medium,
        WhisperSize::LargeV3,
        WhisperSize::Turbo,
    ] {
        let heads = size.alignment_heads();
        assert!(!heads.is_empty(), "{:?} has no alignment heads", size);
    }
}

#[test]
fn prepared_plan_has_concrete_nonzero_capacities() {
    let dims = ModelDimensions::for_size(WhisperSize::LargeV2);
    let plan = WhisperPlan::for_model(&dims, WhisperSize::LargeV2);
    assert!(plan.encoder_batch > 0);
    assert!(plan.decoder_slots > 0);
    assert!(plan.alignment_batch > 0);
    assert!(plan.alignment_batch <= plan.encoder_batch);
    plan.validate().unwrap();
}

/// The replay's rows come in whole tensor-core tiles: 224 text tokens after a
/// multilingual prompt would be 229 rows, which the relaxed tensor-core level
/// lowers to the scalar path; the context bounds the rounding.
#[test_case(224, 3, 448, 240; "the default budget rounds up to a tile")]
#[test_case(11, 3, 448, 16; "a short budget is one tile")]
#[test_case(444, 3, 448, 448; "the context caps the rounding")]
fn alignment_replay_rows_are_whole_tensor_core_tiles(
    max_tokens: usize,
    prompt_len: usize,
    n_text_ctx: usize,
    rows: usize,
) {
    assert_eq!(crate::whisper::aligner::replay_rows(max_tokens, prompt_len, n_text_ctx), rows);
}

#[test]
fn default_decode_policy_is_explicit_openai_fallback() {
    let options = DecodeOptions::default();
    assert_eq!(options.strategy, DecodeStrategy::Beam { size: 5 });
    assert_eq!(options.fallback_temperatures, [0.2, 0.4, 0.6, 0.8, 1.0]);
    assert_eq!(options.compression_ratio_threshold, Some(2.4));
    assert_eq!(options.logprob_threshold, Some(-1.0));
}

#[test]
fn decode_policy_rejects_invalid_geometry_and_temperatures() {
    let invalid_beam = DecodeOptions { strategy: DecodeStrategy::Beam { size: 0 }, ..Default::default() };
    assert!(invalid_beam.validate().is_err());

    let invalid_sample = DecodeOptions { strategy: DecodeStrategy::Sample { temperature: 0.0 }, ..Default::default() };
    assert!(invalid_sample.validate().is_err());

    let invalid_fallback = DecodeOptions { fallback_temperatures: vec![f32::NAN], ..Default::default() };
    assert!(invalid_fallback.validate().is_err());

    let invalid_compression = DecodeOptions { compression_ratio_threshold: Some(0.0), ..Default::default() };
    assert!(invalid_compression.validate().is_err());

    let invalid_logprob = DecodeOptions { logprob_threshold: Some(f32::NEG_INFINITY), ..Default::default() };
    assert!(invalid_logprob.validate().is_err());

    let invalid_silence = DecodeOptions { no_speech_threshold: Some(1.1), ..Default::default() };
    assert!(invalid_silence.validate().is_err());

    // An empty temperature list disables fallback; it is not a geometry error.
    let no_fallback = DecodeOptions { fallback_temperatures: Vec::new(), ..Default::default() };
    no_fallback.validate().unwrap();
}

#[test]
fn no_speech_skip_respects_logprob_override() {
    let options = DecodeOptions::default();
    let mut result = DecodeResult {
        tokens: vec![1, 2],
        token_probs: vec![0.2, 0.3],
        text: "hallucination".to_string(),
        avg_logprob: -2.0,
        no_speech_prob: 0.9,
        temperature: 0.0,
        compression_ratio: 1.0,
        language: Some("en".to_string()),
    };
    assert!(result.should_skip(&options));
    result.clear_speech();
    assert!(result.tokens.is_empty());
    assert!(result.text.is_empty());

    result.avg_logprob = -0.5;
    assert!(!result.should_skip(&options));
}

#[test]
fn confident_silence_cancels_quality_fallback() {
    let options = DecodeOptions::default();
    let result = DecodeResult {
        tokens: Vec::new(),
        token_probs: Vec::new(),
        text: String::new(),
        avg_logprob: -2.0,
        no_speech_prob: 0.9,
        temperature: 0.0,
        compression_ratio: 3.0,
        language: Some("en".to_string()),
    };
    assert!(!crate::whisper::decode::check_fallback(&result, &options));
}
