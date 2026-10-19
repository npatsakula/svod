//! Fixed-capacity decoder-step graph tests.

use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::whisper::decoder::{StepAttentionMode, cached_step_mask};
use crate::whisper::{ModelDimensions, Whisper, WhisperSize};

/// Tiny config so the CPU JIT graph compiles in seconds. `n_text_ctx` is kept
/// small (8) to shrink the self-attention buffers — the step JIT only needs
/// one position of cache populated for this test.
fn tiny_dims() -> ModelDimensions {
    // Start from WhisperSize::Tiny's structural dims, but shrink the text
    // context and vocab so the compile graph is minimal. The step JIT's cache
    // buffers scale with n_text_ctx.
    let mut dims = ModelDimensions::for_size(WhisperSize::Tiny);
    dims.n_text_ctx = 8;
    dims.n_vocab = 64;
    // The step concatenates the passed cache with the K/V it just projected, so
    // both live at `cache_dtype()`; f32 also buys the tolerances asserted below.
    dims.dtype = DType::Float32;
    dims
}

#[test]
fn forward_step_fixed_batch_keeps_batch_concrete() {
    let dims = tiny_dims();
    let model = Whisper::empty(dims.clone());
    let (batch, n_audio_ctx) = (2usize, 8usize);
    let d_head = dims.n_text_state / dims.n_text_head;
    let layer_heads = dims.n_text_layer * dims.n_text_head;
    let token = Tensor::zeros(&[batch, 1], DType::Int32);
    let self_k = Tensor::zeros(&[batch, dims.n_text_ctx, layer_heads, d_head], DType::Float32);
    let self_v = Tensor::zeros(&[batch, dims.n_text_ctx, layer_heads, d_head], DType::Float32);
    let cross_k = Tensor::zeros(&[batch, n_audio_ctx, layer_heads, d_head], DType::Float32);
    let cross_v = Tensor::zeros(&[batch, n_audio_ctx, layer_heads, d_head], DType::Float32);
    let key_lens = Tensor::zeros(&[batch], DType::Int32);
    // Identity: each row reads its own cross cache, the pre-sharing behaviour.
    let cross_map = Tensor::from_slice((0..batch as i32).collect::<Vec<_>>());

    let (logits, new_k, new_v) =
        model.decode_step(&token, &self_k, &self_v, &cross_k, &cross_v, &key_lens, &cross_map).unwrap();
    assert_eq!(logits.dim_const(0).unwrap(), batch);
    assert_eq!(new_k.dim_const(0).unwrap(), batch);
    assert_eq!(new_v.dim_const(0).unwrap(), batch);
    assert!(logits.to_vec::<f32>().unwrap().into_iter().all(f32::is_finite));
}

/// Every lane of one beam attempt shares its owner's cross cache, so pointing
/// both lanes at row 0 must equal physically replicating row 0 into both. The
/// tile kernel reads the map with an index load and the generic path with a
/// gather; this pins the generic path, which is all a CPU or Metal host has.
#[test]
fn shared_cross_cache_map_matches_a_replicated_cache() {
    let dims = tiny_dims();
    let model = Whisper::empty(dims.clone());
    let (batch, n_audio_ctx) = (2usize, 8usize);
    let d_head = dims.n_text_state / dims.n_text_head;
    let layer_heads = dims.n_text_layer * dims.n_text_head;
    let token = Tensor::from_slice([1i32, 2]).try_reshape([batch, 1]).unwrap();
    let self_k = Tensor::randn(&[batch, dims.n_text_ctx, layer_heads, d_head]).unwrap();
    let self_v = Tensor::randn(&[batch, dims.n_text_ctx, layer_heads, d_head]).unwrap();
    let key_lens = Tensor::from_slice([2i32, 5]);

    let cross_k = Tensor::randn(&[batch, n_audio_ctx, layer_heads, d_head]).unwrap();
    let cross_v = Tensor::randn(&[batch, n_audio_ctx, layer_heads, d_head]).unwrap();
    let owner = |cache: &Tensor| {
        let row = cache.narrow(0, 0, 1).unwrap();
        Tensor::cat(&[&row, &row], 0).unwrap()
    };

    let step = |ck: &Tensor, cv: &Tensor, map: &[i32]| {
        model.decode_step(&token, &self_k, &self_v, ck, cv, &key_lens, &Tensor::from_slice(map)).unwrap().0
    };
    let shared = step(&cross_k, &cross_v, &[0, 0]);
    let replicated = step(&owner(&cross_k), &owner(&cross_v), &[0, 1]);
    Tensor::realize_batch([&shared, &replicated]).unwrap();

    let (shared, replicated) = (shared.as_vec::<f32>().unwrap(), replicated.as_vec::<f32>().unwrap());
    let max_abs = shared.iter().zip(&replicated).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(max_abs < 1e-5, "a shared cross cache diverged from a replicated one by {max_abs:e}");
}

/// The cross cache is sized by concurrent windows, so it has fewer rows than the
/// decoder has lanes. A one-row cache read by both lanes must equal a two-row
/// cache holding the same bytes twice -- on the tile path, which resolves the row
/// with an index load, and on the generic path, which gathers.
#[test]
fn cross_cache_with_fewer_rows_than_lanes_matches_a_per_lane_cache() {
    let dims = tiny_dims();
    let model = Whisper::empty(dims.clone());
    let (batch, n_audio_ctx) = (2usize, 8usize);
    let d_head = dims.n_text_state / dims.n_text_head;
    let layer_heads = dims.n_text_layer * dims.n_text_head;
    let token = Tensor::from_slice([1i32, 2]).try_reshape([batch, 1]).unwrap();
    let self_k = Tensor::randn(&[batch, dims.n_text_ctx, layer_heads, d_head]).unwrap();
    let self_v = Tensor::randn(&[batch, dims.n_text_ctx, layer_heads, d_head]).unwrap();
    let key_lens = Tensor::from_slice([2i32, 5]);

    let narrow_k = Tensor::randn(&[1, n_audio_ctx, layer_heads, d_head]).unwrap();
    let narrow_v = Tensor::randn(&[1, n_audio_ctx, layer_heads, d_head]).unwrap();
    let twice = |cache: &Tensor| Tensor::cat(&[cache, cache], 0).unwrap();

    let step = |ck: &Tensor, cv: &Tensor, map: &[i32]| {
        model.decode_step(&token, &self_k, &self_v, ck, cv, &key_lens, &Tensor::from_slice(map)).unwrap().0
    };
    let one_row = step(&narrow_k, &narrow_v, &[0, 0]);
    let two_rows = step(&twice(&narrow_k), &twice(&narrow_v), &[0, 1]);
    Tensor::realize_batch([&one_row, &two_rows]).unwrap();

    let (one_row, two_rows) = (one_row.as_vec::<f32>().unwrap(), two_rows.as_vec::<f32>().unwrap());
    let max_abs = one_row.iter().zip(&two_rows).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(max_abs < 1e-5, "a cache narrower than the lane count diverged by {max_abs:e}");
}

/// `true` = attend: the cached prefix each lane filled, plus the key this step
/// appended at the end of the cache.
#[test]
fn cached_step_key_lengths_admit_only_prefix_and_appended_key() {
    let key_lens = Tensor::from_slice([0i32, 3]);
    let valid = cached_step_mask(&key_lens, 6).unwrap();
    assert_eq!(valid.dims().unwrap(), [2, 6]);
    assert_eq!(
        valid.to_vec::<bool>().unwrap(),
        [false, false, false, false, false, true, true, true, true, false, false, true]
    );
}

#[test]
#[ignore = "GPU: the op layer's cached self/cross attention vs generic SDPA on a tk3 device"]
fn decoder_step_attention_modes_match_generic_gpu_sdpa() {
    let device = Tensor::empty(&[1], DType::Float32).device();
    if !svod_tk3::ops::supported(&device) {
        eprintln!("skip: {device:?} has no tk3 kernels");
        return;
    }

    let mut dims = tiny_dims();
    // Production cross-attention length: split=4 creates 375-key chunks, which
    // also exercises a ragged last key block. The kernels take 16-bit caches
    // only, so the step runs in f16.
    dims.n_audio_ctx = 1500;
    dims.n_text_ctx = 7;
    dims.dtype = DType::Float16;
    let model = Whisper::empty(dims.clone());
    let (batch, d_head) = (2, dims.n_text_state / dims.n_text_head);
    let layer_heads = dims.n_text_layer * dims.n_text_head;
    let token = Tensor::from_slice([1i32, 2]).try_reshape([batch, 1]).unwrap();
    let cache =
        |rows: usize, len: usize| Tensor::randn(&[rows, len, layer_heads, d_head]).unwrap().cast(DType::Float16);
    let (self_k, self_v) = (cache(batch, dims.n_text_ctx), cache(batch, dims.n_text_ctx));
    let (cross_k, cross_v) = (cache(batch, dims.n_audio_ctx), cache(batch, dims.n_audio_ctx));
    // Identity gives each row its own cross cache; `[0, 0]` is one beam attempt
    // whose lanes share the owner's. The kernel resolves the row with an index
    // load and the generic path with a gather, so both maps must agree. The
    // self attention reads the prefix each row filled and scores the row this
    // step projected separately, so the lengths span a row that has decoded
    // nothing and one whose prefix fills the cache.
    let modes = [
        StepAttentionMode::Generic,
        StepAttentionMode::OpSelf,
        StepAttentionMode::OpCross { split: 1 },
        StepAttentionMode::OpCross { split: 4 },
        StepAttentionMode::OpBoth { split: 1 },
        StepAttentionMode::OpBoth { split: 4 },
    ];
    for (map, lens) in [(vec![0i32, 1], [2i32, 5]), (vec![0i32, 0], [2i32, 5]), (vec![0i32, 1], [0i32, 7])] {
        let key_lens = Tensor::from_slice(lens);
        let cross_map = Tensor::from_slice(map.clone());
        let outputs = modes.map(|mode| {
            model
                .decoder
                .forward_step_with_attention_mode(
                    &token, &self_k, &self_v, &cross_k, &cross_v, &key_lens, &cross_map, mode,
                )
                .unwrap()
                .0
        });
        Tensor::realize_batch(outputs.iter()).unwrap();
        let reference = outputs[0].as_vec::<f32>().unwrap();
        let scale = reference.iter().fold(0f32, |m, x| m.max(x.abs()));
        for (mode, output) in modes.into_iter().zip(&outputs).skip(1) {
            let got = output.as_vec::<f32>().unwrap();
            let max_abs = got.iter().zip(&reference).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            // Both paths round the attention output to f16 before the out
            // projection: a few f16 ulps of the logits' range.
            assert!(
                max_abs < 4e-3 * scale,
                "{mode:?} logits differ from generic SDPA by {max_abs:e} (range {scale:e}) under map {map:?} and lengths {lens:?}"
            );
        }
    }
}
