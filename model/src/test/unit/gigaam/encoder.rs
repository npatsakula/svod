//! GigaAM encoder graph tests: the residual stream stays in the compute dtype
//! through every layer norm, the RoPE tables and weights are realized once at
//! load, and the f16 encoder tracks its f32 reference. All on the CPU backend.

use std::collections::BTreeSet;
use std::sync::Arc;

use svod_dtype::{DType, ScalarDType};
use svod_ir::origin::{self, OriginFrame, OriginId};
use svod_ir::{Op, UOp, UnaryOp, ops};
use svod_runtime::{ExecutionPlan, PreparedKernel};
use svod_tensor::Tensor;
use svod_tensor::nn::Module as _;

use crate::gigaam::{GigaAm, GigaAmConfig, Head};
use crate::state::StateDict;

/// Two layers over 16 subsampled frames: every scope of the real model, at a
/// size the CPU backend compiles in seconds. Head dim 16 is what the hand
/// flash-attention kernel accepts on a GPU (it declines the short sequence
/// and the SDPA fallback runs, as on the CPU).
fn config() -> GigaAmConfig {
    let mut cfg = super::super::batch::test_config();
    cfg.max_batch_size = 2;
    cfg.n_heads = 2;
    cfg.max_mel_frames = 64;
    cfg.max_encoder_frames = 16;
    cfg
}

/// A model's parameters keyed the way `GigaAm::from_state_dict` loads them.
pub(super) fn state_dict(model: &GigaAm) -> StateDict {
    let mut sd = model.encoder.subsampling.state_dict("subsampling");
    for (index, layer) in model.encoder.layers.iter().enumerate() {
        sd.extend(layer.state_dict(&format!("layers.{index}")));
    }
    match &model.head {
        Head::Ctc(head) => sd.extend(head.state_dict("head")),
        Head::Rnnt { head, .. } => sd.extend(head.state_dict("head")),
    }
    sd
}

/// One random parameter set loaded at each of `dtypes`, in order.
fn models<const N: usize>(cfg: &GigaAmConfig, dtypes: [DType; N]) -> [GigaAm; N] {
    let sd = state_dict(&GigaAm::with_random_weights(cfg.clone()));
    dtypes.map(|dtype| GigaAm::from_state_dict_with_encoder_dtype(&sd, cfg.clone(), None, dtype).expect("load"))
}

/// Deterministic mel input in `[-1, 1)` plus lengths: lane 0 full, lane 1 half padded.
fn input(cfg: &GigaAmConfig) -> (Tensor, Tensor) {
    let (b, n_mels, t_mel) = (cfg.max_batch_size, cfg.n_mels, cfg.max_mel_frames);
    let mel: Vec<f32> = (0..b * n_mels * t_mel).map(|i| ((i * 7919) % 1000) as f32 / 500.0 - 1.0).collect();
    let mel = Tensor::from_slice(mel).try_reshape([b as isize, n_mels as isize, t_mel as isize]).expect("mel shape");
    (mel, Tensor::from_slice([t_mel as i32, (t_mel / 2) as i32]))
}

fn prepare(model: &GigaAm, cfg: &GigaAmConfig) -> ExecutionPlan {
    let (mel, lengths) = input(cfg);
    model.encoder.forward_batch(&mel, &lengths).expect("forward").prepare().expect("prepare")
}

/// The module scopes of an origin, dropping the call frames beneath them.
fn module_path(id: OriginId) -> String {
    let names = origin::chain(id).into_iter().filter_map(origin::get).filter_map(|origin| match origin.frame {
        OriginFrame::Module { name } => Some(name),
        _ => None,
    });
    names.collect::<Vec<_>>().join(".")
}

/// Kernels that fused an op built directly under `layers.<n>.<scope>` — not
/// under a nested scope such as `layers.<n>.<scope>.norm`.
fn kernels_under<'a>(plan: &'a ExecutionPlan, scope: &str) -> Vec<&'a PreparedKernel> {
    let direct = |id: &OriginId| {
        let path = module_path(*id);
        let parts: Vec<&str> = path.split('.').collect();
        parts.len() == 3 && parts[0] == "layers" && parts[2] == scope
    };
    plan.prepared_kernels().into_iter().filter(|kernel| kernel.origins.iter().any(direct)).collect()
}

/// Whether an address indexes a kernel argument, as opposed to local scratch.
fn addresses_argument(index: &Arc<UOp>) -> bool {
    let buffer = match index.op() {
        Op::Index(ops::Index { buffer, .. }) => buffer,
        Op::Shrink(ops::Shrink { src, .. }) => src,
        _ => return false,
    };
    matches!(buffer.op(), Op::Param(..))
}

/// Element dtypes of every store a kernel makes to its arguments.
fn argument_store_dtypes(kernel: &PreparedKernel) -> BTreeSet<ScalarDType> {
    let stores = kernel.ast.toposort().into_iter().filter_map(|uop| match uop.op() {
        Op::Store(ops::Store { index, value, .. }) if addresses_argument(index) => Some(value.dtype().base()),
        _ => None,
    });
    stores.collect()
}

/// The kernels producing a layer norm's input (the depthwise conv under `conv`,
/// the FFN2 GEMM under `ffn2`) store f16: the norm's own f32 cast is not fused
/// back into them, which would widen a `[B, T, d_model]` store and every read
/// of it. Pins the `.contiguous()` on both norm inputs in the encoder.
#[test]
fn layer_norm_inputs_stay_in_the_compute_dtype() {
    let _capture = origin::capture_for_thread(true);
    let cfg = config();
    let [model] = models(&cfg, [DType::Float16]);
    let plan = prepare(&model, &cfg);

    for scope in ["conv", "ffn2"] {
        let kernels = kernels_under(&plan, scope);
        assert!(kernels.len() >= 2 * cfg.n_layers, "{scope}: {} kernels for {} layers", kernels.len(), cfg.n_layers);
        for kernel in kernels {
            let dtypes = argument_store_dtypes(kernel);
            assert_eq!(
                dtypes,
                BTreeSet::from([ScalarDType::Float16]),
                "{scope} kernel {} stores {dtypes:?}",
                kernel.kernel.entry_point
            );
        }
    }
}

/// The RoPE tables are realized when the encoder is built, whichever way it is
/// built: the encoder plan reads them as inputs, writes them nowhere, and no
/// kernel of it evaluates `sin`.
#[test]
fn rope_tables_are_realized_at_load() {
    let cfg = config();
    let [loaded] = models(&cfg, [DType::Float16]);
    for model in [GigaAm::with_random_weights(cfg.clone()), loaded] {
        let plan = prepare(&model, &cfg);
        let buffers = plan.buffers();
        for (name, table) in [("cos", &model.encoder.cos_cache), ("sin", &model.encoder.sin_cache)] {
            let id = table.buffer().unwrap_or_else(|| panic!("{name} table is lazy")).id();
            assert_eq!(table.dtype(), DType::Float32, "{name} table dtype");
            let slot = buffers.iter().position(|buffer| buffer.id() == id);
            let slot = slot.unwrap_or_else(|| panic!("{name} table is not a plan input"));
            let kernels = plan.prepared_kernels();
            let writers = kernels.iter().filter(|k| k.output_indices.iter().any(|&o| k.buffer_indices[o] == slot));
            let readers = kernels.iter().filter(|k| k.input_indices.iter().any(|&i| k.buffer_indices[i] == slot));
            assert_eq!(writers.count(), 0, "a kernel recomputes the {name} table");
            assert!(readers.count() >= cfg.n_layers, "fewer than one {name} reader per layer");
        }
        // `sin(`, not `sin`: MSL's prelude is `using namespace metal;`, and "u-sin-g"
        // matches a bare substring test in every Metal kernel ever rendered.
        let sines = plan.kernels().filter(|kernel| kernel.code.contains("sin(")).count();
        let sin_ops = plan
            .prepared_kernels()
            .iter()
            .filter(|k| k.ast.toposort().iter().any(|u| matches!(u.op(), Op::Unary(UnaryOp::Sin, _))))
            .count();
        assert_eq!((sines, sin_ops), (0, 0), "an encoder kernel evaluates sin");
    }
}

/// Every encoder parameter is a realized buffer in the compute dtype, so no
/// encoder call re-casts a checkpoint weight.
#[test]
fn encoder_weights_load_as_realized_buffers() {
    let cfg = config();
    let [model] = models(&cfg, [DType::Float16]);
    let encoder_keys = state_dict(&model).into_iter().filter(|(key, _)| !key.starts_with("head."));
    for (key, weight) in encoder_keys {
        let buffer = weight.buffer().unwrap_or_else(|| panic!("{key} is lazy"));
        assert_eq!(buffer.dtype(), DType::Float16, "{key}");
    }
}

/// The f16 encoder against the f32 encoder on the same parameters and input,
/// over the valid frames of both lanes. The residual is rounded to f16 before
/// every norm (as PyTorch fp16 does), so the whole stack carries f16 rounding
/// (unit roundoff 4.9e-4) through two layers of unit-variance activations. The
/// measured gap on this graph is 3.0e-3 max and 7e-4 mean; the bounds leave
/// 4-10x headroom and stay two decades under the O(1) gap of a lost mask or a
/// wrong fusion.
#[test]
fn f16_encoder_tracks_the_f32_reference() {
    let cfg = config();
    let [f16, f32] = models(&cfg, [DType::Float16, DType::Float32]);
    let (mel, lengths) = input(&cfg);
    let run = |model: &GigaAm| -> Vec<f32> {
        let out = model.encoder.forward_batch(&mel, &lengths).expect("forward").cast(DType::Float32);
        out.to_vec::<f32>().expect("read")
    };
    let (half, full) = (run(&f16), run(&f32));
    assert_eq!(half.len(), full.len());

    let t_sub = f16.encoder.subsampling_output_length(cfg.max_mel_frames);
    let valid = [t_sub, f16.encoder.subsampling_output_length(cfg.max_mel_frames / 2)];
    let (mut max_diff, mut sum_diff, mut count) = (0f32, 0f32, 0usize);
    for (index, (a, b)) in half.iter().zip(&full).enumerate() {
        let (lane, frame) = (index / (cfg.d_model * t_sub), index % t_sub);
        if frame < valid[lane] {
            assert!(a.is_finite() && b.is_finite(), "non-finite output at {index}");
            let diff = (a - b).abs();
            max_diff = max_diff.max(diff);
            sum_diff += diff;
            count += 1;
        }
    }
    let mean_diff = sum_diff / count as f32;
    assert!(max_diff <= 3e-2, "max |f16 - f32| = {max_diff}");
    assert!(mean_diff <= 3e-3, "mean |f16 - f32| = {mean_diff}");
}
