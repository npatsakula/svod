use svod_dtype::{DType, DeviceSpec};
use svod_runtime::profiler::ProfileOptions;
use svod_tensor::Tensor;
use svod_tensor::default_device::default_device;
use svod_tensor::nn::Module;

use crate::state::cast_all;
use crate::wavlm::{WavLm, wavlm_large_s80_md};

/// The DiariZen backbone (wavlm-large-s80-md, random weights) in f16, eager,
/// on batch 8 × one 16 s window: best of three profiles of ten replays,
/// GPU ms, dispatches and the flash-attention share (run with
/// `--ignored --nocapture` on a GPU).
#[test]
#[ignore = "perf probe: needs a GPU"]
fn wavlm_forward_profile() {
    if matches!(default_device(), DeviceSpec::Cpu) {
        return;
    }
    let (batch, samples) = (8usize, 256_000usize);
    let mut cfg = wavlm_large_s80_md();
    cfg.max_batch_size = batch;
    let mut model = WavLm::empty(cfg);
    let sd = cast_all(&model.state_dict(""), DType::Float16);
    for t in sd.values() {
        t.realize().unwrap();
    }
    model.load_state_dict(&sd, "").unwrap();
    let data: Vec<f32> = (0..batch * samples).map(|i| (i as f32 * 0.37).sin() * 0.1).collect();
    let wav = Tensor::from_slice(&data).try_reshape([batch as isize, samples as isize]).unwrap();
    let wav = wav.cast(DType::Float16).contiguous();
    wav.realize().unwrap();

    let mut best: Option<(f64, usize, f64)> = None;
    for round in 0..3 {
        let out = model.extract_features_stacked(&wav).unwrap();
        let profile = out.profile(&ProfileOptions { iters: 10, ..ProfileOptions::default() }).unwrap();
        let kernels: Vec<_> = profile.stages.iter().flat_map(|s| &s.kernels).collect();
        let ms = |k: &&svod_runtime::profiler::KernelProfile| match (k.gpu_start_ns, k.gpu_end_ns) {
            (Some(s), Some(e)) => (e - s) as f64 * 1e-6,
            _ => 0.0,
        };
        let total: f64 = kernels.iter().map(ms).sum();
        let attn: f64 = kernels.iter().filter(|k| k.kernel.entry_point.contains("flash_attention")).map(ms).sum();
        eprintln!("round {round}: {total:.2} GPU ms, {} dispatches, flash attention {attn:.2} ms", kernels.len());
        if best.is_none_or(|(b, _, _)| total < b) {
            best = Some((total, kernels.len(), attn));
        }
        if round == 2 {
            eprintln!("{}", profile.render_table());
        }
    }
    let (total, dispatches, attn) = best.unwrap();
    eprintln!("best: {total:.2} GPU ms, {dispatches} dispatches, flash attention {attn:.2} ms");
}
