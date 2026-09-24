//! ModernBERT profiling harness — the forward's wall time plus a per-kernel,
//! origin-attributed profile of the JIT-compiled backbone
//! (`answerdotai/ModernBERT-base` from the HF cache), on deterministic in-vocab
//! ids.
//!
//! ```text
//! # 1×512 on CUDA, BEAM-tuned, one row per dispatch into a CSV:
//! BEAM=4 cargo run -p svod-model --release --example modernbert_profile -- \
//!     --device cuda --batch 1 --seq 512 --csv /tmp/mb.csv
//!
//! # A padded batch of 8, profile JSON:
//! cargo run -p svod-model --release --example modernbert_profile -- \
//!     --device cuda --batch 8 --seq 512 --pad --json /tmp/mb.json
//! ```

use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::Parser;
use svod_dtype::{DType, DeviceSpec};
use svod_model::jit::InputSpec;
use svod_model::modernbert::{ModernBert, ModernBertConfig, ModernBertJit};
use svod_runtime::{KernelProfile, OriginView, RunProfile, StageProfile, aggregate_origins, render_histogram};
use svod_tensor::{PrepareConfig, set_default_device};

#[derive(Parser, Debug)]
struct Args {
    /// HF Hub repository of the checkpoint.
    #[arg(long, default_value = "answerdotai/ModernBERT-base")]
    hub: String,
    /// Batch size; the JIT's batch variable is pinned to it unless `--dynamic-batch`.
    #[arg(long, default_value_t = 1)]
    batch: usize,
    /// Sequence length.
    #[arg(long, default_value_t = 512)]
    seq: usize,
    /// Timed forwards (and profiled ones, merged by minimum).
    #[arg(long, default_value_t = 5)]
    iters: usize,
    /// Untimed forwards before the timed ones.
    #[arg(long, default_value_t = 2)]
    warmup: usize,
    /// Right-pad rows to varying lengths (row b keeps `seq·(B−b)/B` tokens).
    #[arg(long)]
    pad: bool,
    /// Device to run on: `cpu`, `cuda`, `cuda:N`, `amd`, `amd:N`. Exported as
    /// SVOD_DEVICE before runtime init.
    #[arg(long, default_value = "cuda")]
    device: String,
    /// Depth the origin rollup groups scopes at.
    #[arg(long, value_name = "N")]
    origin_depth: Option<usize>,
    /// Write the merged profile as JSON.
    #[arg(long, value_name = "PATH")]
    json: Option<PathBuf>,
    /// Write the last-hidden-state (f32 LE, `(B, L, D)`) and the mask (i64 LE).
    #[arg(long, value_name = "PATH")]
    dump: Option<PathBuf>,
    /// Write one row per dispatch: index, kernel, GPU µs, est. flops/bytes, origins.
    #[arg(long, value_name = "PATH")]
    csv: Option<PathBuf>,
    /// Compute in f32 instead of the device default (the parity reference).
    #[arg(long)]
    f32: bool,
    /// Leave the batch variable free over `1..=batch` instead of pinning it.
    #[arg(long)]
    dynamic_batch: bool,
}

fn parse_device(spec: &str) -> Result<DeviceSpec, Box<dyn std::error::Error>> {
    let lower = spec.trim().to_ascii_lowercase();
    let (name, index) = match lower.split_once(':') {
        Some((name, index)) => (name, Some(index.parse::<usize>()?)),
        None => (lower.as_str(), None),
    };
    match name {
        "cpu" => Ok(DeviceSpec::Cpu),
        "cuda" | "gpu" => Ok(DeviceSpec::Cuda { device_id: index.unwrap_or(0) }),
        "amd" => Ok(DeviceSpec::Amd { device_id: index.unwrap_or(0) }),
        _ => Err(format!("unsupported --device {spec:?}; expected cpu, cuda, cuda:N, amd or amd:N").into()),
    }
}

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    if args.iters == 0 {
        return Err("--iters must be non-zero".into());
    }
    let device = parse_device(&args.device)?;
    // Safety: single-threaded prologue, before any runtime threads spawn.
    unsafe {
        std::env::set_var("SVOD_DEVICE", &args.device);
        if std::env::var("SVOD_ORIGIN").is_err() {
            std::env::set_var("SVOD_ORIGIN", "1");
        }
    }
    set_default_device(device.clone());

    let (b, l) = (args.batch, args.seq);
    let mut cfg = ModernBertConfig { max_batch_size: b, ..Default::default() };
    if args.f32 {
        cfg.dtype = DType::Float32;
    }
    let t = Instant::now();
    let model = ModernBert::from_hub(&args.hub, cfg)?;
    let (vocab, pad_id, dtype) = (model.config.vocab_size, model.config.pad_token_id, model.config.dtype.clone());
    let hidden = model.config.hidden_size;
    println!(
        "=== ModernBERT profile: {} B={b} L={l} dtype {dtype:?} pad {} BEAM={:?} ===",
        args.hub,
        args.pad,
        std::env::var("BEAM").ok()
    );
    println!("load: {:.1} ms", t.elapsed().as_secs_f64() * 1e3);

    let t = Instant::now();
    let mut jit = ModernBertJit::new(model);
    if !args.dynamic_batch {
        jit = jit.with_b_fixed(b);
    }
    jit.prepare_with_config(
        InputSpec::i64(&[b, l]).device_local(),
        InputSpec::i64(&[b, l]).device_local(),
        &PrepareConfig::device_local(),
    )?;
    println!("prepare+compile: {:.1} s", t.elapsed().as_secs_f64());

    // Deterministic in-vocab ids (LCG), clear of the special tokens at the top.
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut ids: Vec<i64> = (0..b * l)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            1000 + ((state >> 33) % (vocab as u64 - 2000)) as i64
        })
        .collect();
    let mut mask = vec![1i64; b * l];
    if args.pad {
        for row in 0..b {
            let keep = l * (b - row) / b;
            for pos in keep..l {
                ids[row * l + pos] = pad_id as i64;
                mask[row * l + pos] = 0;
            }
        }
    }
    jit.input_ids_mut()?.copyin(bytemuck::cast_slice(&ids))?;
    jit.attention_mask_mut()?.copyin(bytemuck::cast_slice(&mask))?;

    for _ in 0..args.warmup {
        jit.execute_bound(b as i64)?;
        jit.hidden()?.synchronize()?;
    }
    let mut forward = Vec::with_capacity(args.iters);
    for _ in 0..args.iters {
        let t = Instant::now();
        jit.execute_bound(b as i64)?;
        jit.hidden()?.synchronize()?;
        forward.push(t.elapsed());
    }
    let min = forward.iter().min().copied().unwrap_or_default();
    println!(
        "forward: median {:.3} ms, min {:.3} ms over {} iters",
        median(forward.clone()).as_secs_f64() * 1e3,
        min.as_secs_f64() * 1e3,
        args.iters
    );

    if let Some(path) = &args.dump {
        let buf = jit.hidden()?;
        let mut bytes = vec![0u8; buf.size()];
        buf.copyout(&mut bytes)?;
        let n = b * l * hidden;
        let floats: Vec<f32> = if dtype == DType::BFloat16 {
            // bf16 is the top half of an f32.
            bytes[..n * 2]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&c| f32::from_bits(u32::from(u16::from_le_bytes(c)) << 16))
                .collect()
        } else if dtype == DType::Float32 {
            bytes[..n * 4].as_chunks::<4>().0.iter().map(|&c| f32::from_le_bytes(c)).collect()
        } else {
            return Err(format!("--dump reads bf16 or f32 activations, not {dtype:?}").into());
        };
        std::fs::write(path, bytemuck::cast_slice(&floats))?;
        std::fs::write(path.with_extension("mask"), bytemuck::cast_slice(&mask))?;
        println!("dumped {} floats to {}", floats.len(), path.display());
    }

    let mut profile = None::<RunProfile>;
    for _ in 0..args.iters {
        let t = Instant::now();
        let kernels = jit.execute_profiled_static()?;
        let run = RunProfile {
            stages: vec![StageProfile::gpu("forward", t.elapsed(), kernels)],
            origin_depth: args.origin_depth,
        };
        match &mut profile {
            Some(merged) => merged.merge_min(run),
            None => profile = Some(run),
        }
    }
    let profile = profile.expect("iters is non-zero");
    let stage = profile.stage("forward").ok_or("no forward stage")?;
    let total: Duration = stage.kernels.iter().map(KernelProfile::gpu_or_wall).sum();
    println!("profiled dispatches: {}, summed kernel time {:.3} ms", stage.kernels.len(), total.as_secs_f64() * 1e3);
    println!("{}", render_histogram(&stage.kernels, 25));
    println!("origin rollup (leaf, exclusive):");
    for row in aggregate_origins(&stage.kernels, OriginView::Exclusive, args.origin_depth).into_iter().take(30) {
        println!(
            "{:>10.3} ms {:>5} {:>9.1} µs  {}",
            row.total.as_secs_f64() * 1e3,
            row.count,
            row.mean.as_secs_f64() * 1e6,
            row.path
        );
    }
    if let Some(path) = &args.csv {
        let mut out = String::from("idx\tkernel\tus\tflops\tbytes\torigins\n");
        for (i, k) in stage.kernels.iter().enumerate() {
            let (flops, bytes) = k.static_info.as_ref().map_or((None, 0), |s| (s.est_flops, s.est_bytes));
            let origins: Vec<String> = k.origins.iter().map(|id| svod_ir::origin::path(*id)).collect();
            out += &format!(
                "{i}\t{}\t{:.2}\t{}\t{bytes}\t{}\n",
                k.kernel.entry_point,
                k.gpu_or_wall().as_secs_f64() * 1e6,
                flops.map_or_else(|| "-".into(), |f| f.to_string()),
                origins.join(" | ")
            );
        }
        std::fs::write(path, out)?;
    }
    if let Some(path) = &args.json {
        std::fs::write(path, profile.to_json_at(args.origin_depth))?;
        println!("profile JSON: {}", path.display());
    }
    Ok(())
}
