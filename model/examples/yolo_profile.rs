//! YOLO26 profiling harness — per-kernel, origin-attributed profile of the
//! JIT-compiled detector plus a wall-clock breakdown of the e2e stages.
//!
//! ```text
//! # CPU (default device), bounded threads, x scale from the HF cache:
//! SVOD_THREADS=4 cargo run -p svod-model --release --example yolo_profile -- \
//!     --scale x --iters 3 --warmup 1 --json /tmp/yolo26x.json
//!
//! # Same harness on CUDA later, unchanged code:
//! SVOD_THREADS=4 cargo run -p svod-model --release --example yolo_profile -- \
//!     --scale x --device cuda
//!
//! # Local checkpoint instead of the Hub:
//! cargo run -p svod-model --release --example yolo_profile -- \
//!     --scale x --local model.safetensors
//! ```

use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::{Parser, ValueEnum};
use svod_dtype::DeviceSpec;
use svod_model::jit::InputSpec;
use svod_model::yolo::{Yolo26Detect, Yolo26DetectJit, YoloConfig, YoloScale, postprocess_raw};
use svod_runtime::{
    KernelProfile, OriginView, RunProfile, StageProfile, aggregate_origins, has_origins, render_histogram,
};
use svod_tensor::PrepareConfig;
use svod_tensor::set_default_device;

const MAX_DET: usize = 300;

#[derive(Copy, Clone, Debug, ValueEnum)]
enum ScaleArg {
    N,
    S,
    M,
    L,
    X,
}

impl From<ScaleArg> for YoloScale {
    fn from(arg: ScaleArg) -> Self {
        match arg {
            ScaleArg::N => YoloScale::Nano,
            ScaleArg::S => YoloScale::Small,
            ScaleArg::M => YoloScale::Medium,
            ScaleArg::L => YoloScale::Large,
            ScaleArg::X => YoloScale::XLarge,
        }
    }
}

impl ScaleArg {
    /// Converted safetensors repos; only `x` is published today, other scales
    /// need `--hub`/`--local` until their repos exist.
    fn hub_id(self) -> &'static str {
        match self {
            ScaleArg::N => "mexus/svod-yolo26n",
            ScaleArg::S => "mexus/svod-yolo26s",
            ScaleArg::M => "mexus/svod-yolo26m",
            ScaleArg::L => "mexus/svod-yolo26l",
            ScaleArg::X => "mexus/svod-yolo26x",
        }
    }

    fn letter(self) -> &'static str {
        match self {
            ScaleArg::N => "n",
            ScaleArg::S => "s",
            ScaleArg::M => "m",
            ScaleArg::L => "l",
            ScaleArg::X => "x",
        }
    }
}

/// Compute dtype for the backbone and neck. The heads always decode in f32.
#[derive(Copy, Clone, Debug, ValueEnum)]
enum DtypeArg {
    F32,
    F16,
    Bf16,
}

impl From<DtypeArg> for svod_dtype::DType {
    fn from(arg: DtypeArg) -> Self {
        match arg {
            DtypeArg::F32 => svod_dtype::DType::Float32,
            DtypeArg::F16 => svod_dtype::DType::Float16,
            DtypeArg::Bf16 => svod_dtype::DType::BFloat16,
        }
    }
}

#[derive(Parser, Debug)]
#[command(about = "YOLO26 per-kernel profiling harness (CPU now, CUDA later)", long_about = None)]
struct Args {
    /// Model scale.
    #[arg(long, value_enum, default_value_t = ScaleArg::X)]
    scale: ScaleArg,

    /// Compute dtype for backbone + neck (heads stay f32). f16 halves weight
    /// bandwidth and doubles tensor-core peak on consumer Ampere, where tf32
    /// buys no FLOPS over plain f32.
    #[arg(long, value_enum, default_value_t = DtypeArg::F32)]
    dtype: DtypeArg,

    /// Number of classes the checkpoint was trained with.
    #[arg(long, default_value_t = 80)]
    nc: usize,

    /// HF Hub repo override (default: mexus/svod-yolo26<scale>).
    #[arg(long)]
    hub: Option<String>,

    /// Local `model.safetensors` to load instead of the Hub.
    #[arg(long, value_name = "PATH")]
    local: Option<PathBuf>,

    /// Raw f32 NCHW input (`batch·3·size²` little-endian floats) instead of the
    /// deterministic gradient; real imagery stresses numerics honestly.
    #[arg(long, value_name = "PATH")]
    input: Option<PathBuf>,

    /// Write the last iteration's raw predictions (f32 LE) for numeric diffs.
    #[arg(long, value_name = "PATH")]
    dump: Option<PathBuf>,

    /// Timed steady-state iterations (kernel times keep the per-kernel min).
    #[arg(long, default_value_t = 3)]
    iters: usize,

    /// Untimed warmup iterations (also absorbs first-run cache effects).
    #[arg(long, default_value_t = 1)]
    warmup: usize,

    /// Square input side in pixels.
    #[arg(long, default_value_t = 640)]
    size: usize,

    /// Batch size to bind at runtime.
    #[arg(long, default_value_t = 1)]
    batch: usize,

    /// Export SVOD_THREADS before runtime init; default leaves it untouched.
    #[arg(long)]
    threads: Option<usize>,

    /// Origin rollup depth: `N` outermost module frames, unset = leaf scope.
    #[arg(long, value_name = "N")]
    origin_depth: Option<usize>,

    /// Write the ProfileExport JSON to this path.
    #[arg(long, value_name = "PATH")]
    json: Option<PathBuf>,

    /// Print the e2e wall-clock stage breakdown (load / compile / steady state).
    #[arg(long)]
    stage_detail: bool,

    /// Device to run on: `cpu`, `cuda`, `cuda:N`, `amd`, `amd:N`. Exported as SVOD_DEVICE
    /// before runtime init; CPU is the default so the GPU is never probed.
    #[arg(long, default_value = "cpu")]
    device: String,
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

/// Deterministic gradient image (same pattern as `yolo_detect`), NCHW f32.
fn synth_input(side: usize, batch: usize) -> Vec<f32> {
    let mut img = vec![0.0f32; batch * 3 * side * side];
    for c in 0..3usize {
        for h in 0..side {
            for w in 0..side {
                let value = ((h + w + c * 213) as f32) / ((side + side + 3 * 213) as f32);
                for b in 0..batch {
                    img[b * 3 * side * side + c * side * side + h * side + w] = value;
                }
            }
        }
    }
    img
}

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn stats(samples: &[Duration]) -> String {
    format!(
        "min {:.1} ms / median {:.1} ms over {} iter(s)",
        samples.iter().min().map_or(0.0, |d| d.as_secs_f64() * 1e3),
        median(samples.to_vec()).as_secs_f64() * 1e3,
        samples.len()
    )
}

/// Exclusive origin rollup at one depth: rows partition the kernel total.
fn print_origin_section(kernels: &[KernelProfile], depth: Option<usize>) {
    let total: Duration = kernels.iter().map(KernelProfile::gpu_or_wall).sum();
    let label = depth.map_or_else(|| "leaf".to_owned(), |d| d.to_string());
    println!("origin rollup (depth {label}, exclusive; rows sum to the total):");
    println!("{:>10}  {:>5}  {:>9}  {:>5}  origin path", "total ms", "count", "mean µs", "%");
    for row in aggregate_origins(kernels, OriginView::Exclusive, depth).into_iter().take(20) {
        let pct = 100.0 * row.total.as_secs_f64() / total.as_secs_f64().max(f64::EPSILON);
        println!(
            "{:>10.3}  {:>5}  {:>9.1}  {:>5.1}  {}",
            row.total.as_secs_f64() * 1e3,
            row.count,
            row.mean.as_secs_f64() * 1e6,
            pct,
            row.path
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    if args.iters == 0 {
        return Err("--iters must be non-zero".into());
    }
    let device = parse_device(&args.device)?;
    // Pin the process default before anything touches the runtime; the
    // thread-local override covers this thread even if a caller already read
    // the env. Origins default on so kernel rows carry scopes.
    // Safety: single-threaded prologue, before any runtime threads spawn.
    unsafe {
        std::env::set_var("SVOD_DEVICE", &args.device);
        if let Some(threads) = args.threads {
            std::env::set_var("SVOD_THREADS", threads.to_string());
        }
        if std::env::var("SVOD_ORIGIN").is_err() {
            std::env::set_var("SVOD_ORIGIN", "1");
        }
    }
    set_default_device(device.clone());
    let threads = std::env::var("SVOD_THREADS").unwrap_or_else(|_| "default".into());

    let tag = if device == DeviceSpec::Cpu { "PROVISIONAL (CPU)" } else { "PROVISIONAL" };
    println!(
        "=== YOLO26 profile: device {}, SVOD_THREADS {threads}, scale {}, nc {}, imgsz {}, batch {} — {tag} ===",
        device.canonicalize(),
        args.scale.letter(),
        args.nc,
        args.size,
        args.batch
    );
    println!("compute dtype: {:?} (heads f32)", svod_dtype::DType::from(args.dtype));

    // --- load -------------------------------------------------------------
    let t_load = Instant::now();
    let cfg = YoloConfig::new(args.scale.into(), args.nc)
        .with_max_batch_size(args.batch.max(1))
        .with_compute_dtype(args.dtype.into());
    let model = if let Some(ref path) = args.local {
        Yolo26Detect::from_safetensors(path, cfg)?
    } else {
        let id = args.hub.as_deref().unwrap_or_else(|| args.scale.hub_id());
        Yolo26Detect::from_hub(id, cfg)?
    };
    let dt_load = t_load.elapsed();

    // --- prepare: graph build + first compile ------------------------------
    let t_prepare = Instant::now();
    let mut jit = Yolo26DetectJit::new(model);
    jit.prepare_with_config(
        InputSpec::f32(&[args.batch, 3, args.size, args.size]).device_local(),
        &PrepareConfig::device_local(),
    )?;
    let dt_prepare = t_prepare.elapsed();

    let input: Vec<f32> = match &args.input {
        Some(path) => {
            let bytes = std::fs::read(path).map_err(|e| format!("--input {}: {e}", path.display()))?;
            let floats: Vec<f32> = bytes.as_chunks::<4>().0.iter().copied().map(f32::from_le_bytes).collect();
            let want = args.batch * 3 * args.size * args.size;
            if floats.len() != want {
                return Err(format!(
                    "--input {} holds {} floats, expected {want} (batch {} × 3 × {}²)",
                    path.display(),
                    floats.len(),
                    args.batch,
                    args.size
                )
                .into());
            }
            floats
        }
        None => synth_input(args.size, args.batch),
    };

    jit.images_mut()?.copyin(bytemuck::cast_slice(&input))?;

    // --- warmup (untimed) --------------------------------------------------
    for _ in 0..args.warmup {
        jit.execute_bound(args.batch as i64)?;
    }

    // --- steady state: forward / readout / postprocess ---------------------
    let mut forward = Vec::with_capacity(args.iters);
    let mut readout = Vec::with_capacity(args.iters);
    let mut post = Vec::with_capacity(args.iters);
    let mut detections = 0;
    for _ in 0..args.iters {
        let t = Instant::now();
        jit.execute_bound(args.batch as i64)?;
        forward.push(t.elapsed());

        let t = Instant::now();
        let shape = jit.predictions_shape()?;
        let data = jit.predictions_to_vec::<f32>()?;
        readout.push(t.elapsed());

        let t = Instant::now();
        let dets = postprocess_raw(&data, &shape, args.nc, MAX_DET)?;
        post.push(t.elapsed());
        detections = dets[0].len();
        if let Some(path) = &args.dump {
            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            std::fs::write(path, &bytes).map_err(|e| format!("--dump {}: {e}", path.display()))?;
        }
    }
    println!("detections (last iter): {detections}");
    println!("forward steady-state: {}", stats(&forward));

    // --- per-kernel profile passes (min-merged across iters) ---------------
    // `execute_profiled_static` drives `ExecutionPlan::profile` (static
    // analysis on) with the plan's current `b` binding, one pass per call;
    // `RunProfile::merge_min` keeps each kernel's fastest sample.
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
    let stage = profile.stage("forward").ok_or("profile produced no forward stage")?;
    if stage.kernels.is_empty() {
        return Err("profiler returned zero kernels — refusing to print empty tables".into());
    }
    println!("profiled dispatches: {}", stage.kernels.len());

    if args.stage_detail {
        println!("\n--- e2e wall breakdown (compile excluded from steady state) ---");
        println!("load:              {:.1} ms", dt_load.as_secs_f64() * 1e3);
        println!("prepare+compile:   {:.1} ms", dt_prepare.as_secs_f64() * 1e3);
        println!("warmup:            {} iter(s), excluded", args.warmup);
        println!("forward:           {}", stats(&forward));
        println!("readout (to_vec):  {}", stats(&readout));
        println!("postprocess_raw:   {}", stats(&post));
        println!("profile pass wall: {:.1} ms (first of {})", stage.wall.as_secs_f64() * 1e3, args.iters);
    }

    println!("\n--- RunProfile table (per-kernel, min over {} pass(es)) ---", args.iters);
    println!("{}", profile.render_table_at(args.origin_depth));

    println!("--- kernel histogram (top 20 by total) ---");
    println!("{}", render_histogram(&stage.kernels, 20));

    if has_origins(&stage.kernels) {
        println!("--- origins ---");
        print_origin_section(&stage.kernels, Some(3));
        println!();
        print_origin_section(&stage.kernels, None);
    } else {
        eprintln!("WARNING: no kernel carried a scope (origin capture off?) — origin sections skipped");
    }

    if let Some(ref path) = args.json {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, profile.to_json_at(args.origin_depth))?;
        println!("\nprofile JSON: {}", path.display());
    }

    Ok(())
}
