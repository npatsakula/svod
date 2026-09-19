//! YOLO v26 object detection — load pretrained weights from HuggingFace,
//! run JIT-compiled inference, and print detected objects.
//!
//! ```text
//! # Download weights from HF and run on a synthetic image:
//! cargo run -p svod-model --release --example yolo_detect -- --hub
//!
//! # Run on a raw NCHW f32 .bin image (1×3×H×W, normalized 0–1):
//! cargo run -p svod-model --release --example yolo_detect -- --hub --image photo.bin --side 640
//!
//! # Use a specific scale:
//! cargo run -p svod-model --release --example yolo_detect -- --hub --scale small
//!
//! # Run a local checkpoint (e.g. converted from an Ultralytics .pt):
//! cargo run -p svod-model --release --example yolo_detect -- \
//!     --weights model.safetensors --scale xlarge --classes 1
//! ```

use std::path::PathBuf;

use bytemuck::cast_slice;
use clap::{Parser, ValueEnum};
use svod_dtype::DType;
use svod_model::jit::InputSpec;
use svod_model::yolo::{Yolo26Detect, Yolo26DetectJit, YoloConfig, YoloScale, postprocess_raw};

#[derive(Copy, Clone, Debug, ValueEnum)]
enum ScaleArg {
    Nano,
    Small,
    Medium,
    Large,
    Xlarge,
}

impl From<ScaleArg> for YoloScale {
    fn from(arg: ScaleArg) -> Self {
        match arg {
            ScaleArg::Nano => YoloScale::Nano,
            ScaleArg::Small => YoloScale::Small,
            ScaleArg::Medium => YoloScale::Medium,
            ScaleArg::Large => YoloScale::Large,
            ScaleArg::Xlarge => YoloScale::XLarge,
        }
    }
}

impl ScaleArg {
    fn hub_id(self) -> &'static str {
        match self {
            ScaleArg::Nano => "ultralytics/yolo26n",
            ScaleArg::Small => "ultralytics/yolo26s",
            ScaleArg::Medium => "ultralytics/yolo26m",
            ScaleArg::Large => "ultralytics/yolo26l",
            ScaleArg::Xlarge => "ultralytics/yolo26x",
        }
    }
}

#[derive(Parser, Debug)]
#[command(about = "YOLO v26 object detection inference", long_about = None)]
struct Args {
    /// Load pretrained weights from HuggingFace Hub.
    #[arg(long)]
    hub: bool,

    /// Use zero weights (no download — for testing graph construction).
    #[arg(long)]
    zero: bool,

    /// HuggingFace model id override (e.g. "ultralytics/yolo26n").
    #[arg(long)]
    hf_id: Option<String>,

    /// Local `model.safetensors` to load instead of downloading from the Hub.
    #[arg(long)]
    weights: Option<PathBuf>,

    /// Model scale.
    #[arg(long, value_enum, default_value_t = ScaleArg::Nano)]
    scale: ScaleArg,

    /// Number of object classes.
    #[arg(long, default_value_t = 80)]
    classes: usize,

    /// Square image side length in pixels.
    #[arg(long, default_value_t = 640)]
    side: usize,

    /// Maximum number of detections to return.
    #[arg(long, default_value_t = 300)]
    max_det: usize,

    /// Path to a raw NCHW f32 .bin file (1×3×side×side). If omitted, a
    /// deterministic pattern is generated so the example runs with no assets.
    #[arg(long)]
    image: Option<PathBuf>,
}

/// Load a raw f32 NCHW .bin image, or synthesize a deterministic gradient.
fn load_input(args: &Args) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let side = args.side;
    let n = 3 * side * side;
    if let Some(ref path) = args.image {
        let bytes = std::fs::read(path)?;
        let expected = n * std::mem::size_of::<f32>();
        if bytes.len() != expected {
            return Err(format!("expected {} bytes ({} floats), got {}", expected, n, bytes.len()).into());
        }
        Ok(cast_slice::<u8, f32>(&bytes).to_vec())
    } else {
        // Deterministic gradient pattern.
        let mut img = vec![0.0f32; n];
        for c in 0..3usize {
            for h in 0..side {
                for w in 0..side {
                    img[c * side * side + h * side + w] = ((h + w + c * 213) as f32) / ((side + side + 3 * 213) as f32);
                }
            }
        }
        Ok(img)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let scale: YoloScale = args.scale.into();
    let cfg = YoloConfig::new(scale, args.classes);

    let model = if args.zero {
        eprintln!("using zero weights (no download)");
        Yolo26Detect::with_zero_weights(cfg)
    } else if let Some(ref path) = args.weights {
        eprintln!("loading weights from {} ...", path.display());
        Yolo26Detect::from_safetensors(path, cfg)?
    } else if args.hub || args.hf_id.is_some() {
        let id = args.hf_id.as_deref().unwrap_or_else(|| args.scale.hub_id());
        eprintln!("downloading weights from {id} ...");
        Yolo26Detect::from_hub(id, cfg)?
    } else {
        eprintln!("no --hub or --zero given; defaulting to zero weights");
        Yolo26Detect::with_zero_weights(cfg)
    };

    let input = load_input(&args)?;
    let side = args.side;

    // --- JIT: compile once, run many -------------------------------------
    // `prepare` bakes the image size (H, W) into the execution plan. The
    // batch dimension is a symbolic variable bound at runtime via
    // `execute_bound`, so the same plan handles any batch ≤ max_batch_size.
    let mut jit = Yolo26DetectJit::new(model);
    jit.prepare_with_config(
        InputSpec::new(&[1, 3, side, side], DType::Float32).device_local(),
        &svod_tensor::PrepareConfig::device_local(),
    )?;

    // Copy the NCHW image into the JIT-managed input buffer.
    jit.images_mut()?.copyin(cast_slice(&input))?;

    // Execute with batch = 1.
    let t0 = std::time::Instant::now();
    jit.execute_bound(1)?;
    let elapsed = t0.elapsed();

    // --- Read output -----------------------------------------------------
    // The wrapper knows the declared output's live shape ([1, 4+nc, A] —
    // decoded xyxy boxes + sigmoid'd class scores) with `b` substituted, so
    // there is no buffer-size arithmetic to do here.
    let shape = jit.predictions_shape()?;
    let data = jit.predictions_to_vec::<f32>()?;

    let detections = postprocess_raw(&data, &shape, args.classes, args.max_det)?;
    let dets = &detections[0];

    // --- Print results ---------------------------------------------------
    println!();
    println!(" detections: {}", dets.len());
    if !dets.is_empty() {
        println!("  {:<5}  {:<8}  {:>10}  {:>10}  {:>10}  {:>10}  conf", "rank", "class", "x1", "y1", "x2", "y2");
        for (i, d) in dets.iter().enumerate() {
            let [x1, y1, x2, y2, conf, class] = *d;
            println!(
                "  {:<5}  {:<8}  {:>10.1}  {:>10.1}  {:>10.1}  {:>10.1}  {:.4}",
                i + 1,
                class as usize,
                x1,
                y1,
                x2,
                y2,
                conf
            );
        }
    }

    let anchors = shape[2];
    println!();
    println!("  inference: {:.2} ms", elapsed.as_secs_f64() * 1e3);
    println!("  image:     {}×{}", side, side);
    println!("  anchors:   {}", anchors);

    Ok(())
}
