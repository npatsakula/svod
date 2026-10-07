//! Mel front-end benchmark: the graph path's prepare and execute times, and
//! its per-kernel profile, for the Whisper (Slaney, 400/160, 80 mels) and
//! GigaAM (HTK, 320/160, 64 mels) configurations over 30 s of 16 kHz audio.
//!
//! Usage:
//!   cargo run --release -p svod-model --example mel_bench
//!   SVOD_DEVICE=CUDA cargo run --release -p svod-model --example mel_bench -- --runs 50
//!   cargo run --release -p svod-model --example mel_bench -- --wav ru_clip_0.wav

use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::Parser;
use svod_model::audio::{MelConfig, MelJit, MelScale, MelSpectrogram};
use svod_model::jit::InputSpec;
use svod_model::whisper::{N_SAMPLES, WhisperMel, WhisperMelJit};
use svod_tensor::PrepareConfig;

#[derive(Parser, Debug)]
#[command(about = "Graph mel front-end timings", long_about = None)]
struct Args {
    /// Timed iterations per path (the median is reported).
    #[arg(long, default_value_t = 20)]
    runs: usize,

    /// 16 kHz mono WAV to use instead of the synthetic 30 s signal.
    #[arg(long)]
    wav: Option<PathBuf>,
}

fn print_kernels(label: &str, kernels: &[svod_runtime::KernelProfile]) {
    println!("\n{label} kernels:");
    for k in kernels {
        println!("  {:>9.3} ms  {}", k.gpu_or_wall().as_secs_f64() * 1e3, k.kernel.entry_point);
    }
    println!();
}

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort();
    samples[samples.len() / 2]
}

fn timed(runs: usize, mut f: impl FnMut()) -> Duration {
    median(
        (0..runs)
            .map(|_| {
                let t = Instant::now();
                f();
                t.elapsed()
            })
            .collect(),
    )
}

fn synthetic(len: usize) -> Vec<f32> {
    let mut state = 12345u32;
    (0..len)
        .map(|i| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
            let t = i as f32 / 16000.0;
            0.4 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                + 0.3 * (2.0 * std::f32::consts::PI * 1375.0 * t).sin()
                + 0.05 * noise
        })
        .collect()
}

fn row(label: &str, prepare: Duration, stage: Duration, execute: Duration) {
    println!(
        "| {label} | {:.0} ms | {:.2} ms | {:.2} ms |",
        prepare.as_secs_f64() * 1e3,
        stage.as_secs_f64() * 1e3,
        execute.as_secs_f64() * 1e3
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let audio = match &args.wav {
        Some(path) => {
            let mut reader = hound::WavReader::open(path)?;
            let mut samples: Vec<f32> =
                reader.samples::<i16>().map(|s| s.map(|v| v as f32 / 32768.0)).collect::<Result<_, _>>()?;
            samples.resize(N_SAMPLES, 0.0);
            samples
        }
        None => synthetic(N_SAMPLES),
    };
    let device = svod_dtype::default_device::default_device();
    println!("device: {device:?}; {} samples; median of {} runs\n", audio.len(), args.runs);
    println!("| front-end | prepare | host stage + upload | execute + sync |");
    println!("|-----------|--------:|--------------------:|---------------:|");

    // ── Whisper ────────────────────────────────────────────────────────────
    let whisper = WhisperMel::new(80);
    let mut jit = WhisperMelJit::new(whisper);
    let t = Instant::now();
    jit.prepare_with_config(InputSpec::f32(&[1, N_SAMPLES]).device_local(), &PrepareConfig::device_local())?;
    let prepare = t.elapsed();
    let mut samples = vec![0.0f32; N_SAMPLES];
    let stage = timed(args.runs, || {
        WhisperMel::pad_or_trim_into(&audio, &mut samples);
        jit.samples_mut().unwrap().copyin(bytemuck::cast_slice(&samples)).unwrap();
    });
    let execute = timed(args.runs, || {
        jit.execute().unwrap();
        jit.output().unwrap().synchronize().unwrap();
    });
    row("whisper 80 mel", prepare, stage, execute);
    print_kernels("whisper", &jit.execute_profiled()?);

    // ── GigaAM ─────────────────────────────────────────────────────────────
    let config = MelConfig {
        sample_rate: 16000,
        n_fft: 320,
        hop_length: 160,
        win_length: 320,
        n_mels: 64,
        center: true,
        mel_scale: MelScale::Htk,
        periodic: true,
        pad_mode: svod_model::audio::PadMode::Reflect,
        preemphasis: None,
        log: svod_model::audio::MelSpectrogram::LOG,
    };
    let mel = MelSpectrogram::new(&config);
    let frames = mel.num_frames(audio.len());
    let framed_len = mel.framed_len(audio.len());
    let mut jit = MelJit::new(mel.clone());
    let t = Instant::now();
    jit.prepare_with_config(
        InputSpec::f32(&[1, framed_len]).device_local(),
        InputSpec::i32(&[1]),
        &PrepareConfig::device_local(),
    )?;
    let prepare = t.elapsed();
    let mut framed = vec![0.0f32; framed_len];
    let stage = timed(args.runs, || {
        mel.frame_into(&audio, &mut framed);
        jit.framed_mut().unwrap().copyin(bytemuck::cast_slice(&framed)).unwrap();
        jit.frames_view_mut::<i32>().unwrap().as_slice_mut().unwrap()[0] = frames as i32;
    });
    let execute = timed(args.runs, || {
        jit.execute().unwrap();
        jit.output().unwrap().synchronize().unwrap();
    });
    row("gigaam 64 mel", prepare, stage, execute);
    print_kernels("gigaam", &jit.execute_profiled()?);
    Ok(())
}
