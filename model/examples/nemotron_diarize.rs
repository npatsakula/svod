//! Nemotron-3-Diarization: a 16 kHz mono WAV in, RTTM speaker segments out.
//!
//! Usage:
//!   cargo run -p svod-model --release --example nemotron_diarize -- audio.wav
//!   cargo run -p svod-model --release --example nemotron_diarize -- audio.wav --stream low
//!
//! `--stream` feeds the audio in 100 ms pieces through a streaming session, as
//! a live microphone would; without it the recording is diarized offline.

use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use svod_arch::diarization::{Binarization, speaker_segments, write_rttm};
use svod_dtype::DType;
use svod_model::nemotron_diar::{Diarizer, NemotronDiar, StreamingMode};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Latency {
    /// 1.04 s
    Low,
    /// 0.64 s
    VeryLow,
    /// 0.32 s
    UltraLow,
}

#[derive(Parser, Debug)]
#[command(about = "Nemotron-3-Diarization speaker diarization", long_about = None)]
struct Args {
    /// Input WAV (16 kHz mono).
    audio: PathBuf,

    /// Stream the audio with this latency instead of diarizing it offline.
    #[arg(long, value_enum)]
    stream: Option<Latency>,

    /// Compute dtype: f32, f16 or bf16. Defaults to the device's 16-bit type on
    /// a GPU (`default_compute_dtype`) and to f32 on the CPU, which emulates
    /// 16-bit arithmetic.
    #[arg(long)]
    dtype: Option<String>,

    /// Diarize once to compile and warm every plan, then time and profile a
    /// second run, printing its per-kernel report.
    #[arg(long)]
    profile: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let dtype = match args.dtype.as_deref() {
        None if svod_dtype::default_device::default_device() == svod_dtype::DeviceSpec::Cpu => DType::Float32,
        None => svod_model::default_compute_dtype(),
        Some("f32") => DType::Float32,
        Some("f16") => DType::Float16,
        Some("bf16") => DType::BFloat16,
        Some(other) => return Err(format!("unknown dtype {other}").into()),
    };
    let (audio, sample_rate) = load_wav(&args.audio)?;
    let duration = audio.len() as f32 / sample_rate as f32;

    let started = Instant::now();
    let model = NemotronDiar::from_hub(dtype, 1)?;
    let frame_sec = model.config.frame_sec();
    let mut diarizer = match args.stream {
        None => Diarizer::offline(model)?,
        Some(mode) => Diarizer::streaming(
            model,
            match mode {
                Latency::Low => StreamingMode::LowLatency,
                Latency::VeryLow => StreamingMode::VeryLowLatency,
                Latency::UltraLow => StreamingMode::UltraLowLatency,
            },
        )?,
    };
    eprintln!("loaded in {:.2}s", started.elapsed().as_secs_f32());

    if args.profile {
        let started = Instant::now();
        diarize(&mut diarizer, &audio, sample_rate, args.stream.is_some())?;
        eprintln!("cold run (compiles every plan): {:.2}s", started.elapsed().as_secs_f32());
    }
    let started = Instant::now();
    let probs = diarize(&mut diarizer, &audio, sample_rate, args.stream.is_some())?;
    let elapsed = started.elapsed().as_secs_f32();
    eprintln!("diarized {duration:.1}s of audio in {elapsed:.2}s (RTF {:.3})", elapsed / duration);
    if args.profile {
        diarizer.start_profiling();
        diarize(&mut diarizer, &audio, sample_rate, args.stream.is_some())?;
        let profile = diarizer.take_profile().expect("profiling was started");
        let steps = profile.stages.len();
        let merged = svod_runtime::StageProfile::gpu(
            format!("{steps} steps"),
            profile.stages.iter().map(|s| s.wall).sum(),
            profile.stages.into_iter().flat_map(|s| s.kernels).collect(),
        );
        let mut run = svod_runtime::RunProfile::default();
        run.push(merged);
        eprintln!("{}", run.render_report_at(svod_runtime::ProfileOptions::from_env().origin_depth));
    }

    let speakers = diarizer.num_speakers();
    let segments = speaker_segments(&probs, speakers, frame_sec, &Binarization::default());
    let recording = args.audio.file_stem().and_then(|s| s.to_str()).unwrap_or("audio");
    write_rttm(std::io::stdout().lock(), recording, &segments)?;
    Ok(())
}

/// Offline, or streamed in 100 ms pieces as a live microphone would feed it.
fn diarize(
    diarizer: &mut Diarizer,
    audio: &[f32],
    sample_rate: u32,
    stream: bool,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    if !stream {
        return Ok(diarizer.diarize(audio, sample_rate as usize)?.probs);
    }
    let mut session = diarizer.session();
    for piece in audio.chunks(sample_rate as usize / 10) {
        session.push(piece)?;
        diarizer.run(&mut [&mut session])?;
    }
    session.finish();
    diarizer.run(&mut [&mut session])?;
    Ok(session.take_probs())
}

fn load_wav(path: &PathBuf) -> Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            reader.samples::<i16>().map(|s| s.map(|v| v as f32 / 32768.0)).collect::<Result<_, _>>()?
        }
    };
    if spec.channels != 1 {
        return Err(format!("{} channels; expected mono", spec.channels).into());
    }
    Ok((samples, spec.sample_rate))
}
