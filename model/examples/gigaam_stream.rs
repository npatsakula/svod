//! GigaAM live streaming demo: a WAV pushed into a [`GigaAmStreamer`] in
//! small pieces, as a microphone or a socket would deliver it, the running
//! hypothesis printed as it grows and the final transcript with word times.
//!
//! A checkpoint fine-tuned for streaming carries its settings in
//! `config.json`; the offline weights stream too (`--chunk-size`,
//! `--left-context`, `--right-context` set the variant) at reduced accuracy.
//!
//! Usage:
//!   cargo run -p svod-model --release --example gigaam_stream -- audio.wav
//!   cargo run -p svod-model --release --example gigaam_stream -- audio.wav --rnnt --right-context 1
//!   cargo run -p svod-model --release --example gigaam_stream -- audio.wav --streams 8 --push-ms 100

use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::Parser;
use svod_dtype::DType;
use svod_model::gigaam::stream::{GigaAmStreamer, StreamId};
use svod_model::gigaam::{GigaAm, StreamingConfig};

#[derive(Parser, Debug)]
#[command(about = "GigaAM streaming demo (CTC + RN-T)", long_about = None)]
struct Args {
    /// Input WAV (16 kHz mono).
    wav: PathBuf,
    /// HF Hub repo or local model directory.
    #[arg(long, default_value = "vpermilp/GigaAM-v3")]
    repo: String,
    /// HF Hub revision; defaults to `ctc`, or `e2e_rnnt` under `--rnnt`.
    #[arg(long)]
    revision: Option<String>,
    #[arg(long)]
    rnnt: bool,
    /// Encoder frames per chunk (40 ms each), for weights without streaming settings.
    #[arg(long, default_value_t = 8)]
    chunk_size: usize,
    /// Chunks of past a query sees; negative for every one.
    #[arg(long, default_value_t = 16)]
    left_context: i64,
    /// 1 adds a chunk of look-ahead at a chunk of extra latency.
    #[arg(long, default_value_t = 0)]
    right_context: usize,
    /// Concurrent copies of the stream, to measure the batched step.
    #[arg(long, default_value_t = 1)]
    streams: usize,
    /// Audio per push.
    #[arg(long, default_value_t = 100)]
    push_ms: usize,
    /// Print the running hypothesis after every push.
    #[arg(long)]
    partials: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    let revision = args.revision.clone().unwrap_or_else(|| if args.rnnt { "e2e_rnnt" } else { "ctc" }.to_string());
    let (wave, sample_rate) = load_wav(&args.wav)?;
    let seconds = wave.len() as f32 / sample_rate as f32;
    println!("audio: {:.1} s @ {sample_rate} Hz", seconds);

    let local = PathBuf::from(&args.repo);
    let mut model = if local.is_dir() {
        GigaAm::from_dir_with_weights_and_encoder_dtype(&local, "model.safetensors", DType::Float16)?
    } else {
        GigaAm::from_hub_with_revision_and_weights_and_encoder_dtype(
            &args.repo,
            &revision,
            "model.safetensors",
            DType::Float16,
        )?
    };
    if sample_rate as usize != model.config.sample_rate {
        return Err(format!("WAV is {sample_rate} Hz; the model expects {} Hz", model.config.sample_rate).into());
    }
    match model.config.streaming {
        Some(streaming) => println!("streaming fine-tune: {streaming:?}"),
        None => {
            let streaming = StreamingConfig {
                chunk_size: args.chunk_size,
                left_context_chunks: usize::try_from(args.left_context).ok(),
                right_context_chunks: args.right_context,
            };
            println!("offline weights streamed as {streaming:?}");
            model.enable_streaming(streaming);
        }
    }

    let t_prepare = Instant::now();
    let mut streamer = GigaAmStreamer::new(model, args.streams)?;
    let mut buffers = streamer.device_buffers()?;
    buffers.sort_unstable_by(|a, b| b.cmp(a));
    let bytes: usize = buffers.iter().sum();
    println!(
        "prepared {} lanes in {:.2} s; latency {:.0} ms; {} device buffers, {:.0} MiB ({:.1} MiB per lane), largest {:?} MiB",
        args.streams,
        t_prepare.elapsed().as_secs_f32(),
        streamer.latency_seconds() * 1e3,
        buffers.len(),
        bytes as f64 / 1048576.0,
        bytes as f64 / 1048576.0 / args.streams as f64,
        buffers.iter().take(6).map(|b| b / 1048576).collect::<Vec<_>>(),
    );

    let ids: Vec<StreamId> = (0..args.streams).map(|_| streamer.open().expect("a free lane")).collect();
    let push = args.push_ms * sample_rate as usize / 1000;
    let (mut t_step, mut steps, mut slowest) = (Duration::ZERO, 0usize, Duration::ZERO);
    let mut finals = vec![None; args.streams];
    let t_run = Instant::now();
    let mut pos = 0;
    while finals.iter().any(Option::is_none) {
        let end = (pos + push).min(wave.len());
        for &id in &ids {
            if pos < wave.len() {
                streamer.push(id, &wave[pos..end])?;
                if end == wave.len() {
                    streamer.finish(id)?;
                }
            }
        }
        pos = end;
        let (t0, before) = (Instant::now(), streamer.rounds());
        let events = streamer.step()?;
        let took = t0.elapsed();
        if streamer.rounds() > before {
            t_step += took;
            steps += 1;
            slowest = slowest.max(took);
        }
        for event in events {
            let i = ids.iter().position(|&id| id == event.id).expect("our stream");
            if let Some(transcript) = event.transcript {
                finals[i] = Some(transcript);
            } else if args.partials && i == 0 {
                println!("[{:6.2} s] {}", pos as f32 / sample_rate as f32, streamer.transcript(event.id)?.text);
            }
        }
    }
    let wall = t_run.elapsed();
    let transcript = finals[0].take().expect("the first stream's transcript");
    println!("\n{}", transcript.text);
    for word in &transcript.words {
        println!("  [{:6.2} - {:6.2}] {}", word.start, word.end, word.text);
    }
    println!(
        "\n{} streams × {:.1} s in {:.2} s wall (RTF {:.4}); {steps} pushes ran a step, {:.1} ms mean, {:.1} ms slowest",
        args.streams,
        seconds,
        wall.as_secs_f32(),
        wall.as_secs_f32() / (seconds * args.streams as f32),
        t_step.as_secs_f32() * 1e3 / steps.max(1) as f32,
        slowest.as_secs_f32() * 1e3,
    );
    Ok(())
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
    Ok((samples, spec.sample_rate))
}
