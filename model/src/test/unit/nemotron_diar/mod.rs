mod config;
mod driver;
mod parity;

/// The published `config.json` and `processor_config.json` of
/// `nvidia/Nemotron-3-Diarization`.
const CONFIG_JSON: &str = include_str!("config.json");
const PROCESSOR_JSON: &str = include_str!("processor_config.json");
