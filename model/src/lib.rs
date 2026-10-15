//! Pretrained models on top of `svod-tensor`: speech recognition (Whisper, GigaAM),
//! voice activity (FireRedVAD, Silero), speech enhancement (GTCRN), speaker analysis
//! (Nemotron-3-Diarization, DiariZen, WeSpeaker), text embeddings and reranking (BGE-M3, Qwen3, ModernBERT)
//! and vision (YOLO26, ResNet).
//!
//! Weights load from the Hugging Face Hub, and each model runs through JIT plans that
//! compile once and replay. See <https://svod.vpermilp.online/docs/models>.

pub(crate) mod attention;
pub mod audio;
pub mod bgem3;
pub mod blocks;
pub mod diarizen;
pub mod encoder_head;
pub mod firered_vad;
pub mod gigaam;
pub mod gtcrn;
pub(crate) mod hub;
pub(crate) mod init;
pub mod jit;
pub(crate) mod linear;
pub mod modernbert;
pub mod nemotron_diar;
pub mod qwen3;
pub mod resnet;
pub mod sentencepiece;
pub mod silero_vad;
mod simd;
pub mod state;
pub mod wavlm;
pub mod wespeaker;
pub mod whisper;
pub mod xlm_roberta;
pub mod yolo;

/// Default compute dtype of the transformer configs (`modernbert`, `qwen3`,
/// `xlm_roberta`): [`compute_dtype_for`] the thread's default device. Callers
/// wanting CPU parity against an f32 reference set `dtype` explicitly.
pub fn default_compute_dtype() -> svod_dtype::DType {
    compute_dtype_for(&svod_dtype::default_device::default_device())
}

/// The dtype a transformer computes in on `device`: bf16, except on a CUDA
/// device without bf16 tensor cores (pre-Ampere), whose profile stores no bf16
/// at all, or one that cannot be opened to tell.
pub fn compute_dtype_for(device: &svod_dtype::DeviceSpec) -> svod_dtype::DType {
    match device {
        svod_dtype::DeviceSpec::Cuda { device_id } => match svod_device::registry::resolve_cuda_arch(*device_id) {
            Ok(arch) if arch.has_bf16_mma() => svod_dtype::DType::BFloat16,
            _ => svod_dtype::DType::Float32,
        },
        _ => svod_dtype::DType::BFloat16,
    }
}

#[cfg(test)]
mod test;
