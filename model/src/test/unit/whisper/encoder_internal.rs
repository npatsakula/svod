//! Whisper encoder internals: the encoder sequence-padding policy and the
//! per-block flash-attention dispatch count it produces.

use crate::whisper::config::ModelDimensions;
use crate::whisper::encoder::{AudioEncoder, encoder_padded_sequence_len};
use svod_dtype::{DType, DeviceSpec};
use svod_tensor::Tensor;

fn encoder_dims(layers: usize) -> ModelDimensions {
    ModelDimensions {
        n_mels: 4,
        n_audio_ctx: 1500,
        n_audio_state: 128,
        n_audio_head: 2,
        n_audio_layer: layers,
        n_vocab: 16,
        n_text_ctx: 8,
        n_text_state: 8,
        n_text_head: 2,
        n_text_layer: 1,
        dtype: DType::Float16,
    }
}

#[test]
fn unsupported_device_keeps_original_encoder_sequence() {
    assert_eq!(encoder_padded_sequence_len(&DeviceSpec::Cpu, 1500), None);

    let encoder = AudioEncoder::empty(&encoder_dims(1));
    let mel = Tensor::zeros(&[1, 4, 3000], DType::Float32).unwrap();
    let out = encoder.forward(&mel).unwrap();
    assert_eq!(out.shape().unwrap().iter().map(|d| d.as_const().unwrap()).collect::<Vec<_>>(), [1, 1500, 128]);
}

#[test]
#[ignore = "GPU: inspect full padded Whisper encoder execution plan"]
fn padded_encoder_plan_has_one_flash_attention_per_block() {
    // The encoder gates on its activations' device, which follows the weights
    // onto the process default device.
    let device = svod_dtype::default_device::default_device();
    if !svod_tk::flash_attention_supported(&device) {
        eprintln!("skipping: flash-attention is not supported on {device:?}");
        return;
    }
    let encoder = AudioEncoder::empty(&encoder_dims(32));
    let mel = Tensor::zeros(&[1, 4, 3000], DType::Float32).unwrap();
    let mut out = encoder.forward(&mel).unwrap();
    assert_eq!(out.shape().unwrap().iter().map(|d| d.as_const().unwrap()).collect::<Vec<_>>(), [1, 1500, 128]);

    let plan = out.prepare().unwrap();
    let flash_attention = plan.kernels().filter(|kernel| kernel.entry_point == "flash_attention").count();
    assert_eq!(flash_attention, 32, "expected one handwritten flash-attention dispatch per encoder block");
}
