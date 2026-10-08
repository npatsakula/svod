//! Whisper encoder internals: the unpadded 1500-frame sequence and its one
//! flash-attention dispatch per block.

use crate::whisper::config::ModelDimensions;
use crate::whisper::encoder::AudioEncoder;
use svod_dtype::DType;
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
fn encoder_keeps_the_sequence_length() {
    let encoder = AudioEncoder::empty(&encoder_dims(1));
    let mel = Tensor::zeros(&[1, 4, 3000], DType::Float32);
    let out = encoder.forward(&mel).unwrap();
    assert_eq!(out.dims().unwrap(), [1, 1500, 128]);
}

#[test]
#[ignore = "GPU: inspect full Whisper encoder execution plan"]
fn encoder_plan_has_one_flash_attention_per_block() {
    // The encoder gates on its activations' device, which follows the weights
    // onto the process default device.
    let device = svod_dtype::default_device::default_device();
    if !svod_tk3::ops::supported(&device) {
        eprintln!("skipping: no tile kernels on {device:?}");
        return;
    }
    let encoder = AudioEncoder::empty(&encoder_dims(32));
    let mel = Tensor::zeros(&[1, 4, 3000], DType::Float32);
    let out = encoder.forward(&mel).unwrap();
    assert_eq!(out.dims().unwrap(), [1, 1500, 128]);

    let plan = out.prepare().unwrap();
    // `unique_kernel_name` suffixes the n-th kernel sharing a name with `n{n-1}`
    // from a PROCESS-wide counter, so whether these dispatches are named
    // `flash_attention` or `flash_attentionn7` depends on what else the test
    // binary compiled first. Match the base name, not the whole entry point.
    let is_flash_attention = |entry: &str| {
        entry.strip_prefix("flash_attention").is_some_and(|suffix| {
            suffix.is_empty()
                || suffix.strip_prefix('n').is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
        })
    };
    let flash_attention = plan.kernels().filter(|kernel| is_flash_attention(&kernel.entry_point)).count();
    assert_eq!(flash_attention, 32, "expected one handwritten flash-attention dispatch per encoder block");
}
