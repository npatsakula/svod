use svod_dtype::DType;

use crate::jit::InputSpec;
use crate::modernbert::{ClassifierPooling, ModernBert, ModernBertConfig, ModernBertJit};

/// Tiny config (2 layers, hidden 32) so the CPU JIT graph compiles in seconds.
fn tiny_cfg() -> ModernBertConfig {
    ModernBertConfig {
        vocab_size: 64,
        hidden_size: 32,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        intermediate_size: 64,
        max_position_embeddings: 128,
        layer_norm_eps: 1e-5,
        global_rope_theta: 10_000.0,
        local_rope_theta: 10_000.0,
        local_attention: 16,
        global_attn_every_n_layers: 3,
        pad_token_id: 0,
        tie_word_embeddings: true,
        decoder_bias: true,
        dtype: DType::Float32,
        max_batch_size: 4,
        num_labels: 3,
        classifier_pooling: ClassifierPooling::Cls,
        classifier_bias: false,
        norm_bias: false,
        id2label: vec![],
    }
}

/// The JIT wrapper must thread `attention_mask` through to the encoder. This
/// is the regression guard for the dropped-mask bug: an all-attend forward
/// (every mask entry 1) must differ from a masked forward (last half of the
/// sequence masked out). If the mask were hardcoded to `None` in the wrapper,
/// both runs would be byte-identical.
#[test]
#[ignore = "heavy: 2-layer ModernBERT JIT graph compile through the CPU backend"]
fn jit_mask_changes_output() {
    let cfg = tiny_cfg();
    let mut jit = ModernBertJit::new(ModernBert::empty(cfg.clone()));
    let seq_len = 8usize;
    jit.prepare(InputSpec::i64(&[cfg.max_batch_size, seq_len]), InputSpec::i64(&[cfg.max_batch_size, seq_len]))
        .expect("prepare");

    let ids: Vec<i64> = (1..=(cfg.max_batch_size * seq_len) as i64).collect();

    // Run A: all tokens attended (mask all ones).
    let out_all = run(&mut jit, &ids, &vec![1i64; cfg.max_batch_size * seq_len], cfg.max_batch_size);
    // Run B: last half of each sequence masked out.
    let mask_half: Vec<i64> = (0..cfg.max_batch_size)
        .flat_map(|_| {
            let mut row = vec![1i64; seq_len];
            for m in &mut row[seq_len / 2..] {
                *m = 0;
            }
            row
        })
        .collect();
    let out_half = run(&mut jit, &ids, &mask_half, cfg.max_batch_size);

    assert_ne!(out_all, out_half, "mask had no effect on JIT output — wrapper dropped it?");
}

/// One compiled plan serves multiple batch sizes: prepare at `max_batch_size`
/// and rebind `b` to 1, 2, … at execute time. This is the JIT batch-rebind
/// contract — only possible because `Tensor::embedding` now carries the
/// symbolic batch dim through (it previously required concrete index shapes).
#[test]
#[ignore = "heavy: 2-layer ModernBERT JIT graph compile through the CPU backend"]
fn jit_rebinds_batch_without_reprepare() {
    let cfg = tiny_cfg();
    let max_batch = cfg.max_batch_size;
    let mut jit = ModernBertJit::new(ModernBert::empty(cfg.clone()));
    let seq_len = 5usize;
    jit.prepare(InputSpec::i64(&[max_batch, seq_len]), InputSpec::i64(&[max_batch, seq_len])).expect("prepare");

    // Prep full-batch buffers; only the first `b` rows are live per execute.
    let ids: Vec<i64> = (1..=(max_batch * seq_len) as i64).collect();
    let mask = vec![1i64; max_batch * seq_len];

    // The output buffer stays sized for `max_batch` (the plan bakes the upper
    // bound), but the declared output's live shape follows `b`, so the read-back
    // is exactly the `(b, L, D)` rows the plan computed.
    let d = cfg.hidden_size;
    for b in [1, 2, max_batch] {
        let out = run(&mut jit, &ids, &mask, b);
        assert_eq!(jit.hidden_shape().expect("live shape"), vec![b, seq_len, d], "live shape must follow b={b}");
        assert_eq!(out.len(), b * seq_len * d);
        assert!(out.iter().all(|v| v.is_finite()), "non-finite output for b={b}");
    }
}

/// Write inputs and execute with batch rebindable to `b`. Returns the live
/// `(b, L, D)` output (only `b` rows read back).
fn run(jit: &mut ModernBertJit, ids: &[i64], mask: &[i64], b: usize) -> Vec<f32> {
    jit.input_ids_view_mut::<i64>().expect("input_ids view").as_slice_mut().expect("contiguous").copy_from_slice(ids);
    jit.attention_mask_view_mut::<i64>()
        .expect("attention_mask view")
        .as_slice_mut()
        .expect("contiguous")
        .copy_from_slice(mask);
    jit.execute_bound(b as i64).expect("execute");
    jit.hidden_to_vec::<f32>().expect("output read")
}
