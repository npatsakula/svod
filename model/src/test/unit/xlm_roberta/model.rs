use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::{Module, StateDict};

use crate::jit::InputSpec;
use crate::test::unit::encoder_head::real_rows;
use crate::test::unit::max_abs_delta;
use crate::xlm_roberta::{XlmRobertaConfig, XlmRobertaJit, XlmRobertaModel};

/// A 2-layer, 32-wide backbone with 20 position rows.
pub fn tiny_cfg() -> XlmRobertaConfig {
    XlmRobertaConfig {
        vocab_size: 100,
        hidden_size: 32,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        intermediate_size: 64,
        max_position_embeddings: 20,
        type_vocab_size: 1,
        layer_norm_eps: 1e-5,
        pad_token_id: 1,
        num_labels: 2,
        id2label: Vec::new(),
        dtype: DType::Float32,
        max_batch_size: 1,
    }
}

#[test]
fn forward_output_shape() {
    let model = XlmRobertaModel::empty(tiny_cfg());
    let ids = svod_tensor::Tensor::from_slice([0i64, 10, 20, 2, 1]).try_reshape([1isize, 5]).unwrap();
    let out = model.forward(&ids, None).unwrap();
    out.realize().unwrap();
    let v = out.as_vec::<f32>().unwrap();
    assert_eq!(v.len(), 5 * 32);
    assert!(v.iter().all(|x| x.is_finite()));
}

/// Every key `#[derive(Module)]` emits for a 2-layer tiny backbone, in the
/// published `BAAI/bge-m3` naming. Drift here is a checkpoint-compatibility
/// break. Norm biases are absent from a freshly-built model and loaded only
/// when the checkpoint carries them.
pub fn expected_keys() -> Vec<String> {
    let mut keys: Vec<String> = ["word_embeddings", "position_embeddings", "token_type_embeddings"]
        .iter()
        .map(|t| format!("embeddings.{t}.weight"))
        .collect();
    keys.push("embeddings.LayerNorm.weight".to_string());
    for i in 0..2 {
        for k in [
            "attention.self.query.weight",
            "attention.self.query.bias",
            "attention.self.key.weight",
            "attention.self.key.bias",
            "attention.self.value.weight",
            "attention.self.value.bias",
            "attention.output.dense.weight",
            "attention.output.dense.bias",
            "attention.output.LayerNorm.weight",
            "intermediate.dense.weight",
            "intermediate.dense.bias",
            "output.dense.weight",
            "output.dense.bias",
            "output.LayerNorm.weight",
        ] {
            keys.push(format!("encoder.layer.{i}.{k}"));
        }
    }
    keys.sort();
    keys
}

pub fn sorted_keys(sd: &StateDict) -> Vec<String> {
    let mut keys: Vec<String> = sd.keys().cloned().collect();
    keys.sort();
    keys
}

/// The emitted key set is exactly the published layout, at the root and nested
/// under a prefix (which must not grow a leading dot).
#[test]
fn state_dict_keys_match_published_layout() {
    let model = XlmRobertaModel::empty(tiny_cfg());
    let want = expected_keys();
    assert_eq!(sorted_keys(&model.state_dict("")), want);

    let want_nested: Vec<String> = want.iter().map(|k| format!("m.{k}")).collect();
    assert_eq!(sorted_keys(&model.state_dict("m")), want_nested);
}

#[test]
fn state_dict_round_trip() {
    let model = XlmRobertaModel::empty(tiny_cfg());
    let sd = model.state_dict("");
    let mut model2 = XlmRobertaModel::empty(tiny_cfg());
    model2.load_state_dict(&sd, "").unwrap();
    assert_eq!(sorted_keys(&sd), sorted_keys(&model2.state_dict("")));
}

/// Under a free JIT batch — BGE-M3's wrappers — a tiling length takes tk's
/// flash attention with the padding mask, where tk has it. Rebound to fewer
/// rows than prepared, the bf16 plan's real tokens track the f32 eager forward
/// (SDPA); the same plan ignoring the padding must not, or the bound proves
/// nothing.
#[test]
#[ignore = "heavy: compiles an XLM-R JIT plan, on a device with flash attention"]
fn free_batch_jit_takes_flash_attention_with_the_mask() {
    let (dtype, device) = (crate::default_compute_dtype(), svod_dtype::default_device::default_device());
    if crate::attention::flash_attention_tile(&device, &dtype).is_none() {
        eprintln!(
            "skip free_batch_jit_takes_flash_attention_with_the_mask: no flash attention for {dtype:?} on {device:?}"
        );
        return;
    }
    let (rows, live, len, short) = (4, 2, svod_tk::FLASH_ATTENTION_SEQUENCE_MULTIPLE, 40);
    // Head dim 16, the narrowest the kernel tiles.
    let cfg = XlmRobertaConfig {
        hidden_size: 64,
        intermediate_size: 128,
        max_position_embeddings: len + 2,
        max_batch_size: rows,
        ..tiny_cfg()
    };
    let reference = XlmRobertaModel::empty(cfg.clone());
    let weights = reference.state_dict("");
    weights.values().for_each(|w| w.realize().expect("realize weight"));
    let model =
        XlmRobertaModel::from_state_dict(&weights, XlmRobertaConfig { dtype, ..cfg.clone() }).expect("bf16 copy");

    // Row 0 fills the length, row 1 is `short` tokens padded with
    // `pad_token_id`; rows past `live` are unused.
    let lens = [len, short, 1, 1];
    let ids: Vec<i64> =
        lens.iter().flat_map(|&n| (0..len).map(move |t| if t < n { 3 + (t as i64 * 7) % 90 } else { 1 })).collect();
    let mask: Vec<i64> = lens.iter().flat_map(|&n| (0..len).map(move |t| i64::from(t < n))).collect();

    let mut jit = XlmRobertaJit::new(model);
    jit.prepare(InputSpec::i64(&[rows, len]), InputSpec::i64(&[rows, len])).expect("prepare");
    let kernels = jit.prepared_kernels().expect("kernels");
    assert!(
        kernels.iter().any(|k| k.kernel.entry_point.starts_with("flash_attention")),
        "the plan has no flash attention"
    );
    let d = cfg.hidden_size;
    let mut run = |mask: &[i64]| -> Vec<f32> {
        jit.input_ids_view_mut::<i64>().expect("ids").as_slice_mut().expect("contiguous").copy_from_slice(&ids);
        jit.attention_mask_view_mut::<i64>().expect("mask").as_slice_mut().expect("contiguous").copy_from_slice(mask);
        jit.execute_bound(live as i64).expect("execute");
        // bf16 is the top half of an f32.
        let mut raw = vec![0u8; live * len * d * 2];
        jit.hidden().expect("hidden").copyout_prefix(&mut raw).expect("read");
        raw.as_chunks::<2>().0.iter().map(|&h| f32::from_bits(u32::from(u16::from_le_bytes(h)) << 16)).collect()
    };

    let live_mask = &mask[..live * len];
    let shape = [live as isize, len as isize];
    let ids_t = Tensor::from_slice(&ids[..live * len]).try_reshape(shape).unwrap();
    let mask_t = Tensor::from_slice(live_mask).try_reshape(shape).unwrap();
    let want = reference.forward(&ids_t, Some(&mask_t)).expect("f32 forward");
    want.realize().expect("realize");
    let want = real_rows(&want.as_vec::<f32>().expect("read"), live_mask, d);

    let drift = max_abs_delta(&real_rows(&run(&mask), live_mask, d), &want);
    let mut unmasked = mask.clone();
    unmasked[..live * len].fill(1);
    let control = max_abs_delta(&real_rows(&run(&unmasked), live_mask, d), &want);
    eprintln!("bf16 free-batch jit max |Δ| from f32: {drift:.3e}; padding ignored: {control:.3e}");
    // The weights are unseeded: over runs, bf16 rounding lands at 0.035-0.049
    // and attending the padding at 0.65-1.2.
    let bound = 0.2;
    assert!(drift < bound, "the free-batch plan drifts from the f32 forward: {drift:.3e}");
    assert!(control > bound, "ignoring the padding stays within the bound ({control:.3e}): it cannot catch a mask bug");
}
