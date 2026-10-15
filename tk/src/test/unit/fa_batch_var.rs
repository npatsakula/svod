//! Flash attention under a JIT `batch_var`: the batch is a runtime variable that
//! reaches the kernel only as its launch grid's z extent, while every buffer stays
//! at the variable's capacity.

use svod_dtype::DType;
use svod_ir::SInt;
use svod_macros::jit_wrapper;
use svod_tensor::jit::{InputSpec, JitError};
use svod_tensor::{Tensor, Variable};
use test_case::test_case;

use crate::kernels::fa::{FA_SUPPORTED_ARCHS, FaOpts, flash_attention_packed, flash_attention_with};

const MAX_B: usize = 3;
const N: usize = 256;
const H: usize = 4;
const D: usize = 64;
/// Valid keys per batch row: a full row, a partial last KV block, a single key.
const LENS: [i32; MAX_B] = [N as i32, 77, 1];

/// How the keys are masked: not at all, by per-row lengths, or by a `[B, N]`
/// validity table.
#[derive(Clone, Copy, Debug)]
enum Mask {
    None,
    Lens,
    Keys,
}

/// The `[MAX_B, N]` key validity: every key, every third one hidden, a single
/// key.
fn key_table() -> Vec<i32> {
    (0..MAX_B * N)
        .map(|i| match i / N {
            0 => 1,
            1 => i32::from(i % N % 3 != 1),
            _ => i32::from(i % N == 5),
        })
        .collect()
}

struct Attention {
    causal: bool,
}

impl Attention {
    fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor, opts: FaOpts) -> Result<Tensor, JitError> {
        let opts = FaOpts { causal: self.causal, ..opts };
        let out = flash_attention_with(q, k, v, opts)
            .map_err(|e| JitError::Build { source: Box::new(e) })?
            .expect("the FA kernel applies on a supported device");
        Ok(out.cast(DType::Float32))
    }

    /// Over one packed `[B, N, 3H, D]` operand, the key table masking it.
    fn forward_packed(&self, qkv: &Tensor, keys: &Tensor) -> Result<Tensor, JitError> {
        let opts = FaOpts { causal: self.causal, key_mask: Some(keys), ..Default::default() };
        let out = flash_attention_packed(qkv, (H, H), opts)
            .map_err(|e| JitError::Build { source: Box::new(e) })?
            .expect("the FA kernel applies on a supported device");
        Ok(out.cast(DType::Float32))
    }
}

jit_wrapper! {
    AttentionJit(Attention) {
        inputs { q: Tensor, k: Tensor, v: Tensor }
        batch_var b: (1, MAX_B),
        outputs { o }

        build(q, k, v) { model.forward(q, k, v, FaOpts::default()) }
    }
}

jit_wrapper! {
    MaskedAttentionJit(Attention) {
        inputs { q: Tensor, k: Tensor, v: Tensor, lens: Tensor }
        batch_var b: (1, MAX_B),
        outputs { o }

        build(q, k, v, lens) { model.forward(q, k, v, FaOpts { key_lens: Some(lens), ..Default::default() }) }
    }
}

jit_wrapper! {
    KeyTableAttentionJit(Attention) {
        inputs { q: Tensor, k: Tensor, v: Tensor, keys: Tensor }
        batch_var b: (1, MAX_B),
        outputs { o }

        build(q, k, v, keys) { model.forward(q, k, v, FaOpts { key_mask: Some(keys), ..Default::default() }) }
    }
}

/// The surface the test drives on either wrapper.
trait BatchPlan {
    /// The plan's flash-attention dispatches as `(entry point, var names)`. Entry
    /// points carry an `n{k}` instance suffix, so the base name is matched.
    fn fa_dispatches(&self) -> Vec<(String, Vec<String>)>;
    /// Execute at batch `b` and read the live output.
    fn run(&mut self, b: usize) -> Vec<f32>;
}

macro_rules! batch_plan {
    ($jit:ty) => {
        impl BatchPlan for $jit {
            fn fa_dispatches(&self) -> Vec<(String, Vec<String>)> {
                self.prepared_kernels()
                    .expect("kernels")
                    .into_iter()
                    .filter(|k| k.kernel.entry_point.starts_with("flash_attention"))
                    .map(|k| (k.kernel.entry_point.clone(), k.kernel.var_names.clone()))
                    .collect()
            }

            fn run(&mut self, b: usize) -> Vec<f32> {
                self.execute_bound(b as i64).expect("execute");
                assert_eq!(self.o_shape().expect("shape"), vec![b, N, H, D], "the output follows the live batch");
                self.o_to_vec::<f32>().expect("read")
            }
        }
    };
}
batch_plan!(AttentionJit);
batch_plan!(MaskedAttentionJit);
jit_wrapper! {
    PackedAttentionJit(Attention) {
        inputs { qkv: Tensor, keys: Tensor }
        batch_var b: (1, MAX_B),
        outputs { o }

        build(qkv, keys) { model.forward_packed(qkv, keys) }
    }
}

batch_plan!(KeyTableAttentionJit);
batch_plan!(PackedAttentionJit);

fn copy_into(t: &Tensor, dst: &mut svod_device::Buffer) {
    t.realize().expect("realize");
    let src = t.buffer().expect("realized buffer");
    let mut bytes = vec![0u8; src.size()];
    src.copyout(&mut bytes).expect("copyout");
    dst.copyin(&bytes).expect("copyin");
}

/// `[MAX_B, N, H, D]` f16 operand, deterministic and away from unit variance.
fn operand(seed: f32) -> Tensor {
    let data: Vec<f32> = (0..MAX_B * N * H * D).map(|i| (i as f32 * 0.37 + seed).sin() * 1.5).collect();
    let t = Tensor::from_slice(data.as_slice()).try_reshape([MAX_B, N, H, D]).expect("reshape").cast(DType::Float16);
    t.realize().expect("realize");
    t
}

/// `SVOD_DEVICE=CUDA:0 cargo test -p svod-tk --lib fa_batch_var -- --ignored --nocapture`.
///
/// One plan prepared at the batch capacity and executed at every bound batch
/// matches both the kernel built for that concrete batch and SDPA, with exactly
/// one flash-attention dispatch whose grid takes the batch variable. A `[B, N]`
/// key table takes the queries' batch variable as the operands do.
#[test_case(false, Mask::None; "bidirectional")]
#[test_case(true, Mask::None; "causal")]
#[test_case(false, Mask::Lens; "bidirectional key mask")]
#[test_case(true, Mask::Lens; "causal key mask")]
#[test_case(false, Mask::Keys; "bidirectional key table")]
#[test_case(true, Mask::Keys; "causal key table")]
#[ignore]
fn fa_batch_var_matches_concrete_batch(causal: bool, mask: Mask) {
    if !super::device_supported(FA_SUPPORTED_ARCHS) {
        eprintln!("skip fa_batch_var_matches_concrete_batch: unsupported device/toolchain");
        return;
    }
    let (q, k, v) = (operand(0.0), operand(1.0), operand(2.0));
    let keys = key_table();
    let spec = InputSpec::new(&[MAX_B, N, H, D], DType::Float16);
    let mut plan: Box<dyn BatchPlan> = match mask {
        Mask::None => {
            let mut jit = AttentionJit::new(Attention { causal });
            jit.prepare(spec.clone(), spec.clone(), spec).expect("prepare");
            copy_into(&q, jit.q_mut().expect("q"));
            copy_into(&k, jit.k_mut().expect("k"));
            copy_into(&v, jit.v_mut().expect("v"));
            Box::new(jit)
        }
        Mask::Lens => {
            let mut jit = MaskedAttentionJit::new(Attention { causal });
            jit.prepare(spec.clone(), spec.clone(), spec, InputSpec::i32(&[MAX_B])).expect("prepare");
            copy_into(&q, jit.q_mut().expect("q"));
            copy_into(&k, jit.k_mut().expect("k"));
            copy_into(&v, jit.v_mut().expect("v"));
            copy_into(&Tensor::from_slice(LENS.as_slice()), jit.lens_mut().expect("lens"));
            Box::new(jit)
        }
        Mask::Keys => {
            let mut jit = KeyTableAttentionJit::new(Attention { causal });
            jit.prepare(spec.clone(), spec.clone(), spec, InputSpec::i32(&[MAX_B, N])).expect("prepare");
            copy_into(&q, jit.q_mut().expect("q"));
            copy_into(&k, jit.k_mut().expect("k"));
            copy_into(&v, jit.v_mut().expect("v"));
            let table = Tensor::from_slice(keys.as_slice()).try_reshape([MAX_B, N]).expect("reshape");
            copy_into(&table, jit.keys_mut().expect("keys"));
            Box::new(jit)
        }
    };
    check_every_batch(plan.as_mut(), (&q, &k, &v), causal, mask);
}

/// `SVOD_DEVICE=CUDA:0 cargo test --release -p svod-tk --lib fa_batch_var -- --ignored --nocapture`.
///
/// The packed `[B, N, 3H, D]` operand takes the batch variable as the split
/// ones do: one plan, the key table masking it, matches the split kernel built
/// for each concrete batch and SDPA.
#[test_case(false; "bidirectional")]
#[test_case(true; "causal")]
#[ignore]
fn fa_batch_var_packed_matches_concrete_batch(causal: bool) {
    if !super::device_supported(FA_SUPPORTED_ARCHS) {
        eprintln!("skip fa_batch_var_packed_matches_concrete_batch: unsupported device/toolchain");
        return;
    }
    let (q, k, v) = (operand(0.0), operand(1.0), operand(2.0));
    let qkv = Tensor::cat(&[&q, &k, &v], 2).expect("pack the heads");
    let mut jit = PackedAttentionJit::new(Attention { causal });
    jit.prepare(InputSpec::new(&[MAX_B, N, 3 * H, D], DType::Float16), InputSpec::i32(&[MAX_B, N])).expect("prepare");
    copy_into(&qkv, jit.qkv_mut().expect("qkv"));
    let table = Tensor::from_slice(key_table().as_slice()).try_reshape([MAX_B, N]).expect("reshape");
    copy_into(&table, jit.keys_mut().expect("keys"));
    check_every_batch(&mut jit, (&q, &k, &v), causal, Mask::Keys);
}

/// Execute `plan` at every bound batch, down and back to one, against the split
/// kernel built for that concrete batch and SDPA; the plan holds exactly one
/// flash-attention dispatch, whose grid takes the batch variable.
fn check_every_batch(plan: &mut dyn BatchPlan, (q, k, v): (&Tensor, &Tensor, &Tensor), causal: bool, mask: Mask) {
    let keys = key_table();
    let dispatches = plan.fa_dispatches();
    assert_eq!(dispatches.len(), 1, "one flash-attention dispatch, got {dispatches:?}");
    assert!(dispatches[0].1.iter().any(|n| n == "b"), "the FA grid takes `b`: {dispatches:?}");

    for b in (1..=MAX_B).rev().chain([1]) {
        let got = plan.run(b);

        let live = |t: &Tensor| t.try_shrink([Some((0, b as isize)), None, None, None]).expect("shrink").contiguous();
        let (qb, kb, vb) = (live(q), live(k), live(v));
        let lens = Tensor::from_slice(&LENS[..b]);
        let table = Tensor::from_slice(&keys[..b * N]).try_reshape([b, N]).expect("reshape");
        let (by_lens, by_keys) = (matches!(mask, Mask::Lens), matches!(mask, Mask::Keys));
        let opts = FaOpts {
            causal,
            key_lens: by_lens.then_some(&lens),
            key_mask: by_keys.then_some(&table),
            ..Default::default()
        };
        let concrete = flash_attention_with(&qb, &kb, &vb, opts).expect("fa").expect("applies").cast(DType::Float32);
        concrete.realize().expect("realize concrete");
        let (lens, keys) = (by_lens.then_some(&LENS[..b]), by_keys.then_some(&keys[..b * N]));
        let reference = super::fa::fa_reference(&qb, &kb, &vb, causal, lens, None, (keys, None));
        reference.realize().expect("realize reference");

        let same = svod_tensor::testing::allclose_f32(&got, &concrete.as_vec::<f32>().expect("read"), 1e-3, 1e-3);
        assert!(same.ok, "b={b} vs concrete batch: {}", same.message);
        let sdpa = svod_tensor::testing::allclose_f32(&got, &reference.as_vec::<f32>().expect("read"), 2e-2, 2e-2);
        assert!(sdpa.ok, "b={b} vs SDPA: {}", sdpa.message);
    }
}

/// A symbolic batch is accepted only as a bare bounded variable on dim 0; the
/// dims come back at its capacity. Host-only — no kernel is built.
#[test]
fn batched_dims_takes_a_bare_batch_variable() {
    let var = Variable::new("b", 1, 6);
    let bound = var.bind(2).expect("bind");
    let buffer = Tensor::empty(&[6, 3], DType::Float32);
    let t = buffer.try_shrink([Some((SInt::Const(0), bound.as_sint())), None]).expect("shrink");
    let (dims, batch) = crate::launch::batched_dims(&t, "k", "t", 2).expect("bare variable");
    assert_eq!(dims, vec![6, 3]);
    let batch = batch.expect("symbolic batch");
    assert_eq!(batch.dim, bound.as_sint());
    assert!(std::sync::Arc::ptr_eq(&batch.var, var.uop()), "the grid extent is the unbound variable");

    let (dims, batch) = crate::launch::batched_dims(&buffer, "k", "t", 2).expect("static");
    assert_eq!((dims, batch.is_none()), (vec![6, 3], true));

    let swapped = t.try_permute(&[1, 0]).expect("permute");
    let err = crate::launch::batched_dims(&swapped, "k", "t", 2).expect_err("a symbolic non-leading dim");
    assert!(matches!(err, crate::launch::Error::OperandSymbolicDim { axis: 1, .. }), "{err:?}");
}
