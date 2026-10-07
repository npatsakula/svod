//! Flash attention under a JIT `batch_var`: the batch is a runtime variable that
//! reaches the kernel only as its launch grid's z extent, while every buffer stays
//! at the variable's capacity.

use svod_dtype::DType;
use svod_ir::SInt;
use svod_macros::jit_wrapper;
use svod_tensor::jit::{InputSpec, JitError};
use svod_tensor::{Tensor, Variable};
use test_case::test_case;

use crate::kernels::fa::{FA_SUPPORTED_ARCHS, FaOpts, flash_attention_with};

const MAX_B: usize = 3;
const N: usize = 256;
const H: usize = 4;
const D: usize = 64;
/// Valid keys per batch row: a full row, a partial last KV block, a single key.
const LENS: [i32; MAX_B] = [N as i32, 77, 1];

struct Attention {
    causal: bool,
}

impl Attention {
    fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor, key_lens: Option<&Tensor>) -> Result<Tensor, JitError> {
        let opts = FaOpts { causal: self.causal, key_lens, seg_start: None };
        let out = flash_attention_with(q, k, v, opts)
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

        build(q, k, v) { model.forward(q, k, v, None) }
    }
}

jit_wrapper! {
    MaskedAttentionJit(Attention) {
        inputs { q: Tensor, k: Tensor, v: Tensor, lens: Tensor }
        batch_var b: (1, MAX_B),
        outputs { o }

        build(q, k, v, lens) { model.forward(q, k, v, Some(lens)) }
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
/// one flash-attention dispatch whose grid takes the batch variable.
#[test_case(false, false; "bidirectional")]
#[test_case(true, false; "causal")]
#[test_case(false, true; "bidirectional key mask")]
#[test_case(true, true; "causal key mask")]
#[ignore]
fn fa_batch_var_matches_concrete_batch(causal: bool, masked: bool) {
    if !super::device_supported(FA_SUPPORTED_ARCHS) {
        eprintln!("skip fa_batch_var_matches_concrete_batch: unsupported device/toolchain");
        return;
    }
    let (q, k, v) = (operand(0.0), operand(1.0), operand(2.0));
    let spec = InputSpec::new(&[MAX_B, N, H, D], DType::Float16);
    let mut plan: Box<dyn BatchPlan> = if masked {
        let mut jit = MaskedAttentionJit::new(Attention { causal });
        jit.prepare(spec.clone(), spec.clone(), spec, InputSpec::i32(&[MAX_B])).expect("prepare");
        copy_into(&q, jit.q_mut().expect("q"));
        copy_into(&k, jit.k_mut().expect("k"));
        copy_into(&v, jit.v_mut().expect("v"));
        copy_into(&Tensor::from_slice(LENS.as_slice()), jit.lens_mut().expect("lens"));
        Box::new(jit)
    } else {
        let mut jit = AttentionJit::new(Attention { causal });
        jit.prepare(spec.clone(), spec.clone(), spec).expect("prepare");
        copy_into(&q, jit.q_mut().expect("q"));
        copy_into(&k, jit.k_mut().expect("k"));
        copy_into(&v, jit.v_mut().expect("v"));
        Box::new(jit)
    };
    let dispatches = plan.fa_dispatches();
    assert_eq!(dispatches.len(), 1, "one flash-attention dispatch, got {dispatches:?}");
    assert!(dispatches[0].1.iter().any(|n| n == "b"), "the FA grid takes `b`: {dispatches:?}");

    for b in (1..=MAX_B).rev().chain([1]) {
        let got = plan.run(b);

        let live = |t: &Tensor| t.try_shrink([Some((0, b as isize)), None, None, None]).expect("shrink").contiguous();
        let (qb, kb, vb) = (live(&q), live(&k), live(&v));
        let lens = Tensor::from_slice(&LENS[..b]);
        let opts = FaOpts { causal, key_lens: masked.then_some(&lens), seg_start: None };
        let concrete = flash_attention_with(&qb, &kb, &vb, opts).expect("fa").expect("applies").cast(DType::Float32);
        concrete.realize().expect("realize concrete");
        let reference = super::fa::fa_reference(&qb, &kb, &vb, causal, masked.then_some(&LENS[..b]), None);
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
