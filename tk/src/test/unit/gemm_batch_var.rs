//! `gemm_nt` under a JIT `batch_var`: each batch is its own GEMM over the
//! static rows behind it, on grid z, whose extent is the runtime batch; every
//! buffer stays at the variable's capacity.

use svod_dtype::DType;
use svod_ir::SInt;
use svod_macros::jit_wrapper;
use svod_tensor::jit::{InputSpec, JitError};
use svod_tensor::{Tensor, Variable};
use test_case::test_case;

use super::gemm::{BF16_REL_TOL, SWIGLU_REL_TOL, operand, pair_rows, to_f32_vec};
use super::rel_err;
use crate::kernels::gemm::{Epilogue, GEMM_NT_SUPPORTED_ARCHS, gemm_nt, gemm_nt_with_epilogue, swiglu_pair_width};

const MAX_B: usize = 3;
const L: usize = 128;
const K: usize = 192;
const N: usize = 256;

#[derive(Clone, Copy, Debug)]
enum Epi {
    Plain,
    Add,
    SwiGlu,
}

/// One projection; the weight is static, `pair` the SwiGLU row arrangement.
struct Proj {
    w: Tensor,
    pair: usize,
}

impl Proj {
    fn forward(&self, x: &Tensor, epi: Epilogue<&Tensor>) -> Result<Tensor, JitError> {
        let y = gemm_nt_with_epilogue(x, &self.w, epi)
            .map_err(|e| JitError::Build { source: Box::new(e) })?
            .expect("the hand GEMM applies on a supported device");
        Ok(y.cast(DType::Float32))
    }
}

jit_wrapper! {
    PlainJit(Proj) {
        inputs { x: Tensor }
        batch_var b: (1, MAX_B),
        outputs { y }
        build(x) { model.forward(x, Epilogue::Plain) }
    }
}

jit_wrapper! {
    AddJit(Proj) {
        inputs { x: Tensor, res: Tensor }
        batch_var b: (1, MAX_B),
        outputs { y }
        build(x, res) { model.forward(x, Epilogue::Add(res)) }
    }
}

jit_wrapper! {
    SwiGluJit(Proj) {
        inputs { x: Tensor }
        batch_var b: (1, MAX_B),
        outputs { y }
        build(x) { model.forward(x, Epilogue::SwiGlu { pair: model.pair }) }
    }
}

/// The surface the test drives on every wrapper.
trait BatchPlan {
    /// The plan's GEMM dispatches as `(entry point, var names)`, matched by base
    /// name (entry points carry an `n{k}` instance suffix).
    fn gemm_dispatches(&self) -> Vec<(String, Vec<String>)>;
    /// Execute at batch `b` and read the live output.
    fn run(&mut self, b: usize) -> Vec<f32>;
}

macro_rules! batch_plan {
    ($jit:ty) => {
        impl BatchPlan for $jit {
            fn gemm_dispatches(&self) -> Vec<(String, Vec<String>)> {
                self.prepared_kernels()
                    .expect("kernels")
                    .into_iter()
                    .filter(|k| k.kernel.entry_point.starts_with("gemm_nt"))
                    .map(|k| (k.kernel.entry_point.clone(), k.kernel.var_names.clone()))
                    .collect()
            }

            fn run(&mut self, b: usize) -> Vec<f32> {
                self.execute_bound(b as i64).expect("execute");
                assert_eq!(self.y_shape().expect("shape")[0], b, "the output follows the live batch");
                self.y_to_vec::<f32>().expect("read")
            }
        }
    };
}
batch_plan!(PlainJit);
batch_plan!(AddJit);
batch_plan!(SwiGluJit);

fn copy_into(t: &Tensor, dst: &mut svod_device::Buffer) {
    t.realize().expect("realize");
    let src = t.buffer().expect("realized buffer");
    let mut bytes = vec![0u8; src.size()];
    src.copyout(&mut bytes).expect("copyout");
    dst.copyin(&bytes).expect("copyin");
}

fn batched(rows: usize, cols: usize, seed: f32) -> Tensor {
    operand(MAX_B * rows, cols, DType::BFloat16, seed).try_reshape([MAX_B, rows, cols]).expect("reshape")
}

fn live(t: &Tensor, b: usize) -> Tensor {
    let t = t.try_shrink([Some((0, b as isize)), None, None]).expect("shrink").contiguous();
    t.realize().expect("realize");
    t
}

/// `SVOD_DEVICE=CUDA:0 cargo test -p svod-tk --lib gemm_batch_var -- --ignored --nocapture`.
///
/// One plan prepared at the batch capacity and executed at every bound batch
/// matches the GEMM built for that concrete batch and the graph it replaces,
/// with exactly one GEMM dispatch whose grid takes the batch variable.
#[test_case(Epi::Plain; "plain")]
#[test_case(Epi::Add; "residual add")]
#[test_case(Epi::SwiGlu; "swiglu")]
#[ignore]
fn gemm_batch_var_matches_concrete_batch(epi: Epi) {
    if !super::device_supported(GEMM_NT_SUPPORTED_ARCHS) {
        eprintln!("skip gemm_batch_var_matches_concrete_batch: unsupported device/toolchain");
        return;
    }
    let x = batched(L, K, 0.31);
    let w = operand(N, K, DType::BFloat16, 0.17);
    let pair = swiglu_pair_width(&x.device()).expect("a common pair width");
    let cols = if matches!(epi, Epi::SwiGlu) { N / 2 } else { N };
    let res = batched(L, cols, 0.53);
    let proj = Proj { w: if matches!(epi, Epi::SwiGlu) { pair_rows(&w, pair) } else { w.clone() }, pair };
    let kernel_w = proj.w.clone();

    let spec = |cols| InputSpec::new(&[MAX_B, L, cols], DType::BFloat16);
    let mut plan: Box<dyn BatchPlan> = match epi {
        Epi::Plain => {
            let mut jit = PlainJit::new(proj);
            jit.prepare(spec(K)).expect("prepare");
            copy_into(&x, jit.x_mut().expect("x"));
            Box::new(jit)
        }
        Epi::Add => {
            let mut jit = AddJit::new(proj);
            jit.prepare(spec(K), spec(cols)).expect("prepare");
            copy_into(&x, jit.x_mut().expect("x"));
            copy_into(&res, jit.res_mut().expect("res"));
            Box::new(jit)
        }
        Epi::SwiGlu => {
            let mut jit = SwiGluJit::new(proj);
            jit.prepare(spec(K)).expect("prepare");
            copy_into(&x, jit.x_mut().expect("x"));
            Box::new(jit)
        }
    };
    let dispatches = plan.gemm_dispatches();
    assert_eq!(dispatches.len(), 1, "one GEMM dispatch, got {dispatches:?}");
    assert!(dispatches[0].1.iter().any(|n| n == "b"), "the GEMM grid takes `b`: {dispatches:?}");

    for b in (1..=MAX_B).rev().chain([1]) {
        let got = plan.run(b);
        let (xb, resb) = (live(&x, b), live(&res, b));
        let kernel_epi = match epi {
            Epi::Plain => Epilogue::Plain,
            Epi::Add => Epilogue::Add(&resb),
            Epi::SwiGlu => Epilogue::SwiGlu { pair },
        };
        let concrete = gemm_nt_with_epilogue(&xb, &kernel_w, kernel_epi).expect("build").expect("applies");
        let linear = xb.linear().weight(&w).call().expect("reference linear");
        let (reference, tol) = match epi {
            Epi::Plain => (linear, BF16_REL_TOL),
            Epi::Add => (linear.try_add(&resb).expect("add"), BF16_REL_TOL),
            Epi::SwiGlu => {
                let halves = linear.split(&[N / 2, N / 2], -1).expect("split");
                (halves[0].silu().expect("silu").try_mul(&halves[1]).expect("gate·up"), SWIGLU_REL_TOL)
            }
        };
        let concrete_err = rel_err(&got, &to_f32_vec(&concrete));
        assert!(concrete_err < BF16_REL_TOL, "{epi:?} b={b} vs concrete batch: {concrete_err}");
        let graph_err = rel_err(&got, &to_f32_vec(&reference));
        assert!(graph_err < tol, "{epi:?} b={b} vs the graph: {graph_err}");
        println!("gemm[{epi:?}] b={b}: vs concrete {concrete_err:e}, vs graph {graph_err:e}");
    }
}

/// `[MAX_B, rows, K]` with its batch bound to a runtime variable.
fn symbolic(rows: usize, cols: usize, seed: f32) -> Tensor {
    let b = Variable::new("b", 1, MAX_B as i64).bind(2).expect("bind");
    batched(rows, cols, seed).try_shrink([Some((SInt::Const(0), b.as_sint())), None, None]).expect("shrink")
}

/// A symbolic batch over rows no tile divides declines (each batch must tile on
/// its own); a residual off `x`'s batch is a malformed request.
#[test]
#[ignore]
fn gemm_batch_var_outcomes_gpu() {
    if !super::device_supported(GEMM_NT_SUPPORTED_ARCHS) {
        eprintln!("skip gemm_batch_var_outcomes_gpu: unsupported device/toolchain");
        return;
    }
    let w = operand(N, K, DType::BFloat16, 0.17);
    assert!(gemm_nt(&symbolic(80, K, 0.31), &w).expect("builds").is_none(), "80 rows per batch do not tile");

    let x = symbolic(L, K, 0.31);
    let res = batched(L, N, 0.53);
    let err = gemm_nt_with_epilogue(&x, &w, Epilogue::Add(&res)).expect_err("a static residual on a symbolic batch");
    assert!(matches!(err, crate::launch::Error::OperandShape { operand: "residual", .. }), "{err:?}");
}

/// The operand gate: static past a leading runtime batch, nothing else symbolic.
#[test]
fn static_past_batch_gates_the_operand() {
    let x = symbolic(L, K, 0.31);
    assert!(crate::static_past_batch(&x));
    assert!(crate::static_past_batch(&batched(L, K, 0.31)));
    assert!(!crate::static_past_batch(&x.try_permute(&[1, 0, 2]).expect("permute")));
}
