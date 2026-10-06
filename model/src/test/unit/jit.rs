//! `jit_wrapper!` end-to-end: prepare, replay, replication, multi-output
//! plans, `batch_var` rebinding, array slots and `state { .. }` recurrences.
//! No `extern crate self as svod_model;` — the expansion is self-contained.

use svod_macros::jit_wrapper;
use svod_tensor::{PrepareConfig, Tensor};

use crate::jit::{InputSpec, JitError, Result};

fn tensor_err(e: svod_tensor::error::Error) -> JitError {
    JitError::Tensor { source: Box::new(e) }
}

struct AddModel;

impl AddModel {
    fn forward(&self, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        x.try_add(y).map_err(tensor_err)
    }
}

jit_wrapper! {
    AddJit(AddModel) {
        x: Tensor,
        y: Tensor,

        build(x, y) {
            model.forward(x, y)
        }
    }
}

struct SplitModel;

impl SplitModel {
    fn forward(&self, x: &Tensor, y: &Tensor) -> Result<(Tensor, Tensor)> {
        let sum = x.try_add(y).map_err(tensor_err)?;
        let diff = x.try_sub(y).map_err(tensor_err)?;
        Ok((sum, diff))
    }
}

// Two same-dtype outputs from one plan.
jit_wrapper! {
    SplitJit(SplitModel) {
        x: Tensor,
        y: Tensor,

        outputs { sum, diff },

        build(x, y) {
            model.forward(x, y)
        }
    }
}

struct MixedModel;

impl MixedModel {
    /// One f32 output (`x + y`) and one int32 output (`argmax(x)`) — exercises
    /// per-output dtype: each output buffer carries its own dtype.
    fn forward(&self, x: &Tensor, y: &Tensor) -> Result<(Tensor, Tensor)> {
        let sum = x.try_add(y).map_err(tensor_err)?;
        let arg = x.argmax(None).map_err(tensor_err)?;
        Ok((sum, arg))
    }
}

jit_wrapper! {
    MixedJit(MixedModel) {
        x: Tensor,
        y: Tensor,

        outputs { sum, arg },

        build(x, y) {
            model.forward(x, y)
        }
    }
}

/// One compiled plan for every batch size up to the bound: `batch_var` shrinks
/// the batched inputs on dim 0, `bias` opts out with `#[unbatched]`.
struct BatchModel;

impl BatchModel {
    fn forward(&self, rows: &Tensor, bias: &Tensor) -> Result<Tensor> {
        rows.try_add(bias).map_err(tensor_err)
    }
}

jit_wrapper! {
    BatchJit(BatchModel) {
        inputs {
            rows: Tensor,
            #[unbatched] bias: Tensor,
        }
        batch_var b: (1, 4),
        outputs { scaled }

        build(rows, bias) {
            model.forward(rows, bias)
        }
    }
}

/// `[B, T, K] @ [K, N]` + an RMS-norm — a matmul whose M axis is the
/// `(batch, time)` pair. That is where `simplify_merge_adjacent` used to fold
/// the symbolic batch range into the constant time range, which cost the kernel
/// every constant-axis opt (upcast/unroll/local/tensor cores).
struct LinearNormModel;

const LN_T: usize = 8;
const LN_K: usize = 16;
const LN_N: usize = 32;
const LN_MAX_B: usize = 4;

impl LinearNormModel {
    fn forward(&self, x: &Tensor, w: &Tensor) -> Result<Tensor> {
        x.matmul(w).and_then(|y| y.rms_norm(-1, 1e-5)).map_err(tensor_err)
    }
}

jit_wrapper! {
    SymBatchJit(LinearNormModel) {
        inputs {
            x: Tensor,
            #[unbatched] w: Tensor,
        }
        batch_var b: (1, LN_MAX_B),
        outputs { y }

        build(x, w) { model.forward(x, w) }
    }
}

jit_wrapper! {
    ConstBatchJit(LinearNormModel) {
        x: Tensor,
        w: Tensor,

        outputs { y }

        build(x, w) { model.forward(x, w) }
    }
}

/// Array-valued input and output slots: N buffers behind one name.
struct FanModel;

impl FanModel {
    fn forward(&self, xs: [&Tensor; 3]) -> Result<([Tensor; 2], Tensor)> {
        let left = xs[0].try_add(xs[1]).map_err(tensor_err)?;
        let right = xs[1].try_add(xs[2]).map_err(tensor_err)?;
        let total = left.try_add(&right).map_err(tensor_err)?;
        Ok(([left, right], total))
    }
}

jit_wrapper! {
    FanJit(FanModel) {
        inputs { xs: [Tensor; 3] }
        outputs { pairs: [Tensor; 2], total }

        build(xs) {
            model.forward(xs)
        }
    }
}

/// A recurrence carried entirely on the device: `h` is an input the plan also
/// writes, so `execute()` advances the state where the next execute reads it.
/// This is what the old host round-tripping `JitRecurrent` did, minus the
/// round trip.
struct RecurrentModel;

impl RecurrentModel {
    /// Each state slot accumulates on its own — no slot reads another's new
    /// value, so the per-buffer read-before-write ordering is unambiguous.
    fn step(&self, x: &Tensor, h: &Tensor, tail: [&Tensor; 2]) -> Result<(Tensor, Tensor, [Tensor; 2])> {
        let next = h.try_add(x).map_err(tensor_err)?;
        let head = tail[0].try_add(&next).map_err(tensor_err)?;
        let rest = tail[1].try_add(x).map_err(tensor_err)?;
        let emitted = next.try_add(&head).map_err(tensor_err)?;
        Ok((emitted, next, [head, rest]))
    }
}

jit_wrapper! {
    RecurrentJit(RecurrentModel) {
        inputs { x: Tensor }
        state { h: Tensor, tail: [Tensor; 2] }
        outputs { emitted }

        build(x, h, tail) {
            model.step(x, h, tail)
        }
    }
}

fn copy_tensor_to_buffer(tensor: &Tensor, dst: &mut svod_device::Buffer) {
    let src_buf = tensor.buffer().unwrap();
    let mut data = vec![0u8; src_buf.size()];
    src_buf.copyout(&mut data).unwrap();
    dst.copyin(&data).unwrap();
}

fn read(buffer: &svod_device::Buffer, len: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; len];
    let bytes = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), len * 4) };
    buffer.copyout_prefix(bytes).unwrap();
    out
}

#[test]
fn test_jit_single_input_prepare_and_execute() {
    let mut jit = AddJit::new(AddModel);

    let cfg = PrepareConfig::default();
    jit.prepare_with_config(InputSpec::f32(&[3]), InputSpec::f32(&[3]), &cfg).unwrap();

    copy_tensor_to_buffer(&Tensor::from_slice([1.0f32, 2.0, 3.0]), jit.x_mut().unwrap());
    copy_tensor_to_buffer(&Tensor::from_slice([10.0f32, 20.0, 30.0]), jit.y_mut().unwrap());

    jit.execute().unwrap();

    assert_eq!(read(jit.output().unwrap(), 3), vec![11.0, 22.0, 33.0]);
}

#[test]
fn test_jit_typed_input_view_writes_in_place() {
    let mut jit = AddJit::new(AddModel);
    jit.prepare(InputSpec::f32(&[3]), InputSpec::f32(&[3])).unwrap();

    for (i, slot) in jit.x_view_mut::<f32>().unwrap().iter_mut().enumerate() {
        *slot = (i + 1) as f32;
    }
    for (i, slot) in jit.y_view_mut::<f32>().unwrap().iter_mut().enumerate() {
        *slot = 10.0 * (i + 1) as f32;
    }
    jit.execute().unwrap();

    assert_eq!(read(jit.output().unwrap(), 3), vec![11.0, 22.0, 33.0]);
}

#[test]
fn test_jit_replay() {
    let mut jit = AddJit::new(AddModel);
    jit.prepare(InputSpec::f32(&[3]), InputSpec::f32(&[3])).unwrap();

    copy_tensor_to_buffer(&Tensor::from_slice([1.0f32, 2.0, 3.0]), jit.x_mut().unwrap());
    copy_tensor_to_buffer(&Tensor::from_slice([10.0f32, 20.0, 30.0]), jit.y_mut().unwrap());
    jit.execute().unwrap();

    copy_tensor_to_buffer(&Tensor::from_slice([100.0f32, 200.0, 300.0]), jit.x_mut().unwrap());
    copy_tensor_to_buffer(&Tensor::from_slice([1.0f32, 2.0, 3.0]), jit.y_mut().unwrap());
    jit.execute().unwrap();

    assert_eq!(read(jit.output().unwrap(), 3), vec![101.0, 202.0, 303.0]);
}

#[test]
fn test_jit_multiple_replays() {
    let mut jit = AddJit::new(AddModel);
    jit.prepare(InputSpec::f32(&[3]), InputSpec::f32(&[3])).unwrap();

    for i in 0..5 {
        copy_tensor_to_buffer(&Tensor::from_slice([i as f32; 3]), jit.x_mut().unwrap());
        copy_tensor_to_buffer(&Tensor::from_slice([(i + 1) as f32; 3]), jit.y_mut().unwrap());

        jit.execute().unwrap();

        assert_eq!(read(jit.output().unwrap(), 3), vec![(i + i + 1) as f32; 3]);
    }
}

#[test]
fn test_jit_profiled_execution_apis() {
    let mut jit = AddJit::new(AddModel);
    jit.prepare(InputSpec::f32(&[3]), InputSpec::f32(&[3])).unwrap();

    copy_tensor_to_buffer(&Tensor::from_slice([1.0f32, 2.0, 3.0]), jit.x_mut().unwrap());
    copy_tensor_to_buffer(&Tensor::from_slice([10.0f32, 20.0, 30.0]), jit.y_mut().unwrap());

    let profiles = jit.execute_profiled().unwrap();
    assert!(!profiles.is_empty(), "expected at least one kernel profile");

    let profiles_with_vars = jit.execute_with_vars_profiled(&[]).unwrap();
    assert!(!profiles_with_vars.is_empty(), "expected at least one kernel profile with vars");

    assert_eq!(read(jit.output().unwrap(), 3), vec![11.0, 22.0, 33.0]);
}

#[test]
fn test_jit_multi_output_same_dtype() {
    let mut jit = SplitJit::new(SplitModel);
    jit.prepare(InputSpec::f32(&[3]), InputSpec::f32(&[3])).unwrap();

    copy_tensor_to_buffer(&Tensor::from_slice([10.0f32, 20.0, 30.0]), jit.x_mut().unwrap());
    copy_tensor_to_buffer(&Tensor::from_slice([1.0f32, 2.0, 3.0]), jit.y_mut().unwrap());
    jit.execute().unwrap();

    assert_eq!(jit.sum_to_vec::<f32>().unwrap(), vec![11.0, 22.0, 33.0]);
    assert_eq!(jit.diff_to_vec::<f32>().unwrap(), vec![9.0, 18.0, 27.0]);
    assert_eq!(jit.sum_shape().unwrap(), vec![3]);
    assert_eq!(jit.sum_view::<f32>().unwrap().shape(), &[3]);
}

#[test]
fn test_jit_multi_output_mixed_dtype() {
    let mut jit = MixedJit::new(MixedModel);
    jit.prepare(InputSpec::f32(&[4]), InputSpec::f32(&[4])).unwrap();

    // argmax(x) is index 2 (value 9.0); x + y is elementwise.
    copy_tensor_to_buffer(&Tensor::from_slice([1.0f32, 5.0, 9.0, 3.0]), jit.x_mut().unwrap());
    copy_tensor_to_buffer(&Tensor::from_slice([10.0f32, 10.0, 10.0, 10.0]), jit.y_mut().unwrap());
    jit.execute().unwrap();

    assert_eq!(jit.sum_to_vec::<f32>().unwrap(), vec![11.0, 15.0, 19.0, 13.0]);
    // The int32 output reads back with its own dtype.
    assert_eq!(jit.arg_to_vec::<i32>().unwrap(), vec![2]);
    // …and refuses a foreign one.
    assert!(matches!(jit.arg_to_vec::<f32>(), Err(JitError::DtypeMismatch { .. })));
}

#[test]
fn test_jit_batch_var_shrinks_and_rebinds() {
    let mut jit = BatchJit::new(BatchModel);
    jit.prepare(InputSpec::f32(&[4, 2]), InputSpec::f32(&[2])).unwrap();

    copy_tensor_to_buffer(&Tensor::from_slice([100.0f32, 200.0]), jit.bias_mut().unwrap());
    let rows = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).try_reshape([4, 2]).unwrap();
    copy_tensor_to_buffer(&rows, jit.rows_mut().unwrap());

    jit.execute_bound(4).unwrap();
    assert_eq!(jit.scaled_shape().unwrap(), vec![4, 2]);
    assert_eq!(jit.scaled_to_vec::<f32>().unwrap(), vec![101.0, 202.0, 103.0, 204.0, 105.0, 206.0, 107.0, 208.0]);

    // Rebinding narrows the live region: shape, view and read-back follow.
    jit.execute_bound(2).unwrap();
    assert_eq!(jit.scaled_shape().unwrap(), vec![2, 2]);
    assert_eq!(jit.scaled_view::<f32>().unwrap().shape(), &[2, 2]);
    assert_eq!(jit.scaled_to_vec::<f32>().unwrap(), vec![101.0, 202.0, 103.0, 204.0]);
}

/// A binding the plan rejects leaves the recorded values alone: the output
/// shape keeps reporting the last run that executed.
#[test]
fn test_jit_rejected_var_keeps_last_output_shape() {
    let mut jit = BatchJit::new(BatchModel);
    jit.prepare(InputSpec::f32(&[4, 2]), InputSpec::f32(&[2])).unwrap();
    copy_tensor_to_buffer(&Tensor::from_slice([100.0f32, 200.0]), jit.bias_mut().unwrap());
    let rows = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).try_reshape([4, 2]).unwrap();
    copy_tensor_to_buffer(&rows, jit.rows_mut().unwrap());

    jit.execute_bound(3).unwrap();
    assert_eq!(jit.scaled_shape().unwrap(), vec![3, 2]);

    // `b` is declared on (1, 4): both directions are rejected by the plan.
    for rejected in [5, 0] {
        assert!(matches!(jit.execute_bound(rejected), Err(JitError::Runtime { .. })));
        assert_eq!(jit.scaled_shape().unwrap(), vec![3, 2]);
        assert_eq!(jit.scaled_to_vec::<f32>().unwrap(), vec![101.0, 202.0, 103.0, 204.0, 105.0, 206.0]);
        assert!(matches!(jit.execute_with_vars_profiled(&[("b", rejected)]), Err(JitError::Runtime { .. })));
        assert_eq!(jit.scaled_shape().unwrap(), vec![3, 2]);
    }

    // An accepted rebinding still moves it.
    jit.execute_bound(2).unwrap();
    assert_eq!(jit.scaled_shape().unwrap(), vec![2, 2]);
}

fn ramp(n: usize, scale: f32) -> Tensor {
    Tensor::from_slice((0..n).map(|i| ((i % 7) as f32 - 3.0) * scale).collect::<Vec<f32>>())
}

/// One plan for every batch size: the results must match a plan compiled for
/// that batch exactly, and the batch must stay its own kernel axis (it shows up
/// as a kernel variable, not folded into a merged `b*T` extent).
#[test]
fn test_jit_symbolic_batch_matches_const_batch() {
    let w = ramp(LN_K * LN_N, 0.05).try_reshape([LN_K, LN_N]).unwrap();
    let x_full = ramp(LN_MAX_B * LN_T * LN_K, 0.1).try_reshape([LN_MAX_B, LN_T, LN_K]).unwrap();

    let mut sym = SymBatchJit::new(LinearNormModel);
    sym.prepare(InputSpec::f32(&[LN_MAX_B, LN_T, LN_K]), InputSpec::f32(&[LN_K, LN_N])).unwrap();
    copy_tensor_to_buffer(&x_full, sym.x_mut().unwrap());
    copy_tensor_to_buffer(&w, sym.w_mut().unwrap());

    // The batch reaches the kernels as a runtime variable rather than being
    // baked into (or merged away with) a constant axis.
    let kernels = sym.prepared_kernels().unwrap();
    assert!(
        kernels.iter().any(|k| k.kernel.var_names.iter().any(|n| n == "b")),
        "no kernel takes `b` as a variable: {:?}",
        kernels.iter().map(|k| &k.kernel.entry_point).collect::<Vec<_>>()
    );

    for b in 1..=LN_MAX_B {
        sym.execute_bound(b as i64).unwrap();
        assert_eq!(sym.y_shape().unwrap(), vec![b, LN_T, LN_N]);
        let got = sym.y_to_vec::<f32>().unwrap();

        let mut fixed = ConstBatchJit::new(LinearNormModel);
        fixed.prepare(InputSpec::f32(&[b, LN_T, LN_K]), InputSpec::f32(&[LN_K, LN_N])).unwrap();
        let rows = x_full.try_shrink([Some((0isize, b as isize)), None, None]).unwrap();
        rows.realize().unwrap();
        copy_tensor_to_buffer(&rows.contiguous(), fixed.x_mut().unwrap());
        copy_tensor_to_buffer(&w, fixed.w_mut().unwrap());
        fixed.execute().unwrap();

        if b == 1 {
            // The kernel name spells out the opt schedule (one token per axis
            // extent). At b=1 the symbolic plan must reach the *same* schedule
            // as the constant one, with the batch added as a symbolic axis
            // (`3F`) — not a merged `b*T` extent that drops the const-axis opts.
            let names = |ks: Vec<&svod_runtime::PreparedKernel>| {
                ks.iter().map(|k| k.kernel.entry_point.clone()).collect::<Vec<_>>()
            };
            let want: Vec<String> =
                names(fixed.prepared_kernels().unwrap()).iter().map(|n| n.replacen('_', "_3F_", 1)).collect();
            assert_eq!(names(sym.prepared_kernels().unwrap()), want);
        }
        let want = fixed.y_to_vec::<f32>().unwrap();
        assert_eq!(got.len(), want.len(), "b={b}");
        for (i, (g, e)) in got.iter().zip(&want).enumerate() {
            assert!((g - e).abs() < 1e-5, "b={b} elem {i}: {g} != {e}");
        }
    }
}

#[test]
fn test_jit_array_slots() {
    let mut jit = FanJit::new(FanModel);
    jit.prepare([InputSpec::f32(&[2]), InputSpec::f32(&[2]), InputSpec::f32(&[2])]).unwrap();

    copy_tensor_to_buffer(&Tensor::from_slice([1.0f32, 1.0]), jit.xs_mut(0).unwrap());
    copy_tensor_to_buffer(&Tensor::from_slice([2.0f32, 2.0]), jit.xs_mut(1).unwrap());
    copy_tensor_to_buffer(&Tensor::from_slice([4.0f32, 4.0]), jit.xs_mut(2).unwrap());
    jit.execute().unwrap();

    assert_eq!(jit.pairs_to_vec::<f32>(0).unwrap(), vec![3.0, 3.0]);
    assert_eq!(jit.pairs_to_vec::<f32>(1).unwrap(), vec![6.0, 6.0]);
    assert_eq!(jit.total_to_vec::<f32>().unwrap(), vec![9.0, 9.0]);
    assert!(matches!(jit.xs_mut(3), Err(JitError::InputBufferNotFound { .. })));
}

/// The `state { .. }` replacement for the old host-side recurrent wrapper:
/// state advances on-device across executes and `reset()` clears every slot.
#[test]
fn test_jit_state_recurrence_and_reset() {
    let mut jit = RecurrentJit::new(RecurrentModel);
    jit.prepare(InputSpec::f32(&[2]), InputSpec::f32(&[2]), [InputSpec::f32(&[2]), InputSpec::f32(&[2])]).unwrap();

    copy_tensor_to_buffer(&Tensor::from_slice([1.0f32, 2.0]), jit.x_mut().unwrap());

    // h accumulates x, tail[0] accumulates h, emitted = h + tail[0].
    jit.execute().unwrap();
    assert_eq!(jit.emitted_to_vec::<f32>().unwrap(), vec![2.0, 4.0]);
    jit.execute().unwrap();
    assert_eq!(jit.emitted_to_vec::<f32>().unwrap(), vec![5.0, 10.0]);
    jit.execute().unwrap();
    assert_eq!(jit.emitted_to_vec::<f32>().unwrap(), vec![9.0, 18.0]);

    jit.reset().unwrap();
    jit.execute().unwrap();
    assert_eq!(jit.emitted_to_vec::<f32>().unwrap(), vec![2.0, 4.0]);

    // Only the declared output is exposed; the state outputs stay internal.
    assert_eq!(jit.output_buffers().unwrap().len(), 4);
    assert_eq!(jit.emitted_shape().unwrap(), vec![2]);
}

#[test]
fn test_jit_replicate_executes_independently() {
    let mut jit = AddJit::new(AddModel);
    jit.prepare(InputSpec::f32(&[3]), InputSpec::f32(&[3])).unwrap();
    copy_tensor_to_buffer(&Tensor::from_slice([1.0f32, 2.0, 3.0]), jit.x_mut().unwrap());
    copy_tensor_to_buffer(&Tensor::from_slice([10.0f32, 20.0, 30.0]), jit.y_mut().unwrap());
    jit.execute().unwrap();

    // The replica snapshots the original's inputs, then diverges.
    let mut replica = jit.replicate().unwrap();
    copy_tensor_to_buffer(&Tensor::from_slice([100.0f32, 200.0, 300.0]), replica.x_mut().unwrap());
    copy_tensor_to_buffer(&Tensor::from_slice([1.0f32, 1.0, 1.0]), replica.y_mut().unwrap());
    replica.execute().unwrap();
    assert_eq!(read(replica.output().unwrap(), 3), vec![101.0, 201.0, 301.0]);

    // The original's inputs and output are untouched by the replica.
    jit.execute().unwrap();
    assert_eq!(read(jit.output().unwrap(), 3), vec![11.0, 22.0, 33.0]);

    // Replicas mint further replicas, snapshotting the replica's state.
    let mut second = replica.replicate().unwrap();
    second.execute().unwrap();
    assert_eq!(read(second.output().unwrap(), 3), vec![101.0, 201.0, 301.0]);
}

/// A replicated recurrence forks its state: the replica advances from the
/// snapshot without touching the original's.
#[test]
fn test_jit_replicate_forks_state() {
    let mut jit = RecurrentJit::new(RecurrentModel);
    jit.prepare(InputSpec::f32(&[2]), InputSpec::f32(&[2]), [InputSpec::f32(&[2]), InputSpec::f32(&[2])]).unwrap();
    copy_tensor_to_buffer(&Tensor::from_slice([1.0f32, 2.0]), jit.x_mut().unwrap());
    jit.execute().unwrap();

    let mut replica = jit.replicate().unwrap();
    replica.execute().unwrap();
    assert_eq!(replica.emitted_to_vec::<f32>().unwrap(), vec![5.0, 10.0]);

    // The original resumes from its own state, unaffected by the replica.
    jit.execute().unwrap();
    assert_eq!(jit.emitted_to_vec::<f32>().unwrap(), vec![5.0, 10.0]);
}

/// Phase-2 acceptance: multiple models preparing AND executing concurrently
/// from several threads must not cross-talk (per-tensor input identities,
/// winner-computes caches, per-plan queues).
#[test]
fn test_concurrent_multi_model_prepare_and_execute() {
    let barrier = std::sync::Barrier::new(4);
    std::thread::scope(|scope| {
        for worker in 0..4u32 {
            let barrier = &barrier;
            scope.spawn(move || {
                for round in 0..8u32 {
                    barrier.wait();
                    let x = [(worker * 100 + round) as f32; 3];
                    let y = [(worker + round * 10) as f32 + 0.5; 3];
                    if worker % 2 == 0 {
                        let mut jit = AddJit::new(AddModel);
                        jit.prepare(InputSpec::f32(&[3]), InputSpec::f32(&[3])).unwrap();
                        copy_tensor_to_buffer(&Tensor::from_slice(x), jit.x_mut().unwrap());
                        copy_tensor_to_buffer(&Tensor::from_slice(y), jit.y_mut().unwrap());
                        jit.execute().unwrap();
                        assert_eq!(read(jit.output().unwrap(), 3), vec![x[0] + y[0]; 3]);
                    } else {
                        let mut jit = SplitJit::new(SplitModel);
                        jit.prepare(InputSpec::f32(&[3]), InputSpec::f32(&[3])).unwrap();
                        copy_tensor_to_buffer(&Tensor::from_slice(x), jit.x_mut().unwrap());
                        copy_tensor_to_buffer(&Tensor::from_slice(y), jit.y_mut().unwrap());
                        jit.execute().unwrap();
                        assert_eq!(jit.sum_to_vec::<f32>().unwrap(), vec![x[0] + y[0]; 3]);
                        assert_eq!(jit.diff_to_vec::<f32>().unwrap(), vec![x[0] - y[0]; 3]);
                    }
                }
            });
        }
    });
}
