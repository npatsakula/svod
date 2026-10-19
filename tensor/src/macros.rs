//! Declarative helpers for **custom (hand-built) kernels** — the generic
//! definition/check surface any kernel author (e.g. the `svod-tk3` tile kernels) reuses.
//!
//! The *definition* half is the runtime helper [`Tensor::graph_kernel`](crate::Tensor::graph_kernel)
//! (wrap a SINK builder as an `Op::Call` graph node). This module adds the *check*
//! half: [`custom_kernel_check!`], which generates a hardware-gated numerical
//! correctness test against a reference op, removing the per-kernel boilerplate of
//! building inputs, running both, casting to f32, and comparing within tolerance.
//! The comparison itself is [`crate::testing::allclose_f32`] — shared with any
//! property-based harness so both inherit its finite/NaN and empty-input guards.

/// Generate a hardware-gated (`#[ignore]`) numerical correctness test for a custom
/// kernel against a reference op.
///
/// Builds one random input tensor per `inputs` name (all of `shape`/`dtype`),
/// runs `run` (the kernel under test) and `reference` (the oracle) — both
/// `FnOnce(&Tensor, …) -> Result<Tensor, _>` for *any* error type — casts both
/// outputs to f32 and asserts [`allclose`](crate::testing::allclose_f32) at
/// `atol = rtol = tol` (so any length/empty or `NaN`/`inf` mismatch also fails).
/// The test is `#[ignore]` (run with `SVOD_DEVICE=<dev> … -- --ignored`), so the
/// inputs land on the env-selected device.
///
/// ```ignore
/// use svod_tensor::{Result, Tensor};
/// use svod_tk3::ops::{Attn, attention};
///
/// // tk3 takes `[B, T, H, D]`; the graph op takes `[B, H, T, D]`.
/// fn causal_sdpa(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
///     let heads_first = |t: &Tensor| t.try_transpose(1, 2);
///     let (q, k, v) = (heads_first(q)?, heads_first(k)?, heads_first(v)?);
///     q.scaled_dot_product_attention().key(&k).value(&v).is_causal(true).call()?.try_transpose(1, 2)
/// }
///
/// svod_tensor::custom_kernel_check! {
///     causal_attention_check,
///     inputs (q, k, v): shape [1, 128, 2, 64], dtype svod_dtype::DType::BFloat16,
///     run: |q, k, v| attention(q, k, v, Attn { causal: true, ..Attn::default() }),
///     reference: causal_sdpa,
///     tol: 2e-2,
/// }
/// ```
#[macro_export]
macro_rules! custom_kernel_check {
    (
        $(#[$meta:meta])*
        $name:ident,
        inputs ( $($arg:ident),+ $(,)? ): shape $shape:expr, dtype $dt:expr,
        run: $run:expr,
        reference: $reference:expr,
        tol: $tol:expr $(,)?
    ) => {
        $(#[$meta])*
        #[test]
        #[ignore]
        fn $name() {
            let shape: &[usize] = &$shape;
            let mk = || {
                let t = $crate::Tensor::randn(shape).expect("custom_kernel_check: randn input");
                let t = t.cast($dt);
                t.realize().expect("custom_kernel_check: realize input");
                t
            };
            $( let $arg = mk(); )+

            let to_f32_vec = |t: $crate::Tensor| -> Vec<f32> {
                t.cast(::svod_dtype::DType::Float32)
                    .to_vec::<f32>()
                    .expect("custom_kernel_check: read f32")
            };

            let run = $run;
            let got = to_f32_vec(run($(&$arg),+).expect("custom_kernel_check: kernel run"));

            let reference = $reference;
            let expected = to_f32_vec(reference($(&$arg),+).expect("custom_kernel_check: reference run"));

            // Combined absolute+relative tolerance via the shared comparator, which
            // also catches finite/non-finite (NaN/inf) and length/empty mismatches.
            let (atol, rtol) = ($tol as f32, $tol as f32);
            let report = $crate::testing::allclose_f32(&got, &expected, atol, rtol);
            println!("custom_kernel_check[{}]: {}", stringify!($name), report.message);
            assert!(report.ok, "custom_kernel_check[{}]: {}", stringify!($name), report.message);
        }
    };
}

/// Open the call frame of a public op, naming `$op` at the caller's source line.
///
/// The enclosing function must be `#[track_caller]` (and so must every wrapper
/// between it and user code). Nested public ops collapse into the outermost
/// frame, so one public call yields exactly one frame; with capture off the
/// guard is a thread-local read and write.
macro_rules! origin_call {
    ($op:literal) => {
        let _origin = ::svod_ir::origin::OriginScope::outer_call($op, ::std::panic::Location::caller());
    };
}
