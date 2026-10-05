---
sidebar_label: Debugging
---

# Debugging and Verifying Kernels

A hand-written kernel is only as trustworthy as your ability to check it. The
[Flash Attention](./flash-attention) walkthrough showed the kind of kernel worth writing by
hand; this chapter is how you come to trust it. The USE face hands you a lazy `Tensor` that
fuses into a big graph — convenient, but a bad place to ask "is this one kernel correct, and how
fast is it?" `tk`'s **DEBUG face** exists for exactly that: run a single kernel against concrete
buffers, read the result back, time it, and prove that a refactor didn't change its behavior.

---

## Direct dispatch: run one kernel, see the bytes

The direct-launch API (`tk/src/launch.rs`) bypasses the tensor scheduler entirely. You give it
a kernel body and real input tensors; it realizes the inputs, allocates the outputs, renders,
compiles, and dispatches, writing the result into an output you can read back:

```rust
// The DEBUG face from tk/src/lib.rs. `outs` are written in place.
run_kernel("tile_add", [1, 1, 1], block, &mut [&mut out], &[&input_a, &input_b], build)?;
let values = out.as_vec::<f32>()?;   // read the GPU result straight back
assert_eq!(values, expected);
```

Because this skips scheduling, fusion, and dependency tracking, what you measure is *just your
kernel* — not a graph that happens to contain it. That isolation is the point: when a number is
wrong, you want to know it's wrong *here*, not somewhere in a fused pipeline.

A note on the path: skipping the *scheduler* is not skipping the *optimizer*. `compile` still runs
the production `optimize_kernel_with_config` over your `SINK` — which applies zero schedule opts to a
hand-lowered body (that's what the `opts_to_apply: Some(vec![])` marker buys) but still performs the
shared rewrites every kernel needs before rendering, index-dtype lowering among them. You get
correct code without the scheduler. The `ArchCaps` come from the buffers' device; a GPU whose
arch does not resolve is an error, and only a host device falls back to `ArchCaps::GFX942` so the
`SINK` still builds.

---

## Timing on real hardware

For performance work, `CompiledLaunch` (from `compile_kernel`) exposes hardware timestamps
rather than wall-clock guesses:

```rust
// Render + compile once …
let launch = compile_kernel("matmul", grid, block, &mut [&mut c], &[&a, &b], build)?;
// … then dispatch in a loop, outside the timed region.
// SAFETY: the bound buffers stay allocated for `launch`'s lifetime.
unsafe { launch.dispatch(true) }?;
let ns = launch.dispatch_gpu_ns()?;   // Option<u64>: device-measured dispatch time
```

`dispatch_gpu_ns()` dispatches once through a profiling context and reads the device's own
timestamp counters around it, so you're measuring time on the device, not the round-trip latency
of launching it — `None` on a backend that stamps nothing. This is the primitive the
[autotuner](./tuning) ranks candidates with, after lifting the clock with the same
`warm_clock` the benches use. The criterion benches reach the same stamps one layer up, through
`plan.profile`; see [Profiling & Benchmarking](./profiling).

---

## Tests without a GPU, tests with one

Building a `SINK` is pure UOp construction and needs no device; only executing it does. The
test module (`tk/src/test/unit/`) uses that split everywhere, and a new kernel should too:

- **Graph-shape tests** run on every `cargo test`. Build the kernel against placeholder
  buffers (`UOp::new_buffer(DeviceSpec::Cpu, size, dtype)`, `ArchCaps::GFX942`), toposort the
  `SINK`, and assert what is and is not in it — `guide.rs` checks the tile-add kernel mints an
  `Op::Special`, contains one `Binary(Add)`, and has no `Wmma` and no `Local` buffer.
- **Hardware tests** are `#[ignore]` and self-skip on an unsupported device:

```bash
SVOD_DEVICE=AMD:0  cargo test -p svod-tk --lib guide::test_tile_add_amd -- --ignored
SVOD_DEVICE=CUDA:0 cargo test -p svod-tk --lib fa::test_fa_graph_check -- --ignored --nocapture
```

The gates are in `tk/src/test/unit/mod.rs`: `device_supported(archs)` for a kernel's
`ArchSet`, `fragment_device()` for anything that needs matrix-core layouts, `is_cdna_device()`
and `wave32_fragment_device()` for layout-specific checks. Each of them also calls
`svod_tk::tune::set_enabled(false)`, so a numerics test does not tune every shape it touches.

For a graph-native kernel, `svod_tensor::custom_kernel_check!` generates the whole comparison:
random inputs of one shape and dtype, the kernel under test, a reference closure, both cast to
f32 and compared at `atol = rtol = tol`.

```rust
svod_tensor::custom_kernel_check! {
    test_fa_graph_check,
    inputs (q, k, v): shape [1, 128, 2, 64], dtype svod_dtype::DType::BFloat16,
    run: |q, k, v| {
        let out = crate::kernels::fa::flash_attention(q, k, v).expect("FA build");
        Ok::<_, crate::LaunchError>(out.expect("the FA kernel applies to [1, 128, 2, 64] bf16 on every supported arch"))
    },
    reference: fa_causal_reference,
    tol: 2e-2,
}
```

A decline (`Ok(None)`) fails loudly here rather than comparing the reference with itself.

---

## Fingerprints: proving a refactor is behavior-preserving

The subtle risk with hand-written kernels: you "clean up" the builder code, the kernel still
compiles and still produces plausible numbers, but the *generated IR* changed in a way that
only shows up on some shape or some architecture later.

`KernelFingerprint` (`tk/src/fingerprint.rs`) guards against this. The LLVM render is not
deterministic run to run (node ids leak into SSA names), but the *graph* is: every UOp carries a
recursive structural `content_hash`, and the fingerprint is that hash of the `SINK` with an
order-independent fold of the node tags beside it — a `u128` `digest`, plus `op_counts` and
`node_count` for a readable diff.

```rust
let fp = kernel_fingerprint(&sink);
assert_eq!(fp.digest, GOLDEN_MATMUL_DIGEST);  // structure unchanged ⇒ behavior unchanged
```

If the fingerprint moves, you changed the emitted IR — intentionally or not — and the golden
test makes you look. `tk/src/test/unit/golden.rs` locks the matmul and the flash-attention
builders (causal, non-causal, masked) this way; a failure prints the new digest to paste, and an
intentional re-baseline is proved by dumping and diffing both graphs. The same digests key the
[autotuner's](./tuning) on-disk store, so a kernel change re-measures its tiles.

---

## The mistakes that don't error

A tile kernel is a dependency graph, and a missing edge is a wrong answer, not a compile error.
The ones the test suite has caught:

| Symptom | Cause | Fix |
|---|---|---|
| An accumulator carries stale state across loop trips | a per-trip re-init (`g.zero(acc)`) with no dependency on the loop counter is hoisted above the loop | `g.zero(lp.reinit(acc))` |
| A loop-carried tile reads the pre-loop value after the loop | the final read is not ordered after the loop's `END` | `acc.after(&lp.close())` or `lp.close_carry(acc)` |
| `finish` debug-asserts, or the linearizer mis-scopes a loop | two stores `END` the same `RANGE` | one closing store per loop; chain the others into it |
| Wrong buffers, right numbers of them | `gl` / `bind_abi` order differs from the launch's `[outs..., ins...]` | declare outputs first, inputs in launch order, optional buffers trailing |
| A kernel silently reads the wrong K/V stream | `k`/`v` bound to `q`'s shape with a different dtype of the same width (`Kernel::gl` checks only the byte width) | validate dtypes in `validate`, as `flash_attention_with` does |
| Correct on CDNA, garbage on RDNA | a lane count or fragment constant hardcoded | read `caps.wave_size` and `ker.frag(role)` ([Layouts and Wave Sizes](./wave-portability)) |

The ones that *do* error are the builder's asserts: a tile dimension that is not a multiple of
its fragment, a `k_step` that is not a multiple of the matrix core's K edge, a block that is not
whole waves, a multi-wave group calling a single-wave op, `ker.frag` on an arch with no
layouts. Each names the offending value.

---

## Which tool for which question

| You're asking… | Use |
|----------------|-----|
| "Does this builder still emit what I think?" | a graph-shape test over the `SINK`'s toposort |
| "Does this kernel produce the right numbers?" | `run_kernel` + `as_vec`, or `custom_kernel_check!` against a reference |
| "How fast is it on this GPU?" | `compile_kernel` + `dispatch_gpu_ns` |
| "Did my refactor change the emitted IR?" | `KernelFingerprint` golden test |
| "Which tile did the tuner pick, and why?" | the store file under `SVOD_TK_TUNE_DIR` ([Autotuning](./tuning)) |
| "Is the *device/driver layer* misbehaving?" | [AMD Backend → Debugging](../backends/amd/debugging), [CUDA Backend → Debugging](../backends/cuda/debugging) |

That last row matters: this chapter is about debugging *kernels* — the IR you authored and the
numbers it produces. When the problem is below that — queue dispatch, memory faults, the driver, the
PTX JIT — the per-backend chapters are the right place:
[AMD](../backends/amd/debugging) and [CUDA](../backends/cuda/debugging).

---

## Why this matters

Hand-authoring trades the optimizer's safety net for control. The DEBUG face is how you make
that trade safely: isolation to localize correctness bugs, hardware timestamps to make
performance claims you can defend, and structural fingerprints so that "I just tidied the code"
can't silently become "I changed the kernel." With those three, a hand-written kernel is as
verifiable as an autotuned one.
