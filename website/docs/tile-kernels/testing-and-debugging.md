---
sidebar_label: Testing and Debugging
---

# Testing and Debugging

Each tk3 kernel is checked at two levels. On the host, its tile program runs in the interpreter
against a direct reference. On a CUDA device, the lowered kernel runs against the interpreter or
against the op's own graph fallback. Tests compare values within tolerances, not hashes of the
IR.

## The interpreter

`interp::run(&program, params, vars)` executes a tile program on the host. It takes one
`Vec<f64>` per parameter and returns every parameter after the run. Each block runs its
statements in order, and a pipeline runs as its serial interleaving. Every value is rounded to
its element type (`interp::round_to`), so the result is what a correct lowering must reproduce
up to accumulation order. Runtime variables are bound by name:

```rust
let out = run(&prog, vec![a.clone(), b.clone(), vec![0.0; m * n]], &[("b", 1)]).unwrap();
```

It returns `interp::Error::{UnboundVar, ParamSize, Raw}`. It does not model threads or
barriers, so ordering bugs show up only on the device.

## Test files

| File | Level | What it checks |
|---|---|---|
| `layout.rs`, `layouts.rs`, `atoms.rs` | host | F2 algebra laws, every atom against its closed form, inference results |
| `interp.rs`, `schedule.rs`, `build.rs` | host | Interpreter semantics, pipeline expansion |
| `ops_plan.rs` | host | Every op's `Plan` and error per shape, dtype and target |
| `attention.rs`, `heads.rs`, `rows.rs` | host + device | Program against an f64 reference, then the lowered kernel against the program |
| `device.rs`, `parts.rs` | device | GEMM (every config the op layer can pick) and the parts attention composes, against the interpreter |
| `ops.rs` | device | Every op against its graph fallback on awkward shapes, with tuning off |
| `tune.rs` | host + device | Store round trip, key stability, a real GEMM tuning |

Device tests skip with a `skipped: no CUDA device` line unless `SVOD_DEVICE` names a CUDA
device. Run the whole suite on the GPU serialized:

```bash
SVOD_DEVICE=CUDA:0 cargo test -p svod-tk3 --lib --release -- --test-threads=1
```

The model-level gate for Nemotron (real weights, generated goldens):

```bash
SVOD_DEVICE=CUDA:0 cargo test -p svod-model --release --lib nemotron_diar::parity::half_precision -- --ignored --nocapture --test-threads=1
```

## Did the kernel run?

An op that falls back builds graph ops instead of a `CALL`. The `ops.rs` tests look for the
kernel name among the calls of the result:

```rust
fn assert_kernel(t: &Tensor, name: &str) {
    let calls: Vec<String> = t
        .uop()
        .toposort()
        .iter()
        .filter_map(|u| match u.op() {
            Op::Call(ops::Call { info, .. }) => info.name.clone(),
            _ => None,
        })
        .collect();
    assert!(calls.iter().any(|n| n == name), "no {name} kernel among {calls:?}");
}
```

To predict the path without a device, call the `ops::shape` planner (see [Op Layer](./op-layer#kernel-or-graph)).

## Environment variables

| Variable | Effect |
|---|---|
| `TK3_DUMP_LIST=1` | Print every emitted instruction (`[i] id op dtype <- sources`) when a program is lowered. Bodies are memoized, so it prints the first lowering of each program only |
| `SVOD_SPEC_DEBUG=1` | When a program fails IR spec verification, print the rejected instruction and its tree |
| `SVOD_TK3_TUNE=0` | No measuring; the first candidate runs (see [Tuning](./tuning)) |
| `SVOD_DEVICE=CUDA:0` | Run on the GPU (the default device decides whether kernels apply) |

## Profiling a model

The Nemotron example warms up once (which also fills the tune store), times a run, then prints
the per-kernel report of a third run:

```bash
cargo run -p svod-model --release --example nemotron_diarize -- audio_1.wav --dtype bf16 --profile
```

tk3 kernels appear under their kernel names (`gemm`, `flash_attention`, `heads`, `layer_norm`,
…) next to graph kernels, timed the same way.

:::tip[Measure on a quiet GPU]
An RTX 3060 idles at 210 MHz, and another process on the GPU skews every number. The probes
and the tune store warm the clock first. Run probes with `--test-threads=1`.
:::
