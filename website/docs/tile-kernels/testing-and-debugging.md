---
sidebar_label: Testing and Debugging
---

# Testing and Debugging

Each tk kernel is checked at two levels. On the host, its tile program runs in the interpreter
against a direct reference, and the lowered program renders and, where the toolchain is
installed, compiles for every target. On a GPU with tk kernels (CUDA or AMD), the lowered kernel
runs against the interpreter or against the op's own graph fallback. Tests compare values within
tolerances, not hashes of the IR.

## The interpreter

`interp::run(&program, params, vars)` executes a tile program on the host. It takes one
`Vec<f64>` per parameter and returns every parameter after the run. Each block runs its
statements in order, and a pipeline runs as its serial interleaving. Every value is rounded to
its element type (`interp::round_to`), so the result is what a correct lowering must reproduce
up to accumulation order. Scalar division truncates toward zero, as the emitted code does.
Runtime variables are bound by name:

```rust
let out = run(&prog, vec![a.clone(), b.clone(), vec![0.0; m * n]], &[("b", 1)]).unwrap();
```

It returns `interp::Error::{UnboundVar, ParamSize, Division, OutOfBounds}`: an access past a
buffer's capacity is an error, not a silent read. It does not model
threads or barriers, so ordering bugs show up only on the device.

## Test files

| File (`test/unit/`) | Level | What it checks |
|---|---|---|
| `atoms.rs`, `layout.rs`, `layouts.rs` | host | Every atom's operand layouts against the vendor ISA lane formulas, F2 algebra laws, inference, relayouts and swizzles |
| `build.rs`, `interp.rs`, `schedule.rs`, `sync.rs`, `emit.rs` | host | Builder records, interpreter semantics, pipeline expansion, the barrier pass, the emitted list |
| `ops_plan.rs` | host | Every op's `Plan` and error per shape, dtype and target |
| `offline.rs` | host | Card targets and costs read from a copied tune store |
| `targets.rs` | host + device | Every kernel family per target lowered, interpreted before and after lowering and compiled (`ptxas`, clang); on the GPU, the device gate |
| `probe.rs` | host + device | Every cost probe builds for every target; on the GPU, `device_costs_are_physical` |
| `attention.rs`, `heads.rs`, `rows.rs`, `conv.rs`, `gather.rs` | host + device | Program against an f64 reference, then the lowered kernel against the program |
| `device.rs`, `parts.rs` | device | GEMM end to end, and the parts attention composes (on `cp.async` devices) |
| `ops.rs` | device | Every op against its graph fallback on awkward shapes |
| `tune.rs` | host + device | Store round trip, key stability, a real GEMM tuning |

## Running the suite

`SVOD_DEVICE` selects the device. Device tests skip with a `skipped: …` line only where there is
nothing to run on: `SVOD_DEVICE=CPU`, a GPU tk has no kernels for, or no `SVOD_DEVICE` and no
supported GPU attached. With a supported GPU attached and `SVOD_DEVICE` unset, they fail and
name the value to set, since a suite whose device tests all skip passes and proves nothing.

```bash
# Host only, no GPU:
SVOD_DEVICE=CPU cargo test -p svod-tk --release --lib
# On the GPU; on a fresh tune store run the cost probes alone first, since the parallel
# suite inflates what they measure and the store keeps the first measurement:
SVOD_DEVICE=CUDA cargo test -p svod-tk --release --lib device_costs_are_physical
SVOD_DEVICE=CUDA cargo test -p svod-tk --release --lib
```

`SVOD_DEVICE=AMD` (or `AMD:N`, in KFD node order) selects an AMD GPU. A target counts as
measured only once `families_match_the_interpreter_on_the_device` passes on it: every family's
first candidate on the device against the interpreter. `register_staged_families_match_on_cuda`
runs the AMD data path (register-staged fills, no `ldmatrix`) on a CUDA GPU.

### On a new GPU

1. Point the tune store at a directory to keep: `SVOD_TK_TUNE_DIR=$PWD/tk_tune_$(hostname)`.
2. On a machine with several AMD nodes, check that the node `SVOD_DEVICE=AMD:N` names is the
   GPU (`/sys/class/kfd/kfd/topology/nodes/*/properties`, `gfx_target_version`).
3. Run `device_costs_are_physical` alone, then `families_match_the_interpreter_on_the_device`,
   then the suite. Do not report numbers for a device whose gate fails.

The model-level gate for Nemotron (real weights, generated goldens):

```bash
SVOD_DEVICE=CUDA cargo test -p svod-model --release --lib nemotron_diar::parity::half_precision -- --ignored --nocapture --test-threads=1
```

## Did the kernel run?

An op that falls back builds graph ops instead of a `CALL`; the `ops.rs` tests look for the
kernel's name among the calls of the result's graph. To predict the path without a device, call
the `ops::shape` planner (see [Op Layer](./op-layer#kernel-or-graph)); on a device, the planner
log says why an op took the graph ([Tuning](./tuning#the-planner-log)).

## Planning for another card

The `tk_dump` example shows what tk would run on a card this machine has no device for: the
planner's ranking with predicted ns and why each config of the lattice was left out
(`plan.txt`), the chosen config's LLVM text and, where the tools are installed, its PTX,
`ptxas -v` resources and SASS (CUDA) or AMDGCN assembly with its resource comments (AMD), and an
opcode histogram of the main loop.

```bash
cargo run -p svod-tk --release --example tk_dump -- --card h100-pcie --op gemm --shape 4096x4096x4096
cargo run -p svod-tk --release --example tk_dump -- --arch gfx942 --op attention --shape 8x16x1024x128 --causal --store DIR
```

`--card` takes a name of `atoms::CARDS` (`rtx3060`, `h100-pcie`, `h100-sxm`, `rtx5070`,
`rtx5090`, `mi300x`, `mi355x`, `rx9070xt`, `strix-halo`); `--arch` takes an architecture with
`--sms`/`--xcds` or the first listed card's. `--store` is a tune store directory copied from the
card, whose probe lines price the ranking; without it the guessed costs do. `--cfg N` emits the
N-th ranked config, `--out DIR` names the output directory (`tk_dump`).

`--op` takes `gemm` (`--shape MxNxK`), `attention` (`BxHxTxD`), `norm` (`RxD`), `conv`
(`BxHxWxCINxCOUT`) or `heads` (`BxTxHxKVxD`), each through the op layer's own shape gate and
tuning. `--cfg tuned` emits the config the tuner chose on the card, read from the `--store`
copy without writing to it: a stored line matches by op, device, dtype, shape and the config list
the op layer keys it by, whatever compiler or kernel build wrote it, and the dump prints each key
it read or says which line is missing. `--compare FILE [--fun NAME]` prints the main loop of a
reference kernel (a `.cubin`, an AMD code object, or SASS, PTX or AMDGCN text) beside tk's,
counted the same way.

```bash
cargo run -p svod-tk --release --example tk_dump -- --card rtx3060 --op gemm --shape 4096x4096x4096 --cfg tuned --store ~/.cache/svod/tk_tune
cargo run -p svod-tk --release --example tk_dump -- --card mi300x --op gemm --shape 4096x4096x4096 --compare hk.co --fun NAME
```

## Environment variables

| Variable | Effect |
|---|---|
| `SVOD_DEVICE` | The default device: `CUDA[:N]`, `AMD[:N]`, `CPU`. It decides whether kernels apply and which device tests run |
| `SVOD_TK_TUNE=0` | No candidate timing; the first clean config runs (see [Tuning](./tuning)) |
| `SVOD_TK_TUNE_DIR` | The tune store's directory |
| `RUST_LOG=svod_tk::plan=debug` | The planner's decisions; `svod_tk=debug` adds the tuner's rounds |
| `TK_DUMP_LIST=1` | Print every emitted instruction (`[i] id op dtype <- sources`) when a program is lowered. Bodies are memoized, so it prints the first lowering of each program only |
| `SVOD_SPEC_DEBUG=1` | When a program fails IR spec verification, print the rejected instruction and its tree |

## Profiling a model

The Nemotron example warms up once (which also fills the tune store), then times and profiles a
second run, printing its per-kernel report:

```bash
cargo run -p svod-model --release --example nemotron_diarize -- audio_1.wav --dtype bf16 --profile
```

tk kernels appear under their kernel names (`gemm`, `conv`, `flash_attention`, `heads`,
`layer_norm`, …) next to graph kernels, timed the same way.

:::tip[Measure on a quiet GPU]
Another process on the GPU skews every number. The probes and the tune store warm the clock
first. Run probes with `--test-threads=1`.
:::
