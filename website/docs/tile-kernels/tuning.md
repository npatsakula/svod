---
sidebar_label: Tuning
---

# Tuning

Every op has a ranked config list per shape in `ops::config` (see the
[Kernel Library](./kernel-library) for the tables), ranked from the device's measured costs
(see [Device costs](#device-costs)). The ranking is a model, and so is a register count, so
neither admits a config. One rule does: the tuner compiles the ranked configs in batches of
eight, in ranked order and in parallel, until it has the first eight tiles whose compiled kernels
do not spill (a kernel reporting scratch bytes spilled; a backend that reports nothing admits).
The ranking lists every pipeline (ring depth, unrolling) of a tile, and a tile is admitted at its
first clean one, so an unrolled ring that spills yields to its rolled twin.
Spilling excludes a config only while clean ones remain: when every config that compiles
spills, the first eight of those are timed anyway, since the time judges and a spilling kernel
still beats the graph. A list of one config takes the same path, and no op falls to the graph
because of spills. The admitted configs, with their variants (key splits for attention,
reduction splits for a starved convolution), are the candidates the tune store measures the
first time a device meets a shape, and it keeps the winner. Attention tunes its softmax forms
and its tiles in turn: the forms on the ranking's first clean config, the tiles under those
forms, then the forms again on the winner (see the [Kernel Library](./kernel-library)). The admitted list is
remembered for the process only; another process recompiles it, which the compiler's object
cache makes cheap. When nothing compiles (no device to compile on) the first eight are used
untested.

With tuning off (`SVOD_TK3_TUNE=0`), admission still compiles and the first admitted config
runs; only candidate timing is off. The device's cost probes still measure, once per device
(about 4 s on gfx1151), since the ranking needs them.

A better fixed default was not enough. Nemotron's GEMMs (M = 704, N 512–2048) ran at about
14 TFLOP/s with the old fixed tile ladder: 48 blocks on 28 SMs. With the tune store, their
mean time fell from 76.5 to about 60 µs on the RTX 3060.

## When measurement happens

The op measures where it is called, while the graph is built, never inside a running plan. Each
candidate is built as a program, launched on scratch buffers at capacity with every runtime
variable bound to its maximum, and timed:

| Step | Setting |
|---|---|
| Warm-up | Back-to-back runs until the clock plateaus (an RTX 3060 idles at 210 MHz) |
| Rounds | 4 round-robin rounds over all candidates |
| Per round | 10 ms of sustain runs, then 5 profiled runs, keeping each candidate's minimum |
| Time of a run | The sum of its kernels' GPU timestamps |

A candidate is a list of programs, a kernel plus the merge of its partial results for split
attention, and its time is the sum of its programs' times. A candidate that fails to build or
run is skipped. If nothing measures, the first candidate is used and nothing is stored.

Two guards keep a time honest:

- **One device lock.** Every timing in the process, the tuner's and the cost probes' alike,
  holds one lock across `tune::time`: two measurements sharing the device would time each other.
  Two threads tuning the same key measure it once; the second waits for the first's answer.
- **The stamp guard.** gfx1151 now and then stamps a long kernel at about a microsecond (seen
  once in some sixty runs of a 90 ms kernel). `tune::best` drops a run under half the median of
  its candidate's runs before keeping the fastest, for the tuner and the probes alike.

## The store

| What | Value |
|---|---|
| Directory | `$SVOD_TK3_TUNE_DIR`, else `$XDG_CACHE_HOME/svod/tk3_tune`, else `~/.cache/svod/tk3_tune` |
| File | One per device and crate version, e.g. `sm_86_28sm-v0.2.0.txt` |
| Line | `op\|device\|dtype\|shape\|candidates\|compiler\|programs index ns` |
| Key | `tune::TuneKey { op, device, dtype, shape, candidates, compiler }`, where `device` is arch and SM count (`sm_86-28sm`) and `compiler` digests the device's compiler (`tune::compiler`) |

A line from a real store: attention, f16, shape `[batch, t, tk, heads, kv_heads, d]`, candidate
1 won at 23.2 µs.

```text
attention|gfx1151-40sm|Float16|5x1x448x20x20x64|f0896a3761c96a19|465b6912a7cb2da1|9c2caed511ea3c95 1 23240
```

The `programs` field fingerprints the candidates' programs and lowerings, and `compiler` the
toolchain, so a kernel or compiler change re-measures. The programs are built (cheap) to
fingerprint them; only a store miss compiles them, so a warm store compiles nothing past
admission. Writes re-read, merge and atomically replace the file. An unreadable or unwritable
store counts as a miss, never an error. A process memo answers repeated calls without building
anything.

## Device costs

The ranking's costs (`atoms::Costs`: a shared byte, a streamed byte, a matrix-core step, a
score's softmax, a pipeline trip, lane moves, global latency, ...) are measured on the device
by `probe::costs` the first time a process plans for it: each field is a pair of programs, a
probe doing all its baseline does plus a known amount of the primitive. The values go in the
same store file as `probe:<field>` lines (index 0, femtoseconds in the `ns` slot), so later
processes read them back. The planner is made once per device; a second caller waits for the
probes. A device without an SM count cannot scale a probe and has no planner. Host tests
declare costs instead: `test::unit::costs()` and `gfx1151_costs()`, the latter pasted from a
gfx1151 store.

## Switching it off

| How | Effect |
|---|---|
| `SVOD_TK3_TUNE=0` | Candidate timing off; the first admitted candidate runs (the cost probes still measure) |
| `svod_tk3::tune::set_enabled(false)` | The same for this process, overriding the environment (tests use it) |

`tune::TuneStore::at(root)` builds a store at another root, or in memory only with `None`.
`tune::measure(candidates)` times any list of `tune::Candidate`
(`Vec<(Program, Lowering)>`) the same way.

## Probe tests

The probe tests are `#[ignore]` tests that print timings and never assert. They rank with the
device's probed costs and time the shortlist the tuner admits (the op layer's own admission),
marking the tuner's pick. Run them one at a time on an idle GPU:

```bash
SVOD_DEVICE=CUDA:0 cargo test -p svod-tk3 --lib --release -- --ignored --nocapture --test-threads=1 gemm_candidates_probe
```

| Probe | Prints |
|---|---|
| `gemm_throughput_probe` | The admitted GEMM configs at 4096³, TFLOP/s, and the tuner's pick |
| `gemm_candidates_probe` | Every admitted GEMM candidate on Nemotron's projection shapes and 4096³, the untuned pick, the fastest and the tuner's pick |
| `attention_throughput_probe` | Flash attention (B 4, H 8, T 2048; d 64/128, causal and not): every admitted config and the tuner's pick with its softmax forms |
| `decode_throughput_probe` | A Whisper large-v3 decoder step's self and cross attention |
| `first_execution_probe` | Host cost of building, lowering and preparing a tk3 GEMM against a graph GEMM |

The last probe measured the host costs that shaped `launch.rs`. Lowering a GEMM body costs
2.3 ms, so lowered bodies are memoized by program, lowering, device and placeholder shapes (a
hit costs 30 µs). Scheduling a body still costs about 0.35 ms more than a graph GEMM at prepare
time.
