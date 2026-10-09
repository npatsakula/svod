---
sidebar_label: Tuning
---

# Tuning

Every op has a candidate list per shape in `ops::config` (see the
[Kernel Library](./kernel-library) for the tables). The first candidate is what runs untuned.
The tune store measures the whole list the first time a device meets a shape and keeps the
winner.

A better fixed default was not enough. Nemotron's GEMMs (M = 704, N 512–2048) ran at about
14 TFLOP/s with the old fixed tile ladder: 48 blocks on 28 SMs. With the tune store, their
mean time fell from 76.5 to about 60 µs on the RTX 3060.

## When measurement happens

The op measures where it is called, while the graph is built, never inside a running plan. Each
candidate is built as a program, launched on scratch buffers at capacity with every runtime
variable bound to its maximum, and timed:

| Step | Setting |
|---|---|
| Warm-up | 500 ms of back-to-back runs (an RTX 3060 idles at 210 MHz) |
| Rounds | 4 round-robin rounds over all candidates |
| Per round | 10 ms of sustain runs, then 5 profiled runs, keeping each candidate's minimum |
| Time of a run | The longest kernel's GPU timestamps |

A candidate is a list of programs, a kernel plus the merge of its partial results for split
attention, and its time is the sum of its programs' times. A candidate that fails to build or
run is skipped. If nothing measures, the first candidate is
used and nothing is stored.

## The store

| What | Value |
|---|---|
| Directory | `$SVOD_TK3_TUNE_DIR`, else `$XDG_CACHE_HOME/svod/tk3_tune`, else `~/.cache/svod/tk3_tune` |
| File | One per device and crate version, e.g. `sm_86_28sm-v0.2.0.txt` |
| Line | `op\|device\|dtype\|shape\|candidates\|programs index ns` |
| Key | `tune::TuneKey { op, device, dtype, shape, candidates }`, where `device` is arch and SM count (`sm_86-28sm`) |

A line from a real store: attention, bf16, shape `[batch, t, tk, heads, kv_heads, d]`, candidate
2 won at 61.4 µs.

```text
attention|sm_86-28sm|BFloat16|1x704x704x8x8x64|3733e8921b15aa79|77aee9d5c98fa463 2 61440
```

The `programs` field fingerprints the built candidate programs and their lowerings, so a kernel
change re-measures. Writes re-read, merge and atomically replace the file. An unreadable or
unwritable store counts as a miss, never an error. A process memo answers repeated calls
without building anything.

## Switching it off

| How | Effect |
|---|---|
| `SVOD_TK3_TUNE=0` | Measuring off; the first candidate runs |
| `svod_tk3::tune::set_enabled(false)` | The same for this process, overriding the environment (tests use it) |

`tune::TuneStore::at(root)` builds a store at another root, or in memory only with `None`.
`tune::measure(candidates)` times any list of `tune::Candidate`
(`Vec<(Program, Lowering)>`) the same way.

## Probes

The probes are `#[ignore]` tests that print timings and never assert. Run them one at a time on
an idle GPU:

```bash
SVOD_DEVICE=CUDA:0 cargo test -p svod-tk3 --lib --release -- --ignored --nocapture --test-threads=1 gemm_candidates_probe
```

| Probe | Prints |
|---|---|
| `gemm_throughput_probe` | tk3 GEMM configs at 4096³, TFLOP/s |
| `gemm_candidates_probe` | Every GEMM candidate on Nemotron's projection shapes and 4096³, the untuned pick and the winner |
| `attention_throughput_probe` | Flash attention (B 4, H 8, T 2048; d 64/128, causal and not) |
| `decode_throughput_probe` | A Whisper large-v3 decoder step's self and cross attention |
| `first_execution_probe` | Host cost of building, lowering and preparing a tk3 GEMM against a graph GEMM |

The last probe measured the host costs that shaped `launch.rs`. Lowering a GEMM body costs
2.3 ms, so lowered bodies are memoized by program, lowering, device and placeholder shapes (a
hit costs 30 µs). Scheduling a body still costs about 0.35 ms more than a graph GEMM at prepare
time.
