---
sidebar_label: Tuning
---

# Tuning

Every op has a ranked config list per shape in `ops::config` (see the
[Kernel Library](./kernel-library) for the tables), ranked from the device's measured costs
(see [Device costs](#device-costs)). The ranking is a model, and so is a register count, so
neither admits a config. One rule does: the tuner compiles the ranked configs in batches of
64 (`config::SHORTLIST`), in ranked order and in parallel, until it has the first 64 tiles whose
compiled kernels do not spill (a kernel reporting scratch bytes spilled; a backend that reports
nothing admits). The shortlist runs that deep because the model orders block tiles well and warp
grids poorly: on the RTX 3060 the measured best of a Nemotron GEMM sat as deep as the 92nd of 109
tiles, and a shortlist of eight lost up to 26 % to it. The ranking lists every pipeline (ring
depth, unrolling) of a tile, and a tile is admitted at its first clean one, so an unrolled ring
that spills yields to its rolled twin. Spilling excludes a config only while clean ones remain:
when every config that compiles spills, the first 64 of those are timed anyway, since the time
judges and a spilling kernel still beats the graph. A list of one config takes the same path, and
no op falls to the graph because of spills. The admitted configs, with their variants (key splits
for attention, reduction splits for a starved convolution), are the candidates the tune store
measures the first time a device meets a shape, and it keeps their timed order; the other
pipelines of every tile within the contender band of the leader, at most eight
(`config::TWIN_TILES`), are then compiled and timed against the winner, spilling or not (among a tile's pipelines the time alone judges, an unrolled ring
that spills a little may be the fastest, and a tile's place in the shortlist is its first
pipeline's), and the fastest is the choice. Attention
tunes its softmax forms and its tiles in turn: the forms on the ranking's first clean config, the
tiles under those forms, then the forms again on the winner (see the
[Kernel Library](./kernel-library)). The admitted list is kept in the store beside the choice, so
a warm store compiles nothing but the winner. When nothing compiles (no device to compile on) the
first 64 tiles are used untested.

With tuning off (`SVOD_TK3_TUNE=0`) nothing is timed: the ranked list's first config that
compiles clean runs, found compiling in ranked batches of eight and kept in the store as a
`first` line (attention's forms pass starts from the same config). The device's cost probes
still measure, once per device (about 4 s on gfx1151), since the ranking needs them.

A better fixed default was not enough. Nemotron's GEMMs (M = 704, N 512–2048) ran at about
14 TFLOP/s with the old fixed tile ladder: 48 blocks on 28 SMs. With the tune store, their
mean time fell from 76.5 to about 60 µs on the RTX 3060.

## When measurement happens

The op measures where it is called, while the graph is built, never inside a running plan. Each
candidate is built as a program, launched on scratch buffers at capacity with every runtime
variable bound to its maximum (a float operand holds values drawn in `[-1, 1)` and an integer one
zeros: zeros toggle no bit, and a card that clocks down under real data, as the RX 9070 XT does by
15 % on the 4096³ GEMM, is timed as it will run), and timed:

| Step | Setting |
|---|---|
| Warm-up | Back-to-back runs until the clock plateaus (an RTX 3060 idles at 210 MHz) |
| Rounds | One round of every candidate, then 3 more of the contenders: those within 1.2× of the first round's best (every candidate, under 16 of them); then 4 more of the two leaders, where more than two contenders ran |
| Per round | 5 ms of device time in back-to-back sustain runs (sized by the candidate's last run), then 5 profiled runs, keeping each candidate's minimum |
| Time of a run | The sum of its kernels' GPU timestamps |

The sustain is sized in device time because launches queue asynchronously: sustaining for 10 ms
of host time once queued about a thousand launches, a second of a millisecond kernel per round,
and tuning eight 3072×3072×1024 GEMMs took a minute. Sized by device time, the same eight take
0.2 s and read the same.

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
| File | One per device and crate version, e.g. `sm_86_28sm-v0.2.0.txt`, headed by its line format (`format 4`, `tune::FORMAT`); a file of another format is dropped whole |
| Line | `kind\|op\|device\|dtype\|shape\|candidates\|compiler\|programs numbers`: a `choice` holds the candidates that ran, fastest first, each `index ns`, a `value` a measured number, an `admitted` the ranked indices admission kept, a `first` the first clean config's index |
| Key | `tune::TuneKey { op, device, dtype, shape, candidates, compiler }`, where `device` is arch and SM count (`sm_86-28sm`) and `compiler` digests the device's compiler (`tune::compiler`) |

Two lines from a real store: attention, bf16, shape `[batch, t, tk, heads, kv_heads, d]`,
candidate 0 won at 62.3 µs; the 4096³ GEMM's admission, the ranked indices of its clean tiles
(shortened).

```text
choice|attention|sm_86-28sm|BFloat16|1x704x704x8x8x64|23f7784292d1b0a0|9154cc72f03d25cd|1640bd140bf8e08b 0 62304 3 63010 1 64877
admitted|gemm|sm_86-28sm|BFloat16|1x4096x4096x4096|47ed81ff601380fe|9154cc72f03d25cd|1f0f4f6116dbeee7 4 10 18 22 30 …
```

The `programs` field fingerprints the candidates' programs and lowerings, and `compiler` the
toolchain, so a kernel or compiler change re-measures. The programs are built (cheap) to
fingerprint them (every fingerprint of a warm Qwen3 run together costs 45 ms); only a store miss
compiles them, so a warm store compiles nothing but the choice: on the RTX 3060 Qwen3 prepares in
2 s warm against 117 s on a fresh store. Writes re-read, merge and atomically replace the file. An unreadable or unwritable
store counts as a miss, never an error. A process memo answers repeated calls without building
anything.

## Device costs

The ranking's costs (`atoms::Costs`: a shared byte, a streamed byte, a matrix-core step under a
full SM and alone on its SIMD, a score's softmax, a pipeline trip over two ring slots (unrolled and rolled) and over one, lane moves, global latency, ...) are measured on the device
by `probe::costs` the first time a process plans for it: each field is a pair of programs, a
probe doing all its baseline does plus a known amount of the primitive. The values go in the
same store file as `probe:<field>` lines (index 0, femtoseconds in the `ns` slot), so later
processes read them back. The planner is made once per device; a second caller waits for the
probes. A probe that fails leaves its field at zero, is warned about, and is named in
`Planner::unmeasured`, which the device gate holds empty: the planner ranks with what it has rather
than leaving the device to the graph. Only a device without an SM count, which no probe can scale
to, has no planner. Host tests
declare costs instead: `test::unit::costs()` and `gfx1151_costs()`, the latter pasted from a
gfx1151 store.

## Switching it off

| How | Effect |
|---|---|
| `SVOD_TK3_TUNE=0` | Candidate timing off; the first ranked config that compiles clean runs and nothing is stored (the cost probes still measure) |
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
