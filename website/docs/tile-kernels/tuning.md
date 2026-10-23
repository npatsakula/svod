---
sidebar_label: Tuning
---

# Tuning

Each op ranks its configs per shape in `ops::config` from the device's measured costs (see
[Device costs](#device-costs)), dropping the configs priced past 2.5× the cheapest once 16 tiles are
kept (`config::PRUNED_PAST`): on sm_86 none of them ran within the contender band. The ranking is a model, and so is a register count, so neither
admits a config. One rule does: the tuner compiles the ranked configs in parallel until it has
the first 64 tiles in ranked order (`config::SHORTLIST`) whose compiled kernels do not spill (a
kernel reporting scratch bytes spilled; a backend that reports nothing admits). A tile is
admitted at its first clean pipeline (ring depth, unrolling), so an unrolled ring that spills
yields to its rolled twin. A pipeline is compiled only when the tile's pipelines ranked ahead of
it spilled or failed, and none past the 64th clean tile: what a walk in ranked order compiling
one config at a time would compile (`config::Search`). The tiles none of whose pipelines compile
clean follow the clean ones within the 64, a pipeline each: the compiled kernel judges what it
can and the time the rest (no op falls to the graph because of spills, and on gfx1201 the 16384²
depth-768 nearest's best tile spills 196 bytes a lane and runs 2 % ahead of the clean pick). A config's
compile starts as soon as the search wants it, on the thread pool (where the pool has room, a tile's next pipeline starts while the one ahead still compiles, its verdict used only where the walk would have asked for it); one whose compiler runs past
30 s (`tune::COMPILE_TIMEOUT`) is killed and rejected as a failure would be, and the planner log
says so (`compile timed out`); one whose lowered program emits more instructions than its target
may (`Target::most_instructions`: 2^18 on AMD, 2^19 on CUDA, above the largest ranked kernels) is
refused by the lowering before any compiler sees it (`lower::Error::TooLarge`, `does not lower` in
the planner log) and rejected alike. Each program is lowered once per tuning: what the store's
fingerprint lowered is the body the compile launches.

The admitted configs, with their variants (key splits for attention, reduction splits for a
starved convolution), are the candidates the tune store measures the first time a device meets
a shape. The other pipelines of the leading tiles, those within the contender band of the
leader and at most eight (`config::TWIN_TILES`), are then compiled and timed against the winner,
spilling or not, and the fastest is the choice. Attention tunes its softmax forms and its tiles
in turn (see the [Kernel Library](./kernel-library#flash-attention)). When nothing compiles (no
device to compile on) the first 64 tiles are used untested.

With tuning off nothing is timed: the ranked list's first config that compiles clean runs,
found compiling in ranked batches of eight and kept in the store as a `first` line. The
device's cost probes still measure, since the ranking needs them.

## When measurement happens

The op measures where it is called, while the graph is built, never inside a running plan. Each
candidate is built as a program, launched on scratch buffers at capacity with every runtime
variable bound to its maximum (float operands hold values drawn in `[-1, 1)`, so a card that
clocks down under real data is timed as it will run; integer operands hold zeros), and timed:

| Step | Setting |
|---|---|
| Warm-up | Back-to-back runs until the clock plateaus |
| Rounds | One round of every candidate in ranked order, where a candidate whose first timed run passes 2× the best so far keeps that run as its time; then 3 more of the contenders: those within 1.2× of the first round's best (within 2×, under 8 candidates), each round past the second only those within 1.05× of the best so far, the later rounds running backwards and forwards in turn so that each candidate runs after both its neighbours; then 4 more of the two leaders, where more than two contenders ran. The first timing of a tuning, which only picks its leading tiles, stops after the second round: those tiles are timed again with their other pipelines |
| Per round | 5 ms of device time in back-to-back sustain runs of the candidate's programs in order (at most 1024 launches, sized by the candidate's last run), then profiled runs of each program, keeping each candidate's minimum: as many as 20 ms of device time holds, 2 to 5 (`tune::runs`), and no sustain for a candidate of 5 ms or more, whose first timed run then counts among them; the leaders' rounds (and the cost probes) keep 5 runs after a sustain |
| Time of a run | The sum of its kernels' GPU timestamps |

The sustain is sized in device time because launches queue asynchronously. A candidate is a
list of programs, a kernel plus the merge of its partial results for a split, and its time is
the sum of its programs' times. Candidates not compiled yet when a timing starts (a split's
variants, a tile's twins) compile on all but one of the pool's threads while the ones ready are
timed, each timed when the ranked order reaches it. A candidate that fails to build or run is skipped. If nothing
measures, the first candidate is used and nothing is stored.

Two guards keep a time honest:

- **One device lock.** Every timing in the process, the tuner's and the cost probes' alike,
  holds one lock: two measurements sharing the device would time each other. Two threads tuning
  the same key measure it once; the second waits for the first's answer.
- **The stamp guard.** A run under half the median of its candidate's runs is a lost stamp and
  is dropped before the fastest is kept, for the tuner and the probes alike.

## The store

| What | Value |
|---|---|
| Directory | `$SVOD_TK_TUNE_DIR`, else `$XDG_CACHE_HOME/svod/tk_tune`, else `~/.cache/svod/tk_tune` |
| File | One per device and crate version, e.g. `sm_86_28sm-v0.2.0.txt`, headed by its line format (`format 5`); a file of another format, or without the line, is dropped whole |
| Line | `kind\|op\|device\|dtype\|shape\|candidates\|compiler\|programs numbers`: a `choice` holds the candidates that ran, fastest first, each `index ns`; a `value` a measured number; an `admitted` the ranked indices admission kept; a `first` the first clean config's index |
| Key | `tune::TuneKey { op, device, dtype, shape, candidates, compiler }`, where `device` is arch and SM count (`sm_86-28sm`) and `compiler` digests the device's compiler (`tune::compiler`) |

Lines from a real store on the RTX 3060: an attention choice (shape `[batch, t, tk, heads,
kv_heads, d]`, candidate 0 at 3904 ns) and a cost probe's value (femtoseconds per unit).

```text
choice|attention|sm_86-28sm|BFloat16|2x10x12x6x2x64|08bd45c243ff1581|9154cc72f03d25cd|ef103f71226add50 0 3904 1 3936
value|probe:cached_1m|sm_86-28sm|Int32||0a2d6c02b82988af|9154cc72f03d25cd|d2a8e296bcce85e4 16601
```

The `programs` field fingerprints the kernels the candidates emit: the lowered program and the
LLVM text the backend is handed, so a change to the kernels or the renderer re-measures and a
refactor that emits the same kernels does not. What a program emits is remembered by what it is
made of, per build of the executable, in `emitted/<build>.txt` under the store's root (a build
unused for a week is removed), so a warm process of the same build lowers nothing and a rebuild
lowers each program once. Only a store miss compiles. Writes re-read, merge and atomically
replace the file. An unreadable or unwritable store counts as a miss, never an error.

## Device costs

The ranking's costs (`atoms::Costs`: a shared byte, a streamed byte, a scattered element, a
matrix-core step under a full SM, alone on its SIMD and as a warpgroup product, a score's
softmax, a pipeline trip over two ring slots unrolled and rolled and over one, the two-slot and
one-slot trips of a block alone on its SM, a lane move, a shared round trip, the global latency
through two ring slots and, where fills are `cp.async`, through one, and the L2 rate and size)
are measured on the device by `probe::costs` the first time a process plans for it. Each field is a pair of programs, a probe
doing all its baseline does plus a known amount of the primitive. The values go in the same
store file as `value|probe:<field>` lines, so later processes read them back. A probe that
fails leaves its field at zero, is warned about, and is named in `Planner::unmeasured`, which
the device gate `device_costs_are_physical` holds empty: the planner ranks with what it has
rather than leaving the device to the graph. Only a device without an SM count has no planner.

Host tests declare costs instead (`test::unit::costs()`, and `gfx1151_costs()` pasted from a
gfx1151 store). For a card this machine has no device for, `Target::offline` or `Target::card`
gives the target, and `probe::stored` reads the costs from a tune store copied from that card;
without one, `probe::GUESSED` stands in and the ranking is a guess. The
[`tk_dump` example](./testing-and-debugging#planning-for-another-card) does both.

## The planner log

`RUST_LOG=svod_tk::plan=debug` logs the planner's decisions: an op that takes the graph and
why, a ranking's first 16 configs with their predicted ns and how many configs it left out, on a
store miss what compiling each config showed (verdict, registers, spilled bytes), and the tuned
pick. `RUST_LOG=svod_tk=debug` adds the tuner's rounds.

## Switching it off

| How | Effect |
|---|---|
| `SVOD_TK_TUNE=0` | Candidate timing off; the first ranked config that compiles clean runs (kept as a `first` line); the cost probes still measure |
| `svod_tk::tune::scoped(false, f)` | The same for `f` on this thread (and the workers it compiles on), over the environment; tests use it |

`tune::TuneStore::at(root)` builds a store at another root, or in memory only with `None`.

## Probe tests

The probe tests are `#[ignore]` tests that print timings and never assert. They rank with the
device's probed costs and time the shortlist the tuner admits, marking the tuner's pick. Run
them one at a time on an idle GPU:

```bash
SVOD_DEVICE=CUDA cargo test -p svod-tk --release --lib gemm_candidates_probe -- --ignored --nocapture --test-threads=1
```

| Probe | Prints |
|---|---|
| `gemm_throughput_probe` | The admitted GEMM configs at 4096³ bf16, TFLOP/s, and the tuner's pick |
| `gemm_candidates_probe` | Every admitted GEMM candidate on Nemotron's projection shapes and 4096³: the untuned pick, the fastest and the tuner's pick |
| `qwen3_gemm_candidates_probe` | Qwen3-Embedding's projections: the shortlist, then the rest of the lattice |
| `attention_throughput_probe` | Flash attention (B 4, H 8, T 2048; d 64/128, causal and not, d 64 with a bias): every admitted config and the tuner's pick with its softmax forms |
| `attention_candidates_probe` | Every attention config of the lattice against the admitted shortlist |
| `decode_throughput_probe` | A Whisper large-v3 decoder step's self and cross attention, tuned splits and unsplit |
| `conv_throughput_probe` | Every admitted convolution candidate per YOLO26 class against the graph convolution |
| `norm_bandwidth_probe` | The row kernels on 8192×1024 bf16, in GB/s |
| `first_execution_probe` | Host cost of building, lowering and preparing a tk GEMM against a graph GEMM |
