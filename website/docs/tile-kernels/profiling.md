---
sidebar_label: Profiling
---

# Profiling with a Thread Trace

A kernel's time tells you that it is slow. A thread trace tells you why: which instruction each
wave waited on, for how long, and which line of the tile program that instruction came from. This
page takes you from "this kernel is slow" to a change in the tk program.

## Why a trace

The other tools see a kernel from outside. The device time says how long it took, the roofline
says how far it is from the memory and compute limits, and the counters say how busy the units
were. None of them says what a wave was doing on a given cycle. On AMD's RDNA cards the SQ thread
trace (SQTT) records exactly that, so you can see whether the main loop waits on global loads, on
the LDS, on a barrier, or on nothing at all.

Svod's profiler has five tiers. Each adds detail to the one below:

| Tier | What it gives | Where |
|---|---|---|
| 1, device time | Each kernel's time on the GPU clock | GPU backends; wall clock on the CPU |
| 2, roofline | GFLOP/s and GB/s from tier 1 and the kernel's static work estimate | Every backend |
| 3, static occupancy | Registers, LDS, scratch and an occupancy estimate, without a dispatch | AMD, CUDA; Metal in part |
| 4, hardware counters | Busy cycles, waves, instructions (`SVOD_PMC`) | AMD (PM4), CUDA (CUPTI) |
| 5, thread trace | Every traced wave's instructions with their cycles, in one extra pass | AMD gfx11 and gfx12 |

## What the trace records

A trace is a stream of tokens per shader engine (SE): when a wave started and ended, and for the
traced SIMD, each instruction it issued and when. Svod decodes the stream, maps every instruction
back to the kernel's disassembly, and splits the main loop's cycles into buckets:

- **issue**: the instruction was issuing, by its class (`matrix`, `valu`, `salu`, `lds`, `vmem_load`, …);
- **stall**: it was ready but its pipe or queue was busy;
- **wait**: an `s_wait_*` waited for a load, an LDS access or a store, charged to the load that bound it;
- **barrier**: the wave was parked at a workgroup barrier;
- **idle**: no record, because the SIMD was issuing for another wave or on a dependency the trace does not show.

Each bucket also has an **exposed** share: its cycles less what other pipes' work hid at the same
time. A load wait of 180 cycles while the matrix pipe is busy for 140 of them costs about 40. The
findings rank by exposed share, because only exposed cycles shorten the kernel when you remove them.

A **trip** is one pass of the tile program's loop body, which may be shorter than a hardware
iteration when the loop is unrolled. Cycles are a traced wave's, per trip.

## Scope and cost

- **Hardware.** AMD gfx11 (RDNA3, RDNA3.5) and gfx12 (RDNA4). Any other target is refused with
  `only gfx11 and gfx12 are traced`. Captures were verified on gfx1201 (RX 9070 XT) and gfx1151
  (Strix Halo). gfx1100 traces decode, but no hardware capture was verified.
- **gfx11 has no ceiling figure.** gfx11 records a matrix op as a one-cycle token, so the pipes'
  busy shares are unknown there. The report prints no `ceiling` line, and its findings use raw
  shares, marked `approx: untimed_matrix`.
- **PM4 queues only.** A device opened with `SVOD_AMD_AQL=1` refuses to arm the trace.
- **Window.** Each SE gets a 64 MiB buffer by default (`SqttConfig::window`, in 4 KiB pages,
  configurable). A buffer that fills loses its tail. The report then opens with `!! TRUNCATED`, and
  its totals undercount.
- **One extra pass per traced kernel.** The profile times its runs untraced, then runs the plan once
  more with the trace armed. That pass synchronizes after each kernel and frees its buffers before
  the next. No timing column comes from it.
- **Two SIMDs.** Instruction tokens come from SE0 and SE1, one SIMD each (`SqttConfig::se_mask`).
  `SVOD_SQTT_SIMD` chooses them: one SIMD for every SE, or a comma list repeated over the SEs
  (default `0,1`, SIMD 0 on SE0 and SIMD 1 on SE1). Per-instruction numbers describe the waves on
  those two SIMDs.

## Clocks and prerequisites

- An AMD card of the targets above, selected with `SVOD_DEVICE=AMD` (or `AMD:N`), running PM4 queues.
- `llvm-objdump` next to the clang Svod compiles with. A capture saves its disassembly, so a replay
  needs the tool only when `kernel.s` is missing.
- **Unpinned clock, the default.** The traced pass runs at whatever clock the device picks, just as
  an untraced run does. `SVOD_SQTT_PSTATE=standard` (or `peak`) pins a stable power state through
  the DRM context. That lowers the shader clock and distorts what the trace shows. Use it only to
  match an older pinned capture.
- A quiet GPU. Tracing perturbs the run anyway: the report prints the traced run's time over the
  untraced one, and above 1.2× it marks the findings `approx: perturbed`. Never compare a traced
  time with an untraced one.

## Tracing from Rust

`ProfileOptions::sqtt` turns tier 5 on for any profiled plan (`ExecutionPlan::profile`) or tensor
(`Tensor::profile`):

```rust
use svod_runtime::{ProfileOptions, SqttOptions};

let opts = ProfileOptions { sqtt: Some(SqttOptions::default()), ..Default::default() };
let profile = plan.profile(&opts)?;
for traced in profile.stages.iter().flat_map(|stage| &stage.sqtt) {
    match &traced.analysis {
        Ok(analysis) => println!("{}: {}", traced.kernel, analysis.summary()),
        Err(error) => println!("{}: {error}", traced.kernel),
    }
}
```

`SqttOptions` holds a `SqttConfig` (SEs, SIMDs, window, power state) and a `Sink`:

- `Sink::Disk(dir)`, the default, saves each kernel's capture under `dir/<index>_<entry point>`
  (`./sqtt` by default), so the pass holds one capture at a time. `Traced::raw` is `Raw::Saved(path)`.
- `Sink::Memory` keeps each capture in `Raw::Held`. A capture can take up to 64 MiB an SE, so a
  whole model's may not fit.

Each `Traced` carries its own `analysis`. A kernel whose trace does not analyze leaves the others
intact. `RunProfile::render_report` ends with one summary line per traced kernel.

## Tracing a model

The model examples' `--profile` flag does not trace. To trace a model, profile its output tensor
with the options above. Every compiled kernel in the plan is traced: tk kernels and graph kernels
alike. `SqttOptions::from_env()` builds the options from the environment for a program that wants
a switch: `SVOD_SQTT=1` traces, `SVOD_SQTT_DIR` names the directory, and `SVOD_SQTT_PSTATE` and
`SVOD_SQTT_SIMD` set the config. It returns `Ok(None)` when `SVOD_SQTT` is not `1`, and an error
for a value that does not parse.

Start with the per-kernel table of the untraced run, pick the kernel that dominates, and read its
directory with `tk_bench sqtt --load` (below). For a tk kernel, trace it on its own next, so
that you can vary its config.

## Tracing one kernel

`tk_bench sqtt` builds one tk op at a shape, times each config untraced as the tuner does, traces
it once, saves the capture and prints its report:

```bash
SVOD_DEVICE=AMD cargo run --release -p svod-tk --example tk_bench -- sqtt nearest 4096 128 16
```

| Command | Traces |
|---|---|
| `sqtt gemm M N K [--residual] [--gated]` | A bf16 GEMM, with a residual or SwiGLU epilogue |
| `sqtt attention B H T D [--causal] [--bias] [--flip FORM,…] [--ring S]` | bf16 attention over `[B, T, H, D]`; `--flip` flips softmax forms of each config, `--ring` sets its ring slots |
| `sqtt nearest N D K [--metric M] [--splits S]` | A self-join of `N` f16 rows at depth `D`, keeping `K` a row; `--splits` overrides the config's splits |
| `sqtt conv H W CIN COUT KERNEL STRIDE [--batch B] [--residual]` | A YOLO block's f16 convolution with bias and SiLU |

All four take `--cfg`, `--top N` (table rows, 12 by default), `--out DIR` (`./sqtt` by default)
and `--json` (the report as one JSON line). `--cfg` is a comma list. `tuned`, the default, is the
config the op would launch. A number `I` is the planner's `I`th ranked config. GEMM also takes a
pinned code `BMxBNxBK:S:WRxWC[:rolled]`, and `nearest` and `conv` take `ranked`, which lists the
ranking instead of tracing. Each config lands in
`DIR/<op>_<shape>_<cfg>/p<program>/<index>_<entry>/`, and the path is printed as `saved …`.

## The saved directory and replay

| File | Holds |
|---|---|
| `se0.att`, `se1.att`, … | Each SE's raw trace stream |
| `kernel.co` | The traced code object; a tk kernel launched on AMD also carries a line table over its `kernel.tk` listing, and the listing itself, in its DWARF |
| `kernel.s` | `llvm-objdump -d` of `kernel.co`; PCs in the report are its addresses |
| `meta.json` | `format` 2, arch, load base, untraced and traced ns, the SEs, labels (op, shape, cfg), the loop model, resources, workgroup and grid, power state, memory clock |

A load checks `format` and asks you to capture again when it differs. Statements are read back from
`kernel.co` on every load. Only AMD tk kernels have them; graph kernels have none (CUDA and Metal
have no thread trace). Without statements, a trip is taken to be one loop iteration, marked
`approx: per_iteration`.

Replay needs no GPU:

```bash
cargo run --release -p svod-tk --example tk_bench -- sqtt --load DIR [--json] [--attribute]
```

From Rust, `svod_runtime::sqtt::Analysis::load(dir)` returns the same analysis.

## Reading the report

The `rdna4_nearest_split` fixture (`device/src/test/fixtures/sqtt/svod/`) is a gfx1201 nearest
kernel, n4096 d128 k16. Abridged, its report reads:

```text
== verdict: cycles are a traced wave's, a trip (1 an iteration, its statements' unroll copies); exposed: not hidden by other pipes' work
ceiling    matrix pipe 14.7% busy over the steady window: removing every stall gains at most 6.82x; …
bound      occupancy_limited  56.6%  SIMD idle 56.6%: 2.0 waves a SIMD, ~4.6 would fill it; limit 8 by vgpr+lds
   knob: fewer waves resident than the kernel allows: more workgroups in flight
 1 barrier            18.6% (raw 23.4%)  barrier (sink): 255/trip parked for the workgroup's slowest wave, 203 exposed; …
   knob: drop or merge a barrier only if its exposed share passes 10%
 2 load_exposed       13.0% (raw 16.8%)  cycles a trip: waiting on loads 183 (141 exposed); stalled issuing 0 (0 exposed)
   0x364 load exposed 129/trip at wait 0x44c (then ds_store_b128 v177, v[105:108]): cover med 91, latency med 238, … @21 fill c staged issue [steady]
   knob: latency: more work between a load and its wait (prefetch distance, ring slots)
 3 loop_overhead       2.7% (raw 3.2%)  valu 26 + salu 9 issue cycles a trip beside matrix 146 (24%), …
kernel     144840 ns untraced, 148360 ns traced (1.02x), wave span 461319 cycles, sclk ~3109 MHz, mclk 875 MHz, pstate unpinned
traced     se0 sa0 wgp0 simd0 (2 waves), se1 sa0 wgp0 simd1 (2 waves) of 2 SEs
== statements: main loop cycles a trip by kernel.tk line (a wait on its binding load's); 104 of 105 loop instructions attributed; top 12 by exposed
 line  pcs    issue    stall     wait  barrier     idle    total  exposed  statement
 1581    1      0.0      0.0      0.0    255.1      0.0    255.1    202.6  await [steady]
   21    8      7.0      2.1    182.7      0.0     32.7    224.5    162.5  fill c staged issue [steady]
   28   24    145.7     11.4      0.0      0.0     75.5    232.6    148.0  mma x·c_s [steady]
```

Read it top down:

1. **`ceiling`**: how busy the matrix pipe was. At 14.7%, this loop is far from the matrix limit,
   and removing every stall would gain at most 6.82×.
2. **`bound`**: the kernel's class, from the pipes' busy shares (`matrix_bound`, `issue_bound` or
   `occupancy_limited`).
3. **Numbered findings**: what the loop's exposed cycles went to, the largest first. Each lists its
   evidence (PCs, the load behind a wait, who arrived last at a barrier), the tk statement after
   `@`, and a `knob` line saying what to change.
4. **`kernel` and `traced`**: the untraced and traced times, the clock, and which SIMDs were traced.
5. **`statements`**: the loop's cycles per trip by `kernel.tk` line. `[steady]` is the loop's steady
   state; `[prologue]`, `[head]`, `[drain]` and `[epilogue]` are the parts around it.

Further sections follow: time by class, occupancy per SE, the main loop's cycles per trip by
bucket, the top stalls, the waits with the loads that bound them, and the hottest instructions.

## Verdict classes and their levers

The thresholds are heuristics, kept as constants in `runtime/src/sqtt/verdict.rs`.

| Class | Fires when | Knob | The tk lever |
|---|---|---|---|
| `matrix_bound` | Matrix pipe busy ≥ 70% of the steady window | `less_matrix_work` | A cheaper type or fewer matrix ops; the schedule is not the problem |
| `issue_bound` | Some pipe busy ≥ 90%, under 60% of that matrix | `cut_issue` | Drop non-matrix instructions from the loop |
| `occupancy_limited` | SIMDs idle ≥ 10% | `free_resources`, `more_workgroups`, `more_waves` | Fewer VGPRs or less LDS a workgroup; a smaller tile or more splits for more workgroups |
| `load_exposed` | Global load waits and stalls ≥ 10% of the loop, exposed | `latency`, `move_use` | A deeper ring or a load issued a step ahead; `move_use` names the use right behind the load |
| `lds_exposed` | LDS waits and stalls ≥ 10%, exposed | `lds_flush`, `lds_use`, `lds_queue` | Fewer and wider `ds` ops, the next fragment read earlier, a conflict-free layout |
| `barrier` | Barrier cycles ≥ 10% of the loop | `barrier` | Fix the other findings first; drop, merge or split a barrier only if its exposed share passes 10% |
| `loop_overhead` | VALU+SALU issue ≥ 10% of the matrix issue, or any divergent branch | `hoist` | Hoist address and guard arithmetic; make guards uniform |

A barrier is a sink: the waves park there for the slowest wave, so its time belongs to whatever
made that wave slow. The finding lists who arrived last and after which instruction.

A finding marked `approx` has figures to read with care: `perturbed`, `untimed_matrix`,
`approx_bindings` (a guessed binding), `implausible_time` (a corrupt stream), `truncated` or
`per_iteration`.

## `--attribute` and `--compare`

`--attribute` is GEMM-only, because only a `tk_bench sqtt gemm` capture carries the cost model's
price. It sets each priced term (`steps`, `reads`, `fill`, `wait`, `lone_trip`, `trip`) beside the
traced cycles of the instructions it prices. The terms overlap, so compare shares and the pace per
trip, not totals.

`--compare A B` sets B's capture against A's (Δ = B − A):

```bash
cargo run --release -p svod-tk --example tk_bench -- sqtt --compare A B [--json] [--top N]
```

It prints both times, the findings that appeared, disappeared or changed, the loop buckets per trip
by |Δ exposed|, the statements by text, and the loop's instructions aligned by mnemonic. It warns
with `!!` when the two captures ran at different power states, when their traced/untraced ratios
are more than 10% apart, or when their trips cover different amounts of the reduction. A capture
with no main loop has nothing to align.

## The workflow

1. Find the slow kernel in the per-kernel table of an untraced profile.
2. Trace it: `tk_bench sqtt <op> <shape>`, or `Tensor::profile` with `SqttOptions` for a kernel
   that only exists inside a model.
3. Read `ceiling` and `bound`. A `matrix_bound` kernel is near its limit: change the work, not the
   schedule.
4. Take the first finding, its evidence and its `@` statement. That statement is the line of the
   tile program to change.
5. Make one change, trace again with `--out` set to a new directory, and run
   `--compare old new`. Check that the targeted bucket moved, and judge the gain by the untraced
   time only.
6. Confirm with `tk_bench` or the tuner on an idle GPU before keeping the change.

## Worked examples

### Attention: loading the bias a step ahead

A directly read attention bias tile was loaded in the step whose scores used it, so its wait sat
right behind the load. That is the pattern `load_exposed` names with `move_use`. The kernel now
loads each tile one step ahead (`pipeline_with` carries it into the next step in
`tk/src/kernels/attention.rs`), so the load has a whole step to land. PR #230 measured −19% at
d128 and −7% at d64.

### Nearest: splitting the barrier on RDNA4

The `rdna4_nearest_joined` and `rdna4_nearest_split` fixtures are the same `NearestCfg` (n4096
d128 k16) before and after RDNA4's split barrier. On gfx12 the register-staged ring's steady loop
now signals the barrier (`arrive`, `s_barrier_signal`) after the LDS commit and waits for it
(`await`, `s_barrier_wait`) at the end of the trip. The tail between the two runs while slower waves
catch up. `--compare` on the fixtures shows it:

```text
time       180320 -> 144840 ns untraced (-19.7%); traced/untraced 1.01x -> 1.02x; …
changed    barrier            46.8% ->  18.6%
appears    load_exposed           - ->  13.0%
bucket                A raw    B raw    Δ raw    A exp    B exp    Δ exp
barrier              1081.7    255.1   -826.6    962.9    202.6   -760.2
   A raw    B raw    Δ raw    A exp    B exp    Δ exp  statement
  1081.7      0.0  -1081.7    962.9      0.0   -962.9  barrier [steady]
     0.0    255.1   +255.1      0.0    202.6   +202.6  await [steady]
```

Barrier exposure falls from 963 to 203 cycles a trip, the `barrier [steady]` statement becomes
`arrive` + `await [steady]`, and the untraced time goes from 180 to 145 µs. Each side is a single
traced run. PR #230's benchmark measured −5.6% for nearest d128 k16. The `load_exposed` finding
that appears is the next lever: with the barrier gone, the global loads' latency shows.

### GEMM: a ragged M

The `rdna4_gemm_1536` and `rdna4_gemm_1500` fixtures are the same GEMM config at M 1536 and
M 1500 (N 1280, K 5120, Whisper's fc2). Both replays say `bound matrix_bound`. Only the 1500 one
adds `loop_overhead`, whose evidence is eight divergent branches in the main loop:

```text
 2 loop_overhead       0.5% (raw 0.7%)  valu 24 + salu 7 issue cycles an iteration beside matrix 443 (7%), 22 valu ops an iteration, 8 divergent branches (approx: per_iteration)
   0x5f0 s_and_saveexec_b32 s1, vcc_lo
   0x5f4 s_cbranch_execz 3
   knob: hoist address and guard arithmetic out of the loop; make guards uniform (no exec branches)
```

An M that is not a multiple of the tile puts the edge guards inside the loop as exec branches,
and the 1536 capture, a multiple of its 128-row tile, has none. The lesson is qualitative: a ragged
edge costs loop instructions that a shape padded to the tile does not pay, so check the padded
shape before tuning the schedule of a ragged one.

## Caveats

- The trace sees two SIMDs. A grid that never lands on them traces nothing there.
- A traced time is never a benchmark. Judge a change by `tk_bench` or the tuner, untraced.
- `exposed` is an estimate of what other pipes hid, so it is not a promise of the time a fix saves.
- The thresholds are heuristics. A finding below 10% is evidence, not a verdict.
- Without statements (a graph kernel, or a capture whose code object has no line table), figures are
  per loop iteration rather than per trip.
- A capture from an older `meta.json` format does not load; capture it again.
