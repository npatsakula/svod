---
name: tk-kernels
description: Write, check, debug and profile tile kernels with the `svod-tk` crate (tk/src). Use when adding or changing a tile program (`build::Kernel`), wiring it into `svod_tk::ops` and the config lattice, running it in the host interpreter or on a GPU, reading a lowering/layout/admission rejection, dumping a tk kernel's instruction list, LLVM IR, PTX/SASS or AMDGCN for a card this machine does not have, comparing its main loop with cuBLAS/CUTLASS/ThunderKittens/HipKittens/AITER ISA, reading the planner log or the tune store (or the config a card's store chose, offline), running the device probes and gates, or checking a change for perf regressions or proving a refactor byte-identical. Covers SVOD_DEVICE for the device gate, the tk_dump example (--cfg tuned, --compare), the tk_digest example, SVOD_DUMP_AMD_ASM, RUST_LOG=svod_tk::plan, TK_DUMP_LIST, SVOD_TK_TUNE, SVOD_TK_TUNE_DIR, device_costs_are_physical, families_match_the_interpreter_on_the_device, the *_probe tests.
---

# tk tile kernels

Reader docs (link, do not duplicate): `website/docs/tile-kernels/{overview,first-kernel,layouts-and-lowering,op-layer,kernel-library,tuning,testing-and-debugging,portability}.md`.
Design: `tk3_design.md`; per-GPU numbers and the commands behind them: `tk3_pr.md` (Tables A/B, "How to add a column").
Both sit untracked at the root of the integrator's checkout. Graph-side debugging is `/svod-debug`.
References here: [new-kernel.md](new-kernel.md) (wiring a kernel end to end).

## Quick start

Always `--release` (~45 s host suite in release, ~14 min in debug). Every device test goes through one gate,
`device_target()` (`tk/src/test/unit/mod.rs`): it **fails** when a GPU tk supports is attached and `SVOD_DEVICE` is
unset, skips (`skipped: …`, test passes) under `SVOD_DEVICE=CPU` or where there is no supported GPU. So: `SVOD_DEVICE=CPU`
for host work, `SVOD_DEVICE=CUDA` / `AMD` (or `CUDA:0`, `AMD:1`) for device work. One timed process per GPU.

| Question | Command |
|----------|---------|
| Host suite (no GPU touched) | `SVOD_DEVICE=CPU cargo test -p svod-tk --release --lib` (818 passed, 12 ignored, 44 s at 2e9d5edb) |
| Host gate: every family lowers on sm_86/sm_90/gfx1201/gfx1151/gfx1100/gfx942, keeps its interpreted numbers, compiles | `SVOD_DEVICE=CPU cargo test -p svod-tk --release --lib every_family_lowers_keeps_its_numbers_and_compiles` (39 s; no clang AMDGPU / NVPTX → `skipped:`; no `ptxas` → PTX not assembled) |
| One `test_case` variant | filter on the generated suffix: `…compiles::test_case_3_gfx1201` |
| Device gate: each family vs the interpreter | `SVOD_DEVICE=CUDA cargo test -p svod-tk --release --lib families_match_the_interpreter_on_the_device -- --nocapture` |
| Cost probes on a fresh store (alone, first) | `SVOD_DEVICE=… cargo test -p svod-tk --release --lib device_costs_are_physical -- --nocapture` |
| What would tk run on an H100 / MI300X I don't have? | `cargo run -p svod-tk --release --example tk_dump -- --card h100-sxm --op gemm --shape 4096x4096x4096` (see Offline) |
| What did the tuner pick on that card? | the same with `--cfg tuned --store <its tune store>` (read-only replay) |
| Main loop beside cuBLAS/TK/HK/AITER | the same with `--compare <.cubin/.co/.s/.sass> [--fun NAME]` |
| Did a refactor change any emitted kernel? | `tk_digest --out A` at base and change, `tk_digest --diff A B` (see Performance-regression protocol) |
| Kernel or graph for a shape, and why | `RUST_LOG=svod_tk::plan=debug` (`graph … why=Shape`, `ranking`, `ranked`, `compiled`, `admitted`, `choice`, `tuned`) |
| Emitted instruction list | `TK_DUMP_LIST=1` (stderr, every `emit` call); pin one config with `SVOD_TK_TUNE=0` or use `tk_dump` |
| LLVM IR / AMDGCN as compiled on a device | `SVOD_DUMP_NVPTX_IR=<dir>` / `SVOD_DUMP_AMD_IR=<dir>` → `<arch>_<kernel>_<ir digest>.ll`, `SVOD_DUMP_AMD_ASM=<dir>` → `.s` |
| Tuning off | `SVOD_TK_TUNE=0` (process) or `tune::scoped(false, f)` (this thread): the first clean ranked config runs; probes still measure |
| GEMM TFLOP/s per config | `SVOD_DEVICE=… cargo test -p svod-tk --release --lib gemm_throughput_probe -- --ignored --nocapture --test-threads=1` |

Not for tk kernels: `SVOD_DUMP_STAGE`, `SVOD_PER_STAGE_UOPS`, `SVOD_DUMP_LINEAR`, `BEAM`, `SVOD_NOOPT`. A tk body is a
pre-linearized PROGRAM (`UOp::linear_program`): the optimizer skips it (`schedule/src/optimizer/mod.rs`, "A PROGRAM is past
optimization"), `do_linearize` returns before the `SVOD_DUMP_LINEAR` dump (`codegen/src/program_pipeline.rs`), and
`adopt_program` only verifies it (`SVOD_SPEC`/`SVOD_SPEC_DEBUG` apply).

## Mental model

1. **Tile program** (`ir.rs`, recorded by `build::Kernel`): statements in program order
   (`Stmt::{Let, Copy, Loop, Pipeline, Warps, If, Sync}`) over typed tiles `Gmem<T>`/`Shared<T>`/`Regs<T>`, shared
   allocations as `SmemRef<T>`; comparisons are `k.compare(a, b, CmpOp::{Lt,Le,Eq,Ne})` → `Regs<Bool>`. No DAG.
2. **`lower::lower`** (`lower/mod.rs`): grid check → `materialize_operands` (non-register `mma` operands become loads;
   on `Core::Group` shared operands stay in place) → `sync::insert_barriers` (fences from shared-memory effects) →
   `schedule::expand` (`Schedule::Uniform { fill, unroll, async_products }` or `PingPong { crew }`) → `layouts::infer`
   (F2 `TileLayout` per register tile, priced relayouts) → `orient_shared` → `emit::emit` (`lower/emit/{mod,address,
   copy,compute,nvptx,amdgpu}.rs`; vendor intrinsics only in the last two). `Lowering { device, schedule, grid,
   swizzle, core }`; `Lowering::uniform(device, unroll, grid)` for plain kernels; `layouts::Swizzle::{None,B32,B64,B128}`.
3. **Target** (`atoms.rs`): matrix cores (`mma: Vec<MmaAtom>`, `group_mma`), `fill: Fill::{CpAsync, Staged}`
   (`Fill::stages` clamps a staged ring to 1–2), `ldmatrix`, `tr_load`, `sync: SyncCaps { warp_barrier, commit_fence,
   priority }`, `occupancy`, `crews`, `sms`, `xcds`. Built by `Target::for_device(&spec)`, `for_arch(arch)`,
   `offline(arch, sms, xcds)` or `card(name)` from `atoms::CARDS` (rtx3060, h100-pcie, h100-sxm, rtx5070, rtx5090,
   mi300x, mi355x, rx9070xt, strix-halo). Kernels never name an instruction.
4. **Costs** (`atoms::Costs`, ns of SM time per unit): `probe::costs(&target)` measures them on the device as
   `[baseline, probe]` pairs stored as `value|probe:<field>|…` lines; `probe::GUESSED` where nothing measured;
   `probe::stored(dir, &target)` reads a store copied from a card; `Costs::trip(fill, stages, unroll)` picks
   `trip`/`trip_rolled`/`trip_one`. `Planner::new(Device)` / `Planner::measured(target, measured)` (both `Option`:
   `None` without an SM count).
5. **Op layer** (`ops/`): `linear`, `attention`, `heads`, `layer_norm`/`rms_norm`/`add_*`, `conv2d` always return a
   Tensor. `ops/shape.rs` gates kernel vs graph (`Plan::Kernel(ranked)` / `Plan::Graph(Fallback::{Target, Dtype,
   Symbolic, Shape, Config})`); `ops/config.rs` ranks a lattice (`Problem::admit`, `Attention::admit` →
   `Result<(ns, regs), Rejected>`); `ops::Tuner` admits the shortlist (compiles, drops spills) and times it once per
   device × shape (`tune.rs`).
6. **Launch** (`launch.rs`): `graph_launch[_all]` wraps the body in a custom kernel, lowered bodies memoized.
7. **Interpreter** (`interp.rs`): f64, rounded to each element type; a pipeline runs serially, `Sync` is a no-op.
   Catches numerics, indexing (`Error::OutOfBounds`), undefined scalar ops (`Error::Division`); never races or fences.

## Writing a kernel

Smallest real kernel: `kernels/rows.rs::norm` (one warp per row, no matrix core). Pipeline + `mma`: `kernels/gemm.rs`.
Masks, `select_if`, `each_warp`, `smem_view_as`, `load_scalar`, `where_`, `mul_add`: `kernels/attention.rs`. Row-mapped
gathers (`Kernel::gather`): `kernels/conv.rs`. Builder call table: `first-kernel.md`. Full wiring: [new-kernel.md](new-kernel.md).

Rules the code enforces:
- Split `XSpec` (shapes, options, `cfg`) / `XCfg::lowering(&self, &Device) -> Lowering` / `pub fn x<T: Elem>(&XSpec)
  -> Program`, dispatched on bf16/f16 by `ops::typed!`. 16-bit operands only; f32 stays on the graph.
- Effects are not declared by hand: async stages inside a `pipeline` belong to the template, `Sync` copies and shared
  stores are fenced by `sync::insert_barriers`; `k.barrier()` is the only hand-placed sync. A block barrier under a
  branch on the warp is `lower::Error::DivergentBarrier`.
- `pin` and `mma_oriented` are for probes/tests; layouts come from inference.
- The warp grid takes every warp (`prog.warps == grid.count()`) unless the extra warps live in `each_warp` bodies.

## Checking correctness

Tests live in `tk/src/test/unit/<topic>.rs` (never in-place; `test_case` tables, `proptest` shapes). Shared helpers in
`mod.rs`: `device_target()`, `device_planner()`, `cp_async_target()`, `group_target()`, `tuned_on_device`, `measure`,
`first_eight`, `costs()` (GUESSED, `step_group` 0), `gfx1151_costs()`, `declared`, `planner`, `rounded`, `decode`,
`assert_rel_close`. Targets: `atoms::sm86()`, `targets::hopper()`, `targets::amd(AmdArch::Gfx1201, 64)`.

| Level | Example |
|---|---|
| Program vs f64 reference | `rows::norm_interpreter_matches_the_reference` (proptest), `attention::attention_matches_a_direct_softmax` |
| Every family, every target, host | `targets::every_family_lowers_keeps_its_numbers_and_compiles` (interp before/after lowering, idempotent barriers, no crop at an offset, per-target facts, clang/ptxas); `targets::each` collects every failure |
| Every candidate lowers | `targets::every_candidate_lowers` |
| Kernel vs interpreter on device | `rows::norms_match_on_device`, `device::gemm_matches_the_interpreter`, `heads::kernel_matches_the_program` |
| Every admitted config on device | `device::every_gemm_candidate_matches_the_interpreter` (via `tuned_on_device`) |
| Op vs graph fallback | `ops::linear_matches_the_graph` (`untuned` = `tune::scoped(false, …)`, `assert_kernel(&y, "gemm")`) |
| Plan per shape (host) | `ops_plan::linear_plans` (`want: Result<(), Fallback>`) |
| Offline card planning | `offline::a_planner_plans_for_a_card_without_it`, `offline::costs_read_from_a_copied_store` |
| Sync pass | `sync::shipped_kernels_need_no_barriers`, `sync::a_block_barrier_under_a_branch_on_the_warp_is_refused` |

## Debugging

**Planner log.** `RUST_LOG=svod_tk::plan=debug` (`ops::PLAN`): `graph op why` (the silent fallback),
`ranking problem admitted rejected`, the first 16 `ranked rank ns regs cfg`, on a store miss `compiled cfg verdict
registers scratch`, then each stage's configs as Debug text (the store's lines hold indices only): `first cfg stored`,
`admitted configs stored` (`stored=false`: nothing compiled, the untested shortlist), `choice timed=[(cfg, ns)…]`,
`twins admitted compiled`, then `tuned op shape pick` / `attention tuned`; a replay adds `replayed lookup`. `RUST_LOG=svod_tk=debug` adds `tuner's first
round`/`leaders`/`rounds`/`twins` and `probe pair timed`/`resolved`. Device tests install the subscriber (`traced()` inside `device_target()`); `tk_dump` installs its own.

**Dumps.** `TK_DUMP_LIST` (`lower/emit/mod.rs`) prints `[i] id=N "Op" dtype <- [src ids]`; the tuner lowers every
candidate and `launch.rs` memoizes, so pin one config (`SVOD_TK_TUNE=0`, one test, or `tk_dump --cfg N|tuned`).
`SVOD_DUMP_{NVPTX,AMD}_IR` and `SVOD_DUMP_AMD_ASM` write on an object-cache hit too, one file per IR
(`<arch>_<kernel name>_<ir digest>.{ll,s}`): every config of `gemm` the tuner compiles gets its own. Host IR for
any target: `lower::render(prog, &lowering)` or `lower::standalone(prog, &lowering)?.text(&target)`.

| Message / symptom | Source and meaning |
|---|---|
| `layout inference: …` | `lower::Error::Layout` (`layouts::Error`): no atom for the dtype pair, a tile that does not tile the warp grid, a carried value that changes layout in a loop |
| `N warps over a grid of M: …` | `lower::Error::Grid` (`targets::a_grid_takes_every_warp_or_whole_grids_run_alone`) |
| `<what> is not lowered yet` | `lower::Error::Unsupported`: cross-warp reductions, a cross-thread transpose, column-major chunk fills, odd warpgroup operands |
| `… is a block barrier under a branch on the warp, which some warps skip` | `lower::Error::DivergentBarrier` (sync pass or a relayout through shared memory) |
| `program assembly: …` / `rendering <name>: …` | `svod_ir::Error` from `UOp::linear_program` / the LLVM renderer; rerun with `SVOD_SPEC_DEBUG=1 TK_DUMP_LIST=1` |
| `element I of NAME is accessed, which holds N` / `scalar A Div B is undefined` (also `Rem`) | `interp::Error::{OutOfBounds, Division}`; also `UnboundVar`, `ParamSize` |
| `Rejected::{Lowers, Shallow, Divides, Smem, Layout, Registers}` | why `admit` left a lattice config out (`tk_dump` counts them, `plan.txt` lists each) |
| op silently runs graph kernels (`E_…`, `r_…` in a profile) | `Plan::Graph(Fallback::…)`: the `graph` line of the planner log says which |
| RDNA4 reads past a view's bounds | transposing loads read a zero run there (`emit::a_transposed_element_past_the_bounds_reads_zero`) |

**Device gate failure**: `check_on_device` prints `<family>: output i max abs diff X` and panics on the first element past
`2e-2·max(|w|,1)`; rerun the family's own device test with `TK_DUMP_LIST=1`. `register_staged_families_match_on_cuda`
runs the AMD data path on NVIDIA. A target counts as measured only once its gate passes.

## Offline: plan and ISA for a card you don't have

`cargo run -p svod-tk --release --example tk_dump -- (--card NAME | --arch sm_90a|gfx942|… [--sms N] [--xcds N])
--op OP --shape S [--dtype bf16|f16] [--cfg I|tuned] [--top N] [--store DIR] [--out DIR] [--compare FILE [--fun NAME]]`
(`tk/examples/tk_dump.rs`; `--out` defaults to `./tk_dump`). `OP`/`S`: `gemm MxNxK` (`--bias --residual --gated
--act silu|gelu`), `attention BxHxTxD` (`--causal`), `norm RxD` (`--norm rms|layer --residual --bias`), `conv
BxHxWxCINxCOUT` (`--kernel --stride --pad --dilation`, epilogue flags), `heads BxTxHxKVxD` (`--qk-norm --rope`), each
through `ops::shape` and `ops::tuned` as the op runs. Prints the costs used (`GUESSED fixture` unless `--store` holds
that card's `<arch>_<sms>sm` file), the top configs with predicted ns and registers, rejection counts by `Rejected`
reason, then writes `plan.txt`, `kernel.ll`, and on CUDA `kernel.ptx`, `ptxas.txt` (`ptxas -v -warn-spills`),
`kernel.cubin`, `kernel.sass` (`nvdisasm -c`; tools from `PATH` or `/opt/cuda/bin`), on AMD `kernel.s` with its
`; NumVgprs … Occupancy` line, and the **main-loop opcode histogram** (`svod_tk::isa::main_loop`: the innermost
back-edge holding a matrix instruction). A split conv/attention emits its first program only.

`--cfg I` is the ranking's `I`th config (`0` is not the tuner's pick: a guessed gfx1201 4096³ ranking leads with a
spilling 256×256 tile). `--cfg tuned` replays the op's tuning through `TuneStore::replay(--store)`, which never
writes (no line, no `emitted/` memo): a line matches by `kind|op|device|dtype|shape|candidates` (`candidates` digests
the salt and the config list's Debug text, so the ranking must be the card's: its stored probes), any compiler
digest (no device here); among several, this build's programs fingerprint (`Own`), else other builds' (`Built`, or
`Disagree`: first taken). Each key read is printed; a `Missing` key is an error with the shape's other lines counted.
A GEMM reads `linear`'s lines, else the device tests'/GEMM probes' (`gemm`, `ops::DEVICE_TEST`); with
`SVOD_TK_TUNE=0` the `first` lines. `TK_DUMP_LIST` and `SVOD_DUMP_NVPTX_IR` work under it too.

## Comparing with vendor kernels offline

Build the reference side, then `tk_dump … --compare <file> [--fun NAME]` prints both main loops per opcode side by
side (`isa::table`; a `.cubin` through `nvdisasm -c`, a `.co`/`.hsaco` through `llvm-objdump -d --symbolize-operands`,
anything else as SASS/PTX/AMDGCN text; `--fun` is required when the file holds several kernels, the error lists them;
`reference.*` keeps the text). Compare per main-loop step: matrix
instructions, global/shared loads and stores, waits and barriers, address arithmetic; plus registers, spills, LDS/smem,
occupancy. Local toolchains: CUDA 13.4 in `/opt/cuda/bin` (`nvcc`, `ptxas`, `nvdisasm`, `cuobjdump`), ROCm in
`/opt/rocm` (`hipcc`, `roc-obj-ls`/`roc-obj-extract`; `llvm-objdump`, `llvm-readelf`, `llvm-mca` in `/opt/rocm/llvm/bin`).
Reference sources are local checkouts under `submodules/` (untracked; may be absent on a fresh clone).

| Reference | How |
|---|---|
| cuBLASLt | `cuobjdump -lelf /opt/cuda/lib64/libcublasLt.so.13.8.0.4` lists the embedded cubins by arch (1376 `sm_90a`, 1608 `sm_120`, 118 `sm_86`); in an empty dir `cuobjdump -arch sm_90a -xelf all <lib>` extracts them (0.4 s, 396 MB), `grep -l -a -F <kernel> *.cubin` finds the one holding a kernel (names read like `sm90_xmma_gemm_bf16bf16_bf16f32_f32_nt_n_tilesize64x256x64_…`), then `cuobjdump -sass -fun <kernel> <cubin>` and `-res-usage` (registers, shared). Which kernel cuBLAS picks for a shape needs the card (`nsys` there). `-fun` over the whole 545 MB library visits every cubin (> 5 min) |
| CUTLASS / ThunderKittens / FA3 | one instantiation, compiled only: `nvcc -std=c++20 -O3 --use_fast_math --expt-extended-lambda --expt-relaxed-constexpr -DNDEBUG -arch=sm_90a -DKITTENS_HOPPER -I…/include -I…/prototype -cubin -Xptxas -v x.cu -o x.cubin` (TK `kernels/gemm/bf16_h100`: 1 s, `matmul_template<2,4,8>` 168 regs, a 170-instruction main loop of 8 `HGMMA.64x256x16`; without `--expt-relaxed-constexpr` CUDA 13 refuses `bit_cast`); `--keep` leaves the PTX. FA3: one file such as `submodules/flash-attention/hopper/instantiations/flash_fwd_hdim128_bf16_sm90.cu` (not built here; the whole extension takes ~1 h) |
| HipKittens / CK | `hipcc -std=c++20 -O3 --offload-arch=gfx942 --offload-device-only -S -Rpass-analysis=kernel-resource-usage -I<hk>/include -I<hk>/prototype -I/opt/rocm/include -I/opt/rocm/include/hip $(python3 -m pybind11 --includes) …/256_256_64_16.cpp -o hk.s` (2 s; remarks give VGPR/AGPR/SGPR/LDS/scratch/occupancy; `-save-temps` keeps every stage); `-c --no-gpu-bundle-output` instead of `-S` gives the ELF `.co` (without it, an offload bundle `llvm-objdump` cannot read). CK: one `example/ck_tile/01_fmha` instance the same way |
| hipBLASLt / AITER | prebuilt `.co` (hipBLASLt's are not installed here: take them from a ROCm wheel or image; AITER ships `hsa/gfx942/*.co`): `llvm-objdump -d --mcpu=gfx942`, `llvm-readelf --notes` for `.vgpr_count`/`.group_segment_fixed_size` |

What a static comparison cannot tell: stall reasons, achieved overlap of copies and products, L2/MALL hit rates and
bank conflicts as they happen, clocks and power. It answers "what does each step issue" (an extra `s_waitcnt`/`BAR.SYNC`,
a spill, address math per `LDGSTS`, an MMA count off the tile/atom ratio); timing needs the card (Nsight Compute,
rocprof, or tk's probes). `llvm-mca` on AMDGPU is a one-wave, no-memory model and prices RDNA WMMA as one VALU cycle.

## Tune store

`$SVOD_TK_TUNE_DIR`, else `$XDG_CACHE_HOME/svod/tk_tune`, else `~/.cache/svod/tk_tune`; one file per device
`<arch>_<sms>sm-v<crate version>.txt` (`sm_86_28sm-v0.2.0.txt`) under a `format 5` line (`tune::FORMAT`); lines
`choice|…` (candidate index + ns pairs, fastest first), `admitted|…`/`first|…` (ranked indices), `value|…` (one number;
probes in femtoseconds) keyed by op, device, dtype, shape, candidate digest, compiler digest and the
fingerprint of the **emitted kernels** (lowered program + LLVM text), so only a change in what a kernel emits re-measures.
The emitted fingerprints are memoized per build in `emitted/<build>.txt` (dropped after 7 days unused). A new FORMAT
orphans every file: the next run re-probes and re-tunes every shape (budget the cold run). Read the probes without
re-running: `grep '^value|probe:' <file>`; indices point into the ranked list: the planner log names their configs,
`tk_dump --cfg tuned` follows them offline.

## Measuring

The `#[ignore]` probes print and never assert; one at a time on an idle GPU:
`-- --ignored --nocapture --test-threads=1`.

| Probe test | Prints |
|---|---|
| `device::gemm_throughput_probe` | 4096³ bf16 TFLOP/s per config, the tuner's pick marked |
| `device::gemm_candidates_probe`, `device::qwen3_gemm_candidates_probe` | every admitted candidate on Nemotron / Qwen3 shapes |
| `attention::{attention_throughput_probe, attention_candidates_probe, decode_throughput_probe}` | FA d64/d128 ± causal ± bias TFLOP/s; decoder step µs |
| `conv::conv_throughput_probe`, `rows::norm_bandwidth_probe` | YOLO conv classes vs graph µs; norm GB/s |
| `attention::key_warp_resources_probe` | registers/LDS/spill per key-warp config |
| `device::first_execution_probe` (cp.async), `parts::mma_peak_probe` (CUDA) | build/lower/prepare cost; `mma.sync` ceiling |
| `probe::probe_signals_are_strong` | every probe pair's signal (run alone) |
| `nearest::knn_probe` | k-nearest lists: the tuned pick's Gpairs/s per `KNN_ROWS`/`KNN_DEPTHS`/`KNN_KS` (f16, NegDot and SquaredL2) and its share of k=1's (register lists to 64, queued into the outputs and score-matrix lists past 32); `KNN_CANDIDATES` for a search instead of a self-join, `KNN_METRICS` one metric, `KNN_GRAPH`/`KNN_GRAPH_K` the graph fallback beside, `KNN_ALL` times every admitted config |
| `nearest::list_time_probe` | list configs untuned, alternating runs (`LIST_CFGS` = `bq:bkv:stages:depth:width[:queue]`;…, `LIST_BK`, `LIST_SPLITS`, `LIST_KS`, `LIST_ROWS`, `LIST_CANDIDATES`, `LIST_DEPTH`, `LIST_RUNS`): the A/B of a list kernel change, split counts forced |
| `nearest::list_ir_dump` | the LLVM IR of those configs into `LIST_IR` (`LIST_TARGET` sm_86/gfx1201), for `ptxas -v` register counts |
| `nearest::live::live_probe` | the op over rows bound to a variable against the op at the bound count (`LIVE_ROWS`, `LIVE_DEPTHS`, `LIVE_KS`): the concrete pick, the live op under that config and under its own pick, rows bound at twice the count run at it |
| `nearest::tuning_probe` | cold tuning wall time, the pick, predicted vs measured per timed config (nearest at `TUNE_KS`, `TUNE_GEMMS`, `TUNE_ATTN`); run on a store of the probes' lines alone |

Ad hoc timing in a test: `super::measure(candidates)` (compile + time, ns per candidate on scratch buffers). Models:
`tk3_pr.md` Table B gives each row's command and line (e.g. `--example nemotron_diarize -- <wav> --dtype bf16 --profile`;
tk kernels appear as `gemm`, `flash_attention`, `heads`, `layer_norm`/`rms_norm`, `conv`).

## Performance-regression protocol

For a change under `ops/config.rs`, `probe.rs`, `tune.rs`, `schedule.rs`, `layouts.rs`, `layout/`, `atoms.rs`, `lower/`:

1. **What changed in the code emitted**: `cargo run -p svod-tk --release --example tk_digest -- --out <dir>` in a
   worktree at the base and at the change (≈45 s each: ~600 kernels × ten targets, `sweep::digested` under the declared
   costs: families, shortlists, attention/conv/norm/heads/probe variants; `--targets sm_86,gfx942`, `--full` keeps the
   `.ll`), then `tk_digest --diff <a> <b>`: kernels changed/removed/added per target and whether `rankings.txt`
   moved; exit 1 on any. Byte-identical output needs no device run; a refactor meant to be neutral proves it this way.
2. **Host**: `cargo fmt`, `cargo clippy --workspace --all-targets`, `SVOD_DEVICE=CPU cargo test -p svod-tk --release
   --lib`, then the workspace: `cargo test --workspace --no-run` and the `svod-codegen`, `svod-tensor`, `svod-model` lib
   tests (model builds tk configs, codegen renders tk kernels; two regressions passed every tk suite and broke them).
3. **Device suite per vendor**, from a detached worktree at the commit (never a checkout others edit mid-run), one
   GPU job per machine: fresh `SVOD_TK_TUNE_DIR`, `device_costs_are_physical` alone first (the suite's parallel tests
   inflate the probes and the store keeps the first measurement), then the suite, each with `SVOD_DEVICE` set.
4. **Probes and models** of what changed, per `tk3_pr.md` "How to add a column", against that GPU's Table A/B column.
5. **A/B across builds on one box**: build both commits in their own worktrees, alternate them ≥2–3 times
   (A B A B …), and compare kernels with tuning off (`tune::scoped(false, …)` / `SVOD_TK_TUNE=0`): a re-tune can pick
   another config from a near-tie (on gfx1201 the 4096³ best tile's standing drifted 4th–10th across processes), which reads as a
   regression of the change. Compare picks separately, through the planner log.

## Key files

| File | Purpose |
|------|---------|
| `tk/src/ir.rs`, `build.rs` | statement tree, `Kernel` builder, `Sc` scalars, `CopyMode::{Sync,Async,Staged}`, `SmemRef`, `CmpOp` |
| `tk/src/kernels/{gemm,attention,conv,heads,rows,mod}.rs` | the kernels; `Batch`, `Act`, `bound`, `batch_offset` |
| `tk/src/lower/{mod,sync}.rs`, `lower/emit/*.rs` | `lower`, `standalone`, `render`, errors; barrier pass; emitter (`TK_DUMP_LIST`) |
| `tk/src/schedule.rs`, `layouts.rs`, `layout/`, `atoms.rs` | templates; inference, `Swizzle`; F2 algebra; `Target`, `CARDS`, `Costs` |
| `tk/src/ops/{mod,shape,config,linear,attention,heads,norm,conv}.rs` | op layer, `PLAN`, plan gate, lattice, `Planner`, `Rejected`, `Tuner` |
| `tk/src/tune.rs`, `probe.rs`, `launch.rs`, `interp.rs` | store, `scoped`, timing; probes, `stored`, `GUESSED`; launch memo; interpreter |
| `tk/src/test/unit/{mod,targets,device,offline,ops,ops_plan,probe,tune,replay,isa,sweep,emit,sync}.rs` | gate, sweep, device tests, probes, store replay, ISA reader |
| `tk/src/isa.rs`, `tk/src/sweep.rs` (`#[doc(hidden)]`) | main loop, functions, side-by-side table; the sweep's targets, families, digest set |
| `tk/examples/tk_dump.rs`, `tk_digest.rs` | offline plan + IR + ISA + `--cfg tuned` + `--compare`; render digests + `--diff` |
