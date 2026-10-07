---
sidebar_label: Kernel search
---

# Kernel Search: Heuristics, BEAM and Tensor Cores

After `apply_pre_optimization` a kernel is a loop nest of `Weak` and `Reduce` ranges. The optimizer decides how those loops execute — which become grid dimensions, workgroups, warps, vector lanes, unrolled bodies or tensor-core fragments — by applying `Opt`s to a `Scheduler`. Two strategies pick the opts: the hand-coded heuristics (default) and BEAM search. Source: `schedule/src/optimizer/{scheduler,opts,heuristics,beam,tc,renderer,config}.rs`, `ir/src/opt.rs`.

## The scheduler and the action space

`Scheduler::new(ast, renderer)` indexes the kernel's `RANGE`s (extent > 1) sorted by `(axis_type.priority(), axis_id)`; `convert_loop_to_global` turns the `Weak` axes that appear in every `STORE` into `Global` when the renderer `has_local` (GPU), and does nothing on CPU. `apply_opt(scheduler, opt, append)` then rewrites one range per call:

| `OptOps` | Effect | Guards (`opts.rs`) |
|----------|--------|--------------------|
| `UPCAST(axis, n)` | split `n` lanes off a `Global`/`Local`/`Weak` axis as `Upcast` | `n <= renderer.upcast_max`; `n = 0` takes the whole axis |
| `UNROLL(axis, n)` | split `n` iterations off a `Reduce`/`GroupReduce` axis as `Unroll` | `axis` indexes `unrollable_dims()`; `n <= 32` |
| `LOCAL(axis, n)` | split a workgroup dimension off a `Global`/`Weak` axis | `has_local`, no earlier `NOLOCALS` |
| `GROUP(axis, n)` / `GROUPTOP(axis, n)` | inner / outer split of a `Reduce` axis into `GroupReduce` (two-stage reduction through shared memory) | `has_local && has_shared`, fits `shared_max`, not nested in another reduce, **rejected once a TC opt was applied** |
| `THREAD(axis, n)` | CPU core dimension on a globalizable `Global`/`Weak` axis | `has_threads`, no existing `Thread` axis, `n <= global_max[0]` |
| `SWAP(a, b)` | exchange two `Global` axes | both `Global` — so never on CPU, where axes stay `Weak` |
| `PADTO(axis, n)` | pad the axis to a multiple of `n`, masking the tail | constant extent, not `Upcast`/`Unroll`/`Thread`, padding below 4× the work, single-index `INDEX` |
| `NOLOCALS` | set `dont_use_locals`, blocking later `LOCAL`s; gpudims launches global-only | no `Local`/`Warp`/`GroupReduce` axis yet |
| `TC(axis_choice, tc_select, tc_opt, use_tc)` | map a matmul onto a tensor core | must be the first opt; see below |

`get_optimized_ast_with_naming` flattens the range lists and attaches `KernelInfo { name, applied_opts, dont_use_locals }`; the name is `r_`/`E_` plus the extents in range order (`r_8_16_4` in the [worked example](../codegen/worked-example.md)).

## Heuristics (`hand_coded_optimizations`)

`hand_coded_optimizations(&mut scheduler, &HeuristicsConfig)` applies, in this order (`heuristics.rs`):

1. **`try_tensor_cores`** — if `tc_enabled != Disabled`, the renderer has cores and (under `TcOpt::Strict`) exactly one reduce axis: `tc::detect_matmul`, then `apply_with_axis_choice` over the axis choices, then `apply_tc_tiling` — `FixedStep`: `UPCAST` M and N by the first of 5/4/3/2 that divides, `LOCAL` N by 4 or 2; `LaneBudget { accum_max: 128 }` (CUDA sm75/80/89): `tc_warp_tile_growth` then a `LOCAL` of `wave_size / tc.threads`. Returns on success.
2. **`apply_image_upcasts`** — image buffers.
3. **`apply_matvec_fast_path`** — the `SVOD_MV*` matvec configuration (`PADTO`, `UPCAST` of the small axes, best-effort `GROUP`, `LOCAL`, `UPCAST`, `UNROLL`). Returns on success.
4. **`try_grouped_reduction`** — `GROUPTOP(axis, 16)` for an output of at most 2048 elements (240 without locals); otherwise **`try_warp_row_reduction`** (`GROUP` by the wave size plus `UNROLL 4`). If a `GroupReduce` axis now exists the function returns.
5. **`apply_masked_upcasts`** — masked axes of size 2–7 with product ≤ 49.
6. **`apply_heuristic_upcasts`** — while the output has ≥ 1024 elements and the upcast product is below 32, `UPCAST` by 3 or 4, axes ranked by `(num_strides, sum_strides, axis, vector rank)`.
7. **`apply_unroll`** — the reduce axis is fully unrolled when ≤ 32 (a second one too when both ≤ 3), otherwise `UNROLL 4`.
8. **`apply_default_upcast`** — `UPCAST 4` on the last upcastable axis if nothing was upcast or unrolled yet.
9. **`apply_local_dims`** — `LOCAL` sizes `[32, 16, 8, 4, 3, 2]` for axis 0 and `[16, 8, 4, 3, 2]` otherwise, cumulative budget 128, at most three, with a `PADTO` fallback.
10. **`apply_threading`** — CPU only: `THREAD` by `[32, 16, 12, 8, 6, 5, 4, 3, 2]` on a `Weak` axis, keeping at least 131072 elements per thread, with a `PADTO` + `THREAD` fallback.

`HeuristicsConfig::from_env` reads `SVOD_TC` (0 disabled, 2 shape-only, else enabled), `SVOD_TC_OPT`/`TC_OPT`, `SVOD_TC_SELECT`/`TC_SELECT`, `SVOD_MV*`, `SVOD_NOLOCALS`, `SVOD_THREADS`. The grouped-reduction thresholds are constants in `heuristics.rs`; `SVOD_K_VECTORIZE` and `SVOD_NO_OUTPUT_UPCAST` set fields nothing on this path reads.

## BEAM search

`BEAM=N` (N > 0) selects `OptStrategy::Beam { width: N }`. `realize` then routes the kernel through `beam_search_cached_remote(scheduler, config, compiler_identity, behavior_fingerprint, compile_wave, benchmark)` (`beam.rs`); the plain `optimize_kernel_with_config` API has no compile-and-time closure and falls back to heuristics.

The search (`beam_search_remote_staged`):

1. Start with `[(scheduler, Duration::MAX)]`.
2. **Expand**: for every beam member, `generate_actions` tries each of the 193 `BEAM_ACTIONS` (200 with `BEAM_PADTO`): `passes_prefilter` (the axis exists; an action whose amount equals the axis size is skipped when the `0` variant exists), `apply_opt`, `validate_limits` (`upcast_prod / tc_up <= max_upcast`, `local_prod <= max_local`). `NOLOCALS` is appended per member when `enable_nolocals`.
3. **Compile** the candidates in a pool of worker processes; a candidate is dropped there if its linearized op count reaches `max_uops` or compilation exceeds `compile_timeout_secs`.
4. **Filter**: candidates with more than 1000× the wave's fewest `compute_ops` are dropped, then duplicates by binary (or source) key.
5. **Time**: `num_runs` runs each, score = minimum; a run is cut short at 3× the incumbent; the global size is capped at 65536 and the time scaled back.
6. **Keep** the best `beam_width`. Stop when the best time no longer improves by `min_progress_ns` (or is already below it); when it did improve the beam collapses to the single winner for the next wave.
7. **Compare**: compile the search's answer and the heuristics' plan (the seed; `BEAM_SEED=0` drops it) together, time them in one batch and keep the faster, the answer on a tie (tinygrad's `BEAM_COMPARE`). The seed never enters a wave, where it would hold a binary the beam may reach later and win or lose that slot by compile order.

The action list (`BEAM_ACTIONS`): `UPCAST` amounts `[0,2,3,4,5,7]` × axes 0..8 (48), `UNROLL` `[0,4,7]` × 0..5 (15), `LOCAL` `[2,3,4,8,13,16,29]` × 0..6 (42) plus `(0,32)` and `(6,2)`, `GROUPTOP` `[13,16,28,29,32,49,64,256]` × 0..3 (24), `GROUP` `[0,4,8,16]` × 0..3 (12), `TC` (one `tc_opt = 0` action plus nine axis choices at `TC_OPT`), `SWAP` pairs within 0..5 (10), `THREAD` `[2,3,4,5,8,12,16,24,32,64]` × 0..3 (30). `BEAM_PADTO` adds `PADTO(axis, 32)` for axes 0..7.

### Cache

Results persist in a `sled` database at `$SVOD_BEAM_CACHE_DIR/beam_cache`, else `~/.cache/svod/beam_cache` (`dirs::cache_dir()`). The key (`CacheKey`, schema 15) is the structural AST hash plus beam width, device, `renderer.cache_fingerprint()`, the compiler identity, the limits (`max_upcast`, `max_local`, `max_uops`, `num_runs`, `min_progress_ns`, `enable_nolocals`, `compile_timeout_secs`), the behavior fingerprint (`transcendental`, `disable_fast_idiv`), a hash of the action space and the seed's plan, so an answer that may be the seed replays only where the same seed would compete again. The value is the `applied_opts` list; a hit is replayed with `replay_opts`, validated and benchmarked once, and invalidated if that fails. `IGNORE_BEAM_CACHE=1` bypasses it, `clear_cache` empties it.

### Environment

| Variable | Default | Meaning |
|----------|---------|---------|
| `BEAM` | 0 | beam width; 0 = heuristics |
| `BEAM_UPCAST_MAX`, `BEAM_LOCAL_MAX`, `BEAM_UOPS_MAX` | 256, 1024, 3000 | `validate_limits` and the worker op cap |
| `BEAM_RUNS` | 3 | timing runs per candidate |
| `BEAM_MIN_PROGRESS` | 10 (µs, stored as ns) | stopping threshold |
| `BEAM_PADTO` | 0 | add the seven `PADTO` actions |
| `NOLOCALS` / `SVOD_NOLOCALS` | unset | add the `NOLOCALS` action |
| `PARALLEL` | 0 | compile workers (GPU defaults to the thread budget, else 1) |
| `BEAM_TIMEOUT_SEC`, `BEAM_MAX_TASKS_PER_CHILD` | 10, 16 | worker watchdog and recycling |
| `TC`, `TC_OPT` | 1, 3 | BEAM's tensor-core actions (`TC_SELECT` is ignored under BEAM: always `Auto`) |
| `BEAM_DEBUG`, `BEAM_LOG_SURPASS_MAX` | unset | diagnostics |
| `IGNORE_BEAM_CACHE`, `SVOD_BEAM_CACHE_DIR` | unset | cache control |

:::tip[BEAM does not read the heuristics switches]
The heuristic seed inside BEAM uses `HeuristicsConfig::from_env()`, but the search's own TC actions read `TC` and `TC_OPT`, not `SVOD_TC`/`SVOD_TC_OPT`. `SVOD_NOOPT` (any value) selects `OptStrategy::None`: no opts at all, but pre- and post-optimization still run.
:::

## Tensor cores

`renderer.rs` holds the core table per `RendererDevice`; dims are `(N, M, K)`:

| Target | Cores (in → out) | threads |
|--------|------------------|---------|
| CUDA sm75 | 8×16×8 f16→f32, f16→f16 | 32 |
| CUDA sm80 | 8×16×16 f16→f32, bf16→f32, f16→f16; 8×16×8 f16→f32, f16→f16; 8×16×32 i8→i32; optional tf32 8×16×8 | 32 |
| CUDA sm89 | sm80 plus 8×16×32 fp8 e4m3/e5m2→f32 | 32 |
| AMD RDNA3 | 16×16×16 f16→f32, f16→f16, bf16→f32, i8→i32 | 32 |
| AMD RDNA4 | RDNA3 plus bf16→bf16 | 32 |
| AMD CDNA3 | 16×16×32 fp8 e5m2/e4m3; 16×16×16 f16/bf16→f32 | 64 |
| AMD CDNA4 | CDNA3 plus 16×16×128 fp8 | 64 |
| Metal | 8×8×8 f32/f16/bf16 variants | 32 |
| Intel Xe | 8×8×16 f16→f32 | 8 |
| WebGPU, CPU | none | — |

`for_cuda_arch` picks the sm80 profile when the capability has bf16 mma, else sm75, and no cores below sm75.

`tc.rs`: `detect_matmul` finds `REDUCE(Add, MUL(in0, in1), reduce_ranges)`; the ranges only `in0` uses are the M candidates, those only `in1` uses are N, the reduce ranges are K, and every `(M, N, K)` triple is an axis choice (an M/N range that is itself a `Reduce` axis is rejected); `select_tensor_core` matches input and output scalar dtypes (an fp8 input without a native core falls back to the f16 core); `apply_with_axis_choice` loops over axis choices × cores within a budget of 64 trials. A core is applied by splitting the axes: one `Warp` range of extent `tc.threads`, an `Upcast` axis of size 2 per `TcOpt::Upcast` entry, each `TcOpt::Local` entry taking a digit of the warp index (`warp % 2`, `warp / 2`), K becoming `log2(K)` `Unroll` axes of size 2; leftover N/M stay `Global` and leftover reduce axes wrap the `WMMA` in a `REDUCE`. `TcUsage::ShapeOnly` (`SVOD_TC=2`) performs the splits but emits no `WMMA`.

`TcOpt` levels (`TC_OPT`): **0 Strict** — one reduce axis only, M/N/K must divide; **1 Relaxed** — same divisibility rule inside `tc.rs`; **2 Padded** (the heuristics' default) — `PADTO` a non-divisible axis when the padding adds at most 25%; **3 Unbounded** (BEAM's default) — pad under `PADTO`'s own 4× limit, and let the timing decide whether the padded tile pays. A symbolic axis never uses a tensor core.

## Programmatic configuration

```rust
use svod_schedule::optimizer::{OptStrategy, OptimizerConfig};
use svod_tensor::PrepareConfig;

let config = PrepareConfig::from(
    OptimizerConfig::builder()
        .strategy(OptStrategy::Beam { width: 8 })
        .build(),
);
tensor.realize_with(&config)?;
```

`OptimizerConfig` (`bon` builder) has `strategy`, `beam: BeamConfig`, `heuristics: HeuristicsConfig`, `transcendental` (`TRANSCENDENTAL`, default 1; ≥ 2 forces the polynomial decompositions), `disable_fast_idiv` (`DISABLE_FAST_IDIV`, default **1**: the magic-number division is opt-in with `DISABLE_FAST_IDIV=0`) and `opts_to_apply` (an explicit opt list, also readable from a kernel `SINK`'s `KernelInfo`; it overrides the strategy and an opt that fails to apply is an error). `PrepareConfig` has `optimizer`, `planner_mode`, `disable_schedule_cache`, `device_local_outputs`, `threads`, and the constructors `Default`, `from_env`, `device_local`, `for_cpu_backend`, `for_{amd,metal,cuda}_if_available`.
