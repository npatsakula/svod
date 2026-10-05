---
sidebar_label: Rangeify & kernel cut
---

# Rangeify, the Kernel Cut and Pre-Optimization

Everything on this page runs before the optimizer sees a kernel. Source: `schedule/src/rangeify/` and `apply_pre_optimization` in `schedule/src/optimizer/mod.rs`.

## Rangeify (`rangeify_with_map`)

Input: the tensor graph a `realize()` call produced (movement ops, `REDUCE` in tensor form with `num_axes > 0`, `CONTIGUOUS`, `COPY`, ...). Output: a graph where every loop is an explicit `RANGE`, every materialization is a `STAGE`, and every read is an `INDEX`.

Passes, in order (`rangeify/transforms.rs`):

1. **Multi-device resolution** — `multi_pm` then `lower_allreduce_pm` (both `graph_rewrite_preserve_calls`); `validate_supported_subset` rejects what the backends cannot run.
2. **`add_tags_patterns`** (bottom-up) numbers every taggable node `[i]`. Tags are tensor identity: after the cut, the output map is rebuilt from the tags that survive. `PARAM`, `CONST`, `RANGE`, `END`, `CALL`, movement ops and all-PARAM `MSTACK`/`MSELECT` are not tagged.
3. **`resolve_calls`** substitutes `FUNCTION` bodies for their arguments and folds `GETTUPLE(TUPLE(..), i)`. Precompiled functions and `CALL` arguments stay opaque.
4. **Earliest rewrites** (bottom-up, one matcher): `movement_op_patterns + early_rewrites + split_reduceop_patterns`. `early_rewrites` drops `DETACH`/`CONTIGUOUS_BACKWARD`, merges untagged `RESHAPE` chains, widens integer products under a widening cast, materializes a resized/reordered `COPY` source with `CONTIGUOUS`, removes same-device `COPY`, and folds zero-size tensors to constants. `split_reduceop` is the two-stage reduction split (see [range optimization](../optimizations/range-optimization.md)).
5. **`run_rangeify`** (`rangeify/indexing.rs`):
   - `pm_generate_realize_map` (bottom-up): marks what must become a buffer — `STORE`, `CONTIGUOUS`, `COPY` and their non-contiguous sources, `MSTACK`/`MSELECT` sources, and the inputs of a hand-written kernel `CALL` (pinned non-removable).
   - `assign_ranges`: a root-to-leaf walk. A realized node gets fresh `Weak` ranges per output dimension (`IndexingContext::new_range`; a size-1 dim is `CONST(0)`). Other nodes inherit their consumers' ranges; when consumers disagree, `merge_consumer_ranges` either merges compatible index expressions (the valid parts are OR-ed into `WHERE(valid, idx, Invalid)`) or allocates new ranges and marks the axis for realization. Movement ops map output ranges to input ranges with `apply_movement_op` (a `PERMUTE` permutes them, an `EXPAND` zeroes the broadcast axis, a `PAD` wraps the range in a validity `WHERE`, a `RESHAPE` goes through `apply_reshape_ranges`). `ending_ranges` propagate broadcast decisions backwards so a `REDUCE` feeding a broadcast is realized before it (the layernorm case).
   - `apply_rangeify_patterns` (bottom-up): tensor-form `REDUCE` → loop-form `REDUCE(src, ranges)` with `num_axes = 0`; `PAD` → `WHERE(valid, src, 0)`; shaped `STACK` → a `WHERE` chain on its leading range; every op gets its realized sources wrapped in `STAGE` + `INDEX` (`transform_sources_with_bufferize`); movement ops are then removed. A buffer-like source (`BUFFER`, `PARAM`, `SLICE`, `AFTER`, ...) gets a single row-major `INDEX` when its shape is static (`linearize_static_indices`); image and symbolic shapes keep one index per coordinate.
6. **Mega-pass** — one fixpoint over `symbolic + pm_reduce_simplify + movement_op_patterns + buffer_folding + dead_axis_removal + pm_remove_bufferize`. The groups feed each other: inlining a `STAGE` exposes range arithmetic that `symbolic` folds, which can make a reduce collapsible. The individual rules are on the [range optimization](../optimizations/range-optimization.md) page.
7. **SINK rebuild** from the tagged backward slice: only `STAGE`, `MSTACK`, `CONST`, `PARAM` and `AFTER` nodes carrying an output tag stay as sink sources, in the original output order.
8. **Buffer limit** — if the device reports `max_buffers`, `buffer_limit_patterns` forces elementwise sources into global `STAGE`s so no kernel exceeds the argument limit.

The result for `x.sum(1)` on an `[8, 64]` tensor:

```text
[67] SINK : Scalar(Void)
└── [66] STAGE : Scalar(Float32) shape=[Const(8)]
    ├── [65] CONTIGUOUS : Scalar(Float32) shape=[]
    │   └── [64] REDUCE(Add, num_axes=0, ranges=[27]) : Scalar(Float32) shape=[]
    │       ├── [62] INDEX : Scalar(Float32) shape=[]
    │       │   ├── [11] PARAM(slot=0) : Scalar(Float32) shape=[Const(512)]
    │       │   │   └── [0] CONST(Int(512)) : Scalar(WeakInt) shape=[]
    │       │   └── [55] Add : Scalar(WeakInt) shape=[]
    │       │       ├── [54] Mul : Scalar(WeakInt) shape=[]
    │       │       │   ├── [26] RANGE(U0, Weak) : Scalar(WeakInt) shape=[]
    │       │       │   │   └── [2] CONST(Int(8)) : Scalar(WeakInt) shape=[]
    │       │       │   └── [3] CONST(Int(64)) : Scalar(WeakInt) shape=[]
    │       │       └── [27] RANGE(U1, Reduce) : Scalar(WeakInt) shape=[]
    │       │           └── [3] → (see above)
    │       └── [27] → (see above)
    └── [26] → (see above)
```

`U0`/`U1` are `AxisId::Unrenumbered`: ranges are renumbered per kernel at the cut. The `PERMUTE`/`RESHAPE` of the input are gone — they became the index expression `U0 * 64 + U1`.

## The kernel cut (`try_get_kernel_graph`)

`kernel_graph_pre_cut` first:

- **`pm_add_buffers_patterns`** (bottom-up, `RangeifyBufferContext`): `movement_op_patterns`, then `flatten_bufferize` (a multi-range `STAGE` becomes a single flat range plus a `RESHAPE` back), `late_buffer_slice` (a DISK `STAGE(BITCAST|CONTIGUOUS)` becomes a `SLICE`), and `bufferize_to_store`. The last one allocates a schedule-local `BUFFER` (`new_lunique_buffer`, slot in the high-bit namespace) and rewrites `STAGE(compute, ranges)` into `AFTER(BUFFER, [END(STORE(INDEX(BUFFER, idx), compute), ranges)])`. A `STAGE(AFTER(..))` reuses the underlying buffer; a `Local` `STAGE` is left alone for `pm_add_local_buffers` later. An already-formed kernel `SINK` (one with `KernelInfo`) is gated so the rewrite does not descend into it.
- **`pm_flatten_range`** once over the whole graph (bottom-up): re-derives the range list of every `END`/`REDUCE` from the `RANGE`s reachable through its sources, so the per-kernel pass below does not re-traverse shared subgraphs.

Then **`split_all_stores`** (bottom-up): every `STORE` or `END(STORE)` with no computational range still open becomes a `CALL`. `split_store` runs `local_to_param_patterns + rangeify_codegen_patterns` on the kernel body: global `BUFFER`/`PARAM` → codegen `PARAM(slot)` numbered in match order by `LocalAddBufferContext::param_slot`; `BIND(var, value)` → the variable, with the binding kept as a `CALL` argument; `AFTER`/`MSTACK`/`MSELECT` → their buffer; `RANGE(end=0)` → `CONST(0)`; `Unrenumbered` axis ids → `Renumbered(n)`; `NOOP` → typed zero; `CONTIGUOUS` → its source, hints harvested. The body is wrapped in `SINK` with a default `KernelInfo`; a `COPY`/`SLICE` value stays a direct call body. `Device` ranges are the one exception to "no open range": they are launch lanes and survive the boundary.

Finally **`validate_normal_kernel_devices`** (one device per non-copy kernel) and **`fix_assign`**: when kernel B reads a buffer kernel A writes, A's `AFTER` is appended to B's `AFTER` deps; a cycle is `KernelSplitDependencyCycle`. With `SVOD_SPEC` on, `verify_kernel_graph` checks the result.

## Per-kernel pre-optimization (`apply_pre_optimization`)

Runs on each kernel body before heuristics or BEAM, in both paths (`optimize_kernel_with_config_impl`, `optimize_kernel_beam`, `prepare_scheduler`). With `SVOD_SPEC` on, `type_verify` against `spec_tensor` runs first.

| Step | Matcher | Direction |
|------|---------|-----------|
| movement ops | `movement_op_patterns` | bottom-up |
| load collapse | `pm_load_collapse` | top-down |
| split ranges | `pm_split_ranges + pm_flatten_range` (`SplitRangesContext`) | top-down |
| symbolic | `sym + pm_fold_cast_const + pm_flatten_range` | top-down |
| simplify ranges | `pm_flatten_range + pm_simplify_ranges` (`SimplifyRangesContext`) | top-down |

**`movement_op_patterns`** has three rules: `INDEX(mop(x), idx)` → `INDEX(x, mop⁻¹(idx))` (`transform_movement_through_index`), `AFTER(mop(x) | INDEX(x), deps)` → `mop(AFTER(x, deps))` (`push_op_through_after`), and `END(mop(x), ranges)` → `END(x, ranges)`. `is_movement()` is exactly `RESHAPE`, `PERMUTE`, `EXPAND`, `PAD`, `SHRINK`, `FLIP`. It is applied bottom-up because the inner movement op must be rewritten before its consumer can match.

**`pm_load_collapse`** eliminates a `REDUCE(Add)` whose body is range-independent after symbolic reasoning (`reduce_load_collapse`): nodes outside the reduce scope are replaced by scalar `PARAM` variables (`UOp::variable("in{n}", vmin, vmax)`), the body is wrapped in a synthetic `REDUCE` over the one range, `build_reduce_load_collapse_matcher` runs, and if no `RANGE` survives the substitution is reversed. The bound patterns it uses are on the [range optimization](../optimizations/range-optimization.md) page.

**`pm_split_ranges`** records every `RANGE % const` whose end is divisible by the constant (`Warp` and `Device` ranges excluded; every range an image `STORE` indexes is pinned) and substitutes `r → outer * c + inner` once, at the `SINK`, with axis ids `r.child(0)` / `r.child(1)`. The substituted graph is then simplified with `symbolic + pm_fold_cast_const`.

**`sym`** is the full tier-3 simplifier ([algebraic simplification](../optimizations/algebraic-simplification.md)); `pm_fold_cast_const` folds `CAST(CONST)`; `pm_flatten_range` keeps the range lists honest after ranges disappear.

**`pm_simplify_ranges`** merges adjacent ranges of an `END`/`REDUCE` when the merged form does not increase the `FloorDiv`/`FloorMod` count (`simplify_merge_adjacent`), and narrows a range to the largest bound any `INDEX` gate proves for it (`mark_gated`; a single ungated use pins the original end; `REDUCE` ranges are protected). Both substitutions happen at the `SINK`.

## Hand-off to the optimizer

`Scheduler::new(ast, renderer)` collects the `RANGE`s with extent > 1, sorted by `(axis_type.priority(), axis_id)`; `convert_loop_to_global` turns `Weak` output axes into `Global` on renderers with `has_local` (it is a no-op on CPU, which is why the row axis in the example above stays `Weak`). Then `hand_coded_optimizations` or BEAM applies `Opt`s and `get_optimized_ast_with_naming` emits the kernel `SINK` with `KernelInfo` metadata (name such as `r_8_16_4`, `dont_use_locals`, `opts_to_apply`). `SVOD_NOOPT` skips the heuristics but not this page nor the post-optimization stages. The search itself is described in [kernel search](../optimizations/kernel-search.md).
