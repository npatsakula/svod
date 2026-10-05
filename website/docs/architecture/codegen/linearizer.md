---
sidebar_label: Late rewrites & linearizer
---

# Late Rewrites, the Program Boundary and the Linearizer (stages 19–20 and beyond)

The last post-optimization stages make the graph renderable for one concrete backend: operations the target lacks are decomposed, validity becomes a `gate`, weak dtypes are committed. Then `svod-codegen` adds control-flow edges, numbers the parameters and flattens the DAG into an instruction list. Source: `optimizer/mod.rs`, `late/gater.rs`, `late/dtype.rs`, `optimizer/implicit_barriers.rs`, `linearize/`, `codegen/src/program_pipeline.rs`.

Every matcher below is built from the renderer's capability table (`renderer.supported_ops()`, `supports_dtype`), which is why `optimize_kernel_with_config` refuses a renderer without one (`OptError::MissingRendererCapabilities`).

## 19 — cast float ALU operands

`pm_cast_float_alu`: for `Sin`, `Log2`, `Exp2`, `Sqrt`, `Reciprocal`, cast the operand to the result dtype. The transcendental decompositions expand into dtype-homogeneous polynomials and must not see a mixed-dtype operand.

## 19b — early decompositions

`early_decomposition_patterns(supported_ops)`:

```text
symbolic_simple + pm_fold_cast_const + pm_mod_to_and + divmod_decomposition_patterns
  + pm_threefry_decomp        if !supports(Threefry)
  + pm_max_decomposition      if !supports(Max) && supports(Lt)
  + pm_erf_decomposition      if !supports(Erf)
```

`divmod_decomposition_patterns` (`ir/src/decompositions/mod.rs`) lowers floor division and modulo (`FloorDiv`/`FloorMod`) to the truncating `CDiv`/`CMod` with a sign correction — the form every backend has. `pm_mod_to_and` is here as well as in the late set so power-of-two modulos fold before the truncating lowering sees them.

## 19c — dtype decompositions

`pm_dtype_decomp_commit = pm_dtype_decomps + pm_commit_weak` with a `DTypeDecompCtx`. The first rule only records which of `FP8E4M3`, `FP8E4M3FNUZ`, `FP8E5M2`, `FP8E5M2FNUZ`, `Float16`, `BFloat16`, `Int64`/`UInt64` appear in the graph; the `SINK` rule then rewrites, bottom-up and in dtype order, every recorded dtype the renderer does not support:

| Unsupported | Emulated as | Matcher |
|-------------|-------------|---------|
| `Int64`, `UInt64` | two `Int32`/`UInt32` words: `PARAM`/`BUFFER` doubled in size, `INDEX` tagged with the word, carries and borrows built from `Lt`, a 64-step shift-subtract divider for `CDiv`/`CMod` | `pm_long_decomp` (`devectorize.rs`) |
| FP8 | `Float16` if supported else `Float32`; storage stays the 8-bit unsigned word, `f2f` does the bit-exact conversion (RNE rounding, FNUZ NaN encoding, saturation in `f2f_clamp`) | `pm_float_decomp` |
| `Float16`, `BFloat16` | `Float32` compute, same `f2f` storage conversion | `pm_float_decomp` |

`get_dtype_decomps` returns the same selection as a list for the renderer's compile cache key.

## 19d — late decompositions

`pm_decomp = early_decomposition_patterns + get_late_rewrite_patterns(renderer, disable_fast_idiv) + get_transcendental_patterns(supported_ops, TRANSCENDENTAL >= 2) (+ renderer.decomposition_matcher())`, run to a fixpoint. The late set is capability-gated ([strength reduction](../optimizations/strength-reduction.md) has every rule):

```text
pm_mod_to_and + pm_half_bf16_cast                       always
+ pm_demorgan                                           if supports(Or)
+ pm_mul_to_shl                                         if supports(Shl)
+ pm_div_to_shr                                         if supports(Shr)
  + fast_division_patterns + pm_mod_to_idiv             if supports(Shr) && DISABLE_FAST_IDIV=0
+ pm_neg_from_mul                                       if supports(Neg)
+ pm_comparison_negations                               if supports(Lt) || supports(Eq)
+ pm_fma_decomposition                                  if supports(MulAcc)
  + pm_shl_add_to_mulacc                                if supports(MulAcc) && supports(Shl)
+ pm_fdiv_to_mul                                        if supports(Fdiv)
```

`get_transcendental_patterns` (`ir/src/decompositions/`) replaces `Exp2`, `Log2`, `Sin` with the `xexp2`/`xlog2`/`xsin` polynomial approximations for f16/f32/f64 (other floats compute in f32) and `Sqrt` with `xpow(x, 0.5)`, for each op the renderer lacks, or for all of them when `TRANSCENDENTAL=2`. The `decomposition_matcher` is the optimizer-side copy of the device `Renderer::decompositor()` hook; Metal installs `amd_decomposition_patterns` there (`Exp`, `Log`, `Cos`, `Tan` and binary `Pow` over native `exp2`/`log2`).

In the worked example this stage turns `R1 * 64` into `R1 << 6` and `R0 * 4 + (R1 << 6)` into `MulAcc(R0, 4, R1 << 6)` — the integer FMA built by `pm_shl_add_to_mulacc`.

## 19e — gates, register lanes, float demotion

`pm_move_gates_from_index` (`late/gater.rs`, port of Tinygrad's `gater.py`) finally moves validity out of the index:

| Before | After |
|--------|-------|
| `LOAD(INDEX(buf, WHERE(g, idx, Invalid)))` (no `alt`, no `gate`) | `LOAD { index: INDEX(buf, idx), alt: 0, gate: g }` |
| `STORE(INDEX(buf, WHERE(g, idx, Invalid)), v)` (no `gate`) | `STORE { index: INDEX(buf, idx), value: v, gate: g }` |
| same two forms on `SHRINK` (coalesced groups) | gated `LOAD`/`STORE` on the cleaned `SHRINK` |
| image two-coordinate `INDEX` with one shared condition | one gated access (checked first) |
| `WHERE(g, LOAD{gate: g}, alt)` and the inverted form | the `alt` folds into the load |

`valid_index` requires the literal `Invalid` constant in the third `WHERE` slot. Then `pm_scalarize_register_stack_index_preserve_deps` resolves `INDEX(AFTER(STACK(..), deps), c)` — a lane of a register stack read after some stores — into the selected `LOAD` with the deps re-attached to its address, and `merge_register_read_ends` merges `END`s that close the same ranges under one register `AFTER` (a debug assertion checks no register-stack `INDEX` survives). `demote_unsupported_floats` (`late/dtype.rs`) runs last: on a renderer without `Float64` ALU (Metal, WebGPU) every internal f64 value is computed in f32, while global f64 storage, its loads and their `alt` values keep the wide dtype.

## 20 — final rewrite

```text
pm_final = pm_commit_weak + pm_cast_weak + pm_decomp (+ renderer.extra_matcher()) + pm_split_ends
```

One fixpoint (`assert_target_renderer_boundary` runs first in debug builds: no static multi-index `INDEX`, no residual singleton broadcast, no mixed `STACK`/vector ALU). `pm_split_ends` turns `END(x, [r1, r2, r3])` into `END(END(END(x, r3), r2), r1)`, ranges sorted descending by `(axis_id, axis_type.priority())`; `Void`/`Bool` sources (reduction backedges) are partitioned out and re-attached on the outermost `END`, and the original tag is preserved so later merge steps still find it. `extra_matcher` is the per-backend hook on `svod_device::device::Renderer`; it runs inside the same fixpoint as the decompositions. The CPU and NVPTX renderers install `bool_storage_patterns` again (`cpu_extra_matcher`), AMD installs `amd_non_native_fp8_patterns` (OCP FP8 ALU widened to f32; storage, conversions and MFMA operands untouched).

Then, as separate passes: `pm_remove_invalid` replaces every remaining data-typed `WHERE(c, x, Invalid)` with `WHERE(c, x, 0)` and every `Invalid` `STACK` lane with zero (a debug assertion checks none is left), and `add_implicit_barriers` inserts `BARRIER`s for local memory: a RAW barrier before an `AFTER` on a local buffer whose deps contain an unbarriered local `STORE`, and a WAR barrier at the end of a loop body that stores to a local buffer another load in the same loop reads. `optimize_kernel_with_config_and_final_rewrite` returns the graph captured just before the barriers, for parity tooling. The `KernelInfo` metadata that `graph_rewrite` dropped is re-attached.

## The program boundary (`program_from_sink`)

`svod-codegen` takes over (`program_pipeline.rs`):

1. **`add_control_flow`** (`linearize/mod.rs`): `pm_split_ends` again (idempotent), then `CFGContext::new(sink)` and `pm_add_control_flow` bottom-up. The context computes, for every `END`, which `END`/`SINK` it is nested in — `END x` is nested in `u` when `u` depends on `x` and `u`'s range is in `x`'s dependencies — groups siblings by parent, orders them by how many siblings they depend on, and records an edge from each later sibling's `RANGE` to its predecessor (the previous sibling's `END`, or the parent's `RANGE` for the first one). `pm_add_control_flow` appends the predecessor to the `RANGE`'s sources; `InScopeRangesProperty` then sees the nesting, which is what gives nested ranges a larger `run_count` below. A predecessor that already contains the range panics (`"edge would create cycle"`).
2. **`number_params`** assigns the final `PARAM` slots (`validate_param_slots` rejects an unassigned or duplicated slot).
3. **`verify_final_sink`** against `spec_program` when `SVOD_SPEC` is on; `ProgramInfo::from_sink` reads the ABI.
4. **`pre_isel_matcher` / `isel_matcher`** — the two instruction-selection hooks on `svod_device::device::Renderer`, both bottom-up (`PreIselContext`, `IselContext`). They exist for ISA-level backends; the LLVM and C renderers leave them `None`.
5. `UOp::program(sink, info, None, None, None)` — the `PROGRAM` node whose later sources are the `LINEAR`, `SOURCE` and `ProgramBinary` stages.

## `linearize`

`do_linearize` calls `svod_schedule::linearize(sink)` (`linearize/linearize.rs`, a direct port of Tinygrad's `linearizer.py`), then `line_rewrite_cleanups`, then `verify_linear_list` against `spec_program`.

The sort key of every node is `(run_count, priority, extra, tuplize rank)`:

| Op | Priority |
|----|----------|
| `PARAM` | −20, ties broken by slot (`extra`) |
| `BUFFER` (global, register) | −18 |
| `BUFFER` (`AddrSpace::Local`) | −17 |
| `END` | −5 |
| `LOAD` | −1 |
| everything else (`CONST`, ALU, `SPECIAL`, …) | 0 |
| `STORE` | +1 |
| `RANGE` | +5 |

`run_count = prod(vmax + 1)` over the node's in-scope ranges (a symbolic extent counts as 1), so code outside a loop sorts before the loop body. The tuplize rank is Tinygrad's `(op, arg, dtype, *src.tuplize)` key computed iteratively over the toposort (`TuplizeKeys`), which makes the order total and deterministic. The linear list is then produced by a max-heap toposort from the `SINK` on these ranks, reversed at the end: a node is emitted once all its consumers have been, so definitions come first, `LOAD`s before their uses, `STORE`s after the computation, and each `RANGE` opens right before its body.

`SVOD_DUMP_LINEAR=<dir>` writes the toposort with in-scope range ids (`tree_<id>.txt`) and the final list (`linear_<id>.txt`).

## `line_rewrite_cleanups`

`line_rewrite` walks the instruction list once; each entry may be replaced by several. The only cleanup is `linearize_cleanup_pattern`: a `STORE` with a `Bool` gate whose address is an `INDEX`/`SHRINK` (possibly behind a `CAST`) becomes `IF(gate)`, the ungated `STORE`, `ENDIF`. Both `IF` and `ENDIF` exist only in the list, never in the graph (`"if not allowed in graph"`), which is why `spec_program` is checked on the list as well as on the sink. Backends that can predicate a store (LLVM, CUDA, Metal) render the triple as a conditional store.

The list is the `LINEAR` stage of the `PROGRAM`; `do_render` hands it to `Renderer::render`, and `do_compile` produces the binary. The [worked example](./worked-example.md) shows the LLVM IR the CPU renderer emits for a row-sum kernel.
