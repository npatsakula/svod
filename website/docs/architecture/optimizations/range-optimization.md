---
sidebar_label: Range & reduce
---

# Range and Reduce Optimization

The rules that decide which loops exist: splitting, merging and narrowing ranges, collapsing reductions to closed forms, inlining or materializing intermediates. They live in `schedule/src/rangeify/{patterns,transforms,kernel}.rs` and run in the rangeify mega-pass, at the kernel cut and in `apply_pre_optimization` (see [Rangeify](../codegen/rangeify.md) for the order). Tinygrad: `schedule/rangeify.py`, `codegen/simplify.py`.

## Range splitting (`pm_split_ranges`)

A `RANGE % c` with `end % c == 0` marks the range; at the `SINK` every marked range is replaced by `outer * c + inner` with `outer = RANGE(end / c)` and `inner = RANGE(c)`, both keeping the axis type and taking the axis ids `axis.child(0)` / `axis.child(1)` (no global id allocation). `Warp` and `Device` ranges are never split; every range an image `STORE` indexes is pinned, because an image address is a coordinate pair, not a flat offset. The substituted graph is simplified with `symbolic + pm_fold_cast_const`, so `inner % c → inner` and `(outer*c + inner) // c → outer` fire immediately.

## Range merging and narrowing (`pm_simplify_ranges`)

`simplify_merge_adjacent` runs on every `END` and `REDUCE` with at least two ranges. For an `END` it tries adjacent pairs; for a `REDUCE` every ordered pair. A pair `(r0, r1)` merges when both have the same axis type, constant ends, and appear in the same `REDUCE`s (consistent scoping): the merged range `R(s0*s1)` replaces `r0` by `R // s1` and `r1` by `R % s1`, the graph is simplified with `symbolic + pm_fold_cast_const + pm_flatten_range`, and the merge is kept only if the `FloorDiv`/`FloorMod` count did not grow (`count_divmod`, memoized per node). A symbolic end is never merged: the divmod count would not change, and the symbolic product would hide the constant axis from every later const-only opt (upcast, unroll, locals, tensor cores).

`mark_gated` collects, from every `INDEX`, the bound each validity clause `range < c` proves for a range; a range used anywhere without a guard is pinned to its own end, and `REDUCE` ranges are protected. At the `SINK` each bounded range is rebuilt with the largest proven bound and the result simplified. Together with `pm_flatten_range` (range lists re-derived from the `RANGE`s reachable through the sources, `Bool`/`Void` backedges kept) this is the whole of stage "simplify ranges".

## Load collapse (`pm_load_collapse`)

`reduce_load_collapse(src, ranges)`, per range: take the nodes in scope of the range (bail on a nested `REDUCE` or `STORE`), replace every external input that is not a constant or `PARAM` by a scalar `PARAM` variable `in{n}` carrying its `vmin`/`vmax` (`UOp::variable`), wrap the body in a synthetic `REDUCE(Add)` over that range, and run `build_reduce_load_collapse_matcher`. If no `RANGE` survives, substitute the variables back. The matcher is `pm_reduce_collapse` plus the `.or_casted()` forms and the `NE` lifting.

The bound rules (`reduce_collapse_inner_patterns`, Tinygrad `simplify.py`):

| Reduce body (over `r ∈ [0, N)`) | Closed form |
|---------------------------------|-------------|
| `WHERE(r < cut, 0, v)` | `clamp(N - cut, 0, N) * v` |
| `WHERE(r < cut, v, 0)` | `clamp(cut, 0, N) * v` |
| `WHERE(r >= lo & r < hi, v, 0)` | two-sided clamp times `v` |
| `WHERE(idx != r, 0, e)`, `WHERE(idx == r, e, 0)` (gather) | `WHERE(0 <= idx < N, e[r := idx], 0)` |

(`min` inside the clamps is spelled `-max(-a, -b)` so the `Max` bounds rule can close boundary cases.) Around them: `pm_reduce_unparented`; the lifting transforms that expose the bounds — `(x + y) < c → x < c - y` and `(x*y) < c → x < ceil(c/y)`, also through a `CAST`, `>=` and `==` likewise, `!=` in the load-collapse variant; the distributive `sum(x + y) → sum(x) + sum(y)`; `x * bool.cast() → WHERE(bool, x, 0)`; `try_param_factor` for a condition that is a range-free `PARAM` clause ANDed with a range clause. The outer `pm_load_collapse` also undoes a lifted `(x + y) < c` when `x` contains a load, so loaded indices never overflow. The same engine with the narrower matcher (no `!=` lifting) is `reduce_collapse`, used by `pm_reduce_simplify` in the mega-pass for `REDUCE(Add)` with `num_axes == 0`.

```text
sum(1 for k in 0..64 if k >= length)   →   max(0, 64 - length)
```

## Reduce unparented and factor hoisting (`pm_reduce_simplify`)

`pm_reduce_unparented`: a reduce range the body does not reference is removed — `Add` multiplies the result by the extent, `Mul` raises it to the extent, `Max` drops the range; `Min` is not matched. `reduce_mul_chain`: in `REDUCE(a * b * .., Add | Max)` the factors that depend on no reduce range move outside (for `Max` only provably non-negative ones), integers only. Both also run in `POST_OPT_SYM` (stage 08) and the `sym` tier.

## Buffer removal (`pm_remove_bufferize`)

`INDEX(STAGE(src, ranges, opts), indices)` is inlined by substituting the stage ranges with the consumer's indices (`substitute_gated`; `CONST` ranges and `Invalid` indices are skipped) unless:

1. `src` is an always-run op (`CONTIGUOUS`, `COPY`, `NOOP`) or the stage is non-removable (a `COPY` consumer, an always-contiguous source, a multi-consumer realize boundary, a custom-kernel input);
2. the compute reads more than three distinct buffers (`AFTER` buffers, global `STAGE`s, `MSTACK`, `PARAM`/`BUFFER`), which would blow up the kernel's argument list;
3. a `REDUCE` inside the compute reads a buffer (`PARAM`, `BUFFER` or `STAGE`) — inlining would re-run the read on every iteration (`argmax(-x)` would load `x` N times instead of once). A reduce over values that touch no buffer is still inlinable.

Two cleanup rules follow the substitution: `STORE(x, x)` → `NOOP`, `END(NOOP)` → `NOOP`.

`buffer_folding`: `STAGE(CONST)`, `INDEX(CONST)`, `COPY(CONST)` and `INDEX(MSTACK(CONST, ..))` fold to the constant; `INDEX(STAGE(compute, ranges), ranges)` with the same ranges is `compute` shrunk to the stage shape, tags merged.

`dead_axis_removal`: a removable `STAGE` (not over `AFTER` or an always-run op, no symbolic end) drops ranges that are `CONST` or unused by the compute, then `RESHAPE`s size-1 dims back in and `EXPAND`s to the original shape. A stage can end with zero ranges; it must still exist, or no `STORE` is produced at the cut.

## Two-stage reductions (`split_reduceop`)

In the earliest rewrite, a tensor-form `REDUCE` whose input/output ratio reaches `SplitReduceOpConfig::split_threshold` (32768) is split: a reduced dimension that is not broadcast (`detect_expanded_dimensions`) and divisible by a divisor in `[8, 256]` (largest first) such that the intermediate output stays under `2^22` elements is reshaped into `[.., divisor, rest, ..]`, reduced over the original axes, materialized with `CONTIGUOUS`, and reduced again over the divisor axis. The first stage then has `divisor` outputs to parallelize over; the second is small.

## Grouped reductions

Not a rangeify rule but the same family: `GROUP`/`GROUPTOP` opts turn part of a `Reduce` axis into `GroupReduce`, and `pm_group_for_reduce` (stage 10) lowers it to a partial `REDUCE` staged in local memory, read back with fresh `Reduce` loops (`axis_id.group_reduce_loop()`) and reduced again. See the [expander page](../codegen/expander.md).
