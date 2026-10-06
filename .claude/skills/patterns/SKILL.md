---
name: patterns
description: Svod `patterns!`/`cached_patterns!` DSL and `graph_rewrite` engine reference (ir/src/pattern, ir/src/rewrite, macros/src/patterns). Use when writing or fixing a rewrite rule, choosing between graph_rewrite / graph_rewrite_bottom_up / graph_rewrite_walk, combining matchers with `+`/`with_context`, or diagnosing a rule that never fires or loops.
---

# Svod pattern DSL and rewrite engine

Long-form reference with real rules from the codebase: `website/docs/architecture/optimizations/pattern-system.md`.
Source of truth: `macros/src/patterns/{parser,codegen}.rs`, `ir/src/pattern/{mod,simplified,helpers}.rs`, `ir/src/rewrite/engine.rs`.

## Quick start

```rust
use svod_schedule::{patterns, cached_patterns, graph_rewrite};   // re-exported from svod_macros / svod_ir
use svod_ir::prelude::*;

// fresh matcher, context-free (TypedPatternMatcher = SimplifiedPatternMatcher<()>)
let pm = patterns! {
    Add[x, @zero] => x,                                   // commutative, bare binding returns a clone
    Mul(x, y) if x.dtype().is_int() => x.try_mul(y).ok(),  // Option: None declines
    Cast { src: Cast { src: x, .. }, dtype } if x.dtype() == *dtype => x.clone(),
};
let out = graph_rewrite(&pm, root, &mut ());

// stateful, built once into a LazyLock and returned as &'static
pub fn my_pass() -> &'static TypedPatternMatcher<MyCtx> {
    cached_patterns! {
        @context MyCtx;                                   // closure receives `ctx: &mut MyCtx`
        r @ Range { axis_type: AxisType::Unroll, .. } if ctx.seen(r) => expand(ctx, r),
    }
}
```

The generated code refers to `svod_ir::{Op, ops, op::{alu, pattern_derived, OpMask}, pattern::helpers}`, so the crate must depend on `svod-ir` under the name `svod_ir`.

```bash
cargo test -p svod-ir pattern                         # DSL + early-reject tests: ir/src/test/unit/pattern/
cargo test -p svod-schedule symbolic                  # symbolic rules: schedule/src/test/
cargo test -p svod-schedule --features z3,proptest    # Z3-checked rewrites (schedule/src/z3/)
```

## Left-hand side

| Form | Meaning |
|------|---------|
| `Add(x, y)` | ALU op by kind, positional, ordered. Names resolve through `svod_ir::op::alu`: `UnaryOp`, `BinaryOp` (`Add Mul Sub FloorMod CMod Max Pow FloorDiv CDiv Fdiv Lt Le Eq Ne Gt Ge And Or Xor Shl Shr Threefry`), `TernaryOp` (`Where`, `MulAcc`). Wrong name or arity is a compile error. |
| `Add[x, y]` | Commutative: both orders tried; guard and body are emitted once, retried per ordering. Nested `[..]` multiply. |
| `Cast { src: x, dtype }` | Struct op by field (`ir/src/op.rs` field names). A field is a child pattern when it is `_`, `@..`, a snake_case name, or an identifier applied to `(..)`/`[..]`/`{..}`/`@`; anything else is a verbatim Rust pattern (`axis_type: AxisType::Upcast`, `index: 2`, `reduce_op: op @ (ReduceOp::Add \| ReduceOp::Max)`). `..` skips the rest. No positional form for struct ops. |
| `Load { alt: None, gate: Some(g), .. }` | `Option<Arc<UOp>>` children match with `Some(pat)` / `None`. |
| `Noop` | unit variant, bare |
| `x` / `_` / `name @ pat` | bind a node / ignore / bind the whole sub-match |
| `c @const(v)` | bind a `CONST` node and its `ConstValue` |
| `c @vconst(vs)` / `c @anyconst(vs)` | `VCONST` lanes / `CONST` or `VCONST`, as `Vec<ConstValue>` |
| `Const(<rust pattern>)` | Rust pattern over the `ConstValue`: `Const(_)`, `Const(ConstValue::Int(0))` |
| `@zero` / `@one` | `CONST` 0 / 1 of any numeric dtype (`helpers::is_zero` also accepts `-0.0`, `false`) |
| repeated name | same node (`Arc::ptr_eq`); a repeated `@const` value name compares values |
| `for op in binary [Add, Mul] { .. }` / `[*]` | one body for several ops of one kind (`unary`/`binary`/`ternary`); `op` is the runtime op value, usable in guard and body. No `(Add \| Mul)(..)` form; mix kinds with separate blocks. |
| `pat if guard => body` | guard sees every binding and `ctx` |

Right-hand side: `Arc<UOp>`, `Option<Arc<UOp>>` (`None` declines) or `RewriteResult` (to `Gate`); `?` works inside a block body; a bare binding returns a clone. A rule that returns the node it was given trips a `debug_assert` in the engine.

Struct field cheat sheet (`ir/src/op.rs`): `Index { buffer, indices }`, `Stage { compute, ranges, opts }`, `Load { index, alt, gate }`, `Store { index, value, gate }`, `Range { end, axis_id, axis_type, deps }`, `End { computation, ranges }`, `Reduce { src, ranges, reduce_op, num_axes }`, `ReduceAxis { src, reduce_op, axes }`, `After { passthrough, deps }`, `Param { shape, arg }`, `Buffer { shape, arg }`, `Stack { sources }`, `VConst { values }`, `Wmma { a, b, c, metadata }`, `Call { body, args, info }`, `GetTuple { src, index }`, `Sink { sources, info }`.

## Matcher type and composition

`SimplifiedPatternMatcher<C>` (`TypedPatternMatcher<C = ()>` alias) is a list of segments, one per `patterns!` block; `rewrite(&self, &Arc<UOp>, &mut C) -> RewriteResult` tries segments in order, skipping those whose root `OpMask` lacks the node's kind, and returns the first non-`NoMatch`. Priority is pure source order; `a + b` appends `b` after `a`.

| API (`ir/src/pattern/simplified.rs`) | Notes |
|------|-------|
| `a + b`, `&a + &b`, `a + &b` | same `C` on both sides |
| `m.with_context::<D>()` | `&self`, only on `SimplifiedPatternMatcher<()>`; re-tags segments so `()` rules run under `D`. Type is inferred from the `+` operand when omitted. A non-`()` matcher cannot be lifted. |
| `m.guarded(\|u\| ..)` | run rules only when the guard accepts the root; stacks with existing guards |
| `m.without_early_reject()` | equivalence hook: must rewrite identically |
| `add(&[OpKey], f)`, `add_rejecting(keys, early_reject, f)`, `add_wildcard(f)` | hand-written `Fn(&Arc<UOp>, &mut C) -> RewriteResult` rules; empty `keys` = wildcard |
| `len, is_empty, wildcard_count, indexed_count, early_rejects(&OpKey)` | diagnostics |
| trait `Matcher<C>` | implement directly for non-DSL matchers (`DemoteFloat` in `schedule/src/late/dtype.rs`) |

Mega-pass pattern (from `rangeify/transforms.rs`):
```rust
let pass = symbolic().with_context::<Ctx>() + pm_reduce_simplify().with_context() + ctx_aware_pass();
```

Helpers (`svod_ir::pattern::helpers`): `is_zero is_one is_neg_one is_nonzero is_vconst is_any_const try_const try_vconst try_any_const_values const_matches`.

## Rewrite engine (`ir/src/rewrite/engine.rs`)

Stack-based port of Tinygrad `RewriteContext.unified_rewrite` (`tinygrad/uop/ops.py`). Per node:

| Stage | What happens |
|-------|--------------|
| 0 PushChildren | `bpm` (if any) applied to a fixpoint on the node *before* descent, seeing ORIGINAL children. `Gate(n)` records `n` as the result and skips the children. |
| 1 ApplyPatterns | children resolved through the replace map (waitlist if not ready); if any changed, the rebuilt node goes back to stage 0. Otherwise `pm` runs, seeing OPTIMIZED children; a `Rewritten` result is pushed at stage 0, so it is fully re-traversed and re-matched (that is the fixpoint). |
| 2 Link | original id → final result |

`RewriteResult::{NoMatch, Rewritten(u), Gate(u)}`; in a `pm` matcher `Gate` counts as `NoMatch`. Memoized by `UOp::id`; `Gate` results are never cached.

| Entry point | Use |
|-------------|-----|
| `graph_rewrite(&pm, root, &mut ctx)` | default; rules see rewritten children (Tinygrad `bottom_up=False`) |
| `graph_rewrite_bottom_up(&bpm, root, &mut ctx)` | rules see original children; `Gate` honoured (Tinygrad `bottom_up=True`). Used by `movement_op_patterns`, `early_rewrites`, kernel cut |
| `graph_rewrite_walk(&bpm, root, &mut ctx)` | one pass, `bpm` once per node, replacement NOT re-traversed (Tinygrad `walk=True`); for replacements that contain their own key (`Buffer → After(Buffer, [Store])`) |
| `graph_rewrite_with_bpm(&pm, &bpm, ..)` | both; no production callers |
| `*_preserve_calls` variants of all of the above | do not enter `CALL`/`FUNCTION` bodies or `PROGRAM` internals (Tinygrad `enter_calls=False`); args are still traversed |

Limits: `REWRITE_STACK_LIMIT = 500_000` stack entries → panic `infinite loop in graph_rewrite (stack too big: ..)`; a `bpm` fixpoint revisiting a node → panic `infinite loop in fixed_point_rewrite`. There is no iteration cap, so every rule must make structural progress (`Neg(Neg(x)) => x` is fine; `Neg(x) => x.neg()` loops).

Pre-built matchers you usually compose rather than rewrite: `svod_schedule::symbolic::{symbolic_simple, symbolic, sym, pm_fold_cast_const, pm_lower_index_dtype}`, `rangeify::patterns::{early_rewrites, movement_op_patterns, apply_rangeify_patterns, buffer_folding, dead_axis_removal, pm_remove_bufferize, split_reduceop_patterns}`, `expand::expander2` (ctx `RangeMap`), `devectorize::{devectorize_patterns, bool_storage_patterns, pm_add_loads, pm_reduce}`, `late::gater::pm_move_gates_from_index`. Where each runs: `website/docs/architecture/codegen/overview.md` pass map.

## What cannot be expressed

| Limitation | Workaround |
|------------|------------|
| negative matching `Add(!Const(_), y)` | guard: `Add(x, y) if !matches!(x.op(), Op::Const(_))` |
| alternatives across kinds, variable-arity chains | separate rules / `for` blocks; write `Add(Add(x, y), z)` explicitly |
| "seen earlier in traversal", consumer counts, cycles | `@context` with manual state, or a pre-pass (`get_consumer_map`) |
| matching through `Option` without `Some` | use `Some(pat)`/`None`; `_` on an Option field is a verbatim wildcard |

## Pitfalls

- Early reject: the op kinds a rule's fixed child positions demand become an `OpMask` checked against `uop.src_ops()` before the body runs (Tinygrad `UPat.early_reject`). It only skips nodes whose children lack a required kind; `Add(Mul(..), _)` on `Add(Const, Mul)` passes the mask and then fails positionally — write `Add[Mul(..), _]` if either side may be the `Mul`.
- Wildcards (`x if ..`, `@anyconst` roots) run for every node, in source position; keep them few and cheap.
- `graph_rewrite` rules that need the *original* nesting (`Index { buffer: Stage { .. } }` before the inner `Stage` was folded) belong in a `graph_rewrite_bottom_up` pass.
- Commutative `[..]` on a non-commutative op silently matches the swapped operands.
- Composition order carries meaning: `symbolic_simple()` runs `propagate_invalid` before `x*0 → 0` on purpose; add new rules where their inputs already exist.

## Debugging a rule

```bash
RUST_LOG=svod_ir::pattern::simplified=trace cargo test -p svod-schedule my_test -- --nocapture
```
logs one `pattern matched` event with `op_key` per successful rewrite — it does NOT list tried or failed rules. To see why a rule misses: dump the tree the pass actually sees (`SVOD_DUMP_STAGE=<label>` for post-opt stages, `RUST_LOG=svod_schedule::optimizer=debug` JSON fields for pre-opt; see `/svod-debug`), check `uop.src_ops()` against the rule's early-reject mask (`matcher.early_rejects(&OpKey::from_op(u.op()))`), and run the rule alone with `graph_rewrite(&patterns!{..}, node, &mut ())` in a unit test under `ir/src/test/unit/pattern/dsl.rs` or `schedule/src/test/`.
