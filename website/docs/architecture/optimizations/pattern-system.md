---
sidebar_label: Pattern engine
sidebar_position: 0
---

# The Pattern Engine

Nearly every pass in Svod is a `graph_rewrite` over a matcher built with the `patterns!` macro: the rangeify stages, the symbolic simplifier, the expander, the devectorizer, the decompositions, the gate movement. The exceptions are a few plain graph walks (`memory_coalescing`, `merge_register_read_ends`, `linearize`) and the line rewrite that runs over the linear instruction list. This page is the reference for the macro and the engine. Source: `macros/src/patterns/`, `ir/src/pattern/`, `ir/src/rewrite/engine.rs`; the Tinygrad counterparts are `UPat` and `graph_rewrite` in `tinygrad/uop/ops.py`.

## The `patterns!` DSL

A block is a list of `pattern [if guard] => body` rules, optionally preceded by `@context Type;`. The left-hand side is Rust pattern syntax extended with what a Rust pattern cannot say across an `Arc<UOp>` edge. Real rules from `schedule/src/symbolic/patterns.rs`:

```rust
// constant folding over thirteen binary ops, one rule body
for op in binary [Add, Mul, Sub, FloorMod, Max, Pow, FloorDiv, Fdiv, And, Or, Xor, Shl, Shr] {
    op(a @const(a_val), _b @const(b_val))
      => eval_binary_op(op, a_val, b_val).and_then(|r| folded_const(a.dtype(), r)),
},

// commutative identity with a guard; `name @ @zero` binds the constant node too
Add[x, zero @ @zero]
    if !x.dtype().is_float()
        || matches!(zero.op(), Op::Const(ConstValueHash(ConstValue::Float(v))) if v.is_sign_negative())
    => x.clone(),
Mul[x, @one] => x.clone(),

// a repeated name means the same node (Arc::ptr_eq), here across a struct field
original @ FloorDiv(x, x) => exact_integer_rewrite(original, 1.into_uop(x.dtype())),
original @ FloorMod(range @ Range { end, .. }, end) => exact_integer_rewrite(original, range.clone()),

// struct ops match by field; `..` skips the rest
Cast { src: Cast { src: x, dtype: intermediate }, dtype: outer }
    if x.dtype() == *outer && can_safe_cast(outer, intermediate)
    => x.clone(),
```

A stateful matcher, from `schedule/src/expand.rs`:

```rust
crate::cached_patterns! {
    @context RangeMap;
    reduce @ Reduce { .. } => expand_reduce(reduce),
    range @ Range { end: _, axis_id, axis_type }
        if matches!(axis_type, AxisType::Upcast | AxisType::Unroll) && ctx.contains_key(axis_id)
        => expand_range(ctx, range),
    Wmma { a, b, c, metadata } if metadata.upcast_axes.is_some() => expand_wmma(ctx, a, b, c, metadata),
}
```

| Form | Meaning |
|------|---------|
| `Add(x, y)` | ALU op by kind, positional, ordered. Names resolve through `svod_ir::op::alu`, so a wrong name or arity is a compile error. `UnaryOp`, `BinaryOp` (`Add, Mul, Sub, FloorMod, CMod, Max, Pow, FloorDiv, CDiv, Fdiv, Lt, Le, Eq, Ne, Gt, Ge, And, Or, Xor, Shl, Shr, Threefry`), `TernaryOp` (`Where`, `MulAcc`). |
| `Add[x, y]` | Commutative: exactly two children, both orders tried; the guard and body are emitted once and retried per ordering. |
| `Cast { src: x, dtype }` | Struct op by field. A field is a child pattern when it is `_`, `@..`, a snake_case name, or an identifier applied to `(..)`/`[..]`/`{..}`/`@`; anything else is a verbatim Rust pattern (`axis_type: AxisType::Upcast`, `index: 2`). `Some(pat)`/`None` match `Option<Arc<UOp>>` children (`Load { alt: None, gate: Some(g), .. }`). Unit ops are bare (`Noop`). |
| `x` / `_` / `name @ pattern` | bind a node / ignore / bind the whole sub-match |
| `c @const(v)` | bind a `CONST` node and its `ConstValue` |
| `c @vconst(vs)` / `c @anyconst(vs)` | `VCONST` lanes / `CONST` or `VCONST` as `Vec<ConstValue>` |
| `Const(<rust pattern>)` | a Rust pattern over the `ConstValue` |
| `@zero` / `@one` | scalar `CONST` 0 / 1 of any numeric dtype (`is_zero` also matches `-0.0` and `false`) |
| repeated name | the same node (`Arc::ptr_eq`); a repeated `@const` value name compares values |
| `for op in binary [A, B]` / `[*]` | one rule body for several ops (or all of a kind); `op` is the runtime op value, usable in guard and body |
| `pat if guard => body` | the guard sees every binding and `ctx` |
| `=> body` | `Arc<UOp>`, `Option<Arc<UOp>>` (`None` declines) or `RewriteResult`; `?` works, a bare binding returns a clone |
| `@context Type;` | first item; the closure receives `ctx: &mut Type` |

`cached_patterns!` has the same grammar and returns `&'static TypedPatternMatcher<C>` from a `LazyLock`; `patterns!` builds a fresh matcher. Both are re-exported from `svod_schedule`.

### What the macro generates

`Op` carries `#[op_enum]`/`PatternEnum`, which generates `svod_ir::op::pattern_derived::OpKey` — one dense index per op kind, with one slot per sub-op for the grouped `Unary`/`Binary`/`Ternary` — and `OpMask`. A `patterns!` block compiles into **one closure** registered with `SimplifiedPatternMatcher::add_block`, plus a constant table of `(root mask, early-reject mask)` per rule:

- Consecutive rules with one constant root kind share a `match __key { __KEY_Add => { .. } .. }`; within an arm the rules keep source order.
- Rules without a constant root — wildcards (`x if ..`), `for` blocks, `@anyconst` roots — are emitted as sequential steps *between* those `match`es, so priority is pure source order, not "indexed first, wildcards last".
- Each rule starts with an early-reject test: the op kinds its fixed child positions require are a bit mask checked against the root's `src_ops` (Tinygrad's `UPat.early_reject`).
- Commutative sites become lazily chained candidate iterators; nested commutative nodes become nested loops; the body is retried per ordering.
- A `for` block is compiled once per rule body; the op variable is bound from the root at runtime.

`SimplifiedPatternMatcher<C>` (`TypedPatternMatcher<C = ()>` is the alias) is a list of segments, one per block, each with a root `OpMask` and the closure. `rewrite(node, ctx)` scans the segments, skips those whose mask lacks the node's kind, and returns the first non-`NoMatch`. `a + b` appends `b`'s segments after `a`'s, so the left operand's rules win. `with_context::<D>()` lifts a `TypedPatternMatcher<()>` into a `D`-context matcher (it takes `&self`); hand-written closures go in with `add`, `add_rejecting`, `add_wildcard`. `Matcher<C>` is the trait (`fn rewrite(&self, &Arc<UOp>, &mut C) -> RewriteResult`); `DemoteFloat` in `late/dtype.rs` implements it directly.

## The rewrite engine

`ir/src/rewrite/engine.rs` is a stack-based port of Tinygrad's `unified_rewrite`. Each node goes through three stages:

| Stage | What happens |
|-------|--------------|
| 0 — PushChildren | If a `bpm` matcher is given, apply it to this node to a fixpoint *before* descending (patterns see the original children). `Gate(node)` records a replacement and skips the children. Then push the children, then a stage-1 entry for this node. |
| 1 — ApplyPatterns | Resolve the children through the replacement map (waitlist if one is not ready). If a child changed, rebuild the node and send the rebuilt node back to stage 0. Otherwise apply `pm`; a `Rewritten` result is pushed at stage 0 — fully re-traversed and re-matched, which is the fixpoint — with a stage-2 link. |
| 2 — Link | Map the original node to the final result of its replacement. |

Results are memoized by `UOp::id` (`replace`, `bpm_cache`; `Gate` is never cached). Two limits: `REWRITE_STACK_LIMIT = 500_000` stack entries (`"infinite loop in graph_rewrite (stack too big: ..)"`), and a per-node `bpm_seen` set that panics when a bottom-up fixpoint revisits a node. There is no iteration cap.

| Entry point | Matchers |
|-------------|----------|
| `graph_rewrite(pm, root, ctx)` | `pm` at stage 1 — rules see rewritten children (Tinygrad default) |
| `graph_rewrite_bottom_up(bpm, root, ctx)` | `bpm` at stage 0 — rules see original children (Tinygrad `bottom_up=True`); `Gate` is honoured |
| `graph_rewrite_with_bpm(pm, bpm, root, ctx)` | both; only used by tests |
| `graph_rewrite_walk(bpm, root, ctx)` | one pass, replacements not re-traversed (Tinygrad `walk=True`) |
| `*_preserve_calls` variants | the same without entering `CALL`/`FUNCTION` bodies or `PROGRAM` internals (Tinygrad `enter_calls=False`) |

`RewriteResult` is `NoMatch`, `Rewritten(Arc<UOp>)` or `Gate(Arc<UOp>)`; in a `pm` matcher `Gate` is treated as `NoMatch`. The kernel cut uses `Gate` to stop `split_all_stores` from descending into an already-formed kernel `SINK` (`rangeify/kernel.rs`). A `debug_assert` fires if a rule returns the node it was given.

The one non-graph driver is `line_rewrite` (`linearize/mod.rs`): it walks the linear instruction list once, lets each entry expand into several, and substitutes later sources through a map. Its only client is `line_rewrite_cleanups`, the gated-`STORE` → `IF`/`STORE`/`ENDIF` expansion.

`RUST_LOG=svod_ir::pattern=trace` logs every match (`op_key`); it does not log which rule fired or which were tried.

## Composition is ordered

Matchers are composed by `+` in a fixed order, and the order carries meaning. `symbolic_simple()` starts with `propagate_invalid` because `x * 0 → 0` would otherwise erase `MUL(0, WHERE(c, x, Invalid))` together with its validity; `with_tier2` orders canonicalization before term combining and ALU folding before the comparison rules because each group exposes matches for the next (see [algebraic simplification](./algebraic-simplification.md)). Adding a rule means choosing where in that order it fires.

## Verifying rewrites with Z3

`schedule/src/z3/` (feature `z3`, optional dependency `z3 = "0.21"`, system `libz3`; the nix flake provides it) checks rewrites instead of trusting them:

- `convert.rs` translates a UOp tree into a Z3 term: `CONST` (int, uint, bool; floats and `Invalid` are rejected), `DefineVar` as a bounded integer, `RANGE` as a fresh variable with `0 <= r < end`, `Neg`, the integer binary ops `Add, Sub, Mul, FloorDiv, FloorMod, CDiv, CMod, Max, Lt, Eq, Ne` (`And`/`Or` on bools), `WHERE` and `MulAcc` on integers, and `CAST` as a dtype-bounded fresh variable tied to its source when the source range fits. `alu.rs` gives `CDiv`/`CMod` C truncation semantics; floor division is built on them. Anything else is a `ConversionError`.
- `verify_equivalence(original, simplified)` converts both into one context and asserts `original != simplified`: `UNSAT` proves the rewrite, `SAT` returns `CounterExample::Found { model, .. }`, a timeout `Unknown`.

```rust
/// The identity elimination `x + 0 = x` is pointer-identical and Z3-proven.
#[test]
fn z3_verify_identity_add_zero(x in arb_var_uop(DType::Int32)) {
    let zero = UOp::native_const(0i32);
    let expr = x.try_add(&zero).expect("ADD accepts matching dtypes");
    let simplified = rewrite(Matchers::simple(), expr.clone());
    prop_assert!(Arc::ptr_eq(&simplified, &x));
    verify_equivalence(&expr, &simplified).expect("Z3 should verify x + 0 = x");
}
```

What is covered (`schedule/src/test/`): hand-written rows through `symbolic_simple` (`unit/z3/symbolic_patterns.rs`), proptest oracles over `arb_arithmetic_tree_bounded_up_to` and `arb_known_property_graph` through `symbolic_simple` and `symbolic` (`property/oracles.rs`, 300–500 cases each), and a dual run of the structural symbolic tests that re-checks each row with Z3 when it converts (`unit/symbolic/mod.rs`). Only `Found` fails a test; `Unknown` and `ConversionFailed` are tolerated, and a liveness test guards that the arithmetic core still converts. Run it with `cargo test -p svod-schedule --features z3,proptest`; CI runs the same features through `nix flake check`.

The proof is over unbounded integers on sampled expressions of a limited op subset — a strong regression net for the index simplifier, not a verification of every pattern.
