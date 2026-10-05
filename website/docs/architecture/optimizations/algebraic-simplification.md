---
sidebar_label: Algebraic simplification
---

# Algebraic Simplification

The symbolic simplifier is three nested matchers in `schedule/src/symbolic/patterns.rs`. Which tier runs where:

| Matcher | Composition | Runs at |
|---------|-------------|---------|
| `symbolic_simple()` | `symbolic_simple_base() + dead_loop_patterns()` | add-loads (13), devectorize (14), the image pass (17), index lowering (17), early decompositions (19b), and the `pm_decomp` set inside the final rewrite |
| `symbolic()` | `symbolic_simple` + the tier-2 groups | the rangeify mega-pass, range splitting/merging (`+ pm_fold_cast_const`), `indexing_simplify`, final symbolic (18) |
| `sym()` | `symbolic` + tier 3 | pre-optimization (`+ pm_fold_cast_const + pm_flatten_range`), post-opt symbolic (08), early symbolic (15), extra symbolic (16, `+ indexing_simplify`) |

`pm_fold_cast_const` (`CAST(CONST) → CONST`) is deliberately *not* inside any tier; the sites that want it add it explicitly, exactly as Tinygrad composes `symbolic + pm_fold_cast_const` only where `UOp.simplify` does.

Every rule that rewrites integer arithmetic goes through `exact_integer_rewrite`, a typed no-wrap proof (`typed_integer_rewrite_is_exact`) that declines the rewrite when either the original or the replacement could overflow its concrete dtype. The groups marked *value-sensitive* are additionally wrapped by `value_sensitive`, which disables them until `weak_float_values_are_committed` holds for the subtree. Bounds come from `VminVmaxProperty` (always available) and `SoundVminVmaxProperty` (`None` for ops whose bounds are not trustworthy: loads, `Pow`, `Fdiv`), both cached per node.

## Composition

```text
symbolic_simple_base()         tier 1
  propagate_invalid                     must be first (before x*0 → 0)
  fold_invalid_load_store
  constant_folding_dsl_patterns         value-sensitive
  vconst_folding_patterns               value-sensitive
  bool_arithmetic_patterns
  identity_and_zero_patterns            value-sensitive
  self_folding_dsl_patterns
  zero_folding_dsl_patterns
  division_dsl_patterns                 value-sensitive
  cast_dsl_patterns
  uint_pack_dsl_patterns
  div_mod_recombine_dsl_patterns
  power_dsl_patterns                    value-sensitive
  boolean_dsl_simple_patterns
  dce_dsl_simple_patterns               value-sensitive
symbolic_simple() = base + dead_loop_patterns

symbolic() = symbolic_simple + tier 2 (with_tier2)
  commutative_canonicalization
  boolean_dsl_patterns
  term_combining_dsl_patterns           value-sensitive
  dce_dsl_patterns
  where_alu_combining_patterns
  vmin_vmax_collapse_patterns           value-sensitive
  minmax_dsl_patterns                   value-sensitive
  alu_folding_dsl_patterns              value-sensitive
  comparison_dsl_patterns               value-sensitive
  range_based_mod_div_patterns
  advanced_division_dsl_patterns
  range_based_cast_patterns
  long_to_int_narrowing_patterns
  after_simplification_patterns
  where_bound_patterns                  value-sensitive

sym() = symbolic + tier 3
  pm_simplify_valid                     (symbolic/valid_simplification.rs)
  alu_vectorize_reorder_patterns
  ne_zero_fold_patterns                 value-sensitive
  cast_where_dsl_patterns
  store_load_folding_patterns
  reduce_sym_patterns                   value-sensitive
  sym_phase3_patterns
```

The order is load-bearing: canonicalization before term combining, ALU folding before the comparison and range rules, because each group exposes matches for the next. Tinygrad's reciprocal distribution rules (`uop/symbolic.py` `sym`) are deliberately absent: all six are IEEE-inexact.

**Notation.** `OP[a, b]` is commutative, `OP(a, b)` ordered; `@zero`/`@one`/`c` are constants; a repeated name is the same node (`Arc::ptr_eq`). `//` is `FloorDiv`, `%` is `FloorMod`; the truncating `CDiv`/`CMod` only appear after the late decompositions.

## Tier 1

### Invalid propagation (`propagate_invalid`)

`Invalid` is `UOp::invalid_marker()`, a `ConstValue::Invalid` constant; `is_invalid_marker` also recognises an all-`Invalid` `VCONST` or `STACK` and movement wrappers around one. Validity is kept as `WHERE(cond, x, Invalid)` and these rules keep that shape intact while arithmetic moves around it:

| Pattern | Result |
|---------|--------|
| `WHERE(Invalid, _, _)` | `Invalid` |
| `WHERE(WHERE(c, x, Inv), a, b)` | `WHERE(c, WHERE(x, a, b), Inv)` |
| `WHERE(c, Inv, x)` | `WHERE(!c, x, Inv)` (`WHERE(c, Inv, Inv)` → `Inv`) |
| `WHERE(c1, WHERE(c2, x, d), d)` | `WHERE(c1 & c2, x, d)` |
| `WHERE(a, WHERE(c, x, Inv), y)`, `y` not `Invalid` | `WHERE(!a \| c, WHERE(a, x, y), Inv)` — and the mirror for the false branch |
| `unary(Inv)`, `CAST(Inv)`, `BITCAST(Inv)` | `Inv` |
| `unary(WHERE(c, x, Inv))`, `CAST`, `BITCAST` | `WHERE(c, unary(x), Inv)` |
| `op(WHERE(c, x, Inv), y)`, `op(y, WHERE(c, x, Inv))` for **every** binary op, comparisons included | `WHERE(c, op(x, y), Inv)` |
| `op(Inv, y)`, `op(y, Inv)` for the 13 non-comparison binary ops | `Inv` |

Why first: `MUL(0, WHERE(c, x, Inv))` must become `WHERE(c, 0, Inv)`, not `0`.

### Dead loads and stores (`fold_invalid_load_store`)

`LOAD(INDEX(buf, Invalid, ..))` (also behind a `CAST`) → the load's `alt` if it has one, else a shaped zero; `STORE(INDEX(buf, Invalid, ..), v)` without a gate → `NOOP`.

### Constant folding

Unary: `Sqrt, Exp2, Log2, Sin, Reciprocal, Trunc` (`Neg` is not an op here — `neg()` builds `MUL(x, -1)`). Binary: `Add, Mul, Sub, FloorMod, Max, Pow, FloorDiv, Fdiv, And, Or, Xor, Shl, Shr`, plus the six comparisons to `Bool`. Ternary: `Where`, `MulAcc`. Results are committed through the dtype's storage format (`Int32` wraps) except for weak dtypes, which keep the untruncated value. `vconst_folding_patterns` does the same lane-wise for `VCONST ⊕ VCONST` and the `CONST`/`VCONST` broadcast mix (11 binary ops + comparisons, 6 unary), skipping weak lanes.

### Bool arithmetic

`Mul[x, y]` → `x & y`, `Add[x, y]` → `x | y`, `Max(x, y)` → `x | y` when both are `Bool`.

### Identity and zero

| Pattern | Result | Guard |
|---------|--------|-------|
| `Add[x, 0]` | `x` | not float, or the zero is `-0.0` (`x + 0.0` is not an identity for `x = -0.0`) |
| `Sub(x, 0)` | `x` | not float, or the zero is `+0.0` |
| `Mul[x, 1]`, `Or[x, 0]`, `Xor[x, 0]`, `FloorDiv(x, 1)`, `Fdiv(x, 1)` | `x` | |
| `FloorMod(x, 1)` | `0` | |
| `Floor/Ceil/Trunc/Round(x)` | `x` | integer `x` |
| `Mul[x, 0]` | `0` | not float (`NaN * 0`, `Inf * 0` are `NaN`) |
| `And[_, 0]` | `0` | |

### Self and zero folding

`FloorDiv(x, x)` → `1`; `FloorDiv(x, -1)` → `MUL(x, -1)`; `FloorMod(FloorMod(x, y), y)` → `FloorMod(x, y)`; `And(x, x)`, `Or(x, x)`, `Max(x, x)` → `x`; `FloorMod(x, x)` → `0`; `Lt(x, x)` → `false` for non-floats, and for floats when the sound bounds prove `x` is not `NaN`; `Ne(x, x)` → `false` for ints and bools.

### Division

`Fdiv(0.0, 0.0)` and `Fdiv(MUL[_, 0.0], 0.0)` → `NaN` (listed first so they beat the next rule); `Fdiv(x, x)` → `1.0` only when `x` is provably finite and nonzero; `FloorDiv(Mul(x, y), y)` → `x`. There is no float `(x*y)/y → x`.

### Casts

`CAST(x, dt)` → `x` when the dtype already matches; `CAST(CAST(x, a), b)` → `x` when `x: b` and `can_safe_cast(b, a)` (`a` holds every value of `b`: same signedness and at least as wide, unsigned→signed needs one extra bit, float↔int never); `CAST(CAST(x, a), b)` → `CAST(x, b)` when `a` does not narrow `x`. `uint_pack_dsl_patterns` cancels the `(hi.cast(u64) << 32) | lo.cast(u64)` packing Threefry builds so the PRNG stays in 32-bit ALU.

### Div-mod recombination

One rule on every `Add`: `fold_add_divmod_recombine`, a port of Tinygrad's. It flattens the `Add` chain, finds a term `(base % div) * mul` and a partner `q * (div * mul)` whose `q` is a quotient of something congruent to `base` modulo `div` (`quotient_base`: `q == b // div`, possibly with merged `(x//c + a)//div` and shifted constants), and replaces the pair by `b * mul`; with `q == (b // div) % d` it folds into the wider `(b % (div*d)) * mul`. This is the family `x%n + (x//n)*n → x` and its scaled, offset and three-term variants, found through the chain rather than as separate rules.

### Power, booleans, DCE

`Pow(x, 0)` → `1`, `Pow(x, 1)` → `x`, `Pow(1, x)` → `1` (scalars only; no other exponent is rewritten — reciprocal/sqrt forms change IEEE rounding). `Not(Not(x))` → `x`, `Xor(x, x)` → `0`, `true | _` → `true`, `false & _` → `false`, `true & x` → `x`, `false | x` → `x` (bool constants only). `WHERE` with a provably constant condition (sound bounds) selects the branch; `WHERE(_, t, t)` → `t`; `WHERE(x, true, false)` → `x`; `WHERE(x, false, true)` → `!x`; `WHERE(a, WHERE(b, c, d), d)` → `WHERE(a & b, c, d)`. `dead_loop_patterns`: a `RANGE` with `vmax < 0` → `CONST(0)`, a `RANGE(CONST)` with `vmin == vmax` → that constant. There is no `END`/`REDUCE` empty-range fold here; `reduce_to_acc` handles those.

## Tier 2

### Commutative canonicalization

For `Add, Mul, Max, And, Or, Xor` (and nominally `Eq`/`Ne`, which never fire because their result is `Bool`) whose **result** dtype is `WeakInt`: swap the operands when `tinygrad_tuplize_cmp(b, a) == Less`, the structural `(op, arg, dtype, *src)` key order the linearizer also uses. Index expressions that are equal up to commutativity then hash-cons to one node, which the recombination and the expander rely on. Other dtypes keep authored order.

### Term combining (`Add`/`Mul` with constants)

| Pattern | Result |
|---------|--------|
| `Add(x, x)` | `x * 2` |
| `Add(Mul[x, c1], Mul[x, c2])` | `x * (c1 + c2)` |
| `Add[x, Mul[x, c]]` | `x * (c + 1)` |
| `Add[Add[y, Mul[x, c0]], Mul[x, c1]]` | `y + x * (c0 + c1)` |
| `Add[Add[y, x], Mul[x, c]]`, `Add[Add[y, Mul[x, c]], x]` | `y + x * (c + 1)` |
| `Add[Add[y, x], x]` | `y + x * 2` |
| `Mul[-1, Add[x, c]]` | `-x + (-c)` |
| `Mul[c, Add[x, k]]`, `x: WeakInt` | `c*x + c*k` |

### Boolean (`boolean_dsl_patterns`)

`Or[x, Not(x)]` → `true`, `And[x, Not(x)]` → `false` (bool only); De Morgan in both directions, `And[Not(x), Not(y)]` → `!(x | y)` and `Or[Not(x), Not(y)]` → `!(x & y)`.

### WHERE

`dce_dsl_patterns`: `WHERE(Not(c), t, f)` → `WHERE(c, f, t)` unless `f` contains `Invalid` (scalar or a `STACK` lane) — swapping would move the marker to the true branch where the gate rules cannot see it. `where_alu_combining_patterns`: `op(WHERE(c, a, b), WHERE(c, d, e))` → `WHERE(c, op(a, d), op(b, e))` for `Add, Mul, Sub, Max, And, Or, Xor` when both true branches or both false branches are constants, and the associative `Add(Add(y, WHERE(c, ..)), WHERE(c, ..))` form. `where_bound_patterns`: `WHERE(Lt(x, c), t, f)` → `t` when `x.vmax < c.vmin`, `f` when `x.vmin >= c.vmax`.

### Bounds collapse and min/max

`vmin_vmax_collapse_patterns`: a `Mul`, `FloorDiv`, `FloorMod`, comparison, `PARAM` or `SPECIAL` whose sound bounds are a single value becomes that constant (floats excluded; `Add`/`Sub`/`Max` deliberately excluded so a trip-1 loop carry is not folded away). `minmax_dsl_patterns`: `Max(x, y)` → `x` when `x.vmin >= y.vmax` (strictly greater for floats, to keep the sign of zero), symmetric for `y`. There is no `Min` op: `Tensor::minimum` is a `WHERE`.

### ALU chain folding (`alu_folding_dsl_patterns`)

Associative folding `(x ⊕ c1) ⊕ c2` → `x ⊕ (c1 ⊕ c2)` for `Add`, `Mul`, `And`, `Or`, `Xor`, `Max`; constant pushing `(x + c) + y` → `(x + y) + c` and `(x * c) * y` → `(x * y) * c` when `y` is not a constant; `(x - c1) + c2`, `(x + c1) - c2` normalised to `x + k` or `x - |k|`; `(x - c1) - c2` → `x - (c1 + c2)`; `Sub(a, Sub(b, x))` → `x + (a - b)`. `Sub` is a first-class op in Svod (Tinygrad spells `a - b` as `a + b*-1`).

### Comparisons (`comparison_dsl_patterns`)

For all six comparisons: `x op x` on non-floats folds (`Lt/Gt/Ne` → `false`, `Le/Ge/Eq` → `true`); constant operands fold; otherwise `ComparisonAnalyzer::analyze` (`ir/src/uop/comparison_analysis.rs`) proves `true` or `false` from the sound bounds — both only for non-weak dtypes. Then: `Lt(Add[c0, x], c1)` → `Lt(x, c1 - c0)`; `Lt(Mul[x, -1], Mul[y, -1])` → `Lt(y, x)`; `Lt(FloorDiv(x, d), c)` → `Lt(x, c * d)` for `d > 0` under a no-wrap check (exact for floor division, any sign of `c`); for `WeakInt`: `Lt(Mul[c0, x], c1)` → `±x < ceil(c1 / |c0|)` and the GCD fold `lt_folding` (`x = d*q + r`, `r ∈ [0, d)`, `d | c` ⇒ `x < c ⇔ q < c/d`).

### Ranges and division

`range_based_mod_div_patterns` and `advanced_division_dsl_patterns` are the index algebra; they are on the [index arithmetic](./index-arithmetic.md) page. `range_based_cast_patterns` collapses `CAST(CAST(x, a), b)` for strong integer `x` whose bounds fit `a`. `long_to_int_narrowing_patterns` rewrites an `Int64` binary op whose operands and result fit `i32` as the `Int32` op cast back, and distributes a signed-int cast over `WeakInt + c`.

### AFTER

`after_simplification_patterns`: deps that are not side effects (`RANGE`, `STORE`, `END`, `CALL`, `BARRIER`, `CUSTOM`, `FUNCTION`) are replaced by their own sources and deduplicated; `NOOP` deps and `END(NOOP)` chains are dropped; `AFTER(x, [])` → `x`.

## Tier 3 (`sym`)

- **`pm_simplify_valid`** (`valid_simplification.rs`): an `And` chain of `Bool` validity clauses is simplified clause by clause (`simplify_valid`), and `WHERE(cond, x, Invalid)` with `x: WeakInt` rewrites `x` under the bounds `cond` implies (`uop_given_valid`): `parse_valid` reads each clause as `expr < c` / `expr >= c`, substitutes a bounded variable and re-simplifies.
- **`alu_vectorize_reorder_patterns`**: `op(STACK(x, x, ..), STACK(y, y, ..))` → `STACK(op(x, y), ..)` for the 13 arithmetic/bitwise ops and the six comparisons, when both operands are broadcasts of one node with the same lane count > 1.
- **`ne_zero_fold_patterns`**: `Ne(x, 0)` → `x.cast(bool)`.
- **`cast_where_dsl_patterns`**: `CAST(WHERE(s, a, b))` → `WHERE(s, CAST(a), CAST(b))`.
- **`store_load_folding_patterns`**: `STORE(_, Invalid)` → `NOOP`; `STORE(INDEX, WHERE(c, v, Invalid))` → the gated index `INDEX(buf, WHERE(c, idx, Invalid))` storing `v`; `STORE(idx, LOAD(idx))` → `NOOP`; `STORE(INDEX, WHERE(g, alt, LOAD(same INDEX)))` → gated store of `alt`.
- **`reduce_sym_patterns`**: `REDUCE(x * c, Add)` → `REDUCE(x, Add) * c` and `reduce_mul_chain_sym` (range-independent factors pulled out of an `Add`/`Max` reduce; for `Max` only non-negative ones), integers only.
- **`sym_phase3_patterns`**: `-1 * (x + y)` → `-x + -y`; `(x + y) * c` → `x*c + y*c` for `WeakInt`; single-source `GROUP` unwrap; `NOOP`/`STACK`/`SINK` flattening into a `SINK`/`GROUP`; `END(NOOP)` → `NOOP`.

## Worked cascade

`(x + 0) * 1 + (3 + 4)` with `x: Int32`:

```text
Add(x, 0)        → x          identity_and_zero
Mul(x, 1)        → x          identity_and_zero
Add(3, 4)        → 7          constant_folding
Add(x, 7)                     stays: no rule
```

The engine rewrites children before parents and re-matches a rebuilt parent, so the three steps happen in one `graph_rewrite`. The identity steps are the ones the [Z3 oracles](./pattern-system.md#verifying-rewrites-with-z3) prove.
