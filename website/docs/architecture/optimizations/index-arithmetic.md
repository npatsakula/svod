---
sidebar_label: Index arithmetic
---

# Index Arithmetic

A `tensor[i, j]` access over shape `[H, W]` is `i * W + j`. After range splitting, merging, unrolling and GPU dimension reconstruction, index expressions accumulate `FloorDiv`/`FloorMod` chains; a division that survives to the backend costs tens of cycles where the rest of the address costs one. This page is the integer algebra that removes them. Source: `range_based_mod_div_patterns`, `advanced_division_dsl_patterns` and `div_mod_recombine_dsl_patterns` in `schedule/src/symbolic/patterns.rs`, `fold_divmod_general` in `schedule/src/symbolic/divmod.rs`, lowering in `schedule/src/symbolic/index_lowering.rs`. Tinygrad: `uop/divandmod.py`, `uop/symbolic.py`, `uop/weak.py`.

Two facts underlie every rule:

- Index arithmetic is `WeakInt` until stage 17 commits it (`UOp::index_const` builds a `WeakInt`); `ScalarDType::Index` is a legacy name that `spec` verifies absent after lowering.
- Every rewrite goes through `exact_integer_rewrite`: a candidate is accepted only if neither the original nor the replacement can wrap in its concrete dtype. The helpers construct candidates in unbounded integer algebra; the proof is at the call site.

## Range-based rules (`range_based_mod_div_patterns`)

`SoundVminVmaxProperty` gives `[vmin, vmax]` of every node (a `RANGE(end)` is `[0, end-1]`):

| Pattern | Result | Guard |
|---------|--------|-------|
| `RANGE(end) % end`, `RANGE(end) // end` | the range, `0` | same `end` node |
| `x % n` | `x` | `0 <= vmin(x)` and `vmax(x) < n` |
| `(a*n + b) % n` | `b % n` | `vmin(b) >= 0`; `n` compared by value |
| `(a*n + b + c) % n` | `(b + c) % n` | `vmin(b + c) >= 0` |
| `(a*n + b) // n` | `a` when `0 <= b < n`, else `a + b // n` | `n > 0`, `vmin(b) >= 0` |
| `x // n` | `k` | `vmin(x) // n == vmax(x) // n` (one bucket) |
| `(a + (x//n)*n) // n` | `x // n` | `0 <= a < n` |
| `(x + c) // d` | `x // d` | `c > 0`, `d > 0`, `vmin(x) >= 0`, and the largest remainder `x % d` can take plus `c` is still below `d` |
| `(x + c) // d` | `(x + c%d) // d + c//d` | `c % d != c`, any `d != 0` |
| `(x + c) // d`, `x <= 0 <= x + c` | `-(-(c%d + x - (d-1)) // d) + c//d` | `d > 0` |

The first two rows are the workhorses: after a split, `RANGE(n) % n` and `(outer*4 + inner) // 4` are the whole story. The remainder bound in the `(x + c) // d` rule is computed on the range `[vmin, vmax]` of `x`, not on its stride, so `(R*4 + 1) // 8` folds only while `R*4` spans fewer than 8 values.

## Advanced division (`advanced_division_dsl_patterns`)

| Pattern | Result | Guard |
|---------|--------|-------|
| `(a // b) // c` | `a // (b*c)` | `b != 0`, `c > 0`, `b*c` does not wrap |
| `expr // d` | `expr.divides(d)` | every additive term of `expr` is exactly divisible (`UOp::divides` recurses through `Add`, so `(a + b) // c` with both divisible is covered) |
| `(a + b) % c` | `b % c` or `a % c` | the dropped term is exactly divisible |
| `(x + c) % d` | `(x + c%d) % d` | `d > 0`, `c % d != c` |
| `x % y`, `x // y` | `fold_divmod_general(..)` | see below |
| `(a - b) // c` | `a//c - b//c` | both exactly divisible |
| `(a//c1 + c2) // c3` | `(a + c1*c2) // (c1*c3)` | `c1, c3 > 0`, `a` and `c2` of one sign, products do not wrap |

There is no generic `c * (a + b)` distribution for integers: only the `WeakInt` and `-1` forms in term combining and `sym_phase3_patterns`.

## `fold_divmod_general`

The port of Tinygrad's `fold_divmod_general`, called for any scalar integer `FloorDiv`/`FloorMod` with a denominator whose range is not exactly `{0}`. Rules in order; the first that produces a candidate wins:

1. **cancel** — if the quotient `x // y` has a single-value range, return it (for `%`: `x - q*y`).
2. **multiple-of guard** — a `PARAM` declared `multiple_of` a divisor that divides it: `% → 0`, `//` untouched.
3. **nested_div** (`//` only) — `(a % (k*c)) // c` → `(a // c) % k`, `k.vmin > 0`.
4. **remove_nested_mod** (`%` only) — `(a % (k*c) + b) % c` → `(a + b) % c`.
5. **congruence** (`fold_divmod_congruence`) — write `x = Σ f_i t_i + k`; for each coefficient choose a residue `r_i ≡ f_i (mod c)` (the smaller of the two signs, both tried for a lone term or an exact tie, `itertools.product` order); if `rem = Σ r_i t_i + k%c` stays in one quotient bucket, `x % c = rem - bucket*c` and `x // c = Σ (f_i - r_i)/c · t_i + (k - k%c + bucket*c)/c`. Sign-agnostic.
6. **gcd_with_remainder** — with `g = gcd(c, all f_i) > 1` and `x/g` non-negative: `((x/g + shift) // (c/g))` or `((x/g + shift) % (c/g)) * g + k%g`.
7. **nest_by_factor** — for every coefficient `f` that properly divides `c` (`2 <= f < c`), rewrite `x // c` as `(x // f) // (c/f)` (recursively folding the inner division) and for `%` as `((x // f) % (c/f)) * f + x % f`; the candidate with the smallest `node_count()` wins. The `%` branch needs `x >= 0` and a low digit provably in `[0, f)`.
8. **divide_by_gcd** (any denominator) — `x op y` → `(x/g) op (y/g)` (times `g` for `%`) when `symbolic_gcd(terms, y)` is not 1. This is what folds `(N*i + j) // N` for a symbolic `N`.
9. **factor_remainder** (any denominator) — split the terms into those exactly divisible by `y` and the rest; `//` → `quotient + rest // y`, `%` → `rest % y`; a constant divisor also reduces a coefficient to its residue (`(r*8 + v) % 7` → `(r + v) % 7`). Needs `x >= 0`, `y >= 0` and a non-negative remainder.

Example: `(R*8 + v) // 8` with `R: [0, 16)`, `v: [0, 8)` — rule 1 fails (16 buckets), rule 5 gives residues `0` for the coefficient `8` and `1` for `v`, `rem = v ∈ [0, 7]` is one bucket, so the quotient is `R` and the remainder `v`.

## Recombination

`fold_add_divmod_recombine`, on every `Add` (tier 1): finds a scaled remainder `(base % div) * mul` and the partner `q * (div*mul)` with `q` a quotient of something congruent to `base` modulo `div`, and returns `b * mul` plus the other terms — or `(b % (div*d)) * mul` when `q` is itself `(b // div) % d`. The variants `x%n + (x//n)*n → x`, `(x//a)%c + (x//(a*c))*c → x//a`, `(x%c1)*c2 + (x//c1)*(c1*c2) → x*c2` and the offset forms all come out of this one search.

## Index dtype lowering

`pm_lower_index_dtype` (stage `17-pm_lower_index_dtype`, [devectorizer page](../codegen/devectorizer.md)) commits `WeakInt` to `Int32` or `Int64`:

- `select_dtype(u)`: `WeakFloat` → the default float; integer bounds inside `[i32::MIN, i32::MAX]` → the default int, else `Int64`.
- Leaves (`CONST`, `VCONST`, scalar `PARAM` variables) become `concrete.cast(WeakInt)`; `Unary`, `Binary`, `WHERE`, `RANGE`, `STACK`, `SPECIAL` unwrap the casts of their sources and are rebuilt at the concrete dtype (`least_upper_dtype` of `select_dtype(u)` and the sources for a binary op); a weak `INDEX` casts its buffer and commits its indices; a consumer that is not weak absorbs the trailing cast on its own edge (`lower_weak_srcs`, memoized per kernel).
- `WHERE(valid, idx, Invalid)` keeps its shape; `Invalid` is never cast. A gated index that came out `Int64` is narrowed back to `Int32` when the buffer's element count fits `i32`.

The validity `WHERE` becomes a LOAD/STORE `gate` two stages later, in `pm_move_gates_from_index` (`late/gater.rs`); `INDEX` itself has no gate field.

## Worked example

`tensor[i, j]` with shape `[4, 8]`, iterated flat over 32 elements by `R0 ∈ [0, 32)`:

```text
row = R0 // 8, col = R0 % 8
addr = row * 8 + col = (R0 // 8) * 8 + (R0 % 8)
```

Recombination gives `addr = R0` immediately. Now unroll by 4: `pm_split_ranges` substitutes `R0 = R1 * 4 + R2` with `R1 ∈ [0, 8)`, `R2 ∈ [0, 4)`:

```text
row = (R1*4 + R2) // 8
col = (R1*4 + R2) % 8
```

Running `graph_rewrite(symbolic(), ..)` on these (the trees below are its output):

```text
row = (R1*4 + R2)//8
[31] FloorDiv : Scalar(WeakInt) shape=[]
├── [13] RANGE(R1, Weak) : Scalar(WeakInt) shape=[]
│   └── [2] CONST(Int(8)) : Scalar(WeakInt) shape=[]
└── [29] CONST(Int(2)) : Scalar(WeakInt) shape=[]

col = (R1*4 + R2)%8
[55] Add : Scalar(WeakInt) shape=[]
├── [54] Mul : Scalar(WeakInt) shape=[]
│   ├── [50] FloorMod : Scalar(WeakInt) shape=[]
│   │   ├── [13] RANGE(R1, Weak) : Scalar(WeakInt) shape=[]
│   │   │   └── [2] CONST(Int(8)) : Scalar(WeakInt) shape=[]
│   │   └── [46] CONST(Int(2)) : Scalar(WeakInt) shape=[]
│   └── [14] CONST(Int(4)) : Scalar(WeakInt) shape=[]
└── [15] RANGE(R2, Weak) : Scalar(WeakInt) shape=[]
    └── [14] → (see above)
```

`row`: rule 1 fails (four buckets), congruence fails (`rem = R1*4 + R2 ∈ [0, 31]` spans four buckets), `gcd(8, 4, 1) = 1`, and `nest_by_factor` with `f = 4` folds `(R1*4 + R2) // 4` to `R1` by congruence (`rem = R2 ∈ [0, 3]`) and leaves `R1 // 2`. `col`: the same factor gives `(R1 % 2) * 4 + R2`. Put back together, `row*8 + col` recombines to `R1*4 + R2` — the address is linear again even though `row` and `col` on their own still carry a division and a modulo. Split once more, `R1 = R3 * 2 + R4` (`R3 ∈ [0, 4)`, `R4 ∈ [0, 2)`), and the range-based rules finish the job:

```text
row = (R3*8 + R4*4 + R2) // 8   → R3                (a*n + b)//n with b = R4*4 + R2 ∈ [0, 7]
col = (R3*8 + R4*4 + R2) % 8    → R4*4 + R2         (a*n + b + c)%n, then x % n → x
addr = row*8 + col              → (R4*4 + R2) + R3*8
```

Zero divisions, zero modulos: the tiled address is the flat address, proven by rewriting. Two more outputs of the same run: `(R*8 + v) // 8` with `R: [0, 16)`, `v: [0, 8)` → `R` (rule 5), and `(R*8 + v) % 7` → `(R + v) % 7` (rule 9's coefficient reduction).
