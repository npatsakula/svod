# svod-schedule

The scheduling and optimization passes of the Svod ML compiler, all written as
`graph_rewrite` over `patterns!` matchers. RANGEIFY turns movement ops into
explicit `RANGE` loops and `STAGE` / `INDEX` nodes, the kernel cut
(`try_get_kernel_graph`) splits the graph into one `CALL` per kernel, and the
optimizer picks each kernel's `Opt`s (`UPCAST`, `UNROLL`, `LOCAL`, `THREAD`,
`GROUP`/`GROUPTOP`, `TC`, `PADTO`, `SWAP`, `NOLOCALS`) with hand-coded
heuristics or BEAM search (`BEAM=N`, results cached in a sled database). Later
passes expand, devectorize, assign GPU dimensions and linearize the kernel for
`svod-codegen`. The symbolic simplifier covers constant folding, identities,
term combining, division and modulo reasoning over value ranges, and dead-branch
removal.

## Example

```rust
use svod_ir::{UOp, uop::eval::eval_add};
use svod_schedule::{TypedPatternMatcher, graph_rewrite, patterns};

let matcher: TypedPatternMatcher = patterns! {
    Add[x, @zero] => x,
    Mul[x, @one] => x,
    Add(a @const(a_val), _b @const(b_val))
        => eval_add(a_val, b_val).map(|r| UOp::const_(a.dtype(), r)),
};
// Rewrites `graph: Arc<UOp>` to a fixed point.
let optimized = graph_rewrite(&matcher, graph, &mut ());
```

## Features

| Feature | Effect |
|---------|--------|
| `z3` | `svod_schedule::z3::verify_equivalence`: proves rewrites with the Z3 SMT solver |
| `proptest` | enables `svod-ir`'s proptest strategies |
| `testing` | tracing-subscriber helpers for tests |

Documentation:

- Pattern engine: <https://svod.vpermilp.online/docs/architecture/optimizations/pattern-system>
- Rangeify and the kernel cut: <https://svod.vpermilp.online/docs/architecture/codegen/rangeify>
- Kernel search (heuristics, BEAM, tensor cores): <https://svod.vpermilp.online/docs/architecture/optimizations/kernel-search>
