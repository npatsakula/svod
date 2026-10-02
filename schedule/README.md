# svod-schedule

Pattern matching engine and RANGEIFY transformation for kernel generation.

## Example

```rust
use svod_schedule::{patterns, graph_rewrite};

let matcher = patterns! {
    // Identity folding
    Add[x, @zero] => x,
    Mul[x, @one] => x,

    // Constant folding (fallible)
    Add(a @const(a_val), b @const(b_val))
      => eval_add(a_val, b_val).map(|r| UOp::const_(a.dtype(), r)),

    // Self-folding
    Neg(Neg(x)) => x,
};
let optimized = graph_rewrite(&matcher, graph, &mut ());
```

## Features

**Supported:**

- Declarative `patterns!` DSL for rewrite rules
- Fixed-point graph rewriting
- RANGEIFY: movement ops -> BUFFERIZE+INDEX -> kernels
- Symbolic simplification (17+ pattern categories)
- Heuristic-based kernel optimization

**Planned:**

- BEAM search optimization
- Multi-device kernel splitting

## Optimizations

| Optimization | Status | Description |
|--------------|--------|-------------|
| **Symbolic Simplification** | | |
| Constant folding | ✅ | Fold unary/binary/ternary ops on constants |
| Identity elimination | ✅ | x+0, x*1, x/1, x|0, x^0 → x |
| Zero propagation | ✅ | x*0, x&0 → 0 |
| Self-folding | ✅ | x//x→1, x&x→x, x|x→x, x^x→0 |
| ALU folding | ✅ | (x+c1)+c2 → x+(c1+c2) |
| Term combining | ✅ | x+x→2*x, (c1*x)+(c2\*x)→(c1+c2)\*x |
| Division distribution | ✅ | (a+b)//c → a//c + b//c when exact |
| Dead code elimination | ✅ | WHERE(true,t,f)→t, dead loops |
| Reduce gate lifting | ✅ | REDUCE(WHERE(c,x,Invalid))→WHERE(c,REDUCE(x),Invalid) |
| **Kernel Optimization** | | |
| Tensor cores (TC) | ✅ | WMMA for matmul patterns |
| BEAM search | ✅ | `BEAM=n` candidate search, cached per kernel |
| Upcasting | ✅ | Vectorization (float4, etc.) |
| Loop unrolling | ✅ | Unroll reductions ≤32 |
| Local memory | ✅ | GPU shared memory allocation |
| Grouped reduction | ✅ | 2-stage GROUP/GROUPTOP |
| Matvec optimization | ✅ | Specialized MV pattern |
| CPU threading | ✅ | `core_id` split, `SVOD_THREADS` ways by default |
| Axis reordering | ✅ | SWAP for memory access |
| PADTO | ✅ | Pad an axis to TC alignment; `BEAM_PADTO` adds it to BEAM |
| Kernel caching | ✅ | In-process dedup, plus a persistent compiled-object cache |
| **RANGEIFY** | | |
| Movement op removal | ✅ | RESHAPE, PERMUTE, EXPAND, etc. |
| Buffer folding | ✅ | Remove noop BUFFERIZE |
| Dead axis removal | ✅ | Eliminate unused dimensions |
| Range flattening | ✅ | Flatten nested RANGE ops |
| Reduce simplification | ✅ | Optimize reduction patterns |
| Kernel splitting | ✅ | Split by STORE operations |
| Reduce splitting | ✅ | 2-stage large reductions |
| Buffer cost analysis | ✅ | PContig cost model |
| Multi-device shard | ✅ | Single-axis `Op::Multi`, host-staged allreduce |
| **Planned** | | |
| Image float4 | ❌ | Image type vectorization |
| Ring allreduce | ❌ | Multi-rank collectives; the shard subset above is host-staged |

## Testing

```bash
cargo test -p svod-schedule
cargo test -p svod-schedule --features z3  # with Z3 verification
```

### Property-Based Testing

Uses [proptest](https://github.com/proptest-rs/proptest) to verify algebraic properties across randomly generated expression trees (1000+ cases per property):

| Category | Properties |
|----------|------------|
| Identity | x+0→x, x*1→x, x-0→x, x/1→x, x|0→x, x^0→x |
| Zero | x*0→0, x&0→0, 0/x→0 |
| Self-fold | x/x→1, x&x→x, x|x→x, x\<x→false, x==x→true |
| Structural | Idempotence, dtype preservation, cost bounds |

### Z3 Verification

Uses [Z3](https://github.com/Z3Prover/z3) SMT solver to formally prove semantic equivalence of rewrites. Z3 is a theorem prover from Microsoft Research that can verify mathematical properties by constraint solving.

**How it works:**

1. Convert original and simplified UOps to Z3 expressions
2. Assert `NOT(original == simplified)`
3. If UNSAT → rewrite is proven correct
4. If SAT → found counterexample (bug)

```bash
cargo test -p svod-schedule --features z3
```

**Resources:**

- [Z3 GitHub](https://github.com/Z3Prover/z3)
- [Z3 Guide](https://microsoft.github.io/z3guide/)
- [z3.rs bindings](https://github.com/prove-rs/z3.rs)
