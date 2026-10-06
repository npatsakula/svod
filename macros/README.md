# svod-macros

Procedural macros for the Svod ML compiler:

| Macro | Purpose | Used through |
|-------|---------|--------------|
| `patterns!` / `cached_patterns!` | declarative rewrite rules compiled to a `TypedPatternMatcher` | `svod_schedule::{patterns, cached_patterns}` |
| `jit_wrapper!` | build-once, run-many wrapper with typed input, output and state buffers | `svod_macros::jit_wrapper` (needs `svod-tensor`) |
| `#[derive(Module)]` | state-dict save/load for a model struct | `svod_tensor::nn::Module` |
| `#[op_enum]` / `#[derive(PatternEnum)]` | per-op structs and `OpKey` dispatch for `svod_ir::Op` | internal to `svod-ir` |

```rust
use svod_schedule::{TypedPatternMatcher, patterns};

let matcher: TypedPatternMatcher = patterns! {
    Add[x, @zero] => x,
    Mul[x, @one] => x,
};
```

Documentation:

- Pattern DSL: <https://svod.vpermilp.online/docs/architecture/optimizations/pattern-system>
- JIT graphs: <https://svod.vpermilp.online/docs/architecture/jit-graphs>
