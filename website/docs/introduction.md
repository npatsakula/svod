---
sidebar_label: Introduction
---

# Svod

> **Alpha software.** Core functionality is tested, but APIs are unstable and may change without notice.

A Rust-based ML compiler inspired by [Tinygrad](https://github.com/tinygrad/tinygrad). Lazy tensor evaluation with UOp-based IR, pattern-driven optimization, and multi-backend code generation.

## Highlights

| Feature | Description |
|---------|-------------|
| **Declarative Optimization** | `patterns!` DSL for graph rewrites with Z3-verified correctness |
| **Lazy Evaluation** | Tensors build computation graphs, compiled only at `realize()` |
| **Origin Tracking** | scopes attribute every kernel to a module path, call site, or ONNX node |
| **80+ IR Operations** | Arithmetic, memory, control flow, WMMA tensor cores |
| **20+ Optimizations** | Constant folding, tensor cores, vectorization, loop unrolling |

For architecture details, see the [documentation site](https://npatsakula.github.io/svod/).

## Workspace

| Crate | Description |
|-------|-------------|
| [dtype](https://github.com/npatsakula/svod/tree/main/dtype/) | Type system: scalars, vectors, pointers, images |
| [macros](https://github.com/npatsakula/svod/tree/main/macros/) | Procedural macros (`patterns!` DSL) |
| [ir](https://github.com/npatsakula/svod/tree/main/ir/) | UOp graph IR: 80+ ops, symbolic integers, origins |
| [device](https://github.com/npatsakula/svod/tree/main/device/) | Buffer management: lazy alloc, zero-copy views, LRU caching |
| [schedule](https://github.com/npatsakula/svod/tree/main/schedule/) | Optimization engine: 20+ passes, RANGEIFY, Z3 verification |
| [codegen](https://github.com/npatsakula/svod/tree/main/codegen/) | Code generation: Clang (default), LLVM JIT |
| [runtime](https://github.com/npatsakula/svod/tree/main/runtime/) | JIT compilation and kernel execution |
| [tensor](https://github.com/npatsakula/svod/tree/main/tensor/) | High-level lazy tensor API |
| [onnx](https://github.com/npatsakula/svod/tree/main/onnx/) | ONNX model importer |
| [arch](https://github.com/npatsakula/svod/tree/main/arch/) | Inference primitives |
| [model](https://github.com/npatsakula/svod/tree/main/model/) | Pretrained models |

## Quick Example

```rust
use svod_tensor::Tensor;
use ndarray::array;

// Zero-copy from ndarray (C-contiguous fast path)
let a = Tensor::from_ndarray(&array![[1.0f32, 2.0], [3.0, 4.0]]);
let b = Tensor::from_ndarray(&array![[5.0f32, 6.0], [7.0, 8.0]]);

// Lazy — nothing executes yet
let c = &a + &b;

// Compile, execute, extract as ndarray
let result = c.to_ndarray::<f32>()?;
assert_eq!(result, array![[6.0, 8.0], [10.0, 12.0]].into_dyn());

// Or extract as flat Vec
let flat = c.to_vec::<f32>()?;
assert_eq!(flat, vec![6.0, 8.0, 10.0, 12.0]);
```

## Pattern DSL Example

```rust
use svod_schedule::patterns;

let optimizer = patterns! {
    // Identity folding (commutative)
    Add[x, @zero] => x,
    Mul[x, @one] => x,

    // Constant folding
    for op in binary [Add, Mul, Sub] {
        op(a @const(av), _b @const(bv))
          => eval_binary_op(op, av, bv).map(|r| UOp::const_(a.dtype(), r)),
    },
};
```

## Development

### Environment setup

#### Nix

This project contains a pre-defined Nix development environment
with all dependencies and compilers included. The same infrastructure
is used for CI/CD, so it's the preferred way to develop and test.

```bash
nix develop # Open development shell
nix flake check # Run CI tests
nix fmt # Format source files
```

#### Bare metal

| Dependency | Version | Required | Description |
|------------|---------|----------|-------------|
| Rust | 1.85+ | yes | Edition 2024 |
| LLVM | 21.x | yes | CPU code generation backend |
| Clang | - | yes | C compiler for LLVM builds |
| pkgconf | - | yes | Build configuration tool |
| protobuf | - | yes | ONNX proto compilation |
| zlib | >=1.3 | yes | Compression library |
| libffi | >=3.4 | yes | Foreign function interface |
| libxml2 | >=2.13 | yes | XML parsing |
| Z3 | >=4.15 | no | SMT solver for optimization verification |
| NVIDIA driver | CUDA >=12.8 (R570) | no | `libcuda.so.1` for the CUDA backend, loaded at runtime; no toolkit needed |
| Clang NVPTX / AMDGPU targets | - | no | GPU kernel compilation (`clang --print-targets`) |

### Threads

`SVOD_THREADS` (default: host parallelism) is the single thread budget: it sizes
the pool that compiles kernels and executes CPU kernels, and is the default
`core_id` split of CPU kernels. `RAYON_NUM_THREADS` is not consulted.

## Test

```bash
cargo test
cargo test --features z3,proptest  # With Z3 verification and PB generated tests
```
