---
sidebar_label: 简介
---

# Svod

> **Alpha 阶段软件。** 核心功能已测试，但 API 尚不稳定，可能随时更改。

Svod 是一个基于 Rust 的 ML 编译器，灵感来自 [Tinygrad](https://github.com/tinygrad/tinygrad)。它具有基于 UOp 的 IR 惰性张量求值、模式驱动优化和多后端代码生成。

## 亮点

| 特性 | 描述 |
|---------|-------------|
| **声明式优化** | `patterns!` DSL 实现图重写，通过 Z3 验证正确性 |
| **惰性求值** | Tensor 构建计算图，仅在 `realize()` 时编译执行 |
| **来源追踪** | 作用域把每个内核归属到模块路径、调用点或 ONNX 节点 |
| **80+ IR 操作** | 算术、内存、控制流、WMMA tensor core |
| **20+ 优化** | 常量折叠、tensor core、向量化、循环展开 |

架构详情请参阅[文档站点](https://npatsakula.github.io/svod/)。

## 工作空间

| Crate | 描述 |
|-------|-------------|
| [dtype](https://github.com/npatsakula/svod/tree/main/dtype/) | 类型系统：标量、向量、指针、图像 |
| [macros](https://github.com/npatsakula/svod/tree/main/macros/) | 过程宏（`patterns!` DSL） |
| [ir](https://github.com/npatsakula/svod/tree/main/ir/) | UOp 图 IR：80+ 操作、符号整数、内核来源 |
| [device](https://github.com/npatsakula/svod/tree/main/device/) | 缓冲区管理：惰性分配、零拷贝视图、LRU 缓存 |
| [schedule](https://github.com/npatsakula/svod/tree/main/schedule/) | 优化引擎：20+ 趟、RANGEIFY、Z3 验证 |
| [codegen](https://github.com/npatsakula/svod/tree/main/codegen/) | 代码生成：Clang（默认）、LLVM JIT |
| [runtime](https://github.com/npatsakula/svod/tree/main/runtime/) | JIT 编译与内核执行 |
| [tensor](https://github.com/npatsakula/svod/tree/main/tensor/) | 高层惰性张量 API |
| [onnx](https://github.com/npatsakula/svod/tree/main/onnx/) | ONNX 模型导入器 |
| [arch](https://github.com/npatsakula/svod/tree/main/arch/) | 推理原语 |
| [model](https://github.com/npatsakula/svod/tree/main/model/) | 预训练模型 |

## 快速示例

```rust
use svod_tensor::Tensor;
use ndarray::array;

// Zero-copy from ndarray (C-contiguous fast path)
let a = Tensor::from_ndarray(&array![[1.0f32, 2.0], [3.0, 4.0]]);
let b = Tensor::from_ndarray(&array![[5.0f32, 6.0], [7.0, 8.0]]);

// Lazy — nothing executes yet
let mut c = &a + &b;
c.realize()?;

// Zero-copy view into the result
let view = c.array_view::<f32>()?;
assert_eq!(view, array![[6.0, 8.0], [10.0, 12.0]].into_dyn());
```

## 模式 DSL 示例

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

## 开发

### 环境配置

#### Nix

本项目包含预定义的 Nix 开发环境，已包含所有依赖和编译器。CI/CD 也使用相同的基础设施，因此这是推荐的开发和测试方式。

```bash
nix develop # Open development shell
nix flake check # Run CI tests
nix fmt # Format source files
```

#### 裸金属

| 依赖 | 版本 | 必需 | 描述 |
|------------|---------|----------|-------------|
| Rust | 1.85+ | 是 | Edition 2024 |
| LLVM | 21.x | 是 | CPU 代码生成后端 |
| Clang | - | 是 | LLVM 构建所需的 C 编译器 |
| pkgconf | - | 是 | 构建配置工具 |
| protobuf | - | 是 | ONNX proto 编译 |
| zlib | >=1.3 | 是 | 压缩库 |
| libffi | >=3.4 | 是 | 外部函数接口 |
| libxml2 | >=2.13 | 是 | XML 解析 |
| Z3 | >=4.15 | 否 | 用于优化验证的 SMT 求解器 |
| NVIDIA 驱动 | CUDA >=12.8 (R570) | 否 | CUDA 后端所需的 `libcuda.so.1`，在运行时加载；无需 toolkit |
| Clang NVPTX / AMDGPU target | - | 否 | GPU 内核编译（`clang --print-targets`） |

## 测试

```bash
cargo test
cargo test --features z3,proptest  # With Z3 verification and PB generated tests
```
