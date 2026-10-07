---
sidebar_label: 简介
---

<div align="center">

# Svod

**用 Rust 编写的深度学习编译器与推理引擎。**

[![CI](https://github.com/npatsakula/svod/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/npatsakula/svod/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-svod.vpermilp.online-blue)](https://svod.vpermilp.online/docs/introduction)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/npatsakula/svod/tree/main/LICENSE)

[文档](https://svod.vpermilp.online/docs/introduction) ·
[模型](https://github.com/npatsakula/svod/tree/main/#models-and-pipelines) ·
[架构](https://svod.vpermilp.online/docs/architecture/pipeline) ·
[演讲](https://github.com/npatsakula/svod/tree/main/#talks-and-writing) ·
[路线图](https://github.com/npatsakula/svod/tree/main/#roadmap)

</div>

Svod 将惰性张量图编译为面向 CPU、AMD 与 NVIDIA GPU 的融合内核，整条链路中没有任何
厂商运行时：不依赖 PyTorch，不依赖 ROCm/HIP，也不依赖 CUDA toolkit。它沿用
[Tinygrad](https://github.com/tinygrad/tinygrad) 的设计：一个小而可验证的 IR（UOp）、
由模式驱动的重写，以及从张量直达机器码的直线流水线。

Svod 自带语音、文本和视觉模型，均已对照各自的 PyTorch 参考实现校验过；其余模型则
通过 ONNX 导入器运行。

## 为什么选择 Svod

- **端到端只有一种表示。** PyTorch 让一个模型经过七种甚至更多 IR，每一道边界都是
  厂商必须搭建的桥梁，也是调试上下文的断点。在 Svod 中，单一的 UOp 图完成构建、
  优化、调度和渲染。新增一种加速器只需要三样东西：代码生成器、缓冲区分配器和内核
  启动器。
- **生产级 Rust。** 静态类型和原生并发，而不是围绕 GIL 的 Python 胶水。部署只是
  一个二进制文件，没有 LibTorch 或 ONNX Runtime 绑定，也没有厂商 SDK。
- **熟悉的 API。** 张量 API 对标 PyTorch，连具名参数都一致，因此移植后的模型读起来
  和参考实现一样，移植过程基本是机械性的。
- **在关键处追求速度。** 当编译器力有不逮时，`tk` tile DSL 让你在同一 IR 中手写
  内核。它们对性能分析器和来源追踪器始终可见，而不是作为不透明二进制游离在外。

动机与设计详见
[Developing Svod](https://blog.vpermilp.online/en/blog/svod-intro/)。

## 模型与流水线

| 领域 | 模型 |
|---|---|
| 语音识别 | Whisper、GigaAM v3（CTC、RN-T） |
| 语音活动检测 | FireRedVAD、Silero VAD |
| 语音增强 | GTCRN |
| 说话人分析 | Nemotron-3-Diarization、DiariZen、WeSpeaker |
| 文本嵌入与重排序 | BGE-M3、Qwen3-Embedding、ModernBERT |
| 视觉 | YOLO26、ResNet |
| 其他一切 | [ONNX 导入器](https://github.com/npatsakula/svod/tree/main/onnx/)（[算子覆盖](https://github.com/npatsakula/svod/tree/main/onnx/PARITY.md)） |

权重直接来自 Hugging Face Hub，输出与参考实现逐一比对。
[`model/`](https://github.com/npatsakula/svod/tree/main/model/) 列出了各个变体、上游链接和可运行示例；
[`arch`](https://github.com/npatsakula/svod/tree/main/arch/) 包含解码器和长音频流水线。

## 引擎

### 图捕获：编译一次，重放多次

模型只被追踪一次，生成执行计划，之后的每次调用只是重放它。符号维度（批大小、
序列长度）在每次调用时绑定，无需重新编译。循环状态在调用之间保留在设备上，内存
规划器通过 TLSF 竞技场复用中间缓冲区。静态链被作为一个硬件图重放：NVIDIA 上是
**CUDA Graphs**，AMD 上是每次重放只敲一次 doorbell 的 AQL/PM4 图（同 HCQGraph），
Metal 上是 indirect command buffers。
参见[JIT 图](https://svod.vpermilp.online/docs/architecture/jit-graphs)。

```rust
jit_wrapper! {
    GigaAmEncoderJit(GigaAm) {
        mel: Tensor,
        lengths: Tensor,

        outputs { frames },

        build(mel, lengths) {
            model.encoder.forward_batch(mel, lengths)
        }
    }
}
// let mut jit = GigaAmEncoderJit::new(model);
// jit.prepare(..)?;   // trace, schedule and compile once
// jit.execute()?;     // replay on every chunk
```

### 经 Z3 检查的重写

图变换以声明式规则写在 `patterns!` DSL 中。核心符号化简规则由 **Z3** 等价性证明固定，
属性测试则把 Z3 当作随机表达式的判定器：求解器证明化简后的表达式与原式相等，否则给出反例。
这些检查在 CI 中运行（`--features z3`），而非运行时。
参见[模式系统](https://svod.vpermilp.online/docs/architecture/optimizations/pattern-system)。

### 面向平台的代码生成

- **Tensor core** 按架构选择：NVIDIA sm_75/80/89、AMD RDNA3、RDNA4 与 CDNA3/4，
  以及 Apple Metal。fp8 在 sm_89 和 CDNA3 上可用。
- **Tile 内核（`tk`）**：Rust 实现的 ThunderKittens 风格 tile DSL，覆盖 GEMM、
  flash attention、RMSNorm 和 k-means。同一份内核源码可下沉为 AMD MFMA/WMMA
  （gfx942、gfx11、gfx12）、CUDA `mma.sync`（sm_80+）和 Apple
  `simdgroup_matrix`（Apple7+）。tile 形状在首次使用时自动
  调优并缓存。参见
  [Tile 内核](https://svod.vpermilp.online/docs/tile-kernels/overview)。

  ```rust
  fn micro_matmul(ker: &Kernel) -> Arc<UOp> {
      let w = ker.warp();
      let a = ker.rt((64, 64), DType::BFloat16, Row, RT_16X16);
      let b = ker.rt((64, 64), DType::BFloat16, Col, RT_16X16);
      let c = ker.rt((64, 64), DType::Float32, Col, RT_16X16);
      let out = w.mma_ab(w.zero(c), &a, &b); // one matrix-core instruction per fragment
      ker.finish(1)
  }
  ```
- **内核搜索**：手写启发式，或在优化空间上做 BEAM 搜索并带持久化磁盘缓存。参见
  [内核搜索](https://svod.vpermilp.online/docs/architecture/optimizations/kernel-search)。
- **CPU**：在进程内编译向量化 LLVM IR，自研 ELF 加载器支持 x86_64、aarch64、
  riscv64、loongarch64 和 ppc64le，并提供多线程内核。

### 零拷贝数据通路

ONNX 初始化器和 `Tensor::from_path` 张量从磁盘按需延迟内存映射。设备缓冲区
支持子视图。宿主代码通过借用的 `ndarray` 视图（`array_view`、`array_view_mut`）
读写已实例化的张量，因此向已捕获的计划喂数据不产生任何拷贝。

### 内核融合与归因

RANGEIFY 调度器把逐元素、归约和搬移操作融合成尽可能少的内核。每个内核都记录
自己的来源（模块路径、ONNX 节点或源码行），因此性能分析器可以把设备时间、
roofline GFLOP/s 与 GB/s、占用率和硬件计数器（AMD SQ、NVIDIA CUPTI）归因回模型
代码。参见[内核来源](https://svod.vpermilp.online/docs/architecture/kernel-origins)。

## 后端

| 设备 | 选择器 | 编译 | 运行时 |
|---|---|---|---|
| CPU | `CPU`（macOS 之外的默认值） | 通过运行时加载的 `libLLVM` 编译 LLVM IR，缺失时回退到 `clang`；Clang C 后端 | 自研 ELF JIT 加载器，多线程 |
| AMD GPU | `AMD:N` | `clang --target=amdgcn-amd-amdhsa` | 直接使用 KFD 队列（AQL/PM4），无 HIP、无 ROCm 运行时 |
| NVIDIA GPU | `CUDA:N` | `clang` NVPTX → PTX → `ptxas` 或驱动 JIT | 运行时加载 `libcuda.so.1`，无 CUDA toolkit |
| Apple GPU | `METAL:N`（macOS 上的默认值） | MSL → metallib | 运行时加载 Metal 框架 |

所有 GPU 后端都已编译进来，只在检测到硬件时才注册。要选择某个后端，设置
`SVOD_DEVICE` 或调用 `Tensor::to(device)`。

CPU 代码在 Linux 和 macOS 上于 x86_64、aarch64、riscv64 和 ppc64le 测试。GPU：
AMD RDNA 3.5、RDNA 4 和 CDNA 3，NVIDIA sm_80 及更新架构，以及 Apple M3 及更新芯片。

## 演讲与文章

[Developing Svod](https://blog.vpermilp.online/en/blog/svod-intro/)，一篇博客
文章（2026 年 8 月），介绍 Svod 为何存在、其架构和路线图。

| 活动 | 演讲 | 语言 |
|---|---|---|
| [Data Fest 2026](https://ods.ai/events/df2026-31-may-online)（线上，2026 年 5 月 31 日） | 在 Svod 上实现最快的 Sber GigaAM 推理 | 俄语 |
| [RustCon 2025](https://rustcon.ru/morok-minimalistichnyy-deep-learning-freymvork-na-rust)（莫斯科，2025 年 11 月） | Morok：一个极简的 Rust 深度学习框架 | 俄语 |
| [Stereo Data Ёлка 2025](https://ods.ai/events/data-elka-2025-vk-offline-spb)（圣彼得堡，2026 年 1 月） | Rust 中的机器学习 | 俄语 |

Morok 是 Svod 之前的名字。

## 工作区

| Crate | 职责 |
|---|---|
| [`dtype`](https://github.com/npatsakula/svod/tree/main/dtype/) | 标量、向量、指针和图像类型，包括 bf16 和 fp8 |
| [`ir`](https://github.com/npatsakula/svod/tree/main/ir/) | 带哈希合并、符号整数和来源信息的 UOp 图 IR |
| [`macros`](https://github.com/npatsakula/svod/tree/main/macros/) | `patterns!` 重写 DSL 与 `jit_wrapper!` |
| [`schedule`](https://github.com/npatsakula/svod/tree/main/schedule/) | RANGEIFY、重写遍、启发式与 BEAM、Z3 验证 |
| [`codegen`](https://github.com/npatsakula/svod/tree/main/codegen/) | LLVM IR（CPU、AMDGPU、NVPTX）、C 和 MSL 渲染器 |
| [`device`](https://github.com/npatsakula/svod/tree/main/device/) | 缓冲区、分配器、mmap、KFD、CUDA 和 Metal 驱动、硬件图 |
| [`runtime`](https://github.com/npatsakula/svod/tree/main/runtime/) | 内核编译、缓存、执行计划和性能分析器 |
| [`tensor`](https://github.com/npatsakula/svod/tree/main/tensor/) | 惰性张量 API、`nn` 模块和内存规划器 |
| [`tk`](https://github.com/npatsakula/svod/tree/main/tk/) | Tile 内核 DSL 与内核库 |
| [`onnx`](https://github.com/npatsakula/svod/tree/main/onnx/) | ONNX 导入器 |
| [`arch`](https://github.com/npatsakula/svod/tree/main/arch/) | 宿主端解码器、VAD 分段和音频流水线 |
| [`model`](https://github.com/npatsakula/svod/tree/main/model/) | 预训练模型与示例 |

## 使用这个库

模型可以串联成流水线。下面是用 GigaAM 做长音频俄语语音识别、由 FireRedVAD 分段
的例子：

```rust
let model = GigaAm::from_hub_with_revision("vpermilp/GigaAM-v3", "ctc")?;
let bounds = EncoderBounds {
    sample_rate: model.config.sample_rate as u32,
    hop_length: model.config.hop_length,
    subsampling_factor: model.config.subsampling_factor,
    max_mel_frames: model.config.max_mel_frames,
    recommended_target_secs: model.recommended_chunk_secs(),
};
let splitter = FireRedVadSplitter::from_hub(&bounds)?;
let mut asr = Asr::assemble(splitter, |max_chunk| GigaAmTranscriber::new(model, opts, max_chunk))?;
let result = asr.transcribe_default(&waveform)?;
```

任何 ONNX 模型都可以编译一次、重复重放。计算图基于你的输入张量进行追踪，
因此写入其中的新数据在每次重放时都可见：

```rust
let proto = ModelProto::decode(std::fs::read("model.onnx")?.as_slice())?;
let input = Tensor::from_ndarray(&first_batch); // [1, 3, 224, 224] f32

let OnnxModel { outputs, .. } = OnnxImporter::new().import_model_with_inputs(
    proto,
    HashMap::from([("input".to_string(), input.clone())]),
    &[("batch", 1)],
)?;

let plan = Tensor::prepare_batch(outputs.values())?; // compile once
plan.execute()?;

for batch in batches {
    input.array_view_mut::<f32>()?.as_slice_mut().unwrap().copy_from_slice(&batch);
    plan.execute()?; // replay: no tracing, no compilation, no allocation
}
```

## 构建 {#building}

Nix flake 锁定了每一个编译器和库的版本，CI 使用同一个 flake：

```bash
nix develop      # development shell
nix flake check  # the CI suite: clippy, nextest (with Z3 and proptest), fmt
```

不使用 Nix 时需要以下依赖：

| 依赖 | 版本 | 必需 | 用途 |
|---|---|---|---|
| Rust | 1.88+ | 是 | Edition 2024 |
| LLVM | ≥ 16 | 是 | CPU 代码生成；`libLLVM` 在运行时加载 |
| Clang | — | 是 | GPU 内核编译、C 后端、`libLLVM` 缺失时的回退 |
| protobuf、pkgconf、zlib、libffi、libxml2 | — | 是 | ONNX 协议与 LLVM 工具链 |
| Z3 | ≥ 4.15 | 否 | 重写验证（`--features z3`） |
| NVIDIA 驱动 | CUDA ≥ 12.0（R525） | 否 | CUDA 后端 |
| amdgpu 内核驱动（KFD） | — | 否 | AMD 后端 |

```bash
cargo test --workspace
cargo test --workspace --features z3,proptest
```

`SVOD_THREADS` 设置用于编译内核和运行 CPU 内核的统一线程预算。

## 路线图

- **AOT 编译：**序列化优化后的图和已编译的内核，使模型即时启动，并能在没有编译器
  的环境（例如 WASM）中运行。
- **数据分析原语：**面向 k-means、kNN、PCA、SVD、(H)DBSCAN、UMAP 和 t-SNE 的
  FlashAttention 风格 GPU 内核，覆盖所有后端。k-means 和 kNN 已随 `tk` 提供。
- **生成代码的形式化验证：**带注解的 C 输出，证明不存在越界访问和有损转换。
- **更多硬件：**服务器（MI300–MI450、H100–B200）、消费级（Ryzen AI、Apple M3–M5、
  RTX 30–50）和嵌入式（Snapdragon X、RK3588）目标统一在一个张量 API 之后，外加一个
  不依赖 AMD 软件栈的用户态 AMD 驱动。

## 许可证

[MIT](https://github.com/npatsakula/svod/tree/main/LICENSE)
