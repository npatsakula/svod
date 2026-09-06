---
sidebar_label: 代码生成
---

# 代码生成：NVPTX target

CUDA 后端复用了 LLVM 文本渲染器，只是多了第三个 target，
`LlvmTarget::Nvptx(CudaArch)`（`LlvmTextRenderer::nvptx(arch)`）。与 AMD 的
发射器一样，`codegen/src/llvm/nvptx/` 组合在 CPU 发射器之上：它拦截那些
NVPTX 后端无法为其通用 LLVM 形式做指令选择的操作（`Special`、`Barrier`、
LOCAL 缓冲区、`Log2`、`Wmma`、fp8 cast），而让其余一切（ALU、INDEX、LOAD、
STORE、CAST、RANGE）原样穿过。这张降低表是用 clang 22 与 `ptxas` 13.3 在
`sm_86` 上核实过的。

---

## 模块形态

```llvm
; ModuleID = 'r_64_32'
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()

define ptx_kernel void @r_64_32(ptr addrspace(1) %data0, ptr addrspace(1) %data1) #0 {
entry:
  ...
  ret void
}

attributes #0 = { nounwind "no-builtins" "no-trapping-math"="true" "nvvm.maxntid"="32" }
```

- 单靠 `ptx_kernel` 就能产出一个 `.visible .entry`；不需要任何
  `!nvvm.annotations`。
- `target datalayout` 是 clang 22 对 `nvptx64` 的默认值；clang 会静默地覆盖
  不匹配的值，所以这一行的存在是为了那些独立读取该模块的工具
  （`opt`、`llvm-as`、IR 转储）。
- `"nvvm.maxntid"` 就是 PTX 的 `.maxntid` **launch bound**，被设为该内核最大的
  local size：`ptxas` 会照着它而不是 1024 线程的最坏情况来给每线程分配
  寄存器预算。更老的 LLVM 会忽略这个字符串属性，只是丢掉这条提示而已。

| 概念 | AMD | NVPTX |
|---|---|---|
| triple | `amdgcn-amd-amdhsa` | `nvptx64-nvidia-cuda` |
| 内核 ABI | `amdgpu_kernel`，`.kd` 描述符 | `ptx_kernel` |
| 工作 id | `llvm.amdgcn.workgroup.id.*` / `workitem.id.*` | `llvm.nvvm.read.ptx.sreg.ctaid.{x,y,z}` / `tid.{x,y,z}` |
| barrier | `fence syncscope("workgroup")` + `s.barrier` | `fence syncscope("block") release; llvm.nvvm.barrier0; fence syncscope("block") acquire`（`bar.sync 0`） |
| 地址空间 | global 1、LDS 3、private 5 | global 1、shared 3；REG 缓冲区保持为普通的通用 `alloca` |
| 共享内存 | `addrspace(3)` 模块全局量 | 同上 |
| launch bound | `"amdgpu-flat-work-group-size"` | `"nvvm.maxntid"` |

NVPTX 把工作组作用域叫做 `"block"`；`syncscope("workgroup")` 会被拒绝。
`@llvm.nvvm.barrier0` 是每一个 LLVM 发行版都会降低为 `bar.sync 0` 的那种写法
（更新的版本会自动升级它）。

---

## 快速数学与除法

渲染器在 GPU target 上把 ` nsz arcp contract afn ` 削减为 ` contract `：
NVPTX 会把 `fdiv ... arcp afn` 降低为 `rcp.approx.f32`，而单纯的 `contract`
则保住精确的 `div.rn.f32`。Tinygrad 的 CUDA 前端同样以精确除法编译。

---

## 超越函数

NVPTX 对通用的 `@llvm.{exp,log,sin,cos,pow}` intrinsic **没有降低**（指令选择
失败），并且会把 `@llvm.erf` 发射成一个只会在 `ptxas` 内部失败的外部调用。
因此渲染器把 `Exp`、`Log`、`Log2`、`Sin`、`Cos`、`Tan`、`Erf`、`Pow`、`Max`
和 `Threefry` 从它的 `supported_ops` 中移除，由调度器用
`nvptx_decomposition_patterns()` 将它们分解：AMD 的那一套（在原生 `exp2` /
`log2` 之上做多项式 `exp`/`log`/三角函数、整数域的 bf16 舍入），外加 f64 的
`Exp2`/`Log2` 展开，因为 NVPTX 只为 f16/f32 降低 `@llvm.exp2`。

保持原生的是：`@llvm.exp2.f32` 选出 `ex2.approx.f32`，`@llvm.sqrt` 选出
`sqrt.rn`，`fma`/`floor`/`rint`/`maxnum` 直接降低。

`Log2` 是刻意为之的那一例。`@llvm.log2.f32` 没有 NVPTX 降低（"no libcall
available for flog2"）；硬件路径是 `@llvm.nvvm.lg2.approx.f` →
`lg2.approx.f32`，一个 2^-22.6 的相对近似，在 AMD 的 `v_log_f32` 与 libm
都精确的地方差 1 ulp。渲染器保留了那条降低路径（`render_log2`：仅标量 f32，
f16 在其外围加宽，向量按 lane 拆分）以备显式使用，但 `Log2` 被排除在
`supported_ops` 之外，于是普通的图走多项式 `xlog2` 路径，并保持共享的测试
容差。

一个未被分解却仍抵达渲染器的超越函数，意味着能力列表出现了漂移；它会在
渲染时以一个 `InvalidGraph` 错误失败，而不是点名一个 LLVM 会静默地变成
外部调用的 intrinsic。

bool 在 PTX 中是谓词寄存器，但在内存中是字节，因此 NVPTX 的
`extra_matcher` 就是 CPU 渲染器的 `bool_storage_patterns`（tinygrad 的
`ptx_matcher`）。优化器 profile 在它的缓存键中记录了
`nvptx-decomposition-v1` 与 `llvm-nvptx-extra-v1`，因此这些选择绝不会与另一个
后端的内核相撞。

---

## Tensor core：`mma.sync`

`Wmma` 降低为一个由 `resolve_mma(arch, in_dtype, acc_dtype, (N, M, K))` 选出的
`@llvm.nvvm.mma.*` intrinsic。每个 CUDA profile 都是一个 `m16n8kK` 形状；
PTX ISA 为每一行规定了最低计算能力：

| 输入 → 累加器 | K | intrinsic 后缀 | 最低 |
|---|---|---|---|
| f16 → f32 / f16 | 8 | `m16n8k8.row.col.f32.f32` / `.f16.f16` | `sm_75` |
| f16 → f32 / f16 | 16 | `m16n8k16.row.col.f32.f32` / `.f16.f16` | `sm_80` |
| bf16 → f32 | 16 | `m16n8k16.row.col.bf16` | `sm_80` |
| tf32（原始 f32 位）→ f32 | 8 | `m16n8k8.row.col.tf32` | `sm_80` |
| int8 → int32 | 32 | `m16n8k32.row.col.satfinite.s8` | `sm_80` |
| e4m3 / e5m2 → f32 | 32 | `m16n8k32.row.col.f32.e4m3.e4m3.f32` / `.e5m2...` | `sm_89` |

任何别的元组，或者一个低于最低要求的 arch，都会返回 `None`，于是调用方抛出
`InvalidGraph`，让优化器在上游做分解。片段遵循 PTX 的寄存器切分（A 是
16×K，B 是 K×8，C/D 是 16×8，全都摊在 32 个 lane 的 32 位寄存器上）：f16
操作数以 `<2 x half>` 对的形式传递，bf16 / tf32 / int8 / fp8 以 `i32` 字传递，
f32 累加器以 `float` 传递；聚合结果会被重新组装回 WMMA 天然的向量。匹配的
`declare` 行由每个调用点的操作数类型合成而来
（`wmma_declaration_from_call`），与 AMD 的 WMMA/MFMA intrinsic 是同一套机制。

两个面向带类型 `CUSTOM` 节点的 warp 级构造器补全了这一套：
`shfl_bfly(value, lane_mask)`（`llvm.nvvm.shfl.sync.bfly.i32`，warp 归约的
蝶形步）与 `globaltimer()`（`llvm.nvvm.read.ptx.sreg.globaltimer`，纳秒级的
GPU 时钟）。

---

## 编译到 PTX

`compile_ir_to_ptx`（`runtime/src/cuda/compile.rs`）把 IR 通过管道灌进宿主的
clang，从 stdin 到 stdout，与 AMD 和 CPU 路径完全一样：

```text
clang -x ir -S -O3 --target=nvptx64-nvidia-cuda -march=sm_86 -Wno-override-module - -o -
```

一个被缓存的 `has_nvptx_target()` 探测（`clang --print-targets`）会把一个不带
NVPTX 的 clang 变成一个干净的 `JitCompilation` 错误。
`SVOD_DUMP_NVPTX_IR=<dir>` 会把每个内核的 `.ll` 写到那里。

在任何 PTX 抵达驱动之前——无论是新鲜编译的还是来自对象缓存的——
`validate_ptx` 都会检查它有一个 `.version`、一个等于设备 `sm_XY` 的
`.target`、一个 `.entry <kernel>(`，以及**没有 `.extern .func`**。最后这一点
很要紧：一个拼错的 `llvm.nvvm.*` 名字并不是编译错误，LLVM 会静默地把它发射
成一个外部调用，而它本来只会以一次 `cuModuleLoadDataEx` 失败的形式浮现。

PTX ISA 版本锁定在 7.8（`--cuda-feature=+ptx78`），而不是 clang 发行版会发出的那个
（clang 22：`.version 8.8`，它需要一个 CUDA 12.8 / R570 驱动）；渲染器选择的每一种
`mma.sync` 形状在 7.8 中都存在。
