---
sidebar_label: Codegen
---

# Codegen: the NVPTX target

The CUDA backend reuses the LLVM text renderer with a third target,
`LlvmTarget::Nvptx(CudaArch)` (`LlvmTextRenderer::nvptx(arch)`). Like the AMD
emitter, `codegen/src/llvm/nvptx/` composes over the CPU emitter: it
intercepts the ops whose generic LLVM form the NVPTX backend cannot select
(`Special`, `Barrier`, LOCAL buffers, `Log2`, `Wmma`, fp8 casts) and lets
everything else (ALU, INDEX, LOAD, STORE, CAST, RANGE) fall through unchanged.
The lowering table was verified with clang 22 and `ptxas` 13.3 at `sm_86`.

---

## Module shape

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

- `ptx_kernel` alone yields a `.visible .entry`; no `!nvvm.annotations` are
  needed.
- The `target datalayout` is clang 22's default for `nvptx64`; clang overrides
  a mismatch silently, so the line exists for tools that read the module
  standalone (`opt`, `llvm-as`, IR dumps).
- `"nvvm.maxntid"` is the PTX `.maxntid` **launch bound**, set to the kernel's
  largest local size: `ptxas` budgets registers per thread against it instead
  of the 1024-thread worst case. An older LLVM ignores the string attribute
  and merely loses the hint.

| Concept | AMD | NVPTX |
|---|---|---|
| triple | `amdgcn-amd-amdhsa` | `nvptx64-nvidia-cuda` |
| kernel ABI | `amdgpu_kernel`, `.kd` descriptor | `ptx_kernel` |
| work ids | `llvm.amdgcn.workgroup.id.*` / `workitem.id.*` | `llvm.nvvm.read.ptx.sreg.ctaid.{x,y,z}` / `tid.{x,y,z}` |
| barrier | `fence syncscope("workgroup")` + `s.barrier` | `fence syncscope("block") release; llvm.nvvm.barrier0; fence syncscope("block") acquire` (`bar.sync 0`) |
| address spaces | global 1, LDS 3, private 5 | global 1, shared 3; REG buffers stay a plain generic `alloca` |
| shared memory | `addrspace(3)` module globals | same |
| launch bound | `"amdgpu-flat-work-group-size"` | `"nvvm.maxntid"` |

NVPTX names the work-group scope `"block"`; `syncscope("workgroup")` is
rejected. `@llvm.nvvm.barrier0` is the spelling every LLVM release lowers to
`bar.sync 0` (newer ones auto-upgrade it).

---

## Fast math and division

The renderer strips ` nsz arcp contract afn ` to ` contract ` on GPU targets:
NVPTX lowers `fdiv ... arcp afn` to `rcp.approx.f32`, whereas plain `contract`
keeps the exact `div.rn.f32`. Tinygrad's CUDA frontend compiles with exact
division as well.

---

## Transcendentals

NVPTX has **no lowering** for the generic `@llvm.{exp,log,sin,cos,pow}`
intrinsics (instruction selection fails) and emits `@llvm.erf` as an external
call that only fails inside `ptxas`. The renderer therefore removes `Exp`,
`Log`, `Log2`, `Sin`, `Cos`, `Tan`, `Erf`, `Pow`, `Max` and `Threefry` from its
`supported_ops`, and the scheduler decomposes them with
`nvptx_decomposition_patterns()`: the AMD set (polynomial `exp`/`log`/trig over
native `exp2`/`log2`, integer-domain bf16 rounding) plus f64 `Exp2`/`Log2`
expansions, because NVPTX lowers `@llvm.exp2` for f16/f32 only.

What stays native: `@llvm.exp2.f32` selects `ex2.approx.f32`, `@llvm.sqrt`
selects `sqrt.rn`, `fma`/`floor`/`rint`/`maxnum` lower directly.

`Log2` is the deliberate case. `@llvm.log2.f32` has no NVPTX lowering ("no
libcall available for flog2"); the hardware path is
`@llvm.nvvm.lg2.approx.f` → `lg2.approx.f32`, a 2^-22.6 relative
approximation, 1 ulp off where AMD's `v_log_f32` and libm are exact. The
renderer keeps that lowering (`render_log2`: scalar f32 only, f16 widens
around it, vectors split per lane) for explicit use, but `Log2` is excluded
from `supported_ops` so ordinary graphs take the polynomial `xlog2` path and
keep the shared test tolerances.

An un-decomposed transcendental that still reaches the renderer is a
capability-list drift; it fails at render time with an `InvalidGraph` error
instead of naming an intrinsic LLVM would silently turn into an external call.

Bools are predicate registers in PTX but bytes in memory, so the NVPTX
`extra_matcher` is the CPU renderer's `bool_storage_patterns` (tinygrad's
`ptx_matcher`). The optimizer profile records `nvptx-decomposition-v1` and
`llvm-nvptx-extra-v1` in its cache key so these choices never collide with
another backend's kernels.

---

## Tensor cores: `mma.sync`

`Wmma` lowers to one `@llvm.nvvm.mma.*` intrinsic chosen by
`resolve_mma(arch, in_dtype, acc_dtype, (N, M, K))`. Every CUDA profile is an
`m16n8kK` shape; the PTX ISA fixes the minimum capability per row:

| Inputs → accumulator | K | Intrinsic suffix | Min |
|---|---|---|---|
| f16 → f32 / f16 | 8 | `m16n8k8.row.col.f32.f32` / `.f16.f16` | `sm_75` |
| f16 → f32 / f16 | 16 | `m16n8k16.row.col.f32.f32` / `.f16.f16` | `sm_80` |
| bf16 → f32 | 16 | `m16n8k16.row.col.bf16` | `sm_80` |
| tf32 (raw f32 bits) → f32 | 8 | `m16n8k8.row.col.tf32` | `sm_80` |
| int8 → int32 | 32 | `m16n8k32.row.col.satfinite.s8` | `sm_80` |
| e4m3 / e5m2 → f32 | 32 | `m16n8k32.row.col.f32.e4m3.e4m3.f32` / `.e5m2...` | `sm_89` |

Any other tuple, or an arch below the minimum, returns `None` and the caller
raises `InvalidGraph` so the optimizer decomposes upstream. Fragments follow
the PTX register split (A is 16×K, B is K×8, C/D 16×8, all over 32 lanes in
32-bit registers): f16 operands travel as `<2 x half>` pairs, bf16 / tf32 /
int8 / fp8 as `i32` words, f32 accumulators as `float`; the aggregate result
is reassembled into the WMMA's natural vector. The matching `declare` lines
are synthesized from each call site's operand types
(`wmma_declaration_from_call`), the same mechanism as the AMD WMMA/MFMA
intrinsics.

Two warp-level builders for typed `CUSTOM` nodes complete the set:
`shfl_bfly(value, lane_mask)` (`llvm.nvvm.shfl.sync.bfly.i32`, the butterfly
step of a warp reduction) and `globaltimer()`
(`llvm.nvvm.read.ptx.sreg.globaltimer`, the nanosecond GPU clock).

---

## Compiling to PTX

`compile_ir_to_ptx` (`runtime/src/cuda/compile.rs`) pipes the IR through the
host clang, stdin to stdout, exactly like the AMD and CPU paths:

```text
clang -x ir -S -O3 --target=nvptx64-nvidia-cuda -march=sm_86 -Wno-override-module - -o -
```

A cached `has_nvptx_target()` probe (`clang --print-targets`) turns a clang
without NVPTX into a clean `JitCompilation` error. `SVOD_DUMP_NVPTX_IR=<dir>`
writes each kernel's `.ll` there.

Before any PTX reaches the driver, fresh or from the object cache,
`validate_ptx` checks that it has a `.version`, a `.target` equal to the
device's `sm_XY`, an `.entry <kernel>(`, and **no `.extern .func`**. The last
one matters: a misspelt `llvm.nvvm.*` name is not a compile error, LLVM
silently emits it as an external call, and it would otherwise only surface as
a `cuModuleLoadDataEx` failure.

The PTX ISA version is pinned to 7.8 (`--cuda-feature=+ptx78`) rather than
whatever the clang release would emit (clang 22: `.version 8.8`, which needs
a CUDA 12.8 / R570 driver); every `mma.sync` shape the renderer selects exists
in 7.8.
