---
sidebar_label: Metal
---

# The Metal Backend

Svod runs on Apple GPUs through Metal. The backend is written against the
Objective-C runtime directly: `libobjc`, `Metal.framework` and the private
`MTLCompiler.framework` are `dlopen`ed at run time and every call is an
`objc_msgSend` with a hand-declared C signature (`device/src/metal/objc.rs`).
There is no `objc2` or `metal` crate, no cargo feature and no `cfg(target_os)`
gate: the module compiles and type-checks on every host, and a Linux box simply
fails the `dlopen` and never registers the device. Kernels are rendered as
Metal Shading Language by the C renderer's Metal dialect
(`codegen/src/c/metal.rs`) and compiled to a metallib in process.

The code lives in `device/src/metal/` (device, allocator, compile, program,
graph, Metal 4 profiler), `runtime/src/devices/metal.rs` (the device factory)
and `codegen/src/c/metal.rs` (the dialect).

---

## Status

The backend landed in September 2026 and has been exercised on one hardware
family: **Apple9** (M3/M4 class) under macOS 26, where the tensor suite, the
ONNX suite and the `tk` hardware tests are green, and flash attention and GEMM
run through `simdgroup_matrix`. The paths written for older systems (the public
`newLibraryWithSource:` compile fallback, the pre-Apple9 indirect-command-buffer
workaround, the `metal3.x` / `metal2.0` language standards) are implemented but
not validated on such hardware. Hardware tests self-skip when no Metal device is
present, so Linux CI runs only the host-side tests.

Only the system default device (`METAL:0`) is supported; `MTLCopyAllDevices`
enumeration is a follow-up (`device/src/metal/device.rs`).

---

## Selecting the device

`METAL[:N]` is the only spelling (`HIP`-style aliases exist for AMD and CUDA,
none for Metal). On macOS it is also the **platform default**: `default_device()`
resolves to `METAL:0` when nothing selects a device, so `SVOD_DEVICE=CPU` is
how a Mac opts back into the CPU backend (`dtype/src/default_device.rs`).

```bash
SVOD_DEVICE=METAL:0 cargo run --release -p svod-model --example gigaam_infer -- ./audio.wav
```

`svod_device::metal::has_devices()` loads the Objective-C runtime and calls
`MTLCreateSystemDefaultDevice`; the runtime's device registry registers the
`"METAL"` factory only when that succeeds. Opening logs one `info` line with the
device name and GPU family (`RUST_LOG=svod_device=info`).

The family is probed with `supportsFamily:` from Apple12 down to Apple1, then
Mac2, and kept as `MetalFamily { Unknown, Mac2, Apple(n) }`. It is the
renderer's `gpu_arch`, keys the object cache and selects the optimizer profile
(`OptimizerRenderer::for_metal_family`): `simdgroup_matrix` tensor cores need
Apple7 or newer.

---

## Codegen: the MSL dialect

`CRenderer::metal()` is the CPU C renderer with `CDialect::Metal`; the Clang
output is unchanged by its existence. A kernel renders as

```c
#include <metal_stdlib>
using namespace metal;

kernel void r_64_32(device float* data0, device float* data1, constant int& data2,
                    uint3 gid [[threadgroup_position_in_grid]],
                    uint3 lid [[thread_position_in_threadgroup]]) {
  threadgroup __attribute__((aligned(16))) float local0[32];
  ...
}
```

| Concept | Clang dialect | Metal dialect |
|---|---|---|
| buffer parameter | `float* restrict data0` | `device float* data0` |
| scalar parameter | `const int data2` | `constant int& data2` |
| launch ids | `core_id` variable | `gid.xyz` (`gidx*` / `idx*`), `lid.xyz` (`lidx*`), appended after the PARAM list |
| local buffer | stack array | `threadgroup __attribute__((aligned(16))) T localN[size]` |
| barrier | none | `threadgroup_barrier(mem_flags::mem_threadgroup)` |
| address spaces | none | `device` / `threadgroup` / `thread` on pointer casts |
| 16-bit floats | `_Float16` | `half`, `bfloat` |
| bitcast | union / memcpy | `as_type<T>()` |

There are no `[[buffer(n)]]` attributes: Metal binds arguments **positionally**,
so the binding index of a parameter is its position in the signature, which the
loader mirrors (below). At most three grid axes exist; the scheduler folds
further global axes (`global_max` in the Metal optimizer profile).

**Types.** Float64, every fp8 format and vectors wider than 4 are rejected at
render (`reject_unsupported_metal_dtypes`, `codegen/src/c/types.rs`); the
scheduler demotes internal f64 to f32 beforehand. bf16 arithmetic promotes
through `float`, and bf16 narrowing uses the integer round-to-nearest-even
pattern set.

**Math.** `sqrt`, `exp2` and `log2` are native; `sin` renders as `precise::sin`;
`exp`, `log`, `cos`, `tan` and `erf` (MSL has no `erf`) are decomposed by the
shared `amd_decomposition_patterns()` over native `exp2`/`log2`, as on AMD. The
renderer's `extra_matcher` is the CPU one (`cpu_extra_matcher()`). Fast math is
off everywhere (`-fno-fast-math`, or `MTLMathModeSafe` on the public path) so the
shared test tolerances hold.

**Tensor cores.** `Wmma` lowers to a per-shape helper over `simdgroup_<T>8x8`
and `simdgroup_multiply_accumulate`: one shape, 8×8×8 over 32 threads with two
elements per lane, for f32→f32, f16→f32, f16→f16, bf16→f32 and bf16→bf16
(`METAL_888` in the optimizer profile). `tk` adds `simd_shuffle`,
`simd_shuffle_xor` and `simdgroup_barrier` builders (`codegen/src/c/metal.rs`),
which is what the Apple flash-attention and GEMM kernels are built from.

---

## Compile path

`compile_msl` (`device/src/metal/compile.rs`) sends the source to Apple's
private `MTLCodeGenService` — the same path tinygrad uses — and receives a
metallib (`MTLB` magic, `ENDT` trailer) through a hand-built Objective-C block
callback, with a 60 s timeout and one request at a time. The flags are

```text
-fno-fast-math -std=<std> --driver-mode=metal -x metal -fno-caret-diagnostics
-fmodules-cache-path=<cache>/metal-modules
```

where `<std>` follows the macOS major version (`metal4.0` on 26+, `metal3.1`
on 14–25, `metal3.0` on 13, `macos-metal2.0` earlier) and the module cache turns
the `metal_stdlib` parse from about 250 ms into about 8 ms.

`MTLCompiler.framework` loads its own libLLVM `RTLD_GLOBAL`, which cannot coexist
with the CPU backend's in-process libLLVM, so the two contend for one slot
(`claim_inprocess_llvm`). The loser of that race, or a system without the private
framework, takes `compile_msl_public`: one `newLibraryWithSource:options:error:`
compile so diagnostics surface, after which the **MSL source itself** is the
payload and the program loader compiles it again at load. Both payloads share
one object-cache entry:

```text
backend:             metal
target_architecture: Apple9/air64
toolchain:           macos=26.0
flags:               -fno-fast-math -std=metal4.0 --driver-mode=metal -x metal -fno-caret-diagnostics
abi:                 msl-kernel-abi-v1
object_format:       metallib-or-msl-v1
```

The transport is deliberately absent from the identity: a BEAM worker that won
the libLLVM slot and its parent that lost it must agree on the key.

---

## Programs and launches

`MetalProgram::load` accepts either payload (`newLibraryWithData:` for a
metallib, `newLibraryWithSource:` for MSL), binds the function with
`newFunctionWithName:` and builds the pipeline with
`setSupportIndirectCommandBuffers:YES`, reading `maxTotalThreadsPerThreadgroup`,
`threadExecutionWidth` and `staticThreadgroupMemoryLength`.

Arguments bind by position: buffers with `setBuffer:offset:atIndex:` — a host
pointer resolves to its `(MTLBuffer, offset)` through the device's
`PointerRegistry`, a `BTreeMap` keyed by buffer base — and scalars with
`setBytes` as 4-byte `i32`; a value outside `i32` is a run-time error. ABI slots
must ascend and may have gaps, up to 31 bindings (`MAX_BUFFER_BINDINGS`).
`global_size` is the threadgroup count and `local_size` the threads per group,
sent through `dispatchThreadgroups:threadsPerThreadgroup:`; a kernel with no
local axes runs one thread per group, and a group over
`maxTotalThreadsPerThreadgroup` is rejected.

Each dispatch is one command buffer, labelled with the kernel name, on a single
queue of depth 1024. With `wait = false` it joins the device's `in_flight` list;
`MetalDevice::synchronize` waits every entry with `waitUntilCompleted` and
surfaces the first `NSError`. Host access to a buffer drains the device first.
`execute_timed` reads the command buffer's `GPUStartTime` / `GPUEndTime`, which
is what BEAM ranks candidates on.

---

## Memory

Every allocation is one `MTLBuffer` in `MTLResourceStorageModeShared`: Apple
silicon is unified memory, so `BufferSpec` flags are ignored and
`copyin` / `copyout` / `_transfer` are host `memcpy` / `memmove` on `contents`
after a `synchronize()`. There are no private or managed buffers and no blit
encoders. A free drains the device first; if the drain fails the allocation is
leaked rather than released under an in-flight kernel.

---

## Graphs

`MetalGraph::capture` records a kernel chain into one `MTLIndirectCommandBuffer`
of `ConcurrentDispatch` commands, each with `setBarrier`, so capture order is
preserved. A replay is one command buffer: `useResources:count:usage:` on the
bound buffers, then `executeCommandsInBuffer:withRange:`. `replay` waits the
previous replay and rebinds only the slots whose buffer changed. Capture
declines (`Ok(None)`, per-call dispatch instead) for an empty chain, a program
that is not a `MetalProgram`, a device whose name contains "virtual"
(paravirtualized CI GPUs break ICBs), an offset over 32 bits, or **any scalar
argument** — a chain with symbolic shapes is not graphed. Below Apple9 the
tinygrad `FIX_METAL_ICB` workaround (one empty dispatch per pipeline) is applied.

---

## Profiling

| Tier | On Metal | Source |
|---|---|---|
| 1 — device time | yes | `GPUStartTime` / `GPUEndTime` per command buffer; inside a graph, a Metal 4 counter heap (macOS 26+) or one command buffer per kernel |
| 2 — roofline | yes | backend-neutral |
| 3 — static resources | partial | `lds_bytes` from `staticThreadgroupMemoryLength`, `wave_size` from `threadExecutionWidth`, `occupancy` as `maxTotalThreadsPerThreadgroup / 1024`; no register counts |
| 4 — hardware counters | no | |

`Mtl4Profiler` (`device/src/metal/mtl4.rs`) exists only for profiled graph
replay: it runs each indirect command alone in a Metal 4 command buffer between
two precise timestamps, with a residency set over the bound buffers and a shared
event wait. An MTL4 encoder silently skips the first execution of an indirect
command whose pipeline it was not given, so every pipeline is set on the encoder
first.

---

## Limitations

- one device (`METAL:0`), one command queue, shared storage only;
- scalars are `i32`; no f64, no fp8, no vectors wider than 4;
- one tensor-core shape (`simdgroup` 8×8×8);
- graphs exclude chains with scalar arguments;
- no hardware counters, no register counts, and per-dispatch timing outside a
  graph is a whole-command-buffer stamp;
- the fast path is Apple's undocumented `MTLCodeGenService`; the public API is
  the fallback.

There are no Metal-specific environment variables. The shared ones apply:
`SVOD_DEVICE`, `SVOD_OBJECT_CACHE` / `SVOD_OBJECT_CACHE_DIR`, `XDG_CACHE_HOME`
for the module cache, and `RUST_LOG=svod_device=debug` for graph capture and
decline messages.

---

## Tests

```bash
cargo test -p svod-device metal          # host tests everywhere; hardware tests self-skip
cargo test -p svod-codegen metal         # MSL golden tests
SVOD_DEVICE=METAL:0 cargo test -p svod-tensor   # codegen_tests! `metal` variants
SVOD_DEVICE=METAL:0 cargo test -p svod-onnx
```
