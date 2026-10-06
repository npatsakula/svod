---
sidebar_label: Overview
---

# Backends

A backend is everything below the rendered kernel: a renderer that turns the
UOp IR into source, a compiler that turns the source into an object, a loader
that turns the object into a callable `Program`, an allocator, and optionally
a graph. Svod ships four, all in the same binary; which ones exist on a given
host is decided at run time.

| Device | Hardware | Renderer | Compile path | Graph replay | Status |
|---|---|---|---|---|---|
| [`CPU`](./cpu.md) | x86_64, aarch64, riscv64, loongarch64, ppc64le | LLVM IR text (default) or C | libLLVM in process, else `clang -c`; [in-memory ELF loader](./jit-loader.md) | none (synchronous calls) | production |
| [`AMD:N`](./amd/overview.md) | CDNA3, RDNA3, RDNA3.5, RDNA4 (Linux, KFD) | LLVM IR text, AMDGPU target | `clang` with the `amdgcn` target → ELF code object loaded into VRAM | AQL command stream (PM4 opt-in) | production |
| [`CUDA:N`](./cuda/overview.md) | NVIDIA, driver CUDA 12.0+ | LLVM IR text, NVPTX target | `clang` with the NVPTX target → PTX, `ptxas` when installed, else driver JIT | CUDA graphs | production |
| [`METAL:N`](./metal.md) | Apple GPUs | C, Metal dialect | Apple's `MTLCodeGenService` in process → metallib | indirect command buffers | validated on Apple9 / macOS 26 |

The compiled objects of every backend go through one on-disk object cache,
keyed by the source and a per-backend `CompilerIdentity`
([CPU page](./cpu.md)).

---

## Selecting a device

`SVOD_DEVICE` picks the default device for tensors and kernels. The value is
parsed case-insensitively as `NAME[:N]` (`dtype/src/default_device.rs`):

| Value | Device |
|---|---|
| `CPU` | `DeviceSpec::Cpu` |
| `AMD[:N]`, `HIP[:N]` | `DeviceSpec::Amd { device_id }` — the N-th GPU node of the KFD topology |
| `CUDA[:N]`, `GPU[:N]` | `DeviceSpec::Cuda { device_id }` |
| `METAL[:N]` | `DeviceSpec::Metal { device_id }` (only `0` exists) |

`NAME` alone is device 0. `NV` is rejected on purpose — the name is reserved
for a future userspace NVIDIA driver. When nothing selects a device the
platform default applies: **`METAL:0` on macOS, `CPU` everywhere else**. The
full precedence is a `with_default_device` scope, then a thread-local
`set_default_device`, then `SVOD_DEVICE` (read once per process), then the
platform default. The GPU arch is never part of the spec: it is a property of
the opened device, so one physical GPU has one identity and the kernel cache
is keyed by what the device reports.

`DeviceSpecExt::parse` in `svod-device` accepts the same spellings plus
`DISK:<path>` (a read-only, memory-mapped file device that cannot run kernels)
and `WEBGPU`, which has no allocator yet and fails with `DeviceUnavailable`.

---

## Runtime-detected registration

Every backend is compiled on every host — there is no cargo feature for AMD,
CUDA or Metal. The CUDA and Metal bindings are plain Rust over `libloading`
and compile everywhere; the AMD kernel-facing modules are `cfg(unix)`. A Linux
or macOS `cargo check` therefore type-checks all of them. Whether a backend is
*available*
is decided when the device factory registry is first touched
(`runtime/src/device_registry.rs`):

```rust
registry.register_factory("CPU", ...);                        // always
if svod_device::amd::has_devices()   { registry.register_factory("AMD", ...); }
if svod_device::metal::has_devices() { registry.register_factory("METAL", ...); }
if svod_device::cuda::has_devices()  { registry.register_factory("CUDA", ...); }
```

Each probe is side-effect-free and memoized: AMD reads the KFD sysfs topology
and asks whether any node is a supported arch; Metal `dlopen`s the Apple
frameworks and asks for the system default device; CUDA loads `libcuda.so.1`,
binds every entry point it uses, calls `cuInit` and counts devices. A host
without the hardware simply has no such device type, and asking for one fails
with `UnsupportedDevice`. The point of compiling everything everywhere is that
a change to the shared `Program` / `PlanContext` / `Graph` traits breaks the
build on any developer machine, not only on the one with the GPU.

The registry caches one `Device` per `DeviceSpec` (`DEVICE_FACTORIES`);
construction — opening KFD, probing the toolchain — runs outside the map locks,
serialized per spec, and a failed construction leaves the slot empty for a
retry. Allocators live in a separate registry in `svod-device`
(`registry::registry()`), where every compute allocator is wrapped in an
`LruAllocator` that pools freed buffers by size and spec.

---

## What a backend implements

A `Device` (`device/src/device.rs`) is five parts:

```rust
pub struct Device {
    pub device: DeviceSpec,
    pub allocator: Arc<dyn Allocator>,
    pub compilers: Vec<CompilerPair>,     // (Arc<dyn Renderer>, Arc<dyn Compiler>)
    pub renderer: Arc<dyn Renderer>,
    pub compiler: Arc<dyn Compiler>,
    pub runtime: RuntimeFactory,          // Fn(&CompiledSpec) -> Result<Box<dyn Program>>
    pub graph: Option<GraphFactory>,      // Fn(&[GraphKernel]) -> Result<Option<Box<dyn Graph>>>
}
```

| Trait | Required | Role |
|---|---|---|
| `Renderer` | `render`, `device`, `supported_ops` | UOp graph → `ProgramSpec` (source, entry, ABI, launch sizes). `gpu_arch` picks the optimizer profile; `decompositor` and `extra_matcher` lower what the target cannot select |
| `Compiler` | `compile`, `cache_key` | `ProgramSpec` → `CompiledSpec` bytes; `cache_key` is the `CompilerIdentity` that keys the object cache |
| `RuntimeFactory` | — | loads a `CompiledSpec` into a `Program`; `Device::new` wraps it so every spec's stage identity is validated first |
| `Program` | `execute`, `name` | one kernel launch; `execute_timed` (GPU-clock duration for BEAM), `new_exec_context`, `resource_usage` and `as_any` are optional |
| `PlanContext` | `dispatch`, `synchronize` | per-plan state minted by `Program::new_exec_context`: lanes, completion tokens, timestamps, counters (`set_pmc`), native linked replay (`replay_linked_plan`) |
| `Allocator` | `_alloc`, `name`, `device_spec` | `_copyin` / `_copyout` / `_transfer` / `_free` / `synchronize` / `supports_device_local` are optional and default to host-memory semantics |
| `Graph` | `replay` | a captured kernel chain replayed with one submission; `completion_token`, `replay_profiled` optional |
| `CompletionToken`, `TimelineSignal`, `DispatchTimestamps` | | the synchronization and profiling handles the executor consumes (`device/src/sync.rs`) |

The launch convention is shared by every GPU backend: `global_size` is the
grid in work-groups, `local_size` the work-group in threads; the CPU uses
`global_size[0]` as the `core_id` split. Kernel arguments are the ABI's
`PARAM` slots in order — pointers, then `i32` scalars — which every loader
packs the same way (`ClikeKernargLayout` on AMD and CUDA, positional
`setBuffer`/`setBytes` on Metal, a libffi CIF on the CPU).

### Adding a backend

The four factories in `runtime/src/devices/` are the template. Each
`create_*_device` does the same five things:

1. gets the allocator from the registry for its `DeviceSpec`;
2. builds a renderer wrapper around a codegen entry point
   (`LlvmTextRenderer::amd(arch)`, `LlvmTextRenderer::nvptx(arch)`,
   `CRenderer::metal()`, or the CPU renderers), declaring `supported_ops`,
   the decomposition patterns and the `gpu_arch`;
3. builds a compiler with a `CompilerIdentity` and an `ObjectCache`, producing
   bytes the loader can validate (`validate_amd_object`, `validate_ptx` /
   `validate_cubin`, `validate_metallib`, the ELF checks on the CPU);
4. installs a `RuntimeFactory` that loads those bytes into the backend's
   `Program`;
5. optionally `with_graph(...)` for capture/replay.

Then it registers the factory under its device-type string, gated on a
`has_devices()` probe, and the scheduler needs an optimizer profile for the
new target (`OptimizerRenderer::for_*`: wave size, tensor-core shapes,
shared-memory and local limits). `create_*_codegen` exists separately on
every backend so BEAM workers can render and compile without opening the
device.
