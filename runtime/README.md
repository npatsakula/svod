# svod-runtime

The execution layer of the Svod ML compiler. It builds a `Device` per
`DeviceSpec` (renderer, compiler, program loader, allocator), compiles rendered
kernels, loads them, and runs prepared kernels and copies in dependency order
through `ExecutionPlan`. It also owns the on-disk object cache, BEAM
benchmarking and the kernel profiler. Most users reach it through `svod-tensor`.

## Example

```rust
use svod_dtype::DeviceSpec;
use svod_runtime::DEVICE_FACTORIES;

// One `Device` per spec, created on first request and cached.
let cpu = DEVICE_FACTORIES.device(&DeviceSpec::Cpu, svod_device::registry::registry())?;
assert_eq!(cpu.device.canonicalize(), "CPU");
```

## Backends

| Device | Compile | Dispatch |
|--------|---------|----------|
| `CPU` | LLVM IR through in-process libLLVM (default), else `clang -x ir`; `SVOD_CPU_BACKEND=clang` renders C for `clang -c` | in-memory ELF loader, libffi call; `core_id`-split kernels run on rayon |
| `AMD:N` | `clang --target=amdgcn-amd-amdhsa` → ELF code object | KFD queues |
| `CUDA:N` | `clang --target=nvptx64-nvidia-cuda` → PTX, `ptxas` when installed, else driver JIT | runtime-loaded `libcuda.so.1`, streams and CUDA graphs |
| `METAL:N` | MSL → metallib through `MTLCodeGenService` | `MTLCommandQueue` |

The CPU factory is always registered; the GPU factories are registered only
when the hardware is present. `SVOD_DEVICE` selects the default device;
without it the default is `METAL:0` on macOS and `CPU` elsewhere.

## Environment variables

| Variable | Effect |
|----------|--------|
| `SVOD_CPU_BACKEND` | `llvm` (default) or `clang` |
| `SVOD_LLVM_LIB` | path of the libLLVM to bind; otherwise `llvm-config --libdir`, the loader path, then Homebrew kegs |
| `SVOD_LLVM_INPROCESS=0` | always compile LLVM IR with the `clang` subprocess |
| `SVOD_THREADS` | thread budget and default CPU kernel split (default: host parallelism) |
| `SVOD_OBJECT_CACHE=0` / `SVOD_OBJECT_CACHE_DIR` | disable / relocate the on-disk object cache |
| `SVOD_CUDA_PTXAS=0` | hand PTX to the driver JIT even when `ptxas` is installed |
| `SVOD_DUMP_AMD_IR` / `SVOD_DUMP_NVPTX_IR` | directory that receives each kernel's LLVM IR |

The `dlopen-fallback` feature loads C-path kernels as shared libraries through
`dlopen` instead of the in-memory ELF loader.

Documentation:

- CPU backend: <https://svod.vpermilp.online/docs/backends/cpu>
- Backends: <https://svod.vpermilp.online/docs/backends/overview>
- JIT loader: <https://svod.vpermilp.online/docs/backends/jit-loader>
