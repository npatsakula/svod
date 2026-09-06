# svod-runtime

Kernel execution interface bridging codegen to hardware.

## Example

```rust
use svod_runtime::CompiledKernel;

let kernel = compile(code)?;
kernel.execute(&[buf_a.ptr(), buf_b.ptr(), buf_out.ptr()])?;
```

## Backends

| Backend | How it works | Feature |
|---------|-------------|---------|
| **LLVM** (default) | Compiles LLVM IR in-process through `libLLVM` (falls back to `clang -x ir`), loads via JIT ELF loader | always |
| **Clang** | Compiles C via `clang -c`, loads via JIT ELF loader | always |

Select at runtime: `SVOD_CPU_BACKEND=clang|llvm` (any other value warns and keeps the LLVM default).

libLLVM is taken from `SVOD_LLVM_LIB` when set, else searched in `llvm-config --libdir`, then on the
loader's default path (dev symlink and runtime SONAMEs such as `libLLVM.so.18.1`, `libLLVM-18.so.1`),
then Homebrew kegs; `SVOD_LLVM_INPROCESS=0` forces the `clang` subprocess.

CPU kernels are split `core_id` ways and run on rayon's global pool. `SVOD_THREADS`
(default: host parallelism) is the single budget for that pool, for the parallel
kernel preparation in `svod-tensor`, and for the default kernel split;
`RAYON_NUM_THREADS` is not consulted. Compiled objects are cached on disk
(`SVOD_OBJECT_CACHE=0` disables, `SVOD_OBJECT_CACHE_DIR` relocates); entries are
published atomically and never locked — concurrent compilers of one key both
publish identical bytes, last rename wins.

## GPU backends

Always compiled, registered only when the hardware is present:

| Device | Compile | Dispatch |
|--------|---------|----------|
| `AMD:N` | `clang --target=amdgcn-amd-amdhsa` → ELF code object | KFD-direct PM4/AQL rings (`svod_device::amd`) |
| `CUDA:N` | `clang --target=nvptx64-nvidia-cuda` → PTX, assembled by `ptxas` when installed (`SVOD_CUDA_PTXAS=0` opts out), else JIT'd by `libcuda.so.1` | streams and CUDA graphs (`svod_device::cuda`) |
| `METAL:N` | MSL → `metallib` through `MTLCodeGenService` | `MTLCommandQueue` (`svod_device::metal`) |

Select with `SVOD_DEVICE=AMD:0`, `CUDA:0`, or `METAL:0`. `SVOD_DUMP_AMD_IR` /
`SVOD_DUMP_NVPTX_IR` name a directory that receives each kernel's LLVM IR.

## Testing

```bash
cargo test -p svod-runtime
```
