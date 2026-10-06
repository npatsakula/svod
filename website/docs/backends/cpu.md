---
sidebar_label: CPU
---

# The CPU Backend

The CPU backend compiles every kernel to native machine code at run time and
calls it in process. It has two code paths, selected by `SVOD_CPU_BACKEND`:

| `SVOD_CPU_BACKEND` | Renderer | Compiler | Cache backend |
|---|---|---|---|
| `llvm` (default; unset or empty selects it) | LLVM IR text (`codegen/src/llvm/cpu/`) | libLLVM bound in process, or `clang -x ir` when it will not load | `cpu-llvm-inprocess` / `cpu-llvm-clang` |
| `clang` | C source (`codegen/src/c/`, Clang dialect) | `clang -x c` subprocess | `cpu-clang` |

Only those two spellings and their upper-case forms are accepted
(`CpuBackend::parse`); anything else logs a warning and selects LLVM. Both
paths produce a relocatable ELF object, which the [JIT loader](./jit-loader.md)
maps and relocates in memory; neither writes a temporary file or `dlopen`s a
shared library. The wiring is in `runtime/src/devices/cpu.rs`.

The `"CPU"` device factory is registered unconditionally, and the CPU is the
default device everywhere except macOS, where it is `METAL:0`
(see [Backends](./overview.md)).

---

## The LLVM path

The renderer emits one LLVM function per kernel, buffers as
`ptr noalias align 32` parameters and scalars as `i32`, with the attributes
`nounwind "no-builtins" "no-trapping-math"="true"`:

```llvm
define void @r_64_32(ptr noalias align 32 %data0, ptr noalias align 32 %data1, i32 %core_id) #0 {
entry:
  ...
  ret void
}
```

### In-process libLLVM

`runtime/src/llvm_inprocess.rs` binds the LLVM-C API from a shared libLLVM with
`libloading`. The library is found once per process, in this order:

1. `SVOD_LLVM_LIB=<path>` — then that file is the only candidate;
2. `llvm-config --libdir` joined with each candidate name;
3. each candidate name through the dynamic loader's own search path;
4. on macOS, `/opt/homebrew/opt/llvm/lib/libLLVM.dylib` and the versioned
   `llvm@N` kegs.

The candidate names are `libLLVM.so` / `libLLVM.dylib`, then for each major from
30 down to 16 the distro SONAMEs (`libLLVM.so.N.1`, `libLLVM-N.so.1`,
`libLLVM-N.so`, `libLLVM.so.N`; `libLLVM-N.dylib`, `libLLVM.N.dylib`). The
version is read with `LLVMGetVersion` and must be **16 or newer**.

A compile parses the IR from a memory buffer (`LLVMParseIRInContext`), verifies
it, runs `LLVMRunPasses("default<O2>")` with loop unrolling, loop vectorization
and SLP vectorization enabled, and emits the object with
`LLVMTargetMachineEmitToMemoryBuffer`. The target machine is
`<arch>-none-unknown-elf` with the host CPU name and feature string
(`LLVMGetHostCPUName` / `LLVMGetHostCPUFeatures`), PIC relocation and the
default code model; on macOS aarch64 the feature string adds `+reserve-x18`.
Error-severity diagnostics from the context's handler fail the compile. Every
thread owns a `Session` (context, target machine, data layout, pass options) in
a thread-local, so kernels compile concurrently.

`SVOD_LLVM_INPROCESS=0` (the exact value `0`) disables the in-process path. Any
failure to bind the library — not found, too old, missing symbol — falls back to
the clang producer below; the fallback is logged at `warn` unless the variable
disabled it on purpose.

:::note[One libLLVM per process]
Apple's Metal compiler framework loads its own libLLVM `RTLD_GLOBAL`, which
cannot share a process with another copy. The CPU and [Metal](./metal.md)
backends therefore arbitrate one slot (`svod_device::claim_inprocess_llvm`):
whichever compiles first keeps it, and the other takes its subprocess or
public-API fallback.
:::

### The clang fallback

`compile_ir_to_object_with` pipes the IR through clang, stdin to stdout:

```text
clang -x ir -c -O2 -march=native -fPIC -fno-math-errno -fno-stack-protector \
      -funroll-loops -fvectorize -fslp-vectorize --target=<arch>-none-unknown-elf [-ffixed-x18] - -o -
```

The in-process and clang producers carry different object-cache identities
(`cpu-llvm-inprocess` with `library=<path>;version=x.y.z` and the host
CPU/features, versus `cpu-llvm-clang` with the clang identity), so objects
never cross producers.

---

## The clang C path

`SVOD_CPU_BACKEND=clang` renders C instead — `void name(T* restrict data0, ...,
const int dataN)` with `__builtin_*` math — and compiles it with

```text
clang -c -x c -O2 <cpu flag> -fPIC -ffreestanding -fno-math-errno -fno-stack-protector \
      -nostdlib -fno-ident --target=<arch>-none-unknown-elf [-ffixed-x18] - -o -
```

where `<cpu flag>` is `-march=native` on x86_64 and loongarch64, `-march=rv64g`
on riscv64, and `-mcpu=native` elsewhere (on ARM `-march=native` sets only the
ISA family). The source is readable, which is what the path is for: inspecting
what a kernel does without reading LLVM IR.

`clang` is resolved on `PATH` only (`ClangToolchain::discover`); there is no
`SVOD_CLANG` or `CC` override. Its identity for the object cache is
`path=...;sha256=<binary digest>;version=<clang --version>`, and the C path
additionally records the resolved `-target-cpu` and features from `clang -###`
(fingerprinted with `/proc/cpuinfo` when a flag says `native`), so an object
compiled on one machine is not reused on another CPU.

:::tip[dlopen fallback]
The `dlopen-fallback` cargo feature of `svod-runtime` replaces the ELF loader
for the **C path only**: `clang -shared ... -lm` writes `kernel.so` into a
temporary directory, which `libloading` opens. It is slower and exists for
platforms where the in-memory loader does not work; the object cache records
it as `elf-shared-dlopen-v1`, and CI runs it as the `test-dlopen-fallback`
check. The LLVM path always uses the in-memory loader.
:::

---

## Math

The CPU renderers keep the transcendentals: `exp`, `log`, `sin`, `cos`, `tan`,
`pow` and friends render as `@llvm.*` intrinsics or `__builtin_*` calls. `sqrt`,
`fma`, `floor`, `rint` and the like lower to instructions when the host ISA has
them; the rest become `libm` calls that the loader resolves with
`dlsym(RTLD_DEFAULT)` at load time.
The LLVM path removes `Erf` from its supported ops (it is decomposed), the C
path keeps it; both drop `Threefry` and `Max`, which decompose to a bare XOR and
a select.

---

## Threads

`SVOD_THREADS` (a positive integer; default: the host's available parallelism)
is the single thread budget. It sizes rayon's global pool, which compiles kernel
cache misses in parallel and runs CPU kernels, and it is the default **`core_id`
split** of every CPU kernel:

- The optimizer's THREAD opt (`schedule/src/optimizer/opts.rs`) moves one global
  loop axis to the outermost position and splits it; the heuristic tries
  32, 16, 12, 8, 6, 5, 4, 3 and 2 chunks, bounded by the budget and by one chunk
  per 131072 elements, picking the first count that divides an axis (else
  padding one). The split is baked into the kernel and its cache identity.
- `gpudims` lowers that axis to a `core_id` variable in `[0, N-1]` instead of
  `gidx`/`lidx`, and `global_size[0]` becomes `N`.
- At run time `execute_kernel` sees `global_size[0] > 1` and runs
  `(0..N).into_par_iter()`, calling the same function pointer with the same
  buffers and `core_id` overwritten per task. Each `core_id` writes a disjoint
  output range, so no synchronization is needed. Inside an existing rayon worker
  the loop runs serially instead of nesting.

The kernel split and the pool size may differ: a kernel split 8 ways runs
correctly on 4 threads. `SVOD_THREADS=1` disables the split. Rayon builds its
global pool once, so the first caller's size wins and a later different request
logs one warning (`ensure_thread_pool`). `RAYON_NUM_THREADS` is not consulted.

---

## Calling a kernel

A loaded kernel is a `JitKernel` (C path, re-exported as `ClangKernel`) or
`LlvmKernel`: the mapped object, the entry pointer, the var names and a
`KernelCif`. The ABI is `void kernel(ptr..., i32...)` in ABI slot order:
storage parameters are `Type::pointer()`, scalars `Type::i32()`, and the `i64`
values the executor carries are truncated to `i32`. The call itself goes through
**libffi** (`cif.call(CodePtr(fn_ptr), &args)`, `runtime/src/dispatch.rs`),
which is why the kernel signature can vary per kernel without a transmute per
arity.

The CPU has no graph factory, no plan context and no timestamps: every kernel is
a synchronous call, `wait` is ignored, and the profiler's device-time tier is
wall-clock.

---

## Object cache

Compiled objects are cached on disk (`runtime/src/object_cache.rs`) under
`SVOD_OBJECT_CACHE_DIR`, else `$XDG_CACHE_HOME/svod/objects`, else
`~/.cache/svod/objects`; `SVOD_OBJECT_CACHE=0` disables it and
`SVOD_OBJECT_CACHE_MAX_BYTES` sets the budget (default 1 GiB, oldest-first
eviction by mtime). An entry is keyed by the SHA-256 of the source and a
`CompilerIdentity` (`schema`, `backend`, `target_architecture`, `toolchain`,
`flags`, `abi`, `object_format`), stored as `<hex>.obj` with a magic, the key,
a payload digest and the payload; a failed integrity check is a miss. Every
object, cached or fresh, is validated against the host ELF architecture,
endianness, object kind and entry symbol before it is loaded. The same cache
holds the AMD, CUDA and Metal objects under their own identities.

---

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `SVOD_CPU_BACKEND` | `llvm` | `clang` selects the C path |
| `SVOD_LLVM_INPROCESS` | on | `0` forces the `clang -x ir` subprocess |
| `SVOD_LLVM_LIB` | unset | Path of the libLLVM to bind; the only candidate when set |
| `SVOD_THREADS` | host parallelism | Thread budget and default `core_id` split |
| `SVOD_OBJECT_CACHE` | on | `0` disables the on-disk object cache |
| `SVOD_OBJECT_CACHE_DIR` | `$XDG_CACHE_HOME/svod/objects` | Relocates the cache |
| `SVOD_OBJECT_CACHE_MAX_BYTES` | 1 GiB | Cache budget |
| `SVOD_DUMP_LLVM_IR` | unset | Directory receiving each kernel's rendered LLVM IR |
| `SVOD_DUMP_POST_O2_IR` | unset | Directory receiving the IR after clang's `-O2` pipeline |
| `RUST_LOG` | unset | `svod_runtime=debug` says which producer compiles (libLLVM or clang) and logs each kernel compile and load; an unintended fallback to clang is a `warn` |
