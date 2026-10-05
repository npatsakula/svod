---
sidebar_label: Debugging
---

# Debugging

The backend has three failure surfaces: the host toolchain (clang, the object
cache), the driver JIT (`cuModuleLoadDataEx`), and the device at run time
(driver errors, memory faults). This page lists the knobs and how to read each
error.

---

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `SVOD_DEVICE` | `CPU` (`METAL:0` on macOS) | `CUDA:N` (alias `GPU`) selects the default tensor device; `NV` is *not* an alias, it stays reserved for a userspace driver |
| `SVOD_DUMP_NVPTX_IR` | unset | Directory receiving each kernel's NVPTX LLVM IR as `sm_XY_<kernel>.ll` |
| `SVOD_CUDA_PTXAS` | on | `0` skips the `ptxas` pre-assembly and hands PTX text to the driver JIT |
| `SVOD_CUDA_SCOPED_SYNC` | on | `0` replaces every scoped wait with a full `cuCtxSynchronize` and makes copies synchronous ([Architecture](./architecture.md)) |
| `SVOD_CUDA_CUPTI` | on | `0` skips loading `libcupti.so.13`, so there are no hardware counters |
| `SVOD_PMC` | unset | `1` for the backend's default counter set, or a comma-separated token list, see [Profiling](./profiling.md) |
| `SVOD_OBJECT_CACHE` | on | `0` disables the on-disk cache of compiled objects |
| `SVOD_OBJECT_CACHE_DIR` | `$XDG_CACHE_HOME/svod/objects`, else `~/.cache/svod/objects` | Relocates the cache |
| `SVOD_OBJECT_CACHE_MAX_BYTES` | 1 GiB | Cache budget |
| `CUDA_PATH` | unset | Last place searched for `ptxas` (`$CUDA_PATH/bin`, after `PATH` and `/opt/cuda/bin`) and CUPTI (`$CUDA_PATH/lib64`, `$CUDA_PATH/extras/CUPTI/lib64`) |
| `SVOD_PROFILE_ITERS`, `SVOD_ORIGIN`, `SVOD_ORIGIN_DEPTH` | | Profiler knobs, see [Profiling](./profiling.md) |
| `RUST_LOG` | unset | `svod_device=info` carries the device open line; `svod_device=debug` adds the JIT info log, graph capture and replay fallbacks; `svod_runtime=debug` logs each NVPTX compile (arch and IR size) and an unusable `ptxas` |

There is no CUDA-specific dispatch dump; the driver JIT log and `tracing` cover
what `SVOD_DEBUG_DISPATCH` does on AMD.

---

## "No CUDA device" on a host that has one

`has_devices()` is `false` when the library does not load, a bound symbol is
missing, `cuInit` fails, or the count is zero. `CudaDevice::open` reports which:

```text
device unavailable: cannot load libcuda.so.1: ...        # no driver on the loader path
device unavailable: libcuda.so.1 has no symbol cu...     # driver too old for a bound entry point
no CUDA GPU available: CUDA cuInit failed: ...           # driver loaded, no usable device
```

Check `ldconfig -p | grep libcuda`, `nvidia-smi`, and that the process can
open `/dev/nvidia*`.

---

## Reading a JIT failure

A PTX the driver rejects surfaces as `Error::CudaJit`, whose display is the
cause followed by the driver's error log:

```text
CUDA JIT of kernel "r_64_32" failed: CUDA_ERROR_INVALID_PTX: a PTX JIT compilation failed
ptxas application ptx input, line 27; error   : ...
```

`CUDA_ERROR_UNSUPPORTED_PTX_VERSION` means the driver is older than the PTX
ISA of the module (the pin follows the compute capability: 7.8 up to sm_88, 8.4 on sm_89
and sm_90, 8.6 to 8.8 across Blackwell), see the
[requirements](./overview.md). The info log (warnings, register spills) is
logged at `debug` level under `svod_device`. When `ptxas` pre-assembled the
kernel there is no driver JIT and the same diagnostic arrives from `ptxas`
itself, as the message of a failed compile.

Two errors come from Svod's own validator before the driver sees anything:

```text
PTX references an unresolved function: .extern .func ...   # an LLVM intrinsic name the NVPTX
                                                            # backend did not recognize
cached PTX targets sm_80, not sm_86                          # a corrupt or foreign cache entry
```

The first one is the important trap: a misspelt `llvm.nvvm.*` intrinsic is
not a clang error, it becomes an external call. The fix is in
`codegen/src/llvm/nvptx/` or the intrinsic declaration table in
`codegen/src/llvm/text/mod.rs`.

A cubin entry is checked instead by `validate_cubin` (a little-endian ELF64
for `EM_CUDA` defining the entry as code), and its `.param` list is checked
against the ABI on the PTX text before assembly, since a cubin does not
carry one. On the PTX path the loader runs that `.param` check itself; a
mismatch reads `PTX entry declares N parameters, the ABI describes M`. An
`llvm.nvvm.*` builder that reaches a non-NVPTX target fails as
`Error::ForeignIntrinsic`.

---

## Offline checks with the toolkit

Nothing at run time *requires* the CUDA toolkit — `ptxas` is used to
pre-assemble kernels when it happens to be installed — and if it is there its
tools also work on the dumped IR:

```bash
SVOD_DUMP_NVPTX_IR=/tmp/nvptx SVOD_DEVICE=CUDA:0 cargo test -p svod-tensor -- some_test

# Reproduce the exact compile, then assemble with ptxas to see the real diagnostics
clang -x ir -S -O3 --target=nvptx64-nvidia-cuda -march=sm_86 --cuda-feature=+ptx78 \
      -Wno-override-module /tmp/nvptx/sm_86_r_64_32.ll -o r_64_32.ptx
ptxas -arch=sm_86 -v r_64_32.ptx -o r_64_32.cubin   # -v prints registers, shared, spills
nvdisasm r_64_32.cubin | less                         # the SASS
```

`ptxas -v` is the quickest way to see why `maxThreadsPerBlock` came out
smaller than the launch wants: register pressure against the `.maxntid`
launch bound. The corresponding run-time error names the numbers:

```text
CUDA kernel 'r_64_32' block [512, 1, 1] (512 threads) exceeds its maxThreadsPerBlock 256 (numRegs 96, sharedSizeBytes 4096, localSizeBytes 0)
```

---

## Driver errors at run time

Every driver call is checked and reported with the driver's own name and text:

```text
CUDA cuStreamSynchronize failed: CUDA_ERROR_ILLEGAL_ADDRESS (700): an illegal memory access was encountered
```

A kernel fault is asynchronous: `cuLaunchKernel` succeeds and the error lands
on the next synchronizing call (`cuStreamSynchronize`, `cuEventSynchronize`,
`cuCtxSynchronize`, a host copy). Codes the driver documents as **sticky**
(`ILLEGAL_ADDRESS`, `LAUNCH_FAILED`, `ILLEGAL_INSTRUCTION`,
`MISALIGNED_ADDRESS`, `ECC_UNCORRECTABLE`, ...) poison the device: every later
call fails immediately with the recorded message, and frees quarantine their
allocations instead of releasing memory a hung kernel may still touch.

Unlike the AMD backend there is no VA registry to classify the faulting
address; the driver does not expose it. To localize a fault, run the same
binary under the toolkit's sanitizer, which works on driver-API programs and
JIT-loaded PTX:

```bash
SVOD_DEVICE=CUDA:0 compute-sanitizer --tool memcheck \
  target/release/examples/gigaam_infer ./audio.wav
```

For a wrong result rather than a fault, the graph replay fallback is worth a
look: `RUST_LOG=svod_device=debug` prints `CUDA graph replay with re-aliased
buffers; using the capture-order chain` when a replay's buffer aliasing
differs from capture, and `SVOD_DEVICE=CUDA:0 cargo test -p svod-tensor` runs
the `codegen_tests!` `cuda` variants, the same tensor-level assertions the CPU
backends pass, one kernel at a time.

:::tip[The pipeline debugger]
For compiler-side issues (which UOps produced which IR) the `/svod-debug` skill
documents the frontend → codegen tracing targets; `SVOD_DUMP_NVPTX_IR` is the
CUDA member of that family next to `SVOD_DUMP_AMD_IR` and `SVOD_DUMP_LLVM_IR`.
:::
