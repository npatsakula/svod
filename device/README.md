# svod-device

The device layer of the Svod ML compiler: buffers with lazy allocation and
zero-copy views, per-device allocators, the `Device` / `Program` / `Graph`
traits the runtime implements, and the hardware bindings for every backend.
All backends are compiled on every host and opened at run time: AMD through KFD
ioctls (Linux), NVIDIA through a runtime-loaded `libcuda.so.1`, and Apple GPUs
through runtime-loaded Metal frameworks. Opening a device whose hardware is
absent returns an error instead of failing the build.

## Example

```rust
use svod_device::{Buffer, BufferSpec, registry};
use svod_dtype::DType;

// Allocated on first use.
let cpu = registry::cpu()?;
let mut dst = Buffer::new(cpu.clone(), DType::Float32, vec![1024], BufferSpec::default());
let src = Buffer::allocate(cpu, DType::Float32, vec![1024], BufferSpec::default())?;

// A view shares storage with its parent; offset and size are in bytes.
let half = src.view(0, 512 * 4)?;
assert_eq!(half.size(), 2048);

dst.copy_from(&src)?;

// Device strings are case-insensitive `NAME[:N]` (or `DISK:<path>`) and cached per spec;
// `cpu_access: false` asks for device-only memory, e.g. VRAM without a host mapping.
let amd = registry::get_device("amd:0")?;
let vram = Buffer::allocate(amd, DType::Float32, vec![1024], BufferSpec { cpu_access: false, ..Default::default() })?;
```

## Allocators

| Device | Allocator | Backing |
|--------|-----------|---------|
| `CPU` | `CpuAllocator` | 64-byte aligned host memory |
| `AMD:N` | `AmdAllocator` | KFD: VRAM or GTT, optional host BAR mapping |
| `CUDA:N` | `CudaAllocator` | device memory; managed or pinned host memory when host-visible |
| `METAL:N` | `MetalAllocator` | `MTLBuffer` in shared storage mode |
| `DISK:path` | `DiskAllocator` | read-only mmap, cannot run kernels |

Every allocator except `DISK` is wrapped in `LruAllocator`, which pools freed
buffers by `(size, BufferSpec)` and re-zeroes reused ones on demand.

Documentation: <https://svod.vpermilp.online/docs/backends/overview>
