# svod-device

Device abstraction with lazy buffer allocation, zero-copy views, and LRU caching.

## Example

```rust
use svod_device::{Buffer, BufferSpec, registry};
use svod_dtype::DType;

// CPU buffer (lazy allocation)
let cpu = registry::cpu()?;
let buf = Buffer::new(cpu, DType::Float32, vec![1024], BufferSpec::default());

// AMD buffer without a host mapping (device-only VRAM)
let amd = registry::get_device("AMD:0")?;
let opts = BufferSpec { cpu_access: false, ..Default::default() };
let vram = Buffer::allocate(amd, DType::Float32, vec![1024], opts)?;

// Zero-copy view
let view = buf.view(0, 512)?;

// Device-to-device copy
dst.copy_from(&src)?;
```

## Allocators

| Device | Allocator | Backing |
|--------|-----------|---------|
| `CPU` | `CpuAllocator` | 64-byte aligned host memory |
| `AMD:N` | `AmdAllocator` | KFD ioctls: VRAM or GTT, optional host BAR mmap |
| `METAL:N` | `MetalAllocator` | `MTLBuffer` with shared storage |
| `DISK:path` | `DiskAllocator` | read-only mmap, no LRU cache |
| `CUDA:N` | `CudaAllocator` | CUDA driver API: device memory, managed for host-visible, pinned for `host` |

Every compute allocator is wrapped in `LruAllocator`, which pools freed
buffers by `(size, BufferSpec)` and re-zeroes on demand. GPU backends are
always compiled and self-register only when their hardware is present.

## Device Registry

```rust
registry::cpu()                 // CPU allocator
registry::get_device("AMD:1")   // Parse string, cached per spec

DeviceSpec::parse("amd:0")      // Case-insensitive parsing
spec.canonicalize()             // → "AMD:0"
```

## Testing

```bash
cargo test -p svod-device
```

GPU tests self-skip when no supported device is present.
