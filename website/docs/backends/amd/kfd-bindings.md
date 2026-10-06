---
sidebar_label: KFD Bindings
---

# KFD Bindings

The backend speaks to the kernel through a small, fixed set of `ioctl` calls on
`/dev/kfd`. This page covers how those calls are bound to Rust, which ones the
backend actually uses, how GPU nodes are discovered, and the allocation flow
that turns an `ioctl` into a mapped GPU buffer. For *why* the backend is
KFD-direct rather than HIP-based, see the [Overview](./overview.md).

---

## How the bindings are generated

KFD's ABI is a C header, `kfd_ioctl.h`, vendored verbatim from the kernel into
`device/include/kfd_ioctl.h` (the upstream AMD file, complete with its ABI
version history). Rust bindings are generated from it at build time by
`bindgen`:

- `device/build.rs` runs `bindgen` on every Unix host (`bindgen` is a
  `cfg(unix)` build-dependency) with no empty-stub branch. It is **hermetic**:
  it needs no system kernel headers. The two headers `kfd_ioctl.h`
  transitively pulls (`<linux/ioctl.h>` for the `_IOC`/`_IO*` macros,
  `<linux/types.h>` for the `__uNN`/`__sNN` aliases) plus a stub `<drm/drm.h>`
  (vestigial — the body uses only `__u32 drm_fd` fields) are themselves vendored
  under `device/include/`, and `build.rs` passes `-I<crate>/include` so bindgen
  resolves them instead of `/usr/include`. The generated file has 60 structs
  and 34 constants. (bindgen needs `libclang`, which ships with the Xcode CLT
  on macOS.)

  It allow-lists exactly the KFD types and constants the backend needs:

  ```text
  allowlist_type:  kfd_ioctl_.*_args, kfd_process_device_apertures,
                   kfd_event_data, kfd_hsa_signal_event_data,
                   kfd_hsa_memory_exception_data, kfd_hsa_hw_exception_data,
                   kfd_memory_exception_failure, __u\d+, __s\d+
  allowlist_var:   KFD_IOC_.*, KFD_MAX_QUEUE_PERCENTAGE, AMDKFD_IOC_.*
  ```

  (The `AMDKFD_IOC_*` request codes are allow-listed but never materialize:
  bindgen can't const-fold their `_IOWR(...)` macro expansions, which is exactly
  why the ioctl numbers are computed Rust-side — see the note below.)

  with `.derive_default(true).layout_tests(false).generate_comments(false)`. The
  output is written to `$OUT_DIR/kfd_sys.rs`.

- `device/src/amd/sys/kfd.rs` is a one-liner that `include!`s the generated
  file.

- A **second bindgen pass** covers the AQL/HSA side: `include/amd_hsa_wrapper.h`
  pulls the vendored ROCm `hsa/` headers and yields `$OUT_DIR/hsa_sys.rs`
  (`hsa_kernel_dispatch_packet_t`, `hsa_queue_t`, `amd_queue_t`, `amd_signal_t`
  and friends, enums as constants), `include!`d by `device/src/amd/sys/hsa.rs`.
  Here `layout_tests` is deliberately left **on**: the 256-byte `amd_queue_t`
  and the 64-byte AQL packet are layout-critical, so a mis-sized struct must
  fail the build.

Compiling the bindings on every Unix host is what makes the AMD backend a
[runtime-detected execution provider](./overview.md) rather than a compile-time
feature: every Unix `cargo check` type-checks the KFD call sites above them
(the kernel-facing modules are `cfg(unix)`; the topology parser and packet
builders compile everywhere), and a host with no GPU simply never registers
the factory.

:::note[Why hand-written ioctl macros]
`bindgen` emits the argument *structs* but not the `_IOWR` ioctl-number macros.
Those are declared by hand in `device/src/amd/sys/ioctl.rs` using
`nix::ioctl_readwrite!`, with the type code `KFD_IOCTL_BASE = b'K'`. Every
ioctl is declared `readwrite` even where the header says `_IOR`/`_IOW` — KFD
treats the argument struct as in/out, and the kernel tolerates both directions.
:::

---

## The ioctls the backend uses

The `(group, opcode, args)` triples come straight from `kfd_ioctl.h`. These are
the ones with live call sites:

| Wrapper | Op | Used for |
|---|---|---|
| `kfd_get_version` | `0x01` | Read the KFD ABI version (gates `RUNTIME_ENABLE`) |
| `kfd_create_queue` | `0x02` | `setup_ring` — create a compute/SDMA queue |
| `kfd_destroy_queue` | `0x03` | `teardown_ring` |
| `kfd_update_queue` | `0x07` | Unmap/remap an AQL queue so CP firmware re-reads its `amd_queue_t` scratch descriptor |
| `kfd_create_event` | `0x08` | The queue-signal, memory-fault, and hw-exception events; binding the event page |
| `kfd_destroy_event` | `0x09` | Tear the three events down on `Drop` |
| `kfd_wait_events` | `0x0C` | `wait_events` — block on completion / fault events |
| `kfd_acquire_vm` | `0x15` | Register the DRM render fd as this process's VM for the GPU |
| `kfd_alloc_memory_of_gpu` | `0x16` | `alloc_raw` — allocate VRAM/GTT |
| `kfd_free_memory_of_gpu` | `0x17` | `free_raw` |
| `kfd_map_memory_to_gpu` | `0x18` | Bind an allocation into the GPU page table |
| `kfd_unmap_memory_from_gpu` | `0x19` | `free_raw` |
| `kfd_runtime_enable` | `0x25` | Enable the runtime (KFD ABI ≥ 1.14 only) |

Five more (`set_memory_policy`, `get_clock_counters`, `get_process_apertures`,
`set_event`, `reset_event`) are declared for completeness but not currently
called.

### Device bring-up sequence

`KfdIface::open` (`device/src/amd/iface.rs`) issues these in order, mirroring
tinygrad's `ops_amd.py`:

```mermaid
flowchart TD
  A["open /dev/kfd (process-shared, one fd)"] --> B["open /dev/dri/renderD(minor) (per node — the DRM render fd)"]
  B --> C["GET_VERSION: capture ABI version"]
  C --> D["ACQUIRE_VM(drm_fd): register this fd as the process VM for the GPU"]
  D --> E["RUNTIME_ENABLE: only if ABI is at least 1.14"]
  E --> F["event page: alloc + bind once per process, map per device"]
  F --> G["CREATE_EVENT x 3: queue-signal, memory-fault, hw-exception"]
```

The chain is strictly ordered: `ACQUIRE_VM` must precede any allocation, and the
event page must be bound before the first `CREATE_QUEUE`.

The DRM render fd is interesting: there are **no DRM ioctls**. The `drm_fd` is
used only two ways — passed *by number* into `ACQUIRE_VM`, and as the `mmap` fd
for host-visible mappings. The doorbell, by contrast, is `mmap`ped from the KFD
fd.

---

## Topology: finding the GPU

GPU nodes are enumerated from sysfs, not via an ioctl.
`device/src/amd/topology.rs` reads
`/sys/devices/virtual/kfd/kfd/topology/nodes/<N>/properties` — one
`key value` pair per line — plus the sibling `<N>/gpu_id` (falling back to an
inline `gpu_id` property), and returns a `Vec<AmdNode>` sorted by node id,
skipping CPU nodes (`gpu_id == 0`). It never panics: a host without that sysfs
tree yields an empty vector.

This same enumeration is what gates the whole backend at runtime.
`topology::has_devices()` — "any node whose `gfx_target_version` resolves to a
supported `AmdArch`" — is the side-effect-free probe the runtime calls to decide
whether to register the `"AMD"` device factory at all (the
[provider model](./overview.md)). No supported node ⇒ no `"AMD"` device type; and
if a factory is asked for a node that isn't there, it returns a clean
`Err(NoAmdGpu)`.

Each `AmdNode` carries the fields the rest of the backend needs: `node_id`,
`gpu_id`, `drm_render_minor`, `gfx_target_version` (e.g. `110000` → gfx1100,
see the [supported-GPU table](./overview.md)), `simd_count`, `array_count`,
`simd_arrays_per_engine`, `simd_per_cu`, `max_waves_per_simd`,
`lds_size_in_kb`, `wave_front_size`, `num_xcc`, `num_cp_queues` and
`max_slots_scratch_cu` — these feed scratch sizing and the PM4-vs-AQL
decision. The wave size the backend uses comes from the arch, not from
`wave_front_size`.

:::tip[Testing without hardware]
The sysfs root is overridable with **`SVOD_KFD_TOPOLOGY`**, so the parser is
unit-tested against a fabricated nodes directory with no GPU present.
:::

---

## The allocation flow

Every buffer follows the same four-step path, implemented once in
`KfdIface::alloc_raw`:

```text
1. reserve_va(size)                     mmap(PROT_NONE, …) — reserve host VA
2. ALLOC_MEMORY_OF_GPU(va, size, flags) → returns handle + mmap_offset
3. if host-visible:                     mmap(va, …, MAP_FIXED, drm_fd, offset)
4. MAP_MEMORY_TO_GPU(handle)            bind into the GPU page table
```

The size is rounded up to 4 KiB first. The host VA is reserved with an
anonymous `PROT_NONE` mapping so the host-visible `mmap` in step 3 can land at
exactly that address (`MAP_FIXED`). A requested zero-fill happens after the
map and requires host visibility. Freeing reverses it:
`UNMAP_MEMORY_FROM_GPU` → `munmap` → `FREE_MEMORY_OF_GPU`.

### Allocation flavors

`alloc_raw` takes an `AllocKind` that selects the KFD flag set — the single
place those flags are composed:

| `AllocKind` | Flags | Used for |
|---|---|---|
| `DeviceVram { executable }` | `VRAM \| WRITABLE \| NO_SUBSTITUTE` (+ `EXECUTABLE` for code, + `PUBLIC` when host-visible) | Tensor data, code objects, scratch; host-visible kernarg arenas and graph / linked-plan control buffers |
| `UncachedGtt` | `GTT \| WRITABLE \| EXECUTABLE \| NO_SUBSTITUTE \| PUBLIC \| COHERENT \| UNCACHED` | Command rings, GART pages, signal slots, the SDMA staging buffer |

The `UNCACHED | COHERENT` GTT flavor matters: the command ring and the signal
slots must be immediately visible between CPU and GPU, or the host spins forever
waiting on a completion value stuck in GPU L2. KFD rejects `CREATE_QUEUE` on a
plain-VRAM ring with `EINVAL`. The one allocation outside `alloc_raw` is the
per-process event page (`alloc_event_page` in `device.rs`), which composes the
same uncached-GTT flags with raw ioctls and is therefore not in the
[VA registry](./debugging.md).

### `cpu_access` follows the copy queue

The allocator (`device/src/amd/allocator.rs`) computes
`cpu_access = options.cpu_access || !self.dev.has_sdma_queue()`. When an SDMA copy
queue is installed (the default on every part — see [Overview](./overview.md)),
an intermediate can be **device-only** VRAM and copies go through DMA:
`_copyin`/`_copyout` stage through the copy queue, and `_transfer` is a
device→device DMA when either side is device-only. When no copy queue is
present, `has_sdma_queue()` is `false`, so every buffer is forced host-visible
and copies fall back to a plain host `memmove` after the scoped `wait_storage`.
The generic `LruAllocator` (`device/src/allocator.rs`) pools freed buffers by
`(size, BufferSpec)`; code objects, the EOP / CWSR context-save buffers, rings,
GART pages, signal slots and scratch never enter that pool — they call the
`AmdAllocator` directly (`alloc_uncached_tagged` / `alloc_host_visible_tagged`
/ `alloc_scratch`) and go straight to the seam.

:::note[Process-shared state]
`/dev/kfd` is opened once per process and shared by all devices (events are
addressed by id against that fd). The 0x8000-byte KFD **event page** is likewise
allocated and bound once per process; subsequent devices only `MAP_MEMORY_TO_GPU`
it into their own `gpu_id`. Both mirror tinygrad's per-process model.
:::

---

## Why this matters

The entire kernel-facing surface is **a handful of vendored headers, thirteen
ioctls, and a sysfs parser**. That is the whole reason the backend can avoid the
ROCm userspace stack: the kernel ABI is small and stable, so binding it directly
is less code than integrating HIP would be — and it leaves the
[backend seam](./overview.md) free to swap KFD out for the userspace
[AM driver](./am-driver.md) without touching anything above it.
