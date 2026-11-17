//! Wraps `_IOWR`-style KFD ioctl numbers as typed `nix` ioctl macros.
//!
//! bindgen emits the struct definitions but not the `_IOR/_IOWR` macros, so
//! we declare them by hand. The `(group, opcode, args_ty)` triples come
//! straight from `kfd_ioctl.h:1560-1655`.

use nix::ioctl_readwrite;

use super::{drm, kfd};

// Type code for KFD ioctls (`#define KFD_IOCTL_BASE 'K'`).
const KFD_IOCTL_BASE: u8 = b'K';

// We declare every ioctl as `readwrite` even when the header uses `_IOR/_IOW`
// — KFD treats the argument struct as in/out, and the kernel side is
// tolerant of both directions.
ioctl_readwrite!(kfd_get_version, KFD_IOCTL_BASE, 0x01, kfd::kfd_ioctl_get_version_args);
ioctl_readwrite!(kfd_create_queue, KFD_IOCTL_BASE, 0x02, kfd::kfd_ioctl_create_queue_args);
ioctl_readwrite!(kfd_destroy_queue, KFD_IOCTL_BASE, 0x03, kfd::kfd_ioctl_destroy_queue_args);
ioctl_readwrite!(kfd_set_memory_policy, KFD_IOCTL_BASE, 0x04, kfd::kfd_ioctl_set_memory_policy_args);
ioctl_readwrite!(kfd_get_clock_counters, KFD_IOCTL_BASE, 0x05, kfd::kfd_ioctl_get_clock_counters_args);
ioctl_readwrite!(kfd_get_process_apertures, KFD_IOCTL_BASE, 0x06, kfd::kfd_ioctl_get_process_apertures_args);
ioctl_readwrite!(kfd_update_queue, KFD_IOCTL_BASE, 0x07, kfd::kfd_ioctl_update_queue_args);
ioctl_readwrite!(kfd_create_event, KFD_IOCTL_BASE, 0x08, kfd::kfd_ioctl_create_event_args);
ioctl_readwrite!(kfd_destroy_event, KFD_IOCTL_BASE, 0x09, kfd::kfd_ioctl_destroy_event_args);
ioctl_readwrite!(kfd_set_event, KFD_IOCTL_BASE, 0x0A, kfd::kfd_ioctl_set_event_args);
ioctl_readwrite!(kfd_reset_event, KFD_IOCTL_BASE, 0x0B, kfd::kfd_ioctl_reset_event_args);
ioctl_readwrite!(kfd_wait_events, KFD_IOCTL_BASE, 0x0C, kfd::kfd_ioctl_wait_events_args);
ioctl_readwrite!(kfd_acquire_vm, KFD_IOCTL_BASE, 0x15, kfd::kfd_ioctl_acquire_vm_args);
ioctl_readwrite!(kfd_alloc_memory_of_gpu, KFD_IOCTL_BASE, 0x16, kfd::kfd_ioctl_alloc_memory_of_gpu_args);
ioctl_readwrite!(kfd_free_memory_of_gpu, KFD_IOCTL_BASE, 0x17, kfd::kfd_ioctl_free_memory_of_gpu_args);
ioctl_readwrite!(kfd_map_memory_to_gpu, KFD_IOCTL_BASE, 0x18, kfd::kfd_ioctl_map_memory_to_gpu_args);
ioctl_readwrite!(kfd_unmap_memory_from_gpu, KFD_IOCTL_BASE, 0x19, kfd::kfd_ioctl_unmap_memory_from_gpu_args);
ioctl_readwrite!(kfd_runtime_enable, KFD_IOCTL_BASE, 0x25, kfd::kfd_ioctl_runtime_enable_args);

// `DRM_IOCTL_AMDGPU_CTX`: `DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_CTX, union drm_amdgpu_ctx)`.
ioctl_readwrite!(amdgpu_ctx, drm::DRM_IOCTL_BASE, drm::DRM_COMMAND_BASE + drm::DRM_AMDGPU_CTX, drm::drm_amdgpu_ctx);
