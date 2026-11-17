/* Vendored empty <drm/drm.h> shim.
 *
 * kfd_ioctl.h includes <drm/drm.h> but uses no drm_* type from it — the only
 * DRM-adjacent fields are plain `__u32 drm_fd`. The include is vestigial; this
 * shim resolves it so the header parses without the system libdrm UAPI headers,
 * keeping the bindgen build hermetic on every platform.
 *
 * amdgpu_drm.h numbers its ioctls from the two bases below, copied verbatim
 * from Linux 7.3 include/uapi/drm/drm.h. */
#ifndef _SVOD_VENDORED_DRM_DRM_H
#define _SVOD_VENDORED_DRM_DRM_H

#include <linux/types.h>

#define DRM_IOCTL_BASE			'd'
#define DRM_COMMAND_BASE                0x40

#endif /* _SVOD_VENDORED_DRM_DRM_H */
