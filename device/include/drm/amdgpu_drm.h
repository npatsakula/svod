/* Vendored minimal amdgpu_drm.h shim: the DRM_IOCTL_AMDGPU_CTX subset, which
 * pins a stable power state for the SQTT profiler. The definitions below are
 * copied verbatim from Linux 7.3 include/uapi/drm/amdgpu_drm.h (Arch
 * linux-api-headers 7.3-1); its "drm.h" include resolves to the shim next to
 * it. Widen it from that header when another amdgpu ioctl is needed. */
/* amdgpu_drm.h -- Public header for the amdgpu driver -*- linux-c -*-
 *
 * Copyright 2000 Precision Insight, Inc., Cedar Park, Texas.
 * Copyright 2000 VA Linux Systems, Inc., Fremont, California.
 * Copyright 2002 Tungsten Graphics, Inc., Cedar Park, Texas.
 * Copyright 2014 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE COPYRIGHT HOLDER(S) OR AUTHOR(S) BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 * Authors:
 *    Kevin E. Martin <martin@valinux.com>
 *    Gareth Hughes <gareth@valinux.com>
 *    Keith Whitwell <keith@tungstengraphics.com>
 */

#ifndef __AMDGPU_DRM_H__
#define __AMDGPU_DRM_H__

#include "drm.h"


#define DRM_AMDGPU_CTX			0x02

/* context related */
#define AMDGPU_CTX_OP_ALLOC_CTX	1
#define AMDGPU_CTX_OP_FREE_CTX	2
#define AMDGPU_CTX_OP_QUERY_STATE	3
#define AMDGPU_CTX_OP_QUERY_STATE2	4
#define AMDGPU_CTX_OP_GET_STABLE_PSTATE	5
#define AMDGPU_CTX_OP_SET_STABLE_PSTATE	6

/* GPU reset status */
#define AMDGPU_CTX_NO_RESET		0
/* this the context caused it */
#define AMDGPU_CTX_GUILTY_RESET		1
/* some other context caused it */
#define AMDGPU_CTX_INNOCENT_RESET	2
/* unknown cause */
#define AMDGPU_CTX_UNKNOWN_RESET	3

/* indicate gpu reset occurred after ctx created */
#define AMDGPU_CTX_QUERY2_FLAGS_RESET    (1<<0)
/* indicate vram lost occurred after ctx created */
#define AMDGPU_CTX_QUERY2_FLAGS_VRAMLOST (1<<1)
/* indicate some job from this context once cause gpu hang */
#define AMDGPU_CTX_QUERY2_FLAGS_GUILTY   (1<<2)
/* indicate some errors are detected by RAS */
#define AMDGPU_CTX_QUERY2_FLAGS_RAS_CE   (1<<3)
#define AMDGPU_CTX_QUERY2_FLAGS_RAS_UE   (1<<4)
/* indicate that the reset hasn't completed yet */
#define AMDGPU_CTX_QUERY2_FLAGS_RESET_IN_PROGRESS (1<<5)

/* Context priority level */
#define AMDGPU_CTX_PRIORITY_UNSET       -2048
#define AMDGPU_CTX_PRIORITY_VERY_LOW    -1023
#define AMDGPU_CTX_PRIORITY_LOW         -512
#define AMDGPU_CTX_PRIORITY_NORMAL      0
/*
 * When used in struct drm_amdgpu_ctx_in, a priority above NORMAL requires
 * CAP_SYS_NICE or DRM_MASTER
*/
#define AMDGPU_CTX_PRIORITY_HIGH        512
#define AMDGPU_CTX_PRIORITY_VERY_HIGH   1023

/* select a stable profiling pstate for perfmon tools */
#define AMDGPU_CTX_STABLE_PSTATE_FLAGS_MASK  0xf
#define AMDGPU_CTX_STABLE_PSTATE_NONE  0
#define AMDGPU_CTX_STABLE_PSTATE_STANDARD  1
#define AMDGPU_CTX_STABLE_PSTATE_MIN_SCLK  2
#define AMDGPU_CTX_STABLE_PSTATE_MIN_MCLK  3
#define AMDGPU_CTX_STABLE_PSTATE_PEAK  4

struct drm_amdgpu_ctx_in {
	/** AMDGPU_CTX_OP_* */
	__u32	op;
	/** Flags */
	__u32	flags;
	__u32	ctx_id;
	/** AMDGPU_CTX_PRIORITY_* */
	__s32	priority;
};

union drm_amdgpu_ctx_out {
		struct {
			__u32	ctx_id;
			__u32	_pad;
		} alloc;

		struct {
			/** For future use, no flags defined so far */
			__u64	flags;
			/** Number of resets caused by this context so far. */
			__u32	hangs;
			/** Reset status since the last call of the ioctl. */
			__u32	reset_status;
		} state;

		struct {
			__u32	flags;
			__u32	_pad;
		} pstate;
};

union drm_amdgpu_ctx {
	struct drm_amdgpu_ctx_in in;
	union drm_amdgpu_ctx_out out;
};
#endif
