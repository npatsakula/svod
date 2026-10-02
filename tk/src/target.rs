//! Target-capability gate for hand-built tile kernels.
//!
//! A tile kernel is built for a specific GPU arch (its matrix-core descriptor, wave
//! width, and lane distribution are arch-specific) and compiles via `clang -x ir`.
//! This gate validates the kernel inputs' [`DeviceSpec`] against the **arch set the
//! kernel declares it supports** ([`ArchSet`]) and that the matching LLVM GPU
//! backend is present — failing fast with a clear message instead of mis-rendering
//! or failing deep in compile.
//!
//! The gate is generic over the supported set: a kernel passes its own [`ArchSet`]
//! (k-means and k-NN declare [`CDNA_RDNA_WMMA`]; flash-attention, matmul and
//! single-query attention add `sm_80+`; norm and the NT GEMM are wave32-only and
//! declare [`RDNA_WMMA`]).
//! Adding a GPU is "declare it here (and supply its arch-specific kernel bits)",
//! not "rewrite this"; the generic launch infra (`compile`/`run_kernel`/
//! `graph_launch`) stays arch-agnostic — only the per-kernel launcher invokes this.
//!
//! It validates **from the `DeviceSpec`** (no full-`Device` open): the specs
//! deliberately omit the arch (it's a hardware property — baking it into the spec
//! invites the "two specs, one physical device" trap; see `svod_dtype::DeviceSpec`),
//! so the arch is resolved from the spec's `device_id` via the backend registry.

use std::fmt;

use svod_dtype::{AmdArch, CudaArch, DeviceSpec, GpuArch, MetalFamily};

use crate::launch::{Result, ToolchainUnavailableSnafu, UnsupportedArchSnafu};

/// The GPU targets a kernel is built for: an explicit AMD arch list (each needs
/// its own fragment tables) plus an open-ended CUDA capability floor (`sm_XY` and
/// newer — the warp is 32 lanes on every generation, so a shuffle-only kernel
/// ports by threshold, not by enumeration). `None` = not ported to CUDA.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArchSet {
    pub amd: &'static [AmdArch],
    pub cuda_min: Option<CudaArch>,
    pub metal_min: Option<MetalFamily>,
}

/// The wave32 WMMA parts tk carries validated fragment tables for: RDNA3.5
/// (gfx11 shapes — replicated inputs, even/odd accumulator) and RDNA4 (the
/// strided 8/lane gfx12 fragment). One family, one config table
/// ([`crate::arch::Family::Rdna`]), so a kernel names the family's parts rather
/// than a single measured card.
pub const RDNA_WMMA: &[AmdArch] = &[AmdArch::Gfx1151, AmdArch::Gfx1200, AmdArch::Gfx1201];

/// [`RDNA_WMMA`] plus the validated CDNA part (gfx942, MFMA wave64) — the AMD
/// list of a kernel whose body is arch-generic across both families.
pub const CDNA_RDNA_WMMA: &[AmdArch] = &[AmdArch::Gfx942, AmdArch::Gfx1151, AmdArch::Gfx1200, AmdArch::Gfx1201];

impl ArchSet {
    /// AMD-only support.
    pub const fn amd(amd: &'static [AmdArch]) -> Self {
        Self { amd, cuda_min: None, metal_min: None }
    }

    /// Also support CUDA at compute capability `min` and above.
    pub const fn with_cuda_from(self, min: CudaArch) -> Self {
        Self { cuda_min: Some(min), ..self }
    }

    /// Also support Apple GPUs from family `min` upward.
    pub const fn with_metal_from(self, min: MetalFamily) -> Self {
        Self { metal_min: Some(min), ..self }
    }

    /// Whether `arch` is in the set.
    pub fn supports(&self, arch: GpuArch) -> bool {
        match arch {
            GpuArch::Amd(amd) => self.amd.contains(&amd),
            GpuArch::Cuda(cuda) => self.cuda_min.is_some_and(|min| cuda >= min),
            GpuArch::Metal(family) => self.metal_min.is_some_and(|min| family >= min),
        }
    }
}

impl fmt::Display for ArchSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AMD {:?}", self.amd)?;
        if let Some(min) = self.cuda_min {
            write!(f, " + CUDA {min}+")?;
        }
        match self.metal_min {
            Some(min) => write!(f, " + Metal {min}+"),
            None => Ok(()),
        }
    }
}

/// Resolve the concrete [`GpuArch`] backing a [`DeviceSpec`] — AMD from the KFD
/// topology, CUDA from the driver's compute capability, Metal from the opened
/// device's GPU family (a host or unreadable device → `None`). The arch is deliberately not in the spec (a hardware
/// property), so it is looked up by `device_id`. [`resolve_supported_arch`] gates
/// on it and returns it; [`check_target`] is the `()`-returning wrapper for callers
/// that only need the gate.
pub fn resolve_arch(spec: &DeviceSpec) -> Option<GpuArch> {
    match spec {
        DeviceSpec::Amd { device_id } => {
            svod_device::registry::resolve_amd_arch_from_topology(*device_id).ok().map(GpuArch::Amd)
        }
        DeviceSpec::Cuda { device_id } => svod_device::registry::resolve_cuda_arch(*device_id).ok().map(GpuArch::Cuda),
        DeviceSpec::Metal { device_id } => {
            svod_device::registry::resolve_metal_family(*device_id).ok().map(GpuArch::Metal)
        }
        DeviceSpec::Cpu | DeviceSpec::WebGpu | DeviceSpec::Disk { .. } => None,
    }
}

/// The compute-unit count (AMD CUs, CUDA SMs) of the device behind `spec`,
/// when the backend reports it: what a kernel's tile crossover measures its
/// launch grid against.
pub fn compute_units(spec: &DeviceSpec) -> Option<usize> {
    match spec {
        DeviceSpec::Amd { device_id } => {
            let node = svod_device::amd::topology::enumerate().into_iter().nth(*device_id)?;
            (node.simd_per_cu > 0).then(|| (node.simd_count / node.simd_per_cu) as usize)
        }
        DeviceSpec::Cuda { device_id } => {
            svod_device::registry::resolve_cuda_limits(*device_id).ok().map(|limits| limits.sm_count as usize)
        }
        DeviceSpec::Metal { .. } | DeviceSpec::Cpu | DeviceSpec::WebGpu | DeviceSpec::Disk { .. } => None,
    }
}

/// How many one-wave workgroups a compute unit of the device behind `spec`
/// keeps resident, when the backend reports it: the wave slots of a CU's SIMDs
/// on AMD, the resident-block cap of an SM on CUDA. What a latency-bound
/// kernel's grid has to reach for the device to be busy.
pub fn resident_waves_per_cu(spec: &DeviceSpec) -> Option<usize> {
    match spec {
        DeviceSpec::Amd { device_id } => {
            let node = svod_device::amd::topology::enumerate().into_iter().nth(*device_id)?;
            let waves = (node.simd_per_cu * node.max_waves_per_simd) as usize;
            (waves > 0).then_some(waves)
        }
        DeviceSpec::Cuda { device_id } => {
            let limits = svod_device::registry::resolve_cuda_limits(*device_id).ok()?;
            let warps = limits.max_threads_per_sm.checked_div(limits.warp_size)?;
            let waves = limits.max_blocks_per_sm.min(warps) as usize;
            (waves > 0).then_some(waves)
        }
        DeviceSpec::Metal { .. } | DeviceSpec::Cpu | DeviceSpec::WebGpu | DeviceSpec::Disk { .. } => None,
    }
}

/// What the device behind `spec` allows one workgroup and one compute unit:
/// `(max threads per workgroup, shared bytes per workgroup, shared bytes per
/// compute unit, registers per compute unit)`. `None` for a field the backend
/// does not report.
///
/// These are the limits a tile is generated against: threads and shared memory
/// bound the tile that *fits*, and the per-compute-unit pair bounds how many of
/// those tiles stay resident, which is what a latency-bound kernel lives on.
/// AMD publishes its LDS but not its register file — KFD has no field for it —
/// so the register term comes back `None` there and a caller that needs one
/// falls back to its own floor.
pub fn workgroup_limits(spec: &DeviceSpec) -> Option<WorkgroupLimits> {
    match spec {
        DeviceSpec::Amd { device_id } => {
            let node = svod_device::amd::topology::enumerate().into_iter().nth(*device_id)?;
            let lds = (node.lds_size_in_kb as usize) * 1024;
            (lds > 0).then_some(WorkgroupLimits {
                max_threads: 1024,
                shared_per_workgroup: lds,
                shared_per_cu: lds,
                registers_per_cu: None,
            })
        }
        DeviceSpec::Cuda { device_id } => {
            let limits = svod_device::registry::resolve_cuda_limits(*device_id).ok()?;
            Some(WorkgroupLimits {
                max_threads: limits.max_threads_per_block as usize,
                shared_per_workgroup: limits.shared_per_block as usize,
                shared_per_cu: limits.shared_per_sm as usize,
                registers_per_cu: Some(limits.registers_per_sm as usize).filter(|&r| r > 0),
            })
        }
        DeviceSpec::Metal { .. } | DeviceSpec::Cpu | DeviceSpec::WebGpu | DeviceSpec::Disk { .. } => None,
    }
}

/// See [`workgroup_limits`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkgroupLimits {
    pub max_threads: usize,
    pub shared_per_workgroup: usize,
    pub shared_per_cu: usize,
    pub registers_per_cu: Option<usize>,
}

/// Gate the kernel inputs' device `spec` to the kernel's `supported` arches
/// **and** verify the matching LLVM GPU backend (`clang` amdgcn / nvptx64) —
/// returning the resolved arch so the launcher can build
/// [`crate::ArchCaps::for_arch`] from it **without a second probe**. A host spec,
/// an unsupported/unreadable device, or a missing toolchain fails. This is the
/// single arch resolution per launch; call it from a kernel launcher with
/// `Tensor::device()`.
pub fn resolve_supported_arch(spec: &DeviceSpec, supported: ArchSet) -> Result<GpuArch> {
    let resolved = resolve_arch(spec);
    let Some(arch) = resolved.filter(|a| supported.supports(*a)) else {
        return UnsupportedArchSnafu { supported, spec: spec.clone(), resolved }.fail();
    };
    let (target, present) = match arch {
        GpuArch::Amd(_) => ("amdgcn", svod_runtime::amd::has_amdgpu_target()),
        GpuArch::Cuda(_) => ("nvptx64", svod_runtime::cuda::has_nvptx_target()),
        // Metal renders MSL source through the system compiler rather than an LLVM
        // GPU backend, so the toolchain probe is the device probe.
        GpuArch::Metal(_) => ("metal", svod_device::metal::has_devices()),
    };
    if !present {
        return ToolchainUnavailableSnafu { target }.fail();
    }
    Ok(arch)
}

/// [`resolve_supported_arch`] discarding the arch — the gate-only wrapper for
/// launchers that don't need the resolved arch (the SDPA-fallback eligibility
/// check folds this into [`resolve_supported_arch`] directly instead).
pub fn check_target(spec: &DeviceSpec, supported: ArchSet) -> Result<()> {
    resolve_supported_arch(spec, supported).map(|_| ())
}
