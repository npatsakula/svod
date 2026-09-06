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
//! (flash-attention declares the AMD pair; single-query attention adds `sm_80+`).
//! Adding a GPU is "declare it here (and supply its arch-specific kernel bits)",
//! not "rewrite this"; the generic launch infra (`compile`/`run_kernel`/
//! `graph_launch`) stays arch-agnostic — only the per-kernel launcher invokes this.
//!
//! It validates **from the `DeviceSpec`** (no full-`Device` open): the specs
//! deliberately omit the arch (it's a hardware property — baking it into the spec
//! invites the "two specs, one physical device" trap; see `svod_dtype::DeviceSpec`),
//! so the arch is resolved from the spec's `device_id` via the backend registry.

use std::fmt;

use svod_dtype::{AmdArch, CudaArch, DeviceSpec, GpuArch};

use crate::launch::{Result, ToolchainUnavailableSnafu, UnsupportedArchSnafu};

/// The GPU targets a kernel is built for: an explicit AMD arch list (each needs
/// its own fragment tables) plus an open-ended CUDA capability floor (`sm_XY` and
/// newer — the warp is 32 lanes on every generation, so a shuffle-only kernel
/// ports by threshold, not by enumeration). `None` = not ported to CUDA.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArchSet {
    pub amd: &'static [AmdArch],
    pub cuda_min: Option<CudaArch>,
}

impl ArchSet {
    /// AMD-only support.
    pub const fn amd(amd: &'static [AmdArch]) -> Self {
        Self { amd, cuda_min: None }
    }

    /// Also support CUDA at compute capability `min` and above.
    pub const fn with_cuda_from(self, min: CudaArch) -> Self {
        Self { cuda_min: Some(min), ..self }
    }

    /// Whether `arch` is in the set.
    pub fn supports(&self, arch: GpuArch) -> bool {
        match arch {
            GpuArch::Amd(amd) => self.amd.contains(&amd),
            GpuArch::Cuda(cuda) => self.cuda_min.is_some_and(|min| cuda >= min),
            GpuArch::Metal(_) => false,
        }
    }
}

impl fmt::Display for ArchSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AMD {:?}", self.amd)?;
        match self.cuda_min {
            Some(min) => write!(f, " + CUDA {min}+"),
            None => Ok(()),
        }
    }
}

/// Resolve the concrete [`GpuArch`] backing a [`DeviceSpec`] — AMD from the KFD
/// topology, CUDA from the driver's compute capability (a host/Metal or unreadable
/// device → `None`). The arch is deliberately not in the spec (a hardware
/// property), so it is looked up by `device_id`. [`resolve_supported_arch`] gates
/// on it and returns it; [`check_target`] is the `()`-returning wrapper for callers
/// that only need the gate.
pub fn resolve_arch(spec: &DeviceSpec) -> Option<GpuArch> {
    match spec {
        DeviceSpec::Amd { device_id } => {
            svod_device::registry::resolve_amd_arch_from_topology(*device_id).ok().map(GpuArch::Amd)
        }
        DeviceSpec::Cuda { device_id } => svod_device::registry::resolve_cuda_arch(*device_id).ok().map(GpuArch::Cuda),
        // Metal has an arch of its own but tk has no lowering for it; the gate
        // reports `UnsupportedArch`.
        DeviceSpec::Metal { .. } | DeviceSpec::Cpu | DeviceSpec::WebGpu | DeviceSpec::Disk { .. } => None,
    }
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
        GpuArch::Metal(_) => unreachable!("ArchSet never admits Metal"),
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
