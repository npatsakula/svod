//! LLVM target selection (CPU, AMD GPU, NVIDIA GPU).
//!
//! Threaded through the renderer so that op-emission helpers (address spaces,
//! kernel attributes, intrinsic names) can branch on the target without
//! introducing separate renderer types.

use svod_dtype::{AddrSpace, AmdArch, CudaArch};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LlvmTarget {
    /// Host CPU. Uses x86/AArch64 ELF triple, generic LLVM IR.
    Cpu,
    /// AMD GPU at the named `gfx{family}` target. Uses
    /// `amdgcn-amd-amdhsa` triple, `amdgpu_kernel` calling convention,
    /// and amdgcn-specific intrinsics for SPECIAL/BARRIER/WMMA.
    Amd(AmdArch),
    /// NVIDIA GPU at the named `sm_XY` compute capability. Uses the
    /// `nvptx64-nvidia-cuda` triple, `ptx_kernel` calling convention, and
    /// nvvm intrinsics for SPECIAL/BARRIER/WMMA; clang emits PTX text that
    /// the CUDA driver JITs at module load.
    Nvptx(CudaArch),
}

impl LlvmTarget {
    pub fn is_amd(&self) -> bool {
        matches!(self, Self::Amd(_))
    }

    pub fn is_nvptx(&self) -> bool {
        matches!(self, Self::Nvptx(_))
    }

    pub fn amd_arch(&self) -> Option<AmdArch> {
        match self {
            Self::Amd(a) => Some(*a),
            Self::Cpu | Self::Nvptx(_) => None,
        }
    }

    pub fn cuda_arch(&self) -> Option<CudaArch> {
        match self {
            Self::Nvptx(a) => Some(*a),
            Self::Cpu | Self::Amd(_) => None,
        }
    }
}

impl std::fmt::Display for LlvmTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => f.write_str("cpu"),
            Self::Amd(arch) => write!(f, "{arch}"),
            Self::Nvptx(arch) => write!(f, "{arch}"),
        }
    }
}

/// Numeric address space encoded in LLVM IR pointer types for this target.
///
/// CPU: addrspace(0) is the generic flat space; LLVM's IR-level distinction
/// between Global and Local doesn't really apply (we use `alloca` for Local).
/// AMD: AMDGPU mandates explicit address spaces — Global=1, Constant=4,
/// Local=3, Private=5, Generic=0. Kernel-arg pointers are passed unannotated
/// (`ptr`) and the backend implicitly promotes to addrspace(1).
/// NVPTX numbers its spaces identically (global=1, shared=3, local=5).
///
/// See <https://llvm.org/docs/AMDGPUUsage.html#address-spaces> and
/// <https://llvm.org/docs/NVPTXUsage.html#address-spaces>.
pub fn addr_space_num(target: LlvmTarget, addrspace: AddrSpace) -> u32 {
    match (target, addrspace) {
        (LlvmTarget::Cpu, AddrSpace::Global) => 0,
        (LlvmTarget::Cpu, AddrSpace::Local) => 3,
        (LlvmTarget::Cpu, AddrSpace::Reg) => 5,
        (LlvmTarget::Amd(_) | LlvmTarget::Nvptx(_), AddrSpace::Global) => 1,
        (LlvmTarget::Amd(_) | LlvmTarget::Nvptx(_), AddrSpace::Local) => 3,
        (LlvmTarget::Amd(_) | LlvmTarget::Nvptx(_), AddrSpace::Reg) => 5,
    }
}
