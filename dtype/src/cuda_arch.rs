//! NVIDIA compute capability (`sm_XY`), the CUDA counterpart of [`super::AmdArch`].
//!
//! Open-ended by design: an arch is its `(major, minor)` pair, so a new GPU
//! generation needs no code change here; only the capability predicates
//! encode generation thresholds.

use core::fmt;
use core::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CudaArch {
    pub major: u8,
    pub minor: u8,
}

impl CudaArch {
    /// Statically declared `.shared` memory a kernel may use (ptxas rejects
    /// more); anything above must be dynamic shared memory opted into per
    /// function with `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`.
    pub const MAX_STATIC_SHARED_BYTES: usize = 48 << 10;

    /// Shared memory one block may opt into on this architecture
    /// (`CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`, per the CUDA
    /// programming guide), for planning without a device; an opened device
    /// reports the authoritative value in `CudaLimits::shared_per_block_optin`.
    /// Unknown generations get the portable 48 KB.
    pub const fn max_shared_per_block_optin(self) -> usize {
        match (self.major, self.minor) {
            (7, 0..=2) => 96 << 10,
            (7, 5) => 64 << 10,
            (8, 0) | (8, 7) => 163 << 10,
            (8, 6) | (8, 9) | (12, _) => 99 << 10,
            (9, 0) | (10 | 11, _) => 227 << 10,
            _ => Self::MAX_STATIC_SHARED_BYTES,
        }
    }

    /// From the driver's `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_{MAJOR,MINOR}`.
    pub const fn from_compute_capability(major: u8, minor: u8) -> Self {
        Self { major, minor }
    }

    /// The number in `sm_XY` (`sm_86` → 86, `sm_100` → 100).
    pub const fn sm(self) -> u32 {
        self.major as u32 * 10 + self.minor as u32
    }

    /// Volta (7.0) introduced `mma.sync`.
    pub const fn has_tensor_cores(self) -> bool {
        self.major >= 7
    }

    /// Ampere (8.0) added the bf16 and tf32 MMA shapes.
    pub const fn has_bf16_mma(self) -> bool {
        self.major >= 8
    }

    /// Ada (8.9) added fp8 (`e4m3` / `e5m2`) MMA.
    pub const fn has_fp8(self) -> bool {
        self.major > 8 || (self.major == 8 && self.minor >= 9)
    }

    /// Warp width; fixed across every CUDA generation.
    pub const fn wave_size(self) -> u32 {
        32
    }
}

impl fmt::Display for CudaArch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sm_{}", self.sm())
    }
}

/// The input did not look like `sm_XY`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseCudaArchError(pub String);

impl fmt::Display for ParseCudaArchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid CUDA arch {:?}: expected sm_XY", self.0)
    }
}

impl std::error::Error for ParseCudaArchError {}

/// Parses `sm_XY` (case-insensitive). Feature-set suffixes (`sm_90a`) are
/// rejected: the arch is the capability, not a compilation mode.
impl FromStr for CudaArch {
    type Err = ParseCudaArchError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parse = || {
            let digits = s.strip_prefix("sm_").or_else(|| s.strip_prefix("SM_"))?;
            if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
                return None;
            }
            let sm: u32 = digits.parse().ok()?;
            Some(Self { major: u8::try_from(sm / 10).ok()?, minor: (sm % 10) as u8 })
        };
        parse().ok_or_else(|| ParseCudaArchError(s.to_string()))
    }
}

#[cfg(test)]
#[path = "test/unit/cuda_arch.rs"]
mod tests;
