//! Backend-agnostic profiling data shared across the device boundary.
//!
//! These types live in `device` (not `runtime`) because the device traits
//! [`crate::DispatchTimestamps::counters`] and [`crate::Program::resource_usage`]
//! return them. `runtime` re-exports them through its `profiler` module so callers
//! see one profiling vocabulary.

use std::collections::BTreeMap;

/// An AMD hardware counter: the SQ block (gfx11/RDNA3.5), which answers the
/// ILP/occupancy question: VALU instructions issued vs SQ-busy cycles, plus
/// waves launched.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AmdCounter {
    /// Cycles the shader sequencer (SQ) was busy.
    SqBusyCycles,
    /// Waves launched.
    SqWaves,
    /// VALU instructions issued.
    SqInstsValu,
}

impl AmdCounter {
    /// Short token used in `SVOD_PMC=…` selection and as a table header.
    pub fn token(self) -> &'static str {
        match self {
            Self::SqBusyCycles => "sqbusy",
            Self::SqWaves => "waves",
            Self::SqInstsValu => "valu",
        }
    }

    /// Every AMD counter, in table order.
    pub fn all() -> [AmdCounter; 3] {
        [Self::SqBusyCycles, Self::SqWaves, Self::SqInstsValu]
    }
}

/// A CUDA hardware counter, collected through the CUPTI range profiler. The set
/// covers issue rate, launch geometry, tensor-pipe residency and DRAM traffic —
/// enough to place a kernel on the roofline. All of them schedule in one pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CudaCounter {
    /// Cycles with at least one warp resident on an SM.
    SmCyclesActive,
    /// Warps launched.
    SmWarpsLaunched,
    /// Warp instructions executed.
    SmspInstExecuted,
    /// Cycles the tensor pipe was active.
    SmPipeTensorCyclesActive,
    /// Bytes moved through DRAM.
    DramBytes,
}

impl CudaCounter {
    /// Short token used in `SVOD_PMC=…` selection and as a table header.
    pub fn token(self) -> &'static str {
        match self {
            Self::SmCyclesActive => "cycles",
            Self::SmWarpsLaunched => "warps",
            Self::SmspInstExecuted => "inst",
            Self::SmPipeTensorCyclesActive => "tensor",
            Self::DramBytes => "dram",
        }
    }

    /// The CUPTI metric name. The `.sum` rollup is required: `ConfigAddMetrics`
    /// rejects a bare base name.
    pub fn metric(self) -> &'static str {
        match self {
            Self::SmCyclesActive => "sm__cycles_active.sum",
            Self::SmWarpsLaunched => "sm__warps_launched.sum",
            Self::SmspInstExecuted => "smsp__inst_executed.sum",
            Self::SmPipeTensorCyclesActive => "sm__pipe_tensor_cycles_active.sum",
            Self::DramBytes => "dram__bytes.sum",
        }
    }

    /// Every CUDA counter, in table order.
    pub fn all() -> [CudaCounter; 5] {
        [
            Self::SmCyclesActive,
            Self::SmWarpsLaunched,
            Self::SmspInstExecuted,
            Self::SmPipeTensorCyclesActive,
            Self::DramBytes,
        ]
    }
}

/// A hardware performance counter selectable via PMC. Counters are
/// backend-specific; a selection may name counters the running backend does not
/// implement, and those are dropped when the counters are armed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PmcCounter {
    Amd(AmdCounter),
    Cuda(CudaCounter),
}

impl PmcCounter {
    /// Short token used in `SVOD_PMC=…` selection and as a table header. Tokens
    /// are unique across backends, so [`from_token`](Self::from_token) needs no
    /// device context.
    pub fn token(self) -> &'static str {
        match self {
            Self::Amd(c) => c.token(),
            Self::Cuda(c) => c.token(),
        }
    }

    /// Parse a `SVOD_PMC` token; unknown tokens return `None`.
    pub fn from_token(s: &str) -> Option<Self> {
        let lowered = s.trim().to_ascii_lowercase();
        // `busy` is a legacy alias for the AMD SQ busy-cycles counter.
        let token = if lowered == "busy" { "sqbusy" } else { lowered.as_str() };
        AmdCounter::all()
            .into_iter()
            .map(Self::Amd)
            .chain(CudaCounter::all().into_iter().map(Self::Cuda))
            .find(|c| c.token() == token)
    }
}

/// Hardware counter values harvested for one dispatch. Sparse: only the
/// requested counters are present.
#[derive(Debug, Clone, Default)]
pub struct CounterSet {
    pub values: BTreeMap<PmcCounter, u64>,
}

/// Per-kernel static GPU resource usage, decoded from the compiled program
/// (AMD: the kernel descriptor; Metal: the pipeline state). Pure static — no
/// runtime cost. Fields a backend cannot see are `None`.
#[derive(Debug, Clone, Copy)]
pub struct KernelResources {
    /// Vector GPRs allocated per lane (AMD).
    pub vgprs: Option<u32>,
    /// Accumulator GPRs allocated per lane beside `vgprs` where the matrix
    /// core has a file of its own (CDNA's AGPRs, gfx90a on); 0 when the
    /// kernel uses none, `None` where the target has no such file.
    pub agprs: Option<u32>,
    /// Scalar GPRs allocated (AMD).
    pub sgprs: Option<u32>,
    /// LDS / threadgroup memory bytes per workgroup.
    pub lds_bytes: u32,
    /// Scratch (private segment) bytes per lane (AMD).
    pub scratch_bytes: Option<u32>,
    /// Wave / SIMD-group width (32 or 64).
    pub wave_size: u32,
    /// Register-limited occupancy (0.0–1.0), the first-order limiter only (LDS
    /// and workgroup limits are not modeled). AMD: resident waves per SIMD
    /// versus the register-file maximum, `None` for unknown geometry. Metal:
    /// the pipeline's `maxTotalThreadsPerThreadgroup` over the device's 1024,
    /// which the compiler lowers as register demand grows.
    pub occupancy: Option<f32>,
}
