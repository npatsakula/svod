//! Backend-agnostic profiling data shared across the device boundary.
//!
//! These types live in `device` (not `runtime`) because the device traits
//! [`crate::DispatchTimestamps::counters`] and [`crate::Program::resource_usage`]
//! return them. `runtime` re-exports them through its `profiler` module so callers
//! see one profiling vocabulary.

use std::collections::BTreeMap;
use std::sync::Arc;

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
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
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

/// What an AMD SQTT (thread trace) capture records. One SIMD per SE emits
/// instruction tokens; every traced SE records wave and register tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SqttConfig {
    /// SEs that emit instruction tokens (bit `i` = SE `i`).
    pub se_mask: u32,
    /// The SIMD of WGP 0, SA 0 whose instructions each SE traces.
    pub simds: SqttSimds,
    /// Trace buffer per SE, a multiple of 4 KiB.
    pub bytes_per_se: usize,
    /// Narrow `COMPUTE_STATIC_THREAD_MGMT_SE*` so a small grid lands on the
    /// traced WGP (restored after the dispatch).
    pub limit_cu: bool,
    /// The stable power state held while armed; `None` traces at the clock the
    /// device picks, as an untraced run does (a pin slows the traced run by its
    /// lower shader clock alone).
    pub pstate: Option<StablePstate>,
}

impl Default for SqttConfig {
    fn default() -> Self {
        Self { se_mask: 0b11, simds: SqttSimds::default(), bytes_per_se: 64 << 20, limit_cu: false, pstate: None }
    }
}

/// The SIMD (0-3) each SE traces, SE `i` the `i % 8`th. The default alternates SIMDs 0 and 1:
/// on RDNA3/4 SIMD bit 1 picks the WGP's CU, so the SEs between them sample both SIMDs of CU 0,
/// which hold a CU-mode workgroup's waves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SqttSimds(pub [u8; 8]);

impl SqttSimds {
    /// `list` repeated over the SEs.
    pub fn cycle(list: &[u8]) -> Self {
        assert!(!list.is_empty(), "a SIMD list names at least one SIMD");
        Self(std::array::from_fn(|se| list[se % list.len()]))
    }

    pub fn of(self, se: u32) -> u8 {
        self.0[se as usize % self.0.len()]
    }
}

impl Default for SqttSimds {
    fn default() -> Self {
        Self::cycle(&[0, 1])
    }
}

/// A SIMD for every SE, or a comma list repeated over the SEs.
impl std::str::FromStr for SqttSimds {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        let list = s.split(',').map(|v| v.trim().parse::<u8>()).collect::<Result<Vec<_>, _>>();
        match list {
            Ok(list) if (1..=8).contains(&list.len()) => Ok(Self::cycle(&list)),
            _ => Err(format!("{s:?}: a SIMD or a comma list of up to 8")),
        }
    }
}

/// An amdgpu stable power state (`AMDGPU_CTX_STABLE_PSTATE_*`): fixed clocks for
/// repeatable cycle counts, `Standard` below the boost clock, `Peak` at it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StablePstate {
    Standard,
    Peak,
}

impl std::str::FromStr for StablePstate {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        match s.trim() {
            "standard" => Ok(Self::Standard),
            "peak" => Ok(Self::Peak),
            other => Err(format!("{other:?}: standard or peak")),
        }
    }
}

/// One SE's raw thread-trace stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeTrace {
    pub se: u32,
    /// Whether this SE emitted instruction tokens.
    pub itrace: bool,
    pub bytes: Vec<u8>,
    /// The SE filled its buffer: the stream's tail is lost.
    pub truncated: bool,
}

/// The traced kernel's code object and where it ran, for mapping trace PCs
/// back to instructions: a PC `p` is ELF address `p - load_base`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CodeImage {
    pub elf: Arc<[u8]>,
    pub load_base: u64,
    pub entry_pc: u64,
}

/// The thread trace of one dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SqttTrace {
    pub arch: svod_dtype::AmdArch,
    pub per_se: Vec<SeTrace>,
    pub code: Arc<CodeImage>,
    /// The memory clock level current when the dispatch was armed (`pp_dpm_mclk`), where sysfs has one.
    pub mclk_mhz: Option<u64>,
}
