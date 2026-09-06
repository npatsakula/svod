//! Optimizer configuration types.
//!
//! Provides typed configuration for kernel optimization with bon builders.
//! Supports both explicit configuration and environment variable fallbacks.

use bon::bon;
use svod_ir::Opt;

fn beam_min_progress_from_env() -> u64 {
    parse_beam_min_progress(std::env::var("BEAM_MIN_PROGRESS").ok().as_deref())
}

fn parse_beam_min_progress(value: Option<&str>) -> u64 {
    value
        .and_then(|value| value.parse::<f64>().ok())
        .map(|microseconds| (microseconds * 1_000.0).max(0.0).min(u64::MAX as f64) as u64)
        .unwrap_or(10)
}

// ============================================================================
// OPTIMIZATION STRATEGY
// ============================================================================

/// Optimization strategy for kernel tuning.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum OptStrategy {
    /// No optimization (for debugging/regression testing).
    None,

    /// Hand-coded heuristics (default).
    #[default]
    Heuristic,

    /// Beam search optimization.
    Beam {
        /// Beam width - number of candidates to keep at each step.
        width: usize,
    },
}

impl OptStrategy {
    /// Get optimization strategy from environment variables.
    ///
    /// # Environment Variables
    ///
    /// * `SVOD_NOOPT=1` - Disable all optimizations
    /// * `BEAM=N` - Use beam search with width N
    pub fn from_env() -> Self {
        if std::env::var("SVOD_NOOPT").is_ok() {
            return Self::None;
        }

        if let Ok(beam_str) = std::env::var("BEAM")
            && let Ok(width) = beam_str.parse::<usize>()
            && width > 0
        {
            return Self::Beam { width };
        }

        Self::Heuristic
    }

    /// Check if this strategy disables optimization.
    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }

    /// Check if this strategy uses beam search.
    pub fn is_beam(&self) -> bool {
        matches!(self, Self::Beam { .. })
    }
}

// ============================================================================
// TENSOR CORE SETTINGS
// ============================================================================

/// Tensor core usage level.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum TcUsage {
    /// Disabled (USE_TC=0).
    Disabled,

    /// Enabled (USE_TC=1, default).
    #[default]
    Enabled,

    /// Shape-only mode (USE_TC=2).
    ShapeOnly,
}

impl TcUsage {
    /// Convert to integer value for internal APIs.
    pub fn as_usize(&self) -> usize {
        match self {
            Self::Disabled => 0,
            Self::Enabled => 1,
            Self::ShapeOnly => 2,
        }
    }
}

/// Tensor core optimization level.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum TcOpt {
    /// Strict matching (TC_OPT=0, default).
    #[default]
    Strict,

    /// Relaxed matching (TC_OPT=1).
    Relaxed,

    /// Padded matching (TC_OPT=2).
    Padded,
}

impl TcOpt {
    /// Convert to integer value for internal APIs.
    pub fn as_usize(&self) -> usize {
        match self {
            Self::Strict => 0,
            Self::Relaxed => 1,
            Self::Padded => 2,
        }
    }
}

/// Tensor core selection mode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum TcSelect {
    /// Auto-select best tensor core (TC_SELECT=-1, default).
    #[default]
    Auto,

    /// Use specific tensor core index.
    Index(usize),
}

impl TcSelect {
    /// Convert to integer value for internal APIs.
    pub fn as_i32(&self) -> i32 {
        match self {
            Self::Auto => -1,
            Self::Index(idx) => *idx as i32,
        }
    }
}

// ============================================================================
// BEAM SEARCH CONFIGURATION
// ============================================================================

/// Configuration for beam search auto-tuning.
///
/// No total search timeout — the loop terminates only on the `min_progress`
/// floor or an empty candidate set. `BEAM_TIMEOUT_SEC` is a per-candidate
/// compile alarm enforced inside the BEAM compile worker,
/// not a global search budget.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct BeamConfig {
    /// Beam width - number of candidates to keep at each step.
    pub beam_width: usize,
    /// Maximum upcast size (product of UPCAST/UNROLL dimensions).
    pub max_upcast: usize,
    /// Maximum local size (product of LOCAL/WARP/GROUP_REDUCE dimensions).
    pub max_local: usize,
    /// Maximum UOps in kernel before rejecting.
    pub max_uops: usize,
    /// Number of benchmark runs per kernel.
    pub num_runs: usize,
    /// Minimum improvement in nanoseconds required to continue searching.
    pub min_progress_ns: u64,
    /// Whether the NOLOCALS action is part of the search space.
    pub enable_nolocals: bool,
    /// Maximum number of candidates compiled concurrently.
    pub compile_workers: usize,
    /// Per-candidate backend compilation timeout in seconds.
    pub compile_timeout_secs: u64,
    /// Tasks handled before replacing a clean spawned worker.
    pub max_tasks_per_child: usize,
    /// Disable disk cache.
    pub disable_cache: bool,
}

impl Default for BeamConfig {
    fn default() -> Self {
        Self {
            beam_width: 4,
            max_upcast: 256,
            max_local: 1024,
            max_uops: 3000,
            num_runs: 3,
            min_progress_ns: 10,
            enable_nolocals: false,
            compile_workers: 0,
            compile_timeout_secs: 10,
            max_tasks_per_child: 16,
            disable_cache: false,
        }
    }
}

#[bon]
impl BeamConfig {
    /// Create a beam configuration with builder pattern.
    ///
    /// Defaults consult the same env vars as `from_env()` so callers
    /// like benches can be overridden via `BEAM_*` without changing
    /// builder call sites.
    #[builder]
    pub fn builder(
        #[builder(default = std::env::var("BEAM").ok().and_then(|s| s.parse().ok()).unwrap_or(4))] beam_width: usize,
        #[builder(default = std::env::var("BEAM_UPCAST_MAX").ok().and_then(|s| s.parse().ok()).unwrap_or(256))]
        max_upcast: usize,
        #[builder(default = std::env::var("BEAM_LOCAL_MAX").ok().and_then(|s| s.parse().ok()).unwrap_or(1024))]
        max_local: usize,
        #[builder(default = std::env::var("BEAM_UOPS_MAX").ok().and_then(|s| s.parse().ok()).unwrap_or(3000))]
        max_uops: usize,
        #[builder(default = std::env::var("BEAM_RUNS").ok().and_then(|s| s.parse().ok()).unwrap_or(3))] num_runs: usize,
        #[builder(default = beam_min_progress_from_env())] min_progress_ns: u64,
        #[builder(default = std::env::var("NOLOCALS").is_ok() || std::env::var("SVOD_NOLOCALS").is_ok())]
        enable_nolocals: bool,
        #[builder(default = std::env::var("PARALLEL").ok().and_then(|s| s.parse().ok()).unwrap_or(0))]
        compile_workers: usize,
        #[builder(default = std::env::var("BEAM_TIMEOUT_SEC").ok().and_then(|s| s.parse().ok()).unwrap_or(10))]
        compile_timeout_secs: u64,
        #[builder(default = std::env::var("BEAM_MAX_TASKS_PER_CHILD").ok().and_then(|s| s.parse().ok()).unwrap_or(16))]
        max_tasks_per_child: usize,
        #[builder(default = std::env::var("IGNORE_BEAM_CACHE").is_ok())] disable_cache: bool,
    ) -> Self {
        Self {
            beam_width,
            max_upcast,
            max_local,
            max_uops,
            num_runs,
            min_progress_ns,
            enable_nolocals,
            compile_workers,
            compile_timeout_secs,
            max_tasks_per_child,
            disable_cache,
        }
    }

    /// Create configuration from environment variables.
    ///
    /// # Environment Variables
    ///
    /// * `BEAM` - Beam width (default: 4)
    /// * `BEAM_UPCAST_MAX` - Max upcast size (default: 256)
    /// * `BEAM_LOCAL_MAX` - Max local memory elements (default: 1024)
    /// * `BEAM_UOPS_MAX` - Max UOps before rejecting (default: 3000)
    /// * `BEAM_RUNS` - Benchmark runs per kernel (default: 3)
    /// * `BEAM_MIN_PROGRESS` - Minimum progress in microseconds (default: 0.01)
    /// * `NOLOCALS` / `SVOD_NOLOCALS` - Include the NOLOCALS action if set
    /// * `PARALLEL` - Maximum concurrent candidate compilations (default: 0 for CPU; GPU resolves to host parallelism)
    /// * `BEAM_TIMEOUT_SEC` - Per-candidate compile timeout in seconds (default: 10)
    /// * `IGNORE_BEAM_CACHE` - Bypass disk cache if set
    pub fn from_env() -> Self {
        let beam_width = std::env::var("BEAM").ok().and_then(|s| s.parse().ok()).unwrap_or(4);
        let max_upcast = std::env::var("BEAM_UPCAST_MAX").ok().and_then(|s| s.parse().ok()).unwrap_or(256);
        let max_local = std::env::var("BEAM_LOCAL_MAX").ok().and_then(|s| s.parse().ok()).unwrap_or(1024);
        let max_uops = std::env::var("BEAM_UOPS_MAX").ok().and_then(|s| s.parse().ok()).unwrap_or(3000);
        let num_runs = std::env::var("BEAM_RUNS").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
        let min_progress_ns = beam_min_progress_from_env();
        let enable_nolocals = std::env::var("NOLOCALS").is_ok() || std::env::var("SVOD_NOLOCALS").is_ok();
        let compile_workers = std::env::var("PARALLEL").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
        let compile_timeout_secs = std::env::var("BEAM_TIMEOUT_SEC").ok().and_then(|s| s.parse().ok()).unwrap_or(10);
        let max_tasks_per_child =
            std::env::var("BEAM_MAX_TASKS_PER_CHILD").ok().and_then(|s| s.parse().ok()).unwrap_or(16);
        let disable_cache = std::env::var("IGNORE_BEAM_CACHE").is_ok();

        Self {
            beam_width,
            max_upcast,
            max_local,
            max_uops,
            num_runs,
            min_progress_ns,
            enable_nolocals,
            compile_workers,
            compile_timeout_secs,
            max_tasks_per_child,
            disable_cache,
        }
    }

    /// Get beam width from strategy if applicable.
    pub fn with_strategy_width(mut self, strategy: &OptStrategy) -> Self {
        if let OptStrategy::Beam { width } = strategy {
            self.beam_width = *width;
        }
        self
    }
}

// ============================================================================
// HEURISTICS CONFIGURATION
// ============================================================================

/// Configuration for heuristic-based optimization.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct HeuristicsConfig {
    // Tensor cores
    /// Tensor core usage level.
    pub tc_enabled: TcUsage,
    /// Tensor core optimization level.
    ///
    /// Defaults to [`TcOpt::Strict`] (`TC_OPT=0`), matching tinygrad
    /// `helpers.py:238` (`ContextVar("TC_OPT", 0)`), which `heuristic.py:28-32`
    /// reads on the hand-coded path. `TC_OPT=2` is only the *BEAM action space*
    /// default (`search.py:22`), so the heuristic path must not inherit it —
    /// tensor cores would silently PADTO a shape the author did not ask to pad.
    /// Benchmarks that want the padded behaviour set it explicitly.
    pub tc_opt: TcOpt,
    /// Tensor core selection mode.
    pub tc_select: TcSelect,

    // Matrix-vector optimization
    /// Enable matrix-vector optimization.
    pub matvec_enabled: bool,
    /// Matrix-vector block size (rows per workgroup).
    pub matvec_blocksize: usize,
    /// Matrix-vector reduction split (threads per reduction row).
    pub threads_per_row: usize,
    /// Matrix-vector output lane split (rows computed per thread).
    pub rows_per_thread: usize,

    // Reduction thresholds
    /// Threshold for applying grouped reduction.
    pub grouped_threshold: usize,
    /// Threshold for applying unroll.
    pub unroll_threshold: usize,

    // Local memory
    /// Disable local memory globally.
    pub disable_locals: bool,

    // Threading
    /// Number of `core_id` chunks a CPU kernel is split into; baked into the
    /// kernel and its cache identity. Defaults to [`thread_budget`] and may
    /// legitimately differ from the runtime pool size: a kernel split N ways
    /// runs correctly on fewer threads. Set to 1 to disable threading.
    pub thread_count: usize,

    // Vectorization
    /// Enable K-axis vectorization for matmul.
    /// When enabled, UPCAST is applied to the reduce (K) axis creating vector accumulators.
    /// Disabled by default: K-vectorization complicates output tiling and horizontal reduce.
    /// Default: false.
    pub k_vectorize: bool,

    /// Enable output dimension upcasting for matmul (register blocking).
    /// When enabled, UPCAST is applied to M/N axes creating register tiles.
    /// Each thread computes an MxN tile instead of a single element.
    /// Default: false (blocked by vector width mismatch issue in expand.rs).
    pub output_upcast: bool,

    // Debug
    /// Debug verbosity level.
    pub debug_level: u8,
}

/// The process-wide thread budget: `SVOD_THREADS` as a positive integer, else
/// the host's available parallelism. Shared by kernel preparation (parallel
/// optimise/compile), CPU execution, and the default kernel `core_id` split.
pub fn thread_budget() -> usize {
    parse_thread_budget(std::env::var("SVOD_THREADS").ok().as_deref())
}

fn parse_thread_budget(value: Option<&str>) -> usize {
    value
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&threads| threads > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map(|p| p.get()).unwrap_or(8))
}

impl HeuristicsConfig {
    /// Create configuration from environment variables.
    ///
    /// # Environment Variables
    ///
    /// * `SVOD_THREADS` - Kernel `core_id` split, the process thread budget (default: available_parallelism)
    /// * `SVOD_MV` - Enable/disable matvec fast-path (`0` disables)
    /// * `SVOD_MV_BLOCKSIZE` / `MV_BLOCKSIZE` - Matvec local block size
    /// * `SVOD_MV_THREADS_PER_ROW` / `MV_THREADS_PER_ROW` - Matvec reduce split
    /// * `SVOD_MV_ROWS_PER_THREAD` / `MV_ROWS_PER_THREAD` - Matvec output split
    /// * `SVOD_K_VECTORIZE` - Enable K-axis vectorization (default: disabled)
    /// * `SVOD_NO_OUTPUT_UPCAST` - Disable output dimension upcasting (default: enabled)
    /// * `SVOD_NOLOCALS` - Disable LOCAL axis selection after grouped-reduction matching
    /// * `SVOD_TC` - Tensor-core usage: `0` disables, `2` shape-only, else enabled
    /// * `TC_OPT` / `SVOD_TC_OPT` - Strict (`0`), relaxed (`1`), or padded (`2`)
    /// * `TC_SELECT` / `SVOD_TC_SELECT` - Auto (`-1`) or a tensor-core index
    pub fn from_env() -> Self {
        let parse_usize = |keys: &[&str], default: usize| {
            keys.iter().find_map(|k| std::env::var(k).ok().and_then(|v| v.parse::<usize>().ok())).unwrap_or(default)
        };

        let thread_count = thread_budget();
        let matvec_enabled = std::env::var("SVOD_MV").map(|v| v != "0").unwrap_or(true);
        let matvec_blocksize = parse_usize(&["SVOD_MV_BLOCKSIZE", "MV_BLOCKSIZE"], 4);
        let threads_per_row = parse_usize(&["SVOD_MV_THREADS_PER_ROW", "MV_THREADS_PER_ROW"], 8);
        let rows_per_thread = parse_usize(&["SVOD_MV_ROWS_PER_THREAD", "MV_ROWS_PER_THREAD"], 4);
        let k_vectorize = std::env::var("SVOD_K_VECTORIZE").is_ok();
        // Default enabled, use SVOD_NO_OUTPUT_UPCAST to disable
        let output_upcast = std::env::var("SVOD_NO_OUTPUT_UPCAST").is_err();
        let disable_locals = std::env::var("SVOD_NOLOCALS").is_ok();
        // Tensor-core usage: `SVOD_TC=0` disables (vector matmul), `2` is
        // shape-only, anything else (or unset) keeps the default Enabled. Lets
        // bit-identity tests pin the numerics to the non-MFMA path.
        let tc_enabled = match std::env::var("SVOD_TC").ok().as_deref() {
            Some("0") => TcUsage::Disabled,
            Some("2") => TcUsage::ShapeOnly,
            _ => TcUsage::Enabled,
        };
        let tc_opt = match parse_usize(&["SVOD_TC_OPT", "TC_OPT"], 0) {
            1 => TcOpt::Relaxed,
            2 => TcOpt::Padded,
            _ => TcOpt::Strict,
        };
        let tc_select = ["SVOD_TC_SELECT", "TC_SELECT"]
            .iter()
            .find_map(|key| std::env::var(key).ok().and_then(|value| value.parse::<i32>().ok()))
            .and_then(|value| usize::try_from(value).ok())
            .map(TcSelect::Index)
            .unwrap_or(TcSelect::Auto);

        Self {
            matvec_enabled,
            matvec_blocksize,
            threads_per_row,
            rows_per_thread,
            thread_count,
            k_vectorize,
            output_upcast,
            disable_locals,
            tc_enabled,
            tc_opt,
            tc_select,
            ..Default::default()
        }
    }
}

impl Default for HeuristicsConfig {
    fn default() -> Self {
        Self {
            tc_enabled: TcUsage::Enabled,
            tc_opt: TcOpt::Strict,
            tc_select: TcSelect::Auto,
            matvec_enabled: true,
            matvec_blocksize: 4,
            threads_per_row: 8,
            rows_per_thread: 4,
            grouped_threshold: 256,
            unroll_threshold: 32,
            disable_locals: false,
            thread_count: thread_budget(),
            k_vectorize: false,
            output_upcast: true,
            debug_level: 0,
        }
    }
}

#[bon]
impl HeuristicsConfig {
    /// Create a heuristics configuration with builder pattern.
    #[builder]
    pub fn builder(
        #[builder(default)] tc_enabled: TcUsage,
        #[builder(default)] tc_opt: TcOpt,
        #[builder(default)] tc_select: TcSelect,
        #[builder(default = true)] matvec_enabled: bool,
        #[builder(default = 4)] matvec_blocksize: usize,
        #[builder(default = 8)] threads_per_row: usize,
        #[builder(default = 4)] rows_per_thread: usize,
        #[builder(default = 256)] grouped_threshold: usize,
        #[builder(default = 32)] unroll_threshold: usize,
        #[builder(default = false)] disable_locals: bool,
        #[builder(default = thread_budget())] thread_count: usize,
        #[builder(default = false)] k_vectorize: bool,
        #[builder(default = true)] output_upcast: bool,
        #[builder(default = 0)] debug_level: u8,
    ) -> Self {
        Self {
            tc_enabled,
            tc_opt,
            tc_select,
            matvec_enabled,
            matvec_blocksize,
            threads_per_row,
            rows_per_thread,
            grouped_threshold,
            unroll_threshold,
            disable_locals,
            thread_count,
            k_vectorize,
            output_upcast,
            debug_level,
        }
    }
}

// ============================================================================
// TOP-LEVEL OPTIMIZER CONFIGURATION
// ============================================================================

/// Top-level optimizer configuration.
///
/// Combines strategy selection, beam search settings, and heuristic parameters.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OptimizerConfig {
    /// Optimization strategy (None, Heuristic, or Beam).
    pub strategy: OptStrategy,
    /// Beam search configuration (used when strategy is Beam).
    pub beam: BeamConfig,
    /// Heuristics configuration (used when strategy is Heuristic).
    pub heuristics: HeuristicsConfig,
    /// Tinygrad-compatible transcendental mode. Values >= 2 force decomposition.
    pub transcendental: i32,
    /// Disable non-power-of-two magic integer division rewrites.
    ///
    /// Defaults to `true`, matching tinygrad `helpers.py:245`
    /// (`DISABLE_FAST_IDIV = ContextVar("DISABLE_FAST_IDIV", 1)`); its own tests
    /// opt in with `Context(DISABLE_FAST_IDIV=0)`.
    pub disable_fast_idiv: bool,
    /// Apply exactly these opts to every kernel instead of `strategy` — the
    /// config-level analog of a SINK's `KernelInfo.opts_to_apply` (which still
    /// wins per kernel). Replays a beam plan deterministically, e.g. in a
    /// regression test for a plan that once miscompiled.
    pub opts_to_apply: Option<Vec<Opt>>,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            strategy: OptStrategy::default(),
            beam: BeamConfig::default(),
            heuristics: HeuristicsConfig::default(),
            transcendental: 1,
            disable_fast_idiv: true,
            opts_to_apply: None,
        }
    }
}

#[bon]
impl OptimizerConfig {
    /// Create an optimizer configuration with builder pattern.
    ///
    /// `beam` and `heuristics` defaults consult env vars (matching
    /// `*::from_env()`) so callers like benches and end-to-end examples
    /// pick up `IGNORE_BEAM_CACHE`, `BEAM_TIMEOUT_SEC`, `BEAM_UOPS_MAX`,
    /// etc. without explicit field setting.
    #[builder]
    pub fn builder(
        #[builder(default)] strategy: OptStrategy,
        #[builder(default = BeamConfig::from_env())] beam: BeamConfig,
        #[builder(default = HeuristicsConfig::from_env())] heuristics: HeuristicsConfig,
        #[builder(default = std::env::var("TRANSCENDENTAL").ok().and_then(|value| value.parse().ok()).unwrap_or(1))]
        transcendental: i32,
        #[builder(default = std::env::var("DISABLE_FAST_IDIV").ok().and_then(|value| value.parse::<i32>().ok()).unwrap_or(1) != 0)]
        disable_fast_idiv: bool,
        opts_to_apply: Option<Vec<Opt>>,
    ) -> Self {
        let beam = beam.with_strategy_width(&strategy);
        Self { strategy, beam, heuristics, transcendental, disable_fast_idiv, opts_to_apply }
    }

    /// Create configuration from environment variables.
    ///
    /// Reads strategy from env, then populates beam and heuristics config accordingly.
    ///
    /// # Environment Variables
    ///
    /// * `SVOD_NOOPT=1` - Disable all optimizations
    /// * `BEAM=N` - Use beam search with width N
    pub fn from_env() -> Self {
        let strategy = OptStrategy::from_env();
        let beam = BeamConfig::from_env().with_strategy_width(&strategy);
        let heuristics = HeuristicsConfig::from_env();
        let transcendental = std::env::var("TRANSCENDENTAL").ok().and_then(|value| value.parse().ok()).unwrap_or(1);
        let disable_fast_idiv =
            std::env::var("DISABLE_FAST_IDIV").ok().and_then(|value| value.parse::<i32>().ok()).unwrap_or(1) != 0;

        Self { strategy, beam, heuristics, transcendental, disable_fast_idiv, opts_to_apply: None }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
#[path = "../test/unit/optimizer/config_internal.rs"]
mod tests;
