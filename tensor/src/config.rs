use std::sync::Arc;

use snafu::ResultExt;
use svod_device::device::Device;
use svod_device::registry::DeviceRegistry;
use svod_ir::DeviceSpec;
use svod_runtime::CpuBackend;
use svod_schedule::OptimizerConfig;

use crate::error::{DeviceFactorySnafu, DeviceSnafu};
use crate::memory_planner::PlannerMode;

/// Resolves a `DeviceSpec` into a concrete `Device` for compilation.
///
/// Implementations control which codegen backend is used for each device type.
/// This enables per-call backend selection instead of relying on the
/// `DEVICE_FACTORIES` singleton (which bakes one backend per device spec).
pub(crate) trait DeviceResolver: Send + Sync {
    fn resolve(&self, spec: &DeviceSpec, registry: &DeviceRegistry) -> crate::Result<Arc<Device>>;
}

/// Default resolver: delegates to `DEVICE_FACTORIES` singleton (reads env vars
/// like `SVOD_CPU_BACKEND` at first device creation, then caches).
struct EnvResolver;

impl DeviceResolver for EnvResolver {
    fn resolve(&self, spec: &DeviceSpec, registry: &DeviceRegistry) -> crate::Result<Arc<Device>> {
        svod_runtime::DEVICE_FACTORIES.device(spec, registry).context(DeviceFactorySnafu)
    }
}

/// Creates CPU devices with a specific backend; delegates other device types
/// to `DEVICE_FACTORIES`. This is the resolver used by `PrepareConfig::for_cpu_backend()`.
struct CpuBackendResolver(CpuBackend);

impl DeviceResolver for CpuBackendResolver {
    fn resolve(&self, spec: &DeviceSpec, registry: &DeviceRegistry) -> crate::Result<Arc<Device>> {
        match spec {
            DeviceSpec::Cpu => svod_runtime::cpu_device_with_backend(registry, self.0).context(DeviceSnafu),
            _ => svod_runtime::DEVICE_FACTORIES.device(spec, registry).context(DeviceFactorySnafu),
        }
    }
}

/// Configuration for `prepare()`/`realize()` that bundles optimizer settings
/// with device resolution (codegen backend selection).
///
/// Instead of relying on the `SVOD_CPU_BACKEND` env var (global mutable state),
/// the backend is selected per-call via a [`DeviceResolver`].
#[allow(rustdoc::private_intra_doc_links)]
#[derive(Clone)]
pub struct PrepareConfig {
    pub optimizer: OptimizerConfig,
    pub(crate) resolver: Arc<dyn DeviceResolver>,
    /// Memory planning policy for this preparation. Keeping this in the config
    /// makes planner-on/off comparisons deterministic without process-global
    /// environment mutation.
    pub planner_mode: PlannerMode,
    /// When `true`, force the cache-cold rangeify/scheduling path even if
    /// `SVOD_DISABLE_SCHEDULE_CACHE` is unset. Primarily useful in tests
    /// that need to compare cache-warm vs cache-cold outputs without mutating
    /// process-global env state.
    pub disable_schedule_cache: bool,
    /// Allocate the plan's OUTPUT buffers device-local (`cpu_access: false`)
    /// on backends that support it. The host then reads results via `copyout`
    /// (staged over the copy engine — SDMA on AMD, ~PCIe speed) instead of
    /// the uncached host mapping. Use for big outputs read back in bulk.
    pub device_local_outputs: bool,
    /// The thread budget this prepare sizes the process-wide pool to — the
    /// pool that optimizes/renders/compiles the kernels it misses in the
    /// optimized-kernel cache and later runs `core_id`-split CPU kernels.
    /// Defaults to [`svod_schedule::thread_budget`] (`SVOD_THREADS`); `1`
    /// compiles inline in schedule order. The pool is built by the first
    /// prepare in the process; later values only warn if they differ.
    pub threads: usize,
}

impl std::fmt::Debug for PrepareConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrepareConfig")
            .field("optimizer", &self.optimizer)
            .field("planner_mode", &self.planner_mode)
            .field("disable_schedule_cache", &self.disable_schedule_cache)
            .field("device_local_outputs", &self.device_local_outputs)
            .field("threads", &self.threads)
            .finish_non_exhaustive()
    }
}

impl Default for PrepareConfig {
    fn default() -> Self {
        Self {
            optimizer: OptimizerConfig::default(),
            resolver: Arc::new(EnvResolver),
            planner_mode: crate::memory_planner::mode_from_env(),
            disable_schedule_cache: false,
            device_local_outputs: false,
            threads: svod_schedule::thread_budget(),
        }
    }
}

impl PrepareConfig {
    /// Read `SVOD_MEMORY_PLANNER`, `SVOD_CPU_BACKEND`, and optimizer env vars.
    pub fn from_env() -> Self {
        Self {
            optimizer: OptimizerConfig::from_env(),
            resolver: Arc::new(EnvResolver),
            planner_mode: crate::memory_planner::mode_from_env(),
            disable_schedule_cache: false,
            device_local_outputs: false,
            threads: svod_schedule::thread_budget(),
        }
    }

    /// Convenience constructor: specific CPU backend with optimizer settings
    /// resolved from env (`BEAM`, `SVOD_NOOPT`, `IGNORE_BEAM_CACHE`,
    /// `BEAM_*`, `SVOD_*`). Used by the `codegen_tests!` macro so a single
    /// `BEAM=4 cargo test` flips every codegen-test target to BEAM
    /// without changing test bodies.
    pub fn for_cpu_backend(backend: CpuBackend) -> Self {
        Self {
            optimizer: OptimizerConfig::from_env(),
            resolver: Arc::new(CpuBackendResolver(backend)),
            planner_mode: crate::memory_planner::mode_from_env(),
            disable_schedule_cache: false,
            device_local_outputs: false,
            threads: svod_schedule::thread_budget(),
        }
    }

    /// AMD variant for the `codegen_tests!` macro. The test runs only when the
    /// active default is a topology-supported AMD device; otherwise it skips.
    pub fn for_amd_if_available() -> Option<Self> {
        let DeviceSpec::Amd { device_id } = svod_dtype::default_device::default_device() else { return None };
        svod_device::registry::resolve_amd_arch_from_topology(device_id).ok()?;
        Some(Self::from_env())
    }

    /// Metal variant for the `codegen_tests!` macro. The test runs only when the
    /// active default is a Metal device and an Apple GPU is present.
    pub fn for_metal_if_available() -> Option<Self> {
        let DeviceSpec::Metal { .. } = svod_dtype::default_device::default_device() else { return None };
        svod_device::metal::has_devices().then(Self::from_env)
    }

    /// CUDA variant for the `codegen_tests!` macro. The test runs only when the
    /// active default is a CUDA device the driver can open.
    pub fn for_cuda_if_available() -> Option<Self> {
        let DeviceSpec::Cuda { device_id } = svod_dtype::default_device::default_device() else { return None };
        svod_device::registry::resolve_cuda_arch(device_id).ok()?;
        Some(Self::from_env())
    }
}

/// Whether a device can hold buffers of `dtype`, per its optimizer profile
/// (Metal and WebGPU have no `double`; CUDA below sm_80 has no bf16). CUDA
/// resolves the opened device's compute capability; the other GPU families
/// are family-level, so arch-specific refinements (AMD fp8 variants) need the
/// opened device's renderer.
pub fn device_supports_storage_dtype(spec: &DeviceSpec, dtype: svod_dtype::ScalarDType) -> bool {
    use svod_schedule::OptimizerRenderer;
    let profile = match spec {
        DeviceSpec::Cpu | DeviceSpec::Disk { .. } => OptimizerRenderer::cpu(),
        DeviceSpec::Cuda { device_id } => svod_device::registry::resolve_cuda_arch(*device_id)
            .map(OptimizerRenderer::for_cuda_arch)
            .unwrap_or_else(|_| OptimizerRenderer::cuda()),
        DeviceSpec::Amd { .. } => OptimizerRenderer::amd_rdna3(),
        DeviceSpec::Metal { .. } => OptimizerRenderer::metal(),
        DeviceSpec::WebGpu => OptimizerRenderer::webgpu(),
    };
    profile.supports_storage_dtype(dtype)
}

/// Detect a supported AMD GPU on this host. Returns the gfx-family arch of
/// device 0 when (a) `/dev/kfd` exists, (b) KFD topology has a GPU node, and
/// (c) the gfx target maps to one of `AmdArch`'s supported families
/// (RDNA3 + CDNA).
pub fn amd_test_arch() -> Option<svod_dtype::AmdArch> {
    let nodes = svod_device::amd::topology::enumerate();
    nodes.into_iter().find_map(|n| svod_dtype::AmdArch::from_gfx_target_version(n.gfx_target_version))
}

/// Detect a CUDA GPU on this host: the compute capability of device 0 when
/// the driver loads and reports one.
pub fn cuda_test_arch() -> Option<svod_dtype::CudaArch> {
    svod_device::cuda::has_devices().then(|| svod_device::registry::resolve_cuda_arch(0).ok()).flatten()
}

impl PrepareConfig {
    /// Resolve a `DeviceSpec` into a `Device` using this config's resolver.
    pub(crate) fn resolve_device(&self, spec: &DeviceSpec, registry: &DeviceRegistry) -> crate::Result<Arc<Device>> {
        self.resolver.resolve(spec, registry)
    }
}

impl From<OptimizerConfig> for PrepareConfig {
    fn from(optimizer: OptimizerConfig) -> Self {
        Self {
            optimizer,
            resolver: Arc::new(EnvResolver),
            planner_mode: crate::memory_planner::mode_from_env(),
            disable_schedule_cache: false,
            device_local_outputs: false,
            threads: svod_schedule::thread_budget(),
        }
    }
}

/// Generate one test per codegen backend (Clang, LLVM) from a single test body.
///
/// Supports three forms:
///
/// **Simple test** (config only, no extra params):
/// ```ignore
/// codegen_tests! {
///     fn test_add(config) {
///         let mut a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
///         a.realize_with(&config).unwrap();
///         let result: Vec<f32> = a.as_vec().unwrap();
///     }
/// }
/// // Generates: test_add::clang, test_add::llvm
/// ```
///
/// **Parameterized test** (extra typed params, use with `#[test_case]`):
/// ```ignore
/// codegen_tests! {
///     #[test_case(128, 0.5; "128x128")]
///     fn test_matmul(config, size: usize, tol: f32) {
///         let mut result = run_matmul(size);
///         result.realize_with(&config).unwrap();
///         assert_close(&result, tol);
///     }
/// }
/// // Generates: test_matmul::clang::test_matmul, test_matmul::llvm::test_matmul
/// ```
///
/// **Proptest** (property-based, params use `in` syntax):
/// ```ignore
/// codegen_tests! {
///     #[proptest_config(ProptestConfig::with_cases(50))]
///     fn test_sort_random(config, data in proptest::collection::vec(-100.0f32..100.0, 1..=16)) {
///         let mut t = Tensor::from_slice(&data);
///         let (sorted, _) = t.sort(-1, false).unwrap();
///         // ...
///     }
/// }
/// // Generates: test_sort_random::clang, test_sort_random::llvm
/// ```
#[macro_export]
macro_rules! codegen_tests {
    // Base case
    () => {};

    // Simple test (config only, no extra params)
    ($(#[$meta:meta])* fn $name:ident($config:ident) $body:block $($rest:tt)*) => {
        mod $name {
            #[allow(unused_imports)]
            use super::*;

            #[test]
            $(#[$meta])*
            fn clang() {
                ::svod_schedule::testing::setup_test_tracing();
                let $config = $crate::PrepareConfig::for_cpu_backend($crate::CpuBackend::Clang);
                $body
            }

            #[test]
            $(#[$meta])*
            fn llvm() {
                ::svod_schedule::testing::setup_test_tracing();
                let $config = $crate::PrepareConfig::for_cpu_backend($crate::CpuBackend::Llvm);
                $body
            }

            /// AMD variant — runs only when a supported AMD GPU is detected
            /// on this host (RDNA3 + CDNA). On unsupported hardware or hosts
            /// without `/dev/kfd` this test exits with a skip message rather
            /// than a failure, so the unified test suite still runs on any
            /// CI runner.
            #[test]
            $(#[$meta])*
            fn amd() {
                ::svod_schedule::testing::setup_test_tracing();
                let $config = match $crate::PrepareConfig::for_amd_if_available() {
                    Some(cfg) => cfg,
                    None => {
                        eprintln!("amd codegen_tests variant: skipped (no supported AMD GPU)");
                        return;
                    }
                };
                $body
            }

            /// Metal variant — runs only under `SVOD_DEVICE=METAL:N` on a host
            /// with an Apple GPU; skips otherwise.
            #[test]
            $(#[$meta])*
            fn metal() {
                ::svod_schedule::testing::setup_test_tracing();
                let $config = match $crate::PrepareConfig::for_metal_if_available() {
                    Some(cfg) => cfg,
                    None => {
                        eprintln!("metal codegen_tests variant: skipped (no Metal device)");
                        return;
                    }
                };
                $body
            }

            /// CUDA variant — runs only under `SVOD_DEVICE=CUDA:N` on a host
            /// with an NVIDIA GPU; skips otherwise.
            #[test]
            $(#[$meta])*
            fn cuda() {
                ::svod_schedule::testing::setup_test_tracing();
                let $config = match $crate::PrepareConfig::for_cuda_if_available() {
                    Some(cfg) => cfg,
                    None => {
                        eprintln!("cuda codegen_tests variant: skipped (no CUDA device)");
                        return;
                    }
                };
                $body
            }
        }
        $crate::codegen_tests!($($rest)*);
    };

    // Proptest with config: #[proptest_config(...)] fn name(config, param in strategy) { body }
    (#[proptest_config($($pc:tt)*)] $(#[$meta:meta])* fn $name:ident($config:ident, $($param:ident in $strategy:expr),+ $(,)?) $body:block $($rest:tt)*) => {
        $crate::codegen_tests!(@proptest $name, $config, [$($param in $strategy),+], $body,
            ::proptest::test_runner::TestRunner::new($($pc)*), [$(#[$meta])*]);
        $crate::codegen_tests!($($rest)*);
    };

    // Proptest with default config: fn name(config, param in strategy) { body }
    ($(#[$meta:meta])* fn $name:ident($config:ident, $($param:ident in $strategy:expr),+ $(,)?) $body:block $($rest:tt)*) => {
        $crate::codegen_tests!(@proptest $name, $config, [$($param in $strategy),+], $body,
            ::proptest::test_runner::TestRunner::default(), [$(#[$meta])*]);
        $crate::codegen_tests!($($rest)*);
    };

    // Internal: proptest code generation (uses TestRunner API directly)
    (@proptest $name:ident, $config:ident, [$($param:ident in $strategy:expr),+], $body:block, $runner:expr, [$(#[$meta:meta])*]) => {
        mod $name {
            #[allow(unused_imports)]
            use super::*;

            #[test]
            #[allow(unused_parens)]
            $(#[$meta])*
            fn clang() {
                ::svod_schedule::testing::setup_test_tracing();
                let mut runner = $runner;
                runner.run(&($($strategy),+), |($($param),+)| {
                    let $config = $crate::PrepareConfig::for_cpu_backend($crate::CpuBackend::Clang);
                    $body
                    Ok(())
                }).unwrap();
            }

            #[test]
            #[allow(unused_parens)]
            $(#[$meta])*
            fn llvm() {
                ::svod_schedule::testing::setup_test_tracing();
                let mut runner = $runner;
                runner.run(&($($strategy),+), |($($param),+)| {
                    let $config = $crate::PrepareConfig::for_cpu_backend($crate::CpuBackend::Llvm);
                    $body
                    Ok(())
                }).unwrap();
            }

            #[test]
            #[allow(unused_parens)]
            $(#[$meta])*
            fn amd() {
                ::svod_schedule::testing::setup_test_tracing();
                let amd_cfg = match $crate::PrepareConfig::for_amd_if_available() {
                    Some(cfg) => cfg,
                    None => {
                        eprintln!("amd codegen_tests variant: skipped (no supported AMD GPU)");
                        return;
                    }
                };
                let mut runner = $runner;
                runner.run(&($($strategy),+), |($($param),+)| {
                    let $config = amd_cfg.clone();
                    $body
                    Ok(())
                }).unwrap();
            }

            #[test]
            #[allow(unused_parens)]
            $(#[$meta])*
            fn metal() {
                ::svod_schedule::testing::setup_test_tracing();
                let metal_cfg = match $crate::PrepareConfig::for_metal_if_available() {
                    Some(cfg) => cfg,
                    None => {
                        eprintln!("metal codegen_tests variant: skipped (no Metal device)");
                        return;
                    }
                };
                let mut runner = $runner;
                runner.run(&($($strategy),+), |($($param),+)| {
                    let $config = metal_cfg.clone();
                    $body
                    Ok(())
                }).unwrap();
            }

            #[test]
            #[allow(unused_parens)]
            $(#[$meta])*
            fn cuda() {
                ::svod_schedule::testing::setup_test_tracing();
                let cuda_cfg = match $crate::PrepareConfig::for_cuda_if_available() {
                    Some(cfg) => cfg,
                    None => {
                        eprintln!("cuda codegen_tests variant: skipped (no CUDA device)");
                        return;
                    }
                };
                let mut runner = $runner;
                runner.run(&($($strategy),+), |($($param),+)| {
                    let $config = cuda_cfg.clone();
                    $body
                    Ok(())
                }).unwrap();
            }
        }
    };

    // Parameterized test (extra typed params — test_case attrs expected, no #[test])
    ($(#[$meta:meta])* fn $name:ident($config:ident, $($param:ident: $ty:ty),+ $(,)?) $body:block $($rest:tt)*) => {
        mod $name {
            mod clang {
                #[allow(unused_imports)]
                use super::super::*;
                use ::test_case::test_case;

                $(#[$meta])*
                fn $name($($param: $ty),+) {
                    ::svod_schedule::testing::setup_test_tracing();
                    let $config = $crate::PrepareConfig::for_cpu_backend($crate::CpuBackend::Clang);
                    $body
                }
            }
            mod llvm {
                #[allow(unused_imports)]
                use super::super::*;
                use ::test_case::test_case;

                $(#[$meta])*
                fn $name($($param: $ty),+) {
                    ::svod_schedule::testing::setup_test_tracing();
                    let $config = $crate::PrepareConfig::for_cpu_backend($crate::CpuBackend::Llvm);
                    $body
                }
            }
            mod amd {
                #[allow(unused_imports)]
                use super::super::*;
                use ::test_case::test_case;

                $(#[$meta])*
                fn $name($($param: $ty),+) {
                    ::svod_schedule::testing::setup_test_tracing();
                    let $config = match $crate::PrepareConfig::for_amd_if_available() {
                        Some(cfg) => cfg,
                        None => {
                            eprintln!("amd codegen_tests variant: skipped (no supported AMD GPU)");
                            return;
                        }
                    };
                    $body
                }
            }
            mod metal {
                #[allow(unused_imports)]
                use super::super::*;
                use ::test_case::test_case;

                $(#[$meta])*
                fn $name($($param: $ty),+) {
                    ::svod_schedule::testing::setup_test_tracing();
                    let $config = match $crate::PrepareConfig::for_metal_if_available() {
                        Some(cfg) => cfg,
                        None => {
                            eprintln!("metal codegen_tests variant: skipped (no Metal device)");
                            return;
                        }
                    };
                    $body
                }
            }
            mod cuda {
                #[allow(unused_imports)]
                use super::super::*;
                use ::test_case::test_case;

                $(#[$meta])*
                fn $name($($param: $ty),+) {
                    ::svod_schedule::testing::setup_test_tracing();
                    let $config = match $crate::PrepareConfig::for_cuda_if_available() {
                        Some(cfg) => cfg,
                        None => {
                            eprintln!("cuda codegen_tests variant: skipped (no CUDA device)");
                            return;
                        }
                    };
                    $body
                }
            }
        }
        $crate::codegen_tests!($($rest)*);
    };
}

#[cfg(test)]
#[path = "test/unit/config.rs"]
mod tests;
