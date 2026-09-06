//! Backend renderer capabilities and tensor core configurations.
//!
//! This module defines the interface between the optimizer and backend code generators.
//! It describes what optimizations a backend supports (local memory, threading, etc.)
//! and provides tensor core configurations for hardware-accelerated matrix multiplication.

use smallvec::SmallVec;
use svod_dtype::{AmdArch, CudaArch, DType, MetalFamily, ScalarDType};
use svod_ir::{RendererDevice, RendererOps, TypedPatternMatcher};

/// Tensor core optimization operation.
///
/// Represents a single transformation step when applying tensor cores.
/// Each operation splits a dimension and assigns it to a new axis type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TcOpt {
    /// Upcast (vectorize) the specified dimension (0=N, 1=M, 2=K).
    Upcast(usize),
    /// Move the specified dimension to local memory (0=N, 1=M, 2=K).
    Local(usize),
}

impl TcOpt {
    /// Get the dimension index (0=N, 1=M, 2=K).
    pub const fn dim(&self) -> usize {
        match self {
            Self::Upcast(dim) | Self::Local(dim) => *dim,
        }
    }

    /// Returns true if this is an upcast operation.
    pub const fn is_upcast(&self) -> bool {
        matches!(self, Self::Upcast(_))
    }

    /// Returns true if this is a local operation.
    pub const fn is_local(&self) -> bool {
        matches!(self, Self::Local(_))
    }
}

impl std::fmt::Display for TcOpt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Upcast(dim) => write!(f, "u{}", dim),
            Self::Local(dim) => write!(f, "l{}", dim),
        }
    }
}

/// Swizzle axis specifier for tensor core data layout transformations.
///
/// Describes axis references in swizzle patterns that remap data layouts
/// for optimal tensor core memory access. Unlike TcOpt (operations),
/// SwizzleAxis describes axis identities in the remapping pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SwizzleAxis {
    /// Upcast axis with index (0, 1, 2, ...).
    Upcast(usize),
    /// Local axis with index (0, 1, 2, ...).
    Local(usize),
    /// Reduce axis with index (0, 1, 2, ...).
    Reduce(usize),
}

impl std::fmt::Display for SwizzleAxis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Upcast(idx) => write!(f, "u{}", idx),
            Self::Local(idx) => write!(f, "l{}", idx),
            Self::Reduce(idx) => write!(f, "r{}", idx),
        }
    }
}

/// Backend renderer capabilities.
///
/// Describes what features and optimizations a particular backend supports.
/// Used by the optimizer to determine valid transformations and enforce device limits.
#[derive(Clone)]
pub struct Renderer {
    /// Backend device identifier.
    pub device: RendererDevice,

    /// Exact target within the broad renderer family, such as `gfx1151`.
    target: Option<String>,

    /// Whether the backend supports local/shared memory (GPU workgroups).
    pub has_local: bool,

    /// Whether the backend supports shared memory across threads in a workgroup.
    pub has_shared: bool,

    /// Whether the backend supports CPU-style threading (not GPU threads).
    pub has_threads: bool,

    /// Maximum shared memory size in bytes.
    ///
    /// Used to validate GROUP/GROUPTOP optimizations that allocate shared memory.
    /// Typical values: 48KB-96KB for modern GPUs.
    pub shared_max: usize,

    /// Maximum global work dimensions [x, y, z].
    ///
    /// Maximum size for each global thread dimension.
    /// Used to validate thread count in THREAD optimization.
    /// None if unlimited or not applicable.
    pub global_max: Option<Vec<usize>>,

    /// Maximum product of global and local size for each hardware axis.
    ///
    /// HIP exposes this separately from the global grid limit.
    pub global_prod_max: Option<Vec<usize>>,

    /// Maximum local work group size.
    ///
    /// Maximum number of threads in a workgroup (product of local dimensions).
    /// Typical values: 256-1024 for GPUs.
    pub local_max: Option<usize>,

    /// Maximum vectorization width (upcast limit).
    ///
    /// Maximum number of elements that can be processed as a vector.
    /// Typical values: 8-16 for SIMD, 4 for GPU float4.
    pub upcast_max: usize,

    /// Maximum number of buffers/arguments per kernel.
    ///
    /// Some backends have limits on kernel arguments.
    /// Metal: 31, WebGPU: 8, CUDA: typically unlimited.
    pub buffer_max: Option<usize>,

    /// Available tensor core configurations.
    ///
    /// Hardware-accelerated matrix multiplication units with specific size constraints.
    /// Empty if tensor cores not available.
    pub tensor_cores: Vec<TensorCore>,

    /// Whether the backend supports vector (float4-style) load/store.
    /// When false, the devectorize pass falls back to scalar fold widths and
    /// skips wide load/store generation.
    pub supports_float4: bool,

    /// Renderer-owned final rewrite rules (`Renderer.extra_matcher` in tinygrad).
    extra_matcher: Option<TypedPatternMatcher>,

    /// Target-specific decomposition rules derived from operations absent from
    /// the renderer's support table.
    decomposition_matcher: Option<TypedPatternMatcher>,

    /// Exact operation table reported by the selected code renderer.
    renderer_ops: Option<RendererOps>,

    /// Scalar storage and conversion formats accepted by this target. Matrix
    /// support is described by `tensor_cores`; ordinary ALU support may be
    /// narrower and is queried separately.
    supported_dtypes: std::collections::HashSet<ScalarDType>,

    /// Stable semantic identities for matcher closures, which cannot be hashed.
    decomposition_profile: &'static str,
    extra_profile: &'static str,
}

impl std::fmt::Debug for Renderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Renderer")
            .field("device", &self.device)
            .field("has_local", &self.has_local)
            .finish_non_exhaustive()
    }
}

impl Renderer {
    fn dtype_set(dtypes: &[ScalarDType]) -> std::collections::HashSet<ScalarDType> {
        dtypes.iter().copied().collect()
    }

    fn common_dtypes() -> std::collections::HashSet<ScalarDType> {
        use ScalarDType::*;
        Self::dtype_set(&[
            Bool, Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64, Float16, BFloat16, Float32, Float64,
        ])
    }

    fn fp8_dtypes() -> std::collections::HashSet<ScalarDType> {
        let mut ret = Self::common_dtypes();
        ret.extend([ScalarDType::FP8E4M3, ScalarDType::FP8E5M2]);
        ret
    }

    fn pre_bf16_dtypes() -> std::collections::HashSet<ScalarDType> {
        let mut ret = Self::common_dtypes();
        ret.remove(&ScalarDType::BFloat16);
        ret
    }

    /// MSL has no `double`.
    fn metal_dtypes() -> std::collections::HashSet<ScalarDType> {
        let mut dtypes = Self::common_dtypes();
        dtypes.remove(&ScalarDType::Float64);
        dtypes
    }

    fn webgpu_dtypes() -> std::collections::HashSet<ScalarDType> {
        use ScalarDType::*;
        Self::dtype_set(&[Bool, Int8, UInt8, Int16, UInt16, Int32, UInt32, Float32])
    }

    /// Create a CPU renderer configuration.
    pub fn cpu() -> Self {
        Self {
            device: RendererDevice::Cpu,
            target: None,
            has_local: false,
            has_shared: false,
            has_threads: true,
            shared_max: 0,
            global_max: Some(vec![super::config::thread_budget()]),
            global_prod_max: None,
            local_max: None,
            upcast_max: 16, // AVX512 can do 16-wide float
            buffer_max: None,
            tensor_cores: vec![],
            supports_float4: true,
            extra_matcher: None,
            decomposition_matcher: None,
            renderer_ops: None,
            supported_dtypes: Self::common_dtypes(),
            decomposition_profile: "none",
            extra_profile: "none",
        }
    }

    /// Tinygrad's abstract base renderer instantiated with a CPU target.
    ///
    /// This models reference compiler stages that intentionally use the base
    /// renderer rather than a concrete CPU backend. Runtime CPU renderers must
    /// continue to use [`Self::cpu`].
    pub fn tinygrad_base_cpu() -> Self {
        Self {
            device: RendererDevice::Cpu,
            target: None,
            has_local: true,
            has_shared: true,
            has_threads: false,
            shared_max: 32768,
            global_max: Some(vec![0x8fff_ffff; 3]),
            global_prod_max: None,
            local_max: Some(0x8fff_ffff),
            upcast_max: 16,
            buffer_max: None,
            tensor_cores: vec![],
            supports_float4: true,
            extra_matcher: None,
            decomposition_matcher: None,
            renderer_ops: Some(RendererOps::default()),
            supported_dtypes: Self::common_dtypes(),
            decomposition_profile: "none",
            extra_profile: "none",
        }
    }

    /// Create a CUDA GPU renderer configuration (SM80/Ampere by default).
    ///
    /// For specific architectures, use `cuda_sm75()`, `cuda_sm80()`, or `cuda_sm89()`.
    pub fn cuda() -> Self {
        Self::cuda_sm80(false) // Default to SM80 (A100) without TF32
    }

    /// Create a CUDA GPU renderer for SM75 (Turing - RTX 20xx, T4).
    pub fn cuda_sm75() -> Self {
        Self {
            device: RendererDevice::CudaSm75,
            target: None,
            has_local: true,
            has_shared: true,
            has_threads: false,
            shared_max: 49152,
            global_max: Some(vec![2147483647, 65535, 65535]),
            global_prod_max: None,
            local_max: Some(1024),
            upcast_max: 8,
            buffer_max: None,
            tensor_cores: TensorCore::sm75_tensor_cores(),
            supports_float4: true,
            extra_matcher: None,
            decomposition_matcher: None,
            renderer_ops: None,
            supported_dtypes: Self::pre_bf16_dtypes(),
            decomposition_profile: "none",
            extra_profile: "none",
        }
    }

    /// Create a CUDA GPU renderer for SM80 (Ampere - A100, RTX 30xx).
    pub fn cuda_sm80(allow_tf32: bool) -> Self {
        Self {
            device: RendererDevice::CudaSm80,
            target: None,
            has_local: true,
            has_shared: true,
            has_threads: false,
            shared_max: 49152,
            global_max: Some(vec![2147483647, 65535, 65535]),
            global_prod_max: None,
            local_max: Some(1024),
            upcast_max: 8,
            buffer_max: None,
            tensor_cores: TensorCore::sm80_tensor_cores(allow_tf32),
            supports_float4: true,
            extra_matcher: None,
            decomposition_matcher: None,
            renderer_ops: None,
            supported_dtypes: Self::common_dtypes(),
            decomposition_profile: "none",
            extra_profile: "none",
        }
    }

    /// Create a CUDA GPU renderer for SM89 (Ada - RTX 40xx, L4; the first
    /// capability with fp8 `mma.sync`).
    pub fn cuda_sm89(allow_tf32: bool) -> Self {
        Self {
            device: RendererDevice::CudaSm89,
            target: None,
            has_local: true,
            has_shared: true,
            has_threads: false,
            shared_max: 49152,
            global_max: Some(vec![2147483647, 65535, 65535]),
            global_prod_max: None,
            local_max: Some(1024),
            upcast_max: 8,
            buffer_max: None,
            tensor_cores: TensorCore::sm89_tensor_cores(allow_tf32),
            supports_float4: true,
            extra_matcher: None,
            decomposition_matcher: None,
            renderer_ops: None,
            supported_dtypes: Self::fp8_dtypes(),
            decomposition_profile: "none",
            extra_profile: "none",
        }
    }

    /// The Metal profile for a concrete GPU family: `simdgroup_matrix` tensor
    /// cores exist from Apple7 (M1) on, so Intel-Mac GPUs (`Mac2`) and older
    /// Apple GPUs run without them.
    pub fn for_metal_family(family: MetalFamily) -> Self {
        let mut renderer = Self::metal();
        if !family.has_simdgroup_matrix() {
            renderer.tensor_cores.clear();
        }
        renderer.target = Some(family.to_string());
        renderer
    }

    /// Create a Metal GPU renderer configuration (family-agnostic: assumes an
    /// Apple7+ GPU; see [`Self::for_metal_family`]).
    pub fn metal() -> Self {
        Self {
            device: RendererDevice::Metal,
            target: None,
            has_local: true,
            has_shared: true,
            has_threads: false,
            shared_max: 32768, // 32KB for Metal
            // Three grid axes: extra global axes are grouped into them (tinygrad's
            // `Renderer.global_max = (0x8FFFFFFF,) * 3` default, which Metal inherits).
            global_max: Some(vec![0x8FFF_FFFF; 3]),
            global_prod_max: None,
            local_max: Some(1024),
            upcast_max: 4,        // float4 for Metal
            buffer_max: Some(31), // Metal has 31 buffer argument limit
            tensor_cores: TensorCore::metal_tensor_cores(),
            supports_float4: true,
            extra_matcher: None,
            decomposition_matcher: None,
            renderer_ops: None,
            supported_dtypes: Self::metal_dtypes(),
            decomposition_profile: "none",
            extra_profile: "none",
        }
    }

    /// Create an AMD RDNA3 GPU renderer (RX 7000 series).
    pub fn amd_rdna3() -> Self {
        Self {
            device: RendererDevice::AmdRdna3,
            target: None,
            has_local: true,
            has_shared: true,
            has_threads: false,
            shared_max: 65536, // 64KB for RDNA3
            global_max: Some(vec![2147483647, 65535, 65535]),
            global_prod_max: Some(vec![u32::MAX as usize; 3]),
            local_max: Some(1024),
            upcast_max: 8,
            buffer_max: None,
            tensor_cores: TensorCore::rdna3_tensor_cores(),
            supports_float4: true,
            extra_matcher: None,
            decomposition_matcher: None,
            renderer_ops: None,
            supported_dtypes: Self::common_dtypes(),
            decomposition_profile: "none",
            extra_profile: "none",
        }
    }

    /// Create an AMD RDNA4 GPU renderer.
    pub fn amd_rdna4() -> Self {
        Self {
            device: RendererDevice::AmdRdna4,
            target: None,
            has_local: true,
            has_shared: true,
            has_threads: false,
            shared_max: 65536,
            global_max: Some(vec![2147483647, 65535, 65535]),
            global_prod_max: Some(vec![u32::MAX as usize; 3]),
            local_max: Some(1024),
            upcast_max: 8,
            buffer_max: None,
            tensor_cores: TensorCore::rdna4_tensor_cores(),
            supports_float4: true,
            extra_matcher: None,
            decomposition_matcher: None,
            renderer_ops: None,
            supported_dtypes: Self::common_dtypes(),
            decomposition_profile: "none",
            extra_profile: "none",
        }
    }

    /// Create an AMD CDNA3 GPU renderer.
    pub fn amd_cdna3() -> Self {
        Self {
            device: RendererDevice::AmdCdna3,
            target: None,
            has_local: true,
            has_shared: true,
            has_threads: false,
            shared_max: 65536, // 64KB for CDNA
            global_max: Some(vec![2147483647, 65535, 65535]),
            global_prod_max: Some(vec![u32::MAX as usize; 3]),
            local_max: Some(1024),
            upcast_max: 8,
            buffer_max: None,
            tensor_cores: TensorCore::cdna3_tensor_cores(),
            supports_float4: true,
            extra_matcher: None,
            decomposition_matcher: None,
            renderer_ops: None,
            // gfx942 has native OCP FP8 MFMA operands even though HIP does not
            // expose FP8 as a general-purpose storage/ALU type on this arch.
            supported_dtypes: Self::fp8_dtypes(),
            decomposition_profile: "none",
            extra_profile: "none",
        }
    }

    /// Create an AMD CDNA4 GPU renderer.
    pub fn amd_cdna4() -> Self {
        Self {
            device: RendererDevice::AmdCdna4,
            target: None,
            has_local: true,
            has_shared: true,
            has_threads: false,
            shared_max: 65536,
            global_max: Some(vec![2147483647, 65535, 65535]),
            global_prod_max: Some(vec![u32::MAX as usize; 3]),
            local_max: Some(1024),
            upcast_max: 8,
            buffer_max: None,
            tensor_cores: TensorCore::cdna4_tensor_cores(),
            supports_float4: true,
            extra_matcher: None,
            decomposition_matcher: None,
            renderer_ops: None,
            supported_dtypes: Self::fp8_dtypes(),
            decomposition_profile: "none",
            extra_profile: "none",
        }
    }

    /// Per-axis local work-group size limits `(x, y, z)`, or `None` when only
    /// the product cap ([`local_max`](Self::local_max)) applies.
    ///
    /// This is distinct from [`local_max`](Self::local_max), which caps the
    /// product of all local axes.
    pub fn local_max_axes(&self) -> Option<[usize; 3]> {
        match self.device {
            RendererDevice::CudaSm75 | RendererDevice::CudaSm80 | RendererDevice::CudaSm89 => Some([1024, 1024, 64]),
            RendererDevice::WebGpu => Some([256, 256, 64]),
            _ => None,
        }
    }

    /// Select the AMD optimizer profile matching a gfx arch. CDNA gfx942 maps
    /// to CDNA3 and gfx950 to CDNA4; RDNA3/RDNA4 families map to their profiles.
    pub fn for_amd_arch(arch: AmdArch) -> Self {
        let mut renderer = match arch {
            AmdArch::Gfx942 => Self::amd_cdna3(),
            AmdArch::Gfx950 => Self::amd_cdna4(),
            _ if arch.is_rdna4() => Self::amd_rdna4(),
            _ => Self::amd_rdna3(),
        };
        renderer.target = Some(arch.mcpu().to_string());
        renderer
    }

    /// Select the CUDA optimizer profile for a compute capability, tinygrad's
    /// `tc.get_cuda` plus int8: sm_80+ has the bf16 / tf32 shapes, `m16n8k16`
    /// and the s8 `m16n8k32`, sm_75 the f16 `m16n8k8` core only (its integer
    /// `mma.sync` shapes are `m8n8k16`, which the NVPTX renderer does not
    /// lower), and anything older runs without tensor cores. The sm_89 fp8
    /// profile is withheld from every capability until the NVPTX renderer
    /// lowers the `cvt.*.e4m3x2` conversions; its fp8 storage dtype would fail
    /// at render time today.
    pub fn for_cuda_arch(arch: CudaArch) -> Self {
        let sm = arch.sm();
        let mut renderer = if arch.has_bf16_mma() { Self::cuda_sm80(false) } else { Self::cuda_sm75() };
        if sm < 75 {
            renderer.tensor_cores.clear();
        }
        renderer.target = Some(arch.to_string());
        renderer
    }

    /// Create an Intel Xe GPU renderer.
    pub fn intel_xe() -> Self {
        Self {
            device: RendererDevice::IntelXe,
            target: None,
            has_local: true,
            has_shared: true,
            has_threads: false,
            shared_max: 65536, // 64KB for Xe
            global_max: Some(vec![2147483647, 65535, 65535]),
            global_prod_max: None,
            local_max: Some(512),
            upcast_max: 8,
            buffer_max: None,
            tensor_cores: TensorCore::intel_tensor_cores(),
            supports_float4: true,
            extra_matcher: None,
            decomposition_matcher: None,
            renderer_ops: None,
            supported_dtypes: Self::common_dtypes(),
            decomposition_profile: "none",
            extra_profile: "none",
        }
    }

    /// Create a WebGPU renderer configuration.
    pub fn webgpu() -> Self {
        Self {
            device: RendererDevice::WebGpu,
            target: None,
            has_local: true,
            has_shared: true,
            has_threads: false,
            shared_max: 16384, // 16KB typical for WebGPU
            global_max: Some(vec![65535, 65535, 65535]),
            global_prod_max: None,
            local_max: Some(256),
            upcast_max: 4,
            buffer_max: Some(8), // WebGPU has 8 buffer limit in some implementations
            tensor_cores: vec![],
            supports_float4: false,
            extra_matcher: None,
            decomposition_matcher: None,
            renderer_ops: None,
            supported_dtypes: Self::webgpu_dtypes(),
            decomposition_profile: "none",
            extra_profile: "none",
        }
    }

    /// Bind the concrete code renderer's operation table and target-local
    /// rewrites to this hardware optimization profile.
    pub fn with_codegen_renderer(mut self, renderer: &dyn svod_device::device::Renderer) -> Self {
        self.renderer_ops = Some(renderer.supported_ops());
        self.decomposition_matcher = renderer.decompositor();
        self.extra_matcher = renderer.extra_matcher();
        self.target = renderer.gpu_arch().map(svod_dtype::GpuArch::target_name);
        self.decomposition_profile = if self.decomposition_matcher.is_some() {
            match self.device {
                RendererDevice::AmdRdna3
                | RendererDevice::AmdRdna4
                | RendererDevice::AmdCdna3
                | RendererDevice::AmdCdna4 => "amd-decomposition-v1",
                RendererDevice::CudaSm75 | RendererDevice::CudaSm80 | RendererDevice::CudaSm89 => {
                    "nvptx-decomposition-v1"
                }
                _ => "backend-decomposition-v1",
            }
        } else {
            "none"
        };
        self.extra_profile = if self.extra_matcher.is_some() {
            match self.device {
                RendererDevice::Cpu => "llvm-cpu-extra-v1",
                RendererDevice::AmdRdna3
                | RendererDevice::AmdRdna4
                | RendererDevice::AmdCdna3
                | RendererDevice::AmdCdna4 => "llvm-amd-fp8-extra-v1",
                RendererDevice::CudaSm75 | RendererDevice::CudaSm80 | RendererDevice::CudaSm89 => "llvm-nvptx-extra-v1",
                _ => "backend-extra-v1",
            }
        } else {
            "none"
        };
        self
    }

    /// Bind explicit renderer capabilities. This is primarily useful for
    /// renderer unit tests and non-device embedding layers.
    pub fn with_rewrite_capabilities(
        mut self,
        supported_ops: RendererOps,
        decomposition_matcher: Option<TypedPatternMatcher>,
        extra_matcher: Option<TypedPatternMatcher>,
    ) -> Self {
        self.renderer_ops = Some(supported_ops);
        self.decomposition_matcher = decomposition_matcher;
        self.extra_matcher = extra_matcher;
        self.decomposition_profile =
            if self.decomposition_matcher.is_some() { "explicit-decomposition-v1" } else { "none" };
        self.extra_profile = if self.extra_matcher.is_some() { "explicit-extra-v1" } else { "none" };
        self
    }

    /// Deterministic identity for every optimizer-visible renderer behavior.
    pub fn cache_fingerprint(&self) -> u64 {
        use std::hash::{Hash, Hasher};

        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        1u32.hash(&mut hasher);
        self.device.hash(&mut hasher);
        self.target.hash(&mut hasher);
        self.has_local.hash(&mut hasher);
        self.has_shared.hash(&mut hasher);
        self.has_threads.hash(&mut hasher);
        self.shared_max.hash(&mut hasher);
        self.global_max.hash(&mut hasher);
        self.global_prod_max.hash(&mut hasher);
        self.local_max.hash(&mut hasher);
        self.local_max_axes().hash(&mut hasher);
        self.upcast_max.hash(&mut hasher);
        self.buffer_max.hash(&mut hasher);
        self.supports_float4.hash(&mut hasher);
        self.decomposition_profile.hash(&mut hasher);
        self.extra_profile.hash(&mut hasher);

        let mut dtypes = self.supported_dtypes.iter().copied().collect::<Vec<_>>();
        dtypes.sort();
        dtypes.hash(&mut hasher);
        self.tensor_cores.hash(&mut hasher);

        if let Some(ops) = &self.renderer_ops {
            let mut unary = ops.unary.iter().map(AsRef::<str>::as_ref).collect::<Vec<_>>();
            let mut binary = ops.binary.iter().map(AsRef::<str>::as_ref).collect::<Vec<_>>();
            let mut ternary = ops.ternary.iter().map(|op| format!("{op:?}")).collect::<Vec<_>>();
            unary.sort_unstable();
            binary.sort_unstable();
            ternary.sort_unstable();
            unary.hash(&mut hasher);
            binary.hash(&mut hasher);
            ternary.hash(&mut hasher);
        }
        hasher.finish()
    }

    pub(crate) fn supported_ops(&self) -> Option<&RendererOps> {
        self.renderer_ops.as_ref()
    }

    pub(crate) fn supports_dtype(&self, dtype: ScalarDType) -> bool {
        self.supported_dtypes.contains(&dtype)
    }

    pub fn supports_storage_dtype(&self, dtype: ScalarDType) -> bool {
        self.supports_dtype(dtype)
    }

    pub fn supports_conversion_dtype(&self, dtype: ScalarDType) -> bool {
        self.supports_dtype(dtype)
    }

    pub fn supports_alu_dtype(&self, dtype: ScalarDType) -> bool {
        self.supports_dtype(dtype)
            && !(matches!(self.device, RendererDevice::AmdCdna3 | RendererDevice::AmdCdna4)
                && matches!(dtype, ScalarDType::FP8E4M3 | ScalarDType::FP8E5M2))
    }

    pub fn supports_matrix_dtype(&self, dtype: ScalarDType) -> bool {
        self.tensor_cores.iter().any(|tensor_core| tensor_core.dtype_in.base() == dtype)
    }

    pub(crate) fn supported_dtypes(&self) -> std::collections::HashSet<ScalarDType> {
        self.supported_dtypes.clone()
    }

    pub(crate) fn decomposition_matcher(&self) -> Option<&TypedPatternMatcher> {
        self.decomposition_matcher.as_ref()
    }

    pub(crate) fn extra_matcher(&self) -> Option<&TypedPatternMatcher> {
        self.extra_matcher.as_ref()
    }
}

/// Tensor core configuration for hardware-accelerated matrix multiplication.
///
/// Describes a specific matrix multiplication unit with fixed dimensions and data types.
/// Based on NVIDIA's WMMA (Warp Matrix Multiply-Accumulate) API and similar accelerators.
///
/// # Matrix Dimensions
///
/// Tensor cores perform: `C[M,N] += A[M,K] × B[K,N]`
/// - `dims.0` (N): Number of output columns
/// - `dims.1` (M): Number of output rows
/// - `dims.2` (K): Reduction dimension size
///
/// # Example
///
/// NVIDIA Tensor Core 16x16x16:
/// - Processes 16×16 output tile
/// - Accumulates across 16 K elements
/// - Uses 32 threads (warp size)
/// - Each thread handles multiple elements via opts
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorCore {
    /// Matrix dimensions (N, M, K).
    pub dims: (usize, usize, usize),

    /// Number of threads required (typically warp size: 32 for CUDA, 64 for AMD).
    pub threads: usize,

    /// Elements per thread in each dimension (N, M, K).
    ///
    /// Describes how the matrix is distributed across threads.
    /// Example: (2, 2, 4) means each thread handles 2×2 output elements
    /// and processes 4 K elements.
    pub elements_per_thread: (usize, usize, usize),

    /// Input matrix data type (A and B matrices).
    pub dtype_in: DType,

    /// Output/accumulator data type (C matrix).
    pub dtype_out: DType,

    /// Optimization sequence for tensor core application.
    ///
    /// A sequence of operations to transform ranges. Each operation splits
    /// a dimension (N, M, or K) and assigns it to a new axis type.
    ///
    /// Example: `[Upcast(0), Local(0), Local(0), Local(1), Local(1), Local(1), Upcast(1)]`
    /// - Upcast N once
    /// - Local split N twice
    /// - Local split M three times
    /// - Upcast M once
    ///
    /// Uses SmallVec to avoid heap allocation for typical tensor cores (≤8 ops).
    pub opts: SmallVec<[TcOpt; 8]>,

    /// Swizzle patterns for input permutation.
    ///
    /// Describes how to permute input matrices to match hardware layout.
    /// Format: ((A_local, A_upcast, A_reduce), (B_local, B_upcast, B_reduce))
    ///
    /// Each tuple contains axis references that describe the permutation pattern
    /// for optimal memory access. The first tuple is for matrix A, second for B.
    ///
    /// Uses SmallVec to avoid heap allocation for typical swizzles (≤8 axes per vec).
    #[allow(clippy::type_complexity)]
    pub swizzle: (
        (SmallVec<[SwizzleAxis; 8]>, SmallVec<[SwizzleAxis; 8]>, SmallVec<[SwizzleAxis; 8]>),
        (SmallVec<[SwizzleAxis; 8]>, SmallVec<[SwizzleAxis; 8]>, SmallVec<[SwizzleAxis; 8]>),
    ),
}

// ============================================================================
// TENSOR CORE CONFIGURATION (Static Const Data)
// ============================================================================

/// Static tensor core configuration for const definitions.
///
/// Uses static slices instead of SmallVec for const-compatibility.
/// Use `build()` to convert to runtime `TensorCore`.
pub struct TcConfig {
    dims: (usize, usize, usize),
    threads: usize,
    ept: (usize, usize, usize),
    opts: &'static [TcOpt],
    swizzle_a: (&'static [SwizzleAxis], &'static [SwizzleAxis], &'static [SwizzleAxis]),
    swizzle_b: (&'static [SwizzleAxis], &'static [SwizzleAxis], &'static [SwizzleAxis]),
}

impl TcConfig {
    /// Build a TensorCore from static config with specified dtypes.
    pub fn build(&self, dtype_in: DType, dtype_out: DType) -> TensorCore {
        TensorCore {
            dims: self.dims,
            threads: self.threads,
            elements_per_thread: self.ept,
            dtype_in,
            dtype_out,
            opts: self.opts.iter().copied().collect(),
            swizzle: (
                (
                    self.swizzle_a.0.iter().copied().collect(),
                    self.swizzle_a.1.iter().copied().collect(),
                    self.swizzle_a.2.iter().copied().collect(),
                ),
                (
                    self.swizzle_b.0.iter().copied().collect(),
                    self.swizzle_b.1.iter().copied().collect(),
                    self.swizzle_b.2.iter().copied().collect(),
                ),
            ),
        }
    }
}

// Aliases for brevity in const definitions
use SwizzleAxis::{Local as SL, Reduce as R, Upcast as SU};
use TcOpt::{Local as L, Upcast as U};

// NVIDIA CUDA Tensor Cores
pub const CUDA_81616: TcConfig = TcConfig {
    dims: (8, 16, 16),
    threads: 32,
    ept: (8, 4, 4),
    opts: &[U(0), L(0), L(0), L(1), L(1), L(1), U(1)],
    swizzle_a: (&[R(1), R(2), SL(2), SL(3), SL(4)], &[SU(1), R(3)], &[SL(0), SL(1), SU(0), R(0)]),
    swizzle_b: (&[R(1), R(2), SU(0), SL(0), SL(1)], &[R(0), R(3)], &[SL(2), SL(3), SL(4), SU(1)]),
};

pub const CUDA_81632: TcConfig = TcConfig {
    dims: (8, 16, 32),
    threads: 32,
    ept: (16, 8, 4),
    opts: &[U(0), L(0), L(0), L(1), L(1), L(1), U(1)],
    swizzle_a: (&[R(2), R(3), SL(2), SL(3), SL(4)], &[SU(1), R(4)], &[SL(0), SL(1), SU(0), R(0), R(1)]),
    swizzle_b: (&[R(2), R(3), SU(0), SL(0), SL(1)], &[R(1), R(4)], &[SL(2), SL(3), SL(4), SU(1), R(0)]),
};

pub const CUDA_8168: TcConfig = TcConfig {
    dims: (8, 16, 8),
    threads: 32,
    ept: (4, 2, 4),
    opts: &[U(0), L(0), L(0), L(1), L(1), L(1), U(1)],
    swizzle_a: (&[R(1), R(2), SL(2), SL(3), SL(4)], &[R(0), SU(1)], &[SL(0), SL(1), SU(0)]),
    swizzle_b: (&[R(1), R(2), SU(0), SL(0), SL(1)], &[SU(1), R(0)], &[SL(2), SL(3), SL(4)]),
};

pub const CUDA_8168_TF32: TcConfig = TcConfig {
    dims: (8, 16, 8),
    threads: 32,
    ept: (4, 2, 4),
    opts: &[U(0), L(0), L(0), L(1), L(1), L(1), U(1)],
    swizzle_a: (&[R(0), R(1), SL(2), SL(3), SL(4)], &[SU(1), R(2)], &[SL(0), SL(1), SU(0)]),
    swizzle_b: (&[R(0), R(1), SU(0), SL(0), SL(1)], &[SU(1), R(2)], &[SL(2), SL(3), SL(4)]),
};

// AMD Tensor Cores
pub const AMD_RDNA3: TcConfig = TcConfig {
    dims: (16, 16, 16),
    threads: 32,
    ept: (16, 16, 8),
    opts: &[L(0), L(0), L(0), L(0), L(1), U(1), U(1), U(1)],
    swizzle_a: (&[SL(4), SU(0), SU(1), SU(2), SL(0)], &[R(1), R(2), R(3)], &[SL(1), SL(2), SL(3), R(0)]),
    swizzle_b: (&[SL(0), SL(1), SL(2), SL(3), SL(4)], &[R(1), R(2), R(3)], &[SU(0), SU(1), SU(2), R(0)]),
};

pub const AMD_RDNA4: TcConfig = TcConfig {
    dims: (16, 16, 16),
    threads: 32,
    ept: (8, 8, 8),
    opts: &[L(0), L(0), L(0), L(0), U(1), U(1), U(1), L(1)],
    swizzle_a: (&[SU(0), SU(1), SU(2), SL(4), R(2)], &[R(0), R(1), R(3)], &[SL(0), SL(1), SL(2), SL(3)]),
    swizzle_b: (&[SL(0), SL(1), SL(2), SL(3), R(2)], &[R(0), R(1), R(3)], &[SL(4), SU(0), SU(1), SU(2)]),
};

pub const AMD_CDNA_161616: TcConfig = TcConfig {
    dims: (16, 16, 16),
    threads: 64,
    ept: (4, 4, 4),
    opts: &[L(0), L(0), L(0), L(0), U(1), U(1), L(1), L(1)],
    swizzle_a: (&[SU(0), SU(1), SL(4), SL(5), R(2), R(3)], &[R(0), R(1)], &[SL(0), SL(1), SL(2), SL(3)]),
    swizzle_b: (&[SL(0), SL(1), SL(2), SL(3), R(2), R(3)], &[R(0), R(1)], &[SL(4), SL(5), SU(0), SU(1)]),
};

pub const AMD_CDNA_161632: TcConfig = TcConfig {
    dims: (16, 16, 32),
    threads: 64,
    ept: (8, 8, 4),
    opts: &[L(0), L(0), L(0), L(0), U(1), U(1), L(1), L(1)],
    swizzle_a: (&[SU(0), SU(1), SL(4), SL(5), R(3), R(4)], &[R(0), R(1)], &[SL(0), SL(1), SL(2), SL(3), R(2)]),
    swizzle_b: (&[SL(0), SL(1), SL(2), SL(3), R(3), R(4)], &[R(0), R(1)], &[SL(4), SL(5), SU(0), SU(1), R(2)]),
};

pub const AMD_CDNA_1616128: TcConfig = TcConfig {
    dims: (16, 16, 128),
    threads: 64,
    ept: (32, 32, 4),
    opts: &[L(0), L(0), L(0), L(0), U(1), U(1), L(1), L(1)],
    swizzle_a: (
        &[SU(0), SU(1), SL(4), SL(5), R(5), R(6)],
        &[R(0), R(1)],
        &[SL(0), SL(1), SL(2), SL(3), R(2), R(3), R(4)],
    ),
    swizzle_b: (
        &[SL(0), SL(1), SL(2), SL(3), R(5), R(6)],
        &[R(0), R(1)],
        &[SL(4), SL(5), SU(0), SU(1), R(2), R(3), R(4)],
    ),
};

// Apple Metal Tensor Cores
pub const METAL_888: TcConfig = TcConfig {
    dims: (8, 8, 8),
    threads: 32,
    ept: (2, 2, 2),
    opts: &[U(0), L(0), L(1), L(1), L(0), L(1)],
    swizzle_a: (&[R(1), SL(1), SL(2), R(2), SL(4)], &[R(0)], &[SU(0), SL(0), SL(3)]),
    swizzle_b: (&[SL(0), R(0), R(1), SL(3), R(2)], &[SU(0)], &[SL(1), SL(2), SL(4)]),
};

// Intel Xe Tensor Cores
pub const INTEL_XE_8816: TcConfig = TcConfig {
    dims: (8, 8, 16),
    threads: 8,
    ept: (16, 16, 8),
    opts: &[L(0), L(0), L(0), U(1), U(1), U(1)],
    swizzle_a: (&[R(1), R(2), R(3)], &[SU(0), SU(1), SU(2)], &[SL(0), SL(1), SL(2), R(0)]),
    swizzle_b: (&[SL(0), SL(1), SL(2)], &[R(1), R(2), R(3)], &[SU(0), SU(1), SU(2), R(0)]),
};

impl TensorCore {
    // ===== Helper Methods =====

    /// Get the axes for reduction unrolling.
    ///
    /// Returns pairs of (dimension_index, unroll_amount) for the K dimension.
    /// Used during TC application to unroll the reduction dimension.
    pub fn get_reduce_axes(&self) -> Vec<(usize, usize)> {
        (0..(self.dims.2 as f64).log2().floor() as usize).map(|i| (i, 2)).collect()
    }

    /// Get the upcast axes configuration for WMMA construction.
    ///
    /// Returns tensor-core expansion axis configuration.
    /// Format: (A_axes, B_axes, output_axes)
    pub fn upcast_axes(&self) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        // This is simplified - actual implementation depends on opts sequence
        // For 16x16x16 WMMA: each has specific upcast patterns
        (vec![0, 1], vec![0, 1], vec![0, 1])
    }

    // ===== Hardware-Specific Collections =====

    /// Get all tensor cores for NVIDIA SM75 architecture (Turing).
    pub fn sm75_tensor_cores() -> Vec<TensorCore> {
        vec![CUDA_8168.build(DType::Float16, DType::Float32), CUDA_8168.build(DType::Float16, DType::Float16)]
    }

    /// Get all tensor cores for NVIDIA SM80 architecture (Ampere). The
    /// `m16n8k32` int8 core shares the fp8 fragment layout (both are one byte
    /// per element), so it reuses `CUDA_81632`; it mirrors the RDNA3
    /// `int8 -> int32` core that quantized linears already engage on AMD.
    pub fn sm80_tensor_cores(allow_tf32: bool) -> Vec<TensorCore> {
        let mut tcs = vec![
            CUDA_81616.build(DType::Float16, DType::Float32),
            CUDA_81616.build(DType::BFloat16, DType::Float32),
            CUDA_81616.build(DType::Float16, DType::Float16),
            CUDA_8168.build(DType::Float16, DType::Float32),
            CUDA_8168.build(DType::Float16, DType::Float16),
            CUDA_81632.build(DType::Int8, DType::Int32),
        ];
        if allow_tf32 {
            tcs.push(CUDA_8168_TF32.build(DType::Float32, DType::Float32));
        }
        tcs
    }

    /// Get all tensor cores for NVIDIA SM89 architecture (Hopper).
    pub fn sm89_tensor_cores(allow_tf32: bool) -> Vec<TensorCore> {
        let mut tcs = Self::sm80_tensor_cores(allow_tf32);
        tcs.push(CUDA_81632.build(DType::FP8E4M3, DType::Float32));
        tcs.push(CUDA_81632.build(DType::FP8E5M2, DType::Float32));
        tcs
    }

    /// Get all tensor cores for AMD RDNA3 architecture (RX 7000 series).
    pub fn rdna3_tensor_cores() -> Vec<TensorCore> {
        vec![
            AMD_RDNA3.build(DType::Float16, DType::Float32),
            AMD_RDNA3.build(DType::Float16, DType::Float16),
            AMD_RDNA3.build(DType::BFloat16, DType::Float32),
            AMD_RDNA3.build(DType::Int8, DType::Int32),
        ]
    }

    /// Get all tensor cores for AMD RDNA4 architecture.
    pub fn rdna4_tensor_cores() -> Vec<TensorCore> {
        vec![
            AMD_RDNA4.build(DType::Float16, DType::Float32),
            AMD_RDNA4.build(DType::Float16, DType::Float16),
            AMD_RDNA4.build(DType::BFloat16, DType::Float32),
            AMD_RDNA4.build(DType::BFloat16, DType::BFloat16),
        ]
    }

    /// Get all tensor cores for AMD CDNA3 architecture.
    pub fn cdna3_tensor_cores() -> Vec<TensorCore> {
        vec![
            AMD_CDNA_161632.build(DType::FP8E5M2, DType::Float32),
            AMD_CDNA_161632.build(DType::FP8E4M3, DType::Float32),
            AMD_CDNA_161616.build(DType::Float16, DType::Float32),
            AMD_CDNA_161616.build(DType::BFloat16, DType::Float32),
        ]
    }

    /// Get all tensor cores for AMD CDNA4 architecture.
    pub fn cdna4_tensor_cores() -> Vec<TensorCore> {
        vec![
            AMD_CDNA_1616128.build(DType::FP8E5M2, DType::Float32),
            AMD_CDNA_1616128.build(DType::FP8E4M3, DType::Float32),
            AMD_CDNA_161632.build(DType::FP8E5M2, DType::Float32),
            AMD_CDNA_161632.build(DType::FP8E4M3, DType::Float32),
            AMD_CDNA_161632.build(DType::Float16, DType::Float32),
            AMD_CDNA_161632.build(DType::BFloat16, DType::Float32),
            AMD_CDNA_161616.build(DType::Float16, DType::Float32),
            AMD_CDNA_161616.build(DType::BFloat16, DType::Float32),
        ]
    }

    /// Get all tensor cores for Apple Metal (M1/M2/M3).
    pub fn metal_tensor_cores() -> Vec<TensorCore> {
        vec![
            METAL_888.build(DType::Float32, DType::Float32),
            METAL_888.build(DType::Float16, DType::Float32),
            METAL_888.build(DType::Float16, DType::Float16),
            METAL_888.build(DType::BFloat16, DType::Float32),
            METAL_888.build(DType::BFloat16, DType::BFloat16),
        ]
    }

    /// Get all tensor cores for Intel Xe architecture.
    pub fn intel_tensor_cores() -> Vec<TensorCore> {
        vec![INTEL_XE_8816.build(DType::Float16, DType::Float32)]
    }
}

#[cfg(test)]
#[path = "../test/unit/optimizer/renderer_internal.rs"]
mod tests;
