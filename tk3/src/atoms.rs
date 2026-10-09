//! Hardware instructions as atoms that carry their own layouts. A kernel never
//! names one; the lowering picks atoms from the [`Target`] and the inference
//! propagates their layouts to the values around them.

use svod_dtype::{AmdArch, CudaArch, DType, GpuArch, ScalarDType};
use svod_ir::{AxisId, RendererDevice, WmmaMetadata, WmmaUpcastAxes};
use svod_schedule::optimizer::{Renderer, TensorCore};

use crate::layout::{self as frag, Dim, Layout};
use crate::schedule::Prefetch;

/// One matrix-core instruction: `C[m, n] += A[m, k] · B[k, n]`, with the
/// register/lane layout of every operand over one wave.
#[derive(Clone, Debug, PartialEq)]
pub struct MmaAtom {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    /// `Row = m, Col = k` over `(Reg, Lane)`.
    pub a: Layout,
    /// `Row = k, Col = n`.
    pub b: Layout,
    /// `Row = m, Col = n`.
    pub c: Layout,
    pub dtype_in: ScalarDType,
    pub dtype_out: ScalarDType,
    pub meta: WmmaMetadata,
}

impl MmaAtom {
    pub fn regs(&self, operand: Dim) -> u32 {
        match operand {
            Dim::Row => self.a.in_size(Dim::Reg),
            Dim::Col => self.b.in_size(Dim::Reg),
            _ => self.c.in_size(Dim::Reg),
        }
    }
}

/// What a target offers the lowering.
#[derive(Clone, Debug)]
pub struct Target {
    pub arch: GpuArch,
    pub wave: u32,
    pub mma: Vec<MmaAtom>,
    /// Asynchronous global→shared copies (`cp.async`).
    pub cp_async: bool,
    /// Warp-collective 8×8 b16 fragment loads from shared memory.
    pub ldmatrix: bool,
    /// Shared memory a block may use without special launch attributes.
    pub smem_bytes: usize,
    /// Streaming multiprocessors (compute units), when the device reports them.
    pub sms: Option<u32>,
}

impl Target {
    pub fn for_arch(arch: GpuArch) -> Self {
        let wave = match arch {
            GpuArch::Amd(a) => a.wave_size(),
            GpuArch::Cuda(c) => c.wave_size(),
            GpuArch::Metal(_) => 32,
        };
        let (cp_async, ldmatrix, smem_bytes) = match arch {
            GpuArch::Cuda(c) => (c.major >= 8, (c.major, c.minor) >= (7, 5), c.max_shared_per_block_optin()),
            GpuArch::Amd(_) => (false, false, 64 << 10),
            GpuArch::Metal(_) => (false, false, 32 << 10),
        };
        Self { arch, wave, mma: mma_atoms(arch), cp_async, ldmatrix, smem_bytes, sms: None }
    }

    /// The target behind a device, when the backend reports its architecture.
    pub fn for_device(spec: &svod_dtype::DeviceSpec) -> Option<Self> {
        use svod_device::registry as reg;
        use svod_dtype::DeviceSpec;
        let mut limits = None;
        let arch = match spec {
            DeviceSpec::Cuda { device_id } => {
                limits = reg::resolve_cuda_limits(*device_id).ok();
                GpuArch::Cuda(reg::resolve_cuda_arch(*device_id).ok()?)
            }
            DeviceSpec::Amd { device_id } => {
                let arch = reg::resolve_amd_arch_from_topology(*device_id).ok()?;
                let node = svod_device::amd::topology::enumerate().into_iter().nth(*device_id);
                let mut target = Self::for_arch(GpuArch::Amd(arch));
                if let Some(node) = node {
                    if node.lds_size_in_kb > 0 {
                        target.smem_bytes = node.lds_size_in_kb as usize * 1024;
                    }
                    target.sms = (node.simd_per_cu > 0).then(|| node.simd_count / node.simd_per_cu);
                }
                return Some(target);
            }
            DeviceSpec::Metal { device_id } => GpuArch::Metal(reg::resolve_metal_family(*device_id).ok()?),
            DeviceSpec::Cpu | DeviceSpec::WebGpu | DeviceSpec::Disk { .. } => return None,
        };
        let mut target = Self::for_arch(arch);
        if let Some(limits) = limits {
            target.smem_bytes = limits.shared_per_block_optin as usize;
            target.sms = Some(limits.sm_count);
        }
        Some(target)
    }

    /// How a pipeline fills shared memory: `cp.async` where the target has
    /// it, else through registers (RDNA3/RDNA4 have no global→LDS copy).
    pub fn prefetch(&self) -> Prefetch {
        if self.cp_async { Prefetch::CpAsync } else { Prefetch::RegisterStaged }
    }

    /// Pipeline stages the target's prefetch can use: register staging
    /// runs one step ahead over two slots.
    pub fn stages(&self, stages: usize) -> usize {
        match self.prefetch() {
            Prefetch::CpAsync => stages,
            Prefetch::RegisterStaged => 2,
        }
    }

    /// The matrix core for `dtype_in → dtype_out`, if the target has one.
    pub fn mma(&self, dtype_in: ScalarDType, dtype_out: ScalarDType) -> Option<&MmaAtom> {
        self.mma.iter().find(|a| a.dtype_in == dtype_in && a.dtype_out == dtype_out)
    }
}

/// The scheduler's tensor-core descriptor as the IR metadata a hand-built
/// `Wmma` carries (mirrors tk1 `wmma_from_tc`: `log2(elements_per_thread)`
/// size-2 upcast axes per operand, no reduce axes).
fn metadata(tc: &TensorCore, device: RendererDevice) -> WmmaMetadata {
    let axes = |ept: usize| -> Vec<(AxisId, usize)> {
        (0..ept.trailing_zeros() as usize).map(|i| (AxisId::Renumbered(4 - i), 2)).collect()
    };
    WmmaMetadata {
        name: tc.wmma_name(),
        dims: tc.dims,
        dtype_in: tc.dtype_in.clone(),
        dtype_out: tc.dtype_out.clone(),
        device,
        threads: tc.threads,
        upcast_axes: Some(WmmaUpcastAxes {
            a: axes(tc.elements_per_thread.0),
            b: axes(tc.elements_per_thread.1),
            c: axes(tc.elements_per_thread.2),
        }),
        reduce_axes: vec![],
    }
}

fn mma_atoms(arch: GpuArch) -> Vec<MmaAtom> {
    let renderer = match arch {
        GpuArch::Amd(a) => Renderer::for_amd_arch(a),
        GpuArch::Cuda(c) => Renderer::for_cuda_arch(c),
        GpuArch::Metal(f) => Renderer::for_metal_family(f),
    };
    // (N, M, K) of the scheduler table, and the operand layouts of that shape.
    let (dims, a, b, c): (_, Layout, Layout, Layout) = match arch {
        GpuArch::Cuda(_) => ((8, 16, 16), frag::mma_sync_a(), frag::mma_sync_b(), frag::mma_sync_c()),
        GpuArch::Amd(amd) => {
            let a = amd_operand(amd);
            ((16, 16, 16), a.clone(), a.transpose(), amd_accumulator(amd))
        }
        GpuArch::Metal(_) => {
            let s = frag::simdgroup_8x8();
            ((8, 8, 8), s.transpose(), s.clone(), s)
        }
    };
    renderer
        .tensor_cores
        .iter()
        .filter(|tc| tc.dims == dims && is_half(&tc.dtype_in) && tc.dtype_out == DType::Float32)
        .map(|tc| MmaAtom {
            m: dims.1 as u32,
            n: dims.0 as u32,
            k: dims.2 as u32,
            a: a.clone(),
            b: b.clone(),
            c: c.clone(),
            dtype_in: tc.dtype_in.scalar().expect("a scalar operand dtype"),
            dtype_out: ScalarDType::Float32,
            meta: metadata(tc, renderer.device),
        })
        .collect()
}

fn is_half(dtype: &DType) -> bool {
    matches!(dtype.scalar(), Some(ScalarDType::BFloat16 | ScalarDType::Float16))
}

fn amd_operand(arch: AmdArch) -> Layout {
    if arch.is_cdna() {
        frag::mfma_16x16x16()
    } else if arch.is_rdna4() {
        frag::wmma_gfx12()
    } else {
        frag::wmma_gfx11_input()
    }
}

fn amd_accumulator(arch: AmdArch) -> Layout {
    if arch.is_cdna() {
        frag::mfma_16x16x16().transpose()
    } else if arch.is_rdna4() {
        frag::wmma_gfx12().transpose()
    } else {
        frag::wmma_gfx11_acc()
    }
}

/// The CUDA sm_86 target of the development machine's RTX 3060 (28 SMs).
pub fn sm86() -> Target {
    Target { sms: Some(28), ..Target::for_arch(GpuArch::Cuda(CudaArch { major: 8, minor: 6 })) }
}
