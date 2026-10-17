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

impl MmaAtom {
    /// Whether the atom can issue [`crate::ir::Orient::Swapped`]: a square
    /// atom, whose `m` and `n` extents are interchangeable.
    pub fn swappable(&self) -> bool {
        self.m == self.n
    }
}

/// A warpgroup matrix instruction: four consecutive warps issue
/// `C[64, n] += A[64, 16] · B[k, n]` together (Hopper's `wgmma.mma_async`),
/// `n` a power of two up to `n_max`. `B` is read from shared memory through
/// a descriptor over the K-major rows a fill leaves, XOR-swizzled as the
/// lowering swizzles them; `A` likewise, or from registers when it is a
/// register tile. The accumulator is spread over the warpgroup, each warp
/// holding its 16 rows in [`frag::wgmma_c`], so it feeds the next product's
/// register `A` ([`frag::wgmma_a`]) as it stands.
///
/// Blackwell's `tcgen05.mma` is the same product over the same descriptors
/// and slots in as a second implementation of this class; the two places it
/// differs are named here: [`AccResidence`] (its accumulator lives in tensor
/// memory and is read out explicitly) and [`IssueScope`] (one thread issues
/// it for the warpgroup).
#[derive(Clone, Debug, PartialEq)]
pub struct GroupMmaAtom {
    pub n_max: u32,
    pub dtype_in: ScalarDType,
    pub dtype_out: ScalarDType,
    pub residence: AccResidence,
    pub scope: IssueScope,
    pub device: RendererDevice,
}

/// Where a warpgroup product's accumulator lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccResidence {
    /// Spread over the warpgroup's registers (Hopper).
    Registers,
    /// Tensor memory, read into registers on request (Blackwell).
    TensorMemory,
}

/// Which threads issue a warpgroup product.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IssueScope {
    /// Every thread of the warpgroup, aligned (Hopper).
    Warpgroup,
    /// One elected thread (Blackwell).
    Thread,
}

impl GroupMmaAtom {
    pub const M: u32 = 64;
    pub const K: u32 = 16;
    pub const WARPS: u32 = 4;

    /// The widest instruction `n` that tiles `cols`: the largest power of
    /// two up to `n_max` dividing it (a 96-wide tile issues three `n32`).
    pub fn n_for(&self, cols: u32) -> Option<u32> {
        (3..=self.n_max.trailing_zeros()).rev().map(|b| 1 << b).find(|n| cols.is_multiple_of(*n))
    }

    pub fn c(&self, n: u32) -> Layout {
        frag::wgmma_c(n)
    }

    pub fn a(&self) -> Layout {
        frag::wgmma_a()
    }
}

/// What an SM (compute unit) runs at once, per lane: its SIMDs, the
/// register file each gives its waves, and how many waves each schedules.
/// [`Target::for_arch`] fills in the architecture's facts; [`Target::for_device`]
/// replaces what the device reports (SIMDs and waves on AMD, the register
/// file and waves on CUDA).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Occupancy {
    pub simds: u32,
    /// 32-bit registers per lane in a SIMD's file, shared by its waves.
    pub file: u32,
    /// Waves a SIMD schedules at most.
    pub waves: u32,
    /// Registers one lane may hold; past it a kernel spills.
    pub registers: u32,
    /// Accumulator registers a lane holds beside `registers` where the
    /// matrix core has a file of its own (CDNA's AGPRs); 0 where accumulators
    /// take vector registers.
    pub acc_file: u32,
}

impl Occupancy {
    /// Whether a lane holds `acc` accumulators and `rest` other registers at once.
    pub fn holds(&self, acc: u32, rest: u32) -> bool {
        if self.acc_file > 0 { acc <= self.acc_file && rest <= self.registers } else { acc + rest <= self.registers }
    }
}

/// What a target offers the lowering.
#[derive(Clone, Debug)]
pub struct Target {
    pub arch: GpuArch,
    pub wave: u32,
    pub mma: Vec<MmaAtom>,
    /// Warpgroup matrix instructions over shared descriptors, where the
    /// target has them (Hopper).
    pub group_mma: Vec<GroupMmaAtom>,
    /// Asynchronous global→shared copies (`cp.async`).
    pub cp_async: bool,
    /// Warp-collective 8×8 b16 fragment loads from shared memory.
    pub ldmatrix: bool,
    /// Transposing 16-byte loads of 16×16 b16 blocks from global memory, a
    /// column run per lane (`global_load_tr_b128`): RDNA4.
    pub tr_load: bool,
    /// A bf16 pair's dot product accumulated in f32 (`v_dot2_f32_bf16`):
    /// RDNA3 on, and gfx950.
    pub dot2_bf16: bool,
    /// Shared memory a block may use without special launch attributes.
    pub smem_bytes: usize,
    /// Streaming multiprocessors (compute units), when the device reports them.
    pub sms: Option<u32>,
    /// Dies a dispatch deals its blocks to round-robin, each with its own L2
    /// (CDNA3's XCDs): the lowering renumbers blocks so consecutive ones share
    /// a die ([`crate::lower::emit::die_order`]). 1 where a device is one die.
    pub xcds: u32,
    pub occupancy: Occupancy,
    /// The machine scheduler hoists a pipeline trip's shared-memory commit above
    /// its products unless fenced (`Sync::Fence` → `sched_barrier`), leaving
    /// the block waiting on global memory with no product in flight: RDNA4.
    pub commit_fence: bool,
    /// An f32 pair rounds to bf16 in one instruction (`cvt.rn.bf16x2.f32`,
    /// `v_cvt_pk_bf16_f32`): sm_80+ and gfx950; elsewhere the rounding is
    /// about seven vector instructions per element.
    pub bf16_cvt: bool,
    /// The matrix products issue on the vector ALUs (RDNA's WMMA), so vector
    /// work saved is matrix time gained; CUDA's tensor cores and CDNA's matrix
    /// cores run beside the vector units.
    pub mma_on_valu: bool,
    /// How a lane reads the register of lane `lane ^ mask`.
    pub lane_move: LaneMove,
    /// An f32 `exp2` is the hardware instruction alone, without the range
    /// reduction that keeps results below 2^-126 denormal (they flush to
    /// zero, which no kernel tells apart): AMD's `v_exp_f32`.
    pub hardware_exp2: bool,
}

/// How a lane reads the register of lane `lane ^ mask`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaneMove {
    /// `shfl.sync.bfly` (CUDA).
    Shuffle,
    /// `simd_shuffle_xor` (Metal).
    SimdShuffle,
    /// `ds_bpermute`, a trip through the LDS crossbar (CDNA).
    Permute,
    /// On the vector unit itself: DPP `row_xmask` within a row of 16 lanes and
    /// `permlanex16` between the two rows of a wave32 (RDNA); a mask past 31
    /// takes [`LaneMove::Permute`].
    Dpp,
}

/// What a device's primitives cost, measured on it by [`crate::probe::costs`]
/// or declared by a test. The throughput fields are ns of one SM's time per
/// unit with the device full (device time × SMs / units), so a ranking sums
/// them per block and per round; the relayout fields (`lane_move`,
/// `smem_trip`, `scatter`) are what one wave waits per element
/// on its dependent chain with one wave per SIMD, since a relayout sits
/// between two products; `latency` is absolute.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Costs {
    /// One byte of a wave's operand fragments read from shared memory.
    pub smem_byte: f64,
    /// One byte streamed from global memory into a block (its fill).
    pub global_byte: f64,
    /// One 16-bit element a wave reads from global memory scattered: a lane's
    /// run under 16 bytes that the next lane does not continue (a line per lane).
    /// Measured on lines in cache, so a bias read from DRAM costs more.
    pub scatter: f64,
    /// One 16×16×16 matrix-core step of a wave, the SM full of waves: the
    /// core's throughput.
    pub step: f64,
    /// The same step issued as part of a warpgroup product
    /// ([`GroupMmaAtom`]), 0 where the target has none.
    pub step_group: f64,
    /// The same step of a wave alone on its SIMD: a dependent chain's
    /// latency, which `step_lone / step` waves a SIMD hide.
    pub step_lone: f64,
    /// One score element's softmax vector work (scale, max, exp2, sum).
    pub score: f64,
    /// One pipeline trip's fixed overhead (barrier, ring bubble), unrolled.
    pub trip: f64,
    /// The same for a rolled ring (slot offsets recomputed per trip).
    pub trip_rolled: f64,
    /// The same over a ring of one slot: its commit waits behind a
    /// block-wide barrier for every warp to leave the slot and is followed
    /// by another, both exposed every trip (a two-slot commit sits behind a
    /// fence); on a target with asynchronous copies, its single-stage
    /// expansion's wait and barriers.
    pub trip_one: f64,
    /// One unit of an exchange's [`crate::layouts::Relayout::work`] (a lane
    /// move, or a select): averaged over the moves a half-wave exchange
    /// issues, on RDNA a `permlanex16` across the rows of 16 and a DPP
    /// `row_xmask` within them.
    pub lane_move: f64,
    /// One element's round trip through shared memory (store, barrier, load).
    pub smem_trip: f64,
    /// Global memory latency one trip waits unhidden, ns.
    pub latency: f64,
    /// One byte streamed into a block from the cache in front of DRAM (L2),
    /// as `global_byte` from a footprint that stays in it.
    pub l2_byte: f64,
    /// That cache's bytes: operands that fit in it come from DRAM once and
    /// are re-read from it. 0 where unmeasured: every fill byte is then
    /// priced at `global_byte`.
    pub l2_bytes: f64,
}

impl Costs {
    /// Every field a pair of programs measures, by name, in declaration
    /// order: all but the cache's, which a series of footprints measures.
    pub fn fields(&mut self) -> [(&'static str, &mut f64); 13] {
        let Costs {
            smem_byte,
            global_byte,
            scatter,
            step,
            step_group,
            step_lone,
            score,
            trip,
            trip_rolled,
            trip_one,
            lane_move,
            smem_trip,
            latency,
            l2_byte: _,
            l2_bytes: _,
        } = self;
        [
            ("smem_byte", smem_byte),
            ("global_byte", global_byte),
            ("scatter", scatter),
            ("step", step),
            ("step_group", step_group),
            ("step_lone", step_lone),
            ("score", score),
            ("trip", trip),
            ("trip_rolled", trip_rolled),
            ("trip_one", trip_one),
            ("lane_move", lane_move),
            ("smem_trip", smem_trip),
            ("latency", latency),
        ]
    }
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
        let commit_fence = matches!(arch, GpuArch::Amd(a) if a.is_rdna4());
        let bf16_cvt = match arch {
            GpuArch::Cuda(c) => c.major >= 8,
            GpuArch::Amd(a) => a == AmdArch::Gfx950,
            GpuArch::Metal(_) => false,
        };
        let mma_on_valu = matches!(arch, GpuArch::Amd(a) if !a.is_cdna());
        let lane_move = match arch {
            GpuArch::Cuda(_) => LaneMove::Shuffle,
            GpuArch::Metal(_) => LaneMove::SimdShuffle,
            GpuArch::Amd(a) if a.is_cdna() => LaneMove::Permute,
            GpuArch::Amd(_) => LaneMove::Dpp,
        };
        let hardware_exp2 = matches!(arch, GpuArch::Amd(_));
        let tr_load = matches!(arch, GpuArch::Amd(a) if a.is_rdna4() && wave == 32);
        let dot2_bf16 = matches!(arch, GpuArch::Amd(a) if !a.is_cdna() || a == AmdArch::Gfx950);
        let occupancy = match arch {
            // Four schedulers over 64K registers; 48 warps on sm_86, sm_89 and
            // sm_120, 64 elsewhere.
            GpuArch::Cuda(c) => {
                let waves = if (c.major == 8 && c.minor >= 6) || c.major == 12 { 12 } else { 16 };
                Occupancy { simds: 4, file: 512, waves, registers: 255, acc_file: 0 }
            }
            // CDNA: four SIMD16s over a unified file of 512 registers a lane, a
            // wave holding up to 256 vector and 256 accumulator (AGPR) ones.
            GpuArch::Amd(a) if a.is_cdna() => {
                Occupancy { simds: 4, file: 512, waves: 8, registers: 256, acc_file: 256 }
            }
            // RDNA: two SIMD32s with 192 KB of registers each.
            GpuArch::Amd(_) => Occupancy { simds: 2, file: 1536, waves: 16, registers: 256, acc_file: 0 },
            GpuArch::Metal(_) => Occupancy { simds: 4, file: 1024, waves: 8, registers: 128, acc_file: 0 },
        };
        Self {
            arch,
            wave,
            mma: mma_atoms(arch),
            group_mma: group_mma_atoms(arch),
            cp_async,
            ldmatrix,
            tr_load,
            dot2_bf16,
            smem_bytes,
            sms: None,
            xcds: 1,
            commit_fence,
            bf16_cvt,
            mma_on_valu,
            occupancy,
            lane_move,
            hardware_exp2,
        }
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
                    if let Some(sms) = node.simd_count.checked_div(node.simd_per_cu).filter(|&sms| sms > 0) {
                        target.sms = Some(sms);
                        target.occupancy.simds = node.simd_per_cu;
                    }
                    if node.max_waves_per_simd > 0 {
                        target.occupancy.waves = node.max_waves_per_simd;
                    }
                    target.xcds = node.num_xcc.max(1);
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
            let lanes = target.occupancy.simds * limits.warp_size.max(1);
            if limits.registers_per_sm > 0 {
                target.occupancy.file = limits.registers_per_sm / lanes;
            }
            if limits.max_threads_per_sm > 0 {
                target.occupancy.waves = limits.max_threads_per_sm / lanes;
            }
        }
        Some(target)
    }

    /// How a pipeline fills shared memory: `cp.async` where the target has
    /// it, else through registers (RDNA3/RDNA4 have no global→LDS copy).
    pub fn prefetch(&self) -> Prefetch {
        if self.cp_async { Prefetch::CpAsync } else { Prefetch::RegisterStaged }
    }

    /// Pipeline stages the target's prefetch can use: register staging
    /// runs one step ahead over one or two slots.
    pub fn stages(&self, stages: usize) -> usize {
        match self.prefetch() {
            Prefetch::CpAsync => stages,
            Prefetch::RegisterStaged => stages.clamp(1, 2),
        }
    }

    /// Registers one lane of a block of `waves` warps may hold: the lane's
    /// limit, or less where the block's warps share a SIMD's file (sixteen
    /// warps over four SIMDs of 512 registers a lane: 128 each).
    pub fn lane_budget(&self, waves: u32) -> u32 {
        let o = self.occupancy;
        o.registers.min(o.file * o.simds / waves.max(1))
    }

    /// The matrix core for `dtype_in → dtype_out`, if the target has one.
    pub fn mma(&self, dtype_in: ScalarDType, dtype_out: ScalarDType) -> Option<&MmaAtom> {
        self.mma.iter().find(|a| a.dtype_in == dtype_in && a.dtype_out == dtype_out)
    }

    /// The warpgroup instruction for `dtype_in → dtype_out`, if the target has one.
    pub fn group_mma(&self, dtype_in: ScalarDType, dtype_out: ScalarDType) -> Option<&GroupMmaAtom> {
        self.group_mma.iter().find(|a| a.dtype_in == dtype_in && a.dtype_out == dtype_out)
    }
}

/// Hopper's `wgmma.mma_async` over bf16 and f16 into f32; nothing elsewhere.
fn group_mma_atoms(arch: GpuArch) -> Vec<GroupMmaAtom> {
    let GpuArch::Cuda(c) = arch else { return vec![] };
    if c.major != 9 {
        return vec![];
    }
    let device = Renderer::for_cuda_arch(c).device;
    [ScalarDType::BFloat16, ScalarDType::Float16]
        .map(|dtype_in| GroupMmaAtom {
            n_max: 256,
            dtype_in,
            dtype_out: ScalarDType::Float32,
            residence: AccResidence::Registers,
            scope: IssueScope::Warpgroup,
            device,
        })
        .to_vec()
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

/// A device as the planner and the lowering see it: its declared facts and
/// its measured costs.
#[derive(Clone, Debug)]
pub struct Device {
    pub target: Target,
    pub costs: Costs,
}

impl Costs {
    /// One pipeline trip's fixed overhead over a ring `stages` deep: a ring
    /// of one slot pays its barrier pair, a `rolled` register-staged ring of
    /// two recomputes its slot offsets every trip.
    pub fn trip(&self, stages: usize, rolled: bool) -> f64 {
        match stages {
            ..=1 => self.trip_one,
            _ if rolled => self.trip_rolled,
            _ => self.trip,
        }
    }
}

/// The CUDA sm_86 target of the development machine's RTX 3060 (28 SMs).
pub fn sm86() -> Target {
    Target { sms: Some(28), ..Target::for_arch(GpuArch::Cuda(CudaArch { major: 8, minor: 6 })) }
}
