//! Hardware instructions as atoms that carry their own layouts. A kernel never
//! names one; the lowering picks atoms from the [`Target`] and the inference
//! propagates their layouts to the values around them.

use svod_dtype::{AmdArch, CudaArch, DType, GpuArch, MetalFamily, ScalarDType};
use svod_ir::{AxisId, RendererDevice, WmmaMetadata, WmmaUpcastAxes};
use svod_schedule::optimizer::{Renderer, TensorCore};

use crate::ir::Orient;
use crate::layout::{self as frag, Layout};
use crate::layouts::{Issue, WarpGrid};

/// Which matrix instruction a config's products issue on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Core {
    /// The warp-level atom ([`MmaAtom`]): operands in registers.
    #[default]
    Warp,
    /// The warpgroup atom ([`GroupMmaAtom`]): a grid's rows are warpgroups
    /// of four warps, and shared operands are read through descriptors.
    Group,
}

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
    /// Whether the atom can issue [`crate::ir::Orient::Swapped`]: a square
    /// atom, whose `m` and `n` extents are interchangeable.
    pub fn swappable(&self) -> bool {
        self.m == self.n
    }

    /// The `[m, n]` product over `grid` ([`Self::issue`]) issued the way
    /// round that spreads a row of its result over the fewest lanes (its
    /// rows fold across the fewest masks,
    /// [`crate::layouts::TileLayout::row_fold_masks`]): where a row
    /// reduction folds cheapest, and where a deal of its rows
    /// ([`crate::ir::TileOp::Deal`]) into as many columns a lane renames
    /// registers ([`crate::layouts::TileLayout::dealt`]). Direct unless
    /// swapped spreads narrower: RDNA and CDNA hold a row in a lane
    /// swapped and over 16 lanes direct; CUDA's `m16n8k16` cannot swap.
    pub fn narrow(&self, grid: WarpGrid, m: usize, n: usize, k: usize) -> crate::layouts::Result<(Orient, Issue)> {
        let direct = self.issue(Orient::Direct, grid, m, n, k)?;
        if !self.swappable() {
            return Ok((Orient::Direct, direct));
        }
        let swapped = self.issue(Orient::Swapped, grid, m, n, k)?;
        let sharing = |issue: &Issue| issue.c.row_fold_masks().len();
        Ok(if sharing(&swapped) < sharing(&direct) { (Orient::Swapped, swapped) } else { (Orient::Direct, direct) })
    }
}

/// A warpgroup matrix instruction: four consecutive warps issue
/// `C[64, n] += A[64, 16] · B[16, n]` together (Hopper's `wgmma.mma_async`),
/// `n` a power of two up to `n_max`. `B` is read from shared memory through
/// a descriptor over the rows a fill leaves, K-major or MN-major (the
/// instruction transposing them), XOR-swizzled as the lowering swizzles them; `A` likewise, or from registers when it is a
/// register tile. The accumulator is spread over the warpgroup, each warp
/// holding its 16 rows in [`frag::wgmma_c`], so it feeds the next product's
/// register `A` ([`frag::wgmma_a`]) as it stands.
#[derive(Clone, Debug, PartialEq)]
pub struct GroupMmaAtom {
    pub n_max: u32,
    pub dtype_in: ScalarDType,
    pub dtype_out: ScalarDType,
    pub residence: AccResidence,
    pub scope: IssueScope,
    pub device: RendererDevice,
}

/// Where a warpgroup product's accumulator lives; only `Registers` is lowered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccResidence {
    /// Spread over the warpgroup's registers (Hopper).
    Registers,
    /// Tensor memory, read into registers on request (Blackwell).
    TensorMemory,
}

/// Which threads issue a warpgroup product; only `Warpgroup` is lowered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IssueScope {
    /// Every thread of the warpgroup, aligned (Hopper).
    Warpgroup,
    /// One elected thread (Blackwell).
    Thread,
}

impl GroupMmaAtom {
    /// The rows of one instruction's product.
    pub const M: u32 = 64;
    /// The depth of one instruction over 16-bit inputs.
    pub const K: u32 = 16;
    /// The warps of a warpgroup.
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
    /// How global memory reaches a shared ring.
    pub fill: Fill,
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
    /// Bytes of device memory, when the device reports them: the share a
    /// planner's scratch may take scales with it.
    pub memory_bytes: Option<u64>,
    /// Crews a block's warps may split into, one stream of clusters offset by
    /// a cluster so one crew's products overlap the other's loads: two where
    /// the matrix core runs beside the vector units and a lane's file holds
    /// two crews' accumulators (CDNA), one elsewhere
    /// ([`crate::schedule::Schedule::PingPong`]).
    pub crews: u32,
    /// Dies a dispatch deals its blocks to round-robin, each with its own L2
    /// (CDNA3's XCDs): the lowering renumbers blocks so consecutive ones share
    /// a die ([`crate::lower::emit::die_order`]). 1 where a device is one die.
    pub xcds: u32,
    pub occupancy: Occupancy,
    /// What the semantic syncs a template places become.
    pub sync: SyncCaps,
    /// How a lane reads the register of lane `lane ^ mask`.
    pub lane_move: LaneMove,
    /// An f32 `exp2` is the hardware instruction alone, without the range
    /// reduction that keeps results below 2^-126 denormal (they flush to
    /// zero, which no kernel tells apart): AMD's `v_exp_f32`.
    pub hardware_exp2: bool,
}

/// How global memory reaches a shared ring.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Fill {
    /// Asynchronous 16-byte copies (`cp.async.cg`, sm_80+), landed by
    /// commit and wait groups, `stages - 1` steps ahead.
    CpAsync,
    /// Global → registers before a step's compute and registers → shared
    /// after it, one step ahead (RDNA and CDNA have no global → LDS copy):
    /// the chunks take registers, and a rolled ring recomputes its slot
    /// offsets every trip.
    Staged,
}

impl Fill {
    /// The ring depth this fill runs for `stages` asked: any, or one or two
    /// slots through registers.
    pub fn stages(self, stages: usize) -> usize {
        match self {
            Fill::CpAsync => stages,
            Fill::Staged => stages.clamp(1, 2),
        }
    }
}

/// The semantic syncs ([`crate::ir::Sync`]) that are an instruction on a
/// target; each is nothing elsewhere.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SyncCaps {
    /// A warp's own barrier is `bar.warp.sync`: CUDA's lanes schedule
    /// independently, where an AMD wave issues its LDS accesses in order and
    /// waits on their counters.
    pub warp_barrier: bool,
    /// The machine scheduler moves a pipeline trip's global loads below its
    /// products, or its shared-memory commit above them, unless fenced
    /// (`Sync::Fence` → `sched_barrier`), leaving the block waiting on global
    /// memory with no product in flight: RDNA (gfx1151 sank a two-slot
    /// 128×128×64 GEMM trip's loads to its commit; gfx1201 hoisted a commit).
    pub commit_fence: bool,
    /// A wave's issue priority is settable (`Sync::Prio` → `s_setprio`): AMD.
    pub priority: bool,
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

impl LaneMove {
    /// The instructions moving a register from lane `lane ^ mask`: on RDNA a
    /// DPP `row_xmask` for the bits within a row of 16 and a `permlanex16`
    /// for the bit across them, one instruction elsewhere.
    pub fn issues(self, mask: u32) -> u32 {
        match self {
            LaneMove::Dpp if mask < 32 => u32::from(mask & 15 != 0) + u32::from(mask & 16 != 0),
            _ => 1,
        }
    }
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
    /// The same step's 16×16×16 products over 8-bit integers into i32
    /// ([`Target::geometry_of`]), the SM full of waves; 0 where the target
    /// has no such core.
    pub step_i8: f64,
    /// The same step of a wave alone on its SIMD: a dependent chain's
    /// latency, which `step_lone / step` waves a SIMD hide.
    pub step_lone: f64,
    /// One score element's softmax vector work (scale, max, exp2, sum).
    pub score: f64,
    /// One pipeline trip's fixed overhead (barrier, ring bubble), unrolled,
    /// with the SM full of blocks: a block-trip's share of the SM's time
    /// once the others hide its waits ([`Self::trip_at`]).
    pub trip: f64,
    /// The same for a rolled ring (slot offsets recomputed per trip),
    /// register-staged: 0 where fills are `cp.async`, whose ring
    /// ([`Self::trip`]) never reads it.
    pub trip_rolled: f64,
    /// The same over a ring of one slot: its commit waits behind a
    /// block-wide barrier for every warp to leave the slot and is followed
    /// by another, both exposed every trip (a two-slot commit sits behind a
    /// fence); on a target with asynchronous copies, its single-stage
    /// expansion's wait and barriers.
    pub trip_one: f64,
    /// [`Self::trip`] of a block alone on its SM: the wall time its waits
    /// and barriers add to the block's trip, which other blocks' work hides
    /// ([`Self::trip_at`]).
    pub trip_lone: f64,
    /// [`Self::trip_one`] of a block alone on its SM.
    pub trip_one_lone: f64,
    /// One unit of an exchange's [`crate::layouts::Relayout::work`] (a lane
    /// move, or a select): averaged over the moves a half-wave exchange
    /// issues, on RDNA a `permlanex16` across the rows of 16 and a DPP
    /// `row_xmask` within them.
    pub lane_move: f64,
    /// One element's round trip through shared memory (store, barrier, load).
    pub smem_trip: f64,
    /// Global memory latency one trip of a two-slot ring waits unhidden, ns:
    /// what prefetching one trip ahead leaves.
    pub latency: f64,
    /// The same of a one-slot ring on a target with asynchronous copies, ns:
    /// its single-stage expansion prefetches nothing, so a trip waits out its
    /// whole load. 0 where fills are register-staged: a one-slot ring loads a
    /// trip ahead as two do ([`Self::latency`]).
    pub latency_one: f64,
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
    pub fn fields(&mut self) -> [(&'static str, &mut f64); 17] {
        let Costs {
            smem_byte,
            global_byte,
            scatter,
            step,
            step_group,
            step_i8,
            step_lone,
            score,
            trip,
            trip_rolled,
            trip_one,
            trip_lone,
            trip_one_lone,
            lane_move,
            smem_trip,
            latency,
            latency_one,
            l2_byte: _,
            l2_bytes: _,
        } = self;
        [
            ("smem_byte", smem_byte),
            ("global_byte", global_byte),
            ("scatter", scatter),
            ("step", step),
            ("step_group", step_group),
            ("step_i8", step_i8),
            ("step_lone", step_lone),
            ("score", score),
            ("trip", trip),
            ("trip_rolled", trip_rolled),
            ("trip_one", trip_one),
            ("trip_lone", trip_lone),
            ("trip_one_lone", trip_one_lone),
            ("lane_move", lane_move),
            ("smem_trip", smem_trip),
            ("latency", latency),
            ("latency_one", latency_one),
        ]
    }
}

impl Target {
    /// The most instructions a lowered program may emit for this target;
    /// past it the lowering refuses the program
    /// ([`crate::lower::Error::TooLarge`]) and the tuner rejects the config
    /// as one that fails to compile, before the backend's compiler is
    /// handed it. The largest kernels the planner ranks (nearest lists at
    /// depth 128, `k` 64–80, 16384² rows) emit 270k on sm_86 (a register
    /// list of 64 over 8-bit rows, a three-slot ring unrolled; the NVPTX
    /// backend folds a 211k one to 25k lines of PTX, 236 registers, no
    /// spill) and 176k on gfx1201 (a queued list of 80 over 8-bit rows,
    /// unrolled: 4 s of clang), where a queued list whose product the
    /// inference issued the wide way round emitted 400k, which took clang
    /// 100 s and 700 MB a config and a pool of them the RX 9070 XT's 60 GB.
    pub fn most_instructions(&self) -> usize {
        match self.arch {
            GpuArch::Amd(_) => 1 << 18,
            GpuArch::Cuda(_) | GpuArch::Metal(_) => 1 << 19,
        }
    }

    pub fn for_arch(arch: GpuArch) -> Self {
        match arch {
            GpuArch::Cuda(c) => cuda(c),
            GpuArch::Amd(a) => amd(a),
            GpuArch::Metal(f) => metal(f),
        }
    }

    /// The target of a card this machine has no device for: what
    /// [`Self::for_device`] reports for it, the architecture's facts with the
    /// card's SMs (CUs) and the dies its dispatch spreads blocks over.
    pub fn offline(arch: GpuArch, sms: u32, xcds: u32) -> Self {
        Self { sms: Some(sms), xcds, ..Self::for_arch(arch) }
    }

    /// The [`Self::offline`] target of a card of [`CARDS`] by name.
    pub fn card(name: &str) -> Option<Self> {
        CARDS.iter().find(|c| c.0 == name).map(|&(_, arch, sms, xcds)| Self::offline(arch, sms, xcds))
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
                    target.memory_bytes = (node.memory_bytes > 0).then_some(node.memory_bytes);
                }
                return Some(target);
            }
            DeviceSpec::Metal { device_id } => GpuArch::Metal(reg::resolve_metal_family(*device_id).ok()?),
            DeviceSpec::Cpu | DeviceSpec::WebGpu | DeviceSpec::Disk { .. } => return None,
        };
        let mut target = Self::for_arch(arch);
        if let DeviceSpec::Cuda { device_id } = spec {
            target.memory_bytes = reg::resolve_cuda_memory(*device_id).ok().map(|b| b as u64);
        }
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

    /// The warp-level matrix core's geometry over 16-bit operands: its
    /// `m × n × k` and operand layouts, which every 16-bit input's atom
    /// shares (`test::unit::atoms::every_atom_of_a_target_shares_its_geometry`).
    /// The planner prices it; the lowering issues the dtype's own
    /// ([`Self::mma`]).
    pub fn geometry(&self) -> Option<&MmaAtom> {
        self.mma.iter().find(|a| a.dtype_in.bytes() == 2)
    }

    /// The warp-level matrix core over `dtype_in` into its accumulator (f32
    /// for a 16-bit float, i32 for an 8-bit integer): an 8-bit atom is as
    /// deep as its bytes allow (`k` 32 on NVIDIA and CDNA, 16 on RDNA).
    pub fn geometry_of(&self, dtype_in: ScalarDType) -> Option<&MmaAtom> {
        self.mma.iter().find(|a| a.dtype_in == dtype_in)
    }

    /// The `[m, n] += [m, k] · [k, n]` product over `grid` on `core`'s
    /// [geometry](Self::geometry) as the planner prices it, issued direct;
    /// `None` where the target lacks the core or the product does not tile
    /// its atoms.
    pub fn issue(&self, core: Core, grid: WarpGrid, m: usize, n: usize, k: usize) -> Option<Issue> {
        match core {
            Core::Warp => self.geometry()?.issue(Orient::Direct, grid, m, n, k).ok(),
            Core::Group => self.group_mma.first()?.issue(grid, m, n, k).ok(),
        }
    }

    /// The warpgroup instruction for `dtype_in → dtype_out`, if the target has one.
    pub fn group_mma(&self, dtype_in: ScalarDType, dtype_out: ScalarDType) -> Option<&GroupMmaAtom> {
        self.group_mma.iter().find(|a| a.dtype_in == dtype_in && a.dtype_out == dtype_out)
    }
}

/// A CUDA part: `cp.async` fills from sm_80, `ldmatrix` from sm_75, the
/// `mma.sync` m16n8k16 core and Hopper's warpgroup one; four schedulers over
/// 64K registers, 48 warps on sm_86, sm_89 and sm_120, 64 elsewhere.
fn cuda(c: CudaArch) -> Target {
    let renderer = Renderer::for_cuda_arch(c);
    let waves = if (c.major == 8 && c.minor >= 6) || c.major == 12 { 12 } else { 16 };
    let mut mma = mma_atoms(&renderer, (8, 16, 16), [frag::mma_sync_a(), frag::mma_sync_b(), frag::mma_sync_c()]);
    let bytes = [frag::mma_sync_a8(), frag::mma_sync_b8(), frag::mma_sync_c()];
    mma.extend(int8_atom(renderer.tensor_cores.iter(), renderer.device, (8, 16, 32), bytes));
    Target {
        arch: GpuArch::Cuda(c),
        wave: c.wave_size(),
        mma,
        group_mma: if c.major == 9 { group_mma_atoms(renderer.device) } else { vec![] },
        fill: if c.major >= 8 { Fill::CpAsync } else { Fill::Staged },
        ldmatrix: (c.major, c.minor) >= (7, 5),
        tr_load: false,
        dot2_bf16: false,
        smem_bytes: c.max_shared_per_block_optin(),
        sms: None,
        memory_bytes: None,
        crews: 1,
        xcds: 1,
        occupancy: Occupancy { simds: 4, file: 512, waves, registers: 255, acc_file: 0 },
        sync: SyncCaps { warp_barrier: true, commit_fence: false, priority: false },
        lane_move: LaneMove::Shuffle,
        hardware_exp2: false,
    }
}

/// An AMD part: register-staged fills into 64 KB of LDS a workgroup (160 KB
/// on gfx950), the 16×16×16 core, settable issue priority and the hardware
/// `exp2`. CDNA: MFMA on wave64, four SIMD16s over a unified file of 512
/// registers a lane (up to 256 vector and 256 accumulator ones a wave),
/// lane moves through the LDS crossbar, two crews. RDNA: WMMA on wave32,
/// two SIMD32s with 192 KB of registers each, DPP lane moves, the bf16 pair
/// dot product; RDNA4 adds transposing loads and needs its commit fenced.
fn amd(a: AmdArch) -> Target {
    let cdna = a.is_cdna();
    let (operand, accumulator) = if cdna {
        (frag::mfma_16x16x16(), frag::mfma_16x16x16().transpose())
    } else if a.is_rdna4() {
        (frag::wmma_gfx12(), frag::wmma_gfx12().transpose())
    } else {
        (frag::wmma_gfx11_input(), frag::wmma_gfx11_acc())
    };
    let layouts = [operand.clone(), operand.transpose(), accumulator.clone()];
    let renderer = Renderer::for_amd_arch(a);
    let mut mma = mma_atoms(&renderer, (16, 16, 16), layouts.clone());
    // CDNA's 8-bit core takes eight bytes a lane, twice the 16-bit depth (the
    // scheduler's table lists it only as fp8, so it is built here); RDNA's
    // `iu8` takes its 16-bit operands' layouts packed into fewer registers.
    let int8 = if cdna {
        let tc = svod_schedule::optimizer::renderer::AMD_CDNA_161632.build(DType::Int8, DType::Int32);
        let operand = frag::mfma_16x16x32_i8();
        int8_atom([tc].iter(), renderer.device, (16, 16, 32), [operand.clone(), operand.transpose(), accumulator])
    } else {
        int8_atom(renderer.tensor_cores.iter(), renderer.device, (16, 16, 16), layouts)
    };
    mma.extend(int8);
    let occupancy = if cdna {
        Occupancy { simds: 4, file: 512, waves: 8, registers: 256, acc_file: 256 }
    } else {
        Occupancy { simds: 2, file: 1536, waves: 16, registers: 256, acc_file: 0 }
    };
    Target {
        arch: GpuArch::Amd(a),
        wave: a.wave_size(),
        mma,
        group_mma: vec![],
        fill: Fill::Staged,
        ldmatrix: false,
        tr_load: a.is_rdna4(),
        dot2_bf16: !cdna || a == AmdArch::Gfx950,
        smem_bytes: if a == AmdArch::Gfx950 { 160 << 10 } else { 64 << 10 },
        sms: None,
        memory_bytes: None,
        crews: if cdna { 2 } else { 1 },
        xcds: 1,
        occupancy,
        sync: SyncCaps { warp_barrier: false, commit_fence: !cdna, priority: true },
        lane_move: if cdna { LaneMove::Permute } else { LaneMove::Dpp },
        hardware_exp2: true,
    }
}

/// An Apple part: `simdgroup_matrix` 8×8×8 products, no kernels yet.
fn metal(f: MetalFamily) -> Target {
    let s = frag::simdgroup_8x8();
    Target {
        arch: GpuArch::Metal(f),
        wave: 32,
        mma: mma_atoms(&Renderer::for_metal_family(f), (8, 8, 8), [s.transpose(), s.clone(), s]),
        group_mma: vec![],
        fill: Fill::Staged,
        ldmatrix: false,
        tr_load: false,
        dot2_bf16: false,
        smem_bytes: 32 << 10,
        sms: None,
        memory_bytes: None,
        crews: 1,
        xcds: 1,
        occupancy: Occupancy { simds: 4, file: 1024, waves: 8, registers: 128, acc_file: 0 },
        sync: SyncCaps { warp_barrier: false, commit_fence: false, priority: false },
        lane_move: LaneMove::SimdShuffle,
        hardware_exp2: false,
    }
}

/// Hopper's `wgmma.mma_async` over bf16 and f16 into f32.
fn group_mma_atoms(device: RendererDevice) -> Vec<GroupMmaAtom> {
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
/// `Wmma` carries: `log2(elements_per_thread)` size-2 upcast axes per
/// operand, no reduce axes.
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

/// The renderer's 16-bit-in, f32-out tensor cores of shape `dims` (the
/// scheduler table's `(N, M, K)`), each with the operand layouts `[a, b, c]`
/// of that shape.
fn mma_atoms(renderer: &Renderer, dims: (usize, usize, usize), layouts: [Layout; 3]) -> Vec<MmaAtom> {
    renderer
        .tensor_cores
        .iter()
        .filter(|tc| tc.dims == dims && is_half(&tc.dtype_in) && tc.dtype_out == DType::Float32)
        .map(|tc| atom(tc, renderer.device, layouts.clone()))
        .collect()
}

/// The 8-bit integer core into i32 of shape `dims` among `cores`, if any,
/// with the operand layouts `[a, b, c]` of its bytes.
fn int8_atom<'a>(
    mut cores: impl Iterator<Item = &'a TensorCore>,
    device: RendererDevice,
    dims: (usize, usize, usize),
    layouts: [Layout; 3],
) -> Option<MmaAtom> {
    let tc = cores.find(|tc| tc.dims == dims && tc.dtype_in == DType::Int8 && tc.dtype_out == DType::Int32)?;
    Some(atom(tc, device, layouts))
}

/// The atom of the tensor core `tc` with the operand layouts `[a, b, c]`.
fn atom(tc: &TensorCore, device: RendererDevice, [a, b, c]: [Layout; 3]) -> MmaAtom {
    let scalar = |d: &DType| d.scalar().expect("a scalar operand dtype");
    MmaAtom {
        m: tc.dims.1 as u32,
        n: tc.dims.0 as u32,
        k: tc.dims.2 as u32,
        a,
        b,
        c,
        dtype_in: scalar(&tc.dtype_in),
        dtype_out: scalar(&tc.dtype_out),
        meta: metadata(tc, device),
    }
}

fn is_half(dtype: &DType) -> bool {
    matches!(dtype.scalar(), Some(ScalarDType::BFloat16 | ScalarDType::Float16))
}

/// A device as the planner and the lowering see it: its declared facts and
/// its measured costs.
#[derive(Clone, Debug)]
pub struct Device {
    pub target: Target,
    pub costs: Costs,
}

impl Costs {
    /// One pipeline trip's fixed overhead over a ring `stages` deep that
    /// `fill` fills, `unroll`ed or not: a ring of one slot pays its barrier
    /// pair, a rolled register-staged ring of two recomputes its slot
    /// offsets every trip; a `cp.async` ring's offsets cost alike either way.
    pub fn trip(&self, fill: Fill, stages: usize, unroll: bool) -> f64 {
        match stages {
            ..=1 => self.trip_one,
            _ if !unroll && fill == Fill::Staged => self.trip_rolled,
            _ => self.trip,
        }
    }

    /// What one block-trip over a ring `stages` deep adds to its SM's time
    /// with `blocks` blocks at once, whose trip with the SM full costs
    /// `trip`: a lone block's trip ([`Self::trip_lone`],
    /// [`Self::trip_one_lone`]) over the blocks to the [`TRIP_HIDING`]
    /// power, never under `trip` (which stands in where the lone trip is
    /// unmeasured).
    pub fn trip_at(&self, stages: usize, trip: f64, blocks: usize) -> f64 {
        let lone = if stages <= 1 { self.trip_one_lone } else { self.trip_lone };
        trip.max(lone / (blocks.max(1) as f64).powi(TRIP_HIDING))
    }
}

/// Model: the power of the blocks an SM runs at once by which a lone
/// block's trip overhead shrinks per block-trip: the blocks overlap one
/// another's waits and share out what a round still waits. Measured per
/// block-trip at one to four blocks an SM, sm_86 falls 127, 41, 16, 3 ns
/// (about the square) and gfx1201 65, 27, 17, 11 (about the blocks); the
/// square ranks both cards' configs closer to their measured order than
/// either card's own exponent.
pub const TRIP_HIDING: i32 = 2;

const fn sm(major: u8, minor: u8) -> GpuArch {
    GpuArch::Cuda(CudaArch { major, minor })
}

/// Cards as a device query reports them: name, architecture, SMs (CUs) and
/// dies (CDNA's XCDs).
pub const CARDS: [(&str, GpuArch, u32, u32); 9] = [
    ("rtx3060", sm(8, 6), 28, 1),
    ("h100-pcie", sm(9, 0), 114, 1),
    ("h100-sxm", sm(9, 0), 132, 1),
    ("rtx5070", sm(12, 0), 48, 1),
    ("rtx5090", sm(12, 0), 170, 1),
    ("mi300x", GpuArch::Amd(AmdArch::Gfx942), 304, 8),
    ("mi355x", GpuArch::Amd(AmdArch::Gfx950), 256, 8),
    ("rx9070xt", GpuArch::Amd(AmdArch::Gfx1201), 64, 1),
    ("strix-halo", GpuArch::Amd(AmdArch::Gfx1151), 40, 1),
];

/// The CUDA sm_86 target of the development machine's RTX 3060 (28 SMs):
/// the host tests' fixture.
#[cfg(test)]
pub fn sm86() -> Target {
    Target::card("rtx3060").expect("a listed card")
}
