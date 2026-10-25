//! Layer 0 of the config rework: what the device says about itself, queried
//! once per device beside its measured [`Costs`](crate::atoms::Costs) —
//! identity, limits, caches and clocks. This phase captures and exposes the
//! facts; every hand value the planner runs on today (the register cap, the
//! occupancy numbers, the smem tables) still drives it unchanged until the
//! model's derivations switch (phase D).
//!
//! Every field is `None` where its vendor offers no query (the KFD topology
//! answers no bus width), never a guess: the fact is absent or it was asked.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use svod_dtype::{DeviceSpec, GpuArch};

use crate::atoms::Target;

/// A device's layer-0 description: what the driver and sysfs report, beside
/// the costs its probes measure. [`Facts::for_device`] queries it;
/// [`Facts::of`] carries what an offline target's tables know, `table` set.
#[derive(Clone, Debug)]
pub struct Facts {
    /// The backend's name for the device; an architecture's name where no
    /// backend reports one.
    pub name: String,
    pub arch: GpuArch,
    /// Compute units the device reports.
    pub sms: u32,
    pub wave: u32,
    /// Shared memory a block declares without launch attributes.
    pub smem_per_block: Option<u64>,
    /// Shared memory a block may opt into (CUDA's dynamic opt-in; AMD's LDS a
    /// work-group allocates): the wide-tile budget.
    pub smem_per_block_optin: Option<u64>,
    /// What the driver reserves per block out of the opt-in: a block uses
    /// `optin − this`.
    pub smem_reserved: Option<u64>,
    /// Shared memory the whole SM divides between its resident blocks.
    pub smem_per_sm: Option<u64>,
    /// 32-bit registers one block may use.
    pub registers_per_block: Option<u32>,
    /// 32-bit registers the SM divides between its resident threads.
    pub registers_per_sm: Option<u32>,
    /// Threads one SM keeps resident.
    pub threads_per_sm: Option<u32>,
    /// Blocks one SM keeps resident, whatever their size.
    pub blocks_per_sm: Option<u32>,
    /// L2 cache bytes (AMD: the sum over the KFD-listed L2 instances).
    pub l2_bytes: Option<u64>,
    /// The largest persisting-access window the L2 keeps.
    pub persisting_l2_bytes: Option<u64>,
    /// Peak core clock as the driver reports it, in kHz (CUDA's
    /// `CLOCK_RATE`; a diagnostic — rates come from measured costs, which a
    /// boost under load settles below the peak).
    pub core_clock_khz: Option<u64>,
    /// Peak memory clock in kHz (CUDA's `MEMORY_CLOCK_RATE`).
    pub memory_clock_khz: Option<u64>,
    /// AMD's KFD `max_engine_clk_fcompute`, the raw sysfs value (kernels
    /// print it in either kHz or MHz; a diagnostic, never derived from).
    pub engine_clk_fcompute: Option<u64>,
    /// Global memory bus, bits.
    pub bus_bits: Option<u32>,
    /// Copy engines (CUDA's `ASYNC_ENGINE_COUNT`, AMD's SDMA engines).
    pub copy_engines: Option<u32>,
    /// Dies a dispatch spreads blocks over (CDNA's XCDs).
    pub xcds: u32,
    /// Device memory bytes.
    pub memory_bytes: Option<u64>,
    /// The cache instances the KFD topology lists for the node, with their
    /// sharing maps (RDNA4's L2 and Infinity Cache, MI300X's per-XCD L2 and
    /// LLC). Empty where the vendor reports caches no other way (CUDA sums
    /// its one L2 into [`Self::l2_bytes`]).
    pub caches: Vec<CacheInstance>,
    /// Whether a live device answered these or the target's tables stand in.
    pub table: bool,
}

/// One cache instance the KFD topology lists: its level, size and which
/// processors share it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheInstance {
    /// 1 for the L1s and LDS, 2 for L2, 3 for the Infinity Cache (LLC).
    pub level: u32,
    pub size_bytes: u64,
    pub line_bytes: u32,
    /// The processors the instance is shared by (the set bits of KFD's
    /// `sibling_map`).
    pub shared_by: u32,
}

impl Facts {
    /// The default device's facts, queried once per device where its costs
    /// are measured ([`crate::ops::planner`]); `None` where the backend
    /// reports no architecture for it.
    pub fn for_device(spec: &DeviceSpec) -> Option<Self> {
        use svod_device::registry as reg;
        match spec {
            DeviceSpec::Cuda { device_id } => {
                let limits = reg::resolve_cuda_limits(*device_id).ok()?;
                let arch = GpuArch::Cuda(reg::resolve_cuda_arch(*device_id).ok()?);
                Some(Self {
                    name: reg::resolve_cuda_name(*device_id).ok()?,
                    arch,
                    sms: limits.sm_count,
                    wave: limits.warp_size,
                    smem_per_block: Some(limits.shared_per_block as u64),
                    smem_per_block_optin: Some(limits.shared_per_block_optin as u64),
                    smem_reserved: Some(limits.reserved_shared_per_block as u64),
                    smem_per_sm: Some(limits.shared_per_sm as u64),
                    registers_per_block: Some(limits.registers_per_block),
                    registers_per_sm: Some(limits.registers_per_sm),
                    threads_per_sm: Some(limits.max_threads_per_sm),
                    blocks_per_sm: Some(limits.max_blocks_per_sm),
                    l2_bytes: Some(limits.l2_cache_bytes as u64),
                    persisting_l2_bytes: Some(limits.persisting_l2_bytes as u64),
                    core_clock_khz: Some(limits.clock_rate_khz as u64),
                    memory_clock_khz: Some(limits.memory_clock_rate_khz as u64),
                    engine_clk_fcompute: None,
                    bus_bits: Some(limits.bus_width_bits),
                    copy_engines: Some(limits.async_engine_count),
                    xcds: 1,
                    memory_bytes: reg::resolve_cuda_memory(*device_id).ok().map(|b| b as u64),
                    caches: vec![],
                    table: false,
                })
            }
            DeviceSpec::Amd { device_id } => amd_of(*device_id),
            _ => None,
        }
    }

    /// What an offline target's tables carry: the fields its arch-keyed ISA
    /// tables answer, every queried field `None`, `table` set — the honest
    /// stand-in [`crate::atoms::Target::offline`] planners run on.
    pub fn of(target: &Target) -> Self {
        Self {
            name: target.arch.target_name().to_string(),
            arch: target.arch,
            sms: target.sms.unwrap_or(0),
            wave: target.wave,
            smem_per_block_optin: Some(target.smem_bytes as u64),
            xcds: target.xcds,
            table: true,
            ..Self::none()
        }
    }

    /// Every queried field absent.
    fn none() -> Self {
        Self {
            name: String::new(),
            arch: GpuArch::Cuda(svod_dtype::CudaArch::from_compute_capability(0, 0)),
            sms: 0,
            wave: 0,
            smem_per_block: None,
            smem_per_block_optin: None,
            smem_reserved: None,
            smem_per_sm: None,
            registers_per_block: None,
            registers_per_sm: None,
            threads_per_sm: None,
            blocks_per_sm: None,
            l2_bytes: None,
            persisting_l2_bytes: None,
            core_clock_khz: None,
            memory_clock_khz: None,
            engine_clk_fcompute: None,
            bus_bits: None,
            copy_engines: None,
            xcds: 1,
            memory_bytes: None,
            caches: vec![],
            table: false,
        }
    }
}

/// The node's facts from the KFD topology: its identity, occupancy geometry,
/// the caches it lists with their sharing, and its peak engine clock — no HIP,
/// no bus width (the KFD topology offers none). `None` where the node maps to
/// no supported architecture.
fn amd_of(device_id: usize) -> Option<Facts> {
    use svod_device::amd::topology;
    let node = topology::enumerate().into_iter().nth(device_id)?;
    let arch = svod_dtype::AmdArch::from_gfx_target_version(node.gfx_target_version)?;
    let caches = node
        .caches
        .iter()
        .map(|cache| CacheInstance {
            level: cache.level,
            size_bytes: cache.size_bytes,
            line_bytes: cache.line_bytes,
            shared_by: cache.shared_by(),
        })
        .collect::<Vec<_>>();
    // The KFD topology has no per-SM block limit and no reserved-LDS split;
    // its one LDS figure is the budget a work-group allocates from.
    let l2 = caches.iter().filter(|c| c.level == 2).map(|c| c.size_bytes).sum::<u64>();
    Some(Facts {
        name: GpuArch::Amd(arch).target_name().to_string(),
        arch: GpuArch::Amd(arch),
        sms: node.simd_count.checked_div(node.simd_per_cu).filter(|&sms| sms > 0).unwrap_or(0),
        wave: node.wave_front_size,
        smem_per_block: Some(node.lds_size_in_kb as u64 * 1024),
        smem_per_block_optin: Some(node.lds_size_in_kb as u64 * 1024),
        smem_reserved: None,
        smem_per_sm: None,
        registers_per_block: None,
        registers_per_sm: None,
        threads_per_sm: None,
        blocks_per_sm: None,
        l2_bytes: (l2 > 0).then_some(l2),
        persisting_l2_bytes: None,
        core_clock_khz: None,
        memory_clock_khz: None,
        engine_clk_fcompute: (node.max_engine_clk_fcompute > 0).then_some(node.max_engine_clk_fcompute),
        bus_bits: None,
        copy_engines: (node.num_sdma_engines > 0).then_some(node.num_sdma_engines),
        xcds: node.num_xcc.max(1),
        memory_bytes: (node.memory_bytes > 0).then_some(node.memory_bytes),
        caches,
        table: false,
    })
}

/// The facts of a device, queried once and kept for the process, beside its
/// planner: opening a device answers its limits once, and the sysfs walk an
/// AMD node costs is not worth repeating.
pub(crate) fn facts_of(spec: &DeviceSpec) -> Option<Arc<Facts>> {
    static FACTS: std::sync::LazyLock<Mutex<HashMap<DeviceSpec, Option<Arc<Facts>>>>> =
        std::sync::LazyLock::new(Mutex::default);
    let mut all = FACTS.lock().expect("facts cache");
    all.entry(spec.clone()).or_insert_with(|| Facts::for_device(spec).map(Arc::new)).clone()
}
