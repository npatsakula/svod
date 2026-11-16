//! Static GPU resource + occupancy decoding from the AMD kernel descriptor.
//!
//! All values come from `COMPUTE_PGM_RSRC1`, `COMPUTE_PGM_RSRC3` (the
//! accumulator offset of gfx90a-class targets) and the descriptor's segment
//! sizes — no execution, no hardware access.

use svod_dtype::AmdArch;

use crate::profile::KernelResources;

// COMPUTE_PGM_RSRC1 bit fields (amdhsa, stable across GFX6–GFX12).
const RSRC1_VGPR_SHIFT: u32 = 0; // GRANULATED_WORKITEM_VGPR_COUNT, bits [5:0]
const RSRC1_VGPR_MASK: u32 = 0x3f;
const RSRC1_SGPR_SHIFT: u32 = 6; // GRANULATED_WAVEFRONT_SGPR_COUNT, bits [9:6]
const RSRC1_SGPR_MASK: u32 = 0xf;
// COMPUTE_PGM_RSRC3 on gfx90a-class targets (gfx90a, gfx942, gfx950):
// ACCUM_OFFSET, bits [5:0], the first AGPR of the unified file in units of 4.
const RSRC3_ACCUM_MASK: u32 = 0x3f;

/// Decode allocated VGPR/SGPR/LDS/scratch and, for architectures with known
/// register-file geometry, the VGPR-limited occupancy.
///
/// `rsrc1`/`rsrc3` are the original `compute_pgm_rsrc1`/`compute_pgm_rsrc3` from
/// the descriptor (NOT Svod's dispatch-patched copy). `lds_bytes`/`scratch_bytes`
/// are the descriptor's exact segment sizes. `wave_size` is 32 or 64.
pub fn decode_resources(
    rsrc1: u32,
    rsrc3: u32,
    lds_bytes: u32,
    scratch_bytes: u32,
    wave_size: u32,
    arch: AmdArch,
) -> KernelResources {
    // VGPRs are allocated/encoded in blocks: 8 per block in wave32, 4 in wave64
    // (GFX10+). Confirmed on RDNA3.5: field 24 → (24+1)*8 = 200 VGPRs. gfx9
    // wave64 allocates its unified file (vector and accumulator registers) in
    // blocks of 8, the accumulators starting at the rsrc3 offset (confirmed on
    // gfx942: `.amdhsa_next_free_vgpr 512`, `.amdhsa_accum_offset 256` encode
    // field 63 in both).
    let gfx9 = arch.gfx_major() == 9;
    let vgpr_block = if wave_size == 32 || gfx9 { 8 } else { 4 };
    let total = (((rsrc1 >> RSRC1_VGPR_SHIFT) & RSRC1_VGPR_MASK) + 1) * vgpr_block;
    let (vgprs, agprs) = if gfx9 {
        let accum = ((rsrc3 & RSRC3_ACCUM_MASK) + 1) * 4;
        (accum.min(total), Some(total - accum.min(total)))
    } else {
        (total, None)
    };
    // SGPRs are encoded in blocks of 16. Informational on GFX10+, where the HW
    // allocates a fixed pool, so this is reported but never an occupancy limiter.
    let sgprs = (((rsrc1 >> RSRC1_SGPR_SHIFT) & RSRC1_SGPR_MASK) + 1) * 16;
    let occupancy = vgpr_limited_occupancy(vgprs, wave_size, arch);
    KernelResources {
        vgprs: Some(vgprs),
        agprs,
        sgprs: Some(sgprs),
        lds_bytes,
        scratch_bytes: Some(scratch_bytes),
        wave_size,
        occupancy,
    }
}

/// VGPR-limited occupancy as a fraction of the SIMD's max resident waves, on the
/// [`Cu`] of a wave32 RDNA arch (`None` otherwise). This is the first-order
/// limiter for compute-bound kernels; LDS and workgroup limits are [`limits`]'.
pub(crate) fn vgpr_limited_occupancy(vgprs: u32, wave_size: u32, arch: AmdArch) -> Option<f32> {
    let cu = Cu::rdna(arch).filter(|_| wave_size == 32)?;
    let waves = (cu.vgprs / vgprs.max(1).next_multiple_of(cu.vgpr_granule)).min(cu.waves_per_simd);
    Some(waves as f32 / cu.waves_per_simd as f32)
}

/// What one compute unit holds at once, in CU mode (svod builds with `-mcumode`: a workgroup's waves
/// stay on one CU's two SIMDs and its LDS half).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Cu {
    pub simds: u32,
    /// Wave slots a SIMD schedules.
    pub waves_per_simd: u32,
    /// Wave32 VGPRs a SIMD's file holds, and the granule a wave allocates them in.
    pub vgprs: u32,
    pub vgpr_granule: u32,
    /// LDS bytes a CU's workgroups share, and the granule a workgroup allocates them in.
    pub lds_bytes: u32,
    pub lds_granule: u32,
    /// Barriers: a workgroup of more than one wave takes one.
    pub barriers: u32,
}

impl Cu {
    /// RDNA3/RDNA4 (gfx11/gfx12) in CU mode, wave32: LLVM's `getMaxWavesPerEU` (16 from gfx10.3),
    /// `getTotalNumVGPRs`/`getVGPRAllocGranule` (1536 in granules of 24 under `Feature1536VGPRs`:
    /// gfx1100, gfx1101, gfx1151, gfx12; 1024 in 16 elsewhere), `getLocalMemorySize` (a WGP's
    /// 128 KiB halved in CU mode), `getMaxWorkGroupsPerCU` (16 barriers), and the
    /// `COMPUTE_PGM_RSRC2.LDS_SIZE` granule of 128 dwords (AMDGPUUsage, GFX7–GFX12).
    pub fn rdna(arch: AmdArch) -> Option<Self> {
        use AmdArch::*;
        let (vgprs, vgpr_granule) = match arch {
            Gfx942 | Gfx950 => return None,
            Gfx1102 => (1024, 16),
            Gfx1100 | Gfx1101 | Gfx1151 | Gfx1200 | Gfx1201 => (1536, 24),
        };
        Some(Self {
            simds: 2,
            waves_per_simd: 16,
            vgprs,
            vgpr_granule,
            lds_bytes: 64 << 10,
            lds_granule: 512,
            barriers: 16,
        })
    }
}

/// The resource that stops a CU holding more of a kernel's workgroups.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Limiter {
    /// The SIMDs' register files.
    Vgpr,
    /// The CU's LDS.
    Lds,
    /// The SIMDs' wave slots.
    Waves,
    /// The CU's barriers.
    Workgroups,
}

impl Limiter {
    pub fn name(self) -> &'static str {
        match self {
            Self::Vgpr => "vgpr",
            Self::Lds => "lds",
            Self::Waves => "waves",
            Self::Workgroups => "workgroups",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Limit {
    pub workgroups_per_cu: u32,
    /// Waves a SIMD holds at once with the CU full of workgroups.
    pub waves_per_simd: u32,
    /// Every resource that allows no more, in the order VGPRs, LDS, barriers, wave slots: freeing
    /// one alone gains nothing while another stays.
    pub by: Vec<Limiter>,
}

/// How many workgroups of `workgroup_threads` threads a CU holds at once, and what stops more: `None`
/// where the resources leave the VGPRs unknown, the wave is not wave32 or the workgroup is empty.
/// A workgroup's waves split evenly over the CU's SIMDs.
pub fn limits(resources: &KernelResources, workgroup_threads: u32, cu: &Cu) -> Option<Limit> {
    if resources.wave_size != 32 || workgroup_threads == 0 {
        return None;
    }
    let waves = workgroup_threads.div_ceil(32);
    let alloc = resources.vgprs?.max(1).next_multiple_of(cu.vgpr_granule);
    let per_simd = (cu.vgprs / alloc).clamp(1, cu.waves_per_simd);
    let by_vgpr = (per_simd < cu.waves_per_simd).then_some(cu.simds * per_simd / waves);
    let lds = resources.lds_bytes.next_multiple_of(cu.lds_granule);
    let by_lds = (lds > 0).then(|| cu.lds_bytes / lds);
    let by_barriers = (waves > 1).then_some(cu.barriers);
    let by_waves = Some(cu.simds * cu.waves_per_simd / waves);
    let bounds = [
        (by_vgpr, Limiter::Vgpr),
        (by_lds, Limiter::Lds),
        (by_barriers, Limiter::Workgroups),
        (by_waves, Limiter::Waves),
    ];
    let bounds: Vec<(u32, Limiter)> = bounds.into_iter().filter_map(|(n, by)| Some((n?, by))).collect();
    let workgroups_per_cu = bounds.iter().map(|b| b.0).min().expect("the wave slots always bound");
    let by = bounds.into_iter().filter(|b| b.0 == workgroups_per_cu).map(|b| b.1).collect();
    let waves_per_simd = (workgroups_per_cu * waves).div_ceil(cu.simds).min(per_simd);
    Some(Limit { workgroups_per_cu, waves_per_simd, by })
}
