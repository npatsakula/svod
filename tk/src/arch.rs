//! Arch-derived capability bundle for the tile DSL.
//!
//! `svod-tk` kernels are built for a specific GPU arch: the wave width, the
//! cross-lane reduce tree, and the per-lane matrix-core fragment layouts are all
//! arch properties, as is the WMMA descriptor itself ([`crate::group`] looks it
//! up from the shared `TensorCore` table by [`ArchCaps::arch`]). [`ArchCaps`] is
//! the single place those are derived from a [`GpuArch`], so the builders thread
//! one value instead of hardcoding gfx942 (wave64) literals.
//!
//! Two layers of support, resolved per arch:
//!
//! - **Control path** — [`ArchCaps::wave_size`] (warp/lane math, launch block).
//!   Defined for every arch; the shuffle-only kernels (single-query attention)
//!   need nothing else.
//! - **Matrix-core fragment layouts** — [`ArchCaps::frag`] / [`ArchCaps::shared_default`]
//!   / [`ArchCaps::shared_swizzled`], the single arch→fragment table kernels stay
//!   arch-blind through (no `is_cdna()` shape branches). CDNA's MFMA accumulator
//!   and input fragments share the wave64 [`crate::tiles::RT_16X16`] layout; RDNA
//!   (gfx11 WMMA, wave32) carries `ept=(16,16,8)`, inputs replicated across the two
//!   wave-halves, and an even/odd-interleaved `<8×float>` accumulator (the
//!   `RT_16X16_W32_*` shapes); CUDA sm_80+ (`mma.sync m16n8k16`, warp32) holds a
//!   16×16 tile as two m16n8 halves ([`crate::layout::LaneMap::MmaSync`], 8/lane
//!   for inputs and accumulator alike — [`crate::tiles::RT_16X16_MMA`]). Unresolved
//!   (`None`) on Metal and pre-Ampere CUDA, so an MMA kernel fails loudly at
//!   fragment resolution instead of rendering a wrong layout.
//!
//! gfx942 is the validated/calibrated target — the register-tile fragment-layout
//! tables ([`crate::tiles`] strides and `group::mma`'s per-lane upcast counts) and
//! the [`crate::WARP_THREADS`] layout-table constant are pinned to it.
//! [`ArchCaps::frag_row_stride`] is the one remaining CDNA-only datum (the legacy
//! direct-launch FA mask — the production rolled-db kernel derives its mask from
//! the accumulator's own lane map instead).

use smallvec::SmallVec;
use svod_dtype::{AmdArch, CudaArch, GpuArch};

use crate::tiles::{
    RT_16X16, RT_16X16_MMA, RT_16X16_W32_ACC, RT_16X16_W32_ACC_T, RT_16X16_W32_IN, RTBaseShape, ST_16X16, ST_16X16_MMA,
    ST_16X16_SWIZZLED, ST_16X16_SWIZZLED_W32, STBaseShape,
};

/// Logical role of a 16×16 matrix-core fragment, independent of arch packing.
/// Resolved to a physical [`RTBaseShape`] by [`ArchCaps::frag`] — kernels select a
/// fragment by *role*, never by naming a per-arch constant.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FragRole {
    /// f32 MMA output / online-softmax accumulator.
    Accumulator,
    /// f16/bf16 WMMA input operand (A or B).
    Operand,
    /// Accumulator transposed for an N-major store (e.g. the FA output `O[q,d]`
    /// from the `[d,q]` PV accumulator).
    AccumulatorT,
}

/// Lanes per wave on `arch`: 64 on CDNA, 32 on RDNA3/4, 32 on every CUDA
/// generation, 32 on Apple GPUs (the SIMD-group width).
pub const fn wave_size_of(arch: GpuArch) -> usize {
    match arch {
        GpuArch::Amd(arch) => arch.wave_size() as usize,
        GpuArch::Cuda(arch) => arch.wave_size() as usize,
        GpuArch::Metal(_) => 32,
    }
}

/// The arch-derived constants the tile builders thread instead of the wave64
/// literals. `Copy`; [`Self::for_arch`] is `const`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArchCaps {
    /// The target GPU arch — drives the matrix-core descriptor lookup
    /// (`Renderer::for_{amd,cuda}_arch`) in [`crate::group`] and the cross-lane
    /// shuffle lowering (`ds_bpermute` on AMD, `shfl.sync` on CUDA).
    pub arch: GpuArch,
    /// Lanes per wave. `threadIdx` splits into warp = `idx / wave_size` and lane
    /// = `idx % wave_size`; the launch block is `warps * wave_size`.
    pub wave_size: usize,
}

impl ArchCaps {
    /// Derive the caps from `arch` (wave size from [`wave_size_of`]).
    pub const fn for_arch(arch: GpuArch) -> Self {
        Self { arch, wave_size: wave_size_of(arch) }
    }

    /// [`Self::for_arch`] for an AMD arch.
    pub const fn for_amd(arch: AmdArch) -> Self {
        Self::for_arch(GpuArch::Amd(arch))
    }

    /// The validated default target: gfx942 (CDNA3, wave64).
    pub const GFX942: ArchCaps = ArchCaps::for_amd(AmdArch::Gfx942);

    /// The AMD arch when this is an AMD target.
    pub fn amd(&self) -> Option<AmdArch> {
        self.arch.amd()
    }

    /// The CUDA compute capability when this is a CUDA target.
    pub fn cuda(&self) -> Option<CudaArch> {
        self.arch.cuda()
    }

    /// Whether the arch is AMD CDNA (MFMA, wave64) — the only arch whose
    /// accumulator and input fragments share one layout.
    fn is_cdna(&self) -> bool {
        self.amd().is_some_and(AmdArch::is_cdna)
    }

    /// Whether tk defines matrix-core fragment layouts for this arch (so
    /// [`Self::frag`] and the shared-tile strips resolve): AMD, and CUDA from
    /// Ampere (the f16/bf16 `m16n8k16` floor).
    pub fn has_matrix_core_layouts(&self) -> bool {
        self.amd().is_some() || self.cuda().is_some_and(CudaArch::has_bf16_mma)
    }

    /// The AMD sibling-gather reduce-tree offsets: a lane folds the partials of the
    /// `wave_size / 16` sibling row-groups, each one WMMA-column span (16 lanes)
    /// apart → wave64 `[16, 32, 48]`, wave32 `[16]`. Correct for **both** the CDNA
    /// MFMA layout *and* the RDNA even/odd accumulator: at wave32 a softmax row
    /// (16 KV) is split across a lane's 8 in-register elements (the even/odd half)
    /// and its sibling lane `L+16` (the other half). The reductions themselves read
    /// the tree off the fragment's [`crate::layout::LaneMap::tree`] (the quad
    /// butterfly on CUDA); this is the AMD form for the graph-shape tests.
    pub fn reduce_tree(&self) -> SmallVec<[i64; 3]> {
        (1..self.wave_size as i64 / 16).map(|i| i * 16).collect()
    }

    /// Per-lane row stride of a 16×16 **CDNA MFMA** accumulator fragment
    /// (`256 / wave_size`): wave64 → 4. Used only by the legacy direct-launch FA
    /// builders' causal/padding mask (which map each `laneid / 16` row-group to KV
    /// rows with this contiguous stride). The production rolled-db FA derives its
    /// mask from the att accumulator's own `lane_rc` instead — arch-correct for both
    /// the CDNA stride and the RDNA even/odd interleave — so it does not call this.
    pub const fn frag_row_stride(&self) -> i64 {
        (16 * 16 / self.wave_size) as i64
    }

    /// Physical register fragment for a logical [`FragRole`] on this arch — the
    /// single arch→fragment table the kernels resolve through. CDNA's MFMA
    /// accumulator and input fragments share a layout, so every role resolves to
    /// [`RT_16X16`]; RDNA (gfx11 WMMA) splits into the even/odd-interleaved
    /// accumulator, the replicated input, and the transposed accumulator; CUDA
    /// sm_80+ resolves every role to the two-half [`RT_16X16_MMA`] (an accumulator
    /// IS the A-operand register order, and the transposed store is the `Col`
    /// reading of the same map). `None` where tk has no fragment table (Metal,
    /// pre-Ampere CUDA) — see the module docs.
    pub fn frag(&self, role: FragRole) -> Option<RTBaseShape> {
        if !self.has_matrix_core_layouts() {
            return None;
        }
        Some(match self.amd() {
            None => RT_16X16_MMA,
            Some(amd) if amd.is_cdna() => RT_16X16,
            Some(_) => match role {
                FragRole::Accumulator => RT_16X16_W32_ACC,
                FragRole::Operand => RT_16X16_W32_IN,
                FragRole::AccumulatorT => RT_16X16_W32_ACC_T,
            },
        })
    }

    /// The canonical LDS strip fragment: plain on CDNA. On RDNA and CUDA the only
    /// ept-8 strip defined is swizzled, so it coincides with [`Self::shared_swizzled`].
    /// Used by kernels whose LDS access does not itself need the XOR swizzle
    /// (flash-attention). `None` where [`Self::frag`] is.
    pub fn shared_default(&self) -> Option<STBaseShape> {
        self.shared_strip(false)
    }

    /// The XOR-swizzled LDS strip fragment, for kernels that swizzle to avoid LDS
    /// bank conflicts (the matmul A/B strips). `None` where [`Self::frag`] is.
    pub fn shared_swizzled(&self) -> Option<STBaseShape> {
        self.shared_strip(true)
    }

    fn shared_strip(&self, swizzled: bool) -> Option<STBaseShape> {
        if !self.has_matrix_core_layouts() {
            return None;
        }
        Some(match self.amd() {
            None => ST_16X16_MMA,
            Some(amd) if amd.is_cdna() => {
                if swizzled {
                    ST_16X16_SWIZZLED
                } else {
                    ST_16X16
                }
            }
            Some(_) => ST_16X16_SWIZZLED_W32,
        })
    }

    /// Whether an MMA accumulator fragment can be reused directly as a WMMA input via
    /// a register copy. True on CDNA (MFMA acc == input fragment) and CUDA (the
    /// two-half 16×16 f32 accumulator holds the m16n8 C fragments in exactly the A
    /// fragment's register order — ThunderKittens `mma_AB(o, att_bf, v)`); false on
    /// RDNA (the even/odd `<8×f32>` accumulator and the replicated `<16×in>` input
    /// differ), where the acc→input handoff must round-trip through LDS instead.
    pub fn acc_reusable_as_input(&self) -> bool {
        self.is_cdna() || self.cuda().is_some()
    }
}
