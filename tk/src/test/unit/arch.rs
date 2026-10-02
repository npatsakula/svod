//! The [`ArchCaps`] capability layer: the wave size / cross-lane reduce tree /
//! WMMA-fragment row stride are *derived from the detected `GpuArch`*, not
//! hand-set per call. gfx942 must reproduce the prior wave64 literals
//! bit-for-bit (the builders now thread these instead of the old constants);
//! gfx1151 (RDNA3.5, wave32) gets the correct control-path caps and is built for
//! by both matmul and FA (in `MATMUL_/FA_SUPPORTED_ARCHS`); its gfx11 WMMA fragment
//! layout is carried by the `RT_16X16_W32_*` tile shapes selected in the kernels,
//! while gfx12 (RDNA4) takes the single strided `RT_16X16_GFX12` for every role.
//! CUDA sm_80+ resolves every role to the two-half `mma.sync` fragment
//! (`RT_16X16_MMA`); pre-Ampere CUDA has no fragment table.

use svod_dtype::{AmdArch, CudaArch, DType, GpuArch};
use svod_schedule::optimizer::Renderer;
use test_case::test_case;

use crate::ArchCaps;
use crate::arch::FragRole;
use crate::arch::FragRole::{Accumulator, AccumulatorT, Operand, OperandB};
use crate::layout::ReduceTree;
use crate::tiles::{
    RT_16X16, RT_16X16_GFX12, RT_16X16_MMA, RT_16X16_W32_ACC, RT_16X16_W32_ACC_T, RT_16X16_W32_IN, ST_16X16,
    ST_16X16_MMA, ST_16X16_SWIZZLED, ST_16X16_SWIZZLED_W32,
};

const SM_86: GpuArch = GpuArch::Cuda(CudaArch::from_compute_capability(8, 6));

/// Behavior-preserving guard: the caps derived for gfx942 reproduce exactly the
/// wave64 literals the builders previously hardcoded (`64`, the `[16,32,48]`
/// `ds_bpermute` sibling tree, the `*4` FA fragment row stride). If the
/// derivation drifts, the gfx942 path is no longer bit-identical — fail loudly.
#[test]
fn gfx942_caps_reproduce_wave64_literals() {
    let c = ArchCaps::for_amd(AmdArch::Gfx942);
    assert_eq!(c, ArchCaps::GFX942, "GFX942 const == for_amd(Gfx942)");
    assert_eq!(c, ArchCaps::for_arch(GpuArch::Amd(AmdArch::Gfx942)), "for_amd is for_arch(Amd)");
    assert_eq!(c.wave_size, 64);
    let acc = c.frag(FragRole::Accumulator).expect("CDNA fragments");
    assert_eq!(
        acc.map.tree(c.wave_size),
        ReduceTree::Gather([16, 32, 48].into_iter().collect()),
        "prior ds_bpermute sibling tree"
    );
}

/// RDNA3.5 (gfx1151) control-path caps: wave32 (the warp/lane math + launch block),
/// the RDNA3.5 classification, and the `[16]` cross-lane reduce tree — correct for
/// the wave32 even/odd accumulator (a softmax row is the lane's 8 in-register
/// elements + its `L+16` sibling, so one `[16]` fold completes it). The RDNA WMMA
/// *fragment* layout (replicated inputs, even/odd accumulator) lives in the
/// `RT_16X16_W32_*` tile shapes, not in these scalar caps; the descriptor resolution
/// is covered by [`wmma_descriptor_resolves_per_detected_arch`].
#[test]
fn gfx1151_caps_are_wave32() {
    let c = ArchCaps::for_amd(AmdArch::Gfx1151);
    assert_eq!(c.wave_size, 32);
    let acc = c.frag(FragRole::Accumulator).expect("RDNA fragments");
    assert_eq!(
        acc.map.tree(c.wave_size),
        ReduceTree::Gather([16].into_iter().collect()),
        "wave32 even/odd accumulator folds one sibling at L+16"
    );
    let amd = c.amd().expect("AMD arch");
    assert!(amd.is_rdna3_5() && !amd.is_rdna3(), "gfx1151 is RDNA3.5, distinct from RDNA3");
    assert_eq!(c.cuda(), None);
}

/// CUDA sm_86 gets the warp32 control path — the same lane math as gfx1151 — and
/// the `mma.sync` fragment table: every role is the two-half [`RT_16X16_MMA`] (an
/// accumulator IS an A operand, so it is reusable as an input), both LDS strips are
/// the swizzled [`ST_16X16_MMA`], and the arch accessors classify it.
#[test]
fn cuda_sm86_caps_resolve_mma_sync_fragments() {
    let c = ArchCaps::for_arch(SM_86);
    assert_eq!(c.wave_size, 32);
    assert_eq!(c.amd(), None);
    assert_eq!(c.cuda(), Some(CudaArch::from_compute_capability(8, 6)));
    assert!(c.has_matrix_core_layouts());
    for role in [Accumulator, Operand, AccumulatorT] {
        assert_eq!(c.frag(role), Some(RT_16X16_MMA), "{role:?}");
    }
    assert_eq!(c.shared_default(), Some(ST_16X16_MMA));
    assert_eq!(c.shared_swizzled(), Some(ST_16X16_MMA));
    assert!(c.acc_reusable_as_input(), "the two-half f32 accumulator is the A-fragment register order");
}

/// Apple7+ resolves to the 8x8 `simdgroup_matrix` fragment, in two orientations.
/// Apple's core reads its operands straight off the fragment map, so tk's `Col`
/// accumulator (holding `Cᵀ`, as on every other arch) is reached by emitting
/// `Cᵀ = Bᵀ·Aᵀ`: the B operand's own `Col` declaration already transposes it under
/// the plain map, while the `Row`-declared **A** operand needs the map flipped.
/// Accumulators — plain and transposed-store alike — keep the plain map, which is
/// what the store, the mask, the RV broadcast and the softmax reduce all assume.
/// Each orientation is pinned by a hardware rung (`matmul_k_addressing`,
/// `fa_output_store_contract`).
/// The LDS strip is unswizzled (an 8-column 16-bit row is already one 16-byte chunk).
#[test]
fn caps_metal_apple7() {
    let c = ArchCaps::for_arch(GpuArch::Metal(svod_dtype::MetalFamily::Apple(9)));
    assert_eq!(c.wave_size, 32);
    assert!(c.has_matrix_core_layouts());
    for role in [Accumulator, AccumulatorT, OperandB] {
        assert_eq!(c.frag(role), Some(crate::tiles::RT_8X8_SIMD), "{role:?}");
    }
    assert_eq!(c.frag(Operand), Some(crate::tiles::RT_8X8_SIMD_T));
    assert_eq!(c.shared_default(), Some(crate::tiles::ST_8X8));
    assert_eq!(c.shared_swizzled(), Some(crate::tiles::ST_8X8));
    assert!(c.acc_reusable_as_input(), "one simdgroup_matrix map serves operand and accumulator");
}

/// Pre-Ampere CUDA (sm_75: f16 `m16n8k8` only, no K=16 f16/bf16 core) and Apple
/// GPUs below family 7 (no `simdgroup_matrix`) have no fragment table: every
/// resolver is `None`, so an MMA kernel fails loudly.
#[test_case(GpuArch::Cuda(CudaArch::from_compute_capability(7, 5)); "sm_75")]
#[test_case(GpuArch::Metal(svod_dtype::MetalFamily::Apple(6)); "apple6")]
#[test_case(GpuArch::Metal(svod_dtype::MetalFamily::Mac2); "mac2")]
fn caps_without_fragment_layouts(arch: GpuArch) {
    let c = ArchCaps::for_arch(arch);
    assert_eq!(c.wave_size, 32);
    assert!(!c.has_matrix_core_layouts());
    for role in [Accumulator, Operand, OperandB, AccumulatorT] {
        assert_eq!(c.frag(role), None, "{role:?}");
    }
    assert_eq!(c.shared_default(), None);
    assert_eq!(c.shared_swizzled(), None);
}

/// The WMMA descriptor is sourced from the shared `TensorCore` table *by the
/// detected arch* (`group::wmma_desc` looks up `Renderer::for_amd_arch(caps.arch)`),
/// so it tracks the GPU in use — not a hand-built descriptor. Confirm the
/// 16×16×16 f16 core resolves with the arch's wave thread count on both the
/// validated CDNA3 path (64) and the deferred RDNA3.5 path (32), that gfx1201
/// (RDNA4) carries the un-replicated `(8,8,8)` widths tk's [`RT_16X16_GFX12`] is
/// sized to, and that CUDA exposes the rectangular `m16n8k16` `(8,16,16)` core with
/// `(8,4,4)` elements per lane (the shape `group::mma` plans two halves over)
/// instead of a square one.
#[test]
fn wmma_descriptor_resolves_per_detected_arch() {
    let core = |ren: Renderer, dims| {
        ren.tensor_cores
            .into_iter()
            .find(|tc| tc.dtype_in == DType::Float16 && tc.dtype_out == DType::Float32 && tc.dims == dims)
            .map(|tc| (tc.threads, tc.elements_per_thread))
    };
    assert_eq!(core(Renderer::for_amd_arch(AmdArch::Gfx942), (16, 16, 16)), Some((64, (4, 4, 4))), "gfx942 MFMA");
    assert_eq!(core(Renderer::for_amd_arch(AmdArch::Gfx1151), (16, 16, 16)), Some((32, (16, 16, 8))), "gfx1151 WMMA");
    assert_eq!(core(Renderer::for_amd_arch(AmdArch::Gfx1201), (16, 16, 16)), Some((32, (8, 8, 8))), "gfx1201 WMMA");
    let sm86 = CudaArch::from_compute_capability(8, 6);
    assert_eq!(core(Renderer::for_cuda_arch(sm86), (16, 16, 16)), None, "sm_86 has no square core");
    assert_eq!(core(Renderer::for_cuda_arch(sm86), (8, 16, 16)), Some((32, (8, 4, 4))), "sm_86 m16n8k16");
}

/// Behavior-preserving guard for the fragment-role resolver (Gap 2): the logical
/// [`FragRole`]s resolve to the *exact* physical fragment constants the kernels
/// previously selected by hand, on both arches — so the refactor is bit-identical on
/// gfx942 and render-identical on gfx1151. If a mapping drifts, the kernels' generated
/// IR changes silently; fail loudly here instead.
#[test]
fn frag_roles_resolve_to_canonical_constants() {
    // gfx942 (CDNA MFMA, wave64): acc == input fragment ⇒ every role is RT_16X16.
    let c = ArchCaps::for_amd(AmdArch::Gfx942);
    assert!(c.has_matrix_core_layouts());
    assert_eq!(c.frag(Accumulator), Some(RT_16X16));
    assert_eq!(c.frag(Operand), Some(RT_16X16));
    assert_eq!(c.frag(AccumulatorT), Some(RT_16X16));
    assert_eq!(c.shared_default(), Some(ST_16X16));
    assert_eq!(c.shared_swizzled(), Some(ST_16X16_SWIZZLED));
    assert!(c.acc_reusable_as_input(), "CDNA acc fragment is reusable as a WMMA input");

    // gfx1151 (RDNA3.5 WMMA, wave32): distinct even/odd-acc / replicated-input /
    // transposed-acc fragments; the acc→input handoff must round-trip through LDS.
    let r = ArchCaps::for_amd(AmdArch::Gfx1151);
    assert!(r.has_matrix_core_layouts());
    assert_eq!(r.frag(Accumulator), Some(RT_16X16_W32_ACC));
    assert_eq!(r.frag(Operand), Some(RT_16X16_W32_IN));
    assert_eq!(r.frag(AccumulatorT), Some(RT_16X16_W32_ACC_T));
    assert_eq!(r.shared_default(), Some(ST_16X16_SWIZZLED_W32));
    assert_eq!(r.shared_swizzled(), Some(ST_16X16_SWIZZLED_W32));
    assert!(!r.acc_reusable_as_input(), "gfx11 acc/input fragments differ ⇒ LDS relayout");
}

/// RDNA4 (gfx1200/gfx1201, wave32) resolves every role to the single strided 8/lane
/// [`RT_16X16_GFX12`] — gfx12 drops RDNA3's wave-half replication and its even/odd
/// accumulator, so operand, B operand, accumulator and the N-major `AccumulatorT`
/// store are one fragment, as on CDNA. Hardware-verified on gfx1201; a regression
/// back onto the `RT_16X16_W32_*` shapes is what made the matmul compute garbage.
/// The *swizzled* strip is gfx12's own: its gather reads one 16-byte chunk of a
/// row per lane, so the chunk-granular [`ST_16X16_MMA`] keeps the fragment one
/// `ds_read_b128` where gfx11's 8-byte-granular XOR would split it in two. The
/// plain strip (flash attention, which does not swizzle) stays the gfx11 one.
/// Sharing one fragment across the roles is also what licenses the register
/// acc→input handoff (hardware-verified on gfx1201), which drops FA's per-warp
/// LDS relayout band.
#[test_case(AmdArch::Gfx1200; "gfx1200")]
#[test_case(AmdArch::Gfx1201; "gfx1201")]
fn rdna4_caps_resolve_gfx12_fragments(arch: AmdArch) {
    let c = ArchCaps::for_amd(arch);
    assert_eq!(c.wave_size, 32);
    assert!(c.has_matrix_core_layouts());
    for role in [Accumulator, Operand, OperandB, AccumulatorT] {
        assert_eq!(c.frag(role), Some(RT_16X16_GFX12), "{role:?}");
    }
    assert_eq!(c.shared_default(), Some(ST_16X16_SWIZZLED_W32));
    assert_eq!(c.shared_swizzled(), Some(ST_16X16_MMA));
    assert!(c.shared_swizzled().is_some_and(|st| st.swizzle.keeps_16b_chunks()), "gfx12 gathers 16-byte chunks");
    assert!(c.acc_reusable_as_input(), "one gfx12 fragment for both roles ⇒ a register copy");
}

/// The scheduling cap: only gfx12 needs the explicit fence that keeps a
/// pipelined trip's LDS commit after its MMAs. The GEMM asks for the property,
/// so a new arch joins by answering here — [`staged_gemm_rdna4_fences_the_commit`]
/// pins what the answer does to the kernel.
#[test_case(GpuArch::Amd(AmdArch::Gfx1201), true; "gfx1201")]
#[test_case(GpuArch::Amd(AmdArch::Gfx1200), true; "gfx1200")]
#[test_case(GpuArch::Amd(AmdArch::Gfx1151), false; "gfx1151")]
#[test_case(GpuArch::Amd(AmdArch::Gfx942), false; "gfx942")]
#[test_case(SM_86, false; "sm_86")]
#[test_case(GpuArch::Metal(svod_dtype::MetalFamily::Apple(9)), false; "metal")]
fn pipeline_commit_fence_is_gfx12_only(arch: GpuArch, want: bool) {
    assert_eq!(ArchCaps::for_arch(arch).needs_pipeline_commit_fence(), want);
}

/// The async-copy cap: only CUDA has `cp.async`. The conv rewrites whose fills
/// have no register-staged form are offered on this answer alone, and the tuner
/// *builds every candidate it is offered* to fingerprint it — so a wrong answer
/// here is a panic in `cp_async_fill`, not a slower kernel. `every_plan_builds`
/// in the conv tests pins the other half of that contract.
#[test_case(SM_86, true; "sm_86")]
#[test_case(GpuArch::Amd(AmdArch::Gfx1201), false; "gfx1201")]
#[test_case(GpuArch::Amd(AmdArch::Gfx1200), false; "gfx1200")]
#[test_case(GpuArch::Amd(AmdArch::Gfx1151), false; "gfx1151")]
#[test_case(GpuArch::Amd(AmdArch::Gfx942), false; "gfx942")]
#[test_case(GpuArch::Metal(svod_dtype::MetalFamily::Apple(9)), false; "metal")]
fn async_copy_is_cuda_only(arch: GpuArch, want: bool) {
    assert_eq!(ArchCaps::for_arch(arch).has_async_copy(), want);
}

/// Every kernel bar the gfx942-only direct-launch FA wrapper admits the RDNA4
/// parts; the RDNA-only sets (norm, NT gemm) admit them without admitting CDNA.
#[test]
fn rdna4_is_in_the_shared_arch_lists() {
    use crate::target::{CDNA_RDNA_WMMA, RDNA_WMMA};

    for arch in [AmdArch::Gfx1200, AmdArch::Gfx1201] {
        let gpu = GpuArch::Amd(arch);
        assert!(crate::ArchSet::amd(RDNA_WMMA).supports(gpu), "{arch:?} in RDNA_WMMA");
        assert!(crate::ArchSet::amd(CDNA_RDNA_WMMA).supports(gpu), "{arch:?} in CDNA_RDNA_WMMA");
    }
    assert!(!crate::ArchSet::amd(RDNA_WMMA).supports(GpuArch::Amd(AmdArch::Gfx942)));
    assert!(crate::ArchSet::amd(CDNA_RDNA_WMMA).supports(GpuArch::Amd(AmdArch::Gfx942)));
}

/// [`ArchSet`](crate::ArchSet) membership: the AMD list is exact, the CUDA floor is
/// open-ended (`sm_80` admits sm_86/sm_90, rejects sm_75), an AMD-only set rejects
/// every CUDA arch, and Metal is never admitted.
#[test_case(GpuArch::Amd(AmdArch::Gfx942), true, true; "gfx942")]
#[test_case(GpuArch::Amd(AmdArch::Gfx950), false, false; "gfx950 absent from the AMD list")]
#[test_case(GpuArch::Cuda(CudaArch::from_compute_capability(7, 5)), false, false; "sm_75 below the floor")]
#[test_case(GpuArch::Cuda(CudaArch::from_compute_capability(8, 0)), false, true; "sm_80 at the floor")]
#[test_case(SM_86, false, true; "sm_86")]
#[test_case(GpuArch::Cuda(CudaArch::from_compute_capability(9, 0)), false, true; "sm_90 above the floor")]
#[test_case(GpuArch::Metal(svod_dtype::MetalFamily::Apple(9)), false, false; "metal")]
fn arch_set_membership(arch: GpuArch, amd_only: bool, with_cuda: bool) {
    let amd = crate::ArchSet::amd(&[AmdArch::Gfx942, AmdArch::Gfx1151]);
    let cuda = amd.with_cuda_from(CudaArch::from_compute_capability(8, 0));
    assert_eq!(amd.supports(arch), amd_only, "AMD-only set");
    assert_eq!(cuda.supports(arch), with_cuda, "AMD + sm_80 set");
    assert_eq!(cuda.to_string(), "AMD [Gfx942, Gfx1151] + CUDA sm_80+");
}
