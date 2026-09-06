mod arch;
mod elementwise;
mod fa;
mod golden;
mod grid;
mod guide;
mod index;
mod kernel_probe;
mod kmeans;
mod knn;
mod layout;
mod loop_scope;
mod masked;
mod math;
mod matmul;
mod movement;
mod proptests;
mod reductions;
mod scaffold;
mod shuffle;
mod sq_attention;
mod swizzle;

/// The env-selected device's caps when tk defines its matrix-core fragment layouts
/// (AMD, CUDA sm_80+), else `None` — the skip gate for fragment-layout HW tests, so
/// a device without them skips instead of panicking at `Kernel::frag`.
pub(crate) fn fragment_device() -> Option<crate::ArchCaps> {
    let dev = svod_tensor::Tensor::rand(&[16, 16]).expect("probe tensor").device();
    crate::target::resolve_arch(&dev).map(crate::ArchCaps::for_arch).filter(crate::ArchCaps::has_matrix_core_layouts)
}

/// Whether the env-selected device is AMD CDNA (gfx942, wave64).
pub(crate) fn is_cdna_device() -> bool {
    fragment_device().and_then(|caps| caps.amd()).is_some_and(svod_dtype::AmdArch::is_cdna)
}

/// The env-selected device's caps when its fragment map folds a `Row` tile's
/// columns per lane row (CDNA's stride map, CUDA's `mma.sync`) — the layouts a
/// `row_reduce` over a `Row` tile is a per-row reduction on; RDNA's even/odd
/// accumulator folds rows instead, so it skips.
pub(crate) fn row_fold_device() -> Option<crate::ArchCaps> {
    fragment_device()
        .filter(|caps| caps.frag(crate::arch::FragRole::Accumulator).is_some_and(|f| f.map.folds_cols(false)))
}

/// Whether the env-selected device is in `archs` with its LLVM backend present —
/// the self-skip gate for the `#[ignore]`d HW tests of a kernel.
pub(crate) fn device_supported(archs: crate::ArchSet) -> bool {
    let spec = svod_tensor::Tensor::empty(&[1], svod_dtype::DType::Float32).device();
    crate::target::check_target(&spec, archs).is_ok()
}
