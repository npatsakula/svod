mod arch;
mod conv;
mod elementwise;
mod fa;
mod gemm;
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
mod norm;
mod proptests;
mod reductions;
mod scaffold;
mod shuffle;
mod sq_attention;
mod swizzle;
mod tiling;
mod tune;

/// The env-selected device's caps when tk defines its matrix-core fragment layouts
/// (AMD, CUDA sm_80+), else `None` — the skip gate for fragment-layout HW tests, so
/// a device without them skips instead of panicking at `Kernel::frag`.
pub(crate) fn fragment_device() -> Option<crate::ArchCaps> {
    crate::tune::set_enabled(false);
    let dev = svod_tensor::Tensor::rand(&[16, 16]).expect("probe tensor").device();
    crate::target::resolve_arch(&dev).map(crate::ArchCaps::for_arch).filter(crate::ArchCaps::has_matrix_core_layouts)
}

/// The env-selected device's caps when its wave is 32 lanes — the gate for the
/// wave32 fragment-map hardware tests (gfx11, gfx12, CUDA, Metal alike).
pub(crate) fn wave32_fragment_device() -> Option<crate::ArchCaps> {
    fragment_device().filter(|caps| caps.wave_size == 32)
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
    // A kernel test checks numerics against the graph; tuning every shape it
    // touches would multiply its GPU time for nothing it asserts.
    crate::tune::set_enabled(false);
    let spec = svod_tensor::Tensor::empty(&[1], svod_dtype::DType::Float32).device();
    crate::target::check_target(&spec, archs).is_ok()
}

/// The largest `|got - want|`, infinite where the two disagree on finiteness
/// (`allclose_f32`'s rule). A plain `f32::max` fold drops a NaN operand, so a NaN
/// output would pass any tolerance.
pub(crate) fn max_abs_err(got: &[f32], want: &[f32]) -> f32 {
    svod_tensor::testing::allclose_f32(got, want, 0.0, 0.0).max_abs_err
}

/// [`max_abs_err`] relative to the reference's largest finite magnitude, the
/// scale a narrow output's rounding is measured against.
pub(crate) fn rel_err(got: &[f32], want: &[f32]) -> f32 {
    let scale = want.iter().filter(|w| w.is_finite()).fold(0f32, |a, w| a.max(w.abs())).max(f32::MIN_POSITIVE);
    max_abs_err(got, want) / scale
}

/// The instructions a tk kernel compiles to on `arch`, in program order: the
/// optimizer and the target-graph lowering `launch_custom` runs, with no device.
pub(crate) fn lowered_program(
    sink: std::sync::Arc<svod_ir::UOp>,
    arch: svod_dtype::GpuArch,
) -> Vec<std::sync::Arc<svod_ir::UOp>> {
    use svod_codegen::traits::Renderer;
    use svod_dtype::GpuArch;

    let optimizer = match arch {
        GpuArch::Cuda(cuda) => svod_schedule::OptimizerRenderer::for_cuda_arch(cuda).with_rewrite_capabilities(
            svod_ir::RendererOps::all(),
            svod_codegen::llvm::LlvmTextRenderer::nvptx(cuda).decompositor(),
            Some(svod_codegen::llvm::nvptx_extra_matcher()),
        ),
        GpuArch::Amd(amd) => svod_schedule::OptimizerRenderer::for_amd_arch(amd).with_rewrite_capabilities(
            svod_ir::RendererOps::all(),
            svod_codegen::llvm::LlvmTextRenderer::amd(amd).decompositor(),
            Some(svod_codegen::llvm::amd_extra_matcher()),
        ),
        other => panic!("no host lowering for {other:?}"),
    };
    let optimized =
        svod_schedule::optimize_kernel_with_config(sink, &optimizer, &svod_schedule::OptimizerConfig::default())
            .expect("optimize");
    let program = svod_codegen::program_pipeline::program_from_sink(optimized, svod_dtype::DeviceSpec::Cpu)
        .expect("final target graph");
    let linearized = svod_codegen::program_pipeline::do_linearize(&program).expect("do_linearize");
    let linear = linearized.toposort().into_iter().find(|u| matches!(u.op(), svod_ir::Op::Linear(..))).expect("LINEAR");
    let svod_ir::Op::Linear(svod_ir::ops::Linear { ops }) = linear.op() else { unreachable!() };
    ops.to_vec()
}

/// The accumulator reads in `program` that sit outside a loop their WMMA runs
/// in. Such a read sees the tile as it stood before that loop, so the loop's
/// steps overwrite the sum instead of adding to it. Empty for a sound kernel.
pub(crate) fn escaped_accumulator_reads(program: &[std::sync::Arc<svod_ir::UOp>]) -> Vec<String> {
    use std::collections::HashMap;
    use svod_ir::{AddrSpace, Op, ops};

    // The loops open at each instruction, by id.
    let mut open: Vec<u64> = Vec::new();
    let mut scope: HashMap<u64, Vec<u64>> = HashMap::new();
    for op in program {
        if let Op::End(ops::End { ranges, .. }) = op.op() {
            open.retain(|id| ranges.iter().all(|range| range.id != *id));
        }
        scope.insert(op.id, open.clone());
        if matches!(op.op(), Op::Range(..)) {
            open.push(op.id);
        }
    }
    let reads_a_register = |load: &svod_ir::UOp| {
        let Op::Load(ops::Load { index, .. }) = load.op() else { return false };
        let mut buffer = index.clone();
        loop {
            buffer = match buffer.op() {
                Op::Index(ops::Index { buffer, .. }) => buffer.clone(),
                Op::Shrink(ops::Shrink { src, .. }) | Op::Cast(ops::Cast { src, .. }) => src.clone(),
                Op::After(ops::After { passthrough, .. }) => passthrough.clone(),
                _ => return buffer.addrspace() == Some(AddrSpace::Reg),
            };
        }
    };

    let mut escaped = Vec::new();
    for wmma in program {
        let Op::Wmma(ops::Wmma { c, .. }) = wmma.op() else { continue };
        let loops = &scope[&wmma.id];
        // The fragment's lane reads: walk the accumulator operand up to the loads,
        // never past another WMMA or a store into an earlier step's chain.
        let mut stack = vec![c.clone()];
        while let Some(node) = stack.pop() {
            if reads_a_register(&node) {
                let at = scope.get(&node.id).map_or(&[][..], Vec::as_slice);
                let outside: Vec<u64> = loops.iter().filter(|id| !at.contains(id)).copied().collect();
                if !outside.is_empty() {
                    escaped.push(format!(
                        "load {} reads the accumulator outside loops {outside:?} of wmma {}",
                        node.id, wmma.id
                    ));
                }
            } else if !matches!(node.op(), Op::Wmma(..) | Op::Store(..) | Op::Range(..) | Op::End(..)) {
                stack.extend(node.op().sources());
            }
        }
    }
    escaped
}
