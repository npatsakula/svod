//! `ExecutionPlan::profile` static tier on a real device. Self-skips unless
//! the active default device is a Metal or CUDA GPU (the AMD tier is covered
//! by tk).

use svod_dtype::DType;
use svod_runtime::ProfileOptions;

use crate::{PrepareConfig, Tensor};

/// Every Metal kernel reports its pipeline limits, and the table shows them in
/// the resource columns with the register counts Metal cannot see left blank.
#[test]
fn metal_profile_reports_pipeline_limits() {
    let Some(config) = PrepareConfig::for_metal_if_available() else {
        eprintln!("skipped: default device is not a Metal GPU");
        return;
    };
    let mut a = Tensor::randn(&[128, 128]).unwrap().cast(DType::Float16).unwrap();
    let mut b = Tensor::randn(&[128, 128]).unwrap().cast(DType::Float16).unwrap();
    a.realize().unwrap();
    b.realize().unwrap();
    let mut c = a.matmul_with().other(&b).dtype(DType::Float32).call().unwrap();
    let plan = c.prepare_with(&config).unwrap();
    let report = plan.profile(&ProfileOptions::default()).unwrap();

    let kernels: Vec<_> = report.stages.iter().flat_map(|stage| &stage.kernels).collect();
    assert!(!kernels.is_empty());
    for kernel in &kernels {
        let resources = kernel.static_info.as_ref().and_then(|info| info.resources).expect("static resources");
        assert_eq!(resources.wave_size, 32, "{}", kernel.kernel.entry_point);
        assert!(resources.occupancy.is_some_and(|occ| occ > 0.0 && occ <= 1.0), "{}", kernel.kernel.entry_point);
        assert_eq!((resources.vgprs, resources.sgprs, resources.scratch_bytes), (None, None, None));
        assert!(kernel.gpu_start_ns.is_some(), "{} has GPU stamps", kernel.kernel.entry_point);
    }
    let table = report.render_table();
    for column in ["VGPR", "LDS", "occ%"] {
        assert!(table.contains(column), "{table}");
    }
}

/// Every CUDA kernel reports its function attributes (registers, shared and
/// local memory, warp width) with the scalar-register column NVIDIA does not
/// expose left blank, and its event-pair GPU stamps are populated and follow
/// dispatch order.
#[test]
fn cuda_profile_reports_function_attributes_and_gpu_stamps() {
    let Some(config) = PrepareConfig::for_cuda_if_available() else {
        eprintln!("skipped: default device is not a CUDA GPU");
        return;
    };
    let mut a = Tensor::randn(&[128, 128]).unwrap().cast(DType::Float16).unwrap();
    let mut b = Tensor::randn(&[128, 128]).unwrap().cast(DType::Float16).unwrap();
    a.realize().unwrap();
    b.realize().unwrap();
    let mut c = a.matmul_with().other(&b).dtype(DType::Float32).call().unwrap();
    let plan = c.prepare_with(&config).unwrap();
    let report = plan.profile(&ProfileOptions::default()).unwrap();

    let kernels: Vec<_> = report.stages.iter().flat_map(|stage| &stage.kernels).collect();
    assert!(!kernels.is_empty());
    let mut previous_end = 0;
    for kernel in &kernels {
        let name = &kernel.kernel.entry_point;
        let resources = kernel.static_info.as_ref().and_then(|info| info.resources).expect("static resources");
        assert_eq!(resources.wave_size, 32, "{name}");
        assert!(resources.vgprs.is_some_and(|regs| regs > 0), "{name}: {resources:?}");
        assert_eq!(resources.sgprs, None, "{name}");
        assert!(resources.scratch_bytes.is_some(), "{name}");
        assert!(resources.occupancy.is_some_and(|occ| occ > 0.0 && occ <= 1.0), "{name}: {resources:?}");
        let (start, end) = (kernel.gpu_start_ns.expect("GPU stamps"), kernel.gpu_end_ns.expect("GPU stamps"));
        assert!(start > 0 && end >= start, "{name}: {start}..{end}");
        assert!(start >= previous_end, "{name} retires in dispatch order: {start} < {previous_end}");
        previous_end = end;
    }
    let table = report.render_table();
    for column in ["VGPR", "SGPR", "LDS", "occ%"] {
        assert!(table.contains(column), "{table}");
    }
}
