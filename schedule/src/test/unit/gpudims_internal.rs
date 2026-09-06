use super::*;
use svod_ir::ops;
use test_case::test_case;

/// Build a Vec<Arc<UOp>> of concrete `index_const` dims for tests that only
/// exercise the numeric grouping/splitting logic.
fn d(vals: &[usize]) -> Vec<Arc<UOp>> {
    vals.iter().map(|&v| UOp::index_const(v as i64)).collect()
}

/// Extract dim_max from a slice — round-trips numeric-only test inputs back
/// through the sint abstraction.
fn dmax(vs: &[Arc<UOp>]) -> Vec<usize> {
    vs.iter().map(dim_max).collect()
}

#[test]
fn thread_extent_maps_to_exact_core_id_cardinality() {
    let thread = UOp::range_axis(UOp::index_const(2), svod_ir::AxisId::Renumbered(0), AxisType::Thread);
    let sink = UOp::sink(vec![thread.clone()]);

    let lowered =
        add_gpudims(&mut GpuDimsContext::from(Renderer::cpu()), &sink).expect("thread range should lower to core_id");
    let core_id = lowered
        .toposort()
        .into_iter()
        .find(|u| matches!(u.op(), Op::Param(ops::Param { arg, .. }) if arg.name.as_deref() == Some("core_id")))
        .expect("lowered graph should contain core_id");

    assert_eq!(core_id.vmin(), &ConstValue::Int(0));
    assert_eq!(core_id.vmax(), &ConstValue::Int(1));

    let info = svod_ir::ProgramInfo::from_sink(&lowered, svod_dtype::DeviceSpec::Cpu);
    assert_eq!(info.global_size[0].vmin(), &ConstValue::Int(2));
    assert_eq!(info.global_size[0].vmax(), &ConstValue::Int(2));

    // One core_id cannot stand in for two THREAD axes: decline, don't panic.
    let second = UOp::range_axis(UOp::index_const(3), svod_ir::AxisId::Renumbered(1), AxisType::Thread);
    assert!(add_gpudims(&mut GpuDimsContext::from(Renderer::cpu()), &UOp::sink(vec![thread, second])).is_none());
}

#[test]
fn existing_special_skips_all_gpudims_lowering() {
    let global = UOp::range_axis(UOp::index_const(4), svod_ir::AxisId::Renumbered(0), AxisType::Global);
    let special = UOp::special(UOp::index_const(8), "gidx0".to_string());
    let sink = UOp::sink(vec![global, special]);

    assert!(add_gpudims(&mut GpuDimsContext::from(Renderer::amd_cdna3()), &sink).is_none());
}

/// Extents of the `gidx*` SPECIALs `add_gpudims` emits for the given axes,
/// sorted so the assertion does not depend on toposort order.
fn global_special_extents(renderer: &Renderer, global_extents: &[i64], local_extents: &[i64]) -> Vec<usize> {
    let mut ranges = Vec::new();
    for (extents, axis_type) in [(global_extents, AxisType::Global), (local_extents, AxisType::Local)] {
        for &extent in extents {
            let axis = svod_ir::AxisId::Renumbered(ranges.len());
            ranges.push(UOp::range_axis(UOp::index_const(extent), axis, axis_type));
        }
    }
    let lowered =
        add_gpudims(&mut GpuDimsContext::from(renderer.clone()), &UOp::sink(ranges)).expect("GPU ranges should lower");
    let mut ends: Vec<usize> = lowered
        .toposort()
        .into_iter()
        .filter_map(|uop| match uop.op() {
            Op::Special(ops::Special { end, name }) if name.starts_with("gidx") => Some(dim_max(end)),
            _ => None,
        })
        .collect();
    ends.sort_unstable();
    ends
}

/// The `lidx*` SPECIAL extents `add_gpudims` emits for a WARP axis of
/// `warp` threads plus `locals`, with `warp_axis` deciding where the warp
/// range sits in axis-id order.
fn local_special_extents(warp: i64, locals: &[i64], warp_axis: usize) -> Vec<(String, usize)> {
    let mut ranges: Vec<Arc<UOp>> = locals
        .iter()
        .enumerate()
        .map(|(i, &extent)| {
            let axis = svod_ir::AxisId::Renumbered(if i < warp_axis { i } else { i + 1 });
            UOp::range_axis(UOp::index_const(extent), axis, AxisType::Local)
        })
        .collect();
    let warp_range = UOp::range_axis(UOp::index_const(warp), svod_ir::AxisId::Renumbered(warp_axis), AxisType::Warp);
    ranges.push(warp_range.clone());
    let sink = UOp::sink(ranges);
    let lowered =
        add_gpudims(&mut GpuDimsContext::from(Renderer::cuda_sm80(false)), &sink).expect("GPU ranges should lower");
    // The warp range must lower to a bare `lidx0` SPECIAL: any div/mod on it
    // means another local was folded into the warp's thread dimension.
    let Op::Sink(ops::Sink { sources, .. }) = lowered.op() else { panic!("sink") };
    let warp_idx = &sources[locals.len()];
    assert!(
        matches!(warp_idx.op(), Op::Special(ops::Special { name, .. }) if name == "lidx0"),
        "warp axis must be lidx0 itself, got {}",
        warp_idx.tree()
    );
    let mut ends: Vec<(String, usize)> = lowered
        .toposort()
        .into_iter()
        .filter_map(|uop| match uop.op() {
            Op::Special(ops::Special { end, name }) if name.starts_with("lidx") => Some((name.clone(), dim_max(end))),
            _ => None,
        })
        .collect();
    ends.sort();
    ends.dedup();
    ends
}

/// A tensor-core WARP axis plus three size-2 LOCALs is four local dims for
/// CUDA's three: the extra local must fold into `lidx1`/`lidx2`, never into
/// the warp's `tid.x` (a BEAM plan `TC, ..., LOCAL, LOCAL, LOCAL` once put a
/// local in the low bit of `tid.x` and scrambled every `mma.sync` lane).
#[test_case(0; "warp numbered first")]
#[test_case(3; "warp numbered last")]
fn warp_axis_keeps_tid_x_to_itself(warp_axis: usize) {
    let ends = local_special_extents(32, &[2, 2, 2], warp_axis);
    assert_eq!(ends, [("lidx0".to_string(), 32), ("lidx1".to_string(), 4), ("lidx2".to_string(), 2)]);
}

#[test_case(&[8], &[4, 1 << 28]; "one local axis divides the work item cap")]
#[test_case(&[0], &[1 << 30]; "zero extent local axis does not divide by zero")]
#[test_case(&[64, 64, 64, 4], &[32, 1 << 25]; "contracted locals still cap the grid")]
fn global_product_cap_accounts_for_local_extent(local_extents: &[i64], expected: &[usize]) {
    assert_eq!(global_special_extents(&Renderer::amd_cdna3(), &[1 << 30], local_extents), expected);
}

#[test_case(DType::Index, 4; "index dtype")]
#[test_case(DType::Int32, 2; "int32 dtype")]
fn device_range_becomes_device_num_and_its_end_drops_all_params(dtype: DType, extent: i64) {
    let end = UOp::const_(dtype.clone(), ConstValue::Int(extent));
    let device = UOp::range_axis_dtype(end, svod_ir::AxisId::Renumbered(0), AxisType::Device, dtype.clone());
    let other = UOp::variable("other".to_string(), 0, 7, dtype.clone());
    let computation = device.add(&UOp::const_(dtype.clone(), ConstValue::Int(1)));
    let ended = computation.end(smallvec::smallvec![device, other]);
    let lowered = crate::rewrite::graph_rewrite(&pm_lower_device_ranges(), ended, &mut ());

    let Op::End(ops::End { ranges, .. }) = lowered.op() else { panic!("target keeps an empty END") };
    assert!(ranges.is_empty(), "Tinygrad removes every PARAM when _device_num is present");
    let device_num =
        lowered.toposort().into_iter().find(is_device_num).expect("DEVICE range should become _device_num");
    assert!(matches!(device_num.op(), Op::Param(ops::Param { arg, .. }) if arg.name.as_deref() == Some("_device_num")));
    assert_eq!(device_num.dtype(), dtype);
    assert_eq!(device_num.vmin().try_int(), Some(0));
    assert_eq!(device_num.vmax().try_int(), Some(extent - 1));
}

#[test]
fn a_non_device_end_keeps_its_params() {
    let other = UOp::variable("other".to_string(), 0, 7, DType::Index);
    let ended = UOp::index_const(1).end(smallvec::smallvec![other.clone()]);
    let lowered = crate::rewrite::graph_rewrite(
        &pm_add_gpudims(),
        ended.clone(),
        &mut GpuDimsContext::from(Renderer::amd_cdna3()),
    );

    assert!(Arc::ptr_eq(&lowered, &ended));
}

#[test_case(UOp::param(0, 16, DType::Float32, None); "bare global param")]
#[test_case(UOp::param(0, 16, DType::Float32, None).after(smallvec::smallvec![UOp::noop()]); "param behind AFTER")]
#[test_case(UOp::stack(smallvec::smallvec![
    UOp::param(0, 16, DType::Float32, None),
    UOp::param(1, 16, DType::Float32, None),
]); "stack of global params")]
fn missing_group_reduce_masks_a_structured_global_param_store(buffer: Arc<UOp>) {
    let group = UOp::range_axis(UOp::index_const(4), svod_ir::AxisId::Renumbered(0), AxisType::GroupReduce);
    // A symbolic offset keeps the INDEX from folding through the STACK row.
    let offset = UOp::variable("off".to_string(), 0, 15, DType::Index);
    let index = UOp::index().buffer(buffer).indices(vec![offset]).call().expect("index");
    let store = index.store(group.cast(DType::Float32));
    let sink = UOp::sink(vec![store]);

    let lowered =
        add_gpudims(&mut GpuDimsContext::from(Renderer::amd_cdna3()), &sink).expect("group range should lower");
    let masked_index = lowered
        .toposort()
        .into_iter()
        .find(|u| matches!(u.op(), Op::Index(ops::Index { indices, .. }) if matches!(indices[0].op(), Op::Ternary(..))))
        .expect("global store index should carry missing GroupReduce validity");
    let Op::Index(ops::Index { indices, .. }) = masked_index.op() else { unreachable!() };
    assert!(
        indices[0]
            .toposort()
            .iter()
            .any(|u| matches!(u.op(), Op::Special(ops::Special { name, .. }) if name == "lidx0"))
    );
}

#[test_case(|| d(&[4, 4]), &[16, 16, 16], Some(&[4, 4][..]); "two dims already fit")]
#[test_case(|| d(&[8, 8, 8]), &[256, 256, 256], Some(&[8, 8, 8][..]); "three dims already fit")]
#[test_case(|| d(&[4, 4, 4, 4]), &[256, 256, 256], Some(&[16, 4, 4][..]); "four dims collapse into three")]
#[test_case(|| d(&[1000]), &[10], None; "no grouping can fit the cap")]
// Regression: with per-axis caps equal to the product, [32,2,2] fits unchanged.
// The old cube-root cap (10 each) made axis 0 unfittable and panicked in split_dims.
#[test_case(|| d(&[32, 2, 2]), &[1024, 1024, 1024], Some(&[32, 2, 2][..]); "non-cubic local shape fits the product cap")]
#[test_case(
    || vec![UOp::variable("n".to_string(), 0, 100, DType::WeakInt), UOp::index_const(4), UOp::index_const(8), UOp::index_const(8)],
    &[2147483647, 65535, 65535],
    Some(&[400, 8, 8][..]);
    "symbolic dim merges under its vmax")]
fn group_dims_fits_the_axis_caps(dims: fn() -> Vec<Arc<UOp>>, max_sizes: &[usize], expected: Option<&[usize]>) {
    assert_eq!(group_dims(&dims(), max_sizes).as_deref().map(dmax).as_deref(), expected);
}

#[test_case(|| d(&[100]), true; "concrete dim splits under the cap")]
#[test_case(|| vec![UOp::variable("n".to_string(), 0, 200, DType::WeakInt)], false; "symbolic dim has no concrete factor")]
fn split_dims_reports_failure_instead_of_a_malformed_split(dims: fn() -> Vec<Arc<UOp>>, splits: bool) {
    let result = split_dims(&dims(), &[64, 64, 64]);
    match result {
        Some(split) => assert!(splits && split.iter().all(|x| dim_max(x) <= 64), "{:?}", dmax(&split)),
        None => assert!(!splits, "expected a split"),
    }
}

#[test_case(1, 1; "one")]
#[test_case(2, 2; "even")]
#[test_case(3, 1; "prime")]
#[test_case(4, 2; "square")]
#[test_case(9, 3; "odd square")]
#[test_case(100, 2; "composite")]
fn find_smallest_divisor_returns_one_for_primes(n: usize, expected: usize) {
    assert_eq!(find_smallest_divisor(n), expected);
}

#[test]
fn symbolic_identity_dims_return_bare_specials() {
    let n = UOp::variable("n".to_string(), 1, 1024, DType::WeakInt);
    let out = get_grouped_dims("gidx", &[n, UOp::index_const(8)], None, true);

    let names: Vec<&str> = out
        .iter()
        .map(|u| match u.op() {
            Op::Special(ops::Special { name, .. }) => name.as_str(),
            _ => panic!("an ungrouped, unsplit shape must not leave a symbolic FloorDiv/FloorMod: {}", u.tree()),
        })
        .collect();
    assert_eq!(names, vec!["gidx1", "gidx0"], "reverse=true names the innermost axis gidx0");
}

/// MSL exposes three grid axes (`gid.xyz`), so a fourth global axis must be
/// grouped into them instead of surfacing as `gidx3` (which the Metal renderer
/// rejects); locals go through the same three-axis cap.
#[test]
fn metal_groups_global_axes_into_the_three_grid_dimensions() {
    let extents = global_special_extents(&Renderer::metal(), &[8, 16, 32, 64], &[4]);
    assert_eq!(extents.len(), 3, "{extents:?}");
    assert_eq!(extents.iter().product::<usize>(), 8 * 16 * 32 * 64);
    assert_eq!(global_special_extents(&Renderer::metal(), &[2, 3], &[]), vec![2, 3]);
}
