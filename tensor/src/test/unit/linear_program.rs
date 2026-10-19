//! `UOp::linear_program`: an author-ordered LINEAR list renders verbatim on the
//! direct path and through a lazy `custom_kernel` graph node.

use std::collections::HashMap;
use std::sync::Arc;

use smallvec::smallvec;
use svod_codegen::program_pipeline::{self, ProgramTarget};
use svod_device::device::{Device, ProgramSpec};
use svod_dtype::{DType, DeviceSpec};
use svod_ir::{AxisId, AxisType, ConstValue, KernelInfo, Op, UOp, ops};

use crate::{CpuBackend, PrepareConfig, Tensor};

const N: usize = 32;
/// `gidx1 × gidx0 × lidx0` threads, each owning `K` loop elements of the
/// first `THREADS * K` and one element of the tail.
const GRID: [i64; 3] = [2, 2, 2];
const THREADS: i64 = GRID[0] * GRID[1] * GRID[2];
const K: i64 = 3;
const NOP: &str = "declare void @llvm.donothing()\ncall void @llvm.donothing()";

fn i32c(v: i64) -> Arc<UOp> {
    UOp::const_(DType::Int32, ConstValue::Int(v))
}

fn range(end: i64, axis: usize) -> Arc<UOp> {
    UOp::range_axis_dtype(i32c(end), AxisId::Renumbered(axis), AxisType::Loop, DType::Int32)
}

fn plus_one_store(y: &Arc<UOp>, x: &Arc<UOp>, i: &Arc<UOp>) -> Arc<UOp> {
    let load = UOp::load().index(UOp::index().buffer(x.clone()).indices(vec![i.clone()]).call().unwrap()).call();
    let value = load.try_add(&UOp::const_(DType::Float32, ConstValue::Float(1.0))).unwrap();
    UOp::index().buffer(y.clone()).indices(vec![i.clone()]).call().unwrap().store(value)
}

/// `y = x + 1` over `N` elements: a loop store, then a barrier no data edge
/// orders, then a tail store, with one `Custom` statement listed twice. On a
/// GPU the thread id is `gidx1/gidx0/lidx0`; the CPU has no SPECIAL, so the
/// same three axes are loops closed at the end.
fn add_one_ops(y: &Arc<UOp>, x: &Arc<UOp>, gpu: bool) -> Vec<Arc<UOp>> {
    let axes: Vec<Arc<UOp>> = if gpu {
        ["gidx1", "gidx0", "lidx0"]
            .iter()
            .zip(GRID)
            .map(|(name, end)| UOp::special_dtype(i32c(end), name.to_string(), DType::Int32))
            .collect()
    } else {
        GRID.iter().enumerate().map(|(axis, &end)| range(end, axis + 1)).collect()
    };
    let thread =
        axes.iter().zip(GRID).fold(i32c(0), |acc, (axis, end)| acc.try_mul(&i32c(end)).unwrap().try_add(axis).unwrap());
    let k = range(K, 0);
    let looped = plus_one_store(y, x, &thread.try_mul(&i32c(K)).unwrap().try_add(&k).unwrap());
    let loop_end = looped.end(smallvec![k.clone()]);
    let barrier = loop_end.barrier(smallvec![]);
    let tail = plus_one_store(y, x, &thread.try_add(&i32c(THREADS * K)).unwrap());
    let nop = UOp::custom(smallvec![], NOP.to_string(), DType::Void);

    // Values used on both sides of the loop are listed before it: an unlisted
    // source lands right before its first user, which would be the loop body.
    let mut list = axes.clone();
    list.extend([thread, k, looped, loop_end, nop.clone(), barrier, tail.clone(), nop]);
    if !gpu {
        let mut last = tail;
        for axis in axes.iter().rev() {
            last = last.end(smallvec![axis.clone()]);
            list.push(last.clone());
        }
    }
    list
}

fn add_one_program(y: &Arc<UOp>, x: &Arc<UOp>, target: DeviceSpec) -> Arc<UOp> {
    let gpu = !matches!(target, DeviceSpec::Cpu);
    let info = KernelInfo { name: Some("linear_add_one".into()), ..Default::default() };
    UOp::linear_program(info, add_one_ops(y, x, gpu), target).expect("well-ordered list")
}

fn linear_ops(program: &Arc<UOp>) -> Vec<Arc<UOp>> {
    let Op::Program(ops::Program { linear: Some(linear), .. }) = program.op() else { panic!("no LINEAR stage") };
    let Op::Linear(ops::Linear { ops }) = linear.op() else { panic!("LINEAR stage is not LINEAR") };
    ops.to_vec()
}

fn config_for(spec: &DeviceSpec) -> PrepareConfig {
    match spec {
        DeviceSpec::Cpu => PrepareConfig::for_cpu_backend(CpuBackend::Llvm),
        _ => PrepareConfig::default(),
    }
}

fn input() -> (Tensor, Vec<f32>) {
    let data: Vec<f32> = (0..N).map(|i| i as f32 * 0.5 - 3.0).collect();
    let x = Tensor::from_slice(&data);
    x.realize().unwrap();
    (x, data.iter().map(|v| v + 1.0).collect())
}

/// The `tk` direct path: optimizer, PROGRAM boundary, render, compile, run.
fn compile_and_run(device: &Device, program: Arc<UOp>, buffers: &[svod_device::Buffer]) -> String {
    let renderer = crate::realize::get_optimizer_renderer(device);
    let optimized = svod_schedule::optimize_kernel_with_config(program.clone(), &renderer, &Default::default())
        .expect("optimizer passes a PROGRAM through");
    assert!(Arc::ptr_eq(&optimized, &program), "the optimizer must not touch a PROGRAM");
    let program = program_pipeline::program_from_sink_with_renderer(optimized, device.renderer.as_ref()).unwrap();
    let rendered = program_pipeline::get_program(
        &program,
        device.renderer.as_ref(),
        device.compiler.as_ref(),
        ProgramTarget::Source,
    )
    .unwrap();
    assert_eq!(linear_ops(&rendered), linear_ops(&program), "the LINEAR stage renders as authored");
    let (compiled_program, compiled) = program_pipeline::do_compile(&rendered, device.compiler.as_ref()).unwrap();
    let spec = ProgramSpec::from_uop(&compiled_program).unwrap();
    let dims = spec.launch_dims(&HashMap::new()).unwrap();
    let ptrs: Vec<*mut u8> = spec
        .globals
        .iter()
        .map(|&slot| {
            buffers[slot].ensure_allocated().unwrap();
            // SAFETY: allocated above and alive for the synchronous dispatch.
            unsafe { buffers[slot].as_raw_ptr() }
        })
        .collect();
    let program = (device.runtime)(&compiled).unwrap();
    // SAFETY: every pointer is a live allocation sized for the kernel.
    unsafe { program.execute(&ptrs, &[], Some(dims.global_size), dims.local_size, true).unwrap() };
    spec.src
}

fn positions(source: &str, needle: &str) -> Vec<usize> {
    source.match_indices(needle).map(|(at, _)| at).collect()
}

/// Store, NOP, barrier (GPU only), store, NOP — the authored order, though no
/// data edge orders the tail store against the barrier.
fn assert_authored_order(source: &str, gpu: bool) {
    let stores = positions(source, "store float");
    let nops = positions(source, "call void @llvm.donothing()");
    assert_eq!(stores.len(), 2, "{source}");
    assert_eq!(nops.len(), 2, "a repeated statement renders at each position:\n{source}");
    assert!(stores[0] < nops[0] && nops[0] < stores[1] && stores[1] < nops[1], "{source}");
    if gpu {
        let barriers = positions(source, "call void @llvm.nvvm.barrier0()");
        assert_eq!(barriers.len(), 1, "{source}");
        assert!(nops[0] < barriers[0] && barriers[0] < stores[1], "{source}");
    }
}

#[test]
fn identical_statements_are_one_node_listed_twice() {
    let nop = || UOp::custom(smallvec![], NOP.to_string(), DType::Void);
    assert!(Arc::ptr_eq(&nop(), &nop()), "hash-consing merges identical statements");
    let y = UOp::param(0, N, DType::Float32, None);
    let x = UOp::param(1, N, DType::Float32, None);
    let program = add_one_program(&y, &x, DeviceSpec::Cpu);
    let listed = linear_ops(&program).iter().filter(|op| Arc::ptr_eq(op, &nop())).count();
    assert_eq!(listed, 2);
}

#[test]
fn derived_sink_reaches_every_param_and_store() {
    let y = UOp::param(0, N, DType::Float32, None);
    let x = UOp::param(1, N, DType::Float32, None);
    let program = add_one_program(&y, &x, DeviceSpec::Cuda { device_id: 0 });
    let Op::Program(ops::Program { sink, info, .. }) = program.op() else { unreachable!() };
    assert!(Arc::ptr_eq(linear_ops(&program).last().unwrap(), sink), "the SINK closes the list");
    assert_eq!((info.globals.clone(), info.outs.clone(), info.ins.clone()), (vec![0, 1], vec![0], vec![1]));
    let sizes = |dims: &[Arc<UOp>]| dims.iter().map(|d| d.vmax().try_int().unwrap()).collect::<Vec<_>>();
    assert_eq!(sizes(&info.global_size), vec![GRID[1], GRID[0], 1]);
    assert_eq!(sizes(info.local_size.as_ref().unwrap()), vec![GRID[2], 1, 1]);
}

#[test]
fn typed_value_repeats_only_when_tagged() {
    let y = UOp::param(0, 1, DType::Int32, None);
    let seven = UOp::custom(smallvec![], "add i32 0, 7".to_string(), DType::Int32);
    let at = |v: &Arc<UOp>| UOp::index().buffer(y.clone()).indices(vec![i32c(0)]).call().unwrap().store(v.clone());
    let info = || KernelInfo { name: Some("repeat".into()), ..Default::default() };

    let err = UOp::linear_program(info(), [seven.clone(), seven.clone()], DeviceSpec::Cpu).unwrap_err();
    assert!(matches!(err, svod_ir::Error::LinearRepeat { .. }), "{err}");

    let again = seven.rtag(Some(smallvec![1]));
    assert!(!Arc::ptr_eq(&seven, &again));
    let sum = seven.try_add(&again).unwrap();
    let program = UOp::linear_program(info(), [seven.clone(), again, at(&sum)], DeviceSpec::Cpu).unwrap();
    let config = PrepareConfig::for_cpu_backend(CpuBackend::Llvm);
    let device = config.resolve_device(&DeviceSpec::Cpu, svod_device::registry::registry()).unwrap();
    let out = Tensor::from_slice([0i32]);
    out.realize().unwrap();
    let source = compile_and_run(&device, program, &[out.buffer().unwrap()]);
    assert_eq!(positions(&source, "add i32 0, 7").len(), 2, "{source}");
    assert_eq!(out.as_vec::<i32>().unwrap(), vec![14]);
}

#[test]
fn direct_launch_renders_authored_order() {
    let (x, expected) = input();
    let spec = x.buffer().unwrap().allocator().device_spec();
    let gpu = !matches!(spec, DeviceSpec::Cpu);
    let device = config_for(&spec).resolve_device(&spec, svod_device::registry::registry()).unwrap();
    let y = Tensor::from_slice(vec![0f32; N]);
    y.realize().unwrap();

    let program =
        add_one_program(&UOp::param(0, N, DType::Float32, None), &UOp::param(1, N, DType::Float32, None), spec.clone());
    let source = compile_and_run(&device, program, &[y.buffer().unwrap(), x.buffer().unwrap()]);
    assert_authored_order(&source, gpu);
    assert_eq!(y.as_vec::<f32>().unwrap(), expected);
}

#[test]
fn graph_kernel_runs_a_linear_program() {
    let (x, expected) = input();
    let spec = x.buffer().unwrap().allocator().device_spec();
    let out = Tensor::empty(&[N], DType::Float32);
    let y =
        Tensor::graph_kernel("linear_add_one", out, &[&x], |ph| add_one_program(&ph[0], &ph[1], spec.clone())).unwrap();
    let y = (&y * 2.0f32).unwrap();

    let plan = y.prepare_with(&config_for(&spec)).unwrap();
    let kernel = plan.kernels().find(|k| k.entry_point == "linear_add_one").expect("the PROGRAM compiles as is");
    assert_authored_order(&kernel.code, !matches!(spec, DeviceSpec::Cpu));
    plan.execute().unwrap();

    let mut actual = vec![0f32; N];
    // SAFETY: `actual` is `N` initialized f32s viewed as bytes.
    let bytes = unsafe { std::slice::from_raw_parts_mut(actual.as_mut_ptr().cast::<u8>(), N * size_of::<f32>()) };
    plan.output_buffer().unwrap().copyout(bytes).unwrap();
    assert_eq!(actual, expected.iter().map(|v| v * 2.0).collect::<Vec<_>>());
}

const ROW: i64 = 8;
const MAX_ROWS: i64 = 4;

/// `y[r, c] = x[r, c] + 1` for the live rows only: the row count is the bound
/// variable `rows`, a launch-grid extent on a GPU and a loop bound on the CPU.
fn rows_program(y: &Arc<UOp>, x: &Arc<UOp>, target: DeviceSpec) -> Arc<UOp> {
    // Slots 0 and 1 are the buffers; the variable binds by name.
    let rows = UOp::scalar_param(2, Some("rows".into()), DType::Int32, 1, MAX_ROWS);
    let (row, col, gpu) = match target {
        DeviceSpec::Cpu => {
            (UOp::range_axis_dtype(rows, AxisId::Renumbered(0), AxisType::Loop, DType::Int32), range(ROW, 1), false)
        }
        _ => (
            UOp::special_dtype(rows, "gidx0".into(), DType::Int32),
            UOp::special_dtype(i32c(ROW), "lidx0".into(), DType::Int32),
            true,
        ),
    };
    let store = plus_one_store(y, x, &row.try_mul(&i32c(ROW)).unwrap().try_add(&col).unwrap());
    let mut ops = vec![row.clone(), col.clone(), store.clone()];
    if !gpu {
        let inner = store.end(smallvec![col]);
        ops.extend([inner.clone(), inner.end(smallvec![row])]);
    }
    let info = KernelInfo { name: Some("linear_rows".into()), ..Default::default() };
    UOp::linear_program(info, ops, target).unwrap()
}

#[test]
fn graph_kernel_binds_a_variable_grid() {
    let spec = svod_dtype::default_device::default_device();
    let config = config_for(&spec);
    let rows = crate::Variable::new("rows", 1, MAX_ROWS);
    let shape = [rows.bind(MAX_ROWS).unwrap().as_sint(), (ROW as usize).into()];
    let x = Tensor::empty_dynamic(&shape, DType::Float32);
    let data: Vec<f32> = (0..MAX_ROWS * ROW).map(|i| i as f32).collect();
    x.assign(&Tensor::from_slice(&data).try_reshape([MAX_ROWS as usize, ROW as usize]).unwrap());
    x.clone().realize_with(&config).unwrap();

    let out = Tensor::empty_dynamic(&shape, DType::Float32);
    let y =
        Tensor::graph_kernel("linear_rows", out, &[&x], |ph| rows_program(&ph[0].base(), &ph[1].base(), spec.clone()))
            .unwrap();
    let mut plan = Tensor::prepare_batch_with([&y], &config).unwrap();
    let kernel = plan.kernels().find(|k| k.entry_point == "linear_rows").expect("compiled once");
    assert_eq!(kernel.var_names, vec!["rows".to_string()]);

    // Ascending, so a row past the live count is one no launch has written.
    for n in [2, 3, MAX_ROWS] {
        plan.execute_with_vars(&[rows.bind(n).unwrap().as_var_val()]).unwrap();
        let mut actual = vec![0f32; (MAX_ROWS * ROW) as usize];
        // SAFETY: `actual` is initialized f32s viewed as bytes.
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(actual.as_mut_ptr().cast::<u8>(), actual.len() * size_of::<f32>())
        };
        y.buffer().unwrap().copyout(bytes).unwrap();
        let live = (n * ROW) as usize;
        assert_eq!(actual[..live], data[..live].iter().map(|v| v + 1.0).collect::<Vec<_>>(), "rows={n}");
        assert!(actual[live..].iter().zip(&data[live..]).all(|(y, x)| *y != x + 1.0), "rows={n} ran past the batch");
    }
}

/// 64 KB of shared memory, above the static 48 KB `ptxas` accepts: thread
/// `t` of 256 stores `x[t]` at the top of the buffer, and after the barrier
/// reads its mirror's slot, so `y = reverse(x)` holds only when the whole
/// buffer is addressable at launch. CUDA only.
#[test]
fn graph_kernel_uses_64k_dynamic_shared_memory() {
    const THREADS: i64 = 256;
    const SMEM: usize = 16 << 10;
    let spec = svod_dtype::default_device::default_device();
    if !matches!(spec, DeviceSpec::Cuda { .. }) {
        return;
    }
    let data: Vec<f32> = (0..THREADS).map(|i| i as f32 * 0.25 - 9.0).collect();
    let x = Tensor::from_slice(&data);
    let program = |y: &Arc<UOp>, x: &Arc<UOp>| {
        let t = UOp::special_dtype(i32c(THREADS), "lidx0".into(), DType::Int32);
        let smem = UOp::buffer(0, SMEM, DType::Float32, svod_dtype::AddrSpace::Local, None);
        let top = i32c(SMEM as i64 - 1);
        let at = |i: Arc<UOp>| UOp::index().buffer(smem.clone()).indices(vec![i]).call().unwrap();
        let load = UOp::load().index(UOp::index().buffer(x.clone()).indices(vec![t.clone()]).call().unwrap()).call();
        let fill = at(top.try_sub(&t).unwrap()).store(load);
        let barrier = fill.barrier(smallvec![]);
        let mirror = at(top.try_sub(&i32c(THREADS - 1)).unwrap().try_add(&t).unwrap());
        let read = UOp::load().index(mirror.clone()).call();
        let store = UOp::index().buffer(y.clone()).indices(vec![t.clone()]).call().unwrap().store(read);
        let info = KernelInfo { name: Some("linear_dynamic_smem".into()), ..Default::default() };
        UOp::linear_program(info, [t, smem.clone(), fill, barrier, mirror, store], spec.clone()).unwrap()
    };
    let y =
        Tensor::graph_kernel("linear_dynamic_smem", Tensor::empty(&[THREADS as usize], DType::Float32), &[&x], |ph| {
            program(&ph[0], &ph[1])
        })
        .unwrap();
    let plan = y.prepare_with(&config_for(&spec)).unwrap();
    let kernel = plan.kernels().find(|k| k.entry_point == "linear_dynamic_smem").expect("compiled as is");
    assert!(kernel.code.contains("@svod_dynamic_shared = external addrspace(3)"), "{}", kernel.code);
    plan.execute().unwrap();
    let mut actual = vec![0f32; THREADS as usize];
    // SAFETY: `actual` is initialized f32s viewed as bytes.
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(actual.as_mut_ptr().cast::<u8>(), actual.len() * size_of::<f32>()) };
    plan.output_buffer().unwrap().copyout(bytes).unwrap();
    assert_eq!(actual, data.iter().rev().copied().collect::<Vec<_>>());
}
