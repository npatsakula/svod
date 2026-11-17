//! A SOURCE stage rendered ahead by [`LocatedAmd`] is what an AMD device
//! compiles: its pipeline takes the PROGRAM as handed, renders nothing, and
//! launches it by the specification its own render gives the bare one.

use std::collections::HashMap;

use svod_codegen::llvm::Location;
use svod_codegen::program_pipeline::do_render;
use svod_dtype::DType;
use svod_ir::{ConstValue, KernelInfo, Op, ops};

use super::*;

const N: usize = 64;

/// `y[i] = x[i] + 1` over `N` threads, pre-linearized for an AMD device,
/// and its store's id.
fn add_one(device: &DeviceSpec) -> (Arc<UOp>, u64) {
    let i32c = |v| UOp::const_(DType::Int32, ConstValue::Int(v));
    let (y, x) = (UOp::param(0, N, DType::Float32, None), UOp::param(1, N, DType::Float32, None));
    let i = UOp::special_dtype(i32c(N as i64), "gidx0".to_string(), DType::Int32);
    let load = UOp::load().index(UOp::index().buffer(x).indices(vec![i.clone()]).call().unwrap()).call();
    let value = load.try_add(&UOp::const_(DType::Float32, ConstValue::Float(1.0))).unwrap();
    let store = UOp::index().buffer(y).indices(vec![i.clone()]).call().unwrap().store(value);
    let info = KernelInfo { name: Some("add_one".into()), ..Default::default() };
    let id = store.id;
    (UOp::linear_program(info, [i, store], device.clone()).expect("a well-ordered list"), id)
}

fn source(program: &Arc<UOp>) -> &str {
    let Op::Program(ops::Program { source: Some(source), .. }) = program.op() else { panic!("no SOURCE stage") };
    let Op::Source(ops::Source { code, .. }) = source.op() else { panic!("the SOURCE stage is no SOURCE") };
    code
}

/// Every field of a specification but its source.
fn launch(spec: &ProgramSpec) -> String {
    let ids = |dims: &[Arc<UOp>]| dims.iter().map(|d| d.id).collect::<Vec<_>>();
    format!(
        "{} {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?} {} {:?}",
        spec.name,
        spec.device,
        ids(&spec.global_size),
        spec.local_size.as_ref().map(|l| ids(l)),
        spec.vars,
        spec.var_names,
        spec.globals,
        spec.outs,
        spec.ins,
        spec.buf_count,
        spec.abi
    )
}

#[test]
fn a_device_takes_a_located_source_as_rendered() {
    let (arch, device) = (AmdArch::Gfx1201, DeviceSpec::Amd { device_id: 0 });
    let own = AmdRendererWrapper { device: device.clone(), arch, integer_bf16: false };
    let (program, store) = add_one(&device);
    let located =
        LocatedAmd::new(device, arch, HashMap::from([(store, Location { line: 1, column: 0 })]), "y = x + 1\n".into());
    let (ahead, _) = do_render(&program, &located).expect("renders");
    let ahead = ahead.without_origins();
    assert!(
        source(&ahead).contains("!dbg") && source(&ahead).contains("source: \"y = x + 1\\0A\""),
        "{}",
        source(&ahead)
    );

    let (taken, skipped) = do_render(&ahead, &own).expect("the device takes it");
    assert!(Arc::ptr_eq(&taken, &ahead), "a PROGRAM with its SOURCE is compiled as handed");
    let (rendered, spec) = do_render(&program, &own).expect("the device renders it");
    assert!(!source(&rendered).contains("!dbg"));
    assert_eq!(skipped.src, source(&ahead));
    assert_eq!(launch(&skipped), launch(&spec));
}
