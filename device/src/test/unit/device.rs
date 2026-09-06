use test_case::test_case;

use crate::{DeviceSpec, registry::DeviceSpecExt};
use svod_dtype::DType;
use svod_ir::ops;
use svod_ir::{Op, UOp};

fn program(
    sink: std::sync::Arc<UOp>,
    target: DeviceSpec,
    linear: Option<std::sync::Arc<UOp>>,
    source: Option<std::sync::Arc<UOp>>,
    binary: Option<std::sync::Arc<UOp>>,
) -> std::sync::Arc<UOp> {
    let info = svod_ir::ProgramInfo::from_sink(&sink, target);
    let source = match (&linear, source) {
        (Some(linear), Some(source)) => match source.op() {
            Op::Source(ops::Source { code, identity: None }) => {
                crate::device::ProgramSpec::validate_program_param_abi(&sink, &info)
                    .ok()
                    .and_then(|abi| crate::device::source_stage_identity(&info, &abi, linear, code).ok())
                    .map_or_else(
                        || Some(source.clone()),
                        |identity| Some(UOp::source_with_identity(code.clone(), identity)),
                    )
            }
            _ => Some(source),
        },
        (_, source) => source,
    };
    let binary = match (&source, binary) {
        (Some(source), Some(binary)) => match (source.op(), binary.op()) {
            (
                Op::Source(ops::Source { identity: Some(source_identity), .. }),
                Op::ProgramBinary(ops::ProgramBinary { bytes, identity: None }),
            ) => Some(UOp::binary_with_identity(
                bytes.clone(),
                crate::device::binary_stage_identity(source_identity.as_ref().clone(), "device-test", bytes),
            )),
            _ => Some(binary),
        },
        (_, binary) => binary,
    };
    UOp::program(sink, info, linear, source, binary)
}

fn slotted_var(name: &str, min: i64, max: i64, slot: usize) -> std::sync::Arc<UOp> {
    let var = UOp::variable(name.to_string(), min, max, DType::Int32);
    let Op::Param(ops::Param { shape, arg }) = var.op() else { panic!("variable PARAM") };
    let mut arg = arg.clone();
    arg.slot = slot;
    UOp::new(Op::Param(ops::Param { shape: shape.clone(), arg }), DType::Int32)
}

#[test]
fn compiled_spec_requires_complete_descriptor_abi() {
    use crate::device::{AbiParamDescriptor, AbiParamKind, CompiledSpec, validate_abi_descriptors};
    use svod_dtype::AddrSpace;

    let abi = vec![
        AbiParamDescriptor {
            slot: 0,
            kind: AbiParamKind::Storage(AddrSpace::Global),
            dtype: DType::Float32,
            name: None,
        },
        AbiParamDescriptor { slot: 5, kind: AbiParamKind::Scalar, dtype: DType::Int32, name: Some("n".into()) },
    ];
    let spec = CompiledSpec::from_source("k".into(), "void k(float*, int) {}".into(), UOp::sink(vec![]), abi)
        .expect("descriptor ABI");
    assert_eq!(spec.buf_count, 1);
    assert_eq!(spec.var_names, ["n"]);

    let err = validate_abi_descriptors(&[], 1, &[]).expect_err("descriptorless argument-bearing ABI must fail");
    assert!(matches!(err, crate::Error::ProgramAbiMismatch { .. }), "{err:?}");
}

/// Spec parsing is case-insensitive and accepts every vendor alias; the
/// canonical form always carries an explicit device id.
#[test]
fn device_spec_parses_aliases_and_round_trips_through_canonicalize() {
    let cases = [
        ("CPU", DeviceSpec::Cpu, "CPU"),
        ("cpu", DeviceSpec::Cpu, "CPU"),
        ("AMD", DeviceSpec::Amd { device_id: 0 }, "AMD:0"),
        ("AMD:1", DeviceSpec::Amd { device_id: 1 }, "AMD:1"),
        ("hip:2", DeviceSpec::Amd { device_id: 2 }, "AMD:2"),
        ("cuda", DeviceSpec::Cuda { device_id: 0 }, "CUDA:0"),
        ("CUDA:1", DeviceSpec::Cuda { device_id: 1 }, "CUDA:1"),
        ("GPU:2", DeviceSpec::Cuda { device_id: 2 }, "CUDA:2"),
        ("metal", DeviceSpec::Metal { device_id: 0 }, "Metal:0"),
        ("Metal:1", DeviceSpec::Metal { device_id: 1 }, "Metal:1"),
        ("webgpu", DeviceSpec::WebGpu, "WebGPU"),
    ];
    for (text, spec, canonical) in cases {
        assert_eq!(DeviceSpec::parse(text).unwrap(), spec, "{text}");
        assert_eq!(spec.canonicalize(), canonical);
    }
}

#[test_case("CUDA:x"; "non numeric id")]
#[test_case("AMD:-1"; "negative id")]
#[test_case("METAL:"; "empty id")]
#[test_case("NV:0"; "alias reserved for a userspace driver")]
#[test_case(""; "empty")]
fn device_spec_parse_rejects_malformed_specs(text: &str) {
    let err = DeviceSpec::parse(text).expect_err(text);
    assert!(matches!(err, crate::Error::InvalidDevice { .. }), "{text}: {err:?}");
}

/// Opening CUDA must never panic: without the driver library, without a GPU,
/// or on a driver failure it returns a typed error; with a GPU it succeeds
/// and `has_devices` agrees.
#[test]
fn cuda_allocator_open_returns_a_clean_result_on_every_host() {
    use crate::error::Error;
    match crate::registry::registry().get(&DeviceSpec::Cuda { device_id: 0 }) {
        Ok(allocator) => {
            assert_eq!(allocator.name(), "CUDA");
            assert!(crate::cuda::has_devices());
        }
        Err(error @ (Error::DeviceUnavailable { .. } | Error::NoCudaGpu { .. })) => {
            assert!(!crate::cuda::has_devices(), "{error:?}");
        }
        Err(Error::CudaDriver { .. }) => {}
        Err(other) => panic!("unexpected CUDA open error: {other:?}"),
    }
}

/// Opening AMD must never panic: without a GPU, on an unsupported arch (e.g.
/// RDNA2/gfx1036), or without permissions it returns a typed error instead.
#[test]
fn amd_device_open_returns_a_clean_result_on_every_host() {
    use crate::error::Error;
    match crate::registry::get_device("AMD:0") {
        Ok(_)
        | Err(
            Error::NoAmdGpu { .. }
            | Error::AmdAllocFailed { .. }
            | Error::AmdIoctl { .. }
            | Error::DeviceUnavailable { .. },
        ) => {}
        Err(other) => panic!("unexpected error variant: {other:?}"),
    }
}

/// Every field of a rebuilt spec comes from the PROGRAM's own stages; stale
/// attached metadata (name, source, buffer counts, I/O sets) is ignored.
#[test]
fn program_spec_from_uop_ignores_stale_metadata() {
    let sink = UOp::sink(vec![UOp::native_const(1.0f32)]);
    let linear = UOp::linear(sink.toposort().into());
    let source = UOp::source("// test kernel".to_string());
    let program = program(sink.clone(), DeviceSpec::Cpu, Some(linear), Some(source), None);

    let mut spec =
        crate::device::ProgramSpec::new("k_test".to_string(), "// old src".to_string(), DeviceSpec::Cpu, sink.clone());
    spec.set_buffer_metadata(vec![1, 0], vec![1], vec![0]);
    spec.set_var_names(vec!["N".to_string()]);
    spec.buf_count = 2;

    let rebuilt = crate::device::ProgramSpec::from_uop(&program.with_metadata(spec)).expect("program spec from uop");
    assert_eq!(rebuilt.name, "test");
    assert_eq!(rebuilt.src, "// test kernel");
    assert_eq!(rebuilt.device, DeviceSpec::Cpu);
    assert_eq!(rebuilt.ast.id, sink.id);
    assert!(rebuilt.var_names.is_empty());
    assert!(rebuilt.globals.is_empty());
    assert!(rebuilt.outs.is_empty());
    assert!(rebuilt.ins.is_empty());
    assert_eq!(rebuilt.buf_count, 0);
}

#[test]
fn program_spec_from_uop_derives_name_and_vars_without_metadata() {
    let var = slotted_var("N", 1, 8, 0);
    let sink = UOp::sink(vec![var]);
    let linear = UOp::linear(sink.toposort().into());
    let source = UOp::source("void var_kernel(float* data0) {}".to_string());
    let program = program(sink.clone(), DeviceSpec::Cpu, Some(linear), Some(source), None);

    let rebuilt = crate::device::ProgramSpec::from_uop(&program).expect("metadata-free from_uop should succeed");
    assert_eq!(rebuilt.name, "test");
    assert_eq!(rebuilt.var_names, vec!["N".to_string()]);
    assert_eq!(rebuilt.vars.len(), 1);
}

fn launch_dims(sink: std::sync::Arc<UOp>, vars: &[(&'static str, i64)]) -> crate::device::ConcreteLaunchDims {
    let linear = UOp::linear(sink.toposort().into());
    let source = UOp::source("void launch_kernel() {}".to_string());
    let staged = program(sink, DeviceSpec::Cpu, Some(linear), Some(source), None);
    let spec = crate::device::ProgramSpec::from_uop(&staged).expect("program spec");
    spec.launch_dims(&vars.iter().copied().collect()).expect("resolve launch dims")
}

/// `gidx`/`lidx` specials split into global and local sizes; a bare `idx`
/// special is a direct global launch with no local size at all.
#[test]
fn program_spec_derives_launch_dims_from_specials() {
    let gidx = UOp::special(UOp::index_const(8), "gidx0".to_string());
    let lidx = UOp::special(UOp::index_const(4), "lidx0".to_string());
    let split = launch_dims(UOp::sink(vec![gidx, lidx]), &[]);
    assert_eq!((split.global_size, split.local_size), ([8, 1, 1], Some([4, 1, 1])));

    let idx = UOp::special(UOp::index_const(16), "idx0".to_string());
    let direct = launch_dims(UOp::sink(vec![idx]), &[]);
    assert_eq!((direct.global_size, direct.local_size), ([16, 1, 1], None));
}

/// The symbolic simplifier fuses `16*ts - 1` (from a reshaped `16*ts` sequence
/// axis) into one MulAcc launch extent, which the evaluator must compute rather
/// than reject.
#[test]
fn program_spec_launch_dims_resolves_mulacc_extent() {
    let extent = UOp::try_mulacc(
        slotted_var("ts", 1, 8, 0),
        UOp::const_(DType::Int32, 16.into()),
        UOp::const_(DType::Int32, (-1).into()),
    )
    .expect("build MulAcc extent");
    let sink = UOp::sink(vec![UOp::special(extent, "gidx0".to_string())]);
    assert_eq!(launch_dims(sink, &[("ts", 8)]).global_size, [127, 1, 1], "ts*16 - 1 = 8*16 - 1 = 127");
}

/// A `core_id` variable sets the CPU global size from its bounds, and attached
/// metadata never overrides the bounds ProgramInfo carries.
#[test]
fn program_spec_core_id_sets_cpu_global_size_over_any_metadata() {
    assert_eq!(launch_dims(UOp::sink(vec![slotted_var("core_id", 0, 7, 0)]), &[]).global_size, [8, 1, 1]);

    let sink = UOp::sink(vec![slotted_var("core_id", 0, 3, 0)]);
    let linear = UOp::linear(sink.toposort().into());
    let source = UOp::source("void core_kernel(int core_id) {}".to_string());
    let staged = program(sink.clone(), DeviceSpec::Cpu, Some(linear), Some(source), None);
    let meta = crate::device::ProgramSpec::new("core".to_string(), "// old".to_string(), DeviceSpec::Cpu, sink);
    let spec = crate::device::ProgramSpec::from_uop(&staged.with_metadata(meta)).expect("program spec");
    assert_eq!(spec.launch_dims(&std::collections::HashMap::new()).unwrap().global_size, [4, 1, 1]);
}

#[test]
fn program_spec_from_uop_derives_buf_count_and_io_without_metadata() {
    let param = UOp::param(0, 16, DType::Float32, None);
    let idx = UOp::index_const(0);
    let load_idx = UOp::index().buffer(param.clone()).indices(vec![idx.clone()]).call().expect("load index");
    let load = UOp::load().index(load_idx).call();
    let store_idx = UOp::index().buffer(param).indices(vec![idx]).call().expect("store index");
    let sink = UOp::sink(vec![store_idx.store(load)]);
    let linear = UOp::linear(sink.toposort().into());
    let source = UOp::source("void io_kernel(float* data0) {}".to_string());
    let program = program(sink.clone(), DeviceSpec::Cpu, Some(linear), Some(source), None);

    let rebuilt = crate::device::ProgramSpec::from_uop(&program).expect("metadata-free from_uop should derive I/O");
    assert_eq!(rebuilt.globals, vec![0]);
    assert_eq!(rebuilt.outs, vec![0]);
    assert_eq!(rebuilt.ins, vec![0]);
    assert_eq!(rebuilt.buf_count, 1);
}

#[test]
fn program_spec_rejects_duplicate_storage_scalar_slots_with_typed_error() {
    let global = UOp::param(0, 1, DType::Float32, None);
    let scalar = UOp::variable("n".to_string(), 1, 8, DType::Int32);
    let Op::Param(ops::Param { shape, arg }) = scalar.op() else { unreachable!() };
    let mut arg = arg.clone();
    arg.slot = 0;
    let scalar = UOp::new(Op::Param(ops::Param { shape: shape.clone(), arg }), DType::Int32);
    let sink = UOp::sink(vec![global, scalar]);
    let linear = UOp::linear(sink.toposort().into());
    let source = UOp::source("void duplicate(float* data0, int n) {}".to_string());
    let program = program(sink, DeviceSpec::Cpu, Some(linear), Some(source), None);

    let err = crate::device::ProgramSpec::from_uop(&program).expect_err("duplicate slot must fail");
    assert!(matches!(err, crate::Error::DuplicateProgramParamSlot { slot: 0, .. }), "{err:?}");
}

#[test]
fn program_spec_rejects_descriptor_equivalent_var_semantic_forgery() {
    let scalar = slotted_var("n", 0, 16, 0);
    let sink = UOp::sink(vec![scalar]);

    for mutation in ["bounds", "multiple_of"] {
        let mut info = svod_ir::ProgramInfo::from_sink(&sink, DeviceSpec::Cpu);
        let Op::Param(ops::Param { shape, arg }) = info.vars[0].op() else { unreachable!() };
        let mut forged_arg = arg.clone();
        if mutation == "bounds" {
            forged_arg.vmin_vmax = Some((
                svod_ir::ConstValueHash(svod_ir::ConstValue::Int(-1000)),
                svod_ir::ConstValueHash(svod_ir::ConstValue::Int(1000)),
            ));
        } else {
            forged_arg.multiple_of = Some(8);
        }
        info.vars[0] = UOp::new(Op::Param(ops::Param { shape: shape.clone(), arg: forged_arg }), DType::Int32);
        let staged = UOp::program(
            sink.clone(),
            info,
            Some(UOp::linear(sink.toposort().into())),
            Some(UOp::source("void forged(int n) {}".to_string())),
            None,
        );

        let err = crate::device::ProgramSpec::from_uop(&staged).expect_err("semantic forgery must fail");
        match err {
            crate::Error::ProgramAbiMismatch { reason } => {
                assert!(reason.contains("ProgramInfo.vars"), "{mutation}: {reason}");
            }
            other => panic!("{mutation}: expected ProgramAbiMismatch, got {other:?}"),
        }
    }
}

#[test]
fn program_spec_accepts_semantically_identical_nonidentical_var() {
    let scalar = slotted_var("n", 0, 16, 0);
    let sink = UOp::sink(vec![scalar]);
    let mut info = svod_ir::ProgramInfo::from_sink(&sink, DeviceSpec::Cpu);
    let sink_var = info.vars[0].clone();
    let reconstructed = UOp::new(sink_var.op().clone(), sink_var.dtype()).with_metadata("detached variable");
    assert!(!std::sync::Arc::ptr_eq(&sink_var, &reconstructed));
    info.vars[0] = reconstructed;
    let staged = UOp::program(sink.clone(), info, Some(UOp::linear(sink.toposort().into())), None, None);
    let Op::Program(ops::Program { info, linear: Some(linear), .. }) = staged.op() else { unreachable!() };
    let abi = crate::device::ProgramSpec::validate_program_param_abi(&sink, info).unwrap();
    let code = "void accepted(int n) {}".to_string();
    let identity = crate::device::source_stage_identity(info, &abi, linear, &code).unwrap();
    let staged = UOp::program(
        sink.clone(),
        info.clone(),
        Some(linear.clone()),
        Some(UOp::source_with_identity(code, identity)),
        None,
    );

    crate::device::ProgramSpec::from_uop(&staged)
        .expect("validation must compare PARAM value semantics rather than allocation identity");
}

#[test]
fn program_spec_from_uop_requires_a_proven_program_source() {
    let sink = UOp::sink(vec![UOp::native_const(3.0f32)]);
    let linear = UOp::linear(sink.toposort().into());
    let program_without_source = program(sink.clone(), DeviceSpec::Cpu, Some(linear), None, None);
    assert!(crate::device::ProgramSpec::from_uop(&program_without_source).is_err());

    let non_program = UOp::native_const(1.0f32);
    assert!(crate::device::ProgramSpec::from_uop(&non_program).is_err());

    let bad_source = UOp::native_const(1.0f32);
    let linear = UOp::linear(sink.toposort().into());
    let bad_program = program(sink, DeviceSpec::Cpu, Some(linear), Some(bad_source), None);
    assert!(crate::device::ProgramSpec::from_uop(&bad_program).is_err());

    let raw_sink = UOp::sink(vec![]);
    let raw_linear = UOp::linear(raw_sink.toposort().into());
    let raw = UOp::program(
        raw_sink.clone(),
        svod_ir::ProgramInfo::from_sink(&raw_sink, DeviceSpec::Cpu),
        Some(raw_linear),
        Some(UOp::source("unproven".into())),
        None,
    );
    let err = crate::device::ProgramSpec::from_uop(&raw).expect_err("identity-less SOURCE must be rejected");
    assert!(matches!(err, crate::Error::ProgramStageMismatch { stage: "SOURCE", .. }), "{err:?}");
}

#[test]
fn program_spec_from_uop_binary_stage_ignores_metadata() {
    let sink = UOp::sink(vec![UOp::native_const(4.0f32)]);
    let linear = UOp::linear(sink.toposort().into());
    let source = UOp::source("// binary source".to_string());
    let program = program(sink.clone(), DeviceSpec::Cpu, Some(linear), Some(source), Some(UOp::binary(vec![1, 2, 3])));

    let mut spec =
        crate::device::ProgramSpec::new("precompiled".to_string(), "// cached src".to_string(), DeviceSpec::Cpu, sink);
    spec.set_var_names(vec!["N".to_string()]);
    spec.buf_count = 3;

    let program = program.with_metadata(spec);
    let rebuilt = crate::device::ProgramSpec::from_uop(&program).expect("program spec from binary+metadata");
    assert_eq!(rebuilt.name, "test");
    assert_eq!(rebuilt.src, "// binary source");
    assert!(rebuilt.var_names.is_empty());
    assert_eq!(rebuilt.buf_count, 0);
}

#[test]
fn program_spec_rejects_empty_binary_compiler_key() {
    let sink = UOp::sink(vec![]);
    let linear = UOp::linear(sink.toposort().into());
    let staged = program(
        sink,
        DeviceSpec::Cpu,
        Some(linear),
        Some(UOp::source("source".into())),
        Some(UOp::binary(vec![1, 2, 3])),
    );
    let Op::Program(ops::Program { sink, info, linear, source, binary: Some(binary) }) = staged.op() else {
        unreachable!()
    };
    let Op::ProgramBinary(ops::ProgramBinary { bytes, identity: Some(identity) }) = binary.op() else { unreachable!() };
    let malformed = UOp::binary_with_identity(
        bytes.clone(),
        svod_ir::BinaryStageIdentity { compiler_key: String::new(), ..identity.as_ref().clone() },
    );
    let staged = UOp::program(sink.clone(), info.clone(), linear.clone(), source.clone(), Some(malformed));
    let err = crate::device::ProgramSpec::from_uop(&staged).expect_err("empty compiler key must be rejected");
    assert!(matches!(err, crate::Error::ProgramStageMismatch { stage: "BINARY", .. }), "{err:?}");
}

#[test]
fn beam_worker_artifact_validates_source_binary_abi_and_compiler_identity() {
    let sink = UOp::sink(vec![UOp::native_const(1i32)]);
    let linear = UOp::linear(sink.toposort().into());
    let info = svod_ir::ProgramInfo { name: "beam_worker".into(), target: DeviceSpec::Cpu, ..Default::default() };
    let source = "void beam_worker(void) {}".to_string();
    let abi = Vec::new();
    let source_identity = crate::device::source_stage_identity(&info, &abi, &linear, &source).unwrap();
    let bytes = vec![1, 2, 3, 4];
    let identity = crate::device::binary_stage_identity(source_identity, "compiler", &bytes);
    let launch = || [UOp::index_const(1), UOp::index_const(1), UOp::index_const(1)];

    crate::device::CompiledSpec::from_beam_worker(
        "beam_worker".into(),
        source.clone(),
        bytes.clone(),
        sink.clone(),
        abi.clone(),
        launch(),
        identity.clone(),
        &DeviceSpec::Cpu,
        "compiler",
    )
    .unwrap();
    assert!(
        crate::device::CompiledSpec::from_beam_worker(
            "beam_worker".into(),
            format!("{source} // tampered"),
            bytes.clone(),
            sink.clone(),
            abi.clone(),
            launch(),
            identity.clone(),
            &DeviceSpec::Cpu,
            "compiler",
        )
        .is_err()
    );
    assert!(
        crate::device::CompiledSpec::from_beam_worker(
            "beam_worker".into(),
            source,
            vec![9],
            sink,
            abi,
            launch(),
            identity,
            &DeviceSpec::Cpu,
            "other-compiler",
        )
        .is_err()
    );
}

/// Downstream stages trust the LINEAR digest minted at render time but
/// re-derive every other identity field from the PROGRAM they are given.
#[test]
fn program_spec_from_uop_rechecks_every_minted_identity_field_but_the_linear_digest() {
    let sink = UOp::sink(vec![UOp::native_const(3.0f32)]);
    let linear = UOp::linear(sink.toposort().into());
    let info = svod_ir::ProgramInfo::from_sink(&sink, DeviceSpec::Cpu);
    let abi = crate::device::ProgramSpec::validate_program_param_abi(&sink, &info).unwrap();
    let code = "void test(void) {}".to_string();
    let minted = crate::device::source_stage_identity(&info, &abi, &linear, &code).unwrap();
    let staged = |source| UOp::program(sink.clone(), info.clone(), Some(linear.clone()), Some(source), None);

    let honest = UOp::source_with_identity(code.clone(), minted.clone());
    assert_eq!(crate::device::minted_source_stage_identity(&info, &abi, &honest).unwrap(), minted);
    crate::device::ProgramSpec::from_uop(&staged(honest)).expect("minted identity is accepted");

    let tampered = [
        UOp::source_with_identity(format!("{code} // edited"), minted.clone()),
        UOp::source_with_identity(
            code.clone(),
            svod_ir::SourceStageIdentity { entry_name: "other".into(), ..minted.clone() },
        ),
        UOp::source_with_identity(code.clone(), svod_ir::SourceStageIdentity { version: minted.version + 1, ..minted }),
    ];
    for source in tampered {
        let err = crate::device::ProgramSpec::from_uop(&staged(source)).expect_err("tampered SOURCE must be rejected");
        assert!(matches!(err, crate::Error::ProgramStageMismatch { stage: "SOURCE", .. }), "{err:?}");
    }
}
