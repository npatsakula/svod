use std::sync::Arc;

use svod_device::device::{CompiledSpec, Compiler, ProgramSpec, Renderer};
use svod_dtype::{DType, DeviceSpec};
use svod_ir::{BinaryOp, InsArg, Op, ParamArg, RewriteResult, TypedPatternMatcher, UOp};

use crate::program_pipeline::{
    ProgramTarget, do_compile, do_linearize, do_render, executable_params, get_program, number_params,
    program_from_sink, program_from_sink_with_renderer,
};
use svod_ir::ops;

fn committed_sink(sources: Vec<Arc<UOp>>) -> Arc<UOp> {
    let sink = UOp::sink(sources);
    let sink = svod_schedule::graph_rewrite(
        &svod_schedule::symbolic::pm_lower_index_dtype(),
        sink,
        &mut svod_schedule::symbolic::WeakMemo::default(),
    );
    assert!(sink.toposort().iter().all(|u| !u.dtype().is_weak()), "end-to-end fixture must commit weak dtypes");
    sink
}

fn program(sink: Arc<UOp>, linear: Option<Arc<UOp>>, source: Option<Arc<UOp>>, binary: Option<Arc<UOp>>) -> Arc<UOp> {
    let info = svod_ir::ProgramInfo::from_sink(&sink, DeviceSpec::Cpu);
    UOp::program(sink, info, linear, source, binary)
}

fn linear_of(sink: &Arc<UOp>) -> Arc<UOp> {
    UOp::linear(svod_schedule::linearize_with_cfg(sink.clone()).into())
}

fn scalar_param(name: &str, slot: usize) -> Arc<UOp> {
    let var = UOp::variable(name.to_string(), 0, 16, DType::Int32);
    let Op::Param(ops::Param { shape, arg }) = var.op() else { unreachable!() };
    let mut arg = arg.clone();
    arg.slot = slot;
    UOp::new(Op::Param(ops::Param { shape: shape.clone(), arg }), DType::Int32)
}

fn param_slot(root: &Arc<UOp>, name: &str) -> usize {
    root.toposort()
        .into_iter()
        .find_map(|u| match u.op() {
            Op::Param(ops::Param { arg, .. }) if arg.name.as_deref() == Some(name) => Some(arg.slot),
            _ => None,
        })
        .unwrap_or_else(|| panic!("missing PARAM {name}"))
}

fn var_slots(info: &svod_ir::ProgramInfo) -> Vec<usize> {
    info.vars
        .iter()
        .map(|var| match var.op() {
            Op::Param(ops::Param { arg, .. }) => arg.slot,
            _ => unreachable!("ProgramInfo.vars must be PARAMs"),
        })
        .collect()
}

// ── renderers and compilers ────────────────────────────────────────────────

struct MockRenderer {
    device: DeviceSpec,
}

/// Asserts it is handed the LINEAR stage rather than the SINK.
struct LinearOnlyRenderer {
    device: DeviceSpec,
}

/// Renders through the real C / LLVM text backends so the ABI they publish is
/// checked against `ProgramInfo`.
struct CAbiRenderer;
struct LlvmAbiRenderer;

/// Publishes a storage-count-only ABI, with no per-parameter descriptors.
struct WrongAbiRenderer;

/// Publishes the real descriptors in the wrong order — same count, same slots.
struct ReversedStorageRenderer;

macro_rules! cpu_renderer {
    ($ty:ty, $render:expr) => {
        impl Renderer for $ty {
            fn supported_ops(&self) -> svod_ir::RendererOps {
                svod_ir::RendererOps::all()
            }

            fn render(&self, ast: &Arc<UOp>, name: Option<&str>) -> svod_device::Result<ProgramSpec> {
                #[allow(clippy::redundant_closure_call)]
                ($render)(ast, name)
            }

            fn device(&self) -> &DeviceSpec {
                static DEVICE: DeviceSpec = DeviceSpec::Cpu;
                &DEVICE
            }
        }
    };
}

fn text_spec(rendered: crate::RenderedKernel, ast: &Arc<UOp>, reverse_abi: bool) -> svod_device::Result<ProgramSpec> {
    let mut spec = ProgramSpec::new(rendered.name, rendered.code, DeviceSpec::Cpu, ast.clone());
    spec.var_names = rendered.var_names;
    spec.buf_count = rendered.buffer_args.len();
    spec.abi = rendered.abi;
    if reverse_abi {
        spec.abi.reverse();
    }
    Ok(spec)
}

fn render_text<E: std::fmt::Display>(
    result: Result<crate::RenderedKernel, E>,
    ast: &Arc<UOp>,
    reverse_abi: bool,
) -> svod_device::Result<ProgramSpec> {
    let rendered =
        result.map_err(|error| svod_device::Error::Runtime { message: format!("text rendering failed: {error}") })?;
    text_spec(rendered, ast, reverse_abi)
}

cpu_renderer!(CAbiRenderer, |ast: &Arc<UOp>, name| render_text(crate::c::render(ast, name), ast, false));
cpu_renderer!(LlvmAbiRenderer, |ast: &Arc<UOp>, name| render_text(crate::llvm::text::render(ast, name), ast, false));
cpu_renderer!(ReversedStorageRenderer, |ast: &Arc<UOp>, name| render_text(crate::c::render(ast, name), ast, true));
cpu_renderer!(WrongAbiRenderer, |ast: &Arc<UOp>, name: Option<&str>| {
    let mut spec = ProgramSpec::new(
        name.unwrap_or("kernel").to_string(),
        "// wrong ABI".to_string(),
        DeviceSpec::Cpu,
        ast.clone(),
    );
    spec.buf_count = 1;
    Ok(spec)
});

impl Renderer for MockRenderer {
    fn supported_ops(&self) -> svod_ir::RendererOps {
        svod_ir::RendererOps::all()
    }

    fn render(&self, ast: &Arc<UOp>, name: Option<&str>) -> svod_device::Result<ProgramSpec> {
        Ok(ProgramSpec::new(
            name.unwrap_or("kernel").to_string(),
            "// mock source".to_string(),
            self.device.clone(),
            ast.clone(),
        ))
    }

    fn device(&self) -> &DeviceSpec {
        &self.device
    }
}

impl Renderer for LinearOnlyRenderer {
    fn supported_ops(&self) -> svod_ir::RendererOps {
        svod_ir::RendererOps::all()
    }

    fn render(&self, ast: &Arc<UOp>, name: Option<&str>) -> svod_device::Result<ProgramSpec> {
        assert!(matches!(ast.op(), Op::Linear(..)), "renderer should receive LINEAR stage");
        Ok(ProgramSpec::new(
            name.unwrap_or("kernel").to_string(),
            "// linear source".to_string(),
            self.device.clone(),
            ast.clone(),
        ))
    }

    fn device(&self) -> &DeviceSpec {
        &self.device
    }
}

fn mock() -> MockRenderer {
    MockRenderer { device: DeviceSpec::Cpu }
}

struct MockCompiler;

impl Compiler for MockCompiler {
    fn compile(&self, spec: &ProgramSpec) -> svod_device::Result<CompiledSpec> {
        let mut compiled =
            CompiledSpec::from_bytes(spec.name.clone(), vec![1, 2, 3], spec.ast.clone(), spec.abi.clone())?;
        compiled.global_size = spec.global_size.clone();
        compiled.local_size = spec.local_size.clone();
        Ok(compiled)
    }

    fn cache_key(&self) -> &str {
        "mock"
    }
}

/// Same cache key as `MockCompiler`, so reaching `compile` at all is the bug.
struct PanicCompiler;

impl Compiler for PanicCompiler {
    fn compile(&self, _spec: &ProgramSpec) -> svod_device::Result<CompiledSpec> {
        panic!("compiler should not be invoked when PROGRAM already has BINARY")
    }

    fn cache_key(&self) -> &str {
        "mock"
    }
}

struct OtherCompiler;

impl Compiler for OtherCompiler {
    fn compile(&self, _spec: &ProgramSpec) -> svod_device::Result<CompiledSpec> {
        panic!("compiler-key mismatch must be rejected before compilation")
    }

    fn cache_key(&self) -> &str {
        "other"
    }
}

/// Renders LINEAR by concatenating the opcodes of its INS list, and records the
/// order its pre-isel / isel matchers fire in.
#[derive(Clone)]
struct MockIsaRenderer {
    device: DeviceSpec,
    events: Arc<std::sync::Mutex<Vec<String>>>,
}

impl Renderer for MockIsaRenderer {
    fn supported_ops(&self) -> svod_ir::RendererOps {
        svod_ir::RendererOps::all()
    }

    fn render(&self, ast: &Arc<UOp>, name: Option<&str>) -> svod_device::Result<ProgramSpec> {
        let Op::Linear(ops::Linear { ops }) = ast.op() else { panic!("ISA renderer must receive LINEAR") };
        let source = ops
            .iter()
            .filter_map(|u| match u.op() {
                Op::Ins(ops::Ins { arg, .. }) => Some(arg.opcode.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        Ok(ProgramSpec::new(name.unwrap_or("kernel").to_string(), source, self.device.clone(), ast.clone()))
    }

    fn device(&self) -> &DeviceSpec {
        &self.device
    }

    fn pre_isel_matcher(&self) -> Option<TypedPatternMatcher<svod_device::isa::PreIselContext>> {
        let events = self.events.clone();
        let mut matcher: TypedPatternMatcher<svod_device::isa::PreIselContext> = TypedPatternMatcher::new();
        matcher.add(&[svod_ir::op::pattern_derived::OpKey::Const], move |u, ctx| {
            let Op::Const(value) = u.op() else { return RewriteResult::NoMatch };
            let temp = ctx.next_temp();
            events.lock().unwrap().push(format!("pre:{:?}:{temp}", value.0));
            RewriteResult::Rewritten(UOp::ins([], u.dtype(), InsArg::new(format!("imm.{:?}", value.0))))
        });
        Some(matcher)
    }

    fn isel_matcher(&self) -> Option<TypedPatternMatcher<svod_device::isa::IselContext>> {
        let events = self.events.clone();
        let mut matcher: TypedPatternMatcher<svod_device::isa::IselContext> = TypedPatternMatcher::new();
        matcher.add(&[svod_ir::op::pattern_derived::OpKey::Binary(BinaryOp::Add)], move |u, ctx| {
            let Op::Binary(BinaryOp::Add, lhs, rhs) = u.op() else { return RewriteResult::NoMatch };
            assert!(matches!(lhs.op(), Op::Ins(..)) && matches!(rhs.op(), Op::Ins(..)));
            assert_eq!(ctx.uses(lhs).len(), 1);
            assert_eq!(ctx.uses(rhs).len(), 1);
            let vreg = ctx.next_vreg();
            events.lock().unwrap().push(format!("isel:add:v{vreg}"));
            RewriteResult::Rewritten(UOp::ins(
                [lhs.clone(), rhs.clone()],
                u.dtype(),
                InsArg::with_attributes("mock.add", vec![("dst".into(), format!("v{vreg}"))]),
            ))
        });
        Some(matcher)
    }
}

// ── ABI slot numbering ─────────────────────────────────────────────────────

fn storage_param(slot: usize, addrspace: svod_ir::AddrSpace) -> Arc<UOp> {
    let shape = svod_ir::shape::shape_to_uop(&smallvec::smallvec![16usize.into()]);
    UOp::new(
        Op::Param(ops::Param { shape, arg: ParamArg::buffer(slot, DType::Float32, addrspace, None).into() }),
        DType::Float32,
    )
}

fn global(slot: usize) -> Arc<UOp> {
    UOp::param(slot, 16, DType::Float32, None)
}

fn var(name: &str) -> Arc<UOp> {
    UOp::variable(name.to_string(), 0, 16, DType::Int32)
}

fn one_global_one_scalar() -> Vec<Arc<UOp>> {
    vec![global(0), var("n")]
}

fn two_globals_two_scalars() -> Vec<Arc<UOp>> {
    vec![global(0), global(1), var("z_first").add(&var("a_second"))]
}

fn sparse_globals_two_scalars() -> Vec<Arc<UOp>> {
    vec![global(0), global(5), var("first").add(&var("second"))]
}

/// Local and register storage lives outside the kernel ABI, whether it arrives
/// as a PARAM or as scratch BUFFER, so it must not consume a scalar slot.
fn non_global_storage_and_one_scalar() -> Vec<Arc<UOp>> {
    vec![
        storage_param(0, svod_ir::AddrSpace::Global),
        storage_param(1, svod_ir::AddrSpace::Local),
        storage_param(2, svod_ir::AddrSpace::Reg),
        UOp::buffer(0, 16, DType::Float32, svod_ir::AddrSpace::Local, None),
        UOp::buffer(1, 16, DType::Float32, svod_ir::AddrSpace::Reg, None),
        var("n"),
    ]
}

/// Authored storage slots are preserved verbatim (including gaps); unassigned
/// scalars are then numbered densely after the highest authored storage slot,
/// in graph-walk order rather than by name.
#[test_case::test_case(one_global_one_scalar(), vec![0], &["n"], vec![1]; "one global reserves slot zero")]
#[test_case::test_case(two_globals_two_scalars(), vec![0, 1], &["z_first", "a_second"], vec![2, 3]; "scalars follow dense globals in walk order")]
#[test_case::test_case(sparse_globals_two_scalars(), vec![0, 5], &["first", "second"], vec![2, 3]; "sparse storage slots keep their gaps")]
#[test_case::test_case(non_global_storage_and_one_scalar(), vec![0, 1, 2], &["n"], vec![3]; "local and register storage stay out of the scalar namespace")]
fn abi_slots_are_assigned_after_authored_storage(
    sources: Vec<Arc<UOp>>,
    globals: Vec<usize>,
    scalar_names: &[&str],
    scalar_slots: Vec<usize>,
) {
    let program = program_from_sink(committed_sink(sources), DeviceSpec::Cpu).expect("final target graph");
    let Op::Program(ops::Program { sink, info, .. }) = program.op() else { panic!("expected PROGRAM") };

    assert_eq!(info.globals, globals);
    assert_eq!(var_slots(info), scalar_slots);
    assert_eq!(scalar_names.iter().map(|name| param_slot(sink, name)).collect::<Vec<_>>(), scalar_slots);
}

/// The C and LLVM signatures agree with the sparse ABI above, gaps included.
#[test]
fn renderers_emit_the_sparse_abi_verbatim() {
    let program =
        program_from_sink(committed_sink(sparse_globals_two_scalars()), DeviceSpec::Cpu).expect("sparse slots");
    let Op::Program(ops::Program { sink, .. }) = program.op() else { panic!("expected PROGRAM") };
    let linear = linear_of(sink);

    let c = crate::c::render(&linear, Some("sparse_abi")).expect("sparse C ABI");
    assert!(
        c.code.contains(
            "void sparse_abi(float* restrict data0, const int data2, const int data3, float* restrict data5)"
        ),
        "{}",
        c.code
    );
    let llvm = crate::llvm::text::render(&linear, Some("sparse_abi")).expect("sparse LLVM ABI");
    assert!(
        llvm.code.contains(
            "define void @sparse_abi(ptr noalias align 32 %data0, i32 %data2, i32 %data3, ptr noalias align 32 %data5)"
        ),
        "{}",
        llvm.code
    );
}

#[test]
fn reused_param_is_one_abi_argument() {
    let global = global(0);
    let program = program_from_sink(committed_sink(vec![global.clone(), global]), DeviceSpec::Cpu)
        .expect("the same PARAM reused is not a duplicate definition");
    let Op::Program(ops::Program { info, .. }) = program.op() else { panic!("PROGRAM") };
    assert_eq!(info.globals, vec![0]);
}

/// A slot claimed twice, or a scalar that never got one, is a typed error —
/// both when numbering a fresh SINK and when validating a prebuilt PROGRAM.
#[test]
fn conflicting_and_unassigned_abi_slots_are_typed_errors() {
    let duplicate = committed_sink(vec![global(0), scalar_param("n", 0)]);
    let err = program_from_sink(duplicate.clone(), DeviceSpec::Cpu).expect_err("duplicate authored ABI slots");
    assert!(matches!(err, svod_device::Error::DuplicateProgramParamSlot { slot: 0, .. }), "{err:?}");

    for (sink, expected_duplicate) in [(duplicate, true), (committed_sink(vec![var("n")]), false)] {
        let err = do_linearize(&program(sink, None, None, None)).expect_err("malformed prebuilt PROGRAM must fail");
        match (&err, expected_duplicate) {
            (svod_device::Error::DuplicateProgramParamSlot { .. }, true) => {}
            (svod_device::Error::UnassignedProgramParam { .. }, false) => {}
            _ => panic!("{err:?}"),
        }
    }
}

#[test]
fn unnamed_scalar_and_non_param_program_info_var_are_typed_errors() {
    let mut arg = ParamArg::variable("n".into(), DType::Int32, 0, 16);
    arg.name = None;
    let unnamed = UOp::new(Op::Param(ops::Param { shape: UOp::index_const(1), arg: arg.into() }), DType::Int32);
    let err = program_from_sink(committed_sink(vec![unnamed]), DeviceSpec::Cpu).expect_err("unnamed scalar must fail");
    assert!(matches!(err, svod_device::Error::ProgramAbiMismatch { .. }), "{err:?}");

    let sink = committed_sink(vec![UOp::const_(DType::Int32, 1.into())]);
    let mut info = svod_ir::ProgramInfo::from_sink(&sink, DeviceSpec::Cpu);
    info.vars.push(UOp::const_(DType::Int32, 2.into()));
    let malformed = UOp::program(sink, info, None, None, None);
    let err = do_linearize(&malformed).expect_err("non-PARAM ProgramInfo var must fail");
    assert!(matches!(err, svod_device::Error::ProgramAbiMismatch { .. }), "{err:?}");
}

/// `ProgramInfo.vars` is a descriptor, not a name list: mutating any field that
/// the ABI depends on must be caught even though the slot and name still match,
/// at every entry point that trusts a prebuilt PROGRAM.
#[test_case::test_case(|arg| arg.vmin_vmax = Some((
    svod_ir::ConstValueHash(svod_ir::ConstValue::Int(-1000)),
    svod_ir::ConstValueHash(svod_ir::ConstValue::Int(1000)),
)); "bounds")]
#[test_case::test_case(|arg| arg.multiple_of = Some(8); "multiple_of")]
#[test_case::test_case(|arg| arg.axis = Some(2); "axis")]
fn prebuilt_program_rejects_descriptor_equivalent_var_forgery(forge: fn(&mut ParamArg)) {
    let sink = committed_sink(vec![scalar_param("n", 0)]);
    let mut info = svod_ir::ProgramInfo::from_sink(&sink, DeviceSpec::Cpu);
    let Op::Param(ops::Param { shape, arg }) = info.vars[0].op() else { unreachable!() };
    let mut forged = arg.clone();
    forge(&mut forged);
    info.vars[0] = UOp::new(Op::Param(ops::Param { shape: shape.clone(), arg: forged }), DType::Int32);

    let prebuilt = UOp::program(sink.clone(), info.clone(), None, None, None);
    let staged = UOp::program(
        sink.clone(),
        info.clone(),
        Some(UOp::linear(svod_schedule::linearize_with_cfg(sink).into())),
        Some(UOp::source("// forged metadata".to_string())),
        None,
    );
    for err in [
        do_linearize(&prebuilt).expect_err("do_linearize must reject forged ProgramInfo.vars"),
        get_program(&prebuilt, &mock(), &MockCompiler, ProgramTarget::Linear)
            .expect_err("get_program must reject forged ProgramInfo.vars"),
        ProgramSpec::from_uop(&staged).expect_err("ProgramSpec must reject forged ProgramInfo.vars"),
    ] {
        match err {
            svod_device::Error::ProgramAbiMismatch { reason } => {
                assert!(reason.contains("ProgramInfo.vars"), "{reason}")
            }
            other => panic!("expected ProgramAbiMismatch, got {other:?}"),
        }
    }
}

#[test]
fn prebuilt_program_accepts_semantically_identical_nonidentical_var() {
    let sink = committed_sink(vec![scalar_param("n", 0)]);
    let mut info = svod_ir::ProgramInfo::from_sink(&sink, DeviceSpec::Cpu);
    let sink_var = info.vars[0].clone();
    let reconstructed = UOp::new(sink_var.op().clone(), sink_var.dtype()).with_metadata("detached variable");
    assert!(!Arc::ptr_eq(&sink_var, &reconstructed));
    assert_eq!(sink_var.content_hash, reconstructed.content_hash);
    info.vars[0] = reconstructed;

    do_linearize(&UOp::program(sink, info, None, None, None))
        .expect("validation must compare PARAM value semantics rather than allocation identity");
}

/// An opaque CALL body has its own PARAM namespace: the formal stays
/// unassigned inside the body and never enters the outer ABI. If it also
/// escapes into the executable graph, that is a typed error.
#[test]
fn opaque_function_formals_stay_out_of_the_outer_abi() {
    let formal = UOp::variable("formal".into(), 0, 16, DType::Int32);
    let outer = UOp::variable("outer".into(), 0, 16, DType::Int32);
    let call = UOp::sink(vec![formal.clone()]).call(smallvec::smallvec![outer], svod_ir::CallInfo::default());
    let sink = number_params(svod_schedule::add_control_flow(committed_sink(vec![call])))
        .expect("opaque formal must not enter PROGRAM ABI");
    let info = svod_ir::ProgramInfo::from_sink(&sink, DeviceSpec::Cpu);
    assert_eq!(param_slot(&sink, "outer"), 0);
    assert_eq!(info.vars.len(), 1);
    let formal_slot = sink
        .toposort_call_aware(true)
        .into_iter()
        .find_map(|node| match node.op() {
            Op::Param(ops::Param { arg, .. }) if arg.name.as_deref() == Some("formal") => Some(arg.slot),
            _ => None,
        })
        .expect("formal PARAM remains in opaque body");
    assert_eq!(formal_slot, usize::MAX);

    let leaked =
        UOp::sink(vec![formal.clone()]).call(smallvec::smallvec![UOp::index_const(1)], svod_ir::CallInfo::default());
    let leaked = svod_schedule::add_control_flow(committed_sink(vec![leaked, formal]));
    let err = executable_params(&leaked).expect_err("leaked opaque formal must fail");
    assert!(matches!(err, svod_device::Error::LeakedOpaqueProgramParam { .. }), "{err:?}");
}

#[test]
fn repeated_program_construction_has_identical_slots_and_identity() {
    let sink = committed_sink(vec![global(0), var("first").add(&var("second"))]);
    let a = program_from_sink(sink.clone(), DeviceSpec::Cpu).expect("first PROGRAM");
    let b = program_from_sink(sink, DeviceSpec::Cpu).expect("second PROGRAM");
    let Op::Program(ops::Program { info: ai, .. }) = a.op() else { unreachable!() };
    let Op::Program(ops::Program { info: bi, .. }) = b.op() else { unreachable!() };
    assert_eq!(ai, bi);
    assert_eq!(a.content_hash, b.content_hash);
}

/// One symbolic kernel end to end: the slot numbering, the rendered C
/// signature, the compiled metadata and the runtime kernarg packing must all
/// describe the same ABI.
#[test]
fn symbolic_program_render_compile_and_runtime_binding_share_canonical_abi() {
    let output = UOp::param(0, 1, DType::Float32, None);
    let n = UOp::variable("n".into(), 1, 16, DType::Int32);
    let index = UOp::index().buffer(output).indices(vec![UOp::index_const(0)]).call().expect("output index");
    let sink = committed_sink(vec![index.store(n.cast(DType::Float32))]);
    let program = program_from_sink(sink, DeviceSpec::Cpu).expect("PROGRAM");
    let Op::Program(ops::Program { info, .. }) = program.op() else { unreachable!() };
    assert_eq!(info.globals, vec![0]);
    assert_eq!(param_slot(&program, "n"), 1);

    let (rendered, spec) = do_render(&program, &CAbiRenderer).expect("C render");
    assert_eq!(spec.globals, vec![0]);
    assert_eq!(spec.var_names, vec!["n"]);
    assert_eq!(spec.buf_count, 1);
    assert!(spec.src.contains("void test(float* restrict data0, const int data1)"), "{}", spec.src);

    let (compiled_program, compiled) = do_compile(&rendered, &MockCompiler).expect("compile");
    assert_eq!(compiled.buf_count, 1);
    assert_eq!(compiled.var_names, vec!["n"]);
    assert!(matches!(compiled_program.op(), Op::Program(ops::Program { binary: Some(_), .. })));

    let mut kernargs = [0u8; 12];
    let written = svod_device::hcq::ClikeKernargLayout::pack_program(
        info,
        &spec.abi,
        &mut kernargs,
        &[0x1122_3344_5566_7788],
        &[7],
    )
    .expect("runtime kernarg binding");
    assert_eq!(written, 12);
    assert_eq!(&kernargs[..8], &0x1122_3344_5566_7788u64.to_le_bytes());
    assert_eq!(&kernargs[8..], &7i32.to_le_bytes());
}

/// Slots must be numbered from the post-`add_control_flow` walk: control flow
/// reorders the graph, so `ProgramInfo::from_sink` on the raw SINK still sees
/// every scalar unassigned.
#[test]
fn number_params_uses_walk_after_control_flow_insertion() {
    let r0 = UOp::range(UOp::index_const(4), 0);
    let r1 = UOp::range(UOp::index_const(4), 1);
    let end0 = var("first").add(&r0.cast(DType::Int32)).end(smallvec::smallvec![r0]);
    let end1 = var("second").add(&r1.cast(DType::Int32)).end(smallvec::smallvec![r1]);
    let raw = committed_sink(vec![end0, end1]);
    let premature = svod_ir::ProgramInfo::from_sink(&raw, DeviceSpec::Cpu);
    assert_eq!(var_slots(&premature), vec![usize::MAX, usize::MAX], "fixture must expose premature slots");

    let prepared = svod_schedule::add_control_flow(raw.clone());
    assert!(
        prepared.toposort().iter().any(|u| matches!(u.op(), Op::Range(ops::Range { deps, .. }) if !deps.is_empty()))
    );
    let expected_names: Vec<String> = prepared
        .toposort()
        .into_iter()
        .filter_map(|u| match u.op() {
            Op::Param(ops::Param { arg, .. }) if arg.slot == usize::MAX && arg.addrspace.is_none() => arg.name.clone(),
            _ => None,
        })
        .collect();

    let program = program_from_sink(raw, DeviceSpec::Cpu).expect("final target graph");
    let Op::Program(ops::Program { sink, info, .. }) = program.op() else { panic!("expected PROGRAM") };
    let actual_names: Vec<String> = info
        .vars
        .iter()
        .map(|u| match u.op() {
            Op::Param(ops::Param { arg, .. }) => arg.name.clone().unwrap(),
            _ => unreachable!(),
        })
        .collect();
    assert_eq!(actual_names, expected_names);
    assert_eq!(var_slots(info), vec![0, 1]);
    assert!(sink.toposort().iter().all(
        |u| !matches!(u.op(), Op::Param(ops::Param { arg, .. }) if arg.addrspace.is_none() && arg.slot == usize::MAX)
    ));
}

// ── staging ────────────────────────────────────────────────────────────────

#[test]
fn test_program_pipeline_sets_all_stages() {
    let sink = committed_sink(vec![UOp::native_const(1.0f32)]);
    let program = program_from_sink(sink.clone(), DeviceSpec::Cpu).expect("final target graph");
    let (program, rendered_spec) = do_render(&program, &mock()).expect("render stage");
    let (program, compiled) = do_compile(&program, &MockCompiler).expect("compile stage");
    let spec = ProgramSpec::from_uop(&program).expect("ProgramSpec::from_uop");

    let sources = program.op().children();
    assert_eq!(sources.len(), 4, "Tinygrad PROGRAM sources are SINK, LINEAR, SOURCE, BINARY");
    assert!(matches!(sources[0].op(), Op::Sink(..)));
    assert!(matches!(sources[1].op(), Op::Linear(..)));
    assert!(matches!(sources[2].op(), Op::Source(..)));
    assert!(matches!(sources[3].op(), Op::ProgramBinary(..)));

    assert_eq!(rendered_spec.name, "test");
    assert_eq!(spec.name, "test");
    assert_eq!(spec.src, "// mock source");
    assert_eq!(spec.ast.id, sink.id);
    assert_eq!(compiled.bytes, vec![1, 2, 3]);
    assert_eq!(compiled.name, "test");
    assert!(spec.globals.is_empty());
    assert!(spec.outs.is_empty());
    assert!(spec.ins.is_empty());
}

#[test]
fn test_do_render_uses_linear_stage_input() {
    let sink = committed_sink(vec![UOp::native_const(5.0f32)]);
    let program = program_from_sink(sink, DeviceSpec::Cpu).expect("final target graph");
    // The renderer asserts internally that it is handed LINEAR, not SINK.
    let (rendered_program, spec) =
        do_render(&program, &LinearOnlyRenderer { device: DeviceSpec::Cpu }).expect("render stage should succeed");

    assert_eq!(spec.name, "test");
    assert!(matches!(spec.ast.op(), Op::Sink(..)));
    let Op::Program(ops::Program { linear: Some(_), source: Some(_), .. }) = rendered_program.op() else {
        panic!("expected PROGRAM with LINEAR and SOURCE stages, got {:?}", rendered_program.op())
    };
}

#[test]
fn prebuilt_program_target_must_match_renderer() {
    let sink = committed_sink(vec![UOp::native_const(1i32)]);
    let program = program_from_sink(sink, DeviceSpec::Cpu).expect("PROGRAM");
    let renderer = LinearOnlyRenderer { device: DeviceSpec::Amd { device_id: 0 } };
    let err = do_render(&program, &renderer).expect_err("target mismatch must fail");
    assert!(matches!(err, svod_device::Error::ProgramTargetMismatch { .. }), "{err:?}");
}

#[test_case::test_case(&WrongAbiRenderer; "storage count with no descriptors")]
#[test_case::test_case(&ReversedStorageRenderer; "same-count descriptors in the wrong order")]
fn do_render_rejects_renderer_abi_disagreeing_with_program_info(renderer: &dyn Renderer) {
    let sink = committed_sink(vec![UOp::param(0, 1, DType::Float32, None), UOp::param(1, 1, DType::Int32, None)]);
    let program = program_from_sink(sink, DeviceSpec::Cpu).expect("PROGRAM");
    let err = do_render(&program, renderer).expect_err("renderer ABI disagreement must fail");
    assert!(matches!(err, svod_device::Error::ProgramAbiMismatch { .. }), "{err:?}");
}

/// The final SINK is verified before the line cleanup, so a malformed SINK is
/// caught whether the pipeline linearizes it itself or reuses a staged LINEAR.
#[test]
fn final_sink_verification_cannot_be_bypassed_by_a_staged_linear() {
    let valid = committed_sink(vec![UOp::native_const(5.0f32)]);
    let malformed = UOp::new(valid.op().clone(), DType::Float32);

    let err = program_from_sink(malformed.clone(), DeviceSpec::Cpu).expect_err("SINK dtype must be checked");
    assert!(format!("{err}").contains("SINK must be void"), "unexpected error: {err:?}");

    let staged = program(malformed, Some(linear_of(&valid)), None, None);
    let err = get_program(&staged, &mock(), &MockCompiler, ProgramTarget::Linear)
        .expect_err("retaining an existing LINEAR stage must still verify its final SINK");
    assert!(format!("{err}").contains("SINK must be void"), "unexpected error: {err:?}");
}

/// A kernel's text does not depend on the UOps the process minted before
/// it: its if-labels are numbered by the kernel's own counter, not by the
/// process-wide UOp id, so the same kernel built after a thousand other
/// UOps renders the same text and finds its compiled object again.
#[test]
fn a_kernels_text_does_not_depend_on_the_uops_minted_before_it() {
    let gated = || {
        let out_index = UOp::index()
            .buffer(UOp::param(0, 16, DType::Float32, None))
            .indices(vec![UOp::index_const(0)])
            .call()
            .expect("index");
        let store = out_index.store_gated(UOp::native_const(1.0f32), UOp::native_const(true));
        let program = program_from_sink(committed_sink(vec![store]), DeviceSpec::Cpu).expect("final target graph");
        let linearized = do_linearize(&program).expect("linearize");
        let Op::Program(ops::Program { linear: Some(linear), .. }) = linearized.op() else { panic!("LINEAR") };
        crate::llvm::text::render(linear, Some("gated")).expect("render").code
    };
    let first = gated();
    let minted: Vec<Arc<UOp>> = (0..1000).map(UOp::native_const).collect();
    let second = gated();
    assert!(first.contains("if_then_"), "{first}");
    assert_eq!(first, second, "{}", minted.len());
}

/// A gated STORE becomes IF / STORE / ENDIF during the cleanup rewrite, and the
/// linearized list is verified *after* that rewrite — otherwise `spec_program`'s
/// IF rule (which rejects a body-carrying IF) would never be reached.
#[test]
fn gated_store_linearizes_to_a_verified_if_block() {
    let out_index = UOp::index()
        .buffer(UOp::param(0, 16, DType::Float32, None))
        .indices(vec![UOp::index_const(0)])
        .call()
        .expect("index");
    let store = out_index.store_gated(UOp::native_const(1.0f32), UOp::native_const(true));
    let program = program_from_sink(committed_sink(vec![store]), DeviceSpec::Cpu).expect("final target graph");

    let linearized = do_linearize(&program).expect("linearize stage should succeed");
    let Op::Program(ops::Program { linear: Some(linear), .. }) = linearized.op() else {
        panic!("expected LINEAR stage")
    };
    let Op::Linear(ops::Linear { ops }) = linear.op() else { panic!("expected LINEAR payload") };

    assert!(ops.iter().any(|u| matches!(u.op(), Op::If(..))), "cleanup must inject IF");
    assert!(ops.iter().any(|u| matches!(u.op(), Op::EndIf(..))), "cleanup must inject ENDIF");
    assert!(
        ops.iter().any(
            |u| matches!(u.op(), Op::Store(ops::Store { index, gate: None, .. }) if matches!(index.op(), Op::Index(..)))
        ),
        "the gate must move onto the IF, off the STORE"
    );

    let bare_if = UOp::new(
        Op::If(ops::If { condition: UOp::native_const(true), body: smallvec::smallvec![UOp::native_const(0i32)] }),
        DType::Void,
    );
    assert!(svod_schedule::spec::type_verify_list(&[bare_if], &svod_schedule::spec::spec_program()).is_err());
}

#[test]
fn test_hand_lowered_final_rewrite_stays_invalid_free_through_linearize() {
    let lane = UOp::special(UOp::index_const(4), "gidx0".to_string());
    let valid = lane.try_cmplt(&UOp::index_const(3)).expect("validity condition");
    let guarded_index = UOp::try_where(valid, lane.clone(), UOp::invalid_marker()).expect("guarded index");
    let input = UOp::param(1, 4, DType::Float32, None);
    let output = UOp::param(0, 4, DType::Float32, None);
    let load_index = UOp::index().buffer(input).indices(vec![guarded_index]).call().expect("input index");
    let value = UOp::load().index(load_index).call();
    let store_index = UOp::index().buffer(output).indices(vec![lane]).call().expect("output index");
    let sink = UOp::sink_with_info(
        vec![store_index.store(value)],
        svod_ir::KernelInfo { opts_to_apply: Some(vec![]), ..Default::default() },
    );
    assert!(sink.toposort().iter().any(UOp::is_invalid_marker), "fixture must contain index validity");

    let optimizer_renderer = svod_schedule::OptimizerRenderer::amd_rdna3().with_rewrite_capabilities(
        svod_ir::RendererOps::all(),
        None,
        None,
    );
    let optimized = svod_schedule::optimize_kernel_with_config(
        sink,
        &optimizer_renderer,
        &svod_schedule::OptimizerConfig::default(),
    )
    .expect("hand-lowered final rewrite");
    assert!(
        optimized.toposort().iter().all(|u| !UOp::is_invalid_marker(u)),
        "mandatory final rewrite must remove Invalid from hand-lowered kernels"
    );

    let program = program_from_sink(optimized, DeviceSpec::Amd { device_id: 0 }).expect("final target graph");
    let linearized = do_linearize(&program).expect("PROGRAM -> LINEAR");
    let Op::Program(ops::Program { linear: Some(linear), .. }) = linearized.op() else {
        panic!("expected LINEAR stage")
    };
    let Op::Linear(ops::Linear { ops }) = linear.op() else { panic!("expected LINEAR op") };
    assert!(
        ops.iter().all(|u| !UOp::is_invalid_marker(u)),
        "stage-20-clean input must remain Invalid-free through PROGRAM -> LINEAR"
    );
}

// ── kernel naming ──────────────────────────────────────────────────────────

#[test]
fn test_structured_custom_name_wins_over_optimizer_shape_name() {
    let sink = UOp::sink_with_info(
        vec![UOp::noop()],
        svod_ir::KernelInfo {
            name: Some("flash_attention".to_string()),
            opts_to_apply: Some(vec![]),
            ..Default::default()
        },
    )
    .with_metadata(svod_schedule::optimizer::KernelInfo::new("E_L2L48", vec![], false));

    let program = program_from_sink(sink, DeviceSpec::Cpu).expect("program");
    let Op::Program(ops::Program { info, .. }) = program.op() else { panic!("expected PROGRAM") };
    assert_eq!(info.name, "flash_attention");
}

/// The structured name keeps whatever the kernel author wrote; sanitisation to
/// a legal identifier happens at the renderer boundary, in every backend.
#[test]
fn test_structured_symbolic_name_is_sanitized_at_renderer_boundary() {
    let sink = UOp::sink_with_info(
        vec![UOp::noop()],
        svod_ir::KernelInfo {
            name: Some("E_\x1b[31mL?\x1b[0mn6".to_string()),
            opts_to_apply: Some(vec![]),
            ..Default::default()
        },
    );

    for renderer in [&CAbiRenderer as &dyn Renderer, &LlvmAbiRenderer as &dyn Renderer] {
        let program = program_from_sink_with_renderer(sink.clone(), renderer).expect("program");
        let Op::Program(ops::Program { info, .. }) = program.op() else { panic!("expected PROGRAM") };
        assert_eq!(info.name, "E_\x1b[31mL?\x1b[0mn6");
        assert_eq!(info.function_name(), "E_L3Fn6");

        let (_, spec) = do_render(&program, renderer).expect("render sanitized PROGRAM");
        assert_eq!(spec.name, "E_L3Fn6");
        assert!(spec.src.contains("E_L3Fn6"), "{}", spec.src);
        assert!(!spec.src.contains('?') && !spec.src.contains('\x1b'), "{}", spec.src);
    }
}

// ── stage identity ─────────────────────────────────────────────────────────

/// A SOURCE or BINARY stage is trusted only when it carries the identity minted
/// for this exact (ProgramInfo, ABI, LINEAR) — so the staged nodes and their
/// parent PROGRAMs are never interned as the ones the pipeline produces.
#[test]
fn semantic_stage_identity_defeats_preinterned_children_and_parent_programs() {
    let initial = program_from_sink(committed_sink(vec![UOp::native_const(11.0f32)]), DeviceSpec::Cpu).unwrap();
    let linearized = do_linearize(&initial).unwrap();
    let Op::Program(ops::Program { sink, info, linear: Some(linear), .. }) = linearized.op() else { unreachable!() };

    let raw_source = UOp::source("// mock source".into());
    let raw_source_parent =
        UOp::program(sink.clone(), info.clone(), Some(linear.clone()), Some(raw_source.clone()), None);
    let abi = ProgramSpec::validate_program_param_abi(sink, info).unwrap();
    let valid_identity = svod_device::device::source_stage_identity(info, &abi, linear, "// mock source").unwrap();
    let different_source = UOp::source_with_identity(
        "// mock source".into(),
        svod_ir::SourceStageIdentity { entry_name: "different".into(), ..valid_identity },
    );
    let different_source_parent =
        UOp::program(sink.clone(), info.clone(), Some(linear.clone()), Some(different_source.clone()), None);

    let (rendered, _) = do_render(&initial, &mock()).unwrap();
    let Op::Program(ops::Program { source: Some(rendered_source), .. }) = rendered.op() else { unreachable!() };
    assert!(!Arc::ptr_eq(rendered_source, &raw_source));
    assert!(!Arc::ptr_eq(rendered_source, &different_source));
    assert!(!Arc::ptr_eq(&rendered, &raw_source_parent));
    assert!(!Arc::ptr_eq(&rendered, &different_source_parent));
    assert!(matches!(rendered_source.op(), Op::Source(ops::Source { identity: Some(_), .. })));

    let raw_binary = UOp::binary(vec![1, 2, 3]);
    let raw_binary_parent = UOp::program(
        sink.clone(),
        info.clone(),
        Some(linear.clone()),
        Some(rendered_source.clone()),
        Some(raw_binary.clone()),
    );
    let Op::Source(ops::Source { identity: Some(source_identity), .. }) = rendered_source.op() else { unreachable!() };
    let different_binary = UOp::binary_with_identity(
        vec![1, 2, 3],
        svod_device::device::binary_stage_identity(source_identity.as_ref().clone(), "other", &[1, 2, 3]),
    );
    let different_binary_parent = UOp::program(
        sink.clone(),
        info.clone(),
        Some(linear.clone()),
        Some(rendered_source.clone()),
        Some(different_binary.clone()),
    );

    let (compiled, _) = do_compile(&rendered, &MockCompiler).unwrap();
    let Op::Program(ops::Program { binary: Some(compiled_binary), .. }) = compiled.op() else { unreachable!() };
    assert!(!Arc::ptr_eq(compiled_binary, &raw_binary));
    assert!(!Arc::ptr_eq(compiled_binary, &different_binary));
    assert!(!Arc::ptr_eq(&compiled, &raw_binary_parent));
    assert!(!Arc::ptr_eq(&compiled, &different_binary_parent));
    assert!(matches!(compiled_binary.op(), Op::ProgramBinary(ops::ProgramBinary { identity: Some(_), .. })));
}

/// An unauthenticated SOURCE is rejected whether it was staged before rendering
/// or swapped in afterwards under an otherwise-valid PROGRAM.
#[test]
fn source_stages_without_a_matching_identity_are_rejected() {
    let sink = committed_sink(vec![UOp::native_const(8.0f32)]);
    let staged = program(sink.clone(), Some(linear_of(&sink)), Some(UOp::source("// stale source".to_string())), None);
    let err = do_render(&staged, &mock()).expect_err("render must reject SOURCE without renderer identity");
    assert!(matches!(err, svod_device::Error::ProgramStageMismatch { stage: "SOURCE", .. }), "{err:?}");

    let initial = program_from_sink(committed_sink(vec![UOp::native_const(4.0f32)]), DeviceSpec::Cpu).unwrap();
    let (rendered, _) = do_render(&initial, &mock()).unwrap();
    let Op::Program(ops::Program { sink, info, linear, .. }) = rendered.op() else { unreachable!() };
    let tampered = UOp::program(
        sink.clone(),
        info.clone(),
        linear.clone(),
        Some(UOp::source("// attacker-controlled source".into())),
        None,
    );
    let err =
        do_compile(&tampered, &MockCompiler).expect_err("same ProgramInfo must not authenticate arbitrary source");
    assert!(matches!(err, svod_device::Error::ProgramStageMismatch { stage: "SOURCE", .. }), "{err:?}");
}

/// A BINARY is reusable only when it was compiled from this exact signature by
/// this exact compiler.
#[test]
fn binary_stages_from_another_signature_or_compiler_are_rejected() {
    let render_compiled = |sink| {
        let program = program_from_sink(sink, DeviceSpec::Cpu).unwrap();
        do_render(&program, &CAbiRenderer).unwrap().0
    };
    let first = render_compiled(committed_sink(vec![UOp::param(0, 4, DType::Float32, None)]));
    let second = render_compiled(committed_sink(vec![
        UOp::param(0, 4, DType::Float32, None),
        UOp::param(5, 4, DType::Float32, None),
    ]));
    let (second, _) = do_compile(&second, &MockCompiler).unwrap();
    let Op::Program(ops::Program { binary: Some(other_binary), .. }) = second.op() else { unreachable!() };
    let Op::Program(ops::Program { sink, info, linear, source, .. }) = first.op() else { unreachable!() };
    let mismatched =
        UOp::program(sink.clone(), info.clone(), linear.clone(), source.clone(), Some(other_binary.clone()));

    let err = do_compile(&mismatched, &MockCompiler)
        .expect_err("binary identity from another signature must not be reusable");
    assert!(matches!(err, svod_device::Error::ProgramStageMismatch { stage: "BINARY", .. }), "{err:?}");

    let program = program_from_sink(committed_sink(vec![UOp::native_const(2.0f32)]), DeviceSpec::Cpu).unwrap();
    let (program, _) = do_render(&program, &mock()).unwrap();
    let (program, _) = do_compile(&program, &MockCompiler).unwrap();
    let err = do_compile(&program, &OtherCompiler).expect_err("binary from another compiler key must not be reused");
    assert!(matches!(err, svod_device::Error::ProgramStageMismatch { stage: "BINARY", .. }), "{err:?}");
}

/// Recompiling a PROGRAM that already carries BINARY returns the same node
/// without invoking the compiler, and its launch dims survive the round trip.
#[test]
fn test_do_compile_reuses_existing_binary_stage() {
    // Launch dims come from the SPECIAL UOps in the SINK (ProgramSpec::from_uop
    // ignores meta work sizes by design), so seed a `gidx0` with bound 4 to get
    // global_size == [4, 1, 1].
    let sink = committed_sink(vec![UOp::native_const(2.0f32), UOp::special(UOp::index_const(4), "gidx0".to_string())]);
    let program = program_from_sink(sink, DeviceSpec::Cpu).unwrap();
    let (program, _) = do_render(&program, &mock()).unwrap();
    let (program, _) = do_compile(&program, &MockCompiler).unwrap();

    let (compiled_program, compiled) = do_compile(&program, &PanicCompiler).expect("binary stage should be reused");

    assert!(Arc::ptr_eq(&compiled_program, &program));
    assert_eq!(compiled.name, "test");
    assert_eq!(compiled.bytes, vec![1, 2, 3]);
    assert_eq!(compiled.src.as_deref(), Some("// mock source"));
    assert!(compiled.var_names.is_empty());
    assert_eq!(compiled.buf_count, 0);
    let launch =
        ProgramSpec::resolve_launch_dims(&compiled.global_size, compiled.local_size.as_ref(), &Default::default())
            .expect("resolve launch dims");
    assert_eq!(launch.global_size, [4, 1, 1]);

    let rebuilt = ProgramSpec::from_uop(&compiled_program).expect("from_uop should support binary+metadata");
    assert_eq!(rebuilt.name, "test");
    assert_eq!(rebuilt.src, "// mock source");
}

fn no_source_stage() -> (Arc<UOp>, &'static str) {
    let sink = committed_sink(vec![UOp::native_const(1.0f32)]);
    (program_from_sink(sink, DeviceSpec::Cpu).expect("final target graph"), "PROGRAM has no SOURCE stage")
}

fn binary_stage_is_not_a_program_binary() -> (Arc<UOp>, &'static str) {
    let sink = committed_sink(vec![UOp::native_const(6.0f32)]);
    let linear = linear_of(&sink);
    let malformed = UOp::const_(DType::Float32, svod_ir::ConstValue::Float(1.0));
    (program(sink, Some(linear), Some(UOp::source("// source".to_string())), Some(malformed)), "ProgramBinary")
}

fn empty_source_stage() -> (Arc<UOp>, &'static str) {
    let sink = committed_sink(vec![UOp::native_const(7.0f32)]);
    let linear = linear_of(&sink);
    let mut meta = ProgramSpec::new("empty_source".to_string(), String::new(), DeviceSpec::Cpu, sink.clone());
    meta.set_var_names(vec!["N".to_string()]);
    meta.buf_count = 1;
    (program(sink, Some(linear), Some(UOp::source(String::new())), None).with_metadata(meta), "empty SOURCE stage")
}

#[test_case::test_case(no_source_stage; "nothing to compile")]
#[test_case::test_case(binary_stage_is_not_a_program_binary; "binary stage is not a ProgramBinary")]
#[test_case::test_case(empty_source_stage; "source stage is empty")]
fn do_compile_rejects_malformed_stages(build: fn() -> (Arc<UOp>, &'static str)) {
    let (program, reason) = build();
    let err = do_compile(&program, &MockCompiler).expect_err("malformed stages must fail to compile");
    assert!(format!("{err}").contains(reason), "unexpected error: {err}");
}

#[test]
fn test_do_render_rejects_program_with_existing_binary_stage() {
    let sink = committed_sink(vec![UOp::native_const(9.0f32)]);
    let program = program(sink.clone(), Some(linear_of(&sink)), None, Some(UOp::binary(vec![1, 2, 3])));

    let err = do_render(&program, &mock()).expect_err("render must reject programs that already have BINARY");
    assert!(format!("{err}").contains("stages must be SINK"), "unexpected error: {err}");
}

/// `get_program` drives a PROGRAM to BINARY from whichever stage it is in, and
/// the BINARY it lands on is reusable.
#[test]
fn get_program_advances_any_staged_program_to_a_reusable_binary() {
    let sink = committed_sink(vec![UOp::native_const(3.0f32)]);
    let stage1 = program(sink.clone(), Some(linear_of(&sink)), None, None);
    let stage2 = do_render(&program_from_sink(sink, DeviceSpec::Cpu).unwrap(), &mock()).unwrap().0;

    for staged in [stage1, stage2] {
        let advanced = get_program(&staged, &mock(), &MockCompiler, ProgramTarget::Binary)
            .expect("staged PROGRAM should advance to BINARY");
        let Op::Program(ops::Program { linear: Some(_), source: Some(_), binary: Some(_), .. }) = advanced.op() else {
            panic!("expected a fully staged PROGRAM, got {:?}", advanced.op())
        };
        let (_, compiled) =
            do_compile(&advanced, &PanicCompiler).expect("binary stage should be reusable after get_program");
        assert_eq!(compiled.bytes, vec![1, 2, 3]);
    }
}

#[test_case::test_case(
    program(committed_sink(vec![UOp::native_const(5.0f32)]), None, Some(UOp::source("// source without linear".into())), None),
    "malformed PROGRAM state"; "source staged without linear")]
#[test_case::test_case(committed_sink(vec![UOp::native_const(1.0f32)]), "expected PROGRAM input"; "bare sink")]
fn get_program_rejects_non_advanceable_input(input: Arc<UOp>, reason: &str) {
    let err = get_program(&input, &mock(), &MockCompiler, ProgramTarget::Binary).expect_err("must be rejected");
    assert!(format!("{err}").contains(reason), "unexpected error: {err}");
}

// ── instruction selection ──────────────────────────────────────────────────

#[test]
fn isa_selection_is_bottom_up_after_program_info_and_renders_program_source() {
    let events = Arc::new(std::sync::Mutex::new(Vec::new()));
    let renderer = MockIsaRenderer { device: DeviceSpec::Cpu, events: events.clone() };
    let sink = committed_sink(vec![UOp::const_(DType::Int32, 1.into()).add(&UOp::const_(DType::Int32, 2.into()))]);
    let expected_info =
        svod_ir::ProgramInfo::from_sink(&svod_schedule::add_control_flow(sink.clone()), DeviceSpec::Cpu);

    let program = program_from_sink_with_renderer(sink, &renderer).expect("ISA PROGRAM");
    let Op::Program(ops::Program { sink: selected, info, .. }) = program.op() else { panic!("expected PROGRAM") };
    assert_eq!(info.as_ref(), &expected_info, "ProgramInfo must be discovered before instruction selection");
    assert!(
        selected.toposort().iter().any(|u| matches!(u.op(), Op::Ins(ops::Ins { arg, .. }) if arg.opcode == "mock.add"))
    );
    svod_schedule::spec::type_verify(selected, &svod_schedule::spec::spec_program()).expect("INS is target-spec legal");
    assert_eq!(
        *events.lock().unwrap(),
        vec!["pre:Int(1):-1", "pre:Int(2):-2", "isel:add:v0"],
        "both ISA passes must walk children before parents",
    );

    let (rendered, spec) = do_render(&program, &renderer).expect("render selected instructions");
    assert_eq!(spec.src, "imm.Int(1)\nimm.Int(2)\nmock.add");
    let Op::Program(ops::Program { source: Some(source), .. }) = rendered.op() else {
        panic!("expected PROGRAM SOURCE")
    };
    assert!(matches!(source.op(), Op::Source(ops::Source { code, .. }) if code == &spec.src));
}
