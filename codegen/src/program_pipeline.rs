use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use svod_device::device::{
    AbiParamDescriptor, CompiledSpec, Compiler, ProgramSpec, Renderer, binary_stage_identity,
    minted_source_stage_identity, source_stage_identity, validate_binary_stage, validate_source_stage,
};
use svod_device::{Error, Result};
use svod_dtype::DeviceSpec;
use svod_ir::ops;
use svod_ir::{Op, ProgramInfo, UOp, UOpKey};
use svod_schedule::linearize::line_rewrite_cleanups;

type ProgramParts = (Arc<UOp>, ProgramInfo, Option<Arc<UOp>>, Option<Arc<UOp>>, Option<Arc<UOp>>);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgramTarget {
    Linear,
    Source,
    Binary,
}

fn invalid_program_state(details: impl Into<String>) -> Error {
    Error::Runtime { message: details.into() }
}

fn unpack_program(program: &Arc<UOp>) -> Result<ProgramParts> {
    let Op::Program(ops::Program { sink, info, linear, source, binary }) = program.op() else {
        return Err(invalid_program_state(format!("expected PROGRAM op, got {:?}", program.op())));
    };
    Ok((sink.clone(), info.as_ref().clone(), linear.clone(), source.clone(), binary.clone()))
}

/// Structural check on the PROGRAM stage tuple. Deliberately does *not* run
/// `type_verify` on the sink: `program_from_sink_impl` already verifies it once
/// against `spec_program`, and this runs at every stage transition
/// (do_linearize / do_render / do_compile / ...), so verifying here re-walked
/// the whole kernel five extra times.
fn validate_program_shape(program: &Arc<UOp>) -> Result<()> {
    let (sink, info, linear, source, binary) = unpack_program(program)?;

    let expected_arity = match (linear.is_some(), source.is_some(), binary.is_some()) {
        (false, false, false) => 1,
        (true, false, false) => 2,
        (true, true, false) => 3,
        (true, true, true) => 4,
        _ => {
            return Err(invalid_program_state(
                "malformed PROGRAM state: stages must be SINK[, LINEAR[, SOURCE[, BINARY]]]",
            ));
        }
    };
    if program.op().sources().len() != expected_arity {
        return Err(invalid_program_state(format!(
            "malformed PROGRAM state: expected {expected_arity} progressive sources, got {}",
            program.op().sources().len()
        )));
    }

    if !matches!(sink.op(), Op::Sink(..)) {
        return Err(invalid_program_state(format!("PROGRAM sink must be SINK op, got {:?}", sink.op())));
    }

    if let Some(linear) = &linear
        && !matches!(linear.op(), Op::Linear(..))
    {
        return Err(invalid_program_state(format!("PROGRAM linear stage must be LINEAR op, got {:?}", linear.op())));
    }

    if let Some(source) = &source
        && !matches!(source.op(), Op::Source(..))
    {
        return Err(invalid_program_state(format!("PROGRAM source stage must be SOURCE op, got {:?}", source.op())));
    }

    if let Some(binary) = &binary
        && !matches!(binary.op(), Op::ProgramBinary(..))
    {
        return Err(invalid_program_state(format!(
            "PROGRAM binary stage must be ProgramBinary op, got {:?}",
            binary.op()
        )));
    }

    validate_program_info(&sink, &info, None)?;

    Ok(())
}

/// Verify the linearized list, after `line_rewrite_cleanups` and any renderer
/// instruction-selection rewrites have run over it.
///
/// The sink is verified before linearization (`verify_final_sink`), which is
/// where the pin does it (`codegen/__init__.py:387`). That leaves the ops only
/// the list ever holds unchecked: the IF/STORE/ENDIF triple the cleanup
/// substitutes for a gated STORE never exists in the graph — `spec_program`'s
/// IF and ENDIF rules had no reachable input at all. tinygrad 1f8b24a6b ran
/// `type_verify(lst, program_spec)` here (`codegen/__init__.py:141`); keep both
/// gates.
fn verify_linear_list(nodes: &[Arc<UOp>]) -> Result<()> {
    if svod_schedule::spec::spec_enabled() {
        svod_schedule::spec::type_verify_list(nodes, &svod_schedule::spec::spec_program())
            .map_err(|error| invalid_program_state(error.to_string()))?;
    }
    Ok(())
}

fn verify_final_sink(sink: &Arc<UOp>) -> Result<()> {
    if svod_schedule::spec::spec_enabled() {
        svod_schedule::spec::type_verify(sink, &svod_schedule::spec::spec_program())
            .map_err(|error| invalid_program_state(error.to_string()))?;
    }
    Ok(())
}

fn preserve_program_context(new_program: Arc<UOp>, old_program: &Arc<UOp>) -> Arc<UOp> {
    new_program.rtag(old_program.tag().clone())
}

fn param_class(node: &Arc<UOp>) -> String {
    let Op::Param(ops::Param { arg, .. }) = node.op() else { return format!("{:?}", node.op()) };
    match arg.addrspace {
        None if arg.name.as_deref() == Some("_device_num") => "device scalar PARAM".to_string(),
        None => format!("scalar PARAM {}", arg.name.as_deref().unwrap_or("<unnamed>")),
        Some(svod_ir::AddrSpace::Global) => "global storage PARAM".to_string(),
        Some(svod_ir::AddrSpace::Local) => "local storage PARAM".to_string(),
        Some(svod_ir::AddrSpace::Reg) => "register storage PARAM".to_string(),
    }
}

fn validate_param_slots(nodes: &[Arc<UOp>], stage: &'static str) -> Result<()> {
    let mut occupied: HashMap<usize, Arc<UOp>> = HashMap::new();
    for node in nodes {
        let Op::Param(ops::Param { arg, .. }) = node.op() else { continue };
        if arg.slot == usize::MAX {
            return Err(Error::UnassignedProgramParam { stage, param: param_class(node) });
        }
        if let Some(first) = occupied.insert(arg.slot, node.clone()) {
            return Err(Error::DuplicateProgramParamSlot {
                slot: arg.slot,
                first: param_class(&first),
                second: param_class(node),
            });
        }
    }
    Ok(())
}

pub(crate) fn executable_params(sink: &Arc<UOp>) -> Result<Vec<Arc<UOp>>> {
    let executable = sink.toposort_call_aware(false);
    let executable_keys = executable.iter().map(|node| UOpKey(node.clone())).collect::<HashSet<_>>();
    for node in &executable {
        if let Op::Special(ops::Special { name, .. }) = node.op()
            && !matches!(name.chars().last().and_then(|axis| axis.to_digit(10)), Some(0..=2))
        {
            return Err(Error::ProgramAbiMismatch { reason: format!("invalid SPECIAL axis name {name:?}") });
        }
        let body = match node.op() {
            Op::Call(ops::Call { body, .. }) | Op::Function(ops::Function { body, .. }) => body,
            _ => continue,
        };
        for formal in body.toposort_call_aware(true) {
            if matches!(formal.op(), Op::Param(..)) && executable_keys.contains(&UOpKey(formal.clone())) {
                return Err(Error::LeakedOpaqueProgramParam { param: param_class(&formal) });
            }
        }
    }
    let mut params = executable.into_iter().filter(|node| matches!(node.op(), Op::Param(..))).collect::<Vec<_>>();
    validate_param_slots(&params, "final PROGRAM ABI validation")?;
    params.sort_by_key(|node| match node.op() {
        Op::Param(ops::Param { arg, .. }) => arg.slot,
        _ => usize::MAX,
    });
    Ok(params)
}

fn validate_program_info(
    sink: &Arc<UOp>,
    info: &ProgramInfo,
    expected_target: Option<&DeviceSpec>,
) -> Result<Vec<AbiParamDescriptor>> {
    if let Some(expected) = expected_target
        && &info.target != expected
    {
        return Err(Error::ProgramTargetMismatch { expected: expected.clone(), actual: info.target.clone() });
    }
    ProgramSpec::validate_program_param_abi(sink, info)
}

/// Port of Tinygrad's `pm_number_params`: assigned PARAM count is the initial
/// slot and every unassigned PARAM is numbered in final topological walk order.
/// Unlike the pinned implementation, occupied sparse slots are skipped rather
/// than collided with; authored slots are never renumbered.
pub(crate) fn number_params(sink: Arc<UOp>) -> Result<Arc<UOp>> {
    let all_nodes = sink.toposort_call_aware(true);
    let nodes = sink.toposort_call_aware(false);
    let mut occupied = HashSet::new();
    let mut authored = Vec::new();
    for node in &nodes {
        if let Op::Param(ops::Param { arg, .. }) = node.op()
            && arg.slot != usize::MAX
        {
            authored.push(node.clone());
            occupied.insert(arg.slot);
        }
    }
    validate_param_slots(&authored, "authored slot validation")?;

    // Pinned Tinygrad seeds numbering with the full authored topological count,
    // including opaque bodies, but its numbering rewrite preserves boundaries.
    let mut next_slot = all_nodes
        .iter()
        .filter(|node| matches!(node.op(), Op::Param(ops::Param { arg, .. }) if arg.slot != usize::MAX))
        .count();
    let mut replacements = HashMap::new();

    for node in nodes {
        let Op::Param(ops::Param { shape, arg }) = node.op() else { continue };
        if arg.slot != usize::MAX {
            continue;
        }
        while occupied.contains(&next_slot) {
            next_slot = next_slot.checked_add(1).ok_or_else(|| Error::UnassignedProgramParam {
                stage: "PARAM numbering slot exhaustion",
                param: param_class(&node),
            })?;
        }
        let mut numbered = arg.clone();
        numbered.slot = next_slot;
        occupied.insert(next_slot);
        next_slot = next_slot.checked_add(1).ok_or_else(|| Error::UnassignedProgramParam {
            stage: "PARAM numbering slot exhaustion",
            param: param_class(&node),
        })?;
        let replacement = UOp::new(Op::Param(ops::Param { shape: shape.clone(), arg: numbered }), node.dtype());
        replacements.insert(UOpKey(node), replacement);
    }

    let sink = sink.substitute_preserve_calls(&replacements);
    executable_params(&sink)?;
    Ok(sink)
}

fn rebuild_program(
    base_program: &Arc<UOp>,
    linear: Option<Arc<UOp>>,
    source: Option<Arc<UOp>>,
    binary: Option<Arc<UOp>>,
) -> Result<Arc<UOp>> {
    let (sink, info, _, _, _) = unpack_program(base_program)?;
    let rebuilt = UOp::program(sink, info, linear, source, binary);
    Ok(preserve_program_context(rebuilt, base_program))
}

/// Verify the final target graph and create initial PROGRAM(sink) state.
pub fn program_from_sink(sink: Arc<UOp>, device: DeviceSpec) -> Result<Arc<UOp>> {
    program_from_sink_impl(sink, device, None)
}

/// Renderer-aware PROGRAM boundary. ProgramInfo is deliberately discovered
/// before either ISA rewrite, matching Tinygrad's `do_to_program` ordering.
pub fn program_from_sink_with_renderer(sink: Arc<UOp>, renderer: &dyn Renderer) -> Result<Arc<UOp>> {
    program_from_sink_impl(sink, renderer.device().clone(), Some(renderer))
}

fn program_from_sink_impl(sink: Arc<UOp>, device: DeviceSpec, renderer: Option<&dyn Renderer>) -> Result<Arc<UOp>> {
    if matches!(sink.op(), Op::Program(..)) {
        return adopt_program(sink, &device, renderer);
    }
    let sink = if matches!(sink.op(), Op::Sink(..)) { sink } else { UOp::sink(vec![sink]) };
    // Hand-authored kernels carry their stable name in structured SINK info.
    // Optimizer metadata may also be present with an auto-generated shape name;
    // it is only the fallback for ordinary scheduled kernels.
    let kernel_name = match sink.op() {
        Op::Sink(ops::Sink { info: Some(info), .. }) if info.name.is_some() => info.name.clone(),
        _ => sink.metadata::<svod_schedule::optimizer::KernelInfo>().map(|info| info.function_name()),
    };
    // Tinygrad's target boundary is final rewrite -> implicit barriers -> CFG ->
    // PARAM numbering -> spec_program -> ProgramInfo -> PROGRAM/linearization.
    let sink = number_params(svod_schedule::add_control_flow(sink))?;
    verify_final_sink(&sink)?;
    let mut info = ProgramInfo::from_sink(&sink, device);
    if let Some(name) = kernel_name {
        info.name = name;
    }
    let sink = if let Some(renderer) = renderer {
        let sink = if let Some(matcher) = renderer.pre_isel_matcher() {
            svod_ir::rewrite::graph_rewrite_bottom_up(&matcher, sink, &mut svod_device::isa::PreIselContext::default())
        } else {
            sink
        };
        if let Some(matcher) = renderer.isel_matcher() {
            let mut context = svod_device::isa::IselContext::new(&sink);
            svod_ir::rewrite::graph_rewrite_bottom_up(&matcher, sink, &mut context)
        } else {
            sink
        }
    } else {
        sink
    };
    let program = UOp::program(sink, info, None, None, None);
    let (sink, info, _, _, _) = unpack_program(&program)?;
    validate_program_info(&sink, &info, renderer.map(Renderer::device))?;
    svod_ir::dump_canonical_stage("program", &program);
    Ok(program)
}

/// An author-built PROGRAM (`UOp::linear_program`) crosses the boundary as
/// is: no CFG, PARAM numbering or instruction selection rewrites it, so only
/// the checks run. Instruction selection would have to rewrite the LINEAR
/// list, which an ISA renderer cannot be handed unselected.
fn adopt_program(program: Arc<UOp>, device: &DeviceSpec, renderer: Option<&dyn Renderer>) -> Result<Arc<UOp>> {
    validate_program_shape(&program)?;
    let (sink, info, linear, _, _) = unpack_program(&program)?;
    if linear.is_some() && renderer.is_some_and(|renderer| renderer.isel_matcher().is_some()) {
        return Err(invalid_program_state("a pre-linearized PROGRAM cannot pass instruction selection"));
    }
    verify_final_sink(&sink)?;
    verify_program_linear(linear.as_ref())?;
    validate_program_info(&sink, &info, Some(device))?;
    Ok(program)
}

fn verify_program_linear(linear: Option<&Arc<UOp>>) -> Result<()> {
    match linear.map(|linear| linear.op()) {
        Some(Op::Linear(ops::Linear { ops })) => verify_linear_list(ops),
        _ => Ok(()),
    }
}

/// PROGRAM -> LINEAR stage.
pub fn do_linearize(program: &Arc<UOp>) -> Result<Arc<UOp>> {
    validate_program_shape(program)?;
    let (sink, _device, linear, source, binary) = unpack_program(program)?;
    if linear.is_some() {
        return Ok(program.clone());
    }

    let linear_ops = svod_schedule::linearize(sink.clone());

    if let Ok(dir) = std::env::var("SVOD_DUMP_LINEAR") {
        use std::io::Write;
        // Dump pre-linearization tree (toposort with scope info)
        let tree_path = format!("{dir}/tree_{}.txt", sink.id);
        if let Ok(mut f) = std::fs::File::create(&tree_path) {
            let topo = sink.toposort();
            for (i, u) in topo.iter().enumerate() {
                let scope = u.in_scope_ranges().iter().map(|k| k.to_string()).collect::<Vec<_>>().join(",");
                let _ = writeln!(f, "[{i:4}] id={} {:?} scope={{{scope}}}", u.id, std::mem::discriminant(u.op()));
            }
        }
        let path = format!("{dir}/linear_{}.txt", sink.id);
        if let Some(parent) = std::path::Path::new(&path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(mut f) = std::fs::File::create(&path) {
            for (i, u) in linear_ops.iter().enumerate() {
                let _ = writeln!(
                    f,
                    "[{i:4}] id={} {:?} scope={{{}}}",
                    u.id,
                    std::mem::discriminant(u.op()),
                    u.in_scope_ranges().iter().map(|k| k.to_string()).collect::<Vec<_>>().join(",")
                );
            }
        }
    }

    let linear_clean = line_rewrite_cleanups(linear_ops);
    verify_linear_list(&linear_clean)?;
    let linear_uop = UOp::linear(linear_clean.into());
    let linearized = rebuild_program(program, Some(linear_uop), source, binary)?;
    svod_ir::dump_canonical_stage("linearized", &linearized);
    Ok(linearized)
}

/// PROGRAM(+LINEAR) -> SOURCE stage via Renderer.
pub fn do_render(program: &Arc<UOp>, renderer: &dyn Renderer) -> Result<(Arc<UOp>, ProgramSpec)> {
    let (input_sink, input_info, _, _, _) = unpack_program(program)?;
    let expected_abi = validate_program_info(&input_sink, &input_info, Some(renderer.device()))?;
    let linearized = do_linearize(program)?;
    let (_sink, info, linear, source, binary) = unpack_program(&linearized)?;

    let linear_uop = linear.clone().ok_or_else(|| invalid_program_state("PROGRAM has no LINEAR stage"))?;

    if let Some(source_uop) = source {
        let Op::Source(ops::Source { code, .. }) = source_uop.op() else {
            return Err(invalid_program_state("PROGRAM source stage is not a SOURCE UOp"));
        };
        let expected = source_stage_identity(&info, &expected_abi, &linear_uop, code)?;
        validate_source_stage(&source_uop, &expected)?;
        let spec = ProgramSpec::from_uop(&linearized)?;
        return Ok((linearized, spec));
    }

    if binary.is_some() {
        return Err(invalid_program_state("PROGRAM BINARY stage cannot exist without SOURCE"));
    }

    let rendered_spec = renderer.render(&linear_uop, Some(&info.function_name()))?;
    let rendered_vars = rendered_spec
        .abi
        .iter()
        .filter(|param| !param.is_storage())
        .map(|param| param.name.clone().unwrap_or_default())
        .collect::<Vec<_>>();
    let rendered_buffers = rendered_spec.abi.iter().filter(|param| param.is_storage()).count();
    if rendered_spec.abi != expected_abi
        || rendered_spec.buf_count != rendered_buffers
        || rendered_spec.var_names != rendered_vars
        || rendered_spec.device != info.target
        || rendered_spec.name != info.function_name()
    {
        return Err(Error::ProgramAbiMismatch {
            reason: format!("ProgramInfo ABI is {expected_abi:?}; renderer reported {:?}", rendered_spec.abi),
        });
    }
    let source_identity = source_stage_identity(&info, &expected_abi, &linear_uop, &rendered_spec.src)?;
    let source_uop = UOp::source_with_identity(rendered_spec.src.clone(), source_identity.clone());
    validate_source_stage(&source_uop, &source_identity)?;
    let rendered = rebuild_program(&linearized, linear, Some(source_uop), None)?;
    let (_, _, _, rebuilt_source, _) = unpack_program(&rendered)?;
    validate_source_stage(
        rebuilt_source.as_ref().ok_or_else(|| invalid_program_state("rebuilt PROGRAM lost SOURCE stage"))?,
        &source_identity,
    )?;
    svod_ir::dump_canonical_stage("source", &rendered);
    let spec = ProgramSpec::from_uop(&rendered)?;
    Ok((rendered, spec))
}

/// PROGRAM(+SOURCE) -> BINARY stage via Compiler.
pub fn do_compile(program: &Arc<UOp>, compiler: &dyn Compiler) -> Result<(Arc<UOp>, CompiledSpec)> {
    validate_program_shape(program)?;
    let (sink, info, linear, source, binary) = unpack_program(program)?;
    let source = source.ok_or_else(|| invalid_program_state("PROGRAM has no SOURCE stage"))?;
    if matches!(source.op(), Op::Source(ops::Source { code, .. }) if code.is_empty()) {
        return Err(invalid_program_state("PROGRAM has empty SOURCE stage"));
    }

    let spec = ProgramSpec::from_uop(program)?;
    let expected_source = minted_source_stage_identity(&info, &spec.abi, &source)?;

    if let Some(binary_uop) = binary {
        let bytes = match binary_uop.op() {
            Op::ProgramBinary(ops::ProgramBinary { bytes, .. }) => bytes,
            _ => return Err(invalid_program_state("PROGRAM binary stage is not a ProgramBinary UOp")),
        };
        let expected_binary = binary_stage_identity(expected_source, compiler.cache_key(), bytes);
        let bytes = validate_binary_stage(&binary_uop, &expected_binary)?;
        let mut compiled = CompiledSpec::from_bytes(spec.name.clone(), bytes, sink, spec.abi.clone())?;
        compiled.src = Some(spec.src.clone());
        compiled.global_size = spec.global_size.clone();
        compiled.local_size = spec.local_size.clone();
        compiled.bind_program_stage(&info.target, compiler.cache_key(), expected_binary)?;
        return Ok((program.clone(), compiled));
    }

    let mut compiled = compiler.compile(&spec)?;
    svod_device::device::validate_abi_descriptors(&compiled.abi, compiled.buf_count, &compiled.var_names)?;
    if compiled.abi != spec.abi
        || compiled.buf_count != spec.buf_count
        || compiled.var_names != spec.var_names
        || compiled.name != spec.name
        || compiled.src.as_ref().is_some_and(|source| source != &spec.src)
    {
        return Err(Error::ProgramAbiMismatch {
            reason: format!(
                "compiler changed PROGRAM ABI: source abi={:?}, buffers={}, vars={:?}; compiled abi={:?}, buffers={}, vars={:?}",
                spec.abi, spec.buf_count, spec.var_names, compiled.abi, compiled.buf_count, compiled.var_names
            ),
        });
    }

    compiled.src = Some(spec.src.clone());
    let binary_identity = binary_stage_identity(expected_source, compiler.cache_key(), &compiled.bytes);
    compiled.bind_program_stage(&info.target, compiler.cache_key(), binary_identity.clone())?;
    let binary_uop = UOp::binary_with_identity(compiled.bytes.clone(), binary_identity.clone());
    validate_binary_stage(&binary_uop, &binary_identity)?;
    let compiled_program = rebuild_program(program, linear, Some(source), Some(binary_uop))?;
    let (_, _, _, rebuilt_source, rebuilt_binary) = unpack_program(&compiled_program)?;
    validate_source_stage(
        rebuilt_source.as_ref().ok_or_else(|| invalid_program_state("rebuilt PROGRAM lost SOURCE stage"))?,
        &binary_identity.source,
    )?;
    validate_binary_stage(
        rebuilt_binary.as_ref().ok_or_else(|| invalid_program_state("rebuilt PROGRAM lost BINARY stage"))?,
        &binary_identity,
    )?;
    ProgramSpec::from_uop(&compiled_program)?;
    svod_ir::dump_canonical_stage("binary", &compiled_program);
    Ok((compiled_program, compiled))
}

/// Rebuild a compiled specification from bytes produced by an isolated compile
/// worker for this exact SOURCE-stage PROGRAM.
pub fn adopt_compiled_bytes(program: &Arc<UOp>, compiler_key: &str, bytes: Vec<u8>) -> Result<CompiledSpec> {
    validate_program_shape(program)?;
    let (sink, info, _, source, binary) = unpack_program(program)?;
    if binary.is_some() {
        return Err(invalid_program_state("cannot adopt bytes into an already compiled PROGRAM"));
    }
    let source = source.ok_or_else(|| invalid_program_state("PROGRAM has no SOURCE stage"))?;
    let spec = ProgramSpec::from_uop(program)?;
    let expected_source = minted_source_stage_identity(&info, &spec.abi, &source)?;
    let identity = binary_stage_identity(expected_source, compiler_key, &bytes);
    let mut compiled = CompiledSpec::from_bytes(spec.name, bytes, sink, spec.abi)?;
    compiled.src = Some(spec.src);
    compiled.global_size = spec.global_size;
    compiled.local_size = spec.local_size;
    compiled.bind_program_stage(&info.target, compiler_key, identity)?;
    Ok(compiled)
}

/// Progressively advance SINK/PROGRAM input to a requested PROGRAM stage.
pub fn get_program(
    input: &Arc<UOp>,
    renderer: &dyn Renderer,
    compiler: &dyn Compiler,
    target: ProgramTarget,
) -> Result<Arc<UOp>> {
    let mut program = match input.op() {
        Op::Program(..) => {
            validate_program_shape(input)?;
            let (sink, info, linear, _, _) = unpack_program(input)?;
            // The only `spec_program` check on an externally supplied PROGRAM:
            // one that already carries a LINEAR stage skips `do_linearize`, so
            // neither its SINK nor its list would otherwise be verified here.
            verify_final_sink(&sink)?;
            verify_program_linear(linear.as_ref())?;
            validate_program_info(&sink, &info, Some(renderer.device()))?;
            input.clone()
        }
        other => return Err(invalid_program_state(format!("expected PROGRAM input, got {other:?}"))),
    };

    if matches!(target, ProgramTarget::Linear | ProgramTarget::Source | ProgramTarget::Binary) {
        let (_, _, linear, _, _) = unpack_program(&program)?;
        if linear.is_none() {
            program = do_linearize(&program)?;
        }
    }

    if matches!(target, ProgramTarget::Source | ProgramTarget::Binary) {
        let (rendered, _) = do_render(&program, renderer)?;
        program = rendered;
    }

    if matches!(target, ProgramTarget::Binary) {
        let (compiled, _) = do_compile(&program, compiler)?;
        program = compiled;
    }

    validate_program_shape(&program)?;
    if matches!(target, ProgramTarget::Source | ProgramTarget::Binary) {
        ProgramSpec::from_uop(&program)?;
    }
    Ok(program)
}
