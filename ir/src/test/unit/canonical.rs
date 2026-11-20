use svod_dtype::{AddrSpace, DType, DeviceSpec, ImageKind};

use crate::ops;
use crate::uop::canonical::CanonicalAxis;
use crate::{
    AxisId, AxisType, BinaryOp, BufferizeOpts, CallInfo, CanonicalArg, CanonicalConst, CanonicalDType, CanonicalGraph,
    CanonicalProgramValue, CanonicalShapeDim, ConstValue, InsArg, KernelInfo, Op, ProgramInfo, ReduceOp,
    RendererDevice, UOp, WmmaMetadata, WmmaUpcastAxes,
};

fn graph_json() -> String {
    let lhs = UOp::const_(DType::Int32, ConstValue::Int(7));
    let rhs = UOp::const_(DType::Int32, ConstValue::Int(2));
    let sub = UOp::new(Op::Binary(BinaryOp::Sub, lhs, rhs), DType::Int32);
    CanonicalGraph::from_root("tensor", &sub).unwrap().to_pretty_json().unwrap()
}

fn range_arg(axis: AxisId, axis_type: AxisType) -> CanonicalArg {
    let range = UOp::range_axis(UOp::index_const(8), axis, axis_type);
    CanonicalGraph::from_root("kernel_ast", &range).unwrap().nodes.last().unwrap().arg.clone()
}

// =========================================================================
// Graph shape and stability
// =========================================================================

/// The oracle is a pure function of graph content: interning ids never leak into it,
/// and it parses as ordinary JSON at the pinned schema version.
#[test]
fn canonical_json_is_stable_generic_json_free_of_runtime_ids() {
    let first = graph_json();
    let second = graph_json();
    assert_eq!(first, second);
    assert!(!first.contains("runtime_id"));

    let parsed: serde_json::Value = serde_json::from_str(&first).unwrap();
    assert_eq!(parsed["schema_version"], 8);
    assert_eq!(parsed["stage"], "tensor");
    assert_eq!(parsed["nodes"][2]["op"], "SUB");
}

/// The binary encoding carries the JSON oracle's content: equal graphs encode
/// identically across interning generations, a semantic change (another
/// constant) changes the bytes, and the layout is fixed-width little-endian
/// bincode so digests over it are machine-independent.
#[test]
fn canonical_binary_encoding_is_stable_and_content_sensitive() {
    let encode = |value: i64| {
        let lhs = UOp::const_(DType::Int32, ConstValue::Int(value));
        let rhs = UOp::const_(DType::Int32, ConstValue::Int(2));
        let sub = UOp::new(Op::Binary(BinaryOp::Sub, lhs, rhs), DType::Int32);
        let mut bytes = Vec::new();
        CanonicalGraph::from_root("tensor", &sub).unwrap().encode_into(&mut bytes).unwrap();
        bytes
    };
    let first = encode(7);
    assert_eq!(first, encode(7));
    assert_ne!(first, encode(8));
    // schema_version as u32, then the u64 length-prefixed stage.
    assert_eq!(first[..16], [8, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, b't', b'e', b'n', b's']);
}

#[test]
fn canonical_verbose_mode_is_explicit_and_not_in_default_oracle() {
    let root = UOp::native_const(7i32).rtag(Some(smallvec::smallvec![9]));
    let default = CanonicalGraph::from_root("tensor", &root).unwrap();
    let verbose = CanonicalGraph::from_root_verbose("tensor", &root).unwrap();

    assert!(default.verbose.is_none());
    assert!(!default.to_pretty_json().unwrap().contains("runtime_id"));
    let diagnostics = verbose.verbose.unwrap();
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(diagnostics[0].runtime_id, root.id);
    assert!(diagnostics[0].backend_dtype.contains("Int32"));
}

/// Shared subgraphs are emitted once and referenced by index, in source order, whether
/// the sharing is within one root or across several roots.
#[test]
fn canonical_graph_preserves_dag_sharing_and_source_order() {
    let shared = UOp::const_(DType::Int32, ConstValue::Int(3));
    let rhs = UOp::const_(DType::Int32, ConstValue::Int(1));
    let sub = UOp::new(Op::Binary(BinaryOp::Sub, shared.clone(), rhs), DType::Int32);
    let root = UOp::new(Op::Binary(BinaryOp::Add, sub, shared.clone()), DType::Int32);
    let graph = CanonicalGraph::from_root("tensor", &root).unwrap();

    assert_eq!(graph.nodes.len(), 4);
    assert_eq!(graph.roots, vec![3]);
    assert_eq!(graph.nodes[2].op, "SUB");
    assert_eq!(graph.nodes[2].src, vec![0, 1]);
    assert_eq!(graph.nodes[3].src, vec![2, 0]);

    let one = UOp::const_(DType::Int32, ConstValue::Int(1));
    let lhs = UOp::new(Op::Binary(BinaryOp::Add, one.clone(), one.clone()), DType::Int32);
    let rhs = UOp::new(Op::Binary(BinaryOp::Mul, one.clone(), one), DType::Int32);
    let graph = CanonicalGraph::from_roots("scheduled", &[lhs, rhs]).unwrap();

    assert_eq!(graph.roots, vec![1, 2]);
    assert_eq!(graph.nodes.len(), 3);
    assert_eq!(graph.nodes[1].src, vec![0, 0]);
    assert_eq!(graph.nodes[2].src, vec![0, 0]);
}

#[test]
fn canonical_symbolic_shape_never_serializes_null() {
    let variable = UOp::variable("n".into(), 1, 16, DType::Int32);
    let source = UOp::param(0, 16, DType::Float32, None);
    let shape = crate::shape::Shape::from_iter([crate::SInt::Symbolic(variable)]);
    let reshaped = source.try_reshape(&shape).unwrap();
    let json = CanonicalGraph::from_root("shape", &reshaped).unwrap().to_pretty_json().unwrap();
    assert!(json.contains("\"kind\": \"symbolic\""));
    assert!(!json.contains("\"node\": null"));
}

/// Identity-only and type-erased metadata has no content-addressable form, so the strict
/// oracle refuses it; verbose mode records a length and a content hash instead.
#[test]
fn canonical_rejects_identity_and_type_erased_metadata() {
    let call = UOp::native_const(1.0f32)
        .function(smallvec::smallvec![], CallInfo { grad_tag: Some("unstable".into()), ..Default::default() });
    let roots = [
        UOp::buffer_id(Some(7)),
        UOp::lunique(Some(9)),
        UOp::source("source".into()),
        UOp::binary(vec![1, 2, 3]),
        call,
    ];
    for root in roots {
        assert!(matches!(CanonicalGraph::from_root("strict", &root), Err(crate::Error::CanonicalSerialization { .. })));
    }

    let verbose = CanonicalGraph::from_root_verbose("strict", &UOp::binary(vec![1, 2, 3])).unwrap();
    assert!(matches!(verbose.nodes[0].arg, CanonicalArg::Binary { length: 3 }));
    assert!(verbose.verbose.unwrap()[0].content_xxh64.is_some());
}

// =========================================================================
// DTypes and constants
// =========================================================================

#[test]
fn canonical_float_constants_use_committed_semantic_bits() {
    let f64_negative_zero = UOp::const_(DType::Float64, ConstValue::Float(-0.0));
    let half = UOp::const_(DType::Float16, ConstValue::Float(1.0 / 123_008.0));
    let fp8 = UOp::vconst(vec![ConstValue::Float(1.0625), ConstValue::Float(-3.2)], DType::FP8E4M3);
    let graph = CanonicalGraph::from_roots("tensor", &[f64_negative_zero, half, fp8]).unwrap();

    assert_eq!(
        graph.nodes[0].arg,
        CanonicalArg::Const { value: CanonicalConst::Float { bits: "0x8000000000000000".to_string() } }
    );
    assert_eq!(
        graph.nodes[1].arg,
        CanonicalArg::Const { value: CanonicalConst::Float { bits: "0x3ee1000000000000".to_string() } }
    );
    assert_eq!(
        graph.nodes[2].arg,
        CanonicalArg::Constants {
            values: vec![
                CanonicalConst::Float { bits: "0x3ff0000000000000".to_string() },
                CanonicalConst::Float { bits: "0xc00a000000000000".to_string() },
            ]
        }
    );
}

#[test]
fn canonical_graph_preserves_vector_pointer_image_and_fnuz_dtype_metadata() {
    let vector = UOp::new(Op::Noop, DType::Float16.vec(4).unwrap());
    let pointer = UOp::new(Op::Noop, DType::Float32.ptr(Some(64), AddrSpace::Local).unwrap());
    let image = UOp::new(Op::Noop, DType::Image { kind: ImageKind::Half, shape: vec![8, 16, 4] });
    let fnuz = UOp::const_(DType::FP8E4M3FNUZ, ConstValue::Float(1.0));
    let graph = CanonicalGraph::from_roots("dtype", &[vector, pointer, image, fnuz]).unwrap();

    assert_eq!(graph.nodes[0].dtype, CanonicalDType::Vector { scalar: "float16".into(), count: 4 });
    assert_eq!(
        graph.nodes[1].dtype,
        CanonicalDType::Pointer {
            base: Box::new(CanonicalDType::Scalar { name: "float32".into() }),
            address_space: "local".into(),
            size: Some(64),
            count: 1,
        }
    );
    assert_eq!(graph.nodes[2].dtype, CanonicalDType::Image { image_kind: "half".into(), shape: vec![8, 16, 4] });
    assert_eq!(graph.nodes[3].dtype, CanonicalDType::Scalar { name: "fp8e4m3fnuz".to_string() });
}

// =========================================================================
// Params, buffers, ranges
// =========================================================================

#[test]
fn canonical_graph_records_structured_param_and_shape_source() {
    let param = UOp::param(4, 32, DType::Float16, None);
    let graph = CanonicalGraph::from_root("kernel_ast", &param).unwrap();

    assert_eq!(graph.nodes[1].src, vec![0]);
    assert_eq!(graph.nodes[1].shape, Some(vec![CanonicalShapeDim::Const { value: 32 }]));
    assert_eq!(graph.nodes[1].dtype, CanonicalDType::Scalar { name: "float16".to_string() });
    assert_eq!(
        graph.nodes[1].arg,
        CanonicalArg::Param {
            slot: 4,
            dtype: CanonicalDType::Scalar { name: "float16".to_string() },
            vmin_vmax: None,
            multiple_of: None,
            name: None,
            address_space: Some("global".to_string()),
            axis: None,
            device: None,
            volatile: false,
            tensor_map: None,
        }
    );
}

/// An authored high-bit slot is content, so it survives every stage; only the slot of a
/// buffer explicitly tagged schedule-local is stripped back to its namespace-free form.
#[test]
fn canonical_buffer_slots_are_stripped_only_when_marked_schedule_local() {
    let slot = (1usize << (usize::BITS - 1)) | 17;
    let buffer = UOp::buffer(slot, 4, DType::Float32, AddrSpace::Global, Some(DeviceSpec::Cpu));
    for stage in ["tensor", "kernel_ast", "scheduled"] {
        let graph = CanonicalGraph::from_root(stage, &buffer).unwrap();
        assert!(
            matches!(graph.nodes.last().unwrap().arg, CanonicalArg::Param { slot: actual, .. } if actual == slot as i128),
            "{stage}"
        );
    }

    let marked = buffer.with_tag(smallvec::smallvec![crate::uop::canonical::TAG_SCHEDULE_LOCAL_BUFFER]);
    let graph = CanonicalGraph::from_root("kernel_ast", &marked).unwrap();
    assert!(matches!(graph.nodes.last().unwrap().arg, CanonicalArg::Param { slot: 17, .. }));
}

#[test]
fn canonical_range_and_special_record_explicit_dtype_and_direct_extent() {
    let end = UOp::native_const(8i32);
    let range = UOp::range_axis_dtype(end.clone(), AxisId::Renumbered(0), AxisType::Global, DType::Int32);
    let special = UOp::special_dtype(end, "gidx0".to_string(), DType::Int32);
    let graph = CanonicalGraph::from_roots("kernel_ast", &[range, special]).unwrap();

    let range = graph.nodes.iter().find(|node| node.op == "RANGE").unwrap();
    let special = graph.nodes.iter().find(|node| node.op == "SPECIAL").unwrap();
    assert_eq!(range.dtype, CanonicalDType::Scalar { name: "int32".to_string() });
    assert_eq!(special.dtype, CanonicalDType::Scalar { name: "int32".to_string() });
    assert_eq!(range.src, vec![0]);
    assert_eq!(special.src, vec![0]);
}

/// The whole axis path is content, grouped-reduce-loop suffix included.
#[test]
fn canonical_range_records_axis_path_and_type() {
    for (axis, axis_type, expected) in [
        (
            AxisId::Renumbered(0),
            AxisType::Weak,
            CanonicalArg::Range { axis: vec![0], renumbered: true, axis_type: "WEAK".to_string() },
        ),
        (
            AxisId::Renumbered(4).child(1).child(0),
            AxisType::Reduce,
            CanonicalArg::Range { axis: vec![4, 1, 0], renumbered: true, axis_type: "REDUCE".to_string() },
        ),
        (
            AxisId::Renumbered(4).child(1).group_reduce_loop(),
            AxisType::Reduce,
            CanonicalArg::Range { axis: vec![4, 1, 2], renumbered: true, axis_type: "REDUCE".to_string() },
        ),
    ] {
        assert_eq!(range_arg(axis, axis_type), expected);
    }
}

/// `Unrenumbered(n)` and a one-element `UnrenumberedPath` denote the same axis, so Eq, Ord
/// and Hash must all agree; the renumbered/unrenumbered namespaces stay disjoint.
#[test]
fn axis_id_equality_ordering_and_hash_agree() {
    use std::hash::{BuildHasher, RandomState};

    let hasher = RandomState::new();
    for (left, right) in [
        (AxisId::Unrenumbered(3), AxisId::UnrenumberedPath(smallvec::smallvec![3])),
        (AxisId::Renumbered(3), AxisId::RenumberedPath(smallvec::smallvec![3])),
    ] {
        assert_eq!(left.cmp(&right) == std::cmp::Ordering::Equal, left == right);
        assert_eq!(hasher.hash_one(&left), hasher.hash_one(&right));
    }
    assert_ne!(AxisId::Unrenumbered(3), AxisId::Renumbered(3));
    assert_ne!(AxisId::Unrenumbered(3), AxisId::UnrenumberedPath(smallvec::smallvec![3, 0]));
}

#[test]
fn grouped_stage_axis_survives_hash_canonical_serde_and_tree() {
    let axis = AxisId::Renumbered(4).child(1).child(0);
    let opts = BufferizeOpts::local_for_axis(axis.clone());
    let encoded = serde_json::to_string(&opts).unwrap();
    assert_eq!(serde_json::from_str::<BufferizeOpts>(&encoded).unwrap(), opts);

    let compute = UOp::native_const(1.0f32);
    let grouped = UOp::stage(compute.clone(), vec![], opts);
    let scalar = UOp::stage(compute, vec![], BufferizeOpts::local_for_axis(AxisId::Renumbered(4)));
    assert!(!std::sync::Arc::ptr_eq(&grouped, &scalar));
    assert!(std::sync::Arc::ptr_eq(&grouped, &grouped.with_sources(grouped.op().sources().into_vec())));

    let graph = CanonicalGraph::from_root("grouped-stage", &grouped).unwrap();
    let node = graph.nodes.iter().find(|node| node.op == "STAGE").unwrap();
    assert_eq!(
        node.arg,
        CanonicalArg::Stage {
            device: None,
            local_axis: Some(CanonicalAxis { path: vec![4, 1, 0], renumbered: true }),
            address_space: "local".to_string(),
            removable: true,
        }
    );
    assert!(grouped.tree().contains("STAGE(local_axis=R4_1_0)"));
}

#[test]
fn ins_survives_hash_canonical_and_tree() {
    let source = UOp::const_(DType::Int32, ConstValue::Int(7));
    let first = UOp::ins(
        [source.clone()],
        DType::Int32,
        InsArg::with_attributes("mock.mov", vec![("z".into(), "2".into()), ("a".into(), "1".into())]),
    );
    let second = UOp::ins(
        [source],
        DType::Int32,
        InsArg::with_attributes("mock.mov", vec![("a".into(), "1".into()), ("z".into(), "2".into())]),
    );
    assert!(std::sync::Arc::ptr_eq(&first, &second), "INS metadata participates deterministically in hash consing");

    let graph = CanonicalGraph::from_root("isa", &first).unwrap();
    assert_eq!(graph.nodes.last().unwrap().op, "INS");
    assert!(matches!(
        &graph.nodes.last().unwrap().arg,
        CanonicalArg::Ins { opcode, attributes }
            if opcode == "mock.mov" && attributes == &vec![("a".into(), "1".into()), ("z".into(), "2".into())]
    ));
    assert!(first.tree().contains("INS(mock.mov)"));
}

// =========================================================================
// Typed op metadata
// =========================================================================

#[test]
fn canonical_call_sink_and_allreduce_metadata_are_typed() {
    let function = UOp::native_const(1.0f32).function(
        smallvec::smallvec![],
        CallInfo {
            grad_tag: None,
            name: Some("call_name".into()),
            precompile: true,
            precompile_backward: true,
            ..Default::default()
        },
    );
    let allreduce = UOp::new(
        Op::AllReduce(ops::AllReduce {
            src: UOp::native_const(2.0f32),
            device: DeviceSpec::Cuda { device_id: 2 },
            reduce_op: ReduceOp::Max,
        }),
        DType::Float32,
    );
    let sink = UOp::sink_with_info(
        vec![function, allreduce],
        KernelInfo {
            name: Some("metadata_kernel".into()),
            opts_to_apply: Some(vec![crate::Opt::upcast(0, 4)]),
            ..Default::default()
        },
    );
    let graph = CanonicalGraph::from_root("metadata", &sink).unwrap();

    assert!(graph.nodes.iter().any(|node| matches!(
        &node.arg,
        CanonicalArg::Call { grad_tag, metadata, name, precompile: true, precompile_backward: true }
            if grad_tag.is_none() && metadata.is_empty() && name.as_deref() == Some("call_name")
    )));
    assert!(graph.nodes.iter().any(|node| matches!(
        &node.arg,
        CanonicalArg::AllReduce { op, device } if op == "MAX" && device == "CUDA:2"
    )));
    assert!(matches!(
        &graph.nodes.last().unwrap().arg,
        CanonicalArg::Sink { name, opts_to_apply: Some(opts), .. }
            if name.as_deref() == Some("metadata_kernel") && opts == &vec![crate::Opt::upcast(0, 4)]
    ));
}

#[test]
fn canonical_wmma_metadata_preserves_axes_and_target() {
    let axis = AxisId::Renumbered(3).child(1);
    let metadata = WmmaMetadata {
        name: "wmma_fixture".into(),
        dims: (16, 8, 16),
        dtype_in: DType::Float16,
        dtype_out: DType::Float32,
        device: RendererDevice::CudaSm80,
        threads: 32,
        upcast_axes: Some(WmmaUpcastAxes {
            a: vec![(axis.clone(), 4)],
            b: vec![(AxisId::Renumbered(4), 2)],
            c: vec![(AxisId::Unrenumbered(5), 8)],
        }),
        reduce_axes: vec![axis],
    };
    let wmma = UOp::new(
        Op::Wmma(ops::Wmma {
            a: UOp::const_(DType::Float16, ConstValue::Float(1.0)),
            b: UOp::const_(DType::Float16, ConstValue::Float(2.0)),
            c: UOp::native_const(0.0f32),
            metadata: metadata.into(),
        }),
        DType::Float32,
    );
    let graph = CanonicalGraph::from_root("wmma", &wmma).unwrap();

    assert!(matches!(
        &graph.nodes.last().unwrap().arg,
        CanonicalArg::Wmma { dims: (16, 8, 16), device, threads: 32, upcast_a, .. }
            if device == "CUDA_SM80"
                && upcast_a[0].axis.path == vec![3, 1]
                && upcast_a[0].extent == 4
    ));
}

#[test]
fn canonical_program_info_preserves_launch_and_abi_metadata() {
    let input = UOp::param(3, 16, DType::Float32, None);
    let output = UOp::param(7, 16, DType::Float32, None);
    let index = UOp::index_const(0);
    let load = UOp::load().index(UOp::index().buffer(input).indices(vec![index.clone()]).call().unwrap()).call();
    let store = UOp::index().buffer(output).indices(vec![index]).call().unwrap().store(load);
    let n = UOp::variable("n".into(), 1, 16, DType::Int32);
    let sink = UOp::sink_with_info(
        vec![store, UOp::special(n.clone(), "gidx0".into())],
        KernelInfo { name: Some("non_default".into()), ..Default::default() },
    );
    let info = ProgramInfo {
        name: "non_default".into(),
        global_size: [n.clone(), UOp::index_const(2), UOp::index_const(1)],
        local_size: Some([UOp::index_const(4), UOp::index_const(1), UOp::index_const(1)]),
        vars: vec![n],
        globals: vec![3, 7],
        outs: vec![7],
        ins: vec![3],
        target: DeviceSpec::Cuda { device_id: 1 },
    };
    let graph = CanonicalGraph::from_root("program", &UOp::program(sink, info, None, None, None)).unwrap();
    let program = graph.nodes.last().unwrap();

    assert!(matches!(
        &program.arg,
        CanonicalArg::Program { name, global_size, local_size: Some(local_size), vars, globals, outs, ins, target }
            if name == "non_default"
                && matches!(global_size[0], CanonicalProgramValue::Node { .. })
                && global_size[1] == CanonicalProgramValue::Int { value: 2 }
                && local_size[0] == CanonicalProgramValue::Int { value: 4 }
                && vars.len() == 1
                && globals == &vec![3, 7]
                && outs == &vec![7]
                && ins == &vec![3]
                && target == "CUDA:1"
    ));
}

/// A symbolic launch bound is a real node in the topology, emitted before the PROGRAM
/// that references it, and shared with the `vars` list rather than duplicated.
#[test]
fn canonical_program_metadata_nodes_are_added_to_topology() {
    let variable = UOp::variable("n".into(), 1, 16, DType::Int32);
    let program = UOp::program(
        UOp::sink(vec![]),
        ProgramInfo {
            global_size: [variable.clone(), UOp::index_const(1), UOp::index_const(1)],
            vars: vec![variable],
            ..Default::default()
        },
        None,
        None,
        None,
    );
    let graph = CanonicalGraph::from_root("program", &program).unwrap();
    let program = graph.nodes.last().unwrap();
    let CanonicalArg::Program { global_size, vars, .. } = &program.arg else { panic!("expected PROGRAM arg") };
    let CanonicalProgramValue::Node { node } = global_size[0] else { panic!("expected symbolic launch node") };
    assert_eq!(vars, &vec![node]);
    assert!(node < program.id);
}

// =========================================================================
// Tinygrad source layouts
// =========================================================================

/// STACK is tinygrad's VECTORIZE: scalar dtype, lane count as a shape, sources in order,
/// at every stage.
#[test]
fn canonical_stack_has_scalar_dtype_shape_and_ordered_sources() {
    let stack = UOp::stack(smallvec::smallvec![UOp::native_const(1i32), UOp::native_const(2i32)]);
    for stage in ["tensor", "kernel_ast"] {
        let graph = CanonicalGraph::from_root(stage, &stack).unwrap();
        assert_eq!(graph.nodes[2].op, "STACK", "{stage}");
        assert_eq!(graph.nodes[2].dtype, CanonicalDType::Scalar { name: "int32".to_string() }, "{stage}");
        assert_eq!(graph.nodes[2].shape, Some(vec![CanonicalShapeDim::Const { value: 2 }]), "{stage}");
        assert_eq!(graph.nodes[2].src, vec![0, 1], "{stage}");
    }
}

#[test]
fn canonical_pad_uses_logical_begin_end_metadata() {
    let source = UOp::stack(smallvec::smallvec![
        UOp::native_const(1.0f32),
        UOp::native_const(2.0f32),
        UOp::native_const(3.0f32),
    ]);
    let padded = source.try_pad(&[(crate::SInt::from(1usize), crate::SInt::from(2usize))]).unwrap();
    let graph = CanonicalGraph::from_root("tensor", &padded).unwrap();
    let pad = graph.nodes.last().unwrap();

    assert_eq!(pad.arg, CanonicalArg::Pad { begin: vec![1], end: vec![2] });
    assert_eq!(pad.src.len(), 1, "representation-specific PAD extent UOps are typed metadata");
    assert_eq!(pad.shape, Some(vec![CanonicalShapeDim::Const { value: 6 }]));
}

/// LOAD/STORE take their address from INDEX alone — no redundant buffer source, and the
/// validity gate stays on INDEX until late gating moves it into the memory op's tail.
#[test]
fn canonical_memory_sources_follow_direct_tinygrad_layout() {
    let param = UOp::param(0, 16, DType::Float32, None);
    let index = UOp::index().buffer(param.clone()).indices(vec![UOp::index_const(3)]).call().unwrap();
    let graph = CanonicalGraph::from_root("tensor", &UOp::load().index(index).call()).unwrap();
    let index_node = graph.nodes.iter().find(|node| node.op == "INDEX").unwrap();
    let load_node = graph.nodes.iter().find(|node| node.op == "LOAD").unwrap();
    assert_eq!(index_node.src.len(), 2, "INDEX sources are param and scalar index");
    assert_eq!(load_node.src, vec![index_node.id], "LOAD has no redundant buffer source");

    let offset = UOp::index_const(3);
    let gate = offset.lt(&UOp::index_const(5));
    let valid = UOp::index().buffer(param).indices(vec![offset.valid(gate)]).call().unwrap();
    let graph = CanonicalGraph::from_root("tensor", &UOp::load().index(valid).call()).unwrap();
    let where_id = graph.nodes.iter().position(|node| node.op == "WHERE").unwrap();
    let index_node = graph.nodes.iter().find(|node| node.op == "INDEX").unwrap();
    let load_node = graph.nodes.iter().find(|node| node.op == "LOAD").unwrap();
    assert_eq!(index_node.src.len(), 2, "INDEX sources are param and validity-bearing index");
    assert_eq!(index_node.src[1], where_id);
    assert_eq!(load_node.src, vec![index_node.id], "validity remains on INDEX before late gating");

    let param = UOp::param(0, 4, DType::Float32, None);
    let index = UOp::index().buffer(param).indices(vec![UOp::index_const(0)]).call().unwrap();
    let gate = UOp::native_const(true);
    let load = UOp::load().index(index.clone()).alt(UOp::native_const(0.0f32)).gate(gate.clone()).call();
    let store = index.store_gated(UOp::native_const(1.0f32), gate);
    let graph = CanonicalGraph::from_roots("kernel_ast", &[load, store]).unwrap();
    let index_id = graph.nodes.iter().position(|node| node.op == "INDEX").unwrap();
    let load_node = graph.nodes.iter().find(|node| node.op == "LOAD").unwrap();
    let store_node = graph.nodes.iter().find(|node| node.op == "STORE").unwrap();
    assert_eq!(load_node.src.len(), 3, "LOAD sources are index, alt, gate");
    assert_eq!(load_node.src[0], index_id);
    assert_eq!(store_node.src.len(), 3, "STORE sources are index, value, gate");
    assert_eq!(store_node.src[0], index_id);
}

#[test]
fn canonical_copy_stores_device_as_metadata_not_source() {
    let src = UOp::new_buffer(DeviceSpec::Disk { path: "/tmp/model.onnx".into() }, 4, DType::UInt8);
    let copy = src.copy_to_device(DeviceSpec::Cpu);
    let graph = CanonicalGraph::from_root("tensor", &copy).unwrap();
    let node = graph.nodes.last().unwrap();

    assert_eq!(node.op, "COPY");
    assert_eq!(node.arg, CanonicalArg::Device { name: "CPU".to_string() });
    assert_eq!(node.src.len(), 1);
    assert!(graph.nodes.iter().all(|node| node.op != "DEVICE"));
}

// =========================================================================
// Reduce
// =========================================================================

/// `num_axes` counts leading shaped axes, is part of REDUCE's structural identity, and
/// may not exceed the source rank.
#[test]
fn canonical_reduce_records_and_hash_conses_leading_shaped_axis_count() {
    let src = UOp::stack(smallvec::smallvec![UOp::native_const(1.0f32), UOp::native_const(2.0f32)]);
    let scalar = src.reduce_with_num_axes(smallvec::smallvec![], ReduceOp::Add, 0);
    let horizontal = src.reduce_with_num_axes(smallvec::smallvec![], ReduceOp::Add, 1);

    let graph = CanonicalGraph::from_root("kernel_ast", &horizontal).unwrap();
    let node = graph.nodes.iter().find(|node| node.op == "REDUCE").unwrap();
    assert_eq!(node.arg, CanonicalArg::Reduce { op: "ADD".to_string(), axes: None, num_axes: Some(1) });
    assert_eq!(node.shape, Some(vec![]));

    assert!(!std::sync::Arc::ptr_eq(&scalar, &horizontal));
    let rebuilt = horizontal.with_sources(horizontal.op().sources().into_vec());
    assert!(std::sync::Arc::ptr_eq(&horizontal, &rebuilt));
    assert!(matches!(rebuilt.op(), Op::Reduce(ops::Reduce { num_axes: 1, .. })));

    let too_many = src.reduce_with_num_axes(smallvec::smallvec![], ReduceOp::Add, 2);
    assert!(matches!(too_many.shape(), Err(crate::Error::ReduceInvalidNumAxes { num_axes: 2, shape_dims: 1 })));
}
