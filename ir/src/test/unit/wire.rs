use std::sync::Arc;

use smallvec::{SmallVec, smallvec};

use crate::ops;
use crate::{
    AxisId, AxisType, BarrierFence, CallInfo, CanonicalArg, DType, DeviceSpec, KernelInfo, Op, Opt, OptimizerWireGraph,
    ParamArg, ReduceOp, RendererDevice, UOp, WmmaMetadata, WmmaUpcastAxes,
};

fn tagged(op: Op, dtype: DType, tag: &[usize]) -> Arc<UOp> {
    UOp::new_tagged(op, dtype, Some(SmallVec::from_slice(tag)))
}

#[test]
fn optimizer_wire_roundtrips_reduce_symbolic_wmma_multi_and_calls() {
    let symbol = UOp::new(Op::DefineVar(ops::DefineVar { name: "n".into(), min_val: 1, max_val: 4096 }), DType::Index);
    let shape =
        UOp::new(Op::Stack(ops::Stack { sources: smallvec![symbol.clone(), UOp::index_const(16)] }), DType::Index);
    let param = UOp::new(
        Op::Param(ops::Param {
            shape: shape.clone(),
            arg: ParamArg::buffer(0, DType::Float16, crate::AddrSpace::Global, Some(DeviceSpec::Cpu)).into(),
        }),
        DType::Float16,
    );
    let range = tagged(
        Op::Range(ops::Range {
            end: symbol,
            axis_id: AxisId::RenumberedPath(smallvec![2, 1]),
            axis_type: AxisType::Reduce,
            deps: smallvec![],
        }),
        DType::Index,
        &[7, 9],
    );
    let reduce = UOp::new(
        Op::Reduce(ops::Reduce {
            src: param.clone(),
            ranges: smallvec![range.clone()],
            reduce_op: ReduceOp::Add,
            num_axes: 1,
        }),
        DType::Float16,
    );
    let wmma = UOp::new(
        Op::Wmma(ops::Wmma {
            a: reduce.clone(),
            b: reduce.clone(),
            c: reduce.clone(),
            metadata: Box::new(WmmaMetadata {
                name: "wmma_test".into(),
                dims: (16, 16, 16),
                dtype_in: DType::Float16,
                dtype_out: DType::Float32,
                device: RendererDevice::CudaSm80,
                threads: 32,
                upcast_axes: Some(WmmaUpcastAxes {
                    a: vec![(AxisId::Renumbered(0), 8)],
                    b: vec![(AxisId::Renumbered(1), 4)],
                    c: vec![(AxisId::Renumbered(2), 4)],
                }),
                reduce_axes: vec![AxisId::RenumberedPath(smallvec![2, 1])],
            }),
        }),
        DType::Float32,
    );
    let function = UOp::new(
        Op::Function(ops::Function {
            body: UOp::new(Op::Tuple(ops::Tuple { src: smallvec![wmma.clone(), reduce] }), DType::Void),
            args: smallvec![param.clone()],
            info: Box::new(CallInfo { name: Some("inner".into()), ..Default::default() }),
        }),
        DType::Void,
    );
    let call = UOp::new(
        Op::Call(ops::Call {
            body: function,
            args: smallvec![UOp::new(Op::Multi(ops::Multi { src: param, axis: 1 }), DType::Float16)],
            info: Box::new(CallInfo { name: Some("outer".into()), precompile: true, ..Default::default() }),
        }),
        DType::Void,
    );
    let root = tagged(
        Op::Sink(ops::Sink {
            sources: smallvec![call],
            info: Some(Box::new(KernelInfo {
                opts_to_apply: Some(vec![Opt::upcast(0, 4)]),
                applied_opts: vec![Opt::local(0, 8)],
                dont_use_locals: true,
                name: Some("wire_kernel".into()),
            })),
        }),
        DType::Void,
        &[11],
    );
    let expected_hash = root.content_hash;
    let expected_tag = root.tag().clone();
    let expected_canonical = crate::CanonicalGraph::from_root("wire", &root).unwrap().to_pretty_json().unwrap();
    let encoded =
        bincode::serde::encode_to_vec(OptimizerWireGraph::from_root(&root).unwrap(), bincode::config::standard())
            .unwrap();
    drop(root);

    let (wire, _): (OptimizerWireGraph, usize) =
        bincode::serde::decode_from_slice(&encoded, bincode::config::standard()).unwrap();
    let decoded = wire.decode_root().unwrap();
    assert_eq!(decoded.content_hash, expected_hash);
    assert_eq!(decoded.tag(), &expected_tag);
    let decoded_range = decoded.toposort().into_iter().find(|node| matches!(node.op(), Op::Range(..))).unwrap();
    assert_eq!(decoded_range.tag().as_deref(), Some(&[7, 9][..]));
    assert_eq!(
        crate::CanonicalGraph::from_root("wire", &decoded).unwrap().to_pretty_json().unwrap(),
        expected_canonical
    );
}

#[test]
fn optimizer_wire_rejects_executable_stages_and_type_erased_metadata() {
    let sink = UOp::sink(vec![UOp::native_const(1i32)]);
    let program = UOp::program(sink.clone(), crate::ProgramInfo::default(), None, None, None);
    assert!(OptimizerWireGraph::from_root(&program).unwrap_err().to_string().contains("not a legal optimizer input"));

    let with_metadata = sink.with_metadata(String::from("opaque"));
    assert!(OptimizerWireGraph::from_root(&with_metadata).unwrap_err().to_string().contains("type-erased metadata"));
}

/// A barrier's fence is part of its identity: it survives the wire and a
/// source rewrite, and the canonical form names an LDS-only fence while a
/// graph barrier keeps Tinygrad's argument-free form.
#[test_case::test_case(BarrierFence::All, CanonicalArg::None; "graph")]
#[test_case::test_case(BarrierFence::Local, CanonicalArg::Name { value: "LOCAL".into() }; "lds only")]
fn barrier_fence_round_trips(fence: BarrierFence, canonical: CanonicalArg) {
    let barrier = UOp::native_const(1i32).fenced_barrier(smallvec![], fence);
    assert_ne!(barrier, UOp::native_const(1i32).fenced_barrier(smallvec![], other(fence)));
    let fence_of = |u: &Arc<UOp>| match u.op() {
        Op::Barrier(ops::Barrier { fence, .. }) => *fence,
        op => panic!("not a barrier: {op:?}"),
    };
    let decoded = OptimizerWireGraph::from_root(&barrier).unwrap().decode_root().unwrap();
    assert_eq!(fence_of(&decoded), fence);
    assert_eq!(fence_of(&barrier.with_sources(vec![UOp::native_const(2i32)])), fence);
    let arg = crate::CanonicalGraph::from_root("wire", &barrier).unwrap().nodes.last().unwrap().arg.clone();
    assert_eq!(arg, canonical);
}

fn other(fence: BarrierFence) -> BarrierFence {
    match fence {
        BarrierFence::All => BarrierFence::Local,
        BarrierFence::Local => BarrierFence::All,
    }
}
