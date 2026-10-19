//! The emitted instruction list: the values the lowering computes and where
//! it lists them.

use std::collections::HashMap;
use std::sync::Arc;

use svod_dtype::{AmdArch, DType};
use svod_ir::uop::eval::{eval_binary_op, eval_unary_op};
use svod_ir::{ConstValue, Op, UOp, ops};
use test_case::test_case;

use crate::atoms::{Costs, Device};
use crate::build::*;
use crate::ir::*;
use crate::layouts::WarpGrid;
use crate::lower::Lowering;
use crate::lower::emit::minimum;

use super::targets::{amd, families, linear};

/// The value of an expression over constants.
fn eval(u: &Arc<UOp>) -> ConstValue {
    match u.op() {
        Op::Const(v) => v.0,
        Op::Unary(op, x) => eval_unary_op(*op, eval(x)).expect("a defined unary"),
        Op::Binary(op, x, y) => eval_binary_op(*op, eval(x), eval(y)).expect("a defined binary"),
        other => panic!("not an expression over constants: {}", other.as_ref()),
    }
}

fn f32(v: f64) -> Arc<UOp> {
    UOp::const_(DType::Float32, ConstValue::Float(v))
}

/// A float minimum is exact where `a + b - max(a, b)` is not: at an
/// infinity (`inf - inf`), and where one side absorbs the other.
#[test_case(f64::INFINITY, 1.0, 1.0; "an infinity")]
#[test_case(f64::NEG_INFINITY, 1.0, f64::NEG_INFINITY; "a negative infinity")]
#[test_case(1e30, 1.0, 1.0; "an absorbed side")]
#[test_case(-3.0, 2.0, -3.0; "a negative side")]
#[test_case(f64::NAN, 2.0, 2.0; "a NaN is ignored as max ignores it")]
fn a_float_minimum_is_exact(a: f64, b: f64, want: f64) {
    for (x, y) in [(a, b), (b, a)] {
        let got = eval(&minimum(&f32(x), &f32(y)).unwrap());
        assert_eq!(got, ConstValue::Float(want), "min({x}, {y})");
    }
}

/// An integer minimum keeps its exact sum form.
#[test_case(-7, 3; "mixed signs")]
#[test_case(5, 5; "equal sides")]
fn an_integer_minimum_is_exact(a: i64, b: i64) {
    let c = |v| UOp::const_(DType::Int32, ConstValue::Int(v));
    assert_eq!(eval(&minimum(&c(a), &c(b)).unwrap()), ConstValue::Int(a.min(b)));
}

/// Lowered, a float tile's elementwise and reduced minimum add and subtract
/// nothing: the minimum is the negated maximum of the negations.
#[test_case(false; "elementwise")]
#[test_case(true; "reduced")]
fn a_lowered_float_minimum_adds_nothing(reduced: bool) {
    let tile = Shape::new(16, 32);
    let mut k = Kernel::new("min");
    k.warps(1);
    let a = k.param::<F32>("a", ParamKind::In, tile.elems());
    let b = k.param::<F32>("b", ParamKind::In, tile.elems());
    let out_shape = if reduced { Shape::new(tile.rows, 1) } else { tile };
    let y = k.param::<F32>("y", ParamKind::Out, out_shape.elems());
    let view = |k: &mut Kernel, p| k.view(p, 0, [tile.cols, 1], tile, [None, None]);
    let (va, vb) = (view(&mut k, a), view(&mut k, b));
    let (ra, rb) = (k.load(va), k.load(vb));
    let m = if reduced { k.reduce(ra, Axis::Row, ReduceOp::Min) } else { k.binary(ra, rb, BinaryOp::Min) };
    let out = k.view(y, 0, [out_shape.cols, 1], out_shape, [None, None]);
    k.store(out, m);
    let device = Device { target: crate::atoms::sm86(), costs: Costs::default() };
    let lowering = Lowering::uniform(&device, false, WarpGrid { rows: 1, cols: 1 });
    let nodes = linear(k.finish(), &lowering).toposort();
    let float = |u: &Arc<UOp>| u.dtype().is_float();
    let sums = nodes
        .iter()
        .filter(|u| matches!(u.op(), Op::Binary(svod_ir::BinaryOp::Add | svod_ir::BinaryOp::Sub, ..)) && float(u));
    assert_eq!(sums.count(), 0, "a float minimum lowered through a sum");
    assert!(nodes.iter().any(|u| matches!(u.op(), Op::Binary(svod_ir::BinaryOp::Max, ..)) && float(u)));
}

/// Every node of `nodes` by the ids of the nodes it reads.
fn users(nodes: &[Arc<UOp>]) -> HashMap<u64, Vec<&Arc<UOp>>> {
    let mut users: HashMap<u64, Vec<&Arc<UOp>>> = HashMap::new();
    for u in nodes {
        for src in u.op().sources() {
            users.entry(src.id).or_default().push(u);
        }
    }
    users
}

/// On RDNA4 an element a transposing load brings from past its view's
/// bounds reads zero, as every register load's does: each element received
/// is selected at its own coordinate, a lane receiving other lanes' rows.
/// Attention's bounded values and a GEMM's bounded residual load this way.
#[test_case("attention d64 causal #0"; "keys past the length")]
#[test_case("attention d64 cache"; "the appended keys")]
#[test_case("gemm gelu scale residual"; "a residual past the output")]
fn a_transposed_element_past_the_bounds_reads_zero(family: &str) {
    let launches = families(&amd(AmdArch::Gfx1201, 64));
    let (_, prog, lowering) = launches.into_iter().find(|(n, ..)| n.starts_with(family)).unwrap();
    let nodes = linear(prog, &lowering).toposort();
    let users = users(&nodes);
    let custom =
        |u: &Arc<UOp>, what: &str| matches!(u.op(), Op::Custom(ops::Custom { code, .. }) if code.contains(what));
    let loads: Vec<_> = nodes.iter().filter(|u| custom(u, "global.load.tr")).collect();
    assert!(!loads.is_empty(), "{family}: no transposing load");
    for load in loads {
        for elem in users[&load.id].iter().filter(|u| custom(u, "extractelement")) {
            let selected = users[&elem.id].iter().filter(|u| !matches!(u.op(), Op::Linear(..))).all(|u| {
                matches!(u.op(), Op::Ternary(svod_ir::TernaryOp::Where, _, t, f) if t.id == elem.id && f.id != elem.id)
            });
            assert!(selected, "{family}: a transposed element is used unselected");
        }
    }
}

/// A scalar read whose index may leave its parameter stays under the branch
/// that keeps the index in range, and one met again past the branch is read
/// again there, unguarded; a read the range analysis keeps inside the parameter is
/// listed where its index is, ahead of the branch.
#[test_case(8, true; "an index the branch keeps in range")]
#[test_case(3, false; "an index always in range")]
fn a_guarded_scalar_read_stays_under_its_guard(max: i64, guarded: bool) {
    let tile = Shape::new(16, 32);
    let mut k = Kernel::new("guarded");
    k.warps(1);
    let lens = k.param::<I32>("lens", ParamKind::In, 4);
    let y = k.param::<F32>("y", ParamKind::Out, 2 * tile.elems());
    let n = k.var("n", 0, max);
    let len = k.load_scalar(lens, n.clone());
    let at = |k: &mut Kernel, i: usize| k.view(y, i * tile.elems(), [tile.cols, 1], tile, [None, None]);
    k.if_(
        n.lt(4),
        |k| {
            let v = k.splat::<F32>(tile, len.clone());
            let out = at(k, 0);
            k.store(out, v);
        },
        |_| {},
    );
    let v = k.splat::<F32>(tile, len);
    let out = at(&mut k, 1);
    k.store(out, v);
    let device = Device { target: crate::atoms::sm86(), costs: Costs::default() };
    let lowering = Lowering::uniform(&device, false, WarpGrid { rows: 1, cols: 1 });
    let linear = linear(k.finish(), &lowering);
    let Op::Linear(ops::Linear { ops: list }) = linear.op() else { panic!("a linear list") };
    let at = |f: &dyn Fn(&Arc<UOp>) -> bool| (0..list.len()).filter(|&i| f(&list[i])).collect::<Vec<_>>();
    let opened = at(&|u| matches!(u.op(), Op::If { .. }));
    let closed = at(&|u| matches!(u.op(), Op::EndIf { .. }));
    let reads = at(&|u| matches!(u.op(), Op::Load { .. }) && u.dtype() == DType::Int32);
    let under = |i: usize| opened.iter().zip(&closed).any(|(&o, &c)| o < i && i < c);
    if guarded {
        assert_eq!(reads.len(), 2, "a read under the branch and one past it");
        assert_eq!(reads.iter().filter(|&&i| under(i)).count(), 1, "reads at {reads:?}, branch {opened:?}..{closed:?}");
    } else {
        assert_eq!(reads.len(), 1, "one read, shared");
        assert!(reads[0] < opened[0], "read at {reads:?}, branch at {opened:?}");
    }
}

/// A tile re-held through the relayout scratch lowers in a program that
/// interns no zero: the scratch is addressed from its start, not at an
/// offset scalar of the program.
#[test]
fn a_relayout_through_shared_memory_needs_no_zero_of_the_program() {
    use crate::layouts::{Relayout, chunked};
    let (tile, lanes) = (Shape::new(16, 32), 32);
    let [from, to] = [Axis::Col, Axis::Row].map(|walk| chunked(tile, 4, 1, lanes, walk).expect("a chunked layout"));
    let costs = super::lane_moves_dear();
    assert_eq!(from.relayout(&to, 1, lanes, &costs), Relayout::ViaSmem);
    let mut k = Kernel::new("relayout");
    k.warps(1);
    let x = k.param::<F32>("x", ParamKind::In, 2 * tile.elems());
    let y = k.param::<F32>("y", ParamKind::Out, 2 * tile.elems());
    let src = k.view(x, tile.elems(), [tile.cols, 1], tile, [None, None]);
    let a = k.load(src);
    k.pin(a, from);
    let b = k.unary(a, UnaryOp::Neg);
    k.pin(b, to);
    let dst = k.view(y, tile.elems(), [tile.cols, 1], tile, [None, None]);
    k.store(dst, b);
    let prog = k.finish();
    assert!(!prog.scalars.contains(&Scalar::Const(0)), "the program interns a zero");
    let device = Device { target: crate::atoms::sm86(), costs };
    let lowering = Lowering::uniform(&device, false, WarpGrid { rows: 1, cols: 1 });
    let nodes = linear(prog, &lowering).toposort();
    assert!(nodes.iter().any(|u| matches!(u.op(), Op::Barrier { .. })), "no round trip through shared memory");
}
