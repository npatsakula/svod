//! The emitted instruction list: the values the lowering computes and where
//! it lists them.

use std::collections::HashMap;
use std::sync::Arc;

use svod_dtype::{AmdArch, DType};
use svod_ir::uop::eval::{eval_binary_op, eval_unary_op};
use svod_ir::{ConstValue, Op, UOp, ops};
use test_case::test_case;

use crate::atoms::{Core, Costs, Device, Target};
use crate::build::*;
use crate::ir::*;
use crate::kernels::Batch;
use crate::kernels::gemm::{Epilogue, GemmCfg, GemmSpec, gemm};
use crate::kernels::rows::{Norm, NormCfg, NormSpec, norm};
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

/// An integer minimum is exact.
#[test_case(-7, 3; "mixed signs")]
#[test_case(5, 5; "equal sides")]
fn an_integer_minimum_is_exact(a: i64, b: i64) {
    let c = |v| UOp::const_(DType::Int32, ConstValue::Int(v));
    assert_eq!(eval(&minimum(&c(a), &c(b)).unwrap()), ConstValue::Int(a.min(b)));
}

/// An integer minimum's range is the minimum of its sides' ranges, plain and
/// sound: no wider than either side, as an index bounded by one side is.
#[test_case((0, 9), (3, 5); "one range inside the other")]
#[test_case((0, 9), (10, 20); "disjoint")]
#[test_case((-4, 6), (2, 30); "overlapping")]
fn an_integer_minimum_keeps_its_range_exact(a: (i64, i64), b: (i64, i64)) {
    use svod_ir::uop::cached_property::CachedProperty;
    use svod_ir::uop::properties::SoundVminVmaxProperty;
    let x = UOp::scalar_param(0, Some("x".into()), DType::Int32, a.0, a.1);
    let y = UOp::scalar_param(1, Some("y".into()), DType::Int32, b.0, b.1);
    let m = minimum(&x, &y).unwrap();
    let want = (ConstValue::Int(a.0.min(b.0)), ConstValue::Int(a.1.min(b.1)));
    assert_eq!((*m.vmin(), *m.vmax()), want, "plain range");
    assert_eq!(*SoundVminVmaxProperty::get(&m), Some(want), "sound range");
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
    let lowering = Lowering::new(&device);
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
/// bounds reads zero, as every register load's does: each 16-byte row run
/// is one lane's address, so a lane whose run lies outside the bounds
/// addresses a run of zeros (not the operand's first elements) and nothing
/// the load returns is selected after it. Attention's bounded values and a
/// GEMM's bounded residual load this way.
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
        // The address the load reads: the operand's run inside the bounds, else zeros.
        let [address] = &load.op().sources()[..] else { panic!("{family}: a load of one address") };
        assert!(custom(address, "@svod_tr_zeros"), "{family}: outside the bounds a lane reads the operand");
        let [_, operand] = &address.op().sources()[..] else { panic!("{family}: a choice of two addresses") };
        let index = operand.op().sources().into_iter().next().expect("a cast address");
        assert!(matches!(index.op(), Op::Index(..)), "{family}: inside the bounds the load reads the operand");
        let elems = users[&load.id].iter().filter(|u| custom(u, "extractelement"));
        let selected = elems.flat_map(|e| &users[&e.id]).any(|u| matches!(u.op(), Op::Ternary(..)));
        assert!(!selected, "{family}: a transposed element is selected after its load");
    }
}

/// A scalar read whose index may leave its parameter stays under the branch
/// that keeps the index in range, and one met again past the branch is read
/// again there, unguarded; a read the range analysis keeps inside the parameter is
/// listed where its index is, ahead of the branch, a truncating quotient's too.
#[test_case(8, 1, true; "an index the branch keeps in range")]
#[test_case(3, 1, false; "an index always in range")]
#[test_case(7, 2, false; "a halved index always in range")]
fn a_guarded_scalar_read_stays_under_its_guard(max: i64, div: i64, guarded: bool) {
    let tile = Shape::new(16, 32);
    let mut k = Kernel::new("guarded");
    k.warps(1);
    let lens = k.param::<I32>("lens", ParamKind::In, 4);
    let y = k.param::<F32>("y", ParamKind::Out, 2 * tile.elems());
    let n = k.var("n", 0, max);
    let index = if div > 1 { n.clone() / div } else { n.clone() };
    let len = k.load_scalar(lens, index);
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
    let lowering = Lowering::new(&device);
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

/// A loop whose extent may fall below zero runs no trip, as the interpreter
/// runs none: the backend compares a range's counter unsigned, so a negative
/// end would run billions of trips (a listing's share of steps starting
/// past the last read 8 GB before its scores on sm_86), and the emitter
/// floors an extent the bounds let below zero. One the bounds keep at or
/// above zero is listed as it is: a share's count less one where it is
/// positive (a register-staged ring's steady trips, `x − [0 < x]`) is
/// bounded as such, not a step under the count's own low.
#[test_case(4, false, true; "a share past the last steps")]
#[test_case(9, false, false; "every share within the steps")]
#[test_case(9, true, false; "a share's steady trips")]
fn a_loop_extent_below_zero_runs_no_trip(steps: i64, steady: bool, floored: bool) {
    let tile = Shape::new(16, 32);
    let mut k = Kernel::new("shares");
    k.warps(1);
    let y = k.param::<F32>("y", ParamKind::Out, 4 * tile.elems());
    k.grid([Sc::from(4), Sc::from(1), Sc::from(1)]);
    // A share of three steps each: the fourth starts at nine.
    let count = (Sc::from(steps) - k.block(0) * 3).min(3);
    let mut count = Sc::Id(k.intern(count));
    if steady {
        count = count.clone() - Sc::from(0).lt(count);
    }
    let zero = k.zeros::<F32>(tile);
    let one = k.fill::<F32>(tile, Const::Float(1.0));
    let [sum] = k.loop_(count, [zero], |k, _, [sum]| [k.binary(sum, one, BinaryOp::Add)]);
    let at = k.block(0) * tile.elems();
    let out = k.view(y, at, [tile.cols, 1], tile, [None, None]);
    k.store(out, sum);
    let device = Device { target: crate::atoms::sm86(), costs: Costs::default() };
    let linear = linear(k.finish(), &Lowering::new(&device));
    let Op::Linear(ops::Linear { ops: list }) = linear.op() else { panic!("a linear list") };
    let ends: Vec<&Arc<UOp>> =
        list.iter().filter_map(|u| if let Op::Range(r) = u.op() { Some(&r.end) } else { None }).collect();
    let [end] = ends[..] else { panic!("one loop, {} ranges", ends.len()) };
    let zero = |u: &Arc<UOp>| matches!(u.op(), Op::Const(c) if c.0 == ConstValue::Int(0));
    let max0 = |u: &Arc<UOp>| matches!(u.op(), Op::Binary(svod_ir::BinaryOp::Max, _, c) if zero(c));
    let end = if let Op::Cast(..) = end.op() { end.op().sources()[0].clone() } else { end.clone() };
    assert_eq!(max0(&end), floored, "range end {:?}", end.op().as_ref());
}

/// A tile re-held through the relayout scratch lowers in a program that
/// interns no zero: the scratch is addressed from its start, not at an
/// offset scalar of the program.
/// A program emitting more instructions than a kernel may on its target
/// ([`crate::atoms::Target::most_instructions`]: the CUDA bound above the
/// AMD one) is refused by the lowering, before any compiler sees it,
/// naming its count and the bound: given up at the statement passing the
/// bound (the count named is under twice it, where the whole program emits
/// four times it), not emitted whole and then refused.
#[test]
fn a_program_past_the_instruction_bound_is_refused() {
    use crate::lower::Error;
    let target = amd(AmdArch::Gfx1201, 64);
    let (most, shape) = (target.most_instructions(), Shape::new(64, 64));
    assert!(most < crate::atoms::sm86().most_instructions());
    let regs = shape.elems() / (4 * target.wave as usize);
    let mut k = Kernel::new("huge");
    k.warps(4);
    let out = k.param::<F32>("o", ParamKind::Out, shape.elems());
    let one = k.fill::<F32>(shape, Const::Float(1.0));
    let mut x = k.zeros::<F32>(shape);
    // Each add lists its loads, adds and stores: four instructions a register.
    for _ in 0..most / regs + 1 {
        x = k.binary(x, one, BinaryOp::Add);
    }
    let v = k.view(out, 0, [shape.cols, 1], shape, [None, None]);
    k.store(v, x);
    let lowering = crate::lower::Lowering::new(&super::declared(target));
    let err = crate::lower::standalone(k.finish(), &lowering).err().expect("refused");
    let Error::TooLarge { name, instructions, most: bound } = &err else { panic!("{err}") };
    assert_eq!((name.as_str(), *bound), ("huge", most));
    assert!(*instructions > most && *instructions < 2 * most, "{instructions} of {most}");
    assert_eq!(err.to_string(), format!("huge emits {instructions} instructions, past the {most} a kernel may"));
}

#[test]
fn a_relayout_through_shared_memory_needs_no_zero_of_the_program() {
    use crate::layouts::{Relayout, chunked};
    let (tile, lanes) = (Shape::new(16, 32), 32);
    let [from, to] = [Axis::Col, Axis::Row].map(|walk| chunked(tile, 4, 1, lanes, walk).expect("a chunked layout"));
    let costs = super::lane_moves_dear();
    assert_eq!(from.relayout(&to, 1, lanes, &costs, false), Relayout::ViaSmem);
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
    let lowering = Lowering::new(&device);
    let nodes = linear(prog, &lowering).toposort();
    assert!(nodes.iter().any(|u| matches!(u.op(), Op::Barrier { .. })), "no round trip through shared memory");
}

/// A register transpose lowers as the builder and the interpreter take it:
/// each thread keeps the elements it holds, its registers reordered, over
/// one warp or several and over a fragment repeated along both dimensions.
/// A result held under another layout has its operand re-held first.
#[test_case(Shape::new(16, 32), 1, false; "one warp")]
#[test_case(Shape::new(64, 32), 4, false; "four warps")]
#[test_case(Shape::new(32, 32), 1, true; "a result laid otherwise")]
fn a_register_transpose_lowers(tile: Shape, warps: u32, pinned: bool) {
    let lanes = 32;
    let mut k = Kernel::new("transpose");
    k.warps(warps);
    let x = k.param::<F32>("x", ParamKind::In, tile.elems());
    let y = k.param::<F32>("y", ParamKind::Out, tile.elems());
    let src = k.view(x, 0, [tile.cols, 1], tile, [None, None]);
    let a = k.load(src);
    let t = k.transpose(a);
    if pinned {
        let l = crate::layouts::chunked(tile.transposed(), 4, warps, lanes, Axis::Row).expect("a layout");
        k.pin(t, l);
    }
    let out = tile.transposed();
    let dst = k.view(y, 0, [out.cols, 1], out, [None, None]);
    k.store(dst, t);
    let device = Device { target: crate::atoms::sm86(), costs: super::costs() };
    let lowering = Lowering::new(&device);
    linear(k.finish(), &lowering);
}

/// The segment mask's last-row start, `seg_start[b·t + min(q_off + bq − 1, t − 1)]`,
/// is inside its parameter wherever the block is, so it is read once ahead
/// of the key loop, not under the pipeline's per-trip consume branch.
#[test_case(false; "warp core on sm_86")]
#[test_case(true; "warpgroup core on sm_90")]
fn the_segment_bounds_are_read_ahead_of_the_key_loop(group: bool) {
    use crate::kernels::Batch;
    use crate::kernels::attention::{AttnMask, AttnSpec, FaCfg, attention};

    let (target, cfg) = if group {
        (super::targets::hopper(), FaCfg { core: Core::Group, ..FaCfg::new(64, 64, 2) })
    } else {
        (crate::atoms::sm86(), FaCfg::new(64, 64, 3))
    };
    let spec = AttnSpec {
        batch: Batch::var("b", 1, 4),
        t: 100,
        tk: 100,
        heads: 4,
        kv_heads: 2,
        d: 64,
        mask: AttnMask { seg_start: true, ..AttnMask::default() },
        cache: None,
        scale: 0.125,
        cfg,
    };
    let prog = attention::<F16>(&spec);
    let slot = prog.params.iter().position(|p| p.name == "seg_start").expect("a segment parameter");
    let lowering = cfg.lowering(&super::declared(target));
    let linear = linear(prog, &lowering);
    let Op::Linear(ops::Linear { ops: list }) = linear.op() else { panic!("a linear list") };
    let reads_segments = |u: &Arc<UOp>| {
        let Op::Load(ops::Load { index, .. }) = u.op() else { return false };
        index.toposort().iter().any(|n| matches!(n.op(), Op::Param(p) if p.arg.slot == slot))
    };
    let reads: Vec<usize> = (0..list.len()).filter(|&i| reads_segments(&list[i])).collect();
    let key_loop = list.iter().position(|u| matches!(u.op(), Op::Range(..))).expect("a key loop");
    assert!(reads.len() >= 2, "the first and last rows' starts, read at {reads:?}");
    assert!(reads.iter().all(|&i| i < key_loop), "reads at {reads:?}, key loop from {key_loop}");
}

/// A scalar's bounds are its operation's over its inputs' (here `n` in
/// `-7..=9`): a quotient and a remainder truncate as C's do, a divisor of
/// either sign bounds both, one that may be zero bounds neither, and a
/// value read from memory is unbounded.
#[test_case(|_, n| n / 2, Some((-3, 4)); "a halved index")]
#[test_case(|_, n| n / -2, Some((-4, 3)); "a negative divisor")]
#[test_case(|_, n| n.clone() / (n + 8), Some((-7, 9)); "a divisor of one sign")]
#[test_case(|_, n| n.clone() / n, None; "a divisor that may be zero")]
#[test_case(|_, n| n % 4, Some((-3, 3)); "a remainder")]
#[test_case(|_, n| n % -4, Some((-3, 3)); "a remainder by a negative divisor")]
#[test_case(|_, n| n.min(3), Some((-7, 3)); "a minimum")]
#[test_case(|_, n| n.max(0), Some((0, 9)); "a maximum")]
#[test_case(|_, n| n * -2, Some((-18, 14)); "a negative factor")]
#[test_case(|_, n| n - 10, Some((-17, -1)); "a difference")]
#[test_case(|_, n| n.lt(2), Some((0, 1)); "a comparison")]
#[test_case(|k, _| k.block(0), Some((0, 4)); "a block index on its grid")]
#[test_case(|k, _| k.warp() + 1, Some((1, 4)); "a warp index")]
#[test_case(|k, n| { let p = k.param::<I32>("p", ParamKind::In, 4); k.load_scalar(p, n) }, None; "a value read")]
fn a_scalars_bounds_are_its_operations(f: fn(&mut Kernel, Sc) -> Sc, want: Option<(i64, i64)>) {
    let mut k = Kernel::new("bounds");
    k.warps(4);
    k.grid([Sc::from(5), Sc::from(1), Sc::from(1)]);
    let n = k.var("n", -7, 9);
    let v = f(&mut k, n);
    let id = k.intern(v);
    assert_eq!(k.finish().bounds(id, &|_| None), want);
}

/// A `cp.async` fill goes in whole 16-byte chunks, the same count per
/// thread ([`crate::layouts::fills_in_chunks`]); the emitter refuses any
/// other, naming the fill, as the rankings' `fits` predicts.
#[test_case(16, 32, true; "64 chunks over 32 threads")]
#[test_case(16, 24, false; "48 chunks over 32 threads")]
#[test_case(16, 20, false; "a row of two chunks and a half")]
fn a_fill_goes_in_whole_chunks_per_thread(rows: usize, cols: usize, fills: bool) {
    let shape = Shape::new(rows, cols);
    assert_eq!(crate::layouts::fills_in_chunks(shape, 2, 32), fills);
    let mut k = Kernel::new("fill");
    k.warps(1);
    let x = k.param::<F16>("x", ParamKind::In, shape.elems());
    let y = k.param::<F16>("y", ParamKind::Out, shape.elems());
    let s = k.smem::<F16>("s", shape.elems());
    let src = k.view(x, 0, [cols, 1], shape, [None, None]);
    let t = k.smem_view::<F16>(s, 0, shape);
    k.stage(t, src, CopyMode::Async);
    let held = Shape::new(rows, 16);
    let v = k.smem_view::<F16>(s, 0, held);
    let r = k.load(v);
    let out = k.view(y, 0, [cols, 1], held, [None, None]);
    k.store(out, r);
    let lowering = Lowering::new(&super::declared(crate::atoms::sm86()));
    match crate::lower::standalone(k.finish(), &lowering) {
        Ok(_) => assert!(fills, "a {rows}×{cols} fill lowered"),
        Err(crate::lower::Error::Unsupported { what }) => {
            assert!(!fills, "{what}");
            assert_eq!(what, format!("a {rows}×{cols} fill by 32 threads"));
        }
        Err(e) => panic!("{e}"),
    }
}

/// The alias contract the LLVM text claims on a real program's buffers:
/// every buffer parameter is `ptr noalias align 32` — each `ParamArg` a
/// distinct buffer of the launch, a view an offset inside its own — while
/// the one runtime scalar (a bound batch's count) carries nothing. What no
/// access may carry beyond that (`align`, `!alias.scope`,
/// `dereferenceable`) is the renderer's own pin, in codegen's
/// `the_text_claims_exactly_the_aliasing_it_knows`.
#[test_case(crate::atoms::sm86(); "sm_86")]
#[test_case(amd(AmdArch::Gfx1201, 64); "gfx1201")]
fn the_text_claims_exactly_the_aliasing_tk_knows(target: Target) {
    let spec = NormSpec {
        norm: Norm::Layer,
        rows: 64,
        d: 256,
        batch: Batch::var("n", 1, 4),
        eps: 1e-5,
        residual: true,
        bias: true,
        cfg: NormCfg { br: 4 },
    };
    let text = crate::lower::render(norm::<BF16>(&spec), &Lowering::new(&super::declared(target))).unwrap();
    let define = text.lines().find(|l| l.starts_with("define ")).expect("a define line");
    let params = define.split('(').nth(1).expect("parameters").trim_end_matches(") #0 {");
    let mut buffers = 0;
    for param in params.split(", ") {
        if param.starts_with("ptr noalias align 32 %data") {
            buffers += 1;
        } else {
            assert_eq!(param, "i32 %data6", "the batch count alone is bare: {define}");
        }
    }
    assert_eq!(buffers, 6, "x, residual, w, b, out, sum: {define}");
}

/// The wide-load fold at a product's operands: where one wide load fills the
/// registers an operand stacks — a whole fragment a lane, as RDNA's WMMA
/// fragments gather — the operand is one shuffle of that load, never the
/// load's elements torn into the register buffer and read back out a lane at
/// a time. On sm_86 the fills are pairs (`ldmatrix` words, two-element
/// runs): several loads a fragment, the closed pattern of one load feeding
/// one stack does not hold, and the scalar loads stack exactly as they did —
/// SLP remains the backstop under them.
#[test_case(amd(AmdArch::Gfx1201, 64); "gfx1201 folds the fragment to its load")]
#[test_case(crate::atoms::sm86(); "sm_86 keeps its pairs")]
fn a_products_operand_is_the_load_that_filled_it(target: Target) {
    let cfg = GemmCfg { tile: [64, 64, 32], stages: 2, warps: [2, 2], unroll: false, core: Core::Warp, crews: 1 };
    let spec = GemmSpec { m: 128, n: 128, k: 128, batch: Batch::Static(1), epilogue: Epilogue::default(), cfg };
    let lowered =
        crate::lower::standalone(gemm::<BF16>(&spec), &Lowering::new(&super::declared(target.clone()))).unwrap();
    let Op::Program(ops::Program { linear: Some(linear), .. }) = lowered.program.op() else {
        unreachable!("a lowering emits a linear program")
    };
    let (mut folded, mut stacked) = (0, 0);
    for u in linear.op().sources() {
        let Op::Wmma(ops::Wmma { a, b, .. }) = u.op() else { continue };
        for operand in [a, b] {
            folded += matches!(operand.op(), Op::Shuffle(..)) as usize;
            stacked += matches!(operand.op(), Op::Stack(..)) as usize;
        }
    }
    let expect = matches!(target.arch, svod_dtype::GpuArch::Amd(_));
    assert_eq!(folded > 0, expect, "shuffles {folded}, stacks {stacked} on {:?}", target.arch);
    if !expect {
        assert!(stacked > 0, "the scalar loads still stack");
    } else {
        assert_eq!(stacked, 0, "no operand is stacked out of the register buffer");
    }
}
