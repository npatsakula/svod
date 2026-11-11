//! The emitted instruction list: the values the lowering computes and where
//! it lists them.

use std::collections::HashMap;
use std::sync::Arc;

use svod_dtype::{AmdArch, DType};
use svod_ir::uop::eval::{eval_binary_op, eval_unary_op};
use svod_ir::{ConstValue, Op, UOp, ops};
use test_case::test_case;

use crate::atoms::{Core, Costs, Device, Fill, Target};
use crate::build::*;
use crate::ir::*;
use crate::kernels::Batch;
use crate::kernels::gemm::{Epilogue, GemmCfg, GemmSpec, gemm};
use crate::kernels::rows::{Norm, NormCfg, NormSpec, norm};
use crate::lower::Lowering;
use crate::lower::emit::minimum;
use crate::sweep::assumed;

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
    let prog = Kernel::record("min", |k| {
        k.warps(1);
        let a = k.input::<F32>("a", tile.elems());
        let b = k.input::<F32>("b", tile.elems());
        let out_shape = if reduced { Shape::new(tile.rows, 1) } else { tile };
        let y = k.output::<F32>("y", out_shape.elems());
        let va = k.view(a, 0, [tile.cols, 1], tile, [None, None]);
        let vb = k.view(b, 0, [tile.cols, 1], tile, [None, None]);
        let (ra, rb) = (k.load(va), k.load(vb));
        let m = if reduced { k.reduce(ra, Axis::Row, ReduceOp::Min) } else { k.binary(ra, rb, BinaryOp::Min) };
        let out = k.view(y, 0, [out_shape.cols, 1], out_shape, [None, None]);
        k.store(out, m);
    });
    let device = Device { target: crate::atoms::sm86(), costs: Costs::default() };
    let lowering = Lowering::new(&device);
    let nodes = linear(prog, &lowering).toposort();
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
    let prog = Kernel::record("guarded", |k| {
        k.warps(1);
        let lens = k.input::<I32>("lens", 4);
        let y = k.output::<F32>("y", 2 * tile.elems());
        let n = k.var("n", 0, max);
        let index = if div > 1 { n.clone() / div } else { n.clone() };
        let len = k.load_scalar(lens, index);
        k.if_(
            n.lt(4),
            |k| {
                let v = k.splat::<F32>(tile, len.clone());
                let out = k.view(y, 0, [tile.cols, 1], tile, [None, None]);
                k.store(out, v);
            },
            |_| {},
        );
        let v = k.splat::<F32>(tile, len);
        let out = k.view(y, tile.elems(), [tile.cols, 1], tile, [None, None]);
        k.store(out, v);
    });
    let device = Device { target: crate::atoms::sm86(), costs: Costs::default() };
    let lowering = Lowering::new(&device);
    let linear = linear(prog, &lowering);
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
    let prog = Kernel::record("shares", |k| {
        k.warps(1);
        let y = k.output::<F32>("y", 4 * tile.elems());
        k.grid([Sc::from(4), Sc::from(1), Sc::from(1)]);
        // A share of three steps each: the fourth starts at nine.
        let count = (Sc::from(steps) - k.block(0) * 3).min(3);
        let mut count = Sc::id(k.intern(count));
        if steady {
            count = count.clone() - Sc::from(0).lt(count);
        }
        let zero = k.zeros::<F32>(tile);
        let one = k.fill::<F32>(tile, Const::Float(1.0));
        let [sum] = k.loop_(count, [zero], |k, _, [sum]| [k.binary(*sum, one, BinaryOp::Add).into()]);
        let at = k.block(0) * tile.elems();
        let out = k.view(y, at, [tile.cols, 1], tile, [None, None]);
        k.store(out, *sum);
    });
    let device = Device { target: crate::atoms::sm86(), costs: Costs::default() };
    let linear = linear(prog, &Lowering::new(&device));
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
    let prog = Kernel::record("huge", |k| {
        k.warps(4);
        let out = k.output::<F32>("o", shape.elems());
        let one = k.fill::<F32>(shape, Const::Float(1.0));
        let mut x = k.zeros::<F32>(shape);
        // Each add lists its loads, adds and stores: four instructions a register.
        for _ in 0..most / regs + 1 {
            x = k.binary(x, one, BinaryOp::Add);
        }
        let v = k.view(out, 0, [shape.cols, 1], shape, [None, None]);
        k.store(v, x);
    });
    let lowering = crate::lower::Lowering::new(&super::declared(target));
    let err = crate::lower::standalone(prog, &lowering).err().expect("refused");
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
    let prog = Kernel::record("relayout", |k| {
        k.warps(1);
        let x = k.input::<F32>("x", 2 * tile.elems());
        let y = k.output::<F32>("y", 2 * tile.elems());
        let src = k.view(x, tile.elems(), [tile.cols, 1], tile, [None, None]);
        let a = k.load(src);
        k.pin(a, from);
        let b = k.unary(a, UnaryOp::Neg);
        k.pin(b, to);
        let dst = k.view(y, tile.elems(), [tile.cols, 1], tile, [None, None]);
        k.store(dst, b);
    });
    let prog = prog;
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
    let prog = Kernel::record("transpose", |k| {
        k.warps(warps);
        let x = k.input::<F32>("x", tile.elems());
        let y = k.output::<F32>("y", tile.elems());
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
    });
    let device = Device { target: crate::atoms::sm86(), costs: super::costs() };
    let lowering = Lowering::new(&device);
    linear(prog, &lowering);
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
    let spec = assumed(
        AttnSpec {
            batch: Batch::var("b", 1, 4),
            t: 100,
            tk: 100,
            heads: 4,
            kv_heads: 2,
            d: 64,
            mask: AttnMask { seg_start: true, ..AttnMask::default() },
            cache: None,
            packed: false,
            scale: 0.125,
        },
        cfg,
    );
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
#[test_case(|k, n| { let p = k.input::<I32>("p", 4); k.load_scalar(p, n) }, None; "a value read")]
fn a_scalars_bounds_are_its_operations(f: for<'k> fn(&mut Kernel<'k>, Sc<'k>) -> Sc<'k>, want: Option<(i64, i64)>) {
    let (prog, id) = Kernel::record_with("bounds", |k| {
        k.warps(4);
        k.grid([Sc::from(5), Sc::from(1), Sc::from(1)]);
        let n = k.var("n", -7, 9);
        let v = f(k, n);
        k.intern(v)
    });
    assert_eq!(prog.bounds(id, &|_| None), want);
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
    let prog = Kernel::record("fill", |k| {
        k.warps(1);
        let x = k.input::<F16>("x", shape.elems());
        let y = k.output::<F16>("y", shape.elems());
        let s = k.smem::<F16>("s", shape.elems());
        let src = k.view(x, 0, [cols, 1], shape, [None, None]);
        let t = k.smem_view::<F16>(s, 0, shape);
        k.stage(t, src, CopyMode::Async);
        let held = Shape::new(rows, 16);
        let v = k.smem_view::<F16>(s, 0, held);
        let r = k.load(v);
        let out = k.view(y, 0, [cols, 1], held, [None, None]);
        k.store(out, r);
    });
    let lowering = Lowering::new(&super::declared(crate::atoms::sm86()));
    match crate::lower::standalone(prog, &lowering) {
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
/// distinct buffer of the launch, a view an offset inside its own — an
/// input `readonly` besides (no store of the kernel reaches it), while
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
    let (mut inputs, mut outputs) = (0, 0);
    for param in params.split(", ") {
        if param.starts_with("ptr noalias readonly align 32 %data") {
            inputs += 1;
        } else if param.starts_with("ptr noalias align 32 %data") {
            outputs += 1;
        } else {
            assert_eq!(param, "i32 %data6", "the batch count alone is bare: {define}");
        }
    }
    assert_eq!((inputs, outputs), (4, 2), "x, residual, w, b read-only; out, sum written: {define}");
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
    let cfg = GemmCfg {
        tile: [64, 64, 32],
        warps: [2, 2],
        core: Core::Warp,
        schedule: Schedule::ring(2, false),
        order: Default::default(),
        tail: None,
    };
    let spec =
        assumed(GemmSpec { m: 128, n: 128, k: 128, batch: Batch::Static(1), epilogue: Epilogue::default() }, cfg);
    let lowered =
        crate::lower::standalone(gemm::<BF16>(&spec), &Lowering::new(&super::declared(target.clone()))).unwrap();
    let (mut folded, mut stacked) = (0, 0);
    for u in lowered.linear.op().sources() {
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

/// Every path of a program past `i32` elements holds the list's type rules,
/// which a mixed-width select or shift breaks, on every target and on a
/// target of the other fill.
#[test]
fn every_path_past_i32_keeps_one_width() {
    let staged = ("staged nvidia", Target { fill: Fill::Staged, ldmatrix: false, ..crate::sweep::hopper() });
    for (name, target) in crate::sweep::targets().into_iter().chain([staged]) {
        let launches = crate::sweep::past_i32(&target);
        assert_eq!(launches.len(), 8, "{name}");
        for (kernel, prog, lowering) in launches {
            assert!(prog.params.iter().any(|p| p.elems > i32::MAX as usize), "{name}: {kernel} fits i32");
            crate::lower::render(prog, &lowering).unwrap_or_else(|e| panic!("{name}: {kernel}: {e}"));
        }
    }
}

/// A fill's run past a column bound reads element 0 in the width of its
/// offset, on the cards that load a bounded run through `load_l2_only`.
#[test_case(crate::sweep::targets().remove(0).1; "sm_86")]
#[test_case(crate::sweep::hopper(); "hopper")]
fn a_bounded_load_past_i32_keeps_one_width(target: Target) {
    let prog = Kernel::record("past", |k| {
        let wide = (i32::MAX as usize + 1) * 2;
        let src = k.input::<I32>("src", wide);
        let out = k.output::<I32>("out", wide);
        k.grid([Sc::from(1), Sc::from(1), Sc::from(1)]);
        k.warps(4);
        let tile = Shape::new(32, 32);
        let from = k.view(src, 0, [32, 1], tile, [None, Some(Sc::from(20))]);
        let loaded = k.load_as_fill(from);
        let to = k.view(out, 0, [32, 1], tile, [None, None]);
        k.store(to, loaded);
    });
    crate::lower::render(prog, &Lowering::new(&super::declared(target))).unwrap();
}

/// A count past `i32` is refused rather than cut to its low 32 bits.
#[test]
fn a_grid_past_i32_is_refused() {
    let prog = Kernel::record("past", |k| {
        let out = k.output::<I32>("out", 1);
        k.grid([Sc::from(1usize << 31), Sc::from(1), Sc::from(1)]);
        k.warps(4);
        let v = k.view(out, 0, [1, 1], Shape::new(1, 1), [None, None]);
        let zero = k.fill::<I32>(Shape::new(1, 1), Const::Int(0));
        k.store(v, zero);
    });
    let target = amd(AmdArch::Gfx942, 304);
    let e = crate::lower::render(prog, &Lowering::new(&super::declared(target))).unwrap_err();
    assert!(e.to_string().contains("past the i32"), "{e}");
}

/// A batch's stride past `i32` stays what it is, not its low 32 bits.
#[test]
fn a_constant_past_i32_keeps_its_value() {
    let cfg = crate::ops::config::gemm_cfg([128, 128, 32], 1, [2, 2], false);
    let spec = assumed(
        GemmSpec { m: 50_001, n: 50_001, k: 50_048, batch: Batch::Static(2), epilogue: Epilogue::DEFAULT },
        cfg,
    );
    let target = amd(AmdArch::Gfx942, 304);
    let text = crate::lower::render(gemm::<BF16>(&spec), &cfg.lowering(&super::declared(target))).unwrap();
    assert!(text.contains("2500100001") && !text.contains("-1794867295"));
}

/// A pipeline runs as its leading ring's schedule says; a second ring its
/// stages reach that holds fewer slots is read past by every slot but its
/// own, which the lowering refuses rather than leaving to the device.
#[test_case(3 => Ok(()); "as deep as the pipeline")]
#[test_case(1 => Err("a view of b reaches elements 0..1536, which holds 512".into()); "one slot short")]
fn a_ring_shallower_than_its_pipeline_is_refused(depth: usize) -> Result<(), String> {
    let tile = Shape::new(16, 32);
    let prog = Kernel::record("ring", |k| {
        k.warps(1);
        let x = k.input::<F32>("x", 8 * tile.elems());
        let y = k.output::<F32>("y", tile.elems());
        let a = k.ring::<F32>("a", tile.elems(), Schedule::ring(3, false));
        let b = k.ring::<F32>("b", tile.elems(), Schedule::ring(depth, false));
        let v = k.view(x, 0, [tile.cols, 1], tile, [None, None]);
        let zero = k.zeros::<F32>(tile);
        let [s] = k.pipeline(
            8,
            a,
            [zero],
            |k, step, slot| {
                let src = k.at(v, step * 16, 0);
                for ring in [a, b] {
                    let t = k.slot(ring, &slot, tile);
                    k.stage(t, src, CopyMode::Async);
                }
            },
            |k, _, slot, [s]| {
                let (ta, tb) = (k.slot(a, &slot, tile), k.slot(b, &slot, tile));
                let (la, lb) = (k.load(ta), k.load(tb));
                let sum = k.binary(la, lb, BinaryOp::Add);
                [k.binary(*s, sum, BinaryOp::Add).into()]
            },
        );
        let out = k.view(y, 0, [tile.cols, 1], tile, [None, None]);
        k.store(out, *s);
    });
    let lowering = Lowering::new(&super::declared(crate::atoms::sm86()));
    crate::lower::standalone(prog, &lowering).map(drop).map_err(|e| e.to_string())
}

/// A tile is read only where its definition dominates the read: one kept
/// from a loop's body and read past the loop (its last trip's, or none), or
/// one a branch defines and the kernel stores past it (undefined where the
/// branch is not taken), is refused by the lowering, naming the scopes; a
/// value handed on as the body's result or the branch's merge lowers.
#[test_case(Escape::Loop => Err(("a loop's body", "the kernel")); "kept from a loop's body")]
#[test_case(Escape::Branch => Err(("a branch's then", "the kernel")); "kept from a branch")]
#[test_case(Escape::If => Err(("a branch's then", "the kernel")); "defined under an if alone")]
#[test_case(Escape::Warps => Err(("the warps' body", "the kernel")); "kept from the warps' body")]
#[test_case(Escape::None => Ok(()); "handed on")]
fn a_tile_read_past_its_scope_is_refused(escape: Escape) -> Result<(), (&'static str, &'static str)> {
    let tile = Shape::new(16, 32);
    let prog = Kernel::record("escape", |k| {
        k.warps(1);
        k.grid([Sc::from(2), Sc::from(1), Sc::from(1)]);
        let y = k.output::<F32>("y", 2 * tile.elems());
        let zero = k.zeros::<F32>(tile);
        let one = k.fill::<F32>(tile, Const::Float(1.0));
        let mut leak = None;
        let [sum] = k.loop_(4, [zero], |k, _, [s]| {
            let t = k.binary(*s, one, BinaryOp::Add);
            if let Escape::Loop = escape {
                leak = Some(t);
            }
            [t.into()]
        });
        let pred = k.block(0).lt(1);
        let [picked] = k.select_if(
            pred,
            |k| {
                let t = k.binary(*sum, one, BinaryOp::Add);
                if let Escape::Branch = escape {
                    leak = Some(t);
                }
                [t]
            },
            |_| [*sum],
        );
        if let Escape::If = escape {
            let pred = k.block(0).lt(1);
            k.if_(pred, |k| leak = Some(k.fill::<F32>(tile, Const::Float(1.0))), |_| {});
        }
        if let Escape::Warps = escape {
            k.each_warp(|k| leak = Some(k.fill::<F32>(tile, Const::Float(2.0))));
        }
        let at = k.block(0) * tile.elems();
        let out = k.view(y, at, [tile.cols, 1], tile, [None, None]);
        k.store(out, leak.unwrap_or(picked));
    });
    let lowering = Lowering::new(&super::declared(crate::atoms::sm86()));
    match crate::lower::standalone(prog, &lowering) {
        Ok(_) => Ok(()),
        Err(crate::lower::Error::Escaped { defined, used, .. }) => Err((defined, used)),
        Err(e) => panic!("{e}"),
    }
}

#[derive(Debug)]
enum Escape {
    Loop,
    Branch,
    If,
    Warps,
    None,
}
