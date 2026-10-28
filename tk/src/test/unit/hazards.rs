//! Lowering hazards found on the device while writing kernels, each a
//! minimal program run there against the interpreter
//! (`SVOD_DEVICE=… cargo test -p svod-tk --release --lib hazards`).

use crate::build::*;
use crate::interp;
use crate::ir::*;
use crate::lower::Lowering;

/// Rows a block holds, a warp each, as [`crate::kernels::nearest::nearest_select`] does.
const ROWS: usize = 4;
/// Columns of the per-row input.
const COLS: usize = 12;

/// The device's outputs of `prog` against the interpreter's, on inputs `0.5·j - i` (row `i`, element `j`).
fn device_matches(prog: Program) {
    let Some(target) = super::device_target() else { return };
    let device = crate::sweep::declared(target);
    let lowering = Lowering::new(&device);
    let params: Vec<Vec<f64>> = prog
        .params
        .iter()
        .map(|p| (0..p.elems).map(|e| 0.5 * (e % COLS) as f64 - (e / COLS) as f64).collect())
        .collect();
    let want = interp::run(&prog, params.clone(), &[]).unwrap();
    let got = super::targets::on_device(prog, &lowering, &params);
    for (g, w) in got.iter().zip(&want) {
        if let Some(g) = g {
            assert_eq!(g, w, "device against the interpreter");
        }
    }
}

/// A warp a row: each step loads its row's `[ROWS, 1]` window of `src` at a
/// column the loop's step sets and folds it into the row's minimum by a row
/// reduction. On sm_86 the device's reduction ran across lanes holding other
/// rows' elements: the reduction's result was laid by its consumer (the
/// lanes' tile's row vector) apart from its operand's vector, a conflict on
/// a result the layout rewrite dropped, and the emitter folded the operand
/// as it is held and stored that as the result's layout. The rewrite now
/// reduces into the operand's vector and re-holds it.
#[test]
fn a_row_window_reduced_in_a_warp_per_row_loop() {
    let mut k = Kernel::new("row_window");
    let src = k.param::<F32>("src", ParamKind::In, ROWS * COLS);
    let out = k.param::<F32>("out", ParamKind::Out, ROWS * 32);
    k.grid([Sc::from(1), Sc::from(1), Sc::from(1)]);
    k.warps(ROWS as u32);
    let lane = Shape::new(ROWS, 32);
    let init = vec![k.fill::<F32>(lane, Const::Float(f64::INFINITY))];
    let view = k.view(src, 0, [COLS, 1], Shape::new(ROWS, 1), [None, None]);
    let [least] = k.loop_vec(COLS, init, |k, step, carried| {
        let v = k.at(view, 0, step);
        let v = k.load(v);
        let low = k.reduce(v, Axis::Row, ReduceOp::Min);
        let zero = k.zeros::<F32>(lane);
        let low = k.binary(low, zero, BinaryOp::Add);
        vec![k.binary(carried[0], low, BinaryOp::Min)]
    })[..] else {
        unreachable!("a tile")
    };
    let v = k.view(out, 0, [32, 1], lane, [None, None]);
    k.store(v, least);
    device_matches(k.finish());
}

/// Read ahead: a loop carries the next `AHEAD` steps' loads in a ring, each
/// step taking the ring's head and loading the step `AHEAD` on under a
/// branch on the step's range, as a listing reading ahead of its work
/// would. On sm_86 every step summed the first loads: a carried value whose
/// next is another carried one (the ring's rotation) was never moved, the
/// emitter moving a next only where the body defines it, and moved in
/// sequence it would read a slot already overwritten. The builder now copies
/// such a next where the body starts.
#[test]
fn a_read_ahead_ring_under_a_branch() {
    const AHEAD: usize = 2;
    let mut k = Kernel::new("read_ahead");
    let src = k.param::<F32>("src", ParamKind::In, ROWS * COLS * 32);
    let out = k.param::<F32>("out", ParamKind::Out, ROWS * 32);
    k.grid([Sc::from(1), Sc::from(1), Sc::from(1)]);
    k.warps(ROWS as u32);
    let lane = Shape::new(ROWS, 32);
    let view = k.view(src, 0, [COLS * 32, 1], lane, [None, None]);
    // The step `at`'s row of 32, zero past the last.
    let fetch = |k: &mut Kernel, at: Sc| {
        let [v] = k.select_if(
            at.clone().lt(COLS),
            |k| {
                let v = k.at(view, 0, at * 32);
                [k.load(v)]
            },
            |k| [k.zeros::<F32>(lane)],
        );
        v
    };
    let mut init = vec![k.zeros::<F32>(lane)];
    init.extend((0..AHEAD).map(|p| fetch(&mut k, Sc::from(p))));
    let sum = k.loop_vec(COLS, init, |k, step, carried| {
        let sum = k.binary(carried[0], carried[1], BinaryOp::Add);
        let next = fetch(k, step + AHEAD);
        let mut out = vec![sum];
        out.extend(carried[2..].iter().copied());
        out.push(next);
        out
    });
    let v = k.view(out, 0, [32, 1], lane, [None, None]);
    k.store(v, sum[0]);
    device_matches(k.finish());
}
