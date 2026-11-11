use crate::build::Kernel;
use crate::ir::*;

/// A block-level GEMM mainloop records as one pipeline whose consumer carries
/// the accumulator, with every view addressed by scalar expressions and the
/// bound batch on grid z.
#[test]
fn gemm_records_as_a_pipeline_with_a_carried_accumulator() {
    let (m, n, kk) = (256usize, 256usize, 1024usize);
    let (bm, bn, bk, stages) = (128usize, 128usize, 32usize, 3usize);
    let batch = crate::kernels::Batch::var("b", 1, 8);
    let prog = super::gemm_nt_batched(m, n, kk, [bm, bn, bk], stages, batch);

    let kinds: Vec<_> = prog.walk().map(|(d, s)| (d, std::mem::discriminant(s))).collect();
    assert_eq!(kinds.iter().filter(|(d, _)| *d == 0).count(), 4, "zeros, pipeline, cast, store at the top level");
    let Stmt::Pipeline(p) = &prog.body.0[1] else { panic!("second statement is the pipeline") };
    assert_eq!((p.schedule.stages(), p.produce.body.0.len(), p.consume.body.0.len()), (3, 2, 1));
    assert_eq!(p.carried.len(), 1);
    assert_eq!(prog.value(p.carried[0].phi).shape, Shape::new(bm, bn));
    assert_eq!(prog.vars, vec![Var { name: "b".into(), min: 1, max: 8 }]);
    assert!(matches!(prog.scalar(prog.grid[2]), Scalar::Var(v) if v == "b"));
}

/// A variable asked for again is the same one; asked for over another
/// range, it is refused rather than kept at its first.
#[test_case::test_case(1, 8; "the same range")]
#[test_case::test_case(1, 16 => panics "variable b is in 1..=8"; "another range")]
fn a_variable_asked_for_again_keeps_its_range(min: i64, max: i64) {
    let prog = Kernel::record("vars", |k| {
        k.var("b", 1, 8);
        k.var("b", min, max);
    });
    assert_eq!(prog.vars, vec![Var { name: "b".into(), min: 1, max: 8 }]);
}

/// A variable's range is a nonempty range of `i32`, as every range is: one
/// past it either way, or an empty one, is refused where it is declared,
/// before any program holds it.
#[test_case::test_case(1, i64::from(i32::MAX) + 1 => panics "variable b in 1..=2147483648 is past i32"; "past the top")]
#[test_case::test_case(i64::from(i32::MIN) - 1, 0 => panics "variable b in -2147483649..=0 is past i32"; "past the bottom")]
#[test_case::test_case(2, 1 => panics "variable b is in the empty 2..=1"; "empty")]
#[test_case::test_case(i64::from(i32::MIN), i64::from(i32::MAX); "all of i32")]
fn a_variable_outside_i32_or_empty_is_refused_when_declared(min: i64, max: i64) {
    let prog = Kernel::record("vars", |k| {
        k.var("b", min, max);
    });
    assert_eq!(prog.vars, vec![Var { name: "b".into(), min, max }]);
}

/// Lone warps are whole copies of the warp grid; a grid set after them
/// would silently drop them, so it is refused.
#[test_case::test_case(false => (2, 8); "the grid first")]
#[test_case::test_case(true => panics "the warp grid is set before the lone warps"; "the grid again after")]
fn the_warp_grid_comes_before_the_lone_warps(again: bool) -> (u32, u32) {
    let prog = Kernel::record("warps", |k| {
        k.warps(2);
        k.lone_warps(8);
        if again {
            k.warps(4);
        }
    });
    (prog.warp_grid.count(), prog.warps)
}
