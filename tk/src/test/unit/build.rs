use crate::ir::*;

/// A block-level GEMM mainloop records as one pipeline whose consumer carries
/// the accumulator, with every view addressed by scalar expressions and the
/// bound batch on grid z.
#[test]
fn gemm_records_as_a_pipeline_with_a_carried_accumulator() {
    let (m, n, kk) = (256usize, 256usize, 1024usize);
    let (bm, bn, bk, stages) = (128usize, 128usize, 32usize, 3usize);
    let batch = crate::kernels::Batch::Var { name: "b".into(), min: 1, max: 8 };
    let prog = super::gemm_nt_batched(m, n, kk, [bm, bn, bk], stages, batch);

    let kinds: Vec<_> = prog.walk().map(|(d, s)| (d, std::mem::discriminant(s))).collect();
    assert_eq!(kinds.iter().filter(|(d, _)| *d == 0).count(), 4, "zeros, pipeline, cast, store at the top level");
    let Stmt::Pipeline(p) = &prog.body.0[1] else { panic!("second statement is the pipeline") };
    assert_eq!((p.stages, p.produce.body.0.len(), p.consume.body.0.len()), (3, 2, 1));
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
    let mut k = crate::build::Kernel::new("vars");
    k.var("b", 1, 8);
    k.var("b", min, max);
    assert_eq!(k.finish().vars, vec![Var { name: "b".into(), min: 1, max: 8 }]);
}

/// A scalar read is of an input and no store writes an input, so the
/// lowering may list a read anywhere its index is known.
#[test]
#[should_panic(expected = "a scalar load reads an input")]
fn a_scalar_read_is_of_an_input() {
    let mut k = crate::build::Kernel::new("read");
    let p = k.param::<crate::build::I32>("p", ParamKind::Out, 4);
    k.load_scalar(p, 0);
}

#[test]
#[should_panic(expected = "a store into an input")]
fn no_store_writes_an_input() {
    use crate::build::{I32, Kernel};
    let mut k = Kernel::new("store");
    let p = k.param::<I32>("p", ParamKind::In, 4);
    let v = k.splat::<I32>(Shape::new(1, 4), 1);
    let out = k.view(p, 0, [4, 1], Shape::new(1, 4), [None, None]);
    k.store(out, v);
}
