use crate::ir::*;

/// A block-level GEMM mainloop records as one pipeline whose consumer carries
/// the accumulator, with every view addressed by scalar expressions.
#[test]
fn gemm_records_as_a_pipeline_with_a_carried_accumulator() {
    let (m, n, kk) = (256usize, 256usize, 1024usize);
    let (bm, bn, bk, stages) = (128usize, 128usize, 32usize, 3usize);
    let prog = super::programs::gemm_nt(m, n, kk, bm, bn, bk, stages);

    let kinds: Vec<_> = prog.walk().map(|(d, s)| (d, std::mem::discriminant(s))).collect();
    assert_eq!(kinds.iter().filter(|(d, _)| *d == 0).count(), 4, "zeros, pipeline, cast, store at the top level");
    let Stmt::Pipeline(p) = &prog.body.0[1] else { panic!("second statement is the pipeline") };
    assert_eq!((p.stages, p.produce.body.0.len(), p.consume.body.0.len()), (3, 2, 1));
    assert_eq!(p.carried.len(), 1);
    assert_eq!(prog.value(p.carried[0].phi).shape, Shape::new(bm, bn));
    assert_eq!(prog.vars, vec!["b".to_string()]);
    assert!(matches!(prog.scalar(prog.grid[2]), Scalar::Var(v) if v == "b"));
}
