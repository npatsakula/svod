use crate::build::*;
use crate::ir::*;

/// `c[m, n] = a[m, k] · b[n, k]ᵀ` as a block-level pipeline: a `[bm, bn]` tile per
/// block over a `stages`-deep ring of `[bm, bk]`/`[bn, bk]` shared slots, with a
/// bound batch variable `b` on grid axis 2 (unused by the addressing).
pub fn gemm_nt(m: usize, n: usize, kk: usize, bm: usize, bn: usize, bk: usize, stages: usize) -> Program {
    let mut k = Kernel::new("gemm");
    let trips = kk / bk;
    let a = k.param::<BF16>("a", ParamKind::In, m * kk);
    let b = k.param::<BF16>("b", ParamKind::In, n * kk);
    let c = k.param::<BF16>("c", ParamKind::Out, m * n);
    let batch = k.var("b");
    let [gm, gn] = [k.c((m / bm) as i64), k.c((n / bn) as i64)];
    k.grid([gm, gn, batch]);
    k.warps(8);
    let a_s = k.smem::<BF16>("a_s", stages * bm * bk);
    let b_s = k.smem::<BF16>("b_s", stages * bn * bk);

    let (bx, by) = (k.block(0), k.block(1));
    let a_view = k.view(a, 0, [kk, 1], Shape::new(bm, bk), [None, None]);
    let b_view = k.view(b, 0, [kk, 1], Shape::new(bn, bk), [None, None]);
    let row0 = k.mul(bx, bm);
    let col0 = k.mul(by, bn);
    let a_view = k.at(a_view, row0, 0);
    let b_view = k.at(b_view, col0, 0);
    let acc0 = k.zeros::<F32>(Shape::new(bm, bn));
    let trips = k.c(trips as i64);
    let [acc] = k.pipeline(
        trips,
        stages,
        [acc0],
        |k, step, slot| {
            let koff = k.mul(step, bk);
            let a_g = k.at(a_view, 0, koff);
            let b_g = k.at(b_view, 0, koff);
            let a_t = k.smem_slot::<BF16>(a_s, slot, Shape::new(bm, bk));
            let b_t = k.smem_slot::<BF16>(b_s, slot, Shape::new(bn, bk));
            k.stage(a_t, a_g, CopyMode::Async);
            k.stage(b_t, b_g, CopyMode::Async);
        },
        |k, _step, slot, [acc]| {
            let a_t = k.smem_slot::<BF16>(a_s, slot, Shape::new(bm, bk));
            let b_t = k.smem_slot::<BF16>(b_s, slot, Shape::new(bn, bk));
            [k.mma(acc, a_t, false, b_t, true)]
        },
    );
    let out = k.cast::<F32, BF16>(acc);
    let c_view = k.view(c, 0, [n, 1], Shape::new(bm, bn), [None, None]);
    let c_view = k.at(c_view, row0, col0);
    k.store(c_view, out);
    k.finish()
}
