use crate::build::*;
use crate::ir::*;

/// `c[m, n] = a[m, k] · b[n, k]ᵀ` as a block-level pipeline: a `[bm, bn]` tile per
/// block over a `stages`-deep ring of `[bm, bk]`/`[bn, bk]` shared slots, with a
/// bound batch variable `b` on grid axis 2 (unused by the addressing).
pub fn gemm_nt(m: usize, n: usize, kk: usize, bm: usize, bn: usize, bk: usize, stages: usize) -> Program {
    gemm_nt_ordered(m, n, kk, bm, bn, bk, stages, 0)
}

/// [`gemm_nt`] walking the tile grid in groups of `group_m` tile rows (0 =
/// row-major), so blocks resident together share B columns in L2.
#[allow(clippy::too_many_arguments)]
pub fn gemm_nt_ordered(
    m: usize,
    n: usize,
    kk: usize,
    bm: usize,
    bn: usize,
    bk: usize,
    stages: usize,
    group_m: usize,
) -> Program {
    let mut k = Kernel::new("gemm");
    let trips = kk / bk;
    let a = k.param::<BF16>("a", ParamKind::In, m * kk);
    let b = k.param::<BF16>("b", ParamKind::In, n * kk);
    let c = k.param::<BF16>("c", ParamKind::Out, m * n);
    let batch = k.var("b", 1, 8);
    let [gm, gn] = [Sc::from(m / bm), Sc::from(n / bn)];
    k.grid([gm, gn, batch]);
    k.warps(8);
    let a_s = k.smem::<BF16>("a_s", stages * bm * bk);
    let b_s = k.smem::<BF16>("b_s", stages * bn * bk);

    let (bx, by) = if group_m == 0 {
        (k.block(0), k.block(1))
    } else {
        let (gm, gn) = (m / bm, n / bn);
        let id = k.block(0) + k.block(1) * gm;
        let group = id.clone() / (group_m * gn);
        let first_m = group * group_m;
        let rows = (Sc::from(gm) - first_m.clone()).min(group_m);
        let within = id % (group_m * gn);
        let bx = first_m + within.clone() % rows.clone();
        let by = within / rows;
        (bx, by)
    };
    let a_view = k.view(a, 0, [kk, 1], Shape::new(bm, bk), [None, None]);
    let b_view = k.view(b, 0, [kk, 1], Shape::new(bn, bk), [None, None]);
    let row0 = bx * bm;
    let col0 = by * bn;
    let a_view = k.at(a_view, row0.clone(), 0);
    let b_view = k.at(b_view, col0.clone(), 0);
    let acc0 = k.zeros::<F32>(Shape::new(bm, bn));
    let [acc] = k.pipeline(
        trips,
        stages,
        [acc0],
        |k, step, slot| {
            let koff = step * bk;
            let a_g = k.at(a_view, 0, koff.clone());
            let b_g = k.at(b_view, 0, koff);
            let a_t = k.smem_slot::<BF16>(a_s, slot.clone(), Shape::new(bm, bk));
            let b_t = k.smem_slot::<BF16>(b_s, slot, Shape::new(bn, bk));
            k.stage(a_t, a_g, CopyMode::Async);
            k.stage(b_t, b_g, CopyMode::Async);
        },
        |k, _step, slot, [acc]| {
            let a_t = k.smem_slot::<BF16>(a_s, slot.clone(), Shape::new(bm, bk));
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
