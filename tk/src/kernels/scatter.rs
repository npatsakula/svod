//! Rows added into a buffer at indices, in any order and with any collisions.

use crate::build::*;
use crate::ir::*;

/// The rows a block adds: a whole number of registers a thread, on a wave of 32
/// or of 64, over [`WARPS`].
pub const TILE: usize = 256;

/// The block's warps; a tile takes them all.
const WARPS: u32 = 4;

#[derive(Clone, Debug)]
pub struct ScatterSpec {
    /// The target's rows.
    pub rows: usize,
    /// The rows added: `index` holds one index each, `values` one row each.
    pub pairs: usize,
    /// The columns of a row.
    pub d: usize,
}

/// `target[index[p]] += values[p]` for every `p`; an index outside the target
/// (−1, say, for an unused row) drops its row. The parameters are `target`,
/// `index`, `values`.
pub fn scatter_add<T: Accumulable>(spec: &ScatterSpec) -> Program {
    let &ScatterSpec { rows, pairs, d } = spec;
    assert!(rows > 0 && pairs > 0 && d > 0, "a scatter of no elements");
    let mut k = Kernel::new("scatter_add");
    let target = k.param::<T>("target", ParamKind::Accum, rows * d);
    let index = k.param::<I32>("index", ParamKind::In, pairs);
    let values = k.param::<T>("values", ParamKind::In, pairs * d);
    k.grid([Sc::from(pairs.div_ceil(TILE)), Sc::from(1), Sc::from(1)]);
    k.warps(WARPS);
    let row = k.block(0) * TILE + k.row();
    // Rows past the last pair read it, in bounds, and add nothing.
    let pair = row.clone().min(pairs - 1);
    let at = k.load_scalar(index, pair.clone());
    let valid = row.lt(pairs).and(Sc::from(-1).lt(at.clone())).and(at.clone().lt(rows));
    let column = Shape::new(TILE, 1);
    k.loop_vec::<T>(d, vec![], |k, c, _| {
        let v = k.gather(values, c.clone(), pair * d, None, column, None);
        let v = k.load(v);
        let dst = k.gather_accum(target, c, at.max(0) * d, valid, column);
        k.atomic_add(dst, v);
        vec![]
    });
    k.finish()
}
