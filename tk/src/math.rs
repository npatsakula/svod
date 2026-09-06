//! Register-tile elementwise math — a port of tinygrad `tiles.py`'s
//! `TileMathMixin`.
//!
//! Each op lowers through [`Group::map`](crate::Group::map): open a loop per
//! logical dim, load the element, combine it with the right-hand operand, and
//! store the result back. The right-hand operand is one of three flavors:
//! - a **scalar** (`*_scalar`): a per-element constant;
//! - a same-shape **tile** (`add`/`sub`/`mul`/`div`/`maximum`): the matching
//!   element of another [`RT`]/[`RV`];
//! - a **register vector** broadcast into an [`RT`] (`*_rv`): per the RT layout,
//!   `ROW` indexes the vector by the row-tile and `COL` by the col-tile, and the
//!   vector slot by the element's [`LaneMap::slot_of`](crate::layout::LaneMap::slot_of).
//!
//! Faithful to the mixin, `sub` is `add(neg)` and `div` is `mul(recip)`.

use std::sync::Arc;

use svod_ir::{ConstValue, UOp};

use crate::Group;
use crate::index::load_at;
use crate::tile::{RT, RV, RegTile};
use crate::tiles::TileLayout;

// ── element combiners (the inner-op is folded in: sub=add(neg), div=mul(recip)) ─
fn mul(x: &Arc<UOp>, y: &Arc<UOp>) -> Arc<UOp> {
    x.try_mul(y).expect("tk mul")
}
fn add(x: &Arc<UOp>, y: &Arc<UOp>) -> Arc<UOp> {
    x.try_add(y).expect("tk add")
}
fn sub(x: &Arc<UOp>, y: &Arc<UOp>) -> Arc<UOp> {
    x.try_add(&y.neg()).expect("tk sub")
}
fn div(x: &Arc<UOp>, y: &Arc<UOp>) -> Arc<UOp> {
    x.try_mul(&UOp::try_reciprocal(y).expect("tk recip")).expect("tk div")
}
fn maximum(x: &Arc<UOp>, y: &Arc<UOp>) -> Arc<UOp> {
    x.try_max(y).expect("tk max")
}

type Combine = fn(&Arc<UOp>, &Arc<UOp>) -> Arc<UOp>;

impl<'k> Group<'k> {
    // ── scalar RHS ────────────────────────────────────────────────────────────
    fn combine_scalar<T: RegTile<'k>>(&self, a: T, s: f64, f: Combine) -> T {
        let elem = a.elem().clone();
        self.map(a, move |x, _| f(x, &UOp::const_(elem.clone(), ConstValue::Float(s))))
    }

    /// `a + s` element-wise.
    pub fn add_scalar<T: RegTile<'k>>(&self, a: T, s: f64) -> T {
        self.combine_scalar(a, s, add)
    }
    /// `a - s` (folded to `a + (-s)`).
    pub fn sub_scalar<T: RegTile<'k>>(&self, a: T, s: f64) -> T {
        self.combine_scalar(a, -s, add)
    }
    /// `a * s` element-wise.
    pub fn mul_scalar<T: RegTile<'k>>(&self, a: T, s: f64) -> T {
        self.combine_scalar(a, s, mul)
    }
    /// `a / s` (folded to `a * (1/s)`).
    pub fn div_scalar<T: RegTile<'k>>(&self, a: T, s: f64) -> T {
        self.combine_scalar(a, 1.0 / s, mul)
    }
    /// `max(a, s)` element-wise.
    pub fn max_scalar<T: RegTile<'k>>(&self, a: T, s: f64) -> T {
        self.combine_scalar(a, s, maximum)
    }

    // ── same-shape tile RHS ───────────────────────────────────────────────────
    fn combine_tile<T: RegTile<'k>>(&self, a: T, b: &T, f: Combine) -> T {
        assert_eq!(a.shape(), b.shape(), "tile op: shape mismatch");
        // Anchor the RHS read so an unrolled constant-address read of a carried
        // tile is not hoisted out of the enclosing rolled loop (`Group::anchor`).
        let (bbuf, bshape, belem, aelem) =
            (self.anchor(b.uop()), b.shape().to_vec(), b.elem().clone(), a.elem().clone());
        self.map(a, move |x, idx| {
            let mut y = load_at(&bbuf, &bshape, idx);
            if belem != aelem {
                y = y.cast(aelem.clone());
            }
            f(x, &y)
        })
    }

    /// `a + b` for tiles of identical shape.
    ///
    /// # Panics
    /// Panics if `a` and `b` have different shapes.
    pub fn add<T: RegTile<'k>>(&self, a: T, b: &T) -> T {
        self.combine_tile(a, b, add)
    }
    /// `a - b` (`add(neg)`).
    ///
    /// # Panics
    /// Panics if `a` and `b` have different shapes.
    pub fn sub<T: RegTile<'k>>(&self, a: T, b: &T) -> T {
        self.combine_tile(a, b, sub)
    }
    /// `a * b`.
    ///
    /// # Panics
    /// Panics if `a` and `b` have different shapes.
    pub fn mul<T: RegTile<'k>>(&self, a: T, b: &T) -> T {
        self.combine_tile(a, b, mul)
    }
    /// `a / b` (`mul(recip)`).
    ///
    /// # Panics
    /// Panics if `a` and `b` have different shapes.
    pub fn div<T: RegTile<'k>>(&self, a: T, b: &T) -> T {
        self.combine_tile(a, b, div)
    }
    /// `max(a, b)`.
    ///
    /// # Panics
    /// Panics if `a` and `b` have different shapes.
    pub fn maximum<T: RegTile<'k>>(&self, a: T, b: &T) -> T {
        self.combine_tile(a, b, maximum)
    }

    // ── register-vector broadcast into an RT ──────────────────────────────────
    fn combine_rv(&self, a: RT<'k>, v: &RV<'k>, f: Combine) -> RT<'k> {
        let (vbuf, vshape, velem, aelem, layout, map) =
            (self.anchor(v.uop()), v.shape().to_vec(), v.elem().clone(), a.elem().clone(), a.layout, a.base.map);
        assert_eq!(vshape[1], map.slots(), "rv broadcast: vector slots must match the tile's lane map");
        self.map(a, move |x, idx| {
            let sel = match layout {
                TileLayout::Row => idx[0].clone(),
                TileLayout::Col => idx[1].clone(),
            };
            let mut y = load_at(&vbuf, &vshape, &[sel, map.slot_of(&idx[2])]);
            if velem != aelem {
                y = y.cast(aelem.clone());
            }
            f(x, &y)
        })
    }

    /// `a + v`, broadcasting `v` across the RT per its layout.
    pub fn add_rv(&self, a: RT<'k>, v: &RV<'k>) -> RT<'k> {
        self.combine_rv(a, v, add)
    }
    /// `a - v` (`add(neg)`).
    pub fn sub_rv(&self, a: RT<'k>, v: &RV<'k>) -> RT<'k> {
        self.combine_rv(a, v, sub)
    }
    /// `a * v`.
    pub fn mul_rv(&self, a: RT<'k>, v: &RV<'k>) -> RT<'k> {
        self.combine_rv(a, v, mul)
    }
    /// `a / v` (`mul(recip)`).
    pub fn div_rv(&self, a: RT<'k>, v: &RV<'k>) -> RT<'k> {
        self.combine_rv(a, v, div)
    }

    // ── unary ─────────────────────────────────────────────────────────────────
    /// `exp2(a)` element-wise.
    pub fn exp2<T: RegTile<'k>>(&self, a: T) -> T {
        self.map(a, |x, _| x.try_exp2().expect("tk exp2"))
    }
}
