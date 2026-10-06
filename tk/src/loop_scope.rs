//! [`Loop`] — a scoped loop that owns its tracked range and the loop-carried
//! bookkeeping.
//!
//! In the tile DSL, loop-carried correctness is expressed by hand-threaded ordering
//! edges, and a *missing* edge is a silent wrong answer (or hang), not a compile
//! error. The two error-prone cases:
//!
//! - **Per-iteration re-init** must depend on the loop RANGE, or the linearizer
//!   schedules it with `run_count = 1` (outside the loop) and it carries stale state
//!   every trip (the same hazard `anchor` guards for unrolled reads).
//! - **Loop close** must end *exactly one* terminal store ([`Kernel::endrange_to`]);
//!   carried tiles then read their final value outside the loop via `t.after([end])`.
//!
//! [`Loop`] makes both declarative — `lp.reinit(t)` cannot be forgotten, and
//! `lp.close()` is the one loop-closing edge — while emitting the **identical** UOp
//! graph the hand-threaded form does (`reinit` is exactly `t.after([range])`,
//! `close` is exactly `endrange_to(1)`). It is a builder-API convenience, not a
//! change to the dependency model: the range is still pushed on the kernel's range
//! stack at creation and consumed by [`Loop::close`] / [`Kernel::finish`].

use std::sync::Arc;

use svod_ir::UOp;

use crate::kernel::Kernel;
use crate::tile::RegTile;

/// A tracked loop range plus the kernel it lives in. Created by
/// [`Kernel::loop_static`] / [`Kernel::loop_dynamic`].
pub struct Loop<'k> {
    ker: &'k Kernel,
    range: Arc<UOp>,
}

impl<'k> Loop<'k> {
    pub(crate) fn new(ker: &'k Kernel, range: Arc<UOp>) -> Self {
        Self { ker, range }
    }

    /// The loop counter — for addressing (parity selects, prefetch indices, masks).
    pub fn index(&self) -> &Arc<UOp> {
        &self.range
    }

    /// Pin a per-iteration re-init to this loop: `t.after([range])`. The re-init
    /// re-runs each trip instead of hoisting above the loop with stale state.
    /// Declarative and impossible to omit — the footgun the hand-threaded
    /// `t.after([loop_range])` edges guard against.
    pub fn reinit<T: RegTile<'k>>(&self, t: T) -> T {
        t.after(&self.range)
    }

    /// Close the loop: end the last terminal store around this range and return the
    /// loop-closing `END` node. Use when several accumulators share the loop and
    /// each reads its final value via `tile.after([end])` (one store anchors the
    /// END; the rest chain off it inside the loop). Identical to
    /// [`Kernel::endrange_to`] with one range.
    pub fn close(&self) -> Arc<UOp> {
        self.ker.endrange_to(1)
    }

    /// Close the loop and rebind a single carried tile to its post-loop value:
    /// `t.rewrap(endrange(1))`. The rewrap is taken from the *terminal store's*
    /// buffer (pre-write-after-wrap), so the loop-closing `END` subsumes the carried
    /// tile's in-loop ordering — identical to the idiomatic
    /// `t.rewrap(ker.endrange(1))`. Use for a single loop-carried accumulator (the
    /// FA `o_reg`); the multi-accumulator case uses [`Self::close`] + `after([end])`.
    pub fn close_carry<T: RegTile<'k>>(&self, t: T) -> T {
        t.rewrap(self.ker.endrange(1))
    }

    /// Close the loop with a per-iteration workgroup fence folded into the
    /// loop-closing `END`: the terminal store becomes the barrier's passthrough (so
    /// the fence emits *after* the body's compute) and `commits` (the cross-iteration
    /// prefetch writes) are its deps. Returns the `END` node, consumed only as an
    /// ordering edge by carried accumulators (`tile.after([end])`). Use for a
    /// software-pipelined loop whose tail fence covers both RAW and WAR.
    /// Identical to [`Kernel::endrange_barrier_to`] with one range.
    ///
    /// NOTE: do not use where the barrier-wrapped END would reorder a `WHERE` past its
    /// consumer (the FA causal mask) — there the in-loop [`crate::Group::war_fence2`]
    /// is used with a plain [`Self::close_carry`] instead.
    pub fn close_barrier(&self, commits: smallvec::SmallVec<[Arc<UOp>; 4]>) -> Arc<UOp> {
        self.ker.endrange_barrier_to(1, commits)
    }
}

impl Kernel {
    /// Open a [`Loop`] over a static trip count (wraps [`Kernel::range`]).
    pub fn loop_static(&self, trips: i64) -> Loop<'_> {
        Loop::new(self, self.range(trips))
    }

    /// Open a [`Loop`] over a dynamic (runtime-valued) trip count (wraps
    /// [`Kernel::range_uop`]).
    pub fn loop_dynamic(&self, bound: Arc<UOp>) -> Loop<'_> {
        Loop::new(self, self.range_uop(bound))
    }
}
