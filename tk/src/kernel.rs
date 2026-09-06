//! The [`Kernel`] builder — the eager context that mints ranges, allocations,
//! and tiles, and assembles the final hand-lowered SINK.
//!
//! All builder methods take `&self` over interior-mutable counters/stacks, so
//! many tiles and [`crate::group::Group`]s can borrow one `&Kernel` at once —
//! the borrow-checker-friendly mapping of tinygrad's `Kernel` context manager.

use std::cell::{Cell, RefCell};
use std::sync::Arc;

use smallvec::{SmallVec, smallvec};
use svod_dtype::{AddrSpace, DType};
use svod_ir::{AxisId, AxisType, KernelInfo, Op, UOp};

use crate::ArchCaps;
use crate::index::cidx;

pub struct Kernel {
    pub name: String,
    /// The arch-derived caps (wave size, reduce tree, WMMA arch) the builder
    /// threads instead of the wave64 literals.
    pub caps: ArchCaps,
    /// `blockIdx.{x,y,z}` as `Special` ops (only rendered if referenced).
    pub block_idx: [Arc<UOp>; 3],
    /// `threadIdx.x` as a `Special` op.
    pub thread_idx: Arc<UOp>,

    /// Flat (1-D pointer) `Param` placeholders, one per bound buffer in
    /// declaration order. The kernel body references these; the concrete buffers
    /// bind positionally at launch (`ProgramSpec.globals` slot order).
    globals: Vec<Arc<UOp>>,
    global_slot: Cell<usize>,
    shared_slot: Cell<usize>,
    reg_slot: Cell<usize>,
    range_id: Cell<usize>,

    /// Tracked ranges, closed together by [`Kernel::finish`] / [`Kernel::endrange`].
    range_stack: RefCell<Vec<Arc<UOp>>>,
    /// Terminal `(store, buffer)` pairs, consumed by `finish`/`endrange`.
    store_stack: RefCell<Vec<(Arc<UOp>, Arc<UOp>)>>,

    /// When set, the register compute primitives ([`crate::Group`]'s
    /// `mma`/`map`/`copy`/`clear`/`transpose`/`reduce`) emit **fully unrolled**
    /// (Rust-`for`, no inner `RANGE`) bodies instead of looped ones, so the FA
    /// QKᵀ/softmax/A·V render as one flat schedulable LLVM region the attention
    /// scheduling comb can weave. Off by default (the looped form keeps the matmul
    /// + rolled-FA IR compact); the cross-tile-pipeline FA builder opts in.
    unroll: Cell<bool>,
}

impl Kernel {
    /// Build a kernel context bound to concrete realized buffers. `buffers` are
    /// the `BUFFER` UOps of the realized tensors in declaration order (output(s)
    /// first, then inputs). Each is converted to a flat 1-D `Param` placeholder
    /// (slot = declaration index); [`Kernel::next_global`] hands those out as GL
    /// tiles bind them, and [`crate::launch`] binds the concrete buffers
    /// positionally at dispatch — the svod analog of tinygrad `sink.call(bufs)`.
    ///
    /// # Panics
    /// In a debug build, panics if the arch wave size is neither 32 nor 64 — tk
    /// only has fragment-layout tables for wave32/wave64.
    pub fn new(name: impl Into<String>, grid: [i64; 3], block: i64, buffers: Vec<Arc<UOp>>, caps: ArchCaps) -> Self {
        // tk's lane math (reduce tree, sibling folds) is calibrated for wave64
        // (gfx942 CDNA) and wave32 (gfx11 RDNA, CUDA). Any other wave size is
        // gated loudly (such an arch is also absent from every kernel's `ArchSet`
        // and falls back before here).
        debug_assert!(
            matches!(caps.wave_size, 32 | 64),
            "tk supports wave32/wave64 lane layouts; got wave{} ({:?})",
            caps.wave_size,
            caps.arch
        );
        let globals = buffers.iter().enumerate().map(|(slot, buf)| flat_param(slot, buf)).collect();
        let block_idx = [
            UOp::special(cidx(grid[0]), "gidx0".to_string()),
            UOp::special(cidx(grid[1]), "gidx1".to_string()),
            UOp::special(cidx(grid[2]), "gidx2".to_string()),
        ];
        let thread_idx = UOp::special(cidx(block), "lidx0".to_string());
        Kernel {
            name: name.into(),
            caps,
            block_idx,
            thread_idx,
            globals,
            global_slot: Cell::new(0),
            shared_slot: Cell::new(0),
            reg_slot: Cell::new(0),
            range_id: Cell::new(0),
            range_stack: RefCell::new(Vec::new()),
            store_stack: RefCell::new(Vec::new()),
            unroll: Cell::new(false),
        }
    }

    /// Opt into fully-unrolled register compute (see [`Self::unrolled`]). The
    /// cross-tile-pipeline FA builder sets this so the QKᵀ/softmax/A·V render as a
    /// flat schedulable region.
    pub fn set_unroll(&self, on: bool) {
        self.unroll.set(on);
    }

    /// Whether the register compute primitives should emit fully-unrolled bodies.
    pub fn unrolled(&self) -> bool {
        self.unroll.get()
    }

    // ── lane / warp helpers ────────────────────────────────────────────────

    /// The wave (warp) index of the current thread within its workgroup —
    /// `threadIdx / wave_size`, derived from the flat thread id.
    ///
    /// # Panics
    /// Panics if the `Index`-typed division cannot be constructed.
    pub fn warpid(&self) -> Arc<UOp> {
        self.thread_idx.try_div(&cidx(self.caps.wave_size as i64)).expect("warpid: index div")
    }
    /// The lane index of the current thread within its wave —
    /// `threadIdx % wave_size`, derived from the flat thread id.
    ///
    /// # Panics
    /// Panics if the `Index`-typed modulo cannot be constructed.
    pub fn laneid(&self) -> Arc<UOp> {
        self.thread_idx.try_mod(&cidx(self.caps.wave_size as i64)).expect("laneid: index mod")
    }

    // ── ranges ─────────────────────────────────────────────────────────────

    fn fresh_range(&self, end: i64, axis_type: AxisType) -> Arc<UOp> {
        let rid = self.range_id.get();
        self.range_id.set(rid + 1);
        UOp::range_axis(cidx(end), AxisId::Renumbered(rid), axis_type)
    }

    /// A tracked `Loop` range closed by `finish`.
    pub fn range(&self, end: i64) -> Arc<UOp> {
        let r = self.fresh_range(end, AxisType::Loop);
        self.range_stack.borrow_mut().push(r.clone());
        r
    }

    /// A tracked `Loop` range with a *dynamic* (runtime-valued) end — e.g. a
    /// `Special`-derived bound for causal block-skip (`q_seq + 1`) — closed by
    /// `finish`/`endrange` like [`Kernel::range`]. `end` must be `Index`-typed
    /// (or const-coercible; `UOp::range_axis` handles the coercion). The renderer
    /// lowers it to a real runtime-trip loop.
    pub fn range_uop(&self, end: Arc<UOp>) -> Arc<UOp> {
        let rid = self.range_id.get();
        self.range_id.set(rid + 1);
        let r = UOp::range_axis(end, AxisId::Renumbered(rid), AxisType::Loop);
        self.range_stack.borrow_mut().push(r.clone());
        r
    }

    /// A range with an explicit axis type; tracked only when `track`.
    pub fn range_typed(&self, end: i64, axis_type: AxisType, track: bool) -> Arc<UOp> {
        let r = self.fresh_range(end, axis_type);
        if track {
            self.range_stack.borrow_mut().push(r.clone());
        }
        r
    }

    /// An untracked range, closed manually via `store(..).end([r])`.
    pub fn raw_range(&self, end: i64, axis_type: AxisType) -> Arc<UOp> {
        self.fresh_range(end, axis_type)
    }

    /// The currently tracked (outer) ranges — tinygrad `ker.range_stack`. A
    /// reduction's per-iteration re-init must depend on these so it re-runs once
    /// per outer-loop iteration instead of hoisting above the enclosing loops.
    pub fn tracked_ranges(&self) -> SmallVec<[Arc<UOp>; 4]> {
        self.range_stack.borrow().iter().cloned().collect()
    }

    // ── allocations ────────────────────────────────────────────────────────

    /// Allocate shared (LDS) memory. The slot is a per-kernel monotonic id (the
    /// renderer names LDS `@local{id}`, so it MUST be unique within a kernel).
    pub fn alloc_local(&self, flat_size: usize, elem: DType) -> Arc<UOp> {
        let slot = self.shared_slot.get();
        self.shared_slot.set(slot + 1);
        UOp::buffer(slot, flat_size, elem, AddrSpace::Local, None)
    }

    /// Allocate register (per-lane) memory. The id is a per-kernel monotonic slot
    /// (NOT a global counter): structurally-identical kernels (e.g. the same FA
    /// kernel built once per transformer layer) allocate regs in the same order →
    /// identical `DefineReg` ids → one content hash → a single compile, instead of
    /// recompiling per instance (tinygrad parity: `DEFINE_REG` arg is a per-kernel
    /// slot). The renderer renumbers regs locally, so the id is identity-only.
    pub fn alloc_reg(&self, flat_size: usize, elem: DType) -> Arc<UOp> {
        let id = self.reg_slot.get();
        self.reg_slot.set(id + 1);
        UOp::buffer(id, flat_size, elem, AddrSpace::Reg, None)
    }

    /// Hand out the next global buffer placeholder (a flat 1-D `Param`) as a GL
    /// tile binds it. Already flat — no `flat_ptr` unwrap is needed.
    ///
    /// # Panics
    /// Panics (index out of bounds) if called more times than the number of
    /// declared GLOBAL buffers.
    pub fn next_global(&self) -> Arc<UOp> {
        let slot = self.global_slot.get();
        self.global_slot.set(slot + 1);
        self.globals[slot].clone()
    }

    // ── store bookkeeping / finalization ───────────────────────────────────

    /// Record a terminal `(store, buffer)` pair for `finish`/`endrange`.
    pub fn push_store(&self, store: Arc<UOp>, buf: Arc<UOp>) {
        self.store_stack.borrow_mut().push((store, buf));
    }

    /// Close every tracked range and group the last `stores` terminal stores
    /// into the final kernel SINK (carrying `opts_to_apply = Some(vec![])`, the
    /// tinygrad `()` analog, so the optimizer applies zero schedule opts to this
    /// hand-lowered body; it still runs the shared pre/post-optimization pipeline).
    ///
    /// # Panics
    /// Panics on store-stack underflow — fewer terminal stores were recorded
    /// (via [`Self::push_store`]) than the `stores` requested here.
    pub fn finish(&self, stores: usize) -> Arc<UOp> {
        let rngs: SmallVec<[Arc<UOp>; 4]> = self.range_stack.borrow_mut().drain(..).collect();

        // A RANGE admits exactly one END (else a double loop footer mis-scopes
        // the linearizer). With multiple stores we'd clone the SAME outer `rngs`
        // into each store's `.end()`, double-ending them — so a kernel that leaves
        // outer ranges open at `finish` must have a single store. In-tree kernels
        // close their loops before `finish`, so `rngs` is empty and this holds.
        debug_assert!(
            rngs.is_empty() || stores == 1,
            "finish: {stores} stores with {} open outer range(s) would double-end a RANGE",
            rngs.len()
        );

        let mut store_uops = Vec::with_capacity(stores);
        for _ in 0..stores {
            let (store, _buf) = self.store_stack.borrow_mut().pop().expect("finish: store stack underflow");
            store_uops.push(store);
        }
        store_uops.reverse(); // restore declaration order

        // Each terminal store is already an `END(STORE)` / `END(GROUP(STORE..))`
        // closing its own loops (the Group ops self-end so their `After`-rewraps
        // carry a completed-loop edge). svod's GROUP may only hold *bare* STOREs,
        // so we don't re-wrap these in a GROUP; instead close any remaining
        // tracked (outer) ranges around each — a no-op `END` when `rngs` is empty
        // (the matmul, whose tile loop `endrange` already consumed) — and SINK
        // them directly (the native `SINK(END(STORE, ..))` kernel shape).
        let sources: Vec<Arc<UOp>> = store_uops.into_iter().map(|s| s.end(rngs.clone())).collect();
        UOp::sink_with_info(
            sources,
            KernelInfo { opts_to_apply: Some(vec![]), name: Some(self.name.clone()), ..Default::default() },
        )
    }

    /// Close `ranges` inner (accumulation) loops around the last store and
    /// return the store's buffer rewrapped with the close as a dependency.
    ///
    /// # Panics
    /// Panics on store-stack underflow (no recorded store to close) or
    /// range-stack underflow (fewer open ranges than `ranges`).
    pub fn endrange(&self, ranges: usize) -> Arc<UOp> {
        let (store, buf) = self.store_stack.borrow_mut().pop().expect("endrange: store stack underflow");
        let mut rngs: Vec<Arc<UOp>> = Vec::with_capacity(ranges);
        for _ in 0..ranges {
            rngs.push(self.range_stack.borrow_mut().pop().expect("endrange: range stack underflow"));
        }
        let ended = store.end(SmallVec::from_vec(rngs));
        buf.after(smallvec![ended])
    }

    /// Like [`Self::endrange`] but returns the loop-closing `END` node directly
    /// (rather than one rewrapped buffer), so several accumulators sharing one K
    /// loop can each be rewrapped `.after([end])` to read the final value
    /// outside the loop. Only the last store is ended (a `RANGE` may have a
    /// single `END`, else a double loop footer): the caller must chain the
    /// other accumulators' stores into it (via a shared input) so they are
    /// scoped inside the loop and survive dead-code elimination.
    ///
    /// # Panics
    /// Panics on store-stack underflow (no recorded store to close) or
    /// range-stack underflow (fewer open ranges than `ranges`).
    pub fn endrange_to(&self, ranges: usize) -> Arc<UOp> {
        let (store, _buf) = self.store_stack.borrow_mut().pop().expect("endrange_to: store stack underflow");
        let mut rngs: Vec<Arc<UOp>> = Vec::with_capacity(ranges);
        for _ in 0..ranges {
            rngs.push(self.range_stack.borrow_mut().pop().expect("endrange_to: range stack underflow"));
        }
        store.end(SmallVec::from_vec(rngs))
    }

    /// Like [`Self::endrange_to`] but wraps the popped terminal store in a
    /// workgroup `BARRIER` carrying `deps`, then closes the loop around the
    /// barrier. The barrier becomes the loop-closing `END`'s computation, so it
    /// is kept live and loop-scoped **with no value consumer** — exactly what a
    /// software-pipelined K-loop's per-iteration fence needs: the barrier's
    /// passthrough (the store, which transitively reads the gathers/MFMAs)
    /// orders it *after* the loop body's compute, while `deps` (the prefetch
    /// commits) order it after the cross-iteration writes.
    ///
    /// This is the tinygrad `STORE.barrier(...)` idiom (and what
    /// [`crate::Group::commit_reg_to_local`] already does). It is legal where the
    /// analogous `.after()` is not: `barrier` accepts an `End` passthrough, but
    /// `after` rejects it (its passthrough must be data-producing); and only a
    /// `Barrier` actually emits the `s.barrier` fence. Anchoring a consumer-less
    /// barrier *post*-loop instead trips the CFG-builder cycle check.
    ///
    /// # Panics
    /// Panics on store-stack underflow (no recorded store to close) or
    /// range-stack underflow (fewer open ranges than `ranges`).
    pub fn endrange_barrier_to(&self, ranges: usize, deps: SmallVec<[Arc<UOp>; 4]>) -> Arc<UOp> {
        let (store, _buf) = self.store_stack.borrow_mut().pop().expect("endrange_barrier_to: store stack underflow");
        let wrapped = if deps.is_empty() { store } else { store.barrier(deps) };
        let mut rngs: Vec<Arc<UOp>> = Vec::with_capacity(ranges);
        for _ in 0..ranges {
            rngs.push(self.range_stack.borrow_mut().pop().expect("endrange_barrier_to: range stack underflow"));
        }
        wrapped.end(SmallVec::from_vec(rngs))
    }
}

/// Mint a flat 1-D `Param` placeholder for a concrete `BUFFER` UOp at `slot`.
///
/// The buffer's element dtype + flat size become a `Ptr` param (global address
/// space), exactly the shape `placeholder_like` builds for a rank-≤1 source.
/// Keeping it flat (no RESHAPE wrapper) means GL tiles index it directly — the
/// renderer's `globals` derivation counts this `Param` and `launch` binds the
/// concrete buffer to its slot. A reshaped/lazy source is unwrapped to its base
/// `BUFFER` first, and a non-buffer source (already a `Param`/`Ptr`) is reused.
fn flat_param(slot: usize, src: &Arc<UOp>) -> Arc<UOp> {
    let base = src.base();
    match base.op() {
        Op::Buffer(..) => {
            let size = base.buffer_size().expect("flat_param requires a concrete BUFFER shape");
            let elem = base.dtype();
            UOp::param(slot, size, elem, None)
        }
        // Already a buffer-like pointer (e.g. a pre-built Param): reuse as-is.
        _ => base,
    }
}
