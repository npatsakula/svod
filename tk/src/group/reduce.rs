//! Cross-lane reductions: the value-only `row_reduce`/`col_reduce` (shared
//! `reduce`/`reduce_u` bodies) and the index-carrying argmin/argmax
//! `row_arg_reduce`/`col_arg_reduce` (shared `arg_reduce` body). Each folds the
//! lane-local elements into the fragment map's per-lane slots
//! ([`LaneMap::slot_of`](crate::layout::LaneMap::slot_of) — one slot on AMD, the
//! `g`/`g+8` pair on `mma.sync`), then completes across lanes per the map's
//! [`ReduceTree`] (the `ds_bpermute` sibling gather, or the `shfl.bfly` quad).

use std::sync::Arc;

use smallvec::{SmallVec, smallvec};
use svod_dtype::DType;
use svod_ir::{AxisType, ConstValue, UOp};

use super::{ArgDir, Group, arg_fold, iadd, imod, imul};
use crate::index::{Idx, cidx, flat_index, load_at};
use crate::layout::{LaneMap, ReduceTree};
use crate::tile::{RT, RV};
use crate::tiles::TileLayout;

impl<'k> Group<'k> {
    /// The source fragment's lane map and cross-lane tree for this wave.
    fn fold_plan(&self, src: &RT<'k>) -> (LaneMap, ReduceTree) {
        (src.base.map, src.base.map.tree(self.ker.caps.wave_size))
    }

    /// Complete a per-lane `partial` across the wave per `tree`: gather the
    /// siblings' ORIGINAL partials (`(laneid + d) % group`, `ds_bpermute`) or
    /// butterfly the RUNNING value with `laneid ^ m` (`shfl.bfly`).
    fn cross_lane<F>(&self, tree: &ReduceTree, partial: &Arc<UOp>, op: &F) -> Arc<UOp>
    where
        F: Fn(&Arc<UOp>, &Arc<UOp>) -> Arc<UOp>,
    {
        let mut acc = partial.clone();
        match tree {
            ReduceTree::Gather(offsets) => {
                let laneid = self.laneid();
                for &d in offsets {
                    let src_lane = imod(&iadd(&laneid, &cidx(d)), self.group_threads() as i64);
                    acc = op(&acc, &self.shuffle_lane(partial, &src_lane));
                }
            }
            ReduceTree::Butterfly(masks) => {
                for &m in masks {
                    acc = op(&acc, &self.shuffle_xor_lane(&acc, m));
                }
            }
        }
        acc
    }
    /// Reduce each row of `src` into `vec` (tinygrad `row_reduce`): per
    /// row-tile `height`, fold `op` over the `(width, inner)` lane-local
    /// elements into a 1-element REG accumulator, publish it to an LDS scratch
    /// slot at this lane, `barrier`, then fold the three sibling 16-lane slots
    /// (`(laneid + (1+i)*16) % group_threads`) to complete the warp-wide reduce,
    /// and fold the result into `vec[height]`.
    ///
    /// # Panics
    /// Panics if the tile rank is less than 3 (it reads the trailing
    /// `[.., height, width, inner]` dims).
    pub fn row_reduce<F>(&self, vec: RV<'k>, src: &RT<'k>, op: F, init_value: f64) -> RV<'k>
    where
        F: Fn(&Arc<UOp>, &Arc<UOp>) -> Arc<UOp>,
    {
        let n = src.shape().len();
        self.reduce(vec, src, op, init_value, src.shape()[n - 3] as i64, src.shape()[n - 2] as i64, true)
    }

    /// Reduce each column of `src` into `vec` (tinygrad `col_reduce`): the
    /// transpose of [`Self::row_reduce`] — outer loop over column-tiles, accumulate
    /// over the `(height, inner)` elements.
    ///
    /// # Panics
    /// Panics if the tile rank is less than 3, or if the group has more than one
    /// warp.
    pub fn col_reduce<F>(&self, vec: RV<'k>, src: &RT<'k>, op: F, init_value: f64) -> RV<'k>
    where
        F: Fn(&Arc<UOp>, &Arc<UOp>) -> Arc<UOp>,
    {
        let n = src.shape().len();
        self.reduce(vec, src, op, init_value, src.shape()[n - 2] as i64, src.shape()[n - 3] as i64, false)
    }

    /// Shared reduction body. `outer_end` is the tile dim mapped to `vec`
    /// (row-tiles for `row_reduce`, col-tiles for `col_reduce`); `acc_end` is the
    /// in-lane reduce dim; `row` selects the `src[outer, acc, inner]` vs
    /// `src[acc, outer, inner]` element order.
    #[allow(clippy::too_many_arguments)]
    fn reduce<F>(
        &self,
        vec: RV<'k>,
        src: &RT<'k>,
        op: F,
        init_value: f64,
        outer_end: i64,
        acc_end: i64,
        row: bool,
    ) -> RV<'k>
    where
        F: Fn(&Arc<UOp>, &Arc<UOp>) -> Arc<UOp>,
    {
        assert_eq!(self.warps, 1, "reduce is a single-warp op");
        if self.ker.unrolled() {
            return self.reduce_u(vec, src, op, init_value, outer_end, acc_end, row);
        }
        let elem = src.elem().clone();
        let ept = src.shape()[src.shape().len() - 1] as i64;
        let (map, tree) = self.fold_plan(src);
        let slots = map.slots();
        assert_eq!(vec.shape()[1], slots, "reduce: vector slots must match the source fragment map");
        let red_reg = self.ker.alloc_reg(slots, elem.clone());

        let init_val = UOp::const_(elem.clone(), ConstValue::Float(init_value));

        let outer = self.ker.raw_range(outer_end, AxisType::Loop);

        // Re-init the REG accumulator slots each outer iteration: the init store
        // must depend on `outer` (and the enclosing tracked loops), or it hoists
        // above them and the accumulator carries stale state across iterations.
        let mut init_deps: SmallVec<[Arc<UOp>; 4]> = smallvec![outer.clone()];
        init_deps.extend(self.ker.tracked_ranges());
        let init_buf = red_reg.after(init_deps);
        let i = self.ker.raw_range(slots as i64, AxisType::Loop);
        let mut latest = flat_index(&init_buf, &[slots], &[Idx::from(&i)]).store(init_val).end(smallvec![i]);

        // In-lane fold over (acc, inner) into the element's slot. The accumulator
        // read must observe both the prior store (`latest`) and the live reduce
        // ranges, else it hoists.
        let acc = self.ker.raw_range(acc_end, AxisType::Reduce);
        let inner = self.ker.raw_range(ept, AxisType::Reduce);
        let slot = map.slot_of(&Idx::from(&inner));
        let acc_read = load_at(
            &red_reg.after(smallvec![latest.clone(), acc.clone(), inner.clone()]),
            &[slots],
            std::slice::from_ref(&slot),
        );
        let src_idx = if row {
            [Idx::from(&outer), Idx::from(&acc), Idx::from(&inner)]
        } else {
            [Idx::from(&acc), Idx::from(&outer), Idx::from(&inner)]
        };
        let src_v = load_at(src.uop(), src.shape(), &src_idx);
        latest = flat_index(&red_reg, &[slots], &[slot]).store(op(&acc_read, &src_v)).end(smallvec![acc, inner]);

        // Cross-lane completion per slot, straight from registers — no LDS and no
        // barrier: the wave executes the shuffle in lockstep, so every lane's
        // partial is live before any lane reads it. On AMD lane L gathers the
        // ORIGINAL partials of {L+16, L+32, L+48} — bit-for-bit the prior LDS
        // sibling tree; on CUDA the quad butterflies its running value.
        let folded = red_reg.after(smallvec![latest]);
        let stores: Vec<Arc<UOp>> = (0..slots as i64)
            .map(|s| {
                let partial = load_at(&folded, &[slots], &[Idx::Const(s)]);
                let acc = self.cross_lane(&tree, &partial, &op);
                // Fold the lane result into vec[outer, s]: the vec read carries the
                // incoming vec state plus `outer` so it accumulates across iterations.
                let at = [Idx::from(&outer), Idx::Const(s)];
                let vec_acc = load_at(&vec.uop().after(smallvec![outer.clone()]), vec.shape(), &at);
                flat_index(vec.uop(), vec.shape(), &at).store(op(&vec_acc, &acc))
            })
            .collect();
        let grouped = super::group_or_single(stores);
        self.finalize_tile(vec, grouped.end(smallvec![outer]))
    }

    /// Fully **unrolled** [`Self::reduce`]: the `outer`/`acc`/`inner` `RANGE`s
    /// become Rust `for`s, so the in-lane fold and the cross-lane `ds_bpermute`
    /// gather render loop-free (the softmax max/sum reduce must sit in the flat
    /// region with the MFMAs for the attention comb). Bit-identical fold order to
    /// the looped form.
    #[allow(clippy::too_many_arguments)]
    fn reduce_u<F>(
        &self,
        vec: RV<'k>,
        src: &RT<'k>,
        op: F,
        init_value: f64,
        outer_end: i64,
        acc_end: i64,
        row: bool,
    ) -> RV<'k>
    where
        F: Fn(&Arc<UOp>, &Arc<UOp>) -> Arc<UOp>,
    {
        let elem = src.elem().clone();
        let ept = src.shape()[src.shape().len() - 1] as i64;
        let (map, tree) = self.fold_plan(src);
        let slots = map.slots();
        assert_eq!(vec.shape()[1], slots, "reduce: vector slots must match the source fragment map");
        // Anchor the `src` read so a constant-address read of a carried tile is
        // not hoisted out of the enclosing rolled loop (see `Group::anchor`).
        let src_buf = self.anchor(src.uop());

        // Chain the per-`outer` vec stores so the LAST scopes them all under the
        // enclosing (rolled KV) loop's `END`.
        let mut vec_prev: Option<Arc<UOp>> = None;
        for o in 0..outer_end {
            // Fresh per-slot accumulator per `outer` (no cross-`outer` reuse, so
            // the unrolled folds stay independent).
            let red_reg = self.ker.alloc_reg(slots, elem.clone());

            // Re-init: anchor the init store inside the enclosing tracked (KV)
            // loop, or — having only a constant input — it hoists above the rolled
            // loop and the accumulator carries stale state across KV iterations
            // (the looped form's `init_deps` invariant). Slot stores chain.
            let init_buf = red_reg.after(self.ker.tracked_ranges());
            let init_val = UOp::const_(elem.clone(), ConstValue::Float(init_value));
            let mut latest = flat_index(&init_buf, &[slots], &[Idx::Const(0)]).store(init_val.clone());
            for s in 1..slots as i64 {
                latest =
                    flat_index(&red_reg.after(smallvec![latest]), &[slots], &[Idx::Const(s)]).store(init_val.clone());
            }

            // In-lane fold over (acc, inner) into each element's slot: each step
            // observes the prior store.
            for a in 0..acc_end {
                for i in 0..ept {
                    let slot = map.slot_of(&Idx::Const(i));
                    let acc_read =
                        load_at(&red_reg.after(smallvec![latest.clone()]), &[slots], std::slice::from_ref(&slot));
                    let src_idx = if row {
                        [Idx::Const(o), Idx::Const(a), Idx::Const(i)]
                    } else {
                        [Idx::Const(a), Idx::Const(o), Idx::Const(i)]
                    };
                    let src_v = load_at(&src_buf, src.shape(), &src_idx);
                    latest = flat_index(&red_reg, &[slots], &[slot]).store(op(&acc_read, &src_v));
                }
            }

            // Cross-lane completion per slot (the same tree as the looped form),
            // then fold into vec[o, s], carrying the incoming (running) vec state;
            // chain across `outer` (and slots) for loop scoping.
            let folded = red_reg.after(smallvec![latest]);
            for s in 0..slots as i64 {
                let partial = load_at(&folded, &[slots], &[Idx::Const(s)]);
                let acc = self.cross_lane(&tree, &partial, &op);
                let vbuf = match &vec_prev {
                    Some(p) => vec.uop().after(smallvec![p.clone()]),
                    None => self.anchor(vec.uop()),
                };
                let at = [Idx::Const(o), Idx::Const(s)];
                let vec_acc = load_at(&vbuf, vec.shape(), &at);
                vec_prev = Some(flat_index(vec.uop(), vec.shape(), &at).store(op(&vec_acc, &acc)));
            }
        }
        let terminal = vec_prev.expect("reduce_u: at least one outer tile");
        self.finalize_tile(vec, terminal)
    }

    /// The global index, along the **folded** axis, contributed by element
    /// `(laneid, inner)` of the stacked fragment `frag`: `frag*extent + lane_rc(..)`.
    /// The `frag` term lifts a fragment's LOCAL `0..extent` index to its GLOBAL
    /// position in the reduced axis (`extent` is the per-frag span of that axis, `16`
    /// for a 16×16 base and `8` for an Apple 8×8 one), so frag `f` starts at element
    /// `f*extent`. Reuses the source fragment's lane map — the same one the value load
    /// uses — and picks the coordinate that *varies with `inner`*
    /// ([`LaneMap::folds_cols`]), since that (with the cross-lane tree) is exactly the
    /// axis the reduce folds, and the axis [`Self::arg_reduce`] derives its whole plan
    /// from. It is the **column** for the gfx942 stride-4, `mma.sync`,
    /// `simdgroup_matrix` and `InterleavedT` layouts, and the **row** for those read
    /// transposed (a `Col` tile) and for the wave32 even/odd `Interleaved`
    /// accumulator either way — there the 16-wide reduced axis is split across a
    /// lane's `inner` elements and its `L+16` sibling, so the reduce folds rows even
    /// on a `Row` tile (the RDNA callers arrange their tile to match).
    fn axis_index_of(&self, src: &RT<'k>, frag: &Arc<UOp>, inner: &Arc<UOp>) -> Arc<UOp> {
        let base_rows = src.base.base.rows as i64;
        let base_cols = src.base.base.cols as i64;
        let transpose = src.layout == TileLayout::Col;
        let (r, c) = src.lane_rc(transpose, &self.laneid(), inner);
        // Which coordinate carries `inner` (the folded axis)?
        let (folded, extent) = if src.base.map.folds_cols(transpose) { (c, base_cols) } else { (r, base_rows) };
        // `frag` sweeps the stacked fragments along the folded axis, each spanning
        // `extent` elements, so the global index is `frag*extent + folded`.
        iadd(&imul(frag, extent), &folded).cast(DType::Int32)
    }

    /// Record one grouped two-output terminal store and rewrap BOTH result tiles
    /// after it — the [`Group::finalize_tile`](super::Group) analog for
    /// arg-reduce's paired value/index outputs. One `END(GROUP(STORE, STORE))`
    /// closes the shared loop exactly once; a per-store `.end()` would
    /// double-`END` the range (cf. the grouped accumulator store in `mma`).
    fn finalize_pair(&self, val: RV<'k>, idx: RV<'k>, ended: Arc<UOp>) -> (RV<'k>, RV<'k>) {
        self.ker.push_store(ended.clone(), val.uop().clone());
        let val = val.rewrap(val.uop().after(smallvec![ended.clone()]));
        let idx = idx.rewrap(idx.uop().after(smallvec![ended]));
        (val, idx)
    }

    /// Argmin/argmax `src` into `(val, idx)` — the index-carrying `reduce`:
    /// threads an `Int32` index accumulator alongside the value through the in-lane
    /// fold and the cross-lane tree, keeping the extremum's value AND its global
    /// index along the folded axis (ties → smaller index, matching
    /// `Tensor::topk`/`argmin`). The partner's index rides its OWN shuffle with its
    /// value, so it is never re-derived from the lane id. The value `RV` is seeded by
    /// `dir` (`+∞`/`−∞`); the index `RV` must be `Int32`. Inside a rolled loop each
    /// trip is a **fresh** reduce (the output pair re-seeds per the enclosing tracked
    /// range), not a running extremum folded across trips.
    ///
    /// Unlike `reduce`, whose row/col orientation the caller picks, there is
    /// no orientation to pick here: a fragment map folds exactly one axis, and the
    /// reduce must report an index along that same axis. So the folded axis is read
    /// off the map ([`LaneMap::folds_cols`]) — the same source `axis_index_of`
    /// derives the element's global index from, so the fold and the index it reports
    /// cannot disagree. It is the tile's **columns** for a `Row` tile under the CDNA
    /// stride, `mma.sync` and `simdgroup_matrix` maps and its **rows** for a `Col`
    /// tile; the RDNA wave32 even/odd accumulator folds rows either way (its 16-wide
    /// reduced axis is split across a lane's `inner` elements and its `L+16` sibling),
    /// which its callers' tile arrangement accounts for.
    ///
    /// The **reduced** axis's stacked fragments sweep as one `frag` `Reduce` range
    /// nested inside the `out` `Loop` over the **kept** axis's fragments, so the whole
    /// reduced axis collapses to one `(val, idx)` pair per kept fragment, carrying the
    /// global index `frag*frag_extent + within_frag_local`.
    ///
    /// The reduced data must be **NaN-free**: the value compare lowers to an unordered
    /// `fcmp ult`, so a NaN can win the fold and propagate as the kept value (unlike
    /// `Tensor::argmin`, whose `==`-mask yields an out-of-range index) — finite KNN
    /// distances satisfy this. A reduced extent that is not a whole number of
    /// fragments must be `±∞`-padded by the caller so padded lanes never win.
    ///
    /// # Panics
    /// Panics if the group has more than one warp, the kernel is unrolled (the flat
    /// form is a follow-up), the value `RV` dtype is not the (float) source dtype, the
    /// index `RV` is not `Int32`, or the output `RV`s do not hold one entry per kept
    /// fragment.
    pub fn arg_reduce(&self, val: RV<'k>, idx: RV<'k>, src: &RT<'k>, dir: ArgDir) -> (RV<'k>, RV<'k>) {
        assert_eq!(self.warps, 1, "arg_reduce is a single-warp op");
        assert!(!self.ker.unrolled(), "arg_reduce: unrolled (flat) form not yet implemented");
        assert!(src.elem().is_float(), "arg_reduce: value dtype must be float");
        assert_eq!(val.elem(), src.elem(), "arg_reduce: value RV dtype must match src");
        assert_eq!(idx.elem(), &DType::Int32, "arg_reduce: index RV must be Int32");

        let (map, tree) = self.fold_plan(src);
        let reduce_cols = map.folds_cols(src.layout == TileLayout::Col);
        let n = src.shape().len();
        let (height, width) = (src.shape()[n - 3] as i64, src.shape()[n - 2] as i64);
        // The kept axis indexes the output RVs; the reduced axis's frags collapse.
        let (kept_end, red_end) = if reduce_cols { (height, width) } else { (width, height) };
        assert_eq!(val.shape()[0] as i64, kept_end, "arg_reduce: output RVs must hold one entry per kept fragment");

        let velem = src.elem().clone();
        let ept = src.shape()[n - 1] as i64;
        let slots = map.slots();
        assert_eq!(val.shape()[1], slots, "arg_reduce: vector slots must match the source fragment map");
        let val_reg = self.ker.alloc_reg(slots, velem.clone());
        let idx_reg = self.ker.alloc_reg(slots, DType::Int32);

        let out = self.ker.raw_range(kept_end, AxisType::Loop);

        // Re-init both accumulators each `out` iteration: the init stores must depend
        // on `out` + the enclosing tracked loops, or they hoist above the loop and
        // carry stale state (cf. `reduce`). One grouped END closes the tiny init loop.
        let mut init_deps: SmallVec<[Arc<UOp>; 4]> = smallvec![out.clone()];
        init_deps.extend(self.ker.tracked_ranges());
        let init_grp = self.arg_init(dir, &velem, &val_reg, &idx_reg, slots, init_deps);

        // In-lane fold over (frag, inner) into the element's slot pair, storing the
        // value and its global axis index under one grouped END.
        let frag = self.ker.raw_range(red_end, AxisType::Reduce);
        let inner = self.ker.raw_range(ept, AxisType::Reduce);
        let slot = map.slot_of(&Idx::from(&inner));
        let red_deps = smallvec![init_grp.clone(), frag.clone(), inner.clone()];
        let va = load_at(&val_reg.after(red_deps.clone()), &[slots], std::slice::from_ref(&slot));
        let ia = load_at(&idx_reg.after(red_deps), &[slots], std::slice::from_ref(&slot));
        let src_idx = if reduce_cols {
            [Idx::from(&out), Idx::from(&frag), Idx::from(&inner)]
        } else {
            [Idx::from(&frag), Idx::from(&out), Idx::from(&inner)]
        };
        let vb = load_at(src.uop(), src.shape(), &src_idx);
        let ib = self.axis_index_of(src, &frag, &inner);
        let (vf, idf) = arg_fold(dir, &va, &ia, &vb, &ib);
        let v_fold = flat_index(&val_reg, &[slots], std::slice::from_ref(&slot)).store(vf);
        let i_fold = flat_index(&idx_reg, &[slots], &[slot]).store(idf);
        let fold_grp = UOp::group(vec![v_fold, i_fold]).end(smallvec![frag, inner]);

        // Cross-lane fold per slot, then fold into the re-seeded output pair.
        let out_grp = self.arg_output(dir, &tree, &val_reg, &idx_reg, &fold_grp, &val, &idx, &out);
        self.finalize_pair(val, idx, out_grp)
    }

    /// Seed the `slots`-wide `(val_reg, idx_reg)` accumulators to `dir.init()`/`-1`
    /// under one grouped END over a tiny slot loop, ordered after `deps`.
    fn arg_init(
        &self,
        dir: ArgDir,
        velem: &DType,
        val_reg: &Arc<UOp>,
        idx_reg: &Arc<UOp>,
        slots: usize,
        deps: SmallVec<[Arc<UOp>; 4]>,
    ) -> Arc<UOp> {
        let i_range = self.ker.raw_range(slots as i64, AxisType::Loop);
        let v_init = flat_index(&val_reg.after(deps.clone()), &[slots], &[Idx::from(&i_range)])
            .store(UOp::const_(velem.clone(), ConstValue::Float(dir.init())));
        let i_init = flat_index(&idx_reg.after(deps), &[slots], &[Idx::from(&i_range)])
            .store(UOp::const_(DType::Int32, ConstValue::Int(-1)));
        UOp::group(vec![v_init, i_init]).end(smallvec![i_range])
    }

    /// Complete the per-slot `(val_reg, idx_reg)` partials across lanes and fold
    /// them into the output pair at `(out, slot)`, re-seeding the OUTPUT pair to
    /// `dir.init()`/`-1` once per `out` trip AND per enclosing tracked loop, so a
    /// reduce *inside* a rolled loop (the KNN corpus stream) starts fresh each trip
    /// instead of folding onto the previous trip's result — the running-extremum
    /// hoist that an `out`-only edge leaves open (the output RVs' seed `clear_rv`
    /// carries no tracked-loop dependency, so it is hoisted to `run_count = 1`; this
    /// re-seed restores the per-trip start). A reduce with no enclosing tracked loop
    /// re-seeds once, identical to the prior single-fold behavior. The fold then
    /// reads THIS seed, not the carried buffer, so it is a fresh per-trip reduce.
    /// Returns the grouped output store ended on `out`.
    #[allow(clippy::too_many_arguments)]
    fn arg_output(
        &self,
        dir: ArgDir,
        tree: &ReduceTree,
        val_reg: &Arc<UOp>,
        idx_reg: &Arc<UOp>,
        fold_grp: &Arc<UOp>,
        val: &RV<'k>,
        idx: &RV<'k>,
        out: &Arc<UOp>,
    ) -> Arc<UOp> {
        let slots = val.shape()[1];
        let velem = val.elem().clone();
        let mut out_init: SmallVec<[Arc<UOp>; 4]> = smallvec![out.clone()];
        out_init.extend(self.ker.tracked_ranges());
        let (mut seeds, mut stores) = (Vec::with_capacity(2 * slots), Vec::with_capacity(2 * slots));
        let mut folds = Vec::with_capacity(slots);
        for s in 0..slots as i64 {
            let at = [Idx::from(out), Idx::Const(s)];
            let (vacc, iacc) = self.arg_cross_lane(dir, tree, val_reg, idx_reg, fold_grp, s);
            seeds.push(
                flat_index(&val.uop().after(out_init.clone()), val.shape(), &at)
                    .store(UOp::const_(velem.clone(), ConstValue::Float(dir.init()))),
            );
            seeds.push(
                flat_index(&idx.uop().after(out_init.clone()), idx.shape(), &at)
                    .store(UOp::const_(DType::Int32, ConstValue::Int(-1))),
            );
            folds.push((at, vacc, iacc));
        }
        let oseed_grp = UOp::group(seeds);
        for (at, vacc, iacc) in folds {
            let v_in = load_at(&val.uop().after(smallvec![oseed_grp.clone(), out.clone()]), val.shape(), &at);
            let i_in = load_at(&idx.uop().after(smallvec![oseed_grp.clone(), out.clone()]), idx.shape(), &at);
            let (vout, iout) = arg_fold(dir, &v_in, &i_in, &vacc, &iacc);
            stores.push(flat_index(val.uop(), val.shape(), &at).store(vout));
            stores.push(flat_index(idx.uop(), idx.shape(), &at).store(iout));
        }
        UOp::group(stores).end(smallvec![out.clone()])
    }

    /// The cross-lane fold of slot `slot` shared by the arg-reduce body:
    /// read this lane's in-lane `(value, index)` partial once, then fold the
    /// partners' partials per `tree` — value and index each ride their OWN shuffle
    /// so the partner's winning index is transported, not re-derived. Returns the
    /// warp-wide `(value, index)` extremum for this lane.
    fn arg_cross_lane(
        &self,
        dir: ArgDir,
        tree: &ReduceTree,
        val_reg: &Arc<UOp>,
        idx_reg: &Arc<UOp>,
        fold_grp: &Arc<UOp>,
        slot: i64,
    ) -> (Arc<UOp>, Arc<UOp>) {
        let slots = &[val_reg.buffer_size().expect("arg_reduce register")];
        let at = [Idx::Const(slot)];
        let v_partial = load_at(&val_reg.after(smallvec![fold_grp.clone()]), slots, &at);
        let i_partial = load_at(&idx_reg.after(smallvec![fold_grp.clone()]), slots, &at);
        let (mut vacc, mut iacc) = (v_partial.clone(), i_partial.clone());
        match tree {
            ReduceTree::Gather(offsets) => {
                // Pin the `ds_bpermute` lane address to THIS reduce's fold scope by
                // materializing `laneid` through a per-reduce 1-element register. The
                // register round-trip — not a bare `laneid.after(fold_grp)` — is the
                // tinygrad anchoring discipline (`llm/kernels/amd.py` anchors ride
                // register buffers): the pinned AFTER spec admits no ALU/SPECIAL
                // passthrough, and a weak-dtyped AFTER is a fixpoint `pm_lower_weak`
                // never strengthens. The Int32 store also commits the WeakInt SPECIAL
                // chain before the index arithmetic below.
                let lane_reg = self.ker.alloc_reg(1, DType::Int32);
                let lane_store = flat_index(&lane_reg, &[1], &[Idx::Const(0)]).store(self.laneid().cast(DType::Int32));
                // Strong load, weak cast outside for the index arithmetic below — the
                // `pm_lower_weak` PARAM discipline.
                let laneid = load_at(&lane_reg.after(smallvec![lane_store, fold_grp.clone()]), &[1], &[Idx::Const(0)])
                    .cast(DType::WeakInt);
                for &d in offsets {
                    let src_lane = imod(&iadd(&laneid, &cidx(d)), self.group_threads() as i64);
                    let pv = self.shuffle_lane(&v_partial, &src_lane);
                    let pi = self.shuffle_lane(&i_partial, &src_lane);
                    (vacc, iacc) = arg_fold(dir, &vacc, &iacc, &pv, &pi);
                }
            }
            ReduceTree::Butterfly(masks) => {
                for &m in masks {
                    let pv = self.shuffle_xor_lane(&vacc, m);
                    let pi = self.shuffle_xor_lane(&iacc, m);
                    (vacc, iacc) = arg_fold(dir, &vacc, &iacc, &pv, &pi);
                }
            }
        }
        (vacc, iacc)
    }
}
