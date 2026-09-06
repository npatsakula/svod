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
        let grouped = if stores.len() == 1 { stores.into_iter().next().unwrap() } else { UOp::group(stores) };
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
    /// `(laneid, inner)` of in-lane fragment-tile `acc` within height/width frag
    /// `frag`: `(frag*frag_extent + acc)*extent + lane_rc(..)`. The `frag` term
    /// (the [`Self::arg_reduce`] cross-frag fold) lifts a stacked frag's LOCAL
    /// `0..extent` index to its GLOBAL position in the reduced axis; it is `0` (no
    /// offset) for a single-fragment source. `frag_extent` is the per-frag span of
    /// the folded axis (`16` for a 16×16 base), so frag `f` starts at element
    /// `f*frag_extent`. Reuses the source fragment's lane map — the same one the
    /// value load uses — and picks the coordinate that *varies with `inner`*
    /// ([`LaneMap::folds_cols`]), since that (with the cross-lane tree) is exactly
    /// the axis the reduce folds. It is the **column** for the normal (gfx942
    /// stride-4) and `InterleavedT` layouts, and the **row** for the `transpose`
    /// (`Col`-layout) and the wave32 even/odd `Interleaved` accumulator — where the
    /// 16-wide reduced axis is split across a lane's `inner` elements and its `L+16`
    /// sibling. So
    /// `row_arg_reduce` on a wave32 accumulator reduces the interleave's
    /// `inner`-carrying axis, exactly as `row_reduce` does (the caller arranges the
    /// tile to match).
    fn axis_index_of(&self, src: &RT<'k>, frag: Option<&Arc<UOp>>, acc: &Arc<UOp>, inner: &Arc<UOp>) -> Arc<UOp> {
        let base_rows = src.base.base.rows as i64;
        let base_cols = src.base.base.cols as i64;
        let transpose = src.layout == TileLayout::Col;
        let (r, c) = src.lane_rc(transpose, &self.laneid(), inner);
        // Which coordinate carries `inner` (the folded axis)?
        let (folded, extent) = if src.base.map.folds_cols(transpose) { (c, base_cols) } else { (r, base_rows) };
        // `acc` (the in-lane reduce frag) and `frag` (the stacked height/width
        // frags the cross-frag fold sweeps) BOTH step by `extent` along the folded
        // axis — the caller stacks the extra frags there — so the global frag index
        // is `frag + acc`. `frag == None` (a single-fragment source) yields the
        // prior `acc*extent + folded` op-tree verbatim, so single-frag callers are
        // bit-identical; `Some(frag)` adds the `frag*extent` cross-frag lift.
        let global_frag = match frag {
            Some(f) => iadd(f, acc),
            None => acc.clone(),
        };
        iadd(&imul(&global_frag, extent), &folded).cast(DType::Int32)
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

    /// Argmin/argmax each row of `src` into `(val, idx)` — the index-carrying
    /// [`Self::row_reduce`]. Folds the reduced-axis `(width, inner)` lane-local
    /// elements and the sibling 16-lane `ds_bpermute` tree, keeping the
    /// extremum's value AND its global column index (ties → smaller index,
    /// matching `Tensor::topk`/`argmin`). The value `RV` is seeded by `dir`
    /// (`+∞`/`−∞`); the index `RV` must be `Int32`. Inside a rolled loop each trip
    /// is a **fresh** reduce (the output pair re-seeds per the enclosing tracked
    /// range), not a running extremum folded across trips.
    ///
    /// The reduced data must be **NaN-free**: the value compare lowers to an
    /// unordered `fcmp ult`, so a NaN can win the fold and propagate as the kept
    /// value (unlike `Tensor::argmin`, whose `==`-mask yields an out-of-range
    /// index) — finite KNN distances satisfy this. A non-16-multiple reduced
    /// width must be `±∞`-padded by the caller so padded lanes never win.
    ///
    /// # Panics
    /// Panics if the group has more than one warp, the kernel is unrolled (the
    /// flat form is a follow-up), the value `RV` dtype is not the (float) source
    /// dtype, or the index `RV` is not `Int32`.
    pub fn row_arg_reduce(&self, val: RV<'k>, idx: RV<'k>, src: &RT<'k>, dir: ArgDir) -> (RV<'k>, RV<'k>) {
        let n = src.shape().len();
        self.arg_reduce(val, idx, src, dir, src.shape()[n - 3] as i64, src.shape()[n - 2] as i64, true)
    }

    /// Argmin/argmax each column of `src` into `(val, idx)` — the transpose of
    /// [`Self::row_arg_reduce`] (folds `(height, inner)`, returns the row index).
    /// Same dtype/padding preconditions.
    pub fn col_arg_reduce(&self, val: RV<'k>, idx: RV<'k>, src: &RT<'k>, dir: ArgDir) -> (RV<'k>, RV<'k>) {
        let n = src.shape().len();
        self.arg_reduce(val, idx, src, dir, src.shape()[n - 2] as i64, src.shape()[n - 3] as i64, false)
    }

    /// Shared arg-reduce body (the index-carrying [`Self::reduce`]): threads a
    /// second `Int32` index accumulator alongside the value through the in-lane
    /// fold and the cross-lane tree. The partner's index rides its OWN
    /// `ds_bpermute` with its value, so it is never re-derived from the lane id.
    /// `outer_end` is the count of stacked height/width fragments along the reduced
    /// axis; `acc_end` is the in-lane reduced dim; `row` selects
    /// `src[outer, acc, inner]` vs `src[acc, outer, inner]`.
    ///
    /// The whole `outer_end` stacked frags fold to a SINGLE `(val[0], idx[0])` pair
    /// per lane, carrying the GLOBAL index `outer*frag_extent + within_frag_local`
    /// (smaller global index wins on ties, via the shared [`arg_fold`]). A
    /// single-fragment source (`outer_end == 1`) keeps the prior structure
    /// bit-for-bit (the `outer` loop is its degenerate one-trip output loop); a
    /// taller source ([`Self::row_arg_reduce`] over the KNN `[TM, query]` Col score
    /// tile's `TM/16` frags) adds the in-primitive cross-frag fold.
    #[allow(clippy::too_many_arguments)]
    fn arg_reduce(
        &self,
        val: RV<'k>,
        idx: RV<'k>,
        src: &RT<'k>,
        dir: ArgDir,
        outer_end: i64,
        acc_end: i64,
        row: bool,
    ) -> (RV<'k>, RV<'k>) {
        assert_eq!(self.warps, 1, "arg_reduce is a single-warp op");
        assert!(!self.ker.unrolled(), "arg_reduce: unrolled (flat) form not yet implemented");
        assert!(src.elem().is_float(), "arg_reduce: value dtype must be float");
        assert_eq!(val.elem(), src.elem(), "arg_reduce: value RV dtype must match src");
        assert_eq!(idx.elem(), &DType::Int32, "arg_reduce: index RV must be Int32");

        // A source one fragment tall along the reduced axis is the prior single-fold
        // form, bit-identical; only a taller source needs the cross-frag fold.
        if outer_end > 1 {
            return self.arg_reduce_folded(val, idx, src, dir, outer_end, acc_end, row);
        }

        let velem = src.elem().clone();
        let ept = src.shape()[src.shape().len() - 1] as i64;
        let (map, tree) = self.fold_plan(src);
        let slots = map.slots();
        assert_eq!(val.shape()[1], slots, "arg_reduce: vector slots must match the source fragment map");
        let val_reg = self.ker.alloc_reg(slots, velem.clone());
        let idx_reg = self.ker.alloc_reg(slots, DType::Int32);

        let outer = self.ker.raw_range(outer_end, AxisType::Loop);

        // Re-init both accumulators each outer iteration: the init stores must
        // depend on `outer` + enclosing tracked loops, or they hoist above the
        // loop and carry stale state (cf. `reduce`). One grouped END closes the
        // tiny init loop once.
        let mut init_deps: SmallVec<[Arc<UOp>; 4]> = smallvec![outer.clone()];
        init_deps.extend(self.ker.tracked_ranges());
        let init_grp = self.arg_init(dir, &velem, &val_reg, &idx_reg, slots, init_deps);

        // In-lane fold over (acc, inner): fold this element's value + its global
        // axis index into the element's slot pair, storing both under one grouped END.
        let acc = self.ker.raw_range(acc_end, AxisType::Reduce);
        let inner = self.ker.raw_range(ept, AxisType::Reduce);
        let slot = map.slot_of(&Idx::from(&inner));
        let va = load_at(
            &val_reg.after(smallvec![init_grp.clone(), acc.clone(), inner.clone()]),
            &[slots],
            std::slice::from_ref(&slot),
        );
        let ia = load_at(
            &idx_reg.after(smallvec![init_grp.clone(), acc.clone(), inner.clone()]),
            &[slots],
            std::slice::from_ref(&slot),
        );
        let src_idx = if row {
            [Idx::from(&outer), Idx::from(&acc), Idx::from(&inner)]
        } else {
            [Idx::from(&acc), Idx::from(&outer), Idx::from(&inner)]
        };
        let vb = load_at(src.uop(), src.shape(), &src_idx);
        let ib = self.axis_index_of(src, None, &acc, &inner);
        let (vf, idf) = arg_fold(dir, &va, &ia, &vb, &ib);
        let v_fold = flat_index(&val_reg, &[slots], std::slice::from_ref(&slot)).store(vf);
        let i_fold = flat_index(&idx_reg, &[slots], &[slot]).store(idf);
        let fold_grp = UOp::group(vec![v_fold, i_fold]).end(smallvec![acc, inner]);

        // Cross-lane fold per slot, then fold into the re-seeded output pair.
        let out_grp = self.arg_output(dir, &tree, &val_reg, &idx_reg, &fold_grp, &val, &idx, &outer);
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

    /// The cross-frag-folding [`Self::arg_reduce`] for a source TALLER than one
    /// fragment along the reduced axis (`outer_end > 1`): the `TM/16` stacked height
    /// (or width) frags fold INTO the in-lane accumulator via a `frag` `Reduce`
    /// range nested OUTSIDE the `(acc, inner)` fold, so the whole stacked reduced
    /// axis collapses to the single per-lane `(val[0], idx[0])`. Each element's
    /// global axis index is `frag*frag_extent + within_frag_local` ([`axis_index_of`]
    /// adds the `frag*extent` lift), so ties break over the true global index — the
    /// in-primitive replacement for the KNN kernel's old inline `fold_partials`.
    #[allow(clippy::too_many_arguments)]
    fn arg_reduce_folded(
        &self,
        val: RV<'k>,
        idx: RV<'k>,
        src: &RT<'k>,
        dir: ArgDir,
        outer_end: i64,
        acc_end: i64,
        row: bool,
    ) -> (RV<'k>, RV<'k>) {
        let velem = src.elem().clone();
        let ept = src.shape()[src.shape().len() - 1] as i64;
        let (map, tree) = self.fold_plan(src);
        let slots = map.slots();
        assert_eq!(val.shape()[1], slots, "arg_reduce: vector slots must match the source fragment map");
        let val_reg = self.ker.alloc_reg(slots, velem.clone());
        let idx_reg = self.ker.alloc_reg(slots, DType::Int32);

        // Re-init both accumulators each enclosing tracked-loop iteration: the init
        // stores must depend on the tracked loops, or they hoist above the loop and
        // carry stale state (cf. the single-fold path's `outer`-keyed re-init). One
        // grouped END closes the tiny init loop once.
        let init_grp = self.arg_init(dir, &velem, &val_reg, &idx_reg, slots, self.ker.tracked_ranges());

        // In-lane fold over (frag, acc, inner): the `frag` Reduce range sweeps the
        // stacked frags so the whole reduced axis collapses into the slot pair; the
        // global axis index carries the `frag*extent` lift.
        let frag = self.ker.raw_range(outer_end, AxisType::Reduce);
        let acc = self.ker.raw_range(acc_end, AxisType::Reduce);
        let inner = self.ker.raw_range(ept, AxisType::Reduce);
        let slot = map.slot_of(&Idx::from(&inner));
        let red_deps = smallvec![init_grp.clone(), frag.clone(), acc.clone(), inner.clone()];
        let va = load_at(&val_reg.after(red_deps.clone()), &[slots], std::slice::from_ref(&slot));
        let ia = load_at(&idx_reg.after(red_deps), &[slots], std::slice::from_ref(&slot));
        let src_idx = if row {
            [Idx::from(&frag), Idx::from(&acc), Idx::from(&inner)]
        } else {
            [Idx::from(&acc), Idx::from(&frag), Idx::from(&inner)]
        };
        let vb = load_at(src.uop(), src.shape(), &src_idx);
        let ib = self.axis_index_of(src, Some(&frag), &acc, &inner);
        let (vf, idf) = arg_fold(dir, &va, &ia, &vb, &ib);
        let v_fold = flat_index(&val_reg, &[slots], std::slice::from_ref(&slot)).store(vf);
        let i_fold = flat_index(&idx_reg, &[slots], &[slot]).store(idf);
        let fold_grp = UOp::group(vec![v_fold, i_fold]).end(smallvec![frag, acc, inner]);

        // The whole reduced axis collapsed to output row 0. A single-trip `out`
        // `Loop` range scopes the output stores (the single-fold path's degenerate
        // end-1 `outer` loop), so the grouped terminal closes exactly once.
        let out = self.ker.raw_range(1, AxisType::Loop);
        let out_grp = self.arg_output(dir, &tree, &val_reg, &idx_reg, &fold_grp, &val, &idx, &out);
        self.finalize_pair(val, idx, out_grp)
    }

    /// The cross-lane fold of slot `slot` shared by both [`Self::arg_reduce`] paths:
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
