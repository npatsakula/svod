//! The load/store path: the public [`Group::load`](super::Group)/`store` entry
//! points (their legal address-space pairs resolved at compile time via
//! `LoadInto`/`StoreInto`), the coalesced GLOBAL↔LDS fills (scalar and vectorized,
//! plus the register-staged prefetch and the CUDA `cp.async` fill), and the
//! GLOBAL/LOCAL↔REG fragment gather/scatter hops (the LOCAL→REG gather of a 16-bit
//! `mma.sync` fragment is one `ldmatrix.x4` on CUDA).

use std::sync::Arc;

use smallvec::{SmallVec, smallvec};
use svod_codegen::llvm::nvptx::smem::{cp_async_16, cp_async_commit, cp_async_wait, ldmatrix};
use svod_ir::{AxisType, ConstValue, Op, UOp};

use super::{Group, MoveIdx, iadd, idiv, idx_mul, imod, imul, wave_offset};
use crate::index::{Idx, cidx, flat_index, flat_offset, index_off, index_off_gated, load_at, load_off, load_off_gated};
use crate::layout::LdmatrixX4;
use crate::tile::{GL, RT, ST};
use crate::tiles::TileLayout;
use svod_ir::ops;

/// Scalar geometry of the coalesced GLOBAL↔LDS fill for one ST tile (the part
/// independent of the global source / tile position). Shared by the direct fill
/// and the register-staged prefetch so both address LDS identically.
struct LdsGeom {
    ept: i64,
    st_cols: i64,
    memcpy_per_row: i64,
    base_rows: i64,
    base_cols: i64,
    total_calls: i64,
    num_valid: i64,
    clamp: bool,
}

impl<'k> Group<'k> {
    /// Move data into `dst` (tinygrad `Group.load`), with the legal (dst, src)
    /// address-space pair resolved at **compile time** via [`LoadInto`](super::LoadInto):
    /// ST←GL (coalesced fill + barrier), RT←ST / RT←GL (fragment gather). An illegal pair
    /// (e.g. RT←RT) has no impl, so it is a compile error — not a runtime panic:
    ///
    /// ```compile_fail
    /// # use svod_tk::{ArchCaps, Kernel, MoveIdx};
    /// # use svod_tk::tiles::{RT_16X16, TileLayout};
    /// # use svod_dtype::DType;
    /// let ker = Kernel::new("x", [1, 1, 1], 64, vec![], ArchCaps::GFX942);
    /// let g = ker.warp();
    /// let a = ker.rt((16, 16), DType::Float32, TileLayout::Row, RT_16X16);
    /// let b = ker.rt((16, 16), DType::Float32, TileLayout::Row, RT_16X16);
    /// let _ = g.load(a, b, MoveIdx::default()); // RT ← RT: no LoadInto impl ⇒ won't compile
    /// ```
    ///
    /// # Panics
    /// A LOCAL→REG (`ST` → `RT`) load panics unless the REG tile fits within the
    /// ST tile — its fragment-grid rows and cols must each be `<=` the ST's.
    pub fn load<Dst, Src>(&self, dst: Dst, src: Src, ix: MoveIdx) -> Src::Output
    where
        Src: super::LoadInto<'k, Dst>,
    {
        src.load_into(self, dst, ix)
    }

    /// Coalesced GLOBAL→LOCAL fill **without** the trailing workgroup barrier —
    /// the software-pipeline primitive (stage ii). The caller is responsible
    /// for inserting one barrier per buffer before the LDS→REG gather (so the
    /// fill is visible) and before the next overwrite (the WAR edge); decoupling
    /// the fill from its sync lets the next block's GLOBAL loads issue *ahead* of
    /// the current block's compute, overlapping memory latency with the MFMA.
    ///
    /// # Panics
    /// Panics if `axis` is out of range for the GLOBAL source's rank (the
    /// row-stride is the product of the dims after `axis`).
    pub fn fill_local_nobar(&self, dst: ST, src: GL, idxs: &[Idx], axis: usize) -> ST {
        self.load_global_to_local(dst, &src, idxs, axis, false)
    }

    /// Stage one tile of `src` (GLOBAL) into a fresh per-lane register buffer —
    /// the GLOBAL→VGPR half of the register prefetch. Uses the *same*
    /// coalesced per-lane addressing as [`Self::load_global_to_local`], but lands
    /// the loaded (unswizzled) values in a flat `[total_calls, ept]` DEFINE_REG
    /// instead of LDS, so the load can be issued ahead of the consuming MFMAs.
    /// Commit it with [`Self::commit_reg_to_local`] (same `st`/`idxs`/`axis`).
    ///
    /// # Panics
    /// Panics if `axis` is out of range for the GLOBAL source's rank (the
    /// row-stride is the product of the dims after `axis`).
    pub fn stage_global_to_reg(&self, st: &ST, src: &GL, idxs: &[Idx], axis: usize) -> Arc<UOp> {
        let geom = self.lds_fill_geom(st);
        let row_stride: i64 = src.shape()[axis + 1..].iter().product::<usize>() as i64;
        let src_i_base = Self::tile_base(st, src, idxs, axis);

        let stage = self.ker.alloc_reg((geom.total_calls * geom.ept) as usize, st.elem().clone());
        let outer = self.ker.raw_range(geom.total_calls, AxisType::Loop);
        let inner = self.ker.raw_range(geom.ept, AxisType::Upcast);
        let (height, width, row, col) = self.fill_lane_rc(&geom, &outer, &inner);

        let off = iadd(
            &src_i_base,
            &iadd(
                &iadd(&imul(&height, geom.base_rows * row_stride), &imul(&width, geom.base_cols)),
                &iadd(&imul(&row, row_stride), &col),
            ),
        );
        let mut load = load_off(src.uop(), off);
        if src.elem() != st.elem() {
            load = load.cast(st.elem().clone());
        }
        let stage_shape = [geom.total_calls as usize, geom.ept as usize];
        let stored = flat_index(&stage, &stage_shape, &[Idx::from(&outer), Idx::from(&inner)])
            .store(load)
            .end(smallvec![outer, inner]);
        self.ker.push_store(stored.clone(), stage.clone());
        stage.after(smallvec![stored])
    }

    /// Commit a staged register buffer (from [`Self::stage_global_to_reg`]) into
    /// the swizzled LDS tile — the VGPR→LDS `ds_write` half of the prefetch.
    /// Recomputes the identical per-lane addressing. Ends in a workgroup barrier
    /// when `barrier` (the single-buffer commit); the double-buffered pipeline
    /// passes `false` and shares one barrier per iteration.
    pub fn commit_reg_to_local(&self, st: ST, stage: &Arc<UOp>, barrier: bool) -> ST {
        // The LDS destination geometry is fully determined by the tile shape (the
        // global tile position only mattered when *staging* into the registers).
        let geom = self.lds_fill_geom(&st);
        let outer = self.ker.raw_range(geom.total_calls, AxisType::Loop);
        let inner = self.ker.raw_range(geom.ept, AxisType::Upcast);
        let (height, width, row, col) = self.fill_lane_rc(&geom, &outer, &inner);
        let (srow, scol) = st.base.swizzle.swizzle_rc(row, col, st.base.base.cols, st.elem().base());

        let stage_shape = [geom.total_calls as usize, geom.ept as usize];
        let load = load_at(stage, &stage_shape, &[Idx::from(&outer), Idx::from(&inner)]);
        let stored = st_index(&st, &[Idx::Uop(height), Idx::Uop(width), Idx::Uop(srow), Idx::Uop(scol)])
            .store(load)
            .end(smallvec![outer, inner]);
        let stored = if barrier { stored.barrier(SmallVec::new()) } else { stored };
        self.finalize_st(st, stored)
    }

    /// Move a register tile `src` out into `dst` (tinygrad `Group.store`), with the
    /// legal address-space pair resolved at **compile time** via [`StoreInto`](super::StoreInto):
    /// RT→ST (fragment scatter, the layout-transpose hop) and RT→GLOBAL (coalesced
    /// write-back). An illegal pair has no impl, so it is a compile error, not a
    /// runtime panic. `ix` carries the wave/global `block` offset and the REG-side
    /// `frag` offset; `ix.axis` is the global-tile row-stride split.
    ///
    /// # Panics
    /// A REG→GLOBAL store panics if `ix.axis` is out of range for the GLOBAL
    /// destination's rank (the row-stride is the product of the dims after it).
    pub fn store<Dst, Src>(&self, dst: Dst, src: Src, ix: MoveIdx) -> Src::Output
    where
        Src: super::StoreInto<'k, Dst>,
    {
        src.store_into(self, dst, ix)
    }

    /// Cross-wave WAR fence over two just-loaded register reads `a`/`b`: builds ONE
    /// workgroup `Barrier` (passthrough `a`, deps = `b` + `extra` — the cross-iteration
    /// prefetch commits the double-buffer pipeline folds in) and returns BOTH reads
    /// re-threaded `.after([sync])`. The barrier is internal (never returned), so a
    /// read cannot be left un-fenced (you get the fenced tiles back) and nothing can
    /// depend on the barrier as a value (the AMD renderer emits the `s.barrier` fence
    /// but registers no SSA value for it). Emits the identical graph as the hand-built
    /// `a.uop().barrier([b] + extra)` + per-read `.after([sync])`.
    pub fn war_fence2<T: crate::tile::RegTile<'k>>(&self, a: T, b: T, extra: &[Arc<UOp>]) -> (T, T) {
        let mut deps: SmallVec<[Arc<UOp>; 4]> = smallvec![b.uop().clone()];
        deps.extend(extra.iter().cloned());
        let sync = a.uop().barrier(deps);
        (a.after(smallvec![sync.clone()]), b.after(smallvec![sync]))
    }

    /// The [`LdsGeom`] for filling `st` collaboratively across all group
    /// threads (`elements_per_thread`, pass count, last-pass clamp).
    fn lds_fill_geom(&self, st: &ST) -> LdsGeom {
        let ept = st.base.base.elements_per_thread() as i64;
        let st_cols = st.cols as i64;
        let base_rows = st.base.base.rows as i64;
        let base_cols = st.base.base.cols as i64;
        let num_elements = st.base.base.num_elements() as i64;
        let n = st.shape().len();
        let total_elems = st.shape()[n - 4] as i64 * st.shape()[n - 3] as i64 * num_elements;
        let slots = self.group_threads() as i64 * ept;
        let total_calls = (total_elems + slots - 1) / slots;
        LdsGeom {
            ept,
            st_cols,
            memcpy_per_row: st_cols / ept,
            base_rows,
            base_cols,
            total_calls,
            num_valid: total_elems / ept,
            clamp: total_calls * slots != total_elems,
        }
    }

    /// The `(height, width, row, col)` LDS fragment coordinate this lane fills at
    /// collaborative pass `(outer, inner)` — the shared per-lane addressing of
    /// the direct fill and the register-staged prefetch (over-subscribed last
    /// pass clamps to the final valid fragment, idempotent).
    fn fill_lane_rc(
        &self,
        geom: &LdsGeom,
        outer: &Arc<UOp>,
        inner: &Arc<UOp>,
    ) -> (Arc<UOp>, Arc<UOp>, Arc<UOp>, Arc<UOp>) {
        let mut load_idx = iadd(&imul(outer, self.group_threads() as i64), &self.laneid());
        if geom.clamp {
            let cond = load_idx.try_cmplt(&cidx(geom.num_valid)).expect("load_idx < num_valid");
            load_idx = UOp::try_where(cond, load_idx.clone(), cidx(geom.num_valid - 1)).expect("clamp load_idx");
        }
        let row0 = idiv(&load_idx, geom.memcpy_per_row);
        let col0 = iadd(&imod(&imul(&load_idx, geom.ept), geom.st_cols), inner);
        (
            idiv(&row0, geom.base_rows),
            idiv(&col0, geom.base_cols),
            imod(&row0, geom.base_rows),
            imod(&col0, geom.base_cols),
        )
    }

    /// Coalesced GLOBAL→LOCAL fill: every group thread streams
    /// `elements_per_thread` contiguous global elements into the swizzled LDS
    /// tile. When `barrier`, it is closed with a workgroup barrier so the
    /// subsequent gather sees it (the default); the software-pipeline path passes
    /// `false` and inserts the barrier itself (see [`Self::fill_local_nobar`]).
    pub(super) fn load_global_to_local(&self, st: ST, src: &GL, idxs: &[Idx], axis: usize, barrier: bool) -> ST {
        let row_stride: i64 = src.shape()[axis + 1..].iter().product::<usize>() as i64;
        let src_i_base = Self::tile_base(&st, src, idxs, axis);

        let ept = st.base.base.elements_per_thread() as i64;
        let st_cols = st.cols as i64;
        let memcpy_per_row = st_cols / ept;
        let base_rows = st.base.base.rows as i64;
        let base_cols = st.base.base.cols as i64;
        let num_elements = st.base.base.num_elements() as i64;
        let n = st.shape().len();
        let height_dim = st.shape()[n - 4] as i64;
        let width_dim = st.shape()[n - 3] as i64;
        let total_elems = height_dim * width_dim * num_elements;
        let slots = self.group_threads() as i64 * ept;
        // Round the pass count *up*: a tile smaller than one full group-pass (the
        // multi-wave FA 16×64 K/V block streamed by 512 threads) would otherwise
        // floor to zero passes and load nothing.
        let total_calls = (total_elems + slots - 1) / slots;
        // Over-subscribed last pass (more lane-loads than fragment-loads): clamp
        // the load index to the last valid fragment so the excess lanes redo it
        // (idempotent — same source, same swizzled slot) instead of writing past
        // the tile. A no-op when the tile divides the group evenly (matmul,
        // single-warp FA): `clamp` is false and the index passes through.
        let num_valid = total_elems / ept;
        let clamp = total_calls * slots != total_elems;

        let outer = self.ker.raw_range(total_calls, AxisType::Loop);
        let inner = self.ker.raw_range(ept, AxisType::Upcast);
        let laneid = self.laneid();

        let mut load_idx = iadd(&imul(&outer, self.group_threads() as i64), &laneid);
        if clamp {
            let cond = load_idx.try_cmplt(&cidx(num_valid)).expect("load_idx < num_valid");
            load_idx = UOp::try_where(cond, load_idx.clone(), cidx(num_valid - 1)).expect("clamp load_idx");
        }
        let row0 = idiv(&load_idx, memcpy_per_row);
        let col0 = iadd(&imod(&imul(&load_idx, ept), st_cols), &inner);
        let height = idiv(&row0, base_rows);
        let width = idiv(&col0, base_cols);
        let row = imod(&row0, base_rows);
        let col = imod(&col0, base_cols);
        let (srow, scol) = st.base.swizzle.swizzle_rc(row.clone(), col.clone(), st.base.base.cols, st.elem().base());

        let off = iadd(
            &src_i_base,
            &iadd(
                &iadd(&imul(&height, base_rows * row_stride), &imul(&width, base_cols)),
                &iadd(&imul(&row, row_stride), &col),
            ),
        );
        let mut load = load_off(src.uop(), off);
        if src.elem() != st.elem() {
            load = load.cast(st.elem().clone());
        }
        let dst_idx = st_index(&st, &[Idx::Uop(height), Idx::Uop(width), Idx::Uop(srow), Idx::Uop(scol)]);
        let stored = dst_idx.store(load).end(smallvec![outer, inner]);
        let ended = if barrier { stored.barrier(SmallVec::new()) } else { stored };
        self.finalize_st(st, ended)
    }

    /// The flat GLOBAL element offset of the `st`-sized tile at block `idxs` of
    /// `src` (row axis `axis`): the block index on `axis` counts `st.rows`-tall
    /// tiles, the last one `st.cols`-wide tiles.
    fn tile_base(st: &ST, src: &GL, idxs: &[Idx], axis: usize) -> Arc<UOp> {
        let idxs_t: Vec<Idx> = idxs
            .iter()
            .enumerate()
            .map(|(i, idx)| {
                let mut e = idx.clone();
                if i == axis {
                    e = idx_mul(&e, st.rows as i64);
                }
                if i == 3 {
                    e = idx_mul(&e, st.cols as i64);
                }
                e
            })
            .collect();
        flat_offset(src.shape(), &idxs_t)
    }

    /// Whether the collaborative fill of `st` from `src` can be `cp.async`
    /// 16-byte copies: a CUDA target, one lane's `elements_per_thread` run is
    /// exactly 16 bytes, no element cast, and the swizzle keeps 16-byte chunks
    /// contiguous ([`crate::swizzle::Swizzle::keeps_16b_chunks`]).
    pub fn cp_async_fill_applies(&self, st: &ST, src: &GL) -> bool {
        self.ker.caps.cuda().is_some()
            && st.base.base.elements_per_thread() * st.elem().bytes() == 16
            && src.elem() == st.elem()
            && st.base.swizzle.keeps_16b_chunks()
    }

    /// Asynchronous GLOBAL→LOCAL tile fill (sm_80+): every group thread issues one
    /// 16-byte `cp.async.cg` per pass (the same per-lane addressing as the scalar
    /// fill, so LDS is laid out identically), then commits them as ONE group.
    /// Returns the `commit` statement; the caller retires it with a
    /// `cp.async.wait_group` and a workgroup barrier before any lane reads the tile
    /// (`wait_group N; bar.sync; consume` — copies of other lanes are visible only
    /// after the barrier). `st` carries the ordering deps of the destination (thread
    /// the WAR barrier through `st.after([..])`).
    ///
    /// # Panics
    /// Panics unless [`Self::cp_async_fill_applies`], and unless the global rows
    /// are 16-byte aligned (`axis` row stride and the innermost extent multiples of
    /// the per-lane run — `D % 8 == 0` for bf16).
    pub fn cp_async_fill(&self, st: &ST, src: &GL, idxs: &[Idx], axis: usize) -> Arc<UOp> {
        assert!(self.cp_async_fill_applies(st, src), "cp.async fill: CUDA, 16-byte lane runs, no cast, chunk swizzle");
        let geom = self.lds_fill_geom(st);
        let row_stride: i64 = src.shape()[axis + 1..].iter().product::<usize>() as i64;
        let inner = *src.shape().last().expect("GL rank") as i64;
        assert_eq!(row_stride % geom.ept, 0, "cp.async fill: row stride {row_stride} not 16-byte aligned");
        assert_eq!(inner % geom.ept, 0, "cp.async fill: innermost extent {inner} not 16-byte aligned");
        let src_i_base = Self::tile_base(st, src, idxs, axis);

        let copies: SmallVec<[Arc<UOp>; 4]> = (0..geom.total_calls)
            .map(|pass| {
                let (height, width, row, col) = self.fill_lane_rc(&geom, &cidx(pass), &cidx(0));
                let (srow, scol) =
                    st.base.swizzle.swizzle_rc(row.clone(), col.clone(), st.base.base.cols, st.elem().base());
                let dst =
                    st_index(st, &[Idx::Uop(height.clone()), Idx::Uop(width.clone()), Idx::Uop(srow), Idx::Uop(scol)]);
                let off = iadd(
                    &src_i_base,
                    &iadd(
                        &iadd(&imul(&height, geom.base_rows * row_stride), &imul(&width, geom.base_cols)),
                        &iadd(&imul(&row, row_stride), &col),
                    ),
                );
                cp_async_16(&dst, &index_off(src.uop(), off))
            })
            .collect();
        cp_async_commit(copies)
    }

    /// Stackd GLOBAL→LOCAL fill: the [`Self::load_global_to_local`]
    /// counterpart that issues **128-bit** (`vec8` bf16) coalesced global loads
    /// (one `global_load_dwordx4`/lane) and commits each into the XOR-swizzled
    /// LDS as `vec8/sw` contiguous `vec_sw` stores. The swizzle's XOR delta is
    /// always a multiple of 8 bytes (`st.cuh:96` `<<3`), so a `sw = 8/itemsize`
    /// element group is never re-ordered (the `vec4` halves stay contiguous);
    /// a single `vec8` LDS store would split on the odd deltas, so we keep the
    /// wide *global* load but narrow the swizzled *LDS* store. Ends in a
    /// workgroup barrier (the matmul fill). bf16-only.
    ///
    /// # Panics
    /// Panics unless the source element itemsize is 2 bytes (bf16), the `src` and
    /// the destination ST element types match (no cast on this path), and the
    /// swizzle period, base cols, ST cols, and source row-stride are all aligned
    /// to the 128-bit vector width.
    pub fn fill_local_vec(&self, dst: ST, src: GL, idxs: &[Idx], axis: usize) -> ST {
        self.load_global_to_local_vec(dst, &src, idxs, axis, true)
    }

    fn load_global_to_local_vec(&self, st: ST, src: &GL, idxs: &[Idx], axis: usize, barrier: bool) -> ST {
        // sm_80+: the 128-bit lane copy is a `cp.async`, retired (`wait_group 0`)
        // under the same trailing barrier the register path ends in.
        if barrier && self.cp_async_fill_applies(&st, src) {
            let commit = self.cp_async_fill(&st, src, idxs, axis);
            let landed = cp_async_wait(0, smallvec![commit]).barrier(SmallVec::new());
            return self.finalize_st(st, landed);
        }
        let itemsize = st.elem().base().bytes() as i64;
        assert_eq!(itemsize, 2, "vec fill: bf16-only (128-bit = vec8)");
        assert_eq!(src.elem(), st.elem(), "vec fill: cast unsupported (use the scalar fill)");
        let vw: i64 = 16 / itemsize; // 8 bf16 — the 128-bit global load width
        let sw: i64 = 8 / itemsize; // 4 bf16 — the swizzle-order-safe LDS store width

        let base_rows = st.base.base.rows as i64;
        let base_cols = st.base.base.cols as i64;
        let st_cols = st.cols as i64;
        // Alignment invariants: the swizzle period and the
        // tile/fragment widths must admit `vw`-aligned 16-byte groups.
        if let Some(period) = st.base.swizzle.period_bytes(st.base.base.cols, itemsize) {
            assert_eq!(period % 16, 0, "vec fill: swizzle period {period}B not 16B-aligned");
        }
        assert_eq!(base_cols % vw, 0, "vec fill: base cols {base_cols} not a multiple of vec width {vw}");
        assert_eq!(st_cols % vw, 0, "vec fill: st cols {st_cols} not a multiple of vec width {vw}");

        let row_stride: i64 = src.shape()[axis + 1..].iter().product::<usize>() as i64;
        assert_eq!(row_stride % vw, 0, "vec fill: row stride {row_stride} not {vw}-aligned (need N % 8 == 0)");

        let src_i_base = Self::tile_base(&st, src, idxs, axis);

        let num_elements = st.base.base.num_elements() as i64;
        let n = st.shape().len();
        let total_elems = st.shape()[n - 4] as i64 * st.shape()[n - 3] as i64 * num_elements;
        let memcpy_per_row = st_cols / vw;
        let slots = self.group_threads() as i64 * vw;
        let total_calls = (total_elems + slots - 1) / slots;
        let num_valid = total_elems / vw;
        let clamp = total_calls * slots != total_elems;

        let outer = self.ker.raw_range(total_calls, AxisType::Loop);
        let mut load_idx = iadd(&imul(&outer, self.group_threads() as i64), &self.laneid());
        if clamp {
            let cond = load_idx.try_cmplt(&cidx(num_valid)).expect("load_idx < num_valid");
            load_idx = UOp::try_where(cond, load_idx.clone(), cidx(num_valid - 1)).expect("clamp load_idx");
        }
        // The thread's `vw`-wide run: row `row0`, columns `[col0, col0+vw)` (a
        // `vw`-aligned slice within one base fragment, since `vw | base_cols`).
        let row0 = idiv(&load_idx, memcpy_per_row);
        let col0 = imod(&imul(&load_idx, vw), st_cols);
        let height = idiv(&row0, base_rows);
        let row = imod(&row0, base_rows);
        let width = idiv(&col0, base_cols);

        // One shaped scalar-dtype load of the contiguous `vw`-run. The compiler
        // may coalesce these logical lanes without widening the storage dtype.
        let off = iadd(&src_i_base, &iadd(&imul(&row0, row_stride), &col0));
        let src_offsets =
            UOp::stack((0..vw).map(|lane| if lane == 0 { off.clone() } else { iadd(&off, &cidx(lane)) }).collect());
        let loaded = UOp::load()
            .index(
                UOp::index()
                    .buffer(src.uop().clone())
                    .indices(vec![src_offsets])
                    .call()
                    .expect("vec fill source INDEX"),
            )
            .call();

        // Commit as `vw/sw` swizzle-safe `vec_sw` LDS stores (delta is constant
        // across the fragment row, so each `sw`-group maps contiguously).
        let stores: Vec<Arc<UOp>> = (0..vw / sw)
            .map(|j| {
                let col = imod(&iadd(&col0, &cidx(j * sw)), base_cols);
                let (srow, scol) = st.base.swizzle.swizzle_rc(row.clone(), col, st.base.base.cols, st.elem().base());
                let didx = [Idx::Uop(height.clone()), Idx::Uop(width.clone()), Idx::Uop(srow), Idx::Uop(scol)];
                let dst = st_index(&st, &didx);
                let Op::Index(ops::Index { buffer, indices }) = dst.op() else {
                    unreachable!("st_index returns INDEX")
                };
                let dst_base = indices[0].clone();
                let dst_offsets = UOp::stack(
                    (0..sw)
                        .map(|lane| if lane == 0 { dst_base.clone() } else { iadd(&dst_base, &cidx(lane)) })
                        .collect(),
                );
                let dst = UOp::index()
                    .buffer(buffer.clone())
                    .indices(vec![dst_offsets])
                    .call()
                    .expect("vec fill destination INDEX");
                let val = UOp::stack(
                    (j * sw..j * sw + sw)
                        .map(|lane| {
                            UOp::index()
                                .buffer(loaded.clone())
                                .indices(vec![cidx(lane)])
                                .call()
                                .expect("vec fill loaded lane INDEX")
                        })
                        .collect(),
                );
                dst.store(val)
            })
            .collect();
        let grouped = if stores.len() == 1 { stores.into_iter().next().unwrap() } else { UOp::group(stores) };
        let stored = grouped.end(smallvec![outer]);
        let ended = if barrier { stored.barrier(SmallVec::new()) } else { stored };
        self.finalize_st(st, ended)
    }

    /// LOCAL→REG fragment gather: each lane reads its WMMA fragment lanes from
    /// the (swizzled) LDS tile.
    pub(super) fn load_local_to_reg(&self, rt: RT<'k>, st: &ST, dst_idxs: &[Idx], idxs: &[Idx]) -> RT<'k> {
        let laneid = self.ker.laneid();
        let ept = rt.base.base.elements_per_thread() as i64;
        let n = rt.shape().len();
        let (rt_h, rt_w) = (rt.shape()[n - 3] as i64, rt.shape()[n - 2] as i64);
        // SI-1 off-by-one guard: the wave's RT sub-tile must fit inside the ST.
        let sn = st.shape().len();
        let (st_h, st_w) = (st.shape()[sn - 4] as i64, st.shape()[sn - 3] as i64);
        assert!(rt_h <= st_h && rt_w <= st_w, "load LOCAL→REG: RT {rt_h}×{rt_w} exceeds ST {st_h}×{st_w}");
        let transpose = rt.layout != st.layout;
        if let Some(plan) = self.ldmatrix_plan(&rt, st, transpose) {
            return self.ldmatrix_local_to_reg(rt, st, dst_idxs, idxs, plan);
        }
        let height = self.ker.raw_range(rt_h, AxisType::Loop);
        let width = self.ker.raw_range(rt_w, AxisType::Loop);
        let inner = self.ker.raw_range(ept, AxisType::Loop);

        let (row, col) = rt.lane_rc(transpose, &laneid, &inner);
        let (srow, scol) = st.base.swizzle.swizzle_rc(row, col, st.base.base.cols, st.elem().base());

        // Wave sub-tile fragment offset (SI-1): the caller passes the wave's
        // `(row_block, col_block)` via `idxs` (already including warp_row/col);
        // empty ⇒ no offset (single-warp).
        let h_idx = wave_offset(idxs.first(), rt_h, &height);
        let w_idx = wave_offset(idxs.get(1), rt_w, &width);
        let src_idx = [h_idx, w_idx, Idx::Uop(srow), Idx::Uop(scol)];
        let mut load = st_load(st, &src_idx);
        if st.elem() != rt.elem() {
            load = load.cast(rt.elem().clone());
        }
        let mut didx: Vec<Idx> = dst_idxs.to_vec();
        didx.extend([Idx::from(&height), Idx::from(&width), Idx::from(&inner)]);
        let ended = flat_index(rt.uop(), rt.shape(), &didx).store(load).end(smallvec![height, width, inner]);
        self.finalize_reg(rt, ended)
    }

    /// The `ldmatrix.x4` plan for the LOCAL→REG hop, when it applies: a CUDA
    /// target, a 16-bit fragment with no cast, the 16×16 / 8-per-lane base, a lane
    /// map [`LaneMap::ldmatrix_x4`](crate::layout::LaneMap::ldmatrix_x4) covers,
    /// and a swizzle that keeps 16-byte row chunks contiguous.
    fn ldmatrix_plan(&self, rt: &RT<'k>, st: &ST, transpose: bool) -> Option<LdmatrixX4> {
        let base = &rt.base.base;
        (self.ker.caps.cuda().is_some()
            && rt.elem().bytes() == 2
            && st.elem() == rt.elem()
            && (base.rows, base.cols, base.elements_per_thread()) == (16, 16, 8)
            && st.base.base == *base
            && st.base.swizzle.keeps_16b_chunks())
        .then(|| rt.base.map.ldmatrix_x4(transpose))
        .flatten()
    }

    /// LOCAL→REG fragment gather as one warp-collective `ldmatrix.x4[.trans]` per
    /// 16×16 fragment: lane `L` supplies the (swizzled) address of row `L % 16`,
    /// columns `8·(L/16)..+8`, and the four returned 32-bit words are scattered onto
    /// the fragment's register pairs per the plan (every register index constant,
    /// so the fragment stays in registers). Replaces the eight scalar `ld.shared.b16`
    /// per fragment of the generic gather.
    fn ldmatrix_local_to_reg(&self, rt: RT<'k>, st: &ST, dst_idxs: &[Idx], idxs: &[Idx], plan: LdmatrixX4) -> RT<'k> {
        let laneid = self.ker.laneid();
        let n = rt.shape().len();
        let (rt_h, rt_w) = (rt.shape()[n - 3] as i64, rt.shape()[n - 2] as i64);
        let row = imod(&laneid, 16);
        let col = imul(&idiv(&laneid, 16), 8);
        let (srow, scol) = st.base.swizzle.swizzle_rc(row, col, st.base.base.cols, st.elem().base());
        let pair = rt.elem().vec(2).expect("16-bit element pair");
        let at = |block: Option<&Idx>, frags: i64, i: i64| match block {
            None => Idx::Const(i),
            Some(b) => Idx::Uop(iadd(&imul(&b.to_uop(), frags), &cidx(i))),
        };
        let mut stores = Vec::with_capacity((rt_h * rt_w * 8) as usize);
        for h in 0..rt_h {
            for w in 0..rt_w {
                let src_idx = [
                    at(idxs.first(), rt_h, h),
                    at(idxs.get(1), rt_w, w),
                    Idx::Uop(srow.clone()),
                    Idx::Uop(scol.clone()),
                ];
                let words = ldmatrix(&st_index(st, &src_idx), 4, plan.trans, pair.clone());
                for (p, &m) in plan.words.iter().enumerate() {
                    for e in 0..2 {
                        let mut didx = dst_idxs.to_vec();
                        didx.extend([Idx::Const(h), Idx::Const(w), Idx::Const(2 * p as i64 + e as i64)]);
                        stores.push(flat_index(rt.uop(), rt.shape(), &didx).store(words[m].index_axes(vec![e])));
                    }
                }
            }
        }
        let ended = if stores.len() == 1 { stores.into_iter().next().unwrap() } else { UOp::group(stores) };
        self.finalize_reg(rt, ended)
    }

    /// The boundary gate for a GLOBAL↔REG hop: `global_row < shape[axis] &
    /// global_col < shape[last]`, restricted to the axes that are actually ragged
    /// (the extent is not a multiple of the per-block tile span — known at build
    /// time, so an aligned axis adds no gate). `srow`/`scol` are the in-tile
    /// coordinates; the block offset from `idxs` is folded back in to recover the
    /// global position. `None` when both axes divide evenly.
    #[allow(clippy::too_many_arguments)]
    fn boundary_gate(
        &self,
        shape: &[usize],
        idxs: &[Idx],
        axis: usize,
        row_tile: i64,
        col_tile: i64,
        srow: &Arc<UOp>,
        scol: &Arc<UOp>,
    ) -> Option<Arc<UOp>> {
        let mut gate: Option<Arc<UOp>> = None;
        let bound_row = shape[axis] as i64;
        if bound_row % row_tile != 0 {
            let blk = idxs.get(axis).map(|i| i.to_uop()).unwrap_or_else(|| cidx(0));
            let g = iadd(&imul(&blk, row_tile), srow).try_cmplt(&cidx(bound_row)).expect("boundary row gate");
            gate = Some(g);
        }
        let bound_col = shape[shape.len() - 1] as i64;
        if bound_col % col_tile != 0 {
            let blk = idxs.get(3).map(|i| i.to_uop()).unwrap_or_else(|| cidx(0));
            let g = iadd(&imul(&blk, col_tile), scol).try_cmplt(&cidx(bound_col)).expect("boundary col gate");
            gate = Some(match gate {
                Some(r) => r.try_and_op(&g).expect("boundary gate and"),
                None => g,
            });
        }
        gate
    }

    /// GLOBAL→REG fragment gather: each lane reads its register fragment
    /// straight from global memory (the FA Q-tile load). The mirror of
    /// [`Self::store_reg_to_global`]. `masked` gates a tile straddling a ragged
    /// edge (see [`Self::boundary_gate`]).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn load_global_to_reg(
        &self,
        rt: RT<'k>,
        src: &GL,
        dst_idxs: &[Idx],
        idxs: &[Idx],
        axis: usize,
        masked: bool,
    ) -> RT<'k> {
        let row_stride: i64 = src.shape()[axis + 1..].iter().product::<usize>() as i64;
        let base_rows = rt.base.base.rows as i64;
        let base_cols = rt.base.base.cols as i64;
        let ept = rt.base.base.elements_per_thread() as i64;
        let n = rt.shape().len();
        let s3 = rt.shape()[n - 3] as i64;
        let s2 = rt.shape()[n - 2] as i64;

        let idxs_t: Vec<Idx> = idxs
            .iter()
            .enumerate()
            .map(|(i, idx)| {
                let mut e = idx.clone();
                if i == axis {
                    e = idx_mul(&e, s3 * base_rows);
                }
                if i == 3 {
                    e = idx_mul(&e, s2 * base_cols);
                }
                e
            })
            .collect();
        let src_i_base = flat_offset(src.shape(), &idxs_t);

        let laneid = self.ker.laneid();
        let height = self.ker.raw_range(s3, AxisType::Loop);
        let width = self.ker.raw_range(s2, AxisType::Loop);
        let inner = self.ker.raw_range(ept, AxisType::Loop);

        let base_row = imul(&height, base_rows);
        let base_col = imul(&width, base_cols);
        let (row, col) = rt.lane_rc(rt.layout == TileLayout::Col, &laneid, &inner);
        let srow = iadd(&base_row, &row);
        let scol = iadd(&base_col, &col);
        let off = iadd(&src_i_base, &iadd(&imul(&srow, row_stride), &scol));

        let gate = masked
            .then(|| self.boundary_gate(src.shape(), idxs, axis, s3 * base_rows, s2 * base_cols, &srow, &scol))
            .flatten();
        let mut load = match gate {
            Some(g) => {
                let zero = if src.elem().is_float() { ConstValue::Float(0.0) } else { ConstValue::Int(0) };
                load_off_gated(src.uop(), off, g, UOp::const_(src.elem().clone(), zero))
            }
            None => load_off(src.uop(), off),
        };
        if src.elem() != rt.elem() {
            load = load.cast(rt.elem().clone());
        }
        let mut didx: Vec<Idx> = dst_idxs.to_vec();
        didx.extend([Idx::from(&height), Idx::from(&width), Idx::from(&inner)]);
        let ended = flat_index(rt.uop(), rt.shape(), &didx).store(load).end(smallvec![height, width, inner]);
        self.finalize_reg(rt, ended)
    }

    /// REG→LOCAL fragment scatter: each lane writes its register fragment into
    /// the (swizzled) LDS tile (the layout-transpose hop before write-back).
    pub(super) fn store_reg_to_local(&self, st: ST, rt: &RT<'k>, idxs: &[Idx], src_idxs: &[Idx]) -> ST {
        let laneid = self.ker.laneid();
        let ept = rt.base.base.elements_per_thread() as i64;
        let n = rt.shape().len();
        let (rt_h, rt_w) = (rt.shape()[n - 3] as i64, rt.shape()[n - 2] as i64);
        let height = self.ker.raw_range(rt_h, AxisType::Loop);
        let width = self.ker.raw_range(rt_w, AxisType::Loop);
        let inner = self.ker.raw_range(ept, AxisType::Loop);

        let (row, col) = rt.lane_rc(rt.layout != st.layout, &laneid, &inner);
        let (srow, scol) = st.base.swizzle.swizzle_rc(row, col, st.base.base.cols, st.elem().base());

        let mut sidx: Vec<Idx> = src_idxs.to_vec();
        sidx.extend([Idx::from(&height), Idx::from(&width), Idx::from(&inner)]);
        let mut load = load_at(rt.uop(), rt.shape(), &sidx);
        if rt.elem() != st.elem() {
            load = load.cast(st.elem().clone());
        }
        // Wave sub-tile fragment offset (SI-1), symmetric with `load_local_to_reg`.
        let h_idx = wave_offset(idxs.first(), rt_h, &height);
        let w_idx = wave_offset(idxs.get(1), rt_w, &width);
        let didx = [h_idx, w_idx, Idx::Uop(srow), Idx::Uop(scol)];
        let ended = st_index(&st, &didx).store(load).end(smallvec![height, width, inner]);
        self.finalize_st(st, ended)
    }

    /// REG→GLOBAL write-back: each lane writes its register fragment to the
    /// correct global position.
    pub(super) fn store_reg_to_global(
        &self,
        dst: GL,
        rt: &RT<'k>,
        idxs: &[Idx],
        src_idxs: &[Idx],
        axis: usize,
        masked: bool,
    ) -> GL {
        let row_stride: i64 = dst.shape()[axis + 1..].iter().product::<usize>() as i64;
        let base_rows = rt.base.base.rows as i64;
        let base_cols = rt.base.base.cols as i64;
        let ept = rt.base.base.elements_per_thread() as i64;
        let n = rt.shape().len();
        let s3 = rt.shape()[n - 3] as i64;
        let s2 = rt.shape()[n - 2] as i64;

        let idxs_t: Vec<Idx> = idxs
            .iter()
            .enumerate()
            .map(|(i, idx)| {
                let mut e = idx.clone();
                if i == axis {
                    e = idx_mul(&e, s3 * base_rows);
                }
                if i == 3 {
                    e = idx_mul(&e, s2 * base_cols);
                }
                e
            })
            .collect();
        let dst_i_base = flat_offset(dst.shape(), &idxs_t);

        let laneid = self.ker.laneid();
        let height = self.ker.raw_range(s3, AxisType::Loop);
        let width = self.ker.raw_range(s2, AxisType::Loop);
        let inner = self.ker.raw_range(ept, AxisType::Loop);

        let base_row = imul(&height, base_rows);
        let base_col = imul(&width, base_cols);
        let (row, col) = rt.lane_rc(rt.layout == TileLayout::Col, &laneid, &inner);
        let srow = iadd(&base_row, &row);
        let scol = iadd(&base_col, &col);
        let off = iadd(&dst_i_base, &iadd(&imul(&srow, row_stride), &scol));

        let mut sidx: Vec<Idx> = src_idxs.to_vec();
        sidx.extend([Idx::from(&height), Idx::from(&width), Idx::from(&inner)]);
        let mut load = load_at(rt.uop(), rt.shape(), &sidx);
        if rt.elem() != dst.elem() {
            load = load.cast(dst.elem().clone());
        }
        let gate = masked
            .then(|| self.boundary_gate(dst.shape(), idxs, axis, s3 * base_rows, s2 * base_cols, &srow, &scol))
            .flatten();
        let target = match gate {
            Some(g) => index_off_gated(dst.uop(), off, g),
            None => index_off(dst.uop(), off),
        };
        let ended = target.store(load).end(smallvec![height, width, inner]);
        self.finalize_gl(dst, ended)
    }
}

/// ST flat INDEX honoring the optional double-buffer parity [`ST::base_offset`].
/// Identical to [`crate::index::flat_index`] for an ordinary (`base_offset:None`)
/// tile; adds the parity offset for a [`Kernel::st_db`](crate::Kernel) half-view.
fn st_index(st: &ST, idxs: &[Idx]) -> Arc<UOp> {
    let mut off = flat_offset(st.shape(), idxs);
    if let Some(bo) = st.base_offset() {
        off = off.try_add(bo).expect("st_index: parity base offset add");
    }
    index_off(st.uop(), off)
}
/// ST flat LOAD honoring [`ST::base_offset`] — the [`crate::index::load_at`] analog.
fn st_load(st: &ST, idxs: &[Idx]) -> Arc<UOp> {
    let idx = st_index(st, idxs);
    UOp::load().index(idx).call()
}
