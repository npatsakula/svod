//! Buffer-bound tile wrappers — the eager tiles that bind a concrete UOp buffer
//! plus a *logical* shape to a [`Kernel`], mirroring tinygrad `tiles.py`'s
//! `GL`/`ST`/`RT`/`RV`. Each wrapper carries its backing `Arc<UOp>` (a flat 1-D
//! pointer), the multi-dim logical shape used for [`crate::index::flat_index`]
//! addressing, its layout/base-shape descriptor, and a borrow of the building
//! [`Kernel`].
//!
//! Unlike tinygrad's autowrapping proxy, these are plain structs (no `Deref` to
//! `Arc<UOp>`); a [`rewrap`](GL::rewrap) swaps the backing buffer after a store
//! (the tinykittens `ruop`) so later reads depend on the store via its
//! `After([END(STORE)])` node.

use std::sync::Arc;

use smallvec::{SmallVec, smallvec};
use svod_dtype::DType;
use svod_ir::UOp;

use crate::Kernel;
use crate::tiles::{RTBaseShape, STBaseShape, TileLayout, VecLayout};

/// A register-backed tile ([`RT`] or [`RV`]) that the elementwise [`map`] /
/// math / reduce ops manipulate uniformly: a flat buffer, a logical shape, an
/// element dtype, and a `rewrap` to swap the backing buffer after a store.
pub trait RegTile<'k>: Clone {
    fn uop(&self) -> &Arc<UOp>;
    fn shape(&self) -> &[usize];
    fn elem(&self) -> &DType;
    fn layout(&self) -> TileLayout;
    fn rewrap(&self, new_uop: Arc<UOp>) -> Self;

    /// Rewrap with extra ordering dependencies (tinygrad `tile.after(dep)`),
    /// e.g. a write-after-read edge that forces this tile's next read to observe
    /// `deps` first. Accepts a single dep, an array, a 2-/3-tuple of mixed deps,
    /// or a raw `SmallVec` (see [`AfterDeps`]) — so a kernel writes `.after(&tile)`
    /// or `.after((range, &vec))` instead of `.after(smallvec![tile.uop().clone()])`.
    fn after(&self, deps: impl AfterDeps) -> Self {
        self.rewrap(self.uop().after(deps.into_afters()))
    }
}

/// A single ordering dependency for [`RegTile::after`] — a raw `Arc<UOp>` (a range
/// counter, a barrier, an asm node) or a register tile (its backing buffer). The
/// `&RT`/`&RV` impls let callers pass a tile directly instead of spelling out
/// `tile.uop().clone()`.
pub trait AfterDep {
    fn into_after(self) -> Arc<UOp>;
}

impl AfterDep for Arc<UOp> {
    fn into_after(self) -> Arc<UOp> {
        self
    }
}
impl AfterDep for &Arc<UOp> {
    fn into_after(self) -> Arc<UOp> {
        self.clone()
    }
}
impl<'k> AfterDep for &RT<'k> {
    fn into_after(self) -> Arc<UOp> {
        self.uop().clone()
    }
}
impl<'k> AfterDep for &RV<'k> {
    fn into_after(self) -> Arc<UOp> {
        self.uop().clone()
    }
}

/// One or more ordering dependencies for [`RegTile::after`]: a single [`AfterDep`],
/// an array `[d; N]`, a heterogeneous 2-/3-tuple (e.g. a loop counter plus a carried
/// vector), or a raw `SmallVec` (the legacy form, kept so existing call sites keep
/// compiling). Every form lowers to the same `SmallVec<[Arc<UOp>; 4]>`.
pub trait AfterDeps {
    fn into_afters(self) -> SmallVec<[Arc<UOp>; 4]>;
}

impl AfterDeps for Arc<UOp> {
    fn into_afters(self) -> SmallVec<[Arc<UOp>; 4]> {
        smallvec![self]
    }
}
impl AfterDeps for &Arc<UOp> {
    fn into_afters(self) -> SmallVec<[Arc<UOp>; 4]> {
        smallvec![self.clone()]
    }
}
impl<'k> AfterDeps for &RT<'k> {
    fn into_afters(self) -> SmallVec<[Arc<UOp>; 4]> {
        smallvec![self.uop().clone()]
    }
}
impl<'k> AfterDeps for &RV<'k> {
    fn into_afters(self) -> SmallVec<[Arc<UOp>; 4]> {
        smallvec![self.uop().clone()]
    }
}
impl<D: AfterDep, const N: usize> AfterDeps for [D; N] {
    fn into_afters(self) -> SmallVec<[Arc<UOp>; 4]> {
        self.into_iter().map(AfterDep::into_after).collect()
    }
}
impl<A: AfterDep, B: AfterDep> AfterDeps for (A, B) {
    fn into_afters(self) -> SmallVec<[Arc<UOp>; 4]> {
        smallvec![self.0.into_after(), self.1.into_after()]
    }
}
impl<A: AfterDep, B: AfterDep, C: AfterDep> AfterDeps for (A, B, C) {
    fn into_afters(self) -> SmallVec<[Arc<UOp>; 4]> {
        smallvec![self.0.into_after(), self.1.into_after(), self.2.into_after()]
    }
}
impl AfterDeps for SmallVec<[Arc<UOp>; 4]> {
    fn into_afters(self) -> SmallVec<[Arc<UOp>; 4]> {
        self
    }
}

impl<'k> RegTile<'k> for RT<'k> {
    fn uop(&self) -> &Arc<UOp> {
        &self.buf
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn elem(&self) -> &DType {
        &self.elem
    }
    fn layout(&self) -> TileLayout {
        self.layout
    }
    fn rewrap(&self, new_uop: Arc<UOp>) -> Self {
        RT::rewrap(self, new_uop)
    }
}

impl<'k> RegTile<'k> for RV<'k> {
    fn uop(&self) -> &Arc<UOp> {
        &self.buf
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn elem(&self) -> &DType {
        &self.elem
    }
    /// An RV is logically a column of values; treat it as `Row` for the generic
    /// ops (it never broadcasts another vector into itself).
    fn layout(&self) -> TileLayout {
        TileLayout::Row
    }
    fn rewrap(&self, new_uop: Arc<UOp>) -> Self {
        RV::rewrap(self, new_uop)
    }
}

/// Element (scalar) dtype backing a pointer tile buffer.
fn elem_of(buf: &Arc<UOp>) -> DType {
    match buf.dtype() {
        DType::Ptr { base, .. } => (*base).clone(),
        dt => dt,
    }
}

/// Macro for the shared tile accessors (`uop`/`shape`/`elem`).
macro_rules! tile_accessors {
    () => {
        /// The backing flat 1-D pointer buffer (or its `After` re-wrap).
        pub fn uop(&self) -> &Arc<UOp> {
            &self.buf
        }
        /// The multi-dim logical shape used for flat addressing.
        pub fn shape(&self) -> &[usize] {
            &self.shape
        }
        /// The element (scalar) dtype.
        pub fn elem(&self) -> &DType {
            &self.elem
        }
    };
}

/// `tile_accessors!` + the `ker: &'k Kernel` accessor for tiles that carry their
/// building kernel (the operator-overload-backed RT/RV).
macro_rules! tile_accessors_ker {
    () => {
        tile_accessors!();

        /// The building kernel.
        pub fn ker(&self) -> &'k Kernel {
            self.ker
        }
    };
}

/// A global-memory tile: the next bound buffer placeholder (`Param`) plus its
/// logical shape (e.g. `[1, 1, N, N]`). Accessed flat in load/store.
#[derive(Clone)]
pub struct GL {
    buf: Arc<UOp>,
    shape: Vec<usize>,
    elem: DType,
}

impl GL {
    tile_accessors!();
    /// Swap the backing buffer (after a store) keeping shape/dtype.
    pub fn rewrap(&self, new_uop: Arc<UOp>) -> Self {
        GL { buf: new_uop, shape: self.shape.clone(), elem: self.elem.clone() }
    }
}

/// A shared-memory (LDS) tile: a grid of [`STBaseShape`] fragments. Logical
/// shape is `[.., height, width, base.rows, base.cols]`.
#[derive(Clone)]
pub struct ST {
    buf: Arc<UOp>,
    shape: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
    pub layout: TileLayout,
    pub base: STBaseShape,
    elem: DType,
    /// Optional additive flat-element offset into the backing buffer — the
    /// software double-buffer parity select (`tile % 2 * half_elems`). `None`
    /// for an ordinary single-buffered tile (the common case). When `Some`, every
    /// LDS access adds it to the computed flat offset, selecting one half of a
    /// `st_db` (2×-size) buffer at runtime.
    base_offset: Option<Arc<UOp>>,
}

impl ST {
    tile_accessors!();
    pub fn rewrap(&self, new_uop: Arc<UOp>) -> Self {
        ST {
            buf: new_uop,
            shape: self.shape.clone(),
            rows: self.rows,
            cols: self.cols,
            layout: self.layout,
            base: self.base,
            elem: self.elem.clone(),
            base_offset: self.base_offset.clone(),
        }
    }

    /// Rewrap with extra ordering dependencies (the [`RegTile::after`] analog): the
    /// tile's next LDS access observes `deps` first (a barrier, an async-copy commit).
    pub fn after(&self, deps: impl AfterDeps) -> ST {
        self.rewrap(self.buf.after(deps.into_afters()))
    }

    /// The per-half flat element count (the full single-half tile size); a
    /// [`Kernel::st_db`] buffer holds two of these. Used to form parity offsets.
    pub fn half_elems(&self) -> usize {
        self.shape.iter().product()
    }
    /// This tile viewing one half of a double buffer: every LDS access adds
    /// `off` (an `Index`-typed element offset, typically `parity * half_elems()`)
    /// to its flat address. Clones the wrapper (shares the backing buffer).
    pub fn with_base_offset(&self, off: Arc<UOp>) -> ST {
        let mut t = self.rewrap(self.buf.clone());
        t.base_offset = Some(off);
        t
    }
    /// The parity base offset, if this is a double-buffer half view.
    pub fn base_offset(&self) -> Option<&Arc<UOp>> {
        self.base_offset.as_ref()
    }
}

/// A register (per-lane) tile: a grid of [`RTBaseShape`] fragments. Logical
/// shape is `[height, width, base.elements_per_thread]`.
#[derive(Clone)]
pub struct RT<'k> {
    buf: Arc<UOp>,
    shape: Vec<usize>,
    pub layout: TileLayout,
    pub base: RTBaseShape,
    elem: DType,
    ker: &'k Kernel,
}

impl<'k> RT<'k> {
    tile_accessors_ker!();
    pub fn rewrap(&self, new_uop: Arc<UOp>) -> Self {
        RT {
            buf: new_uop,
            shape: self.shape.clone(),
            layout: self.layout,
            base: self.base,
            elem: self.elem.clone(),
            ker: self.ker,
        }
    }
}

/// A register vector: logical shape `[outer_dim, inner_dim]` — `[tiles, slots]`
/// for the ortho layout, `slots` the fragment map's per-lane kept values
/// ([`crate::layout::LaneMap::slots`]: 1 on AMD, 2 on `mma.sync`).
#[derive(Clone)]
pub struct RV<'k> {
    buf: Arc<UOp>,
    shape: Vec<usize>,
    pub length: usize,
    pub layout: VecLayout,
    pub base: RTBaseShape,
    elem: DType,
    ker: &'k Kernel,
}

impl<'k> RV<'k> {
    tile_accessors_ker!();
    pub fn rewrap(&self, new_uop: Arc<UOp>) -> Self {
        RV {
            buf: new_uop,
            shape: self.shape.clone(),
            length: self.length,
            layout: self.layout,
            base: self.base,
            elem: self.elem.clone(),
            ker: self.ker,
        }
    }
}

impl Kernel {
    /// Bind the next declared buffer as a [`GL`] tile (tinygrad `ker.gl`). The
    /// element dtype is taken from the bound buffer; `dtype` is the author's
    /// declared dtype. The buffer governs, but a debug build asserts the two have
    /// the same byte width — addressing depends only on the element width, so a
    /// same-width mismatch (e.g. bf16/f16) is benign while a different-width one
    /// (e.g. f32 vs bf16) is a real declaration bug worth catching early.
    pub fn gl(&self, shape: &[usize], dtype: DType) -> GL {
        let buf = self.next_global();
        let elem = elem_of(&buf);
        debug_assert_eq!(
            elem.bytes(),
            dtype.bytes(),
            "gl: declared dtype {dtype:?} ({}B) and bound buffer element {elem:?} ({}B) differ in width",
            dtype.bytes(),
            elem.bytes()
        );
        GL { buf, shape: shape.to_vec(), elem }
    }

    /// Allocate a shared-memory [`ST`] tile (tinygrad `ker.st`). `dims` is the
    /// `(rows, cols)` block size, tiled into a `height×width` grid of `base`
    /// fragments; the LDS buffer is `height×width×base` flat.
    ///
    /// # Panics
    /// Panics unless `rows` is a multiple of `base.base.rows`, `cols` a multiple
    /// of `base.base.cols`, and `cols` a multiple of the per-thread element count.
    pub fn st(&self, dims: (usize, usize), dtype: DType, layout: TileLayout, base: STBaseShape) -> ST {
        let (rows, cols) = dims;
        assert_eq!(rows % base.base.rows, 0, "ST rows {rows} not a multiple of base {}", base.base.rows);
        assert_eq!(cols % base.base.cols, 0, "ST cols {cols} not a multiple of base {}", base.base.cols);
        assert_eq!(cols % base.base.elements_per_thread(), 0, "ST cols {cols} not a multiple of elements_per_thread");
        let height = rows / base.base.rows;
        let width = cols / base.base.cols;
        let shape = vec![height, width, base.base.rows, base.base.cols];
        let flat = shape.iter().product();
        let buf = self.alloc_local(flat, dtype.clone());
        ST { buf, shape, rows, cols, layout, base, elem: dtype, base_offset: None }
    }

    /// Allocate a **double-buffered** shared-memory [`ST`] tile: identical logical
    /// shape and addressing to [`Kernel::st`], but the backing LDS buffer is
    /// **2× the flat size** so the two halves can hold consecutive K-tiles for a
    /// software-pipelined K-loop. The returned tile has `base_offset = None`
    /// (it addresses half 0); the caller forms the two half-views with
    /// [`ST::with_base_offset`]`(parity * `[`ST::half_elems`]`())`.
    ///
    /// # Panics
    /// Panics unless `rows` is a multiple of `base.base.rows`, `cols` a multiple
    /// of `base.base.cols`, and `cols` a multiple of the per-thread element count.
    pub fn st_db(&self, dims: (usize, usize), dtype: DType, layout: TileLayout, base: STBaseShape) -> ST {
        let (rows, cols) = dims;
        assert_eq!(rows % base.base.rows, 0, "ST rows {rows} not a multiple of base {}", base.base.rows);
        assert_eq!(cols % base.base.cols, 0, "ST cols {cols} not a multiple of base {}", base.base.cols);
        assert_eq!(cols % base.base.elements_per_thread(), 0, "ST cols {cols} not a multiple of elements_per_thread");
        let height = rows / base.base.rows;
        let width = cols / base.base.cols;
        let shape = vec![height, width, base.base.rows, base.base.cols];
        let half: usize = shape.iter().product();
        let buf = self.alloc_local(2 * half, dtype.clone());
        ST { buf, shape, rows, cols, layout, base, elem: dtype, base_offset: None }
    }

    /// Allocate a register [`RT`] tile (tinygrad `ker.rt`). `dims` is the
    /// `(rows, cols)` block size, tiled into a `height×width` grid of `base`
    /// fragments; the per-lane buffer is `height×width×elements_per_thread` flat.
    ///
    /// # Panics
    /// Panics unless `rows` is a multiple of `base.base.rows` and `cols` a
    /// multiple of `base.base.cols`.
    pub fn rt(&self, dims: (usize, usize), dtype: DType, layout: TileLayout, base: RTBaseShape) -> RT<'_> {
        let (rows, cols) = dims;
        assert_eq!(rows % base.base.rows, 0, "RT rows {rows} not a multiple of base {}", base.base.rows);
        assert_eq!(cols % base.base.cols, 0, "RT cols {cols} not a multiple of base {}", base.base.cols);
        let height = rows / base.base.rows;
        let width = cols / base.base.cols;
        let ept = base.base.elements_per_thread();
        let shape = vec![height, width, ept];
        let flat = shape.iter().product();
        let buf = self.alloc_reg(flat, dtype.clone());
        RT { buf, shape, layout, base, elem: dtype, ker: self }
    }

    /// Allocate a register vector [`RV`] tile (tinygrad `ker.rv`). `length` is
    /// the logical vector length, floored to a multiple of `base.base.rows`
    /// (the per-fragment row count) to give the fragment-tile count; the inner
    /// width is the fragment map's slot count.
    ///
    /// # Panics
    /// Panics (divide-by-zero) if `base.base.rows == 0`.
    pub fn rv(&self, length: usize, dtype: DType, layout: VecLayout, base: RTBaseShape) -> RV<'_> {
        let tiles = length / base.base.rows;
        let (outer, inner) = match layout {
            VecLayout::Ortho => (tiles, base.map.slots()),
        };
        let shape = vec![outer, inner];
        let buf = self.alloc_reg(outer * inner, dtype.clone());
        RV { buf, shape, length, layout, base, elem: dtype, ker: self }
    }
}
