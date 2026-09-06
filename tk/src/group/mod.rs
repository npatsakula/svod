//! The [`Group`] — a cooperating set of warps that owns the movement, load,
//! store, and WMMA ops, ported from tinygrad `extra/thunder/tiny/tk/group.py`.
//!
//! Each op opens its own *untracked* loops ([`Kernel::raw_range`]), builds the
//! terminal store closing those loops (`store.end(ranges)`, with a workgroup
//! `barrier` for the coalesced GLOBAL→LOCAL fill), records it via
//! [`Kernel::push_store`], and returns the destination tile *rewrapped* with an
//! `After([END(STORE)])` dependency — so a later read of the tile is ordered
//! after the write (tinygrad's `dst.after(dst_store)`).
//!
//! The per-concern op bodies live in submodules — [`elementwise`], [`reduce`],
//! [`shuffle`], [`mma`], [`movement`] — each a single `impl Group<'k>` block;
//! this module holds the shared scaffolding (the [`Group`] struct, the public
//! types, the i64-index helpers, and the low-level methods every concern calls).

use std::sync::Arc;

use smallvec::{SmallVec, smallvec};
use svod_codegen::llvm::nvptx::ops::{shfl_bfly, shfl_idx};
use svod_dtype::{DType, GpuArch};
use svod_ir::{AxisType, UOp};

use crate::index::{Idx, cidx, flat_offset};
use crate::kernel::Kernel;
use crate::tile::{GL, RT, RegTile, ST};

mod elementwise;
mod mma;
mod movement;
mod reduce;
mod shuffle;

/// The index inputs to a [`Group::load`]/[`Group::store`], named by ROLE so the two
/// ops no longer disagree on positional slots. `block` is the wave sub-tile / global
/// block offset (the old `idxs`); `frag` is the REG-side fragment offset (the old
/// `dst_idxs` on load / `src_idxs` on store). `axis` is the global-tile row-stride
/// split (ignored by the LOCAL↔REG hops). Owned `SmallVec` — constructed from any
/// [`IntoIdxs`] (a tuple of `Into<Idx>` elements or a back-compat `&[Idx]`).
#[derive(Clone, Default)]
pub struct MoveIdx {
    pub block: SmallVec<[Idx; 4]>,
    pub frag: SmallVec<[Idx; 4]>,
    pub axis: usize,
    /// Gate the GLOBAL↔REG hop against the tensor's true extent so a tile
    /// straddling a ragged `N`/`M`/`D` edge reads `0.0` (load) / drops the write
    /// (store) instead of touching out-of-bounds memory. No-op when the dims are
    /// tile-aligned (the gate is elided at build time). Only the GLOBAL↔REG hops
    /// honor it; LDS hops are always tile-sized.
    pub masked: bool,
}

impl MoveIdx {
    /// A wave/global `block` offset at `axis` (the common fill/gather/store case).
    pub fn block<I: crate::index::IntoIdxs>(idxs: I, axis: usize) -> Self {
        Self { block: idxs.into_idxs(), frag: SmallVec::new(), axis, masked: false }
    }
    /// A REG-side `frag` offset only.
    pub fn frag<I: crate::index::IntoIdxs>(idxs: I) -> Self {
        Self { frag: idxs.into_idxs(), block: SmallVec::new(), axis: 0, masked: false }
    }
    /// Both a `block` and a `frag` offset at `axis`.
    pub fn at<B: crate::index::IntoIdxs, F: crate::index::IntoIdxs>(block: B, frag: F, axis: usize) -> Self {
        Self { block: block.into_idxs(), frag: frag.into_idxs(), axis, masked: false }
    }
    /// Boundary-mask this GLOBAL↔REG hop (see [`MoveIdx::masked`]).
    pub fn masked(mut self) -> Self {
        self.masked = true;
        self
    }
}

/// A source tile that can be loaded INTO `Dst` by a [`Group`]. The legal
/// address-space pairs each have an impl (ST←GL, RT←ST, RT←GL); an illegal pair is a
/// *compile* error (no impl), not a runtime panic. `Output` is the rewrapped dst.
pub trait LoadInto<'k, Dst> {
    type Output;
    fn load_into(self, g: &Group<'k>, dst: Dst, ix: MoveIdx) -> Self::Output;
}

/// A source REG tile that can be stored INTO `Dst` (`ST` or `GL`); illegal pairs are
/// a compile error.
pub trait StoreInto<'k, Dst> {
    type Output;
    fn store_into(self, g: &Group<'k>, dst: Dst, ix: MoveIdx) -> Self::Output;
}

impl<'k> LoadInto<'k, ST> for GL {
    type Output = ST;
    fn load_into(self, g: &Group<'k>, dst: ST, ix: MoveIdx) -> ST {
        g.load_global_to_local(dst, &self, &ix.block, ix.axis, true)
    }
}
impl<'k> LoadInto<'k, RT<'k>> for ST {
    type Output = RT<'k>;
    fn load_into(self, g: &Group<'k>, dst: RT<'k>, ix: MoveIdx) -> RT<'k> {
        g.load_local_to_reg(dst, &self, &ix.frag, &ix.block)
    }
}
impl<'k> LoadInto<'k, RT<'k>> for GL {
    type Output = RT<'k>;
    fn load_into(self, g: &Group<'k>, dst: RT<'k>, ix: MoveIdx) -> RT<'k> {
        g.load_global_to_reg(dst, &self, &ix.frag, &ix.block, ix.axis, ix.masked)
    }
}
impl<'k> StoreInto<'k, ST> for RT<'k> {
    type Output = ST;
    fn store_into(self, g: &Group<'k>, dst: ST, ix: MoveIdx) -> ST {
        g.store_reg_to_local(dst, &self, &ix.block, &ix.frag)
    }
}
impl<'k> StoreInto<'k, GL> for RT<'k> {
    type Output = GL;
    fn store_into(self, g: &Group<'k>, dst: GL, ix: MoveIdx) -> GL {
        g.store_reg_to_global(dst, &self, &ix.block, &ix.frag, ix.axis, ix.masked)
    }
}

// ── Index (i64-typed) arithmetic helpers ───────────────────────────────────

pub(crate) fn idiv(a: &Arc<UOp>, k: i64) -> Arc<UOp> {
    a.try_div(&cidx(k)).expect("idiv")
}
pub(crate) fn imod(a: &Arc<UOp>, k: i64) -> Arc<UOp> {
    a.try_mod(&cidx(k)).expect("imod")
}
pub(crate) fn imul(a: &Arc<UOp>, k: i64) -> Arc<UOp> {
    if k == 1 { a.clone() } else { a.try_mul(&cidx(k)).expect("imul") }
}
pub(crate) fn iadd(a: &Arc<UOp>, b: &Arc<UOp>) -> Arc<UOp> {
    a.try_add(b).expect("iadd")
}
pub(super) fn ixor(a: &Arc<UOp>, k: i64) -> Arc<UOp> {
    a.try_xor_op(&cidx(k)).expect("ixor")
}
pub(super) fn iand(a: &Arc<UOp>, k: i64) -> Arc<UOp> {
    a.try_and_op(&cidx(k)).expect("iand")
}

/// Compare-exchange direction for [`Group::compare_exchange`] (sorting networks).
#[derive(Clone, Copy, Debug)]
pub enum SwapDir {
    /// The lower-index lane of each pair keeps the min (the larger goes high).
    Ascending,
    /// The lower-index lane keeps the max.
    Descending,
    /// Bitonic merge: ascending where `(laneid & bit) == 0`, else descending.
    ByLaneBit(i64),
}

/// Direction for [`Group::row_arg_reduce`]/[`Group::col_arg_reduce`]: select the
/// minimum (`Min`) or maximum (`Max`) element along the reduced axis and return
/// its index. Ties resolve to the smaller index (matching `Tensor::topk` /
/// `argmin`, whose `Int32` indices the result interoperates with).
#[derive(Clone, Copy, Debug)]
pub enum ArgDir {
    Min,
    Max,
}

impl ArgDir {
    /// The value-accumulator seed: `+∞` for `Min`, `−∞` for `Max`, so the init
    /// always loses to any real element.
    pub(super) fn init(self) -> f64 {
        match self {
            ArgDir::Min => f64::INFINITY,
            ArgDir::Max => f64::NEG_INFINITY,
        }
    }
}

/// Keep the `(value, index)` pair that is the extremum per `dir`, ties → the
/// smaller index (matching `Tensor::topk`/`argmin`). Selecting BOTH outputs by
/// the same predicate guarantees the kept value equals the element at the kept
/// index. `strict` (strict `<`/`>`) and `eq` are mutually exclusive, so
/// `keep = where(va==vb, ia<ib, strict)` needs no boolean and/or. The min is
/// synthesized from `Lt` (no `BinaryOp::Min`, matching the `ReduceOp::Min` /
/// [`Group::compare_exchange`] convention); the comparisons dispatch on dtype
/// (float `fcmp`, signed/unsigned `icmp`), so it is dtype-specific.
pub(super) fn arg_fold(
    dir: ArgDir,
    va: &Arc<UOp>,
    ia: &Arc<UOp>,
    vb: &Arc<UOp>,
    ib: &Arc<UOp>,
) -> (Arc<UOp>, Arc<UOp>) {
    let strict = match dir {
        ArgDir::Min => va.try_cmplt(vb),
        ArgDir::Max => vb.try_cmplt(va),
    }
    .expect("arg_fold: strict cmp");
    let eq = va.try_cmpeq(vb).expect("arg_fold: eq cmp");
    let itie = ia.try_cmplt(ib).expect("arg_fold: index tie cmp");
    let keep = UOp::try_where(eq, itie, strict).expect("arg_fold: keep predicate");
    let v = UOp::try_where(keep.clone(), va.clone(), vb.clone()).expect("arg_fold: value select");
    let i = UOp::try_where(keep, ia.clone(), ib.clone()).expect("arg_fold: index select");
    (v, i)
}

pub(super) fn idx_mul(idx: &Idx, k: i64) -> Idx {
    match idx {
        Idx::Const(c) => Idx::Const(c * k),
        Idx::Uop(u) => Idx::Uop(imul(u, k)),
    }
}

/// The wave sub-tile fragment index for a shared-tile axis (SI-1):
/// `block * frags + local`, where `block` (a wave's row/col in the wave grid,
/// already including `warp_row`/`warp_col`) selects which `frags`-tall slice of
/// the shared tile this wave reads/writes. `None` ⇒ no offset (single-warp).
pub(super) fn wave_offset(block: Option<&Idx>, frags: i64, local: &Arc<UOp>) -> Idx {
    match block {
        None => Idx::from(local),
        Some(b) => Idx::Uop(iadd(&imul(&b.to_uop(), frags), local)),
    }
}

/// A cooperating set of `warps` waves laid out in a `rows_waves × cols_waves`
/// grid (tinygrad `Group` / HK `group<NUM_WARPS>`). Each wave owns a sub-tile of
/// the shared tiles; the GLOBAL→LDS fill is collaborative over all
/// `group_threads`.
pub struct Group<'k> {
    pub warps: usize,
    pub rows_waves: usize,
    pub cols_waves: usize,
    group_threads: usize,
    ker: &'k Kernel,
}

impl Kernel {
    /// A single-warp [`Group`] (tinygrad `ker.warp`) — the `1×1` wave grid that owns
    /// the per-lane register ops (`clear`/`copy`/`map`/`reduce`/`shuffle`/`mma`).
    pub fn warp(&self) -> Group<'_> {
        self.group_2d(1, 1)
    }
    /// An `n`-warp [`Group`] laid out `1×n` (tinygrad `ker.group`) — `n` cooperating
    /// waves for the collaborative GLOBAL→LDS fill.
    pub fn group(&self, n: usize) -> Group<'_> {
        self.group_2d(1, n)
    }
    /// An `R×C`-wave group: one workgroup runs `rows_waves * cols_waves` waves
    /// (`group_threads = warps * 64`), each owning a sub-tile of the shared
    /// tiles (HK 2×4 wave grid, `GEMM:67-68`).
    pub fn group_2d(&self, rows_waves: usize, cols_waves: usize) -> Group<'_> {
        let warps = rows_waves * cols_waves;
        Group { warps, rows_waves, cols_waves, group_threads: warps * self.caps.wave_size, ker: self }
    }
}

impl<'k> Group<'k> {
    /// The group lane id (`threadIdx % group_threads`).
    fn laneid(&self) -> Arc<UOp> {
        imod(&self.ker.thread_idx, self.group_threads as i64)
    }

    /// Total threads in the workgroup (`warps * 64`) — the launch block size.
    pub fn group_threads(&self) -> usize {
        self.group_threads
    }

    /// The wave's flat index within the group (`(threadIdx % group_threads)/64`).
    pub fn warpid_in_group(&self) -> Arc<UOp> {
        idiv(&imod(&self.ker.thread_idx, self.group_threads as i64), self.ker.caps.wave_size as i64)
    }
    /// The wave's row in the `rows_waves × cols_waves` wave grid (`GEMM:67`).
    pub fn warp_row(&self) -> Arc<UOp> {
        idiv(&self.warpid_in_group(), self.cols_waves as i64)
    }
    /// The wave's column in the wave grid (`GEMM:68`).
    pub fn warp_col(&self) -> Arc<UOp> {
        imod(&self.warpid_in_group(), self.cols_waves as i64)
    }

    /// Anchor a tile read to every enclosing tracked loop. This mirrors
    /// tinygrad TK's `tile.after(*ker.range_stack)`: an inner helper range does
    /// not by itself keep a loop-carried REG load inside the outer loop.
    pub(crate) fn anchor(&self, buf: &Arc<UOp>) -> Arc<UOp> {
        let tracked = self.ker.tracked_ranges();
        if tracked.is_empty() { buf.clone() } else { buf.after(tracked) }
    }

    /// Build a per-element register op body — one bare `STORE` per logical
    /// element. Looped (the default): open a `Loop` `RANGE` per dim and close one
    /// store around them. Fully **unrolled** (the kernel's [`Kernel::unrolled`]
    /// flag): emit a bare store per element position, grouped into one node (no
    /// `RANGE`), so the body renders flat for the FA scheduling comb. `store_at`
    /// builds one element's `STORE` from its index tuple.
    fn elementwise<F>(&self, shape: &[usize], store_at: F) -> Arc<UOp>
    where
        F: Fn(&[Idx]) -> Arc<UOp>,
    {
        if self.ker.unrolled() {
            let stores: Vec<Arc<UOp>> = cartesian(shape).iter().map(|idxs| store_at(idxs)).collect();
            if stores.len() == 1 { stores.into_iter().next().unwrap() } else { UOp::group(stores) }
        } else {
            let rngs: Vec<Arc<UOp>> = shape.iter().map(|&d| self.ker.raw_range(d as i64, AxisType::Loop)).collect();
            let idxs: Vec<Idx> = rngs.iter().map(Idx::from).collect();
            store_at(&idxs).end(SmallVec::from_vec(rngs))
        }
    }

    /// Read this lane's `value` from lane `src_lane` within the wave — an
    /// in-register cross-lane gather with no LDS and no barrier, lowered per arch:
    /// `llvm.amdgcn.ds.bpermute` on AMD (i32-typed; lane `L` receives `data` from
    /// lane `byte_addr(L) >> 2`, so f32 is bitcast through i32 and the byte address
    /// is `src_lane * 4`), `shfl.sync.idx` on CUDA. Both ride the typed `Op::Custom`
    /// path (the `declare` is auto-hoisted+deduped to the module prefix).
    ///
    /// # Panics
    /// Panics on an arch without a shuffle lowering (Metal).
    pub(super) fn shuffle_lane(&self, value: &Arc<UOp>, src_lane: &Arc<UOp>) -> Arc<UOp> {
        match self.ker.caps.arch {
            GpuArch::Amd(_) => {
                let is_f32 = value.dtype() == DType::Float32;
                let data_i = if is_f32 { value.bitcast(DType::Int32) } else { value.clone() };
                let addr = imul(src_lane, 4).cast(DType::Int32);
                let sh = UOp::custom(
                    smallvec![addr, data_i],
                    "declare i32 @llvm.amdgcn.ds.bpermute(i32, i32)\n\
                     call i32 @llvm.amdgcn.ds.bpermute(i32 {0}, i32 {1})"
                        .to_string(),
                    DType::Int32,
                );
                if is_f32 { sh.bitcast(DType::Float32) } else { sh }
            }
            GpuArch::Cuda(_) => shfl_idx(value, src_lane),
            GpuArch::Metal(_) => unimplemented!("tk cross-lane shuffle has no Metal lowering"),
        }
    }

    /// Butterfly gather: this lane's `value` from lane `laneid ^ mask` — the
    /// `shfl.sync.bfly` immediate form on CUDA, the same `ds_bpermute` as
    /// [`Self::shuffle_lane`] with a computed partner on AMD.
    pub(super) fn shuffle_xor_lane(&self, value: &Arc<UOp>, mask: i64) -> Arc<UOp> {
        match self.ker.caps.arch {
            GpuArch::Cuda(_) => shfl_bfly(value, &cidx(mask)),
            _ => self.shuffle_lane(value, &ixor(&self.laneid(), mask)),
        }
    }

    // ── store bookkeeping helpers ───────────────────────────────────────────

    pub(super) fn finalize_reg(&self, t: RT<'k>, ended: Arc<UOp>) -> RT<'k> {
        self.finalize_tile(t, ended)
    }
    /// Record `ended` as a terminal store and rewrap the register tile so later
    /// reads order after it (tinygrad `dst.after(dst_store)`).
    pub(super) fn finalize_tile<T: RegTile<'k>>(&self, t: T, ended: Arc<UOp>) -> T {
        self.ker.push_store(ended.clone(), t.uop().clone());
        let after = t.uop().after(smallvec![ended]);
        t.rewrap(after)
    }
    pub(super) fn finalize_st(&self, t: ST, ended: Arc<UOp>) -> ST {
        self.ker.push_store(ended.clone(), t.uop().clone());
        let after = t.uop().after(smallvec![ended]);
        t.rewrap(after)
    }
    pub(super) fn finalize_gl(&self, t: GL, ended: Arc<UOp>) -> GL {
        self.ker.push_store(ended.clone(), t.uop().clone());
        let after = t.uop().after(smallvec![ended]);
        t.rewrap(after)
    }
}

impl<'k> RT<'k> {
    /// This lane's `(row, col)` of register `inner` within a base fragment, per the
    /// tile's [`LaneMap`](crate::layout::LaneMap). `transpose` selects the
    /// "fragment is laid out column-major in registers" reading (group.py: either
    /// `rt.layout != st.layout` for the LDS hops, or `rt.layout == COL` for the
    /// global hops).
    pub(crate) fn lane_rc(&self, transpose: bool, laneid: &Arc<UOp>, inner: &Arc<UOp>) -> (Arc<UOp>, Arc<UOp>) {
        self.base.map.rc(transpose, laneid, self.base.base.rows as i64, self.base.base.cols as i64, inner)
    }
}

impl ST {
    /// A zero-copy view of the `(row_blk, col_blk)`-th sub-rectangle of warp-tile
    /// element size `dims` — folds the per-warp band into the tile's additive base
    /// offset (composing with any existing double-buffer parity offset), so a
    /// subsequent [`Group::load`]/[`Group::store`] needs **no** wave-block index
    /// (pass [`MoveIdx::default`]). `dims` is the consuming register tile's element
    /// shape (the warp-tile size). Addresses the SAME element as the equivalent
    /// `wave_offset` block — the band is whole-fragment-granular (so the LDS swizzle,
    /// applied within a fragment, is unaffected); only the offset op-tree differs
    /// from the folded form (`imul(a·k + local, stride)` → `imul(local, stride) +
    /// imul(a·k, stride)`), so it is correct-by-construction but changes the kernel's
    /// content hash.
    pub fn subtile<R: Into<Idx>, C: Into<Idx>>(&self, dims: (usize, usize), blk: (R, C)) -> ST {
        let blk = (blk.0.into(), blk.1.into());
        let frag_h = (dims.0 / self.base.base.rows) as i64;
        let frag_w = (dims.1 / self.base.base.cols) as i64;
        let band = flat_offset(
            self.shape(),
            &[idx_mul(&blk.0, frag_h), idx_mul(&blk.1, frag_w), Idx::Const(0), Idx::Const(0)],
        );
        let off = match self.base_offset() {
            Some(bo) => band.try_add(bo).expect("subtile band + base offset"),
            None => band,
        };
        self.with_base_offset(off)
    }
}

/// The row-major cartesian product of `0..d` for each `d` in `shape` — the
/// constant index tuples an unrolled register op iterates (the analog of the
/// nested `Loop` `RANGE`s it replaces).
fn cartesian(shape: &[usize]) -> Vec<Vec<Idx>> {
    let mut acc = vec![Vec::new()];
    for &d in shape {
        acc = acc
            .into_iter()
            .flat_map(|prefix| {
                (0..d as i64).map(move |i| {
                    let mut next = prefix.clone();
                    next.push(Idx::Const(i));
                    next
                })
            })
            .collect();
    }
    acc
}
