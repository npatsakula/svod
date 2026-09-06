//! Per-lane fragment layouts: the closed-form `(lane, j) → (row, col)` maps of one
//! matrix-core base fragment, one variant per arch family. Every consumer of the
//! lane→element map — the LDS/GLOBAL fragment hops, `map_position`, the reduce
//! fold/tree and `axis_index_of` — reads through [`LaneMap`], so a layout is
//! described once, as data, and never re-derived from boolean flags.
//!
//! Each map has a **lane axis** (the coordinate fixed per lane; the register axis
//! runs along the other one). A reduce keeps the lane axis: on the AMD maps a lane
//! holds one kept value per fragment, so the fold collapses every in-lane element
//! and the `wave_size/16` sibling lane-groups; the `mma.sync` map spreads a lane's
//! eight elements over two rows (`g`, `g+8`) and four columns, so a lane keeps
//! [`LaneMap::slots`] `= 2` values and the column fold finishes in the 4-lane quad
//! (`shfl.bfly` masks 1, 2).

use std::sync::Arc;

use smallvec::SmallVec;
use svod_ir::UOp;

use crate::group::{iadd, idiv, imod, imul};
use crate::index::Idx;

/// The integer arithmetic a [`LaneMap`] formula is written against — evaluated
/// once over `Index`-typed UOps (kernel build) and once over plain integers
/// (layout tests, bijection proofs), from the SAME expression, so the two can
/// never disagree.
pub trait LaneArith: Clone {
    fn add(&self, o: &Self) -> Self;
    fn mul(&self, k: i64) -> Self;
    fn div(&self, k: i64) -> Self;
    fn rem(&self, k: i64) -> Self;
}

impl LaneArith for Arc<UOp> {
    fn add(&self, o: &Self) -> Self {
        iadd(self, o)
    }
    fn mul(&self, k: i64) -> Self {
        imul(self, k)
    }
    fn div(&self, k: i64) -> Self {
        idiv(self, k)
    }
    fn rem(&self, k: i64) -> Self {
        imod(self, k)
    }
}

impl LaneArith for i64 {
    fn add(&self, o: &Self) -> Self {
        self + o
    }
    fn mul(&self, k: i64) -> Self {
        self * k
    }
    fn div(&self, k: i64) -> Self {
        self / k
    }
    fn rem(&self, k: i64) -> Self {
        self % k
    }
}

/// The cross-lane completion of a fragment reduce, after the in-lane fold.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ReduceTree {
    /// Gather the ORIGINAL partial of every sibling lane-group `lane + d` (mod the
    /// group) and fold them in order — the AMD `ds_bpermute` sibling tree.
    Gather(SmallVec<[i64; 3]>),
    /// Butterfly: fold the RUNNING accumulator with lane `lane ^ m` per mask —
    /// the CUDA `shfl.sync.bfly` quad reduce.
    Butterfly(SmallVec<[i64; 3]>),
}

impl ReduceTree {
    /// Number of shuffles the tree issues.
    pub fn len(&self) -> usize {
        match self {
            ReduceTree::Gather(v) | ReduceTree::Butterfly(v) => v.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Closed-form lane→(row, col) map of element `j` held by lane `L` in a
/// `rows × cols` base fragment. `transpose` (a `Col`-layout register tile, or a
/// `Row`/`Col` mismatch on an LDS hop) reads the map with row/col swapped where the
/// map is orientation-free; the RDNA accumulator maps carry their orientation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LaneMap {
    /// CDNA MFMA (and the RDNA replicated WMMA input at `stride = 0`):
    /// `row = L % rows, col = (L / rows)·stride + j` — K spread over `wave/rows`
    /// lane-groups; `stride = 0` ⇒ every lane-group holds the same full K run.
    Strided { stride: usize },
    /// RDNA3 WMMA f32 accumulator, even/odd row interleave across the two
    /// wave-halves: `row = 2j + L/16, col = L % 16` (tinygrad `ops_python` `c_map`).
    Interleaved,
    /// The transpose of [`Self::Interleaved`] (`row = L % 16, col = 2j + L/16`): an
    /// RDNA accumulator stored along the N-major axis.
    InterleavedT,
    /// NVIDIA `mma.sync.m16n8k16` (ThunderKittens `rt_base`): a 16×16 tile as two
    /// m16n8 halves along the register axis, with `g = L>>2, t = L&3`:
    /// `row = g + 8·((j/2)%2), col = 2t + j%2 + 8·(j/4)`. Registers `0..4` are the
    /// `col 0..8` half, `4..8` the `col 8..16` half; the same 8 registers are the
    /// PTX **A** fragment `a0..a7`, the two **C** fragments `c0..c3` per half, and
    /// (read transposed) the two **B** fragments `{0,1,4,5}` / `{2,3,6,7}`.
    MmaSync,
}

impl LaneMap {
    /// `(row, col)` of element `j` in lane `lane` (see the variant docs).
    pub fn rc<A: LaneArith>(&self, transpose: bool, lane: &A, rows: i64, cols: i64, j: &A) -> (A, A) {
        match *self {
            LaneMap::Strided { stride } => {
                let stride = stride as i64;
                if transpose {
                    (lane.div(cols).mul(stride).add(j), lane.rem(cols))
                } else {
                    (lane.rem(rows), lane.div(rows).mul(stride).add(j))
                }
            }
            LaneMap::Interleaved => (j.mul(2).add(&lane.div(cols)), lane.rem(cols)),
            LaneMap::InterleavedT => (lane.rem(cols), j.mul(2).add(&lane.div(cols))),
            LaneMap::MmaSync => {
                let (g, t) = (lane.div(4), lane.rem(4));
                let r = g.add(&j.div(2).rem(2).mul(8));
                let c = t.mul(2).add(&j.rem(2)).add(&j.div(4).mul(8));
                if transpose { (c, r) } else { (r, c) }
            }
        }
    }

    /// Whether the folded (register) axis is the tile's **column** axis under
    /// `transpose` — the coordinate that varies with `j`; the reduce keeps the other.
    pub fn folds_cols(&self, transpose: bool) -> bool {
        match self {
            LaneMap::Strided { .. } | LaneMap::MmaSync => !transpose,
            LaneMap::Interleaved => false,
            LaneMap::InterleavedT => true,
        }
    }

    /// Kept values a lane holds per fragment after a reduce: the register-vector
    /// inner width. `2` on [`Self::MmaSync`] (rows `g` and `g+8`), else `1`.
    pub const fn slots(&self) -> usize {
        match self {
            LaneMap::MmaSync => 2,
            _ => 1,
        }
    }

    /// The register-vector slot element `j` folds into / broadcasts from: `(j/2)%2`
    /// on [`Self::MmaSync`], the single slot `0` elsewhere (a constant, so the
    /// single-slot graphs are unchanged).
    pub fn slot_of(&self, j: &Idx) -> Idx {
        match (self, j) {
            (LaneMap::MmaSync, Idx::Const(c)) => Idx::Const((c / 2) % 2),
            (LaneMap::MmaSync, Idx::Uop(u)) => Idx::Uop(u.div(2).rem(2)),
            _ => Idx::Const(0),
        }
    }

    /// The cross-lane completion for a `wave_size`-lane wave: the AMD maps fold
    /// the `wave_size/16` sibling lane-groups (`[16, 32, 48]` at wave64, `[16]` at
    /// wave32); [`Self::MmaSync`] folds the quad (`L ^ 1`, `L ^ 2`).
    pub fn tree(&self, wave_size: usize) -> ReduceTree {
        match self {
            LaneMap::MmaSync => ReduceTree::Butterfly([1, 2].into_iter().collect()),
            _ => ReduceTree::Gather((1..wave_size as i64 / 16).map(|i| i * 16).collect()),
        }
    }

    /// The `ldmatrix.x4` load of a 16×16 16-bit fragment held under this map
    /// (`transpose` as in [`Self::rc`]), if one exists: the four 8×8 matrices the
    /// warp fetches (lane `L` addressing row `L % 16`, columns `8·(L / 16)..` — TL,
    /// BL, TR, BR) each land as one 32-bit register pair of adjacent elements, so
    /// the map must give lane `L` register pair `p` = elements `(L/4, 2(L%4) + e)`
    /// of one matrix (or its transpose under `.trans`). Proved by evaluating the
    /// closed form over every lane and register — the plan is derived, never
    /// hand-permuted: [`Self::MmaSync`] reads `[0, 1, 2, 3]` plain and `[0, 2, 1,
    /// 3]` transposed (ThunderKittens `ldsm4t(tmp[0], tmp[2], tmp[1], tmp[3])`).
    pub fn ldmatrix_x4(&self, transpose: bool) -> Option<LdmatrixX4> {
        [false, true].into_iter().find_map(|trans| {
            let mut words = [0usize; 4];
            for (p, word) in words.iter_mut().enumerate() {
                let (r, c) = self.rc(transpose, &0i64, 16, 16, &(2 * p as i64));
                let (rb, cb) = (r / 8, c / 8);
                *word = (rb + 2 * cb) as usize;
                for lane in 0..32i64 {
                    let (g, t) = (lane / 4, lane % 4);
                    for e in 0..2i64 {
                        let want =
                            if trans { (8 * rb + 2 * t + e, 8 * cb + g) } else { (8 * rb + g, 8 * cb + 2 * t + e) };
                        if self.rc(transpose, &lane, 16, 16, &(2 * p as i64 + e)) != want {
                            return None;
                        }
                    }
                }
            }
            Some(LdmatrixX4 { trans, words })
        })
    }
}

/// How one `ldmatrix.sync.aligned.m8n8.x4[.trans]` fills a 16×16 fragment: register
/// pair `p` (elements `2p, 2p+1`) is the fetched matrix `words[p]` (see
/// [`LaneMap::ldmatrix_x4`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct LdmatrixX4 {
    pub trans: bool,
    pub words: [usize; 4],
}
