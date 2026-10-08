//! Relayout classification: what moving a tile from one layout to another costs.

use smallvec::SmallVec;

use super::Dim::{self, Block, Lane, Reg, Row, Warp};
use super::{Echelon, Layout};

/// How a value held under `src` is re-held under `dst`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Conversion {
    Identity,
    /// Free: destination register `j` is source register `perm[j]` of the same lane.
    RegPermute(SmallVec<[u32; 16]>),
    /// Within a warp: see [`ShufflePlan`].
    LaneShuffle(ShufflePlan),
    /// Data crosses warps: a shared-memory round trip.
    ViaSmem,
}

/// `src⁻¹ ∘ dst`: where each destination `(lane, reg)` reads from.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ShufflePlan(Box<Layout>);

impl ShufflePlan {
    /// The linear map destination `(Reg, Lane, ..)` → source `(Reg, Lane, ..)`.
    pub fn map(&self) -> &Layout {
        &self.0
    }

    /// `(source lane, source register)` of destination register `reg` in lane `lane`
    /// of warp `warp` (the source warp is `warp` itself; the plan may depend on it).
    pub fn source(&self, warp: u32, lane: u32, reg: u32) -> (u32, u32) {
        let out = self.0.apply_dims(&[(Warp, warp), (Lane, lane), (Reg, reg)]);
        let get = |dim| out.iter().find(|o| o.0 == dim).map_or(0, |o| o.1);
        (get(Lane), get(Reg))
    }

    /// Per destination register, per destination lane of warp `warp`: the [`Self::source`] it reads.
    pub fn rounds(&self, warp: u32) -> Vec<Vec<(u32, u32)>> {
        let lanes = self.0.in_size(Lane);
        (0..self.0.in_size(Reg)).map(|j| (0..lanes).map(|l| self.source(warp, l, j)).collect()).collect()
    }
}

impl Conversion {
    /// Classifies `src⁻¹ ∘ dst` by the hardware bits it mixes. Panics unless both lay
    /// out the same tile and `src` holds every element `dst` reads.
    pub fn between(src: &Layout, dst: &Layout) -> Self {
        assert_eq!(src.outs, dst.outs, "conversion between different tiles");
        if src == dst {
            return Conversion::Identity;
        }
        let m = src.pseudo_inverse().compose(dst);
        assert!(&src.compose(&m) == dst, "the source layout does not hold every element the destination reads");
        const WIDE: [Dim; 2] = [Warp, Block];
        if m.touches(&[Reg, Lane], &WIDE) || !m.sublayout(&WIDE, &WIDE).is_identity() {
            Conversion::ViaSmem
        } else if m.passes(Lane) && !m.touches(&[Reg], &[Lane]) && !m.touches(&WIDE, &[Reg, Lane]) {
            Conversion::RegPermute(
                (0..dst.in_size(Reg)).map(|j| ShufflePlan(Box::new(m.clone())).source(0, 0, j).1).collect(),
            )
        } else {
            Conversion::LaneShuffle(ShufflePlan(Box::new(m)))
        }
    }
}

/// The lane xor-masks whose butterfly folds a row: a basis (reduced, ascending) of the
/// lane deltas that keep the `Row` coordinate (the kernel of `Lane → Row`).
pub fn lanes_sharing_row(layout: &Layout) -> SmallVec<[u32; 6]> {
    let row = layout.sublayout(&[Lane], &[Row]);
    let mut image = Echelon::default();
    let kernel: SmallVec<[u64; 6]> =
        row.bases(Lane).iter().enumerate().filter_map(|(i, &v)| image.insert(v, 1 << i)).collect();
    let mut basis = Echelon::default();
    for v in kernel {
        basis.insert(v, 0);
    }
    // Back-substitute so every mask carries only its own leading bit among the others'.
    let mut masks: SmallVec<[u64; 6]> = basis.0.iter().map(|p| p.0).collect();
    for i in 0..masks.len() {
        let lead = 1u64 << (63 - masks[i].leading_zeros());
        for k in 0..masks.len() {
            if k != i && masks[k] & lead != 0 {
                masks[k] ^= masks[i];
            }
        }
    }
    let mut masks: SmallVec<[u32; 6]> = masks.into_iter().map(|m| m as u32).collect();
    masks.sort_unstable();
    masks
}
