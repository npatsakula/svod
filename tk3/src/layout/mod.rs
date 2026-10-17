//! F2 linear layouts (Triton "Linear Layouts", arXiv 2505.23819): a layout is a linear
//! map over GF(2) from the bits of named input dimensions to the bits of named output
//! dimensions, stored as one basis vector per input bit. Output values are packed into
//! a `u64`, the output dimensions in [`Dim`] order, each at the offset of the ones
//! before it. Design: `tk3_design.md` §3.2.

mod atoms;
mod conversion;

use std::ops::Range;

use smallvec::SmallVec;

pub use atoms::*;
pub use conversion::{Conversion, ShufflePlan, lanes_sharing_row};

/// A named dimension: the hardware inputs `Reg`/`Lane`/`Warp`/`Block`, and the tile
/// coordinates `Row`/`Col`. Any dimension may be an input or an output (an inverse
/// maps tile coordinates back to hardware, a swizzle maps coordinates to coordinates).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Dim {
    Reg,
    Lane,
    Warp,
    Block,
    Row,
    Col,
}

type Bases = SmallVec<[u64; 8]>;
type Outs = SmallVec<[(Dim, u32); 3]>;

/// Invariants: input and output dimensions sorted by [`Dim`], none of zero bits;
/// output dims are `(dim, log2 size)`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Layout {
    ins: SmallVec<[(Dim, Bases); 4]>,
    outs: Outs,
}

fn log2(size: u32) -> u32 {
    assert!(size.is_power_of_two(), "layout size {size} is not a power of two");
    size.trailing_zeros()
}

fn mask(bits: u32) -> u64 {
    if bits >= 64 { !0 } else { (1 << bits) - 1 }
}

fn bits_of(mut v: u64) -> impl Iterator<Item = u32> {
    std::iter::from_fn(move || {
        (v != 0).then(|| {
            let b = v.trailing_zeros();
            v &= v - 1;
            b
        })
    })
}

/// Row echelon form over GF(2): `(vector, preimage)` pairs with distinct leading bits,
/// kept in descending order so [`Echelon::reduce`] clears every leading bit in one pass.
#[derive(Default)]
struct Echelon(SmallVec<[(u64, u64); 16]>);

impl Echelon {
    fn reduce(&self, mut v: u64) -> (u64, u64) {
        let mut pre = 0;
        for &(pv, pp) in &self.0 {
            if v & (1 << (63 - pv.leading_zeros())) != 0 {
                v ^= pv;
                pre ^= pp;
            }
        }
        (v, pre)
    }

    /// Adds `v` (the image of `pre`); a dependent `v` instead yields its kernel element.
    fn insert(&mut self, v: u64, pre: u64) -> Option<u64> {
        let (r, q) = self.reduce(v);
        if r == 0 {
            return Some(pre ^ q);
        }
        let at = self.0.iter().position(|&(pv, _)| pv < r).unwrap_or(self.0.len());
        self.0.insert(at, (r, pre ^ q));
        None
    }
}

impl Layout {
    fn build(outs: impl IntoIterator<Item = (Dim, u32)>, ins: impl IntoIterator<Item = (Dim, Bases)>) -> Self {
        let mut outs: Outs = outs.into_iter().filter(|o| o.1 > 0).collect();
        outs.sort_unstable_by_key(|o| o.0);
        assert!(outs.windows(2).all(|w| w[0].0 != w[1].0), "duplicate output dim");
        assert!(outs.iter().map(|o| o.1).sum::<u32>() <= 64, "more than 64 output bits");
        let mut ins: SmallVec<[(Dim, Bases); 4]> = ins.into_iter().filter(|i| !i.1.is_empty()).collect();
        ins.sort_unstable_by_key(|i| i.0);
        assert!(ins.windows(2).all(|w| w[0].0 != w[1].0), "duplicate input dim");
        assert!(ins.iter().map(|i| i.1.len()).sum::<usize>() <= 64, "more than 64 input bits");
        Layout { ins, outs }
    }

    fn offset(outs: &Outs, dim: Dim) -> Option<(u32, u32)> {
        let mut off = 0;
        for &(d, b) in outs {
            if d == dim {
                return Some((off, b));
            }
            off += b;
        }
        None
    }

    fn pack(outs: &Outs, vals: impl IntoIterator<Item = (Dim, u32)>) -> u64 {
        vals.into_iter().filter(|v| v.1 != 0).fold(0, |acc, (d, v)| {
            let (off, b) = Self::offset(outs, d).unwrap_or_else(|| panic!("no output dim {d:?}"));
            assert!(u64::from(v) <= mask(b), "{d:?} value {v} exceeds {} bits", b);
            acc ^ (u64::from(v) << off)
        })
    }

    fn unpack(&self, v: u64) -> SmallVec<[(Dim, u32); 3]> {
        let mut off = 0;
        self.outs
            .iter()
            .map(|&(d, b)| {
                let x = (v >> off) & mask(b);
                off += b;
                (d, x as u32)
            })
            .collect()
    }

    fn bases(&self, dim: Dim) -> &[u64] {
        self.ins.iter().find(|i| i.0 == dim).map_or(&[], |i| &i.1)
    }

    fn flat(&self) -> impl Iterator<Item = u64> + '_ {
        self.ins.iter().flat_map(|i| i.1.iter().copied())
    }

    fn echelon(&self) -> Echelon {
        let mut e = Echelon::default();
        for (k, v) in self.flat().enumerate() {
            e.insert(v, 1 << k);
        }
        e
    }

    /// Whether some input bit of `from` reaches an output bit of `to`.
    fn touches(&self, from: &[Dim], to: &[Dim]) -> bool {
        let to: u64 = to.iter().filter_map(|&d| Self::offset(&self.outs, d)).map(|(o, b)| mask(b) << o).sum();
        from.iter().flat_map(|&d| self.bases(d)).any(|v| v & to != 0)
    }

    /// Whether input `dim` maps bit-for-bit onto the same-sized output `dim` and nowhere else.
    fn passes(&self, dim: Dim) -> bool {
        let unit = |i: usize| Self::offset(&self.outs, dim).map(|(o, _)| 1u64 << (o + i as u32));
        self.in_size(dim) == self.out_size(dim) && self.bases(dim).iter().enumerate().all(|(i, &v)| Some(v) == unit(i))
    }

    // ── construction ────────────────────────────────────────────────────────────

    /// `in`'s bits onto the low bits of `out`, both of `size` elements.
    pub fn identity(dim: Dim, size: u32, out: Dim) -> Self {
        let b = log2(size);
        let outs: Outs = [(out, b)].into_iter().collect();
        Self::build(outs, [(dim, (0..b).map(|i| 1 << i).collect())])
    }

    /// `size` elements of `dim` that reach no output: broadcast / replication.
    pub fn zeros(dim: Dim, size: u32) -> Self {
        Self::build([], [(dim, (0..log2(size)).map(|_| 0).collect())])
    }

    /// Outputs `(dim, size)`; per input dim, one basis vector per input bit, each the
    /// output values (in `outs` order) that bit maps to.
    pub fn from_bases<const N: usize>(outs: [(Dim, u32); N], ins: &[(Dim, &[[u32; N]])]) -> Self {
        let packed: Outs = {
            let mut o: Outs = outs.iter().map(|&(d, s)| (d, log2(s))).filter(|o| o.1 > 0).collect();
            o.sort_unstable_by_key(|o| o.0);
            o
        };
        let ins = ins.iter().map(|&(d, bs)| {
            (d, bs.iter().map(|b| Self::pack(&packed, outs.iter().zip(b).map(|(o, &v)| (o.0, v)))).collect())
        });
        Self::build(packed.clone(), ins)
    }

    /// CuTe `Swizzle<B, M, S>` on a row-major `rows × cols` tile, as a `(Row, Col) →
    /// (Row, Col)` map: offset `o = row·cols + col` becomes `o ^ ((o >> S) & (2^B - 1) << M)`.
    pub fn swizzle_xor(rows: u32, cols: u32, b: u32, m: u32, s: u32) -> Self {
        let (rb, cb) = (log2(rows), log2(cols));
        assert!(s >= b && m + s + b <= rb + cb, "Swizzle<{b}, {m}, {s}> on {rows}x{cols}");
        let outs: Outs = [(Dim::Row, rb), (Dim::Col, cb)].into_iter().collect();
        let image = |p: u32| {
            let o = (1u64 << p) | if (m + s..m + s + b).contains(&p) { 1 << (p - s) } else { 0 };
            Self::pack(&outs, [(Dim::Row, (o >> cb) as u32), (Dim::Col, (o & mask(cb)) as u32)])
        };
        Self::build(
            outs.clone(),
            [(Dim::Row, (cb..cb + rb).map(image).collect()), (Dim::Col, (0..cb).map(image).collect())],
        )
    }

    /// Swaps the `Row` and `Col` outputs.
    pub fn transpose(&self) -> Self {
        let swap = |d| match d {
            Dim::Row => Dim::Col,
            Dim::Col => Dim::Row,
            d => d,
        };
        self.map_outs(self.outs.iter().map(|&(d, b)| (swap(d), b)), |d, v| Some((swap(d), v)))
    }

    /// Repacks every basis vector under new outputs, `f` translating each output value.
    fn map_outs(&self, outs: impl IntoIterator<Item = (Dim, u32)>, f: impl Fn(Dim, u32) -> Option<(Dim, u32)>) -> Self {
        let outs: Outs = {
            let mut o: Outs = outs.into_iter().filter(|o| o.1 > 0).collect();
            o.sort_unstable_by_key(|o| o.0);
            o
        };
        let ins = self.ins.iter().map(|(d, bs)| {
            (
                *d,
                bs.iter()
                    .map(|&v| Self::pack(&outs, self.unpack(v).into_iter().filter_map(|(d, x)| f(d, x))))
                    .collect(),
            )
        });
        Self::build(outs.clone(), ins)
    }

    // ── algebra ─────────────────────────────────────────────────────────────────

    /// `self ∘ inner`: `inner`'s outputs feed `self`'s inputs of the same name.
    pub fn compose(&self, inner: &Layout) -> Self {
        for &(d, b) in &inner.outs {
            assert!(b <= self.in_bits(d), "compose: inner output {d:?} has {b} bits, outer input {}", self.in_bits(d));
        }
        let image = |v: u64| {
            inner
                .unpack(v)
                .into_iter()
                .fold(0, |acc, (d, x)| bits_of(x.into()).fold(acc, |a, i| a ^ self.bases(d)[i as usize]))
        };
        Self::build(self.outs.clone(), inner.ins.iter().map(|(d, bs)| (*d, bs.iter().map(|&v| image(v)).collect())))
    }

    /// A linear `P` with `self ∘ P ∘ self = self`: a left inverse when `self` is
    /// injective, a right inverse (free bits set to zero) when it is surjective.
    pub fn pseudo_inverse(&self) -> Self {
        let e = self.echelon();
        let mut off = 0;
        let ins = self.outs.iter().map(|&(d, b)| {
            // `reduce` splits a unit vector into an image part (with its preimage) and a
            // residue on non-leading bits; dropping the residue keeps the map linear.
            let bs = (off..off + b).map(|k| e.reduce(1 << k).1).collect();
            off += b;
            (d, bs)
        });
        let outs = self.ins.iter().map(|(d, bs)| (*d, bs.len() as u32));
        Self::build(outs, ins.collect::<Vec<_>>())
    }

    /// The inverse of a bijection.
    pub fn inverse(&self) -> Option<Self> {
        self.is_bijective().then(|| self.pseudo_inverse())
    }

    /// Direct product: `self` keeps the low bits of every shared input and output
    /// dimension, `outer` stacks above them (an atom tiled over warps or registers).
    pub fn product(&self, outer: &Layout) -> Self {
        let outs: Outs = {
            let mut o: Outs = self.outs.clone();
            for &(d, b) in &outer.outs {
                match o.iter_mut().find(|x| x.0 == d) {
                    Some(x) => x.1 += b,
                    None => o.push((d, b)),
                }
            }
            o.sort_unstable_by_key(|o| o.0);
            o
        };
        let mut ins: SmallVec<[(Dim, Bases); 4]> = SmallVec::new();
        for (d, bs) in &self.ins {
            ins.push((*d, bs.iter().map(|&v| Self::pack(&outs, self.unpack(v))).collect()));
        }
        for (d, bs) in &outer.ins {
            let lifted = bs
                .iter()
                .map(|&v| Self::pack(&outs, outer.unpack(v).into_iter().map(|(od, x)| (od, x << self.out_bits(od)))));
            match ins.iter_mut().find(|i| i.0 == *d) {
                Some(i) => i.1.extend(lifted),
                None => ins.push((*d, lifted.collect())),
            }
        }
        Self::build(outs, ins)
    }

    /// Only the inputs `ins`, projected onto the outputs `outs`.
    pub fn sublayout(&self, ins: &[Dim], outs: &[Dim]) -> Self {
        let kept = self.map_outs(self.outs.iter().copied().filter(|o| outs.contains(&o.0)), |d, v| {
            outs.contains(&d).then_some((d, v))
        });
        Self::build(kept.outs.clone(), kept.ins.into_iter().filter(|i| ins.contains(&i.0)))
    }

    /// Output `dim` restricted to its bits `bits` (the coordinate `(c >> lo) mod 2^len`).
    pub fn slice(&self, dim: Dim, bits: Range<u32>) -> Self {
        assert!(bits.end <= self.out_bits(dim), "slice {bits:?} of {dim:?} with {} bits", self.out_bits(dim));
        let outs = self.outs.iter().map(|&(d, b)| (d, if d == dim { bits.len() as u32 } else { b }));
        self.map_outs(outs, |d, v| {
            Some((d, if d == dim { ((u64::from(v) >> bits.start) & mask(bits.len() as u32)) as u32 } else { v }))
        })
    }

    /// Every output dim's value at the given input coordinates (absent inputs are 0).
    pub fn apply_dims(&self, ins: &[(Dim, u32)]) -> SmallVec<[(Dim, u32); 3]> {
        let v = ins.iter().fold(0, |acc, &(d, x)| {
            assert!(x < self.in_size(d), "{d:?} = {x} out of {}", self.in_size(d));
            bits_of(x.into()).fold(acc, |a, i| a ^ self.bases(d)[i as usize])
        });
        self.unpack(v)
    }

    /// `(row, col)` at the given input coordinates.
    pub fn apply(&self, ins: &[(Dim, u32)]) -> (u32, u32) {
        let out = self.apply_dims(ins);
        let get = |dim| out.iter().find(|o| o.0 == dim).map_or(0, |o| o.1);
        (get(Dim::Row), get(Dim::Col))
    }

    // ── queries ─────────────────────────────────────────────────────────────────

    pub fn in_bits(&self, dim: Dim) -> u32 {
        self.bases(dim).len() as u32
    }

    pub fn out_bits(&self, dim: Dim) -> u32 {
        Self::offset(&self.outs, dim).map_or(0, |o| o.1)
    }

    pub fn in_size(&self, dim: Dim) -> u32 {
        1 << self.in_bits(dim)
    }

    pub fn out_size(&self, dim: Dim) -> u32 {
        1 << self.out_bits(dim)
    }

    pub fn in_dims(&self) -> impl Iterator<Item = Dim> + '_ {
        self.ins.iter().map(|i| i.0)
    }

    pub fn out_dims(&self) -> impl Iterator<Item = Dim> + '_ {
        self.outs.iter().map(|o| o.0)
    }

    /// The output values input bit `bit` of `dim` maps to.
    pub fn basis(&self, dim: Dim, bit: u32) -> SmallVec<[(Dim, u32); 3]> {
        self.unpack(self.bases(dim)[bit as usize])
    }

    pub fn rank(&self) -> u32 {
        self.echelon().0.len() as u32
    }

    pub fn is_injective(&self) -> bool {
        self.rank() as usize == self.flat().count()
    }

    pub fn is_surjective(&self) -> bool {
        self.rank() == self.outs.iter().map(|o| o.1).sum::<u32>()
    }

    pub fn is_bijective(&self) -> bool {
        self.is_injective() && self.is_surjective()
    }

    /// Mask of the input bits of `dim` that reach no output (replication).
    pub fn free_bits(&self, dim: Dim) -> u32 {
        self.bases(dim).iter().enumerate().filter(|b| *b.1 == 0).map(|b| 1 << b.0).sum()
    }

    /// Every input dimension maps bit-for-bit onto the same output dimension.
    pub fn is_identity(&self) -> bool {
        self.ins.len() == self.outs.len() && self.ins.iter().all(|i| self.passes(i.0))
    }
}

/// An integer grid of `reps = [rows, cols]` atoms above a power-of-two `atom`, so a
/// 16×48 tile is three 16×16 atoms. The repetition index is the high part of the
/// register index (`reg / atom regs`), row-major over the grid.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Tiled {
    pub atom: Layout,
    pub reps: [u32; 2],
}

impl Tiled {
    pub fn shape(&self) -> [u32; 2] {
        [self.atom.out_size(Dim::Row) * self.reps[0], self.atom.out_size(Dim::Col) * self.reps[1]]
    }

    pub fn regs(&self) -> u32 {
        self.atom.in_size(Dim::Reg) * self.reps[0] * self.reps[1]
    }

    /// `(row, col)` within the whole tile.
    pub fn apply(&self, ins: &[(Dim, u32)]) -> (u32, u32) {
        let regs = self.atom.in_size(Dim::Reg);
        let rep = ins.iter().find(|i| i.0 == Dim::Reg).map_or(0, |i| i.1 / regs);
        assert!(rep < self.reps[0] * self.reps[1], "repetition {rep} outside {:?}", self.reps);
        let atom: SmallVec<[(Dim, u32); 4]> =
            ins.iter().map(|&(d, v)| (d, if d == Dim::Reg { v % regs } else { v })).collect();
        let (r, c) = self.atom.apply(&atom);
        (rep / self.reps[1] * self.atom.out_size(Dim::Row) + r, rep % self.reps[1] * self.atom.out_size(Dim::Col) + c)
    }

    /// The same map as one layout, when the grid is a power of two on both axes.
    pub fn as_layout(&self) -> Option<Layout> {
        let [r, c] = self.reps;
        (r.is_power_of_two() && c.is_power_of_two()).then(|| {
            self.atom.product(&Layout::identity(Dim::Reg, c, Dim::Col).product(&Layout::identity(
                Dim::Reg,
                r,
                Dim::Row,
            )))
        })
    }
}
