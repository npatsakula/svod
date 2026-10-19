//! F2 linear layouts (Triton "Linear Layouts", arXiv 2505.23819): a layout is a linear
//! map over GF(2) from the bits of named input dimensions to the bits of named output
//! dimensions, stored as one basis vector per input bit. Output values are packed into
//! a `u64`, the output dimensions in [`Dim`] order, each at the offset of the ones
//! before it. Design: `tk3_design.md` §3.2.

mod atoms;

use smallvec::SmallVec;

pub use atoms::*;

/// A named dimension: the hardware inputs `Reg`/`Lane`/`Warp`/`Block`, and the tile
/// coordinates `Row`/`Col`. Any dimension may be an input or an output.
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

    /// The output values input bit `bit` of `dim` maps to.
    pub fn basis(&self, dim: Dim, bit: u32) -> SmallVec<[(Dim, u32); 3]> {
        self.unpack(self.bases(dim)[bit as usize])
    }
}

/// The lane xor-masks whose butterfly folds a row: a basis (reduced, ascending) of the
/// lane deltas that keep the `Row` coordinate (the kernel of `Lane → Row`).
pub fn lanes_sharing_row(layout: &Layout) -> SmallVec<[u32; 6]> {
    let row = layout.sublayout(&[Dim::Lane], &[Dim::Row]);
    let mut image = Echelon::default();
    let kernel: SmallVec<[u64; 6]> =
        row.bases(Dim::Lane).iter().enumerate().filter_map(|(i, &v)| image.insert(v, 1 << i)).collect();
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
