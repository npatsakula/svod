//! F2 layout tests: the tk1 `LaneMap::rc` closed forms (ported verbatim below) and the
//! `mma.sync`/Apple lane tables against every atom, exhaustively; the reduce butterfly
//! against a brute-force partner search; the algebra laws as properties.

use std::collections::{BTreeMap, BTreeSet};

use proptest::prelude::*;
use svod_dtype::AmdArch;
use test_case::test_case;

use crate::layout::Dim::{self, Col, Lane, Reg, Row, Warp};
use crate::layout::*;

/// tk1 `LaneMap`, the integer half of `rc` copied from `tk/src/layout.rs`.
#[derive(Clone, Copy, Debug)]
enum LaneMap {
    Strided { stride: i64 },
    Interleaved,
    InterleavedT,
    MmaSync,
    SimdgroupMatrix,
    SimdgroupMatrixT,
}

fn lane_map_rc(map: LaneMap, transpose: bool, lane: i64, rows: i64, cols: i64, j: i64) -> (i64, i64) {
    match map {
        LaneMap::Strided { stride } => {
            if transpose {
                ((lane / cols) * stride + j, lane % cols)
            } else {
                (lane % rows, (lane / rows) * stride + j)
            }
        }
        LaneMap::Interleaved => (j * 2 + lane / cols, lane % cols),
        LaneMap::InterleavedT => (lane % cols, j * 2 + lane / cols),
        LaneMap::MmaSync => {
            let (g, t) = (lane / 4, lane % 4);
            let (r, c) = (g + (j / 2) % 2 * 8, t * 2 + j % 2 + j / 4 * 8);
            if transpose { (c, r) } else { (r, c) }
        }
        LaneMap::SimdgroupMatrix | LaneMap::SimdgroupMatrixT => {
            let r = (lane / 16) % 2 * 4 + (lane / 2) % 4;
            let c = (lane / 8) % 2 * 4 + lane % 2 * 2 + j;
            if transpose ^ matches!(map, LaneMap::SimdgroupMatrixT) { (c, r) } else { (r, c) }
        }
    }
}

/// Every tk1 fragment (`all_frags` of the tk1 tests) with its F2 atom:
/// `(name, atom, map, tk1 transpose flag, rows, cols, regs, lanes)`.
type Frag = (&'static str, fn() -> Layout, LaneMap, bool, i64, i64, u32, u32);

fn frags() -> Vec<Frag> {
    vec![
        ("gfx942 RT_16X16", mfma_16x16x16, LaneMap::Strided { stride: 4 }, false, 16, 16, 4, 64),
        ("gfx942 RT_16X16 col", || mfma_16x16x16().transpose(), LaneMap::Strided { stride: 4 }, true, 16, 16, 4, 64),
        ("gfx1151 acc", wmma_gfx11_acc, LaneMap::Interleaved, false, 16, 16, 8, 32),
        ("gfx1151 acc_t", || wmma_gfx11_acc().transpose(), LaneMap::InterleavedT, false, 16, 16, 8, 32),
        ("gfx1151 input", wmma_gfx11_input, LaneMap::Strided { stride: 0 }, false, 16, 16, 16, 32),
        ("gfx1151 input col", || wmma_gfx11_input().transpose(), LaneMap::Strided { stride: 0 }, true, 16, 16, 16, 32),
        ("gfx1201 wmma", wmma_gfx12, LaneMap::Strided { stride: 8 }, false, 16, 16, 8, 32),
        ("gfx1201 wmma col", || wmma_gfx12().transpose(), LaneMap::Strided { stride: 8 }, true, 16, 16, 8, 32),
        ("sm_86 mma.sync", mma_sync_16x16, LaneMap::MmaSync, false, 16, 16, 8, 32),
        ("sm_86 mma.sync col", || mma_sync_16x16().transpose(), LaneMap::MmaSync, true, 16, 16, 8, 32),
        ("apple simdgroup", simdgroup_8x8, LaneMap::SimdgroupMatrix, false, 8, 8, 2, 32),
        ("apple simdgroup T", || simdgroup_8x8().transpose(), LaneMap::SimdgroupMatrixT, false, 8, 8, 2, 32),
    ]
}

/// Every input of `l`: each assignment of its input dims.
fn all_inputs(l: &Layout) -> Vec<Vec<(Dim, u32)>> {
    let dims = [Reg, Lane, Warp, Dim::Block, Row, Col].into_iter().filter(|&d| l.in_bits(d) > 0);
    dims.fold(vec![vec![]], |acc, d| {
        acc.into_iter().flat_map(|v| (0..l.in_size(d)).map(move |x| [v.clone(), vec![(d, x)]].concat())).collect()
    })
}

/// How many inputs of `l` hold each `(row, col)` they reach.
fn holders(l: &Layout) -> BTreeMap<(u32, u32), usize> {
    let mut held = BTreeMap::new();
    for x in all_inputs(l) {
        *held.entry(l.apply(&x)).or_insert(0) += 1;
    }
    held
}

/// Whether `l` holds every element of its tile.
pub(super) fn covers(l: &Layout) -> bool {
    holders(l).len() == (l.out_size(Row) * l.out_size(Col)) as usize
}

/// Whether `l` holds every element of its tile exactly once.
fn bijective(l: &Layout) -> bool {
    covers(l) && holders(l).values().all(|&n| n == 1)
}

/// Mask of the input bits of `dim` that reach no output (replication).
fn free_bits(l: &Layout, dim: Dim) -> u32 {
    (0..l.in_bits(dim)).filter(|&b| l.basis(dim, b).iter().all(|o| o.1 == 0)).map(|b| 1 << b).sum()
}

/// Whether every lane of `dst` finds each of its elements in the same lane of `src`.
fn in_lane(src: &Layout, dst: &Layout) -> bool {
    (0..dst.in_size(Lane)).all(|lane| {
        (0..dst.in_size(Reg)).all(|j| {
            let want = dst.apply(&[(Lane, lane), (Reg, j)]);
            (0..src.in_size(Reg)).any(|k| src.apply(&[(Lane, lane), (Reg, k)]) == want)
        })
    })
}

fn at(l: &Layout, lane: u32, j: u32) -> (i64, i64) {
    let (r, c) = l.apply(&[(Lane, lane), (Reg, j)]);
    (r.into(), c.into())
}

/// Each atom equals its tk1 closed form on every (lane, register) and covers the tile
/// exactly `replication` times (RDNA's replicated input: lanes `L`, `L+16` identical,
/// lane bit 4 free).
#[test]
fn atoms_match_lane_map_closed_forms() {
    for (name, atom, map, transpose, rows, cols, regs, lanes) in frags() {
        let l = atom();
        assert_eq!((l.in_size(Reg), l.in_size(Lane)), (regs, lanes), "{name}: input sizes");
        assert_eq!((l.out_size(Row), l.out_size(Col)), (rows as u32, cols as u32), "{name}: tile");
        let mut hits = vec![0u32; (rows * cols) as usize];
        for lane in 0..lanes {
            for j in 0..regs {
                let want = lane_map_rc(map, transpose, lane.into(), rows, cols, j.into());
                assert_eq!(at(&l, lane, j), want, "{name}: lane {lane} reg {j}");
                hits[(want.0 * cols + want.1) as usize] += 1;
            }
        }
        let replication = regs * lanes / (rows * cols) as u32;
        assert!(hits.iter().all(|&h| h == replication), "{name}: every element held {replication} times");
        let free = if replication == 2 { 16 } else { 0 };
        assert_eq!((free_bits(&l, Lane), free_bits(&l, Reg)), (free, 0), "{name}: free bits");
    }
}

/// The 16×16 tile is the C fragment tiled over one more register bit (n-half `h` =
/// registers `4h..4h+4`), and B and C are bijections; where each places its elements
/// is `atoms::atom_layouts_match_the_isa`'s PTX table.
#[test]
fn mma_sync_tile_is_c_over_the_n_halves() {
    let (b, c) = (mma_sync_b(), mma_sync_c());
    for lane in 0..32u32 {
        for h in 0..2 {
            for k in 0..4 {
                let (r, cc) = at(&c, lane, k);
                assert_eq!(at(&mma_sync_16x16(), lane, 4 * h + k), (r, cc + 8 * i64::from(h)), "tile lane {lane}");
            }
        }
    }
    assert!(bijective(&b) && bijective(&c));
}

/// The transposed tile holds the two B fragments in registers `{0,1,4,5}` / `{2,3,6,7}`:
/// a register permutation of B tiled over the n-halves.
#[test]
fn mma_sync_transposed_tile_is_two_b_fragments() {
    let (halves, tile) = (mma_sync_b().product(&Layout::identity(Reg, 2, Col)), mma_sync_16x16().transpose());
    for lane in 0..32 {
        for (j, from) in [0, 1, 4, 5, 2, 3, 6, 7].into_iter().enumerate() {
            assert_eq!(at(&halves, lane, j as u32), at(&tile, lane, from), "lane {lane} register {j}");
        }
    }
}

/// MFMA 32×32×8: A `row = L%32, k = 4(L/32) + j`, B its transpose, C/D `row = 8(j/4) +
/// 4(L/32) + j%4, col = L%32` (CK `xdlops_gemm.hpp` unmerge `groups(4) × blks(2) ×
/// group_size(4)`, `blk = L/32`) — all bijections.
#[test]
fn mfma_32x32x8_tables() {
    let (a, c) = (mfma_32x32x8_a(), mfma_32x32x8_c());
    assert_eq!((a.in_size(Reg), a.in_size(Lane), a.out_size(Row), a.out_size(Col)), (4, 64, 32, 8));
    assert_eq!((c.in_size(Reg), c.in_size(Lane), c.out_size(Row), c.out_size(Col)), (16, 64, 32, 32));
    for lane in 0..64u32 {
        let l = i64::from(lane);
        for j in 0..4 {
            assert_eq!(at(&a, lane, j), (l % 32, 4 * (l / 32) + i64::from(j)), "A lane {lane} reg {j}");
            assert_eq!(at(&a.transpose(), lane, j), (4 * (l / 32) + i64::from(j), l % 32), "B lane {lane} reg {j}");
        }
        for j in 0..16 {
            let j = i64::from(j);
            assert_eq!(at(&c, lane, j as u32), (8 * (j / 4) + 4 * (l / 32) + j % 4, l % 32), "C lane {lane} reg {j}");
        }
    }
    assert!(bijective(&a) && bijective(&c));
}

/// The Apple lane table measured on hardware (Apple9): the source of truth the
/// closed form was pinned from.
#[test_case(0, [(0, 0), (0, 1)]; "lane 0")]
#[test_case(1, [(0, 2), (0, 3)]; "lane 1")]
#[test_case(2, [(1, 0), (1, 1)]; "lane 2")]
#[test_case(7, [(3, 2), (3, 3)]; "lane 7")]
#[test_case(8, [(0, 4), (0, 5)]; "lane 8")]
#[test_case(15, [(3, 6), (3, 7)]; "lane 15")]
#[test_case(16, [(4, 0), (4, 1)]; "lane 16")]
#[test_case(23, [(7, 2), (7, 3)]; "lane 23")]
#[test_case(24, [(4, 4), (4, 5)]; "lane 24")]
#[test_case(31, [(7, 6), (7, 7)]; "lane 31")]
fn simdgroup_matrix_lane_table(lane: u32, want: [(i64, i64); 2]) {
    for (j, expected) in want.into_iter().enumerate() {
        assert_eq!(at(&simdgroup_8x8(), lane, j as u32), expected, "lane {lane} element {j}");
    }
}

/// The derived row-fold butterfly of every atom equals a brute-force partner search:
/// lanes holding the same row at register 0, as xor deltas, are exactly the span of
/// the masks. Spot values: tk1's `ReduceTree` (`mma.sync` `[1, 2]`, Apple `[1, 8]`,
/// gfx942 siblings `16, 32, 48`).
#[test]
fn lanes_sharing_row_matches_brute_force() {
    for (name, atom, ..) in frags() {
        let l = atom();
        let mut by_row: BTreeMap<i64, Vec<u32>> = BTreeMap::new();
        for lane in 0..l.in_size(Lane) {
            by_row.entry(at(&l, lane, 0).0).or_default().push(lane);
        }
        let deltas: BTreeSet<u32> =
            by_row.values().flat_map(|ls| ls.iter().flat_map(move |&a| ls.iter().map(move |&b| a ^ b))).collect();
        let masks = lanes_sharing_row(&l);
        let span: BTreeSet<u32> = (0..1u32 << masks.len())
            .map(|s| (0..masks.len()).filter(|i| s >> i & 1 == 1).fold(0, |a, i| a ^ masks[i]))
            .collect();
        assert_eq!(span, deltas, "{name}: butterfly span");
    }
    assert_eq!(lanes_sharing_row(&mma_sync_16x16()).as_slice(), [1, 2]);
    assert_eq!(lanes_sharing_row(&simdgroup_8x8()).as_slice(), [1, 8]);
    assert_eq!(lanes_sharing_row(&mfma_16x16x16()).as_slice(), [16, 32]);
    assert_eq!(lanes_sharing_row(&mfma_32x32x8_c()).as_slice(), [1, 2, 4, 8, 16]);
}

/// `ldmatrix.x4` (`fa_cuda_references.md` §(c)) as a composition: the 8×8 per-matrix
/// pair layout tiled over the four matrices reproduces the `mma.sync` tile plain, its
/// transpose under `.trans` (ThunderKittens `ldsm4t(tmp[0], tmp[2], tmp[1], tmp[3])`),
/// and the matrices land where the addressing lanes `8m..8m+8` point.
/// A wave32 `global_load_tr_b128` transposes the 8×8 block each group of eight
/// lanes addresses (lane `k` of the group receives element `k` of every row);
/// with the blocks addressed in TL, TR, BL, BR order lane `L` holds column
/// `L % 16` at rows `8·(L/16)..+8`, the RDNA4 WMMA B fragment.
#[test]
fn global_tr_b128_is_the_transposed_block() {
    let (l, rows) = (global_tr_b128(), global_tr_rows());
    assert_eq!(l, strided(8, 32).transpose());
    for lane in 0..32u32 {
        let (half, within) = (i64::from(lane / 16), i64::from(lane % 16));
        for j in 0..8u32 {
            let (r, c) = rows.apply(&[(Lane, 8 * (lane / 8) + j)]);
            assert_eq!(at(&l, lane, j), (i64::from(r), i64::from(c) + i64::from(lane % 8)), "lane {lane} register {j}");
            assert_eq!(at(&l, lane, j), (8 * half + i64::from(j), within));
        }
    }
}

#[test]
fn ldmatrix_x4_is_the_mma_sync_tile() {
    assert_eq!(ldmatrix_x4(false), mma_sync_16x16());
    assert_eq!(ldmatrix_x4(true), mma_sync_16x16().transpose());
    let rows = ldmatrix_x4_rows();
    for (trans, words) in [(false, [0, 1, 2, 3]), (true, [0, 2, 1, 3])] {
        let l = ldmatrix_x4(trans);
        for lane in 0..32u32 {
            let (g, t) = (i64::from(lane / 4), i64::from(lane % 4));
            for (p, &m) in words.iter().enumerate() {
                let (r0, c0) = rows.apply(&[(Lane, 8 * m)]);
                for e in 0..2 {
                    let want = if trans { (2 * t + e, g) } else { (g, 2 * t + e) };
                    let got = at(&l, lane, 2 * p as u32 + e as u32);
                    assert_eq!(
                        got,
                        (i64::from(r0) + want.0, i64::from(c0) + want.1),
                        "trans {trans} lane {lane} pair {p}"
                    );
                }
            }
        }
    }
}

// ── algebra laws ────────────────────────────────────────────────────────────

/// A random linear map `ins → (o0, o1)`.
fn arb(ins: &'static [(Dim, u32)], outs: [(Dim, u32); 2]) -> impl Strategy<Value = Layout> {
    let n: u32 = ins.iter().map(|i| i.1.trailing_zeros()).sum();
    prop::collection::vec((any::<u32>(), any::<u32>()), n as usize).prop_map(move |vals| {
        let vals: Vec<[u32; 2]> = vals.into_iter().map(|(a, b)| [a % outs[0].1, b % outs[1].1]).collect();
        let mut rest = vals.as_slice();
        let bases: Vec<(Dim, &[[u32; 2]])> = ins
            .iter()
            .map(|&(d, s)| {
                let (head, tail) = rest.split_at(s.trailing_zeros() as usize);
                rest = tail;
                (d, head)
            })
            .collect();
        Layout::from_bases(outs, &bases)
    })
}

const HW: &[(Dim, u32)] = &[(Reg, 8), (Lane, 32)];
const TILE: [(Dim, u32); 2] = [(Row, 16), (Col, 16)];

proptest! {
    #[test]
    fn apply_is_linear(a in arb(HW, TILE), x in (0u32..32, 0u32..8), y in (0u32..32, 0u32..8)) {
        let f = |(l, j)| a.apply(&[(Lane, l), (Reg, j)]);
        let (fx, fy, fxy) = (f(x), f(y), f((x.0 ^ y.0, x.1 ^ y.1)));
        prop_assert_eq!(fxy, (fx.0 ^ fy.0, fx.1 ^ fy.1));
    }

    #[test]
    fn product_stacks(a in arb(HW, TILE), b in arb(&[(Reg, 2), (Warp, 4)], [(Row, 4), (Col, 2)]), x in (0u32..32, 0u32..8), y in (0u32..2, 0u32..4)) {
        let p = a.product(&b);
        prop_assert_eq!((p.in_size(Reg), p.in_size(Lane), p.in_size(Warp)), (16, 32, 4));
        prop_assert_eq!((p.out_size(Row), p.out_size(Col)), (64, 32));
        let (r0, c0) = a.apply(&[(Lane, x.0), (Reg, x.1)]);
        let (r1, c1) = b.apply(&[(Reg, y.0), (Warp, y.1)]);
        prop_assert_eq!(p.apply(&[(Lane, x.0), (Reg, x.1 + 8 * y.0), (Warp, y.1)]), (r0 + 16 * r1, c0 + 16 * c1));
        prop_assert_eq!(p.sublayout(&[Lane], &[Row, Col]).apply(&[(Lane, x.0)]), a.apply(&[(Lane, x.0)]));
    }

    #[test]
    fn transpose_swaps(a in arb(HW, TILE), lane in 0u32..32, j in 0u32..8) {
        let (r, c) = a.apply(&[(Lane, lane), (Reg, j)]);
        prop_assert_eq!(a.transpose().apply(&[(Lane, lane), (Reg, j)]), (c, r));
        prop_assert_eq!(&a.transpose().transpose(), &a);
    }

}

/// The accumulator reaches the next product's B operand for free exactly where tk1's
/// `ArchCaps::acc_reusable_as_input` held — gfx12 and CDNA, hardware-verified there —
/// so the F2 algebra subsumes that flag (`tk3_design.md` §3.2). Its A operand never
/// does, and that is the handoff `kernels::attention` asks for: it accumulates the
/// scores as `[bq, bkv]` and feeds them as A, where tk1 accumulates the transpose and
/// feeds it as B.
#[test_case(AmdArch::Gfx1201, true; "gfx1201 rdna4")]
#[test_case(AmdArch::Gfx942, true; "gfx942 cdna3")]
#[test_case(AmdArch::Gfx1151, false; "gfx1151 rdna3.5")]
#[test_case(AmdArch::Gfx1100, false; "gfx1100 rdna3")]
fn amd_accumulator_reuse(arch: AmdArch, reusable: bool) {
    let target = crate::atoms::Target::for_arch(svod_dtype::GpuArch::Amd(arch));
    let atom = target.mma.first().expect("a matrix core");
    assert_eq!(atom.c == atom.b, reusable, "accumulator → B");
    assert!(!in_lane(&atom.c, &atom.a), "accumulator → A crosses lanes");
}
