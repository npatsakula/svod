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

fn at(l: &Layout, lane: u32, j: u32) -> (i64, i64) {
    let (r, c) = l.apply(&[(Lane, lane), (Reg, j)]);
    (r.into(), c.into())
}

/// Each atom equals its tk1 closed form on every (lane, register), covers the tile
/// exactly `replication` times (RDNA's replicated input: lanes `L`, `L+16` identical,
/// lane bit 4 free), and the F2 predicates agree with the brute-force count.
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
        assert!(l.is_surjective(), "{name}: covers the tile");
        assert_eq!(l.is_bijective(), replication == 1, "{name}: bijective");
        let free = if replication == 2 { 16 } else { 0 };
        assert_eq!((l.free_bits(Lane), l.free_bits(Reg)), (free, 0), "{name}: free bits");
    }
}

/// The `mma.sync` A-fragment table (`fa_cuda_references.md` §(a)).
#[test_case(0, [(0,0),(0,1),(8,0),(8,1),(0,8),(0,9),(8,8),(8,9)]; "lane 0")]
#[test_case(1, [(0,2),(0,3),(8,2),(8,3),(0,10),(0,11),(8,10),(8,11)]; "lane 1")]
#[test_case(3, [(0,6),(0,7),(8,6),(8,7),(0,14),(0,15),(8,14),(8,15)]; "lane 3")]
#[test_case(4, [(1,0),(1,1),(9,0),(9,1),(1,8),(1,9),(9,8),(9,9)]; "lane 4")]
#[test_case(13, [(3,2),(3,3),(11,2),(11,3),(3,10),(3,11),(11,10),(11,11)]; "lane 13")]
#[test_case(18, [(4,4),(4,5),(12,4),(12,5),(4,12),(4,13),(12,12),(12,13)]; "lane 18")]
#[test_case(27, [(6,6),(6,7),(14,6),(14,7),(6,14),(6,15),(14,14),(14,15)]; "lane 27")]
#[test_case(31, [(7,6),(7,7),(15,6),(15,7),(7,14),(7,15),(15,14),(15,15)]; "lane 31")]
fn mma_sync_a_fragment_rows(lane: u32, expect: [(i64, i64); 8]) {
    let got: Vec<_> = (0..8).map(|j| at(&mma_sync_a(), lane, j)).collect();
    assert_eq!(got, expect);
}

/// C: `c0..c3 = (g, 2t), (g, 2t+1), (g+8, 2t), (g+8, 2t+1)`; B: `b0..b3 = k (2t, 2t+1,
/// 2t+8, 2t+9) at n = g` — over all 32 lanes, and the 16×16 tile is the C fragment
/// tiled over one more register bit (n-half `h` = registers `4h..4h+4`).
#[test]
fn mma_sync_b_and_c_tables() {
    let (b, c) = (mma_sync_b(), mma_sync_c());
    for lane in 0..32u32 {
        let (g, t) = (i64::from(lane >> 2), i64::from(lane & 3));
        let cs: Vec<_> = (0..4).map(|k| at(&c, lane, k)).collect();
        assert_eq!(cs, [(g, 2 * t), (g, 2 * t + 1), (g + 8, 2 * t), (g + 8, 2 * t + 1)], "C lane {lane}");
        let bs: Vec<_> = (0..4).map(|k| at(&b, lane, k)).collect();
        assert_eq!(bs, [(2 * t, g), (2 * t + 1, g), (2 * t + 8, g), (2 * t + 9, g)], "B lane {lane}");
        for h in 0..2 {
            for k in 0..4 {
                let (r, cc) = at(&c, lane, k);
                assert_eq!(at(&mma_sync_16x16(), lane, 4 * h + k), (r, cc + 8 * i64::from(h)), "tile lane {lane}");
            }
        }
    }
    assert!(b.is_bijective() && c.is_bijective());
}

/// The transposed tile holds the two B fragments in registers `{0,1,4,5}` / `{2,3,6,7}`:
/// a register permutation of B tiled over the n-halves, and the conversion says so.
#[test]
fn mma_sync_transposed_tile_is_two_b_fragments() {
    let halves = mma_sync_b().product(&Layout::identity(Reg, 2, Col));
    let swap_reg_bits = Layout::from_bases(
        [(Reg, 8), (Lane, 32)],
        &[(Reg, &[[1, 0], [4, 0], [2, 0]]), (Lane, &[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]])],
    );
    assert_eq!(halves.compose(&swap_reg_bits), mma_sync_16x16().transpose());
    assert_eq!(
        Conversion::between(&mma_sync_16x16().transpose(), &halves),
        Conversion::RegPermute([0, 1, 4, 5, 2, 3, 6, 7].into_iter().collect())
    );
}

/// tk1's `amd_maps` and `gfx12_map` spot values.
#[test_case(mfma_16x16x16(), 37, 2, (5, 10); "gfx942 row")]
#[test_case(mfma_16x16x16().transpose(), 37, 2, (10, 5); "gfx942 col")]
#[test_case(wmma_gfx11_acc(), 21, 3, (7, 5); "rdna acc")]
#[test_case(wmma_gfx11_acc().transpose(), 21, 3, (5, 7); "rdna acc_t")]
#[test_case(wmma_gfx11_input(), 21, 9, (5, 9); "rdna input")]
#[test_case(wmma_gfx11_input().transpose(), 21, 9, (9, 5); "rdna input col")]
#[test_case(wmma_gfx12(), 5, 0, (5, 0); "gfx12 operand lane 5 reg 0")]
#[test_case(wmma_gfx12(), 21, 3, (5, 11); "gfx12 upper wave-half is the second K run")]
#[test_case(wmma_gfx12().transpose(), 21, 3, (11, 5); "gfx12 col accumulator is the transpose")]
#[test_case(wmma_gfx12(), 31, 7, (15, 15); "gfx12 last lane, last register")]
fn amd_maps(l: Layout, lane: u32, j: u32, expect: (i64, i64)) {
    assert_eq!(at(&l, lane, j), expect);
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
    assert!(a.is_bijective() && c.is_bijective());
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

/// CuTe `Swizzle<B, M, S>` equals its integer formula on every offset, and is an
/// involutive bijection.
#[test_case(8, 64, 3, 3, 3; "128B swizzle of a 8x64 bf16 tile")]
#[test_case(16, 64, 2, 3, 3; "64B")]
#[test_case(32, 32, 3, 0, 5; "element-wise 32x32")]
#[test_case(16, 16, 1, 3, 4; "16x16")]
fn swizzle_matches_cute(rows: u32, cols: u32, b: u32, m: u32, s: u32) {
    let sw = Layout::swizzle_xor(rows, cols, b, m, s);
    for o in 0..rows * cols {
        let want = o ^ ((o >> s) & (((1 << b) - 1) << m));
        let (r, c) = sw.apply(&[(Row, o / cols), (Col, o % cols)]);
        assert_eq!(r * cols + c, want, "offset {o}");
    }
    assert!(sw.is_bijective());
    assert_eq!(sw.inverse(), Some(sw.clone()));
    assert!(sw.compose(&sw).is_identity());
}

#[test]
fn conversions() {
    // The `mma.sync` accumulator is the A operand: tk1's `acc_reusable_as_input`.
    assert_eq!(Conversion::between(&mma_sync_16x16(), &mma_sync_a()), Conversion::Identity);
    // gfx11 accumulator (N-major) → replicated input: lane `L` register `j` reads lane
    // `L%16 + 16·(j%2)`, register `j/2` — a warp shuffle, not tk1's LDS round trip.
    let Conversion::LaneShuffle(plan) = Conversion::between(&wmma_gfx11_acc().transpose(), &wmma_gfx11_input()) else {
        panic!("gfx11 acc → input is a lane shuffle")
    };
    for lane in 0..32 {
        for j in 0..16 {
            assert_eq!(plan.source(0, lane, j), (lane % 16 + 16 * (j % 2), j / 2), "lane {lane} reg {j}");
        }
    }
    assert_eq!(plan.rounds(0).len(), 16);
    // A register-level transpose crosses lanes.
    assert!(matches!(Conversion::between(&wmma_gfx12().transpose(), &wmma_gfx12()), Conversion::LaneShuffle(_)));
    // A 2×2 warp grid placed row-major vs column-major: data crosses warps.
    let grid = |w: &[[u32; 2]]| mma_sync_16x16().product(&Layout::from_bases([(Row, 2), (Col, 2)], &[(Warp, w)]));
    assert_eq!(Conversion::between(&grid(&[[1, 0], [0, 1]]), &grid(&[[0, 1], [1, 0]])), Conversion::ViaSmem);
    // Same warp placement, lane-level relayout inside each warp.
    let tiled = |l: Layout| l.product(&Layout::identity(Warp, 4, Row));
    assert!(matches!(
        Conversion::between(&tiled(wmma_gfx11_acc().transpose()), &tiled(wmma_gfx11_input())),
        Conversion::LaneShuffle(_)
    ));
}

/// 16×48 = three `mma.sync` atoms: the grid is a bijection of 24 registers × 32 lanes
/// onto the tile, with no single-layout form; a power-of-two grid equals the product.
#[test]
fn tiled_grid() {
    let t = Tiled { atom: mma_sync_16x16(), reps: [1, 3] };
    assert_eq!((t.shape(), t.regs(), t.as_layout()), ([16, 48], 24, None));
    let mut seen = BTreeSet::new();
    for lane in 0..32 {
        for j in 0..24 {
            let (r, c) = t.apply(&[(Lane, lane), (Reg, j)]);
            assert!(r < 16 && c < 48);
            assert!(seen.insert((r, c)), "({r}, {c}) held twice");
            assert_eq!(t.apply(&[(Lane, lane), (Reg, j % 8)]), (r, c - 16 * (j / 8)));
        }
    }
    let t = Tiled { atom: mfma_16x16x16(), reps: [2, 4] };
    let l = t.as_layout().expect("power-of-two grid");
    assert_eq!((l.out_size(Row), l.out_size(Col), l.in_size(Reg)), (32, 64, 32));
    for lane in 0..64 {
        for j in 0..t.regs() {
            assert_eq!(t.apply(&[(Lane, lane), (Reg, j)]), l.apply(&[(Lane, lane), (Reg, j)]));
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
const HW_WARPS: &[(Dim, u32)] = &[(Reg, 4), (Lane, 8), (Warp, 2)];
const TILE: [(Dim, u32); 2] = [(Row, 16), (Col, 16)];
const TILE_IN: &[(Dim, u32)] = &[(Row, 16), (Col, 16)];

fn all_inputs(l: &Layout) -> Vec<Vec<(Dim, u32)>> {
    l.in_dims().fold(vec![vec![]], |acc, d| {
        acc.into_iter().flat_map(|v| (0..l.in_size(d)).map(move |x| [v.clone(), vec![(d, x)]].concat())).collect()
    })
}

proptest! {
    #[test]
    fn compose_is_associative(
        a in arb(HW, TILE),
        b in arb(TILE_IN, [(Row, 8), (Col, 32)]),
        c in arb(&[(Row, 8), (Col, 32)], [(Lane, 64), (Warp, 4)]),
    ) {
        prop_assert_eq!(c.compose(&b).compose(&a), c.compose(&b.compose(&a)));
    }

    #[test]
    fn compose_applies_in_sequence(a in arb(HW, TILE), b in arb(TILE_IN, TILE), lane in 0u32..32, j in 0u32..8) {
        let (r, c) = a.apply(&[(Lane, lane), (Reg, j)]);
        prop_assert_eq!(b.compose(&a).apply(&[(Lane, lane), (Reg, j)]), b.apply(&[(Row, r), (Col, c)]));
    }

    #[test]
    fn apply_is_linear(a in arb(HW, TILE), x in (0u32..32, 0u32..8), y in (0u32..32, 0u32..8)) {
        let f = |(l, j)| a.apply(&[(Lane, l), (Reg, j)]);
        let (fx, fy, fxy) = (f(x), f(y), f((x.0 ^ y.0, x.1 ^ y.1)));
        prop_assert_eq!(fxy, (fx.0 ^ fy.0, fx.1 ^ fy.1));
    }

    #[test]
    fn pseudo_inverse_laws(a in arb(HW_WARPS, TILE)) {
        let p = a.pseudo_inverse();
        prop_assert_eq!(&a.compose(&p).compose(&a), &a);
        prop_assert_eq!(a.is_injective(), p.compose(&a).is_identity());
        prop_assert_eq!(a.is_surjective(), a.compose(&p).is_identity());
        prop_assert_eq!(a.inverse().is_some(), a.is_bijective());
        prop_assert!(a.free_bits(Reg).count_ones() + a.free_bits(Lane).count_ones() + a.free_bits(Warp).count_ones() <= 6 - a.rank());
    }

    #[test]
    fn bijections_invert(a in arb(HW, TILE).prop_filter("bijective", Layout::is_bijective)) {
        let inv = a.inverse().expect("bijective");
        prop_assert!(inv.compose(&a).is_identity());
        prop_assert!(a.compose(&inv).is_identity());
        prop_assert_eq!(inv.inverse(), Some(a));
    }

    #[test]
    fn product_stacks(a in arb(HW, TILE), b in arb(&[(Reg, 2), (Warp, 4)], [(Row, 4), (Col, 2)]), x in (0u32..32, 0u32..8), y in (0u32..2, 0u32..4)) {
        let p = a.product(&b);
        prop_assert_eq!((p.in_size(Reg), p.in_size(Lane), p.in_size(Warp)), (16, 32, 4));
        prop_assert_eq!((p.out_size(Row), p.out_size(Col)), (64, 32));
        let (r0, c0) = a.apply(&[(Lane, x.0), (Reg, x.1)]);
        let (r1, c1) = b.apply(&[(Reg, y.0), (Warp, y.1)]);
        prop_assert_eq!(p.apply(&[(Lane, x.0), (Reg, x.1 + 8 * y.0), (Warp, y.1)]), (r0 + 16 * r1, c0 + 16 * c1));
        prop_assert_eq!(p.sublayout(&[Reg, Lane], &[Row, Col]).slice(Row, 0..4).slice(Col, 0..4).sublayout(&[Lane], &[Row, Col]), a.sublayout(&[Lane], &[Row, Col]));
    }

    #[test]
    fn transpose_and_slice(a in arb(HW, TILE), lane in 0u32..32, j in 0u32..8) {
        let (r, c) = a.apply(&[(Lane, lane), (Reg, j)]);
        prop_assert_eq!(a.transpose().apply(&[(Lane, lane), (Reg, j)]), (c, r));
        prop_assert_eq!(&a.transpose().transpose(), &a);
        prop_assert_eq!(a.slice(Row, 1..3).apply(&[(Lane, lane), (Reg, j)]), ((r >> 1) & 3, c));
    }

    #[test]
    fn conversion_plans_reproduce_the_destination(
        src in arb(HW_WARPS, [(Row, 8), (Col, 8)]).prop_filter("bijective", Layout::is_bijective),
        dst in arb(HW_WARPS, [(Row, 8), (Col, 8)]).prop_filter("bijective", Layout::is_bijective),
    ) {
        // Brute force: where each destination element lives in the source.
        let holder: BTreeMap<(u32, u32), Vec<(Dim, u32)>> = all_inputs(&src).into_iter().map(|x| (src.apply(&x), x)).collect();
        let moves = |d: Dim| all_inputs(&dst).iter().any(|x| {
            let get = |v: &[(Dim, u32)]| v.iter().find(|e| e.0 == d).map_or(0, |e| e.1);
            get(&holder[&dst.apply(x)]) != get(x)
        });
        let conv = Conversion::between(&src, &dst);
        prop_assert_eq!(moves(Warp), conv == Conversion::ViaSmem);
        // A lane-dependent register select moves nothing across lanes yet is no static
        // permutation, so only this direction holds; the arms below check the rest.
        if moves(Lane) {
            prop_assert!(matches!(conv, Conversion::LaneShuffle(_) | Conversion::ViaSmem));
        }
        match conv {
            Conversion::Identity => prop_assert_eq!(&src, &dst),
            Conversion::RegPermute(perm) => for x in all_inputs(&dst) {
                let mut moved = x.clone();
                moved.iter_mut().filter(|e| e.0 == Reg).for_each(|e| e.1 = perm[e.1 as usize]);
                prop_assert_eq!(src.apply(&moved), dst.apply(&x));
            },
            Conversion::LaneShuffle(plan) => for x in all_inputs(&dst) {
                let warp: Vec<_> = x.iter().copied().filter(|e| e.0 == Warp).collect();
                let get = |d| x.iter().find(|e| e.0 == d).map_or(0, |e| e.1);
                let (l, j) = plan.source(get(Warp), get(Lane), get(Reg));
                prop_assert_eq!(src.apply(&[[(Lane, l), (Reg, j)].as_slice(), &warp].concat()), dst.apply(&x));
            },
            Conversion::ViaSmem => {}
        }
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
    assert_eq!(Conversion::between(&atom.c, &atom.b) == Conversion::Identity, reusable, "accumulator → B");
    assert!(matches!(Conversion::between(&atom.c, &atom.a), Conversion::LaneShuffle(_)), "accumulator → A");
}
