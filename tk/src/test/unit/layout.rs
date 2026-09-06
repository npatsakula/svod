//! [`LaneMap`] tests: the `mma.sync.m16n8k16` A/B/C lane tables from
//! `fa_cuda_references.md` §(a) verbatim, the lane×register → row×col bijection of
//! every layout/orientation/arch, and the reduce plan (slots, slot assignment,
//! cross-lane tree, folded axis) cross-checked against a brute-force evaluation of
//! the integer map — plus the guarantee that the UOp evaluation of the map folds to
//! the same integers.

use std::collections::{BTreeMap, BTreeSet};

use proptest::prelude::*;
use svod_dtype::{AmdArch, CudaArch, GpuArch};
use svod_ir::UOp;
use test_case::test_case;

use crate::ArchCaps;
use crate::arch::FragRole;
use crate::index::Idx;
use crate::layout::{LaneMap, LdmatrixX4, ReduceTree};
use crate::tiles::{RT_16X16, RT_16X16_MMA, RT_16X16_W32_ACC, RT_16X16_W32_ACC_T, RT_16X16_W32_IN, RTBaseShape};

const SM_86: GpuArch = GpuArch::Cuda(CudaArch::from_compute_capability(8, 6));

/// Every 16×16 matrix-core fragment tk resolves a role to, with its wave size.
/// (The `RT_32X32`/`RT_16X32`/`RT_32X16` constants are unused tinygrad carry-overs
/// whose stride-4 map does not cover a 32-wide fragment; no kernel builds on them.)
fn all_frags() -> Vec<(RTBaseShape, usize, &'static str)> {
    vec![
        (RT_16X16, 64, "gfx942 RT_16X16"),
        (RT_16X16_W32_ACC, 32, "gfx1151 acc"),
        (RT_16X16_W32_IN, 32, "gfx1151 input"),
        (RT_16X16_W32_ACC_T, 32, "gfx1151 acc_t"),
        (RT_16X16_MMA, 32, "sm_86 mma.sync"),
    ]
}

/// The integer `(row, col)` of `(lane, j)` per the fragment map.
fn rc(f: &RTBaseShape, transpose: bool, lane: usize, j: usize) -> (i64, i64) {
    f.map.rc(transpose, &(lane as i64), f.base.rows as i64, f.base.cols as i64, &(j as i64))
}

/// The `mma.sync` A-fragment table (`fa_cuda_references.md` §(a)): register `j` of
/// lane `l` is `(g + 8·((j/2)%2), 2t + j%2 + 8·(j/4))` — the same map is the C
/// fragment per n-half and, read transposed, the B fragments.
#[test_case(0, [(0,0),(0,1),(8,0),(8,1),(0,8),(0,9),(8,8),(8,9)]; "lane 0")]
#[test_case(1, [(0,2),(0,3),(8,2),(8,3),(0,10),(0,11),(8,10),(8,11)]; "lane 1")]
#[test_case(3, [(0,6),(0,7),(8,6),(8,7),(0,14),(0,15),(8,14),(8,15)]; "lane 3")]
#[test_case(4, [(1,0),(1,1),(9,0),(9,1),(1,8),(1,9),(9,8),(9,9)]; "lane 4")]
#[test_case(13, [(3,2),(3,3),(11,2),(11,3),(3,10),(3,11),(11,10),(11,11)]; "lane 13")]
#[test_case(18, [(4,4),(4,5),(12,4),(12,5),(4,12),(4,13),(12,12),(12,13)]; "lane 18")]
#[test_case(27, [(6,6),(6,7),(14,6),(14,7),(6,14),(6,15),(14,14),(14,15)]; "lane 27")]
#[test_case(31, [(7,6),(7,7),(15,6),(15,7),(7,14),(7,15),(15,14),(15,15)]; "lane 31")]
fn mma_sync_a_fragment_rows(lane: usize, expect: [(i64, i64); 8]) {
    let got: Vec<_> = (0..8).map(|j| rc(&RT_16X16_MMA, false, lane, j)).collect();
    assert_eq!(got, expect);
}

/// All 32 lanes of the A table against the closed forms the table was generated
/// from (`row = g + 8·v1, col = 2t + v0 + 8·v2`), and the C table: n-half `h` of a
/// `Row` accumulator is registers `4h..4h+4` = `c0..c3` = `(g, 2t), (g, 2t+1),
/// (g+8, 2t), (g+8, 2t+1)` at columns `8h + ..`.
#[test]
fn mma_sync_a_and_c_tables() {
    for lane in 0..32 {
        let (g, t) = (lane as i64 >> 2, lane as i64 & 3);
        for j in 0..8 {
            let (v0, v1, v2) = (j as i64 % 2, (j as i64 / 2) % 2, j as i64 / 4);
            assert_eq!(rc(&RT_16X16_MMA, false, lane, j), (g + 8 * v1, 2 * t + v0 + 8 * v2), "A lane {lane} a{j}");
        }
        for h in 0..2i64 {
            let c: Vec<_> = (0..4).map(|k| rc(&RT_16X16_MMA, false, lane, (4 * h + k) as usize)).collect();
            let want =
                vec![(g, 2 * t + 8 * h), (g, 2 * t + 1 + 8 * h), (g + 8, 2 * t + 8 * h), (g + 8, 2 * t + 1 + 8 * h)];
            assert_eq!(c, want, "C lane {lane} n-half {h}");
        }
    }
}

/// The B table (`k = 2t + v0 + 8·v1, n = g`, K×N): n-half `h` of a `Col`-read B tile
/// is registers `{2h, 2h+1, 2h+4, 2h+5}` — `b0..b3` at `n = g + 8h`.
#[test_case(0, [(0,0),(1,0),(8,0),(9,0)]; "lane 0")]
#[test_case(1, [(2,0),(3,0),(10,0),(11,0)]; "lane 1")]
#[test_case(7, [(6,1),(7,1),(14,1),(15,1)]; "lane 7")]
#[test_case(12, [(0,3),(1,3),(8,3),(9,3)]; "lane 12")]
#[test_case(22, [(4,5),(5,5),(12,5),(13,5)]; "lane 22")]
#[test_case(31, [(6,7),(7,7),(14,7),(15,7)]; "lane 31")]
fn mma_sync_b_fragment_rows(lane: usize, expect: [(i64, i64); 4]) {
    for h in 0..2i64 {
        let got: Vec<_> =
            [2 * h, 2 * h + 1, 2 * h + 4, 2 * h + 5].map(|j| rc(&RT_16X16_MMA, true, lane, j as usize)).to_vec();
        let want: Vec<_> = expect.iter().map(|&(k, n)| (k, n + 8 * h)).collect();
        assert_eq!(got, want, "B lane {lane} n-half {h}");
    }
}

/// The complete 32-lane B table, all four registers, against the closed form.
#[test]
fn mma_sync_b_table() {
    for lane in 0..32 {
        let (g, t) = (lane as i64 >> 2, lane as i64 & 3);
        for h in 0..2i64 {
            for (b, j) in [2 * h, 2 * h + 1, 2 * h + 4, 2 * h + 5].into_iter().enumerate() {
                let (v0, v1) = (b as i64 % 2, b as i64 / 2);
                assert_eq!(
                    rc(&RT_16X16_MMA, true, lane, j as usize),
                    (2 * t + v0 + 8 * v1, g + 8 * h),
                    "lane {lane} b{b}"
                );
            }
        }
    }
}

/// The legacy AMD maps are unchanged: gfx942 `row = L%16, col = (L/16)·4 + j` (and
/// its transpose), the RDNA even/odd accumulator `row = 2j + L/16, col = L%16`, its
/// transpose, and the replicated input `row = L%16, col = j`.
#[test_case(RT_16X16, false, 37, 2, (5, 10); "gfx942 row")]
#[test_case(RT_16X16, true, 37, 2, (10, 5); "gfx942 col")]
#[test_case(RT_16X16_W32_ACC, false, 21, 3, (7, 5); "rdna acc")]
#[test_case(RT_16X16_W32_ACC, true, 21, 3, (7, 5); "rdna acc ignores transpose")]
#[test_case(RT_16X16_W32_ACC_T, false, 21, 3, (5, 7); "rdna acc_t")]
#[test_case(RT_16X16_W32_IN, false, 21, 9, (5, 9); "rdna input")]
#[test_case(RT_16X16_W32_IN, true, 21, 9, (9, 5); "rdna input col")]
fn amd_maps(f: RTBaseShape, transpose: bool, lane: usize, j: usize, expect: (i64, i64)) {
    assert_eq!(rc(&f, transpose, lane, j), expect);
}

// Every layout is a bijection lane×register → row×col of the fragment (RDNA's
// replicated inputs: a 2-to-1 cover, lanes `L` and `L+16` identical), in both
// orientations, and its UOp evaluation folds to the same integers.
proptest! {
    #[test]
    fn layouts_are_bijections(which in 0usize..5, transpose in any::<bool>()) {
        let (f, wave, name) = all_frags()[which];
        let (rows, cols, ept) = (f.base.rows, f.base.cols, f.base.ept);
        let replication = wave * ept / (rows * cols);
        prop_assert!(replication >= 1, "{name}: ept·wave covers the fragment");
        let mut hits = vec![0usize; rows * cols];
        for lane in 0..wave {
            for j in 0..ept {
                let (r, c) = rc(&f, transpose, lane, j);
                prop_assert!((0..rows as i64).contains(&r) && (0..cols as i64).contains(&c), "{name}: ({lane},{j}) -> ({r},{c})");
                hits[r as usize * cols + c as usize] += 1;
                let lane_u = UOp::index_const(lane as i64);
                let j_u = UOp::index_const(j as i64);
                let (ru, cu) = f.map.rc(transpose, &lane_u, rows as i64, cols as i64, &j_u);
                prop_assert_eq!((super::swizzle::eval_const(&ru), super::swizzle::eval_const(&cu)), (r, c), "{} UOp map", name);
            }
        }
        prop_assert!(hits.iter().all(|&h| h == replication), "{name}: every element held by exactly {replication} (lane, j)");
        if replication > 1 {
            for lane in 0..wave / 2 {
                for j in 0..ept {
                    prop_assert_eq!(rc(&f, transpose, lane, j), rc(&f, transpose, lane + wave / 2, j), "{} replicated halves", name);
                }
            }
        }
    }
}

/// Brute-force reduce plan of a map: the coordinate that varies with `j`
/// (`folds_cols`), the distinct kept coordinates per lane (`slots`, which register
/// feeds which slot), and the set of lanes sharing a kept coordinate (the tree's
/// partner set) — all read off the integer table, then compared with the
/// closed-form [`LaneMap`] answers the reductions build from.
#[test_case(0; "gfx942 RT_16X16")]
#[test_case(1; "gfx1151 acc")]
#[test_case(2; "gfx1151 input")]
#[test_case(3; "gfx1151 acc_t")]
#[test_case(4; "sm_86 mma.sync")]
fn reduce_plan_matches_brute_force(which: usize) {
    let (f, wave, name) = all_frags()[which];
    let (rows, cols, ept) = (f.base.rows, f.base.cols, f.base.ept);
    for transpose in [false, true] {
        let table: Vec<Vec<(i64, i64)>> =
            (0..wave).map(|l| (0..ept).map(|j| rc(&f, transpose, l, j)).collect()).collect();
        let varies = |pick: fn((i64, i64)) -> i64| {
            table.iter().any(|l| l.iter().map(|&e| pick(e)).collect::<BTreeSet<_>>().len() > 1)
        };
        let (rows_vary, cols_vary) = (varies(|e| e.0), varies(|e| e.1));
        let folds_cols = f.map.folds_cols(transpose);
        assert!(if folds_cols { cols_vary } else { rows_vary }, "{name} t={transpose}: the folded axis varies with j");
        // `mma.sync` varies both; the others exactly one.
        assert_eq!(rows_vary && cols_vary, f.map == LaneMap::MmaSync, "{name} t={transpose}: axes varying with j");
        let kept = |e: (i64, i64)| if folds_cols { e.0 } else { e.1 };

        // Slots: distinct kept coordinates per lane, ascending; register j's slot.
        for (lane, regs) in table.iter().enumerate() {
            let distinct: Vec<i64> = regs.iter().map(|&e| kept(e)).collect::<BTreeSet<_>>().into_iter().collect();
            assert_eq!(distinct.len(), f.map.slots(), "{name} t={transpose} lane {lane}: kept values per lane");
            for (j, &e) in regs.iter().enumerate() {
                let want = distinct.iter().position(|&k| k == kept(e)).unwrap() as i64;
                let Idx::Const(slot) = f.map.slot_of(&Idx::Const(j as i64)) else { panic!("constant slot") };
                assert_eq!(slot, want, "{name} t={transpose} lane {lane} j={j}: slot");
            }
        }

        // Partners: lanes holding the same kept coordinate at slot 0, as xor deltas.
        let mut by_kept: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
        for (lane, regs) in table.iter().enumerate() {
            by_kept.entry(kept(regs[0])).or_default().push(lane);
        }
        let deltas: BTreeSet<i64> = by_kept
            .values()
            .flat_map(|lanes| lanes.iter().flat_map(move |&a| lanes.iter().map(move |&b| (a ^ b) as i64)))
            .filter(|&d| d != 0)
            .collect();
        match f.map.tree(wave) {
            ReduceTree::Gather(offsets) => {
                assert_eq!(offsets.iter().copied().collect::<BTreeSet<_>>(), deltas, "{name}: sibling gather offsets");
                assert_eq!(
                    offsets.as_slice(),
                    ArchCaps::for_arch(if wave == 64 {
                        GpuArch::Amd(AmdArch::Gfx942)
                    } else {
                        GpuArch::Amd(AmdArch::Gfx1151)
                    })
                    .reduce_tree()
                    .as_slice(),
                    "{name}: the AMD caps tree"
                );
            }
            ReduceTree::Butterfly(masks) => {
                // The butterfly's masks span the partner group: every delta is a subset-xor of the masks.
                let span: BTreeSet<i64> = (1..1 << masks.len())
                    .map(|bits: usize| {
                        masks
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| bits >> i & 1 == 1)
                            .map(|(_, &m)| m)
                            .fold(0, |a, m| a ^ m)
                    })
                    .collect();
                assert_eq!(span, deltas, "{name}: butterfly span");
            }
        }
        let _ = (rows, cols);
    }
}

/// The per-arch role table resolves onto maps with the slot count the register
/// vectors are allocated with, and `Kernel::rv` follows it.
#[test_case(GpuArch::Amd(AmdArch::Gfx942), 1; "gfx942")]
#[test_case(GpuArch::Amd(AmdArch::Gfx1151), 1; "gfx1151")]
#[test_case(SM_86, 2; "sm_86")]
fn rv_slots_follow_the_accumulator_map(arch: GpuArch, slots: usize) {
    let caps = ArchCaps::for_arch(arch);
    let ker = crate::Kernel::new("rv", [1, 1, 1], caps.wave_size as i64, vec![], caps);
    assert_eq!(caps.frag(FragRole::Accumulator).unwrap().map.slots(), slots);
    assert_eq!(ker.acc_vec(32).shape(), &[2, slots]);
}

/// The `ldmatrix.x4` plan of the `mma.sync` fragment (`fa_cuda_references.md` §(c)):
/// lane `L` addresses row `L % 16`, columns `8·(L/16)..` — matrices TL, BL, TR, BR
/// — so a `Row` read takes the words in order, and a `Col` read (V) needs `.trans`
/// with the ThunderKittens `ldsm4t(tmp[0], tmp[2], tmp[1], tmp[3])` permutation.
/// The AMD maps have no `ldmatrix` form.
#[test_case(RT_16X16_MMA, false, Some(LdmatrixX4 { trans: false, words: [0, 1, 2, 3] }); "mma.sync row")]
#[test_case(RT_16X16_MMA, true, Some(LdmatrixX4 { trans: true, words: [0, 2, 1, 3] }); "mma.sync col is ldsm4t")]
#[test_case(RT_16X16, false, None; "gfx942")]
#[test_case(RT_16X16_W32_IN, false, None; "gfx1151 input")]
#[test_case(RT_16X16_W32_ACC, false, None; "gfx1151 acc")]
fn ldmatrix_x4_plan(f: RTBaseShape, transpose: bool, plan: Option<LdmatrixX4>) {
    assert_eq!(f.map.ldmatrix_x4(transpose), plan);
}

/// Following the plan reproduces the map: register pair `p` of lane `L` is what the
/// hardware hands lane `L` for matrix `words[p]` — `(L/4, 2(L%4) + e)` of the 8×8
/// block, `(2(L%4) + e, L/4)` under `.trans` — at that block's tile offset.
#[test_case(false; "row")]
#[test_case(true; "col")]
fn ldmatrix_x4_plan_reproduces_the_map(transpose: bool) {
    let plan = RT_16X16_MMA.map.ldmatrix_x4(transpose).expect("mma.sync has an ldmatrix form");
    for lane in 0..32i64 {
        let (g, t) = (lane / 4, lane % 4);
        for (p, &m) in plan.words.iter().enumerate() {
            let (rb, cb) = ((m % 2) as i64, (m / 2) as i64); // matrix m = row block + 2·col block
            for e in 0..2i64 {
                let got = rc(&RT_16X16_MMA, transpose, lane as usize, 2 * p + e as usize);
                let want = if plan.trans { (8 * rb + 2 * t + e, 8 * cb + g) } else { (8 * rb + g, 8 * cb + 2 * t + e) };
                assert_eq!(got, want, "lane {lane} pair {p} element {e}");
            }
        }
    }
}
