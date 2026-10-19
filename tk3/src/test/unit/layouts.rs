use std::collections::HashSet;

use test_case::test_case;

use svod_dtype::{AmdArch, GpuArch};

use super::{costs, declared, lane_moves_dear};
use crate::atoms::{Costs, Device, Target, sm86};
use crate::build::*;
use crate::ir::*;
use crate::layout::{self as frag, Dim};
use crate::layouts::{Issue, Laid, Pick, Relayout, TileLayout, WarpGrid, chunked, infer, natural, transposing};

fn coords(l: &TileLayout, warps: u32, lanes: u32) -> Vec<(u32, u32)> {
    (0..warps)
        .flat_map(|w| (0..lanes).flat_map(move |l_| (0..l.regs()).map(move |j| (w, l_, j))))
        .map(|(w, l_, j)| l.coord(w, l_, j))
        .collect()
}

/// The layout transposing loads leave a tile in holds every element in
/// 16-byte column runs, divides the 16×16 blocks among the warps where there
/// are enough, and exists only for wave32 tiles 16 divides.
#[test_case(16, 128, 4, 16; "16 keys of 128")]
#[test_case(32, 64, 4, 16; "32 keys of 64")]
#[test_case(64, 64, 8, 16; "64 keys over 8 warps")]
#[test_case(16, 48, 4, 24; "three blocks over 4 warps replicate")]
fn transposing_layouts_hold_16_byte_column_runs(rows: usize, cols: usize, warps: u32, regs: u32) {
    let l = transposing(Shape::new(rows, cols), warps, 32).unwrap();
    assert_eq!((l.shape(), l.regs()), (Shape::new(rows, cols), regs));
    let held: HashSet<(u32, u32)> = coords(&l, warps, 32).into_iter().collect();
    assert_eq!(held.len(), rows * cols, "every element held");
    assert!(
        l.runs(32, 8, Axis::Row).iter().all(|&(_, w)| w == 8),
        "column runs of eight: {:?}",
        l.runs(32, 8, Axis::Row)
    );
    assert!(l.runs(32, 8, Axis::Col).iter().all(|&(_, w)| w == 1), "no row runs");
    assert!(transposing(Shape::new(rows, cols), warps, 64).is_none(), "wave32 only");
    assert!(transposing(Shape::new(8, cols), warps, 32).is_none(), "blocks of 16");
}

/// The GEMM's accumulator takes the matrix core's C layout tiled over the
/// warp grid, and the output cast inherits it.
#[test]
fn gemm_accumulator_gets_the_core_layout_over_the_warp_grid() {
    let mut prog = super::gemm_nt(256, 256, 256, 128, 128, 32, 3);
    prog.warps = 8;
    let Laid { prog, layouts: lay } = infer(prog, &declared(sm86()), WarpGrid { rows: 2, cols: 4 }, false).unwrap();
    let Stmt::Pipeline(p) = &prog.body.0[1] else { panic!("pipeline") };
    let acc = lay[p.carried[0].phi.index()].as_ref().unwrap();
    assert_eq!(acc.frag, frag::mma_sync_c());
    assert_eq!(acc.reps, [4, 4]);
    assert_eq!(acc.shape(), Shape::new(128, 128));
    assert_eq!(acc.regs(), 64, "128·128 f32 over 256 threads");
    let Stmt::Let { dst, op: TileOp::Cast { .. } } = &prog.body.0[2] else { panic!("cast") };
    assert_eq!(lay[dst.index()], Some(acc.clone()));
    let init = p.carried[0].init;
    assert_eq!(lay[init.index()], Some(acc.clone()), "the zero init adopts the carried layout");
    // every element of the block tile is held exactly once
    let held: HashSet<_> = coords(acc, 8, 32).into_iter().collect();
    assert_eq!(held.len(), 128 * 128);
}

/// Attention: the probabilities come out of QKᵀ in the C layout and feed PV as
/// the A operand; on `mma.sync` the two coincide, so the inserted relayout is
/// the identity. The row statistics take the row-vector layout.
#[test]
fn attention_probabilities_feed_the_second_mma_without_data_movement() {
    let (bq, bkv, d) = (64usize, 64usize, 64usize);
    let mut k = Kernel::new("fa");
    let q = k.param::<BF16>("q", ParamKind::In, bq * d);
    let kk = k.smem::<BF16>("k", bkv * d);
    let vv = k.smem::<BF16>("v", bkv * d);
    let q_g = k.view(q, 0, [d, 1], Shape::new(bq, d), [None, None]);
    let q_r = k.load(q_g);
    let k_s = k.smem_view::<BF16>(kk, 0, Shape::new(bkv, d));
    let v_s = k.smem_view::<BF16>(vv, 0, Shape::new(bkv, d));
    let zero = k.zeros::<F32>(Shape::new(bq, bkv));
    let s = k.mma(zero, q_r, false, k_s, true);
    let m = k.reduce(s, Axis::Row, ReduceOp::Max);
    let p = k.binary(s, m, BinaryOp::Sub);
    let p = k.unary(p, UnaryOp::Exp2);
    let l = k.reduce(p, Axis::Row, ReduceOp::Sum);
    let p16 = k.cast::<F32, BF16>(p);
    let o0 = k.zeros::<F32>(Shape::new(bq, d));
    let o = k.mma(o0, p16, false, v_s, false);
    let o = k.binary(o, l, BinaryOp::Div);
    let out = k.param::<F32>("o", ParamKind::Out, bq * d);
    let o_g = k.view(out, 0, [d, 1], Shape::new(bq, d), [None, None]);
    k.store(o_g, o);
    let mut prog = k.finish();
    prog.warps = 4;
    let Laid { prog, layouts: lay } = infer(prog, &declared(sm86()), WarpGrid { rows: 4, cols: 1 }, false).unwrap();
    let relayouts: Vec<_> = prog
        .walk()
        .filter_map(|(_, s)| match s {
            Stmt::Let { dst, op: TileOp::Relayout { src } } => Some((*dst, *src)),
            _ => None,
        })
        .collect();
    assert_eq!(relayouts.len(), 1, "only P changes role");
    let (dst, src) = relayouts[0];
    let (from, to) = (lay[src.index()].as_ref().unwrap(), lay[dst.index()].as_ref().unwrap());
    assert_eq!(from.frag, frag::mma_sync_c());
    assert_eq!(to.frag, frag::mma_sync_a());
    assert_eq!(from.relayout(to, 4, 32, &costs()), Relayout::Identity);
    assert_eq!(lay[q_r.0.index()].as_ref().unwrap().frag, frag::mma_sync_a());
    let m_l = lay[m.0.index()].as_ref().unwrap();
    assert_eq!((m_l.frag.in_size(Dim::Reg), m_l.shape()), (2, Shape::new(bq, 1)), "two rows per lane");
    assert_eq!(m_l.row_fold_masks().as_slice(), &[1, 2], "quads share a row");
}

#[test_case(64, 64, 4, 32; "square")]
#[test_case(16, 256, 4, 32; "wide")]
#[test_case(256, 8, 8, 64; "narrow, wave64")]
#[test_case(4, 48, 4, 32; "48 wide: 16 spread three times")]
#[test_case(8, 96, 4, 64; "96 wide, wave64")]
fn natural_layouts_hold_every_element_once(rows: usize, cols: usize, warps: u32, lanes: u32) {
    let l = natural(Shape::new(rows, cols), warps, lanes).unwrap();
    assert_eq!(l.shape(), Shape::new(rows, cols));
    let held: HashSet<_> = coords(&l, warps, lanes).into_iter().collect();
    assert_eq!(held.len(), rows * cols);
    assert!(l.regs() as usize * (warps * lanes) as usize >= rows * cols, "warps past the rows replicate");
}

#[test]
fn shapes_that_do_not_tile_the_grid_are_rejected() {
    let target = sm86();
    let atom = target.mma(svod_dtype::ScalarDType::BFloat16, svod_dtype::ScalarDType::Float32).unwrap();
    let direct = crate::ir::Orient::Direct;
    assert!(atom.issue(direct, WarpGrid { rows: 2, cols: 4 }, 48, 128, 32).is_err(), "48 rows over 2 warps of 16");
    assert!(atom.issue(direct, WarpGrid { rows: 2, cols: 4 }, 64, 128, 24).is_err(), "k = 24");
    let Issue { a, b, c, .. } = atom.issue(direct, WarpGrid { rows: 1, cols: 1 }, 16, 48, 32).unwrap();
    assert_eq!((a.reps, b.reps, c.reps), ([1, 2], [2, 6], [1, 6]), "48 columns are six n-halves");
}

/// A register-staged fill's layout holds every element of its tile, each
/// lane's registers in 16-byte row runs, and replicates only warps (or
/// lanes) the tile has no rows for.
#[test_case(128, 32, 2, 4, 32, Axis::Col; "128x32 bf16 on four waves")]
#[test_case(96, 32, 2, 4, 32, Axis::Col; "96 rows repeat by three")]
#[test_case(64, 48, 2, 2, 32, Axis::Col; "48 columns repeat by three")]
#[test_case(16, 64, 2, 1, 32, Axis::Col; "one wave")]
#[test_case(64, 64, 2, 4, 64, Axis::Col; "wave64")]
#[test_case(8, 16, 2, 4, 32, Axis::Col; "fewer chunks than threads")]
#[test_case(32, 32, 4, 2, 32, Axis::Col; "f32")]
#[test_case(16, 128, 2, 4, 32, Axis::Row; "16 keys walked down the rows")]
#[test_case(64, 64, 2, 2, 32, Axis::Row; "64 keys walked down the rows")]
#[test_case(96, 32, 2, 4, 32, Axis::Row; "96 rows walked down the rows")]
fn chunked_layouts_cover_their_tile_in_16_byte_runs(
    rows: usize,
    cols: usize,
    bytes: usize,
    warps: u32,
    lanes: u32,
    walk: Axis,
) {
    let l = chunked(Shape::new(rows, cols), bytes, warps, lanes, walk).unwrap();
    assert_eq!(l.shape(), Shape::new(rows, cols));
    let held: HashSet<(u32, u32)> = coords(&l, warps, lanes).into_iter().collect();
    assert_eq!(held.len(), rows * cols, "every element held");
    let v = (16 / bytes).min(cols) as u32;
    if walk == Axis::Row {
        let (r0, c0) = l.coord(0, 0, 0);
        let (r1, c1) = l.coord(0, 1, 0);
        assert_eq!((r1, c1), (r0 + 1, c0), "the next lane holds the next row");
    }
    let threads = (warps * lanes) as usize;
    if (rows * cols / v as usize).is_multiple_of(threads) {
        assert_eq!(l.regs() as usize * threads, rows * cols, "a tile the threads divide is held once, not replicated");
    }
    for w in 0..warps {
        for lane in 0..lanes {
            for j in (0..l.regs()).step_by(v as usize) {
                let (r, c) = l.coord(w, lane, j);
                assert_eq!(c % v, 0, "a run starts on a 16-byte boundary");
                assert!(
                    (1..v).all(|e| l.coord(w, lane, j + e) == (r, c + e)),
                    "a run is one row's consecutive columns"
                );
            }
        }
    }
}

/// Elementwise ops whose operands are held otherwise than their result: two
/// a product holds as its A and B fragments, and a sum whose result a later
/// product takes as its B while one of its operands is another's A. The operands are re-held, so each op
/// reads them in its result's layout.
#[test]
fn elementwise_operands_reach_the_op_in_its_layout() {
    let mut k = Kernel::new("mixed");
    k.warps(1);
    let x = k.param::<BF16>("x", ParamKind::In, 256);
    let y = k.param::<BF16>("y", ParamKind::In, 256);
    let view = |k: &mut Kernel, p| k.view(p, 0, [16, 1], Shape::new(16, 16), [None, None]);
    let (xg, yg) = (view(&mut k, x), view(&mut k, y));
    let (xr, yr) = (k.load(xg), k.load(yg));
    let acc = k.zeros::<F32>(Shape::new(16, 16));
    let prod = k.mma(acc, xr, false, yr, false);
    let sum = k.binary(xr, yr, BinaryOp::Add);
    let sum = k.cast::<BF16, F32>(sum);
    let both = k.binary(prod, sum, BinaryOp::Mul);
    // Laid only by what follows: the sum as a B operand, `u` as an A one.
    let z = k.param::<BF16>("z", ParamKind::In, 512);
    let (ug, wg) = (view(&mut k, z), k.view(z, 256, [16, 1], Shape::new(16, 16), [None, None]));
    let (u, w) = (k.load(ug), k.load(wg));
    let uw = k.binary(u, w, BinaryOp::Add);
    let acc = k.zeros::<F32>(Shape::new(16, 16));
    let first = k.mma(acc, u, false, yr, false);
    let again = k.mma(first, xr, false, uw, false);
    let both = k.binary(both, again, BinaryOp::Add);
    let out = k.param::<F32>("o", ParamKind::Out, 256);
    let o = k.view(out, 0, [16, 1], Shape::new(16, 16), [None, None]);
    k.store(o, both);
    let Laid { prog, layouts } = infer(k.finish(), &declared(sm86()), WarpGrid { rows: 1, cols: 1 }, false).unwrap();
    let lay = |v: &ValId| layouts[v.index()].clone().unwrap();
    let mut checked = 0;
    for (_, s) in prog.walk() {
        if let Stmt::Let { dst, op: TileOp::Binary { a, b, .. } } = s {
            assert_eq!((lay(a), lay(b)), (lay(dst), lay(dst)), "{s:?}");
            checked += 1;
        }
    }
    assert_eq!(checked, 4);
    assert!(prog.walk().any(|(_, s)| matches!(s, Stmt::Let { op: TileOp::Relayout { .. }, .. })), "one re-held");
}

/// One attention step as `infer` sees it: QKᵀ, a row softmax and PV over four
/// warps, the scores plus a `[bq, bkv]` bias tile read from global memory
/// when `bias`.
fn attention_step(bias: bool) -> Program {
    let (bq, bkv, d) = (64usize, 16usize, 64usize);
    let mut k = Kernel::new("fa");
    let q = k.param::<BF16>("q", ParamKind::In, bq * d);
    let b = k.param::<BF16>("bias", ParamKind::In, bq * bkv);
    let kk = k.smem::<BF16>("k", bkv * d);
    let vv = k.smem::<BF16>("v", bkv * d);
    let q_g = k.view(q, 0, [d, 1], Shape::new(bq, d), [None, None]);
    let q_r = k.load(q_g);
    let k_s = k.smem_view::<BF16>(kk, 0, Shape::new(bkv, d));
    let v_s = k.smem_view::<BF16>(vv, 0, Shape::new(bkv, d));
    let zero = k.zeros::<F32>(Shape::new(bq, bkv));
    let mut s = k.mma(zero, q_r, false, k_s, true);
    if bias {
        let b_g = k.view(b, 0, [bkv, 1], Shape::new(bq, bkv), [None, None]);
        let b_r = k.load(b_g);
        let b_r = k.cast::<BF16, F32>(b_r);
        s = k.binary(s, b_r, BinaryOp::Add);
    }
    let m = k.reduce(s, Axis::Row, ReduceOp::Max);
    let p = k.binary(s, m, BinaryOp::Sub);
    let p = k.unary(p, UnaryOp::Exp2);
    let p16 = k.cast::<F32, BF16>(p);
    let o0 = k.zeros::<F32>(Shape::new(bq, d));
    let o = k.mma(o0, p16, false, v_s, false);
    let out = k.param::<F32>("o", ParamKind::Out, bq * d);
    let o_g = k.view(out, 0, [d, 1], Shape::new(bq, d), [None, None]);
    k.store(o_g, o);
    let mut prog = k.finish();
    prog.warps = 4;
    prog
}

/// The orientation `infer` gives each product of `prog` on gfx1151 (RDNA3.5)
/// at `costs`, and the layouts each inserted relayout re-holds a tile between.
fn gfx1151_attention(prog: Program, costs: &Costs) -> (Vec<Orient>, Vec<(TileLayout, TileLayout)>) {
    let target = Target::for_arch(GpuArch::Amd(AmdArch::Gfx1151));
    let Laid { prog, layouts: lay } =
        infer(prog, &Device { target, costs: *costs }, WarpGrid { rows: 4, cols: 1 }, false).unwrap();
    let orients = prog
        .walk()
        .filter_map(|(_, s)| match s {
            Stmt::Let { op: TileOp::Mma { orient, .. }, .. } => *orient,
            _ => None,
        })
        .collect();
    let relayouts = prog
        .walk()
        .filter_map(|(_, s)| match s {
            Stmt::Let { dst, op: TileOp::Relayout { src } } => {
                Some((lay[src.index()].clone()?, lay[dst.index()].clone()?))
            }
            _ => None,
        })
        .collect();
    (orients, relayouts)
}

/// On gfx11 the score accumulator never feeds the next product as laid. With
/// QKᵀ issued swapped each lane holds one query, so its row statistics stay in
/// the lane, and where two lane moves cost less than a shared-memory round
/// trip (the declared [`costs`]) P reaches PV by the two half-waves trading
/// registers.
#[test]
fn rdna3_attention_trades_probabilities_between_half_waves() {
    let (orients, pairs) = gfx1151_attention(attention_step(false), &costs());
    let relayouts: Vec<Relayout> = pairs.iter().map(|(from, to)| from.relayout(to, 4, 32, &costs())).collect();
    assert_eq!(orients[0], Orient::Swapped, "QKᵀ: {orients:?}");
    assert_eq!(relayouts.len(), 1, "only P changes role: {relayouts:?}");
    let Relayout::Exchange(picks) = &relayouts[0] else { panic!("an exchange: {:?}", relayouts[0]) };
    assert!(picks.iter().all(|p| p.len() <= 2 && p.iter().all(|p| p.mask < 32)), "{picks:?}");
}

/// A bias tile read into the swapped scores would scatter, a row per lane:
/// where a scattered element costs what a round trip does (the declared
/// [`costs`]) the products stay direct and the bias is read along its rows.
#[test]
fn rdna3_attention_reads_a_bias_along_its_rows() {
    let (orients, _) = gfx1151_attention(attention_step(true), &costs());
    assert_eq!(orients, [Orient::Direct, Orient::Direct]);
}

/// Where a lane move costs more than a round trip through shared memory, P
/// reaches PV through shared memory: no exchange is inserted.
#[test]
fn rdna3_attention_takes_the_round_trip_where_lane_moves_are_dear() {
    let (_, pairs) = gfx1151_attention(attention_step(false), &lane_moves_dear());
    assert!(!pairs.is_empty());
    for (from, to) in &pairs {
        assert_eq!(from.relayout(to, 4, 32, &lane_moves_dear()), Relayout::ViaSmem);
    }
}

/// An exchange of work `k` per element becomes the round trip exactly when
/// `k` lane moves cost at least the round trip.
#[test]
fn an_exchange_turns_into_the_round_trip_when_its_moves_cost_as_much() {
    let (_, pairs) = gfx1151_attention(attention_step(false), &costs());
    let (from, to) = &pairs[0];
    let exchange = from.relayout(to, 4, 32, &costs());
    assert!(matches!(exchange, Relayout::Exchange(_)), "an exchange: {exchange:?}");
    let k = exchange.work();
    let at = |lane_move: f64| from.relayout(to, 4, 32, &Costs { lane_move, ..costs() });
    let even = costs().smem_trip / k;
    assert_eq!(at(even), Relayout::ViaSmem, "{k} moves at the round trip's cost");
    assert!(matches!(at(even * 0.99), Relayout::Exchange(_)), "{k} moves just under it");
}

fn pick() -> Pick {
    Pick { mask: 16, reg: 0, lanes: u64::MAX }
}

/// Picks of `(mask, reg)`, per destination register.
fn picks(regs: &[&[(u32, u32)]]) -> Relayout {
    Relayout::Exchange(
        regs.iter().map(|r| r.iter().map(|&(mask, reg)| Pick { mask, reg, lanes: u64::MAX }).collect()).collect(),
    )
}

/// An exchange's work per element is what the emitter issues: each distinct
/// `(mask, reg)` move of a non-zero mask once, a select per pick past a
/// register's first, over its registers.
#[test_case(&[&[(0, 0), (16, 1)], &[(0, 1), (16, 0)]], 2.0; "half swap: two moves, two selects")]
#[test_case(&[&[(16, 0)], &[(16, 1)]], 1.0; "one pick of a mask: two moves")]
#[test_case(&[&[(0, 0)], &[(16, 1)]], 0.5; "one pick, half the registers stay: one move")]
#[test_case(&[&[(0, 0), (16, 1), (1, 0)], &[(0, 1), (16, 0), (1, 1)]], 4.0; "three picks: four moves, four selects")]
#[test_case(&[&[(16, 0), (0, 0)], &[(16, 0), (0, 1)]], 1.5; "a move shared by two registers, two selects")]
fn exchange_work_counts_moves_and_selects(regs: &[&[(u32, u32)]], work: f64) {
    assert_eq!(picks(regs).work(), work);
}

/// The half-wave exchange a relayout infers moves every register once and
/// selects once per register.
#[test_case(4, 16; "across the rows of 16")]
#[test_case(0, 1; "within them")]
fn a_half_swap_moves_and_selects_once_per_register(lane: u32, mask: u32) {
    let plain = natural(Shape::new(64, 64), 4, 32).unwrap();
    let held = crate::probe::half_swapped(&plain, lane);
    let exchange = plain.relayout(&held, 4, 32, &Costs { lane_move: 0.0, ..costs() });
    let Relayout::Exchange(p) = &exchange else { panic!("an exchange: {exchange:?}") };
    let masks: HashSet<u32> = p.iter().flatten().map(|p| p.mask).collect();
    assert_eq!(masks, HashSet::from([0, mask]));
    assert_eq!(exchange.work(), 2.0);
}

/// The per-element cost of each way to re-hold a tile is its work in the
/// declared cost of its primitive, an in-lane permutation free like the
/// identity, so their order follows the costs.
#[test]
fn relayout_costs_follow_the_declared_costs() {
    let costs = costs();
    let ways = [
        Relayout::Identity,
        Relayout::RegPermute(vec![1, 0]),
        Relayout::Exchange(vec![vec![pick(), pick()], vec![pick()]]),
        Relayout::ViaSmem,
    ];
    let priced = ways.each_ref().map(|w| w.cost_per_element(&costs));
    assert_eq!(priced, [0.0, 0.0, costs.lane_move, costs.smem_trip], "one move shared, one select");
}

/// A warpgroup accumulator as the PTX ISA tables it (`wgmma` D fragment,
/// `.m64nNk16` f32): thread `t` of warp `w` holds register `j` at row
/// `16w + t/4 + 8·((j/2)%2)`, column `2·(t%4) + j%2 + 8·(j/4)`.
#[test_case(8; "n8")]
#[test_case(64; "n64")]
#[test_case(256; "n256")]
fn warpgroup_accumulator_follows_the_ptx_table(n: u32) {
    let atom = Target::for_arch(GpuArch::Cuda(svod_dtype::CudaArch { major: 9, minor: 0 })).group_mma[0].clone();
    let issue = atom.issue(WarpGrid { rows: 4, cols: 1 }, 64, n as usize, 16).unwrap();
    assert_eq!(issue.c.regs(), n / 2);
    assert_eq!(issue.c.shape(), Shape::new(64, n as usize));
    for w in 0..4 {
        for t in 0..32 {
            for j in 0..n / 2 {
                let want = (16 * w + t / 4 + 8 * ((j / 2) % 2), 2 * (t % 4) + j % 2 + 8 * (j / 4));
                assert_eq!(issue.c.coord(w, t, j), want, "warp {w} lane {t} register {j}");
            }
        }
    }
    // The register A operand of the next product is the accumulator's own
    // layout, cast: a 16-wide slice of its rows per k16 step.
    let a = issue.a.coord(1, 5, 3);
    assert_eq!(a, issue.c.coord(1, 5, 3));
}

/// A grid of warpgroups: warps `4g..4g + 4` are one group down the rows, the
/// groups walk the columns first; a group holds 64 rows and its share of the
/// columns in the widest instruction that tiles them, repeated along them.
#[test]
fn warpgroups_are_consecutive_warps_down_the_rows() {
    let atom = Target::for_arch(GpuArch::Cuda(svod_dtype::CudaArch { major: 9, minor: 0 })).group_mma[0].clone();
    let grid = WarpGrid { rows: 8, cols: 2 };
    let issue = atom.issue(grid, 128, 192, 64).unwrap();
    assert_eq!((atom.n_for(96), atom.n_for(256), atom.n_for(40), atom.n_for(4)), (Some(32), Some(256), Some(8), None));
    assert_eq!(issue.c.reps, [1, 3]);
    assert_eq!(issue.c.regs(), 48);
    assert_eq!(issue.a.reps, [1, 4]);
    assert_eq!(issue.a.regs(), 32);
    for warp in 0..16 {
        let (group, in_group) = (warp / 4, warp % 4);
        let (row, col) = issue.c.coord(warp, 0, 0);
        assert_eq!((row, col), (64 * (group / 2) + 16 * in_group, 96 * (group % 2)), "warp {warp}");
        assert_eq!(issue.a.coord(warp, 0, 0), (row, 0), "warp {warp}");
    }
    assert!(atom.issue(WarpGrid { rows: 2, cols: 1 }, 32, 64, 16).is_err(), "half a warpgroup");
    assert!(atom.issue(WarpGrid { rows: 4, cols: 1 }, 128, 64, 16).is_err(), "two row blocks a group");
    assert!(atom.issue(WarpGrid { rows: 4, cols: 1 }, 64, 64, 24).is_err(), "a k off the step");
}
