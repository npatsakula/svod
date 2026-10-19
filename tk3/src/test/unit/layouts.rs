use std::collections::HashSet;

use test_case::test_case;

use crate::atoms::sm86;
use crate::build::*;
use crate::ir::*;
use crate::layout::{self as frag, Dim};
use crate::layouts::{Relayout, TileLayout, WarpGrid, chunked, infer, mma_layouts, natural};

fn coords(l: &TileLayout, warps: u32, lanes: u32) -> Vec<(u32, u32)> {
    (0..warps)
        .flat_map(|w| (0..lanes).flat_map(move |l_| (0..l.regs()).map(move |j| (w, l_, j))))
        .map(|(w, l_, j)| l.coord(w, l_, j))
        .collect()
}

/// The GEMM's accumulator takes the matrix core's C layout tiled over the
/// warp grid, and the output cast inherits it.
#[test]
fn gemm_accumulator_gets_the_core_layout_over_the_warp_grid() {
    let mut prog = super::gemm_nt(256, 256, 256, 128, 128, 32, 3);
    prog.warps = 8;
    let lay = infer(&mut prog, &sm86(), WarpGrid { rows: 2, cols: 4 }).unwrap();
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
    let lay = infer(&mut prog, &sm86(), WarpGrid { rows: 4, cols: 1 }).unwrap();
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
    assert_eq!(from.relayout(to, 4, 32), Relayout::Identity);
    assert_eq!(lay[q_r.0.index()].as_ref().unwrap().frag, frag::mma_sync_a());
    let m_l = lay[m.0.index()].as_ref().unwrap();
    assert_eq!((m_l.frag.in_size(Dim::Reg), m_l.shape()), (2, Shape::new(bq, 1)), "two rows per lane");
    assert_eq!(m_l.row_fold_masks().as_slice(), &[1, 2], "quads share a row");
}

#[test_case(64, 64, 4, 32; "square")]
#[test_case(16, 256, 4, 32; "wide")]
#[test_case(256, 8, 8, 64; "narrow, wave64")]
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
    assert!(mma_layouts(atom, WarpGrid { rows: 2, cols: 4 }, 48, 128, 32).is_err(), "48 rows over 2 warps of 16");
    assert!(mma_layouts(atom, WarpGrid { rows: 2, cols: 4 }, 64, 128, 24).is_err(), "k = 24");
    let [a, b, c] = mma_layouts(atom, WarpGrid { rows: 1, cols: 1 }, 16, 48, 32).unwrap();
    assert_eq!((a.reps, b.reps, c.reps), ([1, 2], [2, 6], [1, 6]), "48 columns are six n-halves");
}

/// A register-staged fill's layout holds every element of its tile, each
/// lane's registers in 16-byte row runs, and replicates only warps (or
/// lanes) the tile has no rows for.
#[test_case(128, 32, 2, 4, 32; "128x32 bf16 on four waves")]
#[test_case(96, 32, 2, 4, 32; "96 rows repeat by three")]
#[test_case(64, 48, 2, 2, 32; "48 columns repeat by three")]
#[test_case(16, 64, 2, 1, 32; "one wave")]
#[test_case(64, 64, 2, 4, 64; "wave64")]
#[test_case(8, 16, 2, 4, 32; "fewer chunks than threads")]
#[test_case(32, 32, 4, 2, 32; "f32")]
fn chunked_layouts_cover_their_tile_in_16_byte_runs(rows: usize, cols: usize, bytes: usize, warps: u32, lanes: u32) {
    let l = chunked(Shape::new(rows, cols), bytes, warps, lanes).unwrap();
    assert_eq!(l.shape(), Shape::new(rows, cols));
    let held: HashSet<(u32, u32)> = coords(&l, warps, lanes).into_iter().collect();
    assert_eq!(held.len(), rows * cols, "every element held");
    let v = (16 / bytes).min(cols) as u32;
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
