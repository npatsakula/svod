//! The attention prologue: a fused `[q | k | v]` projection row split into
//! sequence-major heads, `q` and `k` RMS-normalized over the head and rotated.

use super::{Batch, affine, batch_offset, bound, load_f32};
use crate::build::*;
use crate::ir::*;
use crate::kernels::rows::NormCfg;

/// Rotary `(cos, sin)` tables of `[t, d/2]` rows, one set for every batch
/// or one per batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Rope {
    pub per_batch: bool,
}

/// `[batch, t, (heads + 2·kv_heads)·d]` rows of a fused projection.
#[derive(Clone, Debug, PartialEq)]
pub struct HeadsSpec {
    pub batch: Batch,
    pub t: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub d: usize,
    /// RMSNorm `q` (`k`) over `d` with a `[d]` weight.
    pub q_norm: bool,
    pub k_norm: bool,
    pub eps: f64,
    pub rope: Option<Rope>,
    pub cfg: NormCfg,
}

/// Parameters in order: `qkv`, `q_w [d]` and `k_w [d]` where normed, `cos`
/// and `sin` where rotated, then the outputs `q [batch, t, heads, d]`,
/// `k` and `v [batch, t, kv_heads, d]`. A block handles `br` rows of one
/// head slot, in two half-width tiles so the rotation pairs the halves.
pub fn heads<T: Elem>(spec: &HeadsSpec) -> Program {
    let HeadsSpec { ref batch, t, heads, kv_heads, d, q_norm, k_norm, eps, rope, cfg: NormCfg { br } } = *spec;
    let (slots, half) = (heads + 2 * kv_heads, d / 2);
    let cap = batch.capacity();
    let mut k = Kernel::new("heads");
    let qkv = k.param::<T>("qkv", ParamKind::In, cap * t * slots * d);
    let q_w = q_norm.then(|| k.param::<T>("q_w", ParamKind::In, d));
    let k_w = k_norm.then(|| k.param::<T>("k_w", ParamKind::In, d));
    let table_rows = rope.map(|r| if r.per_batch { cap * t } else { t });
    let cos = table_rows.map(|rows| k.param::<T>("cos", ParamKind::In, rows * half));
    let sin = table_rows.map(|rows| k.param::<T>("sin", ParamKind::In, rows * half));
    let q = k.param::<T>("q", ParamKind::Out, cap * t * heads * d);
    let kk = k.param::<T>("k", ParamKind::Out, cap * t * kv_heads * d);
    let v = k.param::<T>("v", ParamKind::Out, cap * t * kv_heads * d);
    let (gz, bb) = batch.axis(&mut k);
    k.grid([Sc::from(t.div_ceil(br)), Sc::from(slots), gz]);
    k.warps(br as u32);
    let row0 = k.block(0) * br;
    let slot = k.block(1);
    let shape = Shape::new(br, half);
    let tiles = |k: &mut Kernel, p: ParamRef<T>, width: usize, col: Sc| {
        let view = k.view(p, batch_offset(&bb, t * width) + col, [width, 1], shape, [bound(t, br), None]);
        [k.at(view, row0.clone(), 0), k.at(view, row0.clone(), half)]
    };
    let src = tiles(&mut k, qkv, slots * d, slot.clone() * d);
    let tables = rope.map(|r| {
        let offset = if r.per_batch { batch_offset(&bb, t * half) } else { Sc::from(0) };
        let table = |k: &mut Kernel, p: ParamRef<T>| {
            let view = k.view(p, offset.clone(), [half, 1], shape, [bound(t, br), None]);
            let tile = k.at(view, row0.clone(), 0);
            load_f32(k, tile)
        };
        (table(&mut k, cos.expect("rotated")), table(&mut k, sin.expect("rotated")))
    });
    let (x1, x2) = (load_f32(&mut k, src[0]), load_f32(&mut k, src[1]));

    // The head's `q`, `k` or `v` path: normed and rotated where asked, then
    // stored to its output at the slot's head.
    let path = |k: &mut Kernel, out: ParamRef<T>, heads: usize, head: Sc, w: Option<ParamRef<T>>, rotate: bool| {
        let (mut y1, mut y2) = (x1, x2);
        if let Some(w) = w {
            let sq1 = k.binary(y1, y1, BinaryOp::Mul);
            let sq2 = k.binary(y2, y2, BinaryOp::Mul);
            let sq = k.binary(sq1, sq2, BinaryOp::Add);
            let sum = k.reduce(sq, Axis::Row, ReduceOp::Sum);
            let var = affine(k, sum, 1.0 / d as f64, eps);
            let inv = k.unary(var, UnaryOp::Rsqrt);
            let weights = |k: &mut Kernel, col: usize| {
                let view = k.view(w, col, [0, 1], Shape::new(1, half), [None, None]);
                load_f32(k, view)
            };
            let (w1, w2) = (weights(k, 0), weights(k, half));
            y1 = k.binary(y1, inv, BinaryOp::Mul);
            y1 = k.binary(y1, w1, BinaryOp::Mul);
            y2 = k.binary(y2, inv, BinaryOp::Mul);
            y2 = k.binary(y2, w2, BinaryOp::Mul);
        }
        if let (true, Some((c, s))) = (rotate, tables) {
            let (c1, s2) = (k.binary(y1, c, BinaryOp::Mul), k.binary(y2, s, BinaryOp::Mul));
            let (s1, c2) = (k.binary(y1, s, BinaryOp::Mul), k.binary(y2, c, BinaryOp::Mul));
            y1 = k.binary(c1, s2, BinaryOp::Sub);
            y2 = k.binary(s1, c2, BinaryOp::Add);
        }
        let dst = tiles(k, out, heads * d, head * d);
        let (y1, y2) = (k.cast::<F32, T>(y1), k.cast::<F32, T>(y2));
        k.store(dst[0], y1);
        k.store(dst[1], y2);
    };
    let is_q = slot.clone().lt(heads);
    let is_k = slot.clone().lt(heads + kv_heads);
    k.if_(
        is_q,
        |k| path(k, q, heads, slot.clone(), q_w, true),
        |k| {
            k.if_(
                is_k,
                |k| path(k, kk, kv_heads, slot.clone() - heads, k_w, true),
                |k| path(k, v, kv_heads, slot.clone() - heads - kv_heads, None, false),
            )
        },
    );
    k.finish()
}
