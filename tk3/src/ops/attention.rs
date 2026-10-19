//! Multi-head attention over sequence-major `[B, T, H, D]` heads.

use snafu::{ResultExt, ensure};
use svod_dtype::DType;
use svod_ir::SInt;
use svod_ir::origin::OriginScope;
use svod_tensor::Tensor;

use super::shape::{self, Plan, extent, shape_of};
use super::{
    DtypeSnafu, GraphSnafu, HeadsSnafu, LaunchSnafu, Result, ShapeSnafu, batch_of, fmt_shape, output, tuned, typed,
};
use crate::kernels::attention::{AttnMask, AttnSpec, attention as fa, key_mask_stride};
use crate::launch;

/// Which keys each batch row attends to.
#[derive(Clone, Copy, Debug, Default)]
pub enum KeyMask<'a> {
    #[default]
    None,
    /// `[B]` integer valid-key counts: keys at and past `lens[b]` are masked.
    Lens(&'a Tensor),
    /// `[B, Tk]` bool (or integer) mask, true (nonzero) where the key is attended.
    Bool(&'a Tensor),
}

/// The masks apply together; a query with no visible key yields NaN, as a
/// softmax over an empty row does.
#[derive(Clone, Copy, Debug, Default)]
pub struct Attn<'a> {
    /// Key `j` is visible to query `i` only when `j ≤ i`.
    pub causal: bool,
    pub keys: KeyMask<'a>,
    /// Query `i` sees keys `[i − left, i + right]` only.
    pub window: Option<(usize, usize)>,
    /// `[B, T]` integer segment starts of packed rows, non-decreasing along
    /// `T`: query `i` does not see keys before `seg_start[b, i]`.
    pub seg_start: Option<&'a Tensor>,
    /// Defaults to `1/√D`.
    pub scale: Option<f32>,
}

const OP: &str = "attention";

/// `softmax(q·kᵀ·scale)·v` over `q [B, T, H, D]` and `k`/`v [B, Tk, H_kv, D]`,
/// query head `h` reading KV head `h / (H / H_kv)`; returns `[B, T, H, D]`.
pub fn attention(q: &Tensor, k: &Tensor, v: &Tensor, opts: Attn) -> Result<Tensor> {
    let shape = |t: &Tensor| shape_of(t).context(GraphSnafu { op: OP });
    let (qs, ks, vs) = (shape(q)?, shape(k)?, shape(v)?);
    let shape_err =
        |operand, got: &[SInt], expected: String| ShapeSnafu { op: OP, operand, got: fmt_shape(got), expected };
    ensure!(qs.len() == 4, shape_err("q", &qs, "[B, T, H, D]".into()));
    let expected = format!("[{}, Tk, H_kv, {}]", qs[0], qs[3]);
    ensure!(ks.len() == 4 && ks[0] == qs[0] && ks[3] == qs[3], shape_err("k", &ks, expected));
    ensure!(vs == ks, shape_err("v", &vs, fmt_shape(&ks)));
    for (operand, t) in [("k", k), ("v", v)] {
        ensure!(t.dtype() == q.dtype(), DtypeSnafu { op: OP, operand, got: t.dtype(), want: q.dtype() });
    }
    if let (Some(heads), Some(kv_heads)) = (qs[2].as_const(), ks[2].as_const()) {
        ensure!(kv_heads > 0 && heads.is_multiple_of(kv_heads), HeadsSnafu { op: OP, heads, kv_heads });
    }
    let (lens, bools) = match opts.keys {
        KeyMask::Lens(lens) => {
            let got = shape(lens)?;
            ensure!(got.len() == 1, shape_err("key lens", &got, format!("[{}]", qs[0])));
            (Some(lens), None)
        }
        KeyMask::Bool(mask) => {
            let got = shape(mask)?;
            ensure!(got == [qs[0].clone(), ks[1].clone()], shape_err("key mask", &got, fmt_shape(&ks[..2])));
            (None, Some(mask))
        }
        KeyMask::None => (None, None),
    };
    if let Some(seg) = opts.seg_start {
        let got = shape(seg)?;
        ensure!(got == qs[..2], shape_err("seg start", &got, fmt_shape(&qs[..2])));
    }

    let (q_ext, var) = extent(&qs).unzip();
    let k_ext = extent(&ks).map(|(e, _)| e);
    let target = super::target(&q.device());
    let plan = shape::attention(target.as_ref(), &[q.dtype(), k.dtype(), v.dtype()], q_ext.as_ref(), k_ext.as_ref());
    let Plan::Kernel(cfgs) = plan else { return graph(q, k, v, opts).context(GraphSnafu { op: OP }) };
    let (q_ext, k_ext, var) = (q_ext.expect("planned"), k_ext.expect("planned"), var.flatten());
    let target = target.expect("planned");
    let ([b, t, heads, d], [_, tk, kv_heads, _]) = (dims4(&q_ext.dims), dims4(&k_ext.dims));
    let batch = batch_of(&var, b);
    let edges = AttnMask { causal: opts.causal, window: opts.window, ..AttnMask::default() };
    let spec = |cfg, mask| AttnSpec {
        batch: batch.clone(),
        t,
        tk,
        heads,
        kv_heads,
        d,
        mask,
        scale: opts.scale.unwrap_or(1.0 / (d as f32).sqrt()),
        cfg,
    };
    // Measured with the static masks only: scratch parameters would be garbage.
    let shape = [batch.capacity(), t, tk, heads, kv_heads, d];
    let cfg = tuned(OP, &target, q.dtype(), &shape, (&batch, edges), &cfgs, |cfg| {
        (typed!(q.dtype(), fa, &spec(cfg, edges)), cfg.lowering(target.clone()))
    });
    let o = output(&q_ext.dims, &var, q.dtype());
    let i32 = |t: &Tensor| if t.dtype() == DType::Int32 { t.clone() } else { t.cast(DType::Int32) };
    let lens = lens.map(i32);
    // Rows padded to the kernel's aligned stride.
    let bools = bools.map(|m| {
        let pad = key_mask_stride(tk) - tk;
        let m = i32(m);
        if pad == 0 { m } else { m.try_pad(&[(0, 0), (0, pad as isize)]).expect("padding the last dim") }
    });
    let segs = opts.seg_start.map(i32);
    let mask = AttnMask { key_lens: lens.is_some(), key_mask: bools.is_some(), seg_start: segs.is_some(), ..edges };
    let ins: Vec<&Tensor> = [Some(q), Some(k), Some(v), Some(&o), lens.as_ref(), bools.as_ref(), segs.as_ref()]
        .into_iter()
        .flatten()
        .collect();
    launch::graph_launch(typed!(q.dtype(), fa, &spec(cfg, mask)), &cfg.lowering(target), &ins)
        .context(LaunchSnafu { op: OP })
}

fn dims4(dims: &[usize]) -> [usize; 4] {
    dims.try_into().expect("rank 4")
}

/// SDPA over head-major `[B, H, T, D]`, with the same masks.
pub(crate) fn graph(q: &Tensor, k: &Tensor, v: &Tensor, opts: Attn) -> svod_tensor::error::Result<Tensor> {
    let head_major = |t: &Tensor| t.try_permute(&[0, 2, 1, 3]);
    let tk = k.dim_const(1)?;
    // Key validity and segment masks are properties of the lengths, shared
    // by every layer: built outside the caller's origin scope so they share.
    let _shared = OriginScope::suspend();
    let valid = match opts.keys {
        KeyMask::Lens(lens) => Some(Tensor::sequence_mask(lens, tk)?),
        KeyMask::Bool(mask) => Some(if mask.dtype() == DType::Bool { mask.clone() } else { mask.try_ne(0i32)? }),
        KeyMask::None => None,
    };
    // Keys before the query's own segment are masked out (`true`).
    let hidden = match opts.seg_start {
        Some(start) => {
            let (b, t) = (start.dim(0)?, start.dim_const(1)?);
            let keys = Tensor::arange(0, Some(tk as i64), None)?.try_reshape([1isize, 1, 1, tk as isize])?;
            Some(keys.try_lt(&start.try_reshape([b, SInt::Const(1), SInt::Const(t), SInt::Const(1)])?)?)
        }
        None => None,
    };
    head_major(q)?
        .scaled_dot_product_attention()
        .key(&head_major(k)?)
        .value(&head_major(v)?)
        .is_causal(opts.causal)
        .maybe_window(opts.window)
        .enable_gqa(q.dim_const(2)? != k.dim_const(2)?)
        .maybe_key_padding_mask(valid.as_ref())
        .maybe_attn_mask(hidden.as_ref())
        .maybe_scale(opts.scale.map(f64::from))
        .call()?
        .try_permute(&[0, 2, 1, 3])
}
