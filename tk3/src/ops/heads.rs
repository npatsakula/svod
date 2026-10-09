//! A fused `[q | k | v]` projection into sequence-major heads, `q` and `k`
//! RMS-normalized over the head and rotated.

use snafu::{ResultExt, ensure};
use svod_dtype::DType;
use svod_ir::SInt;
use svod_tensor::Tensor;

use super::shape::{self, Plan, extent, shape_of};
use super::{GraphSnafu, HeadsSnafu, LaunchSnafu, Result, ShapeSnafu, batch_of, fmt_shape, output, tuned, typed};
use crate::kernels::heads::{HeadsSpec, Rope, heads as kernel};
use crate::launch;

#[derive(Clone, Copy, Debug)]
pub struct Qkv<'a> {
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// `[head_dim]` RMSNorm weights for `q` and for `k`, sharing `eps`.
    pub q_norm: Option<&'a Tensor>,
    pub k_norm: Option<&'a Tensor>,
    pub eps: f64,
    /// Rotary `(cos, sin)` of `[1, T, 1, head_dim / 2]` by position or
    /// `[B, T, 1, head_dim / 2]` by token, applied to `q` and `k` after the
    /// norm, pairing the head's halves.
    pub rope: Option<(&'a Tensor, &'a Tensor)>,
}

const OP: &str = "heads";

/// `qkv [B, T, (heads + 2·kv_heads)·head_dim]` → `(q [B, T, heads, head_dim],
/// k, v [B, T, kv_heads, head_dim])`.
pub fn heads(qkv: &Tensor, opts: Qkv) -> Result<(Tensor, Tensor, Tensor)> {
    let Qkv { heads, kv_heads, head_dim: d, q_norm, k_norm, eps, rope } = opts;
    let shape = |t: &Tensor| shape_of(t).context(GraphSnafu { op: OP });
    let xs = shape(qkv)?;
    let shape_err =
        |operand, got: &[SInt], expected: String| ShapeSnafu { op: OP, operand, got: fmt_shape(got), expected };
    ensure!(kv_heads > 0 && heads.is_multiple_of(kv_heads), HeadsSnafu { op: OP, heads, kv_heads });
    let slots = heads + 2 * kv_heads;
    let width = format!("[B, T, {slots}·{d}]");
    ensure!(xs.len() == 3 && xs[2] == SInt::Const(slots * d), shape_err("qkv", &xs, width));
    for (operand, w) in [("q norm", q_norm), ("k norm", k_norm)] {
        if let Some(w) = w {
            let got = shape(w)?;
            ensure!(got == [SInt::Const(d)], shape_err(operand, &got, format!("[{d}]")));
        }
    }
    if let Some((cos, sin)) = rope {
        let got = shape(cos)?;
        let table = |b: &SInt| vec![b.clone(), xs[1].clone(), SInt::Const(1), SInt::Const(d / 2)];
        let fits = got == table(&SInt::Const(1)) || got == table(&xs[0]);
        ensure!(
            fits,
            shape_err("cos", &got, format!("{} or {}", fmt_shape(&table(&SInt::Const(1))), fmt_shape(&table(&xs[0]))))
        );
        let sin_shape = shape(sin)?;
        ensure!(sin_shape == got, shape_err("sin", &sin_shape, fmt_shape(&got)));
    }

    let (ext, var) = extent(&xs).unzip();
    let target = super::target(&qkv.device());
    // A weight or table off the stream dtype keeps the graph, as the norms do.
    let (cos, sin) = rope.unzip();
    let dtypes: Vec<DType> = [Some(qkv), q_norm, k_norm, cos, sin].into_iter().flatten().map(Tensor::dtype).collect();
    let plan = shape::heads(target.as_ref(), &dtypes, ext.as_ref(), d);
    let Plan::Kernel(cfgs) = plan else { return graph(qkv, opts).context(GraphSnafu { op: OP }) };
    let (ext, var, target) = (ext.expect("planned"), var.flatten(), target.expect("planned"));
    let (b, t) = (ext.dims[0], ext.dims[1]);
    let batch = batch_of(&var, b);
    let per_batch = rope.is_some_and(|(cos, _)| cos.shape().is_ok_and(|s| s[0] != SInt::Const(1)));
    let rope_spec = rope.map(|_| Rope { per_batch });
    let spec = |cfg| HeadsSpec {
        batch: batch.clone(),
        t,
        heads,
        kv_heads,
        d,
        q_norm: q_norm.is_some(),
        k_norm: k_norm.is_some(),
        eps,
        rope: rope_spec,
        cfg,
    };
    let shape = [batch.capacity(), t, heads, kv_heads, d];
    let salt = (&batch, q_norm.is_some(), k_norm.is_some(), rope_spec);
    let cfg = tuned(OP, &target, qkv.dtype(), &shape, salt, &cfgs, |cfg| {
        vec![(typed!(qkv.dtype(), kernel, &spec(cfg)), cfg.lowering(target.clone()))]
    });
    let q = output(&[b, t, heads, d], &var, qkv.dtype());
    let k = output(&[b, t, kv_heads, d], &var, qkv.dtype());
    let v = output(&[b, t, kv_heads, d], &var, qkv.dtype());
    let ins: Vec<&Tensor> =
        [Some(qkv), q_norm, k_norm, cos, sin, Some(&q), Some(&k), Some(&v)].into_iter().flatten().collect();
    let mut outs = launch::graph_launch_all(typed!(qkv.dtype(), kernel, &spec(cfg)), &cfg.lowering(target), &ins)
        .context(LaunchSnafu { op: OP })?;
    let v = outs.pop().expect("v");
    let k = outs.pop().expect("k");
    let q = outs.pop().expect("q");
    Ok((q, k, v))
}

/// The split, head views, norms and rotations as graph ops.
pub(crate) fn graph(qkv: &Tensor, opts: Qkv) -> svod_tensor::error::Result<(Tensor, Tensor, Tensor)> {
    let Qkv { heads, kv_heads, head_dim: d, q_norm, k_norm, eps, rope } = opts;
    let (b, t) = (qkv.dim(0)?, qkv.dim(1)?);
    let parts = qkv.split(&[heads * d, kv_heads * d, kv_heads * d], -1)?;
    let head = |part: &Tensor, n: usize, w: Option<&Tensor>, rotate: bool| -> svod_tensor::error::Result<Tensor> {
        let mut x = part.try_reshape([b.clone(), t.clone(), SInt::Const(n), SInt::Const(d)])?;
        if let Some(w) = w {
            x = x.rms_norm_with().eps(eps).weight(w).call()?;
        }
        if let (true, Some((cos, sin))) = (rotate, rope) {
            x = x.apply_rotary_emb(cos, sin, false)?;
        }
        Ok(x)
    };
    Ok((
        head(&parts[0], heads, q_norm, true)?,
        head(&parts[1], kv_heads, k_norm, true)?,
        head(&parts[2], kv_heads, None, false)?,
    ))
}
