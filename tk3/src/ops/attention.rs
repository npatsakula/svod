//! Multi-head attention over sequence-major `[B, T, H, D]` heads.

use snafu::{ResultExt, ensure};
use svod_dtype::DType;
use svod_ir::SInt;
use svod_ir::origin::OriginScope;
use svod_tensor::Tensor;

use super::shape::{self, Plan, extent, shape_of};
use super::{DtypeSnafu, GraphSnafu, HeadsSnafu, LaunchSnafu, Result, ShapeSnafu, batch_of, fmt_shape, live, typed};
use crate::kernels::attention::{AttnSpec, attention as fa};
use crate::launch;

/// Which keys each batch row attends to.
#[derive(Clone, Copy, Debug, Default)]
pub enum KeyMask<'a> {
    #[default]
    None,
    /// `[B]` integer valid-key counts: keys at and past `lens[b]` are masked.
    Lens(&'a Tensor),
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Attn<'a> {
    /// Key `j` is visible to query `i` only when `j ≤ i`.
    pub causal: bool,
    pub keys: KeyMask<'a>,
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
    let lens = match opts.keys {
        KeyMask::Lens(lens) => {
            let got = shape(lens)?;
            ensure!(got.len() == 1, shape_err("key lens", &got, format!("[{}]", qs[0])));
            Some(lens)
        }
        KeyMask::None => None,
    };

    let (q_ext, var) = extent(&qs).unzip();
    let k_ext = extent(&ks).map(|(e, _)| e);
    let target = super::target(&q.device());
    let plan = shape::attention(target.as_ref(), &[q.dtype(), k.dtype(), v.dtype()], q_ext.as_ref(), k_ext.as_ref());
    let Plan::Kernel(cfg) = plan else { return graph(q, k, v, opts, lens).context(GraphSnafu { op: OP }) };
    let (q_ext, k_ext, var) = (q_ext.expect("planned"), k_ext.expect("planned"), var.flatten());
    let ([b, t, heads, d], [_, tk, kv_heads, _]) = (dims4(&q_ext.dims), dims4(&k_ext.dims));
    let spec = AttnSpec {
        batch: batch_of(&var, b),
        t,
        tk,
        heads,
        kv_heads,
        d,
        causal: opts.causal,
        key_lens: lens.is_some(),
        scale: opts.scale.unwrap_or(1.0 / (d as f32).sqrt()),
        cfg,
    };
    let o = Tensor::empty(&q_ext.dims, q.dtype());
    let lens = lens.map(|l| if l.dtype() == DType::Int32 { l.clone() } else { l.cast(DType::Int32) });
    let ins: Vec<&Tensor> = [Some(q), Some(k), Some(v), Some(&o), lens.as_ref()].into_iter().flatten().collect();
    let o = launch::graph_launch(typed!(q.dtype(), fa, &spec), &cfg.lowering(target.expect("planned")), &ins)
        .context(LaunchSnafu { op: OP })?;
    live(OP, o, &var)
}

fn dims4(dims: &[usize]) -> [usize; 4] {
    dims.try_into().expect("rank 4")
}

/// SDPA over head-major `[B, H, T, D]`, with the same key-length mask.
pub(crate) fn graph(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    opts: Attn,
    lens: Option<&Tensor>,
) -> svod_tensor::error::Result<Tensor> {
    let head_major = |t: &Tensor| t.try_permute(&[0, 2, 1, 3]);
    // `[B, Tk]` key validity is a property of the lengths, shared by every
    // layer: built outside the caller's origin scope so the layers share it.
    let valid = match lens {
        Some(lens) => {
            let _shared = OriginScope::suspend();
            Some(Tensor::sequence_mask(lens, k.dim_const(1)?)?)
        }
        None => None,
    };
    head_major(q)?
        .scaled_dot_product_attention()
        .key(&head_major(k)?)
        .value(&head_major(v)?)
        .is_causal(opts.causal)
        .enable_gqa(q.dim_const(2)? != k.dim_const(2)?)
        .maybe_key_padding_mask(valid.as_ref())
        .maybe_scale(opts.scale.map(f64::from))
        .call()?
        .try_permute(&[0, 2, 1, 3])
}
