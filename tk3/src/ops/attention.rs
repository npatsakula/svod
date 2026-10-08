//! Multi-head attention over sequence-major `[B, T, H, D]` heads.

use snafu::{ResultExt, ensure};
use svod_dtype::DType;
use svod_ir::SInt;
use svod_ir::origin::OriginScope;
use svod_tensor::Tensor;

use super::config;
use super::shape::{self, Plan, extent, shape_of};
use super::{
    DtypeSnafu, GraphSnafu, HeadsSnafu, LaunchSnafu, Result, ShapeSnafu, batch_of, fmt_shape, output, tuned, typed,
};
use crate::kernels;
use crate::kernels::attention::{AttnMask, AttnSpec, CombineSpec, FaCfg, attention as fa, combine, key_mask_stride};
use crate::kernels::rows::NormCfg;
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

/// Keys and values read from a cache `[rows, Tk, H_all, D]` holding several
/// layers' heads: a decoder step's self or cross attention.
#[derive(Clone, Copy, Debug)]
pub struct Cache<'a> {
    /// The first of this attention's `kv_heads` heads in the cache row.
    pub head_start: usize,
    pub kv_heads: usize,
    /// `[B]` integer cache row per batch lane; `None` reads row `b`.
    pub row_map: Option<&'a Tensor>,
    /// `[B, 1, kv_heads, D]` key and value scored after the cached prefix:
    /// the token the step projected, not yet written to the cache. Needs
    /// [`KeyMask::Lens`] naming the prefix.
    pub appended: Option<(&'a Tensor, &'a Tensor)>,
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
    /// `k` and `v` are caches read as [`Cache`] says.
    pub cache: Option<Cache<'a>>,
    /// Key blocks split over this many blocks per query tile, merged by a
    /// second kernel: parallelism for a few long rows (a decoder step).
    /// Capped at the key block count; `None` is one.
    pub splits: Option<usize>,
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
    let rows = match opts.cache {
        Some(Cache { row_map: Some(_), .. }) => "rows".to_string(),
        _ => qs[0].to_string(),
    };
    let expected = format!("[{rows}, Tk, H_kv, {}]", qs[3]);
    let batch_fits = ks[0] == qs[0] || opts.cache.is_some_and(|c| c.row_map.is_some());
    ensure!(ks.len() == 4 && batch_fits && ks[3] == qs[3], shape_err("k", &ks, expected));
    ensure!(vs == ks, shape_err("v", &vs, fmt_shape(&ks)));
    for (operand, t) in [("k", k), ("v", v)] {
        ensure!(t.dtype() == q.dtype(), DtypeSnafu { op: OP, operand, got: t.dtype(), want: q.dtype() });
    }
    let kv_heads = opts.cache.map_or(ks[2].clone(), |c| SInt::Const(c.kv_heads));
    if let (Some(heads), Some(kv_heads)) = (qs[2].as_const(), kv_heads.as_const()) {
        ensure!(kv_heads > 0 && heads.is_multiple_of(kv_heads), HeadsSnafu { op: OP, heads, kv_heads });
    }
    if let Some(cache) = opts.cache {
        let fits = ks[2].as_const().is_some_and(|total| cache.head_start + cache.kv_heads <= total);
        ensure!(fits, shape_err("k", &ks, format!("[rows, Tk, ≥ {}, {}]", cache.head_start + cache.kv_heads, qs[3])));
        if let Some(map) = cache.row_map {
            let got = shape(map)?;
            ensure!(got == [qs[0].clone()], shape_err("row map", &got, format!("[{}]", qs[0])));
        }
        if let Some((ka, va)) = cache.appended {
            let want = vec![qs[0].clone(), SInt::Const(1), kv_heads.clone(), qs[3].clone()];
            for (operand, t) in [("appended k", ka), ("appended v", va)] {
                let got = shape(t)?;
                ensure!(got == want, shape_err(operand, &got, fmt_shape(&want)));
                ensure!(t.dtype() == q.dtype(), DtypeSnafu { op: OP, operand, got: t.dtype(), want: q.dtype() });
            }
            ensure!(
                matches!(opts.keys, KeyMask::Lens(_)),
                shape_err("key lens", &[], "[B] with an appended row".into())
            );
        }
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
    let ([b, t, heads, d], [kv_rows, tk, heads_total, _]) = (dims4(&q_ext.dims), dims4(&k_ext.dims));
    let batch = batch_of(&var, b);
    let kv_heads = opts.cache.map_or(heads_total, |c| c.kv_heads);
    let cache = opts.cache.map(|c| kernels::attention::Cache {
        rows: kv_rows,
        heads_total,
        head_start: c.head_start,
        row_map: c.row_map.is_some(),
        appended: c.appended.is_some(),
    });
    let edges = AttnMask { causal: opts.causal, window: opts.window, ..AttnMask::default() };
    let spec = |cfg, mask, cache| AttnSpec {
        batch: batch.clone(),
        t,
        tk,
        heads,
        kv_heads,
        d,
        mask,
        cache,
        scale: opts.scale.unwrap_or(1.0 / (d as f32).sqrt()),
        cfg,
    };
    // Every config with every split count worth measuring (a given one is
    // capped at the key blocks), the unsplit pick first.
    let cfgs: Vec<FaCfg> = cfgs
        .into_iter()
        .flat_map(|c| {
            let blocks = tk.div_ceil(c.bkv);
            let splits = match opts.splits {
                Some(n) => vec![n.clamp(1, blocks)],
                None => config::split_candidates(&target, batch.capacity() * heads * t.div_ceil(c.bq), blocks),
            };
            splits.into_iter().map(move |splits| FaCfg { splits, ..c })
        })
        .collect();
    let merge = |splits| CombineSpec { batch: batch.clone(), t, heads, d, splits, cfg: NormCfg { br: 4 } };
    // Measured with the static masks only: scratch parameters would be garbage.
    let shape = [batch.capacity(), t, tk, heads, kv_heads, d];
    let salt = (&batch, edges, cache.map(|c| (c.rows, c.heads_total, c.head_start)), opts.splits);
    let cfg = tuned(OP, &target, q.dtype(), &shape, salt, &cfgs, |cfg| {
        let plain = cache.map(|c| kernels::attention::Cache { row_map: false, appended: false, ..c });
        let mut programs = vec![(typed!(q.dtype(), fa, &spec(cfg, edges, plain)), cfg.lowering(target.clone()))];
        if cfg.splits > 1 {
            let merge = merge(cfg.splits);
            programs.push((typed!(q.dtype(), combine, &merge), merge.cfg.lowering(target.clone())));
        }
        programs
    });
    let i32 = |t: &Tensor| if t.dtype() == DType::Int32 { t.clone() } else { t.cast(DType::Int32) };
    let lens = lens.map(i32);
    // Rows padded to the kernel's aligned stride.
    let bools = bools.map(|m| {
        let pad = key_mask_stride(tk) - tk;
        let m = i32(m);
        if pad == 0 { m } else { m.try_pad(&[(0, 0), (0, pad as isize)]).expect("padding the last dim") }
    });
    let segs = opts.seg_start.map(i32);
    let row_map = opts.cache.and_then(|c| c.row_map).map(i32);
    let (k_app, v_app) = opts.cache.and_then(|c| c.appended).unzip();
    let mask = AttnMask { key_lens: lens.is_some(), key_mask: bools.is_some(), seg_start: segs.is_some(), ..edges };
    let spec = spec(cfg, mask, cache);
    let lowering = cfg.lowering(target.clone());
    let extras = [lens.as_ref(), bools.as_ref(), segs.as_ref(), row_map.as_ref(), k_app, v_app];
    if cfg.splits == 1 {
        let o = output(&q_ext.dims, &var, q.dtype());
        let ins: Vec<&Tensor> = [Some(q), Some(k), Some(v), Some(&o)].into_iter().chain(extras).flatten().collect();
        return launch::graph_launch(typed!(q.dtype(), fa, &spec), &lowering, &ins).context(LaunchSnafu { op: OP });
    }
    let part = |dims: &[usize]| Tensor::empty(dims, DType::Float32);
    let (o_part, m_part, l_part) = (
        part(&[cfg.splits, batch.capacity(), t, heads, d]),
        part(&[cfg.splits, batch.capacity(), t, heads]),
        part(&[cfg.splits, batch.capacity(), t, heads]),
    );
    let ins: Vec<&Tensor> = [Some(q), Some(k), Some(v), Some(&o_part), Some(&m_part), Some(&l_part)]
        .into_iter()
        .chain(extras)
        .flatten()
        .collect();
    let parts =
        launch::graph_launch_all(typed!(q.dtype(), fa, &spec), &lowering, &ins).context(LaunchSnafu { op: OP })?;
    let merge = merge(cfg.splits);
    let o = output(&q_ext.dims, &var, q.dtype());
    let ins = [&parts[3], &parts[4], &parts[5], &o];
    launch::graph_launch(typed!(q.dtype(), combine, &merge), &merge.cfg.lowering(target), &ins)
        .context(LaunchSnafu { op: OP })
}

fn dims4(dims: &[usize]) -> [usize; 4] {
    dims.try_into().expect("rank 4")
}

/// SDPA over head-major `[B, H, T, D]`, with the same masks; a cache is
/// narrowed to its heads, gathered by the row map and extended by the
/// appended row.
pub(crate) fn graph(q: &Tensor, k: &Tensor, v: &Tensor, opts: Attn) -> svod_tensor::error::Result<Tensor> {
    let head_major = |t: &Tensor| t.try_permute(&[0, 2, 1, 3]);
    let cached = |t: &Tensor, appended: Option<&Tensor>| -> svod_tensor::error::Result<Tensor> {
        let Some(cache) = opts.cache else { return Ok(t.clone()) };
        let mut t = t.narrow(2, cache.head_start, cache.kv_heads)?;
        if let Some(map) = cache.row_map {
            t = t.index_select(0, map)?;
        }
        match appended {
            Some(row) => Tensor::cat(&[&t, row], 1),
            None => Ok(t),
        }
    };
    let (k_app, v_app) = opts.cache.and_then(|c| c.appended).unzip();
    let (k, v) = (cached(k, k_app)?, cached(v, v_app)?);
    let tk = k.dim_const(1)?;
    // Key validity and segment masks are properties of the lengths, shared
    // by every layer: built outside the caller's origin scope so they share.
    let _shared = OriginScope::suspend();
    let valid = match opts.keys {
        KeyMask::Lens(lens) => {
            let prefix = Tensor::sequence_mask(lens, tk)?;
            Some(match k_app {
                // The appended key, at the end, is always seen.
                Some(_) => prefix.try_bitor(&Tensor::arange(tk as i64, None, None)?.try_eq(tk as i64 - 1)?)?,
                None => prefix,
            })
        }
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
        .key(&head_major(&k)?)
        .value(&head_major(&v)?)
        .is_causal(opts.causal)
        .maybe_window(opts.window)
        .enable_gqa(q.dim_const(2)? != k.dim_const(2)?)
        .maybe_key_padding_mask(valid.as_ref())
        .maybe_attn_mask(hidden.as_ref())
        .maybe_scale(opts.scale.map(f64::from))
        .call()?
        .try_permute(&[0, 2, 1, 3])
}
