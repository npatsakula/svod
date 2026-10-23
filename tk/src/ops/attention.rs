//! Multi-head attention over sequence-major `[B, T, H, D]` heads.

use snafu::{ResultExt, ensure};
use svod_dtype::DType;
use svod_ir::SInt;
use svod_ir::origin::OriginScope;
use svod_tensor::Tensor;

use super::config::Planner;
use super::shape::Fallback;
use super::shape::{self, extent, shape_of};
use super::{
    DtypeSnafu, GraphSnafu, HeadsSnafu, LaunchSnafu, Result, ShapeSnafu, Tuner, batch_of, fmt_shape, kernel_buffer,
    output, typed,
};
use crate::ir::Var;
use crate::kernels;
use crate::kernels::Batch;
pub use crate::kernels::attention::Chunk;
use crate::kernels::attention::{
    AttnMask, AttnSpec, Bias, CombineSpec, FaCfg, Forms, attention as fa, combine, key_mask_stride,
};
use crate::kernels::rows::NormCfg;
use crate::launch;
use crate::lower::Lowering;
use crate::tune::TuneStore;

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
    /// `[B, n, kv_heads, D]` keys and values scored after the cached prefix:
    /// the tokens the step projected, not yet written to the cache. Needs
    /// [`KeyMask::Lens`] naming the prefix; `n` is at most a key block.
    pub appended: Option<(&'a Tensor, &'a Tensor)>,
    /// `[B]` integer counts of visible appended rows, the rest hidden;
    /// `None` sees them all.
    pub app_lens: Option<&'a Tensor>,
}

/// The masks apply together; a query with no visible key yields zeros, as
/// the graph writes it.
#[derive(Clone, Copy, Debug, Default)]
pub struct Attn<'a> {
    /// Key `j` is visible to query `i` only when `j ≤ i`.
    pub causal: bool,
    pub keys: KeyMask<'a>,
    /// Query `i` sees keys `[i − left, i + right]` only.
    pub window: Option<(usize, usize)>,
    /// Block attention over one sequence: query `i` sees the keys of chunk
    /// `i / size`, of the `left` chunks before it and the `right` after.
    pub chunk: Option<Chunk>,
    /// `[B, T]` integer segment starts of packed rows, non-decreasing along
    /// `T`: query `i` does not see keys before `seg_start[b, i]`.
    pub seg_start: Option<&'a Tensor>,
    /// `k` and `v` are caches read as [`Cache`] says.
    pub cache: Option<Cache<'a>>,
    /// Key blocks split over this many blocks per query tile, merged by a
    /// second kernel: parallelism for a few long rows (a decoder step).
    /// Capped at the key block count; `None` lets the tune store choose
    /// among one, two, the counts around two blocks per SM and the block
    /// count (one when tuning is off).
    pub splits: Option<usize>,
    /// Defaults to `1/√D`.
    pub scale: Option<f32>,
    /// `[B, H, T, Tk]` or `[1, H, T, Tk]` added to the scaled scores before
    /// the masks: `softmax(q·kᵀ·scale + bias)`. With appended rows `Tk`
    /// counts them, their columns last. Off the stream dtype keeps the graph.
    pub bias: Option<&'a Tensor>,
}

const OP: &str = "attention";
/// Search bound: the rows (one warp each) of a split's merge block, which
/// the tuner times as part of every split candidate rather than on its own.
const MERGE_ROWS: usize = 4;

/// Measurement: the lanes a shape is measured at, at most.
const TUNE_BATCH: usize = 64;

/// `softmax(q·kᵀ·scale + bias)·v` over `q [B, T, H, D]` and `k`/`v [B, Tk, H_kv, D]`,
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
            let rows = shape(ka)?.get(1).and_then(SInt::as_const).filter(|&n| n > 0);
            let want = vec![qs[0].clone(), SInt::Const(rows.unwrap_or(1)), kv_heads.clone(), qs[3].clone()];
            for (operand, t) in [("appended k", ka), ("appended v", va)] {
                let got = shape(t)?;
                ensure!(rows.is_some() && got == want, shape_err(operand, &got, fmt_shape(&want)));
                ensure!(t.dtype() == q.dtype(), DtypeSnafu { op: OP, operand, got: t.dtype(), want: q.dtype() });
            }
            ensure!(matches!(opts.keys, KeyMask::Lens(_)), shape_err("key lens", &[], "[B] with appended rows".into()));
        }
        if let Some(lens) = cache.app_lens {
            let got = shape(lens)?;
            let fits = cache.appended.is_some() && got == [qs[0].clone()];
            ensure!(fits, shape_err("appended lens", &got, format!("[{}] with appended rows", qs[0])));
        }
    }
    let (lens, bools) = match opts.keys {
        KeyMask::Lens(lens) => {
            let got = shape(lens)?;
            ensure!(got == [qs[0].clone()], shape_err("key lens", &got, format!("[{}]", qs[0])));
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
    let appended = opts.cache.and_then(|c| c.appended).map_or(0, |(ka, _)| ka.dim_const(1).expect("checked"));
    let keys = ks[1].as_const().map_or(ks[1].clone(), |n| SInt::Const(n + appended));
    let bias_rows = match opts.bias {
        Some(bias) => {
            let got = shape(bias)?;
            let want = [qs[0].clone(), qs[2].clone(), qs[1].clone(), keys.clone()];
            let fits = got.len() == 4 && (got[0] == want[0] || got[0] == SInt::Const(1)) && got[1..] == want[1..];
            ensure!(fits, shape_err("bias", &got, format!("[{} | 1, {}, {}, {keys}]", qs[0], qs[2], qs[1])));
            Some(if got[0] == SInt::Const(1) { Bias::Shared } else { Bias::PerBatch })
        }
        None => None,
    };

    let (q_ext, var) = extent(&qs).unzip();
    let k_ext = extent(&ks).map(|(e, _)| e);
    let planner = super::planner(&q.device());
    let dtypes: Vec<DType> = [q, k, v].into_iter().chain(opts.bias).map(Tensor::dtype).collect();
    // A chunk mask skips about as many key blocks as a causal one.
    let causal = opts.causal || opts.chunk.is_some();
    let biased = bias_rows.is_some();
    let plan = shape::attention(planner.as_deref(), &dtypes, q_ext.as_ref(), k_ext.as_ref(), causal, biased);
    let Some(cfgs) = super::kernel(OP, plan) else { return graph(q, k, v, opts).context(GraphSnafu { op: OP }) };
    let cfgs: Vec<FaCfg> = cfgs.iter().copied().filter(|c| c.holds_appended(appended)).collect();
    if cfgs.is_empty() {
        tracing::debug!(target: super::PLAN, op = OP, appended, "graph: no config holds the appended rows");
        super::ran_graph(OP, Fallback::Config);
        return graph(q, k, v, opts).context(GraphSnafu { op: OP });
    }
    let (q_ext, k_ext, var) = (q_ext.expect("planned"), k_ext.expect("planned"), var.flatten());
    let planner = planner.expect("planned");
    let ([b, t, heads, d], [kv_rows, tk, heads_total, _]) = (dims4(&q_ext.dims), dims4(&k_ext.dims));
    let batch = batch_of(&var, b);
    let kv_heads = opts.cache.map_or(heads_total, |c| c.kv_heads);
    let cache = opts.cache.map(|c| kernels::attention::CacheLayout {
        rows: kv_rows,
        heads_total,
        row_map: c.row_map.is_some(),
        appended,
        app_lens: c.app_lens.is_some(),
    });
    let edges = AttnMask { causal: opts.causal, window: opts.window, chunk: opts.chunk, ..AttnMask::default() };
    // `lens`, `bools` and `segs` are set below; the tuner reads the key lengths' presence.
    let mask = AttnMask { key_lens: lens.is_some(), bias: bias_rows, ..edges };
    let scale = opts.scale.unwrap_or(1.0 / (d as f32).sqrt());
    let spec = AttnSpec { batch: batch.clone(), t, tk, heads, kv_heads, d, mask, cache, scale, cfg: cfgs[0] };
    let cfg = tuned(TuneStore::global(), &planner, q.dtype(), &cfgs, spec.clone(), opts.splits).cfg;
    super::ran(OP, "flash_attention", cfg);
    let i32 = |t: &Tensor| if t.dtype() == DType::Int32 { t.clone() } else { t.cast(DType::Int32) };
    let lens = lens.map(i32);
    // Rows padded to the kernel's aligned stride.
    let bools = bools.map(|m| {
        let pad = key_mask_stride(tk) - tk;
        let m = i32(m);
        if pad == 0 { m } else { m.try_pad(&[(0, 0), (0, pad as isize)]).expect("padding the last dim") }
    });
    let segs = opts.seg_start.map(i32);
    // Rows padded to the kernel's aligned stride, like the key mask's.
    let bias = opts.bias.map(|b| {
        let keys = tk + appended;
        let pad = key_mask_stride(keys) - keys;
        if pad == 0 {
            b.clone()
        } else {
            b.try_pad(&[(0, 0), (0, 0), (0, 0), (0, pad as isize)]).expect("padding the last dim")
        }
    });
    let row_map = opts.cache.and_then(|c| c.row_map).map(i32);
    let head_start = opts.cache.map(|c| Tensor::from_slice([c.head_start as i32]));
    let (k_app, v_app) = opts.cache.and_then(|c| c.appended).unzip();
    let app_lens = opts.cache.and_then(|c| c.app_lens).map(i32);
    let mask = AttnMask {
        key_lens: lens.is_some(),
        key_mask: bools.is_some(),
        seg_start: segs.is_some(),
        bias: bias_rows,
        ..edges
    };
    let spec = AttnSpec { mask, cfg, ..spec };
    let lowering = cfg.lowering(&planner.device);
    let extras = [
        lens.as_ref(),
        bools.as_ref(),
        segs.as_ref(),
        row_map.as_ref(),
        head_start.as_ref(),
        k_app,
        v_app,
        app_lens.as_ref(),
        bias.as_ref(),
    ];
    if cfg.splits == 1 {
        let o = output(&q_ext.dims, &var, q.dtype()).context(GraphSnafu { op: OP })?;
        let ins: Vec<&Tensor> = [Some(q), Some(k), Some(v), Some(&o)].into_iter().chain(extras).flatten().collect();
        return launch::graph_launch(typed!(q.dtype(), fa, &spec), &lowering, &ins).context(LaunchSnafu { op: OP });
    }
    let part = |dims: &[usize]| {
        let dims: Vec<SInt> = dims.iter().map(|&d| SInt::Const(d)).collect();
        kernel_buffer(&dims, DType::Float32).context(GraphSnafu { op: OP })
    };
    let (o_part, m_part, l_part) = (
        part(&[cfg.splits, batch.capacity(), t, heads, d])?,
        part(&[cfg.splits, batch.capacity(), t, heads])?,
        part(&[cfg.splits, batch.capacity(), t, heads])?,
    );
    let ins: Vec<&Tensor> = [Some(q), Some(k), Some(v), Some(&o_part), Some(&m_part), Some(&l_part)]
        .into_iter()
        .chain(extras)
        .flatten()
        .collect();
    let parts =
        launch::graph_launch_all(typed!(q.dtype(), fa, &spec), &lowering, &ins).context(LaunchSnafu { op: OP })?;
    let merge = merge(batch, t, heads, d, cfg.splits);
    let o = output(&q_ext.dims, &var, q.dtype()).context(GraphSnafu { op: OP })?;
    let ins = [&parts[3], &parts[4], &parts[5], &o];
    launch::graph_launch(typed!(q.dtype(), combine, &merge), &Lowering::new(&planner.device), &ins)
        .context(LaunchSnafu { op: OP })
}

/// The merge of `splits` partial results.
fn merge(batch: Batch, t: usize, heads: usize, d: usize, splits: usize) -> CombineSpec {
    CombineSpec { batch, t, heads, d, splits, cfg: NormCfg { br: MERGE_ROWS } }
}

/// `spec` under the config [`attention`] launches for its shape and masks
/// (`spec.mask` as launched: the key lengths' presence and the bias are
/// what the tuner reads of it): the tuner's pick among `ranked` (the shape
/// gate's list, the configs that hold `spec`'s appended rows), at the
/// split count `splits` names or every one worth measuring, through `store`.
pub fn tuned(
    store: &TuneStore,
    planner: &Planner,
    dtype: DType,
    ranked: &[FaCfg],
    spec: AttnSpec,
    splits: Option<usize>,
) -> AttnSpec {
    let AttnSpec { t, tk, heads, d, cache, .. } = spec;
    let lens = spec.mask.key_lens;
    let edges =
        AttnMask { causal: spec.mask.causal, window: spec.mask.window, chunk: spec.mask.chunk, ..AttnMask::default() };
    let bias_rows = spec.mask.bias;
    // The configs are admitted at a given split count, else unsplit; then
    // each comes with every split count worth measuring, the unsplit first.
    let split = split_ranked(ranked, splits, tk);
    let expand = |admitted: &[FaCfg]| -> Vec<FaCfg> {
        let counts = |c: &FaCfg| match splits {
            Some(_) => vec![c.splits],
            None => planner.splits(spec.batch.capacity() * heads * t.div_ceil(c.bq), tk.div_ceil(c.bkv)),
        };
        admitted.iter().flat_map(|c| counts(c).into_iter().map(|splits| FaCfg { splits, ..*c })).collect()
    };
    // Measured with the static masks and the bias only (scratch parameters
    // would be garbage; a bias streams the same whatever it holds) and at
    // most `TUNE_BATCH` lanes: more fill the device alike, and a cache's
    // scratch grows with every lane.
    let measured_mask = AttnMask { bias: bias_rows, ..edges };
    let measured = match &spec.batch {
        Batch::Static(n) if *n > TUNE_BATCH => Batch::Static(TUNE_BATCH),
        Batch::Var(v) if v.max > TUNE_BATCH as i64 => {
            Batch::Var(Var { min: v.min.min(TUNE_BATCH as i64), max: TUNE_BATCH as i64, ..v.clone() })
        }
        other => other.clone(),
    };
    // A cache with fewer rows than lanes is read through a map (zeros, in
    // the scratch): lanes sharing a row is what makes a cross cache cheap,
    // and the tuner should see it. Otherwise a row per lane, no map.
    let rows = cache.map(|c| c.rows.min(measured.capacity()));
    let mapped = cache.is_some_and(|c| c.row_map) && rows.is_some_and(|r| r < measured.capacity());
    let shape = [measured.capacity(), t, tk, heads, spec.kv_heads, d];
    let salt = (&measured, measured_mask, cache.map(|c| c.heads_total), rows, mapped, splits);
    let build = |cfg: FaCfg| {
        // Scratch caches hold no appended rows and no garbage: a zero map.
        let plain = cache.map(|c| kernels::attention::CacheLayout {
            rows: rows.expect("a cache"),
            row_map: mapped,
            appended: 0,
            app_lens: false,
            ..c
        });
        let spec = AttnSpec { batch: measured.clone(), mask: measured_mask, cache: plain, cfg, ..spec.clone() };
        let mut programs = timed_keys(tk, cache.is_some() || lens, &cfg)
            .into_iter()
            .map(|tk| (typed!(dtype, fa, &AttnSpec { tk, ..spec.clone() }), cfg.lowering(&planner.device)))
            .collect::<Vec<_>>();
        if cfg.splits > 1 {
            let merge = merge(spec.batch.clone(), t, heads, d, cfg.splits);
            programs.push((typed!(dtype, combine, &merge), Lowering::new(&planner.device)));
        }
        programs
    };
    let mut tuner = Tuner::new(store, &planner.device, dtype.clone(), &shape, salt, build);
    let cfg = tune(&mut tuner, &split, bias_rows.is_some(), expand);
    tracing::debug!(target: super::PLAN, op = OP, ?shape, pick = ?cfg, "attention tuned");
    AttnSpec { cfg, ..spec }
}

/// `cfgs` (ranked) at the split count the caller names, else unsplit. A
/// named count is honoured by the configs with that many key blocks of
/// `tk`, the rest dropped; past the most any config has, it is capped there.
pub(crate) fn split_ranked(cfgs: &[FaCfg], splits: Option<usize>, tk: usize) -> Vec<FaCfg> {
    let blocks = |c: &FaCfg| tk.div_ceil(c.bkv);
    let most = cfgs.iter().map(blocks).max().unwrap_or(1);
    let splits = splits.map_or(1, |n| n.clamp(1, most));
    cfgs.iter().filter(|c| blocks(c) >= splits).map(|&c| FaCfg { splits, ..c }).collect()
}

/// The key counts a candidate is timed at, their times summed: the shape's
/// `tk`, and where the keys live `below` it (a cache's capacity, a length
/// parameter the scratch holds at zero) half of it too, at least the key
/// blocks its splits need. At capacity a decode step's leaders tie within
/// a round's drift, and the step's own length told them apart by 4–12 %
/// on every card, so the length a step runs at most of its life decides.
pub(crate) fn timed_keys(tk: usize, below: bool, cfg: &FaCfg) -> Vec<usize> {
    let half = (tk / 2).max(cfg.splits * cfg.bkv);
    if below && half < tk { vec![tk, half] } else { vec![tk] }
}

/// Search bound: the softmax forms, each flipped in turn by [`forms`]; the
/// last, the bias read directly, only under a bias.
pub(crate) const FORMS: [fn(Forms) -> Forms; 5] = [
    |f| Forms { truncate_p: !f.truncate_p, ..f },
    |f| Forms { skip_rescale: !f.skip_rescale, ..f },
    |f| Forms { raw_max: !f.raw_max, ..f },
    |f| Forms { sum_after: !f.sum_after, ..f },
    |f| Forms { bias_direct: !f.bias_direct, ..f },
];

/// Coordinate descent over the softmax forms from the tuned `winner`: each
/// of [`FORMS`] once (the bias form only with a `bias`), `pick` choosing
/// between the forms so far and the same with that one flipped (the tuner
/// times the pair and keeps the flip only where it measures faster).
pub(crate) fn forms(winner: FaCfg, bias: bool, mut pick: impl FnMut([FaCfg; 2]) -> FaCfg) -> FaCfg {
    let flips = FORMS.iter().take(if bias { FORMS.len() } else { FORMS.len() - 1 });
    flips.fold(winner, |c, flip| pick([c, FaCfg { forms: flip(c.forms), ..c }]))
}

/// The tuned attention config of `ranked`, its softmax forms and tiles
/// tuned in turn, since a tile may fit only under some forms (bf16
/// probabilities held truncated rather than in f32, say): the [`forms`]
/// pass on `first_clean` of the ranking (its first config that compiles
/// without spilling), the tile and split choice (`tiles`) over the ranking
/// under those forms, and the forms pass again on that winner. `pick`
/// chooses within a forms pair, as in [`forms`].
pub(crate) fn descend(
    ranked: &[FaCfg],
    bias: bool,
    first_clean: impl FnOnce(&[FaCfg]) -> FaCfg,
    tiles: impl FnOnce(&[FaCfg]) -> FaCfg,
    mut pick: impl FnMut([FaCfg; 2]) -> FaCfg,
) -> FaCfg {
    let kept = forms(first_clean(ranked), bias, &mut pick).forms;
    let formed: Vec<FaCfg> = ranked.iter().map(|&c| FaCfg { forms: kept, ..c }).collect();
    forms(tiles(&formed), bias, pick)
}

/// The config `tuner` launches from `ranked`, `expand` making a tile pass's
/// candidates as in [`Tuner::tuned`]: the softmax forms (the bias form
/// under a `bias`) and tiles tuned in turn by [`descend`], or with tuning
/// off the first clean, plain.
pub(crate) fn tune<B: Fn(FaCfg) -> crate::tune::Candidate>(
    tuner: &mut Tuner<'_, FaCfg, B>,
    ranked: &[FaCfg],
    bias: bool,
    expand: impl Fn(&[FaCfg]) -> Vec<FaCfg>,
) -> FaCfg {
    if !crate::tune::enabled() {
        return tuner.tuned(OP, ranked, expand);
    }
    let tuner = std::cell::RefCell::new(tuner);
    descend(
        ranked,
        bias,
        |r| tuner.borrow_mut().first_clean(OP, r).expect("a ranked attention config"),
        |r| tuner.borrow_mut().tuned(OP, r, &expand),
        |pair| {
            let mut tuner = tuner.borrow_mut();
            let admitted = tuner.admitted(OP, &pair);
            tuner.fastest(OP, &admitted)
        },
    )
}

fn dims4(dims: &[usize]) -> [usize; 4] {
    dims.try_into().expect("rank 4")
}

/// SDPA over head-major `[B, H, T, D]`, with the same masks and bias; a cache is
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
    let cached_keys = k.dim_const(1)?;
    let (k, v) = (cached(k, k_app)?, cached(v, v_app)?);
    let tk = k.dim_const(1)?;
    // Key validity and segment masks are properties of the lengths, shared
    // by every layer: built outside the caller's origin scope so they share.
    let _shared = OriginScope::suspend();
    let valid = match opts.keys {
        KeyMask::Lens(lens) => {
            let prefix = Tensor::sequence_mask(lens, tk)?;
            Some(match k_app {
                // The appended keys, after the cache, are seen: all, or the counted ones.
                Some(_) => {
                    let offset = Tensor::arange(0, Some(tk as i64), None)?.try_sub(cached_keys as i32)?;
                    let seen = match opts.cache.and_then(|c| c.app_lens) {
                        Some(n) => offset
                            .try_ge(0i32)?
                            .try_bitand(&offset.try_lt(&n.cast(DType::Int32).try_unsqueeze(-1)?)?)?,
                        None => offset.try_ge(0i32)?,
                    };
                    prefix.try_bitor(&seen)?
                }
                None => prefix,
            })
        }
        KeyMask::Bool(mask) => Some(if mask.dtype() == DType::Bool { mask.clone() } else { mask.try_ne(0i32)? }),
        KeyMask::None => None,
    };
    // Keys before the query's own segment are masked out (`true`).
    let mut hidden = match opts.seg_start {
        Some(start) => {
            let (b, t) = (start.dim(0)?, start.dim_const(1)?);
            let keys = Tensor::arange(0, Some(tk as i64), None)?.try_reshape([1isize, 1, 1, tk as isize])?;
            Some(keys.try_lt(&start.try_reshape([b, SInt::Const(1), SInt::Const(t), SInt::Const(1)])?)?)
        }
        None => None,
    };
    // Keys of a chunk past the reach ahead or behind are masked out.
    if let Some(Chunk { size, left, right }) = opts.chunk {
        let t = q.dim_const(1)?;
        let chunk_of = |n: usize, shape: [isize; 4]| -> svod_tensor::error::Result<Tensor> {
            Tensor::arange(0, Some(n as i64), None)?.try_cdiv(size as i32)?.try_reshape(shape)
        };
        let (cq, ck) = (chunk_of(t, [1, 1, t as isize, 1])?, chunk_of(tk, [1, 1, 1, tk as isize])?);
        let mut out = ck.try_gt(&cq.try_add(right as i32)?)?;
        if let Some(left) = left {
            out = out.try_bitor(&ck.try_add(left as i32)?.try_lt(&cq)?)?;
        }
        hidden = Some(match hidden {
            Some(h) => h.try_bitor(&out)?,
            None => out,
        });
    }
    // One attention mask: the segment mask folds into the bias as its floor.
    let attn_mask = match (hidden, opts.bias) {
        (hidden, None) => hidden,
        (None, Some(bias)) => Some(bias.clone()),
        (Some(hidden), Some(bias)) => {
            let bias = bias.cast(DType::Float32);
            let floor = Tensor::const_(f32::MIN, DType::Float32);
            Some(bias.where_(&hidden.logical_not()?, &floor)?)
        }
    };
    head_major(q)?
        .scaled_dot_product_attention()
        .key(&head_major(&k)?)
        .value(&head_major(&v)?)
        .is_causal(opts.causal)
        .maybe_window(opts.window)
        .enable_gqa(q.dim_const(2)? != k.dim_const(2)?)
        .maybe_key_padding_mask(valid.as_ref())
        .maybe_attn_mask(attn_mask.as_ref())
        .maybe_scale(opts.scale.map(f64::from))
        .call()?
        .try_permute(&[0, 2, 1, 3])
}
