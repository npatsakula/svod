//! Flash attention forward: Q resident in registers, K/V streamed through a
//! shared ring, the online-softmax state `(m, l, o)` carried.

use super::rows::NormCfg;
use super::{Batch, batch_offset, bound, load_f32};
use crate::atoms::{Device, Target};
use crate::build::*;
use crate::ir::*;
use crate::layouts::WarpGrid;
use crate::lower::Lowering;

/// A flash-attention config.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FaCfg {
    pub bq: usize,
    pub bkv: usize,
    pub stages: usize,
    /// Key splits: each is a block along grid x writing a partial result
    /// that [`combine`] merges; `1` writes the output directly.
    pub splits: usize,
    /// The ring loop unrolled over its slots.
    pub unroll: bool,
    pub forms: Forms,
    /// Warps sharing a block of one row-warp of queries (`bq` = [`WARP_ROWS`])
    /// by its keys: each walks its own range of key blocks through its own
    /// ring slots with its own softmax state, and the states merge through
    /// shared memory at the end. A one-row decoder step's block is otherwise
    /// a single wave waiting on every key block's load in turn; this many
    /// loads are in flight instead. `1` leaves the keys to one warp.
    pub key_warps: usize,
}

/// The softmax forms: all off as planned; the tuner flips each where it
/// measures faster on the device.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Forms {
    /// Probabilities held in bf16 are truncated rather than rounded, and the
    /// row sum adds what is held, so the normalization stays unbiased: a pair
    /// per instruction where rounding to nearest takes several. An f16 stream
    /// rounds either way.
    pub truncate_p: bool,
    /// The row sum follows the product rather than preceding the rescale,
    /// so both read the held probabilities once; before it, the f32 sum
    /// overlaps the product's operand conversion.
    pub sum_after: bool,
    /// The output's rescale is a [`BinaryOp::Scale`]: skipped under a lane
    /// branch by every wave whose rows all kept their maximum, which under the
    /// lazy rescale is most of them. Else a plain multiply.
    pub skip_rescale: bool,
    /// The running maximum is kept in the raw scores' units (a positive scale
    /// commutes with the maximum, so without a bias the raw scores are
    /// reduced and the maximum alone is scaled, a multiply per row rather
    /// than per score), at the price of a multiply on the step's dependent
    /// chain. Else the scores are scaled first.
    pub raw_max: bool,
}

/// Query rows per warp: the M of every matrix core's atom, so a row's scores
/// and its reductions stay within the warp. The planner refuses a core
/// whose atom differs ([`FaCfg::fits`](Self::fits)).
pub const WARP_ROWS: usize = 16;

impl FaCfg {
    pub const fn new(bq: usize, bkv: usize, stages: usize) -> Self {
        Self {
            bq,
            bkv,
            stages,
            splits: 1,
            unroll: false,
            forms: Forms { truncate_p: false, sum_after: false, skip_rescale: false, raw_max: false },
            key_warps: 1,
        }
    }

    /// One warp per [`WARP_ROWS`] query rows: the warps a tile is spread over.
    pub fn row_warps(&self) -> u32 {
        (self.bq / WARP_ROWS) as u32
    }

    /// The block's warps: the row warps, each split [`key_warps`](Self::key_warps) ways.
    pub fn warps(&self) -> u32 {
        self.row_warps() * self.key_warps as u32
    }

    pub fn lowering(&self, device: &Device) -> Lowering {
        Lowering::uniform(device, self.unroll, WarpGrid { rows: self.row_warps(), cols: 1 })
    }

    /// Static shared memory in bytes: the K and V rings of a 16-bit type, one
    /// per key warp (the states key warps merge through reuse their rings).
    pub fn smem_bytes(&self, d: usize) -> usize {
        self.key_warps * 2 * self.stages * self.bkv * d * 2
    }

    /// The shared scratch the f32 score tile crosses to become the P·V
    /// operand where the core's accumulator cannot feed a product as it
    /// stands (RDNA3); nothing where it can.
    pub fn scratch_bytes(&self, target: &Target) -> usize {
        match target.mma.first() {
            Some(atom) if !atom.accumulator_feeds(target.wave) => self.bq * self.bkv * 4,
            _ => 0,
        }
    }
}

/// Which keys each query sees; the masks apply together. A query with no
/// visible key yields zeros, as the graph writes it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct AttnMask {
    /// Key `j` is visible to query `i` only when `j ≤ i`.
    pub causal: bool,
    /// Keys within `[i − left, i + right]` of query `i`.
    pub window: Option<(usize, usize)>,
    /// Query `i` sees the keys of its own chunk and the chunks before it.
    pub chunk: Option<Chunk>,
    /// A `[batch]` i32 parameter of valid key counts: keys at and past the
    /// count are hidden. A count of 0 sees key 0 unless rows are appended,
    /// so a lane always has a key and a finite output.
    pub key_lens: bool,
    /// A `[batch, tk rounded up to 8]` i32 parameter, nonzero where the key
    /// is visible.
    pub key_mask: bool,
    /// A `[batch, t]` i32 parameter, non-decreasing along `t`: keys before
    /// `seg_start[i]` are hidden from query `i` (packed rows' segment starts).
    pub seg_start: bool,
    /// An additive score bias in the stream type, `scores·scale + bias`
    /// before the masks hide anything; the [`Bias`] rows say whose.
    pub bias: Option<Bias>,
}

/// Block attention: position `p` lies in chunk `p / size`, and a query sees
/// the keys of its own chunk, of the `left` chunks before it (every earlier
/// chunk without a bound) and of the `right` chunks after it. Positions are
/// the query and key indices, so the queries' and keys' sequences must be the
/// same one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Chunk {
    pub size: usize,
    pub left: Option<usize>,
    pub right: usize,
}

/// The rows of the `bias` parameter, `[rows, heads, t, key_mask_stride(keys)]`
/// with `keys` the scored keys: `tk`, plus the appended rows (their columns
/// last).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Bias {
    /// One row every batch shares.
    Shared,
    /// A row per batch.
    PerBatch,
}

/// Keys and values read from a cache of `[rows, tk, heads_total, d]`, the
/// `kv_heads` heads from a `[1]` i32 `head_start` parameter on: a layer's
/// slice is a runtime value, so every layer runs the one program.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Cache {
    pub rows: usize,
    pub heads_total: usize,
    /// A `[batch]` i32 parameter: the cache row each batch reads.
    pub row_map: bool,
    /// Rows of `[batch, appended, kv_heads, d]` key and value parameters
    /// scored after the cached keys (the tokens a step projected), at most a
    /// key block of them; 0 appends nothing.
    pub appended: usize,
    /// A `[batch]` i32 parameter of visible appended rows: the rest are
    /// hidden. Without it every appended row is seen.
    pub app_lens: bool,
}

/// The row stride of the `key_mask` and `bias` parameters: 8-element runs
/// stay aligned.
pub fn key_mask_stride(tk: usize) -> usize {
    tk.next_multiple_of(8)
}

/// `[batch, t, heads, d]` queries against `[batch, tk, kv_heads, d]` keys and
/// values; query head `h` reads KV head `h / (heads / kv_heads)`.
#[derive(Clone, Debug, PartialEq)]
pub struct AttnSpec {
    pub batch: Batch,
    pub t: usize,
    pub tk: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub d: usize,
    pub mask: AttnMask,
    pub cache: Option<Cache>,
    pub scale: f32,
    pub cfg: FaCfg,
}

impl AttnSpec {
    /// Parameters after the outputs, in order.
    pub fn extra_params(&self) -> Vec<&'static str> {
        let AttnMask { key_lens, key_mask, seg_start, .. } = self.mask;
        let cache =
            self.cache.unwrap_or(Cache { rows: 0, heads_total: 0, row_map: false, appended: 0, app_lens: false });
        [
            (key_lens, "key_lens"),
            (key_mask, "key_mask"),
            (seg_start, "seg_start"),
            (cache.row_map, "row_map"),
            (self.cache.is_some(), "head_start"),
            (cache.appended > 0, "k_app"),
            (cache.appended > 0, "v_app"),
            (cache.app_lens, "app_lens"),
            (self.mask.bias.is_some(), "bias"),
        ]
        .into_iter()
        .filter(|(on, _)| *on)
        .map(|(_, name)| name)
        .collect()
    }
}

/// The scores of a key block starting at the scalar, masked.
type Mask<'a> = &'a dyn Fn(&mut Kernel, Regs<F32>, Sc) -> Regs<F32>;

/// The online-softmax step over one key block: `q_r` against `k_t`, the
/// scores (plus `bias(block_start)`, in the scaled log2 units) masked by
/// `mask(block_start)`, folded into `(m, l, o)` with `v_t`. The running
/// maximum `m` is in the raw scores' units when [`raw_max`](Self::raw_max)
/// holds, else in the scaled log2 units of the exponent.
struct Step<'a, T> {
    q_r: Regs<T>,
    scale_log2e: f64,
    shape: Shape,
    bias: Option<Mask<'a>>,
    mask: Mask<'a>,
    cfg: FaCfg,
}

impl<T> Step<'_, T> {
    /// Whether the running maximum is kept in the raw scores' units: the
    /// planned form, where a positive scale commutes with the maximum and no
    /// bias joins the scores before it.
    fn raw_max(&self) -> bool {
        self.cfg.forms.raw_max && self.bias.is_none() && self.scale_log2e > 0.0
    }
}

/// Numeric: how far (in log2 units) a row's running maximum may fall behind
/// its scores before the row rescales (FlashAttention-4's lazy rescale): the
/// probabilities of a block stay below 2^8. The stream type bounds the
/// choice rather than fixing it: f16 holds up to 2^15, bf16 up to 2^127, and
/// the f32 output accumulates `p·v` over every key, so the margin left is
/// what keeps that sum finite for any value magnitude.
const RESCALE_ABOVE: f64 = 8.0;

fn step<T: Elem, PK: TierMark, PV: TierMark>(
    k: &mut Kernel,
    st: &Step<'_, T>,
    k_t: Tile<PK, T>,
    v_t: Tile<PV, T>,
    block_start: Sc,
    [m, l, o]: [Regs<F32>; 3],
) -> [Regs<F32>; 3] {
    let zero = k.zeros::<F32>(st.shape);
    let s = k.mma(zero, st.q_r, false, k_t, true);
    let scale = k.fill::<F32>(st.shape, Const::Float(st.scale_log2e));
    let rows = k.shape(m);
    // With the maximum in raw units the raw scores are masked and reduced,
    // and the scale reaches only the two row values the exponent and the
    // correction take: the exponent's `s·scale − m·scale` is one fma per
    // score. Else the scores are scaled first (which also leaves them
    // canonical for the compiler, sparing the quieting of a matrix core's
    // results before `maxnum`), and a bias joins them before the mask.
    let (s, block_max) = if st.raw_max() {
        let s = (st.mask)(k, s, block_start);
        let block_max = k.reduce(s, Axis::Row, ReduceOp::Max);
        (k.binary(s, scale, BinaryOp::Mul), block_max)
    } else {
        let s = k.binary(s, scale, BinaryOp::Mul);
        let s = st.bias.map_or(s, |bias| bias(k, s, block_start.clone()));
        let s = (st.mask)(k, s, block_start);
        (s, k.reduce(s, Axis::Row, ReduceOp::Max))
    };
    let unit = if st.raw_max() { st.scale_log2e } else { 1.0 };
    let scaled = |k: &mut Kernel, x: Regs<F32>| {
        if !st.raw_max() {
            return x;
        }
        let row_scale = k.fill::<F32>(rows, Const::Float(st.scale_log2e));
        k.binary(x, row_scale, BinaryOp::Mul)
    };
    // A row adopts its new maximum only when it grows by more than
    // `RESCALE_ABOVE` (log2 units), else keeps the old one: its correction is
    // then exactly 1 (and the accumulator's rescale may be skipped), while its
    // probabilities stay below 2^RESCALE_ABOVE. `o` and `l` share the stale
    // maximum, so their quotient is the same.
    let m_max = k.binary(m, block_max, BinaryOp::Max);
    let growth = k.binary(m_max, m, BinaryOp::Sub);
    let limit = k.fill::<F32>(rows, Const::Float(RESCALE_ABOVE / unit));
    let grows = k.compare(limit, growth, BinaryOp::Lt);
    let m_new = k.where_(grows, m_max, m);
    let corr = k.binary(m, m_new, BinaryOp::Sub);
    let corr = scaled(k, corr);
    let corr = k.unary(corr, UnaryOp::Exp2);
    let m_exp = scaled(k, m_new);
    let p = k.binary(s, m_exp, BinaryOp::Sub);
    let p = k.unary(p, UnaryOp::Exp2);
    let truncate = st.cfg.forms.truncate_p && T::DTYPE == svod_dtype::ScalarDType::BFloat16;
    let p16 = if truncate { k.cast_toward_zero::<T>(p) } else { k.cast::<F32, T>(p) };
    // Truncated probabilities are summed as held, so the normalization stays
    // unbiased; rounded ones are summed in f32. Where the sum goes is
    // `sum_after`'s.
    let summed = |k: &mut Kernel| {
        let block_sum =
            if truncate { k.reduce_f32(p16, Axis::Row, ReduceOp::Sum) } else { k.reduce(p, Axis::Row, ReduceOp::Sum) };
        let l = k.binary(l, corr, BinaryOp::Mul);
        k.binary(l, block_sum, BinaryOp::Add)
    };
    let early = (!st.cfg.forms.sum_after).then(|| summed(k));
    let o = if st.cfg.forms.skip_rescale { k.scale(o, corr) } else { k.binary(o, corr, BinaryOp::Mul) };
    let o = k.mma(o, p16, false, v_t, false);
    let l = early.unwrap_or_else(|| summed(k));
    [m_new, l, o]
}

/// `o = softmax(q·kᵀ·scale + bias)·v`. Parameters in order: `q`, `k`, `v`, then `o`
/// (one split) or the f32 partials `o_part [splits, batch, t, heads, d]`,
/// `m_part` and `l_part [splits, batch, t, heads]`, then
/// [`AttnSpec::extra_params`]. Key blocks no query of the tile sees are
/// skipped through the first block and the trip count, the rest shared
/// evenly by the splits; only blocks a mask edge crosses pay for the mask,
/// except under a key mask or a bias, which every block reads.
pub fn attention<T: Elem>(spec: &AttnSpec) -> Program {
    let AttnSpec { ref batch, t, tk, heads, kv_heads, d, mask, cache, scale, cfg } = *spec;
    let AttnMask { causal, window, chunk, key_lens, key_mask, seg_start, bias } = mask;
    let FaCfg { bq, bkv, stages, splits, .. } = cfg;
    let cap = batch.capacity();
    let (kv_rows, heads_total) = cache.map_or((cap, kv_heads), |c| (c.rows, c.heads_total));
    let appended = cache.map_or(0, |c| c.appended);
    assert!(appended <= bkv, "{appended} appended rows overflow a key block of {bkv}");
    let mut k = Kernel::new("flash_attention");
    let q = k.param::<T>("q", ParamKind::In, cap * t * heads * d);
    let kk = k.param::<T>("k", ParamKind::In, kv_rows * tk * heads_total * d);
    let v = k.param::<T>("v", ParamKind::In, kv_rows * tk * heads_total * d);
    let o = (splits == 1).then(|| k.param::<T>("o", ParamKind::Out, cap * t * heads * d));
    let parts = (splits > 1).then(|| {
        let o = k.param::<F32>("o_part", ParamKind::Out, splits * cap * t * heads * d);
        let m = k.param::<F32>("m_part", ParamKind::Out, splits * cap * t * heads);
        let l = k.param::<F32>("l_part", ParamKind::Out, splits * cap * t * heads);
        (o, m, l)
    });
    let lens = key_lens.then(|| k.param::<I32>("key_lens", ParamKind::In, cap));
    let mask_stride = key_mask_stride(tk);
    let masks = key_mask.then(|| k.param::<I32>("key_mask", ParamKind::In, cap * mask_stride));
    let segs = seg_start.then(|| k.param::<I32>("seg_start", ParamKind::In, cap * t));
    let row_map = cache.is_some_and(|c| c.row_map).then(|| k.param::<I32>("row_map", ParamKind::In, cap));
    let head_start = cache.is_some().then(|| k.param::<I32>("head_start", ParamKind::In, 1));
    let app = (appended > 0).then(|| {
        let k_app = k.param::<T>("k_app", ParamKind::In, cap * appended * kv_heads * d);
        let v_app = k.param::<T>("v_app", ParamKind::In, cap * appended * kv_heads * d);
        (k_app, v_app)
    });
    let app_lens = cache.is_some_and(|c| c.app_lens).then(|| k.param::<I32>("app_lens", ParamKind::In, cap));
    let keys = tk + appended;
    let bias_stride = key_mask_stride(keys);
    let bias = bias.map(|rows| {
        let n = if rows == Bias::Shared { 1 } else { cap };
        (k.param::<T>("bias", ParamKind::In, n * heads * t * bias_stride), rows)
    });
    let (gz, bb) = batch.axis(&mut k);
    k.grid([Sc::from(t.div_ceil(bq) * splits), Sc::from(heads), gz]);
    let kw = cfg.key_warps;
    assert!(kw == 1 || bq == WARP_ROWS, "{kw} key warps share a block of one row warp, not {bq} rows");
    // A warp's merged output takes 16·d f32 of its 16-bit K ring.
    assert!(kw == 1 || stages * bkv >= 2 * WARP_ROWS, "{stages} stages of {bkv} keys hold no key warp's output");
    k.warps(cfg.warps());
    let k_s = k.smem::<T>("k_s", kw * stages * bkv * d);
    let v_s = k.smem::<T>("v_s", kw * stages * bkv * d);

    let (order, split, hh) = (k.block(0) / splits, k.block(0) % splits, k.block(1));
    // Where later queries see more keys (a causal mask, chunks with every
    // earlier one) the last query blocks run first, so the grid's tail is its
    // shortest blocks.
    let descending = causal || chunk.is_some_and(|c| c.left.is_none());
    let qb = if descending { Sc::from(t.div_ceil(bq) - 1) - order } else { order };
    let q_off = qb * bq;
    let b = bb.clone().unwrap_or(Sc::from(0));
    // Keys past the length are masked; keys past `tk` too when blocks overhang it.
    let keys_masked = key_lens || !tk.is_multiple_of(bkv);
    let len = match lens {
        Some(lens) if appended > 0 => k.load_scalar(lens, b.clone()).min(tk),
        Some(lens) => k.load_scalar(lens, b.clone()).min(tk),
        None => Sc::from(tk),
    };
    let (q_stride, kv_stride) = (heads * d, heads_total * d);
    let rows = |k: &mut Kernel, p: ParamRef<T>| {
        let v = k.view(
            p,
            batch_offset(&bb, t * q_stride) + hh.clone() * d,
            [q_stride, 1],
            Shape::new(bq, d),
            [bound(t, bq), None],
        );
        k.at(v, q_off.clone(), 0)
    };
    let q_view = rows(&mut k, q);
    let kv_row = match row_map {
        Some(map) => k.load_scalar(map, b.clone()),
        None => b.clone(),
    };
    let kv_head = hh.clone() / (heads / kv_heads);
    let head_start = head_start.map_or(Sc::from(0), |p| k.load_scalar(p, 0));
    let kv_base = kv_row * (tk * kv_stride) + (kv_head.clone() + head_start) * d;
    let kv_bound = keys_masked.then(|| len.clone());
    let k_view = k.view(kk, kv_base.clone(), [kv_stride, 1], Shape::new(bkv, d), [kv_bound.clone(), None]);
    let v_view = k.view(v, kv_base, [kv_stride, 1], Shape::new(bkv, d), [kv_bound, None]);
    let mask_view = masks.map(|m| {
        let view = k.view(m, b.clone() * mask_stride, [mask_stride, 1], Shape::new(1, bkv), [None, bound(tk, bkv)]);
        k.at(view, 0, 0)
    });
    // Past `keys` (the appended columns' tile always overhangs them) reads nothing.
    let bias_view = bias.map(|(p, rows)| {
        let batch = if rows == Bias::Shared { Sc::from(0) } else { batch_offset(&bb, heads * t * bias_stride) };
        let cols = if appended > 0 { Some(Sc::from(keys)) } else { bound(keys, bkv) };
        let bounds = [bound(t, bq), cols];
        k.view(p, batch + hh.clone() * (t * bias_stride), [bias_stride, 1], Shape::new(bq, bkv), bounds)
    });
    // The tile's segment starts, loaded once; the mask is monotonic, so the
    // first and last rows bound the tile's.
    let seg = segs.map(|s| {
        let view = k.view(s, b.clone() * t, [1, 1], Shape::new(bq, 1), [bound(t, bq), None]);
        let tile = k.at(view, q_off.clone(), 0);
        let last = (q_off.clone() + bq - 1).min(t - 1);
        (k.load(tile), k.load_scalar(s, b.clone() * t + q_off.clone()), k.load_scalar(s, b.clone() * t + last))
    });

    // The tile's first and last query chunks; the first sees the most
    // keys ahead, the last the fewest behind.
    let chunks = chunk.map(|c| (q_off.clone() / c.size, (q_off.clone() + (bq - 1)) / c.size));
    // Key blocks `first..end` hold every key some query of the tile sees;
    // this split takes an even share of them.
    let mut first = Sc::from(0);
    if let Some((left, _)) = window {
        first = first.max((q_off.clone().max(left) - left) / bkv);
    }
    if let (Some(Chunk { size, left: Some(left), .. }), Some((cq0, _))) = (chunk, &chunks) {
        first = first.max((cq0.clone().max(left) - left) * size / bkv);
    }
    if let Some((_, seg_first, _)) = &seg {
        first = first.max(seg_first.clone() / bkv);
    }
    let mut end = Sc::from(tk.div_ceil(bkv));
    if causal {
        end = end.min((q_off.clone() + bq + bkv - 1) / bkv);
    }
    if let Some((_, right)) = window {
        end = end.min((q_off.clone() + bq + right + bkv - 1) / bkv);
    }
    if let (Some(Chunk { size, right, .. }), Some((_, cq1))) = (chunk, &chunks) {
        end = end.min(((cq1.clone() + (1 + right)) * size + (bkv - 1)) / bkv);
    }
    if key_lens {
        end = end.min((len.clone() + bkv - 1) / bkv);
    }
    let n = end - first.clone();
    let s_first = first.clone() + n.clone() * split.clone() / splits;
    let s_end = first + n * (split.clone() + 1) / splits;
    let blocks = s_end - s_first.clone();

    let shape = Shape::new(bq, bkv);
    let q_r = k.load(q_view);
    let m0 = k.fill::<F32>(Shape::new(bq, 1), Const::Float(-1e30));
    let l0 = k.zeros::<F32>(Shape::new(bq, 1));
    let o0 = k.zeros::<F32>(Shape::new(bq, d));
    let scale_log2e = (scale * std::f32::consts::LOG2_E) as f64;
    // The key coordinates of a block's scores and the query rows'.
    let coords = |k: &mut Kernel, block_start: Sc| {
        let col = k.coord(shape, Axis::Col);
        let start = k.splat::<I32>(shape, block_start);
        let col = k.binary(col, start, BinaryOp::Add);
        let row = k.coord(shape, Axis::Row);
        let q_off = k.splat::<I32>(shape, q_off.clone());
        (col, k.binary(row, q_off, BinaryOp::Add))
    };
    // The bias tile of the key block at `col`, in the scores' log2 units.
    let biased = |k: &mut Kernel, s: Regs<F32>, col: Sc| -> Regs<F32> {
        let Some(view) = bias_view else { return s };
        let tile = k.at(view, q_off.clone(), col);
        let b = load_f32(k, tile);
        let log2e = k.fill::<F32>(shape, Const::Float(std::f64::consts::LOG2_E));
        let b = k.binary(b, log2e, BinaryOp::Mul);
        k.binary(s, b, BinaryOp::Add)
    };
    let hide = |k: &mut Kernel, keep: Regs<Bool>, plain: Regs<F32>| {
        let hidden = k.fill::<F32>(shape, Const::Float(f64::NEG_INFINITY));
        k.where_(keep, plain, hidden)
    };
    let masked = |k: &mut Kernel, plain: Regs<F32>, block_start: Sc| {
        let (col, row) = coords(k, block_start.clone());
        let mut keep: Option<Regs<Bool>> = None;
        let mut and = |k: &mut Kernel, cond| {
            keep = Some(match keep {
                Some(keep) => k.binary(keep, cond, BinaryOp::And),
                None => cond,
            });
        };
        if causal {
            let cond = k.compare(col, row, BinaryOp::Le);
            and(k, cond);
        }
        if keys_masked {
            let len = k.splat::<I32>(shape, len.clone());
            let cond = k.compare(col, len, BinaryOp::Lt);
            and(k, cond);
        }
        if let Some((left, right)) = window {
            let left = k.splat::<I32>(shape, left);
            let reach = k.binary(col, left, BinaryOp::Add);
            let cond = k.compare(row, reach, BinaryOp::Le);
            and(k, cond);
            let right = k.splat::<I32>(shape, right);
            let reach = k.binary(row, right, BinaryOp::Add);
            let cond = k.compare(col, reach, BinaryOp::Le);
            and(k, cond);
        }
        if let Some(Chunk { size, left, right }) = chunk {
            let size = k.splat::<I32>(shape, size);
            let cq = k.binary(row, size, BinaryOp::Div);
            let ck = k.binary(col, size, BinaryOp::Div);
            let right = k.splat::<I32>(shape, right);
            let reach = k.binary(cq, right, BinaryOp::Add);
            let cond = k.compare(ck, reach, BinaryOp::Le);
            and(k, cond);
            if let Some(left) = left {
                let left = k.splat::<I32>(shape, left);
                let reach = k.binary(ck, left, BinaryOp::Add);
                let cond = k.compare(cq, reach, BinaryOp::Le);
                and(k, cond);
            }
        }
        if let Some(view) = mask_view {
            let tile = k.at(view, 0, block_start.clone());
            let visible = k.load(tile);
            // Against a full-shape zero: the predicate takes the tile's shape.
            let zero = k.zeros::<I32>(shape);
            let cond = k.compare(visible, zero, BinaryOp::Ne);
            and(k, cond);
        }
        if let Some((seg, _, _)) = &seg {
            let cond = k.compare(*seg, col, BinaryOp::Le);
            and(k, cond);
        }
        hide(k, keep.expect("a mask"), plain)
    };
    let edges = causal || keys_masked || window.is_some() || chunk.is_some() || seg.is_some();
    let mask = |k: &mut Kernel, s: Regs<F32>, block_start: Sc| -> Regs<F32> {
        if key_mask {
            return masked(k, s, block_start);
        }
        if !edges {
            return s;
        }
        let block_end = block_start.clone() + bkv;
        // Whether a mask edge crosses this block.
        let mut partial = Sc::from(0);
        if causal {
            partial = partial.or(q_off.clone().lt(block_end.clone()));
        }
        if keys_masked {
            partial = partial.or(len.clone().lt(block_end.clone()));
        }
        if let Some((left, right)) = window {
            let left_edge = (block_start.clone() + left).lt(q_off.clone() + bq - 1);
            let right_edge = (q_off.clone() + right + 1).lt(block_end.clone());
            partial = partial.or(left_edge).or(right_edge);
        }
        if let (Some(Chunk { size, left, right }), Some((cq0, cq1))) = (chunk, &chunks) {
            // Every row sees a block within the last row's reach behind and
            // the first row's reach ahead.
            partial = partial.or(((cq0.clone() + (1 + right)) * size).lt(block_end.clone()));
            if let Some(left) = left {
                partial = partial.or(block_start.clone().lt((cq1.clone().max(left) - left) * size));
            }
        }
        if let Some((_, _, seg_last)) = &seg {
            partial = partial.or(block_start.clone().lt(seg_last.clone()));
        }
        let [out] = k.select_if(partial, |k| [masked(k, s, block_start.clone())], |_| [s]);
        out
    };
    let bias_tile = bias_view.is_some().then_some(&biased as Mask);
    let st = Step { q_r, scale_log2e, shape, bias: bias_tile, mask: &mask, cfg };
    let app_views = app.map(|(k_app, v_app)| {
        let app = |k: &mut Kernel, p: ParamRef<T>| {
            k.view(
                p,
                batch_offset(&bb, appended * kv_heads * d) + kv_head.clone() * d,
                [kv_heads * d, 1],
                Shape::new(bkv, d),
                [Some(Sc::from(appended)), None],
            )
        };
        let (k_g, v_g) = (app(&mut k, k_app), app(&mut k, v_app));
        let visible = match app_lens {
            Some(p) => k.load_scalar(p, b.clone()).min(appended),
            None => Sc::from(appended),
        };
        (k_g, v_g, visible)
    });
    // One warp's share of the split's key blocks (`warp` is `None` for the
    // whole share), folded into `(m, l, o)`, the maximum in the exponent's
    // units at the end.
    let run = |k: &mut Kernel, warp: Option<Sc>| -> [Regs<F32>; 3] {
        let (w_first, w_blocks, slot0, owner) = match &warp {
            Some(w) => (
                s_first.clone() + blocks.clone() * w.clone() / kw,
                blocks.clone() * (w.clone() + 1) / kw - blocks.clone() * w.clone() / kw,
                w.clone() * stages,
                split.clone().eq(splits - 1).and(w.clone().eq(kw - 1)),
            ),
            None => (s_first.clone(), blocks.clone(), Sc::from(0), split.clone().eq(splits - 1)),
        };
        let [m, l, acc] = k.pipeline(
            w_blocks,
            stages,
            [m0, l0, o0],
            |k, step, slot| {
                let kv_off = (w_first.clone() + step) * bkv;
                let k_g = k.at(k_view, kv_off.clone(), 0);
                let v_g = k.at(v_view, kv_off, 0);
                let k_t = k.smem_slot::<T>(k_s, slot0.clone() + slot.clone(), Shape::new(bkv, d));
                let v_t = k.smem_slot::<T>(v_s, slot0.clone() + slot, Shape::new(bkv, d));
                k.stage(k_t, k_g, CopyMode::Async);
                k.stage(v_t, v_g, CopyMode::Async);
            },
            |k, step_, slot, carried| {
                let k_t = k.smem_slot::<T>(k_s, slot0.clone() + slot.clone(), Shape::new(bkv, d));
                let v_t = k.smem_slot::<T>(v_s, slot0.clone() + slot, Shape::new(bkv, d));
                step(k, &st, k_t, v_t, (w_first.clone() + step_) * bkv, carried)
            },
        );
        // The appended rows, a block of keys that every split and key warp
        // scores and all but the last hide: a hidden block leaves `(m, l, o)`
        // as they were, and a branch would cost more than the step.
        let [m, l, acc] = match &app_views {
            Some((k_g, v_g, visible)) => {
                let app_bias = |k: &mut Kernel, s: Regs<F32>, _: Sc| biased(k, s, Sc::from(tk));
                let rows = |k: &mut Kernel, s: Regs<F32>, _: Sc| {
                    let col = k.coord(shape, Axis::Col);
                    let visible = k.splat::<I32>(shape, visible.clone());
                    let seen = k.compare(col, visible, BinaryOp::Lt);
                    let zero = k.zeros::<I32>(shape);
                    let owner = k.splat::<I32>(shape, owner.clone());
                    let owned = k.compare(owner, zero, BinaryOp::Ne);
                    let keep = k.binary(seen, owned, BinaryOp::And);
                    hide(k, keep, s)
                };
                let bias_tile = bias_view.is_some().then_some(&app_bias as Mask);
                let st = Step { q_r, scale_log2e, shape, bias: bias_tile, mask: &rows, cfg };
                step(k, &st, *k_g, *v_g, Sc::from(0), [m, l, acc])
            }
            None => [m, l, acc],
        };
        let m = if st.raw_max() {
            let row_scale = k.fill::<F32>(Shape::new(bq, 1), Const::Float(scale_log2e));
            k.binary(m, row_scale, BinaryOp::Mul)
        } else {
            m
        };
        [m, l, acc]
    };
    let [m, l, acc] = match kw {
        1 => run(&mut k, None),
        // Every warp leaves its state over its own ring slots, dead once its
        // pipeline ends, as f32 views of the 16-bit rings: the output over
        // its key slots, the maximum and sum over its value slots; the block
        // then folds the regions as the split merge does.
        _ => {
            let region = stages * bkv * d * T::DTYPE.bytes() / F32::DTYPE.bytes();
            let views = |k: &mut Kernel, at: Sc| {
                let o = k.smem_view_as::<F32>(k_s, at.clone() * region, Shape::new(WARP_ROWS, d));
                let m = k.smem_view_as::<F32>(v_s, at.clone() * region, Shape::new(WARP_ROWS, 1));
                let l = k.smem_view_as::<F32>(v_s, at * region + WARP_ROWS, Shape::new(WARP_ROWS, 1));
                (o, m, l)
            };
            k.each_warp(|k| {
                let w = k.warp();
                let [m, l, acc] = run(k, Some(w.clone()));
                let (o_v, m_v, l_v) = views(k, w);
                k.store(o_v, acc);
                k.store(m_v, m);
                k.store(l_v, l);
            });
            let parts: Vec<State> = (0..kw)
                .map(|w| {
                    let (o_v, m_v, l_v) = views(&mut k, Sc::from(w));
                    State { m: k.load(m_v), l: k.load(l_v), o: k.load(o_v) }
                })
                .collect();
            let State { m, l, o } = merged(&mut k, &parts);
            [m, l, o]
        }
    };
    match (o, parts) {
        (Some(o), _) => {
            let out = normalized(&mut k, acc, l, bq);
            let out = k.cast::<F32, T>(out);
            let o_view = rows(&mut k, o);
            k.store(o_view, out);
        }
        (None, Some((o_part, m_part, l_part))) => {
            let per_split = cap * t * heads;
            let o_view = k.view(
                o_part,
                split.clone() * (per_split * d) + batch_offset(&bb, t * q_stride) + hh.clone() * d,
                [q_stride, 1],
                Shape::new(bq, d),
                [bound(t, bq), None],
            );
            let o_view = k.at(o_view, q_off.clone(), 0);
            k.store(o_view, acc);
            for (p, val) in [(m_part, m), (l_part, l)] {
                let view = k.view(
                    p,
                    split.clone() * per_split + batch_offset(&bb, t * heads) + hh.clone(),
                    [heads, 1],
                    Shape::new(bq, 1),
                    [bound(t, bq), None],
                );
                let view = k.at(view, q_off.clone(), 0);
                k.store(view, val);
            }
        }
        (None, None) => unreachable!("an output"),
    }
    k.finish()
}

/// `[splits, batch, t, heads]` rows of partial results merged into
/// `o [batch, t, heads, d]`.
#[derive(Clone, Debug, PartialEq)]
pub struct CombineSpec {
    pub batch: Batch,
    pub t: usize,
    pub heads: usize,
    pub d: usize,
    pub splits: usize,
    pub cfg: NormCfg,
}

/// A partial online-softmax state: the maximum in the exponent's units, the
/// sum and the output accumulator over some of the keys.
struct State {
    m: Regs<F32>,
    l: Regs<F32>,
    o: Regs<F32>,
}

/// The state over every part's keys: each part rescaled to the largest
/// maximum and summed.
fn merged(k: &mut Kernel, parts: &[State]) -> State {
    let mut m = parts[0].m;
    for part in &parts[1..] {
        m = k.binary(m, part.m, BinaryOp::Max);
    }
    let (mut l, mut o) = (None, None);
    for part in parts {
        let w = k.binary(part.m, m, BinaryOp::Sub);
        let w = k.unary(w, UnaryOp::Exp2);
        let lw = k.binary(part.l, w, BinaryOp::Mul);
        let ow = k.binary(part.o, w, BinaryOp::Mul);
        l = Some(match l {
            Some(l) => k.binary(l, lw, BinaryOp::Add),
            None => lw,
        });
        o = Some(match o {
            Some(o) => k.binary(o, ow, BinaryOp::Add),
            None => ow,
        });
    }
    State { m, l: l.expect("a part"), o: o.expect("a part") }
}

/// `acc / l` row by row, where a row no key reached (its sum `l` is zero,
/// and so is its accumulator) comes out as zeros, as the graph writes it,
/// rather than as the NaN of a division by zero.
fn normalized(k: &mut Kernel, acc: Regs<F32>, l: Regs<F32>, rows: usize) -> Regs<F32> {
    let zero = k.zeros::<F32>(Shape::new(rows, 1));
    let one = k.fill::<F32>(Shape::new(rows, 1), Const::Float(1.0));
    let empty = k.compare(l, zero, BinaryOp::Eq);
    let l = k.where_(empty, one, l);
    k.binary(acc, l, BinaryOp::Div)
}

/// Parameters in order: `o_part`, `m_part`, `l_part` (f32, as the
/// attention kernel writes them), `o`.
pub fn combine<T: Elem>(spec: &CombineSpec) -> Program {
    let CombineSpec { ref batch, t, heads, d, splits, cfg: NormCfg { br } } = *spec;
    let (cap, rows) = (batch.capacity(), t * heads);
    let mut k = Kernel::new("combine_splits");
    let o_part = k.param::<F32>("o_part", ParamKind::In, splits * cap * rows * d);
    let m_part = k.param::<F32>("m_part", ParamKind::In, splits * cap * rows);
    let l_part = k.param::<F32>("l_part", ParamKind::In, splits * cap * rows);
    let o = k.param::<T>("o", ParamKind::Out, cap * rows * d);
    let (gz, bb) = batch.axis(&mut k);
    k.grid([Sc::from(rows.div_ceil(br)), Sc::from(1), gz]);
    k.warps(br as u32);
    let row0 = k.block(0) * br;
    let part = |k: &mut Kernel, p: ParamRef<F32>, split: usize, width: usize| {
        let base = Sc::from(split * cap * rows * width) + batch_offset(&bb, rows * width);
        let view = k.view(p, base, [width, 1], Shape::new(br, width), [bound(rows, br), None]);
        let tile = k.at(view, row0.clone(), 0);
        k.load(tile)
    };
    let parts: Vec<State> = (0..splits)
        .map(|s| State { m: part(&mut k, m_part, s, 1), l: part(&mut k, l_part, s, 1), o: part(&mut k, o_part, s, d) })
        .collect();
    let State { l, o: acc, .. } = merged(&mut k, &parts);
    let out = normalized(&mut k, acc, l, br);
    let out = k.cast::<F32, T>(out);
    let view = k.view(o, batch_offset(&bb, rows * d), [d, 1], Shape::new(br, d), [bound(rows, br), None]);
    let view = k.at(view, row0, 0);
    k.store(view, out);
    k.finish()
}
