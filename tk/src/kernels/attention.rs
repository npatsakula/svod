//! Flash attention forward: Q resident in registers, K/V streamed through a
//! shared ring, the online-softmax state `(m, l, o)` carried.

use super::rows::NormCfg;
use super::{Admits, Admitted, Batch, Spec, batch_offset, bound, load_f32};
use crate::atoms::{Core, Device};
use crate::build::*;
use crate::ir::*;
use crate::lower::Lowering;
use crate::ops::config::{AttentionProblem, Planner, Rejected};

/// The tune store keys by this text.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FaCfg {
    pub bq: usize,
    pub bkv: usize,
    /// The ring over the key blocks.
    pub schedule: Schedule,
    /// Key splits: each is a block along grid x writing a partial result
    /// that [`combine`] merges; `1` writes the output directly.
    pub splits: usize,
    pub forms: Forms,
    /// Warps sharing a block of one row-warp of queries (`bq` = [`WARP_ROWS`])
    /// by its keys: each walks its own range of key blocks through its own
    /// ring slots with its own softmax state, and the states merge through
    /// shared memory at the end. A one-row decoder step's block is otherwise
    /// a single wave waiting on every key block's load in turn; this many
    /// loads are in flight instead. `1` leaves the keys to one warp.
    pub key_warps: usize,
    /// Query rows a warp owns: [`WARP_ROWS`] (one atom row) or twice that,
    /// where every key and value fragment the warp reads from the ring
    /// feeds two atom rows of products at the price of twice the score and
    /// output accumulators per lane. The warp-level core's alone.
    pub warp_rows: usize,
    /// The matrix instruction both products issue on. On the warpgroup core
    /// each group of four row warps owns 64 query rows: the scores are a
    /// chain over the shared key rows (K-major), the output's a chain of the
    /// probabilities as they stand in registers over the shared value rows
    /// (MN-major); a head wider than a 128-byte row is kept as column panels.
    pub core: Core,
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
    /// A biased kernel reads each key block's bias tile straight from global
    /// memory, in the scores' layout, rather than streaming it through a ring
    /// slot beside K and V: the planned ring fills in 16-byte chunks and
    /// gathers in the layout's runs, which a core whose accumulator holds a
    /// column per lane (gfx11) reads two bytes at a time.
    pub bias_direct: bool,
}

/// The fewest query rows per warp: the M of every matrix core's atom, so a
/// row's scores and its reductions stay within the warp. A warp may own a
/// multiple ([`FaCfg::warp_rows`]); the planner refuses a core whose atom
/// differs ([`FaCfg::fits`](FaCfg::fits)).
pub(crate) const WARP_ROWS: usize = 16;

/// Search bound: the row warps a block of the warp-level core spreads over.
const MOST_ROW_WARPS: usize = 8;

/// The fewest rows a warp of a `bq`-row block owns on `core`, whose atom has
/// `rows`: one atom row (a warpgroup's rows are the instruction's), or on a
/// warp-level core as many as hold the block in [`MOST_ROW_WARPS`].
pub(crate) fn narrowest_warp_rows(core: Core, rows: usize, bq: usize) -> usize {
    match core {
        Core::Warp | Core::Wide => rows * (bq / (MOST_ROW_WARPS * rows)).max(1),
        Core::Group => WARP_ROWS,
    }
}

/// Why [`attention`] refuses a config, whatever the target: what its
/// ranking leaves out as one that does not lower
/// ([`crate::ops::config::Rejected::Lowers`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, snafu::Snafu)]
pub enum Invalid {
    #[snafu(display("{key_warps} key warps share a block of one row warp, not {bq} rows"))]
    KeyWarps { key_warps: usize, bq: usize },
    #[snafu(display("{stages} stages of {bkv} keys hold no key warp's output"))]
    KeyWarpOutput { stages: usize, bkv: usize },
}

impl FaCfg {
    /// Whether [`attention`] runs the config's key warps: several share a
    /// block of one row warp, and a warp's merged output takes `16·d` f32
    /// of its 16-bit K ring.
    pub fn shares_keys(&self) -> Result<(), Invalid> {
        let Self { bq, bkv, key_warps, .. } = *self;
        let stages = self.stages();
        if key_warps == 1 {
            return Ok(());
        }
        if bq != WARP_ROWS {
            return Err(Invalid::KeyWarps { key_warps, bq });
        }
        if stages * bkv < 2 * WARP_ROWS {
            return Err(Invalid::KeyWarpOutput { stages, bkv });
        }
        Ok(())
    }

    pub const fn new(bq: usize, bkv: usize, stages: usize) -> Self {
        Self {
            bq,
            bkv,
            schedule: Schedule::ring(stages, false),
            splits: 1,
            forms: Forms {
                truncate_p: false,
                sum_after: false,
                skip_rescale: false,
                raw_max: false,
                bias_direct: false,
            },
            key_warps: 1,
            warp_rows: WARP_ROWS,
            core: Core::Warp,
        }
    }

    /// One warp per [`warp_rows`](Self::warp_rows) query rows: the warps a
    /// tile is spread over.
    pub fn stages(&self) -> usize {
        self.schedule.stages()
    }

    pub fn unroll(&self) -> bool {
        self.schedule.unroll()
    }

    pub fn row_warps(&self) -> u32 {
        (self.bq / self.warp_rows) as u32
    }

    /// The block's warps: the row warps, each split [`key_warps`](Self::key_warps) ways.
    pub fn warps(&self) -> u32 {
        self.row_warps() * self.key_warps as u32
    }

    pub fn lowering(&self, device: &Device) -> Lowering {
        Lowering { core: self.core, ..Lowering::new(device) }
    }

    /// Whether the config holds `appended` rows: they ride one key block,
    /// read from global memory, which the warpgroup core's shared operands
    /// cannot.
    pub fn holds_appended(&self, appended: usize) -> bool {
        self.bkv >= appended && (appended == 0 || self.core != Core::Group)
    }

    /// Whether a bias rides the ring: under a bias, unless the forms read it
    /// directly.
    pub fn bias_ring(&self, bias: bool) -> bool {
        bias && !self.forms.bias_direct
    }

    /// Static shared memory in bytes: the K and V rings of a 16-bit type (and
    /// the bias ring, [`bias_ring`](Self::bias_ring)), one set per key warp
    /// (the states key warps merge through reuse their rings) — all of it:
    /// where the core's accumulator cannot feed the P·V product (RDNA3), the
    /// probabilities' relayout is pinned to the lane exchange
    /// ([`crate::layouts::TileLayout::relayout`]), so a kernel of the config
    /// declares no scratch for them. (A shared crossing, which the costs
    /// could choose before, the card may refuse: the Strix Halo's "local
    /// memory (73728) exceeds limit (65536)", a 32×128 block of head size 64
    /// whose rings alone are its whole 64 KB.)
    pub fn smem_bytes(&self, d: usize, bias: bool) -> usize {
        let slot = 2 * self.bkv * d + usize::from(self.bias_ring(bias)) * self.bq * self.bkv;
        self.key_warps * self.stages() * slot * 2
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
    /// Block attention ([`Chunk`]): query `i` sees the keys of its own chunk,
    /// of `left` chunks before it and of `right` after.
    pub chunk: Option<Chunk>,
    /// A `[batch]` i32 parameter of valid key counts: keys at and past the
    /// count are hidden, and a lane left with no visible key yields zeros.
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
pub struct CacheLayout {
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
pub(crate) fn key_mask_stride(tk: usize) -> usize {
    tk.next_multiple_of(8)
}

/// `[batch, t, heads, d]` queries against `[batch, tk, kv_heads, d]` keys and
/// values; query head `h` reads KV head `h / (heads / kv_heads)`. A kernel
/// runs it under a config admitted for it ([`AttnSpec::admitted`]).
#[derive(Clone, Debug, PartialEq)]
pub struct AttnSpec {
    pub batch: Batch,
    pub t: usize,
    pub tk: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub d: usize,
    pub mask: AttnMask,
    pub cache: Option<CacheLayout>,
    /// Queries, keys and values are one `[batch, t, heads + 2·kv_heads, d]`
    /// parameter, a fused projection read in place: its keys from head
    /// `heads` on, its values from `heads + kv_heads`. Without a cache, `tk`
    /// is `t`.
    pub packed: bool,
    pub scale: f32,
}

impl Spec for AttnSpec {
    type Cfg = FaCfg;
}

impl Admits for AttnSpec {
    type Problem = AttentionProblem;

    fn problem(&self) -> AttentionProblem {
        let AttnSpec { t, tk, d, mask, .. } = *self;
        let (causal, bias) = (mask.causal, mask.bias.is_some());
        AttentionProblem { tiles: self.batch.capacity() * self.heads, t, tk, d, causal, bias }
    }
}

impl AttnSpec {
    /// `cfg` [admitted](Admitted) for this attention on `planner`, or why it
    /// is not.
    pub fn admitted(self, planner: &Planner, cfg: FaCfg) -> Result<Admitted<'_, Self>, Rejected> {
        self.admit(planner, cfg)
    }

    /// Parameters after the outputs, in order.
    pub fn extra_params(&self) -> Vec<&'static str> {
        let AttnMask { key_lens, key_mask, seg_start, .. } = self.mask;
        let cache =
            self.cache.unwrap_or(CacheLayout { rows: 0, heads_total: 0, row_map: false, appended: 0, app_lens: false });
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
type Mask<'a, 'k> = &'a dyn Fn(&mut Kernel<'k>, Regs<'k, F32>, Sc<'k>) -> Regs<'k, F32>;

/// The online-softmax step over one key block: `q_r` against `k_t`, the
/// scores (plus `bias(block_start)`, in the scaled log2 units) masked by
/// `mask(block_start)`, folded into `(m, l, o)` with `v_t`. The running
/// maximum `m` is in the raw scores' units when [`raw_max`](Self::raw_max)
/// holds, else in the scaled log2 units of the exponent.
struct Step<'a, 'k, T> {
    q_r: Regs<'k, T>,
    scale_log2e: f64,
    shape: Shape,
    bias: Option<Mask<'a, 'k>>,
    mask: Mask<'a, 'k>,
    cfg: FaCfg,
}

impl<T> Clone for Step<'_, '_, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Step<'_, '_, T> {}

/// `s` plus a bias tile, in the scores' log2 units: one fused multiply-add,
/// so the result does not depend on which product the backend would fuse
/// into a separate add (the scaled scores' or the bias's), which varies with
/// the layout around it.
fn biased<'k, T: Elem, P: Readable>(k: &mut Kernel<'k>, s: Regs<'k, F32>, tile: Tile<'k, P, T>) -> Regs<'k, F32> {
    let b = load_f32(k, tile);
    let log2e = k.fill::<F32>(k.shape(s), Const::Float(std::f64::consts::LOG2_E));
    k.mul_add(b, log2e, s)
}

impl<T> Step<'_, '_, T> {
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

fn step<'k, T: Half, PK: Readable, PV: Readable>(
    k: &mut Kernel<'k>,
    st: &Step<'_, 'k, T>,
    k_t: Tile<'k, PK, T>,
    v_t: Tile<'k, PV, T>,
    block_start: Sc<'k>,
    [m, l, o]: [Acc<'k, F32>; 3],
) -> [Acc<'k, F32>; 3] {
    let zero = k.zeros::<F32>(st.shape);
    let s = *k.mma(zero, st.q_r, false, k_t, true);
    let scale = k.fill::<F32>(st.shape, Const::Float(st.scale_log2e));
    let (m, l) = (*m, *l);
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
    let scaled = |k: &mut Kernel<'k>, x: Regs<'k, F32>| {
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
    let grows = k.compare(limit, growth, CmpOp::Lt);
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
    let summed = |k: &mut Kernel<'k>| {
        let block_sum =
            if truncate { k.reduce_f32(p16, Axis::Row, ReduceOp::Sum) } else { k.reduce(p, Axis::Row, ReduceOp::Sum) };
        let l = k.binary(l, corr, BinaryOp::Mul);
        k.binary(l, block_sum, BinaryOp::Add)
    };
    let early = (!st.cfg.forms.sum_after).then(|| summed(k));
    let o = if st.cfg.forms.skip_rescale { k.scale(*o, corr) } else { k.binary(*o, corr, BinaryOp::Mul) };
    let o = k.mma(o, p16, false, v_t, false);
    let l = early.unwrap_or_else(|| summed(k));
    [m_new.into(), l.into(), o]
}

/// `o = softmax(q·kᵀ·scale + bias)·v`. Parameters in order: `q`, `k`, `v`
/// (one `qkv` [packed](AttnSpec::packed)), then `o`
/// (one split) or the f32 partials `o_part [splits, batch, t, heads, d]`,
/// `m_part` and `l_part [splits, batch, t, heads]`, then
/// [`AttnSpec::extra_params`]. Key blocks no query of the tile sees are
/// skipped through the first block and the trip count, the rest shared
/// evenly by the splits; only blocks a mask edge crosses pay for the mask,
/// except under a key mask or a bias, which every block reads.
pub fn attention<T: Half>(spec: &Admitted<AttnSpec>) -> Program {
    Kernel::record(if spec.packed { "flash_attention_packed" } else { "flash_attention" }, |k| {
        attention_body::<T>(k, spec)
    })
}

fn attention_body<'k, T: Half>(k: &mut Kernel<'k>, spec: &Admitted<AttnSpec>) {
    let AttnSpec { ref batch, t, tk, heads, kv_heads, d, mask, cache, packed, scale } = **spec;
    let AttnMask { causal, window, chunk, key_lens, key_mask, seg_start, bias } = mask;
    let cfg = spec.cfg();
    let FaCfg { bq, bkv, splits, .. } = cfg;
    let stages = cfg.stages();
    let cap = batch.capacity();
    let (kv_rows, heads_total) = cache.map_or((cap, kv_heads), |c| (c.rows, c.heads_total));
    let appended = cache.map_or(0, |c| c.appended);
    assert!(appended <= bkv, "{appended} appended rows overflow a key block of {bkv}");
    assert!(!packed || (cache.is_none() && tk == t), "a packed operand holds its own keys");
    let slots = heads + 2 * kv_heads;
    let (q, kk, v) = if packed {
        let qkv = k.input::<T>("qkv", cap * t * slots * d);
        (qkv, qkv, qkv)
    } else {
        let q = k.input::<T>("q", cap * t * heads * d);
        let kk = k.input::<T>("k", kv_rows * tk * heads_total * d);
        (q, kk, k.input::<T>("v", kv_rows * tk * heads_total * d))
    };
    let o = (splits == 1).then(|| k.output::<T>("o", cap * t * heads * d));
    let parts = (splits > 1).then(|| {
        let o = k.output::<F32>("o_part", splits * cap * t * heads * d);
        let m = k.output::<F32>("m_part", splits * cap * t * heads);
        let l = k.output::<F32>("l_part", splits * cap * t * heads);
        (o, m, l)
    });
    let lens = key_lens.then(|| k.input::<I32>("key_lens", cap));
    let mask_stride = key_mask_stride(tk);
    let masks = key_mask.then(|| k.input::<I32>("key_mask", cap * mask_stride));
    let segs = seg_start.then(|| k.input::<I32>("seg_start", cap * t));
    let row_map = cache.is_some_and(|c| c.row_map).then(|| k.input::<I32>("row_map", cap));
    let head_start = cache.is_some().then(|| k.input::<I32>("head_start", 1));
    let app = (appended > 0).then(|| {
        let k_app = k.input::<T>("k_app", cap * appended * kv_heads * d);
        let v_app = k.input::<T>("v_app", cap * appended * kv_heads * d);
        (k_app, v_app)
    });
    let app_lens = cache.is_some_and(|c| c.app_lens).then(|| k.input::<I32>("app_lens", cap));
    let keys = tk + appended;
    let bias_stride = key_mask_stride(keys);
    let bias = bias.map(|rows| {
        let n = if rows == Bias::Shared { 1 } else { cap };
        (k.input::<T>("bias", n * heads * t * bias_stride), rows)
    });
    let (gz, bb) = batch.axis(k);
    k.grid([Sc::from(t.div_ceil(bq) * splits), Sc::from(heads), gz]);
    let kw = cfg.key_warps;
    k.warps(cfg.row_warps());
    k.lone_warps(cfg.warps());
    let ring = Schedule::ring(stages, cfg.unroll());
    let k_s = k.ring_runs::<T>("k_s", bkv * d, ring, kw);
    let v_s = k.ring_runs::<T>("v_s", bkv * d, ring, kw);
    // The bias streams through a ring of its own beside K and V, unless the
    // forms read its tiles directly.
    let bias_s = cfg.bias_ring(bias.is_some()).then(|| k.ring_runs::<T>("bias_s", bq * bkv, ring, kw));

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
        Some(lens) => k.load_scalar(lens, b.clone()).min(tk),
        None => Sc::from(tk),
    };
    // The output's rows hold its heads alone; a packed operand's hold every slot.
    let o_stride = heads * d;
    let (q_stride, kv_stride) = if packed { (slots * d, slots * d) } else { (o_stride, heads_total * d) };
    // The packed operand's keys follow its queries, its values its keys.
    let (k_first, v_first) = if packed { (heads, heads + kv_heads) } else { (0, 0) };
    let rows = Rows { bb: bb.clone(), hh: hh.clone(), q_off: q_off.clone(), t, d, bq };
    let q_view = rows.of(k, q, q_stride);
    let kv_row = match row_map {
        Some(map) => k.load_scalar(map, b.clone()),
        None => b.clone(),
    };
    let kv_head = hh.clone() / (heads / kv_heads);
    let head_start = head_start.map_or(Sc::from(0), |p| k.load_scalar(p, 0));
    let kv_base = kv_row * (tk * kv_stride) + (kv_head.clone() + head_start) * d;
    let kv_bound = keys_masked.then(|| len.clone());
    // A packed operand's keys follow its queries, its values its keys; an
    // unpacked one's K and V start where they always did, with no offset to
    // add (a `+ 0` the emitted program would carry).
    let (k_base, v_base) = match (k_first, v_first) {
        (0, 0) => (kv_base.clone(), kv_base),
        (k_first, v_first) => (kv_base.clone() + k_first * d, kv_base + v_first * d),
    };
    let k_view = k.view(kk, k_base, [kv_stride, 1], Shape::new(bkv, d), [kv_bound.clone(), None]);
    let v_view = k.view(v, v_base, [kv_stride, 1], Shape::new(bkv, d), [kv_bound, None]);
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
    let coords = |k: &mut Kernel<'k>, block_start: Sc<'k>| {
        let col = k.coord(shape, Axis::Col);
        let start = k.splat::<I32>(shape, block_start);
        let col = k.binary(col, start, BinaryOp::Add);
        let row = k.coord(shape, Axis::Row);
        let q_off = k.splat::<I32>(shape, q_off.clone());
        (col, k.binary(row, q_off, BinaryOp::Add))
    };
    // The bias tile of the key block at `col` read from global memory: the
    // appended block's, outside the ring.
    let biased_at = |k: &mut Kernel<'k>, s: Regs<'k, F32>, col: Sc<'k>| -> Regs<'k, F32> {
        let Some(view) = bias_view else { return s };
        let tile = k.at(view, q_off.clone(), col);
        biased(k, s, tile)
    };
    let hide = |k: &mut Kernel<'k>, keep: Regs<'k, Bool>, plain: Regs<'k, F32>| {
        let hidden = k.fill::<F32>(shape, Const::Float(f64::NEG_INFINITY));
        k.where_(keep, plain, hidden)
    };
    let masked = |k: &mut Kernel<'k>, plain: Regs<'k, F32>, block_start: Sc<'k>| {
        let (col, row) = coords(k, block_start.clone());
        let mut keep: Option<Regs<'k, Bool>> = None;
        let mut and = |k: &mut Kernel<'k>, cond| {
            keep = Some(match keep {
                Some(keep) => k.binary(keep, cond, BinaryOp::And),
                None => cond,
            });
        };
        if causal {
            let cond = k.compare(col, row, CmpOp::Le);
            and(k, cond);
        }
        if keys_masked {
            let len = k.splat::<I32>(shape, len.clone());
            let cond = k.compare(col, len, CmpOp::Lt);
            and(k, cond);
        }
        if let Some((left, right)) = window {
            let left = k.splat::<I32>(shape, left);
            let reach = k.binary(col, left, BinaryOp::Add);
            let cond = k.compare(row, reach, CmpOp::Le);
            and(k, cond);
            let right = k.splat::<I32>(shape, right);
            let reach = k.binary(row, right, BinaryOp::Add);
            let cond = k.compare(col, reach, CmpOp::Le);
            and(k, cond);
        }
        if let Some(Chunk { size, left, right }) = chunk {
            let size = k.splat::<I32>(shape, size);
            let cq = k.binary(row, size, BinaryOp::Div);
            let ck = k.binary(col, size, BinaryOp::Div);
            let right = k.splat::<I32>(shape, right);
            let reach = k.binary(cq, right, BinaryOp::Add);
            let cond = k.compare(ck, reach, CmpOp::Le);
            and(k, cond);
            if let Some(left) = left {
                let left = k.splat::<I32>(shape, left);
                let reach = k.binary(ck, left, BinaryOp::Add);
                let cond = k.compare(cq, reach, CmpOp::Le);
                and(k, cond);
            }
        }
        if let Some(view) = mask_view {
            let tile = k.at(view, 0, block_start.clone());
            let visible = k.load(tile);
            // Against a full-shape zero: the predicate takes the tile's shape.
            let zero = k.zeros::<I32>(shape);
            let cond = k.compare(visible, zero, CmpOp::Ne);
            and(k, cond);
        }
        if let Some((seg, _, _)) = &seg {
            let cond = k.compare(*seg, col, CmpOp::Le);
            and(k, cond);
        }
        hide(k, keep.expect("a mask"), plain)
    };
    let edges = causal || keys_masked || window.is_some() || chunk.is_some() || seg.is_some();
    let mask = |k: &mut Kernel<'k>, s: Regs<'k, F32>, block_start: Sc<'k>| -> Regs<'k, F32> {
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
    // The global reader serves the direct form and stands in for the ring's
    // here, so the forms agree with the steps' (a bias keeps the scaled
    // maximum).
    let bias_tile = bias_view.is_some().then_some(&biased_at as Mask);
    let st = Step { q_r, scale_log2e, shape, bias: bias_tile, mask: &mask, cfg };
    let app_views = app.map(|(k_app, v_app)| {
        let app = |k: &mut Kernel<'k>, p: ParamRef<'k, T, In>| {
            k.view(
                p,
                batch_offset(&bb, appended * kv_heads * d) + kv_head.clone() * d,
                [kv_heads * d, 1],
                Shape::new(bkv, d),
                [Some(Sc::from(appended)), None],
            )
        };
        let (k_g, v_g) = (app(k, k_app), app(k, v_app));
        let visible = match app_lens {
            Some(p) => k.load_scalar(p, b.clone()).min(appended),
            None => Sc::from(appended),
        };
        (k_g, v_g, visible)
    });
    // One warp's share of the split's key blocks (`warp` is `None` for the
    // whole share), folded into `(m, l, o)`, the maximum in the exponent's
    // units at the end.
    let run = |k: &mut Kernel<'k>, warp: Option<Sc<'k>>| -> [Regs<'k, F32>; 3] {
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
            k_s,
            [m0, l0, o0],
            |k, step, slot| {
                let (kv_off, slot) = ((w_first.clone() + step) * bkv, slot.past(slot0.clone()));
                let k_g = k.at(k_view, kv_off.clone(), 0);
                let v_g = k.at(v_view, kv_off.clone(), 0);
                let k_t = k.slot(k_s, &slot, Shape::new(bkv, d));
                let v_t = k.slot(v_s, &slot, Shape::new(bkv, d));
                k.stage(k_t, k_g, CopyMode::Async);
                k.stage(v_t, v_g, CopyMode::Async);
                if let (Some(view), Some(ring)) = (bias_view, bias_s) {
                    let b_g = k.at(view, q_off.clone(), kv_off);
                    let b_t = k.slot(ring, &slot, shape);
                    k.stage(b_t, b_g, CopyMode::Async);
                }
            },
            |k, step_, slot, carried| {
                let slot = slot.past(slot0.clone());
                let k_t = k.slot(k_s, &slot, Shape::new(bkv, d));
                let v_t = k.slot(v_s, &slot, Shape::new(bkv, d));
                let b_t = bias_s.map(|ring| k.slot(ring, &slot, shape));
                let from_ring = |k: &mut Kernel<'k>, s, _| biased(k, s, b_t.expect("a bias ring"));
                let st = if b_t.is_some() { Step { bias: Some(&from_ring as Mask), ..st } } else { st };
                step(k, &st, k_t, v_t, (w_first.clone() + step_) * bkv, carried)
            },
        );
        // The appended rows, a block of keys that every split and key warp
        // scores and all but the last hide: a hidden block leaves `(m, l, o)`
        // as they were, and a branch would cost more than the step.
        let [m, l, acc] = match &app_views {
            Some((k_g, v_g, visible)) => {
                let app_bias = |k: &mut Kernel<'k>, s: Regs<'k, F32>, _: Sc<'k>| biased_at(k, s, Sc::from(tk));
                let rows = |k: &mut Kernel<'k>, s: Regs<'k, F32>, _: Sc<'k>| {
                    let col = k.coord(shape, Axis::Col);
                    let visible = k.splat::<I32>(shape, visible.clone());
                    let seen = k.compare(col, visible, CmpOp::Lt);
                    let zero = k.zeros::<I32>(shape);
                    let owner = k.splat::<I32>(shape, owner.clone());
                    let owned = k.compare(owner, zero, CmpOp::Ne);
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
            k.binary(*m, row_scale, BinaryOp::Mul)
        } else {
            *m
        };
        [m, *l, *acc]
    };
    let [m, l, acc] = match kw {
        1 => run(k, None),
        // Every warp leaves its state over its own ring slots, dead once its
        // pipeline ends, as f32 views of the 16-bit rings: the output over
        // its key slots, the maximum and sum over its value slots; the block
        // then folds the regions as the split merge does.
        _ => {
            let region = stages * bkv * d * T::DTYPE.bytes() / F32::DTYPE.bytes();
            // SAFETY: a warp's region is its own run of slots, and both uses
            // come after that warp's pipeline: the warp's own stores once its
            // pipeline ends, the block's loads after every warp's.
            let views = |k: &mut Kernel<'k>, at: Sc<'k>| unsafe {
                let o = k.ring_scratch::<F32>(k_s, at.clone() * region, Shape::new(WARP_ROWS, d));
                let m = k.ring_scratch::<F32>(v_s, at.clone() * region, Shape::new(WARP_ROWS, 1));
                let l = k.ring_scratch::<F32>(v_s, at * region + WARP_ROWS, Shape::new(WARP_ROWS, 1));
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
            let parts: Vec<State<'k>> = (0..kw)
                .map(|w| {
                    let (o_v, m_v, l_v) = views(k, Sc::from(w));
                    State { m: k.load(m_v), l: k.load(l_v), o: k.load(o_v) }
                })
                .collect();
            let State { m, l, o } = merged(k, &parts);
            [m, l, o]
        }
    };
    match (o, parts) {
        (Some(o), _) => {
            let out = normalized(k, acc, l, bq);
            let out = k.cast::<F32, T>(out);
            let o_view = rows.of(k, o, o_stride);
            k.store(o_view, out);
        }
        (None, Some((o_part, m_part, l_part))) => {
            let per_split = cap * t * heads;
            let o_view = k.view(
                o_part,
                split.clone() * (per_split * d) + batch_offset(&bb, t * o_stride) + hh.clone() * d,
                [o_stride, 1],
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
struct State<'k> {
    m: Regs<'k, F32>,
    l: Regs<'k, F32>,
    o: Regs<'k, F32>,
}

/// The state over every part's keys: each part rescaled to the largest
/// maximum and summed.
fn merged<'k>(k: &mut Kernel<'k>, parts: &[State<'k>]) -> State<'k> {
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
fn normalized<'k>(k: &mut Kernel<'k>, acc: Regs<'k, F32>, l: Regs<'k, F32>, rows: usize) -> Regs<'k, F32> {
    let zero = k.zeros::<F32>(Shape::new(rows, 1));
    let one = k.fill::<F32>(Shape::new(rows, 1), Const::Float(1.0));
    let empty = k.compare(l, zero, CmpOp::Eq);
    let l = k.where_(empty, one, l);
    k.binary(acc, l, BinaryOp::Div)
}

/// Parameters in order: `o_part`, `m_part`, `l_part` (f32, as the
/// attention kernel writes them), `o`.
pub fn combine<T: Elem>(spec: &CombineSpec) -> Program {
    Kernel::record("combine_splits", |k| combine_body::<T>(k, spec))
}

fn combine_body<'k, T: Elem>(k: &mut Kernel<'k>, spec: &CombineSpec) {
    let CombineSpec { ref batch, t, heads, d, splits, cfg: NormCfg { br } } = *spec;
    let (cap, rows) = (batch.capacity(), t * heads);
    let o_part = k.input::<F32>("o_part", splits * cap * rows * d);
    let m_part = k.input::<F32>("m_part", splits * cap * rows);
    let l_part = k.input::<F32>("l_part", splits * cap * rows);
    let o = k.output::<T>("o", cap * rows * d);
    let (gz, bb) = batch.axis(k);
    k.grid([Sc::from(rows.div_ceil(br)), Sc::from(1), gz]);
    k.warps(br as u32);
    let row0 = k.block(0) * br;
    let part = |k: &mut Kernel<'k>, p: ParamRef<'k, F32, In>, split: usize, width: usize| {
        let base = Sc::from(split * cap * rows * width) + batch_offset(&bb, rows * width);
        let view = k.view(p, base, [width, 1], Shape::new(br, width), [bound(rows, br), None]);
        let tile = k.at(view, row0.clone(), 0);
        k.load(tile)
    };
    let parts: Vec<State<'k>> = (0..splits)
        .map(|s| State { m: part(k, m_part, s, 1), l: part(k, l_part, s, 1), o: part(k, o_part, s, d) })
        .collect();
    let State { l, o: acc, .. } = merged(k, &parts);
    let out = normalized(k, acc, l, br);
    let out = k.cast::<F32, T>(out);
    let view = k.view(o, batch_offset(&bb, rows * d), [d, 1], Shape::new(br, d), [bound(rows, br), None]);
    let view = k.at(view, row0, 0);
    k.store(view, out);
}

/// The block's `[bq, d]` query rows of head `hh` in a `[batch·t, stride]`
/// parameter, read or written.
struct Rows<'k> {
    bb: Option<Sc<'k>>,
    hh: Sc<'k>,
    q_off: Sc<'k>,
    t: usize,
    d: usize,
    bq: usize,
}

impl<'k> Rows<'k> {
    fn of<T: Elem, A: Access>(&self, k: &mut Kernel<'k>, p: ParamRef<'k, T, A>, stride: usize) -> Gmem<'k, T, A> {
        let Rows { bb, hh, q_off, t, d, bq } = self;
        let offset = batch_offset(bb, t * stride) + hh.clone() * *d;
        let v = k.view(p, offset, [stride, 1], Shape::new(*bq, *d), [bound(*t, *bq), None]);
        k.at(v, q_off.clone(), 0)
    }
}
