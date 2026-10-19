//! Flash attention: the program against a direct softmax reference on the
//! host, and the lowered kernel against the program on the GPU.

use svod_dtype::{DType, ScalarDType};
use svod_tensor::Tensor;
use test_case::test_case;

use super::declared;
use crate::atoms::Target;
use crate::build::BF16;
use crate::interp::{round_to, run};
use crate::ir::Program;
use crate::kernels::Batch;
use crate::kernels::attention::{
    AttnMask, AttnSpec, Bias, Cache, Chunk, CombineSpec, FaCfg, Forms, WARP_ROWS, attention, combine, key_mask_stride,
};
use crate::kernels::gemm::Core;
use crate::kernels::rows::NormCfg;
use crate::launch::graph_launch;
use crate::launch::graph_launch_all;
use crate::ops::config::Planner;

fn flash_attention(spec: &AttnSpec) -> Program {
    attention::<BF16>(spec)
}

/// Segment length of the packed rows the `seg_start` cases mask.
const SEGMENT: usize = 40;

struct Case {
    spec: AttnSpec,
    q: Vec<f64>,
    /// `[rows, tk, heads_total, d]`, the batch's own rows without a cache.
    k: Vec<f64>,
    v: Vec<f64>,
    lens: Vec<i64>,
    /// `[batch, key_mask_stride(tk)]`: key `j` of batch `b` hidden where `(b + j) % 5 == 2`.
    key_mask: Vec<i64>,
    /// `[batch, t]`: segments of `SEGMENT` rows.
    seg_start: Vec<i64>,
    /// `[batch]`: batch `b` reads cache row `rows - 1 - b`.
    row_map: Vec<i64>,
    /// The first cached head this attention reads.
    head_start: i64,
    /// `[batch, appended, kv_heads, d]` appended keys and values.
    k_app: Vec<f64>,
    v_app: Vec<f64>,
    /// `[batch]`: batch `b` sees `appended - b % 3` of them, at least one.
    app_lens: Vec<i64>,
    /// `[rows, heads, t, key_mask_stride(keys)]` in `2·[-1, 1)`, the padding
    /// columns past `keys` a large value nothing may read; empty unbiased.
    bias: Vec<f64>,
}

/// The rows appended per batch.
fn appended(spec: &AttnSpec) -> usize {
    spec.cache.map_or(0, |c| c.appended)
}

/// The scored keys: the cached ones and the appended rows.
fn keys(spec: &AttnSpec) -> usize {
    spec.tk + appended(spec)
}

fn case(spec: AttnSpec, lens: &[i64]) -> Case {
    let mut seed = 5;
    let n = |elems: usize, seed: &mut u64| super::rounded(ScalarDType::BFloat16, elems, seed, |x| x);
    let batch = spec.batch.capacity();
    let (rows, heads_total) = spec.cache.map_or((batch, spec.kv_heads), |c| (c.rows, c.heads_total));
    let (q, k, v) = (
        n(batch * spec.t * spec.heads * spec.d, &mut seed),
        n(rows * spec.tk * heads_total * spec.d, &mut seed),
        n(rows * spec.tk * heads_total * spec.d, &mut seed),
    );
    let stride = key_mask_stride(spec.tk);
    let key_mask = (0..batch * stride).map(|x| i64::from((x / stride + x % stride) % 5 != 2)).collect();
    let seg_start = (0..batch * spec.t).map(|x| (x % spec.t - x % spec.t % SEGMENT) as i64).collect();
    let row_map = (0..batch).map(|b| (rows - 1 - b) as i64).collect();
    let app = batch * appended(&spec) * spec.kv_heads * spec.d;
    let (k_app, v_app) = (n(app, &mut seed), n(app, &mut seed));
    let app_lens = (0..batch).map(|b| (appended(&spec) as i64 - b as i64 % 3).max(1)).collect();
    let bias_rows = match spec.mask.bias {
        Some(Bias::Shared) => 1,
        Some(Bias::PerBatch) => batch,
        None => 0,
    };
    let (keys, stride) = (keys(&spec), key_mask_stride(keys(&spec)));
    let bias = n(bias_rows * spec.heads * spec.t * stride, &mut seed)
        .into_iter()
        .enumerate()
        .map(|(i, x)| if i % stride < keys { 2.0 * x } else { 1e3 })
        .collect();
    let head_start = spec.kv_heads as i64;
    Case { spec, q, k, v, lens: lens.to_vec(), key_mask, seg_start, row_map, head_start, k_app, v_app, app_lens, bias }
}

/// Direct attention in f64 with the same masks and cache reads.
fn reference(c: &Case) -> Vec<f64> {
    let s = &c.spec;
    let m = s.mask;
    let cache =
        s.cache.unwrap_or(Cache { rows: 0, heads_total: s.kv_heads, row_map: false, appended: 0, app_lens: false });
    let head_start = if s.cache.is_some() { c.head_start as usize } else { 0 };
    let (stride, kv_stride, group) = (s.heads * s.d, cache.heads_total * s.d, s.heads / s.kv_heads);
    let mut out = vec![0.0; s.batch.capacity() * s.t * stride];
    for b in 0..s.batch.capacity() {
        let len = if m.key_lens { c.lens[b] as usize } else { s.tk };
        let seen = if cache.app_lens { c.app_lens[b] as usize } else { cache.appended };
        let row = if cache.row_map { c.row_map[b] as usize } else { b };
        let keys = keys(s);
        let bias_stride = key_mask_stride(keys);
        let bias_row = if m.bias == Some(Bias::PerBatch) { b } else { 0 };
        for h in 0..s.heads {
            let kv_head = head_start + h / group;
            let key = |j: usize, dd: usize, cached: &[f64], appended: &[f64]| {
                if j < s.tk {
                    cached[(row * s.tk + j) * kv_stride + kv_head * s.d + dd]
                } else {
                    appended[((b * cache.appended + j - s.tk) * s.kv_heads + h / group) * s.d + dd]
                }
            };
            for i in 0..s.t {
                let qi = &c.q[(b * s.t + i) * stride + h * s.d..][..s.d];
                let chunked = |j: usize| {
                    m.chunk.is_some_and(|Chunk { size, left, right }| {
                        j / size > i / size + right || left.is_some_and(|l| j / size + l < i / size)
                    })
                };
                let hidden = |j: usize| {
                    if j >= s.tk {
                        return j - s.tk >= seen;
                    }
                    j >= len
                        || (m.causal && j > i)
                        || m.window.is_some_and(|(l, r)| j + l < i || j > i + r)
                        || chunked(j)
                        || (m.key_mask && c.key_mask[b * key_mask_stride(s.tk) + j] == 0)
                        || (m.seg_start && (j as i64) < c.seg_start[b * s.t + i])
                };
                let scores: Vec<f64> = (0..keys)
                    .map(|j| {
                        if hidden(j) {
                            return f64::NEG_INFINITY;
                        }
                        let bias = match m.bias {
                            Some(_) => c.bias[((bias_row * s.heads + h) * s.t + i) * bias_stride + j],
                            None => 0.0,
                        };
                        qi.iter().enumerate().map(|(dd, a)| a * key(j, dd, &c.k, &c.k_app)).sum::<f64>()
                            * s.scale as f64
                            + bias
                    })
                    .collect();
                let mx = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                // A query no key reaches attends to nothing: zeros, as the graph writes.
                if mx == f64::NEG_INFINITY {
                    continue;
                }
                let w: Vec<f64> = scores.iter().map(|x| (x - mx).exp()).collect();
                let l: f64 = w.iter().sum();
                for dd in 0..s.d {
                    let acc: f64 = (0..keys).map(|j| w[j] * key(j, dd, &c.v, &c.v_app)).sum();
                    out[(b * s.t + i) * stride + h * s.d + dd] = acc / l;
                }
            }
        }
    }
    out
}

/// The parameters after the outputs in order, as f64 vectors.
fn extra_params(c: &Case) -> Vec<Vec<f64>> {
    let ints = |v: &[i64]| v.iter().map(|&x| x as f64).collect::<Vec<f64>>();
    c.spec
        .extra_params()
        .into_iter()
        .map(|name| match name {
            "key_lens" => ints(&c.lens),
            "key_mask" => ints(&c.key_mask),
            "seg_start" => ints(&c.seg_start),
            "row_map" => ints(&c.row_map),
            "head_start" => vec![c.head_start as f64],
            "k_app" => c.k_app.clone(),
            "v_app" => c.v_app.clone(),
            "app_lens" => ints(&c.app_lens),
            "bias" => c.bias.clone(),
            other => unreachable!("{other}"),
        })
        .collect()
}

/// Every parameter in order; partial outputs when the config splits.
fn params(c: &Case) -> Vec<Vec<f64>> {
    let s = &c.spec;
    let (cap, splits) = (s.batch.capacity(), s.cfg.splits);
    let mut p = vec![c.q.clone(), c.k.clone(), c.v.clone()];
    if splits == 1 {
        p.push(vec![0.0; cap * s.t * s.heads * s.d]);
    } else {
        p.push(vec![0.0; splits * cap * s.t * s.heads * s.d]);
        p.push(vec![0.0; splits * cap * s.t * s.heads]);
        p.push(vec![0.0; splits * cap * s.t * s.heads]);
    }
    p.extend(extra_params(c));
    p
}

/// The output: `o`, or the partials merged by the combine program.
fn output(c: &Case, got: Vec<Vec<f64>>, live: i64) -> Vec<f64> {
    let s = &c.spec;
    if s.cfg.splits == 1 {
        return got[3].clone();
    }
    let merge = combine_spec(s);
    let params =
        vec![got[3].clone(), got[4].clone(), got[5].clone(), vec![0.0; s.batch.capacity() * s.t * s.heads * s.d]];
    run(&combine::<BF16>(&merge), params, &[("b", live)]).unwrap().swap_remove(3)
}

fn combine_spec(s: &AttnSpec) -> CombineSpec {
    CombineSpec { batch: s.batch.clone(), t: s.t, heads: s.heads, d: s.d, splits: s.cfg.splits, cfg: NormCfg { br: 4 } }
}

fn spec(t: usize, tk: usize, d: usize, bq: usize, bkv: usize, causal: bool, key_lens: bool) -> AttnSpec {
    let mask = AttnMask { causal, key_lens, ..AttnMask::default() };
    masked(t, tk, d, bq, bkv, mask)
}

fn masked(t: usize, tk: usize, d: usize, bq: usize, bkv: usize, mask: AttnMask) -> AttnSpec {
    AttnSpec {
        batch: Batch::Var { name: "b".into(), min: 1, max: 2 },
        t,
        tk,
        heads: 2,
        kv_heads: 2,
        d,
        mask,
        cache: None,
        scale: 1.0 / (d as f32).sqrt(),
        cfg: FaCfg::new(bq, bkv, 2),
    }
}

const NONE: AttnMask = AttnMask {
    causal: false,
    window: None,
    chunk: None,
    key_lens: false,
    key_mask: false,
    seg_start: false,
    bias: None,
};
const WINDOW: AttnMask = AttnMask { window: Some((20, 5)), key_lens: true, ..NONE };
const KEY_MASK: AttnMask = AttnMask { key_mask: true, ..NONE };
const SEGMENTS: AttnMask = AttnMask { causal: true, seg_start: true, ..NONE };
/// Chunks of 16 with two behind: the block edges fall inside 32- and 64-wide key blocks.
const CHUNKED: AttnMask = AttnMask { chunk: Some(Chunk { size: 16, left: Some(2), right: 0 }), key_lens: true, ..NONE };
/// Chunks of 24 with every earlier one and one ahead: the edges off the key block grid.
const CHUNKED_ALL: AttnMask = AttnMask { chunk: Some(Chunk { size: 24, left: None, right: 1 }), ..NONE };
const EVERY: AttnMask = AttnMask {
    causal: true,
    window: Some((50, 0)),
    chunk: Some(Chunk { size: 32, left: Some(3), right: 0 }),
    key_lens: true,
    key_mask: true,
    seg_start: true,
    bias: None,
};
const SHARED: AttnMask = AttnMask { bias: Some(Bias::Shared), ..NONE };
const PER_BATCH: AttnMask = AttnMask { bias: Some(Bias::PerBatch), ..NONE };
const CAUSAL_BIAS: AttnMask = AttnMask { causal: true, ..PER_BATCH };
const LENS_BIAS: AttnMask = AttnMask { key_lens: true, ..PER_BATCH };
const EVERY_BIAS: AttnMask = AttnMask { bias: Some(Bias::Shared), ..EVERY };

/// `spec` with its cache's scores biased per batch.
fn biased(spec: AttnSpec) -> AttnSpec {
    AttnSpec { mask: AttnMask { bias: Some(Bias::PerBatch), ..spec.mask }, ..spec }
}

/// The RDNA prefill forms: truncated probabilities, the rescale skipped and
/// the maximum in raw units, on a 64×32 tile.
const RAW_MAX: FaCfg = FaCfg {
    forms: Forms { truncate_p: true, skip_rescale: true, raw_max: true, sum_after: false, bias_direct: false },
    ..FaCfg::new(64, 32, 2)
};

/// One row warp of queries sharing its block with `kw` key warps, `bkv` keys
/// a block, under the plain forms.
const fn key_warps(kw: usize, bkv: usize) -> FaCfg {
    FaCfg { key_warps: kw, ..FaCfg::new(16, bkv, 2) }
}

/// Two atom rows of queries per warp, `bq` by `bkv`, in the plain forms.
const fn rows32(bq: usize, bkv: usize) -> FaCfg {
    FaCfg { warp_rows: 2 * WARP_ROWS, ..FaCfg::new(bq, bkv, 2) }
}

/// A scale of the caller's choosing, in place of `1/√d`.
fn scaled(mut spec: AttnSpec, scale: f32) -> AttnSpec {
    spec.scale = scale;
    spec
}

fn with(cfg: FaCfg, spec: AttnSpec) -> AttnSpec {
    AttnSpec { cfg, ..spec }
}

/// The program in the interpreter against a direct f64 softmax, and on a
/// device its lowered kernel against the program, the ring as deep as the
/// device's fill runs (register staging: two slots at most): every mask form,
/// ragged blocks, caches, appended rows, splits, key warps and two atom rows
/// a warp.
#[test_case(spec(64, 64, 64, 64, 64, false, false), &[64, 64]; "one block")]
#[test_case(spec(128, 192, 64, 64, 64, false, false), &[192, 192]; "three key blocks")]
#[test_case(spec(128, 128, 64, 64, 64, true, false), &[128, 128]; "causal")]
#[test_case(spec(128, 128, 64, 64, 64, false, true), &[100, 7]; "key lengths")]
#[test_case(spec(128, 128, 64, 64, 32, true, true), &[128, 50]; "causal with lengths, narrow kv")]
#[test_case(masked(128, 128, 64, 64, 32, WINDOW), &[128, 90]; "window with lengths")]
#[test_case(masked(128, 128, 64, 64, 64, KEY_MASK), &[128, 128]; "key mask")]
#[test_case(masked(128, 128, 64, 64, 32, SEGMENTS), &[128, 128]; "causal segments")]
#[test_case(masked(128, 128, 64, 64, 32, CHUNKED), &[128, 90]; "chunks with lengths")]
#[test_case(masked(100, 100, 64, 64, 64, CHUNKED_ALL), &[100, 100]; "chunks, every earlier one and one ahead, ragged")]
#[test_case(masked(128, 128, 64, 64, 32, EVERY), &[128, 100]; "every mask")]
#[test_case(masked(96, 96, 48, 64, 64, SEGMENTS), &[96, 96]; "d 48")]
#[test_case(masked(128, 128, 64, 64, 64, PER_BATCH), &[128, 128]; "bias per batch")]
#[test_case(masked(100, 130, 64, 64, 64, SHARED), &[130, 130]; "shared bias, ragged")]
#[test_case(masked(128, 128, 64, 64, 32, CAUSAL_BIAS), &[128, 128]; "causal bias")]
#[test_case(masked(99, 99, 64, 64, 64, LENS_BIAS), &[99, 41]; "bias with key lengths, odd keys")]
#[test_case(masked(128, 128, 128, 64, 32, LENS_BIAS), &[128, 70]; "d 128 bias with key lengths")]
#[test_case(masked(128, 128, 64, 64, 32, EVERY_BIAS), &[128, 100]; "every mask and a bias")]
#[test_case(spec(128, 128, 64, 64, 64, false, true), &[100, 0]; "a lane with no keys")]
#[test_case(spec(128, 128, 64, 64, 32, true, true), &[0, 50]; "causal, a lane with no keys")]
#[test_case(scaled(spec(128, 128, 64, 64, 32, true, true), 0.3), &[128, 50]; "a larger scale")]
#[test_case(with(RAW_MAX, spec(128, 128, 64, 64, 32, true, true)), &[128, 50]; "the maximum in raw units")]
#[test_case(with(RAW_MAX, scaled(spec(128, 128, 64, 64, 32, true, true), 0.3)), &[128, 50]; "the maximum in raw units, a larger scale")]
#[test_case(with(RAW_MAX, scaled(spec(128, 128, 64, 64, 32, true, true), -0.1)), &[128, 50]; "a negative scale keeps the maximum in the exponent's units")]
#[test_case(with(RAW_MAX, masked(128, 128, 64, 64, 32, EVERY_BIAS)), &[128, 100]; "a bias keeps the maximum in the exponent's units")]
#[test_case(with(key_warps(4, 16), spec(16, 200, 64, 16, 16, false, true)), &[200, 70]; "four key warps, key lengths")]
#[test_case(with(key_warps(2, 32), spec(16, 200, 64, 16, 16, true, true)), &[200, 70]; "two key warps, causal with lengths")]
#[test_case(with(key_warps(4, 16), spec(16, 48, 64, 16, 16, false, false)), &[48, 48]; "four key warps over three key blocks")]
#[test_case(with(key_warps(4, 16), spec(16, 16, 64, 16, 16, false, true)), &[16, 0]; "four key warps, a lane with no keys")]
#[test_case(with(FaCfg { forms: RAW_MAX.forms, ..key_warps(4, 16) }, spec(16, 200, 128, 16, 16, false, true)), &[200, 130]; "four key warps, d 128, the RDNA prefill forms")]
#[test_case(with(key_warps(4, 16), masked(16, 128, 64, 16, 16, PER_BATCH)), &[128, 128]; "four key warps, bias per batch")]
#[test_case(with(rows32(64, 32), spec(100, 150, 64, 64, 32, false, true)), &[150, 77]; "two atom rows a warp, plain")]
#[test_case(with(rows32(64, 32), spec(100, 150, 128, 64, 32, true, true)), &[150, 77]; "two atom rows a warp, causal, d 128")]
#[test_case(with(rows32(128, 64), masked(100, 150, 64, 128, 64, AttnMask { key_lens: true, ..PER_BATCH })), &[150, 77]; "two atom rows a warp, bias")]
#[test_case(with(rows32(64, 32), masked(100, 150, 64, 64, 32, AttnMask { causal: true, key_lens: true, ..PER_BATCH })), &[150, 77]; "two atom rows a warp, causal bias")]
#[test_case(with(FaCfg { stages: 1, ..rows32(64, 32) }, spec(100, 150, 64, 64, 32, true, true)), &[150, 77]; "two atom rows a warp, one slot")]
#[test_case(cached(1, 100, [4, 4, 64], false, 0, 1), &[100, 37]; "cache slice")]
#[test_case(cached(1, 100, [4, 2, 64], true, 1, 1), &[100, 37]; "row map, appended, gqa")]
#[test_case(cached(1, 200, [4, 4, 64], true, 1, 4), &[200, 70]; "four splits")]
#[test_case(cached(5, 96, [2, 2, 128], false, 1, 2), &[64, 96]; "d 128, two splits, five queries")]
#[test_case(cached(1, 64, [2, 2, 64], false, 0, 4), &[64, 40]; "more splits than blocks for a row")]
#[test_case(cached(1, 7, [6, 6, 64], false, 1, 1), &[0, 7]; "appended row only, one split")]
#[test_case(cached(1, 7, [6, 6, 64], false, 1, 2), &[0, 3]; "appended row only, two splits")]
#[test_case(cached(8, 128, [4, 4, 48], false, 8, 1), &[128, 40]; "eight queries, eight appended, d 48")]
#[test_case(cached(16, 128, [2, 2, 64], true, 16, 2), &[128, 0]; "sixteen appended, a block of them, two splits")]
#[test_case(counted(cached(8, 128, [4, 4, 48], false, 8, 1)), &[128, 40]; "counted appended rows")]
#[test_case(counted(cached(16, 96, [2, 2, 64], false, 16, 3)), &[0, 96]; "counted appended rows only, three splits")]
#[test_case(biased(cached(1, 100, [4, 2, 64], true, 1, 1)), &[100, 37]; "biased: row map, appended, gqa")]
#[test_case(biased(cached(5, 96, [2, 2, 128], false, 1, 2)), &[64, 96]; "biased: d 128, two splits")]
#[test_case(biased(cached(1, 200, [4, 4, 64], false, 0, 4)), &[200, 70]; "biased: four splits")]
#[test_case(biased(counted(cached(8, 64, [2, 2, 64], false, 8, 2))), &[64, 20]; "biased: counted appended rows")]
#[test_case(cached(1, 100, [4, 4, 64], false, 0, 1), &[100, 0]; "a cached lane with no keys")]
#[test_case(cached(1, 200, [4, 4, 64], false, 0, 4), &[0, 70]; "a cached lane with no keys, four splits")]
#[test_case(scaled(cached(1, 200, [4, 4, 64], true, 1, 4), -0.1), &[200, 70]; "four splits merge a negative scale's partial maxima")]
#[test_case(with(RAW_MAX, cached(1, 200, [4, 4, 64], true, 1, 4)), &[200, 70]; "four splits merge raw partial maxima in the exponent's units")]
#[test_case(with(RAW_MAX, biased(cached(5, 96, [2, 2, 128], false, 1, 2))), &[64, 96]; "biased, two splits: the maximum stays in the exponent's units")]
#[test_case(with(key_warps(4, 16), cached(1, 200, [4, 2, 64], true, 1, 1)), &[200, 37]; "four key warps: row map, appended, gqa")]
#[test_case(with(FaCfg { splits: 3, ..key_warps(2, 32) }, cached(1, 200, [4, 4, 64], true, 1, 1)), &[200, 70]; "two key warps under three splits")]
#[test_case(with(key_warps(4, 16), counted(cached(16, 128, [2, 2, 64], true, 16, 1))), &[128, 0]; "four key warps, sixteen counted appended rows")]
#[test_case(with(key_warps(2, 64), cached(1, 7, [6, 6, 64], false, 1, 1)), &[0, 7]; "two key warps, the appended row only")]
#[test_case(with(key_warps(4, 16), biased(cached(5, 96, [2, 2, 128], false, 1, 1))), &[64, 96]; "four key warps, biased, d 128")]
#[test_case(spec(128, 128, 64, 64, 64, false, false), &[128, 128]; "plain")]
#[test_case(spec(256, 256, 64, 64, 64, true, false), &[256, 256]; "causal, 256 queries")]
#[test_case(spec(128, 256, 64, 64, 64, false, true), &[200, 33]; "key lengths, 256 keys")]
#[test_case(spec(128, 128, 128, 64, 32, true, true), &[128, 64]; "d 128, causal with lengths")]
#[test_case(with(RAW_MAX, spec(128, 128, 128, 0, 0, true, true)), &[128, 64]; "d 128, the RDNA prefill forms")]
#[test_case(with(FaCfg::new(128, 64, 3), spec(256, 192, 64, 0, 0, true, true)), &[192, 70]; "bq 128, three stages")]
#[test_case(with(FaCfg::new(128, 32, 2), spec(200, 130, 128, 0, 0, false, true)), &[130, 99]; "d 128, bq 128, ragged")]
#[test_case(masked(100, 100, 64, 64, 64, KEY_MASK), &[100, 100]; "key mask, ragged")]
#[test_case(masked(128, 128, 128, 64, 32, EVERY), &[128, 100]; "every mask, d 128")]
#[test_case(with(FaCfg::new(64, 64, 3), masked(200, 200, 48, 0, 0, SEGMENTS)), &[200, 200]; "d 48, three stages")]
#[test_case(with(FaCfg::new(64, 64, 2), cached(8, 128, [4, 4, 48], false, 8, 1)), &[128, 40]; "cache: eight appended, d 48")]
#[test_case(counted(cached(16, 128, [2, 2, 64], true, 16, 2)), &[128, 0]; "cache: sixteen counted appended, two splits")]
#[test_case(with(FaCfg { splits: 3, ..FaCfg::new(16, 64, 2) }, cached(1, 1500, [4, 4, 64], true, 0, 1)), &[1500, 1200]; "cache: bq 16, three splits of 1500 keys")]
#[test_case(masked(256, 256, 64, 64, 64, CAUSAL_BIAS), &[256, 256]; "causal bias, 256 queries")]
#[test_case(with(FaCfg::new(128, 32, 2), masked(200, 130, 128, 0, 0, LENS_BIAS)), &[130, 99]; "d 128 bias with lengths, bq 128")]
#[test_case(masked(128, 128, 128, 64, 32, EVERY_BIAS), &[128, 100]; "every mask and a bias, d 128")]
#[test_case(biased(counted(cached(5, 96, [2, 2, 128], false, 8, 2))), &[64, 96]; "cache biased: d 128, eight counted appended, two splits")]
fn attention_matches_a_direct_softmax(spec: AttnSpec, lens: &[i64]) {
    matches_a_direct_softmax(&case(spec.clone(), lens));
    let Some(target) = super::device_target() else { return };
    let spec = AttnSpec { cfg: FaCfg { stages: target.stages(spec.cfg.stages), ..spec.cfg }, ..spec };
    let bias = spec.mask.bias.is_some();
    let smem = spec.cfg.smem_bytes(spec.d, bias) + spec.cfg.scratch_bytes(&target);
    if !spec.cfg.fits(&target, spec.d, bias) || smem > target.smem_bytes {
        return eprintln!("skipped on the device: {:?} does not fit {}", spec.cfg, target.arch.target_name());
    }
    on_device(&spec, lens, target);
}

/// Scores that climb from key block to key block, through a bias ramp along
/// the keys: slowly, a row keeps its stale maximum for several blocks and
/// then adopts a new one; steeply, it adopts one every block. Either way the
/// accumulator is rescaled while it holds earlier blocks' work.
#[test_case(0.1, false; "a slow climb")]
#[test_case(1.0, false; "a steep climb")]
#[test_case(0.1, true; "a slow causal climb")]
fn climbing_scores_rescale_the_accumulator(per_key: f64, causal: bool) {
    let mask = AttnMask { causal, ..PER_BATCH };
    let mut c = case(masked(128, 128, 64, 64, 16, mask), &[128, 128]);
    let stride = key_mask_stride(keys(&c.spec));
    for (i, b) in c.bias.iter_mut().enumerate().filter(|(i, _)| i % stride < 128) {
        *b = round_to(ScalarDType::BFloat16, per_key * (i % stride) as f64);
    }
    matches_a_direct_softmax(&c);
}

fn matches_a_direct_softmax(c: &Case) {
    let spec = &c.spec;
    let got = run(&flash_attention(spec), params(c), &[("b", 2)]).unwrap();
    let got = output(c, got, 2);
    let want = reference(c);
    let mut worst = 0.0f64;
    for (g, w) in got.iter().zip(&want) {
        worst = worst.max((g - w).abs());
    }
    assert!(worst < 2e-2, "max abs diff {worst}");
}

/// A decoder step: `t` query rows against a cache of several layers' heads,
/// rows mapped, the step's own `appended` keys after the prefix.
fn cached(
    t: usize,
    tk: usize,
    [heads, kv_heads, d]: [usize; 3],
    row_map: bool,
    appended: usize,
    splits: usize,
) -> AttnSpec {
    let mut spec = masked(t, tk, d, 64, 32, AttnMask { key_lens: true, ..AttnMask::default() });
    spec.heads = heads;
    spec.kv_heads = kv_heads;
    let cache = Cache { rows: 3, heads_total: 3 * kv_heads, row_map, appended, app_lens: false };
    spec.cache = Some(cache);
    spec.cfg.splits = splits;
    spec
}

/// `spec` with the visible appended rows counted per batch.
fn counted(mut spec: AttnSpec) -> AttnSpec {
    spec.cache = spec.cache.map(|c| Cache { app_lens: true, ..c });
    spec
}

/// A batch whose key mask hides every key attends to nothing: its rows come
/// out as zeros, as the graph writes them, not as the NaN of an empty
/// softmax, whole or merged from key splits.
#[test_case(1; "whole")]
#[test_case(3; "three splits")]
fn a_fully_masked_batch_is_zeros(splits: usize) {
    let spec = masked(128, 128, 64, 64, 32, KEY_MASK);
    let spec = AttnSpec { cfg: FaCfg { splits, ..spec.cfg }, ..spec };
    let mut c = case(spec.clone(), &[128, 128]);
    let stride = key_mask_stride(spec.tk);
    c.key_mask[stride..].fill(0);
    let got = run(&flash_attention(&spec), params(&c), &[("b", 2)]).unwrap();
    let got = output(&c, got, 2);
    let half = spec.t * spec.heads * spec.d;
    assert!(got[..half].iter().any(|x| *x != 0.0), "the visible batch attends");
    assert!(got[half..].iter().all(|x| *x == 0.0), "the hidden batch is zeros");
    let want = reference(&c);
    assert!(got.iter().zip(&want).all(|(g, w)| (g - w).abs() < 2e-2));
}

/// The live batch bound: rows of a batch past it are never written.
#[test]
fn only_the_live_batch_runs() {
    let spec = spec(64, 64, 64, 64, 64, false, false);
    let c = case(spec.clone(), &[64, 64]);
    let got = run(&flash_attention(&spec), params(&c), &[("b", 1)]).unwrap();
    let half = spec.t * spec.heads * spec.d;
    assert!(got[3][..half].iter().any(|x| *x != 0.0));
    assert!(got[3][half..].iter().all(|x| *x == 0.0));
}

/// One row of queries a warp or two give the same bits on the device, under
/// every mask form the probe covers, over partial query and key blocks:
/// a 32-row warp runs the 16-row warp's arithmetic per element (the biased
/// score is one fused multiply-add in both, so neither layout leaves the
/// backend a different product to fuse), and both match the reference.
#[test_case(false, false, 64, 64, 32; "plain")]
#[test_case(true, false, 128, 64, 32; "causal, d 128")]
#[test_case(false, true, 64, 128, 64; "bias")]
#[test_case(true, true, 64, 64, 32; "causal bias")]
fn two_atom_rows_match_one_on_the_device(causal: bool, bias: bool, d: usize, bq: usize, bkv: usize) {
    let Some(planner) = super::device_planner() else { return };
    let device = &planner.device;
    let mask = AttnMask { causal, key_lens: true, bias: bias.then_some(Bias::PerBatch), ..NONE };
    let one = masked(100, 150, d, bq, bkv, mask);
    let two = AttnSpec { cfg: FaCfg { warp_rows: 2 * WARP_ROWS, ..one.cfg }, ..one.clone() };
    if ![&one, &two].iter().all(|s| s.cfg.fits(&device.target, d, bias)) {
        eprintln!("skipped: the tiles do not fit the target");
        return;
    }
    let c = case(one.clone(), &[150, 77]);
    let to_bf16 = |v: &[f64]| Tensor::from_slice(v.iter().map(|&x| x as f32).collect::<Vec<_>>()).cast(DType::BFloat16);
    let (q, k, v) = (to_bf16(&c.q), to_bf16(&c.k), to_bf16(&c.v));
    let elems = one.batch.capacity() * one.t * one.heads * d;
    let extras: Vec<Tensor> = one
        .extra_params()
        .into_iter()
        .zip(extra_params(&c))
        .map(|(name, v)| match name {
            "bias" => to_bf16(&v),
            _ => Tensor::from_slice(v.iter().map(|&x| x as i32).collect::<Vec<_>>()),
        })
        .collect();
    let bits = |spec: &AttnSpec| -> Vec<u16> {
        let o = Tensor::empty(&[elems], DType::BFloat16);
        let mut tensors = vec![&q, &k, &v, &o];
        tensors.extend(&extras);
        let outs = graph_launch_all(flash_attention(spec), &spec.cfg.lowering(device), &tensors).unwrap();
        let mut plan = outs[3].prepare().unwrap();
        plan.execute_with_vars(&[("b", 2)]).unwrap();
        let mut bytes = vec![0u8; elems * 2];
        outs[3].buffer().unwrap().copyout(&mut bytes).unwrap();
        bytes.chunks(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect()
    };
    let (got, want) = (bits(&two), bits(&one));
    let same = got.iter().zip(&want).filter(|(a, b)| a == b).count();
    assert_eq!(same, got.len(), "{} of {} outputs differ from the 16-row kernel's", got.len() - same, got.len());
    let reference = reference(&c);
    let worst = got.iter().zip(&reference).map(|(g, w)| (super::bf16(*g) - w).abs()).fold(0.0, f64::max);
    assert!(worst < 2e-2, "max abs diff {worst} against the reference");
}

/// The cache operand's first elements a decoder step never reads (its rows
/// map past the cache's first row) are an infinity and a NaN, as memory
/// nothing initialised may hold, and its last key block is partial: the
/// kernel reads zeros past the keys, never those elements, so its output is
/// finite and matches the program's, through the transposing loads on RDNA4.
#[test_case(FaCfg { splits: 4, ..FaCfg::new(16, 16, 2) }; "a decoder step over four splits")]
#[test_case(key_warps(4, 16); "four key warps")]
fn a_non_finite_unread_element_never_reaches_the_output(cfg: FaCfg) {
    let Some(target) = super::device_target() else { return };
    let spec = with(cfg, cached(1, 100, [4, 4, 64], true, 0, 1));
    let spec = AttnSpec { cfg: FaCfg { stages: target.stages(spec.cfg.stages), ..spec.cfg }, ..spec };
    let mut c = case(spec, &[100, 37]);
    assert!(c.row_map.iter().all(|&r| r > 0), "no batch reads the cache's first row");
    c.v[0] = f64::INFINITY;
    c.v[1] = f64::NAN;
    let bits = on_device_case(c, target);
    assert!(bits.iter().all(|&b| super::bf16(b).is_finite()), "a non-finite output");
}

/// `spec`'s kernel on the device against the program in the interpreter
/// (within 2e-2 of it), its bf16 output's bits returned.
fn on_device(spec: &AttnSpec, lens: &[i64], target: Target) -> Vec<u16> {
    on_device_case(case(spec.clone(), lens), target)
}

/// [`on_device`] of the parameters `c` holds.
fn on_device_case(c: Case, target: Target) -> Vec<u16> {
    let spec = c.spec.clone();
    let batch = spec.batch.capacity();
    let raw = run(&flash_attention(&spec), params(&c), &[("b", batch as i64)]).unwrap();
    let want = output(&c, raw.clone(), batch as i64);
    let lowering = spec.cfg.lowering(&declared(target.clone()));
    let to_bf16 = |v: &[f64]| Tensor::from_slice(v.iter().map(|&x| x as f32).collect::<Vec<_>>()).cast(DType::BFloat16);
    let (q, k, v) = (to_bf16(&c.q), to_bf16(&c.k), to_bf16(&c.v));
    let elems = batch * spec.t * spec.heads * spec.d;
    let outputs: Vec<Tensor> = if spec.cfg.splits == 1 {
        vec![Tensor::empty(&[elems], DType::BFloat16)]
    } else {
        let per = spec.cfg.splits * batch * spec.t * spec.heads;
        vec![
            Tensor::empty(&[per * spec.d], DType::Float32),
            Tensor::empty(&[per], DType::Float32),
            Tensor::empty(&[per], DType::Float32),
        ]
    };
    let extras: Vec<Tensor> = spec
        .extra_params()
        .into_iter()
        .zip(extra_params(&c))
        .map(|(name, v)| match name {
            "k_app" | "v_app" | "bias" => to_bf16(&v),
            _ => Tensor::from_slice(v.iter().map(|&x| x as i32).collect::<Vec<_>>()),
        })
        .collect();
    let mut tensors = vec![&q, &k, &v];
    tensors.extend(&outputs);
    tensors.extend(&extras);
    let outs = graph_launch_all(flash_attention(&spec), &lowering, &tensors).unwrap();
    // The partials themselves, before the merge: a bug the merge hides is
    // still one. Equal infinities (a split no key reaches) agree; a NaN never.
    for (i, name) in
        ["o_part", "m_part", "l_part"].into_iter().enumerate().take(if spec.cfg.splits > 1 { 3 } else { 0 })
    {
        let part = &outs[3 + i];
        let mut plan = part.prepare().unwrap();
        plan.execute_with_vars(&[("b", batch as i64)]).unwrap();
        let mut bytes = vec![0u8; raw[3 + i].len() * 4];
        part.buffer().unwrap().copyout(&mut bytes).unwrap();
        let got = super::decode(ScalarDType::Float32, &bytes);
        let rel = |(g, w): (&f64, &f64)| if g == w { 0.0 } else { (g - w).abs() / w.abs().max(1.0) };
        let worst = got.iter().zip(&raw[3 + i]).map(rel).fold(0.0, f64::max);
        eprintln!("{name}: max rel diff {worst:.3e} vs interpreter");
        if let Some(at) = got.iter().zip(&raw[3 + i]).position(|p| rel(p).is_nan() || rel(p) > 1e-2) {
            panic!("{name}[{at}] = {}, the interpreter's {}", got[at], raw[3 + i][at]);
        }
    }
    let out = if spec.cfg.splits == 1 {
        outs[3].clone()
    } else {
        let merge = combine_spec(&spec);
        let o = Tensor::empty(&[elems], DType::BFloat16);
        graph_launch(
            combine::<BF16>(&merge),
            &merge.cfg.lowering(&declared(target)),
            &[&outs[3], &outs[4], &outs[5], &o],
        )
        .unwrap()
    };
    let mut plan = out.prepare().unwrap();
    plan.execute_with_vars(&[("b", batch as i64)]).unwrap();
    let mut bytes = vec![0u8; elems * 2];
    out.buffer().unwrap().copyout(&mut bytes).unwrap();
    let bits: Vec<u16> = bytes.chunks(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
    let got: Vec<f64> = bits.iter().map(|&b| super::bf16(b)).collect();
    let reference = reference(&c);
    let mut worst = 0.0f64;
    let (mut over, mut worst_at) = (0usize, 0usize);
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        let d = (g - w).abs();
        if d > worst {
            worst = d;
            worst_at = i;
        }
        over += usize::from(d > 1e-2);
    }
    let vs_ref = got.iter().zip(&reference).map(|(g, w)| (g - w).abs()).fold(0.0, f64::max);
    let stride = spec.heads * spec.d;
    eprintln!(
        "max abs diff {worst:.3e} at batch {} row {} head {} col {} ({over} elements over 1e-2); vs f64 reference {vs_ref:.3e}",
        worst_at / (spec.t * stride),
        worst_at / stride % spec.t,
        worst_at % stride / spec.d,
        worst_at % spec.d
    );
    assert!(worst < 2e-2, "max abs diff {worst}");
    bits
}

/// `spec` on the warpgroup core.
fn grouped(spec: AttnSpec) -> AttnSpec {
    AttnSpec { cfg: FaCfg { core: Core::Group, ..spec.cfg }, ..spec }
}

/// The warpgroup core's kernel (four row warps a group of 64 query rows,
/// both products chains over shared descriptors) against the program on
/// every mask form, ragged blocks and a cache without appended rows, and
/// within one bf16 ulp of the warp-level kernel's output: the two cores
/// accumulate a `k16` step alike (bit for bit in the GEMM,
/// `device::warpgroup_gemm_matches_the_warp_core`), and the softmax around
/// them is the same code.
#[test_case(spec(64, 64, 64, 64, 64, false, false), &[64, 64]; "one key block")]
#[test_case(spec(128, 192, 64, 64, 64, false, false), &[192, 192]; "three key blocks")]
#[test_case(spec(256, 256, 64, 64, 64, true, false), &[256, 256]; "causal")]
#[test_case(spec(128, 256, 64, 64, 64, false, true), &[200, 33]; "key lengths")]
#[test_case(spec(128, 128, 128, 64, 32, true, true), &[128, 64]; "d 128, causal with lengths")]
#[test_case(spec(256, 256, 128, 64, 128, false, false), &[256, 256]; "d 128, 128 keys a block")]
#[test_case(with(FaCfg::new(128, 64, 3), spec(256, 192, 64, 0, 0, true, true)), &[192, 70]; "bq 128, three stages")]
#[test_case(with(FaCfg::new(128, 128, 2), spec(256, 256, 64, 0, 0, false, false)), &[256, 256]; "bq 128, 128 keys a block")]
#[test_case(with(FaCfg::new(128, 32, 2), spec(200, 130, 128, 0, 0, false, true)), &[130, 99]; "d 128, bq 128, ragged")]
#[test_case(with(RAW_MAX, spec(128, 128, 128, 0, 0, true, true)), &[128, 64]; "d 128, the RDNA prefill forms")]
#[test_case(masked(128, 128, 64, 64, 32, WINDOW), &[128, 90]; "window with lengths")]
#[test_case(masked(100, 100, 64, 64, 64, KEY_MASK), &[100, 100]; "key mask, ragged")]
#[test_case(masked(128, 128, 64, 64, 32, SEGMENTS), &[128, 128]; "causal segments")]
#[test_case(masked(128, 128, 64, 64, 32, CHUNKED), &[128, 90]; "chunks with lengths")]
#[test_case(masked(100, 100, 64, 64, 64, CHUNKED_ALL), &[100, 100]; "chunks, every earlier one and one ahead, ragged")]
#[test_case(masked(128, 128, 128, 64, 32, EVERY), &[128, 100]; "every mask, d 128")]
#[test_case(masked(128, 128, 64, 64, 64, PER_BATCH), &[128, 128]; "bias per batch")]
#[test_case(masked(128, 128, 64, 64, 128, PER_BATCH), &[128, 128]; "bias per batch, 128 keys a block")]
#[test_case(masked(100, 130, 64, 64, 64, SHARED), &[130, 130]; "shared bias, ragged")]
#[test_case(masked(256, 256, 64, 64, 64, CAUSAL_BIAS), &[256, 256]; "causal bias")]
#[test_case(masked(99, 99, 64, 64, 64, LENS_BIAS), &[99, 41]; "bias with key lengths, odd keys")]
#[test_case(with(FaCfg::new(128, 32, 2), masked(200, 130, 128, 0, 0, LENS_BIAS)), &[130, 99]; "d 128 bias with lengths, bq 128")]
#[test_case(masked(128, 128, 128, 64, 32, EVERY_BIAS), &[128, 100]; "every mask and a bias, d 128")]
#[test_case(cached(1, 100, [4, 4, 64], false, 0, 1), &[100, 37]; "cache slice")]
#[test_case(cached(1, 200, [4, 4, 64], true, 0, 4), &[200, 70]; "cache: row map, four splits")]
#[test_case(cached(5, 96, [2, 2, 128], false, 0, 2), &[64, 96]; "cache: d 128, two splits")]
#[test_case(biased(cached(1, 200, [4, 4, 64], false, 0, 4)), &[200, 70]; "cache biased: four splits")]
fn warpgroup_kernel_matches_the_program(spec: AttnSpec, lens: &[i64]) {
    let Some(target) = super::group_target() else { return };
    let spec = grouped(spec);
    assert!(spec.cfg.fits(&target, spec.d, spec.mask.bias.is_some()), "{:?} does not fit", spec.cfg);
    let group = on_device(&spec, lens, target.clone());
    let warp = on_device(&AttnSpec { cfg: FaCfg { core: Core::Warp, ..spec.cfg }, ..spec }, lens, target);
    let equal = group.iter().zip(&warp).filter(|(g, w)| g == w).count();
    // Bits as a signed magnitude, so ±0 agree and a sign change counts its steps.
    let ordered = |b: u16| if b & 0x8000 == 0 { i32::from(b) } else { -i32::from(b & 0x7fff) };
    let ulps = group.iter().zip(&warp).map(|(g, w)| (ordered(*g) - ordered(*w)).unsigned_abs()).max().unwrap_or(0);
    eprintln!("{equal} of {} bit-equal to the warp-level kernel, at most {ulps} bf16 ulps apart", group.len());
    assert!(ulps <= 1, "the warpgroup kernel is {ulps} bf16 ulps from the warp-level one");
}

/// The split merge alone: random partials through the kernel against the
/// interpreter.
#[test_case(1, 4, 64, 4, 2; "four rows, four splits")]
#[test_case(5, 2, 128, 2, 2; "ten rows, d 128, two splits")]
#[test_case(1, 4, 64, 3, 1; "three splits, one batch")]
#[test_case(7, 3, 64, 2, 2; "twenty-one rows, partial block")]
fn combine_matches_the_program(t: usize, heads: usize, d: usize, splits: usize, batch: usize) {
    let Some(target) = super::device_target() else { return };
    let spec = CombineSpec { batch: Batch::Static(batch), t, heads, d, splits, cfg: NormCfg { br: 4 } };
    let rows = splits * batch * t * heads;
    let mut seed = 9;
    let o_part = super::rounded(ScalarDType::Float32, rows * d, &mut seed, |x| 10.0 * x);
    let m_part = super::rounded(ScalarDType::Float32, rows, &mut seed, |x| 4.0 * x);
    let l_part = super::rounded(ScalarDType::Float32, rows, &mut seed, |x| 1.5 + x);
    let params = vec![o_part, m_part, l_part, vec![0.0; batch * t * heads * d]];
    let want = run(&combine::<BF16>(&spec), params.clone(), &[]).unwrap().swap_remove(3);
    let tensors: Vec<Tensor> = params
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let t = Tensor::from_slice(v.iter().map(|&x| x as f32).collect::<Vec<_>>());
            if i == 3 { t.cast(DType::BFloat16) } else { t }
        })
        .collect();
    let refs: Vec<&Tensor> = tensors.iter().collect();
    let out = graph_launch(combine::<BF16>(&spec), &spec.cfg.lowering(&declared(target)), &refs).unwrap();
    let got: Vec<f64> = out.cast(DType::Float32).to_vec::<f32>().unwrap().into_iter().map(f64::from).collect();
    let worst = got.iter().zip(&want).map(|(g, w)| (g - w).abs()).fold(0.0, f64::max);
    let bad: Vec<usize> =
        got.iter().zip(&want).enumerate().filter(|(_, (g, w))| (*g - *w).abs() > 2e-2).map(|(i, _)| i).collect();
    eprintln!(
        "max abs diff {worst:.3e}; {} bad of {}; first bad {:?}",
        bad.len(),
        got.len(),
        bad.first().map(|&i| (i / d, i % d))
    );
    assert!(worst < 2e-2, "max abs diff {worst}");
}

/// The attention configs the tuner admits on the device for `spec`'s shape
/// in the plain forms (its shortlist of the device's ranking), the
/// candidates of its last tile pass, and its pick, tuned as the op layer
/// tunes ([`tune`]).
fn device_attention(planner: &Planner, spec: &AttnSpec) -> (Vec<FaCfg>, Vec<FaCfg>, FaCfg) {
    use crate::ops::attention::tune;
    let AttnSpec { t, tk, heads, d, mask, .. } = *spec;
    let tiles = spec.batch.capacity() * heads;
    let ranked = planner.attention_ranked(tiles, t, tk, d, mask.causal, mask.bias.is_some());
    let build = |cfg: FaCfg| vec![(flash_attention(&AttnSpec { cfg, ..spec.clone() }), cfg.lowering(&planner.device))];
    let shape = [spec.batch.capacity(), t, tk, heads, d];
    let mut tuner = crate::ops::Tuner::new(&planner.device, DType::BFloat16, &shape, mask, build);
    let plain = tuner.admitted("attention", &crate::ops::config::representatives(&ranked));
    let formed = std::cell::RefCell::new(vec![]);
    let pick = tune(&mut tuner, &ranked, mask.bias.is_some(), |admitted| {
        *formed.borrow_mut() = admitted.to_vec();
        admitted.to_vec()
    });
    (plain, formed.into_inner(), pick)
}

/// The probe's shape: `batch` lanes of `heads` heads over `t` queries and keys.
fn probe_spec(batch: usize, heads: usize, t: usize, tk: usize, d: usize, mask: AttnMask) -> AttnSpec {
    AttnSpec {
        batch: Batch::Static(batch),
        t,
        tk,
        heads,
        kv_heads: heads,
        d,
        mask,
        cache: None,
        scale: 1.0 / (d as f32).sqrt(),
        cfg: FaCfg::new(64, 64, 2),
    }
}

fn label(c: &FaCfg) -> String {
    let f = c.forms;
    let forms = [
        (f.truncate_p, "truncate_p"),
        (f.skip_rescale, "skip_rescale"),
        (f.raw_max, "raw_max"),
        (f.sum_after, "sum_after"),
        (f.bias_direct, "bias_direct"),
    ];
    let on: Vec<&str> = forms.iter().filter(|(on, _)| *on).map(|(_, name)| *name).collect();
    let rows = if c.warp_rows == WARP_ROWS { String::new() } else { format!(" rows={}", c.warp_rows) };
    let core = if c.core == Core::Warp { "" } else { " group" };
    format!("bq{} bkv{} s{}{rows} unroll={} forms [{}]{core}", c.bq, c.bkv, c.stages, c.unroll, on.join(" "))
}

/// Flash attention throughput at B 4, H 8, T 2048 bf16, causal and plain,
/// head dim 64 and 128, d 64 with a per-batch bias: every config the tuner
/// admits on the device (its probed costs' ranking) timed as the tune store
/// times them, and the tuner's pick with its softmax forms (run with
/// `--ignored --nocapture --release`); never asserts. tk1's
/// `flash_attention_with`, last measured on the 3060 before its removal, in
/// TFLOP/s: d64 23.0, d64 causal 20.7, d128 22.3, d128 causal 16.5.
#[test]
#[ignore = "perf probe: needs a GPU"]
fn attention_throughput_probe() {
    let Some(planner) = super::device_planner() else { return };
    crate::tune::scoped(true, || {
        let (batch, heads, t) = (4usize, 8usize, 2048usize);
        for (d, causal, bias) in [
            (64, false, false),
            (64, false, true),
            (64, true, false),
            (64, true, true),
            (128, false, false),
            (128, true, false),
        ] {
            let mask = AttnMask { causal, bias: bias.then_some(Bias::PerBatch), ..AttnMask::default() };
            let spec = probe_spec(batch, heads, t, t, d, mask);
            let (plain, admitted, pick) = device_attention(&planner, &spec);
            let mut timed = admitted.clone();
            if !timed.contains(&pick) {
                timed.push(pick);
            }
            let build =
                |cfg: FaCfg| vec![(flash_attention(&AttnSpec { cfg, ..spec.clone() }), cfg.lowering(&planner.device))];
            let ns = crate::tune::measure(timed.iter().map(|&c| build(c)));
            let flops = 4.0 * (batch * heads * t * t * d) as f64 / if causal { 2.0 } else { 1.0 };
            eprintln!(
                "== attention d{d} causal={causal} bias={bias}: {} admitted plain, {} under the first forms",
                plain.len(),
                admitted.len()
            );
            for (c, ns) in timed.iter().zip(ns) {
                let what = if *c == pick { " (tuner's pick)" } else { "" };
                let time = ns.map_or("failed".into(), |ns| {
                    format!("{:8.1} us {:5.1} TFLOP/s", ns as f64 / 1e3, flops / ns as f64 / 1e3)
                });
                eprintln!("  {}: {time}{what}", label(c));
            }
        }
    });
}

/// What the backend allocates for a decoder step's key-warp configs on the
/// device: registers, shared memory (the rings alone: the merged states
/// reuse them) and any spill, printed (run with `--ignored --nocapture`);
/// never asserts.
#[test]
#[ignore = "perf probe: needs a GPU"]
fn key_warp_resources_probe() {
    let Some(planner) = super::device_planner() else { return };
    let pool = crate::tune::ScratchPool::default();
    for (key_warps, bkv) in [(1, 16), (2, 16), (4, 16), (1, 32), (2, 32), (4, 32)] {
        let cfg = FaCfg { key_warps, ..FaCfg::new(16, bkv, 2) };
        let spec = probe_spec(5, 20, 1, 1500, 64, AttnMask::default());
        let spec = AttnSpec { cfg, ..spec };
        let lowering = cfg.lowering(&planner.device);
        let Some((plan, _)) = crate::tune::scratch_plan(flash_attention(&spec), &lowering, &pool) else {
            eprintln!("{}: does not compile", label(&cfg));
            continue;
        };
        eprintln!("{} ({} B static smem): {:?}", label(&cfg), cfg.smem_bytes(64, false), crate::tune::resources(&plan));
    }
}

/// The tuner's d = 128 attention kernel on gfx1151 compiles without
/// spilling: admission drops what spills while a clean config remains.
#[test]
fn the_tuned_d128_attention_does_not_spill_on_gfx1151() {
    let Some(planner) = super::device_planner()
        .filter(|p| p.device.target.arch == svod_dtype::GpuArch::Amd(svod_dtype::AmdArch::Gfx1151))
    else {
        eprintln!("skipped: no gfx1151");
        return;
    };
    let spec = probe_spec(4, 8, 2048, 2048, 128, AttnMask::default());
    let (.., formed) = device_attention(&planner, &spec);
    let spec = AttnSpec { cfg: formed, ..spec };
    let prog = flash_attention(&spec);
    let pool = crate::tune::ScratchPool::default();
    let (plan, _) = crate::tune::scratch_plan(prog, &formed.lowering(&planner.device), &pool).expect("compiles");
    let resources = crate::tune::resources(&plan);
    eprintln!("{}: {resources:?}", label(&formed));
    assert!(!resources.is_empty() && !crate::tune::spilled(&resources), "{formed:?} spills: {resources:?}");
}

/// Every attention config the construct spans against the tuner's admitted
/// shortlist, timed on the device for the probe shapes and Qwen3's: prints
/// each config's time, the shortlist's members marked (run with `--ignored
/// --nocapture`); never asserts.
#[test]
#[ignore = "perf probe: needs a GPU"]
fn attention_candidates_probe() {
    let Some(planner) = super::device_planner() else { return };
    let target = &planner.device.target;
    for (batch, heads, t, d, causal) in [
        (8usize, 16usize, 512usize, 128usize, false),
        (4, 8, 2048, 64, false),
        (4, 8, 2048, 64, true),
        (4, 8, 2048, 128, false),
        (4, 8, 2048, 128, true),
        (5, 20, 1, 64, false),
    ] {
        let tk = if t == 1 { 1500 } else { t };
        let spec = probe_spec(batch, heads, t, tk, d, AttnMask { causal, ..AttnMask::default() });
        let (shortlist, ..) = device_attention(&planner, &spec);
        let mut cfgs: Vec<FaCfg> = shortlist.clone();
        for bq in [16, 32, 64, 128] {
            for bkv in [16, 32, 64, 128] {
                for unroll in [false, true] {
                    let cfg = FaCfg { unroll, ..FaCfg::new(bq, bkv, 2) };
                    if cfg.fits(target, d, false)
                        && cfg.smem_bytes(d, false) <= target.smem_bytes
                        && !cfgs.contains(&cfg)
                    {
                        cfgs.push(cfg);
                    }
                }
            }
        }
        let flops = 4.0 * (batch * heads * t * tk * d) as f64 / if causal { 2.0 } else { 1.0 };
        let ns = crate::tune::measure(
            cfgs.iter()
                .map(|&c| vec![(flash_attention(&AttnSpec { cfg: c, ..spec.clone() }), c.lowering(&planner.device))]),
        );
        eprintln!(
            "== attention b{batch} h{heads} t{t} tk{tk} d{d} causal={causal} ({} shortlist, {} lattice)",
            shortlist.len(),
            cfgs.len()
        );
        for (c, ns) in cfgs.iter().zip(ns) {
            let set = if shortlist.contains(c) { "short" } else { "lattice" };
            match ns {
                Some(ns) => eprintln!(
                    "  {set:7} bq{} bkv{} unroll={}: {:8.1} us {:6.1} TFLOP/s",
                    c.bq,
                    c.bkv,
                    c.unroll,
                    ns as f64 / 1e3,
                    flops / ns as f64 / 1e3
                ),
                None => eprintln!("  {set:7} bq{} bkv{} unroll={}: failed", c.bq, c.bkv, c.unroll),
            }
        }
    }
}

/// A Whisper large-v3 decoder step on the device: self attention over a
/// layer-packed cache with the step's key appended, and cross attention
/// over a shared 1500-key cache, tuned splits then unsplit. Prints
/// microseconds per step and never asserts (run with `--ignored --nocapture`).
/// tk1's single-query kernel under its split policy, last measured here on
/// the 3060 before its removal: self 89.1 µs, cross 78.8 µs (636 µs unsplit).
#[test]
#[ignore = "perf probe: needs a GPU"]
fn decode_throughput_probe() {
    use crate::ops::{self as tk, Attn, Cache, KeyMask};
    if super::device_target().is_none() {
        return;
    }
    crate::tune::scoped(true, || {
        let (b, heads, d, layers, layer) = (5usize, 20usize, 64usize, 32usize, 10usize);
        let f16 = |shape: &[usize], seed: u64| {
            let mut s = seed;
            let n: usize = shape.iter().product();
            let data: Vec<f32> = (0..n).map(|_| super::lcg(&mut s) as f32).collect();
            let t =
                Tensor::from_slice(&data).try_reshape(shape.iter().map(|&x| x as isize).collect::<Vec<_>>()).unwrap();
            let t = t.cast(DType::Float16).contiguous();
            t.realize().unwrap();
            t
        };
        let ints = |v: Vec<i32>| {
            let t = Tensor::from_slice(&v);
            t.realize().unwrap();
            t
        };
        let q = f16(&[b, 1, heads, d], 1);
        let self_k = f16(&[b, 448, layers * heads, d], 2);
        let self_v = f16(&[b, 448, layers * heads, d], 3);
        let (k_app, v_app) = (f16(&[b, 1, heads, d], 4), f16(&[b, 1, heads, d], 5));
        let lens = ints(vec![200; b]);
        let cross_k = f16(&[1, 1500, layers * heads, d], 6);
        let cross_v = f16(&[1, 1500, layers * heads, d], 7);
        let map = ints(vec![0; b]);

        let mut plans: Vec<(String, svod_runtime::ExecutionPlan)> = vec![];
        for (name, splits) in [("tuned", None), ("unsplit", Some(1))] {
            let cache = Cache {
                head_start: layer * heads,
                kv_heads: heads,
                row_map: None,
                appended: Some((&k_app, &v_app)),
                app_lens: None,
            };
            let opts = Attn { keys: KeyMask::Lens(&lens), cache: Some(cache), splits, ..Attn::default() };
            let o = tk::attention(&q, &self_k, &self_v, opts).unwrap();
            plans.push((format!("tk3 self {name}"), o.prepare().unwrap()));
            let cache = Cache {
                head_start: layer * heads,
                kv_heads: heads,
                row_map: Some(&map),
                appended: None,
                app_lens: None,
            };
            let o =
                tk::attention(&q, &cross_k, &cross_v, Attn { cache: Some(cache), splits, ..Attn::default() }).unwrap();
            plans.push((format!("tk3 cross {name}"), o.prepare().unwrap()));
        }
        let warm = std::time::Instant::now();
        while warm.elapsed().as_millis() < 500 {
            plans[0].1.execute().unwrap();
        }
        let mut best = vec![f64::INFINITY; plans.len()];
        for _ in 0..4 {
            for (i, (_, plan)) in plans.iter().enumerate() {
                for _ in 0..5 {
                    if let Some(run) = crate::tune::run(plan) {
                        best[i] = best[i].min(run.as_secs_f64() * 1e6);
                    }
                }
            }
        }
        for ((label, _), us) in plans.iter().zip(&best) {
            eprintln!("{label}: {us:.1} us");
        }
    });
}
