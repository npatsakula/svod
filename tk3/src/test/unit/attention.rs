//! Flash attention: the program against a direct softmax reference on the
//! host, and the lowered kernel against the program on the GPU.

use svod_dtype::{DType, DeviceSpec, ScalarDType, default_device::default_device};
use svod_tensor::Tensor;
use test_case::test_case;

use crate::atoms::Target;
use crate::build::BF16;
use crate::interp::{round_to, run};
use crate::ir::Program;
use crate::kernels::Batch;
use crate::kernels::attention::{
    AttnMask, AttnSpec, Bias, Cache, CombineSpec, FaCfg, attention, combine, key_mask_stride,
};
use crate::kernels::rows::NormCfg;
use crate::launch::graph_launch;
use crate::launch::graph_launch_all;

fn flash_attention(spec: &AttnSpec) -> Program {
    attention::<BF16>(spec)
}

fn lcg(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0
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
    /// `[batch, kv_heads, d]` appended keys and values.
    k_app: Vec<f64>,
    v_app: Vec<f64>,
    /// `[rows, heads, t, key_mask_stride(keys)]` in `2·[-1, 1)`, the padding
    /// columns past `keys` a large value nothing may read; empty unbiased.
    bias: Vec<f64>,
}

/// The scored keys: the cached ones and the appended row.
fn keys(spec: &AttnSpec) -> usize {
    spec.tk + usize::from(spec.cache.is_some_and(|c| c.appended))
}

fn case(spec: AttnSpec, lens: &[i64]) -> Case {
    let mut seed = 5;
    let n = |elems: usize, seed: &mut u64| -> Vec<f64> {
        (0..elems).map(|_| round_to(ScalarDType::BFloat16, lcg(seed))).collect()
    };
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
    let (k_app, v_app) = (n(batch * spec.kv_heads * spec.d, &mut seed), n(batch * spec.kv_heads * spec.d, &mut seed));
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
    Case { spec, q, k, v, lens: lens.to_vec(), key_mask, seg_start, row_map, k_app, v_app, bias }
}

/// Direct attention in f64 with the same masks and cache reads.
fn reference(c: &Case) -> Vec<f64> {
    let s = &c.spec;
    let m = s.mask;
    let cache =
        s.cache.unwrap_or(Cache { rows: 0, heads_total: s.kv_heads, head_start: 0, row_map: false, appended: false });
    let (stride, kv_stride, group) = (s.heads * s.d, cache.heads_total * s.d, s.heads / s.kv_heads);
    let mut out = vec![0.0; s.batch.capacity() * s.t * stride];
    for b in 0..s.batch.capacity() {
        // An empty lane sees key 0 unless a row is appended, as the kernel does.
        let floor = usize::from(!cache.appended);
        let len = if m.key_lens { (c.lens[b] as usize).max(floor) } else { s.tk };
        let row = if cache.row_map { c.row_map[b] as usize } else { b };
        let keys = keys(s);
        let bias_stride = key_mask_stride(keys);
        let bias_row = if m.bias == Some(Bias::PerBatch) { b } else { 0 };
        for h in 0..s.heads {
            let kv_head = cache.head_start + h / group;
            let key = |j: usize, dd: usize, cached: &[f64], appended: &[f64]| {
                if j < s.tk {
                    cached[(row * s.tk + j) * kv_stride + kv_head * s.d + dd]
                } else {
                    appended[(b * s.kv_heads + h / group) * s.d + dd]
                }
            };
            for i in 0..s.t {
                let qi = &c.q[(b * s.t + i) * stride + h * s.d..][..s.d];
                let hidden = |j: usize| {
                    j < s.tk
                        && (j >= len
                            || (m.causal && j > i)
                            || m.window.is_some_and(|(l, r)| j + l < i || j > i + r)
                            || (m.key_mask && c.key_mask[b * key_mask_stride(s.tk) + j] == 0)
                            || (m.seg_start && (j as i64) < c.seg_start[b * s.t + i]))
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
            "k_app" => c.k_app.clone(),
            "v_app" => c.v_app.clone(),
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

const NONE: AttnMask =
    AttnMask { causal: false, window: None, key_lens: false, key_mask: false, seg_start: false, bias: None };
const WINDOW: AttnMask = AttnMask { window: Some((20, 5)), key_lens: true, ..NONE };
const KEY_MASK: AttnMask = AttnMask { key_mask: true, ..NONE };
const SEGMENTS: AttnMask = AttnMask { causal: true, seg_start: true, ..NONE };
const EVERY: AttnMask =
    AttnMask { causal: true, window: Some((50, 0)), key_lens: true, key_mask: true, seg_start: true, bias: None };
const SHARED: AttnMask = AttnMask { bias: Some(Bias::Shared), ..NONE };
const PER_BATCH: AttnMask = AttnMask { bias: Some(Bias::PerBatch), ..NONE };
const CAUSAL_BIAS: AttnMask = AttnMask { causal: true, ..PER_BATCH };
const LENS_BIAS: AttnMask = AttnMask { key_lens: true, ..PER_BATCH };
const EVERY_BIAS: AttnMask = AttnMask { bias: Some(Bias::Shared), ..EVERY };

/// `spec` with its cache's scores biased per batch.
fn biased(spec: AttnSpec) -> AttnSpec {
    AttnSpec { mask: AttnMask { bias: Some(Bias::PerBatch), ..spec.mask }, ..spec }
}

fn with(cfg: FaCfg, spec: AttnSpec) -> AttnSpec {
    AttnSpec { cfg, ..spec }
}

#[test_case(spec(64, 64, 64, 64, 64, false, false), &[64, 64]; "one block")]
#[test_case(spec(128, 192, 64, 64, 64, false, false), &[192, 192]; "three key blocks")]
#[test_case(spec(128, 128, 64, 64, 64, true, false), &[128, 128]; "causal")]
#[test_case(spec(128, 128, 64, 64, 64, false, true), &[100, 7]; "key lengths")]
#[test_case(spec(128, 128, 64, 64, 32, true, true), &[128, 50]; "causal with lengths, narrow kv")]
#[test_case(masked(128, 128, 64, 64, 32, WINDOW), &[128, 90]; "window with lengths")]
#[test_case(masked(128, 128, 64, 64, 64, KEY_MASK), &[128, 128]; "key mask")]
#[test_case(masked(128, 128, 64, 64, 32, SEGMENTS), &[128, 128]; "causal segments")]
#[test_case(masked(128, 128, 64, 64, 32, EVERY), &[128, 100]; "every mask")]
#[test_case(masked(96, 96, 48, 64, 64, SEGMENTS), &[96, 96]; "d 48")]
#[test_case(masked(128, 128, 64, 64, 64, PER_BATCH), &[128, 128]; "bias per batch")]
#[test_case(masked(100, 130, 64, 64, 64, SHARED), &[130, 130]; "shared bias, ragged")]
#[test_case(masked(128, 128, 64, 64, 32, CAUSAL_BIAS), &[128, 128]; "causal bias")]
#[test_case(masked(99, 99, 64, 64, 64, LENS_BIAS), &[99, 41]; "bias with key lengths, odd keys")]
#[test_case(masked(128, 128, 128, 64, 32, LENS_BIAS), &[128, 70]; "d 128 bias with key lengths")]
#[test_case(masked(128, 128, 64, 64, 32, EVERY_BIAS), &[128, 100]; "every mask and a bias")]
fn program_matches_a_direct_softmax(spec: AttnSpec, lens: &[i64]) {
    let c = case(spec.clone(), lens);
    let got = run(&flash_attention(&spec), params(&c), &[("b", 2)]).unwrap();
    let got = output(&c, got, 2);
    let want = reference(&c);
    let mut worst = 0.0f64;
    for (g, w) in got.iter().zip(&want) {
        worst = worst.max((g - w).abs());
    }
    assert!(worst < 2e-2, "max abs diff {worst}");
}

/// A decoder step: one query row against a cache of several layers' heads,
/// rows mapped, the step's own key appended after the prefix.
fn cached(
    t: usize,
    tk: usize,
    [heads, kv_heads, d]: [usize; 3],
    row_map: bool,
    appended: bool,
    splits: usize,
) -> AttnSpec {
    let mut spec = masked(t, tk, d, 64, 32, AttnMask { key_lens: true, ..AttnMask::default() });
    spec.heads = heads;
    spec.kv_heads = kv_heads;
    spec.cache = Some(Cache { rows: 3, heads_total: 3 * kv_heads, head_start: kv_heads, row_map, appended });
    spec.cfg.splits = splits;
    spec
}

#[test_case(cached(1, 100, [4, 4, 64], false, false, 1), &[100, 37]; "cache slice")]
#[test_case(cached(1, 100, [4, 2, 64], true, true, 1), &[100, 37]; "row map, appended, gqa")]
#[test_case(cached(1, 200, [4, 4, 64], true, true, 4), &[200, 70]; "four splits")]
#[test_case(cached(5, 96, [2, 2, 128], false, true, 2), &[64, 96]; "d 128, two splits, five queries")]
#[test_case(cached(1, 64, [2, 2, 64], false, false, 4), &[64, 40]; "more splits than blocks for a row")]
#[test_case(cached(1, 7, [6, 6, 64], false, true, 1), &[0, 7]; "appended row only, one split")]
#[test_case(cached(1, 7, [6, 6, 64], false, true, 2), &[0, 3]; "appended row only, two splits")]
#[test_case(biased(cached(1, 100, [4, 2, 64], true, true, 1)), &[100, 37]; "biased: row map, appended, gqa")]
#[test_case(biased(cached(5, 96, [2, 2, 128], false, true, 2)), &[64, 96]; "biased: d 128, two splits")]
#[test_case(biased(cached(1, 200, [4, 4, 64], false, false, 4)), &[200, 70]; "biased: four splits")]
fn cached_program_matches_a_direct_softmax(spec: AttnSpec, lens: &[i64]) {
    program_matches_a_direct_softmax(spec, lens);
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

#[test_case(spec(64, 64, 64, 64, 64, false, false), &[64, 64]; "one key block")]
#[test_case(spec(128, 128, 64, 64, 64, false, false), &[128, 128]; "plain")]
#[test_case(spec(256, 256, 64, 64, 64, true, false), &[256, 256]; "causal")]
#[test_case(spec(128, 256, 64, 64, 64, false, true), &[200, 33]; "key lengths")]
#[test_case(spec(128, 128, 128, 64, 32, true, true), &[128, 64]; "d 128, causal with lengths")]
#[test_case(with(FaCfg::new(128, 64, 3), spec(256, 192, 64, 0, 0, true, true)), &[192, 70]; "bq 128, three stages")]
#[test_case(with(FaCfg::new(128, 32, 2), spec(200, 130, 128, 0, 0, false, true)), &[130, 99]; "d 128, bq 128, ragged")]
#[test_case(masked(128, 128, 64, 64, 32, WINDOW), &[128, 90]; "window with lengths")]
#[test_case(masked(100, 100, 64, 64, 64, KEY_MASK), &[100, 100]; "key mask, ragged")]
#[test_case(masked(128, 128, 64, 64, 32, SEGMENTS), &[128, 128]; "causal segments")]
#[test_case(masked(128, 128, 128, 64, 32, EVERY), &[128, 100]; "every mask, d 128")]
#[test_case(with(FaCfg::new(64, 64, 3), masked(200, 200, 48, 0, 0, SEGMENTS)), &[200, 200]; "d 48, three stages")]
#[test_case(cached(1, 100, [4, 2, 64], true, true, 1), &[100, 37]; "cache: row map, appended, gqa")]
#[test_case(cached(1, 200, [4, 4, 64], true, true, 4), &[200, 70]; "cache: four splits")]
#[test_case(cached(5, 96, [2, 2, 128], false, true, 2), &[64, 96]; "cache: d 128, two splits")]
#[test_case(cached(1, 7, [6, 6, 64], false, true, 1), &[0, 7]; "cache: appended row only")]
#[test_case(cached(1, 7, [6, 6, 64], false, true, 2), &[0, 3]; "cache: appended row only, two splits")]
#[test_case(with(FaCfg { splits: 3, ..FaCfg::new(16, 64, 2) }, cached(1, 1500, [4, 4, 64], true, false, 1)), &[1500, 1200]; "cache: bq 16, three splits of 1500 keys")]
#[test_case(masked(128, 128, 64, 64, 64, PER_BATCH), &[128, 128]; "bias per batch")]
#[test_case(masked(100, 130, 64, 64, 64, SHARED), &[130, 130]; "shared bias, ragged")]
#[test_case(masked(256, 256, 64, 64, 64, CAUSAL_BIAS), &[256, 256]; "causal bias")]
#[test_case(masked(99, 99, 64, 64, 64, LENS_BIAS), &[99, 41]; "bias with key lengths, odd keys")]
#[test_case(with(FaCfg::new(128, 32, 2), masked(200, 130, 128, 0, 0, LENS_BIAS)), &[130, 99]; "d 128 bias with lengths, bq 128")]
#[test_case(masked(128, 128, 128, 64, 32, EVERY_BIAS), &[128, 100]; "every mask and a bias, d 128")]
#[test_case(biased(cached(1, 100, [4, 2, 64], true, true, 1)), &[100, 37]; "cache biased: row map, appended, gqa")]
#[test_case(biased(cached(5, 96, [2, 2, 128], false, true, 2)), &[64, 96]; "cache biased: d 128, two splits")]
fn kernel_matches_the_program(spec: AttnSpec, lens: &[i64]) {
    let device = default_device();
    let Some(target) = matches!(device, DeviceSpec::Cuda { .. }).then(|| Target::for_device(&device)).flatten() else {
        eprintln!("skipped: no CUDA device");
        return;
    };
    let batch = spec.batch.capacity();
    let c = case(spec.clone(), lens);
    let raw = run(&flash_attention(&spec), params(&c), &[("b", batch as i64)]).unwrap();
    let want = output(&c, raw.clone(), batch as i64);
    let lowering = spec.cfg.lowering(target.clone());
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
    // The partials themselves, before the merge.
    for (i, name) in
        ["o_part", "m_part", "l_part"].into_iter().enumerate().take(if spec.cfg.splits > 1 { 3 } else { 0 })
    {
        let part = &outs[3 + i];
        let mut plan = part.prepare().unwrap();
        plan.execute_with_vars(&[("b", batch as i64)]).unwrap();
        let mut bytes = vec![0u8; raw[3 + i].len() * 4];
        part.buffer().unwrap().copyout(&mut bytes).unwrap();
        let got: Vec<f64> = bytes.chunks(4).map(|b| f64::from(f32::from_le_bytes(b.try_into().unwrap()))).collect();
        let worst = got.iter().zip(&raw[3 + i]).map(|(g, w)| (g - w).abs() / w.abs().max(1.0)).fold(0.0, f64::max);
        let at = got.iter().zip(&raw[3 + i]).position(|(g, w)| (g - w).abs() / w.abs().max(1.0) > 1e-2);
        eprintln!("{name}: max rel diff {worst:.3e} vs interpreter, first over 1e-2 at {at:?} of {}", got.len());
    }
    let out = if spec.cfg.splits == 1 {
        outs[3].clone()
    } else {
        let merge = combine_spec(&spec);
        let o = Tensor::empty(&[elems], DType::BFloat16);
        graph_launch(combine::<BF16>(&merge), &merge.cfg.lowering(target), &[&outs[3], &outs[4], &outs[5], &o]).unwrap()
    };
    let mut plan = out.prepare().unwrap();
    plan.execute_with_vars(&[("b", batch as i64)]).unwrap();
    let mut bytes = vec![0u8; elems * 2];
    out.buffer().unwrap().copyout(&mut bytes).unwrap();
    let got: Vec<f32> =
        bytes.chunks(2).map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16)).collect();
    let reference = reference(&c);
    let mut worst = 0.0f64;
    let (mut over, mut worst_at) = (0usize, 0usize);
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        let d = (*g as f64 - w).abs();
        if d > worst {
            worst = d;
            worst_at = i;
        }
        over += usize::from(d > 1e-2);
    }
    let vs_ref = got.iter().zip(&reference).map(|(g, w)| (*g as f64 - w).abs()).fold(0.0, f64::max);
    let stride = spec.heads * spec.d;
    eprintln!(
        "max abs diff {worst:.3e} at batch {} row {} head {} col {} ({over} elements over 1e-2); vs f64 reference {vs_ref:.3e}",
        worst_at / (spec.t * stride),
        worst_at / stride % spec.t,
        worst_at % stride / spec.d,
        worst_at % spec.d
    );
    assert!(worst < 2e-2, "max abs diff {worst}");
}

/// The split merge alone: random partials through the kernel against the
/// interpreter.
#[test_case(1, 4, 64, 4, 2; "four rows, four splits")]
#[test_case(5, 2, 128, 2, 2; "ten rows, d 128, two splits")]
#[test_case(1, 4, 64, 3, 1; "three splits, one batch")]
#[test_case(7, 3, 64, 2, 2; "twenty-one rows, partial block")]
fn combine_matches_the_program(t: usize, heads: usize, d: usize, splits: usize, batch: usize) {
    let device = default_device();
    let Some(target) = matches!(device, DeviceSpec::Cuda { .. }).then(|| Target::for_device(&device)).flatten() else {
        eprintln!("skipped: no CUDA device");
        return;
    };
    let spec = CombineSpec { batch: Batch::Static(batch), t, heads, d, splits, cfg: NormCfg { br: 4 } };
    let rows = splits * batch * t * heads;
    let mut seed = 9;
    let f32s = |n: usize, seed: &mut u64, f: &dyn Fn(f64) -> f64| -> Vec<f64> {
        (0..n).map(|_| round_to(ScalarDType::Float32, f(lcg(seed)))).collect()
    };
    let o_part = f32s(rows * d, &mut seed, &|x| 10.0 * x);
    let m_part = f32s(rows, &mut seed, &|x| 4.0 * x);
    let l_part = f32s(rows, &mut seed, &|x| 1.5 + x);
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
    let out = graph_launch(combine::<BF16>(&spec), &spec.cfg.lowering(target), &refs).unwrap();
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

/// Flash attention throughput at B 4, H 8, T 2048 bf16: causal and plain,
/// head dim 64 and 128, and d 64 with a per-batch bias (run with `--ignored
/// --nocapture --release`). tk1's `flash_attention_with`, last measured here
/// on the 3060 before its removal, in TFLOP/s: d64 23.0, d64 causal 20.7,
/// d128 22.3, d128 causal 16.5.
#[test]
#[ignore = "perf probe: needs a CUDA device"]
fn attention_throughput_probe() {
    let device = default_device();
    let Some(target) = matches!(device, DeviceSpec::Cuda { .. }).then(|| Target::for_device(&device)).flatten() else {
        return;
    };
    let (batch, heads, t) = (4usize, 8usize, 2048usize);
    for (d, causal, bq, bkv, bias) in [
        (64, false, 64, 64, false),
        (64, false, 64, 64, true),
        (64, true, 64, 64, false),
        (64, true, 64, 64, true),
        (128, false, 64, 32, false),
        (128, true, 64, 32, false),
    ] {
        let bias = bias.then_some(Bias::PerBatch);
        let spec = AttnSpec {
            batch: Batch::Var { name: "b".into(), min: 1, max: batch as i64 },
            t,
            tk: t,
            heads,
            kv_heads: heads,
            d,
            mask: AttnMask { causal, bias, ..AttnMask::default() },
            cache: None,
            scale: 1.0 / (d as f32).sqrt(),
            cfg: FaCfg::new(bq, bkv, 2),
        };
        let c = case(spec.clone(), &[]);
        let to_bf16 =
            |v: &[f64]| Tensor::from_slice(v.iter().map(|&x| x as f32).collect::<Vec<_>>()).cast(DType::BFloat16);
        let (q, k, v) = (to_bf16(&c.q), to_bf16(&c.k), to_bf16(&c.v));
        let biases = bias.map(|_| to_bf16(&c.bias));
        for x in [&q, &k, &v].into_iter().chain(&biases) {
            x.realize().unwrap();
        }
        let flops = 4.0 * batch as f64 * heads as f64 * t as f64 * t as f64 * d as f64 / if causal { 2.0 } else { 1.0 };
        let mut plans: Vec<(String, svod_runtime::ExecutionPlan)> = vec![];
        {
            let lowering = spec.cfg.lowering(target.clone());
            let o = Tensor::empty(&[batch * t * heads * d], DType::BFloat16);
            let ins: Vec<&Tensor> = [&q, &k, &v, &o].into_iter().chain(&biases).collect();
            let out = graph_launch(flash_attention(&spec), &lowering, &ins).unwrap();
            let mut plan = out.prepare().unwrap();
            plan.execute_with_vars(&[("b", batch as i64)]).unwrap();
            plans.push((format!("tk3 d{d} causal={causal} bias={} bq{bq} bkv{bkv}", bias.is_some()), plan));
        }
        // The 3060 idles at a low clock: spin the first plan for half a second.
        let warm = std::time::Instant::now();
        while warm.elapsed().as_millis() < 500 {
            plans[0].1.execute().unwrap();
        }
        let mut best = vec![f64::INFINITY; plans.len()];
        for _ in 0..4 {
            for (i, (_, plan)) in plans.iter().enumerate() {
                for _ in 0..5 {
                    let run = plan
                        .execute_profiled()
                        .unwrap()
                        .iter()
                        .filter_map(|kp| Some((kp.gpu_end_ns? - kp.gpu_start_ns?) as f64 * 1e-9))
                        .fold(0.0, f64::max);
                    best[i] = best[i].min(run);
                }
            }
        }
        for ((label, _), secs) in plans.iter().zip(&best) {
            eprintln!("{label}: {:.3} ms, {:.1} TFLOP/s", secs * 1e3, flops / secs / 1e12);
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
#[ignore = "perf probe: needs a CUDA device"]
fn decode_throughput_probe() {
    use crate::ops::{self as tk, Attn, Cache, KeyMask};
    if !matches!(default_device(), DeviceSpec::Cuda { .. }) {
        return;
    }
    crate::tune::set_enabled(true);
    let (b, heads, d, layers, layer) = (5usize, 20usize, 64usize, 32usize, 10usize);
    let f16 = |shape: &[usize], seed: u64| {
        let mut s = seed;
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|_| lcg(&mut s) as f32).collect();
        let t = Tensor::from_slice(&data).try_reshape(shape.iter().map(|&x| x as isize).collect::<Vec<_>>()).unwrap();
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
        let cache =
            Cache { head_start: layer * heads, kv_heads: heads, row_map: None, appended: Some((&k_app, &v_app)) };
        let opts = Attn { keys: KeyMask::Lens(&lens), cache: Some(cache), splits, ..Attn::default() };
        let o = tk::attention(&q, &self_k, &self_v, opts).unwrap();
        plans.push((format!("tk3 self {name}"), o.prepare().unwrap()));
        let cache = Cache { head_start: layer * heads, kv_heads: heads, row_map: Some(&map), appended: None };
        let o = tk::attention(&q, &cross_k, &cross_v, Attn { cache: Some(cache), splits, ..Attn::default() }).unwrap();
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
                let run: f64 = plan
                    .execute_profiled()
                    .unwrap()
                    .iter()
                    .filter_map(|kp| Some((kp.gpu_end_ns? - kp.gpu_start_ns?) as f64 * 1e-3))
                    .sum();
                best[i] = best[i].min(run);
            }
        }
    }
    for ((label, _), us) in plans.iter().zip(&best) {
        eprintln!("{label}: {us:.1} us");
    }
}
