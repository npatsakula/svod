//! Every op against its own graph fallback on a CUDA device, on awkward
//! shapes; skipped unless `SVOD_DEVICE` names one.

use svod_dtype::{DType, DeviceSpec, default_device::default_device};
use svod_ir::{Op, SInt, ops};
use svod_tensor::{Tensor, Variable};
use test_case::test_case;

use crate::ops::{self as tk, Act, Attn, Cache, KeyMask, Linear, Qkv};

/// A CUDA device to run on, with tuning off so a test does not measure
/// every shape it touches (the untuned pick runs).
fn device() -> bool {
    crate::tune::set_enabled(false);
    let device = default_device();
    let ok = matches!(device, DeviceSpec::Cuda { .. }) && tk::supported(&device);
    if !ok {
        eprintln!("skipped: no CUDA device");
    }
    ok
}

/// A realized `shape` tensor of `dtype` with values in `scale·[-1, 1)`.
fn rand(shape: &[usize], seed: u64, scale: f32, dtype: DType) -> Tensor {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let data: Vec<f32> = (0..shape.iter().product())
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32 * 2.0 - 1.0) * scale
        })
        .collect();
    let t = Tensor::from_slice(&data).cast(dtype).try_reshape(shape.iter().map(|&d| d as isize).collect::<Vec<_>>());
    let t = t.unwrap().contiguous();
    t.realize().unwrap();
    t
}

/// `t` on the host; a dim bound to a variable reads at its bound value.
fn values(t: &Tensor) -> Vec<f32> {
    let t = t.cast(DType::Float32);
    let shape = t.shape().unwrap();
    if shape.iter().all(SInt::is_const) {
        return t.to_vec::<f32>().unwrap();
    }
    let live = |d: &SInt| match d {
        SInt::Const(v) => *v,
        SInt::Symbolic(u) => match u.op() {
            Op::Bind(ops::Bind { value, .. }) => match value.op() {
                Op::Const(c) => c.0.try_int().unwrap() as usize,
                other => panic!("bound to {other:?}"),
            },
            other => panic!("unbound dim {other:?}"),
        },
        SInt::Infer => unreachable!(),
    };
    let live_batch = live(&shape[0]);
    let row: usize = shape[1..].iter().map(live).product();
    rebound(&t, live_batch, row)
}

/// The first `batch` rows of `row` elements of `t`'s plan executed with the
/// batch variable `b` bound to `batch`.
fn rebound(t: &Tensor, batch: usize, row: usize) -> Vec<f32> {
    let mut plan = t.cast(DType::Float32).contiguous().prepare().unwrap();
    plan.execute_with_vars(&[("b", batch as i64)]).unwrap();
    let mut bytes = vec![0u8; batch * row * 4];
    plan.output_buffer().unwrap().copyout_prefix(&mut bytes).unwrap();
    bytes.chunks(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
}

/// The graph kernel named `name` is in `t`'s graph.
fn assert_kernel(t: &Tensor, name: &str) {
    let calls: Vec<String> = t
        .uop()
        .toposort()
        .iter()
        .filter_map(|u| match u.op() {
            Op::Call(ops::Call { info, .. }) => info.name.clone(),
            _ => None,
        })
        .collect();
    assert!(calls.iter().any(|n| n == name), "no {name} kernel among {calls:?}");
}

/// `|got - want| ≤ tol·max(|want|, 1)` everywhere.
fn assert_close(what: &str, got: &Tensor, want: &Tensor, tol: f32) {
    assert_close_values(what, &values(got), &values(want), tol);
}

fn assert_close_values(what: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{what}: element count");
    let mut worst = 0.0f32;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let diff = (g - w).abs();
        worst = worst.max(diff);
        assert!(diff <= tol * w.abs().max(1.0), "{what}[{i}] = {g}, graph {w}");
    }
    eprintln!("{what}: max abs diff {worst:.3e} over {} elements", got.len());
}

/// Dim 0 of every tensor shrunk to a bound variable, the way the JIT
/// `batch_var` hands batched inputs to the model.
fn bound(ts: &[&Tensor], var: &svod_tensor::BoundVariable) -> Vec<Tensor> {
    ts.iter()
        .map(|t| {
            let mut ranges: Vec<Option<(SInt, SInt)>> = vec![None; t.shape().unwrap().len()];
            ranges[0] = Some((SInt::Const(0), var.as_sint()));
            t.try_shrink(ranges).unwrap()
        })
        .collect()
}

fn first(ts: &[&Tensor], n: usize) -> Vec<Tensor> {
    ts.iter()
        .map(|t| {
            let mut ranges: Vec<Option<(isize, isize)>> = vec![None; t.shape().unwrap().len()];
            ranges[0] = Some((0, n as isize));
            t.try_shrink(ranges).unwrap().contiguous()
        })
        .collect()
}

// ---- linear ------------------------------------------------------------------

/// The graph over f32 copies of the operands: the kernel rounds once, at the
/// store, so it is held to the unrounded epilogue.
fn linear_reference(x: &Tensor, w: &Tensor, opts: Linear) -> Tensor {
    let f32 = |t: &Tensor| t.cast(DType::Float32);
    let (bias, residual) = (opts.bias.map(f32), opts.residual.map(f32));
    let opts = Linear { bias: bias.as_ref(), residual: residual.as_ref(), ..opts };
    tk::linear::graph(&f32(x), &f32(w), opts).unwrap()
}

#[test_case(&[37], 96, 64, Act::Gelu, true, true; "m37 n96 bias gelu residual")]
#[test_case(&[3, 50], 64, 96, Act::None, false, true; "two lead dims, residual")]
#[test_case(&[300], 200, 48, Act::Silu, true, false; "bk 16, silu")]
#[test_case(&[2048], 512, 512, Act::None, false, false; "large plain")]
fn linear_matches_the_graph(lead: &[usize], n: usize, k: usize, act: Act, bias: bool, residual: bool) {
    if !device() {
        return;
    }
    let x = rand(&[lead, &[k]].concat(), 1, 1.0, DType::BFloat16);
    let w = rand(&[n, k], 2, 0.5, DType::BFloat16);
    let b = rand(&[n], 3, 1.0, DType::BFloat16);
    let r = rand(&[lead, &[n]].concat(), 4, 2.0, DType::BFloat16);
    let opts = Linear { bias: bias.then_some(&b), act, gated: false, residual: residual.then_some(&r) };
    let y = tk::linear(&x, &w, opts).unwrap();
    assert_kernel(&y, "gemm");
    assert_close("linear", &y, &linear_reference(&x, &w, opts), 2e-2);
}

#[test_case(Act::Silu, true; "swiglu with bias")]
#[test_case(Act::Gelu, false; "geglu")]
fn gated_linear_matches_the_graph(act: Act, bias: bool) {
    if !device() {
        return;
    }
    let (m, n, k) = (77, 96, 128);
    let x = rand(&[m, k], 5, 1.0, DType::BFloat16);
    let w = rand(&[2 * n, k], 6, 0.3, DType::BFloat16);
    let b = rand(&[2 * n], 7, 1.0, DType::BFloat16);
    let opts = Linear { bias: bias.then_some(&b), act, gated: true, residual: None };
    let y = tk::linear(&x, &w, opts).unwrap();
    assert_eq!(y.shape().unwrap().to_vec(), [SInt::Const(m), SInt::Const(n)]);
    assert_kernel(&y, "gemm");
    assert_close("gated linear", &y, &linear_reference(&x, &w, opts), 2e-2);
}

/// Operands that are lazy views (a transpose, a reshape) are materialized
/// for the kernel.
#[test]
fn linear_takes_lazy_views() {
    if !device() {
        return;
    }
    let xt = rand(&[64, 40], 8, 1.0, DType::BFloat16);
    let x = xt.try_transpose(0, 1).unwrap().try_reshape([4, 10, 64]).unwrap();
    let w = rand(&[2, 32, 64], 9, 0.5, DType::BFloat16).try_reshape([64, 64]).unwrap();
    let y = tk::linear(&x, &w, Linear::default()).unwrap();
    assert_kernel(&y, "gemm");
    assert_close("linear of views", &y, &tk::linear::graph(&x, &w, Linear::default()).unwrap(), 2e-2);
}

/// A batch bound below its capacity: one GEMM over the live batches only,
/// the output shrunk to the live dim.
#[test]
fn linear_under_a_batch_variable() {
    if !device() {
        return;
    }
    let (cap, live, rows, n, k) = (4, 3, 37, 64, 64);
    let x = rand(&[cap, rows, k], 10, 1.0, DType::BFloat16);
    let r = rand(&[cap, rows, n], 11, 1.0, DType::BFloat16);
    let w = rand(&[2 * n, k], 12, 0.5, DType::BFloat16);
    let b = rand(&[2 * n], 13, 1.0, DType::BFloat16);
    let var = Variable::new("b", 1, cap as i64).bind(live as i64).unwrap();
    let [xb, rb] = <[Tensor; 2]>::try_from(bound(&[&x, &r], &var)).unwrap();
    let opts = Linear { bias: Some(&b), act: Act::Silu, gated: true, residual: Some(&rb) };
    let y = tk::linear(&xb, &w, opts).unwrap();
    assert_eq!(y.shape().unwrap()[0], var.as_sint());
    assert_kernel(&y, "gemm");
    let [xs, rs] = <[Tensor; 2]>::try_from(first(&[&x, &r], live)).unwrap();
    let want = linear_reference(&xs, &w, Linear { residual: Some(&rs), ..opts });
    assert_close("batched linear", &y, &want, 2e-2);
    // The same plan serves a smaller batch.
    let [xs, rs] = <[Tensor; 2]>::try_from(first(&[&x, &r], 1)).unwrap();
    let want = values(&linear_reference(&xs, &w, Linear { residual: Some(&rs), ..opts }));
    assert_close_values("rebound linear", &rebound(&y, 1, rows * n), &want, 2e-2);
}

// ---- attention ----------------------------------------------------------------

struct AttnCase {
    b: usize,
    t: usize,
    tk: usize,
    h: usize,
    h_kv: usize,
    d: usize,
    causal: bool,
    lens: Option<Vec<i32>>,
    window: Option<(usize, usize)>,
    /// A bool key mask hiding every fifth key, and packed segments of 30 rows.
    key_mask: bool,
    segments: bool,
}

struct AttnInputs {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    lens: Option<Tensor>,
    key_mask: Option<Tensor>,
    seg_start: Option<Tensor>,
}

impl AttnInputs {
    fn opts(&self, c: &AttnCase) -> Attn<'_> {
        let keys = match (&self.lens, &self.key_mask) {
            (Some(lens), _) => KeyMask::Lens(lens),
            (None, Some(mask)) => KeyMask::Bool(mask),
            (None, None) => KeyMask::None,
        };
        Attn { causal: c.causal, keys, window: c.window, seg_start: self.seg_start.as_ref(), ..Attn::default() }
    }
}

fn attn_inputs(c: &AttnCase, cap: usize) -> AttnInputs {
    let realized = |t: Tensor| {
        t.realize().unwrap();
        t
    };
    let q = rand(&[cap, c.t, c.h, c.d], 20, 1.0, DType::BFloat16);
    let k = rand(&[cap, c.tk, c.h_kv, c.d], 21, 1.0, DType::BFloat16);
    let v = rand(&[cap, c.tk, c.h_kv, c.d], 22, 1.0, DType::BFloat16);
    let lens = c.lens.as_ref().map(|l| {
        let mut l = l.clone();
        l.resize(cap, c.tk as i32);
        realized(Tensor::from_slice(&l))
    });
    let key_mask = c.key_mask.then(|| {
        let mask: Vec<bool> = (0..cap * c.tk).map(|x| (x / c.tk + x % c.tk) % 5 != 2).collect();
        realized(Tensor::from_slice(&mask).try_reshape([cap, c.tk]).unwrap())
    });
    let seg_start = c.segments.then(|| {
        let seg: Vec<i32> = (0..cap * c.t).map(|x| (x % c.t - x % c.t % 30) as i32).collect();
        realized(Tensor::from_slice(&seg).try_reshape([cap, c.t]).unwrap())
    });
    AttnInputs { q, k, v, lens, key_mask, seg_start }
}

/// `b` batches of `[t, tk]` query/key rows, `[h, h_kv]` heads of width `d`.
fn case(
    b: usize,
    [t, tk]: [usize; 2],
    [h, h_kv]: [usize; 2],
    d: usize,
    causal: bool,
    lens: Option<&[i32]>,
) -> AttnCase {
    let lens = lens.map(<[i32]>::to_vec);
    AttnCase { b, t, tk, h, h_kv, d, causal, lens, window: None, key_mask: false, segments: false }
}

#[test_case(case(2, [100, 100], [4, 4], 64, false, Some(&[100, 61])); "t100 key lens")]
#[test_case(case(1, [200, 200], [2, 2], 64, true, None); "causal t200")]
#[test_case(case(2, [128, 128], [8, 2], 64, true, Some(&[128, 77])); "gqa 8 over 2")]
#[test_case(case(2, [100, 100], [4, 2], 128, true, Some(&[90, 100])); "d128 gqa causal lens")]
#[test_case(case(2, [37, 150], [4, 4], 64, false, None); "cross attention")]
#[test_case(AttnCase { window: Some((64, 64)), ..case(2, [300, 300], [4, 4], 64, false, None) }; "window 64 each side")]
#[test_case(AttnCase { key_mask: true, ..case(2, [100, 100], [4, 4], 64, false, None) }; "bool key mask")]
#[test_case(AttnCase { key_mask: true, window: Some((10, 0)), ..case(2, [130, 130], [4, 2], 128, true, None) }; "d128 causal window bool mask")]
#[test_case(AttnCase { segments: true, ..case(2, [200, 200], [4, 2], 64, true, None) }; "causal packed segments")]
#[test_case(AttnCase { segments: true, key_mask: true, ..case(1, [96, 96], [4, 4], 48, true, None) }; "d48 causal segments bool mask")]
#[test_case(case(2, [150, 150], [4, 4], 48, false, Some(&[150, 100])); "d48 key lens")]
fn attention_matches_the_graph(c: AttnCase) {
    if !device() {
        return;
    }
    let inputs = attn_inputs(&c, c.b);
    let opts = inputs.opts(&c);
    let o = tk::attention(&inputs.q, &inputs.k, &inputs.v, opts).unwrap();
    assert_kernel(&o, "flash_attention");
    assert_close("attention", &o, &tk::attention::graph(&inputs.q, &inputs.k, &inputs.v, opts).unwrap(), 2e-2);
}

#[test]
fn attention_under_a_batch_variable() {
    if !device() {
        return;
    }
    let (cap, live) = (3, 2);
    let c = case(cap, [96, 96], [4, 2], 64, false, Some(&[96, 40, 5]));
    let AttnInputs { q, k, v, lens, .. } = attn_inputs(&c, cap);
    let lens = lens.unwrap();
    let var = Variable::new("b", 1, cap as i64).bind(live as i64).unwrap();
    let [qb, kb, vb, lb] = <[Tensor; 4]>::try_from(bound(&[&q, &k, &v, &lens], &var)).unwrap();
    let opts = |lens| Attn { keys: KeyMask::Lens(lens), scale: Some(0.2), ..Attn::default() };
    let o = tk::attention(&qb, &kb, &vb, opts(&lb)).unwrap();
    assert_eq!(o.shape().unwrap()[0], var.as_sint());
    assert_kernel(&o, "flash_attention");
    let [qs, ks, vs, ls] = <[Tensor; 4]>::try_from(first(&[&q, &k, &v, &lens], live)).unwrap();
    assert_close("batched attention", &o, &tk::attention::graph(&qs, &ks, &vs, opts(&ls)).unwrap(), 2e-2);
}

/// A decoder step: `t` queries per lane against a cache of three layers'
/// heads shared by two windows, each lane reading its window's row, the
/// step's own key appended; with and without key splits.
#[test_case(1, 64, 20, 1500, Some(4); "one query, 1500 keys, four splits")]
#[test_case(1, 128, 8, 300, None; "d 128 unsplit")]
#[test_case(3, 64, 4, 200, Some(2); "three queries, two splits")]
fn cached_attention_matches_the_graph(t: usize, d: usize, h: usize, tk: usize, splits: Option<usize>) {
    if !device() {
        return;
    }
    let (b, rows, layers) = (5, 2, 3);
    let q = rand(&[b, t, h, d], 40, 1.0, DType::BFloat16);
    let k = rand(&[rows, tk, layers * h, d], 41, 1.0, DType::BFloat16);
    let v = rand(&[rows, tk, layers * h, d], 42, 1.0, DType::BFloat16);
    let k_app = rand(&[b, 1, h, d], 43, 1.0, DType::BFloat16);
    let v_app = rand(&[b, 1, h, d], 44, 1.0, DType::BFloat16);
    let realized = |v: Vec<i32>| {
        let t = Tensor::from_slice(&v);
        t.realize().unwrap();
        t
    };
    let lens = realized(vec![tk as i32, 1, 37, tk as i32 - 1, 100]);
    let row_map = realized(vec![0, 1, 1, 0, 1]);
    let cache = Cache { head_start: h, kv_heads: h, row_map: Some(&row_map), appended: Some((&k_app, &v_app)) };
    let opts = Attn { keys: KeyMask::Lens(&lens), cache: Some(cache), splits, ..Attn::default() };
    let o = tk::attention(&q, &k, &v, opts).unwrap();
    assert_kernel(&o, "flash_attention");
    if splits.is_some() {
        assert_kernel(&o, "combine_splits");
    }
    assert_close("cached attention", &o, &tk::attention::graph(&q, &k, &v, opts).unwrap(), 2e-2);
}

// ---- heads ---------------------------------------------------------------------

/// Rotary tables `[rows, t, 1, d / 2]`, realized.
fn rope_tables(rows: usize, t: usize, d: usize, dtype: DType) -> (Tensor, Tensor) {
    let (cos, sin) = Tensor::rope_table(10_000.0, t, d, DType::Float32).unwrap();
    let table = |x: Tensor| {
        let x = x.try_reshape([1, t, 1, d / 2]).unwrap();
        let x = if rows == 1 { x } else { x.try_expand([rows, t, 1, d / 2]).unwrap() };
        let x = x.cast(dtype.clone()).contiguous();
        x.realize().unwrap();
        x
    };
    (table(cos), table(sin))
}

#[test_case(2, 37, [4, 2, 64], true, true, false; "normed, rotated by position")]
#[test_case(2, 20, [8, 2, 128], true, true, true; "d 128, rotated by token")]
#[test_case(3, 9, [4, 4, 64], false, true, false; "rotated only")]
#[test_case(1, 50, [4, 1, 64], true, false, false; "normed only")]
#[test_case(2, 16, [2, 2, 32], false, false, false; "a split")]
fn heads_match_the_graph(b: usize, t: usize, [h, h_kv, d]: [usize; 3], normed: bool, rotated: bool, by_token: bool) {
    if !device() {
        return;
    }
    let qkv = rand(&[b, t, (h + 2 * h_kv) * d], 30, 2.0, DType::BFloat16);
    let q_w = rand(&[d], 31, 0.5, DType::BFloat16);
    let k_w = rand(&[d], 32, 0.5, DType::BFloat16);
    let tables = rotated.then(|| rope_tables(if by_token { b } else { 1 }, t, d, DType::BFloat16));
    let opts = Qkv {
        heads: h,
        kv_heads: h_kv,
        head_dim: d,
        q_norm: normed.then_some(&q_w),
        k_norm: normed.then_some(&k_w),
        eps: 1e-6,
        rope: tables.as_ref().map(|(c, s)| (c, s)),
    };
    let (q, k, v) = tk::heads(&qkv, opts).unwrap();
    assert_kernel(&q, "heads");
    let (wq, wk, wv) = tk::heads::graph(&qkv, opts).unwrap();
    assert_close("q", &q, &wq, 2e-2);
    assert_close("k", &k, &wk, 2e-2);
    assert_close("v", &v, &wv, 0.0);
}

#[test]
fn heads_under_a_batch_variable() {
    if !device() {
        return;
    }
    let (cap, live, t, [h, h_kv, d]) = (3, 2, 12, [4, 2, 64]);
    let qkv = rand(&[cap, t, (h + 2 * h_kv) * d], 33, 2.0, DType::BFloat16);
    let w = rand(&[d], 34, 0.5, DType::BFloat16);
    let (cos, sin) = rope_tables(1, t, d, DType::BFloat16);
    let var = Variable::new("b", 1, cap as i64).bind(live as i64).unwrap();
    let [qkv_b] = <[Tensor; 1]>::try_from(bound(&[&qkv], &var)).unwrap();
    let opts = Qkv {
        heads: h,
        kv_heads: h_kv,
        head_dim: d,
        q_norm: Some(&w),
        k_norm: Some(&w),
        eps: 1e-6,
        rope: Some((&cos, &sin)),
    };
    let (q, k, v) = tk::heads(&qkv_b, opts).unwrap();
    for out in [&q, &k, &v] {
        assert_eq!(out.shape().unwrap()[0], var.as_sint());
    }
    assert_kernel(&q, "heads");
    let [qkv_s] = <[Tensor; 1]>::try_from(first(&[&qkv], live)).unwrap();
    let (wq, wk, wv) = tk::heads::graph(&qkv_s, opts).unwrap();
    assert_close("batched q", &q, &wq, 2e-2);
    assert_close("batched k", &k, &wk, 2e-2);
    assert_close("batched v", &v, &wv, 0.0);
}

// ---- norms ---------------------------------------------------------------------

#[test_case(true, false, true, 1024; "layer 1024")]
#[test_case(true, false, false, 512; "layer 512 without a bias")]
#[test_case(true, true, true, 256; "add layer 256")]
#[test_case(false, false, false, 2048; "rms 2048")]
#[test_case(false, true, false, 1024; "add rms 1024")]
fn norms_match_the_graph(layer: bool, residual: bool, bias: bool, d: usize) {
    if !device() {
        return;
    }
    let x = rand(&[37, d], 30, 2.0, DType::BFloat16);
    let r = rand(&[37, d], 31, 1.0, DType::BFloat16);
    let w = rand(&[d], 32, 1.0, DType::BFloat16);
    let b = rand(&[d], 33, 0.5, DType::BFloat16);
    let b = bias.then_some(&b);
    let (sum, y) = match (layer, residual) {
        (true, false) => (None, tk::layer_norm(&x, &w, b, 1e-5).unwrap()),
        (true, true) => tk::add_layer_norm(&x, &r, &w, b, 1e-5).map(|(s, y)| (Some(s), y)).unwrap(),
        (false, false) => (None, tk::rms_norm(&x, &w, 1e-6).unwrap()),
        (false, true) => tk::add_rms_norm(&x, &r, &w, 1e-6).map(|(s, y)| (Some(s), y)).unwrap(),
    };
    let kind = if layer { crate::kernels::rows::Norm::Layer } else { crate::kernels::rows::Norm::Rms };
    let eps = if layer { 1e-5 } else { 1e-6 };
    let (want_sum, want) = tk::norm::graph(kind, &x, residual.then_some(&r), &w, b, eps).unwrap();
    assert_kernel(&y, if layer { "layer_norm" } else { "rms_norm" });
    assert_close("norm", &y, &want, 2e-2);
    if let (Some(sum), Some(want_sum)) = (sum, want_sum) {
        assert_close("sum", &sum, &want_sum, 0.0);
    }
}

#[test]
fn norm_under_a_batch_variable() {
    if !device() {
        return;
    }
    let (cap, live, d) = (4, 3, 512);
    let x = rand(&[cap, 37, d], 40, 2.0, DType::BFloat16);
    let r = rand(&[cap, 37, d], 41, 1.0, DType::BFloat16);
    let w = rand(&[d], 42, 1.0, DType::BFloat16);
    let b = rand(&[d], 43, 0.5, DType::BFloat16);
    let var = Variable::new("b", 1, cap as i64).bind(live as i64).unwrap();
    let [xb, rb] = <[Tensor; 2]>::try_from(bound(&[&x, &r], &var)).unwrap();
    let (sum, y) = tk::add_layer_norm(&xb, &rb, &w, Some(&b), 1e-5).unwrap();
    assert_eq!(y.shape().unwrap()[0], var.as_sint());
    assert_kernel(&y, "layer_norm");
    let [xs, rs] = <[Tensor; 2]>::try_from(first(&[&x, &r], live)).unwrap();
    let (want_sum, want) =
        tk::norm::graph(crate::kernels::rows::Norm::Layer, &xs, Some(&rs), &w, Some(&b), 1e-5).unwrap();
    assert_close("batched norm", &y, &want, 2e-2);
    assert_close("batched sum", &sum, &want_sum.unwrap(), 0.0);
}

// ---- f16 -------------------------------------------------------------------------

/// The same kernels over f16 operands.
#[test]
fn f16_ops_match_the_graph() {
    if !device() {
        return;
    }
    let x = rand(&[45, 64], 50, 1.0, DType::Float16);
    let w = rand(&[64, 64], 51, 0.5, DType::Float16);
    let y = tk::linear(&x, &w, Linear::default()).unwrap();
    assert_kernel(&y, "gemm");
    assert_close("f16 linear", &y, &tk::linear::graph(&x, &w, Linear::default()).unwrap(), 2e-2);

    let c = case(1, [70, 70], [2, 2], 64, true, None);
    let [q, k, v] = [20, 21, 22].map(|s| rand(&[1, c.t, c.h, c.d], s, 1.0, DType::Float16));
    let opts = Attn { causal: true, ..Attn::default() };
    let o = tk::attention(&q, &k, &v, opts).unwrap();
    assert_kernel(&o, "flash_attention");
    assert_close("f16 attention", &o, &tk::attention::graph(&q, &k, &v, opts).unwrap(), 2e-2);

    let w = rand(&[256], 52, 1.0, DType::Float16);
    let x = rand(&[37, 256], 53, 2.0, DType::Float16);
    let y = tk::rms_norm(&x, &w, 1e-6).unwrap();
    assert_kernel(&y, "rms_norm");
    let want = tk::norm::graph(crate::kernels::rows::Norm::Rms, &x, None, &w, None, 1e-6).unwrap().1;
    assert_close("f16 rms norm", &y, &want, 2e-2);
}

/// The residual is a realized buffer that a graph kernel (a norm the op
/// layer leaves on the graph) also reads; the GEMM consumes that kernel's
/// output.
#[test]
fn residual_shared_with_a_graph_norm() {
    if !device() {
        return;
    }
    let (m, d) = (1536, 384);
    let x = rand(&[1, m, d], 20, 1.0, DType::Float16);
    let g = rand(&[d], 21, 1.0, DType::Float16);
    let w = rand(&[d, d], 22, 0.2, DType::Float16);
    let ln = tk::layer_norm(&x, &g, None, 1e-5).unwrap();
    let opts = Linear { residual: Some(&x), ..Linear::default() };
    let y = tk::linear(&ln, &w, opts).unwrap();
    assert_kernel(&y, "gemm");
    assert_close("residual over graph norm", &y, &linear_reference(&ln, &w, opts), 3e-2);
}
