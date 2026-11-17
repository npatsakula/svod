//! Operands in channel windows of wider buffers: a kernel over windows writes
//! into its output's window exactly what it writes contiguous, and nothing
//! outside it.

use test_case::test_case;

use super::lcg;
use crate::build::F16;
use crate::interp::run;
use crate::kernels::conv::{ConvGeom, ConvSpec, conv};
use crate::kernels::gemm::{Epilogue, GemmSpec, SplitMergeSpec, gemm, split_merge};
use crate::kernels::{Act, Batch, Window, Windows};
use crate::ops::config::gemm_cfg;
use crate::sweep::assumed;

/// What a buffer holds outside every window a test writes.
const UNTOUCHED: f64 = 7.0;

const EPI: Epilogue = Epilogue { bias: true, act: Act::Silu, residual: true, ..Epilogue::DEFAULT };

fn values(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..n).map(|_| lcg(&mut s)).collect()
}

/// `data`'s rows of `width` in window `w` (natural where `None`) of a buffer
/// `fill` holds elsewhere.
fn embedded(data: &[f64], width: usize, w: Option<Window>) -> Vec<f64> {
    let Some(w) = w else { return data.to_vec() };
    let mut buf = vec![UNTOUCHED; w.elems(data.len() / width)];
    for (r, row) in data.chunks(width).enumerate() {
        buf[r * w.ld + w.offset..][..width].copy_from_slice(row);
    }
    buf
}

/// The rows of `width` window `w` of `buf` holds, and whether every element
/// outside it is [`UNTOUCHED`].
fn cut(buf: &[f64], width: usize, w: Option<Window>) -> (Vec<f64>, bool) {
    let Some(w) = w else { return (buf.to_vec(), true) };
    let rows = buf.len() / w.ld;
    let inside = |i: usize| (w.offset..w.offset + width).contains(&(i % w.ld));
    let kept = buf.iter().enumerate().all(|(i, v)| inside(i) || *v == UNTOUCHED);
    ((0..rows).flat_map(|r| buf[r * w.ld + w.offset..][..width].to_vec()).collect(), kept)
}

fn window(offset: usize, ld: usize) -> Option<Window> {
    Some(Window { offset, ld })
}

/// A GEMM over windowed operands, a static batch walking grid z included,
/// stores bit for bit the contiguous GEMM's output in its window alone.
#[test_case(Windows { a: window(16, 96), ..Windows::NATURAL }, 1; "a chunk's rows")]
#[test_case(Windows { residual: window(8, 64), ..Windows::NATURAL }, 1; "a windowed residual")]
#[test_case(Windows { c: window(32, 112), ..Windows::NATURAL }, 1; "into a cat's window")]
#[test_case(Windows { a: window(64, 128), residual: window(0, 96), c: window(48, 96) }, 2; "all three, two batches")]
fn a_windowed_gemm_stores_the_contiguous_one(windows: Windows, batches: usize) {
    let (m, n, k) = (40, 48, 64);
    let cfg = gemm_cfg([32, 32, 32], 2, [2, 2], true);
    let spec = |windows| GemmSpec { m, n, k, batch: Batch::Static(batches), epilogue: EPI, windows };
    let (a, b, bias, r) = (values(batches * m * k, 1), values(n * k, 2), values(n, 3), values(batches * m * n, 4));
    let plain = gemm::<F16>(&assumed(spec(Windows::NATURAL), cfg));
    let want = run(&plain, vec![a.clone(), b.clone(), bias.clone(), r.clone(), vec![0.0; batches * m * n]], &[]);
    let windowed = gemm::<F16>(&assumed(spec(windows), cfg));
    let c = embedded(&vec![UNTOUCHED; batches * m * n], n, windows.c);
    let params = vec![embedded(&a, k, windows.a), b, bias, embedded(&r, n, windows.residual), c];
    let got = run(&windowed, params, &[]).unwrap().pop().unwrap();
    let (got, kept) = cut(&got, n, windows.c);
    assert_eq!(got, want.unwrap().pop().unwrap());
    assert!(kept, "an element outside the output's window was written");
}

/// A split's merge reads its residual from and stores into windows.
#[test]
fn a_windowed_merge_stores_the_contiguous_one() {
    let (m, n, split) = (40, 48, 3);
    let windows = Windows { residual: window(16, 64), c: window(32, 80), a: None };
    let spec = |windows| SplitMergeSpec { m, n, split, epilogue: EPI, tile: [32, 32], parts: None, windows };
    let (partial, bias, r) = (values(split * m * n, 5), values(n, 6), values(m * n, 7));
    let params = vec![partial.clone(), bias.clone(), r.clone(), vec![0.0; m * n]];
    let want = run(&split_merge::<F16>(&spec(Windows::NATURAL)), params, &[]);
    let c = embedded(&vec![UNTOUCHED; m * n], n, windows.c);
    let params = vec![partial, bias, embedded(&r, n, windows.residual), c];
    let got = run(&split_merge::<F16>(&spec(windows)), params, &[]).unwrap().pop().unwrap();
    let (got, kept) = cut(&got, n, windows.c);
    assert_eq!(got, want.unwrap().pop().unwrap());
    assert!(kept, "an element outside the output's window was written");
}

/// A convolution gathers its pixels' channels from a window, a bound batch
/// walking grid z included, and stores into one.
#[test_case(Batch::Static(2), 1; "two images folded into the rows")]
#[test_case(Batch::var("b", 1, 2), 1; "a bound batch")]
#[test_case(Batch::Static(1), 2; "stride two")]
fn a_windowed_convolution_stores_the_contiguous_one(batch: Batch, stride: usize) {
    let g = ConvGeom {
        h: 7,
        w: 6,
        cin: 32,
        cout: 40,
        kernel: [3, 3],
        stride: [stride, stride],
        pad: [1, 1],
        dilation: [1, 1],
    };
    let windows = Windows { a: window(16, 64), residual: window(8, 48), c: window(40, 96) };
    let [ho, wo] = g.out_hw();
    let images = batch.capacity();
    let cfg = gemm_cfg([32, 32, 32], 2, [2, 2], true);
    let spec = |windows| ConvSpec { batch: batch.clone(), geom: g, epilogue: EPI, split: 1, windows };
    let x = values(images * g.h * g.w * g.cin, 8);
    let (w, bias, r) = (values(g.cout * 9 * g.cin, 9), values(g.cout, 10), values(images * ho * wo * g.cout, 11));
    let vars: Vec<(&str, i64)> = match &batch {
        Batch::Var(v) => vec![(v.name.as_str(), v.max)],
        Batch::Static(_) => vec![],
    };
    let plain = conv::<F16>(&assumed(spec(Windows::NATURAL), cfg));
    let params = vec![x.clone(), w.clone(), bias.clone(), r.clone(), vec![0.0; images * ho * wo * g.cout]];
    let want = run(&plain, params, &vars).unwrap().pop().unwrap();
    let c = embedded(&vec![UNTOUCHED; images * ho * wo * g.cout], g.cout, windows.c);
    let params = vec![embedded(&x, g.cin, windows.a), w, bias, embedded(&r, g.cout, windows.residual), c];
    let got = run(&conv::<F16>(&assumed(spec(windows), cfg)), params, &vars).unwrap().pop().unwrap();
    let (got, kept) = cut(&got, g.cout, windows.c);
    assert_eq!(got, want);
    assert!(kept, "an element outside the output's window was written");
}
