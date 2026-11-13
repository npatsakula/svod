//! CSP blocks keep their parts in one buffer, each part stored into its
//! channels where it is made; the forward equals the chunk-and-cat one.

use svod_dtype::DType;
use svod_ir::SInt;
use svod_tensor::Tensor;
use test_case::test_case;

use crate::yolo::{C2f, C3k, C3k2, C3k2Inner, YoloBottleneck, YoloConv};

/// `cv1 → chunk(2) → chain → cat → cv2` with the chain's parts copied
/// together, the forward before the parts shared a buffer.
fn chained<B>(cv1: &YoloConv, cv2: &YoloConv, chain: &[B], step: impl Fn(&B, &Tensor) -> Tensor, x: &Tensor) -> Tensor {
    let halves = cv1.forward(x).unwrap().chunk(2, -1).unwrap();
    let mut parts = halves.clone();
    for blk in chain {
        parts.push(step(blk, parts.last().unwrap()));
    }
    cv2.forward(&Tensor::cat(&parts.iter().collect::<Vec<_>>(), -1).unwrap()).unwrap()
}

fn bottleneck(b: &YoloBottleneck, x: &Tensor) -> Tensor {
    b.forward(x).unwrap()
}

fn c3k(c: &C3k, x: &Tensor) -> Tensor {
    let left = c.m.iter().fold(c.cv1.forward(x).unwrap(), |acc, b| bottleneck(b, &acc));
    let cat = Tensor::cat(&[&left, &c.cv2.forward(x).unwrap()], -1).unwrap();
    c.cv3.forward(&cat).unwrap()
}

fn inner(blk: &C3k2Inner, x: &Tensor) -> Tensor {
    match blk {
        C3k2Inner::Bottleneck(b) => bottleneck(b, x),
        C3k2Inner::C3k(c) => c3k(c, x),
        C3k2Inner::Attn(..) => blk.forward(x).unwrap(),
    }
}

fn input(c: usize, dtype: DType) -> Tensor {
    Tensor::randn(&[2, 6, 5, c]).unwrap().cast(dtype)
}

/// Exact at f32, where every conv takes the graph; at f16 a kernel device runs
/// the windowed kernels against the cat's contiguous ones.
fn assert_matches(got: &Tensor, want: &Tensor) {
    assert_eq!(got.dims().unwrap(), want.dims().unwrap());
    let (got, want) = (got.cast(DType::Float32), want.cast(DType::Float32));
    let (got, want) = (got.to_vec::<f32>().unwrap(), want.to_vec::<f32>().unwrap());
    let exact = got == want;
    let close = got.iter().zip(&want).all(|(g, w)| (g - w).abs() <= 1e-2 * (1.0 + w.abs()));
    assert!(exact || close, "the windowed forward drifted from the cat's");
}

#[test_case(16, 2, true, DType::Float32; "aligned parts, f32")]
#[test_case(16, 2, true, DType::Float16; "aligned parts, f16")]
#[test_case(4, 2, true, DType::Float16; "parts narrower than a run")]
#[test_case(16, 1, false, DType::Float16; "no shortcut")]
fn a_c2f_stores_its_parts_in_place(c: usize, n: usize, shortcut: bool, dtype: DType) {
    let block = C2f::empty(3 * c, 2 * c, n, shortcut, 0.5);
    let x = input(3 * c, dtype);
    let want = chained(&block.cv1, &block.cv2, &block.m, bottleneck, &x);
    assert_matches(&block.forward(&x).unwrap(), &want);
}

#[test_case(false, false, DType::Float32; "bottlenecks, f32")]
#[test_case(true, false, DType::Float32; "c3k, f32")]
#[test_case(true, false, DType::Float16; "c3k, f16")]
#[test_case(false, true, DType::Float32; "attention, f32")]
#[test_case(false, true, DType::Float16; "attention, f16")]
fn a_c3k2_stores_its_parts_in_place(c3k: bool, attn: bool, dtype: DType) {
    let block = C3k2::empty(64, 64, 2, true, 0.5, c3k, attn);
    let x = input(64, dtype);
    let want = chained(&block.cv1, &block.cv2, &block.m, inner, &x);
    assert_matches(&block.forward(&x).unwrap(), &want);
}

/// A C3k stores into another buffer's channels what its forward returns.
#[test_case(DType::Float32; "f32")]
#[test_case(DType::Float16; "f16")]
fn a_c3k_stores_into_a_window(dtype: DType) {
    let block = C3k::empty(32, 32, 2, true, 0.5, 3);
    let x = input(32, dtype.clone());
    let outer = svod_tk::ops::buffer(&[2, 6, 5, 96].map(SInt::Const), dtype).unwrap();
    let got = block.forward_into(&x, (&outer, 32)).unwrap();
    assert_matches(&got.try_shrink([None, None, None, Some((32, 64))]).unwrap(), &c3k(&block, &x));
}
