//! Tests for the Qwen3 SwiGLU feed-forward ([`crate::qwen3::Qwen3MLP`]): the
//! load-time gate/up row interleave the fused GEMM epilogue reads, its inverse
//! on the published state dict, and the forward's agreement with the eager
//! `silu(gate)·up` whichever layout the weight is in.

use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::{Module, StateDict};

use crate::qwen3::Qwen3MLP;

const H: usize = 16;
const I: usize = 32;

/// `[rows, cols]` where element `(r, c)` is `base + r + c/100` — every row
/// distinguishable, so a permutation is visible.
fn marked(rows: usize, cols: usize, base: f32) -> Tensor {
    let v: Vec<f32> = (0..rows * cols).map(|i| base + (i / cols) as f32 + (i % cols) as f32 / 100.0).collect();
    Tensor::from_slice(v).try_reshape([rows as isize, cols as isize]).expect("reshape")
}

fn state(gate: &Tensor, up: &Tensor, down: &Tensor) -> StateDict {
    let mut sd = StateDict::new();
    sd.insert("gate_proj.weight".into(), gate.clone());
    sd.insert("up_proj.weight".into(), up.clone());
    sd.insert("down_proj.weight".into(), down.clone());
    sd
}

fn loaded(gate: &Tensor, up: &Tensor, down: &Tensor) -> Qwen3MLP {
    let mut mlp = Qwen3MLP::empty(H, I, DType::Float32);
    mlp.load_state_dict(&state(gate, up, down), "").expect("load");
    mlp
}

fn rows(t: &Tensor) -> Vec<Vec<f32>> {
    t.realize().expect("realize");
    let cols = t.dim_const(1).expect("dims");
    t.as_vec::<f32>().expect("read").chunks(cols).map(<[f32]>::to_vec).collect()
}

/// The fused weight arrives as alternating `pair`-row gate and up blocks, which
/// is what puts a gate column beside its up column inside one wave's
/// accumulator. The block width is the kernel's, not a number spelled here.
#[test]
fn load_state_dict_interleaves_the_gate_up_rows() {
    let pair = svod_tk::swiglu_pair_width().expect("the GEMM tiles agree on a pair width");
    assert_eq!(I % pair, 0, "the tiny intermediate size must divide by the pair width");
    let (gate, up, down) = (marked(I, H, 100.0), marked(I, H, 200.0), marked(H, I, 300.0));
    let mlp = loaded(&gate, &up, &down);

    assert_eq!(mlp.gate_up_weight.dims().expect("dims"), [2 * I, H]);
    let (got, g, u) = (rows(&mlp.gate_up_weight), rows(&gate), rows(&up));
    for (r, row) in got.iter().enumerate() {
        let (block, within) = (r / pair, r % pair);
        let want = if block % 2 == 0 { &g[(block / 2) * pair + within] } else { &u[(block / 2) * pair + within] };
        assert_eq!(row, want, "row {r} of the fused weight");
    }
}

/// The published state dict is the un-interleaved checkpoint layout, so a
/// load/write round trip is the identity — the row order is an internal detail.
#[test]
fn write_state_un_interleaves_the_rows() {
    let (gate, up, down) = (marked(I, H, 100.0), marked(I, H, 200.0), marked(H, I, 300.0));
    let sd = loaded(&gate, &up, &down).state_dict("");
    assert_eq!(rows(&sd["gate_proj.weight"]), rows(&gate));
    assert_eq!(rows(&sd["up_proj.weight"]), rows(&up));
    assert_eq!(rows(&sd["down_proj.weight"]), rows(&down));
}

/// A never-loaded module keeps the plainly stacked rows, and still publishes
/// the same two keys.
#[test]
fn an_unloaded_module_publishes_the_stacked_halves() {
    let mlp = Qwen3MLP::empty(H, I, DType::Float32);
    let sd = mlp.state_dict("");
    let all = rows(&mlp.gate_up_weight);
    assert_eq!(rows(&sd["gate_proj.weight"]), all[..I].to_vec());
    assert_eq!(rows(&sd["up_proj.weight"]), all[I..].to_vec());
}

/// Whatever the row order, the forward is `down(silu(x·gateᵀ)·(x·upᵀ))`: the
/// interleave is undone by the epilogue's column mapping on a device that runs
/// it, and by the split here on one that does not.
#[test]
fn forward_matches_the_eager_swiglu() {
    let (gate, up, down) = (marked(I, H, 0.01), marked(I, H, 0.02), marked(H, I, 0.03));
    let (gate, up, down) = (gate.try_mul(Tensor::from_slice([0.05f32])).expect("scale"), up, down);
    let mlp = loaded(&gate, &up, &down);

    let x = marked(6, H, 0.1).try_mul(Tensor::from_slice([0.03f32])).expect("scale");
    let got = rows(
        &mlp.forward(&x.try_reshape([1isize, 6, H as isize]).expect("reshape"))
            .expect("forward")
            .try_reshape([6isize, H as isize])
            .expect("flatten"),
    );

    let lin = |a: &Tensor, w: &Tensor| a.linear().weight(w).call().expect("linear");
    let act = lin(&x, &gate).silu().expect("silu").try_mul(lin(&x, &up)).expect("gate·up");
    let want = rows(&lin(&act, &down));

    // Relative to the output's magnitude: a GPU GEMM sums in a different order.
    let want = want.concat();
    let scale = want.iter().fold(0f32, |a, b| a.max(b.abs()));
    let worst = got.concat().iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max) / scale;
    assert!(worst < 1e-5, "the fused layout drifted from the eager SwiGLU by {worst} of its magnitude");
}
