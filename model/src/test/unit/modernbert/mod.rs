mod attention;
mod classifier;
mod config;
mod embedder;
mod head;
mod jit;
mod model;
mod parity;

use svod_arch::pipelines::text::Encoding;

/// Largest `|a - b|`, NaN when any element is: `f32::max` drops a NaN, which
/// would read a broken output as a perfect match.
fn max_abs_delta(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "compared outputs differ in length");
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, nan_max)
}

/// `f32::max` that keeps a NaN where `f32::max` drops it.
fn nan_max(max: f32, d: f32) -> f32 {
    if d.is_nan() || d > max { d } else { max }
}

/// A chunk of `real_ids` (mask 1) followed by `n_pad` pad positions (id 0,
/// mask 0).
fn encoding(real_ids: &[u32], n_pad: usize) -> Encoding {
    let mut ids = real_ids.to_vec();
    let mut mask = vec![1u32; real_ids.len()];
    ids.extend(std::iter::repeat_n(0u32, n_pad));
    mask.extend(std::iter::repeat_n(0u32, n_pad));
    let l = ids.len();
    Encoding {
        input_ids: ids,
        attention_mask: mask,
        token_type_ids: vec![0; l],
        offsets: (0..l).map(|i| (i, i + 1)).collect(),
        special_tokens_mask: vec![0; l],
    }
}

// Every JIT-compile test in this tree is gated behind `#[ignore = "heavy"]`:
// compiling even a 2-layer transformer graph through the CPU backend takes
// seconds per plan, and a head compiles one plan per batch shape it meets. The
// classifier tests share one head (behind a `Mutex`), so each shape they use
// compiles once.
//
// Fast default suite: `cargo test -p svod-model` (these are skipped).
// Full JIT coverage:
//   cargo test -p svod-model --release -- --ignored test::unit::modernbert::
//
// Parity goldens (`golden*.safetensors`) are generated locally and intentionally
// not committed — see `parity`'s module doc for the generate commands. Model
// weights / configs are fetched from HF Hub on first run, at a pinned revision.
