mod attention;
mod classifier;
mod config;
mod embedder;
mod head;
mod jit;
mod model;
mod parity;
mod token_classifier;

/// Largest `|a - b|`, NaN when any element is: `f32::max` drops a NaN, which
/// would read a broken output as a perfect match.
fn max_abs_delta(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "compared outputs differ in length");
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, |max, d| if d.is_nan() || d > max { d } else { max })
}

// Every JIT-compile test in this tree is gated behind `#[ignore = "heavy"]`:
// compiling even a 2-layer transformer graph through the CPU backend takes
// seconds per plan, and a head compiles one plan per batch shape it meets. The
// embedder/classifier/token_classifier tests share one head per fixture (behind
// a `Mutex`), so only the shapes they use compile.
//
// Fast default suite: `cargo test -p svod-model` (these are skipped).
// Full JIT coverage:
//   cargo test -p svod-model --release -- --ignored test::unit::modernbert::
//
// Parity goldens (`golden*.safetensors`) are generated locally and intentionally
// not committed — see `parity`'s module doc for the generate commands. Model
// weights / configs are fetched from HF Hub on first run.
