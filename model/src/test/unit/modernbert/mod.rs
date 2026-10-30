mod attention;
mod classifier;
mod config;
mod embedder;
mod head;
mod jit;
mod model;
mod parity;

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
