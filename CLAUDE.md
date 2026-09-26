# Svod

Use Opus 4.6 powered agents.

## Skills

Use `/svod-debug` for pipeline debugging (IR extraction, LLVM IR, tracing).
Use `/tinygrad` for comparing with Tinygrad's implementation.

## Task planning

- Feel free to use multiple agents to explore different strategies and solutions.
- Avoid assumptions that are not supported by evidence; gather information using agents.
  Don't use cheap model as an agent, it's too stupid for our codebase.
- Keep the main context clear and perform hypothesis testing using agents.
- Avoid fast/hacky solutions; make them robust and scalable.
- Make sure that you understand all type signatures and their implications (some crates may
  have updated their API since the last time you used them).
- Focus on code size and performance: minimize code; use Rust's capabilities
  and ecosystem to optimize performance.
- If you or agent reached limit of 25k tokens during file read, consider chunked file
  reading (e.g. 1000 line each chunk); don't play games with symbols searching, large
  files are source of grand knowledge.

## Task execution

- If you revert the initial plan, you should stop what you're doing and go back to the planning phase.
- If you catch yourself in a situation where you're going to reset changes using git, check that the file
  was committed before; otherwise you may lose your work and will be unable to re-generate it.
- Keep documentation minimal and focused on the most important aspects of the code. Your code
  should be expressive and easy to understand without extra documentation.

### Error handling

- For library crates, use the `snafu` crate for error handling:
  - Use `.context()` method and auto-generated structures (`*Snafu`) for error context capturing.
  - Use `context` field in `Error` enum variants for external error context capturing.
- You can use `.expect()` instead of error handling if:
  - The error is the result of a crate implementation error and the user can't do anything about it.
  - The error is not expected to occur in normal operation.
  - The error is unrecoverable and the user can't do anything about it.

### Dependencies

- Add dependencies to `Cargo.toml` file using `cargo add` command; if there is a new minor/major version
  since the last time you used them, check the difference in interface using an agent.

### Testing

- Each crate has a `test/` module with unit and property-based tests; we don't write tests in-place.
- We use the `proptest` (dep) crate for property-based testing; if we use the `proptest-derive` crate, we
  put the derive under a feature `proptest`.
- We use the `test_case` (dep) crate for unit testing if a test requires different inputs in order to reduce
  code duplication and simplify test understanding.
- We add test infrastructure to the `.tokeignore` file in order to understand the codebase size better.

## Settled decisions

- **BEAM times candidates in batches on a lifted clock** (`TimingBatch` in
  `schedule/src/optimizer/beam.rs`; `warm_clock` + `round_robin_min` in the benchmark closure of
  `tensor/src/realize.rs`). Timing each candidate the moment its compile lands ranks the GPU's
  idle clock, not the kernels: a median 1.5x and up to 6x error on gfx1201, which once made
  BEAM=8 pick a 6x slower plan. The per-batch clock lift is deliberate and not a removable
  extra; the search is not slower with it.
- **The tk convolution gate is a measured rule the kernel owns, not a channel bound the
  model carries** (`svod_tk::conv2d_nhwc_worth_asking`, asked by `YoloConv::tk_eligible`): the
  lattice's edges (`cout % 32`, `cin % 16`) plus `CONV_K_FLOOR = 576` on gfx1201 — K = 288 loses
  ~29 µs a conv to the graph's kernel, K = 864 wins ~8 at BEAM=4 and ties at BEAM=8, K ≥ 3456
  wins outright. On CUDA the kernel also declines a shape only a 32-wide tile serves (the
  table's fine tile or the lattice's edge) unless its grid starves the device: on sm86 in the x
  frame `384→96 k3 @80²` ran 320 µs against the graph's 229 and `96→96 k3 @80²` 64.5 against
  57.3, while `768→96 k3 @20²` ran 63 against 95. The global lattice bound (`a7c2a1fc`) loses at
  m/l and a per-site allowlist cannot tell the shapes apart, so neither replaces this. The
  96-channel bodies are a lever at BEAM=4 and a wash at BEAM=8; do not re-open them per width.
  Measured x/b1: gfx1201 4.767 → 4.515 ms (−5.3%, BEAM=4 under BEAM=8's frame), RTX 3060 −0.6%.
  Re-checked in the frame under clang 22: a floor of 1152 loses at s/m/l/x (+1.7…+5.6%). The conv
  bench's isolated graph arm flatters K = 864 (18 µs against 21.8 in the frame); it cannot move it.
- **Every loop the AMD renderer emits carries `amdgpu.loop.unroll.threshold = 300`, except a loop
  whose counter indexes a register array** (`LOOP_HINT` and `register_indexing_ranges` in
  `codegen/src/llvm/amd/mod.rs`). Without the cap, AMDGPU's +200 per branch on the loop's own
  index fully unrolls a reduce loop over gated loads (a cat, padding): YOLO26-n's `neck.13.cv1`
  became 72 WMMAs and 843 spilled VGPRs, 99 µs instead of 27, and clang 20.1.2 compiled that
  spill into NaN — at BEAM=0 and at BEAM=4 alike, which picks the same kernel. With it, n is right
  and BEAM=4 is −12.4% at n, −0.7…−1.8% at s/m/l and a wash at x; the plans BEAM finds under it
  need it (replayed without it, x runs +45% with NaN boxes). The exemption is not optional:
  capped too, every tk convolution kept 132 B of its register tiles in scratch.
- **The tk tile search times a tile only once its output agrees with its rivals'**
  (`TileBudget::search`, `tiling::agreement`): ranked by time alone it once handed m's bodies a
  tile that computed garbage fast (16 px of box drift). The check reads every candidate back on a
  tune-store miss only; do not trade it for a faster first run.
- **ModernBERT realizes its RoPE tables, its rotated Q/K and every GEMM operand**
  (`.contiguous()` in `model/src/modernbert/{encoder,attention,linear}.rs`). Rangeify inlines a
  staged value into its consumer whenever it reads ≤ 3 buffers — tinygrad's rule, with no count of
  how often the consumer re-evaluates it — so left lazy, RoPE's sin/cos/pow ran once per key
  inside every QKᵀ and GELU's erf once per output feature inside `Wo`: 1×512 took 64 ms instead of
  15 at BEAM=4 on the RTX 3060. `state::cast_all` realizes its casts for the same reason (every GEMM
  otherwise re-reads the f32 checkpoint: 11.1 → 9.8 ms).
- **ModernBERT pads to the flash-attention tile inside the attention, not the sequence**
  (`ModernBertAttention::flash`): the kernel copies its operands anyway, so only those copies grow.
  Padding the sequence makes every GEMM run the padded rows too — 8×300 at 53.7 ms against 43.5,
  8×400 at 72.0 against 56.0.

## Task evaluation

- `cargo fmt`, `cargo clippy` and `cargo test` should pass before I can perform review.
- Don't hack/simplify tests, ensure they are comprehensive and cover all edge cases.
