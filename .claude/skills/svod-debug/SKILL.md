---
name: svod-debug
description: Debug the Svod tensor → rangeify → kernel → codegen pipeline. Use when a test fails, a model gives wrong numbers, a kernel crashes or miscompiles, or you need the UOp tree at a specific pass, the generated LLVM IR / C / PTX, or the env vars and RUST_LOG targets that expose them. Covers SVOD_DUMP_STAGE, scripts/extract-ir.sh, SVOD_DUMP_*_IR, tracing in tests, ONNX node bisection.
---

# Svod pipeline debugging

Pass-by-pass reference with real trees: `website/docs/architecture/codegen/{overview,worked-example}.md`. Backend-specific
knobs: `website/docs/backends/{cuda,amd}/debugging.md`. Compare against Tinygrad with `/tinygrad-debug`.

## Quick start

Tests written with `svod_tensor::codegen_tests!` expand to `<name>::{clang,llvm,amd,cuda,metal}`; GPU variants self-skip
without a device. Filter on the full path to run one backend.

| Question | Command |
|----------|---------|
| Which post-opt pass changes/bloats the kernel? | `SVOD_PER_STAGE_UOPS=1 cargo test -p svod-tensor --lib test_x::llvm -- --nocapture` |
| Tree after one post-opt pass (prefix match on label) | `SVOD_DUMP_STAGE=14 cargo test ...` (`SVOD_DUMP_STAGE=` empty = every stage) |
| Trees for rangeify, kernel cut, pre-opt and post-opt in one file | `./scripts/extract-ir.sh test_x::llvm -p svod-tensor -o /tmp/ir.txt` (needs `rg`, `jaq`; builds `--release`) |
| LLVM IR as compiled (CPU) / after -O2 | `SVOD_DUMP_LLVM_IR=/tmp/ll` / `SVOD_DUMP_POST_O2_IR=/tmp/ll` → `<kernel>.ll` / `<kernel>.post.ll` |
| AMD / NVPTX LLVM IR | `SVOD_DUMP_AMD_IR=/tmp/amd` / `SVOD_DUMP_NVPTX_IR=/tmp/ptx` (one `.ll` per kernel) |
| Linearized instruction list | `SVOD_DUMP_LINEAR=/tmp/lin` → `tree_<id>.txt`, `linear_<id>.txt` from `do_linearize` |
| Is the optimizer the culprit? | `SVOD_NOOPT=1` (no opts; pre/post passes still run), `SVOD_TC=0`, `BEAM=4` |
| Is the backend the culprit? | run `::clang` vs `::llvm`; `SVOD_CPU_BACKEND=clang\|llvm`; `SVOD_LLVM_INPROCESS=0` (shell out instead of in-process LLVM); `SVOD_DEVICE=CPU\|CUDA:0\|AMD:0\|METAL:0` |
| Spec verification | `SVOD_SPEC=0` skips it; `SVOD_SPEC_DEBUG=1` prints the rejected uop and its tree |
| Stale cache suspicion | `SVOD_DISABLE_SCHEDULE_CACHE=1`, `SVOD_OBJECT_CACHE=0`, `IGNORE_BEAM_CACHE=1` |
| Which tensor op produced a kernel | `SVOD_ORIGIN=1` (origin capture; `SVOD_ORIGIN_DEPTH` for the profiler) |
| BEAM action survival / worker drops | `BEAM_DEBUG=1` |
| Canonical JSON of a stage (parity tooling) | `SVOD_DUMP_CANONICAL_STAGE=<prefix>` |

Isolate first: frontend (`tensor.uop().tree()` wrong), transformation (a pass turns a right tree into a wrong one — most
bugs, hardest), or codegen (final tree right, IR/source wrong). Then compare the failing stage with Tinygrad.

## Pipeline map

`tensor/src/realize.rs` chains: `rangeify_with_map` (`schedule/src/rangeify/transforms.rs`, once per SINK) →
`try_get_kernel_graph` (`rangeify/kernel.rs`, STAGE → STORE/END/AFTER, one `CALL(SINK[KERNEL])` per kernel) → per kernel
`apply_pre_optimization` → heuristics or BEAM → `apply_post_optimization_configured_with_capture` (`optimizer/mod.rs`) →
`program_from_sink` → `do_linearize` → `do_render` → `do_compile` (`codegen/src/program_pipeline.rs`).

Two numberings exist for post-opt stages: the `SVOD_DUMP_STAGE` label (also `[per-stage]` lines, follows Tinygrad's
`codegen/__init__.py`) and the `tracing` message. Both listed; `tracing` trees arrive as JSON fields.

| Phase (target) | Field | `SVOD_DUMP_STAGE` label | `tracing` message (debug level) |
|-------|-------|-------|-------|
| rangeify (`svod_schedule::rangeify::transforms`) | `uop.tree` | — | `add_tags complete`, `resolve_function complete`, `earliest rewrites complete`, `Stage 0: range assignment + apply rangeify complete`, `mega-pass complete` (count only), `Stage 7b: buffer limit enforcement complete` (conditional) |
| kernel cut (`svod_schedule::rangeify::kernel`) | — (trace: `tree` after `pm_add_buffers`) | — | `kernel split: pm_add_buffers complete`, `... pm_flatten_range pre-pass complete`, `... split_all_stores complete`, `... fix_assign complete` |
| pre-opt (`svod_schedule::optimizer`) | `ast.initial` (trace), `ast.pre` | — | `kernel initial`; `pre-opt: movement ops complete`, `load collapse`, `split ranges`, `symbolic + flatten`, `simplify ranges` |
| post-opt (`svod_schedule::optimizer`) | `ast.optimized` | `00-initial` | (trace) `kernel initial` |
| | | `08-post_opt_sym` | `Stage 8: after post-opt symbolic` |
| | | `09-pre_expand` | `Stage 9: after pre_expand` |
| | | `10-pm_reduce` | `after pm_reduce` |
| | | `11-local_buffers` | `after add local buffers` |
| | | `12-pm_add_gpudims` | `after pm_add_gpudims` |
| | | `13-pm_add_loads` | `after pm_add_loads` |
| | | `14-devectorize` | `after devectorize` |
| | | `15-early_symbolic` | `after early symbolic` |
| | | `16-memory_coalescing` | `after memory coalescing` |
| | | `17-bottom_up_ew_image` | `after bottom-up elementwise/image pass` |
| | | `16-extra_symbolic` | `after extra symbolic` |
| | | `17-pm_lower_index_dtype` | `after pm_lower_index_dtype` |
| | | `18-final_symbolic` | `after post-index symbolic` |
| | | `19-cast_float_alu` | `after cast float ALU operands` |
| | | `19b-early_decompositions` | `after early decompositions` |
| | | `19c-dtype_decompositions` | `after dtype decompositions` |
| | | `19d-late_decompositions` | `Stage 18: after late decompositions` |
| | | `19e-move_gates_from_index` | `Stage 19: after move gates from index` |
| | | `20-final_rewrite` | `Stage 20: after final rewrite` |
| linearize / render (`svod_codegen::llvm::text`, `svod_codegen::c`) | `generated_code`, `generated_c` (trace) | — | `linearized node` (trace, per op), `llvm codegen: final generated code`, `c codegen: final generated code` |

Labels `16` and `17` repeat, so `SVOD_DUMP_STAGE=16` prints two stages. Output format:
```
[per-stage] 13-pm_add_loads : node_count=30
[dump-stage] 20-final_rewrite :
[282] SINK[KERNEL] : Scalar(Void)
└── [281] END : Scalar(Void) shape=[]
    ├── [279] STORE : Scalar(Void) shape=[]
    │   ├── [278] INDEX : Scalar(Float32) shape=[]
    │   │   ├── [229] PARAM(slot=0) : Scalar(Float32) shape=[Const(2)]
    │   │   │   └── [227] CONST(Int(2)) : Scalar(Int32) shape=[]
    │   │   └── [239] RANGE(R1, Weak) : Scalar(Int32) shape=[]
    │   │       └── [227] → (see above)
    │   └── [273] Add : Scalar(Float32) shape=[]
    ...
    └── [239] → (see above)
[dump-stage] 20-final_rewrite : end
```
`UOp::tree()` prints `[id] OP(args) : dtype shape=[..]` with `├── `/`│   `/`└── ` and `[id] → (see above)` for a node already
printed (hash consing makes sharing visible); `tree_full()` re-expands shared nodes. Ids are allocation order and differ
between runs. Index dtype is `WeakInt` until `17-pm_lower_index_dtype` commits it to `Int32`/`Int64`.

## Tracing

`RUST_LOG` needs a subscriber. `codegen_tests!` installs one; a hand-written test calls
`svod_schedule::testing::setup_test_tracing()` (feature `testing`, already enabled in the `svod-tensor` and `svod-onnx`
dev-dependencies). It is a JSON-lines subscriber on the test writer, so add `-- --nocapture`.

```bash
RUST_LOG=svod_schedule::optimizer=debug cargo test -p svod-tensor --lib test_x::llvm -- --nocapture 2>&1 \
  | rg '^\{' | jaq -r 'select(.fields["ast.optimized"]) | "--- \(.fields.message)\n\(.fields["ast.optimized"])"'
```

`scripts/extract-ir.sh <test> [-p crate] [-o file] [-t target-regex]` runs this for
`rangeify::{transforms,indexing,kernel}` and `linearize` at `debug`, `optimizer` and `svod_codegen` at `trace`
(`kernel initial` and the generated code are trace events), and writes per-stage sections with `[nodes=N] [ms]`
headers, a `KERNEL n` section per kernel, and its generated source.

| Target | Information |
|--------|-------------|
| `svod_schedule::rangeify::transforms=debug` | rangeify stage trees (`uop.tree`) |
| `svod_schedule::rangeify::indexing=debug` | range assignment decisions (`merge_consumer_ranges`, realize axes) |
| `svod_schedule::rangeify::kernel=debug` / `=trace` | kernel split timings / tree after `pm_add_buffers`, `split_store` entries |
| `svod_schedule::optimizer=debug` / `=trace` | `ast.pre`, `ast.optimized` / plus `ast.initial`, dtype emulation decisions |
| `svod_codegen::llvm::text=trace`, `svod_codegen::c=trace` | per-op `linearized node`, final `generated_code` / `generated_c` |
| `svod_tensor::realize=debug` | prepare/realize timings |
| `svod_onnx::importer=debug` / `=trace` | per-node span (`onnx_node{idx, op}`) / realize every node and log `out_name`, `shape`, `first5` (breaks fusion; numerical bisection only) |
| `svod_device=debug`, `svod_runtime=debug` | driver, JIT, graph capture/replay, compile logs |
| `svod_ir::pattern::simplified=trace` | one `pattern matched` event per rewrite (matches only, not attempts) |

## In code

```rust
use svod_ir::prelude::*;
println!("{}", tensor.uop().tree());                   // frontend graph (Tensor::uop reads the registry)
let plan = tensor.prepare()?;                          // ExecutionPlan, nothing executed
for k in plan.kernels() {                              // &CachedKernel
    println!("{} on {}\n{}", k.entry_point, k.device, k.code);   // generated LLVM IR / C / PTX
}
for pk in plan.prepared_kernels() { println!("{}", pk.ast.tree()); }   // kernel AST per PreparedKernel (+ .kernel, .device)

// render a kernel SINK directly (CPU text renderer)
let rendered = svod_codegen::llvm::text::render(&sink, Some("k"))?;    // RenderedKernel { code, name, buffer_args, .. }
```
Other useful accessors: `uop.node_count()`, `uop.toposort()`, `uop.shape()? -> Option<&Shape>`, `uop.src_ops()`,
`uop.vmin()/vmax()`, `uop.ranges()`, `uop.get_consumer_map()`.

## Scenario notes

- Wrong numbers, CPU: diff `::clang` against `::llvm`; then `SVOD_NOOPT=1`; then bisect post-opt with `SVOD_DUMP_STAGE=` and look at `14-devectorize` (lane layout, `STACK`/`INDEX(STACK, c)`), `19e` (gates moved onto `LOAD`/`STORE`), `20`.
- Wrong numbers, ONNX model: `RUST_LOG=svod_onnx::importer=trace`, compare `first5` per node with onnxruntime, fix the first diverging op.
- Crash / SIGSEGV: `SVOD_DUMP_LLVM_IR` + inspect the index arithmetic at `17-pm_lower_index_dtype` (i32 overflow) and `PARAM` sizes vs buffer sizes; try `SVOD_LLVM_INPROCESS=0`.
- Spec failure at a boundary: `SVOD_SPEC_DEBUG=1`, read `schedule/src/spec.rs` for the rule that rejected.
- Slow compile or BEAM stall: `SVOD_PER_STAGE_UOPS=1` for node blow-up per pass; `BEAM_DEBUG=1`, `BEAM_TIMEOUT_SEC`, `PARALLEL`.

## Key files

| File | Purpose |
|------|---------|
| `tensor/src/realize.rs` | pipeline driver, schedule cache, BEAM dispatch |
| `schedule/src/rangeify/{transforms,indexing,kernel,patterns}.rs` | rangeify, range assignment, kernel cut |
| `schedule/src/optimizer/mod.rs` | pre-opt, post-opt stages, `SVOD_DUMP_STAGE`/`SVOD_PER_STAGE_UOPS` |
| `schedule/src/optimizer/{heuristics,beam,config}.rs` | opt selection and env knobs |
| `schedule/src/{expand,devectorize,gpudims}.rs`, `schedule/src/late/` | expander, devectorizer, gates, coalescing |
| `schedule/src/spec.rs` | type verification (`SVOD_SPEC`) |
| `codegen/src/program_pipeline.rs` | `program_from_sink`, `do_linearize` (`SVOD_DUMP_LINEAR`), `do_render`, `do_compile` |
| `codegen/src/llvm/text/mod.rs`, `codegen/src/c/mod.rs` | renderers (`render()`), `generated_code` trace |
| `runtime/src/llvm.rs`, `runtime/src/{amd,cuda}/compile.rs` | `SVOD_DUMP_LLVM_IR`, `SVOD_DUMP_POST_O2_IR`, `SVOD_DUMP_AMD_IR`, `SVOD_DUMP_NVPTX_IR` |
| `ir/src/uop/tree.rs`, `ir/src/uop/canonical.rs` | tree rendering, canonical JSON dump |
| `schedule/src/testing.rs`, `tensor/src/config.rs` | `setup_test_tracing`, `codegen_tests!` |
| `scripts/extract-ir.sh` | JSON trace → readable per-stage file |
