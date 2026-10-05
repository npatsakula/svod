---
name: tinygrad-debug
description: Run and inspect the Tinygrad reference checkout at submodules/tinygrad — kernel AST, generated C/LLVM source, linearized UOps, PatternMatcher rules, DEBUG levels, and the canonical-JSON parity scripts. Use when comparing a Svod tree or kernel with Tinygrad's for the same program, locating the Tinygrad counterpart of a Svod pass, or checking what a Tinygrad pattern does.
---

# Tinygrad investigation

Svod's pass labels (`SVOD_DUMP_STAGE`) follow `tinygrad/codegen/__init__.py`; the Svod-side pass map with Tinygrad
counterparts is `website/docs/architecture/codegen/overview.md`. Svod-side extraction: `/svod-debug`.

## Running

```bash
cd submodules/tinygrad && DEV=CPU uv run python - <<'EOF'
...
EOF
```
`uv run` uses the submodule's `pyproject.toml`/`uv.lock`. Device is the `DEV` context var (`DEV=CPU`, `DEV=CUDA`, ...);
`Device.DEFAULT = ...` raises — use `with Context(DEV="CPU"):` in code. The checked-out commit moves: compare
`git -C submodules/tinygrad rev-parse HEAD` with the pin in `git ls-tree HEAD submodules/tinygrad` before trusting line
numbers; the canonical parity tooling pins `8c8b43de…` (`scripts/check-canonical-parity.sh`, `TINYGRAD_REF` to override).

## Extracting AST, source and linear UOps (verified)

```python
from tinygrad import Tensor
from tinygrad.uop.ops import Ops
from tinygrad.uop.render import pyrender, render_ssa      # pyrender(ast) -> reconstructible Python; UOp.pyrender() too
from tinygrad.engine.realize import lower_and_compile

t = Tensor([1.0, 2.0, 3.0, 4.0]).sum()
print(t.uop.render())                                    # tensor-level graph (Tensor.uop is the attribute)
linear = Tensor.schedule_linear(t)                       # LINEAR of CALLs; bodies are SINK[KernelInfo] or STORE (copies)
for call in linear.src:
    if call.body.op is Ops.SINK: print(pyrender(call.body))   # kernel AST before codegen

linear = lower_and_compile(linear)                       # CALL(PROGRAM(SINK, LINEAR, SOURCE, BINARY))
for call in linear.src:
    prg = call.body
    if prg.op is Ops.PROGRAM:
        print(next(s.arg for s in prg.src if s.op is Ops.SOURCE))            # C / LLVM / PTX source
        print(render_ssa(list(next(s for s in prg.src if s.op is Ops.LINEAR).src)))   # linearized UOps
```
Per-kernel without the schedule: `from tinygrad.codegen import full_rewrite_to_sink, to_program`;
`to_program(ast, Device["CPU"].renderer)` returns the PROGRAM; `full_rewrite_to_sink(ast, renderer)` stops after the passes.
`UOp` fields: `.op`, `.dtype`, `.arg`, `.src`, `.toposort()`, `.render()`, `.key`, `.shape`.

| Env | Effect (`docs/env_vars.md`, prints in `codegen/__init__.py`) |
|-----|--------|
| `DEBUG=3` | applied opts per kernel |
| `DEBUG=4` | generated source (`do_render`/`do_compile`), ISA asm |
| `DEBUG=5` | `render_ssa` of the kernel AST at the start of `full_rewrite_to_sink` |
| `DEBUG=7` | disassembly of the compiled binary |
| `VIZ=1` | graph viewer, records every named `graph_rewrite` |
| `DEBUG_RANGEIFY=1`, `SPEC=0`, `NOOPT=1`, `BEAM=N`, `UPAT_COMPILE=0` | rangeify logging, skip type_verify, no opts, beam, uncompiled UPat matching |

## Where things live (current layout)

| File | Contents |
|------|----------|
| `tinygrad/codegen/__init__.py` | `full_rewrite_to_sink` (the ordered pass list), `expander`, `pm_reduce_local`/`pm_reduce_identity`, `do_devectorize`, `pm_add_loads`, `pm_add_local_buffers`, `pm_cast_float_alu`, `pm_implicit_barriers`, `do_linearize`/`do_render`/`do_compile`, `to_program` |
| `tinygrad/codegen/simplify.py` | `pm_flatten_range`, `pm_simplify_ranges`, `pm_split_ranges`, `pm_load_collapse`, `pm_reduce_unparented` |
| `tinygrad/codegen/late/{linearizer,coalesce,gater,regalloc}.py` | `linearize`, `pm_split_ends`, `memory_coalescing`, `indexing_simplify`, `pm_move_gates_from_index` |
| `tinygrad/codegen/opt/{heuristic,search,postrange}.py` | hand-coded opts, BEAM, `apply_opts` |
| `tinygrad/codegen/decomp/{op,dtype,transcendental}.py`, `codegen/gpudims.py` | late rewrites, dtype emulation, transcendentals, gpudims |
| `tinygrad/schedule/{rangeify,indexing,prepare,multi,memory}.py` | rangeify, range assignment, `pm_mops`, multi-device, memory planning |
| `tinygrad/uop/{ops,upat,symbolic,render,spec,weak,movement,divandmod}.py` | `UOp`, `UPat`, `PatternMatcher`, `graph_rewrite`, `RewriteContext.unified_rewrite`, `sym`/`symbolic`, renderers, spec |
| `tinygrad/renderer/{cstyle,llvmir,ptx}.py` | C-style, `LLVMRenderer`/`CPULLVMRenderer`/`AMDLLVMRenderer`, PTX |

`graph_rewrite(sink, pm, ctx=None, bottom_up=False, name=None, bpm=None, walk=False, enter_calls=False)` maps to Svod's
`graph_rewrite` / `graph_rewrite_bottom_up` / `graph_rewrite_with_bpm` / `graph_rewrite_walk` / `*_preserve_calls`.

## Inspecting a PatternMatcher

```python
from tinygrad.uop.symbolic import sym
for upat, fxn in sym.patterns: print(upat.op, upat.location, fxn.__name__)   # (filename, lineno)
for upat, match, early_reject in sym.pdict[Ops.ADD]: ...                     # rules keyed by root op
# UPat: .op (tuple of Ops), .name, .src, .arg, .dtype, .early_reject, .location
```

## Comparing with Svod

1. Same program on both sides; dump Svod at the matching stage (`SVOD_DUMP_STAGE=<label>`, labels follow
   `full_rewrite_to_sink` order) and Tinygrad with `DEBUG=5` / `pyrender`.
2. Look at RANGE axis types and order, INDEX arithmetic, REDUCE ranges/`num_axes`, where validity sits (inside INDEX as
   `WHERE(valid, idx, Invalid)` until gates move to LOAD/STORE), STACK lane layout after the expander.
3. For exact structural identity use the canonical JSON tooling: `scripts/check-canonical-parity.sh` (fixtures through
   both sides), `scripts/tinygrad-canonical.py <fixture> [--stage tensor|program] [--production-stage ...]`,
   `scripts/canonical-diff.py a.json b.json`; Svod emits the same schema with `SVOD_DUMP_CANONICAL_STAGE=<prefix>` or
   `SVOD_CAPTURE_CANONICAL_STAGE=<label> SVOD_CAPTURE_CANONICAL_PATH=<file>` (schema: `scripts/CANONICAL_SCHEMA_V7.md`).
