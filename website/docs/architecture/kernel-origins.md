---
sidebar_label: Kernel Origins
---

# Kernel Origins

A profile that says `r_128_3_32_4_2_2_2_4_4_192_2` took 100 ms tells you the shape of the
kernel, not whose kernel it is. Origins answer the second question: every dispatched kernel
knows the module path, call site or ONNX node it was built for, and the profiler can roll
time up along that path — per layer, per block, per stage.

This page is the user's guide: how to turn it on, how to instrument a model, and how to read
the output. The mechanism (a hash-consed field on every node, stripped again at the kernel
cut) is summarised at the end and documented in the [IR design](./ir-design.md) and
[op bestiary](./op-bestiary.md) pages.

---

## Turning it on

Capture is off by default and costs nothing while off: nodes carry no origin, hashes are
byte-identical to a build without the feature. Two switches:

| Switch | Effect |
|--------|--------|
| `SVOD_ORIGIN=1` | capture on for every thread of the process (unset, empty or `0` = off) |
| `SVOD_ORIGIN_DEPTH=<n>` | rollups keep the first `n` path segments (unset or `0` = full path, printed as `depth leaf`) |

```bash
SVOD_DEVICE=AMD:0 SVOD_ORIGIN=1 cargo run --release -p svod-model --example gigaam_infer -- \
    audio.wav --profile --origin-depth 3 --profile-json profile.json
```

`--origin-depth` beats `SVOD_ORIGIN_DEPTH`; the Whisper example has only `--profile`.

In tests, flip capture for the current thread only, so parallel tests keep their graph
identity:

```rust
let _capture = svod_ir::origin::capture_for_thread(true); // restored on drop
```

---

## Where origins come from

An origin is a path of frames, root first. Each frame is one of:

| Frame | Rendered as | Opened by |
|-------|-------------|-----------|
| `Module` | `encoder.layers.3.ffn1` | model code, one segment per module (`OriginScope::module`) |
| `Label` | `vad`, `GigaAmCtcJit`, `initializer` | pipeline stages, every `jit_wrapper!` build, the ONNX importer |
| `Onnx` | `/encoder/Conv` or `#12:MatMul` | the ONNX importer, one per node: the node name, or `#index:op_type` when it has none |
| `Call` | `@ linear model/src/gigaam/encoder.rs:43` | every public `Tensor` op, through `origin_call!` at its entry |

Named frames join with `.`; a `Call` frame follows after a space. It is the flat file:line
layer under the module path: an op opens it only when the current frame is not already a
call (outermost wins), so an op implemented on top of other ops (`linear` over `matmul`)
records the user's line once, never svod's own source. The module layers above it are what
model code adds.

### Instrumenting a Rust model

Open a scope in `forward` for each module the way you would spell its state-dict prefix. The
model crate has helpers that do exactly that:

```rust
use crate::state::{scoped, scoped_index};

fn forward(&self, x: &Tensor) -> Result<Tensor> {
    let x = scoped("subsampling", || self.subsampling.forward(x))?;
    let mut x = x;
    for (i, layer) in self.layers.iter().enumerate() {
        x = scoped_index("layers", i, || layer.forward(&x))?;   // layers.0, layers.1, …
    }
    scoped("final_norm", || self.final_norm.forward(&x))
}
```

`scoped` opens `OriginScope::module(name)` around the closure; `scoped_index` formats the
`name.i` segment only while capture is on; `scope_index(name, i)` is the same as a guard for
loop bodies that cannot be closures. Each module opens only its own segment; nesting rebuilds
the full path, so the path a profile prints equals the state-dict key prefix of the weights it
touched. GigaAM and Whisper are instrumented this way, and
`model/src/test/unit/origin.rs` asserts the two sets of paths agree.

Pipeline stages are the root segment. GigaAM wraps the JIT build in a module scope, an
`arch` pipeline uses a label — either way the stage name heads every path minted inside:

```rust
scoped("ctc_head", || jit.prepare_with_config(mel_spec, lengths_spec, &config))?;

let _stage = OriginScope::label(self.profile_label());   // "vad"
self.vad.probs(waveform)?;
```

The `jit_wrapper!` macro opens `OriginScope::label("<WrapperName>")` around every build
closure, which is where the `GigaAmCtcJit` segment below comes from. Anything built outside a
scope lands on the `<unattributed>` row.

### ONNX graphs

Nothing to do. The importer opens one `Onnx` frame per node (index, name, op type, domain,
opset) and a `Label` for each subgraph branch (`then_branch`, `else_branch`) under the node
that owns it, so an `If` body reads `#7:If.then_branch.#0:Add`. Initializers and graph
inputs sit under `initializer` and `input` — but only when they build a computed node:
literals and buffers never carry an origin.

### Hand-written kernels

A `tk` kernel is attributed by the scope active when it is built — the same rule as a graph
kernel. The scheduler never sees its body, so `UOp::custom_kernel` harvests and strips it at
construction (`ir/src/uop/constructors/graph.rs`); two layers launching the same hand kernel
still share one compiled program. Inputs it copies in inherit their producer's origin.

---

## Reading the output

With capture on, `--profile` prints the usual per-kernel table and then two rollups. The
sample is the GigaAM v3 encoder, f16, one 60 s window on gfx1151, cut at depth 3:

```
519 dispatches (519 GPU-stamped), total 444.237 ms
  total ms  count    mean µs      %  name
   103.183     16     6448.9   23.2  r_128_3_32_4_2_2_2_4_4_192_2n1
   100.305     16     6269.1   22.6  r_128_3_32_4_2_2_2_4_4_192_2
    80.530     32     2516.6   18.1  r_128_12_32_4_2_2_2_4_4_48_2
    …
origin rollup (depth 3, exclusive; rows sum to the total):
  total ms  count    mean µs      %  origin path
    27.833     32      869.8    6.3  ctc_head.GigaAmCtcJit.layers.3
    27.678     32      864.9    6.2  ctc_head.GigaAmCtcJit.layers.9
    27.620     32      863.1    6.2  ctc_head.GigaAmCtcJit.layers.0
    …
    23.334      2    11666.8    5.3  ctc_head.GigaAmCtcJit.subsampling
     0.661      4      165.2    0.1  ctc_head.GigaAmCtcJit.head
     0.131      1      131.0    0.0  ctc_head.GigaAmCtcJit
origin rollup (depth 3, inclusive; parents contain children, rows overlap):
  total ms  count    mean µs      %  origin path
   444.237    519      855.9  100.0  ctc_head
   444.237    519      855.9  100.0  ctc_head.GigaAmCtcJit
    27.833     32      869.8    6.3  ctc_head.GigaAmCtcJit.layers.3
    …
```

How to read it:

- **Exclusive** charges each dispatch once, to its *primary* origin: the scope that produced
  the value the kernel stores (or, when that value carries none, the nearest attributed node
  below the root). Rows partition the total, so the sixteen `layers.N` rows plus
  `subsampling`, `head` and the residual `GigaAmCtcJit` row add up to 519 dispatches and
  444 ms. Sixteen layers at 32 dispatches each is the whole encoder; the per-layer spread
  (25.3 to 27.8 ms) is real and is what you would look at first. With fewer than twelve
  exclusive rows, each row also lists its top three kernels as `· name` lines.
- **Inclusive** charges a dispatch to every ancestor of every origin fused into it. A parent
  row contains its children, so `ctc_head` is 100 % and rows overlap. Use it to see how much
  of a block's time hides in kernels that fused across module boundaries.
- **Depth** is the number of path segments kept. Depth 3 gives per-layer rows here; depth 4
  splits a layer into `ffn1`, `mhsa`, `conv`, `ffn2`, `final_norm`; the leaf keeps the full
  path. `Call` frames never form rollup keys — they are detail on the kernel rows and in the
  JSON.
- A kernel that fuses two modules is charged exclusively to the one whose value it stores
  (the residual add lands on the layer, not on `ffn2`) and inclusively to both.

Any `RunProfile` carries both rollups: `render_report()` (the layout above, used by
`gigaam_infer`) and `render_table()` (Whisper's `--profile`, a different kernel table) append
the same origin section, cut at the depth the profile was produced with; `render_report_at(d)`
/ `render_table_at(d)` / `to_json_at(d)` override it.

### JSON

`--profile-json out.json` (or `RunProfile::to_json()`) writes one document per run
(abridged):

```json
{
  "origin_depth": 3,
  "stages": [{
    "name": "ctc_head", "wall_ms": 463.8, "gpu_ms": 444.2, "dispatches": 519, "meta": {},
    "kernels": [{
      "name": "r_128_3_32_4_2_2_2_4_4_192_2", "count": 1, "total_ms": 6.3, "mean_us": 6269.1,
      "origin": "ctc_head.GigaAmCtcJit.layers.3 @ add model/src/gigaam/encoder.rs:596",
      "origin_id": 41, "origins": ["…"], "origin_ids": [41, 39]
    }],
    "origins_exclusive": [{ "path": "ctc_head.GigaAmCtcJit.layers.3", "count": 32, "total_ms": 27.8, "mean_us": 869.8, "percent": 6.3, "kernels": [] }],
    "origins_inclusive": []
  }],
  "origins": [{ "id": 41, "parent": 40, "frame": { "Module": { "name": "layers.3" } } }]
}
```

Kernel rows are keyed by entry point *and* primary origin, so the same program appears once
per scope that dispatched it. `origins` holds only the frames the run referenced, closed
under `parent`, so ids resolve without the process that wrote the file.

---

## Threads

Capture state is per thread: the switch, the current scope, and whether that scope is a call
frame. Scopes do not follow work onto other threads; a scope guard is `!Send` and restores
the thread it was opened on. The rules that fall out:

- Build the graph on the thread that opened the scopes. GigaAM and Whisper do; a stage scope
  opened around `prepare_with_config` covers everything built inside it.
- Scheduling and compiling run detached (`OriginScope::suspend`), on the caller and on the
  rayon workers alike, so an ambient scope never leaks into a kernel body; attribution was
  already harvested onto the CALL by then.
- To carry a scope to a worker you spawn yourself, capture `origin::current()` (an
  `Option<OriginId>`) and re-install it there with `origin::install(id)`. Workers seed their
  switch from `SVOD_ORIGIN` like any other thread.
- BEAM search runs in a child process on origin-free kernel bodies; it never sees a scope.
- **Async code:** scopes must nest, so do not hold one across an `.await`. Open the scope,
  build the graph synchronously, drop it, then await. The guard is `!Send`, so a future that
  keeps one alive across an await cannot be spawned on a multi-threaded executor, and a
  debug build panics when two tasks interleave scopes on one thread (a guard dropped while a
  later one is still active). Graph construction in svod is synchronous, so the natural
  shape of the code already satisfies this.

---

## Costs and trade-offs

- **Off:** nothing. One thread-local read per node, no allocation, hashes unchanged.
- **On:** one interning per scope entry (a mutex over the arena, hundreds of times per
  forward), one thread-local write per public op for the call frame, and a toposort per
  kernel at the cut to harvest the union. GigaAM dispatch counts and GPU time are identical
  with capture on and off.
- **Identity changes.** Origin is part of a node's identity, so two identical expressions
  built under different scopes are two nodes until the cut strips them. Kernel programs are
  unaffected — the strip restores dedup — but a helper that rebuilds the same expression per
  call site (a mask clamp, a table cast, an input copy) would materialise it once per scope.
  Run such helpers under `OriginScope::suspend()` or let the copy inherit its producer's
  origin; `custom_kernel` already does the latter for its inputs. Constants, buffers, params,
  `UNIQUE`, `DEFINE_VAR`, `BIND`, `STACK` and every `Index`-typed node never carry an origin
  for the same reason.
- **Tests that rely on structural identity** (two hand-built graphs expected to hash-cons
  into one node) should run with `capture_for_thread(false)`.

---

## How it works, in one paragraph

Every `UOp` built while a scope is active stores the scope's 4-byte `OriginId` (a
`NonZeroU32`, so the `Option` is still four bytes) and folds it into its content hash, so
identical subgraphs from different scopes stay distinct through rangeify. At the kernel cut,
`split_store` walks the body once, takes the stored value's origin as primary and the union
as the set, stamps both on the kernel `CALL`'s `CallInfo`, and rebuilds the body with origins
cleared. Everything after the cut — optimizer, BEAM, codegen, every kernel cache — sees
origin-free ASTs. The plan copies the CALL's attribution onto each prepared op, the profiler
onto each `KernelProfile`, and the rollups truncate the parent chain to the requested depth.
