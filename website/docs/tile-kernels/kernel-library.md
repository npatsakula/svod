---
sidebar_label: Kernel Library
---

# Kernel Library

Every kernel in `svod_tk::kernels` is a function `fn k<T: Elem>(spec: &Spec) -> Program`,
generic over the 16-bit element type (`BF16` or `F16`); `gemm` and `nearest` also take 8-bit
integers (`I8`, accumulated exactly in i32: `gemm` stores f32, `nearest` scales each row's
products by the rows' `x_scale` and `c_scale`). Each spec has a `batch: Batch`:
`Batch::Static(n)` launches `n` batches, and `Batch::var(name, min, max)` (an `ir::Var`) launches the live
count of a runtime variable bound by name, with buffers sized for `max`. Each config type has a
`lowering(&device)` (an `atoms::Device`: the target and its costs). Models reach these kernels
through the [op layer](./op-layer), which picks the config.

| Kernel | Spec → program | Kernel name | Op |
|---|---|---|---|
| GEMM + epilogue | `GemmSpec` → `gemm::gemm` | `gemm` | `ops::linear` |
| Implicit-GEMM convolution | `ConvSpec` → `conv::conv` (+ `gemm::split_merge` when split) | `conv`, `split_merge` | `ops::conv2d` |
| Flash attention forward | `AttnSpec` → `attention::attention` | `flash_attention` | `ops::attention` |
| Split merge | `CombineSpec` → `attention::combine` | `combine_splits` | `ops::attention` with splits |
| Attention prologue | `HeadsSpec` → `heads::heads` | `heads` | `ops::heads` |
| LayerNorm / RMSNorm | `NormSpec` → `rows::norm` | `layer_norm` / `rms_norm` | `ops::{layer_norm, rms_norm, add_*}` |

## GEMM

`c = scale·act(a·bᵀ + bias) + residual`, with `a [batch·m, k]` and `b [n, k]`. Everything after
the matrix product runs on the f32 accumulator, and the result is rounded once at the store.

| `Epilogue` field | Effect |
|---|---|
| `bias` | `[n]` row added to the accumulator (`[2n]` when gated) |
| `act` | `Act::None`, `Act::Gelu` (erf approximation, error at most 1.5e-7), `Act::Silu` |
| `gated` | `b` is `[2n, k]`, gate rows over up rows; output `act(gate)·up` (SwiGLU, GeGLU) |
| `scale` | Multiplies the activated value before the residual |
| `residual` | `[batch·m, n]` added last |
| `out_f32` | f32 output and residual, the accumulator stored unrounded |

`GemmCfg { tile: [bm, bn, bk], stages, warps: [rows, cols], unroll, core, crews }`. Rows
past `m` and columns past `n` are bounded views, so `m` and `n` are free. `k` must be a multiple
of `bk`. `core: Core::Group` issues the products on `wgmma` (sm_90); `crews: 2` runs the two
warp rows in ping-pong over one slot (CDNA). The blocks walk the output tiles in groups of
8 tile rows (`gemm::GROUP_ROWS`), so blocks resident together read the same B tiles.

There is no tile table. `config::Planner::gemm_ranked` enumerates tiles, `bk`, warp grids, ring
depths and unrolling, keeps what the lowering runs on the target (the atoms tile the warp grid,
the fills split into whole 16-byte chunks per thread, the ring fits shared memory) and ranks the
rest in nanoseconds of SM time from the device's measured `Costs`. Every device input to the
ranking is measured; its few model constants (`BAND`, `CACHE_SHARE`, `CACHE_SPILL`,
`OVERLAP_WAVES` in `ops::config`) are meant to hold for every device. The ranking orders, and
the tuner measures the first 64 tiles that compile without spilling ([Tuning](./tuning)).

## Convolution

`conv::conv` is an implicit GEMM over channels-last `x [b, h, w, cin]` and `w [cout, kh, kw,
cin]`: A is a row-mapped view of the image (`conv::im2col_view`), so padding rows read zero and
nothing is materialized. `K = kh·kw·cin` runs as one pipeline, each step inside one tap (`bk`
divides `cin`). On a starved grid of a single static batch, `ConvCfg::split` splits K over
grid z and `gemm::split_merge` applies the epilogue. `ops::conv2d` takes a 1×1 as `linear` and
keeps `groups ≠ 1`, `cin` off 16 and `cout` off 8 on the graph.

## Flash attention

`o = softmax(q·kᵀ·scale + bias)·v` over sequence-major `q [batch, t, heads, d]` and
`k`, `v [batch, tk, kv_heads, d]`. Q stays in registers and K/V stream through a shared ring.
The online-softmax state `(m, l, o)` is carried in f32 using `exp2`.

| Feature | How |
|---|---|
| GQA | Query head `h` reads KV head `h / (heads / kv_heads)` |
| Cross attention | `tk ≠ t` is a property of the views |
| Any `t`, `tk` | Bounds instead of padding: the query tile and the last key block are bounded views, and keys past `tk` are masked |
| Head dims | Any `d` both products tile, whose fills split into whole chunks per thread (`FaCfg::fits`); the models use 48, 64 and 128 |
| `AttnMask::causal` | Key blocks past the diagonal are skipped through the trip count |
| `AttnMask::window` | `(left, right)` around each query; blocks outside are skipped |
| `AttnMask::chunk` | Block attention: a query sees its chunk, `left` chunks before (all, without a bound) and `right` after |
| `AttnMask::key_lens` | `[batch]` i32 valid key counts; blocks past the length are skipped |
| `AttnMask::key_mask` | `[batch, tk]` i32 visible keys (row stride rounded up to 8), read in every block |
| `AttnMask::seg_start` | `[batch, t]` i32 segment starts of packed rows, non-decreasing along `t` |
| `AttnMask::bias` | `[batch or 1, heads, t, tk]` additive bias in the stream type (row stride rounded up to 8), added to the scaled scores before the masks |

Only key blocks that a causal, window, chunk, length or segment edge crosses compute the
predicate; blocks fully inside run unmasked. A key mask and a bias are read in every block. A
query that sees no key yields zeros, as the graph writes it.

**Cache mode** (`AttnSpec::cache = Some(Cache { .. })`) reads K and V from a cache
`[rows, tk, heads_total, d]` that holds several layers' heads:

| `Cache` field | Effect |
|---|---|
| `rows`, `heads_total` | The cache's shape |
| `row_map` | A `[batch]` i32 parameter: the cache row each batch lane reads |
| `appended` | Rows of `[batch, appended, kv_heads, d]` keys and values scored after the cached prefix (the tokens a decoder step just projected), at most a key block |
| `app_lens` | A `[batch]` i32 parameter of visible appended rows |

The first of this attention's heads is a `[1]` i32 `head_start` parameter, so every layer runs
one program.

**Key splits** (`FaCfg::splits > 1`) give a few long rows, such as a decoder step, more blocks.
Each split writes f32 partials `o_part`, `m_part` and `l_part`, and `combine` merges them.
`Attn::splits = Some(n)` fixes the count; with `None`, `Planner::splits` proposes 1, 2, the
counts around two blocks per SM and one split per key block, and the tuner times each split
candidate together with its merge.

`FaCfg { bq, bkv, stages, splits, unroll, forms, key_warps, warp_rows, core }`:

| Field | Effect |
|---|---|
| `warp_rows` | Query rows per warp: 16 (one atom row) or 32 (each K and V fragment feeds two atom rows); the 32-row form is a variant the tuner times beside its 16-row tile |
| `key_warps` | A one-row-warp block (`bq = 16`, offered for `t ≤ 16`) splits its key blocks over 1, 2 or 4 warps, each with its own ring slots and softmax state, merged through shared memory |
| `core` | `Core::Group` runs both products on `wgmma` (sm_90) |
| `forms` | The softmax forms below |

Candidates come from the same lattice and cost model as the GEMM's
(`Planner::attention_ranked`). Where the fill goes through registers, the value tile is stored
column-major so its fragments gather as 16-byte loads; on RDNA4 the fill itself is a transposing
load (`global_load_tr_b128`).

The softmax adopts a new running maximum only when it grows by more than 8 in log2 units
(FlashAttention-4's lazy rescale). Five `Forms` are tuner knobs, all off as planned; the tuner
flips each where it measures faster (a forms pass, then the tiles under those forms, then the
forms again on the winner):

| `Forms` field | Effect |
|---|---|
| `truncate_p` | bf16 probabilities are truncated rather than rounded, and the row sum adds what is held |
| `sum_after` | The row sum follows the P·V product rather than preceding the rescale |
| `skip_rescale` | The output's rescale is a `Scale`, skipped by every wave whose rows all kept their maximum |
| `raw_max` | The running maximum stays in the raw scores' units: a multiply per row rather than per score |
| `bias_direct` | Each key block's bias tile is read from global memory in the scores' layout rather than through a ring slot |

## Attention prologue

`heads` splits a fused projection `qkv [batch, t, (heads + 2·kv_heads)·d]` into sequence-major
`q [batch, t, heads, d]` and `k`, `v [batch, t, kv_heads, d]`. Optionally it RMS-normalizes `q`
and `k` over the head with `[d]` weights, then applies rotary embedding to them. The rotation
pairs the head's two halves, using `(cos, sin)` tables of `[t, d/2]`, shared or per batch
(`Rope { per_batch }`). A block handles `br` rows of one head slot. `d` must be a power of two
in 16..=256, and candidates are `br ∈ {4, 8, 16}`.

## Norms

`rows::norm` uses one warp per row and a fused reduce-then-map in f32:

| `Norm` | Formula |
|---|---|
| `Layer` | `(x − mean)·rsqrt(var + eps)·w + b` |
| `Rms` | `x·rsqrt(mean(x²) + eps)·w` |

With `residual: true` it normalizes `x + residual` rounded to the element type and also writes
that sum, which is the pre-norm residual stream of a transformer layer. `d` must be a power of
two in 256..=2048, and candidates are `br ∈ {4, 8, 16}` rows per block.
