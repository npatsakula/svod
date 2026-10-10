---
sidebar_label: Kernel Library
---

# Kernel Library

Every kernel in `svod_tk3::kernels` is a function `fn k<T: Elem>(spec: &Spec) -> Program`,
generic over the 16-bit element type (`BF16` or `F16`). Each spec has a `batch: Batch`:
`Batch::Static(n)` launches `n` batches, and `Batch::Var { name, min, max }` launches the live
count of a runtime variable bound by name, with buffers sized for `max`. Each config type has a
`lowering(target)`. Models reach these kernels through the [op layer](./op-layer), which picks
the config.

| Kernel | Spec → program | Kernel name | Op |
|---|---|---|---|
| GEMM + epilogue | `GemmSpec` → `gemm::gemm` | `gemm` | `ops::linear` |
| Implicit-GEMM convolution | `ConvSpec` → `conv::conv` (+ `gemm::split_merge` when split) | `conv`, `split_merge` | `ops::conv2d` |
| Flash attention forward | `AttnSpec` → `attention::attention` | `flash_attention` | `ops::attention` |
| Split merge | `CombineSpec` → `attention::combine` | `combine_splits` | `ops::attention` with `splits` |
| Attention prologue | `HeadsSpec` → `heads::heads` | `heads` | `ops::heads` |
| LayerNorm / RMSNorm | `NormSpec` → `rows::norm` | `layer_norm` / `rms_norm` | `ops::{layer_norm, rms_norm, add_*}` |

## GEMM

`c = act(a·bᵀ + bias) + residual`, with `a [batch·m, k]` and `b [n, k]`. Everything after the
matrix product runs on the f32 accumulator, and the result is rounded once at the store.

| `Epilogue` field | Effect |
|---|---|
| `bias` | `[n]` row added to the accumulator (`[2n]` when gated) |
| `act` | `Act::None`, `Act::Gelu` (erf approximation, error at most 1.5e-7), `Act::Silu` |
| `gated` | `b` is `[2n, k]`, gate rows over up rows; output `act(gate)·up` (SwiGLU, GeGLU) |
| `residual` | `[batch·m, n]` added last |

`GemmCfg { tile: [bm, bn, bk], stages, warps: [rows, cols], group_m, unroll }`. Rows past `m`
and columns past `n` are bounded views, so `m` and `n` are free. `k` must be a multiple of `bk`.

The op layer does not keep a tile table. `config::Planner::gemm_candidates` enumerates the construct's
knobs (output tiles of 32–256 rows and 48–256 columns, `bk` 16–64, nine warp grids, the ring
depths the target's prefetch runs, rolled and unrolled), keeps what the lowering can run (the
atoms tile the warp grid, the fills split into whole 16-byte chunks per thread, the ring fits
shared memory, a lane's registers fit the file) and ranks the rest by a traffic model: the bytes
a block's trip moves through shared memory, plus a fixed cost per trip and per matrix-core step,
over the trips of every round of blocks the device runs, stretched by how few waves an SM holds
and how few steps its ring keeps in flight. One pipeline per output tile and warp grid is kept,
at most eight candidates remain, and the [tune store](./tuning) picks among them. A gated weight
is a `2·bn`-wide B tile and two accumulators to the model, so its lead comes at half width. The
`Planner` of a device holds its target and the problems it has ranked (an op plans its shape at
every call; ranking the lattice takes a millisecond), one planner per device for the process.

The model was fitted on gfx1201 against a measured lattice of 94 configs on Qwen3's four
projections (the measured best is within the shortlist's top five on each) and checked against
the sm_86 facts the earlier tables recorded (4096³ peaks at 128×128 with 3 stages on a 2×4 warp
grid; that config stays in the shortlist). Where the model and a measurement disagree, the
measurement wins: add the shape to `qwen3_gemm_candidates_probe` and refit the constants.

## Flash attention

`o = softmax(q·kᵀ·scale)·v` over sequence-major `q [batch, t, heads, d]` and
`k`, `v [batch, tk, kv_heads, d]`. Q stays in registers and K/V stream through a shared ring.
The online-softmax state `(m, l, o)` is carried in f32 using `exp2`.

| Feature | How |
|---|---|
| GQA | Query head `h` reads KV head `h / (heads / kv_heads)` |
| Cross attention | `tk ≠ t` is a property of the views |
| Any `t`, `tk` | Bounds instead of padding: the query tile and the last key block are bounded views, and keys past `tk` are masked |
| Head dims | 48, 64, 128 |
| `AttnMask::causal` | Key blocks past the diagonal are skipped through the trip count |
| `AttnMask::window` | `(left, right)` around each query; blocks outside are skipped |
| `AttnMask::key_lens` | `[batch]` i32 valid key counts; blocks past the length are skipped |
| `AttnMask::key_mask` | `[batch, tk]` i32 visible keys (row stride rounded up to 8), read in every block |
| `AttnMask::seg_start` | `[batch, t]` i32 segment starts of packed rows, non-decreasing along `t` |
| `AttnMask::bias` | `[batch or 1, heads, t, tk]` additive bias in the stream type (row stride rounded up to 8), read in every block and added to the scaled scores before the masks |

Masks are tile-class work. Only key blocks that a causal, window, length or segment edge
crosses compute the predicate (`select_if` on the block start); blocks fully inside run
unmasked. A bool key mask is applied in every block. A query that sees no key yields NaN, as a
softmax over an empty row does.

**Cache mode** (`AttnSpec::cache = Some(Cache { .. })`) reads K and V from a cache
`[rows, tk, heads_total, d]` that holds several layers' heads:

| `Cache` field | Effect |
|---|---|
| `head_start` | First of this attention's `kv_heads` heads in the cache row |
| `row_map` | A `[batch]` i32 parameter: the cache row each batch lane reads |
| `appended` | `[batch, kv_heads, d]` key and value scored after the cached prefix (the token a decoder step just projected) |

**Key splits** (`FaCfg::splits > 1`) give a few long rows, such as a decoder step, more blocks.
Each split writes f32 partials `o_part`, `m_part` and `l_part`, and `combine` merges them.

`FaCfg { bq, bkv, stages, splits, unroll }` uses one warp per 16 query rows. Candidates come
from the same lattice and traffic model as the GEMM's (`Planner::attention_candidates`): query
blocks of 16–128 rows, key blocks of 16–128 keys, the ring depths the target runs, rolled and
unrolled, kept to what lowers (the fills split into chunks per thread, the ring fits shared
memory, a lane's registers fit) and ranked by the bytes a key block moves (every warp's K and V
fragments in, the fill out), the matrix-core steps, the softmax's vector work per score and the
trip's fixed cost, over the key blocks of every round of query blocks, stretched by how few waves
an SM holds. Two facts the model learned from the ISA on gfx1201: the compiled loop keeps the
query, key, score, probability and value fragments live together, so the register count is
their sum and 16-key blocks are what fits at `d = 128`; and a warp past the problem's queries
hides no latency, so a decoder step leads with one-warp blocks. A causal mask charges each query
block its diagonal. The value tile is stored column-major where its fill goes through registers
(see [Layouts and Lowering](./layouts-and-lowering#layouts-and-lowering)), so its fragments gather
as 16-byte loads like the keys'; measured on gfx1201 that is 5–28% per shape over the row-major
tile's one load per key. On RDNA4 the fill itself is a transposing load (`global_load_tr_b128`),
so its registers hold column runs and store as 16-byte vectors too.

Each tile config is crossed with split counts. `Attn::splits = Some(n)` fixes `n`, capped at the
key block count. With `None`, `config::split_candidates` proposes 1, 2, the counts around two
blocks per SM, and one split per key block. It keeps only counts up to the smaller of the key
block count and twice that SM target. The unsplit
config comes first, and a split candidate is timed together with its merge.

The attention throughput probe (B 4, H 8, T 2048, bf16) measured in TFLOP/s against tk1:

| Case | tk3 | tk1 |
|---|---|---|
| d 64 | 24.1 | 23.0 |
| d 64 causal | 22.6 | 20.7 |
| d 128 | 22.3 | 22.3 |
| d 128 causal | 20.9 | 16.5 |

The decode probe times one Whisper large-v3 decoder step's attention in µs. The Whisper decoder
step calls these kernels:

| Case | tk3 | tk1 |
|---|---|---|
| Self attention, 200 cached keys + the appended one | 21.5 | 89.1 |
| Cross attention, 1500 shared keys | 66.3 | 78.8 (636 unsplit) |

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
two in 256..=2048, and candidates are `br ∈ {4, 8, 16}` rows per block. Measured at 92% of
memory bandwidth on sm_86.
