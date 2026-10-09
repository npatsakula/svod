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

The op layer draws candidates from four tile families, each with its best pipeline as measured
on sm_86:

| Family | Stages | Warp grid | Unroll | Measured note |
|---|---|---|---|---|
| 128×128×32 | 3 | 2×4 | no | 4096³ peaks here: 25.4 TFLOP/s |
| 128×64×32 | 2 | 2×2 | yes | |
| 64×128×32 | 2 | 2×2 | no | |
| 64×64×32 | 3 | 2×2 | yes | M = 704 (Nemotron): 128-row tiles lose to 64×64 |

`config::gemm_candidates` leads with the largest family whose grid gives every SM 8 blocks and
pads at most 1/16 of the output area (else 64×64). It adds that family's variants (the other
stage count, `bk = 64`, flipped `unroll`, transposed warp grid), then the other families. `bk`
halves down to 16 when `k` is not a multiple of it, and configs whose ring exceeds the target's
shared memory are dropped. At most eight candidates remain, and the [tune store](./tuning)
picks among them.

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

`FaCfg { bq, bkv, stages, splits }` uses one warp per 16 query rows. Candidates by head dim,
first is the untuned pick:

| `d` | Candidates `(bq, bkv, stages)` | Measured note (sm_86) |
|---|---|---|
| 48, 64, 128 with `t ≤ 16` | (16, 64, 2), (16, 64, 3), (16, 32, 2) | A decoder step is bandwidth-bound: one-warp blocks leave room for several per SM |
| 48 | (64, 64, 2), (64, 64, 3) | Shapes whose K/V fills divide among the block's threads (96-byte rows) |
| 64 | (64, 64, 2), (64, 64, 3), (128, 64, 2), (64, 32, 2), (128, 32, 2), (64, 32, 3) | |
| 128 | (64, 32, 2), (64, 32, 3), (128, 32, 2), (64, 64, 2), (128, 64, 2), (128, 32, 3) | `bkv = 64` measured 18.2 TFLOP/s against 22.2 for `bkv = 32`: 64 KB per block leaves one block per SM |

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
