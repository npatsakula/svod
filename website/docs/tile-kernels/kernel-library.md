---
sidebar_label: Kernel Library
---

# Kernel Library

Every kernel in `svod_tk3::kernels` is a function `fn k<T: Elem>(spec: &Spec) -> Program`,
generic over the 16-bit element type (`BF16` or `F16`). Each spec has a `batch: Batch`:
`Batch::Static(n)` launches `n` batches, and `Batch::Var { name, min, max }` launches the live
count of a runtime variable bound by name, with buffers sized for `max`. Each config type has a
`lowering(&device)` (an `atoms::Device`: the target and its costs). Models reach these kernels through the [op layer](./op-layer), which picks
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

The op layer does not keep a tile table. `config::Planner::gemm_ranked` enumerates the construct's
knobs (output tiles of 32–256 rows and 48–256 columns, `bk` 16–64, nine warp grids, the ring
depths the target's prefetch runs, rolled and unrolled), keeps what the lowering can run (the
atoms tile the warp grid, the fills split into whole 16-byte chunks per thread, the ring fits
shared memory) and ranks the rest in nanoseconds of SM time from the device's measured `Costs`.
A block's trip does `work`: every wave's operand fragments read from shared memory, the fill
streamed in, the matrix-core steps and the trip's fixed cost. The busiest SM issues the work of
`per_sm = ⌈blocks / sms⌉` blocks, each taken up as another ends, the trip's fixed cost shared
with the blocks it holds at once (`resident`, by shared memory and the register estimate); a round
of those blocks is at least one wave's own issue (a block of fewer waves than the SM has SIMDs
leaves the rest idle) and the memory latency one pipeline stage ahead cannot hide:
`cost = trips × max(per_sm × (work + trip / resident), rounds × max(simds × (work + trip) / waves, latency / (stages − 1)))`
with `rounds = ⌈per_sm / resident⌉` (`Planner::pipelined`); a SIMD holding fewer waves than
`step_lone / step`, the waves that hide a matrix-core chain's latency, runs their issue `work` that
many times slower, the fill streaming on meanwhile. A fill's byte is priced by where it
comes from: the L2 rate where the distinct bytes fit three quarters of the cache, the DRAM rate
past half again as much, a blend between (`Planner::fill_byte`). Without the wave bound,
gfx1151's ranking led with one-warp 256×256
tiles that ran at 1–3 TFLOP/s. The layout's register count only estimates how many blocks are resident;
it admits nothing, since it over-counts where fragments are replicated (gfx11 WMMA). The ranking
keeps every pipeline of a tile; the [tune store](./tuning) picks among the first 64 output
tiles and warp grids whose compiled kernels do not spill, each at its first clean pipeline (all
spilling: the first 64 that compile), then among the winning tile's other pipelines. The compiler
admits and the clock picks because the model orders block tiles well and warp grids poorly (on
sm_86 the measured best sat as deep as the 92nd of 109 tiles); the pipelines it ranks alone until
the winner is known, so an
unrolled ring that spills yields to its rolled twin: on gfx1151 the unrolled 64×16 and 128×16
attention tiles at d = 128 spill and the rolled ones compile clean. A
gated weight
is a `2·bn`-wide B tile and two accumulators to the model, so its lead comes at half width. The
`Planner` of a device holds its target, its measured costs and the problems it has ranked (an op
plans its shape at every call; ranking the lattice takes a millisecond), one planner per device
for the process.

The model has no fitted constants: every cost is the device's own, measured by a probe. Where the
ranking and a measurement disagree, the measurement wins: add the shape to
`qwen3_gemm_candidates_probe` and check which cost the ranking misses.

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

`FaCfg { bq, bkv, stages, splits, unroll, .. }` uses one warp per 16 query rows. Candidates come
from the same lattice and cost model as the GEMM's (`Planner::attention_ranked`): query
blocks of 16–128 rows, key blocks of 16–128 keys, the ring depths the target runs, rolled and
unrolled, kept to what lowers (the fills split into chunks per thread, the ring fits shared
memory) and ranked by what a key block costs (every warp's K and V fragments read, both tiles
streamed in, the matrix-core steps, the softmax's vector work per score and the trip's fixed
cost), over the key blocks of every round of query blocks, each round bound as the GEMM's. The
register estimate is the query, key, score, probability and value fragments, live together in the
compiled loop; whether a config spills is the compiled kernel's to say. A warp past the problem's
queries adds the work of its padded rows, so where the tiles fill the device a decoder step leads
with one-warp blocks; where they leave SIMDs idle a padded warp runs on one of them. A causal
mask charges each query
block its diagonal. The value tile is stored column-major where its fill goes through registers
(see [Layouts and Lowering](./layouts-and-lowering#layouts-and-lowering)), so its fragments gather
as 16-byte loads like the keys'; measured on gfx1201 that is 5–28% per shape over the row-major
tile's one load per key. On RDNA4 the fill itself is a transposing load (`global_load_tr_b128`),
so its registers hold column runs and store as 16-byte vectors too.

The softmax step does as little vector work as it can, which matters most where the matrix
cores share the SIMD's ALUs (RDNA). A row adopts a new running maximum only when it grows by
more than 8 in log2 units (FlashAttention-4's lazy rescale): otherwise the correction is exactly
1, while the probabilities stay below 2⁸. The planner offers every config in the plain forms; the
other four (`FaCfg::forms`: `truncate_p`, `skip_rescale`, `raw_max`, `sum_after`) are tuner knobs, tuned in
turn with the tiles (`ops::attention::tune`, through `descend`), since a tile may compile clean only under some
forms: a forms
pass on the ranking's first config that compiles clean, then the tile and split choice over the
ranking under the forms it kept, then a forms pass again on that winner (store-backed, so cheap
when nothing changes). A forms pass is a coordinate descent: each form once, in that order, a
two-candidate tune of the forms so far against the same with that form
flipped, the flip kept only where it measures faster (a spilling flip is not admitted). Which
form saves registers is not a rule anywhere: the measurement decides. With tuning off every
attention runs the plain forms. With `truncate_p`,
probabilities held in bf16 are truncated rather than rounded, a pair per instruction where the
rounding is several, and the row sum adds the values as held, so the normalization stays unbiased;
targets that round a pair in one instruction (sm_80+, gfx950) round and sum the f32 values, the
extra conversion of the held values having cost 3–5% on sm_86. With `skip_rescale`, the output
accumulator's rescale is a `Scale`, skipped by every wave whose rows all kept their maximum; it
paid wherever the matrix products issue on the vector ALUs, and beside separate matrix cores only
where a full tile's lane holds 64 output registers (sm_86: +4% at d = 128, neutral at d = 64 and
5–7% slower on a decoder step, whose one row keeps the plain multiply). With `raw_max` (paid on RDNA
tiles full of queries), the running maximum stays in the raw scores' units (a positive scale commutes
with the maximum; a bias keeps the scaled form), so only the two row values the exponent and the
correction take are scaled, a multiply per row rather than per score, at the price of a multiply on
the step's dependent chain, which a decoder step's one wave feels (3%) and a full tile does not. With
`sum_after`, the row sum follows the product rather than preceding the rescale; truncated
probabilities are summed as held either way (the bf16 pair dot product where the target has it),
rounded ones in f32. After the product both consumers read truncated values packed once (gfx1201);
before it, on sm_86 with rounding, the order was 3% faster at d = 128 (24.7 against 24.0 TFLOP/s,
22.4 before this work) and neutral at d = 64 and on a decoder step. Under a causal mask (or chunks seeing every earlier one) the last
query blocks, which score the most keys, run first, so a round's tail is its shortest blocks. On
gfx1201 the lazy rescale, the truncation, the skip and the raw `exp2` below took the d = 128 kernel
from 90 to 116 TFLOP/s on random data and Qwen3's causal attention from 63 to 77; the raw maximum,
the pair dot product and the block order took them on to 120 and 79, d = 128 causal 96 → 100 and
d = 64 105 → 108 (causal 87 → 92), the decoder step unchanged. What remains against the 162 TFLOP/s
the loop's WMMAs alone would run at is spread thin: the vector work is 15%, the loads, stores,
barriers and addressing the rest.

A decoder step's block is one row warp of queries (`bq = 16`) walking every key block in turn, a
single wave waiting on each block's load. `FaCfg::key_warps` lets that block's warps share its keys
instead: each walks its own range of key blocks through its own ring slots with its own softmax
state, inside a `Warps` statement (every warp on its own, warp-scoped copies and barriers), and the
states merge through shared memory at the end, as the split merge does. The planner offers 1, 2 and
4 key warps for `t ≤ 16`; the tuner picks. On gfx1201 the Whisper large-v3 cross-attention step
(1500 keys, one cache row shared by five lanes) went from 25.3 to 15.7 µs tuned (four splits) and to
18.2 unsplit (two key warps), tk1's single-query kernel being 16.1; the self-attention step is
unchanged. The states merge through the warps' own ring slots, dead once their pipelines end
(f32 views of the 16-bit rings: a warp's output over its key slots, its maximum and sum over its
value slots, `Kernel::smem_view_as`), so four key warps at `bkv = 16` take the rings' 32 KB alone
and two blocks fit a CU: the unsplit cross-attention step went from 19.6 to 15.8 µs, the four-split
one's time, and the tuner picks it unsplit. The tuner also measures a cache the way the model reads it: fewer cache
rows than lanes are read through a zero row map, so lanes share a row, where it used to time a row
per lane, 38 MB from DRAM for a shape that is 7.7 MB in L2.

Each tile config is crossed with split counts. `Attn::splits = Some(n)` fixes `n`: only the
configs with at least `n` key blocks are ranked, at `n` (capped at the most key blocks any config
has), so a named count is never dropped by a large key block. With `None`, `Planner::splits`
proposes 1, 2, the counts around two
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

Open: the decode `row_map` is still excluded from the tuned shape (the tuner measures on scratch
caches of one row per lane, where a map would read garbage rows), so a tuned cross attention may
not be the fastest with its map: on gfx1151 (2026-10-14) it tuned to 90 µs against 60 µs unsplit.

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
