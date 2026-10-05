---
sidebar_label: The Kernel Library
---

# The Kernel Library

The USE face: every kernel `svod-tk` ships, callable with plain tensors and no knowledge of
tiles. Each returns a lazy `Tensor` (an `Op::Call` node) that composes into a model graph and
realizes through the normal `prepare()` path, and each follows the three-way contract from
[Authoring into the IR](./lowering):

| Result | Meaning |
|---|---|
| `Ok(Some(out))` | the kernel ran |
| `Ok(None)` | it does not apply: the device is outside the kernel's `ArchSet`, its LLVM backend is missing, or the shape does not tile — fall back deliberately |
| `Err(LaunchError)` | the request is malformed (dtype, rank, a symbolic dim, a divisibility rule) — a caller bug |

Operands are bf16 or f16 unless noted; accumulation is f32.

---

## Targets

Each kernel declares its own `ArchSet` (`tk/src/target.rs`): an explicit AMD list plus an
open-ended CUDA capability floor and an Apple GPU-family floor.

| Kernel | gfx942 (CDNA3) | gfx1151 (RDNA3.5) | gfx1200 / gfx1201 (RDNA4) | CUDA sm_80+ | Metal Apple7+ |
|---|---|---|---|---|---|
| `flash_attention` / `_with` / `_tuned` | yes | yes | yes | yes | yes |
| `matmul` (square) | yes | yes | yes | yes | yes |
| `gemm_nt` / `_with` / `_with_epilogue` | — | yes | yes | yes | — |
| `rms_norm` / `add_rms_norm` | — | yes | yes | yes | — |
| `single_query_attention` / `_packed` | yes | yes | yes | yes | — |
| `knn` | yes | yes | yes | — | — |
| `kmeans_assign` | yes | yes | yes | — | — |

The constants are `FA_SUPPORTED_ARCHS`, `MATMUL_SUPPORTED_ARCHS`, `GEMM_NT_SUPPORTED_ARCHS`,
`NORM_SUPPORTED_ARCHS`, `SQ_ATTENTION_SUPPORTED_ARCHS`, `KNN_SUPPORTED_ARCHS` and
`KMEANS_SUPPORTED_ARCHS` in `tk/src/kernels/`. A family joins a kernel by validation and by
measuring its own tile table — `gemm_nt` and the norms are wave32-only because nobody has
measured a wave64 table for them, not because the body cannot run there.
`flash_attention_supported(&device)` answers the arch gate alone, for callers that pad or
bucket a sequence length ahead of the launch.

---

## Flash attention

```rust
pub fn flash_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> LaunchResult<Option<Tensor>>
pub fn flash_attention_with(q, k, v, opts: FaOpts) -> LaunchResult<Option<Tensor>>
pub fn flash_attention_tuned(q, k, v, opts, policy: impl Fn(&DeviceSpec, GpuArch) -> FaPolicy + Copy) -> ..

pub struct FaOpts<'a> {
    pub causal: bool,                      // default true
    pub key_lens: Option<&'a Tensor>,      // [B] i32 valid-key counts: keys >= key_lens[b] are masked
    pub seg_start: Option<&'a Tensor>,     // [B, N] i32: query q of batch b sees no key before seg_start[b, q]
}
```

`q` is `[B, N, H, D]`, `k`/`v` are `[B, N, H_kv, D]` (GQA: `H % H_kv == 0`), the output is
`[B, N, H, D]` in the operand dtype. Sequence-major, not head-major — the model reshapes a
projection straight into it without a transpose.

- `Ok(None)`: off the arch set; `N` not a multiple of `q_blk · 8` (the per-warp Q tile times
  the eight waves of a workgroup; `FLASH_ATTENTION_SEQUENCE_MULTIPLE` is the baseline's
  `128`); a KV length that differs from `N` (cross-attention is not implemented); a head dim
  whose double-buffered K/V tiles exceed the device's shared memory.
- `Err`: dtype outside `{bf16, f16}` or differing between `q` and `k`/`v`; `D % 16 != 0`;
  `H % H_kv != 0`; a `k`/`v` shape other than `[B, N, H_kv, D]`.

`key_lens` masks keys only — padded query rows are still computed and the caller discards
them. A `key_lens[b] == 0` is clamped to `1` so the row stays finite. `seg_start` packs
several sequences into one row: each entry must lie in `0..=q` and leave at least one visible
key. [Flash Attention](./flash-attention) is the worked example; the per-warp tile is
measured on first use ([Autotuning](./tuning)).

---

## GEMM

```rust
pub fn matmul(a: &Tensor, b: &Tensor) -> LaunchResult<Option<Tensor>>              // [n, n] · [n, n] → f32
pub fn gemm_nt(x: &Tensor, w: &Tensor) -> LaunchResult<Option<Tensor>>             // [lead..., K] · [N, K]ᵀ → [lead..., N]
pub fn gemm_nt_with(x, w, cfg: impl Fn(usize, usize, usize) -> Option<GemmCfg> + Copy) -> ..
pub fn gemm_nt_with_epilogue(x, w, epilogue: Epilogue<&Tensor>) -> LaunchResult<Option<Tensor>>

pub enum Epilogue<T> {
    Plain,               // y = x·wᵀ
    Add(T),              // y = x·wᵀ + residual, residual [lead..., N] in the operand dtype
    SwiGlu { pair: usize }, // y = silu(gate)·up off a fused [2I, K] gate/up weight; y is [lead..., N/2]
}
pub fn swiglu_pair_width(spec: &DeviceSpec) -> Option<usize>
```

`matmul` is the square reference kernel: any float dtype in (cast to bf16), f32 out, every
arch. It is the performance canary for the DSL, not a production GEMM.

`gemm_nt` is the production linear layer. `x` is `[lead..., K]` of any rank ≥ 2 (a
`[B, L, K]` activation binds without a reshape or copy), `w` is `[N, K]` as a weight is
stored, and `y` is `[lead..., N]` in the operand dtype — the f32 accumulators are narrowed in
registers, so there is no f32 round trip through memory. `M = ∏lead` and `N` must be
multiples of 64 and `K` a multiple of the 32-wide strip with at least two strips; otherwise
`Ok(None)` and the caller pads to 128 or uses `Tensor::linear`.

The epilogues are why the kernel exists: they fold the pass the graph would pay after the
GEMM into its store. `Add` reads the residual at the store's own offset and adds in the
output dtype, exactly as the graph's `try_add` rounds. `SwiGlu` needs the fused weight's rows
arranged in alternating gate/up blocks of `pair` rows — `swiglu_pair_width(&device)` is
`reg_n / 2` of the device's tile table, or `None` when the tiles disagree (the caller keeps a
separate SwiGLU pass). The model loads the weight in that order once, because `M` picks the
tile at launch time and every candidate tile must read the same arrangement.

The tile comes from `GemmPolicy` (`tk/src/kernels/gemm.rs`): a per-family table —
`CUDA_TILES`, `RDNA_TILES`, `RDNA4_TILES` — measured on first use per shape and epilogue
([Autotuning](./tuning)), or the static `GemmPolicy::cfg` choice when tuning is off.
`gemm_nt_with` takes the chooser from the caller (the benches sweep it this way).

---

## RMS norm

```rust
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> LaunchResult<Option<Tensor>>
pub fn add_rms_norm(x, residual: &Tensor, weight, eps) -> LaunchResult<Option<(Tensor, Tensor)>>   // (h, y)
pub fn select_norm_cfg(rows: usize, d: usize, lanes: usize) -> Option<NormCfg>
```

`x` is `[rows..., D]`, `weight` `[D]`, both in one 16-bit dtype. One wave per row, the row
held in registers, the sum of squares completed by a butterfly shuffle — no LDS, no barrier,
no `RANGE`. Numerics mirror the graph op for op:
`y = dtype((f32(x) · rsqrt(Σx²/D + eps)) · f32(w))`, one rounding at the end; only the
summation order differs. `Ok(None)` when `D` is not a multiple of the wave or exceeds 64
elements per lane (`2048` at wave32).

`add_rms_norm` returns `(h, y)` with `h = x + residual` rounded as the graph's add rounds it
and `y = rms_norm(h)`, so a pre-norm decoder layer writes and reads its residual stream once.
When the preceding projection took the residual through `Epilogue::Add`, the two-pass
`rms_norm` is enough; `model/src/qwen3/decoder_layer.rs` chooses between them per layer.

---

## Single-query attention

```rust
pub fn single_query_attention(q, k, v, opts: SqAttentionOpts<'_>) -> LaunchResult<Option<Tensor>>
pub fn single_query_attention_packed(q, k, v, head_offset: usize, opts) -> LaunchResult<Option<Tensor>>

pub struct SqAttentionOpts<'a> {
    pub key_lens: Option<&'a Tensor>,                 // [B] i32, entries in 0..=N
    pub include_last: bool,                           // also score key N-1 (Whisper's self-cache slot)
    pub appended: Option<(&'a Tensor, &'a Tensor)>,   // the step's own [B, 1, H, D] K/V, scored after the prefix
    pub split: Option<usize>,                         // K/V chunks; None = the device's SqPolicy, tuned on first use
    pub cache_map: Option<&'a Tensor>,                // [B] i32: which K/V row each query row reads
}
```

The decode-step kernel: `q` is `[B, 1, H, D]` in f32, `k`/`v` are `[B, N, H_total, D]` in
f32, f16 or bf16 (or `[1, N, H_total, D]` to serve every row from one cache), the output
`[B, 1, H, D]` in f32.
One wave owns one `(batch, head)`; `Q` stays in registers while K/V stream over `N`; dot
products are XOR-shuffle all-reduces and the softmax is a one-pass online update. There is no
LDS and no matrix core, which is why its `ArchSet` is the widest AMD list plus CUDA.

`_packed` selects heads `head_offset..head_offset + H` of a packed cache without slicing it.
Long unmasked attention splits K/V into contiguous chunks, each a wave, and merges their
softmax states in a second pass; `SqPolicy` sizes the split from the device's resident-wave
budget and measures the nearest divisors on first use.

---

## k-NN and k-means

```rust
pub fn knn(x: &Tensor, c: &Tensor, k: usize) -> LaunchResult<Option<(Tensor, Tensor)>>           // (dists [N, k] f32, idxs [N, k] i32)
pub fn kmeans_assign(x: &Tensor, c: &Tensor) -> LaunchResult<Option<(Tensor, Tensor)>>         // (cluster_ids [N] i32, best_dist [N] f32)
pub fn kmeans_update(x, cluster_ids, old_centroids) -> LaunchResult<(Tensor, Tensor)>           // (new_centroids [K, D], shift [K])
```

Both stream the corpus (centroids) through the matrix core and keep a running top-K
(argmin) from the x²-free score `‖c‖² − 2⟨x, c⟩`, so the `[N, M]` distance matrix is never
formed; the host side casts to bf16, pads `D` (and `N`) to the WMMA edge, and re-adds `‖x‖²`
for exact f32 distances. `knn` takes `k` in `1..=16`. `kmeans_update` is a pure graph op —
`scatter_reduce` per cluster, empty-cluster fixup, per-cluster shift — because the
sort/scatter pattern does not tile; the caller owns the Lloyd loop. AMD-only.

---

## Using a kernel in a model

The policy — which kernel, which fallback — belongs to the model, never to the kernel. The
Qwen3 decoder in `model/src/qwen3/` is the reference integration:

```rust
// model/src/qwen3/linear.rs
pub(crate) fn linear(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    if fusable(x, w)
        && let Some(y) = svod_tk::gemm_nt(x, w).context(TkSnafu)?
    {
        return Ok(y);
    }
    Ok(x.contiguous().linear().weight(w).call()?)
}
```

```rust
// model/src/qwen3/attention.rs
if matches!(q.dtype().base(), ScalarDType::Float16 | ScalarDType::BFloat16)
    && let Some(out) =
        svod_tk::flash_attention_with(q, k, v, svod_tk::FaOpts { causal: true, key_lens: None, seg_start })
            .context(TkSnafu)?
{
    return Ok(out);
}
// else: permute to head-major and run scaled_dot_product_attention
```

Three habits to copy:

- **Gate on `fusable` first.** `Err` means a malformed request, and a 16-bit check on the
  caller's side keeps a legitimate f32 path from being reported as a bug.
- **Bridge the error.** `LaunchError` is boxed into the model's error enum
  (`#[snafu(source(from(svod_tk::LaunchError, Box::new)))]`), so a failed build is a model
  error with the kernel's context attached.
- **Shape for the kernel upstream.** `embed.rs` buckets padded sequence lengths to
  `FLASH_ATTENTION_SEQUENCE_MULTIPLE`, and `feed_forward.rs` interleaves the gate/up weight
  by `swiglu_pair_width` at load time, so the kernels apply instead of declining.

A fusion that knows a model's memory layout lives beside the model: `model/src/qwen3/tk/mod.rs`
is a QKV-norm-RoPE prologue written on the norm kernel's row vocabulary, gated by
`NORM_SUPPORTED_ARCHS`, launched through `graph_launch_multi` with three outputs. The same
`launch_custom` policy applies — it is a tk kernel in every respect except where it lives.
