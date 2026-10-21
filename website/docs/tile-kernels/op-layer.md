---
sidebar_label: Op Layer
---

# Op Layer

`svod_tk::ops` is what models call. Every op returns a lazy `Tensor`. When the device, dtypes
and shapes fit a kernel, a tk kernel computes it. Otherwise the op builds the equivalent graph
itself. A model never checks whether a kernel applies and never pads to a tile.

```rust
pub fn linear(x: &Tensor, w: &Tensor, opts: Linear) -> Result<Tensor>;
pub fn conv2d(x: &Tensor, w: &Tensor, opts: Conv) -> Result<Tensor>;
pub fn attention(q: &Tensor, k: &Tensor, v: &Tensor, opts: Attn) -> Result<Tensor>;
pub fn attention_packed(qkv: &Tensor, heads: usize, kv_heads: usize, opts: Attn) -> Result<Tensor>;
pub fn heads(qkv: &Tensor, opts: Qkv) -> Result<(Tensor, Tensor, Tensor)>;
pub fn layer_norm(x: &Tensor, w: &Tensor, b: Option<&Tensor>, eps: f64) -> Result<Tensor>;
pub fn add_layer_norm(x: &Tensor, residual: &Tensor, w: &Tensor, b: Option<&Tensor>, eps: f64)
    -> Result<(Tensor, Tensor)>;
pub fn rms_norm(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor>;
pub fn add_rms_norm(x: &Tensor, residual: &Tensor, w: &Tensor, eps: f64) -> Result<(Tensor, Tensor)>;
pub fn supported(device: &DeviceSpec) -> bool;
```

| Op | Shapes | Options |
|---|---|---|
| `attention` | `q [B, T, H, D]`, `k`/`v [B, Tk, H_kv, D]` → `[B, T, H, D]` | `Attn { causal, keys, window, chunk, seg_start, cache, splits, scale, bias }` |
| `attention_packed` | `qkv [B, T, (H + 2·H_kv)·D]` → `[B, T, H, D]` | `Attn` without a cache; `heads`, `kv_heads` |
| `linear` | `x [lead..., K]`, `w [N, K]` → `[lead..., N]` | `Linear { bias, act, gated, residual, scale, rope }`; gated `w` is `[2N, K]` |
| `conv2d` | `x [B, H, W, Cin]`, `w [Cout, kh, kw, Cin / groups]` → `[B, Ho, Wo, Cout]` | `Conv { stride, pad, dilation, groups, bias, act, residual, scale, out_dtype }`; a 1×1 is `linear` |
| `heads` | `qkv [B, T, (H + 2·H_kv)·D]` → `q`, `k`, `v` | `Qkv { heads, kv_heads, head_dim, q_norm, k_norm, eps, rope }` |
| `layer_norm`, `rms_norm` | `x [..., D]`, `w`/`b [D]` | `add_*` take `residual` like `x` and return `(x + residual, norm)` |

`Linear::scale` multiplies the activated value before the residual add:
`scale·act(x·wᵀ + bias) + residual`, so a Conformer half-step `x + 0.5·ffn(x)` is one GEMM.

`Linear::rope` is `Rotary { cos, sin, heads, head_dim }`: the output's first `heads` heads of
`head_dim` columns are rotated after the bias by `(cos, sin)` tables of `[1, T, 1, head_dim/2]`,
the head's two halves paired as `apply_rotary_emb` pairs them, for `x [B, T, K]`. Beside an
activation, a residual or a scale the whole linear runs on the graph. Otherwise the GEMM rotates
them in its store when the tables share `x`'s dtype and a tile holds whole heads per warp, and if
not, a plain GEMM runs and the graph rotates its output. `attention_packed` then reads such
a projection's queries, keys and values in place, where `heads` would copy them out.

`Attn::keys` is `KeyMask::None`, `KeyMask::Lens(&lens)` (`[B]` valid key counts) or
`KeyMask::Bool(&mask)` (`[B, Tk]`, true where attended). `Attn::cache` takes
`Cache { head_start, kv_heads, row_map, appended, app_lens }`, and `appended` requires
`KeyMask::Lens`. `Attn::chunk` is block attention (`Chunk { size, left, right }`).
`Attn::scale` defaults to `1/√D`. `Attn::bias` (`[B, H, T, Tk]` or `[1, H, T, Tk]`, the stream
dtype) is added to the scaled scores before the masks, as in WavLM's relative position bias.
`Qkv::rope` is `(cos, sin)` of `[1, T, 1, D/2]` (by position) or `[B, T, 1, D/2]` (by token).
The masks are described with the [attention kernel](./kernel-library#flash-attention).

## Kernel or graph

The decision is a pure function in `ops::shape`, `fn(planner, dtypes, extents, …) -> Plan<Cfg>`,
so it is tested on the host without a GPU (`test/unit/ops_plan.rs`). `Plan::Kernel(configs)`
holds the ranked configs; the [tuner](./tuning) measures the first that compile without spilling
(or, when every one spills, the first that compile: spills never send an op to the graph).
`Plan::Graph(Fallback)` says why the graph runs, and the planner log names it
(`RUST_LOG=svod_tk::plan=debug`).

| `Fallback` | When |
|---|---|
| `Target` | The tensor is not on the default device, the device has no tk kernels (`config::has_kernels`: CUDA below sm_80, Metal, the CPU), or its target reports no SM count |
| `Dtype` | Any operand is f32, or the operands do not all share one 16-bit type with a matrix core. `conv2d`: an output type other than the input's or f32, or a residual not in it |
| `Symbolic` | A dim other than a bound leading one is symbolic (for `linear`, also a symbolic `N`) |
| `Shape` | `linear`: `N` not a multiple of 8 or no rows. `conv2d`: `groups > 1`, `Cin` not a multiple of 16, `Cout` not a multiple of 8, or an empty output. `attention`: no config fits `D` (on sm_86, 32, 48, 64, 128 and 256 plan and 40 does not) or an empty dim. `heads`: `D` not a power of two in 16..=256. Norms: `D` not a multiple of 256 in 256..=2048 |
| `Config` | No tile config fits. For `linear`, `K` is not a multiple of 16 |

`attention` also takes the graph when appended rows are wider than every config's key block.

:::note[Why f32 keeps the graph]
The kernels take 16-bit operands only. Casting an f32 model down would trade about three
decimal digits of precision for speed, so the op layer leaves that choice to the model's dtype.
:::

## Batch variables

Dim 0 of an operand may be symbolic when it is bound to a runtime variable. That variable is
either the JIT's `batch_var` or a `DefineVar`/bounded `Param` with min and max. The kernel then
launches over the live count in grid z, and its buffers hold the maximum. Outputs are allocated
at capacity with the symbolic dim kept in their shape (`Tensor::empty_dynamic_spec`). The next kernel
therefore binds the realized buffer itself, and no copy is made to shrink it. Any other symbolic
dim is `Fallback::Symbolic`.

## Errors

`Err` is reserved for what the graph op would also reject, plus a failed kernel build:

| `ops::Error` | Meaning |
|---|---|
| `Shape { op, operand, got, expected }` | An operand's shape does not fit the op |
| `Dtype { op, operand, got, want }` | `w` (or `k`, `v`, norm weights, rope tables) differs from the input's dtype |
| `Heads { op, heads, kv_heads }` | `heads` is not a multiple of `kv_heads` |
| `Graph { op, source }` | Building the graph fallback failed |
| `Launch { op, source }` | Lowering or binding the kernel failed |

## In a model

Nemotron-3-Diarization's self-attention, from `model/src/nemotron_diar/model.rs`: one stacked
QKV projection, the fused prologue with RoPE, attention under key lengths, and the output
projection with the residual in its epilogue.

```rust
fn forward(&self, x: &Tensor, rope: &(Tensor, Tensor), key_lens: &Tensor, residual: &Tensor) -> Result<Tensor> {
    let (b, s, d) = (x.dim(0)?, x.dim(1)?, x.dim_const(2)?);
    let qkv = ops::linear(x, &self.qkv_weight, ops::Linear::default())?;
    let (cos, sin) = rope;
    let split = Qkv {
        heads: self.num_heads,
        kv_heads: self.num_heads,
        head_dim: d / self.num_heads,
        q_norm: None,
        k_norm: None,
        eps: 0.0,
        rope: Some((cos, sin)),
    };
    let (q, k, v) = ops::heads(&qkv, split)?;
    let opts = Attn { keys: KeyMask::Lens(key_lens), ..Attn::default() };
    let out = ops::attention(&q, &k, &v, opts)?;
    project(&self.o_proj, &out.try_reshape([b, s, SInt::Const(d)])?, Act::None, Some(residual))
}
```

`project` is `ops::linear` with the layer's bias, an activation and an optional residual. The
same file runs its LayerNorms through `ops::layer_norm`.
