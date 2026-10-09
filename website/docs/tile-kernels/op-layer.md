---
sidebar_label: Op Layer
---

# Op Layer

`svod_tk3::ops` is what models call. Every op returns a lazy `Tensor`. When the device, dtypes
and shapes fit a kernel, a tk3 kernel computes it. Otherwise the op builds the equivalent graph
itself. A model never checks whether a kernel applies and never pads to a tile.

```rust
pub fn linear(x: &Tensor, w: &Tensor, opts: Linear) -> Result<Tensor>;
pub fn attention(q: &Tensor, k: &Tensor, v: &Tensor, opts: Attn) -> Result<Tensor>;
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
| `attention` | `q [B, T, H, D]`, `k`/`v [B, Tk, H_kv, D]` → `[B, T, H, D]` | `Attn { causal, keys, window, seg_start, cache, splits, scale, bias }` |
| `linear` | `x [lead..., K]`, `w [N, K]` → `[lead..., N]` | `Linear { bias, act, gated, residual, scale }`; gated `w` is `[2N, K]` |
| `heads` | `qkv [B, T, (H + 2·H_kv)·D]` → `q`, `k`, `v` | `Qkv { heads, kv_heads, head_dim, q_norm, k_norm, eps, rope }` |
| `layer_norm`, `rms_norm` | `x [..., D]`, `w`/`b [D]` | `add_*` take `residual` like `x` and return `(x + residual, norm)` |

`Linear::scale` multiplies the activated value before the residual add: `scale·act(x·wᵀ + bias) + residual`, so a Conformer half-step `x + 0.5·ffn(x)` is one GEMM.

`Attn::keys` is `KeyMask::None`, `KeyMask::Lens(&lens)` (`[B]` valid key counts) or
`KeyMask::Bool(&mask)` (`[B, Tk]`, true where attended). `Attn::cache` takes
`Cache { head_start, kv_heads, row_map, appended }`, and `appended` requires `KeyMask::Lens`.
`Attn::scale` defaults to `1/√D`. `Attn::bias` (`[B, H, T, Tk]` or `[1, H, T, Tk]`, the stream dtype) is added to the scaled scores before the masks, as in WavLM's relative position bias. `Qkv::rope` is `(cos, sin)` of `[1, T, 1, D/2]` (by position)
or `[B, T, 1, D/2]` (by token). The masks are described with the
[attention kernel](./kernel-library#flash-attention).

## Kernel or graph

The decision is a pure function in `ops::shape`, `fn(target, dtypes, extents, …) -> Plan<Cfg>`,
so it is tested on the host without a GPU. `Plan::Kernel(candidates)` lists the configs with the
untuned pick first. `Plan::Graph(Fallback)` says why the graph runs.

| `Fallback` | When |
|---|---|
| `Target` | The tensor is not on the default device, or the device has no tk3 tables (today: anything but CUDA sm_80+) |
| `Dtype` | Any operand is f32, or the operands do not all share one 16-bit type with a matrix core |
| `Symbolic` | A dim other than a bound leading one is symbolic (for `linear`, also a symbolic `N`) |
| `Shape` | `linear`: `N` not a multiple of 8 or no rows. `attention`: `D ∉ {48, 64, 128}` or an empty dim. `heads`: `D` not a power of two in 16..=256. Norms: `D` not a power of two in 256..=2048 |
| `Config` | No tile config fits. For `linear`, `K` is not a multiple of 16 |

Real cases from `test/unit/ops_plan.rs`, on the sm_86 target:

```rust
#[test_case(&[4096, 4096], 4096, false, Ok(BIG); "large grid, deepest ring")]
#[test_case(&[8, 37, 512], 512, false, Ok(SMALL); "medium grid")]
#[test_case(&[37, 40], 96, false, Err(Fallback::Config); "k off every bk")]
#[test_case(&[37, 64], 100, false, Err(Fallback::Shape); "n not a multiple of 8")]
fn linear_plans(x: &[usize], n: usize, gated: bool, want: Result<GemmCfg, Fallback>) {
    assert_eq!(first(shape::linear(Some(&sm86()), &[BF16, BF16], Some(&ext(x)), n, gated)), want);
}
```

:::note[Why f32 keeps the graph]
The kernels take 16-bit operands only. Casting an f32 model down would trade about three
decimal digits of precision for speed, so the op layer leaves that choice to the model's dtype.
:::

## Batch variables

Dim 0 of an operand may be symbolic when it is bound to a runtime variable. That variable is
either the JIT's `batch_var` or a `DefineVar`/bounded `Param` with min and max. The kernel then
launches over the live count in grid z, and its buffers hold the maximum. Outputs are allocated
at capacity with the symbolic dim kept in their shape (`Tensor::empty_dynamic`). The next kernel
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
