---
sidebar_label: Flash Attention
---

# Worked Example: Flash Attention

Flash Attention is the kernel that justifies `tk`'s existence — the one the
[Overview](./overview) named as *not* expressible as a single schedulable reduction, the reason
a hand-authoring surface exists at all. This chapter walks through it: what makes it hard, how
the tile abstractions answer that, and where the [layout split](./wave-portability) shows up in
anger.

We're describing the forward kernel in `tk/src/kernels/fa.rs` (`build_fa_mw_rdb`), reached
through the USE-face `flash_attention(q, k, v)` and `flash_attention_with(q, k, v, opts)`. It is
built for every family in the [target table](./kernel-library#targets): CDNA3, RDNA3.5, RDNA4,
CUDA `sm_80+` and Apple7+.

---

## Why attention can't be autotuned

Plain attention is `softmax(QKᵀ) · V`. Written naively, that means: form the full `N×N` score
matrix, softmax it, multiply by `V`. The score matrix is enormous and never needs to exist all
at once — so Flash Attention streams over blocks of keys and values, maintaining the softmax
*incrementally*.

That word — incrementally — is the problem. The softmax normalization depends on the maximum
and the sum over *all* keys, but we only see one block at a time. So we keep running statistics
and fix up the result as we go. This is **online softmax**, and it's a recurrence: each KV block
reads and updates state the previous block produced.

The optimizer's action space is "tile and unroll this `REDUCE`." There is no `REDUCE` here to
tile — there's a loop whose body depends on its own previous iteration. The search can't find
it. You have to write it.

---

## The algorithm, in tiles

One workgroup is eight waves (`NUM_WARPS`), one `(head, q-block, batch)` triple — the launch
grid is `[H, N / (q_blk · 8), B]`. Each wave owns a `q_blk × D` tile of queries, held in
registers for the whole kernel; all eight share one K/V block in shared memory, filled
collaboratively. For each KV block of `kv_blk` keys the wave runs this body, all in tiles:

```text
for each block of K, V:                          ┌─ everything here is a tile op
    S   = Q · Kᵀ                                 │  mma_atb into a zeroed f32 accumulator
    S   = S · log2(e)/√D                         │  the softmax scale, on the f32 scores
    S   = mask(S)                                │  causal + key-padding + segment masks
    m'  = max(m, colmax(S))                      │  update running max  (cross-lane reduce)
    P   = exp2(S - m')                           │  rescale to the new max (base-2 exp)
    l   = l · exp2(m - m') + colsum(P)           │  update running sum
    O   = O · exp2(m - m') + P · V               │  rescale accumulator, accumulate (mma_atb)
    m   = m'                                     │
O = O / l                                        └─ final normalize, transpose, store
```

Two matrix multiplies per block (`Q·Kᵀ` and `P·V`), two cross-lane reductions (the max and
the sum), and a rescale of the output accumulator every time the running max moves. The
`exp2` — base-2 exponential — is deliberate, so the hardware's fast `exp2` unit can be used
directly. The scale `log2(e)/√D` is applied to the f32 score accumulator rather than folded
into `Q` up front: scaling `Q` would round it a second time to the 16-bit operand dtype, and
that error enters the scores relative to their own magnitude, which `exp2` then amplifies.

Each of those lines is a `Group` operation on tiles (`fa_qk` and `fa_softmax_pv` in
`tk/src/kernels/fa.rs`). The score tile is column-major `(KV, Q)`, so the softmax reduces over
its *height* with `col_reduce` into a per-query `RV`, and the rescales are the operator sugar
from [The Builder API](./builder-reference):

```rust
let max_vec_last = warp.copy(lp.reinit(max_vec_last), &max_vec);
max_vec = warp.col_reduce(max_vec.after(&max_vec_last), &att, |a, b| a.max(b), f64::NEG_INFINITY);
let scale_vec = (max_vec_last - &max_vec).exp2();
o_reg = o_reg * &scale_vec;
norm_vec = norm_vec * &scale_vec;
let att = (att - &max_vec).exp2();
norm_vec = warp.col_reduce(norm_vec.after(&scale_vec), &att, |a, b| a.add(b), 0.0);
```

No lane arithmetic in sight. One detail the recurrence forces: the running max starts at
`f32::MIN`, not `−∞`, so a block the masks hide entirely from a query row leaves
`exp2(m − m') = 1` rather than `−∞ − (−∞) = NaN`.

---

## Streaming: double-buffered KV

This is gap 2 from [Where the FLOPS Hide](./where-flops-hide) in action. While the matrix
core works on the current KV block, the next block should already be on its way into shared
memory. The kernel keeps **two** LDS halves per operand (`ker.shared_db`) and alternates by
the loop counter's parity: compute on half `kv % 2` while loading half `(kv + 1) % 2`.

```text
   load K/V block 0 --> LDS[A]
   ┌─────────────────────────────────────────────────┐
   │ compute on LDS[A]   ║   load block 1 --> LDS[B] │   <- overlap
   │ compute on LDS[B]   ║   load block 2 --> LDS[A] │
   │ ...                                             │
   └─────────────────────────────────────────────────┘
```

How the next block travels is the arch's choice, made by `Group::cp_async_fill_applies`:

- **CUDA sm_80+**: `cp.async` copies global → shared with no register staging. The loop top
  retires the previous issue (`cp_async_wait(0)` + barrier), issues block `kv+1` into the
  other half, and the gathers of the current half — one `ldmatrix.x4` per 16-bit fragment —
  run with the copy in flight.
- **AMD and Metal**: the register-staged stream. `stage_global_to_reg` issues the global loads
  of block `kv+1` into per-lane registers before the MMAs, `commit_regs_to_local` writes them
  into the other half after them, and one `war_fence2` barrier per trip — consumed by the
  gathers, carrying the commit as a dependency — covers both the read-after-write on the half
  just written and the write-after-read on the half every wave just finished gathering.

The prefetch index wraps modulo the block count, so the last trip re-reads block 0 (never
gathered) instead of running off the operand. The KV loop itself is a `Loop` with a *dynamic*
bound: under `causal` each q-block visits only `(block_q_base + 1) · 8 · q_blk / kv_blk`
super-blocks, the causal block-skip.

---

## The layout wrinkle: relayout between the two matmuls

Here's where [Layouts and Wave Sizes](./wave-portability) stops being abstract. The kernel does
two matrix multiplies, and the output of the first (`S = Q·Kᵀ`, after softmax becomes `P`) is the
*input* of the second (`P·V`). Can the score accumulator be fed straight back in as an operand?

- **On CDNA, RDNA4, CUDA and Metal** (`acc_reusable_as_input() == true`): yes. The accumulator
  and the operand share a lane map — the MFMA fragment on CDNA, the one gfx12 fragment, the
  `m16n8` C fragments in A-operand register order on CUDA, one `simdgroup_matrix` map on Apple —
  so `att_mma` is a register `copy` with the f32 → 16-bit cast.
- **On RDNA3** (`acc_reusable_as_input() == false`): no. The even/odd accumulator and the
  replicated operand differ, so `P` makes a **round-trip through LDS**: a `store_local_fenced`
  of the accumulator into this wave's band of a per-workgroup `[8 · kv_blk, q_blk]` scratch
  tile (`att_smem`), then a `load` back under the operand map. `FaPolicy::att_band` reports
  it, so the shared-memory budget counts the band.

The kernel branches on `ker.caps.acc_reusable_as_input()` once, when allocating `att_smem`;
the hot loop reads `Option<ST>`. Same algorithm, two physical realizations — exactly the
portability tax the previous chapter described, here in the hottest loop of the most
important kernel.

---

## Masking

Causal masking (a query can't attend to a future key), key-padding (`key_lens[b]`, ignore keys
at or past the valid count) and packed-sequence segments (`seg_start[b, q]`, ignore keys before
the query's own segment) are applied to the score tile `S` before the softmax. All three are
`mask_where` calls: the position of each score element is implied by which fragment and lane
holds it, so the mask is computed from the tile's own `LaneMap`, not fetched. The optional
mask tensors are trailing globals bound after `o, q, k, v` with plain `ker.gl` calls, so the
unmasked ABI is unchanged.

---

## The per-warp tile

`FaConfig { q_blk, kv_blk, unroll, causal }` is the body's tuning knob. The baseline is
`{16, 32}` — a 16-row Q tile and a 32-key KV super-block, which raises per-wave MMA ILP and
halves the softmax bookkeeping against `{16, 16}`. `FaPolicy` (`tk/src/kernels/fa.rs`) chooses
a taller tile per family once the launch grid covers the device's compute units and the head
dim stays under the family's bound, and declines a head dim whose buffers would exceed shared
memory (64 KiB on AMD, 48 KiB static on CUDA, 32 KiB on Apple). On a first launch the four
`FA_TILES` candidates are timed and the winner cached — [Autotuning](./tuning).

`unroll` is on for CUDA only: the NVPTX backend keeps the accumulators in registers only when
every register index is a constant, so the body is emitted flat (`Kernel::set_unroll`); AMD
keeps the rolled form.

:::tip[For GPU experts]
The compute/memory overlap isn't hand-emitted as raw scheduling intrinsics in `tk` the way it is
in HipKittens' kernels. Instead the KV loop is annotated with
`sched::pipeline(SchedKind::Attention, kv_idx)` (`tk/src/kernels/fa.rs`), a marker threaded
through the in-loop K/V buffers that the post-linearization pass in `codegen/src/llvm/sched.rs`
consumes. The body expresses *what* to overlap; the pass decides the instruction ordering. Today
it lowers the marker to `@llvm.amdgcn.iglp.opt(0)` on CDNA — the backend's canned MFMA/memory
interleave — and leaves it an inert comment elsewhere; a softmax-aware comb that weaves the
exponential work under the matrix ops is the planned replacement for the attention kind.
:::

---

## Why this matters

Flash Attention is the whole section condensed into one file:

- it exists because **online softmax is a recurrence**, not a tileable reduction
  ([Overview](./overview));
- it lives or dies on **streaming and overlap** ([Where the FLOPS Hide](./where-flops-hide));
- it's expressed entirely in **tiles and roles**, never lane indices ([What Tiling Is](./tiling));
- it compiles to **the same UOp IR** as everything else and joins the lazy graph as an
  `Op::Call` ([Authoring into the IR](./lowering));
- and it carries an explicit **accumulator-reuse branch** in its hot loop, one per fragment layout
  ([Layouts and Wave Sizes](./wave-portability)).

That's why it's hand-written, and why `tk` exists to write it. To run it in isolation and check
its numbers, see [Debugging](./debugging).
