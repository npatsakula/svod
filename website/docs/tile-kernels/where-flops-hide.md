---
sidebar_label: Where the FLOPS Hide
---

# Where the FLOPS Hide

A modern AMD matrix core advertises a number that sounds like a promise. The datasheet says
*thousands* of TFLOPS. You write the obvious matrix multiply — three nested loops, a multiply,
an add — and measure. You get a few percent of the number on the box.

The FLOPS didn't vanish. They're hiding — and finding them is what *any* fast matrix kernel has
to do, whether the compiler generates it (see the [Overview](./overview)) or you write it by hand
with a tile library like `tk` (inspired by [HipKittens](https://github.com/HazyResearch/HipKittens)).

This chapter is the "why" behind everything else in this section: it explains what a matrix
core actually needs to run flat-out, and the handful of bottlenecks that stand in the way.

---

## The math is not the bottleneck

Here's the counter-intuitive part. The matrix multiply *instruction* — AMD calls it MFMA on
CDNA, WMMA on RDNA — is extraordinarily fast. A single instruction multiplies two small tiles
and accumulates the result. If you could issue these back-to-back forever, you'd hit the
datasheet number.

You can't, because between every two matrix instructions, the hardware has to:

- compute the *addresses* of the next tiles,
- fetch them from memory that is far slower than the math unit,
- get them into registers in *exactly* the layout the matrix core expects,
- and do all of this without the math unit sitting idle waiting.

> **Roofline intuition.** Every kernel is limited either by how fast it can compute
> (compute-bound) or by how fast it can move data (memory-bound). A naive matmul is
> memory-bound: it spends its time waiting on loads, and the expensive matrix core is starved.
> The goal of tiling is to make a kernel compute-bound — to keep the math unit fed.

So the question "where do the FLOPS go?" really means: **what keeps the matrix instructions
from issuing back-to-back?** There are five recurring answers.

---

## The five gaps

This framing follows HazyResearch's work on ThunderKittens and HipKittens (see the
[HipKittens paper, arXiv:2511.08083](https://arxiv.org/abs/2511.08083)). Their finding,
porting the ideas from NVIDIA to AMD, is that the *tile* and *compute* abstractions carry
over directly — but the decisions around **memory, scheduling, and chip layout** are where the
performance actually lives.

### 1. Layout and address computation

The matrix core wants its inputs in a specific register layout — a particular lane owns a
particular element. If your data arrives in the wrong layout, you pay to shuffle it before
every multiply. And computing *which* address to load from — turning a tile coordinate into a
byte offset into a swizzled buffer — is itself arithmetic that competes with the math.

The fix: size tiles to the matrix-core fragment so data lands already in MMA layout, and
*precompute* the swizzled offsets once instead of recomputing them every iteration.

### 2. Memory latency — and AMD has no `cp.async`

On NVIDIA, asynchronous copy instructions (`cp.async`, and later TMA) let you start a load and
keep computing while it lands — `tk`'s CUDA path uses `cp.async` for exactly this. AMD GPUs don't
have those. Instead, the hardware offers a
**buffer load directly into shared memory (LDS)** that bypasses registers entirely. A fast
kernel streams the *next* block of data into LDS while the matrix core chews on the *current*
one. Get this wrong and the math unit stalls on every load.

### 3. Bank conflicts in shared memory

Shared memory is split into banks. If two lanes in a wave hit the same bank in the same cycle,
the accesses *serialize* — you've turned one memory transaction into many. HipKittens
reverse-engineered the CDNA LDS structure empirically: **64 banks, accessed in two phases of
32 lanes each.** The fix is a carefully chosen XOR *swizzle* of the in-LDS layout so that the
lanes of a wave always spread across distinct banks. `tk` ports these swizzles directly.

### 4. Overlapping compute and memory

NVIDIA hides latency with high *occupancy* — many warps resident, so when one stalls another
runs. AMD matrix-core kernels generally can't lean on that, so they overlap *explicitly*, by
interleaving instruction streams. Two patterns recur:

- **8-wave ping-pong** — a producer/consumer split where some waves only move memory and
  others only compute, handing off through LDS.
- **4-wave interleave** — finer-grained interleaving of matrix instructions against the vector
  ALU and the exponential unit.

Which one wins is *workload-dependent*, not a constant.

### 5. Chiplet thread-block ordering

A datacenter AMD GPU is several chiplets (XCDs), each with its own slice of L2 cache. If two
workgroups that touch the same data land on *different* chiplets, they can't share cache. By
remapping which workgroup ID runs where, you keep cooperating blocks on the same chiplet and
recover real performance for free.

---

## The arch angle: MFMA vs WMMA vs `mma.sync`, wave32 vs wave64

A few hardware facts shape every tile kernel `tk` builds, and they're worth holding onto:

- **CDNA** (datacenter, gfx942) issues matrix multiplies via **MFMA** instructions and
  runs **wave64** — 64 lanes per wavefront.
- **RDNA** (gfx1151 on RDNA3.5, gfx1200/gfx1201 on RDNA4) issues **WMMA** instructions and
  runs **wave32** — 32 lanes. RDNA3 and RDNA4 do not even agree with each other: gfx11
  replicates operands across the two wave halves and interleaves the accumulator's rows,
  gfx12 drops both.
- **NVIDIA** (`sm_80+`) issues **`mma.sync`** and runs a **warp32** — 32 lanes, but a fragment
  layout of its own again: a 16×16 tile held as two `m16n8` halves.
- **Apple** (Apple7+) runs `simdgroup_matrix` over a 32-lane SIMD group on an 8×8 fragment,
  a quarter of everyone else's.

The lane count changes how a tile's elements are distributed across the wave, which changes the
register layout, which changes the reductions — and even at the same width the fragment layout
differs. A kernel written for one and run on another — without accounting for this — is silently
wrong. Keeping a single kernel correct on all of them is its own chapter:
[Layouts and Wave Sizes](./wave-portability).

:::tip[For GPU experts]
HipKittens' `analysis/paper_experiments/` micro-benchmarks quantify the gaps above. They justify
the design:

| Gap | Finding |
|-----|---------|
| gap 3 (bank structure) | The LDS micro-benchmark confirms 64 banks accessed in two phases of 32 lanes on CDNA — the structure the XOR swizzles are tuned against. |
| gap 4 (overlap not universal) | A BF16 GEMM peaks with 8-wave ping-pong; an FP8 GEMM peaks with 4-wave interleave. The optimal wave-overlap strategy varies by dtype. |
| gap 5 (chiplet swizzle) | Remapping workgroup IDs for XCD locality yields a measurable large-GEMM speedup. |

`tk` implements these levers directly: the XOR swizzles live in `tk/src/swizzle.rs` (ported from
HipKittens' shared-tile layouts, plus the 16-byte-chunk `Sw16x16Mma` that `ldmatrix`, `cp.async`
and the gfx12 gather need), the L2/chiplet remap in `tk/src/grid.rs` (`l2_swizzle`, on in the CUDA
GEMM tiles, the widest RDNA4 tile and the gfx942 matmul config), and the compute/memory overlap is expressed as a
`sched::pipeline(SchedKind::Attention, …)` marker on the Flash Attention KV loop that a
post-linearization pass in `codegen/src/llvm/sched.rs` consumes. Today that pass lowers both
`SchedKind`s to `@llvm.amdgcn.iglp.opt(0)` on CDNA — delegating the MFMA/memory interleave to
the AMDGPU machine scheduler — and is a no-op elsewhere; the measured result on gfx942 was that
hand-placed `sched.barrier` fences *regress* a dataflow-scheduled GEMM to 0.6–0.9× of iglp alone,
because they pin the very load/MFMA overlap the double buffer exists to create.

Where a fence does pay, the AUTHOR face exposes the raw controls as typed `Op::Custom` nodes in
`tk/src/asm.rs` — `s_setprio`, `s_waitcnt_lgkmcnt`, `sched_barrier`, `iglp_opt`, each threaded on
a dependency so it lands where the author put it. The one in-tree use is the RDNA4 GEMM:
`ArchCaps::needs_pipeline_commit_fence()` names the arch whose scheduler hoists the pipeline's
whole LDS commit above the trip's MMAs, and `gemm_core` answers with `sched_barrier(0, after_mma)`
so the prefetch stays in flight under the matrix work.
:::

---

## Why this matters

Everything in the rest of this section is a response to one of these five gaps:

- [What Tiling Is](./tiling), the next chapter, answers gaps 1–3: it puts data in the right
  layout, in the right memory, conflict-free.
- [Flash Attention](./flash-attention) shows gaps 2 and 4 in action: double-buffered streaming
  and an explicit pipeline.
- [Layouts and Wave Sizes](./wave-portability) is the portability tax that gap 1, the lane-count
  difference, and the per-arch fragment layouts impose.

The headline: a fast GPU kernel is not "the math, written down." It is *the math, plus an
answer to where every cycle between two matrix instructions goes.* That's where the FLOPS hide.

:::note[Doesn't the compiler already use matrix cores?]
Yes — and nothing here says otherwise. For graph-native kernels, BEAM's `TC` action maps a matmul
straight onto WMMA/MFMA and tiles it against these same gaps; the compiler is perfectly capable of
driving the matrix cores. The five gaps are the *hardware reality* every fast kernel must beat —
compiler-generated and hand-written alike — not a hole in the compiler.

So `tk` is not a competing code path. It is the **instrument for the kernels BEAM can't express**
([Overview](./overview)), and it earns its place by adding *no extra complexity*: a `tk`
kernel emits the same UOp IR the compiler already produces — no second backend, no separate
toolchain, no new debugger. You reach for it only to write an implementation yourself, and even
then you stay inside the one compiler.
:::
