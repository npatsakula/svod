---
sidebar_label: Layouts and Wave Sizes
---

# Keeping One Kernel Correct on Five Layouts

Here is a bug NVIDIA hardware cannot give you. You write a tile kernel, test it on a CDNA
datacenter GPU, and it's perfect. You run the *same* kernel on an RDNA laptop APU and the
numbers are garbage — no crash, no error, just wrong. Nothing in the code looks different.
CUDA is spared this particular trap — a warp is 32 lanes everywhere — but not the reason
behind it: the fragment layout still differs, so the same indirection carries the kernel
onto NVIDIA, and onto Apple.

[What Tiling Is](./tiling) introduced fragments and role-based selection; this chapter explains
why that indirection has to exist. The culprit is the **wavefront size** and, underneath it, the
**lane map** of a fragment; dealing with both cleanly is what separates a tile library that
works on one chip from one that's actually portable.

---

## The 32-vs-64 split

A wavefront (NVIDIA's "warp", Apple's "SIMD group") is the group of lanes that execute in
lockstep. On AMD there are two sizes, and Svod targets both, plus NVIDIA's and Apple's:

| Family | Parts | Matrix op | Wavefront | Fragment | Per lane |
|---|---|---|---|---|---|
| **CDNA3** | gfx942 | MFMA | 64 | 16×16, `Strided { stride: 4 }` for every role | 4 |
| **RDNA3** | gfx1151 | gfx11 WMMA | 32 | 16×16, `Interleaved` accumulator, `Strided { stride: 0 }` operand replicated across the wave halves | 8 acc / 16 operand |
| **RDNA4** | gfx1200, gfx1201 | gfx12 WMMA | 32 | 16×16, `Strided { stride: 8 }` for every role | 8 |
| **CUDA** | sm_80+ | `mma.sync.m16n8k16` | 32 | 16×16 as two `m16n8` halves, `MmaSync` for every role | 8 |
| **Metal** | Apple7+ | `simdgroup_matrix<T, 8, 8>` | 32 | 8×8, `SimdgroupMatrix` (B and accumulator), `SimdgroupMatrixT` (A) | 2 |

(The table is the set of layouts the DSL resolves — `ArchCaps::frag` in `tk/src/arch.rs` and
the constants in `tk/src/tiles.rs`. An individual kernel declares its own `ArchSet` on top of
it; [The Kernel Library](./kernel-library) has the per-kernel matrix.)

That single number ripples through everything. A `16×16` tile has 256 elements. Spread across
64 lanes, that's 4 elements per lane; across 32 lanes, it's 8 — except on RDNA3, where the
operand is replicated and each lane holds 16. Different lanes own different elements. So:

- the **register layout** of a tile differs,
- the **operand layout** the matrix instruction expects differs,
- and any **cross-lane reduction** — the heart of softmax and layernorm — has a different
  number of steps and a different sibling pattern.

A kernel that hardcodes "there are 64 lanes, reduce by gathering lanes 16, 32, 48" computes a
*partial* reduction on a 32-lane machine and silently returns wrong values.

---

## The fix: ask for a role, not a shape

`tk`'s answer is a layer of indirection. A kernel never writes down a concrete fragment shape
like "16×16, 4 elements per lane." Instead it asks for a **role**, and lets the architecture
capabilities resolve it:

```text
   kernel says:  "I need an accumulator fragment"   (FragRole::Accumulator)
                          │
                          ▼
   ArchCaps::frag(role)   ── on CDNA  ──▶  RT_16X16          (wave64, 4/lane)
                          ├─ on gfx11 ──▶  RT_16X16_W32_ACC  (even/odd rows, 8/lane)
                          ├─ on gfx12 ──▶  RT_16X16_GFX12    (strided, 8/lane)
                          ├─ on CUDA  ──▶  RT_16X16_MMA      (two m16n8 halves, 8/lane)
                          └─ on Metal ──▶  RT_8X8_SIMD       (2/lane)
```

The roles are `FragRole::{Accumulator, Operand, OperandB, AccumulatorT}` and the resolver is
`ArchCaps::frag(role)` (kernels reach it as `ker.frag(role)`, or through the shortcuts
`ker.acc` / `ker.operand` / `ker.operand_b` / `ker.acc_t`). The kernel author writes
"accumulator" and "operand"; the *physical* layout — element count per lane, the interleave
map, replication — is filled in for the target. CDNA and gfx12 resolve every role to one
shape; gfx11 splits the accumulator, its transpose and the replicated operand; CUDA resolves
every role to the two-half map; Metal gives the A operand a flipped map because its core
computes `D = A·B` straight off the lane map and tk's accumulators are column-major. `None`
where tk has no table at all — pre-Ampere CUDA, pre-Apple7 Metal — so a matrix-core kernel
fails loudly at `ker.frag` instead of rendering a wrong layout. Write once, run on all five.

This is the same lesson HipKittens learned (see [tk vs HipKittens vs CuTile](./comparison)): its
tile types are keyed off a single compile-time `WARP_THREADS` constant (`64` in the CDNA build), so
a different wave width means a different build of the library. `tk` collapses that into one
runtime-resolved `ArchCaps`.

---

## A bug this actually caught

The reason this indirection exists isn't theoretical. An early `tk` cross-lane all-reduce —
the `shuffle_xor` primitive used to sum a value across a wave — was written with a hardcoded
wave64 reduction tree. On RDNA's 32-lane waves it reduced over lanes that don't participate,
producing wrong sums for exactly the softmax-style reductions attention depends on. The fix was
to drive the reduction off the resolved fragment instead of a constant. The shuffle primitives
in `tk/src/group/shuffle.rs` read `caps.wave_size`; the reductions read the fragment's
`LaneMap`; the bug class is designed out.

:::tip[For GPU experts]
Two things carry most of the layout-specific weight, both read off the resolved fragment's
`LaneMap` (`tk/src/layout.rs`), never off a constant:

- **The reduce tree.** `LaneMap::tree(wave_size)` is the cross-lane completion of a fragment
  reduce after the in-lane fold. The AMD maps gather the *original* partial of every sibling
  lane-group with `ds_bpermute` — offsets `[16, 32, 48]` at wave64, `[16]` at wave32; `MmaSync`
  butterflies the *running* value with `shfl.bfly` over masks `[1, 2]` (a lane's eight elements
  span two rows, so it keeps `LaneMap::slots() == 2` values and the fold finishes in the
  4-lane quad); `SimdgroupMatrix` butterflies over `[1, 8]`. `tk/src/group/reduce.rs` walks
  whichever tree it is handed.
- **`acc_reusable_as_input()`** answers: "can a matrix accumulator be fed straight back in as
  an operand to the next multiply?" True on CDNA (MFMA accumulator and input share `RT_16X16`),
  on gfx12 (one fragment for every role), on CUDA (the two-half f32 accumulator holds the
  `m16n8` C fragments in exactly the A-operand register order) and on Metal (B and accumulator
  share one map). False only on gfx11, where the even/odd `<8×f32>` accumulator and the
  replicated `<16×in>` operand differ, so the value makes a round-trip through LDS to be relaid
  out. [Flash Attention](./flash-attention) branches on it between its two matmuls.

The map is written once against a `LaneArith` trait and evaluated twice: over `Index`-typed
UOps when building a kernel and over plain integers in `tk/src/test/unit/layout.rs`, where
every variant is proved a bijection and the Apple map is pinned to a table measured on
hardware. `LaneMap::ldmatrix_x4` derives the CUDA `ldmatrix` register plan from the same
closed form rather than hand-permuting it. The `ept` field on `BaseShape` (from
[What Tiling Is](./tiling)) exists for the same reason: on gfx11, operands are replicated
across lanes, so elements-per-thread isn't `element_count / wave_size` and must be stored
explicitly.
:::

---

## Why this matters

Portability across wave sizes and fragment layouts is the tax hand-written kernels pay, and it's
why a naive port of an NVIDIA tile library onto AMD doesn't just work. `tk` pays the tax once, in
the `ArchCaps` and `LaneMap` abstractions, so individual kernels stay readable: they speak in
*roles* and let the hardware table sort out the lanes. That the same abstraction then carries a
kernel *onto* NVIDIA and Apple — where the warp is always 32 but the fragment layout is the
core's own — is the payoff for having built it. [Flash Attention](./flash-attention) is where you
see this pay off in a real kernel.
