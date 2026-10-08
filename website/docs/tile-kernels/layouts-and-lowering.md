---
sidebar_label: Layouts and Lowering
---

# Layouts and Lowering

`lower::lower(program, &lowering, params, device)` turns a tile program into a Svod program
whose instruction list is already in order. A `Lowering` holds everything the author did not
decide: the `Target`, the `Schedule`, the `WarpGrid` over the block tile, and whether shared rows
are swizzled. Each kernel config builds its own (`GemmCfg::lowering`, `FaCfg::lowering`,
`NormCfg::lowering`).

| Step | Module | Effect |
|---|---|---|
| 1. Schedule expansion | `schedule::expand` | Every `Pipeline` becomes loops, branches and explicit `Sync` statements |
| 2. Operand materialization | `lower` | A shared or global `mma` operand gets an explicit load into a fresh register tile |
| 3. Barriers | `lower::sync` | Barriers are inserted around author-level shared memory traffic |
| 4. Layout inference | `layouts::infer` | Every register tile gets a layout; conflicts get a `Relayout` |
| 5. Emission | `lower/emit.rs` | Statements are emitted in program order into `UOp::linear_program` |

## F2 linear layouts

A `layout::Layout` is a linear map over GF(2) from the bits of input dimensions to the bits of
output dimensions (the "Linear Layouts" formulation of Triton). The dimensions are `Dim::{Reg,
Lane, Warp, Block, Row, Col}`. The map stores one basis vector per input bit. Every vendor
fragment, every XOR swizzle and every `ldmatrix` plan is such a map. This is the PTX
`mma.m16n8k16` accumulator, from `layout/atoms.rs`:

```rust
/// PTX `mma.m16n8k16` C/D fragment (16×8, M×N): `row = g + 8·(c/2), col = 2t + c%2`.
pub fn mma_sync_c() -> Layout {
    Layout::from_bases(
        [(Row, 16), (Col, 8)],
        &[(Reg, &[[0, 1], [8, 0]]), (Lane, &[[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]])],
    )
}
```

The algebra has `compose`, `inverse`/`pseudo_inverse`, `product` (tile one layout over
another), `transpose`, `sublayout` and `slice`. The tests in `test/unit/layout.rs` check every
atom exhaustively against the closed form it encodes.

A register tile's layout is a `layouts::TileLayout { frag, reps, warps }`. It holds a fragment
over `(Reg, Lane)`, repeated `reps` times inside each warp's sub-tile, with the warps' sub-tile
coordinates.

## Atoms

An `atoms::MmaAtom` is one matrix-core instruction with the layouts of its `a`, `b` and `c`
operands. `Target::mma(dtype_in, dtype_out)` finds it. A kernel never names one: the
inference reads the operand layouts from the atom that consumes the values.

| Target | Atom (bf16/f16 → f32) | Fragment layouts |
|---|---|---|
| CUDA | `mma.sync` m16n8k16 | `mma_sync_a`, `mma_sync_b`, `mma_sync_c` |
| AMD CDNA | MFMA 16×16×16 | `mfma_16x16x16` |
| AMD RDNA3 / RDNA4 | WMMA 16×16×16 | `wmma_gfx11_*` / `wmma_gfx12` |
| Apple | simdgroup 8×8×8 | `simdgroup_8x8` |

Only the CUDA row has kernel tables and runs on hardware today (see [Portability](./portability)).

## Layout inference

1. **Atoms seed.** An `mma` gives its operands and result the atom's layouts tiled over the
   `WarpGrid` (`layouts::mma_layouts`).
2. **Elementwise ops unify.** Operands and the result of `binary`, `cast`, `where_` and the
   carried values of a loop share one layout.
3. **Unconstrained tiles take a natural layout.** At a fixed point, a tile nothing constrains
   gets `layouts::natural`: each lane holds a short row vector of up to 8 elements, lanes walk
   the columns then the rows, and warps split the rows. Tiles are seeded before vectors.
4. **Vectors follow their tile.** A `[rows, 1]` or `[1, cols]` vector, such as the output of a
   `reduce` or a bias row, is demanded in the row or column layout its tile implies. Inference
   does not default it apart from that tile.
5. **Conflicts become relayouts.** A value two consumers want differently gets a
   `TileOp::Relayout` inserted. `TileLayout::relayout` classifies it as `Identity`,
   `RegPermute` (within each lane), `LaneShuffle` or `ViaSmem`. The emitter currently lowers
   both `LaneShuffle` and `ViaSmem` through shared memory.

## Schedule templates

`schedule::Schedule::Uniform { prefetch, unroll }`: every warp loads and computes.

| `prefetch` | Expansion of `pipeline(extent, stages, …)` |
|---|---|
| `CpAsync`, `stages ≥ 2` | One loop of `extent + stages − 1` iterations. Each iteration waits until at most `stages − 2` copy groups are pending, then barriers and consumes step `i − (stages − 1)`. It then issues step `i`'s `cp.async` copies into the slot freed one iteration earlier and commits them as one group (an empty group past the end keeps the count uniform). |
| `CpAsync`, `stages = 1` | Copy, commit, wait for all, barrier, consume, barrier. |
| `RegisterStaged` (2 stages) | Global → register loads for step `i`, consume step `i − 1`, register → shared stores, barrier. |

`unroll` copies the loop body once per ring slot so slot arithmetic folds to constants. Whether
it helps is measured per config, not assumed (see the GEMM table in the
[Kernel Library](./kernel-library#gemm)).

Async and staged copies belong to the template, which fences them. The `lower::sync` pass covers
the rest: a shared tile written by a synchronous copy is fenced before another thread reads it,
and a tile read since the last fence is fenced before it is overwritten.

## Emission

The emitter lists every statement's instructions in program order inside a pre-linearized
program, so Svod's linearizer toposort never runs on a tk3 kernel. Global accesses past a view's
bounds are gated: loads read zero and stores are dropped. Shared rows are XOR-swizzled in
16-byte chunks when the chunk count per row is a power of two. Shared to register loads of
16-bit tiles use `ldmatrix.x4` where the layout allows. The CUDA backend places shared memory
above the 48 KB static limit in one dynamic array. Config candidates are filtered against the
target's opt-in limit (`Target::smem_bytes`).

:::note[Invariants kept by the emitter, not the author]
Pure values are listed at the loop level their inputs require. Register accesses are scalar.
Shared and global accesses are `SHRINK` vector accesses. Gated stores sit inside an `If`.
Carried values move into their `phi` at the end of the block that defines them. These rules are
enforced in `lower/emit.rs`, and a kernel author never handles them.
:::

`lower::Error::Unsupported` is returned for what is recorded but not lowered yet: warp roles,
role barriers, `raw` statements, register transposes, and async copies on a non-CUDA target.
`TK3_DUMP_LIST=1` prints the emitted list (see [Testing and Debugging](./testing-and-debugging)).
