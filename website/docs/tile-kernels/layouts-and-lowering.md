---
sidebar_label: Layouts and Lowering
---

# Layouts and Lowering

`lower::lower(program, &lowering, params, device)` turns a tile program into a Svod program
whose instruction list is already in order. A `Lowering` holds everything the author did not
decide: the `Device` (the target and its costs), the `Schedule`, the `WarpGrid` over the block
tile, whether shared rows are swizzled (`swizzle`), and which matrix instruction the products
issue on (`core`). Each kernel config builds its own (`GemmCfg::lowering`, `FaCfg::lowering`,
`NormCfg::lowering`). `lower::standalone(program, &lowering)` lowers for a target with no device
attached, and `lower::render` returns the LLVM text the backend's compiler would be handed.

| Step | Module | Effect |
|---|---|---|
| 1. Operand materialization | `lower` | A shared or global `mma` operand gets an explicit load into a fresh register tile; on the warpgroup core shared operands stay in place |
| 2. Barriers | `lower::sync` | Shared-memory effects are fenced: barriers, and a commit and wait before an async copy's tile is used. A `Pipeline` is still one statement whose template fences its own ring, so the pass fences only where the ring meets the code around it |
| 3. Schedule expansion | `schedule::expand` | Every `Pipeline` becomes loops, branches and explicit `Sync` statements |
| 4. Layout inference | `layouts::infer` | Every register tile gets a layout; conflicts get a `Relayout` |
| 5. Shared orientation | `lower` | A shared allocation filled through registers is stored along the axis its readers' fragments run along, so a gather is 16-byte loads; `cp.async` tiles stay row-major |
| 6. Emission | `lower/emit/` | Statements are emitted in program order into `UOp::linear_program` |

`lower` fails with `Error::Grid` when the program's warps are not the warp grid's, or a
multiple of it with a `Warps` statement in the body, and with `Error::DivergentBarrier` when a
block barrier sits under a branch on the warp, which some warps would skip.

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

The algebra has `identity`, `zeros`, `from_bases`, `transpose`, `product` (tile one layout over
another), `sublayout` and `apply`. The tests check every atom against the lane formulas of the
vendor ISA documents (`test/unit/atoms.rs`) and against tk1's closed forms (`test/unit/layout.rs`).

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
| CUDA sm_90 | `wgmma` m64nNk16, four warps at once | `wgmma_a`, `wgmma_c`; `B` through a descriptor, never held |
| AMD CDNA | MFMA 16×16×16 | `mfma_16x16x16` |
| AMD RDNA3 / RDNA4 | WMMA 16×16×16 | `wmma_gfx11_input`, `wmma_gfx11_acc` / `wmma_gfx12` |
| Apple | simdgroup 8×8×8 (atom only; no kernels) | `simdgroup_8x8` |

Each target with kernels also has an 8-bit integer atom into i32 (`Target::geometry_of(Int8)`):
`mma.sync` m16n8k32 s8 and CDNA's MFMA 16×16×32 i8 hold the 16-bit fragments' bytes, twice as
deep (`mma_sync_a8`, `mma_sync_b8`, `mfma_16x16x32_i8`); RDNA's `iu8` WMMA keeps the 16-bit
layouts at depth 16. Its `m` and `n` are the 16-bit atom's. The planner prices the 16-bit
geometry (`Target::geometry`) except for a problem over 8-bit rows.

An `atoms::GroupMmaAtom` is a warpgroup instruction: four consecutive warps issue a 64 × n
product together over operands read from shared memory through descriptors (`B` always, `A`
unless it is a register tile), the accumulator spread over the group with each warp holding
16 rows in the `mma.sync` C layout. A config whose `core` is `Core::Group` lowers its products
on it: the grid's rows are warpgroups, shared operands stay in place instead of being loaded,
and the emitter writes each product as one `wgmma` chain over the ring's swizzled rows (the
descriptor's swizzle mode is the lowering's own XOR, `layouts::Swizzle`). The GEMM and
attention configs both offer this core on sm_90.

## Layout inference

1. **Atoms seed.** An `mma` gives its operands and result the atom's layouts tiled over the
   `WarpGrid` (`MmaAtom::issue`), issued whichever way round (`ir::Orient`) leaves the program
   the least relayout cost under the lowering's `Costs`: `Cᵀ = Bᵀ·Aᵀ` is the same instruction
   with its slots exchanged, so on a square atom an accumulator can be held as the next
   product's operand at no cost. A product whose result is dealt (`TileOp::Deal`, a list's lane
   queues) is issued the way round that spreads a row over the fewest lanes (`MmaAtom::narrow`),
   where a deal renames registers, as the op layer prices it. An author may pin the orientation
   (`Kernel::mma_oriented`).
2. **Elementwise ops unify.** Operands and the result of `binary`, `cast`, `where_`, `mul_add`
   and the carried values of a loop share one layout.
3. **Unconstrained tiles take a natural layout.** At a fixed point, a tile nothing constrains
   gets `layouts::natural`: each lane holds a short row vector of up to 8 elements, lanes walk
   the columns then the rows, and warps split the rows.
4. **Vectors follow their tile.** A `[rows, 1]` or `[1, cols]` vector, such as the output of a
   `reduce` or a bias row, is demanded in the row or column layout its tile implies.
5. **Conflicts become relayouts.** A value two consumers want differently gets a
   `TileOp::Relayout`. `TileLayout::relayout` classifies it as `Identity`, `RegPermute`
   (within each lane, free), `Exchange` (each register taken by lane-xor moves from fixed
   partners) or `ViaSmem`. An `Exchange` is kept only while its work costs less than a round
   trip through shared memory (`Relayout::work × Costs::lane_move < Costs::smem_trip`). The
   emitter lowers an `Exchange` with the target's `atoms::LaneMove` (`shfl.sync` on CUDA, DPP
   and `permlanex16` on RDNA, `ds_bpermute` on CDNA).

## Schedule templates

`schedule::Schedule::Uniform { fill, unroll, async_products }`: every warp loads and computes.
`Schedule::PingPong { crew }`: two crews of `crew` warps over one register-staged slot (CDNA
GEMM, `GemmCfg::crews == 2`).

| Fill | Expansion of `pipeline(extent, stages, …)` |
|---|---|
| `Fill::CpAsync`, `stages ≥ 2` | One loop of `extent + stages − 1` iterations. Each iteration waits until at most `stages − 2` copy groups are pending, then barriers and consumes step `i − (stages − 1)`. It then issues step `i`'s `cp.async` copies into the slot freed one iteration earlier and commits them as one group (an empty group past the end keeps the count uniform). |
| `Fill::CpAsync`, `stages ≥ 3`, `async_products` | The same, with the consume's `wgmma` chains left in flight over the next fill: the slot consumed two trips ago is refilled, and after each consume every thread waits for all but its youngest chain (`Sync::WaitProducts`). |
| `Fill::CpAsync`, `stages = 1` | Copy, wait, consume, and a barrier before the slot is refilled. |
| `Fill::Staged`, 2 stages | Three loops over one induction variable: the first trip loads step 0 into registers and stores it to its slot; a steady trip loads step `i` into registers, fences, consumes step `i − 1`, fences, and stores the registers to step `i`'s slot, with no branch in its body (the fences keep the loads ahead of the products and the commit behind them on RDNA); the last trip only consumes. |
| `Fill::Staged`, 1 stage | The same three loops over one slot: the store waits behind a barrier for every warp to finish the slot, then barriers again, so a tile twice as deep fits the same shared memory. |
| `PingPong` | A trip is cut at its products into load clusters and product clusters, a block barrier after each, so one crew's products run from registers at raised issue priority while the other crew loads. |

`unroll` copies the loop body once per ring slot so slot arithmetic folds to constants; the
tuner measures both. Async and staged copies belong to the template, which fences them. The
`lower::sync` pass covers the rest: a shared tile written by a synchronous copy is fenced before
another thread reads it, and a tile read since the last fence is fenced before it is overwritten.

## Emission

The emitter lists every statement's instructions in program order inside a pre-linearized
program, so Svod's linearizer toposort never runs on a tk kernel. Global accesses past a view's
bounds are gated: loads read zero and stores are dropped. Shared rows are XOR-swizzled in
16-byte chunks where the row's chunk count is a power of two (`layouts::Swizzle`). Shared to register loads of 16-bit tiles use `ldmatrix.x4` where the layout
allows, and of 8-bit ones its untransposed form over 16×32 bytes; on RDNA4 global to register loads use `global_load_tr_b128` where the layout is what it
leaves. A `Warps` statement runs its body on each warp alone, with its own copies and barriers.
On AMD an f32 `exp2` is the hardware `v_exp_f32`, which flushes results below 2⁻¹²⁶ to zero.
Shared memory above 48 KB is one dynamic array on CUDA; candidates are filtered against
`Target::smem_bytes`.

:::note[Invariants kept by the emitter, not the author]
Pure values are listed at the loop level their inputs require. Register accesses are scalar.
Shared and global accesses are `SHRINK` vector accesses. Gated stores sit inside an `If`.
Carried values move into their `phi` at the end of the block that defines them.
:::

`TK_DUMP_LIST=1` prints the emitted list (see [Testing and Debugging](./testing-and-debugging)).
