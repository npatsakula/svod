//! WMMA/MFMA/`mma.sync` matrix-multiply tile ops — the four `mma_{ab,abt,atb,atbt}`
//! variants and their shared looped/unrolled bodies. Each `mma` reduces over the
//! K-edge with one [`MmaPlan`] per 16×16 fragment and K-iteration: a single
//! [`Op::Wmma`](svod_ir::Op::Wmma) on the AMD 16×16×16 cores, two `m16n8k16`
//! `Op::Wmma`s (one per n-half) on CUDA.

use std::sync::Arc;

use smallvec::{SmallVec, smallvec};
use svod_dtype::{DType, GpuArch};
use svod_ir::{AxisId, AxisType, RendererDevice, UOp, WmmaMetadata, WmmaUpcastAxes};
use svod_schedule::optimizer::{Renderer, TensorCore};

use super::Group;
use crate::index::{Idx, flat_index, load_at};
use crate::layout::LaneMap;
use crate::tile::RT;
use crate::tiles::TileLayout;

/// Bridge a scheduler [`TensorCore`] (the per-arch×dtype matrix-op table, the
/// single source of truth — `schedule::optimizer::renderer`) into the IR
/// [`WmmaMetadata`] that a hand-built [`Op::Wmma`](svod_ir::Op::Wmma) consumes.
///
/// `dims`/`dtype_in`/`dtype_out`/`threads` copy straight across. The
/// `upcast_axes` are `log2(elements_per_thread)` size-2 entries per operand
/// (mirrors the optimizer's `tc.rs` construction, where every upcast/reduce split
/// is by 2). The direct TK path has no RANGE nodes, so these are stable structural
/// identities for the fragment dimensions rather than scheduler range identities.
/// `reduce_axes` is empty: tk's `mma` carries the K reduce as its own `inner`
/// range, not inside the WMMA metadata.
fn wmma_from_tc(tc: &TensorCore, device: RendererDevice) -> WmmaMetadata {
    let axes = |ept: usize| -> Vec<(AxisId, usize)> {
        (0..(ept as f64).log2() as usize).map(|i| (AxisId::Renumbered(4 - i), 2)).collect()
    };
    WmmaMetadata {
        name: format!("WMMA_{}_{}_{}_{:?}_{:?}", tc.dims.0, tc.dims.1, tc.dims.2, tc.dtype_in, tc.dtype_out),
        dims: tc.dims,
        dtype_in: tc.dtype_in.clone(),
        dtype_out: tc.dtype_out.clone(),
        device,
        threads: tc.threads,
        upcast_axes: Some(WmmaUpcastAxes {
            a: axes(tc.elements_per_thread.0),
            b: axes(tc.elements_per_thread.1),
            c: axes(tc.elements_per_thread.2),
        }),
        reduce_axes: vec![],
    }
}

/// The K=16 matrix-core descriptor for `dtype_in → dtype_out` on `arch`, looked up
/// from the shared per-arch tensor-core table (`Renderer::for_{amd,cuda}_arch`)
/// rather than re-encoded here — so bf16/f16 on CDNA's MFMA cores, the RDNA wave32
/// cores and CUDA's `mma.sync` come from one source. AMD cores are the square
/// `(16,16,16)`; CUDA's `m16n8k16` is `(8,16,16)` (dims are `(N, M, K)`).
fn wmma_desc(arch: GpuArch, dtype_in: &DType, dtype_out: &DType) -> WmmaMetadata {
    let ren = match arch {
        GpuArch::Amd(amd) => Renderer::for_amd_arch(amd),
        GpuArch::Cuda(cuda) => Renderer::for_cuda_arch(cuda),
        GpuArch::Metal(family) => Renderer::for_metal_family(family),
    };
    let dims = if arch.cuda().is_some() { (8, 16, 16) } else { (16, 16, 16) };
    let tc = ren
        .tensor_cores
        .iter()
        .find(|tc| &tc.dtype_in == dtype_in && &tc.dtype_out == dtype_out && tc.dims == dims)
        .unwrap_or_else(|| {
            // Precondition violation by the kernel author, not end-user input: the
            // matrix-core operand dtype must be bf16/f16 with an f32 accumulator on
            // an arch with a K=16 core. The USE-face kernels pre-cast and gate by
            // `ArchSet`; an AUTHOR calling `mma_*` with an unsupported RT dtype or on
            // an arch without a core lands here.
            unimplemented!(
                "mma: {dtype_in:?} → {dtype_out:?} has no {dims:?} matrix core on {arch:?} — operands must be bf16 \
                 or f16 with an f32 accumulator on an AMD matrix-core arch or CUDA sm_80+"
            )
        });
    wmma_from_tc(tc, ren.device)
}

/// Per-lane element count for a WMMA operand = product of its upcast-axis sizes
/// (`wmma_from_tc` builds these as `log2(elements_per_thread)` size-2 entries, so
/// the product is the elements-per-thread). gfx942 16×16×16 → A/B/C = 4/4/4; RDNA
/// → 16/16/8 (replicated 16-wide inputs, 8-wide accumulator); CUDA m16n8k16 →
/// 8/4/4. Empty axes ⇒ 1.
fn upcast_count(axes: &[(AxisId, usize)]) -> i64 {
    axes.iter().map(|(_, sz)| *sz as i64).product()
}

/// One matrix-core instruction of a fragment step: which registers of the A, B
/// and C tiles it consumes (in the intrinsic's operand order) and whether the
/// A/B tiles feed the intrinsic's `(a, b)` slots swapped.
struct MmaStep {
    a: Vec<i64>,
    b: Vec<i64>,
    c: Vec<i64>,
    swap: bool,
}

/// How one `(height, width, k)` 16×16 fragment product lowers on the arch.
struct MmaPlan {
    meta: WmmaMetadata,
    steps: Vec<MmaStep>,
}

impl MmaPlan {
    /// Resolve the plan for `c += a·b` over the tiles' fragment maps.
    ///
    /// AMD: one WMMA over the descriptor's per-lane widths (4/4/4 on gfx942,
    /// 16/16/8 on RDNA) — the operand tiles must carry the 16-column WMMA base.
    ///
    /// CUDA: every tile is the two-half [`LaneMap::MmaSync`] 16×16 (8 registers).
    /// The A tile's 8 registers are the PTX A fragment `a0..a7`; the B tile
    /// (`k×n` read as `Col`, or `n×k` as `Row` — the same registers either way)
    /// holds n-half `h` in registers `{2h, 2h+1, 2h+4, 2h+5}` (`b0..b3`), and a
    /// `Row` accumulator holds n-half `h` in `4h..4h+4` (`c0..c3`) — ThunderKittens
    /// `mma_AB_base`. A `Col` accumulator holds `Cᵀ` in that register order, so it
    /// is computed as `Cᵀ += Bᵀ·Aᵀ`: the B tile supplies the A fragment, the A tile
    /// the two B fragments (`swap`), and the halves split M instead of N.
    fn resolve(arch: GpuArch, c: &RT<'_>, a: &RT<'_>, b: &RT<'_>, a_t: bool, b_t: bool) -> Self {
        // The fragment registers are read as `[m,k]`/`[k,n]` (`Row`/`Col`) or
        // their transposes; a tile declared the other way round is silently
        // multiplied transposed.
        let expect = |name, layout: TileLayout, wanted: TileLayout| {
            assert_eq!(layout, wanted, "mma: operand {name} must be a {wanted:?} tile for this variant");
        };
        expect("A", a.layout, if a_t { TileLayout::Col } else { TileLayout::Row });
        expect("B", b.layout, if b_t { TileLayout::Row } else { TileLayout::Col });
        let meta = wmma_desc(arch, a.elem(), c.elem());
        let steps = if arch.cuda().is_some() {
            for (name, t) in [("A", a), ("B", b), ("C", c)] {
                assert_eq!(
                    (t.base.map, t.base.base.elements_per_thread()),
                    (LaneMap::MmaSync, 8),
                    "mma: operand {name} must carry the two-half mma.sync 16×16 fragment (RT_16X16_MMA)"
                );
            }
            let swap = c.layout == TileLayout::Col;
            (0..2)
                .map(|h| {
                    let full = (0..8).collect();
                    let half = vec![2 * h, 2 * h + 1, 2 * h + 4, 2 * h + 5];
                    let (a, b) = if swap { (half, full) } else { (full, half) };
                    MmaStep { a, b, c: (4 * h..4 * h + 4).collect(), swap }
                })
                .collect()
        } else {
            assert_eq!(a.base.base.cols, 16, "mma: only the 16-col WMMA base is supported");
            let axes = meta.upcast_axes.as_ref().expect("unexpanded WMMA metadata");
            let regs = |axes| (0..upcast_count(axes)).collect();
            vec![MmaStep { a: regs(&axes.a), b: regs(&axes.b), c: regs(&axes.c), swap: false }]
        };
        MmaPlan { meta, steps }
    }

    /// Emit the plan's instructions for one fragment: `at = [a_at, b_at, c_at]`
    /// give the leading `[height/inner, ..]` indices of each tile, `c_src` the
    /// accumulator buffer to read (carrying its loop/chain dependencies). Returns
    /// the grouped accumulator stores.
    fn emit(&self, a: &RT<'_>, b: &RT<'_>, c: &RT<'_>, at: [&[Idx]; 3], c_src: &Arc<UOp>) -> Arc<UOp> {
        let [a_at, b_at, c_at] = at;
        let gather = |t: &RT<'_>, buf: &Arc<UOp>, at: &[Idx], regs: &[i64]| {
            UOp::stack(
                regs.iter()
                    .map(|&i| {
                        let mut idx: SmallVec<[Idx; 4]> = at.iter().cloned().collect();
                        idx.push(Idx::Const(i));
                        load_at(buf, t.shape(), &idx)
                    })
                    .collect(),
            )
        };
        let stores: Vec<Arc<UOp>> = self
            .steps
            .iter()
            .flat_map(|step| {
                let a_in = gather(a, a.uop(), a_at, &step.a);
                let b_in = gather(b, b.uop(), b_at, &step.b);
                let d_in = gather(c, c_src, c_at, &step.c);
                let (x, y) = if step.swap { (b_in, a_in) } else { (a_in, b_in) };
                let out = UOp::wmma(x, y, d_in, self.meta.clone());
                step.c
                    .iter()
                    .enumerate()
                    .map(|(pos, &i)| {
                        let mut idx: SmallVec<[Idx; 4]> = c_at.iter().cloned().collect();
                        idx.push(Idx::Const(i));
                        flat_index(c.uop(), c.shape(), &idx).store(out.index_axes(vec![pos]))
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        UOp::group(stores)
    }
}

impl<'k> Group<'k> {
    /// `C += A·B` over a tile (tinygrad `mma_AB`): for every output fragment
    /// `(height, width)` accumulate `WMMA(A[height,inner], B[inner,width])`
    /// across the reduce axis `inner`. One [`Op::Wmma`](svod_ir::Op::Wmma) per
    /// K-iteration → one `mfma.f32.16x16x16bf16.1k` (two `mma.sync.m16n8k16` on CUDA).
    ///
    /// # Panics
    /// The operand tiles `a`/`b` must be **bf16 or f16** — the only K=16
    /// matrix-core input dtypes — and `c` f32. An operand of any other dtype panics
    /// (a kernel-authoring error); this precondition holds for all four
    /// `mma_{ab,abt,atb,atbt}` variants. Also panics unless the operand tiles carry
    /// the arch's matrix-core fragment (the 16-column WMMA base on AMD, the two-half
    /// `mma.sync` fragment on CUDA), and on an operand-rank mismatch (the index
    /// permutation reads the trailing fragment-grid axes).
    pub fn mma_ab(&self, c: RT<'k>, a: &RT<'k>, b: &RT<'k>) -> RT<'k> {
        self.mma(c, a, b, false, false)
    }

    /// `C += A·Bᵀ` (tinygrad `mma_ABt`): B fragment is read transposed
    /// (`b[width, inner]`); reduce axis stays `a.shape[-2]`.
    ///
    /// # Panics
    /// See [`Self::mma_ab`].
    pub fn mma_abt(&self, c: RT<'k>, a: &RT<'k>, b: &RT<'k>) -> RT<'k> {
        self.mma(c, a, b, false, true)
    }

    /// `C += Aᵀ·B` (tinygrad `mma_AtB`): A fragment is read transposed
    /// (`a[inner, height]`) and the reduce axis is `a.shape[-3]`.
    ///
    /// # Panics
    /// See [`Self::mma_ab`].
    pub fn mma_atb(&self, c: RT<'k>, a: &RT<'k>, b: &RT<'k>) -> RT<'k> {
        self.mma(c, a, b, true, false)
    }

    /// `C += Aᵀ·Bᵀ` (tinygrad `mma_AtBt`): both fragments read transposed.
    ///
    /// # Panics
    /// See [`Self::mma_ab`].
    pub fn mma_atbt(&self, c: RT<'k>, a: &RT<'k>, b: &RT<'k>) -> RT<'k> {
        self.mma(c, a, b, true, true)
    }

    /// The shared matrix-multiply body. The four `mma_{AB,ABt,AtB,AtBt}` variants
    /// differ only in the operand index permutation and the reduce-axis selection:
    /// - `a_t` (Aᵀ): A is read `a[inner, height]` and the reduce axis is
    ///   `a.shape[-3]`; otherwise `a[height, inner]`, reduce axis `a.shape[-2]`.
    /// - `b_t` (Bᵀ): B is read `b[width, inner]`; otherwise `b[inner, width]`.
    ///
    /// Wave-agnostic: each wave runs the matrix op on its own per-lane RT operands
    /// (the wave sub-tile selection happens in the LDS→REG load, not here).
    fn mma(&self, c: RT<'k>, a: &RT<'k>, b: &RT<'k>, a_t: bool, b_t: bool) -> RT<'k> {
        // Flat (cross-tile-pipeline) FA opts into the fully-unrolled body so the
        // QKᵀ / A·V MFMAs render loop-free for the attention scheduling comb.
        if self.ker.unrolled() {
            return self.mma_u(c, a, b, a_t, b_t);
        }
        let plan = MmaPlan::resolve(self.ker.caps.arch, &c, a, b, a_t, b_t);

        let h_end = c.shape()[c.shape().len() - 3] as i64;
        let w_end = c.shape()[c.shape().len() - 2] as i64;
        let k_end = if a_t { a.shape()[a.shape().len() - 3] } else { a.shape()[a.shape().len() - 2] } as i64;
        let height = self.ker.raw_range(h_end, AxisType::Loop);
        let width = self.ker.raw_range(w_end, AxisType::Loop);
        let inner = self.ker.raw_range(k_end, AxisType::Reduce);

        let a_at = if a_t { [Idx::from(&inner), Idx::from(&height)] } else { [Idx::from(&height), Idx::from(&inner)] };
        let b_at = if b_t { [Idx::from(&width), Idx::from(&inner)] } else { [Idx::from(&inner), Idx::from(&width)] };
        let c_at = [Idx::from(&height), Idx::from(&width)];
        // The accumulator read must depend on the reduce range `inner`, or it is
        // loop-invariant w.r.t. the K loop and gets hoisted *out* of it — every
        // K-iteration would then re-read the pre-loop C and the WMMA's
        // accumulation chain breaks. Mirrors svod's `reduce_to_acc`
        // (`acc.after([..reduce_range]).index(..)`): the `After([inner])` keeps
        // the read inside the K loop so it observes the prior iteration's store.
        let c_acc = c.uop().after(smallvec![inner.clone()]);
        let c_store = plan.emit(a, b, &c, [&a_at, &b_at, &c_at], &c_acc).end(smallvec![height, width, inner]);
        self.finalize_reg(c, c_store)
    }

    /// Fully **unrolled** [`Self::mma`]: emit the fragment plan per
    /// `(height, width, k)` via Rust `for` loops — **no inner `RANGE`** — so the
    /// MFMAs render as a *flat* schedulable LLVM region the attention scheduling
    /// comb can weave the online softmax through. tk's direct-launch path skips the
    /// optimizer's `pre_expand`, so the looped [`Self::mma`] stays rolled (three
    /// `loop_body_*` around the mfma); explicit unroll is the only way to flatten
    /// it (route b — the cheap axis-flip is dead on the direct path).
    ///
    /// Each fragment's K-accumulation chains (`c[h,w]`'s k-step read observes the
    /// k−1 store); fragments chain into one terminal store so the enclosing rolled
    /// KV loop's `END` scopes them all (cf. the matmul accumulator chain,
    /// `kernels/matmul.rs:201`). Bit-identical accumulation order to [`Self::mma`].
    fn mma_u(&self, c: RT<'k>, a: &RT<'k>, b: &RT<'k>, a_t: bool, b_t: bool) -> RT<'k> {
        let plan = MmaPlan::resolve(self.ker.caps.arch, &c, a, b, a_t, b_t);

        let h_end = c.shape()[c.shape().len() - 3] as i64;
        let w_end = c.shape()[c.shape().len() - 2] as i64;
        let k_end = if a_t { a.shape()[a.shape().len() - 3] } else { a.shape()[a.shape().len() - 2] } as i64;

        // Fragment-scoping chain: each fragment's first (k=0) accumulator read
        // orders after the previous fragment's terminal store, so the LAST
        // fragment's store transitively scopes them all under one loop `END`.
        let mut prev_frag: Option<Arc<UOp>> = None;
        for h in 0..h_end {
            for w in 0..w_end {
                // Per-fragment K accumulation: the k-step read observes the k−1
                // store to this same fragment (the unrolled analog of the looped
                // `c.after([inner])` loop-carry).
                let mut frag_prev: Option<Arc<UOp>> = None;
                for k in 0..k_end {
                    let a_at = if a_t { [Idx::Const(k), Idx::Const(h)] } else { [Idx::Const(h), Idx::Const(k)] };
                    let b_at = if b_t { [Idx::Const(w), Idx::Const(k)] } else { [Idx::Const(k), Idx::Const(w)] };
                    let c_at = [Idx::Const(h), Idx::Const(w)];
                    // Accumulator source: the prior k-step's store for this
                    // fragment; on k==0 the incoming `c` carrying the
                    // fragment-scoping dep on the previous fragment's store.
                    let deps: SmallVec<[Arc<UOp>; 4]> =
                        frag_prev.as_ref().or(prev_frag.as_ref()).cloned().into_iter().collect();
                    // Anchor the incoming accumulator read (no chain dep yet) to the
                    // enclosing rolled loop so a carried accumulator (`o_reg`) is not
                    // hoisted out (see `Group::anchor`); subsequent k/fragment reads
                    // chain through their stores, which are already loop-scoped.
                    let c_src = if deps.is_empty() { self.anchor(c.uop()) } else { c.uop().after(deps) };
                    frag_prev = Some(plan.emit(a, b, &c, [&a_at, &b_at, &c_at], &c_src));
                }
                prev_frag = frag_prev;
            }
        }
        let terminal = prev_frag.expect("mma_u: at least one (height, width) fragment");
        self.finalize_reg(c, terminal)
    }
}
