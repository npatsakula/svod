//! The tiled GEMM core shared by the square [`matmul`] and the production NT
//! linear-layer kernel [`gemm_nt`].
//!
//! One `block_m × block_n` C tile per workgroup; a `warps_m × warps_n` wave grid
//! where each wave owns `acc_m` col-major `reg_m × reg_n` f32 accumulators reduced
//! over a `k_step`-wide K strip staged through XOR-swizzled shared memory. Two
//! orthogonal knobs generalize the original square builder:
//!
//! - [`BOrder`] — whether B arrives K-major (`B[k, n]`, the square matmul) or as a
//!   row-major `[N, K]` weight (`B[n, k]`, consumed with `mma_abt`). The `Nk` form
//!   needs neither a transposing fill nor `ldmatrix.trans`: the registers of an
//!   `n×k` `Row` fragment and a `k×n` `Col` fragment are the same lane map with the
//!   two coordinates exchanged, so both strips fill and gather identically and the
//!   orientation is carried entirely by the `mma` variant.
//! - [`GemmCfg::stages`] — `1` keeps the single-buffered fill/barrier/gather/mma
//!   loop; `2` runs the software pipeline (the flash-attention K/V pattern): the
//!   next K strip is in flight under the current strip's gathers and MMAs, under
//!   one workgroup barrier per trip. The fill primitive is the arch's, chosen
//!   through [`Group`]: `cp.async` straight into the other shared half where it
//!   applies (CUDA sm_80+), else the register-staged stream — `global_load`
//!   before the MMAs, `ds_write` into the other half after them.
//!
//! The output dtype is the bound C buffer's: a bf16 `c_gl` makes the epilogue cast
//! the f32 accumulators on the way out, with no f32 round trip through memory.
//! [`Epilogue`] then says what else that store does — add a residual, or fold the
//! paired gate/up columns into `silu(gate)·up` — so a fusion the graph would pay
//! a whole extra pass over memory for costs the GEMM nothing but the operand it
//! reads.

use std::sync::Arc;

use smallvec::{SmallVec, smallvec};
use snafu::{ResultExt, ensure};
use svod_codegen::llvm::nvptx::smem::{cp_async_wait, cp_async_wait_all};
use svod_dtype::DType;
use svod_ir::{ConstValue, UOp};
use svod_tensor::Tensor;

use crate::index::{Idx, cidx, load_at, load_off};
use crate::tiles::TileLayout;
use crate::{GL, GlSpec, Group, Kernel, Loop, MoveIdx, RT, RegTile, ST};

/// Where the B operand's K axis lives in global memory.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BOrder {
    /// `B[k, n]` — the K-major operand of the square `A·B` matmul.
    Kn,
    /// `B[n, k]` — a row-major `[N, K]` weight, consumed transposed (`C = A·Bᵀ`).
    Nk,
}

/// The fusion folded into the GEMM's store, generic over how its extra operand
/// is carried: a [`Tensor`] at the launch entry ([`gemm_nt_with_epilogue`]), the
/// bound [`GL`] inside the builder, and `()` where only the shape matters.
///
/// Both fused forms consume the f32 accumulators in registers, so the value the
/// epilogue writes is the only one that ever reaches memory.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Epilogue<T> {
    /// `y = x·wᵀ`.
    Plain,
    /// `y = x·wᵀ + residual`, `residual` `[lead..., N]` in the operand dtype.
    /// The GEMM result is rounded to that dtype and then added, exactly as the
    /// graph's `try_add` over two operand-dtype tensors rounds it.
    Add(T),
    /// `y[.., i] = silu(gate_i)·up_i` — `w` is a fused `[2I, K]` gate/up weight
    /// whose rows alternate gate and up blocks of `pair` rows, so one wave's
    /// accumulator holds a gate block beside its matching up block and `y` comes
    /// out `[lead..., N/2]`. `pair` is [`swiglu_pair_width`].
    SwiGlu { pair: usize },
}

impl<T> Epilogue<T> {
    /// This epilogue with its operand dropped — the shape-only form the kernel
    /// builder is parameterized by.
    pub fn kind(&self) -> Epilogue<()> {
        match self {
            Epilogue::Plain => Epilogue::Plain,
            Epilogue::Add(_) => Epilogue::Add(()),
            Epilogue::SwiGlu { pair } => Epilogue::SwiGlu { pair: *pair },
        }
    }

    /// Columns the epilogue writes for an `n`-column GEMM — halved by
    /// [`Epilogue::SwiGlu`], which folds each gate/up column pair into one.
    pub const fn out_cols(&self, n: usize) -> usize {
        match self {
            Epilogue::SwiGlu { .. } => n / 2,
            _ => n,
        }
    }

    /// A small integer naming the variant (the tuning key's epilogue field).
    pub const fn code(&self) -> usize {
        match self {
            Epilogue::Plain => 0,
            Epilogue::Add(_) => 1,
            Epilogue::SwiGlu { .. } => 2,
        }
    }
}

/// Block / wave geometry of one GEMM workgroup. A `warps_m × warps_n` wave grid
/// computes a `block_m × block_n` C tile; each wave owns `acc_m` col-major
/// `reg_m × reg_n` f32 accumulators (`reg_m = block_m / (warps_m·acc_m)`,
/// `reg_n = block_n / warps_n`) reduced over K in `k_step`-wide strips.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GemmCfg {
    pub block_m: usize,
    pub block_n: usize,
    /// Wave-grid rows — the M side splits into `warps_m · acc_m` row-blocks.
    pub warps_m: usize,
    /// Wave-grid columns — the N side splits into `warps_n` col-blocks.
    pub warps_n: usize,
    /// Accumulators per wave along M.
    pub acc_m: usize,
    /// K-reduction step (the shared strip depth); a multiple of the matrix core's
    /// K-edge, and `K` must be a multiple of `k_step · split_k`.
    pub k_step: usize,
    /// Shared-memory strips in flight: `1` = single-buffered, `2` = the
    /// double-buffered software pipeline (`cp.async` where the arch has it, the
    /// register-staged stream elsewhere; deeper pipelines are `cp.async`-only).
    pub stages: usize,
    /// B's global layout.
    pub b_order: BOrder,
    /// Drive `(pid_m, pid_n)` from a flattened 1-D grid via the chiplet/L2
    /// [`l2_swizzle`](crate::grid::l2_swizzle) instead of the plain 2-D `block_idx`.
    pub l2_swizzle: bool,
    /// Fill the single-buffered strips with 128-bit coalesced loads (`cp.async` on
    /// sm_80+) instead of the scalar path. Ignored when `stages > 1` (the pipeline
    /// has its own fill).
    pub vec_load: bool,
    /// K-slabs the reduction is split across: each `(pid_m, pid_n)` block is
    /// computed by `split_k` workgroups over `K / split_k` each, written as
    /// `split_k` f32 partials and summed by a second pass. `1` disables the split.
    pub split_k: usize,
}

impl GemmCfg {
    /// Whether this tile can carry `epi`. The fused epilogues write the operand
    /// dtype, so they need the direct (un-split) store; [`Epilogue::SwiGlu`]
    /// additionally needs each wave's accumulator to hold a gate block beside its
    /// matching up block — its `reg_n/2` must be the `pair` width the weight rows
    /// were arranged in, and a whole number of `frag_cols`-wide fragments, so the
    /// gate/up split falls on a fragment boundary. `frag_cols` is the arch's
    /// accumulator-fragment width (`None` on an arch with no matrix core).
    pub fn carries(&self, epi: Epilogue<()>, frag_cols: Option<usize>) -> bool {
        match epi {
            Epilogue::Plain => true,
            Epilogue::Add(()) => self.split_k == 1,
            Epilogue::SwiGlu { pair } => {
                self.split_k == 1 && self.reg_n() == 2 * pair && frag_cols.is_some_and(|c| pair.is_multiple_of(c))
            }
        }
    }

    /// Per-accumulator M edge.
    pub const fn reg_m(&self) -> usize {
        self.block_m / (self.warps_m * self.acc_m)
    }
    /// Per-accumulator N edge.
    pub const fn reg_n(&self) -> usize {
        self.block_n / self.warps_n
    }
    /// `reg_m`-blocks per C tile along M (the grid→C-block multiplier).
    pub const fn blocks_m(&self) -> usize {
        self.warps_m * self.acc_m
    }
    /// `reg_n`-blocks per C tile along N.
    pub const fn blocks_n(&self) -> usize {
        self.warps_n
    }
    /// Launch block size (threads).
    pub const fn threads(&self, wave_size: usize) -> i64 {
        (self.warps_m * self.warps_n * wave_size) as i64
    }
    /// Workgroups an `m × n` C launches.
    pub const fn blocks(&self, m: usize, n: usize) -> usize {
        (m / self.block_m) * (n / self.block_n) * self.split_k
    }
    /// Launch grid for an `m × n` C. The K-slab is **always** grid z, so the
    /// builder reads it from one place; the `(pid_m, pid_n)` block comes from the
    /// flattened `x` under the L2 swizzle (which re-derives the pair) and from
    /// `(x, y) = (pid_n, pid_m)` otherwise.
    pub const fn grid_dims(&self, m: usize, n: usize) -> [i64; 3] {
        let (gm, gn) = ((m / self.block_m) as i64, (n / self.block_n) as i64);
        let gk = self.split_k as i64;
        if self.l2_swizzle { [gm * gn, 1, gk] } else { [gn, gm, gk] }
    }
    /// Shared memory one workgroup takes for `in_bytes`-wide operands.
    pub const fn shared_bytes(&self, in_bytes: usize) -> usize {
        self.stages * (self.block_m + self.block_n) * self.k_step * in_bytes
    }
    /// Whether `m`/`k`/`n` tile this config exactly.
    pub const fn tiles(&self, m: usize, k: usize, n: usize) -> bool {
        m.is_multiple_of(self.block_m)
            && n.is_multiple_of(self.block_n)
            && k.is_multiple_of(self.k_step * self.split_k)
            // The pipeline prologue fills one shared half per stage, so a K-slab
            // shorter than the pipeline would write a half twice before the first
            // gather.
            && k / self.split_k / self.k_step >= self.stages
    }
}

/// The M-row C-block coordinate of accumulator `a` (`warp_row + a·warps_m`, in
/// `reg_m`-block units).
fn acc_row(warp_row: &Arc<UOp>, a: usize, cfg: &GemmCfg) -> Arc<UOp> {
    if a == 0 { warp_row.clone() } else { warp_row.add(&cidx((a * cfg.warps_m) as i64)) }
}

/// The `(pid_m, pid_n)` C-block coordinate (in block units) for this workgroup.
fn block_coords(ker: &Kernel, m: usize, n: usize, cfg: &GemmCfg) -> (Arc<UOp>, Arc<UOp>) {
    if cfg.l2_swizzle {
        let (gm, gn) = ((m / cfg.block_m) as i64, (n / cfg.block_n) as i64);
        crate::grid::l2_swizzle(ker.block_idx[0].clone(), gm * gn, gm, gn)
    } else {
        (ker.block_idx[1].clone(), ker.block_idx[0].clone())
    }
}

/// The `(rows, cols)` of B's shared strip.
fn b_strip(cfg: &GemmCfg) -> (usize, usize) {
    match cfg.b_order {
        BOrder::Kn => (cfg.k_step, cfg.block_n),
        BOrder::Nk => (cfg.block_n, cfg.k_step),
    }
}

/// B's global block index at K-strip `tile` of N-block `col`.
fn b_index(cfg: &GemmCfg, col: &Arc<UOp>, tile: &Arc<UOp>) -> [Idx; 4] {
    let (r, c) = match cfg.b_order {
        BOrder::Kn => (tile, col),
        BOrder::Nk => (col, tile),
    };
    [Idx::Const(0), Idx::Const(0), Idx::from(r), Idx::from(c)]
}

/// This wave's B register tile and its shared sub-tile view, per [`BOrder`].
fn b_operand<'k>(ker: &'k Kernel, cfg: &GemmCfg, in_dt: &DType, warp_col: &Arc<UOp>, b_smem: &ST) -> (RT<'k>, ST) {
    let (reg_n, k_step) = (cfg.reg_n(), cfg.k_step);
    match cfg.b_order {
        BOrder::Kn => (
            ker.operand_b((k_step, reg_n), in_dt.clone(), TileLayout::Col),
            b_smem.subtile((k_step, reg_n), (0, warp_col.clone())),
        ),
        BOrder::Nk => (
            ker.operand_b((reg_n, k_step), in_dt.clone(), TileLayout::Row),
            b_smem.subtile((reg_n, k_step), (warp_col.clone(), 0)),
        ),
    }
}

/// The GEMM body into the already-bound `c_gl` — `C[m,n] = A[m,k] · B` with B
/// laid out per [`GemmCfg::b_order`], then `epi` ([`Epilogue`]) applied to each
/// accumulator in registers. The accumulators are f32; the epilogue casts them to
/// `c_gl`'s element dtype on the way out.
///
/// # Panics
/// Panics unless `m` is a multiple of `cfg.block_m`, `n` of `cfg.block_n`,
/// `cfg.k_step` of the arch's matrix-core K-edge, and `k` of `cfg.k_step ·
/// cfg.split_k`; unless `block_m`/`block_n` divide into the wave grid; for a
/// pipelined config, unless the strips admit `cp.async` fills; and, under
/// [`Epilogue::SwiGlu`], unless `cfg.reg_n()` splits into two whole fragment
/// blocks.
pub fn gemm_core(
    ker: &Kernel,
    (m, k, n): (usize, usize, usize),
    cfg: GemmCfg,
    c_gl: GL,
    a_gl: GL,
    b_gl: GL,
    epi: Epilogue<GL>,
) {
    assert_eq!(m % cfg.block_m, 0, "gemm M={m} must be a multiple of the {} block", cfg.block_m);
    assert_eq!(n % cfg.block_n, 0, "gemm N={n} must be a multiple of the {} block", cfg.block_n);
    // The K-edge is the A fragment's column count — 16 on MFMA/`mma.sync`, 8 on
    // Apple's `simdgroup_matrix`.
    let wmma_k = ker.caps.frag(crate::arch::FragRole::Operand).expect("matrix-core fragment").base.cols;
    assert_eq!(
        cfg.k_step % wmma_k,
        0,
        "k_step={} must be a multiple of {wmma_k} (this arch's WMMA K-edge)",
        cfg.k_step
    );
    assert_eq!(k % (cfg.k_step * cfg.split_k), 0, "gemm K={k} must be a multiple of k_step·split_k");
    assert!(cfg.stages >= 1 && k / cfg.split_k / cfg.k_step >= cfg.stages, "gemm K-slab shorter than the pipeline");
    assert_eq!(cfg.block_m % cfg.blocks_m(), 0, "block_m must split into warps_m·acc_m row-blocks");
    assert_eq!(cfg.block_n % cfg.warps_n, 0, "block_n must split into warps_n col-blocks");

    let (reg_m, reg_n, k_step) = (cfg.reg_m(), cfg.reg_n(), cfg.k_step);
    let g = ker.group_2d(cfg.warps_m, cfg.warps_n);
    // The matrix-core input dtype is the operands' (bf16 or f16, both K=16 cores).
    let in_dt = a_gl.elem().clone();

    // A strip [block_m × k_step]; B strip per `b_order`; both XOR-swizzled, and
    // `stages`-deep when the pipeline runs.
    let a_smem = ker.shared_sw_stages((cfg.block_m, k_step), in_dt.clone(), TileLayout::Row, cfg.stages);
    let b_smem = ker.shared_sw_stages(b_strip(&cfg), in_dt.clone(), TileLayout::Row, cfg.stages);

    let (row, col) = block_coords(ker, m, n, &cfg); // (pid_m, pid_n) in block units
    let warp_row = g.warp_row();
    let warp_col = g.warp_col();
    // The K-slab this workgroup reduces (split-K rides grid z); a single slab is
    // the constant 0, so the offset folds away at build time.
    let trips = (k / cfg.split_k / k_step) as i64;
    let slab = (cfg.split_k > 1).then(|| ker.grid_z().mul(&cidx(trips)));

    // `acc_m` col-major reg_m×reg_n f32 accumulators per wave.
    let accs: Vec<RT> = (0..cfg.acc_m).map(|_| g.zero(ker.acc((reg_m, reg_n), TileLayout::Col))).collect();

    let lp = ker.loop_static(trips);
    let strip = Strips { cfg: &cfg, a_gl: &a_gl, b_gl: &b_gl, row: &row, col: &col, slab: slab.clone(), trips };

    let (a_cur, b_cur, stream) =
        if cfg.stages > 1 { strip.pipelined(&g, &lp, a_smem, b_smem) } else { strip.single(&g, &lp, a_smem, b_smem) };

    // Shared B sub-tile (N col-block {warp_col}, same for every accumulator), and
    // per-accumulator A sub-tiles (M row-block {warp_row + a·warps_m}).
    let (b_reg, b_view) = b_operand(ker, &cfg, &in_dt, &warp_col, &b_cur);
    let bb = g.load(b_reg, b_view, MoveIdx::default());
    let a_subs: Vec<RT> = (0..cfg.acc_m)
        .map(|a| {
            g.load(
                ker.operand((reg_m, k_step), in_dt.clone(), TileLayout::Row),
                a_cur.subtile((reg_m, k_step), (acc_row(&warp_row, a, &cfg), 0)),
                MoveIdx::default(),
            )
        })
        .collect();

    // Cross-wave WAR barrier: every wave must finish reading LDS before the next
    // K iteration's collaborative fill overwrites it. The pipelines fence
    // elsewhere — at the loop top (`cp.async`) or in the tail commit (staged) —
    // with one barrier covering both the RAW and the WAR.
    let (bb, a_subs) = if matches!(stream, Stream::Single) {
        let mut bar_deps: SmallVec<[Arc<UOp>; 4]> = smallvec![bb.uop().clone()];
        bar_deps.extend(a_subs.iter().skip(1).map(|t| t.uop().clone()));
        let sync = a_subs[0].uop().barrier(bar_deps);
        let bb = bb.after(smallvec![sync.clone()]);
        (bb, a_subs.into_iter().map(|t| t.after(smallvec![sync.clone()])).collect())
    } else {
        (bb, a_subs)
    };

    // MMA-accumulate each accumulator over the K sub-steps; chain accumulator `a`'s
    // A-input through accumulator `a-1`'s MMA so a single `END` scopes them all
    // inside the K-loop.
    let mut prev_out: Option<Arc<UOp>> = None;
    for (a, a_sub) in a_subs.iter().enumerate() {
        let a_sub = match &prev_out {
            Some(p) => a_sub.after(smallvec![p.clone()]),
            None => a_sub.clone(),
        };
        let acc = accs[a].clone();
        let out = match cfg.b_order {
            BOrder::Kn => g.mma_ab(acc, &a_sub, &bb),
            BOrder::Nk => g.mma_abt(acc, &a_sub, &bb),
        };
        prev_out = Some(out.uop().clone());
    }
    let ended = match &stream {
        // The staged stream's `ds_write` of the next strip lands after this trip's
        // MMAs (ordered through the last one), and the one barrier-wrapped commit
        // of both strips is the loop's terminal store: one fence per trip,
        // covering the RAW on the half just written and the WAR on the half every
        // wave just gathered.
        Stream::Staged { stage, nxt } => {
            let after_mma = prev_out.clone().expect("at least one accumulator");
            let (a_nxt, b_nxt) = (nxt[0].after(&after_mma), nxt[1].after(&after_mma));
            let fenced = g.commit_regs_to_local(&[(&a_nxt, &stage[0]), (&b_nxt, &stage[1])]).barrier(smallvec![]);
            ker.push_store(fenced, a_nxt.uop().clone());
            lp.close()
        }
        _ => lp.close(),
    };
    // Each accumulator reads its fully-reduced register value *outside* the loop.
    let final_accs: Vec<RT> = accs.iter().map(|c| c.after(smallvec![ended.clone()])).collect();
    // No copy may be outstanding at exit: drain the last trip's wrapped prefetch
    // before the epilogue writes (threaded through the GLOBAL tile, so the carried
    // accumulators keep their plain post-loop reads).
    let c_gl = if matches!(stream, Stream::Async) {
        let drained = cp_async_wait_all(smallvec![final_accs[0].uop().clone()]);
        c_gl.rewrap(c_gl.uop().after(smallvec![drained]))
    } else {
        c_gl
    };

    // Epilogue: narrow each col-major accumulator to C's dtype in registers, fold
    // in the fusion, then store it at its reg-block coords.
    let nidx = col.mul(&cidx(cfg.blocks_n() as i64)).add(&warp_col);
    let zslab: Idx = match &slab {
        // Split-K writes one `[split_k, M, N]` partial per grid-z (summed by the
        // second pass), so the slab index is the leading axis.
        Some(_) => Idx::from(ker.grid_z()),
        None => Idx::Const(0),
    };
    let out_dt = c_gl.elem().clone();
    let mut c_t = c_gl;
    for (a, c) in final_accs.into_iter().enumerate() {
        let mrow = row.mul(&cidx(cfg.blocks_m() as i64)).add(&acc_row(&warp_row, a, &cfg));
        let ix = MoveIdx::block((Idx::Const(0), zslab.clone(), mrow, nidx.clone()), 2);
        c_t = match &epi {
            Epilogue::Plain => g.store(c_t, narrow(ker, &g, c, &out_dt), ix),
            // The residual is read at the store's own global offset — the same
            // coalesced block the GEMM is about to overwrite — and added **in the
            // output dtype**, which is what the graph's `try_add` over two
            // operand-dtype tensors does. Folding it into the store's pass keeps
            // the narrowed tile in registers (a second pass over it spills).
            Epilogue::Add(res) => {
                let buf = res.uop().clone();
                let c = narrow(ker, &g, c, &out_dt);
                g.store_global_with(c_t, &c, ix, move |v, off| {
                    v.try_add(&load_off(&buf, off.clone())).expect("gemm epilogue: residual add")
                })
            }
            Epilogue::SwiGlu { .. } => g.store(c_t, swiglu(ker, &g, c, &out_dt), ix),
        };
    }
}

/// `y = silu(gate)·up` straight off the f32 accumulator: the gate block is the
/// tile's first `width/2` fragment columns and its matching up block the second
/// half — the paired row order [`Epilogue::SwiGlu`] requires of the fused weight
/// puts them in one wave's registers — so the epilogue writes `reg_n/2` columns
/// and the `[M, 2I]` intermediate is never formed.
///
/// Each accumulator value is rounded to `out_dt` **before** `silu` and the
/// multiply, which is where the graph rounds it (the GEMM output is an operand-
/// dtype tensor that `Tensor::silu` / `try_mul` then read). The pass is unrolled
/// for the reason [`narrow`]'s is: every register index stays constant, so the
/// accumulator never leaves registers.
fn swiglu<'k>(ker: &'k Kernel, g: &Group<'k>, acc: RT<'k>, out_dt: &DType) -> RT<'k> {
    let (h, w) = (acc.shape()[0], acc.shape()[1]);
    assert_eq!(w % 2, 0, "swiglu epilogue: {w} fragment columns do not split into a gate and an up block");
    let half = w / 2;
    let dst = ker.rt((h * acc.base.base.rows, half * acc.base.base.cols), out_dt.clone(), acc.layout, acc.base);
    let (src, sshape) = (g.anchor(acc.uop()), acc.shape().to_vec());
    let rolled = ker.unrolled();
    ker.set_unroll(true);
    let out = g.map(dst, |_, ix| {
        let at = |w: usize| {
            let idx = [ix[0].clone(), col_at(&ix[1], w), ix[2].clone()];
            load_at(&src, &sshape, &idx).cast(out_dt.clone())
        };
        silu(&at(0), out_dt).try_mul(&at(half)).expect("swiglu epilogue: silu(gate)·up")
    });
    ker.set_unroll(rolled);
    out
}

/// A fragment-column index shifted by `w` blocks.
fn col_at(ix: &Idx, w: usize) -> Idx {
    match ix {
        Idx::Const(c) => Idx::Const(c + w as i64),
        Idx::Uop(u) => Idx::Uop(u.add(&cidx(w as i64))),
    }
}

/// `x·sigmoid(x)` in `x`'s dtype, op for op as [`Tensor::silu`] builds it:
/// `sigmoid(x) = 1/(1 + exp2(x·(−1/ln 2)))`, every step in the operand dtype.
fn silu(x: &Arc<UOp>, dt: &DType) -> Arc<UOp> {
    let c = |v: f64| UOp::const_(dt.clone(), ConstValue::Float(v));
    let e = x.try_mul(&c(-1.0 / std::f64::consts::LN_2)).and_then(|s| s.try_exp2()).expect("silu: exp2");
    let sig = UOp::try_reciprocal(&c(1.0).try_add(&e).expect("silu: 1 + exp2")).expect("silu: reciprocal");
    x.try_mul(&sig).expect("silu: x·sigmoid(x)")
}

/// Convert a finished f32 accumulator to the output dtype **in registers**, as an
/// explicitly unrolled copy into a same-layout tile, so the store that follows
/// moves plain elements.
///
/// Storing the f32 tile and letting the store's own cast do it is a 3.4× cliff on
/// NVPTX: an f32→bf16 cast lowers to a 17-instruction integer round-to-nearest-even
/// ([`svod_ir::decompositions`]), and with that body inside the store's rolled
/// `[height, width, inner]` loops LLVM stops unrolling them — the register tile is
/// then indexed dynamically, falls out of registers into local memory, and the
/// whole K-loop pays for it (measured on sm_86: 256 B/lane of spill, 13.7 → 4.0
/// TFLOP/s). Every index here is a constant, so the accumulator stays in registers
/// and only the narrow result reaches the store.
fn narrow<'k>(ker: &'k Kernel, g: &Group<'k>, acc: RT<'k>, out_dt: &DType) -> RT<'k> {
    if acc.elem() == out_dt {
        return acc;
    }
    let dims = (acc.shape()[0] * acc.base.base.rows, acc.shape()[1] * acc.base.base.cols);
    let dst = ker.rt(dims, out_dt.clone(), acc.layout, acc.base);
    let rolled = ker.unrolled();
    ker.set_unroll(true);
    let out = g.copy(dst, &acc);
    ker.set_unroll(rolled);
    out
}

/// How the K strips reach shared memory, and what [`gemm_core`] owes each stream
/// after the trip's MMAs.
enum Stream {
    /// Single-buffered: the caller fences the gathers against the next fill.
    Single,
    /// The `cp.async` pipeline: fenced at the loop top; drained after the loop.
    Async,
    /// The register-staged pipeline: the next strip is in `stage` (per operand)
    /// and is committed into the `nxt` halves after the MMAs, under the trip's
    /// one barrier.
    Staged { stage: [Arc<UOp>; 2], nxt: Box<[ST; 2]> },
}

/// The K-strip stream: everything the fill strategies share (the operand
/// globals, this workgroup's `(pid_m, pid_n)` and K-slab, and the trip count).
struct Strips<'a> {
    cfg: &'a GemmCfg,
    a_gl: &'a GL,
    b_gl: &'a GL,
    row: &'a Arc<UOp>,
    col: &'a Arc<UOp>,
    slab: Option<Arc<UOp>>,
    trips: i64,
}

impl Strips<'_> {
    /// The `(A, B)` global block indices of K-strip `tile` of this workgroup's slab.
    fn at(&self, tile: &Arc<UOp>) -> ([Idx; 4], [Idx; 4]) {
        let t = match &self.slab {
            Some(base) => base.add(tile),
            None => tile.clone(),
        };
        ([Idx::Const(0), Idx::Const(0), Idx::from(self.row), Idx::from(&t)], b_index(self.cfg, self.col, &t))
    }

    /// Single-buffered: one collaborative GLOBAL→LDS fill per trip, the two strips
    /// sharing ONE barrier (the RAW edge) before the gathers; the WAR edge back to
    /// the next fill is the barrier the caller puts after them (`fence = true`).
    fn single(&self, g: &Group<'_>, lp: &Loop<'_>, a_smem: ST, b_smem: ST) -> (ST, ST, Stream) {
        let (a_idx, b_idx) = self.at(lp.index());
        let (a_f, b_f) = if self.cfg.vec_load {
            (
                g.fill_local_vec_nobar(a_smem, self.a_gl.clone(), &a_idx, 2),
                g.fill_local_vec_nobar(b_smem, self.b_gl.clone(), &b_idx, 2),
            )
        } else {
            (
                g.fill_local_nobar(a_smem, self.a_gl.clone(), &a_idx, 2),
                g.fill_local_nobar(b_smem, self.b_gl.clone(), &b_idx, 2),
            )
        };
        // Depending on B's fill makes "after both fills" a graph edge rather than a
        // linearizer accident, and saves the barrier each strip used to close with.
        let filled = a_f.uop().barrier(smallvec![b_f.uop().clone()]);
        (a_f.after(smallvec![filled.clone()]), b_f.after(smallvec![filled]), Stream::Single)
    }

    /// The software pipeline, on the fill primitive the arch has: `cp.async`
    /// where it applies to both strips ([`Group::cp_async_fill_applies`]), else
    /// the two-deep register-staged stream.
    fn pipelined(&self, g: &Group<'_>, lp: &Loop<'_>, a_smem: ST, b_smem: ST) -> (ST, ST, Stream) {
        if g.cp_async_fill_applies(&a_smem, self.a_gl) && g.cp_async_fill_applies(&b_smem, self.b_gl) {
            self.async_pipelined(g, lp, a_smem, b_smem)
        } else {
            assert_eq!(self.cfg.stages, 2, "the register-staged pipeline is two-deep (one strip in flight)");
            self.staged(g, lp, a_smem, b_smem)
        }
    }

    /// The register-staged double buffer (the flash-attention K/V stream on AMD):
    /// the prologue lands strip 0 in half 0; each trip issues the global loads of
    /// strip `tile + 1` into per-lane registers, where they stay in flight under
    /// the current half's gathers and MMAs, and [`gemm_core`] writes them into the
    /// other half after the MMAs, under the trip's single barrier. The prefetch
    /// index wraps modulo the trip count, so the last trip re-reads strip 0
    /// (never gathered) instead of running off the operand.
    fn staged(&self, g: &Group<'_>, lp: &Loop<'_>, a_smem: ST, b_smem: ST) -> (ST, ST, Stream) {
        let half = |st: &ST, p: &Arc<UOp>| st.with_base_offset(p.mul(&cidx(st.half_elems() as i64)));
        let (a0, b0) = self.at(&cidx(0));
        let s_a = g.stage_global_to_reg(&a_smem, self.a_gl, &a0, 2);
        let s_b = g.stage_global_to_reg(&b_smem, self.b_gl, &b0, 2);
        let (a_half, b_half) = (half(&a_smem, &cidx(0)), half(&b_smem, &cidx(0)));
        let landed = g.commit_regs_to_local(&[(&a_half, &s_a), (&b_half, &s_b)]).barrier(smallvec![]);
        g.kernel().push_store(landed.clone(), a_smem.uop().clone());
        let (a_smem, b_smem) = (a_smem.after(&landed), b_smem.after(&landed));

        let idx = lp.index().clone();
        let nxt = idx.add(&cidx(1));
        let par = |t: &Arc<UOp>| t.try_mod(&cidx(2)).expect("stage parity");
        let (par_cur, par_nxt) = (par(&idx), par(&nxt));
        let pf = nxt.try_mod(&cidx(self.trips)).expect("prefetch strip % trips");
        let (ai, bi) = self.at(&pf);
        let stage =
            [g.stage_global_to_reg(&a_smem, self.a_gl, &ai, 2), g.stage_global_to_reg(&b_smem, self.b_gl, &bi, 2)];
        // The gathers order after the issue, so the loads are in flight under them.
        let issued: SmallVec<[Arc<UOp>; 4]> = smallvec![stage[0].clone(), stage[1].clone()];
        let nxt = Box::new([half(&a_smem, &par_nxt), half(&b_smem, &par_nxt)]);
        (
            half(&a_smem, &par_cur).after(issued.clone()),
            half(&b_smem, &par_cur).after(issued),
            Stream::Staged { stage, nxt },
        )
    }

    /// The `cp.async` software pipeline. A prologue issues the first `stages - 1`
    /// strips into their own shared halves; each trip then retires the oldest
    /// outstanding issue (`wait_group` down to the `stages - 2` strips still in
    /// flight, plus one workgroup barrier — covering both the RAW on the strip that
    /// just landed and the WAR on the half every wave finished gathering `stages - 1`
    /// trips ago), issues strip `tile + stages - 1` into the half it frees, and
    /// hands back the current half for the gathers, which run with `stages - 1`
    /// copies in flight. Prefetch indices wrap modulo the trip count, so the tail
    /// trips re-read strip 0 (never gathered) instead of running off the operand.
    fn async_pipelined(&self, g: &Group<'_>, lp: &Loop<'_>, a_smem: ST, b_smem: ST) -> (ST, ST, Stream) {
        let stages = self.cfg.stages as i64;
        let half = |st: &ST, p: &Arc<UOp>| st.with_base_offset(p.mul(&cidx(st.half_elems() as i64)));
        let issue = |a: &ST, b: &ST, t: &Arc<UOp>| {
            let (ai, bi) = self.at(t);
            [g.cp_async_fill(a, self.a_gl, &ai, 2), g.cp_async_fill(b, self.b_gl, &bi, 2)]
        };
        // Prologue: strips `0..stages-1`, each into its own half.
        let mut deps: SmallVec<[Arc<UOp>; 4]> = SmallVec::new();
        for j in 0..stages - 1 {
            deps.extend(issue(&half(&a_smem, &cidx(j)), &half(&b_smem, &cidx(j)), &cidx(j % self.trips)));
        }
        let a_smem = a_smem.after(deps.clone());
        let b_smem = b_smem.after(deps);

        let idx = lp.index().clone();
        let nxt = idx.add(&cidx(stages - 1));
        let par = |t: &Arc<UOp>| t.try_mod(&cidx(stages)).expect("stage parity");
        let (par_cur, par_nxt) = (par(&idx), par(&nxt));
        let pf = nxt.try_mod(&cidx(self.trips)).expect("prefetch strip % trips");

        // `wait_group N` leaves the `stages - 2` newest strips (two commits each —
        // one per operand) in flight and retires everything older, i.e. strip `idx`.
        let landed = cp_async_wait(2 * (stages as u32).saturating_sub(2), smallvec![idx]).barrier(smallvec![]);
        let mut issued: SmallVec<[Arc<UOp>; 4]> = smallvec![landed.clone()];
        issued.extend(issue(&half(&a_smem, &par_nxt).after(&landed), &half(&b_smem, &par_nxt).after(&landed), &pf));
        (half(&a_smem, &par_cur).after(issued.clone()), half(&b_smem, &par_cur).after(issued), Stream::Async)
    }
}

// ── The NT linear-layer kernel (`y = x · wᵀ`) ────────────────────────────────

/// The arches [`gemm_nt`] is enabled for. The core is arch-generic; a family
/// joins with its own measured tile table in [`GemmPolicy::for_arch`], a new
/// part of a known family with an entry here once validated.
pub const GEMM_NT_SUPPORTED_ARCHS: crate::ArchSet = crate::ArchSet::amd(&[svod_dtype::AmdArch::Gfx1151])
    .with_cuda_from(svod_dtype::CudaArch::from_compute_capability(8, 0));

/// The CUDA default tile: 128×64, a 2×2 wave grid (128 threads), two 32×32 f32
/// accumulators per wave, `k_step = 32` and the two-stage `cp.async` pipeline.
/// 24 KiB of shared memory and ~116 registers, so four blocks are resident per
/// sm_86 SM. Measured fastest on every shape in `benches/gemm.rs`: the
/// 128×128 tile (256 threads, 32 KiB) loses 2-14% — its grid is half as wide, and
/// on the narrow-N shapes that costs more than the extra operand reuse gains.
pub const NT_128X64: GemmCfg = GemmCfg {
    block_m: 128,
    block_n: 64,
    warps_m: 2,
    warps_n: 2,
    acc_m: 2,
    k_step: 32,
    stages: 2,
    b_order: BOrder::Nk,
    l2_swizzle: true,
    vec_load: true,
    split_k: 1,
};

/// 64×64, a 2×2 wave grid, one 32×32 accumulator per wave. Half the M edge, so a
/// short-M request (a batch-1 prefill) still covers the device; it also takes an M
/// that only divides by 64. Its smaller warp tile costs ~7% where the default tile's
/// grid is already big enough, so [`GemmPolicy`] picks it only when that grid is not.
pub const NT_64X64: GemmCfg = GemmCfg { block_m: 64, acc_m: 1, ..NT_128X64 };

/// The split-K tile: [`NT_128X64`] over two K-slabs, writing `[2, M, N]` f32
/// partials that a second pass sums.
///
/// **Not selected by [`GemmPolicy`]** — measured a net loss on this card at every
/// shape tried. On the batch-1 `[128, 1024] · [6144, 1024]ᵀ` it does shorten the
/// GEMM itself, exactly as intended (88.3 → 81.9 µs: twice the workgroups over a
/// grid that covered only 96 of the 112 resident-block slots), but the f32 partials
/// it writes and the reduction reads cost 40 µs — 122 µs end to end against 98 µs
/// unsplit. The RTX 3060's 360 GB/s does not pay for an `M·N·split_k` f32 round
/// trip at these sizes; `split_k = 4` is worse still (149 µs). Kept as a measured
/// configuration: a device with more bandwidth per FLOP, or a much smaller `M·N`,
/// flips the sign.
pub const NT_SPLIT_K: GemmCfg = GemmCfg { split_k: 2, l2_swizzle: false, ..NT_128X64 };

/// The CUDA sm_80+ tiles, widest first.
pub const CUDA_TILES: [GemmCfg; 2] = [NT_128X64, NT_64X64];

/// The RDNA (wave32 WMMA) tiles, measured on gfx1151: the CUDA tiles on the
/// register-staged pipeline without the L2 swizzle (single-XCD parts; ±3%
/// either way), and the fine tile on a 64-deep strip, which halves the barriers
/// per K and wins once the grid is short (batch-1 down projection 12.3 vs 7.1
/// TFLOP/s). A 32-deep 64×64 tile keeps a `K` of 64 or 96 servable. 128×128
/// trailed by 10-15% and `k_step = 64` on the wide tile halved its throughput
/// (48 KiB of LDS), so neither is a candidate.
pub const RDNA_TILES: [GemmCfg; 3] = [
    GemmCfg { l2_swizzle: false, ..NT_128X64 },
    GemmCfg { l2_swizzle: false, k_step: 64, ..NT_64X64 },
    GemmCfg { l2_swizzle: false, ..NT_64X64 },
];

/// Tile selection for the NT GEMM: the family's tile table — the search space
/// [`Self::tuned`] measures on first use — and, for the static choice, the
/// device's compute-unit count, which sets how small a launch grid counts as
/// starving the machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemmPolicy {
    /// Compute units (SMs, CUs) on the device.
    pub compute_units: usize,
    /// The tiles to choose between: the widest first, then the finer ones in
    /// preference order; empty where no one has measured the family (the policy
    /// then declines every shape rather than run another family's constants).
    pub tiles: &'static [GemmCfg],
    /// Blocks of the widest tile per compute unit below which a finer tile wins.
    /// On CUDA the blocks resident per SM (~116 registers and 24 KiB of shared
    /// memory against 64 K and 100 KiB); on RDNA measured (the wide tile leads
    /// at 12.8 blocks per CU and trails at 6.4).
    pub resident: usize,
}

impl GemmPolicy {
    /// The family's tile table with its measured part's compute-unit count (an
    /// RTX 3060's 28 SMs, Strix Halo's 40 CUs); [`Self::for_device`] reads the
    /// real count. A family nobody measured declines.
    pub fn for_arch(arch: svod_dtype::GpuArch) -> Self {
        match crate::arch::Family::of(arch) {
            crate::arch::Family::Cuda => Self { compute_units: 28, tiles: &CUDA_TILES, resident: 4 },
            crate::arch::Family::Rdna => Self { compute_units: 40, tiles: &RDNA_TILES, resident: 8 },
            crate::arch::Family::Cdna | crate::arch::Family::Metal => {
                Self { compute_units: 1, tiles: &[], resident: 1 }
            }
        }
    }

    /// [`Self::for_arch`] with the compute-unit count of the device behind
    /// `spec`, when the backend reports it.
    pub fn for_device(spec: &svod_dtype::DeviceSpec, arch: svod_dtype::GpuArch) -> Self {
        let mut policy = Self::for_arch(arch);
        if let Some(compute_units) = crate::target::compute_units(spec) {
            policy.compute_units = compute_units;
        }
        policy
    }

    /// The tile for an `m × k × n` NT GEMM, or `None` when none tiles it exactly
    /// (the caller pads, or falls back). The widest tile unless its grid would not
    /// even fill the device once, in which case it is tried last — the finer
    /// tiles also cover an `m` the widest does not divide.
    pub fn cfg(&self, m: usize, k: usize, n: usize) -> Option<GemmCfg> {
        let (widest, finer) = self.tiles.split_first()?;
        let starved = widest.blocks(m, n) < self.compute_units * self.resident;
        let mut table: SmallVec<[GemmCfg; 4]> = finer.iter().copied().collect();
        if starved {
            table.push(*widest)
        } else {
            table.insert(0, *widest)
        }
        table.into_iter().find(|cfg| cfg.tiles(m, k, n))
    }

    /// The tile for an `m × k × n` NT GEMM under `epi` as measured on this device
    /// ([`crate::tune`]): every table tile that tiles the shape and carries the
    /// epilogue is timed once, with that epilogue, on synthetic operands, and
    /// the fastest kept in `store`; the static [`Self::cfg`] choice where only
    /// one fits or nothing measured. The launch entry consults
    /// [`crate::tune::enabled`] before coming here.
    pub fn tuned(
        &self,
        store: &crate::tune::TuneStore,
        spec: &svod_dtype::DeviceSpec,
        arch: svod_dtype::GpuArch,
        dtype: &DType,
        (m, k, n): (usize, usize, usize),
        epi: Epilogue<()>,
    ) -> Option<GemmCfg> {
        let caps = crate::ArchCaps::for_arch(arch);
        let frag = caps.frag(crate::arch::FragRole::Accumulator).map(|f| f.base.cols);
        let fits = |cfg: &GemmCfg| cfg.tiles(m, k, n) && cfg.carries(epi, frag);
        let candidates: Vec<GemmCfg> = self.tiles.iter().copied().filter(fits).collect();
        let fallback = || self.cfg(m, k, n).filter(fits);
        if candidates.len() < 2 {
            return fallback();
        }
        let cols = epi.out_cols(n);
        let build = move |ker: &Kernel, cfg: GemmCfg| {
            build_gemm_nt(ker, (m, k, n), cfg, dtype.clone(), dtype.clone(), epi);
            ker.finish(cfg.acc_m)
        };
        // The key covers the candidate kernels themselves: their graphs, built
        // against placeholder buffers, fingerprinted in table order.
        let placeholders = || {
            let mut sizes = vec![m * cols, m * k, n * k];
            if let Epilogue::Add(()) = epi {
                sizes.push(m * cols);
            }
            sizes.into_iter().map(|size| UOp::new_buffer(svod_dtype::DeviceSpec::Cpu, size, dtype.clone())).collect()
        };
        let builds: Vec<u128> = candidates
            .iter()
            .map(|&cfg| {
                let ker =
                    Kernel::new("gemm_nt", cfg.grid_dims(m, n), cfg.threads(caps.wave_size), placeholders(), caps);
                crate::kernel_fingerprint(&build(&ker, cfg)).digest
            })
            .collect();
        let shape = [m, k, n, dtype.bytes(), epi.code()];
        let key = crate::tune::TuneKey::new("gemm_nt", spec, arch, &shape, &builds);
        let compile = |i: usize| {
            let cfg = candidates[i];
            let operand = |shape: &[usize]| Tensor::randn(shape).ok().map(|t| t.cast(dtype.clone()).to(spec.clone()));
            let (x, w) = (operand(&[m, k])?, operand(&[n, k])?);
            let mut ins = vec![x, w];
            if let Epilogue::Add(()) = epi {
                ins.push(operand(&[m, cols])?);
            }
            let ins: Vec<&Tensor> = ins.iter().collect();
            let mut y = Tensor::empty(&[m, cols], dtype.clone()).to(spec.clone());
            let (grid, block) = (cfg.grid_dims(m, n), cfg.threads(caps.wave_size));
            crate::launch::compile_kernel("gemm_nt_tune", grid, block, &mut [&mut y], &ins, move |ker| build(ker, cfg))
                .ok()
        };
        store.select(&key, candidates.len(), compile).map(|i| candidates[i]).or_else(fallback)
    }

    /// The gate/up row-block width an [`Epilogue::SwiGlu`] fused weight must be
    /// laid out in: `reg_n/2` — half a wave's N tile, the two halves its
    /// accumulator holds side by side — common to every tile in the table, or
    /// `None` when they disagree or the table is empty (the caller then keeps a
    /// separate SwiGLU pass). The GEMM's `M` is not known when the weight is
    /// loaded, and `M` is what picks the tile, so the row arrangement has to be
    /// one that **every** candidate tile reads.
    pub fn swiglu_pair_width(&self) -> Option<usize> {
        let pair = self.tiles.first()?.reg_n() / 2;
        self.tiles.iter().all(|cfg| cfg.reg_n() / 2 == pair).then_some(pair)
    }
}

/// [`GemmPolicy::swiglu_pair_width`] for the device behind `spec` — `None` off
/// the supported arches, where the weight stays plainly stacked. The row order
/// is fixed when the weight is loaded, so a model loaded on the host and moved
/// to a GPU afterwards keeps the plain stacking (and the separate SwiGLU pass).
pub fn swiglu_pair_width(spec: &svod_dtype::DeviceSpec) -> Option<usize> {
    let arch = crate::target::resolve_supported_arch(spec, GEMM_NT_SUPPORTED_ARCHS).ok()?;
    GemmPolicy::for_arch(arch).swiglu_pair_width()
}

/// [`GemmPolicy::cfg`] for the CUDA table on its measured 28-SM part — the
/// device-free predicate the applicability tests and [`gemm_nt`]'s docs are
/// written against.
pub fn select_cfg(m: usize, k: usize, n: usize) -> Option<GemmCfg> {
    GemmPolicy::for_arch(svod_dtype::GpuArch::Cuda(svod_dtype::CudaArch::from_compute_capability(8, 6))).cfg(m, k, n)
}

/// **Graph-native** `y[M, N] = x[M, K] · w[N, K]ᵀ` — the linear-layer GEMM, the
/// matmul peer of [`crate::flash_attention_with`]. Returns a lazy output
/// [`Tensor`] (a `custom_kernel` / `Op::Call` node) that composes into a model
/// graph and realizes through the normal `prepare()` path.
///
/// `x` is `[lead..., K]` of any rank ≥ 2 (the leading dims are the rows, so a
/// `[B, L, K]` activation needs no reshape and a realized one is bound without a
/// copy), `w` is `[N, K]`; both statically shaped and **bf16 or f16** (the
/// matrix-core operand dtypes). `y` is `[lead..., N]` in the same dtype, cast
/// from the f32 accumulators inside the kernel's epilogue — there is no f32 round
/// trip through memory.
///
/// The outcome is three-way (via [`crate::launch_custom`]):
///
/// - `Ok(None)` — *doesn't apply here:* the device is not one of
///   [`GEMM_NT_SUPPORTED_ARCHS`] with its LLVM backend, **or** no tile of its
///   table covers the shape ([`GemmPolicy::cfg`]): `M` and `N` must be multiples
///   of 64, `K` a multiple of the 32-wide strip and at least 64 (two strips, one
///   per pipeline stage). The caller pads to 128 or substitutes `Tensor::linear`.
///
/// The tile is the one measured fastest on this device for the shape
/// ([`GemmPolicy::tuned`]): the first request of a shape times every candidate
/// once and caches the winner on disk ([`crate::tune`]); `SVOD_TK_TUNE=0` keeps
/// the table's static choice instead.
/// - `Err` — *malformed request:* a symbolic dim, `x` below rank 2 or `w` not
///   rank 2, a dtype outside {bf16, f16}, a dtype mismatch between `x` and `w`,
///   or `w`'s K disagreeing with `x`'s.
/// - `Ok(Some(y))` — it ran.
///
/// ```no_run
/// use svod_dtype::DType;
/// use svod_tensor::Tensor;
/// let x = Tensor::randn(&[1024, 1024]).unwrap().cast(DType::BFloat16);
/// let w = Tensor::randn(&[4096, 1024]).unwrap().cast(DType::BFloat16);
/// if let Some(mut y) = svod_tk::gemm_nt(&x, &w).unwrap() {
///     y.prepare().unwrap();
/// }
/// ```
pub fn gemm_nt(x: &Tensor, w: &Tensor) -> crate::LaunchResult<Option<Tensor>> {
    gemm_nt_with_epilogue(x, w, Epilogue::Plain)
}

/// [`gemm_nt`] with an explicit tile chooser — the entry point a bench or a
/// caller with its own measured table uses. `cfg` receives `(m, k, n)` and returns
/// the tile, or `None` to decline (`Ok(None)`).
pub fn gemm_nt_with(
    x: &Tensor,
    w: &Tensor,
    cfg: impl Fn(usize, usize, usize) -> Option<GemmCfg> + Copy,
) -> crate::LaunchResult<Option<Tensor>> {
    build_gemm(x, w, Epilogue::Plain, move |_, m, k, n| cfg(m, k, n))
}

/// [`gemm_nt`] with `epilogue` folded into its store, so the fused value is the
/// only one that reaches memory:
///
/// - [`Epilogue::Add`] gives `y = x·wᵀ + residual` — a pre-norm decoder layer's
///   residual add rides the projection that produced it, and the norm that reads
///   the stream is then the two-pass [`rms_norm`](crate::rms_norm) instead of the
///   four-pass [`add_rms_norm`](crate::add_rms_norm).
/// - [`Epilogue::SwiGlu`] gives `y = silu(gate)·up` off a fused gate/up weight,
///   so the `[M, 2I]` intermediate is never written and no separate SwiGLU pass
///   reads it back.
///
/// Shapes, dtypes and the three-way outcome are [`gemm_nt`]'s, plus:
///
/// - `Ok(None)` — the selected tile cannot carry the epilogue: split-K (its
///   store is f32 partials), or, for SwiGLU, a tile whose `reg_n/2` is not the
///   `pair` width the weight rows were arranged in (see
///   [`GemmPolicy::swiglu_pair_width`]).
/// - `Err` — a `residual` whose shape or dtype is not `y`'s, or a SwiGLU `pair`
///   that does not divide `N/2`.
pub fn gemm_nt_with_epilogue(
    x: &Tensor,
    w: &Tensor,
    epilogue: Epilogue<&Tensor>,
) -> crate::LaunchResult<Option<Tensor>> {
    let (spec, dtype, kind) = (x.device(), x.uop().dtype(), epilogue.kind());
    build_gemm(x, w, epilogue, |arch, m, k, n| {
        let policy = GemmPolicy::for_device(&spec, arch);
        if !crate::tune::enabled() {
            let frag = crate::ArchCaps::for_arch(arch).frag(crate::arch::FragRole::Accumulator);
            return policy.cfg(m, k, n).filter(|c| c.carries(kind, frag.map(|f| f.base.cols)));
        }
        policy.tuned(crate::tune::TuneStore::global(), &spec, arch, &dtype, (m, k, n), kind)
    })
}

/// The shared launcher body of the three entries above.
fn build_gemm(
    x: &Tensor,
    w: &Tensor,
    epi: Epilogue<&Tensor>,
    cfg: impl Fn(svod_dtype::GpuArch, usize, usize, usize) -> Option<GemmCfg> + Copy,
) -> crate::LaunchResult<Option<Tensor>> {
    let xd = crate::launch::concrete_dims_at_least(x, "gemm-nt", "x", 2)?;
    let wd = crate::launch::concrete_dims(w, "gemm-nt", "w", 2)?;
    // `x` is `[lead..., K]`: the leading dims are the GEMM's rows and come back
    // on `y` as `[lead..., N]`.
    let (lead, k) = (xd[..xd.len() - 1].to_vec(), xd[xd.len() - 1]);
    let (m, n) = (lead.iter().product::<usize>(), wd[0]);
    let dtype = x.uop().dtype();
    let (w_dtype, kw) = (w.uop().dtype(), wd[1]);
    let err_dtype = dtype.clone();

    // The epilogue's own operand: its dims are resolved up front (a symbolic one
    // is an `Err` like any other operand's), the rest is checked by `validate`.
    let kind = epi.kind();
    let y_shape: Vec<usize> = lead.iter().copied().chain([kind.out_cols(n)]).collect();
    let res_dims = match epi {
        Epilogue::Add(r) => Some(crate::launch::concrete_dims_at_least(r, "gemm-nt", "residual", 2)?),
        _ => None,
    };
    let res_dtype = match epi {
        Epilogue::Add(r) => Some(r.uop().dtype()),
        _ => None,
    };
    let want_res = y_shape.clone();

    crate::launch_custom(
        &x.device(),
        GEMM_NT_SUPPORTED_ARCHS,
        move |_arch| {
            ensure!(
                err_dtype == DType::BFloat16 || err_dtype == DType::Float16,
                crate::launch::DtypeSnafu { kernel: "gemm-nt", got: err_dtype, expected: "bf16 or f16" }
            );
            ensure!(
                w_dtype == err_dtype,
                crate::launch::DtypeSnafu { kernel: "gemm-nt", got: w_dtype, expected: "the dtype of x" }
            );
            ensure!(
                kw == k,
                crate::launch::OperandShapeSnafu { kernel: "gemm-nt", operand: "w", expected: vec![n, k], got: wd }
            );
            if let Some(got) = res_dims {
                ensure!(
                    res_dtype == Some(err_dtype.clone()),
                    crate::launch::DtypeSnafu {
                        kernel: "gemm-nt",
                        got: res_dtype.unwrap_or(err_dtype),
                        expected: "the dtype of x"
                    }
                );
                ensure!(
                    got == want_res,
                    crate::launch::OperandShapeSnafu {
                        kernel: "gemm-nt",
                        operand: "residual",
                        expected: want_res,
                        got
                    }
                );
            }
            if let Epilogue::SwiGlu { pair } = kind {
                ensure!(
                    pair > 0 && n.is_multiple_of(2 * pair),
                    crate::launch::DimMultipleSnafu { kernel: "gemm-nt", dim: "N", value: n, multiple: 2 * pair }
                );
            }
            Ok(())
        },
        move |arch| {
            let frag = crate::ArchCaps::for_arch(arch).frag(crate::arch::FragRole::Accumulator);
            cfg(arch, m, k, n).is_some_and(|c| c.carries(kind, frag.map(|f| f.base.cols)))
        },
        move |arch| {
            let caps = crate::ArchCaps::for_arch(arch);
            let cfg = cfg(arch, m, k, n).expect("checked by the tiling predicate");
            let (grid, block) = (cfg.grid_dims(m, n), cfg.threads(caps.wave_size));
            let (in_dt, split) = (dtype.clone(), cfg.split_k);
            let out_dt = if split > 1 { DType::Float32 } else { dtype.clone() };
            let out = Tensor::empty(&if split > 1 { vec![split, m, n] } else { y_shape.clone() }, out_dt.clone());
            let name = match kind {
                Epilogue::Add(()) => "gemm_nt_add",
                Epilogue::SwiGlu { .. } => "gemm_nt_swiglu",
                Epilogue::Plain if split > 1 => "gemm_nt_split",
                Epilogue::Plain => "gemm_nt",
            };
            let mut ins = vec![x, w];
            if let Epilogue::Add(r) = epi {
                ins.push(r);
            }
            let y = crate::graph_launch(name, grid, block, out, &ins, caps, move |ker| {
                build_gemm_nt(ker, (m, k, n), cfg, in_dt, out_dt, kind);
                ker.finish(cfg.acc_m)
            })?;
            if split == 1 {
                return Ok(y);
            }
            // The partials stay f32, so the reduction adds no rounding beyond the
            // single final cast the non-split path also does.
            let y_shape: Vec<isize> = y_shape.iter().map(|&d| d as isize).collect();
            y.sum(0isize)
                .and_then(|s| s.cast(dtype).contiguous().try_reshape(y_shape))
                .context(crate::launch::OperandSnafu)
        },
    )
}

/// Bind the NT ABI (`y` out; `x[M, K]`, `w[N, K]`, and under [`Epilogue::Add`] a
/// trailing `residual[M, N]`, in) and run [`gemm_core`]. `out_dt` is the output
/// buffer's dtype — the operand dtype for the direct path, f32 for the
/// `[split_k, M, N]` partials.
pub fn build_gemm_nt(
    ker: &Kernel,
    (m, k, n): (usize, usize, usize),
    cfg: GemmCfg,
    in_dt: DType,
    out_dt: DType,
    epi: Epilogue<()>,
) {
    let cols = epi.out_cols(n);
    let out_shape = if cfg.split_k > 1 { vec![1, cfg.split_k, m, cols] } else { vec![1, 1, m, cols] };
    let mut in_specs = vec![GlSpec::new(&[1, 1, m, k], in_dt.clone()), GlSpec::new(&[1, 1, n, k], in_dt.clone())];
    if let Epilogue::Add(()) = epi {
        in_specs.push(GlSpec::new(&[1, 1, m, cols], out_dt.clone()));
    }
    let (outs, ins) = ker.bind_abi(&[GlSpec::new(&out_shape, out_dt)], &in_specs);
    let epi = match epi {
        Epilogue::Plain => Epilogue::Plain,
        Epilogue::Add(()) => Epilogue::Add(ins[2].clone()),
        Epilogue::SwiGlu { pair } => Epilogue::SwiGlu { pair },
    };
    gemm_core(ker, (m, k, n), cfg, outs[0].clone(), ins[0].clone(), ins[1].clone(), epi);
}

// ── The square matmul (`c = a · b`, n×n) ─────────────────────────────────────
//
// A thin wrapper over `gemm_core`: the square K-major ABI plus the per-arch
// `MatmulCfg` tables, which are the AMD- and Metal-validated tuning the NT
// configs above have no counterpart for. HK ships separate gfx942/gfx950/gfx1250
// kernels for the same reason — arch-specific peak tuning lives in the config.

/// K-reduction step (the LDS strip depth, shared by every config). HK `GEMM:6`.
pub const K_STEP: usize = 64;

/// Block / wave geometry of a multi-wave matmul (HK `GEMM:5-8,67-68`): a
/// `wave_rows × wave_cols`-wave workgroup computes a `block × block` C tile,
/// each wave owning `n_accum` col-major `reg × reg` f32 accumulators
/// (`reg = block / wave_cols`) reduced over K in [`K_STEP`]-wide steps. The
/// `wave_cols = wave_rows * n_accum` invariant keeps `reg` square: the M side is
/// split into `wave_rows * n_accum` row-blocks, the N side into `wave_cols`.
#[derive(Clone, Copy)]
pub struct MatmulCfg {
    /// The square C-tile edge (in elements) one workgroup computes.
    pub block: usize,
    /// Wave grid rows — the M side splits into `wave_rows * n_accum` row-blocks.
    pub wave_rows: usize,
    /// Wave grid columns — the N side splits into `wave_cols` col-blocks.
    pub wave_cols: usize,
    /// `reg × reg` f32 accumulators per wave.
    pub n_accum: usize,
    /// Drive `(pid_m, pid_n)` from a flattened 1-D grid via the chiplet/L2
    /// [`l2_swizzle`](crate::grid::l2_swizzle) instead of the plain 2-D
    /// `block_idx`. Grid becomes `[grid² , 1, 1]`.
    pub l2_swizzle: bool,
    /// Fill the GLOBAL→LDS strips with 128-bit (`vec8` bf16) coalesced loads
    /// instead of the scalar/`vec4`-folded path (16-byte `cp.async` copies on CUDA).
    pub vec_load: bool,
    /// K-reduction step (LDS strip depth) for the single-buffered K-loop. Must be a
    /// multiple of 16 (the WMMA K-edge) and divide N. Lowering it cuts the live
    /// operand VGPR/lane (each WMMA input replicates all `k_step`/16 K-sub-steps),
    /// raising occupancy — the dominant occupancy lever on RDNA3.5/wave32
    /// ([`GFX1151_CFG`] uses 32). gfx942 keeps [`K_STEP`]
    /// (64). `0` means "use [`K_STEP`]" so older literal/`..M1_CFG` builders that
    /// predate the field still get the default — see [`MatmulCfg::k_step`].
    pub k_step: usize,
}

impl MatmulCfg {
    /// The per-accumulator square edge (`block / wave_cols`).
    pub const fn reg(&self) -> usize {
        self.block / self.wave_cols
    }
    /// The K-reduction step, resolving the `0` sentinel (older literal builders) to
    /// the default [`K_STEP`]. The resolved value must be a multiple of 16 (the WMMA
    /// K-edge) and divide N; a violation panics in [`gemm_core`].
    pub const fn k_step(&self) -> usize {
        if self.k_step == 0 { K_STEP } else { self.k_step }
    }
    /// `reg`-blocks per C-tile side (= `wave_cols` = `wave_rows * n_accum`); the
    /// grid→C-block coordinate multiplier.
    pub const fn blocks_per_side(&self) -> usize {
        self.block / self.reg()
    }
    /// Launch block size (threads) = `wave_rows * wave_cols * wave_size`.
    pub const fn threads(&self, wave_size: usize) -> i64 {
        (self.wave_rows * self.wave_cols * wave_size) as i64
    }
    /// Grid edge (`n / block`).
    pub const fn grid(&self, n: usize) -> i64 {
        (n / self.block) as i64
    }
    /// Launch grid for a general `m × n` C: a flattened 1-D `[gm·gn, 1, 1]` when
    /// the chiplet swizzle ([`l2_swizzle`]) is on (it re-derives `(pid_m, pid_n)`), else
    /// the plain 2-D `[gn, gm, 1]` (x = n-blocks → `block_idx[0]` = pid_n, y = m-blocks
    /// → `block_idx[1]` = pid_m — matching [`block_coords`]).
    pub const fn grid_dims_mn(&self, m: usize, n: usize) -> [i64; 3] {
        let (gm, gn) = ((m / self.block) as i64, (n / self.block) as i64);
        if self.l2_swizzle { [gm * gn, 1, 1] } else { [gn, gm, 1] }
    }
    /// Square convenience: [`grid_dims_mn`] with `m = n` (the `[grid², 1, 1]` /
    /// `[grid, grid, 1]` the square matmul launches with).
    pub const fn grid_dims(&self, n: usize) -> [i64; 3] {
        self.grid_dims_mn(n, n)
    }
    /// This square K-major config as the general [`GemmCfg`] [`gemm_core`] takes:
    /// a square block, `B[k, n]`, and the single-buffered (`stages = 1`) K-loop.
    pub const fn gemm(&self, k_step: usize) -> GemmCfg {
        GemmCfg {
            block_m: self.block,
            block_n: self.block,
            warps_m: self.wave_rows,
            warps_n: self.wave_cols,
            acc_m: self.n_accum,
            k_step,
            stages: 1,
            b_order: BOrder::Kn,
            l2_swizzle: self.l2_swizzle,
            vec_load: self.vec_load,
            split_k: 1,
        }
    }
}

/// 8-wave (2×4) 256×256 block, two 64×64 accumulators/wave, 512
/// threads, the chiplet/L2 grid swizzle, and 128-bit vectorized LDS fills.
pub const M1_CFG: MatmulCfg =
    MatmulCfg { block: 256, wave_rows: 2, wave_cols: 4, n_accum: 2, l2_swizzle: true, vec_load: true, k_step: K_STEP };
/// Small-N: single-warp 64×64 block, one 64×64 accumulator, 64 threads — the
/// grid is `(n/64)²` workgroups, ~16× the large-N config's at a given N, so a small N keeps the
/// 304-CU machine fed instead of collapsing to a handful of 256×256 blocks.
/// Keeps the plain 2-D grid + scalar fill (the swizzle/vec wins are large-N).
pub const SMALL_CFG: MatmulCfg =
    MatmulCfg { block: 64, wave_rows: 1, wave_cols: 1, n_accum: 1, l2_swizzle: false, vec_load: false, k_step: K_STEP };

/// gfx1151 (RDNA3.5, wave32) config: 64×64 block, 2×2
/// waves (4 waves / 128 threads), ONE
/// 32×32 accumulator/wave, 128-bit vec fills, no L2 swizzle (single-XCD APU), and
/// **`k_step = 32`**. The `reg=32` tile keeps accumulator VGPR ≈ 32/lane; the
/// `k_step=32` halves the live WMMA-input fragment VGPR vs the default 64 (the input
/// replicates all `k_step`/16 K-sub-steps per lane), raising occupancy. `k_step` is
/// the dominant occupancy lever on RDNA3.5/wave32; the single-buffered path has no
/// memory stall a double buffer could hide. gfx942 keeps `k_step = K_STEP` (64). A
/// smaller `k_step` lowers the WMMA-input VGPR but adds barriers, so the tuned value
/// trades occupancy against barrier overhead.
pub const GFX1151_CFG: MatmulCfg =
    MatmulCfg { block: 64, wave_rows: 2, wave_cols: 2, n_accum: 1, l2_swizzle: false, vec_load: true, k_step: 32 };

/// CUDA sm_80+ (`mma.sync`, warp32) config: 128×128 block, 2×4 waves (256 threads),
/// two 32×32 accumulators/wave, 128-bit vec fills, `k_step = 32`. The 8-register
/// two-half fragments make register pressure the lever: a 32×32 f32 accumulator
/// is 32 regs/lane and each 32×32 operand strip 64 packed halves, so `reg = 32`
/// and the short strip keep a warp well under the 255-register ceiling; the
/// 2×(128×32) bf16 strips are 16 KiB of the 48 KiB static shared budget.
pub const SM80_CFG: MatmulCfg =
    MatmulCfg { block: 128, wave_rows: 2, wave_cols: 4, n_accum: 2, l2_swizzle: false, vec_load: true, k_step: 32 };
/// CUDA sm_80+ small-N config (N a multiple of 64 but not 128): 64×64 block, 2×2
/// waves, one 32×32 accumulator/wave.
pub const SM80_SMALL_CFG: MatmulCfg =
    MatmulCfg { block: 64, wave_rows: 2, wave_cols: 2, n_accum: 1, l2_swizzle: false, vec_load: true, k_step: 32 };

/// Apple7+ (`simdgroup_matrix`, SIMD-group 32) config: 64x64 block, 2x2 waves
/// (128 threads), one 32x32 accumulator/wave, `k_step = 16`. `vec_load` is **off**:
/// the 128-bit fill is an 8-lane bf16 vector and MSL has no vector wider than 4
/// lanes. The two 64x16 bf16 strips are 4 KiB of Apple's 32 KiB threadgroup budget.
///
/// `k_step = 16` is measured, not inherited: at N=4096 the 8/16/32/64/128 sweep gives
/// 11.11 / **11.63** / 6.95 / 8.36 / 4.66 TFLOP/s. Each WMMA operand replicates all
/// `k_step / K` sub-steps per lane, so on Apple's 2-elements-per-lane fragment the
/// live operand registers — and with them occupancy — move much faster with `k_step`
/// than on a 16x16 fragment. Inheriting the AMD/CUDA `32` cost 1.67x.
pub const METAL_CFG: MatmulCfg =
    MatmulCfg { block: 64, wave_rows: 2, wave_cols: 2, n_accum: 1, l2_swizzle: false, vec_load: false, k_step: 16 };

/// Size-adaptive config selection: small N (where the 256×256/8-wave grid
/// starves the machine) uses [`SMALL_CFG`]; everything else keeps [`M1_CFG`].
/// Small N uses an occupancy-tuned config; the threshold follows size-adaptive tuning.
pub fn cfg_for_n(n: usize) -> MatmulCfg {
    if n <= 768 && n.is_multiple_of(SMALL_CFG.block) { SMALL_CFG } else { M1_CFG }
}

/// Per-arch config: gfx1151 (RDNA3.5 wave32) uses the occupancy-tuned
/// [`GFX1151_CFG`]; CUDA the register-pressure-tuned [`SM80_CFG`] (or
/// [`SM80_SMALL_CFG`] when N only tiles by 64); gfx942 (CDNA wave64) keeps the
/// size-adaptive [`cfg_for_n`]. Arch-specific peak tuning lives here (the generic
/// optimizer stays generic); this is the tk peer of HK shipping separate
/// gfx942/gfx950/gfx1250 kernels.
pub fn cfg_for_arch(arch: svod_dtype::GpuArch, n: usize) -> MatmulCfg {
    match arch {
        svod_dtype::GpuArch::Amd(svod_dtype::AmdArch::Gfx1151) if n.is_multiple_of(GFX1151_CFG.block) => GFX1151_CFG,
        svod_dtype::GpuArch::Cuda(_) if n.is_multiple_of(SM80_CFG.block) => SM80_CFG,
        svod_dtype::GpuArch::Cuda(_) => SM80_SMALL_CFG,
        svod_dtype::GpuArch::Metal(_) => METAL_CFG,
        _ => cfg_for_n(n),
    }
}

/// The GPU arch(es) the tile matmul is built for: gfx942 (CDNA MFMA, wave64),
/// gfx1151 (RDNA3.5 WMMA, wave32 — the `_W32_*` fragment shapes) and CUDA sm_80+
/// (`mma.sync.m16n8k16`, warp32 — the two-half `RT_16X16_MMA` fragment). The
/// launcher gates against this; see [`crate::target::check_target`]. Validated on
/// gfx942 (CDNA3), gfx1151 (RDNA3.5) and sm_86 (Ampere) — gfx942 before the
/// vector LDS gathers (PR #177), not re-run since.
pub const MATMUL_SUPPORTED_ARCHS: crate::ArchSet =
    crate::ArchSet::amd(&[svod_dtype::AmdArch::Gfx942, svod_dtype::AmdArch::Gfx1151])
        .with_cuda_from(svod_dtype::CudaArch::from_compute_capability(8, 0))
        .with_metal_from(svod_dtype::MetalFamily::Apple(7));

/// **Graph-native** `n×n` matrix multiply — returns a lazy output [`Tensor`] (a
/// `custom_kernel` / `Op::Call` node), the matmul peer of [`crate::flash_attention`].
/// Composes into a model graph and realizes / benchmarks through the normal
/// `prepare()` → `execute_profiled` path like any other tensor op.
///
/// `a`/`b` are square `[n, n]` of **any float dtype**: they are cast to bf16
/// internally (the kernel is a bf16-input matrix-engine GEMM), and the result is
/// the f32 WMMA/MFMA accumulator. So a caller needs no kernel knowledge — pass
/// plain tensors, get a tensor back. The per-arch occupancy config is picked by
/// [`cfg_for_arch`].
///
/// Like [`crate::flash_attention_with`], the outcome is three-way (via
/// [`crate::launch_custom`]): `Ok(None)` when the device can't run the kernel,
/// `Err` when the request is malformed (an operand that isn't a statically-shaped
/// rank-2 tensor, non-square operands, or a size that isn't a multiple of the arch's
/// block), `Ok(Some)` when it ran.
///
/// ```no_run
/// use svod_tensor::Tensor;
/// let a = Tensor::randn(&[256, 256]).unwrap();
/// let b = Tensor::randn(&[256, 256]).unwrap();
/// if let Some(mut c) = svod_tk::matmul(&a, &b).unwrap() { // lazy bf16→f32 GEMM node
///     c.prepare().unwrap();                                // realize through the scheduler
/// }
/// ```
pub fn matmul(a: &Tensor, b: &Tensor) -> crate::LaunchResult<Option<Tensor>> {
    let ad = crate::launch::concrete_dims(a, "matmul", "a", 2)?;
    let bd = crate::launch::concrete_dims(b, "matmul", "b", 2)?;
    let (am, an) = (ad[0], ad[1]);
    let (bm, bn) = (bd[0], bd[1]);
    let n = am;

    crate::launch_custom(
        &a.device(),
        MATMUL_SUPPORTED_ARCHS,
        // Operands must be square + equal-sized; `n % block` (arch-dependent) is checked
        // in `build`. Both are structural request errors (`Err`), not fallback triggers.
        move |_arch| {
            ensure!(
                an == am && bm == am && bn == am,
                crate::launch::NotSquareSnafu { kernel: "matmul", a: [am, an], b: [bm, bn] }
            );
            Ok(())
        },
        |_| true, // no runtime-applicability fallback — a bad size is an error, not `None`.
        move |arch| {
            let caps = crate::ArchCaps::for_arch(arch);
            let cfg = cfg_for_arch(arch, n);
            ensure!(
                n % cfg.block == 0,
                crate::launch::DimMultipleSnafu { kernel: "matmul", dim: "n", value: n, multiple: cfg.block }
            );
            // Operands → bf16 (the matrix-engine operand dtype); a no-op when already
            // bf16, so the ABI's bf16 globals bind directly. Output stays f32 (accumulator).
            let a_bf = a.cast(DType::BFloat16);
            let b_bf = b.cast(DType::BFloat16);
            let out = Tensor::empty(&[n, n], DType::Float32);
            crate::graph_launch(
                "matmul",
                cfg.grid_dims(n),
                cfg.threads(caps.wave_size),
                out,
                &[&a_bf, &b_bf],
                caps,
                move |ker| {
                    build_matmul_cfg(ker, n, cfg);
                    ker.finish(cfg.n_accum)
                },
            )
        },
    )
}

/// The parametrized multi-wave matmul. One `cfg.block × cfg.block` C
/// tile per workgroup, `cfg.n_accum` col-major `reg × reg` accumulators/wave
/// reduced over a tracked K-loop; each wave streams its A-strip rows and shared
/// B-strip cols out of XOR-swizzled LDS. A single `END` closes the K-loop around
/// the last accumulator's store; the rest stay scoped inside it by chaining
/// their A-inputs through the prior accumulator's MFMA (a `RANGE` admits one
/// `END`). The epilogue stores each accumulator to global C at its `reg`-block.
///
/// # Panics
/// Panics on the same preconditions as [`gemm_core`].
pub fn build_matmul_cfg(ker: &Kernel, n: usize, cfg: MatmulCfg) {
    build_matmul_cfg_k(ker, n, cfg, cfg.k_step());
}

/// [`build_matmul_cfg`] with an explicit `k_step` (the LDS strip depth / K-loop
/// reduction step, replacing the hardcoded [`K_STEP`]). A thin wrapper that binds
/// the square `n×n` 16-bit→f32 ABI and runs [`gemm_core`] (the bound operand
/// buffers' dtype — bf16 or f16 — is the matrix-core input dtype).
///
/// # Panics
/// Panics on the same preconditions as [`gemm_core`].
pub fn build_matmul_cfg_k(ker: &Kernel, n: usize, cfg: MatmulCfg, k_step: usize) {
    // ABI: output (c, f32) then inputs (a, b — bf16), fixed by construction. Tiles in
    // `gemm_core` are declared by ROLE via the scaffold shortcuts (`ker.acc`/`operand`/
    // `shared_sw`), which resolve the arch fragment through `caps.frag` (gfx942 CDNA
    // MFMA vs gfx11 RDNA WMMA) — so the kernel names no physical fragment constant.
    let (outs, ins) = ker.bind_abi(
        &[GlSpec::new(&[1, 1, n, n], DType::Float32)],
        &[GlSpec::new(&[1, 1, n, n], DType::BFloat16), GlSpec::new(&[1, 1, n, n], DType::BFloat16)],
    );
    gemm_core(ker, (n, n, n), cfg.gemm(k_step), outs[0].clone(), ins[0].clone(), ins[1].clone(), Epilogue::Plain);
}
