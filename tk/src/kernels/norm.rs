//! Row-wise RMS normalization — the *elementwise* peer of
//! [`gemm_nt`](super::gemm::gemm_nt) and
//! [`flash_attention_with`](crate::flash_attention_with).
//!
//! [`rms_norm`] / [`add_rms_norm`] are `y = rms_norm(x)` and the residual-fused
//! `(h, y) = (x + residual, rms_norm(x + residual))`, so a pre-norm decoder
//! layer writes and reads its residual stream once instead of recomputing the
//! lazy add in the reduce, in the apply, and again in the next layer.
//!
//! Both have the same shape: **one wave per row**, the row held in registers,
//! the sum of squares completed by a butterfly shuffle (no LDS, no barrier),
//! and every global access a `vec`-wide contiguous run per lane so a wave's
//! load is one coalesced transaction. The bodies are straight-line — there is
//! no `RANGE` anywhere, so every register index is a compile-time constant and
//! nothing spills to local memory. That row vocabulary ([`plan`], [`vload`] /
//! [`vstore`], [`inv_rms`], [`scale_by`]) is public, because a model-side
//! fusion built on the same DSL reuses it.
//!
//! Numerics mirror the graph exactly, op for op, so a fused layer and an eager
//! one agree to the last bf16 rounding *except* for the summation order of the
//! row reduce (a butterfly tree here, the scheduler's tree there):
//! `y = bf16((f32(x)·rsqrt(Σx²/D + eps))·f32(w))` — one rounding, at the end.

use std::sync::Arc;

use snafu::ensure;
use svod_dtype::DType;
use svod_ir::{ConstValue, UOp};
use svod_tensor::Tensor;

use crate::group::{iadd, imul};
use crate::index::{cidx, index_off, load_off};
use crate::scaffold::GlSpec;
use crate::{Group, Kernel};

/// The arches these kernels are enabled for. The bodies are arch-generic (wave
/// size + butterfly shuffle), but the vector-width ladder and the block shapes
/// below are measured on `mma.sync` (sm_86); another arch joins by measuring its
/// own, not by inheriting this one.
pub const NORM_SUPPORTED_ARCHS: crate::ArchSet =
    crate::ArchSet::amd(&[]).with_cuda_from(svod_dtype::CudaArch::from_compute_capability(8, 0));

/// Per-lane global-access widths, widest first: 8 bf16 is the 128-bit vector
/// load, and a wave issuing it covers `32 × 16 = 512` contiguous bytes.
const VEC_LADDER: [usize; 4] = [8, 4, 2, 1];

/// Elements one lane may hold. A row lives in registers for its whole life (one
/// pass over memory), so this is the register budget: 64 bf16 values plus their
/// f32 widenings already costs ~96 registers, past which occupancy falls faster
/// than the saved pass pays back.
const MAX_PER_LANE: usize = 64;

/// How a wave covers one contiguous `span` of a row: `vec` consecutive elements
/// per lane per chunk, `chunks` chunks. Element `(c, j)` of lane `L` is
/// `c·(lanes·vec) + L·vec + j`, so each chunk is one coalesced wave-wide
/// transaction. `None` when the span does not divide into the wave.
pub fn plan(span: usize, lanes: usize) -> Option<(usize, usize)> {
    if span == 0 || !span.is_multiple_of(lanes) {
        return None;
    }
    let per_lane = span / lanes;
    if per_lane > MAX_PER_LANE {
        return None;
    }
    let vec = VEC_LADDER.into_iter().find(|v| per_lane.is_multiple_of(*v))?;
    Some((vec, per_lane / vec))
}

fn f32c(v: f64) -> Arc<UOp> {
    UOp::const_(DType::Float32, ConstValue::Float(v))
}

/// A lane's `vec`-wide load at flat element offset `off` — one shaped access,
/// which the renderer widens to a single `ld.global.v4`-class instruction.
pub fn vload(buf: &Arc<UOp>, off: &Arc<UOp>, vec: usize) -> Arc<UOp> {
    if vec == 1 {
        return load_off(buf, off.clone());
    }
    let idx = UOp::index().buffer(buf.clone()).indices(vec![spread(off, vec)]).call().expect("vector load INDEX");
    UOp::load().index(idx).call()
}

/// Element `j` of a [`vload`] result.
pub fn vpick(v: &Arc<UOp>, j: usize, vec: usize) -> Arc<UOp> {
    if vec == 1 { v.clone() } else { v.index_axes(vec![j]) }
}

/// A lane's `vals.len()`-wide store at flat element offset `off`.
pub fn vstore(buf: &Arc<UOp>, off: &Arc<UOp>, vals: Vec<Arc<UOp>>) -> Arc<UOp> {
    if vals.len() == 1 {
        return index_off(buf, off.clone()).store(vals.into_iter().next().expect("one value"));
    }
    UOp::index()
        .buffer(buf.clone())
        .indices(vec![spread(off, vals.len())])
        .call()
        .expect("vector store INDEX")
        .store(UOp::stack(vals.into_iter().collect()))
}

/// `[off, off+1, …, off+w-1]` as one shaped index.
fn spread(off: &Arc<UOp>, w: usize) -> Arc<UOp> {
    UOp::stack((0..w as i64).map(|l| if l == 0 { off.clone() } else { iadd(off, &cidx(l)) }).collect())
}

/// `off + k` with the constant folded away when it is zero.
pub fn shift(off: &Arc<UOp>, k: i64) -> Arc<UOp> {
    if k == 0 { off.clone() } else { iadd(off, &cidx(k)) }
}

/// `rsqrt(Σ/n + eps)`, replicated in every lane of the wave — the graph's
/// `mean(x²) + eps` then `rsqrt`, with the wave butterfly completing the sum.
pub fn inv_rms(warp: &Group<'_>, partial: Arc<UOp>, n: usize, eps: f64) -> Arc<UOp> {
    warp.wave_reduce_scalar(partial, |a, b| a.try_add(b).expect("rms: f32 sum"))
        .try_div(&f32c(n as f64))
        .expect("rms: mean")
        .try_add(&f32c(eps))
        .expect("rms: eps")
        .try_rsqrt()
        .expect("rms: rsqrt")
}

/// The `Σx²` fold over `vals` (already f32), in emission order.
pub fn sum_squares(vals: &[Arc<UOp>]) -> Arc<UOp> {
    vals.iter()
        .map(|v| v.try_mul(v).expect("rms: square"))
        .reduce(|a, b| a.try_add(&b).expect("rms: accumulate"))
        .expect("rms: a row has at least one element")
}

/// `bf16((x·inv)·f32(w))` — the graph's `affine_f32`: the weight multiply runs
/// in f32 and the single rounding to the operand dtype happens at the end.
pub fn scale_by(x: &Arc<UOp>, inv: &Arc<UOp>, w: &Arc<UOp>, dt: &DType) -> Arc<UOp> {
    x.try_mul(inv).expect("rms: normalize").try_mul(&w.cast(DType::Float32)).expect("rms: weight").cast(dt.clone())
}

// ── The row-norm kernel ──────────────────────────────────────────────────────

/// Block shape of the row-norm kernel: `rows_per_block` waves per workgroup,
/// one row each.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct NormCfg {
    pub rows_per_block: usize,
}

/// Workgroup shapes, widest first. Four waves (128 threads) is the measured
/// default; a row count that does not divide by it falls through the ladder.
const ROWS_PER_BLOCK: [usize; 4] = [8, 4, 2, 1];

/// The block shape for `rows` rows of `d` elements on a `lanes`-wide wave, or
/// `None` when the row does not divide into the wave (or is too wide to hold in
/// registers) — see [`plan`].
pub fn select_norm_cfg(rows: usize, d: usize, lanes: usize) -> Option<NormCfg> {
    plan(d, lanes)?;
    let rows_per_block = ROWS_PER_BLOCK.into_iter().find(|r| rows.is_multiple_of(*r))?;
    Some(NormCfg { rows_per_block })
}

/// The row-norm body: `y[r] = rms_norm(x[r])`, and with `residual` bound also
/// `h[r] = x[r] + residual[r]` (the normed row is then `h`'s). ABI is
/// `[y (, h)]` out, `[x (, residual), weight]` in — `h` trails `y` so the
/// one-output kernel keeps slot 0.
fn build_row_norm(ker: &Kernel, rows: usize, d: usize, dt: DType, eps: f64, cfg: NormCfg, fuse_add: bool) {
    let lanes = ker.caps.wave_size;
    let (vec, chunks) = plan(d, lanes).expect("checked by the applicability predicate");
    let row_shape = [rows, d];

    let mut out_specs = vec![GlSpec::new(&row_shape, dt.clone())];
    let mut in_specs = vec![GlSpec::new(&row_shape, dt.clone())];
    if fuse_add {
        out_specs.push(GlSpec::new(&row_shape, dt.clone()));
        in_specs.push(GlSpec::new(&row_shape, dt.clone()));
    }
    in_specs.push(GlSpec::new(&[d], dt.clone()));
    let (outs, ins) = ker.bind_abi(&out_specs, &in_specs);
    let (y_gl, h_gl) = (outs[0].clone(), fuse_add.then(|| outs[1].clone()));
    let (x_gl, res_gl) = (ins[0].clone(), fuse_add.then(|| ins[1].clone()));
    let w_gl = ins.last().expect("weight global").clone();

    let warp = ker.warp();
    let lane = ker.laneid();
    let row = iadd(&imul(&ker.grid_x(), cfg.rows_per_block as i64), &ker.warpid());
    let row_base = imul(&row, d as i64);
    // This lane's `chunks` global offsets into the row, and the matching
    // row-relative offsets the [D] weight is read at.
    let lane_base = imul(&lane, vec as i64);
    let chunk_off: Vec<(Arc<UOp>, Arc<UOp>)> = (0..chunks)
        .map(|c| {
            let inner = shift(&lane_base, (c * lanes * vec) as i64);
            (iadd(&row_base, &inner), inner)
        })
        .collect();

    // One pass over the row: load (and fold in the residual), widen to f32.
    let (mut vals, mut wide) = (Vec::with_capacity(d / lanes), Vec::with_capacity(d / lanes));
    for (off, _) in &chunk_off {
        let xv = vload(x_gl.uop(), off, vec);
        let rv = res_gl.as_ref().map(|g| vload(g.uop(), off, vec));
        for j in 0..vec {
            let mut v = vpick(&xv, j, vec);
            if let Some(rv) = &rv {
                v = v.try_add(&vpick(rv, j, vec)).expect("residual add");
            }
            wide.push(v.cast(DType::Float32));
            vals.push(v);
        }
    }
    let inv = inv_rms(&warp, sum_squares(&wide), d, eps);

    let mut h_stores = Vec::with_capacity(chunks);
    let mut y_stores = Vec::with_capacity(chunks);
    for (c, (off, winner)) in chunk_off.iter().enumerate() {
        let wv = vload(w_gl.uop(), winner, vec);
        let ys = (0..vec).map(|j| scale_by(&wide[c * vec + j], &inv, &vpick(&wv, j, vec), &dt)).collect();
        if let Some(h_gl) = &h_gl {
            h_stores.push(vstore(h_gl.uop(), off, vals[c * vec..(c + 1) * vec].to_vec()));
        }
        y_stores.push(vstore(y_gl.uop(), off, ys));
    }
    ker.push_store(UOp::group(y_stores), y_gl.uop().clone());
    if let Some(h_gl) = &h_gl {
        ker.push_store(UOp::group(h_stores), h_gl.uop().clone());
    }
}

// ── Public entries ───────────────────────────────────────────────────────────

/// Split `x`'s shape into `(rows, D)` — every leading dim is a row.
fn rows_and_d(dims: &[usize]) -> (usize, usize) {
    let d = *dims.last().expect("rank >= 2");
    (dims[..dims.len() - 1].iter().product(), d)
}

/// Structural checks shared by [`rms_norm`] and [`add_rms_norm`]: a matrix-core
/// operand dtype, and a `[D]` weight in that dtype.
fn check_norm_operands(
    kernel: &'static str,
    dtype: &DType,
    w_dtype: &DType,
    wd: &[usize],
    d: usize,
) -> crate::LaunchResult<()> {
    ensure!(
        *dtype == DType::BFloat16 || *dtype == DType::Float16,
        crate::launch::DtypeSnafu { kernel, got: dtype.clone(), expected: "bf16 or f16" }
    );
    ensure!(w_dtype == dtype, crate::launch::DtypeSnafu { kernel, got: w_dtype.clone(), expected: "the dtype of x" });
    ensure!(
        wd == [d],
        crate::launch::OperandShapeSnafu { kernel, operand: "weight", expected: vec![d], got: wd.to_vec() }
    );
    Ok(())
}

/// **Graph-native** `y = rms_norm(x, weight, eps)` over the last axis — the
/// elementwise peer of [`gemm_nt`](super::gemm::gemm_nt). Returns a lazy output
/// [`Tensor`] that composes into a model graph and realizes through the normal
/// `prepare()` path.
///
/// `x` is `[rows..., D]` of any rank ≥ 2 (the leading dims are the rows, so a
/// `[B, L, D]` activation needs no reshape), statically shaped and **bf16 or
/// f16**; `weight` is `[D]` in the same dtype. `y` has `x`'s shape and dtype.
/// The math is the graph's, op for op:
/// `y = dtype((f32(x)·rsqrt(Σf32(x)²/D + eps))·f32(weight))`, one rounding at
/// the end — only the summation order of the row reduce differs (a wave
/// butterfly here).
///
/// The outcome is three-way (via [`crate::launch_custom`]):
///
/// - `Ok(None)` — *doesn't apply here:* the device is not CUDA sm_80+ with its
///   LLVM backend ([`NORM_SUPPORTED_ARCHS`]), **or** the shape does not fit
///   ([`select_norm_cfg`]): `D` must be a multiple of the wave (32) and at most
///   `64·32 = 2048` (a row lives in registers). The caller substitutes
///   `Tensor::rms_norm_with`.
/// - `Err` — *malformed request:* a symbolic dim, `x` below rank 2, `weight`
///   not rank 1 or not `[D]`, a dtype outside {bf16, f16}, or a dtype mismatch.
/// - `Ok(Some(y))` — it ran.
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> crate::LaunchResult<Option<Tensor>> {
    let xd = crate::launch::concrete_dims_at_least(x, "rms-norm", "x", 2)?;
    let wd = crate::launch::concrete_dims(weight, "rms-norm", "weight", 1)?;
    let (rows, d) = rows_and_d(&xd);
    let (dtype, w_dtype) = (x.uop().dtype(), weight.uop().dtype());
    let check = (dtype.clone(), w_dtype, wd, d);

    crate::launch_custom(
        &x.device(),
        NORM_SUPPORTED_ARCHS,
        move |_arch| check_norm_operands("rms-norm", &check.0, &check.1, &check.2, check.3),
        move |arch| select_norm_cfg(rows, d, crate::ArchCaps::for_arch(arch).wave_size).is_some(),
        move |arch| {
            let caps = crate::ArchCaps::for_arch(arch);
            let cfg = select_norm_cfg(rows, d, caps.wave_size).expect("checked by the fit predicate");
            let (grid, block) = launch_dims(rows, cfg.rows_per_block, caps.wave_size);
            let out = Tensor::empty(&xd, dtype.clone());
            let dt = dtype.clone();
            crate::graph_launch("rms_norm", grid, block, out, &[x, weight], caps, move |ker| {
                build_row_norm(ker, rows, d, dt, eps, cfg, false);
                ker.finish(1)
            })
        },
    )
}

/// **Graph-native** residual-fused RMS norm: `h = x + residual` (rounded as the
/// graph's bf16 add rounds it) and `y = rms_norm(h, weight, eps)`, written by
/// one kernel so a pre-norm decoder layer's residual stream is written once and
/// read once per layer instead of being recomputed in the reduce, in the apply,
/// and again by the next consumer of the lazy add.
///
/// Shapes, dtypes and the three-way outcome are [`rms_norm`]'s; `residual` must
/// match `x` exactly. Returns `(h, y)`.
pub fn add_rms_norm(
    x: &Tensor,
    residual: &Tensor,
    weight: &Tensor,
    eps: f64,
) -> crate::LaunchResult<Option<(Tensor, Tensor)>> {
    let xd = crate::launch::concrete_dims_at_least(x, "add-rms-norm", "x", 2)?;
    let rd = crate::launch::concrete_dims_at_least(residual, "add-rms-norm", "residual", 2)?;
    let wd = crate::launch::concrete_dims(weight, "add-rms-norm", "weight", 1)?;
    let (rows, d) = rows_and_d(&xd);
    let (dtype, w_dtype, r_dtype) = (x.uop().dtype(), weight.uop().dtype(), residual.uop().dtype());
    let check = (dtype.clone(), w_dtype, wd, d, r_dtype, rd, xd.clone());

    crate::launch_custom(
        &x.device(),
        NORM_SUPPORTED_ARCHS,
        move |_arch| {
            check_norm_operands("add-rms-norm", &check.0, &check.1, &check.2, check.3)?;
            ensure!(
                check.4 == check.0,
                crate::launch::DtypeSnafu { kernel: "add-rms-norm", got: check.4, expected: "the dtype of x" }
            );
            ensure!(
                check.5 == check.6,
                crate::launch::OperandShapeSnafu {
                    kernel: "add-rms-norm",
                    operand: "residual",
                    expected: check.6,
                    got: check.5
                }
            );
            Ok(())
        },
        move |arch| select_norm_cfg(rows, d, crate::ArchCaps::for_arch(arch).wave_size).is_some(),
        move |arch| {
            let caps = crate::ArchCaps::for_arch(arch);
            let cfg = select_norm_cfg(rows, d, caps.wave_size).expect("checked by the fit predicate");
            let (grid, block) = launch_dims(rows, cfg.rows_per_block, caps.wave_size);
            // ABI: `[y, h]` out, `[x, residual, weight]` in.
            let outs = vec![Tensor::empty(&xd, dtype.clone()), Tensor::empty(&xd, dtype.clone())];
            let dt = dtype.clone();
            let out = crate::graph_launch_multi(
                "add_rms_norm",
                grid,
                block,
                outs,
                &[x, residual, weight],
                caps,
                move |ker| {
                    build_row_norm(ker, rows, d, dt, eps, cfg, true);
                    ker.finish(2)
                },
            )?;
            let mut out = out.into_iter();
            let y = out.next().expect("y output");
            let h = out.next().expect("h output");
            Ok((h, y))
        },
    )
}

/// Launch geometry of the row-norm kernel.
fn launch_dims(rows: usize, rows_per_block: usize, lanes: usize) -> ([i64; 3], i64) {
    ([(rows / rows_per_block) as i64, 1, 1], (rows_per_block * lanes) as i64)
}
