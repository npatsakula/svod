//! Qwen3 fusions written with the [`svod_tk`] tile DSL: kernels that know this
//! model's memory layout, so they belong beside the model rather than in the
//! model-agnostic kernel crate.
//!
//! [`qkv_norm_rope`] is the attention prologue: one read of the fused-QKV GEMM
//! output `[B, L, (H + 2·H_kv)·Dh]` yields the three contiguous tensors the
//! flash kernel wants — `q` and `k` per-head RMS-normed over `Dh` and rotated,
//! `v` copied — instead of five passes over strided head views.
//!
//! The body has the shape of tk's [`rms_norm`](svod_tk::rms_norm): **one wave
//! per head**, the head held in registers, the sum of squares completed by a
//! butterfly shuffle (no LDS, no barrier), every global access a `vec`-wide
//! contiguous run per lane, and no `RANGE` anywhere — so it reuses that
//! kernel's row vocabulary ([`plan`], [`vload`], [`inv_rms`], [`scale_by`]) and
//! its arch gate ([`NORM_SUPPORTED_ARCHS`]). Numerics mirror the graph op for
//! op: one rounding at the end of the norm, and RoPE in the operand dtype, as
//! [`Tensor::apply_rotary_emb`](svod_tensor::Tensor::apply_rotary_emb) does.

use std::sync::Arc;

use snafu::ensure;
use svod_dtype::DType;
use svod_ir::UOp;
use svod_tensor::Tensor;
use svod_tk::group::{iadd, imod, imul};
use svod_tk::kernels::norm::{inv_rms, plan, scale_by, shift, sum_squares, vload, vpick, vstore};
use svod_tk::{GL, GlSpec, Group, Kernel, NORM_SUPPORTED_ARCHS};

/// The head geometry of a fused QKV projection: `h` query heads, `h_kv` key and
/// value heads, `dh` head dim — a `[B, L, (h + 2·h_kv)·dh]` row.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Heads {
    pub h: usize,
    pub h_kv: usize,
    pub dh: usize,
}

impl Heads {
    /// Heads per row, q then k then v.
    const fn total(&self) -> usize {
        self.h + 2 * self.h_kv
    }
    /// Elements per row of the fused GEMM output.
    const fn row(&self) -> usize {
        self.total() * self.dh
    }
}

/// Block shape of the prologue: `warps` waves per workgroup, one workgroup per
/// `(batch, position)` row. Wave `w` owns head `s·warps + w` for each slot `s`,
/// so `warps` must divide both `h` and `h_kv` — then every slot's role (q, k or
/// v) is a build-time constant and its store target is resolved statically.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct QkvCfg {
    pub warps: usize,
}

/// Waves per workgroup, widest first.
const QKV_WARPS: [usize; 4] = [8, 4, 2, 1];

/// The block shape for `heads`, or `None` when no wave count divides both head
/// counts, or the head dim / its halves do not divide into the wave.
pub fn select_qkv_cfg(heads: Heads, lanes: usize) -> Option<QkvCfg> {
    if !heads.dh.is_multiple_of(2) {
        return None;
    }
    plan(heads.dh, lanes)?;
    plan(heads.dh / 2, lanes)?;
    let warps = QKV_WARPS.into_iter().find(|w| heads.h.is_multiple_of(*w) && heads.h_kv.is_multiple_of(*w))?;
    Some(QkvCfg { warps })
}

/// What a slot's wave does with its head.
#[derive(Clone, Copy)]
enum Role {
    /// Normalize over `dh`, rotate, write to `q`.
    Query,
    /// The same, into `k`.
    Key,
    /// Copy into `v`.
    Value,
}

/// Head `first`'s role. Every head of a slot shares it, because the slot's wave
/// count divides `h` and `h_kv`.
fn role_of(first: usize, heads: Heads) -> Role {
    if first < heads.h {
        Role::Query
    } else if first < heads.h + heads.h_kv {
        Role::Key
    } else {
        Role::Value
    }
}

/// One head's RMS norm over `dh` + RoPE, written to `dst` at `dst_base`. The
/// halves are read together so the rotation is entirely in-lane: lane `L` holds
/// `x1[L·vec + j]` and its partner `x2[L·vec + j]`, `half` elements apart.
#[allow(clippy::too_many_arguments)]
fn norm_rope_head(
    warp: &Group<'_>,
    lanes: usize,
    heads: Heads,
    eps: f64,
    dt: &DType,
    gl: (&GL, &GL, &GL, &GL, &GL),
    at: (&Arc<UOp>, &Arc<UOp>, &Arc<UOp>, &Arc<UOp>),
    stores: &mut Vec<Arc<UOp>>,
) {
    let (src, dst, w_gl, cos_gl, sin_gl) = gl;
    let (src_base, dst_base, rope_base, lane_base) = at;
    let (half, (vec, chunks)) = (heads.dh / 2, plan(heads.dh / 2, lanes).expect("checked"));

    // Both halves of the head, widened to f32 for the reduce.
    let inner: Vec<Arc<UOp>> = (0..chunks).map(|c| shift(lane_base, (c * lanes * vec) as i64)).collect();
    let pairs: Vec<(Arc<UOp>, Arc<UOp>)> = inner
        .iter()
        .map(|i| {
            let lo = vload(src.uop(), &iadd(src_base, i), vec);
            (lo, vload(src.uop(), &iadd(src_base, &shift(i, half as i64)), vec))
        })
        .collect();
    let wide: Vec<Arc<UOp>> = pairs
        .iter()
        .flat_map(|(a, b)| (0..vec).flat_map(move |j| [vpick(a, j, vec), vpick(b, j, vec)]))
        .map(|v| v.cast(DType::Float32))
        .collect();
    let inv = inv_rms(warp, sum_squares(&wide), heads.dh, eps);

    for (c, i) in inner.iter().enumerate() {
        let hi = shift(i, half as i64);
        let (w1, w2) = (vload(w_gl.uop(), i, vec), vload(w_gl.uop(), &hi, vec));
        let (cv, sv) = (vload(cos_gl.uop(), &iadd(rope_base, i), vec), vload(sin_gl.uop(), &iadd(rope_base, i), vec));
        let (mut real, mut imag) = (Vec::with_capacity(vec), Vec::with_capacity(vec));
        for j in 0..vec {
            let (x1, x2) = (&wide[2 * (c * vec + j)], &wide[2 * (c * vec + j) + 1]);
            let n1 = scale_by(x1, &inv, &vpick(&w1, j, vec), dt);
            let n2 = scale_by(x2, &inv, &vpick(&w2, j, vec), dt);
            let (cos, sin) = (vpick(&cv, j, vec), vpick(&sv, j, vec));
            let (c1, s1) = (n1.try_mul(&cos).expect("rope"), n1.try_mul(&sin).expect("rope"));
            let (c2, s2) = (n2.try_mul(&cos).expect("rope"), n2.try_mul(&sin).expect("rope"));
            real.push(c1.try_sub(&s2).expect("rope real"));
            imag.push(s1.try_add(&c2).expect("rope imag"));
        }
        stores.push(vstore(dst.uop(), &iadd(dst_base, i), real));
        stores.push(vstore(dst.uop(), &iadd(dst_base, &hi), imag));
    }
}

/// The prologue body. ABI is `[q, k, v]` out, `[qkv, q_weight, k_weight, cos,
/// sin]` in.
fn build_qkv_norm_rope(ker: &Kernel, rows: usize, seq: usize, heads: Heads, dt: DType, eps: f64, cfg: QkvCfg) {
    let lanes = ker.caps.wave_size;
    let (half, dh) = (heads.dh / 2, heads.dh);
    let (vec_v, chunks_v) = plan(dh, lanes).expect("checked by the applicability predicate");

    let (outs, ins) = ker.bind_abi(
        &[
            GlSpec::new(&[rows, heads.h * dh], dt.clone()),
            GlSpec::new(&[rows, heads.h_kv * dh], dt.clone()),
            GlSpec::new(&[rows, heads.h_kv * dh], dt.clone()),
        ],
        &[
            GlSpec::new(&[rows, heads.row()], dt.clone()),
            GlSpec::new(&[dh], dt.clone()),
            GlSpec::new(&[dh], dt.clone()),
            GlSpec::new(&[seq, half], dt.clone()),
            GlSpec::new(&[seq, half], dt.clone()),
        ],
    );
    let (q_gl, k_gl, v_gl) = (outs[0].clone(), outs[1].clone(), outs[2].clone());
    let (qkv_gl, wq_gl, wk_gl) = (ins[0].clone(), ins[1].clone(), ins[2].clone());
    let (cos_gl, sin_gl) = (ins[3].clone(), ins[4].clone());

    let warp = ker.warp();
    let row = ker.grid_x();
    let wave = ker.warpid();
    let src_row = imul(&row, heads.row() as i64);
    // The rope table is indexed by position within the sequence; a single-row
    // batch folds the modulo away.
    let pos = if rows == seq { row.clone() } else { imod(&row, seq as i64) };
    let rope_base = imul(&pos, half as i64);

    let (mut q_stores, mut k_stores, mut v_stores) = (Vec::new(), Vec::new(), Vec::new());
    for slot in 0..heads.total() / cfg.warps {
        let first = slot * cfg.warps;
        let head = shift(&wave, first as i64);
        let src_base = iadd(&src_row, &imul(&head, dh as i64));
        let (out_gl, w_gl, head0, out_heads, stores) = match role_of(first, heads) {
            Role::Query => (&q_gl, &wq_gl, 0, heads.h, &mut q_stores),
            Role::Key => (&k_gl, &wk_gl, heads.h, heads.h_kv, &mut k_stores),
            Role::Value => (&v_gl, &wk_gl, heads.h + heads.h_kv, heads.h_kv, &mut v_stores),
        };
        let out_head = shift(&wave, (first - head0) as i64);
        let dst_base = iadd(&imul(&row, (out_heads * dh) as i64), &imul(&out_head, dh as i64));

        match role_of(first, heads) {
            Role::Value => {
                let lane_base = imul(&ker.laneid(), vec_v as i64);
                for c in 0..chunks_v {
                    let i = shift(&lane_base, (c * lanes * vec_v) as i64);
                    let loaded = vload(qkv_gl.uop(), &iadd(&src_base, &i), vec_v);
                    let vals = (0..vec_v).map(|j| vpick(&loaded, j, vec_v)).collect();
                    stores.push(vstore(out_gl.uop(), &iadd(&dst_base, &i), vals));
                }
            }
            _ => {
                let lane_base = imul(&ker.laneid(), plan(half, lanes).expect("checked").0 as i64);
                norm_rope_head(
                    &warp,
                    lanes,
                    heads,
                    eps,
                    &dt,
                    (&qkv_gl, out_gl, w_gl, &cos_gl, &sin_gl),
                    (&src_base, &dst_base, &rope_base, &lane_base),
                    stores,
                );
            }
        }
    }
    ker.push_store(UOp::group(q_stores), q_gl.uop().clone());
    ker.push_store(UOp::group(k_stores), k_gl.uop().clone());
    ker.push_store(UOp::group(v_stores), v_gl.uop().clone());
}

/// **Graph-native** fused-QKV attention prologue: one read of the realized
/// `[B, L, (h + 2·h_kv)·dh]` GEMM output yields the three contiguous tensors the
/// flash kernel consumes — `q` `[B, L, h, dh]` and `k` `[B, L, h_kv, dh]`,
/// per-head RMS-normed over `dh` and rotated, and `v` `[B, L, h_kv, dh]`, copied.
///
/// The per-head norm is [`svod_tk::rms_norm`]'s (`q_weight`/`k_weight` are `[dh]`), and
/// the rotation is exactly
/// [`Tensor::apply_rotary_emb`](svod_tensor::Tensor::apply_rotary_emb)`(cos, sin,
/// false)` applied after it: halves, `real = x1·cos − x2·sin`,
/// `imag = x1·sin + x2·cos`, in the operand dtype.
///
/// **`cos` and `sin` are the half-width tables laid out row-major as
/// `[L, dh/2]`** — any shape with that element count whose last dim is `dh/2`,
/// so the model's `[1, L, 1, dh/2]` sequence-major cache passes unchanged.
///
/// The outcome is three-way (via [`svod_tk::launch_custom`]):
///
/// - `Ok(None)` — *doesn't apply here:* the device is not one of
///   [`NORM_SUPPORTED_ARCHS`] with its LLVM backend, **or** the geometry does not fit
///   ([`select_qkv_cfg`]): `dh` and `dh/2` must both be multiples of the wave
///   (32) with at most 64 elements per lane, and some wave count must divide
///   both `h` and `h_kv`. The caller substitutes the split / norm / rope graph.
/// - `Err` — *malformed request:* a symbolic dim, `qkv` not rank 3, a last dim
///   that is not `(h + 2·h_kv)·dh`, a weight that is not `[dh]`, a rope table
///   that is not `[L, dh/2]`, a dtype outside {bf16, f16}, or a dtype mismatch.
/// - `Ok(Some((q, k, v)))` — it ran.
pub fn qkv_norm_rope(
    qkv: &Tensor,
    q_weight: &Tensor,
    k_weight: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    eps: f64,
    heads: Heads,
) -> svod_tk::LaunchResult<Option<(Tensor, Tensor, Tensor)>> {
    const K: &str = "qkv-norm-rope";
    let qd = svod_tk::launch::concrete_dims(qkv, K, "qkv", 3)?;
    let (b, seq) = (qd[0], qd[1]);
    let (rows, half) = (b * seq, heads.dh / 2);
    let dtype = qkv.uop().dtype();

    let mut operands: Vec<(&'static str, Vec<usize>, DType, Vec<usize>)> = Vec::with_capacity(5);
    operands.push(("qkv", qd.clone(), dtype.clone(), vec![b, seq, heads.row()]));
    for (name, w) in [("q_weight", q_weight), ("k_weight", k_weight)] {
        let d = svod_tk::launch::concrete_dims(w, K, name, 1)?;
        operands.push((name, d, w.uop().dtype(), vec![heads.dh]));
    }
    // The rope tables are `[L, dh/2]` however the caller spells the unit axes:
    // the element count and the innermost dim are what the flat addressing uses.
    for (name, t) in [("cos", cos), ("sin", sin)] {
        let d = svod_tk::launch::concrete_dims_at_least(t, K, name, 1)?;
        let flat = vec![d.iter().product::<usize>(), *d.last().expect("rank >= 1")];
        operands.push((name, flat, t.uop().dtype(), vec![seq * half, half]));
    }

    svod_tk::launch_custom(
        &qkv.device(),
        NORM_SUPPORTED_ARCHS,
        move |_arch| {
            ensure!(
                dtype == DType::BFloat16 || dtype == DType::Float16,
                svod_tk::launch::DtypeSnafu { kernel: K, got: dtype.clone(), expected: "bf16 or f16" }
            );
            for (operand, got, got_dt, expected) in operands {
                ensure!(
                    got_dt == dtype,
                    svod_tk::launch::DtypeSnafu { kernel: K, got: got_dt, expected: "the dtype of qkv" }
                );
                ensure!(got == expected, svod_tk::launch::OperandShapeSnafu { kernel: K, operand, expected, got });
            }
            Ok(())
        },
        move |arch| heads.dh > 0 && select_qkv_cfg(heads, svod_tk::ArchCaps::for_arch(arch).wave_size).is_some(),
        move |arch| {
            let caps = svod_tk::ArchCaps::for_arch(arch);
            let cfg = select_qkv_cfg(heads, caps.wave_size).expect("checked by the fit predicate");
            let dt = qkv.uop().dtype();
            let shape = |h: usize| vec![b, seq, h, heads.dh];
            let outs = vec![
                Tensor::empty(&shape(heads.h), dt.clone()),
                Tensor::empty(&shape(heads.h_kv), dt.clone()),
                Tensor::empty(&shape(heads.h_kv), dt.clone()),
            ];
            let build_dt = dt.clone();
            let out = svod_tk::graph_launch_multi(
                "qkv_norm_rope",
                [rows as i64, 1, 1],
                (cfg.warps * caps.wave_size) as i64,
                outs,
                &[qkv, q_weight, k_weight, cos, sin],
                caps,
                move |ker| {
                    build_qkv_norm_rope(ker, rows, seq, heads, build_dt, eps, cfg);
                    ker.finish(3)
                },
            )?;
            let mut out = out.into_iter();
            let q = out.next().expect("q output");
            let k = out.next().expect("k output");
            let v = out.next().expect("v output");
            Ok((q, k, v))
        },
    )
}
