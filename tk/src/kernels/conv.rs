//! Implicit-GEMM 2-D convolution over channels-last activations.
//!
//! `y[b, oy, ox, :] = act(Σ_{ky,kx,ci} x[b, oy·s + ky − p, ox·s + kx − p, ci] · w[:, ky, kx, ci] + bias) [+ residual]`
//! is the NT GEMM `C[M, N] = A[M, K] · B[N, K]ᵀ` with `M` the output pixels, `N`
//! the output channels and `K = kh·kw·cin` — except that `A` is never formed:
//! [`gemm_core_with`] gathers each strip row straight from `x`, and a tap that
//! falls in the padding reads as zeros. With `cin` a multiple of the strip
//! depth a strip never straddles a tap, so a lane's run is contiguous in `x`
//! and `w[cout, kh, kw, cin]` is a plain `[N, K]` weight.

use std::cell::OnceCell;
use std::rc::Rc;
use std::sync::Arc;

use snafu::{ResultExt, ensure};
use svod_dtype::DType;
use svod_ir::UOp;
use svod_tensor::Tensor;

use super::gemm::{Epilogue, GEMM_NT_SUPPORTED_ARCHS, GemmCfg, GemmPolicy, RowSource, gemm_core_with};
use crate::group::{iadd, idiv, imod, imul};
use crate::index::cidx;
use crate::{GlSpec, Kernel};

/// The arches the kernel runs on: the NT GEMM's, whose tile tables it reads.
pub const CONV_SUPPORTED_ARCHS: crate::ArchSet = GEMM_NT_SUPPORTED_ARCHS;

/// One convolution's shape: `x[batch, h, w, cin]`, `w[cout, kh, kw, cin]`, a
/// square `stride` and symmetric `pad`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ConvGeom {
    pub batch: usize,
    pub h: usize,
    pub w: usize,
    pub cin: usize,
    pub cout: usize,
    pub kh: usize,
    pub kw: usize,
    pub stride: usize,
    pub pad: usize,
}

impl ConvGeom {
    pub const fn ho(&self) -> usize {
        (self.h + 2 * self.pad - self.kh) / self.stride + 1
    }
    pub const fn wo(&self) -> usize {
        (self.w + 2 * self.pad - self.kw) / self.stride + 1
    }
    /// The GEMM's `(M, K, N)`: output pixels, taps times input channels, output channels.
    pub const fn mkn(&self) -> (usize, usize, usize) {
        (self.batch * self.ho() * self.wo(), self.kh * self.kw * self.cin, self.cout)
    }
    /// Whether `cfg` tiles this convolution: a strip stays inside one tap
    /// (`cin` a multiple of `k_step`), `N` tiles exactly, the K loop is at least
    /// as deep as the pipeline, and the tile carries a fused store (no split-K).
    /// `M` may be ragged.
    pub fn tiles(&self, cfg: &GemmCfg) -> bool {
        let (_, k, n) = self.mkn();
        cfg.split_k == 1
            && cfg.stages == 2
            && self.cin.is_multiple_of(cfg.k_step)
            && n.is_multiple_of(cfg.block_n)
            && k / cfg.k_step >= cfg.stages
    }
    /// Workgroups `cfg` launches: `ceil(M / block_m) · N / block_n`.
    pub const fn blocks(&self, cfg: &GemmCfg) -> usize {
        let (m, _, n) = self.mkn();
        m.div_ceil(cfg.block_m) * (n / cfg.block_n)
    }
    /// The launch grid: N blocks on x, M blocks on y (the plain 2-D form).
    pub const fn grid_dims(&self, cfg: &GemmCfg) -> [i64; 3] {
        let (m, _, n) = self.mkn();
        [(n / cfg.block_n) as i64, m.div_ceil(cfg.block_m) as i64, 1]
    }
}

/// The tile for `geom` from `policy`'s table, or `None` when none tiles it:
/// the widest tile unless its grid would not fill the device, in which case the
/// finer ones come first, as [`GemmPolicy::cfg`] orders them. The L2 swizzle is
/// off: the grid is the plain 2-D one and `M` may be ragged.
pub fn select_conv_cfg(policy: &GemmPolicy, geom: &ConvGeom) -> Option<GemmCfg> {
    let plain = |cfg: &GemmCfg| GemmCfg { l2_swizzle: false, ..*cfg };
    let widest = policy.conv_tiles.first()?;
    let starved = geom.blocks(widest) < policy.compute_units * policy.resident;
    let wide = |cfg: &GemmCfg| cfg.block_m * cfg.block_n >= widest.block_m * widest.block_n;
    let mut table: Vec<GemmCfg> = policy.conv_tiles.iter().map(plain).collect();
    if starved {
        table.sort_by_key(wide);
    }
    table.into_iter().find(|cfg| geom.tiles(cfg))
}

/// [`select_conv_cfg`] as measured on this device ([`crate::tune`]): every
/// table tile that fits is timed once on synthetic operands and the fastest
/// kept in `store`; the static choice where only one fits or nothing measured.
pub fn tuned_conv_cfg(
    store: &crate::tune::TuneStore,
    spec: &svod_dtype::DeviceSpec,
    arch: svod_dtype::GpuArch,
    dtype: &DType,
    geom: ConvGeom,
    epi: Epilogue<()>,
) -> Option<GemmCfg> {
    let policy = GemmPolicy::for_device(spec, arch);
    let caps = crate::ArchCaps::for_arch(arch);
    let candidates: Vec<GemmCfg> = policy
        .conv_tiles
        .iter()
        .map(|cfg| GemmCfg { l2_swizzle: false, ..*cfg })
        .filter(|cfg| geom.tiles(cfg))
        .collect();
    let fallback = || select_conv_cfg(&policy, &geom);
    if candidates.len() < 2 {
        return fallback();
    }
    let (m, k, n) = geom.mkn();
    let residual = matches!(epi, Epilogue::BiasAct { residual: Some(()), .. });
    let build = move |ker: &Kernel, cfg: GemmCfg| {
        build_conv(ker, geom, cfg, dtype.clone(), epi);
        ker.finish(cfg.acc_m)
    };
    let builds = || {
        let placeholders = || {
            let mut sizes = vec![m * n, geom.batch * geom.h * geom.w * geom.cin, n * k, n];
            if residual {
                sizes.push(m * n);
            }
            sizes.into_iter().map(|s| UOp::new_buffer(svod_dtype::DeviceSpec::Cpu, s, dtype.clone())).collect()
        };
        candidates
            .iter()
            .map(|&cfg| {
                let ker =
                    Kernel::new("conv2d_nhwc", geom.grid_dims(&cfg), cfg.threads(caps.wave_size), placeholders(), caps);
                crate::kernel_fingerprint(&build(&ker, cfg)).digest
            })
            .collect()
    };
    let shape = [
        geom.batch,
        geom.h,
        geom.w,
        geom.cin,
        geom.cout,
        geom.kh,
        geom.kw,
        geom.stride,
        geom.pad,
        dtype.bytes(),
        epi.code(),
    ];
    let key = crate::tune::TuneKey::new("conv2d_nhwc", spec, arch, &shape, &(&candidates, dtype));
    let compile = |i: usize| {
        let cfg = candidates[i];
        let operand = |shape: &[usize]| Tensor::randn(shape).ok().map(|t| t.cast(dtype.clone()).to(spec.clone()));
        let (x, w, b) = (
            operand(&[geom.batch, geom.h, geom.w, geom.cin])?,
            operand(&[geom.cout, geom.kh, geom.kw, geom.cin])?,
            operand(&[geom.cout])?,
        );
        let mut ins = vec![x, w, b];
        if residual {
            ins.push(operand(&[m, n])?);
        }
        let ins: Vec<&Tensor> = ins.iter().collect();
        let mut y = Tensor::empty(&[m, n], dtype.clone()).to(spec.clone());
        let (grid, block) = (geom.grid_dims(&cfg), cfg.threads(caps.wave_size));
        crate::launch::compile_kernel("conv2d_nhwc_tune", grid, block, &mut [&mut y], &ins, move |ker| build(ker, cfg))
            .ok()
    };
    store.select(&key, candidates.len(), builds, compile).map(|i| candidates[i]).or_else(fallback)
}

/// The tile for `geom` on the device behind `spec`: measured when tuning is on
/// ([`crate::tune::enabled`]), else the static [`select_conv_cfg`].
fn choose_conv_cfg(
    spec: &svod_dtype::DeviceSpec,
    arch: svod_dtype::GpuArch,
    dtype: &DType,
    geom: ConvGeom,
    epi: Epilogue<()>,
) -> Option<GemmCfg> {
    if crate::tune::enabled() {
        tuned_conv_cfg(crate::tune::TuneStore::global(), spec, arch, dtype, geom, epi)
    } else {
        select_conv_cfg(&GemmPolicy::for_device(spec, arch), &geom)
    }
}

/// The A-strip row source of `geom` under `cfg`: strip row `r` of M block
/// `pid_m` at K trip `tile` is output pixel `m = pid_m·block_m + r`, decoded to
/// `(b, oy, ox)`, at tap `tile·k_step / cin`, decoded to `(ky, kx)`; the row's
/// first strip element is `x[b, oy·s + ky − p, ox·s + kx − p, cin0]` with
/// `cin0 = tile·k_step mod cin`, and the row is valid when that pixel exists
/// (and `m < M`, when `M` is ragged).
fn row_source(geom: ConvGeom, cfg: GemmCfg) -> Box<RowSource<'static>> {
    let (m_total, _, _) = geom.mkn();
    let (ho, wo) = (geom.ho() as i64, geom.wo() as i64);
    let (h, w, cin) = (geom.h as i64, geom.w as i64, geom.cin as i64);
    let (kw, stride, pad) = (geom.kw as i64, geom.stride as i64, geom.pad as i64);
    let ragged = !m_total.is_multiple_of(cfg.block_m);
    let taps = geom.kh * geom.kw;
    Box::new(move |r, pid_m, tile| {
        let lt = |a: &Arc<UOp>, b: &Arc<UOp>| a.try_cmplt(b).expect("conv row: compare");
        let and = |a: Arc<UOp>, b: Arc<UOp>| a.try_and_op(&b).expect("conv row: and");
        let m = iadd(&imul(pid_m, cfg.block_m as i64), r);
        let (b, rem) = (idiv(&m, ho * wo), imod(&m, ho * wo));
        let (oy, ox) = (idiv(&rem, wo), imod(&rem, wo));
        let kbase = imul(tile, cfg.k_step as i64);
        // A 1x1 conv has one tap, so the whole K axis is the channel axis.
        let (tap, cin0) = if taps == 1 { (cidx(0), kbase) } else { (idiv(&kbase, cin), imod(&kbase, cin)) };
        let (ky, kx) = (idiv(&tap, kw), imod(&tap, kw));
        // Padded coordinates (`+ pad`, so never negative): the tap is inside the
        // image when `pad <= c < extent + pad`.
        let iy = iadd(&imul(&oy, stride), &ky);
        let ix = iadd(&imul(&ox, stride), &kx);
        let inside = |c: &Arc<UOp>, extent: i64| and(lt(&cidx(pad), &iadd(c, &cidx(1))), lt(c, &cidx(extent + pad)));
        let mut valid = and(inside(&iy, h), inside(&ix, w));
        if ragged {
            valid = and(valid, lt(&m, &cidx(m_total as i64)));
        }
        let row = iadd(&imul(&iadd(&imul(&b, h), &iy), w), &ix);
        let off = iadd(&imul(&row, cin), &cidx(-pad * (w + 1) * cin));
        (iadd(&off, &cin0), valid)
    })
}

/// The output's `[batch, ho, wo, cout]`.
fn y_dims(geom: &ConvGeom) -> Vec<usize> {
    vec![geom.batch, geom.ho(), geom.wo(), geom.cout]
}

/// Bind the ABI (`y[M, N]` out; `x[batch·h·w, cin]`, `w[N, K]`, `bias[N]` and,
/// under a residual, `residual[M, N]` in) and run the gathered GEMM.
pub fn build_conv(ker: &Kernel, geom: ConvGeom, cfg: GemmCfg, dt: DType, epi: Epilogue<()>) {
    let (m, k, n) = geom.mkn();
    let Epilogue::BiasAct { residual, act, .. } = epi else { panic!("conv2d: the epilogue is BiasAct") };
    let mut ins = vec![
        GlSpec::new(&[1, 1, geom.batch * geom.h * geom.w, geom.cin], dt.clone()),
        GlSpec::new(&[1, 1, n, k], dt.clone()),
        GlSpec::new(&[1, 1, 1, n], dt.clone()),
    ];
    if residual.is_some() {
        ins.push(GlSpec::new(&[1, 1, m, n], dt.clone()));
    }
    let (outs, ins) = ker.bind_abi(&[GlSpec::new(&[1, 1, m, n], dt)], &ins);
    let epi = Epilogue::BiasAct { bias: ins[2].clone(), residual: ins.get(3).cloned(), act };
    let rows = row_source(geom, cfg);
    gemm_core_with(ker, (m, k, n), cfg, outs[0].clone(), ins[0].clone(), ins[1].clone(), epi, Some(&*rows));
}

/// **Graph-native** channels-last convolution: `x[batch, h, w, cin]` and
/// `w[cout, kh, kw, cin]` (both statically shaped, **bf16 or f16**), `bias[cout]`,
/// an optional `residual[batch, ho, wo, cout]`, a square `stride` and symmetric
/// `pad`, and `act` for SiLU after the bias. Returns the lazy
/// `y[batch, ho, wo, cout]` in the operand dtype, accumulated in f32 and rounded
/// once, the bias, activation and residual applied in the store.
///
/// The outcome is three-way, as [`crate::gemm_nt`]'s:
///
/// - `Ok(None)` — the device is not one of [`CONV_SUPPORTED_ARCHS`], or no tile
///   of its table fits ([`ConvGeom::tiles`]: `cin` a multiple of the strip depth,
///   `cout` of the tile's N edge, at least two strips of K). The caller keeps
///   its graph conv.
/// - `Err` — a symbolic dim, a rank other than 4/4/1, a dtype outside
///   {bf16, f16} or one that differs between operands, `w`'s `cin` disagreeing
///   with `x`'s, or a residual whose shape is not `y`'s.
/// - `Ok(Some(y))` — it ran.
pub fn conv2d_nhwc(
    x: &Tensor,
    w: &Tensor,
    bias: &Tensor,
    residual: Option<&Tensor>,
    stride: usize,
    pad: usize,
    act: bool,
) -> crate::LaunchResult<Option<Tensor>> {
    let xd = crate::launch::concrete_dims(x, "conv2d", "x", 4)?;
    let wd = crate::launch::concrete_dims(w, "conv2d", "w", 4)?;
    let bd = crate::launch::concrete_dims(bias, "conv2d", "bias", 1)?;
    let rd = residual.map(|r| crate::launch::concrete_dims(r, "conv2d", "residual", 4)).transpose()?;
    let geom =
        ConvGeom { batch: xd[0], h: xd[1], w: xd[2], cin: xd[3], cout: wd[0], kh: wd[1], kw: wd[2], stride, pad };
    // A dim pinned to one value is that value ([`crate::launch::pinned_dim`]),
    // but the tensor still carries it symbolically and a kernel placeholder
    // needs a static shape: reshape to what the dims resolved to.
    let statically = |t: &Tensor, dims: &[usize]| -> crate::LaunchResult<Tensor> {
        if t.shape().is_ok_and(|s| s.iter().all(|d| d.as_const().is_some())) {
            return Ok(t.clone());
        }
        t.try_reshape(dims.iter().map(|&d| d as isize).collect::<Vec<_>>()).context(crate::launch::OperandSnafu)
    };
    let dtype = x.uop().dtype();
    let y_shape = y_dims(&geom);
    let x = &statically(x, &xd)?;
    let residual = residual.map(|r| statically(r, &y_shape)).transpose()?;
    let residual = residual.as_ref();
    let kind = Epilogue::BiasAct { bias: (), residual: residual.map(|_| ()), act };
    let dtypes: Vec<(&'static str, DType)> = [("w", w.uop().dtype()), ("bias", bias.uop().dtype())]
        .into_iter()
        .chain(residual.map(|r| ("residual", r.uop().dtype())))
        .collect();
    let (err_dtype, want_res) = (dtype.clone(), y_shape.clone());
    // The chooser measures on first use, so it runs once per launch.
    let chosen: Rc<OnceCell<Option<GemmCfg>>> = Rc::default();
    let fit_chosen = chosen.clone();

    crate::launch_custom(
        &x.device(),
        CONV_SUPPORTED_ARCHS,
        move |_arch| {
            ensure!(
                err_dtype == DType::BFloat16 || err_dtype == DType::Float16,
                crate::launch::DtypeSnafu { kernel: "conv2d", got: err_dtype, expected: "bf16 or f16" }
            );
            for (_, dt) in &dtypes {
                ensure!(
                    *dt == err_dtype,
                    crate::launch::DtypeSnafu { kernel: "conv2d", got: dt.clone(), expected: "the dtype of x" }
                );
            }
            ensure!(
                wd[3] == geom.cin,
                crate::launch::OperandShapeSnafu {
                    kernel: "conv2d",
                    operand: "w",
                    expected: vec![geom.cout, geom.kh, geom.kw, geom.cin],
                    got: wd
                }
            );
            ensure!(
                bd[0] == geom.cout,
                crate::launch::OperandShapeSnafu {
                    kernel: "conv2d",
                    operand: "bias",
                    expected: vec![geom.cout],
                    got: bd
                }
            );
            if let Some(got) = rd {
                ensure!(
                    got == want_res,
                    crate::launch::OperandShapeSnafu { kernel: "conv2d", operand: "residual", expected: want_res, got }
                );
            }
            ensure!(
                stride > 0 && geom.h + 2 * pad >= geom.kh && geom.w + 2 * pad >= geom.kw,
                crate::launch::OperandShapeSnafu {
                    kernel: "conv2d",
                    operand: "x",
                    expected: vec![
                        geom.batch,
                        geom.kh.saturating_sub(2 * pad),
                        geom.kw.saturating_sub(2 * pad),
                        geom.cin
                    ],
                    got: xd
                }
            );
            Ok(())
        },
        {
            let (spec, dt) = (x.device(), dtype.clone());
            move |arch| fit_chosen.get_or_init(|| choose_conv_cfg(&spec, arch, &dt, geom, kind)).is_some()
        },
        move |arch| {
            let caps = crate::ArchCaps::for_arch(arch);
            let cfg =
                chosen.get_or_init(|| choose_conv_cfg(&x.device(), arch, &dtype, geom, kind)).expect("checked above");
            let (grid, block) = (geom.grid_dims(&cfg), cfg.threads(caps.wave_size));
            let out = Tensor::empty(&y_shape, dtype.clone());
            let name = match (residual.is_some(), act) {
                (false, false) => "conv2d_nhwc",
                (false, true) => "conv2d_nhwc_silu",
                (true, false) => "conv2d_nhwc_add",
                (true, true) => "conv2d_nhwc_silu_add",
            };
            let mut ins = vec![x, w, bias];
            ins.extend(residual);
            crate::graph_launch(name, grid, block, out, &ins, caps, move |ker| {
                build_conv(ker, geom, cfg, dtype, kind);
                ker.finish(cfg.acc_m)
            })
        },
    )
}
