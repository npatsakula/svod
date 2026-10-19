//! Every kernel family lowered for targets this machine cannot run: the
//! programs the op layer would launch, rendered to LLVM text and, where the
//! toolchain is installed, compiled to the target's code (PTX assembled by
//! `ptxas`, AMDGPU code objects by clang). A missing tool skips, never fails.

use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::Arc;

use svod_codegen::Renderer;
use svod_codegen::llvm::LlvmTextRenderer;
use svod_dtype::default_device::default_device;
use svod_dtype::{AmdArch, CudaArch, DType, DeviceSpec, GpuArch, ScalarDType};
use svod_ir::{Op, UOp, ops};
use svod_tensor::Tensor;
use test_case::test_case;

use crate::atoms::Target;
use crate::build::{BF16, F16};
use crate::interp::{self, round_to};
use crate::ir::{ParamKind, Program};
use crate::kernels::Act;
use crate::kernels::Batch;
use crate::kernels::attention::{AttnMask, AttnSpec, Cache, CombineSpec, attention, combine};
use crate::kernels::conv::ConvSpec;
use crate::kernels::gemm::{Epilogue, GemmSpec, Scale, gemm};
use crate::kernels::heads::{HeadsSpec, Rope, heads};
use crate::kernels::rows::{Norm, NormCfg, NormSpec, norm};
use crate::launch::graph_launch_all;
use crate::lower::{Lowering, lower};
use crate::ops::config;

use super::conv::geom;

/// One launch of a family: its name, program and lowering.
pub(super) type Launch = (String, Program, Lowering);

/// The first candidate the op layer plans for each kernel family on `target`.
pub(super) fn families(target: &Target) -> Vec<Launch> {
    let mut out: Vec<Launch> = vec![];
    let mut push = |name: &str, prog: Program, lowering: Lowering| out.push((name.to_string(), prog, lowering));

    // Shapes stay small for the interpreter; the tiles are those planned
    // for a large GEMM, so the biggest register tiles are lowered too.
    let gemm_spec = |m, n, k, epilogue: Epilogue| {
        let cfg = config::gemm_candidates(target, 1, 4096, 4096, k, epilogue.gated)[0];
        GemmSpec { m, n, k, batch: Batch::Static(1), epilogue, cfg }
    };
    let plain = gemm_spec(130, 192, 128, Epilogue::DEFAULT);
    for (i, cfg) in config::gemm_candidates(target, 1, 4096, 4096, 4096, false).into_iter().enumerate() {
        let spec = GemmSpec { cfg, ..plain.clone() };
        push(&format!("gemm #{i} {:?}x{}", cfg.tile, cfg.stages), gemm::<BF16>(&spec), cfg.lowering(target.clone()));
    }
    let swiglu = gemm_spec(70, 64, 64, Epilogue { bias: true, act: Act::Silu, gated: true, ..Epilogue::DEFAULT });
    push("gemm swiglu", gemm::<F16>(&swiglu), swiglu.cfg.lowering(target.clone()));
    let half_step = Epilogue { residual: true, act: Act::Gelu, scale: Some(Scale::new(0.5)), ..Epilogue::DEFAULT };
    let half_step = GemmSpec {
        cfg: config::gemm_candidates(target, 1, 300, 512, 256, false)[0],
        ..gemm_spec(100, 96, 256, half_step)
    };
    push("gemm gelu scale residual", gemm::<BF16>(&half_step), half_step.cfg.lowering(target.clone()));

    let fa = |t, tk, d, mask, cache: Option<Cache>, batch| {
        let cfg = config::attention_candidates(target, d, t)[0];
        let splits = if cache.is_some() { 2 } else { 1 };
        let cfg = crate::kernels::attention::FaCfg { splits, ..cfg };
        AttnSpec { batch, t, tk, heads: 4, kv_heads: 2, d, mask, cache, scale: 0.125, cfg }
    };
    let causal = AttnMask { causal: true, ..AttnMask::default() };
    for d in [64, 128] {
        let spec = fa(100, 100, d, causal, None, Batch::Static(2));
        for (i, cfg) in config::attention_candidates(target, d, 100).into_iter().enumerate() {
            let spec = AttnSpec { cfg, ..spec.clone() };
            let name = format!("attention d{d} causal #{i} {}x{}x{}", cfg.bq, cfg.bkv, cfg.stages);
            push(&name, attention::<BF16>(&spec), cfg.lowering(target.clone()));
        }
    }
    let masked = AttnMask { key_lens: true, key_mask: true, ..AttnMask::default() };
    let spec = fa(37, 75, 64, masked, None, Batch::Var { name: "b".into(), min: 1, max: 2 });
    push("attention d64 key masks", attention::<F16>(&spec), spec.cfg.lowering(target.clone()));
    let cache = Cache { rows: 6, heads_total: 6, head_start: 2, row_map: true, appended: true };
    let lens = AttnMask { key_lens: true, ..AttnMask::default() };
    let spec = fa(1, 150, 64, lens, Some(cache), Batch::Var { name: "b".into(), min: 1, max: 3 });
    push("attention d64 cache", attention::<F16>(&spec), spec.cfg.lowering(target.clone()));
    let merge = CombineSpec { batch: spec.batch.clone(), t: 1, heads: 4, d: 64, splits: 2, cfg: NormCfg { br: 4 } };
    push("attention combine", combine::<F16>(&merge), merge.cfg.lowering(target.clone()));

    for (name, kind, d, residual, bias) in
        [("layer norm", Norm::Layer, 1024, true, true), ("rms norm", Norm::Rms, 512, false, false)]
    {
        let cfg = config::norm_candidates(target, d)[0];
        let spec = NormSpec { norm: kind, rows: 37, d, batch: Batch::Static(1), eps: 1e-5, residual, bias, cfg };
        push(name, norm::<BF16>(&spec), cfg.lowering(target.clone()));
    }

    for d in [64, 128] {
        let cfg = config::heads_candidates(target, d)[0];
        let spec = HeadsSpec {
            batch: Batch::Static(2),
            t: 20,
            heads: 4,
            kv_heads: 2,
            d,
            q_norm: true,
            k_norm: true,
            eps: 1e-6,
            rope: Some(Rope { per_batch: false }),
            cfg,
        };
        push(&format!("heads d{d}"), heads::<BF16>(&spec), cfg.lowering(target.clone()));
    }

    let silu_bias = Epilogue { bias: true, act: Act::Silu, ..Epilogue::DEFAULT };
    for (name, g, images) in [
        ("conv 3x3", geom([12, 12], 64, 64, 3, 1, 1, 1), 2),
        ("conv 3x3 stride 2 split", geom([16, 16], 128, 64, 3, 2, 1, 1), 1),
    ] {
        let [ho, wo] = g.out_hw();
        let cfgs = config::conv_candidates(target, 1, images * ho * wo, &g);
        let c = cfgs.iter().find(|c| (c.split > 1) == name.ends_with("split")).unwrap_or(&cfgs[0]);
        let spec = ConvSpec { batch: Batch::Static(images), geom: g, epilogue: silu_bias, cfg: c.gemm, split: c.split };
        for (i, (prog, lowering)) in spec.programs::<F16>(target).into_iter().enumerate() {
            push(&format!("{name} #{i}"), prog, lowering);
        }
    }
    out
}

fn device(target: &Target) -> DeviceSpec {
    match target.arch {
        GpuArch::Cuda(_) => DeviceSpec::Cuda { device_id: 0 },
        GpuArch::Amd(_) => DeviceSpec::Amd { device_id: 0 },
        GpuArch::Metal(_) => DeviceSpec::Metal { device_id: 0 },
    }
}

/// The lowered instruction list of `prog`.
pub(super) fn linear(prog: Program, lowering: &Lowering) -> Arc<UOp> {
    let params =
        prog.params.iter().enumerate().map(|(i, p)| UOp::param(i, p.elems, DType::Scalar(p.dtype), None)).collect();
    let name = prog.name.clone();
    let lowered = lower(prog, lowering, params, device(&lowering.target)).unwrap_or_else(|e| panic!("{name}: {e}"));
    let Op::Program(ops::Program { linear: Some(linear), .. }) = lowered.program.op() else { panic!("a program") };
    linear.clone()
}

/// Random parameters of `prog` rounded to their types; every i32 parameter
/// (key counts, masks, segment starts, row maps) is 1.
fn inputs(name: &str, prog: &Program) -> Vec<Vec<f64>> {
    let mut seed = name.len() as u64;
    let params = prog.params.iter().map(|p| match p.dtype {
        ScalarDType::Int32 => vec![1.0; p.elems],
        dtype => (0..p.elems)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                round_to(dtype, ((seed >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0)
            })
            .collect(),
    });
    params.collect()
}

/// The tile program after `lowering` for interpretation.
fn lowered_tile(prog: Program, lowering: &Lowering) -> Program {
    let params =
        prog.params.iter().enumerate().map(|(i, p)| UOp::param(i, p.elems, DType::Scalar(p.dtype), None)).collect();
    lower(prog, lowering, params, device(&lowering.target)).unwrap().tile
}

/// LLVM text of `prog` for its lowering's target.
pub(super) fn render(prog: Program, lowering: &Lowering) -> String {
    let name = prog.name.clone();
    let renderer = match lowering.target.arch {
        GpuArch::Cuda(arch) => LlvmTextRenderer::nvptx(arch),
        GpuArch::Amd(arch) => LlvmTextRenderer::amd(arch),
        GpuArch::Metal(_) => unreachable!("Metal renders C"),
    };
    let linear = linear(prog, lowering);
    renderer.render(&linear, Some(&name)).unwrap_or_else(|e| panic!("{name}: {e}")).code
}

fn run(mut command: Command, input: &[u8], what: &str) -> Result<Vec<u8>, String> {
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("{what}: {e}"))?;
    child.stdin.take().expect("stdin").write_all(input).map_err(|e| format!("{what}: {e}"))?;
    let out = child.wait_with_output().map_err(|e| format!("{what}: {e}"))?;
    if out.status.success() {
        Ok(out.stdout)
    } else {
        Err(format!("{what} failed: {}", String::from_utf8_lossy(&out.stderr)))
    }
}

fn ptxas() -> Option<std::path::PathBuf> {
    let on_path = std::env::var_os("PATH")
        .into_iter()
        .flat_map(|p| std::env::split_paths(&p).collect::<Vec<_>>())
        .map(|dir| dir.join("ptxas"));
    let toolkits = ["/opt/cuda/bin/ptxas", "/usr/local/cuda/bin/ptxas"].map(std::path::PathBuf::from);
    on_path.chain(toolkits).find(|p| p.is_file())
}

pub(super) fn hopper() -> Target {
    Target { sms: Some(132), ..Target::for_arch(GpuArch::Cuda(CudaArch { major: 9, minor: 0 })) }
}

/// The `mma.sync` + `cp.async` path on Hopper: every family lowers for
/// sm_90, clang emits its PTX, and `ptxas` assembles it for `sm_90` and
/// `sm_90a` (nothing in it needs the arch-specific features).
#[test]
fn every_family_assembles_for_hopper() {
    let target = hopper();
    assert!(config::has_tables(&target));
    assert_eq!(target.smem_bytes, 227 << 10);
    let Some(ptxas) = ptxas() else {
        for (_, prog, lowering) in families(&target) {
            render(prog, &lowering);
        }
        eprintln!("skipped: no ptxas; rendered only");
        return;
    };
    let GpuArch::Cuda(arch) = target.arch else { unreachable!() };
    for (name, prog, lowering) in families(&target) {
        let ir = render(prog, &lowering);
        let ptx = match svod_runtime::cuda::compile_ir_to_ptx(&ir, arch) {
            Ok(ptx) => ptx,
            Err(e) if e.to_string().contains("NVPTX target") => {
                eprintln!("skipped: clang has no NVPTX target");
                return;
            }
            Err(e) => panic!("{name}: {e}"),
        };
        for sm in ["sm_90", "sm_90a"] {
            let mut command = Command::new(&ptxas);
            command.args([&format!("-arch={sm}"), "-o", "/dev/null", "/dev/stdin"]);
            run(command, &ptx, &format!("ptxas {sm} {name}")).unwrap_or_else(|e| panic!("{e}"));
            eprintln!("{name}: assembled for {sm}");
        }
    }
}

/// The interpreter gives the same result for a family's program before and
/// after its lowering for `target` (schedule expansion, operand loads,
/// barriers, relayouts): the bookkeeping the target's template adds keeps
/// the numerics.
#[test_case(hopper(); "sm_90")]
#[test_case(amd(AmdArch::Gfx1201, 64); "gfx1201")]
#[test_case(amd(AmdArch::Gfx1151, 40); "gfx1151")]
#[test_case(amd(AmdArch::Gfx1100, 96); "gfx1100")]
#[test_case(amd(AmdArch::Gfx942, 304); "gfx942")]
fn lowering_keeps_the_interpreted_result(target: Target) {
    for (name, prog, lowering) in families(&target) {
        let params = inputs(&name, &prog);
        let vars: Vec<(&str, i64)> = prog.vars.iter().map(|v| (v.name.as_str(), v.max)).collect();
        let want = interp::run(&prog, params.clone(), &vars).unwrap();
        let tile = lowered_tile(prog.clone(), &lowering);
        let got = interp::run(&tile, params, &vars).unwrap();
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            let same = g.iter().zip(w).all(|(a, b)| a == b || (a.is_nan() && b.is_nan()));
            assert!(same, "{name}: parameter {i} differs after lowering");
        }
    }
}

/// An AMD GPU with `cus` compute units and 64 KB of LDS per workgroup.
pub(super) fn amd(arch: AmdArch, cus: u32) -> Target {
    Target { sms: Some(cus), ..Target::for_arch(GpuArch::Amd(arch)) }
}

/// RDNA WMMA and CDNA MFMA with register-staged fills: every family lowers
/// for the target and, where clang has the AMDGPU backend, compiles to a
/// code object holding the kernel's descriptor.
#[test_case(amd(AmdArch::Gfx1201, 64); "gfx1201")]
#[test_case(amd(AmdArch::Gfx1151, 40); "gfx1151")]
#[test_case(amd(AmdArch::Gfx1100, 96); "gfx1100")]
#[test_case(amd(AmdArch::Gfx942, 304); "gfx942")]
fn every_family_compiles_for_amd(target: Target) {
    assert!(config::has_tables(&target));
    let GpuArch::Amd(arch) = target.arch else { unreachable!() };
    for (name, prog, lowering) in families(&target) {
        let kernel = prog.name.clone();
        let ir = render(prog, &lowering);
        let object = match svod_runtime::amd::compile_ir_to_amd_object(&ir, arch) {
            Ok(object) => object,
            Err(e) if e.to_string().contains("AMDGPU target") || e.to_string().contains("too old") => {
                eprintln!("skipped: {e}");
                return;
            }
            Err(e) => panic!("{name}: {e}"),
        };
        assert_eq!(&object[..4], b"\x7fELF", "{name}: an ELF code object");
        let descriptor = format!("{kernel}.kd");
        assert!(object.windows(descriptor.len()).any(|w| w == descriptor.as_bytes()), "{name}: {descriptor}");
        eprintln!("{name}: compiled for {}", arch.mcpu());
    }
}

/// Every output parameter of `prog` after launching it under `lowering` on
/// the default device, `None` for inputs.
fn on_device(prog: Program, lowering: &Lowering, params: &[Vec<f64>]) -> Vec<Option<Vec<f64>>> {
    let decl = prog.params.clone();
    let vars: Vec<(String, i64)> = prog.vars.iter().map(|v| (v.name.clone(), v.max)).collect();
    let vars: Vec<(&str, i64)> = vars.iter().map(|(n, v)| (n.as_str(), *v)).collect();
    let tensors: Vec<Tensor> = decl
        .iter()
        .zip(params)
        .map(|(p, v)| match p.dtype {
            ScalarDType::Int32 => Tensor::from_slice(v.iter().map(|&x| x as i32).collect::<Vec<_>>()),
            dtype => Tensor::from_slice(v.iter().map(|&x| x as f32).collect::<Vec<_>>()).cast(DType::Scalar(dtype)),
        })
        .collect();
    let refs: Vec<&Tensor> = tensors.iter().collect();
    let outs = graph_launch_all(prog, lowering, &refs).unwrap();
    decl.iter()
        .zip(outs)
        .map(|(p, t)| {
            (p.kind != ParamKind::In).then(|| {
                t.prepare().unwrap().execute_with_vars(&vars).unwrap();
                let mut bytes = vec![0u8; p.elems * p.dtype.bytes()];
                t.buffer().unwrap().copyout(&mut bytes).unwrap();
                let half = |b: &[u8]| u16::from_le_bytes([b[0], b[1]]);
                match p.dtype {
                    ScalarDType::Float32 => {
                        bytes.chunks(4).map(|b| f32::from_le_bytes(b.try_into().unwrap()) as f64).collect()
                    }
                    ScalarDType::BFloat16 => {
                        bytes.chunks(2).map(|b| f32::from_bits(u32::from(half(b)) << 16) as f64).collect()
                    }
                    ScalarDType::Float16 => bytes.chunks(2).map(|b| f16_bits(half(b))).collect(),
                    other => unreachable!("{other:?} output"),
                }
            })
        })
        .collect()
}

fn f16_bits(h: u16) -> f64 {
    let (sign, exp, frac) = (if h >> 15 == 1 { -1.0 } else { 1.0 }, (h >> 10) & 0x1f, f64::from(h & 0x3ff));
    sign * match exp {
        0 => frac * 2f64.powi(-24),
        31 if frac == 0.0 => f64::INFINITY,
        31 => f64::NAN,
        e => (1.0 + frac / 1024.0) * 2f64.powi(i32::from(e) - 15),
    }
}

/// Each family on the GPU against the interpreter: within two 16-bit ulps
/// of the output scale (accumulation order differs), NaN where it is NaN.
fn check_on_device(target: &Target) {
    for (name, prog, lowering) in families(target) {
        let params = inputs(&name, &prog);
        let vars: Vec<(&str, i64)> = prog.vars.iter().map(|v| (v.name.as_str(), v.max)).collect();
        let want = interp::run(&prog, params.clone(), &vars).unwrap();
        let got = on_device(prog, &lowering, &params);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            let Some(g) = g else { continue };
            let bad = g
                .iter()
                .zip(w)
                .position(|(g, w)| !(g.is_nan() && w.is_nan() || (g - w).abs() <= 2e-2 * w.abs().max(1.0)));
            let worst = g.iter().zip(w).map(|(g, w)| (g - w).abs()).fold(0.0, f64::max);
            eprintln!("{name}: output {i} max abs diff {worst:.3e}");
            assert!(bad.is_none(), "{name}: output {i}[{}] = {} vs {}", bad.unwrap(), g[bad.unwrap()], w[bad.unwrap()]);
        }
    }
}

/// The device's own target (any vendor with tables): every family's first
/// candidate computes what the interpreter computes. The remote gate for a
/// new GPU; skips without one.
#[test]
fn families_match_the_interpreter_on_the_device() {
    let device = default_device();
    let Some(target) = Target::for_device(&device).filter(config::has_tables) else {
        eprintln!("skipped: no device with tk3 tables");
        return;
    };
    check_on_device(&target);
}

/// The RDNA data path on an NVIDIA GPU: register-staged fills into the
/// chunk layout, vector shared-memory gathers instead of `ldmatrix`, and
/// two-slot rings, measured against the interpreter.
#[test]
fn register_staged_families_match_on_cuda() {
    let device = default_device();
    let Some(target) = matches!(device, DeviceSpec::Cuda { .. }).then(|| Target::for_device(&device)).flatten() else {
        eprintln!("skipped: no CUDA device");
        return;
    };
    check_on_device(&Target { cp_async: false, ldmatrix: false, ..target });
}

/// Every candidate the op layer may plan on AMD lowers (the tune store
/// would otherwise time a failure): the GEMM lists of a large, a small-M and
/// a ragged shape, the attention lists of every head size and both query
/// regimes, and the convolution lists of every YOLO class.
#[test_case(amd(AmdArch::Gfx1201, 64); "gfx1201")]
#[test_case(amd(AmdArch::Gfx1151, 40); "gfx1151")]
#[test_case(amd(AmdArch::Gfx942, 304); "gfx942")]
fn every_amd_candidate_lowers(target: Target) {
    let mut gemms = vec![];
    for (m, n, k) in [(4096, 4096, 4096), (704, 512, 512), (1500, 1280, 5120), (37, 96, 48)] {
        for gated in [false, true] {
            for cfg in config::gemm_candidates(&target, 1, m, n, k, gated) {
                let epilogue = Epilogue { gated, ..Epilogue::DEFAULT };
                gemms.push(GemmSpec { m: 100, n: 96, k: cfg.tile[2] * 2, batch: Batch::Static(1), epilogue, cfg });
            }
        }
    }
    for spec in gemms {
        linear(gemm::<BF16>(&spec), &spec.cfg.lowering(target.clone()));
    }
    for d in [48, 64, 128] {
        for t in [1, 16, 100] {
            for cfg in config::attention_candidates(&target, d, t) {
                let spec = AttnSpec {
                    batch: Batch::Static(1),
                    t,
                    tk: 100,
                    heads: 2,
                    kv_heads: 1,
                    d,
                    mask: AttnMask { causal: t > 1, ..AttnMask::default() },
                    cache: None,
                    scale: 0.125,
                    cfg,
                };
                linear(attention::<BF16>(&spec), &cfg.lowering(target.clone()));
            }
        }
    }
    for class in super::conv::YOLO {
        let g = super::conv::yolo_geom(class);
        let [ho, wo] = g.out_hw();
        for c in config::conv_candidates(&target, 1, ho * wo, &g) {
            let spec =
                ConvSpec { batch: Batch::Static(1), geom: g, epilogue: Epilogue::DEFAULT, cfg: c.gemm, split: c.split };
            for (prog, lowering) in spec.programs::<F16>(&target) {
                linear(prog, &lowering);
            }
        }
    }
}
