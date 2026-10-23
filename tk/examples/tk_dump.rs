//! What tk would run on a card this machine has no device for: the planner's
//! ranking (predicted ns, and why each config of the lattice was left out),
//! the chosen config's LLVM text and, where the tools are installed, its
//! PTX, `ptxas -v` resources and SASS (CUDA) or AMDGCN assembly and its
//! resource comments (AMD), with an opcode histogram of the innermost loop
//! that issues matrix instructions: the main loop.
//!
//! ```text
//! cargo run -p svod-tk --release --example tk_dump -- --arch sm_90a --sms 114 \
//!     --op gemm --shape 4096x4096x4096 [--dtype bf16] [--cfg 0|tuned] [--store DIR] [--out DIR]
//! cargo run -p svod-tk --release --example tk_dump -- --card mi300x --op attention --shape 8x16x1024x128 --causal
//! ```
//!
//! `--op` and its `--shape`: `gemm MxNxK` (`--bias`, `--residual`, `--gated`,
//! `--act silu|gelu`), `attention BxHxTxD` (`--causal`), `norm RxD` (`--norm
//! rms|layer`, `--bias`, `--residual`), `conv BxHxWxCINxCOUT` (`--kernel K`,
//! `--stride S`, `--pad P`, `--dilation D`, `--bias`, `--residual`, `--act`),
//! `heads BxTxHxKVxD` (`--qk-norm`, `--rope`), `nearest NxMxD` (`--metric
//! squared-l2|shifted-l2|l2|neg-dot`, `--row-floor`, `--col-floor`,
//! `--groups`, `--k K` least scores a row). Each goes through the op layer's own shape gate
//! (`ops::shape`) and tuning (`ops::tuned`).
//!
//! `--compare FILE [--fun NAME]` sets the main loop of a reference kernel
//! beside tk's, counted by the same [`svod_tk::isa::main_loop`]: a
//! `.cubin` (`nvdisasm -c`), an AMD code object `.co`/`.hsaco`
//! (`llvm-objdump -d --symbolize-operands` for the target's mcpu), or a
//! SASS/PTX/AMDGCN text; `--fun` picks one kernel of several (a cubin
//! extracted from cuBLASLt holds dozens).
//!
//! `--arch` takes the SMs and dies of the first card of `svod_tk::atoms::CARDS`
//! of the architecture unless `--sms`/`--xcds` say otherwise; `--store` is a
//! tune store directory copied from the card (its probe lines price the
//! ranking), else the guessed costs do. `--cfg I` emits the ranking's `I`th
//! config; `--cfg tuned` the one the tuner chose on the card, read from the
//! store's `admitted`/`choice` lines ([`TuneStore::replay`]): a line matches
//! by op, device, dtype, shape and the digest of the salt and config list
//! the op layer keys it by, whatever compiler and kernel build wrote it, so
//! the ranking must be the card's (its stored costs). A GEMM reads `linear`'s
//! lines, else those the device tests and GEMM probes tuned (`gemm`,
//! `ops::DEVICE_TEST`). `RUST_LOG=svod_tk::plan=debug` logs the planner's
//! decisions.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use svod_dtype::{AmdArch, CudaArch, DType, GpuArch};
use svod_tk::atoms::{CARDS, Target};
use svod_tk::build::{BF16, F16};
use svod_tk::ir::Program;
use svod_tk::isa;
use svod_tk::kernels::attention::{AttnMask, AttnSpec, attention};
use svod_tk::kernels::conv::{ConvGeom, ConvSpec};
use svod_tk::kernels::gemm::{Epilogue, GemmSpec, gemm};
use svod_tk::kernels::heads::{HeadsSpec, Rope, heads};
use svod_tk::kernels::nearest::{Floors, Metric, NearestSpec, nearest};
use svod_tk::kernels::rows::{Norm, NormSpec, norm};
use svod_tk::kernels::{Act, Batch};
use svod_tk::lower::Lowering;
use svod_tk::ops::config::{AttentionProblem, GemmProblem, NearestProblem, Planner, gemm_lattice};
use svod_tk::ops::shape::{self, Extent, Plan};
use svod_tk::ops::tuned;
use svod_tk::probe::{self, Measured};
use svod_tk::tune::{Found, TuneStore};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// What `--cfg` asks for.
#[derive(Clone, Copy)]
enum Pick {
    Ranked(usize),
    Tuned,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter(tracing_subscriber::EnvFilter::from_default_env()).init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| args.iter().any(|a| a == name);
    let opt = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(String::as_str);
    let num = |name: &str| opt(name).map(str::parse::<usize>).transpose();

    let target = target(opt("--card"), opt("--arch"), num("--sms")?, num("--xcds")?)?;
    let device = format!("{}-{}sm", target.arch.target_name(), target.sms.unwrap_or(0));
    let stored = opt("--store").and_then(|dir| probe::stored(Path::new(dir), &target));
    let measured = match stored {
        Some(m) => {
            println!("costs: {device} from {}, unmeasured {:?}", opt("--store").unwrap_or_default(), m.unmeasured);
            m
        }
        None => {
            println!("costs: GUESSED fixture (no stored costs for {device}): the ranking is a guess's");
            Measured { costs: probe::GUESSED, unmeasured: vec![] }
        }
    };
    let planner = Planner::measured(target.clone(), measured)
        .ok_or("the planner needs the card's SM count: pass --sms or --card")?;
    let dims: Vec<usize> =
        opt("--shape").ok_or("--shape")?.split('x').map(str::parse).collect::<std::result::Result<_, _>>()?;
    let dtype = if opt("--dtype").unwrap_or("bf16") == "bf16" { DType::BFloat16 } else { DType::Float16 };
    let bf16 = dtype == DType::BFloat16;
    let out = PathBuf::from(opt("--out").unwrap_or("tk_dump"));
    std::fs::create_dir_all(&out)?;
    let top = num("--top")?.unwrap_or(16);
    let pick = match opt("--cfg") {
        Some("tuned") => Pick::Tuned,
        Some(i) => Pick::Ranked(i.parse()?),
        None => Pick::Ranked(0),
    };
    let store = || -> Result<TuneStore> {
        let dir = opt("--store").ok_or("--cfg tuned reads the tuner's choice from --store DIR")?;
        Ok(TuneStore::replay(PathBuf::from(dir)))
    };
    let act = match opt("--act") {
        None => Act::None,
        Some("silu") => Act::Silu,
        Some("gelu") => Act::Gelu,
        Some(other) => return Err(format!("--act {other}: silu or gelu").into()),
    };
    let epilogue = Epilogue { bias: flag("--bias"), act, residual: flag("--residual"), ..Epilogue::DEFAULT };
    let extent = |dims: &[usize]| Extent { dims: dims.to_vec(), var: false };
    let gate = |op: &'static str| move |why| format!("{op}: the op layer takes the graph ({why:?})");

    let mut plan = String::new();
    let programs: Vec<(Program, Lowering)> = match (opt("--op").ok_or("--op")?, &dims[..]) {
        ("gemm", &[m, n, k]) => {
            let gated = flag("--gated");
            let epilogue = Epilogue { gated, ..epilogue };
            let operands = vec![dtype.clone(); 2 + usize::from(epilogue.bias) + usize::from(epilogue.residual)];
            let ranked = kernel(shape::linear(Some(&planner), &operands, Some(&extent(&[m, k])), n, gated))
                .map_err(gate("linear"))?;
            let p = GemmProblem { batches: 1, m, n, k, halves: if gated { 2 } else { 1 } };
            report(&mut plan, top, &ranked, &gemm_lattice(&target), |c| p.admit(&planner, c));
            let spec = GemmSpec { m, n, k, batch: Batch::Static(1), epilogue, cfg: ranked[0] };
            let spec = match pick {
                Pick::Ranked(i) => GemmSpec { cfg: *ranked.get(i).ok_or("no such ranked config")?, ..spec },
                Pick::Tuned => {
                    let linear = |s: &TuneStore| tuned::linear(s, &planner, dtype.clone(), &ranked, spec.clone());
                    let probed = |s: &TuneStore| tuned::gemm_probe(s, &planner, dtype.clone(), &ranked, spec.clone());
                    replayed(&store()?, "linear", linear).or_else(|why| {
                        println!("{why}\ntrying the device tests' and GEMM probes' tuning (op gemm)");
                        replayed(&store()?, "gemm", probed)
                    })?
                }
            };
            println!("emitting {:?}", spec.cfg);
            vec![(if bf16 { gemm::<BF16>(&spec) } else { gemm::<F16>(&spec) }, spec.cfg.lowering(&planner.device))]
        }
        ("attention", &[b, h, t, d]) => {
            let causal = flag("--causal");
            let qkv = extent(&[b, t, h, d]);
            let operands = [dtype.clone(), dtype.clone(), dtype.clone()];
            let ranked = kernel(shape::attention(Some(&planner), &operands, Some(&qkv), Some(&qkv), causal, false))
                .map_err(gate("attention"))?;
            let a = AttentionProblem { tiles: b * h, t, tk: t, d, causal, bias: false };
            report(&mut plan, top, &ranked, &a.lattice(&target), |c| a.admit(&planner, c));
            let mask = AttnMask { causal, ..AttnMask::default() };
            let scale = 1.0 / (d as f32).sqrt();
            let cfg = ranked[0];
            let spec =
                AttnSpec { batch: Batch::Static(b), t, tk: t, heads: h, kv_heads: h, d, mask, cache: None, scale, cfg };
            let spec = match pick {
                Pick::Ranked(i) => AttnSpec { cfg: *ranked.get(i).ok_or("no such ranked config")?, ..spec },
                Pick::Tuned => replayed(&store()?, "attention", |s| {
                    tuned::attention(s, &planner, dtype.clone(), &ranked, spec.clone(), None)
                })?,
            };
            println!("emitting {:?}", spec.cfg);
            let fa = if bf16 { attention::<BF16>(&spec) } else { attention::<F16>(&spec) };
            vec![(fa, spec.cfg.lowering(&planner.device))]
        }
        ("norm", &[rows, d]) => {
            let kind = match opt("--norm").unwrap_or("rms") {
                "rms" => Norm::Rms,
                "layer" => Norm::Layer,
                other => return Err(format!("--norm {other}: rms or layer").into()),
            };
            let (residual, bias) = (flag("--residual"), flag("--bias") && kind == Norm::Layer);
            let operands = vec![dtype.clone(); 2 + usize::from(residual) + usize::from(bias)];
            let ranked =
                kernel(shape::norm(Some(&planner), &operands, Some(&extent(&[rows, d])))).map_err(gate("norm"))?;
            listed(&mut plan, &ranked);
            let cfg = ranked[0];
            let spec = NormSpec { norm: kind, rows, d, batch: Batch::Static(1), eps: 1e-5, residual, bias, cfg };
            let spec = match pick {
                Pick::Ranked(i) => NormSpec { cfg: *ranked.get(i).ok_or("no such config")?, ..spec },
                Pick::Tuned => {
                    replayed(&store()?, "norm", |s| tuned::norm(s, &planner, dtype.clone(), &ranked, spec.clone()))?
                }
            };
            println!("emitting {:?}", spec.cfg);
            vec![(if bf16 { norm::<BF16>(&spec) } else { norm::<F16>(&spec) }, spec.cfg.lowering(&planner.device))]
        }
        ("conv", &[b, h, w, cin, cout]) => {
            let k = num("--kernel")?.unwrap_or(3);
            let [stride, pad, dilation] =
                [num("--stride")?.unwrap_or(1), num("--pad")?.unwrap_or(k / 2), num("--dilation")?.unwrap_or(1)];
            let geom = ConvGeom {
                h,
                w,
                cin,
                cout,
                kernel: [k, k],
                stride: [stride; 2],
                pad: [pad; 2],
                dilation: [dilation; 2],
            };
            if k == 1 && stride == 1 && pad == 0 {
                return Err("a pointwise convolution runs as a linear: --op gemm".into());
            }
            let operands = vec![dtype.clone(); 2 + usize::from(epilogue.bias)];
            let residual = epilogue.residual.then(|| dtype.clone());
            let x = extent(&[b, h, w, cin]);
            let ranked = kernel(shape::conv2d(Some(&planner), &operands, dtype.clone(), residual, Some(&x), &geom, 1))
                .map_err(gate("conv2d"))?;
            let (batches, m) = shape::conv_grid(&x, &geom);
            let p = GemmProblem { batches, m, n: cout, k: geom.k(), halves: 1 };
            let gemms: Vec<_> = ranked.iter().map(|c| c.gemm).collect();
            report(&mut plan, top, &gemms, &[], |c| p.admit(&planner, c));
            let c = ranked[0];
            let spec = ConvSpec { batch: Batch::Static(b), geom, epilogue, cfg: c.gemm, split: c.split };
            let spec = match pick {
                Pick::Ranked(i) => {
                    let c = *ranked.get(i).ok_or("no such ranked config")?;
                    ConvSpec { cfg: c.gemm, split: c.split, ..spec }
                }
                Pick::Tuned => {
                    replayed(&store()?, "conv2d", |s| tuned::conv(s, &planner, dtype.clone(), &ranked, spec.clone()))?
                }
            };
            println!("emitting {:?} split {}", spec.cfg, spec.split);
            if bf16 { spec.programs::<BF16>(&planner.device) } else { spec.programs::<F16>(&planner.device) }
        }
        ("heads", &[b, t, heads_, kv_heads, d]) => {
            let (qk_norm, rope) = (flag("--qk-norm"), flag("--rope"));
            let operands = vec![dtype.clone(); 1 + 2 * usize::from(qk_norm) + 2 * usize::from(rope)];
            let x = extent(&[b, t, (heads_ + 2 * kv_heads) * d]);
            let ranked = kernel(shape::heads(Some(&planner), &operands, Some(&x), d)).map_err(gate("heads"))?;
            listed(&mut plan, &ranked);
            let spec = HeadsSpec {
                batch: Batch::Static(b),
                t,
                heads: heads_,
                kv_heads,
                d,
                q_norm: qk_norm,
                k_norm: qk_norm,
                eps: 1e-6,
                rope: rope.then_some(Rope { per_batch: false }),
                cfg: ranked[0],
            };
            let spec = match pick {
                Pick::Ranked(i) => HeadsSpec { cfg: *ranked.get(i).ok_or("no such config")?, ..spec },
                Pick::Tuned => {
                    replayed(&store()?, "heads", |s| tuned::heads(s, &planner, dtype.clone(), &ranked, spec.clone()))?
                }
            };
            println!("emitting {:?}", spec.cfg);
            vec![(if bf16 { heads::<BF16>(&spec) } else { heads::<F16>(&spec) }, spec.cfg.lowering(&planner.device))]
        }
        ("nearest", &[n, m, d]) => {
            let metric = match opt("--metric").unwrap_or("squared-l2") {
                "squared-l2" => Metric::SquaredL2,
                "shifted-l2" => Metric::ShiftedL2,
                "l2" => Metric::L2,
                "neg-dot" => Metric::NegDot,
                other => return Err(format!("--metric {other}: squared-l2, shifted-l2, l2 or neg-dot").into()),
            };
            let (floors, groups) = (Floors { row: flag("--row-floor"), col: flag("--col-floor") }, flag("--groups"));
            let k = num("--k")?.unwrap_or(1);
            let operands = [dtype.clone(), dtype.clone()];
            let (x, c) = (extent(&[n, d]), extent(&[m, d]));
            let ranked =
                kernel(shape::nearest(Some(&planner), &operands, Some(&x), Some(&c), metric, floors, groups, k))
                    .map_err(gate("nearest"))?;
            let scalar = dtype.scalar().ok_or("a scalar dtype")?;
            let d = planner.nearest_depth(d, scalar).ok_or("no matrix core")?;
            let p = NearestProblem { n, m, d, dtype: scalar, metric, floors, groups, k };
            report(&mut plan, top, &ranked, &p.lattice(&target), |c| p.admit(&planner, c));
            let spec = NearestSpec { n, m, d, metric, floors, groups, k, cfg: ranked[0], live: Default::default() };
            let spec = match pick {
                Pick::Ranked(i) => NearestSpec { cfg: *ranked.get(i).ok_or("no such ranked config")?, ..spec },
                Pick::Tuned => replayed(&store()?, "nearest", |s| {
                    tuned::nearest(s, &planner, dtype.clone(), &ranked, spec.clone())
                })?,
            };
            println!("emitting {:?} at depth {d}", spec.cfg);
            let prog = if bf16 { nearest::<BF16>(&spec) } else { nearest::<F16>(&spec) };
            vec![(prog, spec.cfg.lowering(&planner.device))]
        }
        (op, dims) => {
            return Err(format!(
                "{op} over {dims:?}: gemm takes MxNxK, attention BxHxTxD, norm RxD, conv BxHxWxCINxCOUT, heads BxTxHxKVxD, nearest NxMxD"
            )
            .into());
        }
    };
    std::fs::write(out.join("plan.txt"), plan)?;
    if programs.len() > 1 {
        println!("{} programs (a split and its merge): the first is emitted", programs.len());
    }
    let (prog, lowering) = programs.into_iter().next().expect("a program");

    let ir = svod_tk::lower::standalone(prog, &lowering)?.text(&target)?;
    write(&out, "kernel.ll", ir.as_bytes())?;
    let (asm, name) = match target.arch {
        GpuArch::Cuda(arch) => (cuda(&out, &ir, arch)?, "SASS"),
        GpuArch::Amd(arch) => (amd(&out, &ir, arch)?, "AMDGCN"),
        GpuArch::Metal(_) => return Err("Metal renders no LLVM text".into()),
    };
    let ours = asm.and_then(|asm| histogram(&asm, name));
    if let Some(path) = opt("--compare") {
        let path = Path::new(path);
        let theirs = reference(path, opt("--fun"), &target, &out)?;
        let label = file_name(path);
        let what = opt("--fun").map_or_else(|| label.clone(), |f| format!("{label}, function {f}"));
        let theirs = isa::main_loop(&theirs).ok_or_else(|| format!("{what}: no loop issues a matrix instruction"))?;
        let ours = ours.ok_or("tk's kernel has no main loop to compare")?;
        println!("main loop per trip, tk's {name} against {what}:");
        print!("{}", isa::table(&[("tk", &ours), (&label, &theirs)]));
    }
    Ok(())
}

fn file_name(path: &Path) -> String {
    path.file_name().map_or_else(|| path.display().to_string(), |n| n.to_string_lossy().into_owned())
}

/// The ranked configs of a kernel plan; the fallback otherwise.
fn kernel<C>(plan: Plan<C>) -> std::result::Result<std::sync::Arc<[C]>, shape::Fallback> {
    match plan {
        Plan::Kernel(ranked) => Ok(ranked),
        Plan::Graph(why) => Err(why),
    }
}

/// What `tune` picked through a replay of the tune store, each key it read
/// printed with what it found; an error when a key had no line, since the
/// pick is then not the tuner's.
fn replayed<S>(store: &TuneStore, op: &str, tune: impl FnOnce(&TuneStore) -> S) -> Result<S> {
    let spec = tune(store);
    let lookups = store.lookups();
    println!("tuned {op}: {} store keys read", lookups.len());
    for l in &lookups {
        let entry = l.entry.as_ref().map_or("-".to_string(), |e| format!("{e:?}"));
        let entry: String = entry.chars().take(96).collect();
        println!("  {:<8} {:<28} {}  {entry}", l.kind.name(), format!("{:?}", l.found), l.key);
    }
    let missing: Vec<_> = lookups.iter().filter(|l| matches!(l.found, Found::Missing { .. })).collect();
    match missing.first() {
        None if lookups.is_empty() => {
            Err(format!("{op}: tuning read no store key (tuning off, or one candidate)").into())
        }
        None => Ok(spec),
        Some(l) => {
            let Found::Missing { others } = l.found else { unreachable!() };
            Err(format!(
                "{op}: no `{}` line for {} ({others} {} lines of the shape under other candidate lists: another \
                 ranking, salt or rule; a ranking moves with the costs, so --store must hold the card's probes)",
                l.kind.name(),
                l.key,
                l.kind.name()
            )
            .into())
        }
    }
}

/// A fixed candidate list (no cost model ranks it) on stdout and into `plan`.
fn listed<C: std::fmt::Debug>(plan: &mut String, ranked: &[C]) {
    for (i, c) in ranked.iter().enumerate() {
        let line = format!("#{i:<4} {c:?}");
        println!("{line}");
        plan.push_str(&line);
        plan.push('\n');
    }
}

/// The target of `--card`, else of `--arch` with the SMs and dies of its
/// first listed card unless given.
fn target(card: Option<&str>, arch: Option<&str>, sms: Option<usize>, xcds: Option<usize>) -> Result<Target> {
    if let Some(card) = card {
        return Target::card(card).ok_or_else(|| format!("{card}: not one of {:?}", CARDS.map(|c| c.0)).into());
    }
    let arch = arch.ok_or("--arch or --card")?;
    let arch = match AmdArch::parse(arch) {
        Some(a) => GpuArch::Amd(a),
        None => GpuArch::Cuda(arch.trim_end_matches('a').parse::<CudaArch>()?),
    };
    let listed = CARDS.iter().find(|c| c.1 == arch);
    let sms = sms.map(|s| s as u32).or(listed.map(|c| c.2)).ok_or("--sms: no listed card of the arch")?;
    let xcds = xcds.map(|x| x as u32).or(listed.map(|c| c.3)).unwrap_or(1);
    Ok(Target::offline(arch, sms, xcds))
}

/// The ranking's first `top` configs with their predicted ns on stdout, and
/// into `plan` every ranked config and every config of `lattice` the
/// ranking left out, with why.
fn report<C: Copy + PartialEq + std::fmt::Debug, R: std::fmt::Debug>(
    plan: &mut String,
    top: usize,
    ranked: &[C],
    lattice: &[C],
    admit: impl Fn(&C) -> std::result::Result<(f64, u32), R>,
) {
    let mut reasons: BTreeMap<String, usize> = BTreeMap::new();
    for (i, c) in ranked.iter().enumerate() {
        let line = match admit(c) {
            Ok((ns, regs)) => format!("#{i:<4} {:>12.0} ns {regs:>4} regs  {c:?}", ns),
            Err(why) => format!("#{i:<4} {why:?}  {c:?}"),
        };
        if i < top {
            println!("{line}");
        }
        plan.push_str(&line);
        plan.push('\n');
    }
    for c in lattice.iter().filter(|c| !ranked.contains(c)) {
        let why = admit(c).err().map_or("admitted, past the ranking".into(), |why| format!("{why:?}"));
        *reasons.entry(why.split([' ', '{']).next().unwrap_or_default().to_string()).or_default() += 1;
        plan.push_str(&format!("rejected {why}  {c:?}\n"));
    }
    println!(
        "{} ranked, {} rejected: {reasons:?} (every one in plan.txt)",
        ranked.len(),
        reasons.values().sum::<usize>()
    );
}

fn write(out: &Path, name: &str, bytes: &[u8]) -> Result<()> {
    std::fs::write(out.join(name), bytes)?;
    println!("wrote {}", out.join(name).display());
    Ok(())
}

/// A CUDA toolkit tool on `PATH` or under `/opt/cuda/bin`.
fn tool(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH").map(|p| std::env::split_paths(&p).collect::<Vec<_>>()).unwrap_or_default();
    path.into_iter().chain([PathBuf::from("/opt/cuda/bin")]).map(|d| d.join(name)).find(|p| p.is_file())
}

/// PTX, `ptxas -v` and SASS; the SASS (else the PTX) for the histogram.
fn cuda(out: &Path, ir: &str, arch: CudaArch) -> Result<Option<String>> {
    let ptx = match svod_runtime::cuda::compile_ir_to_ptx(ir, arch) {
        Ok(ptx) => ptx,
        Err(e) => return skip("PTX", &e.to_string()),
    };
    write(out, "kernel.ptx", &ptx)?;
    let Some(ptxas) = tool("ptxas") else { return skip("cubin, ptxas -v, SASS", "no ptxas") };
    let march = svod_codegen::llvm::nvptx::march(arch);
    let (ptx_path, cubin) = (out.join("kernel.ptx"), out.join("kernel.cubin"));
    let run = Command::new(ptxas)
        .arg(format!("-arch={march}"))
        .args(["-v", "-warn-spills", "-o"])
        .arg(&cubin)
        .arg(&ptx_path)
        .output()?;
    let resources = String::from_utf8_lossy(&run.stderr).into_owned();
    write(out, "ptxas.txt", resources.as_bytes())?;
    resources
        .lines()
        .filter(|l| l.contains("Used") || l.contains("spill") || l.contains("error"))
        .for_each(|l| println!("{l}"));
    let Some(nvdisasm) = tool("nvdisasm").filter(|_| run.status.success()) else {
        return Ok(Some(String::from_utf8_lossy(&ptx).into_owned()));
    };
    let sass = String::from_utf8_lossy(&Command::new(nvdisasm).arg("-c").arg(&cubin).output()?.stdout).into_owned();
    write(out, "kernel.sass", sass.as_bytes())?;
    Ok(Some(sass))
}

/// AMDGCN assembly and its resource comments.
fn amd(out: &Path, ir: &str, arch: AmdArch) -> Result<Option<String>> {
    let asm = match svod_runtime::amd::compile_ir_to_amd_asm(ir, arch) {
        Ok(asm) => String::from_utf8_lossy(&asm).into_owned(),
        Err(e) => return skip("AMDGCN assembly", &e.to_string()),
    };
    write(out, "kernel.s", asm.as_bytes())?;
    let keys = ["NumVgprs:", "NumAgprs:", "TotalNumVgprs:", "NumSgprs:", "ScratchSize:", "Occupancy:", "LDSByteSize:"];
    let summary: Vec<&str> =
        asm.lines().map(str::trim).filter(|l| l.starts_with(';') && keys.iter().any(|k| l.contains(k))).collect();
    println!("{}", summary.join("  "));
    Ok(Some(asm))
}

fn skip(what: &str, why: &str) -> Result<Option<String>> {
    println!("skipped {what}: {why}");
    Ok(None)
}

/// The main loop of `asm` ([`isa::main_loop`]) on stdout, `what` naming its dialect.
fn histogram(asm: &str, what: &str) -> Option<isa::MainLoop> {
    let Some(main) = isa::main_loop(asm) else {
        println!("{what}: no loop issues a matrix instruction (the reduction unrolled whole?)");
        return None;
    };
    println!("{what} main loop ({} instructions):", main.instructions());
    for (op, n) in &main.ops {
        println!("  {n:>5} {op}");
    }
    Some(main)
}

/// The reference kernel's assembly text: a cubin through `nvdisasm -c`, an
/// AMD code object through `llvm-objdump -d --symbolize-operands` for the
/// target's `mcpu`, any other file read as text (SASS, PTX, AMDGCN `.s`);
/// cut to the function `fun`, which a text of several functions needs.
fn reference(path: &Path, fun: Option<&str>, target: &Target, out: &Path) -> Result<String> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or_default();
    let text = match ext {
        "cubin" => {
            let nvdisasm = tool("nvdisasm").ok_or("a cubin is read by nvdisasm, which is not installed")?;
            run(Command::new(nvdisasm).arg("-c").arg(path))?
        }
        "co" | "hsaco" | "o" | "elf" => {
            let GpuArch::Amd(arch) = target.arch else {
                return Err(format!("{}: an AMD code object compares with an AMD target", path.display()).into());
            };
            let objdump = rocm_tool("llvm-objdump").ok_or("an AMD code object is read by llvm-objdump")?;
            let mcpu = format!("--mcpu={}", arch.mcpu());
            run(Command::new(objdump).args(["-d", "--symbolize-operands", &mcpu]).arg(path))?
        }
        _ => std::fs::read_to_string(path)?,
    };
    let names = isa::functions(&text);
    let text = match fun {
        Some(name) => isa::function(&text, name)
            .ok_or_else(|| format!("{}: no function {name}; it holds {names:?}", path.display()))?
            .to_string(),
        None if names.len() > 1 => {
            let shown: Vec<&str> = names.iter().take(20).copied().collect();
            return Err(format!("{}: {} functions, pick one with --fun: {shown:?}", path.display(), names.len()).into());
        }
        None => text,
    };
    write(out, &format!("reference.{}", if ext == "co" || ext == "hsaco" { "s" } else { "txt" }), text.as_bytes())?;
    Ok(text)
}

/// A tool's stdout, an error with its stderr when it fails.
fn run(command: &mut Command) -> Result<String> {
    let done = command.output()?;
    if !done.status.success() {
        return Err(format!("{command:?}: {}", String::from_utf8_lossy(&done.stderr)).into());
    }
    Ok(String::from_utf8_lossy(&done.stdout).into_owned())
}

/// A ROCm LLVM tool on `PATH` or under `/opt/rocm/llvm/bin`.
fn rocm_tool(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH").map(|p| std::env::split_paths(&p).collect::<Vec<_>>()).unwrap_or_default();
    [PathBuf::from("/opt/rocm/llvm/bin")].into_iter().chain(path).map(|d| d.join(name)).find(|p| p.is_file())
}
