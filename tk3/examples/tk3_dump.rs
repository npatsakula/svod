//! What tk3 would run on a card this machine has no device for: the planner's
//! ranking (predicted ns, and why each config of the lattice was left out),
//! the chosen config's LLVM text and, where the tools are installed, its
//! PTX, `ptxas -v` resources and SASS (CUDA) or AMDGCN assembly and its
//! resource comments (AMD), with an opcode histogram of the innermost loop
//! that issues matrix instructions: the main loop.
//!
//! ```text
//! cargo run -p svod-tk3 --release --example tk3_dump -- --arch sm_90a --sms 114 \
//!     --op gemm --shape 4096x4096x4096 [--dtype bf16] [--cfg 0] [--store DIR] [--out DIR]
//! cargo run -p svod-tk3 --release --example tk3_dump -- --card mi300x --op attention --shape 8x16x1024x128 --causal
//! ```
//!
//! `--arch` takes the SMs and dies of the first card of `svod_tk3::atoms::CARDS`
//! of the architecture unless `--sms`/`--xcds` say otherwise; `--store` is a
//! tune store directory copied from the card (its probe lines price the
//! ranking), else the guessed costs do. `RUST_LOG=svod_tk3::plan=debug` logs
//! the planner's decisions.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use svod_dtype::{AmdArch, CudaArch, GpuArch};
use svod_tk3::atoms::{CARDS, Target};
use svod_tk3::build::{BF16, F16};
use svod_tk3::ir::Program;
use svod_tk3::kernels::Batch;
use svod_tk3::kernels::attention::{AttnMask, AttnSpec, attention};
use svod_tk3::kernels::gemm::{Epilogue, GemmSpec, gemm};
use svod_tk3::lower::Lowering;
use svod_tk3::ops::config::{Attention, Planner, Problem, gemm_lattice};
use svod_tk3::probe::{self, Measured};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

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
    let planner = Planner::measured(target.clone(), measured);
    let dims: Vec<usize> =
        opt("--shape").ok_or("--shape")?.split('x').map(str::parse).collect::<std::result::Result<_, _>>()?;
    let bf16 = opt("--dtype").unwrap_or("bf16") == "bf16";
    let out = PathBuf::from(opt("--out").unwrap_or("tk3_dump"));
    std::fs::create_dir_all(&out)?;
    let (top, pick) = (num("--top")?.unwrap_or(16), num("--cfg")?.unwrap_or(0));

    let mut plan = String::new();
    let (prog, lowering): (Program, Lowering) = match (opt("--op").ok_or("--op")?, &dims[..]) {
        ("gemm", &[m, n, k]) => {
            let p = Problem { batches: 1, m, n, k, halves: 1 };
            let ranked = planner.gemm_ranked(1, m, n, k, false);
            let lattice = gemm_lattice(&target);
            report(&mut plan, top, &ranked, &lattice, |c| p.admit(&planner, c));
            let cfg = *ranked.get(pick).ok_or("no such ranked config")?;
            println!("emitting #{pick}: {cfg:?}");
            let spec = GemmSpec { m, n, k, batch: Batch::Static(1), epilogue: Epilogue::DEFAULT, cfg };
            (if bf16 { gemm::<BF16>(&spec) } else { gemm::<F16>(&spec) }, cfg.lowering(&planner.device))
        }
        ("attention", &[b, h, t, d]) => {
            let a = Attention { tiles: b * h, t, tk: t, d, causal: flag("--causal"), bias: false };
            let ranked = planner.attention_ranked(b * h, t, t, d, a.causal, false);
            report(&mut plan, top, &ranked, &a.lattice(&target), |c| a.admit(&planner, c));
            let cfg = *ranked.get(pick).ok_or("no such ranked config")?;
            println!("emitting #{pick}: {cfg:?}");
            let mask = AttnMask { causal: a.causal, ..AttnMask::default() };
            let scale = 1.0 / (d as f32).sqrt();
            let spec =
                AttnSpec { batch: Batch::Static(b), t, tk: t, heads: h, kv_heads: h, d, mask, cache: None, scale, cfg };
            (if bf16 { attention::<BF16>(&spec) } else { attention::<F16>(&spec) }, cfg.lowering(&planner.device))
        }
        (op, dims) => return Err(format!("{op} over {dims:?}: gemm takes MxNxK, attention BxHxTxD").into()),
    };
    std::fs::write(out.join("plan.txt"), plan)?;

    let ir = svod_tk3::lower::standalone(prog, &lowering)?.text(&target)?;
    write(&out, "kernel.ll", ir.as_bytes())?;
    let (asm, name) = match target.arch {
        GpuArch::Cuda(arch) => (cuda(&out, &ir, arch)?, "SASS"),
        GpuArch::Amd(arch) => (amd(&out, &ir, arch)?, "AMDGCN"),
        GpuArch::Metal(_) => return Err("Metal renders no LLVM text".into()),
    };
    if let Some((asm, what)) = asm.map(|a| (a, name)) {
        histogram(&asm, what);
    }
    Ok(())
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

/// One assembly line: a label, or an instruction's opcode with its branch
/// target, from SASS (`nvdisasm`), PTX or AMDGCN text alike.
enum Line {
    Label(String),
    Op(String, Option<String>),
}

fn lines(asm: &str) -> Vec<Line> {
    asm.lines()
        .filter_map(|l| {
            // SASS addresses and encodings sit in `/* */`, PTX comments follow
            // `//`, AMDGCN's `;`, and SASS and PTX end a statement with `;`.
            let mut s = l.split("//").next()?.to_string();
            while let (Some(a), Some(b)) = (s.find("/*"), s.find("*/")) {
                s.replace_range(a..b + 2, "");
            }
            let t = s.split(';').next()?.trim();
            if t.ends_with(':') && !t.contains(char::is_whitespace) {
                return Some(Line::Label(t.trim_end_matches(':').to_string()));
            }
            if t.is_empty() || t.starts_with(['.', '{', '}', '-']) {
                return None;
            }
            let words: Vec<&str> = t.split_whitespace().filter(|w| !w.starts_with('@')).collect();
            let op = words.first()?;
            let branch = ["bra", "BRA", "s_branch", "s_cbranch"].iter().any(|b| op.starts_with(b));
            let target = words.last().filter(|_| branch).map(|w| w.trim_matches(|c| "`(),".contains(c)).to_string());
            Some(Line::Op(op.to_string(), target))
        })
        .collect()
}

/// The opcodes of the innermost loop (a backward branch's span) that issues
/// a matrix instruction, by count: the main loop's step.
fn histogram(asm: &str, what: &str) {
    let lines = lines(asm);
    let at = |name: &str| lines.iter().position(|l| matches!(l, Line::Label(n) if n == name));
    let is_mma = |op: &str| op.contains("MMA") || op.contains("mma") || op.contains("mfma") || op.contains("wmma");
    let loops = lines.iter().enumerate().filter_map(|(j, l)| match l {
        Line::Op(_, Some(target)) => at(target).filter(|&i| i < j).map(|i| (i, j)),
        _ => None,
    });
    let body = |(i, j): (usize, usize)| {
        lines[i..=j].iter().filter_map(|l| if let Line::Op(op, _) = l { Some(op.as_str()) } else { None })
    };
    let Some((i, j)) = loops.filter(|&span| body(span).any(is_mma)).min_by_key(|(i, j)| j - i) else {
        println!("{what}: no loop issues a matrix instruction (the reduction unrolled whole?)");
        return;
    };
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    body((i, j)).for_each(|op| *counts.entry(op).or_default() += 1);
    let mut counts: Vec<(&str, usize)> = counts.into_iter().collect();
    counts.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    println!("{what} main loop ({} instructions):", counts.iter().map(|c| c.1).sum::<usize>());
    for (op, n) in counts {
        println!("  {n:>5} {op}");
    }
}
