//! An agent-facing reading of an AMD thread trace ([`SqttTrace`]): the device's decoder, wave model
//! and stitcher ([`svod_device::sqtt`]) over the kernel's `llvm-objdump` disassembly, summarized as a
//! fixed-format text ([`Analysis`]'s `Display`) or JSON. A [`Capture`] saves the raw streams with the
//! code object and the disassembly, so a trace is re-analyzed offline with no GPU.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde::{Deserialize, Serialize};
use snafu::{OptionExt, ResultExt, Snafu};
use svod_device::profile::{CodeImage, SeTrace, SqttTrace};
use svod_device::sqtt::report::{Report, Stats};
use svod_device::sqtt::stitch::{Stitcher, branch, objdump_lines};
use svod_device::sqtt::{Layout, decode, wave};
use svod_dtype::AmdArch;

#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum Error {
    #[snafu(display("llvm-objdump not found beside clang or on PATH: {source}"))]
    Objdump { source: Box<crate::Error> },
    #[snafu(display("llvm-objdump: {source}"))]
    Spawn { source: std::io::Error },
    #[snafu(display("llvm-objdump failed: {stderr}"))]
    Disassemble { stderr: String },
    #[snafu(display("{}: {source}", path.display()))]
    Io { source: std::io::Error, path: PathBuf },
    #[snafu(display("{}: {source}", path.display()))]
    Json { source: serde_json::Error, path: PathBuf },
    #[snafu(display("the trace carries no code image: set SVOD_SQTT=1 before the device opens"))]
    NoCode,
    #[snafu(display("no instruction-traced shader engine has an RDNA3/RDNA4 trace header"))]
    NoItrace,
    #[snafu(display("{arch}: not an AMD arch"))]
    Arch { arch: String },
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// `llvm-objdump -d` of a code object, the tool found beside the clang svod compiles with.
pub fn disassemble(elf: &[u8]) -> Result<String> {
    let tool = crate::clang::ClangToolchain::discover(None)
        .and_then(|clang| clang.sibling("llvm-objdump"))
        .map_err(Box::new)
        .context(ObjdumpSnafu)?;
    let mut child = Command::new(tool)
        .args(["-d", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context(SpawnSnafu)?;
    let mut stdin = child.stdin.take().expect("piped stdin");
    let output = std::thread::scope(|s| {
        // A pipe objdump closed early is its failure to report, not the writer's.
        s.spawn(move || drop(stdin.write_all(elf)));
        child.wait_with_output()
    })
    .context(SpawnSnafu)?;
    if !output.status.success() {
        return DisassembleSnafu { stderr: String::from_utf8_lossy(&output.stderr).into_owned() }.fail();
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// A kernel's trace with the disassembly its PCs map through: what [`Self::save`] writes and
/// [`Self::load`] reads back.
#[derive(Debug, Clone)]
pub struct Capture {
    pub trace: SqttTrace,
    pub disassembly: String,
    /// The kernel's untraced device time.
    pub kernel_ns: Option<u64>,
    /// The traced dispatch's own device time: the same run and power state as the trace's cycles, so
    /// the clock estimate prefers it.
    pub traced_ns: Option<u64>,
    /// Free-form context (`cfg`, shape) carried into the report header.
    pub labels: BTreeMap<String, String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Meta {
    arch: String,
    load_base: u64,
    entry_pc: u64,
    kernel_ns: Option<u64>,
    #[serde(default)]
    traced_ns: Option<u64>,
    ses: Vec<SeMeta>,
    #[serde(default)]
    labels: BTreeMap<String, String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SeMeta {
    se: u32,
    itrace: bool,
    truncated: bool,
}

fn io<T>(path: &Path, result: std::io::Result<T>) -> Result<T> {
    result.context(IoSnafu { path })
}

impl Capture {
    /// Disassembles the trace's code image.
    pub fn new(trace: SqttTrace, kernel_ns: Option<u64>) -> Result<Self> {
        let disassembly = disassemble(&trace.code.as_ref().context(NoCodeSnafu)?.elf)?;
        Ok(Self { trace, disassembly, kernel_ns, traced_ns: None, labels: BTreeMap::new() })
    }

    /// `se<N>.att` (raw streams), `kernel.co`, `kernel.s` and `meta.json` under `dir`.
    pub fn save(&self, dir: &Path) -> Result<()> {
        let code = self.trace.code.as_ref().context(NoCodeSnafu)?;
        io(dir, std::fs::create_dir_all(dir))?;
        let write = |name: &str, bytes: &[u8]| io(&dir.join(name), std::fs::write(dir.join(name), bytes));
        for se in &self.trace.per_se {
            write(&format!("se{}.att", se.se), &se.bytes)?;
        }
        write("kernel.co", &code.elf)?;
        write("kernel.s", self.disassembly.as_bytes())?;
        let meta = Meta {
            arch: self.trace.arch.to_string(),
            load_base: code.load_base,
            entry_pc: code.entry_pc,
            kernel_ns: self.kernel_ns,
            traced_ns: self.traced_ns,
            ses: self
                .trace
                .per_se
                .iter()
                .map(|s| SeMeta { se: s.se, itrace: s.itrace, truncated: s.truncated })
                .collect(),
            labels: self.labels.clone(),
        };
        let path = dir.join("meta.json");
        write("meta.json", serde_json::to_string_pretty(&meta).context(JsonSnafu { path })?.as_bytes())
    }

    /// What [`Self::save`] wrote; disassembles `kernel.co` only where `kernel.s` is missing.
    pub fn load(dir: &Path) -> Result<Self> {
        let read = |name: &str| io(&dir.join(name), std::fs::read(dir.join(name)));
        let path = dir.join("meta.json");
        let meta: Meta = serde_json::from_slice(&read("meta.json")?).context(JsonSnafu { path })?;
        let arch = AmdArch::parse(&meta.arch).context(ArchSnafu { arch: &meta.arch })?;
        let per_se = meta
            .ses
            .iter()
            .map(|s| {
                Ok(SeTrace {
                    se: s.se,
                    itrace: s.itrace,
                    bytes: read(&format!("se{}.att", s.se))?,
                    truncated: s.truncated,
                })
            })
            .collect::<Result<_>>()?;
        let elf = if dir.join("kernel.co").exists() { read("kernel.co")? } else { vec![] };
        let disassembly = match dir.join("kernel.s").exists() {
            true => String::from_utf8_lossy(&read("kernel.s")?).into_owned(),
            false => disassemble(&elf)?,
        };
        let code = Some(CodeImage { elf, load_base: meta.load_base, entry_pc: meta.entry_pc });
        Ok(Self {
            trace: SqttTrace { arch, per_se, code },
            disassembly,
            kernel_ns: meta.kernel_ns,
            traced_ns: meta.traced_ns,
            labels: meta.labels,
        })
    }

    pub fn analyze(&self) -> Result<Analysis> {
        Analysis::of(self)
    }
}

/// What an instruction's cycles count as, by its mnemonic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Class {
    /// `v_wmma`, `v_swmmac`, `v_mfma`.
    Matrix,
    Valu,
    Salu,
    /// `ds_*`: shared-memory (LDS) issue.
    Lds,
    VmemLoad,
    /// Global stores and atomics.
    VmemStore,
    /// Scalar memory: `s_load`, `s_buffer_load`, ...
    Smem,
    Branch,
    /// `s_waitcnt vmcnt`/`vscnt`, `s_wait_loadcnt`/`storecnt`: waits on vector memory.
    WaitVmem,
    /// `s_waitcnt lgkmcnt`, `s_wait_dscnt`/`kmcnt`: waits on LDS and scalar memory.
    WaitLgkm,
    /// Other immediates: `s_nop`, `s_delay_alu`, `s_clause`, `s_wait_idle`, ...
    Immediate,
    Barrier,
    /// `s_sendmsg`, `s_endpgm`, `s_trap`, `s_setprio`, ...
    Message,
    Other,
}

impl Class {
    pub fn of(asm: &str) -> Self {
        let text = asm.split("//").next().unwrap_or_default().trim();
        let m = text.split_whitespace().next().unwrap_or_default();
        let any = |prefixes: &[&str]| prefixes.iter().any(|p| m.starts_with(p));
        if any(&["v_wmma", "v_swmmac", "v_mfma", "v_smfmac"]) {
            Self::Matrix
        } else if m.starts_with("v_") {
            Self::Valu
        } else if any(&["ds_", "lds_"]) {
            Self::Lds
        } else if any(&["global_", "buffer_", "flat_", "scratch_", "tbuffer_", "image_"]) {
            if m.contains("load") || m.contains("sample") || m.contains("gather") {
                Self::VmemLoad
            } else {
                Self::VmemStore
            }
        } else if any(&[
            "s_load", "s_buffer_", "s_store", "s_dcache", "s_scratch", "s_atomic", "s_prefetch", "s_memtime",
        ]) {
            Self::Smem
        } else if any(&["s_branch", "s_cbranch", "s_setpc", "s_swappc", "s_call", "s_rfe"]) {
            Self::Branch
        } else if any(&["s_waitcnt_depctr", "s_wait_alu", "s_wait_idle", "s_wait_expcnt", "s_wait_event"]) {
            Self::Immediate
        } else if any(&["s_waitcnt", "s_wait_"]) {
            let vmem = ["vmcnt", "vscnt", "loadcnt", "storecnt", "samplecnt", "bvhcnt"];
            if vmem.iter().any(|c| text.contains(c)) {
                Self::WaitVmem
            } else if ["lgkmcnt", "dscnt", "kmcnt"].iter().any(|c| text.contains(c)) {
                Self::WaitLgkm
            } else {
                Self::Immediate
            }
        } else if m.starts_with("s_barrier") {
            Self::Barrier
        } else if any(&["s_sendmsg", "s_endpgm", "s_trap", "s_setprio", "s_sethalt", "s_ttrace", "s_icache"]) {
            Self::Message
        } else if any(&["s_nop", "s_delay", "s_clause", "s_sleep", "s_set_inst", "s_inst_prefetch", "s_version"]) {
            Self::Immediate
        } else if m.starts_with("s_") {
            Self::Salu
        } else {
            Self::Other
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Matrix => "matrix",
            Self::Valu => "valu",
            Self::Salu => "salu",
            Self::Lds => "lds",
            Self::VmemLoad => "vmem_load",
            Self::VmemStore => "vmem_store",
            Self::Smem => "smem",
            Self::Branch => "branch",
            Self::WaitVmem => "wait_vmem",
            Self::WaitLgkm => "wait_lgkm",
            Self::Immediate => "immediate",
            Self::Barrier => "barrier",
            Self::Message => "message",
            Self::Other => "other",
        }
    }
}

/// `n` samples' order statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Spread {
    pub n: usize,
    pub min: i64,
    pub median: i64,
    pub p90: i64,
    pub max: i64,
}

impl Spread {
    pub fn of(mut samples: Vec<i64>) -> Option<Self> {
        samples.sort_unstable();
        let n = samples.len();
        let at = |q: usize| samples[(n * q / 100).min(n - 1)];
        (n > 0).then(|| Self { n, min: samples[0], median: at(50), p90: at(90), max: samples[n - 1] })
    }
}

/// One shader engine's stream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Se {
    pub se: u32,
    pub itrace: bool,
    pub bytes: usize,
    pub truncated: bool,
    /// Waves started on any SIMD of the SE.
    pub waves: usize,
    /// Waves resident on the SE at once, at most.
    pub occupancy_peak: usize,
    /// Cycles from the first wave's start to the last's end.
    pub span: i64,
}

/// The SIMD whose instructions an SE traced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Traced {
    pub sa: u8,
    pub wgp: u8,
    pub simd: u8,
}

/// One instruction's cycles summed over the traced waves; `pc` is its ELF address.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Row {
    pub pc: u64,
    pub asm: String,
    pub class: Class,
    pub hits: u64,
    pub latency: i64,
    pub stall: i64,
    pub idle: i64,
}

/// A class's cycles on the traced SIMD; `latency` includes `stall`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClassTime {
    pub class: Class,
    pub hits: u64,
    pub latency: i64,
    pub stall: i64,
}

/// The innermost loop (a backward branch's span) issuing a matrix instruction, else the hottest one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MainLoop {
    /// ELF addresses of the loop head and its back edge.
    pub head: u64,
    pub back_edge: u64,
    pub matrix: bool,
    /// Back-edge executions of each wave that reached it.
    pub iterations: Option<Spread>,
    /// Cycles between a wave's consecutive back edges.
    pub cycles: Option<Spread>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Analysis {
    pub kernel: String,
    pub arch: String,
    pub labels: BTreeMap<String, String>,
    pub kernel_ns: Option<u64>,
    pub traced_ns: Option<u64>,
    pub ses: Vec<Se>,
    /// The traced SIMD of the first instruction-traced SE.
    pub traced: Traced,
    /// Waves on the traced SIMDs, and those whose end the trace holds.
    pub waves: usize,
    pub complete: usize,
    /// Cycles from start to end of the traced complete waves.
    pub lifetime: Option<Spread>,
    /// Shader clock: the longest instruction-traced SE's wave span over `traced_ns`, else `kernel_ns`.
    pub sclk_mhz: Option<u64>,
    /// Instruction records of the traced waves, and those a PC was attributed to.
    pub records: usize,
    pub stitched: usize,
    pub classes: Vec<ClassTime>,
    pub main_loop: Option<MainLoop>,
    /// Every executed instruction, by PC.
    pub pcs: Vec<Row>,
}

/// The `<name>` of an `llvm-objdump` label line, `0000000000001700 <name>:`, with its address.
fn label(line: &str) -> Option<(u64, &str)> {
    let (at, rest) = line.trim().split_once(' ')?;
    let name = rest.strip_prefix('<')?.strip_suffix(">:")?;
    Some((u64::from_str_radix(at, 16).ok()?, name))
}

fn asm(line: &str) -> String {
    line.split("//").next().unwrap_or_default().split_whitespace().collect::<Vec<_>>().join(" ")
}

impl Analysis {
    /// Disassembles the trace's code object and analyzes it.
    pub fn from_trace(trace: &SqttTrace, kernel_ns: Option<u64>) -> Result<Self> {
        Capture::new(trace.clone(), kernel_ns)?.analyze()
    }

    /// The analysis of a [`Capture::save`]d directory.
    pub fn load(dir: &Path) -> Result<Self> {
        Capture::load(dir)?.analyze()
    }

    fn of(capture: &Capture) -> Result<Self> {
        let code = capture.trace.code.as_ref().context(NoCodeSnafu)?;
        let base = code.load_base;
        let lines: Vec<(u64, &str)> =
            objdump_lines(&capture.disassembly).into_iter().map(|(pc, line)| (pc + base, line)).collect();
        let (mut ses, mut traced, mut waves, mut pcs) = (vec![], None, vec![], BTreeMap::<u64, Stats>::new());
        for se in &capture.trace.per_se {
            let mut trace = wave::analyze(decode(&se.bytes));
            let mut live = 0usize;
            let mut events: Vec<_> = trace.occupancy.iter().map(|o| (o.time, !o.start)).collect();
            events.sort_unstable();
            let peak = events.iter().fold(0, |peak, &(_, end)| {
                live = if end { live.saturating_sub(1) } else { live + 1 };
                peak.max(live)
            });
            let span = events.first().zip(events.last()).map_or(0, |(a, b)| b.0 - a.0);
            let starts = trace.occupancy.iter().filter(|o| o.start).count();
            ses.push(Se {
                se: se.se,
                itrace: se.itrace,
                bytes: se.bytes.len(),
                truncated: se.truncated,
                waves: starts,
                occupancy_peak: peak,
                span,
            });
            let layout = match trace.header.version {
                3 => Layout::Rdna3,
                4 => Layout::Rdna4,
                _ => continue,
            };
            if !se.itrace {
                continue;
            }
            let h = trace.header;
            traced.get_or_insert(Traced { sa: h.dsa, wgp: h.dwgp, simd: h.dsimd });
            let mut stitcher = Stitcher::new(&lines, layout);
            for wave in &mut trace.waves {
                stitcher.stitch(wave);
            }
            for (pc, s) in Report::new(&trace.waves).pcs {
                let at = pcs.entry(pc).or_default();
                (at.hits, at.latency, at.stall, at.idle) =
                    (at.hits + s.hits, at.latency + s.latency, at.stall + s.stall, at.idle + s.idle);
            }
            waves.append(&mut trace.waves);
        }
        let traced = traced.context(NoItraceSnafu)?;
        let text: BTreeMap<u64, &str> = lines.iter().copied().collect();
        let rows: Vec<Row> = pcs
            .iter()
            .map(|(&pc, s)| {
                let line = text.get(&pc).copied().unwrap_or_default();
                let (asm, class) = (asm(line), Class::of(line));
                Row { pc: pc - base, asm, class, hits: s.hits, latency: s.latency, stall: s.stall, idle: s.idle }
            })
            .collect();
        let mut classes = BTreeMap::<Class, ClassTime>::new();
        for r in &rows {
            let c = classes.entry(r.class).or_insert(ClassTime { class: r.class, hits: 0, latency: 0, stall: 0 });
            (c.hits, c.latency, c.stall) = (c.hits + r.hits, c.latency + r.latency, c.stall + r.stall);
        }
        let first_pc = waves.iter().flat_map(|w| &w.instructions).find(|i| i.pc != 0).map(|i| i.pc - base);
        let labels: Vec<(u64, &str)> = capture.disassembly.lines().filter_map(label).collect();
        let entry = code.entry_pc.checked_sub(base);
        let kernel = labels
            .iter()
            .find(|l| Some(l.0) == entry)
            .or_else(|| labels.iter().rev().find(|l| first_pc.is_some_and(|pc| l.0 <= pc)))
            .or(labels.first())
            .map_or("?", |l| l.1);
        let span = ses.iter().filter(|s| s.itrace).map(|s| s.span).max().unwrap_or(0);
        let lifetime = waves.iter().filter(|w| w.complete).map(|w| w.end - w.begin).collect();
        Ok(Self {
            kernel: kernel.to_string(),
            arch: capture.trace.arch.to_string(),
            labels: capture.labels.clone(),
            kernel_ns: capture.kernel_ns,
            traced_ns: capture.traced_ns,
            traced,
            waves: waves.len(),
            complete: waves.iter().filter(|w| w.complete).count(),
            lifetime: Spread::of(lifetime),
            sclk_mhz: capture
                .traced_ns
                .or(capture.kernel_ns)
                .filter(|&ns| ns > 0 && span > 0)
                .map(|ns| span as u64 * 1000 / ns),
            records: waves.iter().map(|w| w.instructions.len()).sum(),
            stitched: waves.iter().flat_map(|w| &w.instructions).filter(|i| i.pc != 0).count(),
            classes: classes.into_values().collect(),
            main_loop: main_loop(&lines, &pcs, &waves, base),
            ses,
            pcs: rows,
        })
    }

    /// The rows of [`Self::main_loop`]'s body.
    pub fn loop_rows(&self) -> impl Iterator<Item = &Row> {
        let range = self.main_loop.as_ref().map(|l| l.head..=l.back_edge);
        self.pcs.iter().filter(move |r| range.as_ref().is_some_and(|range| range.contains(&r.pc)))
    }

    /// The cycles of the traced SIMD's instructions in `classes`, and their stall.
    pub fn cycles(&self, classes: &[Class]) -> (i64, i64) {
        let of = self.classes.iter().filter(|c| classes.contains(&c.class));
        of.fold((0, 0), |(l, s), c| (l + c.latency, s + c.stall))
    }

    /// The analysis as one line of JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("an analysis serializes")
    }

    /// The fixed-format report with `top` rows a table.
    pub fn text(&self, top: usize) -> String {
        use std::fmt::Write;
        let mut o = String::new();
        let shown = self.labels.iter().filter(|(k, _)| !k.starts_with("model."));
        let labels: String = shown.map(|(k, v)| format!(" {k}={v}")).collect();
        let _ = writeln!(o, "== sqtt {} {}{labels}", self.kernel, self.arch);
        for se in self.ses.iter().filter(|s| s.truncated) {
            let _ = writeln!(o, "!! TRUNCATED se{}: its buffer filled and the tail is lost; totals undercount", se.se);
        }
        if self.stitched < self.records {
            let _ =
                writeln!(o, "!! UNSTITCHED {} of {} records have no PC", self.records - self.stitched, self.records);
        }
        let ns = |ns: Option<u64>| ns.map_or("-".into(), |ns| ns.to_string());
        let span = self.ses.iter().filter(|s| s.itrace).map(|s| s.span).max().unwrap_or(0);
        let sclk = self.sclk_mhz.map_or("-".into(), |mhz| format!("~{mhz}"));
        let (untraced, traced) = (ns(self.kernel_ns), ns(self.traced_ns));
        let _ = writeln!(
            o,
            "kernel     {untraced} ns untraced, {traced} ns traced, wave span {span} cycles, sclk {sclk} MHz"
        );
        let itrace: Vec<String> = self.ses.iter().filter(|s| s.itrace).map(|s| format!("se{}", s.se)).collect();
        let t = self.traced;
        let _ = writeln!(
            o,
            "traced     sa{} wgp{} simd{} on {} of {} SEs",
            t.sa,
            t.wgp,
            t.simd,
            itrace.join(","),
            self.ses.len()
        );
        let lifetime = self.lifetime.map_or("-".into(), |l| format!("min {} med {} max {}", l.min, l.median, l.max));
        let starts: usize = self.ses.iter().filter(|s| s.itrace).map(|s| s.waves).sum();
        let peak = self.ses.iter().map(|s| s.occupancy_peak).max().unwrap_or(0);
        let _ = writeln!(
            o,
            "waves      {} traced ({} complete) of {starts} on the traced SEs, occupancy peak {peak}; lifetime {lifetime} cycles",
            self.waves, self.complete
        );
        let _ = writeln!(o, "records    {} stitched of {}", self.stitched, self.records);
        let total: i64 = self.classes.iter().map(|c| c.latency).sum();
        let _ = writeln!(o, "== time by class (traced SIMD, cycles summed over waves)");
        let _ =
            writeln!(o, "{:<11} {:>8} {:>11} {:>6} {:>11} {:>11}", "class", "hits", "cycles", "share", "stall", "exec");
        let share = |c: i64| if total > 0 { 100.0 * c as f64 / total as f64 } else { 0.0 };
        for c in &self.classes {
            let (l, s) = (c.latency, c.stall);
            let _ =
                writeln!(o, "{:<11} {:>8} {l:>11} {:>5.1}% {s:>11} {:>11}", c.class.name(), c.hits, share(l), l - s);
        }
        let (hits, stall): (u64, i64) = self.classes.iter().fold((0, 0), |(h, s), c| (h + c.hits, s + c.stall));
        let _ = writeln!(o, "{:<11} {hits:>8} {total:>11} {:>5.1}% {stall:>11} {:>11}", "total", 100.0, total - stall);
        let table = |o: &mut String, rows: Vec<&Row>| {
            let _ = writeln!(o, "{:>8} {:>7} {:>9} {:>9} {:>9}  asm", "pc", "hits", "cycles", "lat/hit", "stall/hit");
            for r in rows {
                let per = |c: i64| c as f64 / r.hits.max(1) as f64;
                let (l, s) = (per(r.latency), per(r.stall));
                let _ = writeln!(o, "{:>#8x} {:>7} {:>9} {l:>9.1} {s:>9.1}  {}", r.pc, r.hits, r.latency, r.asm);
            }
        };
        match &self.main_loop {
            None => {
                let _ = writeln!(o, "== main loop: none (no executed backward branch)");
            }
            Some(l) => {
                let kind = if l.matrix { "matrix" } else { "hottest, no matrix op" };
                let _ = writeln!(o, "== main loop {:#x}..={:#x} ({kind})", l.head, l.back_edge);
                let spread = |s: Option<Spread>, p: bool| match s {
                    None => "-".into(),
                    Some(s) if p => format!("min {} med {} p90 {} max {} (n {})", s.min, s.median, s.p90, s.max, s.n),
                    Some(s) => format!("min {} med {} max {} (waves {})", s.min, s.median, s.max, s.n),
                };
                let _ = writeln!(o, "iterations/wave  {}", spread(l.iterations, false));
                let _ = writeln!(o, "cycles/iteration {}", spread(l.cycles, true));
                let mut rows: Vec<&Row> = self.loop_rows().collect();
                rows.sort_by_key(|r| (std::cmp::Reverse(r.stall), r.pc));
                let _ = writeln!(o, "== main loop: top {top} by stall");
                table(&mut o, rows.into_iter().take(top).collect());
            }
        }
        let mut rows: Vec<&Row> = self.pcs.iter().collect();
        rows.sort_by_key(|r| (std::cmp::Reverse(r.latency), r.pc));
        let _ = writeln!(o, "== hottest {top} by cycles");
        table(&mut o, rows.into_iter().take(top).collect());
        o
    }

    /// One paragraph: where the traced SIMD's time went and the main loop's pace.
    pub fn summary(&self) -> String {
        let total: i64 = self.classes.iter().map(|c| c.latency).sum::<i64>().max(1);
        let mut classes = self.classes.clone();
        classes.sort_by_key(|c| std::cmp::Reverse(c.latency));
        let top: Vec<String> = classes
            .iter()
            .take(4)
            .map(|c| format!("{} {:.0}%", c.class.name(), 100.0 * c.latency as f64 / total as f64))
            .collect();
        let sclk = self.sclk_mhz.map_or(String::new(), |mhz| format!(", sclk ~{mhz} MHz"));
        let pace = self
            .main_loop
            .as_ref()
            .and_then(|l| Some((l.iterations?, l.cycles?)))
            .map_or("no main loop".into(), |(i, c)| {
                format!("main loop {} iterations/wave at {} cycles (median)", i.median, c.median)
            });
        let warn = if self.ses.iter().any(|s| s.truncated) { " TRUNCATED." } else { "" };
        format!(
            "{} {}: {} traced waves{sclk}; time {}; {pace}.{warn}",
            self.kernel,
            self.arch,
            self.waves,
            top.join(", ")
        )
    }
}

impl std::fmt::Display for Analysis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.text(10))
    }
}

/// The innermost executed backward branch whose span issues a matrix instruction, else the most
/// executed one; its waves' trips and the cycles between their back edges.
fn main_loop(lines: &[(u64, &str)], pcs: &BTreeMap<u64, Stats>, waves: &[wave::Wave], base: u64) -> Option<MainLoop> {
    let edges = lines
        .iter()
        .filter_map(|&(pc, line)| Some((branch(pc, line).filter(|&target| target <= pc)?, pc)))
        .filter(|(_, pc)| pcs.contains_key(pc));
    let matrix = |&(head, edge): &(u64, u64)| {
        lines.iter().any(|&(pc, line)| (head..=edge).contains(&pc) && Class::of(line) == Class::Matrix)
    };
    let edges: Vec<(u64, u64)> = edges.collect();
    let (head, edge) = match edges.iter().filter(|e| matrix(e)).min_by_key(|(h, e)| e - h) {
        Some(&e) => e,
        None => *edges.iter().max_by_key(|(_, pc)| pcs[pc].hits)?,
    };
    let (mut trips, mut cycles) = (vec![], vec![]);
    for wave in waves {
        let times: Vec<i64> = wave.instructions.iter().filter(|i| i.pc == edge).map(|i| i.time).collect();
        if !times.is_empty() {
            trips.push(times.len() as i64);
            cycles.extend(times.windows(2).map(|w| w[1] - w[0]));
        }
    }
    Some(MainLoop {
        head: head - base,
        back_edge: edge - base,
        matrix: matrix(&(head, edge)),
        iterations: Spread::of(trips),
        cycles: Spread::of(cycles),
    })
}

#[cfg(test)]
#[path = "test/unit/sqtt.rs"]
mod tests;
