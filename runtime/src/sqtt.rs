//! An agent-facing reading of an AMD thread trace ([`SqttTrace`]): the device's decoder, wave model
//! and stitcher ([`svod_device::sqtt`]) over the kernel's `llvm-objdump` disassembly, summarized as a
//! fixed-format text ([`Analysis`]'s `Display`) or JSON. A [`Capture`] saves the raw streams with the
//! code object and the disassembly, so a trace is re-analyzed offline with no GPU.

use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use snafu::{OptionExt, ResultExt, Snafu, ensure};
use svod_device::amd::occupancy::{Cu, Limit, limits};
use svod_device::profile::{CodeImage, KernelResources, SeTrace, SqttTrace, StablePstate};
use svod_device::sqtt::asm::{self, Access, Counter, Kind, Memory, Space};
use svod_device::sqtt::hidden::{self, Hidden, Hiding};
pub use svod_device::sqtt::report::Spread;
use svod_device::sqtt::report::{Report, Stats};
use svod_device::sqtt::simd::{PipeBusy, ROBUST, Residency, STEADY, Simd};
use svod_device::sqtt::stitch::{Stitcher, branch, objdump_lines};
use svod_device::sqtt::waits::{Binding, Replay, Waits};
use svod_device::sqtt::{decode, wave};
use svod_dtype::AmdArch;

mod compare;
mod lines;
mod statements;
mod verdict;
pub use compare::{BucketDelta, Caution, Delta, Diff, FindingDelta, LoopRow, Rows, Side, StatementDelta, Unit};
pub use lines::{DwarfError, LISTING, LineRow, LineTable, Lines, Statements, Stmt, line_table, locate};
pub use statements::Statement;
pub use verdict::{
    Approx, BarrierSite, Bucket, Ceiling, Demand, Evidence, Finding, FindingKind, Findings, Knob, Load, Partition,
    PerTrip, Stall, Verdict, WaitOn,
};

#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum Error {
    #[snafu(display("{tool} not found beside clang or on PATH: {source}"))]
    Tool { tool: &'static str, source: Box<crate::Error> },
    #[snafu(display("{tool}: {source}"))]
    Spawn { tool: &'static str, source: std::io::Error },
    #[snafu(display("{tool} failed: {stderr}"))]
    Failed { tool: &'static str, stderr: String },
    #[snafu(display("{}: {source}", path.display()))]
    Io { source: std::io::Error, path: PathBuf },
    #[snafu(display("{}: {source}", path.display()))]
    Json { source: serde_json::Error, path: PathBuf },
    #[snafu(display("no instruction-traced shader engine has an RDNA3/RDNA4 trace header"))]
    NoItrace,
    #[snafu(display("se{se}: a trace header of version {version}, not RDNA3's 3 or RDNA4's 4"))]
    Header { se: u32, version: u8 },
    #[snafu(display("{arch}: not an AMD arch"))]
    Arch { arch: String },
    #[snafu(display("{}: capture format {found:?}, this build reads {FORMAT}; capture it again", path.display()))]
    Format { path: PathBuf, found: Option<u32> },
    #[snafu(display("the kernel's line table: {source}"))]
    LineTable { source: DwarfError },
}

/// The `meta.json` layout [`Capture::load`] reads; bumped whenever it changes.
pub const FORMAT: u32 = 2;

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// `llvm-objdump -d` of a code object, the tool found beside the clang svod compiles with.
pub fn disassemble(elf: &[u8]) -> Result<String> {
    run_tool("llvm-objdump", &["-d", "-"], elf)
}

/// The output of LLVM's `tool`, found beside the clang svod compiles with, run on `input`.
fn run_tool(tool: &'static str, args: &[&str], input: &[u8]) -> Result<String> {
    let path = crate::clang::ClangToolchain::discover(None)
        .and_then(|clang| clang.sibling(tool))
        .map_err(Box::new)
        .context(ToolSnafu { tool })?;
    let mut child = Command::new(path)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context(SpawnSnafu { tool })?;
    let mut stdin = child.stdin.take().expect("piped stdin");
    let output = std::thread::scope(|s| {
        // A pipe the tool closed early is its failure to report, not the writer's.
        s.spawn(move || drop(stdin.write_all(input)));
        child.wait_with_output()
    })
    .context(SpawnSnafu { tool })?;
    if !output.status.success() {
        return FailedSnafu { tool, stderr: String::from_utf8_lossy(&output.stderr).into_owned() }.fail();
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
    /// Free-form context (`cfg`, shape) shown in the report header, read by nothing.
    pub labels: BTreeMap<String, String>,
    /// What the kernel's program says of its main loop; `None` where the capture is no tk kernel's.
    pub model: Option<Model>,
    /// The kernel's registers and LDS, its workgroup's threads along each axis and the grid's
    /// workgroups along each: what bounds how many waves a SIMD holds.
    pub resources: Option<KernelResources>,
    pub workgroup: Option<[u32; 3]>,
    pub grid: Option<[u32; 3]>,
    /// The power state the traced dispatch ran at; `None` where the capture predates the record.
    pub pstate: Option<Pstate>,
    /// Each instruction's tile-program statement, and `kernel.tk`: from the code object's own line
    /// table ([`Statements::of`]); `None` where it has none.
    pub statements: Option<Statements>,
}

/// The power state a traced dispatch ran at: the device's own clock, or a stable state pinned
/// through the DRM context ([`StablePstate`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Pstate {
    Unpinned,
    Standard,
    Peak,
}

impl From<Option<StablePstate>> for Pstate {
    fn from(pin: Option<StablePstate>) -> Self {
        match pin {
            None => Self::Unpinned,
            Some(StablePstate::Standard) => Self::Standard,
            Some(StablePstate::Peak) => Self::Peak,
        }
    }
}

/// The axis a trip of a kernel's main loop reduces along.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Axis {
    /// A GEMM's (or a convolution's implicit GEMM's) K.
    K,
    /// Attention's keys.
    Keys,
    /// A nearest search's feature depth.
    Depth,
}

impl Axis {
    pub fn name(self) -> &'static str {
        match self {
            Self::K => "K",
            Self::Keys => "key",
            Self::Depth => "depth",
        }
    }
}

/// The elements of `axis` one trip covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Reduction {
    pub axis: Axis,
    pub elems: u64,
}

/// A term of the tile planner's GEMM price.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Term {
    Steps,
    Reads,
    Fill,
    Wait,
    LoneTrip,
    Trip,
}

impl Term {
    pub fn name(self) -> &'static str {
        match self {
            Self::Steps => "steps",
            Self::Reads => "reads",
            Self::Fill => "fill",
            Self::Wait => "wait",
            Self::LoneTrip => "lone_trip",
            Self::Trip => "trip",
        }
    }
}

/// The planner's price of a GEMM: its ns on the busiest SM, the trips a block runs, the blocks that
/// SM runs, and per term the price less that term.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Price {
    pub ns: f64,
    pub trips: u64,
    pub blocks_per_sm: Option<u64>,
    pub terms: BTreeMap<Term, f64>,
}

/// What a tk kernel's program says of its main loop: the reduction a trip covers and, for a GEMM,
/// the planner's price.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Model {
    pub per_trip: Option<Reduction>,
    pub price: Option<Price>,
}

impl Pstate {
    pub fn name(self) -> &'static str {
        match self {
            Self::Unpinned => "unpinned",
            Self::Standard => "standard",
            Self::Peak => "peak",
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct Meta {
    format: u32,
    arch: String,
    load_base: u64,
    entry_pc: u64,
    kernel_ns: Option<u64>,
    traced_ns: Option<u64>,
    ses: Vec<SeMeta>,
    labels: BTreeMap<String, String>,
    model: Option<Model>,
    resources: Option<KernelResources>,
    workgroup: Option<[u32; 3]>,
    grid: Option<[u32; 3]>,
    pstate: Option<Pstate>,
    mclk_mhz: Option<u64>,
}

/// A `meta.json`'s format alone, read before the rest.
#[derive(Deserialize)]
struct Version {
    format: Option<u32>,
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
    /// Disassembles the trace's code image, and reads its statements where it carries a line
    /// table over its listing (a tk kernel's, [`Statements::of`]).
    pub fn new(trace: SqttTrace, kernel_ns: Option<u64>) -> Result<Self> {
        let disassembly = disassemble(&trace.code.elf)?;
        let statements = Statements::of(&trace.code.elf, &disassembly).context(LineTableSnafu)?;
        Ok(Self {
            trace,
            disassembly,
            kernel_ns,
            traced_ns: None,
            labels: BTreeMap::new(),
            model: None,
            resources: None,
            workgroup: None,
            grid: None,
            pstate: None,
            statements,
        })
    }

    /// `se<N>.att` (raw streams), `kernel.co`, `kernel.s` and `meta.json` under `dir`: the
    /// statements are `kernel.co`'s own line table, read again on load.
    pub fn save(&self, dir: &Path) -> Result<()> {
        let code = &self.trace.code;
        io(dir, std::fs::create_dir_all(dir))?;
        let write = |name: &str, bytes: &[u8]| io(&dir.join(name), std::fs::write(dir.join(name), bytes));
        for se in &self.trace.per_se {
            write(&format!("se{}.att", se.se), &se.bytes)?;
        }
        write("kernel.co", &code.elf)?;
        write("kernel.s", self.disassembly.as_bytes())?;
        let meta = Meta {
            format: FORMAT,
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
            model: self.model.clone(),
            resources: self.resources,
            workgroup: self.workgroup,
            grid: self.grid,
            pstate: self.pstate,
            mclk_mhz: self.trace.mclk_mhz,
        };
        let path = dir.join("meta.json");
        write("meta.json", serde_json::to_string_pretty(&meta).context(JsonSnafu { path })?.as_bytes())
    }

    /// What [`Self::save`] wrote; disassembles `kernel.co` only where `kernel.s` is missing, and
    /// reads its statements from `kernel.co` as [`Self::new`] does.
    pub fn load(dir: &Path) -> Result<Self> {
        let read = |name: &str| io(&dir.join(name), std::fs::read(dir.join(name)));
        let path = dir.join("meta.json");
        let bytes = read("meta.json")?;
        let found = serde_json::from_slice::<Version>(&bytes).context(JsonSnafu { path: &path })?.format;
        ensure!(found == Some(FORMAT), FormatSnafu { path, found });
        let meta: Meta = serde_json::from_slice(&bytes).context(JsonSnafu { path })?;
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
        let elf: Arc<[u8]> = if dir.join("kernel.co").exists() { read("kernel.co")?.into() } else { Arc::default() };
        let disassembly = match dir.join("kernel.s").exists() {
            true => String::from_utf8_lossy(&read("kernel.s")?).into_owned(),
            false => disassemble(&elf)?,
        };
        let statements = match elf.is_empty() {
            true => None,
            false => Statements::of(&elf, &disassembly).context(LineTableSnafu)?,
        };
        let code = Arc::new(CodeImage { elf, load_base: meta.load_base, entry_pc: meta.entry_pc });
        Ok(Self {
            trace: SqttTrace { arch, per_se, code, mclk_mhz: meta.mclk_mhz },
            disassembly,
            kernel_ns: meta.kernel_ns,
            traced_ns: meta.traced_ns,
            labels: meta.labels,
            model: meta.model,
            resources: meta.resources,
            workgroup: meta.workgroup,
            grid: meta.grid,
            pstate: meta.pstate,
            statements,
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
        Self::from(asm::classify(asm).kind)
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

impl From<Kind> for Class {
    fn from(kind: Kind) -> Self {
        match kind {
            Kind::Matrix => Self::Matrix,
            Kind::Valu => Self::Valu,
            Kind::Salu => Self::Salu,
            Kind::Branch => Self::Branch,
            Kind::Memory(Memory { space: Space::Ds | Space::LdsDirect, .. }) => Self::Lds,
            Kind::Memory(Memory { space: Space::Smem, .. }) => Self::Smem,
            Kind::Memory(Memory { access: Access::Load | Access::Sample | Access::Bvh, .. }) => Self::VmemLoad,
            Kind::Memory(_) => Self::VmemStore,
            Kind::Wait(wait) if wait.idle => Self::Immediate,
            Kind::Wait(wait) => {
                let on = |counters: &[Counter]| wait.counts().any(|(c, _)| counters.contains(&c));
                if on(&[Counter::Load, Counter::Store, Counter::Sample, Counter::Bvh]) {
                    Self::WaitVmem
                } else if on(&[Counter::Ds, Counter::Km, Counter::Lgkm]) {
                    Self::WaitLgkm
                } else {
                    Self::Immediate
                }
            }
            Kind::Barrier => Self::Barrier,
            Kind::SendMsg | Kind::Message => Self::Message,
            Kind::Immediate | Kind::NoToken => Self::Immediate,
            Kind::Other => Self::Other,
        }
    }
}

/// One shader engine's stream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Se {
    pub se: u32,
    pub itrace: bool,
    pub bytes: usize,
    pub truncated: bool,
    /// The version of a `HEADER` the decoder refused: the stream from it on is left out.
    pub refused: Option<u8>,
    /// Waves started on any SIMD of the SE.
    pub waves: usize,
    /// Waves resident on the SE at once, at most.
    pub occupancy_peak: usize,
    /// Cycles from the first wave's start to the last's end.
    pub span: i64,
}

/// The SIMD whose instructions an SE traced, and its waves (those whose end the trace holds).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Traced {
    pub se: u32,
    pub sa: u8,
    pub wgp: u8,
    pub simd: u8,
    pub waves: usize,
    pub complete: usize,
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
    /// Of `idle`, `stall` and the issue cycles (`latency - stall`), those another pipe's activity on
    /// the SIMD overlapped: AMD's hidden latency. `stall - hidden_stall` is the exposed stall.
    pub hidden_idle: i64,
    pub hidden_stall: i64,
    pub hidden_issue: i64,
    /// Its tile-program statement: its `kernel.tk` line and unroll copy, where the capture has lines.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stmt: Option<Stmt>,
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MainLoop {
    /// ELF addresses of the loop head and its back edge.
    pub head: u64,
    pub back_edge: u64,
    pub matrix: bool,
    /// Back-edge executions of each wave that reached it.
    pub iterations: Option<Spread>,
    /// Cycles between a wave's consecutive back edges.
    pub cycles: Option<Spread>,
    pub trips: Trips,
    /// The traced waves' cycles in the loop, over every traced SE.
    pub partition: Partition,
    /// The same, SE by SE.
    pub ses: Vec<SeLoop>,
    /// The loop's barriers: who arrived last and after what.
    pub barriers: Vec<BarrierSite>,
    /// What the loop's waits waited for: [`Waits::bindings`] over the waits in the loop alone.
    pub bindings: Vec<Binding>,
    /// The loop's cycles by statement, where the capture has lines.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub statements: Vec<Statement>,
}

/// The unit per-trip figures are in: a trip of the kernel's main loop where its instructions carry
/// statements, else one loop iteration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Trips {
    /// Trips an iteration runs: the largest unroll copy ([`Stmt::copy`]) among the loop's
    /// statements, at least 1; 1 where they have none.
    pub per_iteration: u32,
    /// Trips the traced waves ran in the loop: their back-edge executions times `per_iteration`.
    pub total: u64,
    /// Whether `per_iteration` is the loop's statements' (else a trip is taken as an iteration).
    pub counted: bool,
}

/// One traced SE's waves in the main loop.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeLoop {
    pub se: u32,
    pub trips: u64,
    pub partition: Partition,
}

/// How many waves the kernel's resources let a SIMD hold, and how many the trace saw resident.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Occupancy {
    /// The CU the limit is for, and the kernel's limit there: where the capture records the kernel's
    /// resources and workgroup on an RDNA arch.
    pub cu: Option<Cu>,
    pub limit: Option<Limit>,
    /// The trace gives matrix ops one issue cycle (gfx11 sends them as VALU tokens): matrix and VALU
    /// busy undercount.
    pub untimed_matrix: bool,
    pub ses: Vec<SeOccupancy>,
}

/// One SE's resident waves over its steady window; on an instruction-traced SE whose traced SIMD
/// held waves, that SIMD's and its pipes' busy cycles in the window.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeOccupancy {
    pub se: u32,
    pub itrace: bool,
    pub residency: Residency,
    pub traced: Option<Simd>,
    pub busy: Option<PipeBusy>,
}

/// A trace's reading. Its JSON ([`Self::to_json`]) keeps these key names, which a reader may rely
/// on: `verdict` (`ceiling`, `demand`, and ranked [`Finding`]s in `bound` and `findings`: `kind`,
/// `share`, `raw`, `evidence`, `knob`, `approx`), `main_loop` (`trips`; `partition` with `waves` and
/// `buckets`, `[bucket, [raw, exposed]]` pairs; the same per SE in `ses`;
/// `barriers`, `bindings`), `occupancy`, `waits`, `pcs`, `classes`, `traced`, `ses`, `pstate`, `model`
/// and the header fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Analysis {
    pub kernel: String,
    pub arch: String,
    /// Lanes a wave runs; `None` where neither the capture's resources nor its code image say.
    pub wave_size: Option<u32>,
    pub labels: BTreeMap<String, String>,
    pub model: Option<Model>,
    pub kernel_ns: Option<u64>,
    pub traced_ns: Option<u64>,
    pub pstate: Option<Pstate>,
    pub ses: Vec<Se>,
    /// Each instruction-traced SE's traced SIMD.
    pub traced: Vec<Traced>,
    /// Waves on the traced SIMDs, and those whose end the trace holds.
    pub waves: usize,
    pub complete: usize,
    /// Cycles from start to end of the traced complete waves.
    pub lifetime: Option<Spread>,
    /// Shader clock: the longest instruction-traced SE's wave span over `traced_ns`, else `kernel_ns`.
    pub sclk_mhz: Option<u64>,
    /// Memory clock level current when the dispatch was armed.
    pub mclk_mhz: Option<u64>,
    /// Instruction records of the traced waves, and those a PC was attributed to.
    pub records: usize,
    pub stitched: usize,
    pub classes: Vec<ClassTime>,
    pub main_loop: Option<MainLoop>,
    /// Every executed instruction, by PC.
    pub pcs: Vec<Row>,
    /// What the traced waves' waits waited for.
    pub waits: Waits,
    pub occupancy: Occupancy,
    /// What limits the main loop.
    pub verdict: Verdict,
    /// `kernel.tk`, line by line, where the capture has lines.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub listing: Vec<String>,
}

/// The `<name>` of an `llvm-objdump` label line, `0000000000001700 <name>:`, with its address.
fn label(line: &str) -> Option<(u64, &str)> {
    let (at, rest) = line.trim().split_once(' ')?;
    let name = rest.strip_prefix('<')?.strip_suffix(">:")?;
    Some((u64::from_str_radix(at, 16).ok()?, name))
}

/// The capture's resources' wave size, else (where the device crate reads AMD code objects) its code
/// image's kernel descriptor's.
#[cfg_attr(not(unix), allow(unused_variables))]
fn wave_size(capture: &Capture, kernel: &str) -> Option<u32> {
    let resources = capture.resources;
    #[cfg(unix)]
    let resources = resources.or_else(|| {
        svod_device::amd::program::kernel_resources(&capture.trace.code.elf, kernel, capture.trace.arch).ok()
    });
    resources.map(|r| r.wave_size)
}

fn asm(line: &str) -> String {
    line.split("//").next().unwrap_or_default().split_whitespace().collect::<Vec<_>>().join(" ")
}

impl Analysis {
    /// The analysis of a [`Capture::save`]d directory.
    pub fn load(dir: &Path) -> Result<Self> {
        Capture::load(dir)?.analyze()
    }

    fn of(capture: &Capture) -> Result<Self> {
        let code = &capture.trace.code;
        let base = code.load_base;
        let lines: Vec<(u64, &str)> =
            objdump_lines(&capture.disassembly).into_iter().map(|(pc, line)| (pc + base, line)).collect();
        let (mut ses, mut traced, mut layout) = (vec![], vec![], None);
        // The traced SEs' waves, SE by SE: `(se, range)` into `waves`.
        let (mut waves, mut by_se, mut hidings) = (vec![], vec![], vec![]);
        let mut pcs = BTreeMap::<u64, Stats>::new();
        let matrix = hidden::matrix_pcs(&lines);
        let (mut hidden, mut occupancy) = (BTreeMap::<u64, Hidden>::new(), vec![]);

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
                refused: trace.refused,
                waves: starts,
                occupancy_peak: peak,
                span,
            });
            let residency = Residency::new(&trace.occupancy);
            let Some(h) = trace.header else { continue };
            if !se.itrace {
                occupancy.extend(residency.map(|residency| SeOccupancy {
                    se: se.se,
                    itrace: false,
                    residency,
                    traced: None,
                    busy: None,
                }));
                continue;
            }
            layout.get_or_insert(h.layout);
            let mut stitcher = Stitcher::new(&lines, h.layout);
            for wave in &mut trace.waves {
                stitcher.stitch(wave);
            }
            for (pc, h) in hidden::analyze([(se.se, &trace)], &matrix, base).by_pc() {
                *hidden.entry(pc).or_default() += h;
            }
            let hiding = Hiding::new(&trace.waves, &trace.other_simd, &matrix);
            occupancy.extend(residency.map(|residency| {
                let traced = residency.simd(h.dsa, h.dwgp, h.dsimd).copied();
                let busy = traced.map(|_| PipeBusy::of(&hiding.pipes, residency.window));
                SeOccupancy { se: se.se, itrace: true, traced, busy, residency }
            }));
            hidings.push(hiding);
            for (pc, s) in Report::new(&trace.waves).pcs {
                pcs.entry(pc).or_default().add(s);
            }
            traced.push(Traced {
                se: se.se,
                sa: h.dsa,
                wgp: h.dwgp,
                simd: h.dsimd,
                waves: trace.waves.len(),
                complete: trace.waves.iter().filter(|w| w.complete).count(),
            });
            by_se.push((se.se, waves.len()..waves.len() + trace.waves.len()));
            waves.append(&mut trace.waves);
        }
        let refused = |s: &&Se| s.itrace && s.refused.is_some();
        let layout = match (layout, ses.iter().find(refused)) {
            (Some(layout), _) => layout,
            (None, Some(s)) => return HeaderSnafu { se: s.se, version: s.refused.expect("refused") }.fail(),
            (None, None) => return NoItraceSnafu.fail(),
        };
        let text: BTreeMap<u64, &str> = lines.iter().copied().collect();
        let rows: Vec<Row> = pcs
            .iter()
            .map(|(&pc, s)| {
                let line = text.get(&pc).copied().unwrap_or_default();
                let (asm, class, pc) = (asm(line), Class::of(line), pc - base);
                let h = hidden.get(&pc).copied().unwrap_or_default();
                Row {
                    pc,
                    asm,
                    class,
                    hits: s.hits,
                    latency: s.latency,
                    stall: s.stall,
                    idle: s.idle,
                    hidden_idle: h.idle,
                    hidden_stall: h.stall,
                    hidden_issue: h.issue,
                    stmt: capture.statements.as_ref().and_then(|s| s.lines.get(&pc).copied()),
                }
            })
            .collect();
        let mut classes = BTreeMap::<Class, ClassTime>::new();
        for r in &rows {
            let c = classes.entry(r.class).or_insert(ClassTime { class: r.class, hits: 0, latency: 0, stall: 0 });
            (c.hits, c.latency, c.stall) =
                (c.hits.saturating_add(r.hits), c.latency.saturating_add(r.latency), c.stall.saturating_add(r.stall));
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
        let mut matrix_issue = waves.iter().flat_map(|w| &w.instructions).filter(|i| matrix.contains(&i.pc));
        let untimed_matrix = matrix_issue.clone().next().is_some() && matrix_issue.all(|i| i.duration - i.stall <= 1);
        let cu = Cu::rdna(capture.trace.arch).filter(|_| capture.resources.is_some() && capture.workgroup.is_some());
        let limit = cu
            .zip(capture.resources)
            .zip(capture.workgroup)
            .and_then(|((cu, resources), workgroup)| limits(&resources, workgroup.iter().product(), &cu));
        let replay = Replay::new(&waves, &lines, layout, base);
        let mut bound: Vec<HashMap<usize, (Counter, u64)>> = vec![HashMap::new(); waves.len()];
        for (wave, seq, counter, pc) in replay.bound() {
            bound[wave].insert(seq, (counter, pc));
        }
        let traced_waves = by_se.iter().zip(&hidings).map(|((se, range), hiding)| SeWaves {
            se: *se,
            waves: &waves[range.clone()],
            bound: &bound[range.clone()],
            hiding,
        });
        let listing: Vec<String> =
            capture.statements.iter().flat_map(|s| s.listing.lines()).map(str::to_string).collect();
        let statements = capture.statements.as_ref().map(|s| (&s.lines, listing.as_slice()));
        let main_loop = main_loop(&lines, &pcs, traced_waves, &replay, base, statements);
        let mut analysis = Self {
            kernel: kernel.to_string(),
            arch: capture.trace.arch.to_string(),
            wave_size: wave_size(capture, kernel),
            labels: capture.labels.clone(),
            model: capture.model.clone(),
            kernel_ns: capture.kernel_ns,
            traced_ns: capture.traced_ns,
            pstate: capture.pstate,
            waves: waves.len(),
            complete: waves.iter().filter(|w| w.complete).count(),
            traced,
            lifetime: Spread::of(lifetime),
            sclk_mhz: capture
                .traced_ns
                .or(capture.kernel_ns)
                .filter(|&ns| ns > 0 && span > 0)
                .and_then(|ns| (span as u128 * 1000 / u128::from(ns)).try_into().ok()),
            mclk_mhz: capture.trace.mclk_mhz,
            records: waves.iter().map(|w| w.instructions.len()).sum(),
            stitched: waves.iter().flat_map(|w| &w.instructions).filter(|i| i.pc != 0).count(),
            classes: classes.into_values().collect(),
            main_loop,
            waits: replay.waits(|_| true),
            occupancy: Occupancy { cu, limit, untimed_matrix, ses: occupancy },
            ses,
            pcs: rows,
            verdict: Verdict::default(),
            listing,
        };
        analysis.verdict = verdict::rank(&analysis);
        Ok(analysis)
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
        let labels: String = self.labels.iter().map(|(k, v)| format!(" {k}={v}")).collect();
        let _ = writeln!(o, "== sqtt {} {}{labels}", self.kernel, self.arch);
        for se in self.ses.iter().filter(|s| s.truncated) {
            let _ = writeln!(o, "!! TRUNCATED se{}: its buffer filled and the tail is lost; totals undercount", se.se);
        }
        for (se, version) in self.ses.iter().filter_map(|s| Some((s.se, s.refused?))) {
            let _ = writeln!(o, "!! UNSUPPORTED se{se}: a header of version {version}; the stream from it is left out");
        }
        if self.stitched < self.records {
            let _ =
                writeln!(o, "!! UNSTITCHED {} of {} records have no PC", self.records - self.stitched, self.records);
        }
        verdict::text(self, &mut o);
        let ns = |ns: Option<u64>| ns.map_or("-".into(), |ns| ns.to_string());
        let span = self.ses.iter().filter(|s| s.itrace).map(|s| s.span).max().unwrap_or(0);
        let sclk = self.sclk_mhz.map_or("-".into(), |mhz| format!("~{mhz}"));
        let mclk = self.mclk_mhz.map_or(String::new(), |mhz| format!(", mclk {mhz} MHz"));
        let (untraced, traced) = (ns(self.kernel_ns), ns(self.traced_ns));
        let ratio = verdict::slowdown(self).map_or(String::new(), |r| format!(" ({r:.2}x)"));
        let pstate = self.pstate.map_or("unrecorded", Pstate::name);
        let _ = writeln!(
            o,
            "kernel     {untraced} ns untraced, {traced} ns traced{ratio}, wave span {span} cycles, sclk {sclk} MHz{mclk}, pstate {pstate}"
        );
        let traced: Vec<String> = self
            .traced
            .iter()
            .map(|t| format!("se{} sa{} wgp{} simd{} ({} waves)", t.se, t.sa, t.wgp, t.simd, t.waves))
            .collect();
        let _ = writeln!(o, "traced     {} of {} SEs", traced.join(", "), self.ses.len());
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
        self.occupancy_text(&mut o);
        let table = |o: &mut String, rows: Vec<&Row>| {
            let _ = writeln!(
                o,
                "{:>8} {:>7} {:>9} {:>9} {:>9} {:>9}  asm",
                "pc", "hits", "cycles", "lat/hit", "stall/hit", "expo/hit"
            );
            for r in rows {
                let per = |c: i64| c as f64 / r.hits.max(1) as f64;
                let (l, s, e) = (per(r.latency), per(r.stall), per(r.stall - r.hidden_stall));
                let _ =
                    writeln!(o, "{:>#8x} {:>7} {:>9} {l:>9.1} {s:>9.1} {e:>9.1}  {}", r.pc, r.hits, r.latency, r.asm);
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
                partition_text(&mut o, l);
                statements::text(self, &mut o, top);
                let mut rows: Vec<&Row> = self.loop_rows().collect();
                rows.sort_by_key(|r| (std::cmp::Reverse(r.stall), r.pc));
                let _ = writeln!(o, "== main loop: top {top} by stall");
                table(&mut o, rows.into_iter().take(top).collect());
            }
        }
        self.waits_text(&mut o, top);
        let mut rows: Vec<&Row> = self.pcs.iter().collect();
        rows.sort_by_key(|r| (std::cmp::Reverse(r.latency), r.pc));
        let _ = writeln!(o, "== hottest {top} by cycles");
        table(&mut o, rows.into_iter().take(top).collect());
        o
    }

    /// The kernel's limit on resident waves, and per SE the waves resident over its steady window
    /// (`mean/peak`, the SE's and its SIMDs'), with the traced SIMD's and its pipes' busy shares.
    fn occupancy_text(&self, o: &mut String) {
        use std::fmt::Write;
        let occ = &self.occupancy;
        let (num, den) = STEADY;
        let _ = writeln!(
            o,
            "== occupancy: resident waves mean/peak over the steady window (the SE at {}% of its p{} or more)",
            100 * num / den,
            100 * ROBUST.0 / ROBUST.1
        );
        match &occ.limit {
            Some(l) => {
                let by: Vec<&str> = l.by.iter().map(|b| b.name()).collect();
                let (waves, groups, by) = (l.waves_per_simd, l.workgroups_per_cu, by.join("+"));
                let _ = writeln!(o, "limit      {waves} waves/SIMD, {groups} workgroups/CU, by {by}");
            }
            None => {
                let _ = writeln!(o, "limit      - (no kernel resources and workgroup in the capture)");
            }
        }
        for se in &occ.ses {
            let r = &se.residency;
            let span = self.ses.iter().find(|s| s.se == se.se).map_or(0, |s| s.span).max(1);
            let window = 100.0 * (r.window.1 - r.window.0) as f64 / span as f64;
            let peak = r.simds.iter().map(|s| s.peak).max().unwrap_or(0);
            let mean = r.simds.iter().map(Simd::mean).sum::<f64>() / r.simds.len().max(1) as f64;
            let traced = se.traced.map_or(String::new(), |t| format!("; traced {:.1}/{}", t.mean(), t.peak));
            let _ = writeln!(
                o,
                "se{:<8} {:.1}/{} (p{} {}) over {window:.0}% of the span; {} SIMDs {mean:.1}/{peak}{traced}",
                se.se,
                r.mean(),
                r.peak,
                100 * ROBUST.0 / ROBUST.1,
                r.robust_peak,
                r.simds.len()
            );
            if let Some(b) = se.busy {
                let pct = |c: i64| 100.0 * b.share(c);
                let _ = writeln!(
                    o,
                    "  busy     matrix {:.1}% valu {:.1}% vmem {:.1}% scalar {:.1}% any {:.1}%",
                    pct(b.matrix),
                    pct(b.valu),
                    pct(b.vmem),
                    pct(b.scalar),
                    pct(b.any)
                );
            }
        }
        if occ.untimed_matrix {
            let _ =
                writeln!(o, "note       matrix ops trace as 1-cycle VALU tokens here: matrix and valu busy undercount");
        }
    }

    /// The bindings whose waits stalled longest: `stalled` of `waits` stalled (latency over those),
    /// `wait` is the wait PC it bound most often and `distance` lists `back edges:instances`.
    fn waits_text(&self, o: &mut String, top: usize) {
        use std::fmt::Write;
        let w = &self.waits;
        let _ = writeln!(
            o,
            "== waits: {} on {} waves ({} skipped), {} unbound, {} approximate; top {top} bindings by stall",
            w.instances, w.waves, w.skipped, w.unbound, w.approx
        );
        let _ = writeln!(
            o,
            "{:>8} {:>7} {:>7} {:>9} {:>7} {:>7} {:>8} {:<12}  asm",
            "pc", "waits", "stalled", "stall", "lat med", "lat p90", "wait", "distance"
        );
        let asm: BTreeMap<u64, &str> = self.pcs.iter().map(|r| (r.pc, r.asm.as_str())).collect();
        for b in w.bindings.iter().take(top) {
            let wait = b.waits.iter().max_by_key(|&&(pc, n, _)| (n, std::cmp::Reverse(pc))).map_or(0, |w| w.0);
            let distance: Vec<String> = b.distance.iter().map(|(d, n)| format!("{d}:{n}")).collect();
            let (med, p90) = b.latency.map_or(("-".into(), "-".into()), |l| (l.median.to_string(), l.p90.to_string()));
            let asm = asm.get(&b.pc).copied().unwrap_or_default();
            let _ = writeln!(
                o,
                "{:>#8x} {:>7} {:>7} {:>9} {med:>7} {p90:>7} {wait:>#8x} {:<12}  {asm}",
                b.pc,
                b.count,
                b.stalled,
                b.stall,
                distance.join(",")
            );
        }
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

/// The main loop's partition a trip: every bucket, raw and exposed, its shares, and each traced SE's.
fn partition_text(o: &mut String, l: &MainLoop) {
    use std::fmt::Write;
    let t = l.trips;
    let unit = if t.counted { "a trip" } else { "an iteration" };
    let _ = writeln!(
        o,
        "trips      {} an iteration ({}), {} in the loop over {} waves",
        t.per_iteration,
        if t.counted { "its statements' unroll copies" } else { "no statements: per-trip figures are per iteration" },
        t.total,
        l.partition.waves
    );
    let ses: String = l.ses.iter().map(|s| format!(" {:>9}", format!("se{}", s.se))).collect();
    let _ = writeln!(
        o,
        "== main loop cycles {unit}: a traced wave's, each cycle once (issue, stall and idle by class); exposed: less what other pipes hid"
    );
    let _ = writeln!(o, "{:<20} {:>9} {:>7} {:>9} {:>7}{ses}", "bucket", "all", "share", "exposed", "share");
    let per = |cycles: i64, trips: u64| cycles as f64 / trips.max(1) as f64;
    let p = &l.partition;
    let total = p.cycles()[0].max(1) as f64;
    let rows = p.buckets.keys().map(|&b| (b.to_string(), Some(b))).chain([("total".into(), None)]);
    for (name, bucket) in rows {
        // A bucket's cycles, else every bucket's.
        let of = |p: &Partition| bucket.map_or(p.cycles(), |b| p.buckets.get(&b).copied().unwrap_or_default());
        let ses: String = l.ses.iter().map(|s| format!(" {:>9.1}", per(of(&s.partition)[0], s.trips))).collect();
        let [all, exposed] = of(p);
        let share = |c: i64| 100.0 * c as f64 / total;
        let _ = writeln!(
            o,
            "{name:<20} {:>9.1} {:>6.1}% {:>9.1} {:>6.1}%{ses}",
            per(all, t.total),
            share(all),
            per(exposed, t.total),
            share(exposed)
        );
    }
}

impl std::fmt::Display for Analysis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.text(10))
    }
}

/// One traced SE's waves, their wait records' bindings by record index, and the SE's pipes.
struct SeWaves<'a> {
    se: u32,
    waves: &'a [wave::Wave],
    bound: &'a [HashMap<usize, (Counter, u64)>],
    hiding: &'a Hiding,
}

/// The innermost executed backward branch whose span issues a matrix instruction, else the most
/// executed one; its waves' trips and the cycles between their back edges, and how the traced waves'
/// cycles in it split, SE by SE, and by the statement `stmts` gives each instruction; an iteration
/// runs as many trips as the largest unroll copy among its statements ([`Trips`]).
fn main_loop<'a>(
    lines: &[(u64, &str)],
    pcs: &BTreeMap<u64, Stats>,
    ses: impl Iterator<Item = SeWaves<'a>> + Clone,
    replay: &Replay,
    base: u64,
    stmts: Option<(&Lines, &[String])>,
) -> Option<MainLoop> {
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
    for wave in ses.clone().flat_map(|s| s.waves) {
        let times: Vec<i64> = wave.instructions.iter().filter(|i| i.pc == edge).map(|i| i.time).collect();
        if !times.is_empty() {
            trips.push(times.len() as i64);
            cycles.extend(times.windows(2).map(|w| w[1] - w[0]));
        }
    }
    let iterations = Spread::of(trips);
    let body: Vec<u64> =
        lines.iter().map(|l| l.0).filter(|pc| (head..=edge).contains(pc)).map(|pc| pc - base).collect();
    let copies =
        stmts.map_or(vec![], |(stmts, _)| body.iter().filter_map(|pc| stmts.get(pc)).map(|s| s.copy).collect());
    let per_iteration = copies.iter().copied().max().unwrap_or(1).max(1);
    let buckets = verdict::Buckets::new(lines);
    let (mut sites, mut charges) = (vec![], BTreeMap::<u64, verdict::Charge>::new());
    let ses: Vec<SeLoop> = ses
        .map(|s| {
            let edges = s.waves.iter().flat_map(|w| &w.instructions).filter(|i| i.pc == edge).count() as u64;
            let pass = buckets.partition(s.waves, s.bound, head..=edge, s.hiding);
            sites.extend(pass.barriers.into_values());
            for (pc, c) in pass.charges {
                charges.entry(pc - base).or_default().add(&c);
            }
            SeLoop { se: s.se, trips: edges * u64::from(per_iteration), partition: pass.partition }
        })
        .filter(|s| s.partition.waves > 0)
        .collect();
    let mut partition = Partition::default();
    ses.iter().for_each(|s| partition.add(&s.partition));
    let trips = Trips { per_iteration, total: ses.iter().map(|s| s.trips).sum(), counted: !copies.is_empty() };
    let statements = stmts.map_or(vec![], |(stmts, listing)| {
        statements::rollup(&charges, body.iter().copied(), stmts, listing, trips.total)
    });
    Some(MainLoop {
        head: head - base,
        back_edge: edge - base,
        matrix: matrix(&(head, edge)),
        iterations,
        cycles: Spread::of(cycles),
        trips,
        partition,
        ses,
        barriers: BarrierSite::merge(sites, base)
            .into_iter()
            .map(|s| BarrierSite { stmt: stmts.and_then(|(lines, _)| lines.get(&s.pc).copied()), ..s })
            .collect(),
        bindings: replay.waits(|pc| (head..=edge).contains(&pc)).bindings,
        statements,
    })
}

#[cfg(test)]
#[path = "test/unit/sqtt.rs"]
mod tests;
