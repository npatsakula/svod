//! The line table over `kernel.tk`: an AMD kernel's SOURCE stage
//! ([`Lowered::source`]) carries a line table and the listing itself, and
//! compiles to the `.text` the device's bare render does, on every AMD
//! target and on either side of clang's machine-sink gate, but for a pinned
//! few whose schedule clang picks by heap addresses; the pipeline
//! takes the stage as a device's own render; and a lowering renders it byte
//! for byte the same each time. Every test fails, not skips, where the
//! host's clang cannot build AMDGPU code: these re-check on every clang
//! upgrade what a capture's statements rest on.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::sync::Arc;

use rayon::prelude::*;
use svod_codegen::Renderer as _;
use svod_device::device::{ProgramSpec, Renderer};
use svod_device::profile::{CodeImage, SqttTrace};
use svod_device::sqtt::stitch::{branch, objdump_lines};
use svod_dtype::AmdArch;
use svod_ir::{Op, UOp, ops};
use svod_runtime::amd::{LocatedAmd, compile_ir_to_amd_object, compile_ir_to_amd_object_as};
use svod_runtime::sqtt::{Capture, Class, Lines, Statements, disassemble, line_table, locate};
use test_case::test_case;

use crate::atoms::{Core, Fill, SyncCaps};
use crate::build::{BF16, I8};
use crate::ir::{Expanded, Program, Stmt, Trip};
use crate::kernels::gemm::{Epilogue, GemmSpec, gemm};
use crate::lower::listing::{Role, listing, roles};
use crate::lower::{Lowered, Lowering, standalone};
use crate::ops::config;
use crate::schedule::expand;
use crate::sweep::{Launch, amd, assumed, digested, planner};

use super::gemm_spec;

/// The share of a ring GEMM's main-loop instructions that carry a line, at
/// least: 100% on the plain tiles, 92–96% where a ragged edge or an 8-bit
/// core adds guard logic CSE shares between statements (gfx1201, gfx1151,
/// 2026-11).
const GEMM_LINED: f64 = 0.90;
/// The same for every other family. Attention measured 66–90%, nearest
/// 79–99% and a tail split's persistent tile walk 81–88%: mask logic CSE
/// shares across statements (`s_or`, `v_cmp`, `s_and`), phi copies
/// (`v_mov`) and spills (`scratch_*`) take no line of LLVM's.
const OTHER_LINED: f64 = 0.60;

/// The LLVM majors on either side of the machine-sink gate: below 23 clang
/// compiles a matrix product with `-disable-machine-sink`.
const GATES: [u32; 2] = [22, 23];

/// The AMD targets of the sweep, by their CUs.
const ARCHES: [(AmdArch, u32); 5] = [
    (AmdArch::Gfx1100, 96),
    (AmdArch::Gfx1151, 40),
    (AmdArch::Gfx1201, 64),
    (AmdArch::Gfx942, 304),
    (AmdArch::Gfx950, 256),
];

/// A kernel of each template the sweep emits: the GEMM's ring, 8-bit core
/// and tail split (its waves and partials), attention's causal ring, cache
/// and combine, the norms, the heads, the nearest search's register, 8-bit,
/// queued and streamed lists, its selection and merge, and a convolution;
/// on CDNA also the GEMM's two crews.
const FAMILIES: [&str; 17] = [
    "gemm #0 ",
    "gemm int8 ",
    "gemm tail waves",
    "gemm tail partials",
    "attention d64 causal #0 ",
    "attention d64 cache",
    "attention combine",
    "layer norm",
    "rms norm",
    "heads d64",
    "nearest squared l2",
    "nearest int8 squared l2",
    "nearest 64 queued squared l2",
    "nearest streamed neg dot",
    "nearest select ",
    "nearest merge",
    "conv 3x3 ",
];

/// The suite's GEMM tiles: plain, unrolled, wide, ragged and 8-bit.
fn tiles(arch: AmdArch, cus: u32) -> Vec<Launch> {
    let planner = planner(amd(arch, cus));
    let mut out = vec![];
    for (name, (m, n, k), cfg) in [
        ("tile 128x128x32:2:4x2", (4096, 4096, 4096), config::gemm_cfg([128, 128, 32], 2, [4, 2], false)),
        ("tile 128x128x32:2:4x2 unrolled", (4096, 4096, 4096), config::gemm_cfg([128, 128, 32], 2, [4, 2], true)),
        ("tile 256x128x32:2:4x2", (4096, 4096, 4096), config::gemm_cfg([256, 128, 32], 2, [4, 2], false)),
        ("tile 128x128x32:2:4x2 ragged m1500", (1500, 1536, 512), config::gemm_cfg([128, 128, 32], 2, [4, 2], false)),
    ] {
        let spec = assumed(gemm_spec(m, n, k), cfg);
        out.push((name.to_string(), gemm::<BF16>(&spec), cfg.lowering(&planner.device)));
    }
    // The 8-bit core over a K deep enough to keep its loop rolled.
    let cfg = config::gemm_cfg([64, 64, 64], 2, [2, 2], true);
    let out_f32 = Epilogue { out_f32: true, ..Epilogue::DEFAULT };
    let spec = assumed(GemmSpec { epilogue: out_f32, ..gemm_spec(1024, 1024, 4096) }, cfg);
    out.push(("tile int8 64x64x64:2:2x2".into(), gemm::<I8>(&spec), cfg.lowering(&planner.device)));
    out
}

/// The first of the sweep's kernels for `arch` per [`FAMILIES`] entry (each
/// must have one), and the suite's GEMM tiles.
fn representatives(arch: AmdArch, cus: u32) -> Vec<Launch> {
    let all = digested(&amd(arch, cus));
    let crews = arch.is_cdna().then_some("gemm two crews");
    let mut out: Vec<Launch> = FAMILIES
        .iter()
        .chain(&crews)
        .map(|family| {
            let found = all.iter().find(|(name, ..)| name.starts_with(family));
            found.unwrap_or_else(|| panic!("{arch}: no {family:?} kernel")).clone()
        })
        .collect();
    out.extend(tiles(arch, cus));
    out
}

/// The SOURCE stage's text of a PROGRAM.
fn code(program: &Arc<UOp>) -> String {
    let Op::Program(ops::Program { source: Some(source), .. }) = program.op() else { panic!("no SOURCE stage") };
    let Op::Source(ops::Source { code, .. }) = source.op() else { panic!("the SOURCE stage is no SOURCE") };
    code.clone()
}

/// The text an AMD device renders of a lowering handed bare: its renderer
/// over the LINEAR stage, under the PROGRAM's function name.
fn bare(low: &Lowered, arch: AmdArch) -> String {
    let Op::Program(ops::Program { linear: Some(linear), info, .. }) = low.program.op() else { panic!("no LINEAR") };
    svod_runtime::amd::renderer(arch).render(linear, Some(&info.function_name())).expect("renders").code
}

/// A code object's `.text`.
fn text(elf: &[u8]) -> Vec<u8> {
    use object::{Object, ObjectSection};
    let file = object::File::parse(elf).expect("an ELF");
    file.section_by_name(".text").expect("a .text").data().expect("its bytes").to_vec()
}

/// A kernel's SOURCE compiled: its lowering, code object, disassembly and
/// its instructions' statements.
struct Located {
    low: Lowered,
    object: Vec<u8>,
    disassembly: String,
    lines: Lines,
}

fn located(arch: AmdArch, prog: Program, lowering: &Lowering) -> Located {
    let low = standalone(prog, lowering).expect("lowers");
    let object = compile_ir_to_amd_object(&code(&low.source(arch).expect("renders")), arch).expect("compiles");
    let disassembly = disassemble(&object).expect("llvm-objdump");
    let table = line_table(&object).expect("its DWARF reads").expect("a line table");
    assert_eq!(table.listing.as_deref(), Some(low.listing().as_str()), "the listing rides in the code object");
    let lines = locate(&table.rows, objdump_lines(&disassembly).into_iter().map(|(pc, _)| pc));
    Located { low, object, disassembly, lines }
}

/// The innermost backward branch's span holding a matrix instruction.
fn main_loop(disassembly: &str) -> Option<(u64, u64)> {
    let ops = objdump_lines(disassembly);
    ops.iter()
        .filter_map(|&(pc, line)| Some((branch(pc, line).filter(|&t| t <= pc)?, pc)))
        .filter(|&(s, e)| ops.iter().any(|&(pc, line)| (s..=e).contains(&pc) && Class::of(line) == Class::Matrix))
        .min_by_key(|(s, e)| e - s)
}

/// One kernel's gate: why it fails, if it does, and its report line. Its
/// SOURCE and the bare render compile to one `.text` on either side of the
/// machine-sink gate, its code object holds the listing and a line table,
/// and on RDNA, which a thread trace reads, enough of its main loop's
/// instructions carry a line (CDNA's crews and ragged edges measured 74% and
/// 90%, past what the floors were set on).
fn gate(arch: AmdArch, (name, prog, lowering): Launch) -> (Option<String>, String) {
    let built = located(arch, prog, &lowering);
    let (source, plain) = (code(&built.low.source(arch).unwrap()), bare(&built.low, arch));
    let differs = GATES.iter().find_map(|&major| {
        let [a, b] = [&plain, &source].map(|ir| text(&compile_ir_to_amd_object_as(ir, arch, major).expect("compiles")));
        (a != b).then(|| format!("{name}: .text differs under LLVM {major}'s flags ({} vs {} bytes)", a.len(), b.len()))
    });
    let rows = line_table(&built.object).unwrap().map_or(0, |t| t.rows.len());
    let ops = objdump_lines(&built.disassembly);
    let body: Vec<&(u64, &str)> = main_loop(&built.disassembly)
        .map_or(vec![], |(s, e)| ops.iter().filter(|(pc, _)| (s..=e).contains(pc)).collect());
    let mut unattributed = BTreeMap::<&str, usize>::new();
    for (_, line) in body.iter().filter(|(pc, _)| !built.lines.contains_key(pc)) {
        *unattributed.entry(line.split_whitespace().next().unwrap_or_default()).or_default() += 1;
    }
    let lined = body.len() - unattributed.values().sum::<usize>();
    let ring = (name.starts_with("gemm") || name.starts_with("tile")) && !name.starts_with("gemm tail");
    let floor = if ring { GEMM_LINED } else { OTHER_LINED };
    let why = if differs.is_some() {
        differs
    } else if rows == 0 {
        Some(format!("{name}: empty .debug_line"))
    } else if !arch.is_cdna() && (lined as f64) < floor * body.len() as f64 {
        Some(format!("{name}: {lined}/{} main-loop instructions carry a line, under {floor}", body.len()))
    } else {
        None
    };
    (why, format!("{name}: rows {rows} | main loop {lined}/{} lined | unattributed {unattributed:?}", body.len()))
}

fn gate_all(arch: AmdArch, launches: Vec<Launch>) {
    let gated: Vec<(Option<String>, String)> = launches.into_par_iter().map(|l| gate(arch, l)).collect();
    eprintln!("{}", gated.iter().map(|g| g.1.as_str()).collect::<Vec<_>>().join("\n"));
    let failed: Vec<&str> = gated.iter().filter_map(|g| g.0.as_deref()).collect();
    assert!(failed.is_empty(), "{arch}:\n{}", failed.join("\n"));
}

#[test_case(AmdArch::Gfx1100, 96; "gfx1100")]
#[test_case(AmdArch::Gfx1151, 40; "gfx1151")]
#[test_case(AmdArch::Gfx1201, 64; "gfx1201")]
#[test_case(AmdArch::Gfx942, 304; "gfx942")]
#[test_case(AmdArch::Gfx950, 256; "gfx950")]
fn a_line_table_leaves_the_code_alone(arch: AmdArch, cus: u32) {
    gate_all(arch, representatives(arch, cus));
}

/// The kernels whose `.text` a line table may change, on either side of the
/// machine-sink gate, pinned against clang 23.1.5: head-dim-48 flash
/// attention at 16 query rows and 16 or 32 keys on RDNA, whose schedule
/// clang picks by heap addresses. The same bare IR compiles to several
/// `.text`s from run to run (four in five compiles of one), one `.text`
/// each with address randomization off (`setarch -R`), where the bare and
/// the located IR, allocating otherwise, compile apart: the same VGPRs and
/// occupancy, other `s_waitcnt` and `s_delay_alu` counts. Which of them a
/// sweep catches differing changes from run to run.
fn rescheduled() -> HashSet<(AmdArch, String)> {
    let fa = |kind: &str, bkv: u32| {
        format!(
            "fa {kind} FaCfg {{ bq: 16, bkv: {bkv}, schedule: Ring {{ depth: 2, unroll: true }}, splits: 1, \
             forms: Forms {{ truncate_p: false, sum_after: false, skip_rescale: false, raw_max: false, \
             bias_direct: false }}, key_warps: 4, warp_rows: 16, core: Warp }}"
        )
    };
    let archs = [AmdArch::Gfx1100, AmdArch::Gfx1151, AmdArch::Gfx1201];
    let kernels = ["d48 t16", "causal_d48"].into_iter().flat_map(|kind| [16, 32].map(|bkv| fa(kind, bkv)));
    kernels.flat_map(|name| archs.map(|arch| (arch, name.clone()))).collect()
}

/// Every kernel of the sweep on every AMD target carries a line table, and
/// compiles to the bare render's `.text` under both gates' flags but for
/// [`rescheduled`]'s, whose code clang does not reproduce even bare.
#[test]
#[ignore = "every kernel of the sweep on every AMD target: ~20 minutes in release"]
fn a_line_table_leaves_every_kernel_alone() {
    let all: Vec<(AmdArch, Launch)> =
        ARCHES.iter().flat_map(|&(arch, cus)| digested(&amd(arch, cus)).into_iter().map(move |l| (arch, l))).collect();
    let known = rescheduled();
    let failed: Vec<String> = all
        .into_par_iter()
        .filter_map(|(arch, (name, prog, lowering))| {
            let low = standalone(prog, &lowering).ok()?;
            let (source, plain) = (code(&low.source(arch).unwrap()), bare(&low, arch));
            let mut rows = 0;
            let differs: Vec<u32> = GATES
                .into_iter()
                .filter(|&major| {
                    let [a, b] =
                        [&plain, &source].map(|ir| compile_ir_to_amd_object_as(ir, arch, major).expect("compiles"));
                    rows = line_table(&b).unwrap().map_or(0, |t| t.rows.len());
                    text(&a) != text(&b)
                })
                .collect();
            if rows == 0 {
                Some(format!("{arch} {name}: no line table"))
            } else if differs.is_empty() || known.contains(&(arch, name.clone())) {
                None
            } else {
                Some(format!("{arch} {name}: .text differs under LLVM {differs:?}'s flags"))
            }
        })
        .collect();
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// What a device's pipeline makes of a PROGRAM: the PROGRAM it compiles,
/// and every field of the specification it launches by but the source.
fn specified(program: &Arc<UOp>, renderer: &dyn Renderer) -> (Arc<UOp>, String) {
    let (rendered, spec) = svod_codegen::program_pipeline::do_render(program, renderer).expect("the pipeline takes it");
    let ProgramSpec {
        name, device, global_size, local_size, vars, var_names, globals, outs, ins, buf_count, abi, ..
    } = spec;
    let ids = |dims: &[Arc<UOp>]| dims.iter().map(|d| d.id).collect::<Vec<_>>();
    let launch = (ids(&global_size), local_size.as_ref().map(|l| ids(l)), vars, var_names, globals, outs, ins);
    (rendered, format!("{name} {device:?} {launch:?} {buf_count} {abi:?}"))
}

/// The pipeline takes a SOURCE stage as minted, origins stripped, and
/// renders nothing more ([`do_render`]'s skip branch): the PROGRAM comes
/// back as handed, launched by the specification its render would give.
///
/// [`do_render`]: svod_codegen::program_pipeline::do_render
#[test_case(AmdArch::Gfx1201, 64; "gfx1201")]
#[test_case(AmdArch::Gfx942, 304; "gfx942")]
fn the_pipeline_takes_the_source_as_rendered(arch: AmdArch, cus: u32) {
    for (name, prog, lowering) in representatives(arch, cus) {
        let low = standalone(prog, &lowering).expect("lowers");
        let device = svod_dtype::DeviceSpec::Amd { device_id: 0 };
        let renderer = LocatedAmd::new(device, arch, low.locations(), low.listing());
        let source = low.source(arch).expect("renders").without_origins();
        let (taken, skipped) = specified(&source, &renderer);
        assert!(Arc::ptr_eq(&taken, &source), "{name}: a PROGRAM with its SOURCE is taken as is");
        let (rendered, spec) = specified(&low.program, &renderer);
        assert_eq!(code(&rendered), code(&source), "{name}");
        assert_eq!(skipped, spec, "{name}");
    }
}

/// Lowering and rendering a kernel again gives the same SOURCE byte for
/// byte, its identity with it: the stage keys the kernel and object caches
/// across processes.
#[test]
fn the_source_is_the_same_every_time() {
    let arch = AmdArch::Gfx1201;
    let stage = |p: &Arc<UOp>| match p.op() {
        Op::Program(ops::Program { source: Some(s), .. }) => s.clone(),
        _ => unreachable!("a SOURCE stage"),
    };
    for (name, prog, lowering) in representatives(arch, 64) {
        let [a, b] =
            [0, 1].map(|_| standalone(prog.clone(), &lowering).expect("lowers").source(arch).expect("renders"));
        assert_eq!(code(&a), code(&b), "{name}");
        assert!(Arc::ptr_eq(&stage(&a), &stage(&b)), "{name}: one SOURCE, hash-consed");
    }
}

/// A located code object loads as its bare build does: the device's
/// loader lays out and relocates the sections it loads alone, so the line
/// table's `.debug_*` sections and their `R_AMDGPU_ABS32`s leave the image
/// and the kernel descriptor as they were.
#[test_case(AmdArch::Gfx1201, 64; "gfx1201")]
#[test_case(AmdArch::Gfx942, 304; "gfx942")]
fn a_located_object_loads_as_the_bare_one(arch: AmdArch, cus: u32) {
    representatives(arch, cus).into_par_iter().for_each(|(name, prog, lowering)| {
        let low = standalone(prog, &lowering).expect("lowers");
        let Op::Program(ops::Program { info, .. }) = low.program.op() else { unreachable!("a PROGRAM") };
        let [located, plain] = [code(&low.source(arch).unwrap()), bare(&low, arch)].map(|ir| {
            let object = compile_ir_to_amd_object(&ir, arch).expect("compiles");
            svod_device::amd::program::parse_kernel(&object, &info.function_name()).expect("loads")
        });
        assert!(located.image == plain.image && located.kd_offset == plain.kd_offset, "{name}");
    });
}

fn tile_gemm(cfg: crate::kernels::gemm::GemmCfg) -> (Program, Lowering) {
    let planner = planner(amd(AmdArch::Gfx1201, 64));
    (gemm::<BF16>(&assumed(gemm_spec(4096, 4096, 4096), cfg)), cfg.lowering(&planner.device))
}

/// The listing names what a GEMM's statements do and the roles the
/// register-staged template gave them, a line a statement.
#[test]
fn the_listing_names_statements_and_roles() {
    let (prog, lowering) = tile_gemm(config::gemm_cfg([128, 128, 32], 2, [4, 2], true));
    let low = standalone(prog, &lowering).unwrap();
    let listing = low.listing();
    let lines: Vec<&str> = listing.lines().map(str::trim).collect();
    assert_eq!(lines.len(), low.tile.walk().count(), "a line a statement:\n{listing}");
    for want in [
        "fill a staged issue [head]",
        "fill b staged issue [steady]",
        "lds read a_s [steady]",
        "mma a_s·b_s [steady]",
        "mma a_s·b_s [drain]",
        "loop x2 [steady]",
        "fence [steady]",
        "barrier [steady]",
    ] {
        assert!(lines.contains(&want), "no `{want}` in\n{listing}");
    }
    let has = |prefix: &str, role: &str| lines.iter().any(|l| l.starts_with(prefix) && l.ends_with(role));
    assert!(has("fill a staged commit -> ", "[steady]"), "{listing}");
    assert!(has("store c", "[epilogue]") && has("", "[prologue]"), "{listing}");
}

/// A template tags the trips it expands a ring into and the listing reads
/// roles from the tags: a `cp.async` ring's one loop runs steady, with a
/// prologue and an epilogue around it, and what stands between two
/// pipelines of one block is the second's prologue, not the first's drain.
#[test_case(Fill::CpAsync, &[Trip::Steady]; "one loop")]
#[test_case(Fill::Staged, &[Trip::Head, Trip::Steady, Trip::Drain]; "peeled")]
fn the_roles_are_the_templates_trips(fill: Fill, trips: &[Trip]) {
    let mut prog = expand(super::gemm_nt(64, 32, 160, 32, 32, 32, 2), fill, Core::Warp, SyncCaps::default());
    let trip = |s: &Stmt<Expanded>| if let Stmt::Loop(l) = s { l.trip } else { None };
    assert_eq!(prog.body.0.iter().filter_map(trip).collect::<Vec<_>>(), trips);
    // The one pipeline twice in a row: roles read only the statements' shapes.
    prog.body.0.extend(prog.body.0.clone());
    let roles = roles(&prog);
    let top: Vec<Option<Role>> = prog.walk().zip(roles).filter(|((depth, _), _)| *depth == 0).map(|(_, r)| r).collect();
    let body = &prog.body.0;
    let second = body.iter().rposition(|s| trip(s) == trips.first().copied()).unwrap();
    let end = body.iter().rposition(|s| trip(s) == trips.last().copied()).unwrap();
    let want: Vec<Option<Role>> = (body.iter().enumerate())
        .map(|(i, s)| match trip(s) {
            Some(t) => Some(t.into()),
            None if i < second => Some(Role::Prologue),
            None if i > end => Some(Role::Epilogue),
            None => unreachable!("no statement of the template's stands between its trips"),
        })
        .collect();
    assert_eq!(top, want, "\n{}", listing(&prog));
}

/// End to end on the host: a GEMM's SOURCE compiled puts its main loop's
/// matrix instructions on the steady loop's product statement, one unroll
/// copy a column; a capture of that code object reads them from it, and
/// keeps them saved and loaded back; a capture of a kernel built bare has
/// none.
#[test]
fn matrix_instructions_sit_on_the_product_statement() {
    let arch = AmdArch::Gfx1201;
    let (prog, lowering) = tile_gemm(config::gemm_cfg([128, 128, 32], 2, [4, 2], true));
    let built = located(arch, prog, &lowering);
    let listing = built.low.listing();
    let text: Vec<&str> = listing.lines().map(str::trim).collect();
    let (head, edge) = main_loop(&built.disassembly).expect("a matrix loop");
    let matrix: Vec<u64> = objdump_lines(&built.disassembly)
        .into_iter()
        .filter(|&(pc, line)| (head..=edge).contains(&pc) && Class::of(line) == Class::Matrix)
        .map(|(pc, _)| pc)
        .collect();
    assert!(!matrix.is_empty());
    for pc in &matrix {
        let line = built.lines[pc].line.get();
        assert_eq!(text[line as usize - 1], "mma a_s·b_s [steady]", "{pc:#x} on line {line}");
    }
    let columns: BTreeSet<u32> = matrix.iter().map(|pc| built.lines[pc].copy).collect();
    assert_eq!(columns, BTreeSet::from([1, 2]), "the steady loop's two unroll copies");

    let trace = |elf: &[u8]| SqttTrace {
        arch,
        per_se: vec![],
        mclk_mhz: None,
        code: Arc::new(CodeImage { elf: elf.into(), load_base: 1 << 40, entry_pc: 1 << 40 }),
    };
    let capture = Capture::new(trace(&built.object), None).unwrap();
    assert_eq!(capture.statements, Some(Statements { lines: built.lines.clone(), listing: listing.clone() }));
    let dir = tempfile::tempdir().unwrap();
    capture.save(dir.path()).unwrap();
    assert_eq!(Capture::load(dir.path()).unwrap().statements, capture.statements, "saved and loaded back");

    let (prog, lowering) = tile_gemm(config::gemm_cfg([64, 64, 64], 2, [2, 2], false));
    let bare =
        compile_ir_to_amd_object(&standalone(prog, &lowering).unwrap().text(&lowering.device.target).unwrap(), arch);
    assert_eq!(Capture::new(trace(&bare.unwrap()), None).unwrap().statements, None, "a kernel without a line table");
}

/// `located` with its line table taken out: the `!dbg` attachments and the
/// debug metadata [`Lowered::source`] adds.
fn unlocated(located: &str) -> String {
    let debug = |l: &str| {
        let node = l.strip_prefix('!').and_then(|l| l.split_once(" = ")).map(|(_, node)| node);
        l.starts_with("!llvm.dbg.cu")
            || l.starts_with("!llvm.module.flags")
            || node.is_some_and(|n| n.trim_start_matches("distinct ").starts_with("!DI") || n.contains(" Version\""))
    };
    let lines = located.split_inclusive('\n').filter(|l| !debug(l)).map(|l| match l.find(" !dbg !") {
        Some(at) => {
            let id = &l[at + " !dbg !".len()..];
            let end = l.len() - id.len() + id.find(|c: char| !c.is_ascii_digit()).unwrap_or(id.len());
            format!("{}{}", l[..at].strip_suffix(',').unwrap_or(&l[..at]), &l[end..])
        }
        None => l.to_string(),
    });
    lines.collect()
}

/// The AMD arch a launch runs on here: this host's device's, else gfx1201
/// (with no device attached a launch is not checked against one), with its
/// CUs.
fn launch_arch() -> (AmdArch, u32) {
    let arch = svod_device::registry::resolve_amd_arch_from_topology(0).unwrap_or(AmdArch::Gfx1201);
    (arch, ARCHES.iter().find(|a| a.0 == arch).map_or(64, |a| a.1))
}

/// The PROGRAM [`launch::graph_launch_once`] hands the graph of `prog`
/// launched on an AMD device over lazy tensors of its parameters.
///
/// [`launch::graph_launch_once`]: crate::launch::graph_launch_once
fn launched(prog: Program, lowering: &Lowering) -> Arc<UOp> {
    let device = svod_dtype::DeviceSpec::Amd { device_id: 0 };
    let out = svod_dtype::default_device::with_default_device(device, || {
        let tensors: Vec<svod_tensor::Tensor> =
            prog.params.iter().map(|p| svod_tensor::Tensor::empty(&[p.elems], p.dtype.into())).collect();
        crate::launch::graph_launch_once(prog, lowering, &tensors.iter().collect::<Vec<_>>()).expect("launches")
    });
    let (mut stack, mut seen) = (vec![out.uop()], std::collections::HashSet::new());
    while let Some(node) = stack.pop() {
        if matches!(node.op(), Op::Program(..)) {
            return node;
        }
        stack.extend(node.op().sources().into_iter().filter(|s| seen.insert(s.id)));
    }
    panic!("the launch holds no PROGRAM")
}

/// The LLVM text an AMD device renders of `prog` launched bare: its
/// renderer over the launch's LINEAR stage, under its function name.
fn launched_text(arch: AmdArch, prog: Program, lowering: &Lowering) -> String {
    let program = launched(prog, lowering);
    let Op::Program(ops::Program { linear: Some(linear), info, .. }) = program.op() else { panic!("no LINEAR") };
    svod_runtime::amd::renderer(arch).render(linear, Some(&info.function_name())).expect("renders").code
}

/// A launch on an AMD device hands the graph its kernel's SOURCE, origins
/// stripped: its code object carries the line table and the listing, and
/// holds the `.text` the device's own render of the bare kernel would.
#[test]
fn a_launch_hands_the_device_its_line_table() {
    let (arch, cus) = launch_arch();
    let cfg = config::gemm_cfg([128, 128, 32], 2, [4, 2], true);
    let prog = gemm::<BF16>(&assumed(gemm_spec(1500, 1280, 5120), cfg));
    let lowering = cfg.lowering(&planner(amd(arch, cus)).device);
    let program = launched(prog.clone(), &lowering);
    assert!(program.toposort().iter().all(|n| n.origin().is_none()), "no origins");
    let object = compile_ir_to_amd_object(&code(&program), arch).expect("compiles");
    let table = line_table(&object).expect("its DWARF reads").expect("a line table");
    assert_eq!(table.listing, Some(standalone(prog.clone(), &lowering).unwrap().listing()));
    assert!(table.rows.iter().any(|r| r.line > 0));
    let bare = compile_ir_to_amd_object(&launched_text(arch, prog, &lowering), arch).expect("compiles");
    assert_eq!(text(&object), text(&bare));
}

/// A launch's SOURCE is the device's bare render of the launched kernel,
/// located: its line table taken out, the text is the bare one, and the two
/// compile to one `.text`; a launch numbers the parameters `[out, ins...]`,
/// whatever order the program declares them in (the GEMM's output last).
#[test]
fn a_launched_source_is_the_bare_kernel_located() {
    let (arch, cus) = launch_arch();
    let planner = planner(amd(arch, cus));
    let tiles = [
        ((1500, 1280, 5120), config::gemm_cfg([128, 128, 32], 2, [4, 2], false)),
        ((1536, 1280, 5120), config::gemm_cfg([128, 128, 32], 2, [4, 2], false)),
        ((4096, 4096, 4096), config::gemm_cfg([128, 128, 32], 2, [4, 2], true)),
        ((1500, 1536, 512), config::gemm_cfg([256, 128, 32], 2, [4, 2], false)),
    ];
    tiles.into_par_iter().for_each(|((m, n, k), cfg)| {
        let prog = gemm::<BF16>(&assumed(gemm_spec(m, n, k), cfg));
        let lowering = cfg.lowering(&planner.device);
        let bare = launched_text(arch, prog.clone(), &lowering);
        let located = code(&launched(prog, &lowering));
        assert!(located.contains(", !dbg !"), "{m}x{n}x{k}: no locations");
        assert_eq!(unlocated(&located), bare, "{m}x{n}x{k} {cfg:?}");
        let [a, b] = [located, bare].map(|ir| text(&compile_ir_to_amd_object(&ir, arch).unwrap()));
        assert!(a == b, "{m}x{n}x{k} {cfg:?}: .text differs");
    });
}
