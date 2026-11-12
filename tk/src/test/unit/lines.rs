//! The line table over `kernel.tk`: an AMDGPU code object compiled from a
//! lowered program's located text holds the same `.text` as one compiled
//! without locations, a non-empty `.debug_line`, and statement lines on its
//! main loop's instructions. The suite gates a representative kernel of
//! each family an RDNA arch runs; the whole sweep is `#[ignore]`d.

use std::collections::{BTreeMap, BTreeSet};

use rayon::prelude::*;
use svod_device::profile::{CodeImage, SqttTrace};
use svod_device::sqtt::stitch::{branch, objdump_lines};
use svod_dtype::AmdArch;
use svod_runtime::amd::compile_ir_to_amd_object;
use svod_runtime::sqtt::{Capture, Class, Lines, debug_line, disassemble, locate, parse_debug_line, text_mismatch};
use test_case::test_case;

use crate::build::{BF16, I8};
use crate::ir::Program;
use crate::kernels::gemm::{Epilogue, GemmSpec, gemm};
use crate::lower::{Lowered, Lowering, standalone};
use crate::ops::config;
use crate::sweep::{Launch, amd, assumed, families, planner};

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

/// The suite's families on `arch` and the GEMM tiles the suite runs on RDNA.
fn launches(arch: AmdArch, cus: u32) -> Vec<Launch> {
    let target = amd(arch, cus);
    let planner = planner(target.clone());
    let mut out = families(&target);
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

/// Whether this host compiles AMDGPU code and has the LLVM tool a line
/// table is read with (it runs, whatever it makes of no input).
fn tooled() -> bool {
    use svod_runtime::sqtt::Error;
    let dwarfdump = !matches!(debug_line(&[]), Err(Error::Tool { .. } | Error::Spawn { .. }));
    let tooled = svod_runtime::amd::has_amdgpu_target() && dwarfdump;
    if !tooled {
        eprintln!("skipped: no clang AMDGPU or llvm-dwarfdump");
    }
    tooled
}

/// A kernel built with a line table: its lowering, code object,
/// disassembly and its instructions' statements.
struct Located {
    low: Lowered,
    object: Vec<u8>,
    disassembly: String,
    lines: Lines,
}

fn located(arch: AmdArch, prog: Program, lowering: &Lowering) -> Located {
    let low = standalone(prog, lowering).expect("lowers");
    let object = compile_ir_to_amd_object(&low.located_text(&lowering.device.target).unwrap(), arch).unwrap();
    let disassembly = disassemble(&object).unwrap();
    let rows = parse_debug_line(&debug_line(&object).unwrap());
    let lines = locate(&rows, objdump_lines(&disassembly).into_iter().map(|(pc, _)| pc));
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

/// One kernel's gate: why it fails, if it does, and its report line.
fn gate(arch: AmdArch, (name, prog, lowering): Launch) -> (Option<String>, String) {
    let built = located(arch, prog, &lowering);
    let plain = compile_ir_to_amd_object(&built.low.text(&lowering.device.target).unwrap(), arch).unwrap();
    let rows = parse_debug_line(&debug_line(&built.object).unwrap()).len();
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
    let why = if let Some(differs) = text_mismatch(&plain, &built.object) {
        Some(format!("{name}: {differs}"))
    } else if rows == 0 {
        Some(format!("{name}: empty .debug_line"))
    } else if (lined as f64) < floor * body.len() as f64 {
        Some(format!("{name}: {lined}/{} main-loop instructions carry a line, under {floor}", body.len()))
    } else {
        None
    };
    (why, format!("{name}: rows {rows} | main loop {lined}/{} lined | unattributed {unattributed:?}", body.len()))
}

fn gate_all(arch: AmdArch, cus: u32, keep: impl Fn(&str) -> bool) {
    if !tooled() {
        return;
    }
    let picked: Vec<Launch> = launches(arch, cus).into_iter().filter(|l| keep(&l.0)).collect();
    let gated: Vec<(Option<String>, String)> = picked.into_par_iter().map(|l| gate(arch, l)).collect();
    eprintln!("{}", gated.iter().map(|g| g.1.as_str()).collect::<Vec<_>>().join("\n"));
    let failed: Vec<&str> = gated.iter().filter_map(|g| g.0.as_deref()).collect();
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// A plain and a ragged GEMM tile, an 8-bit GEMM and a causal attention.
const REPRESENTATIVE: [&str; 4] = [
    "tile 128x128x32:2:4x2 unrolled",
    "tile 128x128x32:2:4x2 ragged m1500",
    "tile int8 64x64x64:2:2x2",
    "attention d64 causal #0",
];

#[test_case(AmdArch::Gfx1201, 64; "gfx1201")]
#[test_case(AmdArch::Gfx1151, 40; "gfx1151")]
fn a_line_table_leaves_the_code_alone(arch: AmdArch, cus: u32) {
    gate_all(arch, cus, |name| REPRESENTATIVE.iter().any(|r| name == *r || name.starts_with(&format!("{r} "))));
}

#[test_case(AmdArch::Gfx1201, 64; "gfx1201")]
#[test_case(AmdArch::Gfx1151, 40; "gfx1151")]
#[ignore = "every family: ~100 s in release"]
fn a_line_table_leaves_every_kernel_alone(arch: AmdArch, cus: u32) {
    gate_all(arch, cus, |_| true);
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

/// End to end on the host: a GEMM lowered with locations and compiled puts
/// its main loop's matrix instructions on the steady loop's product
/// statement, one unroll copy a column; a capture of that code object
/// takes the twin's lines, and a capture of other code refuses them.
#[test]
fn matrix_instructions_sit_on_the_product_statement() {
    if !tooled() {
        return;
    }
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
        let (line, _) = built.lines[pc];
        assert_eq!(text[line as usize - 1], "mma a_s·b_s [steady]", "{pc:#x} on line {line}");
    }
    let columns: BTreeSet<u32> = matrix.iter().map(|pc| built.lines[pc].1).collect();
    assert_eq!(columns, BTreeSet::from([1, 2]), "the steady loop's two unroll copies");

    let trace = |elf: &[u8]| SqttTrace {
        arch,
        per_se: vec![],
        mclk_mhz: None,
        code: Some(CodeImage { elf: elf.to_vec(), load_base: 1 << 40, entry_pc: 1 << 40 }),
    };
    let mut capture = Capture::new(trace(&built.object), None).unwrap();
    assert_eq!(capture.attach_twin(&built.object, &listing).unwrap(), None);
    assert_eq!(capture.lines.as_ref(), Some(&built.lines));
    assert_eq!(capture.listing.as_deref(), Some(listing.as_str()));

    let (prog, lowering) = tile_gemm(config::gemm_cfg([64, 64, 64], 2, [2, 2], false));
    let elsewhere =
        compile_ir_to_amd_object(&standalone(prog, &lowering).unwrap().text(&lowering.device.target).unwrap(), arch);
    let mut capture = Capture::new(trace(&elsewhere.unwrap()), None).unwrap();
    let why = capture.attach_twin(&built.object, &listing).unwrap();
    assert!(why.as_deref().is_some_and(|w| w.starts_with(".text differs")), "{why:?}");
    assert_eq!((capture.lines, capture.listing), (None, None));
}

/// `located` with its line table taken out: the `!dbg` attachments and the
/// debug metadata [`Lowered::located_text_by`] adds.
fn unlocated(located: &str) -> String {
    // The debug nodes are numbered from 2^20, clear of the loop hints.
    let node = |l: &str| l.strip_prefix('!')?.split_once(" = ")?.0.parse::<u32>().ok();
    let debug = |l: &str| {
        l.starts_with("!llvm.dbg.cu") || l.starts_with("!llvm.module.flags") || node(l).is_some_and(|n| n >= 1 << 20)
    };
    let lines = located.split_inclusive('\n').filter(|l| !debug(l)).map(|l| match l.find(", !dbg !DILocation(") {
        Some(at) => format!("{}{}", &l[..at], &l[at + l[at..].find(')').expect("a closed location") + 1..]),
        None => l.replace(" !dbg !1048578", ""),
    });
    lines.collect()
}

/// The LLVM text an AMD device renders of `prog` launched over lazy tensors
/// of its parameters: the PROGRAM [`launch::graph_launch_once`] hands the
/// graph, rendered as the device's renderer renders it, under its function name.
fn launched_text(arch: AmdArch, prog: Program, lowering: &Lowering) -> String {
    use svod_codegen::Renderer;
    use svod_ir::{Op, ops};
    let device = svod_dtype::DeviceSpec::Amd { device_id: 0 };
    let out = svod_dtype::default_device::with_default_device(device, || {
        let tensors: Vec<svod_tensor::Tensor> =
            prog.params.iter().map(|p| svod_tensor::Tensor::empty(&[p.elems], p.dtype.into())).collect();
        crate::launch::graph_launch_once(prog, lowering, &tensors.iter().collect::<Vec<_>>()).expect("launches")
    });
    let (mut stack, mut seen) = (vec![out.uop()], std::collections::HashSet::new());
    while let Some(node) = stack.pop() {
        if let Op::Program(ops::Program { linear: Some(linear), info, .. }) = node.op() {
            let renderer = svod_runtime::amd::renderer(arch);
            return renderer.render(linear, Some(&info.function_name())).expect("renders").code;
        }
        stack.extend(node.op().sources().into_iter().filter(|s| seen.insert(s.id)));
    }
    panic!("the launch holds no PROGRAM")
}

/// The twin a capture's statements come from is the kernel a launch on the
/// device compiles: its located text, the line table taken out, is the
/// launch's text, and the two compile to one `.text`. A launch numbers the
/// parameters `[out, ins...]`; a twin lowered over them in declared order
/// (the GEMM's output last) loads its kernel arguments in another order.
#[test_case(AmdArch::Gfx1201, 64; "gfx1201")]
#[test_case(AmdArch::Gfx1151, 40; "gfx1151")]
fn a_twin_is_the_launched_kernel(arch: AmdArch, cus: u32) {
    if !tooled() {
        return;
    }
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
        let launched = launched_text(arch, prog.clone(), &lowering);
        let twin = crate::launch::lowered_as_launched(prog, &lowering).expect("lowers");
        let located = twin.located_text_by(svod_runtime::amd::renderer(arch)).unwrap();
        assert!(located.contains("!dbg !DILocation"), "{m}x{n}x{k}: no locations");
        assert_eq!(unlocated(&located), launched, "{m}x{n}x{k} {cfg:?}");
        let [twin, launched] = [located, launched].map(|ir| compile_ir_to_amd_object(&ir, arch).unwrap());
        assert_eq!(text_mismatch(&launched, &twin), None, "{m}x{n}x{k} {cfg:?}");
    });
}
