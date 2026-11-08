//! From a tile program to a pre-linearized Svod program: fence the author's
//! shared-memory effects, expand the schedule template, give every register
//! tile a layout, then emit the instruction list in program order.

use std::sync::Arc;

use snafu::{ResultExt, Snafu, ensure};
use svod_codegen::Renderer;
use svod_codegen::llvm::LlvmTextRenderer;
use svod_dtype::{DType, DeviceSpec, GpuArch};
use svod_ir::{Op, UOp, ops};

use crate::atoms::{Core, Device, SharedBanks, Target};
use crate::ir::*;
use crate::layouts::{self, Laid, TileLayout};
use crate::schedule;

pub mod emit;
pub mod sync;

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("layout inference: {source}"))]
    Layout { source: layouts::Error },
    #[snafu(display("{what} is not lowered yet"))]
    Unsupported { what: String },
    #[snafu(display("program assembly: {source}"))]
    Program { source: svod_ir::Error },
    #[snafu(display("{warps} warps over a grid of {grid}: the grid takes them all, or whole grids run on their own"))]
    Grid { warps: u32, grid: u32 },
    #[snafu(display("rendering {name}: {source}"))]
    Render {
        name: String,
        #[snafu(source(from(svod_codegen::Error, Box::new)))]
        source: Box<svod_codegen::Error>,
    },
    #[snafu(display("{what} is a block barrier under a branch on the warp, which some warps skip"))]
    DivergentBarrier { what: &'static str },
    #[snafu(display("{name} emits {instructions} instructions, past the {most} a kernel may"))]
    TooLarge { name: String, instructions: usize, most: usize },
    #[snafu(display("{name} counts to {extent}, past the i32 a range, grid or variable holds"))]
    ExtentPastI32 { name: String, extent: i64 },
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// What the lowering is given beside the program: the target and its
/// costs, and the matrix instruction products issue on. The warp grid and
/// each ring's schedule are the program's own ([`Program::warp_grid`],
/// [`Pipeline`]).
#[derive(Clone, Debug)]
pub struct Lowering {
    /// The target and what its primitives cost: the costs decide each
    /// product's orientation and how a tile is re-held.
    pub device: Device,
    /// On the warpgroup atom ([`crate::atoms::GroupMmaAtom`]) the grid's
    /// rows are warpgroups of four warps, shared operands are read through
    /// descriptors (over the swizzled rows) and never loaded into registers.
    pub core: Core,
}

impl Lowering {
    /// `device`'s warp-level core.
    pub fn new(device: &Device) -> Self {
        Self { device: device.clone(), core: Core::Warp }
    }
}

pub struct Lowered {
    pub program: Arc<UOp>,
    pub tile: Program,
    pub layouts: Vec<Option<TileLayout>>,
    /// The ranges the program's own schedule kept rolled (a loop whose body
    /// holds a product), which [`Lowered::text`] hints every target against
    /// unrolling further.
    pub rolled: Vec<u64>,
}

/// `params` are the flat buffer UOps in `prog.params` order (placeholders on
/// the graph path, `UOp::param`s on the direct one).
pub fn lower(mut prog: Program, lowering: &Lowering, params: Vec<Arc<UOp>>, device: DeviceSpec) -> Result<Lowered> {
    // An input is never stored to ([`crate::build::Kernel::input`]), which the backend may know.
    let params = params
        .iter()
        .zip(&prog.params)
        .map(|(u, p)| if p.kind == ParamKind::In { u.with_readonly() } else { u.clone() })
        .collect();
    let grid = prog.warp_grid.count();
    let lone = prog.walk().any(|(_, s)| matches!(s, Stmt::Warps { .. }));
    ensure!(prog.warps.is_multiple_of(grid) && (prog.warps == grid || lone), GridSnafu { warps: prog.warps, grid });
    materialize_operands(&mut prog, lowering.core);
    sync::insert_barriers(&mut prog)?;
    schedule::expand(&mut prog, lowering.device.target.fill, lowering.core);
    let mut laid = layouts::infer(prog, &lowering.device, lowering.core).context(LayoutSnafu)?;
    orient_shared(&mut laid, &lowering.device.target);
    emit::ROLLED.with(|cell| cell.borrow_mut().clear());
    let program = emit::emit(&laid, lowering, params, device)?;
    let rolled = emit::ROLLED.with(|cell| std::mem::take(&mut *cell.borrow_mut()));
    let Laid { prog, layouts } = laid;
    Ok(Lowered { program, tile: prog, layouts, rolled })
}

/// `prog` lowered for its lowering's target alone, its parameters declared
/// flat: what a graph launch on the target's device would emit, with no
/// graph and no device.
pub fn standalone(prog: Program, lowering: &Lowering) -> Result<Lowered> {
    let params =
        prog.params.iter().enumerate().map(|(i, p)| UOp::param(i, p.elems, DType::Scalar(p.dtype), None)).collect();
    let device = match lowering.device.target.arch {
        GpuArch::Cuda(_) => DeviceSpec::Cuda { device_id: 0 },
        GpuArch::Amd(_) => DeviceSpec::Amd { device_id: 0 },
        GpuArch::Metal(_) => DeviceSpec::Metal { device_id: 0 },
    };
    lower(prog, lowering, params, device)
}

/// The LLVM text of `prog` [`standalone`] ([`Lowered::text`]).
pub fn render(prog: Program, lowering: &Lowering) -> Result<String> {
    standalone(prog, lowering)?.text(&lowering.device.target)
}

impl Lowered {
    /// The instructions the program emitted (within
    /// [`Target::most_instructions`]).
    pub fn instructions(&self) -> usize {
        let Op::Program(ops::Program { linear: Some(linear), .. }) = self.program.op() else {
            unreachable!("a lowering emits a linear program")
        };
        linear.op().sources().len()
    }

    /// The LLVM text of the program for `target`: what the backend's
    /// compiler is handed. Metal renders C, which has no text here.
    pub fn text(&self, target: &Target) -> Result<String> {
        let renderer = match target.arch {
            GpuArch::Cuda(arch) => LlvmTextRenderer::nvptx(arch),
            GpuArch::Amd(arch) => LlvmTextRenderer::amd(arch),
            GpuArch::Metal(_) => return UnsupportedSnafu { what: "Metal's LLVM text" }.fail(),
        }
        .rolled(self.rolled.iter().copied());
        let Op::Program(ops::Program { linear: Some(linear), .. }) = self.program.op() else {
            unreachable!("a lowering emits a linear program")
        };
        if svod_schedule::spec::spec_enabled() {
            let nodes = linear.op().sources();
            svod_schedule::spec::type_verify_list(&nodes, &svod_schedule::spec::spec_program()).expect("a valid list");
        }
        let name = &self.tile.name;
        Ok(renderer.render(linear, Some(name)).context(RenderSnafu { name })?.code)
    }
}

/// Store every shared allocation along the axis its readers' fragments run
/// along: a tile whose consumers each hold runs of rows (a value tile under
/// the product's B operand, say) is kept column-major, so the gathers are
/// 16-byte loads instead of one load per element. Only allocations filled
/// through registers can be stored transposed (the stores scatter; a chunk
/// copy cannot), and readers that disagree keep the row-major default.
fn orient_shared(laid: &mut Laid, target: &Target) {
    let Laid { prog, layouts } = laid;
    let lanes = target.wave;
    let alloc = |prog: &Program, v: ValId| match prog.value(v).place {
        Place::Smem { alloc, .. } => Some(alloc.index()),
        _ => None,
    };
    let mut wants: Vec<Option<Axis>> = vec![None; prog.smem.len()];
    let mut staged = vec![true; prog.smem.len()];
    for (_, stmt) in prog.walk() {
        let Stmt::Copy { dst, src, mode, .. } = stmt else { continue };
        if let (Some(a), Tier::Global) = (alloc(prog, *dst), prog.value(*src).tier()) {
            staged[a] &= *mode == CopyMode::Staged;
        }
        let (Some(a), Tier::Reg) = (alloc(prog, *src), prog.value(*dst).tier()) else { continue };
        // A view under another type is scratch over a dead region: it takes
        // the allocation's axis as it finds it.
        if prog.value(*dst).dtype != prog.smem[a].dtype {
            continue;
        }
        let l = layouts[dst.index()].as_ref().expect("a laid register tile");
        let width = 16 / prog.value(*dst).dtype.bytes() as u32;
        let widest = |along| l.runs(lanes, width, along).iter().map(|&(_, w)| w).max().unwrap_or(1);
        let along = if widest(Axis::Row) > widest(Axis::Col) { Axis::Row } else { Axis::Col };
        wants[a] = Some(match wants[a] {
            Some(other) if other != along => Axis::Col,
            _ => along,
        });
    }
    let transposed: Vec<usize> = (0..prog.smem.len()).filter(|&i| staged[i] && wants[i] == Some(Axis::Row)).collect();
    for &i in &transposed {
        prog.smem[i].along = Axis::Row;
    }
    // The staged registers of a column-major tile hold column runs where the
    // target loads them transposed, else are walked down its rows so a
    // wave's lanes store consecutive elements; a warp's own copy spreads
    // its chunks over that warp's lanes alone.
    for (_, stmt) in prog.walk() {
        let Stmt::Copy { dst, src, mode: CopyMode::Staged, scope } = stmt else { continue };
        if !(alloc(prog, *dst).is_some_and(|a| transposed.contains(&a)) && prog.value(*src).tier() == Tier::Reg) {
            continue;
        }
        let tmp = prog.value(*src);
        let bytes = tmp.dtype.bytes();
        let warps = if *scope == Scope::Warp { 1 } else { prog.warps };
        let l = layouts::transposing(tmp.shape, warps, lanes)
            .filter(|_| target.tr_load && bytes == 2)
            .or_else(|| layouts::chunked(tmp.shape, bytes, warps, lanes, Axis::Row));
        if let Some(l) = l {
            layouts[src.index()] = Some(l);
        }
    }
    if let Some(banks) = target.shared_banks {
        meet_banks(prog, layouts, banks, lanes);
    }
}

/// Exchange the chunks' halves of each shared allocation whose readers reach
/// the target's banks only that way. An allocation qualifies when one tile
/// shape reads it whole (the exchange follows the view's row) and no reader
/// holds a run across a half. Only banks of half a chunk are met this way, on
/// the row bit that doubles the banks the lower bits reach.
fn meet_banks(prog: &mut Program, layouts: &[Option<TileLayout>], banks: SharedBanks, lanes: u32) {
    if 2 * banks.bytes != 16 {
        return;
    }
    let bit = banks.count.ilog2() - 1;
    for a in 0..prog.smem.len() {
        let (along, dtype) = (prog.smem[a].along, prog.smem[a].dtype);
        let views = prog.values.iter().filter(|v| matches!(v.place, Place::Smem { alloc, .. } if alloc.index() == a));
        let shapes: Vec<_> = views.map(|v| (v.shape, v.dtype)).collect();
        let Some(&(shape, _)) = shapes.first() else { continue };
        let bytes = dtype.bytes();
        let cols = if along == Axis::Row { shape.rows } else { shape.cols };
        if bytes > 8 || !cols.is_multiple_of(16 / bytes) || shapes.iter().any(|&s| s != (shape, dtype)) {
            continue;
        }
        let readers: Vec<&TileLayout> = prog
            .walk()
            .filter_map(|(_, stmt)| match stmt {
                Stmt::Copy { dst, src, .. } => match (&prog.value(*src).place, prog.value(*dst).tier()) {
                    (Place::Smem { alloc, .. }, Tier::Reg) if alloc.index() == a => layouts[dst.index()].as_ref(),
                    _ => None,
                },
                _ => None,
            })
            .collect();
        let warps = prog.warp_grid.count();
        let group: Vec<u32> = (0..lanes).collect();
        let width = (16 / bytes) as u32;
        // Whether each `banks.count` consecutive lanes of every read reach as many banks.
        let meets = |halves: Option<u32>| {
            readers.iter().all(|l| {
                let bank = |warp, lane, reg| {
                    let (r, c) = l.coord(warp, lane, reg);
                    let (r, c) = if along == Axis::Row { (c, r) } else { (r, c) };
                    let at = layouts::shared_element(r as usize, c as usize, cols, bytes, halves) * bytes;
                    1u64 << (at / banks.bytes as usize % banks.count as usize)
                };
                (0..warps).all(|warp| {
                    l.runs(lanes, width, along).iter().all(|&(reg, _)| {
                        group.chunks(banks.count as usize).all(|g| {
                            g.iter().fold(0, |seen, &lane| seen | bank(warp, lane, reg)).count_ones() as usize
                                == g.len()
                        })
                    })
                })
            })
        };
        let narrow = readers
            .iter()
            .all(|l| l.runs(lanes, width, along).iter().all(|&(_, w)| w as usize * bytes <= banks.bytes as usize));
        let exchange = !readers.is_empty() && narrow && !meets(None) && meets(Some(bit));
        prog.smem[a].halves = exchange.then_some(bit);
    }
}

/// An `Mma` reads registers: a shared or global operand becomes an explicit
/// load into a fresh register tile just before it, so the inference gives
/// that tile the operand layout and the emitter sees one copy op. A
/// warpgroup product ([`Core::Group`]) reads shared operands in place,
/// through descriptors; only a global one is loaded.
pub(crate) fn materialize_operands(prog: &mut Program, core: Core) {
    let body = std::mem::take(&mut prog.body);
    prog.body = materialize_block(prog, body, Scope::Block, core);
}

fn materialize_block(prog: &mut Program, block: Block, scope: Scope, core: Core) -> Block {
    let mut out = Vec::with_capacity(block.0.len());
    for mut stmt in block.0 {
        match &mut stmt {
            Stmt::Let { op: TileOp::Mma { a, b, .. }, .. } => {
                for operand in [a, b] {
                    let place = &prog.value(*operand).place;
                    if *place != Place::Reg && !(core == Core::Group && matches!(place, Place::Smem { .. })) {
                        let Value { dtype, shape, .. } = prog.value(*operand).clone();
                        prog.values.push(Value { dtype, shape, place: Place::Reg });
                        let tmp = ValId(prog.values.len() as u32 - 1);
                        out.push(Stmt::Copy { dst: tmp, src: *operand, mode: CopyMode::Sync, scope });
                        *operand = tmp;
                    }
                }
            }
            nested => {
                let scope = if matches!(nested, Stmt::Warps { .. }) { Scope::Warp } else { scope };
                for b in nested.blocks_mut() {
                    *b = materialize_block(prog, std::mem::take(b), scope, core);
                }
            }
        }
        out.push(stmt);
    }
    Block(out)
}
