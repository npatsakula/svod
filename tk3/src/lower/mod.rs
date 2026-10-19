//! From a tile program to a pre-linearized Svod program: fence the author's
//! shared-memory effects, expand the schedule template, give every register
//! tile a layout, then emit the instruction list in program order.

use std::sync::Arc;

use snafu::{ResultExt, Snafu, ensure};
use svod_codegen::Renderer;
use svod_codegen::llvm::LlvmTextRenderer;
use svod_dtype::{DType, DeviceSpec, GpuArch};
use svod_ir::{Op, UOp, ops};

use crate::atoms::{Core, Device, Target};
use crate::ir::*;
use crate::layouts::{self, Laid, TileLayout, WarpGrid};
use crate::schedule::{self, Schedule};

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
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Everything the lowering decides that the author did not.
#[derive(Clone, Debug)]
pub struct Lowering {
    /// The target and what its primitives cost: the costs decide each
    /// product's orientation and how a tile is re-held.
    pub device: Device,
    pub schedule: Schedule,
    pub grid: WarpGrid,
    /// XOR-swizzle 16-byte chunks of shared rows against the row index.
    pub swizzle: bool,
    /// The matrix instruction products issue on; on the warpgroup atom
    /// ([`crate::atoms::GroupMmaAtom`]) the grid's rows are warpgroups of
    /// four warps, shared operands are read through descriptors (over the
    /// swizzled rows, so `swizzle` holds) and never loaded into registers.
    pub core: Core,
}

impl Lowering {
    /// `device`'s uniform pipeline over `grid`, its ring `unroll`ed or not,
    /// shared rows swizzled.
    pub fn uniform(device: &Device, unroll: bool, grid: WarpGrid) -> Self {
        let schedule = Schedule::Uniform { fill: device.target.fill, unroll, async_products: false };
        Self { device: device.clone(), schedule, grid, swizzle: true, core: Core::Warp }
    }
}

pub struct Lowered {
    pub program: Arc<UOp>,
    pub tile: Program,
    pub layouts: Vec<Option<TileLayout>>,
}

/// `params` are the flat buffer UOps in `prog.params` order (placeholders on
/// the graph path, `UOp::param`s on the direct one).
pub fn lower(mut prog: Program, lowering: &Lowering, params: Vec<Arc<UOp>>, device: DeviceSpec) -> Result<Lowered> {
    let grid = lowering.grid.count();
    let lone = prog.walk().any(|(_, s)| matches!(s, Stmt::Warps { .. }));
    ensure!(prog.warps.is_multiple_of(grid) && (prog.warps == grid || lone), GridSnafu { warps: prog.warps, grid });
    materialize_operands(&mut prog, lowering.core);
    sync::insert_barriers(&mut prog)?;
    schedule::expand(&mut prog, lowering.schedule);
    let mut laid = layouts::infer(prog, &lowering.device, lowering.grid, lowering.core).context(LayoutSnafu)?;
    orient_shared(&mut laid, &lowering.device.target);
    let program = emit::emit(&laid, lowering, params, device)?;
    let Laid { prog, layouts } = laid;
    Ok(Lowered { program, tile: prog, layouts })
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
    /// The LLVM text of the program for `target`: what the backend's
    /// compiler is handed. Metal renders C, which has no text here.
    pub fn text(&self, target: &Target) -> Result<String> {
        let renderer = match target.arch {
            GpuArch::Cuda(arch) => LlvmTextRenderer::nvptx(arch),
            GpuArch::Amd(arch) => LlvmTextRenderer::amd(arch),
            GpuArch::Metal(_) => return UnsupportedSnafu { what: "Metal's LLVM text" }.fail(),
        };
        let Op::Program(ops::Program { linear: Some(linear), .. }) = self.program.op() else {
            unreachable!("a lowering emits a linear program")
        };
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
            Stmt::Loop(l) => l.body = materialize_block(prog, std::mem::take(&mut l.body), scope, core),
            Stmt::Pipeline(p) => {
                p.produce.body = materialize_block(prog, std::mem::take(&mut p.produce.body), scope, core);
                p.consume.body = materialize_block(prog, std::mem::take(&mut p.consume.body), scope, core);
            }
            Stmt::Warps { body } => *body = materialize_block(prog, std::mem::take(body), Scope::Warp, core),
            Stmt::If { then, otherwise, .. } => {
                *then = materialize_block(prog, std::mem::take(then), scope, core);
                *otherwise = materialize_block(prog, std::mem::take(otherwise), scope, core);
            }
            Stmt::Let { .. } | Stmt::Copy { .. } | Stmt::Sync(_) => {}
        }
        out.push(stmt);
    }
    Block(out)
}
