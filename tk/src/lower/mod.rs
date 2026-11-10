//! From a tile program to a pre-linearized Svod program: fence the author's
//! shared-memory effects, expand the schedule template, give every register
//! tile a layout, then emit the instruction list in program order.

use std::sync::Arc;

use snafu::{ResultExt, Snafu, ensure};
use svod_codegen::Renderer;
use svod_codegen::llvm::LlvmTextRenderer;
use svod_dtype::{DType, DeviceSpec, GpuArch};
use svod_ir::UOp;

use crate::atoms::{Core, Device, Target};
use crate::ir::*;
use crate::layouts;
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
    #[snafu(display("a view of {name} reaches elements {from}..{to}, which holds {held}"))]
    SharedPast { name: String, from: i64, to: i64, held: usize },
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
    /// The instruction list `program` holds (its LINEAR stage).
    pub linear: Arc<UOp>,
    /// The tile program as emitted: laid, with every register value's layout.
    pub tile: Program<Laid>,
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
    within_shared(&prog)?;
    materialize_operands(&mut prog, lowering.core);
    sync::insert_barriers(&mut prog)?;
    let prog = schedule::expand(prog, lowering.device.target.fill, lowering.core);
    let laid = layouts::infer(prog, &lowering.device, lowering.core).context(LayoutSnafu)?;
    emit::ROLLED.with(|cell| cell.borrow_mut().clear());
    let (program, linear) = emit::emit(&laid, lowering, params, device)?;
    let rolled = emit::ROLLED.with(|cell| std::mem::take(&mut *cell.borrow_mut()));
    Ok(Lowered { program, linear, tile: laid, rolled })
}

/// Every shared view within its allocation wherever its offset is bounded,
/// a loop's induction over its trips and a pipeline's slot over its ring:
/// a ring that holds fewer slots than its pipeline runs is refused here, not
/// read past on the device.
fn within_shared(prog: &Program) -> Result<()> {
    let mut ranges = std::collections::HashMap::new();
    for (_, stmt) in prog.walk() {
        let trips = |extent| prog.bounds(extent, &|iv| ranges.get(&iv).copied()).filter(|&(_, hi)| hi > 0);
        match stmt {
            Stmt::Loop(l) => {
                let start = prog.bounds(l.start, &|iv| ranges.get(&iv).copied());
                if let (Some((lo, hi)), Some((_, n))) = (start, trips(l.extent)) {
                    ranges.insert(l.iv, (lo, hi + n - 1));
                }
            }
            Stmt::Pipeline(p) => {
                if let Some((_, n)) = trips(p.extent) {
                    ranges.extend([(p.produce.step, (0, n - 1)), (p.consume.step, (0, n - 1))]);
                }
                let last = p.schedule.stages() as i64 - 1;
                ranges.extend([(p.produce.slot, (0, last)), (p.consume.slot, (0, last))]);
            }
            _ => {}
        }
    }
    for v in &prog.values {
        let Place::Smem { alloc, offset } = v.place else { continue };
        let a = &prog.smem[alloc.index()];
        let held = a.elems * a.dtype.bytes() / v.dtype.bytes();
        if let Some((from, hi)) = prog.bounds(offset, &|iv| ranges.get(&iv).copied()) {
            let to = hi + v.shape.elems() as i64;
            ensure!(from >= 0 && to <= held as i64, SharedPastSnafu { name: a.name.clone(), from, to, held });
        }
    }
    Ok(())
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
        self.linear.op().sources().len()
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
        if svod_schedule::spec::spec_enabled() {
            let nodes = self.linear.op().sources();
            svod_schedule::spec::type_verify_list(&nodes, &svod_schedule::spec::spec_program()).expect("a valid list");
        }
        let name = &self.tile.name;
        Ok(renderer.render(&self.linear, Some(name)).context(RenderSnafu { name })?.code)
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
