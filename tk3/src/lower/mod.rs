//! From a tile program to a pre-linearized Svod program: expand the schedule
//! template, give every register tile a layout, then emit the instruction
//! list in program order.

use std::sync::Arc;

use snafu::{ResultExt, Snafu};
use svod_dtype::DeviceSpec;
use svod_ir::UOp;

use crate::atoms::Target;
use crate::ir::*;
use crate::layouts::{self, Laid, TileLayout, WarpGrid};
use crate::schedule::{self, Schedule};

mod emit;
pub mod sync;

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("layout inference: {source}"))]
    Layout { source: layouts::Error },
    #[snafu(display("{what} is not lowered yet"))]
    Unsupported { what: String },
    #[snafu(display("program assembly: {source}"))]
    Program { source: svod_ir::Error },
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Everything the lowering decides that the author did not.
#[derive(Clone, Debug)]
pub struct Lowering {
    pub target: Target,
    pub schedule: Schedule,
    pub grid: WarpGrid,
    /// XOR-swizzle 16-byte chunks of shared rows against the row index.
    pub swizzle: bool,
}

pub struct Lowered {
    pub program: Arc<UOp>,
    pub tile: Program,
    pub layouts: Vec<Option<TileLayout>>,
}

/// `params` are the flat buffer UOps in `prog.params` order (placeholders on
/// the graph path, `UOp::param`s on the direct one).
pub fn lower(mut prog: Program, lowering: &Lowering, params: Vec<Arc<UOp>>, device: DeviceSpec) -> Result<Lowered> {
    schedule::expand(&mut prog, lowering.schedule);
    materialize_operands(&mut prog);
    sync::insert_barriers(&mut prog);
    let mut laid = layouts::infer(prog, &lowering.target, lowering.grid).context(LayoutSnafu)?;
    orient_shared(&mut laid, &lowering.target);
    let program = emit::emit(&laid, lowering, params, device)?;
    let Laid { prog, layouts } = laid;
    Ok(Lowered { program, tile: prog, layouts })
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
        let Stmt::Copy { dst, src, mode } = stmt else { continue };
        if let (Some(a), Tier::Global) = (alloc(prog, *dst), prog.value(*src).tier()) {
            staged[a] &= *mode == CopyMode::Staged;
        }
        let (Some(a), Tier::Reg) = (alloc(prog, *src), prog.value(*dst).tier()) else { continue };
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
    // wave's lanes store consecutive elements.
    let walked: Vec<(ValId, TileLayout)> = prog
        .walk()
        .filter_map(|(_, stmt)| match stmt {
            Stmt::Copy { dst, src, mode: CopyMode::Staged }
                if alloc(prog, *dst).is_some_and(|a| transposed.contains(&a))
                    && prog.value(*src).tier() == Tier::Reg =>
            {
                let tmp = prog.value(*src);
                let bytes = tmp.dtype.bytes();
                layouts::transposing(tmp.shape, prog.warps, lanes)
                    .filter(|_| target.tr_load && bytes == 2)
                    .or_else(|| layouts::chunked(tmp.shape, bytes, prog.warps, lanes, Axis::Row))
                    .map(|l| (*src, l))
            }
            _ => None,
        })
        .collect();
    for (tmp, l) in walked {
        layouts[tmp.index()] = Some(l);
    }
}

/// An `Mma` reads registers: a shared or global operand becomes an explicit
/// load into a fresh register tile just before it, so the inference gives
/// that tile the operand layout and the emitter sees one copy op.
fn materialize_operands(prog: &mut Program) {
    let body = std::mem::take(&mut prog.body);
    prog.body = materialize_block(prog, body);
}

fn materialize_block(prog: &mut Program, block: Block) -> Block {
    let mut out = Vec::with_capacity(block.0.len());
    for mut stmt in block.0 {
        match &mut stmt {
            Stmt::Let { op: TileOp::Mma { a, b, .. }, .. } => {
                for operand in [a, b] {
                    if prog.value(*operand).place != Place::Reg {
                        let Value { dtype, shape, .. } = prog.value(*operand).clone();
                        prog.values.push(Value { dtype, shape, place: Place::Reg });
                        let tmp = ValId(prog.values.len() as u32 - 1);
                        out.push(Stmt::Copy { dst: tmp, src: *operand, mode: CopyMode::Sync });
                        *operand = tmp;
                    }
                }
            }
            Stmt::Loop(l) => l.body = materialize_block(prog, std::mem::take(&mut l.body)),
            Stmt::Pipeline(p) => {
                p.produce.body = materialize_block(prog, std::mem::take(&mut p.produce.body));
                p.consume.body = materialize_block(prog, std::mem::take(&mut p.consume.body));
            }
            Stmt::Role { body, .. } => *body = materialize_block(prog, std::mem::take(body)),
            Stmt::If { then, otherwise, .. } => {
                *then = materialize_block(prog, std::mem::take(then));
                *otherwise = materialize_block(prog, std::mem::take(otherwise));
            }
            Stmt::Let { .. } | Stmt::Copy { .. } | Stmt::Sync(_) | Stmt::Raw(_) => {}
        }
        out.push(stmt);
    }
    Block(out)
}
