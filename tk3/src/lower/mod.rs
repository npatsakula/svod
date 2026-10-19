//! From a tile program to a pre-linearized Svod program: expand the schedule
//! template, give every register tile a layout, then emit the instruction
//! list in program order.

use std::sync::Arc;

use snafu::{ResultExt, Snafu};
use svod_dtype::DeviceSpec;
use svod_ir::UOp;

use crate::atoms::Target;
use crate::ir::*;
use crate::layouts::{self, TileLayout, WarpGrid};
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
    let layouts = layouts::infer(&mut prog, &lowering.target, lowering.grid).context(LayoutSnafu)?;
    let program = emit::emit(&prog, &layouts, lowering, params, device)?;
    Ok(Lowered { program, tile: prog, layouts })
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
