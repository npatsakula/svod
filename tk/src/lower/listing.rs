//! `kernel.tk`: one line per statement of a laid program, in the pre-order a
//! line table numbers them (line `i + 1` is statement `i` of
//! [`Program::walk`]). A line names what the statement does and the buffers
//! it moves, never a value's number, so two programs' lines compare by text;
//! a statement a schedule template placed carries its role.

use std::collections::HashMap;
use std::fmt::Write;

use crate::ir::*;

/// Where a statement runs relative to the pipeline a schedule template
/// expanded: before its trips, in its head, steady or drain trip loop, or
/// after them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Prologue,
    Head,
    Steady,
    Drain,
    Epilogue,
}

impl Role {
    pub fn name(self) -> &'static str {
        match self {
            Self::Prologue => "prologue",
            Self::Head => "head",
            Self::Steady => "steady",
            Self::Drain => "drain",
            Self::Epilogue => "epilogue",
        }
    }
}

impl From<Trip> for Role {
    fn from(trip: Trip) -> Self {
        match trip {
            Trip::Head => Self::Head,
            Trip::Steady => Self::Steady,
            Trip::Drain => Self::Drain,
        }
    }
}

/// Every statement's role, in [`Program::walk`] order.
pub fn roles<P: Phase>(prog: &Program<P>) -> Vec<Option<Role>> {
    let mut out = Vec::new();
    block_roles(&prog.body, None, &mut out);
    out
}

/// A loop a template placed runs its trip ([`Loop::trip`]); a statement
/// beside one is in the prologue before its pipeline's first trip, with the
/// trip before it between two trips of one pipeline (a crew's stagger), in
/// the epilogue past the last. Elsewhere a statement takes its parent's.
fn block_roles<P: Phase>(block: &Block<P>, inherited: Option<Role>, out: &mut Vec<Option<Role>>) {
    let trip = |s: &Stmt<P>| match s {
        Stmt::Loop(l) => l.trip,
        _ => None,
    };
    // The first trip after each statement.
    let mut next: Vec<Option<Trip>> =
        block.0.iter().rev().scan(None, |after, s| Some(std::mem::replace(after, trip(s).or(*after)))).collect();
    next.reverse();
    let mut before = None;
    for (stmt, next) in block.0.iter().zip(next) {
        let role = match (trip(stmt), before, next) {
            (Some(trip), ..) => {
                before = Some(trip);
                Some(trip.into())
            }
            // A pipeline's trips run head, steady, drain: a later one is the same pipeline's.
            (None, Some(before), Some(next)) if next > before => Some(before.into()),
            (None, _, Some(_)) => Some(Role::Prologue),
            (None, Some(_), None) => Some(Role::Epilogue),
            (None, None, None) => inherited,
        };
        out.push(role);
        stmt.blocks().for_each(|b| block_roles(b, role, out));
    }
}

/// The listing of `prog`: each statement indented by its depth.
pub fn listing<P: Phase>(prog: &Program<P>) -> String {
    let roles = roles(prog);
    // What a register value was read from: a parameter's or an allocation's name.
    let mut origin: HashMap<ValId, &str> = HashMap::new();
    let mut out = String::new();
    for ((depth, stmt), role) in prog.walk().zip(roles) {
        let what = describe(prog, stmt, &mut origin);
        let role = role.map_or(String::new(), |r| format!(" [{}]", r.name()));
        let _ = writeln!(out, "{}{what}{role}", "  ".repeat(depth));
    }
    out
}

fn name<P: Phase>(prog: &Program<P>, v: ValId) -> Option<&str> {
    match &prog.value(v).place {
        Place::Reg => None,
        Place::Smem { alloc, .. } => Some(&prog.smem[alloc.index()].name),
        Place::Global { param, .. } => Some(&prog.params[param.index()].name),
    }
}

fn describe<'p, P: Phase>(prog: &'p Program<P>, stmt: &Stmt<P>, origin: &mut HashMap<ValId, &'p str>) -> String {
    match stmt {
        Stmt::Let { dst, op } => {
            let read = |v: &ValId| origin.get(v).copied();
            let what = match op {
                TileOp::Mma { a, b, .. } => match (read(a), read(b)) {
                    (Some(a), Some(b)) => format!("mma {a}·{b}"),
                    _ => "mma".into(),
                },
                op => op_name(op),
            };
            if let TileOp::Move { src } = op
                && let Some(from) = read(src)
            {
                origin.insert(*dst, from);
            }
            what
        }
        Stmt::Copy { dst, src, mode, scope } => {
            let (d, s) = (prog.value(*dst).tier(), prog.value(*src).tier());
            let (dn, sn) = (name(prog, *dst).unwrap_or("?"), name(prog, *src).unwrap_or("?"));
            let staged = *mode == CopyMode::Staged;
            let what = match (s, d) {
                (Tier::Global, Tier::Smem) => {
                    let mode = match mode {
                        CopyMode::Sync => "",
                        CopyMode::Async => " async",
                        CopyMode::Staged => " staged",
                    };
                    format!("fill {sn} -> {dn}{mode}")
                }
                (Tier::Global, Tier::Reg) => {
                    origin.insert(*dst, sn);
                    if staged { format!("fill {sn} staged issue") } else { format!("load {sn}") }
                }
                (Tier::Reg, Tier::Smem) if staged => match origin.get(src) {
                    Some(from) => format!("fill {from} staged commit -> {dn}"),
                    None => format!("fill {dn} staged commit"),
                },
                (Tier::Reg, Tier::Smem) => format!("lds write {dn}"),
                (Tier::Smem, Tier::Reg) => {
                    origin.insert(*dst, sn);
                    format!("lds read {sn}")
                }
                (Tier::Reg, Tier::Global) => format!("store {dn}"),
                (Tier::Smem, Tier::Global) => format!("store {dn} from {sn}"),
                (Tier::Smem, Tier::Smem) => format!("lds copy {sn} -> {dn}"),
                (Tier::Global, Tier::Global) => format!("copy {sn} -> {dn}"),
                (Tier::Reg, Tier::Reg) => {
                    if let Some(from) = origin.get(src).copied() {
                        origin.insert(*dst, from);
                    }
                    "move".into()
                }
            };
            match scope {
                Scope::Block => what,
                Scope::Warp => format!("{what} (warp)"),
            }
        }
        Stmt::Loop(l) if l.unroll > 1 => format!("loop x{}", l.unroll),
        Stmt::Loop(_) => "loop".into(),
        Stmt::Pipeline(_) => "pipeline".into(),
        Stmt::Warps { .. } => "warps".into(),
        Stmt::If { otherwise, .. } if otherwise.0.is_empty() => "if".into(),
        Stmt::If { .. } => "if/else".into(),
        Stmt::Sync(sync) => match (*sync).into() {
            Sync::Barrier { scope: Scope::Block } => "barrier".into(),
            Sync::Barrier { scope: Scope::Warp } => "warp barrier".into(),
            Sync::Arrive => "arrive".into(),
            Sync::Await => "await".into(),
            Sync::CommitAsync => "commit async".into(),
            Sync::WaitAsync { pending } => format!("wait async {pending}"),
            Sync::Fence => "fence".into(),
            Sync::WaitProducts { pending } => format!("wait products {pending}"),
            Sync::Prio { level } => format!("prio {level}"),
        },
    }
}

/// A tile op's name, snake case, with the function it applies.
fn op_name<P: Phase>(op: &TileOp<P>) -> String {
    let lower = |d: &dyn std::fmt::Debug| snake(&format!("{d:?}"));
    match op {
        TileOp::Fill(_) => "const".into(),
        TileOp::Unary { f, .. } => lower(f),
        TileOp::Binary { f, .. } => lower(f),
        TileOp::Cast { to, .. } => format!("cast {}", format!("{to:?}").to_lowercase()),
        TileOp::Reduce { f, axis, .. } => format!("reduce {} {}", lower(f), lower(axis)),
        TileOp::Coord(axis) => format!("coord {}", lower(axis)),
        op => {
            let debug = format!("{op:?}");
            snake(debug.split([' ', '{', '(']).next().unwrap_or_default())
        }
    }
}

/// `CamelCase` as `camel_case`.
fn snake(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 4);
    for (i, c) in s.chars().enumerate() {
        if c.is_uppercase() && i > 0 {
            out.push('_');
        }
        out.extend(c.to_lowercase());
    }
    out
}
