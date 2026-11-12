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

/// Every statement's role, in [`Program::walk`] order.
pub fn roles<P: Phase>(prog: &Program<P>) -> Vec<Option<Role>> {
    let mut out = Vec::new();
    block_roles(&prog.body, None, &mut out);
    out
}

/// A template expands a pipeline into the trips of one induction (head,
/// steady and drain loops): a block holding loops that share their `iv`
/// holds a pipeline's trips, and its statements take their roles from
/// where they stand among them. Elsewhere a statement takes its parent's.
fn block_roles<P: Phase>(block: &Block<P>, inherited: Option<Role>, out: &mut Vec<Option<Role>>) {
    let mut by_iv: HashMap<ScalarId, Vec<usize>> = HashMap::new();
    for (i, stmt) in block.0.iter().enumerate() {
        if let Stmt::Loop(l) = stmt {
            by_iv.entry(l.iv).or_default().push(i);
        }
    }
    let trips: HashMap<usize, Role> = by_iv
        .into_values()
        .filter(|loops| loops.len() > 1)
        .flat_map(|loops| {
            loops.into_iter().enumerate().map(|(k, i)| (i, [Role::Head, Role::Steady, Role::Drain][k.min(2)]))
        })
        .collect();
    let (first, last) = (trips.keys().min().copied(), trips.keys().max().copied());
    // Between two trip loops (a crew's stagger), the trip before's.
    let mut between = inherited;
    for (i, stmt) in block.0.iter().enumerate() {
        let role = match (trips.get(&i), first, last) {
            (Some(&trip), ..) => {
                between = Some(trip);
                between
            }
            (None, Some(first), _) if i < first => Some(Role::Prologue),
            (None, _, Some(last)) if i > last => Some(Role::Epilogue),
            (None, Some(_), _) => between,
            (None, None, _) => inherited,
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
        Stmt::Sync(sync) => match sync {
            Sync::Barrier { scope: Scope::Block } => "barrier".into(),
            Sync::Barrier { scope: Scope::Warp } => "warp barrier".into(),
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
