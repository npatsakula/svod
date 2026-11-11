//! Every register value is read only where its definition dominates the
//! read: in the scope that defines it, after it, or a scope nested there. A
//! block's values leave it only as what it hands on (a loop's or pipeline's
//! next, a branch's merge), so a tile kept from a body and read past it is
//! refused here, not read undefined (or from the last trip) on the device.

use std::collections::{HashMap, HashSet};

use super::{EscapedSnafu, Result};
use crate::ir::*;

/// What a scope is, as an error names it.
type Scope = &'static str;

pub fn check(prog: &Program) -> Result<()> {
    let mut defined = HashMap::new();
    record(&prog.body, "the kernel", &mut defined);
    Walk { prog, defined, visible: HashSet::new() }.block(&prog.body, "the kernel", &[])
}

/// The scope that defines each register value.
fn record(block: &Block, scope: Scope, into: &mut HashMap<ValId, Scope>) {
    for stmt in &block.0 {
        for v in defines(stmt) {
            into.entry(v).or_insert(scope);
        }
        for (b, inner) in nested(stmt) {
            record(b, inner, into);
        }
    }
}

struct Walk<'p> {
    prog: &'p Program,
    defined: HashMap<ValId, Scope>,
    visible: HashSet<ValId>,
}

impl Walk<'_> {
    /// `block` in `scope`, `inherited` visible in it beside what the scopes
    /// around it define; what it defines is gone past it.
    fn block(&mut self, block: &Block, scope: Scope, inherited: &[ValId]) -> Result<()> {
        self.ending(block, scope, inherited, std::iter::empty())
    }

    fn stmt(&mut self, stmt: &Stmt, scope: Scope) -> Result<()> {
        match stmt {
            Stmt::Let { op, .. } => {
                if let TileOp::Splat(s) = op {
                    self.scalar(*s, scope)?;
                }
                op.srcs().try_for_each(|v| self.read(v, scope))
            }
            Stmt::Copy { src, .. } => self.read(*src, scope),
            Stmt::Loop(l) => {
                l.carried.iter().try_for_each(|c| self.read(c.init, scope))?;
                let phis: Vec<ValId> = l.carried.iter().map(|c| c.phi).collect();
                self.ending(&l.body, "a loop's body", &phis, l.carried.iter().map(|c| c.next))
            }
            Stmt::Pipeline(p) => {
                p.carried.iter().try_for_each(|c| self.read(c.init, scope))?;
                self.block(&p.produce.body, "a pipeline's produce", &[])?;
                let phis: Vec<ValId> = p.carried.iter().map(|c| c.phi).collect();
                self.ending(&p.consume.body, "a pipeline's consume", &phis, p.carried.iter().map(|c| c.next))
            }
            Stmt::Warps { body } => self.block(body, "the warps' body", &[]),
            Stmt::If { pred, then, otherwise, results } => {
                self.scalar(*pred, scope)?;
                self.ending(then, "a branch's then", &[], results.iter().map(|r| r.1))?;
                self.ending(otherwise, "a branch's otherwise", &[], results.iter().filter_map(|r| r.2))
            }
            Stmt::Sync(_) => Ok(()),
        }
    }

    /// `block` as [`Self::block`], `handed` read at its end: what it hands on.
    fn ending(
        &mut self,
        block: &Block,
        scope: Scope,
        inherited: &[ValId],
        mut handed: impl Iterator<Item = ValId>,
    ) -> Result<()> {
        let outer = self.visible.clone();
        self.visible.extend(inherited);
        for stmt in &block.0 {
            self.stmt(stmt, scope)?;
            self.visible.extend(defines(stmt));
        }
        let ends = handed.try_for_each(|v| self.read(v, scope));
        self.visible = outer;
        ends
    }

    /// The register values scalar `id` votes over ([`Scalar::Any`]).
    fn scalar(&mut self, id: ScalarId, scope: Scope) -> Result<()> {
        match self.prog.scalar(id) {
            Scalar::Any(v) => self.read(*v, scope),
            &Scalar::Bin(_, a, b) => {
                self.scalar(a, scope)?;
                self.scalar(b, scope)
            }
            _ => Ok(()),
        }
    }

    fn read(&self, v: ValId, used: Scope) -> Result<()> {
        if self.prog.value(v).place != Place::Reg || self.visible.contains(&v) {
            return Ok(());
        }
        let defined = self.defined.get(&v).copied().unwrap_or("no statement");
        EscapedSnafu { value: v.index(), defined, used }.fail()
    }
}

/// The blocks `stmt` nests and the scopes they are.
fn nested(stmt: &Stmt) -> Vec<(&Block, Scope)> {
    match stmt {
        Stmt::Loop(l) => vec![(&l.body, "a loop's body")],
        Stmt::Pipeline(p) => vec![(&p.produce.body, "a pipeline's produce"), (&p.consume.body, "a pipeline's consume")],
        Stmt::Warps { body } => vec![(body, "the warps' body")],
        Stmt::If { then, otherwise, .. } => vec![(then, "a branch's then"), (otherwise, "a branch's otherwise")],
        _ => vec![],
    }
}

/// The register values `stmt` leaves defined past itself.
fn defines(stmt: &Stmt) -> Vec<ValId> {
    match stmt {
        Stmt::Let { dst, .. } | Stmt::Copy { dst, .. } => vec![*dst],
        Stmt::If { results, .. } => results.iter().map(|r| r.0).collect(),
        Stmt::Loop(Loop { carried, .. }) | Stmt::Pipeline(Pipeline { carried, .. }) => {
            carried.iter().map(|c| c.phi).collect()
        }
        _ => vec![],
    }
}
