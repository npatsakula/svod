//! Barriers from effects, placed before the schedule templates expand: a
//! shared tile written since the last barrier is fenced before it is read or
//! overwritten, a tile read since then is fenced before it is overwritten, and
//! a tile an asynchronous copy is filling is landed (its group committed and
//! waited for) and fenced before either. A [`Pipeline`] is one statement
//! here: its template lands and fences the ring around every consume, so the
//! pass fences only where the ring meets the code around the pipeline and
//! what a consume body does with other tiles. Every commit and wait the
//! templates do not emit is this pass's own.

use std::collections::HashSet;

use snafu::ensure;

use super::{DivergentBarrierSnafu, Result};
use crate::ir::*;

#[derive(Clone, Default)]
struct State {
    /// Allocations written since the last barrier.
    dirty: HashSet<SmemId>,
    /// Allocations read since the last barrier.
    read: HashSet<SmemId>,
    /// Allocations an asynchronous copy is filling, not yet landed.
    pending: HashSet<SmemId>,
}

/// What must come before an access, weakest first.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Need {
    Nothing,
    /// A barrier: another thread's write or read of the tile is ordered first.
    Fence,
    /// Commit, wait for every group, then a barrier.
    Land,
}

impl State {
    fn merge(&mut self, other: &State) {
        self.dirty.extend(other.dirty.iter().copied());
        self.read.extend(other.read.iter().copied());
        self.pending.extend(other.pending.iter().copied());
    }

    fn need(&self, alloc: Option<SmemId>, writes: bool) -> Need {
        match alloc {
            Some(a) if self.pending.contains(&a) => Need::Land,
            Some(a) if self.dirty.contains(&a) || (writes && self.read.contains(&a)) => Need::Fence,
            _ => Need::Nothing,
        }
    }

    /// Emit what `need` asks for into `out`, the barrier over `scope`. A
    /// barrier orders threads but lands no copy, so only the wait clears
    /// `pending`. Under a branch some of the block's warps skip
    /// (`divergent`), a block barrier would wait for them forever.
    fn meet(&mut self, need: Need, scope: Scope, divergent: bool, out: &mut Vec<Stmt>) -> Result<()> {
        if need == Need::Land {
            out.extend([Stmt::Sync(Sync::CommitAsync), Stmt::Sync(Sync::WaitAsync { pending: 0 })]);
            self.pending.clear();
        }
        if need >= Need::Fence {
            ensure!(!divergent || scope == Scope::Warp, DivergentBarrierSnafu { what: "a shared tile's fence" });
            out.push(Stmt::Sync(Sync::Barrier { scope }));
            self.fenced();
        }
        Ok(())
    }

    fn any(&self) -> bool {
        !(self.dirty.is_empty() && self.read.is_empty() && self.pending.is_empty())
    }

    fn fenced(&mut self) {
        self.dirty.clear();
        self.read.clear();
    }
}

pub fn insert_barriers(prog: &mut Program) -> Result<()> {
    let body = std::mem::take(&mut prog.body);
    let (body, _) = walk(prog, body, State::default(), Scope::Block, false)?;
    prog.body = body;
    Ok(())
}

fn alloc(prog: &Program, v: ValId) -> Option<SmemId> {
    match prog.value(v).place {
        Place::Smem { alloc, .. } => Some(alloc),
        _ => None,
    }
}

/// The shared allocations `block` copies into.
fn written(prog: &Program, block: &Block, into: &mut HashSet<SmemId>) {
    for stmt in &block.0 {
        match stmt {
            Stmt::Copy { dst, .. } => into.extend(alloc(prog, *dst)),
            Stmt::Loop(l) => written(prog, &l.body, into),
            Stmt::If { then, otherwise, .. } => {
                written(prog, then, into);
                written(prog, otherwise, into);
            }
            Stmt::Warps { body } => written(prog, body, into),
            Stmt::Pipeline(p) => written(prog, &p.produce.body, into),
            Stmt::Let { .. } | Stmt::Sync(_) | Stmt::Raw(_) => {}
        }
    }
}

/// The shared allocations `block` copies out of.
fn read(prog: &Program, block: &Block, into: &mut HashSet<SmemId>) {
    for stmt in &block.0 {
        match stmt {
            Stmt::Copy { src, .. } => into.extend(alloc(prog, *src)),
            Stmt::Loop(l) => read(prog, &l.body, into),
            Stmt::If { then, otherwise, .. } => {
                read(prog, then, into);
                read(prog, otherwise, into);
            }
            Stmt::Warps { body } => read(prog, body, into),
            Stmt::Pipeline(p) => {
                read(prog, &p.produce.body, into);
                read(prog, &p.consume.body, into);
            }
            Stmt::Let { .. } | Stmt::Sync(_) | Stmt::Raw(_) => {}
        }
    }
}

/// Rewrite `block` under the incoming `state`, its barriers over `scope`,
/// `divergent` under a branch on the warp; returns it with the state at its end.
fn walk(prog: &Program, block: Block, mut state: State, scope: Scope, divergent: bool) -> Result<(Block, State)> {
    let mut out = Vec::with_capacity(block.0.len());
    for stmt in block.0 {
        let stmt = match stmt {
            Stmt::Copy { dst, src, mode, scope: own } => {
                assert_eq!(own, scope, "a copy's scope is its enclosing statement's");
                let (wr, rd) = (alloc(prog, dst), alloc(prog, src));
                state.meet(state.need(rd, false).max(state.need(wr, true)), scope, divergent, &mut out)?;
                if let Some(a) = wr {
                    match mode {
                        CopyMode::Async => state.pending.insert(a),
                        CopyMode::Sync | CopyMode::Staged => state.dirty.insert(a),
                    };
                }
                state.read.extend(rd);
                Stmt::Copy { dst, src, mode, scope: own }
            }
            Stmt::Sync(sync) => {
                match sync {
                    Sync::Barrier { .. } => state.fenced(),
                    Sync::Fence | Sync::WaitProducts { .. } | Sync::Prio { .. } => {}
                    Sync::CommitAsync | Sync::WaitAsync { .. } => {
                        unreachable!("async groups are the templates' and this pass's own")
                    }
                }
                Stmt::Sync(sync)
            }
            Stmt::Loop(mut l) => {
                // The body's end state flows back to its start, and the state
                // before the loop flows past it: it may run no trip.
                let before = state.clone();
                let (_, back) = walk(prog, l.body.clone(), state.clone(), scope, divergent)?;
                state.merge(&back);
                let (body, end) = walk(prog, std::mem::take(&mut l.body), state, scope, divergent)?;
                l.body = body;
                state = end;
                state.merge(&before);
                Stmt::Loop(l)
            }
            Stmt::Pipeline(mut p) => {
                // The produce bodies overwrite the ring from the first step,
                // and copies still filling elsewhere are landed here rather
                // than left to the template's group count.
                let mut ring = HashSet::new();
                written(prog, &p.produce.body, &mut ring);
                let need = ring.iter().map(|&a| state.need(Some(a), true)).max().unwrap_or(Need::Nothing);
                let landing = if state.pending.is_empty() { Need::Nothing } else { Need::Land };
                state.meet(need.max(landing), scope, divergent, &mut out)?;
                // Landed copies are fenced by the template's barrier before
                // the first consume, as is every later consume's start, but
                // an async copy a consume issues is still pending at the next.
                let (_, back) = walk(prog, p.consume.body.clone(), State::default(), scope, divergent)?;
                let start = State { pending: back.pending, ..State::default() };
                let (body, mut end) = walk(prog, std::mem::take(&mut p.consume.body), start, scope, divergent)?;
                p.consume.body = body;
                // No barrier follows the last consume: its effects and its
                // ring reads stay open. With no step, the state passes through.
                end.read.extend(ring);
                state.merge(&end);
                Stmt::Pipeline(p)
            }
            Stmt::If { pred, then, otherwise } => {
                let divergent = divergent || (scope == Scope::Block && prog.varies_by_warp(pred));
                let (then, s1) = walk(prog, then, state.clone(), scope, divergent)?;
                let (otherwise, s2) = walk(prog, otherwise, state, scope, divergent)?;
                state = s1;
                state.merge(&s2);
                Stmt::If { pred, then, otherwise }
            }
            // The warps part on a clean slate (the block's open effects are
            // fenced first, for every warp), fence their own effects among
            // their own lanes, and what any of them wrote or read is the
            // block's open effect afterwards, a copy still in flight its
            // pending one.
            Stmt::Warps { body } => {
                let open = if state.pending.is_empty() { Need::Fence } else { Need::Land };
                if state.any() {
                    state.meet(open, scope, divergent, &mut out)?;
                }
                let (body, end) = walk(prog, body, State::default(), Scope::Warp, divergent)?;
                let mut touched = HashSet::new();
                written(prog, &body, &mut touched);
                read(prog, &body, &mut touched);
                state.dirty.extend(touched);
                state.pending.extend(end.pending);
                Stmt::Warps { body }
            }
            other => other,
        };
        out.push(stmt);
    }
    Ok((Block(out), state))
}
