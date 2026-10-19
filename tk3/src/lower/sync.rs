//! Barriers from effects: a shared tile written by a synchronous copy is
//! fenced before another thread may read it, a tile read since the
//! last fence is fenced before it is overwritten, and a tile an asynchronous
//! copy is filling is waited for and fenced before it is read. A schedule
//! template fences its own ring, so this pass adds nothing inside one; it
//! covers what an author writes outside the templates.

use std::collections::HashSet;

use crate::ir::*;

#[derive(Clone, Default)]
struct State {
    /// Allocations written by a synchronous copy since the last barrier.
    dirty: HashSet<SmemId>,
    /// Allocations read since the last barrier.
    read: HashSet<SmemId>,
    /// Allocations an asynchronous copy is filling, not yet waited for.
    pending: HashSet<SmemId>,
}

impl State {
    fn merge(&mut self, other: &State) {
        self.dirty.extend(other.dirty.iter().copied());
        self.read.extend(other.read.iter().copied());
        self.pending.extend(other.pending.iter().copied());
    }
}

pub fn insert_barriers(prog: &mut Program) {
    let body = std::mem::take(&mut prog.body);
    let (body, _) = walk(prog, body, State::default());
    prog.body = body;
}

fn alloc(prog: &Program, v: ValId) -> Option<SmemId> {
    match prog.value(v).place {
        Place::Smem { alloc, .. } => Some(alloc),
        _ => None,
    }
}

/// Rewrite `block` under the incoming `state`; returns it with the state at
/// its end.
fn walk(prog: &Program, block: Block, mut state: State) -> (Block, State) {
    let mut out = Vec::with_capacity(block.0.len());
    for stmt in block.0 {
        let stmt = match stmt {
            Stmt::Copy { dst, src, mode } => {
                let (wr, rd) = (alloc(prog, dst), alloc(prog, src));
                // Reading, or synchronously overwriting, a tile still filling.
                let unfinished = rd.is_some_and(|a| state.pending.contains(&a))
                    || (mode == CopyMode::Sync && wr.is_some_and(|a| state.pending.contains(&a)));
                if unfinished {
                    // A wait counts committed groups: close the open one first.
                    out.extend([Stmt::Sync(Sync::CommitAsync), Stmt::Sync(Sync::WaitAsync { pending: 0 })]);
                }
                let fence = unfinished
                    || rd.is_some_and(|a| state.dirty.contains(&a))
                    || (mode == CopyMode::Sync && wr.is_some_and(|a| state.read.contains(&a)));
                if fence {
                    out.push(Stmt::Sync(Sync::Barrier { role: None }));
                    state = State::default();
                }
                match (wr, mode) {
                    (Some(a), CopyMode::Sync) => {
                        state.dirty.insert(a);
                    }
                    (Some(a), CopyMode::Async) => {
                        state.pending.insert(a);
                    }
                    // A staged copy belongs to the register-staged template, which fences it.
                    (_, CopyMode::Staged) | (None, _) => {}
                }
                if let Some(a) = rd {
                    state.read.insert(a);
                }
                Stmt::Copy { dst, src, mode }
            }
            Stmt::Sync(sync) => {
                match sync {
                    Sync::Barrier { .. } => state = State::default(),
                    // Waited copies have landed for the thread that issued them.
                    Sync::WaitAsync { .. } => state.dirty.extend(state.pending.drain()),
                    Sync::CommitAsync | Sync::Fence => {}
                }
                Stmt::Sync(sync)
            }
            Stmt::Loop(mut l) => {
                // The body's end state flows back to its start, and the state
                // before the loop flows past it: it may run no trip.
                let before = state.clone();
                let (_, back) = walk(prog, l.body.clone(), state.clone());
                state.merge(&back);
                let (body, end) = walk(prog, std::mem::take(&mut l.body), state);
                l.body = body;
                state = end;
                state.merge(&before);
                Stmt::Loop(l)
            }
            Stmt::If { pred, then, otherwise } => {
                let (then, s1) = walk(prog, then, state.clone());
                let (otherwise, s2) = walk(prog, otherwise, state);
                state = s1;
                state.merge(&s2);
                Stmt::If { pred, then, otherwise }
            }
            Stmt::Role { role, body } => {
                let (body, end) = walk(prog, body, state);
                state = end;
                Stmt::Role { role, body }
            }
            Stmt::Pipeline(_) => unreachable!("pipelines are expanded before the sync pass"),
            other => other,
        };
        out.push(stmt);
    }
    (Block(out), state)
}
