//! Schedule templates: how a [`Pipeline`] statement runs on a target. The
//! template owns every barrier, wait and ring-slot computation; the author's
//! `produce`/`consume` bodies are placed, never edited.

use crate::ir::*;

/// How a uniform (every warp does everything) schedule moves data in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Prefetch {
    /// Asynchronous global→shared copies (`cp.async`), `stages - 1` steps ahead.
    CpAsync,
    /// Global→register loads issued before the compute of a step and written
    /// to shared memory after it, one step ahead.
    RegisterStaged,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Schedule {
    /// Every warp loads and computes; `unroll` copies the loop body once per
    /// ring slot so slot addressing folds to constants.
    Uniform { prefetch: Prefetch, unroll: bool },
}

/// Replace every [`Stmt::Pipeline`] of `prog` by loops, branches and explicit
/// synchronization according to `schedule`.
pub fn expand(prog: &mut Program, schedule: Schedule) {
    let body = std::mem::take(&mut prog.body);
    prog.body = Expander { prog, schedule }.block(body);
}

struct Expander<'p> {
    prog: &'p mut Program,
    schedule: Schedule,
}

impl Expander<'_> {
    fn scalar(&mut self, s: Scalar) -> ScalarId {
        self.prog.scalars.push(s);
        ScalarId(self.prog.scalars.len() as u32 - 1)
    }

    fn c(&mut self, v: i64) -> ScalarId {
        self.scalar(Scalar::Const(v))
    }

    fn bin(&mut self, op: BinOp, a: ScalarId, b: ScalarId) -> ScalarId {
        self.scalar(Scalar::Bin(op, a, b))
    }

    /// Redefine a stage's induction scalar as an expression.
    fn define(&mut self, id: ScalarId, s: Scalar) {
        debug_assert_eq!(self.prog.scalars[id.index()], Scalar::Induction, "a stage parameter");
        self.prog.scalars[id.index()] = s;
    }

    fn block(&mut self, block: Block) -> Block {
        Block(block.0.into_iter().flat_map(|s| self.stmt(s)).collect())
    }

    fn stmt(&mut self, stmt: Stmt) -> Vec<Stmt> {
        match stmt {
            Stmt::Pipeline(p) => self.pipeline(p),
            Stmt::Loop(mut l) => {
                l.body = self.block(l.body);
                vec![Stmt::Loop(l)]
            }
            Stmt::Role { role, body } => vec![Stmt::Role { role, body: self.block(body) }],
            Stmt::If { pred, then, otherwise } => {
                vec![Stmt::If { pred, then: self.block(then), otherwise: self.block(otherwise) }]
            }
            other => vec![other],
        }
    }

    fn pipeline(&mut self, p: Pipeline) -> Vec<Stmt> {
        let Pipeline { extent, stages, carried, produce, consume } = p;
        let produce = Stage { body: self.block(produce.body), ..produce };
        let consume = Stage { body: self.block(consume.body), ..consume };
        let Schedule::Uniform { prefetch, unroll } = self.schedule;
        let mut out = match prefetch {
            Prefetch::CpAsync if stages >= 2 => self.cp_async(extent, stages, carried, produce, consume),
            Prefetch::CpAsync => self.single_stage(extent, carried, produce, consume),
            Prefetch::RegisterStaged => self.register_staged(extent, stages, carried, produce, consume),
        };
        if !unroll {
            for stmt in &mut out {
                if let Stmt::Loop(l) = stmt {
                    l.unroll = 1;
                }
            }
        }
        out
    }

    fn if_(pred: ScalarId, then: Vec<Stmt>) -> Stmt {
        Stmt::If { pred, then: Block(then), otherwise: Block::default() }
    }

    /// `for i in 0..extent + stages - 1`: consume step `i - (stages - 1)` once
    /// its copies landed and everyone can see them, then issue step `i`'s
    /// copies into the slot consumed one iteration ago, and commit them as one
    /// group (an empty one past the end, so the group count stays uniform).
    fn cp_async(
        &mut self,
        extent: ScalarId,
        stages: usize,
        carried: Vec<Carried>,
        produce: Stage,
        consume: Stage,
    ) -> Vec<Stmt> {
        let iv = self.scalar(Scalar::Induction);
        let ahead = self.c(stages as i64 - 1);
        let total = self.bin(BinOp::Add, extent, ahead);
        let nstages = self.c(stages as i64);
        let zero = self.c(0);
        let start = zero;
        self.define(produce.step, Scalar::Bin(BinOp::Add, iv, zero));
        self.define(produce.slot, Scalar::Bin(BinOp::Rem, produce.step, nstages));
        self.define(consume.step, Scalar::Bin(BinOp::Sub, iv, ahead));
        self.define(consume.slot, Scalar::Bin(BinOp::Rem, consume.step, nstages));
        let consuming = self.bin(BinOp::Le, ahead, iv);
        let producing = self.bin(BinOp::Lt, iv, extent);
        let mut consume_body =
            vec![Stmt::Sync(Sync::WaitAsync { pending: stages as u32 - 2 }), Stmt::Sync(Sync::Barrier { role: None })];
        consume_body.extend(consume.body.0);
        let body = vec![
            Self::if_(consuming, consume_body),
            Self::if_(producing, produce.body.0),
            Stmt::Sync(Sync::CommitAsync),
        ];
        vec![Stmt::Loop(Loop { iv, start, extent: total, carried, body: Block(body), unroll: stages as u32 })]
    }

    /// One slot: copy, wait, consume, and fence before the slot is refilled.
    fn single_stage(&mut self, extent: ScalarId, carried: Vec<Carried>, produce: Stage, consume: Stage) -> Vec<Stmt> {
        let iv = self.scalar(Scalar::Induction);
        let zero = self.c(0);
        for stage in [&produce, &consume] {
            self.define(stage.step, Scalar::Bin(BinOp::Add, iv, zero));
            self.define(stage.slot, Scalar::Const(0));
        }
        let start = zero;
        let mut body = produce.body.0;
        body.extend([
            Stmt::Sync(Sync::CommitAsync),
            Stmt::Sync(Sync::WaitAsync { pending: 0 }),
            Stmt::Sync(Sync::Barrier { role: None }),
        ]);
        body.extend(consume.body.0);
        body.push(Stmt::Sync(Sync::Barrier { role: None }));
        vec![Stmt::Loop(Loop { iv, start, extent, carried, body: Block(body), unroll: 1 })]
    }

    /// Two slots, one step ahead: a trip loads step `i` into registers,
    /// consumes step `i - 1` from the other slot, writes the registers to
    /// this step's slot and fences. The first trip only loads and the last
    /// only consumes, so each is peeled into a one-trip loop and the steady
    /// loop has no branches: a trip is one basic block, which a machine
    /// scheduler may reorder freely, and the [`Sync::Fence`] between the
    /// products and the commit is what keeps the two slots' order.
    fn register_staged(
        &mut self,
        extent: ScalarId,
        stages: usize,
        carried: Vec<Carried>,
        produce: Stage,
        consume: Stage,
    ) -> Vec<Stmt> {
        assert_eq!(stages, 2, "register staging runs one step ahead over two slots");
        let iv = self.scalar(Scalar::Induction);
        let one = self.c(1);
        let two = self.c(2);
        let zero = self.c(0);
        // An empty pipeline runs no trip at all (a range never counts down).
        let nonempty = self.bin(BinOp::Lt, zero, extent);
        let steady = self.bin(BinOp::Sub, extent, nonempty);
        self.define(produce.step, Scalar::Bin(BinOp::Add, iv, zero));
        self.define(produce.slot, Scalar::Bin(BinOp::Rem, produce.step, two));
        self.define(consume.step, Scalar::Bin(BinOp::Sub, iv, one));
        self.define(consume.slot, Scalar::Bin(BinOp::Rem, consume.step, two));
        // Split every global→shared copy into its load and its store halves.
        let (mut issue, mut commit) = (vec![], vec![]);
        for stmt in produce.body.0 {
            match stmt {
                Stmt::Copy { dst, src, .. } if self.prog.value(src).tier() == Tier::Global => {
                    let Value { dtype, shape, .. } = self.prog.value(src).clone();
                    self.prog.values.push(Value { dtype, shape, place: Place::Reg });
                    let tmp = ValId(self.prog.values.len() as u32 - 1);
                    issue.push(Stmt::Copy { dst: tmp, src, mode: CopyMode::Staged });
                    commit.push(Stmt::Copy { dst, src: tmp, mode: CopyMode::Staged });
                }
                other => issue.push(other),
            }
        }
        let barrier = Stmt::Sync(Sync::Barrier { role: None });
        let head: Vec<Stmt> = issue.iter().chain(&commit).cloned().chain([barrier.clone()]).collect();
        let fenced = [Stmt::Sync(Sync::Fence)].into_iter().chain(commit).chain([barrier]);
        let body: Vec<Stmt> = issue.into_iter().chain(consume.body.0.iter().cloned()).chain(fenced).collect();
        // The drain's carried registers hold what the steady loop left.
        let drained: Vec<Carried> = carried.iter().map(|c| Carried { init: c.phi, ..*c }).collect();
        let trips = |start, extent, carried, body: Vec<Stmt>, unroll| {
            Stmt::Loop(Loop { iv, start, extent, carried, body: Block(body), unroll })
        };
        vec![
            trips(zero, nonempty, vec![], head, 1),
            trips(one, steady, carried, body, 2),
            trips(extent, nonempty, drained, consume.body.0, 1),
        ]
    }
}
