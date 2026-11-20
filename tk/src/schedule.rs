//! Schedule templates: how a [`Pipeline`] statement runs on a target. The
//! template owns every barrier, wait and ring-slot computation inside the
//! ring (the sync pass fences only where the ring meets the code around it);
//! the author's `produce`/`consume` bodies are placed, never edited.

use crate::atoms::{Core, Fill, GroupMmaAtom, SyncCaps};
use crate::ir::*;
use crate::lower::bulk;

/// The highest issue priority a wave takes (`s_setprio 3`), a crew's over
/// its products.
const RAISED: u8 = 3;

/// Replace every [`Stmt::Pipeline`] of `prog` by loops, branches and explicit
/// synchronization: a ring of one crew runs every warp alike, its fills as
/// the target's `fill` moves data in, its loop unrolled over the slots as
/// the pipeline says; two crews run a cluster apart over one register-staged
/// slot ([`Expander::ping_pong`]). On the warpgroup `core` a `cp.async` ring
/// of three slots or more whose consume only chains its products leaves them
/// in flight over the next trip's fill: the ring refills the slot consumed
/// two trips ago, and the template waits for all but the youngest chain
/// after every consume that places no wait of its own, and for every chain
/// after the loop ([`Sync::WaitProducts`]). Where a block barrier's halves
/// part (`split`), a register-staged ring over two slots arrives as its
/// trip stops touching shared memory and waits at the trip's end
/// ([`Expander::register_staged`]). Each loop a template places is tagged
/// with the trip it runs ([`Loop::trip`]). The result's phase has no
/// pipeline statement. A ring filled in bulk ([`Schedule::Bulk`]) fills its
/// slots so where the target has slot barriers ([`Expander::bulk`]), and is
/// the plain ring elsewhere.
pub fn expand(prog: Program, fill: Fill, core: Core, sync: SyncCaps) -> Program<Expanded> {
    let (mut out, body) = prog.rephase(|_, ()| (), ());
    out.body = Expander { prog: &mut out, fill, core, sync, scope: Scope::Block }.block(body);
    out
}

/// A ring's fill in bulk ([`Expander::bulk`]).
struct BulkFill {
    /// The barrier of the slot a trip produces into.
    bar: SlotBar,
    /// The parity of the phase a trip's consume waits for.
    phase: ScalarId,
    /// The bytes a trip's bulk copies move.
    bytes: u32,
    /// The produce's statements that are those copies.
    moved: Vec<usize>,
    /// Every slot's barrier made, ahead of the ring.
    inits: Vec<Stmt<Expanded>>,
    /// Where the ring's threads leave their slots ([`Expander::left`]): the
    /// barrier of the slot a consume read, the one a produce refills and
    /// the parity it waits for.
    left: Option<(SlotBar, SlotBar, ScalarId)>,
}

struct Expander<'p> {
    prog: &'p mut Program<Expanded>,
    fill: Fill,
    core: Core,
    sync: SyncCaps,
    /// Whose barriers a pipeline's template emits: the block's, or one
    /// warp's own inside a [`Stmt::Warps`].
    scope: Scope,
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

    fn block(&mut self, block: Block) -> Block<Expanded> {
        Block(block.0.into_iter().flat_map(|s| self.stmt(s)).collect())
    }

    /// The statement with its nested blocks expanded; a pipeline becomes
    /// its template's statements.
    fn stmt(&mut self, stmt: Stmt) -> Vec<Stmt<Expanded>> {
        let scope = if matches!(stmt, Stmt::Warps { .. }) { Scope::Warp } else { self.scope };
        let outer = std::mem::replace(&mut self.scope, scope);
        let mapped = stmt.map(&mut |b| self.block(b), &mut |_, op| op.map(|orient| orient, |()| ()));
        self.scope = outer;
        match mapped {
            Ok(stmt) => vec![stmt],
            Err(p) => self.pipeline(p),
        }
    }

    fn barrier(&self) -> Stmt<Expanded> {
        Stmt::Sync(Sync::Barrier { scope: self.scope })
    }

    fn pipeline(&mut self, p: Pipeline) -> Vec<Stmt<Expanded>> {
        let Pipeline { extent, schedule, carried, produce, consume } = p;
        let produce = Stage { step: produce.step, slot: produce.slot, body: self.block(produce.body) };
        let consume = Stage { step: consume.step, slot: consume.slot, body: self.block(consume.body) };
        let Some((stages, unroll, in_bulk)) = schedule.as_ring() else {
            let crew = self.prog.warps / schedule.crews();
            return self.ping_pong(extent, carried, produce, consume, crew);
        };
        let mut out = match self.fill {
            Fill::CpAsync if in_flight(self.fill, self.core, stages) && chains_only(&consume.body) => {
                let fill = if in_bulk { self.bulk(stages, produce.slot, consume.step, &produce.body) } else { None };
                self.cp_async_products(extent, stages, carried, produce, consume, fill)
            }
            Fill::CpAsync if stages >= 2 => {
                let fill = if in_bulk { self.bulk(stages, produce.slot, consume.step, &produce.body) } else { None };
                let fill = fill.map(|fill| self.left(stages, fill, &produce, &consume));
                self.cp_async(extent, stages, carried, produce, consume, fill)
            }
            Fill::CpAsync => self.single_stage(extent, carried, produce, consume),
            Fill::Staged => self.register_staged(extent, stages, carried, produce, consume),
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

    fn if_(pred: ScalarId, then: Vec<Stmt<Expanded>>) -> Stmt<Expanded> {
        Stmt::If { pred, then: Block(then), otherwise: Block::default(), results: vec![] }
    }

    /// `for i in 0..extent + stages - 1`: consume step `i - (stages - 1)` once
    /// its copies landed and everyone can see them, then issue step `i`'s
    /// copies into the slot consumed one iteration ago, and commit them as one
    /// group (an empty one past the end, so the group count stays uniform).
    /// Under a `bulk` fill a consume waits on its slot's barrier, and where
    /// every copy is the bulk fill's the block barrier goes: each warp
    /// leaves the slot it read and the fill of a slot waits for every
    /// warp to have left it ([`Self::left`]), so the block's warpgroups
    /// run their trips apart, one's products over the other's softmax.
    fn cp_async(
        &mut self,
        extent: ScalarId,
        stages: usize,
        carried: Vec<Carried>,
        mut produce: Stage<Expanded>,
        consume: Stage<Expanded>,
        bulk: Option<BulkFill>,
    ) -> Vec<Stmt<Expanded>> {
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
        let (mut consume_body, mut head, mut chunks, mut leave) = (vec![], vec![], true, None);
        if let Some(BulkFill { bar, phase, bytes, moved, inits, left }) = bulk {
            consume_body.push(Stmt::Sync(Sync::SlotWait { bar: SlotBar { slot: consume.slot, ..bar }, phase }));
            chunks = left.is_none();
            for at in moved {
                let Stmt::Copy { mode, .. } = &mut produce.body.0[at] else { unreachable!("a bulk fill is a copy") };
                *mode = CopyMode::Bulk { bar };
            }
            produce.body.0.insert(0, Stmt::Sync(Sync::SlotArrive { bar, bytes }));
            if let Some((read, refilled, phase)) = left {
                produce.body.0.insert(0, Stmt::Sync(Sync::SlotVacant { bar: refilled, phase }));
                leave = Some(Stmt::Sync(Sync::SlotLeave { bar: read }));
            }
            head = inits;
            head.push(self.barrier());
        }
        if chunks {
            consume_body.extend([Stmt::Sync(Sync::WaitAsync { pending: stages as u32 - 2 }), self.barrier()]);
        }
        consume_body.extend(consume.body.0);
        consume_body.extend(leave);
        let mut body = vec![Self::if_(consuming, consume_body), Self::if_(producing, produce.body.0)];
        if chunks {
            body.push(Stmt::Sync(Sync::CommitAsync));
        }
        head.push(Stmt::Loop(Loop {
            iv,
            start,
            extent: total,
            carried,
            body: Block(body),
            unroll: stages as u32,
            trip: Some(Trip::Steady),
        }));
        head
    }

    /// `fill` with its ring's slots left rather than fenced, where every
    /// copy of `produce` is the bulk fill's and the ring is run by every
    /// warp of a block of several warpgroups (one's blocks run shorter
    /// leaving their slots, 18.5 µs against 20.5 traced, but fewer of them
    /// at once and the kernel longer: 0.341 ms against 0.306 over 64×32
    /// tiles of a causal 64×16×512×128, H200, 2026-11-20): a second barrier
    /// a slot, behind the fills' in
    /// their allocation, which every warp arrives at once it has read the
    /// slot's step. The fill of step `p` waits for the phase of step
    /// `p - stages`, of parity `p / stages + 1`: the one before a slot's
    /// first is complete from the start.
    fn left(
        &mut self,
        stages: usize,
        mut fill: BulkFill,
        produce: &Stage<Expanded>,
        consume: &Stage<Expanded>,
    ) -> BulkFill {
        let warps = self.prog.warps;
        let apart = self.core == Core::Group && warps > GroupMmaAtom::WARPS && warps == self.prog.warp_grid.count();
        if fill.moved.len() < produce.body.0.len() || !apart {
            return fill;
        }
        let bars = fill.bar.bars;
        self.prog.smem[bars.index()].elems = 2 * stages;
        let (nstages, one, two) = (self.c(stages as i64), self.c(1), self.c(2));
        let behind = |it: &mut Self, slot: ScalarId| SlotBar { bars, slot: it.bin(BinOp::Add, slot, nstages) };
        let (read, refilled) = (behind(self, consume.slot), behind(self, produce.slot));
        fill.inits.extend(
            (stages..2 * stages).map(|s| Stmt::Sync(Sync::SlotOpen { bar: SlotBar { bars, slot: self.c(s as i64) } })),
        );
        let fills = self.bin(BinOp::Div, produce.step, nstages);
        let after = self.bin(BinOp::Add, fills, one);
        fill.left = Some((read, refilled, self.bin(BinOp::Rem, after, two)));
        fill
    }

    /// [`Self::cp_async`] with the consume's products left in flight: step
    /// `i - 1` is issued at iteration `i` into the slot consumed two
    /// iterations ago (the slot of the chain still running was consumed
    /// one ago), so `stages - 3` groups stay pending at a consume. Every
    /// thread waits for all but its youngest chain at the end of each
    /// consume, and the barrier before the next consume then holds every
    /// warpgroup's chain over a slot complete before that slot is refilled.
    fn cp_async_products(
        &mut self,
        extent: ScalarId,
        stages: usize,
        carried: Vec<Carried>,
        mut produce: Stage<Expanded>,
        consume: Stage<Expanded>,
        bulk: Option<BulkFill>,
    ) -> Vec<Stmt<Expanded>> {
        assert!(stages >= 3, "a chain in flight takes a slot of its own");
        let iv = self.scalar(Scalar::Induction);
        let one = self.c(1);
        let ahead = self.c(stages as i64 - 1);
        let total = self.bin(BinOp::Add, extent, ahead);
        let nstages = self.c(stages as i64);
        let start = self.c(0);
        self.define(produce.step, Scalar::Bin(BinOp::Sub, iv, one));
        self.define(produce.slot, Scalar::Bin(BinOp::Rem, produce.step, nstages));
        self.define(consume.step, Scalar::Bin(BinOp::Sub, iv, ahead));
        self.define(consume.slot, Scalar::Bin(BinOp::Rem, consume.step, nstages));
        let consuming = self.bin(BinOp::Le, ahead, iv);
        let started = self.bin(BinOp::Le, one, iv);
        let before_end = self.bin(BinOp::Lt, produce.step, extent);
        let producing = self.bin(BinOp::And, started, before_end);
        // A consume that places its own waits keeps them: a wait of the
        // template's after them would close a pipeline stage over the
        // accumulators the consume rewrote since, and `ptxas` would
        // serialize every chain of the kernel.
        let waits = consume.body.0.iter().any(|s| matches!(s, Stmt::Sync(Sync::WaitProducts { .. })));
        // A slot's fill is waited for on its barrier where a bulk copy moved
        // it, and as the thread's copy groups where its chunks did.
        let mut consume_body = vec![];
        let (mut head, mut chunks) = (vec![], true);
        if let Some(BulkFill { bar, phase, bytes, moved, inits, .. }) = bulk {
            consume_body.push(Stmt::Sync(Sync::SlotWait { bar: SlotBar { slot: consume.slot, ..bar }, phase }));
            chunks = moved.len() < produce.body.0.len();
            for at in moved {
                let Stmt::Copy { mode, .. } = &mut produce.body.0[at] else { unreachable!("a bulk fill is a copy") };
                *mode = CopyMode::Bulk { bar };
            }
            produce.body.0.insert(0, Stmt::Sync(Sync::SlotArrive { bar, bytes }));
            head = inits;
            head.push(self.barrier());
        }
        if chunks {
            consume_body.push(Stmt::Sync(Sync::WaitAsync { pending: stages as u32 - 3 }));
        }
        consume_body.push(self.barrier());
        consume_body.extend(consume.body.0);
        if !waits {
            consume_body.push(Stmt::Sync(Sync::WaitProducts { pending: 1 }));
        }
        let mut body = vec![Self::if_(consuming, consume_body), Self::if_(producing, produce.body.0)];
        if chunks {
            body.push(Stmt::Sync(Sync::CommitAsync));
        }
        head.extend([
            Stmt::Loop(Loop {
                iv,
                start,
                extent: total,
                carried,
                body: Block(body),
                unroll: stages as u32,
                trip: Some(Trip::Steady),
            }),
            Stmt::Sync(Sync::WaitProducts { pending: 0 }),
        ]);
        head
    }

    /// The bulk fill of a ring of `stages` slots whose trips produce
    /// `produce` into slot `slot`: the block's copies one bulk copy moves
    /// each ([`bulk::frame`]), completing on the slot's barrier of a new
    /// allocation, whose phases alternate with the slot's fills, so the
    /// consume of step `step` waits for parity `step / stages`. `None` where
    /// the target has no slot barriers or no copy is so moved.
    fn bulk(&mut self, stages: usize, slot: ScalarId, step: ScalarId, produce: &Block<Expanded>) -> Option<BulkFill> {
        if !self.sync.slot_barrier || self.scope != Scope::Block {
            return None;
        }
        let frames = produce.0.iter().enumerate().filter_map(|(at, stmt)| match stmt {
            Stmt::Copy { dst, src, mode: CopyMode::Async, .. } => {
                Some((at, bulk::frame(self.prog, self.core, *dst, *src).ok()?.bytes()))
            }
            _ => None,
        });
        let (moved, bytes): (Vec<usize>, Vec<u32>) = frames.unzip();
        if moved.is_empty() {
            return None;
        }
        self.prog.smem.push(SmemAlloc {
            name: "full".into(),
            dtype: svod_dtype::ScalarDType::UInt64,
            elems: stages,
            storage: (),
        });
        let bars = SmemId(self.prog.smem.len() as u32 - 1);
        let inits =
            (0..stages).map(|s| Stmt::Sync(Sync::SlotInit { bar: SlotBar { bars, slot: self.c(s as i64) } })).collect();
        let (nstages, two) = (self.c(stages as i64), self.c(2));
        let fills = self.bin(BinOp::Div, step, nstages);
        let phase = self.bin(BinOp::Rem, fills, two);
        Some(BulkFill { bar: SlotBar { bars, slot }, phase, bytes: bytes.iter().sum(), moved, inits, left: None })
    }

    /// One slot: copy, wait, consume, and fence before the slot is refilled.
    fn single_stage(
        &mut self,
        extent: ScalarId,
        carried: Vec<Carried>,
        produce: Stage<Expanded>,
        consume: Stage<Expanded>,
    ) -> Vec<Stmt<Expanded>> {
        let iv = self.scalar(Scalar::Induction);
        let zero = self.c(0);
        for stage in [&produce, &consume] {
            self.define(stage.step, Scalar::Bin(BinOp::Add, iv, zero));
            self.define(stage.slot, Scalar::Const(0));
        }
        let start = zero;
        let mut body = produce.body.0;
        body.extend([Stmt::Sync(Sync::CommitAsync), Stmt::Sync(Sync::WaitAsync { pending: 0 }), self.barrier()]);
        body.extend(consume.body.0);
        body.push(self.barrier());
        vec![Stmt::Loop(Loop { iv, start, extent, carried, body: Block(body), unroll: 1, trip: Some(Trip::Steady) })]
    }

    /// Every global→shared copy of a produce body split into its load half
    /// (into fresh staging registers) and its store half; what is not such a
    /// copy stays with the loads.
    fn staged_halves(&mut self, body: Block<Expanded>) -> (Vec<Stmt<Expanded>>, Vec<Stmt<Expanded>>) {
        let (mut issue, mut commit) = (vec![], vec![]);
        for stmt in body.0 {
            match stmt {
                Stmt::Copy { dst, src, scope, .. } if self.prog.value(src).tier() == Tier::Global => {
                    let Value { dtype, shape, .. } = self.prog.value(src).clone();
                    self.prog.values.push(Value { dtype, shape, place: Place::Reg });
                    let tmp = ValId(self.prog.values.len() as u32 - 1);
                    issue.push(Stmt::Copy { dst: tmp, src, mode: CopyMode::Staged, scope });
                    commit.push(Stmt::Copy { dst, src: tmp, mode: CopyMode::Staged, scope });
                }
                other => issue.push(other),
            }
        }
        (issue, commit)
    }

    /// One slot, two crews of `crew` warps a cluster apart. The consume body is cut at its
    /// products into clusters: the statements before a product (its fragment
    /// reads) make a load cluster, the product an MFMA cluster, each closed
    /// by a block barrier, so a barrier meets a crew's cluster `k` with the
    /// other crew's `k - 1` and a crew's products, fed from registers, run
    /// while the other crew loads. A trip issues the next step's global
    /// loads at the head of the load clusters before the last, an operand's
    /// all in one (its k-slices of a row meet the row's cache line at once;
    /// spread over clusters, each slice fetches the line again) and the
    /// operands spread apart, reads the last product's fragments a cluster
    /// early, and commits the staged registers in issue order in the last
    /// load cluster, when every warp has read its fragments of the step (the
    /// crew behind is in a product). The head trip fills step 0; the
    /// second crew then waits one barrier more and the first one more after
    /// the loop, which puts the crews a cluster apart. A product runs at
    /// raised issue priority. The last trip stages its own step again: its
    /// commit lands the bytes already there.
    fn ping_pong(
        &mut self,
        extent: ScalarId,
        carried: Vec<Carried>,
        produce: Stage<Expanded>,
        consume: Stage<Expanded>,
        crew: u32,
    ) -> Vec<Stmt<Expanded>> {
        let iv = self.scalar(Scalar::Induction);
        let (zero, one, before) = (self.c(0), self.c(1), self.c(-1));
        let nonempty = self.bin(BinOp::Lt, zero, extent);
        let last = self.bin(BinOp::Sub, extent, one);
        let next = self.bin(BinOp::Add, iv, one);
        self.define(produce.step, Scalar::Bin(BinOp::Min, next, last));
        self.define(produce.slot, Scalar::Const(0));
        self.define(consume.step, Scalar::Bin(BinOp::Add, iv, zero));
        self.define(consume.slot, Scalar::Const(0));
        let (issue, commit) = self.staged_halves(produce.body);
        let barrier = self.barrier();
        let head: Vec<Stmt<Expanded>> = issue.iter().chain(&commit).cloned().chain([barrier.clone()]).collect();
        // The clusters: the loads before each product, and the product.
        let (mut loads, mut products): (Vec<Vec<Stmt<Expanded>>>, Vec<Stmt<Expanded>>) = (vec![vec![]], vec![]);
        for stmt in consume.body.0 {
            if matches!(stmt, Stmt::Let { op: TileOp::Mma { .. }, .. }) {
                products.push(stmt);
                loads.push(vec![]);
            } else {
                loads.last_mut().expect("a cluster").push(stmt);
            }
        }
        let trailing = loads.pop().expect("a cluster");
        let s = products.len();
        assert!(s >= 2, "ping-pong overlaps the loads of one product with another");
        let tail = std::mem::take(&mut loads[s - 1]);
        loads[s - 2].extend(tail);
        // Each issued load's operand, numbered in order of appearance.
        let mut sources: Vec<ParamId> = vec![];
        let groups: Vec<Option<usize>> = issue
            .iter()
            .map(|stmt| {
                let Stmt::Copy { src, .. } = stmt else { return None };
                let Place::Global { param, .. } = self.prog.value(*src).place else { return None };
                Some(sources.iter().position(|&p| p == param).unwrap_or_else(|| {
                    sources.push(param);
                    sources.len() - 1
                }))
            })
            .collect();
        let spread = sources.len().saturating_sub(1).max(1);
        let mut commit: Vec<(usize, Stmt<Expanded>)> = groups.iter().flatten().copied().zip(commit).collect();
        commit.sort_by_key(|&(g, _)| g);
        let mut front: Vec<Vec<Stmt<Expanded>>> = vec![vec![]; s];
        for (stmt, g) in issue.into_iter().zip(groups) {
            front[g.unwrap_or(0) * (s - 2) / spread].push(stmt);
        }
        for (cluster, front) in loads.iter_mut().zip(front) {
            cluster.splice(0..0, front);
        }
        loads[s - 1].extend(commit.into_iter().map(|(_, stmt)| stmt));
        let prio = |level| Stmt::Sync(Sync::Prio { level });
        let mut body = vec![];
        for (k, product) in products.into_iter().enumerate() {
            body.append(&mut loads[k]);
            body.push(barrier.clone());
            body.extend([prio(RAISED), product, prio(0)]);
            if k == s - 1 {
                body.extend(trailing.iter().cloned());
            }
            body.push(barrier.clone());
        }
        let warp = self.scalar(Scalar::Special(Special::Warp));
        let crew = self.c(i64::from(crew));
        let first = self.bin(BinOp::Lt, warp, crew);
        let second = self.bin(BinOp::Eq, first, zero);
        let trips = |trip, start, extent, carried, body: Vec<Stmt<Expanded>>| {
            Stmt::Loop(Loop { iv, start, extent, carried, body: Block(body), unroll: 1, trip: Some(trip) })
        };
        vec![
            trips(Trip::Head, before, nonempty, vec![], head),
            Self::if_(second, vec![barrier.clone()]),
            trips(Trip::Steady, zero, extent, carried, body),
            Self::if_(first, vec![barrier]),
        ]
    }

    /// One step ahead over two slots or one: a trip loads step `i` into
    /// registers, consumes step `i - 1` from its slot, writes the registers
    /// to step `i`'s slot. A fence after the loads keeps them ahead of the
    /// products, where their latency hides behind the products; with two
    /// slots another keeps the commit after the products; with one the
    /// commit waits behind a barrier for every warp to finish the slot, then
    /// barriers again, so the slot holds a step at a time and a tile twice
    /// as deep fits the same shared memory. The first trip only loads and
    /// the last only consumes, so each is peeled into a one-trip loop and
    /// the steady loop has no branches: a trip is one basic block, which a
    /// machine scheduler may reorder freely between the fences. Where the
    /// barrier's halves part, a two-slot consume whose tail past its last
    /// shared access branches ([`Self::branching_tail`]) commits and arrives
    /// before that tail and waits after it: a warp working through a branch
    /// the others skip holds them only a trip later, at its next arrive.
    fn register_staged(
        &mut self,
        extent: ScalarId,
        stages: usize,
        carried: Vec<Carried>,
        produce: Stage<Expanded>,
        consume: Stage<Expanded>,
    ) -> Vec<Stmt<Expanded>> {
        assert!(matches!(stages, 1 | 2), "register staging runs one step ahead over one or two slots (`Fill::ring`)");
        let iv = self.scalar(Scalar::Induction);
        let one = self.c(1);
        let two = self.c(2);
        let zero = self.c(0);
        // An empty pipeline runs no trip at all (a range never counts down).
        let nonempty = self.bin(BinOp::Lt, zero, extent);
        let steady = self.bin(BinOp::Sub, extent, nonempty);
        let slot = |step| if stages == 1 { Scalar::Const(0) } else { Scalar::Bin(BinOp::Rem, step, two) };
        self.define(produce.step, Scalar::Bin(BinOp::Add, iv, zero));
        self.define(produce.slot, slot(produce.step));
        self.define(consume.step, Scalar::Bin(BinOp::Sub, iv, one));
        self.define(consume.slot, slot(consume.step));
        let (issue, commit) = self.staged_halves(produce.body);
        let barrier = self.barrier();
        let head: Vec<Stmt<Expanded>> = issue.iter().chain(&commit).cloned().chain([barrier.clone()]).collect();
        let before = if stages == 1 { barrier.clone() } else { Stmt::Sync(Sync::Fence) };
        let cut = if stages == 2 { self.branching_tail(&consume.body) } else { None };
        let (reads, tail) = consume.body.0.split_at(cut.unwrap_or(consume.body.0.len()));
        let (arrive, wait) = match cut {
            Some(_) => (Stmt::Sync(Sync::Arrive), vec![Stmt::Sync(Sync::Await)]),
            None => (barrier, vec![]),
        };
        let fenced = [before].into_iter().chain(commit).chain([arrive]);
        let issued = issue.into_iter().chain([Stmt::Sync(Sync::Fence)]);
        let body: Vec<Stmt<Expanded>> =
            issued.chain(reads.iter().cloned()).chain(fenced).chain(tail.iter().cloned()).chain(wait).collect();
        // The drain's carried registers hold what the steady loop left.
        let drained: Vec<Carried> = carried.iter().map(|c| Carried { init: c.phi, ..*c }).collect();
        let trips = |trip, start, extent, carried, body: Vec<Stmt<Expanded>>, unroll| {
            Stmt::Loop(Loop { iv, start, extent, carried, body: Block(body), unroll, trip: Some(trip) })
        };
        vec![
            trips(Trip::Head, zero, nonempty, vec![], head, 1),
            trips(Trip::Steady, one, steady, carried, body, stages as u32),
            trips(Trip::Drain, extent, nonempty, drained, consume.body.0, 1),
        ]
    }
}

impl Expander<'_> {
    /// Where `consume`'s tail starts — past its last top-level statement
    /// touching shared memory — where that tail branches and the block's
    /// barrier may part around it; `None` otherwise (a warp's own ring, a
    /// tail of products alone, no tail).
    fn branching_tail(&self, consume: &Block<Expanded>) -> Option<usize> {
        if !self.sync.split_barrier || self.scope != Scope::Block {
            return None;
        }
        let shared = |v: ValId| self.prog.value(v).tier() == Tier::Smem;
        let touches = |s: &Stmt<Expanded>| match s {
            Stmt::Copy { dst, src, .. } => shared(*dst) || shared(*src),
            Stmt::Let { dst, op } => shared(*dst) || op.srcs().any(shared),
            _ => false,
        };
        let within = |s: &Stmt<Expanded>| touches(s) || s.blocks().any(|b| b.walk().any(|(_, s)| touches(s)));
        let cut = consume.0.iter().rposition(within).map_or(0, |i| i + 1);
        let branches = consume.0[cut..].iter().any(|s| matches!(s, Stmt::If { .. } | Stmt::Loop(_)));
        branches.then_some(cut)
    }
}

/// Whether a ring `stages` deep leaves a consume's chained products in
/// flight on `core`, a slot of the ring theirs: its fill then runs one trip
/// less ahead than its slots allow.
pub(crate) fn in_flight(fill: Fill, core: Core, stages: usize) -> bool {
    fill == Fill::CpAsync && core == Core::Group && stages >= 3
}

/// Whether every product `consume` issues feeds only later products'
/// accumulators (or leaves the consume as a carried value): nothing in the
/// consume waits on one.
fn chains_only(consume: &Block<Expanded>) -> bool {
    let mut products = vec![];
    for (_, stmt) in consume.walk() {
        match stmt {
            Stmt::Let { dst, op: TileOp::Mma { acc, a, b, .. } } => {
                if products.contains(a) || products.contains(b) {
                    return false;
                }
                products.retain(|p| p != acc);
                products.push(*dst);
            }
            Stmt::Let { op, .. } if op.srcs().any(|v| products.contains(&v)) => return false,
            Stmt::Copy { src, .. } if products.contains(src) => return false,
            _ => {}
        }
    }
    true
}
