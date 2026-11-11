//! Which instruction each executed wait waited for. A wave is replayed in record order, its memory
//! instructions queued on the counters they increment (LLVM's `SIInsertWaitcnts` events); a wait down
//! to `n` retires all but the `n` youngest of each queue, and the youngest it retires binds it. A
//! counter returns in order, so the binding's issue → the wait's release is that instruction's latency.

use std::collections::{BTreeMap, HashMap, VecDeque};

use serde::{Deserialize, Serialize};

use super::asm::{Access, Counter, Kind, Memory, Space, Wait, classify};
use super::report::Spread;
use super::stitch::branch;
use super::wave::{Instruction, Wave};
use super::{Category, Layout};

/// How an instruction returns relative to the others on a counter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Order {
    Fifo,
    /// Scalar memory and messages.
    Unordered,
    /// `flat_*` counts on both the vector and the LDS counter.
    Flat,
    /// A gfx11 sample or BVH op on `vmcnt`, which plain loads may overtake.
    Image,
}

/// The counters `memory` increments on `layout`.
fn counters(memory: Memory, layout: Layout) -> &'static [(Counter, Order)] {
    use Counter::*;
    use Order::*;
    let gfx12 = layout == Layout::Rdna4;
    let store = matches!(memory.access, Access::Store | Access::Atomic { returns: false } | Access::Writeback);
    match (memory.space, memory.access) {
        (Space::LdsDirect, _) | (_, Access::Prefetch) => &[],
        (Space::Ds, _) if gfx12 => &[(Ds, Fifo)],
        (Space::Ds, _) => &[(Lgkm, Fifo)],
        (Space::Smem, _) if gfx12 => &[(Km, Unordered)],
        (Space::Smem, _) => &[(Lgkm, Unordered)],
        (Space::Flat, _) => match (gfx12, store) {
            (true, true) => &[(Store, Flat), (Ds, Flat)],
            (true, false) => &[(Load, Flat), (Ds, Flat)],
            (false, true) => &[(Store, Flat), (Lgkm, Flat)],
            (false, false) => &[(Load, Flat), (Lgkm, Flat)],
        },
        (_, Access::Sample) if gfx12 => &[(Sample, Fifo)],
        (_, Access::Bvh) if gfx12 => &[(Bvh, Fifo)],
        (_, Access::Sample | Access::Bvh) => &[(Load, Image)],
        (_, Access::Invalidate | Access::Writeback) if !gfx12 => &[],
        _ if store => &[(Store, Fifo)],
        _ => &[(Load, Fifo)],
    }
}

fn message(layout: Layout) -> &'static [(Counter, Order)] {
    match layout {
        Layout::Rdna4 => &[(Counter::Km, Order::Unordered)],
        Layout::Rdna3 => &[(Counter::Lgkm, Order::Unordered)],
    }
}

#[derive(Debug, Clone, Copy)]
struct Pending {
    /// Record index: issue order.
    seq: usize,
    pc: u64,
    issue: i64,
    /// Back edges the wave had taken at issue.
    edges: u32,
    /// In flight on its counter at issue, itself included.
    outstanding: u32,
    order: Order,
}

/// One executed wait: its PC, stall, and what it waited for.
#[derive(Debug, Clone, Copy)]
struct Hit {
    pc: u64,
    stall: i64,
    binding: Option<Bound>,
}

#[derive(Debug, Clone, Copy)]
struct Bound {
    pc: u64,
    /// The binding's issue to the wait's release.
    latency: i64,
    /// Back edges the wave took between the binding's issue and the wait.
    distance: u32,
    outstanding: u32,
    /// A counter that returns out of order, or a flat op, made the binding a guess.
    approx: bool,
}

/// An instruction that bound waits, over every instance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Binding {
    pub pc: u64,
    /// Wait instances it bound, those that stalled, and those whose binding is approximate.
    pub count: usize,
    pub stalled: usize,
    pub approx: usize,
    /// Its issue to the release of a wait it bound that stalled: its latency, give or take the cycles
    /// the released wave waited to be picked. A wait that did not stall bounds it only from above.
    pub latency: Option<Spread>,
    /// `(instructions in flight on its counter at its issue, itself included; latency)` over the stalled
    /// waits: flat means latency-bound, rising means queueing.
    pub by_outstanding: Vec<(u32, Spread)>,
    /// The stall cycles of the waits it bound: its exposed latency.
    pub stall: i64,
    /// `(wait pc, instances)`.
    pub waits: Vec<(u64, usize)>,
    /// `(loop back edges taken between its issue and the wait, instances)`: the prefetch distance.
    pub distance: Vec<(u32, usize)>,
}

/// A wait instruction over every instance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Site {
    pub pc: u64,
    pub instances: usize,
    pub stall: i64,
    /// Instances whose counters were already low enough.
    pub unbound: usize,
    /// `(binding pc, instances)`.
    pub bindings: Vec<(u64, usize)>,
}

/// Waits and the instructions they waited for, over the fully stitched waves.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Waits {
    /// Waves replayed, and those skipped: incomplete, context-switched or not wholly stitched.
    pub waves: usize,
    pub skipped: usize,
    /// Wait records (`IMMED` at a wait instruction), those that bound nothing, and approximate bindings.
    pub instances: usize,
    pub unbound: usize,
    pub approx: usize,
    /// By exposed stall, largest first.
    pub bindings: Vec<Binding>,
    pub sites: Vec<Site>,
}

struct Line {
    kind: Kind,
    /// A backward branch: a loop's back edge when taken.
    back_edge: bool,
}

impl Waits {
    /// `code`: the `(pc, llvm-objdump line)` the waves were stitched against; reported PCs are less `base`.
    pub fn new(waves: &[Wave], code: &[(u64, &str)], layout: Layout, base: u64) -> Self {
        let lines: HashMap<u64, Line> = code
            .iter()
            .map(|&(pc, line)| {
                let back_edge = branch(pc, line).is_some_and(|target| target <= pc);
                (pc, Line { kind: classify(line).kind, back_edge })
            })
            .collect();
        let replayable = |w: &&Wave| w.complete && w.contexts == 0 && w.instructions.iter().all(|i| i.pc != 0);
        let (mut hits, mut waits) = (vec![], Self::default());
        for wave in waves.iter().filter(replayable) {
            replay(&wave.instructions, &lines, layout, &mut hits);
            waits.waves += 1;
        }
        waits.skipped = waves.len() - waits.waves;
        waits.aggregate(&hits, base);
        waits
    }

    fn aggregate(&mut self, hits: &[Hit], base: u64) {
        #[derive(Default)]
        struct Acc {
            count: usize,
            approx: usize,
            latency: Vec<i64>,
            by_outstanding: BTreeMap<u32, Vec<i64>>,
            stall: i64,
            waits: BTreeMap<u64, usize>,
            distance: BTreeMap<u32, usize>,
        }
        let (mut bindings, mut sites) = (BTreeMap::<u64, Acc>::new(), BTreeMap::<u64, Site>::new());
        for hit in hits {
            let site = sites.entry(hit.pc).or_insert_with(|| Site {
                pc: hit.pc - base,
                instances: 0,
                stall: 0,
                unbound: 0,
                bindings: vec![],
            });
            site.instances += 1;
            site.stall += hit.stall;
            let Some(b) = hit.binding else {
                site.unbound += 1;
                continue;
            };
            let acc = bindings.entry(b.pc).or_default();
            acc.count += 1;
            acc.approx += usize::from(b.approx);
            if hit.stall > 0 {
                acc.latency.push(b.latency);
                acc.by_outstanding.entry(b.outstanding).or_default().push(b.latency);
            }
            acc.stall += hit.stall;
            *acc.waits.entry(hit.pc - base).or_default() += 1;
            *acc.distance.entry(b.distance).or_default() += 1;
        }
        for (&pc, acc) in &bindings {
            for (&wait, &n) in &acc.waits {
                sites.get_mut(&(wait + base)).expect("a bound wait's site").bindings.push((pc - base, n));
            }
        }
        let spread = |samples: Vec<i64>| Spread::of(samples).expect("one sample at least");
        self.instances = hits.len();
        self.unbound = sites.values().map(|s| s.unbound).sum();
        self.approx = bindings.values().map(|a| a.approx).sum();
        self.bindings = bindings
            .into_iter()
            .map(|(pc, acc)| Binding {
                pc: pc - base,
                count: acc.count,
                stalled: acc.latency.len(),
                approx: acc.approx,
                latency: Spread::of(acc.latency),
                by_outstanding: acc.by_outstanding.into_iter().map(|(n, l)| (n, spread(l))).collect(),
                stall: acc.stall,
                waits: acc.waits.into_iter().collect(),
                distance: acc.distance.into_iter().collect(),
            })
            .collect();
        self.bindings.sort_by_key(|b| (std::cmp::Reverse(b.stall), b.pc));
        self.sites = sites.into_values().collect();
        for site in &mut self.sites {
            site.bindings.sort_by_key(|&(pc, n)| (std::cmp::Reverse(n), pc));
        }
        self.sites.sort_by_key(|s| (std::cmp::Reverse(s.stall), s.pc));
    }
}

/// One wave from its start: queues every counted instruction, retires them at waits.
fn replay(records: &[Instruction], lines: &HashMap<u64, Line>, layout: Layout, hits: &mut Vec<Hit>) {
    let mut queues: [VecDeque<Pending>; Counter::ALL.len()] = Default::default();
    let mut edges = 0;
    for (seq, record) in records.iter().enumerate() {
        let Some(line) = lines.get(&record.pc) else { continue };
        let counted = match line.kind {
            Kind::Memory(memory) => counters(memory, layout),
            Kind::SendMsg => message(layout),
            Kind::Wait(wait) => {
                let binding = retire(&mut queues, wait, record, edges);
                if record.category == Category::Immed {
                    hits.push(Hit { pc: record.pc, stall: record.stall, binding });
                }
                &[]
            }
            Kind::Branch if line.back_edge && record.category == Category::Jump => {
                edges += 1;
                &[]
            }
            _ => &[],
        };
        for &(counter, order) in counted {
            let queue = &mut queues[counter as usize];
            let outstanding = queue.len() as u32 + 1;
            let issue = record.time + record.stall;
            queue.push_back(Pending { seq, pc: record.pc, issue, edges, outstanding, order });
        }
    }
}

/// Retires what `wait` waits for; the youngest retired binds it.
fn retire(queues: &mut [VecDeque<Pending>], wait: Wait, record: &Instruction, edges: u32) -> Option<Bound> {
    let (mut bound, mut approx) = (None::<Pending>, false);
    for (counter, n) in wait.counts() {
        let queue = &mut queues[counter as usize];
        let n = usize::from(n);
        if queue.len() <= n {
            continue;
        }
        let has = |order: Order| queue.iter().any(|p| p.order == order);
        let mixed_image = has(Order::Image) && queue.iter().any(|p| p.order != Order::Image);
        approx |= (n > 0 && has(Order::Unordered)) || has(Order::Flat) || mixed_image;
        let youngest = queue.drain(..queue.len() - n).next_back().expect("retires one at least");
        if bound.is_none_or(|b| youngest.seq > b.seq) {
            bound = Some(youngest);
        }
    }
    let release = record.time + record.duration;
    bound.map(|p| Bound {
        pc: p.pc,
        latency: release - p.issue,
        distance: edges - p.edges,
        outstanding: p.outstanding,
        approx,
    })
}
