//! AMD's per-wave state machine (`gfx10/rdna_sqtt.cpp`, `gfx10/gfx10wave.cpp`): turns the packets of one
//! shader engine into the instruction records of each wave on the traced SIMD, with AMD's issue time,
//! duration and stall, plus every wave's occupancy.

use std::collections::{HashSet, VecDeque};

use super::{Category, EINST_JUMP, EINST_RFE, EINST_TRAP, Header, Kind, Layout, OpClass, Packet, Unit, WaveStart};

/// One traced instruction. `time` is its issue (or, after a stall, the stall's start); `pc` is 0 until
/// a [`Stitcher`](super::stitch::Stitcher) attributes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Instruction {
    pub time: i64,
    pub category: Category,
    pub duration: i64,
    pub stall: i64,
    pub pc: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Wave {
    pub unit: Unit,
    pub me: u8,
    pub pipe: u8,
    pub begin: i64,
    pub end: i64,
    /// A `WAVE_END` closed it; otherwise the trace stopped first.
    pub complete: bool,
    /// Context saves and restores it went through.
    pub contexts: u8,
    /// gfx12 left its `s_barrier_wait`s out of the trace.
    pub exclude_barrier_wait: bool,
    pub instructions: Vec<Instruction>,
    /// `(instruction index, pc)`: the PCs the trace states (the start, `NEW_PC`s); 0 where it lost one.
    pub pcs: Vec<(usize, u64)>,
}

/// A wave slot filling (`start`) or emptying on any SIMD of the shader engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Occupancy {
    pub time: i64,
    pub unit: Unit,
    pub start: bool,
    pub cause: Cause,
}

/// What filled or emptied a slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cause {
    /// A wave launched or ended.
    Wave,
    /// A context save emptied it or a restore refilled it.
    Context,
    /// A wave ended whose start the trace missed: AMD inserts the start at time 0.
    Orphan,
}

/// A memory op the traced WGP ran for a wave of another SIMD.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OtherSimd {
    pub time: i64,
    pub category: Category,
    pub cycles: u32,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Trace {
    /// The SE's `HEADER`; `None` where its stream announced none the decoder reads.
    pub header: Option<Header>,
    /// The version of a `HEADER` the decoder refused (CDNA's, or a corrupt stream's): the stream
    /// stops there.
    pub refused: Option<u8>,
    /// The traced SIMD's waves, in the order AMD stitches them.
    pub waves: Vec<Wave>,
    pub occupancy: Vec<Occupancy>,
    pub other_simd: Vec<OtherSimd>,
}

/// Runs AMD's wave model over one shader engine's packets.
pub fn analyze(packets: impl IntoIterator<Item = Packet>) -> Trace {
    let mut model = Model::default();
    // AMD's gfx11 generator dates `WAVE_READY` a cycle early and moves it before the tokens of its cycle.
    let mut pending: VecDeque<(i64, Kind)> = VecDeque::new();
    for packet in packets {
        let time = packet.time.min(1 << 60) as i64;
        match packet.kind {
            Kind::Nop | Kind::Time { .. } | Kind::Timestamp { .. } => {}
            Kind::Header(_) => {
                pending.drain(..).for_each(|(time, kind)| model.token(time, kind));
                model.token(time, packet.kind);
            }
            kind if model.header.layout == Layout::Rdna4 => model.token(time, kind),
            kind => {
                let time = time - i64::from(matches!(kind, Kind::WaveReady { .. }));
                match pending.back() {
                    Some(&(back, _)) if time < back => pending.push_front((time, kind)),
                    _ => pending.push_back((time, kind)),
                }
                while pending.front().zip(pending.back()).is_some_and(|(front, back)| front.0 < back.0) {
                    let (time, kind) = pending.pop_front().expect("non-empty");
                    model.token(time, kind);
                }
            }
        }
    }
    pending.drain(..).for_each(|(time, kind)| model.token(time, kind));
    model.finish()
}

const SLOTS: usize = 32;

/// Per `(me & 1, pipe)`, AMD's `PipeArray`.
type Pipes<T> = [[T; 4]; 2];

fn pipe<T>(pipes: &mut Pipes<T>, me: u8, pipe: u8) -> &mut T {
    &mut pipes[usize::from(me & 1)][usize::from(pipe & 3)]
}

#[derive(Default)]
struct Model {
    trace: Trace,
    /// The SE's `HEADER`, layout 3's defaults until one is met.
    header: Header,
    slots: [Vec<State>; SLOTS],
    /// Each pipe's `COMPUTE_PGM`: the entry PC its next wave starts at.
    pgm: Pipes<u64>,
    /// The resident waves' locations.
    running: HashSet<u32>,
    /// The starts AMD inserts at time 0 for waves the trace saw only end, in the order they ended:
    /// they lead the occupancy, the latest first.
    orphans: Vec<Occupancy>,
}

fn location(unit: Unit) -> u32 {
    u32::from(unit.sa_wgp()) << 7 | u32::from(unit.simd) << 5 | u32::from(unit.wave)
}

const COMPUTE_PGM_LO: u16 = 0xc;
const COMPUTE_PGM_HI: u16 = 0xd;

impl Model {
    fn is_target(&self, unit: Unit) -> bool {
        let h = &self.header;
        unit.sa_wgp() == (h.dsa << 7 | h.dwgp) && unit.simd == h.dsimd
    }

    fn occupy(&mut self, time: i64, unit: Unit, start: bool, cause: Cause) {
        self.trace.occupancy.push(Occupancy { time, unit, start, cause });
    }

    fn entry(&mut self, me: u8, pipe_id: u8) -> u64 {
        (*pipe(&mut self.pgm, me, pipe_id) << 8) & ((1 << 48) - 1)
    }

    fn with(&mut self, wave: u8, f: impl FnOnce(&mut State)) {
        if let Some(state) = self.slots.get_mut(usize::from(wave)).and_then(|slot| slot.last_mut()) {
            f(state);
        }
    }

    fn token(&mut self, time: i64, kind: Kind) {
        match kind {
            Kind::Header(Ok(header)) => (self.header, self.trace.header) = (header, Some(header)),
            Kind::Header(Err(version)) => self.trace.refused = Some(version),
            Kind::Misc { fields } if fields & 8 != 0 => {
                self.slots.iter_mut().filter_map(|slot| slot.last_mut()).for_each(|wave| wave.standby = true);
            }
            Kind::WaveStart(start) => self.wave_start(time, start),
            Kind::WaveEnd(unit) => self.wave_end(time, unit),
            Kind::Inst { wave, op, .. } => {
                let class = op.class(&self.header);
                let layout = self.header.layout;
                if class.other_simd {
                    self.trace.other_simd.push(OtherSimd { time, category: class.category, cycles: class.cycles });
                } else {
                    self.with(wave, |s| s.apply_inst(time, op.code, class, layout));
                }
            }
            Kind::ValuInst { wave, .. } => self.with(wave, |s| s.apply_valu(time)),
            Kind::ImmOne { wave } => self.with(wave, |s| s.apply_immediate(time)),
            Kind::Immediate { mask } => {
                (0..16).filter(|i| mask >> i & 1 == 1).for_each(|i| self.with(i, |s| s.apply_immediate(time)));
            }
            Kind::WaveReady { mask } => {
                (0..16).filter(|i| mask >> i & 1 == 1).for_each(|i| self.with(i, |s| s.apply_wave_ready(time)));
            }
            Kind::NewPc { wave, pc } => self.with(wave, |s| s.new_pc(time, pc)),
            Kind::Reg(reg) if reg.cs => {
                let pgm = pipe(&mut self.pgm, reg.me, reg.pipe);
                match reg.addr {
                    COMPUTE_PGM_LO => *pgm = *pgm & !0xffff_ffff | u64::from(reg.data),
                    COMPUTE_PGM_HI => *pgm = *pgm & 0xffff_ffff | u64::from(reg.data) << 32,
                    _ => {}
                }
            }
            _ => {}
        }
    }

    fn wave_start(&mut self, time: i64, start: WaveStart) {
        let WaveStart { unit, me, pipe: pipe_id, count, .. } = start;
        let (loc, target, slot) = (location(unit), self.is_target(unit), usize::from(unit.wave));
        let exbarw = self.header.layout == Layout::Rdna4 && self.header.exbarw;
        if (66..90).contains(&count) {
            let save = 0b00_0011_0011u32 >> (count - 66) & 1 == 1;
            let restore = 0b11_1100_1100u32 >> (count - 66) & 1 == 1;
            if target {
                let waves = &mut self.slots[slot];
                if let Some(state) = waves.last_mut().filter(|_| save) {
                    state.standby = true;
                    state.wave.contexts += 1;
                    state.push(time, Category::Trap, 1, 0);
                }
                if restore {
                    if waves.last().is_none_or(|state| state.wave.complete) {
                        waves.push(State::new(unit, time, 0, exbarw, me, pipe_id));
                    }
                    let state = waves.last_mut().expect("pushed");
                    if let Some(trap) = state.wave.instructions.last_mut().filter(|i| i.category == Category::Trap) {
                        trap.duration = trap.duration.max(time + 1 - trap.time);
                    }
                    state.standby = true;
                    state.wave.contexts += 1;
                }
            }
            if save {
                if self.running.contains(&loc) {
                    self.occupy(time, unit, false, Cause::Context);
                }
            } else if restore {
                self.running.insert(loc);
                self.occupy(time, unit, true, Cause::Context);
            }
            return;
        }
        self.running.insert(loc);
        self.occupy(time, unit, true, Cause::Wave);
        if target {
            while self.slots[slot].last().is_some_and(|state| state.wave.complete) {
                let state = self.slots[slot].pop().expect("non-empty");
                self.trace.waves.push(state.wave);
            }
            let entry = self.entry(me, pipe_id);
            self.slots[slot].push(State::new(unit, time, entry, exbarw, me, pipe_id));
        }
    }

    fn wave_end(&mut self, time: i64, unit: Unit) {
        if self.is_target(unit) {
            self.with(unit.wave, |state| state.complete(time));
        }
        if !self.running.remove(&location(unit)) {
            self.orphans.push(Occupancy { time: 0, unit, start: true, cause: Cause::Orphan });
        }
        self.occupy(time, unit, false, Cause::Wave);
    }

    fn finish(mut self) -> Trace {
        for slot in &mut self.slots {
            self.trace.waves.extend(slot.drain(..).map(|state| state.wave));
        }
        self.trace.occupancy.splice(0..0, self.orphans.into_iter().rev());
        self.trace
    }
}

/// `gfx10::wave_t`: a wave and the state its next token is applied against.
struct State {
    wave: Wave,
    last_state: i64,
    stalled: bool,
    /// A gfx11 `s_barrier`'s `IMMED` stretches into a `MSG` up to the wave's next token.
    extend_barrier: bool,
    /// Trapped or context-switched: tokens are ignored until a trap return.
    standby: bool,
}

impl State {
    fn new(unit: Unit, time: i64, pc: u64, exclude_barrier_wait: bool, me: u8, pipe: u8) -> Self {
        let wave = Wave {
            unit,
            me,
            pipe,
            begin: time,
            end: 0,
            complete: false,
            contexts: 0,
            exclude_barrier_wait,
            instructions: vec![],
            pcs: if pc != 0 { vec![(0, pc)] } else { vec![] },
        };
        Self { wave, last_state: time, stalled: false, extend_barrier: false, standby: false }
    }

    fn push(&mut self, time: i64, category: Category, duration: i64, stall: i64) {
        self.wave.instructions.push(Instruction { time, category, duration, stall, pc: 0 });
    }

    fn update_state(&mut self, stalled: bool, time: i64) {
        self.last_state = self.last_state.max(time);
        self.stalled = stalled;
    }

    /// The issue time and stall of an op whose token came at `time`: a stalled wave issued when it stalled.
    fn stall(&self, time: i64) -> (i64, i64) {
        if self.stalled && self.last_state < time { (self.last_state, time - self.last_state) } else { (time, 0) }
    }

    fn complete(&mut self, time: i64) {
        self.update_state(false, time);
        self.wave.end = time;
        self.wave.complete = true;
        let Some(last) = self.wave.instructions.last() else { return };
        let done = last.time + last.duration;
        // The trace emits no token for `s_endpgm`.
        if done < time && !self.standby {
            self.push(done, Category::Msg, time - done, 0);
        }
    }

    fn apply_wave_ready(&mut self, time: i64) {
        self.update_barrier(time - 1);
        if time > -1 {
            self.update_state(true, time);
        }
    }

    fn new_pc(&mut self, time: i64, pc: u64) {
        self.update_barrier(time);
        let pcs = &mut self.wave.pcs;
        if pcs.is_empty() || self.standby {
            return;
        }
        let at = pcs.len() - pcs.iter().rev().take_while(|(_, pc)| *pc == 0).count();
        if at == pcs.len() {
            pcs.push((self.wave.instructions.len(), 0));
        }
        pcs[at].1 = pc;
    }

    fn apply_valu(&mut self, time: i64) {
        self.update_barrier(time);
        if self.standby {
            return;
        }
        let (issue, stall) = self.stall(time);
        self.push(issue, Category::Valu, 1 + stall, stall);
        self.update_state(false, time);
    }

    fn apply_immediate(&mut self, time: i64) {
        self.update_barrier(time);
        if self.standby {
            return;
        }
        let issue = self.wave.instructions.last().map_or(time, |last| time.min(last.time + last.duration));
        self.push(issue, Category::Immed, time - issue + 1, time - issue);
        self.update_state(false, issue);
        self.update_state(false, time + 1);
    }

    fn update_barrier(&mut self, time: i64) {
        if !std::mem::take(&mut self.extend_barrier) {
            return;
        }
        let Some(&last) = self.wave.instructions.last().filter(|i| i.category == Category::Immed) else { return };
        self.update_state(false, last.time + last.duration);
        self.update_state(false, time + 1);
        let last = self.wave.instructions.last_mut().expect("non-empty");
        last.duration = (time - last.time).max(0);
        last.category = Category::Msg;
    }

    fn apply_inst(&mut self, time: i64, code: u8, class: OpClass, layout: Layout) {
        self.update_barrier(time);
        if code == EINST_RFE {
            self.wave.pcs.push((self.wave.instructions.len(), 0));
            self.standby = false;
            if let Some(trap) = self.wave.instructions.last_mut().filter(|i| i.category == Category::Trap) {
                trap.duration = trap.duration.max(time + 1 - trap.time);
            }
            return;
        }
        if code == EINST_TRAP {
            self.push(time, Category::Trap, 1, 0);
            self.standby = true;
        }
        if self.standby {
            return;
        }
        self.wave.end = time;
        match class.category {
            Category::Immed => {
                self.apply_immediate(time);
                self.extend_barrier = layout == Layout::Rdna3;
            }
            Category::None => {}
            category => {
                let (issue, stall) = self.stall(time);
                self.push(issue, category, i64::from(class.cycles) + stall, stall);
                if code == EINST_JUMP {
                    self.wave.pcs.push((self.wave.instructions.len(), 0));
                }
                self.update_state(false, time);
            }
        }
    }
}
