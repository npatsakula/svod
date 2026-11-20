//! The stamps of a traced lowering ([`crate::trace`]): lane 0 of each warp
//! of a sampled block adds the cycles since its last stamp to the word of
//! the statement that just ended.

use svod_codegen::llvm::nvptx::{clock64, globaltimer, red_add};

use super::*;
use crate::trace::{BARE, COUNT, EVERY, rows};

pub(super) struct Tracer {
    /// The trace parameter: a word per statement and warp.
    words: Arc<UOp>,
    /// The cycle count at this thread's last stamp, then the nanosecond it started at.
    last: Arc<UOp>,
    gate: Arc<UOp>,
    /// The last statement since the last stamp that takes none of its own.
    pending: Option<u32>,
}

impl Emit<'_> {
    pub(super) fn trace_open(&mut self) -> Result<()> {
        if !self.low.trace {
            return Ok(());
        }
        let target = &self.low.device.target;
        let elems = rows(self.lines.len()) * self.prog.warps as usize;
        let declared = self
            .prog
            .params
            .last()
            .is_some_and(|p| p.kind == ParamKind::Accum && p.dtype == ScalarDType::Int64 && p.elems == elems);
        ensure!(
            declared && target.atomic_add && matches!(target.arch, svod_dtype::GpuArch::Cuda(_)),
            UnsupportedSnafu { what: format!("a trace on {} or without its parameter", target.arch.target_name()) }
        );
        let words = self.params.last().expect("the trace parameter").clone();
        let last = self.hoist(UOp::committed_buffer(self.reg_slot, 2, DType::Int64, AddrSpace::Reg));
        self.reg_slot += 1;
        let block = (0..3).map(|a| self.block_index(a)).reduce(|a, b| add(&a, &b)).expect("three axes");
        let sampled = block.try_cmod(&c32(EVERY)).expect("sampled").try_cmpeq(&c32(0)).expect("sampled");
        let gate = self.lane.try_cmpeq(&c32(0)).expect("lane").try_and_op(&sampled).expect("gate");
        let gate = self.hoist(gate);
        self.trace = Some(Tracer { words, last, gate, pending: None });
        let (clock, raw) = (self.read(clock64()), self.read(globaltimer()));
        let timer = raw.cast(DType::Int64);
        self.gated(|e| vec![clock.clone(), e.started(0, clock), raw, timer.clone(), e.started(1, timer)]);
        (0..BARE).for_each(|_| self.stamp(self.lines.len() as u32 + 2));
        Ok(())
    }

    /// The unattributed tail in row 0, the block's nanoseconds in the last row.
    pub(super) fn trace_close(&mut self) {
        if self.trace.is_none() {
            return;
        }
        self.trace_flush();
        let last = self.lines.len() as u32 + 1;
        self.gated(|e| {
            let (clock, raw) = (e.read(clock64()), e.read(globaltimer()));
            let timer = raw.cast(DType::Int64);
            let charged = [(0, 0, clock), (1, last, timer)].map(|(slot, row, now)| e.charged(slot, row, now));
            std::iter::once(raw).chain(charged.into_iter().flatten()).collect()
        });
    }

    /// The statements since the last stamp, charged to the last of them.
    pub(super) fn trace_flush(&mut self) {
        if let Some(line) = self.trace.as_mut().and_then(|t| t.pending.take()) {
            self.stamp(line);
        }
    }

    /// `line` ended: a stamp of its own, or it waits for the next one.
    pub(super) fn trace_after(&mut self, line: u32, own: bool) {
        match &mut self.trace {
            Some(_) if own => self.stamp(line),
            Some(t) => t.pending = Some(line),
            None => {}
        }
    }

    fn stamp(&mut self, line: u32) {
        self.gated(|e| {
            let clock = e.read(clock64());
            let mut us = e.charged(0, line, clock.clone());
            us.push(e.started(0, clock));
            us
        });
    }

    /// A read of `counter` of its own: a tag keeps two apart.
    fn read(&mut self, counter: Arc<UOp>) -> Arc<UOp> {
        let tag = self.next_tag();
        counter.rtag(Some(smallvec![tag as usize]))
    }

    fn slot(&mut self, slot: i64) -> Arc<UOp> {
        let last = self.trace.as_ref().expect("a trace").last.clone();
        self.access(&last, &c32(slot), 1)
    }

    fn started(&mut self, slot: i64, now: Arc<UOp>) -> Arc<UOp> {
        self.slot(slot).store(now)
    }

    /// `now` less what `slot` holds, and one hit, added to `row`'s word of this warp.
    fn charged(&mut self, slot: i64, row: u32, now: Arc<UOp>) -> Vec<Arc<UOp>> {
        let tag = self.next_tag();
        let was = load_at(&self.slot(slot), tag);
        let hit = UOp::const_(DType::Int64, ConstValue::Int(COUNT));
        let word = now.try_sub(&was).expect("elapsed").try_add(&hit).expect("word");
        let words = self.trace.as_ref().expect("a trace").words.clone();
        let warp = add(&self.warp, &c32(i64::from(row) * i64::from(self.prog.warps)));
        let at = self.access(&words, &warp, 1);
        let add = red_add(&at, &word);
        // The address is one node for every stamp of the row: listed where they all reach it.
        self.hoist(add.op().sources()[0].clone());
        vec![now, was, word, add]
    }

    fn gated(&mut self, us: impl FnOnce(&mut Self) -> Vec<Arc<UOp>>) {
        let gate = self.trace.as_ref().expect("a trace").gate.clone();
        let tag = self.next_tag();
        let if_ = self.push(UOp::if_(gate, smallvec![]).rtag(Some(smallvec![tag as usize])));
        for u in us(self) {
            if !self.levels.contains_key(&u.id) {
                self.push(u);
            }
        }
        self.push(UOp::endif(if_));
    }
}
