//! Copies between the tiers: fills of shared memory, register loads from
//! shared and global memory, stores, and relayouts between register layouts.

use svod_codegen::llvm::nvptx::smem::{cp_async_16, cp_async_16_zfill};
use svod_ir::ConstValue;

use crate::layout as frag;
use crate::layouts::{Pick, Relayout};
use crate::lower::DivergentBarrierSnafu;
use snafu::OptionExt;

use super::*;

impl Emit<'_> {
    pub(super) fn copy(&mut self, dst: ValId, src: ValId, mode: CopyMode, scope: Scope) -> Result<()> {
        match (self.value(dst).tier(), self.value(src).tier()) {
            (Tier::Smem, Tier::Global) => self.fill(dst, src, mode, scope),
            (Tier::Reg, Tier::Smem) => self.gather(dst, src),
            (Tier::Reg, Tier::Global) => self.load_regs(dst, src, mode),
            (Tier::Smem | Tier::Global, Tier::Reg) => self.store_regs(dst, src),
            (Tier::Reg, Tier::Reg) => self.relayout(dst, src),
            (d, s) => UnsupportedSnafu { what: format!("a {s:?} → {d:?} copy") }.fail(),
        }
    }

    /// The warps a register tile is spread over: the lowering's grid, which
    /// key warps sharing a block's tiles are not part of.
    pub(super) fn grid_warps(&self) -> u32 {
        self.low.grid.count()
    }

    /// Global → shared, every thread of the copy's scope (the block's, or
    /// one warp's own) moving 16-byte chunks.
    pub(super) fn fill(&mut self, dst: ValId, src: ValId, mode: CopyMode, scope: Scope) -> Result<()> {
        let (d, s) = (self.view(dst), self.view(src));
        let chunk = 16 / d.dtype.bytes() as i64;
        let cols = d.shape.cols as i64;
        let (warps, tid) = match scope {
            Scope::Block => (self.prog.warps, self.tid.clone()),
            Scope::Warp => (1, self.lane.clone()),
        };
        let threads = (warps * self.low.device.target.wave) as i64;
        let chunks = d.shape.rows as i64 * cols / chunk;
        snafu::ensure!(
            cols % chunk == 0 && chunks % threads == 0,
            UnsupportedSnafu { what: format!("a {}×{} fill by {threads} threads", d.shape.rows, cols) }
        );
        snafu::ensure!(self.along(&d).0 == Axis::Col, UnsupportedSnafu { what: "a chunk fill of a column-major tile" });
        let Some(Place::Global { stride, bounds, rows, .. }) = &s.place else { unreachable!() };
        let gathered = rows.is_some();
        snafu::ensure!(
            matches!(self.prog.scalar(stride[1]), Scalar::Const(1)),
            UnsupportedSnafu { what: "a fill from a column-strided view" }
        );
        let bounds = *bounds;
        let cpr = cols / chunk;
        for t in 0..chunks / threads {
            let c = add(&tid, &c32(t * threads));
            let (row, cc) = (c.try_cdiv(&c32(cpr)).expect("row"), c.try_cmod(&c32(cpr)).expect("chunk"));
            let (row, col) = (self.hoist(row), self.hoist(mul(&cc, &c32(chunk))));
            // Rows past a bound re-read the last valid one (the first, under an
            // empty bound); the consumer masks.
            let src_row = match bounds[0] {
                Some(b) => {
                    let bound = self.sc(b);
                    let last = bound.try_sub(&c32(1)).expect("bound").try_max(&c32(0)).expect("floor");
                    let over = row.try_cmplt(&last).expect("cmp");
                    self.hoist(UOp::try_where(over, row.clone(), last).expect("clamp"))
                }
                None => row.clone(),
            };
            let (src_off, gate) = self.address(&s, &Coord::plain(src_row, col.clone()));
            // A plain view's fill clamps its rows (above) and, under a column
            // bound, lands a chunk starting past it as zeros, as a gathered
            // row outside its map does, from element 0 of the operand.
            let gate = gate.filter(|_| gathered || bounds[1].is_some());
            let src_off = match &gate {
                Some(g) => self.hoist(UOp::try_where(g.clone(), src_off, c32(0)).expect("safe offset")),
                None => src_off,
            };
            let (dst_off, _) = self.address(&d, &Coord::plain(row, col));
            if mode == CopyMode::Async && self.low.device.target.fill == Fill::CpAsync {
                let (dst, src) = (self.access(&d.buf, &dst_off, 1), self.access(&s.buf, &src_off, 1));
                match gate {
                    Some(g) => {
                        let bytes = self.hoist(UOp::try_where(g, c32(16), c32(0)).expect("source size"));
                        self.push(cp_async_16_zfill(&dst, &src, &bytes))
                    }
                    None => self.push(cp_async_16(&dst, &src)),
                };
            } else {
                let v = self.mem_load(&s.buf, &src_off, chunk as u32, None);
                let v = match gate {
                    Some(g) => self.zeroed(&g, v),
                    None => v,
                };
                let vals = (0..chunk as usize).map(|e| elem(&v, e, chunk as usize)).collect();
                self.mem_store(&d.buf, &dst_off, vals);
            }
        }
        Ok(())
    }

    /// Shared → registers under the destination's layout: `ldmatrix.x4` per
    /// 16×16 block when the layout is a register permutation of what it
    /// produces, else vector loads per register run.
    pub(super) fn gather(&mut self, dst: ValId, src: ValId) -> Result<()> {
        let l = self.layout(dst).clone();
        let s = self.view(src);
        let [sub_r, sub_c] = l.sub_shape();
        // `ldmatrix` reads 16-byte memory rows: a row-major tile only.
        let rows = self.along(&s).0 == Axis::Col;
        if self.low.device.target.ldmatrix && rows && s.dtype.bytes() == 2 && sub_r % 16 == 0 && sub_c % 16 == 0 {
            for trans in [false, true] {
                let produced = TileLayout {
                    frag: frag::ldmatrix_x4(trans),
                    reps: [sub_r / 16, sub_c / 16],
                    warps: l.warps.clone(),
                };
                if let Some(perm) = l.as_permutation_of(&produced, self.grid_warps(), self.low.device.target.wave) {
                    return self.ldmatrix_gather(dst, &s, &l, trans, &perm);
                }
            }
        }
        for (j, w) in self.runs(&l, &s) {
            let c = self.coord(&l, j);
            let (off, _) = self.address(&s, &c);
            let v = self.mem_load(&s.buf, &off, w, None);
            let vals = (0..w as usize).map(|e| elem(&v, e, w as usize)).collect();
            self.reg_store(dst, j, vals);
        }
        Ok(())
    }

    /// Global → registers: past a plain bound the load is gated; a gathered
    /// row outside its map reads element 0 and one select zeroes the run.
    /// Where the target loads transposed, [`Self::tr_load_regs`] takes over.
    /// The register half of a fill ([`CopyMode::Staged`])
    /// reads as the kernel's fills read: where those are `cp.async.cg`
    /// copies, its 16-byte runs load at L2 only, bypassing L1 as the copies
    /// do; under register staging, as plain loads.
    pub(super) fn load_regs(&mut self, dst: ValId, src: ValId, mode: CopyMode) -> Result<()> {
        let l = self.layout(dst).clone();
        let s = self.view(src);
        if let Some(perm) = self.transposing(&l, &s) {
            return self.tr_load_regs(dst, &s, &l, &perm);
        }
        let gathered = matches!(s.place, Some(Place::Global { rows: Some(_), .. }));
        let l2_only = mode == CopyMode::Staged && self.low.schedule.fill() == Fill::CpAsync;
        for (j, w) in self.runs(&l, &s) {
            let c = self.coord(&l, j);
            let (off, gate) = self.address(&s, &c);
            if l2_only && w as usize * s.dtype.bytes() == 16 {
                let vals = self.load_l2_only(&s, off, gate, w as usize);
                self.reg_store(dst, j, vals);
                continue;
            }
            let v = match gate {
                Some(g) if gathered => {
                    let safe = self.hoist(UOp::try_where(g.clone(), off, c32(0)).expect("safe offset"));
                    let v = self.mem_load(&s.buf, &safe, w, None);
                    self.zeroed(&g, v)
                }
                gate => self.mem_load(&s.buf, &off, w, gate.as_ref()),
            };
            let vals = (0..w as usize).map(|e| elem(&v, e, w as usize)).collect();
            self.reg_store(dst, j, vals);
        }
        Ok(())
    }

    /// The register permutation under which `l` is what transposing 16×16
    /// loads leave a row-major 16-bit `view` in, where the target has them
    /// and a column bound ends a 16-byte run, which such a load reads whole.
    pub(super) fn transposing(&self, l: &TileLayout, view: &View) -> Option<Vec<u32>> {
        let [sub_r, sub_c] = l.sub_shape();
        let Some(Place::Global { bounds, .. }) = view.place else { return None };
        let runs = bounds[1].is_none_or(|b| multiple_of(self.prog, b, 16 / view.dtype.bytes() as i64));
        let global = self.along(view) == (Axis::Col, true) && runs;
        if !self.low.device.target.tr_load || !global || view.dtype.bytes() != 2 || sub_r % 16 != 0 || sub_c % 16 != 0 {
            return None;
        }
        let produced =
            TileLayout { frag: frag::global_tr_b128(), reps: [sub_r / 16, sub_c / 16], warps: l.warps.clone() };
        l.as_permutation_of(&produced, self.grid_warps(), self.low.device.target.wave)
    }

    /// Registers → memory in the runs [`Self::runs`] gives, gated by the
    /// view's bounds; a run that a column bound may end inside (one not
    /// known to be a multiple of its width) is stored element by element,
    /// each gated, so nothing past the bound is written.
    pub(super) fn store_regs(&mut self, dst: ValId, src: ValId) -> Result<()> {
        let l = self.layout(src).clone();
        let d = self.view(dst);
        let ragged = |w: u32| match &d.place {
            Some(Place::Global { bounds: [_, Some(b)], .. }) => {
                self.along(&d).0 == Axis::Col && !multiple_of(self.prog, *b, i64::from(w))
            }
            _ => false,
        };
        let runs: Vec<(u32, u32)> = self
            .runs(&l, &d)
            .into_iter()
            .flat_map(|(j, w)| if w > 1 && ragged(w) { (j..j + w).map(|e| (e, 1)).collect() } else { vec![(j, w)] })
            .collect();
        for (j, w) in runs {
            let c = self.coord(&l, j);
            let (off, gate) = self.address(&d, &c);
            let vals = self.reg_loads(src, j, w);
            match gate {
                Some(gate) => {
                    let tag = self.next_tag();
                    let if_ = self.push(UOp::if_(gate, smallvec![]).rtag(Some(smallvec![tag as usize])));
                    self.mem_store(&d.buf, &off, vals);
                    self.push(UOp::endif(if_));
                }
                None => self.mem_store(&d.buf, &off, vals),
            }
        }
        Ok(())
    }

    pub(super) fn relayout(&mut self, dst: ValId, src: ValId) -> Result<()> {
        let (ls, ld) = (self.layout(src).clone(), self.layout(dst).clone());
        match ls.relayout(&ld, self.grid_warps(), self.low.device.target.wave, &self.low.device.costs) {
            Relayout::Identity => {
                let buf = self.reg_buf(src);
                let root = self.root(dst);
                self.regs.insert(root, buf);
                Ok(())
            }
            Relayout::RegPermute(perm) => {
                for (jd, js) in perm.into_iter().enumerate() {
                    let v = self.reg_load(src, js);
                    self.reg_store(dst, jd as u32, vec![v]);
                }
                Ok(())
            }
            Relayout::Exchange(picks) => {
                // A (mask, register) move is made once and shared by every
                // destination register that selects it.
                let mut moved: HashMap<(u32, u32), Arc<UOp>> = HashMap::new();
                for (jd, picks) in picks.into_iter().enumerate() {
                    let mut out: Option<Arc<UOp>> = None;
                    for p in picks.into_iter().rev() {
                        let v = moved
                            .entry((p.mask, p.reg))
                            .or_insert_with(|| {
                                let x = self.reg_load(src, p.reg);
                                if p.mask == 0 { x } else { self.shuffle_xor(&x, p.mask) }
                            })
                            .clone();
                        out = Some(match out {
                            None => v,
                            Some(rest) => {
                                let mine = self.hoist(self.lane_in(p.lanes));
                                UOp::try_where(mine, v, rest).expect("exchange select")
                            }
                        });
                    }
                    self.reg_store(dst, jd as u32, vec![out.expect("a move per register")]);
                }
                Ok(())
            }
            Relayout::ViaSmem => self.relayout_via_smem(dst, src),
        }
    }

    /// `dst = src` one step on along `axis`, `fill` first: in-lane where a
    /// lane holds the predecessor, else the lane moves of
    /// [`TileLayout::shift`], selected by lane as an exchange's.
    pub(super) fn shift(&mut self, dst: ValId, src: ValId, axis: Axis, fill: Const) -> Result<()> {
        let l = self.layout(src).clone();
        let moves = l
            .shift(axis, self.grid_warps(), self.low.device.target.wave)
            .context(UnsupportedSnafu { what: "a shift whose predecessors are in another warp" })?;
        let dtype = DType::Scalar(self.value(dst).dtype);
        let fill = match fill {
            Const::Int(i) => UOp::const_(dtype, ConstValue::Int(i)),
            Const::Float(f) => UOp::const_(dtype, ConstValue::Float(f)),
        };
        self.moved(dst, src, moves, Some(fill));
        Ok(())
    }

    /// `dst = src` exchanged across the bits of `mask` along `axis`: a
    /// register renaming where a lane holds the partner, else the lane moves
    /// of [`TileLayout::flip`].
    pub(super) fn flip(&mut self, dst: ValId, src: ValId, axis: Axis, mask: u32) -> Result<()> {
        let l = self.layout(src).clone();
        let moves = l
            .flip(axis, mask, self.grid_warps(), self.low.device.target.wave)
            .context(UnsupportedSnafu { what: "a flip whose partners are in another warp" })?;
        self.moved(dst, src, moves, None);
        Ok(())
    }

    /// Each register of `dst` from the `moves` of `src` per lane, `fill` in
    /// the lanes without one.
    fn moved(&mut self, dst: ValId, src: ValId, moves: Vec<(Vec<Pick>, u64)>, fill: Option<Arc<UOp>>) {
        let mut moved: HashMap<(u32, u32), Arc<UOp>> = HashMap::new();
        for (jd, (picks, first)) in moves.into_iter().enumerate() {
            let mut out = (first != 0).then(|| fill.clone().expect("a fill where a lane has no source"));
            for p in picks.into_iter().rev() {
                let v = moved
                    .entry((p.mask, p.reg))
                    .or_insert_with(|| {
                        let x = self.reg_load(src, p.reg);
                        if p.mask == 0 { x } else { self.shuffle_xor(&x, p.mask) }
                    })
                    .clone();
                out = Some(match out {
                    None => v,
                    Some(rest) => {
                        let mine = self.hoist(self.lane_in(p.lanes));
                        UOp::try_where(mine, v, rest).expect("shift select")
                    }
                });
            }
            self.reg_store(dst, jd as u32, vec![out.expect("a source per register")]);
        }
    }

    /// Whether this lane is one of `lanes`: a 32-bit word on a wave of 32,
    /// a 64-bit one on a wave64.
    pub(super) fn lane_in(&self, lanes: u64) -> Arc<UOp> {
        let (dtype, word, lane) = match self.low.device.target.wave {
            ..=32 => (DType::Int32, lanes as u32 as i32 as i64, self.lane.clone()),
            _ => (DType::Int64, lanes as i64, self.lane.cast(DType::Int64)),
        };
        let c = |v: i64| UOp::const_(dtype.clone(), ConstValue::Int(v));
        let bit = and(&c(word).try_shr_op(&lane).expect("lane bit"), &c(1));
        bit.try_cmpne(&c(0)).expect("lane predicate")
    }

    /// A tile re-held through a scratch of shared memory; inside a
    /// [`Stmt::Warps`] every warp takes its own region of it.
    pub(super) fn relayout_via_smem(&mut self, dst: ValId, src: ValId) -> Result<()> {
        let Value { dtype, shape, .. } = self.value(src).clone();
        let elems = shape.elems();
        let regions = if self.scope == Scope::Warp { self.prog.warps as usize } else { 1 };
        let scratch = match self.scratch.get(&dtype) {
            Some((buf, size)) if *size >= elems * regions => buf.clone(),
            _ => {
                let buf = UOp::buffer(self.local_slot, elems * regions, DType::Scalar(dtype), AddrSpace::Local, None);
                self.local_slot += 1;
                let buf = self.hoist(buf);
                self.scratch.insert(dtype, (buf.clone(), elems * regions));
                buf
            }
        };
        // A warp's region starts at its index's multiple of the tile: an
        // offset of every access, never a crop of the scratch, since an
        // access of a crop is folded into the buffer at the summed offset
        // once the program's placeholders are bound, and the sum is listed
        // nowhere.
        let region = (regions > 1).then(|| self.hoist(mul(&self.warp, &c32(elems as i64))));
        let view = View { buf: scratch, place: None, shape, dtype, region };
        snafu::ensure!(
            self.scope == Scope::Warp || self.divergent == 0,
            DivergentBarrierSnafu { what: "a relayout through shared memory" }
        );
        self.sync(Sync::Barrier { scope: self.scope })?;
        self.store_view(&view, src)?;
        self.sync(Sync::Barrier { scope: self.scope })?;
        self.gather_view(dst, &view)
    }

    pub(super) fn store_view(&mut self, d: &View, src: ValId) -> Result<()> {
        let l = self.layout(src).clone();
        for (j, w) in self.runs(&l, d) {
            let c = self.coord(&l, j);
            let (off, _) = self.address(d, &c);
            let vals = self.reg_loads(src, j, w);
            self.mem_store(&d.buf, &off, vals);
        }
        Ok(())
    }

    pub(super) fn gather_view(&mut self, dst: ValId, s: &View) -> Result<()> {
        let l = self.layout(dst).clone();
        for (j, w) in self.runs(&l, s) {
            let c = self.coord(&l, j);
            let (off, _) = self.address(s, &c);
            let v = self.mem_load(&s.buf, &off, w, None);
            let vals = (0..w as usize).map(|e| elem(&v, e, w as usize)).collect();
            self.reg_store(dst, j, vals);
        }
        Ok(())
    }

    /// `v` where `gate` holds, else zeros: one select over the whole run (a
    /// multiply would turn an Inf read in its place into NaN).
    pub(super) fn zeroed(&mut self, gate: &Arc<UOp>, v: Arc<UOp>) -> Arc<UOp> {
        let zero = self.hoist_zero(v.vconst_like(0));
        UOp::try_where(gate.clone(), v, zero).expect("zero fill")
    }

    /// A zero is one hash-consed node per type: a vector of them (rendered as
    /// an instruction chain) is listed in the prologue rather than at its
    /// first use, which another branch would not reach; a scalar is a literal.
    pub(super) fn hoist_zero(&mut self, zero: Arc<UOp>) -> Arc<UOp> {
        if matches!(zero.op(), Op::Const(..) | Op::VConst(..)) { zero } else { self.hoist(zero) }
    }
}
