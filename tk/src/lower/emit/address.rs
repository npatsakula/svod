//! Where a register's element lives in memory: views, the offsets and
//! bounds gates of their coordinates, and the runs a layout is accessed by.

use svod_ir::ops;

use crate::layout::Dim::{Col, Lane, Reg, Row, Warp};
use crate::layout::Layout;
use crate::layouts::Swizzle;

use super::*;

impl Emit<'_> {
    /// The access node of a `w`-wide element run at `off`, listed where its
    /// offset is, so every copy of a loop body reaches it.
    pub(super) fn access(&mut self, buf: &Arc<UOp>, off: &Arc<UOp>, w: u32) -> Arc<UOp> {
        self.hoist(access(buf, off, w as usize))
    }

    pub(super) fn mem_load(&mut self, buf: &Arc<UOp>, off: &Arc<UOp>, w: u32, gate: Option<&Arc<UOp>>) -> Arc<UOp> {
        let tag = self.next_tag();
        let idx = self.access(buf, off, w);
        let u = match gate {
            Some(gate) => {
                let zero = self.hoist_zero(load_zero(&idx));
                load_gated_at(&idx, gate, &zero, tag)
            }
            None => load_at(&idx, tag),
        };
        self.push(u)
    }

    pub(super) fn mem_store(&mut self, buf: &Arc<UOp>, off: &Arc<UOp>, vals: Vec<Arc<UOp>>) {
        let idx = self.access(buf, off, vals.len() as u32);
        let value = self.packed(vals);
        self.push(idx.store(value));
    }

    /// One `w`-wide `val` stored whole at `off`.
    pub(super) fn mem_store_wide(&mut self, buf: &Arc<UOp>, off: &Arc<UOp>, w: u32, val: Arc<UOp>) {
        let idx = self.access(buf, off, w);
        self.push(idx.store(val));
    }

    /// The `w` values as one stored value: the wide value they were all
    /// extracted from, reordered by one shuffle, where every element is an
    /// extract (`elem`) of one source — the [`fill`](Self::fill) path, which
    /// loads a chunk wide and must not tear it into per-element stores. Any
    /// other elements keep the stack: a gated or selected element is not an
    /// extract chain and stays scalar.
    fn packed(&mut self, vals: Vec<Arc<UOp>>) -> Arc<UOp> {
        let lanes: Option<Vec<(&Arc<UOp>, u32)>> = vals.iter().map(|val| lane_of(val)).collect();
        if let Some(lanes) = lanes
            && lanes.len() > 1
            && lanes.iter().all(|(src, _)| Arc::ptr_eq(src, lanes[0].0))
        {
            let mask: smallvec::SmallVec<[u32; 8]> = lanes.iter().map(|(_, lane)| *lane).collect();
            let tag = self.next_tag();
            return UOp::shuffle(lanes[0].0.clone(), mask).rtag(Some(smallvec![tag as usize]));
        }
        store_at_value(vals)
    }

    pub(super) fn view(&mut self, v: ValId) -> View {
        let val = self.value(v).clone();
        let buf = match &val.place {
            Place::Global { param, .. } => self.params[param.index()].clone(),
            Place::Smem { alloc, .. } => self.smem_as(alloc.index(), val.dtype),
            Place::Reg => unreachable!("a memory view"),
        };
        View { buf, place: Some(val.place), shape: val.shape, dtype: val.dtype, region: None }
    }

    /// Allocation `alloc`'s buffer read under `dtype`: its own where they
    /// agree, else the whole of it as a view typed `dtype` (an address runs
    /// over the view's type), made once per pair. A view's offset counts its
    /// own elements, so nothing is scaled.
    pub(super) fn smem_as(&mut self, alloc: usize, dtype: ScalarDType) -> Arc<UOp> {
        let a = &self.prog.smem[alloc];
        if a.dtype == dtype {
            return self.smem[alloc].clone();
        }
        if let Some(buf) = self.typed.get(&(alloc, dtype)) {
            return buf.clone();
        }
        let elems = (a.elems * a.dtype.bytes() / dtype.bytes()) as i64;
        let shrink = ops::Shrink { src: self.smem[alloc].clone(), offsets: c32(0), sizes: c32(elems) };
        let buf = self.hoist(UOp::new(Op::Shrink(shrink), DType::Scalar(dtype)));
        self.typed.insert((alloc, dtype), buf.clone());
        buf
    }

    /// Element offset of a register's coordinate in a memory view, with the
    /// swizzle a shared view takes, and its bounds gate if any. The lane's
    /// part of the offset is one value listed where its inputs are (the
    /// prologue, for a tile). In a plain global view the view's offset is
    /// added to it and the register's constant last, so a compiler keeps the
    /// first two as a running pointer and the constant as an immediate; a
    /// gathered row and a swizzled shared row need the full coordinate.
    pub(super) fn address(&mut self, view: &View, c: &Coord) -> (Arc<UOp>, Option<Arc<UOp>>) {
        match &view.place {
            Some(Place::Global { offset, stride, rows: Some(map), .. }) => {
                let (row, col) = c.full();
                let ((_, offset), (_, s1)) = (self.scalar(*offset), self.scalar(stride[1]));
                let (_, start) = self.scalar_at(map.offset, Some(&row));
                // The view's offset moves per step; the row's start and the
                // column are listed where their inputs are.
                let off = add(&offset, &add(&start, &mul(&col, &s1)));
                let gate = self.inside(view, c);
                (self.hoist(off), gate.map(|g| self.hoist(g)))
            }
            Some(Place::Global { offset, stride, .. }) => {
                let ((_, offset), (_, s0), (_, s1)) =
                    (self.scalar(*offset), self.scalar(stride[0]), self.scalar(stride[1]));
                let lane = self.hoist(add(&mul(&c.row, &s0), &mul(&c.col, &s1)));
                let mut off = add(&offset, &lane);
                let delta =
                    [(c.dr, &s0), (c.dc, &s1)].into_iter().filter(|(d, _)| *d != 0).map(|(d, s)| mul(&c32(d), s));
                if let Some(delta) = delta.reduce(|a, b| add(&a, &b)) {
                    off = add(&off, &delta);
                }
                let gate = self.inside(view, c);
                (self.hoist(off), gate.map(|g| self.hoist(g)))
            }
            Some(Place::Smem { .. }) | None => {
                let (mut row, mut col) = c.full();
                let mut cols = view.shape.cols as i64;
                // A column-major tile is the row-major tile of its transpose.
                if self.along(view).0 == Axis::Row {
                    (row, col) = (col, row);
                    cols = view.shape.rows as i64;
                }
                let offset = match &view.place {
                    Some(Place::Smem { offset, .. }) => self.sc(*offset),
                    _ => c32(0),
                };
                // The relayout scratch is addressed through the warp's region.
                let offset = match &view.region {
                    Some(region) => add(&offset, region),
                    None => offset,
                };
                let bytes = view.dtype.bytes();
                // A row past the widest swizzle span is kept as column panels.
                let panel = (self.along(view).0 == Axis::Col)
                    .then(|| self.panel_cols(view.shape.cols, bytes))
                    .flatten()
                    .map(|pc| pc as i64);
                let (panel_off, cols) = match panel {
                    Some(pc) => {
                        let bits = pc.trailing_zeros();
                        let at = mul(&shr(&col, bits), &c32(view.shape.rows as i64 * pc));
                        col = and(&col, &c32(pc - 1));
                        (Some(at), pc)
                    }
                    None => (None, cols),
                };
                let chunk = 16 / bytes as i64;
                let swizzle = Swizzle::of_row(cols as usize, bytes);
                let col = if swizzle != Swizzle::None {
                    let (c, within) = (shr(&col, chunk.trailing_zeros()), and(&col, &c32(chunk - 1)));
                    let (shift, mask) = (3 - swizzle.bits(), c32((1 << swizzle.bits()) - 1));
                    let sw = if shift == 0 { and(&row, &mask) } else { and(&shr(&row, shift), &mask) };
                    add(&mul(&xor(&c, &sw), &c32(chunk)), &within)
                } else {
                    col
                };
                let lane = add(&mul(&row, &c32(cols)), &col);
                let lane = self.hoist(panel_off.map_or(lane.clone(), |at| add(&at, &lane)));
                (self.hoist(add(&offset, &lane)), None)
            }
            Some(Place::Reg) => unreachable!(),
        }
    }

    /// Whether coordinate `c` of a global view lies within its bounds and,
    /// gathered, inside its row map; `None` where the view has neither.
    pub(super) fn inside(&mut self, view: &View, c: &Coord) -> Option<Arc<UOp>> {
        let Some(Place::Global { bounds, rows, .. }) = &view.place else { return None };
        let (row, col) = c.full();
        let in_rows = match rows {
            Some(map) => map.valid.map(|v| {
                let (_, v) = self.scalar_at(v, Some(&row));
                v.try_cmpne(&c32(0)).expect("row gate")
            }),
            None => bounds[0].map(|b| row.try_cmplt(&self.sc(b)).expect("bound compare")),
        };
        let in_cols = bounds[1].map(|b| col.try_cmplt(&self.sc(b)).expect("bound compare"));
        in_rows.into_iter().chain(in_cols).reduce(|a, b| a.try_and_op(&b).expect("and"))
    }

    /// The coordinate register `j` of this thread holds in layout `l`:
    /// its lane- and warp-dependent part, induction-free and listed in the
    /// prologue, and the constant the register adds to it.
    pub(super) fn coord(&mut self, l: &TileLayout, j: u32) -> Coord {
        let regs = l.frag_regs();
        let rep = j / regs;
        let (rr, rc) = (rep / l.reps[1], rep % l.reps[1]);
        let (fr, fc) = l.frag.apply(&[(Reg, j % regs)]);
        let [fr_n, fc_n] = l.frag_shape();
        let [sr, sc] = l.sub_shape();
        // What the bits of `src` move in each dimension, and the bits they reach.
        let bit_terms = |src: &Arc<UOp>, layout: &Layout, dim| {
            let (mut r, mut c): (Option<Arc<UOp>>, Option<Arc<UOp>>) = (None, None);
            let (mut mr, mut mc) = (0, 0);
            let mix = |acc: &mut Option<Arc<UOp>>, bit: &Arc<UOp>, v: u32| {
                if v != 0 {
                    let term = mul(bit, &c32(v as i64));
                    *acc = Some(acc.as_ref().map_or(term.clone(), |a| xor(a, &term)));
                }
            };
            for b in 0..layout.in_bits(dim) {
                let out = layout.basis(dim, b);
                let get = |d| out.iter().find(|o| o.0 == d).map_or(0, |o| o.1);
                let (vr, vc) = (get(Row), get(Col));
                if vr == 0 && vc == 0 {
                    continue;
                }
                let bit = and(&shr(src, b), &c32(1));
                mix(&mut r, &bit, vr);
                mix(&mut c, &bit, vc);
                (mr, mc) = (mr | vr, mc | vc);
            }
            (r.unwrap_or_else(|| c32(0)), c.unwrap_or_else(|| c32(0)), mr, mc)
        };
        let (lr, lc, mr, mc) = bit_terms(&self.lane, &l.frag, Lane);
        let (wr, wc, ..) = bit_terms(&self.warp, &l.warps, Warp);
        // The register's own bits add to the lane's when they share none (the
        // XOR is then a sum), and the constant stays apart; else it is mixed in.
        let (fr, fc) = (fr as i64, fc as i64);
        let ((lr, dr), (lc, dc)) = if fr & mr as i64 == 0 && fc & mc as i64 == 0 {
            ((lr, fr), (lc, fc))
        } else {
            ((xor(&lr, &c32(fr)), 0), (xor(&lc, &c32(fc)), 0))
        };
        let row = add(&mul(&wr, &c32(sr as i64)), &lr);
        let col = add(&mul(&wc, &c32(sc as i64)), &lc);
        Coord { row: self.hoist(row), col: self.hoist(col), dr: dr + (rr * fr_n) as i64, dc: dc + (rc * fc_n) as i64 }
    }

    /// The axis a view stores consecutive elements along, and whether a run
    /// along it may be one vector access: a global view is contiguous along
    /// the axis of its unit stride (neither: every element on its own), a
    /// shared tile along its allocation's axis.
    pub(super) fn along(&self, view: &View) -> (Axis, bool) {
        match &view.place {
            Some(Place::Global { stride, .. }) => {
                let unit = |s: ScalarId| matches!(self.prog.scalar(s), Scalar::Const(1));
                if unit(stride[1]) { (Axis::Col, true) } else { (Axis::Row, unit(stride[0])) }
            }
            Some(Place::Smem { alloc, .. }) => (self.prog.smem[alloc.index()].along, true),
            None => (Axis::Col, true),
            Some(Place::Reg) => unreachable!(),
        }
    }

    /// The width in elements of the column panels a shared row of `cols`
    /// elements is stored as: under the warpgroup core a row longer than the
    /// widest swizzle span (128 bytes) is kept as panels of that span, each
    /// a row-major region of its own, so a descriptor's canonical layout
    /// reaches every row; `None` keeps the row whole.
    pub(super) fn panel_cols(&self, cols: usize, bytes: usize) -> Option<usize> {
        let (row, span) = (cols * bytes, Swizzle::B128.span());
        (self.low.core == Core::Group && row > span && row.is_multiple_of(span)).then_some(span / bytes)
    }

    /// The register runs of `l` a view is accessed by: 16-byte vectors along
    /// the view's axis where it is contiguous, else single elements.
    /// A row-major global view whose rows are a constant stride apart that
    /// is no multiple of the vector's elements is accessed element by
    /// element: its rows start off the vector alignment.
    pub(super) fn runs(&self, l: &TileLayout, view: &View) -> Vec<(u32, u32)> {
        let (along, contiguous) = self.along(view);
        let width = if contiguous { 16 / view.dtype.bytes() as u32 } else { 1 };
        let misaligned = match &view.place {
            Some(Place::Global { stride, rows: None, .. }) if along == Axis::Col && view.shape.rows > 1 => {
                matches!(self.prog.scalar(stride[0]), Scalar::Const(s) if s % i64::from(width) != 0)
            }
            _ => false,
        };
        l.runs(self.low.device.target.wave, if misaligned { 1 } else { width }, along)
    }
}
