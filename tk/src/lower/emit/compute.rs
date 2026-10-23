//! Tile operations on registers: elementwise, products, reductions, and the
//! lane moves a reduction or an exchange issues.

use svod_codegen::llvm::nvptx::ops::shfl_bfly;
use svod_ir::ConstValue;

use crate::atoms::LaneMove;
use crate::layout::Dim::{Col, Row};
use snafu::OptionExt;

use super::*;

impl Emit<'_> {
    pub(super) fn let_(&mut self, dst: ValId, op: &TileOp) -> Result<()> {
        let dtype = DType::Scalar(self.value(dst).dtype);
        let regs = self.layout(dst).regs();
        match *op {
            TileOp::Fill(c) => {
                let v = match c {
                    Const::Int(i) => UOp::const_(dtype, ConstValue::Int(i)),
                    Const::Float(f) => UOp::const_(dtype, ConstValue::Float(f)),
                };
                for j in 0..regs {
                    self.reg_store(dst, j, vec![v.clone()]);
                }
            }
            TileOp::Splat(id) => {
                let v = self.sc(id).cast(dtype);
                for j in 0..regs {
                    self.reg_store(dst, j, vec![v.clone()]);
                }
            }
            TileOp::Coord(axis) => {
                let l = self.layout(dst).clone();
                for j in 0..regs {
                    let (row, col) = self.coord(&l, j).full();
                    let v = self.hoist(if axis == Axis::Row { row } else { col });
                    self.reg_store(dst, j, vec![v]);
                }
            }
            TileOp::Unary { src, f } => {
                for j in 0..regs {
                    let x = self.reg_load(src, j);
                    let y = match f {
                        UnaryOp::Neg => x.neg(),
                        UnaryOp::Exp2 => self.exp2(&x),
                        UnaryOp::Log2 => x.try_log2().expect("log2"),
                        UnaryOp::Recip => UOp::try_reciprocal(&x).expect("recip"),
                        UnaryOp::Sqrt => x.try_sqrt().expect("sqrt"),
                        UnaryOp::Rsqrt => x.try_rsqrt().expect("rsqrt"),
                        UnaryOp::Abs => x.abs(),
                        UnaryOp::Not => x.not(),
                    };
                    self.reg_store(dst, j, vec![y]);
                }
            }
            TileOp::Binary { a, b, f: BinaryOp::Scale } => self.scale(dst, a, b),
            TileOp::Binary { a, b, f } => {
                let (ja, jb) = (self.operand_regs(dst, a), self.operand_regs(dst, b));
                for j in 0..regs {
                    let (x, y) = (self.reg_load(a, ja[j as usize]), self.reg_load(b, jb[j as usize]));
                    let z = match f {
                        BinaryOp::Add => x.try_add(&y),
                        BinaryOp::Sub => x.try_sub(&y),
                        BinaryOp::Mul | BinaryOp::Scale => x.try_mul(&y),
                        BinaryOp::Div if x.dtype().is_float() => x.try_div(&y),
                        BinaryOp::Div => x.try_cdiv(&y),
                        BinaryOp::Max => x.try_max(&y),
                        BinaryOp::Min => minimum(&x, &y),
                        BinaryOp::Lt => x.try_cmplt(&y),
                        BinaryOp::Le => x.try_cmple(&y),
                        BinaryOp::Eq => x.try_cmpeq(&y),
                        BinaryOp::Ne => x.try_cmpne(&y),
                        BinaryOp::And => x.try_and_op(&y),
                        BinaryOp::Or => x.try_or_op(&y),
                    }
                    .expect("elementwise");
                    self.reg_store(dst, j, vec![z]);
                }
            }
            TileOp::Cast { src, to, round } => {
                for j in 0..regs {
                    let x = self.reg_load(src, j);
                    let y = match round {
                        Round::Nearest => x.cast(DType::Scalar(to)),
                        // The upper half of the f32's bits.
                        Round::Zero => shr(&x.bitcast(DType::Int32), 16).cast(DType::Int16).bitcast(DType::Scalar(to)),
                    };
                    self.reg_store(dst, j, vec![y]);
                }
            }
            TileOp::Where { pred, a, b } => {
                let (jp, ja, jb) = (self.operand_regs(dst, pred), self.operand_regs(dst, a), self.operand_regs(dst, b));
                for j in 0..regs {
                    let p = self.reg_load(pred, jp[j as usize]);
                    let (x, y) = (self.reg_load(a, ja[j as usize]), self.reg_load(b, jb[j as usize]));
                    self.reg_store(dst, j, vec![UOp::try_where(p, x, y).expect("where")]);
                }
            }
            TileOp::MulAdd { a, b, c } => {
                let (ja, jb, jc) = (self.operand_regs(dst, a), self.operand_regs(dst, b), self.operand_regs(dst, c));
                for j in 0..regs {
                    let (x, y) = (self.reg_load(a, ja[j as usize]), self.reg_load(b, jb[j as usize]));
                    let z = self.reg_load(c, jc[j as usize]);
                    self.reg_store(dst, j, vec![UOp::try_mulacc(x, y, z).expect("mul-add")]);
                }
            }
            TileOp::Mma { .. } if self.low.core == Core::Group => self.group_mma(dst, op)?,
            TileOp::Mma { .. } => self.mma(dst, op)?,
            TileOp::Reduce { src, axis, f } => self.reduce(dst, src, axis, f)?,
            TileOp::Transpose { src } => self.transpose(dst, src)?,
            TileOp::Shift { src, axis, fill } => self.shift(dst, src, axis, fill)?,
            TileOp::Flip { src, axis, mask } => self.flip(dst, src, axis, mask)?,
            TileOp::Deal { src, j, col } => self.deal(dst, src, j, col)?,
            TileOp::Relayout { src } => self.relayout(dst, src)?,
            TileOp::Move { src } => {
                for j in 0..regs {
                    let x = self.reg_load(src, j);
                    self.reg_store(dst, j, vec![x]);
                }
            }
        }
        Ok(())
    }

    /// `dst = srcᵀ` by moving registers within each thread: the inference
    /// holds `dst` under `src`'s transposed layout, in which every thread
    /// holds the elements it holds of `src`, at registers of another order
    /// where the layout repeats its fragment along both dimensions.
    pub(super) fn transpose(&mut self, dst: ValId, src: ValId) -> Result<()> {
        let (ls, ld) = (self.layout(src).clone(), self.layout(dst).clone());
        let (warps, lanes) = (self.grid_warps(), self.low.device.target.wave);
        for k in 0..ld.regs() {
            let held = |w: u32, l: u32| {
                let (r, c) = ld.coord(w, l, k);
                (c, r)
            };
            let j = (0..ls.regs()).find(|&j| ls.coord(0, 0, j) == held(0, 0));
            let j = j
                .filter(|&j| (0..warps).all(|w| (0..lanes).all(|l| ls.coord(w, l, j) == held(w, l))))
                .context(UnsupportedSnafu { what: "a transpose that moves elements between threads" })?;
            let x = self.reg_load(src, j);
            self.reg_store(dst, k, vec![x]);
        }
        Ok(())
    }

    /// For every register of `dst`, the register of `operand` holding the
    /// same element (or, for a broadcast vector, the element of its row/column).
    pub(super) fn operand_regs(&self, dst: ValId, operand: ValId) -> Vec<u32> {
        let (ld, lo) = (self.layout(dst), self.layout(operand));
        let os = self.value(operand).shape;
        let ds = self.value(dst).shape;
        if os == ds {
            assert_eq!(ld, lo, "the inference holds elementwise operands in the result's layout");
            return (0..ld.regs()).collect();
        }
        let key = |r: u32, c: u32| if os.cols == 1 { (r, 0) } else { (0, c) };
        let lanes = self.low.device.target.wave;
        (0..ld.regs())
            .map(|j| {
                let (r, c) = ld.coord(0, 0, j);
                let want = key(r, c);
                let found = (0..lo.regs()).find(|&jo| key(lo.coord(0, 0, jo).0, lo.coord(0, 0, jo).1) == want);
                let jo = found.expect("the vector holds every row/column of the tile");
                debug_assert!((1..lanes).all(|lane| {
                    let (r, c) = ld.coord(0, lane, j);
                    key(lo.coord(0, lane, jo).0, lo.coord(0, lane, jo).1) == key(r, c)
                }));
                jo
            })
            .collect()
    }

    pub(super) fn mma(&mut self, dst: ValId, op: &TileOp) -> Result<()> {
        let TileOp::Mma { acc, a, b, a_t, b_t, orient } = *op else { unreachable!("a product") };
        let orient = orient.expect("a laid program orients every product");
        let (va, vd) = (self.value(a).clone(), self.value(dst).clone());
        let atom = self.low.device.target.mma(va.dtype, vd.dtype).expect("inference found the core").clone();
        let k = if a_t { va.shape.rows } else { va.shape.cols };
        // The product as laid, from the one function the inference used: the
        // slot each operand feeds and the registers each fragment holds.
        let issue = atom
            .issue(orient, self.prog.warp_grid, vd.shape.rows, vd.shape.cols, k)
            .expect("inference laid this product");
        let [rm, rn] = issue.c.reps;
        let rk = (k / atom.k as usize) as u32;
        let (ra, rb, rc) = (issue.a.frag_regs(), issue.b.frag_regs(), issue.c.frag_regs());
        let swapped = issue.swapped;
        // Fragment block index of each operand for a (m, n, k) step.
        let block_a = |i: u32, kk: u32| if a_t { kk * rm + i } else { i * rk + kk };
        let block_b = |j: u32, kk: u32| if b_t { j * rk + kk } else { kk * rn + j };
        for i in 0..rm {
            for j in 0..rn {
                let cblock = i * rn + j;
                let mut c = self.reg_load_vec(acc, cblock * rc, rc);
                for kk in 0..rk {
                    let xa = UOp::stack(self.reg_loads(a, block_a(i, kk) * ra, ra).into_iter().collect());
                    let xb = UOp::stack(self.reg_loads(b, block_b(j, kk) * rb, rb).into_iter().collect());
                    let (x, y) = if swapped { (xb, xa) } else { (xa, xb) };
                    c = UOp::wmma(x, y, c, atom.meta.clone());
                }
                self.reg_store_vec(dst, cblock * rc, rc, c);
            }
        }
        Ok(())
    }

    /// The fold runs in the destination's type: a 16-bit source is widened
    /// element by element, except a bf16 sum on a target with `v_dot2_f32_bf16`,
    /// which adds a lane's pairs as held with one instruction each.
    pub(super) fn reduce(&mut self, dst: ValId, src: ValId, axis: Axis, f: ReduceOp) -> Result<()> {
        let (ls, ld) = (self.layout(src).clone(), self.layout(dst).clone());
        let (from, to) = (self.value(src).dtype, self.value(dst).dtype);
        let dot2 =
            from != to && from == ScalarDType::BFloat16 && f == ReduceOp::Sum && self.low.device.target.dot2_bf16;
        let fold_masks = ls.fold_masks(axis);
        let across = ls.warps.out_size(if axis == Axis::Row { Col } else { Row });
        snafu::ensure!(across == 1, UnsupportedSnafu { what: "a reduction across warps" });
        let key = |l: &TileLayout, lane: u32, j: u32| {
            let (r, c) = l.coord(0, lane, j);
            if axis == Axis::Row { r } else { c }
        };
        let combine = |x: &Arc<UOp>, y: &Arc<UOp>| {
            match f {
                ReduceOp::Sum => x.try_add(y),
                ReduceOp::Max => x.try_max(y),
                ReduceOp::Min => minimum(x, y),
            }
            .expect("reduce op")
        };
        for jd in 0..ld.regs() {
            let want = key(&ld, 0, jd);
            let members: Vec<u32> = (0..ls.regs()).filter(|&j| key(&ls, 0, j) == want).collect();
            debug_assert!(!members.is_empty());
            let (mut acc, rest) = if dot2 {
                let mut acc = UOp::const_(DType::Float32, ConstValue::Float(0.0));
                let (pairs, rest) = members.as_chunks::<2>();
                for &[a, b] in pairs {
                    let (x, y) = (self.reg_load(src, a), self.reg_load(src, b));
                    acc = self.dot2_bf16(UOp::stack(smallvec![x, y]), acc);
                }
                (acc, rest.to_vec())
            } else {
                let x = self.reg_load(src, members[0]);
                (if from != to { x.cast(DType::Scalar(to)) } else { x }, members[1..].to_vec())
            };
            for &j in &rest {
                let x = self.reg_load(src, j);
                let x = if from != to { x.cast(DType::Scalar(to)) } else { x };
                acc = combine(&acc, &x);
            }
            for &mask in &fold_masks {
                let partner = self.shuffle_xor(&acc, mask);
                acc = combine(&acc, &partner);
            }
            self.reg_store(dst, jd, vec![acc]);
        }
        Ok(())
    }

    /// `dst = a · b` ([`BinaryOp::Scale`]): `a` as it is, then the products
    /// under a branch on whether any factor this thread holds differs from 1,
    /// which a wave whose threads all hold ones jumps over.
    pub(super) fn scale(&mut self, dst: ValId, a: ValId, b: ValId) {
        let (ja, jb) = (self.operand_regs(dst, a), self.operand_regs(dst, b));
        let regs = ja.len();
        for (j, &src) in ja.iter().enumerate() {
            let x = self.reg_load(a, src);
            self.reg_store(dst, j as u32, vec![x]);
        }
        let mut held = jb.clone();
        held.sort_unstable();
        held.dedup();
        let one = |f: &Arc<UOp>| UOp::const_(f.dtype(), ConstValue::Float(1.0));
        let pred = held
            .into_iter()
            .map(|r| {
                let f = self.reg_load(b, r);
                f.try_cmpne(&one(&f)).expect("factor test")
            })
            .reduce(|p, q| p.try_or_op(&q).expect("or"))
            .expect("a factor");
        let tag = self.next_tag();
        let if_ = self.push(UOp::if_(pred, smallvec![]).rtag(Some(smallvec![tag as usize])));
        for j in 0..regs {
            let (x, y) = (self.reg_load(a, ja[j]), self.reg_load(b, jb[j]));
            self.reg_store(dst, j as u32, vec![x.try_mul(&y).expect("product")]);
        }
        self.push(UOp::endif(if_));
    }

    /// `2^x`. Where the target has [`crate::atoms::Target::hardware_exp2`], an f32
    /// one is the hardware instruction alone (`v_exp_f32`), without the
    /// range reduction LLVM wraps `llvm.exp2` in to keep results below 2^-126
    /// denormal: those flush to zero, which no kernel tells apart.
    pub(super) fn exp2(&mut self, x: &Arc<UOp>) -> Arc<UOp> {
        if self.low.device.target.hardware_exp2 && x.dtype() == DType::Float32 {
            super::amdgpu::exp2_f32(x)
        } else {
            x.try_exp2().expect("exp2")
        }
    }

    /// Whether some element of the boolean tile `v` this warp holds is set
    /// ([`Scalar::Any`]): each lane's own elements or'd, then the warp's
    /// vote, uniform over its lanes. Listed at this point, behind a compiler
    /// memory barrier: the branch on it would otherwise let the backend sink
    /// the loads listed above it (a pipeline's next step, used past the
    /// branch) below it, where their latency is no longer hidden.
    pub(super) fn vote_any(&mut self, v: ValId) -> Arc<UOp> {
        let mut lane = self.reg_load(v, 0);
        for j in 1..self.layout(v).regs() {
            let x = self.reg_load(v, j);
            lane = self.push(lane.try_or_op(&x).expect("a boolean or"));
        }
        if !matches!(self.low.device.target.lane_move, LaneMove::SimdShuffle) {
            let pin = UOp::custom(
                smallvec![lane.clone()],
                "call void asm sideeffect \"\", \"~{{memory}}\"()".to_string(),
                DType::Void,
            );
            self.push(pin);
        }
        let vote = match self.low.device.target.lane_move {
            LaneMove::Shuffle => UOp::custom(
                smallvec![lane],
                "declare i1 @llvm.nvvm.vote.any.sync(i32, i1)\ncall i1 @llvm.nvvm.vote.any.sync(i32 -1, i1 {0})"
                    .to_string(),
                DType::Bool,
            ),
            LaneMove::SimdShuffle => UOp::custom(smallvec![lane], "simd_any({0})".to_string(), DType::Bool),
            LaneMove::Permute | LaneMove::Dpp => {
                let ballot = super::amdgpu::ballot(&lane, self.low.device.target.wave);
                let ballot = self.push(ballot);
                ballot.try_cmpne(&ballot.const_like(0)).expect("a ballot compare")
            }
        };
        self.push(vote)
    }

    pub(super) fn shuffle_xor(&mut self, value: &Arc<UOp>, mask: u32) -> Arc<UOp> {
        match self.low.device.target.lane_move {
            LaneMove::Shuffle => shfl_bfly(value, &c32(mask as i64)),
            LaneMove::SimdShuffle => svod_codegen::c::metal::simd_shuffle_xor(value, &c32(mask as i64)),
            LaneMove::Dpp if mask < 32 => super::amdgpu::dpp_xor(value, mask),
            LaneMove::Permute | LaneMove::Dpp => self.bpermute_xor(value, mask),
        }
    }
}
