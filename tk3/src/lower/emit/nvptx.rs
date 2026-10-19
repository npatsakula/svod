//! NVPTX instruction bodies: `ldmatrix` gathers, `ld.global.cg` loads, the
//! warpgroup products over shared descriptors and a warp's own barrier.

use std::sync::Arc;

use smallvec::smallvec;

use svod_codegen::llvm::nvptx::smem::{ld_global_cg_16, ldmatrix};
use svod_codegen::llvm::nvptx::wgmma;
use svod_dtype::DType;
use svod_ir::{UOp, WmmaMetadata};

use crate::layout::Dim::Lane;
use crate::layout::Layout;
use crate::layouts::TileLayout;
use crate::lower::{Result, UnsupportedSnafu};
use snafu::OptionExt;

use super::*;

impl Emit<'_> {
    pub(super) fn ldmatrix_gather(
        &mut self,
        dst: ValId,
        s: &View,
        l: &TileLayout,
        trans: bool,
        perm: &[u32],
    ) -> Result<()> {
        let [sub_r, sub_c] = l.sub_shape();
        let (blocks_r, blocks_c) = (sub_r / 16, sub_c / 16);
        let pair = DType::Scalar(s.dtype).vec(2).expect("a pair");
        let lane16 = self.hoist(and(&self.lane, &c32(15)));
        let lane_hi = self.hoist(mul(&shr(&self.lane, 4), &c32(8)));
        // Where this warp's sub-tile starts.
        let (wr, wc) =
            self.coord(&TileLayout { frag: Layout::zeros(Lane, 1), reps: [1, 1], warps: l.warps.clone() }, 0).full();
        let (wr, wc) = (self.hoist(mul(&wr, &c32(sub_r as i64))), self.hoist(mul(&wc, &c32(sub_c as i64))));
        for br in 0..blocks_r {
            for bc in 0..blocks_c {
                let row = self.hoist(add(&add(&wr, &c32((br * 16) as i64)), &lane16));
                let col = self.hoist(add(&add(&wc, &c32((bc * 16) as i64)), &lane_hi));
                let (off, _) = self.address(s, &Coord::plain(row, col));
                let idx = self.access(&s.buf, &off, 1);
                let words = ldmatrix(&idx, 4, trans, pair.clone());
                let block = br * blocks_c + bc;
                for (jd, &js) in perm.iter().enumerate() {
                    let jd = jd as u32;
                    if js / 8 != block {
                        continue;
                    }
                    let e = js % 8;
                    // Register pair `p` holds matrix `p` of the TL, BL, TR, BR address
                    // order, with the middle two swapped under `.trans`.
                    let word = if trans { [0, 2, 1, 3][(e / 2) as usize] } else { (e / 2) as usize };
                    let v = elem(&words[word], (e % 2) as usize, 2);
                    self.reg_store(dst, jd, vec![v]);
                }
            }
        }
        Ok(())
    }

    /// A 16-byte run of `w` elements of `s` at `off` loaded at L2 only (as a
    /// `cp.async.cg` fill reads), a run past `gate` reading element 0 and
    /// landing as zeros.
    pub(super) fn load_l2_only(&mut self, s: &View, off: Arc<UOp>, gate: Option<Arc<UOp>>, w: usize) -> Vec<Arc<UOp>> {
        let safe = match &gate {
            Some(g) => self.hoist(UOp::try_where(g.clone(), off, c32(0)).expect("safe offset")),
            None => off,
        };
        let src = self.access(&s.buf, &safe, 1);
        let words = ld_global_cg_16(&src);
        let per_word = w / words.len();
        let vals: Vec<Arc<UOp>> = (0..w)
            .map(|e| {
                let word = &words[e / per_word];
                match per_word {
                    1 if word.dtype() == s.dtype.into() => word.clone(),
                    1 => word.bitcast(s.dtype.into()),
                    n => elem(&word.bitcast(s.dtype.vec(n)), e % n, n),
                }
            })
            .collect();
        match gate {
            Some(g) => vals
                .into_iter()
                .map(|v| UOp::try_where(g.clone(), v.clone(), v.const_like(0)).expect("zeroed"))
                .collect(),
            None => vals,
        }
    }

    /// A product on the warpgroup atom: per instruction column block, one
    /// `Wmma` over the whole K chain, its shared operands as the descriptors
    /// of this warp's group rows (and the block's columns of `B`) at every
    /// `k16` step, `A` as its register fragments when it is a register tile.
    /// `B` is K-major (`b_t`: its rows run along N) or MN-major (its rows run
    /// along K, the instruction transposing them). The descriptor's swizzle
    /// is the lowering's own (the hardware applies it to the absolute
    /// address, so the ring's slots start on 1024-byte boundaries) over the
    /// operand's rows, or over its column panels where a row is longer than
    /// the widest span ([`Self::panel_cols`]). The chain is fenced and
    /// committed inside the one statement (`codegen::llvm::nvptx::wgmma`) and
    /// waited for there too, unless the block holds the
    /// [`Sync::WaitProducts`] that waits for it later.
    pub(super) fn group_mma(&mut self, dst: ValId, op: &TileOp) -> Result<()> {
        let TileOp::Mma { acc, a, b, a_t, b_t, .. } = *op else { unreachable!("a product") };
        let (va, vb, vd) = (self.value(a).clone(), self.value(b).clone(), self.value(dst).clone());
        let atom = self.low.device.target.group_mma(va.dtype, vd.dtype).expect("inference found the core").clone();
        snafu::ensure!(
            !a_t && vb.tier() == Tier::Smem && self.low.swizzle,
            UnsupportedSnafu { what: "a warpgroup product over other than an untransposed A and swizzled shared B" }
        );
        let grid = self.low.grid;
        let k = va.shape.cols;
        let issue = atom.issue(grid, vd.shape.rows, vd.shape.cols, k).expect("inference laid this product");
        let n_atom = atom.n_for(vd.shape.cols as u32 / grid.cols).expect("inference laid this product") as usize;
        let (rn, rc, rk) = (issue.c.reps[1] as usize, issue.c.frag_regs() as usize, k / wgmma::K);
        // This warp's group and the group's place in the grid, columns first.
        let group = self.hoist(shr(&self.warp, 2));
        let cols = c32(grid.cols as i64);
        let group_row = self.hoist(group.try_cdiv(&cols).expect("group row"));
        let group_col = self.hoist(group.try_cmod(&cols).expect("group column"));
        // The `k16` steps' descriptors of a shared operand from `at`: its
        // first row when K-major (rows along M or N), else its first column
        // (rows along K). Panel `p` of `pc` columns starts `p·rows·pc` in.
        let describe = |this: &mut Self, v: ValId, at: Arc<UOp>, k_major: bool| -> Result<Vec<Arc<UOp>>> {
            let view = this.view(v);
            snafu::ensure!(
                this.along(&view).0 == Axis::Col,
                UnsupportedSnafu { what: "a column-major shared operand" }
            );
            let (rows, e) = (view.shape.rows, view.dtype.bytes());
            let pc = this.panel_cols(view.shape.cols, e).unwrap_or(view.shape.cols);
            let swizzle = wgmma::Swizzle::for_row_bytes(pc * e)
                .context(UnsupportedSnafu { what: format!("a warpgroup operand over {}-byte shared rows", pc * e) })?;
            snafu::ensure!(
                k_major || n_atom.is_multiple_of(pc),
                UnsupportedSnafu { what: format!("an MN-major n{n_atom} block within a {pc}-wide panel") }
            );
            // The leading offset steps between an MN-major operand's panels; a
            // K-major one never reads it and declares one unit, as ever.
            let leading = if k_major { 16 } else { (rows * pc * e) as u32 };
            let stride = (8 * pc * e) as u32;
            let Some(Place::Smem { offset, .. }) = view.place else { unreachable!("a shared operand") };
            let (_, offset) = this.scalar(offset);
            let offset = match &view.region {
                Some(region) => add(&offset, region),
                None => offset,
            };
            let base = if k_major {
                mul(&at, &c32(pc as i64))
            } else {
                mul(&shr(&at, pc.trailing_zeros()), &c32((rows * pc) as i64))
            };
            let base = this.hoist(add(&offset, &base));
            Ok((0..rk)
                .map(|kk| {
                    let kk = kk * wgmma::K;
                    let step = if k_major { kk / pc * rows * pc + kk % pc } else { kk * pc };
                    let at = add(&base, &c32(step as i64));
                    let ptr = this.access(&view.buf, &at, 1);
                    let address = this.hoist(wgmma::shared_address(&ptr));
                    this.hoist(wgmma::descriptor(&address, leading, stride, swizzle))
                })
                .collect())
        };
        // One descriptor is the operand itself, not a one-wide vector.
        let gather =
            |ops: Vec<Arc<UOp>>| if ops.len() == 1 { ops[0].clone() } else { UOp::stack(ops.into_iter().collect()) };
        let from_registers = va.tier() == Tier::Reg;
        let a_op = if from_registers {
            let regs = issue.a.frag_regs() as usize;
            UOp::stack((0..rk).flat_map(|kk| self.reg_loads(a, (kk * regs) as u32, regs as u32)).collect())
        } else {
            let rows = self.hoist(mul(&group_row, &c32(wgmma::M as i64)));
            gather(describe(self, a, rows, true)?)
        };
        let meta = WmmaMetadata {
            name: wgmma::name(from_registers, !b_t, !self.async_products),
            dims: (n_atom, wgmma::M, k),
            dtype_in: DType::Scalar(va.dtype),
            dtype_out: DType::Scalar(vd.dtype),
            device: atom.device,
            threads: wgmma::THREADS,
            upcast_axes: None,
            reduce_axes: vec![],
        };
        for j in 0..rn {
            let at = add(&mul(&group_col, &c32((n_atom * rn) as i64)), &c32((j * n_atom) as i64));
            let at = self.hoist(at);
            let b_op = gather(describe(self, b, at, b_t)?);
            let c = UOp::stack(self.reg_loads(acc, (j * rc) as u32, rc as u32).into_iter().collect());
            let d = UOp::wmma(a_op.clone(), b_op, c, meta.clone());
            let vals = (0..rc).map(|p| elem(&d, p, rc)).collect();
            self.reg_store(dst, (j * rc) as u32, vals);
        }
        Ok(())
    }
}

/// A warp's own barrier over its shared accesses (`bar.warp.sync`), after `last`.
pub(super) fn warp_barrier(last: &Arc<UOp>) -> Arc<UOp> {
    UOp::custom(
        smallvec![last.clone()],
        "declare void @llvm.nvvm.bar.warp.sync(i32)\ncall void @llvm.nvvm.bar.warp.sync(i32 -1)".to_string(),
        DType::Void,
    )
}
