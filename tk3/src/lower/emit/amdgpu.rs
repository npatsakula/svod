//! AMDGPU instruction bodies: transposing global loads, the bf16 pair dot
//! product, lane moves on the vector unit and through the LDS crossbar,
//! the hardware `exp2`, and the scheduling and priority barriers.

use std::sync::Arc;

use smallvec::smallvec;
use svod_codegen::llvm::amd::tr::{global_load_tr_b128, global_load_tr_b128_or_zeros};
use svod_dtype::{DType, ScalarDType};
use svod_ir::UOp;

use crate::layout as frag;
use crate::layouts::TileLayout;
use crate::lower::Result;

use super::*;

impl Emit<'_> {
    /// Global → registers by transposing loads, one per 16×16 block, its
    /// lanes addressing the block's rows in [`frag::global_tr_rows`] order and
    /// receiving column runs. Each 16-byte row run is one lane's address and
    /// reaches the lanes it scatters to whole, so a lane whose run lies past
    /// the view's bounds, or whose gathered row is outside its map, addresses
    /// a run of zeros instead and every element it supplies reads zero.
    pub(super) fn tr_load_regs(&mut self, dst: ValId, s: &View, l: &TileLayout, perm: &[u32]) -> Result<()> {
        let [sub_r, sub_c] = l.sub_shape();
        let rows = TileLayout { frag: frag::global_tr_rows(), reps: [sub_r / 16, sub_c / 16], warps: l.warps.clone() };
        for block in 0..rows.regs() {
            let c = self.coord(&rows, block);
            let (off, gate) = self.address(s, &c);
            let idx = self.access(&s.buf, &off, 1);
            let elems = match gate {
                Some(inside) => global_load_tr_b128_or_zeros(&idx, &inside, s.dtype),
                None => global_load_tr_b128(&idx, s.dtype),
            };
            for (jd, &js) in perm.iter().enumerate() {
                if js / 8 == block {
                    self.reg_store(dst, jd as u32, vec![elems[(js % 8) as usize].clone()]);
                }
            }
        }
        Ok(())
    }

    /// `acc + pair·(1, 1)` for a bf16 `pair`, accumulated in f32.
    pub(super) fn dot2_bf16(&mut self, pair: Arc<UOp>, acc: Arc<UOp>) -> Arc<UOp> {
        let call = "declare float @llvm.amdgcn.fdot2.f32.bf16(<2 x bfloat>, <2 x bfloat>, float, i1)\n\
                    call float @llvm.amdgcn.fdot2.f32.bf16(<2 x bfloat> {0}, <2 x bfloat> <bfloat 0xR3F80, bfloat 0xR3F80>, float {1}, i1 false)";
        self.push(UOp::custom(smallvec![pair, acc], call.to_string(), DType::Float32))
    }

    /// `value` of lane `lane ^ mask` through the LDS crossbar (`ds_bpermute`).
    pub(super) fn bpermute_xor(&mut self, value: &Arc<UOp>, mask: u32) -> Arc<UOp> {
        let (word, back) = as_word(value);
        // Pure and lane-only: listed in the prologue, so every branch reaches it.
        let addr = self.hoist(mul(&xor(&self.lane, &c32(mask as i64)), &c32(4)));
        back(UOp::custom(
            smallvec![addr, word],
            "declare i32 @llvm.amdgcn.ds.bpermute(i32, i32)\ncall i32 @llvm.amdgcn.ds.bpermute(i32 {0}, i32 {1})"
                .to_string(),
            DType::Int32,
        ))
    }
}

/// DPP's `row_xmask` control: OR the lane xor-mask within a row of 16.
const DPP_ROW_XMASK: u32 = 0x160;

/// `value` as the 32-bit word a lane move carries, and the way back: a
/// 16-bit value travels zero-extended. A lane move carries one 2- or 4-byte
/// scalar, nothing else.
fn as_word(value: &Arc<UOp>) -> (Arc<UOp>, impl Fn(Arc<UOp>) -> Arc<UOp>) {
    let dtype = value.dtype();
    let bytes = dtype.scalar().map(|s| s.bytes());
    assert!(matches!(bytes, Some(2 | 4)), "a lane move carries a 2- or 4-byte scalar, not {dtype:?}");
    let wide = bytes == Some(4);
    let int = dtype == DType::Int32;
    let word = if int {
        value.clone()
    } else if wide {
        value.bitcast(DType::Int32)
    } else {
        value.bitcast(DType::Scalar(ScalarDType::UInt16)).cast(DType::Int32)
    };
    let back = move |w: Arc<UOp>| {
        if int {
            w
        } else if wide {
            w.bitcast(dtype.clone())
        } else {
            w.cast(DType::Scalar(ScalarDType::UInt16)).bitcast(dtype.clone())
        }
    };
    (word, back)
}

/// `value` of lane `lane ^ mask`, `mask` under 32, on the vector unit: DPP
/// `row_xmask` within a row of 16, `permlanex16` between the rows.
pub(super) fn dpp_xor(value: &Arc<UOp>, mask: u32) -> Arc<UOp> {
    let (mut word, back) = as_word(value);
    if mask & 15 != 0 {
        // `row_xmask:mask`, every row and bank written, so no lane keeps
        // the old value and it need not be set.
        word = UOp::custom(
            smallvec![word],
            format!(
                "declare i32 @llvm.amdgcn.update.dpp.i32(i32, i32, i32, i32, i32, i1)\n\
                 call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 {{0}}, i32 {}, i32 15, i32 15, i1 false)",
                DPP_ROW_XMASK | (mask & 15)
            ),
            DType::Int32,
        );
    }
    if mask & 16 != 0 {
        // Each lane takes the same lane of the other row of 16.
        word = UOp::custom(
            smallvec![word],
            format!(
                "declare i32 @llvm.amdgcn.permlanex16.i32(i32, i32, i32, i32, i1, i1)\n\
                 call i32 @llvm.amdgcn.permlanex16.i32(i32 {{0}}, i32 {{0}}, i32 {}, i32 {}, i1 false, i1 false)",
                0x76543210, 0xFEDCBA98_u32 as i32
            ),
            DType::Int32,
        );
    }
    back(word)
}

/// An f32 `2^x` as the hardware instruction alone (`v_exp_f32`).
pub(super) fn exp2_f32(x: &Arc<UOp>) -> Arc<UOp> {
    UOp::custom(
        smallvec![x.clone()],
        "declare float @llvm.amdgcn.exp2.f32(float)\ncall float @llvm.amdgcn.exp2.f32(float {0})".to_string(),
        DType::Float32,
    )
}

/// No instruction moves across this point (`sched_barrier 0`), after `last`.
pub(super) fn sched_barrier(last: &Arc<UOp>) -> Arc<UOp> {
    UOp::custom(
        smallvec![last.clone()],
        "declare void @llvm.amdgcn.sched.barrier(i32)\ncall void @llvm.amdgcn.sched.barrier(i32 0)".to_string(),
        DType::Void,
    )
}

/// The wave's issue priority from here on (`s_setprio`): a side effect in
/// list order, with no value to depend on, so it may follow a barrier.
pub(super) fn setprio(level: u8) -> Arc<UOp> {
    UOp::custom(
        smallvec![],
        format!("declare void @llvm.amdgcn.s.setprio(i16)\ncall void @llvm.amdgcn.s.setprio(i16 {level})"),
        DType::Void,
    )
}
