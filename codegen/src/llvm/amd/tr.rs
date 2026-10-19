//! `global_load_tr_b128`, the RDNA4 transposing global load: every lane of a
//! wave reads 16 bytes of 16-bit elements, and each group of eight lanes
//! receives the 8×8 block its rows form transposed, lane `k` of the group
//! taking element `k` of every row. Verified against clang 22 at `gfx1201`
//! and on an RX 9070 XT (tk3 `transposing_loads_land_as_their_layout`).

use std::sync::Arc;

use smallvec::{SmallVec, smallvec};
use svod_dtype::{AddrSpace, DType, ScalarDType};
use svod_ir::prelude::*;

use crate::llvm::common::gpu::specific_ptr;
use crate::llvm::common::ldt;

/// The eight elements a lane receives from a transposing load, this lane's
/// 16-byte row read from `global_ptr`.
pub fn global_load_tr_b128(global_ptr: &Arc<UOp>, elem: ScalarDType) -> SmallVec<[Arc<UOp>; 8]> {
    load(specific_ptr(global_ptr, AddrSpace::Global), elem)
}

/// [`global_load_tr_b128`] with this lane's row read from `global_ptr` where
/// `inside` holds, else from a 16-byte run of zeros: one lane's row reaches
/// every lane it scatters to whole, so all it supplies is zero.
pub fn global_load_tr_b128_or_zeros(
    global_ptr: &Arc<UOp>,
    inside: &Arc<UOp>,
    elem: ScalarDType,
) -> SmallVec<[Arc<UOp>; 8]> {
    let zeros = "@svod_tr_zeros = internal unnamed_addr addrspace(1) constant [4 x i32] zeroinitializer, align 16";
    let address = UOp::custom(
        smallvec![inside.clone(), specific_ptr(global_ptr, AddrSpace::Global)],
        format!("{zeros}\nselect i1 {{0}}, ptr addrspace(1) {{1}}, ptr addrspace(1) @svod_tr_zeros"),
        DType::Void.ptr(None, AddrSpace::Global).expect("void is not a pointer"),
    );
    load(address, elem)
}

/// The transposing load of the `ptr addrspace(1)` `address`, element by element.
fn load(address: Arc<UOp>, elem: ScalarDType) -> SmallVec<[Arc<UOp>; 8]> {
    let overload = match elem {
        ScalarDType::BFloat16 => "v8bf16",
        ScalarDType::Float16 => "v8f16",
        other => panic!("a transposing load moves 16-bit elements, not {other:?}"),
    };
    let vec = DType::Scalar(elem).vec(8).expect("eight elements");
    let ty = ldt(&vec);
    let intrinsic = format!("llvm.amdgcn.global.load.tr.b128.{overload}");
    let call = UOp::custom(
        smallvec![address],
        format!("declare {ty} @{intrinsic}(ptr addrspace(1))\ncall {ty} @{intrinsic}(ptr addrspace(1) {{0}})"),
        vec,
    );
    (0..8)
        .map(|i| {
            UOp::custom(smallvec![call.clone()], format!("extractelement {ty} {{0}}, i32 {i}"), DType::Scalar(elem))
        })
        .collect()
}

#[cfg(test)]
#[path = "../../test/unit/llvm_amd_tr.rs"]
mod tests;
