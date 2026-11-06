//! `buffer_load_dwordx4` through a descriptor built around a uniform base,
//! with the row stride and cache-swizzle bit a flat load has no room for.

use std::sync::Arc;

use smallvec::{SmallVec, smallvec};
use svod_dtype::{AddrSpace, DType, ScalarDType};
use svod_ir::prelude::*;

use crate::llvm::common::gpu::specific_ptr;
use crate::llvm::common::ldt;

/// Descriptor word 1 bit 30, in the 16 bits the stride field shares with it.
const CACHE_SWIZZLE: u16 = 0x4000;

/// Descriptor word 3: the `DST_SEL`/format bits a raw 32-bit load reads under.
const FLAGS: u32 = 0x11_0000;

/// The `w` elements (16 bytes) at `voffset` bytes past `base`, a pointer
/// uniform in the wave, whose rows lie `row_bytes` apart (below 2^14). `tag`
/// keeps loads of one address apart, as a load's own tag does.
pub fn swizzled_load_b128(
    base: &Arc<UOp>,
    voffset: &Arc<UOp>,
    row_bytes: u16,
    elem: ScalarDType,
    w: usize,
    tag: usize,
) -> SmallVec<[Arc<UOp>; 8]> {
    assert!(row_bytes < 1 << 14, "the stride field holds 14 bits");
    let field = row_bytes | CACHE_SWIZZLE;
    let make = "llvm.amdgcn.make.buffer.rsrc.p8.p1";
    let rsrc = UOp::custom(
        smallvec![specific_ptr(base, AddrSpace::Global)],
        format!(
            "declare ptr addrspace(8) @{make}(ptr addrspace(1), i16, i32, i32)\n\
             call ptr addrspace(8) @{make}(ptr addrspace(1) {{0}}, i16 {field}, i32 -1, i32 {FLAGS})"
        ),
        DType::Void.ptr(None, AddrSpace::Global).expect("void is not a pointer"),
    )
    .rtag(Some(smallvec![tag]));
    let load = "llvm.amdgcn.raw.ptr.buffer.load.v4i32";
    let words = UOp::custom(
        smallvec![rsrc, voffset.clone()],
        format!(
            "declare <4 x i32> @{load}(ptr addrspace(8), i32, i32, i32)\n\
             call <4 x i32> @{load}(ptr addrspace(8) {{0}}, i32 {{1}}, i32 0, i32 0)"
        ),
        DType::Scalar(ScalarDType::Int32).vec(4).expect("four words"),
    )
    .rtag(Some(smallvec![tag]));
    let vec = DType::Scalar(elem).vec(w).expect("the run's elements");
    let ty = ldt(&vec);
    let run = UOp::custom(smallvec![words], format!("bitcast <4 x i32> {{0}} to {ty}"), vec).rtag(Some(smallvec![tag]));
    (0..w)
        .map(|i| {
            UOp::custom(smallvec![run.clone()], format!("extractelement {ty} {{0}}, i32 {i}"), DType::Scalar(elem))
        })
        .collect()
}
