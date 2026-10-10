//! Lowering helpers shared by the GPU targets (AMD and NVPTX).
//!
//! Both backends agree on the pieces below: the SPECIAL axis-name grammar, the
//! addrspace(3) module global that backs a LOCAL buffer, the work-group upper
//! bound derived from the kernel's `l` axes, and the vector type a WMMA
//! operand is passed as.

use std::sync::Arc;

use svod_dtype::{AddrSpace, DType};
use svod_ir::{Op, ops, prelude::*};

use super::{RenderContext, ldt};
use smallvec::smallvec;

/// Parse a SPECIAL axis name: `'g'/'l'/'i'` prefix + 0/1/2 axis suffix.
///
/// Matches the names `schedule/src/gpudims.rs` mints (`gidx{i}` / `lidx{i}`),
/// which is the producer side for these strings.
pub fn parse_special_axis(name: &str) -> Option<(char, u8)> {
    let prefix = name.chars().next()?;
    if !matches!(prefix, 'g' | 'l' | 'i') {
        return None;
    }
    let suffix_start = name.rfind(|c: char| !c.is_ascii_digit()).map(|i| i + 1).unwrap_or(0);
    if suffix_start == name.len() {
        return None;
    }
    let axis: u8 = name[suffix_start..].parse().ok()?;
    (axis < 3).then_some((prefix, axis))
}

/// Hardware dimension letter of a SPECIAL axis index.
pub const AXIS_LETTERS: [char; 3] = ['x', 'y', 'z'];

/// Upper bound on the work-group size per hardware axis: the `l` SPECIAL
/// bounds (`[1, 1, 1]` for a kernel without local axes). Tinygrad
/// `llvmir.py:259-263`; both GPU backends hand this to the compiler so it
/// sizes registers / scratch for the real launch shape
/// (`amdgpu-flat-work-group-size`, `nvvm.maxntid`). `None` when a bound is
/// not a constant: the compiler then keeps its hardware maximum, since a
/// bound the launch exceeds is a launch failure, not a hint.
pub fn local_bounds(nodes: &[Arc<UOp>]) -> Option<[u64; 3]> {
    let mut bounds = [1u64; 3];
    for node in nodes {
        if let Op::Special(ops::Special { name, end }) = node.op()
            && let Some(('l', axis)) = parse_special_axis(name)
        {
            bounds[axis as usize] = match end.vmax() {
                svod_ir::ConstValue::Int(v) => *v as u64,
                svod_ir::ConstValue::UInt(v) => *v,
                _ => return None,
            };
        }
    }
    Some(bounds)
}

/// LOCAL BUFFER → addrspace(3) module-level global, exposed to the body as a
/// generic pointer.
///
/// The global is declared in the module prefix; the body `addrspacecast`s it
/// to `ptr` so downstream GEP/LOAD/STORE keep the generic pointer type they
/// use everywhere else (a bare `ptr addrspace(3)` operand is a type error
/// there). Both backends recover the fast `ds_*` / `ld.shared` forms through
/// LLVM's address-space inference at `-O3`.
pub fn render_define_local(uop: &Arc<UOp>, ctx: &mut RenderContext, kernel: &mut Vec<String>) -> Option<()> {
    let dst = ctx.name(uop); // e.g. "%local42"
    let (id, base_dtype) = match uop.op() {
        Op::Buffer(ops::Buffer { arg, .. }) if arg.addrspace == Some(svod_ir::AddrSpace::Local) => {
            (arg.slot, arg.dtype.clone())
        }
        _ => unreachable!("render_define_local requires a LOCAL buffer"),
    };
    let size = uop.buffer_size().unwrap_or(1);
    let base_ty = ldt(&base_dtype);
    let global_name = format!("@local{id}");
    ctx.push_module_prefix(format!(
        "{global_name} = internal unnamed_addr addrspace(3) global [{size} x {base_ty}] undef, align 16"
    ));
    kernel.push(format!("  {dst} = addrspacecast ptr addrspace(3) {global_name} to ptr"));
    Some(())
}

/// The LLVM vector type a WMMA operand or result is carried in: its scalar
/// dtype widened to the lane count its shape carries (`<16 x half>`,
/// `<8 x float>`). `DType::scalar()` is `None` for vectors, so this goes
/// through `base()`.
pub fn wmma_operand_dtype(uop: &Arc<UOp>) -> DType {
    let dtype = uop.dtype();
    let count = uop
        .shape()
        .ok()
        .flatten()
        .and_then(|shape| shape.iter().try_fold(1usize, |count, dim| Some(count * dim.as_const()?)))
        .unwrap_or(1);
    if count > 1 { dtype.scalar_dtype().vec(count).expect("WMMA shape must be vectorizable") } else { dtype }
}

#[cfg(test)]
#[path = "../../test/unit/llvm_common_gpu.rs"]
mod tests;

/// `ptr` → `ptr addrspace(N)` for a pointer UOp whose provenance is `space`:
/// Svod pointers stay generic through a kernel body, and NVPTX and AMDGPU
/// both number global as 1 and shared as 3, so an intrinsic that wants a
/// specific space gets a cast LLVM folds back onto the buffer at `-O3`.
pub fn specific_ptr(ptr: &Arc<UOp>, space: AddrSpace) -> Arc<UOp> {
    assert_eq!(ptr.addrspace(), Some(space), "pointer must resolve to a {space:?} buffer");
    let num = match space {
        AddrSpace::Global => 1,
        AddrSpace::Local => 3,
        AddrSpace::Reg => unreachable!("register scratch has no address-space form"),
    };
    let dtype = DType::Void.ptr(None, space).expect("void is not a pointer");
    UOp::custom(smallvec![ptr.clone()], format!("addrspacecast ptr {{0}} to ptr addrspace({num})"), dtype)
}
