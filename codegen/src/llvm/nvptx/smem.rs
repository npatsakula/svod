//! Shared-memory tile traffic for NVPTX: `cp.async` (asynchronous
//! global→shared copies, sm_80+) and `ldmatrix` (warp-collective 8x8 b16
//! fragment loads, sm_75+), as typed `Op::Custom` builders.
//!
//! Svod pointers stay generic (`ptr`) throughout the body, so every builder
//! `addrspacecast`s its operands to the `ptr addrspace(3)` / `ptr addrspace(1)`
//! the intrinsics require; LLVM's address-space inference folds those back onto
//! the addrspace(3) global / the kernel parameter at `-O3`. Intrinsic names and
//! signatures verified against clang 22 + ptxas at `sm_86`.

use std::sync::Arc;

use smallvec::{SmallVec, smallvec};
use svod_dtype::{AddrSpace, DType};
use svod_ir::prelude::*;

use crate::llvm::common::ldt;

/// `ptr` → `ptr addrspace(N)` for a pointer UOp whose provenance is `space`
/// (NVPTX numbers global as 1 and shared as 3).
fn specific_ptr(ptr: &Arc<UOp>, space: AddrSpace) -> Arc<UOp> {
    assert_eq!(ptr.addrspace(), Some(space), "pointer must resolve to a {space:?} buffer");
    let num = match space {
        AddrSpace::Global => 1,
        AddrSpace::Local => 3,
        AddrSpace::Reg => unreachable!("register scratch has no cp.async/ldmatrix form"),
    };
    let dtype = DType::Void.ptr(None, space).expect("void is not a pointer");
    UOp::custom(smallvec![ptr.clone()], format!("addrspacecast ptr {{0}} to ptr addrspace({num})"), dtype)
}

/// Cache policy of a `cp.async` copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CpAsyncCache {
    /// `.cg`: cache at L2 only (bypass L1); 16-byte copies only.
    Cg,
    /// `.ca`: allocate in L1 and L2; 4-, 8- or 16-byte copies.
    Ca,
}

/// `cp.async.{cg,ca}.shared.global [dst], [src], bytes`: this thread's
/// asynchronous `bytes`-wide copy (both addresses `bytes`-aligned). The data
/// lands in shared memory only once a [`cp_async_wait`] /
/// [`cp_async_wait_all`] retires it, and other threads see it after a barrier.
/// A `Void` statement: sequence its consumers with `.after([..])`.
pub fn cp_async(dst_shared: &Arc<UOp>, src_global: &Arc<UOp>, bytes: usize, cache: CpAsyncCache) -> Arc<UOp> {
    let cache = match (cache, bytes) {
        (CpAsyncCache::Cg, 16) => "cg",
        (CpAsyncCache::Ca, 4 | 8 | 16) => "ca",
        (cache, bytes) => panic!("cp.async.{cache:?} has no {bytes}-byte form (cg: 16; ca: 4/8/16)"),
    };
    UOp::custom(
        smallvec![specific_ptr(dst_shared, AddrSpace::Local), specific_ptr(src_global, AddrSpace::Global)],
        format!(
            "declare void @llvm.nvvm.cp.async.{cache}.shared.global.{bytes}(ptr addrspace(3), ptr addrspace(1))\n\
             call void @llvm.nvvm.cp.async.{cache}.shared.global.{bytes}(ptr addrspace(3) {{0}}, ptr addrspace(1) {{1}})"
        ),
        DType::Void,
    )
}

/// The tile-load form: one 16-byte `.cg` copy per thread.
pub fn cp_async_16(dst_shared: &Arc<UOp>, src_global: &Arc<UOp>) -> Arc<UOp> {
    cp_async(dst_shared, src_global, 16, CpAsyncCache::Cg)
}

fn void_call(intrinsic: &str, args: &str, params: &str, deps: SmallVec<[Arc<UOp>; 4]>) -> Arc<UOp> {
    UOp::custom(
        deps,
        format!("declare void @llvm.nvvm.{intrinsic}({params})\ncall void @llvm.nvvm.{intrinsic}({args})"),
        DType::Void,
    )
}

/// `cp.async.commit_group`: close the group of copies issued since the last
/// commit. `deps` are the copies (ordering only; nothing is referenced).
pub fn cp_async_commit(deps: SmallVec<[Arc<UOp>; 4]>) -> Arc<UOp> {
    void_call("cp.async.commit.group", "", "", deps)
}

/// `cp.async.wait_group N`: block until at most `pending` of this thread's
/// committed groups are still in flight (`0` retires everything).
pub fn cp_async_wait(pending: u32, deps: SmallVec<[Arc<UOp>; 4]>) -> Arc<UOp> {
    void_call("cp.async.wait.group", &format!("i32 {pending}"), "i32", deps)
}

/// `cp.async.wait_all`: retire every outstanding copy, committed or not.
pub fn cp_async_wait_all(deps: SmallVec<[Arc<UOp>; 4]>) -> Arc<UOp> {
    void_call("cp.async.wait.all", "", "", deps)
}

/// `ldmatrix.sync.aligned.m8n8.x{count}[.trans].shared.b16`: the warp
/// collectively loads `count` (1, 2 or 4) 8x8 b16 matrices from shared
/// memory, and every thread receives one 32-bit register per matrix holding
/// two adjacent b16 elements — thread `t` gets row `t / 4`, columns
/// `2 * (t % 4)` and `+1` (the `mma.sync` A/B fragment layout); `trans`
/// transposes each matrix on the way in.
///
/// Lane→address contract: each matrix's eight row addresses come from eight
/// consecutive lanes — lanes 0-7 give the rows of matrix 0, 8-15 of matrix 1,
/// 16-23 of matrix 2 and 24-31 of matrix 3 — and every row is 16 contiguous,
/// 16-byte-aligned bytes. Lanes beyond `8 * count` still supply an address,
/// which is ignored. `shared_ptr` is this lane's row address.
///
/// Returns the `count` fragments as `fragment`-typed values (any 4-byte dtype:
/// `Int32`, `<2 x half>`, `<2 x bfloat>`).
pub fn ldmatrix(shared_ptr: &Arc<UOp>, count: usize, trans: bool, fragment: DType) -> SmallVec<[Arc<UOp>; 4]> {
    assert!(matches!(count, 1 | 2 | 4), "ldmatrix loads 1, 2 or 4 matrices, not {count}");
    assert_eq!(fragment.bytes(), 4, "an ldmatrix fragment is one 32-bit register, not {fragment:?}");
    let intrinsic = format!("llvm.nvvm.ldmatrix.sync.aligned.m8n8.x{count}{}.b16", if trans { ".trans" } else { "" });
    // The aggregate's braces are doubled: `{`/`}` delimit CUSTOM placeholders.
    let ret_ty = if count == 1 { "i32".to_string() } else { format!("{{{{ {} }}}}", vec!["i32"; count].join(", ")) };
    let call = UOp::custom(
        smallvec![specific_ptr(shared_ptr, AddrSpace::Local)],
        format!("declare {ret_ty} @{intrinsic}(ptr addrspace(3))\ncall {ret_ty} @{intrinsic}(ptr addrspace(3) {{0}})"),
        // Nominal: an `{ i32, ... }` aggregate has no DType; only the
        // `extractvalue` customs below consume it.
        DType::Int32.vec(count).expect("i32 vectorizes"),
    );
    let as_fragment = |word: Arc<UOp>| if ldt(&fragment) == "i32" { word } else { word.bitcast(fragment.clone()) };
    if count == 1 {
        return smallvec![as_fragment(call)];
    }
    (0..count)
        .map(|i| {
            let word = UOp::custom(smallvec![call.clone()], format!("extractvalue {ret_ty} {{0}}, {i}"), DType::Int32);
            as_fragment(word)
        })
        .collect()
}

#[cfg(test)]
#[path = "../../test/unit/llvm_nvptx_smem.rs"]
mod tests;
