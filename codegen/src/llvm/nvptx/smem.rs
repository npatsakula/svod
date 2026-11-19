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

use crate::llvm::common::gpu::specific_ptr;
use crate::llvm::common::ldt;

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
    let cache = cache_suffix(cache, bytes);
    UOp::custom(
        smallvec![specific_ptr(dst_shared, AddrSpace::Local), specific_ptr(src_global, AddrSpace::Global)],
        format!(
            "declare void @llvm.nvvm.cp.async.{cache}.shared.global.{bytes}(ptr addrspace(3), ptr addrspace(1))\n\
             call void @llvm.nvvm.cp.async.{cache}.shared.global.{bytes}(ptr addrspace(3) {{0}}, ptr addrspace(1) {{1}})"
        ),
        DType::Void,
    )
}

/// The `.cg`/`.ca` suffix for a copy of `bytes`, panicking on a width the cache
/// policy has no form for.
fn cache_suffix(cache: CpAsyncCache, bytes: usize) -> &'static str {
    match (cache, bytes) {
        (CpAsyncCache::Cg, 16) => "cg",
        (CpAsyncCache::Ca, 4 | 8 | 16) => "ca",
        (cache, bytes) => panic!("cp.async.{cache:?} has no {bytes}-byte form (cg: 16; ca: 4/8/16)"),
    }
}

/// The tile-load form: one 16-byte `.cg` copy per thread.
pub fn cp_async_16(dst_shared: &Arc<UOp>, src_global: &Arc<UOp>) -> Arc<UOp> {
    cp_async(dst_shared, src_global, 16, CpAsyncCache::Cg)
}

/// `cp.async.{cg,ca}.shared.global [dst], [src], bytes, src_size`: the
/// zero-filling form of [`cp_async`]. Only the first `src_size` bytes are read
/// from global memory; the copy engine fills the rest of the `bytes`-wide
/// destination with zeros, without a read and without the value ever reaching a
/// register. `src_bytes` is a runtime `i32`, so a **predicated** copy is
/// `src_size = 0`: the whole chunk lands as zeros and global memory is not
/// touched at all.
///
/// # Panics
/// Panics unless `src_bytes` is an `i32` value (the intrinsic's operand type).
pub fn cp_async_zfill(
    dst_shared: &Arc<UOp>,
    src_global: &Arc<UOp>,
    bytes: usize,
    src_bytes: &Arc<UOp>,
    cache: CpAsyncCache,
) -> Arc<UOp> {
    assert_eq!(src_bytes.dtype(), DType::Int32, "cp.async src_size must be i32");
    let cache = cache_suffix(cache, bytes);
    UOp::custom(
        smallvec![
            specific_ptr(dst_shared, AddrSpace::Local),
            specific_ptr(src_global, AddrSpace::Global),
            src_bytes.clone()
        ],
        format!(
            "declare void @llvm.nvvm.cp.async.{cache}.shared.global.{bytes}.s(ptr addrspace(3), ptr addrspace(1), i32)\n\
             call void @llvm.nvvm.cp.async.{cache}.shared.global.{bytes}.s(ptr addrspace(3) {{0}}, ptr addrspace(1) {{1}}, i32 {{2}})"
        ),
        DType::Void,
    )
}

/// The tile-load form of [`cp_async_zfill`]: one predicated 16-byte `.cg` copy
/// per thread.
pub fn cp_async_16_zfill(dst_shared: &Arc<UOp>, src_global: &Arc<UOp>, src_bytes: &Arc<UOp>) -> Arc<UOp> {
    cp_async_zfill(dst_shared, src_global, 16, src_bytes, CpAsyncCache::Cg)
}

/// `ld.global.cg.v4.u32 {..}, [src]`: this thread's 16-byte global load
/// under the policy of a `.cg` [`cp_async`] (cached at L2 only, bypassing
/// L1), into four 32-bit words; the register half of a fill on a target
/// whose fill is `cp.async.cg`. Ordered like a copy (a side effect), so two
/// loads of one address stay two loads.
pub fn ld_global_cg_16(src_global: &Arc<UOp>) -> [Arc<UOp>; 4] {
    // The aggregate's braces are doubled: `{`/`}` delimit CUSTOM placeholders.
    let ret_ty = "{{ i32, i32, i32, i32 }}";
    let call = UOp::custom(
        smallvec![specific_ptr(src_global, AddrSpace::Global)],
        format!(
            "call {ret_ty} asm sideeffect \"ld.global.cg.v4.u32 {{{{$0, $1, $2, $3}}}}, [$4];\", \"=r,=r,=r,=r,l\"(ptr addrspace(1) {{0}})"
        ),
        // Nominal, as for `ldmatrix`: only the `extractvalue`s consume it.
        DType::Int32.vec(4).expect("i32 vectorizes"),
    );
    std::array::from_fn(|i| {
        UOp::custom(smallvec![call.clone()], format!("extractvalue {ret_ty} {{0}}, {i}"), DType::Int32)
    })
}

/// `mbarrier.init`: the 64-bit barrier object at `bar` (shared memory)
/// completes a phase once `arrivals` arrivals and every byte they expect
/// have come in. One thread initializes it; the others see it past a block
/// barrier. sm_80+; the forms below, sm_90.
pub fn mbarrier_init(bar: &Arc<UOp>, arrivals: u32) -> Arc<UOp> {
    UOp::custom(
        smallvec![specific_ptr(bar, AddrSpace::Local)],
        format!(
            "declare void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3), i32)\n\
             call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) {{0}}, i32 {arrivals})"
        ),
        DType::Void,
    )
}

/// `mbarrier.arrive.expect_tx`: this thread's arrival at `bar`, expecting
/// `bytes` of bulk copies that name the barrier: the phase completes once
/// they have landed.
pub fn mbarrier_arrive_expect(bar: &Arc<UOp>, bytes: u32) -> Arc<UOp> {
    let intrinsic = "llvm.nvvm.mbarrier.arrive.expect.tx.scope.cta.space.cta";
    UOp::custom(
        smallvec![specific_ptr(bar, AddrSpace::Local)],
        format!(
            "declare i64 @{intrinsic}(ptr addrspace(3), i32)\n\
             call i64 @{intrinsic}(ptr addrspace(3) {{0}}, i32 {bytes})"
        ),
        DType::Int64,
    )
}

/// `mbarrier.try_wait.parity` until it succeeds: this thread waits for the
/// phase of `bar` whose parity is `parity` (an `i32`, 0 or 1; a barrier's
/// phases alternate from 0). One statement holds the retry loop: the
/// intrinsic is a single try.
///
/// # Panics
/// Panics unless `parity` is an `i32` value.
pub fn mbarrier_wait(bar: &Arc<UOp>, parity: &Arc<UOp>) -> Arc<UOp> {
    assert_eq!(parity.dtype(), DType::Int32, "an mbarrier phase parity is i32");
    let address = UOp::custom(
        smallvec![specific_ptr(bar, AddrSpace::Local)],
        "ptrtoint ptr addrspace(3) {0} to i32".to_string(),
        DType::Int32,
    );
    // The braces scope the predicate and the label, so every wait of a
    // kernel may spell them alike; doubled, as CUSTOM placeholders are `{N}`.
    let wait =
        "{{\\0A.reg .pred p;\\0AWAIT: mbarrier.try_wait.parity.shared::cta.b64 p, [$0], $1;\\0A@!p bra WAIT;\\0A}}";
    UOp::custom(
        smallvec![address, parity.clone()],
        format!("call void asm sideeffect \"{wait}\", \"r,r,~{{{{memory}}}}\"(i32 {{0}}, i32 {{1}})"),
        DType::Void,
    )
}

/// `cp.async.bulk.tensor.2d.shared::cluster.global.tile`: this thread's
/// bulk copy of the box of tensor map `map` at column `at[0]`, row `at[1]`
/// (`i32`s) into shared memory at `dst_shared`, its bytes completing on the
/// barrier `bar` ([`mbarrier_arrive_expect`]). `map` is a tensor-map PARAM
/// and `buffer` the storage PARAM it maps, which the copy keeps a source.
/// sm_90.
///
/// # Panics
/// Panics unless both coordinates are `i32` values.
pub fn bulk_tensor_load(
    dst_shared: &Arc<UOp>,
    bar: &Arc<UOp>,
    map: &Arc<UOp>,
    at: [&Arc<UOp>; 2],
    buffer: &Arc<UOp>,
) -> Arc<UOp> {
    assert!(at.iter().all(|c| c.dtype() == DType::Int32), "a tensor-map coordinate is i32");
    // The destination is in the cluster's shared window (addrspace 7), of
    // which the block's own is a part.
    let dst = UOp::custom(
        smallvec![specific_ptr(dst_shared, AddrSpace::Local)],
        "addrspacecast ptr addrspace(3) {0} to ptr addrspace(7)".to_string(),
        DType::Void.ptr(None, AddrSpace::Local).expect("void is not a pointer"),
    );
    let intrinsic = "llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d";
    UOp::custom(
        smallvec![dst, specific_ptr(bar, AddrSpace::Local), map.clone(), at[0].clone(), at[1].clone(), buffer.clone()],
        format!(
            "declare void @{intrinsic}(ptr addrspace(7), ptr addrspace(3), ptr, i32, i32, i16, i64, i1, i1, i32)\n\
             call void @{intrinsic}(ptr addrspace(7) {{0}}, ptr addrspace(3) {{1}}, ptr {{2}}, i32 {{3}}, i32 {{4}}, \
             i16 0, i64 0, i1 false, i1 false, i32 0)"
        ),
        DType::Void,
    )
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

/// One thread's bulk copy of a shared tile into global memory through the
/// tensor map `map` (`cp.async.bulk.tensor`, shared → global), its box at
/// column `at[0]`, row `at[1]` of the tensor; what of the box lies past the
/// tensor is not written. `written` is an index into the parameter the map
/// describes, which the statement so writes in place
/// ([`UOp::read_modify_writes`]). The copy joins the thread's bulk group
/// ([`bulk_store_wait`]).
pub fn bulk_tensor_store(src_shared: &Arc<UOp>, map: &Arc<UOp>, at: [&Arc<UOp>; 2], written: &Arc<UOp>) -> Arc<UOp> {
    assert!(matches!(written.op(), Op::Index(..)), "a bulk store names the memory it writes by an index");
    assert!(at.iter().all(|c| c.dtype() == DType::Int32), "a tensor-map coordinate is i32");
    let intrinsic = "llvm.nvvm.cp.async.bulk.tensor.s2g.tile.2d";
    UOp::custom(
        smallvec![
            specific_ptr(src_shared, AddrSpace::Local),
            map.clone(),
            at[0].clone(),
            at[1].clone(),
            written.clone()
        ],
        format!(
            "declare void @{intrinsic}(ptr addrspace(3), ptr, i32, i32, i64, i1)\n\
             call void @{intrinsic}(ptr addrspace(3) {{0}}, ptr {{1}}, i32 {{2}}, i32 {{3}}, i64 0, i1 false)"
        ),
        DType::Void,
    )
}

/// What the generic proxy stored in shared memory at and past `shared`,
/// fenced for the bulk copies that read it through the async proxy
/// (`fence.proxy.async`).
pub fn bulk_store_fence(shared: &Arc<UOp>) -> Arc<UOp> {
    UOp::custom(
        smallvec![specific_ptr(shared, AddrSpace::Local)],
        "call void asm sideeffect \"fence.proxy.async.shared::cta;\", \"~{{{{memory}}}}\"()".to_string(),
        DType::Void,
    )
}

/// The thread's bulk stores so far, committed as a group and complete.
pub fn bulk_store_wait(last: &Arc<UOp>) -> Arc<UOp> {
    UOp::custom(
        smallvec![last.clone()],
        "call void asm sideeffect \"cp.async.bulk.commit_group;\\0Acp.async.bulk.wait_group 0;\", \"~{{{{memory}}}}\"()"
            .to_string(),
        DType::Void,
    )
}
