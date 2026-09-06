use smallvec::smallvec;
use svod_dtype::{AddrSpace, DType};
use svod_ir::{BinaryOp, UOp};

use super::*;
use crate::llvm::common::ldt;
use crate::llvm::nvptx::ops::tests::{SM86, assert_ptx_compiles, render_nvptx_linearized};
use crate::llvm::nvptx::shfl_idx;

fn lane() -> Arc<UOp> {
    UOp::special(UOp::native_const(32i32), "lidx0".to_string())
}

fn global_at(slot: usize, dtype: DType, index: Arc<UOp>) -> Arc<UOp> {
    UOp::index().buffer(UOp::param(slot, 256, dtype, None)).indices(vec![index]).call().unwrap()
}

fn shared_at(tile: &Arc<UOp>, index: Arc<UOp>) -> Arc<UOp> {
    UOp::index().buffer(tile.clone()).indices(vec![index]).call().unwrap()
}

fn imul(lhs: &Arc<UOp>, rhs: i32) -> Arc<UOp> {
    lhs.try_mul(&UOp::native_const(rhs)).unwrap()
}

/// Each lane copies one 16-byte chunk of a 16x16 bf16 tile, commits, waits,
/// and re-reads its chunk after the barrier.
fn copy_kernel(bytes: usize, cache: CpAsyncCache) -> Arc<UOp> {
    let l = lane();
    let tile = UOp::buffer(0, 256, DType::BFloat16, AddrSpace::Local, None);
    let chunk = imul(&l, 8);
    let copy = cp_async(&shared_at(&tile, chunk.clone()), &global_at(0, DType::BFloat16, chunk.clone()), bytes, cache);
    let wait = cp_async_wait(1, smallvec![cp_async_commit(smallvec![copy])]);
    let drained = cp_async_wait_all(smallvec![wait]);
    let barrier = drained.barrier(smallvec![]);
    let value = UOp::load().index(shared_at(&tile.after(smallvec![barrier]), chunk.clone())).call();
    UOp::sink(vec![global_at(1, DType::BFloat16, chunk).store(value)])
}

#[test]
fn cp_async_16_casts_pointers_to_their_address_spaces() {
    let rendered = render_nvptx_linearized(&copy_kernel(16, CpAsyncCache::Cg), SM86, "nvptx_cp_async");
    for needle in [
        "= addrspacecast ptr %",
        " to ptr addrspace(3)",
        " to ptr addrspace(1)",
        "call void @llvm.nvvm.cp.async.cg.shared.global.16(ptr addrspace(3) %",
        ", ptr addrspace(1) %",
        "call void @llvm.nvvm.cp.async.commit.group()",
        "call void @llvm.nvvm.cp.async.wait.group(i32 1)",
        "call void @llvm.nvvm.cp.async.wait.all()",
    ] {
        assert!(rendered.code.contains(needle), "missing {needle}:\n{}", rendered.code);
    }
    for decl in [
        "declare void @llvm.nvvm.cp.async.cg.shared.global.16(ptr addrspace(3), ptr addrspace(1))",
        "declare void @llvm.nvvm.cp.async.commit.group()",
        "declare void @llvm.nvvm.cp.async.wait.group(i32)",
        "declare void @llvm.nvvm.cp.async.wait.all()",
    ] {
        assert_eq!(rendered.code.matches(decl).count(), 1, "{decl} hoisted once:\n{}", rendered.code);
    }
    let calls: Vec<&str> = rendered.code.lines().map(str::trim).filter(|line| line.contains("call ")).collect();
    let at = |needle: &str| calls.iter().position(|line| line.contains(needle)).unwrap();
    let (copy, commit, wait, drain, sync) = (
        at("cp.async.cg.shared.global.16("),
        at("cp.async.commit.group()"),
        at("cp.async.wait.group(i32 1)"),
        at("cp.async.wait.all()"),
        at("@llvm.nvvm.barrier0()"),
    );
    assert!(copy < commit && commit < wait && wait < drain && drain < sync, "{}", rendered.code);
    if let Some(ptx) = assert_ptx_compiles(&rendered.code, SM86) {
        for needle in [
            "cp.async.cg.shared.global",
            "cp.async.commit_group",
            "cp.async.wait_group 	1",
            "cp.async.wait_all",
            "bar.sync",
            "ld.shared",
        ] {
            assert!(ptx.contains(needle), "missing {needle}:\n{ptx}");
        }
        assert!(!ptx.contains("cvta.to.shared"), "the shared cast must fold onto the global:\n{ptx}");
    }
}

#[test_case::test_case(4; "ca 4")]
#[test_case::test_case(8; "ca 8")]
#[test_case::test_case(16; "ca 16")]
fn cp_async_ca_selects_the_sized_form(bytes: usize) {
    let rendered = render_nvptx_linearized(&copy_kernel(bytes, CpAsyncCache::Ca), SM86, "nvptx_cp_async_ca");
    let call = format!("call void @llvm.nvvm.cp.async.ca.shared.global.{bytes}(ptr addrspace(3) %");
    assert!(rendered.code.contains(&call), "missing {call}:\n{}", rendered.code);
    if let Some(ptx) = assert_ptx_compiles(&rendered.code, SM86) {
        assert!(ptx.contains("cp.async.ca.shared.global [%rd") && ptx.contains(&format!("], {bytes};")), "{ptx}");
    }
}

#[test]
#[should_panic(expected = "no 8-byte form")]
fn cp_async_cg_is_16_bytes_only() {
    copy_kernel(8, CpAsyncCache::Cg);
}

#[test]
#[should_panic(expected = "must resolve to a Local buffer")]
fn cp_async_rejects_a_global_destination() {
    let src = global_at(0, DType::BFloat16, lane());
    cp_async_16(&src, &src);
}

#[test]
#[should_panic(expected = "must resolve to a Global buffer")]
fn cp_async_rejects_a_shared_source() {
    let tile = UOp::buffer(0, 256, DType::BFloat16, AddrSpace::Local, None);
    let dst = shared_at(&tile, lane());
    cp_async_16(&dst, &dst);
}

/// Lane `l` addresses row `l % 16` at column block `l / 16` of a row-major
/// 16x16 b16 tile: lanes 0-7 / 8-15 rows 0-15 of columns 0-7 (matrices 0/1),
/// 16-23 / 24-31 the same rows of columns 8-15 (matrices 2/3).
fn row_address(tile: &Arc<UOp>, l: &Arc<UOp>) -> Arc<UOp> {
    let row = UOp::alu(BinaryOp::CMod, l.clone(), UOp::native_const(16i32));
    let block = UOp::alu(BinaryOp::CDiv, l.clone(), UOp::native_const(16i32));
    shared_at(tile, imul(&row, 16).try_add(&imul(&block, 8)).unwrap())
}

fn ldmatrix_kernel(count: usize, trans: bool, fragment: DType) -> Arc<UOp> {
    let l = lane();
    let tile = UOp::buffer(0, 256, DType::BFloat16, AddrSpace::Local, None);
    let fill = shared_at(&tile, l.clone()).store(UOp::load().index(global_at(0, DType::BFloat16, l.clone())).call());
    let ready = tile.after(smallvec![fill.barrier(smallvec![])]);
    let frags = ldmatrix(&row_address(&ready, &l), count, trans, fragment.clone());
    assert_eq!(frags.len(), count);
    let stores = frags
        .iter()
        .enumerate()
        .map(|(i, frag)| {
            let out =
                global_at(1, fragment.clone(), imul(&l, count as i32).try_add(&UOp::native_const(i as i32)).unwrap());
            out.store(frag.clone())
        })
        .collect();
    UOp::sink(stores)
}

#[test_case::test_case(1, false, DType::Int32, "i32"; "x1 i32")]
#[test_case::test_case(2, false, DType::UInt32, "{ i32, i32 }"; "x2 u32")]
#[test_case::test_case(4, false, DType::Float16.vec(2).unwrap(), "{ i32, i32, i32, i32 }"; "x4 half pairs")]
#[test_case::test_case(4, true, DType::BFloat16.vec(2).unwrap(), "{ i32, i32, i32, i32 }"; "x4 trans bf16 pairs")]
fn ldmatrix_forms(count: usize, trans: bool, fragment: DType, ret_ty: &str) {
    let rendered = render_nvptx_linearized(&ldmatrix_kernel(count, trans, fragment.clone()), SM86, "nvptx_ldmatrix");
    let intrinsic = format!("llvm.nvvm.ldmatrix.sync.aligned.m8n8.x{count}{}.b16", if trans { ".trans" } else { "" });
    let call = format!("call {ret_ty} @{intrinsic}(ptr addrspace(3) %");
    assert!(rendered.code.contains(&call), "missing {call}:\n{}", rendered.code);
    let decl = format!("declare {ret_ty} @{intrinsic}(ptr addrspace(3))");
    assert_eq!(rendered.code.matches(&decl).count(), 1, "{decl} hoisted once:\n{}", rendered.code);
    // One `extractvalue` per aggregate member, none for the scalar `x1` form.
    let extracted: Vec<&str> = rendered
        .code
        .lines()
        .filter(|line| line.contains(&format!("= extractvalue {ret_ty} %")))
        .map(|line| line.rsplit(", ").next().unwrap())
        .collect();
    let expected: Vec<String> = if count > 1 { (0..count).map(|i| i.to_string()).collect() } else { Vec::new() };
    assert_eq!(extracted, expected, "{}", rendered.code);
    if fragment.is_vector() {
        assert_eq!(rendered.code.matches("bitcast i32 %").count(), count, "{}", rendered.code);
        assert!(rendered.code.contains(&format!("to {}", ldt(&fragment))), "{}", rendered.code);
    } else {
        assert!(!rendered.code.contains("bitcast i32"), "{}", rendered.code);
    }
    if let Some(ptx) = assert_ptx_compiles(&rendered.code, SM86) {
        let needle = format!("ldmatrix.sync.aligned.m8n8.x{count}{}.shared.b16", if trans { ".trans" } else { "" });
        assert!(ptx.contains(&needle), "missing {needle}:\n{ptx}");
    }
}

#[test]
#[should_panic(expected = "1, 2 or 4 matrices")]
fn ldmatrix_rejects_other_counts() {
    let tile = UOp::buffer(0, 256, DType::BFloat16, AddrSpace::Local, None);
    ldmatrix(&shared_at(&tile, lane()), 3, false, DType::Int32);
}

#[test]
#[should_panic(expected = "one 32-bit register")]
fn ldmatrix_rejects_narrow_fragments() {
    let tile = UOp::buffer(0, 256, DType::BFloat16, AddrSpace::Local, None);
    ldmatrix(&shared_at(&tile, lane()), 4, false, DType::BFloat16);
}

/// The flash-attention tile prologue end to end: every lane `cp.async`s its
/// 16-byte chunk of a 16x16 bf16 tile global→shared, commits and waits, the
/// warp `ldmatrix.x4`s the tile into mma fragments, and the first fragment is
/// exchanged with the neighbouring lane. ptxas must accept the result and the
/// PTX must carry each primitive.
#[test]
fn tile_prologue_compiles_to_ptx() {
    let l = lane();
    let tile = UOp::buffer(0, 256, DType::BFloat16, AddrSpace::Local, None);
    let chunk = imul(&l, 8);
    let copy = cp_async_16(&shared_at(&tile, chunk.clone()), &global_at(0, DType::BFloat16, chunk));
    let wait = cp_async_wait(0, smallvec![cp_async_commit(smallvec![copy])]);
    let ready = tile.after(smallvec![wait.barrier(smallvec![])]);
    let frags = ldmatrix(&row_address(&ready, &l), 4, false, DType::Int32);
    let neighbour = UOp::alu(BinaryOp::Xor, l.clone(), UOp::native_const(1i32));
    let swapped = shfl_idx(&frags[0], &neighbour);
    let sum = frags[1..].iter().fold(swapped, |acc, frag| acc.try_add(frag).unwrap());
    let sink = UOp::sink(vec![global_at(1, DType::Int32, l).store(sum)]);

    let rendered = render_nvptx_linearized(&sink, SM86, "nvptx_tile_prologue");
    let Some(ptx) = assert_ptx_compiles(&rendered.code, SM86) else { return };
    for needle in [
        "cp.async.cg.shared.global",
        "cp.async.commit_group",
        "cp.async.wait_group",
        "bar.sync",
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16",
        "shfl.sync.idx.b32",
    ] {
        assert!(ptx.contains(needle), "missing {needle}:\n{ptx}");
    }
    assert!(!ptx.contains("cvta.to.shared"), "the shared casts must fold onto the tile global:\n{ptx}");
}
