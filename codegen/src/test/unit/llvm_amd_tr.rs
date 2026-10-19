use svod_dtype::{AddrSpace, AmdArch, DType, ScalarDType};
use svod_ir::{BinaryOp, UOp};

use super::*;
use crate::llvm::text::tests::{assert_amd_ir_compiles, render_amd_linearized};

fn lane() -> Arc<UOp> {
    UOp::special(UOp::native_const(32i32), "lidx0".to_string())
}

fn global_at(slot: usize, dtype: DType, index: Arc<UOp>) -> Arc<UOp> {
    UOp::index().buffer(UOp::param(slot, 256, dtype, None)).indices(vec![index]).call().unwrap()
}

/// Lane `L` addresses row `L % 16` at column `8·(L/16)` of a row-major 16×16
/// tile and writes its eight transposed elements out in lane order.
fn tr_kernel(elem: ScalarDType) -> Arc<UOp> {
    let l = lane();
    let row = UOp::alu(BinaryOp::CMod, l.clone(), UOp::native_const(16i32));
    let half = UOp::alu(BinaryOp::CDiv, l.clone(), UOp::native_const(16i32));
    let at = UOp::alu(BinaryOp::Mul, row, UOp::native_const(16i32))
        .try_add(&UOp::alu(BinaryOp::Mul, half, UOp::native_const(8i32)))
        .unwrap();
    let elems = global_load_tr_b128(&global_at(0, DType::Scalar(elem), at), elem);
    let stores = elems
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let out = UOp::alu(BinaryOp::Mul, l.clone(), UOp::native_const(8i32))
                .try_add(&UOp::native_const(i as i32))
                .unwrap();
            global_at(1, DType::Scalar(elem), out).store(e.clone())
        })
        .collect();
    UOp::sink(stores)
}

#[test_case::test_case(ScalarDType::BFloat16, "<8 x bfloat>", "v8bf16"; "bf16")]
#[test_case::test_case(ScalarDType::Float16, "<8 x half>", "v8f16"; "f16")]
fn transposing_loads_take_a_global_pointer(elem: ScalarDType, ty: &str, overload: &str) {
    let code = render_amd_linearized(&tr_kernel(elem), AmdArch::Gfx1201, "amd_tr").code;
    let intrinsic = format!("llvm.amdgcn.global.load.tr.b128.{overload}");
    assert_eq!(code.matches(&format!("declare {ty} @{intrinsic}(ptr addrspace(1))")).count(), 1, "{code}");
    assert!(code.contains(&format!("call {ty} @{intrinsic}(ptr addrspace(1) %")), "{code}");
    assert!(code.contains("addrspacecast ptr %"), "{code}");
    assert_eq!(code.matches(&format!("= extractelement {ty} %")).count(), 8, "{code}");
    assert_amd_ir_compiles(&code, "gfx1201");
}

#[test]
#[should_panic(expected = "16-bit elements")]
fn transposing_loads_move_16_bit_elements() {
    global_load_tr_b128(&global_at(0, DType::Float32, lane()), ScalarDType::Float32);
}

#[test]
#[should_panic(expected = "Global")]
fn transposing_loads_reject_a_shared_source() {
    let tile = UOp::buffer(0, 256, DType::BFloat16, AddrSpace::Local, None);
    let at = UOp::index().buffer(tile).indices(vec![lane()]).call().unwrap();
    global_load_tr_b128(&at, ScalarDType::BFloat16);
}
