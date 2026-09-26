//! Pure graph-shape tests for the register-tile math ops (`TileMathMixin`).

use svod_dtype::{AmdArch, CudaArch, DType, GpuArch};
use svod_ir::{BinaryOp, Op, UnaryOp, ops};
use test_case::test_case;

use crate::tiles::{RT_16X16, TileLayout, VecLayout};
use crate::{ArchCaps, Kernel};

const GFX942: GpuArch = GpuArch::Amd(AmdArch::Gfx942);
const GFX1201: GpuArch = GpuArch::Amd(AmdArch::Gfx1201);
const SM_86: GpuArch = GpuArch::Cuda(CudaArch::from_compute_capability(8, 6));

fn probe() -> Kernel {
    Kernel::new("math_probe", [1, 1, 1], 64, vec![], crate::ArchCaps::GFX942)
}

fn probe_on(arch: GpuArch) -> Kernel {
    let caps = ArchCaps::for_arch(arch);
    Kernel::new("math_probe", [1, 1, 1], caps.wave_size as i64, vec![], caps)
}

/// The LLVM call a typed `Custom` renders, when the graph holds one.
fn custom_call(uop: &svod_ir::UOp) -> Option<&str> {
    match uop.op() {
        Op::Custom(ops::Custom { code, .. }) => code.lines().find(|l| l.starts_with("call ")),
        _ => None,
    }
}

/// `exp2` maps an `Exp2` unary over the tile, except an f32 tile on AMD, which
/// takes the bare `v_exp_f32` (no denormal fix-up) as a typed `Custom`.
#[test_case(GFX942, DType::Float32, Some("call float @llvm.amdgcn.exp2.f32(float {0})"); "gfx942 f32 flushes")]
#[test_case(GFX1201, DType::Float32, Some("call float @llvm.amdgcn.exp2.f32(float {0})"); "gfx1201 f32 flushes")]
#[test_case(GFX1201, DType::BFloat16, None; "gfx1201 bf16 keeps exp2")]
#[test_case(SM_86, DType::Float32, None; "sm_86 f32 keeps exp2")]
fn test_exp2_lowering(arch: GpuArch, dtype: DType, custom: Option<&str>) {
    let ker = probe_on(arch);
    let warp = ker.warp();
    let a = warp.zero(ker.rt((16, 16), dtype, TileLayout::Row, RT_16X16));
    let topo = warp.exp2(a).uop().toposort();
    let calls: Vec<&str> = topo.iter().filter_map(|u| custom_call(u)).collect();
    let unary = topo.iter().any(|u| matches!(u.op(), Op::Unary(UnaryOp::Exp2, _)));
    match custom {
        Some(call) => assert!(calls == [call] && !unary, "{arch:?}: want only {call}, got {calls:?}, Exp2 {unary}"),
        None => assert!(calls.is_empty() && unary, "{arch:?}: want Unary(Exp2), got {calls:?}"),
    }
}

/// `max_num` is the native `maxnum` on gfx12 f32 only; elsewhere the `Max` the
/// renderers decompose into a compare and a select.
#[test_case(GFX1201, DType::Float32, true; "gfx1201 f32 native")]
#[test_case(GFX1201, DType::Int32, false; "gfx1201 i32 max")]
#[test_case(GFX942, DType::Float32, false; "gfx942 f32 max")]
#[test_case(SM_86, DType::Float32, false; "sm_86 f32 max")]
fn test_max_num_lowering(arch: GpuArch, dtype: DType, native: bool) {
    let ker = probe_on(arch);
    let (x, y) = (
        svod_ir::UOp::const_(dtype.clone(), svod_ir::ConstValue::Int(1)),
        svod_ir::UOp::const_(dtype, svod_ir::ConstValue::Int(2)),
    );
    let out = ker.warp().max_num(&x, &y);
    if native {
        assert_eq!(custom_call(&out), Some("call float @llvm.maxnum.f32(float {0}, float {1})"));
        assert!(out.op().sources().iter().map(|s| s.id).eq([x.id, y.id]), "maxnum reads x then y");
    } else {
        assert!(matches!(out.op(), Op::Binary(BinaryOp::Max, _, _)), "{arch:?}: want Max, got {:?}", out.op());
    }
}

/// `mul_scalar` folds the scalar into a `Mul` against a constant.
#[test]
fn test_mul_scalar_emits_mul_const() {
    let ker = probe();
    let warp = ker.warp();
    let a = warp.zero(ker.rt((16, 16), DType::Float32, TileLayout::Row, RT_16X16));
    let out = warp.mul_scalar(a, 1.5);
    let topo = out.uop().toposort();
    assert!(topo.iter().any(|u| matches!(u.op(), Op::Binary(BinaryOp::Mul, _, _))), "mul_scalar emits a Mul");
    assert!(topo.iter().any(|u| matches!(u.op(), Op::Const(_))), "mul_scalar references a constant operand");
}

/// `div` is `mul(reciprocal)`, not a raw `Fdiv` (faithful to the mixin).
#[test]
fn test_div_is_mul_reciprocal() {
    let ker = probe();
    let warp = ker.warp();
    let a = warp.zero(ker.rt((16, 16), DType::Float32, TileLayout::Row, RT_16X16));
    let b = warp.ones(ker.rt((16, 16), DType::Float32, TileLayout::Row, RT_16X16));
    let out = warp.div(a, &b);
    let topo = out.uop().toposort();
    assert!(topo.iter().any(|u| matches!(u.op(), Op::Unary(UnaryOp::Reciprocal, _))), "div lowers to mul(reciprocal)");
    assert!(topo.iter().any(|u| matches!(u.op(), Op::Binary(BinaryOp::Mul, _, _))), "div uses a Mul");
    assert!(!topo.iter().any(|u| matches!(u.op(), Op::Binary(BinaryOp::Fdiv, _, _))), "div is not a raw Fdiv");
}

/// `sub_rv` broadcasts the register vector into the RT (`add(neg)`), reading the
/// vector once per output element.
#[test]
fn test_sub_rv_broadcast_is_add_neg() {
    let ker = probe();
    let warp = ker.warp();
    let a = warp.zero(ker.rt((32, 32), DType::Float32, TileLayout::Row, RT_16X16));
    let v = warp.zero_rv(ker.rv(32, DType::Float32, VecLayout::Ortho, RT_16X16));
    let out = warp.sub_rv(a, &v);
    let topo = out.uop().toposort();
    // sub = add(neg): the neg is MUL(-1); the combine is an Add.
    assert!(topo.iter().any(|u| matches!(u.op(), Op::Binary(BinaryOp::Add, _, _))), "sub_rv combines with Add");
    // The RV buffer is loaded (broadcast) inside the map body.
    assert!(topo.iter().any(|u| matches!(u.op(), Op::Load(..))), "sub_rv loads the broadcast vector element");
}

/// `maximum` on same-shape tiles emits a `Max`.
#[test]
fn test_maximum_emits_max() {
    let ker = probe();
    let warp = ker.warp();
    let a = warp.zero(ker.rt((16, 16), DType::Float32, TileLayout::Row, RT_16X16));
    let b = warp.zero(ker.rt((16, 16), DType::Float32, TileLayout::Row, RT_16X16));
    let out = warp.maximum(a, &b);
    assert!(
        out.uop().toposort().iter().any(|u| matches!(u.op(), Op::Binary(BinaryOp::Max, _, _))),
        "maximum emits a Max"
    );
}
