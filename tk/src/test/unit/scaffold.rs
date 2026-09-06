//! Unit tests for the kernel-scaffold shortcuts ([`crate::scaffold`]): the role
//! tile builders resolve the arch fragment via `caps` (on BOTH arches — the
//! gfx942-only golden tests don't cover gfx1151), and `bind_abi` binds outputs
//! before inputs. The end-to-end graph-identity is covered by the golden digests.

use svod_dtype::{AmdArch, CudaArch, DType, DeviceSpec, GpuArch};
use svod_ir::UOp;

use crate::arch::FragRole;
use crate::tiles::TileLayout;
use crate::{ArchCaps, GlSpec, Kernel};

/// `acc`/`operand`/`acc_t`/`shared`/`shared_sw` resolve to the same fragment the
/// `caps` resolver would — on both gfx942 (CDNA) and gfx1151 (RDNA), proving the
/// shortcuts are arch-blind (they never name a physical fragment constant).
#[test]
fn role_tiles_resolve_arch_fragment() {
    for arch in [AmdArch::Gfx942, AmdArch::Gfx1151] {
        let caps = ArchCaps::for_amd(arch);
        let ker = Kernel::new("scaf", [1, 1, 1], 64, vec![], caps);
        let (row, col) = (TileLayout::Row, TileLayout::Col);
        assert_eq!(Some(ker.acc((16, 16), col).base), caps.frag(FragRole::Accumulator), "{arch:?} acc");
        assert_eq!(
            Some(ker.operand((16, 16), DType::BFloat16, row).base),
            caps.frag(FragRole::Operand),
            "{arch:?} operand"
        );
        assert_eq!(Some(ker.acc_t((16, 16), row).base), caps.frag(FragRole::AccumulatorT), "{arch:?} acc_t");
        assert_eq!(Some(ker.shared((16, 16), DType::BFloat16, row).base), caps.shared_default(), "{arch:?} shared");
        assert_eq!(
            Some(ker.shared_sw((16, 16), DType::BFloat16, row).base),
            caps.shared_swizzled(),
            "{arch:?} shared_sw"
        );
    }
}

/// On an arch without fragment layouts (pre-Ampere CUDA: no f16/bf16 `m16n8k16`)
/// the role tiles refuse loudly instead of silently building a layout the hardware
/// does not have.
#[test]
#[should_panic(expected = "sm_75: tk defines no Accumulator fragment layout")]
fn role_tiles_panic_without_fragment_layouts() {
    let caps = ArchCaps::for_arch(GpuArch::Cuda(CudaArch::from_compute_capability(7, 5)));
    let ker = Kernel::new("scaf", [1, 1, 1], 32, vec![], caps);
    let _ = ker.acc((16, 16), TileLayout::Col);
}

/// `bind_abi` binds outputs first, then inputs, preserving order and shapes — so the
/// ABI slot order is fixed by the call structure, not by statement order.
#[test]
fn bind_abi_orders_outputs_before_inputs() {
    // 3 dummy buffers (1 output + 2 inputs) for the GL bindings to claim.
    let bufs = vec![
        UOp::new_buffer(DeviceSpec::Cpu, 6, DType::Float32),
        UOp::new_buffer(DeviceSpec::Cpu, 20, DType::BFloat16),
        UOp::new_buffer(DeviceSpec::Cpu, 42, DType::BFloat16),
    ];
    let ker = Kernel::new("scaf", [1, 1, 1], 64, bufs, ArchCaps::GFX942);
    let (outs, ins) = ker.bind_abi(
        &[GlSpec::new(&[2, 3], DType::Float32)],
        &[GlSpec::new(&[4, 5], DType::BFloat16), GlSpec::new(&[6, 7], DType::BFloat16)],
    );
    assert_eq!(outs.len(), 1);
    assert_eq!(ins.len(), 2);
    assert_eq!(outs[0].shape(), &[2, 3], "output shape preserved");
    assert_eq!(ins[0].shape(), &[4, 5], "first input shape/order preserved");
    assert_eq!(ins[1].shape(), &[6, 7], "second input shape/order preserved");
}

/// `assert_divisible` accepts a multiple and panics otherwise.
#[test]
#[should_panic(expected = "Q_BLK")]
fn assert_divisible_rejects_non_multiple() {
    Kernel::assert_divisible(48, 16, "test"); // ok
    Kernel::assert_divisible(17, 16, "Q_BLK"); // panics
}
