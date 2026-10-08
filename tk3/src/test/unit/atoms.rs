use svod_dtype::{AmdArch, CudaArch, GpuArch, ScalarDType};
use test_case::test_case;

use crate::atoms::Target;
use crate::layout::Dim::{Col, Lane, Reg, Row};

/// Every target's half-precision matrix core has operand layouts that tile its
/// instruction shape bijectively over one wave.
#[test_case(GpuArch::Cuda(CudaArch { major: 8, minor: 6 }), 32, (16, 8, 16), (8, 4, 4); "sm_86 mma.sync")]
#[test_case(GpuArch::Amd(AmdArch::Gfx942), 64, (16, 16, 16), (4, 4, 4); "gfx942 mfma")]
#[test_case(GpuArch::Amd(AmdArch::Gfx1201), 32, (16, 16, 16), (8, 8, 8); "gfx1201 wmma")]
fn matrix_cores_carry_bijective_layouts(arch: GpuArch, wave: u32, mnk: (u32, u32, u32), regs: (u32, u32, u32)) {
    let target = Target::for_arch(arch);
    assert_eq!(target.wave, wave);
    for dtype in [ScalarDType::BFloat16, ScalarDType::Float16] {
        let atom = target.mma(dtype, ScalarDType::Float32).unwrap_or_else(|| panic!("{arch:?} {dtype:?} core"));
        assert_eq!((atom.m, atom.n, atom.k), mnk);
        for (operand, (rows, cols), want) in [
            (&atom.a, (atom.m, atom.k), regs.0),
            (&atom.b, (atom.k, atom.n), regs.1),
            (&atom.c, (atom.m, atom.n), regs.2),
        ] {
            assert_eq!((operand.out_size(Row), operand.out_size(Col)), (rows, cols));
            assert_eq!(operand.in_size(Reg), want);
            assert_eq!(operand.in_size(Lane), wave);
            assert!(operand.is_surjective(), "every element of the operand is held");
        }
        assert_eq!(atom.meta.dims, (atom.n as usize, atom.m as usize, atom.k as usize));
    }
    assert!(target.mma(ScalarDType::Float32, ScalarDType::Float32).is_none(), "no f32 core");
}
