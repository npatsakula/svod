use svod_dtype::{AmdArch, CudaArch, GpuArch, ScalarDType};
use test_case::test_case;

use crate::atoms::Target;
use crate::layout::Dim::{Col, Lane, Reg, Row};

/// Where the instruction reads operand element `(lane, reg)` from, per the
/// vendor ISA documents, written independently of the layout algebra:
/// `(a: (m, k), b: (k, n), c: (m, n))` for register `j` of lane `l`.
type Isa = fn(u32, u32) -> [(u32, u32); 3];

/// PTX ISA `mma.m16n8k16` (`g = l/4`, `t = l%4`): A `a_j` at
/// `(g + 8·((j/2)%2), 2t + j%2 + 8·(j/4))`, B `b_j` at `(2t + j%2 + 8·(j/2), g)`,
/// C `c_j` at `(g + 8·(j/2), 2t + j%2)`.
fn ptx_m16n8k16(l: u32, j: u32) -> [(u32, u32); 3] {
    let (g, t) = (l / 4, l % 4);
    [
        (g + 8 * ((j / 2) % 2), 2 * t + j % 2 + 8 * (j / 4)),
        (2 * t + j % 2 + 8 * (j / 2), g),
        (g + 8 * (j / 2), 2 * t + j % 2),
    ]
}

/// RDNA3 ISA §7.9 `V_WMMA_F32_16X16X16_{F16,BF16}`, wave32: lane `l` holds
/// row `l%16` of A and column `l%16` of B over all sixteen K (the two half
/// waves replicate), and the f32 D in VGPR `j` at row `2j + l/16`.
fn rdna3_wmma(l: u32, j: u32) -> [(u32, u32); 3] {
    [(l % 16, j), (j, l % 16), (2 * j + l / 16, l % 16)]
}

/// RDNA4 ISA `V_WMMA_F32_16X16X16_{F16,BF16}`, wave32: half wave `h = l/16`
/// holds K `8h..8h+8` of row (A) or column (B) `l%16`, and D's VGPR `j` is
/// row `8h + j` of column `l%16`.
fn rdna4_wmma(l: u32, j: u32) -> [(u32, u32); 3] {
    let h = l / 16;
    [(l % 16, 8 * h + j), (8 * h + j, l % 16), (8 * h + j, l % 16)]
}

/// CDNA3 ISA `V_MFMA_F32_16X16X16_{F16,BF16}`, wave64: quarter `q = l/16`
/// holds K `4q..4q+4` of row (A) or column (B) `l%16`; D's VGPR `j` is row
/// `4q + j` of column `l%16`.
fn cdna3_mfma(l: u32, j: u32) -> [(u32, u32); 3] {
    let q = l / 16;
    [(l % 16, 4 * q + j), (4 * q + j, l % 16), (4 * q + j, l % 16)]
}

/// Every target's half-precision matrix cores, bf16 and f16 alike, place every
/// operand element exactly where the instruction reads and writes it, tiling the
/// instruction shape over one wave in `regs` registers of A, B and C; there is no
/// f32 core.
#[test_case(GpuArch::Cuda(CudaArch { major: 8, minor: 6 }), ptx_m16n8k16, 32, (16, 8, 16), (8, 4, 4); "sm_86 mma.sync")]
#[test_case(GpuArch::Cuda(CudaArch { major: 9, minor: 0 }), ptx_m16n8k16, 32, (16, 8, 16), (8, 4, 4); "sm_90 mma.sync")]
#[test_case(GpuArch::Amd(AmdArch::Gfx1100), rdna3_wmma, 32, (16, 16, 16), (16, 16, 8); "gfx1100 wmma")]
#[test_case(GpuArch::Amd(AmdArch::Gfx1151), rdna3_wmma, 32, (16, 16, 16), (16, 16, 8); "gfx1151 wmma")]
#[test_case(GpuArch::Amd(AmdArch::Gfx1201), rdna4_wmma, 32, (16, 16, 16), (8, 8, 8); "gfx1201 wmma")]
#[test_case(GpuArch::Amd(AmdArch::Gfx942), cdna3_mfma, 64, (16, 16, 16), (4, 4, 4); "gfx942 mfma")]
fn atom_layouts_match_the_isa(arch: GpuArch, isa: Isa, wave: u32, mnk: (u32, u32, u32), regs: (u32, u32, u32)) {
    let target = Target::for_arch(arch);
    assert_eq!(target.wave, wave);
    for dtype in [ScalarDType::BFloat16, ScalarDType::Float16] {
        let atom = target.mma(dtype, ScalarDType::Float32).unwrap_or_else(|| panic!("{arch:?} {dtype:?} core"));
        assert_eq!((atom.m, atom.n, atom.k), mnk);
        assert_eq!(atom.meta.dims, (atom.n as usize, atom.m as usize, atom.k as usize));
        for (i, (layout, (rows, cols), want)) in [
            (&atom.a, (atom.m, atom.k), regs.0),
            (&atom.b, (atom.k, atom.n), regs.1),
            (&atom.c, (atom.m, atom.n), regs.2),
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!((layout.out_size(Row), layout.out_size(Col)), (rows, cols), "{dtype:?} operand {i}");
            assert_eq!((layout.in_size(Reg), layout.in_size(Lane)), (want, wave), "{dtype:?} operand {i}");
            for l in 0..wave {
                for j in 0..want {
                    let at = layout.apply(&[(Lane, l), (Reg, j)]);
                    assert_eq!(at, isa(l, j)[i], "{dtype:?} operand {i}, lane {l}, register {j}");
                }
            }
            assert!(layout.is_surjective(), "{dtype:?} operand {i}: every element is held");
        }
    }
    assert!(target.mma(ScalarDType::Float32, ScalarDType::Float32).is_none(), "no f32 core");
}
