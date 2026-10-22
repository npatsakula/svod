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

/// PTX ISA `mma.m16n8k32` s8 (`g = l/4`, `t = l%4`): A `a_j` at
/// `(g + 8·((j/4)%2), 4t + j%4 + 16·(j/8))`, B `b_j` at `(4t + j%4 + 16·(j/4), g)`,
/// C as `m16n8k16`'s.
fn ptx_m16n8k32_s8(l: u32, j: u32) -> [(u32, u32); 3] {
    let (g, t) = (l / 4, l % 4);
    [(g + 8 * ((j / 4) % 2), 4 * t + j % 4 + 16 * (j / 8)), (4 * t + j % 4 + 16 * (j / 4), g), ptx_m16n8k16(l, j)[2]]
}

/// CDNA3 ISA `V_MFMA_I32_16X16X32_I8`, wave64: quarter `q = l/16` holds K
/// `8q..8q+8` of row (A) or column (B) `l%16`; D as the f32 core's.
fn cdna3_mfma_i8(l: u32, j: u32) -> [(u32, u32); 3] {
    let q = l / 16;
    [(l % 16, 8 * q + j), (8 * q + j, l % 16), cdna3_mfma(l, j)[2]]
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
    for dtype in [ScalarDType::BFloat16, ScalarDType::Float16] {
        matches_the_isa(&target, dtype, ScalarDType::Float32, isa, wave, mnk, regs);
    }
    assert!(target.mma(ScalarDType::Float32, ScalarDType::Float32).is_none(), "no f32 core");
}

/// The 8-bit integer cores into i32 likewise: twice the 16-bit depth in the
/// same registers on NVIDIA (`m16n8k32`) and CDNA (`16x16x32`), RDNA's `iu8`
/// at the 16-bit depth and layouts.
#[test_case(GpuArch::Cuda(CudaArch { major: 8, minor: 6 }), ptx_m16n8k32_s8, 32, (16, 8, 32), (16, 8, 4); "sm_86 mma.sync s8")]
#[test_case(GpuArch::Cuda(CudaArch { major: 12, minor: 0 }), ptx_m16n8k32_s8, 32, (16, 8, 32), (16, 8, 4); "sm_120 mma.sync s8")]
#[test_case(GpuArch::Amd(AmdArch::Gfx1100), rdna3_wmma, 32, (16, 16, 16), (16, 16, 8); "gfx1100 wmma iu8")]
#[test_case(GpuArch::Amd(AmdArch::Gfx1151), rdna3_wmma, 32, (16, 16, 16), (16, 16, 8); "gfx1151 wmma iu8")]
#[test_case(GpuArch::Amd(AmdArch::Gfx1201), rdna4_wmma, 32, (16, 16, 16), (8, 8, 8); "gfx1201 wmma iu8")]
#[test_case(GpuArch::Amd(AmdArch::Gfx942), cdna3_mfma_i8, 64, (16, 16, 32), (8, 8, 4); "gfx942 mfma i8")]
#[test_case(GpuArch::Amd(AmdArch::Gfx950), cdna3_mfma_i8, 64, (16, 16, 32), (8, 8, 4); "gfx950 mfma i8")]
fn int8_atom_layouts_match_the_isa(arch: GpuArch, isa: Isa, wave: u32, mnk: (u32, u32, u32), regs: (u32, u32, u32)) {
    let target = Target::for_arch(arch);
    matches_the_isa(&target, ScalarDType::Int8, ScalarDType::Int32, isa, wave, mnk, regs);
    assert!(target.mma(ScalarDType::Int8, ScalarDType::Float32).is_none(), "an 8-bit core accumulates in i32");
}

/// `target`'s `dtype_in → dtype_out` atom places every element as `isa`
/// says, in `regs` registers of A, B and C over `wave` lanes.
fn matches_the_isa(
    target: &Target,
    dtype_in: ScalarDType,
    dtype_out: ScalarDType,
    isa: Isa,
    wave: u32,
    mnk: (u32, u32, u32),
    regs: (u32, u32, u32),
) {
    assert_eq!(target.wave, wave);
    let arch = target.arch;
    let atom = target.mma(dtype_in, dtype_out).unwrap_or_else(|| panic!("{arch:?} {dtype_in:?} core"));
    assert_eq!(target.geometry_of(dtype_in), Some(atom));
    assert_eq!((atom.m, atom.n, atom.k), mnk);
    assert_eq!(atom.meta.dims, (atom.n as usize, atom.m as usize, atom.k as usize));
    for (i, (layout, (rows, cols), want)) in
        [(&atom.a, (atom.m, atom.k), regs.0), (&atom.b, (atom.k, atom.n), regs.1), (&atom.c, (atom.m, atom.n), regs.2)]
            .into_iter()
            .enumerate()
    {
        assert_eq!((layout.out_size(Row), layout.out_size(Col)), (rows, cols), "{dtype_in:?} operand {i}");
        assert_eq!((layout.in_size(Reg), layout.in_size(Lane)), (want, wave), "{dtype_in:?} operand {i}");
        for l in 0..wave {
            for j in 0..want {
                let at = layout.apply(&[(Lane, l), (Reg, j)]);
                assert_eq!(at, isa(l, j)[i], "{dtype_in:?} operand {i}, lane {l}, register {j}");
            }
        }
        assert!(super::layout::covers(layout), "{dtype_in:?} operand {i}: every element is held");
    }
}

/// The planner prices one 16-bit atom a target ([`Target::geometry`]) and
/// the lowering issues the dtype's own: every 16-bit atom of every target,
/// bf16 and f16 alike, has the geometry's shape and operand layouts, so what
/// is priced is what runs. An 8-bit atom ([`Target::geometry_of`]), on every
/// target with kernels, has the geometry's `m` and `n` (an 8-bit tile is laid
/// over the warps as a 16-bit one) and its `k` or twice it.
#[test]
fn every_atom_of_a_target_shares_its_geometry() {
    let metal = Target::for_arch(GpuArch::Metal(svod_dtype::MetalFamily::Apple(9)));
    let cards = crate::atoms::CARDS.iter().map(|c| Target::card(c.0).expect("a listed card"));
    let targets = crate::sweep::targets().into_iter().map(|(_, t)| t).chain(cards).chain([metal]);
    for target in targets {
        let geometry = target.geometry().unwrap_or_else(|| panic!("{:?}: a matrix core", target.arch));
        let shape = |a: &crate::atoms::MmaAtom| (a.m, a.n, a.k, a.a.clone(), a.b.clone(), a.c.clone());
        for atom in target.mma.iter().filter(|a| a.dtype_in.bytes() == 2) {
            assert_eq!(shape(atom), shape(geometry), "{:?} {:?}", target.arch, atom.dtype_in);
        }
        for dtype in [ScalarDType::BFloat16, ScalarDType::Float16] {
            assert!(target.mma(dtype, ScalarDType::Float32).is_some(), "{:?} {dtype:?}", target.arch);
        }
        let int8 = target.geometry_of(ScalarDType::Int8);
        assert_eq!(int8.is_some(), crate::ops::config::has_kernels(&target), "{:?}: an 8-bit core", target.arch);
        if let Some(atom) = int8 {
            assert_eq!((atom.m, atom.n, atom.dtype_out), (geometry.m, geometry.n, ScalarDType::Int32));
            assert!([geometry.k, 2 * geometry.k].contains(&atom.k), "{:?}: an 8-bit depth of {}", target.arch, atom.k);
        }
    }
}
