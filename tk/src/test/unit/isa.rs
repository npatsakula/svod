//! The main loop found in every assembly dialect the comparisons read, a
//! kernel cut out of a text of several, and main loops set side by side.

use test_case::test_case;

use crate::isa::{MainLoop, function, functions, main_loop, table};

/// `nvdisasm -c`: an outer loop around a matrix loop, and an inner copy
/// loop without one; the innermost loop issuing `HMMA` is the main loop.
const NVDISASM: &str = "\t.text.gemm:
        /*0000*/                   MOV R1, c[0x0][0x28] ;                 /* 0x00000a0000017a02 */
.L_x_0:
        /*0010*/                   LDGSTS.E.BYPASS.128 [R2], [R4.64] ;    /* 0x0000000004027fae */
.L_x_1:
        /*0020*/                   LDSM.16.M88.4 R8, [R2] ;
        /*0030*/                   HMMA.16816.F32.BF16 R12, R8, R10, R12 ;
        /*0040*/                   HMMA.16816.F32.BF16 R16, R8, R11, R16 ;
        /*0050*/               @P0 BRA `(.L_x_1) ;
.L_x_2:
        /*0060*/                   STS [R3], R5 ;
        /*0070*/               @P1 BRA `(.L_x_2) ;
        /*0080*/               @P2 BRA `(.L_x_0) ;
        /*0090*/                   EXIT ;
";

/// `cuobjdump -sass`: no labels, branches to addresses.
const CUOBJDUMP: &str = "\t\tFunction : gemm
        /*0000*/                   MOV R1, c[0x0][0x28] ;                 /* 0x00000a0000017a02 */
        /*0010*/                   LDSM.16.M88.4 R8, [R2] ;               /* 0x000000000208783b */
        /*0020*/                   HGMMA.64x128x16.F32.BF16 R24, gdesc[UR4], R24 ;
        /*0030*/                   BAR.SYNC.DEFER_BLOCKING 0x0 ;
        /*0040*/               @P0 BRA 0x10 ;                             /* 0xfffffff000000947 */
        /*0050*/                   EXIT ;
";

/// PTX: `$L__` labels and predicated `bra`.
const PTX: &str = ".visible .entry gemm(
\t.param .u64 gemm_param_0
)
{
\tmov.u32 \t%r1, 0;
$L__BB0_1:
\tldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r2, %r3, %r4, %r5}, [%r6];
\tmma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%f1, %f2, %f3, %f4}, {%r2, %r3, %r4, %r5}, {%r7, %r8}, {%f1, %f2, %f3, %f4};
\tsetp.lt.u32 \t%p1, %r1, 64;
\t@%p1 bra \t$L__BB0_1;
\tret;
}
";

/// `clang -S` AMDGCN: `.LBB` labels, `;` comments.
const AMDGCN: &str = "\t.type\tgemm,@function
gemm:                                   ; @gemm
; %bb.0:
\ts_load_dwordx4 s[8:11], s[0:1], 0x0
.LBB0_1:                                ; =>This Inner Loop Header: Depth=1
\tds_read_b128 v[0:3], v4
\ts_waitcnt lgkmcnt(0)
\tv_mfma_f32_32x32x8_bf16 a[0:15], v[0:1], v[2:3], a[0:15]
\tv_mfma_f32_32x32x8_bf16 a[16:31], v[0:1], v[2:3], a[16:31]
\ts_cbranch_scc1 .LBB0_1
\ts_endpgm
.Lfunc_end0:
\t.size\tgemm, .Lfunc_end0-gemm
";

/// `llvm-objdump -d --symbolize-operands`: `<address> <name>:` labels,
/// encodings after `//`.
const OBJDUMP: &str = "0000000000000000 <gemm>:
\ts_load_dwordx4 s[8:11], s[0:1], 0x0                        // 000000000000: C00A0200 00000000
00000000000003f0 <L0>:
\tglobal_load_dwordx4 v[0:3], v[4:5], off                    // 0000000003F0: DC5C0000 007F0004
\tv_wmma_f32_16x16x16_bf16 v[8:15], v[0:3], v[0:3], v[8:15]  // 0000000003F8: CC420008 1C220100
\ts_cbranch_scc1 L0                                          // 000000000404: BF85FFFA
\ts_endpgm                                                   // 000000000408: BF810000
";

#[test_case(NVDISASM, &[("HMMA.16816.F32.BF16", 2), ("BRA", 1), ("LDSM.16.M88.4", 1)]; "nvdisasm")]
#[test_case(CUOBJDUMP, &[("BAR.SYNC.DEFER_BLOCKING", 1), ("BRA", 1), ("HGMMA.64x128x16.F32.BF16", 1), ("LDSM.16.M88.4", 1)]; "cuobjdump")]
#[test_case(PTX, &[("bra", 1), ("ldmatrix.sync.aligned.m8n8.x4.shared.b16", 1), ("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32", 1), ("setp.lt.u32", 1)]; "ptx")]
#[test_case(AMDGCN, &[("v_mfma_f32_32x32x8_bf16", 2), ("ds_read_b128", 1), ("s_cbranch_scc1", 1), ("s_waitcnt", 1)]; "amdgcn")]
#[test_case(OBJDUMP, &[("global_load_dwordx4", 1), ("s_cbranch_scc1", 1), ("v_wmma_f32_16x16x16_bf16", 1)]; "llvm-objdump")]
fn the_main_loop_is_the_innermost_loop_issuing_a_matrix_instruction(asm: &str, want: &[(&str, usize)]) {
    let want = MainLoop { ops: want.iter().map(|&(op, n)| (op.to_string(), n)).collect() };
    assert_eq!(main_loop(asm), Some(want));
}

/// Without a loop, or with loops of no matrix instruction, there is no
/// main loop.
#[test_case("\tHMMA.16816.F32.BF16 R12, R8, R10, R12 ;\n\tEXIT ;\n"; "a matrix instruction outside any loop")]
#[test_case(".L_x_0:\n\tSTS [R3], R5 ;\n\t@P0 BRA `(.L_x_0) ;\n"; "a loop without one")]
#[test_case("\tHMMA.16816.F32.BF16 R12, R8, R10, R12 ;\n\tBRA `(.L_x_9) ;\n.L_x_9:\n"; "a forward branch")]
fn no_matrix_loop_is_no_main_loop(asm: &str) {
    assert_eq!(main_loop(asm), None);
}

/// One kernel of a text holding several, in each dialect: from its start to
/// the next kernel's, and the functions listed in order.
#[test_case(&format!("{NVDISASM}\t.text.other:\n\tEXIT ;\n"), "\t.text.other:\n"; "nvdisasm")]
#[test_case(&format!("{CUOBJDUMP}\t\tFunction : other\n\tEXIT ;\n"), "\t\tFunction : other\n"; "cuobjdump")]
#[test_case(&format!("gemm:\n{NVDISASM}other:\n\t.text.other:\n\tEXIT ;\n"), "other:\n"; "nvdisasm with symbols")]
#[test_case(&format!("{PTX}.visible .entry other(\n)\n"), ".visible .entry other(\n"; "ptx")]
#[test_case(&format!("{AMDGCN}other:                                   ; @other\n\ts_endpgm\n"), "other:"; "amdgcn")]
#[test_case(&format!("{OBJDUMP}0000000000000500 <other>:\n\ts_endpgm\n"), "0000000000000500 <other>:\n"; "llvm-objdump")]
fn a_function_is_cut_from_its_start_to_the_next(asm: &str, second: &str) {
    assert_eq!(functions(asm), ["gemm", "other"]);
    let gemm = function(asm, "gemm").expect("gemm");
    assert!(!gemm.contains("other") && main_loop(gemm).is_some(), "{gemm}");
    assert!(function(asm, "other").expect("other").starts_with(second));
    assert_eq!(function(asm, "missing"), None);
}

/// AMDGCN metadata keys and block labels start no function.
#[test]
fn metadata_and_blocks_are_not_functions() {
    let asm = format!("{AMDGCN}\t.amdgpu_metadata\n---\namdhsa.kernels:\n  - .name: gemm\n...\n");
    assert_eq!(functions(&asm), ["gemm"]);
    assert_eq!(functions(OBJDUMP), ["gemm"]);
}

/// Every opcode of either loop is a row, the most issued first, with its
/// count in each column and zero where a loop issues none.
#[test]
fn main_loops_sit_side_by_side() {
    let tk = MainLoop { ops: vec![("HMMA".into(), 32), ("LDSM".into(), 12), ("BRA".into(), 1)] };
    let reference = MainLoop { ops: vec![("HMMA".into(), 32), ("LDGSTS".into(), 16)] };
    let text = table(&[("tk", &tk), ("cublas", &reference)]);
    let rows: Vec<Vec<&str>> = text.lines().map(|l| l.split_whitespace().collect()).collect();
    assert_eq!(
        rows,
        [
            vec!["tk", "cublas"],
            vec!["instructions", "45", "48"],
            vec!["HMMA", "32", "32"],
            vec!["LDGSTS", "0", "16"],
            vec!["LDSM", "12", "0"],
            vec!["BRA", "1", "0"],
        ]
    );
}
