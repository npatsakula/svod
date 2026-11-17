
kernel.co:	file format elf64-amdgpu
	.amdgcn_target "amdgpu-amd-amdhsa-unknown-gfx1201"

Disassembly of section .text:

0000000000001700 <r_32_32_16_2>:
	s_load_b128 s[4:7], s[0:1], 0x0                            // 000000001700: F4004100 F8000000
	s_lshl_b32 s2, ttmp7, 5                                    // 000000001708: 84028573
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4002000 F8000010
	v_lshl_or_b32 v1, v0, 1, s2                                // 000000001714: D6560001 00090300
	v_lshl_add_u32 v3, v0, 6, ttmp9                            // 00000000171C: D6460003 01D50D00
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000001724: BF870112
	v_ashrrev_i32_e32 v2, 31, v1                               // 000000001728: 3404029F
	v_ashrrev_i32_e32 v4, 31, v3                               // 00000000172C: 3408069F
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000001730: BF870112
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]                        // 000000001734: 3E020282
	v_lshlrev_b64_e32 v[3:4], 2, v[3:4]                        // 000000001738: 3E060682
	s_wait_kmcnt 0x0                                           // 00000000173C: BFC70000
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000001740: BF870192
	v_add_co_u32 v1, vcc_lo, s6, v1                            // 000000001744: D7006A01 00020206
	v_add_co_ci_u32_e32 v2, vcc_lo, s7, v2, vcc_lo             // 00000000174C: 40040407
	s_delay_alu instid0(VALU_DEP_3)                            // 000000001750: BF870003
	v_add_co_u32 v3, vcc_lo, s0, v3                            // 000000001754: D7006A03 00020600
	s_wait_alu 0xfffd                                          // 00000000175C: BF88FFFD
	v_add_co_ci_u32_e32 v4, vcc_lo, s1, v4, vcc_lo             // 000000001760: 40080801
	s_mov_b32 s0, exec_lo                                      // 000000001764: BE80007E
	global_load_b64 v[1:2], v[1:2], off                        // 000000001768: EE05407C 00000001 00000001
	s_clause 0x1                                               // 000000001774: BF850001
	global_load_b32 v5, v[3:4], off                            // 000000001778: EE05007C 00000005 00000003
	global_load_b32 v3, v[3:4], off offset:128                 // 000000001784: EE05007C 00000003 00008003
	s_wait_loadcnt 0x1                                         // 000000001790: BFC00001
	v_fma_f32 v4, v1, v5, 0                                    // 000000001794: D6130004 02020B01
	v_lshlrev_b32_e32 v5, 2, v0                                // 00000000179C: 300A0082
	s_wait_loadcnt 0x0                                         // 0000000017A0: BFC00000
	s_delay_alu instid0(VALU_DEP_2)                            // 0000000017A4: BF870002
	v_dual_mov_b32 v1, 0 :: v_dual_fmac_f32 v4, v2, v3         // 0000000017A8: CA000080 01040702
	ds_store_b32 v5, v4                                        // 0000000017B0: D8340000 00000405
	s_wait_dscnt 0x0                                           // 0000000017B8: BFC60000
	s_wait_dscnt 0x0                                           // 0000000017BC: BFC60000
	v_cmpx_eq_u32_e32 0, v0                                    // 0000000017C0: 7D940080
	s_cbranch_execz 46                                         // 0000000017C4: BFA5002E <r_32_32_16_2+0x180>
	ds_load_b128 v[2:5], v1                                    // 0000000017C8: DBFC0000 02000001
	ds_load_b128 v[6:9], v1 offset:16                          // 0000000017D0: DBFC0010 06000001
	s_wait_alu 0xfffe                                          // 0000000017D8: BF88FFFE
	s_add_co_i32 s0, s2, ttmp9                                 // 0000000017DC: 81007502
	s_wait_alu 0xfffe                                          // 0000000017E0: BF88FFFE
	s_ashr_i32 s1, s0, 31                                      // 0000000017E4: 86019F00
	s_wait_alu 0xfffe                                          // 0000000017E8: BF88FFFE
	s_lshl_b64 s[0:1], s[0:1], 2                               // 0000000017EC: 84808200
	s_wait_alu 0xfffe                                          // 0000000017F0: BF88FFFE
	s_add_nc_u64 s[0:1], s[4:5], s[0:1]                        // 0000000017F4: A9800004
	s_wait_dscnt 0x1                                           // 0000000017F8: BFC60001
	v_add_f32_e32 v0, 0, v2                                    // 0000000017FC: 06000480
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001800: BF870091
	v_add_f32_e32 v0, v0, v3                                   // 000000001804: 06000700
	v_add_f32_e32 v0, v0, v4                                   // 000000001808: 06000900
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 00000000180C: BF8700C1
	v_add_f32_e32 v0, v0, v5                                   // 000000001810: 06000B00
	ds_load_b128 v[2:5], v1 offset:32                          // 000000001814: DBFC0020 02000001
	s_wait_dscnt 0x1                                           // 00000000181C: BFC60001
	v_add_f32_e32 v0, v0, v6                                   // 000000001820: 06000D00
	v_add_f32_e32 v0, v0, v7                                   // 000000001824: 06000F00
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001828: BF870091
	v_add_f32_e32 v0, v0, v8                                   // 00000000182C: 06001100
	v_add_f32_e32 v0, v0, v9                                   // 000000001830: 06001300
	ds_load_b128 v[6:9], v1 offset:48                          // 000000001834: DBFC0030 06000001
	s_wait_dscnt 0x1                                           // 00000000183C: BFC60001
	v_add_f32_e32 v0, v0, v2                                   // 000000001840: 06000500
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001844: BF870091
	v_add_f32_e32 v0, v0, v3                                   // 000000001848: 06000700
	v_add_f32_e32 v0, v0, v4                                   // 00000000184C: 06000900
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001850: BF8700A1
	v_add_f32_e32 v0, v0, v5                                   // 000000001854: 06000B00
	s_wait_dscnt 0x0                                           // 000000001858: BFC60000
	v_add_f32_e32 v0, v0, v6                                   // 00000000185C: 06000D00
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001860: BF870091
	v_add_f32_e32 v0, v0, v7                                   // 000000001864: 06000F00
	v_add_f32_e32 v0, v0, v8                                   // 000000001868: 06001100
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000186C: BF870001
	v_add_f32_e32 v0, v0, v9                                   // 000000001870: 06001300
	global_store_b32 v1, v0, s[0:1]                            // 000000001874: EE068000 00000000 00000001
	s_nop 0                                                    // 000000001880: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001884: BFB60003
	s_endpgm                                                   // 000000001888: BFB00000
	s_code_end                                                 // 00000000188C: BF9F0000
	s_code_end                                                 // 000000001890: BF9F0000
	s_code_end                                                 // 000000001894: BF9F0000
	s_code_end                                                 // 000000001898: BF9F0000
	s_code_end                                                 // 00000000189C: BF9F0000
	s_code_end                                                 // 0000000018A0: BF9F0000
	s_code_end                                                 // 0000000018A4: BF9F0000
	s_code_end                                                 // 0000000018A8: BF9F0000
	s_code_end                                                 // 0000000018AC: BF9F0000
	s_code_end                                                 // 0000000018B0: BF9F0000
	s_code_end                                                 // 0000000018B4: BF9F0000
	s_code_end                                                 // 0000000018B8: BF9F0000
	s_code_end                                                 // 0000000018BC: BF9F0000
	s_code_end                                                 // 0000000018C0: BF9F0000
	s_code_end                                                 // 0000000018C4: BF9F0000
	s_code_end                                                 // 0000000018C8: BF9F0000
	s_code_end                                                 // 0000000018CC: BF9F0000
	s_code_end                                                 // 0000000018D0: BF9F0000
	s_code_end                                                 // 0000000018D4: BF9F0000
	s_code_end                                                 // 0000000018D8: BF9F0000
	s_code_end                                                 // 0000000018DC: BF9F0000
	s_code_end                                                 // 0000000018E0: BF9F0000
	s_code_end                                                 // 0000000018E4: BF9F0000
	s_code_end                                                 // 0000000018E8: BF9F0000
	s_code_end                                                 // 0000000018EC: BF9F0000
	s_code_end                                                 // 0000000018F0: BF9F0000
	s_code_end                                                 // 0000000018F4: BF9F0000
	s_code_end                                                 // 0000000018F8: BF9F0000
	s_code_end                                                 // 0000000018FC: BF9F0000
	s_code_end                                                 // 000000001900: BF9F0000
	s_code_end                                                 // 000000001904: BF9F0000
	s_code_end                                                 // 000000001908: BF9F0000
	s_code_end                                                 // 00000000190C: BF9F0000
	s_code_end                                                 // 000000001910: BF9F0000
	s_code_end                                                 // 000000001914: BF9F0000
	s_code_end                                                 // 000000001918: BF9F0000
	s_code_end                                                 // 00000000191C: BF9F0000
	s_code_end                                                 // 000000001920: BF9F0000
	s_code_end                                                 // 000000001924: BF9F0000
	s_code_end                                                 // 000000001928: BF9F0000
	s_code_end                                                 // 00000000192C: BF9F0000
	s_code_end                                                 // 000000001930: BF9F0000
	s_code_end                                                 // 000000001934: BF9F0000
	s_code_end                                                 // 000000001938: BF9F0000
	s_code_end                                                 // 00000000193C: BF9F0000
	s_code_end                                                 // 000000001940: BF9F0000
	s_code_end                                                 // 000000001944: BF9F0000
	s_code_end                                                 // 000000001948: BF9F0000
	s_code_end                                                 // 00000000194C: BF9F0000
	s_code_end                                                 // 000000001950: BF9F0000
	s_code_end                                                 // 000000001954: BF9F0000
	s_code_end                                                 // 000000001958: BF9F0000
	s_code_end                                                 // 00000000195C: BF9F0000
	s_code_end                                                 // 000000001960: BF9F0000
	s_code_end                                                 // 000000001964: BF9F0000
	s_code_end                                                 // 000000001968: BF9F0000
	s_code_end                                                 // 00000000196C: BF9F0000
	s_code_end                                                 // 000000001970: BF9F0000
	s_code_end                                                 // 000000001974: BF9F0000
	s_code_end                                                 // 000000001978: BF9F0000
	s_code_end                                                 // 00000000197C: BF9F0000
	s_code_end                                                 // 000000001980: BF9F0000
	s_code_end                                                 // 000000001984: BF9F0000
	s_code_end                                                 // 000000001988: BF9F0000
	s_code_end                                                 // 00000000198C: BF9F0000
	s_code_end                                                 // 000000001990: BF9F0000
	s_code_end                                                 // 000000001994: BF9F0000
	s_code_end                                                 // 000000001998: BF9F0000
	s_code_end                                                 // 00000000199C: BF9F0000
	s_code_end                                                 // 0000000019A0: BF9F0000
	s_code_end                                                 // 0000000019A4: BF9F0000
	s_code_end                                                 // 0000000019A8: BF9F0000
	s_code_end                                                 // 0000000019AC: BF9F0000
	s_code_end                                                 // 0000000019B0: BF9F0000
	s_code_end                                                 // 0000000019B4: BF9F0000
	s_code_end                                                 // 0000000019B8: BF9F0000
	s_code_end                                                 // 0000000019BC: BF9F0000
	s_code_end                                                 // 0000000019C0: BF9F0000
	s_code_end                                                 // 0000000019C4: BF9F0000
	s_code_end                                                 // 0000000019C8: BF9F0000
	s_code_end                                                 // 0000000019CC: BF9F0000
	s_code_end                                                 // 0000000019D0: BF9F0000
	s_code_end                                                 // 0000000019D4: BF9F0000
	s_code_end                                                 // 0000000019D8: BF9F0000
	s_code_end                                                 // 0000000019DC: BF9F0000
	s_code_end                                                 // 0000000019E0: BF9F0000
	s_code_end                                                 // 0000000019E4: BF9F0000
	s_code_end                                                 // 0000000019E8: BF9F0000
	s_code_end                                                 // 0000000019EC: BF9F0000
	s_code_end                                                 // 0000000019F0: BF9F0000
	s_code_end                                                 // 0000000019F4: BF9F0000
	s_code_end                                                 // 0000000019F8: BF9F0000
	s_code_end                                                 // 0000000019FC: BF9F0000
	s_code_end                                                 // 000000001A00: BF9F0000
	s_code_end                                                 // 000000001A04: BF9F0000
	s_code_end                                                 // 000000001A08: BF9F0000
	s_code_end                                                 // 000000001A0C: BF9F0000
	s_code_end                                                 // 000000001A10: BF9F0000
	s_code_end                                                 // 000000001A14: BF9F0000
	s_code_end                                                 // 000000001A18: BF9F0000
	s_code_end                                                 // 000000001A1C: BF9F0000
	s_code_end                                                 // 000000001A20: BF9F0000
	s_code_end                                                 // 000000001A24: BF9F0000
	s_code_end                                                 // 000000001A28: BF9F0000
	s_code_end                                                 // 000000001A2C: BF9F0000
	s_code_end                                                 // 000000001A30: BF9F0000
	s_code_end                                                 // 000000001A34: BF9F0000
	s_code_end                                                 // 000000001A38: BF9F0000
	s_code_end                                                 // 000000001A3C: BF9F0000
	s_code_end                                                 // 000000001A40: BF9F0000
	s_code_end                                                 // 000000001A44: BF9F0000
	s_code_end                                                 // 000000001A48: BF9F0000
	s_code_end                                                 // 000000001A4C: BF9F0000
	s_code_end                                                 // 000000001A50: BF9F0000
	s_code_end                                                 // 000000001A54: BF9F0000
	s_code_end                                                 // 000000001A58: BF9F0000
	s_code_end                                                 // 000000001A5C: BF9F0000
	s_code_end                                                 // 000000001A60: BF9F0000
	s_code_end                                                 // 000000001A64: BF9F0000
	s_code_end                                                 // 000000001A68: BF9F0000
	s_code_end                                                 // 000000001A6C: BF9F0000
	s_code_end                                                 // 000000001A70: BF9F0000
	s_code_end                                                 // 000000001A74: BF9F0000
	s_code_end                                                 // 000000001A78: BF9F0000
	s_code_end                                                 // 000000001A7C: BF9F0000
