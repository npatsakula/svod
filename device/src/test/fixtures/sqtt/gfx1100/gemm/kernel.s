
kernel.co:	file format elf64-amdgpu
	.amdgcn_target "amdgpu-amd-amdhsa-unknown-gfx1100"

Disassembly of section .text:

0000000000001700 <r_32_32_16_2>:
	s_load_b128 s[4:7], s[0:1], null                           // 000000001700: F4080100 F8000000
	s_lshl_b32 s2, s15, 5                                      // 000000001708: 8402850F
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	v_lshl_or_b32 v1, v0, 1, s2                                // 000000001714: D6560001 00090300
	v_lshl_add_u32 v3, v0, 6, s14                              // 00000000171C: D6460003 00390D00
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000001724: BF870112
	v_ashrrev_i32_e32 v2, 31, v1                               // 000000001728: 3404029F
	v_ashrrev_i32_e32 v4, 31, v3                               // 00000000172C: 3408069F
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000001730: BF870112
	v_lshlrev_b64 v[1:2], 2, v[1:2]                            // 000000001734: D73C0001 00020282
	v_lshlrev_b64 v[3:4], 2, v[3:4]                            // 00000000173C: D73C0003 00020682
	s_waitcnt lgkmcnt(0)                                       // 000000001744: BF89FC07
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000001748: BF870192
	v_add_co_u32 v1, vcc_lo, s6, v1                            // 00000000174C: D7006A01 00020206
	v_add_co_ci_u32_e32 v2, vcc_lo, s7, v2, vcc_lo             // 000000001754: 40040407
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000001758: BF870213
	v_add_co_u32 v3, vcc_lo, s0, v3                            // 00000000175C: D7006A03 00020600
	v_add_co_ci_u32_e32 v4, vcc_lo, s1, v4, vcc_lo             // 000000001764: 40080801
	s_mov_b32 s0, exec_lo                                      // 000000001768: BE80007E
	global_load_b64 v[1:2], v[1:2], off                        // 00000000176C: DC560000 017C0001
	s_clause 0x1                                               // 000000001774: BF850001
	global_load_b32 v5, v[3:4], off                            // 000000001778: DC520000 057C0003
	global_load_b32 v3, v[3:4], off offset:128                 // 000000001780: DC520080 037C0003
	s_waitcnt vmcnt(1)                                         // 000000001788: BF8907F7
	v_fma_f32 v4, v1, v5, 0                                    // 00000000178C: D6130004 02020B01
	v_lshlrev_b32_e32 v5, 2, v0                                // 000000001794: 300A0082
	s_waitcnt vmcnt(0)                                         // 000000001798: BF8903F7
	s_delay_alu instid0(VALU_DEP_2)                            // 00000000179C: BF870002
	v_dual_mov_b32 v1, 0 :: v_dual_fmac_f32 v4, v2, v3         // 0000000017A0: CA000080 01040702
	ds_store_b32 v5, v4                                        // 0000000017A8: D8340000 00000405
	s_waitcnt lgkmcnt(0)                                       // 0000000017B0: BF89FC07
	v_cmpx_eq_u32_e32 0, v0                                    // 0000000017B4: 7D940080
	s_cbranch_execz 43                                         // 0000000017B8: BFA5002B <r_32_32_16_2+0x168>
	ds_load_b128 v[2:5], v1                                    // 0000000017BC: DBFC0000 02000001
	ds_load_b128 v[6:9], v1 offset:16                          // 0000000017C4: DBFC0010 06000001
	s_add_i32 s0, s2, s14                                      // 0000000017CC: 81000E02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000017D0: BF870499
	s_ashr_i32 s1, s0, 31                                      // 0000000017D4: 86019F00
	s_lshl_b64 s[0:1], s[0:1], 2                               // 0000000017D8: 84808200
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 0000000017DC: BF8700C9
	s_add_u32 s0, s4, s0                                       // 0000000017E0: 80000004
	s_addc_u32 s1, s5, s1                                      // 0000000017E4: 82010105
	s_waitcnt lgkmcnt(1)                                       // 0000000017E8: BF89FC17
	v_add_f32_e32 v0, 0, v2                                    // 0000000017EC: 06000480
	v_add_f32_e32 v0, v0, v3                                   // 0000000017F0: 06000700
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017F4: BF870091
	v_add_f32_e32 v0, v0, v4                                   // 0000000017F8: 06000900
	v_add_f32_e32 v0, v0, v5                                   // 0000000017FC: 06000B00
	ds_load_b128 v[2:5], v1 offset:32                          // 000000001800: DBFC0020 02000001
	s_waitcnt lgkmcnt(1)                                       // 000000001808: BF89FC17
	v_add_f32_e32 v0, v0, v6                                   // 00000000180C: 06000D00
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001810: BF870091
	v_add_f32_e32 v0, v0, v7                                   // 000000001814: 06000F00
	v_add_f32_e32 v0, v0, v8                                   // 000000001818: 06001100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 00000000181C: BF8700C1
	v_add_f32_e32 v0, v0, v9                                   // 000000001820: 06001300
	ds_load_b128 v[6:9], v1 offset:48                          // 000000001824: DBFC0030 06000001
	s_waitcnt lgkmcnt(1)                                       // 00000000182C: BF89FC17
	v_add_f32_e32 v0, v0, v2                                   // 000000001830: 06000500
	v_add_f32_e32 v0, v0, v3                                   // 000000001834: 06000700
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001838: BF870091
	v_add_f32_e32 v0, v0, v4                                   // 00000000183C: 06000900
	v_add_f32_e32 v0, v0, v5                                   // 000000001840: 06000B00
	s_waitcnt lgkmcnt(0)                                       // 000000001844: BF89FC07
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001848: BF870091
	v_add_f32_e32 v0, v0, v6                                   // 00000000184C: 06000D00
	v_add_f32_e32 v0, v0, v7                                   // 000000001850: 06000F00
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001854: BF870091
	v_add_f32_e32 v0, v0, v8                                   // 000000001858: 06001100
	v_add_f32_e32 v0, v0, v9                                   // 00000000185C: 06001300
	global_store_b32 v1, v0, s[0:1]                            // 000000001860: DC6A0000 00000001
	s_endpgm                                                   // 000000001868: BFB00000
	s_code_end                                                 // 00000000186C: BF9F0000
	s_code_end                                                 // 000000001870: BF9F0000
	s_code_end                                                 // 000000001874: BF9F0000
	s_code_end                                                 // 000000001878: BF9F0000
	s_code_end                                                 // 00000000187C: BF9F0000
	s_code_end                                                 // 000000001880: BF9F0000
	s_code_end                                                 // 000000001884: BF9F0000
	s_code_end                                                 // 000000001888: BF9F0000
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
