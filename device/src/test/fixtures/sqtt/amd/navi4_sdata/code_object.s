
code_object.out:	file format elf64-amdgpu
	.amdgcn_target "amdgpu-amd-amdhsa-unknown-gfx1200"

Disassembly of section .text:

0000000000001800 <_Z7squaredf>:
	s_wait_loadcnt_dscnt 0x0                                   // 000000001800: BFC80000
	s_wait_expcnt 0x0                                          // 000000001804: BFC40000
	s_wait_samplecnt 0x0                                       // 000000001808: BFC20000
	s_wait_bvhcnt 0x0                                          // 00000000180C: BFC30000
	s_wait_kmcnt 0x0                                           // 000000001810: BFC70000
	s_xor_saveexec_b32 s0, -1                                  // 000000001814: BE8024C1
	scratch_store_b32 off, v1, s32                             // 000000001818: ED068020 00800000 00000000
	s_wait_alu 0xfffe                                          // 000000001824: BF88FFFE
	s_mov_b32 exec_lo, s0                                      // 000000001828: BEFE0000
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000182C: BF870009
	v_writelane_b32 v1, exec_lo, 0                             // 000000001830: D7610001 0001007E
	v_mul_f32_e32 v0, v0, v0                                   // 000000001838: 10000100
	s_ttracedata_imm 0x7                                       // 00000000183C: BFBB0007
	s_xor_saveexec_b32 s0, -1                                  // 000000001840: BE8024C1
	scratch_load_b32 v1, off, s32                              // 000000001844: ED050020 00000001 00000000
	s_wait_alu 0xfffe                                          // 000000001850: BF88FFFE
	s_mov_b32 exec_lo, s0                                      // 000000001854: BEFE0000
	s_wait_loadcnt 0x0                                         // 000000001858: BFC00000
	s_setpc_b64 s[30:31]                                       // 00000000185C: BE80481E

0000000000001860 <_Z3rmsff>:
	s_wait_loadcnt_dscnt 0x0                                   // 000000001860: BFC80000
	s_wait_expcnt 0x0                                          // 000000001864: BFC40000
	s_wait_samplecnt 0x0                                       // 000000001868: BFC20000
	s_wait_bvhcnt 0x0                                          // 00000000186C: BFC30000
	s_wait_kmcnt 0x0                                           // 000000001870: BFC70000
	s_mov_b32 s1, s33                                          // 000000001874: BE810021
	s_mov_b32 s33, s32                                         // 000000001878: BEA10020
	s_xor_saveexec_b32 s0, -1                                  // 00000000187C: BE8024C1
	scratch_store_b32 off, v6, s33                             // 000000001880: ED068021 03000000 00000000
	s_wait_alu 0xfffe                                          // 00000000188C: BF88FFFE
	s_mov_b32 exec_lo, s0                                      // 000000001890: BEFE0000
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001894: BF870009
	v_writelane_b32 v6, exec_lo, 2                             // 000000001898: D7610006 0001047E
	v_writelane_b32 v6, s30, 0                                 // 0000000018A0: D7610006 0001001E
	s_add_co_i32 s32, s32, 16                                  // 0000000018A8: 81209020
	v_writelane_b32 v6, s31, 1                                 // 0000000018AC: D7610006 0001021F
	v_mov_b32_e32 v2, v1                                       // 0000000018B4: 7E040301
	s_ttracedata                                               // 0000000018B8: BFBA0000
	s_nop 0                                                    // 0000000018BC: BF800000
	s_getpc_b64 s[2:3]                                         // 0000000018C0: BE824700
	s_wait_alu 0xfffe                                          // 0000000018C4: BF88FFFE
	s_sext_i32_i16 s3, s3                                      // 0000000018C8: BE830F03
	s_add_co_u32 s2, s2, 0xffffff3c                            // 0000000018CC: 8002FF02 FFFFFF3C
	s_wait_alu 0xfffe                                          // 0000000018D4: BF88FFFE
	s_add_co_ci_u32 s3, s3, lit(0xffffffff)                    // 0000000018D8: 8203FF03 FFFFFFFF
	s_wait_alu 0xfffe                                          // 0000000018E0: BF88FFFE
	s_swappc_b64 s[30:31], s[2:3]                              // 0000000018E4: BE9E4902
	v_dual_mov_b32 v3, v0 :: v_dual_mov_b32 v0, v2             // 0000000018E8: CA100100 03000102
	s_wait_alu 0xfffe                                          // 0000000018F0: BF88FFFE
	s_swappc_b64 s[30:31], s[2:3]                              // 0000000018F4: BE9E4902
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)// 0000000018F8: BF8701C1
	v_add_f32_e32 v0, v3, v0                                   // 0000000018FC: 06000103
	v_readlane_b32 s30, v6, 0                                  // 000000001900: D760001E 00010106
	v_readlane_b32 s31, v6, 1                                  // 000000001908: D760001F 00010306
	s_mov_b32 s32, s33                                         // 000000001910: BEA00021
	v_mul_f32_e32 v1, 0x4f800000, v0                           // 000000001914: 100200FF 4F800000
	v_cmp_gt_f32_e32 vcc_lo, 0xf800000, v0                     // 00000000191C: 7C2800FF 0F800000
	s_wait_alu 0xfffd                                          // 000000001924: BF88FFFD
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001928: BF870092
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 00000000192C: 02000300
	v_sqrt_f32_e32 v1, v0                                      // 000000001930: 7E026700
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001934: BF870125
	v_add_nc_u32_e32 v2, -1, v1                                // 000000001938: 4A0402C1
	v_add_nc_u32_e32 v3, 1, v1                                 // 00000000193C: 4A060281
	v_fma_f32 v4, -v2, v1, v0                                  // 000000001940: D6130004 24020302
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000001948: BF870112
	v_fma_f32 v5, -v3, v1, v0                                  // 00000000194C: D6130005 24020303
	v_cmp_ge_f32_e64 s0, 0, v4                                 // 000000001954: D4160000 00020880
	s_wait_alu 0xf1ff                                          // 00000000195C: BF88F1FF
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000001960: BF870191
	v_cndmask_b32_e64 v1, v1, v2, s0                           // 000000001964: D5010001 00020501
	v_cmp_lt_f32_e64 s0, 0, v5                                 // 00000000196C: D4110000 00020A80
	s_wait_alu 0xf1ff                                          // 000000001974: BF88F1FF
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001978: BF870091
	v_cndmask_b32_e64 v1, v1, v3, s0                           // 00000000197C: D5010001 00020701
	v_mul_f32_e32 v2, 0x37800000, v1                           // 000000001984: 100402FF 37800000
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 00000000198C: BF870131
	v_cndmask_b32_e32 v1, v1, v2, vcc_lo                       // 000000001990: 02020501
	v_cmp_class_f32_e64 vcc_lo, v0, 0x260                      // 000000001994: D47E006A 0001FF00 00000260
	s_wait_alu 0xfffd                                          // 0000000019A0: BF88FFFD
	v_cndmask_b32_e32 v0, v1, v0, vcc_lo                       // 0000000019A4: 02000101
	s_xor_saveexec_b32 s0, -1                                  // 0000000019A8: BE8024C1
	scratch_load_b32 v6, off, s33                              // 0000000019AC: ED050021 00000006 00000000
	s_wait_alu 0xfffe                                          // 0000000019B8: BF88FFFE
	s_mov_b32 exec_lo, s0                                      // 0000000019BC: BEFE0000
	s_mov_b32 s33, s1                                          // 0000000019C0: BEA10001
	s_wait_loadcnt 0x0                                         // 0000000019C4: BFC00000
	s_wait_alu 0xfffe                                          // 0000000019C8: BF88FFFE
	s_setpc_b64 s[30:31]                                       // 0000000019CC: BE80481E
	s_nop 0                                                    // 0000000019D0: BF800000
	s_nop 0                                                    // 0000000019D4: BF800000
	s_nop 0                                                    // 0000000019D8: BF800000
	s_nop 0                                                    // 0000000019DC: BF800000
	s_nop 0                                                    // 0000000019E0: BF800000
	s_nop 0                                                    // 0000000019E4: BF800000
	s_nop 0                                                    // 0000000019E8: BF800000
	s_nop 0                                                    // 0000000019EC: BF800000
	s_nop 0                                                    // 0000000019F0: BF800000
	s_nop 0                                                    // 0000000019F4: BF800000
	s_nop 0                                                    // 0000000019F8: BF800000
	s_nop 0                                                    // 0000000019FC: BF800000

0000000000001a00 <_Z9vectoropsv>:
	s_load_b32 s0, s[0:1], 0xc                                 // 000000001A00: F4000000 F800000C
	v_and_b32_e32 v1, 0x3ff, v0                                // 000000001A08: 360200FF 000003FF
	v_bfe_u32 v0, v0, 10, 10                                   // 000000001A10: D6100000 02291500
	s_mov_b32 s32, 0                                           // 000000001A18: BEA00080
	global_wb scope:SCOPE_SE                                   // 000000001A1C: EE0B007C 00040000 00000000
	s_wait_kmcnt 0x0                                           // 000000001A28: BFC70000
	s_barrier_signal -1                                        // 000000001A2C: BE804EC1
	s_barrier_wait 0xffff                                      // 000000001A30: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000001A34: EE0AC07C 00040000 00000000
	s_lshr_b32 s1, s0, 16                                      // 000000001A40: 85019000
	s_and_b32 s0, s0, 0xffff                                   // 000000001A44: 8B00FF00 0000FFFF
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001A4C: BF870099
	v_mad_co_u64_u32 v[1:2], null, ttmp9, s0, v[1:2]           // 000000001A50: D6FE7C01 04040075
	v_mad_co_u64_u32 v[2:3], null, ttmp7, s1, v[0:1]           // 000000001A58: D6FE7C02 04000273
	v_cvt_f32_i32_e32 v0, v1                                   // 000000001A60: 7E000B01
	s_getpc_b64 s[0:1]                                         // 000000001A64: BE804700
	s_wait_alu 0xfffe                                          // 000000001A68: BF88FFFE
	s_sext_i32_i16 s1, s1                                      // 000000001A6C: BE810F01
	s_add_co_u32 s0, s0, 0xfffffdf8                            // 000000001A70: 8000FF00 FFFFFDF8
	s_wait_alu 0xfffe                                          // 000000001A78: BF88FFFE
	s_add_co_ci_u32 s1, s1, lit(0xffffffff)                    // 000000001A7C: 8201FF01 FFFFFFFF
	v_cvt_f32_i32_e32 v1, v2                                   // 000000001A84: 7E020B02
	s_wait_alu 0xfffe                                          // 000000001A88: BF88FFFE
	s_swappc_b64 s[30:31], s[0:1]                              // 000000001A8C: BE9E4900
	s_mov_b32 s2, 0                                            // 000000001A90: BE820080
	s_mov_b32 s1, 0                                            // 000000001A94: BE810080
	s_mov_b32 s3, 0                                            // 000000001A98: BE830080
	s_mov_b32 s4, 0                                            // 000000001A9C: BE840080
	s_mov_b32 s5, 0                                            // 000000001AA0: BE850080
	s_mov_b32 s6, 0                                            // 000000001AA4: BE860080
	s_mov_b32 s7, 0                                            // 000000001AA8: BE870080
	s_mov_b32 s8, 0                                            // 000000001AAC: BE880080
	s_mov_b32 s9, 0                                            // 000000001AB0: BE890080
	s_mov_b32 s10, 0                                           // 000000001AB4: BE8A0080
	s_mov_b32 s11, 0                                           // 000000001AB8: BE8B0080
	s_mov_b32 s12, 0                                           // 000000001ABC: BE8C0080
	s_mov_b32 s14, 0                                           // 000000001AC0: BE8E0080
	s_mov_b32 s16, 0                                           // 000000001AC4: BE900080
	s_mov_b32 s13, 0                                           // 000000001AC8: BE8D0080
	s_mov_b32 s17, 0                                           // 000000001ACC: BE910080
	s_mov_b32 s15, 0                                           // 000000001AD0: BE8F0080
	global_wb scope:SCOPE_SE                                   // 000000001AD4: EE0B007C 00040000 00000000
	s_wait_storecnt 0x0                                        // 000000001AE0: BFC10000
	s_barrier_signal -1                                        // 000000001AE4: BE804EC1
	s_barrier_wait 0xffff                                      // 000000001AE8: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000001AEC: EE0AC07C 00040000 00000000
	s_branch 624                                               // 000000001AF8: BFA00270 <_Z9vectoropsv+0xabc>
	v_cmp_gt_f32_e32 vcc_lo, 0xf800000, v0                     // 000000001AFC: 7C2800FF 0F800000
	s_add_f32 s12, s18, 1.0                                    // 000000001B04: A00CF212
	v_mul_f32_e32 v1, 0x4f800000, v0                           // 000000001B08: 100200FF 4F800000
	s_add_f32 s13, s19, 1.0                                    // 000000001B10: A00DF213
	s_add_f32 s15, s21, 1.0                                    // 000000001B14: A00FF215
	s_add_f32 s14, s20, 1.0                                    // 000000001B18: A00EF214
	s_add_f32 s17, s0, 1.0                                     // 000000001B1C: A011F200
	s_wait_alu 0xfffd                                          // 000000001B20: BF88FFFD
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 000000001B24: 02000300
	s_add_f32 s16, s22, 1.0                                    // 000000001B28: A010F216
	s_add_f32 s5, s5, 1.0                                      // 000000001B2C: A005F205
	s_add_f32 s4, s4, 1.0                                      // 000000001B30: A004F204
	s_add_f32 s7, s7, 1.0                                      // 000000001B34: A007F207
	v_sqrt_f32_e32 v1, v0                                      // 000000001B38: 7E026700
	s_add_f32 s8, s8, 1.0                                      // 000000001B3C: A008F208
	s_add_f32 s9, s9, 1.0                                      // 000000001B40: A009F209
	s_add_f32 s3, s3, 1.0                                      // 000000001B44: A003F203
	s_wait_alu 0xfffe                                          // 000000001B48: BF88FFFE
	v_dual_mov_b32 v8, s7 :: v_dual_mov_b32 v9, s8             // 000000001B4C: CA100007 08080008
	s_add_f32 s11, s11, 1.0                                    // 000000001B54: A00BF20B
	v_dual_mov_b32 v10, s9 :: v_dual_mov_b32 v11, s3           // 000000001B58: CA100009 0A0A0003
	s_add_f32 s2, s2, 1.0                                      // 000000001B60: A002F202
	s_delay_alu instid0(TRANS32_DEP_1)                         // 000000001B64: BF870005
	v_add_nc_u32_e32 v2, -1, v1                                // 000000001B68: 4A0402C1
	v_add_nc_u32_e32 v3, 1, v1                                 // 000000001B6C: 4A060281
	s_add_f32 s10, s10, 1.0                                    // 000000001B70: A00AF20A
	s_add_f32 s6, s6, 1.0                                      // 000000001B74: A006F206
	s_wait_alu 0xfffe                                          // 000000001B78: BF88FFFE
	v_dual_mov_b32 v13, s17 :: v_dual_mov_b32 v14, s11         // 000000001B7C: CA100011 0D0E000B
	v_fma_f32 v4, -v2, v1, v0                                  // 000000001B84: D6130004 24020302
	v_mov_b32_e32 v15, s6                                      // 000000001B8C: 7E1E0206
	s_add_f32 s7, s7, 1.0                                      // 000000001B90: A007F207
	s_add_f32 s11, s11, 1.0                                    // 000000001B94: A00BF20B
	s_add_f32 s8, s8, 1.0                                      // 000000001B98: A008F208
	v_cmp_ge_f32_e64 s0, 0, v4                                 // 000000001B9C: D4160000 00020880
	v_mov_b32_e32 v4, s14                                      // 000000001BA4: 7E08020E
	v_fma_f32 v5, -v3, v1, v0                                  // 000000001BA8: D6130005 24020303
	s_add_f32 s14, s14, 1.0                                    // 000000001BB0: A00EF20E
	s_add_f32 s9, s9, 1.0                                      // 000000001BB4: A009F209
	v_cndmask_b32_e64 v1, v1, v2, s0                           // 000000001BB8: D5010001 00020501
	v_mov_b32_e32 v2, s12                                      // 000000001BC0: 7E04020C
	v_cmp_lt_f32_e64 s0, 0, v5                                 // 000000001BC4: D4110000 00020A80
	v_mov_b32_e32 v5, s15                                      // 000000001BCC: 7E0A020F
	s_add_f32 s12, s12, 1.0                                    // 000000001BD0: A00CF20C
	s_add_f32 s15, s15, 1.0                                    // 000000001BD4: A00FF20F
	s_add_f32 s3, s3, 1.0                                      // 000000001BD8: A003F203
	v_cndmask_b32_e64 v1, v1, v3, s0                           // 000000001BDC: D5010001 00020701
	v_dual_mov_b32 v6, s16 :: v_dual_mov_b32 v3, s13           // 000000001BE4: CA100010 0602000D
	s_add_f32 s13, s13, 1.0                                    // 000000001BEC: A00DF20D
	s_add_f32 s16, s16, 1.0                                    // 000000001BF0: A010F210
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 000000001BF4: BF8700C2
	v_dual_mul_f32 v7, 0x37800000, v1 :: v_dual_mov_b32 v12, s5// 000000001BF8: C8D002FF 070C0005 37800000
	s_add_f32 s5, s5, 1.0                                      // 000000001C04: A005F205
	s_add_f32 s17, s17, 1.0                                    // 000000001C08: A011F211
	s_add_f32 s6, s6, 1.0                                      // 000000001C0C: A006F206
	v_cndmask_b32_e32 v1, v1, v7, vcc_lo                       // 000000001C10: 02020F01
	v_cmp_class_f32_e64 vcc_lo, v0, 0x260                      // 000000001C14: D47E006A 0001FF00 00000260
	v_dual_mov_b32 v7, s4 :: v_dual_mov_b32 v16, s10           // 000000001C20: CA100004 0710000A
	s_add_f32 s4, s4, 1.0                                      // 000000001C28: A004F204
	s_add_f32 s10, s10, 1.0                                    // 000000001C2C: A00AF20A
	s_wait_alu 0xfffd                                          // 000000001C30: BF88FFFD
	v_cndmask_b32_e32 v0, v1, v0, vcc_lo                       // 000000001C34: 02000101
	v_add_f32_e32 v0, v0, v2                                   // 000000001C38: 06000500
	v_add_f32_e32 v0, v0, v3                                   // 000000001C3C: 06000700
	v_add_f32_e32 v0, v0, v4                                   // 000000001C40: 06000900
	v_add_f32_e32 v0, v0, v5                                   // 000000001C44: 06000B00
	v_add_f32_e32 v0, v0, v6                                   // 000000001C48: 06000D00
	v_add_f32_e32 v0, v0, v8                                   // 000000001C4C: 06001100
	v_add_f32_e32 v0, v0, v9                                   // 000000001C50: 06001300
	v_add_f32_e32 v0, v0, v10                                  // 000000001C54: 06001500
	v_add_f32_e32 v0, v0, v11                                  // 000000001C58: 06001700
	v_add_f32_e32 v0, v0, v7                                   // 000000001C5C: 06000F00
	v_add_f32_e32 v0, v0, v12                                  // 000000001C60: 06001900
	v_add_f32_e32 v0, v0, v13                                  // 000000001C64: 06001B00
	v_mov_b32_e32 v1, s2                                       // 000000001C68: 7E020202
	v_add_f32_e32 v0, v0, v1                                   // 000000001C6C: 06000300
	v_add_f32_e32 v0, v0, v14                                  // 000000001C70: 06001D00
	v_add_f32_e32 v0, v0, v15                                  // 000000001C74: 06001F00
	v_add_f32_e32 v0, v0, v16                                  // 000000001C78: 06002100
	v_mul_f32_e32 v1, 0x4f800000, v0                           // 000000001C7C: 100200FF 4F800000
	v_cmp_gt_f32_e32 vcc_lo, 0xf800000, v0                     // 000000001C84: 7C2800FF 0F800000
	s_wait_alu 0xfffe                                          // 000000001C8C: BF88FFFE
	v_dual_mov_b32 v9, s8 :: v_dual_mov_b32 v10, s9            // 000000001C90: CA100008 090A0009
	s_add_f32 s2, s2, 1.0                                      // 000000001C98: A002F202
	v_dual_mov_b32 v11, s3 :: v_dual_mov_b32 v12, s5           // 000000001C9C: CA100003 0B0C0005
	s_wait_alu 0xfffd                                          // 000000001CA4: BF88FFFD
	v_dual_cndmask_b32 v0, v0, v1 :: v_dual_mov_b32 v13, s17   // 000000001CA8: CA500300 000C0011
	v_dual_mov_b32 v16, s10 :: v_dual_mov_b32 v15, s6          // 000000001CB0: CA10000A 100E0006
	s_add_f32 s9, s9, 1.0                                      // 000000001CB8: A009F209
	s_delay_alu instid0(VALU_DEP_2)                            // 000000001CBC: BF870002
	v_sqrt_f32_e32 v1, v0                                      // 000000001CC0: 7E026700
	s_add_f32 s3, s3, 1.0                                      // 000000001CC4: A003F203
	s_add_f32 s8, s8, 1.0                                      // 000000001CC8: A008F208
	s_add_f32 s5, s5, 1.0                                      // 000000001CCC: A005F205
	s_add_f32 s10, s10, 1.0                                    // 000000001CD0: A00AF20A
	s_add_f32 s17, s17, 1.0                                    // 000000001CD4: A011F211
	s_add_f32 s6, s6, 1.0                                      // 000000001CD8: A006F206
	s_add_co_i32 s1, s1, 8                                     // 000000001CDC: 81018801
	v_mov_b32_e32 v6, s16                                      // 000000001CE0: 7E0C0210
	s_add_f32 s16, s16, 1.0                                    // 000000001CE4: A010F210
	v_add_nc_u32_e32 v2, -1, v1                                // 000000001CE8: 4A0402C1
	v_add_nc_u32_e32 v3, 1, v1                                 // 000000001CEC: 4A060281
	s_wait_alu 0xfffe                                          // 000000001CF0: BF88FFFE
	s_cmp_eq_u32 s1, 0x80                                      // 000000001CF4: BF06FF01 00000080
	v_mov_b32_e32 v8, s7                                       // 000000001CFC: 7E100207
	s_add_f32 s7, s7, 1.0                                      // 000000001D00: A007F207
	v_fma_f32 v4, -v2, v1, v0                                  // 000000001D04: D6130004 24020302
	v_mov_b32_e32 v14, s11                                     // 000000001D0C: 7E1C020B
	s_add_f32 s11, s11, 1.0                                    // 000000001D10: A00BF20B
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_3)// 000000001D14: BF8701C2
	v_cmp_ge_f32_e64 s0, 0, v4                                 // 000000001D18: D4160000 00020880
	v_mov_b32_e32 v4, s14                                      // 000000001D20: 7E08020E
	v_fma_f32 v5, -v3, v1, v0                                  // 000000001D24: D6130005 24020303
	s_add_f32 s14, s14, 1.0                                    // 000000001D2C: A00EF20E
	v_cndmask_b32_e64 v1, v1, v2, s0                           // 000000001D30: D5010001 00020501
	v_mov_b32_e32 v2, s12                                      // 000000001D38: 7E04020C
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_2)// 000000001D3C: BF870143
	v_cmp_lt_f32_e64 s0, 0, v5                                 // 000000001D40: D4110000 00020A80
	v_mov_b32_e32 v5, s15                                      // 000000001D48: 7E0A020F
	s_add_f32 s12, s12, 1.0                                    // 000000001D4C: A00CF20C
	s_add_f32 s15, s15, 1.0                                    // 000000001D50: A00FF20F
	v_cndmask_b32_e64 v1, v1, v3, s0                           // 000000001D54: D5010001 00020701
	v_mov_b32_e32 v3, s13                                      // 000000001D5C: 7E06020D
	s_add_f32 s13, s13, 1.0                                    // 000000001D60: A00DF20D
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001D64: BF870092
	v_mul_f32_e32 v7, 0x37800000, v1                           // 000000001D68: 100E02FF 37800000
	v_cndmask_b32_e32 v1, v1, v7, vcc_lo                       // 000000001D70: 02020F01
	v_cmp_class_f32_e64 vcc_lo, v0, 0x260                      // 000000001D74: D47E006A 0001FF00 00000260
	v_mov_b32_e32 v7, s4                                       // 000000001D80: 7E0E0204
	s_add_f32 s4, s4, 1.0                                      // 000000001D84: A004F204
	s_wait_alu 0xfffd                                          // 000000001D88: BF88FFFD
	v_cndmask_b32_e32 v0, v1, v0, vcc_lo                       // 000000001D8C: 02000101
	v_add_f32_e32 v0, v0, v2                                   // 000000001D90: 06000500
	v_add_f32_e32 v0, v0, v3                                   // 000000001D94: 06000700
	v_add_f32_e32 v0, v0, v4                                   // 000000001D98: 06000900
	v_add_f32_e32 v0, v0, v5                                   // 000000001D9C: 06000B00
	v_add_f32_e32 v0, v0, v6                                   // 000000001DA0: 06000D00
	v_add_f32_e32 v0, v0, v8                                   // 000000001DA4: 06001100
	v_add_f32_e32 v0, v0, v9                                   // 000000001DA8: 06001300
	v_add_f32_e32 v0, v0, v10                                  // 000000001DAC: 06001500
	v_add_f32_e32 v0, v0, v11                                  // 000000001DB0: 06001700
	v_add_f32_e32 v0, v0, v7                                   // 000000001DB4: 06000F00
	v_add_f32_e32 v0, v0, v12                                  // 000000001DB8: 06001900
	v_add_f32_e32 v0, v0, v13                                  // 000000001DBC: 06001B00
	v_mov_b32_e32 v1, s2                                       // 000000001DC0: 7E020202
	v_add_f32_e32 v0, v0, v1                                   // 000000001DC4: 06000300
	v_add_f32_e32 v0, v0, v14                                  // 000000001DC8: 06001D00
	v_add_f32_e32 v0, v0, v15                                  // 000000001DCC: 06001F00
	v_add_f32_e32 v0, v0, v16                                  // 000000001DD0: 06002100
	v_mul_f32_e32 v1, 0x4f800000, v0                           // 000000001DD4: 100200FF 4F800000
	v_cmp_gt_f32_e32 vcc_lo, 0xf800000, v0                     // 000000001DDC: 7C2800FF 0F800000
	v_dual_mov_b32 v9, s8 :: v_dual_mov_b32 v12, s5            // 000000001DE4: CA100008 090C0005
	s_add_f32 s2, s2, 1.0                                      // 000000001DEC: A002F202
	v_mov_b32_e32 v6, s16                                      // 000000001DF0: 7E0C0210
	s_wait_alu 0xfffd                                          // 000000001DF4: BF88FFFD
	v_dual_cndmask_b32 v0, v0, v1 :: v_dual_mov_b32 v13, s17   // 000000001DF8: CA500300 000C0011
	v_mov_b32_e32 v15, s6                                      // 000000001E00: 7E1E0206
	s_add_f32 s16, s16, 1.0                                    // 000000001E04: A010F210
	s_add_f32 s5, s5, 1.0                                      // 000000001E08: A005F205
	s_delay_alu instid0(VALU_DEP_2)                            // 000000001E0C: BF870002
	v_sqrt_f32_e32 v1, v0                                      // 000000001E10: 7E026700
	s_add_f32 s8, s8, 1.0                                      // 000000001E14: A008F208
	s_add_f32 s17, s17, 1.0                                    // 000000001E18: A011F211
	s_add_f32 s6, s6, 1.0                                      // 000000001E1C: A006F206
	s_wait_alu 0xfffe                                          // 000000001E20: BF88FFFE
	v_mov_b32_e32 v8, s7                                       // 000000001E24: 7E100207
	s_add_f32 s7, s7, 1.0                                      // 000000001E28: A007F207
	v_mov_b32_e32 v10, s9                                      // 000000001E2C: 7E140209
	s_add_f32 s9, s9, 1.0                                      // 000000001E30: A009F209
	v_mov_b32_e32 v14, s11                                     // 000000001E34: 7E1C020B
	v_mov_b32_e32 v16, s10                                     // 000000001E38: 7E20020A
	v_add_nc_u32_e32 v2, -1, v1                                // 000000001E3C: 4A0402C1
	v_add_nc_u32_e32 v3, 1, v1                                 // 000000001E40: 4A060281
	s_add_f32 s10, s10, 1.0                                    // 000000001E44: A00AF20A
	s_add_f32 s11, s11, 1.0                                    // 000000001E48: A00BF20B
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000001E4C: BF870112
	v_fma_f32 v4, -v2, v1, v0                                  // 000000001E50: D6130004 24020302
	v_fma_f32 v5, -v3, v1, v0                                  // 000000001E58: D6130005 24020303
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 000000001E60: BF870132
	v_cmp_ge_f32_e64 s0, 0, v4                                 // 000000001E64: D4160000 00020880
	v_mov_b32_e32 v4, s14                                      // 000000001E6C: 7E08020E
	s_add_f32 s14, s14, 1.0                                    // 000000001E70: A00EF20E
	v_cndmask_b32_e64 v1, v1, v2, s0                           // 000000001E74: D5010001 00020501
	v_cmp_lt_f32_e64 s0, 0, v5                                 // 000000001E7C: D4110000 00020A80
	v_dual_mov_b32 v2, s12 :: v_dual_mov_b32 v5, s15           // 000000001E84: CA10000C 0204000F
	s_add_f32 s12, s12, 1.0                                    // 000000001E8C: A00CF20C
	s_add_f32 s15, s15, 1.0                                    // 000000001E90: A00FF20F
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 000000001E94: BF870132
	v_cndmask_b32_e64 v1, v1, v3, s0                           // 000000001E98: D5010001 00020701
	v_mov_b32_e32 v3, s13                                      // 000000001EA0: 7E06020D
	s_add_f32 s13, s13, 1.0                                    // 000000001EA4: A00DF20D
	v_mul_f32_e32 v7, 0x37800000, v1                           // 000000001EA8: 100E02FF 37800000
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001EB0: BF870001
	v_cndmask_b32_e32 v1, v1, v7, vcc_lo                       // 000000001EB4: 02020F01
	v_cmp_class_f32_e64 vcc_lo, v0, 0x260                      // 000000001EB8: D47E006A 0001FF00 00000260
	v_mov_b32_e32 v11, s3                                      // 000000001EC4: 7E160203
	v_mov_b32_e32 v7, s4                                       // 000000001EC8: 7E0E0204
	s_add_f32 s3, s3, 1.0                                      // 000000001ECC: A003F203
	s_add_f32 s4, s4, 1.0                                      // 000000001ED0: A004F204
	s_wait_alu 0xfffd                                          // 000000001ED4: BF88FFFD
	v_cndmask_b32_e32 v0, v1, v0, vcc_lo                       // 000000001ED8: 02000101
	v_add_f32_e32 v0, v0, v2                                   // 000000001EDC: 06000500
	v_add_f32_e32 v0, v0, v3                                   // 000000001EE0: 06000700
	v_add_f32_e32 v0, v0, v4                                   // 000000001EE4: 06000900
	v_add_f32_e32 v0, v0, v5                                   // 000000001EE8: 06000B00
	v_add_f32_e32 v0, v0, v6                                   // 000000001EEC: 06000D00
	v_add_f32_e32 v0, v0, v8                                   // 000000001EF0: 06001100
	v_add_f32_e32 v0, v0, v9                                   // 000000001EF4: 06001300
	v_add_f32_e32 v0, v0, v10                                  // 000000001EF8: 06001500
	v_add_f32_e32 v0, v0, v11                                  // 000000001EFC: 06001700
	v_add_f32_e32 v0, v0, v7                                   // 000000001F00: 06000F00
	v_add_f32_e32 v0, v0, v12                                  // 000000001F04: 06001900
	v_add_f32_e32 v0, v0, v13                                  // 000000001F08: 06001B00
	v_mov_b32_e32 v1, s2                                       // 000000001F0C: 7E020202
	v_add_f32_e32 v0, v0, v1                                   // 000000001F10: 06000300
	v_add_f32_e32 v0, v0, v14                                  // 000000001F14: 06001D00
	v_add_f32_e32 v0, v0, v15                                  // 000000001F18: 06001F00
	v_add_f32_e32 v0, v0, v16                                  // 000000001F1C: 06002100
	v_mul_f32_e32 v1, 0x4f800000, v0                           // 000000001F20: 100200FF 4F800000
	v_cmp_gt_f32_e32 vcc_lo, 0xf800000, v0                     // 000000001F28: 7C2800FF 0F800000
	v_mov_b32_e32 v6, s16                                      // 000000001F30: 7E0C0210
	s_add_f32 s2, s2, 1.0                                      // 000000001F34: A002F202
	s_wait_alu 0xfffe                                          // 000000001F38: BF88FFFE
	v_dual_mov_b32 v11, s3 :: v_dual_mov_b32 v14, s11          // 000000001F3C: CA100003 0B0E000B
	s_wait_alu 0xfffd                                          // 000000001F44: BF88FFFD
	v_dual_cndmask_b32 v0, v0, v1 :: v_dual_mov_b32 v13, s17   // 000000001F48: CA500300 000C0011
	s_add_f32 s16, s16, 1.0                                    // 000000001F50: A010F210
	s_add_f32 s11, s11, 1.0                                    // 000000001F54: A00BF20B
	s_add_f32 s3, s3, 1.0                                      // 000000001F58: A003F203
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001F5C: BF870001
	v_sqrt_f32_e32 v1, v0                                      // 000000001F60: 7E026700
	s_add_f32 s17, s17, 1.0                                    // 000000001F64: A011F211
	s_wait_alu 0xfffe                                          // 000000001F68: BF88FFFE
	s_add_f32 s22, s11, 1.0                                    // 000000001F6C: A016F20B
	s_add_f32 s18, s3, 1.0                                     // 000000001F70: A012F203
	v_mov_b32_e32 v10, s9                                      // 000000001F74: 7E140209
	s_add_f32 s9, s9, 1.0                                      // 000000001F78: A009F209
	v_mov_b32_e32 v8, s7                                       // 000000001F7C: 7E100207
	s_add_f32 s7, s7, 1.0                                      // 000000001F80: A007F207
	v_mov_b32_e32 v12, s5                                      // 000000001F84: 7E180205
	s_add_f32 s5, s5, 1.0                                      // 000000001F88: A005F205
	v_add_nc_u32_e32 v2, -1, v1                                // 000000001F8C: 4A0402C1
	v_dual_mov_b32 v16, s10 :: v_dual_add_nc_u32 v3, 1, v1     // 000000001F90: CA20000A 10020281
	s_add_f32 s10, s10, 1.0                                    // 000000001F98: A00AF20A
	s_wait_alu 0xfffe                                          // 000000001F9C: BF88FFFE
	s_add_f32 s20, s5, 1.0                                     // 000000001FA0: A014F205
	v_fma_f32 v4, -v2, v1, v0                                  // 000000001FA4: D6130004 24020302
	v_fma_f32 v5, -v3, v1, v0                                  // 000000001FAC: D6130005 24020303
	s_add_f32 s24, s10, 1.0                                    // 000000001FB4: A018F20A
	s_delay_alu instid0(VALU_DEP_2)                            // 000000001FB8: BF870002
	v_cmp_ge_f32_e64 s0, 0, v4                                 // 000000001FBC: D4160000 00020880
	v_mov_b32_e32 v4, s14                                      // 000000001FC4: 7E08020E
	s_add_f32 s14, s14, 1.0                                    // 000000001FC8: A00EF20E
	v_mov_b32_e32 v15, s6                                      // 000000001FCC: 7E1E0206
	s_add_f32 s6, s6, 1.0                                      // 000000001FD0: A006F206
	v_cndmask_b32_e64 v1, v1, v2, s0                           // 000000001FD4: D5010001 00020501
	v_cmp_lt_f32_e64 s0, 0, v5                                 // 000000001FDC: D4110000 00020A80
	v_dual_mov_b32 v2, s12 :: v_dual_mov_b32 v5, s15           // 000000001FE4: CA10000C 0204000F
	s_add_f32 s12, s12, 1.0                                    // 000000001FEC: A00CF20C
	s_add_f32 s15, s15, 1.0                                    // 000000001FF0: A00FF20F
	s_delay_alu instid0(VALU_DEP_2)                            // 000000001FF4: BF870002
	v_cndmask_b32_e64 v1, v1, v3, s0                           // 000000001FF8: D5010001 00020701
	v_mov_b32_e32 v3, s13                                      // 000000002000: 7E06020D
	s_add_f32 s13, s13, 1.0                                    // 000000002004: A00DF20D
	s_wait_alu 0xfffe                                          // 000000002008: BF88FFFE
	s_add_f32 s23, s6, 1.0                                     // 00000000200C: A017F206
	v_mul_f32_e32 v7, 0x37800000, v1                           // 000000002010: 100E02FF 37800000
	s_delay_alu instid0(VALU_DEP_1)                            // 000000002018: BF870001
	v_cndmask_b32_e32 v1, v1, v7, vcc_lo                       // 00000000201C: 02020F01
	v_cmp_class_f32_e64 vcc_lo, v0, 0x260                      // 000000002020: D47E006A 0001FF00 00000260
	v_mov_b32_e32 v9, s8                                       // 00000000202C: 7E120208
	v_mov_b32_e32 v7, s4                                       // 000000002030: 7E0E0204
	s_add_f32 s8, s8, 1.0                                      // 000000002034: A008F208
	s_add_f32 s4, s4, 1.0                                      // 000000002038: A004F204
	s_wait_alu 0xfffd                                          // 00000000203C: BF88FFFD
	v_cndmask_b32_e32 v0, v1, v0, vcc_lo                       // 000000002040: 02000101
	v_add_f32_e32 v0, v0, v2                                   // 000000002044: 06000500
	v_add_f32_e32 v0, v0, v3                                   // 000000002048: 06000700
	v_add_f32_e32 v0, v0, v4                                   // 00000000204C: 06000900
	v_add_f32_e32 v0, v0, v5                                   // 000000002050: 06000B00
	v_add_f32_e32 v0, v0, v6                                   // 000000002054: 06000D00
	v_add_f32_e32 v0, v0, v8                                   // 000000002058: 06001100
	v_add_f32_e32 v0, v0, v9                                   // 00000000205C: 06001300
	v_add_f32_e32 v0, v0, v10                                  // 000000002060: 06001500
	v_add_f32_e32 v0, v0, v11                                  // 000000002064: 06001700
	v_add_f32_e32 v0, v0, v7                                   // 000000002068: 06000F00
	v_add_f32_e32 v0, v0, v12                                  // 00000000206C: 06001900
	v_add_f32_e32 v0, v0, v13                                  // 000000002070: 06001B00
	v_mov_b32_e32 v1, s2                                       // 000000002074: 7E020202
	v_add_f32_e32 v0, v0, v1                                   // 000000002078: 06000300
	v_add_f32_e32 v0, v0, v14                                  // 00000000207C: 06001D00
	v_add_f32_e32 v0, v0, v15                                  // 000000002080: 06001F00
	v_add_f32_e32 v0, v0, v16                                  // 000000002084: 06002100
	v_dual_mul_f32 v1, 0x4f800000, v0 :: v_dual_mov_b32 v6, s16// 000000002088: C8D000FF 01060010 4F800000
	v_cmp_gt_f32_e32 vcc_lo, 0xf800000, v0                     // 000000002094: 7C2800FF 0F800000
	s_wait_alu 0xfffe                                          // 00000000209C: BF88FFFE
	v_dual_mov_b32 v8, s7 :: v_dual_mov_b32 v9, s8             // 0000000020A0: CA100007 08080008
	v_mov_b32_e32 v14, s11                                     // 0000000020A8: 7E1C020B
	s_add_f32 s2, s2, 1.0                                      // 0000000020AC: A002F202
	s_wait_alu 0xfffd                                          // 0000000020B0: BF88FFFD
	v_dual_cndmask_b32 v0, v0, v1 :: v_dual_mov_b32 v11, s3    // 0000000020B4: CA500300 000A0003
	v_mov_b32_e32 v16, s10                                     // 0000000020BC: 7E20020A
	v_mov_b32_e32 v12, s5                                      // 0000000020C0: 7E180205
	s_add_f32 s16, s16, 1.0                                    // 0000000020C4: A010F210
	s_delay_alu instid0(VALU_DEP_3)                            // 0000000020C8: BF870003
	v_sqrt_f32_e32 v1, v0                                      // 0000000020CC: 7E026700
	v_mov_b32_e32 v13, s17                                     // 0000000020D0: 7E1A0211
	v_dual_mov_b32 v15, s6 :: v_dual_mov_b32 v10, s9           // 0000000020D4: CA100006 0F0A0009
	s_add_f32 s7, s7, 1.0                                      // 0000000020DC: A007F207
	s_add_f32 s9, s9, 1.0                                      // 0000000020E0: A009F209
	s_wait_alu 0xfffe                                          // 0000000020E4: BF88FFFE
	s_add_f32 s21, s2, 1.0                                     // 0000000020E8: A015F202
	s_add_f32 s8, s8, 1.0                                      // 0000000020EC: A008F208
	s_add_f32 s19, s4, 1.0                                     // 0000000020F0: A013F204
	s_add_f32 s17, s17, 1.0                                    // 0000000020F4: A011F211
	v_add_nc_u32_e32 v2, -1, v1                                // 0000000020F8: 4A0402C1
	v_add_nc_u32_e32 v3, 1, v1                                 // 0000000020FC: 4A060281
	s_add_f32 s6, s16, 1.0                                     // 000000002100: A006F210
	s_add_f32 s10, s18, 1.0                                    // 000000002104: A00AF212
	s_wait_alu 0xfffe                                          // 000000002108: BF88FFFE
	s_add_f32 s11, s19, 1.0                                    // 00000000210C: A00BF213
	v_fma_f32 v4, -v2, v1, v0                                  // 000000002110: D6130004 24020302
	v_fma_f32 v5, -v3, v1, v0                                  // 000000002118: D6130005 24020303
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 000000002120: BF870132
	v_cmp_ge_f32_e64 s0, 0, v4                                 // 000000002124: D4160000 00020880
	v_mov_b32_e32 v4, s14                                      // 00000000212C: 7E08020E
	s_add_f32 s14, s14, 1.0                                    // 000000002130: A00EF20E
	v_cndmask_b32_e64 v1, v1, v2, s0                           // 000000002134: D5010001 00020501
	v_cmp_lt_f32_e64 s0, 0, v5                                 // 00000000213C: D4110000 00020A80
	v_mov_b32_e32 v2, s12                                      // 000000002144: 7E04020C
	s_add_f32 s12, s12, 1.0                                    // 000000002148: A00CF20C
	v_mov_b32_e32 v5, s15                                      // 00000000214C: 7E0A020F
	s_add_f32 s15, s15, 1.0                                    // 000000002150: A00FF20F
	v_cndmask_b32_e64 v1, v1, v3, s0                           // 000000002154: D5010001 00020701
	v_mov_b32_e32 v3, s13                                      // 00000000215C: 7E06020D
	s_add_f32 s13, s13, 1.0                                    // 000000002160: A00DF20D
	s_wait_alu 0xfffe                                          // 000000002164: BF88FFFE
	s_add_f32 s5, s15, 1.0                                     // 000000002168: A005F20F
	v_mul_f32_e32 v7, 0x37800000, v1                           // 00000000216C: 100E02FF 37800000
	s_add_f32 s3, s13, 1.0                                     // 000000002174: A003F20D
	s_delay_alu instid0(VALU_DEP_1)                            // 000000002178: BF870001
	v_cndmask_b32_e32 v1, v1, v7, vcc_lo                       // 00000000217C: 02020F01
	v_cmp_class_f32_e64 vcc_lo, v0, 0x260                      // 000000002180: D47E006A 0001FF00 00000260
	v_mov_b32_e32 v7, s4                                       // 00000000218C: 7E0E0204
	s_add_f32 s4, s14, 1.0                                     // 000000002190: A004F20E
	s_wait_alu 0xfffd                                          // 000000002194: BF88FFFD
	v_cndmask_b32_e32 v0, v1, v0, vcc_lo                       // 000000002198: 02000101
	v_add_f32_e32 v0, v0, v2                                   // 00000000219C: 06000500
	v_add_f32_e32 v0, v0, v3                                   // 0000000021A0: 06000700
	v_add_f32_e32 v0, v0, v4                                   // 0000000021A4: 06000900
	v_add_f32_e32 v0, v0, v5                                   // 0000000021A8: 06000B00
	v_add_f32_e32 v0, v0, v6                                   // 0000000021AC: 06000D00
	v_add_f32_e32 v0, v0, v8                                   // 0000000021B0: 06001100
	v_add_f32_e32 v0, v0, v9                                   // 0000000021B4: 06001300
	v_add_f32_e32 v0, v0, v10                                  // 0000000021B8: 06001500
	v_add_f32_e32 v0, v0, v11                                  // 0000000021BC: 06001700
	v_add_f32_e32 v0, v0, v7                                   // 0000000021C0: 06000F00
	v_add_f32_e32 v0, v0, v12                                  // 0000000021C4: 06001900
	v_add_f32_e32 v0, v0, v13                                  // 0000000021C8: 06001B00
	v_mov_b32_e32 v1, s2                                       // 0000000021CC: 7E020202
	v_add_f32_e32 v0, v0, v1                                   // 0000000021D0: 06000300
	v_add_f32_e32 v0, v0, v14                                  // 0000000021D4: 06001D00
	v_mov_b32_e32 v6, s16                                      // 0000000021D8: 7E0C0210
	v_add_f32_e32 v0, v0, v15                                  // 0000000021DC: 06001F00
	v_add_f32_e32 v0, v0, v16                                  // 0000000021E0: 06002100
	v_dual_mul_f32 v1, 0x4f800000, v0 :: v_dual_mov_b32 v8, s7 // 0000000021E4: C8D000FF 01080007 4F800000
	v_cmp_gt_f32_e32 vcc_lo, 0xf800000, v0                     // 0000000021F0: 7C2800FF 0F800000
	v_mov_b32_e32 v10, s9                                      // 0000000021F8: 7E140209
	s_add_f32 s2, s12, 1.0                                     // 0000000021FC: A002F20C
	v_dual_mov_b32 v9, s8 :: v_dual_mov_b32 v16, s24           // 000000002200: CA100008 09100018
	s_wait_alu 0xfffd                                          // 000000002208: BF88FFFD
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 00000000220C: 02000300
	v_dual_mov_b32 v12, s20 :: v_dual_mov_b32 v11, s18         // 000000002210: CA100014 0C0A0012
	s_add_f32 s7, s7, 1.0                                      // 000000002218: A007F207
	v_mov_b32_e32 v13, s17                                     // 00000000221C: 7E1A0211
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(TRANS32_DEP_1)// 000000002220: BF8702D3
	v_sqrt_f32_e32 v1, v0                                      // 000000002224: 7E026700
	v_dual_mov_b32 v14, s22 :: v_dual_mov_b32 v15, s23         // 000000002228: CA100016 0E0E0017
	s_add_f32 s9, s9, 1.0                                      // 000000002230: A009F209
	s_add_f32 s8, s8, 1.0                                      // 000000002234: A008F208
	s_add_f32 s16, s21, 1.0                                    // 000000002238: A010F215
	v_add_nc_u32_e32 v2, -1, v1                                // 00000000223C: 4A0402C1
	v_add_nc_u32_e32 v3, 1, v1                                 // 000000002240: 4A060281
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000002244: BF870112
	v_fma_f32 v4, -v2, v1, v0                                  // 000000002248: D6130004 24020302
	v_fma_f32 v5, -v3, v1, v0                                  // 000000002250: D6130005 24020303
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)// 000000002258: BF870142
	v_cmp_ge_f32_e64 s0, 0, v4                                 // 00000000225C: D4160000 00020880
	v_mov_b32_e32 v4, s14                                      // 000000002264: 7E08020E
	s_add_f32 s14, s17, 1.0                                    // 000000002268: A00EF211
	s_add_f32 s17, s23, 1.0                                    // 00000000226C: A011F217
	v_cndmask_b32_e64 v1, v1, v2, s0                           // 000000002270: D5010001 00020501
	v_cmp_lt_f32_e64 s0, 0, v5                                 // 000000002278: D4110000 00020A80
	v_mov_b32_e32 v2, s12                                      // 000000002280: 7E04020C
	s_add_f32 s12, s20, 1.0                                    // 000000002284: A00CF214
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 000000002288: BF870132
	v_cndmask_b32_e64 v1, v1, v3, s0                           // 00000000228C: D5010001 00020701
	v_mov_b32_e32 v3, s13                                      // 000000002294: 7E06020D
	s_add_f32 s13, s22, 1.0                                    // 000000002298: A00DF216
	v_mul_f32_e32 v7, 0x37800000, v1                           // 00000000229C: 100E02FF 37800000
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000022A4: BF870001
	v_cndmask_b32_e32 v1, v1, v7, vcc_lo                       // 0000000022A8: 02020F01
	v_cmp_class_f32_e64 vcc_lo, v0, 0x260                      // 0000000022AC: D47E006A 0001FF00 00000260
	v_mov_b32_e32 v5, s15                                      // 0000000022B8: 7E0A020F
	v_mov_b32_e32 v7, s19                                      // 0000000022BC: 7E0E0213
	s_add_f32 s15, s24, 1.0                                    // 0000000022C0: A00FF218
	s_wait_alu 0xfffd                                          // 0000000022C4: BF88FFFD
	v_cndmask_b32_e32 v0, v1, v0, vcc_lo                       // 0000000022C8: 02000101
	v_add_f32_e32 v0, v0, v2                                   // 0000000022CC: 06000500
	v_add_f32_e32 v0, v0, v3                                   // 0000000022D0: 06000700
	v_add_f32_e32 v0, v0, v4                                   // 0000000022D4: 06000900
	v_add_f32_e32 v0, v0, v5                                   // 0000000022D8: 06000B00
	v_add_f32_e32 v0, v0, v6                                   // 0000000022DC: 06000D00
	v_add_f32_e32 v0, v0, v8                                   // 0000000022E0: 06001100
	v_add_f32_e32 v0, v0, v9                                   // 0000000022E4: 06001300
	v_add_f32_e32 v0, v0, v10                                  // 0000000022E8: 06001500
	v_add_f32_e32 v0, v0, v11                                  // 0000000022EC: 06001700
	v_add_f32_e32 v0, v0, v7                                   // 0000000022F0: 06000F00
	v_add_f32_e32 v0, v0, v12                                  // 0000000022F4: 06001900
	v_add_f32_e32 v0, v0, v13                                  // 0000000022F8: 06001B00
	v_dual_mov_b32 v1, s21 :: v_dual_mov_b32 v6, s6            // 0000000022FC: CA100015 01060006
	v_add_f32_e32 v0, v0, v1                                   // 000000002304: 06000300
	v_add_f32_e32 v0, v0, v14                                  // 000000002308: 06001D00
	s_wait_alu 0xfffe                                          // 00000000230C: BF88FFFE
	v_mov_b32_e32 v8, s7                                       // 000000002310: 7E100207
	v_add_f32_e32 v0, v0, v15                                  // 000000002314: 06001F00
	v_add_f32_e32 v0, v0, v16                                  // 000000002318: 06002100
	v_dual_mul_f32 v1, 0x4f800000, v0 :: v_dual_mov_b32 v10, s9// 00000000231C: C8D000FF 010A0009 4F800000
	v_cmp_gt_f32_e32 vcc_lo, 0xf800000, v0                     // 000000002328: 7C2800FF 0F800000
	v_mov_b32_e32 v12, s12                                     // 000000002330: 7E18020C
	v_mov_b32_e32 v14, s13                                     // 000000002334: 7E1C020D
	v_dual_mov_b32 v16, s15 :: v_dual_mov_b32 v9, s8           // 000000002338: CA10000F 10080008
	s_wait_alu 0xfffd                                          // 000000002340: BF88FFFD
	v_dual_cndmask_b32 v0, v0, v1 :: v_dual_mov_b32 v11, s10   // 000000002344: CA500300 000A000A
	v_mov_b32_e32 v13, s14                                     // 00000000234C: 7E1A020E
	v_mov_b32_e32 v15, s17                                     // 000000002350: 7E1E0211
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)// 000000002354: BF870293
	v_sqrt_f32_e32 v1, v0                                      // 000000002358: 7E026700
	v_add_nc_u32_e32 v2, -1, v1                                // 00000000235C: 4A0402C1
	v_add_nc_u32_e32 v3, 1, v1                                 // 000000002360: 4A060281
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000002364: BF870112
	v_fma_f32 v4, -v2, v1, v0                                  // 000000002368: D6130004 24020302
	v_fma_f32 v5, -v3, v1, v0                                  // 000000002370: D6130005 24020303
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 000000002378: BF870132
	v_cmp_ge_f32_e64 s0, 0, v4                                 // 00000000237C: D4160000 00020880
	v_mov_b32_e32 v4, s4                                       // 000000002384: 7E080204
	s_wait_alu 0xf1ff                                          // 000000002388: BF88F1FF
	v_cndmask_b32_e64 v1, v1, v2, s0                           // 00000000238C: D5010001 00020501
	v_cmp_lt_f32_e64 s0, 0, v5                                 // 000000002394: D4110000 00020A80
	v_dual_mov_b32 v2, s2 :: v_dual_mov_b32 v5, s5             // 00000000239C: CA100002 02040005
	s_wait_alu 0xf1ff                                          // 0000000023A4: BF88F1FF
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000023A8: BF870092
	v_cndmask_b32_e64 v1, v1, v3, s0                           // 0000000023AC: D5010001 00020701
	v_mul_f32_e32 v7, 0x37800000, v1                           // 0000000023B4: 100E02FF 37800000
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)// 0000000023BC: BF8701C1
	v_cndmask_b32_e32 v1, v1, v7, vcc_lo                       // 0000000023C0: 02020F01
	v_cmp_class_f32_e64 vcc_lo, v0, 0x260                      // 0000000023C4: D47E006A 0001FF00 00000260
	v_mov_b32_e32 v3, s3                                       // 0000000023D0: 7E060203
	s_wait_alu 0xfffd                                          // 0000000023D4: BF88FFFD
	v_dual_mov_b32 v7, s11 :: v_dual_cndmask_b32 v0, v1, v0    // 0000000023D8: CA12000B 07000101
	v_add_f32_e32 v0, v0, v2                                   // 0000000023E0: 06000500
	v_add_f32_e32 v0, v0, v3                                   // 0000000023E4: 06000700
	v_add_f32_e32 v0, v0, v4                                   // 0000000023E8: 06000900
	v_add_f32_e32 v0, v0, v5                                   // 0000000023EC: 06000B00
	v_add_f32_e32 v0, v0, v6                                   // 0000000023F0: 06000D00
	v_add_f32_e32 v0, v0, v8                                   // 0000000023F4: 06001100
	v_add_f32_e32 v0, v0, v9                                   // 0000000023F8: 06001300
	v_add_f32_e32 v0, v0, v10                                  // 0000000023FC: 06001500
	v_add_f32_e32 v0, v0, v11                                  // 000000002400: 06001700
	v_add_f32_e32 v0, v0, v7                                   // 000000002404: 06000F00
	v_add_f32_e32 v0, v0, v12                                  // 000000002408: 06001900
	v_add_f32_e32 v0, v0, v13                                  // 00000000240C: 06001B00
	v_mov_b32_e32 v1, s16                                      // 000000002410: 7E020210
	v_add_f32_e32 v0, v0, v1                                   // 000000002414: 06000300
	v_add_f32_e32 v0, v0, v14                                  // 000000002418: 06001D00
	v_add_f32_e32 v0, v0, v15                                  // 00000000241C: 06001F00
	v_add_f32_e32 v0, v0, v16                                  // 000000002420: 06002100
	v_mul_f32_e32 v1, 0x4f800000, v0                           // 000000002424: 100200FF 4F800000
	v_cmp_gt_f32_e32 vcc_lo, 0xf800000, v0                     // 00000000242C: 7C2800FF 0F800000
	s_wait_alu 0xfffd                                          // 000000002434: BF88FFFD
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000002438: BF870092
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 00000000243C: 02000300
	v_sqrt_f32_e32 v1, v0                                      // 000000002440: 7E026700
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000002444: BF870125
	v_add_nc_u32_e32 v2, -1, v1                                // 000000002448: 4A0402C1
	v_add_nc_u32_e32 v3, 1, v1                                 // 00000000244C: 4A060281
	v_fma_f32 v4, -v2, v1, v0                                  // 000000002450: D6130004 24020302
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000002458: BF870112
	v_fma_f32 v5, -v3, v1, v0                                  // 00000000245C: D6130005 24020303
	v_cmp_ge_f32_e64 s0, 0, v4                                 // 000000002464: D4160000 00020880
	s_wait_alu 0xf1ff                                          // 00000000246C: BF88F1FF
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000002470: BF870191
	v_cndmask_b32_e64 v1, v1, v2, s0                           // 000000002474: D5010001 00020501
	v_cmp_lt_f32_e64 s0, 0, v5                                 // 00000000247C: D4110000 00020A80
	s_wait_alu 0xf1ff                                          // 000000002484: BF88F1FF
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000002488: BF870091
	v_cndmask_b32_e64 v1, v1, v3, s0                           // 00000000248C: D5010001 00020701
	v_mul_f32_e32 v2, 0x37800000, v1                           // 000000002494: 100402FF 37800000
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 00000000249C: BF870131
	v_cndmask_b32_e32 v1, v1, v2, vcc_lo                       // 0000000024A0: 02020501
	v_cmp_class_f32_e64 vcc_lo, v0, 0x260                      // 0000000024A4: D47E006A 0001FF00 00000260
	s_wait_alu 0xfffd                                          // 0000000024B0: BF88FFFD
	v_cndmask_b32_e32 v0, v1, v0, vcc_lo                       // 0000000024B4: 02000101
	s_cbranch_scc1 63                                          // 0000000024B8: BFA2003F <_Z9vectoropsv+0xbb8>
	s_wait_alu 0xfffe                                          // 0000000024BC: BF88FFFE
	s_add_f32 s18, s2, 1.0                                     // 0000000024C0: A012F202
	s_add_f32 s19, s3, 1.0                                     // 0000000024C4: A013F203
	s_add_f32 s20, s4, 1.0                                     // 0000000024C8: A014F204
	s_add_f32 s22, s6, 1.0                                     // 0000000024CC: A016F206
	s_add_f32 s21, s5, 1.0                                     // 0000000024D0: A015F205
	s_add_f32 s7, s7, 1.0                                      // 0000000024D4: A007F207
	s_wait_alu 0xfffe                                          // 0000000024D8: BF88FFFE
	v_dual_mov_b32 v1, s18 :: v_dual_mov_b32 v2, s19           // 0000000024DC: CA100012 01020013
	v_add_f32_e32 v0, v0, v1                                   // 0000000024E4: 06000300
	v_add_f32_e32 v0, v0, v2                                   // 0000000024E8: 06000500
	s_add_f32 s8, s8, 1.0                                      // 0000000024EC: A008F208
	s_add_f32 s9, s9, 1.0                                      // 0000000024F0: A009F209
	v_dual_mov_b32 v1, s20 :: v_dual_mov_b32 v2, s22           // 0000000024F4: CA100014 01020016
	v_add_f32_e32 v0, v0, v1                                   // 0000000024FC: 06000300
	v_dual_mov_b32 v3, s21 :: v_dual_mov_b32 v4, s7            // 000000002500: CA100015 03040007
	v_add_f32_e32 v0, v0, v3                                   // 000000002508: 06000700
	s_add_f32 s3, s10, 1.0                                     // 00000000250C: A003F20A
	s_add_f32 s5, s12, 1.0                                     // 000000002510: A005F20C
	v_add_f32_e32 v0, v0, v2                                   // 000000002514: 06000500
	s_add_f32 s4, s11, 1.0                                     // 000000002518: A004F20B
	s_add_f32 s0, s14, 1.0                                     // 00000000251C: A000F20E
	v_add_f32_e32 v0, v0, v4                                   // 000000002520: 06000900
	s_wait_alu 0xfffe                                          // 000000002524: BF88FFFE
	v_dual_mov_b32 v1, s8 :: v_dual_mov_b32 v2, s9             // 000000002528: CA100008 01020009
	v_add_f32_e32 v0, v0, v1                                   // 000000002530: 06000300
	v_add_f32_e32 v0, v0, v2                                   // 000000002534: 06000500
	s_add_f32 s2, s16, 1.0                                     // 000000002538: A002F210
	s_add_f32 s6, s17, 1.0                                     // 00000000253C: A006F211
	v_dual_mov_b32 v3, s3 :: v_dual_mov_b32 v4, s5             // 000000002540: CA100003 03040005
	v_add_f32_e32 v0, v0, v3                                   // 000000002548: 06000700
	s_add_f32 s11, s13, 1.0                                    // 00000000254C: A00BF20D
	s_add_f32 s10, s15, 1.0                                    // 000000002550: A00AF20F
	v_dual_mov_b32 v1, s4 :: v_dual_mov_b32 v2, s0             // 000000002554: CA100004 01020000
	v_add_f32_e32 v0, v0, v1                                   // 00000000255C: 06000300
	v_add_f32_e32 v0, v0, v4                                   // 000000002560: 06000900
	v_add_f32_e32 v0, v0, v2                                   // 000000002564: 06000500
	s_wait_alu 0xfffe                                          // 000000002568: BF88FFFE
	v_dual_mov_b32 v1, s2 :: v_dual_mov_b32 v2, s6             // 00000000256C: CA100002 01020006
	v_add_f32_e32 v0, v0, v1                                   // 000000002574: 06000300
	v_dual_mov_b32 v3, s11 :: v_dual_mov_b32 v4, s10           // 000000002578: CA10000B 0304000A
	v_add_f32_e32 v0, v0, v3                                   // 000000002580: 06000700
	s_bitcmp0_b32 s1, 3                                        // 000000002584: BF0C8301
	v_add_f32_e32 v0, v0, v2                                   // 000000002588: 06000500
	v_add_f32_e32 v0, v0, v4                                   // 00000000258C: 06000900
	s_cbranch_scc0 64858                                       // 000000002590: BFA1FD5A <_Z9vectoropsv+0xfc>
	global_wb scope:SCOPE_SE                                   // 000000002594: EE0B007C 00040000 00000000
	s_barrier_signal -1                                        // 0000000025A0: BE804EC1
	s_barrier_wait 0xffff                                      // 0000000025A4: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 0000000025A8: EE0AC07C 00040000 00000000
	s_branch 64849                                             // 0000000025B4: BFA0FD51 <_Z9vectoropsv+0xfc>
	s_endpgm                                                   // 0000000025B8: BFB00000
	s_code_end                                                 // 0000000025BC: BF9F0000
	s_code_end                                                 // 0000000025C0: BF9F0000
	s_code_end                                                 // 0000000025C4: BF9F0000
	s_code_end                                                 // 0000000025C8: BF9F0000
	s_code_end                                                 // 0000000025CC: BF9F0000
	s_code_end                                                 // 0000000025D0: BF9F0000
	s_code_end                                                 // 0000000025D4: BF9F0000
	s_code_end                                                 // 0000000025D8: BF9F0000
	s_code_end                                                 // 0000000025DC: BF9F0000
	s_code_end                                                 // 0000000025E0: BF9F0000
	s_code_end                                                 // 0000000025E4: BF9F0000
	s_code_end                                                 // 0000000025E8: BF9F0000
	s_code_end                                                 // 0000000025EC: BF9F0000
	s_code_end                                                 // 0000000025F0: BF9F0000
	s_code_end                                                 // 0000000025F4: BF9F0000
	s_code_end                                                 // 0000000025F8: BF9F0000
	s_code_end                                                 // 0000000025FC: BF9F0000
	s_code_end                                                 // 000000002600: BF9F0000
	s_code_end                                                 // 000000002604: BF9F0000
	s_code_end                                                 // 000000002608: BF9F0000
	s_code_end                                                 // 00000000260C: BF9F0000
	s_code_end                                                 // 000000002610: BF9F0000
	s_code_end                                                 // 000000002614: BF9F0000
	s_code_end                                                 // 000000002618: BF9F0000
	s_code_end                                                 // 00000000261C: BF9F0000
	s_code_end                                                 // 000000002620: BF9F0000
	s_code_end                                                 // 000000002624: BF9F0000
	s_code_end                                                 // 000000002628: BF9F0000
	s_code_end                                                 // 00000000262C: BF9F0000
	s_code_end                                                 // 000000002630: BF9F0000
	s_code_end                                                 // 000000002634: BF9F0000
	s_code_end                                                 // 000000002638: BF9F0000
	s_code_end                                                 // 00000000263C: BF9F0000
	s_code_end                                                 // 000000002640: BF9F0000
	s_code_end                                                 // 000000002644: BF9F0000
	s_code_end                                                 // 000000002648: BF9F0000
	s_code_end                                                 // 00000000264C: BF9F0000
	s_code_end                                                 // 000000002650: BF9F0000
	s_code_end                                                 // 000000002654: BF9F0000
	s_code_end                                                 // 000000002658: BF9F0000
	s_code_end                                                 // 00000000265C: BF9F0000
	s_code_end                                                 // 000000002660: BF9F0000
	s_code_end                                                 // 000000002664: BF9F0000
	s_code_end                                                 // 000000002668: BF9F0000
	s_code_end                                                 // 00000000266C: BF9F0000
	s_code_end                                                 // 000000002670: BF9F0000
	s_code_end                                                 // 000000002674: BF9F0000
	s_code_end                                                 // 000000002678: BF9F0000
	s_code_end                                                 // 00000000267C: BF9F0000
	s_code_end                                                 // 000000002680: BF9F0000
	s_code_end                                                 // 000000002684: BF9F0000
	s_code_end                                                 // 000000002688: BF9F0000
	s_code_end                                                 // 00000000268C: BF9F0000
	s_code_end                                                 // 000000002690: BF9F0000
	s_code_end                                                 // 000000002694: BF9F0000
	s_code_end                                                 // 000000002698: BF9F0000
	s_code_end                                                 // 00000000269C: BF9F0000
	s_code_end                                                 // 0000000026A0: BF9F0000
	s_code_end                                                 // 0000000026A4: BF9F0000
	s_code_end                                                 // 0000000026A8: BF9F0000
	s_code_end                                                 // 0000000026AC: BF9F0000
	s_code_end                                                 // 0000000026B0: BF9F0000
	s_code_end                                                 // 0000000026B4: BF9F0000
	s_code_end                                                 // 0000000026B8: BF9F0000
	s_code_end                                                 // 0000000026BC: BF9F0000
	s_code_end                                                 // 0000000026C0: BF9F0000
	s_code_end                                                 // 0000000026C4: BF9F0000
	s_code_end                                                 // 0000000026C8: BF9F0000
	s_code_end                                                 // 0000000026CC: BF9F0000
	s_code_end                                                 // 0000000026D0: BF9F0000
	s_code_end                                                 // 0000000026D4: BF9F0000
	s_code_end                                                 // 0000000026D8: BF9F0000
	s_code_end                                                 // 0000000026DC: BF9F0000
	s_code_end                                                 // 0000000026E0: BF9F0000
	s_code_end                                                 // 0000000026E4: BF9F0000
	s_code_end                                                 // 0000000026E8: BF9F0000
	s_code_end                                                 // 0000000026EC: BF9F0000
	s_code_end                                                 // 0000000026F0: BF9F0000
	s_code_end                                                 // 0000000026F4: BF9F0000
	s_code_end                                                 // 0000000026F8: BF9F0000
	s_code_end                                                 // 0000000026FC: BF9F0000
	s_code_end                                                 // 000000002700: BF9F0000
	s_code_end                                                 // 000000002704: BF9F0000
	s_code_end                                                 // 000000002708: BF9F0000
	s_code_end                                                 // 00000000270C: BF9F0000
	s_code_end                                                 // 000000002710: BF9F0000
	s_code_end                                                 // 000000002714: BF9F0000
	s_code_end                                                 // 000000002718: BF9F0000
	s_code_end                                                 // 00000000271C: BF9F0000
	s_code_end                                                 // 000000002720: BF9F0000
	s_code_end                                                 // 000000002724: BF9F0000
	s_code_end                                                 // 000000002728: BF9F0000
	s_code_end                                                 // 00000000272C: BF9F0000
	s_code_end                                                 // 000000002730: BF9F0000
	s_code_end                                                 // 000000002734: BF9F0000
	s_code_end                                                 // 000000002738: BF9F0000
	s_code_end                                                 // 00000000273C: BF9F0000
	s_code_end                                                 // 000000002740: BF9F0000
	s_code_end                                                 // 000000002744: BF9F0000
	s_code_end                                                 // 000000002748: BF9F0000
	s_code_end                                                 // 00000000274C: BF9F0000
	s_code_end                                                 // 000000002750: BF9F0000
	s_code_end                                                 // 000000002754: BF9F0000
	s_code_end                                                 // 000000002758: BF9F0000
	s_code_end                                                 // 00000000275C: BF9F0000
	s_code_end                                                 // 000000002760: BF9F0000
	s_code_end                                                 // 000000002764: BF9F0000
	s_code_end                                                 // 000000002768: BF9F0000
	s_code_end                                                 // 00000000276C: BF9F0000
	s_code_end                                                 // 000000002770: BF9F0000
	s_code_end                                                 // 000000002774: BF9F0000
	s_code_end                                                 // 000000002778: BF9F0000
	s_code_end                                                 // 00000000277C: BF9F0000
