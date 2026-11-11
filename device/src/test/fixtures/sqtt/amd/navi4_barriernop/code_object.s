
code_object.out:	file format elf64-amdgpu
	.amdgcn_target "amdgpu-amd-amdhsa-unknown-gfx1201"

Disassembly of section .text:

0000000000001800 <_Z10barriernopv>:
	s_load_b32 s0, s[0:1], 0xc                                 // 000000001800: F4000000 F800000C
	v_cmp_gt_u32_e32 vcc_lo, 32, v0                            // 000000001808: 7C9800A0
	s_wait_kmcnt 0x0                                           // 00000000180C: BFC70000
	s_and_b32 s0, s0, 0xffff                                   // 000000001810: 8B00FF00 0000FFFF
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001818: BF870129
	v_mad_co_u64_u32 v[2:3], null, ttmp9, s0, v[0:1]           // 00000000181C: D6FE7C02 04000075
	v_lshlrev_b32_e32 v1, 2, v0                                // 000000001824: 30020082
	v_lshlrev_b32_e32 v0, 1, v2                                // 000000001828: 30000481
	s_barrier_signal -1                                        // 00000000182C: BE804EC1
	s_barrier_wait 0xffff                                      // 000000001830: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000001834: EE0AC07C 00040000 00000000
	ds_store_2addr_stride64_b32 v1, v2, v0 offset1:4           // 000000001840: D83C0400 00000201
	s_wait_loadcnt_dscnt 0x0                                   // 000000001848: BFC80000
	s_barrier_signal -1                                        // 00000000184C: BE804EC1
	s_barrier_wait 0xffff                                      // 000000001850: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000001854: EE0AC07C 00040000 00000000
	s_nop 1                                                    // 000000001860: BF800001
	s_nop 1                                                    // 000000001864: BF800001
	s_and_saveexec_b32 s0, vcc_lo                              // 000000001868: BE80206A
	s_cbranch_execz 21                                         // 00000000186C: BFA50015 <_Z10barriernopv+0xc4>
	ds_load_b32 v0, v1 offset:4                                // 000000001870: D8D80004 00000001
	s_wait_dscnt 0x0                                           // 000000001878: BFC60000
	v_cvt_f32_i32_e32 v0, v0                                   // 00000000187C: 7E000B00
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001880: BF870001
	v_mul_f32_e32 v0, 0.5, v0                                  // 000000001884: 100000F0
	s_wait_loadcnt 0x0                                         // 000000001888: BFC00000
	s_barrier_signal -1                                        // 00000000188C: BE804EC1
	s_barrier_wait 0xffff                                      // 000000001890: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000001894: EE0AC07C 00040000 00000000
	s_nop 1                                                    // 0000000018A0: BF800001
	s_wait_loadcnt 0x0                                         // 0000000018A4: BFC00000
	s_barrier_signal -1                                        // 0000000018A8: BE804EC1
	s_barrier_wait 0xffff                                      // 0000000018AC: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 0000000018B0: EE0AC07C 00040000 00000000
	v_add_f32_e64 v0, v0, 1                                    // 0000000018BC: D5030000 00010300
	s_endpgm                                                   // 0000000018C4: BFB00000
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
