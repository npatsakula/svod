
kernel.co:	file format elf64-amdgpu
	.amdgcn_target "amdgpu-amd-amdhsa-unknown-gfx1201"

Disassembly of section .text:

0000000000000100 <.text>:
	s_mov_b32 s1, 8                                            // 000000000100: BE810088
	s_mov_b32 s20, 0                                           // 000000000104: BE940080
	s_min_i32 s30, 0, s0                                       // 000000000108: 891E0080
	s_mov_b32 s40, 0                                           // 00000000010C: BEA80080
	s_mul_i32 s14, s12, 32                                     // 000000000110: 960EA00C
	s_mov_b32 s21, 1                                           // 000000000114: BE950081
	s_min_i32 s31, 1, s0                                       // 000000000118: 891F0081
	s_mov_b32 s41, 1                                           // 00000000011C: BEA90081
	s_mul_i32 s15, s13, 32                                     // 000000000120: 960FA00D
	s_mov_b32 s22, 2                                           // 000000000124: BE960082
	s_min_i32 s32, 2, s0                                       // 000000000128: 89200082
	s_mov_b32 s42, 2                                           // 00000000012C: BEAA0082
	s_mul_i32 s16, s14, 32                                     // 000000000130: 9610A00E
	s_mov_b32 s23, 3                                           // 000000000134: BE970083
	s_min_i32 s33, 3, s0                                       // 000000000138: 89210083
	s_mov_b32 s43, 3                                           // 00000000013C: BEAB0083
	s_mul_i32 s17, s15, 32                                     // 000000000140: 9611A00F
	s_mov_b32 s24, 4                                           // 000000000144: BE980084
	s_min_i32 s34, 4, s0                                       // 000000000148: 89220084
	s_mov_b32 s44, 4                                           // 00000000014C: BEAC0084
	s_mul_i32 s18, s16, 32                                     // 000000000150: 9612A010
	v_mov_b32_e32 v20, 0                                       // 000000000154: 7E280280
	v_lshlrev_b64_e32 v[30:31], 2, v[12:13]                    // 000000000158: 3E3C1882
	v_mad_co_u64_u32 v[40:41], null, v12, v13, v[14:15]        // 00000000015C: D6FE7C28 043A1B0C
	v_mov_b32_e32 v21, 1                                       // 000000000164: 7E2A0281
	v_lshlrev_b64_e32 v[32:33], 2, v[13:14]                    // 000000000168: 3E401A82
	v_mad_co_u64_u32 v[42:43], null, v13, v14, v[15:16]        // 00000000016C: D6FE7C2A 043E1D0D
	v_mov_b32_e32 v22, 2                                       // 000000000174: 7E2C0282
	v_lshlrev_b64_e32 v[34:35], 2, v[14:15]                    // 000000000178: 3E441C82
	v_mad_co_u64_u32 v[44:45], null, v14, v15, v[16:17]        // 00000000017C: D6FE7C2C 04421F0E
	v_mov_b32_e32 v23, 3                                       // 000000000184: 7E2E0283
	v_lshlrev_b64_e32 v[36:37], 2, v[15:16]                    // 000000000188: 3E481E82
	v_mad_co_u64_u32 v[46:47], null, v15, v16, v[17:18]        // 00000000018C: D6FE7C2E 0446210F
	v_mov_b32_e32 v24, 4                                       // 000000000194: 7E300284
	v_lshlrev_b64_e32 v[38:39], 2, v[16:17]                    // 000000000198: 3E4C2082
	v_mad_co_u64_u32 v[48:49], null, v16, v17, v[18:19]        // 00000000019C: D6FE7C30 044A2310
	v_mov_b32_e32 v20, 0                                       // 0000000001A4: 7E280280
	v_s_rcp_f32 s20, s12                                       // 0000000001A8: D6840014 0000000C
	v_rcp_f32_e32 v30, v12                                     // 0000000001B0: 7E3C550C
	v_s_exp_f32 s30, s12                                       // 0000000001B4: D680001E 0000000C
	v_mov_b32_e32 v21, 1                                       // 0000000001BC: 7E2A0281
	v_s_rcp_f32 s21, s13                                       // 0000000001C0: D6840015 0000000D
	v_rcp_f32_e32 v31, v13                                     // 0000000001C8: 7E3E550D
	v_s_exp_f32 s31, s13                                       // 0000000001CC: D680001F 0000000D
	v_mov_b32_e32 v22, 2                                       // 0000000001D4: 7E2C0282
	v_s_rcp_f32 s22, s14                                       // 0000000001D8: D6840016 0000000E
	v_rcp_f32_e32 v32, v14                                     // 0000000001E0: 7E40550E
	v_s_exp_f32 s32, s14                                       // 0000000001E4: D6800020 0000000E
	v_mov_b32_e32 v23, 3                                       // 0000000001EC: 7E2E0283
	v_s_rcp_f32 s23, s15                                       // 0000000001F0: D6840017 0000000F
	v_rcp_f32_e32 v33, v15                                     // 0000000001F8: 7E42550F
	v_s_exp_f32 s33, s15                                       // 0000000001FC: D6800021 0000000F
	v_mov_b32_e32 v24, 4                                       // 000000000204: 7E300284
	v_s_rcp_f32 s24, s16                                       // 000000000208: D6840018 00000010
	v_rcp_f32_e32 v34, v16                                     // 000000000210: 7E445510
	v_s_exp_f32 s34, s16                                       // 000000000214: D6800022 00000010
	v_wmma_f32_16x16x16_f16 v[38:45], v[30:33], v[34:37], v[38:45]// 00000000021C: CC404026 1C9A451E
	v_wmma_i32_16x16x16_iu8 v[50:57], v[46:47], v[48:49], v[50:57]// 000000000224: CC444032 1CCA612E
	v_wmma_f32_16x16x16_f16 v[78:85], v[70:73], v[74:77], v[78:85]// 00000000022C: CC40404E 1D3A9546
	v_wmma_i32_16x16x16_iu8 v[90:97], v[86:87], v[88:89], v[90:97]// 000000000234: CC44405A 1D6AB156
	v_wmma_f32_16x16x16_f16 v[118:125], v[110:113], v[114:117], v[118:125]// 00000000023C: CC404076 1DDAE56E
	v_wmma_i32_16x16x16_iu8 v[130:137], v[126:127], v[128:129], v[130:137]// 000000000244: CC444082 1E0B017E
	v_wmma_f32_16x16x16_f16 v[158:165], v[150:153], v[154:157], v[158:165]// 00000000024C: CC40409E 1E7B3596
	v_wmma_i32_16x16x16_iu8 v[170:177], v[166:167], v[168:169], v[170:177]// 000000000254: CC4440AA 1EAB51A6
	v_wmma_f32_16x16x16_f16 v[198:205], v[190:193], v[194:197], v[198:205]// 00000000025C: CC4040C6 1F1B85BE
	v_wmma_i32_16x16x16_iu8 v[210:217], v[206:207], v[208:209], v[210:217]// 000000000264: CC4440D2 1F4BA1CE
	s_add_co_i32 s1, s1, -1                                    // 00000000026C: 8101C101
	s_cmp_eq_i32 s1, 0                                         // 000000000270: BF008001
	s_cbranch_scc0 65443                                       // 000000000274: BFA1FFA3 <.text+0x4>
	s_endpgm                                                   // 000000000278: BFB00000
	s_code_end                                                 // 00000000027C: BF9F0000
	s_code_end                                                 // 000000000280: BF9F0000
	s_code_end                                                 // 000000000284: BF9F0000
	s_code_end                                                 // 000000000288: BF9F0000
	s_code_end                                                 // 00000000028C: BF9F0000
	s_code_end                                                 // 000000000290: BF9F0000
	s_code_end                                                 // 000000000294: BF9F0000
	s_code_end                                                 // 000000000298: BF9F0000
	s_code_end                                                 // 00000000029C: BF9F0000
	s_code_end                                                 // 0000000002A0: BF9F0000
	s_code_end                                                 // 0000000002A4: BF9F0000
	s_code_end                                                 // 0000000002A8: BF9F0000
	s_code_end                                                 // 0000000002AC: BF9F0000
	s_code_end                                                 // 0000000002B0: BF9F0000
	s_code_end                                                 // 0000000002B4: BF9F0000
	s_code_end                                                 // 0000000002B8: BF9F0000
	s_code_end                                                 // 0000000002BC: BF9F0000
	s_code_end                                                 // 0000000002C0: BF9F0000
	s_code_end                                                 // 0000000002C4: BF9F0000
	s_code_end                                                 // 0000000002C8: BF9F0000
	s_code_end                                                 // 0000000002CC: BF9F0000
	s_code_end                                                 // 0000000002D0: BF9F0000
	s_code_end                                                 // 0000000002D4: BF9F0000
	s_code_end                                                 // 0000000002D8: BF9F0000
	s_code_end                                                 // 0000000002DC: BF9F0000
	s_code_end                                                 // 0000000002E0: BF9F0000
	s_code_end                                                 // 0000000002E4: BF9F0000
	s_code_end                                                 // 0000000002E8: BF9F0000
	s_code_end                                                 // 0000000002EC: BF9F0000
	s_code_end                                                 // 0000000002F0: BF9F0000
	s_code_end                                                 // 0000000002F4: BF9F0000
	s_code_end                                                 // 0000000002F8: BF9F0000
	s_code_end                                                 // 0000000002FC: BF9F0000
	s_code_end                                                 // 000000000300: BF9F0000
	s_code_end                                                 // 000000000304: BF9F0000
	s_code_end                                                 // 000000000308: BF9F0000
	s_code_end                                                 // 00000000030C: BF9F0000
	s_code_end                                                 // 000000000310: BF9F0000
	s_code_end                                                 // 000000000314: BF9F0000
	s_code_end                                                 // 000000000318: BF9F0000
	s_code_end                                                 // 00000000031C: BF9F0000
	s_code_end                                                 // 000000000320: BF9F0000
	s_code_end                                                 // 000000000324: BF9F0000
	s_code_end                                                 // 000000000328: BF9F0000
	s_code_end                                                 // 00000000032C: BF9F0000
	s_code_end                                                 // 000000000330: BF9F0000
	s_code_end                                                 // 000000000334: BF9F0000
	s_code_end                                                 // 000000000338: BF9F0000
	s_code_end                                                 // 00000000033C: BF9F0000
	s_code_end                                                 // 000000000340: BF9F0000
	s_code_end                                                 // 000000000344: BF9F0000
	s_code_end                                                 // 000000000348: BF9F0000
	s_code_end                                                 // 00000000034C: BF9F0000
	s_code_end                                                 // 000000000350: BF9F0000
	s_code_end                                                 // 000000000354: BF9F0000
	s_code_end                                                 // 000000000358: BF9F0000
	s_code_end                                                 // 00000000035C: BF9F0000
	s_code_end                                                 // 000000000360: BF9F0000
	s_code_end                                                 // 000000000364: BF9F0000
	s_code_end                                                 // 000000000368: BF9F0000
	s_code_end                                                 // 00000000036C: BF9F0000
	s_code_end                                                 // 000000000370: BF9F0000
	s_code_end                                                 // 000000000374: BF9F0000
	s_code_end                                                 // 000000000378: BF9F0000
	s_code_end                                                 // 00000000037C: BF9F0000
	s_code_end                                                 // 000000000380: BF9F0000
	s_code_end                                                 // 000000000384: BF9F0000
	s_code_end                                                 // 000000000388: BF9F0000
	s_code_end                                                 // 00000000038C: BF9F0000
	s_code_end                                                 // 000000000390: BF9F0000
	s_code_end                                                 // 000000000394: BF9F0000
	s_code_end                                                 // 000000000398: BF9F0000
	s_code_end                                                 // 00000000039C: BF9F0000
	s_code_end                                                 // 0000000003A0: BF9F0000
	s_code_end                                                 // 0000000003A4: BF9F0000
	s_code_end                                                 // 0000000003A8: BF9F0000
	s_code_end                                                 // 0000000003AC: BF9F0000
	s_code_end                                                 // 0000000003B0: BF9F0000
	s_code_end                                                 // 0000000003B4: BF9F0000
	s_code_end                                                 // 0000000003B8: BF9F0000
	s_code_end                                                 // 0000000003BC: BF9F0000
	s_code_end                                                 // 0000000003C0: BF9F0000
	s_code_end                                                 // 0000000003C4: BF9F0000
	s_code_end                                                 // 0000000003C8: BF9F0000
	s_code_end                                                 // 0000000003CC: BF9F0000
	s_code_end                                                 // 0000000003D0: BF9F0000
	s_code_end                                                 // 0000000003D4: BF9F0000
	s_code_end                                                 // 0000000003D8: BF9F0000
	s_code_end                                                 // 0000000003DC: BF9F0000
	s_code_end                                                 // 0000000003E0: BF9F0000
	s_code_end                                                 // 0000000003E4: BF9F0000
	s_code_end                                                 // 0000000003E8: BF9F0000
	s_code_end                                                 // 0000000003EC: BF9F0000
	s_code_end                                                 // 0000000003F0: BF9F0000
	s_code_end                                                 // 0000000003F4: BF9F0000
	s_code_end                                                 // 0000000003F8: BF9F0000
	s_code_end                                                 // 0000000003FC: BF9F0000
	s_code_end                                                 // 000000000400: BF9F0000
	s_code_end                                                 // 000000000404: BF9F0000
	s_code_end                                                 // 000000000408: BF9F0000
	s_code_end                                                 // 00000000040C: BF9F0000
	s_code_end                                                 // 000000000410: BF9F0000
	s_code_end                                                 // 000000000414: BF9F0000
	s_code_end                                                 // 000000000418: BF9F0000
	s_code_end                                                 // 00000000041C: BF9F0000
	s_code_end                                                 // 000000000420: BF9F0000
	s_code_end                                                 // 000000000424: BF9F0000
	s_code_end                                                 // 000000000428: BF9F0000
	s_code_end                                                 // 00000000042C: BF9F0000
	s_code_end                                                 // 000000000430: BF9F0000
	s_code_end                                                 // 000000000434: BF9F0000
	s_code_end                                                 // 000000000438: BF9F0000
	s_code_end                                                 // 00000000043C: BF9F0000
	s_code_end                                                 // 000000000440: BF9F0000
	s_code_end                                                 // 000000000444: BF9F0000
	s_code_end                                                 // 000000000448: BF9F0000
	s_code_end                                                 // 00000000044C: BF9F0000
	s_code_end                                                 // 000000000450: BF9F0000
	s_code_end                                                 // 000000000454: BF9F0000
	s_code_end                                                 // 000000000458: BF9F0000
	s_code_end                                                 // 00000000045C: BF9F0000
	s_code_end                                                 // 000000000460: BF9F0000
	s_code_end                                                 // 000000000464: BF9F0000
	s_code_end                                                 // 000000000468: BF9F0000
	s_code_end                                                 // 00000000046C: BF9F0000
	s_code_end                                                 // 000000000470: BF9F0000
	s_code_end                                                 // 000000000474: BF9F0000
	s_code_end                                                 // 000000000478: BF9F0000
	s_code_end                                                 // 00000000047C: BF9F0000
	s_code_end                                                 // 000000000480: BF9F0000
	s_code_end                                                 // 000000000484: BF9F0000
	s_code_end                                                 // 000000000488: BF9F0000
