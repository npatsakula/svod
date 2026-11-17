
<stdin>:	file format elf64-amdgpu
	.amdgcn_target "amdgpu-amd-amdhsa-unknown-gfx1201"

Disassembly of section .text:

0000000000000000 <nearest>:
	v_lshrrev_b32_e32 v2, 5, v0                                // 000000000000: 32040085
	v_dual_mov_b32 v66, -1.0 :: v_dual_and_b32 v15, 15, v0     // 000000000004: CA2400F3 420E008F
	v_bfe_u32 v49, v0, 4, 1                                    // 00000000000C: D6100031 02050900
	s_load_b256 s[12:19], s[0:1], 0x0                          // 000000000014: F4006300 F8000000
	s_lshl_b32 s0, ttmp9, 13                                   // 00000000001C: 84008D75
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000000020: BF870112
	v_lshl_or_b32 v48, v2, 4, v15                              // 000000000024: D6560030 043D0902
	v_dual_mov_b32 v103, 0x7f800000 :: v_dual_lshlrev_b32 v70, 3, v49// 00000000002C: CA2200FF 67466283 7F800000
	v_dual_mov_b32 v104, -1.0 :: v_dual_lshlrev_b32 v3, 7, v49 // 000000000038: CA2200F3 68026287
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)// 000000000040: BF8701A3
	v_dual_mov_b32 v68, 0x7f800000 :: v_dual_lshlrev_b32 v1, 7, v48// 000000000044: CA2200FF 44006087 7F800000
	v_dual_mov_b32 v67, -1.0 :: v_dual_lshlrev_b32 v4, 3, v0   // 000000000050: CA2200F3 43040083
	v_lshl_or_b32 v76, v2, 10, v3                              // 000000000058: D656004C 040D1502
	s_delay_alu instid0(VALU_DEP_3)                            // 000000000060: BF870003
	v_or3_b32 v1, v1, s0, v70                                  // 000000000064: D6580001 05180101
	v_or_b32_e32 v77, 2, v49                                   // 00000000006C: 389A6282
	v_or_b32_e32 v78, 4, v49                                   // 000000000070: 389C6284
	v_or_b32_e32 v80, 6, v49                                   // 000000000074: 38A06286
	v_and_or_b32 v75, 0x78, v4, v76                            // 000000000078: D657004B 053208FF 00000078
	v_ashrrev_i32_e32 v2, 31, v1                               // 000000000084: 3404029F
	v_dual_mov_b32 v69, 0x7f800000 :: v_dual_lshlrev_b32 v76, 1, v76// 000000000088: CA2200FF 454C9881 7F800000
	v_lshlrev_b32_e32 v105, 8, v15                             // 000000000094: 30D21E88
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000000098: BF870214
	v_lshlrev_b32_e32 v16, 1, v75                              // 00000000009C: 30209681
	v_lshlrev_b64_e32 v[1:2], 1, v[1:2]                        // 0000000000A0: 3E020281
	v_or_b32_e32 v79, 16, v70                                  // 0000000000A4: 389E8C90
	v_or_b32_e32 v82, 8, v49                                   // 0000000000A8: 38A46288
	v_or_b32_e32 v83, 10, v49                                  // 0000000000AC: 38A6628A
	v_or_b32_e32 v84, 12, v49                                  // 0000000000B0: 38A8628C
	v_or_b32_e32 v85, 14, v49                                  // 0000000000B4: 38AA628E
	s_wait_kmcnt 0x0                                           // 0000000000B8: BFC70000
	v_add_co_u32 v13, vcc_lo, s14, v1                          // 0000000000BC: D7006A0D 0202020E
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000000C4: BF870001
	v_add_co_ci_u32_e64 v14, null, s15, v2, vcc_lo             // 0000000000C8: D5207C0E 01AA040F
	s_clause 0x3                                               // 0000000000D0: BF850003
	global_load_b128 v[1:4], v16, s[16:17]                     // 0000000000D4: EE05C010 00000001 00000010
	global_load_b128 v[5:8], v16, s[16:17] offset:512          // 0000000000E0: EE05C010 00000005 00020010
	global_load_b128 v[9:12], v16, s[16:17] offset:1024        // 0000000000EC: EE05C010 00000009 00040010
	global_load_b128 v[98:101], v16, s[16:17] offset:1536      // 0000000000F8: EE05C010 00000062 00060010
	s_clause 0x7                                               // 000000000104: BF850007
	global_load_b128 v[44:47], v[13:14], off                   // 000000000108: EE05C07C 0000002C 0000000D
	global_load_b128 v[40:43], v[13:14], off offset:32         // 000000000114: EE05C07C 00000028 0000200D
	global_load_b128 v[36:39], v[13:14], off offset:64         // 000000000120: EE05C07C 00000024 0000400D
	global_load_b128 v[32:35], v[13:14], off offset:96         // 00000000012C: EE05C07C 00000020 0000600D
	global_load_b128 v[28:31], v[13:14], off offset:128        // 000000000138: EE05C07C 0000001C 0000800D
	global_load_b128 v[24:27], v[13:14], off offset:160        // 000000000144: EE05C07C 00000018 0000A00D
	global_load_b128 v[20:23], v[13:14], off offset:192        // 000000000150: EE05C07C 00000014 0000C00D
	global_load_b128 v[16:19], v[13:14], off offset:224        // 00000000015C: EE05C07C 00000010 0000E00D
	v_and_b32_e32 v13, 7, v0                                   // 000000000168: 361A0087
	v_and_b32_e32 v0, 16, v0                                   // 00000000016C: 36000090
	v_xor_b32_e32 v14, v15, v49                                // 000000000170: 3A1C630F
	v_dual_mov_b32 v72, -1.0 :: v_dual_mov_b32 v71, -1.0       // 000000000174: CA1000F3 484600F3
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 00000000017C: BF870214
	v_lshlrev_b32_e32 v106, 4, v13                             // 000000000180: 30D41A84
	v_cmp_ne_u32_e64 s0, 0, v0                                 // 000000000184: D44D0000 02020080
	v_xor_b32_e32 v0, v15, v77                                 // 00000000018C: 3A009B0F
	v_xor_b32_e32 v77, v15, v78                                // 000000000190: 3A9A9D0F
	v_xor_b32_e32 v15, v15, v80                                // 000000000194: 3A1EA10F
	v_lshl_or_b32 v93, v14, 4, v76                             // 000000000198: D656005D 0531090E
	v_xor_b32_e32 v81, v49, v13                                // 0000000001A0: 3AA21B31
	v_lshl_or_b32 v94, v0, 4, v76                              // 0000000001A4: D656005E 05310900
	v_lshl_or_b32 v95, v77, 4, v76                             // 0000000001AC: D656005F 0531094D
	v_lshl_or_b32 v96, v15, 4, v76                             // 0000000001B4: D6560060 0531090F
	v_lshlrev_b32_e32 v0, 1, v79                               // 0000000001BC: 30009E81
	v_xor_b32_e32 v78, v78, v13                                // 0000000001C0: 3A9C1B4E
	v_xor_b32_e32 v107, v80, v13                               // 0000000001C4: 3AD61B50
	v_xor_b32_e32 v108, v82, v13                               // 0000000001C8: 3AD81B52
	v_xor_b32_e32 v109, v83, v13                               // 0000000001CC: 3ADA1B53
	v_xor_b32_e32 v110, v84, v13                               // 0000000001D0: 3ADC1B54
	v_xor_b32_e32 v13, v85, v13                                // 0000000001D4: 3A1A1B55
	v_xor_b32_e32 v0, v106, v0                                 // 0000000001D8: 3A00016A
	v_dual_mov_b32 v74, 0x7f800000 :: v_dual_mov_b32 v73, 0x7f800000// 0000000001DC: CA1000FF 4A4800FF 7F800000
	v_dual_mov_b32 v64, -1.0 :: v_dual_mov_b32 v65, 0x7f800000 // 0000000001E8: CA1000F3 404000FF 7F800000
	v_dual_mov_b32 v58, -1.0 :: v_dual_mov_b32 v59, 0x7f800000 // 0000000001F4: CA1000F3 3A3A00FF 7F800000
	v_dual_mov_b32 v61, -1.0 :: v_dual_mov_b32 v60, 0x7f800000 // 000000000200: CA1000F3 3D3C00FF 7F800000
	v_dual_mov_b32 v63, -1.0 :: v_dual_mov_b32 v62, 0x7f800000 // 00000000020C: CA1000F3 3F3E00FF 7F800000
	v_dual_mov_b32 v56, -1.0 :: v_dual_mov_b32 v55, 0x7f800000 // 000000000218: CA1000F3 383600FF 7F800000
	v_dual_mov_b32 v54, -1.0 :: v_dual_mov_b32 v57, 0x7f800000 // 000000000224: CA1000F3 363800FF 7F800000
	v_dual_mov_b32 v51, -1.0 :: v_dual_mov_b32 v50, -1.0       // 000000000230: CA1000F3 333200F3
	v_dual_mov_b32 v52, 0x7f800000 :: v_dual_mov_b32 v53, 0x7f800000// 000000000238: CA1000FF 343400FF 7F800000
	v_cmp_eq_u32_e64 s1, 0, v49                                // 000000000244: D44A0001 02026280
	v_or_b32_e32 v80, 1, v70                                   // 00000000024C: 38A08C81
	v_or_b32_e32 v82, 2, v70                                   // 000000000250: 38A48C82
	v_or_b32_e32 v83, 3, v70                                   // 000000000254: 38A68C83
	v_or_b32_e32 v85, 4, v70                                   // 000000000258: 38AA8C84
	v_or_b32_e32 v86, 5, v70                                   // 00000000025C: 38AC8C85
	v_or_b32_e32 v87, 6, v70                                   // 000000000260: 38AE8C86
	v_or_b32_e32 v89, 7, v70                                   // 000000000264: 38B28C87
	v_or_b32_e32 v90, 17, v70                                  // 000000000268: 38B48C91
	v_or_b32_e32 v91, 18, v70                                  // 00000000026C: 38B68C92
	v_or_b32_e32 v92, 19, v70                                  // 000000000270: 38B88C93
	v_or_b32_e32 v97, 20, v70                                  // 000000000274: 38C28C94
	v_or_b32_e32 v102, 23, v70                                 // 000000000278: 38CC8C97
	v_lshl_or_b32 v88, v78, 4, v105                            // 00000000027C: D6560058 05A5094E
	v_lshl_or_b32 v84, v107, 4, v105                           // 000000000284: D6560054 05A5096B
	v_lshl_or_b32 v78, v109, 4, v105                           // 00000000028C: D656004E 05A5096D
	v_lshl_or_b32 v77, v110, 4, v105                           // 000000000294: D656004D 05A5096E
	v_lshl_or_b32 v76, v13, 4, v105                            // 00000000029C: D656004C 05A5090D
	s_mov_b32 s2, 1                                            // 0000000002A4: BE820081
	s_mov_b32 s3, 0                                            // 0000000002A8: BE830080
	s_mov_b32 s10, 0x76543210                                  // 0000000002AC: BE8A00FF 76543210
	s_xor_b32 s9, s0, -1                                       // 0000000002B4: 8D09C100
	s_wait_loadcnt 0xb                                         // 0000000002B8: BFC0000B
	ds_store_b128 v93, v[1:4]                                  // 0000000002BC: DB7C0000 0000015D
	s_wait_loadcnt 0xa                                         // 0000000002C4: BFC0000A
	ds_store_b128 v94, v[5:8] offset:512                       // 0000000002C8: DB7C0200 0000055E
	s_wait_loadcnt 0x9                                         // 0000000002D0: BFC00009
	ds_store_b128 v95, v[9:12] offset:1024                     // 0000000002D4: DB7C0400 0000095F
	s_wait_loadcnt 0x8                                         // 0000000002DC: BFC00008
	ds_store_b128 v96, v[98:101] offset:1536                   // 0000000002E0: DB7C0600 00006260
	v_or_b32_e32 v100, 21, v70                                 // 0000000002E8: 38C88C95
	v_or_b32_e32 v101, 22, v70                                 // 0000000002EC: 38CA8C96
	v_lshl_or_b32 v98, v81, 4, v105                            // 0000000002F0: D6560062 05A50951
	v_lshl_or_b32 v81, v108, 4, v105                           // 0000000002F8: D6560051 05A5096C
	v_or_b32_e32 v99, v0, v105                                 // 000000000300: 38C6D300
	s_wait_dscnt 0x0                                           // 000000000304: BFC60000
	s_barrier_signal -1                                        // 000000000308: BE804EC1
	s_barrier_wait 0xffff                                      // 00000000030C: BF94FFFF
	s_branch 7                                                 // 000000000310: BFA00007 <nearest+0x330>
	s_add_co_i32 s2, s11, 1                                    // 000000000314: 8102810B
	s_mov_b32 s3, s11                                          // 000000000318: BE83000B
	s_wait_alu depctr_sa_sdst(0)                               // 00000000031C: BF88FF9E
	s_cmp_eq_u32 s2, 0x80                                      // 000000000320: BF06FF02 00000080
	s_barrier_wait 0xffff                                      // 000000000328: BF94FFFF
	s_cbranch_scc1 3006                                        // 00000000032C: BFA20BBE <nearest+0x3228>
	s_mov_b32 s11, s2                                          // 000000000330: BE8B0002
	s_lshl_b32 s2, s2, 13                                      // 000000000334: 84028D02
	s_lshl_b32 s4, s3, 13                                      // 000000000338: 84048D03
	s_wait_alu depctr_sa_sdst(0)                               // 00000000033C: BF88FF9E
	v_lshl_or_b32 v0, v75, 1, s2                               // 000000000340: D6560000 0009034B
	s_and_b32 s4, s4, 0x2000                                   // 000000000348: 8B04FF04 00002000
	s_and_b32 s2, s2, 0x2000                                   // 000000000350: 8B02FF02 00002000
	s_wait_alu depctr_sa_sdst(0)                               // 000000000358: BF88FF9E
	v_or_b32_e32 v1, s4, v99                                   // 00000000035C: 3802C604
	s_clause 0x3                                               // 000000000360: BF850003
	global_load_b128 v[105:108], v0, s[16:17]                  // 000000000364: EE05C010 00000069 00000000
	global_load_b128 v[109:112], v0, s[16:17] offset:512       // 000000000370: EE05C010 0000006D 00020000
	global_load_b128 v[113:116], v0, s[16:17] offset:1024      // 00000000037C: EE05C010 00000071 00040000
	global_load_b128 v[117:120], v0, s[16:17] offset:1536      // 000000000388: EE05C010 00000075 00060000
	v_or_b32_e32 v0, s4, v98                                   // 000000000394: 3800C404
	v_or_b32_e32 v2, s4, v88                                   // 000000000398: 3804B004
	v_or_b32_e32 v3, s4, v84                                   // 00000000039C: 3806A804
	v_or_b32_e32 v4, s4, v81                                   // 0000000003A0: 3808A204
	v_or_b32_e32 v5, s4, v78                                   // 0000000003A4: 380A9C04
	v_or_b32_e32 v6, s4, v77                                   // 0000000003A8: 380C9A04
	v_or_b32_e32 v7, s4, v76                                   // 0000000003AC: 380E9804
	v_add_nc_u32_e32 v177, s2, v93                             // 0000000003B0: 4B62BA02
	v_add_nc_u32_e32 v178, s2, v94                             // 0000000003B4: 4B64BC02
	v_add_nc_u32_e32 v179, s2, v95                             // 0000000003B8: 4B66BE02
	v_add_nc_u32_e32 v180, s2, v96                             // 0000000003BC: 4B68C002
	ds_load_b128 v[8:11], v0                                   // 0000000003C0: DBFC0000 08000000
	ds_load_b128 v[121:124], v0 offset:4096                    // 0000000003C8: DBFC1000 79000000
	ds_load_b128 v[12:15], v1                                  // 0000000003D0: DBFC0000 0C000001
	ds_load_b128 v[125:128], v1 offset:4096                    // 0000000003D8: DBFC1000 7D000001
	ds_load_b128 v[129:132], v2                                // 0000000003E0: DBFC0000 81000002
	ds_load_b128 v[133:136], v2 offset:4096                    // 0000000003E8: DBFC1000 85000002
	ds_load_b128 v[137:140], v3                                // 0000000003F0: DBFC0000 89000003
	ds_load_b128 v[141:144], v3 offset:4096                    // 0000000003F8: DBFC1000 8D000003
	ds_load_b128 v[145:148], v4                                // 000000000400: DBFC0000 91000004
	ds_load_b128 v[149:152], v4 offset:4096                    // 000000000408: DBFC1000 95000004
	ds_load_b128 v[153:156], v5                                // 000000000410: DBFC0000 99000005
	ds_load_b128 v[157:160], v5 offset:4096                    // 000000000418: DBFC1000 9D000005
	ds_load_b128 v[161:164], v6                                // 000000000420: DBFC0000 A1000006
	ds_load_b128 v[165:168], v6 offset:4096                    // 000000000428: DBFC1000 A5000006
	ds_load_b128 v[169:172], v7                                // 000000000430: DBFC0000 A9000007
	ds_load_b128 v[173:176], v7 offset:4096                    // 000000000438: DBFC1000 AD000007
	s_wait_loadcnt_dscnt 0xb0f                                 // 000000000440: BFC80B0F
	v_wmma_f32_16x16x16_f16 v[0:7], v[8:11], v[44:47], 0       // 000000000444: CC404000 1A025908
	s_wait_loadcnt 0x3                                         // 00000000044C: BFC00003
	ds_store_b128 v177, v[105:108]                             // 000000000450: DB7C0000 000069B1
	s_wait_loadcnt 0x2                                         // 000000000458: BFC00002
	ds_store_b128 v178, v[109:112] offset:512                  // 00000000045C: DB7C0200 00006DB2
	s_wait_loadcnt 0x1                                         // 000000000464: BFC00001
	ds_store_b128 v179, v[113:116] offset:1024                 // 000000000468: DB7C0400 000071B3
	s_wait_loadcnt 0x0                                         // 000000000470: BFC00000
	ds_store_b128 v180, v[117:120] offset:1536                 // 000000000474: DB7C0600 000075B4
	s_wait_dscnt 0x0                                           // 00000000047C: BFC60000
	v_wmma_f32_16x16x16_f16 v[0:7], v[12:15], v[40:43], v[0:7] // 000000000480: CC404000 1C02510C
	v_wmma_f32_16x16x16_f16 v[8:15], v[121:124], v[44:47], 0   // 000000000488: CC404008 1A025979
	s_barrier_signal -1                                        // 000000000490: BE804EC1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000000494: BF870112
	v_wmma_f32_16x16x16_f16 v[0:7], v[129:132], v[36:39], v[0:7]// 000000000498: CC404000 1C024981
	v_wmma_f32_16x16x16_f16 v[8:15], v[125:128], v[40:43], v[8:15]// 0000000004A0: CC404008 1C22517D
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000004A8: BF870112
	v_wmma_f32_16x16x16_f16 v[0:7], v[137:140], v[32:35], v[0:7]// 0000000004AC: CC404000 1C024189
	v_wmma_f32_16x16x16_f16 v[8:15], v[133:136], v[36:39], v[8:15]// 0000000004B4: CC404008 1C224985
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000004BC: BF870112
	v_wmma_f32_16x16x16_f16 v[0:7], v[145:148], v[28:31], v[0:7]// 0000000004C0: CC404000 1C023991
	v_wmma_f32_16x16x16_f16 v[8:15], v[141:144], v[32:35], v[8:15]// 0000000004C8: CC404008 1C22418D
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000004D0: BF870112
	v_wmma_f32_16x16x16_f16 v[0:7], v[153:156], v[24:27], v[0:7]// 0000000004D4: CC404000 1C023199
	v_wmma_f32_16x16x16_f16 v[8:15], v[149:152], v[28:31], v[8:15]// 0000000004DC: CC404008 1C223995
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000004E4: BF870112
	v_wmma_f32_16x16x16_f16 v[0:7], v[161:164], v[20:23], v[0:7]// 0000000004E8: CC404000 1C0229A1
	v_wmma_f32_16x16x16_f16 v[8:15], v[157:160], v[24:27], v[8:15]// 0000000004F0: CC404008 1C22319D
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000004F8: BF870112
	v_wmma_f32_16x16x16_f16 v[0:7], v[169:172], v[16:19], v[0:7]// 0000000004FC: CC404000 1C0221A9
	v_wmma_f32_16x16x16_f16 v[8:15], v[165:168], v[20:23], v[8:15]// 000000000504: CC404008 1C2229A5
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000050C: BF870112
	v_min3_num_f32 v121, -v0, -v1, -v2                         // 000000000510: D6290079 E40A0300
	v_wmma_f32_16x16x16_f16 v[8:15], v[173:176], v[16:19], v[8:15]// 000000000518: CC404008 1C2221AD
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000000520: BF870112
	v_min3_num_f32 v121, v121, -v3, -v4                        // 000000000524: D6290079 C4120779
	v_max_num_f32_e64 v122, -v15, -v15                         // 00000000052C: D516007A 62021F0F
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000000534: BF870092
	v_min3_num_f32 v121, v121, -v5, -v6                        // 000000000538: D6290079 C41A0B79
	v_min3_num_f32 v121, v121, -v7, -v8                        // 000000000540: D6290079 C4220F79
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000000548: BF870091
	v_min3_num_f32 v121, v121, -v9, -v10                       // 00000000054C: D6290079 C42A1379
	v_min3_num_f32 v121, v121, -v11, -v12                      // 000000000554: D6290079 C4321779
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000055C: BF870091
	v_min3_num_f32 v121, v121, -v13, -v14                      // 000000000560: D6290079 C43A1B79
	v_min_num_f32_e32 v121, v121, v122                         // 000000000568: 2AF2F579
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000056C: BF870091
	v_mov_b32_e32 v122, v121                                   // 000000000570: 7EF40379
	v_permlanex16_b32 v122, v122, s10, 0xfedcba98              // 000000000574: D65C007A 03FC157A FEDCBA98
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000000580: BF870091
	v_max_num_f32_e64 v122, -v122, -v122                       // 000000000584: D516007A 6202F57A
	v_max_num_f32_e64 v121, -v121, v122                        // 00000000058C: D5160079 2202F579
	s_delay_alu instid0(VALU_DEP_1)                            // 000000000594: BF870001
	v_cmp_ge_f32_e64 s2, v103, -v121                           // 000000000598: D4160002 4202F367
	s_mov_b32 vcc_lo, s2                                       // 0000000005A0: BEEA0002
	s_wait_alu depctr_sa_sdst(0)                               // 0000000005A4: BF88FF9E
	s_cbranch_vccz 65370                                       // 0000000005A8: BFA3FF5A <nearest+0x314>
	v_cmp_ge_f32_e64 s4, v103, -v0                             // 0000000005AC: D4160004 42020167
	v_cmp_ge_f32_e64 s5, v103, -v1                             // 0000000005B4: D4160005 42020367
	s_lshl_b32 s7, s3, 5                                       // 0000000005BC: 84078503
	s_wait_alu depctr_sa_sdst(0)                               // 0000000005C0: BF88FF9E
	v_or_b32_e32 v106, s7, v70                                 // 0000000005C4: 38D48C07
	s_and_b32 s4, s4, s2                                       // 0000000005C8: 8B040204
	s_and_b32 s6, s5, s2                                       // 0000000005CC: 8B060205
	v_or_b32_e32 v107, s7, v80                                 // 0000000005D0: 38D6A007
	s_wait_alu depctr_sa_sdst(0)                               // 0000000005D4: BF88FF9E
	s_or_b32 s3, s4, s6                                        // 0000000005D8: 8C030604
	s_wait_alu depctr_sa_sdst(0)                               // 0000000005DC: BF88FF9E
	v_cndmask_b32_e64 v105, 0, 1, s3                           // 0000000005E0: D5010069 000D0280
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000005E8: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v105                           // 0000000005EC: 7C9AD280
	v_cvt_f32_u32_e32 v105, v106                               // 0000000005F0: 7ED20D6A
	v_cvt_f32_u32_e32 v106, v107                               // 0000000005F4: 7ED40D6B
	s_cbranch_vccz 30                                          // 0000000005F8: BFA3001E <nearest+0x674>
	v_cmp_gt_f32_e64 s8, v103, -v0                             // 0000000005FC: D4140008 42020167
	s_delay_alu instid0(VALU_DEP_3)                            // 000000000604: BF870003
	v_cmp_gt_f32_e32 vcc_lo, v104, v105                        // 000000000608: 7C28D368
	v_cmp_gt_f32_e64 s14, v103, -v1                            // 00000000060C: D414000E 42020367
	v_cmp_gt_f32_e64 s3, v104, v106                            // 000000000614: D4140003 0202D568
	v_cmp_lt_f32_e64 s5, -1.0, v66                             // 00000000061C: D4110005 020284F3
	s_or_b32 s8, vcc_lo, s8                                    // 000000000624: 8C08086A
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v67                         // 000000000628: 7C2286F3
	s_or_b32 s14, s3, s14                                      // 00000000062C: 8C0E0E03
	s_and_b32 s3, s8, s4                                       // 000000000630: 8B030408
	s_wait_alu depctr_sa_sdst(0)                               // 000000000634: BF88FF9E
	s_and_b32 s4, s14, s6                                      // 000000000638: 8B04060E
	s_and_b32 s5, s5, s3                                       // 00000000063C: 8B050305
	s_wait_alu depctr_sa_sdst(0)                               // 000000000640: BF88FF9E
	s_and_b32 s6, vcc_lo, s4                                   // 000000000644: 8B06046A
	s_wait_alu depctr_sa_sdst(0)                               // 000000000648: BF88FF9E
	s_or_b32 s5, s5, s6                                        // 00000000064C: 8C050605
	s_wait_alu depctr_sa_sdst(0)                               // 000000000650: BF88FF9E
	v_cndmask_b32_e64 v107, 0, 1, s5                           // 000000000654: D501006B 00150280
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000065C: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v107                           // 000000000660: 7C9AD680
	s_cbranch_vccz 7                                           // 000000000664: BFA30007 <nearest+0x684>
	s_mov_b32 s6, 1.0                                          // 000000000668: BE8600F2
	s_mov_b32 s14, 0                                           // 00000000066C: BE8E0080
	s_branch 27                                                // 000000000670: BFA0001B <nearest+0x6e0>
	s_mov_b32 s14, 0x41000000                                  // 000000000674: BE8E00FF 41000000
	s_mov_b32 s6, 0                                            // 00000000067C: BE860080
	s_branch 23                                                // 000000000680: BFA00017 <nearest+0x6e0>
	v_cndmask_b32_e64 v107, v74, -v0, s3                       // 000000000684: D501006B 400E014A
	v_cndmask_b32_e64 v108, v73, -v1, s4                       // 00000000068C: D501006C 40120349
	v_cndmask_b32_e64 v109, v72, v105, s3                      // 000000000694: D501006D 000ED348
	v_cndmask_b32_e64 v110, v71, v106, s4                      // 00000000069C: D501006E 0012D547
	v_cndmask_b32_e64 v68, v68, v74, s3                        // 0000000006A4: D5010044 000E9544
	v_cndmask_b32_e64 v69, v69, v73, s4                        // 0000000006AC: D5010045 00129345
	v_cndmask_b32_e64 v66, v66, v72, s3                        // 0000000006B4: D5010042 000E9142
	v_cndmask_b32_e64 v67, v67, v71, s4                        // 0000000006BC: D5010043 00128F43
	v_dual_mov_b32 v73, v108 :: v_dual_mov_b32 v74, v107       // 0000000006C4: CA10016C 494A016B
	v_dual_mov_b32 v71, v110 :: v_dual_mov_b32 v72, v109       // 0000000006CC: CA10016E 4748016D
	s_mov_b32 s6, 0                                            // 0000000006D4: BE860080
	s_mov_b32 s14, 0x41000000                                  // 0000000006D8: BE8E00FF 41000000
	s_wait_alu depctr_sa_sdst(0)                               // 0000000006E0: BF88FF9E
	s_cmp_lt_f32 s6, 0.5                                       // 0000000006E4: BF41F006
	v_cmp_ge_f32_e64 s3, v103, -v2                             // 0000000006E8: D4160003 42020567
	v_cmp_ge_f32_e64 s5, v103, -v3                             // 0000000006F0: D4160005 42020767
	v_or_b32_e32 v108, s7, v82                                 // 0000000006F8: 38D8A407
	s_cselect_b32 s4, -1, 0                                    // 0000000006FC: 980480C1
	v_or_b32_e32 v109, s7, v83                                 // 000000000700: 38DAA607
	s_wait_alu depctr_sa_sdst(0)                               // 000000000704: BF88FF9E
	s_and_b32 s8, s2, s4                                       // 000000000708: 8B080402
	s_and_b32 s4, s3, s8                                       // 00000000070C: 8B040803
	s_and_b32 s8, s5, s8                                       // 000000000710: 8B080805
	s_wait_alu depctr_sa_sdst(0)                               // 000000000714: BF88FF9E
	s_or_b32 s3, s4, s8                                        // 000000000718: 8C030804
	s_wait_alu depctr_sa_sdst(0)                               // 00000000071C: BF88FF9E
	v_cndmask_b32_e64 v107, 0, 1, s3                           // 000000000720: D501006B 000D0280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000000728: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v107                           // 00000000072C: 7C9AD680
	v_cvt_f32_u32_e32 v107, v108                               // 000000000730: 7ED60D6C
	v_cvt_f32_u32_e32 v108, v109                               // 000000000734: 7ED80D6D
	s_cbranch_vccz 50                                          // 000000000738: BFA30032 <nearest+0x804>
	v_cmp_gt_f32_e64 s15, v103, -v2                            // 00000000073C: D414000F 42020567
	s_delay_alu instid0(VALU_DEP_3)                            // 000000000744: BF870003
	v_cmp_gt_f32_e32 vcc_lo, v104, v107                        // 000000000748: 7C28D768
	v_cmp_gt_f32_e64 s20, v103, -v3                            // 00000000074C: D4140014 42020767
	v_cmp_gt_f32_e64 s3, v104, v108                            // 000000000754: D4140003 0202D968
	v_cmp_lt_f32_e64 s5, -1.0, v66                             // 00000000075C: D4110005 020284F3
	s_or_b32 s15, vcc_lo, s15                                  // 000000000764: 8C0F0F6A
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v67                         // 000000000768: 7C2286F3
	s_or_b32 s20, s3, s20                                      // 00000000076C: 8C141403
	s_wait_alu depctr_sa_sdst(0)                               // 000000000770: BF88FF9E
	s_and_b32 s3, s15, s4                                      // 000000000774: 8B03040F
	s_and_b32 s4, s20, s8                                      // 000000000778: 8B040814
	s_wait_alu depctr_sa_sdst(0)                               // 00000000077C: BF88FF9E
	s_and_b32 s5, s5, s3                                       // 000000000780: 8B050305
	s_and_b32 s8, s4, vcc_lo                                   // 000000000784: 8B086A04
	s_wait_alu depctr_sa_sdst(0)                               // 000000000788: BF88FF9E
	s_or_b32 s5, s5, s8                                        // 00000000078C: 8C050805
	s_wait_alu depctr_sa_sdst(0)                               // 000000000790: BF88FF9E
	v_cndmask_b32_e64 v109, 0, 1, s5                           // 000000000794: D501006D 00150280
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000079C: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v109                           // 0000000007A0: 7C9ADA80
	s_cbranch_vccz 3                                           // 0000000007A4: BFA30003 <nearest+0x7b4>
	s_mov_b32 s14, 1.0                                         // 0000000007A8: BE8E00F2
	s_mov_b32 s6, 1.0                                          // 0000000007AC: BE8600F2
	s_branch 20                                                // 0000000007B0: BFA00014 <nearest+0x804>
	v_cndmask_b32_e64 v109, v74, -v2, s3                       // 0000000007B4: D501006D 400E054A
	v_cndmask_b32_e64 v110, v73, -v3, s4                       // 0000000007BC: D501006E 40120749
	v_cndmask_b32_e64 v111, v72, v107, s3                      // 0000000007C4: D501006F 000ED748
	v_cndmask_b32_e64 v112, v71, v108, s4                      // 0000000007CC: D5010070 0012D947
	v_cndmask_b32_e64 v68, v68, v74, s3                        // 0000000007D4: D5010044 000E9544
	v_cndmask_b32_e64 v69, v69, v73, s4                        // 0000000007DC: D5010045 00129345
	v_cndmask_b32_e64 v66, v66, v72, s3                        // 0000000007E4: D5010042 000E9142
	v_cndmask_b32_e64 v67, v67, v71, s4                        // 0000000007EC: D5010043 00128F43
	v_dual_mov_b32 v73, v110 :: v_dual_mov_b32 v74, v109       // 0000000007F4: CA10016E 494A016D
	v_dual_mov_b32 v71, v112 :: v_dual_mov_b32 v72, v111       // 0000000007FC: CA100170 4748016F
	s_wait_alu depctr_sa_sdst(0)                               // 000000000804: BF88FF9E
	s_cmp_lt_f32 s6, 0.5                                       // 000000000808: BF41F006
	v_cmp_ge_f32_e64 s3, v103, -v4                             // 00000000080C: D4160003 42020967
	v_cmp_ge_f32_e64 s5, v103, -v5                             // 000000000814: D4160005 42020B67
	v_or_b32_e32 v110, s7, v85                                 // 00000000081C: 38DCAA07
	s_cselect_b32 s4, -1, 0                                    // 000000000820: 980480C1
	v_or_b32_e32 v111, s7, v86                                 // 000000000824: 38DEAC07
	s_wait_alu depctr_sa_sdst(0)                               // 000000000828: BF88FF9E
	s_and_b32 s8, s2, s4                                       // 00000000082C: 8B080402
	s_and_b32 s4, s3, s8                                       // 000000000830: 8B040803
	s_and_b32 s8, s5, s8                                       // 000000000834: 8B080805
	s_wait_alu depctr_sa_sdst(0)                               // 000000000838: BF88FF9E
	s_or_b32 s3, s4, s8                                        // 00000000083C: 8C030804
	s_wait_alu depctr_sa_sdst(0)                               // 000000000840: BF88FF9E
	v_cndmask_b32_e64 v109, 0, 1, s3                           // 000000000844: D501006D 000D0280
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000084C: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v109                           // 000000000850: 7C9ADA80
	v_cvt_f32_u32_e32 v109, v110                               // 000000000854: 7EDA0D6E
	v_cvt_f32_u32_e32 v110, v111                               // 000000000858: 7EDC0D6F
	s_cbranch_vccz 50                                          // 00000000085C: BFA30032 <nearest+0x928>
	v_cmp_gt_f32_e64 s15, v103, -v4                            // 000000000860: D414000F 42020967
	s_delay_alu instid0(VALU_DEP_3)                            // 000000000868: BF870003
	v_cmp_gt_f32_e32 vcc_lo, v104, v109                        // 00000000086C: 7C28DB68
	v_cmp_gt_f32_e64 s20, v103, -v5                            // 000000000870: D4140014 42020B67
	v_cmp_gt_f32_e64 s3, v104, v110                            // 000000000878: D4140003 0202DD68
	v_cmp_lt_f32_e64 s5, -1.0, v66                             // 000000000880: D4110005 020284F3
	s_or_b32 s15, vcc_lo, s15                                  // 000000000888: 8C0F0F6A
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v67                         // 00000000088C: 7C2286F3
	s_or_b32 s20, s3, s20                                      // 000000000890: 8C141403
	s_wait_alu depctr_sa_sdst(0)                               // 000000000894: BF88FF9E
	s_and_b32 s3, s15, s4                                      // 000000000898: 8B03040F
	s_and_b32 s4, s20, s8                                      // 00000000089C: 8B040814
	s_wait_alu depctr_sa_sdst(0)                               // 0000000008A0: BF88FF9E
	s_and_b32 s5, s5, s3                                       // 0000000008A4: 8B050305
	s_and_b32 s8, s4, vcc_lo                                   // 0000000008A8: 8B086A04
	s_wait_alu depctr_sa_sdst(0)                               // 0000000008AC: BF88FF9E
	s_or_b32 s5, s5, s8                                        // 0000000008B0: 8C050805
	s_wait_alu depctr_sa_sdst(0)                               // 0000000008B4: BF88FF9E
	v_cndmask_b32_e64 v111, 0, 1, s5                           // 0000000008B8: D501006F 00150280
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000008C0: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v111                           // 0000000008C4: 7C9ADE80
	s_cbranch_vccz 3                                           // 0000000008C8: BFA30003 <nearest+0x8d8>
	s_mov_b32 s6, 1.0                                          // 0000000008CC: BE8600F2
	s_mov_b32 s14, 2.0                                         // 0000000008D0: BE8E00F4
	s_branch 20                                                // 0000000008D4: BFA00014 <nearest+0x928>
	v_cndmask_b32_e64 v111, v74, -v4, s3                       // 0000000008D8: D501006F 400E094A
	v_cndmask_b32_e64 v112, v73, -v5, s4                       // 0000000008E0: D5010070 40120B49
	v_cndmask_b32_e64 v113, v72, v109, s3                      // 0000000008E8: D5010071 000EDB48
	v_cndmask_b32_e64 v114, v71, v110, s4                      // 0000000008F0: D5010072 0012DD47
	v_cndmask_b32_e64 v68, v68, v74, s3                        // 0000000008F8: D5010044 000E9544
	v_cndmask_b32_e64 v69, v69, v73, s4                        // 000000000900: D5010045 00129345
	v_cndmask_b32_e64 v66, v66, v72, s3                        // 000000000908: D5010042 000E9142
	v_cndmask_b32_e64 v67, v67, v71, s4                        // 000000000910: D5010043 00128F43
	v_dual_mov_b32 v73, v112 :: v_dual_mov_b32 v74, v111       // 000000000918: CA100170 494A016F
	v_dual_mov_b32 v71, v114 :: v_dual_mov_b32 v72, v113       // 000000000920: CA100172 47480171
	s_wait_alu depctr_sa_sdst(0)                               // 000000000928: BF88FF9E
	s_cmp_lt_f32 s6, 0.5                                       // 00000000092C: BF41F006
	v_cmp_ge_f32_e64 s3, v103, -v6                             // 000000000930: D4160003 42020D67
	v_cmp_ge_f32_e64 s5, v103, -v7                             // 000000000938: D4160005 42020F67
	v_or_b32_e32 v112, s7, v87                                 // 000000000940: 38E0AE07
	s_cselect_b32 s4, -1, 0                                    // 000000000944: 980480C1
	v_or_b32_e32 v113, s7, v89                                 // 000000000948: 38E2B207
	s_wait_alu depctr_sa_sdst(0)                               // 00000000094C: BF88FF9E
	s_and_b32 s8, s2, s4                                       // 000000000950: 8B080402
	s_and_b32 s4, s3, s8                                       // 000000000954: 8B040803
	s_and_b32 s8, s5, s8                                       // 000000000958: 8B080805
	s_wait_alu depctr_sa_sdst(0)                               // 00000000095C: BF88FF9E
	s_or_b32 s3, s4, s8                                        // 000000000960: 8C030804
	s_wait_alu depctr_sa_sdst(0)                               // 000000000964: BF88FF9E
	v_cndmask_b32_e64 v111, 0, 1, s3                           // 000000000968: D501006F 000D0280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000000970: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v111                           // 000000000974: 7C9ADE80
	v_cvt_f32_u32_e32 v111, v112                               // 000000000978: 7EDE0D70
	v_cvt_f32_u32_e32 v112, v113                               // 00000000097C: 7EE00D71
	s_cbranch_vccz 51                                          // 000000000980: BFA30033 <nearest+0xa50>
	v_cmp_gt_f32_e64 s15, v103, -v6                            // 000000000984: D414000F 42020D67
	s_delay_alu instid0(VALU_DEP_3)                            // 00000000098C: BF870003
	v_cmp_gt_f32_e32 vcc_lo, v104, v111                        // 000000000990: 7C28DF68
	v_cmp_gt_f32_e64 s20, v103, -v7                            // 000000000994: D4140014 42020F67
	v_cmp_gt_f32_e64 s3, v104, v112                            // 00000000099C: D4140003 0202E168
	v_cmp_lt_f32_e64 s5, -1.0, v66                             // 0000000009A4: D4110005 020284F3
	s_or_b32 s15, vcc_lo, s15                                  // 0000000009AC: 8C0F0F6A
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v67                         // 0000000009B0: 7C2286F3
	s_or_b32 s20, s3, s20                                      // 0000000009B4: 8C141403
	s_wait_alu depctr_sa_sdst(0)                               // 0000000009B8: BF88FF9E
	s_and_b32 s3, s15, s4                                      // 0000000009BC: 8B03040F
	s_and_b32 s4, s20, s8                                      // 0000000009C0: 8B040814
	s_wait_alu depctr_sa_sdst(0)                               // 0000000009C4: BF88FF9E
	s_and_b32 s5, s5, s3                                       // 0000000009C8: 8B050305
	s_and_b32 s8, s4, vcc_lo                                   // 0000000009CC: 8B086A04
	s_wait_alu depctr_sa_sdst(0)                               // 0000000009D0: BF88FF9E
	s_or_b32 s5, s5, s8                                        // 0000000009D4: 8C050805
	s_wait_alu depctr_sa_sdst(0)                               // 0000000009D8: BF88FF9E
	v_cndmask_b32_e64 v113, 0, 1, s5                           // 0000000009DC: D5010071 00150280
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000009E4: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v113                           // 0000000009E8: 7C9AE280
	s_cbranch_vccz 4                                           // 0000000009EC: BFA30004 <nearest+0xa00>
	s_mov_b32 s6, 1.0                                          // 0000000009F0: BE8600F2
	s_mov_b32 s14, 0x40400000                                  // 0000000009F4: BE8E00FF 40400000
	s_branch 20                                                // 0000000009FC: BFA00014 <nearest+0xa50>
	v_cndmask_b32_e64 v113, v74, -v6, s3                       // 000000000A00: D5010071 400E0D4A
	v_cndmask_b32_e64 v114, v73, -v7, s4                       // 000000000A08: D5010072 40120F49
	v_cndmask_b32_e64 v115, v72, v111, s3                      // 000000000A10: D5010073 000EDF48
	v_cndmask_b32_e64 v116, v71, v112, s4                      // 000000000A18: D5010074 0012E147
	v_cndmask_b32_e64 v68, v68, v74, s3                        // 000000000A20: D5010044 000E9544
	v_cndmask_b32_e64 v69, v69, v73, s4                        // 000000000A28: D5010045 00129345
	v_cndmask_b32_e64 v66, v66, v72, s3                        // 000000000A30: D5010042 000E9142
	v_cndmask_b32_e64 v67, v67, v71, s4                        // 000000000A38: D5010043 00128F43
	v_dual_mov_b32 v73, v114 :: v_dual_mov_b32 v74, v113       // 000000000A40: CA100172 494A0171
	v_dual_mov_b32 v71, v116 :: v_dual_mov_b32 v72, v115       // 000000000A48: CA100174 47480173
	s_wait_alu depctr_sa_sdst(0)                               // 000000000A50: BF88FF9E
	s_cmp_lt_f32 s6, 0.5                                       // 000000000A54: BF41F006
	v_cmp_ge_f32_e64 s3, v103, -v8                             // 000000000A58: D4160003 42021167
	v_cmp_ge_f32_e64 s5, v103, -v9                             // 000000000A60: D4160005 42021367
	v_or_b32_e32 v114, s7, v79                                 // 000000000A68: 38E49E07
	s_cselect_b32 s4, -1, 0                                    // 000000000A6C: 980480C1
	v_or_b32_e32 v115, s7, v90                                 // 000000000A70: 38E6B407
	s_wait_alu depctr_sa_sdst(0)                               // 000000000A74: BF88FF9E
	s_and_b32 s8, s2, s4                                       // 000000000A78: 8B080402
	s_and_b32 s4, s3, s8                                       // 000000000A7C: 8B040803
	s_and_b32 s8, s5, s8                                       // 000000000A80: 8B080805
	s_wait_alu depctr_sa_sdst(0)                               // 000000000A84: BF88FF9E
	s_or_b32 s3, s4, s8                                        // 000000000A88: 8C030804
	s_wait_alu depctr_sa_sdst(0)                               // 000000000A8C: BF88FF9E
	v_cndmask_b32_e64 v113, 0, 1, s3                           // 000000000A90: D5010071 000D0280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000000A98: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v113                           // 000000000A9C: 7C9AE280
	v_cvt_f32_u32_e32 v113, v114                               // 000000000AA0: 7EE20D72
	v_cvt_f32_u32_e32 v114, v115                               // 000000000AA4: 7EE40D73
	s_cbranch_vccz 50                                          // 000000000AA8: BFA30032 <nearest+0xb74>
	v_cmp_gt_f32_e64 s15, v103, -v8                            // 000000000AAC: D414000F 42021167
	s_delay_alu instid0(VALU_DEP_3)                            // 000000000AB4: BF870003
	v_cmp_gt_f32_e32 vcc_lo, v104, v113                        // 000000000AB8: 7C28E368
	v_cmp_gt_f32_e64 s20, v103, -v9                            // 000000000ABC: D4140014 42021367
	v_cmp_gt_f32_e64 s3, v104, v114                            // 000000000AC4: D4140003 0202E568
	v_cmp_lt_f32_e64 s5, -1.0, v66                             // 000000000ACC: D4110005 020284F3
	s_or_b32 s15, vcc_lo, s15                                  // 000000000AD4: 8C0F0F6A
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v67                         // 000000000AD8: 7C2286F3
	s_or_b32 s20, s3, s20                                      // 000000000ADC: 8C141403
	s_wait_alu depctr_sa_sdst(0)                               // 000000000AE0: BF88FF9E
	s_and_b32 s3, s15, s4                                      // 000000000AE4: 8B03040F
	s_and_b32 s4, s20, s8                                      // 000000000AE8: 8B040814
	s_wait_alu depctr_sa_sdst(0)                               // 000000000AEC: BF88FF9E
	s_and_b32 s5, s5, s3                                       // 000000000AF0: 8B050305
	s_and_b32 s8, s4, vcc_lo                                   // 000000000AF4: 8B086A04
	s_wait_alu depctr_sa_sdst(0)                               // 000000000AF8: BF88FF9E
	s_or_b32 s5, s5, s8                                        // 000000000AFC: 8C050805
	s_wait_alu depctr_sa_sdst(0)                               // 000000000B00: BF88FF9E
	v_cndmask_b32_e64 v115, 0, 1, s5                           // 000000000B04: D5010073 00150280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000000B0C: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v115                           // 000000000B10: 7C9AE680
	s_cbranch_vccz 3                                           // 000000000B14: BFA30003 <nearest+0xb24>
	s_mov_b32 s6, 1.0                                          // 000000000B18: BE8600F2
	s_mov_b32 s14, 4.0                                         // 000000000B1C: BE8E00F6
	s_branch 20                                                // 000000000B20: BFA00014 <nearest+0xb74>
	v_cndmask_b32_e64 v115, v74, -v8, s3                       // 000000000B24: D5010073 400E114A
	v_cndmask_b32_e64 v116, v73, -v9, s4                       // 000000000B2C: D5010074 40121349
	v_cndmask_b32_e64 v117, v72, v113, s3                      // 000000000B34: D5010075 000EE348
	v_cndmask_b32_e64 v118, v71, v114, s4                      // 000000000B3C: D5010076 0012E547
	v_cndmask_b32_e64 v68, v68, v74, s3                        // 000000000B44: D5010044 000E9544
	v_cndmask_b32_e64 v69, v69, v73, s4                        // 000000000B4C: D5010045 00129345
	v_cndmask_b32_e64 v66, v66, v72, s3                        // 000000000B54: D5010042 000E9142
	v_cndmask_b32_e64 v67, v67, v71, s4                        // 000000000B5C: D5010043 00128F43
	v_dual_mov_b32 v73, v116 :: v_dual_mov_b32 v74, v115       // 000000000B64: CA100174 494A0173
	v_dual_mov_b32 v71, v118 :: v_dual_mov_b32 v72, v117       // 000000000B6C: CA100176 47480175
	s_wait_alu depctr_sa_sdst(0)                               // 000000000B74: BF88FF9E
	s_cmp_lt_f32 s6, 0.5                                       // 000000000B78: BF41F006
	v_cmp_ge_f32_e64 s3, v103, -v10                            // 000000000B7C: D4160003 42021567
	v_cmp_ge_f32_e64 s5, v103, -v11                            // 000000000B84: D4160005 42021767
	v_or_b32_e32 v116, s7, v91                                 // 000000000B8C: 38E8B607
	s_cselect_b32 s4, -1, 0                                    // 000000000B90: 980480C1
	v_or_b32_e32 v117, s7, v92                                 // 000000000B94: 38EAB807
	s_wait_alu depctr_sa_sdst(0)                               // 000000000B98: BF88FF9E
	s_and_b32 s8, s2, s4                                       // 000000000B9C: 8B080402
	s_and_b32 s4, s3, s8                                       // 000000000BA0: 8B040803
	s_and_b32 s8, s5, s8                                       // 000000000BA4: 8B080805
	s_wait_alu depctr_sa_sdst(0)                               // 000000000BA8: BF88FF9E
	s_or_b32 s3, s4, s8                                        // 000000000BAC: 8C030804
	s_wait_alu depctr_sa_sdst(0)                               // 000000000BB0: BF88FF9E
	v_cndmask_b32_e64 v115, 0, 1, s3                           // 000000000BB4: D5010073 000D0280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000000BBC: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v115                           // 000000000BC0: 7C9AE680
	v_cvt_f32_u32_e32 v115, v116                               // 000000000BC4: 7EE60D74
	v_cvt_f32_u32_e32 v116, v117                               // 000000000BC8: 7EE80D75
	s_cbranch_vccz 51                                          // 000000000BCC: BFA30033 <nearest+0xc9c>
	v_cmp_gt_f32_e64 s15, v103, -v10                           // 000000000BD0: D414000F 42021567
	s_delay_alu instid0(VALU_DEP_3)                            // 000000000BD8: BF870003
	v_cmp_gt_f32_e32 vcc_lo, v104, v115                        // 000000000BDC: 7C28E768
	v_cmp_gt_f32_e64 s20, v103, -v11                           // 000000000BE0: D4140014 42021767
	v_cmp_gt_f32_e64 s3, v104, v116                            // 000000000BE8: D4140003 0202E968
	v_cmp_lt_f32_e64 s5, -1.0, v66                             // 000000000BF0: D4110005 020284F3
	s_or_b32 s15, vcc_lo, s15                                  // 000000000BF8: 8C0F0F6A
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v67                         // 000000000BFC: 7C2286F3
	s_or_b32 s20, s3, s20                                      // 000000000C00: 8C141403
	s_wait_alu depctr_sa_sdst(0)                               // 000000000C04: BF88FF9E
	s_and_b32 s3, s15, s4                                      // 000000000C08: 8B03040F
	s_and_b32 s4, s20, s8                                      // 000000000C0C: 8B040814
	s_wait_alu depctr_sa_sdst(0)                               // 000000000C10: BF88FF9E
	s_and_b32 s5, s5, s3                                       // 000000000C14: 8B050305
	s_and_b32 s8, s4, vcc_lo                                   // 000000000C18: 8B086A04
	s_wait_alu depctr_sa_sdst(0)                               // 000000000C1C: BF88FF9E
	s_or_b32 s5, s5, s8                                        // 000000000C20: 8C050805
	s_wait_alu depctr_sa_sdst(0)                               // 000000000C24: BF88FF9E
	v_cndmask_b32_e64 v117, 0, 1, s5                           // 000000000C28: D5010075 00150280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000000C30: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v117                           // 000000000C34: 7C9AEA80
	s_cbranch_vccz 4                                           // 000000000C38: BFA30004 <nearest+0xc4c>
	s_mov_b32 s6, 1.0                                          // 000000000C3C: BE8600F2
	s_mov_b32 s14, 0x40a00000                                  // 000000000C40: BE8E00FF 40A00000
	s_branch 20                                                // 000000000C48: BFA00014 <nearest+0xc9c>
	v_cndmask_b32_e64 v117, v74, -v10, s3                      // 000000000C4C: D5010075 400E154A
	v_cndmask_b32_e64 v118, v73, -v11, s4                      // 000000000C54: D5010076 40121749
	v_cndmask_b32_e64 v119, v72, v115, s3                      // 000000000C5C: D5010077 000EE748
	v_cndmask_b32_e64 v120, v71, v116, s4                      // 000000000C64: D5010078 0012E947
	v_cndmask_b32_e64 v68, v68, v74, s3                        // 000000000C6C: D5010044 000E9544
	v_cndmask_b32_e64 v69, v69, v73, s4                        // 000000000C74: D5010045 00129345
	v_cndmask_b32_e64 v66, v66, v72, s3                        // 000000000C7C: D5010042 000E9142
	v_cndmask_b32_e64 v67, v67, v71, s4                        // 000000000C84: D5010043 00128F43
	v_dual_mov_b32 v73, v118 :: v_dual_mov_b32 v74, v117       // 000000000C8C: CA100176 494A0175
	v_dual_mov_b32 v71, v120 :: v_dual_mov_b32 v72, v119       // 000000000C94: CA100178 47480177
	s_wait_alu depctr_sa_sdst(0)                               // 000000000C9C: BF88FF9E
	s_cmp_lt_f32 s6, 0.5                                       // 000000000CA0: BF41F006
	v_cmp_ge_f32_e64 s3, v103, -v12                            // 000000000CA4: D4160003 42021967
	v_cmp_ge_f32_e64 s5, v103, -v13                            // 000000000CAC: D4160005 42021B67
	v_or_b32_e32 v118, s7, v97                                 // 000000000CB4: 38ECC207
	s_cselect_b32 s4, -1, 0                                    // 000000000CB8: 980480C1
	v_or_b32_e32 v119, s7, v100                                // 000000000CBC: 38EEC807
	s_wait_alu depctr_sa_sdst(0)                               // 000000000CC0: BF88FF9E
	s_and_b32 s8, s2, s4                                       // 000000000CC4: 8B080402
	s_and_b32 s4, s3, s8                                       // 000000000CC8: 8B040803
	s_and_b32 s8, s5, s8                                       // 000000000CCC: 8B080805
	s_wait_alu depctr_sa_sdst(0)                               // 000000000CD0: BF88FF9E
	s_or_b32 s3, s4, s8                                        // 000000000CD4: 8C030804
	s_wait_alu depctr_sa_sdst(0)                               // 000000000CD8: BF88FF9E
	v_cndmask_b32_e64 v117, 0, 1, s3                           // 000000000CDC: D5010075 000D0280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000000CE4: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v117                           // 000000000CE8: 7C9AEA80
	v_cvt_f32_u32_e32 v117, v118                               // 000000000CEC: 7EEA0D76
	v_cvt_f32_u32_e32 v118, v119                               // 000000000CF0: 7EEC0D77
	s_cbranch_vccz 51                                          // 000000000CF4: BFA30033 <nearest+0xdc4>
	v_cmp_gt_f32_e64 s15, v103, -v12                           // 000000000CF8: D414000F 42021967
	s_delay_alu instid0(VALU_DEP_3)                            // 000000000D00: BF870003
	v_cmp_gt_f32_e32 vcc_lo, v104, v117                        // 000000000D04: 7C28EB68
	v_cmp_gt_f32_e64 s20, v103, -v13                           // 000000000D08: D4140014 42021B67
	v_cmp_gt_f32_e64 s3, v104, v118                            // 000000000D10: D4140003 0202ED68
	v_cmp_lt_f32_e64 s5, -1.0, v66                             // 000000000D18: D4110005 020284F3
	s_or_b32 s15, vcc_lo, s15                                  // 000000000D20: 8C0F0F6A
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v67                         // 000000000D24: 7C2286F3
	s_or_b32 s20, s3, s20                                      // 000000000D28: 8C141403
	s_wait_alu depctr_sa_sdst(0)                               // 000000000D2C: BF88FF9E
	s_and_b32 s3, s15, s4                                      // 000000000D30: 8B03040F
	s_and_b32 s4, s20, s8                                      // 000000000D34: 8B040814
	s_wait_alu depctr_sa_sdst(0)                               // 000000000D38: BF88FF9E
	s_and_b32 s5, s5, s3                                       // 000000000D3C: 8B050305
	s_and_b32 s8, s4, vcc_lo                                   // 000000000D40: 8B086A04
	s_wait_alu depctr_sa_sdst(0)                               // 000000000D44: BF88FF9E
	s_or_b32 s5, s5, s8                                        // 000000000D48: 8C050805
	s_wait_alu depctr_sa_sdst(0)                               // 000000000D4C: BF88FF9E
	v_cndmask_b32_e64 v119, 0, 1, s5                           // 000000000D50: D5010077 00150280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000000D58: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v119                           // 000000000D5C: 7C9AEE80
	s_cbranch_vccz 4                                           // 000000000D60: BFA30004 <nearest+0xd74>
	s_mov_b32 s6, 1.0                                          // 000000000D64: BE8600F2
	s_mov_b32 s14, 0x40c00000                                  // 000000000D68: BE8E00FF 40C00000
	s_branch 20                                                // 000000000D70: BFA00014 <nearest+0xdc4>
	v_cndmask_b32_e64 v119, v74, -v12, s3                      // 000000000D74: D5010077 400E194A
	v_cndmask_b32_e64 v120, v73, -v13, s4                      // 000000000D7C: D5010078 40121B49
	v_cndmask_b32_e64 v121, v72, v117, s3                      // 000000000D84: D5010079 000EEB48
	v_cndmask_b32_e64 v122, v71, v118, s4                      // 000000000D8C: D501007A 0012ED47
	v_cndmask_b32_e64 v68, v68, v74, s3                        // 000000000D94: D5010044 000E9544
	v_cndmask_b32_e64 v69, v69, v73, s4                        // 000000000D9C: D5010045 00129345
	v_cndmask_b32_e64 v66, v66, v72, s3                        // 000000000DA4: D5010042 000E9142
	v_cndmask_b32_e64 v67, v67, v71, s4                        // 000000000DAC: D5010043 00128F43
	v_dual_mov_b32 v73, v120 :: v_dual_mov_b32 v74, v119       // 000000000DB4: CA100178 494A0177
	v_dual_mov_b32 v71, v122 :: v_dual_mov_b32 v72, v121       // 000000000DBC: CA10017A 47480179
	s_wait_alu depctr_sa_sdst(0)                               // 000000000DC4: BF88FF9E
	s_cmp_lt_f32 s6, 0.5                                       // 000000000DC8: BF41F006
	v_cmp_ge_f32_e64 s3, v103, -v14                            // 000000000DCC: D4160003 42021D67
	v_cmp_ge_f32_e64 s4, v103, -v15                            // 000000000DD4: D4160004 42021F67
	v_or_b32_e32 v120, s7, v101                                // 000000000DDC: 38F0CA07
	s_cselect_b32 s5, -1, 0                                    // 000000000DE0: 980580C1
	v_or_b32_e32 v121, s7, v102                                // 000000000DE4: 38F2CC07
	s_wait_alu depctr_sa_sdst(0)                               // 000000000DE8: BF88FF9E
	s_and_b32 s2, s2, s5                                       // 000000000DEC: 8B020502
	s_wait_alu depctr_sa_sdst(0)                               // 000000000DF0: BF88FF9E
	s_and_b32 s3, s3, s2                                       // 000000000DF4: 8B030203
	s_and_b32 s5, s4, s2                                       // 000000000DF8: 8B050204
	s_wait_alu depctr_sa_sdst(0)                               // 000000000DFC: BF88FF9E
	s_or_b32 s2, s3, s5                                        // 000000000E00: 8C020503
	s_wait_alu depctr_sa_sdst(0)                               // 000000000E04: BF88FF9E
	v_cndmask_b32_e64 v119, 0, 1, s2                           // 000000000E08: D5010077 00090280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000000E10: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v119                           // 000000000E14: 7C9AEE80
	v_cvt_f32_u32_e32 v119, v120                               // 000000000E18: 7EEE0D78
	v_cvt_f32_u32_e32 v120, v121                               // 000000000E1C: 7EF00D79
	s_cbranch_vccz 30                                          // 000000000E20: BFA3001E <nearest+0xe9c>
	v_cmp_gt_f32_e64 s7, v103, -v14                            // 000000000E24: D4140007 42021D67
	s_delay_alu instid0(VALU_DEP_3)                            // 000000000E2C: BF870003
	v_cmp_gt_f32_e32 vcc_lo, v104, v119                        // 000000000E30: 7C28EF68
	v_cmp_gt_f32_e64 s8, v103, -v15                            // 000000000E34: D4140008 42021F67
	v_cmp_gt_f32_e64 s2, v104, v120                            // 000000000E3C: D4140002 0202F168
	v_cmp_lt_f32_e64 s4, -1.0, v66                             // 000000000E44: D4110004 020284F3
	s_or_b32 s7, vcc_lo, s7                                    // 000000000E4C: 8C07076A
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v67                         // 000000000E50: 7C2286F3
	s_or_b32 s8, s2, s8                                        // 000000000E54: 8C080802
	s_wait_alu depctr_sa_sdst(0)                               // 000000000E58: BF88FF9E
	s_and_b32 s2, s7, s3                                       // 000000000E5C: 8B020307
	s_and_b32 s3, s8, s5                                       // 000000000E60: 8B030508
	s_wait_alu depctr_sa_sdst(0)                               // 000000000E64: BF88FF9E
	s_and_b32 s4, s4, s2                                       // 000000000E68: 8B040204
	s_and_b32 s5, s3, vcc_lo                                   // 000000000E6C: 8B056A03
	s_wait_alu depctr_sa_sdst(0)                               // 000000000E70: BF88FF9E
	s_or_b32 s4, s4, s5                                        // 000000000E74: 8C040504
	s_wait_alu depctr_sa_sdst(0)                               // 000000000E78: BF88FF9E
	v_cndmask_b32_e64 v121, 0, 1, s4                           // 000000000E7C: D5010079 00110280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000000E84: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v121                           // 000000000E88: 7C9AF280
	s_cbranch_vccz 2255                                        // 000000000E8C: BFA308CF <nearest+0x31cc>
	s_mov_b32 s6, 1.0                                          // 000000000E90: BE8600F2
	s_mov_b32 s14, 0x40e00000                                  // 000000000E94: BE8E00FF 40E00000
	s_wait_alu depctr_sa_sdst(0)                               // 000000000E9C: BF88FF9E
	s_cmp_ngt_f32 s6, 0.5                                      // 000000000EA0: BF4BF006
	s_cbranch_scc1 64795                                       // 000000000EA4: BFA2FD1B <nearest+0x314>
	v_cmp_eq_f32_e32 vcc_lo, v73, v74                          // 000000000EA8: 7C249549
	v_cmp_lt_f32_e64 s2, v71, v72                              // 000000000EAC: D4110002 02029147
	v_cmp_lt_f32_e64 s4, v73, v74                              // 000000000EB4: D4110004 02029549
	v_cmp_lt_f32_e64 s3, v72, v71                              // 000000000EBC: D4110003 02028F48
	v_cmp_lt_f32_e64 s5, v74, v73                              // 000000000EC4: D4110005 0202934A
	v_cmp_lt_f32_e64 s6, v68, v69                              // 000000000ECC: D4110006 02028B44
	s_and_b32 s2, s2, vcc_lo                                   // 000000000ED4: 8B026A02
	s_wait_alu depctr_sa_sdst(0)                               // 000000000ED8: BF88FF9E
	s_or_b32 s2, s4, s2                                        // 000000000EDC: 8C020204
	s_and_b32 s3, s3, vcc_lo                                   // 000000000EE0: 8B036A03
	s_wait_alu depctr_sa_sdst(0)                               // 000000000EE4: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000000EE8: 8D6A0002
	s_or_b32 s3, s5, s3                                        // 000000000EEC: 8C030305
	s_wait_alu depctr_sa_sdst(0)                               // 000000000EF0: BF88FF9E
	v_cndmask_b32_e32 v103, v74, v73, vcc_lo                   // 000000000EF4: 02CE934A
	s_xor_b32 s2, s3, s9                                       // 000000000EF8: 8D020903
	v_cmp_lt_f32_e64 s3, v67, v66                              // 000000000EFC: D4110003 02028543
	s_wait_alu depctr_sa_sdst(0)                               // 000000000F04: BF88FF9E
	v_cndmask_b32_e64 v73, v73, v74, s2                        // 000000000F08: D5010049 000A9549
	v_cndmask_b32_e32 v74, v72, v71, vcc_lo                    // 000000000F10: 02948F48
	v_cndmask_b32_e64 v71, v71, v72, s2                        // 000000000F14: D5010047 000A9147
	v_mov_b32_e32 v104, v103                                   // 000000000F1C: 7ED00367
	v_cmp_eq_f32_e32 vcc_lo, v69, v68                          // 000000000F20: 7C248945
	v_cmp_lt_f32_e64 s4, v66, v67                              // 000000000F24: D4110004 02028742
	v_cmp_lt_f32_e64 s5, v69, v68                              // 000000000F2C: D4110005 02028945
	v_mov_b32_e32 v123, v71                                    // 000000000F34: 7EF60347
	v_permlanex16_b32 v104, v104, s10, 0xfedcba98              // 000000000F38: D65C0068 03FC1568 FEDCBA98
	v_mov_b32_e32 v122, v74                                    // 000000000F44: 7EF4034A
	s_and_b32 s2, s3, vcc_lo                                   // 000000000F48: 8B026A03
	s_and_b32 s3, s4, vcc_lo                                   // 000000000F4C: 8B036A04
	s_wait_alu depctr_sa_sdst(0)                               // 000000000F50: BF88FF9E
	s_or_b32 s2, s5, s2                                        // 000000000F54: 8C020205
	s_or_b32 s3, s6, s3                                        // 000000000F58: 8C030306
	v_permlanex16_b32 v122, v122, s10, 0xfedcba98              // 000000000F5C: D65C007A 03FC157A FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 000000000F68: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000000F6C: 8D6A0002
	s_xor_b32 s2, s3, s9                                       // 000000000F70: 8D020903
	s_wait_alu depctr_sa_sdst(0)                               // 000000000F74: BF88FF9E
	v_dual_cndmask_b32 v72, v68, v69 :: v_dual_mov_b32 v121, v73// 000000000F78: CA508B44 48780149
	v_cndmask_b32_e64 v68, v69, v68, s2                        // 000000000F80: D5010044 000A8945
	v_cndmask_b32_e32 v69, v66, v67, vcc_lo                    // 000000000F88: 028A8742
	v_cmp_eq_f32_e32 vcc_lo, v103, v104                        // 000000000F8C: 7C24D167
	v_cmp_gt_f32_e64 s4, v74, v122                             // 000000000F90: D4140004 0202F54A
	v_cmp_gt_f32_e64 s6, v103, v104                            // 000000000F98: D4140006 0202D167
	v_cndmask_b32_e64 v66, v67, v66, s2                        // 000000000FA0: D5010042 000A8543
	v_permlanex16_b32 v121, v121, s10, 0xfedcba98              // 000000000FA8: D65C0079 03FC1579 FEDCBA98
	v_permlanex16_b32 v123, v123, s10, 0xfedcba98              // 000000000FB4: D65C007B 03FC157B FEDCBA98
	s_and_b32 s2, vcc_lo, s4                                   // 000000000FC0: 8B02046A
	s_mov_b32 s15, 4                                           // 000000000FC4: BE8F0084
	s_wait_alu depctr_sa_sdst(0)                               // 000000000FC8: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 000000000FCC: 8C020206
	v_cmp_eq_f32_e64 s3, v73, v121                             // 000000000FD0: D4120003 0202F349
	s_wait_alu depctr_sa_sdst(0)                               // 000000000FD8: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000000FDC: 8D6A0002
	v_cmp_gt_f32_e64 s5, v71, v123                             // 000000000FE0: D4140005 0202F747
	s_wait_alu depctr_sa_sdst(0)                               // 000000000FE8: BF88FF9E
	v_dual_cndmask_b32 v67, v103, v104 :: v_dual_mov_b32 v104, v68// 000000000FEC: CA50D167 43680144
	v_dual_mov_b32 v103, v72 :: v_dual_mov_b32 v124, v69       // 000000000FF4: CA100148 677C0145
	v_cmp_gt_f32_e64 s7, v73, v121                             // 000000000FFC: D4140007 0202F349
	s_and_b32 s3, s3, s5                                       // 000000001004: 8B030503
	v_permlanex16_b32 v104, v104, s10, 0xfedcba98              // 000000001008: D65C0068 03FC1568 FEDCBA98
	v_permlanex16_b32 v103, v103, s10, 0xfedcba98              // 000000001014: D65C0067 03FC1567 FEDCBA98
	v_permlanex16_b32 v124, v124, s10, 0xfedcba98              // 000000001020: D65C007C 03FC157C FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 00000000102C: BF88FF9E
	s_or_b32 s3, s7, s3                                        // 000000001030: 8C030307
	v_cndmask_b32_e32 v74, v74, v122, vcc_lo                   // 000000001034: 0294F54A
	s_wait_alu depctr_sa_sdst(0)                               // 000000001038: BF88FF9E
	s_xor_b32 s2, s3, s0                                       // 00000000103C: 8D020003
	v_cmp_neq_f32_e32 vcc_lo, v72, v103                        // 000000001040: 7C3ACF48
	v_cmp_ngt_f32_e64 s3, v69, v124                            // 000000001044: D41B0003 0202F945
	v_cmp_ngt_f32_e64 s6, v72, v103                            // 00000000104C: D41B0006 0202CF48
	s_wait_alu depctr_sa_sdst(0)                               // 000000001054: BF88FF9E
	v_cndmask_b32_e64 v73, v73, v121, s2                       // 000000001058: D5010049 000AF349
	v_cndmask_b32_e64 v71, v71, v123, s2                       // 000000001060: D5010047 000AF747
	v_cmp_neq_f32_e64 s4, v68, v104                            // 000000001068: D41D0004 0202D144
	s_or_b32 s2, vcc_lo, s3                                    // 000000001070: 8C02036A
	v_cmp_ngt_f32_e64 s7, v68, v104                            // 000000001074: D41B0007 0202D144
	s_wait_alu depctr_sa_sdst(0)                               // 00000000107C: BF88FF9E
	s_and_b32 s2, s6, s2                                       // 000000001080: 8B020206
	s_wait_alu depctr_sa_sdst(0)                               // 000000001084: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000001088: 8D6A0002
	s_wait_alu depctr_sa_sdst(0)                               // 00000000108C: BF88FF9E
	v_dual_mov_b32 v125, v66 :: v_dual_cndmask_b32 v72, v72, v103// 000000001090: CA120142 7D48CF48
	v_cndmask_b32_e32 v69, v69, v124, vcc_lo                   // 000000001098: 028AF945
	v_cmp_eq_f32_e32 vcc_lo, v73, v67                          // 00000000109C: 7C248749
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000010A0: BF870093
	v_permlanex16_b32 v125, v125, s10, 0xfedcba98              // 0000000010A4: D65C007D 03FC157D FEDCBA98
	v_cmp_ngt_f32_e64 s5, v66, v125                            // 0000000010B0: D41B0005 0202FB42
	s_or_b32 s3, s4, s5                                        // 0000000010B8: 8C030504
	v_cmp_lt_f32_e64 s4, v74, v71                              // 0000000010BC: D4110004 02028F4A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000010C4: BF88FF9E
	s_and_b32 s3, s7, s3                                       // 0000000010C8: 8B030307
	v_cmp_lt_f32_e64 s5, v73, v67                              // 0000000010CC: D4110005 02028749
	s_wait_alu depctr_sa_sdst(0)                               // 0000000010D4: BF88FF9E
	s_xor_b32 s2, s3, s0                                       // 0000000010D8: 8D020003
	v_cmp_lt_f32_e64 s3, v71, v74                              // 0000000010DC: D4110003 02029547
	s_wait_alu depctr_sa_sdst(0)                               // 0000000010E4: BF88FF9E
	v_cndmask_b32_e64 v68, v68, v104, s2                       // 0000000010E8: D5010044 000AD144
	v_cndmask_b32_e64 v66, v66, v125, s2                       // 0000000010F0: D5010042 000AFB42
	v_cmp_lt_f32_e64 s2, v67, v73                              // 0000000010F8: D4110002 02029343
	s_and_b32 s4, vcc_lo, s4                                   // 000000001100: 8B04046A
	s_and_b32 s3, vcc_lo, s3                                   // 000000001104: 8B03036A
	s_wait_alu depctr_sa_sdst(0)                               // 000000001108: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 00000000110C: 8C6A0305
	s_or_b32 s2, s2, s4                                        // 000000001110: 8C020402
	s_wait_alu depctr_sa_sdst(0)                               // 000000001114: BF88FF9E
	v_cndmask_b32_e32 v103, v67, v73, vcc_lo                   // 000000001118: 02CE9343
	v_cndmask_b32_e64 v67, v67, v73, s2                        // 00000000111C: D5010043 000A9343
	v_cndmask_b32_e32 v73, v74, v71, vcc_lo                    // 000000001124: 02928F4A
	v_cmp_eq_f32_e32 vcc_lo, v68, v72                          // 000000001128: 7C249144
	v_cmp_lt_f32_e64 s3, v66, v69                              // 00000000112C: D4110003 02028B42
	v_cmp_lt_f32_e64 s4, v69, v66                              // 000000001134: D4110004 02028545
	v_cmp_lt_f32_e64 s5, v68, v72                              // 00000000113C: D4110005 02029144
	v_cndmask_b32_e64 v71, v74, v71, s2                        // 000000001144: D5010047 000A8F4A
	v_cmp_lt_f32_e64 s2, v72, v68                              // 00000000114C: D4110002 02028948
	s_and_b32 s3, vcc_lo, s3                                   // 000000001154: 8B03036A
	s_and_b32 s4, vcc_lo, s4                                   // 000000001158: 8B04046A
	s_wait_alu depctr_sa_sdst(0)                               // 00000000115C: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 000000001160: 8C6A0305
	s_or_b32 s2, s2, s4                                        // 000000001164: 8C020402
	s_wait_alu depctr_sa_sdst(0)                               // 000000001168: BF88FF9E
	v_cndmask_b32_e32 v74, v68, v72, vcc_lo                    // 00000000116C: 02949144
	v_cndmask_b32_e64 v68, v68, v72, s2                        // 000000001170: D5010044 000A9144
	v_cndmask_b32_e32 v72, v66, v69, vcc_lo                    // 000000001178: 02908B42
	v_cndmask_b32_e64 v66, v66, v69, s2                        // 00000000117C: D5010042 000A8B42
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 000000001184: BF870224
	v_cmp_eq_f32_e32 vcc_lo, v103, v74                         // 000000001188: 7C249567
	v_cmp_lt_f32_e64 s5, v103, v74                             // 00000000118C: D4110005 02029567
	v_cmp_lt_f32_e64 s3, v73, v72                              // 000000001194: D4110003 02029149
	v_cmp_eq_f32_e64 s2, v67, v68                              // 00000000119C: D4120002 02028943
	v_cmp_lt_f32_e64 s4, v71, v66                              // 0000000011A4: D4110004 02028547
	v_cmp_lt_f32_e64 s6, v67, v68                              // 0000000011AC: D4110006 02028943
	s_and_b32 s3, vcc_lo, s3                                   // 0000000011B4: 8B03036A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000011B8: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 0000000011BC: 8C6A0305
	s_and_b32 s2, s2, s4                                       // 0000000011C0: 8B020402
	s_wait_alu depctr_sa_sdst(0)                               // 0000000011C4: BF88FF9E
	v_cndmask_b32_e32 v69, v74, v103, vcc_lo                   // 0000000011C8: 028ACF4A
	s_or_b32 s2, s6, s2                                        // 0000000011CC: 8C020206
	v_cndmask_b32_e32 v74, v103, v74, vcc_lo                   // 0000000011D0: 02949567
	s_wait_alu depctr_sa_sdst(0)                               // 0000000011D4: BF88FF9E
	v_cndmask_b32_e64 v104, v68, v67, s2                       // 0000000011D8: D5010068 000A8744
	v_cndmask_b32_e64 v67, v67, v68, s2                        // 0000000011E0: D5010043 000A8943
	v_mov_b32_e32 v103, v69                                    // 0000000011E8: 7ECE0345
	v_dual_cndmask_b32 v121, v72, v73 :: v_dual_cndmask_b32 v68, v73, v72// 0000000011EC: CA529348 79449149
	v_cndmask_b32_e64 v122, v66, v71, s2                       // 0000000011F4: D501007A 000A8F42
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)// 0000000011FC: BF870194
	v_dual_mov_b32 v123, v104 :: v_dual_mov_b32 v72, v67       // 000000001200: CA100168 7B480143
	v_dual_mov_b32 v124, v121 :: v_dual_mov_b32 v73, v68       // 000000001208: CA100179 7C480144
	v_permlanex16_b32 v103, v103, s10, 0xfedcba98              // 000000001210: D65C0067 03FC1567 FEDCBA98
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 00000000121C: BF870223
	v_permlanex16_b32 v123, v123, s10, 0xfedcba98              // 000000001220: D65C007B 03FC157B FEDCBA98
	v_cndmask_b32_e64 v66, v71, v66, s2                        // 00000000122C: D5010042 000A8547
	v_permlanex16_b32 v124, v124, s10, 0xfedcba98              // 000000001234: D65C007C 03FC157C FEDCBA98
	v_permlanex16_b32 v73, v73, s10, 0xfedcba98                // 000000001240: D65C0049 03FC1549 FEDCBA98
	v_mov_b32_e32 v125, v122                                   // 00000000124C: 7EFA037A
	v_cmp_eq_f32_e32 vcc_lo, v69, v103                         // 000000001250: 7C24CF45
	v_cmp_eq_f32_e64 s3, v104, v123                            // 000000001254: D4120003 0202F768
	v_cmp_gt_f32_e64 s4, v121, v124                            // 00000000125C: D4140004 0202F979
	v_cmp_gt_f32_e64 s6, v69, v103                             // 000000001264: D4140006 0202CF45
	v_permlanex16_b32 v125, v125, s10, 0xfedcba98              // 00000000126C: D65C007D 03FC157D FEDCBA98
	v_cmp_gt_f32_e64 s7, v104, v123                            // 000000001278: D4140007 0202F768
	v_permlanex16_b32 v72, v72, s10, 0xfedcba98                // 000000001280: D65C0048 03FC1548 FEDCBA98
	s_and_b32 s2, vcc_lo, s4                                   // 00000000128C: 8B02046A
	v_cmp_gt_f32_e64 s4, v68, v73                              // 000000001290: D4140004 02029344
	v_cmp_gt_f32_e64 s5, v122, v125                            // 000000001298: D4140005 0202FB7A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000012A0: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 0000000012A4: 8C020206
	s_wait_alu depctr_sa_sdst(0)                               // 0000000012A8: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 0000000012AC: 8D6A0002
	s_and_b32 s3, s3, s5                                       // 0000000012B0: 8B030503
	s_wait_alu depctr_sa_sdst(0)                               // 0000000012B4: BF88FF9E
	v_cndmask_b32_e32 v69, v69, v103, vcc_lo                   // 0000000012B8: 028ACF45
	s_or_b32 s3, s7, s3                                        // 0000000012BC: 8C030307
	v_mov_b32_e32 v103, v66                                    // 0000000012C0: 7ECE0342
	s_wait_alu depctr_sa_sdst(0)                               // 0000000012C4: BF88FF9E
	s_xor_b32 s2, s3, s0                                       // 0000000012C8: 8D020003
	v_cndmask_b32_e32 v121, v121, v124, vcc_lo                 // 0000000012CC: 02F2F979
	s_wait_alu depctr_sa_sdst(0)                               // 0000000012D0: BF88FF9E
	v_cndmask_b32_e64 v104, v104, v123, s2                     // 0000000012D4: D5010068 000AF768
	v_mov_b32_e32 v71, v74                                     // 0000000012DC: 7E8E034A
	v_permlanex16_b32 v103, v103, s10, 0xfedcba98              // 0000000012E0: D65C0067 03FC1567 FEDCBA98
	v_cmp_eq_f32_e64 s3, v67, v72                              // 0000000012EC: D4120003 02029143
	v_cmp_gt_f32_e64 s7, v67, v72                              // 0000000012F4: D4140007 02029143
	v_cndmask_b32_e64 v122, v122, v125, s2                     // 0000000012FC: D501007A 000AFB7A
	v_permlanex16_b32 v71, v71, s10, 0xfedcba98                // 000000001304: D65C0047 03FC1547 FEDCBA98
	v_cmp_gt_f32_e64 s5, v66, v103                             // 000000001310: D4140005 0202CF42
	s_delay_alu instid0(VALU_DEP_2)                            // 000000001318: BF870002
	v_cmp_eq_f32_e32 vcc_lo, v74, v71                          // 00000000131C: 7C248F4A
	v_cmp_gt_f32_e64 s6, v74, v71                              // 000000001320: D4140006 02028F4A
	s_and_b32 s3, s3, s5                                       // 000000001328: 8B030503
	v_cmp_lt_f32_e64 s5, v104, v69                             // 00000000132C: D4110005 02028B68
	s_wait_alu depctr_sa_sdst(0)                               // 000000001334: BF88FF9E
	s_or_b32 s3, s7, s3                                        // 000000001338: 8C030307
	s_and_b32 s2, vcc_lo, s4                                   // 00000000133C: 8B02046A
	v_cmp_lt_f32_e64 s4, v121, v122                            // 000000001340: D4110004 0202F579
	s_wait_alu depctr_sa_sdst(0)                               // 000000001348: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 00000000134C: 8C020206
	s_wait_alu depctr_sa_sdst(0)                               // 000000001350: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000001354: 8D6A0002
	s_xor_b32 s2, s3, s0                                       // 000000001358: 8D020003
	s_wait_alu depctr_sa_sdst(0)                               // 00000000135C: BF88FF9E
	v_dual_cndmask_b32 v71, v74, v71 :: v_dual_cndmask_b32 v68, v68, v73// 000000001360: CA528F4A 47449344
	v_cmp_eq_f32_e32 vcc_lo, v104, v69                         // 000000001368: 7C248B68
	v_cmp_lt_f32_e64 s3, v122, v121                            // 00000000136C: D4110003 0202F37A
	v_cndmask_b32_e64 v67, v67, v72, s2                        // 000000001374: D5010043 000A9143
	v_cndmask_b32_e64 v66, v66, v103, s2                       // 00000000137C: D5010042 000ACF42
	v_cmp_lt_f32_e64 s2, v69, v104                             // 000000001384: D4110002 0202D145
	s_and_b32 s4, vcc_lo, s4                                   // 00000000138C: 8B04046A
	s_and_b32 s3, vcc_lo, s3                                   // 000000001390: 8B03036A
	s_wait_alu depctr_sa_sdst(0)                               // 000000001394: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 000000001398: 8C6A0305
	s_or_b32 s2, s2, s4                                        // 00000000139C: 8C020402
	s_wait_alu depctr_sa_sdst(0)                               // 0000000013A0: BF88FF9E
	v_cndmask_b32_e32 v72, v69, v104, vcc_lo                   // 0000000013A4: 0290D145
	v_cndmask_b32_e32 v73, v121, v122, vcc_lo                  // 0000000013A8: 0292F579
	v_cmp_eq_f32_e32 vcc_lo, v67, v71                          // 0000000013AC: 7C248F43
	v_cmp_lt_f32_e64 s3, v66, v68                              // 0000000013B0: D4110003 02028942
	v_cmp_lt_f32_e64 s4, v68, v66                              // 0000000013B8: D4110004 02028544
	v_cndmask_b32_e64 v69, v69, v104, s2                       // 0000000013C0: D5010045 000AD145
	v_cmp_lt_f32_e64 s5, v67, v71                              // 0000000013C8: D4110005 02028F43
	v_cndmask_b32_e64 v74, v121, v122, s2                      // 0000000013D0: D501004A 000AF579
	v_cmp_lt_f32_e64 s2, v71, v67                              // 0000000013D8: D4110002 02028747
	s_and_b32 s3, vcc_lo, s3                                   // 0000000013E0: 8B03036A
	s_and_b32 s4, vcc_lo, s4                                   // 0000000013E4: 8B04046A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000013E8: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 0000000013EC: 8C6A0305
	v_permlanex16_b32 v69, v69, s10, 0xfedcba98                // 0000000013F0: D65C0045 03FC1545 FEDCBA98
	s_or_b32 s2, s2, s4                                        // 0000000013FC: 8C020402
	s_wait_alu depctr_sa_sdst(0)                               // 000000001400: BF88FF9E
	v_cndmask_b32_e32 v103, v71, v67, vcc_lo                   // 000000001404: 02CE8747
	v_cndmask_b32_e64 v67, v71, v67, s2                        // 000000001408: D5010043 000A8747
	v_permlanex16_b32 v72, v72, s10, 0xfedcba98                // 000000001410: D65C0048 03FC1548 FEDCBA98
	v_permlanex16_b32 v74, v74, s10, 0xfedcba98                // 00000000141C: D65C004A 03FC154A FEDCBA98
	v_permlanex16_b32 v73, v73, s10, 0xfedcba98                // 000000001428: D65C0049 03FC1549 FEDCBA98
	v_cndmask_b32_e32 v71, v68, v66, vcc_lo                    // 000000001434: 028E8544
	v_cndmask_b32_e64 v66, v68, v66, s2                        // 000000001438: D5010042 000A8544
	v_cmp_eq_f32_e32 vcc_lo, v65, v69                          // 000000001440: 7C248B41
	v_cmp_eq_f32_e64 s3, v59, v72                              // 000000001444: D4120003 0202913B
	v_cmp_lt_f32_e64 s4, v64, v74                              // 00000000144C: D4110004 02029540
	v_cmp_lt_f32_e64 s5, v58, v73                              // 000000001454: D4110005 0202933A
	v_permlanex16_b32 v67, v67, s10, 0xfedcba98                // 00000000145C: D65C0043 03FC1543 FEDCBA98
	v_permlanex16_b32 v66, v66, s10, 0xfedcba98                // 000000001468: D65C0042 03FC1542 FEDCBA98
	v_cmp_lt_f32_e64 s2, v65, v69                              // 000000001474: D4110002 02028B41
	v_permlanex16_b32 v103, v103, s10, 0xfedcba98              // 00000000147C: D65C0067 03FC1567 FEDCBA98
	v_permlanex16_b32 v71, v71, s10, 0xfedcba98                // 000000001488: D65C0047 03FC1547 FEDCBA98
	s_and_b32 s4, vcc_lo, s4                                   // 000000001494: 8B04046A
	s_and_b32 s8, s3, s5                                       // 000000001498: 8B080503
	v_cmp_eq_f32_e64 s3, v60, v67                              // 00000000149C: D4120003 0202873C
	v_cmp_lt_f32_e64 s5, v61, v66                              // 0000000014A4: D4110005 0202853D
	v_cmp_lt_f32_e64 s6, v59, v72                              // 0000000014AC: D4110006 0202913B
	s_wait_alu depctr_sa_sdst(0)                               // 0000000014B4: BF88FF9E
	s_or_b32 vcc_lo, s2, s4                                    // 0000000014B8: 8C6A0402
	v_cmp_lt_f32_e64 s2, v60, v67                              // 0000000014BC: D4110002 0202873C
	v_cmp_eq_f32_e64 s4, v62, v103                             // 0000000014C4: D4120004 0202CF3E
	v_cmp_lt_f32_e64 s7, v63, v71                              // 0000000014CC: D4110007 02028F3F
	s_and_b32 s3, s3, s5                                       // 0000000014D4: 8B030503
	s_or_b32 s6, s6, s8                                        // 0000000014D8: 8C060806
	v_cmp_lt_f32_e64 s8, v62, v103                             // 0000000014DC: D4110008 0202CF3E
	s_wait_alu depctr_sa_sdst(0)                               // 0000000014E4: BF88FF9E
	s_or_b32 s2, s2, s3                                        // 0000000014E8: 8C020302
	s_and_b32 s4, s4, s7                                       // 0000000014EC: 8B040704
	s_wait_alu depctr_sa_sdst(0)                               // 0000000014F0: BF88FF9E
	v_cndmask_b32_e64 v60, v67, v60, s2                        // 0000000014F4: D501003C 000A7943
	v_cndmask_b32_e64 v61, v66, v61, s2                        // 0000000014FC: D501003D 000A7B42
	s_or_b32 s3, s8, s4                                        // 000000001504: 8C030408
	v_cndmask_b32_e32 v65, v69, v65, vcc_lo                    // 000000001508: 02828345
	s_wait_alu depctr_sa_sdst(0)                               // 00000000150C: BF88FF9E
	v_cndmask_b32_e64 v62, v103, v62, s3                       // 000000001510: D501003E 000E7D67
	v_cndmask_b32_e64 v63, v71, v63, s3                        // 000000001518: D501003F 000E7F47
	v_cndmask_b32_e32 v64, v74, v64, vcc_lo                    // 000000001520: 0280814A
	v_cmp_eq_f32_e32 vcc_lo, v52, v60                          // 000000001524: 7C247934
	v_cmp_lt_f32_e64 s3, v51, v61                              // 000000001528: D4110003 02027B33
	v_cmp_lt_f32_e64 s5, v52, v60                              // 000000001530: D4110005 02027934
	v_cmp_eq_f32_e64 s2, v53, v62                              // 000000001538: D4120002 02027D35
	v_cmp_lt_f32_e64 s4, v50, v63                              // 000000001540: D4110004 02027F32
	v_cndmask_b32_e64 v59, v72, v59, s6                        // 000000001548: D501003B 001A7748
	s_and_b32 s3, vcc_lo, s3                                   // 000000001550: 8B03036A
	v_cndmask_b32_e64 v58, v73, v58, s6                        // 000000001554: D501003A 001A7549
	v_cmp_lt_f32_e64 s6, v53, v62                              // 00000000155C: D4110006 02027D35
	s_wait_alu depctr_sa_sdst(0)                               // 000000001564: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 000000001568: 8C6A0305
	s_and_b32 s2, s2, s4                                       // 00000000156C: 8B020402
	s_wait_alu depctr_sa_sdst(0)                               // 000000001570: BF88FF9E
	v_cndmask_b32_e32 v66, v60, v52, vcc_lo                    // 000000001574: 0284693C
	v_cndmask_b32_e32 v68, v61, v51, vcc_lo                    // 000000001578: 0288673D
	v_dual_cndmask_b32 v52, v52, v60 :: v_dual_cndmask_b32 v51, v51, v61// 00000000157C: CA527934 34327B33
	v_cmp_eq_f32_e32 vcc_lo, v55, v65                          // 000000001584: 7C248337
	v_cmp_lt_f32_e64 s4, v56, v64                              // 000000001588: D4110004 02028138
	s_or_b32 s2, s6, s2                                        // 000000001590: 8C020206
	v_cmp_lt_f32_e64 s6, v55, v65                              // 000000001594: D4110006 02028337
	s_wait_alu depctr_sa_sdst(0)                               // 00000000159C: BF88FF9E
	v_cndmask_b32_e64 v67, v62, v53, s2                        // 0000000015A0: D5010043 000A6B3E
	v_cndmask_b32_e64 v53, v53, v62, s2                        // 0000000015A8: D5010035 000A7D35
	s_and_b32 s4, vcc_lo, s4                                   // 0000000015B0: 8B04046A
	v_cmp_eq_f32_e64 s3, v57, v59                              // 0000000015B4: D4120003 02027739
	s_wait_alu depctr_sa_sdst(0)                               // 0000000015BC: BF88FF9E
	s_or_b32 vcc_lo, s6, s4                                    // 0000000015C0: 8C6A0406
	v_cmp_lt_f32_e64 s5, v54, v58                              // 0000000015C4: D4110005 02027536
	s_wait_alu depctr_sa_sdst(0)                               // 0000000015CC: BF88FF9E
	v_cndmask_b32_e32 v60, v65, v55, vcc_lo                    // 0000000015D0: 02786F41
	v_cndmask_b32_e32 v62, v64, v56, vcc_lo                    // 0000000015D4: 027C7140
	v_cndmask_b32_e64 v69, v63, v50, s2                        // 0000000015D8: D5010045 000A653F
	v_cndmask_b32_e64 v50, v50, v63, s2                        // 0000000015E0: D5010032 000A7F32
	v_cmp_lt_f32_e64 s2, v57, v59                              // 0000000015E8: D4110002 02027739
	s_and_b32 s3, s3, s5                                       // 0000000015F0: 8B030503
	v_dual_cndmask_b32 v55, v55, v65 :: v_dual_cndmask_b32 v56, v56, v64// 0000000015F4: CA528337 37388138
	v_cmp_eq_f32_e32 vcc_lo, v66, v60                          // 0000000015FC: 7C247942
	v_cmp_lt_f32_e64 s4, v68, v62                              // 000000001600: D4110004 02027D44
	s_wait_alu depctr_sa_sdst(0)                               // 000000001608: BF88FF9E
	s_or_b32 s2, s2, s3                                        // 00000000160C: 8C020302
	v_cmp_lt_f32_e64 s6, v66, v60                              // 000000001610: D4110006 02027942
	s_wait_alu depctr_sa_sdst(0)                               // 000000001618: BF88FF9E
	v_cndmask_b32_e64 v61, v59, v57, s2                        // 00000000161C: D501003D 000A733B
	v_cndmask_b32_e64 v63, v58, v54, s2                        // 000000001624: D501003F 000A6D3A
	s_and_b32 s4, vcc_lo, s4                                   // 00000000162C: 8B04046A
	v_cndmask_b32_e64 v54, v54, v58, s2                        // 000000001630: D5010036 000A7536
	s_wait_alu depctr_sa_sdst(0)                               // 000000001638: BF88FF9E
	s_or_b32 vcc_lo, s6, s4                                    // 00000000163C: 8C6A0406
	v_cmp_eq_f32_e64 s3, v67, v61                              // 000000001640: D4120003 02027B43
	v_cmp_lt_f32_e64 s5, v69, v63                              // 000000001648: D4110005 02027F45
	s_wait_alu depctr_sa_sdst(0)                               // 000000001650: BF88FF9E
	v_cndmask_b32_e32 v58, v60, v66, vcc_lo                    // 000000001654: 0274853C
	v_cndmask_b32_e32 v64, v62, v68, vcc_lo                    // 000000001658: 0280893E
	v_cndmask_b32_e32 v60, v66, v60, vcc_lo                    // 00000000165C: 02787942
	v_cndmask_b32_e32 v62, v68, v62, vcc_lo                    // 000000001660: 027C7D44
	v_cmp_eq_f32_e32 vcc_lo, v52, v55                          // 000000001664: 7C246F34
	v_cmp_lt_f32_e64 s4, v51, v56                              // 000000001668: D4110004 02027133
	v_cndmask_b32_e64 v57, v57, v59, s2                        // 000000001670: D5010039 000A7739
	v_cmp_lt_f32_e64 s2, v67, v61                              // 000000001678: D4110002 02027B43
	v_cmp_lt_f32_e64 s6, v52, v55                              // 000000001680: D4110006 02026F34
	s_and_b32 s3, s3, s5                                       // 000000001688: 8B030503
	s_and_b32 s4, vcc_lo, s4                                   // 00000000168C: 8B04046A
	v_cmp_lt_f32_e64 s5, v50, v54                              // 000000001690: D4110005 02026D32
	s_wait_alu depctr_sa_sdst(0)                               // 000000001698: BF88FF9E
	s_or_b32 s2, s2, s3                                        // 00000000169C: 8C020302
	s_or_b32 vcc_lo, s6, s4                                    // 0000000016A0: 8C6A0406
	s_wait_alu depctr_sa_sdst(0)                               // 0000000016A4: BF88FF9E
	v_cndmask_b32_e64 v65, v63, v69, s2                        // 0000000016A8: D5010041 000A8B3F
	v_cndmask_b32_e32 v66, v55, v52, vcc_lo                    // 0000000016B0: 02846937
	v_dual_cndmask_b32 v52, v52, v55 :: v_dual_mov_b32 v55, v58// 0000000016B4: CA506F34 3436013A
	v_mov_b32_e32 v72, v64                                     // 0000000016BC: 7E900340
	v_cndmask_b32_e64 v59, v61, v67, s2                        // 0000000016C0: D501003B 000A873D
	v_dual_cndmask_b32 v68, v56, v51 :: v_dual_cndmask_b32 v51, v51, v56// 0000000016C8: CA526738 44327133
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 0000000016D0: BF870214
	v_permlanex16_b32 v55, v55, s10, 0xfedcba98                // 0000000016D4: D65C0037 03FC1537 FEDCBA98
	v_permlanex16_b32 v72, v72, s10, 0xfedcba98                // 0000000016E0: D65C0048 03FC1548 FEDCBA98
	v_cmp_eq_f32_e64 s3, v53, v57                              // 0000000016EC: D4120003 02027335
	v_cndmask_b32_e64 v61, v67, v61, s2                        // 0000000016F4: D501003D 000A7B43
	v_cndmask_b32_e64 v63, v69, v63, s2                        // 0000000016FC: D501003F 000A7F45
	v_cmp_lt_f32_e64 s2, v53, v57                              // 000000001704: D4110002 02027335
	v_cmp_eq_f32_e32 vcc_lo, v58, v55                          // 00000000170C: 7C246F3A
	v_cmp_gt_f32_e64 s4, v64, v72                              // 000000001710: D4140004 02029140
	s_and_b32 s3, s3, s5                                       // 000000001718: 8B030503
	v_cmp_gt_f32_e64 s6, v58, v55                              // 00000000171C: D4140006 02026F3A
	s_wait_alu depctr_sa_sdst(0)                               // 000000001724: BF88FF9E
	s_or_b32 s2, s2, s3                                        // 000000001728: 8C020302
	v_mov_b32_e32 v71, v59                                     // 00000000172C: 7E8E033B
	s_wait_alu depctr_sa_sdst(0)                               // 000000001730: BF88FF9E
	v_cndmask_b32_e64 v67, v57, v53, s2                        // 000000001734: D5010043 000A6B39
	v_cndmask_b32_e64 v69, v54, v50, s2                        // 00000000173C: D5010045 000A6536
	v_mov_b32_e32 v73, v65                                     // 000000001744: 7E920341
	v_cndmask_b32_e64 v53, v53, v57, s2                        // 000000001748: D5010035 000A7335
	v_cndmask_b32_e64 v50, v50, v54, s2                        // 000000001750: D5010032 000A6D32
	s_and_b32 s2, vcc_lo, s4                                   // 000000001758: 8B02046A
	v_permlanex16_b32 v71, v71, s10, 0xfedcba98                // 00000000175C: D65C0047 03FC1547 FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 000000001768: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 00000000176C: 8C020206
	v_permlanex16_b32 v73, v73, s10, 0xfedcba98                // 000000001770: D65C0049 03FC1549 FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 00000000177C: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000001780: 8D6A0002
	v_dual_mov_b32 v56, v61 :: v_dual_mov_b32 v57, v62         // 000000001784: CA10013D 3838013E
	s_wait_alu depctr_sa_sdst(0)                               // 00000000178C: BF88FF9E
	v_dual_cndmask_b32 v54, v58, v55 :: v_dual_mov_b32 v55, v60// 000000001790: CA506F3A 3636013C
	v_cmp_eq_f32_e64 s3, v59, v71                              // 000000001798: D4120003 02028F3B
	v_cmp_gt_f32_e64 s5, v65, v73                              // 0000000017A0: D4140005 02029341
	v_permlanex16_b32 v57, v57, s10, 0xfedcba98                // 0000000017A8: D65C0039 03FC1539 FEDCBA98
	v_cmp_gt_f32_e64 s7, v59, v71                              // 0000000017B4: D4140007 02028F3B
	v_permlanex16_b32 v55, v55, s10, 0xfedcba98                // 0000000017BC: D65C0037 03FC1537 FEDCBA98
	v_cndmask_b32_e32 v64, v64, v72, vcc_lo                    // 0000000017C8: 02809140
	s_and_b32 s3, s3, s5                                       // 0000000017CC: 8B030503
	v_cmp_gt_f32_e64 s4, v62, v57                              // 0000000017D0: D4140004 0202733E
	s_wait_alu depctr_sa_sdst(0)                               // 0000000017D8: BF88FF9E
	s_or_b32 s3, s7, s3                                        // 0000000017DC: 8C030307
	v_cmp_eq_f32_e32 vcc_lo, v60, v55                          // 0000000017E0: 7C246F3C
	v_cmp_gt_f32_e64 s6, v60, v55                              // 0000000017E4: D4140006 02026F3C
	s_wait_alu depctr_sa_sdst(0)                               // 0000000017EC: BF88FF9E
	s_xor_b32 s2, s3, s0                                       // 0000000017F0: 8D020003
	v_mov_b32_e32 v58, v63                                     // 0000000017F4: 7E74033F
	s_wait_alu depctr_sa_sdst(0)                               // 0000000017F8: BF88FF9E
	v_cndmask_b32_e64 v59, v59, v71, s2                        // 0000000017FC: D501003B 000A8F3B
	v_cndmask_b32_e64 v65, v65, v73, s2                        // 000000001804: D5010041 000A9341
	s_and_b32 s2, vcc_lo, s4                                   // 00000000180C: 8B02046A
	v_permlanex16_b32 v56, v56, s10, 0xfedcba98                // 000000001810: D65C0038 03FC1538 FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 00000000181C: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 000000001820: 8C020206
	v_mov_b32_e32 v71, v67                                     // 000000001824: 7E8E0343
	s_wait_alu depctr_sa_sdst(0)                               // 000000001828: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 00000000182C: 8D6A0002
	v_mov_b32_e32 v72, v68                                     // 000000001830: 7E900344
	s_wait_alu depctr_sa_sdst(0)                               // 000000001834: BF88FF9E
	v_dual_cndmask_b32 v60, v60, v55 :: v_dual_mov_b32 v73, v69// 000000001838: CA506F3C 3C480145
	v_mov_b32_e32 v55, v66                                     // 000000001840: 7E6E0342
	v_permlanex16_b32 v58, v58, s10, 0xfedcba98                // 000000001844: D65C003A 03FC153A FEDCBA98
	v_cmp_eq_f32_e64 s3, v61, v56                              // 000000001850: D4120003 0202713D
	v_cmp_gt_f32_e64 s7, v61, v56                              // 000000001858: D4140007 0202713D
	v_permlanex16_b32 v71, v71, s10, 0xfedcba98                // 000000001860: D65C0047 03FC1547 FEDCBA98
	v_permlanex16_b32 v55, v55, s10, 0xfedcba98                // 00000000186C: D65C0037 03FC1537 FEDCBA98
	v_cmp_gt_f32_e64 s5, v63, v58                              // 000000001878: D4140005 0202753F
	v_permlanex16_b32 v73, v73, s10, 0xfedcba98                // 000000001880: D65C0049 03FC1549 FEDCBA98
	s_delay_alu instid0(VALU_DEP_3)                            // 00000000188C: BF870003
	v_cmp_gt_f32_e64 s6, v66, v55                              // 000000001890: D4140006 02026F42
	s_and_b32 s3, s3, s5                                       // 000000001898: 8B030503
	v_cmp_gt_f32_e64 s5, v69, v73                              // 00000000189C: D4140005 02029345
	s_wait_alu depctr_sa_sdst(0)                               // 0000000018A4: BF88FF9E
	s_or_b32 s3, s7, s3                                        // 0000000018A8: 8C030307
	v_cmp_gt_f32_e64 s7, v67, v71                              // 0000000018AC: D4140007 02028F43
	s_wait_alu depctr_sa_sdst(0)                               // 0000000018B4: BF88FF9E
	s_xor_b32 s2, s3, s0                                       // 0000000018B8: 8D020003
	v_cmp_eq_f32_e64 s3, v67, v71                              // 0000000018BC: D4120003 02028F43
	s_wait_alu depctr_sa_sdst(0)                               // 0000000018C4: BF88FF9E
	v_cndmask_b32_e64 v56, v61, v56, s2                        // 0000000018C8: D5010038 000A713D
	v_cndmask_b32_e32 v61, v62, v57, vcc_lo                    // 0000000018D0: 027A733E
	v_permlanex16_b32 v72, v72, s10, 0xfedcba98                // 0000000018D4: D65C0048 03FC1548 FEDCBA98
	v_cmp_eq_f32_e32 vcc_lo, v66, v55                          // 0000000018E0: 7C246F42
	v_cndmask_b32_e64 v62, v63, v58, s2                        // 0000000018E4: D501003E 000A753F
	v_mov_b32_e32 v58, v51                                     // 0000000018EC: 7E740333
	s_and_b32 s3, s3, s5                                       // 0000000018F0: 8B030503
	v_cmp_gt_f32_e64 s4, v68, v72                              // 0000000018F4: D4140004 02029144
	s_wait_alu depctr_sa_sdst(0)                               // 0000000018FC: BF88FF9E
	s_or_b32 s3, s7, s3                                        // 000000001900: 8C030307
	v_permlanex16_b32 v58, v58, s10, 0xfedcba98                // 000000001904: D65C003A 03FC153A FEDCBA98
	s_and_b32 s2, vcc_lo, s4                                   // 000000001910: 8B02046A
	s_wait_alu depctr_sa_sdst(0)                               // 000000001914: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 000000001918: 8C020206
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000191C: BF870001
	v_cmp_gt_f32_e64 s4, v51, v58                              // 000000001920: D4140004 02027533
	s_wait_alu depctr_sa_sdst(0)                               // 000000001928: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 00000000192C: 8D6A0002
	s_wait_alu depctr_sa_sdst(0)                               // 000000001930: BF88FF9E
	v_dual_mov_b32 v57, v53 :: v_dual_cndmask_b32 v68, v68, v72// 000000001934: CA120135 39449144
	v_cndmask_b32_e32 v63, v66, v55, vcc_lo                    // 00000000193C: 027E6F42
	v_dual_mov_b32 v55, v52 :: v_dual_mov_b32 v66, v50         // 000000001940: CA100134 37420132
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001948: BF870123
	v_permlanex16_b32 v57, v57, s10, 0xfedcba98                // 00000000194C: D65C0039 03FC1539 FEDCBA98
	s_xor_b32 s2, s3, s0                                       // 000000001958: 8D020003
	v_permlanex16_b32 v55, v55, s10, 0xfedcba98                // 00000000195C: D65C0037 03FC1537 FEDCBA98
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000001968: BF870193
	v_permlanex16_b32 v66, v66, s10, 0xfedcba98                // 00000000196C: D65C0042 03FC1542 FEDCBA98
	v_cmp_eq_f32_e64 s3, v53, v57                              // 000000001978: D4120003 02027335
	v_cmp_gt_f32_e64 s7, v53, v57                              // 000000001980: D4140007 02027335
	s_wait_alu depctr_sa_sdst(0)                               // 000000001988: BF88FF9E
	v_cndmask_b32_e64 v67, v67, v71, s2                        // 00000000198C: D5010043 000A8F43
	v_cmp_eq_f32_e32 vcc_lo, v52, v55                          // 000000001994: 7C246F34
	v_cmp_gt_f32_e64 s5, v50, v66                              // 000000001998: D4140005 02028532
	v_cmp_gt_f32_e64 s6, v52, v55                              // 0000000019A0: D4140006 02026F34
	v_cndmask_b32_e64 v69, v69, v73, s2                        // 0000000019A8: D5010045 000A9345
	s_and_b32 s2, vcc_lo, s4                                   // 0000000019B0: 8B02046A
	s_and_b32 s3, s3, s5                                       // 0000000019B4: 8B030503
	s_wait_alu depctr_sa_sdst(0)                               // 0000000019B8: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 0000000019BC: 8C020206
	s_or_b32 s3, s7, s3                                        // 0000000019C0: 8C030307
	s_wait_alu depctr_sa_sdst(0)                               // 0000000019C4: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 0000000019C8: 8D6A0002
	s_xor_b32 s2, s3, s0                                       // 0000000019CC: 8D020003
	s_wait_alu depctr_sa_sdst(0)                               // 0000000019D0: BF88FF9E
	v_cndmask_b32_e32 v71, v52, v55, vcc_lo                    // 0000000019D4: 028E6F34
	v_cndmask_b32_e32 v73, v51, v58, vcc_lo                    // 0000000019D8: 02927533
	v_cmp_eq_f32_e32 vcc_lo, v59, v54                          // 0000000019DC: 7C246D3B
	v_cmp_lt_f32_e64 s4, v64, v65                              // 0000000019E0: D4110004 02028340
	v_cndmask_b32_e64 v72, v53, v57, s2                        // 0000000019E8: D5010048 000A7335
	v_cmp_lt_f32_e64 s3, v65, v64                              // 0000000019F0: D4110003 02028141
	v_cndmask_b32_e64 v66, v50, v66, s2                        // 0000000019F8: D5010042 000A8532
	v_cmp_lt_f32_e64 s2, v54, v59                              // 000000001A00: D4110002 02027736
	v_cmp_lt_f32_e64 s5, v59, v54                              // 000000001A08: D4110005 02026D3B
	s_and_b32 s4, vcc_lo, s4                                   // 000000001A10: 8B04046A
	s_and_b32 s3, vcc_lo, s3                                   // 000000001A14: 8B03036A
	v_cmp_lt_f32_e64 s7, v60, v56                              // 000000001A18: D4110007 0202713C
	s_wait_alu depctr_sa_sdst(0)                               // 000000001A20: BF88FF9E
	s_or_b32 s2, s2, s4                                        // 000000001A24: 8C020402
	s_or_b32 vcc_lo, s5, s3                                    // 000000001A28: 8C6A0305
	s_wait_alu depctr_sa_sdst(0)                               // 000000001A2C: BF88FF9E
	v_cndmask_b32_e64 v53, v54, v59, s2                        // 000000001A30: D5010035 000A7736
	v_cndmask_b32_e64 v50, v64, v65, s2                        // 000000001A38: D5010032 000A8340
	v_cmp_eq_f32_e64 s2, v56, v60                              // 000000001A40: D4120002 02027938
	v_cmp_lt_f32_e64 s3, v62, v61                              // 000000001A48: D4110003 02027B3E
	v_cndmask_b32_e32 v52, v54, v59, vcc_lo                    // 000000001A50: 02687736
	v_cmp_lt_f32_e64 s4, v61, v62                              // 000000001A54: D4110004 02027D3D
	v_cmp_eq_f32_e64 s5, v72, v71                              // 000000001A5C: D4120005 02028F48
	v_cmp_lt_f32_e64 s6, v73, v66                              // 000000001A64: D4110006 02028549
	s_and_b32 s8, s2, s3                                       // 000000001A6C: 8B080302
	v_cmp_lt_f32_e64 s3, v71, v72                              // 000000001A70: D4110003 02029147
	s_and_b32 s4, s2, s4                                       // 000000001A78: 8B040402
	s_and_b32 s2, s5, s6                                       // 000000001A7C: 8B020605
	v_cmp_lt_f32_e64 s6, v63, v67                              // 000000001A80: D4110006 0202873F
	s_wait_alu depctr_sa_sdst(0)                               // 000000001A88: BF88FF9E
	s_or_b32 s2, s3, s2                                        // 000000001A8C: 8C020203
	v_cmp_lt_f32_e64 s3, v69, v68                              // 000000001A90: D4110003 02028945
	s_wait_alu depctr_sa_sdst(0)                               // 000000001A98: BF88FF9E
	v_cndmask_b32_e64 v59, v71, v72, s2                        // 000000001A9C: D501003B 000A9147
	v_cndmask_b32_e32 v51, v64, v65, vcc_lo                    // 000000001AA4: 02668340
	v_cmp_lt_f32_e32 vcc_lo, v56, v60                          // 000000001AA8: 7C227938
	v_cndmask_b32_e64 v58, v73, v66, s2                        // 000000001AAC: D501003A 000A8549
	s_or_b32 s2, s7, s4                                        // 000000001AB4: 8C020407
	v_cmp_lt_f32_e64 s4, v68, v69                              // 000000001AB8: D4110004 02028B44
	s_wait_alu depctr_sa_sdst(0)                               // 000000001AC0: BF88FF9E
	v_cndmask_b32_e64 v57, v60, v56, s2                        // 000000001AC4: D5010039 000A713C
	s_or_b32 vcc_lo, vcc_lo, s8                                // 000000001ACC: 8C6A086A
	v_cndmask_b32_e64 v54, v61, v62, s2                        // 000000001AD0: D5010036 000A7D3D
	s_wait_alu depctr_sa_sdst(0)                               // 000000001AD8: BF88FF9E
	v_dual_cndmask_b32 v55, v60, v56 :: v_dual_max_num_f32 v60, v59, v59// 000000001ADC: CA54713C 373C773B
	v_cndmask_b32_e32 v56, v61, v62, vcc_lo                    // 000000001AE4: 02707D3D
	v_max_num_f32_e32 v61, v58, v58                            // 000000001AE8: 2C7A753A
	v_cmp_eq_f32_e64 s2, v67, v63                              // 000000001AEC: D4120002 02027F43
	v_cmp_lt_f32_e32 vcc_lo, v67, v63                          // 000000001AF4: 7C227F43
	v_max_num_f32_e32 v60, 0xff800000, v60                     // 000000001AF8: 2C7878FF FF800000
	s_and_b32 s3, s2, s3                                       // 000000001B00: 8B030302
	s_and_b32 s2, s2, s4                                       // 000000001B04: 8B020402
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001B08: BF870001
	v_cndmask_b32_e64 v74, v60, 0xff800000, s1                 // 000000001B0C: D501004A 0005FF3C FF800000
	v_max_num_f32_e32 v61, 0xff800000, v61                     // 000000001B18: 2C7A7AFF FF800000
	s_wait_alu depctr_sa_sdst(0)                               // 000000001B20: BF88FF9E
	s_or_b32 vcc_lo, vcc_lo, s3                                // 000000001B24: 8C6A036A
	v_cmp_lt_f32_e64 s3, v72, v71                              // 000000001B28: D4110003 02028F48
	s_wait_alu depctr_sa_sdst(0)                               // 000000001B30: BF88FF9E
	v_cndmask_b32_e32 v60, v63, v67, vcc_lo                    // 000000001B34: 0278873F
	v_mov_b32_e32 v64, v74                                     // 000000001B38: 7E80034A
	v_cndmask_b32_e64 v104, v61, 0xff800000, s1                // 000000001B3C: D5010068 0005FF3D FF800000
	v_cndmask_b32_e32 v61, v68, v69, vcc_lo                    // 000000001B48: 027A8B44
	v_cmp_lt_f32_e32 vcc_lo, v66, v73                          // 000000001B4C: 7C229342
	s_or_b32 s2, s6, s2                                        // 000000001B50: 8C020206
	v_permlanex16_b32 v64, v64, s10, 0xfedcba98                // 000000001B54: D65C0040 03FC1540 FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 000000001B60: BF88FF9E
	v_cndmask_b32_e64 v62, v63, v67, s2                        // 000000001B64: D501003E 000A873F
	v_cndmask_b32_e64 v63, v68, v69, s2                        // 000000001B6C: D501003F 000A8B44
	s_and_b32 s2, s5, vcc_lo                                   // 000000001B74: 8B026A05
	v_mov_b32_e32 v69, 0x7f800000                              // 000000001B78: 7E8A02FF 7F800000
	s_wait_alu depctr_sa_sdst(0)                               // 000000001B80: BF88FF9E
	s_or_b32 vcc_lo, s3, s2                                    // 000000001B84: 8C6A0203
	v_mov_b32_e32 v65, v104                                    // 000000001B88: 7E820368
	s_wait_alu depctr_sa_sdst(0)                               // 000000001B8C: BF88FF9E
	v_dual_max_num_f32 v67, v64, v64 :: v_dual_cndmask_b32 v64, v73, v66// 000000001B90: CA928140 43408549
	v_dual_mov_b32 v73, 0x7f800000 :: v_dual_mov_b32 v66, -1.0 // 000000001B98: CA1000FF 494200F3 7F800000
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001BA4: BF870093
	v_permlanex16_b32 v65, v65, s10, 0xfedcba98                // 000000001BA8: D65C0041 03FC1541 FEDCBA98
	v_dual_max_num_f32 v68, v65, v65 :: v_dual_cndmask_b32 v65, v71, v72// 000000001BB4: CA928341 44409147
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)// 000000001BBC: BF8701A4
	v_dual_max_num_f32 v103, v74, v67 :: v_dual_mov_b32 v72, -1.0// 000000001BC0: CA90874A 674800F3
	v_mov_b32_e32 v74, 0x7f800000                              // 000000001BC8: 7E9402FF 7F800000
	v_dual_max_num_f32 v104, v104, v68 :: v_dual_mov_b32 v71, -1.0// 000000001BD0: CA908968 684600F3
	v_dual_mov_b32 v68, 0x7f800000 :: v_dual_mov_b32 v67, -1.0 // 000000001BD8: CA1000FF 444200F3 7F800000
	s_branch 4                                                 // 000000001BE4: BFA00004 <nearest+0x1bf8>
	s_add_co_i32 s15, s15, -1                                  // 000000001BE8: 810FC10F
	s_wait_alu depctr_sa_sdst(0)                               // 000000001BEC: BF88FF9E
	s_cmp_lg_u32 s15, 0                                        // 000000001BF0: BF07800F
	s_cbranch_scc0 63943                                       // 000000001BF4: BFA1F9C7 <nearest+0x314>
	s_cmp_nlt_f32 s14, 0x41000000                              // 000000001BF8: BF4EFF0E 41000000
	s_cbranch_scc1 65529                                       // 000000001C00: BFA2FFF9 <nearest+0x1be8>
	s_cmp_le_f32 s14, 0                                        // 000000001C04: BF43800E
	v_cmp_ge_f32_e64 s2, v103, -v0                             // 000000001C08: D4160002 42020167
	v_cmp_ge_f32_e64 s4, v103, -v1                             // 000000001C10: D4160004 42020367
	s_cselect_b32 s5, -1, 0                                    // 000000001C18: 980580C1
	s_wait_alu depctr_sa_sdst(0)                               // 000000001C1C: BF88FF9E
	s_and_b32 s3, s2, s5                                       // 000000001C20: 8B030502
	s_and_b32 s6, s4, s5                                       // 000000001C24: 8B060504
	s_mov_b32 s5, 0                                            // 000000001C28: BE850080
	s_wait_alu depctr_sa_sdst(0)                               // 000000001C2C: BF88FF9E
	s_or_b32 vcc_lo, s3, s6                                    // 000000001C30: 8C6A0603
	s_wait_alu depctr_sa_sdst(0)                               // 000000001C34: BF88FF9E
	s_cbranch_vccz 29                                          // 000000001C38: BFA3001D <nearest+0x1cb0>
	v_cmp_gt_f32_e64 s5, v103, -v0                             // 000000001C3C: D4140005 42020167
	v_cmp_gt_f32_e32 vcc_lo, v104, v105                        // 000000001C44: 7C28D368
	v_cmp_gt_f32_e64 s7, v103, -v1                             // 000000001C48: D4140007 42020367
	v_cmp_gt_f32_e64 s2, v104, v106                            // 000000001C50: D4140002 0202D568
	v_cmp_lt_f32_e64 s4, -1.0, v66                             // 000000001C58: D4110004 020284F3
	s_or_b32 s5, s5, vcc_lo                                    // 000000001C60: 8C056A05
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v67                         // 000000001C64: 7C2286F3
	s_or_b32 s7, s7, s2                                        // 000000001C68: 8C070207
	s_wait_alu depctr_sa_sdst(0)                               // 000000001C6C: BF88FF9E
	s_and_b32 s2, s5, s3                                       // 000000001C70: 8B020305
	s_and_b32 s3, s7, s6                                       // 000000001C74: 8B030607
	s_wait_alu depctr_sa_sdst(0)                               // 000000001C78: BF88FF9E
	s_and_b32 s4, s4, s2                                       // 000000001C7C: 8B040204
	s_and_b32 s5, vcc_lo, s3                                   // 000000001C80: 8B05036A
	s_wait_alu depctr_sa_sdst(0)                               // 000000001C84: BF88FF9E
	s_or_b32 s4, s4, s5                                        // 000000001C88: 8C040504
	s_wait_alu depctr_sa_sdst(0)                               // 000000001C8C: BF88FF9E
	v_cndmask_b32_e64 v121, 0, 1, s4                           // 000000001C90: D5010079 00110280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001C98: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v121                           // 000000001C9C: 7C9AF280
	s_cbranch_vccz 6                                           // 000000001CA0: BFA30006 <nearest+0x1cbc>
	s_mov_b32 s6, 0                                            // 000000001CA4: BE860080
	s_mov_b32 s5, 1.0                                          // 000000001CA8: BE8500F2
	s_branch 26                                                // 000000001CAC: BFA0001A <nearest+0x1d18>
	s_mov_b32 s6, 0x41000000                                   // 000000001CB0: BE8600FF 41000000
	s_branch 23                                                // 000000001CB8: BFA00017 <nearest+0x1d18>
	v_cndmask_b32_e64 v121, v74, -v0, s2                       // 000000001CBC: D5010079 400A014A
	v_cndmask_b32_e64 v122, v73, -v1, s3                       // 000000001CC4: D501007A 400E0349
	v_cndmask_b32_e64 v123, v72, v105, s2                      // 000000001CCC: D501007B 000AD348
	v_cndmask_b32_e64 v124, v71, v106, s3                      // 000000001CD4: D501007C 000ED547
	v_cndmask_b32_e64 v68, v68, v74, s2                        // 000000001CDC: D5010044 000A9544
	v_cndmask_b32_e64 v69, v69, v73, s3                        // 000000001CE4: D5010045 000E9345
	v_cndmask_b32_e64 v66, v66, v72, s2                        // 000000001CEC: D5010042 000A9142
	v_cndmask_b32_e64 v67, v67, v71, s3                        // 000000001CF4: D5010043 000E8F43
	v_dual_mov_b32 v72, v123 :: v_dual_mov_b32 v71, v124       // 000000001CFC: CA10017B 4846017C
	v_dual_mov_b32 v74, v121 :: v_dual_mov_b32 v73, v122       // 000000001D04: CA100179 4A48017A
	s_mov_b32 s6, 0x41000000                                   // 000000001D0C: BE8600FF 41000000
	s_mov_b32 s5, 0                                            // 000000001D14: BE850080
	s_wait_alu depctr_sa_sdst(0)                               // 000000001D18: BF88FF9E
	s_cmp_lt_f32 s5, 0.5                                       // 000000001D1C: BF41F005
	v_cmp_ge_f32_e64 s3, v103, -v2                             // 000000001D20: D4160003 42020567
	v_cmp_ge_f32_e64 s4, v103, -v3                             // 000000001D28: D4160004 42020767
	s_cselect_b32 s2, -1, 0                                    // 000000001D30: 980280C1
	s_cmp_le_f32 s14, 1.0                                      // 000000001D34: BF43F20E
	s_cselect_b32 s7, -1, 0                                    // 000000001D38: 980780C1
	s_wait_alu depctr_sa_sdst(0)                               // 000000001D3C: BF88FF9E
	s_and_b32 s2, s7, s2                                       // 000000001D40: 8B020207
	s_wait_alu depctr_sa_sdst(0)                               // 000000001D44: BF88FF9E
	s_and_b32 s3, s3, s2                                       // 000000001D48: 8B030203
	s_and_b32 s7, s4, s2                                       // 000000001D4C: 8B070204
	s_wait_alu depctr_sa_sdst(0)                               // 000000001D50: BF88FF9E
	s_or_b32 vcc_lo, s3, s7                                    // 000000001D54: 8C6A0703
	s_wait_alu depctr_sa_sdst(0)                               // 000000001D58: BF88FF9E
	s_cbranch_vccz 48                                          // 000000001D5C: BFA30030 <nearest+0x1e20>
	v_cmp_gt_f32_e64 s8, v103, -v2                             // 000000001D60: D4140008 42020567
	v_cmp_gt_f32_e32 vcc_lo, v104, v107                        // 000000001D68: 7C28D768
	v_cmp_gt_f32_e64 s20, v103, -v3                            // 000000001D6C: D4140014 42020767
	v_cmp_gt_f32_e64 s2, v104, v108                            // 000000001D74: D4140002 0202D968
	v_cmp_lt_f32_e64 s4, -1.0, v66                             // 000000001D7C: D4110004 020284F3
	s_or_b32 s8, s8, vcc_lo                                    // 000000001D84: 8C086A08
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v67                         // 000000001D88: 7C2286F3
	s_or_b32 s20, s20, s2                                      // 000000001D8C: 8C140214
	s_and_b32 s2, s8, s3                                       // 000000001D90: 8B020308
	s_and_b32 s3, s20, s7                                      // 000000001D94: 8B030714
	s_wait_alu depctr_sa_sdst(0)                               // 000000001D98: BF88FF9E
	s_and_b32 s4, s4, s2                                       // 000000001D9C: 8B040204
	s_and_b32 s7, vcc_lo, s3                                   // 000000001DA0: 8B07036A
	s_wait_alu depctr_sa_sdst(0)                               // 000000001DA4: BF88FF9E
	s_or_b32 s4, s4, s7                                        // 000000001DA8: 8C040704
	s_wait_alu depctr_sa_sdst(0)                               // 000000001DAC: BF88FF9E
	v_cndmask_b32_e64 v121, 0, 1, s4                           // 000000001DB0: D5010079 00110280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001DB8: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v121                           // 000000001DBC: 7C9AF280
	s_cbranch_vccz 3                                           // 000000001DC0: BFA30003 <nearest+0x1dd0>
	s_mov_b32 s5, 1.0                                          // 000000001DC4: BE8500F2
	s_mov_b32 s6, 1.0                                          // 000000001DC8: BE8600F2
	s_branch 20                                                // 000000001DCC: BFA00014 <nearest+0x1e20>
	v_cndmask_b32_e64 v121, v74, -v2, s2                       // 000000001DD0: D5010079 400A054A
	v_cndmask_b32_e64 v122, v73, -v3, s3                       // 000000001DD8: D501007A 400E0749
	v_cndmask_b32_e64 v123, v72, v107, s2                      // 000000001DE0: D501007B 000AD748
	v_cndmask_b32_e64 v124, v71, v108, s3                      // 000000001DE8: D501007C 000ED947
	v_cndmask_b32_e64 v68, v68, v74, s2                        // 000000001DF0: D5010044 000A9544
	v_cndmask_b32_e64 v69, v69, v73, s3                        // 000000001DF8: D5010045 000E9345
	v_cndmask_b32_e64 v66, v66, v72, s2                        // 000000001E00: D5010042 000A9142
	v_cndmask_b32_e64 v67, v67, v71, s3                        // 000000001E08: D5010043 000E8F43
	v_dual_mov_b32 v72, v123 :: v_dual_mov_b32 v71, v124       // 000000001E10: CA10017B 4846017C
	v_dual_mov_b32 v74, v121 :: v_dual_mov_b32 v73, v122       // 000000001E18: CA100179 4A48017A
	s_wait_alu depctr_sa_sdst(0)                               // 000000001E20: BF88FF9E
	s_cmp_lt_f32 s5, 0.5                                       // 000000001E24: BF41F005
	v_cmp_ge_f32_e64 s3, v103, -v4                             // 000000001E28: D4160003 42020967
	v_cmp_ge_f32_e64 s4, v103, -v5                             // 000000001E30: D4160004 42020B67
	s_cselect_b32 s2, -1, 0                                    // 000000001E38: 980280C1
	s_cmp_le_f32 s14, 2.0                                      // 000000001E3C: BF43F40E
	s_cselect_b32 s7, -1, 0                                    // 000000001E40: 980780C1
	s_wait_alu depctr_sa_sdst(0)                               // 000000001E44: BF88FF9E
	s_and_b32 s2, s7, s2                                       // 000000001E48: 8B020207
	s_wait_alu depctr_sa_sdst(0)                               // 000000001E4C: BF88FF9E
	s_and_b32 s3, s3, s2                                       // 000000001E50: 8B030203
	s_and_b32 s7, s4, s2                                       // 000000001E54: 8B070204
	s_wait_alu depctr_sa_sdst(0)                               // 000000001E58: BF88FF9E
	s_or_b32 vcc_lo, s3, s7                                    // 000000001E5C: 8C6A0703
	s_wait_alu depctr_sa_sdst(0)                               // 000000001E60: BF88FF9E
	s_cbranch_vccz 48                                          // 000000001E64: BFA30030 <nearest+0x1f28>
	v_cmp_gt_f32_e64 s8, v103, -v4                             // 000000001E68: D4140008 42020967
	v_cmp_gt_f32_e32 vcc_lo, v104, v109                        // 000000001E70: 7C28DB68
	v_cmp_gt_f32_e64 s20, v103, -v5                            // 000000001E74: D4140014 42020B67
	v_cmp_gt_f32_e64 s2, v104, v110                            // 000000001E7C: D4140002 0202DD68
	v_cmp_lt_f32_e64 s4, -1.0, v66                             // 000000001E84: D4110004 020284F3
	s_or_b32 s8, s8, vcc_lo                                    // 000000001E8C: 8C086A08
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v67                         // 000000001E90: 7C2286F3
	s_or_b32 s20, s20, s2                                      // 000000001E94: 8C140214
	s_and_b32 s2, s8, s3                                       // 000000001E98: 8B020308
	s_and_b32 s3, s20, s7                                      // 000000001E9C: 8B030714
	s_wait_alu depctr_sa_sdst(0)                               // 000000001EA0: BF88FF9E
	s_and_b32 s4, s4, s2                                       // 000000001EA4: 8B040204
	s_and_b32 s7, vcc_lo, s3                                   // 000000001EA8: 8B07036A
	s_wait_alu depctr_sa_sdst(0)                               // 000000001EAC: BF88FF9E
	s_or_b32 s4, s4, s7                                        // 000000001EB0: 8C040704
	s_wait_alu depctr_sa_sdst(0)                               // 000000001EB4: BF88FF9E
	v_cndmask_b32_e64 v121, 0, 1, s4                           // 000000001EB8: D5010079 00110280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001EC0: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v121                           // 000000001EC4: 7C9AF280
	s_cbranch_vccz 3                                           // 000000001EC8: BFA30003 <nearest+0x1ed8>
	s_mov_b32 s6, 2.0                                          // 000000001ECC: BE8600F4
	s_mov_b32 s5, 1.0                                          // 000000001ED0: BE8500F2
	s_branch 20                                                // 000000001ED4: BFA00014 <nearest+0x1f28>
	v_cndmask_b32_e64 v121, v74, -v4, s2                       // 000000001ED8: D5010079 400A094A
	v_cndmask_b32_e64 v122, v73, -v5, s3                       // 000000001EE0: D501007A 400E0B49
	v_cndmask_b32_e64 v123, v72, v109, s2                      // 000000001EE8: D501007B 000ADB48
	v_cndmask_b32_e64 v124, v71, v110, s3                      // 000000001EF0: D501007C 000EDD47
	v_cndmask_b32_e64 v68, v68, v74, s2                        // 000000001EF8: D5010044 000A9544
	v_cndmask_b32_e64 v69, v69, v73, s3                        // 000000001F00: D5010045 000E9345
	v_cndmask_b32_e64 v66, v66, v72, s2                        // 000000001F08: D5010042 000A9142
	v_cndmask_b32_e64 v67, v67, v71, s3                        // 000000001F10: D5010043 000E8F43
	v_dual_mov_b32 v72, v123 :: v_dual_mov_b32 v71, v124       // 000000001F18: CA10017B 4846017C
	v_dual_mov_b32 v74, v121 :: v_dual_mov_b32 v73, v122       // 000000001F20: CA100179 4A48017A
	s_wait_alu depctr_sa_sdst(0)                               // 000000001F28: BF88FF9E
	s_cmp_lt_f32 s5, 0.5                                       // 000000001F2C: BF41F005
	v_cmp_ge_f32_e64 s3, v103, -v6                             // 000000001F30: D4160003 42020D67
	v_cmp_ge_f32_e64 s4, v103, -v7                             // 000000001F38: D4160004 42020F67
	s_cselect_b32 s2, -1, 0                                    // 000000001F40: 980280C1
	s_cmp_le_f32 s14, 0x40400000                               // 000000001F44: BF43FF0E 40400000
	s_cselect_b32 s7, -1, 0                                    // 000000001F4C: 980780C1
	s_wait_alu depctr_sa_sdst(0)                               // 000000001F50: BF88FF9E
	s_and_b32 s2, s7, s2                                       // 000000001F54: 8B020207
	s_wait_alu depctr_sa_sdst(0)                               // 000000001F58: BF88FF9E
	s_and_b32 s3, s3, s2                                       // 000000001F5C: 8B030203
	s_and_b32 s7, s4, s2                                       // 000000001F60: 8B070204
	s_wait_alu depctr_sa_sdst(0)                               // 000000001F64: BF88FF9E
	s_or_b32 vcc_lo, s3, s7                                    // 000000001F68: 8C6A0703
	s_wait_alu depctr_sa_sdst(0)                               // 000000001F6C: BF88FF9E
	s_cbranch_vccz 49                                          // 000000001F70: BFA30031 <nearest+0x2038>
	v_cmp_gt_f32_e64 s8, v103, -v6                             // 000000001F74: D4140008 42020D67
	v_cmp_gt_f32_e32 vcc_lo, v104, v111                        // 000000001F7C: 7C28DF68
	v_cmp_gt_f32_e64 s20, v103, -v7                            // 000000001F80: D4140014 42020F67
	v_cmp_gt_f32_e64 s2, v104, v112                            // 000000001F88: D4140002 0202E168
	v_cmp_lt_f32_e64 s4, -1.0, v66                             // 000000001F90: D4110004 020284F3
	s_or_b32 s8, s8, vcc_lo                                    // 000000001F98: 8C086A08
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v67                         // 000000001F9C: 7C2286F3
	s_or_b32 s20, s20, s2                                      // 000000001FA0: 8C140214
	s_and_b32 s2, s8, s3                                       // 000000001FA4: 8B020308
	s_and_b32 s3, s20, s7                                      // 000000001FA8: 8B030714
	s_wait_alu depctr_sa_sdst(0)                               // 000000001FAC: BF88FF9E
	s_and_b32 s4, s4, s2                                       // 000000001FB0: 8B040204
	s_and_b32 s7, vcc_lo, s3                                   // 000000001FB4: 8B07036A
	s_wait_alu depctr_sa_sdst(0)                               // 000000001FB8: BF88FF9E
	s_or_b32 s4, s4, s7                                        // 000000001FBC: 8C040704
	s_wait_alu depctr_sa_sdst(0)                               // 000000001FC0: BF88FF9E
	v_cndmask_b32_e64 v121, 0, 1, s4                           // 000000001FC4: D5010079 00110280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001FCC: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v121                           // 000000001FD0: 7C9AF280
	s_cbranch_vccz 4                                           // 000000001FD4: BFA30004 <nearest+0x1fe8>
	s_mov_b32 s6, 0x40400000                                   // 000000001FD8: BE8600FF 40400000
	s_mov_b32 s5, 1.0                                          // 000000001FE0: BE8500F2
	s_branch 20                                                // 000000001FE4: BFA00014 <nearest+0x2038>
	v_cndmask_b32_e64 v121, v74, -v6, s2                       // 000000001FE8: D5010079 400A0D4A
	v_cndmask_b32_e64 v122, v73, -v7, s3                       // 000000001FF0: D501007A 400E0F49
	v_cndmask_b32_e64 v123, v72, v111, s2                      // 000000001FF8: D501007B 000ADF48
	v_cndmask_b32_e64 v124, v71, v112, s3                      // 000000002000: D501007C 000EE147
	v_cndmask_b32_e64 v68, v68, v74, s2                        // 000000002008: D5010044 000A9544
	v_cndmask_b32_e64 v69, v69, v73, s3                        // 000000002010: D5010045 000E9345
	v_cndmask_b32_e64 v66, v66, v72, s2                        // 000000002018: D5010042 000A9142
	v_cndmask_b32_e64 v67, v67, v71, s3                        // 000000002020: D5010043 000E8F43
	v_dual_mov_b32 v72, v123 :: v_dual_mov_b32 v71, v124       // 000000002028: CA10017B 4846017C
	v_dual_mov_b32 v74, v121 :: v_dual_mov_b32 v73, v122       // 000000002030: CA100179 4A48017A
	s_wait_alu depctr_sa_sdst(0)                               // 000000002038: BF88FF9E
	s_cmp_lt_f32 s5, 0.5                                       // 00000000203C: BF41F005
	v_cmp_ge_f32_e64 s3, v103, -v8                             // 000000002040: D4160003 42021167
	v_cmp_ge_f32_e64 s4, v103, -v9                             // 000000002048: D4160004 42021367
	s_cselect_b32 s2, -1, 0                                    // 000000002050: 980280C1
	s_cmp_le_f32 s14, 4.0                                      // 000000002054: BF43F60E
	s_cselect_b32 s7, -1, 0                                    // 000000002058: 980780C1
	s_wait_alu depctr_sa_sdst(0)                               // 00000000205C: BF88FF9E
	s_and_b32 s2, s7, s2                                       // 000000002060: 8B020207
	s_wait_alu depctr_sa_sdst(0)                               // 000000002064: BF88FF9E
	s_and_b32 s3, s3, s2                                       // 000000002068: 8B030203
	s_and_b32 s7, s4, s2                                       // 00000000206C: 8B070204
	s_wait_alu depctr_sa_sdst(0)                               // 000000002070: BF88FF9E
	s_or_b32 vcc_lo, s3, s7                                    // 000000002074: 8C6A0703
	s_wait_alu depctr_sa_sdst(0)                               // 000000002078: BF88FF9E
	s_cbranch_vccz 48                                          // 00000000207C: BFA30030 <nearest+0x2140>
	v_cmp_gt_f32_e64 s8, v103, -v8                             // 000000002080: D4140008 42021167
	v_cmp_gt_f32_e32 vcc_lo, v104, v113                        // 000000002088: 7C28E368
	v_cmp_gt_f32_e64 s20, v103, -v9                            // 00000000208C: D4140014 42021367
	v_cmp_gt_f32_e64 s2, v104, v114                            // 000000002094: D4140002 0202E568
	v_cmp_lt_f32_e64 s4, -1.0, v66                             // 00000000209C: D4110004 020284F3
	s_or_b32 s8, s8, vcc_lo                                    // 0000000020A4: 8C086A08
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v67                         // 0000000020A8: 7C2286F3
	s_or_b32 s20, s20, s2                                      // 0000000020AC: 8C140214
	s_and_b32 s2, s8, s3                                       // 0000000020B0: 8B020308
	s_and_b32 s3, s20, s7                                      // 0000000020B4: 8B030714
	s_wait_alu depctr_sa_sdst(0)                               // 0000000020B8: BF88FF9E
	s_and_b32 s4, s4, s2                                       // 0000000020BC: 8B040204
	s_and_b32 s7, vcc_lo, s3                                   // 0000000020C0: 8B07036A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000020C4: BF88FF9E
	s_or_b32 s4, s4, s7                                        // 0000000020C8: 8C040704
	s_wait_alu depctr_sa_sdst(0)                               // 0000000020CC: BF88FF9E
	v_cndmask_b32_e64 v121, 0, 1, s4                           // 0000000020D0: D5010079 00110280
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000020D8: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v121                           // 0000000020DC: 7C9AF280
	s_cbranch_vccz 3                                           // 0000000020E0: BFA30003 <nearest+0x20f0>
	s_mov_b32 s6, 4.0                                          // 0000000020E4: BE8600F6
	s_mov_b32 s5, 1.0                                          // 0000000020E8: BE8500F2
	s_branch 20                                                // 0000000020EC: BFA00014 <nearest+0x2140>
	v_cndmask_b32_e64 v121, v74, -v8, s2                       // 0000000020F0: D5010079 400A114A
	v_cndmask_b32_e64 v122, v73, -v9, s3                       // 0000000020F8: D501007A 400E1349
	v_cndmask_b32_e64 v123, v72, v113, s2                      // 000000002100: D501007B 000AE348
	v_cndmask_b32_e64 v124, v71, v114, s3                      // 000000002108: D501007C 000EE547
	v_cndmask_b32_e64 v68, v68, v74, s2                        // 000000002110: D5010044 000A9544
	v_cndmask_b32_e64 v69, v69, v73, s3                        // 000000002118: D5010045 000E9345
	v_cndmask_b32_e64 v66, v66, v72, s2                        // 000000002120: D5010042 000A9142
	v_cndmask_b32_e64 v67, v67, v71, s3                        // 000000002128: D5010043 000E8F43
	v_dual_mov_b32 v72, v123 :: v_dual_mov_b32 v71, v124       // 000000002130: CA10017B 4846017C
	v_dual_mov_b32 v74, v121 :: v_dual_mov_b32 v73, v122       // 000000002138: CA100179 4A48017A
	s_wait_alu depctr_sa_sdst(0)                               // 000000002140: BF88FF9E
	s_cmp_lt_f32 s5, 0.5                                       // 000000002144: BF41F005
	v_cmp_ge_f32_e64 s3, v103, -v10                            // 000000002148: D4160003 42021567
	v_cmp_ge_f32_e64 s4, v103, -v11                            // 000000002150: D4160004 42021767
	s_cselect_b32 s2, -1, 0                                    // 000000002158: 980280C1
	s_cmp_le_f32 s14, 0x40a00000                               // 00000000215C: BF43FF0E 40A00000
	s_cselect_b32 s7, -1, 0                                    // 000000002164: 980780C1
	s_wait_alu depctr_sa_sdst(0)                               // 000000002168: BF88FF9E
	s_and_b32 s2, s7, s2                                       // 00000000216C: 8B020207
	s_wait_alu depctr_sa_sdst(0)                               // 000000002170: BF88FF9E
	s_and_b32 s3, s3, s2                                       // 000000002174: 8B030203
	s_and_b32 s7, s4, s2                                       // 000000002178: 8B070204
	s_wait_alu depctr_sa_sdst(0)                               // 00000000217C: BF88FF9E
	s_or_b32 vcc_lo, s3, s7                                    // 000000002180: 8C6A0703
	s_wait_alu depctr_sa_sdst(0)                               // 000000002184: BF88FF9E
	s_cbranch_vccz 49                                          // 000000002188: BFA30031 <nearest+0x2250>
	v_cmp_gt_f32_e64 s8, v103, -v10                            // 00000000218C: D4140008 42021567
	v_cmp_gt_f32_e32 vcc_lo, v104, v115                        // 000000002194: 7C28E768
	v_cmp_gt_f32_e64 s20, v103, -v11                           // 000000002198: D4140014 42021767
	v_cmp_gt_f32_e64 s2, v104, v116                            // 0000000021A0: D4140002 0202E968
	v_cmp_lt_f32_e64 s4, -1.0, v66                             // 0000000021A8: D4110004 020284F3
	s_or_b32 s8, s8, vcc_lo                                    // 0000000021B0: 8C086A08
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v67                         // 0000000021B4: 7C2286F3
	s_or_b32 s20, s20, s2                                      // 0000000021B8: 8C140214
	s_and_b32 s2, s8, s3                                       // 0000000021BC: 8B020308
	s_and_b32 s3, s20, s7                                      // 0000000021C0: 8B030714
	s_wait_alu depctr_sa_sdst(0)                               // 0000000021C4: BF88FF9E
	s_and_b32 s4, s4, s2                                       // 0000000021C8: 8B040204
	s_and_b32 s7, vcc_lo, s3                                   // 0000000021CC: 8B07036A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000021D0: BF88FF9E
	s_or_b32 s4, s4, s7                                        // 0000000021D4: 8C040704
	s_wait_alu depctr_sa_sdst(0)                               // 0000000021D8: BF88FF9E
	v_cndmask_b32_e64 v121, 0, 1, s4                           // 0000000021DC: D5010079 00110280
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000021E4: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v121                           // 0000000021E8: 7C9AF280
	s_cbranch_vccz 4                                           // 0000000021EC: BFA30004 <nearest+0x2200>
	s_mov_b32 s6, 0x40a00000                                   // 0000000021F0: BE8600FF 40A00000
	s_mov_b32 s5, 1.0                                          // 0000000021F8: BE8500F2
	s_branch 20                                                // 0000000021FC: BFA00014 <nearest+0x2250>
	v_cndmask_b32_e64 v121, v74, -v10, s2                      // 000000002200: D5010079 400A154A
	v_cndmask_b32_e64 v122, v73, -v11, s3                      // 000000002208: D501007A 400E1749
	v_cndmask_b32_e64 v123, v72, v115, s2                      // 000000002210: D501007B 000AE748
	v_cndmask_b32_e64 v124, v71, v116, s3                      // 000000002218: D501007C 000EE947
	v_cndmask_b32_e64 v68, v68, v74, s2                        // 000000002220: D5010044 000A9544
	v_cndmask_b32_e64 v69, v69, v73, s3                        // 000000002228: D5010045 000E9345
	v_cndmask_b32_e64 v66, v66, v72, s2                        // 000000002230: D5010042 000A9142
	v_cndmask_b32_e64 v67, v67, v71, s3                        // 000000002238: D5010043 000E8F43
	v_dual_mov_b32 v72, v123 :: v_dual_mov_b32 v71, v124       // 000000002240: CA10017B 4846017C
	v_dual_mov_b32 v74, v121 :: v_dual_mov_b32 v73, v122       // 000000002248: CA100179 4A48017A
	s_wait_alu depctr_sa_sdst(0)                               // 000000002250: BF88FF9E
	s_cmp_lt_f32 s5, 0.5                                       // 000000002254: BF41F005
	v_cmp_ge_f32_e64 s3, v103, -v12                            // 000000002258: D4160003 42021967
	v_cmp_ge_f32_e64 s4, v103, -v13                            // 000000002260: D4160004 42021B67
	s_cselect_b32 s2, -1, 0                                    // 000000002268: 980280C1
	s_cmp_le_f32 s14, 0x40c00000                               // 00000000226C: BF43FF0E 40C00000
	s_cselect_b32 s7, -1, 0                                    // 000000002274: 980780C1
	s_wait_alu depctr_sa_sdst(0)                               // 000000002278: BF88FF9E
	s_and_b32 s2, s7, s2                                       // 00000000227C: 8B020207
	s_wait_alu depctr_sa_sdst(0)                               // 000000002280: BF88FF9E
	s_and_b32 s3, s3, s2                                       // 000000002284: 8B030203
	s_and_b32 s7, s4, s2                                       // 000000002288: 8B070204
	s_wait_alu depctr_sa_sdst(0)                               // 00000000228C: BF88FF9E
	s_or_b32 vcc_lo, s3, s7                                    // 000000002290: 8C6A0703
	s_wait_alu depctr_sa_sdst(0)                               // 000000002294: BF88FF9E
	s_cbranch_vccz 49                                          // 000000002298: BFA30031 <nearest+0x2360>
	v_cmp_gt_f32_e64 s8, v103, -v12                            // 00000000229C: D4140008 42021967
	v_cmp_gt_f32_e32 vcc_lo, v104, v117                        // 0000000022A4: 7C28EB68
	v_cmp_gt_f32_e64 s20, v103, -v13                           // 0000000022A8: D4140014 42021B67
	v_cmp_gt_f32_e64 s2, v104, v118                            // 0000000022B0: D4140002 0202ED68
	v_cmp_lt_f32_e64 s4, -1.0, v66                             // 0000000022B8: D4110004 020284F3
	s_or_b32 s8, s8, vcc_lo                                    // 0000000022C0: 8C086A08
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v67                         // 0000000022C4: 7C2286F3
	s_or_b32 s20, s20, s2                                      // 0000000022C8: 8C140214
	s_and_b32 s2, s8, s3                                       // 0000000022CC: 8B020308
	s_and_b32 s3, s20, s7                                      // 0000000022D0: 8B030714
	s_wait_alu depctr_sa_sdst(0)                               // 0000000022D4: BF88FF9E
	s_and_b32 s4, s4, s2                                       // 0000000022D8: 8B040204
	s_and_b32 s7, vcc_lo, s3                                   // 0000000022DC: 8B07036A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000022E0: BF88FF9E
	s_or_b32 s4, s4, s7                                        // 0000000022E4: 8C040704
	s_wait_alu depctr_sa_sdst(0)                               // 0000000022E8: BF88FF9E
	v_cndmask_b32_e64 v121, 0, 1, s4                           // 0000000022EC: D5010079 00110280
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000022F4: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v121                           // 0000000022F8: 7C9AF280
	s_cbranch_vccz 4                                           // 0000000022FC: BFA30004 <nearest+0x2310>
	s_mov_b32 s6, 0x40c00000                                   // 000000002300: BE8600FF 40C00000
	s_mov_b32 s5, 1.0                                          // 000000002308: BE8500F2
	s_branch 20                                                // 00000000230C: BFA00014 <nearest+0x2360>
	v_cndmask_b32_e64 v121, v74, -v12, s2                      // 000000002310: D5010079 400A194A
	v_cndmask_b32_e64 v122, v73, -v13, s3                      // 000000002318: D501007A 400E1B49
	v_cndmask_b32_e64 v123, v72, v117, s2                      // 000000002320: D501007B 000AEB48
	v_cndmask_b32_e64 v124, v71, v118, s3                      // 000000002328: D501007C 000EED47
	v_cndmask_b32_e64 v68, v68, v74, s2                        // 000000002330: D5010044 000A9544
	v_cndmask_b32_e64 v69, v69, v73, s3                        // 000000002338: D5010045 000E9345
	v_cndmask_b32_e64 v66, v66, v72, s2                        // 000000002340: D5010042 000A9142
	v_cndmask_b32_e64 v67, v67, v71, s3                        // 000000002348: D5010043 000E8F43
	v_dual_mov_b32 v72, v123 :: v_dual_mov_b32 v71, v124       // 000000002350: CA10017B 4846017C
	v_dual_mov_b32 v74, v121 :: v_dual_mov_b32 v73, v122       // 000000002358: CA100179 4A48017A
	s_wait_alu depctr_sa_sdst(0)                               // 000000002360: BF88FF9E
	s_cmp_lt_f32 s5, 0.5                                       // 000000002364: BF41F005
	v_cmp_ge_f32_e64 s3, v103, -v14                            // 000000002368: D4160003 42021D67
	v_cmp_ge_f32_e64 s4, v103, -v15                            // 000000002370: D4160004 42021F67
	s_cselect_b32 s2, -1, 0                                    // 000000002378: 980280C1
	s_cmp_le_f32 s14, 0x40e00000                               // 00000000237C: BF43FF0E 40E00000
	s_cselect_b32 s7, -1, 0                                    // 000000002384: 980780C1
	s_wait_alu depctr_sa_sdst(0)                               // 000000002388: BF88FF9E
	s_and_b32 s2, s7, s2                                       // 00000000238C: 8B020207
	s_wait_alu depctr_sa_sdst(0)                               // 000000002390: BF88FF9E
	s_and_b32 s3, s3, s2                                       // 000000002394: 8B030203
	s_and_b32 s7, s4, s2                                       // 000000002398: 8B070204
	s_wait_alu depctr_sa_sdst(0)                               // 00000000239C: BF88FF9E
	s_or_b32 vcc_lo, s3, s7                                    // 0000000023A0: 8C6A0703
	s_wait_alu depctr_sa_sdst(0)                               // 0000000023A4: BF88FF9E
	s_cbranch_vccz 29                                          // 0000000023A8: BFA3001D <nearest+0x2420>
	v_cmp_gt_f32_e64 s8, v103, -v14                            // 0000000023AC: D4140008 42021D67
	v_cmp_gt_f32_e32 vcc_lo, v104, v119                        // 0000000023B4: 7C28EF68
	v_cmp_gt_f32_e64 s14, v103, -v15                           // 0000000023B8: D414000E 42021F67
	v_cmp_gt_f32_e64 s2, v104, v120                            // 0000000023C0: D4140002 0202F168
	v_cmp_lt_f32_e64 s4, -1.0, v66                             // 0000000023C8: D4110004 020284F3
	s_or_b32 s8, s8, vcc_lo                                    // 0000000023D0: 8C086A08
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v67                         // 0000000023D4: 7C2286F3
	s_or_b32 s14, s14, s2                                      // 0000000023D8: 8C0E020E
	s_and_b32 s2, s8, s3                                       // 0000000023DC: 8B020308
	s_wait_alu depctr_sa_sdst(0)                               // 0000000023E0: BF88FF9E
	s_and_b32 s3, s14, s7                                      // 0000000023E4: 8B03070E
	s_and_b32 s4, s4, s2                                       // 0000000023E8: 8B040204
	s_wait_alu depctr_sa_sdst(0)                               // 0000000023EC: BF88FF9E
	s_and_b32 s7, vcc_lo, s3                                   // 0000000023F0: 8B07036A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000023F4: BF88FF9E
	s_or_b32 s4, s4, s7                                        // 0000000023F8: 8C040704
	s_wait_alu depctr_sa_sdst(0)                               // 0000000023FC: BF88FF9E
	v_cndmask_b32_e64 v121, 0, 1, s4                           // 000000002400: D5010079 00110280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000002408: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v121                           // 00000000240C: 7C9AF280
	s_cbranch_vccz 854                                         // 000000002410: BFA30356 <nearest+0x316c>
	s_mov_b32 s6, 0x40e00000                                   // 000000002414: BE8600FF 40E00000
	s_mov_b32 s5, 1.0                                          // 00000000241C: BE8500F2
	s_wait_alu depctr_sa_sdst(0)                               // 000000002420: BF88FF9E
	s_cmp_ngt_f32 s5, 0.5                                      // 000000002424: BF4BF005
	s_mov_b32 s14, s6                                          // 000000002428: BE8E0006
	s_cbranch_scc1 65006                                       // 00000000242C: BFA2FDEE <nearest+0x1be8>
	v_cmp_eq_f32_e32 vcc_lo, v73, v74                          // 000000002430: 7C249549
	v_cmp_lt_f32_e64 s2, v71, v72                              // 000000002434: D4110002 02029147
	v_cmp_lt_f32_e64 s4, v73, v74                              // 00000000243C: D4110004 02029549
	v_cmp_lt_f32_e64 s3, v72, v71                              // 000000002444: D4110003 02028F48
	v_cmp_lt_f32_e64 s5, v74, v73                              // 00000000244C: D4110005 0202934A
	v_cmp_lt_f32_e64 s6, v68, v69                              // 000000002454: D4110006 02028B44
	s_and_b32 s2, vcc_lo, s2                                   // 00000000245C: 8B02026A
	s_wait_alu depctr_sa_sdst(0)                               // 000000002460: BF88FF9E
	s_or_b32 s2, s4, s2                                        // 000000002464: 8C020204
	s_and_b32 s3, vcc_lo, s3                                   // 000000002468: 8B03036A
	s_wait_alu depctr_sa_sdst(0)                               // 00000000246C: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000002470: 8D6A0002
	s_or_b32 s3, s5, s3                                        // 000000002474: 8C030305
	s_wait_alu depctr_sa_sdst(0)                               // 000000002478: BF88FF9E
	v_cndmask_b32_e32 v103, v74, v73, vcc_lo                   // 00000000247C: 02CE934A
	s_xor_b32 s2, s3, s9                                       // 000000002480: 8D020903
	v_cmp_lt_f32_e64 s3, v67, v66                              // 000000002484: D4110003 02028543
	s_wait_alu depctr_sa_sdst(0)                               // 00000000248C: BF88FF9E
	v_cndmask_b32_e64 v73, v73, v74, s2                        // 000000002490: D5010049 000A9549
	v_cndmask_b32_e32 v74, v72, v71, vcc_lo                    // 000000002498: 02948F48
	v_cndmask_b32_e64 v71, v71, v72, s2                        // 00000000249C: D5010047 000A9147
	v_mov_b32_e32 v104, v103                                   // 0000000024A4: 7ED00367
	v_cmp_eq_f32_e32 vcc_lo, v69, v68                          // 0000000024A8: 7C248945
	v_cmp_lt_f32_e64 s4, v66, v67                              // 0000000024AC: D4110004 02028742
	v_cmp_lt_f32_e64 s5, v69, v68                              // 0000000024B4: D4110005 02028945
	v_mov_b32_e32 v123, v71                                    // 0000000024BC: 7EF60347
	v_permlanex16_b32 v104, v104, s10, 0xfedcba98              // 0000000024C0: D65C0068 03FC1568 FEDCBA98
	v_mov_b32_e32 v122, v74                                    // 0000000024CC: 7EF4034A
	s_and_b32 s2, vcc_lo, s3                                   // 0000000024D0: 8B02036A
	s_and_b32 s3, vcc_lo, s4                                   // 0000000024D4: 8B03046A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000024D8: BF88FF9E
	s_or_b32 s2, s5, s2                                        // 0000000024DC: 8C020205
	s_or_b32 s3, s6, s3                                        // 0000000024E0: 8C030306
	v_permlanex16_b32 v122, v122, s10, 0xfedcba98              // 0000000024E4: D65C007A 03FC157A FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 0000000024F0: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 0000000024F4: 8D6A0002
	s_xor_b32 s2, s3, s9                                       // 0000000024F8: 8D020903
	s_wait_alu depctr_sa_sdst(0)                               // 0000000024FC: BF88FF9E
	v_dual_cndmask_b32 v72, v68, v69 :: v_dual_mov_b32 v121, v73// 000000002500: CA508B44 48780149
	v_cndmask_b32_e64 v68, v69, v68, s2                        // 000000002508: D5010044 000A8945
	v_cndmask_b32_e32 v69, v66, v67, vcc_lo                    // 000000002510: 028A8742
	v_cmp_eq_f32_e32 vcc_lo, v103, v104                        // 000000002514: 7C24D167
	v_cmp_gt_f32_e64 s4, v74, v122                             // 000000002518: D4140004 0202F54A
	v_cmp_gt_f32_e64 s6, v103, v104                            // 000000002520: D4140006 0202D167
	v_cndmask_b32_e64 v66, v67, v66, s2                        // 000000002528: D5010042 000A8543
	v_permlanex16_b32 v121, v121, s10, 0xfedcba98              // 000000002530: D65C0079 03FC1579 FEDCBA98
	v_permlanex16_b32 v123, v123, s10, 0xfedcba98              // 00000000253C: D65C007B 03FC157B FEDCBA98
	s_and_b32 s2, vcc_lo, s4                                   // 000000002548: 8B02046A
	s_wait_alu depctr_sa_sdst(0)                               // 00000000254C: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 000000002550: 8C020206
	v_cmp_eq_f32_e64 s3, v73, v121                             // 000000002554: D4120003 0202F349
	s_wait_alu depctr_sa_sdst(0)                               // 00000000255C: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000002560: 8D6A0002
	v_cmp_gt_f32_e64 s5, v71, v123                             // 000000002564: D4140005 0202F747
	s_wait_alu depctr_sa_sdst(0)                               // 00000000256C: BF88FF9E
	v_dual_cndmask_b32 v67, v103, v104 :: v_dual_mov_b32 v104, v68// 000000002570: CA50D167 43680144
	v_dual_mov_b32 v103, v72 :: v_dual_mov_b32 v124, v69       // 000000002578: CA100148 677C0145
	v_cmp_gt_f32_e64 s7, v73, v121                             // 000000002580: D4140007 0202F349
	s_and_b32 s3, s3, s5                                       // 000000002588: 8B030503
	v_permlanex16_b32 v104, v104, s10, 0xfedcba98              // 00000000258C: D65C0068 03FC1568 FEDCBA98
	v_permlanex16_b32 v103, v103, s10, 0xfedcba98              // 000000002598: D65C0067 03FC1567 FEDCBA98
	v_permlanex16_b32 v124, v124, s10, 0xfedcba98              // 0000000025A4: D65C007C 03FC157C FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 0000000025B0: BF88FF9E
	s_or_b32 s3, s7, s3                                        // 0000000025B4: 8C030307
	v_cndmask_b32_e32 v74, v74, v122, vcc_lo                   // 0000000025B8: 0294F54A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000025BC: BF88FF9E
	s_xor_b32 s2, s3, s0                                       // 0000000025C0: 8D020003
	v_cmp_neq_f32_e32 vcc_lo, v72, v103                        // 0000000025C4: 7C3ACF48
	v_cmp_ngt_f32_e64 s3, v69, v124                            // 0000000025C8: D41B0003 0202F945
	v_cmp_ngt_f32_e64 s6, v72, v103                            // 0000000025D0: D41B0006 0202CF48
	s_wait_alu depctr_sa_sdst(0)                               // 0000000025D8: BF88FF9E
	v_cndmask_b32_e64 v73, v73, v121, s2                       // 0000000025DC: D5010049 000AF349
	v_cndmask_b32_e64 v71, v71, v123, s2                       // 0000000025E4: D5010047 000AF747
	v_cmp_neq_f32_e64 s4, v68, v104                            // 0000000025EC: D41D0004 0202D144
	s_or_b32 s2, vcc_lo, s3                                    // 0000000025F4: 8C02036A
	v_cmp_ngt_f32_e64 s7, v68, v104                            // 0000000025F8: D41B0007 0202D144
	s_wait_alu depctr_sa_sdst(0)                               // 000000002600: BF88FF9E
	s_and_b32 s2, s6, s2                                       // 000000002604: 8B020206
	s_wait_alu depctr_sa_sdst(0)                               // 000000002608: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 00000000260C: 8D6A0002
	s_wait_alu depctr_sa_sdst(0)                               // 000000002610: BF88FF9E
	v_dual_mov_b32 v125, v66 :: v_dual_cndmask_b32 v72, v72, v103// 000000002614: CA120142 7D48CF48
	v_cndmask_b32_e32 v69, v69, v124, vcc_lo                   // 00000000261C: 028AF945
	v_cmp_eq_f32_e32 vcc_lo, v73, v67                          // 000000002620: 7C248749
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000002624: BF870093
	v_permlanex16_b32 v125, v125, s10, 0xfedcba98              // 000000002628: D65C007D 03FC157D FEDCBA98
	v_cmp_ngt_f32_e64 s5, v66, v125                            // 000000002634: D41B0005 0202FB42
	s_or_b32 s3, s4, s5                                        // 00000000263C: 8C030504
	v_cmp_lt_f32_e64 s4, v74, v71                              // 000000002640: D4110004 02028F4A
	s_wait_alu depctr_sa_sdst(0)                               // 000000002648: BF88FF9E
	s_and_b32 s3, s7, s3                                       // 00000000264C: 8B030307
	v_cmp_lt_f32_e64 s5, v73, v67                              // 000000002650: D4110005 02028749
	s_wait_alu depctr_sa_sdst(0)                               // 000000002658: BF88FF9E
	s_xor_b32 s2, s3, s0                                       // 00000000265C: 8D020003
	v_cmp_lt_f32_e64 s3, v71, v74                              // 000000002660: D4110003 02029547
	s_wait_alu depctr_sa_sdst(0)                               // 000000002668: BF88FF9E
	v_cndmask_b32_e64 v68, v68, v104, s2                       // 00000000266C: D5010044 000AD144
	v_cndmask_b32_e64 v66, v66, v125, s2                       // 000000002674: D5010042 000AFB42
	v_cmp_lt_f32_e64 s2, v67, v73                              // 00000000267C: D4110002 02029343
	s_and_b32 s4, vcc_lo, s4                                   // 000000002684: 8B04046A
	s_and_b32 s3, vcc_lo, s3                                   // 000000002688: 8B03036A
	s_wait_alu depctr_sa_sdst(0)                               // 00000000268C: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 000000002690: 8C6A0305
	s_or_b32 s2, s2, s4                                        // 000000002694: 8C020402
	s_wait_alu depctr_sa_sdst(0)                               // 000000002698: BF88FF9E
	v_cndmask_b32_e32 v103, v67, v73, vcc_lo                   // 00000000269C: 02CE9343
	v_cndmask_b32_e64 v67, v67, v73, s2                        // 0000000026A0: D5010043 000A9343
	v_cndmask_b32_e32 v73, v74, v71, vcc_lo                    // 0000000026A8: 02928F4A
	v_cmp_eq_f32_e32 vcc_lo, v68, v72                          // 0000000026AC: 7C249144
	v_cmp_lt_f32_e64 s3, v66, v69                              // 0000000026B0: D4110003 02028B42
	v_cmp_lt_f32_e64 s4, v69, v66                              // 0000000026B8: D4110004 02028545
	v_cmp_lt_f32_e64 s5, v68, v72                              // 0000000026C0: D4110005 02029144
	v_cndmask_b32_e64 v71, v74, v71, s2                        // 0000000026C8: D5010047 000A8F4A
	v_cmp_lt_f32_e64 s2, v72, v68                              // 0000000026D0: D4110002 02028948
	s_and_b32 s3, vcc_lo, s3                                   // 0000000026D8: 8B03036A
	s_and_b32 s4, vcc_lo, s4                                   // 0000000026DC: 8B04046A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000026E0: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 0000000026E4: 8C6A0305
	s_or_b32 s2, s2, s4                                        // 0000000026E8: 8C020402
	s_wait_alu depctr_sa_sdst(0)                               // 0000000026EC: BF88FF9E
	v_cndmask_b32_e32 v74, v68, v72, vcc_lo                    // 0000000026F0: 02949144
	v_cndmask_b32_e64 v68, v68, v72, s2                        // 0000000026F4: D5010044 000A9144
	v_cndmask_b32_e32 v72, v66, v69, vcc_lo                    // 0000000026FC: 02908B42
	v_cndmask_b32_e64 v66, v66, v69, s2                        // 000000002700: D5010042 000A8B42
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 000000002708: BF870224
	v_cmp_eq_f32_e32 vcc_lo, v103, v74                         // 00000000270C: 7C249567
	v_cmp_lt_f32_e64 s5, v103, v74                             // 000000002710: D4110005 02029567
	v_cmp_lt_f32_e64 s3, v73, v72                              // 000000002718: D4110003 02029149
	v_cmp_eq_f32_e64 s2, v67, v68                              // 000000002720: D4120002 02028943
	v_cmp_lt_f32_e64 s4, v71, v66                              // 000000002728: D4110004 02028547
	v_cmp_lt_f32_e64 s6, v67, v68                              // 000000002730: D4110006 02028943
	s_and_b32 s3, vcc_lo, s3                                   // 000000002738: 8B03036A
	s_wait_alu depctr_sa_sdst(0)                               // 00000000273C: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 000000002740: 8C6A0305
	s_and_b32 s2, s2, s4                                       // 000000002744: 8B020402
	s_wait_alu depctr_sa_sdst(0)                               // 000000002748: BF88FF9E
	v_cndmask_b32_e32 v69, v74, v103, vcc_lo                   // 00000000274C: 028ACF4A
	s_or_b32 s2, s6, s2                                        // 000000002750: 8C020206
	v_cndmask_b32_e32 v74, v103, v74, vcc_lo                   // 000000002754: 02949567
	s_wait_alu depctr_sa_sdst(0)                               // 000000002758: BF88FF9E
	v_cndmask_b32_e64 v104, v68, v67, s2                       // 00000000275C: D5010068 000A8744
	v_cndmask_b32_e64 v67, v67, v68, s2                        // 000000002764: D5010043 000A8943
	v_mov_b32_e32 v103, v69                                    // 00000000276C: 7ECE0345
	v_dual_cndmask_b32 v121, v72, v73 :: v_dual_cndmask_b32 v68, v73, v72// 000000002770: CA529348 79449149
	v_cndmask_b32_e64 v122, v66, v71, s2                       // 000000002778: D501007A 000A8F42
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000002780: BF870194
	v_dual_mov_b32 v123, v104 :: v_dual_mov_b32 v72, v67       // 000000002784: CA100168 7B480143
	v_dual_mov_b32 v124, v121 :: v_dual_mov_b32 v73, v68       // 00000000278C: CA100179 7C480144
	v_permlanex16_b32 v103, v103, s10, 0xfedcba98              // 000000002794: D65C0067 03FC1567 FEDCBA98
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 0000000027A0: BF870223
	v_permlanex16_b32 v123, v123, s10, 0xfedcba98              // 0000000027A4: D65C007B 03FC157B FEDCBA98
	v_cndmask_b32_e64 v66, v71, v66, s2                        // 0000000027B0: D5010042 000A8547
	v_permlanex16_b32 v124, v124, s10, 0xfedcba98              // 0000000027B8: D65C007C 03FC157C FEDCBA98
	v_permlanex16_b32 v73, v73, s10, 0xfedcba98                // 0000000027C4: D65C0049 03FC1549 FEDCBA98
	v_mov_b32_e32 v125, v122                                   // 0000000027D0: 7EFA037A
	v_cmp_eq_f32_e32 vcc_lo, v69, v103                         // 0000000027D4: 7C24CF45
	v_cmp_eq_f32_e64 s3, v104, v123                            // 0000000027D8: D4120003 0202F768
	v_cmp_gt_f32_e64 s4, v121, v124                            // 0000000027E0: D4140004 0202F979
	v_cmp_gt_f32_e64 s6, v69, v103                             // 0000000027E8: D4140006 0202CF45
	v_permlanex16_b32 v125, v125, s10, 0xfedcba98              // 0000000027F0: D65C007D 03FC157D FEDCBA98
	v_cmp_gt_f32_e64 s7, v104, v123                            // 0000000027FC: D4140007 0202F768
	v_permlanex16_b32 v72, v72, s10, 0xfedcba98                // 000000002804: D65C0048 03FC1548 FEDCBA98
	s_and_b32 s2, vcc_lo, s4                                   // 000000002810: 8B02046A
	v_cmp_gt_f32_e64 s4, v68, v73                              // 000000002814: D4140004 02029344
	v_cmp_gt_f32_e64 s5, v122, v125                            // 00000000281C: D4140005 0202FB7A
	s_wait_alu depctr_sa_sdst(0)                               // 000000002824: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 000000002828: 8C020206
	s_wait_alu depctr_sa_sdst(0)                               // 00000000282C: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000002830: 8D6A0002
	s_and_b32 s3, s3, s5                                       // 000000002834: 8B030503
	s_wait_alu depctr_sa_sdst(0)                               // 000000002838: BF88FF9E
	v_cndmask_b32_e32 v69, v69, v103, vcc_lo                   // 00000000283C: 028ACF45
	s_or_b32 s3, s7, s3                                        // 000000002840: 8C030307
	v_mov_b32_e32 v103, v66                                    // 000000002844: 7ECE0342
	s_wait_alu depctr_sa_sdst(0)                               // 000000002848: BF88FF9E
	s_xor_b32 s2, s3, s0                                       // 00000000284C: 8D020003
	v_cndmask_b32_e32 v121, v121, v124, vcc_lo                 // 000000002850: 02F2F979
	s_wait_alu depctr_sa_sdst(0)                               // 000000002854: BF88FF9E
	v_cndmask_b32_e64 v104, v104, v123, s2                     // 000000002858: D5010068 000AF768
	v_mov_b32_e32 v71, v74                                     // 000000002860: 7E8E034A
	v_permlanex16_b32 v103, v103, s10, 0xfedcba98              // 000000002864: D65C0067 03FC1567 FEDCBA98
	v_cmp_eq_f32_e64 s3, v67, v72                              // 000000002870: D4120003 02029143
	v_cmp_gt_f32_e64 s7, v67, v72                              // 000000002878: D4140007 02029143
	v_cndmask_b32_e64 v122, v122, v125, s2                     // 000000002880: D501007A 000AFB7A
	v_permlanex16_b32 v71, v71, s10, 0xfedcba98                // 000000002888: D65C0047 03FC1547 FEDCBA98
	v_cmp_gt_f32_e64 s5, v66, v103                             // 000000002894: D4140005 0202CF42
	s_delay_alu instid0(VALU_DEP_2)                            // 00000000289C: BF870002
	v_cmp_eq_f32_e32 vcc_lo, v74, v71                          // 0000000028A0: 7C248F4A
	v_cmp_gt_f32_e64 s6, v74, v71                              // 0000000028A4: D4140006 02028F4A
	s_and_b32 s3, s3, s5                                       // 0000000028AC: 8B030503
	v_cmp_lt_f32_e64 s5, v104, v69                             // 0000000028B0: D4110005 02028B68
	s_wait_alu depctr_sa_sdst(0)                               // 0000000028B8: BF88FF9E
	s_or_b32 s3, s7, s3                                        // 0000000028BC: 8C030307
	s_and_b32 s2, vcc_lo, s4                                   // 0000000028C0: 8B02046A
	v_cmp_lt_f32_e64 s4, v121, v122                            // 0000000028C4: D4110004 0202F579
	s_wait_alu depctr_sa_sdst(0)                               // 0000000028CC: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 0000000028D0: 8C020206
	s_wait_alu depctr_sa_sdst(0)                               // 0000000028D4: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 0000000028D8: 8D6A0002
	s_xor_b32 s2, s3, s0                                       // 0000000028DC: 8D020003
	s_wait_alu depctr_sa_sdst(0)                               // 0000000028E0: BF88FF9E
	v_dual_cndmask_b32 v71, v74, v71 :: v_dual_cndmask_b32 v68, v68, v73// 0000000028E4: CA528F4A 47449344
	v_cmp_eq_f32_e32 vcc_lo, v104, v69                         // 0000000028EC: 7C248B68
	v_cmp_lt_f32_e64 s3, v122, v121                            // 0000000028F0: D4110003 0202F37A
	v_cndmask_b32_e64 v67, v67, v72, s2                        // 0000000028F8: D5010043 000A9143
	v_cndmask_b32_e64 v66, v66, v103, s2                       // 000000002900: D5010042 000ACF42
	v_cmp_lt_f32_e64 s2, v69, v104                             // 000000002908: D4110002 0202D145
	s_and_b32 s4, vcc_lo, s4                                   // 000000002910: 8B04046A
	s_and_b32 s3, vcc_lo, s3                                   // 000000002914: 8B03036A
	s_wait_alu depctr_sa_sdst(0)                               // 000000002918: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 00000000291C: 8C6A0305
	s_or_b32 s2, s2, s4                                        // 000000002920: 8C020402
	s_wait_alu depctr_sa_sdst(0)                               // 000000002924: BF88FF9E
	v_cndmask_b32_e32 v72, v69, v104, vcc_lo                   // 000000002928: 0290D145
	v_cndmask_b32_e32 v73, v121, v122, vcc_lo                  // 00000000292C: 0292F579
	v_cmp_eq_f32_e32 vcc_lo, v67, v71                          // 000000002930: 7C248F43
	v_cmp_lt_f32_e64 s3, v66, v68                              // 000000002934: D4110003 02028942
	v_cmp_lt_f32_e64 s4, v68, v66                              // 00000000293C: D4110004 02028544
	v_cndmask_b32_e64 v69, v69, v104, s2                       // 000000002944: D5010045 000AD145
	v_cmp_lt_f32_e64 s5, v67, v71                              // 00000000294C: D4110005 02028F43
	v_cndmask_b32_e64 v74, v121, v122, s2                      // 000000002954: D501004A 000AF579
	v_cmp_lt_f32_e64 s2, v71, v67                              // 00000000295C: D4110002 02028747
	s_and_b32 s3, vcc_lo, s3                                   // 000000002964: 8B03036A
	s_and_b32 s4, vcc_lo, s4                                   // 000000002968: 8B04046A
	s_wait_alu depctr_sa_sdst(0)                               // 00000000296C: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 000000002970: 8C6A0305
	v_permlanex16_b32 v69, v69, s10, 0xfedcba98                // 000000002974: D65C0045 03FC1545 FEDCBA98
	s_or_b32 s2, s2, s4                                        // 000000002980: 8C020402
	s_wait_alu depctr_sa_sdst(0)                               // 000000002984: BF88FF9E
	v_cndmask_b32_e32 v103, v71, v67, vcc_lo                   // 000000002988: 02CE8747
	v_cndmask_b32_e64 v67, v71, v67, s2                        // 00000000298C: D5010043 000A8747
	v_permlanex16_b32 v72, v72, s10, 0xfedcba98                // 000000002994: D65C0048 03FC1548 FEDCBA98
	v_permlanex16_b32 v74, v74, s10, 0xfedcba98                // 0000000029A0: D65C004A 03FC154A FEDCBA98
	v_permlanex16_b32 v73, v73, s10, 0xfedcba98                // 0000000029AC: D65C0049 03FC1549 FEDCBA98
	v_cndmask_b32_e32 v71, v68, v66, vcc_lo                    // 0000000029B8: 028E8544
	v_cndmask_b32_e64 v66, v68, v66, s2                        // 0000000029BC: D5010042 000A8544
	v_cmp_eq_f32_e32 vcc_lo, v65, v69                          // 0000000029C4: 7C248B41
	v_cmp_eq_f32_e64 s3, v59, v72                              // 0000000029C8: D4120003 0202913B
	v_cmp_lt_f32_e64 s4, v64, v74                              // 0000000029D0: D4110004 02029540
	v_cmp_lt_f32_e64 s5, v58, v73                              // 0000000029D8: D4110005 0202933A
	v_permlanex16_b32 v67, v67, s10, 0xfedcba98                // 0000000029E0: D65C0043 03FC1543 FEDCBA98
	v_permlanex16_b32 v66, v66, s10, 0xfedcba98                // 0000000029EC: D65C0042 03FC1542 FEDCBA98
	v_cmp_lt_f32_e64 s2, v65, v69                              // 0000000029F8: D4110002 02028B41
	v_permlanex16_b32 v103, v103, s10, 0xfedcba98              // 000000002A00: D65C0067 03FC1567 FEDCBA98
	v_permlanex16_b32 v71, v71, s10, 0xfedcba98                // 000000002A0C: D65C0047 03FC1547 FEDCBA98
	s_and_b32 s4, vcc_lo, s4                                   // 000000002A18: 8B04046A
	s_and_b32 s8, s3, s5                                       // 000000002A1C: 8B080503
	v_cmp_eq_f32_e64 s3, v60, v67                              // 000000002A20: D4120003 0202873C
	v_cmp_lt_f32_e64 s5, v61, v66                              // 000000002A28: D4110005 0202853D
	v_cmp_lt_f32_e64 s6, v59, v72                              // 000000002A30: D4110006 0202913B
	s_wait_alu depctr_sa_sdst(0)                               // 000000002A38: BF88FF9E
	s_or_b32 vcc_lo, s2, s4                                    // 000000002A3C: 8C6A0402
	v_cmp_lt_f32_e64 s2, v60, v67                              // 000000002A40: D4110002 0202873C
	v_cmp_eq_f32_e64 s4, v62, v103                             // 000000002A48: D4120004 0202CF3E
	v_cmp_lt_f32_e64 s7, v63, v71                              // 000000002A50: D4110007 02028F3F
	s_and_b32 s3, s3, s5                                       // 000000002A58: 8B030503
	s_or_b32 s6, s6, s8                                        // 000000002A5C: 8C060806
	v_cmp_lt_f32_e64 s8, v62, v103                             // 000000002A60: D4110008 0202CF3E
	s_wait_alu depctr_sa_sdst(0)                               // 000000002A68: BF88FF9E
	s_or_b32 s2, s2, s3                                        // 000000002A6C: 8C020302
	s_and_b32 s4, s4, s7                                       // 000000002A70: 8B040704
	s_wait_alu depctr_sa_sdst(0)                               // 000000002A74: BF88FF9E
	v_cndmask_b32_e64 v60, v67, v60, s2                        // 000000002A78: D501003C 000A7943
	v_cndmask_b32_e64 v61, v66, v61, s2                        // 000000002A80: D501003D 000A7B42
	s_or_b32 s3, s8, s4                                        // 000000002A88: 8C030408
	v_cndmask_b32_e32 v65, v69, v65, vcc_lo                    // 000000002A8C: 02828345
	s_wait_alu depctr_sa_sdst(0)                               // 000000002A90: BF88FF9E
	v_cndmask_b32_e64 v62, v103, v62, s3                       // 000000002A94: D501003E 000E7D67
	v_cndmask_b32_e64 v63, v71, v63, s3                        // 000000002A9C: D501003F 000E7F47
	v_cndmask_b32_e32 v64, v74, v64, vcc_lo                    // 000000002AA4: 0280814A
	v_cmp_eq_f32_e32 vcc_lo, v52, v60                          // 000000002AA8: 7C247934
	v_cmp_lt_f32_e64 s3, v51, v61                              // 000000002AAC: D4110003 02027B33
	v_cmp_lt_f32_e64 s5, v52, v60                              // 000000002AB4: D4110005 02027934
	v_cmp_eq_f32_e64 s2, v53, v62                              // 000000002ABC: D4120002 02027D35
	v_cmp_lt_f32_e64 s4, v50, v63                              // 000000002AC4: D4110004 02027F32
	v_cndmask_b32_e64 v59, v72, v59, s6                        // 000000002ACC: D501003B 001A7748
	s_and_b32 s3, vcc_lo, s3                                   // 000000002AD4: 8B03036A
	v_cndmask_b32_e64 v58, v73, v58, s6                        // 000000002AD8: D501003A 001A7549
	v_cmp_lt_f32_e64 s6, v53, v62                              // 000000002AE0: D4110006 02027D35
	s_wait_alu depctr_sa_sdst(0)                               // 000000002AE8: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 000000002AEC: 8C6A0305
	s_and_b32 s2, s2, s4                                       // 000000002AF0: 8B020402
	s_wait_alu depctr_sa_sdst(0)                               // 000000002AF4: BF88FF9E
	v_cndmask_b32_e32 v66, v60, v52, vcc_lo                    // 000000002AF8: 0284693C
	v_cndmask_b32_e32 v68, v61, v51, vcc_lo                    // 000000002AFC: 0288673D
	v_dual_cndmask_b32 v52, v52, v60 :: v_dual_cndmask_b32 v51, v51, v61// 000000002B00: CA527934 34327B33
	v_cmp_eq_f32_e32 vcc_lo, v55, v65                          // 000000002B08: 7C248337
	v_cmp_lt_f32_e64 s4, v56, v64                              // 000000002B0C: D4110004 02028138
	s_or_b32 s2, s6, s2                                        // 000000002B14: 8C020206
	v_cmp_lt_f32_e64 s6, v55, v65                              // 000000002B18: D4110006 02028337
	s_wait_alu depctr_sa_sdst(0)                               // 000000002B20: BF88FF9E
	v_cndmask_b32_e64 v67, v62, v53, s2                        // 000000002B24: D5010043 000A6B3E
	v_cndmask_b32_e64 v53, v53, v62, s2                        // 000000002B2C: D5010035 000A7D35
	s_and_b32 s4, vcc_lo, s4                                   // 000000002B34: 8B04046A
	v_cmp_eq_f32_e64 s3, v57, v59                              // 000000002B38: D4120003 02027739
	s_wait_alu depctr_sa_sdst(0)                               // 000000002B40: BF88FF9E
	s_or_b32 vcc_lo, s6, s4                                    // 000000002B44: 8C6A0406
	v_cmp_lt_f32_e64 s5, v54, v58                              // 000000002B48: D4110005 02027536
	s_wait_alu depctr_sa_sdst(0)                               // 000000002B50: BF88FF9E
	v_cndmask_b32_e32 v60, v65, v55, vcc_lo                    // 000000002B54: 02786F41
	v_cndmask_b32_e32 v62, v64, v56, vcc_lo                    // 000000002B58: 027C7140
	v_cndmask_b32_e64 v69, v63, v50, s2                        // 000000002B5C: D5010045 000A653F
	v_cndmask_b32_e64 v50, v50, v63, s2                        // 000000002B64: D5010032 000A7F32
	v_cmp_lt_f32_e64 s2, v57, v59                              // 000000002B6C: D4110002 02027739
	s_and_b32 s3, s3, s5                                       // 000000002B74: 8B030503
	v_dual_cndmask_b32 v55, v55, v65 :: v_dual_cndmask_b32 v56, v56, v64// 000000002B78: CA528337 37388138
	v_cmp_eq_f32_e32 vcc_lo, v66, v60                          // 000000002B80: 7C247942
	v_cmp_lt_f32_e64 s4, v68, v62                              // 000000002B84: D4110004 02027D44
	s_wait_alu depctr_sa_sdst(0)                               // 000000002B8C: BF88FF9E
	s_or_b32 s2, s2, s3                                        // 000000002B90: 8C020302
	v_cmp_lt_f32_e64 s6, v66, v60                              // 000000002B94: D4110006 02027942
	s_wait_alu depctr_sa_sdst(0)                               // 000000002B9C: BF88FF9E
	v_cndmask_b32_e64 v61, v59, v57, s2                        // 000000002BA0: D501003D 000A733B
	v_cndmask_b32_e64 v63, v58, v54, s2                        // 000000002BA8: D501003F 000A6D3A
	s_and_b32 s4, vcc_lo, s4                                   // 000000002BB0: 8B04046A
	v_cndmask_b32_e64 v54, v54, v58, s2                        // 000000002BB4: D5010036 000A7536
	s_wait_alu depctr_sa_sdst(0)                               // 000000002BBC: BF88FF9E
	s_or_b32 vcc_lo, s6, s4                                    // 000000002BC0: 8C6A0406
	v_cmp_eq_f32_e64 s3, v67, v61                              // 000000002BC4: D4120003 02027B43
	v_cmp_lt_f32_e64 s5, v69, v63                              // 000000002BCC: D4110005 02027F45
	s_wait_alu depctr_sa_sdst(0)                               // 000000002BD4: BF88FF9E
	v_cndmask_b32_e32 v58, v60, v66, vcc_lo                    // 000000002BD8: 0274853C
	v_cndmask_b32_e32 v64, v62, v68, vcc_lo                    // 000000002BDC: 0280893E
	v_cndmask_b32_e32 v60, v66, v60, vcc_lo                    // 000000002BE0: 02787942
	v_cndmask_b32_e32 v62, v68, v62, vcc_lo                    // 000000002BE4: 027C7D44
	v_cmp_eq_f32_e32 vcc_lo, v52, v55                          // 000000002BE8: 7C246F34
	v_cmp_lt_f32_e64 s4, v51, v56                              // 000000002BEC: D4110004 02027133
	v_cndmask_b32_e64 v57, v57, v59, s2                        // 000000002BF4: D5010039 000A7739
	v_cmp_lt_f32_e64 s2, v67, v61                              // 000000002BFC: D4110002 02027B43
	v_cmp_lt_f32_e64 s6, v52, v55                              // 000000002C04: D4110006 02026F34
	s_and_b32 s3, s3, s5                                       // 000000002C0C: 8B030503
	s_and_b32 s4, vcc_lo, s4                                   // 000000002C10: 8B04046A
	v_cmp_lt_f32_e64 s5, v50, v54                              // 000000002C14: D4110005 02026D32
	s_wait_alu depctr_sa_sdst(0)                               // 000000002C1C: BF88FF9E
	s_or_b32 s2, s2, s3                                        // 000000002C20: 8C020302
	s_or_b32 vcc_lo, s6, s4                                    // 000000002C24: 8C6A0406
	s_wait_alu depctr_sa_sdst(0)                               // 000000002C28: BF88FF9E
	v_cndmask_b32_e64 v65, v63, v69, s2                        // 000000002C2C: D5010041 000A8B3F
	v_cndmask_b32_e32 v66, v55, v52, vcc_lo                    // 000000002C34: 02846937
	v_dual_cndmask_b32 v52, v52, v55 :: v_dual_mov_b32 v55, v58// 000000002C38: CA506F34 3436013A
	v_mov_b32_e32 v72, v64                                     // 000000002C40: 7E900340
	v_cndmask_b32_e64 v59, v61, v67, s2                        // 000000002C44: D501003B 000A873D
	v_dual_cndmask_b32 v68, v56, v51 :: v_dual_cndmask_b32 v51, v51, v56// 000000002C4C: CA526738 44327133
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000002C54: BF870214
	v_permlanex16_b32 v55, v55, s10, 0xfedcba98                // 000000002C58: D65C0037 03FC1537 FEDCBA98
	v_permlanex16_b32 v72, v72, s10, 0xfedcba98                // 000000002C64: D65C0048 03FC1548 FEDCBA98
	v_cmp_eq_f32_e64 s3, v53, v57                              // 000000002C70: D4120003 02027335
	v_cndmask_b32_e64 v61, v67, v61, s2                        // 000000002C78: D501003D 000A7B43
	v_cndmask_b32_e64 v63, v69, v63, s2                        // 000000002C80: D501003F 000A7F45
	v_cmp_lt_f32_e64 s2, v53, v57                              // 000000002C88: D4110002 02027335
	v_cmp_eq_f32_e32 vcc_lo, v58, v55                          // 000000002C90: 7C246F3A
	v_cmp_gt_f32_e64 s4, v64, v72                              // 000000002C94: D4140004 02029140
	s_and_b32 s3, s3, s5                                       // 000000002C9C: 8B030503
	v_cmp_gt_f32_e64 s6, v58, v55                              // 000000002CA0: D4140006 02026F3A
	s_wait_alu depctr_sa_sdst(0)                               // 000000002CA8: BF88FF9E
	s_or_b32 s2, s2, s3                                        // 000000002CAC: 8C020302
	v_mov_b32_e32 v71, v59                                     // 000000002CB0: 7E8E033B
	s_wait_alu depctr_sa_sdst(0)                               // 000000002CB4: BF88FF9E
	v_cndmask_b32_e64 v67, v57, v53, s2                        // 000000002CB8: D5010043 000A6B39
	v_cndmask_b32_e64 v69, v54, v50, s2                        // 000000002CC0: D5010045 000A6536
	v_mov_b32_e32 v73, v65                                     // 000000002CC8: 7E920341
	v_cndmask_b32_e64 v53, v53, v57, s2                        // 000000002CCC: D5010035 000A7335
	v_cndmask_b32_e64 v50, v50, v54, s2                        // 000000002CD4: D5010032 000A6D32
	s_and_b32 s2, vcc_lo, s4                                   // 000000002CDC: 8B02046A
	v_permlanex16_b32 v71, v71, s10, 0xfedcba98                // 000000002CE0: D65C0047 03FC1547 FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 000000002CEC: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 000000002CF0: 8C020206
	v_permlanex16_b32 v73, v73, s10, 0xfedcba98                // 000000002CF4: D65C0049 03FC1549 FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 000000002D00: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000002D04: 8D6A0002
	v_dual_mov_b32 v56, v61 :: v_dual_mov_b32 v57, v62         // 000000002D08: CA10013D 3838013E
	s_wait_alu depctr_sa_sdst(0)                               // 000000002D10: BF88FF9E
	v_dual_cndmask_b32 v54, v58, v55 :: v_dual_mov_b32 v55, v60// 000000002D14: CA506F3A 3636013C
	v_cmp_eq_f32_e64 s3, v59, v71                              // 000000002D1C: D4120003 02028F3B
	v_cmp_gt_f32_e64 s5, v65, v73                              // 000000002D24: D4140005 02029341
	v_permlanex16_b32 v57, v57, s10, 0xfedcba98                // 000000002D2C: D65C0039 03FC1539 FEDCBA98
	v_cmp_gt_f32_e64 s7, v59, v71                              // 000000002D38: D4140007 02028F3B
	v_permlanex16_b32 v55, v55, s10, 0xfedcba98                // 000000002D40: D65C0037 03FC1537 FEDCBA98
	v_cndmask_b32_e32 v64, v64, v72, vcc_lo                    // 000000002D4C: 02809140
	s_and_b32 s3, s3, s5                                       // 000000002D50: 8B030503
	v_cmp_gt_f32_e64 s4, v62, v57                              // 000000002D54: D4140004 0202733E
	s_wait_alu depctr_sa_sdst(0)                               // 000000002D5C: BF88FF9E
	s_or_b32 s3, s7, s3                                        // 000000002D60: 8C030307
	v_cmp_eq_f32_e32 vcc_lo, v60, v55                          // 000000002D64: 7C246F3C
	v_cmp_gt_f32_e64 s6, v60, v55                              // 000000002D68: D4140006 02026F3C
	s_wait_alu depctr_sa_sdst(0)                               // 000000002D70: BF88FF9E
	s_xor_b32 s2, s3, s0                                       // 000000002D74: 8D020003
	v_mov_b32_e32 v58, v63                                     // 000000002D78: 7E74033F
	s_wait_alu depctr_sa_sdst(0)                               // 000000002D7C: BF88FF9E
	v_cndmask_b32_e64 v59, v59, v71, s2                        // 000000002D80: D501003B 000A8F3B
	v_cndmask_b32_e64 v65, v65, v73, s2                        // 000000002D88: D5010041 000A9341
	s_and_b32 s2, vcc_lo, s4                                   // 000000002D90: 8B02046A
	v_permlanex16_b32 v56, v56, s10, 0xfedcba98                // 000000002D94: D65C0038 03FC1538 FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 000000002DA0: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 000000002DA4: 8C020206
	v_mov_b32_e32 v71, v67                                     // 000000002DA8: 7E8E0343
	s_wait_alu depctr_sa_sdst(0)                               // 000000002DAC: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000002DB0: 8D6A0002
	v_mov_b32_e32 v72, v68                                     // 000000002DB4: 7E900344
	s_wait_alu depctr_sa_sdst(0)                               // 000000002DB8: BF88FF9E
	v_dual_cndmask_b32 v60, v60, v55 :: v_dual_mov_b32 v73, v69// 000000002DBC: CA506F3C 3C480145
	v_mov_b32_e32 v55, v66                                     // 000000002DC4: 7E6E0342
	v_permlanex16_b32 v58, v58, s10, 0xfedcba98                // 000000002DC8: D65C003A 03FC153A FEDCBA98
	v_cmp_eq_f32_e64 s3, v61, v56                              // 000000002DD4: D4120003 0202713D
	v_cmp_gt_f32_e64 s7, v61, v56                              // 000000002DDC: D4140007 0202713D
	v_permlanex16_b32 v71, v71, s10, 0xfedcba98                // 000000002DE4: D65C0047 03FC1547 FEDCBA98
	v_permlanex16_b32 v55, v55, s10, 0xfedcba98                // 000000002DF0: D65C0037 03FC1537 FEDCBA98
	v_cmp_gt_f32_e64 s5, v63, v58                              // 000000002DFC: D4140005 0202753F
	v_permlanex16_b32 v73, v73, s10, 0xfedcba98                // 000000002E04: D65C0049 03FC1549 FEDCBA98
	s_delay_alu instid0(VALU_DEP_3)                            // 000000002E10: BF870003
	v_cmp_gt_f32_e64 s6, v66, v55                              // 000000002E14: D4140006 02026F42
	s_and_b32 s3, s3, s5                                       // 000000002E1C: 8B030503
	v_cmp_gt_f32_e64 s5, v69, v73                              // 000000002E20: D4140005 02029345
	s_wait_alu depctr_sa_sdst(0)                               // 000000002E28: BF88FF9E
	s_or_b32 s3, s7, s3                                        // 000000002E2C: 8C030307
	v_cmp_gt_f32_e64 s7, v67, v71                              // 000000002E30: D4140007 02028F43
	s_wait_alu depctr_sa_sdst(0)                               // 000000002E38: BF88FF9E
	s_xor_b32 s2, s3, s0                                       // 000000002E3C: 8D020003
	v_cmp_eq_f32_e64 s3, v67, v71                              // 000000002E40: D4120003 02028F43
	s_wait_alu depctr_sa_sdst(0)                               // 000000002E48: BF88FF9E
	v_cndmask_b32_e64 v56, v61, v56, s2                        // 000000002E4C: D5010038 000A713D
	v_cndmask_b32_e32 v61, v62, v57, vcc_lo                    // 000000002E54: 027A733E
	v_permlanex16_b32 v72, v72, s10, 0xfedcba98                // 000000002E58: D65C0048 03FC1548 FEDCBA98
	v_cmp_eq_f32_e32 vcc_lo, v66, v55                          // 000000002E64: 7C246F42
	v_cndmask_b32_e64 v62, v63, v58, s2                        // 000000002E68: D501003E 000A753F
	v_mov_b32_e32 v58, v51                                     // 000000002E70: 7E740333
	s_and_b32 s3, s3, s5                                       // 000000002E74: 8B030503
	v_cmp_gt_f32_e64 s4, v68, v72                              // 000000002E78: D4140004 02029144
	s_wait_alu depctr_sa_sdst(0)                               // 000000002E80: BF88FF9E
	s_or_b32 s3, s7, s3                                        // 000000002E84: 8C030307
	v_permlanex16_b32 v58, v58, s10, 0xfedcba98                // 000000002E88: D65C003A 03FC153A FEDCBA98
	s_and_b32 s2, vcc_lo, s4                                   // 000000002E94: 8B02046A
	s_wait_alu depctr_sa_sdst(0)                               // 000000002E98: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 000000002E9C: 8C020206
	s_delay_alu instid0(VALU_DEP_1)                            // 000000002EA0: BF870001
	v_cmp_gt_f32_e64 s4, v51, v58                              // 000000002EA4: D4140004 02027533
	s_wait_alu depctr_sa_sdst(0)                               // 000000002EAC: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000002EB0: 8D6A0002
	s_wait_alu depctr_sa_sdst(0)                               // 000000002EB4: BF88FF9E
	v_dual_mov_b32 v57, v53 :: v_dual_cndmask_b32 v68, v68, v72// 000000002EB8: CA120135 39449144
	v_cndmask_b32_e32 v63, v66, v55, vcc_lo                    // 000000002EC0: 027E6F42
	v_dual_mov_b32 v55, v52 :: v_dual_mov_b32 v66, v50         // 000000002EC4: CA100134 37420132
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000002ECC: BF870123
	v_permlanex16_b32 v57, v57, s10, 0xfedcba98                // 000000002ED0: D65C0039 03FC1539 FEDCBA98
	s_xor_b32 s2, s3, s0                                       // 000000002EDC: 8D020003
	v_permlanex16_b32 v55, v55, s10, 0xfedcba98                // 000000002EE0: D65C0037 03FC1537 FEDCBA98
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000002EEC: BF870193
	v_permlanex16_b32 v66, v66, s10, 0xfedcba98                // 000000002EF0: D65C0042 03FC1542 FEDCBA98
	v_cmp_eq_f32_e64 s3, v53, v57                              // 000000002EFC: D4120003 02027335
	v_cmp_gt_f32_e64 s7, v53, v57                              // 000000002F04: D4140007 02027335
	s_wait_alu depctr_sa_sdst(0)                               // 000000002F0C: BF88FF9E
	v_cndmask_b32_e64 v67, v67, v71, s2                        // 000000002F10: D5010043 000A8F43
	v_cmp_eq_f32_e32 vcc_lo, v52, v55                          // 000000002F18: 7C246F34
	v_cmp_gt_f32_e64 s5, v50, v66                              // 000000002F1C: D4140005 02028532
	v_cmp_gt_f32_e64 s6, v52, v55                              // 000000002F24: D4140006 02026F34
	v_cndmask_b32_e64 v69, v69, v73, s2                        // 000000002F2C: D5010045 000A9345
	s_and_b32 s2, vcc_lo, s4                                   // 000000002F34: 8B02046A
	s_and_b32 s3, s3, s5                                       // 000000002F38: 8B030503
	s_wait_alu depctr_sa_sdst(0)                               // 000000002F3C: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 000000002F40: 8C020206
	s_or_b32 s3, s7, s3                                        // 000000002F44: 8C030307
	s_wait_alu depctr_sa_sdst(0)                               // 000000002F48: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000002F4C: 8D6A0002
	s_xor_b32 s2, s3, s0                                       // 000000002F50: 8D020003
	s_wait_alu depctr_sa_sdst(0)                               // 000000002F54: BF88FF9E
	v_cndmask_b32_e32 v71, v52, v55, vcc_lo                    // 000000002F58: 028E6F34
	v_cndmask_b32_e32 v73, v51, v58, vcc_lo                    // 000000002F5C: 02927533
	v_cmp_eq_f32_e32 vcc_lo, v59, v54                          // 000000002F60: 7C246D3B
	v_cmp_lt_f32_e64 s4, v64, v65                              // 000000002F64: D4110004 02028340
	v_cndmask_b32_e64 v72, v53, v57, s2                        // 000000002F6C: D5010048 000A7335
	v_cmp_lt_f32_e64 s3, v65, v64                              // 000000002F74: D4110003 02028141
	v_cndmask_b32_e64 v66, v50, v66, s2                        // 000000002F7C: D5010042 000A8532
	v_cmp_lt_f32_e64 s2, v54, v59                              // 000000002F84: D4110002 02027736
	v_cmp_lt_f32_e64 s5, v59, v54                              // 000000002F8C: D4110005 02026D3B
	s_and_b32 s4, vcc_lo, s4                                   // 000000002F94: 8B04046A
	s_and_b32 s3, vcc_lo, s3                                   // 000000002F98: 8B03036A
	v_cmp_lt_f32_e64 s7, v60, v56                              // 000000002F9C: D4110007 0202713C
	s_wait_alu depctr_sa_sdst(0)                               // 000000002FA4: BF88FF9E
	s_or_b32 s2, s2, s4                                        // 000000002FA8: 8C020402
	s_or_b32 vcc_lo, s5, s3                                    // 000000002FAC: 8C6A0305
	s_wait_alu depctr_sa_sdst(0)                               // 000000002FB0: BF88FF9E
	v_cndmask_b32_e64 v53, v54, v59, s2                        // 000000002FB4: D5010035 000A7736
	v_cndmask_b32_e64 v50, v64, v65, s2                        // 000000002FBC: D5010032 000A8340
	v_cmp_eq_f32_e64 s2, v56, v60                              // 000000002FC4: D4120002 02027938
	v_cmp_lt_f32_e64 s3, v62, v61                              // 000000002FCC: D4110003 02027B3E
	v_cndmask_b32_e32 v52, v54, v59, vcc_lo                    // 000000002FD4: 02687736
	v_cmp_lt_f32_e64 s4, v61, v62                              // 000000002FD8: D4110004 02027D3D
	v_cmp_eq_f32_e64 s5, v72, v71                              // 000000002FE0: D4120005 02028F48
	v_cmp_lt_f32_e64 s6, v73, v66                              // 000000002FE8: D4110006 02028549
	s_and_b32 s8, s2, s3                                       // 000000002FF0: 8B080302
	v_cmp_lt_f32_e64 s3, v71, v72                              // 000000002FF4: D4110003 02029147
	s_and_b32 s4, s2, s4                                       // 000000002FFC: 8B040402
	s_and_b32 s2, s5, s6                                       // 000000003000: 8B020605
	v_cmp_lt_f32_e64 s6, v63, v67                              // 000000003004: D4110006 0202873F
	s_wait_alu depctr_sa_sdst(0)                               // 00000000300C: BF88FF9E
	s_or_b32 s2, s3, s2                                        // 000000003010: 8C020203
	v_cmp_lt_f32_e64 s3, v69, v68                              // 000000003014: D4110003 02028945
	s_wait_alu depctr_sa_sdst(0)                               // 00000000301C: BF88FF9E
	v_cndmask_b32_e64 v59, v71, v72, s2                        // 000000003020: D501003B 000A9147
	v_cndmask_b32_e32 v51, v64, v65, vcc_lo                    // 000000003028: 02668340
	v_cmp_lt_f32_e32 vcc_lo, v56, v60                          // 00000000302C: 7C227938
	v_cndmask_b32_e64 v58, v73, v66, s2                        // 000000003030: D501003A 000A8549
	s_or_b32 s2, s7, s4                                        // 000000003038: 8C020407
	v_cmp_lt_f32_e64 s4, v68, v69                              // 00000000303C: D4110004 02028B44
	s_wait_alu depctr_sa_sdst(0)                               // 000000003044: BF88FF9E
	v_cndmask_b32_e64 v57, v60, v56, s2                        // 000000003048: D5010039 000A713C
	s_or_b32 vcc_lo, vcc_lo, s8                                // 000000003050: 8C6A086A
	v_cndmask_b32_e64 v54, v61, v62, s2                        // 000000003054: D5010036 000A7D3D
	s_wait_alu depctr_sa_sdst(0)                               // 00000000305C: BF88FF9E
	v_dual_cndmask_b32 v55, v60, v56 :: v_dual_max_num_f32 v60, v59, v59// 000000003060: CA54713C 373C773B
	v_cndmask_b32_e32 v56, v61, v62, vcc_lo                    // 000000003068: 02707D3D
	v_max_num_f32_e32 v61, v58, v58                            // 00000000306C: 2C7A753A
	v_cmp_eq_f32_e64 s2, v67, v63                              // 000000003070: D4120002 02027F43
	v_cmp_lt_f32_e32 vcc_lo, v67, v63                          // 000000003078: 7C227F43
	v_max_num_f32_e32 v60, 0xff800000, v60                     // 00000000307C: 2C7878FF FF800000
	s_and_b32 s3, s2, s3                                       // 000000003084: 8B030302
	s_and_b32 s2, s2, s4                                       // 000000003088: 8B020402
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000308C: BF870001
	v_cndmask_b32_e64 v74, v60, 0xff800000, s1                 // 000000003090: D501004A 0005FF3C FF800000
	v_max_num_f32_e32 v61, 0xff800000, v61                     // 00000000309C: 2C7A7AFF FF800000
	s_wait_alu depctr_sa_sdst(0)                               // 0000000030A4: BF88FF9E
	s_or_b32 vcc_lo, vcc_lo, s3                                // 0000000030A8: 8C6A036A
	v_cmp_lt_f32_e64 s3, v72, v71                              // 0000000030AC: D4110003 02028F48
	s_wait_alu depctr_sa_sdst(0)                               // 0000000030B4: BF88FF9E
	v_cndmask_b32_e32 v60, v63, v67, vcc_lo                    // 0000000030B8: 0278873F
	v_mov_b32_e32 v64, v74                                     // 0000000030BC: 7E80034A
	v_cndmask_b32_e64 v104, v61, 0xff800000, s1                // 0000000030C0: D5010068 0005FF3D FF800000
	v_cndmask_b32_e32 v61, v68, v69, vcc_lo                    // 0000000030CC: 027A8B44
	v_cmp_lt_f32_e32 vcc_lo, v66, v73                          // 0000000030D0: 7C229342
	s_or_b32 s2, s6, s2                                        // 0000000030D4: 8C020206
	v_permlanex16_b32 v64, v64, s10, 0xfedcba98                // 0000000030D8: D65C0040 03FC1540 FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 0000000030E4: BF88FF9E
	v_cndmask_b32_e64 v62, v63, v67, s2                        // 0000000030E8: D501003E 000A873F
	v_cndmask_b32_e64 v63, v68, v69, s2                        // 0000000030F0: D501003F 000A8B44
	s_and_b32 s2, s5, vcc_lo                                   // 0000000030F8: 8B026A05
	v_mov_b32_e32 v69, 0x7f800000                              // 0000000030FC: 7E8A02FF 7F800000
	s_wait_alu depctr_sa_sdst(0)                               // 000000003104: BF88FF9E
	s_or_b32 vcc_lo, s3, s2                                    // 000000003108: 8C6A0203
	v_mov_b32_e32 v65, v104                                    // 00000000310C: 7E820368
	s_wait_alu depctr_sa_sdst(0)                               // 000000003110: BF88FF9E
	v_dual_max_num_f32 v67, v64, v64 :: v_dual_cndmask_b32 v64, v73, v66// 000000003114: CA928140 43408549
	v_dual_mov_b32 v73, 0x7f800000 :: v_dual_mov_b32 v66, -1.0 // 00000000311C: CA1000FF 494200F3 7F800000
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000003128: BF870093
	v_permlanex16_b32 v65, v65, s10, 0xfedcba98                // 00000000312C: D65C0041 03FC1541 FEDCBA98
	v_dual_max_num_f32 v68, v65, v65 :: v_dual_cndmask_b32 v65, v71, v72// 000000003138: CA928341 44409147
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000003140: BF870114
	v_dual_mov_b32 v72, -1.0 :: v_dual_max_num_f32 v103, v74, v67// 000000003144: CA1400F3 4866874A
	v_dual_mov_b32 v71, -1.0 :: v_dual_max_num_f32 v104, v104, v68// 00000000314C: CA1400F3 47688968
	v_mov_b32_e32 v74, 0x7f800000                              // 000000003154: 7E9402FF 7F800000
	v_dual_mov_b32 v68, 0x7f800000 :: v_dual_mov_b32 v67, -1.0 // 00000000315C: CA1000FF 444200F3 7F800000
	s_branch 64159                                             // 000000003168: BFA0FA9F <nearest+0x1be8>
	v_cndmask_b32_e64 v121, v74, -v14, s2                      // 00000000316C: D5010079 400A1D4A
	v_cndmask_b32_e64 v122, v73, -v15, s3                      // 000000003174: D501007A 400E1F49
	v_cndmask_b32_e64 v123, v72, v119, s2                      // 00000000317C: D501007B 000AEF48
	v_cndmask_b32_e64 v124, v71, v120, s3                      // 000000003184: D501007C 000EF147
	v_cndmask_b32_e64 v68, v68, v74, s2                        // 00000000318C: D5010044 000A9544
	v_cndmask_b32_e64 v69, v69, v73, s3                        // 000000003194: D5010045 000E9345
	v_cndmask_b32_e64 v66, v66, v72, s2                        // 00000000319C: D5010042 000A9142
	v_cndmask_b32_e64 v67, v67, v71, s3                        // 0000000031A4: D5010043 000E8F43
	v_dual_mov_b32 v72, v123 :: v_dual_mov_b32 v71, v124       // 0000000031AC: CA10017B 4846017C
	v_dual_mov_b32 v74, v121 :: v_dual_mov_b32 v73, v122       // 0000000031B4: CA100179 4A48017A
	s_cmp_ngt_f32 s5, 0.5                                      // 0000000031BC: BF4BF005
	s_mov_b32 s14, s6                                          // 0000000031C0: BE8E0006
	s_cbranch_scc0 64666                                       // 0000000031C4: BFA1FC9A <nearest+0x2430>
	s_branch 64135                                             // 0000000031C8: BFA0FA87 <nearest+0x1be8>
	v_cndmask_b32_e64 v121, v74, -v14, s2                      // 0000000031CC: D5010079 400A1D4A
	v_cndmask_b32_e64 v122, v73, -v15, s3                      // 0000000031D4: D501007A 400E1F49
	v_cndmask_b32_e64 v123, v72, v119, s2                      // 0000000031DC: D501007B 000AEF48
	v_cndmask_b32_e64 v124, v71, v120, s3                      // 0000000031E4: D501007C 000EF147
	v_cndmask_b32_e64 v68, v68, v74, s2                        // 0000000031EC: D5010044 000A9544
	v_cndmask_b32_e64 v69, v69, v73, s3                        // 0000000031F4: D5010045 000E9345
	v_cndmask_b32_e64 v66, v66, v72, s2                        // 0000000031FC: D5010042 000A9142
	v_cndmask_b32_e64 v67, v67, v71, s3                        // 000000003204: D5010043 000E8F43
	v_dual_mov_b32 v73, v122 :: v_dual_mov_b32 v74, v121       // 00000000320C: CA10017A 494A0179
	v_dual_mov_b32 v71, v124 :: v_dual_mov_b32 v72, v123       // 000000003214: CA10017C 4748017B
	s_cmp_ngt_f32 s6, 0.5                                      // 00000000321C: BF4BF006
	s_cbranch_scc0 63265                                       // 000000003220: BFA1F721 <nearest+0xea8>
	s_branch 62523                                             // 000000003224: BFA0F43B <nearest+0x314>
	ds_load_b128 v[8:11], v98 offset:8192                      // 000000003228: DBFC2000 08000062
	ds_load_b128 v[12:15], v99 offset:8192                     // 000000003230: DBFC2000 0C000063
	ds_load_b128 v[89:92], v88 offset:8192                     // 000000003238: DBFC2000 59000058
	ds_load_b128 v[93:96], v98 offset:12288                    // 000000003240: DBFC3000 5D000062
	ds_load_b128 v[105:108], v84 offset:8192                   // 000000003248: DBFC2000 69000054
	ds_load_b128 v[109:112], v81 offset:8192                   // 000000003250: DBFC2000 6D000051
	ds_load_b128 v[97:100], v99 offset:12288                   // 000000003258: DBFC3000 61000063
	ds_load_b128 v[113:116], v78 offset:8192                   // 000000003260: DBFC2000 7100004E
	ds_load_b128 v[85:88], v88 offset:12288                    // 000000003268: DBFC3000 55000058
	ds_load_b128 v[117:120], v77 offset:8192                   // 000000003270: DBFC2000 7500004D
	ds_load_b128 v[121:124], v84 offset:12288                  // 000000003278: DBFC3000 79000054
	s_mov_b32 s2, 0x76543210                                   // 000000003280: BE8200FF 76543210
	s_wait_dscnt 0xa                                           // 000000003288: BFC6000A
	v_wmma_f32_16x16x16_f16 v[0:7], v[8:11], v[44:47], 0       // 00000000328C: CC404000 1A025908
	s_wait_dscnt 0x9                                           // 000000003294: BFC60009
	s_delay_alu instid0(VALU_DEP_1)                            // 000000003298: BF870001
	v_wmma_f32_16x16x16_f16 v[0:7], v[12:15], v[40:43], v[0:7] // 00000000329C: CC404000 1C02510C
	s_wait_dscnt 0x7                                           // 0000000032A4: BFC60007
	v_wmma_f32_16x16x16_f16 v[8:15], v[93:96], v[44:47], 0     // 0000000032A8: CC404008 1A02595D
	ds_load_b128 v[44:47], v78 offset:12288                    // 0000000032B0: DBFC3000 2C00004E
	v_wmma_f32_16x16x16_f16 v[0:7], v[89:92], v[36:39], v[0:7] // 0000000032B8: CC404000 1C024959
	ds_load_b128 v[89:92], v76 offset:8192                     // 0000000032C0: DBFC2000 5900004C
	ds_load_b128 v[79:82], v81 offset:12288                    // 0000000032C8: DBFC3000 4F000051
	s_wait_dscnt 0x7                                           // 0000000032D0: BFC60007
	v_wmma_f32_16x16x16_f16 v[8:15], v[97:100], v[40:43], v[8:15]// 0000000032D4: CC404008 1C225161
	ds_load_b128 v[40:43], v77 offset:12288                    // 0000000032DC: DBFC3000 2800004D
	v_wmma_f32_16x16x16_f16 v[0:7], v[105:108], v[32:35], v[0:7]// 0000000032E4: CC404000 1C024169
	s_wait_dscnt 0x6                                           // 0000000032EC: BFC60006
	v_wmma_f32_16x16x16_f16 v[8:15], v[85:88], v[36:39], v[8:15]// 0000000032F0: CC404008 1C224955
	ds_load_b128 v[36:39], v76 offset:12288                    // 0000000032F8: DBFC3000 2400004C
	v_wmma_f32_16x16x16_f16 v[0:7], v[109:112], v[28:31], v[0:7]// 000000003300: CC404000 1C02396D
	s_wait_dscnt 0x5                                           // 000000003308: BFC60005
	v_wmma_f32_16x16x16_f16 v[8:15], v[121:124], v[32:35], v[8:15]// 00000000330C: CC404008 1C224179
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)// 000000003314: BF8701B2
	v_wmma_f32_16x16x16_f16 v[0:7], v[113:116], v[24:27], v[0:7]// 000000003318: CC404000 1C023171
	v_wmma_f32_16x16x16_f16 v[0:7], v[117:120], v[20:23], v[0:7]// 000000003320: CC404000 1C022975
	s_wait_dscnt 0x2                                           // 000000003328: BFC60002
	v_wmma_f32_16x16x16_f16 v[8:15], v[79:82], v[28:31], v[8:15]// 00000000332C: CC404008 1C22394F
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000003334: BF870112
	v_wmma_f32_16x16x16_f16 v[0:7], v[89:92], v[16:19], v[0:7] // 000000003338: CC404000 1C022159
	v_wmma_f32_16x16x16_f16 v[8:15], v[44:47], v[24:27], v[8:15]// 000000003340: CC404008 1C22312C
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000003348: BF870122
	v_min3_num_f32 v24, -v0, -v1, -v2                          // 00000000334C: D6290018 E40A0300
	s_wait_dscnt 0x1                                           // 000000003354: BFC60001
	v_wmma_f32_16x16x16_f16 v[8:15], v[40:43], v[20:23], v[8:15]// 000000003358: CC404008 1C222928
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000003360: BF870122
	v_min3_num_f32 v20, v24, -v3, -v4                          // 000000003364: D6290014 C4120718
	s_wait_dscnt 0x0                                           // 00000000336C: BFC60000
	v_wmma_f32_16x16x16_f16 v[8:15], v[36:39], v[16:19], v[8:15]// 000000003370: CC404008 1C222124
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000003378: BF870112
	v_min3_num_f32 v16, v20, -v5, -v6                          // 00000000337C: D6290010 C41A0B14
	v_max_num_f32_e64 v17, -v15, -v15                          // 000000003384: D5160011 62021F0F
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000338C: BF870092
	v_min3_num_f32 v16, v16, -v7, -v8                          // 000000003390: D6290010 C4220F10
	v_min3_num_f32 v16, v16, -v9, -v10                         // 000000003398: D6290010 C42A1310
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000033A0: BF870091
	v_min3_num_f32 v16, v16, -v11, -v12                        // 0000000033A4: D6290010 C4321710
	v_min3_num_f32 v16, v16, -v13, -v14                        // 0000000033AC: D6290010 C43A1B10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000033B4: BF870091
	v_min_num_f32_e32 v16, v16, v17                            // 0000000033B8: 2A202310
	v_mov_b32_e32 v17, v16                                     // 0000000033BC: 7E220310
	s_wait_alu depctr_sa_sdst(0)                               // 0000000033C0: BF88FF9E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000033C4: BF870091
	v_permlanex16_b32 v17, v17, s2, 0xfedcba98                 // 0000000033C8: D65C0011 03FC0511 FEDCBA98
	v_max_num_f32_e64 v17, -v17, -v17                          // 0000000033D4: D5160011 62022311
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000033DC: BF870091
	v_max_num_f32_e64 v16, -v16, v17                           // 0000000033E0: D5160010 22022310
	v_cmp_ge_f32_e64 s2, v103, -v16                            // 0000000033E8: D4160002 42022167
	s_mov_b32 vcc_lo, s2                                       // 0000000033F0: BEEA0002
	s_wait_alu depctr_sa_sdst(0)                               // 0000000033F4: BF88FF9E
	s_cbranch_vccz 2853                                        // 0000000033F8: BFA30B25 <nearest+0x6090>
	v_cmp_ge_f32_e64 s3, v103, -v0                             // 0000000033FC: D4160003 42020167
	v_cmp_ge_f32_e64 s5, v103, -v1                             // 000000003404: D4160005 42020367
	v_or_b32_e32 v17, 0xfe0, v70                               // 00000000340C: 38228CFF 00000FE0
	v_or_b32_e32 v18, 0xfe1, v70                               // 000000003414: 38248CFF 00000FE1
	s_and_b32 s4, s3, s2                                       // 00000000341C: 8B040203
	s_and_b32 s7, s5, s2                                       // 000000003420: 8B070205
	s_mov_b32 s6, 0                                            // 000000003424: BE860080
	s_wait_alu depctr_sa_sdst(0)                               // 000000003428: BF88FF9E
	s_or_b32 s3, s4, s7                                        // 00000000342C: 8C030704
	s_wait_alu depctr_sa_sdst(0)                               // 000000003430: BF88FF9E
	v_cndmask_b32_e64 v16, 0, 1, s3                            // 000000003434: D5010010 000D0280
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000343C: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v16                            // 000000003440: 7C9A2080
	v_cvt_f32_u32_e32 v16, v17                                 // 000000003444: 7E200D11
	v_cvt_f32_u32_e32 v17, v18                                 // 000000003448: 7E220D12
	s_cbranch_vccz 51                                          // 00000000344C: BFA30033 <nearest+0x351c>
	v_cmp_gt_f32_e64 s6, v103, -v0                             // 000000003450: D4140006 42020167
	s_delay_alu instid0(VALU_DEP_3)                            // 000000003458: BF870003
	v_cmp_gt_f32_e32 vcc_lo, v104, v16                         // 00000000345C: 7C282168
	v_cmp_gt_f32_e64 s8, v103, -v1                             // 000000003460: D4140008 42020367
	v_cmp_gt_f32_e64 s3, v104, v17                             // 000000003468: D4140003 02022368
	v_cmp_lt_f32_e64 s5, -1.0, v66                             // 000000003470: D4110005 020284F3
	s_or_b32 s6, vcc_lo, s6                                    // 000000003478: 8C06066A
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v67                         // 00000000347C: 7C2286F3
	s_or_b32 s8, s3, s8                                        // 000000003480: 8C080803
	s_wait_alu depctr_sa_sdst(0)                               // 000000003484: BF88FF9E
	s_and_b32 s3, s6, s4                                       // 000000003488: 8B030406
	s_and_b32 s4, s8, s7                                       // 00000000348C: 8B040708
	s_wait_alu depctr_sa_sdst(0)                               // 000000003490: BF88FF9E
	s_and_b32 s5, s5, s3                                       // 000000003494: 8B050305
	s_and_b32 s6, vcc_lo, s4                                   // 000000003498: 8B06046A
	s_wait_alu depctr_sa_sdst(0)                               // 00000000349C: BF88FF9E
	s_or_b32 s5, s5, s6                                        // 0000000034A0: 8C050605
	s_mov_b32 s6, 0                                            // 0000000034A4: BE860080
	s_wait_alu depctr_sa_sdst(0)                               // 0000000034A8: BF88FF9E
	v_cndmask_b32_e64 v18, 0, 1, s5                            // 0000000034AC: D5010012 00150280
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000034B4: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v18                            // 0000000034B8: 7C9A2480
	s_cbranch_vccz 3                                           // 0000000034BC: BFA30003 <nearest+0x34cc>
	s_mov_b32 s6, 1.0                                          // 0000000034C0: BE8600F2
	s_mov_b32 s10, 0                                           // 0000000034C4: BE8A0080
	s_branch 22                                                // 0000000034C8: BFA00016 <nearest+0x3524>
	v_cndmask_b32_e64 v18, v74, -v0, s3                        // 0000000034CC: D5010012 400E014A
	v_cndmask_b32_e64 v19, v73, -v1, s4                        // 0000000034D4: D5010013 40120349
	v_cndmask_b32_e64 v20, v72, v16, s3                        // 0000000034DC: D5010014 000E2148
	v_cndmask_b32_e64 v21, v71, v17, s4                        // 0000000034E4: D5010015 00122347
	v_cndmask_b32_e64 v68, v68, v74, s3                        // 0000000034EC: D5010044 000E9544
	v_cndmask_b32_e64 v69, v69, v73, s4                        // 0000000034F4: D5010045 00129345
	v_cndmask_b32_e64 v66, v66, v72, s3                        // 0000000034FC: D5010042 000E9142
	v_cndmask_b32_e64 v67, v67, v71, s4                        // 000000003504: D5010043 00128F43
	v_dual_mov_b32 v73, v19 :: v_dual_mov_b32 v74, v18         // 00000000350C: CA100113 494A0112
	v_dual_mov_b32 v71, v21 :: v_dual_mov_b32 v72, v20         // 000000003514: CA100115 47480114
	s_mov_b32 s10, 0x41000000                                  // 00000000351C: BE8A00FF 41000000
	s_wait_alu depctr_sa_sdst(0)                               // 000000003524: BF88FF9E
	s_cmp_lt_f32 s6, 0.5                                       // 000000003528: BF41F006
	v_cmp_ge_f32_e64 s3, v103, -v2                             // 00000000352C: D4160003 42020567
	v_cmp_ge_f32_e64 s5, v103, -v3                             // 000000003534: D4160005 42020767
	v_or_b32_e32 v19, 0xfe2, v70                               // 00000000353C: 38268CFF 00000FE2
	s_cselect_b32 s4, -1, 0                                    // 000000003544: 980480C1
	v_or_b32_e32 v20, 0xfe3, v70                               // 000000003548: 38288CFF 00000FE3
	s_wait_alu depctr_sa_sdst(0)                               // 000000003550: BF88FF9E
	s_and_b32 s7, s2, s4                                       // 000000003554: 8B070402
	s_wait_alu depctr_sa_sdst(0)                               // 000000003558: BF88FF9E
	s_and_b32 s4, s3, s7                                       // 00000000355C: 8B040703
	s_and_b32 s7, s5, s7                                       // 000000003560: 8B070705
	s_wait_alu depctr_sa_sdst(0)                               // 000000003564: BF88FF9E
	s_or_b32 s3, s4, s7                                        // 000000003568: 8C030704
	s_wait_alu depctr_sa_sdst(0)                               // 00000000356C: BF88FF9E
	v_cndmask_b32_e64 v18, 0, 1, s3                            // 000000003570: D5010012 000D0280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000003578: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v18                            // 00000000357C: 7C9A2480
	v_cvt_f32_u32_e32 v18, v19                                 // 000000003580: 7E240D13
	v_cvt_f32_u32_e32 v19, v20                                 // 000000003584: 7E260D14
	s_cbranch_vccz 50                                          // 000000003588: BFA30032 <nearest+0x3654>
	v_cmp_gt_f32_e64 s8, v103, -v2                             // 00000000358C: D4140008 42020567
	s_delay_alu instid0(VALU_DEP_3)                            // 000000003594: BF870003
	v_cmp_gt_f32_e32 vcc_lo, v104, v18                         // 000000003598: 7C282568
	v_cmp_gt_f32_e64 s11, v103, -v3                            // 00000000359C: D414000B 42020767
	v_cmp_gt_f32_e64 s3, v104, v19                             // 0000000035A4: D4140003 02022768
	v_cmp_lt_f32_e64 s5, -1.0, v66                             // 0000000035AC: D4110005 020284F3
	s_or_b32 s8, vcc_lo, s8                                    // 0000000035B4: 8C08086A
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v67                         // 0000000035B8: 7C2286F3
	s_or_b32 s11, s3, s11                                      // 0000000035BC: 8C0B0B03
	s_and_b32 s3, s8, s4                                       // 0000000035C0: 8B030408
	s_wait_alu depctr_sa_sdst(0)                               // 0000000035C4: BF88FF9E
	s_and_b32 s4, s11, s7                                      // 0000000035C8: 8B04070B
	s_and_b32 s5, s5, s3                                       // 0000000035CC: 8B050305
	s_wait_alu depctr_sa_sdst(0)                               // 0000000035D0: BF88FF9E
	s_and_b32 s7, s4, vcc_lo                                   // 0000000035D4: 8B076A04
	s_wait_alu depctr_sa_sdst(0)                               // 0000000035D8: BF88FF9E
	s_or_b32 s5, s5, s7                                        // 0000000035DC: 8C050705
	s_wait_alu depctr_sa_sdst(0)                               // 0000000035E0: BF88FF9E
	v_cndmask_b32_e64 v20, 0, 1, s5                            // 0000000035E4: D5010014 00150280
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000035EC: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v20                            // 0000000035F0: 7C9A2880
	s_cbranch_vccz 3                                           // 0000000035F4: BFA30003 <nearest+0x3604>
	s_mov_b32 s10, 1.0                                         // 0000000035F8: BE8A00F2
	s_mov_b32 s6, 1.0                                          // 0000000035FC: BE8600F2
	s_branch 20                                                // 000000003600: BFA00014 <nearest+0x3654>
	v_cndmask_b32_e64 v20, v74, -v2, s3                        // 000000003604: D5010014 400E054A
	v_cndmask_b32_e64 v21, v73, -v3, s4                        // 00000000360C: D5010015 40120749
	v_cndmask_b32_e64 v22, v72, v18, s3                        // 000000003614: D5010016 000E2548
	v_cndmask_b32_e64 v23, v71, v19, s4                        // 00000000361C: D5010017 00122747
	v_cndmask_b32_e64 v68, v68, v74, s3                        // 000000003624: D5010044 000E9544
	v_cndmask_b32_e64 v69, v69, v73, s4                        // 00000000362C: D5010045 00129345
	v_cndmask_b32_e64 v66, v66, v72, s3                        // 000000003634: D5010042 000E9142
	v_cndmask_b32_e64 v67, v67, v71, s4                        // 00000000363C: D5010043 00128F43
	v_dual_mov_b32 v73, v21 :: v_dual_mov_b32 v74, v20         // 000000003644: CA100115 494A0114
	v_dual_mov_b32 v71, v23 :: v_dual_mov_b32 v72, v22         // 00000000364C: CA100117 47480116
	s_wait_alu depctr_sa_sdst(0)                               // 000000003654: BF88FF9E
	s_cmp_lt_f32 s6, 0.5                                       // 000000003658: BF41F006
	v_cmp_ge_f32_e64 s3, v103, -v4                             // 00000000365C: D4160003 42020967
	v_cmp_ge_f32_e64 s5, v103, -v5                             // 000000003664: D4160005 42020B67
	v_or_b32_e32 v21, 0xfe4, v70                               // 00000000366C: 382A8CFF 00000FE4
	s_cselect_b32 s4, -1, 0                                    // 000000003674: 980480C1
	v_or_b32_e32 v22, 0xfe5, v70                               // 000000003678: 382C8CFF 00000FE5
	s_wait_alu depctr_sa_sdst(0)                               // 000000003680: BF88FF9E
	s_and_b32 s7, s2, s4                                       // 000000003684: 8B070402
	s_wait_alu depctr_sa_sdst(0)                               // 000000003688: BF88FF9E
	s_and_b32 s4, s3, s7                                       // 00000000368C: 8B040703
	s_and_b32 s7, s5, s7                                       // 000000003690: 8B070705
	s_wait_alu depctr_sa_sdst(0)                               // 000000003694: BF88FF9E
	s_or_b32 s3, s4, s7                                        // 000000003698: 8C030704
	s_wait_alu depctr_sa_sdst(0)                               // 00000000369C: BF88FF9E
	v_cndmask_b32_e64 v20, 0, 1, s3                            // 0000000036A0: D5010014 000D0280
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000036A8: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v20                            // 0000000036AC: 7C9A2880
	v_cvt_f32_u32_e32 v20, v21                                 // 0000000036B0: 7E280D15
	v_cvt_f32_u32_e32 v21, v22                                 // 0000000036B4: 7E2A0D16
	s_cbranch_vccz 50                                          // 0000000036B8: BFA30032 <nearest+0x3784>
	v_cmp_gt_f32_e64 s8, v103, -v4                             // 0000000036BC: D4140008 42020967
	s_delay_alu instid0(VALU_DEP_3)                            // 0000000036C4: BF870003
	v_cmp_gt_f32_e32 vcc_lo, v104, v20                         // 0000000036C8: 7C282968
	v_cmp_gt_f32_e64 s11, v103, -v5                            // 0000000036CC: D414000B 42020B67
	v_cmp_gt_f32_e64 s3, v104, v21                             // 0000000036D4: D4140003 02022B68
	v_cmp_lt_f32_e64 s5, -1.0, v66                             // 0000000036DC: D4110005 020284F3
	s_or_b32 s8, vcc_lo, s8                                    // 0000000036E4: 8C08086A
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v67                         // 0000000036E8: 7C2286F3
	s_or_b32 s11, s3, s11                                      // 0000000036EC: 8C0B0B03
	s_and_b32 s3, s8, s4                                       // 0000000036F0: 8B030408
	s_wait_alu depctr_sa_sdst(0)                               // 0000000036F4: BF88FF9E
	s_and_b32 s4, s11, s7                                      // 0000000036F8: 8B04070B
	s_and_b32 s5, s5, s3                                       // 0000000036FC: 8B050305
	s_wait_alu depctr_sa_sdst(0)                               // 000000003700: BF88FF9E
	s_and_b32 s7, s4, vcc_lo                                   // 000000003704: 8B076A04
	s_wait_alu depctr_sa_sdst(0)                               // 000000003708: BF88FF9E
	s_or_b32 s5, s5, s7                                        // 00000000370C: 8C050705
	s_wait_alu depctr_sa_sdst(0)                               // 000000003710: BF88FF9E
	v_cndmask_b32_e64 v22, 0, 1, s5                            // 000000003714: D5010016 00150280
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000371C: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v22                            // 000000003720: 7C9A2C80
	s_cbranch_vccz 3                                           // 000000003724: BFA30003 <nearest+0x3734>
	s_mov_b32 s6, 1.0                                          // 000000003728: BE8600F2
	s_mov_b32 s10, 2.0                                         // 00000000372C: BE8A00F4
	s_branch 20                                                // 000000003730: BFA00014 <nearest+0x3784>
	v_cndmask_b32_e64 v22, v74, -v4, s3                        // 000000003734: D5010016 400E094A
	v_cndmask_b32_e64 v23, v73, -v5, s4                        // 00000000373C: D5010017 40120B49
	v_cndmask_b32_e64 v24, v72, v20, s3                        // 000000003744: D5010018 000E2948
	v_cndmask_b32_e64 v25, v71, v21, s4                        // 00000000374C: D5010019 00122B47
	v_cndmask_b32_e64 v68, v68, v74, s3                        // 000000003754: D5010044 000E9544
	v_cndmask_b32_e64 v69, v69, v73, s4                        // 00000000375C: D5010045 00129345
	v_cndmask_b32_e64 v66, v66, v72, s3                        // 000000003764: D5010042 000E9142
	v_cndmask_b32_e64 v67, v67, v71, s4                        // 00000000376C: D5010043 00128F43
	v_dual_mov_b32 v73, v23 :: v_dual_mov_b32 v74, v22         // 000000003774: CA100117 494A0116
	v_dual_mov_b32 v71, v25 :: v_dual_mov_b32 v72, v24         // 00000000377C: CA100119 47480118
	s_wait_alu depctr_sa_sdst(0)                               // 000000003784: BF88FF9E
	s_cmp_lt_f32 s6, 0.5                                       // 000000003788: BF41F006
	v_cmp_ge_f32_e64 s3, v103, -v6                             // 00000000378C: D4160003 42020D67
	v_cmp_ge_f32_e64 s5, v103, -v7                             // 000000003794: D4160005 42020F67
	v_or_b32_e32 v23, 0xfe6, v70                               // 00000000379C: 382E8CFF 00000FE6
	s_cselect_b32 s4, -1, 0                                    // 0000000037A4: 980480C1
	v_or_b32_e32 v24, 0xfe7, v70                               // 0000000037A8: 38308CFF 00000FE7
	s_wait_alu depctr_sa_sdst(0)                               // 0000000037B0: BF88FF9E
	s_and_b32 s7, s2, s4                                       // 0000000037B4: 8B070402
	s_wait_alu depctr_sa_sdst(0)                               // 0000000037B8: BF88FF9E
	s_and_b32 s4, s3, s7                                       // 0000000037BC: 8B040703
	s_and_b32 s7, s5, s7                                       // 0000000037C0: 8B070705
	s_wait_alu depctr_sa_sdst(0)                               // 0000000037C4: BF88FF9E
	s_or_b32 s3, s4, s7                                        // 0000000037C8: 8C030704
	s_wait_alu depctr_sa_sdst(0)                               // 0000000037CC: BF88FF9E
	v_cndmask_b32_e64 v22, 0, 1, s3                            // 0000000037D0: D5010016 000D0280
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000037D8: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v22                            // 0000000037DC: 7C9A2C80
	v_cvt_f32_u32_e32 v22, v23                                 // 0000000037E0: 7E2C0D17
	v_cvt_f32_u32_e32 v23, v24                                 // 0000000037E4: 7E2E0D18
	s_cbranch_vccz 51                                          // 0000000037E8: BFA30033 <nearest+0x38b8>
	v_cmp_gt_f32_e64 s8, v103, -v6                             // 0000000037EC: D4140008 42020D67
	s_delay_alu instid0(VALU_DEP_3)                            // 0000000037F4: BF870003
	v_cmp_gt_f32_e32 vcc_lo, v104, v22                         // 0000000037F8: 7C282D68
	v_cmp_gt_f32_e64 s11, v103, -v7                            // 0000000037FC: D414000B 42020F67
	v_cmp_gt_f32_e64 s3, v104, v23                             // 000000003804: D4140003 02022F68
	v_cmp_lt_f32_e64 s5, -1.0, v66                             // 00000000380C: D4110005 020284F3
	s_or_b32 s8, vcc_lo, s8                                    // 000000003814: 8C08086A
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v67                         // 000000003818: 7C2286F3
	s_or_b32 s11, s3, s11                                      // 00000000381C: 8C0B0B03
	s_and_b32 s3, s8, s4                                       // 000000003820: 8B030408
	s_wait_alu depctr_sa_sdst(0)                               // 000000003824: BF88FF9E
	s_and_b32 s4, s11, s7                                      // 000000003828: 8B04070B
	s_and_b32 s5, s5, s3                                       // 00000000382C: 8B050305
	s_wait_alu depctr_sa_sdst(0)                               // 000000003830: BF88FF9E
	s_and_b32 s7, s4, vcc_lo                                   // 000000003834: 8B076A04
	s_wait_alu depctr_sa_sdst(0)                               // 000000003838: BF88FF9E
	s_or_b32 s5, s5, s7                                        // 00000000383C: 8C050705
	s_wait_alu depctr_sa_sdst(0)                               // 000000003840: BF88FF9E
	v_cndmask_b32_e64 v24, 0, 1, s5                            // 000000003844: D5010018 00150280
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000384C: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v24                            // 000000003850: 7C9A3080
	s_cbranch_vccz 4                                           // 000000003854: BFA30004 <nearest+0x3868>
	s_mov_b32 s6, 1.0                                          // 000000003858: BE8600F2
	s_mov_b32 s10, 0x40400000                                  // 00000000385C: BE8A00FF 40400000
	s_branch 20                                                // 000000003864: BFA00014 <nearest+0x38b8>
	v_cndmask_b32_e64 v24, v74, -v6, s3                        // 000000003868: D5010018 400E0D4A
	v_cndmask_b32_e64 v25, v73, -v7, s4                        // 000000003870: D5010019 40120F49
	v_cndmask_b32_e64 v26, v72, v22, s3                        // 000000003878: D501001A 000E2D48
	v_cndmask_b32_e64 v27, v71, v23, s4                        // 000000003880: D501001B 00122F47
	v_cndmask_b32_e64 v68, v68, v74, s3                        // 000000003888: D5010044 000E9544
	v_cndmask_b32_e64 v69, v69, v73, s4                        // 000000003890: D5010045 00129345
	v_cndmask_b32_e64 v66, v66, v72, s3                        // 000000003898: D5010042 000E9142
	v_cndmask_b32_e64 v67, v67, v71, s4                        // 0000000038A0: D5010043 00128F43
	v_dual_mov_b32 v73, v25 :: v_dual_mov_b32 v74, v24         // 0000000038A8: CA100119 494A0118
	v_dual_mov_b32 v71, v27 :: v_dual_mov_b32 v72, v26         // 0000000038B0: CA10011B 4748011A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000038B8: BF88FF9E
	s_cmp_lt_f32 s6, 0.5                                       // 0000000038BC: BF41F006
	v_cmp_ge_f32_e64 s3, v103, -v8                             // 0000000038C0: D4160003 42021167
	v_cmp_ge_f32_e64 s5, v103, -v9                             // 0000000038C8: D4160005 42021367
	v_or_b32_e32 v25, 0xff0, v70                               // 0000000038D0: 38328CFF 00000FF0
	s_cselect_b32 s4, -1, 0                                    // 0000000038D8: 980480C1
	v_or_b32_e32 v26, 0xff1, v70                               // 0000000038DC: 38348CFF 00000FF1
	s_wait_alu depctr_sa_sdst(0)                               // 0000000038E4: BF88FF9E
	s_and_b32 s7, s2, s4                                       // 0000000038E8: 8B070402
	s_wait_alu depctr_sa_sdst(0)                               // 0000000038EC: BF88FF9E
	s_and_b32 s4, s3, s7                                       // 0000000038F0: 8B040703
	s_and_b32 s7, s5, s7                                       // 0000000038F4: 8B070705
	s_wait_alu depctr_sa_sdst(0)                               // 0000000038F8: BF88FF9E
	s_or_b32 s3, s4, s7                                        // 0000000038FC: 8C030704
	s_wait_alu depctr_sa_sdst(0)                               // 000000003900: BF88FF9E
	v_cndmask_b32_e64 v24, 0, 1, s3                            // 000000003904: D5010018 000D0280
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000390C: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v24                            // 000000003910: 7C9A3080
	v_cvt_f32_u32_e32 v24, v25                                 // 000000003914: 7E300D19
	v_cvt_f32_u32_e32 v25, v26                                 // 000000003918: 7E320D1A
	s_cbranch_vccz 50                                          // 00000000391C: BFA30032 <nearest+0x39e8>
	v_cmp_gt_f32_e64 s8, v103, -v8                             // 000000003920: D4140008 42021167
	s_delay_alu instid0(VALU_DEP_3)                            // 000000003928: BF870003
	v_cmp_gt_f32_e32 vcc_lo, v104, v24                         // 00000000392C: 7C283168
	v_cmp_gt_f32_e64 s11, v103, -v9                            // 000000003930: D414000B 42021367
	v_cmp_gt_f32_e64 s3, v104, v25                             // 000000003938: D4140003 02023368
	v_cmp_lt_f32_e64 s5, -1.0, v66                             // 000000003940: D4110005 020284F3
	s_or_b32 s8, vcc_lo, s8                                    // 000000003948: 8C08086A
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v67                         // 00000000394C: 7C2286F3
	s_or_b32 s11, s3, s11                                      // 000000003950: 8C0B0B03
	s_and_b32 s3, s8, s4                                       // 000000003954: 8B030408
	s_wait_alu depctr_sa_sdst(0)                               // 000000003958: BF88FF9E
	s_and_b32 s4, s11, s7                                      // 00000000395C: 8B04070B
	s_and_b32 s5, s5, s3                                       // 000000003960: 8B050305
	s_wait_alu depctr_sa_sdst(0)                               // 000000003964: BF88FF9E
	s_and_b32 s7, s4, vcc_lo                                   // 000000003968: 8B076A04
	s_wait_alu depctr_sa_sdst(0)                               // 00000000396C: BF88FF9E
	s_or_b32 s5, s5, s7                                        // 000000003970: 8C050705
	s_wait_alu depctr_sa_sdst(0)                               // 000000003974: BF88FF9E
	v_cndmask_b32_e64 v26, 0, 1, s5                            // 000000003978: D501001A 00150280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000003980: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v26                            // 000000003984: 7C9A3480
	s_cbranch_vccz 3                                           // 000000003988: BFA30003 <nearest+0x3998>
	s_mov_b32 s6, 1.0                                          // 00000000398C: BE8600F2
	s_mov_b32 s10, 4.0                                         // 000000003990: BE8A00F6
	s_branch 20                                                // 000000003994: BFA00014 <nearest+0x39e8>
	v_cndmask_b32_e64 v26, v74, -v8, s3                        // 000000003998: D501001A 400E114A
	v_cndmask_b32_e64 v27, v73, -v9, s4                        // 0000000039A0: D501001B 40121349
	v_cndmask_b32_e64 v28, v72, v24, s3                        // 0000000039A8: D501001C 000E3148
	v_cndmask_b32_e64 v29, v71, v25, s4                        // 0000000039B0: D501001D 00123347
	v_cndmask_b32_e64 v68, v68, v74, s3                        // 0000000039B8: D5010044 000E9544
	v_cndmask_b32_e64 v69, v69, v73, s4                        // 0000000039C0: D5010045 00129345
	v_cndmask_b32_e64 v66, v66, v72, s3                        // 0000000039C8: D5010042 000E9142
	v_cndmask_b32_e64 v67, v67, v71, s4                        // 0000000039D0: D5010043 00128F43
	v_dual_mov_b32 v73, v27 :: v_dual_mov_b32 v74, v26         // 0000000039D8: CA10011B 494A011A
	v_dual_mov_b32 v71, v29 :: v_dual_mov_b32 v72, v28         // 0000000039E0: CA10011D 4748011C
	s_wait_alu depctr_sa_sdst(0)                               // 0000000039E8: BF88FF9E
	s_cmp_lt_f32 s6, 0.5                                       // 0000000039EC: BF41F006
	v_cmp_ge_f32_e64 s3, v103, -v10                            // 0000000039F0: D4160003 42021567
	v_cmp_ge_f32_e64 s5, v103, -v11                            // 0000000039F8: D4160005 42021767
	v_or_b32_e32 v27, 0xff2, v70                               // 000000003A00: 38368CFF 00000FF2
	s_cselect_b32 s4, -1, 0                                    // 000000003A08: 980480C1
	v_or_b32_e32 v28, 0xff3, v70                               // 000000003A0C: 38388CFF 00000FF3
	s_wait_alu depctr_sa_sdst(0)                               // 000000003A14: BF88FF9E
	s_and_b32 s7, s2, s4                                       // 000000003A18: 8B070402
	s_wait_alu depctr_sa_sdst(0)                               // 000000003A1C: BF88FF9E
	s_and_b32 s4, s3, s7                                       // 000000003A20: 8B040703
	s_and_b32 s7, s5, s7                                       // 000000003A24: 8B070705
	s_wait_alu depctr_sa_sdst(0)                               // 000000003A28: BF88FF9E
	s_or_b32 s3, s4, s7                                        // 000000003A2C: 8C030704
	s_wait_alu depctr_sa_sdst(0)                               // 000000003A30: BF88FF9E
	v_cndmask_b32_e64 v26, 0, 1, s3                            // 000000003A34: D501001A 000D0280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000003A3C: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v26                            // 000000003A40: 7C9A3480
	v_cvt_f32_u32_e32 v26, v27                                 // 000000003A44: 7E340D1B
	v_cvt_f32_u32_e32 v27, v28                                 // 000000003A48: 7E360D1C
	s_cbranch_vccz 51                                          // 000000003A4C: BFA30033 <nearest+0x3b1c>
	v_cmp_gt_f32_e64 s8, v103, -v10                            // 000000003A50: D4140008 42021567
	s_delay_alu instid0(VALU_DEP_3)                            // 000000003A58: BF870003
	v_cmp_gt_f32_e32 vcc_lo, v104, v26                         // 000000003A5C: 7C283568
	v_cmp_gt_f32_e64 s11, v103, -v11                           // 000000003A60: D414000B 42021767
	v_cmp_gt_f32_e64 s3, v104, v27                             // 000000003A68: D4140003 02023768
	v_cmp_lt_f32_e64 s5, -1.0, v66                             // 000000003A70: D4110005 020284F3
	s_or_b32 s8, vcc_lo, s8                                    // 000000003A78: 8C08086A
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v67                         // 000000003A7C: 7C2286F3
	s_or_b32 s11, s3, s11                                      // 000000003A80: 8C0B0B03
	s_and_b32 s3, s8, s4                                       // 000000003A84: 8B030408
	s_wait_alu depctr_sa_sdst(0)                               // 000000003A88: BF88FF9E
	s_and_b32 s4, s11, s7                                      // 000000003A8C: 8B04070B
	s_and_b32 s5, s5, s3                                       // 000000003A90: 8B050305
	s_wait_alu depctr_sa_sdst(0)                               // 000000003A94: BF88FF9E
	s_and_b32 s7, s4, vcc_lo                                   // 000000003A98: 8B076A04
	s_wait_alu depctr_sa_sdst(0)                               // 000000003A9C: BF88FF9E
	s_or_b32 s5, s5, s7                                        // 000000003AA0: 8C050705
	s_wait_alu depctr_sa_sdst(0)                               // 000000003AA4: BF88FF9E
	v_cndmask_b32_e64 v28, 0, 1, s5                            // 000000003AA8: D501001C 00150280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000003AB0: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v28                            // 000000003AB4: 7C9A3880
	s_cbranch_vccz 4                                           // 000000003AB8: BFA30004 <nearest+0x3acc>
	s_mov_b32 s6, 1.0                                          // 000000003ABC: BE8600F2
	s_mov_b32 s10, 0x40a00000                                  // 000000003AC0: BE8A00FF 40A00000
	s_branch 20                                                // 000000003AC8: BFA00014 <nearest+0x3b1c>
	v_cndmask_b32_e64 v28, v74, -v10, s3                       // 000000003ACC: D501001C 400E154A
	v_cndmask_b32_e64 v29, v73, -v11, s4                       // 000000003AD4: D501001D 40121749
	v_cndmask_b32_e64 v30, v72, v26, s3                        // 000000003ADC: D501001E 000E3548
	v_cndmask_b32_e64 v31, v71, v27, s4                        // 000000003AE4: D501001F 00123747
	v_cndmask_b32_e64 v68, v68, v74, s3                        // 000000003AEC: D5010044 000E9544
	v_cndmask_b32_e64 v69, v69, v73, s4                        // 000000003AF4: D5010045 00129345
	v_cndmask_b32_e64 v66, v66, v72, s3                        // 000000003AFC: D5010042 000E9142
	v_cndmask_b32_e64 v67, v67, v71, s4                        // 000000003B04: D5010043 00128F43
	v_dual_mov_b32 v73, v29 :: v_dual_mov_b32 v74, v28         // 000000003B0C: CA10011D 494A011C
	v_dual_mov_b32 v71, v31 :: v_dual_mov_b32 v72, v30         // 000000003B14: CA10011F 4748011E
	s_wait_alu depctr_sa_sdst(0)                               // 000000003B1C: BF88FF9E
	s_cmp_lt_f32 s6, 0.5                                       // 000000003B20: BF41F006
	v_cmp_ge_f32_e64 s3, v103, -v12                            // 000000003B24: D4160003 42021967
	v_cmp_ge_f32_e64 s5, v103, -v13                            // 000000003B2C: D4160005 42021B67
	v_or_b32_e32 v29, 0xff4, v70                               // 000000003B34: 383A8CFF 00000FF4
	s_cselect_b32 s4, -1, 0                                    // 000000003B3C: 980480C1
	v_or_b32_e32 v30, 0xff5, v70                               // 000000003B40: 383C8CFF 00000FF5
	s_wait_alu depctr_sa_sdst(0)                               // 000000003B48: BF88FF9E
	s_and_b32 s7, s2, s4                                       // 000000003B4C: 8B070402
	s_wait_alu depctr_sa_sdst(0)                               // 000000003B50: BF88FF9E
	s_and_b32 s4, s3, s7                                       // 000000003B54: 8B040703
	s_and_b32 s7, s5, s7                                       // 000000003B58: 8B070705
	s_wait_alu depctr_sa_sdst(0)                               // 000000003B5C: BF88FF9E
	s_or_b32 s3, s4, s7                                        // 000000003B60: 8C030704
	s_wait_alu depctr_sa_sdst(0)                               // 000000003B64: BF88FF9E
	v_cndmask_b32_e64 v28, 0, 1, s3                            // 000000003B68: D501001C 000D0280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000003B70: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v28                            // 000000003B74: 7C9A3880
	v_cvt_f32_u32_e32 v28, v29                                 // 000000003B78: 7E380D1D
	v_cvt_f32_u32_e32 v29, v30                                 // 000000003B7C: 7E3A0D1E
	s_cbranch_vccz 51                                          // 000000003B80: BFA30033 <nearest+0x3c50>
	v_cmp_gt_f32_e64 s8, v103, -v12                            // 000000003B84: D4140008 42021967
	s_delay_alu instid0(VALU_DEP_3)                            // 000000003B8C: BF870003
	v_cmp_gt_f32_e32 vcc_lo, v104, v28                         // 000000003B90: 7C283968
	v_cmp_gt_f32_e64 s11, v103, -v13                           // 000000003B94: D414000B 42021B67
	v_cmp_gt_f32_e64 s3, v104, v29                             // 000000003B9C: D4140003 02023B68
	v_cmp_lt_f32_e64 s5, -1.0, v66                             // 000000003BA4: D4110005 020284F3
	s_or_b32 s8, vcc_lo, s8                                    // 000000003BAC: 8C08086A
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v67                         // 000000003BB0: 7C2286F3
	s_or_b32 s11, s3, s11                                      // 000000003BB4: 8C0B0B03
	s_and_b32 s3, s8, s4                                       // 000000003BB8: 8B030408
	s_wait_alu depctr_sa_sdst(0)                               // 000000003BBC: BF88FF9E
	s_and_b32 s4, s11, s7                                      // 000000003BC0: 8B04070B
	s_and_b32 s5, s5, s3                                       // 000000003BC4: 8B050305
	s_wait_alu depctr_sa_sdst(0)                               // 000000003BC8: BF88FF9E
	s_and_b32 s7, s4, vcc_lo                                   // 000000003BCC: 8B076A04
	s_wait_alu depctr_sa_sdst(0)                               // 000000003BD0: BF88FF9E
	s_or_b32 s5, s5, s7                                        // 000000003BD4: 8C050705
	s_wait_alu depctr_sa_sdst(0)                               // 000000003BD8: BF88FF9E
	v_cndmask_b32_e64 v30, 0, 1, s5                            // 000000003BDC: D501001E 00150280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000003BE4: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v30                            // 000000003BE8: 7C9A3C80
	s_cbranch_vccz 4                                           // 000000003BEC: BFA30004 <nearest+0x3c00>
	s_mov_b32 s6, 1.0                                          // 000000003BF0: BE8600F2
	s_mov_b32 s10, 0x40c00000                                  // 000000003BF4: BE8A00FF 40C00000
	s_branch 20                                                // 000000003BFC: BFA00014 <nearest+0x3c50>
	v_cndmask_b32_e64 v30, v74, -v12, s3                       // 000000003C00: D501001E 400E194A
	v_cndmask_b32_e64 v31, v73, -v13, s4                       // 000000003C08: D501001F 40121B49
	v_cndmask_b32_e64 v32, v72, v28, s3                        // 000000003C10: D5010020 000E3948
	v_cndmask_b32_e64 v33, v71, v29, s4                        // 000000003C18: D5010021 00123B47
	v_cndmask_b32_e64 v68, v68, v74, s3                        // 000000003C20: D5010044 000E9544
	v_cndmask_b32_e64 v69, v69, v73, s4                        // 000000003C28: D5010045 00129345
	v_cndmask_b32_e64 v66, v66, v72, s3                        // 000000003C30: D5010042 000E9142
	v_cndmask_b32_e64 v67, v67, v71, s4                        // 000000003C38: D5010043 00128F43
	v_dual_mov_b32 v73, v31 :: v_dual_mov_b32 v74, v30         // 000000003C40: CA10011F 494A011E
	v_dual_mov_b32 v71, v33 :: v_dual_mov_b32 v72, v32         // 000000003C48: CA100121 47480120
	s_wait_alu depctr_sa_sdst(0)                               // 000000003C50: BF88FF9E
	s_cmp_lt_f32 s6, 0.5                                       // 000000003C54: BF41F006
	v_cmp_ge_f32_e64 s3, v103, -v14                            // 000000003C58: D4160003 42021D67
	v_cmp_ge_f32_e64 s4, v103, -v15                            // 000000003C60: D4160004 42021F67
	v_or_b32_e32 v31, 0xff6, v70                               // 000000003C68: 383E8CFF 00000FF6
	s_cselect_b32 s5, -1, 0                                    // 000000003C70: 980580C1
	v_or_b32_e32 v32, 0xff7, v70                               // 000000003C74: 38408CFF 00000FF7
	s_wait_alu depctr_sa_sdst(0)                               // 000000003C7C: BF88FF9E
	s_and_b32 s2, s2, s5                                       // 000000003C80: 8B020502
	s_wait_alu depctr_sa_sdst(0)                               // 000000003C84: BF88FF9E
	s_and_b32 s3, s3, s2                                       // 000000003C88: 8B030203
	s_and_b32 s5, s4, s2                                       // 000000003C8C: 8B050204
	s_wait_alu depctr_sa_sdst(0)                               // 000000003C90: BF88FF9E
	s_or_b32 s2, s3, s5                                        // 000000003C94: 8C020503
	s_wait_alu depctr_sa_sdst(0)                               // 000000003C98: BF88FF9E
	v_cndmask_b32_e64 v30, 0, 1, s2                            // 000000003C9C: D501001E 00090280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000003CA4: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v30                            // 000000003CA8: 7C9A3C80
	v_cvt_f32_u32_e32 v30, v31                                 // 000000003CAC: 7E3C0D1F
	v_cvt_f32_u32_e32 v31, v32                                 // 000000003CB0: 7E3E0D20
	s_cbranch_vccz 30                                          // 000000003CB4: BFA3001E <nearest+0x3d30>
	v_cmp_gt_f32_e64 s7, v103, -v14                            // 000000003CB8: D4140007 42021D67
	s_delay_alu instid0(VALU_DEP_3)                            // 000000003CC0: BF870003
	v_cmp_gt_f32_e32 vcc_lo, v104, v30                         // 000000003CC4: 7C283D68
	v_cmp_gt_f32_e64 s8, v103, -v15                            // 000000003CC8: D4140008 42021F67
	v_cmp_gt_f32_e64 s2, v104, v31                             // 000000003CD0: D4140002 02023F68
	v_cmp_lt_f32_e64 s4, -1.0, v66                             // 000000003CD8: D4110004 020284F3
	s_or_b32 s7, vcc_lo, s7                                    // 000000003CE0: 8C07076A
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v67                         // 000000003CE4: 7C2286F3
	s_or_b32 s8, s2, s8                                        // 000000003CE8: 8C080802
	s_wait_alu depctr_sa_sdst(0)                               // 000000003CEC: BF88FF9E
	s_and_b32 s2, s7, s3                                       // 000000003CF0: 8B020307
	s_and_b32 s3, s8, s5                                       // 000000003CF4: 8B030508
	s_wait_alu depctr_sa_sdst(0)                               // 000000003CF8: BF88FF9E
	s_and_b32 s4, s4, s2                                       // 000000003CFC: 8B040204
	s_and_b32 s5, s3, vcc_lo                                   // 000000003D00: 8B056A03
	s_wait_alu depctr_sa_sdst(0)                               // 000000003D04: BF88FF9E
	s_or_b32 s4, s4, s5                                        // 000000003D08: 8C040504
	s_wait_alu depctr_sa_sdst(0)                               // 000000003D0C: BF88FF9E
	v_cndmask_b32_e64 v32, 0, 1, s4                            // 000000003D10: D5010020 00110280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000003D18: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v32                            // 000000003D1C: 7C9A4080
	s_cbranch_vccz 2245                                        // 000000003D20: BFA308C5 <nearest+0x6038>
	s_mov_b32 s6, 1.0                                          // 000000003D24: BE8600F2
	s_mov_b32 s10, 0x40e00000                                  // 000000003D28: BE8A00FF 40E00000
	s_wait_alu depctr_sa_sdst(0)                               // 000000003D30: BF88FF9E
	s_cmp_ngt_f32 s6, 0.5                                      // 000000003D34: BF4BF006
	s_cbranch_scc1 2261                                        // 000000003D38: BFA208D5 <nearest+0x6090>
	v_cmp_eq_f32_e32 vcc_lo, v73, v74                          // 000000003D3C: 7C249549
	v_cmp_lt_f32_e64 s2, v71, v72                              // 000000003D40: D4110002 02029147
	v_cmp_lt_f32_e64 s3, v72, v71                              // 000000003D48: D4110003 02028F48
	v_cmp_lt_f32_e64 s4, v73, v74                              // 000000003D50: D4110004 02029549
	v_cmp_lt_f32_e64 s5, v74, v73                              // 000000003D58: D4110005 0202934A
	s_mov_b32 s11, 0x76543210                                  // 000000003D60: BE8B00FF 76543210
	s_and_b32 s2, s2, vcc_lo                                   // 000000003D68: 8B026A02
	s_and_b32 s3, s3, vcc_lo                                   // 000000003D6C: 8B036A03
	s_wait_alu depctr_sa_sdst(0)                               // 000000003D70: BF88FF9E
	s_or_b32 s2, s4, s2                                        // 000000003D74: 8C020204
	s_or_b32 s3, s5, s3                                        // 000000003D78: 8C030305
	s_wait_alu depctr_sa_sdst(0)                               // 000000003D7C: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000003D80: 8D6A0002
	s_xor_b32 s2, s3, s9                                       // 000000003D84: 8D020903
	s_wait_alu depctr_sa_sdst(0)                               // 000000003D88: BF88FF9E
	v_cndmask_b32_e32 v32, v74, v73, vcc_lo                    // 000000003D8C: 0240934A
	v_cndmask_b32_e32 v34, v72, v71, vcc_lo                    // 000000003D90: 02448F48
	v_cmp_eq_f32_e32 vcc_lo, v69, v68                          // 000000003D94: 7C248945
	v_cmp_lt_f32_e64 s3, v67, v66                              // 000000003D98: D4110003 02028543
	v_cmp_lt_f32_e64 s5, v69, v68                              // 000000003DA0: D4110005 02028945
	v_cmp_lt_f32_e64 s4, v66, v67                              // 000000003DA8: D4110004 02028742
	v_cndmask_b32_e64 v33, v73, v74, s2                        // 000000003DB0: D5010021 000A9549
	v_cndmask_b32_e64 v35, v71, v72, s2                        // 000000003DB8: D5010023 000A9147
	s_and_b32 s2, s3, vcc_lo                                   // 000000003DC0: 8B026A03
	v_mov_b32_e32 v37, v32                                     // 000000003DC4: 7E4A0320
	s_wait_alu depctr_sa_sdst(0)                               // 000000003DC8: BF88FF9E
	s_or_b32 s2, s5, s2                                        // 000000003DCC: 8C020205
	s_and_b32 s3, s4, vcc_lo                                   // 000000003DD0: 8B036A04
	s_wait_alu depctr_sa_sdst(0)                               // 000000003DD4: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000003DD8: 8D6A0002
	v_mov_b32_e32 v38, v33                                     // 000000003DDC: 7E4C0321
	s_wait_alu depctr_sa_sdst(0)                               // 000000003DE0: BF88FF9E
	v_cndmask_b32_e32 v42, v66, v67, vcc_lo                    // 000000003DE4: 02548742
	v_permlanex16_b32 v37, v37, s11, 0xfedcba98                // 000000003DE8: D65C0025 03FC1725 FEDCBA98
	v_cmp_lt_f32_e64 s6, v68, v69                              // 000000003DF4: D4110006 02028B44
	v_mov_b32_e32 v72, -1.0                                    // 000000003DFC: 7E9002F3
	v_permlanex16_b32 v38, v38, s11, 0xfedcba98                // 000000003E00: D65C0026 03FC1726 FEDCBA98
	v_dual_mov_b32 v45, v42 :: v_dual_cndmask_b32 v36, v68, v69// 000000003E0C: CA12012A 2D248B44
	v_mov_b32_e32 v39, v34                                     // 000000003E14: 7E4E0322
	v_cmp_eq_f32_e32 vcc_lo, v32, v37                          // 000000003E18: 7C244B20
	s_or_b32 s3, s6, s3                                        // 000000003E1C: 8C030306
	v_cmp_gt_f32_e64 s6, v32, v37                              // 000000003E20: D4140006 02024B20
	s_wait_alu depctr_sa_sdst(0)                               // 000000003E28: BF88FF9E
	s_xor_b32 s2, s3, s9                                       // 000000003E2C: 8D020903
	v_permlanex16_b32 v39, v39, s11, 0xfedcba98                // 000000003E30: D65C0027 03FC1727 FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 000000003E3C: BF88FF9E
	v_cndmask_b32_e64 v41, v69, v68, s2                        // 000000003E40: D5010029 000A8945
	v_cndmask_b32_e64 v43, v67, v66, s2                        // 000000003E48: D501002B 000A8543
	v_cmp_eq_f32_e64 s3, v33, v38                              // 000000003E50: D4120003 02024D21
	v_cmp_gt_f32_e64 s7, v33, v38                              // 000000003E58: D4140007 02024D21
	v_cmp_gt_f32_e64 s4, v34, v39                              // 000000003E60: D4140004 02024F22
	v_mov_b32_e32 v44, v41                                     // 000000003E68: 7E580329
	v_mov_b32_e32 v46, v43                                     // 000000003E6C: 7E5C032B
	v_permlanex16_b32 v45, v45, s11, 0xfedcba98                // 000000003E70: D65C002D 03FC172D FEDCBA98
	v_mov_b32_e32 v74, 0x7f800000                              // 000000003E7C: 7E9402FF 7F800000
	s_and_b32 s2, vcc_lo, s4                                   // 000000003E84: 8B02046A
	v_permlanex16_b32 v44, v44, s11, 0xfedcba98                // 000000003E88: D65C002C 03FC172C FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 000000003E94: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 000000003E98: 8C020206
	v_permlanex16_b32 v46, v46, s11, 0xfedcba98                // 000000003E9C: D65C002E 03FC172E FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 000000003EA8: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000003EAC: 8D6A0002
	v_mov_b32_e32 v68, 0x7f800000                              // 000000003EB0: 7E8802FF 7F800000
	s_wait_alu depctr_sa_sdst(0)                               // 000000003EB8: BF88FF9E
	v_cndmask_b32_e32 v34, v34, v39, vcc_lo                    // 000000003EBC: 02444F22
	v_mov_b32_e32 v40, v35                                     // 000000003EC0: 7E500323
	v_cndmask_b32_e32 v32, v32, v37, vcc_lo                    // 000000003EC4: 02404B20
	v_mov_b32_e32 v37, v36                                     // 000000003EC8: 7E4A0324
	v_cmp_neq_f32_e64 s4, v41, v44                             // 000000003ECC: D41D0004 02025929
	v_mov_b32_e32 v66, -1.0                                    // 000000003ED4: 7E8402F3
	v_permlanex16_b32 v40, v40, s11, 0xfedcba98                // 000000003ED8: D65C0028 03FC1728 FEDCBA98
	v_permlanex16_b32 v37, v37, s11, 0xfedcba98                // 000000003EE4: D65C0025 03FC1725 FEDCBA98
	s_mov_b32 s14, 4                                           // 000000003EF0: BE8E0084
	v_mov_b32_e32 v73, 0x7f800000                              // 000000003EF4: 7E9202FF 7F800000
	v_cmp_gt_f32_e64 s5, v35, v40                              // 000000003EFC: D4140005 02025123
	v_mov_b32_e32 v71, -1.0                                    // 000000003F04: 7E8E02F3
	v_cmp_neq_f32_e32 vcc_lo, v36, v37                         // 000000003F08: 7C3A4B24
	v_cmp_ngt_f32_e64 s6, v36, v37                             // 000000003F0C: D41B0006 02024B24
	v_mov_b32_e32 v67, -1.0                                    // 000000003F14: 7E8602F3
	s_and_b32 s3, s3, s5                                       // 000000003F18: 8B030503
	v_cmp_ngt_f32_e64 s5, v43, v46                             // 000000003F1C: D41B0005 02025D2B
	s_wait_alu depctr_sa_sdst(0)                               // 000000003F24: BF88FF9E
	s_or_b32 s3, s7, s3                                        // 000000003F28: 8C030307
	v_cmp_ngt_f32_e64 s7, v41, v44                             // 000000003F2C: D41B0007 02025929
	s_wait_alu depctr_sa_sdst(0)                               // 000000003F34: BF88FF9E
	s_xor_b32 s2, s3, s0                                       // 000000003F38: 8D020003
	v_cmp_ngt_f32_e64 s3, v42, v45                             // 000000003F3C: D41B0003 02025B2A
	s_wait_alu depctr_sa_sdst(0)                               // 000000003F44: BF88FF9E
	v_cndmask_b32_e64 v33, v33, v38, s2                        // 000000003F48: D5010021 000A4D21
	v_cndmask_b32_e64 v35, v35, v40, s2                        // 000000003F50: D5010023 000A5123
	v_mov_b32_e32 v69, 0x7f800000                              // 000000003F58: 7E8A02FF 7F800000
	s_or_b32 s2, vcc_lo, s3                                    // 000000003F60: 8C02036A
	s_or_b32 s3, s4, s5                                        // 000000003F64: 8C030504
	s_wait_alu depctr_sa_sdst(0)                               // 000000003F68: BF88FF9E
	s_and_b32 s2, s6, s2                                       // 000000003F6C: 8B020206
	s_and_b32 s3, s7, s3                                       // 000000003F70: 8B030307
	s_wait_alu depctr_sa_sdst(0)                               // 000000003F74: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000003F78: 8D6A0002
	s_xor_b32 s2, s3, s0                                       // 000000003F7C: 8D020003
	s_wait_alu depctr_sa_sdst(0)                               // 000000003F80: BF88FF9E
	v_cndmask_b32_e32 v36, v36, v37, vcc_lo                    // 000000003F84: 02484B24
	v_cndmask_b32_e32 v38, v42, v45, vcc_lo                    // 000000003F88: 024C5B2A
	v_cmp_eq_f32_e32 vcc_lo, v33, v32                          // 000000003F8C: 7C244121
	v_cmp_lt_f32_e64 s3, v35, v34                              // 000000003F90: D4110003 02024523
	v_cmp_lt_f32_e64 s4, v34, v35                              // 000000003F98: D4110004 02024722
	v_cndmask_b32_e64 v37, v41, v44, s2                        // 000000003FA0: D5010025 000A5929
	v_cmp_lt_f32_e64 s5, v33, v32                              // 000000003FA8: D4110005 02024121
	v_cndmask_b32_e64 v39, v43, v46, s2                        // 000000003FB0: D5010027 000A5D2B
	v_cmp_lt_f32_e64 s2, v32, v33                              // 000000003FB8: D4110002 02024320
	s_and_b32 s3, vcc_lo, s3                                   // 000000003FC0: 8B03036A
	s_and_b32 s4, vcc_lo, s4                                   // 000000003FC4: 8B04046A
	s_wait_alu depctr_sa_sdst(0)                               // 000000003FC8: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 000000003FCC: 8C6A0305
	v_cmp_lt_f32_e64 s3, v39, v38                              // 000000003FD0: D4110003 02024D27
	s_or_b32 s2, s2, s4                                        // 000000003FD8: 8C020402
	s_wait_alu depctr_sa_sdst(0)                               // 000000003FDC: BF88FF9E
	v_cndmask_b32_e32 v40, v32, v33, vcc_lo                    // 000000003FE0: 02504320
	v_cndmask_b32_e64 v32, v32, v33, s2                        // 000000003FE4: D5010020 000A4320
	v_cndmask_b32_e32 v33, v34, v35, vcc_lo                    // 000000003FEC: 02424722
	v_cmp_eq_f32_e32 vcc_lo, v37, v36                          // 000000003FF0: 7C244925
	v_cmp_lt_f32_e64 s4, v38, v39                              // 000000003FF4: D4110004 02024F26
	v_cmp_lt_f32_e64 s5, v37, v36                              // 000000003FFC: D4110005 02024925
	v_cndmask_b32_e64 v34, v34, v35, s2                        // 000000004004: D5010022 000A4722
	v_cmp_lt_f32_e64 s2, v36, v37                              // 00000000400C: D4110002 02024B24
	s_and_b32 s3, vcc_lo, s3                                   // 000000004014: 8B03036A
	s_and_b32 s4, vcc_lo, s4                                   // 000000004018: 8B04046A
	s_wait_alu depctr_sa_sdst(0)                               // 00000000401C: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 000000004020: 8C6A0305
	s_or_b32 s2, s2, s4                                        // 000000004024: 8C020402
	s_wait_alu depctr_sa_sdst(0)                               // 000000004028: BF88FF9E
	v_cndmask_b32_e32 v35, v37, v36, vcc_lo                    // 00000000402C: 02464925
	v_cndmask_b32_e64 v36, v37, v36, s2                        // 000000004030: D5010024 000A4925
	v_cndmask_b32_e32 v37, v39, v38, vcc_lo                    // 000000004038: 024A4D27
	v_cndmask_b32_e64 v38, v39, v38, s2                        // 00000000403C: D5010026 000A4D27
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 000000004044: BF870224
	v_cmp_eq_f32_e32 vcc_lo, v40, v35                          // 000000004048: 7C244728
	v_cmp_lt_f32_e64 s5, v40, v35                              // 00000000404C: D4110005 02024728
	v_cmp_lt_f32_e64 s3, v33, v37                              // 000000004054: D4110003 02024B21
	v_cmp_eq_f32_e64 s2, v32, v36                              // 00000000405C: D4120002 02024920
	v_cmp_lt_f32_e64 s4, v34, v38                              // 000000004064: D4110004 02024D22
	v_cmp_lt_f32_e64 s6, v32, v36                              // 00000000406C: D4110006 02024920
	s_and_b32 s3, vcc_lo, s3                                   // 000000004074: 8B03036A
	s_wait_alu depctr_sa_sdst(0)                               // 000000004078: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 00000000407C: 8C6A0305
	s_and_b32 s2, s2, s4                                       // 000000004080: 8B020402
	s_wait_alu depctr_sa_sdst(0)                               // 000000004084: BF88FF9E
	v_dual_cndmask_b32 v39, v35, v40 :: v_dual_cndmask_b32 v42, v37, v33// 000000004088: CA525123 272A4325
	v_cndmask_b32_e32 v35, v40, v35, vcc_lo                    // 000000004090: 02464728
	v_cndmask_b32_e32 v33, v33, v37, vcc_lo                    // 000000004094: 02424B21
	s_or_b32 s2, s6, s2                                        // 000000004098: 8C020206
	s_delay_alu instid0(VALU_DEP_3)                            // 00000000409C: BF870003
	v_dual_mov_b32 v40, v39 :: v_dual_mov_b32 v45, v42         // 0000000040A0: CA100127 282C012A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000040A8: BF88FF9E
	v_cndmask_b32_e64 v41, v36, v32, s2                        // 0000000040AC: D5010029 000A4124
	v_cndmask_b32_e64 v43, v38, v34, s2                        // 0000000040B4: D501002B 000A4526
	v_cndmask_b32_e64 v32, v32, v36, s2                        // 0000000040BC: D5010020 000A4920
	v_permlanex16_b32 v40, v40, s11, 0xfedcba98                // 0000000040C4: D65C0028 03FC1728 FEDCBA98
	v_permlanex16_b32 v45, v45, s11, 0xfedcba98                // 0000000040D0: D65C002D 03FC172D FEDCBA98
	v_cndmask_b32_e64 v34, v34, v38, s2                        // 0000000040DC: D5010022 000A4D22
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 0000000040E4: BF870214
	v_dual_mov_b32 v37, v35 :: v_dual_mov_b32 v38, v32         // 0000000040E8: CA100123 25260120
	v_cmp_eq_f32_e32 vcc_lo, v39, v40                          // 0000000040F0: 7C245127
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 0000000040F4: BF870224
	v_cmp_gt_f32_e64 s4, v42, v45                              // 0000000040F8: D4140004 02025B2A
	v_cmp_gt_f32_e64 s6, v39, v40                              // 000000004100: D4140006 02025127
	v_permlanex16_b32 v37, v37, s11, 0xfedcba98                // 000000004108: D65C0025 03FC1725 FEDCBA98
	v_mov_b32_e32 v44, v41                                     // 000000004114: 7E580329
	v_permlanex16_b32 v38, v38, s11, 0xfedcba98                // 000000004118: D65C0026 03FC1726 FEDCBA98
	s_and_b32 s2, vcc_lo, s4                                   // 000000004124: 8B02046A
	s_wait_alu depctr_sa_sdst(0)                               // 000000004128: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 00000000412C: 8C020206
	v_permlanex16_b32 v44, v44, s11, 0xfedcba98                // 000000004130: D65C002C 03FC172C FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 00000000413C: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000004140: 8D6A0002
	v_cmp_gt_f32_e64 s6, v35, v37                              // 000000004144: D4140006 02024B23
	s_wait_alu depctr_sa_sdst(0)                               // 00000000414C: BF88FF9E
	v_cndmask_b32_e32 v36, v39, v40, vcc_lo                    // 000000004150: 02485127
	v_dual_mov_b32 v46, v43 :: v_dual_mov_b32 v39, v33         // 000000004154: CA10012B 2E260121
	v_mov_b32_e32 v40, v34                                     // 00000000415C: 7E500322
	v_cmp_eq_f32_e64 s3, v41, v44                              // 000000004160: D4120003 02025929
	v_cmp_gt_f32_e64 s7, v41, v44                              // 000000004168: D4140007 02025929
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004170: BF870004
	v_permlanex16_b32 v46, v46, s11, 0xfedcba98                // 000000004174: D65C002E 03FC172E FEDCBA98
	v_permlanex16_b32 v39, v39, s11, 0xfedcba98                // 000000004180: D65C0027 03FC1727 FEDCBA98
	v_permlanex16_b32 v40, v40, s11, 0xfedcba98                // 00000000418C: D65C0028 03FC1728 FEDCBA98
	v_cndmask_b32_e32 v42, v42, v45, vcc_lo                    // 000000004198: 02545B2A
	v_cmp_eq_f32_e32 vcc_lo, v35, v37                          // 00000000419C: 7C244B23
	v_cmp_gt_f32_e64 s5, v43, v46                              // 0000000041A0: D4140005 02025D2B
	v_cmp_gt_f32_e64 s4, v33, v39                              // 0000000041A8: D4140004 02024F21
	s_and_b32 s3, s3, s5                                       // 0000000041B0: 8B030503
	v_cmp_gt_f32_e64 s5, v34, v40                              // 0000000041B4: D4140005 02025122
	s_wait_alu depctr_sa_sdst(0)                               // 0000000041BC: BF88FF9E
	s_or_b32 s3, s7, s3                                        // 0000000041C0: 8C030307
	v_cmp_gt_f32_e64 s7, v32, v38                              // 0000000041C4: D4140007 02024D20
	s_wait_alu depctr_sa_sdst(0)                               // 0000000041CC: BF88FF9E
	s_xor_b32 s2, s3, s0                                       // 0000000041D0: 8D020003
	v_cmp_eq_f32_e64 s3, v32, v38                              // 0000000041D4: D4120003 02024D20
	s_wait_alu depctr_sa_sdst(0)                               // 0000000041DC: BF88FF9E
	v_cndmask_b32_e64 v41, v41, v44, s2                        // 0000000041E0: D5010029 000A5929
	v_cndmask_b32_e64 v43, v43, v46, s2                        // 0000000041E8: D501002B 000A5D2B
	s_and_b32 s2, vcc_lo, s4                                   // 0000000041F0: 8B02046A
	s_and_b32 s3, s3, s5                                       // 0000000041F4: 8B030503
	s_wait_alu depctr_sa_sdst(0)                               // 0000000041F8: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 0000000041FC: 8C020206
	s_or_b32 s3, s7, s3                                        // 000000004200: 8C030307
	s_wait_alu depctr_sa_sdst(0)                               // 000000004204: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000004208: 8D6A0002
	s_xor_b32 s2, s3, s0                                       // 00000000420C: 8D020003
	s_wait_alu depctr_sa_sdst(0)                               // 000000004210: BF88FF9E
	v_cndmask_b32_e32 v35, v35, v37, vcc_lo                    // 000000004214: 02464B23
	v_cndmask_b32_e32 v33, v33, v39, vcc_lo                    // 000000004218: 02424F21
	v_cmp_eq_f32_e32 vcc_lo, v41, v36                          // 00000000421C: 7C244929
	v_cmp_lt_f32_e64 s3, v43, v42                              // 000000004220: D4110003 0202552B
	v_cmp_lt_f32_e64 s4, v42, v43                              // 000000004228: D4110004 0202572A
	v_cmp_lt_f32_e64 s5, v41, v36                              // 000000004230: D4110005 02024929
	v_cndmask_b32_e64 v32, v32, v38, s2                        // 000000004238: D5010020 000A4D20
	v_cndmask_b32_e64 v34, v34, v40, s2                        // 000000004240: D5010022 000A5122
	v_cmp_lt_f32_e64 s2, v36, v41                              // 000000004248: D4110002 02025324
	s_and_b32 s3, vcc_lo, s3                                   // 000000004250: 8B03036A
	s_and_b32 s4, vcc_lo, s4                                   // 000000004254: 8B04046A
	s_wait_alu depctr_sa_sdst(0)                               // 000000004258: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 00000000425C: 8C6A0305
	v_cmp_lt_f32_e64 s3, v34, v33                              // 000000004260: D4110003 02024322
	s_wait_alu depctr_sa_sdst(0)                               // 000000004268: BF88FF9E
	v_cndmask_b32_e32 v37, v36, v41, vcc_lo                    // 00000000426C: 024A5324
	s_or_b32 s2, s2, s4                                        // 000000004270: 8C020402
	v_cndmask_b32_e32 v38, v42, v43, vcc_lo                    // 000000004274: 024C572A
	v_cmp_eq_f32_e32 vcc_lo, v32, v35                          // 000000004278: 7C244720
	v_cmp_lt_f32_e64 s4, v33, v34                              // 00000000427C: D4110004 02024521
	s_wait_alu depctr_sa_sdst(0)                               // 000000004284: BF88FF9E
	v_cndmask_b32_e64 v36, v36, v41, s2                        // 000000004288: D5010024 000A5324
	v_cmp_lt_f32_e64 s5, v32, v35                              // 000000004290: D4110005 02024720
	v_cndmask_b32_e64 v39, v42, v43, s2                        // 000000004298: D5010027 000A572A
	v_cmp_lt_f32_e64 s2, v35, v32                              // 0000000042A0: D4110002 02024123
	s_and_b32 s3, vcc_lo, s3                                   // 0000000042A8: 8B03036A
	s_and_b32 s4, vcc_lo, s4                                   // 0000000042AC: 8B04046A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000042B0: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 0000000042B4: 8C6A0305
	v_permlanex16_b32 v36, v36, s11, 0xfedcba98                // 0000000042B8: D65C0024 03FC1724 FEDCBA98
	s_or_b32 s2, s2, s4                                        // 0000000042C4: 8C020402
	v_permlanex16_b32 v39, v39, s11, 0xfedcba98                // 0000000042C8: D65C0027 03FC1727 FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 0000000042D4: BF88FF9E
	v_cndmask_b32_e32 v40, v35, v32, vcc_lo                    // 0000000042D8: 02504123
	v_cndmask_b32_e64 v32, v35, v32, s2                        // 0000000042DC: D5010020 000A4123
	v_permlanex16_b32 v37, v37, s11, 0xfedcba98                // 0000000042E4: D65C0025 03FC1725 FEDCBA98
	v_permlanex16_b32 v38, v38, s11, 0xfedcba98                // 0000000042F0: D65C0026 03FC1726 FEDCBA98
	v_cndmask_b32_e32 v35, v33, v34, vcc_lo                    // 0000000042FC: 02464521
	v_cndmask_b32_e64 v33, v33, v34, s2                        // 000000004300: D5010021 000A4521
	v_cmp_eq_f32_e32 vcc_lo, v65, v36                          // 000000004308: 7C244941
	v_cmp_lt_f32_e64 s4, v64, v39                              // 00000000430C: D4110004 02024F40
	v_cmp_eq_f32_e64 s3, v59, v37                              // 000000004314: D4120003 02024B3B
	v_cmp_lt_f32_e64 s5, v58, v38                              // 00000000431C: D4110005 02024D3A
	v_cmp_lt_f32_e64 s2, v65, v36                              // 000000004324: D4110002 02024941
	v_permlanex16_b32 v32, v32, s11, 0xfedcba98                // 00000000432C: D65C0020 03FC1720 FEDCBA98
	v_permlanex16_b32 v40, v40, s11, 0xfedcba98                // 000000004338: D65C0028 03FC1728 FEDCBA98
	v_permlanex16_b32 v33, v33, s11, 0xfedcba98                // 000000004344: D65C0021 03FC1721 FEDCBA98
	v_permlanex16_b32 v35, v35, s11, 0xfedcba98                // 000000004350: D65C0023 03FC1723 FEDCBA98
	v_cmp_lt_f32_e64 s6, v59, v37                              // 00000000435C: D4110006 02024B3B
	s_and_b32 s4, vcc_lo, s4                                   // 000000004364: 8B04046A
	s_and_b32 s8, s3, s5                                       // 000000004368: 8B080503
	s_wait_alu depctr_sa_sdst(0)                               // 00000000436C: BF88FF9E
	s_or_b32 vcc_lo, s2, s4                                    // 000000004370: 8C6A0402
	v_cmp_eq_f32_e64 s3, v60, v32                              // 000000004374: D4120003 0202413C
	v_cmp_eq_f32_e64 s4, v62, v40                              // 00000000437C: D4120004 0202513E
	v_cmp_lt_f32_e64 s5, v61, v33                              // 000000004384: D4110005 0202433D
	v_cmp_lt_f32_e64 s7, v63, v35                              // 00000000438C: D4110007 0202473F
	v_cmp_lt_f32_e64 s2, v60, v32                              // 000000004394: D4110002 0202413C
	s_or_b32 s6, s6, s8                                        // 00000000439C: 8C060806
	v_cmp_lt_f32_e64 s8, v62, v40                              // 0000000043A0: D4110008 0202513E
	s_and_b32 s3, s3, s5                                       // 0000000043A8: 8B030503
	s_and_b32 s4, s4, s7                                       // 0000000043AC: 8B040704
	s_wait_alu depctr_sa_sdst(0)                               // 0000000043B0: BF88FF9E
	s_or_b32 s2, s2, s3                                        // 0000000043B4: 8C020302
	v_cndmask_b32_e32 v34, v36, v65, vcc_lo                    // 0000000043B8: 02448324
	s_or_b32 s3, s8, s4                                        // 0000000043BC: 8C030408
	v_cndmask_b32_e64 v36, v37, v59, s6                        // 0000000043C0: D5010024 001A7725
	s_wait_alu depctr_sa_sdst(0)                               // 0000000043C8: BF88FF9E
	v_cndmask_b32_e64 v32, v32, v60, s2                        // 0000000043CC: D5010020 000A7920
	v_cndmask_b32_e64 v37, v40, v62, s3                        // 0000000043D4: D5010025 000E7D28
	v_cndmask_b32_e64 v33, v33, v61, s2                        // 0000000043DC: D5010021 000A7B21
	v_cndmask_b32_e64 v35, v35, v63, s3                        // 0000000043E4: D5010023 000E7F23
	v_cndmask_b32_e32 v39, v39, v64, vcc_lo                    // 0000000043EC: 024E8127
	v_cmp_eq_f32_e32 vcc_lo, v52, v32                          // 0000000043F0: 7C244134
	v_cmp_eq_f32_e64 s2, v53, v37                              // 0000000043F4: D4120002 02024B35
	v_cmp_lt_f32_e64 s3, v51, v33                              // 0000000043FC: D4110003 02024333
	v_cmp_lt_f32_e64 s4, v50, v35                              // 000000004404: D4110004 02024732
	v_cmp_lt_f32_e64 s5, v52, v32                              // 00000000440C: D4110005 02024134
	v_cndmask_b32_e64 v38, v38, v58, s6                        // 000000004414: D5010026 001A7526
	v_cmp_lt_f32_e64 s6, v53, v37                              // 00000000441C: D4110006 02024B35
	s_and_b32 s3, vcc_lo, s3                                   // 000000004424: 8B03036A
	s_and_b32 s2, s2, s4                                       // 000000004428: 8B020402
	s_wait_alu depctr_sa_sdst(0)                               // 00000000442C: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 000000004430: 8C6A0305
	v_cmp_eq_f32_e64 s3, v57, v36                              // 000000004434: D4120003 02024939
	s_or_b32 s2, s6, s2                                        // 00000000443C: 8C020206
	v_cmp_lt_f32_e64 s5, v54, v38                              // 000000004440: D4110005 02024D36
	s_wait_alu depctr_sa_sdst(0)                               // 000000004448: BF88FF9E
	v_cndmask_b32_e64 v41, v37, v53, s2                        // 00000000444C: D5010029 000A6B25
	v_cndmask_b32_e64 v43, v35, v50, s2                        // 000000004454: D501002B 000A6523
	v_cndmask_b32_e64 v37, v53, v37, s2                        // 00000000445C: D5010025 000A4B35
	v_cndmask_b32_e64 v35, v50, v35, s2                        // 000000004464: D5010023 000A4732
	v_cmp_lt_f32_e64 s2, v57, v36                              // 00000000446C: D4110002 02024939
	s_and_b32 s3, s3, s5                                       // 000000004474: 8B030503
	v_cndmask_b32_e32 v40, v32, v52, vcc_lo                    // 000000004478: 02506920
	v_cndmask_b32_e32 v42, v33, v51, vcc_lo                    // 00000000447C: 02546721
	v_cndmask_b32_e32 v32, v52, v32, vcc_lo                    // 000000004480: 02404134
	s_wait_alu depctr_sa_sdst(0)                               // 000000004484: BF88FF9E
	s_or_b32 s2, s2, s3                                        // 000000004488: 8C020302
	v_cndmask_b32_e32 v33, v51, v33, vcc_lo                    // 00000000448C: 02424333
	s_wait_alu depctr_sa_sdst(0)                               // 000000004490: BF88FF9E
	v_cndmask_b32_e64 v45, v36, v57, s2                        // 000000004494: D501002D 000A7324
	v_cndmask_b32_e64 v47, v38, v54, s2                        // 00000000449C: D501002F 000A6D26
	v_cndmask_b32_e64 v36, v57, v36, s2                        // 0000000044A4: D5010024 000A4939
	v_cndmask_b32_e64 v38, v54, v38, s2                        // 0000000044AC: D5010026 000A4D36
	v_cmp_eq_f32_e32 vcc_lo, v55, v34                          // 0000000044B4: 7C244537
	v_cmp_eq_f32_e64 s3, v41, v45                              // 0000000044B8: D4120003 02025B29
	v_cmp_lt_f32_e64 s5, v43, v47                              // 0000000044C0: D4110005 02025F2B
	v_cmp_lt_f32_e64 s2, v41, v45                              // 0000000044C8: D4110002 02025B29
	v_cmp_lt_f32_e64 s4, v56, v39                              // 0000000044D0: D4110004 02024F38
	v_cmp_lt_f32_e64 s6, v55, v34                              // 0000000044D8: D4110006 02024537
	s_and_b32 s3, s3, s5                                       // 0000000044E0: 8B030503
	v_cmp_lt_f32_e64 s5, v35, v38                              // 0000000044E4: D4110005 02024D23
	s_wait_alu depctr_sa_sdst(0)                               // 0000000044EC: BF88FF9E
	s_or_b32 s2, s2, s3                                        // 0000000044F0: 8C020302
	v_cmp_eq_f32_e64 s3, v37, v36                              // 0000000044F4: D4120003 02024925
	s_wait_alu depctr_sa_sdst(0)                               // 0000000044FC: BF88FF9E
	v_cndmask_b32_e64 v51, v45, v41, s2                        // 000000004500: D5010033 000A532D
	v_cndmask_b32_e64 v53, v47, v43, s2                        // 000000004508: D5010035 000A572F
	v_cndmask_b32_e64 v41, v41, v45, s2                        // 000000004510: D5010029 000A5B29
	v_cndmask_b32_e64 v43, v43, v47, s2                        // 000000004518: D501002B 000A5F2B
	v_cmp_lt_f32_e64 s2, v37, v36                              // 000000004520: D4110002 02024925
	s_and_b32 s4, vcc_lo, s4                                   // 000000004528: 8B04046A
	s_and_b32 s3, s3, s5                                       // 00000000452C: 8B030503
	s_wait_alu depctr_sa_sdst(0)                               // 000000004530: BF88FF9E
	s_or_b32 vcc_lo, s6, s4                                    // 000000004534: 8C6A0406
	v_mov_b32_e32 v54, v51                                     // 000000004538: 7E6C0333
	s_or_b32 s2, s2, s3                                        // 00000000453C: 8C020302
	s_wait_alu depctr_sa_sdst(0)                               // 000000004540: BF88FF9E
	v_cndmask_b32_e32 v46, v39, v56, vcc_lo                    // 000000004544: 025C7127
	v_cndmask_b32_e64 v45, v36, v37, s2                        // 000000004548: D501002D 000A4B24
	v_cndmask_b32_e64 v36, v37, v36, s2                        // 000000004550: D5010024 000A4925
	v_cndmask_b32_e32 v44, v34, v55, vcc_lo                    // 000000004558: 02586F22
	v_dual_cndmask_b32 v34, v55, v34 :: v_dual_cndmask_b32 v39, v56, v39// 00000000455C: CA524537 22264F38
	v_cmp_lt_f32_e64 s4, v42, v46                              // 000000004564: D4110004 02025D2A
	v_mov_b32_e32 v56, v53                                     // 00000000456C: 7E700335
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004570: BF870004
	v_cmp_eq_f32_e32 vcc_lo, v40, v44                          // 000000004574: 7C245928
	v_cmp_lt_f32_e64 s6, v40, v44                              // 000000004578: D4110006 02025928
	v_permlanex16_b32 v54, v54, s11, 0xfedcba98                // 000000004580: D65C0036 03FC1736 FEDCBA98
	v_cndmask_b32_e64 v47, v38, v35, s2                        // 00000000458C: D501002F 000A4726
	v_permlanex16_b32 v56, v56, s11, 0xfedcba98                // 000000004594: D65C0038 03FC1738 FEDCBA98
	s_and_b32 s4, vcc_lo, s4                                   // 0000000045A0: 8B04046A
	v_cndmask_b32_e64 v35, v35, v38, s2                        // 0000000045A4: D5010023 000A4D23
	s_wait_alu depctr_sa_sdst(0)                               // 0000000045AC: BF88FF9E
	s_or_b32 vcc_lo, s6, s4                                    // 0000000045B0: 8C6A0406
	v_cmp_lt_f32_e64 s4, v33, v39                              // 0000000045B4: D4110004 02024F21
	s_wait_alu depctr_sa_sdst(0)                               // 0000000045BC: BF88FF9E
	v_cndmask_b32_e32 v50, v44, v40, vcc_lo                    // 0000000045C0: 0264512C
	v_cndmask_b32_e32 v52, v46, v42, vcc_lo                    // 0000000045C4: 0268552E
	v_cndmask_b32_e32 v40, v40, v44, vcc_lo                    // 0000000045C8: 02505928
	v_cndmask_b32_e32 v42, v42, v46, vcc_lo                    // 0000000045CC: 02545D2A
	v_cmp_eq_f32_e32 vcc_lo, v32, v34                          // 0000000045D0: 7C244520
	v_cmp_lt_f32_e64 s6, v32, v34                              // 0000000045D4: D4110006 02024520
	v_mov_b32_e32 v55, v52                                     // 0000000045DC: 7E6E0334
	v_cmp_eq_f32_e64 s3, v51, v54                              // 0000000045E0: D4120003 02026D33
	v_cmp_gt_f32_e64 s5, v53, v56                              // 0000000045E8: D4140005 02027135
	s_and_b32 s4, vcc_lo, s4                                   // 0000000045F0: 8B04046A
	v_mov_b32_e32 v37, v40                                     // 0000000045F4: 7E4A0328
	s_wait_alu depctr_sa_sdst(0)                               // 0000000045F8: BF88FF9E
	s_or_b32 vcc_lo, s6, s4                                    // 0000000045FC: 8C6A0406
	v_permlanex16_b32 v55, v55, s11, 0xfedcba98                // 000000004600: D65C0037 03FC1737 FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 00000000460C: BF88FF9E
	v_cndmask_b32_e32 v44, v34, v32, vcc_lo                    // 000000004610: 02584122
	v_cndmask_b32_e32 v32, v32, v34, vcc_lo                    // 000000004614: 02404520
	v_mov_b32_e32 v34, v50                                     // 000000004618: 7E440332
	v_dual_cndmask_b32 v46, v39, v33 :: v_dual_cndmask_b32 v33, v33, v39// 00000000461C: CA524327 2E204F21
	v_cmp_gt_f32_e64 s4, v52, v55                              // 000000004624: D4140004 02026F34
	v_mov_b32_e32 v39, v42                                     // 00000000462C: 7E4E032A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004630: BF870004
	v_permlanex16_b32 v34, v34, s11, 0xfedcba98                // 000000004634: D65C0022 03FC1722 FEDCBA98
	v_cmp_gt_f32_e64 s7, v51, v54                              // 000000004640: D4140007 02026D33
	s_and_b32 s3, s3, s5                                       // 000000004648: 8B030503
	v_permlanex16_b32 v37, v37, s11, 0xfedcba98                // 00000000464C: D65C0025 03FC1725 FEDCBA98
	v_permlanex16_b32 v39, v39, s11, 0xfedcba98                // 000000004658: D65C0027 03FC1727 FEDCBA98
	v_cmp_eq_f32_e32 vcc_lo, v50, v34                          // 000000004664: 7C244532
	v_cmp_gt_f32_e64 s6, v50, v34                              // 000000004668: D4140006 02024532
	s_wait_alu depctr_sa_sdst(0)                               // 000000004670: BF88FF9E
	s_or_b32 s3, s7, s3                                        // 000000004674: 8C030307
	v_mov_b32_e32 v38, v41                                     // 000000004678: 7E4C0329
	s_and_b32 s2, vcc_lo, s4                                   // 00000000467C: 8B02046A
	v_cmp_gt_f32_e64 s4, v42, v39                              // 000000004680: D4140004 02024F2A
	s_wait_alu depctr_sa_sdst(0)                               // 000000004688: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 00000000468C: 8C020206
	v_cmp_gt_f32_e64 s6, v40, v37                              // 000000004690: D4140006 02024B28
	s_wait_alu depctr_sa_sdst(0)                               // 000000004698: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 00000000469C: 8D6A0002
	s_xor_b32 s2, s3, s0                                       // 0000000046A0: 8D020003
	s_wait_alu depctr_sa_sdst(0)                               // 0000000046A4: BF88FF9E
	v_cndmask_b32_e32 v34, v50, v34, vcc_lo                    // 0000000046A8: 02444532
	v_cndmask_b32_e64 v51, v51, v54, s2                        // 0000000046AC: D5010033 000A6D33
	v_cndmask_b32_e32 v54, v52, v55, vcc_lo                    // 0000000046B4: 026C6F34
	v_cmp_eq_f32_e32 vcc_lo, v40, v37                          // 0000000046B8: 7C244B28
	v_mov_b32_e32 v50, v43                                     // 0000000046BC: 7E64032B
	v_cndmask_b32_e64 v55, v53, v56, s2                        // 0000000046C0: D5010037 000A7135
	v_permlanex16_b32 v38, v38, s11, 0xfedcba98                // 0000000046C8: D65C0026 03FC1726 FEDCBA98
	v_mov_b32_e32 v53, v46                                     // 0000000046D4: 7E6A032E
	s_and_b32 s2, vcc_lo, s4                                   // 0000000046D8: 8B02046A
	v_permlanex16_b32 v50, v50, s11, 0xfedcba98                // 0000000046DC: D65C0032 03FC1732 FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 0000000046E8: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 0000000046EC: 8C020206
	v_cmp_eq_f32_e64 s3, v41, v38                              // 0000000046F0: D4120003 02024D29
	s_wait_alu depctr_sa_sdst(0)                               // 0000000046F8: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 0000000046FC: 8D6A0002
	v_permlanex16_b32 v53, v53, s11, 0xfedcba98                // 000000004700: D65C0035 03FC1735 FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 00000000470C: BF88FF9E
	v_cndmask_b32_e32 v37, v40, v37, vcc_lo                    // 000000004710: 024A4B28
	v_mov_b32_e32 v40, v44                                     // 000000004714: 7E50032C
	v_cmp_gt_f32_e64 s5, v43, v50                              // 000000004718: D4140005 0202652B
	v_cmp_gt_f32_e64 s7, v41, v38                              // 000000004720: D4140007 02024D29
	v_cndmask_b32_e32 v39, v42, v39, vcc_lo                    // 000000004728: 024E4F2A
	v_cmp_gt_f32_e64 s4, v46, v53                              // 00000000472C: D4140004 02026B2E
	v_permlanex16_b32 v40, v40, s11, 0xfedcba98                // 000000004734: D65C0028 03FC1728 FEDCBA98
	s_and_b32 s3, s3, s5                                       // 000000004740: 8B030503
	v_mov_b32_e32 v52, v45                                     // 000000004744: 7E68032D
	s_wait_alu depctr_sa_sdst(0)                               // 000000004748: BF88FF9E
	s_or_b32 s3, s7, s3                                        // 00000000474C: 8C030307
	v_mov_b32_e32 v56, v47                                     // 000000004750: 7E70032F
	v_cmp_eq_f32_e32 vcc_lo, v44, v40                          // 000000004754: 7C24512C
	v_cmp_gt_f32_e64 s6, v44, v40                              // 000000004758: D4140006 0202512C
	s_wait_alu depctr_sa_sdst(0)                               // 000000004760: BF88FF9E
	s_xor_b32 s2, s3, s0                                       // 000000004764: 8D020003
	v_permlanex16_b32 v52, v52, s11, 0xfedcba98                // 000000004768: D65C0034 03FC1734 FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 000000004774: BF88FF9E
	v_cndmask_b32_e64 v38, v41, v38, s2                        // 000000004778: D5010026 000A4D29
	v_cndmask_b32_e64 v41, v43, v50, s2                        // 000000004780: D5010029 000A652B
	s_and_b32 s2, vcc_lo, s4                                   // 000000004788: 8B02046A
	v_permlanex16_b32 v56, v56, s11, 0xfedcba98                // 00000000478C: D65C0038 03FC1738 FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 000000004798: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 00000000479C: 8C020206
	v_mov_b32_e32 v42, v32                                     // 0000000047A0: 7E540320
	s_wait_alu depctr_sa_sdst(0)                               // 0000000047A4: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 0000000047A8: 8D6A0002
	v_mov_b32_e32 v43, v36                                     // 0000000047AC: 7E560324
	s_wait_alu depctr_sa_sdst(0)                               // 0000000047B0: BF88FF9E
	v_cndmask_b32_e32 v40, v44, v40, vcc_lo                    // 0000000047B4: 0250512C
	v_mov_b32_e32 v44, v33                                     // 0000000047B8: 7E580321
	v_mov_b32_e32 v50, v35                                     // 0000000047BC: 7E640323
	v_cmp_eq_f32_e64 s3, v45, v52                              // 0000000047C0: D4120003 0202692D
	v_cmp_gt_f32_e64 s5, v47, v56                              // 0000000047C8: D4140005 0202712F
	v_cmp_gt_f32_e64 s7, v45, v52                              // 0000000047D0: D4140007 0202692D
	v_permlanex16_b32 v42, v42, s11, 0xfedcba98                // 0000000047D8: D65C002A 03FC172A FEDCBA98
	v_permlanex16_b32 v44, v44, s11, 0xfedcba98                // 0000000047E4: D65C002C 03FC172C FEDCBA98
	v_permlanex16_b32 v43, v43, s11, 0xfedcba98                // 0000000047F0: D65C002B 03FC172B FEDCBA98
	v_permlanex16_b32 v50, v50, s11, 0xfedcba98                // 0000000047FC: D65C0032 03FC1732 FEDCBA98
	s_and_b32 s3, s3, s5                                       // 000000004808: 8B030503
	v_cndmask_b32_e32 v46, v46, v53, vcc_lo                    // 00000000480C: 025C6B2E
	s_wait_alu depctr_sa_sdst(0)                               // 000000004810: BF88FF9E
	s_or_b32 s3, s7, s3                                        // 000000004814: 8C030307
	v_cmp_eq_f32_e32 vcc_lo, v32, v42                          // 000000004818: 7C245520
	v_cmp_gt_f32_e64 s4, v33, v44                              // 00000000481C: D4140004 02025921
	s_wait_alu depctr_sa_sdst(0)                               // 000000004824: BF88FF9E
	s_xor_b32 s2, s3, s0                                       // 000000004828: 8D020003
	v_cmp_eq_f32_e64 s3, v36, v43                              // 00000000482C: D4120003 02025724
	v_cmp_gt_f32_e64 s5, v35, v50                              // 000000004834: D4140005 02026523
	v_cmp_gt_f32_e64 s6, v32, v42                              // 00000000483C: D4140006 02025520
	v_cmp_gt_f32_e64 s7, v36, v43                              // 000000004844: D4140007 02025724
	s_wait_alu depctr_sa_sdst(0)                               // 00000000484C: BF88FF9E
	v_cndmask_b32_e64 v45, v45, v52, s2                        // 000000004850: D501002D 000A692D
	v_cndmask_b32_e64 v47, v47, v56, s2                        // 000000004858: D501002F 000A712F
	s_and_b32 s2, vcc_lo, s4                                   // 000000004860: 8B02046A
	s_and_b32 s3, s3, s5                                       // 000000004864: 8B030503
	s_wait_alu depctr_sa_sdst(0)                               // 000000004868: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 00000000486C: 8C020206
	s_or_b32 s3, s7, s3                                        // 000000004870: 8C030307
	s_wait_alu depctr_sa_sdst(0)                               // 000000004874: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000004878: 8D6A0002
	s_xor_b32 s2, s3, s0                                       // 00000000487C: 8D020003
	s_wait_alu depctr_sa_sdst(0)                               // 000000004880: BF88FF9E
	v_dual_cndmask_b32 v32, v32, v42 :: v_dual_cndmask_b32 v33, v33, v44// 000000004884: CA525520 20205921
	v_cmp_eq_f32_e32 vcc_lo, v51, v34                          // 00000000488C: 7C244533
	v_cmp_lt_f32_e64 s4, v54, v55                              // 000000004890: D4110004 02026F36
	v_cndmask_b32_e64 v36, v36, v43, s2                        // 000000004898: D5010024 000A5724
	v_cmp_lt_f32_e64 s3, v55, v54                              // 0000000048A0: D4110003 02026D37
	v_cndmask_b32_e64 v35, v35, v50, s2                        // 0000000048A8: D5010023 000A6523
	v_cmp_lt_f32_e64 s2, v34, v51                              // 0000000048B0: D4110002 02026722
	v_cmp_lt_f32_e64 s5, v51, v34                              // 0000000048B8: D4110005 02024533
	s_and_b32 s4, vcc_lo, s4                                   // 0000000048C0: 8B04046A
	s_and_b32 s3, vcc_lo, s3                                   // 0000000048C4: 8B03036A
	v_cmp_lt_f32_e64 s6, v33, v35                              // 0000000048C8: D4110006 02024721
	s_wait_alu depctr_sa_sdst(0)                               // 0000000048D0: BF88FF9E
	s_or_b32 s2, s2, s4                                        // 0000000048D4: 8C020402
	s_or_b32 vcc_lo, s5, s3                                    // 0000000048D8: 8C6A0305
	s_wait_alu depctr_sa_sdst(0)                               // 0000000048DC: BF88FF9E
	v_cndmask_b32_e64 v53, v34, v51, s2                        // 0000000048E0: D5010035 000A6722
	v_cndmask_b32_e64 v50, v54, v55, s2                        // 0000000048E8: D5010032 000A6F36
	v_cmp_eq_f32_e64 s2, v38, v37                              // 0000000048F0: D4120002 02024B26
	v_cmp_lt_f32_e64 s3, v41, v39                              // 0000000048F8: D4110003 02024F29
	v_cmp_lt_f32_e64 s4, v39, v41                              // 000000004900: D4110004 02025327
	v_cmp_eq_f32_e64 s5, v36, v32                              // 000000004908: D4120005 02024124
	v_cndmask_b32_e32 v52, v34, v51, vcc_lo                    // 000000004910: 02686722
	v_cndmask_b32_e32 v51, v54, v55, vcc_lo                    // 000000004914: 02666F36
	s_and_b32 s8, s2, s3                                       // 000000004918: 8B080302
	v_cmp_lt_f32_e64 s3, v32, v36                              // 00000000491C: D4110003 02024920
	s_and_b32 s4, s2, s4                                       // 000000004924: 8B040402
	s_and_b32 s2, s5, s6                                       // 000000004928: 8B020605
	v_cmp_lt_f32_e32 vcc_lo, v38, v37                          // 00000000492C: 7C224B26
	v_cmp_lt_f32_e64 s7, v37, v38                              // 000000004930: D4110007 02024D25
	s_wait_alu depctr_sa_sdst(0)                               // 000000004938: BF88FF9E
	s_or_b32 s2, s3, s2                                        // 00000000493C: 8C020203
	v_cmp_lt_f32_e64 s3, v47, v46                              // 000000004940: D4110003 02025D2F
	s_wait_alu depctr_sa_sdst(0)                               // 000000004948: BF88FF9E
	v_cndmask_b32_e64 v59, v32, v36, s2                        // 00000000494C: D501003B 000A4920
	v_cndmask_b32_e64 v58, v33, v35, s2                        // 000000004954: D501003A 000A4721
	s_or_b32 vcc_lo, vcc_lo, s8                                // 00000000495C: 8C6A086A
	s_or_b32 s2, s7, s4                                        // 000000004960: 8C020407
	s_wait_alu depctr_sa_sdst(0)                               // 000000004964: BF88FF9E
	v_cndmask_b32_e32 v55, v37, v38, vcc_lo                    // 000000004968: 026E4D25
	v_cndmask_b32_e64 v57, v37, v38, s2                        // 00000000496C: D5010039 000A4D25
	v_dual_max_num_f32 v34, v59, v59 :: v_dual_max_num_f32 v37, v58, v58// 000000004974: CA94773B 2224753A
	v_cndmask_b32_e64 v54, v39, v41, s2                        // 00000000497C: D5010036 000A5327
	v_cmp_eq_f32_e64 s2, v45, v40                              // 000000004984: D4120002 0202512D
	v_cndmask_b32_e32 v56, v39, v41, vcc_lo                    // 00000000498C: 02705327
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004990: BF870004
	v_dual_max_num_f32 v34, 0xff800000, v34 :: v_dual_max_num_f32 v37, 0xff800000, v37// 000000004994: CA9444FF 22244AFF FF800000
	v_cmp_lt_f32_e32 vcc_lo, v45, v40                          // 0000000049A0: 7C22512D
	v_cmp_lt_f32_e64 s4, v46, v47                              // 0000000049A4: D4110004 02025F2E
	s_and_b32 s3, s2, s3                                       // 0000000049AC: 8B030302
	v_cndmask_b32_e64 v34, v34, 0xff800000, s1                 // 0000000049B0: D5010022 0005FF22 FF800000
	v_cndmask_b32_e64 v37, v37, 0xff800000, s1                 // 0000000049BC: D5010025 0005FF25 FF800000
	v_cmp_lt_f32_e64 s6, v40, v45                              // 0000000049C8: D4110006 02025B28
	s_wait_alu depctr_sa_sdst(0)                               // 0000000049D0: BF88FF9E
	s_or_b32 vcc_lo, vcc_lo, s3                                // 0000000049D4: 8C6A036A
	s_and_b32 s2, s2, s4                                       // 0000000049D8: 8B020402
	v_dual_mov_b32 v38, v34 :: v_dual_mov_b32 v39, v37         // 0000000049DC: CA100122 26260125
	s_wait_alu depctr_sa_sdst(0)                               // 0000000049E4: BF88FF9E
	v_dual_cndmask_b32 v60, v40, v45 :: v_dual_cndmask_b32 v61, v46, v47// 0000000049E8: CA525B28 3C3C5F2E
	v_cmp_lt_f32_e32 vcc_lo, v35, v33                          // 0000000049F0: 7C224323
	s_delay_alu instid0(VALU_DEP_3)                            // 0000000049F4: BF870003
	v_permlanex16_b32 v38, v38, s11, 0xfedcba98                // 0000000049F8: D65C0026 03FC1726 FEDCBA98
	v_permlanex16_b32 v39, v39, s11, 0xfedcba98                // 000000004A04: D65C0027 03FC1727 FEDCBA98
	v_cmp_lt_f32_e64 s3, v36, v32                              // 000000004A10: D4110003 02024124
	s_or_b32 s2, s6, s2                                        // 000000004A18: 8C020206
	v_dual_max_num_f32 v38, v38, v38 :: v_dual_max_num_f32 v39, v39, v39// 000000004A1C: CA944D26 26264F27
	s_wait_alu depctr_sa_sdst(0)                               // 000000004A24: BF88FF9E
	v_cndmask_b32_e64 v62, v40, v45, s2                        // 000000004A28: D501003E 000A5B28
	v_cndmask_b32_e64 v63, v46, v47, s2                        // 000000004A30: D501003F 000A5F2E
	s_and_b32 s2, s5, vcc_lo                                   // 000000004A38: 8B026A05
	s_wait_alu depctr_sa_sdst(0)                               // 000000004A3C: BF88FF9E
	s_or_b32 vcc_lo, s3, s2                                    // 000000004A40: 8C6A0203
	s_wait_alu depctr_sa_sdst(0)                               // 000000004A44: BF88FF9E
	v_dual_cndmask_b32 v65, v32, v36 :: v_dual_cndmask_b32 v64, v33, v35// 000000004A48: CA524920 41404721
	v_dual_max_num_f32 v32, v34, v38 :: v_dual_max_num_f32 v33, v37, v39// 000000004A50: CA944D22 20204F25
	s_branch 4                                                 // 000000004A58: BFA00004 <nearest+0x4a6c>
	s_add_co_i32 s14, s14, -1                                  // 000000004A5C: 810EC10E
	s_wait_alu depctr_sa_sdst(0)                               // 000000004A60: BF88FF9E
	s_cmp_lg_u32 s14, 0                                        // 000000004A64: BF07800E
	s_cbranch_scc0 1417                                        // 000000004A68: BFA10589 <nearest+0x6090>
	s_cmp_nlt_f32 s10, 0x41000000                              // 000000004A6C: BF4EFF0A 41000000
	s_cbranch_scc1 65529                                       // 000000004A74: BFA2FFF9 <nearest+0x4a5c>
	s_cmp_le_f32 s10, 0                                        // 000000004A78: BF43800A
	v_cmp_ge_f32_e64 s2, v32, -v0                              // 000000004A7C: D4160002 42020120
	v_cmp_ge_f32_e64 s4, v32, -v1                              // 000000004A84: D4160004 42020320
	s_cselect_b32 s5, -1, 0                                    // 000000004A8C: 980580C1
	s_wait_alu depctr_sa_sdst(0)                               // 000000004A90: BF88FF9E
	s_and_b32 s3, s2, s5                                       // 000000004A94: 8B030502
	s_and_b32 s6, s4, s5                                       // 000000004A98: 8B060504
	s_mov_b32 s5, 0                                            // 000000004A9C: BE850080
	s_wait_alu depctr_sa_sdst(0)                               // 000000004AA0: BF88FF9E
	s_or_b32 vcc_lo, s3, s6                                    // 000000004AA4: 8C6A0603
	s_wait_alu depctr_sa_sdst(0)                               // 000000004AA8: BF88FF9E
	s_cbranch_vccz 29                                          // 000000004AAC: BFA3001D <nearest+0x4b24>
	v_cmp_gt_f32_e64 s5, v32, -v0                              // 000000004AB0: D4140005 42020120
	v_cmp_gt_f32_e32 vcc_lo, v33, v16                          // 000000004AB8: 7C282121
	v_cmp_gt_f32_e64 s7, v32, -v1                              // 000000004ABC: D4140007 42020320
	v_cmp_gt_f32_e64 s2, v33, v17                              // 000000004AC4: D4140002 02022321
	v_cmp_lt_f32_e64 s4, -1.0, v66                             // 000000004ACC: D4110004 020284F3
	s_or_b32 s5, s5, vcc_lo                                    // 000000004AD4: 8C056A05
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v67                         // 000000004AD8: 7C2286F3
	s_or_b32 s7, s7, s2                                        // 000000004ADC: 8C070207
	s_wait_alu depctr_sa_sdst(0)                               // 000000004AE0: BF88FF9E
	s_and_b32 s2, s5, s3                                       // 000000004AE4: 8B020305
	s_and_b32 s3, s7, s6                                       // 000000004AE8: 8B030607
	s_wait_alu depctr_sa_sdst(0)                               // 000000004AEC: BF88FF9E
	s_and_b32 s4, s4, s2                                       // 000000004AF0: 8B040204
	s_and_b32 s5, vcc_lo, s3                                   // 000000004AF4: 8B05036A
	s_wait_alu depctr_sa_sdst(0)                               // 000000004AF8: BF88FF9E
	s_or_b32 s4, s4, s5                                        // 000000004AFC: 8C040504
	s_wait_alu depctr_sa_sdst(0)                               // 000000004B00: BF88FF9E
	v_cndmask_b32_e64 v34, 0, 1, s4                            // 000000004B04: D5010022 00110280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000004B0C: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v34                            // 000000004B10: 7C9A4480
	s_cbranch_vccz 6                                           // 000000004B14: BFA30006 <nearest+0x4b30>
	s_mov_b32 s6, 0                                            // 000000004B18: BE860080
	s_mov_b32 s5, 1.0                                          // 000000004B1C: BE8500F2
	s_branch 26                                                // 000000004B20: BFA0001A <nearest+0x4b8c>
	s_mov_b32 s6, 0x41000000                                   // 000000004B24: BE8600FF 41000000
	s_branch 23                                                // 000000004B2C: BFA00017 <nearest+0x4b8c>
	v_cndmask_b32_e64 v34, v74, -v0, s2                        // 000000004B30: D5010022 400A014A
	v_cndmask_b32_e64 v35, v73, -v1, s3                        // 000000004B38: D5010023 400E0349
	v_cndmask_b32_e64 v36, v72, v16, s2                        // 000000004B40: D5010024 000A2148
	v_cndmask_b32_e64 v37, v71, v17, s3                        // 000000004B48: D5010025 000E2347
	v_cndmask_b32_e64 v68, v68, v74, s2                        // 000000004B50: D5010044 000A9544
	v_cndmask_b32_e64 v69, v69, v73, s3                        // 000000004B58: D5010045 000E9345
	v_cndmask_b32_e64 v66, v66, v72, s2                        // 000000004B60: D5010042 000A9142
	v_cndmask_b32_e64 v67, v67, v71, s3                        // 000000004B68: D5010043 000E8F43
	v_dual_mov_b32 v72, v36 :: v_dual_mov_b32 v71, v37         // 000000004B70: CA100124 48460125
	v_dual_mov_b32 v74, v34 :: v_dual_mov_b32 v73, v35         // 000000004B78: CA100122 4A480123
	s_mov_b32 s6, 0x41000000                                   // 000000004B80: BE8600FF 41000000
	s_mov_b32 s5, 0                                            // 000000004B88: BE850080
	s_wait_alu depctr_sa_sdst(0)                               // 000000004B8C: BF88FF9E
	s_cmp_lt_f32 s5, 0.5                                       // 000000004B90: BF41F005
	v_cmp_ge_f32_e64 s3, v32, -v2                              // 000000004B94: D4160003 42020520
	v_cmp_ge_f32_e64 s4, v32, -v3                              // 000000004B9C: D4160004 42020720
	s_cselect_b32 s2, -1, 0                                    // 000000004BA4: 980280C1
	s_cmp_le_f32 s10, 1.0                                      // 000000004BA8: BF43F20A
	s_cselect_b32 s7, -1, 0                                    // 000000004BAC: 980780C1
	s_wait_alu depctr_sa_sdst(0)                               // 000000004BB0: BF88FF9E
	s_and_b32 s2, s7, s2                                       // 000000004BB4: 8B020207
	s_wait_alu depctr_sa_sdst(0)                               // 000000004BB8: BF88FF9E
	s_and_b32 s3, s3, s2                                       // 000000004BBC: 8B030203
	s_and_b32 s7, s4, s2                                       // 000000004BC0: 8B070204
	s_wait_alu depctr_sa_sdst(0)                               // 000000004BC4: BF88FF9E
	s_or_b32 vcc_lo, s3, s7                                    // 000000004BC8: 8C6A0703
	s_wait_alu depctr_sa_sdst(0)                               // 000000004BCC: BF88FF9E
	s_cbranch_vccz 49                                          // 000000004BD0: BFA30031 <nearest+0x4c98>
	v_cmp_gt_f32_e64 s8, v32, -v2                              // 000000004BD4: D4140008 42020520
	v_cmp_gt_f32_e32 vcc_lo, v33, v18                          // 000000004BDC: 7C282521
	v_cmp_gt_f32_e64 s15, v32, -v3                             // 000000004BE0: D414000F 42020720
	v_cmp_gt_f32_e64 s2, v33, v19                              // 000000004BE8: D4140002 02022721
	v_cmp_lt_f32_e64 s4, -1.0, v66                             // 000000004BF0: D4110004 020284F3
	s_or_b32 s8, s8, vcc_lo                                    // 000000004BF8: 8C086A08
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v67                         // 000000004BFC: 7C2286F3
	s_or_b32 s15, s15, s2                                      // 000000004C00: 8C0F020F
	s_and_b32 s2, s8, s3                                       // 000000004C04: 8B020308
	s_wait_alu depctr_sa_sdst(0)                               // 000000004C08: BF88FF9E
	s_and_b32 s3, s15, s7                                      // 000000004C0C: 8B03070F
	s_and_b32 s4, s4, s2                                       // 000000004C10: 8B040204
	s_wait_alu depctr_sa_sdst(0)                               // 000000004C14: BF88FF9E
	s_and_b32 s7, vcc_lo, s3                                   // 000000004C18: 8B07036A
	s_wait_alu depctr_sa_sdst(0)                               // 000000004C1C: BF88FF9E
	s_or_b32 s4, s4, s7                                        // 000000004C20: 8C040704
	s_wait_alu depctr_sa_sdst(0)                               // 000000004C24: BF88FF9E
	v_cndmask_b32_e64 v34, 0, 1, s4                            // 000000004C28: D5010022 00110280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000004C30: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v34                            // 000000004C34: 7C9A4480
	s_cbranch_vccz 3                                           // 000000004C38: BFA30003 <nearest+0x4c48>
	s_mov_b32 s5, 1.0                                          // 000000004C3C: BE8500F2
	s_mov_b32 s6, 1.0                                          // 000000004C40: BE8600F2
	s_branch 20                                                // 000000004C44: BFA00014 <nearest+0x4c98>
	v_cndmask_b32_e64 v34, v74, -v2, s2                        // 000000004C48: D5010022 400A054A
	v_cndmask_b32_e64 v35, v73, -v3, s3                        // 000000004C50: D5010023 400E0749
	v_cndmask_b32_e64 v36, v72, v18, s2                        // 000000004C58: D5010024 000A2548
	v_cndmask_b32_e64 v37, v71, v19, s3                        // 000000004C60: D5010025 000E2747
	v_cndmask_b32_e64 v68, v68, v74, s2                        // 000000004C68: D5010044 000A9544
	v_cndmask_b32_e64 v69, v69, v73, s3                        // 000000004C70: D5010045 000E9345
	v_cndmask_b32_e64 v66, v66, v72, s2                        // 000000004C78: D5010042 000A9142
	v_cndmask_b32_e64 v67, v67, v71, s3                        // 000000004C80: D5010043 000E8F43
	v_dual_mov_b32 v72, v36 :: v_dual_mov_b32 v71, v37         // 000000004C88: CA100124 48460125
	v_dual_mov_b32 v74, v34 :: v_dual_mov_b32 v73, v35         // 000000004C90: CA100122 4A480123
	s_wait_alu depctr_sa_sdst(0)                               // 000000004C98: BF88FF9E
	s_cmp_lt_f32 s5, 0.5                                       // 000000004C9C: BF41F005
	v_cmp_ge_f32_e64 s3, v32, -v4                              // 000000004CA0: D4160003 42020920
	v_cmp_ge_f32_e64 s4, v32, -v5                              // 000000004CA8: D4160004 42020B20
	s_cselect_b32 s2, -1, 0                                    // 000000004CB0: 980280C1
	s_cmp_le_f32 s10, 2.0                                      // 000000004CB4: BF43F40A
	s_cselect_b32 s7, -1, 0                                    // 000000004CB8: 980780C1
	s_wait_alu depctr_sa_sdst(0)                               // 000000004CBC: BF88FF9E
	s_and_b32 s2, s7, s2                                       // 000000004CC0: 8B020207
	s_wait_alu depctr_sa_sdst(0)                               // 000000004CC4: BF88FF9E
	s_and_b32 s3, s3, s2                                       // 000000004CC8: 8B030203
	s_and_b32 s7, s4, s2                                       // 000000004CCC: 8B070204
	s_wait_alu depctr_sa_sdst(0)                               // 000000004CD0: BF88FF9E
	s_or_b32 vcc_lo, s3, s7                                    // 000000004CD4: 8C6A0703
	s_wait_alu depctr_sa_sdst(0)                               // 000000004CD8: BF88FF9E
	s_cbranch_vccz 49                                          // 000000004CDC: BFA30031 <nearest+0x4da4>
	v_cmp_gt_f32_e64 s8, v32, -v4                              // 000000004CE0: D4140008 42020920
	v_cmp_gt_f32_e32 vcc_lo, v33, v20                          // 000000004CE8: 7C282921
	v_cmp_gt_f32_e64 s15, v32, -v5                             // 000000004CEC: D414000F 42020B20
	v_cmp_gt_f32_e64 s2, v33, v21                              // 000000004CF4: D4140002 02022B21
	v_cmp_lt_f32_e64 s4, -1.0, v66                             // 000000004CFC: D4110004 020284F3
	s_or_b32 s8, s8, vcc_lo                                    // 000000004D04: 8C086A08
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v67                         // 000000004D08: 7C2286F3
	s_or_b32 s15, s15, s2                                      // 000000004D0C: 8C0F020F
	s_and_b32 s2, s8, s3                                       // 000000004D10: 8B020308
	s_wait_alu depctr_sa_sdst(0)                               // 000000004D14: BF88FF9E
	s_and_b32 s3, s15, s7                                      // 000000004D18: 8B03070F
	s_and_b32 s4, s4, s2                                       // 000000004D1C: 8B040204
	s_wait_alu depctr_sa_sdst(0)                               // 000000004D20: BF88FF9E
	s_and_b32 s7, vcc_lo, s3                                   // 000000004D24: 8B07036A
	s_wait_alu depctr_sa_sdst(0)                               // 000000004D28: BF88FF9E
	s_or_b32 s4, s4, s7                                        // 000000004D2C: 8C040704
	s_wait_alu depctr_sa_sdst(0)                               // 000000004D30: BF88FF9E
	v_cndmask_b32_e64 v34, 0, 1, s4                            // 000000004D34: D5010022 00110280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000004D3C: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v34                            // 000000004D40: 7C9A4480
	s_cbranch_vccz 3                                           // 000000004D44: BFA30003 <nearest+0x4d54>
	s_mov_b32 s6, 2.0                                          // 000000004D48: BE8600F4
	s_mov_b32 s5, 1.0                                          // 000000004D4C: BE8500F2
	s_branch 20                                                // 000000004D50: BFA00014 <nearest+0x4da4>
	v_cndmask_b32_e64 v34, v74, -v4, s2                        // 000000004D54: D5010022 400A094A
	v_cndmask_b32_e64 v35, v73, -v5, s3                        // 000000004D5C: D5010023 400E0B49
	v_cndmask_b32_e64 v36, v72, v20, s2                        // 000000004D64: D5010024 000A2948
	v_cndmask_b32_e64 v37, v71, v21, s3                        // 000000004D6C: D5010025 000E2B47
	v_cndmask_b32_e64 v68, v68, v74, s2                        // 000000004D74: D5010044 000A9544
	v_cndmask_b32_e64 v69, v69, v73, s3                        // 000000004D7C: D5010045 000E9345
	v_cndmask_b32_e64 v66, v66, v72, s2                        // 000000004D84: D5010042 000A9142
	v_cndmask_b32_e64 v67, v67, v71, s3                        // 000000004D8C: D5010043 000E8F43
	v_dual_mov_b32 v72, v36 :: v_dual_mov_b32 v71, v37         // 000000004D94: CA100124 48460125
	v_dual_mov_b32 v74, v34 :: v_dual_mov_b32 v73, v35         // 000000004D9C: CA100122 4A480123
	s_wait_alu depctr_sa_sdst(0)                               // 000000004DA4: BF88FF9E
	s_cmp_lt_f32 s5, 0.5                                       // 000000004DA8: BF41F005
	v_cmp_ge_f32_e64 s3, v32, -v6                              // 000000004DAC: D4160003 42020D20
	v_cmp_ge_f32_e64 s4, v32, -v7                              // 000000004DB4: D4160004 42020F20
	s_cselect_b32 s2, -1, 0                                    // 000000004DBC: 980280C1
	s_cmp_le_f32 s10, 0x40400000                               // 000000004DC0: BF43FF0A 40400000
	s_cselect_b32 s7, -1, 0                                    // 000000004DC8: 980780C1
	s_wait_alu depctr_sa_sdst(0)                               // 000000004DCC: BF88FF9E
	s_and_b32 s2, s7, s2                                       // 000000004DD0: 8B020207
	s_wait_alu depctr_sa_sdst(0)                               // 000000004DD4: BF88FF9E
	s_and_b32 s3, s3, s2                                       // 000000004DD8: 8B030203
	s_and_b32 s7, s4, s2                                       // 000000004DDC: 8B070204
	s_wait_alu depctr_sa_sdst(0)                               // 000000004DE0: BF88FF9E
	s_or_b32 vcc_lo, s3, s7                                    // 000000004DE4: 8C6A0703
	s_wait_alu depctr_sa_sdst(0)                               // 000000004DE8: BF88FF9E
	s_cbranch_vccz 50                                          // 000000004DEC: BFA30032 <nearest+0x4eb8>
	v_cmp_gt_f32_e64 s8, v32, -v6                              // 000000004DF0: D4140008 42020D20
	v_cmp_gt_f32_e32 vcc_lo, v33, v22                          // 000000004DF8: 7C282D21
	v_cmp_gt_f32_e64 s15, v32, -v7                             // 000000004DFC: D414000F 42020F20
	v_cmp_gt_f32_e64 s2, v33, v23                              // 000000004E04: D4140002 02022F21
	v_cmp_lt_f32_e64 s4, -1.0, v66                             // 000000004E0C: D4110004 020284F3
	s_or_b32 s8, s8, vcc_lo                                    // 000000004E14: 8C086A08
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v67                         // 000000004E18: 7C2286F3
	s_or_b32 s15, s15, s2                                      // 000000004E1C: 8C0F020F
	s_and_b32 s2, s8, s3                                       // 000000004E20: 8B020308
	s_wait_alu depctr_sa_sdst(0)                               // 000000004E24: BF88FF9E
	s_and_b32 s3, s15, s7                                      // 000000004E28: 8B03070F
	s_and_b32 s4, s4, s2                                       // 000000004E2C: 8B040204
	s_wait_alu depctr_sa_sdst(0)                               // 000000004E30: BF88FF9E
	s_and_b32 s7, vcc_lo, s3                                   // 000000004E34: 8B07036A
	s_wait_alu depctr_sa_sdst(0)                               // 000000004E38: BF88FF9E
	s_or_b32 s4, s4, s7                                        // 000000004E3C: 8C040704
	s_wait_alu depctr_sa_sdst(0)                               // 000000004E40: BF88FF9E
	v_cndmask_b32_e64 v34, 0, 1, s4                            // 000000004E44: D5010022 00110280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000004E4C: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v34                            // 000000004E50: 7C9A4480
	s_cbranch_vccz 4                                           // 000000004E54: BFA30004 <nearest+0x4e68>
	s_mov_b32 s6, 0x40400000                                   // 000000004E58: BE8600FF 40400000
	s_mov_b32 s5, 1.0                                          // 000000004E60: BE8500F2
	s_branch 20                                                // 000000004E64: BFA00014 <nearest+0x4eb8>
	v_cndmask_b32_e64 v34, v74, -v6, s2                        // 000000004E68: D5010022 400A0D4A
	v_cndmask_b32_e64 v35, v73, -v7, s3                        // 000000004E70: D5010023 400E0F49
	v_cndmask_b32_e64 v36, v72, v22, s2                        // 000000004E78: D5010024 000A2D48
	v_cndmask_b32_e64 v37, v71, v23, s3                        // 000000004E80: D5010025 000E2F47
	v_cndmask_b32_e64 v68, v68, v74, s2                        // 000000004E88: D5010044 000A9544
	v_cndmask_b32_e64 v69, v69, v73, s3                        // 000000004E90: D5010045 000E9345
	v_cndmask_b32_e64 v66, v66, v72, s2                        // 000000004E98: D5010042 000A9142
	v_cndmask_b32_e64 v67, v67, v71, s3                        // 000000004EA0: D5010043 000E8F43
	v_dual_mov_b32 v72, v36 :: v_dual_mov_b32 v71, v37         // 000000004EA8: CA100124 48460125
	v_dual_mov_b32 v74, v34 :: v_dual_mov_b32 v73, v35         // 000000004EB0: CA100122 4A480123
	s_wait_alu depctr_sa_sdst(0)                               // 000000004EB8: BF88FF9E
	s_cmp_lt_f32 s5, 0.5                                       // 000000004EBC: BF41F005
	v_cmp_ge_f32_e64 s3, v32, -v8                              // 000000004EC0: D4160003 42021120
	v_cmp_ge_f32_e64 s4, v32, -v9                              // 000000004EC8: D4160004 42021320
	s_cselect_b32 s2, -1, 0                                    // 000000004ED0: 980280C1
	s_cmp_le_f32 s10, 4.0                                      // 000000004ED4: BF43F60A
	s_cselect_b32 s7, -1, 0                                    // 000000004ED8: 980780C1
	s_wait_alu depctr_sa_sdst(0)                               // 000000004EDC: BF88FF9E
	s_and_b32 s2, s7, s2                                       // 000000004EE0: 8B020207
	s_wait_alu depctr_sa_sdst(0)                               // 000000004EE4: BF88FF9E
	s_and_b32 s3, s3, s2                                       // 000000004EE8: 8B030203
	s_and_b32 s7, s4, s2                                       // 000000004EEC: 8B070204
	s_wait_alu depctr_sa_sdst(0)                               // 000000004EF0: BF88FF9E
	s_or_b32 vcc_lo, s3, s7                                    // 000000004EF4: 8C6A0703
	s_wait_alu depctr_sa_sdst(0)                               // 000000004EF8: BF88FF9E
	s_cbranch_vccz 49                                          // 000000004EFC: BFA30031 <nearest+0x4fc4>
	v_cmp_gt_f32_e64 s8, v32, -v8                              // 000000004F00: D4140008 42021120
	v_cmp_gt_f32_e32 vcc_lo, v33, v24                          // 000000004F08: 7C283121
	v_cmp_gt_f32_e64 s15, v32, -v9                             // 000000004F0C: D414000F 42021320
	v_cmp_gt_f32_e64 s2, v33, v25                              // 000000004F14: D4140002 02023321
	v_cmp_lt_f32_e64 s4, -1.0, v66                             // 000000004F1C: D4110004 020284F3
	s_or_b32 s8, s8, vcc_lo                                    // 000000004F24: 8C086A08
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v67                         // 000000004F28: 7C2286F3
	s_or_b32 s15, s15, s2                                      // 000000004F2C: 8C0F020F
	s_and_b32 s2, s8, s3                                       // 000000004F30: 8B020308
	s_wait_alu depctr_sa_sdst(0)                               // 000000004F34: BF88FF9E
	s_and_b32 s3, s15, s7                                      // 000000004F38: 8B03070F
	s_and_b32 s4, s4, s2                                       // 000000004F3C: 8B040204
	s_wait_alu depctr_sa_sdst(0)                               // 000000004F40: BF88FF9E
	s_and_b32 s7, vcc_lo, s3                                   // 000000004F44: 8B07036A
	s_wait_alu depctr_sa_sdst(0)                               // 000000004F48: BF88FF9E
	s_or_b32 s4, s4, s7                                        // 000000004F4C: 8C040704
	s_wait_alu depctr_sa_sdst(0)                               // 000000004F50: BF88FF9E
	v_cndmask_b32_e64 v34, 0, 1, s4                            // 000000004F54: D5010022 00110280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000004F5C: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v34                            // 000000004F60: 7C9A4480
	s_cbranch_vccz 3                                           // 000000004F64: BFA30003 <nearest+0x4f74>
	s_mov_b32 s6, 4.0                                          // 000000004F68: BE8600F6
	s_mov_b32 s5, 1.0                                          // 000000004F6C: BE8500F2
	s_branch 20                                                // 000000004F70: BFA00014 <nearest+0x4fc4>
	v_cndmask_b32_e64 v34, v74, -v8, s2                        // 000000004F74: D5010022 400A114A
	v_cndmask_b32_e64 v35, v73, -v9, s3                        // 000000004F7C: D5010023 400E1349
	v_cndmask_b32_e64 v36, v72, v24, s2                        // 000000004F84: D5010024 000A3148
	v_cndmask_b32_e64 v37, v71, v25, s3                        // 000000004F8C: D5010025 000E3347
	v_cndmask_b32_e64 v68, v68, v74, s2                        // 000000004F94: D5010044 000A9544
	v_cndmask_b32_e64 v69, v69, v73, s3                        // 000000004F9C: D5010045 000E9345
	v_cndmask_b32_e64 v66, v66, v72, s2                        // 000000004FA4: D5010042 000A9142
	v_cndmask_b32_e64 v67, v67, v71, s3                        // 000000004FAC: D5010043 000E8F43
	v_dual_mov_b32 v72, v36 :: v_dual_mov_b32 v71, v37         // 000000004FB4: CA100124 48460125
	v_dual_mov_b32 v74, v34 :: v_dual_mov_b32 v73, v35         // 000000004FBC: CA100122 4A480123
	s_wait_alu depctr_sa_sdst(0)                               // 000000004FC4: BF88FF9E
	s_cmp_lt_f32 s5, 0.5                                       // 000000004FC8: BF41F005
	v_cmp_ge_f32_e64 s3, v32, -v10                             // 000000004FCC: D4160003 42021520
	v_cmp_ge_f32_e64 s4, v32, -v11                             // 000000004FD4: D4160004 42021720
	s_cselect_b32 s2, -1, 0                                    // 000000004FDC: 980280C1
	s_cmp_le_f32 s10, 0x40a00000                               // 000000004FE0: BF43FF0A 40A00000
	s_cselect_b32 s7, -1, 0                                    // 000000004FE8: 980780C1
	s_wait_alu depctr_sa_sdst(0)                               // 000000004FEC: BF88FF9E
	s_and_b32 s2, s7, s2                                       // 000000004FF0: 8B020207
	s_wait_alu depctr_sa_sdst(0)                               // 000000004FF4: BF88FF9E
	s_and_b32 s3, s3, s2                                       // 000000004FF8: 8B030203
	s_and_b32 s7, s4, s2                                       // 000000004FFC: 8B070204
	s_wait_alu depctr_sa_sdst(0)                               // 000000005000: BF88FF9E
	s_or_b32 vcc_lo, s3, s7                                    // 000000005004: 8C6A0703
	s_wait_alu depctr_sa_sdst(0)                               // 000000005008: BF88FF9E
	s_cbranch_vccz 50                                          // 00000000500C: BFA30032 <nearest+0x50d8>
	v_cmp_gt_f32_e64 s8, v32, -v10                             // 000000005010: D4140008 42021520
	v_cmp_gt_f32_e32 vcc_lo, v33, v26                          // 000000005018: 7C283521
	v_cmp_gt_f32_e64 s15, v32, -v11                            // 00000000501C: D414000F 42021720
	v_cmp_gt_f32_e64 s2, v33, v27                              // 000000005024: D4140002 02023721
	v_cmp_lt_f32_e64 s4, -1.0, v66                             // 00000000502C: D4110004 020284F3
	s_or_b32 s8, s8, vcc_lo                                    // 000000005034: 8C086A08
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v67                         // 000000005038: 7C2286F3
	s_or_b32 s15, s15, s2                                      // 00000000503C: 8C0F020F
	s_and_b32 s2, s8, s3                                       // 000000005040: 8B020308
	s_wait_alu depctr_sa_sdst(0)                               // 000000005044: BF88FF9E
	s_and_b32 s3, s15, s7                                      // 000000005048: 8B03070F
	s_and_b32 s4, s4, s2                                       // 00000000504C: 8B040204
	s_wait_alu depctr_sa_sdst(0)                               // 000000005050: BF88FF9E
	s_and_b32 s7, vcc_lo, s3                                   // 000000005054: 8B07036A
	s_wait_alu depctr_sa_sdst(0)                               // 000000005058: BF88FF9E
	s_or_b32 s4, s4, s7                                        // 00000000505C: 8C040704
	s_wait_alu depctr_sa_sdst(0)                               // 000000005060: BF88FF9E
	v_cndmask_b32_e64 v34, 0, 1, s4                            // 000000005064: D5010022 00110280
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000506C: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v34                            // 000000005070: 7C9A4480
	s_cbranch_vccz 4                                           // 000000005074: BFA30004 <nearest+0x5088>
	s_mov_b32 s6, 0x40a00000                                   // 000000005078: BE8600FF 40A00000
	s_mov_b32 s5, 1.0                                          // 000000005080: BE8500F2
	s_branch 20                                                // 000000005084: BFA00014 <nearest+0x50d8>
	v_cndmask_b32_e64 v34, v74, -v10, s2                       // 000000005088: D5010022 400A154A
	v_cndmask_b32_e64 v35, v73, -v11, s3                       // 000000005090: D5010023 400E1749
	v_cndmask_b32_e64 v36, v72, v26, s2                        // 000000005098: D5010024 000A3548
	v_cndmask_b32_e64 v37, v71, v27, s3                        // 0000000050A0: D5010025 000E3747
	v_cndmask_b32_e64 v68, v68, v74, s2                        // 0000000050A8: D5010044 000A9544
	v_cndmask_b32_e64 v69, v69, v73, s3                        // 0000000050B0: D5010045 000E9345
	v_cndmask_b32_e64 v66, v66, v72, s2                        // 0000000050B8: D5010042 000A9142
	v_cndmask_b32_e64 v67, v67, v71, s3                        // 0000000050C0: D5010043 000E8F43
	v_dual_mov_b32 v72, v36 :: v_dual_mov_b32 v71, v37         // 0000000050C8: CA100124 48460125
	v_dual_mov_b32 v74, v34 :: v_dual_mov_b32 v73, v35         // 0000000050D0: CA100122 4A480123
	s_wait_alu depctr_sa_sdst(0)                               // 0000000050D8: BF88FF9E
	s_cmp_lt_f32 s5, 0.5                                       // 0000000050DC: BF41F005
	v_cmp_ge_f32_e64 s3, v32, -v12                             // 0000000050E0: D4160003 42021920
	v_cmp_ge_f32_e64 s4, v32, -v13                             // 0000000050E8: D4160004 42021B20
	s_cselect_b32 s2, -1, 0                                    // 0000000050F0: 980280C1
	s_cmp_le_f32 s10, 0x40c00000                               // 0000000050F4: BF43FF0A 40C00000
	s_cselect_b32 s7, -1, 0                                    // 0000000050FC: 980780C1
	s_wait_alu depctr_sa_sdst(0)                               // 000000005100: BF88FF9E
	s_and_b32 s2, s7, s2                                       // 000000005104: 8B020207
	s_wait_alu depctr_sa_sdst(0)                               // 000000005108: BF88FF9E
	s_and_b32 s3, s3, s2                                       // 00000000510C: 8B030203
	s_and_b32 s7, s4, s2                                       // 000000005110: 8B070204
	s_wait_alu depctr_sa_sdst(0)                               // 000000005114: BF88FF9E
	s_or_b32 vcc_lo, s3, s7                                    // 000000005118: 8C6A0703
	s_wait_alu depctr_sa_sdst(0)                               // 00000000511C: BF88FF9E
	s_cbranch_vccz 50                                          // 000000005120: BFA30032 <nearest+0x51ec>
	v_cmp_gt_f32_e64 s8, v32, -v12                             // 000000005124: D4140008 42021920
	v_cmp_gt_f32_e32 vcc_lo, v33, v28                          // 00000000512C: 7C283921
	v_cmp_gt_f32_e64 s15, v32, -v13                            // 000000005130: D414000F 42021B20
	v_cmp_gt_f32_e64 s2, v33, v29                              // 000000005138: D4140002 02023B21
	v_cmp_lt_f32_e64 s4, -1.0, v66                             // 000000005140: D4110004 020284F3
	s_or_b32 s8, s8, vcc_lo                                    // 000000005148: 8C086A08
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v67                         // 00000000514C: 7C2286F3
	s_or_b32 s15, s15, s2                                      // 000000005150: 8C0F020F
	s_and_b32 s2, s8, s3                                       // 000000005154: 8B020308
	s_wait_alu depctr_sa_sdst(0)                               // 000000005158: BF88FF9E
	s_and_b32 s3, s15, s7                                      // 00000000515C: 8B03070F
	s_and_b32 s4, s4, s2                                       // 000000005160: 8B040204
	s_wait_alu depctr_sa_sdst(0)                               // 000000005164: BF88FF9E
	s_and_b32 s7, vcc_lo, s3                                   // 000000005168: 8B07036A
	s_wait_alu depctr_sa_sdst(0)                               // 00000000516C: BF88FF9E
	s_or_b32 s4, s4, s7                                        // 000000005170: 8C040704
	s_wait_alu depctr_sa_sdst(0)                               // 000000005174: BF88FF9E
	v_cndmask_b32_e64 v34, 0, 1, s4                            // 000000005178: D5010022 00110280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000005180: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v34                            // 000000005184: 7C9A4480
	s_cbranch_vccz 4                                           // 000000005188: BFA30004 <nearest+0x519c>
	s_mov_b32 s6, 0x40c00000                                   // 00000000518C: BE8600FF 40C00000
	s_mov_b32 s5, 1.0                                          // 000000005194: BE8500F2
	s_branch 20                                                // 000000005198: BFA00014 <nearest+0x51ec>
	v_cndmask_b32_e64 v34, v74, -v12, s2                       // 00000000519C: D5010022 400A194A
	v_cndmask_b32_e64 v35, v73, -v13, s3                       // 0000000051A4: D5010023 400E1B49
	v_cndmask_b32_e64 v36, v72, v28, s2                        // 0000000051AC: D5010024 000A3948
	v_cndmask_b32_e64 v37, v71, v29, s3                        // 0000000051B4: D5010025 000E3B47
	v_cndmask_b32_e64 v68, v68, v74, s2                        // 0000000051BC: D5010044 000A9544
	v_cndmask_b32_e64 v69, v69, v73, s3                        // 0000000051C4: D5010045 000E9345
	v_cndmask_b32_e64 v66, v66, v72, s2                        // 0000000051CC: D5010042 000A9142
	v_cndmask_b32_e64 v67, v67, v71, s3                        // 0000000051D4: D5010043 000E8F43
	v_dual_mov_b32 v72, v36 :: v_dual_mov_b32 v71, v37         // 0000000051DC: CA100124 48460125
	v_dual_mov_b32 v74, v34 :: v_dual_mov_b32 v73, v35         // 0000000051E4: CA100122 4A480123
	s_wait_alu depctr_sa_sdst(0)                               // 0000000051EC: BF88FF9E
	s_cmp_lt_f32 s5, 0.5                                       // 0000000051F0: BF41F005
	v_cmp_ge_f32_e64 s3, v32, -v14                             // 0000000051F4: D4160003 42021D20
	v_cmp_ge_f32_e64 s4, v32, -v15                             // 0000000051FC: D4160004 42021F20
	s_cselect_b32 s2, -1, 0                                    // 000000005204: 980280C1
	s_cmp_le_f32 s10, 0x40e00000                               // 000000005208: BF43FF0A 40E00000
	s_cselect_b32 s7, -1, 0                                    // 000000005210: 980780C1
	s_wait_alu depctr_sa_sdst(0)                               // 000000005214: BF88FF9E
	s_and_b32 s2, s7, s2                                       // 000000005218: 8B020207
	s_wait_alu depctr_sa_sdst(0)                               // 00000000521C: BF88FF9E
	s_and_b32 s3, s3, s2                                       // 000000005220: 8B030203
	s_and_b32 s7, s4, s2                                       // 000000005224: 8B070204
	s_wait_alu depctr_sa_sdst(0)                               // 000000005228: BF88FF9E
	s_or_b32 vcc_lo, s3, s7                                    // 00000000522C: 8C6A0703
	s_wait_alu depctr_sa_sdst(0)                               // 000000005230: BF88FF9E
	s_cbranch_vccz 29                                          // 000000005234: BFA3001D <nearest+0x52ac>
	v_cmp_gt_f32_e64 s8, v32, -v14                             // 000000005238: D4140008 42021D20
	v_cmp_gt_f32_e32 vcc_lo, v33, v30                          // 000000005240: 7C283D21
	v_cmp_gt_f32_e64 s10, v32, -v15                            // 000000005244: D414000A 42021F20
	v_cmp_gt_f32_e64 s2, v33, v31                              // 00000000524C: D4140002 02023F21
	v_cmp_lt_f32_e64 s4, -1.0, v66                             // 000000005254: D4110004 020284F3
	s_or_b32 s8, s8, vcc_lo                                    // 00000000525C: 8C086A08
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v67                         // 000000005260: 7C2286F3
	s_or_b32 s10, s10, s2                                      // 000000005264: 8C0A020A
	s_and_b32 s2, s8, s3                                       // 000000005268: 8B020308
	s_wait_alu depctr_sa_sdst(0)                               // 00000000526C: BF88FF9E
	s_and_b32 s3, s10, s7                                      // 000000005270: 8B03070A
	s_and_b32 s4, s4, s2                                       // 000000005274: 8B040204
	s_wait_alu depctr_sa_sdst(0)                               // 000000005278: BF88FF9E
	s_and_b32 s7, vcc_lo, s3                                   // 00000000527C: 8B07036A
	s_wait_alu depctr_sa_sdst(0)                               // 000000005280: BF88FF9E
	s_or_b32 s4, s4, s7                                        // 000000005284: 8C040704
	s_wait_alu depctr_sa_sdst(0)                               // 000000005288: BF88FF9E
	v_cndmask_b32_e64 v34, 0, 1, s4                            // 00000000528C: D5010022 00110280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000005294: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v34                            // 000000005298: 7C9A4480
	s_cbranch_vccz 846                                         // 00000000529C: BFA3034E <nearest+0x5fd8>
	s_mov_b32 s6, 0x40e00000                                   // 0000000052A0: BE8600FF 40E00000
	s_mov_b32 s5, 1.0                                          // 0000000052A8: BE8500F2
	s_wait_alu depctr_sa_sdst(0)                               // 0000000052AC: BF88FF9E
	s_cmp_ngt_f32 s5, 0.5                                      // 0000000052B0: BF4BF005
	s_mov_b32 s10, s6                                          // 0000000052B4: BE8A0006
	s_cbranch_scc1 65000                                       // 0000000052B8: BFA2FDE8 <nearest+0x4a5c>
	v_cmp_eq_f32_e32 vcc_lo, v73, v74                          // 0000000052BC: 7C249549
	v_cmp_lt_f32_e64 s2, v71, v72                              // 0000000052C0: D4110002 02029147
	v_cmp_lt_f32_e64 s3, v72, v71                              // 0000000052C8: D4110003 02028F48
	v_cmp_lt_f32_e64 s4, v73, v74                              // 0000000052D0: D4110004 02029549
	v_cmp_lt_f32_e64 s5, v74, v73                              // 0000000052D8: D4110005 0202934A
	v_cmp_lt_f32_e64 s6, v68, v69                              // 0000000052E0: D4110006 02028B44
	s_and_b32 s2, vcc_lo, s2                                   // 0000000052E8: 8B02026A
	s_and_b32 s3, vcc_lo, s3                                   // 0000000052EC: 8B03036A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000052F0: BF88FF9E
	s_or_b32 s2, s4, s2                                        // 0000000052F4: 8C020204
	s_or_b32 s3, s5, s3                                        // 0000000052F8: 8C030305
	s_wait_alu depctr_sa_sdst(0)                               // 0000000052FC: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000005300: 8D6A0002
	s_xor_b32 s2, s3, s9                                       // 000000005304: 8D020903
	s_wait_alu depctr_sa_sdst(0)                               // 000000005308: BF88FF9E
	v_cndmask_b32_e32 v32, v74, v73, vcc_lo                    // 00000000530C: 0240934A
	v_cndmask_b32_e32 v34, v72, v71, vcc_lo                    // 000000005310: 02448F48
	v_cmp_eq_f32_e32 vcc_lo, v69, v68                          // 000000005314: 7C248945
	v_cmp_lt_f32_e64 s3, v67, v66                              // 000000005318: D4110003 02028543
	v_cmp_lt_f32_e64 s5, v69, v68                              // 000000005320: D4110005 02028945
	v_cmp_lt_f32_e64 s4, v66, v67                              // 000000005328: D4110004 02028742
	v_cndmask_b32_e64 v33, v73, v74, s2                        // 000000005330: D5010021 000A9549
	v_cndmask_b32_e64 v35, v71, v72, s2                        // 000000005338: D5010023 000A9147
	s_and_b32 s2, vcc_lo, s3                                   // 000000005340: 8B02036A
	v_mov_b32_e32 v37, v32                                     // 000000005344: 7E4A0320
	s_wait_alu depctr_sa_sdst(0)                               // 000000005348: BF88FF9E
	s_or_b32 s2, s5, s2                                        // 00000000534C: 8C020205
	s_and_b32 s3, vcc_lo, s4                                   // 000000005350: 8B03046A
	s_wait_alu depctr_sa_sdst(0)                               // 000000005354: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000005358: 8D6A0002
	v_mov_b32_e32 v38, v33                                     // 00000000535C: 7E4C0321
	s_wait_alu depctr_sa_sdst(0)                               // 000000005360: BF88FF9E
	v_cndmask_b32_e32 v42, v66, v67, vcc_lo                    // 000000005364: 02548742
	v_permlanex16_b32 v37, v37, s11, 0xfedcba98                // 000000005368: D65C0025 03FC1725 FEDCBA98
	s_or_b32 s3, s6, s3                                        // 000000005374: 8C030306
	v_mov_b32_e32 v74, 0x7f800000                              // 000000005378: 7E9402FF 7F800000
	s_wait_alu depctr_sa_sdst(0)                               // 000000005380: BF88FF9E
	s_xor_b32 s2, s3, s9                                       // 000000005384: 8D020903
	v_dual_mov_b32 v45, v42 :: v_dual_cndmask_b32 v36, v68, v69// 000000005388: CA12012A 2D248B44
	v_mov_b32_e32 v39, v34                                     // 000000005390: 7E4E0322
	v_cmp_eq_f32_e32 vcc_lo, v32, v37                          // 000000005394: 7C244B20
	v_cmp_gt_f32_e64 s6, v32, v37                              // 000000005398: D4140006 02024B20
	s_wait_alu depctr_sa_sdst(0)                               // 0000000053A0: BF88FF9E
	v_cndmask_b32_e64 v41, v69, v68, s2                        // 0000000053A4: D5010029 000A8945
	v_cndmask_b32_e64 v43, v67, v66, s2                        // 0000000053AC: D501002B 000A8543
	v_permlanex16_b32 v39, v39, s11, 0xfedcba98                // 0000000053B4: D65C0027 03FC1727 FEDCBA98
	v_permlanex16_b32 v38, v38, s11, 0xfedcba98                // 0000000053C0: D65C0026 03FC1726 FEDCBA98
	v_permlanex16_b32 v45, v45, s11, 0xfedcba98                // 0000000053CC: D65C002D 03FC172D FEDCBA98
	v_mov_b32_e32 v44, v41                                     // 0000000053D8: 7E580329
	v_mov_b32_e32 v46, v43                                     // 0000000053DC: 7E5C032B
	v_cmp_gt_f32_e64 s4, v34, v39                              // 0000000053E0: D4140004 02024F22
	v_cmp_eq_f32_e64 s3, v33, v38                              // 0000000053E8: D4120003 02024D21
	v_cmp_gt_f32_e64 s7, v33, v38                              // 0000000053F0: D4140007 02024D21
	v_permlanex16_b32 v44, v44, s11, 0xfedcba98                // 0000000053F8: D65C002C 03FC172C FEDCBA98
	v_permlanex16_b32 v46, v46, s11, 0xfedcba98                // 000000005404: D65C002E 03FC172E FEDCBA98
	s_and_b32 s2, vcc_lo, s4                                   // 000000005410: 8B02046A
	v_mov_b32_e32 v72, -1.0                                    // 000000005414: 7E9002F3
	s_wait_alu depctr_sa_sdst(0)                               // 000000005418: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 00000000541C: 8C020206
	v_cmp_neq_f32_e64 s4, v41, v44                             // 000000005420: D41D0004 02025929
	s_wait_alu depctr_sa_sdst(0)                               // 000000005428: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 00000000542C: 8D6A0002
	v_mov_b32_e32 v68, 0x7f800000                              // 000000005430: 7E8802FF 7F800000
	s_wait_alu depctr_sa_sdst(0)                               // 000000005438: BF88FF9E
	v_cndmask_b32_e32 v34, v34, v39, vcc_lo                    // 00000000543C: 02444F22
	v_mov_b32_e32 v40, v35                                     // 000000005440: 7E500323
	v_cndmask_b32_e32 v32, v32, v37, vcc_lo                    // 000000005444: 02404B20
	v_dual_mov_b32 v37, v36 :: v_dual_mov_b32 v66, -1.0        // 000000005448: CA100124 254200F3
	v_mov_b32_e32 v71, -1.0                                    // 000000005450: 7E8E02F3
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 000000005454: BF870224
	v_permlanex16_b32 v40, v40, s11, 0xfedcba98                // 000000005458: D65C0028 03FC1728 FEDCBA98
	v_mov_b32_e32 v73, 0x7f800000                              // 000000005464: 7E9202FF 7F800000
	v_permlanex16_b32 v37, v37, s11, 0xfedcba98                // 00000000546C: D65C0025 03FC1725 FEDCBA98
	v_mov_b32_e32 v67, -1.0                                    // 000000005478: 7E8602F3
	v_mov_b32_e32 v69, 0x7f800000                              // 00000000547C: 7E8A02FF 7F800000
	v_cmp_gt_f32_e64 s5, v35, v40                              // 000000005484: D4140005 02025123
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000548C: BF870004
	v_cmp_neq_f32_e32 vcc_lo, v36, v37                         // 000000005490: 7C3A4B24
	v_cmp_ngt_f32_e64 s6, v36, v37                             // 000000005494: D41B0006 02024B24
	s_and_b32 s3, s3, s5                                       // 00000000549C: 8B030503
	v_cmp_ngt_f32_e64 s5, v43, v46                             // 0000000054A0: D41B0005 02025D2B
	s_wait_alu depctr_sa_sdst(0)                               // 0000000054A8: BF88FF9E
	s_or_b32 s3, s7, s3                                        // 0000000054AC: 8C030307
	v_cmp_ngt_f32_e64 s7, v41, v44                             // 0000000054B0: D41B0007 02025929
	s_wait_alu depctr_sa_sdst(0)                               // 0000000054B8: BF88FF9E
	s_xor_b32 s2, s3, s0                                       // 0000000054BC: 8D020003
	v_cmp_ngt_f32_e64 s3, v42, v45                             // 0000000054C0: D41B0003 02025B2A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000054C8: BF88FF9E
	v_cndmask_b32_e64 v33, v33, v38, s2                        // 0000000054CC: D5010021 000A4D21
	v_cndmask_b32_e64 v35, v35, v40, s2                        // 0000000054D4: D5010023 000A5123
	s_or_b32 s2, vcc_lo, s3                                    // 0000000054DC: 8C02036A
	s_or_b32 s3, s4, s5                                        // 0000000054E0: 8C030504
	s_wait_alu depctr_sa_sdst(0)                               // 0000000054E4: BF88FF9E
	s_and_b32 s2, s6, s2                                       // 0000000054E8: 8B020206
	s_and_b32 s3, s7, s3                                       // 0000000054EC: 8B030307
	s_wait_alu depctr_sa_sdst(0)                               // 0000000054F0: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 0000000054F4: 8D6A0002
	s_xor_b32 s2, s3, s0                                       // 0000000054F8: 8D020003
	s_wait_alu depctr_sa_sdst(0)                               // 0000000054FC: BF88FF9E
	v_cndmask_b32_e32 v36, v36, v37, vcc_lo                    // 000000005500: 02484B24
	v_cndmask_b32_e32 v38, v42, v45, vcc_lo                    // 000000005504: 024C5B2A
	v_cmp_eq_f32_e32 vcc_lo, v33, v32                          // 000000005508: 7C244121
	v_cmp_lt_f32_e64 s3, v35, v34                              // 00000000550C: D4110003 02024523
	v_cmp_lt_f32_e64 s4, v34, v35                              // 000000005514: D4110004 02024722
	v_cndmask_b32_e64 v37, v41, v44, s2                        // 00000000551C: D5010025 000A5929
	v_cmp_lt_f32_e64 s5, v33, v32                              // 000000005524: D4110005 02024121
	v_cndmask_b32_e64 v39, v43, v46, s2                        // 00000000552C: D5010027 000A5D2B
	v_cmp_lt_f32_e64 s2, v32, v33                              // 000000005534: D4110002 02024320
	s_and_b32 s3, vcc_lo, s3                                   // 00000000553C: 8B03036A
	s_and_b32 s4, vcc_lo, s4                                   // 000000005540: 8B04046A
	s_wait_alu depctr_sa_sdst(0)                               // 000000005544: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 000000005548: 8C6A0305
	v_cmp_lt_f32_e64 s3, v39, v38                              // 00000000554C: D4110003 02024D27
	s_or_b32 s2, s2, s4                                        // 000000005554: 8C020402
	s_wait_alu depctr_sa_sdst(0)                               // 000000005558: BF88FF9E
	v_cndmask_b32_e32 v40, v32, v33, vcc_lo                    // 00000000555C: 02504320
	v_cndmask_b32_e64 v32, v32, v33, s2                        // 000000005560: D5010020 000A4320
	v_cndmask_b32_e32 v33, v34, v35, vcc_lo                    // 000000005568: 02424722
	v_cmp_eq_f32_e32 vcc_lo, v37, v36                          // 00000000556C: 7C244925
	v_cmp_lt_f32_e64 s4, v38, v39                              // 000000005570: D4110004 02024F26
	v_cmp_lt_f32_e64 s5, v37, v36                              // 000000005578: D4110005 02024925
	v_cndmask_b32_e64 v34, v34, v35, s2                        // 000000005580: D5010022 000A4722
	v_cmp_lt_f32_e64 s2, v36, v37                              // 000000005588: D4110002 02024B24
	s_and_b32 s3, vcc_lo, s3                                   // 000000005590: 8B03036A
	s_and_b32 s4, vcc_lo, s4                                   // 000000005594: 8B04046A
	s_wait_alu depctr_sa_sdst(0)                               // 000000005598: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 00000000559C: 8C6A0305
	s_or_b32 s2, s2, s4                                        // 0000000055A0: 8C020402
	s_wait_alu depctr_sa_sdst(0)                               // 0000000055A4: BF88FF9E
	v_cndmask_b32_e32 v35, v37, v36, vcc_lo                    // 0000000055A8: 02464925
	v_cndmask_b32_e64 v36, v37, v36, s2                        // 0000000055AC: D5010024 000A4925
	v_cndmask_b32_e32 v37, v39, v38, vcc_lo                    // 0000000055B4: 024A4D27
	v_cndmask_b32_e64 v38, v39, v38, s2                        // 0000000055B8: D5010026 000A4D27
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 0000000055C0: BF870224
	v_cmp_eq_f32_e32 vcc_lo, v40, v35                          // 0000000055C4: 7C244728
	v_cmp_lt_f32_e64 s5, v40, v35                              // 0000000055C8: D4110005 02024728
	v_cmp_lt_f32_e64 s3, v33, v37                              // 0000000055D0: D4110003 02024B21
	v_cmp_eq_f32_e64 s2, v32, v36                              // 0000000055D8: D4120002 02024920
	v_cmp_lt_f32_e64 s4, v34, v38                              // 0000000055E0: D4110004 02024D22
	v_cmp_lt_f32_e64 s6, v32, v36                              // 0000000055E8: D4110006 02024920
	s_and_b32 s3, vcc_lo, s3                                   // 0000000055F0: 8B03036A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000055F4: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 0000000055F8: 8C6A0305
	s_and_b32 s2, s2, s4                                       // 0000000055FC: 8B020402
	s_wait_alu depctr_sa_sdst(0)                               // 000000005600: BF88FF9E
	v_dual_cndmask_b32 v39, v35, v40 :: v_dual_cndmask_b32 v42, v37, v33// 000000005604: CA525123 272A4325
	v_cndmask_b32_e32 v35, v40, v35, vcc_lo                    // 00000000560C: 02464728
	v_cndmask_b32_e32 v33, v33, v37, vcc_lo                    // 000000005610: 02424B21
	s_or_b32 s2, s6, s2                                        // 000000005614: 8C020206
	s_delay_alu instid0(VALU_DEP_3)                            // 000000005618: BF870003
	v_dual_mov_b32 v40, v39 :: v_dual_mov_b32 v45, v42         // 00000000561C: CA100127 282C012A
	s_wait_alu depctr_sa_sdst(0)                               // 000000005624: BF88FF9E
	v_cndmask_b32_e64 v41, v36, v32, s2                        // 000000005628: D5010029 000A4124
	v_cndmask_b32_e64 v43, v38, v34, s2                        // 000000005630: D501002B 000A4526
	v_cndmask_b32_e64 v32, v32, v36, s2                        // 000000005638: D5010020 000A4920
	v_permlanex16_b32 v40, v40, s11, 0xfedcba98                // 000000005640: D65C0028 03FC1728 FEDCBA98
	v_permlanex16_b32 v45, v45, s11, 0xfedcba98                // 00000000564C: D65C002D 03FC172D FEDCBA98
	v_cndmask_b32_e64 v34, v34, v38, s2                        // 000000005658: D5010022 000A4D22
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000005660: BF870214
	v_dual_mov_b32 v37, v35 :: v_dual_mov_b32 v38, v32         // 000000005664: CA100123 25260120
	v_cmp_eq_f32_e32 vcc_lo, v39, v40                          // 00000000566C: 7C245127
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 000000005670: BF870224
	v_cmp_gt_f32_e64 s4, v42, v45                              // 000000005674: D4140004 02025B2A
	v_cmp_gt_f32_e64 s6, v39, v40                              // 00000000567C: D4140006 02025127
	v_permlanex16_b32 v37, v37, s11, 0xfedcba98                // 000000005684: D65C0025 03FC1725 FEDCBA98
	v_mov_b32_e32 v44, v41                                     // 000000005690: 7E580329
	v_permlanex16_b32 v38, v38, s11, 0xfedcba98                // 000000005694: D65C0026 03FC1726 FEDCBA98
	s_and_b32 s2, vcc_lo, s4                                   // 0000000056A0: 8B02046A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000056A4: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 0000000056A8: 8C020206
	v_permlanex16_b32 v44, v44, s11, 0xfedcba98                // 0000000056AC: D65C002C 03FC172C FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 0000000056B8: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 0000000056BC: 8D6A0002
	v_cmp_gt_f32_e64 s6, v35, v37                              // 0000000056C0: D4140006 02024B23
	s_wait_alu depctr_sa_sdst(0)                               // 0000000056C8: BF88FF9E
	v_cndmask_b32_e32 v36, v39, v40, vcc_lo                    // 0000000056CC: 02485127
	v_dual_mov_b32 v46, v43 :: v_dual_mov_b32 v39, v33         // 0000000056D0: CA10012B 2E260121
	v_mov_b32_e32 v40, v34                                     // 0000000056D8: 7E500322
	v_cmp_eq_f32_e64 s3, v41, v44                              // 0000000056DC: D4120003 02025929
	v_cmp_gt_f32_e64 s7, v41, v44                              // 0000000056E4: D4140007 02025929
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000056EC: BF870004
	v_permlanex16_b32 v46, v46, s11, 0xfedcba98                // 0000000056F0: D65C002E 03FC172E FEDCBA98
	v_permlanex16_b32 v39, v39, s11, 0xfedcba98                // 0000000056FC: D65C0027 03FC1727 FEDCBA98
	v_permlanex16_b32 v40, v40, s11, 0xfedcba98                // 000000005708: D65C0028 03FC1728 FEDCBA98
	v_cndmask_b32_e32 v42, v42, v45, vcc_lo                    // 000000005714: 02545B2A
	v_cmp_eq_f32_e32 vcc_lo, v35, v37                          // 000000005718: 7C244B23
	v_cmp_gt_f32_e64 s5, v43, v46                              // 00000000571C: D4140005 02025D2B
	v_cmp_gt_f32_e64 s4, v33, v39                              // 000000005724: D4140004 02024F21
	s_and_b32 s3, s3, s5                                       // 00000000572C: 8B030503
	v_cmp_gt_f32_e64 s5, v34, v40                              // 000000005730: D4140005 02025122
	s_wait_alu depctr_sa_sdst(0)                               // 000000005738: BF88FF9E
	s_or_b32 s3, s7, s3                                        // 00000000573C: 8C030307
	v_cmp_gt_f32_e64 s7, v32, v38                              // 000000005740: D4140007 02024D20
	s_wait_alu depctr_sa_sdst(0)                               // 000000005748: BF88FF9E
	s_xor_b32 s2, s3, s0                                       // 00000000574C: 8D020003
	v_cmp_eq_f32_e64 s3, v32, v38                              // 000000005750: D4120003 02024D20
	s_wait_alu depctr_sa_sdst(0)                               // 000000005758: BF88FF9E
	v_cndmask_b32_e64 v41, v41, v44, s2                        // 00000000575C: D5010029 000A5929
	v_cndmask_b32_e64 v43, v43, v46, s2                        // 000000005764: D501002B 000A5D2B
	s_and_b32 s2, vcc_lo, s4                                   // 00000000576C: 8B02046A
	s_and_b32 s3, s3, s5                                       // 000000005770: 8B030503
	s_wait_alu depctr_sa_sdst(0)                               // 000000005774: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 000000005778: 8C020206
	s_or_b32 s3, s7, s3                                        // 00000000577C: 8C030307
	s_wait_alu depctr_sa_sdst(0)                               // 000000005780: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000005784: 8D6A0002
	s_xor_b32 s2, s3, s0                                       // 000000005788: 8D020003
	s_wait_alu depctr_sa_sdst(0)                               // 00000000578C: BF88FF9E
	v_cndmask_b32_e32 v35, v35, v37, vcc_lo                    // 000000005790: 02464B23
	v_cndmask_b32_e32 v33, v33, v39, vcc_lo                    // 000000005794: 02424F21
	v_cmp_eq_f32_e32 vcc_lo, v41, v36                          // 000000005798: 7C244929
	v_cmp_lt_f32_e64 s3, v43, v42                              // 00000000579C: D4110003 0202552B
	v_cmp_lt_f32_e64 s4, v42, v43                              // 0000000057A4: D4110004 0202572A
	v_cmp_lt_f32_e64 s5, v41, v36                              // 0000000057AC: D4110005 02024929
	v_cndmask_b32_e64 v32, v32, v38, s2                        // 0000000057B4: D5010020 000A4D20
	v_cndmask_b32_e64 v34, v34, v40, s2                        // 0000000057BC: D5010022 000A5122
	v_cmp_lt_f32_e64 s2, v36, v41                              // 0000000057C4: D4110002 02025324
	s_and_b32 s3, vcc_lo, s3                                   // 0000000057CC: 8B03036A
	s_and_b32 s4, vcc_lo, s4                                   // 0000000057D0: 8B04046A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000057D4: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 0000000057D8: 8C6A0305
	v_cmp_lt_f32_e64 s3, v34, v33                              // 0000000057DC: D4110003 02024322
	s_wait_alu depctr_sa_sdst(0)                               // 0000000057E4: BF88FF9E
	v_cndmask_b32_e32 v37, v36, v41, vcc_lo                    // 0000000057E8: 024A5324
	s_or_b32 s2, s2, s4                                        // 0000000057EC: 8C020402
	v_cndmask_b32_e32 v38, v42, v43, vcc_lo                    // 0000000057F0: 024C572A
	v_cmp_eq_f32_e32 vcc_lo, v32, v35                          // 0000000057F4: 7C244720
	v_cmp_lt_f32_e64 s4, v33, v34                              // 0000000057F8: D4110004 02024521
	s_wait_alu depctr_sa_sdst(0)                               // 000000005800: BF88FF9E
	v_cndmask_b32_e64 v36, v36, v41, s2                        // 000000005804: D5010024 000A5324
	v_cmp_lt_f32_e64 s5, v32, v35                              // 00000000580C: D4110005 02024720
	v_cndmask_b32_e64 v39, v42, v43, s2                        // 000000005814: D5010027 000A572A
	v_cmp_lt_f32_e64 s2, v35, v32                              // 00000000581C: D4110002 02024123
	s_and_b32 s3, vcc_lo, s3                                   // 000000005824: 8B03036A
	s_and_b32 s4, vcc_lo, s4                                   // 000000005828: 8B04046A
	s_wait_alu depctr_sa_sdst(0)                               // 00000000582C: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 000000005830: 8C6A0305
	v_permlanex16_b32 v36, v36, s11, 0xfedcba98                // 000000005834: D65C0024 03FC1724 FEDCBA98
	s_or_b32 s2, s2, s4                                        // 000000005840: 8C020402
	v_permlanex16_b32 v39, v39, s11, 0xfedcba98                // 000000005844: D65C0027 03FC1727 FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 000000005850: BF88FF9E
	v_cndmask_b32_e32 v40, v35, v32, vcc_lo                    // 000000005854: 02504123
	v_cndmask_b32_e64 v32, v35, v32, s2                        // 000000005858: D5010020 000A4123
	v_permlanex16_b32 v37, v37, s11, 0xfedcba98                // 000000005860: D65C0025 03FC1725 FEDCBA98
	v_permlanex16_b32 v38, v38, s11, 0xfedcba98                // 00000000586C: D65C0026 03FC1726 FEDCBA98
	v_cndmask_b32_e32 v35, v33, v34, vcc_lo                    // 000000005878: 02464521
	v_cndmask_b32_e64 v33, v33, v34, s2                        // 00000000587C: D5010021 000A4521
	v_cmp_eq_f32_e32 vcc_lo, v65, v36                          // 000000005884: 7C244941
	v_cmp_lt_f32_e64 s4, v64, v39                              // 000000005888: D4110004 02024F40
	v_cmp_eq_f32_e64 s3, v59, v37                              // 000000005890: D4120003 02024B3B
	v_cmp_lt_f32_e64 s5, v58, v38                              // 000000005898: D4110005 02024D3A
	v_cmp_lt_f32_e64 s2, v65, v36                              // 0000000058A0: D4110002 02024941
	v_permlanex16_b32 v32, v32, s11, 0xfedcba98                // 0000000058A8: D65C0020 03FC1720 FEDCBA98
	v_permlanex16_b32 v40, v40, s11, 0xfedcba98                // 0000000058B4: D65C0028 03FC1728 FEDCBA98
	v_permlanex16_b32 v33, v33, s11, 0xfedcba98                // 0000000058C0: D65C0021 03FC1721 FEDCBA98
	v_permlanex16_b32 v35, v35, s11, 0xfedcba98                // 0000000058CC: D65C0023 03FC1723 FEDCBA98
	v_cmp_lt_f32_e64 s6, v59, v37                              // 0000000058D8: D4110006 02024B3B
	s_and_b32 s4, vcc_lo, s4                                   // 0000000058E0: 8B04046A
	s_and_b32 s8, s3, s5                                       // 0000000058E4: 8B080503
	s_wait_alu depctr_sa_sdst(0)                               // 0000000058E8: BF88FF9E
	s_or_b32 vcc_lo, s2, s4                                    // 0000000058EC: 8C6A0402
	v_cmp_eq_f32_e64 s3, v60, v32                              // 0000000058F0: D4120003 0202413C
	v_cmp_eq_f32_e64 s4, v62, v40                              // 0000000058F8: D4120004 0202513E
	v_cmp_lt_f32_e64 s5, v61, v33                              // 000000005900: D4110005 0202433D
	v_cmp_lt_f32_e64 s7, v63, v35                              // 000000005908: D4110007 0202473F
	v_cmp_lt_f32_e64 s2, v60, v32                              // 000000005910: D4110002 0202413C
	s_or_b32 s6, s6, s8                                        // 000000005918: 8C060806
	v_cmp_lt_f32_e64 s8, v62, v40                              // 00000000591C: D4110008 0202513E
	s_and_b32 s3, s3, s5                                       // 000000005924: 8B030503
	s_and_b32 s4, s4, s7                                       // 000000005928: 8B040704
	s_wait_alu depctr_sa_sdst(0)                               // 00000000592C: BF88FF9E
	s_or_b32 s2, s2, s3                                        // 000000005930: 8C020302
	v_cndmask_b32_e32 v34, v36, v65, vcc_lo                    // 000000005934: 02448324
	s_or_b32 s3, s8, s4                                        // 000000005938: 8C030408
	v_cndmask_b32_e64 v36, v37, v59, s6                        // 00000000593C: D5010024 001A7725
	s_wait_alu depctr_sa_sdst(0)                               // 000000005944: BF88FF9E
	v_cndmask_b32_e64 v32, v32, v60, s2                        // 000000005948: D5010020 000A7920
	v_cndmask_b32_e64 v37, v40, v62, s3                        // 000000005950: D5010025 000E7D28
	v_cndmask_b32_e64 v33, v33, v61, s2                        // 000000005958: D5010021 000A7B21
	v_cndmask_b32_e64 v35, v35, v63, s3                        // 000000005960: D5010023 000E7F23
	v_cndmask_b32_e32 v39, v39, v64, vcc_lo                    // 000000005968: 024E8127
	v_cmp_eq_f32_e32 vcc_lo, v52, v32                          // 00000000596C: 7C244134
	v_cmp_eq_f32_e64 s2, v53, v37                              // 000000005970: D4120002 02024B35
	v_cmp_lt_f32_e64 s3, v51, v33                              // 000000005978: D4110003 02024333
	v_cmp_lt_f32_e64 s4, v50, v35                              // 000000005980: D4110004 02024732
	v_cmp_lt_f32_e64 s5, v52, v32                              // 000000005988: D4110005 02024134
	v_cndmask_b32_e64 v38, v38, v58, s6                        // 000000005990: D5010026 001A7526
	v_cmp_lt_f32_e64 s6, v53, v37                              // 000000005998: D4110006 02024B35
	s_and_b32 s3, vcc_lo, s3                                   // 0000000059A0: 8B03036A
	s_and_b32 s2, s2, s4                                       // 0000000059A4: 8B020402
	s_wait_alu depctr_sa_sdst(0)                               // 0000000059A8: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 0000000059AC: 8C6A0305
	v_cmp_eq_f32_e64 s3, v57, v36                              // 0000000059B0: D4120003 02024939
	s_or_b32 s2, s6, s2                                        // 0000000059B8: 8C020206
	v_cmp_lt_f32_e64 s5, v54, v38                              // 0000000059BC: D4110005 02024D36
	s_wait_alu depctr_sa_sdst(0)                               // 0000000059C4: BF88FF9E
	v_cndmask_b32_e64 v41, v37, v53, s2                        // 0000000059C8: D5010029 000A6B25
	v_cndmask_b32_e64 v43, v35, v50, s2                        // 0000000059D0: D501002B 000A6523
	v_cndmask_b32_e64 v37, v53, v37, s2                        // 0000000059D8: D5010025 000A4B35
	v_cndmask_b32_e64 v35, v50, v35, s2                        // 0000000059E0: D5010023 000A4732
	v_cmp_lt_f32_e64 s2, v57, v36                              // 0000000059E8: D4110002 02024939
	s_and_b32 s3, s3, s5                                       // 0000000059F0: 8B030503
	v_cndmask_b32_e32 v40, v32, v52, vcc_lo                    // 0000000059F4: 02506920
	v_cndmask_b32_e32 v42, v33, v51, vcc_lo                    // 0000000059F8: 02546721
	v_cndmask_b32_e32 v32, v52, v32, vcc_lo                    // 0000000059FC: 02404134
	s_wait_alu depctr_sa_sdst(0)                               // 000000005A00: BF88FF9E
	s_or_b32 s2, s2, s3                                        // 000000005A04: 8C020302
	v_cndmask_b32_e32 v33, v51, v33, vcc_lo                    // 000000005A08: 02424333
	s_wait_alu depctr_sa_sdst(0)                               // 000000005A0C: BF88FF9E
	v_cndmask_b32_e64 v45, v36, v57, s2                        // 000000005A10: D501002D 000A7324
	v_cndmask_b32_e64 v47, v38, v54, s2                        // 000000005A18: D501002F 000A6D26
	v_cndmask_b32_e64 v36, v57, v36, s2                        // 000000005A20: D5010024 000A4939
	v_cndmask_b32_e64 v38, v54, v38, s2                        // 000000005A28: D5010026 000A4D36
	v_cmp_eq_f32_e32 vcc_lo, v55, v34                          // 000000005A30: 7C244537
	v_cmp_eq_f32_e64 s3, v41, v45                              // 000000005A34: D4120003 02025B29
	v_cmp_lt_f32_e64 s5, v43, v47                              // 000000005A3C: D4110005 02025F2B
	v_cmp_lt_f32_e64 s2, v41, v45                              // 000000005A44: D4110002 02025B29
	v_cmp_lt_f32_e64 s4, v56, v39                              // 000000005A4C: D4110004 02024F38
	v_cmp_lt_f32_e64 s6, v55, v34                              // 000000005A54: D4110006 02024537
	s_and_b32 s3, s3, s5                                       // 000000005A5C: 8B030503
	v_cmp_lt_f32_e64 s5, v35, v38                              // 000000005A60: D4110005 02024D23
	s_wait_alu depctr_sa_sdst(0)                               // 000000005A68: BF88FF9E
	s_or_b32 s2, s2, s3                                        // 000000005A6C: 8C020302
	v_cmp_eq_f32_e64 s3, v37, v36                              // 000000005A70: D4120003 02024925
	s_wait_alu depctr_sa_sdst(0)                               // 000000005A78: BF88FF9E
	v_cndmask_b32_e64 v51, v45, v41, s2                        // 000000005A7C: D5010033 000A532D
	v_cndmask_b32_e64 v53, v47, v43, s2                        // 000000005A84: D5010035 000A572F
	v_cndmask_b32_e64 v41, v41, v45, s2                        // 000000005A8C: D5010029 000A5B29
	v_cndmask_b32_e64 v43, v43, v47, s2                        // 000000005A94: D501002B 000A5F2B
	v_cmp_lt_f32_e64 s2, v37, v36                              // 000000005A9C: D4110002 02024925
	s_and_b32 s4, vcc_lo, s4                                   // 000000005AA4: 8B04046A
	s_and_b32 s3, s3, s5                                       // 000000005AA8: 8B030503
	s_wait_alu depctr_sa_sdst(0)                               // 000000005AAC: BF88FF9E
	s_or_b32 vcc_lo, s6, s4                                    // 000000005AB0: 8C6A0406
	v_mov_b32_e32 v54, v51                                     // 000000005AB4: 7E6C0333
	s_or_b32 s2, s2, s3                                        // 000000005AB8: 8C020302
	s_wait_alu depctr_sa_sdst(0)                               // 000000005ABC: BF88FF9E
	v_cndmask_b32_e32 v46, v39, v56, vcc_lo                    // 000000005AC0: 025C7127
	v_cndmask_b32_e64 v45, v36, v37, s2                        // 000000005AC4: D501002D 000A4B24
	v_cndmask_b32_e64 v36, v37, v36, s2                        // 000000005ACC: D5010024 000A4925
	v_cndmask_b32_e32 v44, v34, v55, vcc_lo                    // 000000005AD4: 02586F22
	v_dual_cndmask_b32 v34, v55, v34 :: v_dual_cndmask_b32 v39, v56, v39// 000000005AD8: CA524537 22264F38
	v_cmp_lt_f32_e64 s4, v42, v46                              // 000000005AE0: D4110004 02025D2A
	v_mov_b32_e32 v56, v53                                     // 000000005AE8: 7E700335
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005AEC: BF870004
	v_cmp_eq_f32_e32 vcc_lo, v40, v44                          // 000000005AF0: 7C245928
	v_cmp_lt_f32_e64 s6, v40, v44                              // 000000005AF4: D4110006 02025928
	v_permlanex16_b32 v54, v54, s11, 0xfedcba98                // 000000005AFC: D65C0036 03FC1736 FEDCBA98
	v_cndmask_b32_e64 v47, v38, v35, s2                        // 000000005B08: D501002F 000A4726
	v_permlanex16_b32 v56, v56, s11, 0xfedcba98                // 000000005B10: D65C0038 03FC1738 FEDCBA98
	s_and_b32 s4, vcc_lo, s4                                   // 000000005B1C: 8B04046A
	v_cndmask_b32_e64 v35, v35, v38, s2                        // 000000005B20: D5010023 000A4D23
	s_wait_alu depctr_sa_sdst(0)                               // 000000005B28: BF88FF9E
	s_or_b32 vcc_lo, s6, s4                                    // 000000005B2C: 8C6A0406
	v_cmp_lt_f32_e64 s4, v33, v39                              // 000000005B30: D4110004 02024F21
	s_wait_alu depctr_sa_sdst(0)                               // 000000005B38: BF88FF9E
	v_cndmask_b32_e32 v50, v44, v40, vcc_lo                    // 000000005B3C: 0264512C
	v_cndmask_b32_e32 v52, v46, v42, vcc_lo                    // 000000005B40: 0268552E
	v_cndmask_b32_e32 v40, v40, v44, vcc_lo                    // 000000005B44: 02505928
	v_cndmask_b32_e32 v42, v42, v46, vcc_lo                    // 000000005B48: 02545D2A
	v_cmp_eq_f32_e32 vcc_lo, v32, v34                          // 000000005B4C: 7C244520
	v_cmp_lt_f32_e64 s6, v32, v34                              // 000000005B50: D4110006 02024520
	v_mov_b32_e32 v55, v52                                     // 000000005B58: 7E6E0334
	v_cmp_eq_f32_e64 s3, v51, v54                              // 000000005B5C: D4120003 02026D33
	v_cmp_gt_f32_e64 s5, v53, v56                              // 000000005B64: D4140005 02027135
	s_and_b32 s4, vcc_lo, s4                                   // 000000005B6C: 8B04046A
	v_mov_b32_e32 v37, v40                                     // 000000005B70: 7E4A0328
	s_wait_alu depctr_sa_sdst(0)                               // 000000005B74: BF88FF9E
	s_or_b32 vcc_lo, s6, s4                                    // 000000005B78: 8C6A0406
	v_permlanex16_b32 v55, v55, s11, 0xfedcba98                // 000000005B7C: D65C0037 03FC1737 FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 000000005B88: BF88FF9E
	v_cndmask_b32_e32 v44, v34, v32, vcc_lo                    // 000000005B8C: 02584122
	v_cndmask_b32_e32 v32, v32, v34, vcc_lo                    // 000000005B90: 02404520
	v_mov_b32_e32 v34, v50                                     // 000000005B94: 7E440332
	v_dual_cndmask_b32 v46, v39, v33 :: v_dual_cndmask_b32 v33, v33, v39// 000000005B98: CA524327 2E204F21
	v_cmp_gt_f32_e64 s4, v52, v55                              // 000000005BA0: D4140004 02026F34
	v_mov_b32_e32 v39, v42                                     // 000000005BA8: 7E4E032A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005BAC: BF870004
	v_permlanex16_b32 v34, v34, s11, 0xfedcba98                // 000000005BB0: D65C0022 03FC1722 FEDCBA98
	v_cmp_gt_f32_e64 s7, v51, v54                              // 000000005BBC: D4140007 02026D33
	s_and_b32 s3, s3, s5                                       // 000000005BC4: 8B030503
	v_permlanex16_b32 v37, v37, s11, 0xfedcba98                // 000000005BC8: D65C0025 03FC1725 FEDCBA98
	v_permlanex16_b32 v39, v39, s11, 0xfedcba98                // 000000005BD4: D65C0027 03FC1727 FEDCBA98
	v_cmp_eq_f32_e32 vcc_lo, v50, v34                          // 000000005BE0: 7C244532
	v_cmp_gt_f32_e64 s6, v50, v34                              // 000000005BE4: D4140006 02024532
	s_wait_alu depctr_sa_sdst(0)                               // 000000005BEC: BF88FF9E
	s_or_b32 s3, s7, s3                                        // 000000005BF0: 8C030307
	v_mov_b32_e32 v38, v41                                     // 000000005BF4: 7E4C0329
	s_and_b32 s2, vcc_lo, s4                                   // 000000005BF8: 8B02046A
	v_cmp_gt_f32_e64 s4, v42, v39                              // 000000005BFC: D4140004 02024F2A
	s_wait_alu depctr_sa_sdst(0)                               // 000000005C04: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 000000005C08: 8C020206
	v_cmp_gt_f32_e64 s6, v40, v37                              // 000000005C0C: D4140006 02024B28
	s_wait_alu depctr_sa_sdst(0)                               // 000000005C14: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000005C18: 8D6A0002
	s_xor_b32 s2, s3, s0                                       // 000000005C1C: 8D020003
	s_wait_alu depctr_sa_sdst(0)                               // 000000005C20: BF88FF9E
	v_cndmask_b32_e32 v34, v50, v34, vcc_lo                    // 000000005C24: 02444532
	v_cndmask_b32_e64 v51, v51, v54, s2                        // 000000005C28: D5010033 000A6D33
	v_cndmask_b32_e32 v54, v52, v55, vcc_lo                    // 000000005C30: 026C6F34
	v_cmp_eq_f32_e32 vcc_lo, v40, v37                          // 000000005C34: 7C244B28
	v_mov_b32_e32 v50, v43                                     // 000000005C38: 7E64032B
	v_cndmask_b32_e64 v55, v53, v56, s2                        // 000000005C3C: D5010037 000A7135
	v_permlanex16_b32 v38, v38, s11, 0xfedcba98                // 000000005C44: D65C0026 03FC1726 FEDCBA98
	v_mov_b32_e32 v53, v46                                     // 000000005C50: 7E6A032E
	s_and_b32 s2, vcc_lo, s4                                   // 000000005C54: 8B02046A
	v_permlanex16_b32 v50, v50, s11, 0xfedcba98                // 000000005C58: D65C0032 03FC1732 FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 000000005C64: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 000000005C68: 8C020206
	v_cmp_eq_f32_e64 s3, v41, v38                              // 000000005C6C: D4120003 02024D29
	s_wait_alu depctr_sa_sdst(0)                               // 000000005C74: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000005C78: 8D6A0002
	v_permlanex16_b32 v53, v53, s11, 0xfedcba98                // 000000005C7C: D65C0035 03FC1735 FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 000000005C88: BF88FF9E
	v_cndmask_b32_e32 v37, v40, v37, vcc_lo                    // 000000005C8C: 024A4B28
	v_mov_b32_e32 v40, v44                                     // 000000005C90: 7E50032C
	v_cmp_gt_f32_e64 s5, v43, v50                              // 000000005C94: D4140005 0202652B
	v_cmp_gt_f32_e64 s7, v41, v38                              // 000000005C9C: D4140007 02024D29
	v_cndmask_b32_e32 v39, v42, v39, vcc_lo                    // 000000005CA4: 024E4F2A
	v_cmp_gt_f32_e64 s4, v46, v53                              // 000000005CA8: D4140004 02026B2E
	v_permlanex16_b32 v40, v40, s11, 0xfedcba98                // 000000005CB0: D65C0028 03FC1728 FEDCBA98
	s_and_b32 s3, s3, s5                                       // 000000005CBC: 8B030503
	v_mov_b32_e32 v52, v45                                     // 000000005CC0: 7E68032D
	s_wait_alu depctr_sa_sdst(0)                               // 000000005CC4: BF88FF9E
	s_or_b32 s3, s7, s3                                        // 000000005CC8: 8C030307
	v_mov_b32_e32 v56, v47                                     // 000000005CCC: 7E70032F
	v_cmp_eq_f32_e32 vcc_lo, v44, v40                          // 000000005CD0: 7C24512C
	v_cmp_gt_f32_e64 s6, v44, v40                              // 000000005CD4: D4140006 0202512C
	s_wait_alu depctr_sa_sdst(0)                               // 000000005CDC: BF88FF9E
	s_xor_b32 s2, s3, s0                                       // 000000005CE0: 8D020003
	v_permlanex16_b32 v52, v52, s11, 0xfedcba98                // 000000005CE4: D65C0034 03FC1734 FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 000000005CF0: BF88FF9E
	v_cndmask_b32_e64 v38, v41, v38, s2                        // 000000005CF4: D5010026 000A4D29
	v_cndmask_b32_e64 v41, v43, v50, s2                        // 000000005CFC: D5010029 000A652B
	s_and_b32 s2, vcc_lo, s4                                   // 000000005D04: 8B02046A
	v_permlanex16_b32 v56, v56, s11, 0xfedcba98                // 000000005D08: D65C0038 03FC1738 FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 000000005D14: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 000000005D18: 8C020206
	v_mov_b32_e32 v42, v32                                     // 000000005D1C: 7E540320
	s_wait_alu depctr_sa_sdst(0)                               // 000000005D20: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000005D24: 8D6A0002
	v_mov_b32_e32 v43, v36                                     // 000000005D28: 7E560324
	s_wait_alu depctr_sa_sdst(0)                               // 000000005D2C: BF88FF9E
	v_cndmask_b32_e32 v40, v44, v40, vcc_lo                    // 000000005D30: 0250512C
	v_mov_b32_e32 v44, v33                                     // 000000005D34: 7E580321
	v_mov_b32_e32 v50, v35                                     // 000000005D38: 7E640323
	v_cmp_eq_f32_e64 s3, v45, v52                              // 000000005D3C: D4120003 0202692D
	v_cmp_gt_f32_e64 s5, v47, v56                              // 000000005D44: D4140005 0202712F
	v_cmp_gt_f32_e64 s7, v45, v52                              // 000000005D4C: D4140007 0202692D
	v_permlanex16_b32 v42, v42, s11, 0xfedcba98                // 000000005D54: D65C002A 03FC172A FEDCBA98
	v_permlanex16_b32 v44, v44, s11, 0xfedcba98                // 000000005D60: D65C002C 03FC172C FEDCBA98
	v_permlanex16_b32 v43, v43, s11, 0xfedcba98                // 000000005D6C: D65C002B 03FC172B FEDCBA98
	v_permlanex16_b32 v50, v50, s11, 0xfedcba98                // 000000005D78: D65C0032 03FC1732 FEDCBA98
	s_and_b32 s3, s3, s5                                       // 000000005D84: 8B030503
	v_cndmask_b32_e32 v46, v46, v53, vcc_lo                    // 000000005D88: 025C6B2E
	s_wait_alu depctr_sa_sdst(0)                               // 000000005D8C: BF88FF9E
	s_or_b32 s3, s7, s3                                        // 000000005D90: 8C030307
	v_cmp_eq_f32_e32 vcc_lo, v32, v42                          // 000000005D94: 7C245520
	v_cmp_gt_f32_e64 s4, v33, v44                              // 000000005D98: D4140004 02025921
	s_wait_alu depctr_sa_sdst(0)                               // 000000005DA0: BF88FF9E
	s_xor_b32 s2, s3, s0                                       // 000000005DA4: 8D020003
	v_cmp_eq_f32_e64 s3, v36, v43                              // 000000005DA8: D4120003 02025724
	v_cmp_gt_f32_e64 s5, v35, v50                              // 000000005DB0: D4140005 02026523
	v_cmp_gt_f32_e64 s6, v32, v42                              // 000000005DB8: D4140006 02025520
	v_cmp_gt_f32_e64 s7, v36, v43                              // 000000005DC0: D4140007 02025724
	s_wait_alu depctr_sa_sdst(0)                               // 000000005DC8: BF88FF9E
	v_cndmask_b32_e64 v45, v45, v52, s2                        // 000000005DCC: D501002D 000A692D
	v_cndmask_b32_e64 v47, v47, v56, s2                        // 000000005DD4: D501002F 000A712F
	s_and_b32 s2, vcc_lo, s4                                   // 000000005DDC: 8B02046A
	s_and_b32 s3, s3, s5                                       // 000000005DE0: 8B030503
	s_wait_alu depctr_sa_sdst(0)                               // 000000005DE4: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 000000005DE8: 8C020206
	s_or_b32 s3, s7, s3                                        // 000000005DEC: 8C030307
	s_wait_alu depctr_sa_sdst(0)                               // 000000005DF0: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000005DF4: 8D6A0002
	s_xor_b32 s2, s3, s0                                       // 000000005DF8: 8D020003
	s_wait_alu depctr_sa_sdst(0)                               // 000000005DFC: BF88FF9E
	v_dual_cndmask_b32 v32, v32, v42 :: v_dual_cndmask_b32 v33, v33, v44// 000000005E00: CA525520 20205921
	v_cmp_eq_f32_e32 vcc_lo, v51, v34                          // 000000005E08: 7C244533
	v_cmp_lt_f32_e64 s4, v54, v55                              // 000000005E0C: D4110004 02026F36
	v_cndmask_b32_e64 v36, v36, v43, s2                        // 000000005E14: D5010024 000A5724
	v_cmp_lt_f32_e64 s3, v55, v54                              // 000000005E1C: D4110003 02026D37
	v_cndmask_b32_e64 v35, v35, v50, s2                        // 000000005E24: D5010023 000A6523
	v_cmp_lt_f32_e64 s2, v34, v51                              // 000000005E2C: D4110002 02026722
	v_cmp_lt_f32_e64 s5, v51, v34                              // 000000005E34: D4110005 02024533
	s_and_b32 s4, vcc_lo, s4                                   // 000000005E3C: 8B04046A
	s_and_b32 s3, vcc_lo, s3                                   // 000000005E40: 8B03036A
	v_cmp_lt_f32_e64 s6, v33, v35                              // 000000005E44: D4110006 02024721
	s_wait_alu depctr_sa_sdst(0)                               // 000000005E4C: BF88FF9E
	s_or_b32 s2, s2, s4                                        // 000000005E50: 8C020402
	s_or_b32 vcc_lo, s5, s3                                    // 000000005E54: 8C6A0305
	s_wait_alu depctr_sa_sdst(0)                               // 000000005E58: BF88FF9E
	v_cndmask_b32_e64 v53, v34, v51, s2                        // 000000005E5C: D5010035 000A6722
	v_cndmask_b32_e64 v50, v54, v55, s2                        // 000000005E64: D5010032 000A6F36
	v_cmp_eq_f32_e64 s2, v38, v37                              // 000000005E6C: D4120002 02024B26
	v_cmp_lt_f32_e64 s3, v41, v39                              // 000000005E74: D4110003 02024F29
	v_cmp_lt_f32_e64 s4, v39, v41                              // 000000005E7C: D4110004 02025327
	v_cmp_eq_f32_e64 s5, v36, v32                              // 000000005E84: D4120005 02024124
	v_cndmask_b32_e32 v52, v34, v51, vcc_lo                    // 000000005E8C: 02686722
	v_cndmask_b32_e32 v51, v54, v55, vcc_lo                    // 000000005E90: 02666F36
	s_and_b32 s8, s2, s3                                       // 000000005E94: 8B080302
	v_cmp_lt_f32_e64 s3, v32, v36                              // 000000005E98: D4110003 02024920
	s_and_b32 s4, s2, s4                                       // 000000005EA0: 8B040402
	s_and_b32 s2, s5, s6                                       // 000000005EA4: 8B020605
	v_cmp_lt_f32_e32 vcc_lo, v38, v37                          // 000000005EA8: 7C224B26
	v_cmp_lt_f32_e64 s7, v37, v38                              // 000000005EAC: D4110007 02024D25
	s_wait_alu depctr_sa_sdst(0)                               // 000000005EB4: BF88FF9E
	s_or_b32 s2, s3, s2                                        // 000000005EB8: 8C020203
	v_cmp_lt_f32_e64 s3, v47, v46                              // 000000005EBC: D4110003 02025D2F
	s_wait_alu depctr_sa_sdst(0)                               // 000000005EC4: BF88FF9E
	v_cndmask_b32_e64 v59, v32, v36, s2                        // 000000005EC8: D501003B 000A4920
	v_cndmask_b32_e64 v58, v33, v35, s2                        // 000000005ED0: D501003A 000A4721
	s_or_b32 vcc_lo, vcc_lo, s8                                // 000000005ED8: 8C6A086A
	s_or_b32 s2, s7, s4                                        // 000000005EDC: 8C020407
	s_wait_alu depctr_sa_sdst(0)                               // 000000005EE0: BF88FF9E
	v_cndmask_b32_e32 v55, v37, v38, vcc_lo                    // 000000005EE4: 026E4D25
	v_cndmask_b32_e64 v57, v37, v38, s2                        // 000000005EE8: D5010039 000A4D25
	v_dual_max_num_f32 v34, v59, v59 :: v_dual_max_num_f32 v37, v58, v58// 000000005EF0: CA94773B 2224753A
	v_cndmask_b32_e64 v54, v39, v41, s2                        // 000000005EF8: D5010036 000A5327
	v_cmp_eq_f32_e64 s2, v45, v40                              // 000000005F00: D4120002 0202512D
	v_cndmask_b32_e32 v56, v39, v41, vcc_lo                    // 000000005F08: 02705327
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005F0C: BF870004
	v_dual_max_num_f32 v34, 0xff800000, v34 :: v_dual_max_num_f32 v37, 0xff800000, v37// 000000005F10: CA9444FF 22244AFF FF800000
	v_cmp_lt_f32_e32 vcc_lo, v45, v40                          // 000000005F1C: 7C22512D
	v_cmp_lt_f32_e64 s4, v46, v47                              // 000000005F20: D4110004 02025F2E
	s_and_b32 s3, s2, s3                                       // 000000005F28: 8B030302
	v_cndmask_b32_e64 v34, v34, 0xff800000, s1                 // 000000005F2C: D5010022 0005FF22 FF800000
	v_cndmask_b32_e64 v37, v37, 0xff800000, s1                 // 000000005F38: D5010025 0005FF25 FF800000
	v_cmp_lt_f32_e64 s6, v40, v45                              // 000000005F44: D4110006 02025B28
	s_wait_alu depctr_sa_sdst(0)                               // 000000005F4C: BF88FF9E
	s_or_b32 vcc_lo, vcc_lo, s3                                // 000000005F50: 8C6A036A
	s_and_b32 s2, s2, s4                                       // 000000005F54: 8B020402
	v_dual_mov_b32 v38, v34 :: v_dual_mov_b32 v39, v37         // 000000005F58: CA100122 26260125
	s_wait_alu depctr_sa_sdst(0)                               // 000000005F60: BF88FF9E
	v_dual_cndmask_b32 v60, v40, v45 :: v_dual_cndmask_b32 v61, v46, v47// 000000005F64: CA525B28 3C3C5F2E
	v_cmp_lt_f32_e32 vcc_lo, v35, v33                          // 000000005F6C: 7C224323
	s_delay_alu instid0(VALU_DEP_3)                            // 000000005F70: BF870003
	v_permlanex16_b32 v38, v38, s11, 0xfedcba98                // 000000005F74: D65C0026 03FC1726 FEDCBA98
	v_permlanex16_b32 v39, v39, s11, 0xfedcba98                // 000000005F80: D65C0027 03FC1727 FEDCBA98
	v_cmp_lt_f32_e64 s3, v36, v32                              // 000000005F8C: D4110003 02024124
	s_or_b32 s2, s6, s2                                        // 000000005F94: 8C020206
	v_dual_max_num_f32 v38, v38, v38 :: v_dual_max_num_f32 v39, v39, v39// 000000005F98: CA944D26 26264F27
	s_wait_alu depctr_sa_sdst(0)                               // 000000005FA0: BF88FF9E
	v_cndmask_b32_e64 v62, v40, v45, s2                        // 000000005FA4: D501003E 000A5B28
	v_cndmask_b32_e64 v63, v46, v47, s2                        // 000000005FAC: D501003F 000A5F2E
	s_and_b32 s2, s5, vcc_lo                                   // 000000005FB4: 8B026A05
	s_wait_alu depctr_sa_sdst(0)                               // 000000005FB8: BF88FF9E
	s_or_b32 vcc_lo, s3, s2                                    // 000000005FBC: 8C6A0203
	s_wait_alu depctr_sa_sdst(0)                               // 000000005FC0: BF88FF9E
	v_dual_cndmask_b32 v65, v32, v36 :: v_dual_cndmask_b32 v64, v33, v35// 000000005FC4: CA524920 41404721
	v_dual_max_num_f32 v32, v34, v38 :: v_dual_max_num_f32 v33, v37, v39// 000000005FCC: CA944D22 20204F25
	s_branch 64161                                             // 000000005FD4: BFA0FAA1 <nearest+0x4a5c>
	v_cndmask_b32_e64 v34, v74, -v14, s2                       // 000000005FD8: D5010022 400A1D4A
	v_cndmask_b32_e64 v35, v73, -v15, s3                       // 000000005FE0: D5010023 400E1F49
	v_cndmask_b32_e64 v36, v72, v30, s2                        // 000000005FE8: D5010024 000A3D48
	v_cndmask_b32_e64 v37, v71, v31, s3                        // 000000005FF0: D5010025 000E3F47
	v_cndmask_b32_e64 v68, v68, v74, s2                        // 000000005FF8: D5010044 000A9544
	v_cndmask_b32_e64 v69, v69, v73, s3                        // 000000006000: D5010045 000E9345
	v_cndmask_b32_e64 v66, v66, v72, s2                        // 000000006008: D5010042 000A9142
	v_cndmask_b32_e64 v67, v67, v71, s3                        // 000000006010: D5010043 000E8F43
	v_dual_mov_b32 v72, v36 :: v_dual_mov_b32 v71, v37         // 000000006018: CA100124 48460125
	v_dual_mov_b32 v74, v34 :: v_dual_mov_b32 v73, v35         // 000000006020: CA100122 4A480123
	s_cmp_ngt_f32 s5, 0.5                                      // 000000006028: BF4BF005
	s_mov_b32 s10, s6                                          // 00000000602C: BE8A0006
	s_cbranch_scc0 64674                                       // 000000006030: BFA1FCA2 <nearest+0x52bc>
	s_branch 64137                                             // 000000006034: BFA0FA89 <nearest+0x4a5c>
	v_cndmask_b32_e64 v32, v74, -v14, s2                       // 000000006038: D5010020 400A1D4A
	v_cndmask_b32_e64 v33, v73, -v15, s3                       // 000000006040: D5010021 400E1F49
	v_cndmask_b32_e64 v34, v72, v30, s2                        // 000000006048: D5010022 000A3D48
	v_cndmask_b32_e64 v35, v71, v31, s3                        // 000000006050: D5010023 000E3F47
	v_cndmask_b32_e64 v68, v68, v74, s2                        // 000000006058: D5010044 000A9544
	v_cndmask_b32_e64 v69, v69, v73, s3                        // 000000006060: D5010045 000E9345
	v_cndmask_b32_e64 v66, v66, v72, s2                        // 000000006068: D5010042 000A9142
	v_cndmask_b32_e64 v67, v67, v71, s3                        // 000000006070: D5010043 000E8F43
	v_dual_mov_b32 v73, v33 :: v_dual_mov_b32 v74, v32         // 000000006078: CA100121 494A0120
	v_dual_mov_b32 v71, v35 :: v_dual_mov_b32 v72, v34         // 000000006080: CA100123 47480122
	s_cmp_ngt_f32 s6, 0.5                                      // 000000006088: BF4BF006
	s_cbranch_scc0 63275                                       // 00000000608C: BFA1F72B <nearest+0x3d3c>
	v_lshlrev_b32_e32 v0, 1, v49                               // 000000006090: 30006281
	v_lshlrev_b32_e32 v1, 4, v48                               // 000000006094: 30026084
	s_lshl_b32 s1, ttmp9, 10                                   // 000000006098: 84018A75
	v_cmp_eq_f32_e32 vcc_lo, v73, v74                          // 00000000609C: 7C249549
	v_cmp_lt_f32_e64 s2, v72, v71                              // 0000000060A0: D4110002 02028F48
	v_cmp_lt_f32_e64 s3, v73, v74                              // 0000000060A8: D4110003 02029549
	s_wait_alu depctr_sa_sdst(0)                               // 0000000060B0: BF88FF9E
	v_or3_b32 v0, v1, s1, v0                                   // 0000000060B4: D6580000 04000301
	v_cmp_lt_f32_e64 s1, v71, v72                              // 0000000060BC: D4110001 02029147
	v_cmp_lt_f32_e64 s4, v74, v73                              // 0000000060C4: D4110004 0202934A
	s_and_b32 s2, s2, vcc_lo                                   // 0000000060CC: 8B026A02
	s_mov_b32 s8, 0x76543210                                   // 0000000060D0: BE8800FF 76543210
	v_ashrrev_i32_e32 v1, 31, v0                               // 0000000060D8: 3402009F
	s_and_b32 s1, s1, vcc_lo                                   // 0000000060DC: 8B016A01
	s_wait_alu depctr_sa_sdst(0)                               // 0000000060E0: BF88FF9E
	s_or_b32 s2, s4, s2                                        // 0000000060E4: 8C020204
	s_or_b32 s1, s3, s1                                        // 0000000060E8: 8C010103
	v_cmp_lt_f32_e64 s4, v69, v68                              // 0000000060EC: D4110004 02028945
	v_lshlrev_b64_e32 v[2:3], 2, v[0:1]                        // 0000000060F4: 3E040082
	s_wait_alu depctr_sa_sdst(0)                               // 0000000060F8: BF88FF9E
	s_xor_b32 vcc_lo, s1, s0                                   // 0000000060FC: 8D6A0001
	v_cmp_lt_f32_e64 s3, v66, v67                              // 000000006100: D4110003 02028742
	s_wait_alu depctr_sa_sdst(0)                               // 000000006108: BF88FF9E
	v_cndmask_b32_e32 v4, v74, v73, vcc_lo                     // 00000000610C: 0208934A
	v_cndmask_b32_e32 v6, v72, v71, vcc_lo                     // 000000006110: 020C8F48
	v_cmp_eq_f32_e32 vcc_lo, v69, v68                          // 000000006114: 7C248945
	v_add_co_u32 v0, s1, s18, v2                               // 000000006118: D7000100 02020412
	s_wait_alu depctr_va_sdst(0)                               // 000000006120: BF88F19F
	v_add_co_ci_u32_e64 v1, null, s19, v3, s1                  // 000000006124: D5207C01 00060613
	s_xor_b32 s1, s2, s9                                       // 00000000612C: 8D010902
	v_cmp_lt_f32_e64 s2, v67, v66                              // 000000006130: D4110002 02028543
	s_wait_alu depctr_sa_sdst(0)                               // 000000006138: BF88FF9E
	v_cndmask_b32_e64 v5, v73, v74, s1                         // 00000000613C: D5010005 00069549
	v_cndmask_b32_e64 v7, v71, v72, s1                         // 000000006144: D5010007 00069147
	v_mov_b32_e32 v9, v4                                       // 00000000614C: 7E120304
	v_cmp_lt_f32_e64 s5, v68, v69                              // 000000006150: D4110005 02028B44
	s_and_b32 s1, s2, vcc_lo                                   // 000000006158: 8B016A02
	s_and_b32 s2, s3, vcc_lo                                   // 00000000615C: 8B026A03
	s_wait_alu depctr_sa_sdst(0)                               // 000000006160: BF88FF9E
	s_or_b32 s1, s4, s1                                        // 000000006164: 8C010104
	v_permlanex16_b32 v9, v9, s8, 0xfedcba98                   // 000000006168: D65C0009 03FC1109 FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 000000006174: BF88FF9E
	s_xor_b32 vcc_lo, s1, s0                                   // 000000006178: 8D6A0001
	s_or_b32 s2, s5, s2                                        // 00000000617C: 8C020205
	s_wait_alu depctr_sa_sdst(0)                               // 000000006180: BF88FF9E
	v_cndmask_b32_e32 v14, v66, v67, vcc_lo                    // 000000006184: 021C8742
	s_xor_b32 s1, s2, s9                                       // 000000006188: 8D010902
	v_cmp_gt_f32_e64 s5, v4, v9                                // 00000000618C: D4140005 02021304
	s_wait_alu depctr_sa_sdst(0)                               // 000000006194: BF88FF9E
	v_cndmask_b32_e64 v13, v69, v68, s1                        // 000000006198: D501000D 00068945
	v_cndmask_b32_e64 v15, v67, v66, s1                        // 0000000061A0: D501000F 00068543
	v_dual_mov_b32 v17, v14 :: v_dual_cndmask_b32 v8, v68, v69 // 0000000061A8: CA12010E 11088B44
	v_mov_b32_e32 v11, v6                                      // 0000000061B0: 7E160306
	v_cmp_eq_f32_e32 vcc_lo, v4, v9                            // 0000000061B4: 7C241304
	v_mov_b32_e32 v10, v5                                      // 0000000061B8: 7E140305
	v_mov_b32_e32 v16, v13                                     // 0000000061BC: 7E20030D
	v_mov_b32_e32 v18, v15                                     // 0000000061C0: 7E24030F
	v_permlanex16_b32 v11, v11, s8, 0xfedcba98                 // 0000000061C4: D65C000B 03FC110B FEDCBA98
	v_permlanex16_b32 v17, v17, s8, 0xfedcba98                 // 0000000061D0: D65C0011 03FC1111 FEDCBA98
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 0000000061DC: BF870214
	v_permlanex16_b32 v16, v16, s8, 0xfedcba98                 // 0000000061E0: D65C0010 03FC1110 FEDCBA98
	v_permlanex16_b32 v18, v18, s8, 0xfedcba98                 // 0000000061EC: D65C0012 03FC1112 FEDCBA98
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000061F8: BF870004
	v_cmp_gt_f32_e64 s3, v6, v11                               // 0000000061FC: D4140003 02021706
	s_and_b32 s1, vcc_lo, s3                                   // 000000006204: 8B01036A
	v_cmp_neq_f32_e64 s3, v13, v16                             // 000000006208: D41D0003 0202210D
	s_wait_alu depctr_sa_sdst(0)                               // 000000006210: BF88FF9E
	s_or_b32 s1, s5, s1                                        // 000000006214: 8C010105
	s_wait_alu depctr_sa_sdst(0)                               // 000000006218: BF88FF9E
	s_xor_b32 vcc_lo, s1, s0                                   // 00000000621C: 8D6A0001
	s_wait_alu depctr_sa_sdst(0)                               // 000000006220: BF88FF9E
	v_cndmask_b32_e32 v6, v6, v11, vcc_lo                      // 000000006224: 020C1706
	v_mov_b32_e32 v12, v7                                      // 000000006228: 7E180307
	v_cndmask_b32_e32 v4, v4, v9, vcc_lo                       // 00000000622C: 02081304
	v_mov_b32_e32 v9, v8                                       // 000000006230: 7E120308
	v_permlanex16_b32 v10, v10, s8, 0xfedcba98                 // 000000006234: D65C000A 03FC110A FEDCBA98
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000006240: BF870194
	v_permlanex16_b32 v12, v12, s8, 0xfedcba98                 // 000000006244: D65C000C 03FC110C FEDCBA98
	v_permlanex16_b32 v9, v9, s8, 0xfedcba98                   // 000000006250: D65C0009 03FC1109 FEDCBA98
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 00000000625C: BF870223
	v_cmp_eq_f32_e64 s2, v5, v10                               // 000000006260: D4120002 02021505
	v_cmp_gt_f32_e64 s6, v5, v10                               // 000000006268: D4140006 02021505
	v_cmp_gt_f32_e64 s4, v7, v12                               // 000000006270: D4140004 02021907
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006278: BF870004
	v_cmp_neq_f32_e32 vcc_lo, v8, v9                           // 00000000627C: 7C3A1308
	v_cmp_ngt_f32_e64 s5, v8, v9                               // 000000006280: D41B0005 02021308
	s_and_b32 s2, s2, s4                                       // 000000006288: 8B020402
	v_cmp_ngt_f32_e64 s4, v15, v18                             // 00000000628C: D41B0004 0202250F
	s_wait_alu depctr_sa_sdst(0)                               // 000000006294: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 000000006298: 8C020206
	v_cmp_ngt_f32_e64 s6, v13, v16                             // 00000000629C: D41B0006 0202210D
	s_wait_alu depctr_sa_sdst(0)                               // 0000000062A4: BF88FF9E
	s_xor_b32 s1, s2, s0                                       // 0000000062A8: 8D010002
	v_cmp_ngt_f32_e64 s2, v14, v17                             // 0000000062AC: D41B0002 0202230E
	s_wait_alu depctr_sa_sdst(0)                               // 0000000062B4: BF88FF9E
	v_cndmask_b32_e64 v5, v5, v10, s1                          // 0000000062B8: D5010005 00061505
	v_cndmask_b32_e64 v7, v7, v12, s1                          // 0000000062C0: D5010007 00061907
	s_or_b32 s1, vcc_lo, s2                                    // 0000000062C8: 8C01026A
	s_or_b32 s2, s3, s4                                        // 0000000062CC: 8C020403
	s_wait_alu depctr_sa_sdst(0)                               // 0000000062D0: BF88FF9E
	s_and_b32 s1, s5, s1                                       // 0000000062D4: 8B010105
	s_and_b32 s2, s6, s2                                       // 0000000062D8: 8B020206
	s_wait_alu depctr_sa_sdst(0)                               // 0000000062DC: BF88FF9E
	s_xor_b32 vcc_lo, s1, s0                                   // 0000000062E0: 8D6A0001
	s_xor_b32 s1, s2, s0                                       // 0000000062E4: 8D010002
	s_wait_alu depctr_sa_sdst(0)                               // 0000000062E8: BF88FF9E
	v_cndmask_b32_e32 v8, v8, v9, vcc_lo                       // 0000000062EC: 02101308
	v_cndmask_b32_e32 v10, v14, v17, vcc_lo                    // 0000000062F0: 0214230E
	v_cmp_eq_f32_e32 vcc_lo, v5, v4                            // 0000000062F4: 7C240905
	v_cmp_lt_f32_e64 s2, v7, v6                                // 0000000062F8: D4110002 02020D07
	v_cmp_lt_f32_e64 s3, v6, v7                                // 000000006300: D4110003 02020F06
	v_cndmask_b32_e64 v9, v13, v16, s1                         // 000000006308: D5010009 0006210D
	v_cmp_lt_f32_e64 s4, v5, v4                                // 000000006310: D4110004 02020905
	v_cndmask_b32_e64 v11, v15, v18, s1                        // 000000006318: D501000B 0006250F
	v_cmp_lt_f32_e64 s1, v4, v5                                // 000000006320: D4110001 02020B04
	s_and_b32 s2, vcc_lo, s2                                   // 000000006328: 8B02026A
	s_and_b32 s3, vcc_lo, s3                                   // 00000000632C: 8B03036A
	s_wait_alu depctr_sa_sdst(0)                               // 000000006330: BF88FF9E
	s_or_b32 vcc_lo, s4, s2                                    // 000000006334: 8C6A0204
	v_cmp_lt_f32_e64 s2, v11, v10                              // 000000006338: D4110002 0202150B
	s_or_b32 s1, s1, s3                                        // 000000006340: 8C010301
	s_wait_alu depctr_sa_sdst(0)                               // 000000006344: BF88FF9E
	v_cndmask_b32_e32 v12, v4, v5, vcc_lo                      // 000000006348: 02180B04
	v_cndmask_b32_e64 v4, v4, v5, s1                           // 00000000634C: D5010004 00060B04
	v_cndmask_b32_e32 v5, v6, v7, vcc_lo                       // 000000006354: 020A0F06
	v_cmp_eq_f32_e32 vcc_lo, v9, v8                            // 000000006358: 7C241109
	v_cmp_lt_f32_e64 s3, v10, v11                              // 00000000635C: D4110003 0202170A
	v_cmp_lt_f32_e64 s4, v9, v8                                // 000000006364: D4110004 02021109
	v_cndmask_b32_e64 v6, v6, v7, s1                           // 00000000636C: D5010006 00060F06
	v_cmp_lt_f32_e64 s1, v8, v9                                // 000000006374: D4110001 02021308
	s_and_b32 s2, vcc_lo, s2                                   // 00000000637C: 8B02026A
	s_and_b32 s3, vcc_lo, s3                                   // 000000006380: 8B03036A
	s_wait_alu depctr_sa_sdst(0)                               // 000000006384: BF88FF9E
	s_or_b32 vcc_lo, s4, s2                                    // 000000006388: 8C6A0204
	v_add_co_u32 v2, s5, s12, v2                               // 00000000638C: D7000502 0202040C
	s_or_b32 s1, s1, s3                                        // 000000006394: 8C010301
	s_wait_alu depctr_sa_sdst(0)                               // 000000006398: BF88FF9E
	v_cndmask_b32_e32 v7, v9, v8, vcc_lo                       // 00000000639C: 020E1109
	v_cndmask_b32_e64 v8, v9, v8, s1                           // 0000000063A0: D5010008 00061109
	v_cndmask_b32_e32 v9, v11, v10, vcc_lo                     // 0000000063A8: 0212150B
	v_cndmask_b32_e64 v10, v11, v10, s1                        // 0000000063AC: D501000A 0006150B
	v_add_co_ci_u32_e64 v3, null, s13, v3, s5                  // 0000000063B4: D5207C03 0016060D
	v_cmp_eq_f32_e64 s1, v12, v7                               // 0000000063BC: D4120001 02020F0C
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000063C4: BF870004
	v_cmp_lt_f32_e64 s3, v5, v9                                // 0000000063C8: D4110003 02021305
	v_cmp_lt_f32_e32 vcc_lo, v12, v7                           // 0000000063D0: 7C220F0C
	v_cmp_eq_f32_e64 s2, v4, v8                                // 0000000063D4: D4120002 02021104
	v_cmp_lt_f32_e64 s4, v6, v10                               // 0000000063DC: D4110004 02021506
	v_cmp_lt_f32_e64 s5, v4, v8                                // 0000000063E4: D4110005 02021104
	s_and_b32 s1, s1, s3                                       // 0000000063EC: 8B010301
	s_wait_alu depctr_sa_sdst(0)                               // 0000000063F0: BF88FF9E
	s_or_b32 vcc_lo, vcc_lo, s1                                // 0000000063F4: 8C6A016A
	s_and_b32 s2, s2, s4                                       // 0000000063F8: 8B020402
	s_wait_alu depctr_sa_sdst(0)                               // 0000000063FC: BF88FF9E
	v_dual_cndmask_b32 v11, v7, v12 :: v_dual_cndmask_b32 v14, v9, v5// 000000006400: CA521907 0B0E0B09
	v_cndmask_b32_e32 v7, v12, v7, vcc_lo                      // 000000006408: 020E0F0C
	v_cndmask_b32_e32 v5, v5, v9, vcc_lo                       // 00000000640C: 020A1305
	s_or_b32 s1, s5, s2                                        // 000000006410: 8C010205
	s_delay_alu instid0(VALU_DEP_3)                            // 000000006414: BF870003
	v_dual_mov_b32 v12, v11 :: v_dual_mov_b32 v17, v14         // 000000006418: CA10010B 0C10010E
	s_wait_alu depctr_sa_sdst(0)                               // 000000006420: BF88FF9E
	v_cndmask_b32_e64 v13, v8, v4, s1                          // 000000006424: D501000D 00060908
	v_cndmask_b32_e64 v15, v10, v6, s1                         // 00000000642C: D501000F 00060D0A
	v_cndmask_b32_e64 v4, v4, v8, s1                           // 000000006434: D5010004 00061104
	v_permlanex16_b32 v12, v12, s8, 0xfedcba98                 // 00000000643C: D65C000C 03FC110C FEDCBA98
	v_permlanex16_b32 v17, v17, s8, 0xfedcba98                 // 000000006448: D65C0011 03FC1111 FEDCBA98
	v_cndmask_b32_e64 v6, v6, v10, s1                          // 000000006454: D5010006 00061506
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 00000000645C: BF870214
	v_dual_mov_b32 v9, v7 :: v_dual_mov_b32 v10, v4            // 000000006460: CA100107 090A0104
	v_cmp_eq_f32_e32 vcc_lo, v11, v12                          // 000000006468: 7C24190B
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 00000000646C: BF870224
	v_cmp_gt_f32_e64 s3, v14, v17                              // 000000006470: D4140003 0202230E
	v_cmp_gt_f32_e64 s5, v11, v12                              // 000000006478: D4140005 0202190B
	v_permlanex16_b32 v9, v9, s8, 0xfedcba98                   // 000000006480: D65C0009 03FC1109 FEDCBA98
	v_mov_b32_e32 v16, v13                                     // 00000000648C: 7E20030D
	v_permlanex16_b32 v10, v10, s8, 0xfedcba98                 // 000000006490: D65C000A 03FC110A FEDCBA98
	s_and_b32 s1, vcc_lo, s3                                   // 00000000649C: 8B01036A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000064A0: BF88FF9E
	s_or_b32 s1, s5, s1                                        // 0000000064A4: 8C010105
	v_permlanex16_b32 v16, v16, s8, 0xfedcba98                 // 0000000064A8: D65C0010 03FC1110 FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 0000000064B4: BF88FF9E
	s_xor_b32 vcc_lo, s1, s0                                   // 0000000064B8: 8D6A0001
	v_cmp_gt_f32_e64 s5, v7, v9                                // 0000000064BC: D4140005 02021307
	s_wait_alu depctr_sa_sdst(0)                               // 0000000064C4: BF88FF9E
	v_cndmask_b32_e32 v8, v11, v12, vcc_lo                     // 0000000064C8: 0210190B
	v_dual_mov_b32 v18, v15 :: v_dual_mov_b32 v11, v5          // 0000000064CC: CA10010F 120A0105
	v_mov_b32_e32 v12, v6                                      // 0000000064D4: 7E180306
	v_cmp_eq_f32_e64 s2, v13, v16                              // 0000000064D8: D4120002 0202210D
	v_cmp_gt_f32_e64 s6, v13, v16                              // 0000000064E0: D4140006 0202210D
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000064E8: BF870004
	v_permlanex16_b32 v18, v18, s8, 0xfedcba98                 // 0000000064EC: D65C0012 03FC1112 FEDCBA98
	v_permlanex16_b32 v11, v11, s8, 0xfedcba98                 // 0000000064F8: D65C000B 03FC110B FEDCBA98
	v_permlanex16_b32 v12, v12, s8, 0xfedcba98                 // 000000006504: D65C000C 03FC110C FEDCBA98
	v_cndmask_b32_e32 v14, v14, v17, vcc_lo                    // 000000006510: 021C230E
	v_cmp_eq_f32_e32 vcc_lo, v7, v9                            // 000000006514: 7C241307
	v_cmp_gt_f32_e64 s4, v15, v18                              // 000000006518: D4140004 0202250F
	v_cmp_gt_f32_e64 s3, v5, v11                               // 000000006520: D4140003 02021705
	s_and_b32 s2, s2, s4                                       // 000000006528: 8B020402
	v_cmp_gt_f32_e64 s4, v6, v12                               // 00000000652C: D4140004 02021906
	s_wait_alu depctr_sa_sdst(0)                               // 000000006534: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 000000006538: 8C020206
	v_cmp_gt_f32_e64 s6, v4, v10                               // 00000000653C: D4140006 02021504
	s_wait_alu depctr_sa_sdst(0)                               // 000000006544: BF88FF9E
	s_xor_b32 s1, s2, s0                                       // 000000006548: 8D010002
	v_cmp_eq_f32_e64 s2, v4, v10                               // 00000000654C: D4120002 02021504
	s_wait_alu depctr_sa_sdst(0)                               // 000000006554: BF88FF9E
	v_cndmask_b32_e64 v13, v13, v16, s1                        // 000000006558: D501000D 0006210D
	v_cndmask_b32_e64 v15, v15, v18, s1                        // 000000006560: D501000F 0006250F
	s_and_b32 s1, vcc_lo, s3                                   // 000000006568: 8B01036A
	s_and_b32 s2, s2, s4                                       // 00000000656C: 8B020402
	s_wait_alu depctr_sa_sdst(0)                               // 000000006570: BF88FF9E
	s_or_b32 s1, s5, s1                                        // 000000006574: 8C010105
	s_or_b32 s2, s6, s2                                        // 000000006578: 8C020206
	s_wait_alu depctr_sa_sdst(0)                               // 00000000657C: BF88FF9E
	s_xor_b32 vcc_lo, s1, s0                                   // 000000006580: 8D6A0001
	s_xor_b32 s1, s2, s0                                       // 000000006584: 8D010002
	s_wait_alu depctr_sa_sdst(0)                               // 000000006588: BF88FF9E
	v_cndmask_b32_e32 v7, v7, v9, vcc_lo                       // 00000000658C: 020E1307
	v_cndmask_b32_e32 v5, v5, v11, vcc_lo                      // 000000006590: 020A1705
	v_cmp_eq_f32_e32 vcc_lo, v13, v8                           // 000000006594: 7C24110D
	v_cmp_lt_f32_e64 s2, v15, v14                              // 000000006598: D4110002 02021D0F
	v_cmp_lt_f32_e64 s3, v14, v15                              // 0000000065A0: D4110003 02021F0E
	v_cmp_lt_f32_e64 s4, v13, v8                               // 0000000065A8: D4110004 0202110D
	v_cndmask_b32_e64 v4, v4, v10, s1                          // 0000000065B0: D5010004 00061504
	v_cndmask_b32_e64 v6, v6, v12, s1                          // 0000000065B8: D5010006 00061906
	v_cmp_lt_f32_e64 s1, v8, v13                               // 0000000065C0: D4110001 02021B08
	s_and_b32 s2, vcc_lo, s2                                   // 0000000065C8: 8B02026A
	s_and_b32 s3, vcc_lo, s3                                   // 0000000065CC: 8B03036A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000065D0: BF88FF9E
	s_or_b32 vcc_lo, s4, s2                                    // 0000000065D4: 8C6A0204
	v_cmp_lt_f32_e64 s2, v6, v5                                // 0000000065D8: D4110002 02020B06
	s_wait_alu depctr_sa_sdst(0)                               // 0000000065E0: BF88FF9E
	v_cndmask_b32_e32 v9, v8, v13, vcc_lo                      // 0000000065E4: 02121B08
	s_or_b32 s1, s1, s3                                        // 0000000065E8: 8C010301
	v_cndmask_b32_e32 v10, v14, v15, vcc_lo                    // 0000000065EC: 02141F0E
	v_cmp_eq_f32_e32 vcc_lo, v4, v7                            // 0000000065F0: 7C240F04
	v_cmp_lt_f32_e64 s3, v5, v6                                // 0000000065F4: D4110003 02020D05
	s_wait_alu depctr_sa_sdst(0)                               // 0000000065FC: BF88FF9E
	v_cndmask_b32_e64 v8, v8, v13, s1                          // 000000006600: D5010008 00061B08
	v_cmp_lt_f32_e64 s4, v4, v7                                // 000000006608: D4110004 02020F04
	v_cndmask_b32_e64 v11, v14, v15, s1                        // 000000006610: D501000B 00061F0E
	v_cmp_lt_f32_e64 s1, v7, v4                                // 000000006618: D4110001 02020907
	s_and_b32 s2, vcc_lo, s2                                   // 000000006620: 8B02026A
	s_and_b32 s3, vcc_lo, s3                                   // 000000006624: 8B03036A
	s_wait_alu depctr_sa_sdst(0)                               // 000000006628: BF88FF9E
	s_or_b32 vcc_lo, s4, s2                                    // 00000000662C: 8C6A0204
	v_permlanex16_b32 v8, v8, s8, 0xfedcba98                   // 000000006630: D65C0008 03FC1108 FEDCBA98
	s_or_b32 s1, s1, s3                                        // 00000000663C: 8C010301
	v_permlanex16_b32 v11, v11, s8, 0xfedcba98                 // 000000006640: D65C000B 03FC110B FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 00000000664C: BF88FF9E
	v_cndmask_b32_e32 v12, v7, v4, vcc_lo                      // 000000006650: 02180907
	v_cndmask_b32_e64 v4, v7, v4, s1                           // 000000006654: D5010004 00060907
	v_permlanex16_b32 v9, v9, s8, 0xfedcba98                   // 00000000665C: D65C0009 03FC1109 FEDCBA98
	v_permlanex16_b32 v10, v10, s8, 0xfedcba98                 // 000000006668: D65C000A 03FC110A FEDCBA98
	v_cndmask_b32_e32 v7, v5, v6, vcc_lo                       // 000000006674: 020E0D05
	v_cndmask_b32_e64 v5, v5, v6, s1                           // 000000006678: D5010005 00060D05
	v_cmp_eq_f32_e32 vcc_lo, v65, v8                           // 000000006680: 7C241141
	v_cmp_lt_f32_e64 s3, v64, v11                              // 000000006684: D4110003 02021740
	v_cmp_eq_f32_e64 s2, v59, v9                               // 00000000668C: D4120002 0202133B
	v_cmp_lt_f32_e64 s4, v58, v10                              // 000000006694: D4110004 0202153A
	v_cmp_lt_f32_e64 s1, v65, v8                               // 00000000669C: D4110001 02021141
	v_permlanex16_b32 v4, v4, s8, 0xfedcba98                   // 0000000066A4: D65C0004 03FC1104 FEDCBA98
	v_permlanex16_b32 v12, v12, s8, 0xfedcba98                 // 0000000066B0: D65C000C 03FC110C FEDCBA98
	v_permlanex16_b32 v5, v5, s8, 0xfedcba98                   // 0000000066BC: D65C0005 03FC1105 FEDCBA98
	v_permlanex16_b32 v7, v7, s8, 0xfedcba98                   // 0000000066C8: D65C0007 03FC1107 FEDCBA98
	v_cmp_lt_f32_e64 s5, v59, v9                               // 0000000066D4: D4110005 0202133B
	s_and_b32 s3, vcc_lo, s3                                   // 0000000066DC: 8B03036A
	s_and_b32 s7, s2, s4                                       // 0000000066E0: 8B070402
	s_wait_alu depctr_sa_sdst(0)                               // 0000000066E4: BF88FF9E
	s_or_b32 vcc_lo, s1, s3                                    // 0000000066E8: 8C6A0301
	v_cmp_eq_f32_e64 s2, v60, v4                               // 0000000066EC: D4120002 0202093C
	v_cmp_eq_f32_e64 s3, v62, v12                              // 0000000066F4: D4120003 0202193E
	v_cmp_lt_f32_e64 s4, v61, v5                               // 0000000066FC: D4110004 02020B3D
	v_cmp_lt_f32_e64 s6, v63, v7                               // 000000006704: D4110006 02020F3F
	v_cmp_lt_f32_e64 s1, v60, v4                               // 00000000670C: D4110001 0202093C
	s_or_b32 s5, s5, s7                                        // 000000006714: 8C050705
	v_cmp_lt_f32_e64 s7, v62, v12                              // 000000006718: D4110007 0202193E
	s_and_b32 s2, s2, s4                                       // 000000006720: 8B020402
	s_and_b32 s3, s3, s6                                       // 000000006724: 8B030603
	s_wait_alu depctr_sa_sdst(0)                               // 000000006728: BF88FF9E
	s_or_b32 s1, s1, s2                                        // 00000000672C: 8C010201
	v_cndmask_b32_e32 v6, v8, v65, vcc_lo                      // 000000006730: 020C8308
	s_or_b32 s2, s7, s3                                        // 000000006734: 8C020307
	v_cndmask_b32_e64 v8, v9, v59, s5                          // 000000006738: D5010008 00167709
	s_wait_alu depctr_sa_sdst(0)                               // 000000006740: BF88FF9E
	v_cndmask_b32_e64 v4, v4, v60, s1                          // 000000006744: D5010004 00067904
	v_cndmask_b32_e64 v9, v12, v62, s2                         // 00000000674C: D5010009 000A7D0C
	v_cndmask_b32_e64 v5, v5, v61, s1                          // 000000006754: D5010005 00067B05
	v_cndmask_b32_e64 v7, v7, v63, s2                          // 00000000675C: D5010007 000A7F07
	v_cndmask_b32_e32 v11, v11, v64, vcc_lo                    // 000000006764: 0216810B
	v_cmp_eq_f32_e32 vcc_lo, v52, v4                           // 000000006768: 7C240934
	v_cmp_eq_f32_e64 s1, v53, v9                               // 00000000676C: D4120001 02021335
	v_cmp_lt_f32_e64 s2, v51, v5                               // 000000006774: D4110002 02020B33
	v_cmp_lt_f32_e64 s3, v50, v7                               // 00000000677C: D4110003 02020F32
	v_cmp_lt_f32_e64 s4, v52, v4                               // 000000006784: D4110004 02020934
	v_cndmask_b32_e64 v10, v10, v58, s5                        // 00000000678C: D501000A 0016750A
	v_cmp_lt_f32_e64 s5, v53, v9                               // 000000006794: D4110005 02021335
	s_and_b32 s2, vcc_lo, s2                                   // 00000000679C: 8B02026A
	s_and_b32 s1, s1, s3                                       // 0000000067A0: 8B010301
	s_wait_alu depctr_sa_sdst(0)                               // 0000000067A4: BF88FF9E
	s_or_b32 vcc_lo, s4, s2                                    // 0000000067A8: 8C6A0204
	v_cmp_eq_f32_e64 s2, v57, v8                               // 0000000067AC: D4120002 02021139
	s_or_b32 s1, s5, s1                                        // 0000000067B4: 8C010105
	v_cmp_lt_f32_e64 s4, v54, v10                              // 0000000067B8: D4110004 02021536
	s_wait_alu depctr_sa_sdst(0)                               // 0000000067C0: BF88FF9E
	v_cndmask_b32_e64 v13, v9, v53, s1                         // 0000000067C4: D501000D 00066B09
	v_cndmask_b32_e64 v15, v7, v50, s1                         // 0000000067CC: D501000F 00066507
	v_cndmask_b32_e64 v9, v53, v9, s1                          // 0000000067D4: D5010009 00061335
	v_cndmask_b32_e64 v7, v50, v7, s1                          // 0000000067DC: D5010007 00060F32
	v_cmp_lt_f32_e64 s1, v57, v8                               // 0000000067E4: D4110001 02021139
	s_and_b32 s2, s2, s4                                       // 0000000067EC: 8B020402
	v_cndmask_b32_e32 v12, v4, v52, vcc_lo                     // 0000000067F0: 02186904
	v_cndmask_b32_e32 v4, v52, v4, vcc_lo                      // 0000000067F4: 02080934
	v_cmp_lt_f32_e64 s3, v56, v11                              // 0000000067F8: D4110003 02021738
	s_wait_alu depctr_sa_sdst(0)                               // 000000006800: BF88FF9E
	s_or_b32 s1, s1, s2                                        // 000000006804: 8C010201
	v_cmp_lt_f32_e64 s5, v55, v6                               // 000000006808: D4110005 02020D37
	s_wait_alu depctr_sa_sdst(0)                               // 000000006810: BF88FF9E
	v_cndmask_b32_e64 v17, v8, v57, s1                         // 000000006814: D5010011 00067308
	v_cndmask_b32_e64 v19, v10, v54, s1                        // 00000000681C: D5010013 00066D0A
	v_cndmask_b32_e64 v8, v57, v8, s1                          // 000000006824: D5010008 00061139
	v_cndmask_b32_e64 v10, v54, v10, s1                        // 00000000682C: D501000A 00061536
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000006834: BF870214
	v_cmp_eq_f32_e64 s2, v13, v17                              // 000000006838: D4120002 0202230D
	v_cmp_lt_f32_e64 s4, v15, v19                              // 000000006840: D4110004 0202270F
	v_cmp_lt_f32_e64 s1, v13, v17                              // 000000006848: D4110001 0202230D
	s_and_b32 s2, s2, s4                                       // 000000006850: 8B020402
	v_cmp_lt_f32_e64 s4, v7, v10                               // 000000006854: D4110004 02021507
	s_wait_alu depctr_sa_sdst(0)                               // 00000000685C: BF88FF9E
	s_or_b32 s1, s1, s2                                        // 000000006860: 8C010201
	v_cmp_eq_f32_e64 s2, v9, v8                                // 000000006864: D4120002 02021109
	s_wait_alu depctr_sa_sdst(0)                               // 00000000686C: BF88FF9E
	v_cndmask_b32_e64 v21, v17, v13, s1                        // 000000006870: D5010015 00061B11
	v_cndmask_b32_e64 v23, v19, v15, s1                        // 000000006878: D5010017 00061F13
	v_cndmask_b32_e64 v13, v13, v17, s1                        // 000000006880: D501000D 0006230D
	v_cndmask_b32_e64 v15, v15, v19, s1                        // 000000006888: D501000F 0006270F
	v_cmp_lt_f32_e64 s1, v9, v8                                // 000000006890: D4110001 02021109
	s_and_b32 s2, s2, s4                                       // 000000006898: 8B020402
	v_mov_b32_e32 v24, v21                                     // 00000000689C: 7E300315
	v_mov_b32_e32 v26, v23                                     // 0000000068A0: 7E340317
	s_wait_alu depctr_sa_sdst(0)                               // 0000000068A4: BF88FF9E
	s_or_b32 s1, s1, s2                                        // 0000000068A8: 8C010201
	s_wait_alu depctr_sa_sdst(0)                               // 0000000068AC: BF88FF9E
	v_cndmask_b32_e64 v17, v8, v9, s1                          // 0000000068B0: D5010011 00061308
	v_cndmask_b32_e64 v8, v9, v8, s1                           // 0000000068B8: D5010008 00061109
	v_dual_cndmask_b32 v14, v5, v51 :: v_dual_cndmask_b32 v5, v51, v5// 0000000068C0: CA526705 0E040B33
	v_cmp_eq_f32_e32 vcc_lo, v55, v6                           // 0000000068C8: 7C240D37
	v_cndmask_b32_e64 v19, v10, v7, s1                         // 0000000068CC: D5010013 00060F0A
	v_cndmask_b32_e64 v7, v7, v10, s1                          // 0000000068D4: D5010007 00061507
	v_permlanex16_b32 v24, v24, s8, 0xfedcba98                 // 0000000068DC: D65C0018 03FC1118 FEDCBA98
	v_permlanex16_b32 v26, v26, s8, 0xfedcba98                 // 0000000068E8: D65C001A 03FC111A FEDCBA98
	s_and_b32 s3, vcc_lo, s3                                   // 0000000068F4: 8B03036A
	v_mov_b32_e32 v9, v13                                      // 0000000068F8: 7E12030D
	s_wait_alu depctr_sa_sdst(0)                               // 0000000068FC: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 000000006900: 8C6A0305
	v_cmp_eq_f32_e64 s2, v21, v24                              // 000000006904: D4120002 02023115
	s_wait_alu depctr_sa_sdst(0)                               // 00000000690C: BF88FF9E
	v_cndmask_b32_e32 v16, v6, v55, vcc_lo                     // 000000006910: 02206F06
	v_cndmask_b32_e32 v18, v11, v56, vcc_lo                    // 000000006914: 0224710B
	v_dual_cndmask_b32 v6, v55, v6 :: v_dual_cndmask_b32 v11, v56, v11// 000000006918: CA520D37 060A1738
	v_cmp_gt_f32_e64 s4, v23, v26                              // 000000006920: D4140004 02023517
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000006928: BF870214
	v_cmp_eq_f32_e32 vcc_lo, v12, v16                          // 00000000692C: 7C24210C
	v_cmp_lt_f32_e64 s3, v14, v18                              // 000000006930: D4110003 0202250E
	v_cmp_lt_f32_e64 s5, v12, v16                              // 000000006938: D4110005 0202210C
	v_cmp_gt_f32_e64 s6, v21, v24                              // 000000006940: D4140006 02023115
	s_and_b32 s2, s2, s4                                       // 000000006948: 8B020402
	v_permlanex16_b32 v9, v9, s8, 0xfedcba98                   // 00000000694C: D65C0009 03FC1109 FEDCBA98
	s_and_b32 s3, vcc_lo, s3                                   // 000000006958: 8B03036A
	s_wait_alu depctr_sa_sdst(0)                               // 00000000695C: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 000000006960: 8C6A0305
	v_cmp_lt_f32_e64 s3, v5, v11                               // 000000006964: D4110003 02021705
	s_wait_alu depctr_sa_sdst(0)                               // 00000000696C: BF88FF9E
	v_cndmask_b32_e32 v20, v16, v12, vcc_lo                    // 000000006970: 02281910
	v_cndmask_b32_e32 v22, v18, v14, vcc_lo                    // 000000006974: 022C1D12
	v_cndmask_b32_e32 v12, v12, v16, vcc_lo                    // 000000006978: 0218210C
	v_cndmask_b32_e32 v14, v14, v18, vcc_lo                    // 00000000697C: 021C250E
	v_cmp_eq_f32_e32 vcc_lo, v4, v6                            // 000000006980: 7C240D04
	v_cmp_lt_f32_e64 s5, v4, v6                                // 000000006984: D4110005 02020D04
	v_mov_b32_e32 v25, v22                                     // 00000000698C: 7E320316
	s_or_b32 s2, s6, s2                                        // 000000006990: 8C020206
	v_cmp_gt_f32_e64 s6, v13, v9                               // 000000006994: D4140006 0202130D
	s_and_b32 s3, vcc_lo, s3                                   // 00000000699C: 8B03036A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000069A0: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 0000000069A4: 8C6A0305
	v_permlanex16_b32 v25, v25, s8, 0xfedcba98                 // 0000000069A8: D65C0019 03FC1119 FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 0000000069B4: BF88FF9E
	v_cndmask_b32_e32 v16, v6, v4, vcc_lo                      // 0000000069B8: 02200906
	v_cndmask_b32_e32 v4, v4, v6, vcc_lo                       // 0000000069BC: 02080D04
	v_mov_b32_e32 v6, v20                                      // 0000000069C0: 7E0C0314
	v_dual_cndmask_b32 v18, v11, v5 :: v_dual_cndmask_b32 v5, v5, v11// 0000000069C4: CA520B0B 12041705
	v_cmp_gt_f32_e64 s3, v22, v25                              // 0000000069CC: D4140003 02023316
	v_mov_b32_e32 v11, v14                                     // 0000000069D4: 7E16030E
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000069D8: BF870114
	v_permlanex16_b32 v6, v6, s8, 0xfedcba98                   // 0000000069DC: D65C0006 03FC1106 FEDCBA98
	v_permlanex16_b32 v11, v11, s8, 0xfedcba98                 // 0000000069E8: D65C000B 03FC110B FEDCBA98
	s_delay_alu instid0(VALU_DEP_2)                            // 0000000069F4: BF870002
	v_cmp_eq_f32_e32 vcc_lo, v20, v6                           // 0000000069F8: 7C240D14
	v_cmp_gt_f32_e64 s5, v20, v6                               // 0000000069FC: D4140005 02020D14
	s_and_b32 s1, vcc_lo, s3                                   // 000000006A04: 8B01036A
	v_cmp_gt_f32_e64 s3, v14, v11                              // 000000006A08: D4140003 0202170E
	s_wait_alu depctr_sa_sdst(0)                               // 000000006A10: BF88FF9E
	s_or_b32 s1, s5, s1                                        // 000000006A14: 8C010105
	s_wait_alu depctr_sa_sdst(0)                               // 000000006A18: BF88FF9E
	s_xor_b32 vcc_lo, s1, s0                                   // 000000006A1C: 8D6A0001
	s_xor_b32 s1, s2, s0                                       // 000000006A20: 8D010002
	s_wait_alu depctr_sa_sdst(0)                               // 000000006A24: BF88FF9E
	v_cndmask_b32_e32 v10, v20, v6, vcc_lo                     // 000000006A28: 02140D14
	v_mov_b32_e32 v6, v12                                      // 000000006A2C: 7E0C030C
	v_cndmask_b32_e32 v22, v22, v25, vcc_lo                    // 000000006A30: 022C3316
	v_mov_b32_e32 v20, v15                                     // 000000006A34: 7E28030F
	v_cndmask_b32_e64 v21, v21, v24, s1                        // 000000006A38: D5010015 00063115
	v_cndmask_b32_e64 v23, v23, v26, s1                        // 000000006A40: D5010017 00063517
	v_permlanex16_b32 v6, v6, s8, 0xfedcba98                   // 000000006A48: D65C0006 03FC1106 FEDCBA98
	v_mov_b32_e32 v25, v18                                     // 000000006A54: 7E320312
	v_permlanex16_b32 v20, v20, s8, 0xfedcba98                 // 000000006A58: D65C0014 03FC1114 FEDCBA98
	v_cmp_eq_f32_e64 s2, v13, v9                               // 000000006A64: D4120002 0202130D
	v_mov_b32_e32 v24, v17                                     // 000000006A6C: 7E300311
	v_cmp_eq_f32_e32 vcc_lo, v12, v6                           // 000000006A70: 7C240D0C
	v_cmp_gt_f32_e64 s5, v12, v6                               // 000000006A74: D4140005 02020D0C
	v_cmp_gt_f32_e64 s4, v15, v20                              // 000000006A7C: D4140004 0202290F
	v_permlanex16_b32 v25, v25, s8, 0xfedcba98                 // 000000006A84: D65C0019 03FC1119 FEDCBA98
	v_mov_b32_e32 v26, v19                                     // 000000006A90: 7E340313
	s_and_b32 s1, vcc_lo, s3                                   // 000000006A94: 8B01036A
	v_permlanex16_b32 v24, v24, s8, 0xfedcba98                 // 000000006A98: D65C0018 03FC1118 FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 000000006AA4: BF88FF9E
	s_or_b32 s1, s5, s1                                        // 000000006AA8: 8C010105
	s_and_b32 s2, s2, s4                                       // 000000006AAC: 8B020402
	s_wait_alu depctr_sa_sdst(0)                               // 000000006AB0: BF88FF9E
	s_xor_b32 vcc_lo, s1, s0                                   // 000000006AB4: 8D6A0001
	v_cmp_gt_f32_e64 s3, v18, v25                              // 000000006AB8: D4140003 02023312
	s_wait_alu depctr_sa_sdst(0)                               // 000000006AC0: BF88FF9E
	v_cndmask_b32_e32 v6, v12, v6, vcc_lo                      // 000000006AC4: 020C0D0C
	v_dual_mov_b32 v12, v16 :: v_dual_cndmask_b32 v11, v14, v11// 000000006AC8: CA120110 0C0A170E
	s_or_b32 s2, s6, s2                                        // 000000006AD0: 8C020206
	v_permlanex16_b32 v26, v26, s8, 0xfedcba98                 // 000000006AD4: D65C001A 03FC111A FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 000000006AE0: BF88FF9E
	s_xor_b32 s1, s2, s0                                       // 000000006AE4: 8D010002
	v_permlanex16_b32 v12, v12, s8, 0xfedcba98                 // 000000006AE8: D65C000C 03FC110C FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 000000006AF4: BF88FF9E
	v_cndmask_b32_e64 v9, v13, v9, s1                          // 000000006AF8: D5010009 0006130D
	v_cndmask_b32_e64 v13, v15, v20, s1                        // 000000006B00: D501000D 0006290F
	v_mov_b32_e32 v14, v4                                      // 000000006B08: 7E1C0304
	v_cmp_eq_f32_e64 s2, v17, v24                              // 000000006B0C: D4120002 02023111
	v_cmp_eq_f32_e32 vcc_lo, v16, v12                          // 000000006B14: 7C241910
	v_cmp_gt_f32_e64 s5, v16, v12                              // 000000006B18: D4140005 02021910
	v_cmp_gt_f32_e64 s4, v19, v26                              // 000000006B20: D4140004 02023513
	v_dual_mov_b32 v15, v8 :: v_dual_mov_b32 v20, v7           // 000000006B28: CA100108 0F140107
	s_and_b32 s1, vcc_lo, s3                                   // 000000006B30: 8B01036A
	v_permlanex16_b32 v14, v14, s8, 0xfedcba98                 // 000000006B34: D65C000E 03FC110E FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 000000006B40: BF88FF9E
	s_or_b32 s1, s5, s1                                        // 000000006B44: 8C010105
	v_cmp_gt_f32_e64 s6, v17, v24                              // 000000006B48: D4140006 02023111
	s_wait_alu depctr_sa_sdst(0)                               // 000000006B50: BF88FF9E
	s_xor_b32 vcc_lo, s1, s0                                   // 000000006B54: 8D6A0001
	s_and_b32 s2, s2, s4                                       // 000000006B58: 8B020402
	s_wait_alu depctr_sa_sdst(0)                               // 000000006B5C: BF88FF9E
	v_cndmask_b32_e32 v12, v16, v12, vcc_lo                    // 000000006B60: 02181910
	v_mov_b32_e32 v16, v5                                      // 000000006B64: 7E200305
	v_permlanex16_b32 v15, v15, s8, 0xfedcba98                 // 000000006B68: D65C000F 03FC110F FEDCBA98
	v_permlanex16_b32 v20, v20, s8, 0xfedcba98                 // 000000006B74: D65C0014 03FC1114 FEDCBA98
	v_cndmask_b32_e32 v18, v18, v25, vcc_lo                    // 000000006B80: 02243312
	v_cmp_eq_f32_e32 vcc_lo, v4, v14                           // 000000006B84: 7C241D04
	v_permlanex16_b32 v16, v16, s8, 0xfedcba98                 // 000000006B88: D65C0010 03FC1110 FEDCBA98
	s_or_b32 s2, s6, s2                                        // 000000006B94: 8C020206
	v_cmp_gt_f32_e64 s5, v4, v14                               // 000000006B98: D4140005 02021D04
	s_wait_alu depctr_sa_sdst(0)                               // 000000006BA0: BF88FF9E
	s_xor_b32 s1, s2, s0                                       // 000000006BA4: 8D010002
	v_cmp_eq_f32_e64 s2, v8, v15                               // 000000006BA8: D4120002 02021F08
	v_cmp_gt_f32_e64 s3, v5, v16                               // 000000006BB0: D4140003 02022105
	v_cmp_gt_f32_e64 s4, v7, v20                               // 000000006BB8: D4140004 02022907
	s_wait_alu depctr_sa_sdst(0)                               // 000000006BC0: BF88FF9E
	v_cndmask_b32_e64 v17, v17, v24, s1                        // 000000006BC4: D5010011 00063111
	v_cndmask_b32_e64 v19, v19, v26, s1                        // 000000006BCC: D5010013 00063513
	v_cmp_gt_f32_e64 s6, v8, v15                               // 000000006BD4: D4140006 02021F08
	s_and_b32 s1, vcc_lo, s3                                   // 000000006BDC: 8B01036A
	s_and_b32 s2, s2, s4                                       // 000000006BE0: 8B020402
	s_wait_alu depctr_sa_sdst(0)                               // 000000006BE4: BF88FF9E
	s_or_b32 s1, s5, s1                                        // 000000006BE8: 8C010105
	v_cmp_lt_f32_e64 s3, v21, v10                              // 000000006BEC: D4110003 02021515
	s_wait_alu depctr_sa_sdst(0)                               // 000000006BF4: BF88FF9E
	s_xor_b32 vcc_lo, s1, s0                                   // 000000006BF8: 8D6A0001
	s_or_b32 s2, s6, s2                                        // 000000006BFC: 8C020206
	s_wait_alu depctr_sa_sdst(0)                               // 000000006C00: BF88FF9E
	v_cndmask_b32_e32 v14, v4, v14, vcc_lo                     // 000000006C04: 021C1D04
	v_cndmask_b32_e32 v16, v5, v16, vcc_lo                     // 000000006C08: 02202105
	v_cmp_eq_f32_e32 vcc_lo, v21, v10                          // 000000006C0C: 7C241515
	v_cmp_lt_f32_e64 s1, v23, v22                              // 000000006C10: D4110001 02022D17
	s_xor_b32 s0, s2, s0                                       // 000000006C18: 8D000002
	v_cmp_lt_f32_e64 s2, v22, v23                              // 000000006C1C: D4110002 02022F16
	s_wait_alu depctr_sa_sdst(0)                               // 000000006C24: BF88FF9E
	v_cndmask_b32_e64 v15, v8, v15, s0                         // 000000006C28: D501000F 00021F08
	v_cndmask_b32_e64 v20, v7, v20, s0                         // 000000006C30: D5010014 00022907
	s_and_b32 s1, vcc_lo, s1                                   // 000000006C38: 8B01016A
	v_cmp_lt_f32_e64 s0, v10, v21                              // 000000006C3C: D4110000 02022B0A
	s_and_b32 s5, vcc_lo, s2                                   // 000000006C44: 8B05026A
	s_wait_alu depctr_sa_sdst(0)                               // 000000006C48: BF88FF9E
	s_or_b32 vcc_lo, s3, s1                                    // 000000006C4C: 8C6A0103
	v_cmp_eq_f32_e64 s2, v9, v6                                // 000000006C50: D4120002 02020D09
	v_cmp_lt_f32_e64 s3, v13, v11                              // 000000006C58: D4110003 0202170D
	v_cmp_lt_f32_e64 s1, v9, v6                                // 000000006C60: D4110001 02020D09
	v_cmp_lt_f32_e64 s4, v11, v13                              // 000000006C68: D4110004 02021B0B
	s_or_b32 s0, s0, s5                                        // 000000006C70: 8C000500
	v_cmp_lt_f32_e64 s5, v6, v9                                // 000000006C74: D4110005 02021306
	s_and_b32 s3, s2, s3                                       // 000000006C7C: 8B030302
	s_wait_alu depctr_sa_sdst(0)                               // 000000006C80: BF88FF9E
	v_cndmask_b32_e32 v24, v22, v23, vcc_lo                    // 000000006C84: 02302F16
	s_or_b32 s1, s1, s3                                        // 000000006C88: 8C010301
	v_cndmask_b32_e64 v22, v22, v23, s0                        // 000000006C8C: D5010016 00022F16
	s_and_b32 s2, s2, s4                                       // 000000006C94: 8B020402
	s_wait_alu depctr_sa_sdst(0)                               // 000000006C98: BF88FF9E
	v_cndmask_b32_e64 v4, v6, v9, s1                           // 000000006C9C: D5010004 00061306
	v_cndmask_b32_e64 v23, v11, v13, s1                        // 000000006CA4: D5010017 00061B0B
	v_cmp_eq_f32_e64 s1, v17, v12                              // 000000006CAC: D4120001 02021911
	v_cmp_lt_f32_e64 s3, v19, v18                              // 000000006CB4: D4110003 02022513
	s_or_b32 s2, s5, s2                                        // 000000006CBC: 8C020205
	v_cmp_lt_f32_e64 s4, v18, v19                              // 000000006CC0: D4110004 02022712
	v_cmp_lt_f32_e64 s5, v17, v12                              // 000000006CC8: D4110005 02021911
	s_wait_alu depctr_sa_sdst(0)                               // 000000006CD0: BF88FF9E
	v_cndmask_b32_e64 v5, v6, v9, s2                           // 000000006CD4: D5010005 000A1306
	v_cndmask_b32_e64 v25, v11, v13, s2                        // 000000006CDC: D5010019 000A1B0B
	v_cmp_lt_f32_e64 s2, v12, v17                              // 000000006CE4: D4110002 0202230C
	s_and_b32 s3, s1, s3                                       // 000000006CEC: 8B030301
	s_and_b32 s4, s1, s4                                       // 000000006CF0: 8B040401
	s_wait_alu depctr_sa_sdst(0)                               // 000000006CF4: BF88FF9E
	s_or_b32 s1, s5, s3                                        // 000000006CF8: 8C010305
	v_cmp_lt_f32_e64 s3, v20, v16                              // 000000006CFC: D4110003 02022114
	s_wait_alu depctr_sa_sdst(0)                               // 000000006D04: BF88FF9E
	v_cndmask_b32_e64 v6, v12, v17, s1                         // 000000006D08: D5010006 0006230C
	s_or_b32 s2, s2, s4                                        // 000000006D10: 8C020402
	v_cndmask_b32_e64 v26, v18, v19, s1                        // 000000006D14: D501001A 00062712
	v_cmp_eq_f32_e64 s1, v15, v14                              // 000000006D1C: D4120001 02021D0F
	v_cmp_lt_f32_e64 s4, v16, v20                              // 000000006D24: D4110004 02022910
	s_wait_alu depctr_sa_sdst(0)                               // 000000006D2C: BF88FF9E
	v_cndmask_b32_e64 v7, v12, v17, s2                         // 000000006D30: D5010007 000A230C
	v_cmp_lt_f32_e64 s5, v15, v14                              // 000000006D38: D4110005 02021D0F
	v_cndmask_b32_e64 v17, v18, v19, s2                        // 000000006D40: D5010011 000A2712
	v_cmp_lt_f32_e64 s2, v14, v15                              // 000000006D48: D4110002 02021F0E
	s_and_b32 s3, s1, s3                                       // 000000006D50: 8B030301
	s_and_b32 s4, s1, s4                                       // 000000006D54: 8B040401
	s_wait_alu depctr_sa_sdst(0)                               // 000000006D58: BF88FF9E
	s_or_b32 s1, s5, s3                                        // 000000006D5C: 8C010305
	v_cndmask_b32_e64 v11, v10, v21, s0                        // 000000006D60: D501000B 00022B0A
	s_or_b32 s2, s2, s4                                        // 000000006D68: 8C020402
	s_wait_alu depctr_sa_sdst(0)                               // 000000006D6C: BF88FF9E
	v_cndmask_b32_e64 v18, v16, v20, s1                        // 000000006D70: D5010012 00062910
	v_cndmask_b32_e64 v19, v16, v20, s2                        // 000000006D78: D5010013 000A2910
	v_cndmask_b32_e32 v10, v10, v21, vcc_lo                    // 000000006D80: 02142B0A
	v_cndmask_b32_e64 v8, v14, v15, s1                         // 000000006D84: D5010008 00061F0E
	v_cndmask_b32_e64 v9, v14, v15, s2                         // 000000006D8C: D5010009 000A1F0E
	v_cvt_i32_f32_e32 v13, v22                                 // 000000006D94: 7E1A1116
	v_cvt_i32_f32_e32 v12, v24                                 // 000000006D98: 7E181118
	v_cvt_i32_f32_e32 v15, v25                                 // 000000006D9C: 7E1E1119
	v_cvt_i32_f32_e32 v14, v23                                 // 000000006DA0: 7E1C1117
	v_cvt_i32_f32_e32 v17, v17                                 // 000000006DA4: 7E221111
	v_cvt_i32_f32_e32 v16, v26                                 // 000000006DA8: 7E20111A
	v_cvt_i32_f32_e32 v19, v19                                 // 000000006DAC: 7E261113
	v_cvt_i32_f32_e32 v18, v18                                 // 000000006DB0: 7E241112
	s_clause 0x3                                               // 000000006DB4: BF850003
	global_store_b64 v[2:3], v[10:11], off                     // 000000006DB8: EE06C07C 05000000 00000002
	global_store_b64 v[2:3], v[4:5], off offset:16             // 000000006DC4: EE06C07C 02000000 00001002
	global_store_b64 v[2:3], v[6:7], off offset:32             // 000000006DD0: EE06C07C 03000000 00002002
	global_store_b64 v[2:3], v[8:9], off offset:48             // 000000006DDC: EE06C07C 04000000 00003002
	s_clause 0x3                                               // 000000006DE8: BF850003
	global_store_b64 v[0:1], v[12:13], off                     // 000000006DEC: EE06C07C 06000000 00000000
	global_store_b64 v[0:1], v[14:15], off offset:16           // 000000006DF8: EE06C07C 07000000 00001000
	global_store_b64 v[0:1], v[16:17], off offset:32           // 000000006E04: EE06C07C 08000000 00002000
	global_store_b64 v[0:1], v[18:19], off offset:48           // 000000006E10: EE06C07C 09000000 00003000
	s_nop 0                                                    // 000000006E1C: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000006E20: BFB60003
	s_endpgm                                                   // 000000006E24: BFB00000
	s_code_end                                                 // 000000006E28: BF9F0000
	s_code_end                                                 // 000000006E2C: BF9F0000
	s_code_end                                                 // 000000006E30: BF9F0000
	s_code_end                                                 // 000000006E34: BF9F0000
	s_code_end                                                 // 000000006E38: BF9F0000
	s_code_end                                                 // 000000006E3C: BF9F0000
	s_code_end                                                 // 000000006E40: BF9F0000
	s_code_end                                                 // 000000006E44: BF9F0000
	s_code_end                                                 // 000000006E48: BF9F0000
	s_code_end                                                 // 000000006E4C: BF9F0000
	s_code_end                                                 // 000000006E50: BF9F0000
	s_code_end                                                 // 000000006E54: BF9F0000
	s_code_end                                                 // 000000006E58: BF9F0000
	s_code_end                                                 // 000000006E5C: BF9F0000
	s_code_end                                                 // 000000006E60: BF9F0000
	s_code_end                                                 // 000000006E64: BF9F0000
	s_code_end                                                 // 000000006E68: BF9F0000
	s_code_end                                                 // 000000006E6C: BF9F0000
	s_code_end                                                 // 000000006E70: BF9F0000
	s_code_end                                                 // 000000006E74: BF9F0000
	s_code_end                                                 // 000000006E78: BF9F0000
	s_code_end                                                 // 000000006E7C: BF9F0000
	s_code_end                                                 // 000000006E80: BF9F0000
	s_code_end                                                 // 000000006E84: BF9F0000
	s_code_end                                                 // 000000006E88: BF9F0000
	s_code_end                                                 // 000000006E8C: BF9F0000
	s_code_end                                                 // 000000006E90: BF9F0000
	s_code_end                                                 // 000000006E94: BF9F0000
	s_code_end                                                 // 000000006E98: BF9F0000
	s_code_end                                                 // 000000006E9C: BF9F0000
	s_code_end                                                 // 000000006EA0: BF9F0000
	s_code_end                                                 // 000000006EA4: BF9F0000
	s_code_end                                                 // 000000006EA8: BF9F0000
	s_code_end                                                 // 000000006EAC: BF9F0000
	s_code_end                                                 // 000000006EB0: BF9F0000
	s_code_end                                                 // 000000006EB4: BF9F0000
	s_code_end                                                 // 000000006EB8: BF9F0000
	s_code_end                                                 // 000000006EBC: BF9F0000
	s_code_end                                                 // 000000006EC0: BF9F0000
	s_code_end                                                 // 000000006EC4: BF9F0000
	s_code_end                                                 // 000000006EC8: BF9F0000
	s_code_end                                                 // 000000006ECC: BF9F0000
	s_code_end                                                 // 000000006ED0: BF9F0000
	s_code_end                                                 // 000000006ED4: BF9F0000
	s_code_end                                                 // 000000006ED8: BF9F0000
	s_code_end                                                 // 000000006EDC: BF9F0000
	s_code_end                                                 // 000000006EE0: BF9F0000
	s_code_end                                                 // 000000006EE4: BF9F0000
	s_code_end                                                 // 000000006EE8: BF9F0000
	s_code_end                                                 // 000000006EEC: BF9F0000
	s_code_end                                                 // 000000006EF0: BF9F0000
	s_code_end                                                 // 000000006EF4: BF9F0000
	s_code_end                                                 // 000000006EF8: BF9F0000
	s_code_end                                                 // 000000006EFC: BF9F0000
	s_code_end                                                 // 000000006F00: BF9F0000
	s_code_end                                                 // 000000006F04: BF9F0000
	s_code_end                                                 // 000000006F08: BF9F0000
	s_code_end                                                 // 000000006F0C: BF9F0000
	s_code_end                                                 // 000000006F10: BF9F0000
	s_code_end                                                 // 000000006F14: BF9F0000
	s_code_end                                                 // 000000006F18: BF9F0000
	s_code_end                                                 // 000000006F1C: BF9F0000
	s_code_end                                                 // 000000006F20: BF9F0000
	s_code_end                                                 // 000000006F24: BF9F0000
	s_code_end                                                 // 000000006F28: BF9F0000
	s_code_end                                                 // 000000006F2C: BF9F0000
	s_code_end                                                 // 000000006F30: BF9F0000
	s_code_end                                                 // 000000006F34: BF9F0000
	s_code_end                                                 // 000000006F38: BF9F0000
	s_code_end                                                 // 000000006F3C: BF9F0000
	s_code_end                                                 // 000000006F40: BF9F0000
	s_code_end                                                 // 000000006F44: BF9F0000
	s_code_end                                                 // 000000006F48: BF9F0000
	s_code_end                                                 // 000000006F4C: BF9F0000
	s_code_end                                                 // 000000006F50: BF9F0000
	s_code_end                                                 // 000000006F54: BF9F0000
	s_code_end                                                 // 000000006F58: BF9F0000
	s_code_end                                                 // 000000006F5C: BF9F0000
	s_code_end                                                 // 000000006F60: BF9F0000
	s_code_end                                                 // 000000006F64: BF9F0000
	s_code_end                                                 // 000000006F68: BF9F0000
	s_code_end                                                 // 000000006F6C: BF9F0000
	s_code_end                                                 // 000000006F70: BF9F0000
	s_code_end                                                 // 000000006F74: BF9F0000
	s_code_end                                                 // 000000006F78: BF9F0000
	s_code_end                                                 // 000000006F7C: BF9F0000
	s_code_end                                                 // 000000006F80: BF9F0000
	s_code_end                                                 // 000000006F84: BF9F0000
	s_code_end                                                 // 000000006F88: BF9F0000
	s_code_end                                                 // 000000006F8C: BF9F0000
	s_code_end                                                 // 000000006F90: BF9F0000
	s_code_end                                                 // 000000006F94: BF9F0000
	s_code_end                                                 // 000000006F98: BF9F0000
	s_code_end                                                 // 000000006F9C: BF9F0000
	s_code_end                                                 // 000000006FA0: BF9F0000
	s_code_end                                                 // 000000006FA4: BF9F0000
	s_code_end                                                 // 000000006FA8: BF9F0000
	s_code_end                                                 // 000000006FAC: BF9F0000
	s_code_end                                                 // 000000006FB0: BF9F0000
	s_code_end                                                 // 000000006FB4: BF9F0000
	s_code_end                                                 // 000000006FB8: BF9F0000
	s_code_end                                                 // 000000006FBC: BF9F0000
	s_code_end                                                 // 000000006FC0: BF9F0000
	s_code_end                                                 // 000000006FC4: BF9F0000
	s_code_end                                                 // 000000006FC8: BF9F0000
	s_code_end                                                 // 000000006FCC: BF9F0000
	s_code_end                                                 // 000000006FD0: BF9F0000
	s_code_end                                                 // 000000006FD4: BF9F0000
	s_code_end                                                 // 000000006FD8: BF9F0000
	s_code_end                                                 // 000000006FDC: BF9F0000
	s_code_end                                                 // 000000006FE0: BF9F0000
	s_code_end                                                 // 000000006FE4: BF9F0000
	s_code_end                                                 // 000000006FE8: BF9F0000
	s_code_end                                                 // 000000006FEC: BF9F0000
	s_code_end                                                 // 000000006FF0: BF9F0000
	s_code_end                                                 // 000000006FF4: BF9F0000
	s_code_end                                                 // 000000006FF8: BF9F0000
	s_code_end                                                 // 000000006FFC: BF9F0000
