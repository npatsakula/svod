
<stdin>:	file format elf64-amdgpu
	.amdgcn_target "amdgpu-amd-amdhsa-unknown-gfx1201"

Disassembly of section .text:

0000000000000000 <nearest>:
	v_lshrrev_b32_e32 v2, 5, v0                                // 000000000000: 32040085
	v_dual_mov_b32 v82, -1.0 :: v_dual_and_b32 v15, 15, v0     // 000000000004: CA2400F3 520E008F
	v_bfe_u32 v65, v0, 4, 1                                    // 00000000000C: D6100041 02050900
	s_load_b256 s[12:19], s[0:1], 0x0                          // 000000000014: F4006300 F8000000
	s_lshl_b32 s0, ttmp9, 13                                   // 00000000001C: 84008D75
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000000020: BF870112
	v_lshl_or_b32 v64, v2, 4, v15                              // 000000000024: D6560040 043D0902
	v_dual_mov_b32 v119, 0x7f800000 :: v_dual_lshlrev_b32 v86, 3, v65// 00000000002C: CA2200FF 77568283 7F800000
	v_dual_mov_b32 v120, -1.0 :: v_dual_lshlrev_b32 v3, 7, v65 // 000000000038: CA2200F3 78028287
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)// 000000000040: BF8701A3
	v_dual_mov_b32 v84, 0x7f800000 :: v_dual_lshlrev_b32 v1, 7, v64// 000000000044: CA2200FF 54008087 7F800000
	v_dual_mov_b32 v83, -1.0 :: v_dual_lshlrev_b32 v4, 3, v0   // 000000000050: CA2200F3 53040083
	v_lshl_or_b32 v52, v2, 10, v3                              // 000000000058: D6560034 040D1502
	s_delay_alu instid0(VALU_DEP_3)                            // 000000000060: BF870003
	v_or3_b32 v1, v1, s0, v86                                  // 000000000064: D6580001 05580101
	v_or_b32_e32 v53, 2, v65                                   // 00000000006C: 386A8282
	v_or_b32_e32 v54, 4, v65                                   // 000000000070: 386C8284
	v_or_b32_e32 v55, 6, v65                                   // 000000000074: 386E8286
	v_and_or_b32 v91, 0x78, v4, v52                            // 000000000078: D657005B 04D208FF 00000078
	v_ashrrev_i32_e32 v2, 31, v1                               // 000000000084: 3404029F
	v_dual_mov_b32 v85, 0x7f800000 :: v_dual_lshlrev_b32 v52, 1, v52// 000000000088: CA2200FF 55346881 7F800000
	v_lshlrev_b32_e32 v61, 8, v15                              // 000000000094: 307A1E88
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000000098: BF870214
	v_lshlrev_b32_e32 v16, 1, v91                              // 00000000009C: 3020B681
	v_lshlrev_b64_e32 v[1:2], 1, v[1:2]                        // 0000000000A0: 3E020281
	v_or_b32_e32 v95, 16, v86                                  // 0000000000A4: 38BEAC90
	v_or_b32_e32 v57, 8, v65                                   // 0000000000A8: 38728288
	v_or_b32_e32 v58, 10, v65                                  // 0000000000AC: 3874828A
	v_or_b32_e32 v59, 12, v65                                  // 0000000000B0: 3876828C
	v_or_b32_e32 v60, 14, v65                                  // 0000000000B4: 3878828E
	s_wait_kmcnt 0x0                                           // 0000000000B8: BFC70000
	v_add_co_u32 v13, vcc_lo, s14, v1                          // 0000000000BC: D7006A0D 0202020E
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000000C4: BF870001
	v_add_co_ci_u32_e64 v14, null, s15, v2, vcc_lo             // 0000000000C8: D5207C0E 01AA040F
	s_clause 0x3                                               // 0000000000D0: BF850003
	global_load_b128 v[1:4], v16, s[16:17]                     // 0000000000D4: EE05C010 00000001 00000010
	global_load_b128 v[5:8], v16, s[16:17] offset:512          // 0000000000E0: EE05C010 00000005 00020010
	global_load_b128 v[9:12], v16, s[16:17] offset:1024        // 0000000000EC: EE05C010 00000009 00040010
	global_load_b128 v[48:51], v16, s[16:17] offset:1536       // 0000000000F8: EE05C010 00000030 00060010
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
	v_xor_b32_e32 v14, v15, v65                                // 000000000170: 3A1C830F
	v_dual_mov_b32 v88, -1.0 :: v_dual_mov_b32 v87, -1.0       // 000000000174: CA1000F3 585600F3
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 00000000017C: BF870214
	v_lshlrev_b32_e32 v62, 4, v13                              // 000000000180: 307C1A84
	v_cmp_ne_u32_e64 s0, 0, v0                                 // 000000000184: D44D0000 02020080
	v_xor_b32_e32 v0, v15, v53                                 // 00000000018C: 3A006B0F
	v_xor_b32_e32 v53, v15, v54                                // 000000000190: 3A6A6D0F
	v_xor_b32_e32 v15, v15, v55                                // 000000000194: 3A1E6F0F
	v_lshl_or_b32 v109, v14, 4, v52                            // 000000000198: D656006D 04D1090E
	v_xor_b32_e32 v56, v65, v13                                // 0000000001A0: 3A701B41
	v_lshl_or_b32 v110, v0, 4, v52                             // 0000000001A4: D656006E 04D10900
	v_lshl_or_b32 v111, v53, 4, v52                            // 0000000001AC: D656006F 04D10935
	v_lshl_or_b32 v112, v15, 4, v52                            // 0000000001B4: D6560070 04D1090F
	v_lshlrev_b32_e32 v0, 1, v95                               // 0000000001BC: 3000BE81
	v_xor_b32_e32 v54, v54, v13                                // 0000000001C0: 3A6C1B36
	v_xor_b32_e32 v55, v55, v13                                // 0000000001C4: 3A6E1B37
	v_xor_b32_e32 v57, v57, v13                                // 0000000001C8: 3A721B39
	v_xor_b32_e32 v58, v58, v13                                // 0000000001CC: 3A741B3A
	v_xor_b32_e32 v59, v59, v13                                // 0000000001D0: 3A761B3B
	v_xor_b32_e32 v13, v60, v13                                // 0000000001D4: 3A1A1B3C
	v_xor_b32_e32 v0, v62, v0                                  // 0000000001D8: 3A00013E
	v_dual_mov_b32 v90, 0x7f800000 :: v_dual_mov_b32 v89, 0x7f800000// 0000000001DC: CA1000FF 5A5800FF 7F800000
	v_dual_mov_b32 v80, -1.0 :: v_dual_mov_b32 v81, 0x7f800000 // 0000000001E8: CA1000F3 505000FF 7F800000
	v_dual_mov_b32 v74, -1.0 :: v_dual_mov_b32 v75, 0x7f800000 // 0000000001F4: CA1000F3 4A4A00FF 7F800000
	v_dual_mov_b32 v77, -1.0 :: v_dual_mov_b32 v76, 0x7f800000 // 000000000200: CA1000F3 4D4C00FF 7F800000
	v_dual_mov_b32 v79, -1.0 :: v_dual_mov_b32 v78, 0x7f800000 // 00000000020C: CA1000F3 4F4E00FF 7F800000
	v_dual_mov_b32 v72, -1.0 :: v_dual_mov_b32 v71, 0x7f800000 // 000000000218: CA1000F3 484600FF 7F800000
	v_dual_mov_b32 v70, -1.0 :: v_dual_mov_b32 v73, 0x7f800000 // 000000000224: CA1000F3 464800FF 7F800000
	v_dual_mov_b32 v67, -1.0 :: v_dual_mov_b32 v66, -1.0       // 000000000230: CA1000F3 434200F3
	v_dual_mov_b32 v68, 0x7f800000 :: v_dual_mov_b32 v69, 0x7f800000// 000000000238: CA1000FF 444400FF 7F800000
	v_cmp_eq_u32_e64 s1, 0, v65                                // 000000000244: D44A0001 02028280
	v_or_b32_e32 v96, 1, v86                                   // 00000000024C: 38C0AC81
	v_or_b32_e32 v98, 2, v86                                   // 000000000250: 38C4AC82
	v_or_b32_e32 v99, 3, v86                                   // 000000000254: 38C6AC83
	v_or_b32_e32 v101, 4, v86                                  // 000000000258: 38CAAC84
	v_or_b32_e32 v102, 5, v86                                  // 00000000025C: 38CCAC85
	v_or_b32_e32 v103, 6, v86                                  // 000000000260: 38CEAC86
	v_or_b32_e32 v105, 7, v86                                  // 000000000264: 38D2AC87
	v_or_b32_e32 v106, 17, v86                                 // 000000000268: 38D4AC91
	v_or_b32_e32 v107, 18, v86                                 // 00000000026C: 38D6AC92
	v_or_b32_e32 v108, 19, v86                                 // 000000000270: 38D8AC93
	v_or_b32_e32 v113, 20, v86                                 // 000000000274: 38E2AC94
	v_or_b32_e32 v116, 21, v86                                 // 000000000278: 38E8AC95
	v_or_b32_e32 v117, 22, v86                                 // 00000000027C: 38EAAC96
	v_or_b32_e32 v118, 23, v86                                 // 000000000280: 38ECAC97
	v_lshl_or_b32 v114, v56, 4, v61                            // 000000000284: D6560072 04F50938
	v_lshl_or_b32 v104, v54, 4, v61                            // 00000000028C: D6560068 04F50936
	v_lshl_or_b32 v100, v55, 4, v61                            // 000000000294: D6560064 04F50937
	v_lshl_or_b32 v97, v57, 4, v61                             // 00000000029C: D6560061 04F50939
	v_lshl_or_b32 v94, v58, 4, v61                             // 0000000002A4: D656005E 04F5093A
	v_lshl_or_b32 v93, v59, 4, v61                             // 0000000002AC: D656005D 04F5093B
	v_lshl_or_b32 v92, v13, 4, v61                             // 0000000002B4: D656005C 04F5090D
	v_or_b32_e32 v115, v0, v61                                 // 0000000002BC: 38E67B00
	s_mov_b32 s2, 1                                            // 0000000002C0: BE820081
	s_mov_b32 s3, 0                                            // 0000000002C4: BE830080
	s_mov_b32 s10, 0x76543210                                  // 0000000002C8: BE8A00FF 76543210
	s_xor_b32 s9, s0, -1                                       // 0000000002D0: 8D09C100
	s_wait_loadcnt 0xb                                         // 0000000002D4: BFC0000B
	ds_store_b128 v109, v[1:4]                                 // 0000000002D8: DB7C0000 0000016D
	s_wait_loadcnt 0xa                                         // 0000000002E0: BFC0000A
	ds_store_b128 v110, v[5:8] offset:512                      // 0000000002E4: DB7C0200 0000056E
	s_wait_loadcnt 0x9                                         // 0000000002EC: BFC00009
	ds_store_b128 v111, v[9:12] offset:1024                    // 0000000002F0: DB7C0400 0000096F
	s_wait_loadcnt 0x8                                         // 0000000002F8: BFC00008
	ds_store_b128 v112, v[48:51] offset:1536                   // 0000000002FC: DB7C0600 00003070
	s_wait_dscnt 0x0                                           // 000000000304: BFC60000
	s_barrier_signal -1                                        // 000000000308: BE804EC1
	s_barrier_wait 0xffff                                      // 00000000030C: BF94FFFF
	s_branch 30                                                // 000000000310: BFA0001E <nearest+0x38c>
	s_and_b32 s2, s14, 0x1000                                  // 000000000314: 8B02FF0E 00001000
	s_wait_alu depctr_sa_sdst(0)                               // 00000000031C: BF88FF9E
	s_lshl_b32 s2, s2, 1                                       // 000000000320: 84028102
	s_wait_alu depctr_sa_sdst(0)                               // 000000000324: BF88FF9E
	v_add_nc_u32_e32 v0, s2, v109                              // 000000000328: 4A00DA02
	v_add_nc_u32_e32 v1, s2, v110                              // 00000000032C: 4A02DC02
	v_add_nc_u32_e32 v2, s2, v111                              // 000000000330: 4A04DE02
	v_add_nc_u32_e32 v3, s2, v112                              // 000000000334: 4A06E002
	s_wait_loadcnt 0x3                                         // 000000000338: BFC00003
	ds_store_b128 v0, v[48:51]                                 // 00000000033C: DB7C0000 00003000
	s_wait_loadcnt 0x2                                         // 000000000344: BFC00002
	ds_store_b128 v1, v[52:55] offset:512                      // 000000000348: DB7C0200 00003401
	s_wait_loadcnt 0x1                                         // 000000000350: BFC00001
	ds_store_b128 v2, v[56:59] offset:1024                     // 000000000354: DB7C0400 00003802
	s_wait_loadcnt 0x0                                         // 00000000035C: BFC00000
	ds_store_b128 v3, v[60:63] offset:1536                     // 000000000360: DB7C0600 00003C03
	s_wait_dscnt 0x0                                           // 000000000368: BFC60000
	s_barrier_signal -1                                        // 00000000036C: BE804EC1
	s_add_co_i32 s2, s11, 1                                    // 000000000370: 8102810B
	s_mov_b32 s3, s11                                          // 000000000374: BE83000B
	s_wait_alu depctr_sa_sdst(0)                               // 000000000378: BF88FF9E
	s_cmp_eq_u32 s2, 0x80                                      // 00000000037C: BF06FF02 00000080
	s_barrier_wait 0xffff                                      // 000000000384: BF94FFFF
	s_cbranch_scc1 2993                                        // 000000000388: BFA20BB1 <nearest+0x3250>
	s_lshl_b32 s14, s2, 12                                     // 00000000038C: 840E8C02
	s_mov_b32 s11, s2                                          // 000000000390: BE8B0002
	s_wait_alu depctr_sa_sdst(0)                               // 000000000394: BF88FF9E
	v_or_b32_e32 v0, s14, v91                                  // 000000000398: 3800B60E
	s_lshl_b32 s2, s3, 13                                      // 00000000039C: 84028D03
	s_wait_alu depctr_sa_sdst(0)                               // 0000000003A0: BF88FF9E
	s_and_b32 s2, s2, 0x2000                                   // 0000000003A4: 8B02FF02 00002000
	s_wait_alu depctr_sa_sdst(0)                               // 0000000003AC: BF88FF9E
	v_or_b32_e32 v1, s2, v115                                  // 0000000003B0: 3802E602
	v_lshlrev_b32_e32 v0, 1, v0                                // 0000000003B4: 30000081
	v_or_b32_e32 v145, s2, v104                                // 0000000003B8: 3922D002
	v_or_b32_e32 v153, s2, v100                                // 0000000003BC: 3932C802
	v_or_b32_e32 v157, s2, v97                                 // 0000000003C0: 393AC202
	s_clause 0x3                                               // 0000000003C4: BF850003
	global_load_b128 v[48:51], v0, s[16:17]                    // 0000000003C8: EE05C010 00000030 00000000
	global_load_b128 v[52:55], v0, s[16:17] offset:512         // 0000000003D4: EE05C010 00000034 00020000
	global_load_b128 v[56:59], v0, s[16:17] offset:1024        // 0000000003E0: EE05C010 00000038 00040000
	global_load_b128 v[60:63], v0, s[16:17] offset:1536        // 0000000003EC: EE05C010 0000003C 00060000
	v_or_b32_e32 v0, s2, v114                                  // 0000000003F8: 3800E402
	v_or_b32_e32 v161, s2, v94                                 // 0000000003FC: 3942BC02
	v_or_b32_e32 v162, s2, v93                                 // 000000000400: 3944BA02
	v_or_b32_e32 v163, s2, v92                                 // 000000000404: 3946B802
	ds_load_b128 v[8:11], v0                                   // 000000000408: DBFC0000 08000000
	ds_load_b128 v[12:15], v1                                  // 000000000410: DBFC0000 0C000001
	ds_load_b128 v[121:124], v145                              // 000000000418: DBFC0000 79000091
	ds_load_b128 v[125:128], v0 offset:4096                    // 000000000420: DBFC1000 7D000000
	ds_load_b128 v[129:132], v153                              // 000000000428: DBFC0000 81000099
	ds_load_b128 v[133:136], v157                              // 000000000430: DBFC0000 8500009D
	ds_load_b128 v[137:140], v1 offset:4096                    // 000000000438: DBFC1000 89000001
	ds_load_b128 v[141:144], v161                              // 000000000440: DBFC0000 8D0000A1
	ds_load_b128 v[145:148], v145 offset:4096                  // 000000000448: DBFC1000 91000091
	ds_load_b128 v[149:152], v162                              // 000000000450: DBFC0000 950000A2
	ds_load_b128 v[153:156], v153 offset:4096                  // 000000000458: DBFC1000 99000099
	s_wait_loadcnt_dscnt 0xb0a                                 // 000000000460: BFC80B0A
	v_wmma_f32_16x16x16_f16 v[0:7], v[8:11], v[44:47], 0       // 000000000464: CC404000 1A025908
	s_wait_loadcnt_dscnt 0xa09                                 // 00000000046C: BFC80A09
	s_delay_alu instid0(VALU_DEP_1)                            // 000000000470: BF870001
	v_wmma_f32_16x16x16_f16 v[0:7], v[12:15], v[40:43], v[0:7] // 000000000474: CC404000 1C02510C
	s_wait_dscnt 0x7                                           // 00000000047C: BFC60007
	v_wmma_f32_16x16x16_f16 v[8:15], v[125:128], v[44:47], 0   // 000000000480: CC404008 1A02597D
	ds_load_b128 v[125:128], v161 offset:4096                  // 000000000488: DBFC1000 7D0000A1
	s_wait_loadcnt 0x9                                         // 000000000490: BFC00009
	v_wmma_f32_16x16x16_f16 v[0:7], v[121:124], v[36:39], v[0:7]// 000000000494: CC404000 1C024979
	ds_load_b128 v[121:124], v163                              // 00000000049C: DBFC0000 790000A3
	ds_load_b128 v[157:160], v157 offset:4096                  // 0000000004A4: DBFC1000 9D00009D
	s_wait_dscnt 0x7                                           // 0000000004AC: BFC60007
	v_wmma_f32_16x16x16_f16 v[8:15], v[137:140], v[40:43], v[8:15]// 0000000004B0: CC404008 1C225189
	s_wait_loadcnt 0x8                                         // 0000000004B8: BFC00008
	v_wmma_f32_16x16x16_f16 v[0:7], v[129:132], v[32:35], v[0:7]// 0000000004BC: CC404000 1C024181
	ds_load_b128 v[129:132], v162 offset:4096                  // 0000000004C4: DBFC1000 810000A2
	s_wait_dscnt 0x6                                           // 0000000004CC: BFC60006
	v_wmma_f32_16x16x16_f16 v[8:15], v[145:148], v[36:39], v[8:15]// 0000000004D0: CC404008 1C224991
	s_wait_loadcnt 0x7                                         // 0000000004D8: BFC00007
	v_wmma_f32_16x16x16_f16 v[0:7], v[133:136], v[28:31], v[0:7]// 0000000004DC: CC404000 1C023985
	ds_load_b128 v[133:136], v163 offset:4096                  // 0000000004E4: DBFC1000 850000A3
	s_wait_dscnt 0x5                                           // 0000000004EC: BFC60005
	v_wmma_f32_16x16x16_f16 v[8:15], v[153:156], v[32:35], v[8:15]// 0000000004F0: CC404008 1C224199
	s_wait_loadcnt 0x6                                         // 0000000004F8: BFC00006
	v_wmma_f32_16x16x16_f16 v[0:7], v[141:144], v[24:27], v[0:7]// 0000000004FC: CC404000 1C02318D
	s_wait_loadcnt 0x5                                         // 000000000504: BFC00005
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)// 000000000508: BF870141
	v_wmma_f32_16x16x16_f16 v[0:7], v[149:152], v[20:23], v[0:7]// 00000000050C: CC404000 1C022995
	s_wait_dscnt 0x2                                           // 000000000514: BFC60002
	v_wmma_f32_16x16x16_f16 v[8:15], v[157:160], v[28:31], v[8:15]// 000000000518: CC404008 1C22399D
	s_wait_loadcnt 0x4                                         // 000000000520: BFC00004
	v_wmma_f32_16x16x16_f16 v[0:7], v[121:124], v[16:19], v[0:7]// 000000000524: CC404000 1C022179
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000052C: BF870112
	v_wmma_f32_16x16x16_f16 v[8:15], v[125:128], v[24:27], v[8:15]// 000000000530: CC404008 1C22317D
	v_min3_num_f32 v121, -v0, -v1, -v2                         // 000000000538: D6290079 E40A0300
	s_wait_dscnt 0x1                                           // 000000000540: BFC60001
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000000544: BF870112
	v_wmma_f32_16x16x16_f16 v[8:15], v[129:132], v[20:23], v[8:15]// 000000000548: CC404008 1C222981
	v_min3_num_f32 v121, v121, -v3, -v4                        // 000000000550: D6290079 C4120779
	s_wait_dscnt 0x0                                           // 000000000558: BFC60000
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000055C: BF870112
	v_wmma_f32_16x16x16_f16 v[8:15], v[133:136], v[16:19], v[8:15]// 000000000560: CC404008 1C222185
	v_min3_num_f32 v121, v121, -v5, -v6                        // 000000000568: D6290079 C41A0B79
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000000570: BF870112
	v_max_num_f32_e64 v122, -v15, -v15                         // 000000000574: D516007A 62021F0F
	v_min3_num_f32 v121, v121, -v7, -v8                        // 00000000057C: D6290079 C4220F79
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000000584: BF870091
	v_min3_num_f32 v121, v121, -v9, -v10                       // 000000000588: D6290079 C42A1379
	v_min3_num_f32 v121, v121, -v11, -v12                      // 000000000590: D6290079 C4321779
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000000598: BF870091
	v_min3_num_f32 v121, v121, -v13, -v14                      // 00000000059C: D6290079 C43A1B79
	v_min_num_f32_e32 v121, v121, v122                         // 0000000005A4: 2AF2F579
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000005A8: BF870091
	v_mov_b32_e32 v122, v121                                   // 0000000005AC: 7EF40379
	v_permlanex16_b32 v122, v122, s10, 0xfedcba98              // 0000000005B0: D65C007A 03FC157A FEDCBA98
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000005BC: BF870091
	v_max_num_f32_e64 v122, -v122, -v122                       // 0000000005C0: D516007A 6202F57A
	v_max_num_f32_e64 v121, -v121, v122                        // 0000000005C8: D5160079 2202F579
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000005D0: BF870001
	v_cmp_ge_f32_e64 s2, v119, -v121                           // 0000000005D4: D4160002 4202F377
	s_mov_b32 vcc_lo, s2                                       // 0000000005DC: BEEA0002
	s_wait_alu depctr_sa_sdst(0)                               // 0000000005E0: BF88FF9E
	s_cbranch_vccz 65355                                       // 0000000005E4: BFA3FF4B <nearest+0x314>
	v_cmp_ge_f32_e64 s4, v119, -v0                             // 0000000005E8: D4160004 42020177
	v_cmp_ge_f32_e64 s5, v119, -v1                             // 0000000005F0: D4160005 42020377
	s_lshl_b32 s7, s3, 5                                       // 0000000005F8: 84078503
	s_wait_alu depctr_sa_sdst(0)                               // 0000000005FC: BF88FF9E
	v_or_b32_e32 v122, s7, v86                                 // 000000000600: 38F4AC07
	s_and_b32 s4, s4, s2                                       // 000000000604: 8B040204
	s_and_b32 s6, s5, s2                                       // 000000000608: 8B060205
	v_or_b32_e32 v123, s7, v96                                 // 00000000060C: 38F6C007
	s_wait_alu depctr_sa_sdst(0)                               // 000000000610: BF88FF9E
	s_or_b32 s3, s4, s6                                        // 000000000614: 8C030604
	s_wait_alu depctr_sa_sdst(0)                               // 000000000618: BF88FF9E
	v_cndmask_b32_e64 v121, 0, 1, s3                           // 00000000061C: D5010079 000D0280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000000624: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v121                           // 000000000628: 7C9AF280
	v_cvt_f32_u32_e32 v121, v122                               // 00000000062C: 7EF20D7A
	v_cvt_f32_u32_e32 v122, v123                               // 000000000630: 7EF40D7B
	s_cbranch_vccz 30                                          // 000000000634: BFA3001E <nearest+0x6b0>
	v_cmp_gt_f32_e64 s8, v119, -v0                             // 000000000638: D4140008 42020177
	s_delay_alu instid0(VALU_DEP_3)                            // 000000000640: BF870003
	v_cmp_gt_f32_e32 vcc_lo, v120, v121                        // 000000000644: 7C28F378
	v_cmp_gt_f32_e64 s15, v119, -v1                            // 000000000648: D414000F 42020377
	v_cmp_gt_f32_e64 s3, v120, v122                            // 000000000650: D4140003 0202F578
	v_cmp_lt_f32_e64 s5, -1.0, v82                             // 000000000658: D4110005 0202A4F3
	s_or_b32 s8, vcc_lo, s8                                    // 000000000660: 8C08086A
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v83                         // 000000000664: 7C22A6F3
	s_or_b32 s15, s3, s15                                      // 000000000668: 8C0F0F03
	s_and_b32 s3, s8, s4                                       // 00000000066C: 8B030408
	s_wait_alu depctr_sa_sdst(0)                               // 000000000670: BF88FF9E
	s_and_b32 s4, s15, s6                                      // 000000000674: 8B04060F
	s_and_b32 s5, s5, s3                                       // 000000000678: 8B050305
	s_wait_alu depctr_sa_sdst(0)                               // 00000000067C: BF88FF9E
	s_and_b32 s6, vcc_lo, s4                                   // 000000000680: 8B06046A
	s_wait_alu depctr_sa_sdst(0)                               // 000000000684: BF88FF9E
	s_or_b32 s5, s5, s6                                        // 000000000688: 8C050605
	s_wait_alu depctr_sa_sdst(0)                               // 00000000068C: BF88FF9E
	v_cndmask_b32_e64 v123, 0, 1, s5                           // 000000000690: D501007B 00150280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000000698: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v123                           // 00000000069C: 7C9AF680
	s_cbranch_vccz 7                                           // 0000000006A0: BFA30007 <nearest+0x6c0>
	s_mov_b32 s6, 1.0                                          // 0000000006A4: BE8600F2
	s_mov_b32 s15, 0                                           // 0000000006A8: BE8F0080
	s_branch 27                                                // 0000000006AC: BFA0001B <nearest+0x71c>
	s_mov_b32 s15, 0x41000000                                  // 0000000006B0: BE8F00FF 41000000
	s_mov_b32 s6, 0                                            // 0000000006B8: BE860080
	s_branch 23                                                // 0000000006BC: BFA00017 <nearest+0x71c>
	v_cndmask_b32_e64 v123, v90, -v0, s3                       // 0000000006C0: D501007B 400E015A
	v_cndmask_b32_e64 v124, v89, -v1, s4                       // 0000000006C8: D501007C 40120359
	v_cndmask_b32_e64 v125, v88, v121, s3                      // 0000000006D0: D501007D 000EF358
	v_cndmask_b32_e64 v126, v87, v122, s4                      // 0000000006D8: D501007E 0012F557
	v_cndmask_b32_e64 v84, v84, v90, s3                        // 0000000006E0: D5010054 000EB554
	v_cndmask_b32_e64 v85, v85, v89, s4                        // 0000000006E8: D5010055 0012B355
	v_cndmask_b32_e64 v82, v82, v88, s3                        // 0000000006F0: D5010052 000EB152
	v_cndmask_b32_e64 v83, v83, v87, s4                        // 0000000006F8: D5010053 0012AF53
	v_dual_mov_b32 v89, v124 :: v_dual_mov_b32 v90, v123       // 000000000700: CA10017C 595A017B
	v_dual_mov_b32 v87, v126 :: v_dual_mov_b32 v88, v125       // 000000000708: CA10017E 5758017D
	s_mov_b32 s6, 0                                            // 000000000710: BE860080
	s_mov_b32 s15, 0x41000000                                  // 000000000714: BE8F00FF 41000000
	s_wait_alu depctr_sa_sdst(0)                               // 00000000071C: BF88FF9E
	s_cmp_lt_f32 s6, 0.5                                       // 000000000720: BF41F006
	v_cmp_ge_f32_e64 s3, v119, -v2                             // 000000000724: D4160003 42020577
	v_cmp_ge_f32_e64 s5, v119, -v3                             // 00000000072C: D4160005 42020777
	v_or_b32_e32 v124, s7, v98                                 // 000000000734: 38F8C407
	s_cselect_b32 s4, -1, 0                                    // 000000000738: 980480C1
	v_or_b32_e32 v125, s7, v99                                 // 00000000073C: 38FAC607
	s_wait_alu depctr_sa_sdst(0)                               // 000000000740: BF88FF9E
	s_and_b32 s8, s2, s4                                       // 000000000744: 8B080402
	s_and_b32 s4, s3, s8                                       // 000000000748: 8B040803
	s_and_b32 s8, s5, s8                                       // 00000000074C: 8B080805
	s_wait_alu depctr_sa_sdst(0)                               // 000000000750: BF88FF9E
	s_or_b32 s3, s4, s8                                        // 000000000754: 8C030804
	s_wait_alu depctr_sa_sdst(0)                               // 000000000758: BF88FF9E
	v_cndmask_b32_e64 v123, 0, 1, s3                           // 00000000075C: D501007B 000D0280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000000764: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v123                           // 000000000768: 7C9AF680
	v_cvt_f32_u32_e32 v123, v124                               // 00000000076C: 7EF60D7C
	v_cvt_f32_u32_e32 v124, v125                               // 000000000770: 7EF80D7D
	s_cbranch_vccz 49                                          // 000000000774: BFA30031 <nearest+0x83c>
	v_cmp_gt_f32_e64 s20, v119, -v2                            // 000000000778: D4140014 42020577
	s_delay_alu instid0(VALU_DEP_3)                            // 000000000780: BF870003
	v_cmp_gt_f32_e32 vcc_lo, v120, v123                        // 000000000784: 7C28F778
	v_cmp_gt_f32_e64 s21, v119, -v3                            // 000000000788: D4140015 42020777
	v_cmp_gt_f32_e64 s3, v120, v124                            // 000000000790: D4140003 0202F978
	v_cmp_lt_f32_e64 s5, -1.0, v82                             // 000000000798: D4110005 0202A4F3
	s_or_b32 s20, vcc_lo, s20                                  // 0000000007A0: 8C14146A
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v83                         // 0000000007A4: 7C22A6F3
	s_or_b32 s21, s3, s21                                      // 0000000007A8: 8C151503
	s_and_b32 s3, s20, s4                                      // 0000000007AC: 8B030414
	s_and_b32 s4, s21, s8                                      // 0000000007B0: 8B040815
	s_wait_alu depctr_sa_sdst(0)                               // 0000000007B4: BF88FF9E
	s_and_b32 s5, s5, s3                                       // 0000000007B8: 8B050305
	s_and_b32 s8, s4, vcc_lo                                   // 0000000007BC: 8B086A04
	s_wait_alu depctr_sa_sdst(0)                               // 0000000007C0: BF88FF9E
	s_or_b32 s5, s5, s8                                        // 0000000007C4: 8C050805
	s_wait_alu depctr_sa_sdst(0)                               // 0000000007C8: BF88FF9E
	v_cndmask_b32_e64 v125, 0, 1, s5                           // 0000000007CC: D501007D 00150280
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000007D4: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v125                           // 0000000007D8: 7C9AFA80
	s_cbranch_vccz 3                                           // 0000000007DC: BFA30003 <nearest+0x7ec>
	s_mov_b32 s15, 1.0                                         // 0000000007E0: BE8F00F2
	s_mov_b32 s6, 1.0                                          // 0000000007E4: BE8600F2
	s_branch 20                                                // 0000000007E8: BFA00014 <nearest+0x83c>
	v_cndmask_b32_e64 v125, v90, -v2, s3                       // 0000000007EC: D501007D 400E055A
	v_cndmask_b32_e64 v126, v89, -v3, s4                       // 0000000007F4: D501007E 40120759
	v_cndmask_b32_e64 v127, v88, v123, s3                      // 0000000007FC: D501007F 000EF758
	v_cndmask_b32_e64 v128, v87, v124, s4                      // 000000000804: D5010080 0012F957
	v_cndmask_b32_e64 v84, v84, v90, s3                        // 00000000080C: D5010054 000EB554
	v_cndmask_b32_e64 v85, v85, v89, s4                        // 000000000814: D5010055 0012B355
	v_cndmask_b32_e64 v82, v82, v88, s3                        // 00000000081C: D5010052 000EB152
	v_cndmask_b32_e64 v83, v83, v87, s4                        // 000000000824: D5010053 0012AF53
	v_dual_mov_b32 v89, v126 :: v_dual_mov_b32 v90, v125       // 00000000082C: CA10017E 595A017D
	v_dual_mov_b32 v87, v128 :: v_dual_mov_b32 v88, v127       // 000000000834: CA100180 5758017F
	s_wait_alu depctr_sa_sdst(0)                               // 00000000083C: BF88FF9E
	s_cmp_lt_f32 s6, 0.5                                       // 000000000840: BF41F006
	v_cmp_ge_f32_e64 s3, v119, -v4                             // 000000000844: D4160003 42020977
	v_cmp_ge_f32_e64 s5, v119, -v5                             // 00000000084C: D4160005 42020B77
	v_or_b32_e32 v126, s7, v101                                // 000000000854: 38FCCA07
	s_cselect_b32 s4, -1, 0                                    // 000000000858: 980480C1
	v_or_b32_e32 v127, s7, v102                                // 00000000085C: 38FECC07
	s_wait_alu depctr_sa_sdst(0)                               // 000000000860: BF88FF9E
	s_and_b32 s8, s2, s4                                       // 000000000864: 8B080402
	s_and_b32 s4, s3, s8                                       // 000000000868: 8B040803
	s_and_b32 s8, s5, s8                                       // 00000000086C: 8B080805
	s_wait_alu depctr_sa_sdst(0)                               // 000000000870: BF88FF9E
	s_or_b32 s3, s4, s8                                        // 000000000874: 8C030804
	s_wait_alu depctr_sa_sdst(0)                               // 000000000878: BF88FF9E
	v_cndmask_b32_e64 v125, 0, 1, s3                           // 00000000087C: D501007D 000D0280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000000884: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v125                           // 000000000888: 7C9AFA80
	v_cvt_f32_u32_e32 v125, v126                               // 00000000088C: 7EFA0D7E
	v_cvt_f32_u32_e32 v126, v127                               // 000000000890: 7EFC0D7F
	s_cbranch_vccz 49                                          // 000000000894: BFA30031 <nearest+0x95c>
	v_cmp_gt_f32_e64 s20, v119, -v4                            // 000000000898: D4140014 42020977
	s_delay_alu instid0(VALU_DEP_3)                            // 0000000008A0: BF870003
	v_cmp_gt_f32_e32 vcc_lo, v120, v125                        // 0000000008A4: 7C28FB78
	v_cmp_gt_f32_e64 s21, v119, -v5                            // 0000000008A8: D4140015 42020B77
	v_cmp_gt_f32_e64 s3, v120, v126                            // 0000000008B0: D4140003 0202FD78
	v_cmp_lt_f32_e64 s5, -1.0, v82                             // 0000000008B8: D4110005 0202A4F3
	s_or_b32 s20, vcc_lo, s20                                  // 0000000008C0: 8C14146A
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v83                         // 0000000008C4: 7C22A6F3
	s_or_b32 s21, s3, s21                                      // 0000000008C8: 8C151503
	s_and_b32 s3, s20, s4                                      // 0000000008CC: 8B030414
	s_and_b32 s4, s21, s8                                      // 0000000008D0: 8B040815
	s_wait_alu depctr_sa_sdst(0)                               // 0000000008D4: BF88FF9E
	s_and_b32 s5, s5, s3                                       // 0000000008D8: 8B050305
	s_and_b32 s8, s4, vcc_lo                                   // 0000000008DC: 8B086A04
	s_wait_alu depctr_sa_sdst(0)                               // 0000000008E0: BF88FF9E
	s_or_b32 s5, s5, s8                                        // 0000000008E4: 8C050805
	s_wait_alu depctr_sa_sdst(0)                               // 0000000008E8: BF88FF9E
	v_cndmask_b32_e64 v127, 0, 1, s5                           // 0000000008EC: D501007F 00150280
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000008F4: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v127                           // 0000000008F8: 7C9AFE80
	s_cbranch_vccz 3                                           // 0000000008FC: BFA30003 <nearest+0x90c>
	s_mov_b32 s6, 1.0                                          // 000000000900: BE8600F2
	s_mov_b32 s15, 2.0                                         // 000000000904: BE8F00F4
	s_branch 20                                                // 000000000908: BFA00014 <nearest+0x95c>
	v_cndmask_b32_e64 v127, v90, -v4, s3                       // 00000000090C: D501007F 400E095A
	v_cndmask_b32_e64 v128, v89, -v5, s4                       // 000000000914: D5010080 40120B59
	v_cndmask_b32_e64 v129, v88, v125, s3                      // 00000000091C: D5010081 000EFB58
	v_cndmask_b32_e64 v130, v87, v126, s4                      // 000000000924: D5010082 0012FD57
	v_cndmask_b32_e64 v84, v84, v90, s3                        // 00000000092C: D5010054 000EB554
	v_cndmask_b32_e64 v85, v85, v89, s4                        // 000000000934: D5010055 0012B355
	v_cndmask_b32_e64 v82, v82, v88, s3                        // 00000000093C: D5010052 000EB152
	v_cndmask_b32_e64 v83, v83, v87, s4                        // 000000000944: D5010053 0012AF53
	v_dual_mov_b32 v89, v128 :: v_dual_mov_b32 v90, v127       // 00000000094C: CA100180 595A017F
	v_dual_mov_b32 v87, v130 :: v_dual_mov_b32 v88, v129       // 000000000954: CA100182 57580181
	s_wait_alu depctr_sa_sdst(0)                               // 00000000095C: BF88FF9E
	s_cmp_lt_f32 s6, 0.5                                       // 000000000960: BF41F006
	v_cmp_ge_f32_e64 s3, v119, -v6                             // 000000000964: D4160003 42020D77
	v_cmp_ge_f32_e64 s5, v119, -v7                             // 00000000096C: D4160005 42020F77
	v_or_b32_e32 v128, s7, v103                                // 000000000974: 3900CE07
	s_cselect_b32 s4, -1, 0                                    // 000000000978: 980480C1
	v_or_b32_e32 v129, s7, v105                                // 00000000097C: 3902D207
	s_wait_alu depctr_sa_sdst(0)                               // 000000000980: BF88FF9E
	s_and_b32 s8, s2, s4                                       // 000000000984: 8B080402
	s_and_b32 s4, s3, s8                                       // 000000000988: 8B040803
	s_and_b32 s8, s5, s8                                       // 00000000098C: 8B080805
	s_wait_alu depctr_sa_sdst(0)                               // 000000000990: BF88FF9E
	s_or_b32 s3, s4, s8                                        // 000000000994: 8C030804
	s_wait_alu depctr_sa_sdst(0)                               // 000000000998: BF88FF9E
	v_cndmask_b32_e64 v127, 0, 1, s3                           // 00000000099C: D501007F 000D0280
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000009A4: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v127                           // 0000000009A8: 7C9AFE80
	v_cvt_f32_u32_e32 v127, v128                               // 0000000009AC: 7EFE0D80
	v_cvt_f32_u32_e32 v128, v129                               // 0000000009B0: 7F000D81
	s_cbranch_vccz 50                                          // 0000000009B4: BFA30032 <nearest+0xa80>
	v_cmp_gt_f32_e64 s20, v119, -v6                            // 0000000009B8: D4140014 42020D77
	s_delay_alu instid0(VALU_DEP_3)                            // 0000000009C0: BF870003
	v_cmp_gt_f32_e32 vcc_lo, v120, v127                        // 0000000009C4: 7C28FF78
	v_cmp_gt_f32_e64 s21, v119, -v7                            // 0000000009C8: D4140015 42020F77
	v_cmp_gt_f32_e64 s3, v120, v128                            // 0000000009D0: D4140003 02030178
	v_cmp_lt_f32_e64 s5, -1.0, v82                             // 0000000009D8: D4110005 0202A4F3
	s_or_b32 s20, vcc_lo, s20                                  // 0000000009E0: 8C14146A
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v83                         // 0000000009E4: 7C22A6F3
	s_or_b32 s21, s3, s21                                      // 0000000009E8: 8C151503
	s_and_b32 s3, s20, s4                                      // 0000000009EC: 8B030414
	s_and_b32 s4, s21, s8                                      // 0000000009F0: 8B040815
	s_wait_alu depctr_sa_sdst(0)                               // 0000000009F4: BF88FF9E
	s_and_b32 s5, s5, s3                                       // 0000000009F8: 8B050305
	s_and_b32 s8, s4, vcc_lo                                   // 0000000009FC: 8B086A04
	s_wait_alu depctr_sa_sdst(0)                               // 000000000A00: BF88FF9E
	s_or_b32 s5, s5, s8                                        // 000000000A04: 8C050805
	s_wait_alu depctr_sa_sdst(0)                               // 000000000A08: BF88FF9E
	v_cndmask_b32_e64 v129, 0, 1, s5                           // 000000000A0C: D5010081 00150280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000000A14: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v129                           // 000000000A18: 7C9B0280
	s_cbranch_vccz 4                                           // 000000000A1C: BFA30004 <nearest+0xa30>
	s_mov_b32 s6, 1.0                                          // 000000000A20: BE8600F2
	s_mov_b32 s15, 0x40400000                                  // 000000000A24: BE8F00FF 40400000
	s_branch 20                                                // 000000000A2C: BFA00014 <nearest+0xa80>
	v_cndmask_b32_e64 v129, v90, -v6, s3                       // 000000000A30: D5010081 400E0D5A
	v_cndmask_b32_e64 v130, v89, -v7, s4                       // 000000000A38: D5010082 40120F59
	v_cndmask_b32_e64 v131, v88, v127, s3                      // 000000000A40: D5010083 000EFF58
	v_cndmask_b32_e64 v132, v87, v128, s4                      // 000000000A48: D5010084 00130157
	v_cndmask_b32_e64 v84, v84, v90, s3                        // 000000000A50: D5010054 000EB554
	v_cndmask_b32_e64 v85, v85, v89, s4                        // 000000000A58: D5010055 0012B355
	v_cndmask_b32_e64 v82, v82, v88, s3                        // 000000000A60: D5010052 000EB152
	v_cndmask_b32_e64 v83, v83, v87, s4                        // 000000000A68: D5010053 0012AF53
	v_dual_mov_b32 v89, v130 :: v_dual_mov_b32 v90, v129       // 000000000A70: CA100182 595A0181
	v_dual_mov_b32 v87, v132 :: v_dual_mov_b32 v88, v131       // 000000000A78: CA100184 57580183
	s_wait_alu depctr_sa_sdst(0)                               // 000000000A80: BF88FF9E
	s_cmp_lt_f32 s6, 0.5                                       // 000000000A84: BF41F006
	v_cmp_ge_f32_e64 s3, v119, -v8                             // 000000000A88: D4160003 42021177
	v_cmp_ge_f32_e64 s5, v119, -v9                             // 000000000A90: D4160005 42021377
	v_or_b32_e32 v130, s7, v95                                 // 000000000A98: 3904BE07
	s_cselect_b32 s4, -1, 0                                    // 000000000A9C: 980480C1
	v_or_b32_e32 v131, s7, v106                                // 000000000AA0: 3906D407
	s_wait_alu depctr_sa_sdst(0)                               // 000000000AA4: BF88FF9E
	s_and_b32 s8, s2, s4                                       // 000000000AA8: 8B080402
	s_and_b32 s4, s3, s8                                       // 000000000AAC: 8B040803
	s_and_b32 s8, s5, s8                                       // 000000000AB0: 8B080805
	s_wait_alu depctr_sa_sdst(0)                               // 000000000AB4: BF88FF9E
	s_or_b32 s3, s4, s8                                        // 000000000AB8: 8C030804
	s_wait_alu depctr_sa_sdst(0)                               // 000000000ABC: BF88FF9E
	v_cndmask_b32_e64 v129, 0, 1, s3                           // 000000000AC0: D5010081 000D0280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000000AC8: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v129                           // 000000000ACC: 7C9B0280
	v_cvt_f32_u32_e32 v129, v130                               // 000000000AD0: 7F020D82
	v_cvt_f32_u32_e32 v130, v131                               // 000000000AD4: 7F040D83
	s_cbranch_vccz 49                                          // 000000000AD8: BFA30031 <nearest+0xba0>
	v_cmp_gt_f32_e64 s20, v119, -v8                            // 000000000ADC: D4140014 42021177
	s_delay_alu instid0(VALU_DEP_3)                            // 000000000AE4: BF870003
	v_cmp_gt_f32_e32 vcc_lo, v120, v129                        // 000000000AE8: 7C290378
	v_cmp_gt_f32_e64 s21, v119, -v9                            // 000000000AEC: D4140015 42021377
	v_cmp_gt_f32_e64 s3, v120, v130                            // 000000000AF4: D4140003 02030578
	v_cmp_lt_f32_e64 s5, -1.0, v82                             // 000000000AFC: D4110005 0202A4F3
	s_or_b32 s20, vcc_lo, s20                                  // 000000000B04: 8C14146A
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v83                         // 000000000B08: 7C22A6F3
	s_or_b32 s21, s3, s21                                      // 000000000B0C: 8C151503
	s_and_b32 s3, s20, s4                                      // 000000000B10: 8B030414
	s_and_b32 s4, s21, s8                                      // 000000000B14: 8B040815
	s_wait_alu depctr_sa_sdst(0)                               // 000000000B18: BF88FF9E
	s_and_b32 s5, s5, s3                                       // 000000000B1C: 8B050305
	s_and_b32 s8, s4, vcc_lo                                   // 000000000B20: 8B086A04
	s_wait_alu depctr_sa_sdst(0)                               // 000000000B24: BF88FF9E
	s_or_b32 s5, s5, s8                                        // 000000000B28: 8C050805
	s_wait_alu depctr_sa_sdst(0)                               // 000000000B2C: BF88FF9E
	v_cndmask_b32_e64 v131, 0, 1, s5                           // 000000000B30: D5010083 00150280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000000B38: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v131                           // 000000000B3C: 7C9B0680
	s_cbranch_vccz 3                                           // 000000000B40: BFA30003 <nearest+0xb50>
	s_mov_b32 s6, 1.0                                          // 000000000B44: BE8600F2
	s_mov_b32 s15, 4.0                                         // 000000000B48: BE8F00F6
	s_branch 20                                                // 000000000B4C: BFA00014 <nearest+0xba0>
	v_cndmask_b32_e64 v131, v90, -v8, s3                       // 000000000B50: D5010083 400E115A
	v_cndmask_b32_e64 v132, v89, -v9, s4                       // 000000000B58: D5010084 40121359
	v_cndmask_b32_e64 v133, v88, v129, s3                      // 000000000B60: D5010085 000F0358
	v_cndmask_b32_e64 v134, v87, v130, s4                      // 000000000B68: D5010086 00130557
	v_cndmask_b32_e64 v84, v84, v90, s3                        // 000000000B70: D5010054 000EB554
	v_cndmask_b32_e64 v85, v85, v89, s4                        // 000000000B78: D5010055 0012B355
	v_cndmask_b32_e64 v82, v82, v88, s3                        // 000000000B80: D5010052 000EB152
	v_cndmask_b32_e64 v83, v83, v87, s4                        // 000000000B88: D5010053 0012AF53
	v_dual_mov_b32 v89, v132 :: v_dual_mov_b32 v90, v131       // 000000000B90: CA100184 595A0183
	v_dual_mov_b32 v87, v134 :: v_dual_mov_b32 v88, v133       // 000000000B98: CA100186 57580185
	s_wait_alu depctr_sa_sdst(0)                               // 000000000BA0: BF88FF9E
	s_cmp_lt_f32 s6, 0.5                                       // 000000000BA4: BF41F006
	v_cmp_ge_f32_e64 s3, v119, -v10                            // 000000000BA8: D4160003 42021577
	v_cmp_ge_f32_e64 s5, v119, -v11                            // 000000000BB0: D4160005 42021777
	v_or_b32_e32 v132, s7, v107                                // 000000000BB8: 3908D607
	s_cselect_b32 s4, -1, 0                                    // 000000000BBC: 980480C1
	v_or_b32_e32 v133, s7, v108                                // 000000000BC0: 390AD807
	s_wait_alu depctr_sa_sdst(0)                               // 000000000BC4: BF88FF9E
	s_and_b32 s8, s2, s4                                       // 000000000BC8: 8B080402
	s_and_b32 s4, s3, s8                                       // 000000000BCC: 8B040803
	s_and_b32 s8, s5, s8                                       // 000000000BD0: 8B080805
	s_wait_alu depctr_sa_sdst(0)                               // 000000000BD4: BF88FF9E
	s_or_b32 s3, s4, s8                                        // 000000000BD8: 8C030804
	s_wait_alu depctr_sa_sdst(0)                               // 000000000BDC: BF88FF9E
	v_cndmask_b32_e64 v131, 0, 1, s3                           // 000000000BE0: D5010083 000D0280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000000BE8: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v131                           // 000000000BEC: 7C9B0680
	v_cvt_f32_u32_e32 v131, v132                               // 000000000BF0: 7F060D84
	v_cvt_f32_u32_e32 v132, v133                               // 000000000BF4: 7F080D85
	s_cbranch_vccz 50                                          // 000000000BF8: BFA30032 <nearest+0xcc4>
	v_cmp_gt_f32_e64 s20, v119, -v10                           // 000000000BFC: D4140014 42021577
	s_delay_alu instid0(VALU_DEP_3)                            // 000000000C04: BF870003
	v_cmp_gt_f32_e32 vcc_lo, v120, v131                        // 000000000C08: 7C290778
	v_cmp_gt_f32_e64 s21, v119, -v11                           // 000000000C0C: D4140015 42021777
	v_cmp_gt_f32_e64 s3, v120, v132                            // 000000000C14: D4140003 02030978
	v_cmp_lt_f32_e64 s5, -1.0, v82                             // 000000000C1C: D4110005 0202A4F3
	s_or_b32 s20, vcc_lo, s20                                  // 000000000C24: 8C14146A
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v83                         // 000000000C28: 7C22A6F3
	s_or_b32 s21, s3, s21                                      // 000000000C2C: 8C151503
	s_and_b32 s3, s20, s4                                      // 000000000C30: 8B030414
	s_and_b32 s4, s21, s8                                      // 000000000C34: 8B040815
	s_wait_alu depctr_sa_sdst(0)                               // 000000000C38: BF88FF9E
	s_and_b32 s5, s5, s3                                       // 000000000C3C: 8B050305
	s_and_b32 s8, s4, vcc_lo                                   // 000000000C40: 8B086A04
	s_wait_alu depctr_sa_sdst(0)                               // 000000000C44: BF88FF9E
	s_or_b32 s5, s5, s8                                        // 000000000C48: 8C050805
	s_wait_alu depctr_sa_sdst(0)                               // 000000000C4C: BF88FF9E
	v_cndmask_b32_e64 v133, 0, 1, s5                           // 000000000C50: D5010085 00150280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000000C58: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v133                           // 000000000C5C: 7C9B0A80
	s_cbranch_vccz 4                                           // 000000000C60: BFA30004 <nearest+0xc74>
	s_mov_b32 s6, 1.0                                          // 000000000C64: BE8600F2
	s_mov_b32 s15, 0x40a00000                                  // 000000000C68: BE8F00FF 40A00000
	s_branch 20                                                // 000000000C70: BFA00014 <nearest+0xcc4>
	v_cndmask_b32_e64 v133, v90, -v10, s3                      // 000000000C74: D5010085 400E155A
	v_cndmask_b32_e64 v134, v89, -v11, s4                      // 000000000C7C: D5010086 40121759
	v_cndmask_b32_e64 v135, v88, v131, s3                      // 000000000C84: D5010087 000F0758
	v_cndmask_b32_e64 v136, v87, v132, s4                      // 000000000C8C: D5010088 00130957
	v_cndmask_b32_e64 v84, v84, v90, s3                        // 000000000C94: D5010054 000EB554
	v_cndmask_b32_e64 v85, v85, v89, s4                        // 000000000C9C: D5010055 0012B355
	v_cndmask_b32_e64 v82, v82, v88, s3                        // 000000000CA4: D5010052 000EB152
	v_cndmask_b32_e64 v83, v83, v87, s4                        // 000000000CAC: D5010053 0012AF53
	v_dual_mov_b32 v89, v134 :: v_dual_mov_b32 v90, v133       // 000000000CB4: CA100186 595A0185
	v_dual_mov_b32 v87, v136 :: v_dual_mov_b32 v88, v135       // 000000000CBC: CA100188 57580187
	s_wait_alu depctr_sa_sdst(0)                               // 000000000CC4: BF88FF9E
	s_cmp_lt_f32 s6, 0.5                                       // 000000000CC8: BF41F006
	v_cmp_ge_f32_e64 s3, v119, -v12                            // 000000000CCC: D4160003 42021977
	v_cmp_ge_f32_e64 s5, v119, -v13                            // 000000000CD4: D4160005 42021B77
	v_or_b32_e32 v134, s7, v113                                // 000000000CDC: 390CE207
	s_cselect_b32 s4, -1, 0                                    // 000000000CE0: 980480C1
	v_or_b32_e32 v135, s7, v116                                // 000000000CE4: 390EE807
	s_wait_alu depctr_sa_sdst(0)                               // 000000000CE8: BF88FF9E
	s_and_b32 s8, s2, s4                                       // 000000000CEC: 8B080402
	s_and_b32 s4, s3, s8                                       // 000000000CF0: 8B040803
	s_and_b32 s8, s5, s8                                       // 000000000CF4: 8B080805
	s_wait_alu depctr_sa_sdst(0)                               // 000000000CF8: BF88FF9E
	s_or_b32 s3, s4, s8                                        // 000000000CFC: 8C030804
	s_wait_alu depctr_sa_sdst(0)                               // 000000000D00: BF88FF9E
	v_cndmask_b32_e64 v133, 0, 1, s3                           // 000000000D04: D5010085 000D0280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000000D0C: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v133                           // 000000000D10: 7C9B0A80
	v_cvt_f32_u32_e32 v133, v134                               // 000000000D14: 7F0A0D86
	v_cvt_f32_u32_e32 v134, v135                               // 000000000D18: 7F0C0D87
	s_cbranch_vccz 50                                          // 000000000D1C: BFA30032 <nearest+0xde8>
	v_cmp_gt_f32_e64 s20, v119, -v12                           // 000000000D20: D4140014 42021977
	s_delay_alu instid0(VALU_DEP_3)                            // 000000000D28: BF870003
	v_cmp_gt_f32_e32 vcc_lo, v120, v133                        // 000000000D2C: 7C290B78
	v_cmp_gt_f32_e64 s21, v119, -v13                           // 000000000D30: D4140015 42021B77
	v_cmp_gt_f32_e64 s3, v120, v134                            // 000000000D38: D4140003 02030D78
	v_cmp_lt_f32_e64 s5, -1.0, v82                             // 000000000D40: D4110005 0202A4F3
	s_or_b32 s20, vcc_lo, s20                                  // 000000000D48: 8C14146A
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v83                         // 000000000D4C: 7C22A6F3
	s_or_b32 s21, s3, s21                                      // 000000000D50: 8C151503
	s_and_b32 s3, s20, s4                                      // 000000000D54: 8B030414
	s_and_b32 s4, s21, s8                                      // 000000000D58: 8B040815
	s_wait_alu depctr_sa_sdst(0)                               // 000000000D5C: BF88FF9E
	s_and_b32 s5, s5, s3                                       // 000000000D60: 8B050305
	s_and_b32 s8, s4, vcc_lo                                   // 000000000D64: 8B086A04
	s_wait_alu depctr_sa_sdst(0)                               // 000000000D68: BF88FF9E
	s_or_b32 s5, s5, s8                                        // 000000000D6C: 8C050805
	s_wait_alu depctr_sa_sdst(0)                               // 000000000D70: BF88FF9E
	v_cndmask_b32_e64 v135, 0, 1, s5                           // 000000000D74: D5010087 00150280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000000D7C: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v135                           // 000000000D80: 7C9B0E80
	s_cbranch_vccz 4                                           // 000000000D84: BFA30004 <nearest+0xd98>
	s_mov_b32 s6, 1.0                                          // 000000000D88: BE8600F2
	s_mov_b32 s15, 0x40c00000                                  // 000000000D8C: BE8F00FF 40C00000
	s_branch 20                                                // 000000000D94: BFA00014 <nearest+0xde8>
	v_cndmask_b32_e64 v135, v90, -v12, s3                      // 000000000D98: D5010087 400E195A
	v_cndmask_b32_e64 v136, v89, -v13, s4                      // 000000000DA0: D5010088 40121B59
	v_cndmask_b32_e64 v137, v88, v133, s3                      // 000000000DA8: D5010089 000F0B58
	v_cndmask_b32_e64 v138, v87, v134, s4                      // 000000000DB0: D501008A 00130D57
	v_cndmask_b32_e64 v84, v84, v90, s3                        // 000000000DB8: D5010054 000EB554
	v_cndmask_b32_e64 v85, v85, v89, s4                        // 000000000DC0: D5010055 0012B355
	v_cndmask_b32_e64 v82, v82, v88, s3                        // 000000000DC8: D5010052 000EB152
	v_cndmask_b32_e64 v83, v83, v87, s4                        // 000000000DD0: D5010053 0012AF53
	v_dual_mov_b32 v89, v136 :: v_dual_mov_b32 v90, v135       // 000000000DD8: CA100188 595A0187
	v_dual_mov_b32 v87, v138 :: v_dual_mov_b32 v88, v137       // 000000000DE0: CA10018A 57580189
	s_wait_alu depctr_sa_sdst(0)                               // 000000000DE8: BF88FF9E
	s_cmp_lt_f32 s6, 0.5                                       // 000000000DEC: BF41F006
	v_cmp_ge_f32_e64 s3, v119, -v14                            // 000000000DF0: D4160003 42021D77
	v_cmp_ge_f32_e64 s4, v119, -v15                            // 000000000DF8: D4160004 42021F77
	v_or_b32_e32 v136, s7, v117                                // 000000000E00: 3910EA07
	s_cselect_b32 s5, -1, 0                                    // 000000000E04: 980580C1
	v_or_b32_e32 v137, s7, v118                                // 000000000E08: 3912EC07
	s_wait_alu depctr_sa_sdst(0)                               // 000000000E0C: BF88FF9E
	s_and_b32 s2, s2, s5                                       // 000000000E10: 8B020502
	s_wait_alu depctr_sa_sdst(0)                               // 000000000E14: BF88FF9E
	s_and_b32 s3, s3, s2                                       // 000000000E18: 8B030203
	s_and_b32 s5, s4, s2                                       // 000000000E1C: 8B050204
	s_wait_alu depctr_sa_sdst(0)                               // 000000000E20: BF88FF9E
	s_or_b32 s2, s3, s5                                        // 000000000E24: 8C020503
	s_wait_alu depctr_sa_sdst(0)                               // 000000000E28: BF88FF9E
	v_cndmask_b32_e64 v135, 0, 1, s2                           // 000000000E2C: D5010087 00090280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000000E34: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v135                           // 000000000E38: 7C9B0E80
	v_cvt_f32_u32_e32 v135, v136                               // 000000000E3C: 7F0E0D88
	v_cvt_f32_u32_e32 v136, v137                               // 000000000E40: 7F100D89
	s_cbranch_vccz 30                                          // 000000000E44: BFA3001E <nearest+0xec0>
	v_cmp_gt_f32_e64 s7, v119, -v14                            // 000000000E48: D4140007 42021D77
	s_delay_alu instid0(VALU_DEP_3)                            // 000000000E50: BF870003
	v_cmp_gt_f32_e32 vcc_lo, v120, v135                        // 000000000E54: 7C290F78
	v_cmp_gt_f32_e64 s8, v119, -v15                            // 000000000E58: D4140008 42021F77
	v_cmp_gt_f32_e64 s2, v120, v136                            // 000000000E60: D4140002 02031178
	v_cmp_lt_f32_e64 s4, -1.0, v82                             // 000000000E68: D4110004 0202A4F3
	s_or_b32 s7, vcc_lo, s7                                    // 000000000E70: 8C07076A
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v83                         // 000000000E74: 7C22A6F3
	s_or_b32 s8, s2, s8                                        // 000000000E78: 8C080802
	s_wait_alu depctr_sa_sdst(0)                               // 000000000E7C: BF88FF9E
	s_and_b32 s2, s7, s3                                       // 000000000E80: 8B020307
	s_and_b32 s3, s8, s5                                       // 000000000E84: 8B030508
	s_wait_alu depctr_sa_sdst(0)                               // 000000000E88: BF88FF9E
	s_and_b32 s4, s4, s2                                       // 000000000E8C: 8B040204
	s_and_b32 s5, s3, vcc_lo                                   // 000000000E90: 8B056A03
	s_wait_alu depctr_sa_sdst(0)                               // 000000000E94: BF88FF9E
	s_or_b32 s4, s4, s5                                        // 000000000E98: 8C040504
	s_wait_alu depctr_sa_sdst(0)                               // 000000000E9C: BF88FF9E
	v_cndmask_b32_e64 v137, 0, 1, s4                           // 000000000EA0: D5010089 00110280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000000EA8: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v137                           // 000000000EAC: 7C9B1280
	s_cbranch_vccz 2256                                        // 000000000EB0: BFA308D0 <nearest+0x31f4>
	s_mov_b32 s6, 1.0                                          // 000000000EB4: BE8600F2
	s_mov_b32 s15, 0x40e00000                                  // 000000000EB8: BE8F00FF 40E00000
	s_wait_alu depctr_sa_sdst(0)                               // 000000000EC0: BF88FF9E
	s_cmp_ngt_f32 s6, 0.5                                      // 000000000EC4: BF4BF006
	s_cbranch_scc1 64786                                       // 000000000EC8: BFA2FD12 <nearest+0x314>
	v_cmp_eq_f32_e32 vcc_lo, v89, v90                          // 000000000ECC: 7C24B559
	v_cmp_lt_f32_e64 s2, v87, v88                              // 000000000ED0: D4110002 0202B157
	v_cmp_lt_f32_e64 s4, v89, v90                              // 000000000ED8: D4110004 0202B559
	v_cmp_lt_f32_e64 s3, v88, v87                              // 000000000EE0: D4110003 0202AF58
	v_cmp_lt_f32_e64 s5, v90, v89                              // 000000000EE8: D4110005 0202B35A
	v_cmp_lt_f32_e64 s6, v84, v85                              // 000000000EF0: D4110006 0202AB54
	s_and_b32 s2, s2, vcc_lo                                   // 000000000EF8: 8B026A02
	s_wait_alu depctr_sa_sdst(0)                               // 000000000EFC: BF88FF9E
	s_or_b32 s2, s4, s2                                        // 000000000F00: 8C020204
	s_and_b32 s3, s3, vcc_lo                                   // 000000000F04: 8B036A03
	s_wait_alu depctr_sa_sdst(0)                               // 000000000F08: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000000F0C: 8D6A0002
	s_or_b32 s3, s5, s3                                        // 000000000F10: 8C030305
	s_wait_alu depctr_sa_sdst(0)                               // 000000000F14: BF88FF9E
	v_cndmask_b32_e32 v119, v90, v89, vcc_lo                   // 000000000F18: 02EEB35A
	s_xor_b32 s2, s3, s9                                       // 000000000F1C: 8D020903
	v_cmp_lt_f32_e64 s3, v83, v82                              // 000000000F20: D4110003 0202A553
	s_wait_alu depctr_sa_sdst(0)                               // 000000000F28: BF88FF9E
	v_cndmask_b32_e64 v89, v89, v90, s2                        // 000000000F2C: D5010059 000AB559
	v_cndmask_b32_e32 v90, v88, v87, vcc_lo                    // 000000000F34: 02B4AF58
	v_cndmask_b32_e64 v87, v87, v88, s2                        // 000000000F38: D5010057 000AB157
	v_mov_b32_e32 v120, v119                                   // 000000000F40: 7EF00377
	v_cmp_eq_f32_e32 vcc_lo, v85, v84                          // 000000000F44: 7C24A955
	v_cmp_lt_f32_e64 s4, v82, v83                              // 000000000F48: D4110004 0202A752
	v_cmp_lt_f32_e64 s5, v85, v84                              // 000000000F50: D4110005 0202A955
	v_mov_b32_e32 v139, v87                                    // 000000000F58: 7F160357
	v_permlanex16_b32 v120, v120, s10, 0xfedcba98              // 000000000F5C: D65C0078 03FC1578 FEDCBA98
	v_mov_b32_e32 v138, v90                                    // 000000000F68: 7F14035A
	s_and_b32 s2, s3, vcc_lo                                   // 000000000F6C: 8B026A03
	s_and_b32 s3, s4, vcc_lo                                   // 000000000F70: 8B036A04
	s_wait_alu depctr_sa_sdst(0)                               // 000000000F74: BF88FF9E
	s_or_b32 s2, s5, s2                                        // 000000000F78: 8C020205
	s_or_b32 s3, s6, s3                                        // 000000000F7C: 8C030306
	v_permlanex16_b32 v138, v138, s10, 0xfedcba98              // 000000000F80: D65C008A 03FC158A FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 000000000F8C: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000000F90: 8D6A0002
	s_xor_b32 s2, s3, s9                                       // 000000000F94: 8D020903
	s_wait_alu depctr_sa_sdst(0)                               // 000000000F98: BF88FF9E
	v_dual_cndmask_b32 v88, v84, v85 :: v_dual_mov_b32 v137, v89// 000000000F9C: CA50AB54 58880159
	v_cndmask_b32_e64 v84, v85, v84, s2                        // 000000000FA4: D5010054 000AA955
	v_cndmask_b32_e32 v85, v82, v83, vcc_lo                    // 000000000FAC: 02AAA752
	v_cmp_eq_f32_e32 vcc_lo, v119, v120                        // 000000000FB0: 7C24F177
	v_cmp_gt_f32_e64 s4, v90, v138                             // 000000000FB4: D4140004 0203155A
	v_cmp_gt_f32_e64 s6, v119, v120                            // 000000000FBC: D4140006 0202F177
	v_cndmask_b32_e64 v82, v83, v82, s2                        // 000000000FC4: D5010052 000AA553
	v_permlanex16_b32 v137, v137, s10, 0xfedcba98              // 000000000FCC: D65C0089 03FC1589 FEDCBA98
	v_permlanex16_b32 v139, v139, s10, 0xfedcba98              // 000000000FD8: D65C008B 03FC158B FEDCBA98
	s_and_b32 s2, vcc_lo, s4                                   // 000000000FE4: 8B02046A
	s_mov_b32 s20, 4                                           // 000000000FE8: BE940084
	s_wait_alu depctr_sa_sdst(0)                               // 000000000FEC: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 000000000FF0: 8C020206
	v_cmp_eq_f32_e64 s3, v89, v137                             // 000000000FF4: D4120003 02031359
	s_wait_alu depctr_sa_sdst(0)                               // 000000000FFC: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000001000: 8D6A0002
	v_cmp_gt_f32_e64 s5, v87, v139                             // 000000001004: D4140005 02031757
	s_wait_alu depctr_sa_sdst(0)                               // 00000000100C: BF88FF9E
	v_dual_cndmask_b32 v83, v119, v120 :: v_dual_mov_b32 v120, v84// 000000001010: CA50F177 53780154
	v_dual_mov_b32 v119, v88 :: v_dual_mov_b32 v140, v85       // 000000001018: CA100158 778C0155
	v_cmp_gt_f32_e64 s7, v89, v137                             // 000000001020: D4140007 02031359
	s_and_b32 s3, s3, s5                                       // 000000001028: 8B030503
	v_permlanex16_b32 v120, v120, s10, 0xfedcba98              // 00000000102C: D65C0078 03FC1578 FEDCBA98
	v_permlanex16_b32 v119, v119, s10, 0xfedcba98              // 000000001038: D65C0077 03FC1577 FEDCBA98
	v_permlanex16_b32 v140, v140, s10, 0xfedcba98              // 000000001044: D65C008C 03FC158C FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 000000001050: BF88FF9E
	s_or_b32 s3, s7, s3                                        // 000000001054: 8C030307
	v_cndmask_b32_e32 v90, v90, v138, vcc_lo                   // 000000001058: 02B5155A
	s_wait_alu depctr_sa_sdst(0)                               // 00000000105C: BF88FF9E
	s_xor_b32 s2, s3, s0                                       // 000000001060: 8D020003
	v_cmp_neq_f32_e32 vcc_lo, v88, v119                        // 000000001064: 7C3AEF58
	v_cmp_ngt_f32_e64 s3, v85, v140                            // 000000001068: D41B0003 02031955
	v_cmp_ngt_f32_e64 s6, v88, v119                            // 000000001070: D41B0006 0202EF58
	s_wait_alu depctr_sa_sdst(0)                               // 000000001078: BF88FF9E
	v_cndmask_b32_e64 v89, v89, v137, s2                       // 00000000107C: D5010059 000B1359
	v_cndmask_b32_e64 v87, v87, v139, s2                       // 000000001084: D5010057 000B1757
	v_cmp_neq_f32_e64 s4, v84, v120                            // 00000000108C: D41D0004 0202F154
	s_or_b32 s2, vcc_lo, s3                                    // 000000001094: 8C02036A
	v_cmp_ngt_f32_e64 s7, v84, v120                            // 000000001098: D41B0007 0202F154
	s_wait_alu depctr_sa_sdst(0)                               // 0000000010A0: BF88FF9E
	s_and_b32 s2, s6, s2                                       // 0000000010A4: 8B020206
	s_wait_alu depctr_sa_sdst(0)                               // 0000000010A8: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 0000000010AC: 8D6A0002
	s_wait_alu depctr_sa_sdst(0)                               // 0000000010B0: BF88FF9E
	v_dual_mov_b32 v141, v82 :: v_dual_cndmask_b32 v88, v88, v119// 0000000010B4: CA120152 8D58EF58
	v_cndmask_b32_e32 v85, v85, v140, vcc_lo                   // 0000000010BC: 02AB1955
	v_cmp_eq_f32_e32 vcc_lo, v89, v83                          // 0000000010C0: 7C24A759
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000010C4: BF870093
	v_permlanex16_b32 v141, v141, s10, 0xfedcba98              // 0000000010C8: D65C008D 03FC158D FEDCBA98
	v_cmp_ngt_f32_e64 s5, v82, v141                            // 0000000010D4: D41B0005 02031B52
	s_or_b32 s3, s4, s5                                        // 0000000010DC: 8C030504
	v_cmp_lt_f32_e64 s4, v90, v87                              // 0000000010E0: D4110004 0202AF5A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000010E8: BF88FF9E
	s_and_b32 s3, s7, s3                                       // 0000000010EC: 8B030307
	v_cmp_lt_f32_e64 s5, v89, v83                              // 0000000010F0: D4110005 0202A759
	s_wait_alu depctr_sa_sdst(0)                               // 0000000010F8: BF88FF9E
	s_xor_b32 s2, s3, s0                                       // 0000000010FC: 8D020003
	v_cmp_lt_f32_e64 s3, v87, v90                              // 000000001100: D4110003 0202B557
	s_wait_alu depctr_sa_sdst(0)                               // 000000001108: BF88FF9E
	v_cndmask_b32_e64 v84, v84, v120, s2                       // 00000000110C: D5010054 000AF154
	v_cndmask_b32_e64 v82, v82, v141, s2                       // 000000001114: D5010052 000B1B52
	v_cmp_lt_f32_e64 s2, v83, v89                              // 00000000111C: D4110002 0202B353
	s_and_b32 s4, vcc_lo, s4                                   // 000000001124: 8B04046A
	s_and_b32 s3, vcc_lo, s3                                   // 000000001128: 8B03036A
	s_wait_alu depctr_sa_sdst(0)                               // 00000000112C: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 000000001130: 8C6A0305
	s_or_b32 s2, s2, s4                                        // 000000001134: 8C020402
	s_wait_alu depctr_sa_sdst(0)                               // 000000001138: BF88FF9E
	v_cndmask_b32_e32 v119, v83, v89, vcc_lo                   // 00000000113C: 02EEB353
	v_cndmask_b32_e64 v83, v83, v89, s2                        // 000000001140: D5010053 000AB353
	v_cndmask_b32_e32 v89, v90, v87, vcc_lo                    // 000000001148: 02B2AF5A
	v_cmp_eq_f32_e32 vcc_lo, v84, v88                          // 00000000114C: 7C24B154
	v_cmp_lt_f32_e64 s3, v82, v85                              // 000000001150: D4110003 0202AB52
	v_cmp_lt_f32_e64 s4, v85, v82                              // 000000001158: D4110004 0202A555
	v_cmp_lt_f32_e64 s5, v84, v88                              // 000000001160: D4110005 0202B154
	v_cndmask_b32_e64 v87, v90, v87, s2                        // 000000001168: D5010057 000AAF5A
	v_cmp_lt_f32_e64 s2, v88, v84                              // 000000001170: D4110002 0202A958
	s_and_b32 s3, vcc_lo, s3                                   // 000000001178: 8B03036A
	s_and_b32 s4, vcc_lo, s4                                   // 00000000117C: 8B04046A
	s_wait_alu depctr_sa_sdst(0)                               // 000000001180: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 000000001184: 8C6A0305
	s_or_b32 s2, s2, s4                                        // 000000001188: 8C020402
	s_wait_alu depctr_sa_sdst(0)                               // 00000000118C: BF88FF9E
	v_cndmask_b32_e32 v90, v84, v88, vcc_lo                    // 000000001190: 02B4B154
	v_cndmask_b32_e64 v84, v84, v88, s2                        // 000000001194: D5010054 000AB154
	v_cndmask_b32_e32 v88, v82, v85, vcc_lo                    // 00000000119C: 02B0AB52
	v_cndmask_b32_e64 v82, v82, v85, s2                        // 0000000011A0: D5010052 000AAB52
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 0000000011A8: BF870224
	v_cmp_eq_f32_e32 vcc_lo, v119, v90                         // 0000000011AC: 7C24B577
	v_cmp_lt_f32_e64 s5, v119, v90                             // 0000000011B0: D4110005 0202B577
	v_cmp_lt_f32_e64 s3, v89, v88                              // 0000000011B8: D4110003 0202B159
	v_cmp_eq_f32_e64 s2, v83, v84                              // 0000000011C0: D4120002 0202A953
	v_cmp_lt_f32_e64 s4, v87, v82                              // 0000000011C8: D4110004 0202A557
	v_cmp_lt_f32_e64 s6, v83, v84                              // 0000000011D0: D4110006 0202A953
	s_and_b32 s3, vcc_lo, s3                                   // 0000000011D8: 8B03036A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000011DC: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 0000000011E0: 8C6A0305
	s_and_b32 s2, s2, s4                                       // 0000000011E4: 8B020402
	s_wait_alu depctr_sa_sdst(0)                               // 0000000011E8: BF88FF9E
	v_cndmask_b32_e32 v85, v90, v119, vcc_lo                   // 0000000011EC: 02AAEF5A
	s_or_b32 s2, s6, s2                                        // 0000000011F0: 8C020206
	v_cndmask_b32_e32 v90, v119, v90, vcc_lo                   // 0000000011F4: 02B4B577
	s_wait_alu depctr_sa_sdst(0)                               // 0000000011F8: BF88FF9E
	v_cndmask_b32_e64 v120, v84, v83, s2                       // 0000000011FC: D5010078 000AA754
	v_cndmask_b32_e64 v83, v83, v84, s2                        // 000000001204: D5010053 000AA953
	v_mov_b32_e32 v119, v85                                    // 00000000120C: 7EEE0355
	v_dual_cndmask_b32 v137, v88, v89 :: v_dual_cndmask_b32 v84, v89, v88// 000000001210: CA52B358 8954B159
	v_cndmask_b32_e64 v138, v82, v87, s2                       // 000000001218: D501008A 000AAF52
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000001220: BF870194
	v_dual_mov_b32 v139, v120 :: v_dual_mov_b32 v88, v83       // 000000001224: CA100178 8B580153
	v_dual_mov_b32 v140, v137 :: v_dual_mov_b32 v89, v84       // 00000000122C: CA100189 8C580154
	v_permlanex16_b32 v119, v119, s10, 0xfedcba98              // 000000001234: D65C0077 03FC1577 FEDCBA98
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 000000001240: BF870223
	v_permlanex16_b32 v139, v139, s10, 0xfedcba98              // 000000001244: D65C008B 03FC158B FEDCBA98
	v_cndmask_b32_e64 v82, v87, v82, s2                        // 000000001250: D5010052 000AA557
	v_permlanex16_b32 v140, v140, s10, 0xfedcba98              // 000000001258: D65C008C 03FC158C FEDCBA98
	v_permlanex16_b32 v89, v89, s10, 0xfedcba98                // 000000001264: D65C0059 03FC1559 FEDCBA98
	v_mov_b32_e32 v141, v138                                   // 000000001270: 7F1A038A
	v_cmp_eq_f32_e32 vcc_lo, v85, v119                         // 000000001274: 7C24EF55
	v_cmp_eq_f32_e64 s3, v120, v139                            // 000000001278: D4120003 02031778
	v_cmp_gt_f32_e64 s4, v137, v140                            // 000000001280: D4140004 02031989
	v_cmp_gt_f32_e64 s6, v85, v119                             // 000000001288: D4140006 0202EF55
	v_permlanex16_b32 v141, v141, s10, 0xfedcba98              // 000000001290: D65C008D 03FC158D FEDCBA98
	v_cmp_gt_f32_e64 s7, v120, v139                            // 00000000129C: D4140007 02031778
	v_permlanex16_b32 v88, v88, s10, 0xfedcba98                // 0000000012A4: D65C0058 03FC1558 FEDCBA98
	s_and_b32 s2, vcc_lo, s4                                   // 0000000012B0: 8B02046A
	v_cmp_gt_f32_e64 s4, v84, v89                              // 0000000012B4: D4140004 0202B354
	v_cmp_gt_f32_e64 s5, v138, v141                            // 0000000012BC: D4140005 02031B8A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000012C4: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 0000000012C8: 8C020206
	s_wait_alu depctr_sa_sdst(0)                               // 0000000012CC: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 0000000012D0: 8D6A0002
	s_and_b32 s3, s3, s5                                       // 0000000012D4: 8B030503
	s_wait_alu depctr_sa_sdst(0)                               // 0000000012D8: BF88FF9E
	v_cndmask_b32_e32 v85, v85, v119, vcc_lo                   // 0000000012DC: 02AAEF55
	s_or_b32 s3, s7, s3                                        // 0000000012E0: 8C030307
	v_mov_b32_e32 v119, v82                                    // 0000000012E4: 7EEE0352
	s_wait_alu depctr_sa_sdst(0)                               // 0000000012E8: BF88FF9E
	s_xor_b32 s2, s3, s0                                       // 0000000012EC: 8D020003
	v_cndmask_b32_e32 v137, v137, v140, vcc_lo                 // 0000000012F0: 03131989
	s_wait_alu depctr_sa_sdst(0)                               // 0000000012F4: BF88FF9E
	v_cndmask_b32_e64 v120, v120, v139, s2                     // 0000000012F8: D5010078 000B1778
	v_mov_b32_e32 v87, v90                                     // 000000001300: 7EAE035A
	v_permlanex16_b32 v119, v119, s10, 0xfedcba98              // 000000001304: D65C0077 03FC1577 FEDCBA98
	v_cmp_eq_f32_e64 s3, v83, v88                              // 000000001310: D4120003 0202B153
	v_cmp_gt_f32_e64 s7, v83, v88                              // 000000001318: D4140007 0202B153
	v_cndmask_b32_e64 v138, v138, v141, s2                     // 000000001320: D501008A 000B1B8A
	v_permlanex16_b32 v87, v87, s10, 0xfedcba98                // 000000001328: D65C0057 03FC1557 FEDCBA98
	v_cmp_gt_f32_e64 s5, v82, v119                             // 000000001334: D4140005 0202EF52
	s_delay_alu instid0(VALU_DEP_2)                            // 00000000133C: BF870002
	v_cmp_eq_f32_e32 vcc_lo, v90, v87                          // 000000001340: 7C24AF5A
	v_cmp_gt_f32_e64 s6, v90, v87                              // 000000001344: D4140006 0202AF5A
	s_and_b32 s3, s3, s5                                       // 00000000134C: 8B030503
	v_cmp_lt_f32_e64 s5, v120, v85                             // 000000001350: D4110005 0202AB78
	s_wait_alu depctr_sa_sdst(0)                               // 000000001358: BF88FF9E
	s_or_b32 s3, s7, s3                                        // 00000000135C: 8C030307
	s_and_b32 s2, vcc_lo, s4                                   // 000000001360: 8B02046A
	v_cmp_lt_f32_e64 s4, v137, v138                            // 000000001364: D4110004 02031589
	s_wait_alu depctr_sa_sdst(0)                               // 00000000136C: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 000000001370: 8C020206
	s_wait_alu depctr_sa_sdst(0)                               // 000000001374: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000001378: 8D6A0002
	s_xor_b32 s2, s3, s0                                       // 00000000137C: 8D020003
	s_wait_alu depctr_sa_sdst(0)                               // 000000001380: BF88FF9E
	v_dual_cndmask_b32 v87, v90, v87 :: v_dual_cndmask_b32 v84, v84, v89// 000000001384: CA52AF5A 5754B354
	v_cmp_eq_f32_e32 vcc_lo, v120, v85                         // 00000000138C: 7C24AB78
	v_cmp_lt_f32_e64 s3, v138, v137                            // 000000001390: D4110003 0203138A
	v_cndmask_b32_e64 v83, v83, v88, s2                        // 000000001398: D5010053 000AB153
	v_cndmask_b32_e64 v82, v82, v119, s2                       // 0000000013A0: D5010052 000AEF52
	v_cmp_lt_f32_e64 s2, v85, v120                             // 0000000013A8: D4110002 0202F155
	s_and_b32 s4, vcc_lo, s4                                   // 0000000013B0: 8B04046A
	s_and_b32 s3, vcc_lo, s3                                   // 0000000013B4: 8B03036A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000013B8: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 0000000013BC: 8C6A0305
	s_or_b32 s2, s2, s4                                        // 0000000013C0: 8C020402
	s_wait_alu depctr_sa_sdst(0)                               // 0000000013C4: BF88FF9E
	v_cndmask_b32_e32 v88, v85, v120, vcc_lo                   // 0000000013C8: 02B0F155
	v_cndmask_b32_e32 v89, v137, v138, vcc_lo                  // 0000000013CC: 02B31589
	v_cmp_eq_f32_e32 vcc_lo, v83, v87                          // 0000000013D0: 7C24AF53
	v_cmp_lt_f32_e64 s3, v82, v84                              // 0000000013D4: D4110003 0202A952
	v_cmp_lt_f32_e64 s4, v84, v82                              // 0000000013DC: D4110004 0202A554
	v_cndmask_b32_e64 v85, v85, v120, s2                       // 0000000013E4: D5010055 000AF155
	v_cmp_lt_f32_e64 s5, v83, v87                              // 0000000013EC: D4110005 0202AF53
	v_cndmask_b32_e64 v90, v137, v138, s2                      // 0000000013F4: D501005A 000B1589
	v_cmp_lt_f32_e64 s2, v87, v83                              // 0000000013FC: D4110002 0202A757
	s_and_b32 s3, vcc_lo, s3                                   // 000000001404: 8B03036A
	s_and_b32 s4, vcc_lo, s4                                   // 000000001408: 8B04046A
	s_wait_alu depctr_sa_sdst(0)                               // 00000000140C: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 000000001410: 8C6A0305
	v_permlanex16_b32 v85, v85, s10, 0xfedcba98                // 000000001414: D65C0055 03FC1555 FEDCBA98
	s_or_b32 s2, s2, s4                                        // 000000001420: 8C020402
	s_wait_alu depctr_sa_sdst(0)                               // 000000001424: BF88FF9E
	v_cndmask_b32_e32 v119, v87, v83, vcc_lo                   // 000000001428: 02EEA757
	v_cndmask_b32_e64 v83, v87, v83, s2                        // 00000000142C: D5010053 000AA757
	v_permlanex16_b32 v88, v88, s10, 0xfedcba98                // 000000001434: D65C0058 03FC1558 FEDCBA98
	v_permlanex16_b32 v90, v90, s10, 0xfedcba98                // 000000001440: D65C005A 03FC155A FEDCBA98
	v_permlanex16_b32 v89, v89, s10, 0xfedcba98                // 00000000144C: D65C0059 03FC1559 FEDCBA98
	v_cndmask_b32_e32 v87, v84, v82, vcc_lo                    // 000000001458: 02AEA554
	v_cndmask_b32_e64 v82, v84, v82, s2                        // 00000000145C: D5010052 000AA554
	v_cmp_eq_f32_e32 vcc_lo, v81, v85                          // 000000001464: 7C24AB51
	v_cmp_eq_f32_e64 s3, v75, v88                              // 000000001468: D4120003 0202B14B
	v_cmp_lt_f32_e64 s4, v80, v90                              // 000000001470: D4110004 0202B550
	v_cmp_lt_f32_e64 s5, v74, v89                              // 000000001478: D4110005 0202B34A
	v_permlanex16_b32 v83, v83, s10, 0xfedcba98                // 000000001480: D65C0053 03FC1553 FEDCBA98
	v_permlanex16_b32 v82, v82, s10, 0xfedcba98                // 00000000148C: D65C0052 03FC1552 FEDCBA98
	v_cmp_lt_f32_e64 s2, v81, v85                              // 000000001498: D4110002 0202AB51
	v_permlanex16_b32 v119, v119, s10, 0xfedcba98              // 0000000014A0: D65C0077 03FC1577 FEDCBA98
	v_permlanex16_b32 v87, v87, s10, 0xfedcba98                // 0000000014AC: D65C0057 03FC1557 FEDCBA98
	s_and_b32 s4, vcc_lo, s4                                   // 0000000014B8: 8B04046A
	s_and_b32 s8, s3, s5                                       // 0000000014BC: 8B080503
	v_cmp_eq_f32_e64 s3, v76, v83                              // 0000000014C0: D4120003 0202A74C
	v_cmp_lt_f32_e64 s5, v77, v82                              // 0000000014C8: D4110005 0202A54D
	v_cmp_lt_f32_e64 s6, v75, v88                              // 0000000014D0: D4110006 0202B14B
	s_wait_alu depctr_sa_sdst(0)                               // 0000000014D8: BF88FF9E
	s_or_b32 vcc_lo, s2, s4                                    // 0000000014DC: 8C6A0402
	v_cmp_lt_f32_e64 s2, v76, v83                              // 0000000014E0: D4110002 0202A74C
	v_cmp_eq_f32_e64 s4, v78, v119                             // 0000000014E8: D4120004 0202EF4E
	v_cmp_lt_f32_e64 s7, v79, v87                              // 0000000014F0: D4110007 0202AF4F
	s_and_b32 s3, s3, s5                                       // 0000000014F8: 8B030503
	s_or_b32 s6, s6, s8                                        // 0000000014FC: 8C060806
	v_cmp_lt_f32_e64 s8, v78, v119                             // 000000001500: D4110008 0202EF4E
	s_wait_alu depctr_sa_sdst(0)                               // 000000001508: BF88FF9E
	s_or_b32 s2, s2, s3                                        // 00000000150C: 8C020302
	s_and_b32 s4, s4, s7                                       // 000000001510: 8B040704
	s_wait_alu depctr_sa_sdst(0)                               // 000000001514: BF88FF9E
	v_cndmask_b32_e64 v76, v83, v76, s2                        // 000000001518: D501004C 000A9953
	v_cndmask_b32_e64 v77, v82, v77, s2                        // 000000001520: D501004D 000A9B52
	s_or_b32 s3, s8, s4                                        // 000000001528: 8C030408
	v_cndmask_b32_e32 v81, v85, v81, vcc_lo                    // 00000000152C: 02A2A355
	s_wait_alu depctr_sa_sdst(0)                               // 000000001530: BF88FF9E
	v_cndmask_b32_e64 v78, v119, v78, s3                       // 000000001534: D501004E 000E9D77
	v_cndmask_b32_e64 v79, v87, v79, s3                        // 00000000153C: D501004F 000E9F57
	v_cndmask_b32_e32 v80, v90, v80, vcc_lo                    // 000000001544: 02A0A15A
	v_cmp_eq_f32_e32 vcc_lo, v68, v76                          // 000000001548: 7C249944
	v_cmp_lt_f32_e64 s3, v67, v77                              // 00000000154C: D4110003 02029B43
	v_cmp_lt_f32_e64 s5, v68, v76                              // 000000001554: D4110005 02029944
	v_cmp_eq_f32_e64 s2, v69, v78                              // 00000000155C: D4120002 02029D45
	v_cmp_lt_f32_e64 s4, v66, v79                              // 000000001564: D4110004 02029F42
	v_cndmask_b32_e64 v75, v88, v75, s6                        // 00000000156C: D501004B 001A9758
	s_and_b32 s3, vcc_lo, s3                                   // 000000001574: 8B03036A
	v_cndmask_b32_e64 v74, v89, v74, s6                        // 000000001578: D501004A 001A9559
	v_cmp_lt_f32_e64 s6, v69, v78                              // 000000001580: D4110006 02029D45
	s_wait_alu depctr_sa_sdst(0)                               // 000000001588: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 00000000158C: 8C6A0305
	s_and_b32 s2, s2, s4                                       // 000000001590: 8B020402
	s_wait_alu depctr_sa_sdst(0)                               // 000000001594: BF88FF9E
	v_cndmask_b32_e32 v82, v76, v68, vcc_lo                    // 000000001598: 02A4894C
	v_cndmask_b32_e32 v84, v77, v67, vcc_lo                    // 00000000159C: 02A8874D
	v_dual_cndmask_b32 v68, v68, v76 :: v_dual_cndmask_b32 v67, v67, v77// 0000000015A0: CA529944 44429B43
	v_cmp_eq_f32_e32 vcc_lo, v71, v81                          // 0000000015A8: 7C24A347
	v_cmp_lt_f32_e64 s4, v72, v80                              // 0000000015AC: D4110004 0202A148
	s_or_b32 s2, s6, s2                                        // 0000000015B4: 8C020206
	v_cmp_lt_f32_e64 s6, v71, v81                              // 0000000015B8: D4110006 0202A347
	s_wait_alu depctr_sa_sdst(0)                               // 0000000015C0: BF88FF9E
	v_cndmask_b32_e64 v83, v78, v69, s2                        // 0000000015C4: D5010053 000A8B4E
	v_cndmask_b32_e64 v69, v69, v78, s2                        // 0000000015CC: D5010045 000A9D45
	s_and_b32 s4, vcc_lo, s4                                   // 0000000015D4: 8B04046A
	v_cmp_eq_f32_e64 s3, v73, v75                              // 0000000015D8: D4120003 02029749
	s_wait_alu depctr_sa_sdst(0)                               // 0000000015E0: BF88FF9E
	s_or_b32 vcc_lo, s6, s4                                    // 0000000015E4: 8C6A0406
	v_cmp_lt_f32_e64 s5, v70, v74                              // 0000000015E8: D4110005 02029546
	s_wait_alu depctr_sa_sdst(0)                               // 0000000015F0: BF88FF9E
	v_cndmask_b32_e32 v76, v81, v71, vcc_lo                    // 0000000015F4: 02988F51
	v_cndmask_b32_e32 v78, v80, v72, vcc_lo                    // 0000000015F8: 029C9150
	v_cndmask_b32_e64 v85, v79, v66, s2                        // 0000000015FC: D5010055 000A854F
	v_cndmask_b32_e64 v66, v66, v79, s2                        // 000000001604: D5010042 000A9F42
	v_cmp_lt_f32_e64 s2, v73, v75                              // 00000000160C: D4110002 02029749
	s_and_b32 s3, s3, s5                                       // 000000001614: 8B030503
	v_dual_cndmask_b32 v71, v71, v81 :: v_dual_cndmask_b32 v72, v72, v80// 000000001618: CA52A347 4748A148
	v_cmp_eq_f32_e32 vcc_lo, v82, v76                          // 000000001620: 7C249952
	v_cmp_lt_f32_e64 s4, v84, v78                              // 000000001624: D4110004 02029D54
	s_wait_alu depctr_sa_sdst(0)                               // 00000000162C: BF88FF9E
	s_or_b32 s2, s2, s3                                        // 000000001630: 8C020302
	v_cmp_lt_f32_e64 s6, v82, v76                              // 000000001634: D4110006 02029952
	s_wait_alu depctr_sa_sdst(0)                               // 00000000163C: BF88FF9E
	v_cndmask_b32_e64 v77, v75, v73, s2                        // 000000001640: D501004D 000A934B
	v_cndmask_b32_e64 v79, v74, v70, s2                        // 000000001648: D501004F 000A8D4A
	s_and_b32 s4, vcc_lo, s4                                   // 000000001650: 8B04046A
	v_cndmask_b32_e64 v70, v70, v74, s2                        // 000000001654: D5010046 000A9546
	s_wait_alu depctr_sa_sdst(0)                               // 00000000165C: BF88FF9E
	s_or_b32 vcc_lo, s6, s4                                    // 000000001660: 8C6A0406
	v_cmp_eq_f32_e64 s3, v83, v77                              // 000000001664: D4120003 02029B53
	v_cmp_lt_f32_e64 s5, v85, v79                              // 00000000166C: D4110005 02029F55
	s_wait_alu depctr_sa_sdst(0)                               // 000000001674: BF88FF9E
	v_cndmask_b32_e32 v74, v76, v82, vcc_lo                    // 000000001678: 0294A54C
	v_cndmask_b32_e32 v80, v78, v84, vcc_lo                    // 00000000167C: 02A0A94E
	v_cndmask_b32_e32 v76, v82, v76, vcc_lo                    // 000000001680: 02989952
	v_cndmask_b32_e32 v78, v84, v78, vcc_lo                    // 000000001684: 029C9D54
	v_cmp_eq_f32_e32 vcc_lo, v68, v71                          // 000000001688: 7C248F44
	v_cmp_lt_f32_e64 s4, v67, v72                              // 00000000168C: D4110004 02029143
	v_cndmask_b32_e64 v73, v73, v75, s2                        // 000000001694: D5010049 000A9749
	v_cmp_lt_f32_e64 s2, v83, v77                              // 00000000169C: D4110002 02029B53
	v_cmp_lt_f32_e64 s6, v68, v71                              // 0000000016A4: D4110006 02028F44
	s_and_b32 s3, s3, s5                                       // 0000000016AC: 8B030503
	s_and_b32 s4, vcc_lo, s4                                   // 0000000016B0: 8B04046A
	v_cmp_lt_f32_e64 s5, v66, v70                              // 0000000016B4: D4110005 02028D42
	s_wait_alu depctr_sa_sdst(0)                               // 0000000016BC: BF88FF9E
	s_or_b32 s2, s2, s3                                        // 0000000016C0: 8C020302
	s_or_b32 vcc_lo, s6, s4                                    // 0000000016C4: 8C6A0406
	s_wait_alu depctr_sa_sdst(0)                               // 0000000016C8: BF88FF9E
	v_cndmask_b32_e64 v81, v79, v85, s2                        // 0000000016CC: D5010051 000AAB4F
	v_cndmask_b32_e32 v82, v71, v68, vcc_lo                    // 0000000016D4: 02A48947
	v_dual_cndmask_b32 v68, v68, v71 :: v_dual_mov_b32 v71, v74// 0000000016D8: CA508F44 4446014A
	v_mov_b32_e32 v88, v80                                     // 0000000016E0: 7EB00350
	v_cndmask_b32_e64 v75, v77, v83, s2                        // 0000000016E4: D501004B 000AA74D
	v_dual_cndmask_b32 v84, v72, v67 :: v_dual_cndmask_b32 v67, v67, v72// 0000000016EC: CA528748 54429143
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 0000000016F4: BF870214
	v_permlanex16_b32 v71, v71, s10, 0xfedcba98                // 0000000016F8: D65C0047 03FC1547 FEDCBA98
	v_permlanex16_b32 v88, v88, s10, 0xfedcba98                // 000000001704: D65C0058 03FC1558 FEDCBA98
	v_cmp_eq_f32_e64 s3, v69, v73                              // 000000001710: D4120003 02029345
	v_cndmask_b32_e64 v77, v83, v77, s2                        // 000000001718: D501004D 000A9B53
	v_cndmask_b32_e64 v79, v85, v79, s2                        // 000000001720: D501004F 000A9F55
	v_cmp_lt_f32_e64 s2, v69, v73                              // 000000001728: D4110002 02029345
	v_cmp_eq_f32_e32 vcc_lo, v74, v71                          // 000000001730: 7C248F4A
	v_cmp_gt_f32_e64 s4, v80, v88                              // 000000001734: D4140004 0202B150
	s_and_b32 s3, s3, s5                                       // 00000000173C: 8B030503
	v_cmp_gt_f32_e64 s6, v74, v71                              // 000000001740: D4140006 02028F4A
	s_wait_alu depctr_sa_sdst(0)                               // 000000001748: BF88FF9E
	s_or_b32 s2, s2, s3                                        // 00000000174C: 8C020302
	v_mov_b32_e32 v87, v75                                     // 000000001750: 7EAE034B
	s_wait_alu depctr_sa_sdst(0)                               // 000000001754: BF88FF9E
	v_cndmask_b32_e64 v83, v73, v69, s2                        // 000000001758: D5010053 000A8B49
	v_cndmask_b32_e64 v85, v70, v66, s2                        // 000000001760: D5010055 000A8546
	v_mov_b32_e32 v89, v81                                     // 000000001768: 7EB20351
	v_cndmask_b32_e64 v69, v69, v73, s2                        // 00000000176C: D5010045 000A9345
	v_cndmask_b32_e64 v66, v66, v70, s2                        // 000000001774: D5010042 000A8D42
	s_and_b32 s2, vcc_lo, s4                                   // 00000000177C: 8B02046A
	v_permlanex16_b32 v87, v87, s10, 0xfedcba98                // 000000001780: D65C0057 03FC1557 FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 00000000178C: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 000000001790: 8C020206
	v_permlanex16_b32 v89, v89, s10, 0xfedcba98                // 000000001794: D65C0059 03FC1559 FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 0000000017A0: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 0000000017A4: 8D6A0002
	v_dual_mov_b32 v72, v77 :: v_dual_mov_b32 v73, v78         // 0000000017A8: CA10014D 4848014E
	s_wait_alu depctr_sa_sdst(0)                               // 0000000017B0: BF88FF9E
	v_dual_cndmask_b32 v70, v74, v71 :: v_dual_mov_b32 v71, v76// 0000000017B4: CA508F4A 4646014C
	v_cmp_eq_f32_e64 s3, v75, v87                              // 0000000017BC: D4120003 0202AF4B
	v_cmp_gt_f32_e64 s5, v81, v89                              // 0000000017C4: D4140005 0202B351
	v_permlanex16_b32 v73, v73, s10, 0xfedcba98                // 0000000017CC: D65C0049 03FC1549 FEDCBA98
	v_cmp_gt_f32_e64 s7, v75, v87                              // 0000000017D8: D4140007 0202AF4B
	v_permlanex16_b32 v71, v71, s10, 0xfedcba98                // 0000000017E0: D65C0047 03FC1547 FEDCBA98
	v_cndmask_b32_e32 v80, v80, v88, vcc_lo                    // 0000000017EC: 02A0B150
	s_and_b32 s3, s3, s5                                       // 0000000017F0: 8B030503
	v_cmp_gt_f32_e64 s4, v78, v73                              // 0000000017F4: D4140004 0202934E
	s_wait_alu depctr_sa_sdst(0)                               // 0000000017FC: BF88FF9E
	s_or_b32 s3, s7, s3                                        // 000000001800: 8C030307
	v_cmp_eq_f32_e32 vcc_lo, v76, v71                          // 000000001804: 7C248F4C
	v_cmp_gt_f32_e64 s6, v76, v71                              // 000000001808: D4140006 02028F4C
	s_wait_alu depctr_sa_sdst(0)                               // 000000001810: BF88FF9E
	s_xor_b32 s2, s3, s0                                       // 000000001814: 8D020003
	v_mov_b32_e32 v74, v79                                     // 000000001818: 7E94034F
	s_wait_alu depctr_sa_sdst(0)                               // 00000000181C: BF88FF9E
	v_cndmask_b32_e64 v75, v75, v87, s2                        // 000000001820: D501004B 000AAF4B
	v_cndmask_b32_e64 v81, v81, v89, s2                        // 000000001828: D5010051 000AB351
	s_and_b32 s2, vcc_lo, s4                                   // 000000001830: 8B02046A
	v_permlanex16_b32 v72, v72, s10, 0xfedcba98                // 000000001834: D65C0048 03FC1548 FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 000000001840: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 000000001844: 8C020206
	v_mov_b32_e32 v87, v83                                     // 000000001848: 7EAE0353
	s_wait_alu depctr_sa_sdst(0)                               // 00000000184C: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000001850: 8D6A0002
	v_mov_b32_e32 v88, v84                                     // 000000001854: 7EB00354
	s_wait_alu depctr_sa_sdst(0)                               // 000000001858: BF88FF9E
	v_dual_cndmask_b32 v76, v76, v71 :: v_dual_mov_b32 v89, v85// 00000000185C: CA508F4C 4C580155
	v_mov_b32_e32 v71, v82                                     // 000000001864: 7E8E0352
	v_permlanex16_b32 v74, v74, s10, 0xfedcba98                // 000000001868: D65C004A 03FC154A FEDCBA98
	v_cmp_eq_f32_e64 s3, v77, v72                              // 000000001874: D4120003 0202914D
	v_cmp_gt_f32_e64 s7, v77, v72                              // 00000000187C: D4140007 0202914D
	v_permlanex16_b32 v87, v87, s10, 0xfedcba98                // 000000001884: D65C0057 03FC1557 FEDCBA98
	v_permlanex16_b32 v71, v71, s10, 0xfedcba98                // 000000001890: D65C0047 03FC1547 FEDCBA98
	v_cmp_gt_f32_e64 s5, v79, v74                              // 00000000189C: D4140005 0202954F
	v_permlanex16_b32 v89, v89, s10, 0xfedcba98                // 0000000018A4: D65C0059 03FC1559 FEDCBA98
	s_delay_alu instid0(VALU_DEP_3)                            // 0000000018B0: BF870003
	v_cmp_gt_f32_e64 s6, v82, v71                              // 0000000018B4: D4140006 02028F52
	s_and_b32 s3, s3, s5                                       // 0000000018BC: 8B030503
	v_cmp_gt_f32_e64 s5, v85, v89                              // 0000000018C0: D4140005 0202B355
	s_wait_alu depctr_sa_sdst(0)                               // 0000000018C8: BF88FF9E
	s_or_b32 s3, s7, s3                                        // 0000000018CC: 8C030307
	v_cmp_gt_f32_e64 s7, v83, v87                              // 0000000018D0: D4140007 0202AF53
	s_wait_alu depctr_sa_sdst(0)                               // 0000000018D8: BF88FF9E
	s_xor_b32 s2, s3, s0                                       // 0000000018DC: 8D020003
	v_cmp_eq_f32_e64 s3, v83, v87                              // 0000000018E0: D4120003 0202AF53
	s_wait_alu depctr_sa_sdst(0)                               // 0000000018E8: BF88FF9E
	v_cndmask_b32_e64 v72, v77, v72, s2                        // 0000000018EC: D5010048 000A914D
	v_cndmask_b32_e32 v77, v78, v73, vcc_lo                    // 0000000018F4: 029A934E
	v_permlanex16_b32 v88, v88, s10, 0xfedcba98                // 0000000018F8: D65C0058 03FC1558 FEDCBA98
	v_cmp_eq_f32_e32 vcc_lo, v82, v71                          // 000000001904: 7C248F52
	v_cndmask_b32_e64 v78, v79, v74, s2                        // 000000001908: D501004E 000A954F
	v_mov_b32_e32 v74, v67                                     // 000000001910: 7E940343
	s_and_b32 s3, s3, s5                                       // 000000001914: 8B030503
	v_cmp_gt_f32_e64 s4, v84, v88                              // 000000001918: D4140004 0202B154
	s_wait_alu depctr_sa_sdst(0)                               // 000000001920: BF88FF9E
	s_or_b32 s3, s7, s3                                        // 000000001924: 8C030307
	v_permlanex16_b32 v74, v74, s10, 0xfedcba98                // 000000001928: D65C004A 03FC154A FEDCBA98
	s_and_b32 s2, vcc_lo, s4                                   // 000000001934: 8B02046A
	s_wait_alu depctr_sa_sdst(0)                               // 000000001938: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 00000000193C: 8C020206
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001940: BF870001
	v_cmp_gt_f32_e64 s4, v67, v74                              // 000000001944: D4140004 02029543
	s_wait_alu depctr_sa_sdst(0)                               // 00000000194C: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000001950: 8D6A0002
	s_wait_alu depctr_sa_sdst(0)                               // 000000001954: BF88FF9E
	v_dual_mov_b32 v73, v69 :: v_dual_cndmask_b32 v84, v84, v88// 000000001958: CA120145 4954B154
	v_cndmask_b32_e32 v79, v82, v71, vcc_lo                    // 000000001960: 029E8F52
	v_dual_mov_b32 v71, v68 :: v_dual_mov_b32 v82, v66         // 000000001964: CA100144 47520142
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 00000000196C: BF870123
	v_permlanex16_b32 v73, v73, s10, 0xfedcba98                // 000000001970: D65C0049 03FC1549 FEDCBA98
	s_xor_b32 s2, s3, s0                                       // 00000000197C: 8D020003
	v_permlanex16_b32 v71, v71, s10, 0xfedcba98                // 000000001980: D65C0047 03FC1547 FEDCBA98
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 00000000198C: BF870193
	v_permlanex16_b32 v82, v82, s10, 0xfedcba98                // 000000001990: D65C0052 03FC1552 FEDCBA98
	v_cmp_eq_f32_e64 s3, v69, v73                              // 00000000199C: D4120003 02029345
	v_cmp_gt_f32_e64 s7, v69, v73                              // 0000000019A4: D4140007 02029345
	s_wait_alu depctr_sa_sdst(0)                               // 0000000019AC: BF88FF9E
	v_cndmask_b32_e64 v83, v83, v87, s2                        // 0000000019B0: D5010053 000AAF53
	v_cmp_eq_f32_e32 vcc_lo, v68, v71                          // 0000000019B8: 7C248F44
	v_cmp_gt_f32_e64 s5, v66, v82                              // 0000000019BC: D4140005 0202A542
	v_cmp_gt_f32_e64 s6, v68, v71                              // 0000000019C4: D4140006 02028F44
	v_cndmask_b32_e64 v85, v85, v89, s2                        // 0000000019CC: D5010055 000AB355
	s_and_b32 s2, vcc_lo, s4                                   // 0000000019D4: 8B02046A
	s_and_b32 s3, s3, s5                                       // 0000000019D8: 8B030503
	s_wait_alu depctr_sa_sdst(0)                               // 0000000019DC: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 0000000019E0: 8C020206
	s_or_b32 s3, s7, s3                                        // 0000000019E4: 8C030307
	s_wait_alu depctr_sa_sdst(0)                               // 0000000019E8: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 0000000019EC: 8D6A0002
	s_xor_b32 s2, s3, s0                                       // 0000000019F0: 8D020003
	s_wait_alu depctr_sa_sdst(0)                               // 0000000019F4: BF88FF9E
	v_cndmask_b32_e32 v87, v68, v71, vcc_lo                    // 0000000019F8: 02AE8F44
	v_cndmask_b32_e32 v89, v67, v74, vcc_lo                    // 0000000019FC: 02B29543
	v_cmp_eq_f32_e32 vcc_lo, v75, v70                          // 000000001A00: 7C248D4B
	v_cmp_lt_f32_e64 s4, v80, v81                              // 000000001A04: D4110004 0202A350
	v_cndmask_b32_e64 v88, v69, v73, s2                        // 000000001A0C: D5010058 000A9345
	v_cmp_lt_f32_e64 s3, v81, v80                              // 000000001A14: D4110003 0202A151
	v_cndmask_b32_e64 v82, v66, v82, s2                        // 000000001A1C: D5010052 000AA542
	v_cmp_lt_f32_e64 s2, v70, v75                              // 000000001A24: D4110002 02029746
	v_cmp_lt_f32_e64 s5, v75, v70                              // 000000001A2C: D4110005 02028D4B
	s_and_b32 s4, vcc_lo, s4                                   // 000000001A34: 8B04046A
	s_and_b32 s3, vcc_lo, s3                                   // 000000001A38: 8B03036A
	v_cmp_lt_f32_e64 s7, v76, v72                              // 000000001A3C: D4110007 0202914C
	s_wait_alu depctr_sa_sdst(0)                               // 000000001A44: BF88FF9E
	s_or_b32 s2, s2, s4                                        // 000000001A48: 8C020402
	s_or_b32 vcc_lo, s5, s3                                    // 000000001A4C: 8C6A0305
	s_wait_alu depctr_sa_sdst(0)                               // 000000001A50: BF88FF9E
	v_cndmask_b32_e64 v69, v70, v75, s2                        // 000000001A54: D5010045 000A9746
	v_cndmask_b32_e64 v66, v80, v81, s2                        // 000000001A5C: D5010042 000AA350
	v_cmp_eq_f32_e64 s2, v72, v76                              // 000000001A64: D4120002 02029948
	v_cmp_lt_f32_e64 s3, v78, v77                              // 000000001A6C: D4110003 02029B4E
	v_cndmask_b32_e32 v68, v70, v75, vcc_lo                    // 000000001A74: 02889746
	v_cmp_lt_f32_e64 s4, v77, v78                              // 000000001A78: D4110004 02029D4D
	v_cmp_eq_f32_e64 s5, v88, v87                              // 000000001A80: D4120005 0202AF58
	v_cmp_lt_f32_e64 s6, v89, v82                              // 000000001A88: D4110006 0202A559
	s_and_b32 s8, s2, s3                                       // 000000001A90: 8B080302
	v_cmp_lt_f32_e64 s3, v87, v88                              // 000000001A94: D4110003 0202B157
	s_and_b32 s4, s2, s4                                       // 000000001A9C: 8B040402
	s_and_b32 s2, s5, s6                                       // 000000001AA0: 8B020605
	v_cmp_lt_f32_e64 s6, v79, v83                              // 000000001AA4: D4110006 0202A74F
	s_wait_alu depctr_sa_sdst(0)                               // 000000001AAC: BF88FF9E
	s_or_b32 s2, s3, s2                                        // 000000001AB0: 8C020203
	v_cmp_lt_f32_e64 s3, v85, v84                              // 000000001AB4: D4110003 0202A955
	s_wait_alu depctr_sa_sdst(0)                               // 000000001ABC: BF88FF9E
	v_cndmask_b32_e64 v75, v87, v88, s2                        // 000000001AC0: D501004B 000AB157
	v_cndmask_b32_e32 v67, v80, v81, vcc_lo                    // 000000001AC8: 0286A350
	v_cmp_lt_f32_e32 vcc_lo, v72, v76                          // 000000001ACC: 7C229948
	v_cndmask_b32_e64 v74, v89, v82, s2                        // 000000001AD0: D501004A 000AA559
	s_or_b32 s2, s7, s4                                        // 000000001AD8: 8C020407
	v_cmp_lt_f32_e64 s4, v84, v85                              // 000000001ADC: D4110004 0202AB54
	s_wait_alu depctr_sa_sdst(0)                               // 000000001AE4: BF88FF9E
	v_cndmask_b32_e64 v73, v76, v72, s2                        // 000000001AE8: D5010049 000A914C
	s_or_b32 vcc_lo, vcc_lo, s8                                // 000000001AF0: 8C6A086A
	v_cndmask_b32_e64 v70, v77, v78, s2                        // 000000001AF4: D5010046 000A9D4D
	s_wait_alu depctr_sa_sdst(0)                               // 000000001AFC: BF88FF9E
	v_dual_cndmask_b32 v71, v76, v72 :: v_dual_max_num_f32 v76, v75, v75// 000000001B00: CA54914C 474C974B
	v_cndmask_b32_e32 v72, v77, v78, vcc_lo                    // 000000001B08: 02909D4D
	v_max_num_f32_e32 v77, v74, v74                            // 000000001B0C: 2C9A954A
	v_cmp_eq_f32_e64 s2, v83, v79                              // 000000001B10: D4120002 02029F53
	v_cmp_lt_f32_e32 vcc_lo, v83, v79                          // 000000001B18: 7C229F53
	v_max_num_f32_e32 v76, 0xff800000, v76                     // 000000001B1C: 2C9898FF FF800000
	s_and_b32 s3, s2, s3                                       // 000000001B24: 8B030302
	s_and_b32 s2, s2, s4                                       // 000000001B28: 8B020402
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001B2C: BF870001
	v_cndmask_b32_e64 v90, v76, 0xff800000, s1                 // 000000001B30: D501005A 0005FF4C FF800000
	v_max_num_f32_e32 v77, 0xff800000, v77                     // 000000001B3C: 2C9A9AFF FF800000
	s_wait_alu depctr_sa_sdst(0)                               // 000000001B44: BF88FF9E
	s_or_b32 vcc_lo, vcc_lo, s3                                // 000000001B48: 8C6A036A
	v_cmp_lt_f32_e64 s3, v88, v87                              // 000000001B4C: D4110003 0202AF58
	s_wait_alu depctr_sa_sdst(0)                               // 000000001B54: BF88FF9E
	v_cndmask_b32_e32 v76, v79, v83, vcc_lo                    // 000000001B58: 0298A74F
	v_mov_b32_e32 v80, v90                                     // 000000001B5C: 7EA0035A
	v_cndmask_b32_e64 v120, v77, 0xff800000, s1                // 000000001B60: D5010078 0005FF4D FF800000
	v_cndmask_b32_e32 v77, v84, v85, vcc_lo                    // 000000001B6C: 029AAB54
	v_cmp_lt_f32_e32 vcc_lo, v82, v89                          // 000000001B70: 7C22B352
	s_or_b32 s2, s6, s2                                        // 000000001B74: 8C020206
	v_permlanex16_b32 v80, v80, s10, 0xfedcba98                // 000000001B78: D65C0050 03FC1550 FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 000000001B84: BF88FF9E
	v_cndmask_b32_e64 v78, v79, v83, s2                        // 000000001B88: D501004E 000AA74F
	v_cndmask_b32_e64 v79, v84, v85, s2                        // 000000001B90: D501004F 000AAB54
	s_and_b32 s2, s5, vcc_lo                                   // 000000001B98: 8B026A05
	v_mov_b32_e32 v85, 0x7f800000                              // 000000001B9C: 7EAA02FF 7F800000
	s_wait_alu depctr_sa_sdst(0)                               // 000000001BA4: BF88FF9E
	s_or_b32 vcc_lo, s3, s2                                    // 000000001BA8: 8C6A0203
	v_mov_b32_e32 v81, v120                                    // 000000001BAC: 7EA20378
	s_wait_alu depctr_sa_sdst(0)                               // 000000001BB0: BF88FF9E
	v_dual_max_num_f32 v83, v80, v80 :: v_dual_cndmask_b32 v80, v89, v82// 000000001BB4: CA92A150 5350A559
	v_dual_mov_b32 v89, 0x7f800000 :: v_dual_mov_b32 v82, -1.0 // 000000001BBC: CA1000FF 595200F3 7F800000
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001BC8: BF870093
	v_permlanex16_b32 v81, v81, s10, 0xfedcba98                // 000000001BCC: D65C0051 03FC1551 FEDCBA98
	v_dual_max_num_f32 v84, v81, v81 :: v_dual_cndmask_b32 v81, v87, v88// 000000001BD8: CA92A351 5450B157
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)// 000000001BE0: BF8701A4
	v_dual_max_num_f32 v119, v90, v83 :: v_dual_mov_b32 v88, -1.0// 000000001BE4: CA90A75A 775800F3
	v_mov_b32_e32 v90, 0x7f800000                              // 000000001BEC: 7EB402FF 7F800000
	v_dual_max_num_f32 v120, v120, v84 :: v_dual_mov_b32 v87, -1.0// 000000001BF4: CA90A978 785600F3
	v_dual_mov_b32 v84, 0x7f800000 :: v_dual_mov_b32 v83, -1.0 // 000000001BFC: CA1000FF 545200F3 7F800000
	s_branch 4                                                 // 000000001C08: BFA00004 <nearest+0x1c1c>
	s_add_co_i32 s20, s20, -1                                  // 000000001C0C: 8114C114
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001C10: BF870009
	s_cmp_lg_u32 s20, 0                                        // 000000001C14: BF078014
	s_cbranch_scc0 63934                                       // 000000001C18: BFA1F9BE <nearest+0x314>
	s_wait_alu depctr_sa_sdst(0)                               // 000000001C1C: BF88FF9E
	s_cmp_nlt_f32 s15, 0x41000000                              // 000000001C20: BF4EFF0F 41000000
	s_cbranch_scc1 65528                                       // 000000001C28: BFA2FFF8 <nearest+0x1c0c>
	s_cmp_le_f32 s15, 0                                        // 000000001C2C: BF43800F
	v_cmp_ge_f32_e64 s2, v119, -v0                             // 000000001C30: D4160002 42020177
	v_cmp_ge_f32_e64 s4, v119, -v1                             // 000000001C38: D4160004 42020377
	s_cselect_b32 s5, -1, 0                                    // 000000001C40: 980580C1
	s_wait_alu depctr_sa_sdst(0)                               // 000000001C44: BF88FF9E
	s_and_b32 s3, s2, s5                                       // 000000001C48: 8B030502
	s_and_b32 s6, s4, s5                                       // 000000001C4C: 8B060504
	s_mov_b32 s5, 0                                            // 000000001C50: BE850080
	s_wait_alu depctr_sa_sdst(0)                               // 000000001C54: BF88FF9E
	s_or_b32 vcc_lo, s3, s6                                    // 000000001C58: 8C6A0603
	s_wait_alu depctr_sa_sdst(0)                               // 000000001C5C: BF88FF9E
	s_cbranch_vccz 29                                          // 000000001C60: BFA3001D <nearest+0x1cd8>
	v_cmp_gt_f32_e64 s5, v119, -v0                             // 000000001C64: D4140005 42020177
	v_cmp_gt_f32_e32 vcc_lo, v120, v121                        // 000000001C6C: 7C28F378
	v_cmp_gt_f32_e64 s7, v119, -v1                             // 000000001C70: D4140007 42020377
	v_cmp_gt_f32_e64 s2, v120, v122                            // 000000001C78: D4140002 0202F578
	v_cmp_lt_f32_e64 s4, -1.0, v82                             // 000000001C80: D4110004 0202A4F3
	s_or_b32 s5, s5, vcc_lo                                    // 000000001C88: 8C056A05
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v83                         // 000000001C8C: 7C22A6F3
	s_or_b32 s7, s7, s2                                        // 000000001C90: 8C070207
	s_wait_alu depctr_sa_sdst(0)                               // 000000001C94: BF88FF9E
	s_and_b32 s2, s5, s3                                       // 000000001C98: 8B020305
	s_and_b32 s3, s7, s6                                       // 000000001C9C: 8B030607
	s_wait_alu depctr_sa_sdst(0)                               // 000000001CA0: BF88FF9E
	s_and_b32 s4, s4, s2                                       // 000000001CA4: 8B040204
	s_and_b32 s5, vcc_lo, s3                                   // 000000001CA8: 8B05036A
	s_wait_alu depctr_sa_sdst(0)                               // 000000001CAC: BF88FF9E
	s_or_b32 s4, s4, s5                                        // 000000001CB0: 8C040504
	s_wait_alu depctr_sa_sdst(0)                               // 000000001CB4: BF88FF9E
	v_cndmask_b32_e64 v137, 0, 1, s4                           // 000000001CB8: D5010089 00110280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001CC0: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v137                           // 000000001CC4: 7C9B1280
	s_cbranch_vccz 6                                           // 000000001CC8: BFA30006 <nearest+0x1ce4>
	s_mov_b32 s6, 0                                            // 000000001CCC: BE860080
	s_mov_b32 s5, 1.0                                          // 000000001CD0: BE8500F2
	s_branch 26                                                // 000000001CD4: BFA0001A <nearest+0x1d40>
	s_mov_b32 s6, 0x41000000                                   // 000000001CD8: BE8600FF 41000000
	s_branch 23                                                // 000000001CE0: BFA00017 <nearest+0x1d40>
	v_cndmask_b32_e64 v137, v90, -v0, s2                       // 000000001CE4: D5010089 400A015A
	v_cndmask_b32_e64 v138, v89, -v1, s3                       // 000000001CEC: D501008A 400E0359
	v_cndmask_b32_e64 v139, v88, v121, s2                      // 000000001CF4: D501008B 000AF358
	v_cndmask_b32_e64 v140, v87, v122, s3                      // 000000001CFC: D501008C 000EF557
	v_cndmask_b32_e64 v84, v84, v90, s2                        // 000000001D04: D5010054 000AB554
	v_cndmask_b32_e64 v85, v85, v89, s3                        // 000000001D0C: D5010055 000EB355
	v_cndmask_b32_e64 v82, v82, v88, s2                        // 000000001D14: D5010052 000AB152
	v_cndmask_b32_e64 v83, v83, v87, s3                        // 000000001D1C: D5010053 000EAF53
	v_dual_mov_b32 v88, v139 :: v_dual_mov_b32 v87, v140       // 000000001D24: CA10018B 5856018C
	v_dual_mov_b32 v90, v137 :: v_dual_mov_b32 v89, v138       // 000000001D2C: CA100189 5A58018A
	s_mov_b32 s6, 0x41000000                                   // 000000001D34: BE8600FF 41000000
	s_mov_b32 s5, 0                                            // 000000001D3C: BE850080
	s_wait_alu depctr_sa_sdst(0)                               // 000000001D40: BF88FF9E
	s_cmp_lt_f32 s5, 0.5                                       // 000000001D44: BF41F005
	v_cmp_ge_f32_e64 s3, v119, -v2                             // 000000001D48: D4160003 42020577
	v_cmp_ge_f32_e64 s4, v119, -v3                             // 000000001D50: D4160004 42020777
	s_cselect_b32 s2, -1, 0                                    // 000000001D58: 980280C1
	s_cmp_le_f32 s15, 1.0                                      // 000000001D5C: BF43F20F
	s_cselect_b32 s7, -1, 0                                    // 000000001D60: 980780C1
	s_wait_alu depctr_sa_sdst(0)                               // 000000001D64: BF88FF9E
	s_and_b32 s2, s7, s2                                       // 000000001D68: 8B020207
	s_wait_alu depctr_sa_sdst(0)                               // 000000001D6C: BF88FF9E
	s_and_b32 s3, s3, s2                                       // 000000001D70: 8B030203
	s_and_b32 s7, s4, s2                                       // 000000001D74: 8B070204
	s_wait_alu depctr_sa_sdst(0)                               // 000000001D78: BF88FF9E
	s_or_b32 vcc_lo, s3, s7                                    // 000000001D7C: 8C6A0703
	s_wait_alu depctr_sa_sdst(0)                               // 000000001D80: BF88FF9E
	s_cbranch_vccz 48                                          // 000000001D84: BFA30030 <nearest+0x1e48>
	v_cmp_gt_f32_e64 s8, v119, -v2                             // 000000001D88: D4140008 42020577
	v_cmp_gt_f32_e32 vcc_lo, v120, v123                        // 000000001D90: 7C28F778
	v_cmp_gt_f32_e64 s21, v119, -v3                            // 000000001D94: D4140015 42020777
	v_cmp_gt_f32_e64 s2, v120, v124                            // 000000001D9C: D4140002 0202F978
	v_cmp_lt_f32_e64 s4, -1.0, v82                             // 000000001DA4: D4110004 0202A4F3
	s_or_b32 s8, s8, vcc_lo                                    // 000000001DAC: 8C086A08
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v83                         // 000000001DB0: 7C22A6F3
	s_or_b32 s21, s21, s2                                      // 000000001DB4: 8C150215
	s_and_b32 s2, s8, s3                                       // 000000001DB8: 8B020308
	s_and_b32 s3, s21, s7                                      // 000000001DBC: 8B030715
	s_wait_alu depctr_sa_sdst(0)                               // 000000001DC0: BF88FF9E
	s_and_b32 s4, s4, s2                                       // 000000001DC4: 8B040204
	s_and_b32 s7, vcc_lo, s3                                   // 000000001DC8: 8B07036A
	s_wait_alu depctr_sa_sdst(0)                               // 000000001DCC: BF88FF9E
	s_or_b32 s4, s4, s7                                        // 000000001DD0: 8C040704
	s_wait_alu depctr_sa_sdst(0)                               // 000000001DD4: BF88FF9E
	v_cndmask_b32_e64 v137, 0, 1, s4                           // 000000001DD8: D5010089 00110280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001DE0: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v137                           // 000000001DE4: 7C9B1280
	s_cbranch_vccz 3                                           // 000000001DE8: BFA30003 <nearest+0x1df8>
	s_mov_b32 s5, 1.0                                          // 000000001DEC: BE8500F2
	s_mov_b32 s6, 1.0                                          // 000000001DF0: BE8600F2
	s_branch 20                                                // 000000001DF4: BFA00014 <nearest+0x1e48>
	v_cndmask_b32_e64 v137, v90, -v2, s2                       // 000000001DF8: D5010089 400A055A
	v_cndmask_b32_e64 v138, v89, -v3, s3                       // 000000001E00: D501008A 400E0759
	v_cndmask_b32_e64 v139, v88, v123, s2                      // 000000001E08: D501008B 000AF758
	v_cndmask_b32_e64 v140, v87, v124, s3                      // 000000001E10: D501008C 000EF957
	v_cndmask_b32_e64 v84, v84, v90, s2                        // 000000001E18: D5010054 000AB554
	v_cndmask_b32_e64 v85, v85, v89, s3                        // 000000001E20: D5010055 000EB355
	v_cndmask_b32_e64 v82, v82, v88, s2                        // 000000001E28: D5010052 000AB152
	v_cndmask_b32_e64 v83, v83, v87, s3                        // 000000001E30: D5010053 000EAF53
	v_dual_mov_b32 v88, v139 :: v_dual_mov_b32 v87, v140       // 000000001E38: CA10018B 5856018C
	v_dual_mov_b32 v90, v137 :: v_dual_mov_b32 v89, v138       // 000000001E40: CA100189 5A58018A
	s_wait_alu depctr_sa_sdst(0)                               // 000000001E48: BF88FF9E
	s_cmp_lt_f32 s5, 0.5                                       // 000000001E4C: BF41F005
	v_cmp_ge_f32_e64 s3, v119, -v4                             // 000000001E50: D4160003 42020977
	v_cmp_ge_f32_e64 s4, v119, -v5                             // 000000001E58: D4160004 42020B77
	s_cselect_b32 s2, -1, 0                                    // 000000001E60: 980280C1
	s_cmp_le_f32 s15, 2.0                                      // 000000001E64: BF43F40F
	s_cselect_b32 s7, -1, 0                                    // 000000001E68: 980780C1
	s_wait_alu depctr_sa_sdst(0)                               // 000000001E6C: BF88FF9E
	s_and_b32 s2, s7, s2                                       // 000000001E70: 8B020207
	s_wait_alu depctr_sa_sdst(0)                               // 000000001E74: BF88FF9E
	s_and_b32 s3, s3, s2                                       // 000000001E78: 8B030203
	s_and_b32 s7, s4, s2                                       // 000000001E7C: 8B070204
	s_wait_alu depctr_sa_sdst(0)                               // 000000001E80: BF88FF9E
	s_or_b32 vcc_lo, s3, s7                                    // 000000001E84: 8C6A0703
	s_wait_alu depctr_sa_sdst(0)                               // 000000001E88: BF88FF9E
	s_cbranch_vccz 48                                          // 000000001E8C: BFA30030 <nearest+0x1f50>
	v_cmp_gt_f32_e64 s8, v119, -v4                             // 000000001E90: D4140008 42020977
	v_cmp_gt_f32_e32 vcc_lo, v120, v125                        // 000000001E98: 7C28FB78
	v_cmp_gt_f32_e64 s21, v119, -v5                            // 000000001E9C: D4140015 42020B77
	v_cmp_gt_f32_e64 s2, v120, v126                            // 000000001EA4: D4140002 0202FD78
	v_cmp_lt_f32_e64 s4, -1.0, v82                             // 000000001EAC: D4110004 0202A4F3
	s_or_b32 s8, s8, vcc_lo                                    // 000000001EB4: 8C086A08
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v83                         // 000000001EB8: 7C22A6F3
	s_or_b32 s21, s21, s2                                      // 000000001EBC: 8C150215
	s_and_b32 s2, s8, s3                                       // 000000001EC0: 8B020308
	s_and_b32 s3, s21, s7                                      // 000000001EC4: 8B030715
	s_wait_alu depctr_sa_sdst(0)                               // 000000001EC8: BF88FF9E
	s_and_b32 s4, s4, s2                                       // 000000001ECC: 8B040204
	s_and_b32 s7, vcc_lo, s3                                   // 000000001ED0: 8B07036A
	s_wait_alu depctr_sa_sdst(0)                               // 000000001ED4: BF88FF9E
	s_or_b32 s4, s4, s7                                        // 000000001ED8: 8C040704
	s_wait_alu depctr_sa_sdst(0)                               // 000000001EDC: BF88FF9E
	v_cndmask_b32_e64 v137, 0, 1, s4                           // 000000001EE0: D5010089 00110280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001EE8: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v137                           // 000000001EEC: 7C9B1280
	s_cbranch_vccz 3                                           // 000000001EF0: BFA30003 <nearest+0x1f00>
	s_mov_b32 s6, 2.0                                          // 000000001EF4: BE8600F4
	s_mov_b32 s5, 1.0                                          // 000000001EF8: BE8500F2
	s_branch 20                                                // 000000001EFC: BFA00014 <nearest+0x1f50>
	v_cndmask_b32_e64 v137, v90, -v4, s2                       // 000000001F00: D5010089 400A095A
	v_cndmask_b32_e64 v138, v89, -v5, s3                       // 000000001F08: D501008A 400E0B59
	v_cndmask_b32_e64 v139, v88, v125, s2                      // 000000001F10: D501008B 000AFB58
	v_cndmask_b32_e64 v140, v87, v126, s3                      // 000000001F18: D501008C 000EFD57
	v_cndmask_b32_e64 v84, v84, v90, s2                        // 000000001F20: D5010054 000AB554
	v_cndmask_b32_e64 v85, v85, v89, s3                        // 000000001F28: D5010055 000EB355
	v_cndmask_b32_e64 v82, v82, v88, s2                        // 000000001F30: D5010052 000AB152
	v_cndmask_b32_e64 v83, v83, v87, s3                        // 000000001F38: D5010053 000EAF53
	v_dual_mov_b32 v88, v139 :: v_dual_mov_b32 v87, v140       // 000000001F40: CA10018B 5856018C
	v_dual_mov_b32 v90, v137 :: v_dual_mov_b32 v89, v138       // 000000001F48: CA100189 5A58018A
	s_wait_alu depctr_sa_sdst(0)                               // 000000001F50: BF88FF9E
	s_cmp_lt_f32 s5, 0.5                                       // 000000001F54: BF41F005
	v_cmp_ge_f32_e64 s3, v119, -v6                             // 000000001F58: D4160003 42020D77
	v_cmp_ge_f32_e64 s4, v119, -v7                             // 000000001F60: D4160004 42020F77
	s_cselect_b32 s2, -1, 0                                    // 000000001F68: 980280C1
	s_cmp_le_f32 s15, 0x40400000                               // 000000001F6C: BF43FF0F 40400000
	s_cselect_b32 s7, -1, 0                                    // 000000001F74: 980780C1
	s_wait_alu depctr_sa_sdst(0)                               // 000000001F78: BF88FF9E
	s_and_b32 s2, s7, s2                                       // 000000001F7C: 8B020207
	s_wait_alu depctr_sa_sdst(0)                               // 000000001F80: BF88FF9E
	s_and_b32 s3, s3, s2                                       // 000000001F84: 8B030203
	s_and_b32 s7, s4, s2                                       // 000000001F88: 8B070204
	s_wait_alu depctr_sa_sdst(0)                               // 000000001F8C: BF88FF9E
	s_or_b32 vcc_lo, s3, s7                                    // 000000001F90: 8C6A0703
	s_wait_alu depctr_sa_sdst(0)                               // 000000001F94: BF88FF9E
	s_cbranch_vccz 49                                          // 000000001F98: BFA30031 <nearest+0x2060>
	v_cmp_gt_f32_e64 s8, v119, -v6                             // 000000001F9C: D4140008 42020D77
	v_cmp_gt_f32_e32 vcc_lo, v120, v127                        // 000000001FA4: 7C28FF78
	v_cmp_gt_f32_e64 s21, v119, -v7                            // 000000001FA8: D4140015 42020F77
	v_cmp_gt_f32_e64 s2, v120, v128                            // 000000001FB0: D4140002 02030178
	v_cmp_lt_f32_e64 s4, -1.0, v82                             // 000000001FB8: D4110004 0202A4F3
	s_or_b32 s8, s8, vcc_lo                                    // 000000001FC0: 8C086A08
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v83                         // 000000001FC4: 7C22A6F3
	s_or_b32 s21, s21, s2                                      // 000000001FC8: 8C150215
	s_and_b32 s2, s8, s3                                       // 000000001FCC: 8B020308
	s_and_b32 s3, s21, s7                                      // 000000001FD0: 8B030715
	s_wait_alu depctr_sa_sdst(0)                               // 000000001FD4: BF88FF9E
	s_and_b32 s4, s4, s2                                       // 000000001FD8: 8B040204
	s_and_b32 s7, vcc_lo, s3                                   // 000000001FDC: 8B07036A
	s_wait_alu depctr_sa_sdst(0)                               // 000000001FE0: BF88FF9E
	s_or_b32 s4, s4, s7                                        // 000000001FE4: 8C040704
	s_wait_alu depctr_sa_sdst(0)                               // 000000001FE8: BF88FF9E
	v_cndmask_b32_e64 v137, 0, 1, s4                           // 000000001FEC: D5010089 00110280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001FF4: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v137                           // 000000001FF8: 7C9B1280
	s_cbranch_vccz 4                                           // 000000001FFC: BFA30004 <nearest+0x2010>
	s_mov_b32 s6, 0x40400000                                   // 000000002000: BE8600FF 40400000
	s_mov_b32 s5, 1.0                                          // 000000002008: BE8500F2
	s_branch 20                                                // 00000000200C: BFA00014 <nearest+0x2060>
	v_cndmask_b32_e64 v137, v90, -v6, s2                       // 000000002010: D5010089 400A0D5A
	v_cndmask_b32_e64 v138, v89, -v7, s3                       // 000000002018: D501008A 400E0F59
	v_cndmask_b32_e64 v139, v88, v127, s2                      // 000000002020: D501008B 000AFF58
	v_cndmask_b32_e64 v140, v87, v128, s3                      // 000000002028: D501008C 000F0157
	v_cndmask_b32_e64 v84, v84, v90, s2                        // 000000002030: D5010054 000AB554
	v_cndmask_b32_e64 v85, v85, v89, s3                        // 000000002038: D5010055 000EB355
	v_cndmask_b32_e64 v82, v82, v88, s2                        // 000000002040: D5010052 000AB152
	v_cndmask_b32_e64 v83, v83, v87, s3                        // 000000002048: D5010053 000EAF53
	v_dual_mov_b32 v88, v139 :: v_dual_mov_b32 v87, v140       // 000000002050: CA10018B 5856018C
	v_dual_mov_b32 v90, v137 :: v_dual_mov_b32 v89, v138       // 000000002058: CA100189 5A58018A
	s_wait_alu depctr_sa_sdst(0)                               // 000000002060: BF88FF9E
	s_cmp_lt_f32 s5, 0.5                                       // 000000002064: BF41F005
	v_cmp_ge_f32_e64 s3, v119, -v8                             // 000000002068: D4160003 42021177
	v_cmp_ge_f32_e64 s4, v119, -v9                             // 000000002070: D4160004 42021377
	s_cselect_b32 s2, -1, 0                                    // 000000002078: 980280C1
	s_cmp_le_f32 s15, 4.0                                      // 00000000207C: BF43F60F
	s_cselect_b32 s7, -1, 0                                    // 000000002080: 980780C1
	s_wait_alu depctr_sa_sdst(0)                               // 000000002084: BF88FF9E
	s_and_b32 s2, s7, s2                                       // 000000002088: 8B020207
	s_wait_alu depctr_sa_sdst(0)                               // 00000000208C: BF88FF9E
	s_and_b32 s3, s3, s2                                       // 000000002090: 8B030203
	s_and_b32 s7, s4, s2                                       // 000000002094: 8B070204
	s_wait_alu depctr_sa_sdst(0)                               // 000000002098: BF88FF9E
	s_or_b32 vcc_lo, s3, s7                                    // 00000000209C: 8C6A0703
	s_wait_alu depctr_sa_sdst(0)                               // 0000000020A0: BF88FF9E
	s_cbranch_vccz 48                                          // 0000000020A4: BFA30030 <nearest+0x2168>
	v_cmp_gt_f32_e64 s8, v119, -v8                             // 0000000020A8: D4140008 42021177
	v_cmp_gt_f32_e32 vcc_lo, v120, v129                        // 0000000020B0: 7C290378
	v_cmp_gt_f32_e64 s21, v119, -v9                            // 0000000020B4: D4140015 42021377
	v_cmp_gt_f32_e64 s2, v120, v130                            // 0000000020BC: D4140002 02030578
	v_cmp_lt_f32_e64 s4, -1.0, v82                             // 0000000020C4: D4110004 0202A4F3
	s_or_b32 s8, s8, vcc_lo                                    // 0000000020CC: 8C086A08
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v83                         // 0000000020D0: 7C22A6F3
	s_or_b32 s21, s21, s2                                      // 0000000020D4: 8C150215
	s_and_b32 s2, s8, s3                                       // 0000000020D8: 8B020308
	s_and_b32 s3, s21, s7                                      // 0000000020DC: 8B030715
	s_wait_alu depctr_sa_sdst(0)                               // 0000000020E0: BF88FF9E
	s_and_b32 s4, s4, s2                                       // 0000000020E4: 8B040204
	s_and_b32 s7, vcc_lo, s3                                   // 0000000020E8: 8B07036A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000020EC: BF88FF9E
	s_or_b32 s4, s4, s7                                        // 0000000020F0: 8C040704
	s_wait_alu depctr_sa_sdst(0)                               // 0000000020F4: BF88FF9E
	v_cndmask_b32_e64 v137, 0, 1, s4                           // 0000000020F8: D5010089 00110280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000002100: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v137                           // 000000002104: 7C9B1280
	s_cbranch_vccz 3                                           // 000000002108: BFA30003 <nearest+0x2118>
	s_mov_b32 s6, 4.0                                          // 00000000210C: BE8600F6
	s_mov_b32 s5, 1.0                                          // 000000002110: BE8500F2
	s_branch 20                                                // 000000002114: BFA00014 <nearest+0x2168>
	v_cndmask_b32_e64 v137, v90, -v8, s2                       // 000000002118: D5010089 400A115A
	v_cndmask_b32_e64 v138, v89, -v9, s3                       // 000000002120: D501008A 400E1359
	v_cndmask_b32_e64 v139, v88, v129, s2                      // 000000002128: D501008B 000B0358
	v_cndmask_b32_e64 v140, v87, v130, s3                      // 000000002130: D501008C 000F0557
	v_cndmask_b32_e64 v84, v84, v90, s2                        // 000000002138: D5010054 000AB554
	v_cndmask_b32_e64 v85, v85, v89, s3                        // 000000002140: D5010055 000EB355
	v_cndmask_b32_e64 v82, v82, v88, s2                        // 000000002148: D5010052 000AB152
	v_cndmask_b32_e64 v83, v83, v87, s3                        // 000000002150: D5010053 000EAF53
	v_dual_mov_b32 v88, v139 :: v_dual_mov_b32 v87, v140       // 000000002158: CA10018B 5856018C
	v_dual_mov_b32 v90, v137 :: v_dual_mov_b32 v89, v138       // 000000002160: CA100189 5A58018A
	s_wait_alu depctr_sa_sdst(0)                               // 000000002168: BF88FF9E
	s_cmp_lt_f32 s5, 0.5                                       // 00000000216C: BF41F005
	v_cmp_ge_f32_e64 s3, v119, -v10                            // 000000002170: D4160003 42021577
	v_cmp_ge_f32_e64 s4, v119, -v11                            // 000000002178: D4160004 42021777
	s_cselect_b32 s2, -1, 0                                    // 000000002180: 980280C1
	s_cmp_le_f32 s15, 0x40a00000                               // 000000002184: BF43FF0F 40A00000
	s_cselect_b32 s7, -1, 0                                    // 00000000218C: 980780C1
	s_wait_alu depctr_sa_sdst(0)                               // 000000002190: BF88FF9E
	s_and_b32 s2, s7, s2                                       // 000000002194: 8B020207
	s_wait_alu depctr_sa_sdst(0)                               // 000000002198: BF88FF9E
	s_and_b32 s3, s3, s2                                       // 00000000219C: 8B030203
	s_and_b32 s7, s4, s2                                       // 0000000021A0: 8B070204
	s_wait_alu depctr_sa_sdst(0)                               // 0000000021A4: BF88FF9E
	s_or_b32 vcc_lo, s3, s7                                    // 0000000021A8: 8C6A0703
	s_wait_alu depctr_sa_sdst(0)                               // 0000000021AC: BF88FF9E
	s_cbranch_vccz 49                                          // 0000000021B0: BFA30031 <nearest+0x2278>
	v_cmp_gt_f32_e64 s8, v119, -v10                            // 0000000021B4: D4140008 42021577
	v_cmp_gt_f32_e32 vcc_lo, v120, v131                        // 0000000021BC: 7C290778
	v_cmp_gt_f32_e64 s21, v119, -v11                           // 0000000021C0: D4140015 42021777
	v_cmp_gt_f32_e64 s2, v120, v132                            // 0000000021C8: D4140002 02030978
	v_cmp_lt_f32_e64 s4, -1.0, v82                             // 0000000021D0: D4110004 0202A4F3
	s_or_b32 s8, s8, vcc_lo                                    // 0000000021D8: 8C086A08
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v83                         // 0000000021DC: 7C22A6F3
	s_or_b32 s21, s21, s2                                      // 0000000021E0: 8C150215
	s_and_b32 s2, s8, s3                                       // 0000000021E4: 8B020308
	s_and_b32 s3, s21, s7                                      // 0000000021E8: 8B030715
	s_wait_alu depctr_sa_sdst(0)                               // 0000000021EC: BF88FF9E
	s_and_b32 s4, s4, s2                                       // 0000000021F0: 8B040204
	s_and_b32 s7, vcc_lo, s3                                   // 0000000021F4: 8B07036A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000021F8: BF88FF9E
	s_or_b32 s4, s4, s7                                        // 0000000021FC: 8C040704
	s_wait_alu depctr_sa_sdst(0)                               // 000000002200: BF88FF9E
	v_cndmask_b32_e64 v137, 0, 1, s4                           // 000000002204: D5010089 00110280
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000220C: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v137                           // 000000002210: 7C9B1280
	s_cbranch_vccz 4                                           // 000000002214: BFA30004 <nearest+0x2228>
	s_mov_b32 s6, 0x40a00000                                   // 000000002218: BE8600FF 40A00000
	s_mov_b32 s5, 1.0                                          // 000000002220: BE8500F2
	s_branch 20                                                // 000000002224: BFA00014 <nearest+0x2278>
	v_cndmask_b32_e64 v137, v90, -v10, s2                      // 000000002228: D5010089 400A155A
	v_cndmask_b32_e64 v138, v89, -v11, s3                      // 000000002230: D501008A 400E1759
	v_cndmask_b32_e64 v139, v88, v131, s2                      // 000000002238: D501008B 000B0758
	v_cndmask_b32_e64 v140, v87, v132, s3                      // 000000002240: D501008C 000F0957
	v_cndmask_b32_e64 v84, v84, v90, s2                        // 000000002248: D5010054 000AB554
	v_cndmask_b32_e64 v85, v85, v89, s3                        // 000000002250: D5010055 000EB355
	v_cndmask_b32_e64 v82, v82, v88, s2                        // 000000002258: D5010052 000AB152
	v_cndmask_b32_e64 v83, v83, v87, s3                        // 000000002260: D5010053 000EAF53
	v_dual_mov_b32 v88, v139 :: v_dual_mov_b32 v87, v140       // 000000002268: CA10018B 5856018C
	v_dual_mov_b32 v90, v137 :: v_dual_mov_b32 v89, v138       // 000000002270: CA100189 5A58018A
	s_wait_alu depctr_sa_sdst(0)                               // 000000002278: BF88FF9E
	s_cmp_lt_f32 s5, 0.5                                       // 00000000227C: BF41F005
	v_cmp_ge_f32_e64 s3, v119, -v12                            // 000000002280: D4160003 42021977
	v_cmp_ge_f32_e64 s4, v119, -v13                            // 000000002288: D4160004 42021B77
	s_cselect_b32 s2, -1, 0                                    // 000000002290: 980280C1
	s_cmp_le_f32 s15, 0x40c00000                               // 000000002294: BF43FF0F 40C00000
	s_cselect_b32 s7, -1, 0                                    // 00000000229C: 980780C1
	s_wait_alu depctr_sa_sdst(0)                               // 0000000022A0: BF88FF9E
	s_and_b32 s2, s7, s2                                       // 0000000022A4: 8B020207
	s_wait_alu depctr_sa_sdst(0)                               // 0000000022A8: BF88FF9E
	s_and_b32 s3, s3, s2                                       // 0000000022AC: 8B030203
	s_and_b32 s7, s4, s2                                       // 0000000022B0: 8B070204
	s_wait_alu depctr_sa_sdst(0)                               // 0000000022B4: BF88FF9E
	s_or_b32 vcc_lo, s3, s7                                    // 0000000022B8: 8C6A0703
	s_wait_alu depctr_sa_sdst(0)                               // 0000000022BC: BF88FF9E
	s_cbranch_vccz 49                                          // 0000000022C0: BFA30031 <nearest+0x2388>
	v_cmp_gt_f32_e64 s8, v119, -v12                            // 0000000022C4: D4140008 42021977
	v_cmp_gt_f32_e32 vcc_lo, v120, v133                        // 0000000022CC: 7C290B78
	v_cmp_gt_f32_e64 s21, v119, -v13                           // 0000000022D0: D4140015 42021B77
	v_cmp_gt_f32_e64 s2, v120, v134                            // 0000000022D8: D4140002 02030D78
	v_cmp_lt_f32_e64 s4, -1.0, v82                             // 0000000022E0: D4110004 0202A4F3
	s_or_b32 s8, s8, vcc_lo                                    // 0000000022E8: 8C086A08
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v83                         // 0000000022EC: 7C22A6F3
	s_or_b32 s21, s21, s2                                      // 0000000022F0: 8C150215
	s_and_b32 s2, s8, s3                                       // 0000000022F4: 8B020308
	s_and_b32 s3, s21, s7                                      // 0000000022F8: 8B030715
	s_wait_alu depctr_sa_sdst(0)                               // 0000000022FC: BF88FF9E
	s_and_b32 s4, s4, s2                                       // 000000002300: 8B040204
	s_and_b32 s7, vcc_lo, s3                                   // 000000002304: 8B07036A
	s_wait_alu depctr_sa_sdst(0)                               // 000000002308: BF88FF9E
	s_or_b32 s4, s4, s7                                        // 00000000230C: 8C040704
	s_wait_alu depctr_sa_sdst(0)                               // 000000002310: BF88FF9E
	v_cndmask_b32_e64 v137, 0, 1, s4                           // 000000002314: D5010089 00110280
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000231C: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v137                           // 000000002320: 7C9B1280
	s_cbranch_vccz 4                                           // 000000002324: BFA30004 <nearest+0x2338>
	s_mov_b32 s6, 0x40c00000                                   // 000000002328: BE8600FF 40C00000
	s_mov_b32 s5, 1.0                                          // 000000002330: BE8500F2
	s_branch 20                                                // 000000002334: BFA00014 <nearest+0x2388>
	v_cndmask_b32_e64 v137, v90, -v12, s2                      // 000000002338: D5010089 400A195A
	v_cndmask_b32_e64 v138, v89, -v13, s3                      // 000000002340: D501008A 400E1B59
	v_cndmask_b32_e64 v139, v88, v133, s2                      // 000000002348: D501008B 000B0B58
	v_cndmask_b32_e64 v140, v87, v134, s3                      // 000000002350: D501008C 000F0D57
	v_cndmask_b32_e64 v84, v84, v90, s2                        // 000000002358: D5010054 000AB554
	v_cndmask_b32_e64 v85, v85, v89, s3                        // 000000002360: D5010055 000EB355
	v_cndmask_b32_e64 v82, v82, v88, s2                        // 000000002368: D5010052 000AB152
	v_cndmask_b32_e64 v83, v83, v87, s3                        // 000000002370: D5010053 000EAF53
	v_dual_mov_b32 v88, v139 :: v_dual_mov_b32 v87, v140       // 000000002378: CA10018B 5856018C
	v_dual_mov_b32 v90, v137 :: v_dual_mov_b32 v89, v138       // 000000002380: CA100189 5A58018A
	s_wait_alu depctr_sa_sdst(0)                               // 000000002388: BF88FF9E
	s_cmp_lt_f32 s5, 0.5                                       // 00000000238C: BF41F005
	v_cmp_ge_f32_e64 s3, v119, -v14                            // 000000002390: D4160003 42021D77
	v_cmp_ge_f32_e64 s4, v119, -v15                            // 000000002398: D4160004 42021F77
	s_cselect_b32 s2, -1, 0                                    // 0000000023A0: 980280C1
	s_cmp_le_f32 s15, 0x40e00000                               // 0000000023A4: BF43FF0F 40E00000
	s_cselect_b32 s7, -1, 0                                    // 0000000023AC: 980780C1
	s_wait_alu depctr_sa_sdst(0)                               // 0000000023B0: BF88FF9E
	s_and_b32 s2, s7, s2                                       // 0000000023B4: 8B020207
	s_wait_alu depctr_sa_sdst(0)                               // 0000000023B8: BF88FF9E
	s_and_b32 s3, s3, s2                                       // 0000000023BC: 8B030203
	s_and_b32 s7, s4, s2                                       // 0000000023C0: 8B070204
	s_wait_alu depctr_sa_sdst(0)                               // 0000000023C4: BF88FF9E
	s_or_b32 vcc_lo, s3, s7                                    // 0000000023C8: 8C6A0703
	s_wait_alu depctr_sa_sdst(0)                               // 0000000023CC: BF88FF9E
	s_cbranch_vccz 29                                          // 0000000023D0: BFA3001D <nearest+0x2448>
	v_cmp_gt_f32_e64 s8, v119, -v14                            // 0000000023D4: D4140008 42021D77
	v_cmp_gt_f32_e32 vcc_lo, v120, v135                        // 0000000023DC: 7C290F78
	v_cmp_gt_f32_e64 s15, v119, -v15                           // 0000000023E0: D414000F 42021F77
	v_cmp_gt_f32_e64 s2, v120, v136                            // 0000000023E8: D4140002 02031178
	v_cmp_lt_f32_e64 s4, -1.0, v82                             // 0000000023F0: D4110004 0202A4F3
	s_or_b32 s8, s8, vcc_lo                                    // 0000000023F8: 8C086A08
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v83                         // 0000000023FC: 7C22A6F3
	s_or_b32 s15, s15, s2                                      // 000000002400: 8C0F020F
	s_and_b32 s2, s8, s3                                       // 000000002404: 8B020308
	s_wait_alu depctr_sa_sdst(0)                               // 000000002408: BF88FF9E
	s_and_b32 s3, s15, s7                                      // 00000000240C: 8B03070F
	s_and_b32 s4, s4, s2                                       // 000000002410: 8B040204
	s_wait_alu depctr_sa_sdst(0)                               // 000000002414: BF88FF9E
	s_and_b32 s7, vcc_lo, s3                                   // 000000002418: 8B07036A
	s_wait_alu depctr_sa_sdst(0)                               // 00000000241C: BF88FF9E
	s_or_b32 s4, s4, s7                                        // 000000002420: 8C040704
	s_wait_alu depctr_sa_sdst(0)                               // 000000002424: BF88FF9E
	v_cndmask_b32_e64 v137, 0, 1, s4                           // 000000002428: D5010089 00110280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000002430: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v137                           // 000000002434: 7C9B1280
	s_cbranch_vccz 854                                         // 000000002438: BFA30356 <nearest+0x3194>
	s_mov_b32 s6, 0x40e00000                                   // 00000000243C: BE8600FF 40E00000
	s_mov_b32 s5, 1.0                                          // 000000002444: BE8500F2
	s_wait_alu depctr_sa_sdst(0)                               // 000000002448: BF88FF9E
	s_cmp_ngt_f32 s5, 0.5                                      // 00000000244C: BF4BF005
	s_mov_b32 s15, s6                                          // 000000002450: BE8F0006
	s_cbranch_scc1 65005                                       // 000000002454: BFA2FDED <nearest+0x1c0c>
	v_cmp_eq_f32_e32 vcc_lo, v89, v90                          // 000000002458: 7C24B559
	v_cmp_lt_f32_e64 s2, v87, v88                              // 00000000245C: D4110002 0202B157
	v_cmp_lt_f32_e64 s4, v89, v90                              // 000000002464: D4110004 0202B559
	v_cmp_lt_f32_e64 s3, v88, v87                              // 00000000246C: D4110003 0202AF58
	v_cmp_lt_f32_e64 s5, v90, v89                              // 000000002474: D4110005 0202B35A
	v_cmp_lt_f32_e64 s6, v84, v85                              // 00000000247C: D4110006 0202AB54
	s_and_b32 s2, vcc_lo, s2                                   // 000000002484: 8B02026A
	s_wait_alu depctr_sa_sdst(0)                               // 000000002488: BF88FF9E
	s_or_b32 s2, s4, s2                                        // 00000000248C: 8C020204
	s_and_b32 s3, vcc_lo, s3                                   // 000000002490: 8B03036A
	s_wait_alu depctr_sa_sdst(0)                               // 000000002494: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000002498: 8D6A0002
	s_or_b32 s3, s5, s3                                        // 00000000249C: 8C030305
	s_wait_alu depctr_sa_sdst(0)                               // 0000000024A0: BF88FF9E
	v_cndmask_b32_e32 v119, v90, v89, vcc_lo                   // 0000000024A4: 02EEB35A
	s_xor_b32 s2, s3, s9                                       // 0000000024A8: 8D020903
	v_cmp_lt_f32_e64 s3, v83, v82                              // 0000000024AC: D4110003 0202A553
	s_wait_alu depctr_sa_sdst(0)                               // 0000000024B4: BF88FF9E
	v_cndmask_b32_e64 v89, v89, v90, s2                        // 0000000024B8: D5010059 000AB559
	v_cndmask_b32_e32 v90, v88, v87, vcc_lo                    // 0000000024C0: 02B4AF58
	v_cndmask_b32_e64 v87, v87, v88, s2                        // 0000000024C4: D5010057 000AB157
	v_mov_b32_e32 v120, v119                                   // 0000000024CC: 7EF00377
	v_cmp_eq_f32_e32 vcc_lo, v85, v84                          // 0000000024D0: 7C24A955
	v_cmp_lt_f32_e64 s4, v82, v83                              // 0000000024D4: D4110004 0202A752
	v_cmp_lt_f32_e64 s5, v85, v84                              // 0000000024DC: D4110005 0202A955
	v_mov_b32_e32 v139, v87                                    // 0000000024E4: 7F160357
	v_permlanex16_b32 v120, v120, s10, 0xfedcba98              // 0000000024E8: D65C0078 03FC1578 FEDCBA98
	v_mov_b32_e32 v138, v90                                    // 0000000024F4: 7F14035A
	s_and_b32 s2, vcc_lo, s3                                   // 0000000024F8: 8B02036A
	s_and_b32 s3, vcc_lo, s4                                   // 0000000024FC: 8B03046A
	s_wait_alu depctr_sa_sdst(0)                               // 000000002500: BF88FF9E
	s_or_b32 s2, s5, s2                                        // 000000002504: 8C020205
	s_or_b32 s3, s6, s3                                        // 000000002508: 8C030306
	v_permlanex16_b32 v138, v138, s10, 0xfedcba98              // 00000000250C: D65C008A 03FC158A FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 000000002518: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 00000000251C: 8D6A0002
	s_xor_b32 s2, s3, s9                                       // 000000002520: 8D020903
	s_wait_alu depctr_sa_sdst(0)                               // 000000002524: BF88FF9E
	v_dual_cndmask_b32 v88, v84, v85 :: v_dual_mov_b32 v137, v89// 000000002528: CA50AB54 58880159
	v_cndmask_b32_e64 v84, v85, v84, s2                        // 000000002530: D5010054 000AA955
	v_cndmask_b32_e32 v85, v82, v83, vcc_lo                    // 000000002538: 02AAA752
	v_cmp_eq_f32_e32 vcc_lo, v119, v120                        // 00000000253C: 7C24F177
	v_cmp_gt_f32_e64 s4, v90, v138                             // 000000002540: D4140004 0203155A
	v_cmp_gt_f32_e64 s6, v119, v120                            // 000000002548: D4140006 0202F177
	v_cndmask_b32_e64 v82, v83, v82, s2                        // 000000002550: D5010052 000AA553
	v_permlanex16_b32 v137, v137, s10, 0xfedcba98              // 000000002558: D65C0089 03FC1589 FEDCBA98
	v_permlanex16_b32 v139, v139, s10, 0xfedcba98              // 000000002564: D65C008B 03FC158B FEDCBA98
	s_and_b32 s2, vcc_lo, s4                                   // 000000002570: 8B02046A
	s_wait_alu depctr_sa_sdst(0)                               // 000000002574: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 000000002578: 8C020206
	v_cmp_eq_f32_e64 s3, v89, v137                             // 00000000257C: D4120003 02031359
	s_wait_alu depctr_sa_sdst(0)                               // 000000002584: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000002588: 8D6A0002
	v_cmp_gt_f32_e64 s5, v87, v139                             // 00000000258C: D4140005 02031757
	s_wait_alu depctr_sa_sdst(0)                               // 000000002594: BF88FF9E
	v_dual_cndmask_b32 v83, v119, v120 :: v_dual_mov_b32 v120, v84// 000000002598: CA50F177 53780154
	v_dual_mov_b32 v119, v88 :: v_dual_mov_b32 v140, v85       // 0000000025A0: CA100158 778C0155
	v_cmp_gt_f32_e64 s7, v89, v137                             // 0000000025A8: D4140007 02031359
	s_and_b32 s3, s3, s5                                       // 0000000025B0: 8B030503
	v_permlanex16_b32 v120, v120, s10, 0xfedcba98              // 0000000025B4: D65C0078 03FC1578 FEDCBA98
	v_permlanex16_b32 v119, v119, s10, 0xfedcba98              // 0000000025C0: D65C0077 03FC1577 FEDCBA98
	v_permlanex16_b32 v140, v140, s10, 0xfedcba98              // 0000000025CC: D65C008C 03FC158C FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 0000000025D8: BF88FF9E
	s_or_b32 s3, s7, s3                                        // 0000000025DC: 8C030307
	v_cndmask_b32_e32 v90, v90, v138, vcc_lo                   // 0000000025E0: 02B5155A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000025E4: BF88FF9E
	s_xor_b32 s2, s3, s0                                       // 0000000025E8: 8D020003
	v_cmp_neq_f32_e32 vcc_lo, v88, v119                        // 0000000025EC: 7C3AEF58
	v_cmp_ngt_f32_e64 s3, v85, v140                            // 0000000025F0: D41B0003 02031955
	v_cmp_ngt_f32_e64 s6, v88, v119                            // 0000000025F8: D41B0006 0202EF58
	s_wait_alu depctr_sa_sdst(0)                               // 000000002600: BF88FF9E
	v_cndmask_b32_e64 v89, v89, v137, s2                       // 000000002604: D5010059 000B1359
	v_cndmask_b32_e64 v87, v87, v139, s2                       // 00000000260C: D5010057 000B1757
	v_cmp_neq_f32_e64 s4, v84, v120                            // 000000002614: D41D0004 0202F154
	s_or_b32 s2, vcc_lo, s3                                    // 00000000261C: 8C02036A
	v_cmp_ngt_f32_e64 s7, v84, v120                            // 000000002620: D41B0007 0202F154
	s_wait_alu depctr_sa_sdst(0)                               // 000000002628: BF88FF9E
	s_and_b32 s2, s6, s2                                       // 00000000262C: 8B020206
	s_wait_alu depctr_sa_sdst(0)                               // 000000002630: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000002634: 8D6A0002
	s_wait_alu depctr_sa_sdst(0)                               // 000000002638: BF88FF9E
	v_dual_mov_b32 v141, v82 :: v_dual_cndmask_b32 v88, v88, v119// 00000000263C: CA120152 8D58EF58
	v_cndmask_b32_e32 v85, v85, v140, vcc_lo                   // 000000002644: 02AB1955
	v_cmp_eq_f32_e32 vcc_lo, v89, v83                          // 000000002648: 7C24A759
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000264C: BF870093
	v_permlanex16_b32 v141, v141, s10, 0xfedcba98              // 000000002650: D65C008D 03FC158D FEDCBA98
	v_cmp_ngt_f32_e64 s5, v82, v141                            // 00000000265C: D41B0005 02031B52
	s_or_b32 s3, s4, s5                                        // 000000002664: 8C030504
	v_cmp_lt_f32_e64 s4, v90, v87                              // 000000002668: D4110004 0202AF5A
	s_wait_alu depctr_sa_sdst(0)                               // 000000002670: BF88FF9E
	s_and_b32 s3, s7, s3                                       // 000000002674: 8B030307
	v_cmp_lt_f32_e64 s5, v89, v83                              // 000000002678: D4110005 0202A759
	s_wait_alu depctr_sa_sdst(0)                               // 000000002680: BF88FF9E
	s_xor_b32 s2, s3, s0                                       // 000000002684: 8D020003
	v_cmp_lt_f32_e64 s3, v87, v90                              // 000000002688: D4110003 0202B557
	s_wait_alu depctr_sa_sdst(0)                               // 000000002690: BF88FF9E
	v_cndmask_b32_e64 v84, v84, v120, s2                       // 000000002694: D5010054 000AF154
	v_cndmask_b32_e64 v82, v82, v141, s2                       // 00000000269C: D5010052 000B1B52
	v_cmp_lt_f32_e64 s2, v83, v89                              // 0000000026A4: D4110002 0202B353
	s_and_b32 s4, vcc_lo, s4                                   // 0000000026AC: 8B04046A
	s_and_b32 s3, vcc_lo, s3                                   // 0000000026B0: 8B03036A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000026B4: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 0000000026B8: 8C6A0305
	s_or_b32 s2, s2, s4                                        // 0000000026BC: 8C020402
	s_wait_alu depctr_sa_sdst(0)                               // 0000000026C0: BF88FF9E
	v_cndmask_b32_e32 v119, v83, v89, vcc_lo                   // 0000000026C4: 02EEB353
	v_cndmask_b32_e64 v83, v83, v89, s2                        // 0000000026C8: D5010053 000AB353
	v_cndmask_b32_e32 v89, v90, v87, vcc_lo                    // 0000000026D0: 02B2AF5A
	v_cmp_eq_f32_e32 vcc_lo, v84, v88                          // 0000000026D4: 7C24B154
	v_cmp_lt_f32_e64 s3, v82, v85                              // 0000000026D8: D4110003 0202AB52
	v_cmp_lt_f32_e64 s4, v85, v82                              // 0000000026E0: D4110004 0202A555
	v_cmp_lt_f32_e64 s5, v84, v88                              // 0000000026E8: D4110005 0202B154
	v_cndmask_b32_e64 v87, v90, v87, s2                        // 0000000026F0: D5010057 000AAF5A
	v_cmp_lt_f32_e64 s2, v88, v84                              // 0000000026F8: D4110002 0202A958
	s_and_b32 s3, vcc_lo, s3                                   // 000000002700: 8B03036A
	s_and_b32 s4, vcc_lo, s4                                   // 000000002704: 8B04046A
	s_wait_alu depctr_sa_sdst(0)                               // 000000002708: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 00000000270C: 8C6A0305
	s_or_b32 s2, s2, s4                                        // 000000002710: 8C020402
	s_wait_alu depctr_sa_sdst(0)                               // 000000002714: BF88FF9E
	v_cndmask_b32_e32 v90, v84, v88, vcc_lo                    // 000000002718: 02B4B154
	v_cndmask_b32_e64 v84, v84, v88, s2                        // 00000000271C: D5010054 000AB154
	v_cndmask_b32_e32 v88, v82, v85, vcc_lo                    // 000000002724: 02B0AB52
	v_cndmask_b32_e64 v82, v82, v85, s2                        // 000000002728: D5010052 000AAB52
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 000000002730: BF870224
	v_cmp_eq_f32_e32 vcc_lo, v119, v90                         // 000000002734: 7C24B577
	v_cmp_lt_f32_e64 s5, v119, v90                             // 000000002738: D4110005 0202B577
	v_cmp_lt_f32_e64 s3, v89, v88                              // 000000002740: D4110003 0202B159
	v_cmp_eq_f32_e64 s2, v83, v84                              // 000000002748: D4120002 0202A953
	v_cmp_lt_f32_e64 s4, v87, v82                              // 000000002750: D4110004 0202A557
	v_cmp_lt_f32_e64 s6, v83, v84                              // 000000002758: D4110006 0202A953
	s_and_b32 s3, vcc_lo, s3                                   // 000000002760: 8B03036A
	s_wait_alu depctr_sa_sdst(0)                               // 000000002764: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 000000002768: 8C6A0305
	s_and_b32 s2, s2, s4                                       // 00000000276C: 8B020402
	s_wait_alu depctr_sa_sdst(0)                               // 000000002770: BF88FF9E
	v_cndmask_b32_e32 v85, v90, v119, vcc_lo                   // 000000002774: 02AAEF5A
	s_or_b32 s2, s6, s2                                        // 000000002778: 8C020206
	v_cndmask_b32_e32 v90, v119, v90, vcc_lo                   // 00000000277C: 02B4B577
	s_wait_alu depctr_sa_sdst(0)                               // 000000002780: BF88FF9E
	v_cndmask_b32_e64 v120, v84, v83, s2                       // 000000002784: D5010078 000AA754
	v_cndmask_b32_e64 v83, v83, v84, s2                        // 00000000278C: D5010053 000AA953
	v_mov_b32_e32 v119, v85                                    // 000000002794: 7EEE0355
	v_dual_cndmask_b32 v137, v88, v89 :: v_dual_cndmask_b32 v84, v89, v88// 000000002798: CA52B358 8954B159
	v_cndmask_b32_e64 v138, v82, v87, s2                       // 0000000027A0: D501008A 000AAF52
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)// 0000000027A8: BF870194
	v_dual_mov_b32 v139, v120 :: v_dual_mov_b32 v88, v83       // 0000000027AC: CA100178 8B580153
	v_dual_mov_b32 v140, v137 :: v_dual_mov_b32 v89, v84       // 0000000027B4: CA100189 8C580154
	v_permlanex16_b32 v119, v119, s10, 0xfedcba98              // 0000000027BC: D65C0077 03FC1577 FEDCBA98
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 0000000027C8: BF870223
	v_permlanex16_b32 v139, v139, s10, 0xfedcba98              // 0000000027CC: D65C008B 03FC158B FEDCBA98
	v_cndmask_b32_e64 v82, v87, v82, s2                        // 0000000027D8: D5010052 000AA557
	v_permlanex16_b32 v140, v140, s10, 0xfedcba98              // 0000000027E0: D65C008C 03FC158C FEDCBA98
	v_permlanex16_b32 v89, v89, s10, 0xfedcba98                // 0000000027EC: D65C0059 03FC1559 FEDCBA98
	v_mov_b32_e32 v141, v138                                   // 0000000027F8: 7F1A038A
	v_cmp_eq_f32_e32 vcc_lo, v85, v119                         // 0000000027FC: 7C24EF55
	v_cmp_eq_f32_e64 s3, v120, v139                            // 000000002800: D4120003 02031778
	v_cmp_gt_f32_e64 s4, v137, v140                            // 000000002808: D4140004 02031989
	v_cmp_gt_f32_e64 s6, v85, v119                             // 000000002810: D4140006 0202EF55
	v_permlanex16_b32 v141, v141, s10, 0xfedcba98              // 000000002818: D65C008D 03FC158D FEDCBA98
	v_cmp_gt_f32_e64 s7, v120, v139                            // 000000002824: D4140007 02031778
	v_permlanex16_b32 v88, v88, s10, 0xfedcba98                // 00000000282C: D65C0058 03FC1558 FEDCBA98
	s_and_b32 s2, vcc_lo, s4                                   // 000000002838: 8B02046A
	v_cmp_gt_f32_e64 s4, v84, v89                              // 00000000283C: D4140004 0202B354
	v_cmp_gt_f32_e64 s5, v138, v141                            // 000000002844: D4140005 02031B8A
	s_wait_alu depctr_sa_sdst(0)                               // 00000000284C: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 000000002850: 8C020206
	s_wait_alu depctr_sa_sdst(0)                               // 000000002854: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000002858: 8D6A0002
	s_and_b32 s3, s3, s5                                       // 00000000285C: 8B030503
	s_wait_alu depctr_sa_sdst(0)                               // 000000002860: BF88FF9E
	v_cndmask_b32_e32 v85, v85, v119, vcc_lo                   // 000000002864: 02AAEF55
	s_or_b32 s3, s7, s3                                        // 000000002868: 8C030307
	v_mov_b32_e32 v119, v82                                    // 00000000286C: 7EEE0352
	s_wait_alu depctr_sa_sdst(0)                               // 000000002870: BF88FF9E
	s_xor_b32 s2, s3, s0                                       // 000000002874: 8D020003
	v_cndmask_b32_e32 v137, v137, v140, vcc_lo                 // 000000002878: 03131989
	s_wait_alu depctr_sa_sdst(0)                               // 00000000287C: BF88FF9E
	v_cndmask_b32_e64 v120, v120, v139, s2                     // 000000002880: D5010078 000B1778
	v_mov_b32_e32 v87, v90                                     // 000000002888: 7EAE035A
	v_permlanex16_b32 v119, v119, s10, 0xfedcba98              // 00000000288C: D65C0077 03FC1577 FEDCBA98
	v_cmp_eq_f32_e64 s3, v83, v88                              // 000000002898: D4120003 0202B153
	v_cmp_gt_f32_e64 s7, v83, v88                              // 0000000028A0: D4140007 0202B153
	v_cndmask_b32_e64 v138, v138, v141, s2                     // 0000000028A8: D501008A 000B1B8A
	v_permlanex16_b32 v87, v87, s10, 0xfedcba98                // 0000000028B0: D65C0057 03FC1557 FEDCBA98
	v_cmp_gt_f32_e64 s5, v82, v119                             // 0000000028BC: D4140005 0202EF52
	s_delay_alu instid0(VALU_DEP_2)                            // 0000000028C4: BF870002
	v_cmp_eq_f32_e32 vcc_lo, v90, v87                          // 0000000028C8: 7C24AF5A
	v_cmp_gt_f32_e64 s6, v90, v87                              // 0000000028CC: D4140006 0202AF5A
	s_and_b32 s3, s3, s5                                       // 0000000028D4: 8B030503
	v_cmp_lt_f32_e64 s5, v120, v85                             // 0000000028D8: D4110005 0202AB78
	s_wait_alu depctr_sa_sdst(0)                               // 0000000028E0: BF88FF9E
	s_or_b32 s3, s7, s3                                        // 0000000028E4: 8C030307
	s_and_b32 s2, vcc_lo, s4                                   // 0000000028E8: 8B02046A
	v_cmp_lt_f32_e64 s4, v137, v138                            // 0000000028EC: D4110004 02031589
	s_wait_alu depctr_sa_sdst(0)                               // 0000000028F4: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 0000000028F8: 8C020206
	s_wait_alu depctr_sa_sdst(0)                               // 0000000028FC: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000002900: 8D6A0002
	s_xor_b32 s2, s3, s0                                       // 000000002904: 8D020003
	s_wait_alu depctr_sa_sdst(0)                               // 000000002908: BF88FF9E
	v_dual_cndmask_b32 v87, v90, v87 :: v_dual_cndmask_b32 v84, v84, v89// 00000000290C: CA52AF5A 5754B354
	v_cmp_eq_f32_e32 vcc_lo, v120, v85                         // 000000002914: 7C24AB78
	v_cmp_lt_f32_e64 s3, v138, v137                            // 000000002918: D4110003 0203138A
	v_cndmask_b32_e64 v83, v83, v88, s2                        // 000000002920: D5010053 000AB153
	v_cndmask_b32_e64 v82, v82, v119, s2                       // 000000002928: D5010052 000AEF52
	v_cmp_lt_f32_e64 s2, v85, v120                             // 000000002930: D4110002 0202F155
	s_and_b32 s4, vcc_lo, s4                                   // 000000002938: 8B04046A
	s_and_b32 s3, vcc_lo, s3                                   // 00000000293C: 8B03036A
	s_wait_alu depctr_sa_sdst(0)                               // 000000002940: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 000000002944: 8C6A0305
	s_or_b32 s2, s2, s4                                        // 000000002948: 8C020402
	s_wait_alu depctr_sa_sdst(0)                               // 00000000294C: BF88FF9E
	v_cndmask_b32_e32 v88, v85, v120, vcc_lo                   // 000000002950: 02B0F155
	v_cndmask_b32_e32 v89, v137, v138, vcc_lo                  // 000000002954: 02B31589
	v_cmp_eq_f32_e32 vcc_lo, v83, v87                          // 000000002958: 7C24AF53
	v_cmp_lt_f32_e64 s3, v82, v84                              // 00000000295C: D4110003 0202A952
	v_cmp_lt_f32_e64 s4, v84, v82                              // 000000002964: D4110004 0202A554
	v_cndmask_b32_e64 v85, v85, v120, s2                       // 00000000296C: D5010055 000AF155
	v_cmp_lt_f32_e64 s5, v83, v87                              // 000000002974: D4110005 0202AF53
	v_cndmask_b32_e64 v90, v137, v138, s2                      // 00000000297C: D501005A 000B1589
	v_cmp_lt_f32_e64 s2, v87, v83                              // 000000002984: D4110002 0202A757
	s_and_b32 s3, vcc_lo, s3                                   // 00000000298C: 8B03036A
	s_and_b32 s4, vcc_lo, s4                                   // 000000002990: 8B04046A
	s_wait_alu depctr_sa_sdst(0)                               // 000000002994: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 000000002998: 8C6A0305
	v_permlanex16_b32 v85, v85, s10, 0xfedcba98                // 00000000299C: D65C0055 03FC1555 FEDCBA98
	s_or_b32 s2, s2, s4                                        // 0000000029A8: 8C020402
	s_wait_alu depctr_sa_sdst(0)                               // 0000000029AC: BF88FF9E
	v_cndmask_b32_e32 v119, v87, v83, vcc_lo                   // 0000000029B0: 02EEA757
	v_cndmask_b32_e64 v83, v87, v83, s2                        // 0000000029B4: D5010053 000AA757
	v_permlanex16_b32 v88, v88, s10, 0xfedcba98                // 0000000029BC: D65C0058 03FC1558 FEDCBA98
	v_permlanex16_b32 v90, v90, s10, 0xfedcba98                // 0000000029C8: D65C005A 03FC155A FEDCBA98
	v_permlanex16_b32 v89, v89, s10, 0xfedcba98                // 0000000029D4: D65C0059 03FC1559 FEDCBA98
	v_cndmask_b32_e32 v87, v84, v82, vcc_lo                    // 0000000029E0: 02AEA554
	v_cndmask_b32_e64 v82, v84, v82, s2                        // 0000000029E4: D5010052 000AA554
	v_cmp_eq_f32_e32 vcc_lo, v81, v85                          // 0000000029EC: 7C24AB51
	v_cmp_eq_f32_e64 s3, v75, v88                              // 0000000029F0: D4120003 0202B14B
	v_cmp_lt_f32_e64 s4, v80, v90                              // 0000000029F8: D4110004 0202B550
	v_cmp_lt_f32_e64 s5, v74, v89                              // 000000002A00: D4110005 0202B34A
	v_permlanex16_b32 v83, v83, s10, 0xfedcba98                // 000000002A08: D65C0053 03FC1553 FEDCBA98
	v_permlanex16_b32 v82, v82, s10, 0xfedcba98                // 000000002A14: D65C0052 03FC1552 FEDCBA98
	v_cmp_lt_f32_e64 s2, v81, v85                              // 000000002A20: D4110002 0202AB51
	v_permlanex16_b32 v119, v119, s10, 0xfedcba98              // 000000002A28: D65C0077 03FC1577 FEDCBA98
	v_permlanex16_b32 v87, v87, s10, 0xfedcba98                // 000000002A34: D65C0057 03FC1557 FEDCBA98
	s_and_b32 s4, vcc_lo, s4                                   // 000000002A40: 8B04046A
	s_and_b32 s8, s3, s5                                       // 000000002A44: 8B080503
	v_cmp_eq_f32_e64 s3, v76, v83                              // 000000002A48: D4120003 0202A74C
	v_cmp_lt_f32_e64 s5, v77, v82                              // 000000002A50: D4110005 0202A54D
	v_cmp_lt_f32_e64 s6, v75, v88                              // 000000002A58: D4110006 0202B14B
	s_wait_alu depctr_sa_sdst(0)                               // 000000002A60: BF88FF9E
	s_or_b32 vcc_lo, s2, s4                                    // 000000002A64: 8C6A0402
	v_cmp_lt_f32_e64 s2, v76, v83                              // 000000002A68: D4110002 0202A74C
	v_cmp_eq_f32_e64 s4, v78, v119                             // 000000002A70: D4120004 0202EF4E
	v_cmp_lt_f32_e64 s7, v79, v87                              // 000000002A78: D4110007 0202AF4F
	s_and_b32 s3, s3, s5                                       // 000000002A80: 8B030503
	s_or_b32 s6, s6, s8                                        // 000000002A84: 8C060806
	v_cmp_lt_f32_e64 s8, v78, v119                             // 000000002A88: D4110008 0202EF4E
	s_wait_alu depctr_sa_sdst(0)                               // 000000002A90: BF88FF9E
	s_or_b32 s2, s2, s3                                        // 000000002A94: 8C020302
	s_and_b32 s4, s4, s7                                       // 000000002A98: 8B040704
	s_wait_alu depctr_sa_sdst(0)                               // 000000002A9C: BF88FF9E
	v_cndmask_b32_e64 v76, v83, v76, s2                        // 000000002AA0: D501004C 000A9953
	v_cndmask_b32_e64 v77, v82, v77, s2                        // 000000002AA8: D501004D 000A9B52
	s_or_b32 s3, s8, s4                                        // 000000002AB0: 8C030408
	v_cndmask_b32_e32 v81, v85, v81, vcc_lo                    // 000000002AB4: 02A2A355
	s_wait_alu depctr_sa_sdst(0)                               // 000000002AB8: BF88FF9E
	v_cndmask_b32_e64 v78, v119, v78, s3                       // 000000002ABC: D501004E 000E9D77
	v_cndmask_b32_e64 v79, v87, v79, s3                        // 000000002AC4: D501004F 000E9F57
	v_cndmask_b32_e32 v80, v90, v80, vcc_lo                    // 000000002ACC: 02A0A15A
	v_cmp_eq_f32_e32 vcc_lo, v68, v76                          // 000000002AD0: 7C249944
	v_cmp_lt_f32_e64 s3, v67, v77                              // 000000002AD4: D4110003 02029B43
	v_cmp_lt_f32_e64 s5, v68, v76                              // 000000002ADC: D4110005 02029944
	v_cmp_eq_f32_e64 s2, v69, v78                              // 000000002AE4: D4120002 02029D45
	v_cmp_lt_f32_e64 s4, v66, v79                              // 000000002AEC: D4110004 02029F42
	v_cndmask_b32_e64 v75, v88, v75, s6                        // 000000002AF4: D501004B 001A9758
	s_and_b32 s3, vcc_lo, s3                                   // 000000002AFC: 8B03036A
	v_cndmask_b32_e64 v74, v89, v74, s6                        // 000000002B00: D501004A 001A9559
	v_cmp_lt_f32_e64 s6, v69, v78                              // 000000002B08: D4110006 02029D45
	s_wait_alu depctr_sa_sdst(0)                               // 000000002B10: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 000000002B14: 8C6A0305
	s_and_b32 s2, s2, s4                                       // 000000002B18: 8B020402
	s_wait_alu depctr_sa_sdst(0)                               // 000000002B1C: BF88FF9E
	v_cndmask_b32_e32 v82, v76, v68, vcc_lo                    // 000000002B20: 02A4894C
	v_cndmask_b32_e32 v84, v77, v67, vcc_lo                    // 000000002B24: 02A8874D
	v_dual_cndmask_b32 v68, v68, v76 :: v_dual_cndmask_b32 v67, v67, v77// 000000002B28: CA529944 44429B43
	v_cmp_eq_f32_e32 vcc_lo, v71, v81                          // 000000002B30: 7C24A347
	v_cmp_lt_f32_e64 s4, v72, v80                              // 000000002B34: D4110004 0202A148
	s_or_b32 s2, s6, s2                                        // 000000002B3C: 8C020206
	v_cmp_lt_f32_e64 s6, v71, v81                              // 000000002B40: D4110006 0202A347
	s_wait_alu depctr_sa_sdst(0)                               // 000000002B48: BF88FF9E
	v_cndmask_b32_e64 v83, v78, v69, s2                        // 000000002B4C: D5010053 000A8B4E
	v_cndmask_b32_e64 v69, v69, v78, s2                        // 000000002B54: D5010045 000A9D45
	s_and_b32 s4, vcc_lo, s4                                   // 000000002B5C: 8B04046A
	v_cmp_eq_f32_e64 s3, v73, v75                              // 000000002B60: D4120003 02029749
	s_wait_alu depctr_sa_sdst(0)                               // 000000002B68: BF88FF9E
	s_or_b32 vcc_lo, s6, s4                                    // 000000002B6C: 8C6A0406
	v_cmp_lt_f32_e64 s5, v70, v74                              // 000000002B70: D4110005 02029546
	s_wait_alu depctr_sa_sdst(0)                               // 000000002B78: BF88FF9E
	v_cndmask_b32_e32 v76, v81, v71, vcc_lo                    // 000000002B7C: 02988F51
	v_cndmask_b32_e32 v78, v80, v72, vcc_lo                    // 000000002B80: 029C9150
	v_cndmask_b32_e64 v85, v79, v66, s2                        // 000000002B84: D5010055 000A854F
	v_cndmask_b32_e64 v66, v66, v79, s2                        // 000000002B8C: D5010042 000A9F42
	v_cmp_lt_f32_e64 s2, v73, v75                              // 000000002B94: D4110002 02029749
	s_and_b32 s3, s3, s5                                       // 000000002B9C: 8B030503
	v_dual_cndmask_b32 v71, v71, v81 :: v_dual_cndmask_b32 v72, v72, v80// 000000002BA0: CA52A347 4748A148
	v_cmp_eq_f32_e32 vcc_lo, v82, v76                          // 000000002BA8: 7C249952
	v_cmp_lt_f32_e64 s4, v84, v78                              // 000000002BAC: D4110004 02029D54
	s_wait_alu depctr_sa_sdst(0)                               // 000000002BB4: BF88FF9E
	s_or_b32 s2, s2, s3                                        // 000000002BB8: 8C020302
	v_cmp_lt_f32_e64 s6, v82, v76                              // 000000002BBC: D4110006 02029952
	s_wait_alu depctr_sa_sdst(0)                               // 000000002BC4: BF88FF9E
	v_cndmask_b32_e64 v77, v75, v73, s2                        // 000000002BC8: D501004D 000A934B
	v_cndmask_b32_e64 v79, v74, v70, s2                        // 000000002BD0: D501004F 000A8D4A
	s_and_b32 s4, vcc_lo, s4                                   // 000000002BD8: 8B04046A
	v_cndmask_b32_e64 v70, v70, v74, s2                        // 000000002BDC: D5010046 000A9546
	s_wait_alu depctr_sa_sdst(0)                               // 000000002BE4: BF88FF9E
	s_or_b32 vcc_lo, s6, s4                                    // 000000002BE8: 8C6A0406
	v_cmp_eq_f32_e64 s3, v83, v77                              // 000000002BEC: D4120003 02029B53
	v_cmp_lt_f32_e64 s5, v85, v79                              // 000000002BF4: D4110005 02029F55
	s_wait_alu depctr_sa_sdst(0)                               // 000000002BFC: BF88FF9E
	v_cndmask_b32_e32 v74, v76, v82, vcc_lo                    // 000000002C00: 0294A54C
	v_cndmask_b32_e32 v80, v78, v84, vcc_lo                    // 000000002C04: 02A0A94E
	v_cndmask_b32_e32 v76, v82, v76, vcc_lo                    // 000000002C08: 02989952
	v_cndmask_b32_e32 v78, v84, v78, vcc_lo                    // 000000002C0C: 029C9D54
	v_cmp_eq_f32_e32 vcc_lo, v68, v71                          // 000000002C10: 7C248F44
	v_cmp_lt_f32_e64 s4, v67, v72                              // 000000002C14: D4110004 02029143
	v_cndmask_b32_e64 v73, v73, v75, s2                        // 000000002C1C: D5010049 000A9749
	v_cmp_lt_f32_e64 s2, v83, v77                              // 000000002C24: D4110002 02029B53
	v_cmp_lt_f32_e64 s6, v68, v71                              // 000000002C2C: D4110006 02028F44
	s_and_b32 s3, s3, s5                                       // 000000002C34: 8B030503
	s_and_b32 s4, vcc_lo, s4                                   // 000000002C38: 8B04046A
	v_cmp_lt_f32_e64 s5, v66, v70                              // 000000002C3C: D4110005 02028D42
	s_wait_alu depctr_sa_sdst(0)                               // 000000002C44: BF88FF9E
	s_or_b32 s2, s2, s3                                        // 000000002C48: 8C020302
	s_or_b32 vcc_lo, s6, s4                                    // 000000002C4C: 8C6A0406
	s_wait_alu depctr_sa_sdst(0)                               // 000000002C50: BF88FF9E
	v_cndmask_b32_e64 v81, v79, v85, s2                        // 000000002C54: D5010051 000AAB4F
	v_cndmask_b32_e32 v82, v71, v68, vcc_lo                    // 000000002C5C: 02A48947
	v_dual_cndmask_b32 v68, v68, v71 :: v_dual_mov_b32 v71, v74// 000000002C60: CA508F44 4446014A
	v_mov_b32_e32 v88, v80                                     // 000000002C68: 7EB00350
	v_cndmask_b32_e64 v75, v77, v83, s2                        // 000000002C6C: D501004B 000AA74D
	v_dual_cndmask_b32 v84, v72, v67 :: v_dual_cndmask_b32 v67, v67, v72// 000000002C74: CA528748 54429143
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000002C7C: BF870214
	v_permlanex16_b32 v71, v71, s10, 0xfedcba98                // 000000002C80: D65C0047 03FC1547 FEDCBA98
	v_permlanex16_b32 v88, v88, s10, 0xfedcba98                // 000000002C8C: D65C0058 03FC1558 FEDCBA98
	v_cmp_eq_f32_e64 s3, v69, v73                              // 000000002C98: D4120003 02029345
	v_cndmask_b32_e64 v77, v83, v77, s2                        // 000000002CA0: D501004D 000A9B53
	v_cndmask_b32_e64 v79, v85, v79, s2                        // 000000002CA8: D501004F 000A9F55
	v_cmp_lt_f32_e64 s2, v69, v73                              // 000000002CB0: D4110002 02029345
	v_cmp_eq_f32_e32 vcc_lo, v74, v71                          // 000000002CB8: 7C248F4A
	v_cmp_gt_f32_e64 s4, v80, v88                              // 000000002CBC: D4140004 0202B150
	s_and_b32 s3, s3, s5                                       // 000000002CC4: 8B030503
	v_cmp_gt_f32_e64 s6, v74, v71                              // 000000002CC8: D4140006 02028F4A
	s_wait_alu depctr_sa_sdst(0)                               // 000000002CD0: BF88FF9E
	s_or_b32 s2, s2, s3                                        // 000000002CD4: 8C020302
	v_mov_b32_e32 v87, v75                                     // 000000002CD8: 7EAE034B
	s_wait_alu depctr_sa_sdst(0)                               // 000000002CDC: BF88FF9E
	v_cndmask_b32_e64 v83, v73, v69, s2                        // 000000002CE0: D5010053 000A8B49
	v_cndmask_b32_e64 v85, v70, v66, s2                        // 000000002CE8: D5010055 000A8546
	v_mov_b32_e32 v89, v81                                     // 000000002CF0: 7EB20351
	v_cndmask_b32_e64 v69, v69, v73, s2                        // 000000002CF4: D5010045 000A9345
	v_cndmask_b32_e64 v66, v66, v70, s2                        // 000000002CFC: D5010042 000A8D42
	s_and_b32 s2, vcc_lo, s4                                   // 000000002D04: 8B02046A
	v_permlanex16_b32 v87, v87, s10, 0xfedcba98                // 000000002D08: D65C0057 03FC1557 FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 000000002D14: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 000000002D18: 8C020206
	v_permlanex16_b32 v89, v89, s10, 0xfedcba98                // 000000002D1C: D65C0059 03FC1559 FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 000000002D28: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000002D2C: 8D6A0002
	v_dual_mov_b32 v72, v77 :: v_dual_mov_b32 v73, v78         // 000000002D30: CA10014D 4848014E
	s_wait_alu depctr_sa_sdst(0)                               // 000000002D38: BF88FF9E
	v_dual_cndmask_b32 v70, v74, v71 :: v_dual_mov_b32 v71, v76// 000000002D3C: CA508F4A 4646014C
	v_cmp_eq_f32_e64 s3, v75, v87                              // 000000002D44: D4120003 0202AF4B
	v_cmp_gt_f32_e64 s5, v81, v89                              // 000000002D4C: D4140005 0202B351
	v_permlanex16_b32 v73, v73, s10, 0xfedcba98                // 000000002D54: D65C0049 03FC1549 FEDCBA98
	v_cmp_gt_f32_e64 s7, v75, v87                              // 000000002D60: D4140007 0202AF4B
	v_permlanex16_b32 v71, v71, s10, 0xfedcba98                // 000000002D68: D65C0047 03FC1547 FEDCBA98
	v_cndmask_b32_e32 v80, v80, v88, vcc_lo                    // 000000002D74: 02A0B150
	s_and_b32 s3, s3, s5                                       // 000000002D78: 8B030503
	v_cmp_gt_f32_e64 s4, v78, v73                              // 000000002D7C: D4140004 0202934E
	s_wait_alu depctr_sa_sdst(0)                               // 000000002D84: BF88FF9E
	s_or_b32 s3, s7, s3                                        // 000000002D88: 8C030307
	v_cmp_eq_f32_e32 vcc_lo, v76, v71                          // 000000002D8C: 7C248F4C
	v_cmp_gt_f32_e64 s6, v76, v71                              // 000000002D90: D4140006 02028F4C
	s_wait_alu depctr_sa_sdst(0)                               // 000000002D98: BF88FF9E
	s_xor_b32 s2, s3, s0                                       // 000000002D9C: 8D020003
	v_mov_b32_e32 v74, v79                                     // 000000002DA0: 7E94034F
	s_wait_alu depctr_sa_sdst(0)                               // 000000002DA4: BF88FF9E
	v_cndmask_b32_e64 v75, v75, v87, s2                        // 000000002DA8: D501004B 000AAF4B
	v_cndmask_b32_e64 v81, v81, v89, s2                        // 000000002DB0: D5010051 000AB351
	s_and_b32 s2, vcc_lo, s4                                   // 000000002DB8: 8B02046A
	v_permlanex16_b32 v72, v72, s10, 0xfedcba98                // 000000002DBC: D65C0048 03FC1548 FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 000000002DC8: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 000000002DCC: 8C020206
	v_mov_b32_e32 v87, v83                                     // 000000002DD0: 7EAE0353
	s_wait_alu depctr_sa_sdst(0)                               // 000000002DD4: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000002DD8: 8D6A0002
	v_mov_b32_e32 v88, v84                                     // 000000002DDC: 7EB00354
	s_wait_alu depctr_sa_sdst(0)                               // 000000002DE0: BF88FF9E
	v_dual_cndmask_b32 v76, v76, v71 :: v_dual_mov_b32 v89, v85// 000000002DE4: CA508F4C 4C580155
	v_mov_b32_e32 v71, v82                                     // 000000002DEC: 7E8E0352
	v_permlanex16_b32 v74, v74, s10, 0xfedcba98                // 000000002DF0: D65C004A 03FC154A FEDCBA98
	v_cmp_eq_f32_e64 s3, v77, v72                              // 000000002DFC: D4120003 0202914D
	v_cmp_gt_f32_e64 s7, v77, v72                              // 000000002E04: D4140007 0202914D
	v_permlanex16_b32 v87, v87, s10, 0xfedcba98                // 000000002E0C: D65C0057 03FC1557 FEDCBA98
	v_permlanex16_b32 v71, v71, s10, 0xfedcba98                // 000000002E18: D65C0047 03FC1547 FEDCBA98
	v_cmp_gt_f32_e64 s5, v79, v74                              // 000000002E24: D4140005 0202954F
	v_permlanex16_b32 v89, v89, s10, 0xfedcba98                // 000000002E2C: D65C0059 03FC1559 FEDCBA98
	s_delay_alu instid0(VALU_DEP_3)                            // 000000002E38: BF870003
	v_cmp_gt_f32_e64 s6, v82, v71                              // 000000002E3C: D4140006 02028F52
	s_and_b32 s3, s3, s5                                       // 000000002E44: 8B030503
	v_cmp_gt_f32_e64 s5, v85, v89                              // 000000002E48: D4140005 0202B355
	s_wait_alu depctr_sa_sdst(0)                               // 000000002E50: BF88FF9E
	s_or_b32 s3, s7, s3                                        // 000000002E54: 8C030307
	v_cmp_gt_f32_e64 s7, v83, v87                              // 000000002E58: D4140007 0202AF53
	s_wait_alu depctr_sa_sdst(0)                               // 000000002E60: BF88FF9E
	s_xor_b32 s2, s3, s0                                       // 000000002E64: 8D020003
	v_cmp_eq_f32_e64 s3, v83, v87                              // 000000002E68: D4120003 0202AF53
	s_wait_alu depctr_sa_sdst(0)                               // 000000002E70: BF88FF9E
	v_cndmask_b32_e64 v72, v77, v72, s2                        // 000000002E74: D5010048 000A914D
	v_cndmask_b32_e32 v77, v78, v73, vcc_lo                    // 000000002E7C: 029A934E
	v_permlanex16_b32 v88, v88, s10, 0xfedcba98                // 000000002E80: D65C0058 03FC1558 FEDCBA98
	v_cmp_eq_f32_e32 vcc_lo, v82, v71                          // 000000002E8C: 7C248F52
	v_cndmask_b32_e64 v78, v79, v74, s2                        // 000000002E90: D501004E 000A954F
	v_mov_b32_e32 v74, v67                                     // 000000002E98: 7E940343
	s_and_b32 s3, s3, s5                                       // 000000002E9C: 8B030503
	v_cmp_gt_f32_e64 s4, v84, v88                              // 000000002EA0: D4140004 0202B154
	s_wait_alu depctr_sa_sdst(0)                               // 000000002EA8: BF88FF9E
	s_or_b32 s3, s7, s3                                        // 000000002EAC: 8C030307
	v_permlanex16_b32 v74, v74, s10, 0xfedcba98                // 000000002EB0: D65C004A 03FC154A FEDCBA98
	s_and_b32 s2, vcc_lo, s4                                   // 000000002EBC: 8B02046A
	s_wait_alu depctr_sa_sdst(0)                               // 000000002EC0: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 000000002EC4: 8C020206
	s_delay_alu instid0(VALU_DEP_1)                            // 000000002EC8: BF870001
	v_cmp_gt_f32_e64 s4, v67, v74                              // 000000002ECC: D4140004 02029543
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ED4: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000002ED8: 8D6A0002
	s_wait_alu depctr_sa_sdst(0)                               // 000000002EDC: BF88FF9E
	v_dual_mov_b32 v73, v69 :: v_dual_cndmask_b32 v84, v84, v88// 000000002EE0: CA120145 4954B154
	v_cndmask_b32_e32 v79, v82, v71, vcc_lo                    // 000000002EE8: 029E8F52
	v_dual_mov_b32 v71, v68 :: v_dual_mov_b32 v82, v66         // 000000002EEC: CA100144 47520142
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000002EF4: BF870123
	v_permlanex16_b32 v73, v73, s10, 0xfedcba98                // 000000002EF8: D65C0049 03FC1549 FEDCBA98
	s_xor_b32 s2, s3, s0                                       // 000000002F04: 8D020003
	v_permlanex16_b32 v71, v71, s10, 0xfedcba98                // 000000002F08: D65C0047 03FC1547 FEDCBA98
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000002F14: BF870193
	v_permlanex16_b32 v82, v82, s10, 0xfedcba98                // 000000002F18: D65C0052 03FC1552 FEDCBA98
	v_cmp_eq_f32_e64 s3, v69, v73                              // 000000002F24: D4120003 02029345
	v_cmp_gt_f32_e64 s7, v69, v73                              // 000000002F2C: D4140007 02029345
	s_wait_alu depctr_sa_sdst(0)                               // 000000002F34: BF88FF9E
	v_cndmask_b32_e64 v83, v83, v87, s2                        // 000000002F38: D5010053 000AAF53
	v_cmp_eq_f32_e32 vcc_lo, v68, v71                          // 000000002F40: 7C248F44
	v_cmp_gt_f32_e64 s5, v66, v82                              // 000000002F44: D4140005 0202A542
	v_cmp_gt_f32_e64 s6, v68, v71                              // 000000002F4C: D4140006 02028F44
	v_cndmask_b32_e64 v85, v85, v89, s2                        // 000000002F54: D5010055 000AB355
	s_and_b32 s2, vcc_lo, s4                                   // 000000002F5C: 8B02046A
	s_and_b32 s3, s3, s5                                       // 000000002F60: 8B030503
	s_wait_alu depctr_sa_sdst(0)                               // 000000002F64: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 000000002F68: 8C020206
	s_or_b32 s3, s7, s3                                        // 000000002F6C: 8C030307
	s_wait_alu depctr_sa_sdst(0)                               // 000000002F70: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000002F74: 8D6A0002
	s_xor_b32 s2, s3, s0                                       // 000000002F78: 8D020003
	s_wait_alu depctr_sa_sdst(0)                               // 000000002F7C: BF88FF9E
	v_cndmask_b32_e32 v87, v68, v71, vcc_lo                    // 000000002F80: 02AE8F44
	v_cndmask_b32_e32 v89, v67, v74, vcc_lo                    // 000000002F84: 02B29543
	v_cmp_eq_f32_e32 vcc_lo, v75, v70                          // 000000002F88: 7C248D4B
	v_cmp_lt_f32_e64 s4, v80, v81                              // 000000002F8C: D4110004 0202A350
	v_cndmask_b32_e64 v88, v69, v73, s2                        // 000000002F94: D5010058 000A9345
	v_cmp_lt_f32_e64 s3, v81, v80                              // 000000002F9C: D4110003 0202A151
	v_cndmask_b32_e64 v82, v66, v82, s2                        // 000000002FA4: D5010052 000AA542
	v_cmp_lt_f32_e64 s2, v70, v75                              // 000000002FAC: D4110002 02029746
	v_cmp_lt_f32_e64 s5, v75, v70                              // 000000002FB4: D4110005 02028D4B
	s_and_b32 s4, vcc_lo, s4                                   // 000000002FBC: 8B04046A
	s_and_b32 s3, vcc_lo, s3                                   // 000000002FC0: 8B03036A
	v_cmp_lt_f32_e64 s7, v76, v72                              // 000000002FC4: D4110007 0202914C
	s_wait_alu depctr_sa_sdst(0)                               // 000000002FCC: BF88FF9E
	s_or_b32 s2, s2, s4                                        // 000000002FD0: 8C020402
	s_or_b32 vcc_lo, s5, s3                                    // 000000002FD4: 8C6A0305
	s_wait_alu depctr_sa_sdst(0)                               // 000000002FD8: BF88FF9E
	v_cndmask_b32_e64 v69, v70, v75, s2                        // 000000002FDC: D5010045 000A9746
	v_cndmask_b32_e64 v66, v80, v81, s2                        // 000000002FE4: D5010042 000AA350
	v_cmp_eq_f32_e64 s2, v72, v76                              // 000000002FEC: D4120002 02029948
	v_cmp_lt_f32_e64 s3, v78, v77                              // 000000002FF4: D4110003 02029B4E
	v_cndmask_b32_e32 v68, v70, v75, vcc_lo                    // 000000002FFC: 02889746
	v_cmp_lt_f32_e64 s4, v77, v78                              // 000000003000: D4110004 02029D4D
	v_cmp_eq_f32_e64 s5, v88, v87                              // 000000003008: D4120005 0202AF58
	v_cmp_lt_f32_e64 s6, v89, v82                              // 000000003010: D4110006 0202A559
	s_and_b32 s8, s2, s3                                       // 000000003018: 8B080302
	v_cmp_lt_f32_e64 s3, v87, v88                              // 00000000301C: D4110003 0202B157
	s_and_b32 s4, s2, s4                                       // 000000003024: 8B040402
	s_and_b32 s2, s5, s6                                       // 000000003028: 8B020605
	v_cmp_lt_f32_e64 s6, v79, v83                              // 00000000302C: D4110006 0202A74F
	s_wait_alu depctr_sa_sdst(0)                               // 000000003034: BF88FF9E
	s_or_b32 s2, s3, s2                                        // 000000003038: 8C020203
	v_cmp_lt_f32_e64 s3, v85, v84                              // 00000000303C: D4110003 0202A955
	s_wait_alu depctr_sa_sdst(0)                               // 000000003044: BF88FF9E
	v_cndmask_b32_e64 v75, v87, v88, s2                        // 000000003048: D501004B 000AB157
	v_cndmask_b32_e32 v67, v80, v81, vcc_lo                    // 000000003050: 0286A350
	v_cmp_lt_f32_e32 vcc_lo, v72, v76                          // 000000003054: 7C229948
	v_cndmask_b32_e64 v74, v89, v82, s2                        // 000000003058: D501004A 000AA559
	s_or_b32 s2, s7, s4                                        // 000000003060: 8C020407
	v_cmp_lt_f32_e64 s4, v84, v85                              // 000000003064: D4110004 0202AB54
	s_wait_alu depctr_sa_sdst(0)                               // 00000000306C: BF88FF9E
	v_cndmask_b32_e64 v73, v76, v72, s2                        // 000000003070: D5010049 000A914C
	s_or_b32 vcc_lo, vcc_lo, s8                                // 000000003078: 8C6A086A
	v_cndmask_b32_e64 v70, v77, v78, s2                        // 00000000307C: D5010046 000A9D4D
	s_wait_alu depctr_sa_sdst(0)                               // 000000003084: BF88FF9E
	v_dual_cndmask_b32 v71, v76, v72 :: v_dual_max_num_f32 v76, v75, v75// 000000003088: CA54914C 474C974B
	v_cndmask_b32_e32 v72, v77, v78, vcc_lo                    // 000000003090: 02909D4D
	v_max_num_f32_e32 v77, v74, v74                            // 000000003094: 2C9A954A
	v_cmp_eq_f32_e64 s2, v83, v79                              // 000000003098: D4120002 02029F53
	v_cmp_lt_f32_e32 vcc_lo, v83, v79                          // 0000000030A0: 7C229F53
	v_max_num_f32_e32 v76, 0xff800000, v76                     // 0000000030A4: 2C9898FF FF800000
	s_and_b32 s3, s2, s3                                       // 0000000030AC: 8B030302
	s_and_b32 s2, s2, s4                                       // 0000000030B0: 8B020402
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000030B4: BF870001
	v_cndmask_b32_e64 v90, v76, 0xff800000, s1                 // 0000000030B8: D501005A 0005FF4C FF800000
	v_max_num_f32_e32 v77, 0xff800000, v77                     // 0000000030C4: 2C9A9AFF FF800000
	s_wait_alu depctr_sa_sdst(0)                               // 0000000030CC: BF88FF9E
	s_or_b32 vcc_lo, vcc_lo, s3                                // 0000000030D0: 8C6A036A
	v_cmp_lt_f32_e64 s3, v88, v87                              // 0000000030D4: D4110003 0202AF58
	s_wait_alu depctr_sa_sdst(0)                               // 0000000030DC: BF88FF9E
	v_cndmask_b32_e32 v76, v79, v83, vcc_lo                    // 0000000030E0: 0298A74F
	v_mov_b32_e32 v80, v90                                     // 0000000030E4: 7EA0035A
	v_cndmask_b32_e64 v120, v77, 0xff800000, s1                // 0000000030E8: D5010078 0005FF4D FF800000
	v_cndmask_b32_e32 v77, v84, v85, vcc_lo                    // 0000000030F4: 029AAB54
	v_cmp_lt_f32_e32 vcc_lo, v82, v89                          // 0000000030F8: 7C22B352
	s_or_b32 s2, s6, s2                                        // 0000000030FC: 8C020206
	v_permlanex16_b32 v80, v80, s10, 0xfedcba98                // 000000003100: D65C0050 03FC1550 FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 00000000310C: BF88FF9E
	v_cndmask_b32_e64 v78, v79, v83, s2                        // 000000003110: D501004E 000AA74F
	v_cndmask_b32_e64 v79, v84, v85, s2                        // 000000003118: D501004F 000AAB54
	s_and_b32 s2, s5, vcc_lo                                   // 000000003120: 8B026A05
	v_mov_b32_e32 v85, 0x7f800000                              // 000000003124: 7EAA02FF 7F800000
	s_wait_alu depctr_sa_sdst(0)                               // 00000000312C: BF88FF9E
	s_or_b32 vcc_lo, s3, s2                                    // 000000003130: 8C6A0203
	v_mov_b32_e32 v81, v120                                    // 000000003134: 7EA20378
	s_wait_alu depctr_sa_sdst(0)                               // 000000003138: BF88FF9E
	v_dual_max_num_f32 v83, v80, v80 :: v_dual_cndmask_b32 v80, v89, v82// 00000000313C: CA92A150 5350A559
	v_dual_mov_b32 v89, 0x7f800000 :: v_dual_mov_b32 v82, -1.0 // 000000003144: CA1000FF 595200F3 7F800000
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000003150: BF870093
	v_permlanex16_b32 v81, v81, s10, 0xfedcba98                // 000000003154: D65C0051 03FC1551 FEDCBA98
	v_dual_max_num_f32 v84, v81, v81 :: v_dual_cndmask_b32 v81, v87, v88// 000000003160: CA92A351 5450B157
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000003168: BF870114
	v_dual_mov_b32 v88, -1.0 :: v_dual_max_num_f32 v119, v90, v83// 00000000316C: CA1400F3 5876A75A
	v_dual_mov_b32 v87, -1.0 :: v_dual_max_num_f32 v120, v120, v84// 000000003174: CA1400F3 5778A978
	v_mov_b32_e32 v90, 0x7f800000                              // 00000000317C: 7EB402FF 7F800000
	v_dual_mov_b32 v84, 0x7f800000 :: v_dual_mov_b32 v83, -1.0 // 000000003184: CA1000FF 545200F3 7F800000
	s_branch 64158                                             // 000000003190: BFA0FA9E <nearest+0x1c0c>
	v_cndmask_b32_e64 v137, v90, -v14, s2                      // 000000003194: D5010089 400A1D5A
	v_cndmask_b32_e64 v138, v89, -v15, s3                      // 00000000319C: D501008A 400E1F59
	v_cndmask_b32_e64 v139, v88, v135, s2                      // 0000000031A4: D501008B 000B0F58
	v_cndmask_b32_e64 v140, v87, v136, s3                      // 0000000031AC: D501008C 000F1157
	v_cndmask_b32_e64 v84, v84, v90, s2                        // 0000000031B4: D5010054 000AB554
	v_cndmask_b32_e64 v85, v85, v89, s3                        // 0000000031BC: D5010055 000EB355
	v_cndmask_b32_e64 v82, v82, v88, s2                        // 0000000031C4: D5010052 000AB152
	v_cndmask_b32_e64 v83, v83, v87, s3                        // 0000000031CC: D5010053 000EAF53
	v_dual_mov_b32 v88, v139 :: v_dual_mov_b32 v87, v140       // 0000000031D4: CA10018B 5856018C
	v_dual_mov_b32 v90, v137 :: v_dual_mov_b32 v89, v138       // 0000000031DC: CA100189 5A58018A
	s_cmp_ngt_f32 s5, 0.5                                      // 0000000031E4: BF4BF005
	s_mov_b32 s15, s6                                          // 0000000031E8: BE8F0006
	s_cbranch_scc0 64666                                       // 0000000031EC: BFA1FC9A <nearest+0x2458>
	s_branch 64134                                             // 0000000031F0: BFA0FA86 <nearest+0x1c0c>
	v_cndmask_b32_e64 v137, v90, -v14, s2                      // 0000000031F4: D5010089 400A1D5A
	v_cndmask_b32_e64 v138, v89, -v15, s3                      // 0000000031FC: D501008A 400E1F59
	v_cndmask_b32_e64 v139, v88, v135, s2                      // 000000003204: D501008B 000B0F58
	v_cndmask_b32_e64 v140, v87, v136, s3                      // 00000000320C: D501008C 000F1157
	v_cndmask_b32_e64 v84, v84, v90, s2                        // 000000003214: D5010054 000AB554
	v_cndmask_b32_e64 v85, v85, v89, s3                        // 00000000321C: D5010055 000EB355
	v_cndmask_b32_e64 v82, v82, v88, s2                        // 000000003224: D5010052 000AB152
	v_cndmask_b32_e64 v83, v83, v87, s3                        // 00000000322C: D5010053 000EAF53
	v_dual_mov_b32 v89, v138 :: v_dual_mov_b32 v90, v137       // 000000003234: CA10018A 595A0189
	v_dual_mov_b32 v87, v140 :: v_dual_mov_b32 v88, v139       // 00000000323C: CA10018C 5758018B
	s_cmp_ngt_f32 s6, 0.5                                      // 000000003244: BF4BF006
	s_cbranch_scc0 63264                                       // 000000003248: BFA1F720 <nearest+0xecc>
	s_branch 62513                                             // 00000000324C: BFA0F431 <nearest+0x314>
	ds_load_b128 v[8:11], v114 offset:8192                     // 000000003250: DBFC2000 08000072
	ds_load_b128 v[12:15], v115 offset:8192                    // 000000003258: DBFC2000 0C000073
	ds_load_b128 v[48:51], v104 offset:8192                    // 000000003260: DBFC2000 30000068
	ds_load_b128 v[52:55], v114 offset:12288                   // 000000003268: DBFC3000 34000072
	ds_load_b128 v[56:59], v100 offset:8192                    // 000000003270: DBFC2000 38000064
	ds_load_b128 v[60:63], v97 offset:8192                     // 000000003278: DBFC2000 3C000061
	ds_load_b128 v[105:108], v115 offset:12288                 // 000000003280: DBFC3000 69000073
	ds_load_b128 v[109:112], v94 offset:8192                   // 000000003288: DBFC2000 6D00005E
	ds_load_b128 v[101:104], v104 offset:12288                 // 000000003290: DBFC3000 65000068
	ds_load_b128 v[113:116], v93 offset:8192                   // 000000003298: DBFC2000 7100005D
	ds_load_b128 v[121:124], v100 offset:12288                 // 0000000032A0: DBFC3000 79000064
	s_mov_b32 s2, 0x76543210                                   // 0000000032A8: BE8200FF 76543210
	s_wait_dscnt 0xa                                           // 0000000032B0: BFC6000A
	v_wmma_f32_16x16x16_f16 v[0:7], v[8:11], v[44:47], 0       // 0000000032B4: CC404000 1A025908
	s_wait_dscnt 0x9                                           // 0000000032BC: BFC60009
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000032C0: BF870001
	v_wmma_f32_16x16x16_f16 v[0:7], v[12:15], v[40:43], v[0:7] // 0000000032C4: CC404000 1C02510C
	s_wait_dscnt 0x7                                           // 0000000032CC: BFC60007
	v_wmma_f32_16x16x16_f16 v[8:15], v[52:55], v[44:47], 0     // 0000000032D0: CC404008 1A025934
	ds_load_b128 v[44:47], v94 offset:12288                    // 0000000032D8: DBFC3000 2C00005E
	v_wmma_f32_16x16x16_f16 v[0:7], v[48:51], v[36:39], v[0:7] // 0000000032E0: CC404000 1C024930
	ds_load_b128 v[48:51], v92 offset:8192                     // 0000000032E8: DBFC2000 3000005C
	ds_load_b128 v[95:98], v97 offset:12288                    // 0000000032F0: DBFC3000 5F000061
	s_wait_dscnt 0x7                                           // 0000000032F8: BFC60007
	v_wmma_f32_16x16x16_f16 v[8:15], v[105:108], v[40:43], v[8:15]// 0000000032FC: CC404008 1C225169
	ds_load_b128 v[40:43], v93 offset:12288                    // 000000003304: DBFC3000 2800005D
	v_wmma_f32_16x16x16_f16 v[0:7], v[56:59], v[32:35], v[0:7] // 00000000330C: CC404000 1C024138
	s_wait_dscnt 0x6                                           // 000000003314: BFC60006
	v_wmma_f32_16x16x16_f16 v[8:15], v[101:104], v[36:39], v[8:15]// 000000003318: CC404008 1C224965
	ds_load_b128 v[36:39], v92 offset:12288                    // 000000003320: DBFC3000 2400005C
	v_wmma_f32_16x16x16_f16 v[0:7], v[60:63], v[28:31], v[0:7] // 000000003328: CC404000 1C02393C
	s_wait_dscnt 0x5                                           // 000000003330: BFC60005
	v_wmma_f32_16x16x16_f16 v[8:15], v[121:124], v[32:35], v[8:15]// 000000003334: CC404008 1C224179
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)// 00000000333C: BF8701B2
	v_wmma_f32_16x16x16_f16 v[0:7], v[109:112], v[24:27], v[0:7]// 000000003340: CC404000 1C02316D
	v_wmma_f32_16x16x16_f16 v[0:7], v[113:116], v[20:23], v[0:7]// 000000003348: CC404000 1C022971
	s_wait_dscnt 0x2                                           // 000000003350: BFC60002
	v_wmma_f32_16x16x16_f16 v[8:15], v[95:98], v[28:31], v[8:15]// 000000003354: CC404008 1C22395F
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000335C: BF870112
	v_wmma_f32_16x16x16_f16 v[0:7], v[48:51], v[16:19], v[0:7] // 000000003360: CC404000 1C022130
	v_wmma_f32_16x16x16_f16 v[8:15], v[44:47], v[24:27], v[8:15]// 000000003368: CC404008 1C22312C
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000003370: BF870122
	v_min3_num_f32 v24, -v0, -v1, -v2                          // 000000003374: D6290018 E40A0300
	s_wait_dscnt 0x1                                           // 00000000337C: BFC60001
	v_wmma_f32_16x16x16_f16 v[8:15], v[40:43], v[20:23], v[8:15]// 000000003380: CC404008 1C222928
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000003388: BF870122
	v_min3_num_f32 v20, v24, -v3, -v4                          // 00000000338C: D6290014 C4120718
	s_wait_dscnt 0x0                                           // 000000003394: BFC60000
	v_wmma_f32_16x16x16_f16 v[8:15], v[36:39], v[16:19], v[8:15]// 000000003398: CC404008 1C222124
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000033A0: BF870112
	v_min3_num_f32 v16, v20, -v5, -v6                          // 0000000033A4: D6290010 C41A0B14
	v_max_num_f32_e64 v17, -v15, -v15                          // 0000000033AC: D5160011 62021F0F
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000033B4: BF870092
	v_min3_num_f32 v16, v16, -v7, -v8                          // 0000000033B8: D6290010 C4220F10
	v_min3_num_f32 v16, v16, -v9, -v10                         // 0000000033C0: D6290010 C42A1310
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000033C8: BF870091
	v_min3_num_f32 v16, v16, -v11, -v12                        // 0000000033CC: D6290010 C4321710
	v_min3_num_f32 v16, v16, -v13, -v14                        // 0000000033D4: D6290010 C43A1B10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000033DC: BF870091
	v_min_num_f32_e32 v16, v16, v17                            // 0000000033E0: 2A202310
	v_mov_b32_e32 v17, v16                                     // 0000000033E4: 7E220310
	s_wait_alu depctr_sa_sdst(0)                               // 0000000033E8: BF88FF9E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000033EC: BF870091
	v_permlanex16_b32 v17, v17, s2, 0xfedcba98                 // 0000000033F0: D65C0011 03FC0511 FEDCBA98
	v_max_num_f32_e64 v17, -v17, -v17                          // 0000000033FC: D5160011 62022311
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000003404: BF870091
	v_max_num_f32_e64 v16, -v16, v17                           // 000000003408: D5160010 22022310
	v_cmp_ge_f32_e64 s2, v119, -v16                            // 000000003410: D4160002 42022177
	s_mov_b32 vcc_lo, s2                                       // 000000003418: BEEA0002
	s_wait_alu depctr_sa_sdst(0)                               // 00000000341C: BF88FF9E
	s_cbranch_vccz 2853                                        // 000000003420: BFA30B25 <nearest+0x60b8>
	v_cmp_ge_f32_e64 s3, v119, -v0                             // 000000003424: D4160003 42020177
	v_cmp_ge_f32_e64 s5, v119, -v1                             // 00000000342C: D4160005 42020377
	v_or_b32_e32 v17, 0xfe0, v86                               // 000000003434: 3822ACFF 00000FE0
	v_or_b32_e32 v18, 0xfe1, v86                               // 00000000343C: 3824ACFF 00000FE1
	s_and_b32 s4, s3, s2                                       // 000000003444: 8B040203
	s_and_b32 s7, s5, s2                                       // 000000003448: 8B070205
	s_mov_b32 s6, 0                                            // 00000000344C: BE860080
	s_wait_alu depctr_sa_sdst(0)                               // 000000003450: BF88FF9E
	s_or_b32 s3, s4, s7                                        // 000000003454: 8C030704
	s_wait_alu depctr_sa_sdst(0)                               // 000000003458: BF88FF9E
	v_cndmask_b32_e64 v16, 0, 1, s3                            // 00000000345C: D5010010 000D0280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000003464: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v16                            // 000000003468: 7C9A2080
	v_cvt_f32_u32_e32 v16, v17                                 // 00000000346C: 7E200D11
	v_cvt_f32_u32_e32 v17, v18                                 // 000000003470: 7E220D12
	s_cbranch_vccz 51                                          // 000000003474: BFA30033 <nearest+0x3544>
	v_cmp_gt_f32_e64 s6, v119, -v0                             // 000000003478: D4140006 42020177
	s_delay_alu instid0(VALU_DEP_3)                            // 000000003480: BF870003
	v_cmp_gt_f32_e32 vcc_lo, v120, v16                         // 000000003484: 7C282178
	v_cmp_gt_f32_e64 s8, v119, -v1                             // 000000003488: D4140008 42020377
	v_cmp_gt_f32_e64 s3, v120, v17                             // 000000003490: D4140003 02022378
	v_cmp_lt_f32_e64 s5, -1.0, v82                             // 000000003498: D4110005 0202A4F3
	s_or_b32 s6, vcc_lo, s6                                    // 0000000034A0: 8C06066A
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v83                         // 0000000034A4: 7C22A6F3
	s_or_b32 s8, s3, s8                                        // 0000000034A8: 8C080803
	s_wait_alu depctr_sa_sdst(0)                               // 0000000034AC: BF88FF9E
	s_and_b32 s3, s6, s4                                       // 0000000034B0: 8B030406
	s_and_b32 s4, s8, s7                                       // 0000000034B4: 8B040708
	s_wait_alu depctr_sa_sdst(0)                               // 0000000034B8: BF88FF9E
	s_and_b32 s5, s5, s3                                       // 0000000034BC: 8B050305
	s_and_b32 s6, vcc_lo, s4                                   // 0000000034C0: 8B06046A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000034C4: BF88FF9E
	s_or_b32 s5, s5, s6                                        // 0000000034C8: 8C050605
	s_mov_b32 s6, 0                                            // 0000000034CC: BE860080
	s_wait_alu depctr_sa_sdst(0)                               // 0000000034D0: BF88FF9E
	v_cndmask_b32_e64 v18, 0, 1, s5                            // 0000000034D4: D5010012 00150280
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000034DC: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v18                            // 0000000034E0: 7C9A2480
	s_cbranch_vccz 3                                           // 0000000034E4: BFA30003 <nearest+0x34f4>
	s_mov_b32 s6, 1.0                                          // 0000000034E8: BE8600F2
	s_mov_b32 s10, 0                                           // 0000000034EC: BE8A0080
	s_branch 22                                                // 0000000034F0: BFA00016 <nearest+0x354c>
	v_cndmask_b32_e64 v18, v90, -v0, s3                        // 0000000034F4: D5010012 400E015A
	v_cndmask_b32_e64 v19, v89, -v1, s4                        // 0000000034FC: D5010013 40120359
	v_cndmask_b32_e64 v20, v88, v16, s3                        // 000000003504: D5010014 000E2158
	v_cndmask_b32_e64 v21, v87, v17, s4                        // 00000000350C: D5010015 00122357
	v_cndmask_b32_e64 v84, v84, v90, s3                        // 000000003514: D5010054 000EB554
	v_cndmask_b32_e64 v85, v85, v89, s4                        // 00000000351C: D5010055 0012B355
	v_cndmask_b32_e64 v82, v82, v88, s3                        // 000000003524: D5010052 000EB152
	v_cndmask_b32_e64 v83, v83, v87, s4                        // 00000000352C: D5010053 0012AF53
	v_dual_mov_b32 v89, v19 :: v_dual_mov_b32 v90, v18         // 000000003534: CA100113 595A0112
	v_dual_mov_b32 v87, v21 :: v_dual_mov_b32 v88, v20         // 00000000353C: CA100115 57580114
	s_mov_b32 s10, 0x41000000                                  // 000000003544: BE8A00FF 41000000
	s_wait_alu depctr_sa_sdst(0)                               // 00000000354C: BF88FF9E
	s_cmp_lt_f32 s6, 0.5                                       // 000000003550: BF41F006
	v_cmp_ge_f32_e64 s3, v119, -v2                             // 000000003554: D4160003 42020577
	v_cmp_ge_f32_e64 s5, v119, -v3                             // 00000000355C: D4160005 42020777
	v_or_b32_e32 v19, 0xfe2, v86                               // 000000003564: 3826ACFF 00000FE2
	s_cselect_b32 s4, -1, 0                                    // 00000000356C: 980480C1
	v_or_b32_e32 v20, 0xfe3, v86                               // 000000003570: 3828ACFF 00000FE3
	s_wait_alu depctr_sa_sdst(0)                               // 000000003578: BF88FF9E
	s_and_b32 s7, s2, s4                                       // 00000000357C: 8B070402
	s_wait_alu depctr_sa_sdst(0)                               // 000000003580: BF88FF9E
	s_and_b32 s4, s3, s7                                       // 000000003584: 8B040703
	s_and_b32 s7, s5, s7                                       // 000000003588: 8B070705
	s_wait_alu depctr_sa_sdst(0)                               // 00000000358C: BF88FF9E
	s_or_b32 s3, s4, s7                                        // 000000003590: 8C030704
	s_wait_alu depctr_sa_sdst(0)                               // 000000003594: BF88FF9E
	v_cndmask_b32_e64 v18, 0, 1, s3                            // 000000003598: D5010012 000D0280
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000035A0: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v18                            // 0000000035A4: 7C9A2480
	v_cvt_f32_u32_e32 v18, v19                                 // 0000000035A8: 7E240D13
	v_cvt_f32_u32_e32 v19, v20                                 // 0000000035AC: 7E260D14
	s_cbranch_vccz 50                                          // 0000000035B0: BFA30032 <nearest+0x367c>
	v_cmp_gt_f32_e64 s8, v119, -v2                             // 0000000035B4: D4140008 42020577
	s_delay_alu instid0(VALU_DEP_3)                            // 0000000035BC: BF870003
	v_cmp_gt_f32_e32 vcc_lo, v120, v18                         // 0000000035C0: 7C282578
	v_cmp_gt_f32_e64 s11, v119, -v3                            // 0000000035C4: D414000B 42020777
	v_cmp_gt_f32_e64 s3, v120, v19                             // 0000000035CC: D4140003 02022778
	v_cmp_lt_f32_e64 s5, -1.0, v82                             // 0000000035D4: D4110005 0202A4F3
	s_or_b32 s8, vcc_lo, s8                                    // 0000000035DC: 8C08086A
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v83                         // 0000000035E0: 7C22A6F3
	s_or_b32 s11, s3, s11                                      // 0000000035E4: 8C0B0B03
	s_and_b32 s3, s8, s4                                       // 0000000035E8: 8B030408
	s_wait_alu depctr_sa_sdst(0)                               // 0000000035EC: BF88FF9E
	s_and_b32 s4, s11, s7                                      // 0000000035F0: 8B04070B
	s_and_b32 s5, s5, s3                                       // 0000000035F4: 8B050305
	s_wait_alu depctr_sa_sdst(0)                               // 0000000035F8: BF88FF9E
	s_and_b32 s7, s4, vcc_lo                                   // 0000000035FC: 8B076A04
	s_wait_alu depctr_sa_sdst(0)                               // 000000003600: BF88FF9E
	s_or_b32 s5, s5, s7                                        // 000000003604: 8C050705
	s_wait_alu depctr_sa_sdst(0)                               // 000000003608: BF88FF9E
	v_cndmask_b32_e64 v20, 0, 1, s5                            // 00000000360C: D5010014 00150280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000003614: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v20                            // 000000003618: 7C9A2880
	s_cbranch_vccz 3                                           // 00000000361C: BFA30003 <nearest+0x362c>
	s_mov_b32 s10, 1.0                                         // 000000003620: BE8A00F2
	s_mov_b32 s6, 1.0                                          // 000000003624: BE8600F2
	s_branch 20                                                // 000000003628: BFA00014 <nearest+0x367c>
	v_cndmask_b32_e64 v20, v90, -v2, s3                        // 00000000362C: D5010014 400E055A
	v_cndmask_b32_e64 v21, v89, -v3, s4                        // 000000003634: D5010015 40120759
	v_cndmask_b32_e64 v22, v88, v18, s3                        // 00000000363C: D5010016 000E2558
	v_cndmask_b32_e64 v23, v87, v19, s4                        // 000000003644: D5010017 00122757
	v_cndmask_b32_e64 v84, v84, v90, s3                        // 00000000364C: D5010054 000EB554
	v_cndmask_b32_e64 v85, v85, v89, s4                        // 000000003654: D5010055 0012B355
	v_cndmask_b32_e64 v82, v82, v88, s3                        // 00000000365C: D5010052 000EB152
	v_cndmask_b32_e64 v83, v83, v87, s4                        // 000000003664: D5010053 0012AF53
	v_dual_mov_b32 v89, v21 :: v_dual_mov_b32 v90, v20         // 00000000366C: CA100115 595A0114
	v_dual_mov_b32 v87, v23 :: v_dual_mov_b32 v88, v22         // 000000003674: CA100117 57580116
	s_wait_alu depctr_sa_sdst(0)                               // 00000000367C: BF88FF9E
	s_cmp_lt_f32 s6, 0.5                                       // 000000003680: BF41F006
	v_cmp_ge_f32_e64 s3, v119, -v4                             // 000000003684: D4160003 42020977
	v_cmp_ge_f32_e64 s5, v119, -v5                             // 00000000368C: D4160005 42020B77
	v_or_b32_e32 v21, 0xfe4, v86                               // 000000003694: 382AACFF 00000FE4
	s_cselect_b32 s4, -1, 0                                    // 00000000369C: 980480C1
	v_or_b32_e32 v22, 0xfe5, v86                               // 0000000036A0: 382CACFF 00000FE5
	s_wait_alu depctr_sa_sdst(0)                               // 0000000036A8: BF88FF9E
	s_and_b32 s7, s2, s4                                       // 0000000036AC: 8B070402
	s_wait_alu depctr_sa_sdst(0)                               // 0000000036B0: BF88FF9E
	s_and_b32 s4, s3, s7                                       // 0000000036B4: 8B040703
	s_and_b32 s7, s5, s7                                       // 0000000036B8: 8B070705
	s_wait_alu depctr_sa_sdst(0)                               // 0000000036BC: BF88FF9E
	s_or_b32 s3, s4, s7                                        // 0000000036C0: 8C030704
	s_wait_alu depctr_sa_sdst(0)                               // 0000000036C4: BF88FF9E
	v_cndmask_b32_e64 v20, 0, 1, s3                            // 0000000036C8: D5010014 000D0280
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000036D0: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v20                            // 0000000036D4: 7C9A2880
	v_cvt_f32_u32_e32 v20, v21                                 // 0000000036D8: 7E280D15
	v_cvt_f32_u32_e32 v21, v22                                 // 0000000036DC: 7E2A0D16
	s_cbranch_vccz 50                                          // 0000000036E0: BFA30032 <nearest+0x37ac>
	v_cmp_gt_f32_e64 s8, v119, -v4                             // 0000000036E4: D4140008 42020977
	s_delay_alu instid0(VALU_DEP_3)                            // 0000000036EC: BF870003
	v_cmp_gt_f32_e32 vcc_lo, v120, v20                         // 0000000036F0: 7C282978
	v_cmp_gt_f32_e64 s11, v119, -v5                            // 0000000036F4: D414000B 42020B77
	v_cmp_gt_f32_e64 s3, v120, v21                             // 0000000036FC: D4140003 02022B78
	v_cmp_lt_f32_e64 s5, -1.0, v82                             // 000000003704: D4110005 0202A4F3
	s_or_b32 s8, vcc_lo, s8                                    // 00000000370C: 8C08086A
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v83                         // 000000003710: 7C22A6F3
	s_or_b32 s11, s3, s11                                      // 000000003714: 8C0B0B03
	s_and_b32 s3, s8, s4                                       // 000000003718: 8B030408
	s_wait_alu depctr_sa_sdst(0)                               // 00000000371C: BF88FF9E
	s_and_b32 s4, s11, s7                                      // 000000003720: 8B04070B
	s_and_b32 s5, s5, s3                                       // 000000003724: 8B050305
	s_wait_alu depctr_sa_sdst(0)                               // 000000003728: BF88FF9E
	s_and_b32 s7, s4, vcc_lo                                   // 00000000372C: 8B076A04
	s_wait_alu depctr_sa_sdst(0)                               // 000000003730: BF88FF9E
	s_or_b32 s5, s5, s7                                        // 000000003734: 8C050705
	s_wait_alu depctr_sa_sdst(0)                               // 000000003738: BF88FF9E
	v_cndmask_b32_e64 v22, 0, 1, s5                            // 00000000373C: D5010016 00150280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000003744: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v22                            // 000000003748: 7C9A2C80
	s_cbranch_vccz 3                                           // 00000000374C: BFA30003 <nearest+0x375c>
	s_mov_b32 s6, 1.0                                          // 000000003750: BE8600F2
	s_mov_b32 s10, 2.0                                         // 000000003754: BE8A00F4
	s_branch 20                                                // 000000003758: BFA00014 <nearest+0x37ac>
	v_cndmask_b32_e64 v22, v90, -v4, s3                        // 00000000375C: D5010016 400E095A
	v_cndmask_b32_e64 v23, v89, -v5, s4                        // 000000003764: D5010017 40120B59
	v_cndmask_b32_e64 v24, v88, v20, s3                        // 00000000376C: D5010018 000E2958
	v_cndmask_b32_e64 v25, v87, v21, s4                        // 000000003774: D5010019 00122B57
	v_cndmask_b32_e64 v84, v84, v90, s3                        // 00000000377C: D5010054 000EB554
	v_cndmask_b32_e64 v85, v85, v89, s4                        // 000000003784: D5010055 0012B355
	v_cndmask_b32_e64 v82, v82, v88, s3                        // 00000000378C: D5010052 000EB152
	v_cndmask_b32_e64 v83, v83, v87, s4                        // 000000003794: D5010053 0012AF53
	v_dual_mov_b32 v89, v23 :: v_dual_mov_b32 v90, v22         // 00000000379C: CA100117 595A0116
	v_dual_mov_b32 v87, v25 :: v_dual_mov_b32 v88, v24         // 0000000037A4: CA100119 57580118
	s_wait_alu depctr_sa_sdst(0)                               // 0000000037AC: BF88FF9E
	s_cmp_lt_f32 s6, 0.5                                       // 0000000037B0: BF41F006
	v_cmp_ge_f32_e64 s3, v119, -v6                             // 0000000037B4: D4160003 42020D77
	v_cmp_ge_f32_e64 s5, v119, -v7                             // 0000000037BC: D4160005 42020F77
	v_or_b32_e32 v23, 0xfe6, v86                               // 0000000037C4: 382EACFF 00000FE6
	s_cselect_b32 s4, -1, 0                                    // 0000000037CC: 980480C1
	v_or_b32_e32 v24, 0xfe7, v86                               // 0000000037D0: 3830ACFF 00000FE7
	s_wait_alu depctr_sa_sdst(0)                               // 0000000037D8: BF88FF9E
	s_and_b32 s7, s2, s4                                       // 0000000037DC: 8B070402
	s_wait_alu depctr_sa_sdst(0)                               // 0000000037E0: BF88FF9E
	s_and_b32 s4, s3, s7                                       // 0000000037E4: 8B040703
	s_and_b32 s7, s5, s7                                       // 0000000037E8: 8B070705
	s_wait_alu depctr_sa_sdst(0)                               // 0000000037EC: BF88FF9E
	s_or_b32 s3, s4, s7                                        // 0000000037F0: 8C030704
	s_wait_alu depctr_sa_sdst(0)                               // 0000000037F4: BF88FF9E
	v_cndmask_b32_e64 v22, 0, 1, s3                            // 0000000037F8: D5010016 000D0280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000003800: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v22                            // 000000003804: 7C9A2C80
	v_cvt_f32_u32_e32 v22, v23                                 // 000000003808: 7E2C0D17
	v_cvt_f32_u32_e32 v23, v24                                 // 00000000380C: 7E2E0D18
	s_cbranch_vccz 51                                          // 000000003810: BFA30033 <nearest+0x38e0>
	v_cmp_gt_f32_e64 s8, v119, -v6                             // 000000003814: D4140008 42020D77
	s_delay_alu instid0(VALU_DEP_3)                            // 00000000381C: BF870003
	v_cmp_gt_f32_e32 vcc_lo, v120, v22                         // 000000003820: 7C282D78
	v_cmp_gt_f32_e64 s11, v119, -v7                            // 000000003824: D414000B 42020F77
	v_cmp_gt_f32_e64 s3, v120, v23                             // 00000000382C: D4140003 02022F78
	v_cmp_lt_f32_e64 s5, -1.0, v82                             // 000000003834: D4110005 0202A4F3
	s_or_b32 s8, vcc_lo, s8                                    // 00000000383C: 8C08086A
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v83                         // 000000003840: 7C22A6F3
	s_or_b32 s11, s3, s11                                      // 000000003844: 8C0B0B03
	s_and_b32 s3, s8, s4                                       // 000000003848: 8B030408
	s_wait_alu depctr_sa_sdst(0)                               // 00000000384C: BF88FF9E
	s_and_b32 s4, s11, s7                                      // 000000003850: 8B04070B
	s_and_b32 s5, s5, s3                                       // 000000003854: 8B050305
	s_wait_alu depctr_sa_sdst(0)                               // 000000003858: BF88FF9E
	s_and_b32 s7, s4, vcc_lo                                   // 00000000385C: 8B076A04
	s_wait_alu depctr_sa_sdst(0)                               // 000000003860: BF88FF9E
	s_or_b32 s5, s5, s7                                        // 000000003864: 8C050705
	s_wait_alu depctr_sa_sdst(0)                               // 000000003868: BF88FF9E
	v_cndmask_b32_e64 v24, 0, 1, s5                            // 00000000386C: D5010018 00150280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000003874: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v24                            // 000000003878: 7C9A3080
	s_cbranch_vccz 4                                           // 00000000387C: BFA30004 <nearest+0x3890>
	s_mov_b32 s6, 1.0                                          // 000000003880: BE8600F2
	s_mov_b32 s10, 0x40400000                                  // 000000003884: BE8A00FF 40400000
	s_branch 20                                                // 00000000388C: BFA00014 <nearest+0x38e0>
	v_cndmask_b32_e64 v24, v90, -v6, s3                        // 000000003890: D5010018 400E0D5A
	v_cndmask_b32_e64 v25, v89, -v7, s4                        // 000000003898: D5010019 40120F59
	v_cndmask_b32_e64 v26, v88, v22, s3                        // 0000000038A0: D501001A 000E2D58
	v_cndmask_b32_e64 v27, v87, v23, s4                        // 0000000038A8: D501001B 00122F57
	v_cndmask_b32_e64 v84, v84, v90, s3                        // 0000000038B0: D5010054 000EB554
	v_cndmask_b32_e64 v85, v85, v89, s4                        // 0000000038B8: D5010055 0012B355
	v_cndmask_b32_e64 v82, v82, v88, s3                        // 0000000038C0: D5010052 000EB152
	v_cndmask_b32_e64 v83, v83, v87, s4                        // 0000000038C8: D5010053 0012AF53
	v_dual_mov_b32 v89, v25 :: v_dual_mov_b32 v90, v24         // 0000000038D0: CA100119 595A0118
	v_dual_mov_b32 v87, v27 :: v_dual_mov_b32 v88, v26         // 0000000038D8: CA10011B 5758011A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000038E0: BF88FF9E
	s_cmp_lt_f32 s6, 0.5                                       // 0000000038E4: BF41F006
	v_cmp_ge_f32_e64 s3, v119, -v8                             // 0000000038E8: D4160003 42021177
	v_cmp_ge_f32_e64 s5, v119, -v9                             // 0000000038F0: D4160005 42021377
	v_or_b32_e32 v25, 0xff0, v86                               // 0000000038F8: 3832ACFF 00000FF0
	s_cselect_b32 s4, -1, 0                                    // 000000003900: 980480C1
	v_or_b32_e32 v26, 0xff1, v86                               // 000000003904: 3834ACFF 00000FF1
	s_wait_alu depctr_sa_sdst(0)                               // 00000000390C: BF88FF9E
	s_and_b32 s7, s2, s4                                       // 000000003910: 8B070402
	s_wait_alu depctr_sa_sdst(0)                               // 000000003914: BF88FF9E
	s_and_b32 s4, s3, s7                                       // 000000003918: 8B040703
	s_and_b32 s7, s5, s7                                       // 00000000391C: 8B070705
	s_wait_alu depctr_sa_sdst(0)                               // 000000003920: BF88FF9E
	s_or_b32 s3, s4, s7                                        // 000000003924: 8C030704
	s_wait_alu depctr_sa_sdst(0)                               // 000000003928: BF88FF9E
	v_cndmask_b32_e64 v24, 0, 1, s3                            // 00000000392C: D5010018 000D0280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000003934: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v24                            // 000000003938: 7C9A3080
	v_cvt_f32_u32_e32 v24, v25                                 // 00000000393C: 7E300D19
	v_cvt_f32_u32_e32 v25, v26                                 // 000000003940: 7E320D1A
	s_cbranch_vccz 50                                          // 000000003944: BFA30032 <nearest+0x3a10>
	v_cmp_gt_f32_e64 s8, v119, -v8                             // 000000003948: D4140008 42021177
	s_delay_alu instid0(VALU_DEP_3)                            // 000000003950: BF870003
	v_cmp_gt_f32_e32 vcc_lo, v120, v24                         // 000000003954: 7C283178
	v_cmp_gt_f32_e64 s11, v119, -v9                            // 000000003958: D414000B 42021377
	v_cmp_gt_f32_e64 s3, v120, v25                             // 000000003960: D4140003 02023378
	v_cmp_lt_f32_e64 s5, -1.0, v82                             // 000000003968: D4110005 0202A4F3
	s_or_b32 s8, vcc_lo, s8                                    // 000000003970: 8C08086A
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v83                         // 000000003974: 7C22A6F3
	s_or_b32 s11, s3, s11                                      // 000000003978: 8C0B0B03
	s_and_b32 s3, s8, s4                                       // 00000000397C: 8B030408
	s_wait_alu depctr_sa_sdst(0)                               // 000000003980: BF88FF9E
	s_and_b32 s4, s11, s7                                      // 000000003984: 8B04070B
	s_and_b32 s5, s5, s3                                       // 000000003988: 8B050305
	s_wait_alu depctr_sa_sdst(0)                               // 00000000398C: BF88FF9E
	s_and_b32 s7, s4, vcc_lo                                   // 000000003990: 8B076A04
	s_wait_alu depctr_sa_sdst(0)                               // 000000003994: BF88FF9E
	s_or_b32 s5, s5, s7                                        // 000000003998: 8C050705
	s_wait_alu depctr_sa_sdst(0)                               // 00000000399C: BF88FF9E
	v_cndmask_b32_e64 v26, 0, 1, s5                            // 0000000039A0: D501001A 00150280
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000039A8: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v26                            // 0000000039AC: 7C9A3480
	s_cbranch_vccz 3                                           // 0000000039B0: BFA30003 <nearest+0x39c0>
	s_mov_b32 s6, 1.0                                          // 0000000039B4: BE8600F2
	s_mov_b32 s10, 4.0                                         // 0000000039B8: BE8A00F6
	s_branch 20                                                // 0000000039BC: BFA00014 <nearest+0x3a10>
	v_cndmask_b32_e64 v26, v90, -v8, s3                        // 0000000039C0: D501001A 400E115A
	v_cndmask_b32_e64 v27, v89, -v9, s4                        // 0000000039C8: D501001B 40121359
	v_cndmask_b32_e64 v28, v88, v24, s3                        // 0000000039D0: D501001C 000E3158
	v_cndmask_b32_e64 v29, v87, v25, s4                        // 0000000039D8: D501001D 00123357
	v_cndmask_b32_e64 v84, v84, v90, s3                        // 0000000039E0: D5010054 000EB554
	v_cndmask_b32_e64 v85, v85, v89, s4                        // 0000000039E8: D5010055 0012B355
	v_cndmask_b32_e64 v82, v82, v88, s3                        // 0000000039F0: D5010052 000EB152
	v_cndmask_b32_e64 v83, v83, v87, s4                        // 0000000039F8: D5010053 0012AF53
	v_dual_mov_b32 v89, v27 :: v_dual_mov_b32 v90, v26         // 000000003A00: CA10011B 595A011A
	v_dual_mov_b32 v87, v29 :: v_dual_mov_b32 v88, v28         // 000000003A08: CA10011D 5758011C
	s_wait_alu depctr_sa_sdst(0)                               // 000000003A10: BF88FF9E
	s_cmp_lt_f32 s6, 0.5                                       // 000000003A14: BF41F006
	v_cmp_ge_f32_e64 s3, v119, -v10                            // 000000003A18: D4160003 42021577
	v_cmp_ge_f32_e64 s5, v119, -v11                            // 000000003A20: D4160005 42021777
	v_or_b32_e32 v27, 0xff2, v86                               // 000000003A28: 3836ACFF 00000FF2
	s_cselect_b32 s4, -1, 0                                    // 000000003A30: 980480C1
	v_or_b32_e32 v28, 0xff3, v86                               // 000000003A34: 3838ACFF 00000FF3
	s_wait_alu depctr_sa_sdst(0)                               // 000000003A3C: BF88FF9E
	s_and_b32 s7, s2, s4                                       // 000000003A40: 8B070402
	s_wait_alu depctr_sa_sdst(0)                               // 000000003A44: BF88FF9E
	s_and_b32 s4, s3, s7                                       // 000000003A48: 8B040703
	s_and_b32 s7, s5, s7                                       // 000000003A4C: 8B070705
	s_wait_alu depctr_sa_sdst(0)                               // 000000003A50: BF88FF9E
	s_or_b32 s3, s4, s7                                        // 000000003A54: 8C030704
	s_wait_alu depctr_sa_sdst(0)                               // 000000003A58: BF88FF9E
	v_cndmask_b32_e64 v26, 0, 1, s3                            // 000000003A5C: D501001A 000D0280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000003A64: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v26                            // 000000003A68: 7C9A3480
	v_cvt_f32_u32_e32 v26, v27                                 // 000000003A6C: 7E340D1B
	v_cvt_f32_u32_e32 v27, v28                                 // 000000003A70: 7E360D1C
	s_cbranch_vccz 51                                          // 000000003A74: BFA30033 <nearest+0x3b44>
	v_cmp_gt_f32_e64 s8, v119, -v10                            // 000000003A78: D4140008 42021577
	s_delay_alu instid0(VALU_DEP_3)                            // 000000003A80: BF870003
	v_cmp_gt_f32_e32 vcc_lo, v120, v26                         // 000000003A84: 7C283578
	v_cmp_gt_f32_e64 s11, v119, -v11                           // 000000003A88: D414000B 42021777
	v_cmp_gt_f32_e64 s3, v120, v27                             // 000000003A90: D4140003 02023778
	v_cmp_lt_f32_e64 s5, -1.0, v82                             // 000000003A98: D4110005 0202A4F3
	s_or_b32 s8, vcc_lo, s8                                    // 000000003AA0: 8C08086A
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v83                         // 000000003AA4: 7C22A6F3
	s_or_b32 s11, s3, s11                                      // 000000003AA8: 8C0B0B03
	s_and_b32 s3, s8, s4                                       // 000000003AAC: 8B030408
	s_wait_alu depctr_sa_sdst(0)                               // 000000003AB0: BF88FF9E
	s_and_b32 s4, s11, s7                                      // 000000003AB4: 8B04070B
	s_and_b32 s5, s5, s3                                       // 000000003AB8: 8B050305
	s_wait_alu depctr_sa_sdst(0)                               // 000000003ABC: BF88FF9E
	s_and_b32 s7, s4, vcc_lo                                   // 000000003AC0: 8B076A04
	s_wait_alu depctr_sa_sdst(0)                               // 000000003AC4: BF88FF9E
	s_or_b32 s5, s5, s7                                        // 000000003AC8: 8C050705
	s_wait_alu depctr_sa_sdst(0)                               // 000000003ACC: BF88FF9E
	v_cndmask_b32_e64 v28, 0, 1, s5                            // 000000003AD0: D501001C 00150280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000003AD8: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v28                            // 000000003ADC: 7C9A3880
	s_cbranch_vccz 4                                           // 000000003AE0: BFA30004 <nearest+0x3af4>
	s_mov_b32 s6, 1.0                                          // 000000003AE4: BE8600F2
	s_mov_b32 s10, 0x40a00000                                  // 000000003AE8: BE8A00FF 40A00000
	s_branch 20                                                // 000000003AF0: BFA00014 <nearest+0x3b44>
	v_cndmask_b32_e64 v28, v90, -v10, s3                       // 000000003AF4: D501001C 400E155A
	v_cndmask_b32_e64 v29, v89, -v11, s4                       // 000000003AFC: D501001D 40121759
	v_cndmask_b32_e64 v30, v88, v26, s3                        // 000000003B04: D501001E 000E3558
	v_cndmask_b32_e64 v31, v87, v27, s4                        // 000000003B0C: D501001F 00123757
	v_cndmask_b32_e64 v84, v84, v90, s3                        // 000000003B14: D5010054 000EB554
	v_cndmask_b32_e64 v85, v85, v89, s4                        // 000000003B1C: D5010055 0012B355
	v_cndmask_b32_e64 v82, v82, v88, s3                        // 000000003B24: D5010052 000EB152
	v_cndmask_b32_e64 v83, v83, v87, s4                        // 000000003B2C: D5010053 0012AF53
	v_dual_mov_b32 v89, v29 :: v_dual_mov_b32 v90, v28         // 000000003B34: CA10011D 595A011C
	v_dual_mov_b32 v87, v31 :: v_dual_mov_b32 v88, v30         // 000000003B3C: CA10011F 5758011E
	s_wait_alu depctr_sa_sdst(0)                               // 000000003B44: BF88FF9E
	s_cmp_lt_f32 s6, 0.5                                       // 000000003B48: BF41F006
	v_cmp_ge_f32_e64 s3, v119, -v12                            // 000000003B4C: D4160003 42021977
	v_cmp_ge_f32_e64 s5, v119, -v13                            // 000000003B54: D4160005 42021B77
	v_or_b32_e32 v29, 0xff4, v86                               // 000000003B5C: 383AACFF 00000FF4
	s_cselect_b32 s4, -1, 0                                    // 000000003B64: 980480C1
	v_or_b32_e32 v30, 0xff5, v86                               // 000000003B68: 383CACFF 00000FF5
	s_wait_alu depctr_sa_sdst(0)                               // 000000003B70: BF88FF9E
	s_and_b32 s7, s2, s4                                       // 000000003B74: 8B070402
	s_wait_alu depctr_sa_sdst(0)                               // 000000003B78: BF88FF9E
	s_and_b32 s4, s3, s7                                       // 000000003B7C: 8B040703
	s_and_b32 s7, s5, s7                                       // 000000003B80: 8B070705
	s_wait_alu depctr_sa_sdst(0)                               // 000000003B84: BF88FF9E
	s_or_b32 s3, s4, s7                                        // 000000003B88: 8C030704
	s_wait_alu depctr_sa_sdst(0)                               // 000000003B8C: BF88FF9E
	v_cndmask_b32_e64 v28, 0, 1, s3                            // 000000003B90: D501001C 000D0280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000003B98: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v28                            // 000000003B9C: 7C9A3880
	v_cvt_f32_u32_e32 v28, v29                                 // 000000003BA0: 7E380D1D
	v_cvt_f32_u32_e32 v29, v30                                 // 000000003BA4: 7E3A0D1E
	s_cbranch_vccz 51                                          // 000000003BA8: BFA30033 <nearest+0x3c78>
	v_cmp_gt_f32_e64 s8, v119, -v12                            // 000000003BAC: D4140008 42021977
	s_delay_alu instid0(VALU_DEP_3)                            // 000000003BB4: BF870003
	v_cmp_gt_f32_e32 vcc_lo, v120, v28                         // 000000003BB8: 7C283978
	v_cmp_gt_f32_e64 s11, v119, -v13                           // 000000003BBC: D414000B 42021B77
	v_cmp_gt_f32_e64 s3, v120, v29                             // 000000003BC4: D4140003 02023B78
	v_cmp_lt_f32_e64 s5, -1.0, v82                             // 000000003BCC: D4110005 0202A4F3
	s_or_b32 s8, vcc_lo, s8                                    // 000000003BD4: 8C08086A
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v83                         // 000000003BD8: 7C22A6F3
	s_or_b32 s11, s3, s11                                      // 000000003BDC: 8C0B0B03
	s_and_b32 s3, s8, s4                                       // 000000003BE0: 8B030408
	s_wait_alu depctr_sa_sdst(0)                               // 000000003BE4: BF88FF9E
	s_and_b32 s4, s11, s7                                      // 000000003BE8: 8B04070B
	s_and_b32 s5, s5, s3                                       // 000000003BEC: 8B050305
	s_wait_alu depctr_sa_sdst(0)                               // 000000003BF0: BF88FF9E
	s_and_b32 s7, s4, vcc_lo                                   // 000000003BF4: 8B076A04
	s_wait_alu depctr_sa_sdst(0)                               // 000000003BF8: BF88FF9E
	s_or_b32 s5, s5, s7                                        // 000000003BFC: 8C050705
	s_wait_alu depctr_sa_sdst(0)                               // 000000003C00: BF88FF9E
	v_cndmask_b32_e64 v30, 0, 1, s5                            // 000000003C04: D501001E 00150280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000003C0C: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v30                            // 000000003C10: 7C9A3C80
	s_cbranch_vccz 4                                           // 000000003C14: BFA30004 <nearest+0x3c28>
	s_mov_b32 s6, 1.0                                          // 000000003C18: BE8600F2
	s_mov_b32 s10, 0x40c00000                                  // 000000003C1C: BE8A00FF 40C00000
	s_branch 20                                                // 000000003C24: BFA00014 <nearest+0x3c78>
	v_cndmask_b32_e64 v30, v90, -v12, s3                       // 000000003C28: D501001E 400E195A
	v_cndmask_b32_e64 v31, v89, -v13, s4                       // 000000003C30: D501001F 40121B59
	v_cndmask_b32_e64 v32, v88, v28, s3                        // 000000003C38: D5010020 000E3958
	v_cndmask_b32_e64 v33, v87, v29, s4                        // 000000003C40: D5010021 00123B57
	v_cndmask_b32_e64 v84, v84, v90, s3                        // 000000003C48: D5010054 000EB554
	v_cndmask_b32_e64 v85, v85, v89, s4                        // 000000003C50: D5010055 0012B355
	v_cndmask_b32_e64 v82, v82, v88, s3                        // 000000003C58: D5010052 000EB152
	v_cndmask_b32_e64 v83, v83, v87, s4                        // 000000003C60: D5010053 0012AF53
	v_dual_mov_b32 v89, v31 :: v_dual_mov_b32 v90, v30         // 000000003C68: CA10011F 595A011E
	v_dual_mov_b32 v87, v33 :: v_dual_mov_b32 v88, v32         // 000000003C70: CA100121 57580120
	s_wait_alu depctr_sa_sdst(0)                               // 000000003C78: BF88FF9E
	s_cmp_lt_f32 s6, 0.5                                       // 000000003C7C: BF41F006
	v_cmp_ge_f32_e64 s3, v119, -v14                            // 000000003C80: D4160003 42021D77
	v_cmp_ge_f32_e64 s4, v119, -v15                            // 000000003C88: D4160004 42021F77
	v_or_b32_e32 v31, 0xff6, v86                               // 000000003C90: 383EACFF 00000FF6
	s_cselect_b32 s5, -1, 0                                    // 000000003C98: 980580C1
	v_or_b32_e32 v32, 0xff7, v86                               // 000000003C9C: 3840ACFF 00000FF7
	s_wait_alu depctr_sa_sdst(0)                               // 000000003CA4: BF88FF9E
	s_and_b32 s2, s2, s5                                       // 000000003CA8: 8B020502
	s_wait_alu depctr_sa_sdst(0)                               // 000000003CAC: BF88FF9E
	s_and_b32 s3, s3, s2                                       // 000000003CB0: 8B030203
	s_and_b32 s5, s4, s2                                       // 000000003CB4: 8B050204
	s_wait_alu depctr_sa_sdst(0)                               // 000000003CB8: BF88FF9E
	s_or_b32 s2, s3, s5                                        // 000000003CBC: 8C020503
	s_wait_alu depctr_sa_sdst(0)                               // 000000003CC0: BF88FF9E
	v_cndmask_b32_e64 v30, 0, 1, s2                            // 000000003CC4: D501001E 00090280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000003CCC: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v30                            // 000000003CD0: 7C9A3C80
	v_cvt_f32_u32_e32 v30, v31                                 // 000000003CD4: 7E3C0D1F
	v_cvt_f32_u32_e32 v31, v32                                 // 000000003CD8: 7E3E0D20
	s_cbranch_vccz 30                                          // 000000003CDC: BFA3001E <nearest+0x3d58>
	v_cmp_gt_f32_e64 s7, v119, -v14                            // 000000003CE0: D4140007 42021D77
	s_delay_alu instid0(VALU_DEP_3)                            // 000000003CE8: BF870003
	v_cmp_gt_f32_e32 vcc_lo, v120, v30                         // 000000003CEC: 7C283D78
	v_cmp_gt_f32_e64 s8, v119, -v15                            // 000000003CF0: D4140008 42021F77
	v_cmp_gt_f32_e64 s2, v120, v31                             // 000000003CF8: D4140002 02023F78
	v_cmp_lt_f32_e64 s4, -1.0, v82                             // 000000003D00: D4110004 0202A4F3
	s_or_b32 s7, vcc_lo, s7                                    // 000000003D08: 8C07076A
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v83                         // 000000003D0C: 7C22A6F3
	s_or_b32 s8, s2, s8                                        // 000000003D10: 8C080802
	s_wait_alu depctr_sa_sdst(0)                               // 000000003D14: BF88FF9E
	s_and_b32 s2, s7, s3                                       // 000000003D18: 8B020307
	s_and_b32 s3, s8, s5                                       // 000000003D1C: 8B030508
	s_wait_alu depctr_sa_sdst(0)                               // 000000003D20: BF88FF9E
	s_and_b32 s4, s4, s2                                       // 000000003D24: 8B040204
	s_and_b32 s5, s3, vcc_lo                                   // 000000003D28: 8B056A03
	s_wait_alu depctr_sa_sdst(0)                               // 000000003D2C: BF88FF9E
	s_or_b32 s4, s4, s5                                        // 000000003D30: 8C040504
	s_wait_alu depctr_sa_sdst(0)                               // 000000003D34: BF88FF9E
	v_cndmask_b32_e64 v32, 0, 1, s4                            // 000000003D38: D5010020 00110280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000003D40: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v32                            // 000000003D44: 7C9A4080
	s_cbranch_vccz 2245                                        // 000000003D48: BFA308C5 <nearest+0x6060>
	s_mov_b32 s6, 1.0                                          // 000000003D4C: BE8600F2
	s_mov_b32 s10, 0x40e00000                                  // 000000003D50: BE8A00FF 40E00000
	s_wait_alu depctr_sa_sdst(0)                               // 000000003D58: BF88FF9E
	s_cmp_ngt_f32 s6, 0.5                                      // 000000003D5C: BF4BF006
	s_cbranch_scc1 2261                                        // 000000003D60: BFA208D5 <nearest+0x60b8>
	v_cmp_eq_f32_e32 vcc_lo, v89, v90                          // 000000003D64: 7C24B559
	v_cmp_lt_f32_e64 s2, v87, v88                              // 000000003D68: D4110002 0202B157
	v_cmp_lt_f32_e64 s3, v88, v87                              // 000000003D70: D4110003 0202AF58
	v_cmp_lt_f32_e64 s4, v89, v90                              // 000000003D78: D4110004 0202B559
	v_cmp_lt_f32_e64 s5, v90, v89                              // 000000003D80: D4110005 0202B35A
	s_mov_b32 s11, 0x76543210                                  // 000000003D88: BE8B00FF 76543210
	s_and_b32 s2, s2, vcc_lo                                   // 000000003D90: 8B026A02
	s_and_b32 s3, s3, vcc_lo                                   // 000000003D94: 8B036A03
	s_wait_alu depctr_sa_sdst(0)                               // 000000003D98: BF88FF9E
	s_or_b32 s2, s4, s2                                        // 000000003D9C: 8C020204
	s_or_b32 s3, s5, s3                                        // 000000003DA0: 8C030305
	s_wait_alu depctr_sa_sdst(0)                               // 000000003DA4: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000003DA8: 8D6A0002
	s_xor_b32 s2, s3, s9                                       // 000000003DAC: 8D020903
	s_wait_alu depctr_sa_sdst(0)                               // 000000003DB0: BF88FF9E
	v_cndmask_b32_e32 v32, v90, v89, vcc_lo                    // 000000003DB4: 0240B35A
	v_cndmask_b32_e32 v34, v88, v87, vcc_lo                    // 000000003DB8: 0244AF58
	v_cmp_eq_f32_e32 vcc_lo, v85, v84                          // 000000003DBC: 7C24A955
	v_cmp_lt_f32_e64 s3, v83, v82                              // 000000003DC0: D4110003 0202A553
	v_cmp_lt_f32_e64 s5, v85, v84                              // 000000003DC8: D4110005 0202A955
	v_cmp_lt_f32_e64 s4, v82, v83                              // 000000003DD0: D4110004 0202A752
	v_cndmask_b32_e64 v33, v89, v90, s2                        // 000000003DD8: D5010021 000AB559
	v_cndmask_b32_e64 v35, v87, v88, s2                        // 000000003DE0: D5010023 000AB157
	s_and_b32 s2, s3, vcc_lo                                   // 000000003DE8: 8B026A03
	v_mov_b32_e32 v37, v32                                     // 000000003DEC: 7E4A0320
	s_wait_alu depctr_sa_sdst(0)                               // 000000003DF0: BF88FF9E
	s_or_b32 s2, s5, s2                                        // 000000003DF4: 8C020205
	s_and_b32 s3, s4, vcc_lo                                   // 000000003DF8: 8B036A04
	s_wait_alu depctr_sa_sdst(0)                               // 000000003DFC: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000003E00: 8D6A0002
	v_mov_b32_e32 v38, v33                                     // 000000003E04: 7E4C0321
	s_wait_alu depctr_sa_sdst(0)                               // 000000003E08: BF88FF9E
	v_cndmask_b32_e32 v42, v82, v83, vcc_lo                    // 000000003E0C: 0254A752
	v_permlanex16_b32 v37, v37, s11, 0xfedcba98                // 000000003E10: D65C0025 03FC1725 FEDCBA98
	v_cmp_lt_f32_e64 s6, v84, v85                              // 000000003E1C: D4110006 0202AB54
	v_mov_b32_e32 v88, -1.0                                    // 000000003E24: 7EB002F3
	v_permlanex16_b32 v38, v38, s11, 0xfedcba98                // 000000003E28: D65C0026 03FC1726 FEDCBA98
	v_dual_mov_b32 v45, v42 :: v_dual_cndmask_b32 v36, v84, v85// 000000003E34: CA12012A 2D24AB54
	v_mov_b32_e32 v39, v34                                     // 000000003E3C: 7E4E0322
	v_cmp_eq_f32_e32 vcc_lo, v32, v37                          // 000000003E40: 7C244B20
	s_or_b32 s3, s6, s3                                        // 000000003E44: 8C030306
	v_cmp_gt_f32_e64 s6, v32, v37                              // 000000003E48: D4140006 02024B20
	s_wait_alu depctr_sa_sdst(0)                               // 000000003E50: BF88FF9E
	s_xor_b32 s2, s3, s9                                       // 000000003E54: 8D020903
	v_permlanex16_b32 v39, v39, s11, 0xfedcba98                // 000000003E58: D65C0027 03FC1727 FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 000000003E64: BF88FF9E
	v_cndmask_b32_e64 v41, v85, v84, s2                        // 000000003E68: D5010029 000AA955
	v_cndmask_b32_e64 v43, v83, v82, s2                        // 000000003E70: D501002B 000AA553
	v_cmp_eq_f32_e64 s3, v33, v38                              // 000000003E78: D4120003 02024D21
	v_cmp_gt_f32_e64 s7, v33, v38                              // 000000003E80: D4140007 02024D21
	v_cmp_gt_f32_e64 s4, v34, v39                              // 000000003E88: D4140004 02024F22
	v_mov_b32_e32 v44, v41                                     // 000000003E90: 7E580329
	v_mov_b32_e32 v46, v43                                     // 000000003E94: 7E5C032B
	v_permlanex16_b32 v45, v45, s11, 0xfedcba98                // 000000003E98: D65C002D 03FC172D FEDCBA98
	v_mov_b32_e32 v90, 0x7f800000                              // 000000003EA4: 7EB402FF 7F800000
	s_and_b32 s2, vcc_lo, s4                                   // 000000003EAC: 8B02046A
	v_permlanex16_b32 v44, v44, s11, 0xfedcba98                // 000000003EB0: D65C002C 03FC172C FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 000000003EBC: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 000000003EC0: 8C020206
	v_permlanex16_b32 v46, v46, s11, 0xfedcba98                // 000000003EC4: D65C002E 03FC172E FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 000000003ED0: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000003ED4: 8D6A0002
	v_mov_b32_e32 v84, 0x7f800000                              // 000000003ED8: 7EA802FF 7F800000
	s_wait_alu depctr_sa_sdst(0)                               // 000000003EE0: BF88FF9E
	v_cndmask_b32_e32 v34, v34, v39, vcc_lo                    // 000000003EE4: 02444F22
	v_mov_b32_e32 v40, v35                                     // 000000003EE8: 7E500323
	v_cndmask_b32_e32 v32, v32, v37, vcc_lo                    // 000000003EEC: 02404B20
	v_mov_b32_e32 v37, v36                                     // 000000003EF0: 7E4A0324
	v_cmp_neq_f32_e64 s4, v41, v44                             // 000000003EF4: D41D0004 02025929
	v_mov_b32_e32 v82, -1.0                                    // 000000003EFC: 7EA402F3
	v_permlanex16_b32 v40, v40, s11, 0xfedcba98                // 000000003F00: D65C0028 03FC1728 FEDCBA98
	v_permlanex16_b32 v37, v37, s11, 0xfedcba98                // 000000003F0C: D65C0025 03FC1725 FEDCBA98
	s_mov_b32 s14, 4                                           // 000000003F18: BE8E0084
	v_mov_b32_e32 v89, 0x7f800000                              // 000000003F1C: 7EB202FF 7F800000
	v_cmp_gt_f32_e64 s5, v35, v40                              // 000000003F24: D4140005 02025123
	v_mov_b32_e32 v87, -1.0                                    // 000000003F2C: 7EAE02F3
	v_cmp_neq_f32_e32 vcc_lo, v36, v37                         // 000000003F30: 7C3A4B24
	v_cmp_ngt_f32_e64 s6, v36, v37                             // 000000003F34: D41B0006 02024B24
	v_mov_b32_e32 v83, -1.0                                    // 000000003F3C: 7EA602F3
	s_and_b32 s3, s3, s5                                       // 000000003F40: 8B030503
	v_cmp_ngt_f32_e64 s5, v43, v46                             // 000000003F44: D41B0005 02025D2B
	s_wait_alu depctr_sa_sdst(0)                               // 000000003F4C: BF88FF9E
	s_or_b32 s3, s7, s3                                        // 000000003F50: 8C030307
	v_cmp_ngt_f32_e64 s7, v41, v44                             // 000000003F54: D41B0007 02025929
	s_wait_alu depctr_sa_sdst(0)                               // 000000003F5C: BF88FF9E
	s_xor_b32 s2, s3, s0                                       // 000000003F60: 8D020003
	v_cmp_ngt_f32_e64 s3, v42, v45                             // 000000003F64: D41B0003 02025B2A
	s_wait_alu depctr_sa_sdst(0)                               // 000000003F6C: BF88FF9E
	v_cndmask_b32_e64 v33, v33, v38, s2                        // 000000003F70: D5010021 000A4D21
	v_cndmask_b32_e64 v35, v35, v40, s2                        // 000000003F78: D5010023 000A5123
	v_mov_b32_e32 v85, 0x7f800000                              // 000000003F80: 7EAA02FF 7F800000
	s_or_b32 s2, vcc_lo, s3                                    // 000000003F88: 8C02036A
	s_or_b32 s3, s4, s5                                        // 000000003F8C: 8C030504
	s_wait_alu depctr_sa_sdst(0)                               // 000000003F90: BF88FF9E
	s_and_b32 s2, s6, s2                                       // 000000003F94: 8B020206
	s_and_b32 s3, s7, s3                                       // 000000003F98: 8B030307
	s_wait_alu depctr_sa_sdst(0)                               // 000000003F9C: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000003FA0: 8D6A0002
	s_xor_b32 s2, s3, s0                                       // 000000003FA4: 8D020003
	s_wait_alu depctr_sa_sdst(0)                               // 000000003FA8: BF88FF9E
	v_cndmask_b32_e32 v36, v36, v37, vcc_lo                    // 000000003FAC: 02484B24
	v_cndmask_b32_e32 v38, v42, v45, vcc_lo                    // 000000003FB0: 024C5B2A
	v_cmp_eq_f32_e32 vcc_lo, v33, v32                          // 000000003FB4: 7C244121
	v_cmp_lt_f32_e64 s3, v35, v34                              // 000000003FB8: D4110003 02024523
	v_cmp_lt_f32_e64 s4, v34, v35                              // 000000003FC0: D4110004 02024722
	v_cndmask_b32_e64 v37, v41, v44, s2                        // 000000003FC8: D5010025 000A5929
	v_cmp_lt_f32_e64 s5, v33, v32                              // 000000003FD0: D4110005 02024121
	v_cndmask_b32_e64 v39, v43, v46, s2                        // 000000003FD8: D5010027 000A5D2B
	v_cmp_lt_f32_e64 s2, v32, v33                              // 000000003FE0: D4110002 02024320
	s_and_b32 s3, vcc_lo, s3                                   // 000000003FE8: 8B03036A
	s_and_b32 s4, vcc_lo, s4                                   // 000000003FEC: 8B04046A
	s_wait_alu depctr_sa_sdst(0)                               // 000000003FF0: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 000000003FF4: 8C6A0305
	v_cmp_lt_f32_e64 s3, v39, v38                              // 000000003FF8: D4110003 02024D27
	s_or_b32 s2, s2, s4                                        // 000000004000: 8C020402
	s_wait_alu depctr_sa_sdst(0)                               // 000000004004: BF88FF9E
	v_cndmask_b32_e32 v40, v32, v33, vcc_lo                    // 000000004008: 02504320
	v_cndmask_b32_e64 v32, v32, v33, s2                        // 00000000400C: D5010020 000A4320
	v_cndmask_b32_e32 v33, v34, v35, vcc_lo                    // 000000004014: 02424722
	v_cmp_eq_f32_e32 vcc_lo, v37, v36                          // 000000004018: 7C244925
	v_cmp_lt_f32_e64 s4, v38, v39                              // 00000000401C: D4110004 02024F26
	v_cmp_lt_f32_e64 s5, v37, v36                              // 000000004024: D4110005 02024925
	v_cndmask_b32_e64 v34, v34, v35, s2                        // 00000000402C: D5010022 000A4722
	v_cmp_lt_f32_e64 s2, v36, v37                              // 000000004034: D4110002 02024B24
	s_and_b32 s3, vcc_lo, s3                                   // 00000000403C: 8B03036A
	s_and_b32 s4, vcc_lo, s4                                   // 000000004040: 8B04046A
	s_wait_alu depctr_sa_sdst(0)                               // 000000004044: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 000000004048: 8C6A0305
	s_or_b32 s2, s2, s4                                        // 00000000404C: 8C020402
	s_wait_alu depctr_sa_sdst(0)                               // 000000004050: BF88FF9E
	v_cndmask_b32_e32 v35, v37, v36, vcc_lo                    // 000000004054: 02464925
	v_cndmask_b32_e64 v36, v37, v36, s2                        // 000000004058: D5010024 000A4925
	v_cndmask_b32_e32 v37, v39, v38, vcc_lo                    // 000000004060: 024A4D27
	v_cndmask_b32_e64 v38, v39, v38, s2                        // 000000004064: D5010026 000A4D27
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 00000000406C: BF870224
	v_cmp_eq_f32_e32 vcc_lo, v40, v35                          // 000000004070: 7C244728
	v_cmp_lt_f32_e64 s5, v40, v35                              // 000000004074: D4110005 02024728
	v_cmp_lt_f32_e64 s3, v33, v37                              // 00000000407C: D4110003 02024B21
	v_cmp_eq_f32_e64 s2, v32, v36                              // 000000004084: D4120002 02024920
	v_cmp_lt_f32_e64 s4, v34, v38                              // 00000000408C: D4110004 02024D22
	v_cmp_lt_f32_e64 s6, v32, v36                              // 000000004094: D4110006 02024920
	s_and_b32 s3, vcc_lo, s3                                   // 00000000409C: 8B03036A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000040A0: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 0000000040A4: 8C6A0305
	s_and_b32 s2, s2, s4                                       // 0000000040A8: 8B020402
	s_wait_alu depctr_sa_sdst(0)                               // 0000000040AC: BF88FF9E
	v_dual_cndmask_b32 v39, v35, v40 :: v_dual_cndmask_b32 v42, v37, v33// 0000000040B0: CA525123 272A4325
	v_cndmask_b32_e32 v35, v40, v35, vcc_lo                    // 0000000040B8: 02464728
	v_cndmask_b32_e32 v33, v33, v37, vcc_lo                    // 0000000040BC: 02424B21
	s_or_b32 s2, s6, s2                                        // 0000000040C0: 8C020206
	s_delay_alu instid0(VALU_DEP_3)                            // 0000000040C4: BF870003
	v_dual_mov_b32 v40, v39 :: v_dual_mov_b32 v45, v42         // 0000000040C8: CA100127 282C012A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000040D0: BF88FF9E
	v_cndmask_b32_e64 v41, v36, v32, s2                        // 0000000040D4: D5010029 000A4124
	v_cndmask_b32_e64 v43, v38, v34, s2                        // 0000000040DC: D501002B 000A4526
	v_cndmask_b32_e64 v32, v32, v36, s2                        // 0000000040E4: D5010020 000A4920
	v_permlanex16_b32 v40, v40, s11, 0xfedcba98                // 0000000040EC: D65C0028 03FC1728 FEDCBA98
	v_permlanex16_b32 v45, v45, s11, 0xfedcba98                // 0000000040F8: D65C002D 03FC172D FEDCBA98
	v_cndmask_b32_e64 v34, v34, v38, s2                        // 000000004104: D5010022 000A4D22
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 00000000410C: BF870214
	v_dual_mov_b32 v37, v35 :: v_dual_mov_b32 v38, v32         // 000000004110: CA100123 25260120
	v_cmp_eq_f32_e32 vcc_lo, v39, v40                          // 000000004118: 7C245127
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 00000000411C: BF870224
	v_cmp_gt_f32_e64 s4, v42, v45                              // 000000004120: D4140004 02025B2A
	v_cmp_gt_f32_e64 s6, v39, v40                              // 000000004128: D4140006 02025127
	v_permlanex16_b32 v37, v37, s11, 0xfedcba98                // 000000004130: D65C0025 03FC1725 FEDCBA98
	v_mov_b32_e32 v44, v41                                     // 00000000413C: 7E580329
	v_permlanex16_b32 v38, v38, s11, 0xfedcba98                // 000000004140: D65C0026 03FC1726 FEDCBA98
	s_and_b32 s2, vcc_lo, s4                                   // 00000000414C: 8B02046A
	s_wait_alu depctr_sa_sdst(0)                               // 000000004150: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 000000004154: 8C020206
	v_permlanex16_b32 v44, v44, s11, 0xfedcba98                // 000000004158: D65C002C 03FC172C FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 000000004164: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000004168: 8D6A0002
	v_cmp_gt_f32_e64 s6, v35, v37                              // 00000000416C: D4140006 02024B23
	s_wait_alu depctr_sa_sdst(0)                               // 000000004174: BF88FF9E
	v_cndmask_b32_e32 v36, v39, v40, vcc_lo                    // 000000004178: 02485127
	v_dual_mov_b32 v46, v43 :: v_dual_mov_b32 v39, v33         // 00000000417C: CA10012B 2E260121
	v_mov_b32_e32 v40, v34                                     // 000000004184: 7E500322
	v_cmp_eq_f32_e64 s3, v41, v44                              // 000000004188: D4120003 02025929
	v_cmp_gt_f32_e64 s7, v41, v44                              // 000000004190: D4140007 02025929
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004198: BF870004
	v_permlanex16_b32 v46, v46, s11, 0xfedcba98                // 00000000419C: D65C002E 03FC172E FEDCBA98
	v_permlanex16_b32 v39, v39, s11, 0xfedcba98                // 0000000041A8: D65C0027 03FC1727 FEDCBA98
	v_permlanex16_b32 v40, v40, s11, 0xfedcba98                // 0000000041B4: D65C0028 03FC1728 FEDCBA98
	v_cndmask_b32_e32 v42, v42, v45, vcc_lo                    // 0000000041C0: 02545B2A
	v_cmp_eq_f32_e32 vcc_lo, v35, v37                          // 0000000041C4: 7C244B23
	v_cmp_gt_f32_e64 s5, v43, v46                              // 0000000041C8: D4140005 02025D2B
	v_cmp_gt_f32_e64 s4, v33, v39                              // 0000000041D0: D4140004 02024F21
	s_and_b32 s3, s3, s5                                       // 0000000041D8: 8B030503
	v_cmp_gt_f32_e64 s5, v34, v40                              // 0000000041DC: D4140005 02025122
	s_wait_alu depctr_sa_sdst(0)                               // 0000000041E4: BF88FF9E
	s_or_b32 s3, s7, s3                                        // 0000000041E8: 8C030307
	v_cmp_gt_f32_e64 s7, v32, v38                              // 0000000041EC: D4140007 02024D20
	s_wait_alu depctr_sa_sdst(0)                               // 0000000041F4: BF88FF9E
	s_xor_b32 s2, s3, s0                                       // 0000000041F8: 8D020003
	v_cmp_eq_f32_e64 s3, v32, v38                              // 0000000041FC: D4120003 02024D20
	s_wait_alu depctr_sa_sdst(0)                               // 000000004204: BF88FF9E
	v_cndmask_b32_e64 v41, v41, v44, s2                        // 000000004208: D5010029 000A5929
	v_cndmask_b32_e64 v43, v43, v46, s2                        // 000000004210: D501002B 000A5D2B
	s_and_b32 s2, vcc_lo, s4                                   // 000000004218: 8B02046A
	s_and_b32 s3, s3, s5                                       // 00000000421C: 8B030503
	s_wait_alu depctr_sa_sdst(0)                               // 000000004220: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 000000004224: 8C020206
	s_or_b32 s3, s7, s3                                        // 000000004228: 8C030307
	s_wait_alu depctr_sa_sdst(0)                               // 00000000422C: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000004230: 8D6A0002
	s_xor_b32 s2, s3, s0                                       // 000000004234: 8D020003
	s_wait_alu depctr_sa_sdst(0)                               // 000000004238: BF88FF9E
	v_cndmask_b32_e32 v35, v35, v37, vcc_lo                    // 00000000423C: 02464B23
	v_cndmask_b32_e32 v33, v33, v39, vcc_lo                    // 000000004240: 02424F21
	v_cmp_eq_f32_e32 vcc_lo, v41, v36                          // 000000004244: 7C244929
	v_cmp_lt_f32_e64 s3, v43, v42                              // 000000004248: D4110003 0202552B
	v_cmp_lt_f32_e64 s4, v42, v43                              // 000000004250: D4110004 0202572A
	v_cmp_lt_f32_e64 s5, v41, v36                              // 000000004258: D4110005 02024929
	v_cndmask_b32_e64 v32, v32, v38, s2                        // 000000004260: D5010020 000A4D20
	v_cndmask_b32_e64 v34, v34, v40, s2                        // 000000004268: D5010022 000A5122
	v_cmp_lt_f32_e64 s2, v36, v41                              // 000000004270: D4110002 02025324
	s_and_b32 s3, vcc_lo, s3                                   // 000000004278: 8B03036A
	s_and_b32 s4, vcc_lo, s4                                   // 00000000427C: 8B04046A
	s_wait_alu depctr_sa_sdst(0)                               // 000000004280: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 000000004284: 8C6A0305
	v_cmp_lt_f32_e64 s3, v34, v33                              // 000000004288: D4110003 02024322
	s_wait_alu depctr_sa_sdst(0)                               // 000000004290: BF88FF9E
	v_cndmask_b32_e32 v37, v36, v41, vcc_lo                    // 000000004294: 024A5324
	s_or_b32 s2, s2, s4                                        // 000000004298: 8C020402
	v_cndmask_b32_e32 v38, v42, v43, vcc_lo                    // 00000000429C: 024C572A
	v_cmp_eq_f32_e32 vcc_lo, v32, v35                          // 0000000042A0: 7C244720
	v_cmp_lt_f32_e64 s4, v33, v34                              // 0000000042A4: D4110004 02024521
	s_wait_alu depctr_sa_sdst(0)                               // 0000000042AC: BF88FF9E
	v_cndmask_b32_e64 v36, v36, v41, s2                        // 0000000042B0: D5010024 000A5324
	v_cmp_lt_f32_e64 s5, v32, v35                              // 0000000042B8: D4110005 02024720
	v_cndmask_b32_e64 v39, v42, v43, s2                        // 0000000042C0: D5010027 000A572A
	v_cmp_lt_f32_e64 s2, v35, v32                              // 0000000042C8: D4110002 02024123
	s_and_b32 s3, vcc_lo, s3                                   // 0000000042D0: 8B03036A
	s_and_b32 s4, vcc_lo, s4                                   // 0000000042D4: 8B04046A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000042D8: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 0000000042DC: 8C6A0305
	v_permlanex16_b32 v36, v36, s11, 0xfedcba98                // 0000000042E0: D65C0024 03FC1724 FEDCBA98
	s_or_b32 s2, s2, s4                                        // 0000000042EC: 8C020402
	v_permlanex16_b32 v39, v39, s11, 0xfedcba98                // 0000000042F0: D65C0027 03FC1727 FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 0000000042FC: BF88FF9E
	v_cndmask_b32_e32 v40, v35, v32, vcc_lo                    // 000000004300: 02504123
	v_cndmask_b32_e64 v32, v35, v32, s2                        // 000000004304: D5010020 000A4123
	v_permlanex16_b32 v37, v37, s11, 0xfedcba98                // 00000000430C: D65C0025 03FC1725 FEDCBA98
	v_permlanex16_b32 v38, v38, s11, 0xfedcba98                // 000000004318: D65C0026 03FC1726 FEDCBA98
	v_cndmask_b32_e32 v35, v33, v34, vcc_lo                    // 000000004324: 02464521
	v_cndmask_b32_e64 v33, v33, v34, s2                        // 000000004328: D5010021 000A4521
	v_cmp_eq_f32_e32 vcc_lo, v81, v36                          // 000000004330: 7C244951
	v_cmp_lt_f32_e64 s4, v80, v39                              // 000000004334: D4110004 02024F50
	v_cmp_eq_f32_e64 s3, v75, v37                              // 00000000433C: D4120003 02024B4B
	v_cmp_lt_f32_e64 s5, v74, v38                              // 000000004344: D4110005 02024D4A
	v_cmp_lt_f32_e64 s2, v81, v36                              // 00000000434C: D4110002 02024951
	v_permlanex16_b32 v32, v32, s11, 0xfedcba98                // 000000004354: D65C0020 03FC1720 FEDCBA98
	v_permlanex16_b32 v40, v40, s11, 0xfedcba98                // 000000004360: D65C0028 03FC1728 FEDCBA98
	v_permlanex16_b32 v33, v33, s11, 0xfedcba98                // 00000000436C: D65C0021 03FC1721 FEDCBA98
	v_permlanex16_b32 v35, v35, s11, 0xfedcba98                // 000000004378: D65C0023 03FC1723 FEDCBA98
	v_cmp_lt_f32_e64 s6, v75, v37                              // 000000004384: D4110006 02024B4B
	s_and_b32 s4, vcc_lo, s4                                   // 00000000438C: 8B04046A
	s_and_b32 s8, s3, s5                                       // 000000004390: 8B080503
	s_wait_alu depctr_sa_sdst(0)                               // 000000004394: BF88FF9E
	s_or_b32 vcc_lo, s2, s4                                    // 000000004398: 8C6A0402
	v_cmp_eq_f32_e64 s3, v76, v32                              // 00000000439C: D4120003 0202414C
	v_cmp_eq_f32_e64 s4, v78, v40                              // 0000000043A4: D4120004 0202514E
	v_cmp_lt_f32_e64 s5, v77, v33                              // 0000000043AC: D4110005 0202434D
	v_cmp_lt_f32_e64 s7, v79, v35                              // 0000000043B4: D4110007 0202474F
	v_cmp_lt_f32_e64 s2, v76, v32                              // 0000000043BC: D4110002 0202414C
	s_or_b32 s6, s6, s8                                        // 0000000043C4: 8C060806
	v_cmp_lt_f32_e64 s8, v78, v40                              // 0000000043C8: D4110008 0202514E
	s_and_b32 s3, s3, s5                                       // 0000000043D0: 8B030503
	s_and_b32 s4, s4, s7                                       // 0000000043D4: 8B040704
	s_wait_alu depctr_sa_sdst(0)                               // 0000000043D8: BF88FF9E
	s_or_b32 s2, s2, s3                                        // 0000000043DC: 8C020302
	v_cndmask_b32_e32 v34, v36, v81, vcc_lo                    // 0000000043E0: 0244A324
	s_or_b32 s3, s8, s4                                        // 0000000043E4: 8C030408
	v_cndmask_b32_e64 v36, v37, v75, s6                        // 0000000043E8: D5010024 001A9725
	s_wait_alu depctr_sa_sdst(0)                               // 0000000043F0: BF88FF9E
	v_cndmask_b32_e64 v32, v32, v76, s2                        // 0000000043F4: D5010020 000A9920
	v_cndmask_b32_e64 v37, v40, v78, s3                        // 0000000043FC: D5010025 000E9D28
	v_cndmask_b32_e64 v33, v33, v77, s2                        // 000000004404: D5010021 000A9B21
	v_cndmask_b32_e64 v35, v35, v79, s3                        // 00000000440C: D5010023 000E9F23
	v_cndmask_b32_e32 v39, v39, v80, vcc_lo                    // 000000004414: 024EA127
	v_cmp_eq_f32_e32 vcc_lo, v68, v32                          // 000000004418: 7C244144
	v_cmp_eq_f32_e64 s2, v69, v37                              // 00000000441C: D4120002 02024B45
	v_cmp_lt_f32_e64 s3, v67, v33                              // 000000004424: D4110003 02024343
	v_cmp_lt_f32_e64 s4, v66, v35                              // 00000000442C: D4110004 02024742
	v_cmp_lt_f32_e64 s5, v68, v32                              // 000000004434: D4110005 02024144
	v_cndmask_b32_e64 v38, v38, v74, s6                        // 00000000443C: D5010026 001A9526
	v_cmp_lt_f32_e64 s6, v69, v37                              // 000000004444: D4110006 02024B45
	s_and_b32 s3, vcc_lo, s3                                   // 00000000444C: 8B03036A
	s_and_b32 s2, s2, s4                                       // 000000004450: 8B020402
	s_wait_alu depctr_sa_sdst(0)                               // 000000004454: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 000000004458: 8C6A0305
	v_cmp_eq_f32_e64 s3, v73, v36                              // 00000000445C: D4120003 02024949
	s_or_b32 s2, s6, s2                                        // 000000004464: 8C020206
	v_cmp_lt_f32_e64 s5, v70, v38                              // 000000004468: D4110005 02024D46
	s_wait_alu depctr_sa_sdst(0)                               // 000000004470: BF88FF9E
	v_cndmask_b32_e64 v41, v37, v69, s2                        // 000000004474: D5010029 000A8B25
	v_cndmask_b32_e64 v43, v35, v66, s2                        // 00000000447C: D501002B 000A8523
	v_cndmask_b32_e64 v37, v69, v37, s2                        // 000000004484: D5010025 000A4B45
	v_cndmask_b32_e64 v35, v66, v35, s2                        // 00000000448C: D5010023 000A4742
	v_cmp_lt_f32_e64 s2, v73, v36                              // 000000004494: D4110002 02024949
	s_and_b32 s3, s3, s5                                       // 00000000449C: 8B030503
	v_cndmask_b32_e32 v40, v32, v68, vcc_lo                    // 0000000044A0: 02508920
	v_cndmask_b32_e32 v42, v33, v67, vcc_lo                    // 0000000044A4: 02548721
	v_cndmask_b32_e32 v32, v68, v32, vcc_lo                    // 0000000044A8: 02404144
	s_wait_alu depctr_sa_sdst(0)                               // 0000000044AC: BF88FF9E
	s_or_b32 s2, s2, s3                                        // 0000000044B0: 8C020302
	v_cndmask_b32_e32 v33, v67, v33, vcc_lo                    // 0000000044B4: 02424343
	s_wait_alu depctr_sa_sdst(0)                               // 0000000044B8: BF88FF9E
	v_cndmask_b32_e64 v45, v36, v73, s2                        // 0000000044BC: D501002D 000A9324
	v_cndmask_b32_e64 v47, v38, v70, s2                        // 0000000044C4: D501002F 000A8D26
	v_cndmask_b32_e64 v36, v73, v36, s2                        // 0000000044CC: D5010024 000A4949
	v_cndmask_b32_e64 v38, v70, v38, s2                        // 0000000044D4: D5010026 000A4D46
	v_cmp_eq_f32_e32 vcc_lo, v71, v34                          // 0000000044DC: 7C244547
	v_cmp_eq_f32_e64 s3, v41, v45                              // 0000000044E0: D4120003 02025B29
	v_cmp_lt_f32_e64 s5, v43, v47                              // 0000000044E8: D4110005 02025F2B
	v_cmp_lt_f32_e64 s2, v41, v45                              // 0000000044F0: D4110002 02025B29
	v_cmp_lt_f32_e64 s4, v72, v39                              // 0000000044F8: D4110004 02024F48
	v_cmp_lt_f32_e64 s6, v71, v34                              // 000000004500: D4110006 02024547
	s_and_b32 s3, s3, s5                                       // 000000004508: 8B030503
	v_cmp_lt_f32_e64 s5, v35, v38                              // 00000000450C: D4110005 02024D23
	s_wait_alu depctr_sa_sdst(0)                               // 000000004514: BF88FF9E
	s_or_b32 s2, s2, s3                                        // 000000004518: 8C020302
	v_cmp_eq_f32_e64 s3, v37, v36                              // 00000000451C: D4120003 02024925
	s_wait_alu depctr_sa_sdst(0)                               // 000000004524: BF88FF9E
	v_cndmask_b32_e64 v49, v45, v41, s2                        // 000000004528: D5010031 000A532D
	v_cndmask_b32_e64 v51, v47, v43, s2                        // 000000004530: D5010033 000A572F
	v_cndmask_b32_e64 v41, v41, v45, s2                        // 000000004538: D5010029 000A5B29
	v_cndmask_b32_e64 v43, v43, v47, s2                        // 000000004540: D501002B 000A5F2B
	v_cmp_lt_f32_e64 s2, v37, v36                              // 000000004548: D4110002 02024925
	s_and_b32 s4, vcc_lo, s4                                   // 000000004550: 8B04046A
	s_and_b32 s3, s3, s5                                       // 000000004554: 8B030503
	s_wait_alu depctr_sa_sdst(0)                               // 000000004558: BF88FF9E
	s_or_b32 vcc_lo, s6, s4                                    // 00000000455C: 8C6A0406
	v_mov_b32_e32 v52, v49                                     // 000000004560: 7E680331
	s_or_b32 s2, s2, s3                                        // 000000004564: 8C020302
	s_wait_alu depctr_sa_sdst(0)                               // 000000004568: BF88FF9E
	v_cndmask_b32_e32 v46, v39, v72, vcc_lo                    // 00000000456C: 025C9127
	v_cndmask_b32_e64 v45, v36, v37, s2                        // 000000004570: D501002D 000A4B24
	v_cndmask_b32_e64 v36, v37, v36, s2                        // 000000004578: D5010024 000A4925
	v_cndmask_b32_e32 v44, v34, v71, vcc_lo                    // 000000004580: 02588F22
	v_dual_cndmask_b32 v34, v71, v34 :: v_dual_cndmask_b32 v39, v72, v39// 000000004584: CA524547 22264F48
	v_cmp_lt_f32_e64 s4, v42, v46                              // 00000000458C: D4110004 02025D2A
	v_mov_b32_e32 v54, v51                                     // 000000004594: 7E6C0333
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004598: BF870004
	v_cmp_eq_f32_e32 vcc_lo, v40, v44                          // 00000000459C: 7C245928
	v_cmp_lt_f32_e64 s6, v40, v44                              // 0000000045A0: D4110006 02025928
	v_permlanex16_b32 v52, v52, s11, 0xfedcba98                // 0000000045A8: D65C0034 03FC1734 FEDCBA98
	v_cndmask_b32_e64 v47, v38, v35, s2                        // 0000000045B4: D501002F 000A4726
	v_permlanex16_b32 v54, v54, s11, 0xfedcba98                // 0000000045BC: D65C0036 03FC1736 FEDCBA98
	s_and_b32 s4, vcc_lo, s4                                   // 0000000045C8: 8B04046A
	v_cndmask_b32_e64 v35, v35, v38, s2                        // 0000000045CC: D5010023 000A4D23
	s_wait_alu depctr_sa_sdst(0)                               // 0000000045D4: BF88FF9E
	s_or_b32 vcc_lo, s6, s4                                    // 0000000045D8: 8C6A0406
	v_cmp_lt_f32_e64 s4, v33, v39                              // 0000000045DC: D4110004 02024F21
	s_wait_alu depctr_sa_sdst(0)                               // 0000000045E4: BF88FF9E
	v_cndmask_b32_e32 v48, v44, v40, vcc_lo                    // 0000000045E8: 0260512C
	v_cndmask_b32_e32 v50, v46, v42, vcc_lo                    // 0000000045EC: 0264552E
	v_cndmask_b32_e32 v40, v40, v44, vcc_lo                    // 0000000045F0: 02505928
	v_cndmask_b32_e32 v42, v42, v46, vcc_lo                    // 0000000045F4: 02545D2A
	v_cmp_eq_f32_e32 vcc_lo, v32, v34                          // 0000000045F8: 7C244520
	v_cmp_lt_f32_e64 s6, v32, v34                              // 0000000045FC: D4110006 02024520
	v_mov_b32_e32 v53, v50                                     // 000000004604: 7E6A0332
	v_mov_b32_e32 v37, v40                                     // 000000004608: 7E4A0328
	v_cmp_eq_f32_e64 s3, v49, v52                              // 00000000460C: D4120003 02026931
	s_and_b32 s4, vcc_lo, s4                                   // 000000004614: 8B04046A
	v_cmp_gt_f32_e64 s5, v51, v54                              // 000000004618: D4140005 02026D33
	s_wait_alu depctr_sa_sdst(0)                               // 000000004620: BF88FF9E
	s_or_b32 vcc_lo, s6, s4                                    // 000000004624: 8C6A0406
	v_permlanex16_b32 v53, v53, s11, 0xfedcba98                // 000000004628: D65C0035 03FC1735 FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 000000004634: BF88FF9E
	v_cndmask_b32_e32 v44, v34, v32, vcc_lo                    // 000000004638: 02584122
	v_cndmask_b32_e32 v32, v32, v34, vcc_lo                    // 00000000463C: 02404520
	v_mov_b32_e32 v34, v48                                     // 000000004640: 7E440330
	v_dual_cndmask_b32 v46, v39, v33 :: v_dual_cndmask_b32 v33, v33, v39// 000000004644: CA524327 2E204F21
	v_cmp_gt_f32_e64 s4, v50, v53                              // 00000000464C: D4140004 02026B32
	v_mov_b32_e32 v39, v42                                     // 000000004654: 7E4E032A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004658: BF870004
	v_permlanex16_b32 v34, v34, s11, 0xfedcba98                // 00000000465C: D65C0022 03FC1722 FEDCBA98
	v_permlanex16_b32 v37, v37, s11, 0xfedcba98                // 000000004668: D65C0025 03FC1725 FEDCBA98
	v_cmp_gt_f32_e64 s7, v49, v52                              // 000000004674: D4140007 02026931
	s_and_b32 s3, s3, s5                                       // 00000000467C: 8B030503
	v_permlanex16_b32 v39, v39, s11, 0xfedcba98                // 000000004680: D65C0027 03FC1727 FEDCBA98
	v_cmp_eq_f32_e32 vcc_lo, v48, v34                          // 00000000468C: 7C244530
	v_cmp_gt_f32_e64 s6, v48, v34                              // 000000004690: D4140006 02024530
	s_wait_alu depctr_sa_sdst(0)                               // 000000004698: BF88FF9E
	s_or_b32 s3, s7, s3                                        // 00000000469C: 8C030307
	v_mov_b32_e32 v38, v41                                     // 0000000046A0: 7E4C0329
	s_and_b32 s2, vcc_lo, s4                                   // 0000000046A4: 8B02046A
	v_cmp_gt_f32_e64 s4, v42, v39                              // 0000000046A8: D4140004 02024F2A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000046B0: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 0000000046B4: 8C020206
	v_cmp_gt_f32_e64 s6, v40, v37                              // 0000000046B8: D4140006 02024B28
	s_wait_alu depctr_sa_sdst(0)                               // 0000000046C0: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 0000000046C4: 8D6A0002
	s_xor_b32 s2, s3, s0                                       // 0000000046C8: 8D020003
	s_wait_alu depctr_sa_sdst(0)                               // 0000000046CC: BF88FF9E
	v_cndmask_b32_e32 v34, v48, v34, vcc_lo                    // 0000000046D0: 02444530
	v_cndmask_b32_e32 v50, v50, v53, vcc_lo                    // 0000000046D4: 02646B32
	v_cmp_eq_f32_e32 vcc_lo, v40, v37                          // 0000000046D8: 7C244B28
	v_mov_b32_e32 v48, v43                                     // 0000000046DC: 7E60032B
	v_cndmask_b32_e64 v49, v49, v52, s2                        // 0000000046E0: D5010031 000A6931
	v_cndmask_b32_e64 v51, v51, v54, s2                        // 0000000046E8: D5010033 000A6D33
	v_permlanex16_b32 v38, v38, s11, 0xfedcba98                // 0000000046F0: D65C0026 03FC1726 FEDCBA98
	s_and_b32 s2, vcc_lo, s4                                   // 0000000046FC: 8B02046A
	v_permlanex16_b32 v48, v48, s11, 0xfedcba98                // 000000004700: D65C0030 03FC1730 FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 00000000470C: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 000000004710: 8C020206
	v_mov_b32_e32 v53, v46                                     // 000000004714: 7E6A032E
	s_wait_alu depctr_sa_sdst(0)                               // 000000004718: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 00000000471C: 8D6A0002
	v_cmp_eq_f32_e64 s3, v41, v38                              // 000000004720: D4120003 02024D29
	s_wait_alu depctr_sa_sdst(0)                               // 000000004728: BF88FF9E
	v_cndmask_b32_e32 v37, v40, v37, vcc_lo                    // 00000000472C: 024A4B28
	v_mov_b32_e32 v40, v44                                     // 000000004730: 7E50032C
	v_cmp_gt_f32_e64 s5, v43, v48                              // 000000004734: D4140005 0202612B
	v_permlanex16_b32 v53, v53, s11, 0xfedcba98                // 00000000473C: D65C0035 03FC1735 FEDCBA98
	v_cmp_gt_f32_e64 s7, v41, v38                              // 000000004748: D4140007 02024D29
	v_cndmask_b32_e32 v39, v42, v39, vcc_lo                    // 000000004750: 024E4F2A
	v_permlanex16_b32 v40, v40, s11, 0xfedcba98                // 000000004754: D65C0028 03FC1728 FEDCBA98
	s_and_b32 s3, s3, s5                                       // 000000004760: 8B030503
	v_cmp_gt_f32_e64 s4, v46, v53                              // 000000004764: D4140004 02026B2E
	s_wait_alu depctr_sa_sdst(0)                               // 00000000476C: BF88FF9E
	s_or_b32 s3, s7, s3                                        // 000000004770: 8C030307
	v_mov_b32_e32 v52, v45                                     // 000000004774: 7E68032D
	v_cmp_eq_f32_e32 vcc_lo, v44, v40                          // 000000004778: 7C24512C
	v_cmp_gt_f32_e64 s6, v44, v40                              // 00000000477C: D4140006 0202512C
	s_wait_alu depctr_sa_sdst(0)                               // 000000004784: BF88FF9E
	s_xor_b32 s2, s3, s0                                       // 000000004788: 8D020003
	v_mov_b32_e32 v54, v47                                     // 00000000478C: 7E6C032F
	s_wait_alu depctr_sa_sdst(0)                               // 000000004790: BF88FF9E
	v_cndmask_b32_e64 v38, v41, v38, s2                        // 000000004794: D5010026 000A4D29
	v_cndmask_b32_e64 v41, v43, v48, s2                        // 00000000479C: D5010029 000A612B
	s_and_b32 s2, vcc_lo, s4                                   // 0000000047A4: 8B02046A
	v_permlanex16_b32 v52, v52, s11, 0xfedcba98                // 0000000047A8: D65C0034 03FC1734 FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 0000000047B4: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 0000000047B8: 8C020206
	v_permlanex16_b32 v54, v54, s11, 0xfedcba98                // 0000000047BC: D65C0036 03FC1736 FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 0000000047C8: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 0000000047CC: 8D6A0002
	v_mov_b32_e32 v42, v32                                     // 0000000047D0: 7E540320
	s_wait_alu depctr_sa_sdst(0)                               // 0000000047D4: BF88FF9E
	v_cndmask_b32_e32 v40, v44, v40, vcc_lo                    // 0000000047D8: 0250512C
	v_dual_mov_b32 v44, v33 :: v_dual_mov_b32 v43, v36         // 0000000047DC: CA100121 2C2A0124
	v_mov_b32_e32 v48, v35                                     // 0000000047E4: 7E600323
	v_cmp_eq_f32_e64 s3, v45, v52                              // 0000000047E8: D4120003 0202692D
	v_cmp_gt_f32_e64 s5, v47, v54                              // 0000000047F0: D4140005 02026D2F
	v_cmp_gt_f32_e64 s7, v45, v52                              // 0000000047F8: D4140007 0202692D
	v_permlanex16_b32 v42, v42, s11, 0xfedcba98                // 000000004800: D65C002A 03FC172A FEDCBA98
	v_permlanex16_b32 v44, v44, s11, 0xfedcba98                // 00000000480C: D65C002C 03FC172C FEDCBA98
	v_permlanex16_b32 v43, v43, s11, 0xfedcba98                // 000000004818: D65C002B 03FC172B FEDCBA98
	v_permlanex16_b32 v48, v48, s11, 0xfedcba98                // 000000004824: D65C0030 03FC1730 FEDCBA98
	s_and_b32 s3, s3, s5                                       // 000000004830: 8B030503
	v_cndmask_b32_e32 v46, v46, v53, vcc_lo                    // 000000004834: 025C6B2E
	s_wait_alu depctr_sa_sdst(0)                               // 000000004838: BF88FF9E
	s_or_b32 s3, s7, s3                                        // 00000000483C: 8C030307
	v_cmp_eq_f32_e32 vcc_lo, v32, v42                          // 000000004840: 7C245520
	v_cmp_gt_f32_e64 s4, v33, v44                              // 000000004844: D4140004 02025921
	s_wait_alu depctr_sa_sdst(0)                               // 00000000484C: BF88FF9E
	s_xor_b32 s2, s3, s0                                       // 000000004850: 8D020003
	v_cmp_eq_f32_e64 s3, v36, v43                              // 000000004854: D4120003 02025724
	v_cmp_gt_f32_e64 s5, v35, v48                              // 00000000485C: D4140005 02026123
	v_cmp_gt_f32_e64 s6, v32, v42                              // 000000004864: D4140006 02025520
	v_cmp_gt_f32_e64 s7, v36, v43                              // 00000000486C: D4140007 02025724
	s_wait_alu depctr_sa_sdst(0)                               // 000000004874: BF88FF9E
	v_cndmask_b32_e64 v45, v45, v52, s2                        // 000000004878: D501002D 000A692D
	v_cndmask_b32_e64 v47, v47, v54, s2                        // 000000004880: D501002F 000A6D2F
	s_and_b32 s2, vcc_lo, s4                                   // 000000004888: 8B02046A
	s_and_b32 s3, s3, s5                                       // 00000000488C: 8B030503
	s_wait_alu depctr_sa_sdst(0)                               // 000000004890: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 000000004894: 8C020206
	s_or_b32 s3, s7, s3                                        // 000000004898: 8C030307
	s_wait_alu depctr_sa_sdst(0)                               // 00000000489C: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 0000000048A0: 8D6A0002
	s_xor_b32 s2, s3, s0                                       // 0000000048A4: 8D020003
	s_wait_alu depctr_sa_sdst(0)                               // 0000000048A8: BF88FF9E
	v_dual_cndmask_b32 v32, v32, v42 :: v_dual_cndmask_b32 v33, v33, v44// 0000000048AC: CA525520 20205921
	v_cmp_eq_f32_e32 vcc_lo, v49, v34                          // 0000000048B4: 7C244531
	v_cmp_lt_f32_e64 s4, v50, v51                              // 0000000048B8: D4110004 02026732
	v_cndmask_b32_e64 v36, v36, v43, s2                        // 0000000048C0: D5010024 000A5724
	v_cmp_lt_f32_e64 s3, v51, v50                              // 0000000048C8: D4110003 02026533
	v_cndmask_b32_e64 v35, v35, v48, s2                        // 0000000048D0: D5010023 000A6123
	v_cmp_lt_f32_e64 s2, v34, v49                              // 0000000048D8: D4110002 02026322
	v_cmp_lt_f32_e64 s5, v49, v34                              // 0000000048E0: D4110005 02024531
	s_and_b32 s4, vcc_lo, s4                                   // 0000000048E8: 8B04046A
	s_and_b32 s3, vcc_lo, s3                                   // 0000000048EC: 8B03036A
	v_cmp_lt_f32_e64 s6, v33, v35                              // 0000000048F0: D4110006 02024721
	s_wait_alu depctr_sa_sdst(0)                               // 0000000048F8: BF88FF9E
	s_or_b32 s2, s2, s4                                        // 0000000048FC: 8C020402
	s_or_b32 vcc_lo, s5, s3                                    // 000000004900: 8C6A0305
	s_wait_alu depctr_sa_sdst(0)                               // 000000004904: BF88FF9E
	v_cndmask_b32_e64 v69, v34, v49, s2                        // 000000004908: D5010045 000A6322
	v_cndmask_b32_e64 v66, v50, v51, s2                        // 000000004910: D5010042 000A6732
	v_cmp_eq_f32_e64 s2, v38, v37                              // 000000004918: D4120002 02024B26
	v_cmp_lt_f32_e64 s3, v41, v39                              // 000000004920: D4110003 02024F29
	v_cmp_lt_f32_e64 s4, v39, v41                              // 000000004928: D4110004 02025327
	v_cmp_eq_f32_e64 s5, v36, v32                              // 000000004930: D4120005 02024124
	v_cndmask_b32_e32 v68, v34, v49, vcc_lo                    // 000000004938: 02886322
	v_cndmask_b32_e32 v67, v50, v51, vcc_lo                    // 00000000493C: 02866732
	s_and_b32 s8, s2, s3                                       // 000000004940: 8B080302
	v_cmp_lt_f32_e64 s3, v32, v36                              // 000000004944: D4110003 02024920
	s_and_b32 s4, s2, s4                                       // 00000000494C: 8B040402
	s_and_b32 s2, s5, s6                                       // 000000004950: 8B020605
	v_cmp_lt_f32_e32 vcc_lo, v38, v37                          // 000000004954: 7C224B26
	v_cmp_lt_f32_e64 s7, v37, v38                              // 000000004958: D4110007 02024D25
	s_wait_alu depctr_sa_sdst(0)                               // 000000004960: BF88FF9E
	s_or_b32 s2, s3, s2                                        // 000000004964: 8C020203
	v_cmp_lt_f32_e64 s3, v47, v46                              // 000000004968: D4110003 02025D2F
	s_wait_alu depctr_sa_sdst(0)                               // 000000004970: BF88FF9E
	v_cndmask_b32_e64 v75, v32, v36, s2                        // 000000004974: D501004B 000A4920
	v_cndmask_b32_e64 v74, v33, v35, s2                        // 00000000497C: D501004A 000A4721
	s_or_b32 vcc_lo, vcc_lo, s8                                // 000000004984: 8C6A086A
	s_or_b32 s2, s7, s4                                        // 000000004988: 8C020407
	s_wait_alu depctr_sa_sdst(0)                               // 00000000498C: BF88FF9E
	v_cndmask_b32_e32 v71, v37, v38, vcc_lo                    // 000000004990: 028E4D25
	v_cndmask_b32_e64 v73, v37, v38, s2                        // 000000004994: D5010049 000A4D25
	v_dual_max_num_f32 v34, v75, v75 :: v_dual_max_num_f32 v37, v74, v74// 00000000499C: CA94974B 2224954A
	v_cndmask_b32_e64 v70, v39, v41, s2                        // 0000000049A4: D5010046 000A5327
	v_cmp_eq_f32_e64 s2, v45, v40                              // 0000000049AC: D4120002 0202512D
	v_cndmask_b32_e32 v72, v39, v41, vcc_lo                    // 0000000049B4: 02905327
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000049B8: BF870004
	v_dual_max_num_f32 v34, 0xff800000, v34 :: v_dual_max_num_f32 v37, 0xff800000, v37// 0000000049BC: CA9444FF 22244AFF FF800000
	v_cmp_lt_f32_e32 vcc_lo, v45, v40                          // 0000000049C8: 7C22512D
	v_cmp_lt_f32_e64 s4, v46, v47                              // 0000000049CC: D4110004 02025F2E
	s_and_b32 s3, s2, s3                                       // 0000000049D4: 8B030302
	v_cndmask_b32_e64 v34, v34, 0xff800000, s1                 // 0000000049D8: D5010022 0005FF22 FF800000
	v_cndmask_b32_e64 v37, v37, 0xff800000, s1                 // 0000000049E4: D5010025 0005FF25 FF800000
	v_cmp_lt_f32_e64 s6, v40, v45                              // 0000000049F0: D4110006 02025B28
	s_wait_alu depctr_sa_sdst(0)                               // 0000000049F8: BF88FF9E
	s_or_b32 vcc_lo, vcc_lo, s3                                // 0000000049FC: 8C6A036A
	s_and_b32 s2, s2, s4                                       // 000000004A00: 8B020402
	v_dual_mov_b32 v38, v34 :: v_dual_mov_b32 v39, v37         // 000000004A04: CA100122 26260125
	s_wait_alu depctr_sa_sdst(0)                               // 000000004A0C: BF88FF9E
	v_dual_cndmask_b32 v76, v40, v45 :: v_dual_cndmask_b32 v77, v46, v47// 000000004A10: CA525B28 4C4C5F2E
	v_cmp_lt_f32_e32 vcc_lo, v35, v33                          // 000000004A18: 7C224323
	s_delay_alu instid0(VALU_DEP_3)                            // 000000004A1C: BF870003
	v_permlanex16_b32 v38, v38, s11, 0xfedcba98                // 000000004A20: D65C0026 03FC1726 FEDCBA98
	v_permlanex16_b32 v39, v39, s11, 0xfedcba98                // 000000004A2C: D65C0027 03FC1727 FEDCBA98
	v_cmp_lt_f32_e64 s3, v36, v32                              // 000000004A38: D4110003 02024124
	s_or_b32 s2, s6, s2                                        // 000000004A40: 8C020206
	v_dual_max_num_f32 v38, v38, v38 :: v_dual_max_num_f32 v39, v39, v39// 000000004A44: CA944D26 26264F27
	s_wait_alu depctr_sa_sdst(0)                               // 000000004A4C: BF88FF9E
	v_cndmask_b32_e64 v78, v40, v45, s2                        // 000000004A50: D501004E 000A5B28
	v_cndmask_b32_e64 v79, v46, v47, s2                        // 000000004A58: D501004F 000A5F2E
	s_and_b32 s2, s5, vcc_lo                                   // 000000004A60: 8B026A05
	s_wait_alu depctr_sa_sdst(0)                               // 000000004A64: BF88FF9E
	s_or_b32 vcc_lo, s3, s2                                    // 000000004A68: 8C6A0203
	s_wait_alu depctr_sa_sdst(0)                               // 000000004A6C: BF88FF9E
	v_dual_cndmask_b32 v81, v32, v36 :: v_dual_cndmask_b32 v80, v33, v35// 000000004A70: CA524920 51504721
	v_dual_max_num_f32 v32, v34, v38 :: v_dual_max_num_f32 v33, v37, v39// 000000004A78: CA944D22 20204F25
	s_branch 4                                                 // 000000004A80: BFA00004 <nearest+0x4a94>
	s_add_co_i32 s14, s14, -1                                  // 000000004A84: 810EC10E
	s_wait_alu depctr_sa_sdst(0)                               // 000000004A88: BF88FF9E
	s_cmp_lg_u32 s14, 0                                        // 000000004A8C: BF07800E
	s_cbranch_scc0 1417                                        // 000000004A90: BFA10589 <nearest+0x60b8>
	s_cmp_nlt_f32 s10, 0x41000000                              // 000000004A94: BF4EFF0A 41000000
	s_cbranch_scc1 65529                                       // 000000004A9C: BFA2FFF9 <nearest+0x4a84>
	s_cmp_le_f32 s10, 0                                        // 000000004AA0: BF43800A
	v_cmp_ge_f32_e64 s2, v32, -v0                              // 000000004AA4: D4160002 42020120
	v_cmp_ge_f32_e64 s4, v32, -v1                              // 000000004AAC: D4160004 42020320
	s_cselect_b32 s5, -1, 0                                    // 000000004AB4: 980580C1
	s_wait_alu depctr_sa_sdst(0)                               // 000000004AB8: BF88FF9E
	s_and_b32 s3, s2, s5                                       // 000000004ABC: 8B030502
	s_and_b32 s6, s4, s5                                       // 000000004AC0: 8B060504
	s_mov_b32 s5, 0                                            // 000000004AC4: BE850080
	s_wait_alu depctr_sa_sdst(0)                               // 000000004AC8: BF88FF9E
	s_or_b32 vcc_lo, s3, s6                                    // 000000004ACC: 8C6A0603
	s_wait_alu depctr_sa_sdst(0)                               // 000000004AD0: BF88FF9E
	s_cbranch_vccz 29                                          // 000000004AD4: BFA3001D <nearest+0x4b4c>
	v_cmp_gt_f32_e64 s5, v32, -v0                              // 000000004AD8: D4140005 42020120
	v_cmp_gt_f32_e32 vcc_lo, v33, v16                          // 000000004AE0: 7C282121
	v_cmp_gt_f32_e64 s7, v32, -v1                              // 000000004AE4: D4140007 42020320
	v_cmp_gt_f32_e64 s2, v33, v17                              // 000000004AEC: D4140002 02022321
	v_cmp_lt_f32_e64 s4, -1.0, v82                             // 000000004AF4: D4110004 0202A4F3
	s_or_b32 s5, s5, vcc_lo                                    // 000000004AFC: 8C056A05
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v83                         // 000000004B00: 7C22A6F3
	s_or_b32 s7, s7, s2                                        // 000000004B04: 8C070207
	s_wait_alu depctr_sa_sdst(0)                               // 000000004B08: BF88FF9E
	s_and_b32 s2, s5, s3                                       // 000000004B0C: 8B020305
	s_and_b32 s3, s7, s6                                       // 000000004B10: 8B030607
	s_wait_alu depctr_sa_sdst(0)                               // 000000004B14: BF88FF9E
	s_and_b32 s4, s4, s2                                       // 000000004B18: 8B040204
	s_and_b32 s5, vcc_lo, s3                                   // 000000004B1C: 8B05036A
	s_wait_alu depctr_sa_sdst(0)                               // 000000004B20: BF88FF9E
	s_or_b32 s4, s4, s5                                        // 000000004B24: 8C040504
	s_wait_alu depctr_sa_sdst(0)                               // 000000004B28: BF88FF9E
	v_cndmask_b32_e64 v34, 0, 1, s4                            // 000000004B2C: D5010022 00110280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000004B34: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v34                            // 000000004B38: 7C9A4480
	s_cbranch_vccz 6                                           // 000000004B3C: BFA30006 <nearest+0x4b58>
	s_mov_b32 s6, 0                                            // 000000004B40: BE860080
	s_mov_b32 s5, 1.0                                          // 000000004B44: BE8500F2
	s_branch 26                                                // 000000004B48: BFA0001A <nearest+0x4bb4>
	s_mov_b32 s6, 0x41000000                                   // 000000004B4C: BE8600FF 41000000
	s_branch 23                                                // 000000004B54: BFA00017 <nearest+0x4bb4>
	v_cndmask_b32_e64 v34, v90, -v0, s2                        // 000000004B58: D5010022 400A015A
	v_cndmask_b32_e64 v35, v89, -v1, s3                        // 000000004B60: D5010023 400E0359
	v_cndmask_b32_e64 v36, v88, v16, s2                        // 000000004B68: D5010024 000A2158
	v_cndmask_b32_e64 v37, v87, v17, s3                        // 000000004B70: D5010025 000E2357
	v_cndmask_b32_e64 v84, v84, v90, s2                        // 000000004B78: D5010054 000AB554
	v_cndmask_b32_e64 v85, v85, v89, s3                        // 000000004B80: D5010055 000EB355
	v_cndmask_b32_e64 v82, v82, v88, s2                        // 000000004B88: D5010052 000AB152
	v_cndmask_b32_e64 v83, v83, v87, s3                        // 000000004B90: D5010053 000EAF53
	v_dual_mov_b32 v88, v36 :: v_dual_mov_b32 v87, v37         // 000000004B98: CA100124 58560125
	v_dual_mov_b32 v90, v34 :: v_dual_mov_b32 v89, v35         // 000000004BA0: CA100122 5A580123
	s_mov_b32 s6, 0x41000000                                   // 000000004BA8: BE8600FF 41000000
	s_mov_b32 s5, 0                                            // 000000004BB0: BE850080
	s_wait_alu depctr_sa_sdst(0)                               // 000000004BB4: BF88FF9E
	s_cmp_lt_f32 s5, 0.5                                       // 000000004BB8: BF41F005
	v_cmp_ge_f32_e64 s3, v32, -v2                              // 000000004BBC: D4160003 42020520
	v_cmp_ge_f32_e64 s4, v32, -v3                              // 000000004BC4: D4160004 42020720
	s_cselect_b32 s2, -1, 0                                    // 000000004BCC: 980280C1
	s_cmp_le_f32 s10, 1.0                                      // 000000004BD0: BF43F20A
	s_cselect_b32 s7, -1, 0                                    // 000000004BD4: 980780C1
	s_wait_alu depctr_sa_sdst(0)                               // 000000004BD8: BF88FF9E
	s_and_b32 s2, s7, s2                                       // 000000004BDC: 8B020207
	s_wait_alu depctr_sa_sdst(0)                               // 000000004BE0: BF88FF9E
	s_and_b32 s3, s3, s2                                       // 000000004BE4: 8B030203
	s_and_b32 s7, s4, s2                                       // 000000004BE8: 8B070204
	s_wait_alu depctr_sa_sdst(0)                               // 000000004BEC: BF88FF9E
	s_or_b32 vcc_lo, s3, s7                                    // 000000004BF0: 8C6A0703
	s_wait_alu depctr_sa_sdst(0)                               // 000000004BF4: BF88FF9E
	s_cbranch_vccz 49                                          // 000000004BF8: BFA30031 <nearest+0x4cc0>
	v_cmp_gt_f32_e64 s8, v32, -v2                              // 000000004BFC: D4140008 42020520
	v_cmp_gt_f32_e32 vcc_lo, v33, v18                          // 000000004C04: 7C282521
	v_cmp_gt_f32_e64 s15, v32, -v3                             // 000000004C08: D414000F 42020720
	v_cmp_gt_f32_e64 s2, v33, v19                              // 000000004C10: D4140002 02022721
	v_cmp_lt_f32_e64 s4, -1.0, v82                             // 000000004C18: D4110004 0202A4F3
	s_or_b32 s8, s8, vcc_lo                                    // 000000004C20: 8C086A08
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v83                         // 000000004C24: 7C22A6F3
	s_or_b32 s15, s15, s2                                      // 000000004C28: 8C0F020F
	s_and_b32 s2, s8, s3                                       // 000000004C2C: 8B020308
	s_wait_alu depctr_sa_sdst(0)                               // 000000004C30: BF88FF9E
	s_and_b32 s3, s15, s7                                      // 000000004C34: 8B03070F
	s_and_b32 s4, s4, s2                                       // 000000004C38: 8B040204
	s_wait_alu depctr_sa_sdst(0)                               // 000000004C3C: BF88FF9E
	s_and_b32 s7, vcc_lo, s3                                   // 000000004C40: 8B07036A
	s_wait_alu depctr_sa_sdst(0)                               // 000000004C44: BF88FF9E
	s_or_b32 s4, s4, s7                                        // 000000004C48: 8C040704
	s_wait_alu depctr_sa_sdst(0)                               // 000000004C4C: BF88FF9E
	v_cndmask_b32_e64 v34, 0, 1, s4                            // 000000004C50: D5010022 00110280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000004C58: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v34                            // 000000004C5C: 7C9A4480
	s_cbranch_vccz 3                                           // 000000004C60: BFA30003 <nearest+0x4c70>
	s_mov_b32 s5, 1.0                                          // 000000004C64: BE8500F2
	s_mov_b32 s6, 1.0                                          // 000000004C68: BE8600F2
	s_branch 20                                                // 000000004C6C: BFA00014 <nearest+0x4cc0>
	v_cndmask_b32_e64 v34, v90, -v2, s2                        // 000000004C70: D5010022 400A055A
	v_cndmask_b32_e64 v35, v89, -v3, s3                        // 000000004C78: D5010023 400E0759
	v_cndmask_b32_e64 v36, v88, v18, s2                        // 000000004C80: D5010024 000A2558
	v_cndmask_b32_e64 v37, v87, v19, s3                        // 000000004C88: D5010025 000E2757
	v_cndmask_b32_e64 v84, v84, v90, s2                        // 000000004C90: D5010054 000AB554
	v_cndmask_b32_e64 v85, v85, v89, s3                        // 000000004C98: D5010055 000EB355
	v_cndmask_b32_e64 v82, v82, v88, s2                        // 000000004CA0: D5010052 000AB152
	v_cndmask_b32_e64 v83, v83, v87, s3                        // 000000004CA8: D5010053 000EAF53
	v_dual_mov_b32 v88, v36 :: v_dual_mov_b32 v87, v37         // 000000004CB0: CA100124 58560125
	v_dual_mov_b32 v90, v34 :: v_dual_mov_b32 v89, v35         // 000000004CB8: CA100122 5A580123
	s_wait_alu depctr_sa_sdst(0)                               // 000000004CC0: BF88FF9E
	s_cmp_lt_f32 s5, 0.5                                       // 000000004CC4: BF41F005
	v_cmp_ge_f32_e64 s3, v32, -v4                              // 000000004CC8: D4160003 42020920
	v_cmp_ge_f32_e64 s4, v32, -v5                              // 000000004CD0: D4160004 42020B20
	s_cselect_b32 s2, -1, 0                                    // 000000004CD8: 980280C1
	s_cmp_le_f32 s10, 2.0                                      // 000000004CDC: BF43F40A
	s_cselect_b32 s7, -1, 0                                    // 000000004CE0: 980780C1
	s_wait_alu depctr_sa_sdst(0)                               // 000000004CE4: BF88FF9E
	s_and_b32 s2, s7, s2                                       // 000000004CE8: 8B020207
	s_wait_alu depctr_sa_sdst(0)                               // 000000004CEC: BF88FF9E
	s_and_b32 s3, s3, s2                                       // 000000004CF0: 8B030203
	s_and_b32 s7, s4, s2                                       // 000000004CF4: 8B070204
	s_wait_alu depctr_sa_sdst(0)                               // 000000004CF8: BF88FF9E
	s_or_b32 vcc_lo, s3, s7                                    // 000000004CFC: 8C6A0703
	s_wait_alu depctr_sa_sdst(0)                               // 000000004D00: BF88FF9E
	s_cbranch_vccz 49                                          // 000000004D04: BFA30031 <nearest+0x4dcc>
	v_cmp_gt_f32_e64 s8, v32, -v4                              // 000000004D08: D4140008 42020920
	v_cmp_gt_f32_e32 vcc_lo, v33, v20                          // 000000004D10: 7C282921
	v_cmp_gt_f32_e64 s15, v32, -v5                             // 000000004D14: D414000F 42020B20
	v_cmp_gt_f32_e64 s2, v33, v21                              // 000000004D1C: D4140002 02022B21
	v_cmp_lt_f32_e64 s4, -1.0, v82                             // 000000004D24: D4110004 0202A4F3
	s_or_b32 s8, s8, vcc_lo                                    // 000000004D2C: 8C086A08
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v83                         // 000000004D30: 7C22A6F3
	s_or_b32 s15, s15, s2                                      // 000000004D34: 8C0F020F
	s_and_b32 s2, s8, s3                                       // 000000004D38: 8B020308
	s_wait_alu depctr_sa_sdst(0)                               // 000000004D3C: BF88FF9E
	s_and_b32 s3, s15, s7                                      // 000000004D40: 8B03070F
	s_and_b32 s4, s4, s2                                       // 000000004D44: 8B040204
	s_wait_alu depctr_sa_sdst(0)                               // 000000004D48: BF88FF9E
	s_and_b32 s7, vcc_lo, s3                                   // 000000004D4C: 8B07036A
	s_wait_alu depctr_sa_sdst(0)                               // 000000004D50: BF88FF9E
	s_or_b32 s4, s4, s7                                        // 000000004D54: 8C040704
	s_wait_alu depctr_sa_sdst(0)                               // 000000004D58: BF88FF9E
	v_cndmask_b32_e64 v34, 0, 1, s4                            // 000000004D5C: D5010022 00110280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000004D64: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v34                            // 000000004D68: 7C9A4480
	s_cbranch_vccz 3                                           // 000000004D6C: BFA30003 <nearest+0x4d7c>
	s_mov_b32 s6, 2.0                                          // 000000004D70: BE8600F4
	s_mov_b32 s5, 1.0                                          // 000000004D74: BE8500F2
	s_branch 20                                                // 000000004D78: BFA00014 <nearest+0x4dcc>
	v_cndmask_b32_e64 v34, v90, -v4, s2                        // 000000004D7C: D5010022 400A095A
	v_cndmask_b32_e64 v35, v89, -v5, s3                        // 000000004D84: D5010023 400E0B59
	v_cndmask_b32_e64 v36, v88, v20, s2                        // 000000004D8C: D5010024 000A2958
	v_cndmask_b32_e64 v37, v87, v21, s3                        // 000000004D94: D5010025 000E2B57
	v_cndmask_b32_e64 v84, v84, v90, s2                        // 000000004D9C: D5010054 000AB554
	v_cndmask_b32_e64 v85, v85, v89, s3                        // 000000004DA4: D5010055 000EB355
	v_cndmask_b32_e64 v82, v82, v88, s2                        // 000000004DAC: D5010052 000AB152
	v_cndmask_b32_e64 v83, v83, v87, s3                        // 000000004DB4: D5010053 000EAF53
	v_dual_mov_b32 v88, v36 :: v_dual_mov_b32 v87, v37         // 000000004DBC: CA100124 58560125
	v_dual_mov_b32 v90, v34 :: v_dual_mov_b32 v89, v35         // 000000004DC4: CA100122 5A580123
	s_wait_alu depctr_sa_sdst(0)                               // 000000004DCC: BF88FF9E
	s_cmp_lt_f32 s5, 0.5                                       // 000000004DD0: BF41F005
	v_cmp_ge_f32_e64 s3, v32, -v6                              // 000000004DD4: D4160003 42020D20
	v_cmp_ge_f32_e64 s4, v32, -v7                              // 000000004DDC: D4160004 42020F20
	s_cselect_b32 s2, -1, 0                                    // 000000004DE4: 980280C1
	s_cmp_le_f32 s10, 0x40400000                               // 000000004DE8: BF43FF0A 40400000
	s_cselect_b32 s7, -1, 0                                    // 000000004DF0: 980780C1
	s_wait_alu depctr_sa_sdst(0)                               // 000000004DF4: BF88FF9E
	s_and_b32 s2, s7, s2                                       // 000000004DF8: 8B020207
	s_wait_alu depctr_sa_sdst(0)                               // 000000004DFC: BF88FF9E
	s_and_b32 s3, s3, s2                                       // 000000004E00: 8B030203
	s_and_b32 s7, s4, s2                                       // 000000004E04: 8B070204
	s_wait_alu depctr_sa_sdst(0)                               // 000000004E08: BF88FF9E
	s_or_b32 vcc_lo, s3, s7                                    // 000000004E0C: 8C6A0703
	s_wait_alu depctr_sa_sdst(0)                               // 000000004E10: BF88FF9E
	s_cbranch_vccz 50                                          // 000000004E14: BFA30032 <nearest+0x4ee0>
	v_cmp_gt_f32_e64 s8, v32, -v6                              // 000000004E18: D4140008 42020D20
	v_cmp_gt_f32_e32 vcc_lo, v33, v22                          // 000000004E20: 7C282D21
	v_cmp_gt_f32_e64 s15, v32, -v7                             // 000000004E24: D414000F 42020F20
	v_cmp_gt_f32_e64 s2, v33, v23                              // 000000004E2C: D4140002 02022F21
	v_cmp_lt_f32_e64 s4, -1.0, v82                             // 000000004E34: D4110004 0202A4F3
	s_or_b32 s8, s8, vcc_lo                                    // 000000004E3C: 8C086A08
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v83                         // 000000004E40: 7C22A6F3
	s_or_b32 s15, s15, s2                                      // 000000004E44: 8C0F020F
	s_and_b32 s2, s8, s3                                       // 000000004E48: 8B020308
	s_wait_alu depctr_sa_sdst(0)                               // 000000004E4C: BF88FF9E
	s_and_b32 s3, s15, s7                                      // 000000004E50: 8B03070F
	s_and_b32 s4, s4, s2                                       // 000000004E54: 8B040204
	s_wait_alu depctr_sa_sdst(0)                               // 000000004E58: BF88FF9E
	s_and_b32 s7, vcc_lo, s3                                   // 000000004E5C: 8B07036A
	s_wait_alu depctr_sa_sdst(0)                               // 000000004E60: BF88FF9E
	s_or_b32 s4, s4, s7                                        // 000000004E64: 8C040704
	s_wait_alu depctr_sa_sdst(0)                               // 000000004E68: BF88FF9E
	v_cndmask_b32_e64 v34, 0, 1, s4                            // 000000004E6C: D5010022 00110280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000004E74: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v34                            // 000000004E78: 7C9A4480
	s_cbranch_vccz 4                                           // 000000004E7C: BFA30004 <nearest+0x4e90>
	s_mov_b32 s6, 0x40400000                                   // 000000004E80: BE8600FF 40400000
	s_mov_b32 s5, 1.0                                          // 000000004E88: BE8500F2
	s_branch 20                                                // 000000004E8C: BFA00014 <nearest+0x4ee0>
	v_cndmask_b32_e64 v34, v90, -v6, s2                        // 000000004E90: D5010022 400A0D5A
	v_cndmask_b32_e64 v35, v89, -v7, s3                        // 000000004E98: D5010023 400E0F59
	v_cndmask_b32_e64 v36, v88, v22, s2                        // 000000004EA0: D5010024 000A2D58
	v_cndmask_b32_e64 v37, v87, v23, s3                        // 000000004EA8: D5010025 000E2F57
	v_cndmask_b32_e64 v84, v84, v90, s2                        // 000000004EB0: D5010054 000AB554
	v_cndmask_b32_e64 v85, v85, v89, s3                        // 000000004EB8: D5010055 000EB355
	v_cndmask_b32_e64 v82, v82, v88, s2                        // 000000004EC0: D5010052 000AB152
	v_cndmask_b32_e64 v83, v83, v87, s3                        // 000000004EC8: D5010053 000EAF53
	v_dual_mov_b32 v88, v36 :: v_dual_mov_b32 v87, v37         // 000000004ED0: CA100124 58560125
	v_dual_mov_b32 v90, v34 :: v_dual_mov_b32 v89, v35         // 000000004ED8: CA100122 5A580123
	s_wait_alu depctr_sa_sdst(0)                               // 000000004EE0: BF88FF9E
	s_cmp_lt_f32 s5, 0.5                                       // 000000004EE4: BF41F005
	v_cmp_ge_f32_e64 s3, v32, -v8                              // 000000004EE8: D4160003 42021120
	v_cmp_ge_f32_e64 s4, v32, -v9                              // 000000004EF0: D4160004 42021320
	s_cselect_b32 s2, -1, 0                                    // 000000004EF8: 980280C1
	s_cmp_le_f32 s10, 4.0                                      // 000000004EFC: BF43F60A
	s_cselect_b32 s7, -1, 0                                    // 000000004F00: 980780C1
	s_wait_alu depctr_sa_sdst(0)                               // 000000004F04: BF88FF9E
	s_and_b32 s2, s7, s2                                       // 000000004F08: 8B020207
	s_wait_alu depctr_sa_sdst(0)                               // 000000004F0C: BF88FF9E
	s_and_b32 s3, s3, s2                                       // 000000004F10: 8B030203
	s_and_b32 s7, s4, s2                                       // 000000004F14: 8B070204
	s_wait_alu depctr_sa_sdst(0)                               // 000000004F18: BF88FF9E
	s_or_b32 vcc_lo, s3, s7                                    // 000000004F1C: 8C6A0703
	s_wait_alu depctr_sa_sdst(0)                               // 000000004F20: BF88FF9E
	s_cbranch_vccz 49                                          // 000000004F24: BFA30031 <nearest+0x4fec>
	v_cmp_gt_f32_e64 s8, v32, -v8                              // 000000004F28: D4140008 42021120
	v_cmp_gt_f32_e32 vcc_lo, v33, v24                          // 000000004F30: 7C283121
	v_cmp_gt_f32_e64 s15, v32, -v9                             // 000000004F34: D414000F 42021320
	v_cmp_gt_f32_e64 s2, v33, v25                              // 000000004F3C: D4140002 02023321
	v_cmp_lt_f32_e64 s4, -1.0, v82                             // 000000004F44: D4110004 0202A4F3
	s_or_b32 s8, s8, vcc_lo                                    // 000000004F4C: 8C086A08
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v83                         // 000000004F50: 7C22A6F3
	s_or_b32 s15, s15, s2                                      // 000000004F54: 8C0F020F
	s_and_b32 s2, s8, s3                                       // 000000004F58: 8B020308
	s_wait_alu depctr_sa_sdst(0)                               // 000000004F5C: BF88FF9E
	s_and_b32 s3, s15, s7                                      // 000000004F60: 8B03070F
	s_and_b32 s4, s4, s2                                       // 000000004F64: 8B040204
	s_wait_alu depctr_sa_sdst(0)                               // 000000004F68: BF88FF9E
	s_and_b32 s7, vcc_lo, s3                                   // 000000004F6C: 8B07036A
	s_wait_alu depctr_sa_sdst(0)                               // 000000004F70: BF88FF9E
	s_or_b32 s4, s4, s7                                        // 000000004F74: 8C040704
	s_wait_alu depctr_sa_sdst(0)                               // 000000004F78: BF88FF9E
	v_cndmask_b32_e64 v34, 0, 1, s4                            // 000000004F7C: D5010022 00110280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000004F84: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v34                            // 000000004F88: 7C9A4480
	s_cbranch_vccz 3                                           // 000000004F8C: BFA30003 <nearest+0x4f9c>
	s_mov_b32 s6, 4.0                                          // 000000004F90: BE8600F6
	s_mov_b32 s5, 1.0                                          // 000000004F94: BE8500F2
	s_branch 20                                                // 000000004F98: BFA00014 <nearest+0x4fec>
	v_cndmask_b32_e64 v34, v90, -v8, s2                        // 000000004F9C: D5010022 400A115A
	v_cndmask_b32_e64 v35, v89, -v9, s3                        // 000000004FA4: D5010023 400E1359
	v_cndmask_b32_e64 v36, v88, v24, s2                        // 000000004FAC: D5010024 000A3158
	v_cndmask_b32_e64 v37, v87, v25, s3                        // 000000004FB4: D5010025 000E3357
	v_cndmask_b32_e64 v84, v84, v90, s2                        // 000000004FBC: D5010054 000AB554
	v_cndmask_b32_e64 v85, v85, v89, s3                        // 000000004FC4: D5010055 000EB355
	v_cndmask_b32_e64 v82, v82, v88, s2                        // 000000004FCC: D5010052 000AB152
	v_cndmask_b32_e64 v83, v83, v87, s3                        // 000000004FD4: D5010053 000EAF53
	v_dual_mov_b32 v88, v36 :: v_dual_mov_b32 v87, v37         // 000000004FDC: CA100124 58560125
	v_dual_mov_b32 v90, v34 :: v_dual_mov_b32 v89, v35         // 000000004FE4: CA100122 5A580123
	s_wait_alu depctr_sa_sdst(0)                               // 000000004FEC: BF88FF9E
	s_cmp_lt_f32 s5, 0.5                                       // 000000004FF0: BF41F005
	v_cmp_ge_f32_e64 s3, v32, -v10                             // 000000004FF4: D4160003 42021520
	v_cmp_ge_f32_e64 s4, v32, -v11                             // 000000004FFC: D4160004 42021720
	s_cselect_b32 s2, -1, 0                                    // 000000005004: 980280C1
	s_cmp_le_f32 s10, 0x40a00000                               // 000000005008: BF43FF0A 40A00000
	s_cselect_b32 s7, -1, 0                                    // 000000005010: 980780C1
	s_wait_alu depctr_sa_sdst(0)                               // 000000005014: BF88FF9E
	s_and_b32 s2, s7, s2                                       // 000000005018: 8B020207
	s_wait_alu depctr_sa_sdst(0)                               // 00000000501C: BF88FF9E
	s_and_b32 s3, s3, s2                                       // 000000005020: 8B030203
	s_and_b32 s7, s4, s2                                       // 000000005024: 8B070204
	s_wait_alu depctr_sa_sdst(0)                               // 000000005028: BF88FF9E
	s_or_b32 vcc_lo, s3, s7                                    // 00000000502C: 8C6A0703
	s_wait_alu depctr_sa_sdst(0)                               // 000000005030: BF88FF9E
	s_cbranch_vccz 50                                          // 000000005034: BFA30032 <nearest+0x5100>
	v_cmp_gt_f32_e64 s8, v32, -v10                             // 000000005038: D4140008 42021520
	v_cmp_gt_f32_e32 vcc_lo, v33, v26                          // 000000005040: 7C283521
	v_cmp_gt_f32_e64 s15, v32, -v11                            // 000000005044: D414000F 42021720
	v_cmp_gt_f32_e64 s2, v33, v27                              // 00000000504C: D4140002 02023721
	v_cmp_lt_f32_e64 s4, -1.0, v82                             // 000000005054: D4110004 0202A4F3
	s_or_b32 s8, s8, vcc_lo                                    // 00000000505C: 8C086A08
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v83                         // 000000005060: 7C22A6F3
	s_or_b32 s15, s15, s2                                      // 000000005064: 8C0F020F
	s_and_b32 s2, s8, s3                                       // 000000005068: 8B020308
	s_wait_alu depctr_sa_sdst(0)                               // 00000000506C: BF88FF9E
	s_and_b32 s3, s15, s7                                      // 000000005070: 8B03070F
	s_and_b32 s4, s4, s2                                       // 000000005074: 8B040204
	s_wait_alu depctr_sa_sdst(0)                               // 000000005078: BF88FF9E
	s_and_b32 s7, vcc_lo, s3                                   // 00000000507C: 8B07036A
	s_wait_alu depctr_sa_sdst(0)                               // 000000005080: BF88FF9E
	s_or_b32 s4, s4, s7                                        // 000000005084: 8C040704
	s_wait_alu depctr_sa_sdst(0)                               // 000000005088: BF88FF9E
	v_cndmask_b32_e64 v34, 0, 1, s4                            // 00000000508C: D5010022 00110280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000005094: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v34                            // 000000005098: 7C9A4480
	s_cbranch_vccz 4                                           // 00000000509C: BFA30004 <nearest+0x50b0>
	s_mov_b32 s6, 0x40a00000                                   // 0000000050A0: BE8600FF 40A00000
	s_mov_b32 s5, 1.0                                          // 0000000050A8: BE8500F2
	s_branch 20                                                // 0000000050AC: BFA00014 <nearest+0x5100>
	v_cndmask_b32_e64 v34, v90, -v10, s2                       // 0000000050B0: D5010022 400A155A
	v_cndmask_b32_e64 v35, v89, -v11, s3                       // 0000000050B8: D5010023 400E1759
	v_cndmask_b32_e64 v36, v88, v26, s2                        // 0000000050C0: D5010024 000A3558
	v_cndmask_b32_e64 v37, v87, v27, s3                        // 0000000050C8: D5010025 000E3757
	v_cndmask_b32_e64 v84, v84, v90, s2                        // 0000000050D0: D5010054 000AB554
	v_cndmask_b32_e64 v85, v85, v89, s3                        // 0000000050D8: D5010055 000EB355
	v_cndmask_b32_e64 v82, v82, v88, s2                        // 0000000050E0: D5010052 000AB152
	v_cndmask_b32_e64 v83, v83, v87, s3                        // 0000000050E8: D5010053 000EAF53
	v_dual_mov_b32 v88, v36 :: v_dual_mov_b32 v87, v37         // 0000000050F0: CA100124 58560125
	v_dual_mov_b32 v90, v34 :: v_dual_mov_b32 v89, v35         // 0000000050F8: CA100122 5A580123
	s_wait_alu depctr_sa_sdst(0)                               // 000000005100: BF88FF9E
	s_cmp_lt_f32 s5, 0.5                                       // 000000005104: BF41F005
	v_cmp_ge_f32_e64 s3, v32, -v12                             // 000000005108: D4160003 42021920
	v_cmp_ge_f32_e64 s4, v32, -v13                             // 000000005110: D4160004 42021B20
	s_cselect_b32 s2, -1, 0                                    // 000000005118: 980280C1
	s_cmp_le_f32 s10, 0x40c00000                               // 00000000511C: BF43FF0A 40C00000
	s_cselect_b32 s7, -1, 0                                    // 000000005124: 980780C1
	s_wait_alu depctr_sa_sdst(0)                               // 000000005128: BF88FF9E
	s_and_b32 s2, s7, s2                                       // 00000000512C: 8B020207
	s_wait_alu depctr_sa_sdst(0)                               // 000000005130: BF88FF9E
	s_and_b32 s3, s3, s2                                       // 000000005134: 8B030203
	s_and_b32 s7, s4, s2                                       // 000000005138: 8B070204
	s_wait_alu depctr_sa_sdst(0)                               // 00000000513C: BF88FF9E
	s_or_b32 vcc_lo, s3, s7                                    // 000000005140: 8C6A0703
	s_wait_alu depctr_sa_sdst(0)                               // 000000005144: BF88FF9E
	s_cbranch_vccz 50                                          // 000000005148: BFA30032 <nearest+0x5214>
	v_cmp_gt_f32_e64 s8, v32, -v12                             // 00000000514C: D4140008 42021920
	v_cmp_gt_f32_e32 vcc_lo, v33, v28                          // 000000005154: 7C283921
	v_cmp_gt_f32_e64 s15, v32, -v13                            // 000000005158: D414000F 42021B20
	v_cmp_gt_f32_e64 s2, v33, v29                              // 000000005160: D4140002 02023B21
	v_cmp_lt_f32_e64 s4, -1.0, v82                             // 000000005168: D4110004 0202A4F3
	s_or_b32 s8, s8, vcc_lo                                    // 000000005170: 8C086A08
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v83                         // 000000005174: 7C22A6F3
	s_or_b32 s15, s15, s2                                      // 000000005178: 8C0F020F
	s_and_b32 s2, s8, s3                                       // 00000000517C: 8B020308
	s_wait_alu depctr_sa_sdst(0)                               // 000000005180: BF88FF9E
	s_and_b32 s3, s15, s7                                      // 000000005184: 8B03070F
	s_and_b32 s4, s4, s2                                       // 000000005188: 8B040204
	s_wait_alu depctr_sa_sdst(0)                               // 00000000518C: BF88FF9E
	s_and_b32 s7, vcc_lo, s3                                   // 000000005190: 8B07036A
	s_wait_alu depctr_sa_sdst(0)                               // 000000005194: BF88FF9E
	s_or_b32 s4, s4, s7                                        // 000000005198: 8C040704
	s_wait_alu depctr_sa_sdst(0)                               // 00000000519C: BF88FF9E
	v_cndmask_b32_e64 v34, 0, 1, s4                            // 0000000051A0: D5010022 00110280
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000051A8: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v34                            // 0000000051AC: 7C9A4480
	s_cbranch_vccz 4                                           // 0000000051B0: BFA30004 <nearest+0x51c4>
	s_mov_b32 s6, 0x40c00000                                   // 0000000051B4: BE8600FF 40C00000
	s_mov_b32 s5, 1.0                                          // 0000000051BC: BE8500F2
	s_branch 20                                                // 0000000051C0: BFA00014 <nearest+0x5214>
	v_cndmask_b32_e64 v34, v90, -v12, s2                       // 0000000051C4: D5010022 400A195A
	v_cndmask_b32_e64 v35, v89, -v13, s3                       // 0000000051CC: D5010023 400E1B59
	v_cndmask_b32_e64 v36, v88, v28, s2                        // 0000000051D4: D5010024 000A3958
	v_cndmask_b32_e64 v37, v87, v29, s3                        // 0000000051DC: D5010025 000E3B57
	v_cndmask_b32_e64 v84, v84, v90, s2                        // 0000000051E4: D5010054 000AB554
	v_cndmask_b32_e64 v85, v85, v89, s3                        // 0000000051EC: D5010055 000EB355
	v_cndmask_b32_e64 v82, v82, v88, s2                        // 0000000051F4: D5010052 000AB152
	v_cndmask_b32_e64 v83, v83, v87, s3                        // 0000000051FC: D5010053 000EAF53
	v_dual_mov_b32 v88, v36 :: v_dual_mov_b32 v87, v37         // 000000005204: CA100124 58560125
	v_dual_mov_b32 v90, v34 :: v_dual_mov_b32 v89, v35         // 00000000520C: CA100122 5A580123
	s_wait_alu depctr_sa_sdst(0)                               // 000000005214: BF88FF9E
	s_cmp_lt_f32 s5, 0.5                                       // 000000005218: BF41F005
	v_cmp_ge_f32_e64 s3, v32, -v14                             // 00000000521C: D4160003 42021D20
	v_cmp_ge_f32_e64 s4, v32, -v15                             // 000000005224: D4160004 42021F20
	s_cselect_b32 s2, -1, 0                                    // 00000000522C: 980280C1
	s_cmp_le_f32 s10, 0x40e00000                               // 000000005230: BF43FF0A 40E00000
	s_cselect_b32 s7, -1, 0                                    // 000000005238: 980780C1
	s_wait_alu depctr_sa_sdst(0)                               // 00000000523C: BF88FF9E
	s_and_b32 s2, s7, s2                                       // 000000005240: 8B020207
	s_wait_alu depctr_sa_sdst(0)                               // 000000005244: BF88FF9E
	s_and_b32 s3, s3, s2                                       // 000000005248: 8B030203
	s_and_b32 s7, s4, s2                                       // 00000000524C: 8B070204
	s_wait_alu depctr_sa_sdst(0)                               // 000000005250: BF88FF9E
	s_or_b32 vcc_lo, s3, s7                                    // 000000005254: 8C6A0703
	s_wait_alu depctr_sa_sdst(0)                               // 000000005258: BF88FF9E
	s_cbranch_vccz 29                                          // 00000000525C: BFA3001D <nearest+0x52d4>
	v_cmp_gt_f32_e64 s8, v32, -v14                             // 000000005260: D4140008 42021D20
	v_cmp_gt_f32_e32 vcc_lo, v33, v30                          // 000000005268: 7C283D21
	v_cmp_gt_f32_e64 s10, v32, -v15                            // 00000000526C: D414000A 42021F20
	v_cmp_gt_f32_e64 s2, v33, v31                              // 000000005274: D4140002 02023F21
	v_cmp_lt_f32_e64 s4, -1.0, v82                             // 00000000527C: D4110004 0202A4F3
	s_or_b32 s8, s8, vcc_lo                                    // 000000005284: 8C086A08
	v_cmp_lt_f32_e32 vcc_lo, -1.0, v83                         // 000000005288: 7C22A6F3
	s_or_b32 s10, s10, s2                                      // 00000000528C: 8C0A020A
	s_and_b32 s2, s8, s3                                       // 000000005290: 8B020308
	s_wait_alu depctr_sa_sdst(0)                               // 000000005294: BF88FF9E
	s_and_b32 s3, s10, s7                                      // 000000005298: 8B03070A
	s_and_b32 s4, s4, s2                                       // 00000000529C: 8B040204
	s_wait_alu depctr_sa_sdst(0)                               // 0000000052A0: BF88FF9E
	s_and_b32 s7, vcc_lo, s3                                   // 0000000052A4: 8B07036A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000052A8: BF88FF9E
	s_or_b32 s4, s4, s7                                        // 0000000052AC: 8C040704
	s_wait_alu depctr_sa_sdst(0)                               // 0000000052B0: BF88FF9E
	v_cndmask_b32_e64 v34, 0, 1, s4                            // 0000000052B4: D5010022 00110280
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000052BC: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v34                            // 0000000052C0: 7C9A4480
	s_cbranch_vccz 846                                         // 0000000052C4: BFA3034E <nearest+0x6000>
	s_mov_b32 s6, 0x40e00000                                   // 0000000052C8: BE8600FF 40E00000
	s_mov_b32 s5, 1.0                                          // 0000000052D0: BE8500F2
	s_wait_alu depctr_sa_sdst(0)                               // 0000000052D4: BF88FF9E
	s_cmp_ngt_f32 s5, 0.5                                      // 0000000052D8: BF4BF005
	s_mov_b32 s10, s6                                          // 0000000052DC: BE8A0006
	s_cbranch_scc1 65000                                       // 0000000052E0: BFA2FDE8 <nearest+0x4a84>
	v_cmp_eq_f32_e32 vcc_lo, v89, v90                          // 0000000052E4: 7C24B559
	v_cmp_lt_f32_e64 s2, v87, v88                              // 0000000052E8: D4110002 0202B157
	v_cmp_lt_f32_e64 s3, v88, v87                              // 0000000052F0: D4110003 0202AF58
	v_cmp_lt_f32_e64 s4, v89, v90                              // 0000000052F8: D4110004 0202B559
	v_cmp_lt_f32_e64 s5, v90, v89                              // 000000005300: D4110005 0202B35A
	v_cmp_lt_f32_e64 s6, v84, v85                              // 000000005308: D4110006 0202AB54
	s_and_b32 s2, vcc_lo, s2                                   // 000000005310: 8B02026A
	s_and_b32 s3, vcc_lo, s3                                   // 000000005314: 8B03036A
	s_wait_alu depctr_sa_sdst(0)                               // 000000005318: BF88FF9E
	s_or_b32 s2, s4, s2                                        // 00000000531C: 8C020204
	s_or_b32 s3, s5, s3                                        // 000000005320: 8C030305
	s_wait_alu depctr_sa_sdst(0)                               // 000000005324: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000005328: 8D6A0002
	s_xor_b32 s2, s3, s9                                       // 00000000532C: 8D020903
	s_wait_alu depctr_sa_sdst(0)                               // 000000005330: BF88FF9E
	v_cndmask_b32_e32 v32, v90, v89, vcc_lo                    // 000000005334: 0240B35A
	v_cndmask_b32_e32 v34, v88, v87, vcc_lo                    // 000000005338: 0244AF58
	v_cmp_eq_f32_e32 vcc_lo, v85, v84                          // 00000000533C: 7C24A955
	v_cmp_lt_f32_e64 s3, v83, v82                              // 000000005340: D4110003 0202A553
	v_cmp_lt_f32_e64 s5, v85, v84                              // 000000005348: D4110005 0202A955
	v_cmp_lt_f32_e64 s4, v82, v83                              // 000000005350: D4110004 0202A752
	v_cndmask_b32_e64 v33, v89, v90, s2                        // 000000005358: D5010021 000AB559
	v_cndmask_b32_e64 v35, v87, v88, s2                        // 000000005360: D5010023 000AB157
	s_and_b32 s2, vcc_lo, s3                                   // 000000005368: 8B02036A
	v_mov_b32_e32 v37, v32                                     // 00000000536C: 7E4A0320
	s_wait_alu depctr_sa_sdst(0)                               // 000000005370: BF88FF9E
	s_or_b32 s2, s5, s2                                        // 000000005374: 8C020205
	s_and_b32 s3, vcc_lo, s4                                   // 000000005378: 8B03046A
	s_wait_alu depctr_sa_sdst(0)                               // 00000000537C: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000005380: 8D6A0002
	v_mov_b32_e32 v38, v33                                     // 000000005384: 7E4C0321
	s_wait_alu depctr_sa_sdst(0)                               // 000000005388: BF88FF9E
	v_cndmask_b32_e32 v42, v82, v83, vcc_lo                    // 00000000538C: 0254A752
	v_permlanex16_b32 v37, v37, s11, 0xfedcba98                // 000000005390: D65C0025 03FC1725 FEDCBA98
	s_or_b32 s3, s6, s3                                        // 00000000539C: 8C030306
	v_mov_b32_e32 v90, 0x7f800000                              // 0000000053A0: 7EB402FF 7F800000
	s_wait_alu depctr_sa_sdst(0)                               // 0000000053A8: BF88FF9E
	s_xor_b32 s2, s3, s9                                       // 0000000053AC: 8D020903
	v_dual_mov_b32 v45, v42 :: v_dual_cndmask_b32 v36, v84, v85// 0000000053B0: CA12012A 2D24AB54
	v_mov_b32_e32 v39, v34                                     // 0000000053B8: 7E4E0322
	v_cmp_eq_f32_e32 vcc_lo, v32, v37                          // 0000000053BC: 7C244B20
	v_cmp_gt_f32_e64 s6, v32, v37                              // 0000000053C0: D4140006 02024B20
	s_wait_alu depctr_sa_sdst(0)                               // 0000000053C8: BF88FF9E
	v_cndmask_b32_e64 v41, v85, v84, s2                        // 0000000053CC: D5010029 000AA955
	v_cndmask_b32_e64 v43, v83, v82, s2                        // 0000000053D4: D501002B 000AA553
	v_permlanex16_b32 v39, v39, s11, 0xfedcba98                // 0000000053DC: D65C0027 03FC1727 FEDCBA98
	v_permlanex16_b32 v38, v38, s11, 0xfedcba98                // 0000000053E8: D65C0026 03FC1726 FEDCBA98
	v_permlanex16_b32 v45, v45, s11, 0xfedcba98                // 0000000053F4: D65C002D 03FC172D FEDCBA98
	v_mov_b32_e32 v44, v41                                     // 000000005400: 7E580329
	v_mov_b32_e32 v46, v43                                     // 000000005404: 7E5C032B
	v_cmp_gt_f32_e64 s4, v34, v39                              // 000000005408: D4140004 02024F22
	v_cmp_eq_f32_e64 s3, v33, v38                              // 000000005410: D4120003 02024D21
	v_cmp_gt_f32_e64 s7, v33, v38                              // 000000005418: D4140007 02024D21
	v_permlanex16_b32 v44, v44, s11, 0xfedcba98                // 000000005420: D65C002C 03FC172C FEDCBA98
	v_permlanex16_b32 v46, v46, s11, 0xfedcba98                // 00000000542C: D65C002E 03FC172E FEDCBA98
	s_and_b32 s2, vcc_lo, s4                                   // 000000005438: 8B02046A
	v_mov_b32_e32 v88, -1.0                                    // 00000000543C: 7EB002F3
	s_wait_alu depctr_sa_sdst(0)                               // 000000005440: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 000000005444: 8C020206
	v_cmp_neq_f32_e64 s4, v41, v44                             // 000000005448: D41D0004 02025929
	s_wait_alu depctr_sa_sdst(0)                               // 000000005450: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000005454: 8D6A0002
	v_mov_b32_e32 v84, 0x7f800000                              // 000000005458: 7EA802FF 7F800000
	s_wait_alu depctr_sa_sdst(0)                               // 000000005460: BF88FF9E
	v_cndmask_b32_e32 v34, v34, v39, vcc_lo                    // 000000005464: 02444F22
	v_mov_b32_e32 v40, v35                                     // 000000005468: 7E500323
	v_cndmask_b32_e32 v32, v32, v37, vcc_lo                    // 00000000546C: 02404B20
	v_dual_mov_b32 v37, v36 :: v_dual_mov_b32 v82, -1.0        // 000000005470: CA100124 255200F3
	v_mov_b32_e32 v87, -1.0                                    // 000000005478: 7EAE02F3
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 00000000547C: BF870224
	v_permlanex16_b32 v40, v40, s11, 0xfedcba98                // 000000005480: D65C0028 03FC1728 FEDCBA98
	v_mov_b32_e32 v89, 0x7f800000                              // 00000000548C: 7EB202FF 7F800000
	v_permlanex16_b32 v37, v37, s11, 0xfedcba98                // 000000005494: D65C0025 03FC1725 FEDCBA98
	v_mov_b32_e32 v83, -1.0                                    // 0000000054A0: 7EA602F3
	v_mov_b32_e32 v85, 0x7f800000                              // 0000000054A4: 7EAA02FF 7F800000
	v_cmp_gt_f32_e64 s5, v35, v40                              // 0000000054AC: D4140005 02025123
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000054B4: BF870004
	v_cmp_neq_f32_e32 vcc_lo, v36, v37                         // 0000000054B8: 7C3A4B24
	v_cmp_ngt_f32_e64 s6, v36, v37                             // 0000000054BC: D41B0006 02024B24
	s_and_b32 s3, s3, s5                                       // 0000000054C4: 8B030503
	v_cmp_ngt_f32_e64 s5, v43, v46                             // 0000000054C8: D41B0005 02025D2B
	s_wait_alu depctr_sa_sdst(0)                               // 0000000054D0: BF88FF9E
	s_or_b32 s3, s7, s3                                        // 0000000054D4: 8C030307
	v_cmp_ngt_f32_e64 s7, v41, v44                             // 0000000054D8: D41B0007 02025929
	s_wait_alu depctr_sa_sdst(0)                               // 0000000054E0: BF88FF9E
	s_xor_b32 s2, s3, s0                                       // 0000000054E4: 8D020003
	v_cmp_ngt_f32_e64 s3, v42, v45                             // 0000000054E8: D41B0003 02025B2A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000054F0: BF88FF9E
	v_cndmask_b32_e64 v33, v33, v38, s2                        // 0000000054F4: D5010021 000A4D21
	v_cndmask_b32_e64 v35, v35, v40, s2                        // 0000000054FC: D5010023 000A5123
	s_or_b32 s2, vcc_lo, s3                                    // 000000005504: 8C02036A
	s_or_b32 s3, s4, s5                                        // 000000005508: 8C030504
	s_wait_alu depctr_sa_sdst(0)                               // 00000000550C: BF88FF9E
	s_and_b32 s2, s6, s2                                       // 000000005510: 8B020206
	s_and_b32 s3, s7, s3                                       // 000000005514: 8B030307
	s_wait_alu depctr_sa_sdst(0)                               // 000000005518: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 00000000551C: 8D6A0002
	s_xor_b32 s2, s3, s0                                       // 000000005520: 8D020003
	s_wait_alu depctr_sa_sdst(0)                               // 000000005524: BF88FF9E
	v_cndmask_b32_e32 v36, v36, v37, vcc_lo                    // 000000005528: 02484B24
	v_cndmask_b32_e32 v38, v42, v45, vcc_lo                    // 00000000552C: 024C5B2A
	v_cmp_eq_f32_e32 vcc_lo, v33, v32                          // 000000005530: 7C244121
	v_cmp_lt_f32_e64 s3, v35, v34                              // 000000005534: D4110003 02024523
	v_cmp_lt_f32_e64 s4, v34, v35                              // 00000000553C: D4110004 02024722
	v_cndmask_b32_e64 v37, v41, v44, s2                        // 000000005544: D5010025 000A5929
	v_cmp_lt_f32_e64 s5, v33, v32                              // 00000000554C: D4110005 02024121
	v_cndmask_b32_e64 v39, v43, v46, s2                        // 000000005554: D5010027 000A5D2B
	v_cmp_lt_f32_e64 s2, v32, v33                              // 00000000555C: D4110002 02024320
	s_and_b32 s3, vcc_lo, s3                                   // 000000005564: 8B03036A
	s_and_b32 s4, vcc_lo, s4                                   // 000000005568: 8B04046A
	s_wait_alu depctr_sa_sdst(0)                               // 00000000556C: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 000000005570: 8C6A0305
	v_cmp_lt_f32_e64 s3, v39, v38                              // 000000005574: D4110003 02024D27
	s_or_b32 s2, s2, s4                                        // 00000000557C: 8C020402
	s_wait_alu depctr_sa_sdst(0)                               // 000000005580: BF88FF9E
	v_cndmask_b32_e32 v40, v32, v33, vcc_lo                    // 000000005584: 02504320
	v_cndmask_b32_e64 v32, v32, v33, s2                        // 000000005588: D5010020 000A4320
	v_cndmask_b32_e32 v33, v34, v35, vcc_lo                    // 000000005590: 02424722
	v_cmp_eq_f32_e32 vcc_lo, v37, v36                          // 000000005594: 7C244925
	v_cmp_lt_f32_e64 s4, v38, v39                              // 000000005598: D4110004 02024F26
	v_cmp_lt_f32_e64 s5, v37, v36                              // 0000000055A0: D4110005 02024925
	v_cndmask_b32_e64 v34, v34, v35, s2                        // 0000000055A8: D5010022 000A4722
	v_cmp_lt_f32_e64 s2, v36, v37                              // 0000000055B0: D4110002 02024B24
	s_and_b32 s3, vcc_lo, s3                                   // 0000000055B8: 8B03036A
	s_and_b32 s4, vcc_lo, s4                                   // 0000000055BC: 8B04046A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000055C0: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 0000000055C4: 8C6A0305
	s_or_b32 s2, s2, s4                                        // 0000000055C8: 8C020402
	s_wait_alu depctr_sa_sdst(0)                               // 0000000055CC: BF88FF9E
	v_cndmask_b32_e32 v35, v37, v36, vcc_lo                    // 0000000055D0: 02464925
	v_cndmask_b32_e64 v36, v37, v36, s2                        // 0000000055D4: D5010024 000A4925
	v_cndmask_b32_e32 v37, v39, v38, vcc_lo                    // 0000000055DC: 024A4D27
	v_cndmask_b32_e64 v38, v39, v38, s2                        // 0000000055E0: D5010026 000A4D27
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 0000000055E8: BF870224
	v_cmp_eq_f32_e32 vcc_lo, v40, v35                          // 0000000055EC: 7C244728
	v_cmp_lt_f32_e64 s5, v40, v35                              // 0000000055F0: D4110005 02024728
	v_cmp_lt_f32_e64 s3, v33, v37                              // 0000000055F8: D4110003 02024B21
	v_cmp_eq_f32_e64 s2, v32, v36                              // 000000005600: D4120002 02024920
	v_cmp_lt_f32_e64 s4, v34, v38                              // 000000005608: D4110004 02024D22
	v_cmp_lt_f32_e64 s6, v32, v36                              // 000000005610: D4110006 02024920
	s_and_b32 s3, vcc_lo, s3                                   // 000000005618: 8B03036A
	s_wait_alu depctr_sa_sdst(0)                               // 00000000561C: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 000000005620: 8C6A0305
	s_and_b32 s2, s2, s4                                       // 000000005624: 8B020402
	s_wait_alu depctr_sa_sdst(0)                               // 000000005628: BF88FF9E
	v_dual_cndmask_b32 v39, v35, v40 :: v_dual_cndmask_b32 v42, v37, v33// 00000000562C: CA525123 272A4325
	v_cndmask_b32_e32 v35, v40, v35, vcc_lo                    // 000000005634: 02464728
	v_cndmask_b32_e32 v33, v33, v37, vcc_lo                    // 000000005638: 02424B21
	s_or_b32 s2, s6, s2                                        // 00000000563C: 8C020206
	s_delay_alu instid0(VALU_DEP_3)                            // 000000005640: BF870003
	v_dual_mov_b32 v40, v39 :: v_dual_mov_b32 v45, v42         // 000000005644: CA100127 282C012A
	s_wait_alu depctr_sa_sdst(0)                               // 00000000564C: BF88FF9E
	v_cndmask_b32_e64 v41, v36, v32, s2                        // 000000005650: D5010029 000A4124
	v_cndmask_b32_e64 v43, v38, v34, s2                        // 000000005658: D501002B 000A4526
	v_cndmask_b32_e64 v32, v32, v36, s2                        // 000000005660: D5010020 000A4920
	v_permlanex16_b32 v40, v40, s11, 0xfedcba98                // 000000005668: D65C0028 03FC1728 FEDCBA98
	v_permlanex16_b32 v45, v45, s11, 0xfedcba98                // 000000005674: D65C002D 03FC172D FEDCBA98
	v_cndmask_b32_e64 v34, v34, v38, s2                        // 000000005680: D5010022 000A4D22
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000005688: BF870214
	v_dual_mov_b32 v37, v35 :: v_dual_mov_b32 v38, v32         // 00000000568C: CA100123 25260120
	v_cmp_eq_f32_e32 vcc_lo, v39, v40                          // 000000005694: 7C245127
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 000000005698: BF870224
	v_cmp_gt_f32_e64 s4, v42, v45                              // 00000000569C: D4140004 02025B2A
	v_cmp_gt_f32_e64 s6, v39, v40                              // 0000000056A4: D4140006 02025127
	v_permlanex16_b32 v37, v37, s11, 0xfedcba98                // 0000000056AC: D65C0025 03FC1725 FEDCBA98
	v_mov_b32_e32 v44, v41                                     // 0000000056B8: 7E580329
	v_permlanex16_b32 v38, v38, s11, 0xfedcba98                // 0000000056BC: D65C0026 03FC1726 FEDCBA98
	s_and_b32 s2, vcc_lo, s4                                   // 0000000056C8: 8B02046A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000056CC: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 0000000056D0: 8C020206
	v_permlanex16_b32 v44, v44, s11, 0xfedcba98                // 0000000056D4: D65C002C 03FC172C FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 0000000056E0: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 0000000056E4: 8D6A0002
	v_cmp_gt_f32_e64 s6, v35, v37                              // 0000000056E8: D4140006 02024B23
	s_wait_alu depctr_sa_sdst(0)                               // 0000000056F0: BF88FF9E
	v_cndmask_b32_e32 v36, v39, v40, vcc_lo                    // 0000000056F4: 02485127
	v_dual_mov_b32 v46, v43 :: v_dual_mov_b32 v39, v33         // 0000000056F8: CA10012B 2E260121
	v_mov_b32_e32 v40, v34                                     // 000000005700: 7E500322
	v_cmp_eq_f32_e64 s3, v41, v44                              // 000000005704: D4120003 02025929
	v_cmp_gt_f32_e64 s7, v41, v44                              // 00000000570C: D4140007 02025929
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005714: BF870004
	v_permlanex16_b32 v46, v46, s11, 0xfedcba98                // 000000005718: D65C002E 03FC172E FEDCBA98
	v_permlanex16_b32 v39, v39, s11, 0xfedcba98                // 000000005724: D65C0027 03FC1727 FEDCBA98
	v_permlanex16_b32 v40, v40, s11, 0xfedcba98                // 000000005730: D65C0028 03FC1728 FEDCBA98
	v_cndmask_b32_e32 v42, v42, v45, vcc_lo                    // 00000000573C: 02545B2A
	v_cmp_eq_f32_e32 vcc_lo, v35, v37                          // 000000005740: 7C244B23
	v_cmp_gt_f32_e64 s5, v43, v46                              // 000000005744: D4140005 02025D2B
	v_cmp_gt_f32_e64 s4, v33, v39                              // 00000000574C: D4140004 02024F21
	s_and_b32 s3, s3, s5                                       // 000000005754: 8B030503
	v_cmp_gt_f32_e64 s5, v34, v40                              // 000000005758: D4140005 02025122
	s_wait_alu depctr_sa_sdst(0)                               // 000000005760: BF88FF9E
	s_or_b32 s3, s7, s3                                        // 000000005764: 8C030307
	v_cmp_gt_f32_e64 s7, v32, v38                              // 000000005768: D4140007 02024D20
	s_wait_alu depctr_sa_sdst(0)                               // 000000005770: BF88FF9E
	s_xor_b32 s2, s3, s0                                       // 000000005774: 8D020003
	v_cmp_eq_f32_e64 s3, v32, v38                              // 000000005778: D4120003 02024D20
	s_wait_alu depctr_sa_sdst(0)                               // 000000005780: BF88FF9E
	v_cndmask_b32_e64 v41, v41, v44, s2                        // 000000005784: D5010029 000A5929
	v_cndmask_b32_e64 v43, v43, v46, s2                        // 00000000578C: D501002B 000A5D2B
	s_and_b32 s2, vcc_lo, s4                                   // 000000005794: 8B02046A
	s_and_b32 s3, s3, s5                                       // 000000005798: 8B030503
	s_wait_alu depctr_sa_sdst(0)                               // 00000000579C: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 0000000057A0: 8C020206
	s_or_b32 s3, s7, s3                                        // 0000000057A4: 8C030307
	s_wait_alu depctr_sa_sdst(0)                               // 0000000057A8: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 0000000057AC: 8D6A0002
	s_xor_b32 s2, s3, s0                                       // 0000000057B0: 8D020003
	s_wait_alu depctr_sa_sdst(0)                               // 0000000057B4: BF88FF9E
	v_cndmask_b32_e32 v35, v35, v37, vcc_lo                    // 0000000057B8: 02464B23
	v_cndmask_b32_e32 v33, v33, v39, vcc_lo                    // 0000000057BC: 02424F21
	v_cmp_eq_f32_e32 vcc_lo, v41, v36                          // 0000000057C0: 7C244929
	v_cmp_lt_f32_e64 s3, v43, v42                              // 0000000057C4: D4110003 0202552B
	v_cmp_lt_f32_e64 s4, v42, v43                              // 0000000057CC: D4110004 0202572A
	v_cmp_lt_f32_e64 s5, v41, v36                              // 0000000057D4: D4110005 02024929
	v_cndmask_b32_e64 v32, v32, v38, s2                        // 0000000057DC: D5010020 000A4D20
	v_cndmask_b32_e64 v34, v34, v40, s2                        // 0000000057E4: D5010022 000A5122
	v_cmp_lt_f32_e64 s2, v36, v41                              // 0000000057EC: D4110002 02025324
	s_and_b32 s3, vcc_lo, s3                                   // 0000000057F4: 8B03036A
	s_and_b32 s4, vcc_lo, s4                                   // 0000000057F8: 8B04046A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000057FC: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 000000005800: 8C6A0305
	v_cmp_lt_f32_e64 s3, v34, v33                              // 000000005804: D4110003 02024322
	s_wait_alu depctr_sa_sdst(0)                               // 00000000580C: BF88FF9E
	v_cndmask_b32_e32 v37, v36, v41, vcc_lo                    // 000000005810: 024A5324
	s_or_b32 s2, s2, s4                                        // 000000005814: 8C020402
	v_cndmask_b32_e32 v38, v42, v43, vcc_lo                    // 000000005818: 024C572A
	v_cmp_eq_f32_e32 vcc_lo, v32, v35                          // 00000000581C: 7C244720
	v_cmp_lt_f32_e64 s4, v33, v34                              // 000000005820: D4110004 02024521
	s_wait_alu depctr_sa_sdst(0)                               // 000000005828: BF88FF9E
	v_cndmask_b32_e64 v36, v36, v41, s2                        // 00000000582C: D5010024 000A5324
	v_cmp_lt_f32_e64 s5, v32, v35                              // 000000005834: D4110005 02024720
	v_cndmask_b32_e64 v39, v42, v43, s2                        // 00000000583C: D5010027 000A572A
	v_cmp_lt_f32_e64 s2, v35, v32                              // 000000005844: D4110002 02024123
	s_and_b32 s3, vcc_lo, s3                                   // 00000000584C: 8B03036A
	s_and_b32 s4, vcc_lo, s4                                   // 000000005850: 8B04046A
	s_wait_alu depctr_sa_sdst(0)                               // 000000005854: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 000000005858: 8C6A0305
	v_permlanex16_b32 v36, v36, s11, 0xfedcba98                // 00000000585C: D65C0024 03FC1724 FEDCBA98
	s_or_b32 s2, s2, s4                                        // 000000005868: 8C020402
	v_permlanex16_b32 v39, v39, s11, 0xfedcba98                // 00000000586C: D65C0027 03FC1727 FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 000000005878: BF88FF9E
	v_cndmask_b32_e32 v40, v35, v32, vcc_lo                    // 00000000587C: 02504123
	v_cndmask_b32_e64 v32, v35, v32, s2                        // 000000005880: D5010020 000A4123
	v_permlanex16_b32 v37, v37, s11, 0xfedcba98                // 000000005888: D65C0025 03FC1725 FEDCBA98
	v_permlanex16_b32 v38, v38, s11, 0xfedcba98                // 000000005894: D65C0026 03FC1726 FEDCBA98
	v_cndmask_b32_e32 v35, v33, v34, vcc_lo                    // 0000000058A0: 02464521
	v_cndmask_b32_e64 v33, v33, v34, s2                        // 0000000058A4: D5010021 000A4521
	v_cmp_eq_f32_e32 vcc_lo, v81, v36                          // 0000000058AC: 7C244951
	v_cmp_lt_f32_e64 s4, v80, v39                              // 0000000058B0: D4110004 02024F50
	v_cmp_eq_f32_e64 s3, v75, v37                              // 0000000058B8: D4120003 02024B4B
	v_cmp_lt_f32_e64 s5, v74, v38                              // 0000000058C0: D4110005 02024D4A
	v_cmp_lt_f32_e64 s2, v81, v36                              // 0000000058C8: D4110002 02024951
	v_permlanex16_b32 v32, v32, s11, 0xfedcba98                // 0000000058D0: D65C0020 03FC1720 FEDCBA98
	v_permlanex16_b32 v40, v40, s11, 0xfedcba98                // 0000000058DC: D65C0028 03FC1728 FEDCBA98
	v_permlanex16_b32 v33, v33, s11, 0xfedcba98                // 0000000058E8: D65C0021 03FC1721 FEDCBA98
	v_permlanex16_b32 v35, v35, s11, 0xfedcba98                // 0000000058F4: D65C0023 03FC1723 FEDCBA98
	v_cmp_lt_f32_e64 s6, v75, v37                              // 000000005900: D4110006 02024B4B
	s_and_b32 s4, vcc_lo, s4                                   // 000000005908: 8B04046A
	s_and_b32 s8, s3, s5                                       // 00000000590C: 8B080503
	s_wait_alu depctr_sa_sdst(0)                               // 000000005910: BF88FF9E
	s_or_b32 vcc_lo, s2, s4                                    // 000000005914: 8C6A0402
	v_cmp_eq_f32_e64 s3, v76, v32                              // 000000005918: D4120003 0202414C
	v_cmp_eq_f32_e64 s4, v78, v40                              // 000000005920: D4120004 0202514E
	v_cmp_lt_f32_e64 s5, v77, v33                              // 000000005928: D4110005 0202434D
	v_cmp_lt_f32_e64 s7, v79, v35                              // 000000005930: D4110007 0202474F
	v_cmp_lt_f32_e64 s2, v76, v32                              // 000000005938: D4110002 0202414C
	s_or_b32 s6, s6, s8                                        // 000000005940: 8C060806
	v_cmp_lt_f32_e64 s8, v78, v40                              // 000000005944: D4110008 0202514E
	s_and_b32 s3, s3, s5                                       // 00000000594C: 8B030503
	s_and_b32 s4, s4, s7                                       // 000000005950: 8B040704
	s_wait_alu depctr_sa_sdst(0)                               // 000000005954: BF88FF9E
	s_or_b32 s2, s2, s3                                        // 000000005958: 8C020302
	v_cndmask_b32_e32 v34, v36, v81, vcc_lo                    // 00000000595C: 0244A324
	s_or_b32 s3, s8, s4                                        // 000000005960: 8C030408
	v_cndmask_b32_e64 v36, v37, v75, s6                        // 000000005964: D5010024 001A9725
	s_wait_alu depctr_sa_sdst(0)                               // 00000000596C: BF88FF9E
	v_cndmask_b32_e64 v32, v32, v76, s2                        // 000000005970: D5010020 000A9920
	v_cndmask_b32_e64 v37, v40, v78, s3                        // 000000005978: D5010025 000E9D28
	v_cndmask_b32_e64 v33, v33, v77, s2                        // 000000005980: D5010021 000A9B21
	v_cndmask_b32_e64 v35, v35, v79, s3                        // 000000005988: D5010023 000E9F23
	v_cndmask_b32_e32 v39, v39, v80, vcc_lo                    // 000000005990: 024EA127
	v_cmp_eq_f32_e32 vcc_lo, v68, v32                          // 000000005994: 7C244144
	v_cmp_eq_f32_e64 s2, v69, v37                              // 000000005998: D4120002 02024B45
	v_cmp_lt_f32_e64 s3, v67, v33                              // 0000000059A0: D4110003 02024343
	v_cmp_lt_f32_e64 s4, v66, v35                              // 0000000059A8: D4110004 02024742
	v_cmp_lt_f32_e64 s5, v68, v32                              // 0000000059B0: D4110005 02024144
	v_cndmask_b32_e64 v38, v38, v74, s6                        // 0000000059B8: D5010026 001A9526
	v_cmp_lt_f32_e64 s6, v69, v37                              // 0000000059C0: D4110006 02024B45
	s_and_b32 s3, vcc_lo, s3                                   // 0000000059C8: 8B03036A
	s_and_b32 s2, s2, s4                                       // 0000000059CC: 8B020402
	s_wait_alu depctr_sa_sdst(0)                               // 0000000059D0: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 0000000059D4: 8C6A0305
	v_cmp_eq_f32_e64 s3, v73, v36                              // 0000000059D8: D4120003 02024949
	s_or_b32 s2, s6, s2                                        // 0000000059E0: 8C020206
	v_cmp_lt_f32_e64 s5, v70, v38                              // 0000000059E4: D4110005 02024D46
	s_wait_alu depctr_sa_sdst(0)                               // 0000000059EC: BF88FF9E
	v_cndmask_b32_e64 v41, v37, v69, s2                        // 0000000059F0: D5010029 000A8B25
	v_cndmask_b32_e64 v43, v35, v66, s2                        // 0000000059F8: D501002B 000A8523
	v_cndmask_b32_e64 v37, v69, v37, s2                        // 000000005A00: D5010025 000A4B45
	v_cndmask_b32_e64 v35, v66, v35, s2                        // 000000005A08: D5010023 000A4742
	v_cmp_lt_f32_e64 s2, v73, v36                              // 000000005A10: D4110002 02024949
	s_and_b32 s3, s3, s5                                       // 000000005A18: 8B030503
	v_cndmask_b32_e32 v40, v32, v68, vcc_lo                    // 000000005A1C: 02508920
	v_cndmask_b32_e32 v42, v33, v67, vcc_lo                    // 000000005A20: 02548721
	v_cndmask_b32_e32 v32, v68, v32, vcc_lo                    // 000000005A24: 02404144
	s_wait_alu depctr_sa_sdst(0)                               // 000000005A28: BF88FF9E
	s_or_b32 s2, s2, s3                                        // 000000005A2C: 8C020302
	v_cndmask_b32_e32 v33, v67, v33, vcc_lo                    // 000000005A30: 02424343
	s_wait_alu depctr_sa_sdst(0)                               // 000000005A34: BF88FF9E
	v_cndmask_b32_e64 v45, v36, v73, s2                        // 000000005A38: D501002D 000A9324
	v_cndmask_b32_e64 v47, v38, v70, s2                        // 000000005A40: D501002F 000A8D26
	v_cndmask_b32_e64 v36, v73, v36, s2                        // 000000005A48: D5010024 000A4949
	v_cndmask_b32_e64 v38, v70, v38, s2                        // 000000005A50: D5010026 000A4D46
	v_cmp_eq_f32_e32 vcc_lo, v71, v34                          // 000000005A58: 7C244547
	v_cmp_eq_f32_e64 s3, v41, v45                              // 000000005A5C: D4120003 02025B29
	v_cmp_lt_f32_e64 s5, v43, v47                              // 000000005A64: D4110005 02025F2B
	v_cmp_lt_f32_e64 s2, v41, v45                              // 000000005A6C: D4110002 02025B29
	v_cmp_lt_f32_e64 s4, v72, v39                              // 000000005A74: D4110004 02024F48
	v_cmp_lt_f32_e64 s6, v71, v34                              // 000000005A7C: D4110006 02024547
	s_and_b32 s3, s3, s5                                       // 000000005A84: 8B030503
	v_cmp_lt_f32_e64 s5, v35, v38                              // 000000005A88: D4110005 02024D23
	s_wait_alu depctr_sa_sdst(0)                               // 000000005A90: BF88FF9E
	s_or_b32 s2, s2, s3                                        // 000000005A94: 8C020302
	v_cmp_eq_f32_e64 s3, v37, v36                              // 000000005A98: D4120003 02024925
	s_wait_alu depctr_sa_sdst(0)                               // 000000005AA0: BF88FF9E
	v_cndmask_b32_e64 v49, v45, v41, s2                        // 000000005AA4: D5010031 000A532D
	v_cndmask_b32_e64 v51, v47, v43, s2                        // 000000005AAC: D5010033 000A572F
	v_cndmask_b32_e64 v41, v41, v45, s2                        // 000000005AB4: D5010029 000A5B29
	v_cndmask_b32_e64 v43, v43, v47, s2                        // 000000005ABC: D501002B 000A5F2B
	v_cmp_lt_f32_e64 s2, v37, v36                              // 000000005AC4: D4110002 02024925
	s_and_b32 s4, vcc_lo, s4                                   // 000000005ACC: 8B04046A
	s_and_b32 s3, s3, s5                                       // 000000005AD0: 8B030503
	s_wait_alu depctr_sa_sdst(0)                               // 000000005AD4: BF88FF9E
	s_or_b32 vcc_lo, s6, s4                                    // 000000005AD8: 8C6A0406
	v_mov_b32_e32 v52, v49                                     // 000000005ADC: 7E680331
	s_or_b32 s2, s2, s3                                        // 000000005AE0: 8C020302
	s_wait_alu depctr_sa_sdst(0)                               // 000000005AE4: BF88FF9E
	v_cndmask_b32_e32 v46, v39, v72, vcc_lo                    // 000000005AE8: 025C9127
	v_cndmask_b32_e64 v45, v36, v37, s2                        // 000000005AEC: D501002D 000A4B24
	v_cndmask_b32_e64 v36, v37, v36, s2                        // 000000005AF4: D5010024 000A4925
	v_cndmask_b32_e32 v44, v34, v71, vcc_lo                    // 000000005AFC: 02588F22
	v_dual_cndmask_b32 v34, v71, v34 :: v_dual_cndmask_b32 v39, v72, v39// 000000005B00: CA524547 22264F48
	v_cmp_lt_f32_e64 s4, v42, v46                              // 000000005B08: D4110004 02025D2A
	v_mov_b32_e32 v54, v51                                     // 000000005B10: 7E6C0333
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005B14: BF870004
	v_cmp_eq_f32_e32 vcc_lo, v40, v44                          // 000000005B18: 7C245928
	v_cmp_lt_f32_e64 s6, v40, v44                              // 000000005B1C: D4110006 02025928
	v_permlanex16_b32 v52, v52, s11, 0xfedcba98                // 000000005B24: D65C0034 03FC1734 FEDCBA98
	v_cndmask_b32_e64 v47, v38, v35, s2                        // 000000005B30: D501002F 000A4726
	v_permlanex16_b32 v54, v54, s11, 0xfedcba98                // 000000005B38: D65C0036 03FC1736 FEDCBA98
	s_and_b32 s4, vcc_lo, s4                                   // 000000005B44: 8B04046A
	v_cndmask_b32_e64 v35, v35, v38, s2                        // 000000005B48: D5010023 000A4D23
	s_wait_alu depctr_sa_sdst(0)                               // 000000005B50: BF88FF9E
	s_or_b32 vcc_lo, s6, s4                                    // 000000005B54: 8C6A0406
	v_cmp_lt_f32_e64 s4, v33, v39                              // 000000005B58: D4110004 02024F21
	s_wait_alu depctr_sa_sdst(0)                               // 000000005B60: BF88FF9E
	v_cndmask_b32_e32 v48, v44, v40, vcc_lo                    // 000000005B64: 0260512C
	v_cndmask_b32_e32 v50, v46, v42, vcc_lo                    // 000000005B68: 0264552E
	v_cndmask_b32_e32 v40, v40, v44, vcc_lo                    // 000000005B6C: 02505928
	v_cndmask_b32_e32 v42, v42, v46, vcc_lo                    // 000000005B70: 02545D2A
	v_cmp_eq_f32_e32 vcc_lo, v32, v34                          // 000000005B74: 7C244520
	v_cmp_lt_f32_e64 s6, v32, v34                              // 000000005B78: D4110006 02024520
	v_mov_b32_e32 v53, v50                                     // 000000005B80: 7E6A0332
	v_mov_b32_e32 v37, v40                                     // 000000005B84: 7E4A0328
	v_cmp_eq_f32_e64 s3, v49, v52                              // 000000005B88: D4120003 02026931
	s_and_b32 s4, vcc_lo, s4                                   // 000000005B90: 8B04046A
	v_cmp_gt_f32_e64 s5, v51, v54                              // 000000005B94: D4140005 02026D33
	s_wait_alu depctr_sa_sdst(0)                               // 000000005B9C: BF88FF9E
	s_or_b32 vcc_lo, s6, s4                                    // 000000005BA0: 8C6A0406
	v_permlanex16_b32 v53, v53, s11, 0xfedcba98                // 000000005BA4: D65C0035 03FC1735 FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 000000005BB0: BF88FF9E
	v_cndmask_b32_e32 v44, v34, v32, vcc_lo                    // 000000005BB4: 02584122
	v_cndmask_b32_e32 v32, v32, v34, vcc_lo                    // 000000005BB8: 02404520
	v_mov_b32_e32 v34, v48                                     // 000000005BBC: 7E440330
	v_dual_cndmask_b32 v46, v39, v33 :: v_dual_cndmask_b32 v33, v33, v39// 000000005BC0: CA524327 2E204F21
	v_cmp_gt_f32_e64 s4, v50, v53                              // 000000005BC8: D4140004 02026B32
	v_mov_b32_e32 v39, v42                                     // 000000005BD0: 7E4E032A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005BD4: BF870004
	v_permlanex16_b32 v34, v34, s11, 0xfedcba98                // 000000005BD8: D65C0022 03FC1722 FEDCBA98
	v_permlanex16_b32 v37, v37, s11, 0xfedcba98                // 000000005BE4: D65C0025 03FC1725 FEDCBA98
	v_cmp_gt_f32_e64 s7, v49, v52                              // 000000005BF0: D4140007 02026931
	s_and_b32 s3, s3, s5                                       // 000000005BF8: 8B030503
	v_permlanex16_b32 v39, v39, s11, 0xfedcba98                // 000000005BFC: D65C0027 03FC1727 FEDCBA98
	v_cmp_eq_f32_e32 vcc_lo, v48, v34                          // 000000005C08: 7C244530
	v_cmp_gt_f32_e64 s6, v48, v34                              // 000000005C0C: D4140006 02024530
	s_wait_alu depctr_sa_sdst(0)                               // 000000005C14: BF88FF9E
	s_or_b32 s3, s7, s3                                        // 000000005C18: 8C030307
	v_mov_b32_e32 v38, v41                                     // 000000005C1C: 7E4C0329
	s_and_b32 s2, vcc_lo, s4                                   // 000000005C20: 8B02046A
	v_cmp_gt_f32_e64 s4, v42, v39                              // 000000005C24: D4140004 02024F2A
	s_wait_alu depctr_sa_sdst(0)                               // 000000005C2C: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 000000005C30: 8C020206
	v_cmp_gt_f32_e64 s6, v40, v37                              // 000000005C34: D4140006 02024B28
	s_wait_alu depctr_sa_sdst(0)                               // 000000005C3C: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000005C40: 8D6A0002
	s_xor_b32 s2, s3, s0                                       // 000000005C44: 8D020003
	s_wait_alu depctr_sa_sdst(0)                               // 000000005C48: BF88FF9E
	v_cndmask_b32_e32 v34, v48, v34, vcc_lo                    // 000000005C4C: 02444530
	v_cndmask_b32_e32 v50, v50, v53, vcc_lo                    // 000000005C50: 02646B32
	v_cmp_eq_f32_e32 vcc_lo, v40, v37                          // 000000005C54: 7C244B28
	v_mov_b32_e32 v48, v43                                     // 000000005C58: 7E60032B
	v_cndmask_b32_e64 v49, v49, v52, s2                        // 000000005C5C: D5010031 000A6931
	v_cndmask_b32_e64 v51, v51, v54, s2                        // 000000005C64: D5010033 000A6D33
	v_permlanex16_b32 v38, v38, s11, 0xfedcba98                // 000000005C6C: D65C0026 03FC1726 FEDCBA98
	s_and_b32 s2, vcc_lo, s4                                   // 000000005C78: 8B02046A
	v_permlanex16_b32 v48, v48, s11, 0xfedcba98                // 000000005C7C: D65C0030 03FC1730 FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 000000005C88: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 000000005C8C: 8C020206
	v_mov_b32_e32 v53, v46                                     // 000000005C90: 7E6A032E
	s_wait_alu depctr_sa_sdst(0)                               // 000000005C94: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000005C98: 8D6A0002
	v_cmp_eq_f32_e64 s3, v41, v38                              // 000000005C9C: D4120003 02024D29
	s_wait_alu depctr_sa_sdst(0)                               // 000000005CA4: BF88FF9E
	v_cndmask_b32_e32 v37, v40, v37, vcc_lo                    // 000000005CA8: 024A4B28
	v_mov_b32_e32 v40, v44                                     // 000000005CAC: 7E50032C
	v_cmp_gt_f32_e64 s5, v43, v48                              // 000000005CB0: D4140005 0202612B
	v_permlanex16_b32 v53, v53, s11, 0xfedcba98                // 000000005CB8: D65C0035 03FC1735 FEDCBA98
	v_cmp_gt_f32_e64 s7, v41, v38                              // 000000005CC4: D4140007 02024D29
	v_cndmask_b32_e32 v39, v42, v39, vcc_lo                    // 000000005CCC: 024E4F2A
	v_permlanex16_b32 v40, v40, s11, 0xfedcba98                // 000000005CD0: D65C0028 03FC1728 FEDCBA98
	s_and_b32 s3, s3, s5                                       // 000000005CDC: 8B030503
	v_cmp_gt_f32_e64 s4, v46, v53                              // 000000005CE0: D4140004 02026B2E
	s_wait_alu depctr_sa_sdst(0)                               // 000000005CE8: BF88FF9E
	s_or_b32 s3, s7, s3                                        // 000000005CEC: 8C030307
	v_mov_b32_e32 v52, v45                                     // 000000005CF0: 7E68032D
	v_cmp_eq_f32_e32 vcc_lo, v44, v40                          // 000000005CF4: 7C24512C
	v_cmp_gt_f32_e64 s6, v44, v40                              // 000000005CF8: D4140006 0202512C
	s_wait_alu depctr_sa_sdst(0)                               // 000000005D00: BF88FF9E
	s_xor_b32 s2, s3, s0                                       // 000000005D04: 8D020003
	v_mov_b32_e32 v54, v47                                     // 000000005D08: 7E6C032F
	s_wait_alu depctr_sa_sdst(0)                               // 000000005D0C: BF88FF9E
	v_cndmask_b32_e64 v38, v41, v38, s2                        // 000000005D10: D5010026 000A4D29
	v_cndmask_b32_e64 v41, v43, v48, s2                        // 000000005D18: D5010029 000A612B
	s_and_b32 s2, vcc_lo, s4                                   // 000000005D20: 8B02046A
	v_permlanex16_b32 v52, v52, s11, 0xfedcba98                // 000000005D24: D65C0034 03FC1734 FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 000000005D30: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 000000005D34: 8C020206
	v_permlanex16_b32 v54, v54, s11, 0xfedcba98                // 000000005D38: D65C0036 03FC1736 FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 000000005D44: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000005D48: 8D6A0002
	v_mov_b32_e32 v42, v32                                     // 000000005D4C: 7E540320
	s_wait_alu depctr_sa_sdst(0)                               // 000000005D50: BF88FF9E
	v_cndmask_b32_e32 v40, v44, v40, vcc_lo                    // 000000005D54: 0250512C
	v_dual_mov_b32 v44, v33 :: v_dual_mov_b32 v43, v36         // 000000005D58: CA100121 2C2A0124
	v_mov_b32_e32 v48, v35                                     // 000000005D60: 7E600323
	v_cmp_eq_f32_e64 s3, v45, v52                              // 000000005D64: D4120003 0202692D
	v_cmp_gt_f32_e64 s5, v47, v54                              // 000000005D6C: D4140005 02026D2F
	v_cmp_gt_f32_e64 s7, v45, v52                              // 000000005D74: D4140007 0202692D
	v_permlanex16_b32 v42, v42, s11, 0xfedcba98                // 000000005D7C: D65C002A 03FC172A FEDCBA98
	v_permlanex16_b32 v44, v44, s11, 0xfedcba98                // 000000005D88: D65C002C 03FC172C FEDCBA98
	v_permlanex16_b32 v43, v43, s11, 0xfedcba98                // 000000005D94: D65C002B 03FC172B FEDCBA98
	v_permlanex16_b32 v48, v48, s11, 0xfedcba98                // 000000005DA0: D65C0030 03FC1730 FEDCBA98
	s_and_b32 s3, s3, s5                                       // 000000005DAC: 8B030503
	v_cndmask_b32_e32 v46, v46, v53, vcc_lo                    // 000000005DB0: 025C6B2E
	s_wait_alu depctr_sa_sdst(0)                               // 000000005DB4: BF88FF9E
	s_or_b32 s3, s7, s3                                        // 000000005DB8: 8C030307
	v_cmp_eq_f32_e32 vcc_lo, v32, v42                          // 000000005DBC: 7C245520
	v_cmp_gt_f32_e64 s4, v33, v44                              // 000000005DC0: D4140004 02025921
	s_wait_alu depctr_sa_sdst(0)                               // 000000005DC8: BF88FF9E
	s_xor_b32 s2, s3, s0                                       // 000000005DCC: 8D020003
	v_cmp_eq_f32_e64 s3, v36, v43                              // 000000005DD0: D4120003 02025724
	v_cmp_gt_f32_e64 s5, v35, v48                              // 000000005DD8: D4140005 02026123
	v_cmp_gt_f32_e64 s6, v32, v42                              // 000000005DE0: D4140006 02025520
	v_cmp_gt_f32_e64 s7, v36, v43                              // 000000005DE8: D4140007 02025724
	s_wait_alu depctr_sa_sdst(0)                               // 000000005DF0: BF88FF9E
	v_cndmask_b32_e64 v45, v45, v52, s2                        // 000000005DF4: D501002D 000A692D
	v_cndmask_b32_e64 v47, v47, v54, s2                        // 000000005DFC: D501002F 000A6D2F
	s_and_b32 s2, vcc_lo, s4                                   // 000000005E04: 8B02046A
	s_and_b32 s3, s3, s5                                       // 000000005E08: 8B030503
	s_wait_alu depctr_sa_sdst(0)                               // 000000005E0C: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 000000005E10: 8C020206
	s_or_b32 s3, s7, s3                                        // 000000005E14: 8C030307
	s_wait_alu depctr_sa_sdst(0)                               // 000000005E18: BF88FF9E
	s_xor_b32 vcc_lo, s2, s0                                   // 000000005E1C: 8D6A0002
	s_xor_b32 s2, s3, s0                                       // 000000005E20: 8D020003
	s_wait_alu depctr_sa_sdst(0)                               // 000000005E24: BF88FF9E
	v_dual_cndmask_b32 v32, v32, v42 :: v_dual_cndmask_b32 v33, v33, v44// 000000005E28: CA525520 20205921
	v_cmp_eq_f32_e32 vcc_lo, v49, v34                          // 000000005E30: 7C244531
	v_cmp_lt_f32_e64 s4, v50, v51                              // 000000005E34: D4110004 02026732
	v_cndmask_b32_e64 v36, v36, v43, s2                        // 000000005E3C: D5010024 000A5724
	v_cmp_lt_f32_e64 s3, v51, v50                              // 000000005E44: D4110003 02026533
	v_cndmask_b32_e64 v35, v35, v48, s2                        // 000000005E4C: D5010023 000A6123
	v_cmp_lt_f32_e64 s2, v34, v49                              // 000000005E54: D4110002 02026322
	v_cmp_lt_f32_e64 s5, v49, v34                              // 000000005E5C: D4110005 02024531
	s_and_b32 s4, vcc_lo, s4                                   // 000000005E64: 8B04046A
	s_and_b32 s3, vcc_lo, s3                                   // 000000005E68: 8B03036A
	v_cmp_lt_f32_e64 s6, v33, v35                              // 000000005E6C: D4110006 02024721
	s_wait_alu depctr_sa_sdst(0)                               // 000000005E74: BF88FF9E
	s_or_b32 s2, s2, s4                                        // 000000005E78: 8C020402
	s_or_b32 vcc_lo, s5, s3                                    // 000000005E7C: 8C6A0305
	s_wait_alu depctr_sa_sdst(0)                               // 000000005E80: BF88FF9E
	v_cndmask_b32_e64 v69, v34, v49, s2                        // 000000005E84: D5010045 000A6322
	v_cndmask_b32_e64 v66, v50, v51, s2                        // 000000005E8C: D5010042 000A6732
	v_cmp_eq_f32_e64 s2, v38, v37                              // 000000005E94: D4120002 02024B26
	v_cmp_lt_f32_e64 s3, v41, v39                              // 000000005E9C: D4110003 02024F29
	v_cmp_lt_f32_e64 s4, v39, v41                              // 000000005EA4: D4110004 02025327
	v_cmp_eq_f32_e64 s5, v36, v32                              // 000000005EAC: D4120005 02024124
	v_cndmask_b32_e32 v68, v34, v49, vcc_lo                    // 000000005EB4: 02886322
	v_cndmask_b32_e32 v67, v50, v51, vcc_lo                    // 000000005EB8: 02866732
	s_and_b32 s8, s2, s3                                       // 000000005EBC: 8B080302
	v_cmp_lt_f32_e64 s3, v32, v36                              // 000000005EC0: D4110003 02024920
	s_and_b32 s4, s2, s4                                       // 000000005EC8: 8B040402
	s_and_b32 s2, s5, s6                                       // 000000005ECC: 8B020605
	v_cmp_lt_f32_e32 vcc_lo, v38, v37                          // 000000005ED0: 7C224B26
	v_cmp_lt_f32_e64 s7, v37, v38                              // 000000005ED4: D4110007 02024D25
	s_wait_alu depctr_sa_sdst(0)                               // 000000005EDC: BF88FF9E
	s_or_b32 s2, s3, s2                                        // 000000005EE0: 8C020203
	v_cmp_lt_f32_e64 s3, v47, v46                              // 000000005EE4: D4110003 02025D2F
	s_wait_alu depctr_sa_sdst(0)                               // 000000005EEC: BF88FF9E
	v_cndmask_b32_e64 v75, v32, v36, s2                        // 000000005EF0: D501004B 000A4920
	v_cndmask_b32_e64 v74, v33, v35, s2                        // 000000005EF8: D501004A 000A4721
	s_or_b32 vcc_lo, vcc_lo, s8                                // 000000005F00: 8C6A086A
	s_or_b32 s2, s7, s4                                        // 000000005F04: 8C020407
	s_wait_alu depctr_sa_sdst(0)                               // 000000005F08: BF88FF9E
	v_cndmask_b32_e32 v71, v37, v38, vcc_lo                    // 000000005F0C: 028E4D25
	v_cndmask_b32_e64 v73, v37, v38, s2                        // 000000005F10: D5010049 000A4D25
	v_dual_max_num_f32 v34, v75, v75 :: v_dual_max_num_f32 v37, v74, v74// 000000005F18: CA94974B 2224954A
	v_cndmask_b32_e64 v70, v39, v41, s2                        // 000000005F20: D5010046 000A5327
	v_cmp_eq_f32_e64 s2, v45, v40                              // 000000005F28: D4120002 0202512D
	v_cndmask_b32_e32 v72, v39, v41, vcc_lo                    // 000000005F30: 02905327
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005F34: BF870004
	v_dual_max_num_f32 v34, 0xff800000, v34 :: v_dual_max_num_f32 v37, 0xff800000, v37// 000000005F38: CA9444FF 22244AFF FF800000
	v_cmp_lt_f32_e32 vcc_lo, v45, v40                          // 000000005F44: 7C22512D
	v_cmp_lt_f32_e64 s4, v46, v47                              // 000000005F48: D4110004 02025F2E
	s_and_b32 s3, s2, s3                                       // 000000005F50: 8B030302
	v_cndmask_b32_e64 v34, v34, 0xff800000, s1                 // 000000005F54: D5010022 0005FF22 FF800000
	v_cndmask_b32_e64 v37, v37, 0xff800000, s1                 // 000000005F60: D5010025 0005FF25 FF800000
	v_cmp_lt_f32_e64 s6, v40, v45                              // 000000005F6C: D4110006 02025B28
	s_wait_alu depctr_sa_sdst(0)                               // 000000005F74: BF88FF9E
	s_or_b32 vcc_lo, vcc_lo, s3                                // 000000005F78: 8C6A036A
	s_and_b32 s2, s2, s4                                       // 000000005F7C: 8B020402
	v_dual_mov_b32 v38, v34 :: v_dual_mov_b32 v39, v37         // 000000005F80: CA100122 26260125
	s_wait_alu depctr_sa_sdst(0)                               // 000000005F88: BF88FF9E
	v_dual_cndmask_b32 v76, v40, v45 :: v_dual_cndmask_b32 v77, v46, v47// 000000005F8C: CA525B28 4C4C5F2E
	v_cmp_lt_f32_e32 vcc_lo, v35, v33                          // 000000005F94: 7C224323
	s_delay_alu instid0(VALU_DEP_3)                            // 000000005F98: BF870003
	v_permlanex16_b32 v38, v38, s11, 0xfedcba98                // 000000005F9C: D65C0026 03FC1726 FEDCBA98
	v_permlanex16_b32 v39, v39, s11, 0xfedcba98                // 000000005FA8: D65C0027 03FC1727 FEDCBA98
	v_cmp_lt_f32_e64 s3, v36, v32                              // 000000005FB4: D4110003 02024124
	s_or_b32 s2, s6, s2                                        // 000000005FBC: 8C020206
	v_dual_max_num_f32 v38, v38, v38 :: v_dual_max_num_f32 v39, v39, v39// 000000005FC0: CA944D26 26264F27
	s_wait_alu depctr_sa_sdst(0)                               // 000000005FC8: BF88FF9E
	v_cndmask_b32_e64 v78, v40, v45, s2                        // 000000005FCC: D501004E 000A5B28
	v_cndmask_b32_e64 v79, v46, v47, s2                        // 000000005FD4: D501004F 000A5F2E
	s_and_b32 s2, s5, vcc_lo                                   // 000000005FDC: 8B026A05
	s_wait_alu depctr_sa_sdst(0)                               // 000000005FE0: BF88FF9E
	s_or_b32 vcc_lo, s3, s2                                    // 000000005FE4: 8C6A0203
	s_wait_alu depctr_sa_sdst(0)                               // 000000005FE8: BF88FF9E
	v_dual_cndmask_b32 v81, v32, v36 :: v_dual_cndmask_b32 v80, v33, v35// 000000005FEC: CA524920 51504721
	v_dual_max_num_f32 v32, v34, v38 :: v_dual_max_num_f32 v33, v37, v39// 000000005FF4: CA944D22 20204F25
	s_branch 64161                                             // 000000005FFC: BFA0FAA1 <nearest+0x4a84>
	v_cndmask_b32_e64 v34, v90, -v14, s2                       // 000000006000: D5010022 400A1D5A
	v_cndmask_b32_e64 v35, v89, -v15, s3                       // 000000006008: D5010023 400E1F59
	v_cndmask_b32_e64 v36, v88, v30, s2                        // 000000006010: D5010024 000A3D58
	v_cndmask_b32_e64 v37, v87, v31, s3                        // 000000006018: D5010025 000E3F57
	v_cndmask_b32_e64 v84, v84, v90, s2                        // 000000006020: D5010054 000AB554
	v_cndmask_b32_e64 v85, v85, v89, s3                        // 000000006028: D5010055 000EB355
	v_cndmask_b32_e64 v82, v82, v88, s2                        // 000000006030: D5010052 000AB152
	v_cndmask_b32_e64 v83, v83, v87, s3                        // 000000006038: D5010053 000EAF53
	v_dual_mov_b32 v88, v36 :: v_dual_mov_b32 v87, v37         // 000000006040: CA100124 58560125
	v_dual_mov_b32 v90, v34 :: v_dual_mov_b32 v89, v35         // 000000006048: CA100122 5A580123
	s_cmp_ngt_f32 s5, 0.5                                      // 000000006050: BF4BF005
	s_mov_b32 s10, s6                                          // 000000006054: BE8A0006
	s_cbranch_scc0 64674                                       // 000000006058: BFA1FCA2 <nearest+0x52e4>
	s_branch 64137                                             // 00000000605C: BFA0FA89 <nearest+0x4a84>
	v_cndmask_b32_e64 v32, v90, -v14, s2                       // 000000006060: D5010020 400A1D5A
	v_cndmask_b32_e64 v33, v89, -v15, s3                       // 000000006068: D5010021 400E1F59
	v_cndmask_b32_e64 v34, v88, v30, s2                        // 000000006070: D5010022 000A3D58
	v_cndmask_b32_e64 v35, v87, v31, s3                        // 000000006078: D5010023 000E3F57
	v_cndmask_b32_e64 v84, v84, v90, s2                        // 000000006080: D5010054 000AB554
	v_cndmask_b32_e64 v85, v85, v89, s3                        // 000000006088: D5010055 000EB355
	v_cndmask_b32_e64 v82, v82, v88, s2                        // 000000006090: D5010052 000AB152
	v_cndmask_b32_e64 v83, v83, v87, s3                        // 000000006098: D5010053 000EAF53
	v_dual_mov_b32 v89, v33 :: v_dual_mov_b32 v90, v32         // 0000000060A0: CA100121 595A0120
	v_dual_mov_b32 v87, v35 :: v_dual_mov_b32 v88, v34         // 0000000060A8: CA100123 57580122
	s_cmp_ngt_f32 s6, 0.5                                      // 0000000060B0: BF4BF006
	s_cbranch_scc0 63275                                       // 0000000060B4: BFA1F72B <nearest+0x3d64>
	v_lshlrev_b32_e32 v0, 1, v65                               // 0000000060B8: 30008281
	v_lshlrev_b32_e32 v1, 4, v64                               // 0000000060BC: 30028084
	s_lshl_b32 s1, ttmp9, 10                                   // 0000000060C0: 84018A75
	v_cmp_eq_f32_e32 vcc_lo, v89, v90                          // 0000000060C4: 7C24B559
	v_cmp_lt_f32_e64 s2, v88, v87                              // 0000000060C8: D4110002 0202AF58
	v_cmp_lt_f32_e64 s3, v89, v90                              // 0000000060D0: D4110003 0202B559
	s_wait_alu depctr_sa_sdst(0)                               // 0000000060D8: BF88FF9E
	v_or3_b32 v0, v1, s1, v0                                   // 0000000060DC: D6580000 04000301
	v_cmp_lt_f32_e64 s1, v87, v88                              // 0000000060E4: D4110001 0202B157
	v_cmp_lt_f32_e64 s4, v90, v89                              // 0000000060EC: D4110004 0202B35A
	s_and_b32 s2, s2, vcc_lo                                   // 0000000060F4: 8B026A02
	s_mov_b32 s8, 0x76543210                                   // 0000000060F8: BE8800FF 76543210
	v_ashrrev_i32_e32 v1, 31, v0                               // 000000006100: 3402009F
	s_and_b32 s1, s1, vcc_lo                                   // 000000006104: 8B016A01
	s_wait_alu depctr_sa_sdst(0)                               // 000000006108: BF88FF9E
	s_or_b32 s2, s4, s2                                        // 00000000610C: 8C020204
	s_or_b32 s1, s3, s1                                        // 000000006110: 8C010103
	v_cmp_lt_f32_e64 s4, v85, v84                              // 000000006114: D4110004 0202A955
	v_lshlrev_b64_e32 v[2:3], 2, v[0:1]                        // 00000000611C: 3E040082
	s_wait_alu depctr_sa_sdst(0)                               // 000000006120: BF88FF9E
	s_xor_b32 vcc_lo, s1, s0                                   // 000000006124: 8D6A0001
	v_cmp_lt_f32_e64 s3, v82, v83                              // 000000006128: D4110003 0202A752
	s_wait_alu depctr_sa_sdst(0)                               // 000000006130: BF88FF9E
	v_cndmask_b32_e32 v4, v90, v89, vcc_lo                     // 000000006134: 0208B35A
	v_cndmask_b32_e32 v6, v88, v87, vcc_lo                     // 000000006138: 020CAF58
	v_cmp_eq_f32_e32 vcc_lo, v85, v84                          // 00000000613C: 7C24A955
	v_add_co_u32 v0, s1, s18, v2                               // 000000006140: D7000100 02020412
	s_wait_alu depctr_va_sdst(0)                               // 000000006148: BF88F19F
	v_add_co_ci_u32_e64 v1, null, s19, v3, s1                  // 00000000614C: D5207C01 00060613
	s_xor_b32 s1, s2, s9                                       // 000000006154: 8D010902
	v_cmp_lt_f32_e64 s2, v83, v82                              // 000000006158: D4110002 0202A553
	s_wait_alu depctr_sa_sdst(0)                               // 000000006160: BF88FF9E
	v_cndmask_b32_e64 v5, v89, v90, s1                         // 000000006164: D5010005 0006B559
	v_cndmask_b32_e64 v7, v87, v88, s1                         // 00000000616C: D5010007 0006B157
	v_mov_b32_e32 v9, v4                                       // 000000006174: 7E120304
	v_cmp_lt_f32_e64 s5, v84, v85                              // 000000006178: D4110005 0202AB54
	s_and_b32 s1, s2, vcc_lo                                   // 000000006180: 8B016A02
	s_and_b32 s2, s3, vcc_lo                                   // 000000006184: 8B026A03
	s_wait_alu depctr_sa_sdst(0)                               // 000000006188: BF88FF9E
	s_or_b32 s1, s4, s1                                        // 00000000618C: 8C010104
	v_permlanex16_b32 v9, v9, s8, 0xfedcba98                   // 000000006190: D65C0009 03FC1109 FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 00000000619C: BF88FF9E
	s_xor_b32 vcc_lo, s1, s0                                   // 0000000061A0: 8D6A0001
	s_or_b32 s2, s5, s2                                        // 0000000061A4: 8C020205
	s_wait_alu depctr_sa_sdst(0)                               // 0000000061A8: BF88FF9E
	v_cndmask_b32_e32 v14, v82, v83, vcc_lo                    // 0000000061AC: 021CA752
	s_xor_b32 s1, s2, s9                                       // 0000000061B0: 8D010902
	v_cmp_gt_f32_e64 s5, v4, v9                                // 0000000061B4: D4140005 02021304
	s_wait_alu depctr_sa_sdst(0)                               // 0000000061BC: BF88FF9E
	v_cndmask_b32_e64 v13, v85, v84, s1                        // 0000000061C0: D501000D 0006A955
	v_cndmask_b32_e64 v15, v83, v82, s1                        // 0000000061C8: D501000F 0006A553
	v_dual_mov_b32 v17, v14 :: v_dual_cndmask_b32 v8, v84, v85 // 0000000061D0: CA12010E 1108AB54
	v_mov_b32_e32 v11, v6                                      // 0000000061D8: 7E160306
	v_cmp_eq_f32_e32 vcc_lo, v4, v9                            // 0000000061DC: 7C241304
	v_mov_b32_e32 v10, v5                                      // 0000000061E0: 7E140305
	v_mov_b32_e32 v16, v13                                     // 0000000061E4: 7E20030D
	v_mov_b32_e32 v18, v15                                     // 0000000061E8: 7E24030F
	v_permlanex16_b32 v11, v11, s8, 0xfedcba98                 // 0000000061EC: D65C000B 03FC110B FEDCBA98
	v_permlanex16_b32 v17, v17, s8, 0xfedcba98                 // 0000000061F8: D65C0011 03FC1111 FEDCBA98
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000006204: BF870214
	v_permlanex16_b32 v16, v16, s8, 0xfedcba98                 // 000000006208: D65C0010 03FC1110 FEDCBA98
	v_permlanex16_b32 v18, v18, s8, 0xfedcba98                 // 000000006214: D65C0012 03FC1112 FEDCBA98
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006220: BF870004
	v_cmp_gt_f32_e64 s3, v6, v11                               // 000000006224: D4140003 02021706
	s_and_b32 s1, vcc_lo, s3                                   // 00000000622C: 8B01036A
	v_cmp_neq_f32_e64 s3, v13, v16                             // 000000006230: D41D0003 0202210D
	s_wait_alu depctr_sa_sdst(0)                               // 000000006238: BF88FF9E
	s_or_b32 s1, s5, s1                                        // 00000000623C: 8C010105
	s_wait_alu depctr_sa_sdst(0)                               // 000000006240: BF88FF9E
	s_xor_b32 vcc_lo, s1, s0                                   // 000000006244: 8D6A0001
	s_wait_alu depctr_sa_sdst(0)                               // 000000006248: BF88FF9E
	v_cndmask_b32_e32 v6, v6, v11, vcc_lo                      // 00000000624C: 020C1706
	v_mov_b32_e32 v12, v7                                      // 000000006250: 7E180307
	v_cndmask_b32_e32 v4, v4, v9, vcc_lo                       // 000000006254: 02081304
	v_mov_b32_e32 v9, v8                                       // 000000006258: 7E120308
	v_permlanex16_b32 v10, v10, s8, 0xfedcba98                 // 00000000625C: D65C000A 03FC110A FEDCBA98
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000006268: BF870194
	v_permlanex16_b32 v12, v12, s8, 0xfedcba98                 // 00000000626C: D65C000C 03FC110C FEDCBA98
	v_permlanex16_b32 v9, v9, s8, 0xfedcba98                   // 000000006278: D65C0009 03FC1109 FEDCBA98
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 000000006284: BF870223
	v_cmp_eq_f32_e64 s2, v5, v10                               // 000000006288: D4120002 02021505
	v_cmp_gt_f32_e64 s6, v5, v10                               // 000000006290: D4140006 02021505
	v_cmp_gt_f32_e64 s4, v7, v12                               // 000000006298: D4140004 02021907
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000062A0: BF870004
	v_cmp_neq_f32_e32 vcc_lo, v8, v9                           // 0000000062A4: 7C3A1308
	v_cmp_ngt_f32_e64 s5, v8, v9                               // 0000000062A8: D41B0005 02021308
	s_and_b32 s2, s2, s4                                       // 0000000062B0: 8B020402
	v_cmp_ngt_f32_e64 s4, v15, v18                             // 0000000062B4: D41B0004 0202250F
	s_wait_alu depctr_sa_sdst(0)                               // 0000000062BC: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 0000000062C0: 8C020206
	v_cmp_ngt_f32_e64 s6, v13, v16                             // 0000000062C4: D41B0006 0202210D
	s_wait_alu depctr_sa_sdst(0)                               // 0000000062CC: BF88FF9E
	s_xor_b32 s1, s2, s0                                       // 0000000062D0: 8D010002
	v_cmp_ngt_f32_e64 s2, v14, v17                             // 0000000062D4: D41B0002 0202230E
	s_wait_alu depctr_sa_sdst(0)                               // 0000000062DC: BF88FF9E
	v_cndmask_b32_e64 v5, v5, v10, s1                          // 0000000062E0: D5010005 00061505
	v_cndmask_b32_e64 v7, v7, v12, s1                          // 0000000062E8: D5010007 00061907
	s_or_b32 s1, vcc_lo, s2                                    // 0000000062F0: 8C01026A
	s_or_b32 s2, s3, s4                                        // 0000000062F4: 8C020403
	s_wait_alu depctr_sa_sdst(0)                               // 0000000062F8: BF88FF9E
	s_and_b32 s1, s5, s1                                       // 0000000062FC: 8B010105
	s_and_b32 s2, s6, s2                                       // 000000006300: 8B020206
	s_wait_alu depctr_sa_sdst(0)                               // 000000006304: BF88FF9E
	s_xor_b32 vcc_lo, s1, s0                                   // 000000006308: 8D6A0001
	s_xor_b32 s1, s2, s0                                       // 00000000630C: 8D010002
	s_wait_alu depctr_sa_sdst(0)                               // 000000006310: BF88FF9E
	v_cndmask_b32_e32 v8, v8, v9, vcc_lo                       // 000000006314: 02101308
	v_cndmask_b32_e32 v10, v14, v17, vcc_lo                    // 000000006318: 0214230E
	v_cmp_eq_f32_e32 vcc_lo, v5, v4                            // 00000000631C: 7C240905
	v_cmp_lt_f32_e64 s2, v7, v6                                // 000000006320: D4110002 02020D07
	v_cmp_lt_f32_e64 s3, v6, v7                                // 000000006328: D4110003 02020F06
	v_cndmask_b32_e64 v9, v13, v16, s1                         // 000000006330: D5010009 0006210D
	v_cmp_lt_f32_e64 s4, v5, v4                                // 000000006338: D4110004 02020905
	v_cndmask_b32_e64 v11, v15, v18, s1                        // 000000006340: D501000B 0006250F
	v_cmp_lt_f32_e64 s1, v4, v5                                // 000000006348: D4110001 02020B04
	s_and_b32 s2, vcc_lo, s2                                   // 000000006350: 8B02026A
	s_and_b32 s3, vcc_lo, s3                                   // 000000006354: 8B03036A
	s_wait_alu depctr_sa_sdst(0)                               // 000000006358: BF88FF9E
	s_or_b32 vcc_lo, s4, s2                                    // 00000000635C: 8C6A0204
	v_cmp_lt_f32_e64 s2, v11, v10                              // 000000006360: D4110002 0202150B
	s_or_b32 s1, s1, s3                                        // 000000006368: 8C010301
	s_wait_alu depctr_sa_sdst(0)                               // 00000000636C: BF88FF9E
	v_cndmask_b32_e32 v12, v4, v5, vcc_lo                      // 000000006370: 02180B04
	v_cndmask_b32_e64 v4, v4, v5, s1                           // 000000006374: D5010004 00060B04
	v_cndmask_b32_e32 v5, v6, v7, vcc_lo                       // 00000000637C: 020A0F06
	v_cmp_eq_f32_e32 vcc_lo, v9, v8                            // 000000006380: 7C241109
	v_cmp_lt_f32_e64 s3, v10, v11                              // 000000006384: D4110003 0202170A
	v_cmp_lt_f32_e64 s4, v9, v8                                // 00000000638C: D4110004 02021109
	v_cndmask_b32_e64 v6, v6, v7, s1                           // 000000006394: D5010006 00060F06
	v_cmp_lt_f32_e64 s1, v8, v9                                // 00000000639C: D4110001 02021308
	s_and_b32 s2, vcc_lo, s2                                   // 0000000063A4: 8B02026A
	s_and_b32 s3, vcc_lo, s3                                   // 0000000063A8: 8B03036A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000063AC: BF88FF9E
	s_or_b32 vcc_lo, s4, s2                                    // 0000000063B0: 8C6A0204
	v_add_co_u32 v2, s5, s12, v2                               // 0000000063B4: D7000502 0202040C
	s_or_b32 s1, s1, s3                                        // 0000000063BC: 8C010301
	s_wait_alu depctr_sa_sdst(0)                               // 0000000063C0: BF88FF9E
	v_cndmask_b32_e32 v7, v9, v8, vcc_lo                       // 0000000063C4: 020E1109
	v_cndmask_b32_e64 v8, v9, v8, s1                           // 0000000063C8: D5010008 00061109
	v_cndmask_b32_e32 v9, v11, v10, vcc_lo                     // 0000000063D0: 0212150B
	v_cndmask_b32_e64 v10, v11, v10, s1                        // 0000000063D4: D501000A 0006150B
	v_add_co_ci_u32_e64 v3, null, s13, v3, s5                  // 0000000063DC: D5207C03 0016060D
	v_cmp_eq_f32_e64 s1, v12, v7                               // 0000000063E4: D4120001 02020F0C
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000063EC: BF870004
	v_cmp_lt_f32_e64 s3, v5, v9                                // 0000000063F0: D4110003 02021305
	v_cmp_lt_f32_e32 vcc_lo, v12, v7                           // 0000000063F8: 7C220F0C
	v_cmp_eq_f32_e64 s2, v4, v8                                // 0000000063FC: D4120002 02021104
	v_cmp_lt_f32_e64 s4, v6, v10                               // 000000006404: D4110004 02021506
	v_cmp_lt_f32_e64 s5, v4, v8                                // 00000000640C: D4110005 02021104
	s_and_b32 s1, s1, s3                                       // 000000006414: 8B010301
	s_wait_alu depctr_sa_sdst(0)                               // 000000006418: BF88FF9E
	s_or_b32 vcc_lo, vcc_lo, s1                                // 00000000641C: 8C6A016A
	s_and_b32 s2, s2, s4                                       // 000000006420: 8B020402
	s_wait_alu depctr_sa_sdst(0)                               // 000000006424: BF88FF9E
	v_dual_cndmask_b32 v11, v7, v12 :: v_dual_cndmask_b32 v14, v9, v5// 000000006428: CA521907 0B0E0B09
	v_cndmask_b32_e32 v7, v12, v7, vcc_lo                      // 000000006430: 020E0F0C
	v_cndmask_b32_e32 v5, v5, v9, vcc_lo                       // 000000006434: 020A1305
	s_or_b32 s1, s5, s2                                        // 000000006438: 8C010205
	s_delay_alu instid0(VALU_DEP_3)                            // 00000000643C: BF870003
	v_dual_mov_b32 v12, v11 :: v_dual_mov_b32 v17, v14         // 000000006440: CA10010B 0C10010E
	s_wait_alu depctr_sa_sdst(0)                               // 000000006448: BF88FF9E
	v_cndmask_b32_e64 v13, v8, v4, s1                          // 00000000644C: D501000D 00060908
	v_cndmask_b32_e64 v15, v10, v6, s1                         // 000000006454: D501000F 00060D0A
	v_cndmask_b32_e64 v4, v4, v8, s1                           // 00000000645C: D5010004 00061104
	v_permlanex16_b32 v12, v12, s8, 0xfedcba98                 // 000000006464: D65C000C 03FC110C FEDCBA98
	v_permlanex16_b32 v17, v17, s8, 0xfedcba98                 // 000000006470: D65C0011 03FC1111 FEDCBA98
	v_cndmask_b32_e64 v6, v6, v10, s1                          // 00000000647C: D5010006 00061506
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000006484: BF870214
	v_dual_mov_b32 v9, v7 :: v_dual_mov_b32 v10, v4            // 000000006488: CA100107 090A0104
	v_cmp_eq_f32_e32 vcc_lo, v11, v12                          // 000000006490: 7C24190B
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 000000006494: BF870224
	v_cmp_gt_f32_e64 s3, v14, v17                              // 000000006498: D4140003 0202230E
	v_cmp_gt_f32_e64 s5, v11, v12                              // 0000000064A0: D4140005 0202190B
	v_permlanex16_b32 v9, v9, s8, 0xfedcba98                   // 0000000064A8: D65C0009 03FC1109 FEDCBA98
	v_mov_b32_e32 v16, v13                                     // 0000000064B4: 7E20030D
	v_permlanex16_b32 v10, v10, s8, 0xfedcba98                 // 0000000064B8: D65C000A 03FC110A FEDCBA98
	s_and_b32 s1, vcc_lo, s3                                   // 0000000064C4: 8B01036A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000064C8: BF88FF9E
	s_or_b32 s1, s5, s1                                        // 0000000064CC: 8C010105
	v_permlanex16_b32 v16, v16, s8, 0xfedcba98                 // 0000000064D0: D65C0010 03FC1110 FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 0000000064DC: BF88FF9E
	s_xor_b32 vcc_lo, s1, s0                                   // 0000000064E0: 8D6A0001
	v_cmp_gt_f32_e64 s5, v7, v9                                // 0000000064E4: D4140005 02021307
	s_wait_alu depctr_sa_sdst(0)                               // 0000000064EC: BF88FF9E
	v_cndmask_b32_e32 v8, v11, v12, vcc_lo                     // 0000000064F0: 0210190B
	v_dual_mov_b32 v18, v15 :: v_dual_mov_b32 v11, v5          // 0000000064F4: CA10010F 120A0105
	v_mov_b32_e32 v12, v6                                      // 0000000064FC: 7E180306
	v_cmp_eq_f32_e64 s2, v13, v16                              // 000000006500: D4120002 0202210D
	v_cmp_gt_f32_e64 s6, v13, v16                              // 000000006508: D4140006 0202210D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006510: BF870004
	v_permlanex16_b32 v18, v18, s8, 0xfedcba98                 // 000000006514: D65C0012 03FC1112 FEDCBA98
	v_permlanex16_b32 v11, v11, s8, 0xfedcba98                 // 000000006520: D65C000B 03FC110B FEDCBA98
	v_permlanex16_b32 v12, v12, s8, 0xfedcba98                 // 00000000652C: D65C000C 03FC110C FEDCBA98
	v_cndmask_b32_e32 v14, v14, v17, vcc_lo                    // 000000006538: 021C230E
	v_cmp_eq_f32_e32 vcc_lo, v7, v9                            // 00000000653C: 7C241307
	v_cmp_gt_f32_e64 s4, v15, v18                              // 000000006540: D4140004 0202250F
	v_cmp_gt_f32_e64 s3, v5, v11                               // 000000006548: D4140003 02021705
	s_and_b32 s2, s2, s4                                       // 000000006550: 8B020402
	v_cmp_gt_f32_e64 s4, v6, v12                               // 000000006554: D4140004 02021906
	s_wait_alu depctr_sa_sdst(0)                               // 00000000655C: BF88FF9E
	s_or_b32 s2, s6, s2                                        // 000000006560: 8C020206
	v_cmp_gt_f32_e64 s6, v4, v10                               // 000000006564: D4140006 02021504
	s_wait_alu depctr_sa_sdst(0)                               // 00000000656C: BF88FF9E
	s_xor_b32 s1, s2, s0                                       // 000000006570: 8D010002
	v_cmp_eq_f32_e64 s2, v4, v10                               // 000000006574: D4120002 02021504
	s_wait_alu depctr_sa_sdst(0)                               // 00000000657C: BF88FF9E
	v_cndmask_b32_e64 v13, v13, v16, s1                        // 000000006580: D501000D 0006210D
	v_cndmask_b32_e64 v15, v15, v18, s1                        // 000000006588: D501000F 0006250F
	s_and_b32 s1, vcc_lo, s3                                   // 000000006590: 8B01036A
	s_and_b32 s2, s2, s4                                       // 000000006594: 8B020402
	s_wait_alu depctr_sa_sdst(0)                               // 000000006598: BF88FF9E
	s_or_b32 s1, s5, s1                                        // 00000000659C: 8C010105
	s_or_b32 s2, s6, s2                                        // 0000000065A0: 8C020206
	s_wait_alu depctr_sa_sdst(0)                               // 0000000065A4: BF88FF9E
	s_xor_b32 vcc_lo, s1, s0                                   // 0000000065A8: 8D6A0001
	s_xor_b32 s1, s2, s0                                       // 0000000065AC: 8D010002
	s_wait_alu depctr_sa_sdst(0)                               // 0000000065B0: BF88FF9E
	v_cndmask_b32_e32 v7, v7, v9, vcc_lo                       // 0000000065B4: 020E1307
	v_cndmask_b32_e32 v5, v5, v11, vcc_lo                      // 0000000065B8: 020A1705
	v_cmp_eq_f32_e32 vcc_lo, v13, v8                           // 0000000065BC: 7C24110D
	v_cmp_lt_f32_e64 s2, v15, v14                              // 0000000065C0: D4110002 02021D0F
	v_cmp_lt_f32_e64 s3, v14, v15                              // 0000000065C8: D4110003 02021F0E
	v_cmp_lt_f32_e64 s4, v13, v8                               // 0000000065D0: D4110004 0202110D
	v_cndmask_b32_e64 v4, v4, v10, s1                          // 0000000065D8: D5010004 00061504
	v_cndmask_b32_e64 v6, v6, v12, s1                          // 0000000065E0: D5010006 00061906
	v_cmp_lt_f32_e64 s1, v8, v13                               // 0000000065E8: D4110001 02021B08
	s_and_b32 s2, vcc_lo, s2                                   // 0000000065F0: 8B02026A
	s_and_b32 s3, vcc_lo, s3                                   // 0000000065F4: 8B03036A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000065F8: BF88FF9E
	s_or_b32 vcc_lo, s4, s2                                    // 0000000065FC: 8C6A0204
	v_cmp_lt_f32_e64 s2, v6, v5                                // 000000006600: D4110002 02020B06
	s_wait_alu depctr_sa_sdst(0)                               // 000000006608: BF88FF9E
	v_cndmask_b32_e32 v9, v8, v13, vcc_lo                      // 00000000660C: 02121B08
	s_or_b32 s1, s1, s3                                        // 000000006610: 8C010301
	v_cndmask_b32_e32 v10, v14, v15, vcc_lo                    // 000000006614: 02141F0E
	v_cmp_eq_f32_e32 vcc_lo, v4, v7                            // 000000006618: 7C240F04
	v_cmp_lt_f32_e64 s3, v5, v6                                // 00000000661C: D4110003 02020D05
	s_wait_alu depctr_sa_sdst(0)                               // 000000006624: BF88FF9E
	v_cndmask_b32_e64 v8, v8, v13, s1                          // 000000006628: D5010008 00061B08
	v_cmp_lt_f32_e64 s4, v4, v7                                // 000000006630: D4110004 02020F04
	v_cndmask_b32_e64 v11, v14, v15, s1                        // 000000006638: D501000B 00061F0E
	v_cmp_lt_f32_e64 s1, v7, v4                                // 000000006640: D4110001 02020907
	s_and_b32 s2, vcc_lo, s2                                   // 000000006648: 8B02026A
	s_and_b32 s3, vcc_lo, s3                                   // 00000000664C: 8B03036A
	s_wait_alu depctr_sa_sdst(0)                               // 000000006650: BF88FF9E
	s_or_b32 vcc_lo, s4, s2                                    // 000000006654: 8C6A0204
	v_permlanex16_b32 v8, v8, s8, 0xfedcba98                   // 000000006658: D65C0008 03FC1108 FEDCBA98
	s_or_b32 s1, s1, s3                                        // 000000006664: 8C010301
	v_permlanex16_b32 v11, v11, s8, 0xfedcba98                 // 000000006668: D65C000B 03FC110B FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 000000006674: BF88FF9E
	v_cndmask_b32_e32 v12, v7, v4, vcc_lo                      // 000000006678: 02180907
	v_cndmask_b32_e64 v4, v7, v4, s1                           // 00000000667C: D5010004 00060907
	v_permlanex16_b32 v9, v9, s8, 0xfedcba98                   // 000000006684: D65C0009 03FC1109 FEDCBA98
	v_permlanex16_b32 v10, v10, s8, 0xfedcba98                 // 000000006690: D65C000A 03FC110A FEDCBA98
	v_cndmask_b32_e32 v7, v5, v6, vcc_lo                       // 00000000669C: 020E0D05
	v_cndmask_b32_e64 v5, v5, v6, s1                           // 0000000066A0: D5010005 00060D05
	v_cmp_eq_f32_e32 vcc_lo, v81, v8                           // 0000000066A8: 7C241151
	v_cmp_lt_f32_e64 s3, v80, v11                              // 0000000066AC: D4110003 02021750
	v_cmp_eq_f32_e64 s2, v75, v9                               // 0000000066B4: D4120002 0202134B
	v_cmp_lt_f32_e64 s4, v74, v10                              // 0000000066BC: D4110004 0202154A
	v_cmp_lt_f32_e64 s1, v81, v8                               // 0000000066C4: D4110001 02021151
	v_permlanex16_b32 v4, v4, s8, 0xfedcba98                   // 0000000066CC: D65C0004 03FC1104 FEDCBA98
	v_permlanex16_b32 v12, v12, s8, 0xfedcba98                 // 0000000066D8: D65C000C 03FC110C FEDCBA98
	v_permlanex16_b32 v5, v5, s8, 0xfedcba98                   // 0000000066E4: D65C0005 03FC1105 FEDCBA98
	v_permlanex16_b32 v7, v7, s8, 0xfedcba98                   // 0000000066F0: D65C0007 03FC1107 FEDCBA98
	v_cmp_lt_f32_e64 s5, v75, v9                               // 0000000066FC: D4110005 0202134B
	s_and_b32 s3, vcc_lo, s3                                   // 000000006704: 8B03036A
	s_and_b32 s7, s2, s4                                       // 000000006708: 8B070402
	s_wait_alu depctr_sa_sdst(0)                               // 00000000670C: BF88FF9E
	s_or_b32 vcc_lo, s1, s3                                    // 000000006710: 8C6A0301
	v_cmp_eq_f32_e64 s2, v76, v4                               // 000000006714: D4120002 0202094C
	v_cmp_eq_f32_e64 s3, v78, v12                              // 00000000671C: D4120003 0202194E
	v_cmp_lt_f32_e64 s4, v77, v5                               // 000000006724: D4110004 02020B4D
	v_cmp_lt_f32_e64 s6, v79, v7                               // 00000000672C: D4110006 02020F4F
	v_cmp_lt_f32_e64 s1, v76, v4                               // 000000006734: D4110001 0202094C
	s_or_b32 s5, s5, s7                                        // 00000000673C: 8C050705
	v_cmp_lt_f32_e64 s7, v78, v12                              // 000000006740: D4110007 0202194E
	s_and_b32 s2, s2, s4                                       // 000000006748: 8B020402
	s_and_b32 s3, s3, s6                                       // 00000000674C: 8B030603
	s_wait_alu depctr_sa_sdst(0)                               // 000000006750: BF88FF9E
	s_or_b32 s1, s1, s2                                        // 000000006754: 8C010201
	v_cndmask_b32_e32 v6, v8, v81, vcc_lo                      // 000000006758: 020CA308
	s_or_b32 s2, s7, s3                                        // 00000000675C: 8C020307
	v_cndmask_b32_e64 v8, v9, v75, s5                          // 000000006760: D5010008 00169709
	s_wait_alu depctr_sa_sdst(0)                               // 000000006768: BF88FF9E
	v_cndmask_b32_e64 v4, v4, v76, s1                          // 00000000676C: D5010004 00069904
	v_cndmask_b32_e64 v9, v12, v78, s2                         // 000000006774: D5010009 000A9D0C
	v_cndmask_b32_e64 v5, v5, v77, s1                          // 00000000677C: D5010005 00069B05
	v_cndmask_b32_e64 v7, v7, v79, s2                          // 000000006784: D5010007 000A9F07
	v_cndmask_b32_e32 v11, v11, v80, vcc_lo                    // 00000000678C: 0216A10B
	v_cmp_eq_f32_e32 vcc_lo, v68, v4                           // 000000006790: 7C240944
	v_cmp_eq_f32_e64 s1, v69, v9                               // 000000006794: D4120001 02021345
	v_cmp_lt_f32_e64 s2, v67, v5                               // 00000000679C: D4110002 02020B43
	v_cmp_lt_f32_e64 s3, v66, v7                               // 0000000067A4: D4110003 02020F42
	v_cmp_lt_f32_e64 s4, v68, v4                               // 0000000067AC: D4110004 02020944
	v_cndmask_b32_e64 v10, v10, v74, s5                        // 0000000067B4: D501000A 0016950A
	v_cmp_lt_f32_e64 s5, v69, v9                               // 0000000067BC: D4110005 02021345
	s_and_b32 s2, vcc_lo, s2                                   // 0000000067C4: 8B02026A
	s_and_b32 s1, s1, s3                                       // 0000000067C8: 8B010301
	s_wait_alu depctr_sa_sdst(0)                               // 0000000067CC: BF88FF9E
	s_or_b32 vcc_lo, s4, s2                                    // 0000000067D0: 8C6A0204
	v_cmp_eq_f32_e64 s2, v73, v8                               // 0000000067D4: D4120002 02021149
	s_or_b32 s1, s5, s1                                        // 0000000067DC: 8C010105
	v_cmp_lt_f32_e64 s4, v70, v10                              // 0000000067E0: D4110004 02021546
	s_wait_alu depctr_sa_sdst(0)                               // 0000000067E8: BF88FF9E
	v_cndmask_b32_e64 v13, v9, v69, s1                         // 0000000067EC: D501000D 00068B09
	v_cndmask_b32_e64 v15, v7, v66, s1                         // 0000000067F4: D501000F 00068507
	v_cndmask_b32_e64 v9, v69, v9, s1                          // 0000000067FC: D5010009 00061345
	v_cndmask_b32_e64 v7, v66, v7, s1                          // 000000006804: D5010007 00060F42
	v_cmp_lt_f32_e64 s1, v73, v8                               // 00000000680C: D4110001 02021149
	s_and_b32 s2, s2, s4                                       // 000000006814: 8B020402
	v_cndmask_b32_e32 v12, v4, v68, vcc_lo                     // 000000006818: 02188904
	v_cndmask_b32_e32 v4, v68, v4, vcc_lo                      // 00000000681C: 02080944
	v_cmp_lt_f32_e64 s3, v72, v11                              // 000000006820: D4110003 02021748
	s_wait_alu depctr_sa_sdst(0)                               // 000000006828: BF88FF9E
	s_or_b32 s1, s1, s2                                        // 00000000682C: 8C010201
	v_cmp_lt_f32_e64 s5, v71, v6                               // 000000006830: D4110005 02020D47
	s_wait_alu depctr_sa_sdst(0)                               // 000000006838: BF88FF9E
	v_cndmask_b32_e64 v17, v8, v73, s1                         // 00000000683C: D5010011 00069308
	v_cndmask_b32_e64 v19, v10, v70, s1                        // 000000006844: D5010013 00068D0A
	v_cndmask_b32_e64 v8, v73, v8, s1                          // 00000000684C: D5010008 00061149
	v_cndmask_b32_e64 v10, v70, v10, s1                        // 000000006854: D501000A 00061546
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 00000000685C: BF870214
	v_cmp_eq_f32_e64 s2, v13, v17                              // 000000006860: D4120002 0202230D
	v_cmp_lt_f32_e64 s4, v15, v19                              // 000000006868: D4110004 0202270F
	v_cmp_lt_f32_e64 s1, v13, v17                              // 000000006870: D4110001 0202230D
	s_and_b32 s2, s2, s4                                       // 000000006878: 8B020402
	v_cmp_lt_f32_e64 s4, v7, v10                               // 00000000687C: D4110004 02021507
	s_wait_alu depctr_sa_sdst(0)                               // 000000006884: BF88FF9E
	s_or_b32 s1, s1, s2                                        // 000000006888: 8C010201
	v_cmp_eq_f32_e64 s2, v9, v8                                // 00000000688C: D4120002 02021109
	s_wait_alu depctr_sa_sdst(0)                               // 000000006894: BF88FF9E
	v_cndmask_b32_e64 v21, v17, v13, s1                        // 000000006898: D5010015 00061B11
	v_cndmask_b32_e64 v23, v19, v15, s1                        // 0000000068A0: D5010017 00061F13
	v_cndmask_b32_e64 v13, v13, v17, s1                        // 0000000068A8: D501000D 0006230D
	v_cndmask_b32_e64 v15, v15, v19, s1                        // 0000000068B0: D501000F 0006270F
	v_cmp_lt_f32_e64 s1, v9, v8                                // 0000000068B8: D4110001 02021109
	s_and_b32 s2, s2, s4                                       // 0000000068C0: 8B020402
	v_mov_b32_e32 v24, v21                                     // 0000000068C4: 7E300315
	v_mov_b32_e32 v26, v23                                     // 0000000068C8: 7E340317
	s_wait_alu depctr_sa_sdst(0)                               // 0000000068CC: BF88FF9E
	s_or_b32 s1, s1, s2                                        // 0000000068D0: 8C010201
	s_wait_alu depctr_sa_sdst(0)                               // 0000000068D4: BF88FF9E
	v_cndmask_b32_e64 v17, v8, v9, s1                          // 0000000068D8: D5010011 00061308
	v_cndmask_b32_e64 v8, v9, v8, s1                           // 0000000068E0: D5010008 00061109
	v_dual_cndmask_b32 v14, v5, v67 :: v_dual_cndmask_b32 v5, v67, v5// 0000000068E8: CA528705 0E040B43
	v_cmp_eq_f32_e32 vcc_lo, v71, v6                           // 0000000068F0: 7C240D47
	v_cndmask_b32_e64 v19, v10, v7, s1                         // 0000000068F4: D5010013 00060F0A
	v_cndmask_b32_e64 v7, v7, v10, s1                          // 0000000068FC: D5010007 00061507
	v_permlanex16_b32 v24, v24, s8, 0xfedcba98                 // 000000006904: D65C0018 03FC1118 FEDCBA98
	v_permlanex16_b32 v26, v26, s8, 0xfedcba98                 // 000000006910: D65C001A 03FC111A FEDCBA98
	s_and_b32 s3, vcc_lo, s3                                   // 00000000691C: 8B03036A
	v_mov_b32_e32 v9, v13                                      // 000000006920: 7E12030D
	s_wait_alu depctr_sa_sdst(0)                               // 000000006924: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 000000006928: 8C6A0305
	v_cmp_eq_f32_e64 s2, v21, v24                              // 00000000692C: D4120002 02023115
	s_wait_alu depctr_sa_sdst(0)                               // 000000006934: BF88FF9E
	v_cndmask_b32_e32 v16, v6, v71, vcc_lo                     // 000000006938: 02208F06
	v_cndmask_b32_e32 v18, v11, v72, vcc_lo                    // 00000000693C: 0224910B
	v_dual_cndmask_b32 v6, v71, v6 :: v_dual_cndmask_b32 v11, v72, v11// 000000006940: CA520D47 060A1748
	v_cmp_gt_f32_e64 s4, v23, v26                              // 000000006948: D4140004 02023517
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000006950: BF870214
	v_cmp_eq_f32_e32 vcc_lo, v12, v16                          // 000000006954: 7C24210C
	v_cmp_lt_f32_e64 s3, v14, v18                              // 000000006958: D4110003 0202250E
	v_cmp_lt_f32_e64 s5, v12, v16                              // 000000006960: D4110005 0202210C
	v_cmp_gt_f32_e64 s6, v21, v24                              // 000000006968: D4140006 02023115
	s_and_b32 s2, s2, s4                                       // 000000006970: 8B020402
	v_permlanex16_b32 v9, v9, s8, 0xfedcba98                   // 000000006974: D65C0009 03FC1109 FEDCBA98
	s_and_b32 s3, vcc_lo, s3                                   // 000000006980: 8B03036A
	s_wait_alu depctr_sa_sdst(0)                               // 000000006984: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 000000006988: 8C6A0305
	v_cmp_lt_f32_e64 s3, v5, v11                               // 00000000698C: D4110003 02021705
	s_wait_alu depctr_sa_sdst(0)                               // 000000006994: BF88FF9E
	v_cndmask_b32_e32 v20, v16, v12, vcc_lo                    // 000000006998: 02281910
	v_cndmask_b32_e32 v22, v18, v14, vcc_lo                    // 00000000699C: 022C1D12
	v_cndmask_b32_e32 v12, v12, v16, vcc_lo                    // 0000000069A0: 0218210C
	v_cndmask_b32_e32 v14, v14, v18, vcc_lo                    // 0000000069A4: 021C250E
	v_cmp_eq_f32_e32 vcc_lo, v4, v6                            // 0000000069A8: 7C240D04
	v_cmp_lt_f32_e64 s5, v4, v6                                // 0000000069AC: D4110005 02020D04
	v_mov_b32_e32 v25, v22                                     // 0000000069B4: 7E320316
	s_or_b32 s2, s6, s2                                        // 0000000069B8: 8C020206
	v_cmp_gt_f32_e64 s6, v13, v9                               // 0000000069BC: D4140006 0202130D
	s_and_b32 s3, vcc_lo, s3                                   // 0000000069C4: 8B03036A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000069C8: BF88FF9E
	s_or_b32 vcc_lo, s5, s3                                    // 0000000069CC: 8C6A0305
	v_permlanex16_b32 v25, v25, s8, 0xfedcba98                 // 0000000069D0: D65C0019 03FC1119 FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 0000000069DC: BF88FF9E
	v_cndmask_b32_e32 v16, v6, v4, vcc_lo                      // 0000000069E0: 02200906
	v_cndmask_b32_e32 v4, v4, v6, vcc_lo                       // 0000000069E4: 02080D04
	v_mov_b32_e32 v6, v20                                      // 0000000069E8: 7E0C0314
	v_dual_cndmask_b32 v18, v11, v5 :: v_dual_cndmask_b32 v5, v5, v11// 0000000069EC: CA520B0B 12041705
	v_cmp_gt_f32_e64 s3, v22, v25                              // 0000000069F4: D4140003 02023316
	v_mov_b32_e32 v11, v14                                     // 0000000069FC: 7E16030E
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000006A00: BF870114
	v_permlanex16_b32 v6, v6, s8, 0xfedcba98                   // 000000006A04: D65C0006 03FC1106 FEDCBA98
	v_permlanex16_b32 v11, v11, s8, 0xfedcba98                 // 000000006A10: D65C000B 03FC110B FEDCBA98
	s_delay_alu instid0(VALU_DEP_2)                            // 000000006A1C: BF870002
	v_cmp_eq_f32_e32 vcc_lo, v20, v6                           // 000000006A20: 7C240D14
	v_cmp_gt_f32_e64 s5, v20, v6                               // 000000006A24: D4140005 02020D14
	s_and_b32 s1, vcc_lo, s3                                   // 000000006A2C: 8B01036A
	v_cmp_gt_f32_e64 s3, v14, v11                              // 000000006A30: D4140003 0202170E
	s_wait_alu depctr_sa_sdst(0)                               // 000000006A38: BF88FF9E
	s_or_b32 s1, s5, s1                                        // 000000006A3C: 8C010105
	s_wait_alu depctr_sa_sdst(0)                               // 000000006A40: BF88FF9E
	s_xor_b32 vcc_lo, s1, s0                                   // 000000006A44: 8D6A0001
	s_xor_b32 s1, s2, s0                                       // 000000006A48: 8D010002
	s_wait_alu depctr_sa_sdst(0)                               // 000000006A4C: BF88FF9E
	v_cndmask_b32_e32 v10, v20, v6, vcc_lo                     // 000000006A50: 02140D14
	v_mov_b32_e32 v6, v12                                      // 000000006A54: 7E0C030C
	v_cndmask_b32_e32 v22, v22, v25, vcc_lo                    // 000000006A58: 022C3316
	v_mov_b32_e32 v20, v15                                     // 000000006A5C: 7E28030F
	v_cndmask_b32_e64 v21, v21, v24, s1                        // 000000006A60: D5010015 00063115
	v_cndmask_b32_e64 v23, v23, v26, s1                        // 000000006A68: D5010017 00063517
	v_permlanex16_b32 v6, v6, s8, 0xfedcba98                   // 000000006A70: D65C0006 03FC1106 FEDCBA98
	v_mov_b32_e32 v25, v18                                     // 000000006A7C: 7E320312
	v_permlanex16_b32 v20, v20, s8, 0xfedcba98                 // 000000006A80: D65C0014 03FC1114 FEDCBA98
	v_cmp_eq_f32_e64 s2, v13, v9                               // 000000006A8C: D4120002 0202130D
	v_mov_b32_e32 v24, v17                                     // 000000006A94: 7E300311
	v_cmp_eq_f32_e32 vcc_lo, v12, v6                           // 000000006A98: 7C240D0C
	v_cmp_gt_f32_e64 s5, v12, v6                               // 000000006A9C: D4140005 02020D0C
	v_cmp_gt_f32_e64 s4, v15, v20                              // 000000006AA4: D4140004 0202290F
	v_permlanex16_b32 v25, v25, s8, 0xfedcba98                 // 000000006AAC: D65C0019 03FC1119 FEDCBA98
	v_mov_b32_e32 v26, v19                                     // 000000006AB8: 7E340313
	s_and_b32 s1, vcc_lo, s3                                   // 000000006ABC: 8B01036A
	v_permlanex16_b32 v24, v24, s8, 0xfedcba98                 // 000000006AC0: D65C0018 03FC1118 FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 000000006ACC: BF88FF9E
	s_or_b32 s1, s5, s1                                        // 000000006AD0: 8C010105
	s_and_b32 s2, s2, s4                                       // 000000006AD4: 8B020402
	s_wait_alu depctr_sa_sdst(0)                               // 000000006AD8: BF88FF9E
	s_xor_b32 vcc_lo, s1, s0                                   // 000000006ADC: 8D6A0001
	v_cmp_gt_f32_e64 s3, v18, v25                              // 000000006AE0: D4140003 02023312
	s_wait_alu depctr_sa_sdst(0)                               // 000000006AE8: BF88FF9E
	v_cndmask_b32_e32 v6, v12, v6, vcc_lo                      // 000000006AEC: 020C0D0C
	v_dual_mov_b32 v12, v16 :: v_dual_cndmask_b32 v11, v14, v11// 000000006AF0: CA120110 0C0A170E
	s_or_b32 s2, s6, s2                                        // 000000006AF8: 8C020206
	v_permlanex16_b32 v26, v26, s8, 0xfedcba98                 // 000000006AFC: D65C001A 03FC111A FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 000000006B08: BF88FF9E
	s_xor_b32 s1, s2, s0                                       // 000000006B0C: 8D010002
	v_permlanex16_b32 v12, v12, s8, 0xfedcba98                 // 000000006B10: D65C000C 03FC110C FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 000000006B1C: BF88FF9E
	v_cndmask_b32_e64 v9, v13, v9, s1                          // 000000006B20: D5010009 0006130D
	v_cndmask_b32_e64 v13, v15, v20, s1                        // 000000006B28: D501000D 0006290F
	v_mov_b32_e32 v14, v4                                      // 000000006B30: 7E1C0304
	v_cmp_eq_f32_e64 s2, v17, v24                              // 000000006B34: D4120002 02023111
	v_cmp_eq_f32_e32 vcc_lo, v16, v12                          // 000000006B3C: 7C241910
	v_cmp_gt_f32_e64 s5, v16, v12                              // 000000006B40: D4140005 02021910
	v_cmp_gt_f32_e64 s4, v19, v26                              // 000000006B48: D4140004 02023513
	v_dual_mov_b32 v15, v8 :: v_dual_mov_b32 v20, v7           // 000000006B50: CA100108 0F140107
	s_and_b32 s1, vcc_lo, s3                                   // 000000006B58: 8B01036A
	v_permlanex16_b32 v14, v14, s8, 0xfedcba98                 // 000000006B5C: D65C000E 03FC110E FEDCBA98
	s_wait_alu depctr_sa_sdst(0)                               // 000000006B68: BF88FF9E
	s_or_b32 s1, s5, s1                                        // 000000006B6C: 8C010105
	v_cmp_gt_f32_e64 s6, v17, v24                              // 000000006B70: D4140006 02023111
	s_wait_alu depctr_sa_sdst(0)                               // 000000006B78: BF88FF9E
	s_xor_b32 vcc_lo, s1, s0                                   // 000000006B7C: 8D6A0001
	s_and_b32 s2, s2, s4                                       // 000000006B80: 8B020402
	s_wait_alu depctr_sa_sdst(0)                               // 000000006B84: BF88FF9E
	v_cndmask_b32_e32 v12, v16, v12, vcc_lo                    // 000000006B88: 02181910
	v_mov_b32_e32 v16, v5                                      // 000000006B8C: 7E200305
	v_permlanex16_b32 v15, v15, s8, 0xfedcba98                 // 000000006B90: D65C000F 03FC110F FEDCBA98
	v_permlanex16_b32 v20, v20, s8, 0xfedcba98                 // 000000006B9C: D65C0014 03FC1114 FEDCBA98
	v_cndmask_b32_e32 v18, v18, v25, vcc_lo                    // 000000006BA8: 02243312
	v_cmp_eq_f32_e32 vcc_lo, v4, v14                           // 000000006BAC: 7C241D04
	v_permlanex16_b32 v16, v16, s8, 0xfedcba98                 // 000000006BB0: D65C0010 03FC1110 FEDCBA98
	s_or_b32 s2, s6, s2                                        // 000000006BBC: 8C020206
	v_cmp_gt_f32_e64 s5, v4, v14                               // 000000006BC0: D4140005 02021D04
	s_wait_alu depctr_sa_sdst(0)                               // 000000006BC8: BF88FF9E
	s_xor_b32 s1, s2, s0                                       // 000000006BCC: 8D010002
	v_cmp_eq_f32_e64 s2, v8, v15                               // 000000006BD0: D4120002 02021F08
	v_cmp_gt_f32_e64 s3, v5, v16                               // 000000006BD8: D4140003 02022105
	v_cmp_gt_f32_e64 s4, v7, v20                               // 000000006BE0: D4140004 02022907
	s_wait_alu depctr_sa_sdst(0)                               // 000000006BE8: BF88FF9E
	v_cndmask_b32_e64 v17, v17, v24, s1                        // 000000006BEC: D5010011 00063111
	v_cndmask_b32_e64 v19, v19, v26, s1                        // 000000006BF4: D5010013 00063513
	v_cmp_gt_f32_e64 s6, v8, v15                               // 000000006BFC: D4140006 02021F08
	s_and_b32 s1, vcc_lo, s3                                   // 000000006C04: 8B01036A
	s_and_b32 s2, s2, s4                                       // 000000006C08: 8B020402
	s_wait_alu depctr_sa_sdst(0)                               // 000000006C0C: BF88FF9E
	s_or_b32 s1, s5, s1                                        // 000000006C10: 8C010105
	v_cmp_lt_f32_e64 s3, v21, v10                              // 000000006C14: D4110003 02021515
	s_wait_alu depctr_sa_sdst(0)                               // 000000006C1C: BF88FF9E
	s_xor_b32 vcc_lo, s1, s0                                   // 000000006C20: 8D6A0001
	s_or_b32 s2, s6, s2                                        // 000000006C24: 8C020206
	s_wait_alu depctr_sa_sdst(0)                               // 000000006C28: BF88FF9E
	v_cndmask_b32_e32 v14, v4, v14, vcc_lo                     // 000000006C2C: 021C1D04
	v_cndmask_b32_e32 v16, v5, v16, vcc_lo                     // 000000006C30: 02202105
	v_cmp_eq_f32_e32 vcc_lo, v21, v10                          // 000000006C34: 7C241515
	v_cmp_lt_f32_e64 s1, v23, v22                              // 000000006C38: D4110001 02022D17
	s_xor_b32 s0, s2, s0                                       // 000000006C40: 8D000002
	v_cmp_lt_f32_e64 s2, v22, v23                              // 000000006C44: D4110002 02022F16
	s_wait_alu depctr_sa_sdst(0)                               // 000000006C4C: BF88FF9E
	v_cndmask_b32_e64 v15, v8, v15, s0                         // 000000006C50: D501000F 00021F08
	v_cndmask_b32_e64 v20, v7, v20, s0                         // 000000006C58: D5010014 00022907
	s_and_b32 s1, vcc_lo, s1                                   // 000000006C60: 8B01016A
	v_cmp_lt_f32_e64 s0, v10, v21                              // 000000006C64: D4110000 02022B0A
	s_and_b32 s5, vcc_lo, s2                                   // 000000006C6C: 8B05026A
	s_wait_alu depctr_sa_sdst(0)                               // 000000006C70: BF88FF9E
	s_or_b32 vcc_lo, s3, s1                                    // 000000006C74: 8C6A0103
	v_cmp_eq_f32_e64 s2, v9, v6                                // 000000006C78: D4120002 02020D09
	v_cmp_lt_f32_e64 s3, v13, v11                              // 000000006C80: D4110003 0202170D
	v_cmp_lt_f32_e64 s1, v9, v6                                // 000000006C88: D4110001 02020D09
	v_cmp_lt_f32_e64 s4, v11, v13                              // 000000006C90: D4110004 02021B0B
	s_or_b32 s0, s0, s5                                        // 000000006C98: 8C000500
	v_cmp_lt_f32_e64 s5, v6, v9                                // 000000006C9C: D4110005 02021306
	s_and_b32 s3, s2, s3                                       // 000000006CA4: 8B030302
	s_wait_alu depctr_sa_sdst(0)                               // 000000006CA8: BF88FF9E
	v_cndmask_b32_e32 v24, v22, v23, vcc_lo                    // 000000006CAC: 02302F16
	s_or_b32 s1, s1, s3                                        // 000000006CB0: 8C010301
	v_cndmask_b32_e64 v22, v22, v23, s0                        // 000000006CB4: D5010016 00022F16
	s_and_b32 s2, s2, s4                                       // 000000006CBC: 8B020402
	s_wait_alu depctr_sa_sdst(0)                               // 000000006CC0: BF88FF9E
	v_cndmask_b32_e64 v4, v6, v9, s1                           // 000000006CC4: D5010004 00061306
	v_cndmask_b32_e64 v23, v11, v13, s1                        // 000000006CCC: D5010017 00061B0B
	v_cmp_eq_f32_e64 s1, v17, v12                              // 000000006CD4: D4120001 02021911
	v_cmp_lt_f32_e64 s3, v19, v18                              // 000000006CDC: D4110003 02022513
	s_or_b32 s2, s5, s2                                        // 000000006CE4: 8C020205
	v_cmp_lt_f32_e64 s4, v18, v19                              // 000000006CE8: D4110004 02022712
	v_cmp_lt_f32_e64 s5, v17, v12                              // 000000006CF0: D4110005 02021911
	s_wait_alu depctr_sa_sdst(0)                               // 000000006CF8: BF88FF9E
	v_cndmask_b32_e64 v5, v6, v9, s2                           // 000000006CFC: D5010005 000A1306
	v_cndmask_b32_e64 v25, v11, v13, s2                        // 000000006D04: D5010019 000A1B0B
	v_cmp_lt_f32_e64 s2, v12, v17                              // 000000006D0C: D4110002 0202230C
	s_and_b32 s3, s1, s3                                       // 000000006D14: 8B030301
	s_and_b32 s4, s1, s4                                       // 000000006D18: 8B040401
	s_wait_alu depctr_sa_sdst(0)                               // 000000006D1C: BF88FF9E
	s_or_b32 s1, s5, s3                                        // 000000006D20: 8C010305
	v_cmp_lt_f32_e64 s3, v20, v16                              // 000000006D24: D4110003 02022114
	s_wait_alu depctr_sa_sdst(0)                               // 000000006D2C: BF88FF9E
	v_cndmask_b32_e64 v6, v12, v17, s1                         // 000000006D30: D5010006 0006230C
	s_or_b32 s2, s2, s4                                        // 000000006D38: 8C020402
	v_cndmask_b32_e64 v26, v18, v19, s1                        // 000000006D3C: D501001A 00062712
	v_cmp_eq_f32_e64 s1, v15, v14                              // 000000006D44: D4120001 02021D0F
	v_cmp_lt_f32_e64 s4, v16, v20                              // 000000006D4C: D4110004 02022910
	s_wait_alu depctr_sa_sdst(0)                               // 000000006D54: BF88FF9E
	v_cndmask_b32_e64 v7, v12, v17, s2                         // 000000006D58: D5010007 000A230C
	v_cmp_lt_f32_e64 s5, v15, v14                              // 000000006D60: D4110005 02021D0F
	v_cndmask_b32_e64 v17, v18, v19, s2                        // 000000006D68: D5010011 000A2712
	v_cmp_lt_f32_e64 s2, v14, v15                              // 000000006D70: D4110002 02021F0E
	s_and_b32 s3, s1, s3                                       // 000000006D78: 8B030301
	s_and_b32 s4, s1, s4                                       // 000000006D7C: 8B040401
	s_wait_alu depctr_sa_sdst(0)                               // 000000006D80: BF88FF9E
	s_or_b32 s1, s5, s3                                        // 000000006D84: 8C010305
	v_cndmask_b32_e64 v11, v10, v21, s0                        // 000000006D88: D501000B 00022B0A
	s_or_b32 s2, s2, s4                                        // 000000006D90: 8C020402
	s_wait_alu depctr_sa_sdst(0)                               // 000000006D94: BF88FF9E
	v_cndmask_b32_e64 v18, v16, v20, s1                        // 000000006D98: D5010012 00062910
	v_cndmask_b32_e64 v19, v16, v20, s2                        // 000000006DA0: D5010013 000A2910
	v_cndmask_b32_e32 v10, v10, v21, vcc_lo                    // 000000006DA8: 02142B0A
	v_cndmask_b32_e64 v8, v14, v15, s1                         // 000000006DAC: D5010008 00061F0E
	v_cndmask_b32_e64 v9, v14, v15, s2                         // 000000006DB4: D5010009 000A1F0E
	v_cvt_i32_f32_e32 v13, v22                                 // 000000006DBC: 7E1A1116
	v_cvt_i32_f32_e32 v12, v24                                 // 000000006DC0: 7E181118
	v_cvt_i32_f32_e32 v15, v25                                 // 000000006DC4: 7E1E1119
	v_cvt_i32_f32_e32 v14, v23                                 // 000000006DC8: 7E1C1117
	v_cvt_i32_f32_e32 v17, v17                                 // 000000006DCC: 7E221111
	v_cvt_i32_f32_e32 v16, v26                                 // 000000006DD0: 7E20111A
	v_cvt_i32_f32_e32 v19, v19                                 // 000000006DD4: 7E261113
	v_cvt_i32_f32_e32 v18, v18                                 // 000000006DD8: 7E241112
	s_clause 0x3                                               // 000000006DDC: BF850003
	global_store_b64 v[2:3], v[10:11], off                     // 000000006DE0: EE06C07C 05000000 00000002
	global_store_b64 v[2:3], v[4:5], off offset:16             // 000000006DEC: EE06C07C 02000000 00001002
	global_store_b64 v[2:3], v[6:7], off offset:32             // 000000006DF8: EE06C07C 03000000 00002002
	global_store_b64 v[2:3], v[8:9], off offset:48             // 000000006E04: EE06C07C 04000000 00003002
	s_clause 0x3                                               // 000000006E10: BF850003
	global_store_b64 v[0:1], v[12:13], off                     // 000000006E14: EE06C07C 06000000 00000000
	global_store_b64 v[0:1], v[14:15], off offset:16           // 000000006E20: EE06C07C 07000000 00001000
	global_store_b64 v[0:1], v[16:17], off offset:32           // 000000006E2C: EE06C07C 08000000 00002000
	global_store_b64 v[0:1], v[18:19], off offset:48           // 000000006E38: EE06C07C 09000000 00003000
	s_nop 0                                                    // 000000006E44: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000006E48: BFB60003
	s_endpgm                                                   // 000000006E4C: BFB00000
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
