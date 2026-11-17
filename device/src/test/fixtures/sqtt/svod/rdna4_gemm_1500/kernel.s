
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000000000 <gemm>:
	s_mul_i32 s2, ttmp7, 12                                    // 000000000000: 96028C73
	v_bfe_u32 v28, v0, 6, 1                                    // 000000000004: D610001C 02050D00
	s_add_co_i32 s2, s2, ttmp9                                 // 00000000000C: 81027502
	v_lshrrev_b32_e32 v23, 7, v0                               // 000000000010: 322E0087
	s_mul_hi_i32 s3, s2, 0x66666667                            // 000000000014: 9703FF02 66666667
	v_bfe_u32 v21, v0, 5, 1                                    // 00000000001C: D6100015 02050B00
	s_lshr_b32 s4, s3, 31                                      // 000000000024: 85049F03
	s_ashr_i32 s3, s3, 5                                       // 000000000028: 86038503
	v_lshrrev_b32_e32 v2, 2, v0                                // 00000000002C: 32040082
	s_add_co_i32 s3, s3, s4                                    // 000000000030: 81030403
	v_bfe_u32 v82, v0, 4, 1                                    // 000000000034: D6100052 02050900
	s_lshl_b32 s8, s3, 3                                       // 00000000003C: 84088303
	v_lshlrev_b32_e32 v24, 1, v28                              // 000000000040: 30303881
	s_max_i32 s4, s8, 4                                        // 000000000044: 8A048408
	v_dual_mov_b32 v14, 0 :: v_dual_lshlrev_b32 v25, 2, v23    // 000000000048: CA220080 0E182E82
	s_sub_co_i32 s6, 12, s4                                    // 000000000050: 8186048C
	v_dual_mov_b32 v13, 0 :: v_dual_and_b32 v20, 2, v2         // 000000000054: CA240080 0D140482
	s_abs_i32 s4, s6                                           // 00000000005C: BE841506
	v_dual_mov_b32 v15, 0 :: v_dual_lshlrev_b32 v22, 2, v82    // 000000000060: CA220080 0F16A482
	s_cvt_f32_u32 s5, s4                                       // 000000000068: BE856504
	v_or3_b32 v2, v24, v25, v21                                // 00000000006C: D6580002 04563318
	s_sub_co_i32 s7, 0, s4                                     // 000000000074: 81870480
	s_mulk_i32 s3, 0x50                                        // 000000000078: B8030050
	v_rcp_iflag_f32_e32 v1, s5                                 // 00000000007C: 7E025605
	s_sub_co_i32 s2, s2, s3                                    // 000000000080: 81820302
	v_bfe_u32 v19, v0, 2, 1                                    // 000000000084: D6100013 02050500
	v_or_b32_e32 v29, v22, v20                                 // 00000000008C: 383A2916
	v_lshlrev_b32_e32 v2, 4, v2                                // 000000000090: 30040484
	s_abs_i32 s3, s2                                           // 000000000094: BE831502
	v_mov_b32_e32 v16, 0                                       // 000000000098: 7E200280
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)// 00000000009C: BF870292
	v_or3_b32 v26, v2, v19, v29                                // 0000000000A0: D658001A 04762702
	v_readfirstlane_b32 s5, v1                                 // 0000000000A8: 7E0A0501
	v_lshlrev_b32_e32 v1, 3, v0                                // 0000000000AC: 30020083
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 0000000000B0: BF870123
	v_mul_lo_u32 v2, 0x1400, v26                               // 0000000000B4: D72C0002 000234FF 00001400
	s_mul_f32 s5, s5, 0x4f7ffffe                               // 0000000000C0: A205FF05 4F7FFFFE
	v_and_b32_e32 v27, 24, v1                                  // 0000000000C8: 36360298
	s_wait_alu depctr_sa_sdst(0)                               // 0000000000CC: BF88FF9E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000000D0: BF870099
	s_cvt_u32_f32 s5, s5                                       // 0000000000D4: BE856705
	v_or_b32_e32 v73, v2, v27                                  // 0000000000D8: 38923702
	s_wait_alu depctr_sa_sdst(0)                               // 0000000000DC: BF88FF9E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000000E0: BF870499
	s_mul_i32 s7, s7, s5                                       // 0000000000E4: 96070507
	s_mul_hi_u32 s7, s5, s7                                    // 0000000000E8: 96870705
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000000EC: BF870009
	s_add_co_i32 s5, s5, s7                                    // 0000000000F0: 81050705
	s_xor_b32 s7, s2, s6                                       // 0000000000F4: 8D070602
	s_wait_alu depctr_sa_sdst(0)                               // 0000000000F8: BF88FF9E
	s_mul_hi_u32 s5, s3, s5                                    // 0000000000FC: 96850503
	s_ashr_i32 s7, s7, 31                                      // 000000000100: 86079F07
	s_wait_alu depctr_sa_sdst(0)                               // 000000000104: BF88FF9E
	s_mul_i32 s9, s5, s4                                       // 000000000108: 96090405
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000010C: BF870009
	s_sub_co_i32 s3, s3, s9                                    // 000000000110: 81830903
	s_add_co_i32 s9, s5, 1                                     // 000000000114: 81098105
	s_sub_co_i32 s10, s3, s4                                   // 000000000118: 818A0403
	s_cmp_ge_u32 s3, s4                                        // 00000000011C: BF090403
	s_cselect_b32 s5, s9, s5                                   // 000000000120: 98050509
	s_cselect_b32 s3, s10, s3                                  // 000000000124: 9803030A
	s_wait_alu depctr_sa_sdst(0)                               // 000000000128: BF88FF9E
	s_add_co_i32 s9, s5, 1                                     // 00000000012C: 81098105
	s_cmp_ge_u32 s3, s4                                        // 000000000130: BF090403
	s_cselect_b32 s3, s9, s5                                   // 000000000134: 98030509
	s_load_b64 s[4:5], s[0:1], 0x10                            // 000000000138: F4002100 F8000010
	s_xor_b32 s3, s3, s7                                       // 000000000140: 8D030703
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000000144: BF870499
	s_sub_co_i32 s3, s3, s7                                    // 000000000148: 81830703
	v_mad_co_u64_u32 v[1:2], null, 0xa0000, s3, v[73:74]       // 00000000014C: D6FE7C01 052406FF 000A0000
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000000158: BF870091
	v_ashrrev_i32_e32 v2, 31, v1                               // 00000000015C: 3404029F
	v_lshlrev_b64_e32 v[1:2], 1, v[1:2]                        // 000000000160: 3E020281
	s_wait_kmcnt 0x0                                           // 000000000164: BFC70000
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000000168: BF870091
	v_add_co_u32 v74, vcc_lo, s4, v1                           // 00000000016C: D7006A4A 00020204
	v_add_co_ci_u32_e64 v75, null, s5, v2, vcc_lo              // 000000000174: D5207C4B 01AA0405
	s_clause 0x1                                               // 00000000017C: BF850001
	global_load_b128 v[5:8], v[74:75], off                     // 000000000180: EE05C07C 00000005 0000004A
	global_load_b128 v[1:4], v[74:75], off offset:81920        // 00000000018C: EE05C07C 00000001 0140004A
	s_mul_i32 s4, s3, s6                                       // 000000000198: 96040603
	s_wait_alu depctr_sa_sdst(0)                               // 00000000019C: BF88FF9E
	s_sub_co_i32 s11, s2, s4                                   // 0000000001A0: 818B0402
	s_load_b128 s[4:7], s[0:1], 0x0                            // 0000000001A4: F4004100 F8000000
	s_add_co_i32 s11, s11, s8                                  // 0000000001AC: 810B080B
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)// 0000000001B0: BF8704D9
	s_mul_i32 s2, s11, 0xa0000                                 // 0000000001B4: 9602FF0B 000A0000
	s_lshl_b32 s0, s11, 7                                      // 0000000001BC: 8400870B
	s_wait_alu depctr_sa_sdst(0)                               // 0000000001C0: BF88FF9E
	v_add_nc_u32_e32 v9, s2, v73                               // 0000000001C4: 4A129202
	s_sub_co_i32 s10, 0x5dc, s0                                // 0000000001C8: 818A00FF 000005DC
	v_cmp_gt_i32_e32 vcc_lo, s10, v26                          // 0000000001D0: 7C88340A
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000001D4: BF870092
	v_ashrrev_i32_e32 v10, 31, v9                              // 0000000001D8: 3414129F
	v_lshlrev_b64_e32 v[9:10], 1, v[9:10]                      // 0000000001DC: 3E121281
	s_wait_kmcnt 0x0                                           // 0000000001E0: BFC70000
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000001E4: BF870091
	v_add_co_u32 v17, s0, s6, v9                               // 0000000001E8: D7000011 00021206
	v_add_co_ci_u32_e64 v18, null, s7, v10, s0                 // 0000000001F0: D5207C12 00021407
	v_mov_b32_e32 v9, 0                                        // 0000000001F8: 7E120280
	s_and_saveexec_b32 s0, vcc_lo                              // 0000000001FC: BE80206A
	s_cbranch_execz 3                                          // 000000000200: BFA50003 <gemm+0x210>
	global_load_b128 v[13:16], v[17:18], off                   // 000000000204: EE05C07C 0000000D 00000011
	s_wait_alu depctr_sa_sdst(0)                               // 000000000210: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s0                              // 000000000214: 8C7E007E
	v_or_b32_e32 v30, 8, v26                                   // 000000000218: 383C3488
	v_dual_mov_b32 v10, 0 :: v_dual_mov_b32 v11, 0             // 00000000021C: CA100080 0A0A0080
	v_mov_b32_e32 v12, 0                                       // 000000000224: 7E180280
	s_delay_alu instid0(VALU_DEP_3)                            // 000000000228: BF870003
	v_cmp_gt_i32_e64 s0, s10, v30                              // 00000000022C: D4440000 00023C0A
	s_and_saveexec_b32 s1, s0                                  // 000000000234: BE812000
	s_cbranch_execz 3                                          // 000000000238: BFA50003 <gemm+0x248>
	global_load_b128 v[9:12], v[17:18], off offset:81920       // 00000000023C: EE05C07C 00000009 01400011
	s_wait_alu depctr_sa_sdst(0)                               // 000000000248: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s1                              // 00000000024C: 8C7E017E
	v_dual_mov_b32 v56, 0 :: v_dual_lshlrev_b32 v17, 1, v27    // 000000000250: CA220080 38103681
	v_lshlrev_b32_e32 v18, 3, v29                              // 000000000258: 30243A83
	v_lshlrev_b32_e32 v28, 5, v28                              // 00000000025C: 30383885
	v_and_b32_e32 v29, 15, v0                                  // 000000000260: 363A008F
	v_bfe_u32 v0, v0, 1, 2                                     // 000000000264: D6100000 02090300
	v_add3_u32 v24, v25, v24, v21                              // 00000000026C: D6550018 04563119
	v_mov_b32_e32 v57, v56                                     // 000000000274: 7E720338
	v_xor_b32_e32 v17, v18, v17                                // 000000000278: 3A222312
	v_or_b32_e32 v18, 2, v82                                   // 00000000027C: 3824A482
	v_mov_b32_e32 v59, v56                                     // 000000000280: 7E760338
	v_lshl_or_b32 v91, v23, 6, v28                             // 000000000284: D656005B 04710D17
	v_xor_b32_e32 v23, v82, v0                                 // 00000000028C: 3A2E0152
	v_lshl_or_b32 v85, v26, 6, v17                             // 000000000290: D6560055 04450D1A
	v_xor_b32_e32 v0, v18, v0                                  // 000000000298: 3A000112
	v_lshl_or_b32 v18, v24, 4, v22                             // 00000000029C: D6560012 04590918
	v_lshl_or_b32 v83, v30, 6, v17                             // 0000000002A4: D6560053 04450D1E
	s_wait_loadcnt 0x0                                         // 0000000002AC: BFC00000
	ds_store_b128 v85, v[13:16]                                // 0000000002B0: DB7C0000 00000D55
	ds_store_b128 v85, v[5:8] offset:16384                     // 0000000002B8: DB7C4000 00000555
	ds_store_b128 v83, v[9:12]                                 // 0000000002C0: DB7C0000 00000953
	ds_store_b128 v83, v[1:4] offset:16384                     // 0000000002C8: DB7C4000 00000153
	v_add3_u32 v1, v18, v20, v19                               // 0000000002D0: D6550001 044E2912
	s_wait_dscnt 0x0                                           // 0000000002D8: BFC60000
	s_barrier_signal -1                                        // 0000000002DC: BE804EC1
	v_mov_b32_e32 v58, v56                                     // 0000000002E0: 7E740338
	v_lshl_or_b32 v92, v21, 6, v29                             // 0000000002E4: D656005C 04750D15
	v_or_b32_e32 v17, v91, v29                                 // 0000000002EC: 38223B5B
	v_dual_mov_b32 v49, v56 :: v_dual_lshlrev_b32 v4, 4, v0    // 0000000002F0: CA220138 31040084
	v_mad_co_u64_u32 v[0:1], null, 0x1400, v1, s[2:3]          // 0000000002F8: D6FE7C00 000A02FF 00001400
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_4)// 000000000304: BF870253
	v_dual_mov_b32 v63, v56 :: v_dual_lshlrev_b32 v2, 6, v17   // 000000000308: CA220138 3F022286
	v_dual_mov_b32 v62, v56 :: v_dual_lshlrev_b32 v3, 4, v23   // 000000000310: CA220138 3E022E84
	v_lshl_or_b32 v1, v92, 6, 0x4000                           // 000000000318: D6560001 03FD0D5C 00004000
	v_dual_mov_b32 v60, v56 :: v_dual_add_nc_u32 v89, 0x4000, v85// 000000000324: CA200138 3C58AAFF 00004000
	v_dual_mov_b32 v61, v56 :: v_dual_add_nc_u32 v88, 0x4000, v83// 000000000330: CA200138 3D58A6FF 00004000
	v_or_b32_e32 v86, v2, v3                                   // 00000000033C: 38AC0702
	v_or_b32_e32 v84, v2, v4                                   // 000000000340: 38A80902
	v_or_b32_e32 v90, v1, v3                                   // 000000000344: 38B40701
	v_or_b32_e32 v87, v1, v4                                   // 000000000348: 38AE0901
	v_add3_u32 v76, v0, v27, 32                                // 00000000034C: D655004C 02823700
	v_dual_mov_b32 v48, v56 :: v_dual_mov_b32 v51, v56         // 000000000354: CA100138 30320138
	v_dual_mov_b32 v50, v56 :: v_dual_mov_b32 v53, v56         // 00000000035C: CA100138 32340138
	v_dual_mov_b32 v52, v56 :: v_dual_mov_b32 v55, v56         // 000000000364: CA100138 34360138
	v_dual_mov_b32 v54, v56 :: v_dual_mov_b32 v41, v56         // 00000000036C: CA100138 36280138
	v_dual_mov_b32 v40, v56 :: v_dual_mov_b32 v43, v56         // 000000000374: CA100138 282A0138
	v_dual_mov_b32 v42, v56 :: v_dual_mov_b32 v45, v56         // 00000000037C: CA100138 2A2C0138
	v_dual_mov_b32 v44, v56 :: v_dual_mov_b32 v47, v56         // 000000000384: CA100138 2C2E0138
	v_dual_mov_b32 v46, v56 :: v_dual_mov_b32 v33, v56         // 00000000038C: CA100138 2E200138
	v_dual_mov_b32 v32, v56 :: v_dual_mov_b32 v35, v56         // 000000000394: CA100138 20220138
	v_dual_mov_b32 v34, v56 :: v_dual_mov_b32 v37, v56         // 00000000039C: CA100138 22240138
	v_dual_mov_b32 v36, v56 :: v_dual_mov_b32 v39, v56         // 0000000003A4: CA100138 24260138
	v_dual_mov_b32 v38, v56 :: v_dual_mov_b32 v25, v56         // 0000000003AC: CA100138 26180138
	v_dual_mov_b32 v24, v56 :: v_dual_mov_b32 v27, v56         // 0000000003B4: CA100138 181A0138
	v_dual_mov_b32 v26, v56 :: v_dual_mov_b32 v29, v56         // 0000000003BC: CA100138 1A1C0138
	v_dual_mov_b32 v28, v56 :: v_dual_mov_b32 v31, v56         // 0000000003C4: CA100138 1C1E0138
	v_dual_mov_b32 v30, v56 :: v_dual_mov_b32 v17, v56         // 0000000003CC: CA100138 1E100138
	v_dual_mov_b32 v16, v56 :: v_dual_mov_b32 v19, v56         // 0000000003D4: CA100138 10120138
	v_dual_mov_b32 v18, v56 :: v_dual_mov_b32 v21, v56         // 0000000003DC: CA100138 12140138
	v_dual_mov_b32 v20, v56 :: v_dual_mov_b32 v23, v56         // 0000000003E4: CA100138 14160138
	v_dual_mov_b32 v22, v56 :: v_dual_mov_b32 v9, v56          // 0000000003EC: CA100138 16080138
	v_dual_mov_b32 v8, v56 :: v_dual_mov_b32 v11, v56          // 0000000003F4: CA100138 080A0138
	v_dual_mov_b32 v10, v56 :: v_dual_mov_b32 v13, v56         // 0000000003FC: CA100138 0A0C0138
	v_dual_mov_b32 v12, v56 :: v_dual_mov_b32 v15, v56         // 000000000404: CA100138 0C0E0138
	v_dual_mov_b32 v14, v56 :: v_dual_mov_b32 v1, v56          // 00000000040C: CA100138 0E000138
	v_dual_mov_b32 v0, v56 :: v_dual_mov_b32 v3, v56           // 000000000414: CA100138 00020138
	v_dual_mov_b32 v2, v56 :: v_dual_mov_b32 v5, v56           // 00000000041C: CA100138 02040138
	v_dual_mov_b32 v4, v56 :: v_dual_mov_b32 v7, v56           // 000000000424: CA100138 04060138
	v_mov_b32_e32 v6, v56                                      // 00000000042C: 7E0C0338
	s_mov_b64 s[8:9], 0                                        // 000000000430: BE880180
	s_barrier_wait 0xffff                                      // 000000000434: BF94FFFF
	s_branch 96                                                // 000000000438: BFA00060 <gemm+0x5bc>
	s_wait_alu depctr_sa_sdst(0)                               // 00000000043C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s1                              // 000000000440: 8C7E017E
	s_clause 0x1                                               // 000000000444: BF850001
	global_load_b128 v[93:96], v[77:78], off offset:128        // 000000000448: EE05C07C 0000005D 0000804D
	global_load_b128 v[77:80], v[77:78], off offset:82048      // 000000000454: EE05C07C 0000004D 0140804D
	ds_load_b128 v[97:100], v86 offset:8192                    // 000000000460: DBFC2000 61000056
	ds_load_b128 v[101:104], v90 offset:8192                   // 000000000468: DBFC2000 6500005A
	ds_load_b128 v[105:108], v90 offset:9216                   // 000000000470: DBFC2400 6900005A
	ds_load_b128 v[109:112], v90 offset:10240                  // 000000000478: DBFC2800 6D00005A
	ds_load_b128 v[113:116], v90 offset:11264                  // 000000000480: DBFC2C00 7100005A
	ds_load_b128 v[117:120], v86 offset:9216                   // 000000000488: DBFC2400 75000056
	ds_load_b128 v[121:124], v84 offset:8192                   // 000000000490: DBFC2000 79000054
	ds_load_b128 v[125:128], v87 offset:8192                   // 000000000498: DBFC2000 7D000057
	ds_load_b128 v[129:132], v87 offset:9216                   // 0000000004A0: DBFC2400 81000057
	ds_load_b128 v[133:136], v87 offset:10240                  // 0000000004A8: DBFC2800 85000057
	ds_load_b128 v[137:140], v87 offset:11264                  // 0000000004B0: DBFC2C00 89000057
	ds_load_b128 v[141:144], v84 offset:9216                   // 0000000004B8: DBFC2400 8D000054
	s_wait_dscnt 0xa                                           // 0000000004C0: BFC6000A
	v_wmma_f32_16x16x16_bf16 v[56:63], v[97:100], v[101:104], v[56:63]// 0000000004C4: CC414038 1CE2CB61
	s_wait_dscnt 0x9                                           // 0000000004CC: BFC60009
	v_wmma_f32_16x16x16_bf16 v[48:55], v[97:100], v[105:108], v[48:55]// 0000000004D0: CC414030 1CC2D361
	s_wait_dscnt 0x8                                           // 0000000004D8: BFC60008
	v_wmma_f32_16x16x16_bf16 v[40:47], v[97:100], v[109:112], v[40:47]// 0000000004DC: CC414028 1CA2DB61
	s_wait_dscnt 0x7                                           // 0000000004E4: BFC60007
	v_wmma_f32_16x16x16_bf16 v[32:39], v[97:100], v[113:116], v[32:39]// 0000000004E8: CC414020 1C82E361
	s_wait_dscnt 0x6                                           // 0000000004F0: BFC60006
	v_wmma_f32_16x16x16_bf16 v[24:31], v[117:120], v[101:104], v[24:31]// 0000000004F4: CC414018 1C62CB75
	v_wmma_f32_16x16x16_bf16 v[16:23], v[117:120], v[105:108], v[16:23]// 0000000004FC: CC414010 1C42D375
	v_wmma_f32_16x16x16_bf16 v[8:15], v[117:120], v[109:112], v[8:15]// 000000000504: CC414008 1C22DB75
	v_wmma_f32_16x16x16_bf16 v[0:7], v[117:120], v[113:116], v[0:7]// 00000000050C: CC414000 1C02E375
	s_wait_dscnt 0x4                                           // 000000000514: BFC60004
	v_wmma_f32_16x16x16_bf16 v[56:63], v[121:124], v[125:128], v[56:63]// 000000000518: CC414038 1CE2FB79
	s_wait_dscnt 0x3                                           // 000000000520: BFC60003
	v_wmma_f32_16x16x16_bf16 v[48:55], v[121:124], v[129:132], v[48:55]// 000000000524: CC414030 1CC30379
	s_wait_dscnt 0x2                                           // 00000000052C: BFC60002
	v_wmma_f32_16x16x16_bf16 v[40:47], v[121:124], v[133:136], v[40:47]// 000000000530: CC414028 1CA30B79
	s_wait_dscnt 0x1                                           // 000000000538: BFC60001
	v_wmma_f32_16x16x16_bf16 v[32:39], v[121:124], v[137:140], v[32:39]// 00000000053C: CC414020 1C831379
	s_wait_dscnt 0x0                                           // 000000000544: BFC60000
	v_wmma_f32_16x16x16_bf16 v[24:31], v[141:144], v[125:128], v[24:31]// 000000000548: CC414018 1C62FB8D
	v_wmma_f32_16x16x16_bf16 v[16:23], v[141:144], v[129:132], v[16:23]// 000000000550: CC414010 1C43038D
	v_wmma_f32_16x16x16_bf16 v[8:15], v[141:144], v[133:136], v[8:15]// 000000000558: CC414008 1C230B8D
	v_wmma_f32_16x16x16_bf16 v[0:7], v[141:144], v[137:140], v[0:7]// 000000000560: CC414000 1C03138D
	s_wait_loadcnt 0x2                                         // 000000000568: BFC00002
	ds_store_b128 v85, v[64:67]                                // 00000000056C: DB7C0000 00004055
	s_wait_loadcnt 0x1                                         // 000000000574: BFC00001
	ds_store_b128 v89, v[93:96]                                // 000000000578: DB7C0000 00005D59
	ds_store_b128 v83, v[68:71]                                // 000000000580: DB7C0000 00004453
	s_wait_loadcnt 0x0                                         // 000000000588: BFC00000
	ds_store_b128 v88, v[77:80]                                // 00000000058C: DB7C0000 00004D58
	s_wait_dscnt 0x0                                           // 000000000594: BFC60000
	s_barrier_signal -1                                        // 000000000598: BE804EC1
	v_add_nc_u32_e32 v76, 64, v76                              // 00000000059C: 4A9898C0
	s_add_nc_u64 s[8:9], s[8:9], 0x80                          // 0000000005A0: A988FF08 00000080
	s_wait_alu depctr_sa_sdst(0)                               // 0000000005A8: BF88FF9E
	s_cmp_eq_u32 s8, 0x2780                                    // 0000000005AC: BF06FF08 00002780
	s_barrier_wait 0xffff                                      // 0000000005B4: BF94FFFF
	s_cbranch_scc1 154                                         // 0000000005B8: BFA2009A <gemm+0x824>
	v_ashrrev_i32_e32 v77, 31, v76                             // 0000000005BC: 349A989F
	v_dual_mov_b32 v65, 0 :: v_dual_mov_b32 v66, 0             // 0000000005C0: CA100080 41420080
	v_mov_b32_e32 v67, 0                                       // 0000000005C8: 7E860280
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000005CC: BF870093
	v_lshlrev_b64_e32 v[68:69], 1, v[76:77]                    // 0000000005D0: 3E889881
	v_add_co_u32 v77, s1, s6, v68                              // 0000000005D4: D700014D 00028806
	s_wait_alu depctr_va_sdst(0)                               // 0000000005DC: BF88F19F
	s_delay_alu instid0(VALU_DEP_2)                            // 0000000005E0: BF870002
	v_add_co_ci_u32_e64 v78, null, s7, v69, s1                 // 0000000005E4: D5207C4E 00068A07
	v_mov_b32_e32 v68, 0                                       // 0000000005EC: 7E880280
	s_and_saveexec_b32 s1, vcc_lo                              // 0000000005F0: BE81206A
	s_cbranch_execz 3                                          // 0000000005F4: BFA50003 <gemm+0x604>
	global_load_b128 v[65:68], v[77:78], off                   // 0000000005F8: EE05C07C 00000041 0000004D
	s_wait_alu depctr_sa_sdst(0)                               // 000000000604: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s1                              // 000000000608: 8C7E017E
	v_dual_mov_b32 v64, 0 :: v_dual_mov_b32 v69, 0             // 00000000060C: CA100080 40440080
	v_dual_mov_b32 v70, 0 :: v_dual_mov_b32 v71, 0             // 000000000614: CA100080 46460080
	v_mov_b32_e32 v72, 0                                       // 00000000061C: 7E900280
	s_and_saveexec_b32 s1, s0                                  // 000000000620: BE812000
	s_cbranch_execz 3                                          // 000000000624: BFA50003 <gemm+0x634>
	global_load_b128 v[69:72], v[77:78], off offset:81920      // 000000000628: EE05C07C 00000045 0140004D
	s_wait_alu depctr_sa_sdst(0)                               // 000000000634: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s1                              // 000000000638: 8C7E017E
	v_add_co_u32 v77, s1, v74, s8                              // 00000000063C: D700014D 0000114A
	s_wait_alu depctr_va_sdst(0)                               // 000000000644: BF88F19F
	v_add_co_ci_u32_e64 v78, null, s9, v75, s1                 // 000000000648: D5207C4E 00069609
	v_add_nc_u32_e32 v79, 32, v76                              // 000000000650: 4A9E98A0
	s_clause 0x1                                               // 000000000654: BF850001
	global_load_b128 v[93:96], v[77:78], off offset:64         // 000000000658: EE05C07C 0000005D 0000404D
	global_load_b128 v[97:100], v[77:78], off offset:81984     // 000000000664: EE05C07C 00000061 0140404D
	v_ashrrev_i32_e32 v80, 31, v79                             // 000000000670: 34A09E9F
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000000674: BF870091
	v_lshlrev_b64_e32 v[79:80], 1, v[79:80]                    // 000000000678: 3E9E9E81
	v_add_co_u32 v79, s1, s6, v79                              // 00000000067C: D700014F 00029E06
	s_wait_alu depctr_va_sdst(0)                               // 000000000684: BF88F19F
	s_delay_alu instid0(VALU_DEP_2)                            // 000000000688: BF870002
	v_add_co_ci_u32_e64 v80, null, s7, v80, s1                 // 00000000068C: D5207C50 0006A007
	ds_load_b128 v[101:104], v86                               // 000000000694: DBFC0000 65000056
	ds_load_b128 v[105:108], v90                               // 00000000069C: DBFC0000 6900005A
	ds_load_b128 v[109:112], v90 offset:1024                   // 0000000006A4: DBFC0400 6D00005A
	ds_load_b128 v[113:116], v90 offset:2048                   // 0000000006AC: DBFC0800 7100005A
	ds_load_b128 v[117:120], v90 offset:3072                   // 0000000006B4: DBFC0C00 7500005A
	ds_load_b128 v[121:124], v86 offset:1024                   // 0000000006BC: DBFC0400 79000056
	ds_load_b128 v[125:128], v84                               // 0000000006C4: DBFC0000 7D000054
	ds_load_b128 v[129:132], v87                               // 0000000006CC: DBFC0000 81000057
	ds_load_b128 v[133:136], v87 offset:1024                   // 0000000006D4: DBFC0400 85000057
	ds_load_b128 v[137:140], v87 offset:2048                   // 0000000006DC: DBFC0800 89000057
	ds_load_b128 v[141:144], v87 offset:3072                   // 0000000006E4: DBFC0C00 8D000057
	ds_load_b128 v[145:148], v84 offset:1024                   // 0000000006EC: DBFC0400 91000054
	s_wait_dscnt 0xa                                           // 0000000006F4: BFC6000A
	v_wmma_f32_16x16x16_bf16 v[56:63], v[101:104], v[105:108], v[56:63]// 0000000006F8: CC414038 1CE2D365
	s_wait_dscnt 0x9                                           // 000000000700: BFC60009
	v_wmma_f32_16x16x16_bf16 v[48:55], v[101:104], v[109:112], v[48:55]// 000000000704: CC414030 1CC2DB65
	s_wait_dscnt 0x8                                           // 00000000070C: BFC60008
	v_wmma_f32_16x16x16_bf16 v[40:47], v[101:104], v[113:116], v[40:47]// 000000000710: CC414028 1CA2E365
	s_wait_dscnt 0x7                                           // 000000000718: BFC60007
	v_wmma_f32_16x16x16_bf16 v[32:39], v[101:104], v[117:120], v[32:39]// 00000000071C: CC414020 1C82EB65
	s_wait_dscnt 0x6                                           // 000000000724: BFC60006
	v_wmma_f32_16x16x16_bf16 v[24:31], v[121:124], v[105:108], v[24:31]// 000000000728: CC414018 1C62D379
	v_wmma_f32_16x16x16_bf16 v[16:23], v[121:124], v[109:112], v[16:23]// 000000000730: CC414010 1C42DB79
	v_wmma_f32_16x16x16_bf16 v[8:15], v[121:124], v[113:116], v[8:15]// 000000000738: CC414008 1C22E379
	v_wmma_f32_16x16x16_bf16 v[0:7], v[121:124], v[117:120], v[0:7]// 000000000740: CC414000 1C02EB79
	s_wait_dscnt 0x4                                           // 000000000748: BFC60004
	v_wmma_f32_16x16x16_bf16 v[56:63], v[125:128], v[129:132], v[56:63]// 00000000074C: CC414038 1CE3037D
	s_wait_dscnt 0x3                                           // 000000000754: BFC60003
	v_wmma_f32_16x16x16_bf16 v[48:55], v[125:128], v[133:136], v[48:55]// 000000000758: CC414030 1CC30B7D
	s_wait_dscnt 0x2                                           // 000000000760: BFC60002
	v_wmma_f32_16x16x16_bf16 v[40:47], v[125:128], v[137:140], v[40:47]// 000000000764: CC414028 1CA3137D
	s_wait_dscnt 0x1                                           // 00000000076C: BFC60001
	v_wmma_f32_16x16x16_bf16 v[32:39], v[125:128], v[141:144], v[32:39]// 000000000770: CC414020 1C831B7D
	s_wait_dscnt 0x0                                           // 000000000778: BFC60000
	v_wmma_f32_16x16x16_bf16 v[24:31], v[145:148], v[129:132], v[24:31]// 00000000077C: CC414018 1C630391
	v_wmma_f32_16x16x16_bf16 v[16:23], v[145:148], v[133:136], v[16:23]// 000000000784: CC414010 1C430B91
	v_wmma_f32_16x16x16_bf16 v[8:15], v[145:148], v[137:140], v[8:15]// 00000000078C: CC414008 1C231391
	v_wmma_f32_16x16x16_bf16 v[0:7], v[145:148], v[141:144], v[0:7]// 000000000794: CC414000 1C031B91
	s_wait_loadcnt 0x2                                         // 00000000079C: BFC00002
	ds_store_b128 v85, v[65:68] offset:8192                    // 0000000007A0: DB7C2000 00004155
	s_wait_loadcnt 0x1                                         // 0000000007A8: BFC00001
	ds_store_b128 v89, v[93:96] offset:8192                    // 0000000007AC: DB7C2000 00005D59
	ds_store_b128 v83, v[69:72] offset:8192                    // 0000000007B4: DB7C2000 00004553
	s_wait_loadcnt 0x0                                         // 0000000007BC: BFC00000
	ds_store_b128 v88, v[97:100] offset:8192                   // 0000000007C0: DB7C2000 00006158
	s_wait_dscnt 0x0                                           // 0000000007C8: BFC60000
	s_barrier_signal -1                                        // 0000000007CC: BE804EC1
	v_dual_mov_b32 v65, 0 :: v_dual_mov_b32 v66, 0             // 0000000007D0: CA100080 41420080
	v_mov_b32_e32 v67, 0                                       // 0000000007D8: 7E860280
	s_barrier_wait 0xffff                                      // 0000000007DC: BF94FFFF
	s_and_saveexec_b32 s1, vcc_lo                              // 0000000007E0: BE81206A
	s_cbranch_execz 3                                          // 0000000007E4: BFA50003 <gemm+0x7f4>
	global_load_b128 v[64:67], v[79:80], off                   // 0000000007E8: EE05C07C 00000040 0000004F
	s_wait_alu depctr_sa_sdst(0)                               // 0000000007F4: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000007F8: 8C7E017E
	v_dual_mov_b32 v68, 0 :: v_dual_mov_b32 v69, 0             // 0000000007FC: CA100080 44440080
	v_dual_mov_b32 v70, 0 :: v_dual_mov_b32 v71, 0             // 000000000804: CA100080 46460080
	s_and_saveexec_b32 s1, s0                                  // 00000000080C: BE812000
	s_cbranch_execz 65290                                      // 000000000810: BFA5FF0A <gemm+0x43c>
	global_load_b128 v[68:71], v[79:80], off offset:81920      // 000000000814: EE05C07C 00000044 0140004F
	s_branch 65286                                             // 000000000820: BFA0FF06 <gemm+0x43c>
	s_clause 0x1                                               // 000000000824: BF850001
	global_load_b128 v[68:71], v[74:75], off offset:10176      // 000000000828: EE05C07C 00000044 0027C04A
	global_load_b128 v[64:67], v[74:75], off offset:92096      // 000000000834: EE05C07C 00000040 0167C04A
	v_add3_u32 v72, v73, s2, 0x13e0                            // 000000000840: D6550048 03FC0549 000013E0
	v_dual_mov_b32 v76, 0 :: v_dual_mov_b32 v79, 0             // 00000000084C: CA100080 4C4E0080
	v_dual_mov_b32 v78, 0 :: v_dual_mov_b32 v77, 0             // 000000000854: CA100080 4E4C0080
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000085C: BF870093
	v_ashrrev_i32_e32 v73, 31, v72                             // 000000000860: 3492909F
	v_lshlrev_b64_e32 v[72:73], 1, v[72:73]                    // 000000000864: 3E909081
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000000868: BF870121
	v_add_co_u32 v80, s1, s6, v72                              // 00000000086C: D7000150 00029006
	s_wait_alu depctr_va_sdst(0)                               // 000000000874: BF88F19F
	v_add_co_ci_u32_e64 v81, null, s7, v73, s1                 // 000000000878: D5207C51 00069207
	v_mov_b32_e32 v72, 0                                       // 000000000880: 7E900280
	s_and_saveexec_b32 s1, vcc_lo                              // 000000000884: BE81206A
	s_cbranch_execz 3                                          // 000000000888: BFA50003 <gemm+0x898>
	global_load_b128 v[76:79], v[80:81], off                   // 00000000088C: EE05C07C 0000004C 00000050
	s_wait_alu depctr_sa_sdst(0)                               // 000000000898: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s1                              // 00000000089C: 8C7E017E
	v_dual_mov_b32 v73, 0 :: v_dual_mov_b32 v74, 0             // 0000000008A0: CA100080 494A0080
	v_mov_b32_e32 v75, 0                                       // 0000000008A8: 7E960280
	s_and_saveexec_b32 s1, s0                                  // 0000000008AC: BE812000
	s_cbranch_execz 3                                          // 0000000008B0: BFA50003 <gemm+0x8c0>
	global_load_b128 v[72:75], v[80:81], off offset:81920      // 0000000008B4: EE05C07C 00000048 01400050
	s_wait_alu depctr_sa_sdst(0)                               // 0000000008C0: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000008C4: 8C7E017E
	v_lshl_or_b32 v82, v82, 3, v91                             // 0000000008C8: D6560052 056D0752
	s_lshl_b32 s0, s3, 7                                       // 0000000008D0: 84008703
	s_mul_i32 s11, s11, 0x28000                                // 0000000008D4: 960BFF0B 00028000
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 0000000008DC: BF870121
	v_mul_u32_u24_e32 v80, 0x500, v82                          // 0000000008E0: 16A0A4FF 00000500
	v_cmp_gt_i32_e32 vcc_lo, s10, v82                          // 0000000008E8: 7C88A40A
	v_or_b32_e32 v80, v80, v92                                 // 0000000008EC: 38A0B950
	s_wait_alu depctr_sa_sdst(0)                               // 0000000008F0: BF88FF9E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000008F4: BF870091
	v_add3_u32 v80, s11, s0, v80                               // 0000000008F8: D6550050 0540000B
	v_ashrrev_i32_e32 v81, 31, v80                             // 000000000900: 34A2A09F
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000000904: BF870091
	v_lshlrev_b64_e32 v[80:81], 1, v[80:81]                    // 000000000908: 3EA0A081
	v_add_co_u32 v80, s0, s4, v80                              // 00000000090C: D7000050 0002A004
	s_wait_alu depctr_va_sdst(0)                               // 000000000914: BF88F19F
	s_delay_alu instid0(VALU_DEP_2)                            // 000000000918: BF870002
	v_add_co_ci_u32_e64 v81, null, s5, v81, s0                 // 00000000091C: D5207C51 0002A205
	ds_load_b128 v[91:94], v86                                 // 000000000924: DBFC0000 5B000056
	ds_load_b128 v[95:98], v90                                 // 00000000092C: DBFC0000 5F00005A
	ds_load_b128 v[99:102], v90 offset:1024                    // 000000000934: DBFC0400 6300005A
	ds_load_b128 v[103:106], v90 offset:2048                   // 00000000093C: DBFC0800 6700005A
	ds_load_b128 v[107:110], v90 offset:3072                   // 000000000944: DBFC0C00 6B00005A
	ds_load_b128 v[111:114], v86 offset:1024                   // 00000000094C: DBFC0400 6F000056
	ds_load_b128 v[115:118], v84                               // 000000000954: DBFC0000 73000054
	ds_load_b128 v[119:122], v87                               // 00000000095C: DBFC0000 77000057
	ds_load_b128 v[123:126], v87 offset:1024                   // 000000000964: DBFC0400 7B000057
	ds_load_b128 v[127:130], v87 offset:2048                   // 00000000096C: DBFC0800 7F000057
	ds_load_b128 v[131:134], v87 offset:3072                   // 000000000974: DBFC0C00 83000057
	ds_load_b128 v[135:138], v84 offset:1024                   // 00000000097C: DBFC0400 87000054
	s_wait_dscnt 0xa                                           // 000000000984: BFC6000A
	v_wmma_f32_16x16x16_bf16 v[56:63], v[91:94], v[95:98], v[56:63]// 000000000988: CC414038 1CE2BF5B
	s_wait_dscnt 0x9                                           // 000000000990: BFC60009
	v_wmma_f32_16x16x16_bf16 v[48:55], v[91:94], v[99:102], v[48:55]// 000000000994: CC414030 1CC2C75B
	s_wait_dscnt 0x8                                           // 00000000099C: BFC60008
	v_wmma_f32_16x16x16_bf16 v[40:47], v[91:94], v[103:106], v[40:47]// 0000000009A0: CC414028 1CA2CF5B
	s_wait_dscnt 0x7                                           // 0000000009A8: BFC60007
	v_wmma_f32_16x16x16_bf16 v[32:39], v[91:94], v[107:110], v[32:39]// 0000000009AC: CC414020 1C82D75B
	s_wait_dscnt 0x6                                           // 0000000009B4: BFC60006
	v_wmma_f32_16x16x16_bf16 v[24:31], v[111:114], v[95:98], v[24:31]// 0000000009B8: CC414018 1C62BF6F
	v_wmma_f32_16x16x16_bf16 v[16:23], v[111:114], v[99:102], v[16:23]// 0000000009C0: CC414010 1C42C76F
	v_wmma_f32_16x16x16_bf16 v[8:15], v[111:114], v[103:106], v[8:15]// 0000000009C8: CC414008 1C22CF6F
	v_wmma_f32_16x16x16_bf16 v[0:7], v[111:114], v[107:110], v[0:7]// 0000000009D0: CC414000 1C02D76F
	s_wait_dscnt 0x4                                           // 0000000009D8: BFC60004
	v_wmma_f32_16x16x16_bf16 v[56:63], v[115:118], v[119:122], v[56:63]// 0000000009DC: CC414038 1CE2EF73
	s_wait_dscnt 0x3                                           // 0000000009E4: BFC60003
	v_wmma_f32_16x16x16_bf16 v[48:55], v[115:118], v[123:126], v[48:55]// 0000000009E8: CC414030 1CC2F773
	s_wait_dscnt 0x2                                           // 0000000009F0: BFC60002
	v_wmma_f32_16x16x16_bf16 v[40:47], v[115:118], v[127:130], v[40:47]// 0000000009F4: CC414028 1CA2FF73
	s_wait_dscnt 0x1                                           // 0000000009FC: BFC60001
	v_wmma_f32_16x16x16_bf16 v[32:39], v[115:118], v[131:134], v[32:39]// 000000000A00: CC414020 1C830773
	s_wait_dscnt 0x0                                           // 000000000A08: BFC60000
	v_wmma_f32_16x16x16_bf16 v[24:31], v[135:138], v[119:122], v[24:31]// 000000000A0C: CC414018 1C62EF87
	v_wmma_f32_16x16x16_bf16 v[16:23], v[135:138], v[123:126], v[16:23]// 000000000A14: CC414010 1C42F787
	v_wmma_f32_16x16x16_bf16 v[8:15], v[135:138], v[127:130], v[8:15]// 000000000A1C: CC414008 1C22FF87
	v_wmma_f32_16x16x16_bf16 v[0:7], v[135:138], v[131:134], v[0:7]// 000000000A24: CC414000 1C030787
	s_wait_loadcnt 0x0                                         // 000000000A2C: BFC00000
	ds_store_b128 v85, v[76:79] offset:8192                    // 000000000A30: DB7C2000 00004C55
	ds_store_b128 v89, v[68:71] offset:8192                    // 000000000A38: DB7C2000 00004459
	ds_store_b128 v83, v[72:75] offset:8192                    // 000000000A40: DB7C2000 00004853
	ds_store_b128 v88, v[64:67] offset:8192                    // 000000000A48: DB7C2000 00004058
	s_wait_dscnt 0x0                                           // 000000000A50: BFC60000
	s_barrier_signal -1                                        // 000000000A54: BE804EC1
	s_barrier_wait 0xffff                                      // 000000000A58: BF94FFFF
	ds_load_b128 v[64:67], v86 offset:8192                     // 000000000A5C: DBFC2000 40000056
	ds_load_b128 v[68:71], v90 offset:8192                     // 000000000A64: DBFC2000 4400005A
	ds_load_b128 v[72:75], v90 offset:9216                     // 000000000A6C: DBFC2400 4800005A
	ds_load_b128 v[76:79], v90 offset:10240                    // 000000000A74: DBFC2800 4C00005A
	ds_load_b128 v[88:91], v90 offset:11264                    // 000000000A7C: DBFC2C00 5800005A
	ds_load_b128 v[92:95], v86 offset:9216                     // 000000000A84: DBFC2400 5C000056
	ds_load_b128 v[96:99], v84 offset:8192                     // 000000000A8C: DBFC2000 60000054
	ds_load_b128 v[100:103], v87 offset:8192                   // 000000000A94: DBFC2000 64000057
	ds_load_b128 v[104:107], v87 offset:9216                   // 000000000A9C: DBFC2400 68000057
	ds_load_b128 v[108:111], v87 offset:10240                  // 000000000AA4: DBFC2800 6C000057
	ds_load_b128 v[112:115], v87 offset:11264                  // 000000000AAC: DBFC2C00 70000057
	ds_load_b128 v[83:86], v84 offset:9216                     // 000000000AB4: DBFC2400 53000054
	s_wait_dscnt 0xa                                           // 000000000ABC: BFC6000A
	v_wmma_f32_16x16x16_bf16 v[56:63], v[64:67], v[68:71], v[56:63]// 000000000AC0: CC414038 1CE28940
	s_wait_dscnt 0x9                                           // 000000000AC8: BFC60009
	v_wmma_f32_16x16x16_bf16 v[48:55], v[64:67], v[72:75], v[48:55]// 000000000ACC: CC414030 1CC29140
	s_wait_dscnt 0x8                                           // 000000000AD4: BFC60008
	v_wmma_f32_16x16x16_bf16 v[40:47], v[64:67], v[76:79], v[40:47]// 000000000AD8: CC414028 1CA29940
	s_wait_dscnt 0x7                                           // 000000000AE0: BFC60007
	v_wmma_f32_16x16x16_bf16 v[32:39], v[64:67], v[88:91], v[32:39]// 000000000AE4: CC414020 1C82B140
	s_wait_dscnt 0x6                                           // 000000000AEC: BFC60006
	v_wmma_f32_16x16x16_bf16 v[24:31], v[92:95], v[68:71], v[24:31]// 000000000AF0: CC414018 1C62895C
	v_wmma_f32_16x16x16_bf16 v[16:23], v[92:95], v[72:75], v[16:23]// 000000000AF8: CC414010 1C42915C
	v_wmma_f32_16x16x16_bf16 v[8:15], v[92:95], v[76:79], v[8:15]// 000000000B00: CC414008 1C22995C
	v_wmma_f32_16x16x16_bf16 v[0:7], v[92:95], v[88:91], v[0:7]// 000000000B08: CC414000 1C02B15C
	s_wait_dscnt 0x4                                           // 000000000B10: BFC60004
	v_wmma_f32_16x16x16_bf16 v[56:63], v[96:99], v[100:103], v[56:63]// 000000000B14: CC414038 1CE2C960
	s_wait_dscnt 0x3                                           // 000000000B1C: BFC60003
	v_wmma_f32_16x16x16_bf16 v[48:55], v[96:99], v[104:107], v[48:55]// 000000000B20: CC414030 1CC2D160
	s_wait_dscnt 0x2                                           // 000000000B28: BFC60002
	v_wmma_f32_16x16x16_bf16 v[40:47], v[96:99], v[108:111], v[40:47]// 000000000B2C: CC414028 1CA2D960
	s_wait_dscnt 0x1                                           // 000000000B34: BFC60001
	v_wmma_f32_16x16x16_bf16 v[32:39], v[96:99], v[112:115], v[32:39]// 000000000B38: CC414020 1C82E160
	s_wait_dscnt 0x0                                           // 000000000B40: BFC60000
	v_wmma_f32_16x16x16_bf16 v[24:31], v[83:86], v[100:103], v[24:31]// 000000000B44: CC414018 1C62C953
	v_wmma_f32_16x16x16_bf16 v[16:23], v[83:86], v[104:107], v[16:23]// 000000000B4C: CC414010 1C42D153
	v_wmma_f32_16x16x16_bf16 v[8:15], v[83:86], v[108:111], v[8:15]// 000000000B54: CC414008 1C22D953
	v_wmma_f32_16x16x16_bf16 v[0:7], v[83:86], v[112:115], v[0:7]// 000000000B5C: CC414000 1C02E153
	s_and_saveexec_b32 s1, vcc_lo                              // 000000000B64: BE81206A
	s_cbranch_execz 16                                         // 000000000B68: BFA50010 <gemm+0xbac>
	v_bfe_u32 v64, v56, 16, 1                                  // 000000000B6C: D6100040 02052138
	v_or_b32_e32 v65, 0x400000, v56                            // 000000000B74: 388270FF 00400000
	v_cmp_u_f32_e64 s0, v56, v56                               // 000000000B7C: D4180000 00027138
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000000B84: BF8700A3
	v_add3_u32 v64, v64, v56, 0x7fff                           // 000000000B88: D6550040 03FE7140 00007FFF
	s_wait_alu depctr_va_sdst(0)                               // 000000000B94: BF88F19F
	v_cndmask_b32_e64 v56, v64, v65, s0                        // 000000000B98: D5010038 00028340
	global_store_d16_hi_b16 v[80:81], v56, off                 // 000000000BA0: EE09407C 1C000000 00000050
	s_wait_alu depctr_sa_sdst(0)                               // 000000000BAC: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s1                              // 000000000BB0: 8C7E017E
	v_or_b32_e32 v56, 1, v82                                   // 000000000BB4: 3870A481
	s_delay_alu instid0(VALU_DEP_1)                            // 000000000BB8: BF870001
	v_cmp_gt_i32_e64 s0, s10, v56                              // 000000000BBC: D4440000 0002700A
	s_and_saveexec_b32 s2, s0                                  // 000000000BC4: BE822000
	s_cbranch_execz 16                                         // 000000000BC8: BFA50010 <gemm+0xc0c>
	v_bfe_u32 v56, v57, 16, 1                                  // 000000000BCC: D6100038 02052139
	v_or_b32_e32 v64, 0x400000, v57                            // 000000000BD4: 388072FF 00400000
	v_cmp_u_f32_e64 s1, v57, v57                               // 000000000BDC: D4180001 00027339
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000000BE4: BF8700A3
	v_add3_u32 v56, v56, v57, 0x7fff                           // 000000000BE8: D6550038 03FE7338 00007FFF
	s_wait_alu depctr_va_sdst(0)                               // 000000000BF4: BF88F19F
	v_cndmask_b32_e64 v56, v56, v64, s1                        // 000000000BF8: D5010038 00068138
	global_store_d16_hi_b16 v[80:81], v56, off offset:2560     // 000000000C00: EE09407C 1C000000 000A0050
	s_wait_alu depctr_sa_sdst(0)                               // 000000000C0C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s2                              // 000000000C10: 8C7E027E
	v_or_b32_e32 v56, 2, v82                                   // 000000000C14: 3870A482
	s_delay_alu instid0(VALU_DEP_1)                            // 000000000C18: BF870001
	v_cmp_gt_i32_e64 s1, s10, v56                              // 000000000C1C: D4440001 0002700A
	s_and_saveexec_b32 s3, s1                                  // 000000000C24: BE832001
	s_cbranch_execz 16                                         // 000000000C28: BFA50010 <gemm+0xc6c>
	v_bfe_u32 v56, v58, 16, 1                                  // 000000000C2C: D6100038 0205213A
	v_or_b32_e32 v57, 0x400000, v58                            // 000000000C34: 387274FF 00400000
	v_cmp_u_f32_e64 s2, v58, v58                               // 000000000C3C: D4180002 0002753A
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000000C44: BF8700A3
	v_add3_u32 v56, v56, v58, 0x7fff                           // 000000000C48: D6550038 03FE7538 00007FFF
	s_wait_alu depctr_va_sdst(0)                               // 000000000C54: BF88F19F
	v_cndmask_b32_e64 v56, v56, v57, s2                        // 000000000C58: D5010038 000A7338
	global_store_d16_hi_b16 v[80:81], v56, off offset:5120     // 000000000C60: EE09407C 1C000000 00140050
	s_wait_alu depctr_sa_sdst(0)                               // 000000000C6C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s3                              // 000000000C70: 8C7E037E
	v_or_b32_e32 v56, 3, v82                                   // 000000000C74: 3870A483
	s_delay_alu instid0(VALU_DEP_1)                            // 000000000C78: BF870001
	v_cmp_gt_i32_e64 s2, s10, v56                              // 000000000C7C: D4440002 0002700A
	s_and_saveexec_b32 s4, s2                                  // 000000000C84: BE842002
	s_cbranch_execz 16                                         // 000000000C88: BFA50010 <gemm+0xccc>
	v_bfe_u32 v56, v59, 16, 1                                  // 000000000C8C: D6100038 0205213B
	v_or_b32_e32 v57, 0x400000, v59                            // 000000000C94: 387276FF 00400000
	v_cmp_u_f32_e64 s3, v59, v59                               // 000000000C9C: D4180003 0002773B
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000000CA4: BF8700A3
	v_add3_u32 v56, v56, v59, 0x7fff                           // 000000000CA8: D6550038 03FE7738 00007FFF
	s_wait_alu depctr_va_sdst(0)                               // 000000000CB4: BF88F19F
	v_cndmask_b32_e64 v56, v56, v57, s3                        // 000000000CB8: D5010038 000E7338
	global_store_d16_hi_b16 v[80:81], v56, off offset:7680     // 000000000CC0: EE09407C 1C000000 001E0050
	s_wait_alu depctr_sa_sdst(0)                               // 000000000CCC: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s4                              // 000000000CD0: 8C7E047E
	v_or_b32_e32 v56, 4, v82                                   // 000000000CD4: 3870A484
	s_delay_alu instid0(VALU_DEP_1)                            // 000000000CD8: BF870001
	v_cmp_gt_i32_e64 s3, s10, v56                              // 000000000CDC: D4440003 0002700A
	s_and_saveexec_b32 s5, s3                                  // 000000000CE4: BE852003
	s_cbranch_execz 16                                         // 000000000CE8: BFA50010 <gemm+0xd2c>
	v_bfe_u32 v56, v60, 16, 1                                  // 000000000CEC: D6100038 0205213C
	v_or_b32_e32 v57, 0x400000, v60                            // 000000000CF4: 387278FF 00400000
	v_cmp_u_f32_e64 s4, v60, v60                               // 000000000CFC: D4180004 0002793C
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000000D04: BF8700A3
	v_add3_u32 v56, v56, v60, 0x7fff                           // 000000000D08: D6550038 03FE7938 00007FFF
	s_wait_alu depctr_va_sdst(0)                               // 000000000D14: BF88F19F
	v_cndmask_b32_e64 v56, v56, v57, s4                        // 000000000D18: D5010038 00127338
	global_store_d16_hi_b16 v[80:81], v56, off offset:10240    // 000000000D20: EE09407C 1C000000 00280050
	s_wait_alu depctr_sa_sdst(0)                               // 000000000D2C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s5                              // 000000000D30: 8C7E057E
	v_or_b32_e32 v56, 5, v82                                   // 000000000D34: 3870A485
	s_delay_alu instid0(VALU_DEP_1)                            // 000000000D38: BF870001
	v_cmp_gt_i32_e64 s4, s10, v56                              // 000000000D3C: D4440004 0002700A
	s_and_saveexec_b32 s6, s4                                  // 000000000D44: BE862004
	s_cbranch_execz 16                                         // 000000000D48: BFA50010 <gemm+0xd8c>
	v_bfe_u32 v56, v61, 16, 1                                  // 000000000D4C: D6100038 0205213D
	v_or_b32_e32 v57, 0x400000, v61                            // 000000000D54: 38727AFF 00400000
	v_cmp_u_f32_e64 s5, v61, v61                               // 000000000D5C: D4180005 00027B3D
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000000D64: BF8700A3
	v_add3_u32 v56, v56, v61, 0x7fff                           // 000000000D68: D6550038 03FE7B38 00007FFF
	s_wait_alu depctr_va_sdst(0)                               // 000000000D74: BF88F19F
	v_cndmask_b32_e64 v56, v56, v57, s5                        // 000000000D78: D5010038 00167338
	global_store_d16_hi_b16 v[80:81], v56, off offset:12800    // 000000000D80: EE09407C 1C000000 00320050
	s_wait_alu depctr_sa_sdst(0)                               // 000000000D8C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s6                              // 000000000D90: 8C7E067E
	v_or_b32_e32 v56, 6, v82                                   // 000000000D94: 3870A486
	s_delay_alu instid0(VALU_DEP_1)                            // 000000000D98: BF870001
	v_cmp_gt_i32_e64 s5, s10, v56                              // 000000000D9C: D4440005 0002700A
	s_and_saveexec_b32 s7, s5                                  // 000000000DA4: BE872005
	s_cbranch_execz 16                                         // 000000000DA8: BFA50010 <gemm+0xdec>
	v_bfe_u32 v56, v62, 16, 1                                  // 000000000DAC: D6100038 0205213E
	v_or_b32_e32 v57, 0x400000, v62                            // 000000000DB4: 38727CFF 00400000
	v_cmp_u_f32_e64 s6, v62, v62                               // 000000000DBC: D4180006 00027D3E
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000000DC4: BF8700A3
	v_add3_u32 v56, v56, v62, 0x7fff                           // 000000000DC8: D6550038 03FE7D38 00007FFF
	s_wait_alu depctr_va_sdst(0)                               // 000000000DD4: BF88F19F
	v_cndmask_b32_e64 v56, v56, v57, s6                        // 000000000DD8: D5010038 001A7338
	global_store_d16_hi_b16 v[80:81], v56, off offset:15360    // 000000000DE0: EE09407C 1C000000 003C0050
	s_wait_alu depctr_sa_sdst(0)                               // 000000000DEC: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s7                              // 000000000DF0: 8C7E077E
	v_or_b32_e32 v56, 7, v82                                   // 000000000DF4: 3870A487
	s_delay_alu instid0(VALU_DEP_1)                            // 000000000DF8: BF870001
	v_cmp_gt_i32_e64 s6, s10, v56                              // 000000000DFC: D4440006 0002700A
	s_and_saveexec_b32 s8, s6                                  // 000000000E04: BE882006
	s_cbranch_execnz 384                                       // 000000000E08: BFA60180 <gemm+0x140c>
	s_wait_alu depctr_sa_sdst(0)                               // 000000000E0C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000000E10: 8C7E087E
	s_and_saveexec_b32 s8, vcc_lo                              // 000000000E14: BE88206A
	s_cbranch_execnz 400                                       // 000000000E18: BFA60190 <gemm+0x145c>
	s_wait_alu depctr_sa_sdst(0)                               // 000000000E1C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000000E20: 8C7E087E
	s_and_saveexec_b32 s8, s0                                  // 000000000E24: BE882000
	s_cbranch_execnz 416                                       // 000000000E28: BFA601A0 <gemm+0x14ac>
	s_wait_alu depctr_sa_sdst(0)                               // 000000000E2C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000000E30: 8C7E087E
	s_and_saveexec_b32 s8, s1                                  // 000000000E34: BE882001
	s_cbranch_execnz 432                                       // 000000000E38: BFA601B0 <gemm+0x14fc>
	s_wait_alu depctr_sa_sdst(0)                               // 000000000E3C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000000E40: 8C7E087E
	s_and_saveexec_b32 s8, s2                                  // 000000000E44: BE882002
	s_cbranch_execnz 448                                       // 000000000E48: BFA601C0 <gemm+0x154c>
	s_wait_alu depctr_sa_sdst(0)                               // 000000000E4C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000000E50: 8C7E087E
	s_and_saveexec_b32 s8, s3                                  // 000000000E54: BE882003
	s_cbranch_execnz 464                                       // 000000000E58: BFA601D0 <gemm+0x159c>
	s_wait_alu depctr_sa_sdst(0)                               // 000000000E5C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000000E60: 8C7E087E
	s_and_saveexec_b32 s8, s4                                  // 000000000E64: BE882004
	s_cbranch_execnz 480                                       // 000000000E68: BFA601E0 <gemm+0x15ec>
	s_wait_alu depctr_sa_sdst(0)                               // 000000000E6C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000000E70: 8C7E087E
	s_and_saveexec_b32 s8, s5                                  // 000000000E74: BE882005
	s_cbranch_execnz 496                                       // 000000000E78: BFA601F0 <gemm+0x163c>
	s_wait_alu depctr_sa_sdst(0)                               // 000000000E7C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000000E80: 8C7E087E
	s_and_saveexec_b32 s8, s6                                  // 000000000E84: BE882006
	s_cbranch_execnz 512                                       // 000000000E88: BFA60200 <gemm+0x168c>
	s_wait_alu depctr_sa_sdst(0)                               // 000000000E8C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000000E90: 8C7E087E
	s_and_saveexec_b32 s8, vcc_lo                              // 000000000E94: BE88206A
	s_cbranch_execnz 528                                       // 000000000E98: BFA60210 <gemm+0x16dc>
	s_wait_alu depctr_sa_sdst(0)                               // 000000000E9C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000000EA0: 8C7E087E
	s_and_saveexec_b32 s8, s0                                  // 000000000EA4: BE882000
	s_cbranch_execnz 544                                       // 000000000EA8: BFA60220 <gemm+0x172c>
	s_wait_alu depctr_sa_sdst(0)                               // 000000000EAC: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000000EB0: 8C7E087E
	s_and_saveexec_b32 s8, s1                                  // 000000000EB4: BE882001
	s_cbranch_execnz 560                                       // 000000000EB8: BFA60230 <gemm+0x177c>
	s_wait_alu depctr_sa_sdst(0)                               // 000000000EBC: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000000EC0: 8C7E087E
	s_and_saveexec_b32 s8, s2                                  // 000000000EC4: BE882002
	s_cbranch_execnz 576                                       // 000000000EC8: BFA60240 <gemm+0x17cc>
	s_wait_alu depctr_sa_sdst(0)                               // 000000000ECC: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000000ED0: 8C7E087E
	s_and_saveexec_b32 s8, s3                                  // 000000000ED4: BE882003
	s_cbranch_execnz 592                                       // 000000000ED8: BFA60250 <gemm+0x181c>
	s_wait_alu depctr_sa_sdst(0)                               // 000000000EDC: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000000EE0: 8C7E087E
	s_and_saveexec_b32 s8, s4                                  // 000000000EE4: BE882004
	s_cbranch_execnz 608                                       // 000000000EE8: BFA60260 <gemm+0x186c>
	s_wait_alu depctr_sa_sdst(0)                               // 000000000EEC: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000000EF0: 8C7E087E
	s_and_saveexec_b32 s8, s5                                  // 000000000EF4: BE882005
	s_cbranch_execnz 624                                       // 000000000EF8: BFA60270 <gemm+0x18bc>
	s_wait_alu depctr_sa_sdst(0)                               // 000000000EFC: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000000F00: 8C7E087E
	s_and_saveexec_b32 s8, s6                                  // 000000000F04: BE882006
	s_cbranch_execnz 640                                       // 000000000F08: BFA60280 <gemm+0x190c>
	s_wait_alu depctr_sa_sdst(0)                               // 000000000F0C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000000F10: 8C7E087E
	s_and_saveexec_b32 s7, vcc_lo                              // 000000000F14: BE87206A
	s_cbranch_execnz 656                                       // 000000000F18: BFA60290 <gemm+0x195c>
	s_wait_alu depctr_sa_sdst(0)                               // 000000000F1C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s7                              // 000000000F20: 8C7E077E
	s_and_saveexec_b32 s7, s0                                  // 000000000F24: BE872000
	s_cbranch_execnz 670                                       // 000000000F28: BFA6029E <gemm+0x19a4>
	s_wait_alu depctr_sa_sdst(0)                               // 000000000F2C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s7                              // 000000000F30: 8C7E077E
	s_and_saveexec_b32 s0, s1                                  // 000000000F34: BE802001
	s_cbranch_execnz 684                                       // 000000000F38: BFA602AC <gemm+0x19ec>
	s_wait_alu depctr_sa_sdst(0)                               // 000000000F3C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s0                              // 000000000F40: 8C7E007E
	s_and_saveexec_b32 s0, s2                                  // 000000000F44: BE802002
	s_cbranch_execnz 698                                       // 000000000F48: BFA602BA <gemm+0x1a34>
	s_wait_alu depctr_sa_sdst(0)                               // 000000000F4C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s0                              // 000000000F50: 8C7E007E
	s_and_saveexec_b32 s0, s3                                  // 000000000F54: BE802003
	s_cbranch_execnz 712                                       // 000000000F58: BFA602C8 <gemm+0x1a7c>
	s_wait_alu depctr_sa_sdst(0)                               // 000000000F5C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s0                              // 000000000F60: 8C7E007E
	s_and_saveexec_b32 s0, s4                                  // 000000000F64: BE802004
	s_cbranch_execnz 726                                       // 000000000F68: BFA602D6 <gemm+0x1ac4>
	s_wait_alu depctr_sa_sdst(0)                               // 000000000F6C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s0                              // 000000000F70: 8C7E007E
	s_and_saveexec_b32 s0, s5                                  // 000000000F74: BE802005
	s_cbranch_execnz 740                                       // 000000000F78: BFA602E4 <gemm+0x1b0c>
	s_wait_alu depctr_sa_sdst(0)                               // 000000000F7C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s0                              // 000000000F80: 8C7E007E
	s_and_saveexec_b32 s0, s6                                  // 000000000F84: BE802006
	s_cbranch_execz 14                                         // 000000000F88: BFA5000E <gemm+0xfc4>
	v_bfe_u32 v32, v39, 16, 1                                  // 000000000F8C: D6100020 02052127
	v_or_b32_e32 v33, 0x400000, v39                            // 000000000F94: 38424EFF 00400000
	v_cmp_u_f32_e32 vcc_lo, v39, v39                           // 000000000F9C: 7C304F27
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000000FA0: BF8700A3
	v_add3_u32 v32, v32, v39, 0x7fff                           // 000000000FA4: D6550020 03FE4F20 00007FFF
	s_wait_alu depctr_va_vcc(0)                                // 000000000FB0: BF88FF9D
	v_cndmask_b32_e32 v32, v32, v33, vcc_lo                    // 000000000FB4: 02404320
	global_store_d16_hi_b16 v[80:81], v32, off offset:18016    // 000000000FB8: EE09407C 10000000 00466050
	s_wait_alu depctr_sa_sdst(0)                               // 000000000FC4: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s0                              // 000000000FC8: 8C7E007E
	v_or_b32_e32 v32, 16, v82                                  // 000000000FCC: 3840A490
	s_delay_alu instid0(VALU_DEP_1)                            // 000000000FD0: BF870001
	v_cmp_gt_i32_e32 vcc_lo, s10, v32                          // 000000000FD4: 7C88400A
	s_and_saveexec_b32 s1, vcc_lo                              // 000000000FD8: BE81206A
	s_cbranch_execz 16                                         // 000000000FDC: BFA50010 <gemm+0x1020>
	v_bfe_u32 v32, v24, 16, 1                                  // 000000000FE0: D6100020 02052118
	v_or_b32_e32 v33, 0x400000, v24                            // 000000000FE8: 384230FF 00400000
	v_cmp_u_f32_e64 s0, v24, v24                               // 000000000FF0: D4180000 00023118
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000000FF8: BF8700A3
	v_add3_u32 v32, v32, v24, 0x7fff                           // 000000000FFC: D6550020 03FE3120 00007FFF
	s_wait_alu depctr_va_sdst(0)                               // 000000001008: BF88F19F
	v_cndmask_b32_e64 v24, v32, v33, s0                        // 00000000100C: D5010018 00024320
	global_store_d16_hi_b16 v[80:81], v24, off offset:40960    // 000000001014: EE09407C 0C000000 00A00050
	s_wait_alu depctr_sa_sdst(0)                               // 000000001020: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s1                              // 000000001024: 8C7E017E
	v_or_b32_e32 v24, 17, v82                                  // 000000001028: 3830A491
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000102C: BF870001
	v_cmp_gt_i32_e64 s0, s10, v24                              // 000000001030: D4440000 0002300A
	s_and_saveexec_b32 s2, s0                                  // 000000001038: BE822000
	s_cbranch_execz 16                                         // 00000000103C: BFA50010 <gemm+0x1080>
	v_bfe_u32 v24, v25, 16, 1                                  // 000000001040: D6100018 02052119
	v_or_b32_e32 v32, 0x400000, v25                            // 000000001048: 384032FF 00400000
	v_cmp_u_f32_e64 s1, v25, v25                               // 000000001050: D4180001 00023319
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001058: BF8700A3
	v_add3_u32 v24, v24, v25, 0x7fff                           // 00000000105C: D6550018 03FE3318 00007FFF
	s_wait_alu depctr_va_sdst(0)                               // 000000001068: BF88F19F
	v_cndmask_b32_e64 v24, v24, v32, s1                        // 00000000106C: D5010018 00064118
	global_store_d16_hi_b16 v[80:81], v24, off offset:43520    // 000000001074: EE09407C 0C000000 00AA0050
	s_wait_alu depctr_sa_sdst(0)                               // 000000001080: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s2                              // 000000001084: 8C7E027E
	v_or_b32_e32 v24, 18, v82                                  // 000000001088: 3830A492
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000108C: BF870001
	v_cmp_gt_i32_e64 s1, s10, v24                              // 000000001090: D4440001 0002300A
	s_and_saveexec_b32 s3, s1                                  // 000000001098: BE832001
	s_cbranch_execz 16                                         // 00000000109C: BFA50010 <gemm+0x10e0>
	v_bfe_u32 v24, v26, 16, 1                                  // 0000000010A0: D6100018 0205211A
	v_or_b32_e32 v25, 0x400000, v26                            // 0000000010A8: 383234FF 00400000
	v_cmp_u_f32_e64 s2, v26, v26                               // 0000000010B0: D4180002 0002351A
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000010B8: BF8700A3
	v_add3_u32 v24, v24, v26, 0x7fff                           // 0000000010BC: D6550018 03FE3518 00007FFF
	s_wait_alu depctr_va_sdst(0)                               // 0000000010C8: BF88F19F
	v_cndmask_b32_e64 v24, v24, v25, s2                        // 0000000010CC: D5010018 000A3318
	global_store_d16_hi_b16 v[80:81], v24, off offset:46080    // 0000000010D4: EE09407C 0C000000 00B40050
	s_wait_alu depctr_sa_sdst(0)                               // 0000000010E0: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s3                              // 0000000010E4: 8C7E037E
	v_or_b32_e32 v24, 19, v82                                  // 0000000010E8: 3830A493
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000010EC: BF870001
	v_cmp_gt_i32_e64 s2, s10, v24                              // 0000000010F0: D4440002 0002300A
	s_and_saveexec_b32 s4, s2                                  // 0000000010F8: BE842002
	s_cbranch_execz 16                                         // 0000000010FC: BFA50010 <gemm+0x1140>
	v_bfe_u32 v24, v27, 16, 1                                  // 000000001100: D6100018 0205211B
	v_or_b32_e32 v25, 0x400000, v27                            // 000000001108: 383236FF 00400000
	v_cmp_u_f32_e64 s3, v27, v27                               // 000000001110: D4180003 0002371B
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001118: BF8700A3
	v_add3_u32 v24, v24, v27, 0x7fff                           // 00000000111C: D6550018 03FE3718 00007FFF
	s_wait_alu depctr_va_sdst(0)                               // 000000001128: BF88F19F
	v_cndmask_b32_e64 v24, v24, v25, s3                        // 00000000112C: D5010018 000E3318
	global_store_d16_hi_b16 v[80:81], v24, off offset:48640    // 000000001134: EE09407C 0C000000 00BE0050
	s_wait_alu depctr_sa_sdst(0)                               // 000000001140: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s4                              // 000000001144: 8C7E047E
	v_or_b32_e32 v24, 20, v82                                  // 000000001148: 3830A494
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000114C: BF870001
	v_cmp_gt_i32_e64 s3, s10, v24                              // 000000001150: D4440003 0002300A
	s_and_saveexec_b32 s5, s3                                  // 000000001158: BE852003
	s_cbranch_execz 16                                         // 00000000115C: BFA50010 <gemm+0x11a0>
	v_bfe_u32 v24, v28, 16, 1                                  // 000000001160: D6100018 0205211C
	v_or_b32_e32 v25, 0x400000, v28                            // 000000001168: 383238FF 00400000
	v_cmp_u_f32_e64 s4, v28, v28                               // 000000001170: D4180004 0002391C
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001178: BF8700A3
	v_add3_u32 v24, v24, v28, 0x7fff                           // 00000000117C: D6550018 03FE3918 00007FFF
	s_wait_alu depctr_va_sdst(0)                               // 000000001188: BF88F19F
	v_cndmask_b32_e64 v24, v24, v25, s4                        // 00000000118C: D5010018 00123318
	global_store_d16_hi_b16 v[80:81], v24, off offset:51200    // 000000001194: EE09407C 0C000000 00C80050
	s_wait_alu depctr_sa_sdst(0)                               // 0000000011A0: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s5                              // 0000000011A4: 8C7E057E
	v_or_b32_e32 v24, 21, v82                                  // 0000000011A8: 3830A495
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000011AC: BF870001
	v_cmp_gt_i32_e64 s4, s10, v24                              // 0000000011B0: D4440004 0002300A
	s_and_saveexec_b32 s6, s4                                  // 0000000011B8: BE862004
	s_cbranch_execz 16                                         // 0000000011BC: BFA50010 <gemm+0x1200>
	v_bfe_u32 v24, v29, 16, 1                                  // 0000000011C0: D6100018 0205211D
	v_or_b32_e32 v25, 0x400000, v29                            // 0000000011C8: 38323AFF 00400000
	v_cmp_u_f32_e64 s5, v29, v29                               // 0000000011D0: D4180005 00023B1D
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000011D8: BF8700A3
	v_add3_u32 v24, v24, v29, 0x7fff                           // 0000000011DC: D6550018 03FE3B18 00007FFF
	s_wait_alu depctr_va_sdst(0)                               // 0000000011E8: BF88F19F
	v_cndmask_b32_e64 v24, v24, v25, s5                        // 0000000011EC: D5010018 00163318
	global_store_d16_hi_b16 v[80:81], v24, off offset:53760    // 0000000011F4: EE09407C 0C000000 00D20050
	s_wait_alu depctr_sa_sdst(0)                               // 000000001200: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s6                              // 000000001204: 8C7E067E
	v_or_b32_e32 v24, 22, v82                                  // 000000001208: 3830A496
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000120C: BF870001
	v_cmp_gt_i32_e64 s5, s10, v24                              // 000000001210: D4440005 0002300A
	s_and_saveexec_b32 s7, s5                                  // 000000001218: BE872005
	s_cbranch_execz 16                                         // 00000000121C: BFA50010 <gemm+0x1260>
	v_bfe_u32 v24, v30, 16, 1                                  // 000000001220: D6100018 0205211E
	v_or_b32_e32 v25, 0x400000, v30                            // 000000001228: 38323CFF 00400000
	v_cmp_u_f32_e64 s6, v30, v30                               // 000000001230: D4180006 00023D1E
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001238: BF8700A3
	v_add3_u32 v24, v24, v30, 0x7fff                           // 00000000123C: D6550018 03FE3D18 00007FFF
	s_wait_alu depctr_va_sdst(0)                               // 000000001248: BF88F19F
	v_cndmask_b32_e64 v24, v24, v25, s6                        // 00000000124C: D5010018 001A3318
	global_store_d16_hi_b16 v[80:81], v24, off offset:56320    // 000000001254: EE09407C 0C000000 00DC0050
	s_wait_alu depctr_sa_sdst(0)                               // 000000001260: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s7                              // 000000001264: 8C7E077E
	v_or_b32_e32 v24, 23, v82                                  // 000000001268: 3830A497
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000126C: BF870001
	v_cmp_gt_i32_e64 s6, s10, v24                              // 000000001270: D4440006 0002300A
	s_and_saveexec_b32 s8, s6                                  // 000000001278: BE882006
	s_cbranch_execnz 566                                       // 00000000127C: BFA60236 <gemm+0x1b58>
	s_wait_alu depctr_sa_sdst(0)                               // 000000001280: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000001284: 8C7E087E
	s_and_saveexec_b32 s8, vcc_lo                              // 000000001288: BE88206A
	s_cbranch_execnz 582                                       // 00000000128C: BFA60246 <gemm+0x1ba8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000001290: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000001294: 8C7E087E
	s_and_saveexec_b32 s8, s0                                  // 000000001298: BE882000
	s_cbranch_execnz 598                                       // 00000000129C: BFA60256 <gemm+0x1bf8>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000012A0: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 0000000012A4: 8C7E087E
	s_and_saveexec_b32 s8, s1                                  // 0000000012A8: BE882001
	s_cbranch_execnz 614                                       // 0000000012AC: BFA60266 <gemm+0x1c48>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000012B0: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 0000000012B4: 8C7E087E
	s_and_saveexec_b32 s8, s2                                  // 0000000012B8: BE882002
	s_cbranch_execnz 630                                       // 0000000012BC: BFA60276 <gemm+0x1c98>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000012C0: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 0000000012C4: 8C7E087E
	s_and_saveexec_b32 s8, s3                                  // 0000000012C8: BE882003
	s_cbranch_execnz 646                                       // 0000000012CC: BFA60286 <gemm+0x1ce8>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000012D0: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 0000000012D4: 8C7E087E
	s_and_saveexec_b32 s8, s4                                  // 0000000012D8: BE882004
	s_cbranch_execnz 662                                       // 0000000012DC: BFA60296 <gemm+0x1d38>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000012E0: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 0000000012E4: 8C7E087E
	s_and_saveexec_b32 s8, s5                                  // 0000000012E8: BE882005
	s_cbranch_execnz 678                                       // 0000000012EC: BFA602A6 <gemm+0x1d88>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000012F0: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 0000000012F4: 8C7E087E
	s_and_saveexec_b32 s8, s6                                  // 0000000012F8: BE882006
	s_cbranch_execnz 694                                       // 0000000012FC: BFA602B6 <gemm+0x1dd8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000001300: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000001304: 8C7E087E
	s_and_saveexec_b32 s8, vcc_lo                              // 000000001308: BE88206A
	s_cbranch_execnz 710                                       // 00000000130C: BFA602C6 <gemm+0x1e28>
	s_wait_alu depctr_sa_sdst(0)                               // 000000001310: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000001314: 8C7E087E
	s_and_saveexec_b32 s8, s0                                  // 000000001318: BE882000
	s_cbranch_execnz 726                                       // 00000000131C: BFA602D6 <gemm+0x1e78>
	s_wait_alu depctr_sa_sdst(0)                               // 000000001320: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000001324: 8C7E087E
	s_and_saveexec_b32 s8, s1                                  // 000000001328: BE882001
	s_cbranch_execnz 742                                       // 00000000132C: BFA602E6 <gemm+0x1ec8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000001330: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000001334: 8C7E087E
	s_and_saveexec_b32 s8, s2                                  // 000000001338: BE882002
	s_cbranch_execnz 758                                       // 00000000133C: BFA602F6 <gemm+0x1f18>
	s_wait_alu depctr_sa_sdst(0)                               // 000000001340: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000001344: 8C7E087E
	s_and_saveexec_b32 s8, s3                                  // 000000001348: BE882003
	s_cbranch_execnz 774                                       // 00000000134C: BFA60306 <gemm+0x1f68>
	s_wait_alu depctr_sa_sdst(0)                               // 000000001350: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000001354: 8C7E087E
	s_and_saveexec_b32 s8, s4                                  // 000000001358: BE882004
	s_cbranch_execnz 790                                       // 00000000135C: BFA60316 <gemm+0x1fb8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000001360: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000001364: 8C7E087E
	s_and_saveexec_b32 s8, s5                                  // 000000001368: BE882005
	s_cbranch_execnz 806                                       // 00000000136C: BFA60326 <gemm+0x2008>
	s_wait_alu depctr_sa_sdst(0)                               // 000000001370: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000001374: 8C7E087E
	s_and_saveexec_b32 s8, s6                                  // 000000001378: BE882006
	s_cbranch_execnz 822                                       // 00000000137C: BFA60336 <gemm+0x2058>
	s_wait_alu depctr_sa_sdst(0)                               // 000000001380: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000001384: 8C7E087E
	s_and_saveexec_b32 s7, vcc_lo                              // 000000001388: BE87206A
	s_cbranch_execnz 838                                       // 00000000138C: BFA60346 <gemm+0x20a8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000001390: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s7                              // 000000001394: 8C7E077E
	s_and_saveexec_b32 s7, s0                                  // 000000001398: BE872000
	s_cbranch_execnz 852                                       // 00000000139C: BFA60354 <gemm+0x20f0>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000013A0: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s7                              // 0000000013A4: 8C7E077E
	s_and_saveexec_b32 s0, s1                                  // 0000000013A8: BE802001
	s_cbranch_execnz 866                                       // 0000000013AC: BFA60362 <gemm+0x2138>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000013B0: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s0                              // 0000000013B4: 8C7E007E
	s_and_saveexec_b32 s0, s2                                  // 0000000013B8: BE802002
	s_cbranch_execnz 880                                       // 0000000013BC: BFA60370 <gemm+0x2180>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000013C0: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s0                              // 0000000013C4: 8C7E007E
	s_and_saveexec_b32 s0, s3                                  // 0000000013C8: BE802003
	s_cbranch_execnz 894                                       // 0000000013CC: BFA6037E <gemm+0x21c8>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000013D0: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s0                              // 0000000013D4: 8C7E007E
	s_and_saveexec_b32 s0, s4                                  // 0000000013D8: BE802004
	s_cbranch_execnz 908                                       // 0000000013DC: BFA6038C <gemm+0x2210>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000013E0: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s0                              // 0000000013E4: 8C7E007E
	s_and_saveexec_b32 s0, s5                                  // 0000000013E8: BE802005
	s_cbranch_execnz 922                                       // 0000000013EC: BFA6039A <gemm+0x2258>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000013F0: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s0                              // 0000000013F4: 8C7E007E
	s_and_saveexec_b32 s0, s6                                  // 0000000013F8: BE802006
	s_cbranch_execnz 936                                       // 0000000013FC: BFA603A8 <gemm+0x22a0>
	s_nop 0                                                    // 000000001400: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001404: BFB60003
	s_endpgm                                                   // 000000001408: BFB00000
	v_bfe_u32 v56, v63, 16, 1                                  // 00000000140C: D6100038 0205213F
	v_or_b32_e32 v57, 0x400000, v63                            // 000000001414: 38727EFF 00400000
	v_cmp_u_f32_e64 s7, v63, v63                               // 00000000141C: D4180007 00027F3F
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001424: BF8700A3
	v_add3_u32 v56, v56, v63, 0x7fff                           // 000000001428: D6550038 03FE7F38 00007FFF
	s_wait_alu depctr_va_sdst(0)                               // 000000001434: BF88F19F
	v_cndmask_b32_e64 v56, v56, v57, s7                        // 000000001438: D5010038 001E7338
	global_store_d16_hi_b16 v[80:81], v56, off offset:17920    // 000000001440: EE09407C 1C000000 00460050
	s_wait_alu depctr_sa_sdst(0)                               // 00000000144C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000001450: 8C7E087E
	s_and_saveexec_b32 s8, vcc_lo                              // 000000001454: BE88206A
	s_cbranch_execz 65136                                      // 000000001458: BFA5FE70 <gemm+0xe1c>
	v_bfe_u32 v56, v48, 16, 1                                  // 00000000145C: D6100038 02052130
	v_or_b32_e32 v57, 0x400000, v48                            // 000000001464: 387260FF 00400000
	v_cmp_u_f32_e64 s7, v48, v48                               // 00000000146C: D4180007 00026130
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001474: BF8700A3
	v_add3_u32 v56, v56, v48, 0x7fff                           // 000000001478: D6550038 03FE6138 00007FFF
	s_wait_alu depctr_va_sdst(0)                               // 000000001484: BF88F19F
	v_cndmask_b32_e64 v48, v56, v57, s7                        // 000000001488: D5010030 001E7338
	global_store_d16_hi_b16 v[80:81], v48, off offset:32       // 000000001490: EE09407C 18000000 00002050
	s_wait_alu depctr_sa_sdst(0)                               // 00000000149C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 0000000014A0: 8C7E087E
	s_and_saveexec_b32 s8, s0                                  // 0000000014A4: BE882000
	s_cbranch_execz 65120                                      // 0000000014A8: BFA5FE60 <gemm+0xe2c>
	v_bfe_u32 v48, v49, 16, 1                                  // 0000000014AC: D6100030 02052131
	v_or_b32_e32 v56, 0x400000, v49                            // 0000000014B4: 387062FF 00400000
	v_cmp_u_f32_e64 s7, v49, v49                               // 0000000014BC: D4180007 00026331
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000014C4: BF8700A3
	v_add3_u32 v48, v48, v49, 0x7fff                           // 0000000014C8: D6550030 03FE6330 00007FFF
	s_wait_alu depctr_va_sdst(0)                               // 0000000014D4: BF88F19F
	v_cndmask_b32_e64 v48, v48, v56, s7                        // 0000000014D8: D5010030 001E7130
	global_store_d16_hi_b16 v[80:81], v48, off offset:2592     // 0000000014E0: EE09407C 18000000 000A2050
	s_wait_alu depctr_sa_sdst(0)                               // 0000000014EC: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 0000000014F0: 8C7E087E
	s_and_saveexec_b32 s8, s1                                  // 0000000014F4: BE882001
	s_cbranch_execz 65104                                      // 0000000014F8: BFA5FE50 <gemm+0xe3c>
	v_bfe_u32 v48, v50, 16, 1                                  // 0000000014FC: D6100030 02052132
	v_or_b32_e32 v49, 0x400000, v50                            // 000000001504: 386264FF 00400000
	v_cmp_u_f32_e64 s7, v50, v50                               // 00000000150C: D4180007 00026532
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001514: BF8700A3
	v_add3_u32 v48, v48, v50, 0x7fff                           // 000000001518: D6550030 03FE6530 00007FFF
	s_wait_alu depctr_va_sdst(0)                               // 000000001524: BF88F19F
	v_cndmask_b32_e64 v48, v48, v49, s7                        // 000000001528: D5010030 001E6330
	global_store_d16_hi_b16 v[80:81], v48, off offset:5152     // 000000001530: EE09407C 18000000 00142050
	s_wait_alu depctr_sa_sdst(0)                               // 00000000153C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000001540: 8C7E087E
	s_and_saveexec_b32 s8, s2                                  // 000000001544: BE882002
	s_cbranch_execz 65088                                      // 000000001548: BFA5FE40 <gemm+0xe4c>
	v_bfe_u32 v48, v51, 16, 1                                  // 00000000154C: D6100030 02052133
	v_or_b32_e32 v49, 0x400000, v51                            // 000000001554: 386266FF 00400000
	v_cmp_u_f32_e64 s7, v51, v51                               // 00000000155C: D4180007 00026733
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001564: BF8700A3
	v_add3_u32 v48, v48, v51, 0x7fff                           // 000000001568: D6550030 03FE6730 00007FFF
	s_wait_alu depctr_va_sdst(0)                               // 000000001574: BF88F19F
	v_cndmask_b32_e64 v48, v48, v49, s7                        // 000000001578: D5010030 001E6330
	global_store_d16_hi_b16 v[80:81], v48, off offset:7712     // 000000001580: EE09407C 18000000 001E2050
	s_wait_alu depctr_sa_sdst(0)                               // 00000000158C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000001590: 8C7E087E
	s_and_saveexec_b32 s8, s3                                  // 000000001594: BE882003
	s_cbranch_execz 65072                                      // 000000001598: BFA5FE30 <gemm+0xe5c>
	v_bfe_u32 v48, v52, 16, 1                                  // 00000000159C: D6100030 02052134
	v_or_b32_e32 v49, 0x400000, v52                            // 0000000015A4: 386268FF 00400000
	v_cmp_u_f32_e64 s7, v52, v52                               // 0000000015AC: D4180007 00026934
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000015B4: BF8700A3
	v_add3_u32 v48, v48, v52, 0x7fff                           // 0000000015B8: D6550030 03FE6930 00007FFF
	s_wait_alu depctr_va_sdst(0)                               // 0000000015C4: BF88F19F
	v_cndmask_b32_e64 v48, v48, v49, s7                        // 0000000015C8: D5010030 001E6330
	global_store_d16_hi_b16 v[80:81], v48, off offset:10272    // 0000000015D0: EE09407C 18000000 00282050
	s_wait_alu depctr_sa_sdst(0)                               // 0000000015DC: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 0000000015E0: 8C7E087E
	s_and_saveexec_b32 s8, s4                                  // 0000000015E4: BE882004
	s_cbranch_execz 65056                                      // 0000000015E8: BFA5FE20 <gemm+0xe6c>
	v_bfe_u32 v48, v53, 16, 1                                  // 0000000015EC: D6100030 02052135
	v_or_b32_e32 v49, 0x400000, v53                            // 0000000015F4: 38626AFF 00400000
	v_cmp_u_f32_e64 s7, v53, v53                               // 0000000015FC: D4180007 00026B35
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001604: BF8700A3
	v_add3_u32 v48, v48, v53, 0x7fff                           // 000000001608: D6550030 03FE6B30 00007FFF
	s_wait_alu depctr_va_sdst(0)                               // 000000001614: BF88F19F
	v_cndmask_b32_e64 v48, v48, v49, s7                        // 000000001618: D5010030 001E6330
	global_store_d16_hi_b16 v[80:81], v48, off offset:12832    // 000000001620: EE09407C 18000000 00322050
	s_wait_alu depctr_sa_sdst(0)                               // 00000000162C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000001630: 8C7E087E
	s_and_saveexec_b32 s8, s5                                  // 000000001634: BE882005
	s_cbranch_execz 65040                                      // 000000001638: BFA5FE10 <gemm+0xe7c>
	v_bfe_u32 v48, v54, 16, 1                                  // 00000000163C: D6100030 02052136
	v_or_b32_e32 v49, 0x400000, v54                            // 000000001644: 38626CFF 00400000
	v_cmp_u_f32_e64 s7, v54, v54                               // 00000000164C: D4180007 00026D36
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001654: BF8700A3
	v_add3_u32 v48, v48, v54, 0x7fff                           // 000000001658: D6550030 03FE6D30 00007FFF
	s_wait_alu depctr_va_sdst(0)                               // 000000001664: BF88F19F
	v_cndmask_b32_e64 v48, v48, v49, s7                        // 000000001668: D5010030 001E6330
	global_store_d16_hi_b16 v[80:81], v48, off offset:15392    // 000000001670: EE09407C 18000000 003C2050
	s_wait_alu depctr_sa_sdst(0)                               // 00000000167C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000001680: 8C7E087E
	s_and_saveexec_b32 s8, s6                                  // 000000001684: BE882006
	s_cbranch_execz 65024                                      // 000000001688: BFA5FE00 <gemm+0xe8c>
	v_bfe_u32 v48, v55, 16, 1                                  // 00000000168C: D6100030 02052137
	v_or_b32_e32 v49, 0x400000, v55                            // 000000001694: 38626EFF 00400000
	v_cmp_u_f32_e64 s7, v55, v55                               // 00000000169C: D4180007 00026F37
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000016A4: BF8700A3
	v_add3_u32 v48, v48, v55, 0x7fff                           // 0000000016A8: D6550030 03FE6F30 00007FFF
	s_wait_alu depctr_va_sdst(0)                               // 0000000016B4: BF88F19F
	v_cndmask_b32_e64 v48, v48, v49, s7                        // 0000000016B8: D5010030 001E6330
	global_store_d16_hi_b16 v[80:81], v48, off offset:17952    // 0000000016C0: EE09407C 18000000 00462050
	s_wait_alu depctr_sa_sdst(0)                               // 0000000016CC: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 0000000016D0: 8C7E087E
	s_and_saveexec_b32 s8, vcc_lo                              // 0000000016D4: BE88206A
	s_cbranch_execz 65008                                      // 0000000016D8: BFA5FDF0 <gemm+0xe9c>
	v_bfe_u32 v48, v40, 16, 1                                  // 0000000016DC: D6100030 02052128
	v_or_b32_e32 v49, 0x400000, v40                            // 0000000016E4: 386250FF 00400000
	v_cmp_u_f32_e64 s7, v40, v40                               // 0000000016EC: D4180007 00025128
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000016F4: BF8700A3
	v_add3_u32 v48, v48, v40, 0x7fff                           // 0000000016F8: D6550030 03FE5130 00007FFF
	s_wait_alu depctr_va_sdst(0)                               // 000000001704: BF88F19F
	v_cndmask_b32_e64 v40, v48, v49, s7                        // 000000001708: D5010028 001E6330
	global_store_d16_hi_b16 v[80:81], v40, off offset:64       // 000000001710: EE09407C 14000000 00004050
	s_wait_alu depctr_sa_sdst(0)                               // 00000000171C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000001720: 8C7E087E
	s_and_saveexec_b32 s8, s0                                  // 000000001724: BE882000
	s_cbranch_execz 64992                                      // 000000001728: BFA5FDE0 <gemm+0xeac>
	v_bfe_u32 v40, v41, 16, 1                                  // 00000000172C: D6100028 02052129
	v_or_b32_e32 v48, 0x400000, v41                            // 000000001734: 386052FF 00400000
	v_cmp_u_f32_e64 s7, v41, v41                               // 00000000173C: D4180007 00025329
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001744: BF8700A3
	v_add3_u32 v40, v40, v41, 0x7fff                           // 000000001748: D6550028 03FE5328 00007FFF
	s_wait_alu depctr_va_sdst(0)                               // 000000001754: BF88F19F
	v_cndmask_b32_e64 v40, v40, v48, s7                        // 000000001758: D5010028 001E6128
	global_store_d16_hi_b16 v[80:81], v40, off offset:2624     // 000000001760: EE09407C 14000000 000A4050
	s_wait_alu depctr_sa_sdst(0)                               // 00000000176C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000001770: 8C7E087E
	s_and_saveexec_b32 s8, s1                                  // 000000001774: BE882001
	s_cbranch_execz 64976                                      // 000000001778: BFA5FDD0 <gemm+0xebc>
	v_bfe_u32 v40, v42, 16, 1                                  // 00000000177C: D6100028 0205212A
	v_or_b32_e32 v41, 0x400000, v42                            // 000000001784: 385254FF 00400000
	v_cmp_u_f32_e64 s7, v42, v42                               // 00000000178C: D4180007 0002552A
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001794: BF8700A3
	v_add3_u32 v40, v40, v42, 0x7fff                           // 000000001798: D6550028 03FE5528 00007FFF
	s_wait_alu depctr_va_sdst(0)                               // 0000000017A4: BF88F19F
	v_cndmask_b32_e64 v40, v40, v41, s7                        // 0000000017A8: D5010028 001E5328
	global_store_d16_hi_b16 v[80:81], v40, off offset:5184     // 0000000017B0: EE09407C 14000000 00144050
	s_wait_alu depctr_sa_sdst(0)                               // 0000000017BC: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 0000000017C0: 8C7E087E
	s_and_saveexec_b32 s8, s2                                  // 0000000017C4: BE882002
	s_cbranch_execz 64960                                      // 0000000017C8: BFA5FDC0 <gemm+0xecc>
	v_bfe_u32 v40, v43, 16, 1                                  // 0000000017CC: D6100028 0205212B
	v_or_b32_e32 v41, 0x400000, v43                            // 0000000017D4: 385256FF 00400000
	v_cmp_u_f32_e64 s7, v43, v43                               // 0000000017DC: D4180007 0002572B
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000017E4: BF8700A3
	v_add3_u32 v40, v40, v43, 0x7fff                           // 0000000017E8: D6550028 03FE5728 00007FFF
	s_wait_alu depctr_va_sdst(0)                               // 0000000017F4: BF88F19F
	v_cndmask_b32_e64 v40, v40, v41, s7                        // 0000000017F8: D5010028 001E5328
	global_store_d16_hi_b16 v[80:81], v40, off offset:7744     // 000000001800: EE09407C 14000000 001E4050
	s_wait_alu depctr_sa_sdst(0)                               // 00000000180C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000001810: 8C7E087E
	s_and_saveexec_b32 s8, s3                                  // 000000001814: BE882003
	s_cbranch_execz 64944                                      // 000000001818: BFA5FDB0 <gemm+0xedc>
	v_bfe_u32 v40, v44, 16, 1                                  // 00000000181C: D6100028 0205212C
	v_or_b32_e32 v41, 0x400000, v44                            // 000000001824: 385258FF 00400000
	v_cmp_u_f32_e64 s7, v44, v44                               // 00000000182C: D4180007 0002592C
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001834: BF8700A3
	v_add3_u32 v40, v40, v44, 0x7fff                           // 000000001838: D6550028 03FE5928 00007FFF
	s_wait_alu depctr_va_sdst(0)                               // 000000001844: BF88F19F
	v_cndmask_b32_e64 v40, v40, v41, s7                        // 000000001848: D5010028 001E5328
	global_store_d16_hi_b16 v[80:81], v40, off offset:10304    // 000000001850: EE09407C 14000000 00284050
	s_wait_alu depctr_sa_sdst(0)                               // 00000000185C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000001860: 8C7E087E
	s_and_saveexec_b32 s8, s4                                  // 000000001864: BE882004
	s_cbranch_execz 64928                                      // 000000001868: BFA5FDA0 <gemm+0xeec>
	v_bfe_u32 v40, v45, 16, 1                                  // 00000000186C: D6100028 0205212D
	v_or_b32_e32 v41, 0x400000, v45                            // 000000001874: 38525AFF 00400000
	v_cmp_u_f32_e64 s7, v45, v45                               // 00000000187C: D4180007 00025B2D
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001884: BF8700A3
	v_add3_u32 v40, v40, v45, 0x7fff                           // 000000001888: D6550028 03FE5B28 00007FFF
	s_wait_alu depctr_va_sdst(0)                               // 000000001894: BF88F19F
	v_cndmask_b32_e64 v40, v40, v41, s7                        // 000000001898: D5010028 001E5328
	global_store_d16_hi_b16 v[80:81], v40, off offset:12864    // 0000000018A0: EE09407C 14000000 00324050
	s_wait_alu depctr_sa_sdst(0)                               // 0000000018AC: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 0000000018B0: 8C7E087E
	s_and_saveexec_b32 s8, s5                                  // 0000000018B4: BE882005
	s_cbranch_execz 64912                                      // 0000000018B8: BFA5FD90 <gemm+0xefc>
	v_bfe_u32 v40, v46, 16, 1                                  // 0000000018BC: D6100028 0205212E
	v_or_b32_e32 v41, 0x400000, v46                            // 0000000018C4: 38525CFF 00400000
	v_cmp_u_f32_e64 s7, v46, v46                               // 0000000018CC: D4180007 00025D2E
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000018D4: BF8700A3
	v_add3_u32 v40, v40, v46, 0x7fff                           // 0000000018D8: D6550028 03FE5D28 00007FFF
	s_wait_alu depctr_va_sdst(0)                               // 0000000018E4: BF88F19F
	v_cndmask_b32_e64 v40, v40, v41, s7                        // 0000000018E8: D5010028 001E5328
	global_store_d16_hi_b16 v[80:81], v40, off offset:15424    // 0000000018F0: EE09407C 14000000 003C4050
	s_wait_alu depctr_sa_sdst(0)                               // 0000000018FC: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000001900: 8C7E087E
	s_and_saveexec_b32 s8, s6                                  // 000000001904: BE882006
	s_cbranch_execz 64896                                      // 000000001908: BFA5FD80 <gemm+0xf0c>
	v_bfe_u32 v40, v47, 16, 1                                  // 00000000190C: D6100028 0205212F
	v_or_b32_e32 v41, 0x400000, v47                            // 000000001914: 38525EFF 00400000
	v_cmp_u_f32_e64 s7, v47, v47                               // 00000000191C: D4180007 00025F2F
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001924: BF8700A3
	v_add3_u32 v40, v40, v47, 0x7fff                           // 000000001928: D6550028 03FE5F28 00007FFF
	s_wait_alu depctr_va_sdst(0)                               // 000000001934: BF88F19F
	v_cndmask_b32_e64 v40, v40, v41, s7                        // 000000001938: D5010028 001E5328
	global_store_d16_hi_b16 v[80:81], v40, off offset:17984    // 000000001940: EE09407C 14000000 00464050
	s_wait_alu depctr_sa_sdst(0)                               // 00000000194C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000001950: 8C7E087E
	s_and_saveexec_b32 s7, vcc_lo                              // 000000001954: BE87206A
	s_cbranch_execz 64880                                      // 000000001958: BFA5FD70 <gemm+0xf1c>
	v_bfe_u32 v40, v32, 16, 1                                  // 00000000195C: D6100028 02052120
	v_or_b32_e32 v41, 0x400000, v32                            // 000000001964: 385240FF 00400000
	v_cmp_u_f32_e32 vcc_lo, v32, v32                           // 00000000196C: 7C304120
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001970: BF8700A3
	v_add3_u32 v40, v40, v32, 0x7fff                           // 000000001974: D6550028 03FE4128 00007FFF
	s_wait_alu depctr_va_vcc(0)                                // 000000001980: BF88FF9D
	v_cndmask_b32_e32 v32, v40, v41, vcc_lo                    // 000000001984: 02405328
	global_store_d16_hi_b16 v[80:81], v32, off offset:96       // 000000001988: EE09407C 10000000 00006050
	s_wait_alu depctr_sa_sdst(0)                               // 000000001994: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s7                              // 000000001998: 8C7E077E
	s_and_saveexec_b32 s7, s0                                  // 00000000199C: BE872000
	s_cbranch_execz 64866                                      // 0000000019A0: BFA5FD62 <gemm+0xf2c>
	v_bfe_u32 v32, v33, 16, 1                                  // 0000000019A4: D6100020 02052121
	v_or_b32_e32 v40, 0x400000, v33                            // 0000000019AC: 385042FF 00400000
	v_cmp_u_f32_e32 vcc_lo, v33, v33                           // 0000000019B4: 7C304321
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000019B8: BF8700A3
	v_add3_u32 v32, v32, v33, 0x7fff                           // 0000000019BC: D6550020 03FE4320 00007FFF
	s_wait_alu depctr_va_vcc(0)                                // 0000000019C8: BF88FF9D
	v_cndmask_b32_e32 v32, v32, v40, vcc_lo                    // 0000000019CC: 02405120
	global_store_d16_hi_b16 v[80:81], v32, off offset:2656     // 0000000019D0: EE09407C 10000000 000A6050
	s_wait_alu depctr_sa_sdst(0)                               // 0000000019DC: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s7                              // 0000000019E0: 8C7E077E
	s_and_saveexec_b32 s0, s1                                  // 0000000019E4: BE802001
	s_cbranch_execz 64852                                      // 0000000019E8: BFA5FD54 <gemm+0xf3c>
	v_bfe_u32 v32, v34, 16, 1                                  // 0000000019EC: D6100020 02052122
	v_or_b32_e32 v33, 0x400000, v34                            // 0000000019F4: 384244FF 00400000
	v_cmp_u_f32_e32 vcc_lo, v34, v34                           // 0000000019FC: 7C304522
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001A00: BF8700A3
	v_add3_u32 v32, v32, v34, 0x7fff                           // 000000001A04: D6550020 03FE4520 00007FFF
	s_wait_alu depctr_va_vcc(0)                                // 000000001A10: BF88FF9D
	v_cndmask_b32_e32 v32, v32, v33, vcc_lo                    // 000000001A14: 02404320
	global_store_d16_hi_b16 v[80:81], v32, off offset:5216     // 000000001A18: EE09407C 10000000 00146050
	s_wait_alu depctr_sa_sdst(0)                               // 000000001A24: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s0                              // 000000001A28: 8C7E007E
	s_and_saveexec_b32 s0, s2                                  // 000000001A2C: BE802002
	s_cbranch_execz 64838                                      // 000000001A30: BFA5FD46 <gemm+0xf4c>
	v_bfe_u32 v32, v35, 16, 1                                  // 000000001A34: D6100020 02052123
	v_or_b32_e32 v33, 0x400000, v35                            // 000000001A3C: 384246FF 00400000
	v_cmp_u_f32_e32 vcc_lo, v35, v35                           // 000000001A44: 7C304723
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001A48: BF8700A3
	v_add3_u32 v32, v32, v35, 0x7fff                           // 000000001A4C: D6550020 03FE4720 00007FFF
	s_wait_alu depctr_va_vcc(0)                                // 000000001A58: BF88FF9D
	v_cndmask_b32_e32 v32, v32, v33, vcc_lo                    // 000000001A5C: 02404320
	global_store_d16_hi_b16 v[80:81], v32, off offset:7776     // 000000001A60: EE09407C 10000000 001E6050
	s_wait_alu depctr_sa_sdst(0)                               // 000000001A6C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s0                              // 000000001A70: 8C7E007E
	s_and_saveexec_b32 s0, s3                                  // 000000001A74: BE802003
	s_cbranch_execz 64824                                      // 000000001A78: BFA5FD38 <gemm+0xf5c>
	v_bfe_u32 v32, v36, 16, 1                                  // 000000001A7C: D6100020 02052124
	v_or_b32_e32 v33, 0x400000, v36                            // 000000001A84: 384248FF 00400000
	v_cmp_u_f32_e32 vcc_lo, v36, v36                           // 000000001A8C: 7C304924
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001A90: BF8700A3
	v_add3_u32 v32, v32, v36, 0x7fff                           // 000000001A94: D6550020 03FE4920 00007FFF
	s_wait_alu depctr_va_vcc(0)                                // 000000001AA0: BF88FF9D
	v_cndmask_b32_e32 v32, v32, v33, vcc_lo                    // 000000001AA4: 02404320
	global_store_d16_hi_b16 v[80:81], v32, off offset:10336    // 000000001AA8: EE09407C 10000000 00286050
	s_wait_alu depctr_sa_sdst(0)                               // 000000001AB4: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s0                              // 000000001AB8: 8C7E007E
	s_and_saveexec_b32 s0, s4                                  // 000000001ABC: BE802004
	s_cbranch_execz 64810                                      // 000000001AC0: BFA5FD2A <gemm+0xf6c>
	v_bfe_u32 v32, v37, 16, 1                                  // 000000001AC4: D6100020 02052125
	v_or_b32_e32 v33, 0x400000, v37                            // 000000001ACC: 38424AFF 00400000
	v_cmp_u_f32_e32 vcc_lo, v37, v37                           // 000000001AD4: 7C304B25
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001AD8: BF8700A3
	v_add3_u32 v32, v32, v37, 0x7fff                           // 000000001ADC: D6550020 03FE4B20 00007FFF
	s_wait_alu depctr_va_vcc(0)                                // 000000001AE8: BF88FF9D
	v_cndmask_b32_e32 v32, v32, v33, vcc_lo                    // 000000001AEC: 02404320
	global_store_d16_hi_b16 v[80:81], v32, off offset:12896    // 000000001AF0: EE09407C 10000000 00326050
	s_wait_alu depctr_sa_sdst(0)                               // 000000001AFC: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s0                              // 000000001B00: 8C7E007E
	s_and_saveexec_b32 s0, s5                                  // 000000001B04: BE802005
	s_cbranch_execz 64796                                      // 000000001B08: BFA5FD1C <gemm+0xf7c>
	v_bfe_u32 v32, v38, 16, 1                                  // 000000001B0C: D6100020 02052126
	v_or_b32_e32 v33, 0x400000, v38                            // 000000001B14: 38424CFF 00400000
	v_cmp_u_f32_e32 vcc_lo, v38, v38                           // 000000001B1C: 7C304D26
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001B20: BF8700A3
	v_add3_u32 v32, v32, v38, 0x7fff                           // 000000001B24: D6550020 03FE4D20 00007FFF
	s_wait_alu depctr_va_vcc(0)                                // 000000001B30: BF88FF9D
	v_cndmask_b32_e32 v32, v32, v33, vcc_lo                    // 000000001B34: 02404320
	global_store_d16_hi_b16 v[80:81], v32, off offset:15456    // 000000001B38: EE09407C 10000000 003C6050
	s_wait_alu depctr_sa_sdst(0)                               // 000000001B44: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s0                              // 000000001B48: 8C7E007E
	s_and_saveexec_b32 s0, s6                                  // 000000001B4C: BE802006
	s_cbranch_execnz 64782                                     // 000000001B50: BFA6FD0E <gemm+0xf8c>
	s_branch 64795                                             // 000000001B54: BFA0FD1B <gemm+0xfc4>
	v_bfe_u32 v24, v31, 16, 1                                  // 000000001B58: D6100018 0205211F
	v_or_b32_e32 v25, 0x400000, v31                            // 000000001B60: 38323EFF 00400000
	v_cmp_u_f32_e64 s7, v31, v31                               // 000000001B68: D4180007 00023F1F
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001B70: BF8700A3
	v_add3_u32 v24, v24, v31, 0x7fff                           // 000000001B74: D6550018 03FE3F18 00007FFF
	s_wait_alu depctr_va_sdst(0)                               // 000000001B80: BF88F19F
	v_cndmask_b32_e64 v24, v24, v25, s7                        // 000000001B84: D5010018 001E3318
	global_store_d16_hi_b16 v[80:81], v24, off offset:58880    // 000000001B8C: EE09407C 0C000000 00E60050
	s_wait_alu depctr_sa_sdst(0)                               // 000000001B98: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000001B9C: 8C7E087E
	s_and_saveexec_b32 s8, vcc_lo                              // 000000001BA0: BE88206A
	s_cbranch_execz 64954                                      // 000000001BA4: BFA5FDBA <gemm+0x1290>
	v_bfe_u32 v24, v16, 16, 1                                  // 000000001BA8: D6100018 02052110
	v_or_b32_e32 v25, 0x400000, v16                            // 000000001BB0: 383220FF 00400000
	v_cmp_u_f32_e64 s7, v16, v16                               // 000000001BB8: D4180007 00022110
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001BC0: BF8700A3
	v_add3_u32 v24, v24, v16, 0x7fff                           // 000000001BC4: D6550018 03FE2118 00007FFF
	s_wait_alu depctr_va_sdst(0)                               // 000000001BD0: BF88F19F
	v_cndmask_b32_e64 v16, v24, v25, s7                        // 000000001BD4: D5010010 001E3318
	global_store_d16_hi_b16 v[80:81], v16, off offset:40992    // 000000001BDC: EE09407C 08000000 00A02050
	s_wait_alu depctr_sa_sdst(0)                               // 000000001BE8: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000001BEC: 8C7E087E
	s_and_saveexec_b32 s8, s0                                  // 000000001BF0: BE882000
	s_cbranch_execz 64938                                      // 000000001BF4: BFA5FDAA <gemm+0x12a0>
	v_bfe_u32 v16, v17, 16, 1                                  // 000000001BF8: D6100010 02052111
	v_or_b32_e32 v24, 0x400000, v17                            // 000000001C00: 383022FF 00400000
	v_cmp_u_f32_e64 s7, v17, v17                               // 000000001C08: D4180007 00022311
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001C10: BF8700A3
	v_add3_u32 v16, v16, v17, 0x7fff                           // 000000001C14: D6550010 03FE2310 00007FFF
	s_wait_alu depctr_va_sdst(0)                               // 000000001C20: BF88F19F
	v_cndmask_b32_e64 v16, v16, v24, s7                        // 000000001C24: D5010010 001E3110
	global_store_d16_hi_b16 v[80:81], v16, off offset:43552    // 000000001C2C: EE09407C 08000000 00AA2050
	s_wait_alu depctr_sa_sdst(0)                               // 000000001C38: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000001C3C: 8C7E087E
	s_and_saveexec_b32 s8, s1                                  // 000000001C40: BE882001
	s_cbranch_execz 64922                                      // 000000001C44: BFA5FD9A <gemm+0x12b0>
	v_bfe_u32 v16, v18, 16, 1                                  // 000000001C48: D6100010 02052112
	v_or_b32_e32 v17, 0x400000, v18                            // 000000001C50: 382224FF 00400000
	v_cmp_u_f32_e64 s7, v18, v18                               // 000000001C58: D4180007 00022512
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001C60: BF8700A3
	v_add3_u32 v16, v16, v18, 0x7fff                           // 000000001C64: D6550010 03FE2510 00007FFF
	s_wait_alu depctr_va_sdst(0)                               // 000000001C70: BF88F19F
	v_cndmask_b32_e64 v16, v16, v17, s7                        // 000000001C74: D5010010 001E2310
	global_store_d16_hi_b16 v[80:81], v16, off offset:46112    // 000000001C7C: EE09407C 08000000 00B42050
	s_wait_alu depctr_sa_sdst(0)                               // 000000001C88: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000001C8C: 8C7E087E
	s_and_saveexec_b32 s8, s2                                  // 000000001C90: BE882002
	s_cbranch_execz 64906                                      // 000000001C94: BFA5FD8A <gemm+0x12c0>
	v_bfe_u32 v16, v19, 16, 1                                  // 000000001C98: D6100010 02052113
	v_or_b32_e32 v17, 0x400000, v19                            // 000000001CA0: 382226FF 00400000
	v_cmp_u_f32_e64 s7, v19, v19                               // 000000001CA8: D4180007 00022713
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001CB0: BF8700A3
	v_add3_u32 v16, v16, v19, 0x7fff                           // 000000001CB4: D6550010 03FE2710 00007FFF
	s_wait_alu depctr_va_sdst(0)                               // 000000001CC0: BF88F19F
	v_cndmask_b32_e64 v16, v16, v17, s7                        // 000000001CC4: D5010010 001E2310
	global_store_d16_hi_b16 v[80:81], v16, off offset:48672    // 000000001CCC: EE09407C 08000000 00BE2050
	s_wait_alu depctr_sa_sdst(0)                               // 000000001CD8: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000001CDC: 8C7E087E
	s_and_saveexec_b32 s8, s3                                  // 000000001CE0: BE882003
	s_cbranch_execz 64890                                      // 000000001CE4: BFA5FD7A <gemm+0x12d0>
	v_bfe_u32 v16, v20, 16, 1                                  // 000000001CE8: D6100010 02052114
	v_or_b32_e32 v17, 0x400000, v20                            // 000000001CF0: 382228FF 00400000
	v_cmp_u_f32_e64 s7, v20, v20                               // 000000001CF8: D4180007 00022914
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001D00: BF8700A3
	v_add3_u32 v16, v16, v20, 0x7fff                           // 000000001D04: D6550010 03FE2910 00007FFF
	s_wait_alu depctr_va_sdst(0)                               // 000000001D10: BF88F19F
	v_cndmask_b32_e64 v16, v16, v17, s7                        // 000000001D14: D5010010 001E2310
	global_store_d16_hi_b16 v[80:81], v16, off offset:51232    // 000000001D1C: EE09407C 08000000 00C82050
	s_wait_alu depctr_sa_sdst(0)                               // 000000001D28: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000001D2C: 8C7E087E
	s_and_saveexec_b32 s8, s4                                  // 000000001D30: BE882004
	s_cbranch_execz 64874                                      // 000000001D34: BFA5FD6A <gemm+0x12e0>
	v_bfe_u32 v16, v21, 16, 1                                  // 000000001D38: D6100010 02052115
	v_or_b32_e32 v17, 0x400000, v21                            // 000000001D40: 38222AFF 00400000
	v_cmp_u_f32_e64 s7, v21, v21                               // 000000001D48: D4180007 00022B15
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001D50: BF8700A3
	v_add3_u32 v16, v16, v21, 0x7fff                           // 000000001D54: D6550010 03FE2B10 00007FFF
	s_wait_alu depctr_va_sdst(0)                               // 000000001D60: BF88F19F
	v_cndmask_b32_e64 v16, v16, v17, s7                        // 000000001D64: D5010010 001E2310
	global_store_d16_hi_b16 v[80:81], v16, off offset:53792    // 000000001D6C: EE09407C 08000000 00D22050
	s_wait_alu depctr_sa_sdst(0)                               // 000000001D78: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000001D7C: 8C7E087E
	s_and_saveexec_b32 s8, s5                                  // 000000001D80: BE882005
	s_cbranch_execz 64858                                      // 000000001D84: BFA5FD5A <gemm+0x12f0>
	v_bfe_u32 v16, v22, 16, 1                                  // 000000001D88: D6100010 02052116
	v_or_b32_e32 v17, 0x400000, v22                            // 000000001D90: 38222CFF 00400000
	v_cmp_u_f32_e64 s7, v22, v22                               // 000000001D98: D4180007 00022D16
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001DA0: BF8700A3
	v_add3_u32 v16, v16, v22, 0x7fff                           // 000000001DA4: D6550010 03FE2D10 00007FFF
	s_wait_alu depctr_va_sdst(0)                               // 000000001DB0: BF88F19F
	v_cndmask_b32_e64 v16, v16, v17, s7                        // 000000001DB4: D5010010 001E2310
	global_store_d16_hi_b16 v[80:81], v16, off offset:56352    // 000000001DBC: EE09407C 08000000 00DC2050
	s_wait_alu depctr_sa_sdst(0)                               // 000000001DC8: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000001DCC: 8C7E087E
	s_and_saveexec_b32 s8, s6                                  // 000000001DD0: BE882006
	s_cbranch_execz 64842                                      // 000000001DD4: BFA5FD4A <gemm+0x1300>
	v_bfe_u32 v16, v23, 16, 1                                  // 000000001DD8: D6100010 02052117
	v_or_b32_e32 v17, 0x400000, v23                            // 000000001DE0: 38222EFF 00400000
	v_cmp_u_f32_e64 s7, v23, v23                               // 000000001DE8: D4180007 00022F17
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001DF0: BF8700A3
	v_add3_u32 v16, v16, v23, 0x7fff                           // 000000001DF4: D6550010 03FE2F10 00007FFF
	s_wait_alu depctr_va_sdst(0)                               // 000000001E00: BF88F19F
	v_cndmask_b32_e64 v16, v16, v17, s7                        // 000000001E04: D5010010 001E2310
	global_store_d16_hi_b16 v[80:81], v16, off offset:58912    // 000000001E0C: EE09407C 08000000 00E62050
	s_wait_alu depctr_sa_sdst(0)                               // 000000001E18: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000001E1C: 8C7E087E
	s_and_saveexec_b32 s8, vcc_lo                              // 000000001E20: BE88206A
	s_cbranch_execz 64826                                      // 000000001E24: BFA5FD3A <gemm+0x1310>
	v_bfe_u32 v16, v8, 16, 1                                   // 000000001E28: D6100010 02052108
	v_or_b32_e32 v17, 0x400000, v8                             // 000000001E30: 382210FF 00400000
	v_cmp_u_f32_e64 s7, v8, v8                                 // 000000001E38: D4180007 00021108
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001E40: BF8700A3
	v_add3_u32 v16, v16, v8, 0x7fff                            // 000000001E44: D6550010 03FE1110 00007FFF
	s_wait_alu depctr_va_sdst(0)                               // 000000001E50: BF88F19F
	v_cndmask_b32_e64 v8, v16, v17, s7                         // 000000001E54: D5010008 001E2310
	global_store_d16_hi_b16 v[80:81], v8, off offset:41024     // 000000001E5C: EE09407C 04000000 00A04050
	s_wait_alu depctr_sa_sdst(0)                               // 000000001E68: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000001E6C: 8C7E087E
	s_and_saveexec_b32 s8, s0                                  // 000000001E70: BE882000
	s_cbranch_execz 64810                                      // 000000001E74: BFA5FD2A <gemm+0x1320>
	v_bfe_u32 v8, v9, 16, 1                                    // 000000001E78: D6100008 02052109
	v_or_b32_e32 v16, 0x400000, v9                             // 000000001E80: 382012FF 00400000
	v_cmp_u_f32_e64 s7, v9, v9                                 // 000000001E88: D4180007 00021309
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001E90: BF8700A3
	v_add3_u32 v8, v8, v9, 0x7fff                              // 000000001E94: D6550008 03FE1308 00007FFF
	s_wait_alu depctr_va_sdst(0)                               // 000000001EA0: BF88F19F
	v_cndmask_b32_e64 v8, v8, v16, s7                          // 000000001EA4: D5010008 001E2108
	global_store_d16_hi_b16 v[80:81], v8, off offset:43584     // 000000001EAC: EE09407C 04000000 00AA4050
	s_wait_alu depctr_sa_sdst(0)                               // 000000001EB8: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000001EBC: 8C7E087E
	s_and_saveexec_b32 s8, s1                                  // 000000001EC0: BE882001
	s_cbranch_execz 64794                                      // 000000001EC4: BFA5FD1A <gemm+0x1330>
	v_bfe_u32 v8, v10, 16, 1                                   // 000000001EC8: D6100008 0205210A
	v_or_b32_e32 v9, 0x400000, v10                             // 000000001ED0: 381214FF 00400000
	v_cmp_u_f32_e64 s7, v10, v10                               // 000000001ED8: D4180007 0002150A
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001EE0: BF8700A3
	v_add3_u32 v8, v8, v10, 0x7fff                             // 000000001EE4: D6550008 03FE1508 00007FFF
	s_wait_alu depctr_va_sdst(0)                               // 000000001EF0: BF88F19F
	v_cndmask_b32_e64 v8, v8, v9, s7                           // 000000001EF4: D5010008 001E1308
	global_store_d16_hi_b16 v[80:81], v8, off offset:46144     // 000000001EFC: EE09407C 04000000 00B44050
	s_wait_alu depctr_sa_sdst(0)                               // 000000001F08: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000001F0C: 8C7E087E
	s_and_saveexec_b32 s8, s2                                  // 000000001F10: BE882002
	s_cbranch_execz 64778                                      // 000000001F14: BFA5FD0A <gemm+0x1340>
	v_bfe_u32 v8, v11, 16, 1                                   // 000000001F18: D6100008 0205210B
	v_or_b32_e32 v9, 0x400000, v11                             // 000000001F20: 381216FF 00400000
	v_cmp_u_f32_e64 s7, v11, v11                               // 000000001F28: D4180007 0002170B
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001F30: BF8700A3
	v_add3_u32 v8, v8, v11, 0x7fff                             // 000000001F34: D6550008 03FE1708 00007FFF
	s_wait_alu depctr_va_sdst(0)                               // 000000001F40: BF88F19F
	v_cndmask_b32_e64 v8, v8, v9, s7                           // 000000001F44: D5010008 001E1308
	global_store_d16_hi_b16 v[80:81], v8, off offset:48704     // 000000001F4C: EE09407C 04000000 00BE4050
	s_wait_alu depctr_sa_sdst(0)                               // 000000001F58: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000001F5C: 8C7E087E
	s_and_saveexec_b32 s8, s3                                  // 000000001F60: BE882003
	s_cbranch_execz 64762                                      // 000000001F64: BFA5FCFA <gemm+0x1350>
	v_bfe_u32 v8, v12, 16, 1                                   // 000000001F68: D6100008 0205210C
	v_or_b32_e32 v9, 0x400000, v12                             // 000000001F70: 381218FF 00400000
	v_cmp_u_f32_e64 s7, v12, v12                               // 000000001F78: D4180007 0002190C
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001F80: BF8700A3
	v_add3_u32 v8, v8, v12, 0x7fff                             // 000000001F84: D6550008 03FE1908 00007FFF
	s_wait_alu depctr_va_sdst(0)                               // 000000001F90: BF88F19F
	v_cndmask_b32_e64 v8, v8, v9, s7                           // 000000001F94: D5010008 001E1308
	global_store_d16_hi_b16 v[80:81], v8, off offset:51264     // 000000001F9C: EE09407C 04000000 00C84050
	s_wait_alu depctr_sa_sdst(0)                               // 000000001FA8: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000001FAC: 8C7E087E
	s_and_saveexec_b32 s8, s4                                  // 000000001FB0: BE882004
	s_cbranch_execz 64746                                      // 000000001FB4: BFA5FCEA <gemm+0x1360>
	v_bfe_u32 v8, v13, 16, 1                                   // 000000001FB8: D6100008 0205210D
	v_or_b32_e32 v9, 0x400000, v13                             // 000000001FC0: 38121AFF 00400000
	v_cmp_u_f32_e64 s7, v13, v13                               // 000000001FC8: D4180007 00021B0D
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001FD0: BF8700A3
	v_add3_u32 v8, v8, v13, 0x7fff                             // 000000001FD4: D6550008 03FE1B08 00007FFF
	s_wait_alu depctr_va_sdst(0)                               // 000000001FE0: BF88F19F
	v_cndmask_b32_e64 v8, v8, v9, s7                           // 000000001FE4: D5010008 001E1308
	global_store_d16_hi_b16 v[80:81], v8, off offset:53824     // 000000001FEC: EE09407C 04000000 00D24050
	s_wait_alu depctr_sa_sdst(0)                               // 000000001FF8: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 000000001FFC: 8C7E087E
	s_and_saveexec_b32 s8, s5                                  // 000000002000: BE882005
	s_cbranch_execz 64730                                      // 000000002004: BFA5FCDA <gemm+0x1370>
	v_bfe_u32 v8, v14, 16, 1                                   // 000000002008: D6100008 0205210E
	v_or_b32_e32 v9, 0x400000, v14                             // 000000002010: 38121CFF 00400000
	v_cmp_u_f32_e64 s7, v14, v14                               // 000000002018: D4180007 00021D0E
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000002020: BF8700A3
	v_add3_u32 v8, v8, v14, 0x7fff                             // 000000002024: D6550008 03FE1D08 00007FFF
	s_wait_alu depctr_va_sdst(0)                               // 000000002030: BF88F19F
	v_cndmask_b32_e64 v8, v8, v9, s7                           // 000000002034: D5010008 001E1308
	global_store_d16_hi_b16 v[80:81], v8, off offset:56384     // 00000000203C: EE09407C 04000000 00DC4050
	s_wait_alu depctr_sa_sdst(0)                               // 000000002048: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 00000000204C: 8C7E087E
	s_and_saveexec_b32 s8, s6                                  // 000000002050: BE882006
	s_cbranch_execz 64714                                      // 000000002054: BFA5FCCA <gemm+0x1380>
	v_bfe_u32 v8, v15, 16, 1                                   // 000000002058: D6100008 0205210F
	v_or_b32_e32 v9, 0x400000, v15                             // 000000002060: 38121EFF 00400000
	v_cmp_u_f32_e64 s7, v15, v15                               // 000000002068: D4180007 00021F0F
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000002070: BF8700A3
	v_add3_u32 v8, v8, v15, 0x7fff                             // 000000002074: D6550008 03FE1F08 00007FFF
	s_wait_alu depctr_va_sdst(0)                               // 000000002080: BF88F19F
	v_cndmask_b32_e64 v8, v8, v9, s7                           // 000000002084: D5010008 001E1308
	global_store_d16_hi_b16 v[80:81], v8, off offset:58944     // 00000000208C: EE09407C 04000000 00E64050
	s_wait_alu depctr_sa_sdst(0)                               // 000000002098: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s8                              // 00000000209C: 8C7E087E
	s_and_saveexec_b32 s7, vcc_lo                              // 0000000020A0: BE87206A
	s_cbranch_execz 64698                                      // 0000000020A4: BFA5FCBA <gemm+0x1390>
	v_bfe_u32 v8, v0, 16, 1                                    // 0000000020A8: D6100008 02052100
	v_or_b32_e32 v9, 0x400000, v0                              // 0000000020B0: 381200FF 00400000
	v_cmp_u_f32_e32 vcc_lo, v0, v0                             // 0000000020B8: 7C300100
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000020BC: BF8700A3
	v_add3_u32 v8, v8, v0, 0x7fff                              // 0000000020C0: D6550008 03FE0108 00007FFF
	s_wait_alu depctr_va_vcc(0)                                // 0000000020CC: BF88FF9D
	v_cndmask_b32_e32 v0, v8, v9, vcc_lo                       // 0000000020D0: 02001308
	global_store_d16_hi_b16 v[80:81], v0, off offset:41056     // 0000000020D4: EE09407C 00000000 00A06050
	s_wait_alu depctr_sa_sdst(0)                               // 0000000020E0: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s7                              // 0000000020E4: 8C7E077E
	s_and_saveexec_b32 s7, s0                                  // 0000000020E8: BE872000
	s_cbranch_execz 64684                                      // 0000000020EC: BFA5FCAC <gemm+0x13a0>
	v_bfe_u32 v0, v1, 16, 1                                    // 0000000020F0: D6100000 02052101
	v_or_b32_e32 v8, 0x400000, v1                              // 0000000020F8: 381002FF 00400000
	v_cmp_u_f32_e32 vcc_lo, v1, v1                             // 000000002100: 7C300301
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000002104: BF8700A3
	v_add3_u32 v0, v0, v1, 0x7fff                              // 000000002108: D6550000 03FE0300 00007FFF
	s_wait_alu depctr_va_vcc(0)                                // 000000002114: BF88FF9D
	v_cndmask_b32_e32 v0, v0, v8, vcc_lo                       // 000000002118: 02001100
	global_store_d16_hi_b16 v[80:81], v0, off offset:43616     // 00000000211C: EE09407C 00000000 00AA6050
	s_wait_alu depctr_sa_sdst(0)                               // 000000002128: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s7                              // 00000000212C: 8C7E077E
	s_and_saveexec_b32 s0, s1                                  // 000000002130: BE802001
	s_cbranch_execz 64670                                      // 000000002134: BFA5FC9E <gemm+0x13b0>
	v_bfe_u32 v0, v2, 16, 1                                    // 000000002138: D6100000 02052102
	v_or_b32_e32 v1, 0x400000, v2                              // 000000002140: 380204FF 00400000
	v_cmp_u_f32_e32 vcc_lo, v2, v2                             // 000000002148: 7C300502
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 00000000214C: BF8700A3
	v_add3_u32 v0, v0, v2, 0x7fff                              // 000000002150: D6550000 03FE0500 00007FFF
	s_wait_alu depctr_va_vcc(0)                                // 00000000215C: BF88FF9D
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 000000002160: 02000300
	global_store_d16_hi_b16 v[80:81], v0, off offset:46176     // 000000002164: EE09407C 00000000 00B46050
	s_wait_alu depctr_sa_sdst(0)                               // 000000002170: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s0                              // 000000002174: 8C7E007E
	s_and_saveexec_b32 s0, s2                                  // 000000002178: BE802002
	s_cbranch_execz 64656                                      // 00000000217C: BFA5FC90 <gemm+0x13c0>
	v_bfe_u32 v0, v3, 16, 1                                    // 000000002180: D6100000 02052103
	v_or_b32_e32 v1, 0x400000, v3                              // 000000002188: 380206FF 00400000
	v_cmp_u_f32_e32 vcc_lo, v3, v3                             // 000000002190: 7C300703
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000002194: BF8700A3
	v_add3_u32 v0, v0, v3, 0x7fff                              // 000000002198: D6550000 03FE0700 00007FFF
	s_wait_alu depctr_va_vcc(0)                                // 0000000021A4: BF88FF9D
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 0000000021A8: 02000300
	global_store_d16_hi_b16 v[80:81], v0, off offset:48736     // 0000000021AC: EE09407C 00000000 00BE6050
	s_wait_alu depctr_sa_sdst(0)                               // 0000000021B8: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s0                              // 0000000021BC: 8C7E007E
	s_and_saveexec_b32 s0, s3                                  // 0000000021C0: BE802003
	s_cbranch_execz 64642                                      // 0000000021C4: BFA5FC82 <gemm+0x13d0>
	v_bfe_u32 v0, v4, 16, 1                                    // 0000000021C8: D6100000 02052104
	v_or_b32_e32 v1, 0x400000, v4                              // 0000000021D0: 380208FF 00400000
	v_cmp_u_f32_e32 vcc_lo, v4, v4                             // 0000000021D8: 7C300904
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000021DC: BF8700A3
	v_add3_u32 v0, v0, v4, 0x7fff                              // 0000000021E0: D6550000 03FE0900 00007FFF
	s_wait_alu depctr_va_vcc(0)                                // 0000000021EC: BF88FF9D
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 0000000021F0: 02000300
	global_store_d16_hi_b16 v[80:81], v0, off offset:51296     // 0000000021F4: EE09407C 00000000 00C86050
	s_wait_alu depctr_sa_sdst(0)                               // 000000002200: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s0                              // 000000002204: 8C7E007E
	s_and_saveexec_b32 s0, s4                                  // 000000002208: BE802004
	s_cbranch_execz 64628                                      // 00000000220C: BFA5FC74 <gemm+0x13e0>
	v_bfe_u32 v0, v5, 16, 1                                    // 000000002210: D6100000 02052105
	v_or_b32_e32 v1, 0x400000, v5                              // 000000002218: 38020AFF 00400000
	v_cmp_u_f32_e32 vcc_lo, v5, v5                             // 000000002220: 7C300B05
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000002224: BF8700A3
	v_add3_u32 v0, v0, v5, 0x7fff                              // 000000002228: D6550000 03FE0B00 00007FFF
	s_wait_alu depctr_va_vcc(0)                                // 000000002234: BF88FF9D
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 000000002238: 02000300
	global_store_d16_hi_b16 v[80:81], v0, off offset:53856     // 00000000223C: EE09407C 00000000 00D26050
	s_wait_alu depctr_sa_sdst(0)                               // 000000002248: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s0                              // 00000000224C: 8C7E007E
	s_and_saveexec_b32 s0, s5                                  // 000000002250: BE802005
	s_cbranch_execz 64614                                      // 000000002254: BFA5FC66 <gemm+0x13f0>
	v_bfe_u32 v0, v6, 16, 1                                    // 000000002258: D6100000 02052106
	v_or_b32_e32 v1, 0x400000, v6                              // 000000002260: 38020CFF 00400000
	v_cmp_u_f32_e32 vcc_lo, v6, v6                             // 000000002268: 7C300D06
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 00000000226C: BF8700A3
	v_add3_u32 v0, v0, v6, 0x7fff                              // 000000002270: D6550000 03FE0D00 00007FFF
	s_wait_alu depctr_va_vcc(0)                                // 00000000227C: BF88FF9D
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 000000002280: 02000300
	global_store_d16_hi_b16 v[80:81], v0, off offset:56416     // 000000002284: EE09407C 00000000 00DC6050
	s_wait_alu depctr_sa_sdst(0)                               // 000000002290: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s0                              // 000000002294: 8C7E007E
	s_and_saveexec_b32 s0, s6                                  // 000000002298: BE802006
	s_cbranch_execz 64600                                      // 00000000229C: BFA5FC58 <gemm+0x1400>
	v_bfe_u32 v0, v7, 16, 1                                    // 0000000022A0: D6100000 02052107
	v_or_b32_e32 v1, 0x400000, v7                              // 0000000022A8: 38020EFF 00400000
	v_cmp_u_f32_e32 vcc_lo, v7, v7                             // 0000000022B0: 7C300F07
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000022B4: BF8700A3
	v_add3_u32 v0, v0, v7, 0x7fff                              // 0000000022B8: D6550000 03FE0F00 00007FFF
	s_wait_alu depctr_va_vcc(0)                                // 0000000022C4: BF88FF9D
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 0000000022C8: 02000300
	global_store_d16_hi_b16 v[80:81], v0, off offset:58976     // 0000000022CC: EE09407C 00000000 00E66050
	s_nop 0                                                    // 0000000022D8: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000022DC: BFB60003
	s_endpgm                                                   // 0000000022E0: BFB00000
	s_code_end                                                 // 0000000022E4: BF9F0000
	s_code_end                                                 // 0000000022E8: BF9F0000
	s_code_end                                                 // 0000000022EC: BF9F0000
	s_code_end                                                 // 0000000022F0: BF9F0000
	s_code_end                                                 // 0000000022F4: BF9F0000
	s_code_end                                                 // 0000000022F8: BF9F0000
	s_code_end                                                 // 0000000022FC: BF9F0000
	s_code_end                                                 // 000000002300: BF9F0000
	s_code_end                                                 // 000000002304: BF9F0000
	s_code_end                                                 // 000000002308: BF9F0000
	s_code_end                                                 // 00000000230C: BF9F0000
	s_code_end                                                 // 000000002310: BF9F0000
	s_code_end                                                 // 000000002314: BF9F0000
	s_code_end                                                 // 000000002318: BF9F0000
	s_code_end                                                 // 00000000231C: BF9F0000
	s_code_end                                                 // 000000002320: BF9F0000
	s_code_end                                                 // 000000002324: BF9F0000
	s_code_end                                                 // 000000002328: BF9F0000
	s_code_end                                                 // 00000000232C: BF9F0000
	s_code_end                                                 // 000000002330: BF9F0000
	s_code_end                                                 // 000000002334: BF9F0000
	s_code_end                                                 // 000000002338: BF9F0000
	s_code_end                                                 // 00000000233C: BF9F0000
	s_code_end                                                 // 000000002340: BF9F0000
	s_code_end                                                 // 000000002344: BF9F0000
	s_code_end                                                 // 000000002348: BF9F0000
	s_code_end                                                 // 00000000234C: BF9F0000
	s_code_end                                                 // 000000002350: BF9F0000
	s_code_end                                                 // 000000002354: BF9F0000
	s_code_end                                                 // 000000002358: BF9F0000
	s_code_end                                                 // 00000000235C: BF9F0000
	s_code_end                                                 // 000000002360: BF9F0000
	s_code_end                                                 // 000000002364: BF9F0000
	s_code_end                                                 // 000000002368: BF9F0000
	s_code_end                                                 // 00000000236C: BF9F0000
	s_code_end                                                 // 000000002370: BF9F0000
	s_code_end                                                 // 000000002374: BF9F0000
	s_code_end                                                 // 000000002378: BF9F0000
	s_code_end                                                 // 00000000237C: BF9F0000
	s_code_end                                                 // 000000002380: BF9F0000
	s_code_end                                                 // 000000002384: BF9F0000
	s_code_end                                                 // 000000002388: BF9F0000
	s_code_end                                                 // 00000000238C: BF9F0000
	s_code_end                                                 // 000000002390: BF9F0000
	s_code_end                                                 // 000000002394: BF9F0000
	s_code_end                                                 // 000000002398: BF9F0000
	s_code_end                                                 // 00000000239C: BF9F0000
	s_code_end                                                 // 0000000023A0: BF9F0000
	s_code_end                                                 // 0000000023A4: BF9F0000
	s_code_end                                                 // 0000000023A8: BF9F0000
	s_code_end                                                 // 0000000023AC: BF9F0000
	s_code_end                                                 // 0000000023B0: BF9F0000
	s_code_end                                                 // 0000000023B4: BF9F0000
	s_code_end                                                 // 0000000023B8: BF9F0000
	s_code_end                                                 // 0000000023BC: BF9F0000
	s_code_end                                                 // 0000000023C0: BF9F0000
	s_code_end                                                 // 0000000023C4: BF9F0000
	s_code_end                                                 // 0000000023C8: BF9F0000
	s_code_end                                                 // 0000000023CC: BF9F0000
	s_code_end                                                 // 0000000023D0: BF9F0000
	s_code_end                                                 // 0000000023D4: BF9F0000
	s_code_end                                                 // 0000000023D8: BF9F0000
	s_code_end                                                 // 0000000023DC: BF9F0000
	s_code_end                                                 // 0000000023E0: BF9F0000
	s_code_end                                                 // 0000000023E4: BF9F0000
	s_code_end                                                 // 0000000023E8: BF9F0000
	s_code_end                                                 // 0000000023EC: BF9F0000
	s_code_end                                                 // 0000000023F0: BF9F0000
	s_code_end                                                 // 0000000023F4: BF9F0000
	s_code_end                                                 // 0000000023F8: BF9F0000
	s_code_end                                                 // 0000000023FC: BF9F0000
	s_code_end                                                 // 000000002400: BF9F0000
	s_code_end                                                 // 000000002404: BF9F0000
	s_code_end                                                 // 000000002408: BF9F0000
	s_code_end                                                 // 00000000240C: BF9F0000
	s_code_end                                                 // 000000002410: BF9F0000
	s_code_end                                                 // 000000002414: BF9F0000
	s_code_end                                                 // 000000002418: BF9F0000
	s_code_end                                                 // 00000000241C: BF9F0000
	s_code_end                                                 // 000000002420: BF9F0000
	s_code_end                                                 // 000000002424: BF9F0000
	s_code_end                                                 // 000000002428: BF9F0000
	s_code_end                                                 // 00000000242C: BF9F0000
	s_code_end                                                 // 000000002430: BF9F0000
	s_code_end                                                 // 000000002434: BF9F0000
	s_code_end                                                 // 000000002438: BF9F0000
	s_code_end                                                 // 00000000243C: BF9F0000
	s_code_end                                                 // 000000002440: BF9F0000
	s_code_end                                                 // 000000002444: BF9F0000
	s_code_end                                                 // 000000002448: BF9F0000
	s_code_end                                                 // 00000000244C: BF9F0000
	s_code_end                                                 // 000000002450: BF9F0000
	s_code_end                                                 // 000000002454: BF9F0000
	s_code_end                                                 // 000000002458: BF9F0000
	s_code_end                                                 // 00000000245C: BF9F0000
	s_code_end                                                 // 000000002460: BF9F0000
	s_code_end                                                 // 000000002464: BF9F0000
	s_code_end                                                 // 000000002468: BF9F0000
	s_code_end                                                 // 00000000246C: BF9F0000
	s_code_end                                                 // 000000002470: BF9F0000
	s_code_end                                                 // 000000002474: BF9F0000
	s_code_end                                                 // 000000002478: BF9F0000
	s_code_end                                                 // 00000000247C: BF9F0000
