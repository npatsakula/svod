
code_object.out:	file format elf64-amdgpu
	.amdgcn_target "amdgpu-amd-amdhsa-unknown-gfx1200"

Disassembly of section .text:

0000000000001600 <_Z11fifo_kernelv>:
	global_wb scope:SCOPE_SE                                   // 000000001600: EE0B007C 00040000 00000000
	s_barrier_signal -1                                        // 00000000160C: BE804EC1
	s_barrier_wait 0xffff                                      // 000000001610: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000001614: EE0AC07C 00040000 00000000
	ds_load_b128 v[1:4], v0                                    // 000000001620: DBFC0000 01000000
	ds_load_b128 v[1:4], v0                                    // 000000001628: DBFC0000 01000000
	ds_load_b128 v[1:4], v0                                    // 000000001630: DBFC0000 01000000
	ds_load_b128 v[1:4], v0                                    // 000000001638: DBFC0000 01000000
	ds_load_b128 v[1:4], v0                                    // 000000001640: DBFC0000 01000000
	ds_load_b128 v[1:4], v0                                    // 000000001648: DBFC0000 01000000
	ds_load_b128 v[1:4], v0                                    // 000000001650: DBFC0000 01000000
	ds_load_b128 v[1:4], v0                                    // 000000001658: DBFC0000 01000000
	ds_load_b128 v[1:4], v0                                    // 000000001660: DBFC0000 01000000
	ds_load_b128 v[1:4], v0                                    // 000000001668: DBFC0000 01000000
	ds_load_b128 v[1:4], v0                                    // 000000001670: DBFC0000 01000000
	ds_load_b128 v[1:4], v0                                    // 000000001678: DBFC0000 01000000
	ds_load_b128 v[1:4], v0                                    // 000000001680: DBFC0000 01000000
	ds_load_b128 v[1:4], v0                                    // 000000001688: DBFC0000 01000000
	ds_load_b128 v[1:4], v0                                    // 000000001690: DBFC0000 01000000
	ds_load_b128 v[1:4], v0                                    // 000000001698: DBFC0000 01000000
	ds_load_b128 v[1:4], v0                                    // 0000000016A0: DBFC0000 01000000
	ds_load_b128 v[1:4], v0                                    // 0000000016A8: DBFC0000 01000000
	ds_load_b128 v[1:4], v0                                    // 0000000016B0: DBFC0000 01000000
	ds_load_b128 v[1:4], v0                                    // 0000000016B8: DBFC0000 01000000
	ds_load_b128 v[1:4], v0                                    // 0000000016C0: DBFC0000 01000000
	ds_load_b128 v[1:4], v0                                    // 0000000016C8: DBFC0000 01000000
	ds_load_b128 v[1:4], v0                                    // 0000000016D0: DBFC0000 01000000
	ds_load_b128 v[1:4], v0                                    // 0000000016D8: DBFC0000 01000000
	ds_load_b128 v[1:4], v0                                    // 0000000016E0: DBFC0000 01000000
	ds_load_b128 v[1:4], v0                                    // 0000000016E8: DBFC0000 01000000
	ds_load_b128 v[1:4], v0                                    // 0000000016F0: DBFC0000 01000000
	ds_load_b128 v[1:4], v0                                    // 0000000016F8: DBFC0000 01000000
	ds_load_b128 v[1:4], v0                                    // 000000001700: DBFC0000 01000000
	ds_load_b128 v[1:4], v0                                    // 000000001708: DBFC0000 01000000
	ds_load_b128 v[1:4], v0                                    // 000000001710: DBFC0000 01000000
	ds_load_b128 v[0:3], v0                                    // 000000001718: DBFC0000 00000000
	global_wb scope:SCOPE_SE                                   // 000000001720: EE0B007C 00040000 00000000
	s_barrier_signal -1                                        // 00000000172C: BE804EC1
	s_barrier_wait 0xffff                                      // 000000001730: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000001734: EE0AC07C 00040000 00000000
	v_add_f32_e64 v0, v0, 1                                    // 000000001740: D5030000 00010300
	s_endpgm                                                   // 000000001748: BFB00000
	s_code_end                                                 // 00000000174C: BF9F0000
	s_code_end                                                 // 000000001750: BF9F0000
	s_code_end                                                 // 000000001754: BF9F0000
	s_code_end                                                 // 000000001758: BF9F0000
	s_code_end                                                 // 00000000175C: BF9F0000
	s_code_end                                                 // 000000001760: BF9F0000
	s_code_end                                                 // 000000001764: BF9F0000
	s_code_end                                                 // 000000001768: BF9F0000
	s_code_end                                                 // 00000000176C: BF9F0000
	s_code_end                                                 // 000000001770: BF9F0000
	s_code_end                                                 // 000000001774: BF9F0000
	s_code_end                                                 // 000000001778: BF9F0000
	s_code_end                                                 // 00000000177C: BF9F0000
	s_code_end                                                 // 000000001780: BF9F0000
	s_code_end                                                 // 000000001784: BF9F0000
	s_code_end                                                 // 000000001788: BF9F0000
	s_code_end                                                 // 00000000178C: BF9F0000
	s_code_end                                                 // 000000001790: BF9F0000
	s_code_end                                                 // 000000001794: BF9F0000
	s_code_end                                                 // 000000001798: BF9F0000
	s_code_end                                                 // 00000000179C: BF9F0000
	s_code_end                                                 // 0000000017A0: BF9F0000
	s_code_end                                                 // 0000000017A4: BF9F0000
	s_code_end                                                 // 0000000017A8: BF9F0000
	s_code_end                                                 // 0000000017AC: BF9F0000
	s_code_end                                                 // 0000000017B0: BF9F0000
	s_code_end                                                 // 0000000017B4: BF9F0000
	s_code_end                                                 // 0000000017B8: BF9F0000
	s_code_end                                                 // 0000000017BC: BF9F0000
	s_code_end                                                 // 0000000017C0: BF9F0000
	s_code_end                                                 // 0000000017C4: BF9F0000
	s_code_end                                                 // 0000000017C8: BF9F0000
	s_code_end                                                 // 0000000017CC: BF9F0000
	s_code_end                                                 // 0000000017D0: BF9F0000
	s_code_end                                                 // 0000000017D4: BF9F0000
	s_code_end                                                 // 0000000017D8: BF9F0000
	s_code_end                                                 // 0000000017DC: BF9F0000
	s_code_end                                                 // 0000000017E0: BF9F0000
	s_code_end                                                 // 0000000017E4: BF9F0000
	s_code_end                                                 // 0000000017E8: BF9F0000
	s_code_end                                                 // 0000000017EC: BF9F0000
	s_code_end                                                 // 0000000017F0: BF9F0000
	s_code_end                                                 // 0000000017F4: BF9F0000
	s_code_end                                                 // 0000000017F8: BF9F0000
	s_code_end                                                 // 0000000017FC: BF9F0000
	s_code_end                                                 // 000000001800: BF9F0000
	s_code_end                                                 // 000000001804: BF9F0000
	s_code_end                                                 // 000000001808: BF9F0000
	s_code_end                                                 // 00000000180C: BF9F0000
	s_code_end                                                 // 000000001810: BF9F0000
	s_code_end                                                 // 000000001814: BF9F0000
	s_code_end                                                 // 000000001818: BF9F0000
	s_code_end                                                 // 00000000181C: BF9F0000
	s_code_end                                                 // 000000001820: BF9F0000
	s_code_end                                                 // 000000001824: BF9F0000
	s_code_end                                                 // 000000001828: BF9F0000
	s_code_end                                                 // 00000000182C: BF9F0000
	s_code_end                                                 // 000000001830: BF9F0000
	s_code_end                                                 // 000000001834: BF9F0000
	s_code_end                                                 // 000000001838: BF9F0000
	s_code_end                                                 // 00000000183C: BF9F0000
	s_code_end                                                 // 000000001840: BF9F0000
	s_code_end                                                 // 000000001844: BF9F0000
	s_code_end                                                 // 000000001848: BF9F0000
	s_code_end                                                 // 00000000184C: BF9F0000
	s_code_end                                                 // 000000001850: BF9F0000
	s_code_end                                                 // 000000001854: BF9F0000
	s_code_end                                                 // 000000001858: BF9F0000
	s_code_end                                                 // 00000000185C: BF9F0000
	s_code_end                                                 // 000000001860: BF9F0000
	s_code_end                                                 // 000000001864: BF9F0000
	s_code_end                                                 // 000000001868: BF9F0000
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
