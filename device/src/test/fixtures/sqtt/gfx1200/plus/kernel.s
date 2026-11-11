
kernel.co:	file format elf64-amdgpu
	.amdgcn_target "amdgpu-amd-amdhsa-unknown-gfx1201"

Disassembly of section .text:

0000000000001600 <E_3>:
	s_clause 0x1                                               // 000000001600: BF850001
	s_load_b128 s[4:7], s[0:1], 0x0                            // 000000001604: F4004100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000160C: F4002000 F8000010
	v_lshlrev_b32_e32 v0, 2, v0                                // 000000001614: 30000082
	s_wait_kmcnt 0x0                                           // 000000001618: BFC70000
	s_clause 0x1                                               // 00000000161C: BF850001
	global_load_b32 v1, v0, s[6:7]                             // 000000001620: EE050006 00000001 00000000
	global_load_b32 v2, v0, s[0:1]                             // 00000000162C: EE050000 00000002 00000000
	s_wait_loadcnt 0x0                                         // 000000001638: BFC00000
	v_add_f32_e32 v1, v1, v2                                   // 00000000163C: 06020501
	global_store_b32 v0, v1, s[4:5]                            // 000000001640: EE068004 00800000 00000000
	s_nop 0                                                    // 00000000164C: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001650: BFB60003
	s_endpgm                                                   // 000000001654: BFB00000
	s_code_end                                                 // 000000001658: BF9F0000
	s_code_end                                                 // 00000000165C: BF9F0000
	s_code_end                                                 // 000000001660: BF9F0000
	s_code_end                                                 // 000000001664: BF9F0000
	s_code_end                                                 // 000000001668: BF9F0000
	s_code_end                                                 // 00000000166C: BF9F0000
	s_code_end                                                 // 000000001670: BF9F0000
	s_code_end                                                 // 000000001674: BF9F0000
	s_code_end                                                 // 000000001678: BF9F0000
	s_code_end                                                 // 00000000167C: BF9F0000
	s_code_end                                                 // 000000001680: BF9F0000
	s_code_end                                                 // 000000001684: BF9F0000
	s_code_end                                                 // 000000001688: BF9F0000
	s_code_end                                                 // 00000000168C: BF9F0000
	s_code_end                                                 // 000000001690: BF9F0000
	s_code_end                                                 // 000000001694: BF9F0000
	s_code_end                                                 // 000000001698: BF9F0000
	s_code_end                                                 // 00000000169C: BF9F0000
	s_code_end                                                 // 0000000016A0: BF9F0000
	s_code_end                                                 // 0000000016A4: BF9F0000
	s_code_end                                                 // 0000000016A8: BF9F0000
	s_code_end                                                 // 0000000016AC: BF9F0000
	s_code_end                                                 // 0000000016B0: BF9F0000
	s_code_end                                                 // 0000000016B4: BF9F0000
	s_code_end                                                 // 0000000016B8: BF9F0000
	s_code_end                                                 // 0000000016BC: BF9F0000
	s_code_end                                                 // 0000000016C0: BF9F0000
	s_code_end                                                 // 0000000016C4: BF9F0000
	s_code_end                                                 // 0000000016C8: BF9F0000
	s_code_end                                                 // 0000000016CC: BF9F0000
	s_code_end                                                 // 0000000016D0: BF9F0000
	s_code_end                                                 // 0000000016D4: BF9F0000
	s_code_end                                                 // 0000000016D8: BF9F0000
	s_code_end                                                 // 0000000016DC: BF9F0000
	s_code_end                                                 // 0000000016E0: BF9F0000
	s_code_end                                                 // 0000000016E4: BF9F0000
	s_code_end                                                 // 0000000016E8: BF9F0000
	s_code_end                                                 // 0000000016EC: BF9F0000
	s_code_end                                                 // 0000000016F0: BF9F0000
	s_code_end                                                 // 0000000016F4: BF9F0000
	s_code_end                                                 // 0000000016F8: BF9F0000
	s_code_end                                                 // 0000000016FC: BF9F0000
	s_code_end                                                 // 000000001700: BF9F0000
	s_code_end                                                 // 000000001704: BF9F0000
	s_code_end                                                 // 000000001708: BF9F0000
	s_code_end                                                 // 00000000170C: BF9F0000
	s_code_end                                                 // 000000001710: BF9F0000
	s_code_end                                                 // 000000001714: BF9F0000
	s_code_end                                                 // 000000001718: BF9F0000
	s_code_end                                                 // 00000000171C: BF9F0000
	s_code_end                                                 // 000000001720: BF9F0000
	s_code_end                                                 // 000000001724: BF9F0000
	s_code_end                                                 // 000000001728: BF9F0000
	s_code_end                                                 // 00000000172C: BF9F0000
	s_code_end                                                 // 000000001730: BF9F0000
	s_code_end                                                 // 000000001734: BF9F0000
	s_code_end                                                 // 000000001738: BF9F0000
	s_code_end                                                 // 00000000173C: BF9F0000
	s_code_end                                                 // 000000001740: BF9F0000
	s_code_end                                                 // 000000001744: BF9F0000
	s_code_end                                                 // 000000001748: BF9F0000
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
