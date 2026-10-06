---
sidebar_label: AM Driver
---

# AM Driver (Userspace)

**AM** `device/src/amd/am/` के अंतर्गत एक experimental, अभी तक न चुना जा सकने वाला userspace
driver है जो `amdgpu`/KFD से होकर जाने के बजाय AMD GPU के PCI BARs से सीधे बात करता है। यह
भावना में tinygrad के AM driver का अनुसरण करता है, लेकिन एक अलग प्रकार के hardware को target
करता है (bare metal के बजाय एक virtual function)। यह **scaffolding** है: pure-logic हिस्से
implement और unit-tested हैं, bring-up एक बार एक live GPU पर परखा गया, और इसके माध्यम से कभी
कोई कर्नेल execute नहीं हुआ।

:::caution[न चुना जा सकता है, न चलाया जा सकता है]
`SVOD_AMD_BACKEND=am` reject होता है (`unknown SVOD_AMD_BACKEND=am (only 'kfd'
supported)`, `device/src/amd/device.rs`): `am/` में कुछ भी [`AmdIface`](./overview.md) seam
implement नहीं करता, इसलिए `AmdDevice` इसका उपयोग नहीं कर सकता। code केवल standalone
`device/examples/am_*.rs` programs के माध्यम से पहुँचा जा सकता है। इसका अंतिम functional
बदलाव जून 2026 का है; बाद के commits cleanups हैं। नीचे के स्थिति-कथन बताते हैं कि उस commit
ने क्या परखा, कोई चालू गारंटी नहीं — इसके लिए कोई CI या hardware test नहीं है।
:::

module हर Unix host पर compile होता है (`cfg(unix)`, बाकी बैकएंड की तरह), इसलिए यह हमेशा
type-checked, linted होता है, और इसका logic unit-tested है (`device/src/test/unit/amd/am/`
के अंतर्गत लगभग 40 tests: page tables, TLSF, memory manager, register tables, discovery
parser)।

---

## Target: एक CDNA3 SR-IOV VF (gfx9.4.x)

`AmDev::open` केवल ऐसे GPU का **SR-IOV virtual function** स्वीकार करता है जिसका GC IP version
9.4 (CDNA3) हो — bring-up hardware एक KVM guest में pass किया गया MI300-class VF था। non-VF
function या कोई दूसरा GC version reject होता है (`device/src/amd/am/dev.rs`)। gfx11
page-table encoding implemented और unit-tested बनी हुई है, और इसके geometry व
physical-range helpers वही हैं जिन्हें gfx9 path reuse करता है।

bare metal के बजाय VF होना पूरे driver को आकार देता है:

- **GC MMIO host-gated है।** GC register का direct read `0xffffffff` लौटाता है; GC / GCVM
  registers **RLC के माध्यम से indirectly** जाते हैं (value को RLC scratch में stage करें,
  `RLC_SPARE_INT` kick करें, poll करें)।
- **VRAM और IP discovery grant होने तक gated हैं।** host **GIM** (SR-IOV host driver) को एक
  **mailbox handshake** के माध्यम से access grant करना होता है, जो discovery से पहले चलता है।
- **host PF privileged subsystems का स्वामी है:** PSP, SMU, clocks, firmware / world-switch,
  और **doorbell aperture routing**। AM प्रति-VF state program करता है (page-table context0,
  प्रति-engine invalidation ranges, TLB flushes, ring/queue MQDs) और कुछ PF-owned registers
  (L2 cache config, system और identity apertures, `GB_ADDR_CONFIG`, `RLC_CNTL`,
  `SH_MEM_BASES`) best-effort लिखता है, rejections को ignore करते हुए।

tinygrad का AM इसका उल्टा है: केवल bare-metal, `amdgpu` को unbind करके पूरे डिवाइस का स्वामी
बनता है। VF प्रकार को mailbox, RLCG indirect path और केवल प्रति-VF hub programming चाहिए, और
यह कभी engines का स्वामी नहीं बन पाता।

---

## क्या मौजूद है

| समूह | Module(s) | क्या करता है | स्थिति |
|---|---|---|---|
| Discovery | `pci.rs`, `discovery.rs` | sysfs BAR mmap (BAR0 VRAM / BAR2 doorbell / BAR5 MMIO), config-space r/w, bounds-checked IP-discovery parser (प्रति-XCC segment bases, `gc_info` v1/v2) | VF पर चला; parser unit-tested |
| Register access | `regaccess.rs`, `rlcg.rs`, `mailbox.rs`, `regs.rs`, `regs_gen.rs` | VF↔GIM mailbox handshake, प्रति XCC RLCG indirect GC/GCVM r/w, MMIO/RLCG router, `select` / `find` / `encode` के साथ vendored register tables | VF पर चला (scratch echo, हर XCC पर `GRBM_STATUS`); table logic unit-tested |
| Memory (GMMU) | `mm/{tlsf,pagetable,manager,mod}.rs` | VA, physical VRAM और page-table pool के लिए TLSF allocators; 4-level / 48-bit walk (`va_shifts = [12, 21, 30, 39]`); gfx9 और gfx11 PTE/PDE encoding; huge pages; table reclaim; `valloc` / `vfree` | unit-tested; page tables BAR0 पर VRAM में लिखी गईं और CPU द्वारा वापस walk की गईं — उनके माध्यम से कोई GPU translation पुष्ट नहीं है |
| GMC bring-up | `ip/gmc.rs` | दोनों hubs का context0 (base/start/end + CNTL), MX_L1_TLB, प्रति-engine invalidation ranges, ENG17 TLB flush, HDP flush, raw fault-status read | VF पर context programming तक चला, हर XCC पर flush ACK के साथ |
| GFX bring-up | `ip/gfx.rs` | MEC enable (unchecked writes), v9 compute MQD, HQD activation, `WRITE_DATA` PM4 | `CP_HQD_ACTIVE` 1 पढ़ता है; queue ने कभी कोई packet consume नहीं किया |
| SDMA bring-up | `ip/sdma.rs` | F32 को unhalt करना, RB base/rptr/wptr + doorbell program करना, submit, `wait_idle` | programmed; कोई copy कभी पूरी नहीं हुई |
| Orchestrator | `dev.rs` | `AmDev::open` = mailbox → discovery → GMMU → GMC context0 → flush; `valloc`, `vram_read` / `vram_write`, `release` | VF पर GMC तक चला |

Page tables एक injectable `PhysMem` trait पर आधारित हैं — tests में एक plain buffer, driver
में BAR-mapped VRAM (`VramPhys`)। leaf encoding ही एकमात्र arch-specific हिस्सा है: gfx9
MTYPE को bit 57 पर रखता है, PDB1 table entries पर `bfs` और PDB0 table entries पर
translate-further set करता है, और PDB1/PDB2 leaves को `PDE_PTE` mark करता है; gfx12
`unimplemented!` है (constants captured; एक test panic को assert करता है)।

### Register tables एक बार generate होती हैं, फिर vendored

tinygrad एक कभी-कभी अनुपस्थित submodule है, इसलिए build कभी उस पर निर्भर नहीं करता।
`device/tools/gen_am_regs.py` हाथ से चलाया जाता है: यह tinygrad का `autogen/am/regs.py`
parse करता है और committed `am/regs_gen.rs` emit करता है। boot पर `select` उसी major वाला
सबसे बड़ा table version `≤ ip_ver` चुनता है। committed tables gfx9.4.3 set (`gc_9_4_3`,
`mmhub_1_8_0`, `osssys_4_4_2`, `sdma_4_4_2`, `nbio_7_9_0`, `hdp_4_4_2`, `mp_11_0_0`,
`mp_13_0_0`) और gfx11.5.0 set (`gc_11_5_0`, `mmhub_3_3_0`, `mp_14_0_2`, `nbio_7_11_0`,
`hdp_6_0_0`, `osssys_6_0_0`) को cover करती हैं; gfx11 GC table वही है जिसे KFD path के
hardware counters उपयोग करते हैं (`amd/pmc.rs`)।

---

## Examples

हर `device/examples/am_*.rs` program एक standalone bring-up oracle है। जून 2026 के run ने
क्या स्थापित किया:

| Example | क्या करता है | परिणाम |
|---|---|---|
| `am_discovery` | BAR map + IP discovery, read-only; bound `amdgpu` के साथ coexist करता है | 8 GC 9.4.3 instances, SDMA और AIDs enumerated |
| `am_own` | mailbox grant + RLCG scratch echo + हर XCC पर `GRBM_STATUS` | asserts pass |
| `am_gmc` | GC + MM context0 programmed; हर XCC पर ENG17 TLB-flush ACK; fault status printed | हर XCC पर ACKs |
| `am_sdma` | SDMA ring setup + उसके माध्यम से एक copy | engine ring consume नहीं करता |
| `am_compute` | MEC enable + MQD activate + `WRITE_DATA`, doorbell और direct `CP_HQD_PQ_WPTR` write दोनों से kicked | HQD activate होता है; sentinel कभी नहीं पहुँचता |

दीवार engine hand-off है: doorbell aperture routing और engine boot host PF के स्वामित्व में
हैं। VF से aperture enable करना (`_PF` BIF doorbell registers) VF↔GIM mailbox को अटका देता है
और VM reboot की ज़रूरत पड़ती है, इसलिए `enable_doorbell_aperture` `ip/gfx.rs` में मौजूद है पर
VF पर do-not-call mark है और `am_compute` में commented out है।

---

## क्या मौजूद नहीं है

- **एक `AmdIface` implementation** — इसलिए AM device बैकएंड नहीं बन सकता।
- **PSP firmware load**, **SMU / clocks** — VF पर GIM के स्वामित्व में; bare metal पर ये
  सबसे बड़ा और सबसे जोखिम भरा port होंगे।
- **एक interrupt handler** — कोई `ip/ih.rs` नहीं है; OSSSYS table केवल `am_discovery`
  उपयोग करता है। Bring-up poll करता है।
- **प्रमाण कि कोई GPU engine AM की page tables के माध्यम से काम execute करता है।**

दो debug knobs केवल GMC bring-up के लिए मौजूद हैं: `SVOD_AM_DEBUG` (कोई भी value) registers
लिखने के बाद उन्हें वापस पढ़ता है और rejected GC writes log करता है, और `SVOD_AM_MCBASE`
(`raw`, `fb` या `fbxgmi`) MC aperture base को override करता है।

---

## यह यहाँ क्यों है

प्रेरणा kernel scheduler है: single-XCC gfx11+ parts पर, aggressive multi-queue dispatch CP
micro-engines को ऐसे waits में अटका सकता है जिन्हें MES firmware preempt नहीं कर सकता, यही
कारण है कि [lane pool](./queues-and-dispatch.md) conservative रहता है। GPU का स्वामी बनना
kernel को dispatch path से बाहर कर देगा। वह तर्क अभी उस पर लागू नहीं होता जिसे AM support करता
है — VF पर host अब भी scheduling, world-switch और doorbells का स्वामी है — और bare metal पर
वहाँ पहुँचने का अर्थ है मौजूदा चीज़ों के ऊपर PSP, SMU, interrupts और seam implementation। आज
design का मूल्य ख़ुद seam है: यदि कभी कोई AM बैकएंड आता है, तो वह `KfdIface` जैसे ही पाँच
methods और तीन hooks implement करेगा, और seam के ऊपर कुछ नहीं बदलेगा।
