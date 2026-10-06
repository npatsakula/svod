---
sidebar_label: Debugging
---

# Debugging और Fault Triage

जब एक GPU kernel ऐसी memory छूता है जो उसे नहीं छूनी चाहिए, KFD एक raw virtual address और बहुत
कम के साथ एक fault report करता है। यह पेज उन tools को कवर करता है जो बैकएंड के पास उसे एक
diagnosis में बदलने के लिए हैं: वह VA→allocation registry जो faulting address को classify
करता है, वह poison latch जो device को साफ़-सुथरे ढंग से रोक देता है, और dispatch/tracing
instrumentation।

---

## समस्या: एक नंगी faulting VA

एक `WAIT_EVENTS` memory-fault event एक `kfd_hsa_memory_exception_data` वापस सौंपता है जिसमें
faulting `va`, failure flags (`NotPresent`, `ReadOnly`, `NoExecute`, `imprecise`), और एक
`ErrorType` होती है। यह आपको बताता है कि GPU ने *कहाँ* fault किया लेकिन यह नहीं कि *वहाँ क्या
था* — और यही वह सवाल है जो असल में bug को localize करता है। जो panic आख़िरकार इसे सामने लाता
है वह अगले `synchronize()` पर एक delayed re-throw है, fault site से बहुत दूर।

---

## VA registry

`device/src/amd/va_registry.rs` एक diagnostic side-table है जो हर live GPU VA range को वापस
उसके मालिक allocation से map करता है। यह विशुद्ध bookkeeping है — कोई GPU dependency नहीं —
इसलिए classification logic को किसी भी host पर unit- और property-test किया जाता है। एक
`VaRegistry` `KfdIface` पर रहता है (एक fault पूरे VM को corrupt कर देता है, इसलिए per-device
सही granularity है)।

इसे एक allocation के जीवन के दोनों सिरों पर बनाए रखा जाता है:

- **`alloc_raw`** `MAP_MEMORY_TO_GPU` सफल होने के बाद `va.insert(base, size, handle, tag)`
  call करता है।
- **`free_raw`** unmap से *पहले* `va.remove(base)` call करता है — ताकि एक fault जो एक
  अभी-अभी-freed VA पर land करे वह एक live allocation के बजाय use-after-free के रूप में
  classify हो।

### Tags

हर allocation को उसके उद्देश्य (`AllocTag`) से tag किया जाता है। `Vram` और `Gtt` वे defaults
हैं जो `AllocKind` से derive होते हैं; बारीक tags `alloc_*_tagged` call sites द्वारा explicit
रूप से pass किए जाते हैं:

| Tag | किसे cover करता है |
|---|---|
| `Vram` | General device VRAM — tensor data, code objects, EOP/ctx-save |
| `Gtt` | Host-visible control memory — GTT, और graph / linked-plan IB तथा control buffers |
| `Kernarg` | Kernarg arenas (host-visible VRAM) — per-dispatch, graph, और linked-plan argument pages |
| `SignalPool` | GTT signal-slot pool |
| `QueueRing` / `QueueGart` / `QueueInactive` | एक queue का ring, GART page, और queue-inactive signal |
| `Staging` | GTT SDMA bounce buffer |
| `Scratch` | Register-spill scratch — GPU-only VRAM, प्रति kernel realloc'd |

जो भेद मायने रखता है वह है **scratch बनाम बाक़ी सब कुछ**: scratch एकमात्र shared, GPU-only,
dynamically realloc'd-and-freed region है, और ऐतिहासिक `NotPresent` अपराधी। per-process KFD
event page `alloc_raw` के बाहर allocate होता है और track नहीं किया जाता: वहाँ का fault unmapped
के रूप में classify होता है।

### Classification

registry live ranges का एक `BTreeMap` रखता है (range queries के लिए base VA से keyed) साथ ही
**256** सबसे-हाल-में-freed regions (`FREED_HISTORY`) का एक bounded ring। `classify(va)` एक
faulting address को इस precedence के साथ resolve करता है:

```text
1. Live    — va is inside a currently-mapped allocation
             (live takes precedence, so a re-allocated VA reads Live, not stale)
2. Freed   — va is inside a recently-freed region → use-after-free
3. Unmapped — va is in no tracked region; report nearest live neighbours + gaps
```

`Display` rendering ही वह है जो fault message में उतरता है:

```text
Live:     va is at offset +0x40 within a LIVE scratch allocation
          [0x7f…000, 0x7f…400) (handle=0x42)

Freed:    va is within a RECENTLY-FREED scratch region [0x…, 0x…) (handle=0x…)
          — use-after-free: a stale/recycled VA still referenced by an
          in-flight kernel

Unmapped: va is in NO tracked allocation; nearest live below: VRAM buffer
          [0x…, 0x…) (va is +0x80 past its end); nearest live above: …
```

---

## एक fault कैसे report होता है

`KfdIface::wait_events` (`device/src/amd/iface.rs`) में, जब memory-fault event fire हो चुका
हो (`gpu_id != 0`), तो fields को bindgen union payload से locals में copy किया जाता है, VA को
classify किया जाता है, और एक enriched message बनाया जाता है:

```text
AMD GPU memory fault on gpu_id=… va=0x… (NotPresent=true ReadOnly=false NoExecute=false
Imprecise=false ErrorType=…) — va is at offset +0x40 within a LIVE scratch …
```

इसे एक `fault_logged: AtomicBool` latch और एक `tracing::error!` के माध्यम से **एक बार** log
किया जाता है। one-shot मायने रखता है: memory-fault event auto-reset नहीं होता, इसलिए बाद की
poll-fault calls (`wait_events(0)`) वही fault फिर से observe करती हैं — हर बार log करना spam
करता। फिर इसे एक typed `Error::GpuFault` के रूप में return किया जाता है, जिसका `Display` ऊपर
वाली string ही है; poison latch उस text को record करता है और हर बाद के entry point पर उसे एक
`Error::Runtime` (prefix `runtime error: ` के साथ) के रूप में फिर से throw करता है। बिना fault के
expire होने वाला wait एक अलग `Error::TimelineTimeout` है। (एक hardware-exception event, slot `[2]`, इसके बजाय
`reset_type`/`reset_cause`/`memory_lost` report करता है — उनके पास classify करने के लिए कोई
faulting VA नहीं होती।)

---

## Poison latch

एक memory fault पूरे per-VM page table को corrupt कर देता है, इसलिए एक के बाद device मर जाता
है। `AmdDeviceCore` (`device/src/amd/device.rs`) एक poison latch रखता है —
`poisoned: AtomicBool` + `error_msg: OnceLock<String>` — जो हर dispatch और synchronize entry
point पर check होता है:

- `poison(msg)` message को एक बार record करता है और flag सेट करता है;
- `is_poisoned()` hot-path gate है;
- `poison_error()` poisoned होने पर record किया गया `Error::Runtime` return करता है;
- `poll_faults_nonblocking()` तब `wait_events(0)` जारी करता है जब कोई signal wait अपनी deadline
  (dispatch, copy और lane acquisition के लिए 30 s) तक पहुँच जाए, ताकि असली error एक नंगी
  deadline के बजाय timeout से attach हो जाए। (spin-escalation path भी एक fault पर जल्दी बाहर
  निकलता है, पर इस poll के बजाय एक छोटे *blocking* `wait_events` के माध्यम से।) `poison()`
  lane pool पर parked हर thread को भी जगा देता है।

एक बार poisoned हो जाने पर, device पर किसी भी lane के विरुद्ध हर `synchronize`/`execute`
fail-fast होता है — GPU state और cached mappings अब भरोसेमंद नहीं रहते।

---

## Dispatch instrumentation: `SVOD_DEBUG_DISPATCH`

`SVOD_DEBUG_DISPATCH` (किसी भी चीज़ पर) सेट करना दो बिंदुओं पर `eprintln` dumps चालू कर देता
है, दोनों `device/src/amd/program.rs` में:

- **`[program-load]`** — प्रति program: kernarg/private/group sizes,
  `kernel_code_properties` (bit-by-bit decoded), user-SGPR count, `wave32`, और raw
  `rsrc1/2/3`। यह उन `kernel_code_properties` bits को flag करता है जिन्हें loader populate
  *नहीं* करता (जो kernel को garbage pointers पढ़ने और fault करने पर मजबूर कर देता)।
- **`[dispatch tv=…]`** — प्रति dispatch: kernel name, `grid`, `local`,
  `is_pm4`, kernarg GPU VA, scratch VA, और हर buffer की VA।

यह ठीक-ठीक यह देखने का सबसे तेज़ तरीक़ा है कि एक faulting dispatch ने किन VAs को छुआ, ताकि
registry की classification के विरुद्ध cross-reference किया जा सके।

---

## Tracing setup (`RUST_LOG`)

बैकएंड `tracing` crate (`debug!`, `tracing::error!`) का उपयोग करता है लेकिन **कोई subscriber
install नहीं करता** — यह host binary का काम है। `alloc_raw`/`free_raw` की `debug!` lines और
one-shot fault `error!` केवल तभी दिखती हैं जब एक subscriber install हो और level अनुमति दे।

जो example binaries एक install करती हैं वे `main` में `tracing_subscriber::fmt::init()` call
करती हैं (यह `RUST_LOG` का सम्मान करता है):

```bash
# Surface the alloc/free debug lines and the fault error from gigaam_infer:
RUST_LOG=svod_device=debug \
SVOD_DEVICE=AMD:0 \
  cargo run --release -p svod-model --example gigaam_infer -- ./audio.wav
```

:::tip[Pipeline debugger]
driver के बजाय *compiler*-side issues (IR extraction, LLVM IR, UOp trees) के लिए, project एक
`/svod-debug` skill ship करता है जो frontend → codegen tracing targets (`SVOD_DUMP_LLVM_IR`,
`SVOD_DUMP_AMD_IR`, per-stage `RUST_LOG` targets, `setup_test_tracing()`) का दस्तावेज़ीकरण
करता है। यह इस पेज की driver-side fault triage से एक अलग toolkit है।
:::

---

## एक worked triage

जब एक `NotPresent` fault फिर से आता है, workflow यह है:

1. fault message पहले ही class का नाम बता देता है — उसे पहले पढ़ें। "LIVE scratch" scratch
   realloc path की ओर इशारा करता है; "RECENTLY-FREED" एक ऐसे buffer का use-after-free है जो
   तब free हुआ जब एक kernel अब भी उसे reference कर रहा था; एक पास के neighbour के साथ "NO
   tracked allocation" एक overrun है (gap आपको बताता है कि कितने से)।
2. faulting dispatch की exact VAs देखने के लिए `SVOD_DEBUG_DISPATCH` सेट करके re-run करें, और
   उस तक ले जाने वाली alloc/free history देखने के लिए `RUST_LOG=svod_device=debug`।
3. faulting VA को dumped scratch/kernarg/buffer VAs के विरुद्ध cross-reference करें।

`NotPresent` के लिए प्रमुख संदिग्ध **scratch** है (`Scratch` tag के अनुसार) — एकमात्र shared,
GPU-only, dynamically realloc'd-and-freed region, जहाँ एक realloc-बनाम-dispatch race एक
kernel को एक freed buffer की ओर इशारा करते हुए छोड़ सकती है।

---

## यह क्यों ज़रूरी है

registry से पहले, एक fault आपको एक hex address देता था और कुछ नहीं। अब fault *message ख़ुद*
बताता है कि VA live scratch है, एक freed/stale VA है, या wild — एक अंधे शिकार को एक directed
शिकार में बदलते हुए। poison latch (जो corrupted state को फैलने देने के बजाय device को साफ़-सुथरे
ढंग से रोक देता है) और dispatch dumps के साथ जोड़ा जाए, तो बैकएंड GPU से एक debugger attach
किए बिना एक memory fault को localize कर सकता है।
