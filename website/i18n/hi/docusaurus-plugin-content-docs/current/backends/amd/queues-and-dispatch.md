---
sidebar_label: Queues और Dispatch
---

# Queues और Dispatch

AMD बैकएंड Tinygrad के validated PM4, AQL, और SDMA packet semantics को बनाए रखता है, पर queue
scheduling और failure handling के लिए Rust ownership का उपयोग करता है। केंद्रीय नियम सरल है:
**एक non-clone lease ही किसी compute lane का एकमात्र publication authority है**।

## Compute lanes

`AmdDeviceCore` एक bounded `QueuePool` का मालिक है। उसके slots fixed `OnceLock`s हैं और queues
`SVOD_AMD_HW_QUEUES` तक lazily बनाई जाती हैं, जिसे 1 से 64 के बीच clamp किया जाता है और जिसका
default multi-XCC CDNA पर 4 है, बाक़ी हर जगह 1। एक atomic bitset leases को track करता है:

- एक initialized idle lane claim करना एक atomic compare-exchange है;
- queue creation एक cold serialized path है;
- जब हर lane leased हो, तो callers एक condition variable पर park करते हैं;
- `QueueLease` drop होने पर bit clear होता है और एक waiter जागता है;
- queues कभी host publishers के साथ co-tenant नहीं होतीं।

Programs और graph templates कभी `QueueLease` नहीं रखते। `OwnerCtx` logical plan state रखता
है — device core और allocator, सबसे नया completion, PMC selection, एक optional linked replay
template — और एक `session` slot, जिसे केवल direct-dispatch fallback अपनी epoch lease से
भरता है।

एक thread प्रति wait अधिकतम 30 s park करता है; लगातार दो expiries "AMD lane acquisition" के
लिए `Error::TimelineTimeout` return करती हैं, और एक poisoned device हर waiter को अपनी poison
error के साथ जगा देता है।

Direct semantic fallback एक replay epoch के सभी kernels में एक ही lease बनाए रखता है, फिर
`PlanContext::finish_replay` उसे release करता है। एक बाद वाला epoch दूसरी lane acquire करने से
पहले पिछले finalizer का wait करता है, क्योंकि एक अलग queue पुरानी queue की FIFO ordering
विरासत में नहीं लेती। Graph और native linked replay पहले से ही अपनी mutable kernarg/control
storage दोबारा उपयोग करने से पहले wait करते हैं और हर publication epoch के लिए एक lane lease
करते हैं।

## Native queues

`AmdComputeQueue` एक 16 MiB host-visible ring, GART read/write pointers, एक doorbell mapping,
और KFD queue backing का मालिक है। Packet format एक बार चुना जाता है:

```text
PM4 = num_xcc == 1 && SVOD_AMD_AQL is unset or "0"
AQL = otherwise
```

- PM4 queues raw dwords publish करती हैं और अगले dword index पर ring करती हैं।
- AQL queues 64-byte packets publish करती हैं और आख़िरी लिखे गए packet के index पर ring
  करती हैं।
- AQL kernel `completion_signal` zero ही रहता है। Vendor-IB PM4 waits/stores ख़ुद timeline
  completion के मालिक हैं, multi-XCC hardware पर XCC0 `PRED_EXEC` के साथ।

lane lease compute co-tenancy को ख़त्म कर देता है। `AmdComputeQueue.inner` अब भी एक mutex का
उपयोग एक Rust aliasing guard के रूप में करता है; सामान्य compute path पर वह uncontended है।
singleton SDMA queue स्वतंत्र रूप से mutex-protected है, क्योंकि अलग-अलग plans की copies उसे
साझा कर सकती हैं।

## Publication

Native linked replay एक submission को preparation और publication में बाँटता है:

1. program identity, concrete buffer ownership, ABI, launch geometry, patch tables, और
   hardware stream limits validate करें।
2. kernargs/control data reserve और write करें।
3. ring headroom acquire करें।
4. जब device-wide drains को किसी plan-owned timeline को observe करना हो, तो एक prepared
   finalizer register करें।
5. packets और doorbells publish करें।
6. finalizer को published चिह्नित करें।

यदि registration के बाद कोई error unwind करता है, तो prepared finalizer failed हो जाता है। एक
concurrent drain जागकर तुरंत fail हो जाता है, बजाय इसके कि वह उस terminal store का इंतज़ार करे
जो कभी publish हुआ ही नहीं। फिर physical device poison कर दिया जाता है, इसलिए lane दोबारा
उपयोग नहीं की जा सकती और hardware-referenced allocations quarantine कर दी जाती हैं।

Ordinary per-kernel dispatch (`submit_hcq_dispatch`) सरल है: उसका finalizer doorbell के बाद
published रूप में ही जन्म लेता है और caller उसे in flight register करता है; doorbell से पहले
की failure इसके बजाय ring और timeline reservation को roll back कर देती है।

PM4, AQL, और SDMA publication — तीनों अपने rings को wrap करने से पहले KFD read pointer जाँचते
हैं (AQL और SDMA एक monotonic value पढ़ते हैं; PM4 pointer queue-relative है और उसका epoch
producer index से दोबारा बनाया जाता है)। Ordinary dispatch अतिरिक्त रूप से in-flight timeline values को
bound करता है। PM4 timeline values 2^31 watermark पर drain होकर reset हो जाते हैं, क्योंकि
hardware wait/store packets निचले 32 bits की तुलना करते हैं।

## Resource lifetime

हर direct submission finalizer अपना code object retain करता है। Graphs और linked plans उन सभी
code objects को retain करते हैं जिन्हें वे link करते हैं। Persistent kernarg, resident command,
control, timestamp, और PMC allocations तब तक owned रहती हैं जब तक उनका ठीक वही replay
completion retire न हो जाए।

Queue lifecycle explicit है:

```text
Constructing -> Active
Constructing -> Destroyed | Quarantined
Active -> Destroyed
Active -> Quarantined
```

Orderly compute teardown है drain, KFD `DESTROY_QUEUE`, scratch release, फिर
ring/GART/context release। एक failed drain या destroy physical device को poison कर देता है और
हर संभावित रूप से referenced backing को mapped छोड़ देता है। सफल queue destruction के बाद
doorbell unmap का fail होना एक host mapping leak के रूप में report होता है, पर वह safe GPU
backing को अनावश्यक रूप से quarantine नहीं करता।

यदि `CREATE_QUEUE` सफल होता है पर doorbell mapping और rollback destruction दोनों fail होते हैं,
तो `setup_ring` `AmdQueueStillActive` return करता है। Caller allocation guards के unwind होने
से पहले device को poison कर देता है, जो एक live KFD queue को freed ring memory observe करने से
रोकता है।

किसी lane से होकर unwind होने वाला panic उस lane को quarantine करता है पर device को poison
नहीं करता; केवल वह reservation जो अपने doorbell के ring होने के *बाद* abandon हुआ हो
(`TimelineReservation`, `PreparedPublication`) poison करता है, चाहे कारण error हो या panic।
Signal slots बिना शर्त pool को वापस जाते हैं, सिवाय तब जब device poisoned हो — उस स्थिति में
उन्हें रोक लिया जाता है, क्योंकि कोई abandoned queue अब भी उन्हें target कर सकती है।

## Device-wide drains

हर lane एक queue timeline और non-queue finalizers की एक FIFO की मालिक है। device core हर
initialized lane के weak references रखता है। `synchronize_all` उन lanes का snapshot लेता है और
publication locks लिए बिना उनके timelines का wait करता है। Host reads, writes, और destructive
frees scoped `wait_storage` को प्राथमिकता देते हैं, जो केवल उन्हीं submissions का wait करता है
जो उस storage base के विरुद्ध record हुई हैं, और किसी unknown VA पर या `SVOD_AMD_SCOPED_SYNC=0`
के तहत पूरे drain पर fall back करता है।

Native replay republish करने से पहले हर operation को अतिरिक्त रूप से re-validate करता है: एक
PROGRAM को अब भी एक ऐसा `AmdProgram` होना चाहिए जिसका core ठीक वही `Arc` हो (`Arc::ptr_eq`, न
कि कोई allocator जो केवल `AMD:N` report कर रहा हो) और जिसके PM4 तथा AQL program addresses
unchanged हों, और एक COPY lane को एक installed SDMA queue चाहिए।

## Backend seam

KFD operations `AmdIface` के पीछे isolated हैं:

```rust
pub trait AmdIface: Send + Sync + std::fmt::Debug {
    fn alloc_raw(
        &self,
        size: usize,
        kind: AllocKind,
        tag: AllocTag,
        cpu_access: bool,
        zero: bool,
    ) -> Result<AllocResult>;
    fn free_raw(&self, gpu_va: u64, size: usize, handle: u64);
    fn setup_ring(&self, desc: &RingDesc) -> Result<QueueHandle>;
    fn teardown_ring(&self, queue_id: u32, doorbell_base: NonNull<u8>) -> Result<QueueTeardown>;
    fn wait_events(&self, timeout_ms: u32) -> Result<Option<Error>>;

    // Defaulted hooks. `KfdIface` overrides the first and the last; the
    // test mock overrides `publication_checkpoint`.
    fn queue_event_mailbox(&self) -> Option<QueueEventMailbox> {
        None
    }
    fn publication_checkpoint(&self, _stage: PublicationStage) -> Result<()> {
        Ok(())
    }
    fn update_queue_percentage(&self, _queue_id: u32, _ring_gpu: u64, _ring_size: u32, _percentage: u32) -> Result<()> {
        Ok(())
    }
}
```

Ring, GART, EOP, context-save, और inactive-signal buffers इस seam के ऊपर allocate होते हैं।
`setup_ring` उन resources को activate करता है और doorbell map करता है।
`update_queue_percentage` ही वह चीज़ है जो एक AQL queue को दोबारा map करती है ताकि CP firmware
अपना cached `amd_queue_t` scratch descriptor दोबारा पढ़े — lane creation पर और जब भी scratch
बढ़े; एक failed remap device को poison कर देता है। जब `queue_event_mailbox` मौजूद हो, तो हर
completion store के साथ एक KFD event-interrupt companion जाता है, ताकि `wait_events` बिना
polling के जाग सके।

## Configuration

| Variable | Default | प्रभाव |
|---|---|---|
| `SVOD_DEVICE` | `CPU` (macOS पर `METAL:0`) | default tensor device चुनें, उदाहरण के लिए `AMD:0` (`HIP` एक alias है) |
| `SVOD_AMD_BACKEND` | `kfd` | AMD बैकएंड; फ़िलहाल केवल `kfd` स्वीकार्य है |
| `SVOD_AMD_HW_QUEUES` | multi-XCC पर 4, वरना 1 | Bounded compute-lane count, 1 से 64 तक clamp किया गया |
| `SVOD_AMD_AQL` | unset | `0` के अलावा कोई भी value single-XCC hardware पर AQL को force करती है; हर queue creation पर पढ़ा जाता है |
| `SVOD_AMD_SCOPED_SYNC` | unset | `=0` हर storage-scoped host wait को पूरे device drain से बदल देता है और producer recording रोक देता है; प्रति process एक बार latch होता है |
| `SVOD_PM4_GRAPH` | unset | `=1` PM4 graph capture enable करता है; केवल `1` गिना जाता है; device creation पर एक बार पढ़ा जाता है |
| `AMD_DISABLE_SDMA` | unset | SDMA copy queue skip करने के लिए किसी भी value पर सेट करें, जो buffers को host-visible बना देता है |
| `SVOD_KFD_TOPOLOGY` | sysfs | tests के लिए KFD topology root override करें |
| `SVOD_DEBUG_DISPATCH` | unset | program-load और dispatch grid, kernarg, scratch, तथा buffer addresses print करने के लिए किसी भी value पर सेट करें |
| `SVOD_DUMP_AMD_IR` | unset | वह directory जिसमें हर kernel का AMD LLVM IR `<mcpu>_<module>.ll` के रूप में लिखा जाता है |
| `SVOD_OBJECT_CACHE`, `SVOD_OBJECT_CACHE_DIR`, `SVOD_OBJECT_CACHE_MAX_BYTES` | on | साझा on-disk object cache, देखें [CPU पेज](../cpu.md) |
| `SVOD_AM_DEBUG` | unset | केवल AM bring-up: registers लिखने के बाद उन्हें वापस पढ़ें |
| `SVOD_AM_MCBASE` | unset | केवल AM bring-up: `raw`, `fb`, या `fbxgmi` MC aperture base |

कोई `SVOD_AMD_SINGLE_QUEUE` नहीं है। जब एक single hardware lane चाहिए, तो
`SVOD_AMD_HW_QUEUES=1` सेट करें।
