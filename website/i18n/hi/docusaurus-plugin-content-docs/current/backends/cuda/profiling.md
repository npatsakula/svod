---
sidebar_label: Profiling
---

# CUDA पर Profiling

[स्तरित profiler](../../tile-kernels/profiling.md) `DispatchTimestamps` और
`KernelResources` handles के ऊपर backend-neutral है। यह पेज बताता है कि CUDA बैकएंड उन
handles में क्या डालता है, और कौन-से tiers मौजूद हैं।

| Tier | CUDA पर | Source |
|---|---|---|
| **1 — device time** | हाँ | हर launch के इर्द-गिर्द CUDA event जोड़ियाँ |
| **2 — roofline** | हाँ | backend-neutral (IR FLOP estimate, plan buffers) |
| **3 — static occupancy** | हाँ | `cuFuncGetAttribute` + `cuOccupancyMaxActiveBlocksPerMultiprocessor` |
| **4 — hardware counters** | **नहीं** | CUPTI चाहिए; bound नहीं है |

```bash
SVOD_DEVICE=CUDA:0 SVOD_PROFILE_ITERS=20 cargo run --release -p svod-model --example gigaam_infer -- ./audio.wav
```

---

## Tier 1: event timestamps

`profile` सेट होने पर `CudaPlanCtx::dispatch` plan की stream पर launch से पहले और बाद में
एक **timing event** record करता है और एक `CudaDispatchTimestamps` return करता है जो दोनों
का मालिक है। `timestamps_ns` को GPU clock पर nanoseconds report करने होते हैं, इसलिए वह यह
गणना करता है

```text
start    = cuEventElapsedTime(base_event, start_event)   // ms since the device opened
duration = cuEventElapsedTime(start_event, end_event)
end      = start + duration
```

Base event `CudaDevice::open` पर एक बार record होता है और वही timeline का शून्य है।
Duration को सीधे जोड़ी के बीच मापा जाता है (पूरा event resolution, लगभग आधा microsecond);
absolute position एक `f32` millisecond count से होकर जाता है जो process के पुराने होने के
साथ मोटा होता जाता है, यही कारण है कि `end` को base के विरुद्ध मापने के बजाय `start` से
derive किया जाता है। दोनों events का पूरा हो जाना (`cuEventQuery`) ज़रूरी है, अन्यथा handle
`None` report करता है।

Graph replays को भी उसी तरह profile किया जाता है: `replay_profiled` एक chain executable
चलाता है जिसमें हर kernel से पहले और बाद में एक event-record node होता है और यह प्रति
captured kernel एक handle return करता है ([Architecture](./architecture.md))।

`Program::execute_timed`, जिसका उपयोग BEAM करता है, dispatch stream पर वही event जोड़ी है,
जो एक `Duration` के रूप में return होती है।

---

## Tier 3: static resources

`CudaProgram::resource_usage` `KernelResources` को load पर पढ़ी गई function attributes से
भरता है:

| Column | Field | Source |
|---|---|---|
| `VGPR` | `vgprs` | `CU_FUNC_ATTRIBUTE_NUM_REGS` (प्रति thread registers) |
| `SGPR` | `sgprs` | `-` (NVIDIA पर कोई scalar register file नहीं) |
| `LDS` | `lds_bytes` | `CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES` (static `.shared`) |
| `scratch` | `scratch_bytes` | `CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES` (प्रति thread `.local`) |
| `occ%` | `occupancy` | `cuOccupancyMaxActiveBlocksPerMultiprocessor(block) × block / max threads per SM` |

`wave_size` device का warp size (32) है। Occupancy query को एक block size चाहिए: program
अपने **नवीनतम launch** का block याद रखता है और किसी भी launch से पहले function के
`maxThreadsPerBlock` पर वापस आ जाता है। AMD के आँकड़े के विपरीत, जो केवल register-limited है,
driver की गिनती में registers, shared memory और per-SM block limit पहले से ही शामिल हैं।

---

## Tier 4: उपलब्ध नहीं

कोई CUPTI binding नहीं है, इसलिए CUDA पर `PlanContext::pmc_available()` `false` है।
`SVOD_PMC=1` सेट करना fail नहीं होता: profiler घटकर Tiers 1-3 पर आ जाता है और अपना एक-line
नोट print करता है कि counters उपलब्ध नहीं हैं। `PmcCounter` enum आज AMD-SQ-specific है; उसे
चौड़ा करना [roadmap](./limitations.md) का हिस्सा है।

In-kernel timing प्रयोगों के लिए, `svod_codegen::llvm::nvptx::globaltimer()` एक `CUSTOM`
node बनाता है जो `%globaltimer`, यानी nanosecond GPU clock, पढ़ता है।
