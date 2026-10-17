---
sidebar_label: Profiling
---

# CUDA पर Profiling

स्तरित profiler (`ExecutionPlan::profile`) `DispatchTimestamps` और
`KernelResources` handles के ऊपर backend-neutral है। यह पेज बताता है कि CUDA बैकएंड उन
handles में क्या डालता है, और कौन-से tiers मौजूद हैं।

| Tier | CUDA पर | Source |
|---|---|---|
| **1 — device time** | हाँ | हर launch के इर्द-गिर्द CUDA event जोड़ियाँ |
| **2 — roofline** | हाँ | backend-neutral (IR FLOP estimate, plan buffers) |
| **3 — static occupancy** | हाँ | `cuFuncGetAttribute` + `cuOccupancyMaxActiveBlocksPerMultiprocessor` |
| **4 — hardware counters** | हाँ | CUPTI range profiler (`libcupti.so.13`) |

```bash
SVOD_DEVICE=CUDA:0 cargo run --release -p svod-model --example gigaam_infer -- --profile ./audio.wav
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
| `occ%` | `occupancy` | `cuOccupancyMaxActiveBlocksPerMultiprocessor(block) × block / max threads per SM`, 1 पर clamp किया गया |

`wave_size` device का warp size (32) है। Occupancy query को एक block size चाहिए: program
अपने **नवीनतम launch** का block याद रखता है और किसी भी launch से पहले function के
`maxThreadsPerBlock` पर वापस आ जाता है। AMD के आँकड़े के विपरीत, जो केवल register-limited है,
driver की गिनती में registers, shared memory और per-SM block limit पहले से ही शामिल हैं।

---

## Tier 4: hardware counters

Counters CUPTI के range profiler से आते हैं, जो runtime पर एक ही soname,
`libcupti.so.13`, से bind होता है — पहले loader path पर और फिर `/opt/cuda/lib64`,
`/usr/local/cuda/extras/CUPTI/lib64` तथा `$CUDA_PATH/{lib64,extras/CUPTI/lib64}`
में खोजा जाता है (`device/src/cuda/cupti.rs`)। CUDA 13 ने PerfWorks का host API
CUPTI में ही समेट दिया, इसलिए पूरी sequence यही एक library उठाती है — और
`libnvperf_host.so` को CUPTI खुद `dlopen` करता है, तो वह loader को मिलनी चाहिए;
न मिलने पर यह `CUPTI_ERROR_NOT_INITIALIZED` की तरह दिखता है। यह binding उतना ही
optional है जितना `ptxas`: library न हो, काम की न हो, या `SVOD_CUDA_CUPTI=0` से बंद
कर दी गई हो, तो `pmc_available()` `false` रहता है और profiler अपनी एक-line नोट के
साथ Tiers 1-3 पर घट जाता है।

CUDA 13.3 में params वाले दो structs बड़े हो गए, इसलिए हर call पहले सबसे नया
`struct_size` भेजती है और `CUPTI_ERROR_INVALID_PARAMETER` पर एक size पीछे हट जाती है —
`cuptiProfilerGetCounterAvailability` (41, फिर 40) और `cuptiProfilerHostInitialize`
(56, फिर 48) — और `abi_ladder` वह पहला size याद रखता है जिसे installed CUPTI ने bad parameter
के रूप में reject नहीं किया।

`SVOD_PMC=1` इस backend का default set चुनता है:

| Token | Metric | अर्थ |
|---|---|---|
| `cycles` | `sm__cycles_active.sum` | वे cycles जिनमें कम से कम एक warp resident था |
| `warps` | `sm__warps_launched.sum` | launch हुए warps |
| `inst` | `smsp__inst_executed.sum` | execute हुए warp instructions |
| `tensor` | `sm__pipe_tensor_cycles_active.sum` | tensor pipe के active cycles |
| `dram` | `dram__bytes.sum` | DRAM से गुज़रे bytes |

Subset tokens से चुनें — `SVOD_PMC=tensor,dram`। Tokens सभी backends में unique
हैं, इसलिए CUDA पर AMD का token गिरा दिया जाता है, किसी दूसरे block को गलत
program नहीं करता; `set_pmc` बताता है कि उसने कितने रखे और executor यह stderr पर
कह देता है:

```text
SVOD_PMC: 2 of 5 requested counters are not collected on this backend
```

जिस list के सारे tokens अनजान हों वह default set पर वापस आ जाती है, और ऐसा चयन
जिसमें से कुछ भी न बचे एक सामान्य, बिना-arm किए timing run के बराबर है। `cycles`
के सापेक्ष `tensor` किसी matmul या flash-attention kernel का tensor-core
utilization है; और `dram` bandwidth-bound kernel को issue-bound से अलग कर देता है।

```bash
SVOD_DEVICE=CUDA:0 SVOD_PMC=1 cargo run --release -p svod-model --example gigaam_infer -- --profile ./audio.wav
```

tk3 के throughput probes सिर्फ़ hand kernels का समय मापते हैं। वे event timestamps पढ़ते हैं,
counters नहीं:

```bash
SVOD_DEVICE=CUDA:0 cargo test -p svod-tk3 --release --lib gemm_throughput_probe -- --ignored --nocapture
```

### Privileges

Driver default रूप से counter collection सिर्फ़ admin users तक सीमित रखता है, और
यह पाबंदी वहाँ नहीं दिखती जहाँ आप उम्मीद करेंगे: `cuptiRangeProfilerEnable` और
`cuptiRangeProfilerSetConfig` बिना privileges के भी सफल होते हैं, और
`CUPTI_ERROR_INSUFFICIENT_PRIVILEGES` सिर्फ़ counter availability image तथा
`cuptiRangeProfilerStart` पर आता है। इसीलिए `pmc_available()` उसी availability
image को probe करता है। पाबंदी हटाने के लिए:

```bash
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' \
  | sudo tee /etc/modprobe.d/nvidia-profiling.conf
# अपने distro के हिसाब से initramfs दोबारा बनाएँ, फिर reboot करें
```

`scripts/cupti_probe.cu` पूरी sequence एक saxpy kernel के विरुद्ध अलग से चलाकर
बताता है कि वह कहाँ रुकती है — privilege की दिक्कत को toolkit की दिक्कत से अलग
करने का यह सबसे तेज़ तरीका है; उसे `sudo` के नीचे चलाना बिना reboot किए इस
निदान की पुष्टि कर देता है।

### Collection की क़ीमत

Capture `CUPTI_AutoRange` में `CUPTI_KernelReplay` के साथ चलता है: CUPTI हर launch
पर एक range खोलता है और multi-pass config पूरा करने के लिए kernel को अंदर ही
replay करता है (ऊपर के पाँचों counters हर उस chip पर एक ही pass में schedule हुए
हैं जिस पर हमने चलाया है; जिस set को इससे ज़्यादा चाहिए वह `Stop` पर पकड़ा जाता है
और timing पर घट जाता है)। दो नतीजे आपके लिए पहले ही सँभाल लिए गए हैं:

- Capture किया गया CUDA graph एक अपारदर्शी submission की तरह replay होता है और
  उसके handles कभी counters नहीं ले जाते, इसलिए counters वाला run per-dispatch
  रास्ता लेता है।
- Kernel replay उसी dispatch की event pair को कई गुना बढ़ा देता है, इसलिए counters
  वाला run एक disarmed pass और जोड़ता है; `merge_min` उसकी timing को counted pass
  के counters के साथ रखता है। यानी एक ही table में timing और counters अलग-अलग
  passes से आते हैं।

Readback host-driven है और एक session अगली से overlap नहीं कर सकता, इसलिए counters
वाला dispatch वहीं synchronize करता है। कोई भी CUPTI failure उस dispatch को सिर्फ़
timing तक घटा देता है, पूरा run fail नहीं करता।

In-kernel timing प्रयोगों के लिए, `svod_codegen::llvm::nvptx::globaltimer()` एक `CUSTOM`
node बनाता है जो `%globaltimer`, यानी nanosecond GPU clock, पढ़ता है।
