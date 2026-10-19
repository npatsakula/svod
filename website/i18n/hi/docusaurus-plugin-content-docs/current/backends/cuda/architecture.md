---
sidebar_label: आर्किटेक्चर
---

# आर्किटेक्चर

यह पेज बैकएंड का अनुसरण करता है, driver binding से लेकर graph replay तक। नीचे जो कुछ भी है
वह `device/src/cuda/` में है जब तक कि अन्यथा न कहा गया हो।

```text
mod.rs        the dl_api! runtime-binding macro and the module's re-exports
sys.rs        the bound driver entry points (libloading), CUresult, handles
device.rs     CudaDevice: primary context, limits, lanes, base event, scoped-sync tables, poison latch
allocator.rs  CudaAllocator: device / managed / pinned memory, staged copies
program.rs    CudaProgram: cubin or PTX module load, cuLaunchKernel, execute_timed, resources
sync.rs       CudaPlanCtx, CudaDispatchTimestamps, CudaCompletionToken
graph.rs      CudaGraph: a CUDA graph DAG from GraphKernel::deps, patched replays
cupti.rs      the CUPTI range profiler behind Tier 4 (see Profiling)
```

---

## Driver bindings

`sys.rs` बैकएंड द्वारा उपयोग किए जाने वाले हर entry point को एक ही `dl_api!` block में
declare करता है: Rust field, exact export name, और C prototype; macro ख़ुद `mod.rs` में
रहता है और `cupti.rs` उसे फिर से उपयोग करता है। `Api::load`
`libcuda.so.1` खोलता है और उन सबको पहले ही resolve कर लेता है, इसलिए एक ग़ायब या नाम बदला
हुआ symbol पहले उपयोग पर नहीं, बल्कि एक ही बार, load पर, `Error::DeviceUnavailable`
(`libcuda.so.1 has no symbol ...`) के रूप में fail होता है। नाम वे **versioned** exports हैं
जिन पर `cuda.h` remap करता है: `cuMemAlloc_v2`, `cuDevicePrimaryCtxRelease_v2`,
`cuGraphAddKernelNode_v2`, `cuGraphExecKernelNodeSetParams_v2`,
`cuGraphInstantiateWithFlags` (unversioned `cuGraphInstantiate` एक legacy पाँच-argument ABI
है और उसे कभी छुआ नहीं जाता)। `cuEventElapsedTime` जान-बूझकर रखा गया अपवाद है: उसका `_v2`
CUDA 12.8 का है, जो driver की न्यूनतम सीमा को R570 तक उठा देता।

Handles `#[repr(transparent)]` pointer newtypes हैं (`CUcontext`, `CUmodule`, `CUfunction`,
`CUstream`, `CUevent`, `CUgraph`, `CUgraphExec`, ...); `CUdeviceptr` एक `u64` है।
`CUresult` एक integer newtype है ताकि किसी नए driver के codes भी round-trip करें;
`CUresult::check("cuLaunchKernel")` एक failure को इसमें बदल देता है

```text
CUDA cuLaunchKernel failed: CUDA_ERROR_INVALID_VALUE (1): invalid argument
```

driver के अपने `cuGetErrorName` / `cuGetErrorString` का उपयोग करते हुए।
`CudaKernelNodeParams` struct `CUDA_KERNEL_NODE_PARAMS_v2` को compile-time size और offset
assertions के साथ mirror करता है।

---

## Device, context, streams

`CudaDevice::open(id)` प्रति process cached है। यह `cuInit` चलाता है, device के
**primary context** को retain करता है (`cuDevicePrimaryCtxRetain`), उन `CudaLimits` को
पढ़ता है जिनकी उसे ज़रूरत है (`cuDeviceGetAttribute`: SM count, प्रति block और प्रति SM
threads, प्रति SM blocks, प्रति block shared memory, warp size, और यह कि managed memory
supported और concurrently accessible दोनों है या नहीं), दो non-blocking streams बनाता है
(allocator के लिए एक **copy stream** और per-call `Program::execute` के लिए एक **dispatch
stream**), और legacy default stream पर एक **base event** record करता है — device return
होने से पहले उसकी प्रतीक्षा की जाती है — जो हर GPU-clock timestamp का शून्य है।

Driver current context को प्रति thread रखता है, इसलिए बैकएंड का हर entry point `enter()`
से शुरू होता है: यदि device poisoned है तो मना कर दो, फिर `cuCtxSetCurrent`। एक **sticky**
`CUresult` (`ILLEGAL_ADDRESS`, `LAUNCH_FAILED`, `ILLEGAL_INSTRUCTION`, `MISALIGNED_ADDRESS`,
`ECC_UNCORRECTABLE`, `LAUNCH_TIMEOUT`, `ASSERT`, `HARDWARE_STACK_ERROR`,
`INVALID_ADDRESS_SPACE`, `INVALID_PC`, `DEINITIALIZED`, और सावधानीवश `UNKNOWN` — `sys.rs`)
poison flag को अपने message के
साथ latch कर देता है; device पर हर बाद की call उसी message के साथ fail-fast होती है, जैसा
AMD पर होता है।

---

## Memory

एक `RawBuffer::Cuda` एक device pointer, एक optional host pointer, और अपनी `CudaMemory`
kind रखता है, जिसे `BufferSpec` से चुना जाता है:

| `BufferSpec` | Kind | Driver call |
|---|---|---|
| default | `Device` | `cuMemAlloc` — device memory, कोई host mapping नहीं |
| `cpu_access` | `Managed` यदि device concurrent managed access report करता हो, अन्यथा `Pinned` | `cuMemAllocManaged(ATTACH_GLOBAL)`, एक ही address दोनों ओर valid; अन्यथा `cuMemHostAlloc(PORTABLE \| DEVICEMAP)` + `cuMemHostGetDevicePointer` |
| `host` | `Pinned` | `cuMemHostAlloc(PORTABLE \| DEVICEMAP)`, kernels इसे bus के ऊपर से पढ़ते हैं |

`supports_device_local()` `true` है, इसलिए intermediates device पर ही रहते हैं।
Host <-> device copies पहले storage के in-flight producers और readers की प्रतीक्षा करती
हैं (`CudaDevice::wait_storage`, नीचे — host access lanes के विरुद्ध ordered नहीं है), फिर
data ले जाती हैं। 4 MiB तक एक copy-out एक synchronous `cuMemcpyDtoH` है, जबकि एक copy-in
copy lane पर एक `cuMemcpyHtoDAsync` है जो storage के नए producer के रूप में publish होती
है; यह stream को केवल तभी synchronize करती है जब source ऐसी memory हो जिसे driver track
करता है (pinned, registered या managed, `cuPointerGetAttribute` से पूछा गया), क्योंकि
driver एक pageable source को return करने से पहले stage करता है। 4 MiB से ऊपर दोनों दिशाएँ
एक lazily allocate किए गए **pinned staging buffer** के माध्यम से `cuMemcpyHtoDAsync` /
`cuMemcpyDtoHAsync` के साथ 4 MiB chunks में जाती हैं, प्रति chunk stream को synchronize
करते हुए। Pinned buffers सीधे `memcpy` कर दिए जाते हैं। Device-to-device `_transfer` और
zero-fills copy lane पर asynchronous हैं: `cuStreamWaitEvent` से producers के बाद ordered,
दोनों ranges के नए producer के रूप में publish, और किसी भी lane पर हर बाद के launch द्वारा
प्रतीक्षित, इसलिए वे host को block नहीं करतीं — सिवाय एक allocation के अंदर overlapping range
के, जो `memmove` semantics बनाए रखने के लिए एक temporary से होकर गुज़रती है और उस temporary
के free होने पर प्रतीक्षा करती है। Free करना पहले
storage के producers की प्रतीक्षा करता है; यदि यह प्रतीक्षा fail होती है (poisoned context)
तो allocation को एक in-flight kernel के नीचे free करने के बजाय **quarantine** (leak) कर
दिया जाता है। हर compute allocator की तरह यह `LruAllocator` के नीचे बैठता है, जो एक
recycle की गई allocation को उसके पिछले owner के producers पर fence करता है।

---

## Programs और launches

`CudaProgram::load` `is_cubin` पर branch करता है — एक ELF image `validate_cubin` से होकर
जाती है, PTX text entry की `.param` जाँच से — और दोनों 16 KiB error और info log buffers
के साथ उसी `cuModuleLoadDataEx` तक पहुँचते हैं, ताकि एक JIT failure
`Error::CudaJit { kernel, cause, log }`
के रूप में सामने आए जो `ptxas` का अपना message रखता है (देखें [Debugging](./debugging.md));
info log `tracing::debug!` पर जाता है। फिर यह entry को `cuModuleGetFunction` से bind करता
है और function attributes `MAX_THREADS_PER_BLOCK`, `NUM_REGS`, `SHARED_SIZE_BYTES`,
`LOCAL_SIZE_BYTES` पढ़ता है। Module किसी भी graph के साथ `Arc`-shared होता है जिसने उसे
capture किया, और आख़िरी drop पर unload होता है।

Kernel arguments `cuLaunchKernel` की `extra` array में **एक packed blob** के रूप में जाते
हैं (`CU_LAUNCH_PARAM_BUFFER_POINTER` / `_SIZE` / `_END`), जिसे साझा `ClikeKernargLayout`
lay out करता है: 8-byte device pointers, 4-byte `i32` scalars, PARAM slot order में, जो
ठीक-ठीक PTX का स्वाभाविक `.param` layout है। `global_size` **blocks में grid** है और
`local_size` **threads में block** (वही work-group convention जो AMD और Metal उपयोग करते
हैं); function के `maxThreadsPerBlock` से बड़ा block launch से पहले ही reject कर दिया जाता
है, message में register, shared और local memory के आँकड़ों के साथ।

`Program::execute` device की dispatch stream पर launch करता है और वैकल्पिक रूप से उस पर
wait करता है; `execute_timed` launch के इर्द-गिर्द एक timing event pair record करता है और
`cuEventElapsedTime` return करता है, ताकि BEAM candidates को GPU time पर rank करे।

---

## Plan contexts, tokens, timelines

हर execution plan को एक `CudaPlanCtx` मिलता है: **एक non-blocking stream**, जो उसकी lane
है, और साथ में CUPTI counter selection तथा session जब counters armed हों। `dispatch` उस पर
launch करता है; `profile` सेट होने पर यह launch को timing events से
घेरता है और एक `CudaDispatchTimestamps` return करता है ([Profiling](./profiling.md))।
`completion_token` एक completion-only event (`CU_EVENT_DISABLE_TIMING`) record करता है
जिसका `wait` `cuEventSynchronize` है और जिसका `retired` `cuEventQuery` है; `synchronize`
`cuStreamSynchronize` है।

### Scoped synchronization

Lanes आपस में एक-दूसरे के विरुद्ध ordered नहीं हैं, इसलिए `CudaDevice` तीन tables रखता है
(`device/src/cuda/device.rs` के module docs):

- **producers** — storage base -> प्रति lane वह नवीनतम completion token जिसने उसे पढ़ा या
  लिखा, और क्या उसने लिखा (एक host overwrite in-flight readers के विरुद्ध भी एक WAR hazard
  है)। Execution plan हर execute के बाद अपना या अपने graph का token उन सभी storages पर publish
  करता है जिन्हें वह छूता है, और जहाँ वह लिखता है वहाँ उसे write के रूप में चिह्नित करता है;
  allocator हर transfer या memset के बाद एक copy-lane token publish करता है।
  `wait_storage(base)` नीचे बताई गई lanes को drain करता है, फिर उन tokens की प्रतीक्षा
  करता है, फिर उन्हें table से हटा देता है। जिस storage को table नहीं जानता — इसमें वह भी
  शामिल है जिसका नवीनतम token किसी दूसरे backend का हो — वह `cuCtxSynchronize` पर वापस
  गिर जाता है।
- **lanes** — हर जीवित lane और उसके पास कितनी submissions हैं जिनके लिए कोई token publish
  नहीं हुआ (per-call `Program::execute`, एक plan जो बीच में fail हुआ, एक graph replay उसका
  token लिए जाने से पहले)। `wait_storage` ऐसी lanes को host पर drain करता है; एक copy
  इसके बजाय हर एक पर एक tail event record करती है और उसकी प्रतीक्षा GPU पर करती है। Copy
  lane ख़ुद इस table में नहीं है।
- **copy tail** — नवीनतम copy-lane event; हर launch चलने से पहले उसकी प्रतीक्षा GPU पर
  करता है, ताकि asynchronous copies हर बाद के kernel से पहले आएँ।

हर execute से पहले plan अपनी lane (या अपने graph की lane) को दूसरी lanes के टकराने वाले
tokens के बाद `cuStreamWaitEvent` से order करता है (`CudaDevice::order_lane_after`): हर उस
storage के writers के बाद जिसे वह पढ़ता है, और हर उस storage के सभी tokens के बाद जिसे वह
लिखता है। जो plans किसी storage को सिर्फ़ पढ़ते हैं (जैसे shared weights), वे concurrent
रहते हैं। Tokens अपनी lanes से अधिक जीते हैं, इसलिए एक one-shot `realize` जिसका plan drop हो
चुका है, फिर भी अगले को order करता है।

`SVOD_CUDA_SCOPED_SYNC=0` इस सबको disable कर देता है: तब हर wait context को drain करती है,
हर copy copy stream को synchronize करती है, और हर plan execute पहले context को drain करता है।

Executor की cross-plan ordering हर backend पर, CUDA सहित, एक host signal
(`CpuTimelineSignal`) है; इसका अपना कोई `TimelineSignal` implementation नहीं है। Host
signal को critical path से बाहर जो रखता है वह ऊपर की machinery है: tables GPU work को GPU
work के विरुद्ध `cuStreamWaitEvent` से order करती हैं, इसलिए एक plan host पर केवल उसी के
लिए प्रतीक्षा करता है जिसे वह वास्तव में पढ़ता है।

---

## Graphs

`CudaGraph::capture` एक captured kernel chain को एक असली **CUDA graph** में बदल देता है:
प्रति kernel एक `cuGraphAddKernelNode_v2` जिसकी dependency list ठीक-ठीक
`GraphKernel::deps` है, यानी host hazard analysis। इसलिए स्वतंत्र kernels device पर overlap
कर सकते हैं (AMD बैकएंड `deps` को छोड़ देता है क्योंकि एक single in-order ring उन्हें
अनावश्यक बना देता है)। हर node के params उसी `extra` protocol के माध्यम से उस kernel के
kernarg blob की ओर इशारा करते हैं जो eager launches में है; graph को
`cuGraphInstantiateWithFlags` से instantiate किया जाता है। Capture एक ख़ाली chain, एक
non-CUDA program, या किसी दूसरे device के program के लिए मना कर देता है (`Ok(None)`), और एक
poisoned device या किसी बाद वाले kernel पर dependency होने पर fail (`Err`) होता है; plan
दोनों को per-call dispatch मानता है।

`replay(buffers, vals)` केवल उन kernels को फिर से pack करता है जिनका `(buffers, vals)`
slice बदला है और उन nodes को `cuGraphExecKernelNodeSetParams_v2` से update करता है, फिर
graph की अपनी stream पर `cuGraphLaunch` करता है। एक सूक्ष्मता: record किए गए hazards केवल
उसी **aliasing** के लिए valid हैं जिसके साथ chain capture हुई थी। यदि एक replay buffers को
इस तरह bind करे कि अब slots की कोई दूसरी जोड़ी एक ही address साझा करे, तो graph एक lazily
बनाई गई **capture-order chain** पर switch कर जाता है (हर kernel पिछले के बाद), जो हमेशा
सही होती है।

`replay_profiled` एक तीसरे executable का उपयोग करता है, वही chain हर kernel के इर्द-गिर्द
एक `cuGraphAddEventRecordNode` जोड़ी के साथ; events को प्रति launch फिर से arm किया जाता है
(`cuGraphExecEventRecordNodeSetEvent`) ताकि पहले ही सौंपे जा चुके handles अपने stamps रखें,
और प्रति captured kernel एक `CudaDispatchTimestamps` capture order में return होता है।

---

## Object cache identity

Compiled PTX साझा on-disk object cache से होकर जाता है, जिसकी key rendered IR और एक
`CompilerIdentity` है:

```text
schema:              OBJECT_CACHE_SCHEMA
backend:             nvptx-clang
target_architecture: nvptx64-nvidia-cuda/sm_86
toolchain:           <clang identity>[;ptxas:path=...;version=...]
flags:               -x ir -S -O3 --target=nvptx64-nvidia-cuda -march=sm_86 --cuda-feature=+ptx78 -Wno-override-module - -o -
                     [-arch=sm_86 -o /dev/stdout /dev/stdin]
abi:                 ptx-kernel-abi-v1;warp-size=32
object_format:       ptx-text-v1 | cubin-v1
```

कोष्ठकों वाले हिस्से `ptxas` के रास्ते के हैं: assembler के साथ cached object एक **cubin**
होता है, उसके बिना **PTX text**, जिसे driver load पर assemble करता है और SASS को अपने ख़ुद
के `~/.nv/ComputeCache` में रखता है। ये दोनों formats कभी एक ही entry साझा नहीं करते,
क्योंकि वे `toolchain`, `flags` और `object_format` — तीनों में भिन्न हैं। Rendered IR भी
पूरी key नहीं है: ABI descriptors उसके साथ जोड़ दिए जाते हैं, क्योंकि एक cubin की entry की
जाँच compile time पर उन्हीं के विरुद्ध होती है। हर cache hit को driver तक पहुँचने से पहले
उसके format के validator द्वारा फिर से validate किया जाता है — `validate_cubin` या
`validate_ptx`, देखें [Codegen](./codegen.md)।
`SVOD_OBJECT_CACHE=0` cache को disable करता है, `SVOD_OBJECT_CACHE_DIR` उसे स्थानांतरित
करता है और `SVOD_OBJECT_CACHE_MAX_BYTES` उसका budget तय करता है (देखें
[CPU पेज](../cpu.md))।

Device factory (`create_cuda_device`) ऐसे device को भी मना कर देता है जिसकी per-block
shared memory limit optimizer profile की static `shared_max` से कम हो, क्योंकि profile के
हिसाब से sized एक kernel अन्यथा केवल JIT पर fail होता।
