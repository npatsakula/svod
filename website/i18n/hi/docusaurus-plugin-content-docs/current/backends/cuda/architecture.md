---
sidebar_label: आर्किटेक्चर
---

# आर्किटेक्चर

यह पेज बैकएंड का अनुसरण करता है, driver binding से लेकर graph replay तक। नीचे जो कुछ भी है
वह `device/src/cuda/` में है जब तक कि अन्यथा न कहा गया हो।

```text
sys.rs        the bound driver entry points (libloading), CUresult, handles
device.rs     CudaDevice: primary context, limits, streams, base event, poison latch
allocator.rs  CudaAllocator: device / managed / pinned memory, staged copies
program.rs    CudaProgram: PTX module load, cuLaunchKernel, execute_timed, resources
sync.rs       CudaPlanCtx, CudaDispatchTimestamps, CudaCompletionToken, CudaTimelineSignal
graph.rs      CudaGraph: a CUDA graph DAG from GraphKernel::deps, patched replays
```

---

## Driver bindings

`sys.rs` बैकएंड द्वारा उपयोग किए जाने वाले हर entry point को एक ही `cuda_api!` macro में
declare करता है: Rust field, exact export name, और C prototype। `Api::load`
`libcuda.so.1` खोलता है और उन सबको पहले ही resolve कर लेता है, इसलिए एक ग़ायब या नाम बदला
हुआ symbol पहले उपयोग पर नहीं, बल्कि एक ही बार, load पर, `Error::DeviceUnavailable`
(`libcuda.so.1 has no symbol ...`) के रूप में fail होता है। नाम वे **versioned** exports हैं
जिन पर `cuda.h` remap करता है: `cuMemAlloc_v2`, `cuCtxDestroy_v2`,
`cuGraphAddKernelNode_v2`, `cuGraphExecKernelNodeSetParams_v2`,
`cuGraphInstantiateWithFlags` (unversioned `cuGraphInstantiate` एक legacy पाँच-argument ABI
है और उसे कभी छुआ नहीं जाता)।

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
threads और shared memory, registers, warp size, L2, managed-memory support), दो
non-blocking streams बनाता है (allocator के लिए एक **copy stream** और per-call
`Program::execute` के लिए एक **dispatch stream**), और एक **base event** record करता है जो
हर GPU-clock timestamp का शून्य है।

Driver current context को प्रति thread रखता है, इसलिए बैकएंड का हर entry point `enter()`
से शुरू होता है: यदि device poisoned है तो मना कर दो, फिर `cuCtxSetCurrent`। एक **sticky**
`CUresult` (`ILLEGAL_ADDRESS`, `LAUNCH_FAILED`, `ILLEGAL_INSTRUCTION`, `ECC_UNCORRECTABLE`,
... वे codes जिन्हें driver context के लिए घातक बताता है) poison flag को अपने message के
साथ latch कर देता है; device पर हर बाद की call उसी message के साथ fail-fast होती है, जैसा
AMD पर होता है।

---

## Memory

एक `RawBuffer::Cuda` एक device pointer, एक optional host pointer, और अपनी `CudaMemory`
kind रखता है, जिसे `BufferSpec` से चुना जाता है:

| `BufferSpec` | Kind | Driver call |
|---|---|---|
| default | `Device` | `cuMemAlloc` — device memory, कोई host mapping नहीं |
| `cpu_access` | `Managed` यदि device concurrent managed access report करता हो, अन्यथा `Pinned` (WDDM, pre-Pascal) | `cuMemAllocManaged`, एक ही address दोनों ओर valid |
| `host` | `Pinned` | `cuMemHostAlloc(PORTABLE \| DEVICEMAP)`, kernels इसे bus के ऊपर से पढ़ते हैं |

`supports_device_local()` `true` है, इसलिए intermediates device पर ही रहते हैं।
Host <-> device copies पहले context को drain करती हैं (`cuCtxSynchronize`: host access
plan streams के विरुद्ध ordered नहीं है), फिर copy stream पर `cuMemcpyHtoDAsync` /
`cuMemcpyDtoHAsync` के साथ एक lazily allocate किए गए **pinned staging buffer** के माध्यम से
data को 4 MiB chunks में ले जाती हैं, प्रति chunk stream को synchronize करते हुए। Pinned
buffers सीधे `memcpy` कर दिए जाते हैं। Device-to-device `_transfer` `cuMemcpyDtoDAsync` है;
एक allocation के अंदर overlapping range `memmove` semantics बनाए रखने के लिए एक temporary
से होकर गुज़रती है। Free करना पहले drain करता है; यदि drain fail होता है (poisoned context)
तो allocation को एक in-flight kernel के नीचे free करने के बजाय **quarantine** (leak) कर
दिया जाता है। हर compute allocator की तरह यह `LruAllocator` के नीचे बैठता है।

---

## Programs और launches

`CudaProgram::load` PTX text को 16 KiB error और info log buffers के साथ
`cuModuleLoadDataEx` को सौंपता है, ताकि एक JIT failure `Error::CudaJit { kernel, cause, log }`
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
है, message में register और shared memory के आँकड़ों के साथ।

`Program::execute` device की dispatch stream पर launch करता है और वैकल्पिक रूप से उस पर
wait करता है; `execute_timed` launch के इर्द-गिर्द एक timing event pair record करता है और
`cuEventElapsedTime` return करता है, ताकि BEAM candidates को GPU time पर rank करे।

---

## Plan contexts, tokens, timelines

हर execution plan को एक `CudaPlanCtx` मिलता है: **एक non-blocking stream**, जो उसकी lane
है। `dispatch` उस पर launch करता है; `profile` सेट होने पर यह launch को timing events से
घेरता है और एक `CudaDispatchTimestamps` return करता है ([Profiling](./profiling.md))।
`completion_token` एक completion-only event (`CU_EVENT_DISABLE_TIMING`) record करता है
जिसका `wait` `cuEventSynchronize` है और जिसका `retired` `cuEventQuery` है; `synchronize`
`cuStreamSynchronize` है।

Executor की cross-plan ordering `CudaTimelineSignal` का उपयोग करती है, एक timeline जो
events द्वारा publish होती है: `signal(stream, value)` stream की tail पर एक event record
करता है और `(value, event)` फ़ाइल करता है; `value()` हर retired publication को floor में
मोड़ देता है; `wait(target)` उस event पर block करता है जो `>= target` वाली सबसे छोटी value
रखता है; `wait_on_stream` उसके बाद GPU work को `cuStreamWaitEvent` से order करता है। Slots
`Arc`'d हैं ताकि एक waiter अपने event को ज़िंदा रखे जबकि दूसरा thread उसे मोड़कर हटा दे।

---

## Graphs

`CudaGraph::capture` एक captured kernel chain को एक असली **CUDA graph** में बदल देता है:
प्रति kernel एक `cuGraphAddKernelNode_v2` जिसकी dependency list ठीक-ठीक
`GraphKernel::deps` है, यानी host hazard analysis। इसलिए स्वतंत्र kernels device पर overlap
कर सकते हैं (AMD बैकएंड `deps` को छोड़ देता है क्योंकि एक single in-order ring उन्हें
अनावश्यक बना देता है)। हर node के params उसी `extra` protocol के माध्यम से उस kernel के
kernarg blob की ओर इशारा करते हैं जो eager launches में है; graph को
`cuGraphInstantiateWithFlags` से instantiate किया जाता है। Capture एक ख़ाली chain, एक
non-CUDA program, या किसी दूसरे device के program के लिए मना कर देता है (`Ok(None)`)।

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
backend:             nvptx-clang
target_architecture: nvptx64-nvidia-cuda/sm_86
toolchain:           <clang identity>
flags:               -x ir -S -O3 --target=nvptx64-nvidia-cuda -march=sm_86 -Wno-override-module - -o -
abi:                 ptx-kernel-abi-v1;warp-size=32
object_format:       ptx-text-v1
```

Cache **PTX text** store करता है, कभी cubin नहीं: driver उसे load पर assemble करता है और
SASS को अपने ख़ुद के `~/.nv/ComputeCache` में रखता है। हर cache hit को driver तक पहुँचने से
पहले फिर से validate किया जाता है (`validate_ptx`), देखें [Codegen](./codegen.md)।
`SVOD_OBJECT_CACHE=0` cache को disable करता है और `SVOD_OBJECT_CACHE_DIR` उसे स्थानांतरित
करता है।

Device factory (`create_cuda_device`) ऐसे device को भी मना कर देता है जिसकी per-block
shared memory limit optimizer profile की static `shared_max` से कम हो, क्योंकि profile के
हिसाब से sized एक kernel अन्यथा केवल JIT पर fail होता।
