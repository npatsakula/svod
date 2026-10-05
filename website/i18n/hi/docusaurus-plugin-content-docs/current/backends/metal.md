---
sidebar_label: Metal
---

# Metal बैकएंड

Svod Metal के माध्यम से Apple GPUs पर चलता है। बैकएंड सीधे Objective-C runtime के विरुद्ध
लिखा गया है: `libobjc`, `Metal.framework` और private `MTLCompiler.framework` run time पर
`dlopen` किए जाते हैं और हर call एक hand-declared C signature के साथ `objc_msgSend` है
(`device/src/metal/objc.rs`)। कोई `objc2` या `metal` crate नहीं, कोई cargo feature नहीं और
कोई `cfg(target_os)` gate नहीं: module हर host पर compile और type-check होता है, और Linux
machine बस `dlopen` में fail होती है और डिवाइस कभी register नहीं करती। कर्नेल C renderer की
Metal dialect (`codegen/src/c/metal.rs`) द्वारा Metal Shading Language के रूप में render होते
हैं और process में एक metallib में compile होते हैं।

code `device/src/metal/` (device, allocator, compile, program, graph, Metal 4 profiler),
`runtime/src/devices/metal.rs` (device factory) और `codegen/src/c/metal.rs` (dialect) में
रहता है।

---

## स्थिति

बैकएंड सितंबर 2026 में आया और इसे एक hardware family पर परखा गया है: macOS 26 के अंतर्गत
**Apple9** (M3/M4 class), जहाँ tensor suite, ONNX suite और `tk` hardware tests green हैं,
और flash attention व GEMM `simdgroup_matrix` के माध्यम से चलते हैं। पुराने systems के लिए
लिखे गए paths (public `newLibraryWithSource:` compile fallback, pre-Apple9
indirect-command-buffer workaround, `metal3.x` / `metal2.0` language standards) implement
किए गए हैं पर ऐसे hardware पर validated नहीं हैं। कोई Metal डिवाइस मौजूद न होने पर hardware
tests ख़ुद skip हो जाते हैं, इसलिए Linux CI केवल host-side tests चलाता है।

केवल system default डिवाइस (`METAL:0`) supported है; `MTLCopyAllDevices` enumeration एक
follow-up है (`device/src/metal/device.rs`)।

---

## डिवाइस चुनना

`METAL[:N]` एकमात्र spelling है (`HIP`-style aliases AMD और CUDA के लिए मौजूद हैं, Metal के
लिए कोई नहीं)। macOS पर यह **platform default** भी है: जब कुछ भी डिवाइस नहीं चुनता तो
`default_device()` `METAL:0` पर resolve होता है, इसलिए Mac पर CPU बैकएंड पर वापस जाने का
तरीका `SVOD_DEVICE=CPU` है (`dtype/src/default_device.rs`)।

```bash
SVOD_DEVICE=METAL:0 cargo run --release -p svod-model --example gigaam_infer -- ./audio.wav
```

`svod_device::metal::has_devices()` Objective-C runtime load करता है और
`MTLCreateSystemDefaultDevice` call करता है; runtime की device registry `"METAL"` factory को
केवल तभी register करती है जब यह सफल हो। खोलने पर डिवाइस नाम और GPU family के साथ एक `info`
line log होती है (`RUST_LOG=svod_device=info`)।

family को `supportsFamily:` से Apple12 से नीचे Apple1 तक, फिर Mac2, probe किया जाता है, और
`MetalFamily { Unknown, Mac2, Apple(n) }` के रूप में रखा जाता है। यह renderer का
`gpu_arch` है, object cache की key बनाता है और optimizer profile चुनता है
(`OptimizerRenderer::for_metal_family`): `simdgroup_matrix` tensor cores को Apple7 या नया
चाहिए।

---

## Codegen: MSL dialect

`CRenderer::metal()` `CDialect::Metal` के साथ CPU C renderer है; इसके अस्तित्व से Clang
output नहीं बदलता। एक कर्नेल इस तरह render होता है

```c
#include <metal_stdlib>
using namespace metal;

kernel void r_64_32(device float* data0, device float* data1, constant int& data2,
                    uint3 gid [[threadgroup_position_in_grid]],
                    uint3 lid [[thread_position_in_threadgroup]]) {
  threadgroup __attribute__((aligned(16))) float local0[32];
  ...
}
```

| अवधारणा | Clang dialect | Metal dialect |
|---|---|---|
| buffer parameter | `float* restrict data0` | `device float* data0` |
| scalar parameter | `const int data2` | `constant int& data2` |
| launch ids | `core_id` variable | `gid.xyz` (`gidx*` / `idx*`), `lid.xyz` (`lidx*`), PARAM list के बाद जोड़े गए |
| local buffer | stack array | `threadgroup __attribute__((aligned(16))) T localN[size]` |
| barrier | कोई नहीं | `threadgroup_barrier(mem_flags::mem_threadgroup)` |
| address spaces | कोई नहीं | pointer casts पर `device` / `threadgroup` / `thread` |
| 16-bit floats | `_Float16` | `half`, `bfloat` |
| bitcast | union / memcpy | `as_type<T>()` |

कोई `[[buffer(n)]]` attributes नहीं हैं: Metal arguments को **positionally** bind करता है,
इसलिए किसी parameter का binding index signature में उसकी स्थिति है, जिसे loader mirror करता
है (नीचे)। अधिकतम तीन grid axes होते हैं; scheduler अतिरिक्त global axes को fold करता है
(Metal optimizer profile में `global_max`)।

**Types.** Float64, हर fp8 format और 4 से चौड़े vectors render पर reject होते हैं
(`reject_unsupported_metal_dtypes`, `codegen/src/c/types.rs`); scheduler पहले ही internal
f64 को f32 में demote कर देता है। bf16 arithmetic `float` के माध्यम से promote होता है, और
bf16 narrowing integer round-to-nearest-even pattern set उपयोग करता है।

**Math.** `sqrt`, `exp2` और `log2` native हैं; `sin` `precise::sin` के रूप में render होता
है; `exp`, `log`, `cos`, `tan` और `erf` (MSL में `erf` नहीं है) साझा
`amd_decomposition_patterns()` द्वारा native `exp2`/`log2` के ऊपर decompose होते हैं, जैसे
AMD पर। renderer का `extra_matcher` CPU वाला है (`cpu_extra_matcher()`)। Fast math हर जगह
off है (`-fno-fast-math`, या public path पर `MTLMathModeSafe`) ताकि साझा test tolerances
बनी रहें।

**Tensor cores.** `Wmma` `simdgroup_<T>8x8` और `simdgroup_multiply_accumulate` के ऊपर एक
प्रति-shape helper में lower होता है: एक shape, 32 threads पर 8×8×8, प्रति लेन दो elements,
f32→f32, f16→f32, f16→f16, bf16→f32 और bf16→bf16 के लिए (optimizer profile में
`METAL_888`)। `tk` `simd_shuffle`, `simd_shuffle_xor` और `simdgroup_barrier` builders जोड़ता
है (`codegen/src/c/metal.rs`), और Apple flash-attention व GEMM कर्नेल इन्हीं से बने हैं।

---

## Compile path

`compile_msl` (`device/src/metal/compile.rs`) source को Apple के private
`MTLCodeGenService` को भेजता है — वही path जो tinygrad उपयोग करता है — और एक hand-built
Objective-C block callback के माध्यम से metallib (`MTLB` magic, `ENDT` trailer) प्राप्त करता
है, 60 s timeout और एक समय में एक request के साथ। flags हैं

```text
-fno-fast-math -std=<std> --driver-mode=metal -x metal -fno-caret-diagnostics
-fmodules-cache-path=<cache>/metal-modules
```

जहाँ `<std>` macOS major version का अनुसरण करता है (26+ पर `metal4.0`, 14–25 पर
`metal3.1`, 13 पर `metal3.0`, उससे पहले `macos-metal2.0`) और module cache `metal_stdlib`
parse को लगभग 250 ms से लगभग 8 ms कर देता है।

`MTLCompiler.framework` अपना libLLVM `RTLD_GLOBAL` load करता है, जो CPU बैकएंड के
in-process libLLVM के साथ coexist नहीं कर सकता, इसलिए दोनों एक slot
(`claim_inprocess_llvm`) के लिए होड़ करते हैं। उस race का हारने वाला, या private framework
के बिना system, `compile_msl_public` लेता है: एक `newLibraryWithSource:options:error:`
compile ताकि diagnostics सामने आएँ, जिसके बाद **MSL source ख़ुद** payload है और program
loader उसे load पर फिर से compile करता है। दोनों payloads एक object-cache entry साझा करते हैं:

```text
backend:             metal
target_architecture: Apple9/air64
toolchain:           macos=26.0
flags:               -fno-fast-math -std=metal4.0 --driver-mode=metal -x metal -fno-caret-diagnostics
abi:                 msl-kernel-abi-v1
object_format:       metallib-or-msl-v1
```

transport जान-बूझकर identity से अनुपस्थित है: एक BEAM worker जिसने libLLVM slot जीता और
उसका parent जो हार गया, दोनों को key पर सहमत होना चाहिए।

---

## Programs और launches

`MetalProgram::load` कोई भी payload स्वीकार करता है (metallib के लिए
`newLibraryWithData:`, MSL के लिए `newLibraryWithSource:`), function को
`newFunctionWithName:` से bind करता है और pipeline को
`setSupportIndirectCommandBuffers:YES` के साथ बनाता है, `maxTotalThreadsPerThreadgroup`,
`threadExecutionWidth` और `staticThreadgroupMemoryLength` पढ़ते हुए।

Arguments स्थिति के अनुसार bind होते हैं: buffers `setBuffer:offset:atIndex:` से — एक host
pointer डिवाइस के `PointerRegistry` के माध्यम से अपने `(MTLBuffer, offset)` पर resolve होता
है, जो buffer base से keyed एक `BTreeMap` है — और scalars `setBytes` से 4-byte `i32` के रूप
में; `i32` से बाहर की value एक run-time error है। ABI slots बढ़ते क्रम में होने चाहिए और
उनमें gaps हो सकते हैं, 31 bindings तक (`MAX_BUFFER_BINDINGS`)। `global_size` threadgroup
count है और `local_size` प्रति group threads, जो
`dispatchThreadgroups:threadsPerThreadgroup:` से भेजे जाते हैं; बिना local axes वाला कर्नेल
प्रति group एक thread चलाता है, और `maxTotalThreadsPerThreadgroup` से बड़ा group reject होता
है।

हर dispatch एक command buffer है, कर्नेल नाम से labelled, depth 1024 की एक single queue पर।
`wait = false` के साथ यह डिवाइस की `in_flight` list में जुड़ता है;
`MetalDevice::synchronize` हर entry का `waitUntilCompleted` से इंतज़ार करता है और पहला
`NSError` सामने लाता है। किसी buffer तक host access पहले डिवाइस को drain करता है।
`execute_timed` command buffer का `GPUStartTime` / `GPUEndTime` पढ़ता है, और इसी पर BEAM
candidates को rank करता है।

---

## Memory

हर allocation `MTLResourceStorageModeShared` में एक `MTLBuffer` है: Apple silicon unified
memory है, इसलिए `BufferSpec` flags ignore होते हैं और `copyin` / `copyout` / `_transfer`
एक `synchronize()` के बाद `contents` पर host `memcpy` / `memmove` हैं। कोई private या
managed buffers नहीं और कोई blit encoders नहीं। एक free पहले डिवाइस को drain करता है; यदि
drain fail हो तो allocation को in-flight कर्नेल के नीचे release करने के बजाय leak कर दिया
जाता है।

---

## Graphs

`MetalGraph::capture` एक कर्नेल chain को `ConcurrentDispatch` commands के एक
`MTLIndirectCommandBuffer` में record करता है, हर एक `setBarrier` के साथ, ताकि capture क्रम
बना रहे। एक replay एक command buffer है: bound buffers पर `useResources:count:usage:`, फिर
`executeCommandsInBuffer:withRange:`। `replay` पिछले replay का इंतज़ार करता है और केवल उन
slots को rebind करता है जिनका buffer बदला। Capture मना करता है (`Ok(None)`, बदले में प्रति-call
dispatch) खाली chain, ऐसे program जो `MetalProgram` नहीं है, ऐसे डिवाइस जिसके नाम में
"virtual" है (paravirtualized CI GPUs ICBs को तोड़ते हैं), 32 bits से बड़े offset, या **किसी
भी scalar argument** के लिए — symbolic shapes वाली chain graph नहीं होती। Apple9 से नीचे
tinygrad का `FIX_METAL_ICB` workaround (प्रति pipeline एक खाली dispatch) लागू होता है।

---

## Profiling

| Tier | Metal पर | स्रोत |
|---|---|---|
| 1 — device time | हाँ | प्रति command buffer `GPUStartTime` / `GPUEndTime`; graph के भीतर, एक Metal 4 counter heap (macOS 26+) या प्रति कर्नेल एक command buffer |
| 2 — roofline | हाँ | backend-neutral |
| 3 — static resources | आंशिक | `staticThreadgroupMemoryLength` से `lds_bytes`, `threadExecutionWidth` से `wave_size`, `occupancy` `maxTotalThreadsPerThreadgroup / 1024` के रूप में; कोई register counts नहीं |
| 4 — hardware counters | नहीं | |

`Mtl4Profiler` (`device/src/metal/mtl4.rs`) केवल profiled graph replay के लिए मौजूद है: यह हर
indirect command को अकेले एक Metal 4 command buffer में दो precise timestamps के बीच चलाता
है, bound buffers पर एक residency set और एक shared event wait के साथ। एक MTL4 encoder चुपचाप
उस indirect command का पहला execution skip कर देता है जिसकी pipeline उसे नहीं दी गई थी, इसलिए
हर pipeline पहले encoder पर set की जाती है।

---

## सीमाएँ

- एक डिवाइस (`METAL:0`), एक command queue, केवल shared storage;
- scalars `i32` हैं; कोई f64 नहीं, कोई fp8 नहीं, 4 से चौड़े vectors नहीं;
- एक tensor-core shape (`simdgroup` 8×8×8);
- graphs scalar arguments वाली chains को बाहर रखते हैं;
- कोई hardware counters नहीं, कोई register counts नहीं, और graph के बाहर प्रति-dispatch timing
  पूरे command buffer का stamp है;
- fast path Apple का undocumented `MTLCodeGenService` है; public API fallback है।

कोई Metal-specific environment variables नहीं हैं। साझा वाले लागू होते हैं: `SVOD_DEVICE`,
`SVOD_OBJECT_CACHE` / `SVOD_OBJECT_CACHE_DIR`, module cache के लिए `XDG_CACHE_HOME`, और
graph capture व decline messages के लिए `RUST_LOG=svod_device=debug`।

---

## Tests

```bash
cargo test -p svod-device metal          # host tests everywhere; hardware tests self-skip
cargo test -p svod-codegen metal         # MSL golden tests
SVOD_DEVICE=METAL:0 cargo test -p svod-tensor   # codegen_tests! `metal` variants
SVOD_DEVICE=METAL:0 cargo test -p svod-onnx
```
