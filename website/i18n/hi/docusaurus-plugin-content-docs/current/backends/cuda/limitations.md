---
sidebar_label: Limitations और Roadmap
---

# Limitations और Roadmap

बैकएंड अभी तक क्या नहीं करता, source में उसका ठोस कारण, और क्या योजना में है। यहाँ कुछ भी
चुपचाप fail नहीं होता: हर कमी या तो एक साफ़ error है या एक दस्तावेज़ीकृत fallback।

---

## Implement नहीं किया गया

| Gap | आज | कहाँ |
|---|---|---|
| **fp8 conversions** | `FP8E4M3` / `FP8E5M2` से या उसकी ओर एक cast render पर fail होता है (`NVPTX fp8 cast ...`); sm_89 वाले `cvt.*.e4m3x2` intrinsics emit नहीं होते। fp8 `mma.sync` rows `resolve_mma` में मौजूद हैं लेकिन उन्हें feed नहीं किया जा सकता। | `codegen/src/llvm/nvptx/ops.rs` |
| **Stream-ordered frees** | `cuMemFree*` पूरे device को synchronize करता है और इस दौरान हर दूसरे thread की driver call को block कर देता है; `LruAllocator` के नीचे `_free` विरल है, पर `cuMemFreeAsync` bound नहीं है। | `device/src/cuda/allocator.rs` |
| **Peer-to-peer copies** | `cuMemcpyPeerAsync` / `cuDeviceCanAccessPeer` bound नहीं हैं। एक `CUDA:0 → CUDA:1` copy executor में `SyncStrategy::PeerToPeer` लेती है, जो `Buffer::copy_from` पर fall back करती है; दो allocators दो devices हैं, इसलिए bytes एक host `Vec` से होकर उछलते हैं। | `runtime/src/executor.rs`, `device/src/buffer.rs` |
| **Dynamic shared memory** | Launches `shared_mem_bytes = 0` पास करते हैं; केवल static `.shared` उपयोग होती है और `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)` कभी call नहीं होता, इसलिए एक kernel जिसे default per-block limit से ज़्यादा चाहिए वह JIT पर fail होता है। Device factory ऐसे device को पहले ही मना कर देता है जिसकी limit profile की `shared_max` (48 KiB) से नीचे हो। | `device/src/cuda/program.rs`, `runtime/src/devices/cuda.rs` |
| **Hopper / Blackwell matrix paths** | केवल `mma.sync` (`m16n8kK`) lower होता है; कोई `wgmma` नहीं, कोई `tcgen05` नहीं। | `codegen/src/llvm/nvptx/wmma.rs` |
| **`ptxas` के बिना hosts पर cubins** | बिना CUDA toolkit के object cache PTX text store करता है और हर ताज़ा load driver JIT की क़ीमत चुकाता है (driver द्वारा `~/.nv/ComputeCache` में cached)। कोई bundled assembler नहीं है: `ptxas` तभी उपयोग होता है जब वह installed हो (`object_format: cubin-v1`), अन्यथा काम driver करता है। | `runtime/src/cuda/compile.rs` |
| **Userspace NV driver** | Tinygrad के `ops_nv` (सीधी GPU-FIFO submission) को प्रति driver branch एक generated ABI चाहिए; Svod stable `libcuda.so.1` API पर ही रहता है। `SVOD_DEVICE` में `NV` जान-बूझकर स्वीकार *नहीं* किया जाता (केवल `CUDA` और `GPU` हैं); यह नाम उसी भविष्य के बैकएंड के लिए आरक्षित है। | `device/src/registry.rs` |

कमियों के बजाय numerical टिप्पणियाँ: f64 `Exp2` / `Log2` और सभी transcendentals polynomial
path लेते हैं ([Codegen](./codegen.md)); `lg2.approx.f32` renderer के लिए उपलब्ध है लेकिन
सामान्य graphs उसका उपयोग नहीं करते।

---

## वे आवश्यकताएँ जिन पर आज कोई समझौता नहीं

- Driver कम से कम CUDA 12.0 / R525 होना चाहिए: CUDA graph के entry points अपने 12.0
  versioned नामों से bind हैं, और पुराना driver बैकएंड को चुपचाप unregistered छोड़ देता है। PTX ISA का pin (`--cuda-feature`) compute capability का
  अनुसरण करता है — sm_88 तक 7.8, sm_89 और sm_90 पर 8.4 (CUDA 12.4 / R550), फिर Blackwell भर में 8.6,
  8.7 और 8.8 (CUDA 12.9 तक) — यानी कोई Blackwell part floor को अपनी ही ISA वाले driver तक उठा
  देता है, पर नया clang उसे नहीं बढ़ाता।
- `clang` में NVPTX target होना चाहिए; कोई NVRTC fallback नहीं है।

---

## Roadmap

जो बचा है, प्राथमिकता के क्रम में:

1. **Stream-ordered frees**: device memory के लिए copy lane पर `cuMemFreeAsync`, ताकि एक
   free device को drain करना बंद कर दे।
2. **असली P2P**: `cuDeviceCanAccessPeer` / `cuCtxEnablePeerAccess` / `cuMemcpyPeerAsync` को
   bind करना और `SyncStrategy::PeerToPeer` को उनके माध्यम से route करना।
3. **fp8**: sm_89 वाले `cvt` intrinsics को lower करना ताकि fp8 `mma.sync` rows तक पहुँचा जा
   सके, और `for_cuda_arch` को sm_89 profile बनाने देना।
4. `cuFuncSetAttribute` के माध्यम से **Dynamic shared memory**, ताकि एक kernel default
   48 KiB per-block limit को पार कर सके।

Scoped synchronization और CUPTI hardware counters बाक़ी दो items थे; दोनों आ चुके हैं
([Architecture](./architecture.md), [Profiling](./profiling.md))।
