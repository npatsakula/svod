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
| **Scoped synchronization** | Host reads और writes केवल buffer के producers का wait करने के बजाय पूरे context को drain करते हैं (`_copyin` / `_copyout` में `cuCtxSynchronize`)। Plans और graphs event-आधारित `CompletionToken` ज़रूर सौंपते हैं। | `device/src/cuda/allocator.rs` |
| **Peer-to-peer copies** | `cuMemcpyPeerAsync` / `cuDeviceCanAccessPeer` bound नहीं हैं। एक `CUDA:0 → CUDA:1` copy executor में `SyncStrategy::PeerToPeer` लेती है, जो `Buffer::copy_from` पर fall back करती है; दो allocators दो devices हैं, इसलिए bytes एक host `Vec` से होकर उछलते हैं। | `runtime/src/executor.rs`, `device/src/buffer.rs` |
| **Hardware counters (Tier 4)** | कोई CUPTI नहीं; `pmc_available()` `false` है, `SVOD_PMC` एक नोट के साथ घट जाता है। | [Profiling](./profiling.md) |
| **Tile kernels (`tk`)** | केवल AMD: `resolve_arch` एक CUDA spec के लिए कोई `AmdArch` नहीं देता, इसलिए एक `tk` launch `UnsupportedArch` report करता है। | `tk/src/target.rs` |
| **Dynamic shared memory** | Launches `shared_mem_bytes = 0` पास करते हैं; केवल static `.shared` उपयोग होती है और `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)` कभी call नहीं होता, इसलिए एक kernel जिसे default per-block limit से ज़्यादा चाहिए वह JIT पर fail होता है। Device factory ऐसे device को पहले ही मना कर देता है जिसकी limit profile की `shared_max` (48 KiB) से नीचे हो। | `device/src/cuda/program.rs`, `runtime/src/devices/cuda.rs` |
| **Hopper / Blackwell matrix paths** | केवल `mma.sync` (`m16n8kK`) lower होता है; कोई `wgmma` नहीं, कोई `tcgen05` नहीं। | `codegen/src/llvm/nvptx/wmma.rs` |
| **Pre-assembled objects** | Object cache PTX text store करता है; हर ताज़ा load driver JIT की क़ीमत चुकाता है (driver द्वारा `~/.nv/ComputeCache` में cached)। एक `ptxas` pre-assembly (`object_format: cubin`) wire नहीं की गई है। | `runtime/src/devices/cuda.rs` |
| **Userspace NV driver** | Tinygrad के `ops_nv` (सीधी GPU-FIFO submission) को प्रति driver branch एक generated ABI चाहिए; Svod stable `libcuda.so.1` API पर ही रहता है। `SVOD_DEVICE` में `NV` को `CUDA` के alias के रूप में स्वीकार किया जाता है और उसे उसी भविष्य के बैकएंड के लिए आरक्षित रखा गया है। | `nvidia_backend_plan.md` |

कमियों के बजाय numerical टिप्पणियाँ: f64 `Exp2` / `Log2` और सभी transcendentals polynomial
path लेते हैं ([Codegen](./codegen.md)); `lg2.approx.f32` renderer के लिए उपलब्ध है लेकिन
सामान्य graphs उसका उपयोग नहीं करते।

---

## वे आवश्यकताएँ जिन पर आज कोई समझौता नहीं

- Driver कम से कम CUDA 12.0 / R525 होना चाहिए: CUDA graph के entry points अपने 12.0
  versioned नामों से bind हैं। PTX ISA `--cuda-feature=+ptx78` से 7.8 पर pin है, इसलिए
  नया clang यह floor नहीं बढ़ाता।
- `clang` में NVPTX target होना चाहिए; कोई NVRTC fallback नहीं है।

---

## Roadmap

उसी क्रम में जिसमें plan (`nvidia_backend_plan.md`, phase 5) उन्हें सूचीबद्ध करता है:

1. **Scoped sync**: प्रति buffer एक producer table ताकि host access `cuCtxSynchronize` के
   बजाय events का wait करे।
2. **असली P2P**: `cuDeviceCanAccessPeer` / `cuCtxEnablePeerAccess` / `cuMemcpyPeerAsync` को
   bind करना और `SyncStrategy::PeerToPeer` को उनके माध्यम से route करना।
3. **CUPTI counters**: `PmcCounter` को AMD SQ set से आगे चौड़ा करना और एक Tier 4 provider
   जोड़ना।
4. **fp8**: sm_89 वाले `cvt` intrinsics को lower करना ताकि fp8 `mma.sync` rows तक पहुँचा जा
   सके।
5. Toolkit मौजूद होने पर **`ptxas` pre-assembly**, एक cubin के रूप में cached।
6. `cuFuncSetAttribute` के माध्यम से **Dynamic shared memory**, और `GpuArch` के ऊपर `tk`
   ताकि tile kernels CUDA पर चलें।
