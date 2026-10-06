---
sidebar_label: अवलोकन
---

# बैकएंड

बैकएंड वह सब कुछ है जो rendered कर्नेल के नीचे है: एक renderer जो UOp IR को source में
बदलता है, एक compiler जो source को object में बदलता है, एक loader जो object को callable
`Program` में बदलता है, एक allocator, और वैकल्पिक रूप से एक graph। Svod चार बैकएंड ship
करता है, सभी एक ही binary में; किसी दिए गए host पर कौन-से मौजूद हैं, यह run time पर तय होता है।

| डिवाइस | Hardware | Renderer | Compile path | Graph replay | स्थिति |
|---|---|---|---|---|---|
| [`CPU`](./cpu.md) | x86_64, aarch64, riscv64, loongarch64, ppc64le | LLVM IR text (default) या C | process में libLLVM, अन्यथा `clang -c`; [in-memory ELF loader](./jit-loader.md) | कोई नहीं (synchronous calls) | production |
| [`AMD:N`](./amd/overview.md) | CDNA3, RDNA3, RDNA3.5, RDNA4 (Linux, KFD) | LLVM IR text, AMDGPU target | `amdgcn` target के साथ `clang` → VRAM में load किया गया ELF code object | AQL command stream (PM4 opt-in) | production |
| [`CUDA:N`](./cuda/overview.md) | NVIDIA, driver CUDA 12.0+ | LLVM IR text, NVPTX target | NVPTX target के साथ `clang` → PTX, installed होने पर `ptxas`, अन्यथा driver JIT | CUDA graphs | production |
| [`METAL:N`](./metal.md) | Apple GPUs | C, Metal dialect | process में Apple का `MTLCodeGenService` → metallib | indirect command buffers | Apple9 / macOS 26 पर validated |

हर बैकएंड के compiled objects एक ही on-disk object cache से गुज़रते हैं, जिसकी key source
और प्रति-बैकएंड `CompilerIdentity` है ([CPU पेज](./cpu.md))।

---

## डिवाइस चुनना

`SVOD_DEVICE` tensors और कर्नेल के लिए default डिवाइस चुनता है। value को case-insensitive
रूप से `NAME[:N]` के तौर पर parse किया जाता है (`dtype/src/default_device.rs`):

| Value | डिवाइस |
|---|---|
| `CPU` | `DeviceSpec::Cpu` |
| `AMD[:N]`, `HIP[:N]` | `DeviceSpec::Amd { device_id }` — KFD topology का N-वाँ GPU node |
| `CUDA[:N]`, `GPU[:N]` | `DeviceSpec::Cuda { device_id }` |
| `METAL[:N]` | `DeviceSpec::Metal { device_id }` (केवल `0` मौजूद है) |

अकेला `NAME` डिवाइस 0 है। `NV` जान-बूझकर reject किया जाता है — यह नाम भविष्य के एक userspace
NVIDIA driver के लिए आरक्षित है। जब कुछ भी डिवाइस नहीं चुनता, तो platform default लागू होता
है: **macOS पर `METAL:0`, बाकी हर जगह `CPU`**। पूरी precedence यह है: एक
`with_default_device` scope, फिर thread-local `set_default_device`, फिर `SVOD_DEVICE` (प्रति
process एक बार पढ़ा जाता है), फिर platform default। GPU arch कभी spec का हिस्सा नहीं होता:
यह खोले गए डिवाइस की property है, इसलिए एक physical GPU की एक ही identity होती है और कर्नेल
cache की key वही है जो डिवाइस report करता है।

`svod-device` में `DeviceSpecExt::parse` यही spellings स्वीकार करता है, साथ में
`DISK:<path>` (एक read-only, memory-mapped file डिवाइस जो कर्नेल नहीं चला सकता) और
`WEBGPU`, जिसका अभी कोई allocator नहीं है और जो `DeviceUnavailable` के साथ fail होता है।

---

## Runtime-detected registration

हर बैकएंड हर host पर compile होता है — AMD, CUDA या Metal के लिए कोई cargo feature नहीं है।
CUDA और Metal bindings `libloading` के ऊपर plain Rust हैं और हर जगह compile होती हैं; AMD
के kernel-facing modules `cfg(unix)` हैं। इसलिए Linux या macOS पर `cargo check` उन सभी को
type-check करता है। कोई बैकएंड *उपलब्ध* है या नहीं,
यह तब तय होता है जब device factory registry को पहली बार छुआ जाता है
(`runtime/src/device_registry.rs`):

```rust
registry.register_factory("CPU", ...);                        // always
if svod_device::amd::has_devices()   { registry.register_factory("AMD", ...); }
if svod_device::metal::has_devices() { registry.register_factory("METAL", ...); }
if svod_device::cuda::has_devices()  { registry.register_factory("CUDA", ...); }
```

हर probe side-effect-free और memoized है: AMD KFD sysfs topology पढ़ता है और पूछता है कि
क्या कोई node supported arch है; Metal Apple frameworks को `dlopen` करता है और system
default device माँगता है; CUDA `libcuda.so.1` load करता है, अपने उपयोग के हर entry point को
bind करता है, `cuInit` call करता है और डिवाइस गिनता है। बिना hardware वाले host पर बस वह
device type नहीं होता, और उसे माँगना `UnsupportedDevice` के साथ fail होता है। सब कुछ हर जगह
compile करने का मकसद यह है कि साझा `Program` / `PlanContext` / `Graph` traits में बदलाव
किसी भी developer machine पर build तोड़ दे, केवल GPU वाली machine पर नहीं।

registry प्रति `DeviceSpec` एक `Device` cache करती है (`DEVICE_FACTORIES`); construction —
KFD खोलना, toolchain probe करना — map locks के बाहर चलता है, प्रति spec serialized, और
failed construction slot को retry के लिए खाली छोड़ देता है। Allocators `svod-device` में एक
अलग registry (`registry::registry()`) में रहते हैं, जहाँ हर compute allocator एक
`LruAllocator` में wrap होता है जो freed buffers को size और spec के अनुसार pool करता है।

---

## बैकएंड क्या implement करता है

एक `Device` (`device/src/device.rs`) के पाँच हिस्से हैं:

```rust
pub struct Device {
    pub device: DeviceSpec,
    pub allocator: Arc<dyn Allocator>,
    pub compilers: Vec<CompilerPair>,     // (Arc<dyn Renderer>, Arc<dyn Compiler>)
    pub renderer: Arc<dyn Renderer>,
    pub compiler: Arc<dyn Compiler>,
    pub runtime: RuntimeFactory,          // Fn(&CompiledSpec) -> Result<Box<dyn Program>>
    pub graph: Option<GraphFactory>,      // Fn(&[GraphKernel]) -> Result<Option<Box<dyn Graph>>>
}
```

| Trait | आवश्यक | भूमिका |
|---|---|---|
| `Renderer` | `render`, `device`, `supported_ops` | UOp graph → `ProgramSpec` (source, entry, ABI, launch sizes)। `gpu_arch` optimizer profile चुनता है; `decompositor` और `extra_matcher` वह lower करते हैं जिसे target select नहीं कर सकता |
| `Compiler` | `compile`, `cache_key` | `ProgramSpec` → `CompiledSpec` bytes; `cache_key` वह `CompilerIdentity` है जो object cache की key है |
| `RuntimeFactory` | — | `CompiledSpec` को `Program` में load करता है; `Device::new` इसे wrap करता है ताकि हर spec की stage identity पहले validate हो |
| `Program` | `execute`, `name` | एक कर्नेल launch; `execute_timed` (BEAM के लिए GPU-clock duration), `new_exec_context`, `resource_usage` और `as_any` वैकल्पिक हैं |
| `PlanContext` | `dispatch`, `synchronize` | `Program::new_exec_context` द्वारा बनाया गया प्रति-plan state: lanes, completion tokens, timestamps, counters (`set_pmc`), native linked replay (`replay_linked_plan`) |
| `Allocator` | `_alloc`, `name`, `device_spec` | `_copyin` / `_copyout` / `_transfer` / `_free` / `synchronize` / `supports_device_local` वैकल्पिक हैं और default रूप से host-memory semantics रखते हैं |
| `Graph` | `replay` | एक captured कर्नेल chain जो एक submission से replay होती है; `completion_token`, `replay_profiled` वैकल्पिक |
| `CompletionToken`, `TimelineSignal`, `DispatchTimestamps` | | synchronization और profiling handles जिन्हें executor उपयोग करता है (`device/src/sync.rs`) |

launch convention हर GPU बैकएंड में साझा है: `global_size` वर्क-ग्रुप्स में grid है,
`local_size` threads में वर्क-ग्रुप; CPU `global_size[0]` को `core_id` split के रूप में
उपयोग करता है। कर्नेल arguments क्रम में ABI के `PARAM` slots हैं — pointers, फिर `i32`
scalars — जिन्हें हर loader एक ही तरह pack करता है (AMD और CUDA पर `ClikeKernargLayout`,
Metal पर positional `setBuffer`/`setBytes`, CPU पर एक libffi CIF)।

### नया बैकएंड जोड़ना

`runtime/src/devices/` में चार factories template हैं। हर `create_*_device` वही पाँच काम
करता है:

1. अपने `DeviceSpec` के लिए registry से allocator लेता है;
2. एक codegen entry point के चारों ओर renderer wrapper बनाता है
   (`LlvmTextRenderer::amd(arch)`, `LlvmTextRenderer::nvptx(arch)`,
   `CRenderer::metal()`, या CPU renderers), जो `supported_ops`, decomposition patterns और
   `gpu_arch` declare करता है;
3. एक `CompilerIdentity` और `ObjectCache` के साथ compiler बनाता है, जो ऐसे bytes बनाता है
   जिन्हें loader validate कर सके (`validate_amd_object`, `validate_ptx` /
   `validate_cubin`, `validate_metallib`, CPU पर ELF checks);
4. एक `RuntimeFactory` install करता है जो उन bytes को बैकएंड के `Program` में load करता है;
5. वैकल्पिक रूप से capture/replay के लिए `with_graph(...)`।

फिर यह factory को अपने device-type string के तहत register करता है, एक `has_devices()`
probe पर gated, और scheduler को नए target के लिए एक optimizer profile चाहिए
(`OptimizerRenderer::for_*`: wave size, tensor-core shapes, shared-memory और local limits)।
`create_*_codegen` हर बैकएंड पर अलग से मौजूद है ताकि BEAM workers डिवाइस खोले बिना render
और compile कर सकें।
