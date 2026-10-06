---
sidebar_label: CPU
---

# CPU बैकएंड

CPU बैकएंड हर कर्नेल को run time पर native machine code में compile करता है और उसे process
के भीतर call करता है। इसके दो code paths हैं, जिन्हें `SVOD_CPU_BACKEND` चुनता है:

| `SVOD_CPU_BACKEND` | Renderer | Compiler | Cache backend |
|---|---|---|---|
| `llvm` (default; unset या खाली होने पर यही चुना जाता है) | LLVM IR text (`codegen/src/llvm/cpu/`) | process में bound libLLVM, या load न हो तो `clang -x ir` | `cpu-llvm-inprocess` / `cpu-llvm-clang` |
| `clang` | C source (`codegen/src/c/`, Clang dialect) | `clang -x c` subprocess | `cpu-clang` |

केवल ये दो spellings और उनके upper-case रूप स्वीकार होते हैं (`CpuBackend::parse`); कुछ भी
और एक warning log करता है और LLVM चुनता है। दोनों paths एक relocatable ELF object बनाते हैं,
जिसे [JIT loader](./jit-loader.md) memory में map और relocate करता है; कोई भी temporary file
नहीं लिखता या shared library को `dlopen` नहीं करता। wiring `runtime/src/devices/cpu.rs` में है।

`"CPU"` device factory बिना शर्त register होती है, और CPU macOS को छोड़कर हर जगह default
डिवाइस है; macOS पर default `METAL:0` है ([बैकएंड](./overview.md) देखें)।

---

## LLVM path

renderer प्रति कर्नेल एक LLVM function emit करता है, buffers `ptr noalias align 32`
parameters के रूप में और scalars `i32` के रूप में, attributes
`nounwind "no-builtins" "no-trapping-math"="true"` के साथ:

```llvm
define void @r_64_32(ptr noalias align 32 %data0, ptr noalias align 32 %data1, i32 %core_id) #0 {
entry:
  ...
  ret void
}
```

### In-process libLLVM

`runtime/src/llvm_inprocess.rs` `libloading` के साथ एक shared libLLVM से LLVM-C API bind
करता है। library प्रति process एक बार, इस क्रम में खोजी जाती है:

1. `SVOD_LLVM_LIB=<path>` — तब वही file एकमात्र candidate है;
2. `llvm-config --libdir` हर candidate नाम के साथ जोड़कर;
3. dynamic loader के अपने search path से हर candidate नाम;
4. macOS पर, `/opt/homebrew/opt/llvm/lib/libLLVM.dylib` और versioned `llvm@N` kegs।

candidate नाम हैं `libLLVM.so` / `libLLVM.dylib`, फिर 30 से 16 तक हर major के लिए distro
SONAMEs (`libLLVM.so.N.1`, `libLLVM-N.so.1`, `libLLVM-N.so`, `libLLVM.so.N`;
`libLLVM-N.dylib`, `libLLVM.N.dylib`)। version `LLVMGetVersion` से पढ़ा जाता है और
**16 या नया** होना चाहिए।

एक compile IR को memory buffer से parse करता है (`LLVMParseIRInContext`), उसे verify करता
है, loop unrolling, loop vectorization और SLP vectorization enabled के साथ
`LLVMRunPasses("default<O2>")` चलाता है, और `LLVMTargetMachineEmitToMemoryBuffer` से object
emit करता है। target machine `<arch>-none-unknown-elf` है, host CPU नाम और feature string
(`LLVMGetHostCPUName` / `LLVMGetHostCPUFeatures`), PIC relocation और default code model के
साथ; macOS aarch64 पर feature string में `+reserve-x18` जुड़ता है। context के handler से
error-severity diagnostics compile को fail करते हैं। हर thread एक thread-local में अपना
`Session` (context, target machine, data layout, pass options) रखता है, इसलिए कर्नेल
concurrently compile होते हैं।

`SVOD_LLVM_INPROCESS=0` (ठीक value `0`) in-process path को disable करता है। library bind
करने में कोई भी विफलता — नहीं मिली, बहुत पुरानी, symbol गायब — नीचे के clang producer पर
fallback करती है; fallback `warn` पर log होता है, जब तक कि variable ने इसे जान-बूझकर disable
न किया हो।

:::note[प्रति process एक libLLVM]
Apple का Metal compiler framework अपना libLLVM `RTLD_GLOBAL` load करता है, जो दूसरी copy के
साथ process साझा नहीं कर सकता। इसलिए CPU और [Metal](./metal.md) बैकएंड एक slot
(`svod_device::claim_inprocess_llvm`) पर मध्यस्थता करते हैं: जो पहले compile करता है वह उसे
रखता है, और दूसरा अपना subprocess या public-API fallback लेता है।
:::

### clang fallback

`compile_ir_to_object_with` IR को clang से pipe करता है, stdin से stdout:

```text
clang -x ir -c -O2 -march=native -fPIC -fno-math-errno -fno-stack-protector \
      -funroll-loops -fvectorize -fslp-vectorize --target=<arch>-none-unknown-elf [-ffixed-x18] - -o -
```

in-process और clang producers की object-cache identities अलग हैं (`cpu-llvm-inprocess`
`library=<path>;version=x.y.z` और host CPU/features के साथ, बनाम `cpu-llvm-clang` clang
identity के साथ), इसलिए objects कभी producers के बीच पार नहीं होते।

---

## clang C path

`SVOD_CPU_BACKEND=clang` इसके बजाय C render करता है — `void name(T* restrict data0, ...,
const int dataN)` `__builtin_*` math के साथ — और उसे इससे compile करता है

```text
clang -c -x c -O2 <cpu flag> -fPIC -ffreestanding -fno-math-errno -fno-stack-protector \
      -nostdlib -fno-ident --target=<arch>-none-unknown-elf [-ffixed-x18] - -o -
```

जहाँ `<cpu flag>` x86_64 और loongarch64 पर `-march=native`, riscv64 पर `-march=rv64g`, और
बाकी जगह `-mcpu=native` है (ARM पर `-march=native` केवल ISA family set करता है)। source
पढ़ने योग्य है, और यही इस path का उद्देश्य है: LLVM IR पढ़े बिना यह देखना कि कर्नेल क्या
करता है।

`clang` केवल `PATH` पर resolve होता है (`ClangToolchain::discover`); कोई `SVOD_CLANG` या
`CC` override नहीं है। object cache के लिए इसकी identity
`path=...;sha256=<binary digest>;version=<clang --version>` है, और C path अतिरिक्त रूप से
`clang -###` से resolved `-target-cpu` और features record करता है (जब कोई flag `native`
कहता है तो `/proc/cpuinfo` से fingerprint किया गया), इसलिए एक machine पर compiled object
दूसरे CPU पर reuse नहीं होता।

:::tip[dlopen fallback]
`svod-runtime` का `dlopen-fallback` cargo feature **केवल C path** के लिए ELF loader को
बदलता है: `clang -shared ... -lm` एक temporary directory में `kernel.so` लिखता है, जिसे
`libloading` खोलता है। यह धीमा है और उन platforms के लिए है जहाँ in-memory loader काम नहीं
करता; object cache इसे `elf-shared-dlopen-v1` के रूप में record करता है, और CI इसे
`test-dlopen-fallback` check के रूप में चलाता है। LLVM path हमेशा in-memory loader उपयोग
करता है।
:::

---

## Math

CPU renderers transcendentals रखते हैं: `exp`, `log`, `sin`, `cos`, `tan`, `pow` आदि
`@llvm.*` intrinsics या `__builtin_*` calls के रूप में render होते हैं। `sqrt`, `fma`,
`floor`, `rint` जैसे host ISA में होने पर instructions में lower होते हैं; बाकी `libm` calls
बन जाते हैं जिन्हें loader load time पर `dlsym(RTLD_DEFAULT)` से resolve करता है।
LLVM path अपने supported ops से `Erf` हटा देता है (वह decompose होता है), C path उसे रखता
है; दोनों `Threefry` और `Max` छोड़ देते हैं, जो एक bare XOR और एक select में decompose होते हैं।

---

## Threads

`SVOD_THREADS` (एक positive integer; default: host का available parallelism) एकमात्र thread
budget है। यह rayon के global pool का आकार तय करता है, जो कर्नेल cache misses को parallel में
compile करता है और CPU कर्नेल चलाता है, और यह हर CPU कर्नेल का default **`core_id` split** है:

- optimizer का THREAD opt (`schedule/src/optimizer/opts.rs`) एक global loop axis को सबसे
  बाहरी स्थिति में ले जाता है और उसे split करता है; heuristic 32, 16, 12, 8, 6, 5, 4, 3 और 2
  chunks आज़माता है, budget और प्रति 131072 elements एक chunk से सीमित, और पहली ऐसी count
  चुनता है जो किसी axis को divide करे (अन्यथा एक को pad करता है)। split कर्नेल और उसकी cache
  identity में baked होता है।
- `gpudims` उस axis को `gidx`/`lidx` के बजाय `[0, N-1]` में एक `core_id` variable में lower
  करता है, और `global_size[0]` `N` बन जाता है।
- run time पर `execute_kernel` देखता है कि `global_size[0] > 1` है और
  `(0..N).into_par_iter()` चलाता है, उसी function pointer को उन्हीं buffers के साथ call करते
  हुए, हर task के लिए `core_id` overwrite करके। हर `core_id` एक disjoint output range लिखता
  है, इसलिए किसी synchronization की ज़रूरत नहीं। किसी मौजूदा rayon worker के भीतर loop nest
  होने के बजाय serially चलता है।

कर्नेल split और pool size अलग हो सकते हैं: 8 तरह split किया गया कर्नेल 4 threads पर सही
चलता है। `SVOD_THREADS=1` split को disable करता है। Rayon अपना global pool एक बार बनाता है,
इसलिए पहले caller का size जीतता है और बाद का अलग request एक warning log करता है
(`ensure_thread_pool`)। `RAYON_NUM_THREADS` नहीं देखा जाता।

---

## कर्नेल call करना

एक loaded कर्नेल `JitKernel` (C path, `ClangKernel` के रूप में re-exported) या
`LlvmKernel` है: mapped object, entry pointer, var names और एक `KernelCif`। ABI ABI slot
क्रम में `void kernel(ptr..., i32...)` है: storage parameters `Type::pointer()` हैं, scalars
`Type::i32()`, और executor जो `i64` values रखता है वे `i32` में truncate होती हैं। call ख़ुद
**libffi** से होकर जाता है (`cif.call(CodePtr(fn_ptr), &args)`, `runtime/src/dispatch.rs`),
इसीलिए कर्नेल signature प्रति arity transmute के बिना प्रति कर्नेल बदल सकता है।

CPU के पास कोई graph factory, कोई plan context और कोई timestamps नहीं हैं: हर कर्नेल एक
synchronous call है, `wait` ignore होता है, और profiler का device-time tier wall-clock है।

---

## Object cache

compiled objects disk पर cache होते हैं (`runtime/src/object_cache.rs`)
`SVOD_OBJECT_CACHE_DIR` के अंतर्गत, अन्यथा `$XDG_CACHE_HOME/svod/objects`, अन्यथा
`~/.cache/svod/objects`; `SVOD_OBJECT_CACHE=0` इसे disable करता है और
`SVOD_OBJECT_CACHE_MAX_BYTES` budget set करता है (default 1 GiB, mtime के अनुसार
oldest-first eviction)। एक entry की key source का SHA-256 और एक `CompilerIdentity`
(`schema`, `backend`, `target_architecture`, `toolchain`, `flags`, `abi`,
`object_format`) है, जो `<hex>.obj` के रूप में एक magic, key, payload digest और payload के
साथ stored है; failed integrity check एक miss है। हर object, cached या fresh, load होने से
पहले host ELF architecture, endianness, object kind और entry symbol के विरुद्ध validate होता
है। यही cache AMD, CUDA और Metal objects को उनकी अपनी identities के तहत रखता है।

---

## Environment variables

| Variable | Default | प्रभाव |
|---|---|---|
| `SVOD_CPU_BACKEND` | `llvm` | `clang` C path चुनता है |
| `SVOD_LLVM_INPROCESS` | on | `0` `clang -x ir` subprocess को मजबूर करता है |
| `SVOD_LLVM_LIB` | unset | bind करने के लिए libLLVM का path; set होने पर एकमात्र candidate |
| `SVOD_THREADS` | host parallelism | thread budget और default `core_id` split |
| `SVOD_OBJECT_CACHE` | on | `0` on-disk object cache को disable करता है |
| `SVOD_OBJECT_CACHE_DIR` | `$XDG_CACHE_HOME/svod/objects` | cache को relocate करता है |
| `SVOD_OBJECT_CACHE_MAX_BYTES` | 1 GiB | cache budget |
| `SVOD_DUMP_LLVM_IR` | unset | वह directory जो हर कर्नेल का rendered LLVM IR प्राप्त करती है |
| `SVOD_DUMP_POST_O2_IR` | unset | वह directory जो clang की `-O2` pipeline के बाद का IR प्राप्त करती है |
| `RUST_LOG` | unset | `svod_runtime=debug` बताता है कि कौन-सा producer compile करता है (libLLVM या clang) और हर कर्नेल compile और load log करता है; clang पर अनचाहा fallback एक `warn` है |
