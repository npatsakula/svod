---
sidebar_label: Codegen
---

# Codegen: NVPTX target

CUDA बैकएंड LLVM text renderer को एक तीसरे target, `LlvmTarget::Nvptx(CudaArch)`
(`LlvmTextRenderer::nvptx(arch)`) के साथ फिर से उपयोग करता है। AMD emitter की तरह,
`codegen/src/llvm/nvptx/` CPU emitter के ऊपर compose करता है: यह उन ops को intercept करता है
जिनके generic LLVM रूप को NVPTX backend select नहीं कर सकता (`Special`, `Barrier`, LOCAL
buffers, `Log2`, `Wmma`, fp8 casts) और बाक़ी सब कुछ (ALU, INDEX, LOAD, STORE, CAST, RANGE)
बिना बदले गुज़र जाने देता है। Lowering table को clang 22 और `ptxas` 13.3 के साथ `sm_86` पर
verify किया गया था।

---

## Module का आकार

```llvm
; ModuleID = 'r_64_32'
source_filename = "r_64_32"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()

define ptx_kernel void @r_64_32(ptr noalias align 32 %data0, ptr noalias align 32 %data1) #0 {
entry:
  ...
  ret void
}

attributes #0 = { nounwind "no-builtins" "no-trapping-math"="true" "nvvm.maxntid"="32" }
```

- अकेला `ptx_kernel` एक `.visible .entry` देता है; किसी `!nvvm.annotations` की ज़रूरत नहीं।
- `target datalayout` clang 22 का `nvptx64` के लिए default है; clang एक mismatch को चुपचाप
  override कर देता है, इसलिए यह line उन tools के लिए मौजूद है जो module को अकेले पढ़ते हैं
  (`opt`, `llvm-as`, IR dumps)।
- `"nvvm.maxntid"` PTX का `.maxntid` **launch bound** है, हर axis पर एक bound: kernel के
  local sizes `nx[, ny[, nz]]` की तरह render होते हैं (`"16,8"` से `.maxntid 16, 8` बनता है;
  अंत की 1 वाली axes हटा दी जाती हैं), और `ptxas` 1024-thread worst case के बजाय उसके
  विरुद्ध प्रति thread registers का बजट बनाता है। bound हर extent का `vmax` है, इसलिए integer
  upper bound वाले symbolic extent को भी एक bound मिलता है; जिस extent का कोई integer bound
  नहीं, वहाँ attribute छोड़ दिया जाता है और hardware maximum लागू होता है। एक पुराना LLVM string attribute को नज़रअंदाज़ कर देता है
  और केवल hint खो देता है।

| Concept | AMD | NVPTX |
|---|---|---|
| triple | `amdgcn-amd-amdhsa` | `nvptx64-nvidia-cuda` |
| kernel ABI | `amdgpu_kernel`, `.kd` descriptor | `ptx_kernel` |
| work ids | `llvm.amdgcn.workgroup.id.*` / `workitem.id.*` | `llvm.nvvm.read.ptx.sreg.ctaid.{x,y,z}` / `tid.{x,y,z}` |
| barrier | `fence syncscope("workgroup")` + `s.barrier` | `fence syncscope("block") release; llvm.nvvm.barrier0; fence syncscope("block") acquire` (`bar.sync 0`) |
| address spaces | global 1, LDS 3, private 5 | `cp.async` / `ldmatrix` casts में shared 3, global 1; kernel parameters generic pointers हैं और REG buffers एक सादे `alloca` ही रहते हैं |
| shared memory | `addrspace(3)` module globals | वही |
| launch bound | `"amdgpu-flat-work-group-size"` | `"nvvm.maxntid"` |

NVPTX work-group scope को `"block"` नाम देता है; `syncscope("workgroup")` reject हो जाता है।
`@llvm.nvvm.barrier0` वह spelling है जिसे हर LLVM release `bar.sync 0` में lower करती है
(नई releases उसे auto-upgrade कर देती हैं)।

---

## Fast math और division

Renderer GPU targets पर ` nsz arcp contract afn ` को घटाकर ` contract ` कर देता है: NVPTX
`fdiv ... arcp afn` को `rcp.approx.f32` में lower करता है, जबकि सादा `contract` सटीक
`div.rn.f32` बनाए रखता है। Tinygrad का CUDA frontend भी सटीक division के साथ compile करता
है।

---

## Transcendentals

NVPTX के पास generic `@llvm.{exp,log,sin,cos,pow}` intrinsics के लिए **कोई lowering नहीं**
है (instruction selection fail हो जाती है) और वह `@llvm.erf` को एक external call के रूप में
emit करता है जो केवल `ptxas` के अंदर fail होती है। इसलिए CUDA renderer wrapper
(`runtime/src/devices/cuda.rs`) `Exp`, `Log`, `Log2`, `Sin`, `Cos`, `Tan`, `Erf`, `Pow`, `Max`
और `Threefry` को अपने `supported_ops` से हटा देता है, और scheduler उन्हें rendering से पहले
decompose करता है। इसका `decompositor` `nvptx_decomposition_patterns()` है: AMD वाला set
(native `exp2`/`log2` के ऊपर polynomial `exp`/`log`/`cos`/`tan`/`pow`, integer-domain bf16
rounding) साथ में f64 `Exp2`/`Log2` expansions, क्योंकि NVPTX `@llvm.exp2` को केवल f16/f32 के
लिए lower करता है। `Sin` और f32/f16 `Log2` `supported_ops` पर keyed साझा transcendental
patterns से होकर जाते हैं; `Erf`, `Max` और `Threefry` को optimizer के अपने passes rewrite
करते हैं (एक polynomial, एक select, पूरा `threefry2x32` mixing)। `Max`, `Pow` और `Threefry`
हर GPU renderer के लिए हटाए जाते हैं, किसी NVPTX-विशिष्ट निर्णय के रूप में नहीं।

जो native रहता है: `@llvm.exp2.f32` `ex2.approx.f32` select करता है, `@llvm.sqrt` `sqrt.rn`
select करता है, `fma`/`floor`/`rint`/`maxnum` सीधे lower होते हैं।

`Log2` जान-बूझकर चुना गया मामला है। `@llvm.log2.f32` की कोई NVPTX lowering नहीं है ("no
libcall available for flog2"); hardware path `@llvm.nvvm.lg2.approx.f` → `lg2.approx.f32`
है, एक 2^-22.6 relative approximation, 1 ulp दूर वहाँ जहाँ AMD का `v_log_f32` और libm सटीक
हैं। Renderer उस lowering को स्पष्ट उपयोग के लिए रखता है (`render_log2`: केवल scalar f32,
f16 उसके इर्द-गिर्द widen होता है, vectors प्रति lane split होते हैं), लेकिन `Log2` को
`supported_ops` से बाहर रखा गया है ताकि सामान्य graphs polynomial `xlog2` path लें और साझा
test tolerances बनाए रखें।

एक बिना-decompose हुआ transcendental जो फिर भी renderer तक पहुँच जाए, वह एक capability-list
drift है; वह render time पर एक `InvalidGraph` error के साथ fail होता है, बजाय इसके कि किसी
ऐसे intrinsic का नाम ले जिसे LLVM चुपचाप एक external call में बदल देता।

PTX में bools predicate registers हैं लेकिन memory में bytes, इसलिए NVPTX का
`extra_matcher` CPU renderer का `bool_storage_patterns` है (tinygrad का `ptx_matcher`)।
Optimizer profile अपनी cache key में `nvptx-decomposition-v1` और `llvm-nvptx-extra-v1`
record करता है ताकि ये विकल्प किसी दूसरे बैकएंड के kernels से कभी न टकराएँ।

---

## Tensor cores: `mma.sync`

`Wmma` एक `@llvm.nvvm.mma.*` intrinsic में lower होता है जिसे
`resolve_mma(arch, in_dtype, acc_dtype, (N, M, K))` चुनता है। हर CUDA profile एक `m16n8kK`
shape है; PTX ISA प्रति row न्यूनतम capability तय करता है:

| Inputs → accumulator | K | Intrinsic suffix | Min |
|---|---|---|---|
| f16 → f32 / f16 | 8 | `m16n8k8.row.col.f32.f32` / `.f16.f16` | `sm_75` |
| f16 → f32 / f16 | 16 | `m16n8k16.row.col.f32.f32` / `.f16.f16` | `sm_80` |
| bf16 → f32 | 16 | `m16n8k16.row.col.bf16` | `sm_80` |
| tf32 (raw f32 bits) → f32 | 8 | `m16n8k8.row.col.tf32` | `sm_80` |
| int8 → int32 | 32 | `m16n8k32.row.col.satfinite.s8` | `sm_80` |
| e4m3 / e5m2 → f32 | 32 | `m16n8k32.row.col.f32.e4m3.e4m3.f32` / `.e5m2...` | `sm_89` |

कोई भी दूसरा tuple, या न्यूनतम से नीचे का arch, `None` return करता है और caller
`InvalidGraph` उठाता है ताकि optimizer ऊपर की ओर decompose करे। Fragments PTX register
split का अनुसरण करते हैं (A 16×K है, B K×8, C/D 16×8, सब 32 lanes पर 32-bit registers में):
f16 operands और f16 accumulators `<2 x half>` जोड़ियों के रूप में जाते हैं, bf16 / tf32 /
int8 / fp8 operands और i32 accumulators `i32` words के रूप में, f32 accumulators `float` के रूप में; aggregate result को WMMA के स्वाभाविक
vector में फिर से जोड़ दिया जाता है। मेल खाती `declare` lines हर call site के operand types
से synthesize होती हैं (`wmma_declaration_from_call`), वही mechanism जो AMD WMMA/MFMA
intrinsics का है।

Typed `CUSTOM` nodes के दो और परिवार इस set को पूरा करते हैं। Warp builders हैं `shfl` और उसके
चार modes — `shfl_bfly(value, lane_mask)` (`llvm.nvvm.shfl.sync.bfly.i32`, एक warp reduction का
butterfly step), `shfl_idx`, `shfl_up` और `shfl_down` — तथा `globaltimer()`
(`llvm.nvvm.read.ptx.sreg.globaltimer`, nanosecond GPU clock)।

Shared-memory builders (`codegen/src/llvm/nvptx/smem.rs`) वही हैं जो एक tile kernel को `.shared`
से होकर stage करने देते हैं, ठीक जैसे AMD पर: `ldmatrix` (sm_75+) एक matrix fragment को उसी
register layout में load करता है जिसकी `mma.sync` अपेक्षा करता है, और `cp_async` / `cp_async_16`
(sm_80+) global → shared copy को registers से बचाकर करते हैं, जिन्हें `cp_async_commit`,
`cp_async_wait` और `cp_async_wait_all` से commit और await किया जाता है। `CpAsyncCache` उसी copy
की cache policy है — `.cg` (केवल L2, 16 bytes) या `.ca` (L1 और L2, 4, 8 या 16)। हर builder अपना
`declare` ख़ुद साथ लाता है; `dedup_declares` प्रति function name पहला वाला रखता है, इसलिए
`cp.async` से भरा एक kernel उस intrinsic को एक ही बार declare करता है।

---

## PTX में compile करना

`compile_ir_to_ptx` (`runtime/src/cuda/compile.rs`) IR को host clang के माध्यम से pipe करता
है, stdin से stdout तक, ठीक वैसे ही जैसे AMD और CPU paths करते हैं:

```text
clang -x ir -S -O3 --target=nvptx64-nvidia-cuda -march=sm_86 --cuda-feature=+ptx78 -Wno-override-module - -o -
```

एक cached `clang --print-targets` probe NVPTX के बिना एक clang को एक साफ़ `JitCompilation`
error में बदल देता है। `SVOD_DUMP_NVPTX_IR=<dir>` हर kernel का IR वहाँ
`sm_XY_<module>.ll` के रूप में लिख देता है।

कोई भी PTX driver तक पहुँचे उससे पहले, चाहे ताज़ा हो या object cache से, `validate_ptx`
जाँचता है कि उसमें एक `.version` है, device के `sm_XY` के बराबर एक `.target` है, एक
`.entry <kernel>(` है, और **कोई `.extern .func` नहीं** है। आख़िरी वाला मायने रखता है: एक
ग़लत लिखा गया `llvm.nvvm.*` नाम compile error नहीं है, LLVM उसे चुपचाप एक external call के
रूप में emit कर देता है, और वह अन्यथा केवल एक `cuModuleLoadDataEx` failure के रूप में सामने
आता।

जब `ptxas` installed हो तो PTX की जाँच यहीं, compile time पर होती है — इसमें ABI के विरुद्ध
उसकी `.param` list भी शामिल है, जिसे एक cubin अब साथ नहीं रखता — और cache में assemble किया
गया cubin ही जाता है; तब एक hit इसके बजाय `validate_cubin` से होकर गुज़रता है (एक
little-endian ELF64, `EM_CUDA` के लिए, जो entry को code के रूप में define करता है)।

PTX ISA version arch के अनुसार pin है, clang पर नहीं छोड़ा जाता, जिसका default उस CUDA
toolkit पर निर्भर है जो उसे मिलता है (CUDA 13 के साथ clang 22: `.version 8.8`, जिसे एक
CUDA 12.9 driver चाहिए; बिना toolkit: किसी भी tensor core के लिए बहुत पुराना version)।
`ptx_isa` (`codegen/src/llvm/nvptx/mod.rs`) capability को उस सबसे पुराने ISA से map करता है
जो उस part को जानता है: sm_88 तक `+ptx78`, sm_89 से लेकर हर 9.x तक `+ptx84` (fp8 `mma.sync`
shapes 8.4 से मौजूद हैं), sm_100 से sm_102 पर `+ptx86`, sm_120 पर `+ptx87`, और हर दूसरे 10.x
तथा नए part पर `+ptx88` (sm_103, sm_110, sm_121, ...)। Clang इससे पुराने को मना कर देता है:
`PTX version 8.4 does not support target 'sm_120'. Minimum required PTX version is 8.7`।
