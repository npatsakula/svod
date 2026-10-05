---
sidebar_label: Debugging
---

# Debugging

बैकएंड की तीन failure surfaces हैं: host toolchain (clang, object cache), driver JIT
(`cuModuleLoadDataEx`), और run time पर device (driver errors, memory faults)। यह पेज उन
knobs की सूची देता है और बताता है कि हर error को कैसे पढ़ा जाए।

---

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `SVOD_DEVICE` | `CPU` (macOS पर `METAL:0`) | `CUDA:N` (alias `GPU`) default tensor device चुनता है; `NV` alias *नहीं* है, वह एक userspace driver के लिए सुरक्षित रहता है |
| `SVOD_DUMP_NVPTX_IR` | unset | वह directory जिसमें हर kernel का NVPTX LLVM IR `sm_XY_<kernel>.ll` के रूप में जाता है |
| `SVOD_CUDA_PTXAS` | on | `0` `ptxas` वाली pre-assembly छोड़ देता है और PTX text को driver JIT को सौंप देता है |
| `SVOD_CUDA_SCOPED_SYNC` | on | `0` हर scoped wait को एक पूरे `cuCtxSynchronize` से बदल देता है और copies को synchronous बना देता है ([Architecture](./architecture.md)) |
| `SVOD_CUDA_CUPTI` | on | `0` `libcupti.so.13` को load करना छोड़ देता है, इसलिए कोई hardware counters नहीं होते |
| `SVOD_PMC` | unset | बैकएंड के default counter set के लिए `1`, या एक comma-separated token list, देखें [Profiling](./profiling.md) |
| `SVOD_OBJECT_CACHE` | on | `0` compiled objects की on-disk cache को disable करता है |
| `SVOD_OBJECT_CACHE_DIR` | `$XDG_CACHE_HOME/svod/objects`, अन्यथा `~/.cache/svod/objects` | cache को स्थानांतरित करता है |
| `SVOD_OBJECT_CACHE_MAX_BYTES` | 1 GiB | cache का budget |
| `CUDA_PATH` | unset | `ptxas` (`$CUDA_PATH/bin`, `PATH` और `/opt/cuda/bin` के बाद) और CUPTI (`$CUDA_PATH/lib64`, `$CUDA_PATH/extras/CUPTI/lib64`) के लिए आख़िरी में खोजी जाने वाली जगह |
| `SVOD_PROFILE_ITERS`, `SVOD_ORIGIN`, `SVOD_ORIGIN_DEPTH` | | Profiler knobs, देखें [Profiling](./profiling.md) |
| `RUST_LOG` | unset | `svod_device=info` device open line देता है; `svod_device=debug` उसमें JIT info log, graph capture और replay fallbacks जोड़ देता है; `svod_runtime=debug` हर NVPTX compile (arch और IR size) और एक अनुपयोगी `ptxas` log करता है |

कोई CUDA-specific dispatch dump नहीं है; driver JIT log और `tracing` वही कवर करते हैं जो
AMD पर `SVOD_DEBUG_DISPATCH` करता है।

---

## ऐसे host पर "No CUDA device" जिसमें एक है

`has_devices()` तब `false` होता है जब library load न हो, कोई bound symbol ग़ायब हो, `cuInit`
fail हो, या count शून्य हो। `CudaDevice::open` बताता है कि कौन-सा:

```text
device unavailable: cannot load libcuda.so.1: ...        # no driver on the loader path
device unavailable: libcuda.so.1 has no symbol cu...     # driver too old for a bound entry point
no CUDA GPU available: CUDA cuInit failed: ...           # driver loaded, no usable device
```

`ldconfig -p | grep libcuda`, `nvidia-smi` जाँचें, और यह भी कि process `/dev/nvidia*` खोल
सकता है।

---

## एक JIT failure पढ़ना

जिस PTX को driver reject करता है वह `Error::CudaJit` के रूप में सामने आती है, जिसका display
cause है और उसके बाद driver का error log:

```text
CUDA JIT of kernel "r_64_32" failed: CUDA_ERROR_INVALID_PTX: a PTX JIT compilation failed
ptxas application ptx input, line 27; error   : ...
```

`CUDA_ERROR_UNSUPPORTED_PTX_VERSION` का अर्थ है कि driver module की PTX ISA से पुराना है
(pin compute capability का अनुसरण करता है: sm_88 तक 7.8, sm_89 और sm_90 पर 8.4, Blackwell भर में
8.6 से 8.8), देखें [आवश्यकताएँ](./overview.md)। Info log
(warnings, register spills) `svod_device` के अंतर्गत `debug` level पर log होता है। जब
`ptxas` ने kernel को पहले ही assemble कर दिया हो तो कोई driver JIT होता ही नहीं और वही
diagnostic ख़ुद `ptxas` से, एक असफल compile के message के रूप में, आती है।

दो errors Svod के अपने validator से आती हैं, driver के कुछ भी देखने से पहले:

```text
PTX references an unresolved function: .extern .func ...   # an LLVM intrinsic name the NVPTX
                                                            # backend did not recognize
cached PTX targets sm_80, not sm_86                          # a corrupt or foreign cache entry
```

पहला वाला ही असली जाल है: एक ग़लत लिखा गया `llvm.nvvm.*` intrinsic clang के लिए error नहीं
है, वह एक external call बन जाता है। इसका fix `codegen/src/llvm/nvptx/` में है या
`codegen/src/llvm/text/mod.rs` की intrinsic declaration table में।

एक cubin entry की जाँच इसके बजाय `validate_cubin` करता है (एक little-endian ELF64,
`EM_CUDA` के लिए, जो entry को code के रूप में define करता है), और उसकी `.param` list की
जाँच assembly से पहले PTX text पर ABI के विरुद्ध होती है, क्योंकि एक cubin वह साथ नहीं
रखता। PTX वाले रास्ते पर loader वह `.param` जाँच ख़ुद चलाता है; mismatch होने पर संदेश
`PTX entry declares N parameters, the ABI describes M` होता है। एक `llvm.nvvm.*` builder जो
किसी non-NVPTX target तक पहुँच जाए, `Error::ForeignIntrinsic` के रूप में fail होता है।

---

## Toolkit के साथ offline checks

Run time पर किसी चीज़ को CUDA toolkit की *ज़रूरत* नहीं है — `ptxas` का उपयोग kernels को
पहले से assemble करने के लिए तब होता है जब वह संयोग से installed हो — और यदि वह वहाँ है तो
उसके tools dumped IR पर भी काम करते हैं:

```bash
SVOD_DUMP_NVPTX_IR=/tmp/nvptx SVOD_DEVICE=CUDA:0 cargo test -p svod-tensor -- some_test

# Reproduce the exact compile, then assemble with ptxas to see the real diagnostics
clang -x ir -S -O3 --target=nvptx64-nvidia-cuda -march=sm_86 --cuda-feature=+ptx78 \
      -Wno-override-module /tmp/nvptx/sm_86_r_64_32.ll -o r_64_32.ptx
ptxas -arch=sm_86 -v r_64_32.ptx -o r_64_32.cubin   # -v prints registers, shared, spills
nvdisasm r_64_32.cubin | less                         # the SASS
```

यह देखने का सबसे तेज़ तरीक़ा `ptxas -v` है कि `maxThreadsPerBlock` उससे छोटा क्यों निकला
जितना launch चाहता है: `.maxntid` launch bound के विरुद्ध register pressure। उससे मेल खाती
run-time error आँकड़ों का नाम लेती है:

```text
CUDA kernel 'r_64_32' block [512, 1, 1] (512 threads) exceeds its maxThreadsPerBlock 256 (numRegs 96, sharedSizeBytes 4096, localSizeBytes 0)
```

---

## Run time पर driver errors

हर driver call जाँची जाती है और driver के अपने नाम और text के साथ report होती है:

```text
CUDA cuStreamSynchronize failed: CUDA_ERROR_ILLEGAL_ADDRESS (700): an illegal memory access was encountered
```

एक kernel fault asynchronous होता है: `cuLaunchKernel` सफल होती है और error अगली
synchronizing call (`cuStreamSynchronize`, `cuEventSynchronize`, `cuCtxSynchronize`, एक host
copy) पर उतरती है। जिन codes को driver **sticky** बताता है (`ILLEGAL_ADDRESS`,
`LAUNCH_FAILED`, `ILLEGAL_INSTRUCTION`, `MISALIGNED_ADDRESS`, `ECC_UNCORRECTABLE`, ...) वे
device को poison कर देते हैं: हर बाद की call record किए गए message के साथ तुरंत fail होती
है, और frees अपनी allocations को quarantine कर देती हैं, बजाय इसके कि वह memory release करें
जिसे एक अटका हुआ kernel अब भी छू सकता है।

AMD बैकएंड के विपरीत faulting address को classify करने के लिए कोई VA registry नहीं है;
driver उसे expose नहीं करता। एक fault को localize करने के लिए, वही binary toolkit के
sanitizer के नीचे चलाएँ, जो driver-API programs और JIT-loaded PTX पर काम करता है:

```bash
SVOD_DEVICE=CUDA:0 compute-sanitizer --tool memcheck \
  target/release/examples/gigaam_infer ./audio.wav
```

Fault के बजाय एक ग़लत result के लिए, graph replay fallback देखने लायक़ है:
`RUST_LOG=svod_device=debug` तब `CUDA graph replay with re-aliased buffers; using the
capture-order chain` print करता है जब एक replay की buffer aliasing capture से अलग हो, और
`SVOD_DEVICE=CUDA:0 cargo test -p svod-tensor` `codegen_tests!` के `cuda` variants चलाता है,
वही tensor-level assertions जो CPU बैकएंड pass करते हैं, एक बार में एक kernel।

:::tip[Pipeline debugger]
Compiler-side issues (कौन-से UOps ने कौन-सी IR बनाई) के लिए `/svod-debug` skill
frontend → codegen tracing targets का दस्तावेज़ीकरण करती है; `SVOD_DUMP_NVPTX_IR` उसी परिवार
का CUDA सदस्य है, `SVOD_DUMP_AMD_IR` और `SVOD_DUMP_LLVM_IR` के बगल में।
:::
