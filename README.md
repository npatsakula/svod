<div align="center">

# Svod

**A deep learning compiler and inference engine written in Rust.**

[![CI](https://github.com/npatsakula/svod/actions/workflows/ci.yml/badge.svg)](https://github.com/npatsakula/svod/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-svod.vpermilp.online-blue)](https://svod.vpermilp.online/docs/introduction)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

[Documentation](https://svod.vpermilp.online/docs/introduction) ·
[Models](#models-and-pipelines) ·
[Architecture](https://svod.vpermilp.online/docs/architecture/pipeline) ·
[Talks](#talks-and-writing) ·
[Roadmap](#roadmap)

</div>

Svod compiles lazy tensor graphs into fused kernels for CPUs, AMD and NVIDIA GPUs
with no vendor runtime in the loop: no PyTorch, no ROCm/HIP, no CUDA toolkit.
It follows [Tinygrad](https://github.com/tinygrad/tinygrad)'s design: a small,
verifiable IR (UOps), pattern-driven rewrites, and a straight pipeline from
tensors to machine code.

Svod ships with speech, text and vision models that are checked against their
PyTorch references, plus an ONNX importer for everything else.

## Why Svod

- **One representation, end to end.** PyTorch moves a model through seven or
  more IRs, and every boundary is a bridge a vendor has to build and a break in
  the debugging context. In Svod a single UOp graph is built, optimized,
  scheduled and rendered. A new accelerator needs only three pieces: a code
  generator, a buffer allocator and a kernel launcher.
- **Production Rust.** Static types and native concurrency instead of Python
  glue around the GIL. Deployment is one binary, with no LibTorch or ONNX
  Runtime bindings and no vendor SDKs.
- **A familiar API.** The tensor API mirrors PyTorch, down to named arguments,
  so a model port reads like the reference, and porting it is mostly mechanical.
- **Speed where it counts.** When the compiler is not enough, the `tk` tile DSL
  gives you hand-written kernels in the same IR. They stay visible to the
  profiler and the origin tracker instead of living off to the side as opaque
  binaries.

The motivation and design are covered in
[Developing Svod](https://blog.vpermilp.online/en/blog/svod-intro/).

## Models and pipelines

| Domain | Models |
|---|---|
| Speech recognition | Whisper, GigaAM v3 (CTC, RN-T) |
| Voice activity | FireRedVAD, Silero VAD |
| Speech enhancement | GTCRN |
| Speaker analysis | DiariZen, WeSpeaker |
| Text embeddings and reranking | BGE-M3, Qwen3-Embedding, ModernBERT |
| Vision | YOLO26, ResNet |
| Anything else | [ONNX importer](onnx/) ([op coverage](onnx/PARITY.md)) |

Weights come straight from the Hugging Face Hub, and outputs are checked
against the reference implementations. [`model/`](model/) lists variants,
upstream links and runnable examples; [`arch`](arch/) holds the decoders and the long-form audio pipeline.

## Engine

### Graph capture: compile once, replay many times

A model is traced once into an execution plan, and each call after that only
replays it. Symbolic dimensions (batch, sequence length) are bound per call
without recompiling. Recurrent state stays on the device between calls, and a
memory planner reuses intermediate buffers through a TLSF arena. Static chains
are replayed as one hardware graph: **CUDA Graphs** on NVIDIA, a doorbell-per-replay
AQL/PM4 graph on AMD (as in HCQGraph), and indirect command buffers on Metal.
See [JIT graphs](https://svod.vpermilp.online/docs/architecture/jit-graphs).

```rust
jit_wrapper! {
    GigaAmEncoderJit(GigaAm) {
        mel: Tensor,
        lengths: Tensor,

        outputs { frames },

        build(mel, lengths) {
            model.encoder.forward_batch(mel, lengths)
        }
    }
}
// let mut jit = GigaAmEncoderJit::new(model);
// jit.prepare(..)?;   // trace, schedule and compile once
// jit.execute()?;     // replay on every chunk
```

### Rewrites proven with Z3

Every optimization is a declarative rewrite in the `patterns!` DSL. The
algebraic and index simplifications are checked with the **Z3 SMT solver**: it
proves that the rewritten expression equals the original for every input, or
returns a counterexample. Property-based tests cover the rest of the pipeline.
See [Pattern system](https://svod.vpermilp.online/docs/architecture/optimizations/pattern-system).

### Platform-specific code generation

- **Tensor cores** are chosen per architecture: NVIDIA sm_75/80/89, AMD RDNA3,
  RDNA4 and CDNA3/4, plus Apple Metal. fp8 is available on sm_89 and CDNA3.
- **Tile kernels (`tk`)**: a ThunderKittens-style tile DSL in Rust for GEMM,
  flash attention, RMSNorm and k-means. A single kernel source lowers to AMD
  MFMA/WMMA (gfx942, gfx11, gfx12), CUDA `mma.sync` (sm_80+) and Apple
  `simdgroup_matrix` (Apple7+). Tile shapes are
  autotuned on first use and cached. See
  [Tile kernels](https://svod.vpermilp.online/docs/tile-kernels/overview).

  ```rust
  fn micro_matmul(ker: &Kernel) -> Arc<UOp> {
      let w = ker.warp();
      let a = ker.rt((64, 64), DType::BFloat16, Row, RT_16X16);
      let b = ker.rt((64, 64), DType::BFloat16, Col, RT_16X16);
      let c = ker.rt((64, 64), DType::Float32, Col, RT_16X16);
      let out = w.mma_ab(w.zero(c), &a, &b); // one matrix-core instruction per fragment
      ker.finish(1)
  }
  ```
- **Kernel search**: hand-written heuristics, or BEAM search over the
  optimization space with a persistent on-disk cache. See
  [Kernel search](https://svod.vpermilp.online/docs/architecture/optimizations/kernel-search).
- **CPU**: vectorized LLVM IR compiled in-process, a custom ELF loader for
  x86_64, aarch64, riscv64, loongarch64 and ppc64le, and multi-threaded
  kernels.

### Zero-copy data paths

ONNX initializers and `Tensor::from_path` tensors are memory-mapped lazily from disk. Device buffers
support sub-views. Host code reads and writes realized tensors through borrowed
`ndarray` views (`array_view`, `array_view_mut`), so feeding a captured plan
copies nothing.

### Kernel fusion and attribution

The RANGEIFY scheduler fuses elementwise, reduction and movement ops into as
few kernels as possible. Each kernel records where it came from (module path,
ONNX node or source line), so the profiler can attribute device time, roofline
GFLOP/s and GB/s, occupancy and hardware counters (AMD SQ, NVIDIA CUPTI) back to
the model code. See [Kernel origins](https://svod.vpermilp.online/docs/architecture/kernel-origins).

## Backends

| Device | Selector | Compilation | Runtime |
|---|---|---|---|
| CPU | `CPU` (default off macOS) | LLVM IR via `libLLVM` loaded at runtime, falling back to `clang`; Clang C backend | Own ELF JIT loader, multi-threaded |
| AMD GPU | `AMD:N` | `clang --target=amdgcn-amd-amdhsa` | Direct KFD queues (AQL/PM4), no HIP or ROCm runtime |
| NVIDIA GPU | `CUDA:N` | `clang` NVPTX → PTX → `ptxas` or the driver JIT | `libcuda.so.1` loaded at runtime, no CUDA toolkit |
| Apple GPU | `METAL:N` (default on macOS) | MSL → metallib | Metal frameworks loaded at runtime |

Every GPU backend is compiled in and registers only when the hardware is
present. To pick one, set `SVOD_DEVICE` or call `Tensor::to(device)`.

CPU code is tested on x86_64, aarch64, riscv64 and ppc64le, on Linux and
macOS. GPUs: AMD RDNA 3.5, RDNA 4 and CDNA 3, NVIDIA sm_80 and newer, and Apple M3 and newer.

## Talks and writing

[Developing Svod](https://blog.vpermilp.online/en/blog/svod-intro/), a blog
post (August 2026) on why Svod exists, its architecture and the roadmap.

| Event | Talk | Language |
|---|---|---|
| [Data Fest 2026](https://ods.ai/events/df2026-31-may-online) (online, 31 May 2026) | Writing the fastest Sber GigaAM inference on Svod | Russian |
| [RustCon 2025](https://rustcon.ru/morok-minimalistichnyy-deep-learning-freymvork-na-rust) (Moscow, November 2025) | Morok: a minimalist deep learning framework in Rust | Russian |
| [Stereo Data Ёлка 2025](https://ods.ai/events/data-elka-2025-vk-offline-spb) (St. Petersburg, January 2026) | ML in Rust | Russian |

Morok was Svod's earlier name.

## Workspace

| Crate | Role |
|---|---|
| [`dtype`](dtype/) | Scalar, vector, pointer and image types, including bf16 and fp8 |
| [`ir`](ir/) | UOp graph IR with hash-consing, symbolic integers and origins |
| [`macros`](macros/) | `patterns!` rewrite DSL and `jit_wrapper!` |
| [`schedule`](schedule/) | RANGEIFY, rewrite passes, heuristics and BEAM, Z3 verification |
| [`codegen`](codegen/) | LLVM IR (CPU, AMDGPU, NVPTX), C and MSL renderers |
| [`device`](device/) | Buffers, allocators, mmap, KFD, CUDA and Metal drivers, hardware graphs |
| [`runtime`](runtime/) | Kernel compilation, caching, execution plans and the profiler |
| [`tensor`](tensor/) | Lazy tensor API, `nn` modules and the memory planner |
| [`tk`](tk/) | Tile kernel DSL and kernel library |
| [`onnx`](onnx/) | ONNX importer |
| [`arch`](arch/) | Host-side decoders, VAD segmentation and audio pipelines |
| [`model`](model/) | Pretrained models and examples |

## Using the library

Models chain into pipelines. Here is long-form Russian speech recognition with
GigaAM, segmented by FireRedVAD:

```rust
let model = GigaAm::from_hub_with_revision("vpermilp/GigaAM-v3", "ctc")?;
let bounds = EncoderBounds {
    sample_rate: model.config.sample_rate as u32,
    hop_length: model.config.hop_length,
    subsampling_factor: model.config.subsampling_factor,
    max_mel_frames: model.config.max_mel_frames,
    recommended_target_secs: model.recommended_chunk_secs(),
};
let splitter = FireRedVadSplitter::from_hub(&bounds)?;
let mut asr = Asr::assemble(splitter, |max_chunk| GigaAmTranscriber::new(model, opts, max_chunk))?;
let result = asr.transcribe_default(&waveform)?;
```

Any ONNX model can be compiled once and replayed. The graph is traced on your
input tensor, so new data written into it is seen by every replay:

```rust
let proto = ModelProto::decode(std::fs::read("model.onnx")?.as_slice())?;
let input = Tensor::from_ndarray(&first_batch); // [1, 3, 224, 224] f32

let OnnxModel { outputs, .. } = OnnxImporter::new().import_model_with_inputs(
    proto,
    HashMap::from([("input".to_string(), input.clone())]),
    &[("batch", 1)],
)?;

let plan = Tensor::prepare_batch(outputs.values())?; // compile once
plan.execute()?;

for batch in batches {
    input.array_view_mut::<f32>()?.as_slice_mut().unwrap().copy_from_slice(&batch);
    plan.execute()?; // replay: no tracing, no compilation, no allocation
}
```

## Building

The Nix flake pins every compiler and library, and CI uses the same flake:

```bash
nix develop      # development shell
nix flake check  # the CI suite: clippy, nextest (with Z3 and proptest), fmt
```

Without Nix you need the following:

| Dependency | Version | Required | Purpose |
|---|---|---|---|
| Rust | 1.88+ | yes | Edition 2024 |
| LLVM | ≥ 16 | yes | CPU code generation; `libLLVM` is loaded at runtime |
| Clang | — | yes | GPU kernel compilation, C backend, fallback when `libLLVM` is missing |
| protobuf, pkgconf, zlib, libffi, libxml2 | — | yes | ONNX protos and the LLVM toolchain |
| Z3 | ≥ 4.15 | no | Rewrite verification (`--features z3`) |
| NVIDIA driver | CUDA ≥ 12.0 (R525) | no | CUDA backend |
| amdgpu kernel driver (KFD) | — | no | AMD backend |

```bash
cargo test --workspace
cargo test --workspace --features z3,proptest
```

`SVOD_THREADS` sets the single thread budget used to compile kernels and run
CPU kernels.

## Roadmap

- **AOT compilation:** serialize optimized graphs and compiled kernels, so a
  model starts instantly and runs where no compiler is available (e.g. WASM).
- **Data analysis primitives:** FlashAttention-style GPU kernels for k-means,
  kNN, PCA, SVD, (H)DBSCAN, UMAP and t-SNE on every backend. k-means and kNN
  already ship in `tk`.
- **Formal verification of generated code:** annotated C output that proves
  the absence of out-of-bounds accesses and lossy casts.
- **More hardware:** server (MI300–MI450, H100–B200), consumer (Ryzen AI,
  Apple M3–M5, RTX 30–50) and embedded (Snapdragon X, RK3588) targets behind
  one tensor API, plus a userspace AMD driver with no dependency on the AMD software stack.

## License

[MIT](LICENSE)
