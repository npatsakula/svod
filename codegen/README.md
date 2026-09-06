# svod-codegen

Backend code generation from optimized UOp graphs.

## Example

```rust
use svod_codegen::{Renderer, render};

let code = render(&kernel_graph, backend)?;
```

## Backends

| Backend | Output | Feature | Default |
|---------|--------|---------|---------|
| **LLVM** (default) | LLVM IR text → in-process `libLLVM` (or `clang -x ir`) → JIT ELF loader | always | no |
| **Clang** | C source → `clang -c` → JIT ELF loader | always | yes |

Select at runtime via `SVOD_CPU_BACKEND` env var (`clang` or `llvm`).

GPU targets share the LLVM text renderer (`llvm::LlvmTextRenderer`) or the C renderer:

| Target | Output |
|--------|--------|
| **AMD** (`LlvmTextRenderer::amd(arch)`) | amdgcn LLVM IR → `clang --target=amdgcn-amd-amdhsa` → code object |
| **NVIDIA** (`LlvmTextRenderer::nvptx(arch)`) | nvptx64 LLVM IR → `clang --target=nvptx64-nvidia-cuda` → PTX, JIT'd by the CUDA driver |
| **Metal** (`c::CRenderer::metal()`) | MSL source → `metallib` via `MTLCodeGenService` |

**Planned:**

- WebGPU (WGSL) renderer

## Testing

```bash
cargo test -p svod-codegen
```
