# svod-codegen

Backend code generation for the Svod ML compiler. It takes an optimized kernel
AST through the staged `PROGRAM` pipeline (`program_pipeline::program_from_sink`,
then `get_program` up to the LINEAR, SOURCE or BINARY stage) and renders the
linearized instruction list into source text for each backend. The compilers and
loaders the BINARY stage uses live in `svod-runtime`.

| Renderer | Target | Downstream compile (in `svod-runtime`) |
|----------|--------|----------------------------------------|
| `llvm::LlvmTextRenderer::new()` | CPU, LLVM IR (default) | in-process libLLVM, else `clang -x ir` |
| `c::CRenderer::new()` | CPU, C (`SVOD_CPU_BACKEND=clang`) | `clang -c` |
| `llvm::LlvmTextRenderer::amd(arch)` | AMD, amdgcn LLVM IR | `clang --target=amdgcn-amd-amdhsa` → code object |
| `llvm::LlvmTextRenderer::nvptx(arch)` | NVIDIA, NVPTX LLVM IR | `clang --target=nvptx64-nvidia-cuda` → PTX, `ptxas` when installed, else driver JIT |
| `c::CRenderer::metal()` | Apple GPUs, Metal Shading Language | `MTLCodeGenService` → metallib |

Every renderer implements the `Renderer` trait
(`render(&uop, name) -> Result<RenderedKernel>`). It expects a LINEAR-stage
UOp: the staged entry points in `program_pipeline` run the required cleanup
pass themselves, direct callers must run
`svod_schedule::linearize::line_rewrite_cleanups` first.

Documentation:

- Codegen pipeline: <https://svod.vpermilp.online/docs/architecture/codegen/overview>
- Backends: <https://svod.vpermilp.online/docs/backends/overview>
