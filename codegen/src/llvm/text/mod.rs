//! Text-based LLVM IR code generation (main entry point).
//!
//! This module generates LLVM IR as plain strings using `format!` macros,
//! following Tinygrad's approach in `renderer/llvmir.py`.
//!
//! # Kernel Signature
//!
//! Generates a single function with direct typed parameters and `noalias align 32`
//! buffer annotations:
//! ```llvm
//! define void @kernel(ptr noalias align 32 %buf0, ..., i32 %N) #0 { ... }
//! ```

use std::sync::Arc;

use svod_dtype::{AmdArch, CudaArch};
use svod_ir::pattern::TypedPatternMatcher;
use svod_ir::{Op, prelude::*};

use crate::common::{collect_abi_params, is_output_buffer};
use crate::llvm::amd;
use crate::llvm::common::gpu::max_local_threads;
use crate::llvm::common::{LlvmTarget, RenderContext, ldt};
use crate::llvm::cpu;
use crate::llvm::nvptx;
use crate::{BufferArg, Error, RenderedKernel, RenderedOperation, Renderer, Result};
use svod_ir::ops;

/// Text-based LLVM IR renderer.
///
/// Generates LLVM IR as strings, suitable for compilation via external clang.
/// Produces a single function with direct typed parameters. The active
/// [`LlvmTarget`] selects between the CPU emitter, the AMDGPU emitter
/// (`amdgpu_kernel` ABI, addrspace(3) LDS, amdgcn intrinsics) and the NVPTX
/// emitter (`ptx_kernel` ABI, addrspace(3) shared memory, nvvm intrinsics).
pub struct LlvmTextRenderer {
    target: LlvmTarget,
}

impl LlvmTextRenderer {
    /// Renderer for the host CPU target (default for backwards compatibility).
    pub fn new() -> Self {
        Self { target: LlvmTarget::Cpu }
    }

    /// Renderer for an AMD GPU at the named `gfx{family}` target.
    pub fn amd(arch: AmdArch) -> Self {
        Self { target: LlvmTarget::Amd(arch) }
    }

    /// Renderer for an NVIDIA GPU at the named `sm_XY` compute capability.
    pub fn nvptx(arch: CudaArch) -> Self {
        Self { target: LlvmTarget::Nvptx(arch) }
    }

    /// Construct with an explicit target.
    pub fn with_target(target: LlvmTarget) -> Self {
        Self { target }
    }

    pub fn target(&self) -> LlvmTarget {
        self.target
    }
}

impl Default for LlvmTextRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl Renderer for LlvmTextRenderer {
    fn render(&self, uop: &Arc<UOp>, name: Option<&str>) -> Result<RenderedKernel> {
        let kernel_name = name.unwrap_or("kernel");

        let nodes: Vec<Arc<UOp>> = match uop.op() {
            Op::Linear(ops::Linear { ops }) => ops.iter().cloned().collect(),
            other => {
                return Err(Error::InvalidGraph {
                    reason: format!("LLVM text renderer expects LINEAR input, got {other:?}"),
                });
            }
        };
        crate::common::reject_unsupported_fnuz(&nodes, "LLVM")?;

        // Instruction-scheduling pass: lower any `sched::pipeline` markers into the
        // gfx9 machine scheduling controls (s_setprio brackets, sched.barrier fences,
        // the attention interleave comb). No-op on non-CDNA targets / unmarked kernels.
        let nodes = crate::llvm::sched::apply_pipeline_scheduling(nodes, self.target);

        for (i, node) in nodes.iter().enumerate() {
            tracing::trace!(position = i, op = node.op().as_ref(), id = node.id, "linearized node");
        }

        let mut ctx = RenderContext::new();
        let mut kernel: Vec<String> = Vec::new();
        let mut operations = Vec::new();
        let mut buffer_args: Vec<BufferArg> = Vec::new();
        let mut var_names: Vec<String> = Vec::new();

        let abi_params = collect_abi_params(&nodes)?;

        for buf in abi_params
            .iter()
            .filter(|param| matches!(param.op(), Op::Param(ops::Param { arg, .. }) if arg.addrspace.is_some()))
        {
            if let Op::Param(ops::Param { arg, .. }) = buf.op() {
                let is_output = is_output_buffer(buf, &nodes);
                buffer_args.push(BufferArg {
                    index: arg.slot,
                    name: format!("data{}", arg.slot),
                    dtype: buf.dtype(),
                    is_output,
                });
            }
        }

        for var in abi_params
            .iter()
            .filter(|param| matches!(param.op(), Op::Param(ops::Param { arg, .. }) if arg.addrspace.is_none()))
        {
            let name = match var.op() {
                Op::Param(ops::Param { arg, .. }) => arg.name.as_ref().ok_or_else(|| Error::InvalidGraph {
                    reason: format!("scalar PARAM in slot {} has no name", arg.slot),
                })?,
                other => return Err(Error::InvalidGraph { reason: format!("non-PARAM in ABI list: {other:?}") }),
            };
            var_names.push(name.clone());
        }
        // -- Build function parameters --
        let mut inner_params: Vec<String> = Vec::new();

        for param in &abi_params {
            let Op::Param(ops::Param { arg, .. }) = param.op() else {
                return Err(Error::InvalidGraph { reason: "non-PARAM in ABI list".into() });
            };
            let source_name = format!("%data{}", arg.slot);
            if arg.addrspace.is_some() {
                inner_params.push(format!("ptr noalias align 32 {source_name}"));
            } else {
                inner_params.push(format!("{} {source_name}", ldt(&param.dtype())));
            }
            ctx.register(param.id, source_name);
        }

        // -- Build function body --
        kernel.push("".to_string());

        for node in &nodes {
            if matches!(node.op(), Op::Noop | Op::Group(..)) {
                ctx.register(node.id, String::new());
                continue;
            }
            if let Some(intrinsic) = foreign_intrinsic(node, self.target) {
                return Err(Error::ForeignIntrinsic { intrinsic, target: self.target.to_string() });
            }
            let first_line = kernel.len();
            match self.target {
                LlvmTarget::Cpu => {
                    cpu::render_uop(node, &mut ctx, &mut kernel);
                }
                LlvmTarget::Amd(_) => {
                    amd::render_uop(node, &mut ctx, &mut kernel, self.target);
                }
                LlvmTarget::Nvptx(_) => {
                    nvptx::render_uop(node, &mut ctx, &mut kernel, self.target);
                }
            }
            if let Some(err) = ctx.take_error() {
                return Err(err);
            }
            operations.push(RenderedOperation {
                uop_id: node.id,
                op: node.op().as_ref().to_string(),
                source_ids: node.op().sources().iter().map(|source| source.id).collect(),
                result: ctx.try_get(node).map(str::to_string),
                lines: kernel[first_line..].to_vec(),
            });
        }

        match self.target {
            LlvmTarget::Cpu => {}
            // Both GPU backends turn `arcp`/`afn` division into an unrefined
            // reciprocal: AMDGPU selects `v_rcp_f32`, and NVPTX lowers
            // `fdiv nsz arcp contract afn` to `rcp.approx.f32` where plain
            // `contract` keeps the exact `div.rn.f32` (probed on clang 22).
            // Tinygrad's CUDA and AMD C frontends both compile with exact
            // division, so keep `contract` only.
            LlvmTarget::Amd(_) | LlvmTarget::Nvptx(_) => {
                for line in &mut kernel {
                    *line = line.replace(" nsz arcp contract afn ", " contract ");
                }
            }
        }

        if !ctx.open_ranges().is_empty() {
            return Err(Error::InvalidGraph { reason: format!("unclosed LLVM ranges: {:?}", ctx.open_ranges()) });
        }

        kernel.push("  ret void".to_string());

        let abi = match self.target {
            LlvmTarget::Cpu => "void",
            LlvmTarget::Amd(_) => "amdgpu_kernel void",
            // `ptx_kernel` alone yields `.visible .entry`; no `!nvvm.annotations`.
            LlvmTarget::Nvptx(_) => "ptx_kernel void",
        };

        let attrs = build_function_attributes(&self.target, &nodes);

        // Module-level prefix:
        //   1. amdgcn intrinsic declarations + CPU intrinsic declarations
        //   2. fp8 helper (AMD-only, only when the kernel uses fp8)
        //   3. addrspace(3) LDS globals from LOCAL BUFFERs (AMD-only)
        let mut module_blocks: Vec<String> = Vec::new();
        module_blocks.push(generate_intrinsic_declarations(&kernel, &self.target));
        if self.target.is_amd()
            && let Some(helper) = amd::ops::fp8_helper_prefix(&nodes)
        {
            module_blocks.push(helper.to_string());
        }
        if !ctx.module_prefix().is_empty() {
            module_blocks.push(ctx.module_prefix().join("\n"));
        }

        // A `declare` can originate from both the auto-scan and a hoisted
        // CUSTOM body line; LLVM forbids redefining a function, so keep only
        // the first occurrence of each identical declaration.
        let module_prefix = dedup_declares(module_blocks.join("\n\n"));

        let target_triple_line = match self.target {
            LlvmTarget::Cpu => String::new(),
            LlvmTarget::Amd(_) => "target triple = \"amdgcn-amd-amdhsa\"\n".to_string(),
            // clang overrides a mismatching module datalayout silently, so the
            // line exists for tools that parse the module standalone
            // (`llvm-as`, `opt`, IR dumps) and would otherwise assume the
            // host layout. This is clang 22's default for nvptx64, and every
            // spec in it (`p6` = the 32-bit `.param` space, `i256`) parses on
            // older LLVMs as well.
            LlvmTarget::Nvptx(_) => {
                "target datalayout = \"e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64\"\n\
                                    target triple = \"nvptx64-nvidia-cuda\"\n"
                    .to_string()
            }
        };

        let ir = format!(
            r#"; ModuleID = '{kernel_name}'
source_filename = "{kernel_name}"
{target_triple_line}
{module_prefix}

define {abi} @{kernel_name}({inner_params}) #0 {{
entry:
{inner_body}
}}

attributes #0 = {{ {attrs} }}
"#,
            module_prefix = module_prefix,
            inner_params = inner_params.join(", "),
            inner_body = kernel.join("\n"),
        );

        tracing::trace!(generated_code = ir, "llvm codegen: final generated code");

        let mut result = RenderedKernel::new(ir, kernel_name.to_string());
        result.buffer_args = buffer_args;
        result.var_names = var_names;
        result.abi = abi_params
            .iter()
            .map(|param| {
                svod_device::device::AbiParamDescriptor::from_param(param)
                    .map_err(|error| Error::InvalidGraph { reason: error.to_string() })
            })
            .collect::<Result<Vec<_>>>()?;
        result.operations = operations;

        Ok(result)
    }

    fn backend_name(&self) -> &str {
        match self.target {
            LlvmTarget::Cpu | LlvmTarget::Amd(_) => "llvm-text",
            LlvmTarget::Nvptx(_) => "nvptx",
        }
    }

    fn decompositor(&self) -> Option<TypedPatternMatcher<()>> {
        None
    }
}

/// The first `@llvm.nvvm.*` / `@llvm.amdgcn.*` reference in a CUSTOM body
/// that `target` cannot lower. The typed builders (`nvptx::{ops,smem}`, tk's
/// gfx9 `asm`) are target-specific; on the wrong target clang emits the name
/// as an extern call that only the device assembler rejects.
fn foreign_intrinsic(node: &Arc<UOp>, target: LlvmTarget) -> Option<String> {
    let code = match node.op() {
        Op::Custom(ops::Custom { code, .. }) | Op::CustomI(ops::CustomI { code, .. }) => code,
        _ => return None,
    };
    code.match_indices("@llvm.")
        .map(|(at, _)| code[at + 1..].split(|c: char| c.is_whitespace() || c == '(').next().unwrap_or_default())
        .find(|name| {
            (name.starts_with("llvm.nvvm.") && !target.is_nvptx())
                || (name.starts_with("llvm.amdgcn.") && !target.is_amd())
        })
        .map(str::to_string)
}

fn mangle_type(llvm_type: &str) -> String {
    match llvm_type {
        "float" => "f32".to_string(),
        "double" => "f64".to_string(),
        "half" => "f16".to_string(),
        "i8" => "i8".to_string(),
        "i16" => "i16".to_string(),
        "i32" => "i32".to_string(),
        "i64" => "i64".to_string(),
        _ if llvm_type.starts_with('<') && llvm_type.ends_with('>') => {
            let inner = &llvm_type[1..llvm_type.len() - 1];
            let parts: Vec<&str> = inner.split(" x ").collect();
            if parts.len() == 2 {
                let count = parts[0].trim();
                let base = mangle_type(parts[1].trim());
                format!("v{count}{base}")
            } else {
                llvm_type.to_string()
            }
        }
        _ => llvm_type.to_string(),
    }
}

/// Remove duplicate `declare ...` lines from an assembled module prefix,
/// keeping the first occurrence per **function name** (the `@name` token). Two
/// declares for the same intrinsic with different signatures — e.g. a wave64
/// and a wave32 `@llvm.amdgcn.wmma.*` call in the same kernel — are treated as
/// duplicates so the second is dropped, avoiding clang's "invalid redefinition"
/// error. Non-`declare` lines pass through unchanged.
fn dedup_declares(prefix: String) -> String {
    let mut seen = std::collections::HashSet::new();
    let mut out: Vec<&str> = Vec::new();
    for line in prefix.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("declare ") {
            // Extract the function name: the token between `@` and `(`.
            let key = trimmed
                .find('@')
                .and_then(|at| trimmed[at + 1..].find('(').map(|p| &trimmed[at + 1..at + 1 + p]))
                .unwrap_or(trimmed);
            if !seen.insert(key.to_string()) {
                continue;
            }
        }
        out.push(line);
    }
    out.join("\n")
}

fn generate_intrinsic_declarations(kernel: &[String], target: &LlvmTarget) -> String {
    let mut decls = Vec::new();
    let kernel_str = kernel.join("\n");

    for intrinsic in &[
        "sqrt", "exp", "exp2", "log", "log2", "sin", "cos", "pow", "fabs", "floor", "ceil", "trunc", "round", "maxnum",
        "minnum", "fmuladd", "erf",
    ] {
        for llvm_type in
            &["float", "double", "half", "<2 x float>", "<4 x float>", "<8 x float>", "<2 x double>", "<4 x double>"]
        {
            let mangled = mangle_type(llvm_type);
            let pattern = format!("@llvm.{intrinsic}.{mangled}");
            if kernel_str.contains(&pattern) {
                let decl = match *intrinsic {
                    "fmuladd" => format!(
                        "declare {llvm_type} @llvm.{intrinsic}.{mangled}({llvm_type}, {llvm_type}, {llvm_type})"
                    ),
                    "pow" | "maxnum" | "minnum" => {
                        format!("declare {llvm_type} @llvm.{intrinsic}.{mangled}({llvm_type}, {llvm_type})")
                    }
                    _ => format!("declare {llvm_type} @llvm.{intrinsic}.{mangled}({llvm_type})"),
                };
                decls.push(decl);
            }
        }
    }

    for bits in &["i8", "i16", "i32", "i64"] {
        let pattern = format!("@llvm.abs.{bits}");
        if kernel_str.contains(&pattern) {
            decls.push(format!("declare {bits} @llvm.abs.{bits}({bits}, i1)"));
        }
    }

    match target {
        LlvmTarget::Cpu => {}
        LlvmTarget::Amd(_) => {
            // Only the f64 transcendentals the AMDGPU backend cannot select stay on
            // ROCm device libraries; everything else is an `@llvm.*` intrinsic
            // declared by the generic loop above. See `amd::ops::render_float_unary`.
            for op in ["log2", "exp2"] {
                let name = format!("@__ocml_{op}_f64");
                if kernel_str.contains(&name) {
                    decls.push(format!("declare double {name}(double)"));
                }
            }
            // Scalar (non-mangled) amdgcn intrinsics; declared whenever referenced
            // in the kernel body. Source: AMDGPU LLVM intrinsic reference.
            for (pattern, decl) in [
                ("@llvm.amdgcn.s.barrier", "declare void @llvm.amdgcn.s.barrier()"),
                ("@llvm.amdgcn.workgroup.id.x", "declare i32 @llvm.amdgcn.workgroup.id.x()"),
                ("@llvm.amdgcn.workgroup.id.y", "declare i32 @llvm.amdgcn.workgroup.id.y()"),
                ("@llvm.amdgcn.workgroup.id.z", "declare i32 @llvm.amdgcn.workgroup.id.z()"),
                ("@llvm.amdgcn.workitem.id.x", "declare i32 @llvm.amdgcn.workitem.id.x()"),
                ("@llvm.amdgcn.workitem.id.y", "declare i32 @llvm.amdgcn.workitem.id.y()"),
                ("@llvm.amdgcn.workitem.id.z", "declare i32 @llvm.amdgcn.workitem.id.z()"),
                ("@llvm.amdgcn.cvt.f32.fp8", "declare float @llvm.amdgcn.cvt.f32.fp8(i32, i32)"),
                ("@llvm.amdgcn.cvt.f32.bf8", "declare float @llvm.amdgcn.cvt.f32.bf8(i32, i32)"),
                ("@llvm.amdgcn.cvt.pk.fp8.f32", "declare i32 @llvm.amdgcn.cvt.pk.fp8.f32(float, float, i32, i1)"),
                ("@llvm.amdgcn.cvt.pk.bf8.f32", "declare i32 @llvm.amdgcn.cvt.pk.bf8.f32(float, float, i32, i1)"),
                ("@llvm.amdgcn.fmed3.f32", "declare float @llvm.amdgcn.fmed3.f32(float, float, float)"),
            ] {
                if kernel_str.contains(pattern) {
                    decls.push(decl.to_string());
                }
            }
        }
        LlvmTarget::Nvptx(_) => {
            // Scalar nvvm intrinsics referenced by `nvptx::ops`. Names are exact:
            // a misspelt nvvm intrinsic (`lg2.approx.f32`) is emitted as an
            // external call and only fails inside ptxas.
            for (pattern, decl) in [
                ("@llvm.nvvm.barrier0", "declare void @llvm.nvvm.barrier0()"),
                ("@llvm.nvvm.lg2.approx.f", "declare float @llvm.nvvm.lg2.approx.f(float)"),
                ("@llvm.nvvm.read.ptx.sreg.ctaid.x", "declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()"),
                ("@llvm.nvvm.read.ptx.sreg.ctaid.y", "declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()"),
                ("@llvm.nvvm.read.ptx.sreg.ctaid.z", "declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()"),
                ("@llvm.nvvm.read.ptx.sreg.tid.x", "declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()"),
                ("@llvm.nvvm.read.ptx.sreg.tid.y", "declare i32 @llvm.nvvm.read.ptx.sreg.tid.y()"),
                ("@llvm.nvvm.read.ptx.sreg.tid.z", "declare i32 @llvm.nvvm.read.ptx.sreg.tid.z()"),
            ] {
                if kernel_str.contains(pattern) {
                    decls.push(decl.to_string());
                }
            }
        }
    }
    if !matches!(target, LlvmTarget::Cpu) {
        // WMMA / MFMA / mma.sync intrinsics: the signature varies by family
        // and dtype, so we synthesize each `declare` from its call site's
        // operand types. The operands already carry the intrinsic-required
        // wire types (bf16 as i16 or packed i32, fp8 as a packed integer,
        // f16 as `<2 x half>` pairs), so the declaration matches the call by
        // construction. Dedup identical lines (a tiled matmul emits many
        // calls to the same intrinsic).
        for line in kernel.iter() {
            if let Some(decl) = wmma_declaration_from_call(line)
                && !decls.contains(&decl)
            {
                decls.push(decl);
            }
        }
    }

    decls.join("\n")
}

/// Synthesize a `declare` line for a `@llvm.amdgcn.{wmma,mfma}.*` or
/// `@llvm.nvvm.mma.*` call by echoing the call's argument types. Returns
/// `None` if the line isn't a matrix-core call site.
fn wmma_declaration_from_call(line: &str) -> Option<String> {
    const NEEDLES: [&str; 3] = ["@llvm.amdgcn.wmma.", "@llvm.amdgcn.mfma.", "@llvm.nvvm.mma."];
    let needle = NEEDLES.into_iter().find(|needle| line.contains(needle))?;
    // `  %vN = call <ret_ty> @llvm.amdgcn.wmma.<rest>(<args>)`; the NVPTX
    // return type is an aggregate (`{ float, float, float, float }`).
    let call_start = line.find("call ")?;
    let after_call = &line[call_start + "call ".len()..];
    let ret_end = after_call.find(" @")?;
    let ret_ty = &after_call[..ret_end];
    let name_start = call_start + "call ".len() + ret_end + 2; // skip " @"
    let paren = line[name_start..].find('(')?;
    let intrinsic_name = &line[name_start..name_start + paren];
    if !intrinsic_name.starts_with(&needle[1..]) {
        return None;
    }
    // Extract the argument list (between the matching parens).
    let args_start = name_start + paren + 1;
    let args_end = line[args_start..].rfind(')')?;
    let args_chunk = &line[args_start..args_start + args_end];
    // Pull out types — entries are `<ty> %name` or `<ty> <const>`.
    let mut param_types: Vec<String> = Vec::new();
    let mut depth = 0;
    let mut current = String::new();
    let mut parts: Vec<String> = Vec::new();
    for ch in args_chunk.chars() {
        match ch {
            '<' => {
                depth += 1;
                current.push(ch);
            }
            '>' => {
                depth -= 1;
                current.push(ch);
            }
            ',' if depth == 0 => {
                parts.push(current.trim().to_string());
                current.clear();
            }
            _ => current.push(ch),
        }
    }
    if !current.trim().is_empty() {
        parts.push(current.trim().to_string());
    }
    for part in parts {
        let trimmed = part.trim();
        // The leading *type* token. A `<…>` vector/aggregate type runs to its
        // matching `>` (the value or name follows it); a scalar type is the
        // token before the first space (`i32 0`, `i1 false`). Splitting on the
        // first space would truncate `<16 x half>` to `<16` — the bug this
        // replaces — since the type itself contains spaces.
        let ty = if trimmed.starts_with('<') {
            let mut depth = 0usize;
            let mut end = trimmed.len();
            for (i, ch) in trimmed.char_indices() {
                match ch {
                    '<' => depth += 1,
                    '>' => {
                        depth -= 1;
                        if depth == 0 {
                            end = i + 1;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            &trimmed[..end]
        } else {
            trimmed.split_whitespace().next().unwrap_or(trimmed)
        };
        param_types.push(ty.to_string());
    }
    Some(format!("declare {ret_ty} @{intrinsic_name}({})", param_types.join(", ")))
}

/// Build the per-target `attributes #0` body.
fn build_function_attributes(target: &LlvmTarget, nodes: &[Arc<UOp>]) -> String {
    match target {
        LlvmTarget::Cpu => "nounwind \"no-builtins\" \"no-trapping-math\"=\"true\"".to_string(),
        // Tinygrad `llvmir.py:259-263`: include the upper bound on the local
        // workgroup size so the AMDGPU backend can size scratch allocations /
        // waves correctly.
        LlvmTarget::Amd(_) => format!(
            "alwaysinline nounwind \"no-builtins\" \"amdgpu-flat-work-group-size\"=\"1,{}\" \
             \"no-trapping-math\"=\"true\"",
            max_local_threads(nodes)
        ),
        // `nvvm.maxntid` is the PTX `.maxntid` launch bound: ptxas budgets
        // registers per thread against it instead of the 1024-thread worst
        // case. Older LLVMs ignore the unknown string attribute (they only read
        // the `!nvvm.annotations` form), which merely costs the hint.
        LlvmTarget::Nvptx(_) => format!(
            "nounwind \"no-builtins\" \"no-trapping-math\"=\"true\" \"nvvm.maxntid\"=\"{}\"",
            max_local_threads(nodes)
        ),
    }
}

pub fn render(uop: &Arc<UOp>, name: Option<&str>) -> Result<RenderedKernel> {
    let renderer = LlvmTextRenderer::new();
    renderer.render(uop, name)
}

#[cfg(test)]
#[path = "../../test/unit/llvm_text.rs"]
mod tests;
