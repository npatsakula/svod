//! C source code rendering for individual UOp operations.
//!
//! Generates C expressions/statements for each Op variant.
//! Uses SSA inlining: single-use values are inlined as expressions,
//! multi-use values get local variable declarations.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use svod_dtype::{DType, ScalarDType};
use svod_ir::{BinaryOp, Op, ReduceOp, TernaryOp, UnaryOp, prelude::*};

use super::dialect::{CDialect, addr_qualifier};
use super::types::{MathFn, c_cast, c_dtype, math_fn, pointer_type, vector_literal};
use crate::common::{access_dtype, format_custom_template_strict, shaped_dtype};
use svod_ir::ops;

/// Context for C code generation, tracking variable names and SSA inlining.
pub struct CContext {
    /// UOp ID -> C expression or variable name
    names: HashMap<u64, String>,
    /// UOp ID -> reference count (how many times used)
    ref_counts: HashMap<u64, usize>,
    /// Variable counter for generating unique names
    counter: usize,
    /// Current indentation depth
    depth: usize,
    /// Pending reduce accumulator info: reduce_id -> (acc_name, dtype)
    pending_reduces: HashMap<u64, (String, DType)>,
    /// UOp IDs that escape their declaration scope — need function-scope declaration.
    scope_escaping: HashSet<u64>,
    /// Function-scope declarations for hoisted variables (emitted before kernel body).
    pub hoisted_declarations: Vec<String>,
    /// Side-channel error set by `render_uop` when it detects a graph invariant
    /// violation. The render loop drains this after each call and propagates as
    /// a typed [`crate::Error`].
    pending_error: Option<crate::Error>,
    /// Which C-family language is being emitted.
    pub dialect: CDialect,
}

impl CContext {
    pub fn new(ref_counts: HashMap<u64, usize>, scope_escaping: HashSet<u64>, dialect: CDialect) -> Self {
        Self {
            dialect,
            names: HashMap::new(),
            ref_counts,
            counter: 0,
            depth: 1,
            pending_reduces: HashMap::new(),
            scope_escaping,
            hoisted_declarations: Vec::new(),
            pending_error: None,
        }
    }

    /// Record an `InvalidGraph` error from a renderer op handler.
    pub fn set_invalid_graph(&mut self, reason: impl Into<String>) {
        if self.pending_error.is_none() {
            self.pending_error = Some(crate::Error::InvalidGraph { reason: reason.into() });
        }
    }

    /// Record an `UnsupportedOp` error from a renderer op handler that reached an
    /// op variant it cannot lower.
    pub fn set_unsupported_op(&mut self, op: impl Into<String>) {
        if self.pending_error.is_none() {
            self.pending_error = Some(crate::Error::UnsupportedOp { op: op.into() });
        }
    }

    /// Drain any error recorded via [`Self::set_invalid_graph`].
    pub fn take_error(&mut self) -> Option<crate::Error> {
        self.pending_error.take()
    }

    /// Get the C expression for a UOp. Panics if not registered.
    pub fn get(&self, uop: &Arc<UOp>) -> &str {
        self.names
            .get(&uop.id)
            .map(|s| s.as_str())
            .unwrap_or_else(|| panic!("UOp {} ({}) not in C context", uop.id, uop.op().as_ref()))
    }

    /// Register a name/expression for a UOp ID.
    pub fn register(&mut self, id: u64, expr: String) {
        self.names.insert(id, expr);
    }

    /// Check if a value should be inlined (single-use, expression-safe).
    pub fn should_inline(&self, id: u64) -> bool {
        self.ref_counts.get(&id).copied().unwrap_or(0) <= 1
    }

    /// Generate a unique variable name with given prefix.
    pub fn next_name(&mut self, prefix: &str) -> String {
        let name = format!("{}{}", prefix, self.counter);
        self.counter += 1;
        name
    }

    /// Get current indentation string.
    pub fn indent(&self) -> String {
        "  ".repeat(self.depth)
    }

    /// Increase indentation depth.
    pub fn push_indent(&mut self) {
        self.depth += 1;
    }

    /// Decrease indentation depth.
    pub fn pop_indent(&mut self) {
        self.depth = self.depth.saturating_sub(1);
    }

    /// Register a pending reduce final load.
    pub fn register_reduce_pending(&mut self, reduce_id: u64, acc_name: String, dtype: DType) {
        self.pending_reduces.insert(reduce_id, (acc_name, dtype));
    }

    /// Take all pending reduces.
    pub fn take_pending_reduces(&mut self) -> HashMap<u64, (String, DType)> {
        std::mem::take(&mut self.pending_reduces)
    }

    /// Emit a C expression, either as an inline expression or a variable declaration.
    /// Returns the name/expression to reference this value.
    ///
    /// Variables that escape their declaration scope are hoisted: declared at function
    /// scope and assigned at current depth. This prevents "use of undeclared identifier"
    /// errors when the linearizer places a shared node inside a loop but consumers exist
    /// outside the loop.
    pub fn emit_expr(&mut self, uop: &Arc<UOp>, expr: String, prefix: &str, kernel: &mut Vec<String>) -> String {
        if self.should_inline(uop.id) {
            self.register(uop.id, expr.clone());
            expr
        } else {
            let dtype = shaped_dtype(uop);
            self.emit_expr_dtype(uop, expr, prefix, kernel, &dtype, false)
        }
    }

    /// Register an address (INDEX/SHRINK/pointer CAST) expression.
    ///
    /// Addresses are inlined like tinygrad's `cstyle.py` INDEX rendering, which
    /// is safe there because its linearizer never places a node inside a range
    /// that is consumed outside it. Morok's can, so an escaping address is
    /// hoisted to function scope and assigned at its declaration depth —
    /// otherwise the inlined text references a loop variable that is out of
    /// scope at the use site.
    pub fn emit_address(&mut self, uop: &Arc<UOp>, expr: String, kernel: &mut Vec<String>) -> String {
        if !self.scope_escaping.contains(&uop.id) {
            self.register(uop.id, expr.clone());
            return expr;
        }
        let declared = if is_address_value(uop) {
            let dtype = uop.dtype();
            let elem = match &dtype {
                DType::Ptr { base, .. } => base.as_ref(),
                other => other,
            };
            pointer_type(elem, uop.addrspace(), self.dialect)
        } else {
            c_dtype(&shaped_dtype(uop), self.dialect)
        };
        let name = self.next_name("bidx");
        let indent = self.indent();
        self.hoisted_declarations.push(format!("  {declared} {name};"));
        kernel.push(format!("{indent}{name} = {expr};"));
        self.register(uop.id, name.clone());
        name
    }

    pub(super) fn emit_expr_dtype(
        &mut self,
        uop: &Arc<UOp>,
        expr: String,
        prefix: &str,
        kernel: &mut Vec<String>,
        dtype: &DType,
        allow_inline: bool,
    ) -> String {
        if allow_inline && self.should_inline(uop.id) {
            self.register(uop.id, expr.clone());
            expr
        } else {
            let name = self.next_name(prefix);
            self.emit_named(uop, expr, name, dtype, kernel)
        }
    }

    /// Declare `name` with `dtype` at the current depth (hoisted to function
    /// scope when the value escapes). The name is fixed by the caller — SPECIAL
    /// axes keep their IR name.
    pub(super) fn emit_named(
        &mut self,
        uop: &Arc<UOp>,
        expr: String,
        name: String,
        dtype: &DType,
        kernel: &mut Vec<String>,
    ) -> String {
        let dtype = c_dtype(dtype, self.dialect);
        let indent = self.indent();
        if self.scope_escaping.contains(&uop.id) {
            // Hoist: declare at function scope, assign at current depth
            self.hoisted_declarations.push(format!("  {dtype} {name};"));
            kernel.push(format!("{indent}{name} = {expr};"));
        } else {
            kernel.push(format!("{indent}{dtype} {name} = {expr};"));
        }
        self.register(uop.id, name.clone());
        name
    }
}

/// Render a single UOp to C source code.
///
/// Returns `Some(())` if code was emitted, `None` for meta-ops.
pub fn render_uop(uop: &Arc<UOp>, ctx: &mut CContext, kernel: &mut Vec<String>) -> Option<()> {
    match uop.op() {
        Op::Special(ops::Special { name, .. }) if ctx.dialect.is_metal() => {
            super::metal::render_special(uop, name, ctx, kernel)
        }
        Op::Barrier(..) if ctx.dialect.is_metal() => {
            let indent = ctx.indent();
            kernel.push(format!("{indent}threadgroup_barrier(mem_flags::mem_threadgroup);"));
            ctx.register(uop.id, String::new());
            Some(())
        }
        Op::Wmma(ops::Wmma { a, b, c, metadata }) if ctx.dialect.is_metal() => {
            super::metal::render_wmma(uop, a, b, c, metadata, ctx, kernel)
        }

        // Meta-ops: no code emitted
        Op::Const(_)
        | Op::VConst(..)
        | Op::Param(..)
        | Op::DefineVar(..)
        | Op::Noop
        | Op::Sink(..)
        | Op::Group(..)
        | Op::Unique(_)
        | Op::Call(..)
        | Op::Barrier(..) => None,

        Op::Buffer(ops::Buffer { arg, .. }) if arg.addrspace == Some(svod_ir::AddrSpace::Reg) => {
            let base_dtype = arg.dtype.clone();
            let alloc_size = uop.buffer_size().unwrap_or(1);
            let name = ctx.next_name("reg");
            let indent = ctx.indent();
            kernel.push(format!("{indent}{} {name}[{alloc_size}];", c_dtype(&base_dtype, ctx.dialect)));
            ctx.register(uop.id, name);
            Some(())
        }

        Op::Buffer(..) => None,

        Op::Index(ops::Index { buffer, indices, .. }) => {
            let buf = ctx.get(buffer).to_string();

            if indices.is_empty() {
                // No index - just alias the buffer pointer
                ctx.emit_address(uop, buf, kernel);
            } else {
                let idx = if indices.len() == 1 {
                    ctx.get(&indices[0]).to_string()
                } else {
                    ctx.set_invalid_graph(format!(
                        "C renderer requires linearized INDEX (single-axis), found {} indices on uop {}",
                        indices.len(),
                        uop.id
                    ));
                    return None;
                };
                // Tinygrad render_index: ALU values use lane extraction; values
                // carrying an address space remain addresses for LOAD/STORE.
                let expr =
                    if buffer.addrspace().is_none() { format!("({buf})[{idx}]") } else { format!("{buf} + {idx}") };
                ctx.emit_address(uop, expr, kernel);
            }
            Some(())
        }

        Op::Shrink(ops::Shrink { src, offsets, sizes: _ }) => {
            let expr = format!("{} + {}", ctx.get(src), ctx.get(offsets));
            ctx.emit_address(uop, expr, kernel);
            Some(())
        }

        Op::Load(ops::Load { index, alt, gate }) => {
            // Defense-in-depth: `UOp::new` (ir hash_consing.rs `new_tagged`) already
            // asserts the alt/gate pairing, the bool gate and the alt dtype, so no
            // legal construction path reaches these branches.
            if alt.is_some() != gate.is_some() {
                ctx.set_invalid_graph(format!("LOAD on uop {} must have either neither or both alt and gate", uop.id));
                return None;
            }
            if let (Some(alt), Some(gate)) = (alt, gate) {
                if gate.dtype() != DType::Bool {
                    ctx.set_invalid_graph(format!("gated LOAD on uop {} requires a scalar bool gate", uop.id));
                    return None;
                }
                if alt.dtype() != uop.dtype() {
                    ctx.set_invalid_graph(format!(
                        "gated LOAD on uop {} requires alt dtype to match the load dtype",
                        uop.id
                    ));
                    return None;
                }
            }
            let idx = ctx.get(index).to_string();
            let load_dtype = shaped_dtype(uop);
            let deref_expr = render_access(index, &load_dtype, &idx, ctx.dialect);
            let expr = match (alt, gate) {
                (None, None) => deref_expr,
                (Some(alt), Some(gate)) => {
                    format!("({} ? {deref_expr} : {})", ctx.get(gate), ctx.get(alt))
                }
                _ => unreachable!(),
            };
            // Tinygrad materializes non-register loads even when they have one
            // consumer. This keeps address expressions from being duplicated
            // through later aliases.
            ctx.emit_expr_dtype(
                uop,
                expr,
                "val",
                kernel,
                &load_dtype,
                index.addrspace() == Some(svod_ir::AddrSpace::Reg),
            );
            Some(())
        }

        Op::Store(ops::Store { index, value, gate }) => {
            if gate.is_some() {
                ctx.set_invalid_graph(format!(
                    "gated STORE on uop {} reached C codegen; linear cleanup must rewrite it to IF/STORE/ENDIF",
                    uop.id
                ));
                return None;
            }
            let idx = ctx.get(index).to_string();
            let val = ctx.get(value).to_string();
            let indent = ctx.indent();
            let val_dtype = access_dtype(index, value);
            kernel.push(format!("{indent}{} = {val};", render_access(index, &val_dtype, &idx, ctx.dialect)));
            Some(())
        }

        Op::Binary(op, lhs, rhs) => {
            let l = ctx.get(lhs).to_string();
            let r = ctx.get(rhs).to_string();
            let expr = render_binary(*op, &l, &r, &lhs.dtype(), ctx.dialect);
            ctx.emit_expr(uop, expr, "alu", kernel);
            Some(())
        }

        Op::Unary(op, src) => {
            let s = ctx.get(src).to_string();
            let expr = render_unary(*op, &s, &src.dtype(), ctx.dialect);
            ctx.emit_expr(uop, expr, "alu", kernel);
            Some(())
        }

        Op::Ternary(TernaryOp::Where, cond, t, f) => {
            let c = ctx.get(cond).to_string();
            let tv = ctx.get(t).to_string();
            let fv = ctx.get(f).to_string();
            let expr = format!("({c} ? {tv} : {fv})");
            // WHERE is always an SSA boundary in Tinygrad's C renderer. Inlining
            // nested Threefry selects causes exponential expression growth.
            let dtype = shaped_dtype(uop);
            ctx.emit_expr_dtype(uop, expr, "alu", kernel, &dtype, false);
            Some(())
        }

        Op::Ternary(TernaryOp::MulAcc, a, b, c) => {
            let av = ctx.get(a).to_string();
            let bv = ctx.get(b).to_string();
            let cv = ctx.get(c).to_string();
            let expr = if a.dtype().is_float() {
                let dtype = a.dtype();
                let d = ctx.dialect;
                let args = [av, bv, cv].map(|arg| bf16_arg(&arg, &dtype, d));
                narrow_float(
                    format!("{}({}, {}, {})", math_fn(MathFn::Fma, &dtype, d), args[0], args[1], args[2]),
                    &dtype,
                    d,
                )
            } else {
                format!("(({av} * {bv}) + {cv})")
            };
            ctx.emit_expr(uop, expr, "alu", kernel);
            Some(())
        }

        Op::Cast(ops::Cast { src, dtype }) => {
            let s = ctx.get(src).to_string();
            if is_address_value(src) {
                let elem = match dtype {
                    DType::Ptr { base, .. } => base.as_ref(),
                    other => other,
                };
                let expr = format!("(({})({s}))", pointer_type(elem, uop.addrspace(), ctx.dialect));
                ctx.emit_address(uop, expr, kernel);
                return Some(());
            }
            let rendered_dtype = shaped_dtype(uop);

            // Vector casts use __builtin_convertvector for element-wise conversion
            // (a plain C cast would reinterpret bits, not convert values)
            let expr = if rendered_dtype.vcount() > 1 && !matches!(rendered_dtype, DType::Ptr { .. }) {
                format!("__builtin_convertvector({s}, {})", c_dtype(&rendered_dtype, ctx.dialect))
            } else {
                c_cast(&s, &src.dtype(), dtype, ctx.dialect)
            };
            let allow_inline = rendered_dtype.vcount() == 1;
            ctx.emit_expr_dtype(uop, expr, "cast", kernel, &rendered_dtype, allow_inline);
            Some(())
        }

        Op::BitCast(ops::BitCast { src, dtype: _ }) => {
            let s = ctx.get(src).to_string();
            let from_type = c_dtype(&shaped_dtype(src), ctx.dialect);
            let rendered_dtype = shaped_dtype(uop);
            let to_type = c_dtype(&rendered_dtype, ctx.dialect);
            if from_type == to_type {
                ctx.register(uop.id, s);
            } else {
                let expr = match ctx.dialect {
                    CDialect::Clang => format!("__builtin_bit_cast({to_type}, ({from_type})({s}))"),
                    CDialect::Metal => format!("as_type<{to_type}>(({from_type})({s}))"),
                };
                ctx.emit_expr(uop, expr, "cast", kernel);
            }
            Some(())
        }

        Op::Reshape(ops::Reshape { src, .. }) => {
            let s = ctx.get(src).to_string();
            ctx.register(uop.id, s);
            Some(())
        }

        Op::Range(ops::Range { end, axis_id, .. }) => {
            let end_val = ctx.get(end).to_string();
            let id = axis_id.name();
            let range_dtype = c_dtype(&uop.dtype(), ctx.dialect);
            let var_name = format!("ridx{id}");
            let indent = ctx.indent();
            kernel.push(format!("{indent}for ({range_dtype} {var_name} = 0; {var_name} < {end_val}; {var_name}++) {{"));
            ctx.register(uop.id, var_name);
            ctx.push_indent();
            Some(())
        }

        Op::End(ops::End { ranges, .. }) => {
            for range in ranges.iter() {
                if let Op::Range(..) = range.op() {
                    ctx.pop_indent();
                    let indent = ctx.indent();
                    kernel.push(format!("{indent}}}"));
                }
            }

            // After closing loops, resolve pending reduces.
            // In C, the accumulator variable already holds the final value
            // (unlike LLVM where we need to load from alloca).
            let pending = ctx.take_pending_reduces();
            for (reduce_id, (acc_name, _dtype)) in pending {
                // Re-register the reduce with the accumulator name
                // so downstream users reference the accumulated value.
                ctx.register(reduce_id, acc_name);
            }
            Some(())
        }

        Op::Reduce(ops::Reduce { src, ranges, reduce_op, .. }) => {
            let src_val = ctx.get(src).to_string();
            let dtype = &uop.dtype();

            if ranges.is_empty() {
                // Passthrough reduce
                ctx.register(uop.id, src_val);
            } else {
                // Accumulator was pre-declared in mod.rs with name acc{uop.id}
                let acc_name = ctx.get(uop).to_string();
                let indent = ctx.indent();

                let acc_expr = render_reduce_accumulate(*reduce_op, &acc_name, &src_val, dtype, ctx.dialect);
                kernel.push(format!("{indent}{acc_expr}"));

                // Register pending for End to emit the final value
                ctx.register_reduce_pending(uop.id, acc_name, dtype.clone());
            }
            Some(())
        }

        Op::Stack(ops::Stack { sources }) => {
            if sources.is_empty() {
                return None;
            }
            let vals: Vec<String> = sources.iter().map(|source| ctx.get(source).to_string()).collect();
            if matches!(uop.dtype(), DType::Ptr { .. }) {
                // Ptr types can't be vectorized in C (no compound literal for pointers).
                // All elements should be the same scalar pointer — use the first one.
                ctx.emit_expr(uop, vals[0].clone(), "vec", kernel);
            } else {
                let packed_dtype = uop.dtype().scalar_dtype().vec(sources.len()).expect("STACK source dtype is scalar");
                let out_dtype = c_dtype(&packed_dtype, ctx.dialect);
                let expr = vector_literal(&out_dtype, &vals.join(", "), ctx.dialect);
                ctx.emit_expr_dtype(uop, expr, "vec", kernel, &packed_dtype, true);
            }
            Some(())
        }

        Op::Shuffle(ops::Shuffle { src, mask }) => {
            let s = ctx.get(src).to_string();
            let lanes: Vec<String> = mask.iter().map(|lane| format!("{s}[{lane}]")).collect();
            let packed_dtype = uop.dtype().scalar_dtype().vec(mask.len()).expect("a SHUFFLE takes a vector");
            let out_dtype = c_dtype(&packed_dtype, ctx.dialect);
            let expr = vector_literal(&out_dtype, &lanes.join(", "), ctx.dialect);
            ctx.emit_expr_dtype(uop, expr, "shuf", kernel, &packed_dtype, true);
            Some(())
        }

        Op::CustomI(ops::CustomI { deps, code }) => {
            let args: Vec<String> = deps.iter().map(|dep| ctx.get(dep).to_string()).collect();
            let expr = match format_custom_template_strict(code, &args) {
                Ok(s) => s,
                Err(e) => {
                    ctx.set_invalid_graph(format!("CUSTOMI template error on uop {}: {e}", uop.id));
                    return None;
                }
            };
            // CUSTOMI is always inline in Tinygrad's cstyle renderer.
            ctx.register(uop.id, expr);
            Some(())
        }

        Op::Custom(ops::Custom { deps, code }) => {
            // Scheduling markers are LLVM line comments carried on a Void CUSTOM;
            // `;` is not a comment in MSL, so drop them rather than splice them.
            if crate::llvm::sched::is_scheduling_marker(code) {
                ctx.register(uop.id, String::new());
                return Some(());
            }
            let args: Vec<String> = deps.iter().map(|dep| ctx.get(dep).to_string()).collect();
            let rendered = match format_custom_template_strict(code, &args) {
                Ok(s) => s,
                Err(e) => {
                    ctx.set_invalid_graph(format!("CUSTOM template error on uop {}: {e}", uop.id));
                    return None;
                }
            };
            let indent = ctx.indent();

            if uop.dtype() == DType::Void {
                let stmt = if rendered.trim_end().ends_with(';') { rendered } else { format!("{rendered};") };
                kernel.push(format!("{indent}{stmt}"));
                ctx.register(uop.id, String::new());
            } else {
                let name = ctx.next_name("custom");
                let dtype = c_dtype(&uop.dtype(), ctx.dialect);
                if ctx.scope_escaping.contains(&uop.id) {
                    ctx.hoisted_declarations.push(format!("  {dtype} {name};"));
                    kernel.push(format!("{indent}{name} = {rendered};"));
                } else {
                    kernel.push(format!("{indent}{dtype} {name} = {rendered};"));
                }
                ctx.register(uop.id, name);
            }
            Some(())
        }

        Op::Detach(ops::Detach { src }) => {
            let s = ctx.get(src).to_string();
            ctx.register(uop.id, s);
            None
        }

        Op::After(ops::After { passthrough, .. }) => {
            assert!(
                !matches!(passthrough.op(), Op::Group(..)),
                "BUG: AFTER passthrough is GROUP (id={}). AFTER tree:\n{}",
                passthrough.id,
                uop.tree()
            );
            let s = ctx.get(passthrough).to_string();
            ctx.register(uop.id, s);
            None
        }

        Op::Bind(ops::Bind { var, value }) => {
            let v = ctx.get(value).to_string();
            ctx.register(var.id, v);
            None
        }

        Op::If(ops::If { condition, .. }) => {
            let cond = ctx.get(condition).to_string();
            let indent = ctx.indent();
            kernel.push(format!("{indent}if ({cond}) {{"));
            ctx.push_indent();
            Some(())
        }

        Op::EndIf(..) => {
            ctx.pop_indent();
            let indent = ctx.indent();
            kernel.push(format!("{indent}}}"));
            Some(())
        }

        op => {
            // An op variant the C backend has no lowering for. Surface it as a typed
            // error instead of emitting a comment + None that would detonate later
            // when a consumer calls `ctx.get` on this missing value.
            ctx.set_unsupported_op(op.as_ref());
            None
        }
    }
}

/// Direct equivalent of Tinygrad CStyleLanguage.render_access.
fn render_access(index: &Arc<UOp>, access_dtype: &DType, address: &str, d: CDialect) -> String {
    let source_dtype = match index.op() {
        Op::Index(ops::Index { buffer, .. }) => buffer.dtype(),
        Op::Shrink(ops::Shrink { src, .. }) => src.dtype(),
        Op::Cast(ops::Cast { src, .. }) => src.dtype(),
        _ => index.dtype(),
    };
    if access_dtype.vcount() > 1 || *access_dtype != source_dtype {
        format!("*(({}{}*)({address}))", addr_qualifier(d, index.addrspace()), c_dtype(access_dtype, d))
    } else {
        format!("*({address})")
    }
}

/// C evaluates scalar arithmetic on types narrower than `int` at `int` width,
/// and an inlined expression never rounds back down. Cast such results to the
/// IR dtype so `int8` wraps like every other backend.
fn narrow_int(expr: String, dtype: &DType, d: CDialect) -> String {
    let narrow =
        matches!(dtype.base(), ScalarDType::Int8 | ScalarDType::UInt8 | ScalarDType::Int16 | ScalarDType::UInt16);
    if narrow && dtype.vcount() == 1 { format!("(({}){expr})", c_dtype(dtype, d)) } else { expr }
}

/// The `f`-suffixed builtins compute at f32; round a half or bfloat16 result
/// back to its own width so the C backend matches the native-width backends.
fn narrow_float(expr: String, dtype: &DType, d: CDialect) -> String {
    if matches!(dtype.base(), ScalarDType::Float16 | ScalarDType::BFloat16) && dtype.vcount() == 1 {
        format!("(({}){expr})", c_dtype(dtype, d))
    } else {
        expr
    }
}

/// MSL has no `bfloat` overloads of the math functions; promote the operand
/// (`narrow_float` rounds the result back). Identity for clang, whose
/// `__builtin_*f` promote implicitly.
fn bf16_arg(s: &str, dtype: &DType, d: CDialect) -> String {
    if d.is_metal() && dtype.base() == ScalarDType::BFloat16 { format!("(float){s}") } else { s.to_string() }
}

/// `fn(args...)` through the dialect's math function, with bf16 promotion.
fn math_call(f: MathFn, args: &[&str], dtype: &DType, d: CDialect) -> String {
    let args: Vec<String> = args.iter().map(|arg| bf16_arg(arg, dtype, d)).collect();
    narrow_float(format!("{}({})", math_fn(f, dtype, d), args.join(", ")), dtype, d)
}

/// Render a binary operation as a C expression.
fn render_binary(op: BinaryOp, l: &str, r: &str, dtype: &DType, d: CDialect) -> String {
    match op {
        BinaryOp::FloorDiv | BinaryOp::FloorMod => unreachable!("floor div/mod must be decomposed before C rendering"),
        BinaryOp::Add => narrow_int(format!("({l} + {r})"), dtype, d),
        BinaryOp::Sub => narrow_int(format!("({l} - {r})"), dtype, d),
        BinaryOp::Mul => narrow_int(format!("({l} * {r})"), dtype, d),
        BinaryOp::Fdiv => format!("({l} / {r})"),
        BinaryOp::CDiv => narrow_int(format!("({l} / {r})"), dtype, d),
        BinaryOp::CMod => {
            if dtype.is_float() {
                math_call(MathFn::Fmod, &[l, r], dtype, d)
            } else {
                narrow_int(format!("({l} % {r})"), dtype, d)
            }
        }
        BinaryOp::Max => {
            if dtype.is_float() {
                math_call(MathFn::Fmax, &[l, r], dtype, d)
            } else {
                narrow_int(format!("({l} > {r} ? {l} : {r})"), dtype, d)
            }
        }
        BinaryOp::Lt => format!("({l} < {r})"),
        BinaryOp::Le => format!("({l} <= {r})"),
        BinaryOp::Gt => format!("({l} > {r})"),
        BinaryOp::Ge => format!("({l} >= {r})"),
        BinaryOp::Eq => format!("({l} == {r})"),
        BinaryOp::Ne => format!("({l} != {r})"),
        BinaryOp::And => narrow_int(format!("({l} & {r})"), dtype, d),
        BinaryOp::Or => narrow_int(format!("({l} | {r})"), dtype, d),
        BinaryOp::Xor => narrow_int(format!("({l} ^ {r})"), dtype, d),
        BinaryOp::Shl => narrow_int(format!("({l} << {r})"), dtype, d),
        BinaryOp::Shr => narrow_int(format!("({l} >> {r})"), dtype, d),
        BinaryOp::Pow => {
            if dtype.is_float() {
                math_call(MathFn::Pow, &[l, r], dtype, d)
            } else {
                // Integer pow through the widest float the dialect has.
                let (wide, pow) = match d {
                    CDialect::Clang => ("double", "__builtin_pow"),
                    CDialect::Metal => ("float", "pow"),
                };
                format!("(({}){pow}(({wide}){l}, ({wide}){r}))", c_dtype(&DType::Scalar(dtype.base()), d))
            }
        }
        BinaryOp::Threefry => narrow_int(format!("({l} ^ {r})"), dtype, d),
    }
}

/// Render a unary operation as a C expression.
fn render_unary(op: UnaryOp, s: &str, dtype: &DType, d: CDialect) -> String {
    let one = if matches!(dtype.base(), ScalarDType::Float64) { "1.0" } else { "1.0f" };
    match op {
        UnaryOp::Neg => narrow_int(format!("(-{s})"), dtype, d),
        UnaryOp::Not => {
            if dtype.is_bool() {
                format!("(!{s})")
            } else {
                narrow_int(format!("(~{s})"), dtype, d)
            }
        }
        UnaryOp::Abs => {
            if dtype.is_float() {
                math_call(MathFn::Fabs, &[s], dtype, d)
            } else {
                narrow_int(format!("({s} < 0 ? -{s} : {s})"), dtype, d)
            }
        }
        UnaryOp::Sqrt => math_call(MathFn::Sqrt, &[s], dtype, d),
        UnaryOp::Rsqrt => {
            narrow_float(format!("({one} / {}({}))", math_fn(MathFn::Sqrt, dtype, d), bf16_arg(s, dtype, d)), dtype, d)
        }
        UnaryOp::Reciprocal => format!("({one} / {s})"),
        UnaryOp::Exp => math_call(MathFn::Exp, &[s], dtype, d),
        UnaryOp::Exp2 => math_call(MathFn::Exp2, &[s], dtype, d),
        UnaryOp::Log => math_call(MathFn::Log, &[s], dtype, d),
        UnaryOp::Log2 => math_call(MathFn::Log2, &[s], dtype, d),
        UnaryOp::Sin => math_call(MathFn::Sin, &[s], dtype, d),
        UnaryOp::Cos => math_call(MathFn::Cos, &[s], dtype, d),
        UnaryOp::Tan => math_call(MathFn::Tan, &[s], dtype, d),
        UnaryOp::Floor => math_call(MathFn::Floor, &[s], dtype, d),
        UnaryOp::Ceil => math_call(MathFn::Ceil, &[s], dtype, d),
        UnaryOp::Trunc => math_call(MathFn::Trunc, &[s], dtype, d),
        UnaryOp::Round => math_call(MathFn::Rint, &[s], dtype, d),
        UnaryOp::Erf => math_call(MathFn::Erf, &[s], dtype, d),
        UnaryOp::Sign => {
            if dtype.is_float() {
                let zero = if matches!(dtype.base(), ScalarDType::Float64) { "0.0" } else { "0.0f" };
                format!("(({s} > {zero}) - ({s} < {zero}))")
            } else {
                format!("(({s} > 0) - ({s} < 0))")
            }
        }
        UnaryOp::Square => format!("({s} * {s})"),
    }
}

/// Render a reduce accumulation statement.
fn render_reduce_accumulate(op: ReduceOp, acc: &str, val: &str, dtype: &DType, d: CDialect) -> String {
    let float_fn = |f: MathFn| {
        format!("{acc} = {}({}, {});", math_fn(f, dtype, d), bf16_arg(acc, dtype, d), bf16_arg(val, dtype, d))
    };
    match op {
        ReduceOp::Add => format!("{acc} += {val};"),
        ReduceOp::Mul => format!("{acc} *= {val};"),
        ReduceOp::Max => {
            if dtype.is_float() {
                float_fn(MathFn::Fmax)
            } else {
                format!("{acc} = ({acc} > {val} ? {acc} : {val});")
            }
        }
        ReduceOp::Min => {
            if dtype.is_float() {
                float_fn(MathFn::Fmin)
            } else {
                format!("{acc} = ({acc} < {val} ? {acc} : {val});")
            }
        }
    }
}

fn is_address_value(uop: &Arc<UOp>) -> bool {
    match uop.op() {
        Op::Param(ops::Param { arg, .. }) | Op::Buffer(ops::Buffer { arg, .. }) => arg.addrspace.is_some(),
        Op::Slice(..) => true,
        Op::Index(ops::Index { buffer, .. }) => is_address_value(buffer),
        Op::Shrink(ops::Shrink { src, .. })
        | Op::Cast(ops::Cast { src, .. })
        | Op::After(ops::After { passthrough: src, .. })
        | Op::Precast(ops::Precast { src }) => is_address_value(src),
        _ => false,
    }
}

/// Count references for each UOp ID in the linearized stream.
/// Used to determine which values should be inlined vs declared.
pub fn count_references(nodes: &[Arc<UOp>]) -> HashMap<u64, usize> {
    let mut counts: HashMap<u64, usize> = HashMap::new();
    for node in nodes {
        for child in node.op().children() {
            *counts.entry(child.id).or_insert(0) += 1;
        }
    }
    counts
}
