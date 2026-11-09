//! Proc-macros for svod.
//!
//! This crate provides:
//! - `#[derive(PatternEnum)]` for generating pattern matching infrastructure from Op enum
//! - `patterns!` macro for declarative pattern rewrite rules

use proc_macro::TokenStream;
use syn::{DeriveInput, parse_macro_input};

mod jit;
mod module;
mod pattern_enum;
mod patterns;

/// Derive macro for generating pattern matching infrastructure from an Op enum.
///
/// This macro analyzes your `Op` enum and generates:
/// - `OpKey` enum for O(1) pattern dispatch
/// - `OpKey::from_op()` method to extract the key from an `Op`
/// - the `alu` module: one marker per grouped op with `AluOp::{key, destructure}`
///
/// # Usage
///
/// ```ignore
/// #[derive(PatternEnum)]
/// #[pattern(grouped = [Unary, Binary, Ternary])]
/// pub enum Op {
///     Const(ConstValue),
///     Unary(UnaryOp, Arc<UOp>),
///     Binary(BinaryOp, Arc<UOp>, Arc<UOp>),
///     #[pattern(skip)]
///     Invalid,
/// }
/// ```
///
/// # Attributes
///
/// ## Enum-level
///
/// - `#[pattern(grouped = [Variant1, Variant2, ...])]` - Marks variants where the first
///   field is a sub-enum discriminant. For example, `Binary(BinaryOp, ...)` has `BinaryOp`
///   as a sub-discriminant, so `OpKey::Binary(BinaryOp::Add)` differs from `OpKey::Binary(BinaryOp::Mul)`.
///
/// ## Variant-level
///
/// - `#[pattern(skip)]` - Skip pattern generation for this variant (e.g., `Invalid`).
///
/// # Field Type Detection
///
/// The macro automatically classifies field types:
/// - `Arc<UOp>` → child operand (fixed arity)
/// - `SmallVec<[Arc<UOp>; N]>` or `Vec<Arc<UOp>>` → variadic children
/// - `Option<Arc<UOp>>` → optional child
/// - Other types → filter/metadata (e.g., `DType`, `DeviceSpec`)
///
/// # Generated Items
///
/// ```ignore
/// mod pattern_derived {
///     // Discriminant enum for O(1) dispatch
///     pub enum OpKey {
///         Const,
///         Unary(UnaryOp),
///         Binary(BinaryOp),
///         // ...
///     }
///
///     impl OpKey {
///         pub fn from_op(op: &Op) -> Self { ... }
///     }
/// }
/// ```
#[proc_macro_derive(PatternEnum, attributes(pattern))]
pub fn derive_pattern_enum(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match pattern_enum::generate(&input) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// Attribute macro giving every struct-like variant of the `Op` enum its own struct.
///
/// Write the enum with named fields as usual; the macro rewrites each such variant
/// to wrap a struct of the same name in a sibling `ops` module, adds
/// `From<ops::X> for Op`, and derives the same
/// pattern-matching infrastructure as [`PatternEnum`] from the original field
/// layout. Must precede `#[derive(...)]` so the derives see the rewritten enum;
/// `#[pattern(...)]` attributes are consumed.
#[proc_macro_attribute]
pub fn op_enum(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let item = parse_macro_input!(item as syn::ItemEnum);
    match pattern_enum::expand_op_enum(item) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// Proc-macro for declarative pattern rewrite rules.
///
/// Generates a `SimplifiedPatternMatcher` (in `svod_ir::pattern`) from a list of
/// rules, compiled to inline Rust matching with O(1) dispatch via `OpKey` and
/// Tinygrad-style early reject on the root's children.
///
/// # Syntax Overview
///
/// ```text
/// patterns! {
///     Add(x, @zero) => x,                       // rewrite: anything `IntoRewriteResult`
///     Mul(x, y) if is_power_of_two(y) => { .. }, // guard before the arrow
///     for op in binary [Add, Mul, Sub] {         // one rule per op; `[*]` for all
///         op(x, @zero) => x,
///     }
/// }
/// ```
///
/// The right-hand side may evaluate to `Arc<UOp>`, `Option<Arc<UOp>>` (`None`
/// declines) or a `RewriteResult`; `?` works inside it.
///
/// # Pattern Syntax
///
/// ```text
/// Add(x, y)                 // ALU op by kind; `Add[x, y]` also tries the swapped order
/// Cast { src, dtype }       // struct op: child fields nest, other fields are Rust patterns
/// Range { axis_type: AxisType::Upcast, .. }
/// Noop                      // unit op
/// Const(ConstValue::Int(0)) // Rust pattern over the ConstValue; `Const(v)` binds it
/// @zero / @one              // zero / one of any numeric type
/// c @const(cv)              // binds the UOp to `c` and its ConstValue to `cv`
/// c @vconst(vs) / c @anyconst(vs)
/// gate: Some(g) / gate: None
/// result @ Add(x, y)        // bind the whole match
/// Add(x, x)                 // repeated names must be the same node (`Arc::ptr_eq`)
/// ```
///
/// Op names resolve against `svod_ir::op::alu` (ALU kinds) and `svod_ir::ops`
/// (struct ops), so a typo is a normal resolution error at its span.
///
/// # Context Types
///
/// `@context MyContext;` at the start makes `ctx: &mut MyContext` available to
/// every rewrite body.
#[proc_macro]
pub fn patterns(input: TokenStream) -> TokenStream {
    let pattern_list = parse_macro_input!(input as patterns::PatternList);

    match patterns::generate_simplified_pattern_matcher(&pattern_list) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// Like `patterns!` but wraps the matcher in `LazyLock` for zero-cost reuse.
///
/// Returns `&'static SimplifiedPatternMatcher<C>` instead of an owned matcher.
/// The matcher is constructed only once on first call and cached globally.
///
/// Use this for stateless `pm_*()` functions that are called repeatedly
/// (e.g., once per kernel). Avoids re-constructing closures and hashmaps
/// on every call.
///
/// # Example
///
/// ```ignore
/// pub fn example_patterns() -> &'static TypedPatternMatcher {
///     cached_patterns! { ... }
/// }
/// ```
#[proc_macro]
pub fn cached_patterns(input: TokenStream) -> TokenStream {
    let pattern_list = parse_macro_input!(input as patterns::PatternList);

    match patterns::generate_cached_pattern_matcher(&pattern_list) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// Proc-macro for generating a JIT wrapper for a model.
///
/// Generates a struct with typed input/output buffer accessors and
/// prepare/execute methods for zero-overhead repeated inference.
///
/// The expansion refers to `::svod_tensor::jit` only, so the invoking crate
/// needs `svod-tensor` and nothing else.
///
/// # Syntax
///
/// Every clause is optional and order-free; a bare `name: Tensor` line at the
/// top level is an input, so the pre-v2 form parses unchanged.
///
/// ```ignore
/// jit_wrapper! {
///     MyModelJit(MyModel) {
///         inputs { spec: Tensor, #[unbatched] mask: Tensor, taps: [Tensor; 3] }
///         batch_var b: (1, model.config.max_batch_size),
///         vars { t: (1, 512) }
///         state { h: Tensor, caches: [Tensor; 8] }
///         outputs { logits, hidden: [Tensor; 2] }
///
///         build(spec, mask, taps, h, caches, t) {
///             model.forward_stream(spec, mask, taps, h, caches, &t)
///         }
///     }
/// }
/// ```
///
/// - `name: [Tensor; N]` declares N buffers behind one name: `prepare` takes
///   `[InputSpec; N]`, `build` gets `[&Tensor; N]`, accessors take an index.
/// - `batch_var` declares a variable **and** shrinks every batched input's
///   dim 0 to it after realization; `#[unbatched]` opts an input out.
/// - `state` slots are inputs the plan also writes: the build tuple is
///   `(declared outputs…, state values…)`, the macro assigns each new value
///   back into its own device-local buffer, and they are never exposed as
///   outputs.
/// - The build tuple has one element per declared output slot plus one per
///   state slot — and no tuple at all when there is exactly one of them.
/// - `MyModelJit<const W: usize>(MyModel)` — generics after the name land on
///   the struct and its impl and stay in scope inside `build`, so one wrapper
///   traces a different graph per instantiation.
///
/// # Generated API
///
/// - `new(model)`, `with_<var>_bound` / `_min_bound` / `_fixed`
/// - `prepare(<spec per slot>)` / `prepare_with_config(.., &PrepareConfig)`
/// - `<input>_mut()` / `<input>_host_write::<T>()` — buffer and typed host
///   write guard (`HostWrite`)
/// - `copy_output_to_<input>(out_pos, dst_off, src_off, len)`
/// - `<output>()`, `<output>_shape()`, `<output>_view::<T>()`,
///   `<output>_to_vec::<T>()` — the shape and both reads follow the live
///   variable bindings
/// - `reset()` — zero every state slot
/// - `execute()`, `execute_bound(<var values>)`, `execute_with_vars()`,
///   the `_profiled` variants
/// - `replicate()` — deep copy of the prepared JIT (`Result<Self>`): forked
///   input/intermediate/output buffers, shared model/weights and compiled
///   kernels, independent backend queue for concurrent execution
#[proc_macro]
pub fn jit_wrapper(input: TokenStream) -> TokenStream {
    let jit = parse_macro_input!(input as jit::JitWrapper);
    match jit::generate(jit) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// Derive `svod_tensor::nn::Module`: a state dict built from the field types.
///
/// Fields are classified by type syntax — a last path segment of `Tensor` is a
/// parameter, `Option<Tensor>` an optional parameter, primitives and containers
/// of primitives are ignored, and everything else delegates to that field's own
/// `Module` impl. Keys always go through `nn::prefixed`, so a root module
/// (`prefix == ""`) never emits a leading dot.
///
/// # Attributes
///
/// - `#[module(key = "attention.q_proj")]` — replaces the field-name segment;
///   may contain dots or digits. `key = ""` passes the parent prefix through
///   unchanged (flatten).
/// - `#[module(skip)]` — ignore a non-primitive field (config, dtype, mode).
/// - `#[module(optional)]` — required on `Option<Tensor>`: save when `Some`,
///   load with an absent-tolerant lookup.
/// - `#[module(optional = "<predicate over self>")]` — the key is required when
///   the predicate holds and skipped otherwise.
/// - `#[module(crate = "::my_tensor")]` on the type — where the `nn` module lives.
///
/// Enum variants derive too: a newtype variant is transparent (same prefix), a
/// tuple variant indexes `.0`, `.1`, …, and a struct variant uses field names.
/// `#[module(key = "…")]` on a variant nests all of its fields one segment deeper.
///
/// The derive also emits an inherent
/// `const MODULE_FIELDS: &'static [(&'static str, &'static str)]` mapping each
/// weight-carrying named field ident to its key segment.
#[proc_macro_derive(Module, attributes(module))]
pub fn derive_module(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match module::generate(&input) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.to_compile_error().into(),
    }
}
