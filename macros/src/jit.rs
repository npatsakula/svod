use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{
    Attribute, Error, Expr, Generics, Ident, LitInt, Result, Token, Type, braced, bracketed,
    parse::{Parse, ParseStream},
    token::{Brace, Comma, Paren},
};

pub(crate) struct JitWrapper {
    name: Ident,
    /// Optional wrapper generics (`RnntBlockJit<const W: usize>`): declared on
    /// the struct and its impl, and in scope inside `build`, so one wrapper can
    /// be instantiated at several compile-time shapes.
    generics: Generics,
    model_ty: Type,
    /// Host-written inputs, in `prepare` order. A bare `name: Tensor` line at
    /// the top level and an `inputs { .. }` entry produce the same slot.
    inputs: Vec<Slot>,
    /// Inputs that are also outputs: the build tuple carries a new value for
    /// each, and the macro assigns it back into the input's own storage.
    state: Vec<Slot>,
    /// Declared outputs. Empty (with no `state`) = the classic single-output
    /// form where `build` returns one `Tensor`.
    outputs: Vec<Slot>,
    vars: Vec<VarDecl>,
    /// Index into `vars` of the `batch_var` declaration, if any.
    batch_var: Option<usize>,
    build_args: Vec<Ident>,
    build_body: TokenStream,
}

/// One declared slot: a single `Tensor`, or `[Tensor; N]` — N buffers behind
/// one name, indexed by the generated accessors.
struct Slot {
    name: Ident,
    count: Option<usize>,
    /// `#[unbatched]`: opt out of the `batch_var` dim-0 shrink.
    unbatched: bool,
}

impl Slot {
    fn leaves(&self) -> usize {
        self.count.unwrap_or(1)
    }

    /// Unique identifier for leaf `i`; the bare name for a scalar slot.
    fn leaf(&self, i: usize) -> Ident {
        match self.count {
            Some(_) => format_ident!("{}_{}", self.name, i),
            None => self.name.clone(),
        }
    }

    fn label(&self, i: usize) -> String {
        match self.count {
            Some(_) => format!("{}[{i}]", self.name),
            None => self.name.to_string(),
        }
    }

    /// The `InputSpec` parameter type this slot contributes to `prepare`.
    fn spec_ty(&self, jit: &TokenStream) -> TokenStream {
        match self.count {
            Some(n) => quote! { [#jit::InputSpec; #n] },
            None => quote! { #jit::InputSpec },
        }
    }

    /// Local holding leaf `i`'s `InputSpec` inside `prepare_with_config`.
    fn spec_local(&self, i: usize) -> Ident {
        match self.count {
            Some(_) => format_ident!("__jit_spec_{}_{}", self.name, i),
            None => self.name.clone(),
        }
    }
}

#[cfg(test)]
mod test;

struct VarDecl {
    name: Ident,
    min: Expr,
    max: Expr,
}

fn parse_bounds(input: ParseStream) -> Result<(Expr, Expr)> {
    let bounds;
    syn::parenthesized!(bounds in input);
    let min: Expr = bounds.parse()?;
    bounds.parse::<Comma>()?;
    let max: Expr = bounds.parse()?;
    if !bounds.is_empty() {
        return Err(Error::new(bounds.span(), "expected bounds as (min, max)"));
    }
    Ok((min, max))
}

/// The `: Tensor` / `: [Tensor; N]` tail of a slot declaration. The element
/// type is informational — the macro allocates from the `InputSpec`.
fn parse_slot_tail(input: ParseStream, name: Ident, unbatched: bool) -> Result<Slot> {
    let mut count = None;
    if input.peek(Token![:]) {
        input.parse::<Token![:]>()?;
        if input.peek(syn::token::Bracket) {
            let elem;
            bracketed!(elem in input);
            let _: Type = elem.parse()?;
            elem.parse::<Token![;]>()?;
            let len: LitInt = elem.parse()?;
            let n = len.base10_parse::<usize>()?;
            if n == 0 {
                return Err(Error::new(len.span(), "array slot length must be greater than zero"));
            }
            count = Some(n);
        } else {
            let _: Type = input.parse()?;
        }
    }
    Ok(Slot { name, count, unbatched })
}

fn parse_slot(input: ParseStream) -> Result<Slot> {
    let mut unbatched = false;
    for attr in input.call(Attribute::parse_outer)? {
        if attr.path().is_ident("unbatched") {
            unbatched = true;
        } else {
            return Err(Error::new_spanned(attr, "unknown attribute; expected `#[unbatched]`"));
        }
    }
    let name: Ident = input.parse()?;
    parse_slot_tail(input, name, unbatched)
}

fn parse_slots(block: ParseStream) -> Result<Vec<Slot>> {
    let mut slots = Vec::new();
    while !block.is_empty() {
        slots.push(parse_slot(block)?);
        if block.peek(Comma) {
            block.parse::<Comma>()?;
        }
    }
    Ok(slots)
}

impl Parse for JitWrapper {
    fn parse(input: ParseStream) -> Result<Self> {
        let name: Ident = input.parse()?;
        let generics: Generics = input.parse()?;
        let content;
        syn::parenthesized!(content in input);
        let model_ty: Type = content.parse()?;

        let body;
        braced!(body in input);

        let mut inputs = Vec::new();
        let mut state = Vec::new();
        let mut outputs = Vec::new();
        let mut vars = Vec::new();
        let mut batch_var = None;
        let mut build_args = Vec::new();
        let mut build_body = None;

        // Every clause keyword is recognized only in its own shape, so a bare
        // input named `state`/`inputs`/`outputs` keeps parsing as an input.
        while !body.is_empty() {
            let first: Ident = body.parse()?;

            if first == "build" && body.peek(Paren) {
                let args;
                syn::parenthesized!(args in body);
                build_args = args.parse_terminated(Ident::parse, Comma)?.into_iter().collect();

                let block;
                braced!(block in body);
                build_body = Some(block.parse()?);
                continue;
            } else if first == "inputs" && body.peek(Brace) {
                let block;
                braced!(block in body);
                inputs.extend(parse_slots(&block)?);
            } else if first == "state" && body.peek(Brace) {
                let block;
                braced!(block in body);
                state.extend(parse_slots(&block)?);
            } else if first == "outputs" && body.peek(Brace) {
                let block;
                braced!(block in body);
                outputs.extend(parse_slots(&block)?);
            } else if first == "vars" && body.peek(Brace) {
                let block;
                braced!(block in body);
                while !block.is_empty() {
                    let name: Ident = block.parse()?;
                    block.parse::<Token![:]>()?;
                    let (min, max) = parse_bounds(&block)?;
                    vars.push(VarDecl { name, min, max });
                    if block.peek(Comma) {
                        block.parse::<Comma>()?;
                    }
                }
            } else if first == "batch_var" && body.peek(Ident) {
                if batch_var.is_some() {
                    return Err(Error::new(first.span(), "duplicate `batch_var` declaration"));
                }
                let name: Ident = body.parse()?;
                body.parse::<Token![:]>()?;
                let (min, max) = parse_bounds(&body)?;
                batch_var = Some(vars.len());
                vars.push(VarDecl { name, min, max });
            } else {
                inputs.push(parse_slot_tail(&body, first, false)?);
            }

            // Every clause may be followed by a separating comma.
            if body.peek(Comma) {
                body.parse::<Comma>()?;
            }
        }

        let build_body = build_body.ok_or_else(|| Error::new(name.span(), "missing `build(...) { ... }` block"))?;

        Ok(JitWrapper { name, generics, model_ty, inputs, state, outputs, vars, batch_var, build_args, build_body })
    }
}

/// Fixed method names the generated impl always contains; a declared output
/// with one of these names would emit a duplicate definition.
const GENERATED_METHODS: &[&str] = &[
    "new",
    "prepare",
    "prepare_with_config",
    "replicate",
    "reset",
    "output",
    "buffers",
    "output_buffers",
    "input_buffer_ids",
    "prepared_kernels",
    "execute",
    "execute_bound",
    "execute_profiled",
    "execute_profiled_static",
    "execute_with_vars",
    "execute_with_vars_profiled",
];

/// A single tensor behind a slot: what the generated code actually tracks.
struct Leaf {
    /// Unique local/field-name stem.
    ident: Ident,
    /// Human-readable slot name for error messages.
    label: String,
    /// The `InputSpec` local this leaf is allocated from (plan inputs only).
    spec: Ident,
    is_state: bool,
    batched: bool,
}

fn flatten(slots: &[Slot], is_state: bool, batched_default: bool) -> Vec<Leaf> {
    slots
        .iter()
        .flat_map(|slot| {
            (0..slot.leaves()).map(move |i| Leaf {
                ident: slot.leaf(i),
                label: slot.label(i),
                spec: slot.spec_local(i),
                is_state,
                batched: batched_default && !is_state && !slot.unbatched,
            })
        })
        .collect()
}

pub(crate) fn generate(jit: JitWrapper) -> Result<TokenStream> {
    use std::collections::HashMap;

    let tensor = quote! { ::svod_tensor };
    let jit_path = quote! { ::svod_tensor::jit };
    let rt = quote! { ::svod_tensor::jit::rt };

    let name = &jit.name;
    let (impl_generics, ty_generics, where_clause) = jit.generics.split_for_impl();
    let model_ty = &jit.model_ty;
    let state_name = format_ident!("{}State", name);
    let name_str = name.to_string();

    // ---- validation ------------------------------------------------------
    let mut declared: HashMap<String, &'static str> = HashMap::new();
    for (slots, kind) in [(&jit.inputs, "input"), (&jit.state, "state slot"), (&jit.outputs, "output")] {
        for slot in slots {
            if let Some(prev) = declared.insert(slot.name.to_string(), kind) {
                return Err(Error::new(slot.name.span(), format!("name already declared as an {prev}")));
            }
        }
    }
    for var in &jit.vars {
        if let Some(prev) = declared.insert(var.name.to_string(), "variable") {
            return Err(Error::new(var.name.span(), format!("variable name already declared as an {prev}")));
        }
    }
    // Array slots expand to `name_0`, `name_1`, … locals and fields.
    for slot in jit.inputs.iter().chain(&jit.state).chain(&jit.outputs).filter(|s| s.count.is_some()) {
        for i in 0..slot.leaves() {
            if declared.contains_key(&slot.leaf(i).to_string()) {
                return Err(Error::new(
                    slot.name.span(),
                    format!("array slot expands to `{}`, which is already declared", slot.leaf(i)),
                ));
            }
        }
    }
    for out in &jit.outputs {
        if GENERATED_METHODS.contains(&out.name.to_string().as_str()) {
            return Err(Error::new(out.name.span(), "output name collides with a generated method"));
        }
    }
    for slot in jit.state.iter().chain(&jit.outputs) {
        if slot.unbatched {
            return Err(Error::new(slot.name.span(), "`#[unbatched]` applies to inputs only"));
        }
    }
    if jit.batch_var.is_none() && jit.inputs.iter().any(|s| s.unbatched) {
        return Err(Error::new(
            jit.inputs.iter().find(|s| s.unbatched).expect("checked").name.span(),
            "`#[unbatched]` requires a `batch_var` declaration",
        ));
    }
    if !jit.state.is_empty() && jit.outputs.is_empty() {
        return Err(Error::new(
            jit.state[0].name.span(),
            "a `state { .. }` block requires an `outputs { .. }` block: the build tuple is \
             (declared outputs.., state values..)",
        ));
    }
    for arg in &jit.build_args {
        let key = arg.to_string();
        if !matches!(declared.get(&key), Some(&"input") | Some(&"state slot") | Some(&"variable")) {
            return Err(Error::new(arg.span(), "build arg must match an input, a state slot or a declared variable"));
        }
    }

    // ---- flattened views -------------------------------------------------
    let batched = jit.batch_var.is_some();
    let input_leaves = flatten(&jit.inputs, false, batched);
    let state_leaves = flatten(&jit.state, true, batched);
    let out_leaves = flatten(&jit.outputs, false, false);
    // Inputs and state slots are both host-written plan inputs: allocated,
    // identity-checked, declared and replicated the same way.
    let plan_inputs: Vec<&Leaf> = input_leaves.iter().chain(&state_leaves).collect();
    let n_declared_outputs = out_leaves.len();
    let n_plan_outputs = n_declared_outputs + state_leaves.len();
    let multi_output = n_plan_outputs > 0;

    let idx_field = |leaf: &Leaf| format_ident!("{}_idx", leaf.ident);
    let buf_field = |leaf: &Leaf| format_ident!("{}_buffer_id", leaf.ident);
    let realized = |leaf: &Leaf| format_ident!("__jit_input_{}", leaf.ident);

    let idx_fields: Vec<Ident> = plan_inputs.iter().map(|l| idx_field(l)).collect();
    let buf_fields: Vec<Ident> = plan_inputs.iter().map(|l| buf_field(l)).collect();

    let var_names: Vec<&Ident> = jit.vars.iter().map(|v| &v.name).collect();
    let var_fields: Vec<Ident> = jit.vars.iter().map(|v| format_ident!("__var_{}", v.name)).collect();
    let n_vars = jit.vars.len();

    // ---- construction ----------------------------------------------------
    let var_inits = jit.vars.iter().zip(&var_fields).map(|(var, field)| {
        let (var_name, min, max) = (&var.name, &var.min, &var.max);
        quote! {
            let #field = #tensor::Variable::new(stringify!(#var_name), (#min) as i64, (#max) as i64);
        }
    });

    // Per variable: override the upper bound, the lower bound, or pin both
    // (making the variable a JIT-time constant). All three panic on an empty
    // `[min, max]` so misuse fails at construction instead of at execute time.
    // Must be chained before `prepare` — the plan captures the bounds when the
    // build closure runs.
    let with_var_bound_methods = jit.vars.iter().zip(&var_fields).flat_map(|(var, field)| {
        let var_name = &var.name;
        let (max_setter, min_setter, fixed_setter) = (
            format_ident!("with_{}_bound", var_name),
            format_ident!("with_{}_min_bound", var_name),
            format_ident!("with_{}_fixed", var_name),
        );
        let doc = |what: &str| format!("{what} for `{var_name}`. Must be called before `prepare`.");
        let (max_doc, min_doc, fixed_doc) = (
            doc("Override the upper bound"),
            doc("Override the lower bound"),
            doc("Pin both bounds to one value, making the variable a JIT-time constant"),
        );
        let var_str = var_name.to_string();
        [
            quote! {
                #[doc = #max_doc]
                pub fn #max_setter(mut self, max: usize) -> Self {
                    let (min, _) = self.#field.bounds();
                    let max = max as i64;
                    assert!(max >= min, "{}: with_{}_bound({max}) creates empty range (min={min})", #var_str, #var_str);
                    self.#field = #tensor::Variable::new(stringify!(#var_name), min, max);
                    self
                }
            },
            quote! {
                #[doc = #min_doc]
                pub fn #min_setter(mut self, min: usize) -> Self {
                    let (_, max) = self.#field.bounds();
                    let min = min as i64;
                    assert!(min <= max, "{}: with_{}_min_bound({min}) exceeds max={max}", #var_str, #var_str);
                    self.#field = #tensor::Variable::new(stringify!(#var_name), min, max);
                    self
                }
            },
            quote! {
                #[doc = #fixed_doc]
                pub fn #fixed_setter(mut self, value: usize) -> Self {
                    assert!(value > 0, "{}: with_{}_fixed(0) is not allowed", #var_str, #var_str);
                    let v = value as i64;
                    self.#field = #tensor::Variable::new(stringify!(#var_name), v, v);
                    self
                }
            },
        ]
    });

    // ---- prepare body ----------------------------------------------------
    let prepare_params: Vec<TokenStream> = jit
        .inputs
        .iter()
        .chain(&jit.state)
        .map(|slot| {
            let (slot_name, ty) = (&slot.name, slot.spec_ty(&jit_path));
            quote! { #slot_name: #ty }
        })
        .collect();
    let prepare_args: Vec<&Ident> = jit.inputs.iter().chain(&jit.state).map(|s| &s.name).collect();

    // Array slots arrive as one `[InputSpec; N]`; split it into per-leaf specs.
    let spec_destructure = jit.inputs.iter().chain(&jit.state).filter(|s| s.count.is_some()).map(|slot| {
        let slot_name = &slot.name;
        let locals: Vec<Ident> = (0..slot.leaves()).map(|i| slot.spec_local(i)).collect();
        quote! { let [#(#locals),*] = #slot_name; }
    });

    // Placeholder inputs are allocated directly (`from_bytes_shaped_spec`
    // mints a unique-slot BUFFER UOp) rather than realized from
    // `Tensor::zeros`: a zeros graph is pure and hash-consed, so every
    // same-shape placeholder in the process would share ONE UOp identity —
    // and a concurrent prepare's `apply_map_to_tensors` could then rebind
    // this JIT's input to a foreign buffer. Per-tensor identity closes that
    // race at the source. Trade-off (accepted): allocation resolves through
    // the global device registry and panics on allocator failure, bypassing
    // `PrepareConfig`'s resolver.
    let input_realizations = plan_inputs.iter().map(|leaf| {
        let (local, spec) = (realized(leaf), &leaf.spec);
        // State lives on the device across executes; the host reaches it only
        // through `reset`/`copyin`, never through a mapping.
        let cpu_access = if leaf.is_state {
            quote! { false }
        } else {
            quote! { !#spec.device_local }
        };
        quote! {
            let #local = {
                let numel: usize = #spec.shape.iter().product();
                #tensor::Tensor::from_bytes_shaped_spec(
                    &vec![0u8; numel * #spec.dtype.bytes()],
                    &#spec.shape,
                    #spec.dtype.clone(),
                    #tensor::default_device::default_device(),
                    #rt::BufferSpec { cpu_access: #cpu_access, ..Default::default() },
                )
            };
        }
    });

    let ast_id_local = |leaf: &Leaf| format_ident!("{}_ast_id", leaf.ident);
    let buffer_id_extractions = plan_inputs.iter().map(|leaf| {
        let (local, buf, ast) = (realized(leaf), buf_field(leaf), ast_id_local(leaf));
        quote! {
            let #buf = #local.buffer().ok_or(#jit_path::JitError::NotPrepared)?.id();
            let #ast = #local.uop().id;
        }
    });

    // Inputs must not share a buffer: the plan would collapse them into one
    // slot. State slots are inputs here too — the check never compares against
    // an OUTPUT, which for state deliberately IS the input's storage.
    let duplicate_input_checks = plan_inputs.iter().enumerate().flat_map(|(i, left)| {
        let (left_label, left_buf, jit_path) = (left.label.clone(), buf_field(left), jit_path.clone());
        plan_inputs.iter().skip(i + 1).map(move |right| {
            let (right_label, right_buf) = (&right.label, buf_field(right));
            let (left_label, left_buf) = (left_label.clone(), left_buf.clone());
            quote! {
                if #left_buf == #right_buf {
                    return Err(#jit_path::JitError::DuplicateInputBuffer {
                        name: #right_label,
                        duplicate_of: #left_label,
                        buffer_id: #right_buf,
                    });
                }
            }
        })
    });

    let prepare_var_bindings = jit.vars.iter().zip(&var_fields).map(|(var, field)| {
        let var_name = &var.name;
        quote! { let #var_name = self.#field.bind(self.#field.bounds().1)?; }
    });

    // `batch_var b` shrinks every batched input's dim 0 to `b` after
    // realization: the buffers stay allocated at the maximum, the graph sees
    // the live extent. State is never shrunk — it is carried whole.
    let batch_ident = jit.batch_var.map(|i| var_names[i]);
    let arg_local = |ident: &Ident| format_ident!("__jit_arg_{ident}");
    let input_arg_bindings = input_leaves.iter().chain(&state_leaves).map(|leaf| {
        let (local, arg) = (realized(leaf), arg_local(&leaf.ident));
        match (leaf.batched, batch_ident) {
            (true, Some(b)) => {
                let shrunk = format_ident!("__jit_shrunk_{}", leaf.ident);
                quote! {
                    let #shrunk = #jit_path::shrink_batch(&#local, &#b)?;
                    let #arg: &#tensor::Tensor = &#shrunk;
                }
            }
            _ => quote! { let #arg: &#tensor::Tensor = &#local; },
        }
    });
    // Bind each slot to the name the build closure uses: `&Tensor` for a
    // scalar slot, `[&Tensor; N]` for an array slot.
    let build_slot_bindings = jit.inputs.iter().chain(&jit.state).map(|slot| {
        let slot_name = &slot.name;
        let args: Vec<Ident> = (0..slot.leaves()).map(|i| arg_local(&slot.leaf(i))).collect();
        match slot.count {
            Some(n) => quote! { let #slot_name: [&#tensor::Tensor; #n] = [#(#args),*]; },
            None => {
                let arg = &args[0];
                quote! { let #slot_name = #arg; }
            }
        }
    });

    // The build closure runs once, at capture, on the caller's thread, so it
    // inherits the caller's origin scope; the wrapper's name roots every kernel
    // the plan ends up owning.
    let build_args = &jit.build_args;
    let build_body = &jit.build_body;
    // Variables are cloned into the closure: the wrapper still needs them
    // afterwards for the captured output shapes and the recorded bindings.
    let build_arg_sources: Vec<TokenStream> = build_args
        .iter()
        .map(|arg| {
            if var_names.contains(&arg) {
                quote! { #arg.clone() }
            } else {
                quote! { #arg }
            }
        })
        .collect();
    let build_closure = quote! {
        (|| {
            let __jit_origin = #rt::OriginScope::label(#name_str);
            let model: &#model_ty = &self.model;
            let (#(#build_args),*) = (#(#build_arg_sources),*);
            #build_body
        })()
    };

    let out_leaf_idents: Vec<&Ident> = out_leaves.iter().map(|l| &l.ident).collect();
    // Flattened once: a nested repetition inside the capture list would make
    // `quote!` bind `var_names` at the wrong level.
    let var_refs = quote! { &[#(&#var_names),*] };
    let state_value = |slot: &Slot| format_ident!("__jit_state_value_{}", slot.name);
    let state_new = |ident: &Ident| format_ident!("__jit_state_new_{ident}");
    let state_out = |ident: &Ident| format_ident!("__jit_state_out_{ident}");

    // Destructure the build tuple: declared outputs first, then one value per
    // state slot. Array slots arrive as one `[Tensor; N]` and are split here.
    let build_and_compile = if multi_output {
        let out_slot_names: Vec<&Ident> = jit.outputs.iter().map(|s| &s.name).collect();
        let state_values: Vec<Ident> = jit.state.iter().map(state_value).collect();
        let out_destructure = jit.outputs.iter().filter(|s| s.count.is_some()).map(|slot| {
            let slot_name = &slot.name;
            let leaves: Vec<Ident> = (0..slot.leaves()).map(|i| slot.leaf(i)).collect();
            quote! { let [#(#leaves),*] = #slot_name; }
        });
        let state_destructure = jit.state.iter().map(|slot| {
            let value = state_value(slot);
            let news: Vec<Ident> = (0..slot.leaves()).map(|i| state_new(&slot.leaf(i))).collect();
            match slot.count {
                Some(_) => quote! { let [#(#news),*] = #value; },
                None => {
                    let new = &news[0];
                    quote! { let #new = #value; }
                }
            }
        });
        // A state output is an in-place write into the input's own storage:
        // `AFTER(in_buf, STORE(in_buf, value))`. The plan stores the new value
        // where the next execute reads it, so state recycles on-device and the
        // host never copies output→input between executes.
        let state_assign_backs = state_leaves.iter().map(|leaf| {
            let (local, new, out) = (realized(leaf), state_new(&leaf.ident), state_out(&leaf.ident));
            quote! {
                let #out = {
                    let __jit_out = #tensor::Tensor::from_lazy(#local.uop());
                    __jit_out.try_assign(&#new)?;
                    __jit_out
                };
            }
        });
        let state_outs: Vec<Ident> = state_leaves.iter().map(|l| state_out(&l.ident)).collect();
        // The build tuple has one element per declared output slot and one per
        // state slot — and no tuple at all when there is exactly one of them.
        let bindings = if jit.outputs.len() + jit.state.len() == 1 {
            let single = match out_slot_names.first() {
                Some(out) => *out,
                None => &state_values[0],
            };
            quote! { let #single }
        } else {
            quote! { let (#(#out_slot_names,)* #(#state_values,)*) }
        };
        quote! {
            #bindings = #build_closure
                .map_err(|e| #jit_path::JitError::Build { source: Box::new(e) as _ })?;
            #(#out_destructure)*
            #(#state_destructure)*
            #(#state_assign_backs)*
            // Live output shapes, captured before `prepare_batch_with` rewrites
            // the tensors onto their plan buffers.
            let __jit_out_shapes: Vec<#jit_path::OutputShape> = vec![
                #(#jit_path::OutputShape::capture(&#out_leaf_idents, #var_refs)?,)*
            ];
            let __jit_outputs: [#tensor::Tensor; #n_plan_outputs] = [#(#out_leaf_idents,)* #(#state_outs,)*];
            let plan = #tensor::Tensor::prepare_batch_with(__jit_outputs.iter(), config)?;
            if plan.num_outputs() != #n_plan_outputs {
                return Err(#jit_path::JitError::OutputCountMismatch {
                    declared: #n_plan_outputs,
                    actual: plan.num_outputs(),
                });
            }
        }
    } else {
        quote! {
            let output: #tensor::Tensor = #build_closure
                .map_err(|e| #jit_path::JitError::Build { source: Box::new(e) as _ })?;
            let __jit_out_shapes: Vec<#jit_path::OutputShape> = Vec::new();
            let plan = #tensor::Tensor::prepare_batch_with(std::iter::once(&output), config)?;
        }
    };

    // Eagerly resolve each input's plan buffer index at prepare time. A
    // missing input fails loud here (instead of at first accessor use), and
    // the stored plain `usize` survives replication, where buffer handle ids
    // are re-minted but indices are preserved.
    let index_resolution = plan_inputs.iter().map(|leaf| {
        let (idx, buf, ast, label) = (idx_field(leaf), buf_field(leaf), ast_id_local(leaf), &leaf.label);
        quote! {
            let #idx = plan
                .ast_to_buffer_map()
                .get(&#ast)
                .copied()
                .or_else(|| plan.buffers().iter().position(|b| b.id() == #buf))
                .ok_or(#jit_path::JitError::InputBufferNotFound { name: #label })?;
        }
    });

    // Post-resolution identity checks: indices must be pairwise distinct and
    // must resolve back to the realized input buffers. This guards the uop
    // channel (`ast_to_buffer`) — the one cross-plan aliasing corrupts —
    // where the pre-plan `DuplicateInputBuffer` check only sees the local
    // buffer handles.
    let index_conflict_checks = plan_inputs.iter().enumerate().flat_map(|(i, left)| {
        let (left_label, left_idx, jit_path) = (left.label.clone(), idx_field(left), jit_path.clone());
        plan_inputs.iter().skip(i + 1).map(move |right| {
            let (right_label, right_idx, right_buf) = (&right.label, idx_field(right), buf_field(right));
            let (left_label, left_idx) = (left_label.clone(), left_idx.clone());
            quote! {
                if #left_idx == #right_idx {
                    return Err(#jit_path::JitError::DuplicateInputBuffer {
                        name: #right_label,
                        duplicate_of: #left_label,
                        buffer_id: #right_buf,
                    });
                }
            }
        })
    });
    let identity_checks = plan_inputs.iter().map(|leaf| {
        let (idx, buf, label) = (idx_field(leaf), buf_field(leaf), &leaf.label);
        quote! {
            {
                let resolved = plan.buffers()[#idx].id();
                if resolved != #buf {
                    return Err(#jit_path::JitError::InputAliased {
                        name: #label,
                        expected: #buf,
                        actual: resolved,
                    });
                }
            }
        }
    });

    // Declare each input on the plan: the plan's write analysis cannot see
    // host writes (`copyin`, the `copy_output_to_*` state recycling), and
    // `replicate` snapshot-forks declared inputs.
    let input_declarations = if plan_inputs.is_empty() {
        quote! {}
    } else {
        let declares = idx_fields.iter().map(|idx| {
            quote! { plan.declare_input(#idx).map_err(|e| #jit_path::JitError::Runtime { source: e })?; }
        });
        quote! {
            let mut plan = plan;
            #(#declares)*
        }
    };

    // ---- state-struct API ------------------------------------------------
    let state_init = quote! {
        #state_name {
            plan,
            #(#idx_fields,)*
            #(#buf_fields,)*
            __jit_out_shapes,
            __jit_var_values: vec![#(#var_names.value(),)*],
        }
    };

    // One `..._mut` per input/state slot (`(i)`-indexed for array slots) plus
    // a typed write view for host-visible inputs.
    let slot_index_arm = |slot: &Slot, leaf_ident: &dyn Fn(usize) -> Ident| -> TokenStream {
        match slot.count {
            Some(_) => {
                let arms = (0..slot.leaves()).map(|i| {
                    let field = leaf_ident(i);
                    quote! { #i => self.#field, }
                });
                let label = slot.name.to_string();
                quote! {
                    let __jit_idx = match index {
                        #(#arms)*
                        _ => return Err(#jit_path::JitError::InputBufferNotFound { name: #label }),
                    };
                }
            }
            None => {
                let field = leaf_ident(0);
                quote! { let __jit_idx = self.#field; }
            }
        }
    };

    let host_slots = || jit.inputs.iter().map(|s| (s, false)).chain(jit.state.iter().map(|s| (s, true)));
    let input_accessor_impls = host_slots().map(|(slot, is_state)| {
        let (slot_name, label) = (&slot.name, slot.name.to_string());
        let accessor = format_ident!("{}_mut", slot_name);
        let view_mut = format_ident!("{}_view_mut", slot_name);
        let params = if slot.count.is_some() {
            quote! { , index: usize }
        } else {
            quote! {}
        };
        let resolve = slot_index_arm(slot, &|i| format_ident!("{}_idx", slot.leaf(i)));
        // State buffers have no host mapping, so no typed write view.
        let view = (!is_state).then(|| {
            quote! {
                fn #view_mut<T: #rt::HasDType>(&mut self #params) -> #jit_path::Result<#rt::ArrayViewMutD<'_, T>> {
                    #resolve
                    let buffer = self.plan
                        .buffer_at_mut(__jit_idx)
                        .ok_or(#jit_path::JitError::InputBufferNotFound { name: #label })?;
                    #jit_path::view_mut::<T>(buffer)
                }
            }
        });
        quote! {
            fn #accessor(&mut self #params) -> #jit_path::Result<&mut #rt::Buffer> {
                #resolve
                self.plan.buffer_at_mut(__jit_idx).ok_or(#jit_path::JitError::InputBufferNotFound { name: #label })
            }
            #view
        }
    });

    // Per-input on-device copy helpers: copy a region of declared output
    // `out_pos` into the input's buffer with NO host round-trip (the plan owns
    // both buffers; the split borrow lives in the runtime).
    let copy_helper_impls = jit.inputs.iter().map(|slot| {
        let helper = format_ident!("copy_output_to_{}", slot.name);
        let params = if slot.count.is_some() {
            quote! { index: usize, }
        } else {
            quote! {}
        };
        let resolve = slot_index_arm(slot, &|i| format_ident!("{}_idx", slot.leaf(i)));
        quote! {
            fn #helper(
                &mut self,
                #params
                out_pos: usize,
                dst_off: usize,
                src_off: usize,
                len: usize,
            ) -> #jit_path::Result<()> {
                #resolve
                self.plan
                    .copy_output_region_to_buffer(out_pos, __jit_idx, dst_off, src_off, len)
                    .map_err(|e| #jit_path::JitError::Runtime { source: e })
            }
        }
    });

    // One accessor group per declared output, backed by positional
    // `output_buffer_at(i)` (i = declared order = `prepare_batch_with` order).
    let mut out_pos = 0usize;
    let output_impls: Vec<TokenStream> = jit
        .outputs
        .iter()
        .map(|slot| {
            let base = out_pos;
            out_pos += slot.leaves();
            let slot_name = &slot.name;
            let shape_fn = format_ident!("{}_shape", slot_name);
            let view_fn = format_ident!("{}_view", slot_name);
            let to_vec_fn = format_ident!("{}_to_vec", slot_name);
            let (params, pos) = match slot.count {
                Some(n) => (
                    quote! { index: usize },
                    quote! { { assert!(index < #n, "output index out of range"); #base + index } },
                ),
                None => (quote! {}, quote! { #base }),
            };
            let arg = if slot.count.is_some() {
                quote! { index }
            } else {
                quote! {}
            };
            quote! {
                fn #slot_name(&self, #params) -> #jit_path::Result<&#rt::Buffer> {
                    self.plan.output_buffer_at(#pos).ok_or(#jit_path::JitError::NotPrepared)
                }

                fn #shape_fn(&self, #params) -> #jit_path::Result<Vec<usize>> {
                    Ok(self.__jit_out_shapes[#pos].resolve(&self.__jit_var_values))
                }

                fn #view_fn<T: #rt::HasDType>(&self, #params) -> #jit_path::Result<#rt::ArrayViewD<'_, T>> {
                    let shape = self.#shape_fn(#arg)?;
                    #jit_path::view::<T>(self.#slot_name(#arg)?, &shape)
                }

                fn #to_vec_fn<T: #rt::HasDType + Default + Clone>(
                    &self,
                    #params
                ) -> #jit_path::Result<Vec<T>> {
                    let numel = self.__jit_out_shapes[#pos].numel(&self.__jit_var_values);
                    #jit_path::to_vec::<T>(self.#slot_name(#arg)?, numel)
                }
            }
        })
        .collect();

    let reset_impl = (!state_leaves.is_empty()).then(|| {
        let zeros = state_leaves.iter().map(|leaf| {
            let (idx, label) = (idx_field(leaf), &leaf.label);
            quote! {
                let buffer = self.plan
                    .buffer_at_mut(self.#idx)
                    .ok_or(#jit_path::JitError::InputBufferNotFound { name: #label })?;
                #jit_path::zero_fill(buffer)?;
            }
        });
        quote! {
            fn reset(&mut self) -> #jit_path::Result<()> {
                #(#zeros)*
                Ok(())
            }
        }
    });

    let execute_bound_impl = (n_vars > 0).then(|| {
        quote! {
            fn execute_bound(&mut self, #(#var_names: i64,)*) -> #jit_path::Result<()> {
                self.execute_with_vars(&[#((stringify!(#var_names), #var_names),)*])
            }
        }
    });

    // ---- wrapper forwarding ----------------------------------------------
    // Every method above lives on the state struct and is forwarded verbatim
    // through the wrapper's `Option<state>`.
    let forward = |sig: TokenStream, call: TokenStream, mutable: bool| {
        let borrow = if mutable {
            quote! { self.state.as_mut().ok_or(#jit_path::JitError::NotPrepared)? }
        } else {
            quote! { self.state.as_ref().ok_or(#jit_path::JitError::NotPrepared)? }
        };
        quote! { pub #sig { #borrow.#call } }
    };

    let mut wrapper_api: Vec<TokenStream> = Vec::new();
    for (slot, is_state) in host_slots() {
        let slot_name = &slot.name;
        let accessor = format_ident!("{}_mut", slot_name);
        let view_mut = format_ident!("{}_view_mut", slot_name);
        let (params, arg) =
            if slot.count.is_some() { (quote! { , index: usize }, quote! { index }) } else { (quote! {}, quote! {}) };
        wrapper_api.push(forward(
            quote! { fn #accessor(&mut self #params) -> #jit_path::Result<&mut #rt::Buffer> },
            quote! { #accessor(#arg) },
            true,
        ));
        if !is_state {
            wrapper_api.push(forward(
                quote! {
                    fn #view_mut<T: #rt::HasDType>(&mut self #params) -> #jit_path::Result<#rt::ArrayViewMutD<'_, T>>
                },
                quote! { #view_mut::<T>(#arg) },
                true,
            ));
        }
    }
    for slot in &jit.inputs {
        let helper = format_ident!("copy_output_to_{}", slot.name);
        let (params, arg) =
            if slot.count.is_some() { (quote! { index: usize, }, quote! { index, }) } else { (quote! {}, quote! {}) };
        wrapper_api.push(forward(
            quote! {
                fn #helper(
                    &mut self,
                    #params
                    out_pos: usize,
                    dst_off: usize,
                    src_off: usize,
                    len: usize,
                ) -> #jit_path::Result<()>
            },
            quote! { #helper(#arg out_pos, dst_off, src_off, len) },
            true,
        ));
    }
    for slot in &jit.outputs {
        let slot_name = &slot.name;
        let shape_fn = format_ident!("{}_shape", slot_name);
        let view_fn = format_ident!("{}_view", slot_name);
        let to_vec_fn = format_ident!("{}_to_vec", slot_name);
        let (params, arg) =
            if slot.count.is_some() { (quote! { index: usize }, quote! { index }) } else { (quote! {}, quote! {}) };
        wrapper_api.push(forward(
            quote! { fn #slot_name(&self, #params) -> #jit_path::Result<&#rt::Buffer> },
            quote! { #slot_name(#arg) },
            false,
        ));
        wrapper_api.push(forward(
            quote! { fn #shape_fn(&self, #params) -> #jit_path::Result<Vec<usize>> },
            quote! { #shape_fn(#arg) },
            false,
        ));
        wrapper_api.push(forward(
            quote! { fn #view_fn<T: #rt::HasDType>(&self, #params) -> #jit_path::Result<#rt::ArrayViewD<'_, T>> },
            quote! { #view_fn::<T>(#arg) },
            false,
        ));
        wrapper_api.push(forward(
            quote! {
                fn #to_vec_fn<T: #rt::HasDType + Default + Clone>(&self, #params) -> #jit_path::Result<Vec<T>>
            },
            quote! { #to_vec_fn::<T>(#arg) },
            false,
        ));
    }
    if reset_impl.is_some() {
        wrapper_api.push(forward(quote! { fn reset(&mut self) -> #jit_path::Result<()> }, quote! { reset() }, true));
    }
    if execute_bound_impl.is_some() {
        wrapper_api.push(forward(
            quote! { fn execute_bound(&mut self, #(#var_names: i64,)*) -> #jit_path::Result<()> },
            quote! { execute_bound(#(#var_names,)*) },
            true,
        ));
    }
    for (sig, call, mutable) in [
        (quote! { fn output(&self) -> #jit_path::Result<&#rt::Buffer> }, quote! { output() }, false),
        (quote! { fn buffers(&self) -> #jit_path::Result<&[#rt::Buffer]> }, quote! { buffers() }, false),
        (
            quote! { fn output_buffers(&self) -> #jit_path::Result<Vec<&#rt::Buffer>> },
            quote! { output_buffers() },
            false,
        ),
        (
            quote! { fn input_buffer_ids(&self) -> #jit_path::Result<Vec<#rt::BufferId>> },
            quote! { input_buffer_ids() },
            false,
        ),
        (
            quote! { fn prepared_kernels(&self) -> #jit_path::Result<Vec<&#rt::PreparedKernel>> },
            quote! { prepared_kernels() },
            false,
        ),
        (quote! { fn execute(&mut self) -> #jit_path::Result<()> }, quote! { execute() }, true),
        (
            quote! { fn execute_profiled(&mut self) -> #jit_path::Result<Vec<#rt::KernelProfile>> },
            quote! { execute_profiled() },
            true,
        ),
        (
            quote! { fn execute_profiled_static(&mut self) -> #jit_path::Result<Vec<#rt::KernelProfile>> },
            quote! { execute_profiled_static() },
            true,
        ),
        (
            quote! { fn execute_with_vars(&mut self, vars: &[(&str, i64)]) -> #jit_path::Result<()> },
            quote! { execute_with_vars(vars) },
            true,
        ),
        (
            quote! {
                fn execute_with_vars_profiled(
                    &mut self,
                    vars: &[(&str, i64)],
                ) -> #jit_path::Result<Vec<#rt::KernelProfile>>
            },
            quote! { execute_with_vars_profiled(vars) },
            true,
        ),
    ] {
        wrapper_api.push(forward(sig, call, mutable));
    }

    // `replicate` on the state: the plan itself knows every declared input, so
    // replication is a single argument-less call; only the replica's buffer
    // ids need re-resolving at the preserved indices.
    let replicate_copy_indices = idx_fields.iter().map(|idx| quote! { #idx: self.#idx, });
    let replicate_rebinds = idx_fields.iter().zip(&buf_fields).map(|(idx, buf)| {
        quote! { #buf: plan.buffers()[self.#idx].id(), }
    });

    let expanded = quote! {
        pub struct #name #impl_generics #where_clause {
            /// `Arc` so `replicate` can return `Self` without an `M: Clone`
            /// bound; the model only feeds `&`-access graph building. Note
            /// the wrapper is therefore `Send`/`Sync` iff `M: Send + Sync`.
            model: std::sync::Arc<#model_ty>,
            state: Option<#state_name>,
            #(#var_fields: #tensor::Variable,)*
        }

        struct #state_name {
            plan: #rt::ExecutionPlan,
            #(#idx_fields: usize,)*
            #(#buf_fields: #rt::BufferId,)*
            /// Per declared output, the shape captured at prepare with its
            /// symbolic dims resolved against `__jit_var_values`.
            __jit_out_shapes: Vec<#jit_path::OutputShape>,
            /// Last bound value per declared variable, positionally.
            __jit_var_values: Vec<i64>,
        }

        impl #state_name {
            #(#input_accessor_impls)*
            #(#copy_helper_impls)*
            #(#output_impls)*
            #reset_impl
            #execute_bound_impl

            /// Track the bindings the live output shapes are resolved against.
            fn record_var_values(&mut self, vars: &[(&str, i64)]) {
                const NAMES: [&str; #n_vars] = [#(stringify!(#var_names),)*];
                for &(name, value) in vars {
                    if let Some(i) = NAMES.iter().position(|n| *n == name) {
                        self.__jit_var_values[i] = value;
                    }
                }
            }

            fn output(&self) -> #jit_path::Result<&#rt::Buffer> {
                self.plan.output_buffer().ok_or(#jit_path::JitError::NotPrepared)
            }

            fn buffers(&self) -> #jit_path::Result<&[#rt::Buffer]> {
                Ok(self.plan.buffers())
            }

            fn output_buffers(&self) -> #jit_path::Result<Vec<&#rt::Buffer>> {
                Ok(self.plan.output_buffers())
            }

            fn input_buffer_ids(&self) -> #jit_path::Result<Vec<#rt::BufferId>> {
                Ok(vec![#(self.#buf_fields),*])
            }

            fn prepared_kernels(&self) -> #jit_path::Result<Vec<&#rt::PreparedKernel>> {
                Ok(self.plan.prepared_kernels())
            }

            fn execute(&mut self) -> #jit_path::Result<()> {
                self.plan.execute().map_err(|e| #jit_path::JitError::Runtime { source: e })
            }

            fn execute_profiled(&mut self) -> #jit_path::Result<Vec<#rt::KernelProfile>> {
                self.plan.execute_profiled().map_err(|e| #jit_path::JitError::Runtime { source: e })
            }

            fn execute_profiled_static(&mut self) -> #jit_path::Result<Vec<#rt::KernelProfile>> {
                self.plan.profile(&#rt::ProfileOptions::default())
                    .map(|mut profile| profile.stages.pop().map_or_else(Vec::new, |stage| stage.kernels))
                    .map_err(|e| #jit_path::JitError::Runtime { source: e })
            }

            // Values are recorded only once the plan accepted them: a binding
            // the plan rejects (out of its declared bounds) must not move the
            // shapes the outputs still report for the last successful run.
            fn execute_with_vars(&mut self, vars: &[(&str, i64)]) -> #jit_path::Result<()> {
                self.plan.execute_with_vars(vars).map_err(|e| #jit_path::JitError::Runtime { source: e })?;
                self.record_var_values(vars);
                Ok(())
            }

            fn execute_with_vars_profiled(
                &mut self,
                vars: &[(&str, i64)],
            ) -> #jit_path::Result<Vec<#rt::KernelProfile>> {
                let profiles = self
                    .plan
                    .execute_with_vars_profiled(vars)
                    .map_err(|e| #jit_path::JitError::Runtime { source: e })?;
                self.record_var_values(vars);
                Ok(profiles)
            }

            fn replicate(&self) -> #jit_path::Result<Self> {
                let plan = self.plan.replicate().map_err(|e| #jit_path::JitError::Runtime { source: e })?;
                Ok(Self {
                    #(#replicate_copy_indices)*
                    #(#replicate_rebinds)*
                    __jit_out_shapes: self.__jit_out_shapes.clone(),
                    __jit_var_values: self.__jit_var_values.clone(),
                    plan,
                })
            }
        }

        impl #impl_generics #name #ty_generics #where_clause {
            pub fn new(model: #model_ty) -> Self {
                #(#var_inits)*
                Self {
                    model: std::sync::Arc::new(model),
                    state: None,
                    #(#var_fields,)*
                }
            }

            #(#with_var_bound_methods)*

            // A wrapper with many slots legitimately has a wide `prepare`.
            #[allow(clippy::too_many_arguments)]
            pub fn prepare(&mut self, #(#prepare_params),*) -> #jit_path::Result<()> {
                let config = #tensor::PrepareConfig::from_env();
                self.prepare_with_config(#(#prepare_args,)* &config)
            }

            #[allow(clippy::too_many_arguments)]
            pub fn prepare_with_config(
                &mut self,
                #(#prepare_params,)*
                config: &#tensor::PrepareConfig,
            ) -> #jit_path::Result<()> {
                #(#spec_destructure)*
                #(#input_realizations)*
                #(#buffer_id_extractions)*
                #(#duplicate_input_checks)*

                #(#prepare_var_bindings)*
                #(#input_arg_bindings)*
                #(#build_slot_bindings)*

                #build_and_compile

                #(#index_resolution)*
                #(#index_conflict_checks)*
                #(#identity_checks)*

                #input_declarations

                self.state = Some(#state_init);
                Ok(())
            }

            /// Deep-copy the prepared JIT for concurrent execution: forks
            /// every written storage (bare) and the declared input buffers
            /// (with a snapshot of their current contents), shares the model,
            /// weights and compiled kernels, and gives the replica its own
            /// backend queue. Replicate only while the JIT is idle.
            pub fn replicate(&self) -> #jit_path::Result<Self> {
                let state = self.state.as_ref().ok_or(#jit_path::JitError::NotPrepared)?;
                Ok(Self {
                    model: std::sync::Arc::clone(&self.model),
                    state: Some(state.replicate()?),
                    #(#var_fields: self.#var_fields.clone(),)*
                })
            }

            #(#wrapper_api)*
        }
    };

    Ok(expanded)
}
