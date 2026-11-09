use proc_macro2::TokenStream;
use quote::quote;

use super::{JitWrapper, generate};

/// The expansion with every space removed, so assertions can be written as
/// compact needles instead of `quote!`'s token spacing.
fn expand(input: TokenStream) -> String {
    let jit: JitWrapper = syn::parse2(input).expect("parse failed");
    generate(jit).expect("expansion failed").to_string().replace([' ', '\n'], "")
}

/// The error message of an invocation that must not compile — from the parser
/// or from `generate`'s validation.
fn error(input: TokenStream) -> String {
    match syn::parse2::<JitWrapper>(input) {
        Err(e) => e.to_string(),
        Ok(jit) => generate(jit).expect_err("expected a compile error").to_string(),
    }
}

// ---------------------------------------------------------------------------
// Grammar: the pre-v2 forms must expand exactly as before.
// ---------------------------------------------------------------------------

#[test]
fn bare_input_lines_and_single_output() {
    let out = expand(quote! {
        AddJit(AddModel) {
            x: Tensor,
            y: Tensor,

            build(x, y) {
                model.forward(x, y)
            }
        }
    });
    assert!(out.contains("pubstructAddJit"));
    assert!(out.contains("pubfnprepare(&mutself,x:::svod_tensor::jit::InputSpec,y:::svod_tensor::jit::InputSpec)"));
    assert!(out.contains("pubfnx_mut(&mutself)->::svod_tensor::jit::Result<&mut::svod_tensor::jit::rt::Buffer>"));
    assert!(out.contains("pubfncopy_output_to_x(&mutself,out_pos:usize"));
    // Single-output form: no tuple destructuring, no output-count check.
    assert!(out.contains("letoutput:::svod_tensor::Tensor="));
    assert!(!out.contains("OutputCountMismatch"));
    // No `svod_model` / `svod_device` / `svod_runtime` paths: an invoking crate
    // needs only `svod-tensor`.
    for foreign in ["svod_model", "svod_device::", "svod_runtime::", "svod_dtype::"] {
        assert!(!out.contains(foreign), "expansion still mentions {foreign}");
    }
}

#[test]
fn outputs_block_and_vars_are_unchanged() {
    let out = expand(quote! {
        SplitJit(SplitModel) {
            x: Tensor,

            vars { b: (1, 8) }

            outputs { sum, diff },

            build(x, b) {
                model.forward(x, &b)
            }
        }
    });
    assert!(out.contains("let(sum,diff,)="));
    assert!(out.contains("plan.num_outputs()!=2usize"));
    assert!(out.contains("pubfnwith_b_bound(mutself,max:usize)"));
    assert!(out.contains("pubfnwith_b_min_bound(mutself,min:usize)"));
    assert!(out.contains("pubfnwith_b_fixed(mutself,value:usize)"));
    // A declared variable is cloned into the build closure so the wrapper can
    // still resolve output shapes against it afterwards.
    assert!(out.contains("let(x,b)=(x,b.clone());"));
}

// ---------------------------------------------------------------------------
// v2 grammar.
// ---------------------------------------------------------------------------

#[test]
fn inputs_block_parses_like_bare_lines() {
    let bare = expand(quote! {
        J(M) {
            x: Tensor,
            y: Tensor,
            build(x, y) { model.forward(x, y) }
        }
    });
    let block = expand(quote! {
        J(M) {
            inputs { x: Tensor, y: Tensor }
            build(x, y) { model.forward(x, y) }
        }
    });
    assert_eq!(bare, block);
}

#[test]
fn batch_var_declares_a_variable_and_shrinks_batched_inputs() {
    let out = expand(quote! {
        ScaleJit(Scale) {
            inputs {
                x: Tensor,
                #[unbatched] bias: Tensor,
            }
            batch_var b: (1, model.config.max_batch_size),
            outputs { y }

            build(x, bias) { model.forward(x, bias) }
        }
    });
    // Declared like any other variable …
    assert!(out.contains("pubfnwith_b_bound(mutself,max:usize)"));
    assert!(out.contains(r#"::svod_tensor::Variable::new(stringify!(b),(1)asi64,(model.config.max_batch_size)asi64)"#));
    // … and shrinks every batched input on dim 0 after realization.
    assert!(out.contains("let__jit_shrunk_x=::svod_tensor::jit::shrink_batch(&__jit_input_x,&b)?;"));
    assert!(!out.contains("__jit_shrunk_bias"));
    assert!(out.contains("let__jit_arg_bias:&::svod_tensor::Tensor=&__jit_input_bias;"));
    // Typed positional execute.
    assert!(out.contains("pubfnexecute_bound(&mutself,b:i64,)"));
}

#[test]
fn array_slots_expand_to_indexed_accessors() {
    let out = expand(quote! {
        FanJit(Fan) {
            inputs { xs: [Tensor; 3] }
            outputs { pairs: [Tensor; 2], total }

            build(xs) { model.forward(xs) }
        }
    });
    assert!(out.contains("pubfnprepare(&mutself,xs:[::svod_tensor::jit::InputSpec;3usize])"));
    assert!(out.contains("let[__jit_spec_xs_0,__jit_spec_xs_1,__jit_spec_xs_2]=xs;"));
    assert!(out.contains("letxs:[&::svod_tensor::Tensor;3usize]=[__jit_arg_xs_0,__jit_arg_xs_1,__jit_arg_xs_2];"));
    assert!(out.contains("pubfnxs_mut(&mutself,index:usize)"));
    assert!(out.contains("let[pairs_0,pairs_1]=pairs;"));
    assert!(out.contains("pubfnpairs(&self,index:usize)"));
    assert!(out.contains("pubfntotal(&self,)"));
    assert!(out.contains("plan.num_outputs()!=3usize"));
}

#[test]
fn state_slots_assign_back_and_stay_internal() {
    let out = expand(quote! {
        AccJit(Acc) {
            inputs { x: Tensor }
            state { h: Tensor, caches: [Tensor; 2] }
            outputs { y }

            build(x, h, caches) { model.step(x, h, caches) }
        }
    });
    // State is a prepare parameter, allocated device-local …
    assert!(out.contains("h:::svod_tensor::jit::InputSpec"));
    assert!(out.contains("::svod_tensor::jit::rt::BufferSpec{cpu_access:false,..Default::default()}"));
    // … an input the host can seed, but with no host write guard.
    assert!(out.contains("pubfnh_mut(&mutself)"));
    assert!(!out.contains("h_host_write"));
    assert!(out.contains("pubfnx_host_write"));
    // … whose new value is stored back into its own buffer.
    assert!(out.contains("let__jit_state_out_h={let__jit_out=::svod_tensor::Tensor::from_lazy(__jit_input_h.uop());"));
    assert!(out.contains("__jit_out.try_assign(&__jit_state_new_h)?;"));
    assert!(out.contains("let[__jit_state_new_caches_0,__jit_state_new_caches_1]=__jit_state_value_caches;"));
    // The build tuple is (declared outputs.., state values..).
    assert!(out.contains("let(y,__jit_state_value_h,__jit_state_value_caches,)="));
    // Declared + state outputs are what the plan must keep …
    assert!(out.contains("plan.num_outputs()!=4usize"));
    // … but only the declared ones are exposed.
    assert!(out.contains("pubfny(&self,)"));
    assert!(!out.contains("pubfnh(&self,)"));
    // Zero-fill per state slot.
    assert!(out.contains("pubfnreset(&mutself)"));
    assert!(out.contains("::svod_tensor::jit::zero_fill(buffer)?;"));
}

#[test]
fn state_slots_are_not_shrunk_by_batch_var() {
    let out = expand(quote! {
        J(M) {
            inputs { x: Tensor }
            batch_var b: (1, 4),
            state { h: Tensor }
            outputs { y }
            build(x, h) { model.step(x, h) }
        }
    });
    assert!(out.contains("__jit_shrunk_x"));
    assert!(!out.contains("__jit_shrunk_h"));
}

#[test]
fn declared_outputs_carry_live_shapes_views_and_reads() {
    let out = expand(quote! {
        J(M) {
            inputs { x: Tensor }
            batch_var b: (1, 4),
            outputs { y }
            build(x) { model.forward(x) }
        }
    });
    assert!(out.contains("::svod_tensor::jit::OutputShape::capture(&y,&[&b])?"));
    assert!(out.contains("pubfny_shape(&self,)->::svod_tensor::jit::Result<Vec<usize>>"));
    assert!(out.contains("pubfny_view<T:::svod_tensor::jit::rt::HasDType>(&self,)"));
    assert!(out.contains("pubfny_to_vec<T:::svod_tensor::jit::rt::HasDType+Default+Clone>(&self,)"));
    // Shapes resolve against the last bound values, which every execute path
    // that takes bindings records.
    assert!(out.contains("self.__jit_out_shapes[0usize].resolve(&self.__jit_var_values)"));
    assert!(out.contains("constNAMES:[&str;1usize]=[stringify!(b),];"));
}

#[test]
fn single_declared_output_needs_no_tuple() {
    let out = expand(quote! {
        J(M) {
            inputs { x: Tensor }
            outputs { y }
            build(x) { model.forward(x) }
        }
    });
    assert!(out.contains("lety=(||{"));
    assert!(!out.contains("let(y,)="));
}

// ---------------------------------------------------------------------------
// Rejected invocations.
// ---------------------------------------------------------------------------

#[test]
fn state_without_outputs_is_rejected() {
    let err = error(quote! {
        J(M) {
            inputs { x: Tensor }
            state { h: Tensor }
            build(x, h) { model.step(x, h) }
        }
    });
    assert!(err.contains("requires an `outputs"), "{err}");
}

#[test]
fn unbatched_without_batch_var_is_rejected() {
    let err = error(quote! {
        J(M) {
            inputs { #[unbatched] x: Tensor }
            build(x) { model.forward(x) }
        }
    });
    assert!(err.contains("requires a `batch_var` declaration"), "{err}");
}

#[test]
fn unbatched_on_state_or_output_is_rejected() {
    let err = error(quote! {
        J(M) {
            inputs { x: Tensor }
            batch_var b: (1, 4),
            state { #[unbatched] h: Tensor }
            outputs { y }
            build(x, h) { model.step(x, h) }
        }
    });
    assert!(err.contains("applies to inputs only"), "{err}");
}

#[test]
fn unknown_attribute_is_rejected() {
    let err = error(quote! {
        J(M) {
            inputs { #[batched] x: Tensor }
            build(x) { model.forward(x) }
        }
    });
    assert!(err.contains("expected `#[unbatched]`"), "{err}");
}

#[test]
fn duplicate_names_are_rejected() {
    let err = error(quote! {
        J(M) {
            inputs { x: Tensor }
            outputs { x }
            build(x) { model.forward(x) }
        }
    });
    assert!(err.contains("already declared as an input"), "{err}");

    let err = error(quote! {
        J(M) {
            inputs { x: Tensor }
            vars { x: (1, 4) }
            build(x) { model.forward(x) }
        }
    });
    assert!(err.contains("variable name already declared as an input"), "{err}");
}

#[test]
fn array_expansion_may_not_shadow_a_declared_name() {
    let err = error(quote! {
        J(M) {
            inputs { xs: [Tensor; 2], xs_0: Tensor }
            build(xs) { model.forward(xs) }
        }
    });
    assert!(err.contains("already declared"), "{err}");
}

#[test]
fn zero_length_array_slot_is_rejected() {
    let err = error(quote! {
        J(M) {
            inputs { xs: [Tensor; 0] }
            build(xs) { model.forward(xs) }
        }
    });
    assert!(err.contains("greater than zero"), "{err}");
}

#[test]
fn duplicate_batch_var_is_rejected() {
    let err = error(quote! {
        J(M) {
            inputs { x: Tensor }
            batch_var b: (1, 4),
            batch_var c: (1, 4),
            build(x) { model.forward(x) }
        }
    });
    assert!(err.contains("duplicate `batch_var`"), "{err}");
}

#[test]
fn unknown_build_arg_is_rejected() {
    let err = error(quote! {
        J(M) {
            inputs { x: Tensor }
            build(x, z) { model.forward(x, z) }
        }
    });
    assert!(err.contains("build arg must match"), "{err}");
}

#[test]
fn output_named_like_a_generated_method_is_rejected() {
    let err = error(quote! {
        J(M) {
            inputs { x: Tensor }
            outputs { execute, other }
            build(x) { model.forward(x) }
        }
    });
    assert!(err.contains("collides with a generated method"), "{err}");
}

#[test]
fn missing_build_block_is_rejected() {
    let err = error(quote! {
        J(M) {
            inputs { x: Tensor }
        }
    });
    assert!(err.contains("missing `build"), "{err}");
}

#[test]
fn malformed_var_bounds_are_rejected() {
    let err = error(quote! {
        J(M) {
            inputs { x: Tensor }
            vars { b: (1, 4, 9) }
            build(x) { model.forward(x) }
        }
    });
    assert!(err.contains("expected bounds as (min, max)"), "{err}");
}

/// Wrapper generics land on the struct and its impl and stay in scope inside
/// `build`, so one wrapper can be instantiated at several compile-time shapes
/// (`RnntBlockJit<const W: usize>` traces a different decode window per `W`).
#[test]
fn wrapper_generics_reach_the_struct_and_impl() {
    let out = expand(quote! {
        WindowJit<const W: usize>(M) {
            inputs { x: Tensor }
            build(x) { model.forward::<W>(x) }
        }
    });
    assert!(out.contains("pubstructWindowJit<constW:usize>"), "{out}");
    assert!(out.contains("impl<constW:usize>WindowJit<W>"), "{out}");
    assert!(out.contains("model.forward::<W>(x)"), "{out}");
    // The state struct carries no wrapper generics — it holds only buffers.
    assert!(out.contains("structWindowJitState{"), "{out}");
}
