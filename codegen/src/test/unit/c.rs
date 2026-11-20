//! C renderer tests for code generation verification.

use smallvec::SmallVec;
use svod_dtype::{AddrSpace, DType, DeviceSpec};
use svod_ir::{AxisId, AxisType, ConstValue, Op, ParamArg, ReduceOp, UOp};

use crate::c::CDialect;
use crate::c::render;
use crate::c::types::c_const;
use svod_ir::ops;

fn render_linearized(root: &std::sync::Arc<UOp>, name: Option<&str>) -> crate::Result<crate::RenderedKernel> {
    let linear = UOp::linear(svod_schedule::linearize_with_cfg(root.clone()).into());
    render(&linear, name)
}

fn concrete_range(end: i64, axis_type: AxisType) -> std::sync::Arc<UOp> {
    let end = UOp::const_(DType::Int32, ConstValue::Int(end));
    UOp::new(
        Op::Range(ops::Range { end, axis_id: AxisId::Renumbered(0), axis_type, deps: SmallVec::new() }),
        DType::Int32,
    )
}

fn slotted_var(name: &str, slot: usize) -> std::sync::Arc<UOp> {
    let var = UOp::variable(name.to_string(), 0, 16, DType::Int32);
    let Op::Param(ops::Param { shape, arg }) = var.op() else { unreachable!() };
    let mut arg = arg.clone();
    arg.slot = slot;
    UOp::new(Op::Param(ops::Param { shape: shape.clone(), arg }), DType::Int32)
}

fn volatile_param(slot: usize, size: usize) -> std::sync::Arc<UOp> {
    let mut arg = ParamArg::buffer(slot, DType::Float32, AddrSpace::Global, None);
    arg.volatile = true;
    UOp::new(Op::Param(ops::Param { shape: UOp::index_const(size as i64), arg: arg.into() }), DType::Float32)
}

fn shrink4(src: std::sync::Arc<UOp>, dtype: DType) -> std::sync::Arc<UOp> {
    UOp::new(
        Op::Shrink(ops::Shrink {
            src,
            offsets: UOp::const_(DType::Int32, ConstValue::Int(0)),
            sizes: UOp::const_(DType::Int32, ConstValue::Int(4)),
        }),
        dtype,
    )
}

fn element(buffer: std::sync::Arc<UOp>, lane: i64) -> std::sync::Arc<UOp> {
    UOp::index().buffer(buffer).indices(vec![UOp::const_(DType::Index, ConstValue::Int(lane))]).call().unwrap()
}

#[test]
fn c_signature_uses_canonical_mixed_param_slot_order() {
    let sink = UOp::sink(vec![
        slotted_var("high", 3),
        UOp::param(2, 1, DType::Float32, None),
        slotted_var("low", 1),
        UOp::param(0, 1, DType::Float32, None),
    ]);
    let rendered = render_linearized(&sink, Some("mixed_abi")).expect("render mixed ABI");
    assert_eq!(rendered.buffer_args.iter().map(|arg| arg.index).collect::<Vec<_>>(), vec![0, 2]);
    assert_eq!(rendered.var_names, vec!["low", "high"]);
    assert!(
        rendered
            .code
            .contains("void mixed_abi(float* restrict data0, const int data1, float* restrict data2, const int data3)"),
        "{}",
        rendered.code
    );
}

#[test]
fn c_qualifies_only_volatile_buffer_parameters() {
    let sink = UOp::sink(vec![volatile_param(0, 1), UOp::param(1, 1, DType::Float32, None)]);
    let rendered = render_linearized(&sink, Some("volatile_params")).expect("render volatile C ABI");
    assert!(
        rendered.code.contains("void volatile_params(volatile float* restrict data0, float* restrict data1)"),
        "{}",
        rendered.code
    );
}

#[test]
fn test_render_linear_input_succeeds() {
    let sink = UOp::sink(vec![UOp::const_(DType::Float32, ConstValue::Float(1.0))]);
    let linear = UOp::linear(svod_schedule::linearize_with_cfg(sink.clone()).into());

    let rendered = render(&linear, Some("test_linear")).expect("C codegen from LINEAR should succeed");
    assert!(rendered.code.contains("test_linear"));
}

/// A SHRINK on both endpoints groups into one vector load and one vector store,
/// with the vector typedef named after the shape width and the element type.
#[test_case::test_case(DType::Float32, "float4", "typedef float float4"; "f32")]
#[test_case::test_case(DType::Float16, "half4", "typedef _Float16 half4"; "f16")]
fn grouped_shrink_renders_single_vector_load_and_store(dtype: DType, vector: &str, typedef: &str) {
    let output = shrink4(UOp::param(0, 8, dtype.clone(), None), dtype.clone());
    let input = shrink4(UOp::param(1, 8, dtype.clone(), None), dtype);
    let sink = UOp::sink(vec![output.store(UOp::load().index(input).call())]);

    let rendered = render_linearized(&sink, Some("grouped_memory")).expect("render grouped C memory");
    assert!(rendered.code.contains(typedef), "{}", rendered.code);
    // Two typedef lines, the load and the store.
    assert_eq!(rendered.code.matches(vector).count(), 4, "{}", rendered.code);
    assert!(!rendered.code.contains("void4"), "{}", rendered.code);
    assert_c_compiles(&rendered.code);
}

#[test]
fn clang_materializes_tinygrad_ssa_boundaries() {
    let selected = UOp::try_where(
        UOp::const_(DType::Bool, ConstValue::Bool(true)),
        UOp::const_(DType::Int32, ConstValue::Int(2)),
        UOp::const_(DType::Int32, ConstValue::Int(3)),
    )
    .unwrap();
    let out_index = element(UOp::param(0, 1, DType::Int32, None), 0);
    let rendered = render_linearized(&UOp::sink(vec![out_index.store(selected)]), Some("materialize_where"))
        .expect("render WHERE materialization");
    assert!(rendered.code.contains("int alu0 = (1 ? 2 : 3);"), "{}", rendered.code);
    assert!(rendered.code.contains("= alu0;"), "{}", rendered.code);

    let input_index = element(UOp::param(0, 1, DType::Float32, None), 0);
    let output_index = element(UOp::param(1, 1, DType::Float32, None), 0);
    let rendered = render_linearized(
        &UOp::sink(vec![output_index.store(UOp::load().index(input_index).call())]),
        Some("materialize_load"),
    )
    .expect("render LOAD materialization");
    assert!(rendered.code.contains("float val0 = *(data0 + 0LL);"), "{}", rendered.code);
    assert!(rendered.code.contains("= val0;"), "{}", rendered.code);

    let vector = UOp::vconst(vec![ConstValue::UInt(1); 4], DType::UInt32);
    let cast = vector.cast(DType::UInt32.vec(4).unwrap()).cast(DType::Int32.vec(4).unwrap());
    let rendered = render_linearized(&UOp::sink(vec![cast]), Some("materialize_vector_cast"))
        .expect("render vector CAST materialization");
    assert!(rendered.code.contains("int4 cast0 = __builtin_convertvector"), "{}", rendered.code);
}

#[test]
fn clang_preserves_shape_width_across_materialized_values_and_store_aliases() {
    let shaped = UOp::stack((0..4).map(|value| UOp::const_(DType::UInt32, ConstValue::UInt(value))).collect());
    let rendered = render_linearized(&UOp::sink(vec![shaped.cast(DType::Float32)]), Some("materialize_shaped_cast"))
        .expect("render scalar-dtype shaped CAST");
    assert!(rendered.code.contains("typedef float float4"), "{}", rendered.code);
    assert!(rendered.code.contains("float4 cast0 = __builtin_convertvector"), "{}", rendered.code);

    let output = shrink4(UOp::param(0, 4, DType::Int32, None), DType::Int32);
    let value = UOp::stack((0..4).map(|value| UOp::const_(DType::Int32, ConstValue::Int(value))).collect()).detach();
    let rendered = render_linearized(&UOp::sink(vec![output.store(value)]), Some("store_shaped_alias"))
        .expect("render shaped alias STORE");
    assert!(rendered.code.contains("*((int4*)(data0 + 0)) ="), "{}", rendered.code);
}

#[test]
fn clang_hoists_addresses_that_escape_their_loop() {
    // The linearizer can place a shared node inside a range whose consumer sits
    // outside it. Addresses are inlined, so without a hoist the STORE after the
    // loop would reference `ridx0` — an out-of-scope identifier.
    let param = UOp::param(0, 8, DType::Float32, None);
    let bound = UOp::const_(DType::Int32, ConstValue::Int(4));
    let range = UOp::range_axis_dtype(bound.clone(), AxisId::Renumbered(0), AxisType::Loop, DType::Int32);
    let index = UOp::index().buffer(param.clone()).indices(vec![range.clone()]).call().unwrap();
    let load = UOp::load().index(index.clone()).call();
    let end = load.end(smallvec::smallvec![range.clone()]);
    let store = index.clone().store(load.clone());
    let linear = UOp::linear(smallvec::smallvec![bound, param, range, index, load, end, store]);

    let rendered = render(&linear, Some("escaping_address")).expect("render escaping address");
    assert!(rendered.code.contains("  float* bidx0;"), "{}", rendered.code);
    let (_, after_loop) = rendered.code.split_once("\n  }\n").expect("loop must close");
    assert!(!after_loop.contains("ridx0"), "{}", rendered.code);
    assert_c_compiles(&rendered.code);
}

#[test]
fn clang_store_width_follows_the_stored_value() {
    let index = element(UOp::param(0, 8, DType::Float32, None), 0);
    let value = UOp::vconst(vec![ConstValue::Float(1.0); 4], DType::Float32);
    let rendered = render_linearized(&UOp::sink(vec![index.store(value)]), Some("store_vector_through_scalar_index"))
        .expect("render vector STORE through a scalar-width index");
    assert!(rendered.code.contains("*((float4*)(data0 + 0"), "{}", rendered.code);
    assert_c_compiles(&rendered.code);
}

#[test]
fn clang_stack_packs_loaded_index_lanes() {
    let input = shrink4(UOp::param(0, 8, DType::Float32, None), DType::Float32);
    let lanes = (0..4).map(|lane| UOp::load().index(element(input.clone(), lane)).call()).collect();
    let output = shrink4(UOp::param(1, 8, DType::Float32, None), DType::Float32);

    let rendered =
        render_linearized(&UOp::sink(vec![output.store(UOp::stack(lanes).detach())]), Some("stack_index_lanes"))
            .expect("render STACK of loaded INDEX lanes");
    assert!(rendered.code.contains("(float4){val0, val1, val2, val3}"), "{}", rendered.code);
    assert_c_compiles(&rendered.code);
}

#[test]
fn clang_preserves_address_casts_as_pointers() {
    let address = element(UOp::param(0, 1, DType::Float32, None), 0).cast(DType::Int32);
    let output = element(UOp::param(1, 1, DType::Int32, None), 0);
    let rendered =
        render_linearized(&UOp::sink(vec![output.store(UOp::load().index(address).call())]), Some("address_cast"))
            .expect("render address CAST");
    assert!(rendered.code.contains("((int*)(data0 + 0LL))"), "{}", rendered.code);
    assert!(!rendered.code.contains("__builtin_convertvector"), "{}", rendered.code);
}

#[test]
fn clang_vector_alignment_rounds_down_like_tinygrad() {
    let vector = UOp::const_(DType::Float32.vec(3).unwrap(), ConstValue::Float(0.0));
    let rendered = render_linearized(&UOp::sink(vec![vector]), Some("float3_alignment")).expect("render float3");

    assert!(
        rendered.code.contains("typedef float float3 __attribute__((aligned(8),ext_vector_type(3)))"),
        "{}",
        rendered.code
    );
}

/// C evaluates sub-`int` scalar arithmetic at `int` width and the `f`-suffixed
/// builtins at f32; both results must be cast back to the IR dtype so an
/// inlined consumer sees the wrapped or rounded value.
#[test_case::test_case(DType::Int8, "signed char"; "int8 add")]
#[test_case::test_case(DType::UInt16, "unsigned short"; "uint16 add")]
#[test_case::test_case(DType::Float16, "_Float16"; "half sqrt")]
#[test_case::test_case(DType::BFloat16, "__bf16"; "bfloat16 sqrt")]
fn clang_narrows_promoted_arithmetic(dtype: DType, c_type: &str) {
    let a = UOp::load().index(element(UOp::param(1, 4, dtype.clone(), None), 0)).call();
    let value = if dtype.is_float() { a.try_sqrt().expect("sqrt") } else { a.try_add(&a).expect("add") };
    let wide = value.cast(DType::Int32);
    let sink = UOp::sink(vec![element(UOp::param(0, 4, DType::Int32, None), 0).store(wide)]);

    let rendered = render_linearized(&sink, Some("narrow")).expect("render narrowed C arithmetic");
    assert!(rendered.code.contains(&format!("({c_type})")), "{}", rendered.code);
    assert_c_compiles(&rendered.code);
}

#[test]
fn test_render_rejects_non_linear_inputs() {
    let sink = UOp::sink(vec![UOp::const_(DType::Float32, ConstValue::Float(1.0))]);
    let info = svod_ir::ProgramInfo::from_sink(&sink, DeviceSpec::Cpu);
    let program = UOp::program(sink.clone(), info, None, None, None);

    let err = render(&program, Some("test_program_input")).expect_err("PROGRAM input must fail");
    assert!(format!("{err}").contains("expects LINEAR input"), "unexpected error: {err:?}");

    let err = render(&sink, Some("test_sink_input")).expect_err("SINK input must fail");
    assert!(format!("{err}").contains("expects LINEAR input"), "unexpected error: {err:?}");
}

#[test]
fn test_getaddr_must_be_resolved_before_codegen() {
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::UInt8);
    let address = buffer.getaddr(None);
    let linear = UOp::linear(vec![buffer, address].into());
    let err = render(&linear, Some("getaddr")).expect_err("GETADDR is an HCQ runtime op, not a kernel op");
    assert!(format!("{err}").contains("GetAddr"), "unexpected error: {err:?}");
}

#[test]
fn test_render_rejects_fnuz_without_fallback() {
    let constant = UOp::const_(DType::FP8E5M2FNUZ, ConstValue::Float(1.0));
    let linear = UOp::linear(vec![constant].into());
    let err = render(&linear, Some("fnuz")).expect_err("FNUZ rendering must fail");
    let message = format!("{err}");
    assert!(message.contains("does not support FP8E5M2FNUZ"), "unexpected error: {message}");
    assert!(message.contains("cannot use OCP FP8 decomposition or raw-byte fallback"), "unexpected error: {message}");
}

#[test]
fn c_constants_consume_committed_values_and_fp8_bits() {
    let f32_value = UOp::const_(DType::Float32, ConstValue::Float(-3.2));
    let Op::Const(f32_value) = f32_value.op() else { unreachable!() };
    assert_eq!(c_const(&f32_value.0, &DType::Float32, CDialect::Clang), "-3.2e0f");

    let fp8 = UOp::const_(DType::FP8E4M3, ConstValue::Float(1.1875));
    let Op::Const(fp8) = fp8.op() else { unreachable!() };
    assert_eq!(c_const(&fp8.0, &DType::FP8E4M3, CDialect::Clang), "58");
}

fn empty_loop_sink() -> std::sync::Arc<UOp> {
    let range = concrete_range(10, AxisType::Loop);
    UOp::sink(vec![UOp::noop().end(smallvec::smallvec![range])])
}

fn reduce_sink(value: f64, extent: i64, op: ReduceOp) -> std::sync::Arc<UOp> {
    let range = concrete_range(extent, AxisType::Reduce);
    let reduce = UOp::const_(DType::Float32, ConstValue::Float(value)).reduce(smallvec::smallvec![range.clone()], op);
    UOp::sink(vec![reduce.end(smallvec::smallvec![range])])
}

#[test_case::test_case(empty_loop_sink(), &["for", "ridx0", "< 10"]; "loop over a concrete range")]
#[test_case::test_case(reduce_sink(5.0, 10, ReduceOp::Add), &["acc", "for", "+=", "0.0f"]; "sum accumulates from the identity")]
#[test_case::test_case(reduce_sink(3.0, 5, ReduceOp::Max), &["acc", "fmaxf", "-__builtin_inff()"]; "max starts from -inf and uses fmaxf")]
#[test_case::test_case(reduce_sink(3.0, 5, ReduceOp::Min), &["acc", "fminf", "__builtin_inff()"]; "min starts from +inf and uses fminf")]
fn c_renders_range_and_reduce_structure(sink: std::sync::Arc<UOp>, needles: &[&str]) {
    let result = render_linearized(&sink, Some("range_or_reduce")).expect("C codegen failed");
    for needle in needles {
        assert!(result.code.contains(needle), "missing {needle}:\n{}", result.code);
    }
    assert_c_compiles(&result.code);
}

#[test]
fn reduce_without_ranges_renders_as_a_plain_value() {
    let reduce = UOp::const_(DType::Float32, ConstValue::Float(42.0)).reduce(smallvec::smallvec![], ReduceOp::Add);
    let result = render_linearized(&UOp::sink(vec![reduce]), Some("test_reduce_empty")).expect("C codegen failed");
    assert!(!result.code.contains("for"), "a rangeless REDUCE must not emit a loop:\n{}", result.code);
    assert_c_compiles(&result.code);
}

#[test]
fn test_multi_index_requires_linearization() {
    let buffer = UOp::param(0, 1024, DType::Float32, None);
    let i = UOp::const_(DType::Index, ConstValue::Int(1));
    let j = UOp::const_(DType::Index, ConstValue::Int(2));
    let index = UOp::index().buffer(buffer).indices(vec![i, j]).call().unwrap();
    let sink = UOp::sink(vec![index]);

    let linear = UOp::linear(sink.toposort().into());
    let err = render(&linear, Some("test_multi_index_requires_linearization"))
        .expect_err("multi-index INDEX must surface as InvalidGraph");
    assert!(
        matches!(&err, crate::Error::InvalidGraph { reason } if reason.contains("linearized INDEX")),
        "expected InvalidGraph(linearized INDEX), got {err:?}",
    );
}

#[test]
fn test_gated_load_emits_conditional_dereference() {
    let buffer = UOp::param(0, 1024, DType::Float32, None);
    let gate = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let alt = UOp::const_(DType::Float32, ConstValue::Float(7.0));
    let load = UOp::load().index(element(buffer, 1)).alt(alt).gate(gate).call();
    let out_idx = element(UOp::param(1, 1024, DType::Float32, None), 0);

    let rendered = render_linearized(&UOp::sink(vec![out_idx.store(load)]), Some("gated_load"))
        .expect("C backend should render gated load");
    assert!(
        rendered.code.contains("1 ? *(data0 + 1LL) : 7.0f"),
        "gated load should conditionally evaluate the dereference:\n{}",
        rendered.code
    );
}

#[test]
fn test_custom_statement_is_materialized() {
    let one = UOp::const_(DType::Int32, ConstValue::Int(1));
    let expr = UOp::custom(smallvec::smallvec![one], "({0} + 3)".to_string(), DType::Int32);
    let stmt = UOp::custom(smallvec::smallvec![expr], "sink({0})".to_string(), DType::Void);

    let result = render_linearized(&UOp::sink(vec![stmt]), Some("test_custom_stmt")).expect("C codegen failed");

    assert!(
        result.code.contains("int custom0 = (1 + 3);"),
        "CUSTOM should materialize to a statement:\n{}",
        result.code
    );
    assert!(
        result.code.contains("sink(custom0);"),
        "CUSTOM consumer should reference materialized value:\n{}",
        result.code
    );
}

#[test]
fn test_customi_is_inline_and_formats_placeholders() {
    let operands = (1..=3).map(|value| UOp::const_(DType::Int32, ConstValue::Int(value))).collect();
    let inline = UOp::customi(operands, "{0} + {2} + {1}".to_string(), DType::Int32);
    let stmt = UOp::custom(smallvec::smallvec![inline], "emit({0})".to_string(), DType::Void);

    let result = render_linearized(&UOp::sink(vec![stmt]), Some("test_customi_inline")).expect("C codegen failed");

    assert!(
        result.code.contains("emit(1 + 3 + 2);"),
        "CUSTOMI should stay inline and substitute placeholders in-order:\n{}",
        result.code
    );
    assert!(!result.code.contains("custom0 ="), "CUSTOMI must not create temp statements:\n{}", result.code);
}

fn out_of_bounds_placeholder() -> std::sync::Arc<UOp> {
    let one = UOp::const_(DType::Int32, ConstValue::Int(1));
    UOp::custom(smallvec::smallvec![one], "emit({1})".to_string(), DType::Void)
}

fn unmatched_brace() -> std::sync::Arc<UOp> {
    let one = UOp::const_(DType::Int32, ConstValue::Int(1));
    UOp::custom(smallvec::smallvec![one], "emit({0".to_string(), DType::Void)
}

fn mixed_placeholder_modes() -> std::sync::Arc<UOp> {
    let a = UOp::const_(DType::Int32, ConstValue::Int(1));
    let b = UOp::const_(DType::Int32, ConstValue::Int(2));
    UOp::customi(smallvec::smallvec![a, b], "{} + {1}".to_string(), DType::Int32)
}

#[test_case::test_case(out_of_bounds_placeholder, "out of bounds"; "index past the last operand")]
#[test_case::test_case(unmatched_brace, "unmatched"; "unclosed brace")]
#[test_case::test_case(mixed_placeholder_modes, "mixes automatic"; "automatic and manual indices mixed")]
fn custom_template_errors(build: fn() -> std::sync::Arc<UOp>, reason: &str) {
    let err = render_linearized(&UOp::sink(vec![build()]), Some("bad_template"))
        .expect_err("malformed CUSTOM template must fail");
    assert!(format!("{err}").contains(reason), "unexpected error: {err}");
}

/// Pipe `src` through `clang -fsyntax-only` and assert it parses. Returns
/// without asserting when no clang is on PATH. Mirrors
/// `assert_llvm_ir_assembles` in `llvm_text.rs`.
fn assert_c_compiles(src: &str) {
    use std::io::Write;
    use std::process::{Command, Stdio};

    let Ok(mut child) = Command::new("clang")
        .args(["-fsyntax-only", "-Wno-unused-value", "-x", "c", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
    else {
        return;
    };
    child.stdin.take().unwrap().write_all(src.as_bytes()).expect("write C source to clang");
    let output = child.wait_with_output().expect("wait for clang");
    assert!(
        output.status.success(),
        "clang rejected the emitted C:\n{src}\n--- clang stderr ---\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Neither C dialect passes a tensor map: the PARAM is refused by slot, not
/// rendered as a scalar.
#[test]
fn c_dialects_reject_a_tensor_map_param() {
    let map = svod_ir::TensorMap::new(
        0,
        svod_dtype::ScalarDType::Float32,
        [64, 32],
        [16, 8],
        svod_ir::TensorMapSwizzle::None,
    )
    .unwrap();
    let read = UOp::custom(
        smallvec::smallvec![UOp::tensor_map_param(1, map), UOp::param(0, 64 * 32, DType::Float32, None)],
        "/* {0} {1} */".to_string(),
        DType::Void,
    );
    let linear = UOp::linear(svod_schedule::linearize_with_cfg(UOp::sink(vec![read])).into());
    for rendered in [render(&linear, Some("k")), crate::c::render_metal(&linear, Some("k"))] {
        let error = rendered.unwrap_err();
        assert!(matches!(error, crate::Error::UnsupportedTensorMapParam { slot: 1, .. }), "{error:?}");
    }
}
