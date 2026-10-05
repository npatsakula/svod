//! Metal-only pieces of the C-family renderer: SPECIAL launch axes and the
//! simdgroup-matrix lowering of WMMA.

use std::collections::BTreeMap;
use std::sync::Arc;

use svod_ir::{Op, UOp, WmmaMetadata, ops};

use super::dialect::CDialect;
use super::ops::CContext;
use super::types::c_scalar;
use crate::common::shaped_dtype;

/// Parse a SPECIAL axis name: `'g'/'l'/'i'` prefix + 0/1/2 axis suffix. Same
/// grammar as `crate::llvm::amd`; the producer is `ProgramSpec::special_launch_axis`.
fn parse_special_axis(name: &str) -> Option<(char, u8)> {
    let prefix = name.chars().next()?;
    if !matches!(prefix, 'g' | 'l' | 'i') {
        return None;
    }
    let suffix_start = name.rfind(|c: char| !c.is_ascii_digit()).map(|i| i + 1).unwrap_or(0);
    if suffix_start == name.len() {
        return None;
    }
    let axis: u8 = name[suffix_start..].parse().ok()?;
    (axis < 3).then_some((prefix, axis))
}

const AXIS_LETTERS: [char; 3] = ['x', 'y', 'z'];

/// `gidx*` reads the threadgroup position, `lidx*` the thread position in the
/// threadgroup. `idx*` (NOLOCALS) also reads the threadgroup position:
/// `ProgramInfo::from_sink` drops `local_size` for the `i` prefix and the
/// runtime dispatches one thread per threadgroup, so the group index is the
/// flat global index (same lowering as AMD). The value is declared with the IR
/// dtype so index arithmetic stays signed — `gid`/`lid` are `uint3`.
pub(super) fn render_special(uop: &Arc<UOp>, name: &str, ctx: &mut CContext, kernel: &mut Vec<String>) -> Option<()> {
    let Some((kind, axis)) = parse_special_axis(name) else {
        ctx.set_invalid_graph(format!("Metal renderer: malformed SPECIAL axis name {name:?}"));
        return None;
    };
    let dim = AXIS_LETTERS[axis as usize];
    let expr = if kind == 'l' { format!("lid.{dim}") } else { format!("gid.{dim}") };
    ctx.emit_named(uop, expr, name.to_string(), &uop.dtype(), kernel);
    Some(())
}

const SIMDGROUP_DIMS: (usize, usize, usize) = (8, 8, 8);
const SIMDGROUP_THREADS: usize = 32;

/// `D = __WMMA_8_8_8_<in>_<out>(a, b, c)` over the helper from
/// [`wmma_helper_prefix`]; every operand is the 2-element per-thread fragment
/// of an 8x8 simdgroup matrix.
pub(super) fn render_wmma(
    uop: &Arc<UOp>,
    a: &Arc<UOp>,
    b: &Arc<UOp>,
    c: &Arc<UOp>,
    metadata: &WmmaMetadata,
    ctx: &mut CContext,
    kernel: &mut Vec<String>,
) -> Option<()> {
    if metadata.dims != SIMDGROUP_DIMS || metadata.threads != SIMDGROUP_THREADS {
        ctx.set_invalid_graph(format!(
            "Metal renderer: only the 8x8x8/32-thread simdgroup matrix exists; got dims={:?} threads={}",
            metadata.dims, metadata.threads
        ));
        return None;
    }
    for (label, operand) in [("a", a), ("b", b), ("c", c)] {
        let width = shaped_dtype(operand).vcount();
        if width != 2 {
            ctx.set_invalid_graph(format!(
                "Metal WMMA operand {label} on uop {} must be 2-wide (elements_per_thread=(2,2,2)), got {width}",
                uop.id
            ));
            return None;
        }
    }
    let expr = format!("__{}({}, {}, {})", metadata.name, ctx.get(a), ctx.get(b), ctx.get(c));
    let dtype = shaped_dtype(uop);
    ctx.emit_expr_dtype(uop, expr, "wmma", kernel, &dtype, false);
    Some(())
}

/// One helper per distinct WMMA shape/dtype pair, in name order.
pub(super) fn wmma_helper_prefix(nodes: &[Arc<UOp>]) -> Vec<String> {
    let mut helpers = BTreeMap::new();
    for node in nodes {
        if let Op::Wmma(ops::Wmma { metadata, .. }) = node.op() {
            helpers.entry(metadata.name.clone()).or_insert_with(|| wmma_helper(metadata));
        }
    }
    helpers.into_values().collect()
}

fn wmma_helper(metadata: &WmmaMetadata) -> String {
    let inp = c_scalar(metadata.dtype_in.base(), CDialect::Metal);
    let out = c_scalar(metadata.dtype_out.base(), CDialect::Metal);
    let name = &metadata.name;
    format!(
        "{out}2 __{name}({inp}2 a, {inp}2 b, {out}2 c){{\n\
         \x20 simdgroup_{inp}8x8 mat_a, mat_b; simdgroup_{out}8x8 mat_c;\n\
         \x20 mat_a.thread_elements()[0] = a[0]; mat_b.thread_elements()[0] = b[0]; mat_c.thread_elements()[0] = c[0];\n\
         \x20 mat_a.thread_elements()[1] = a[1]; mat_b.thread_elements()[1] = b[1]; mat_c.thread_elements()[1] = c[1];\n\
         \x20 simdgroup_multiply_accumulate(mat_c, mat_a, mat_b, mat_c);\n\
         \x20 return {out}2(mat_c.thread_elements()[0], mat_c.thread_elements()[1]);\n\
         }}"
    )
}

// ── Cross-lane builders ───────────────────────────────────────────────────────
//
// The MSL analog of `llvm::nvptx::ops::shfl_*`. MSL's `simd_*` are overloaded
// functions in `metal_stdlib` (already included by the prelude), so unlike the
// AMD `ds_bpermute` path there is no declaration to hoist, no i32 round-trip and
// no lane-address arithmetic — the whole lowering is the call text. These build
// `Op::Custom` (a typed SSA binding), never `Op::CustomI`: a `CustomI` is spliced
// once per consumer, and re-evaluating a cross-lane op per use is a convergence
// hazard, not just a redundant instruction.

/// MSL's `simd_*` overload set covers the 32-bit-and-narrower scalars and their
/// vectors; there is **no 64-bit overload** (`long`/`ulong` fail to compile,
/// verified on Apple9). NVPTX's `shfl` has the same width limit and splits a 64-bit
/// value through two 32-bit exchanges; nothing in tk shuffles 64 bits (reduces carry
/// f32 values and Int32 indices), so rather than carry an unexercised split this
/// rejects the width at the builder, turning an opaque MSL compile failure into a
/// precise one. A caller that needs it splits at the call site — the same place
/// NVPTX's would have to.
fn reject_64_bit(op: &str, dtype: &svod_dtype::DType) {
    assert!(
        dtype.bytes() <= 4,
        "Metal `{op}`: MSL has no 64-bit cross-lane overload, got {dtype:?} — split through two 32-bit halves"
    );
}

/// `simd_shuffle(value, lane)` — read `value` from `lane` of the SIMD group.
///
/// # Panics
/// Panics on a value wider than 32 bits (see `reject_64_bit`).
pub fn simd_shuffle(value: &Arc<UOp>, lane: &Arc<UOp>) -> Arc<UOp> {
    let dtype = value.dtype();
    reject_64_bit("simd_shuffle", &dtype);
    UOp::custom(smallvec::smallvec![value.clone(), lane.clone()], "simd_shuffle({0}, (ushort)({1}))".to_string(), dtype)
}

/// `simd_shuffle_xor(value, mask)` — the butterfly exchange with lane `L ^ mask`.
///
/// # Panics
/// Panics on a value wider than 32 bits (see `reject_64_bit`).
pub fn simd_shuffle_xor(value: &Arc<UOp>, mask: &Arc<UOp>) -> Arc<UOp> {
    let dtype = value.dtype();
    reject_64_bit("simd_shuffle_xor", &dtype);
    UOp::custom(
        smallvec::smallvec![value.clone(), mask.clone()],
        "simd_shuffle_xor({0}, (ushort)({1}))".to_string(),
        dtype,
    )
}

/// `simdgroup_barrier(mem_flags::mem_threadgroup)` — the SIMD-group-scoped fence,
/// cheaper than the threadgroup barrier `Op::Barrier` lowers to.
pub fn simdgroup_barrier(deps: smallvec::SmallVec<[Arc<UOp>; 4]>) -> Arc<UOp> {
    UOp::custom(deps, "simdgroup_barrier(mem_flags::mem_threadgroup)".to_string(), svod_dtype::DType::Void)
}
