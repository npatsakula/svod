//! The predicates that decide whether a hand kernel gets the work.
//!
//! `svod_tk` splits a refusal two ways: `Ok(None)` where the device or the
//! runtime geometry does not suit it (the caller substitutes the graph), and
//! `Err` where an operand's dtype or shape is not the one its ABI declares — a
//! caller bug that propagates out of `forward` instead of falling back. A gate
//! that reads only the activation lets the second kind through, so these tests
//! pin the operand properties each gate has to mirror.
//!
//! The checks are device-free: `launch_custom` answers `Ok(None)` before it
//! validates anything on a host without the kernel's arch, so only the gate
//! itself can be exercised here.

use svod_dtype::DType;
use svod_tensor::Tensor;
use test_case::test_case;

use crate::qwen3::norm_fusable;
use crate::qwen3::tk::{Heads, fusable as prologue_fusable};

/// A `[B, L, D]` activation and the `[D]` norm weight that goes with it.
const ROWS: [usize; 3] = [2, 8, 64];

#[test]
fn the_norm_gate_takes_a_matching_activation_and_weight() {
    let x = Tensor::empty(&ROWS, DType::BFloat16);
    assert!(norm_fusable(&x, &Tensor::empty(&[ROWS[2]], DType::BFloat16)));
}

/// `svod_tk::rms_norm` calls a weight that is not `[D]` in the activation's own
/// dtype malformed (`check_norm_operands`), so the gate has to catch it before
/// the launch turns it into an error the forward cannot recover from.
#[test_case(&[64], DType::Float32; "an f32 weight beside a bf16 activation")]
#[test_case(&[1, 64], DType::BFloat16; "a weight the kernel cannot read as a row")]
fn the_norm_gate_refuses_a_weight_the_kernel_would_reject(dims: &[usize], dtype: DType) {
    let x = Tensor::empty(&ROWS, DType::BFloat16);
    assert!(!norm_fusable(&x, &Tensor::empty(dims, dtype)));
}

/// A 32-bit stream has no matrix-core path at all, matching weight or not.
#[test]
fn the_norm_gate_refuses_a_thirty_two_bit_activation() {
    let x = Tensor::empty(&ROWS, DType::Float32);
    assert!(!norm_fusable(&x, &Tensor::empty(&[ROWS[2]], DType::Float32)));
}

const HEADS: Heads = Heads { h: 4, h_kv: 2, dh: 32 };
const BATCH: usize = 2;
const SEQ: usize = 8;

/// The fused QKV row width the head geometry implies.
fn qkv_row() -> usize {
    (HEADS.h + 2 * HEADS.h_kv) * HEADS.dh
}

/// The prologue's operands at the shapes and dtype the attention layer hands
/// it: the GEMM output, the two per-head norm weights, and the sequence-major
/// half-width rope tables the model caches.
fn prologue_operands(dtype: DType) -> [Tensor; 5] {
    let rope = || Tensor::empty(&[1, SEQ, 1, HEADS.dh / 2], dtype.clone());
    [
        Tensor::empty(&[BATCH, SEQ, qkv_row()], dtype.clone()),
        Tensor::empty(&[HEADS.dh], dtype.clone()),
        Tensor::empty(&[HEADS.dh], dtype.clone()),
        rope(),
        rope(),
    ]
}

fn gate(operands: &[Tensor; 5]) -> bool {
    let [qkv, q_weight, k_weight, cos, sin] = operands;
    prologue_fusable(qkv, q_weight, k_weight, cos, sin, HEADS)
}

#[test_case(DType::BFloat16; "bf16")]
fn the_prologue_gate_takes_matching_operands(dtype: DType) {
    assert!(gate(&prologue_operands(dtype)));
}

/// Every operand must carry `qkv`'s dtype, weights and rope tables included —
/// the kernel reads them through one ABI and rejects a mismatch outright.
#[test_case(0; "an f32 activation")]
#[test_case(1; "an f32 query norm weight")]
fn the_prologue_gate_refuses_a_dtype_the_kernel_would_reject(operand: usize) {
    let mut operands = prologue_operands(DType::BFloat16);
    operands[operand] = Tensor::empty(&operands[operand].dims().unwrap(), DType::Float32);
    assert!(!gate(&operands));
}

/// The row the kernel addresses is `(h + 2·h_kv)·dh` wide and exactly rank 3.
#[test_case(&[BATCH, SEQ, 320]; "a row the head geometry does not add up to")]
#[test_case(&[BATCH * SEQ, 256]; "a row already flattened")]
fn the_prologue_gate_refuses_an_activation_shape_the_kernel_would_reject(dims: &[usize]) {
    let mut operands = prologue_operands(DType::BFloat16);
    operands[0] = Tensor::empty(dims, DType::BFloat16);
    assert!(!gate(&operands));
}

/// The rope tables are `dh/2` wide over one row per position or per token; a
/// full-width table has the right element count and the wrong layout.
#[test_case(&[1, SEQ, 1, HEADS.dh]; "a full-width table")]
fn the_prologue_gate_refuses_a_rope_table_the_kernel_would_reject(dims: &[usize]) {
    let mut operands = prologue_operands(DType::BFloat16);
    operands[3] = Tensor::empty(dims, DType::BFloat16);
    assert!(!gate(&operands));
}

/// One table per token and one per position address different rows, so the two
/// have to agree with each other.
#[test]
fn the_prologue_gate_refuses_rope_tables_of_two_different_layouts() {
    let mut operands = prologue_operands(DType::BFloat16);
    operands[3] = Tensor::empty(&[BATCH, SEQ, 1, HEADS.dh / 2], DType::BFloat16);
    assert!(!gate(&operands));
}

/// A `[dh]` weight each: the kernel norms one head per wave over that axis.
#[test_case(1; "the query norm weight")]
fn the_prologue_gate_refuses_a_norm_weight_of_the_wrong_width(operand: usize) {
    let mut operands = prologue_operands(DType::BFloat16);
    operands[operand] = Tensor::empty(&[HEADS.dh / 2], DType::BFloat16);
    assert!(!gate(&operands));
}
