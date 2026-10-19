//! Warpgroup matrix products for NVPTX (`wgmma.mma_async`, sm_90a): the one
//! place that writes PTX by hand. LLVM 22 exposes the warpgroup fence, commit
//! and wait as intrinsics but no `mma_async` form, so a product is an inline
//! `asm` block holding a whole K chain between its fence and its wait: the
//! accumulator registers are tied in and out of the one statement, so
//! nothing the compiler schedules can touch them while the core still owns
//! them (ptxas would otherwise serialize the chain and warn).
//!
//! The core reads shared memory through the async proxy, so a chain opens
//! with `fence.proxy.async` over what the generic proxy stored there.
//!
//! A shared operand reaches the core through a 64-bit matrix descriptor
//! ([`descriptor`]) over its shared address, the K-major canonical layouts
//! whose swizzle modes are those of the TMA (and of tk3's XOR swizzle of
//! 16-byte chunks against the row).

use std::sync::Arc;

use smallvec::smallvec;
use svod_dtype::{AddrSpace, DType, ScalarDType};
use svod_ir::{WmmaMetadata, prelude::*};

use crate::llvm::common::gpu::{specific_ptr, wmma_operand_dtype};
use crate::llvm::common::{RenderContext, ldt};
use crate::llvm::nvptx::wmma::{Wire, join_fragment, split_fragment};

/// Rows of the product every warpgroup instruction computes.
pub const M: usize = 64;
/// Depth of one `k16` instruction.
pub const K: usize = 16;
/// The widest `N` of a bf16 instruction; any multiple of 8 up to it issues.
pub const N_MAX: usize = 256;
/// Threads of a warpgroup.
pub const THREADS: usize = 128;

/// The `metadata.name` of a warpgroup product whose both operands are shared
/// descriptors.
pub const SS: &str = "wgmma.ss";
/// The `metadata.name` of one whose A operand is held in registers.
pub const RS: &str = "wgmma.rs";
/// The suffix of a chain left in flight: committed but not waited for, a
/// later [`wait`] being the caller's.
pub const ASYNC: &str = ".async";

/// The `metadata.name` of a product: its A operand's form, and whether the
/// statement waits for its chain.
pub fn name(from_registers: bool, waits: bool) -> String {
    format!("{}{}", if from_registers { RS } else { SS }, if waits { "" } else { ASYNC })
}

/// The swizzle a descriptor declares over its operand's shared rows: the
/// 16-byte chunk index of a row XORed with row bits, exactly as the TMA
/// modes and tk3's shared layout do it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Swizzle {
    None = 0,
    /// 128-byte rows: chunk `^= row & 7`.
    B128 = 1,
    /// 64-byte rows: chunk `^= (row >> 1) & 3`.
    B64 = 2,
    /// 32-byte rows: chunk `^= (row >> 2) & 1`.
    B32 = 3,
}

impl Swizzle {
    /// The mode of a K-major operand whose rows are `row_bytes` long, when
    /// one of the modes spans a whole row.
    pub fn for_row_bytes(row_bytes: usize) -> Option<Self> {
        match row_bytes {
            128 => Some(Self::B128),
            64 => Some(Self::B64),
            32 => Some(Self::B32),
            _ => None,
        }
    }
}

/// `wgmma.fence.sync.aligned`: orders the warpgroup's register writes before
/// the products that read them.
pub fn fence() -> Arc<UOp> {
    void_call("wgmma.fence.sync.aligned", "", "")
}

/// `wgmma.commit_group.sync.aligned`.
pub fn commit() -> Arc<UOp> {
    void_call("wgmma.commit_group.sync.aligned", "", "")
}

/// `wgmma.wait_group.sync.aligned N`.
pub fn wait(pending: u32) -> Arc<UOp> {
    void_call("wgmma.wait_group.sync.aligned", &format!("i64 {pending}"), "i64")
}

fn void_call(intrinsic: &str, args: &str, params: &str) -> Arc<UOp> {
    UOp::custom(
        smallvec![],
        format!("declare void @llvm.nvvm.{intrinsic}({params})\ncall void @llvm.nvvm.{intrinsic}({args})"),
        DType::Void,
    )
}

/// The 32-bit shared-window address of a shared pointer.
pub fn shared_address(ptr: &Arc<UOp>) -> Arc<UOp> {
    UOp::custom(
        smallvec![specific_ptr(ptr, AddrSpace::Local)],
        "ptrtoint ptr addrspace(3) {0} to i32".to_string(),
        DType::Int32,
    )
}

/// The matrix descriptor of a K-major shared operand at the 32-bit shared
/// `address`: its 16-byte units in bits 0–13, the leading byte offset in
/// 16–29, the stride byte offset (between groups of eight rows) in 32–45
/// and the swizzle mode in 62–63. The leading offset is unread for a
/// swizzled K-major operand whose instruction depth lies within one swizzle
/// span (a `k16` of 16-bit elements always does); it is declared as 16.
pub fn descriptor(address: &Arc<UOp>, stride_bytes: u32, swizzle: Swizzle) -> Arc<UOp> {
    let c64 = |v: i64| UOp::const_(DType::Int64, ConstValue::Int(v));
    let start = address.cast(DType::Int64).try_shr_op(&c64(4)).expect("shr").try_and_op(&c64(0x3fff)).expect("and");
    let fixed = (1i64 << 16) | ((stride_bytes as i64 >> 4) << 32) | ((swizzle as i64) << 62);
    start.try_or_op(&c64(fixed)).expect("or")
}

/// The `i64` descriptors an operand carries: itself when scalar, else its
/// elements.
fn split_descriptors(kernel: &mut Vec<String>, dst: &str, operand: &str, dtype: &DType, name: &str) -> Vec<String> {
    let lanes = dtype.vcount();
    if lanes == 1 {
        return vec![name.to_string()];
    }
    let natural = ldt(dtype);
    (0..lanes)
        .map(|i| {
            let value = format!("{dst}.{operand}{i}");
            kernel.push(format!("  {value} = extractelement {natural} {name}, i32 {i}"));
            value
        })
        .collect()
}

/// Render a warpgroup product: `metadata.name` is [`SS`] or [`RS`], its dims
/// `(n, 64, k)` with `k` a multiple of 16, `b` the `k / 16` B descriptors
/// as one `i64` vector, `a` likewise for A (`SS`) or the `k / 16` register
/// fragments of A (`RS`, eight 16-bit elements each), and `c` the `n / 2`
/// f32 accumulators of this thread. Returns `None` where no form exists.
pub fn render(
    uop: &Arc<UOp>,
    a: &Arc<UOp>,
    b: &Arc<UOp>,
    c: &Arc<UOp>,
    metadata: &WmmaMetadata,
    ctx: &mut RenderContext,
    kernel: &mut Vec<String>,
) -> Option<()> {
    let (n, m, k) = metadata.dims;
    let (form, waits) = match metadata.name.strip_suffix(ASYNC) {
        Some(form) => (form, false),
        None => (metadata.name.as_str(), true),
    };
    let from_registers = match form {
        SS => false,
        RS => true,
        name => {
            ctx.set_invalid_graph(format!("NVPTX renderer: no warpgroup product named {name}"));
            return None;
        }
    };
    let in_scalar = metadata.dtype_in.base();
    let valid = m == M && n % 8 == 0 && (8..=N_MAX).contains(&n) && k % K == 0 && k > 0;
    let ty = match in_scalar {
        ScalarDType::BFloat16 => "bf16",
        ScalarDType::Float16 => "f16",
        _ => "",
    };
    if !valid || ty.is_empty() || metadata.dtype_out.base() != ScalarDType::Float32 {
        ctx.set_invalid_graph(format!(
            "NVPTX renderer: no wgmma form for in={in_scalar:?} out={:?} dims=({n},{m},{k})",
            metadata.dtype_out.base()
        ));
        return None;
    }
    let dst = ctx.name(uop);
    let (a_name, b_name, c_name) = (ctx.get(a).to_string(), ctx.get(b).to_string(), ctx.get(c).to_string());
    let steps = k / K;
    let regs = n / 2;
    let acc = split_fragment(kernel, &dst, "c", &wmma_operand_dtype(c), &c_name, Wire::Word("float"));
    if acc.len() != regs {
        ctx.set_invalid_graph(format!("NVPTX renderer: wgmma n{n} holds {regs} accumulators, got {}", acc.len()));
        return None;
    }
    let descs_b = split_descriptors(kernel, &dst, "b", &wmma_operand_dtype(b), &b_name);
    let a_words = if from_registers {
        split_fragment(kernel, &dst, "a", &wmma_operand_dtype(a), &a_name, Wire::Word("i32"))
    } else {
        split_descriptors(kernel, &dst, "a", &wmma_operand_dtype(a), &a_name)
    };
    let a_per_step = if from_registers { 4 } else { 1 };
    if descs_b.len() != steps || a_words.len() != steps * a_per_step {
        ctx.set_invalid_graph(format!(
            "NVPTX renderer: wgmma k{k} takes {steps} B descriptors and {} A words, got {} and {}",
            steps * a_per_step,
            descs_b.len(),
            a_words.len()
        ));
        return None;
    }
    // Operand numbering: the accumulators out (0..regs), then per step its
    // A words and B descriptor, then the accumulators tied in.
    let d_list = (0..regs).map(|i| format!("${i}")).collect::<Vec<_>>().join(",");
    // The core reads shared memory through the async proxy: what the generic
    // proxy wrote there (a staged store, a landed `cp.async`) is fenced first.
    let mut text = String::from("fence.proxy.async.shared::cta;\\0Awgmma.fence.sync.aligned;\\0A");
    let mut operand = regs;
    for _ in 0..steps {
        let a_text = if from_registers {
            let words = (0..4).map(|i| format!("${}", operand + i)).collect::<Vec<_>>().join(",");
            operand += 4;
            format!("{{{words}}}")
        } else {
            operand += 1;
            format!("${}", operand - 1)
        };
        let b_text = format!("${operand}");
        operand += 1;
        // scale-d = 1 (accumulate), scale-a = scale-b = 1, no transposes (a
        // register A has no transpose flag: its last operand is B's).
        let tail = if from_registers { "p,1,1,0" } else { "p,1,1,0,0" };
        text.push_str(&format!(
            "{{.reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m{M}n{n}k{K}.f32.{ty}.{ty} {{{d_list}}},{a_text},{b_text},{tail};}}\\0A"
        ));
    }
    text.push_str("wgmma.commit_group.sync.aligned;");
    if waits {
        text.push_str("\\0Awgmma.wait_group.sync.aligned 0;");
    }
    let mut constraints: Vec<String> = vec!["=f".to_string(); regs];
    let mut args: Vec<String> = vec![];
    for step in 0..steps {
        for word in &a_words[step * a_per_step..(step + 1) * a_per_step] {
            constraints.push(if from_registers { "r" } else { "l" }.to_string());
            args.push(format!("{} {word}", if from_registers { "i32" } else { "i64" }));
        }
        constraints.push("l".to_string());
        args.push(format!("i64 {}", descs_b[step]));
    }
    for (i, value) in acc.iter().enumerate() {
        constraints.push(i.to_string());
        args.push(format!("float {value}"));
    }
    // A memory clobber: no shared access moves across the chain.
    constraints.push("~{memory}".to_string());
    let ret_ty = format!("{{ {} }}", vec!["float"; regs].join(", "));
    let call = format!("{dst}.wgmma");
    kernel.push(format!(
        "  {call} = call {ret_ty} asm sideeffect \"{text}\", \"{}\"({})",
        constraints.join(","),
        args.join(", ")
    ));
    let out = wmma_operand_dtype(uop);
    debug_assert_eq!(ldt(&out), format!("<{regs} x float>"));
    join_fragment(kernel, &dst, &call, &ret_ty, &out, Wire::Word("float"), regs);
    Some(())
}
