//! Golden structural fingerprints of the production kernel builders — the committed
//! regression oracle. Because the LLVM render is non-deterministic
//! ([`crate::fingerprint`]), behavior preservation is checked on the build-time UOp
//! graph: a refactor that changes a kernel's graph changes its digest. Update an
//! `expected` const ONLY for an intentional graph change — the failure message
//! prints the new value to paste.
//!
//! One caveat: the digest is the SINK's `content_hash`, which folds each node's
//! op-data through `Hash`. Changing an op-data type's `Hash` impl (e.g. `AxisId`,
//! carried by every `RANGE`) therefore moves the digest with a byte-identical
//! graph. Such a re-baseline is proved by dumping both graphs and diffing them,
//! not by reading the new digest off the failure.

use std::sync::Arc;

use svod_dtype::{DType, DeviceSpec};
use svod_ir::UOp;

use crate::kernels::fa::{FaConfig, build_fa_mw_rdb};
use crate::kernels::gemm::{M1_CFG, build_matmul_cfg};
use crate::{ArchCaps, Kernel, kernel_fingerprint};
use svod_ir::ops;

fn matmul_sink() -> Arc<UOp> {
    let n = 512usize;
    let bufs = vec![
        UOp::new_buffer(DeviceSpec::Cpu, n * n, DType::Float32),
        UOp::new_buffer(DeviceSpec::Cpu, n * n, DType::BFloat16),
        UOp::new_buffer(DeviceSpec::Cpu, n * n, DType::BFloat16),
    ];
    let ker =
        Kernel::new("matmul_cfg", M1_CFG.grid_dims(n), M1_CFG.threads(crate::WARP_THREADS), bufs, ArchCaps::GFX942);
    build_matmul_cfg(&ker, n, M1_CFG);
    ker.finish(M1_CFG.n_accum)
}

/// FA dims shared by the golden builders. `o,q,k,v` are bf16; the masked variant
/// appends a 5th `[B]` i32 `key_lens` global.
const FA_DIMS: (usize, usize, usize, usize, usize) = (1, 2, 2, 64, 128); // (b, h, h_kv, d, n)

fn fa_bufs(masked: bool) -> Vec<Arc<UOp>> {
    let (b, h, h_kv, d, n) = FA_DIMS;
    let mut bufs = vec![
        UOp::new_buffer(DeviceSpec::Cpu, b * n * h * d, DType::BFloat16),
        UOp::new_buffer(DeviceSpec::Cpu, b * n * h * d, DType::BFloat16),
        UOp::new_buffer(DeviceSpec::Cpu, b * n * h_kv * d, DType::BFloat16),
        UOp::new_buffer(DeviceSpec::Cpu, b * n * h_kv * d, DType::BFloat16),
    ];
    if masked {
        bufs.push(UOp::new_buffer(DeviceSpec::Cpu, b, DType::Int32)); // key_lens [B], trailing
    }
    bufs
}

fn fa_sink_cfg(causal: bool, masked: bool) -> Arc<UOp> {
    let (b, h, h_kv, d, n) = FA_DIMS;
    let ker =
        Kernel::new("fa_mw_rdb", [h as i64, (n / 16 / 8) as i64, b as i64], 8 * 64, fa_bufs(masked), ArchCaps::GFX942);
    build_fa_mw_rdb(
        &ker,
        b,
        n,
        h,
        h_kv,
        d,
        FaConfig { q_blk: 16, kv_blk: 16, causal, ..Default::default() },
        DType::BFloat16,
        masked,
    );
    ker.finish(1)
}

fn fa_sink() -> Arc<UOp> {
    fa_sink_cfg(true, false)
}

// Committed structural golden digests. Update ONLY for an intentional graph change.
const MATMUL_DIGEST: u128 = 0x0678_fad7_5395_74af_0000_0000_0000_0000;
const MATMUL_NODES: usize = 536;
const FA_DIGEST: u128 = 0x8a20_2349_757e_64b0_0000_0000_0000_0000;
const FA_NODES: usize = 883;
// Non-causal and non-causal+key-masked build variants (pin the `causal:false` and
// `key_lens:Some` branches GPU-free). The FA all-masked-row NaN fix is a key_lens
// clamp at the kernel ENTRY (a tensor-graph op), so the SINK graph is unchanged.
// The FA digests last moved when the Q tile lost its f32 staging copy: the gather
// lands the 16-bit operand dtype straight in registers (the softmax scale already
// rides on the f32 `QKᵀ` accumulator), so each variant drops those 16 nodes.
const FA_NONCAUSAL_DIGEST: u128 = 0xd629_128d_5adc_ef74_0000_0000_0000_0000;
const FA_NONCAUSAL_NODES: usize = 857;
const FA_MASKED_DIGEST: u128 = 0xf841_8880_061b_92f5_0000_0000_0000_0000;
const FA_MASKED_NODES: usize = 881;

fn check(name: &str, sink: Arc<UOp>, digest: u128, nodes: usize) {
    let fp = kernel_fingerprint(&sink);
    assert_eq!(
        (fp.digest, fp.node_count),
        (digest, nodes),
        "{name} graph changed. If intentional, set the const to:\n  \
         DIGEST = 0x{:032x}; NODES = {};\nop_counts = {:#?}",
        fp.digest,
        fp.node_count,
        fp.op_counts
    );
}

#[test]
fn golden_matmul_cfg() {
    check("matmul_cfg", matmul_sink(), MATMUL_DIGEST, MATMUL_NODES);
}

#[test]
fn golden_fa_mw_rdb() {
    check("fa_mw_rdb", fa_sink(), FA_DIGEST, FA_NODES);
}

#[test]
fn golden_fa_mw_rdb_noncausal() {
    check("fa_mw_rdb[noncausal]", fa_sink_cfg(false, false), FA_NONCAUSAL_DIGEST, FA_NONCAUSAL_NODES);
}

#[test]
fn golden_fa_mw_rdb_masked() {
    check("fa_mw_rdb[noncausal,masked]", fa_sink_cfg(false, true), FA_MASKED_DIGEST, FA_MASKED_NODES);
}

/// The fingerprint is invariant to the global id counter: building the same kernel
/// twice in one process (fresh ids each time) yields the same digest.
#[test]
fn fingerprint_is_build_deterministic() {
    assert_eq!(kernel_fingerprint(&matmul_sink()).digest, kernel_fingerprint(&matmul_sink()).digest);
    assert_eq!(kernel_fingerprint(&fa_sink()).digest, kernel_fingerprint(&fa_sink()).digest);
}

/// Sorted, de-duped local/register `BUFFER` slots in a kernel graph.
fn local_slots_and_reg_ids(sink: &Arc<UOp>) -> (Vec<usize>, Vec<usize>) {
    let (mut locals, mut regs) = (Vec::new(), Vec::new());
    for u in sink.toposort() {
        match u.op() {
            svod_ir::Op::Buffer(ops::Buffer { arg, .. }) if arg.addrspace == Some(svod_ir::AddrSpace::Local) => {
                locals.push(arg.slot)
            }
            svod_ir::Op::Buffer(ops::Buffer { arg, .. }) if arg.addrspace == Some(svod_ir::AddrSpace::Reg) => {
                regs.push(arg.slot)
            }
            _ => {}
        }
    }
    for v in [&mut locals, &mut regs] {
        v.sort_unstable();
        v.dedup();
    }
    (locals, regs)
}

/// The per-kernel local/register `BUFFER` slots are deterministic across two
/// builds AND a dense `0..n` range — the contract the custom-kernel compile-dedup
/// relies on (structurally identical kernels mint identical LDS slot / register
/// slots → hash-cons to ONE compiled artifact; the `@local{slot}` LDS name is
/// stable). The fingerprint guards this only indirectly; this pins it directly.
#[test]
fn define_ids_are_deterministic_and_dense() {
    for build in [matmul_sink as fn() -> Arc<UOp>, fa_sink] {
        let (l1, r1) = local_slots_and_reg_ids(&build());
        let (l2, r2) = local_slots_and_reg_ids(&build());
        assert_eq!(l1, l2, "local BUFFER slots differ across two builds (dedup would break)");
        assert_eq!(r1, r2, "register BUFFER slots differ across two builds (dedup would break)");
        assert_eq!(l1, (0..l1.len()).collect::<Vec<_>>(), "local BUFFER slots must be dense 0..n, got {l1:?}");
        assert_eq!(r1, (0..r1.len()).collect::<Vec<_>>(), "register BUFFER slots must be dense 0..n, got {r1:?}");
    }
}
