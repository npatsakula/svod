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

use crate::kernels::fa::{FaConfig, FaMask, build_fa_mw_rdb};
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

fn fa_bufs(mask: FaMask) -> Vec<Arc<UOp>> {
    let (b, h, h_kv, d, n) = FA_DIMS;
    let mut bufs = vec![
        UOp::new_buffer(DeviceSpec::Cpu, b * n * h * d, DType::BFloat16),
        UOp::new_buffer(DeviceSpec::Cpu, b * n * h * d, DType::BFloat16),
        UOp::new_buffer(DeviceSpec::Cpu, b * n * h_kv * d, DType::BFloat16),
        UOp::new_buffer(DeviceSpec::Cpu, b * n * h_kv * d, DType::BFloat16),
    ];
    if mask.key_lens {
        bufs.push(UOp::new_buffer(DeviceSpec::Cpu, b, DType::Int32)); // key_lens [B], trailing
    }
    if mask.seg_start {
        bufs.push(UOp::new_buffer(DeviceSpec::Cpu, b * n, DType::Int32)); // seg_start [B, N]
    }
    if mask.key_mask {
        bufs.push(UOp::new_buffer(DeviceSpec::Cpu, b * n, DType::Int32)); // key_mask [B, N], last
    }
    bufs
}

fn fa_sink_cfg(causal: bool, mask: FaMask) -> Arc<UOp> {
    fa_sink_windowed(causal, None, mask)
}

fn fa_sink_windowed(causal: bool, window: Option<(usize, usize)>, mask: FaMask) -> Arc<UOp> {
    let (b, h, h_kv, d, n) = FA_DIMS;
    let ker =
        Kernel::new("fa_mw_rdb", [h as i64, (n / 16 / 8) as i64, b as i64], 8 * 64, fa_bufs(mask), ArchCaps::GFX942);
    build_fa_mw_rdb(
        &ker,
        b,
        n,
        h,
        h_kv,
        d,
        FaConfig { q_blk: 16, kv_blk: 16, causal, window, ..Default::default() },
        DType::BFloat16,
        mask,
    );
    ker.finish(1)
}

fn fa_sink() -> Arc<UOp> {
    fa_sink_cfg(true, FaMask::NONE)
}

// Committed structural golden digests. Update ONLY for an intentional graph change.
//
// Every digest here last moved in PR #177 for two changes, one commit each,
// re-baselined WITHOUT a gfx942 run (these are gfx942 builds; the kernels were
// validated on gfx1151 and sm_86): the LOCAL→REG gather of a Strided operand
// fragment became `ept / group` unrolled vector reads (the matmul's node count
// doubles: no gather loops, constant register indices), and the register-staged
// K/V commit became one fenced store node handed to the gathers' WAR fence (the
// FA graphs lose the per-commit `After`s and their fences).
const MATMUL_DIGEST: u128 = 0xbd81_3d05_5b61_250e_0000_0000_0000_0000;
const MATMUL_NODES: usize = 1208;
const FA_DIGEST: u128 = 0x1ace_1f69_0db4_e959_0000_0000_0000_0000;
const FA_NODES: usize = 818;
// Every FA digest moved, 10 nodes more, when the softmax weights began narrowing
// to bf16 through the integer rounding bias on AMD (`Group::narrow_finite`).
// Every FA digest moved, node counts unchanged, when tk's f32 `exp2` became the bare
// `v_exp_f32` on AMD (`exp2_flush`: one CUSTOM in place of each EXP2).
// The FA digests moved again for the packed-row segment mask: the running max
// starts at the finite f32 floor instead of `-∞` (one constant node per graph).
// Non-causal and non-causal+key-masked build variants (pin the `causal:false` and
// `key_lens:Some` branches GPU-free). The FA all-masked-row NaN fix is a key_lens
// clamp at the kernel ENTRY (a tensor-graph op), so the SINK graph is unchanged.
// Before #177 the FA digests moved when the Q tile lost its f32 staging copy: the
// gather lands the 16-bit operand dtype straight in registers (the softmax scale
// already rides on the f32 `QKᵀ` accumulator), so each variant drops those 16 nodes.
const FA_NONCAUSAL_DIGEST: u128 = 0x7cb4_8621_0f71_d50c_0000_0000_0000_0000;
const FA_NONCAUSAL_NODES: usize = 792;
const FA_MASKED_DIGEST: u128 = 0xf9f1_a064_2f80_e35d_0000_0000_0000_0000;
const FA_MASKED_NODES: usize = 816;
// Causal + segment-masked (packed rows): the `seg_start:Some` branch, a per-row
// table read inside the score mask.
const FA_SEGMENTED_DIGEST: u128 = 0x4f29_9e69_ce5e_992b_0000_0000_0000_0000;
const FA_SEGMENTED_NODES: usize = 846;
// Sliding window (the band's KV block range per workgroup, the band mask and the
// empty-row norm floor) and the general `[B, N]` key mask (a per-key table read
// inside the score mask, and the same floor).
const FA_WINDOWED_DIGEST: u128 = 0xddbb_d6b9_f92d_aa3d_0000_0000_0000_0000;
const FA_WINDOWED_NODES: usize = 882;
const FA_KEY_MASKED_DIGEST: u128 = 0xe7dc_a488_18f9_e689_0000_0000_0000_0000;
const FA_KEY_MASKED_NODES: usize = 839;

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
    check("fa_mw_rdb[noncausal]", fa_sink_cfg(false, FaMask::NONE), FA_NONCAUSAL_DIGEST, FA_NONCAUSAL_NODES);
}

#[test]
fn golden_fa_mw_rdb_masked() {
    let mask = FaMask { key_lens: true, ..FaMask::NONE };
    check("fa_mw_rdb[noncausal,masked]", fa_sink_cfg(false, mask), FA_MASKED_DIGEST, FA_MASKED_NODES);
}

#[test]
fn golden_fa_mw_rdb_segmented() {
    let mask = FaMask { seg_start: true, ..FaMask::NONE };
    check("fa_mw_rdb[causal,segmented]", fa_sink_cfg(true, mask), FA_SEGMENTED_DIGEST, FA_SEGMENTED_NODES);
}

#[test]
fn golden_fa_mw_rdb_windowed() {
    let sink = fa_sink_windowed(false, Some((64, 64)), FaMask::NONE);
    check("fa_mw_rdb[noncausal,window]", sink, FA_WINDOWED_DIGEST, FA_WINDOWED_NODES);
}

#[test]
fn golden_fa_mw_rdb_key_masked() {
    let mask = FaMask { key_mask: true, ..FaMask::NONE };
    check("fa_mw_rdb[noncausal,key_mask]", fa_sink_cfg(false, mask), FA_KEY_MASKED_DIGEST, FA_KEY_MASKED_NODES);
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
