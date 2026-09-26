//! PM4 packet builders for AMD GPU command processor.
//!
//! The PM4 opcodes/bitfields are AMD hardware constants — the SOC15 PM4
//! definitions for gfx9 (CDNA) and the Navi definitions for gfx10+ (RDNA) —
//! stable across kernel versions. Most are shared across gfx9/10/11/12, but the
//! RELEASE_MEM / ACQUIRE_MEM cache-flush encodings differ on gfx9 (the `*_gfx9`
//! builders and `EOP_*` / `COHER_*` constants below cover that split).
//!
//! `<< 0` for offset-zero bitfields is kept intentionally so every member of a
//! bitfield family reads uniformly (`x << OFFSET`) — clippy's `identity_op`
//! would obscure that pattern.
#![allow(clippy::identity_op)]
//!
//! Used by [`AmdComputeQueue`] to emit signal/barrier/wait packets through
//! the AQL vendor-specific indirect-buffer mechanism. The AQL kernel-dispatch
//! packet's HSA `completion_signal` is deliberately unused; PM4 `RELEASE_MEM`
//! inside a barriered AQL vendor IB advances the queue-owned timeline instead.

#![allow(dead_code)]

// ── PACKET3 header ────────────────────────────────────────────────────────

pub const PACKET_TYPE3: u32 = 3;

/// Build a PACKET3 header dword.
/// `n` is the number of data dwords following the header (i.e. total dwords - 1).
pub const fn packet3(op: u32, n: u32) -> u32 {
    (PACKET_TYPE3 << 30) | ((op & 0xFF) << 8) | ((n & 0x3FFF) << 16)
}

// ── PACKET3 opcodes (gfx10+) ──────────────────────────────────────────────

pub const PACKET3_NOP: u32 = 0x10;
pub const PACKET3_WAIT_REG_MEM: u32 = 0x3C;
pub const PACKET3_INDIRECT_BUFFER: u32 = 0x3F;
pub const PACKET3_RELEASE_MEM: u32 = 0x49;
pub const PACKET3_ACQUIRE_MEM: u32 = 0x58;
/// `PRED_EXEC` — conditionally execute the following dwords on a subset of
/// XCCs (gfx9 multi-XCC only). Used to fan a single signal write to one XCC.
pub const PACKET3_PRED_EXEC: u32 = 0x23;

/// `(1 << 23)` bit for the PACKET3_INDIRECT_BUFFER count dword, marking the
/// IB as valid for the CP to execute.
pub const INDIRECT_BUFFER_VALID: u32 = 1 << 23;
/// Number of dwords representable by the PACKET3_INDIRECT_BUFFER size field.
pub const INDIRECT_BUFFER_SIZE_MASK: u32 = (1 << 20) - 1;

// ── RELEASE_MEM bitfields (DW1) ───────────────────────────────────────────

/// VGT event type for "cache flush + invalidate timestamp".
pub const CACHE_FLUSH_AND_INV_TS_EVENT: u32 = 20;

/// `event_index = end_of_pipe` for `RELEASE_MEM`.
pub const EVENT_INDEX_END_OF_PIPE: u32 = 5;

pub const fn release_mem_event_type(x: u32) -> u32 {
    x << 0
}
pub const fn release_mem_event_index(x: u32) -> u32 {
    x << 8
}

pub const RELEASE_MEM_GCR_GLM_WB: u32 = 1 << 12;
pub const RELEASE_MEM_GCR_GLM_INV: u32 = 1 << 13;
pub const RELEASE_MEM_GCR_GLV_INV: u32 = 1 << 14;
pub const RELEASE_MEM_GCR_GL1_INV: u32 = 1 << 15;
pub const RELEASE_MEM_GCR_GL2_INV: u32 = 1 << 20;
pub const RELEASE_MEM_GCR_GL2_WB: u32 = 1 << 21;
pub const RELEASE_MEM_GCR_SEQ: u32 = 1 << 22;

/// All cache-flush flags ORed — uses the `cache_flush=True` path.
pub const RELEASE_MEM_CACHE_FLUSH_ALL: u32 = RELEASE_MEM_GCR_GLV_INV
    | RELEASE_MEM_GCR_GL1_INV
    | RELEASE_MEM_GCR_GL2_INV
    | RELEASE_MEM_GCR_GLM_WB
    | RELEASE_MEM_GCR_GLM_INV
    | RELEASE_MEM_GCR_GL2_WB
    | RELEASE_MEM_GCR_SEQ;

// ── RELEASE_MEM cache-flush bits, gfx9 / SOC15 (DW1) ──────────────────────
// gfx9 (CDNA) encodes the end-of-pipe cache flush via TC action bits, NOT the
// gfx10+ GCR bitfield above. Using the GCR bits on gfx9 leaves the
// CACHE_FLUSH_AND_INV_TS event waiting on a cache op that never completes, so
// the signal write at the end of the packet never lands.
pub const EOP_TC_WB_ACTION_EN: u32 = 1 << 15;
pub const EOP_TC_NC_ACTION_EN: u32 = 1 << 19;

/// gfx9 end-of-pipe cache flush: write-back + non-coherent invalidate.
pub const EOP_CACHE_FLUSH_GFX9: u32 = EOP_TC_WB_ACTION_EN | EOP_TC_NC_ACTION_EN;

// ── RELEASE_MEM bitfields (DW2) ───────────────────────────────────────────

pub const fn release_mem_dst_sel(x: u32) -> u32 {
    x << 16
}
pub const fn release_mem_int_sel(x: u32) -> u32 {
    x << 24
}
pub const fn release_mem_data_sel(x: u32) -> u32 {
    x << 29
}

/// `data_sel = send_32_bit_low` — write the low 32 bits of the value field
/// (DW5..6) to the target address. Used by the `signal()` path.
pub const DATA_SEL_SEND_32_BIT_LOW: u32 = 1;
/// `data_sel = send_64_bit_data`.
pub const DATA_SEL_SEND_64_BIT_DATA: u32 = 2;
/// `data_sel = send_gpu_clock_counter` — the CP writes the 64-bit GPU clock
/// counter (architected 100 MHz, 10 ns/tick) to the target address; the value
/// fields are ignored. Used to bracket a PM4 dispatch with start/end timestamps.
pub const DATA_SEL_SEND_GPU_CLOCK: u32 = 3;
/// `int_sel = send_interrupt_after_write_confirm`.
pub const INT_SEL_INTERRUPT_AFTER_WRITE: u32 = 2;
/// `dst_sel = 0` for ordinary memory destinations.
pub const DST_SEL_MEMORY: u32 = 0;

// ── WAIT_REG_MEM bitfields (DW1: info) ────────────────────────────────────

pub const fn wait_reg_mem_function(x: u32) -> u32 {
    x << 0
}
pub const fn wait_reg_mem_mem_space(x: u32) -> u32 {
    x << 4
}
pub const fn wait_reg_mem_operation(x: u32) -> u32 {
    x << 6
}
pub const fn wait_reg_mem_engine(x: u32) -> u32 {
    x << 8
}

/// `function = >= comparison`. Used for signal-value waits.
pub const WAIT_REG_MEM_FUNC_GEQ: u32 = 5;
/// `function = == comparison`. Used for the HDP flush handshake.
pub const WAIT_REG_MEM_FUNC_EQ: u32 = 3;

// ── HDP flush register handshake ──────────────────────────────────────────
//
// A register-space WAIT_REG_MEM against the BIF HDP flush register pair is
// emitted before `acquire_mem` in `memory_barrier`. Without this handshake,
// host writes through the BAR (kernarg arena) may not have left the HDP
// write buffers when the command processor consumes the next packet —
// manifests as a hung dispatch with the signal never firing.
//
// The REQ/DONE pair is one handshake channel per bit — CP0..CP9, SDMA0/1,
// then twenty reserved engines — shared by every agent on the GPU. amdgpu
// gives each of its rings a bit of its own (`amdgpu_gfx_get_hdp_flush_mask`:
// gfx `cp0 << pipe`, compute `cp2 << pipe`, MES cp8, KIQ cp9) and waits on
// that bit alone. Requesting and waiting on all 32 bits at once, as tinygrad
// does, races those kernel handshakes: once a concurrent MES/KIQ handshake
// retires its bit, DONE never reads all-ones again and the CP polls forever
// (seen on gfx1201 on roughly every other GigaAM load, rptr frozen on this
// packet). A user queue owns no pipe under MES, so svod takes a channel no
// kernel ring uses on any supported arch: reserved engine 0.
//
// The register pair address is **uniform across supported arches**:
// gfx9 and gfx10+ both define `NBIO_BASE_INST0_SEG2 = 0x0D20`, and every
// NBIO/NBIF dictionary defines the polled register at `(offset=262,
// segment=2)` — only the symbolic name (PF0 vs PF1) differs across NBIO
// versions (Strix Point's nbio_7_11_0 calls it `PF1_GPU_HDP_FLUSH_REQ`
// while older nbios call the same physical register `PF0_GPU_HDP_FLUSH_REQ`;
// this is purely a name disambiguation, not an address change).
pub const HDP_FLUSH_REQ_ADDR: u32 = 0xD20 + 262;
pub const HDP_FLUSH_DONE_ADDR: u32 = 0xD20 + 263;
/// `GPU_HDP_FLUSH_REQ/DONE.RSVD_ENG0`, laid out identically in nbio 7.9,
/// nbio 4.3 and nbif 6.3.1.
pub const HDP_FLUSH_CHANNEL: u32 = 1 << 12;

// ── PACKET3 opcodes & constants (PM4 dispatch path, single-XCC) ──────────
//
// These constants drive the raw-PM4 HCQ submission path used
// when `xccs == 1` (the gfx11/gfx12 default).

pub const PACKET3_DISPATCH_DIRECT: u32 = 0x15;
pub const PACKET3_SET_SH_REG: u32 = 0x76;
pub const PACKET3_SET_SH_REG_START: u32 = 0x2c00;
pub const PACKET3_EVENT_WRITE: u32 = 0x46;
pub const PACKET3_COPY_DATA: u32 = 0x40;
pub const PACKET3_SET_UCONFIG_REG: u32 = 0x79;
pub const PACKET3_SET_UCONFIG_REG_START: u32 = 0xc000;

// ── SH-relative COMPUTE_* register offsets ────────────────────────────────
//
// Each constant is `(GC_BASE_INST0_SEG0 + field_offset) - PACKET3_SET_SH_REG_START`.
// The pair is uniform across the supported arch matrix because gfx9 uses
// `GC_BASE=0x2000` with `field_offset=0xe00`, while gfx11/12 use
// `GC_BASE=0x1260` with `field_offset=0x1ba0` — both sums equal `0x2e00`,
// yielding `0x200` once `PACKET3_SET_SH_REG_START=0x2c00` is subtracted.

pub const COMPUTE_DISPATCH_INITIATOR: u32 = 0x200;
pub const COMPUTE_START_X: u32 = 0x204;
pub const COMPUTE_NUM_THREAD_X: u32 = 0x207;
pub const COMPUTE_PGM_LO: u32 = 0x20c;
pub const COMPUTE_DISPATCH_SCRATCH_BASE_LO: u32 = 0x210;
pub const COMPUTE_PGM_RSRC1: u32 = 0x212;
pub const COMPUTE_PGM_RSRC2: u32 = 0x213;
pub const COMPUTE_RESOURCE_LIMITS: u32 = 0x215;
pub const COMPUTE_TMPRING_SIZE: u32 = 0x218;
pub const COMPUTE_RESTART_X: u32 = 0x21b;
pub const COMPUTE_PGM_RSRC3: u32 = 0x228;
// gfx9 (CDNA) places `regCOMPUTE_PGM_RSRC3` at a different SH-register offset:
// vega `(3629 + GC_SEG0 0x2000) - SET_SH_REG_START 0x2c00 = 0x22d` vs navi
// `(7112 + 0x1260) - 0x2c00 = 0x228`. Every *other* COMPUTE_* reg lands
// identically across both arches, so only RSRC3 needs the split. Reached only on
// a single-XCC gfx9 PM4 dispatch; multi-XCC CDNA uses the AQL path.
pub const COMPUTE_PGM_RSRC3_GFX9: u32 = 0x22d;
pub const COMPUTE_USER_DATA_0: u32 = 0x240;

// ── COMPUTE_DISPATCH_INITIATOR field bits ─────────────────────────────────
// The `cs_w32_en` bit is only defined on gfx11/12; gfx9 (CDNA) ignores it.

pub const DISPATCH_INITIATOR_COMPUTE_SHADER_EN: u32 = 1 << 0;
pub const DISPATCH_INITIATOR_FORCE_START_AT_000: u32 = 1 << 2;
pub const DISPATCH_INITIATOR_CS_W32_EN: u32 = 1 << 15;

// ── Event constants ───────────────────────────────────────────────────────
// `EVENT_WRITE(CS_PARTIAL_FLUSH, EVENT_INDEX=4)` terminates a PM4 dispatch
// stream — flushes CS state so the next dispatch sees clean queue state.

pub const CS_PARTIAL_FLUSH: u32 = 7;
pub const EVENT_INDEX_PARTIAL_FLUSH: u32 = 4;

// ── Builders ──────────────────────────────────────────────────────────────

/// Build a PM4 RELEASE_MEM packet that writes `value` (low 32 bits) to
/// `addr`. With `cache_flush = true`, also flushes/invalidates all GPU
/// caches before the write — required so prior kernel stores are visible
/// at the signal site.
///
/// Layout (8 dwords total):
/// ```text
/// dw0  PACKET3(RELEASE_MEM, 6)   header (count = body_len - 1)
/// dw1  event_dw                   event type + index + cache flush bits
/// dw2  memsel_dw                  DATA_SEL | INT_SEL | DST_SEL
/// dw3  addr lo
/// dw4  addr hi
/// dw5  value lo
/// dw6  value hi
/// dw7  ctxid                      always 0 in our usage
/// ```
pub fn release_mem(addr: u64, value: u32, cache_flush: bool, is_gfx9: bool) -> [u32; 8] {
    release_mem_write(addr, value as u64, false, cache_flush, true, is_gfx9)
}

/// General RELEASE_MEM memory write used by HCQ stores. `interrupt` is kept
/// explicit because queue timeline stores are polled and need no KFD event.
pub fn release_mem_write(
    addr: u64,
    value: u64,
    write_64: bool,
    cache_flush: bool,
    interrupt: bool,
    is_gfx9: bool,
) -> [u32; 8] {
    // gfx9 (CDNA) and gfx10+ (RDNA) encode the cache flush differently: gfx9
    // uses the TC action bits in DW1, gfx10+ the GCR bitfield. DST_SEL only
    // exists on gfx10+ (it is implicitly memory on gfx9).
    let (cache, memsel_dw) = if is_gfx9 {
        let cache = if cache_flush { EOP_CACHE_FLUSH_GFX9 } else { 0 };
        let memsel = release_mem_data_sel(if write_64 { DATA_SEL_SEND_64_BIT_DATA } else { DATA_SEL_SEND_32_BIT_LOW })
            | if interrupt { release_mem_int_sel(INT_SEL_INTERRUPT_AFTER_WRITE) } else { 0 };
        (cache, memsel)
    } else {
        let cache = if cache_flush { RELEASE_MEM_CACHE_FLUSH_ALL } else { 0 };
        let memsel = release_mem_data_sel(if write_64 { DATA_SEL_SEND_64_BIT_DATA } else { DATA_SEL_SEND_32_BIT_LOW })
            | if interrupt { release_mem_int_sel(INT_SEL_INTERRUPT_AFTER_WRITE) } else { 0 }
            | release_mem_dst_sel(DST_SEL_MEMORY);
        (cache, memsel)
    };
    let event_dw =
        release_mem_event_type(CACHE_FLUSH_AND_INV_TS_EVENT) | release_mem_event_index(EVENT_INDEX_END_OF_PIPE) | cache;
    [
        packet3(PACKET3_RELEASE_MEM, 6),
        event_dw,
        memsel_dw,
        addr as u32,
        (addr >> 32) as u32,
        value as u32,
        (value >> 32) as u32,
        0, // ctxid
    ]
}

/// KFD queue-event mailbox write: store `event_id` at `addr`, raise the
/// completion interrupt, and carry the id in `ctxid` so KFD routes the
/// interrupt to that event. Mirrors the second RELEASE_MEM of Tinygrad's
/// `AMDComputeQueue.signal` (`ops_amd.py:392-393`). No cache flush — the
/// timeline value store that precedes it already carries one.
pub fn release_mem_event(addr: u64, event_id: u32, is_gfx9: bool) -> [u32; 8] {
    let mut packet = release_mem_write(
        addr,
        u64::from(event_id),
        /*write_64=*/ false,
        /*cache_flush=*/ false,
        true,
        is_gfx9,
    );
    packet[7] = event_id;
    packet
}

/// End-of-pipe ordering packet with no memory write, matching Tinygrad's first
/// timestamp RELEASE_MEM. The zero address is ignored when DATA_SEL is none.
pub fn release_mem_order(is_gfx9: bool) -> [u32; 8] {
    let memsel = release_mem_int_sel(INT_SEL_INTERRUPT_AFTER_WRITE)
        | if is_gfx9 { 0 } else { release_mem_dst_sel(DST_SEL_MEMORY) };
    [
        packet3(PACKET3_RELEASE_MEM, 6),
        release_mem_event_type(CACHE_FLUSH_AND_INV_TS_EVENT) | release_mem_event_index(EVENT_INDEX_END_OF_PIPE),
        memsel,
        0,
        0,
        0,
        0,
        0,
    ]
}

/// Build a PM4 RELEASE_MEM that writes the 64-bit GPU clock counter to `addr`
/// at end-of-pipe, with NO cache flush and NO interrupt — a lightweight
/// dispatch-timestamp probe. Bracketing a PM4 dispatch with two of these
/// (`start` before `DISPATCH_DIRECT`, `end` after) gives the host on-device
/// kernel duration on the single-XCC PM4 path, which — unlike the AQL path's
/// `ENABLE_PROFILING` — gets no CP-auto-stamped `start_ts`/`end_ts`. `addr` must
/// be 8-byte aligned (a signal slot's `start_ts`/`end_ts` fields, at +32/+40).
pub fn release_mem_timestamp(addr: u64, is_gfx9: bool) -> [u32; 8] {
    // GPU-clock data_sel writes 64 bits; no INT_SEL (no waiter to wake) and no
    // cache-flush bits (a bare timestamp probe, not the coherence-carrying
    // completion `release_mem`). DST_SEL exists only on gfx10+.
    let memsel = if is_gfx9 {
        release_mem_data_sel(DATA_SEL_SEND_GPU_CLOCK)
    } else {
        release_mem_data_sel(DATA_SEL_SEND_GPU_CLOCK) | release_mem_dst_sel(DST_SEL_MEMORY)
    };
    // `CACHE_FLUSH_AND_INV_TS_EVENT` names the timestamp-CAPABLE end-of-pipe
    // event; despite the name it triggers NO cache action here — the flush /
    // invalidate is gated by the GCR (gfx10+) / TC-action (gfx9) bits, which are
    // deliberately omitted (cf. the completion `release_mem`, which OR's them in).
    let event_dw =
        release_mem_event_type(CACHE_FLUSH_AND_INV_TS_EVENT) | release_mem_event_index(EVENT_INDEX_END_OF_PIPE);
    [
        packet3(PACKET3_RELEASE_MEM, 6),
        event_dw,
        memsel,
        addr as u32,
        (addr >> 32) as u32,
        0, // value lo — ignored for gpu-clock data_sel
        0, // value hi
        0, // ctxid
    ]
}

// ── ACQUIRE_MEM GCR_CNTL cache-flag bit positions (gfx10+) ────────────────
// Each constant is the `(x) << N` shift position; here we encode the
// `1`-set value directly.

pub const GCR_GLI_INV: u32 = 1 << 0;
pub const GCR_GL1_RANGE: u32 = 1 << 2; // not used; included for documentation
pub const GCR_GLM_WB: u32 = 1 << 4;
pub const GCR_GLM_INV: u32 = 1 << 5;
pub const GCR_GLK_WB: u32 = 1 << 6;
pub const GCR_GLK_INV: u32 = 1 << 7;
pub const GCR_GLV_INV: u32 = 1 << 8;
pub const GCR_GL1_INV: u32 = 1 << 9;
pub const GCR_GL2_INV: u32 = 1 << 14;
pub const GCR_GL2_WB: u32 = 1 << 15;

/// All cache levels invalidated/written-back. Equivalent to the
/// `acquire_mem()` default (`gli=glm=glk=glv=gl1=gl2=1`). Yields 0xC3F1.
pub const GCR_FLAGS_ALL: u32 = GCR_GLI_INV
    | GCR_GLM_WB
    | GCR_GLM_INV
    | GCR_GLK_WB
    | GCR_GLK_INV
    | GCR_GLV_INV
    | GCR_GL1_INV
    | GCR_GL2_INV
    | GCR_GL2_WB;

/// Mild flush: skips GLI and GL2 invalidate/writeback. Equivalent to the
/// pre-dispatch `acquire_mem(gli=0, gl2=0)`. Yields 0x03F0.
pub const GCR_FLAGS_NO_GLI_GL2: u32 = GCR_GLM_WB | GCR_GLM_INV | GCR_GLK_WB | GCR_GLK_INV | GCR_GLV_INV | GCR_GL1_INV;

/// Build a PM4 ACQUIRE_MEM packet (cache invalidate). `memory_barrier()`
/// issues this after a WAIT_REG_MEM on the HDP-flush register. `cache_flags`
/// selects which cache levels to invalidate/write-back — use
/// [`GCR_FLAGS_ALL`] for the `memory_barrier` site and
/// [`GCR_FLAGS_NO_GLI_GL2`] for the pre-dispatch site in `build_exec_pm4`
/// (`acquire_mem(gli=0, gl2=0)`).
///
/// Layout (8 dwords total):
/// ```text
/// dw0  PACKET3(ACQUIRE_MEM, 6)   header
/// dw1  0                          (CP_COHER_CNTL — unused on gfx10+)
/// dw2  size lo                    0xFFFFFFFF = full VA range
/// dw3  size hi                    0xFFFFFFFF
/// dw4  addr lo                    0
/// dw5  addr hi                    0
/// dw6  0                          poll interval (unused for ACQUIRE_MEM)
/// dw7  cache flags                GLI/GLM/GLK/GLV/GL1/GL2 inv + WBs
/// ```
pub fn acquire_mem_with(cache_flags: u32) -> [u32; 8] {
    [packet3(PACKET3_ACQUIRE_MEM, 6), 0, 0xFFFF_FFFF, 0xFFFF_FFFF, 0, 0, 0, cache_flags]
}

/// Shorthand for `acquire_mem_with(GCR_FLAGS_ALL)` — invalidates and writes
/// back every cache level. Used by `memory_barrier`.
pub fn acquire_mem() -> [u32; 8] {
    acquire_mem_with(GCR_FLAGS_ALL)
}

// ── gfx9 / SOC15 ACQUIRE_MEM (CP_COHER_CNTL, DW1) ─────────────────────────
// gfx9 has no GCR_CNTL dword; cache actions live in CP_COHER_CNTL (DW1) and
// the packet is one dword shorter than the gfx10+ form.
pub const COHER_SH_ICACHE_ACTION_ENA: u32 = 1 << 29;
pub const COHER_SH_KCACHE_ACTION_ENA: u32 = 1 << 27;
pub const COHER_TC_ACTION_ENA: u32 = 1 << 23;
pub const COHER_TCL1_ACTION_ENA: u32 = 1 << 22;
pub const COHER_TC_WB_ACTION_ENA: u32 = 1 << 18;

/// gfx9 ACQUIRE_MEM (full cache invalidate + write-back). 7 dwords:
/// `dw0` header, `dw1` CP_COHER_CNTL, `dw2..3` coher size (full VA),
/// `dw4..5` coher base (0), `dw6` poll interval (`0x0A`).
pub fn acquire_mem_gfx9() -> [u32; 7] {
    let cp_coher_cntl = COHER_SH_ICACHE_ACTION_ENA
        | COHER_SH_KCACHE_ACTION_ENA
        | COHER_TC_ACTION_ENA
        | COHER_TCL1_ACTION_ENA
        | COHER_TC_WB_ACTION_ENA;
    [packet3(PACKET3_ACQUIRE_MEM, 5), cp_coher_cntl, 0xFFFF_FFFF, 0xFFFF_FFFF, 0, 0, 0x0000_000A]
}

/// gfx9 ACQUIRE_MEM, **narrowed**: invalidate the per-CU caches (I-cache,
/// K-cache, vector TCL1) but **skip the L2 (TC) invalidate + write-back** —
/// the per-exec `acquire_mem(gli=0, gl2=0)` form.
///
/// Safe as a per-dispatch prologue because a PM4 queue has one XCC, whose L2
/// every dispatch on the device reads through; memory written behind L2's back
/// (host, SDMA) is acquired by a full `MemoryBarrier` (HDP flush, L2 invalidate)
/// before the dispatch that reads it — the full L2 invalidate here would be
/// pure redundant stall.
pub fn acquire_mem_gfx9_narrow() -> [u32; 7] {
    let cp_coher_cntl = COHER_SH_ICACHE_ACTION_ENA | COHER_SH_KCACHE_ACTION_ENA | COHER_TCL1_ACTION_ENA;
    [packet3(PACKET3_ACQUIRE_MEM, 5), cp_coher_cntl, 0xFFFF_FFFF, 0xFFFF_FFFF, 0, 0, 0x0000_000A]
}

/// PRED_EXEC: execute the following `exec_count` dwords only on the XCCs whose
/// bit is set in `xcc_mask`. gfx9 multi-XCC only, and only valid inside an IB
/// (the CP ignores PRED_EXEC issued straight into a ring). Used to emit the
/// end-of-pipe signal from a single XCC so the timeline is written exactly
/// once instead of once per XCC.
pub fn pred_exec(xcc_mask: u32, exec_count: u32) -> [u32; 2] {
    [packet3(PACKET3_PRED_EXEC, 0), (xcc_mask << 24) | (exec_count & 0x3FFF)]
}

/// Build a PM4 WAIT_REG_MEM packet that polls memory at `addr` until
/// `(*addr & mask) >= value`. Used by the AQL `wait()` to block on
/// a signal slot reaching the target value.
pub fn wait_reg_mem(addr: u64, value: u32, mask: u32) -> [u32; 7] {
    let info = wait_reg_mem_mem_space(1) // memory (not register)
        | wait_reg_mem_function(WAIT_REG_MEM_FUNC_GEQ)
        | wait_reg_mem_engine(0); // ME engine
    [
        packet3(PACKET3_WAIT_REG_MEM, 5),
        info,
        addr as u32,
        (addr >> 32) as u32,
        value,
        mask,
        4, // poll interval (4 dword units)
    ]
}

/// Build a register-space WAIT_REG_MEM packet that writes `value` to
/// `reg_req` and polls `reg_done` until `(reg_done & mask) == value`. This is
/// the "register handshake" variant: `mem_space = 0` (register),
/// `operation = 1` (write-then-poll), the same encoding and poll interval as
/// amdgpu's `gfx_v12_0_ring_emit_hdp_flush`.
///
/// Layout: 7 dwords (same shape as memory-space variant), but the two
/// address dwords carry `(reg_req, reg_done)` instead of `(addr_lo, addr_hi)`.
pub fn wait_reg_mem_register(reg_req: u32, reg_done: u32, value: u32, mask: u32) -> [u32; 7] {
    let info = wait_reg_mem_mem_space(0)            // 0 = register space
        | wait_reg_mem_operation(1)                  // 1 = REQ→DONE handshake
        | wait_reg_mem_function(WAIT_REG_MEM_FUNC_EQ)
        | wait_reg_mem_engine(0);
    [packet3(PACKET3_WAIT_REG_MEM, 5), info, reg_req, reg_done, value, mask, 0x20]
}

/// Build the HDP flush handshake packet for `memory_barrier`: request a
/// flush on svod's own [`HDP_FLUSH_CHANNEL`] and wait for that bit alone.
pub fn hdp_flush() -> [u32; 7] {
    wait_reg_mem_register(HDP_FLUSH_REQ_ADDR, HDP_FLUSH_DONE_ADDR, HDP_FLUSH_CHANNEL, HDP_FLUSH_CHANNEL)
}

/// Build a `PACKET3_SET_SH_REG`. `reg_offset` is the SH-relative offset
/// (e.g. [`COMPUTE_PGM_LO`]). Successive `values` are written to consecutive
/// registers, matching the `wreg(reg, *args)` semantics (the same `wreg`
/// call sets `PGM_LO` + `PGM_HI` in one packet, etc.).
///
/// Output layout (2 + N dwords):
/// ```text
/// dw0  PACKET3(SET_SH_REG, N)        header (count = data dwords)
/// dw1  reg_offset                    SH-relative, bits 0..15
/// dw2..  values...                   N consecutive register payloads
/// ```
pub fn set_sh_reg(reg_offset: u32, values: &[u32]) -> Vec<u32> {
    let n = values.len() as u32;
    let mut v = Vec::with_capacity(2 + values.len());
    v.push(packet3(PACKET3_SET_SH_REG, n));
    v.push(reg_offset & 0xFFFF);
    v.extend_from_slice(values);
    v
}

/// Build a `PACKET3_SET_UCONFIG_REG`. Same layout as [`set_sh_reg`] but for the
/// UCONFIG register window; `reg_offset` is UCONFIG-relative
/// (`abs_dword - PACKET3_SET_UCONFIG_REG_START`). Used for perf-counter control
/// registers (SQ_PERFCOUNTER*, CP_PERFMON_CNTL, GRBM_GFX_INDEX).
pub fn set_uconfig_reg(reg_offset: u32, values: &[u32]) -> Vec<u32> {
    let n = values.len() as u32;
    let mut v = Vec::with_capacity(2 + values.len());
    v.push(packet3(PACKET3_SET_UCONFIG_REG, n));
    v.push(reg_offset & 0xFFFF);
    v.extend_from_slice(values);
    v
}

/// Build a `PACKET3_COPY_DATA` that copies a performance-counter register into
/// GPU memory (`src_sel=perfcounter`, `dst_sel=tc_l2`). `src_reg` is the source
/// register's ABSOLUTE dword address; `dst` is the destination GPU VA. The final
/// dispatch `release_mem` cache flush makes the L2 write visible to the host.
///
/// Layout (6 dwords): header, control `(2<<8)|4`, src reg, src hi (0), dst lo, dst hi.
pub fn copy_data_reg_to_mem(src_reg: u32, dst: u64) -> [u32; 6] {
    [packet3(PACKET3_COPY_DATA, 4), (2 << 8) | 4, src_reg, 0, dst as u32, (dst >> 32) as u32]
}

/// Build a `PACKET3_DISPATCH_DIRECT` packet — 5 dwords total. The PM4 CP
/// reads `dim_x/y/z` and the `dispatch_initiator` value, then launches a
/// compute dispatch with the previously-set COMPUTE_* registers.
pub fn dispatch_direct(grid: [u32; 3], dispatch_initiator: u32) -> [u32; 5] {
    [packet3(PACKET3_DISPATCH_DIRECT, 3), grid[0], grid[1], grid[2], dispatch_initiator]
}

/// Build a `PACKET3_EVENT_WRITE` — 2 dwords. We use this to emit
/// `CS_PARTIAL_FLUSH` after a `DISPATCH_DIRECT` so back-to-back dispatches
/// see clean queue state.
pub fn event_write(event_type: u32, event_index: u32) -> [u32; 2] {
    [packet3(PACKET3_EVENT_WRITE, 0), (event_type & 0x3f) | ((event_index & 0xf) << 8)]
}
