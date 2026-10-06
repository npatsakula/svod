---
sidebar_label: AM Driver
---

# The AM Driver (Userspace)

**AM** is an experimental, not-yet-selectable userspace driver under
`device/src/amd/am/` that talks to an AMD GPU's PCI BARs directly instead of
going through `amdgpu`/KFD. It follows tinygrad's AM driver in spirit, but
targets a different flavor of hardware (a virtual function rather than bare
metal). It is **scaffolding**: the pure-logic parts are implemented and
unit-tested, the bring-up was exercised once on a live GPU, and no kernel has
ever executed through it.

:::caution[Not selectable, not runnable]
`SVOD_AMD_BACKEND=am` is rejected (`unknown SVOD_AMD_BACKEND=am (only 'kfd'
supported)`, `device/src/amd/device.rs`): nothing in `am/` implements the
[`AmdIface`](./overview.md) seam, so `AmdDevice` cannot use it. The code is
reachable only through the standalone `device/examples/am_*.rs` programs. Its
last functional change is from June 2026; later commits are cleanups. The
status statements below describe what that commit exercised, not an ongoing
guarantee — there is no CI or hardware test for it.
:::

The module compiles on every Unix host (`cfg(unix)`, like the rest of the
backend), so it is always type-checked, linted, and its logic unit-tested
(about 40 tests under `device/src/test/unit/amd/am/`: page tables, TLSF,
memory manager, register tables, discovery parser).

---

## Target: a CDNA3 SR-IOV VF (gfx9.4.x)

`AmDev::open` accepts only an **SR-IOV virtual function** of a GPU whose GC IP
version is 9.4 (CDNA3) — the bring-up hardware was an MI300-class VF passed
into a KVM guest. A non-VF function or another GC version is rejected
(`device/src/amd/am/dev.rs`). The gfx11 page-table encoding remains implemented
and unit-tested, and its geometry and physical-range helpers are what the gfx9
path reuses.

Being a VF rather than bare metal shapes the whole driver:

- **GC MMIO is host-gated.** A direct read of a GC register returns
  `0xffffffff`; GC / GCVM registers go **indirectly through the RLC** (stage
  the value in RLC scratch, kick `RLC_SPARE_INT`, poll).
- **VRAM and IP discovery are gated until granted.** The host **GIM** (the
  SR-IOV host driver) must grant access through a **mailbox handshake**, which
  runs before discovery.
- **The host PF owns the privileged subsystems:** PSP, SMU, clocks,
  firmware / world-switch, and the **doorbell aperture routing**. AM programs
  the per-VF state (page-table context0, per-engine invalidation ranges, TLB
  flushes, ring/queue MQDs) and writes a few PF-owned registers (L2 cache
  config, system and identity apertures, `GB_ADDR_CONFIG`, `RLC_CNTL`,
  `SH_MEM_BASES`) best-effort, ignoring rejections.

tinygrad's AM is the inverse: bare-metal only, unbinding `amdgpu` and owning
the whole device. The VF flavor needs the mailbox, the RLCG indirect path and
per-VF-only hub programming, and never gets to own the engines.

---

## What exists

| Group | Module(s) | What it does | Status |
|---|---|---|---|
| Discovery | `pci.rs`, `discovery.rs` | sysfs BAR mmap (BAR0 VRAM / BAR2 doorbell / BAR5 MMIO), config-space r/w, bounds-checked IP-discovery parser (per-XCC segment bases, `gc_info` v1/v2) | ran on the VF; parser unit-tested |
| Register access | `regaccess.rs`, `rlcg.rs`, `mailbox.rs`, `regs.rs`, `regs_gen.rs` | the VF↔GIM mailbox handshake, RLCG indirect GC/GCVM r/w per XCC, the MMIO/RLCG router, vendored register tables with `select` / `find` / `encode` | ran on the VF (scratch echo, `GRBM_STATUS` on every XCC); table logic unit-tested |
| Memory (GMMU) | `mm/{tlsf,pagetable,manager,mod}.rs` | TLSF allocators for VA, physical VRAM and the page-table pool; 4-level / 48-bit walk (`va_shifts = [12, 21, 30, 39]`); gfx9 and gfx11 PTE/PDE encoding; huge pages; table reclaim; `valloc` / `vfree` | unit-tested; page tables written to VRAM over BAR0 and walked back by the CPU — no GPU translation through them is confirmed |
| GMC bring-up | `ip/gmc.rs` | both hubs' context0 (base/start/end + CNTL), MX_L1_TLB, per-engine invalidation ranges, ENG17 TLB flush, HDP flush, raw fault-status read | ran on the VF to context programming, with the flush ACK on every XCC |
| GFX bring-up | `ip/gfx.rs` | MEC enable (unchecked writes), v9 compute MQD, HQD activation, `WRITE_DATA` PM4 | `CP_HQD_ACTIVE` reads 1; the queue never consumed a packet |
| SDMA bring-up | `ip/sdma.rs` | unhalt the F32, program RB base/rptr/wptr + doorbell, submit, `wait_idle` | programmed; a copy never completed |
| Orchestrator | `dev.rs` | `AmDev::open` = mailbox → discovery → GMMU → GMC context0 → flush; `valloc`, `vram_read` / `vram_write`, `release` | ran on the VF through GMC |

Page tables are backed by an injectable `PhysMem` trait — a plain buffer in
tests, BAR-mapped VRAM (`VramPhys`) in the driver. The leaf encoding is the
only arch-specific part: gfx9 puts MTYPE at bit 57, sets `bfs` on PDB1 table
entries and translate-further on PDB0 table entries, and marks PDB1/PDB2
leaves `PDE_PTE`; gfx12 is `unimplemented!` (constants captured; a test
asserts the panic).

### Register tables are generated once, then vendored

tinygrad is a sometimes-absent submodule, so the build never depends on it.
`device/tools/gen_am_regs.py` is run manually: it parses tinygrad's
`autogen/am/regs.py` and emits the committed `am/regs_gen.rs`. At boot
`select` picks the greatest table version `≤ ip_ver` with the same major. The
committed tables cover the gfx9.4.3 set (`gc_9_4_3`, `mmhub_1_8_0`,
`osssys_4_4_2`, `sdma_4_4_2`, `nbio_7_9_0`, `hdp_4_4_2`, `mp_11_0_0`,
`mp_13_0_0`) and the gfx11.5.0 set (`gc_11_5_0`, `mmhub_3_3_0`, `mp_14_0_2`,
`nbio_7_11_0`, `hdp_6_0_0`, `osssys_6_0_0`); the gfx11 GC table is also what
the KFD path's hardware counters use (`amd/pmc.rs`).

---

## The examples

Each `device/examples/am_*.rs` program is a standalone bring-up oracle. What
the June 2026 run established:

| Example | What it does | Outcome |
|---|---|---|
| `am_discovery` | BAR map + IP discovery, read-only; coexists with a bound `amdgpu` | 8 GC 9.4.3 instances, SDMA and AIDs enumerated |
| `am_own` | mailbox grant + RLCG scratch echo + `GRBM_STATUS` on every XCC | asserts pass |
| `am_gmc` | GC + MM context0 programmed; ENG17 TLB-flush ACK on every XCC; fault status printed | ACKs on every XCC |
| `am_sdma` | SDMA ring setup + a copy through it | the engine does not consume the ring |
| `am_compute` | MEC enable + MQD activate + `WRITE_DATA`, kicked through both the doorbell and a direct `CP_HQD_PQ_WPTR` write | the HQD activates; the sentinel never lands |

The wall is the engine hand-off: the doorbell aperture routing and engine boot
are owned by the host PF. Enabling the aperture from the VF (the `_PF` BIF
doorbell registers) wedges the VF↔GIM mailbox and needs a VM reboot, so
`enable_doorbell_aperture` exists in `ip/gfx.rs` but is marked do-not-call on
the VF and commented out in `am_compute`.

---

## What does not exist

- **An `AmdIface` implementation** — so AM cannot be a device backend.
- **PSP firmware load**, **SMU / clocks** — GIM-owned on a VF; on bare metal
  they would be the largest and riskiest port.
- **An interrupt handler** — there is no `ip/ih.rs`; the OSSSYS table is used
  only by `am_discovery`. Bring-up polls.
- **Proof that a GPU engine executes work** through AM's page tables.

Two debug knobs exist for the GMC bring-up only: `SVOD_AM_DEBUG` (any value)
reads registers back after writing them and logs rejected GC writes, and
`SVOD_AM_MCBASE` (`raw`, `fb` or `fbxgmi`) overrides the MC aperture base.

---

## Why it is here

The motivation is the kernel scheduler: on single-XCC gfx11+ parts, aggressive
multi-queue dispatch can park CP micro-engines in waits the MES firmware cannot
preempt, which is why the [lane pool](./queues-and-dispatch.md) stays
conservative. Owning the GPU would take the kernel out of the dispatch path.
That argument does not yet apply to what AM supports — on a VF the host still
owns scheduling, world-switch and the doorbells — and reaching it on bare metal
means PSP, SMU, interrupts and the seam implementation on top of what exists.
The design value today is the seam itself: if an AM backend ever lands, it
implements the same five methods and three hooks as `KfdIface`, and nothing
above the seam changes.
