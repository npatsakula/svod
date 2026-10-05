//! SDMA (copy engine) ring bring-up for gfx942 (SDMA IP 4.4.2).
//!
//! SDMA registers are direct MMIO (not RLCG-gated). The ring's doorbell is
//! enabled (SDMA 4.4.2 advances via the BAR2 doorbell, whose aperture the GIM
//! routes on a VF); `submit` writes the byte-offset write pointer to both
//! `RB_WPTR` and the doorbell, and `RB_RPTR` reports progress in bytes.

use crate::error::Error;

use super::super::pci::Bar;
use super::super::regaccess::Regs;

type Result<T> = std::result::Result<T, Error>;

const RPTR_POLL: u32 = 2_000_000;

/// gfx942 SDMA engine-0 doorbell index (`AMDGPU_NAVI10_DOORBELL_sDMA_ENGINE0`);
/// per engine the stride is `0xA`.
pub const DOORBELL_SDMA_ENGINE0: u64 = 0x100;

/// `regSDMA_F32_CNTL` dword offset in the SDMA segment (sdma_4_4_2, seg 0).
const F32_CNTL_OFFSET: u32 = 0x2;
/// `SDMA_F32_CNTL.HALT` is bit 0.
const F32_CNTL_HALT: u32 = 1;

/// A single SDMA GFX ring on one SDMA instance.
pub struct SdmaRing {
    inst: u16,
    ring_va: u64,
    ring_size: u64,
    wptr_bytes: u64,
    /// BAR2 doorbell index (u64 units) for this ring.
    pub doorbell_index: u64,
}

impl SdmaRing {
    /// Bring up SDMA instance `inst` and program its GFX ring.
    /// `ring_va` is the GPU VA of the ring buffer (mapped through vmid 0);
    /// `rptr_va` is where the engine writes back the read pointer.
    pub fn setup(regs: &Regs, inst: u16, ring_va: u64, ring_size: u64, rptr_va: u64) -> Result<Self> {
        // Unhalt the F32 microengine. amdgpu clears SDMA_F32_CNTL.HALT (bit 0)
        // on VF resume without reloading microcode (GIM loaded it once and it
        // persists); the engine is left halted when amdgpu releases the VF.
        // SDMA_F32_CNTL is offset 0x2 / seg 0, absent from the vendored table.
        let f32 = regs.read_raw("sdma", inst, 0, F32_CNTL_OFFSET);
        regs.write_raw("sdma", inst, 0, F32_CNTL_OFFSET, f32 & !F32_CNTL_HALT);

        // Per-instance engine enable (trap + UTCL1) — SDMA 4.4 path.
        regs.write_fields("sdma", "regSDMA_CNTL", inst, &[("trap_enable", 1), ("utc_l1_enable", 1)])?;

        regs.write("sdma", "regSDMA_GFX_MINOR_PTR_UPDATE", inst, 1)?;
        regs.write("sdma", "regSDMA_GFX_RB_RPTR", inst, 0)?;
        regs.write("sdma", "regSDMA_GFX_RB_RPTR_HI", inst, 0)?;
        regs.write("sdma", "regSDMA_GFX_RB_WPTR", inst, 0)?;
        regs.write("sdma", "regSDMA_GFX_RB_WPTR_HI", inst, 0)?;
        regs.write("sdma", "regSDMA_GFX_RB_BASE", inst, (ring_va >> 8) as u32)?;
        regs.write("sdma", "regSDMA_GFX_RB_BASE_HI", inst, (ring_va >> 40) as u32)?;
        regs.write("sdma", "regSDMA_GFX_RB_RPTR_ADDR_LO", inst, rptr_va as u32)?;
        regs.write("sdma", "regSDMA_GFX_RB_RPTR_ADDR_HI", inst, (rptr_va >> 32) as u32)?;

        // Enable the doorbell (4.4.2 advances the ring via the BAR2 doorbell;
        // the aperture is already routed by the GIM on this VF).
        let doorbell_index = DOORBELL_SDMA_ENGINE0 + (inst as u64) * 0xA;
        regs.write_fields("sdma", "regSDMA_GFX_DOORBELL_OFFSET", inst, &[("offset", (doorbell_index * 2) as u32)])?;
        regs.write_fields("sdma", "regSDMA_GFX_DOORBELL", inst, &[("enable", 1)])?;
        regs.write("sdma", "regSDMA_GFX_MINOR_PTR_UPDATE", inst, 0)?;

        // rb_size encodes log2(ring size in dwords).
        let log2_dwords = 63 - (ring_size / 4).leading_zeros();
        regs.write_fields(
            "sdma",
            "regSDMA_GFX_RB_CNTL",
            inst,
            &[
                ("rb_vmid", 0),
                ("rptr_writeback_enable", 1),
                ("rptr_writeback_timer", 4),
                ("rb_priv", 1),
                ("rb_size", log2_dwords),
                ("rb_enable", 1),
            ],
        )?;
        regs.write_fields("sdma", "regSDMA_GFX_IB_CNTL", inst, &[("ib_enable", 1)])?;
        Ok(Self { inst, ring_va, ring_size, wptr_bytes: 0, doorbell_index })
    }

    pub fn ring_va(&self) -> u64 {
        self.ring_va
    }

    /// Advance the write pointer by `n_dwords` (the packets must already be in
    /// the ring) and kick the engine. Writes both the RB_WPTR register and the
    /// BAR2 doorbell (byte-offset wptr) — whichever the engine honors.
    pub fn submit(&mut self, regs: &Regs, doorbell: &Bar, n_dwords: u64) -> Result<()> {
        self.wptr_bytes += n_dwords * 4;
        debug_assert!(self.wptr_bytes <= self.ring_size, "M2 ring is single-shot; no wrap");
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        regs.write("sdma", "regSDMA_GFX_RB_WPTR", self.inst, self.wptr_bytes as u32)?;
        regs.write("sdma", "regSDMA_GFX_RB_WPTR_HI", self.inst, (self.wptr_bytes >> 32) as u32)?;
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        doorbell.write_u64(self.doorbell_index as usize, self.wptr_bytes);
        Ok(())
    }

    /// Poll RB_RPTR until it reaches the write pointer (engine drained).
    pub fn wait_idle(&self, regs: &Regs) -> Result<()> {
        for _ in 0..RPTR_POLL {
            if regs.read("sdma", "regSDMA_GFX_RB_RPTR", self.inst)? as u64 >= self.wptr_bytes {
                return Ok(());
            }
            std::hint::spin_loop();
        }
        Err(Error::Runtime {
            message: format!(
                "SDMA inst{} rptr {:#x} never reached wptr {:#x}",
                self.inst,
                regs.read("sdma", "regSDMA_GFX_RB_RPTR", self.inst).unwrap_or(0),
                self.wptr_bytes
            ),
        })
    }
}
