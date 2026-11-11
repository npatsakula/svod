//! GC register writes from a PM4 stream, shared by the PMC and SQTT profilers.
//!
//! A register resolves by name in the GC table vendored for the target's IP
//! version, and its absolute dword address is `base[segment] + offset` with the
//! GC segment bases of the target's SOC family. The write is a `SET_SH_REG` or
//! `SET_UCONFIG_REG` packet, chosen by the window the address falls in.

use svod_dtype::AmdArch;

use crate::amd::am::regs::{self, RegDef};
use crate::amd::sys::pm4;

/// `GC_BASE__INST0_SEG*` of the SOC15 parts (`vega_offsets`, gfx9).
const GC_BASES_SOC15: [u64; 2] = [0x2000, 0xA000];
/// `GC_BASE__INST0_SEG*` of the Navi parts (`navi_offsets`, gfx10-12). Checked
/// against Mesa: `regGRBM_GFX_INDEX` lands at byte 0x30800 (R_030800) and
/// `regSQ_PERFCOUNTER0_SELECT` at 0x36700 (R_036700).
const GC_BASES_NAVI: [u64; 4] = [0x1260, 0xA000, 0x1C000, 0x0240_2C00];

const SET_SH_START: u64 = pm4::PACKET3_SET_SH_REG_START as u64;
const SET_SH_END: u64 = 0x3000;
const SET_UCONFIG_START: u64 = pm4::PACKET3_SET_UCONFIG_REG_START as u64;

/// The GC IP version a target reports in IP discovery.
pub fn gc_ip_version(arch: AmdArch) -> (u8, u8, u8) {
    match arch {
        AmdArch::Gfx942 => (9, 4, 3),
        AmdArch::Gfx950 => (9, 5, 0),
        AmdArch::Gfx1100 => (11, 0, 0),
        AmdArch::Gfx1101 => (11, 0, 3),
        AmdArch::Gfx1102 => (11, 0, 2),
        AmdArch::Gfx1151 => (11, 5, 1),
        AmdArch::Gfx1200 => (12, 0, 0),
        AmdArch::Gfx1201 => (12, 0, 1),
    }
}

/// A GC register table plus the segment bases that place it.
#[derive(Clone, Copy, Debug)]
pub struct RegWriter {
    table: &'static [RegDef],
    bases: &'static [u64],
}

impl RegWriter {
    /// The writer for a GC IP version: its vendored table, or for a GC 11.0
    /// part, which has none, the 11.5 one. Their profiler registers (SQ
    /// perf counters, thread trace, SPI_CONFIG_CNTL, GRBM_GFX_INDEX, the static
    /// thread masks) sit at the same addresses with the same field offsets.
    pub fn gc(ip: (u8, u8, u8)) -> Option<Self> {
        let table = regs::select("gc", ip).or_else(|| regs::select("gc", (ip.0, u8::MAX, u8::MAX)))?;
        let bases: &'static [u64] = match ip.0 {
            9 => &GC_BASES_SOC15,
            10..=12 => &GC_BASES_NAVI,
            _ => return None,
        };
        Some(Self { table, bases })
    }

    pub fn for_arch(arch: AmdArch) -> Option<Self> {
        Self::gc(gc_ip_version(arch))
    }

    /// The named register. Panics when the table lacks it: the profilers only
    /// name registers every table they select carries.
    pub fn reg(&self, name: &str) -> &'static RegDef {
        regs::find(self.table, name).unwrap_or_else(|| panic!("GC register table missing {name}"))
    }

    /// Absolute dword address of the named register.
    pub fn addr(&self, name: &str) -> u32 {
        abs_addr(self.bases, self.reg(name))
    }

    /// Write `fields` (each `value << field.lo`) to the named register.
    pub fn wreg(&self, out: &mut Vec<u32>, name: &str, fields: &[(&str, u32)]) {
        let r = self.reg(name);
        self.wreg_value(out, name, r.encode(fields) as u32);
    }

    /// Write a raw value to the named register.
    pub fn wreg_value(&self, out: &mut Vec<u32>, name: &str, value: u32) {
        let a = abs_addr(self.bases, self.reg(name)) as u64;
        if (SET_SH_START..SET_SH_END).contains(&a) {
            out.extend(pm4::set_sh_reg((a - SET_SH_START) as u32, &[value]));
        } else {
            out.extend(pm4::set_uconfig_reg((a - SET_UCONFIG_START) as u32, &[value]));
        }
    }

    /// `GRBM_GFX_INDEX` broadcasting to every SE, SA and instance.
    pub fn grbm_broadcast(&self, out: &mut Vec<u32>) {
        self.wreg(
            out,
            "regGRBM_GFX_INDEX",
            &[("se_broadcast_writes", 1), ("sa_broadcast_writes", 1), ("instance_broadcast_writes", 1)],
        );
    }

    /// `GRBM_GFX_INDEX` indexed to one (se, sa, wgp); `instance_index` packs
    /// the WGP into bits [2+].
    pub fn grbm_index(&self, out: &mut Vec<u32>, se: u32, sa: u32, wgp: u32) {
        self.wreg(out, "regGRBM_GFX_INDEX", &[("se_index", se), ("sa_index", sa), ("instance_index", wgp << 2)]);
    }

    /// `GRBM_GFX_INDEX` selecting SE `se`, SA 0, all instances.
    pub fn grbm_se(&self, out: &mut Vec<u32>, se: u32) {
        self.wreg(out, "regGRBM_GFX_INDEX", &[("se_index", se), ("sa_index", 0), ("instance_broadcast_writes", 1)]);
    }
}

fn abs_addr(bases: &[u64], r: &RegDef) -> u32 {
    (bases[r.segment as usize] + r.offset as u64) as u32
}
