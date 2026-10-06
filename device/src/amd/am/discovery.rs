//! IP discovery: the firmware table at the top of VRAM that names every IP
//! block, its `(major, minor, revision)` and its register segment bases.
//!
//! Pure parsing over a byte slice (the 10 KiB table read from `vram_size −
//! 64 KiB`) so it is unit-testable without hardware; only the helpers that
//! locate the table touch MMIO. Port of `AMDev._run_discovery`.

use std::collections::BTreeMap;

use snafu::ensure;

use crate::error::{Error, RuntimeSnafu};

type Result<T> = std::result::Result<T, Error>;

pub const BINARY_SIGNATURE: u32 = 0x2821_1407;
pub const DISCOVERY_TABLE_SIGNATURE: u32 = 0x5344_5049;
/// Fixed dword index of `RCC_CONFIG_MEMSIZE` (readable without IP bases).
pub const MM_RCC_CONFIG_MEMSIZE: usize = 0xde3;
/// Table lives at `vram_size - 64 KiB`, 10 KiB long.
pub const TABLE_TAIL_OFFSET: u64 = 64 << 10;
pub const TABLE_SIZE: usize = 10 << 10;

// Hardware IP indices we use (subset of amdgpu's `enum amd_hw_ip_block_type`).
pub const GC_HWIP: u16 = 1;
pub const HDP_HWIP: u16 = 2;
pub const SDMA0_HWIP: u16 = 3;
pub const MMHUB_HWIP: u16 = 12;
pub const NBIO_HWIP: u16 = 14;
pub const MP0_HWIP: u16 = 15;
pub const MP1_HWIP: u16 = 16;
pub const OSSSYS_HWIP: u16 = 23;

/// `hwip → discovery hw_id` (amdgpu `hw_id_map`, subset we resolve).
const HW_ID_MAP: &[(u16, u16)] = &[
    (GC_HWIP, 11),
    (HDP_HWIP, 41),
    (SDMA0_HWIP, 42),
    (4, 43), // SDMA1
    (5, 68), // SDMA2
    (6, 69), // SDMA3
    (MMHUB_HWIP, 34),
    (13, 35), // ATHUB
    (NBIO_HWIP, 108),
    (MP0_HWIP, 255),
    (MP1_HWIP, 1),
    (OSSSYS_HWIP, 40),
];

/// GC properties (subset shared by `gc_info` v1/v2 that we need; field order
/// differs by major version).
#[derive(Clone, Copy, Debug, Default)]
pub struct GcInfo {
    pub num_se: u32,
    pub num_cu_per_sh: u32,
    pub num_sh_per_se: u32,
    pub wave_size: u32,
    pub max_waves_per_simd: u32,
    pub max_scratch_slots_per_cu: u32,
    pub lds_size_kb: u32,
}

#[derive(Debug, Default)]
pub struct Discovery {
    pub vram_size: u64,
    pub reserved_vram_size: u64,
    /// `hwip → version` for the blocks in `HW_ID_MAP`.
    pub ip_ver: BTreeMap<u16, (u8, u8, u8)>,
    /// `hwip → instance → register segment bases`.
    pub regs_offset: BTreeMap<u16, BTreeMap<u16, Vec<u64>>>,
    pub gc: GcInfo,
}

impl Discovery {
    /// Number of GC instances = XCC count (8 on CDNA3 multi-XCC, 1 on single-XCC parts).
    pub fn xccs(&self) -> usize {
        self.regs_offset.get(&GC_HWIP).map_or(1, BTreeMap::len)
    }

    pub fn vmhubs(&self) -> usize {
        self.regs_offset.get(&MMHUB_HWIP).map_or(1, BTreeMap::len)
    }

    /// Parse the 10 KiB discovery blob. `vram_size` from `RCC_CONFIG_MEMSIZE`.
    pub fn parse(tbl: &[u8], vram_size: u64) -> Result<Self> {
        ensure!(tbl.len() >= TABLE_SIZE, RuntimeSnafu { message: "discovery blob too short".to_string() });
        let sig = rd32(tbl, 0)?;
        ensure!(sig == BINARY_SIGNATURE, RuntimeSnafu { message: format!("discovery binary signature {sig:#x}") });

        // binary_header.table_list[6] of (offset, checksum, size, pad) u16s at +12.
        let table_off = |idx: usize| -> Result<usize> { Ok(rd16(tbl, 12 + idx * 8)? as usize) };
        let ihdr = table_off(0)?; // IP_DISCOVERY
        let ihdr_sig = rd32(tbl, ihdr)?;
        ensure!(
            ihdr_sig == DISCOVERY_TABLE_SIGNATURE,
            RuntimeSnafu { message: format!("ip table signature {ihdr_sig:#x}") }
        );
        let num_dies = rd16(tbl, ihdr + 12)? as usize;
        let base_64bit = rd16(tbl, ihdr + 12 + 2 + 16 * 4)? & 1 != 0; // post die_info[16]

        let mut d = Self { vram_size, ..Default::default() };
        for die in 0..num_dies {
            let die_off = rd16(tbl, ihdr + 14 + die * 4 + 2)? as usize;
            let num_ips = rd16(tbl, die_off + 2)? as usize;
            let mut ip = die_off + 4;
            for _ in 0..num_ips {
                // ip_v4 (packed): hw_id u16, instance u8, num_base u8, maj/min/rev u8.
                let (hw_id, inst) = (rd16(tbl, ip)?, byte(tbl, ip + 2)? as u16);
                let nbase = byte(tbl, ip + 3)? as usize;
                let ver = (byte(tbl, ip + 4)?, byte(tbl, ip + 5)?, byte(tbl, ip + 6)?);
                let bases: Vec<u64> =
                    (0..nbase)
                        .map(|i| {
                            if base_64bit {
                                rd64(tbl, ip + 8 + i * 8)
                            } else {
                                rd32(tbl, ip + 8 + i * 4).map(u64::from)
                            }
                        })
                        .collect::<Result<_>>()?;
                for &(hwip, id) in HW_ID_MAP {
                    if id == hw_id {
                        d.regs_offset.entry(hwip).or_default().insert(inst, bases.clone());
                        d.ip_ver.insert(hwip, ver);
                    }
                }
                ip += 8 + nbase * if base_64bit { 8 } else { 4 };
            }
        }

        let gc_ver = *d.ip_ver.get(&GC_HWIP).unwrap_or(&(0, 0, 0));
        d.reserved_vram_size = if matches!((gc_ver.0, gc_ver.1), (9, 4) | (9, 5)) { 384 << 20 } else { 64 << 20 };

        // gc_info: gpu_info_header (table_id u32, ver u16/u16, size u32) + fields.
        let gc = table_off(1)?;
        let ver_major = rd16(tbl, gc + 4)?;
        let f = |i: usize| -> Result<u32> { rd32(tbl, gc + 12 + i * 4) };
        d.gc = match ver_major {
            1 => GcInfo {
                num_se: f(0)?,
                // v1: wgp0+wgp1 per SA, sa_per_se at field 16.
                num_cu_per_sh: 2 * (f(1)? + f(2)?),
                num_sh_per_se: f(16)?,
                wave_size: f(11)?,
                max_waves_per_simd: f(12)?,
                max_scratch_slots_per_cu: f(13)?,
                lds_size_kb: f(14)?,
            },
            2 => GcInfo {
                num_se: f(0)?,
                num_cu_per_sh: f(1)?,
                num_sh_per_se: f(2)?,
                wave_size: f(11)?,
                max_waves_per_simd: f(12)?,
                max_scratch_slots_per_cu: f(13)?,
                lds_size_kb: f(14)?,
            },
            v => return Err(Error::Runtime { message: format!("unsupported gc_info v{v}") }),
        };
        Ok(d)
    }
}

/// Bounds-checked slice: returns `Err` instead of panicking when the table is
/// short/garbage (e.g. a partial read before the GIM fully ungates the FB).
#[inline]
fn chunk(b: &[u8], off: usize, n: usize) -> Result<&[u8]> {
    b.get(off..off + n).ok_or_else(|| Error::Runtime {
        message: format!("discovery: {n}B read at {off:#x} past table end {}", b.len()),
    })
}
#[inline]
fn byte(b: &[u8], off: usize) -> Result<u8> {
    b.get(off)
        .copied()
        .ok_or_else(|| Error::Runtime { message: format!("discovery: byte at {off:#x} past table end {}", b.len()) })
}
#[inline]
fn rd16(b: &[u8], off: usize) -> Result<u16> {
    Ok(u16::from_le_bytes(chunk(b, off, 2)?.try_into().unwrap()))
}
#[inline]
fn rd32(b: &[u8], off: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(chunk(b, off, 4)?.try_into().unwrap()))
}
#[inline]
fn rd64(b: &[u8], off: usize) -> Result<u64> {
    Ok(u64::from_le_bytes(chunk(b, off, 8)?.try_into().unwrap()))
}
