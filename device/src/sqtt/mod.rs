//! SQTT (SQ thread trace) packet decoder for RDNA3 (layout 3, gfx11) and RDNA4 (layout 4, gfx12).
//!
//! A port of tinygrad's `renderer/amd/sqtt.py` decoder: the stream is nibble-based, each packet is
//! identified by the low byte of a 64-bit shift register and carries a time delta that accumulates
//! into [`Packet::time`]. Field names follow tinygrad's packet classes.

mod decode;

use std::borrow::Cow;

pub use decode::{Decoder, decode};

/// The packet layout a trace announces in its first `LAYOUT_HEADER`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Layout {
    /// Layout 3: gfx11 (RDNA3, RDNA3.5).
    Rdna3,
    /// Layout 4: gfx12 (RDNA4).
    Rdna4,
}

/// One decoded packet: its kind and the accumulated time (in SQTT clock units) at which it ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Packet {
    pub time: u64,
    pub kind: Kind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Nop,
    /// `simd` is the SIMD selected for instruction tracing.
    LayoutHeader {
        layout: u8,
        simd: u8,
        group: u8,
        sel_a: u8,
        sel_b: u8,
        flag4: bool,
    },
    /// `cu` is `wgp | sa << 3` on RDNA3 and `wgp | sa << 4` on RDNA4.
    WaveStart {
        sa: u8,
        simd: u8,
        wgp: u8,
        cu: u8,
        wave: u8,
        id7: u16,
    },
    WaveEnd {
        sa: u8,
        simd: u8,
        wgp: u8,
        cu: u8,
        wave: u8,
    },
    WaveRdy {
        mask: u16,
    },
    WaveAlloc,
    /// An instruction of `wave` on the traced SIMD. `flag1`/`flag2` exist on RDNA3, `w64h` on RDNA4.
    Inst {
        wave: u8,
        op: InstOp,
        flag1: bool,
        flag2: bool,
        w64h: bool,
    },
    ValuInst {
        wave: u8,
        flag: bool,
    },
    Immediate {
        wave: u8,
    },
    /// One immediate instruction for each wave whose bit is set.
    ImmediateMask {
        mask: u16,
    },
    AluExec {
        src: AluSrc,
    },
    VmemExec {
        src: MemSrc,
    },
    /// `big` marks tinygrad's `EVENT_BIG` (21-bit event) against `EVENT` (13-bit).
    Event {
        event: u32,
        big: bool,
    },
    Reg {
        slot: u8,
        hi_byte: u8,
        subop: u16,
        val32: u32,
    },
    Snapshot {
        snap: u64,
    },
    UtilCtr {
        ctr: u64,
    },
    Perf {
        arg: u32,
    },
    TsWaveState {
        coarse: u8,
    },
    TsDeltaShort {
        delta: u64,
    },
    TsDeltaS5W2 {
        delta: u64,
    },
    TsDeltaS5W3 {
        delta: u64,
    },
    /// AMD's `NEW_PC` (a wave's PC after an indirect branch): 16 nibbles on RDNA3, 18 on RDNA4.
    TsDeltaS8W3 {
        delta: u64,
    },
    /// RDNA4 `s_ttracedata_imm` (AMD's `SHADER_DATA_SHORT`, 7-bit encoding `0b1000110`); tinygrad lacks it.
    /// `cu` as in [`Kind::WaveStart`].
    ShaderDataShort {
        sa: u8,
        simd: u8,
        wgp: u8,
        cu: u8,
        wave: u8,
        data: u8,
        private: bool,
    },
    /// `tl` exists on RDNA4 only. A marker (`rt && !pl`) adds nothing to the time.
    TsDeltaOrMark {
        delta: u64,
        pl: bool,
        rt: bool,
        tl: bool,
    },
}

impl Kind {
    pub fn is_marker(&self) -> bool {
        matches!(self, Self::TsDeltaOrMark { rt: true, pl: false, .. })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemSrc {
    Lds,
    LdsAlt,
    Vmem,
    VmemAlt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AluSrc {
    None,
    Salu,
    Valu,
    ValuSalu,
}

impl MemSrc {
    pub fn name(self) -> &'static str {
        ["LDS", "LDS_ALT", "VMEM", "VMEM_ALT"][self as usize]
    }
}

impl AluSrc {
    pub fn name(self) -> &'static str {
        ["NONE", "SALU", "VALU", "VALU_SALU"][self as usize]
    }
}

/// An `INST` packet's operation code, interpreted by the layout that produced it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InstOp {
    pub layout: Layout,
    pub code: u8,
}

const JUMP: u8 = 0x3;
const JUMP_NO: u8 = 0x4;

#[rustfmt::skip]
const RDNA3_OPS: &[(u8, &str)] = &[
    (0x00, "SALU"), (0x01, "SMEM_RD"), (JUMP, "JUMP"), (JUMP_NO, "JUMP_NO"), (0x05, "CALL"), (0x09, "MESSAGE"),
    (0x0b, "VALUT_4"), (0x0d, "VALUB_2"), (0x0e, "VALUB_4"), (0x0f, "VALUB_16"), (0x12, "VINTERP"), (0x13, "BARRIER"),
    (0x1c, "FLAT_RD_2"), (0x1d, "FLAT_WR_3"), (0x1e, "FLAT_WR_4"), (0x1f, "FLAT_WR_5"), (0x20, "FLAT_WR_6"),
    (0x21, "SGMEM_RD_1"), (0x22, "SGMEM_RD_2"), (0x24, "SGMEM_WR_2"), (0x25, "SGMEM_WR_3"), (0x26, "SGMEM_WR_4"),
    (0x27, "SGMEM_WR_5"), (0x28, "SGMEM_WR_6"), (0x29, "LDS_RD"), (0x2a, "LDS_WR_1"), (0x2b, "LDS_WR_2"),
    (0x2c, "LDS_WR_3"), (0x2d, "LDS_WR_4"), (0x2e, "LDS_WR_5"), (0x72, "SALU_WR_EXEC"), (0x73, "VALU1_WR_EXEC"),
];

#[rustfmt::skip]
const RDNA4_OPS: &[(u8, &str)] = &[
    (0x00, "SALU"), (0x01, "SMEM"), (0x02, "SMEM_WR"), (JUMP, "JUMP"), (JUMP_NO, "JUMP_NO"), (0x05, "CALL"),
    (0x07, "SALU_NO_EXEC"), (0x09, "MESSAGE"), (0x0a, "VALU_1"), (0x0b, "VALUT_4"), (0x0c, "VALUB_1"),
    (0x0d, "VALUB_2"), (0x0e, "VALUB_4"), (0x0f, "VALUB_16"), (0x12, "VINTERP"), (0x13, "BARRIER_WAIT"),
    (0x1c, "FLAT_RD_2"), (0x1d, "FLAT_WR_3"), (0x1e, "FLAT_WR_4"), (0x1f, "FLAT_WR_5"), (0x20, "FLAT_WR_6"),
    (0x21, "VMEM_RD_1"), (0x22, "VMEM_RD_2"), (0x23, "VMEM_WR_1"), (0x24, "VMEM_WR_2"), (0x25, "VMEM_WR_3"),
    (0x26, "VMEM_WR_4"), (0x27, "VMEM_WR_5"), (0x28, "VMEM_WR_6"), (0x29, "LDS_RD"), (0x2a, "LDS_WR_1"),
    (0x2b, "LDS_WR_2"), (0x2c, "LDS_WR_3"), (0x2d, "LDS_WR_4"), (0x2e, "LDS_WR_5"), (0x2f, "BUF_RD_1"),
    (0x30, "BUF_RD_2"), (0x31, "BUF_WR_1"), (0x32, "BUF_WR_2"), (0x33, "BUF_WR_3"), (0x34, "BUF_WR_4"),
    (0x35, "BUF_WR_5"), (0x36, "BUF_WR_6"), (0x6e, "LDS_DIR_LOAD"), (0x6f, "LDS_PARAM_LOAD"),
    (0x72, "SALU_WR_EXEC"), (0x73, "VALU1_WR_EXEC"), (0x74, "VALU_WR_EXEC_2"), (0x77, "OTHER_LDS_6"),
    (0x78, "OTHER_LDS_10"), (0x7a, "BARRIER_SIGNAL"), (0x87, "DYN_VGPR"), (0x8a, "BARRIER_JOIN"), (0x8c, "WMMA_8"),
    (0x8d, "WMMA_16"), (0x8e, "WMMA_32"), (0x8f, "WMMA_64"), (0x92, "VALU_DPFP"), (0x98, "SALU_FLOAT_3"),
    (0x99, "VALU_SCL_TRANS"), (0x9b, "SALU_2"), (0x9c, "SALU_5"),
];

/// Memory ops executed by another SIMD: `(category, first, last, base cycles)`, named
/// `OTHER_{category}_{code - first + base}`.
const RDNA3_OTHER: &[(&str, u8, u8, u8)] = &[("LDS", 0x50, 0x54, 1), ("FLAT", 0x55, 0x59, 2), ("VMEM", 0x5a, 0x66, 1)];
const RDNA4_OTHER: &[(&str, u8, u8, u8)] = &[("LDS", 0x50, 0x54, 1), ("FLAT", 0x55, 0x59, 2), ("VMEM", 0xbc, 0xdd, 1)];

impl InstOp {
    /// tinygrad's `InstOp`/`InstOpRDNA4` member name; `None` for a code neither enum knows.
    pub fn name(self) -> Option<Cow<'static, str>> {
        let (named, other) = match self.layout {
            Layout::Rdna3 => (RDNA3_OPS, RDNA3_OTHER),
            Layout::Rdna4 => (RDNA4_OPS, RDNA4_OTHER),
        };
        if let Some(&(_, name)) = named.iter().find(|(code, _)| *code == self.code) {
            return Some(Cow::Borrowed(name));
        }
        other.iter().find(|(_, lo, hi, _)| (*lo..=*hi).contains(&self.code)).map(|&(category, lo, _, base)| {
            Cow::Owned(format!("OTHER_{category}_{}", u16::from(self.code - lo) + u16::from(base)))
        })
    }

    /// A taken branch: the wave's next instruction is the branch target.
    pub fn is_jump(self) -> bool {
        self.code == JUMP
    }

    /// A branch not taken: the wave falls through.
    pub fn is_jump_no(self) -> bool {
        self.code == JUMP_NO
    }

    /// An op executed by a SIMD other than the traced one; it does not advance the traced wave's PC.
    pub fn is_other_simd(self) -> bool {
        self.name().is_some_and(|name| name.starts_with("OTHER_"))
    }
}
