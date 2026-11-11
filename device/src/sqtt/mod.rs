//! SQTT (SQ thread trace) analysis for RDNA3 (layout 3, gfx11) and RDNA4 (layout 4, gfx12).
//!
//! A port of AMD's MIT-licensed rocprof-trace-decoder (`source/gfx10`, `gfx11`, `gfx12`, `stitch`):
//! [`decode`] splits the nibble stream into packets with AMD's token tables, [`wave::analyze`] runs AMD's
//! per-wave state machine over them, [`stitch::Stitcher`] attributes each traced instruction to a PC of
//! a disassembly, and [`report::Report`] aggregates per-PC hit, latency, stall and idle cycles. Token and
//! field names follow AMD's; tinygrad's `renderer/amd/sqtt.py` adds the `ALU_EXEC`/`VMEM_EXEC` tokens AMD
//! leaves unknown. On top, [`asm::classify`] reads what a disassembly line does and [`waits::Waits`]
//! binds each executed wait to the instruction it waited for.

pub mod asm;
mod decode;
pub mod report;
pub mod stitch;
pub mod waits;
pub mod wave;

use std::borrow::Cow;

pub use decode::{Decoder, decode};

/// The packet layout a trace announces in its `HEADER`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Layout {
    /// Layout 3: gfx11 (RDNA3, RDNA3.5).
    Rdna3,
    /// Layout 4: gfx12 (RDNA4).
    Rdna4,
}

/// One decoded packet: the accumulated time (shader clock) at which it ends, its first 64 bits and kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Packet {
    pub time: u64,
    pub raw: u64,
    pub kind: Kind,
}

/// A wave slot: `wgp` on shader array `sa`, `simd`, slot `wave`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Unit {
    pub sa: u8,
    pub wgp: u8,
    pub simd: u8,
    pub wave: u8,
}

impl Unit {
    /// AMD's `get_sa_wgp`: the shader array at bit 7, the WGP below.
    pub fn sa_wgp(self) -> u8 {
        self.sa << 7 | self.wgp
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Header {
    /// The trace version: 3 on gfx11, 4 on gfx12.
    pub version: u8,
    /// The instruction-traced SIMD, WGP and shader array.
    pub dsimd: u8,
    pub dwgp: u8,
    pub dsa: u8,
    pub nsa: u8,
    pub nwgp: u8,
    /// Wave start mode; on gfx12 a `WAVE_START_EXT` is 40 bits when it is 3, else 48.
    pub wsm: u8,
    pub ucf: bool,
    pub dp_rate: u8,
    pub dp_derate: u8,
    /// `EXEC_POPCOUNT` tokens follow exec writes.
    pub exec: bool,
    pub dbl: bool,
    /// gfx12 `s_barrier_wait` tokens are excluded; the stitcher re-inserts them.
    pub exbarw: bool,
}

impl Header {
    /// The issue cycles of a double-precision VALU op (`dprate` in AMD's `rdna_sqtt.cpp`).
    pub fn dp_cycles(self) -> u32 {
        match self.dp_rate & if self.version < 3 { 0xf } else { 7 } {
            0 => 1,
            dpr => 1 << (dpr - 1),
        }
    }

    pub fn derate(self) -> u32 {
        1 << self.dp_derate
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WaveStart {
    pub unit: Unit,
    pub pipe: u8,
    pub me: u8,
    /// Wave launch count; 66..90 are context save and restore.
    pub count: u8,
    /// A gfx12 `WAVE_START_EXT` carrying a workgroup id.
    pub ext: bool,
    pub wgid: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reg {
    pub pipe: u8,
    pub me: u8,
    /// A compute (`CS`) register; otherwise user data or graphics.
    pub cs: bool,
    pub addr: u16,
    pub data: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShaderData {
    pub unit: Unit,
    pub data: u32,
    pub short: bool,
    pub private: bool,
    pub invalid: bool,
}

/// An exec-mask popcount of slot `wave`'s oldest pending SALU (`!valu`) or VALU exec write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Popcount {
    pub wave: u8,
    pub count: u8,
    pub valu: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Nop,
    Header(Header),
    /// `WAVE_START` and `WAVE_START_EXT`.
    WaveStart(WaveStart),
    WaveEnd(Unit),
    WaveAlloc,
    WaveReady {
        mask: u16,
    },
    /// An instruction of slot `wave` on the traced SIMD.
    Inst {
        wave: u8,
        op: InstOp,
        w64h: bool,
    },
    ValuInst {
        wave: u8,
        w64h: bool,
    },
    /// An immediate (`s_nop`, `s_waitcnt`, ...) of one wave.
    ImmOne {
        wave: u8,
    },
    /// One immediate for each wave whose bit is set.
    Immediate {
        mask: u16,
    },
    /// tinygrad's `ALUEXEC`; AMD leaves the encoding unknown.
    AluExec {
        src: AluSrc,
    },
    /// tinygrad's `VMEMEXEC`; AMD leaves the encoding unknown.
    VmemExec {
        src: MemSrc,
    },
    /// `EVENT` and, with `sync`, `EVENT_SYNC`.
    Event {
        id: u8,
        me: u8,
        pipe: u8,
        bop: bool,
        sync: bool,
    },
    Reg(Reg),
    RegInit {
        me: u8,
        pipe: u8,
        ty: u8,
        data: u32,
    },
    UtilCounter,
    ShaderData(ShaderData),
    Misc {
        fields: u8,
    },
    /// Adds `delta + 4`.
    Time {
        delta: u64,
    },
    /// The byte address a wave continues at after an indirect branch.
    NewPc {
        wave: u8,
        pc: u64,
    },
    /// Adds `delta` unless `rt` (a realtime sample); `pl` without `rt` reports lost packets.
    Timestamp {
        delta: u64,
        pl: bool,
        rt: bool,
        tl: bool,
    },
    ExecPopcount1(Popcount),
    ExecPopcount3([Popcount; 3]),
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

/// AMD's `WaveInstCategory`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Category {
    None,
    Smem,
    Salu,
    Vmem,
    Flat,
    Lds,
    Valu,
    /// A taken branch.
    Jump,
    /// A branch not taken.
    Next,
    Immed,
    /// A trap or a context save/restore.
    Trap,
    Msg,
    Bvh,
}

/// An `INST` packet's operation code (AMD's `EINST`), interpreted by the layout that produced it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InstOp {
    pub layout: Layout,
    pub code: u8,
}

/// What an op does: its category, its issue cycles and whether a SIMD other than the traced one runs it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpClass {
    pub category: Category,
    pub cycles: u32,
    pub other_simd: bool,
}

const DP: u8 = 0xfe;
const DP_DERATE: u8 = 0xff;

/// `(first code, last code, name, category, cycles)`; a range names its codes `{name}{cycles}`, its cycles
/// counting up from the first's. [`DP`]: the header's double-precision cycles, [`DP_DERATE`]: times the derate.
type Row = (u8, u8, &'static str, Category, u8);

#[rustfmt::skip]
const COMMON: &[Row] = &[
    (0, 0, "salu", Category::Salu, 1), (1, 1, "smem_rd", Category::Smem, 1), (2, 2, "smem_wr", Category::Smem, 1),
    (3, 3, "branch_taken", Category::Jump, 1), (4, 4, "branch_not_taken", Category::Next, 1),
    (5, 5, "jump", Category::Salu, 1), (6, 6, "trap", Category::Trap, 1), (7, 7, "salu_no_exec", Category::Salu, 1),
    (8, 8, "fatal_halt", Category::Trap, 1), (9, 9, "message", Category::Msg, 1), (11, 11, "valut_4", Category::Valu, 4),
    (12, 12, "valub_1", Category::Valu, 1), (13, 13, "valub_2", Category::Valu, 2), (14, 14, "valub_4", Category::Valu, 4),
    (18, 18, "vinterp", Category::Valu, 1), (20, 20, "expreq_gds", Category::None, 0),
    (21, 21, "expreq_gfx", Category::None, 0), (28, 28, "flat_rd_2", Category::Flat, 2),
    (29, 32, "flat_wr_", Category::Flat, 3), (33, 34, "sgmem_rd_", Category::Vmem, 1),
    (35, 40, "sgmem_wr_", Category::Vmem, 1), (41, 41, "lds_rd", Category::Lds, 1), (42, 46, "lds_wr_", Category::Lds, 1),
    (47, 48, "buf_rd_", Category::Vmem, 1), (49, 54, "buf_wr_", Category::Vmem, 1),
    (55, 66, "img_sample_", Category::Vmem, 1), (67, 67, "img_sample_reserved", Category::None, 0),
    (68, 71, "img_rd_", Category::Vmem, 1), (72, 78, "img_wr_", Category::Vmem, 2),
    (103, 103, "raytrace8", Category::Bvh, 8), (104, 104, "raytrace9", Category::Bvh, 9),
    (105, 105, "raytrace11", Category::Bvh, 11), (106, 106, "raytrace12", Category::Bvh, 12),
    (110, 110, "lds_dir_load", Category::Lds, 1), (111, 111, "lds_param_load", Category::Lds, 1),
    (114, 114, "salu_wr_exec", Category::Salu, 1), (115, 115, "valu1_wr_exec", Category::Valu, 1),
    (116, 116, "valu_b2_wr_exec", Category::Valu, 2), (117, 117, "rfe", Category::Salu, 1),
];

/// `gfx11wave.cpp`.
#[rustfmt::skip]
const GFX11: &[Row] = &[
    (10, 10, "valu_1", Category::Valu, DP), (15, 15, "valub_16", Category::Valu, DP),
    (16, 16, "valub_dfdp", Category::Valu, 1), (17, 17, "valub_dfdp_derate", Category::Valu, 1),
    (19, 19, "barrier", Category::Immed, 1), (108, 108, "wmma8", Category::Valu, 8), (109, 109, "wmma16", Category::Valu, 16),
    (112, 112, "subv_loop_begin", Category::Salu, 1), (113, 113, "subv_loop_end", Category::Salu, 1),
    (118, 118, "lds_bvh_6", Category::Lds, 6),
];

#[rustfmt::skip]
const GFX11_OTHER: &[Row] = &[
    (80, 84, "lds_other_simd_", Category::Lds, 1), (85, 89, "flat_other_simd_", Category::Flat, 2),
    (90, 101, "vmem_other_simd_", Category::Vmem, 1), (119, 119, "lds_other_simd_6", Category::Lds, 6),
];

/// `gfx12wave.cpp`.
#[rustfmt::skip]
const GFX12: &[Row] = &[
    (10, 10, "valu_1", Category::Valu, 1), (15, 15, "valub_16", Category::Valu, 16),
    (19, 19, "barrier_wait", Category::Immed, 1), (118, 118, "lds_bvh_6", Category::Bvh, 6),
    (121, 121, "lds_bvh_10", Category::Bvh, 10), (122, 122, "barrier_signal", Category::Msg, 1),
    (135, 135, "dyn_vgpr", Category::Salu, 1), (136, 136, "try_lock", Category::Salu, 1),
    (137, 137, "unlock", Category::Salu, 1), (138, 138, "barrier_join", Category::Msg, 1),
    (140, 140, "wmma8", Category::Valu, 8), (141, 141, "wmma16", Category::Valu, 16),
    (142, 142, "wmma32", Category::Valu, 32), (143, 143, "wmma64", Category::Valu, 64),
    (146, 146, "valu_dpfp", Category::Valu, DP), (147, 147, "valu_derate", Category::Valu, DP_DERATE),
    (150, 150, "icpref", Category::Smem, 1), (151, 151, "kcpref", Category::Smem, 1),
    (152, 152, "salu_float3", Category::Salu, 3), (153, 153, "valu_scl_trans", Category::Valu, 1),
    (155, 155, "salu2", Category::Salu, 2), (156, 156, "salu5", Category::Salu, 5),
    (222, 255, "block_store_", Category::Vmem, 1),
];

#[rustfmt::skip]
const GFX12_OTHER: &[Row] = &[
    (80, 84, "lds_other_simd_", Category::Lds, 1), (85, 89, "flat_other_simd_", Category::Flat, 2),
    (119, 119, "lds_other_simd_6", Category::Lds, 6), (120, 120, "lds_other_simd_10", Category::Lds, 10),
    (188, 221, "vmem_other_simd_", Category::Vmem, 1),
];

/// Codes both layouts share: an indirect jump (`s_setpc`), a trap and a trap return.
pub(crate) const EINST_JUMP: u8 = 5;
pub(crate) const EINST_TRAP: u8 = 6;
pub(crate) const EINST_RFE: u8 = 117;

impl InstOp {
    fn row(self) -> Option<(&'static Row, bool)> {
        let (own, other) = match self.layout {
            Layout::Rdna3 => (GFX11, GFX11_OTHER),
            Layout::Rdna4 => (GFX12, GFX12_OTHER),
        };
        let hit = |rows: &'static [Row]| rows.iter().find(|r| (r.0..=r.1).contains(&self.code));
        hit(own).or_else(|| hit(COMMON)).map(|r| (r, false)).or_else(|| hit(other).map(|r| (r, true)))
    }

    /// AMD's `EINST` name; `None` for a code AMD does not know.
    pub fn name(self) -> Option<Cow<'static, str>> {
        self.row().map(|((first, last, name, _, cycles), _)| match first == last {
            true => Cow::Borrowed(*name),
            false => Cow::Owned(format!("{name}{}", u32::from(self.code - first) + u32::from(*cycles))),
        })
    }

    /// AMD's `map_to_common_type` under a header's double-precision rate and derate.
    pub fn class(self, header: &Header) -> OpClass {
        let Some(((first, _, _, category, cycles), other_simd)) = self.row() else {
            return OpClass { category: Category::None, cycles: 0, other_simd: false };
        };
        let cycles = match *cycles {
            DP => header.dp_cycles(),
            DP_DERATE => header.dp_cycles() * header.derate(),
            base => u32::from(base) + u32::from(self.code - first),
        };
        OpClass { category: *category, cycles, other_simd }
    }

    /// A taken branch: the wave's next instruction is the branch target.
    pub fn is_jump(self) -> bool {
        self.code == 3
    }

    /// A branch not taken: the wave falls through.
    pub fn is_jump_no(self) -> bool {
        self.code == 4
    }

    /// An op executed by a SIMD other than the traced one; it does not advance the traced wave's PC.
    pub fn is_other_simd(self) -> bool {
        self.row().is_some_and(|(_, other)| other)
    }
}
