use std::iter::FusedIterator;

use super::{AluSrc, InstOp, Kind, Layout, MemSrc, Packet};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Code {
    ValuInst,
    VmemExec,
    AluExec,
    Immediate,
    ImmediateMask,
    WaveRdy,
    TsDeltaS8W3,
    WaveEnd,
    WaveStart,
    TsDeltaS5W2,
    WaveAlloc,
    TsDeltaS5W3,
    Perf,
    UtilCtr,
    TsDeltaShort,
    Nop,
    TsWaveState,
    Event,
    EventBig,
    Reg,
    Snapshot,
    TsDeltaOrMark,
    LayoutHeader,
    Inst,
}

/// A packet type: the low `width` bits of the register equal `enc`; the packet spans `nibbles` and its
/// delta field is `delta = (lo, width)`, both indexed by layout (RDNA3, RDNA4).
struct Spec {
    code: Code,
    enc: u8,
    width: u8,
    nibbles: [u8; 2],
    delta: [(u8, u8); 2],
}

const fn spec(code: Code, enc: u8, width: u8, nibbles: [u8; 2], delta: [(u8, u8); 2]) -> Spec {
    Spec { code, enc, width, nibbles, delta }
}

/// In rocprof type-ID order, which breaks ties in the opcode table.
#[rustfmt::skip]
const SPECS: [Spec; 24] = [
    spec(Code::ValuInst, 0b011, 3, [3, 3], [(3, 3), (3, 3)]),
    spec(Code::VmemExec, 0b1111, 4, [2, 2], [(4, 2), (4, 2)]),
    spec(Code::AluExec, 0b1110, 4, [2, 2], [(4, 2), (4, 2)]),
    spec(Code::Immediate, 0b1101, 4, [3, 3], [(4, 3), (4, 3)]),
    spec(Code::ImmediateMask, 0b00100, 5, [6, 6], [(5, 3), (5, 3)]),
    spec(Code::WaveRdy, 0b10100, 5, [6, 6], [(5, 3), (5, 3)]),
    spec(Code::TsDeltaS8W3, 0b0100001, 7, [18, 18], [(8, 3), (8, 3)]),
    spec(Code::WaveEnd, 0b10101, 5, [5, 5], [(5, 3), (5, 3)]),
    spec(Code::WaveStart, 0b01100, 5, [8, 8], [(5, 2), (5, 2)]),
    spec(Code::TsDeltaS5W2, 0b11100, 5, [12, 10], [(5, 2), (5, 2)]),
    spec(Code::WaveAlloc, 0b00101, 5, [5, 6], [(5, 3), (5, 3)]),
    spec(Code::TsDeltaS5W3, 0b00110, 5, [13, 14], [(5, 3), (7, 3)]),
    spec(Code::Perf, 0b10110, 5, [7, 8], [(5, 3), (7, 3)]),
    spec(Code::UtilCtr, 0b0110001, 7, [12, 12], [(7, 2), (7, 2)]),
    spec(Code::TsDeltaShort, 0b1000, 4, [2, 2], [(4, 4), (4, 4)]),
    spec(Code::Nop, 0b0000, 4, [1, 1], [(0, 0), (0, 0)]),
    spec(Code::TsWaveState, 0b1010001, 7, [6, 6], [(7, 9), (7, 9)]),
    spec(Code::Event, 0b01100001, 8, [6, 6], [(8, 3), (8, 3)]),
    spec(Code::EventBig, 0b11100001, 8, [8, 8], [(8, 3), (8, 3)]),
    spec(Code::Reg, 0b1001, 4, [16, 16], [(4, 3), (4, 3)]),
    spec(Code::Snapshot, 0b1110001, 7, [16, 16], [(7, 3), (7, 3)]),
    spec(Code::TsDeltaOrMark, 0b0000001, 7, [12, 16], [(12, 36), (12, 52)]),
    spec(Code::LayoutHeader, 0b0010001, 7, [16, 16], [(0, 0), (0, 0)]),
    spec(Code::Inst, 0b010, 3, [5, 5], [(4, 3), (3, 3)]),
];

const NOP: u8 = 15;

/// Low register byte -> index into [`SPECS`]: the widest matching encoding, the first on a tie, NOP when
/// none matches. Encodings are the same in both layouts.
const OPCODES: [u8; 256] = {
    let mut table = [NOP; 256];
    let mut byte = 0;
    while byte < 256 {
        let (mut i, mut best) = (0, 0);
        while i < SPECS.len() {
            let s = &SPECS[i];
            if s.width > best && (byte as u8) & ((1u16 << s.width) - 1) as u8 == s.enc {
                table[byte] = i as u8;
                best = s.width;
            }
            i += 1;
        }
        byte += 1;
    }
    table
};

fn bits(reg: u64, hi: u32, lo: u32) -> u64 {
    (reg >> lo) & (u64::MAX >> (63 - (hi - lo)))
}

fn flag(reg: u64, bit: u32) -> bool {
    bits(reg, bit, bit) != 0
}

/// Decodes a raw SQTT blob. Starts in layout 3 and follows the `LAYOUT_HEADER`s it meets.
pub fn decode(data: &[u8]) -> Decoder<'_> {
    Decoder { data, pos: 0, reg: 0, nib_off: 0, nib_count: 16, time: 0, layout: Layout::Rdna3, unsupported: None }
}

/// The packet iterator over one SQTT blob.
///
/// The register is a 64-bit window over the nibble stream; each step shifts out the previous packet's
/// nibbles. A packet wider than the window (`TS_DELTA_S8_W3`, 18 nibbles) shifts at most eight bytes in
/// and skips the rest, as tinygrad does.
pub struct Decoder<'a> {
    data: &'a [u8],
    pos: usize,
    reg: u64,
    nib_off: usize,
    nib_count: usize,
    time: u64,
    layout: Layout,
    unsupported: Option<u8>,
}

impl Decoder<'_> {
    /// The layout the packets that follow are decoded with.
    pub fn layout(&self) -> Layout {
        self.layout
    }

    /// The layout of a `LAYOUT_HEADER` that is neither 3 nor 4 (CDNA, or not a header at all); decoding
    /// stops there, where tinygrad switches to its CDNA tables.
    pub fn unsupported_layout(&self) -> Option<u8> {
        self.unsupported
    }

    fn shift_in(&mut self) {
        let data = self.data;
        let need = self.nib_count - self.nib_off;
        if self.nib_off == 1 {
            self.reg = (self.reg >> 4) | (u64::from(data[self.pos] >> 4) << 60);
            self.pos += 1;
        }
        let byte_count = need >> 1;
        if byte_count > 0 {
            let read = byte_count.min(8);
            let mut chunk = [0u8; 8];
            chunk[..read].copy_from_slice(&data[self.pos..self.pos + read]);
            let chunk = u64::from_le_bytes(chunk);
            self.reg = if read == 8 { chunk } else { (self.reg >> (read * 8)) | (chunk << (64 - read * 8)) };
            self.pos += byte_count;
        }
        self.nib_off = need & 1;
        if self.nib_off == 1 {
            self.reg = (self.reg >> 4) | (u64::from(data[self.pos] & 0xf) << 60);
        }
    }
}

impl Iterator for Decoder<'_> {
    type Item = Packet;

    fn next(&mut self) -> Option<Packet> {
        if self.unsupported.is_some() || self.pos + ((self.nib_count + self.nib_off + 1) >> 1) > self.data.len() {
            return None;
        }
        self.shift_in();
        let reg = self.reg;
        let spec = &SPECS[usize::from(OPCODES[(reg & 0xff) as usize])];
        let li = self.layout as usize;
        let (lo, width) = spec.delta[li];
        let delta = if width == 0 { 0 } else { bits(reg, u32::from(lo + width - 1), u32::from(lo)) };
        let kind = kind(spec.code, reg, self.layout, delta);
        self.time = self.time.saturating_add(match kind {
            _ if kind.is_marker() => 0,
            Kind::TsDeltaShort { .. } => delta + 4,
            _ => delta,
        });
        self.nib_count = usize::from(spec.nibbles[li]);
        if let Kind::LayoutHeader { layout, .. } = kind {
            match layout {
                3 => {}
                4 => self.layout = Layout::Rdna4,
                other => {
                    self.unsupported = Some(other);
                    return None;
                }
            }
        }
        Some(Packet { time: self.time, kind })
    }
}

impl FusedIterator for Decoder<'_> {}

fn kind(code: Code, r: u64, layout: Layout, delta: u64) -> Kind {
    let b = |hi, lo| bits(r, hi, lo);
    let rdna4 = layout == Layout::Rdna4;
    let cu = |wgp: u64, sa: u64| (wgp | sa << if rdna4 { 4 } else { 3 }) as u8;
    match code {
        Code::Nop => Kind::Nop,
        Code::LayoutHeader => Kind::LayoutHeader {
            layout: b(12, 7) as u8,
            simd: b(14, 13) as u8,
            group: b(17, 15) as u8,
            sel_a: b(31, 28) as u8,
            sel_b: b(36, 33) as u8,
            flag4: flag(r, 59),
        },
        Code::WaveStart => {
            let (wgp, wave, id7) =
                if rdna4 { (b(13, 10), b(19, 15), b(31, 20)) } else { (b(12, 10), b(17, 13), b(31, 18)) };
            let sa = b(7, 7);
            Kind::WaveStart {
                sa: sa as u8,
                simd: b(9, 8) as u8,
                wgp: wgp as u8,
                cu: cu(wgp, sa),
                wave: wave as u8,
                id7: id7 as u16,
            }
        }
        Code::WaveEnd => {
            let (wgp, sa) = (if rdna4 { b(14, 11) } else { b(13, 11) }, b(8, 8));
            Kind::WaveEnd { sa: sa as u8, simd: b(10, 9) as u8, wgp: wgp as u8, cu: cu(wgp, sa), wave: b(19, 15) as u8 }
        }
        Code::WaveRdy => Kind::WaveRdy { mask: b(23, 8) as u16 },
        Code::WaveAlloc => Kind::WaveAlloc,
        Code::Inst if rdna4 => Kind::Inst {
            wave: b(11, 7) as u8,
            op: InstOp { layout, code: b(19, 12) as u8 },
            flag1: false,
            flag2: false,
            w64h: flag(r, 6),
        },
        Code::Inst => Kind::Inst {
            wave: b(12, 8) as u8,
            op: InstOp { layout, code: b(19, 13) as u8 },
            flag1: flag(r, 3),
            flag2: flag(r, 7),
            w64h: false,
        },
        Code::ValuInst => Kind::ValuInst { wave: b(11, 7) as u8, flag: flag(r, 6) },
        Code::Immediate => Kind::Immediate { wave: b(11, 7) as u8 },
        Code::ImmediateMask => Kind::ImmediateMask { mask: b(23, 8) as u16 },
        Code::AluExec => {
            Kind::AluExec { src: [AluSrc::None, AluSrc::Salu, AluSrc::Valu, AluSrc::ValuSalu][b(7, 6) as usize] }
        }
        Code::VmemExec => {
            Kind::VmemExec { src: [MemSrc::Lds, MemSrc::LdsAlt, MemSrc::Vmem, MemSrc::VmemAlt][b(7, 6) as usize] }
        }
        Code::Event => Kind::Event { event: b(23, 11) as u32, big: false },
        Code::EventBig => Kind::Event { event: b(31, 11) as u32, big: true },
        Code::Reg => {
            Kind::Reg { slot: b(9, 7) as u8, hi_byte: b(15, 8) as u8, subop: b(31, 16) as u16, val32: b(63, 32) as u32 }
        }
        Code::Snapshot => Kind::Snapshot { snap: b(63, 10) },
        Code::UtilCtr => Kind::UtilCtr { ctr: b(47, 9) },
        Code::Perf if rdna4 => Kind::Perf { arg: b(31, 10) as u32 },
        Code::Perf => Kind::Perf { arg: b(27, 8) as u32 },
        Code::TsWaveState => Kind::TsWaveState { coarse: b(23, 16) as u8 },
        Code::TsDeltaShort => Kind::TsDeltaShort { delta },
        Code::TsDeltaS5W2 => Kind::TsDeltaS5W2 { delta },
        Code::TsDeltaS5W3 => Kind::TsDeltaS5W3 { delta },
        Code::TsDeltaS8W3 => Kind::TsDeltaS8W3 { delta },
        Code::TsDeltaOrMark if rdna4 => Kind::TsDeltaOrMark { delta, rt: flag(r, 7), pl: flag(r, 8), tl: flag(r, 9) },
        Code::TsDeltaOrMark => Kind::TsDeltaOrMark { delta, pl: flag(r, 8), rt: flag(r, 9), tl: false },
    }
}
