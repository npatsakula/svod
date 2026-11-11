use std::iter::FusedIterator;

use super::{AluSrc, Header, InstOp, Kind, Layout, MemSrc, Packet, Popcount, Reg, ShaderData, Unit, WaveStart};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Token {
    Nop,
    Inst,
    ValuInst,
    ImmOne,
    Immediate,
    WaveReady,
    NewPc,
    WaveStart,
    WaveStartExt,
    WaveAlloc,
    WaveEnd,
    ShaderData,
    ShaderDataShort,
    UtilCounter,
    Time,
    Misc,
    Event,
    EventSync,
    Reg,
    RegInit,
    Timestamp,
    Header,
    ExecPopcount1,
    ExecPopcount3,
    AluExec,
    VmemExec,
}

/// A table slot: the token, its length in nibbles and its time delta field `(lo, width)` in bits.
#[derive(Clone, Copy)]
struct Slot {
    token: Token,
    nibbles: u8,
    lo: u8,
    width: u8,
}

/// `(token, pattern, pattern bits, length bits, time begin, time end)`, AMD's `encoding_t`.
type Encoding = (Token, u8, u8, u8, u8, u8);

/// `gfx10token.cpp`.
#[rustfmt::skip]
const GFX10: &[Encoding] = &[
    (Token::Inst, 0b010, 3, 20, 4, 7), (Token::ValuInst, 0b011, 3, 12, 3, 6), (Token::ImmOne, 0b1101, 4, 12, 4, 7),
    (Token::Immediate, 0b00100, 5, 24, 5, 8), (Token::WaveReady, 0b10100, 5, 24, 5, 8),
    (Token::NewPc, 0b0100001, 7, 64, 8, 11), (Token::WaveStart, 0b01100, 5, 32, 5, 7),
    (Token::WaveStartExt, 0b11100, 5, 48, 5, 7), (Token::WaveAlloc, 0b00101, 5, 20, 5, 8),
    (Token::WaveEnd, 0b10101, 5, 20, 5, 8), (Token::ShaderData, 0b00110, 5, 52, 5, 8),
    (Token::ShaderDataShort, 0b10110, 5, 28, 5, 8), (Token::UtilCounter, 0b0110001, 7, 64, 7, 9),
    (Token::Time, 0b1000, 4, 8, 4, 8), (Token::Nop, 0b0000, 4, 4, 0, 0), (Token::Misc, 0b1010001, 7, 24, 7, 16),
    (Token::Event, 0b01100001, 8, 24, 8, 11), (Token::EventSync, 0b11100001, 8, 32, 8, 11),
    (Token::Reg, 0b1001, 4, 64, 4, 7), (Token::RegInit, 0b1110001, 7, 64, 7, 10),
    (Token::Timestamp, 0b0000001, 7, 64, 16, 64), (Token::Header, 0b0010001, 7, 64, 0, 0),
    // tinygrad's; AMD leaves these 4-bit encodings unknown.
    (Token::AluExec, 0b1110, 4, 8, 4, 6), (Token::VmemExec, 0b1111, 4, 8, 4, 6),
];

/// `gfx11token.cpp`.
#[rustfmt::skip]
const GFX11: &[Encoding] = &[
    (Token::Misc, 0b1010001, 7, 24, 7, 16), (Token::UtilCounter, 0b0110001, 7, 48, 7, 9),
    (Token::Timestamp, 0b0000001, 7, 48, 12, 48),
];

/// `gfx12token.cpp`; `WAVE_START_EXT` is 40 bits only under header WSM 3 (see [`Decoder::next`]).
#[rustfmt::skip]
const GFX12: &[Encoding] = &[
    (Token::Inst, 0b010, 3, 20, 3, 6), (Token::ShaderData, 0b0000110, 7, 56, 7, 10),
    (Token::ShaderDataShort, 0b1000110, 7, 32, 7, 10), (Token::ExecPopcount1, 0b1100110, 7, 24, 7, 10),
    (Token::ExecPopcount3, 0b010110, 6, 48, 6, 9), (Token::NewPc, 0b0100001, 7, 72, 8, 11),
    (Token::WaveStartExt, 0b11100, 5, 40, 5, 7), (Token::WaveAlloc, 0b00101, 5, 24, 5, 8),
    (Token::Timestamp, 0b0000001, 7, 64, 12, 64),
];

const fn table(layers: &[&[Encoding]]) -> [Slot; 256] {
    let mut table = [Slot { token: Token::Nop, nibbles: 1, lo: 0, width: 0 }; 256];
    let mut l = 0;
    while l < layers.len() {
        let mut e = 0;
        while e < layers[l].len() {
            let (token, pattern, bits, len, lo, hi) = layers[l][e];
            let mut byte = pattern as usize;
            while byte < 256 {
                table[byte] = Slot { token, nibbles: len / 4, lo, width: hi - lo };
                byte += 1 << bits;
            }
            e += 1;
        }
        l += 1;
    }
    table
}

const TABLES: [[Slot; 256]; 2] = [table(&[GFX10, GFX11]), table(&[GFX10, GFX11, GFX12])];

fn bits(reg: u64, hi: u32, lo: u32) -> u64 {
    (reg >> lo) & (u64::MAX >> (63 - (hi - lo)))
}

fn flag(reg: u64, bit: u32) -> bool {
    bits(reg, bit, bit) != 0
}

/// Decodes a raw SQTT blob. Starts in layout 3 and follows the `HEADER`s it meets.
pub fn decode(data: &[u8]) -> Decoder<'_> {
    Decoder { data, nibble: 0, time: 0, layout: Layout::Rdna3, wsm: 0, unsupported: None }
}

/// The packet iterator over one SQTT blob: a 64-bit window at each packet's first nibble, zero past the
/// end of the data.
///
/// Like AMD's generators, a layout-3 stream decodes every packet that starts inside the data and a
/// layout-4 one stops at the first that ends past it; tinygrad stops once the window leaves the data.
pub struct Decoder<'a> {
    data: &'a [u8],
    nibble: usize,
    time: u64,
    layout: Layout,
    wsm: u8,
    unsupported: Option<u8>,
}

impl Decoder<'_> {
    /// The layout the packets that follow are decoded with.
    pub fn layout(&self) -> Layout {
        self.layout
    }

    /// The version of a `HEADER` that is neither 3 nor 4 (CDNA, or not a header at all); decoding stops there.
    pub fn unsupported_layout(&self) -> Option<u8> {
        self.unsupported
    }

    fn window(&self, nibble: usize) -> u64 {
        let (byte, mut chunk) = (nibble / 2, [0u8; 9]);
        let tail = self.data.get(byte..).unwrap_or_default();
        let n = tail.len().min(9);
        chunk[..n].copy_from_slice(&tail[..n]);
        let low = u64::from_le_bytes(chunk[..8].try_into().expect("eight bytes"));
        if nibble.is_multiple_of(2) { low } else { low >> 4 | u64::from(chunk[8]) << 60 }
    }
}

impl Iterator for Decoder<'_> {
    type Item = Packet;

    fn next(&mut self) -> Option<Packet> {
        let end = 2 * self.data.len();
        if self.unsupported.is_some() || self.nibble >= end {
            return None;
        }
        let raw = self.window(self.nibble);
        let rdna4 = self.layout == Layout::Rdna4;
        let slot = TABLES[usize::from(rdna4)][(raw & 0xff) as usize];
        let nibbles = match slot.token {
            Token::WaveStartExt if rdna4 && self.wsm != 3 => 12,
            _ => usize::from(slot.nibbles),
        };
        if rdna4 && self.nibble + nibbles > end {
            self.nibble = end;
            return None;
        }
        let delta =
            if slot.width == 0 { 0 } else { bits(raw, u32::from(slot.lo + slot.width - 1), u32::from(slot.lo)) };
        let extra = if nibbles > 16 { self.window(self.nibble + 16) } else { 0 };
        let kind = kind(slot.token, raw, extra, self.layout, delta);
        self.time = self.time.saturating_add(match kind {
            Kind::Timestamp { rt: true, .. } => 0,
            Kind::Time { .. } => delta + 4,
            _ => delta,
        });
        self.nibble += nibbles;
        if let Kind::Header(header) = kind {
            match header.version {
                3 => self.layout = Layout::Rdna3,
                4 => self.layout = Layout::Rdna4,
                other => {
                    self.unsupported = Some(other);
                    return None;
                }
            }
            self.wsm = header.wsm;
        }
        Some(Packet { time: self.time, raw, kind })
    }
}

impl FusedIterator for Decoder<'_> {}

fn kind(token: Token, r: u64, extra: u64, layout: Layout, delta: u64) -> Kind {
    let b = |hi, lo| bits(r, hi, lo) as u8;
    let rdna4 = layout == Layout::Rdna4;
    let unit = |sa, simd: (u32, u32), wgp: (u32, u32), wave: (u32, u32)| Unit {
        sa: b(sa, sa),
        simd: b(simd.0, simd.1),
        wgp: b(wgp.0, wgp.1),
        wave: b(wave.0, wave.1),
    };
    let popcount = |lo: u32| Popcount { wave: b(lo + 4, lo), count: b(lo + 11, lo + 5), valu: flag(r, lo + 12) };
    match token {
        Token::Nop => Kind::Nop,
        Token::Header => Kind::Header(Header {
            version: b(12, 7),
            dsimd: b(14, 13),
            dwgp: b(17, 15),
            dsa: b(19, 19),
            nsa: b(20, 20),
            nwgp: b(23, 21),
            wsm: b(26, 25),
            ucf: flag(r, 27),
            dp_rate: b(31, 28),
            dp_derate: b(34, 33),
            exec: flag(r, 42),
            dbl: flag(r, 43),
            exbarw: flag(r, 59),
        }),
        Token::WaveStart | Token::WaveStartExt if rdna4 => {
            let ext = flag(r, 4) && !flag(r, 32);
            Kind::WaveStart(WaveStart {
                unit: unit(7, (9, 8), (13, 10), (19, 15)),
                pipe: b(21, 20),
                me: b(22, 22),
                count: b(31, 25),
                ext,
                wgid: if ext { b(37, 33) } else { 0 },
            })
        }
        Token::WaveStart | Token::WaveStartExt => Kind::WaveStart(WaveStart {
            unit: unit(7, (9, 8), (12, 10), (17, 13)),
            pipe: b(22, 21),
            me: b(23, 23),
            count: b(31, 25),
            ext: false,
            wgid: 0,
        }),
        Token::WaveEnd => Kind::WaveEnd(unit(8, (10, 9), if rdna4 { (14, 11) } else { (13, 11) }, (19, 15))),
        Token::WaveAlloc => Kind::WaveAlloc,
        Token::WaveReady => Kind::WaveReady { mask: bits(r, 23, 8) as u16 },
        Token::Inst if rdna4 => Kind::Inst { wave: b(11, 7), op: InstOp { layout, code: b(19, 12) }, w64h: flag(r, 6) },
        Token::Inst => Kind::Inst { wave: b(12, 8), op: InstOp { layout, code: b(19, 13) }, w64h: flag(r, 7) },
        Token::ValuInst => Kind::ValuInst { wave: b(11, 7), w64h: flag(r, 6) },
        Token::ImmOne => Kind::ImmOne { wave: b(11, 7) },
        Token::Immediate => Kind::Immediate { mask: bits(r, 23, 8) as u16 },
        Token::AluExec => {
            Kind::AluExec { src: [AluSrc::None, AluSrc::Salu, AluSrc::Valu, AluSrc::ValuSalu][b(7, 6) as usize] }
        }
        Token::VmemExec => {
            Kind::VmemExec { src: [MemSrc::Lds, MemSrc::LdsAlt, MemSrc::Vmem, MemSrc::VmemAlt][b(7, 6) as usize] }
        }
        Token::Event | Token::EventSync => Kind::Event {
            id: b(23, 18),
            me: b(17, 16),
            pipe: b(15, 14),
            bop: flag(r, 11),
            sync: token == Token::EventSync,
        },
        Token::Reg => Kind::Reg(Reg {
            pipe: b(8, 7),
            me: b(10, 9),
            cs: flag(r, 15),
            addr: bits(r, 31, 16) as u16,
            data: bits(r, 63, 32) as u32,
        }),
        Token::RegInit => Kind::RegInit { me: b(17, 16), pipe: b(15, 14), ty: b(19, 18), data: bits(r, 51, 20) as u32 },
        Token::UtilCounter => Kind::UtilCounter,
        Token::ShaderData | Token::ShaderDataShort => {
            let short = token == Token::ShaderDataShort;
            let (unit, data) = match (rdna4, short) {
                (true, false) => (unit(10, (12, 11), (16, 13), (21, 17)), bits(r, 53, 22) as u32),
                (true, true) => (unit(10, (12, 11), (16, 13), (21, 17)), b(29, 22).into()),
                (false, false) => (unit(8, (10, 9), (13, 11), (19, 15)), bits(r, 51, 20) as u32),
                (false, true) => (unit(8, (10, 9), (13, 11), (19, 15)), b(27, 20).into()),
            };
            let private = rdna4 && if short { b(31, 30) != 0 } else { b(55, 54) != 0 };
            Kind::ShaderData(ShaderData { unit, data, short, private, invalid: !rdna4 && flag(r, 14) })
        }
        Token::Misc => Kind::Misc { fields: b(23, 16) },
        Token::Time => Kind::Time { delta },
        Token::NewPc if rdna4 => Kind::NewPc { wave: b(15, 11), pc: (bits(r, 63, 16) | (extra & 0x7f) << 48) << 2 },
        Token::NewPc => Kind::NewPc { wave: b(15, 11), pc: bits(r, 61, 16) << 2 },
        Token::Timestamp if rdna4 => Kind::Timestamp { delta, rt: flag(r, 7), pl: flag(r, 8), tl: flag(r, 9) },
        Token::Timestamp => Kind::Timestamp { delta, pl: flag(r, 8), rt: flag(r, 9), tl: false },
        Token::ExecPopcount1 => Kind::ExecPopcount1(popcount(10)),
        Token::ExecPopcount3 => Kind::ExecPopcount3([popcount(9), popcount(22), popcount(35)]),
    }
}
