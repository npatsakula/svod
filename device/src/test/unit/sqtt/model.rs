//! Hand-built token streams through the decoder, the wave model, the stitcher and the report.

use proptest::prelude::*;
use test_case::test_case;

use crate::sqtt::hidden;
use crate::sqtt::report::{Report, Stats};
use crate::sqtt::simd::Residency;
use crate::sqtt::stitch::{Stitcher, objdump_lines};
use crate::sqtt::waits::Replay;
use crate::sqtt::wave::{Instruction, Wave, analyze};
use crate::sqtt::{Category, Header, InstOp, Kind, Layout, Packet, Popcount, Unit, decode};

/// Packs `(value, nibbles)` tokens into a little-endian nibble stream, NOP-padded to whole bytes.
fn stream(tokens: &[(u128, usize)]) -> Vec<u8> {
    let mut nibbles: Vec<u8> =
        tokens.iter().flat_map(|&(v, n)| (0..n).map(move |i| (v >> (4 * i) & 0xf) as u8)).collect();
    nibbles.resize(nibbles.len().next_multiple_of(2), 0);
    nibbles.chunks(2).map(|pair| pair[0] | pair[1] << 4).collect()
}

/// A `HEADER` instruction-tracing SIMD 0 of WGP 0 on shader array 0.
fn header(h: Header) -> (u128, usize) {
    header_of(if h.layout == Layout::Rdna4 { 4 } else { 3 }, h)
}

/// `h` announcing `version`, which [`Header::layout`] cannot hold where it is neither 3 nor 4.
fn header_of(version: u8, h: Header) -> (u128, usize) {
    let v = 0b0010001
        | u128::from(version) << 7
        | u128::from(h.dsimd) << 13
        | u128::from(h.dwgp) << 15
        | u128::from(h.dsa) << 19
        | u128::from(h.nsa) << 20
        | u128::from(h.nwgp) << 21
        | u128::from(h.wsm) << 25
        | u128::from(h.ucf) << 27
        | u128::from(h.dp_rate) << 28
        | u128::from(h.dp_derate) << 33
        | u128::from(h.exec) << 42
        | u128::from(h.dbl) << 43
        | u128::from(h.exbarw) << 59;
    (v, 16)
}

fn version(v: u8) -> (u128, usize) {
    header_of(v, Header::default())
}

fn layout(v: u8) -> Layout {
    if v == 4 { Layout::Rdna4 } else { Layout::Rdna3 }
}

fn time(delta: u128) -> (u128, usize) {
    (0b1000 | delta << 4, 2)
}

fn inst(rdna4: bool, wave: u128, code: u128, delta: u128) -> (u128, usize) {
    let v =
        if rdna4 { 0b010 | delta << 3 | wave << 7 | code << 12 } else { 0b010 | delta << 4 | wave << 8 | code << 13 };
    (v, 5)
}

fn valu(wave: u128, delta: u128) -> (u128, usize) {
    (0b011 | delta << 3 | wave << 7, 3)
}

fn imm_one(wave: u128, delta: u128) -> (u128, usize) {
    (0b1101 | delta << 4 | wave << 7, 3)
}

fn wave_ready(mask: u128, delta: u128) -> (u128, usize) {
    (0b10100 | delta << 5 | mask << 8, 6)
}

fn wave_start(rdna4: bool, wave: u128, count: u128) -> (u128, usize) {
    let slot = if rdna4 { wave << 15 } else { wave << 13 };
    (0b01100 | slot | count << 25, 8)
}

fn wave_end(wave: u128) -> (u128, usize) {
    (0b10101 | wave << 15, 5)
}

fn packets(tokens: &[(u128, usize)]) -> Vec<Packet> {
    decode(&stream(tokens)).filter(|p| p.kind != Kind::Nop).collect()
}

#[test]
fn header_fields() {
    let h = Header {
        layout: Layout::Rdna4,
        dsimd: 2,
        dwgp: 5,
        dsa: 1,
        nsa: 1,
        nwgp: 6,
        wsm: 3,
        ucf: true,
        dp_rate: 3,
        dp_derate: 2,
        exec: true,
        dbl: true,
        exbarw: true,
    };
    assert_eq!(packets(&[header(h)])[0].kind, Kind::Header(Ok(h)));
}

#[test_case(0, 1)]
#[test_case(1, 1)]
#[test_case(2, 2)]
#[test_case(4, 8)]
#[test_case(0xf, 64 ; "masked to three bits")]
fn dp_cycles(dp_rate: u8, cycles: u32) {
    assert_eq!(Header { dp_rate, ..Header::default() }.dp_cycles(), cycles);
}

/// A `HEADER` of neither layout (CDNA's 2, or not a header at all) is the last packet, refused.
#[test_case(2)]
#[test_case(5)]
fn foreign_version_stops_decoding(v: u8) {
    let data = stream(&[version(v), time(3)]);
    let mut decoder = decode(&data);
    assert_eq!(decoder.next().map(|p| p.kind), Some(Kind::Header(Err(v))));
    assert_eq!(decoder.unsupported_layout(), Some(v));
    assert_eq!(decoder.next(), None);
    let trace = analyze(decode(&data));
    assert_eq!((trace.header, trace.refused), (None, Some(v)), "the wave model names it");
}

#[test_case(3, Layout::Rdna3)]
#[test_case(4, Layout::Rdna4)]
fn header_selects_layout(v: u8, layout: Layout) {
    let data = stream(&[version(v)]);
    let mut decoder = decode(&data);
    decoder.next();
    assert_eq!(decoder.layout(), layout);
}

// A realtime timestamp (`rt`) carries no delta; otherwise it adds its delta, lost packets (`pl`) or not.
#[test_case(4, 1 << 7, 0 ; "rdna4 realtime")]
#[test_case(4, 1 << 8, 100 ; "rdna4 lost packets")]
#[test_case(4, 1 << 9, 100 ; "rdna4 tl")]
#[test_case(3, 1 << 9, 0 ; "rdna3 realtime")]
#[test_case(3, 1 << 8 | 1 << 9, 0 ; "rdna3 realtime lost packets")]
#[test_case(3, 1 << 8, 100 ; "rdna3 lost packets")]
fn timestamp_delta(v: u8, flags: u128, added: u64) {
    let nibbles = if v == 4 { 16 } else { 12 };
    let decoded = packets(&[version(v), time(3), (0b0000001 | flags | 100 << 12, nibbles), time(0)]);
    assert_eq!(decoded[1].time, 7);
    assert!(matches!(decoded[2].kind, Kind::Timestamp { delta: 100, .. }), "{:?}", decoded[2]);
    assert_eq!(decoded[2].time, 7 + added);
    assert_eq!(decoded[3].time, 7 + added + 4);
}

#[test_case(3, 10 ; "wsm 3")]
#[test_case(0, 12 ; "otherwise")]
fn wave_start_ext_width_follows_wsm(wsm: u8, nibbles: usize) {
    let ext = 0b11100 | 1 << 15 | 9 << 33;
    let decoded =
        packets(&[header(Header { layout: Layout::Rdna4, wsm, ..Header::default() }), (ext, nibbles), valu(3, 2)]);
    let Kind::WaveStart(start) = decoded[1].kind else { panic!("{:?}", decoded[1]) };
    assert_eq!((start.ext, start.wgid, start.unit.wave), (true, 9, 1));
    assert_eq!(decoded[2].kind, Kind::ValuInst { wave: 3, w64h: false });
    assert_eq!(decoded.len(), 3);
}

// `NEW_PC` is 64 bits on gfx11 and 72 on gfx12, whose top byte extends the dword address.
#[test_case(3, 16, 0x7fff_ffff_fffc ; "rdna3")]
#[test_case(4, 18, 0x1ff_ffff_ffff_fffc ; "rdna4")]
fn new_pc(v: u8, nibbles: usize, pc: u64) {
    let raw = 0b0100001 | 2 << 8 | 5 << 11 | u128::from(pc >> 2) << 16;
    let decoded = packets(&[version(v), (raw, nibbles), valu(1, 1)]);
    assert_eq!(decoded[1], Packet { time: 2, raw: raw as u64, kind: Kind::NewPc { wave: 5, pc } });
    assert_eq!(decoded[2].kind, Kind::ValuInst { wave: 1, w64h: false });
    assert_eq!(decoded.len(), 3);
}

#[test]
fn exec_popcounts() {
    let one = 0b1100110 | 3 << 10 | 64 << 15 | 1 << 22;
    let three = 0b010110 | 1 << 9 | 2 << 14 | 2 << 22 | 32 << 27 | 1 << 34 | 31 << 35 | 127 << 40;
    let decoded = packets(&[version(4), (one, 6), (three, 12)]);
    assert_eq!(decoded[1].kind, Kind::ExecPopcount1(Popcount { wave: 3, count: 64, valu: true }));
    assert_eq!(
        decoded[2].kind,
        Kind::ExecPopcount3([
            Popcount { wave: 1, count: 2, valu: false },
            Popcount { wave: 2, count: 32, valu: true },
            Popcount { wave: 31, count: 127, valu: false },
        ])
    );
}

// A gfx11 stream decodes a token that starts inside the data, a gfx12 one stops at a token ending past it.
#[test_case(3, 2)]
#[test_case(4, 1)]
fn truncated_tail(v: u8, count: usize) {
    let mut data = stream(&[version(v), inst(v == 4, 1, 0, 1)]);
    data.pop();
    assert_eq!(decode(&data).count(), count);
}

#[test_case(Layout::Rdna3, 19, Some("barrier"), Category::Immed, 1, false)]
#[test_case(Layout::Rdna4, 19, Some("barrier_wait"), Category::Immed, 1, false)]
#[test_case(Layout::Rdna3, 10, Some("valu_1"), Category::Valu, 4, false ; "gfx11 valu_1 at dp rate")]
#[test_case(Layout::Rdna4, 10, Some("valu_1"), Category::Valu, 1, false)]
#[test_case(Layout::Rdna4, 146, Some("valu_dpfp"), Category::Valu, 4, false)]
#[test_case(Layout::Rdna4, 147, Some("valu_derate"), Category::Valu, 16, false)]
#[test_case(Layout::Rdna4, 30, Some("flat_wr_4"), Category::Flat, 4, false)]
#[test_case(Layout::Rdna4, 143, Some("wmma64"), Category::Valu, 64, false)]
#[test_case(Layout::Rdna4, 230, Some("block_store_9"), Category::Vmem, 9, false)]
#[test_case(Layout::Rdna3, 92, Some("vmem_other_simd_3"), Category::Vmem, 3, true)]
#[test_case(Layout::Rdna4, 190, Some("vmem_other_simd_3"), Category::Vmem, 3, true)]
#[test_case(Layout::Rdna3, 20, Some("expreq_gds"), Category::None, 0, false)]
#[test_case(Layout::Rdna3, 140, None, Category::None, 0, false)]
fn inst_op_class(layout: Layout, code: u8, name: Option<&str>, category: Category, cycles: u32, other: bool) {
    let op = InstOp { layout, code };
    let header = Header { layout: Layout::Rdna4, dp_rate: 3, dp_derate: 2, ..Header::default() };
    assert_eq!(op.name().as_deref(), name);
    let class = op.class(&header);
    assert_eq!((class.category, class.cycles, class.other_simd), (category, cycles, other));
    assert_eq!(op.is_other_simd(), other);
}

fn record(time: i64, category: Category, duration: i64, stall: i64) -> Instruction {
    Instruction { time, category, duration, stall, pc: 0 }
}

#[test]
fn stalls_immediates_and_endpgm() {
    let trace = analyze(packets(&[
        version(4),
        time(6),
        wave_start(true, 1, 0),
        inst(true, 1, 0, 2),
        wave_ready(1 << 1, 3),
        time(6),
        valu(1, 0),
        imm_one(1, 2),
        time(6),
        wave_end(1),
    ]));
    let [wave] = trace.waves.as_slice() else { panic!("{:?}", trace.waves) };
    assert_eq!((wave.begin, wave.end, wave.complete), (10, 37, true));
    assert_eq!(
        wave.instructions,
        [
            record(12, Category::Salu, 1, 0),
            // Ready since 15, issued at 25: the wave stalled 10 cycles.
            record(15, Category::Valu, 11, 10),
            record(26, Category::Immed, 2, 1),
            // `s_endpgm` sends no token: its record fills the gap to the wave's end.
            record(28, Category::Msg, 9, 0),
        ]
    );
}

#[test]
fn other_waves_and_simds_are_not_traced() {
    let start = (0b01100 | 1 << 8 | 2 << 15, 8);
    let trace = analyze(packets(&[version(4), start, inst(true, 2, 0, 1), inst(true, 2, 190, 1)]));
    assert!(trace.waves.is_empty());
    assert_eq!(trace.occupancy.len(), 1);
    assert_eq!(trace.other_simd.len(), 1);
    assert_eq!((trace.other_simd[0].category, trace.other_simd[0].cycles), (Category::Vmem, 3));
}

// gfx11 `s_barrier` traces as an `IMMED` stretched into a `MSG` up to the wave's next token.
#[test]
fn gfx11_barrier_becomes_message() {
    let trace = analyze(packets(&[
        version(3),
        wave_start(false, 0, 0),
        inst(false, 0, 19, 2),
        time(6),
        valu(0, 0),
        wave_end(0),
    ]));
    let categories: Vec<_> = trace.waves[0].instructions.iter().map(|i| (i.time, i.category, i.duration)).collect();
    assert_eq!(categories, [(2, Category::Msg, 10), (12, Category::Valu, 1)]);
}

// Count 66 saves the wave's context and 68 restores it; the wave ignores tokens until a trap return.
#[test]
fn context_save_and_restore() {
    let trace = analyze(packets(&[
        version(4),
        wave_start(true, 0, 0),
        valu(0, 1),
        wave_start(true, 0, 66),
        time(6),
        wave_start(true, 0, 68),
        valu(0, 1),
        inst(true, 0, 117, 1),
        valu(0, 1),
        wave_end(0),
    ]));
    let wave = &trace.waves[0];
    assert_eq!(wave.contexts, 2);
    let categories: Vec<_> = wave.instructions.iter().map(|i| (i.time, i.category, i.duration)).collect();
    // The trap spans the save up to the return; the return leaves the PC to find from there on.
    assert_eq!(categories, [(1, Category::Valu, 1), (1, Category::Trap, 13), (14, Category::Valu, 1)]);
    assert_eq!(wave.pcs, [(2, 0)]);
}

const LOOP: &str = "
0000000000001600 <k>:
\ts_mov_b32 s0, 0                // 000000001600: BE800080
\tv_add_f32_e32 v0, v0, v0       // 000000001604: 06000100
\ts_add_u32 s0, s0, 1            // 000000001608: 80008100
\ts_cbranch_scc1 65533           // 00000000160C: BF85FFFD
\ts_endpgm                       // 000000001610: BFB00000
";

fn wave(categories: &[Category], start: u64) -> Wave {
    Wave {
        unit: Unit::default(),
        me: 0,
        pipe: 0,
        begin: 0,
        end: 0,
        complete: true,
        contexts: 0,
        exclude_barrier_wait: false,
        instructions: categories.iter().enumerate().map(|(t, &c)| record(t as i64, c, 1, 0)).collect(),
        pcs: vec![(0, start)],
    }
}

#[test]
fn objdump_pcs() {
    let lines = objdump_lines(LOOP);
    assert_eq!(lines.iter().map(|(pc, _)| *pc).collect::<Vec<_>>(), [0x1600, 0x1604, 0x1608, 0x160c, 0x1610]);
    assert!(lines[3].1.contains("s_cbranch_scc1"));
}

#[test]
fn stitch_follows_branches() {
    use Category::{Jump, Msg, Next, Salu, Valu};
    let code: Vec<_> = objdump_lines(LOOP).into_iter().map(|(pc, line)| (pc + 0x1000_0000, line)).collect();
    let mut stitcher = Stitcher::new(&code, Layout::Rdna3);
    let mut w = wave(&[Salu, Valu, Salu, Jump, Valu, Salu, Next, Msg], 0x1000_1600);
    assert_eq!(stitcher.stitch(&mut w), 8);
    let pcs: Vec<_> = w.instructions.iter().map(|i| i.pc - 0x1000_0000).collect();
    assert_eq!(pcs, [0x1600, 0x1604, 0x1608, 0x160c, 0x1604, 0x1608, 0x160c, 0x1610]);
}

#[test]
fn stitch_stops_at_a_mismatch() {
    use Category::{Lds, Salu};
    let code = objdump_lines(LOOP);
    let mut w = wave(&[Salu, Lds, Salu], 0x1600);
    assert_eq!(Stitcher::new(&code, Layout::Rdna3).stitch(&mut w), 1);
    assert_eq!(w.instructions.iter().map(|i| i.pc).collect::<Vec<_>>(), [0x1600, 0, 0]);
}

#[test]
fn branch_target_from_decimal_operand() {
    use Category::{Jump, Salu};
    let code = [(0x100, "s_branch 2"), (0x104, "s_nop 0"), (0x108, "s_nop 0"), (0x10c, "s_mov_b32 s0, 0")];
    let mut w = wave(&[Jump, Salu], 0x100);
    assert_eq!(Stitcher::new(&code, Layout::Rdna3).stitch(&mut w), 2);
    assert_eq!(w.instructions[1].pc, 0x10c);
}

// gfx12 `v_mov_b64` issues over at least two cycles past its stall.
#[test_case(Layout::Rdna4, 5)]
#[test_case(Layout::Rdna3, 1)]
fn v_mov_b64_takes_two_cycles(layout: Layout, duration: i64) {
    let code = [(0x100, "s_mov_b32 s0, 0"), (0x104, "v_mov_b64_e32 v[0:1], 0"), (0x108, "s_endpgm")];
    let mut w = wave(&[Category::Salu, Category::Valu], 0x100);
    w.instructions[1].stall = 3;
    assert_eq!(Stitcher::new(&code, layout).stitch(&mut w), 2);
    assert_eq!(w.instructions[1].duration, duration);
}

// gfx12 can leave `s_barrier_wait` out of the trace; the stitcher re-inserts it as the gap it left.
#[test]
fn excluded_barrier_wait_is_reinserted() {
    let code = [(0x100, "s_mov_b32 s0, 0"), (0x104, "s_barrier_wait 0xffff"), (0x108, "v_mov_b32_e32 v0, 0")];
    let mut w = wave(&[Category::Salu, Category::Valu], 0x100);
    w.instructions[1].time = 10;
    w.exclude_barrier_wait = true;
    assert_eq!(Stitcher::new(&code, Layout::Rdna4).stitch(&mut w), 2);
    assert_eq!(
        w.instructions,
        [
            Instruction { time: 0, category: Category::Salu, duration: 1, stall: 0, pc: 0x100 },
            Instruction { time: 1, category: Category::Immed, duration: 9, stall: 0, pc: 0x104 },
            Instruction { time: 10, category: Category::Valu, duration: 1, stall: 0, pc: 0x108 },
        ]
    );
}

#[test]
fn report_idle_is_the_gap_since_the_previous_record_ended() {
    let mut w = wave(&[], 0);
    w.begin = 2;
    w.instructions = vec![
        Instruction { time: 4, category: Category::Valu, duration: 5, stall: 2, pc: 0xa },
        Instruction { time: 6, category: Category::Valu, duration: 1, stall: 0, pc: 0xb },
        Instruction { time: 9, category: Category::Salu, duration: 1, stall: 0, pc: 0 },
        Instruction { time: 15, category: Category::Valu, duration: 2, stall: 1, pc: 0xa },
    ];
    let report = Report::new(&[w]);
    assert_eq!(report.pcs[&0xa], Stats { hits: 2, latency: 7, stall: 3, idle: 2 + 6 });
    assert_eq!(report.pcs[&0xb], Stats { hits: 1, latency: 1, stall: 0, idle: 0 });
    assert_eq!(report.waves[0].records, 4);
    assert_eq!(report.waves[0].stats, Stats { hits: 3, latency: 8, stall: 3, idle: 8 });
}

proptest! {
    #[test]
    fn arbitrary_bytes_decode_monotonically(data in prop::collection::vec(any::<u8>(), 0..4096)) {
        let times: Vec<u64> = decode(&data).map(|p| p.time).collect();
        prop_assert!(times.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn arbitrary_tokens_after_a_header_analyze_and_stitch(
        v in 3u8..=4,
        exbarw: bool,
        wsm in 0u8..4,
        body in prop::collection::vec(any::<u8>(), 0..4096),
    ) {
        let mut data = stream(&[header(Header { layout: layout(v), wsm, exbarw, ..Header::default() })]);
        data.extend(body);
        analyzes_and_stitches(&data, layout(v))?;
    }

    /// Waves whose records straddle time jumps up to the wave model's 2^60 clamp analyze without overflow.
    #[test]
    fn huge_time_jumps_analyze(tokens in prop::collection::vec((0u8..5, 0u128..16, 0u128..(1 << 52)), 0..512)) {
        let mut all = vec![version(4)];
        all.extend(tokens.into_iter().map(|(token, wave, delta)| match token {
            0 => wave_start(true, wave, 0),
            1 => valu(wave, delta & 0xf),
            2 => inst(true, wave, delta & 0x7f, delta & 0x7),
            3 => (0b0000001 | delta << 12, 16),
            _ => wave_end(wave),
        }));
        analyzes_and_stitches(&stream(&all), Layout::Rdna4)?;
    }
}

/// The wave model, the stitcher over [`LOOP`], the report, AMD's hidden latency, the residency and
/// the wait replay over one SE's `data`, checked for consistency.
fn analyzes_and_stitches(data: &[u8], layout: Layout) -> Result<(), TestCaseError> {
    let mut trace = analyze(decode(data));
    let code = objdump_lines(LOOP);
    let mut stitcher = Stitcher::new(&code, layout);
    for wave in &mut trace.waves {
        let records = wave.instructions.len();
        prop_assert!(stitcher.stitch(wave) <= records);
    }
    let report = Report::new(&trace.waves);
    prop_assert!(report.pcs.keys().all(|pc| code.iter().any(|(line, _)| line == pc)));
    let hidden = hidden::analyze([(0, &trace)], &hidden::matrix_pcs(&code));
    prop_assert!(hidden.by_pc.keys().all(|pc| report.pcs.contains_key(pc)));
    if let Some(residency) = Residency::new(&trace.occupancy) {
        prop_assert!(residency.window.0 <= residency.window.1);
    }
    let replay = Replay::new(&trace.waves, &code, layout);
    prop_assert!(replay.bound().all(|(wave, _, _, pc)| wave < trace.waves.len() && report.pcs.contains_key(&pc)));
    Ok(())
}
