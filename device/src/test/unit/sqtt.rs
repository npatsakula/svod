use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;

use proptest::prelude::*;
use serde_json::Value;
use sha2::{Digest, Sha256};
use test_case::test_case;

use crate::sqtt::{InstOp, Kind, Layout, Packet, decode};

/// tinygrad's packet class name, which the goldens record.
fn tinygrad_name(kind: &Kind, layout: Layout) -> String {
    let base = match kind {
        Kind::Nop => "NOP",
        Kind::LayoutHeader { .. } => return "LAYOUT_HEADER".into(),
        Kind::WaveStart { .. } => "WAVESTART",
        Kind::WaveEnd { .. } => "WAVEEND",
        Kind::WaveRdy { .. } => "WAVERDY",
        Kind::WaveAlloc => "WAVEALLOC",
        Kind::Inst { .. } => "INST",
        Kind::ValuInst { .. } => "VALUINST",
        Kind::Immediate { .. } => "IMMEDIATE",
        Kind::ImmediateMask { .. } => "IMMEDIATE_MASK",
        Kind::AluExec { .. } => "ALUEXEC",
        Kind::VmemExec { .. } => "VMEMEXEC",
        Kind::Event { big: false, .. } => "EVENT",
        Kind::Event { big: true, .. } => "EVENT_BIG",
        Kind::Reg { .. } => "REG",
        Kind::Snapshot { .. } => "SNAPSHOT",
        Kind::UtilCtr { .. } => "UTILCTR",
        Kind::Perf { .. } => "PERF",
        Kind::TsWaveState { .. } => "TS_WAVE_STATE",
        Kind::TsDeltaShort { .. } => "TS_DELTA_SHORT",
        Kind::TsDeltaS5W2 { .. } => "TS_DELTA_S5_W2",
        Kind::TsDeltaS5W3 { .. } => "TS_DELTA_S5_W3",
        Kind::TsDeltaS8W3 { .. } => "TS_DELTA_S8_W3",
        Kind::TsDeltaOrMark { .. } => "TS_DELTA_OR_MARK",
        Kind::ShaderDataShort { .. } => return "SHADER_DATA_SHORT".into(),
    };
    let rdna4_class = matches!(
        kind,
        Kind::WaveStart { .. }
            | Kind::WaveEnd { .. }
            | Kind::WaveAlloc
            | Kind::Inst { .. }
            | Kind::Perf { .. }
            | Kind::TsDeltaS5W2 { .. }
            | Kind::TsDeltaS5W3 { .. }
            | Kind::TsDeltaOrMark { .. }
    );
    if layout == Layout::Rdna4 && rdna4_class { format!("{base}_RDNA4") } else { base.into() }
}

fn op_name(op: InstOp) -> String {
    op.name().map_or_else(|| op.code.to_string(), Into::into)
}

/// The fields tinygrad's class declares, sorted by name, `delta` only on the `TS_DELTA_*` classes.
fn fields(kind: &Kind, layout: Layout) -> BTreeMap<&'static str, String> {
    let b = |v: bool| u8::from(v).to_string();
    let rdna4 = layout == Layout::Rdna4;
    let f: Vec<(&str, String)> = match *kind {
        Kind::Nop | Kind::WaveAlloc => vec![],
        Kind::LayoutHeader { layout, simd, group, sel_a, sel_b, flag4 } => vec![
            ("layout", layout.to_string()),
            ("simd", simd.to_string()),
            ("group", group.to_string()),
            ("sel_a", sel_a.to_string()),
            ("sel_b", sel_b.to_string()),
            ("flag4", b(flag4)),
        ],
        Kind::WaveStart { sa, simd, wgp, cu, wave, id7 } => vec![
            ("sa", sa.to_string()),
            ("simd", simd.to_string()),
            ("wgp", wgp.to_string()),
            ("cu", cu.to_string()),
            ("wave", wave.to_string()),
            ("id7", id7.to_string()),
        ],
        Kind::WaveEnd { sa, simd, wgp, cu, wave } => vec![
            ("sa", sa.to_string()),
            ("simd", simd.to_string()),
            ("wgp", wgp.to_string()),
            ("cu", cu.to_string()),
            ("wave", wave.to_string()),
        ],
        Kind::WaveRdy { mask } | Kind::ImmediateMask { mask } => vec![("mask", mask.to_string())],
        Kind::Inst { wave, op, flag1, flag2, w64h } => {
            let mut f = vec![("wave", wave.to_string()), ("op", op_name(op))];
            if rdna4 {
                f.push(("w64h", b(w64h)));
            } else {
                f.extend([("flag1", b(flag1)), ("flag2", b(flag2))]);
            }
            f
        }
        Kind::ValuInst { wave, flag } => vec![("wave", wave.to_string()), ("flag", b(flag))],
        Kind::Immediate { wave } => vec![("wave", wave.to_string())],
        Kind::AluExec { src } => vec![("src", src.name().into())],
        Kind::VmemExec { src } => vec![("src", src.name().into())],
        Kind::Event { event, .. } => vec![("event", event.to_string())],
        Kind::Reg { slot, hi_byte, subop, val32 } => vec![
            ("slot", slot.to_string()),
            ("hi_byte", hi_byte.to_string()),
            ("subop", subop.to_string()),
            ("val32", val32.to_string()),
        ],
        Kind::Snapshot { snap } => vec![("snap", snap.to_string())],
        Kind::UtilCtr { ctr } => vec![("ctr", ctr.to_string())],
        Kind::Perf { arg } => vec![("arg", arg.to_string())],
        Kind::TsWaveState { coarse } => vec![("coarse", coarse.to_string())],
        Kind::ShaderDataShort { sa, simd, wgp, cu, wave, data, private } => vec![
            ("sa", sa.to_string()),
            ("simd", simd.to_string()),
            ("wgp", wgp.to_string()),
            ("cu", cu.to_string()),
            ("wave", wave.to_string()),
            ("data", data.to_string()),
            ("private", b(private)),
        ],
        Kind::TsDeltaShort { delta }
        | Kind::TsDeltaS5W2 { delta }
        | Kind::TsDeltaS5W3 { delta }
        | Kind::TsDeltaS8W3 { delta } => vec![("delta", delta.to_string())],
        Kind::TsDeltaOrMark { delta, pl, rt, tl } => {
            let mut f = vec![("delta", delta.to_string()), ("pl", b(pl)), ("rt", b(rt))];
            if rdna4 {
                f.push(("tl", b(tl)));
            }
            f
        }
    };
    f.into_iter().collect()
}

/// Decodes `data`, pairing each packet with the layout it was decoded under.
fn decode_with_layout(data: &[u8]) -> Vec<(Packet, Layout)> {
    let mut decoder = decode(data);
    let mut out = vec![];
    let mut layout = decoder.layout();
    while let Some(packet) = decoder.next() {
        out.push((packet, layout));
        layout = decoder.layout();
    }
    assert_eq!(decoder.unsupported_layout(), None);
    out
}

fn sha256(lines: impl Iterator<Item = String>) -> String {
    let mut hasher = Sha256::new();
    lines.for_each(|line| hasher.update(line));
    hasher.finalize().iter().map(|byte| format!("{byte:02x}")).collect()
}

fn json_fields(value: &Value) -> BTreeMap<&str, String> {
    let to_string = |v: &Value| v.as_str().map_or_else(|| v.to_string(), str::to_string);
    value.as_object().expect("fields object").iter().map(|(k, v)| (k.as_str(), to_string(v))).collect()
}

fn fixture(arch: &str, example: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/test/fixtures/sqtt").join(arch).join(example)
}

#[test_case("gfx1100", "empty", Layout::Rdna3)]
#[test_case("gfx1100", "gemm", Layout::Rdna3)]
#[test_case("gfx1100", "plus", Layout::Rdna3)]
#[test_case("gfx1100", "sync", Layout::Rdna3)]
#[test_case("gfx1200", "empty", Layout::Rdna4)]
#[test_case("gfx1200", "gemm", Layout::Rdna4)]
#[test_case("gfx1200", "handwritten", Layout::Rdna4)]
#[test_case("gfx1200", "plus", Layout::Rdna4)]
#[test_case("gfx1200", "sync", Layout::Rdna4)]
fn matches_tinygrad(arch: &str, example: &str, layout: Layout) {
    let dir = fixture(arch, example);
    let golden: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("golden.json")).unwrap()).unwrap();
    let mut decoded = BTreeMap::new();
    for event in golden["events"].as_array().unwrap() {
        let index = event["index"].as_u64().unwrap();
        let data = std::fs::read(dir.join(format!("trace{index}.bin"))).unwrap();
        assert_eq!(data.len() as u64, event["bytes"].as_u64().unwrap());
        let packets = decode_with_layout(&data);
        let ctx = format!("{arch}/{example} event {index}");
        assert_eq!(packets.len() as u64, event["packets"].as_u64().unwrap(), "{ctx}");
        assert!(matches!(packets[0].0.kind, Kind::LayoutHeader { .. }), "{ctx}");
        assert!(packets[1..].iter().all(|(_, l)| *l == layout), "{ctx}");
        assert_eq!(packets.last().map_or(0, |(p, _)| p.time), event["last_time"].as_u64().unwrap(), "{ctx}");
        assert!(packets.windows(2).all(|w| w[0].0.time <= w[1].0.time), "{ctx}: time decreased");

        let mut kinds = BTreeMap::new();
        packets.iter().for_each(|(p, l)| *kinds.entry(tinygrad_name(&p.kind, *l)).or_insert(0u64) += 1);
        let golden_kinds: BTreeMap<String, u64> =
            event["kinds"].as_object().unwrap().iter().map(|(k, v)| (k.clone(), v.as_u64().unwrap())).collect();
        assert_eq!(kinds, golden_kinds, "{ctx}");

        let kind_time = sha256(packets.iter().map(|(p, l)| format!("{} {}\n", tinygrad_name(&p.kind, *l), p.time)));
        assert_eq!(kind_time, event["sha256_kind_time"].as_str().unwrap(), "{ctx}");
        let all_fields = sha256(packets.iter().map(|(p, l)| {
            let f = fields(&p.kind, *l).into_iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join(" ");
            format!("{} {} {f}\n", tinygrad_name(&p.kind, *l), p.time)
        }));
        assert_eq!(all_fields, event["sha256_fields"].as_str().unwrap(), "{ctx}");

        for (i, head) in event["head"].as_array().unwrap().iter().enumerate() {
            let (p, l) = &packets[i];
            assert_eq!(tinygrad_name(&p.kind, *l), head[0].as_str().unwrap(), "{ctx} packet {i}");
            assert_eq!(p.time, head[1].as_u64().unwrap(), "{ctx} packet {i}");
            assert_eq!(fields(&p.kind, *l), json_fields(&head[2]), "{ctx} packet {i}");
        }

        if event["itrace"].as_bool().unwrap() {
            waves_pair(&packets, &ctx);
        }
        decoded.insert(index, packets);
    }

    // Every packet tinygrad's mapper attributes to an instruction exists with that wave.
    for mapped in golden["mapped"].as_array().unwrap() {
        let (index, name, time, wave) = (
            mapped[0].as_u64().unwrap(),
            mapped[1].as_str().unwrap(),
            mapped[2].as_u64().unwrap(),
            mapped[3].as_u64().unwrap(),
        );
        let packets = &decoded[&index];
        let found = packets.iter().any(|(p, l)| {
            p.time == time
                && tinygrad_name(&p.kind, *l) == name
                && match p.kind {
                    Kind::Inst { wave: w, op, .. } => u64::from(w) == wave && !op.is_other_simd(),
                    Kind::ValuInst { wave: w, .. } | Kind::Immediate { wave: w } => u64::from(w) == wave,
                    Kind::ImmediateMask { mask } => mask >> wave & 1 == 1,
                    _ => false,
                }
        });
        assert!(found, "{arch}/{example}: mapped {mapped} has no packet");
    }
}

/// On the instruction-traced SE every wave that starts on a unit ends before another starts there.
fn waves_pair(packets: &[(Packet, Layout)], ctx: &str) {
    let mut live = HashSet::new();
    for (p, _) in packets {
        match p.kind {
            Kind::WaveStart { cu, simd, wave, .. } => assert!(live.insert((cu, simd, wave)), "{ctx}: {p:?} twice"),
            Kind::WaveEnd { cu, simd, wave, .. } => assert!(live.remove(&(cu, simd, wave)), "{ctx}: {p:?} unstarted"),
            _ => {}
        }
    }
    assert!(live.is_empty(), "{ctx}: waves never ended: {live:?}");
}

/// Packs `(value, nibbles)` packets into a little-endian nibble stream, zero (NOP) padded so the
/// decoder's 16-nibble window reaches past the last packet.
fn stream(packets: &[(u64, usize)]) -> Vec<u8> {
    let mut nibbles: Vec<u8> =
        packets.iter().flat_map(|&(v, n)| (0..n).map(move |i| (v >> (4 * i) & 0xf) as u8)).collect();
    nibbles.resize(nibbles.len() + 32 + nibbles.len() % 2, 0);
    nibbles.chunks(2).map(|pair| pair[0] | pair[1] << 4).collect()
}

fn header(layout: u64) -> (u64, usize) {
    (0b0010001 | layout << 7, 16)
}

fn timed(data: &[u8]) -> Vec<Packet> {
    decode(data).filter(|p| p.kind != Kind::Nop).collect()
}

#[test_case(3, Layout::Rdna3)]
#[test_case(4, Layout::Rdna4)]
fn layout_header_selects_tables(code: u64, layout: Layout) {
    let data = stream(&[header(code)]);
    let mut decoder = decode(&data);
    let first = decoder.next().unwrap();
    assert_eq!(first.time, 0);
    assert!(matches!(first.kind, Kind::LayoutHeader { layout: l, .. } if u64::from(l) == code));
    assert_eq!(decoder.layout(), layout);
    assert!(decoder.all(|p| p.kind == Kind::Nop));
}

#[test]
fn foreign_layout_stops_decoding() {
    let data = stream(&[header(5)]);
    let mut decoder = decode(&data);
    assert_eq!(decoder.next(), None);
    assert_eq!(decoder.unsupported_layout(), Some(5));
    assert_eq!(decoder.next(), None);
}

#[test]
fn delta_short_adds_four() {
    let packets = timed(&stream(&[header(3), (0b1000 | 3 << 4, 2)]));
    assert_eq!(packets[1], Packet { time: 7, kind: Kind::TsDeltaShort { delta: 3 } });
}

// RDNA3 puts `pl` at bit 8 and `rt` at bit 9, RDNA4 `rt` at bit 7 and `pl` at bit 8.
#[test_case(3, 1 << 9, 12, 0 ; "rdna3 marker")]
#[test_case(3, 1 << 8 | 1 << 9, 12, 100 ; "rdna3 pl set")]
#[test_case(3, 0, 12, 100 ; "rdna3 delta")]
#[test_case(4, 1 << 7, 16, 0 ; "rdna4 marker")]
#[test_case(4, 1 << 9, 16, 100 ; "rdna4 tl only")]
fn delta_or_mark(layout: u64, flags: u64, nibbles: usize, time: u64) {
    let packets = timed(&stream(&[header(layout), (0b0000001 | flags | 100 << 12, nibbles), (0b1000, 2)]));
    assert!(matches!(packets[1].kind, Kind::TsDeltaOrMark { delta: 100, .. }));
    assert_eq!(packets[1].time, time);
    assert_eq!(packets[2].time, time + 4);
}

fn inst(layout: u64, wave: u64, delta: u64) -> (u64, usize) {
    let raw = if layout == 4 { 0b010 | delta << 3 | wave << 7 } else { 0b010 | delta << 4 | wave << 8 };
    (raw, 5)
}

fn waves(packets: &[Packet]) -> Vec<(u64, u8)> {
    packets
        .iter()
        .filter_map(|p| if let Kind::Inst { wave, .. } = p.kind { Some((p.time, wave)) } else { None })
        .collect()
}

// AMD's `NEW_PC` is 64 bits on gfx11 and 72 on gfx12 (rocprof-trace-decoder `gfx10token.cpp`,
// `gfx12token.cpp`); tinygrad sizes it 18 nibbles on both and, past the 64-bit window, lands two
// nibbles short, which desynchronised its decoding of AMD's navi4_cwsr/setpc/sdata traces.
#[test_case(3, 16 ; "rdna3")]
#[test_case(4, 18 ; "rdna4")]
fn new_pc_spans_its_width(layout: u64, nibbles: usize) {
    // Ones above the delta, so a misaligned window decodes other packets.
    let new_pc = 0b0100001 | 2 << 8 | !0u64 << 11;
    let mut packets = vec![header(layout), (new_pc, 16)];
    if nibbles > 16 {
        packets.push((0xff, nibbles - 16));
    }
    packets.extend((0..8).map(|i| inst(layout, i + 1, 1)));
    let decoded = timed(&stream(&packets));
    assert_eq!(decoded[1], Packet { time: 2, kind: Kind::TsDeltaS8W3 { delta: 2 } });
    assert_eq!(waves(&decoded[2..]), (0..8).map(|i| (3 + i, i as u8 + 1)).collect::<Vec<_>>());
    assert_eq!(decoded.len(), 10);
}

#[test]
fn rdna4_shader_data_short() {
    let raw = 0b1000110 | 1 << 7 | 1 << 10 | 2 << 11 | 3 << 13 | 4 << 17 | 0xab << 22 | 1 << 30;
    let decoded = timed(&stream(&[header(4), (raw, 8), inst(4, 7, 2)]));
    assert_eq!(
        decoded[1],
        Packet {
            time: 1,
            kind: Kind::ShaderDataShort { sa: 1, simd: 2, wgp: 3, cu: 19, wave: 4, data: 0xab, private: true }
        }
    );
    assert_eq!(waves(&decoded[2..]), [(3, 7)]);
    // RDNA3 has no 7-bit split: the low five bits make it a 13-nibble `TS_DELTA_S5_W3`.
    let decoded = timed(&stream(&[header(3), (raw, 13), inst(3, 7, 2)]));
    assert!(matches!(decoded[1].kind, Kind::TsDeltaS5W3 { .. }), "{:?}", decoded[1]);
    assert_eq!(waves(&decoded[2..]), [(decoded[1].time + 2, 7)]);
}

#[test_case(Layout::Rdna3, 0b0000_0010 | 5 << 8 | 0x03 << 13, 5, true ; "rdna3 jump")]
#[test_case(Layout::Rdna4, 0b0000_0010 | 5 << 7 | 0x03 << 12, 5, true ; "rdna4 jump")]
#[test_case(Layout::Rdna3, 0b0000_0010 | 9 << 8 | 0x04 << 13, 9, false ; "rdna3 jump_no")]
fn inst_fields(layout: Layout, raw: u64, wave: u8, jump: bool) {
    let code = if layout == Layout::Rdna4 { 4 } else { 3 };
    let packets = timed(&stream(&[header(code), (raw, 5)]));
    let Kind::Inst { wave: w, op, .. } = packets[1].kind else { panic!("{:?}", packets[1]) };
    assert_eq!((w, op.layout, op.is_jump(), op.is_jump_no()), (wave, layout, jump, !jump));
}

#[test]
fn wave_start_end_cu() {
    // RDNA4: start sa=1 simd=2 wgp=5 wave=7; end sa=1 simd=3 wgp=9 wave=4.
    let start = 0b01100 | 1 << 7 | 2 << 8 | 5 << 10 | 7 << 15;
    let end = 0b10101 | 1 << 8 | 3 << 9 | 9 << 11 | 4 << 15;
    let packets = timed(&stream(&[header(4), (start, 8), (end, 5)]));
    assert_eq!(packets[1].kind, Kind::WaveStart { sa: 1, simd: 2, wgp: 5, cu: 21, wave: 7, id7: 0 });
    assert_eq!(packets[2].kind, Kind::WaveEnd { sa: 1, simd: 3, wgp: 9, cu: 25, wave: 4 });
}

#[test_case(Layout::Rdna3, 0x00, Some("SALU"))]
#[test_case(Layout::Rdna3, 0x13, Some("BARRIER"))]
#[test_case(Layout::Rdna3, 0x50, Some("OTHER_LDS_1"))]
#[test_case(Layout::Rdna3, 0x55, Some("OTHER_FLAT_2"))]
#[test_case(Layout::Rdna3, 0x66, Some("OTHER_VMEM_13"))]
#[test_case(Layout::Rdna3, 0x67, None)]
#[test_case(Layout::Rdna3, 0x8c, None)]
#[test_case(Layout::Rdna4, 0x13, Some("BARRIER_WAIT"))]
#[test_case(Layout::Rdna4, 0x77, Some("OTHER_LDS_6"))]
#[test_case(Layout::Rdna4, 0x8c, Some("WMMA_8"))]
#[test_case(Layout::Rdna4, 0x5a, None)]
#[test_case(Layout::Rdna4, 0xbc, Some("OTHER_VMEM_1"))]
#[test_case(Layout::Rdna4, 0xdd, Some("OTHER_VMEM_34"))]
fn inst_op_names(layout: Layout, code: u8, name: Option<&str>) {
    let op = InstOp { layout, code };
    assert_eq!(op.name().as_deref(), name);
    assert_eq!(op.is_other_simd(), name.is_some_and(|n| n.starts_with("OTHER_")));
}

proptest! {
    #[test]
    fn arbitrary_bytes_decode_monotonically(data in prop::collection::vec(any::<u8>(), 0..4096)) {
        let times: Vec<u64> = decode(&data).map(|p| p.time).collect();
        prop_assert!(times.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn arbitrary_packets_after_header_decode_monotonically(
        layout in 3u64..=4,
        body in prop::collection::vec(any::<u8>(), 0..4096),
    ) {
        let mut data = stream(&[header(layout)]);
        data.truncate(8);
        data.extend(body);
        let mut decoder = decode(&data);
        let mut last = 0;
        for packet in decoder.by_ref() {
            prop_assert!(packet.time >= last);
            last = packet.time;
        }
        prop_assert!(decoder.unsupported_layout().is_none_or(|l| l != 3 && l != 4));
    }
}
