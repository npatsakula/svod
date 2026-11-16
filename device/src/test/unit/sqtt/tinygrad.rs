//! tinygrad's recorded example traces: our token boundaries and times against its goldens, and the wave
//! model plus stitcher against the PCs its mapper attributes.

use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;

use serde_json::Value;
use sha2::{Digest, Sha256};
use test_case::test_case;

use crate::sqtt::report::Report;
use crate::sqtt::stitch::{Stitcher, objdump_lines};
use crate::sqtt::{Kind, Layout, Packet, decode, wave};

/// A tinygrad packet class: `(name, encoding, width, delta (lo, width) per layout, RDNA4 only)`.
type Class = (&'static str, u8, u8, [(u8, u8); 2], bool);

#[rustfmt::skip]
const CLASSES: [Class; 25] = [
    ("VALUINST", 0b011, 3, [(3, 3), (3, 3)], false),
    ("VMEMEXEC", 0b1111, 4, [(4, 2), (4, 2)], false),
    ("ALUEXEC", 0b1110, 4, [(4, 2), (4, 2)], false),
    ("IMMEDIATE", 0b1101, 4, [(4, 3), (4, 3)], false),
    ("IMMEDIATE_MASK", 0b00100, 5, [(5, 3), (5, 3)], false),
    ("WAVERDY", 0b10100, 5, [(5, 3), (5, 3)], false),
    ("TS_DELTA_S8_W3", 0b0100001, 7, [(8, 3), (8, 3)], false),
    ("WAVEEND", 0b10101, 5, [(5, 3), (5, 3)], false),
    ("WAVESTART", 0b01100, 5, [(5, 2), (5, 2)], false),
    ("TS_DELTA_S5_W2", 0b11100, 5, [(5, 2), (5, 2)], false),
    ("WAVEALLOC", 0b00101, 5, [(5, 3), (5, 3)], false),
    ("TS_DELTA_S5_W3", 0b00110, 5, [(5, 3), (7, 3)], false),
    ("PERF", 0b10110, 5, [(5, 3), (7, 3)], false),
    ("UTILCTR", 0b0110001, 7, [(7, 2), (7, 2)], false),
    ("TS_DELTA_SHORT", 0b1000, 4, [(4, 4), (4, 4)], false),
    ("NOP", 0b0000, 4, [(0, 0), (0, 0)], false),
    ("TS_WAVE_STATE", 0b1010001, 7, [(7, 9), (7, 9)], false),
    ("EVENT", 0b01100001, 8, [(8, 3), (8, 3)], false),
    ("EVENT_BIG", 0b11100001, 8, [(8, 3), (8, 3)], false),
    ("REG", 0b1001, 4, [(4, 3), (4, 3)], false),
    ("SNAPSHOT", 0b1110001, 7, [(7, 3), (7, 3)], false),
    ("TS_DELTA_OR_MARK", 0b0000001, 7, [(12, 36), (12, 52)], false),
    ("LAYOUT_HEADER", 0b0010001, 7, [(0, 0), (0, 0)], false),
    ("INST", 0b010, 3, [(4, 3), (3, 3)], false),
    ("SHADER_DATA_SHORT", 0b1000110, 7, [(0, 0), (7, 3)], true),
];

const RDNA4_SUFFIXED: [&str; 8] =
    ["WAVESTART", "WAVEEND", "WAVEALLOC", "INST", "PERF", "TS_DELTA_S5_W2", "TS_DELTA_S5_W3", "TS_DELTA_OR_MARK"];

#[rustfmt::skip]
const RDNA3_OPS: &[(u8, &str)] = &[
    (0x00, "SALU"), (0x01, "SMEM_RD"), (0x03, "JUMP"), (0x04, "JUMP_NO"), (0x05, "CALL"), (0x09, "MESSAGE"),
    (0x0b, "VALUT_4"), (0x0d, "VALUB_2"), (0x0e, "VALUB_4"), (0x0f, "VALUB_16"), (0x12, "VINTERP"), (0x13, "BARRIER"),
    (0x1c, "FLAT_RD_2"), (0x1d, "FLAT_WR_3"), (0x1e, "FLAT_WR_4"), (0x1f, "FLAT_WR_5"), (0x20, "FLAT_WR_6"),
    (0x21, "SGMEM_RD_1"), (0x22, "SGMEM_RD_2"), (0x24, "SGMEM_WR_2"), (0x25, "SGMEM_WR_3"), (0x26, "SGMEM_WR_4"),
    (0x27, "SGMEM_WR_5"), (0x28, "SGMEM_WR_6"), (0x29, "LDS_RD"), (0x2a, "LDS_WR_1"), (0x2b, "LDS_WR_2"),
    (0x2c, "LDS_WR_3"), (0x2d, "LDS_WR_4"), (0x2e, "LDS_WR_5"), (0x72, "SALU_WR_EXEC"), (0x73, "VALU1_WR_EXEC"),
];

#[rustfmt::skip]
const RDNA4_OPS: &[(u8, &str)] = &[
    (0x00, "SALU"), (0x01, "SMEM"), (0x02, "SMEM_WR"), (0x03, "JUMP"), (0x04, "JUMP_NO"), (0x05, "CALL"),
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

/// Ops another SIMD executes, `(category, first, last, base)`, named `OTHER_{category}_{code - first + base}`.
const RDNA3_OTHER: &[(&str, u8, u8, u8)] = &[("LDS", 0x50, 0x54, 1), ("FLAT", 0x55, 0x59, 2), ("VMEM", 0x5a, 0x66, 1)];
const RDNA4_OTHER: &[(&str, u8, u8, u8)] = &[("LDS", 0x50, 0x54, 1), ("FLAT", 0x55, 0x59, 2), ("VMEM", 0xbc, 0xdd, 1)];

/// tinygrad's `InstOp`/`InstOpRDNA4` member name, or the code when neither knows it.
fn op_name(code: u8, rdna4: bool) -> String {
    let (named, other) = if rdna4 { (RDNA4_OPS, RDNA4_OTHER) } else { (RDNA3_OPS, RDNA3_OTHER) };
    if let Some((_, name)) = named.iter().find(|(c, _)| *c == code) {
        return (*name).into();
    }
    other
        .iter()
        .find(|(_, lo, hi, _)| (*lo..=*hi).contains(&code))
        .map_or_else(|| code.to_string(), |(cat, lo, _, base)| format!("OTHER_{cat}_{}", code - lo + base))
}

fn bits(r: u64, hi: u32, lo: u32) -> u64 {
    (r >> lo) & (u64::MAX >> (63 - (hi - lo)))
}

/// tinygrad's class name and the fields it declares, sorted by name, decoded from the packet's raw bits.
fn tinygrad(raw: u64, layout: Layout) -> (String, BTreeMap<&'static str, u64>) {
    let rdna4 = layout == Layout::Rdna4;
    let (name, _, _, delta, _) = CLASSES
        .iter()
        .filter(|(_, enc, width, _, only4)| (rdna4 || !only4) && raw & ((1 << width) - 1) == u64::from(*enc))
        .max_by_key(|(_, _, width, _, _)| *width)
        .copied()
        .expect("NOP matches everything");
    let b = |hi, lo| bits(raw, hi, lo);
    let cu = |wgp: u64, sa: u64| wgp | sa << if rdna4 { 4 } else { 3 };
    let f: Vec<(&str, u64)> = match name {
        "LAYOUT_HEADER" => vec![
            ("layout", b(12, 7)),
            ("simd", b(14, 13)),
            ("group", b(17, 15)),
            ("sel_a", b(31, 28)),
            ("sel_b", b(36, 33)),
            ("flag4", b(59, 59)),
        ],
        "WAVESTART" => {
            let (wgp, wave, id7) =
                if rdna4 { (b(13, 10), b(19, 15), b(31, 20)) } else { (b(12, 10), b(17, 13), b(31, 18)) };
            let sa = b(7, 7);
            vec![("sa", sa), ("simd", b(9, 8)), ("wgp", wgp), ("cu", cu(wgp, sa)), ("wave", wave), ("id7", id7)]
        }
        "WAVEEND" => {
            let (wgp, sa) = (if rdna4 { b(14, 11) } else { b(13, 11) }, b(8, 8));
            vec![("sa", sa), ("simd", b(10, 9)), ("wgp", wgp), ("cu", cu(wgp, sa)), ("wave", b(19, 15))]
        }
        "WAVERDY" | "IMMEDIATE_MASK" => vec![("mask", b(23, 8))],
        "INST" if rdna4 => vec![("wave", b(11, 7)), ("op", b(19, 12)), ("w64h", b(6, 6))],
        "INST" => vec![("wave", b(12, 8)), ("op", b(19, 13)), ("flag1", b(3, 3)), ("flag2", b(7, 7))],
        "VALUINST" => vec![("wave", b(11, 7)), ("flag", b(6, 6))],
        "IMMEDIATE" => vec![("wave", b(11, 7))],
        "ALUEXEC" | "VMEMEXEC" => vec![("src", b(7, 6))],
        "EVENT" => vec![("event", b(23, 11))],
        "EVENT_BIG" => vec![("event", b(31, 11))],
        "REG" => vec![("slot", b(9, 7)), ("hi_byte", b(15, 8)), ("subop", b(31, 16)), ("val32", b(63, 32))],
        "SNAPSHOT" => vec![("snap", b(63, 10))],
        "UTILCTR" => vec![("ctr", b(47, 9))],
        "PERF" => vec![("arg", if rdna4 { b(31, 10) } else { b(27, 8) })],
        "TS_WAVE_STATE" => vec![("coarse", b(23, 16))],
        "SHADER_DATA_SHORT" => {
            let (wgp, sa) = (b(16, 13), b(10, 10));
            vec![
                ("sa", sa),
                ("simd", b(12, 11)),
                ("wgp", wgp),
                ("cu", cu(wgp, sa)),
                ("wave", b(21, 17)),
                ("data", b(29, 22)),
                ("private", u64::from(b(31, 30) != 0)),
            ]
        }
        "TS_DELTA_OR_MARK" if rdna4 => vec![("rt", b(7, 7)), ("pl", b(8, 8)), ("tl", b(9, 9))],
        "TS_DELTA_OR_MARK" => vec![("pl", b(8, 8)), ("rt", b(9, 9))],
        _ => vec![],
    };
    let mut fields: BTreeMap<_, _> = f.into_iter().collect();
    if name.starts_with("TS_DELTA") {
        let (lo, width) = delta[usize::from(rdna4)];
        fields.insert("delta", b(u32::from(lo + width - 1), u32::from(lo)));
    }
    let name = if rdna4 && RDNA4_SUFFIXED.contains(&name) { format!("{name}_RDNA4") } else { name.into() };
    (name, fields)
}

fn render(name: &str, fields: &BTreeMap<&str, u64>, rdna4: bool) -> BTreeMap<String, String> {
    let value = |k: &str, v: u64| match (name, k) {
        (_, "op") => op_name(v as u8, rdna4),
        ("ALUEXEC", "src") => ["NONE", "SALU", "VALU", "VALU_SALU"][v as usize].into(),
        ("VMEMEXEC", "src") => ["LDS", "LDS_ALT", "VMEM", "VMEM_ALT"][v as usize].into(),
        _ => v.to_string(),
    };
    fields.iter().map(|(k, v)| ((*k).to_string(), value(k, *v))).collect()
}

/// Decodes `data`, pairing each packet with the layout it was decoded under.
fn decode_with_layout(data: &[u8]) -> Vec<(Packet, Layout)> {
    let mut layout = Layout::Rdna3;
    let mut decoder = decode(data);
    let packets = decoder
        .by_ref()
        .map(|p| {
            let at = layout;
            if let Kind::Header(Ok(h)) = p.kind {
                layout = h.layout;
            }
            (p, at)
        })
        .collect();
    assert_eq!(decoder.unsupported_layout(), None);
    packets
}

fn sha256(lines: impl Iterator<Item = String>) -> String {
    let mut hasher = Sha256::new();
    lines.for_each(|line| hasher.update(line));
    hasher.finalize().iter().map(|byte| format!("{byte:02x}")).collect()
}

fn json_fields(value: &Value) -> BTreeMap<String, String> {
    let to_string = |v: &Value| v.as_str().map_or_else(|| v.to_string(), str::to_string);
    value.as_object().expect("fields object").iter().map(|(k, v)| (k.clone(), to_string(v))).collect()
}

fn fixture(arch: &str, example: &str) -> (PathBuf, Value) {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/test/fixtures/sqtt").join(arch).join(example);
    let golden = serde_json::from_str(&std::fs::read_to_string(dir.join("golden.json")).unwrap()).unwrap();
    (dir, golden)
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
    let (dir, golden) = fixture(arch, example);
    let rdna4 = layout == Layout::Rdna4;
    for event in golden["events"].as_array().unwrap() {
        let index = event["index"].as_u64().unwrap();
        let data = std::fs::read(dir.join(format!("trace{index}.bin"))).unwrap();
        assert_eq!(data.len() as u64, event["bytes"].as_u64().unwrap());
        let ctx = format!("{arch}/{example} event {index}");
        let packets = decode_with_layout(&data);
        // tinygrad stops once its 64-bit window leaves the data; AMD decodes the tail packets.
        let count = event["packets"].as_u64().unwrap() as usize;
        assert!((count..count + 16).contains(&packets.len()), "{ctx}: {} packets", packets.len());
        let packets = &packets[..count];
        assert!(matches!(packets[0].0.kind, Kind::Header(Ok(_))), "{ctx}");
        assert!(packets[1..].iter().all(|(_, l)| *l == layout), "{ctx}");
        assert_eq!(packets.last().unwrap().0.time, event["last_time"].as_u64().unwrap(), "{ctx}");

        let named: Vec<_> = packets.iter().map(|(p, l)| (p.time, tinygrad(p.raw, *l))).collect();
        let mut kinds = BTreeMap::new();
        named.iter().for_each(|(_, (name, _))| *kinds.entry(name.clone()).or_insert(0u64) += 1);
        let golden_kinds: BTreeMap<String, u64> =
            event["kinds"].as_object().unwrap().iter().map(|(k, v)| (k.clone(), v.as_u64().unwrap())).collect();
        assert_eq!(kinds, golden_kinds, "{ctx}");

        let kind_time = sha256(named.iter().map(|(time, (name, _))| format!("{name} {time}\n")));
        assert_eq!(kind_time, event["sha256_kind_time"].as_str().unwrap(), "{ctx}");
        let all_fields = sha256(named.iter().map(|(time, (name, fields))| {
            let f = render(name, fields, rdna4).into_iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>();
            format!("{name} {time} {}\n", f.join(" "))
        }));
        assert_eq!(all_fields, event["sha256_fields"].as_str().unwrap(), "{ctx}");

        for (i, head) in event["head"].as_array().unwrap().iter().enumerate() {
            let (time, (name, fields)) = &named[i];
            assert_eq!(name, head[0].as_str().unwrap(), "{ctx} packet {i}");
            assert_eq!(*time, head[1].as_u64().unwrap(), "{ctx} packet {i}");
            assert_eq!(render(name, fields, rdna4), json_fields(&head[2]), "{ctx} packet {i}");
        }
        if event["itrace"].as_bool().unwrap() {
            waves_pair(packets, &ctx);
        }
    }
}

/// On an instruction-traced SE every wave that starts on a slot ends before another starts there.
fn waves_pair(packets: &[(Packet, Layout)], ctx: &str) {
    let mut live = HashSet::new();
    for (p, _) in packets {
        match p.kind {
            Kind::WaveStart(start) => assert!(live.insert(start.unit), "{ctx}: {p:?} twice"),
            Kind::WaveEnd(unit) => assert!(live.remove(&unit), "{ctx}: {p:?} unstarted"),
            _ => {}
        }
    }
    assert!(live.is_empty(), "{ctx}: waves never ended: {live:?}");
}

/// Stitching tinygrad's kernels attributes every record, and to each PC as many as tinygrad's mapper
/// does; `s_endpgm` emits no token, and AMD synthesises at most one per wave that ran instructions.
#[test_case("gfx1100", "empty")]
#[test_case("gfx1100", "gemm")]
#[test_case("gfx1100", "plus")]
#[test_case("gfx1200", "empty")]
#[test_case("gfx1200", "gemm")]
#[test_case("gfx1200", "plus")]
fn stitches_tinygrad(arch: &str, example: &str) {
    let (dir, golden) = fixture(arch, example);
    let base = golden["base"].as_u64().unwrap();
    let dump = std::fs::read_to_string(dir.join("kernel.s")).unwrap();
    let code: Vec<(u64, &str)> = objdump_lines(&dump).into_iter().map(|(pc, line)| (pc + base, line)).collect();
    let endpgm = code.iter().find(|(_, line)| line.trim_start().starts_with("s_endpgm")).unwrap().0 - base;
    let mut ours = BTreeMap::new();
    for event in golden["events"].as_array().unwrap().iter().filter(|e| e["itrace"].as_bool().unwrap()) {
        let index = event["index"].as_u64().unwrap();
        let mut trace = wave::analyze(decode(&std::fs::read(dir.join(format!("trace{index}.bin"))).unwrap()));
        let mut stitcher = Stitcher::new(&code, trace.header.unwrap_or_default().layout);
        for wave in &mut trace.waves {
            assert_eq!(stitcher.stitch(wave), wave.instructions.len(), "{arch}/{example} event {index}");
        }
        let report = Report::new(&trace.waves);
        let ended = trace.waves.iter().filter(|w| w.complete && !w.instructions.is_empty()).count() as u64;
        assert!(report.pcs.get(&(endpgm + base)).is_none_or(|s| s.hits <= ended), "{arch}/{example} event {index}");
        ours.extend(
            report.pcs.into_iter().filter(|(pc, _)| pc - base != endpgm).map(|(pc, s)| ((index, pc - base), s.hits)),
        );
    }
    let mut mapped = BTreeMap::new();
    for m in golden["mapped"].as_array().unwrap() {
        *mapped.entry((m[0].as_u64().unwrap(), m[4].as_u64().unwrap())).or_insert(0) += 1;
    }
    assert_eq!(ours, mapped, "{arch}/{example}");
}
