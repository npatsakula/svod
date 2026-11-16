//! SQTT capture streams, arming and readback, host-only.

use std::sync::Arc;

use svod_dtype::AmdArch;
use test_case::test_case;

use super::test_support::{mock_device, mock_device_queues};
use crate::amd::connector::{OwnerCtx, SubmissionFinalizer};
use crate::amd::regwrite::RegWriter;
use crate::amd::sqtt::{Capture, SqttTopology, build_streams, check_target, cu_masks, current_dpm_mhz, trace_bytes};
use crate::amd::sys::pm4;
use crate::device::PlanContext;
use crate::error::Error;
use crate::profile::{CodeImage, SqttConfig, SqttSimds};
use crate::sync::DispatchTimestamps;

/// A PM4 packet of the profiler streams, register writes at absolute dwords.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Packet {
    Reg(u32, u32),
    WaitReg { reg: u32, mask: u32 },
    Event(u32),
    Copy { src: u32, dst: u64, control: u32 },
    Acquire,
}

pub(super) fn decode(stream: &[u32]) -> Vec<Packet> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < stream.len() {
        let header = stream[i];
        assert_eq!(header >> 30, 3, "PACKET3 at dword {i}");
        let (op, n) = ((header >> 8) & 0xFF, ((header >> 16) & 0x3FFF) as usize + 1);
        let body = &stream[i + 1..i + 1 + n];
        match op {
            pm4::PACKET3_SET_SH_REG | pm4::PACKET3_SET_UCONFIG_REG => {
                let window = if op == pm4::PACKET3_SET_SH_REG {
                    pm4::PACKET3_SET_SH_REG_START
                } else {
                    pm4::PACKET3_SET_UCONFIG_REG_START
                };
                out.extend(body[1..].iter().enumerate().map(|(k, &v)| Packet::Reg(window + body[0] + k as u32, v)));
            }
            pm4::PACKET3_WAIT_REG_MEM => {
                assert_eq!(body, [3, body[1], 0, 0, body[4], 4], "register == 0 wait");
                out.push(Packet::WaitReg { reg: body[1], mask: body[4] });
            }
            pm4::PACKET3_EVENT_WRITE => out.push(Packet::Event(body[0])),
            pm4::PACKET3_COPY_DATA => {
                out.push(Packet::Copy { src: body[1], dst: body[3] as u64 | (body[4] as u64) << 32, control: body[0] })
            }
            pm4::PACKET3_ACQUIRE_MEM => out.push(Packet::Acquire),
            _ => panic!("unexpected PM4 opcode {op:#x}"),
        }
        i += 1 + n;
    }
    out
}

const BUF: u64 = 0x0000_9234_5678_0000; // 48-bit, so gfx11 packs 0x9 into BUF0_SIZE.base_hi
const WPTRS: u64 = 0x0000_7000_0000_1000;
const WINDOW: usize = 1 << 20;

fn gfx(arch: AmdArch) -> RegWriter {
    RegWriter::for_arch(arch).unwrap()
}

fn topology(se: u32) -> SqttTopology {
    SqttTopology { se, cu_per_se: 10, cu_per_sa: 10 }
}

fn config(se_mask: u32) -> SqttConfig {
    SqttConfig { se_mask, simds: SqttSimds::default(), bytes_per_se: WINDOW, limit_cu: false, pstate: None }
}

fn streams(arch: AmdArch, se: u32, config: &SqttConfig) -> (Vec<Packet>, Vec<Packet>) {
    let (start, stop) = build_streams(config, arch, &topology(se), 64, BUF, WPTRS).unwrap();
    (decode(&start), decode(&stop))
}

/// Writes to `name` in stream order.
pub(super) fn writes(gc: &RegWriter, packets: &[Packet], name: &str) -> Vec<u32> {
    let addr = gc.addr(name);
    packets
        .iter()
        .filter_map(|p| match p {
            Packet::Reg(a, v) if *a == addr => Some(*v),
            _ => None,
        })
        .collect()
}

/// Byte addresses agree with Mesa's `R_xxxxxx` register names.
#[test_case(AmdArch::Gfx1151, "regSQ_THREAD_TRACE_BUF0_BASE", 0x367A0)]
#[test_case(AmdArch::Gfx1151, "regSQ_THREAD_TRACE_BUF0_SIZE", 0x367A4)]
#[test_case(AmdArch::Gfx1151, "regSQ_THREAD_TRACE_WPTR", 0x367BC)]
#[test_case(AmdArch::Gfx1151, "regSQ_THREAD_TRACE_STATUS", 0x367D0)]
#[test_case(AmdArch::Gfx1100, "regGRBM_GFX_INDEX", 0x30800)]
#[test_case(AmdArch::Gfx1201, "regSQ_THREAD_TRACE_BUF0_SIZE", 0x36798)]
#[test_case(AmdArch::Gfx1201, "regSQ_THREAD_TRACE_BUF0_BASE_LO", 0x3679C)]
#[test_case(AmdArch::Gfx1201, "regSQ_THREAD_TRACE_BUF0_BASE_HI", 0x367A0)]
#[test_case(AmdArch::Gfx1201, "regSPI_CONFIG_CNTL", 0x31100)]
#[test_case(AmdArch::Gfx1201, "regCOMPUTE_THREAD_TRACE_ENABLE", 0xB878)]
#[test_case(AmdArch::Gfx942, "regGRBM_GFX_INDEX", 0x30800)]
fn registers_land_at_their_mesa_addresses(arch: AmdArch, name: &str, byte: u32) {
    assert_eq!(gfx(arch).addr(name) * 4, byte);
}

#[test_case(AmdArch::Gfx1151, 2)]
#[test_case(AmdArch::Gfx1201, 4)]
fn every_se_is_selected_and_programmed_in_order(arch: AmdArch, ses: u32) {
    let gc = gfx(arch);
    let (start, _) = streams(arch, ses, &config(0b11));
    let buf0: &[&str] = if arch.gfx_major() == 12 {
        &["regSQ_THREAD_TRACE_BUF0_SIZE", "regSQ_THREAD_TRACE_BUF0_BASE_LO", "regSQ_THREAD_TRACE_BUF0_BASE_HI"]
    } else {
        &["regSQ_THREAD_TRACE_BUF0_SIZE", "regSQ_THREAD_TRACE_BUF0_BASE"]
    };
    let per_se = ["regGRBM_GFX_INDEX"].into_iter().chain(buf0.iter().copied()).chain([
        "regSQ_THREAD_TRACE_MASK",
        "regSQ_THREAD_TRACE_TOKEN_MASK",
        "regSQ_THREAD_TRACE_CTRL",
    ]);
    let mut expected = vec!["acquire", "regSPI_CONFIG_CNTL"];
    for _ in 0..ses {
        expected.extend(per_se.clone());
    }
    expected.extend(["regGRBM_GFX_INDEX", "regCOMPUTE_THREAD_TRACE_ENABLE", "acquire"]);
    let named: Vec<&str> = start
        .iter()
        .map(|p| match p {
            Packet::Acquire => "acquire",
            Packet::Reg(a, _) => expected.iter().copied().find(|n| *n != "acquire" && gc.addr(n) == *a).unwrap(),
            other => panic!("unexpected start packet {other:?}"),
        })
        .collect();
    assert_eq!(named, expected);

    let grbm = writes(&gc, &start, "regGRBM_GFX_INDEX");
    let selected: Vec<u32> = (0..ses).map(|se| (1 << 30) | (se << 16)).collect();
    assert_eq!(grbm[..ses as usize], selected, "SE n, SA 0, every instance");
    assert_eq!(grbm[ses as usize], 0xE000_0000, "broadcast");
    assert_eq!(writes(&gc, &start, "regSPI_CONFIG_CNTL"), [spi(true)]);
    assert_eq!(writes(&gc, &start, "regCOMPUTE_THREAD_TRACE_ENABLE"), [1]);
}

fn spi(tracing: bool) -> u32 {
    0x2c688 | (3 << 21) | (3 << 30) | (tracing as u32) << 24 | (tracing as u32) << 25
}

#[test]
fn gfx11_buf0_packs_the_high_address_bits_into_size() {
    let gc = gfx(AmdArch::Gfx1151);
    let (start, _) = streams(AmdArch::Gfx1151, 2, &config(0b11));
    let sizes = writes(&gc, &start, "regSQ_THREAD_TRACE_BUF0_SIZE");
    let bases = writes(&gc, &start, "regSQ_THREAD_TRACE_BUF0_BASE");
    let window = WINDOW as u64;
    assert_eq!(sizes, [(256 << 8) | (BUF >> 44) as u32, (256 << 8) | ((BUF + window) >> 44) as u32]);
    assert_eq!(bases, [(BUF >> 12) as u32, ((BUF + window) >> 12) as u32]);
}

#[test]
fn gfx12_buf0_splits_the_base_over_two_registers() {
    let gc = gfx(AmdArch::Gfx1201);
    let (start, _) = streams(AmdArch::Gfx1201, 4, &config(0b11));
    let window = WINDOW as u64;
    let se_base = |se: u64| BUF + se * window;
    assert_eq!(writes(&gc, &start, "regSQ_THREAD_TRACE_BUF0_SIZE"), [256; 4]);
    assert_eq!(
        writes(&gc, &start, "regSQ_THREAD_TRACE_BUF0_BASE_LO"),
        (0..4).map(|se| (se_base(se) >> 12) as u32).collect::<Vec<_>>()
    );
    assert_eq!(
        writes(&gc, &start, "regSQ_THREAD_TRACE_BUF0_BASE_HI"),
        (0..4).map(|se| (se_base(se) >> 44) as u32).collect::<Vec<_>>()
    );
}

/// Only the SEs of `se_mask` record instruction tokens; gfx12 also leaves out
/// barrier waits. gfx11 sets no `ttrace_exec` (bit 11): its 11-bit `token_exclude` has no PERF bit.
#[test_case(AmdArch::Gfx1151, 0b01, &[0x1F_1000, 0x1F_1127]; "gfx11 se0")]
#[test_case(AmdArch::Gfx1151, 0b10, &[0x1F_1127, 0x1F_1000]; "gfx11 se1")]
#[test_case(AmdArch::Gfx1201, 0b0101, &[0x1F_6000, 0x1F_6927, 0x1F_6000, 0x1F_6927]; "gfx12 even")]
#[test_case(AmdArch::Gfx1201, 0, &[0x1F_6927; 4]; "gfx12 none")]
fn token_mask_follows_the_itrace_ses(arch: AmdArch, se_mask: u32, expected: &[u32]) {
    let gc = gfx(arch);
    let (start, _) = streams(arch, expected.len() as u32, &config(se_mask));
    assert_eq!(writes(&gc, &start, "regSQ_THREAD_TRACE_TOKEN_MASK"), expected);
}

/// MASK traces CS waves on each SE's SIMD of WGP 0, SA 0: by default SIMDs 0 and 1 (CU 0's)
/// alternate over the SEs; a list repeats over them.
#[test_case(AmdArch::Gfx1151, SqttSimds::default(), &[0, 1]; "gfx11 default")]
#[test_case(AmdArch::Gfx1201, SqttSimds::default(), &[0, 1, 0, 1]; "gfx12 default")]
#[test_case(AmdArch::Gfx1151, SqttSimds::cycle(&[3]), &[3, 3]; "gfx11 one simd")]
#[test_case(AmdArch::Gfx1201, SqttSimds::cycle(&[2, 3, 0]), &[2, 3, 0, 2]; "gfx12 list repeats")]
fn mask_traces_each_se_simd(arch: AmdArch, simds: SqttSimds, expected: &[u32]) {
    let gc = gfx(arch);
    let (start, _) = streams(arch, expected.len() as u32, &SqttConfig { simds, ..config(0b11) });
    let masks: Vec<u32> = expected.iter().map(|&simd| (0x40 << 10) | simd).collect();
    assert_eq!(writes(&gc, &start, "regSQ_THREAD_TRACE_MASK"), masks);
}

#[test_case("1" => Ok(SqttSimds([1; 8])); "one for every se")]
#[test_case(" 0, 1 " => Ok(SqttSimds::default()); "the default list")]
#[test_case("2,3,0" => Ok(SqttSimds([2, 3, 0, 2, 3, 0, 2, 3])); "repeated")]
#[test_case("" => Err("\"\": a SIMD or a comma list of up to 8".to_string()); "empty")]
#[test_case("0,x" => Err("\"0,x\": a SIMD or a comma list of up to 8".to_string()); "garbage")]
#[test_case("0,1,2,3,0,1,2,3,0" => Err("\"0,1,2,3,0,1,2,3,0\": a SIMD or a comma list of up to 8".to_string()); "past 8 ses")]
fn simds_parse(s: &str) -> Result<SqttSimds, String> {
    s.parse()
}

#[test_case("0: 1000Mhz \n1: 1800Mhz *\n2: 1800Mhz *\n" => Some(1800); "two marked")]
#[test_case("0: 96Mhz *\n1: 456Mhz\n" => Some(96); "lowest")]
#[test_case("0: 96Mhz\n1: 456Mhz\n" => None; "none marked")]
#[test_case("S: 19Mhz *\n" => Some(19); "deep sleep")]
#[test_case("0: fast *\n" => None; "garbage")]
fn dpm_level_parses(levels: &str) -> Option<u64> {
    current_dpm_mhz(levels)
}

/// CTRL starts each SE in mode 1 and the stop returns it to mode 0; gfx11
/// alone has `rt_freq`, and the two place `util_timer` differently.
#[test_case(AmdArch::Gfx1151, 0x8002_3C40)]
#[test_case(AmdArch::Gfx1201, 0x8000_5C40)]
fn trace_ctrl_toggles_mode(arch: AmdArch, idle: u32) {
    let gc = gfx(arch);
    let (start, stop) = streams(arch, 2, &config(0b11));
    assert_eq!(writes(&gc, &start, "regSQ_THREAD_TRACE_CTRL"), [idle | 1; 2]);
    assert_eq!(writes(&gc, &stop, "regSQ_THREAD_TRACE_CTRL"), [idle; 2]);
}

#[test_case(AmdArch::Gfx1151, 2)]
#[test_case(AmdArch::Gfx1201, 4)]
fn stop_drains_every_se_and_copies_its_write_pointer(arch: AmdArch, ses: u32) {
    let gc = gfx(arch);
    let (_, stop) = streams(arch, ses, &config(0b11));
    let (status, wptr) = (gc.addr("regSQ_THREAD_TRACE_STATUS"), gc.addr("regSQ_THREAD_TRACE_WPTR"));
    let grbm = gc.addr("regGRBM_GFX_INDEX");
    let ctrl = gc.addr("regSQ_THREAD_TRACE_CTRL");
    let idle = writes(&gc, &stop, "regSQ_THREAD_TRACE_CTRL")[0];
    let mut expected = vec![
        Packet::Event(pm4::THREAD_TRACE_MARKER),
        Packet::Acquire,
        Packet::Reg(grbm, 0xE000_0000),
        Packet::Reg(gc.addr("regCOMPUTE_THREAD_TRACE_ENABLE"), 0),
        Packet::Event(pm4::THREAD_TRACE_FINISH),
    ];
    for se in 0..ses {
        expected.extend([
            Packet::Reg(grbm, (1 << 30) | (se << 16)),
            Packet::WaitReg { reg: status, mask: 0xFFF },
            Packet::Reg(ctrl, idle),
            Packet::WaitReg { reg: status, mask: 1 << 25 },
            Packet::Event(pm4::CS_PARTIAL_FLUSH | (pm4::EVENT_INDEX_PARTIAL_FLUSH << 8)),
            Packet::Copy { src: wptr, dst: WPTRS + se as u64 * 4, control: (1 << 20) | (2 << 8) | 4 },
        ]);
    }
    expected.extend([
        Packet::Reg(grbm, 0xE000_0000),
        Packet::Reg(gc.addr("regSPI_CONFIG_CNTL"), spi(false)),
        Packet::Acquire,
    ]);
    assert_eq!(stop, expected);
}

/// `limit_cu` narrows the static thread masks after the start and gives every
/// CU back after the stop.
#[test]
fn limit_cu_narrows_and_restores_the_thread_masks() {
    let arch = AmdArch::Gfx1201;
    let gc = gfx(arch);
    let topology = topology(4);
    let cfg = SqttConfig { limit_cu: true, ..config(0b1) };
    let (start, stop) = build_streams(&cfg, arch, &topology, 100, BUF, WPTRS).unwrap();
    let (start, stop) = (decode(&start), decode(&stop));
    let masks = cu_masks(&topology, 100).unwrap();
    for (se, &mask) in masks.iter().enumerate() {
        let name = format!("regCOMPUTE_STATIC_THREAD_MGMT_SE{se}");
        assert_eq!(writes(&gc, &start, &name), [mask]);
        assert_eq!(writes(&gc, &stop, &name), [u32::MAX]);
    }
    assert!(
        matches!(stop.last(), Some(Packet::Reg(a, u32::MAX)) if *a == gc.addr("regCOMPUTE_STATIC_THREAD_MGMT_SE7"))
    );
}

/// Tinygrad's masks: `workgroups / (cu_per_se * 4)` CUs per SE, one more on
/// SE 0, within SA 0's `cu_per_sa / 2` bits.
#[test_case(10, 0 => [0b1, 0, 0, 0, 0, 0, 0, 0]; "one CU")]
#[test_case(10, 100 => [0b111, 0b11, 0b11, 0b11, 0b11, 0b11, 0b11, 0b11]; "two CUs per SE")]
#[test_case(4, 100_000 => [0b11; 8]; "clamped to the SA")]
fn cu_masks_follow_tinygrad(cu_per_sa: u32, workgroups: u64) -> [u32; 8] {
    cu_masks(&SqttTopology { se: 4, cu_per_se: 10, cu_per_sa }, workgroups).unwrap()
}

#[test]
fn limit_cu_needs_a_known_topology() {
    let unknown = SqttTopology { se: 1, cu_per_se: 1, cu_per_sa: 0 };
    assert!(matches!(cu_masks(&unknown, 1), Err(Error::SqttUnsupported { .. })));
}

#[test_case(AmdArch::Gfx942, true; "cdna")]
#[test_case(AmdArch::Gfx950, true; "cdna4")]
#[test_case(AmdArch::Gfx1151, false; "aql")]
fn unsupported_targets_are_refused(arch: AmdArch, pm4: bool) {
    assert!(matches!(check_target(arch, pm4), Err(Error::SqttUnsupported { .. })));
}

#[test]
fn streams_refuse_cdna() {
    assert!(matches!(
        build_streams(&config(1), AmdArch::Gfx942, &topology(1), 1, BUF, WPTRS),
        Err(Error::SqttUnsupported { .. })
    ));
}

#[test_case(SqttConfig { simds: SqttSimds::cycle(&[0, 4]), ..config(1) }; "simd")]
#[test_case(SqttConfig { bytes_per_se: 0, ..config(1) }; "empty window")]
#[test_case(SqttConfig { bytes_per_se: 4096 + 32, ..config(1) }; "unaligned window")]
#[test_case(SqttConfig { bytes_per_se: 16 << 30, ..config(1) }; "window past the size field")]
fn bad_configs_are_refused(cfg: SqttConfig) {
    assert!(matches!(
        build_streams(&cfg, AmdArch::Gfx1201, &topology(1), 1, BUF, WPTRS),
        Err(Error::SqttUnsupported { .. })
    ));
}

/// `(wptr & 0x1FFF_FFFF) * 32` bytes, truncated once the pointer reaches the
/// window's last 32 bytes; gfx11.0 counts from address 0, not the window.
#[test_case(AmdArch::Gfx1151, 0, 0x1000 => (0, false); "empty")]
#[test_case(AmdArch::Gfx1201, 0x8000_0010, 0x1000 => (0x200, false); "buffer id bit ignored")]
#[test_case(AmdArch::Gfx1201, 0x7F, 0x1000 => (0xFE0, true); "last line")]
#[test_case(AmdArch::Gfx1151, 0x80, 0x1000 => (0x1000, true); "full")]
#[test_case(AmdArch::Gfx1151, 0x81, 0x1000 => (0x1000, true); "past the window")]
#[test_case(AmdArch::Gfx1100, ((0x1_0000_2000u64 / 32) as u32 & 0x1FFF_FFFF) + 3, 0x1000 => (96, false); "gfx11.0 absolute")]
fn readback_bytes_follow_the_write_pointer(arch: AmdArch, wptr: u32, window: usize) -> (usize, bool) {
    trace_bytes(arch, wptr, 0x1_0000_2000, window)
}

#[test]
fn set_sqtt_refuses_aql_queues() {
    let (_iface, allocator) = mock_device_queues(1, false);
    let owner = OwnerCtx::new(Arc::clone(allocator.dev.core()), allocator);
    assert!(matches!(owner.set_sqtt(Some(SqttConfig::default())), Err(Error::SqttUnsupported { .. })));
    assert_eq!(owner.sqtt_config(), None);
    owner.set_sqtt(None).unwrap();
}

/// Arming pins no clock by default; a requested state is held once while re-armed at it, and taken
/// again once another state or a disarm released it.
#[test]
fn set_sqtt_pins_only_a_requested_pstate() {
    use crate::profile::StablePstate::{Peak, Standard};
    let (iface, allocator) = mock_device(1);
    let owner = OwnerCtx::new(Arc::clone(allocator.dev.core()), allocator);
    let pinned = |pstate| Some(SqttConfig { pstate, ..SqttConfig::default() });
    assert_eq!(SqttConfig::default().pstate, None, "a trace runs at the untraced clock by default");
    owner.set_sqtt(pinned(None)).unwrap();
    assert_eq!(iface.pstate_holds(), []);
    owner.set_sqtt(pinned(Some(Standard))).unwrap();
    owner.set_sqtt(pinned(Some(Standard))).unwrap();
    assert_eq!(iface.pstate_holds(), [Standard], "a held pin carries over a re-arm");
    owner.set_sqtt(pinned(Some(Peak))).unwrap();
    owner.set_sqtt(None).unwrap();
    owner.set_sqtt(pinned(Some(Peak))).unwrap();
    assert_eq!(iface.pstate_holds(), [Standard, Peak, Peak]);
    assert_eq!(owner.sqtt_config(), pinned(Some(Peak)));
}

#[test_case("standard" => Ok(crate::profile::StablePstate::Standard))]
#[test_case(" peak\n" => Ok(crate::profile::StablePstate::Peak))]
#[test_case("min_sclk" => Err("\"min_sclk\": standard or peak".to_string()))]
fn pstate_parses(s: &str) -> Result<crate::profile::StablePstate, String> {
    s.parse()
}

#[test]
fn set_sqtt_arms_and_disarms() {
    let (_iface, allocator) = mock_device(1);
    let core = Arc::clone(allocator.dev.core());
    let owner = OwnerCtx::new(Arc::clone(&core), allocator);
    assert!(matches!(
        owner.set_sqtt(Some(SqttConfig { simds: SqttSimds::cycle(&[9]), ..SqttConfig::default() })),
        Err(Error::SqttUnsupported { .. })
    ));
    owner.set_sqtt(Some(SqttConfig::default())).unwrap();
    assert_eq!(owner.sqtt_config(), Some(SqttConfig::default()));
    owner.set_sqtt(None).unwrap();
    assert_eq!(owner.sqtt_config(), None);
}

/// The handle reads each SE's filled window once the dispatch retired.
#[test]
fn handle_reads_each_se_window_after_retirement() {
    let (_iface, allocator) = mock_device(1);
    let pool = crate::amd::signal::SignalPool::new(&allocator, 64).unwrap();
    let signal = Arc::new(pool.acquire().unwrap());
    let finalizer = SubmissionFinalizer::timeline(Arc::clone(&signal), 1, None);
    let code = Arc::new(CodeImage { elf: b"\x7fELF"[..].into(), load_base: 0x1000, entry_pc: 0x1100 });
    let cfg = SqttConfig { bytes_per_se: 4096, se_mask: 0b1, ..SqttConfig::default() };
    let capture = Capture::new(&allocator, cfg, 1, Arc::clone(&code)).unwrap();
    let copies: Vec<u64> = decode(&capture.stop)
        .into_iter()
        .filter_map(|p| match p {
            Packet::Copy { dst, .. } => Some(dst),
            _ => None,
        })
        .collect();
    let [wptr_va] = copies[..] else { panic!("one SE on the synthetic device") };
    let gc = gfx(AmdArch::Gfx1100);
    let start = decode(&capture.start);
    let [size] = writes(&gc, &start, "regSQ_THREAD_TRACE_BUF0_SIZE")[..] else { panic!("one SE") };
    let [base] = writes(&gc, &start, "regSQ_THREAD_TRACE_BUF0_BASE")[..] else { panic!("one SE") };
    let trace_va = ((size as u64 & 0xF) << 44) | (base as u64) << 12;
    let handle = capture.into_handle(Arc::clone(&signal) as Arc<dyn DispatchTimestamps>, Arc::clone(&finalizer));
    // The synthetic device is gfx1100, whose pointer counts from address 0.
    // SAFETY: mock GPU VAs are host addresses of live allocations.
    unsafe {
        std::ptr::write_bytes(trace_va as *mut u8, 0xAB, 96);
        (wptr_va as *mut u32).write(((trace_va / 32) as u32 & 0x1FFF_FFFF) + 3);
    }
    assert_eq!(handle.sqtt(), None, "not retired");
    signal.reset(1);
    let trace = handle.sqtt().expect("retired");
    assert_eq!(trace.arch, AmdArch::Gfx1100);
    assert!(Arc::ptr_eq(&trace.code, &code), "the trace shares the program's image");
    assert_eq!(trace.mclk_mhz, None, "the synthetic device has no render node");
    let [se] = &trace.per_se[..] else { panic!("one SE") };
    assert_eq!((se.se, se.itrace, se.truncated), (0, true, false));
    assert_eq!(se.bytes, vec![0xAB; 96]);
}
