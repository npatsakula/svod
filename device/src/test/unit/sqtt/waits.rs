//! Wait → instruction binding: hand-built streams, one counter rule each, then the fixtures.

use proptest::prelude::*;
use test_case::test_case;

use super::fixture::stitched;
use crate::sqtt::asm::{Kind, classify};
use crate::sqtt::waits::Waits;
use crate::sqtt::wave::{Instruction, Wave};
use crate::sqtt::{Category, Layout, Unit};

/// `s_cbranch_scc1` from line `from` to line `to` of a program at 4 bytes a line.
fn branch(from: usize, to: usize) -> String {
    format!("s_cbranch_scc1 {}", (to as i64 - from as i64 - 1) as u16)
}

/// Where the programs load: PC 0 is an unstitched record's.
const BASE: u64 = 0x1000;

/// A wave running `program` (line `i` at PC `BASE + 4 * i`) along `path`, a line every 10 cycles; a wait
/// stalls `stall` cycles, and a branch is taken where the path does not fall through.
fn wave(program: &[String], path: &[usize], stall: i64) -> Wave {
    let instructions = path
        .iter()
        .enumerate()
        .map(|(at, &line)| {
            let (time, pc) = (10 * at as i64, BASE + 4 * line as u64);
            let category = match classify(&program[line]).kind {
                Kind::Wait(_) => {
                    return Instruction { time, category: Category::Immed, duration: 1 + stall, stall, pc };
                }
                Kind::Branch if path.get(at + 1).is_some_and(|&next| next != line + 1) => Category::Jump,
                Kind::Branch => Category::Next,
                Kind::Memory(_) => Category::Vmem,
                _ => Category::Salu,
            };
            Instruction { time, category, duration: 1, stall: 0, pc }
        })
        .collect::<Vec<_>>();
    let end = instructions.last().map_or(0, |i| i.time + i.duration);
    let pcs = vec![(0, 0)];
    Wave {
        unit: Unit::default(),
        me: 0,
        pipe: 0,
        begin: 0,
        end,
        complete: true,
        contexts: 0,
        exclude_barrier_wait: false,
        instructions,
        pcs,
    }
}

fn analyze(layout: Layout, program: &[String], waves: &[Wave]) -> Waits {
    let code: Vec<(u64, &str)> = program.iter().enumerate().map(|(i, l)| (BASE + 4 * i as u64, l.as_str())).collect();
    Waits::new(waves, &code, layout, BASE)
}

fn lines(program: &[&str]) -> Vec<String> {
    program.iter().map(|l| l.to_string()).collect()
}

const LOAD: &str = "global_load_b32 v0, v[0:1], off";
const STORE: &str = "global_store_b32 v[0:1], v0, off";
const DS_LOAD: &str = "ds_load_b32 v0, v1";
const DS_STORE: &str = "ds_store_b32 v1, v0";
const SMEM: &str = "s_load_b32 s0, s[0:1], 0x0";
const FLAT: &str = "flat_load_b32 v0, v[0:1]";
const SAMPLE: &str = "image_sample v[0:3], v0, s[0:7], s[8:11] dmask:0xf";

/// `expected`: `(wait line, binding line)` for each wait the straight-line program runs; `approx`:
/// how many of those bindings are guesses.
#[test_case(Layout::Rdna4, &[LOAD, LOAD, LOAD, "s_wait_loadcnt 0x1"], &[(3, Some(1))], 0 ; "loads retire oldest first")]
#[test_case(Layout::Rdna4, &[LOAD, "s_wait_loadcnt 0x0", "s_wait_loadcnt 0x0"], &[(1, Some(0)), (2, None)], 0 ; "a retired load binds once")]
#[test_case(Layout::Rdna4, &[LOAD, "s_wait_loadcnt 0x3"], &[(1, None)], 0 ; "a wait above the count binds nothing")]
#[test_case(Layout::Rdna4, &[STORE, LOAD, "s_wait_storecnt 0x0", "s_wait_loadcnt 0x0"], &[(2, Some(0)), (3, Some(1))], 0 ; "stores count apart from loads")]
#[test_case(Layout::Rdna4, &["global_atomic_add_u32 v0, v1, s[0:1]", "global_atomic_add_u32 v0, v1, v2, s[0:1] th:TH_ATOMIC_RETURN", "s_wait_storecnt 0x0", "s_wait_loadcnt 0x0"], &[(2, Some(0)), (3, Some(1))], 0 ; "an atomic returning data is a load")]
#[test_case(Layout::Rdna4, &[DS_STORE, DS_LOAD, LOAD, "s_wait_dscnt 0x0"], &[(3, Some(1))], 0 ; "dscnt counts LDS loads and stores")]
#[test_case(Layout::Rdna4, &[DS_LOAD, LOAD, "s_wait_loadcnt_dscnt 0x0"], &[(2, Some(1))], 0 ; "a combined wait binds the youngest of both")]
#[test_case(Layout::Rdna4, &[SMEM, SMEM, "s_wait_kmcnt 0x0"], &[(2, Some(1))], 0 ; "kmcnt is exact at zero")]
#[test_case(Layout::Rdna4, &[SMEM, SMEM, "s_wait_kmcnt 0x1"], &[(2, Some(0))], 1 ; "kmcnt above zero is a guess")]
#[test_case(Layout::Rdna4, &[SMEM, "s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)", "s_wait_kmcnt 0x0"], &[(2, Some(1))], 0 ; "a message counts on kmcnt")]
#[test_case(Layout::Rdna4, &[DS_LOAD, SMEM, "s_wait_dscnt 0x0"], &[(2, Some(0))], 0 ; "scalar memory leaves dscnt alone")]
#[test_case(Layout::Rdna4, &[FLAT, "s_wait_dscnt 0x0", "s_wait_loadcnt 0x0"], &[(1, Some(0)), (2, Some(0))], 2 ; "flat counts on both and is a guess")]
#[test_case(Layout::Rdna4, &[SAMPLE, LOAD, "s_wait_samplecnt 0x0", "s_wait_loadcnt 0x0"], &[(2, Some(0)), (3, Some(1))], 0 ; "samples have their own counter on gfx12")]
#[test_case(Layout::Rdna4, &["global_inv scope:SCOPE_SE", "global_wb scope:SCOPE_SE", "s_wait_loadcnt 0x0", "s_wait_storecnt 0x0"], &[(2, Some(0)), (3, Some(1))], 0 ; "invalidates load and writebacks store")]
#[test_case(Layout::Rdna4, &[LOAD, DS_LOAD, SMEM, STORE, "s_wait_idle"], &[(4, Some(3))], 0 ; "idle waits on everything")]
#[test_case(Layout::Rdna4, &["global_prefetch_b8 v[0:1], off", "lds_param_load v0, attr0.x", "s_wait_loadcnt 0x0", "s_wait_dscnt 0x0"], &[(2, None), (3, None)], 0 ; "prefetches and LDS direct loads count nowhere waited on")]
#[test_case(Layout::Rdna3, &[LOAD, LOAD, "s_waitcnt vmcnt(1)"], &[(2, Some(0))], 0 ; "vmcnt retires oldest first")]
#[test_case(Layout::Rdna3, &[STORE, LOAD, "s_waitcnt_vscnt null, 0x0", "s_waitcnt vmcnt(0)"], &[(2, Some(0)), (3, Some(1))], 0 ; "vscnt counts stores")]
#[test_case(Layout::Rdna3, &[DS_LOAD, DS_STORE, DS_LOAD, "s_waitcnt lgkmcnt(1)"], &[(3, Some(1))], 0 ; "lgkmcnt over LDS alone is exact")]
#[test_case(Layout::Rdna3, &[SMEM, DS_LOAD, "s_waitcnt lgkmcnt(1)"], &[(2, Some(0))], 1 ; "lgkmcnt with scalar memory in flight is a guess")]
#[test_case(Layout::Rdna3, &[SMEM, DS_LOAD, "s_waitcnt lgkmcnt(0)"], &[(2, Some(1))], 0 ; "lgkmcnt at zero is exact")]
#[test_case(Layout::Rdna3, &["s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)", DS_LOAD, "s_waitcnt lgkmcnt(1)"], &[(2, Some(0))], 1 ; "a message counts on lgkmcnt")]
#[test_case(Layout::Rdna3, &[FLAT, "s_waitcnt lgkmcnt(0)"], &[(1, Some(0))], 1 ; "flat counts on lgkmcnt")]
#[test_case(Layout::Rdna3, &[SAMPLE, LOAD, "s_waitcnt vmcnt(1)"], &[(2, Some(0))], 1 ; "samples mixed with loads on vmcnt are a guess")]
#[test_case(Layout::Rdna3, &[SAMPLE, SAMPLE, "s_waitcnt vmcnt(1)"], &[(2, Some(0))], 0 ; "samples alone on vmcnt are exact")]
#[test_case(Layout::Rdna3, &["buffer_gl0_inv", "s_waitcnt vmcnt(0)"], &[(1, None)], 0 ; "gfx11 cache invalidates count nowhere")]
#[test_case(Layout::Rdna3, &[LOAD, DS_LOAD, "s_waitcnt vmcnt(0) lgkmcnt(0)"], &[(2, Some(1))], 0 ; "a two-counter waitcnt binds the youngest")]
fn binds(layout: Layout, program: &[&str], expected: &[(usize, Option<usize>)], approx: usize) {
    let program = lines(program);
    let w = analyze(layout, &program, &[wave(&program, &(0..program.len()).collect::<Vec<_>>(), 5)]);
    assert_eq!((w.waves, w.skipped, w.instances, w.approx), (1, 0, expected.len(), approx), "{w:?}");
    for &(at, binding) in expected {
        let site = w.sites.iter().find(|s| s.pc == 4 * at as u64).unwrap();
        let bindings: Vec<(u64, usize)> = binding.map(|b| (4 * b as u64, 1)).into_iter().collect();
        assert_eq!(
            (site.instances, site.bindings.clone(), site.unbound),
            (1, bindings, usize::from(binding.is_none()))
        );
    }
    for bound in &w.bindings {
        let b = bound.pc as usize / 4;
        let ats: Vec<usize> = expected.iter().filter(|e| e.1 == Some(b)).map(|e| e.0).collect();
        let latencies: Vec<i64> = ats.iter().map(|&at| 10 * (at - b) as i64 + 6).collect();
        let spread = bound.latency.unwrap();
        assert_eq!((spread.n, spread.min, spread.max), (ats.len(), latencies[0], *latencies.last().unwrap()));
        assert_eq!((bound.stall, bound.distance.clone()), (5 * ats.len() as i64, vec![(0, ats.len())]));
    }
}

/// The in-flight count at issue keys the latency curve, and a wait that did not stall gives none.
#[test]
fn latency_by_outstanding() {
    let program = lines(&[LOAD, LOAD, LOAD, "s_wait_loadcnt 0x0"]);
    let w = analyze(Layout::Rdna4, &program, &[wave(&program, &[0, 1, 2, 3], 5)]);
    let b = &w.bindings[0];
    assert_eq!((b.pc, b.count, b.stalled), (8, 1, 1));
    assert_eq!(b.by_outstanding.iter().map(|(n, l)| (*n, l.n, l.median)).collect::<Vec<_>>(), [(3, 1, 16)]);
    let hidden = analyze(Layout::Rdna4, &program, &[wave(&program, &[0, 1, 2, 3], 0)]);
    let b = &hidden.bindings[0];
    assert_eq!((b.count, b.stalled, b.latency, b.by_outstanding.is_empty()), (1, 0, None, true));
}

/// A load issued before the back edge and waited after it was fetched a trip ahead; the first trip's
/// wait binds the prologue's load.
#[test]
fn prefetch_distance_counts_back_edges() {
    // 0: load (prologue); 1: wait; 2: load (next trip); 3: back edge to 1; 4: wait (epilogue).
    let program =
        vec![LOAD.to_string(), "s_wait_loadcnt 0x0".into(), LOAD.into(), branch(3, 1), "s_wait_loadcnt 0x0".into()];
    let path = [0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 4];
    let w = analyze(Layout::Rdna4, &program, &[wave(&program, &path, 5)]);
    let distance = |pc: u64| w.bindings.iter().find(|b| b.pc == pc).unwrap().distance.clone();
    assert_eq!(distance(0), [(0, 1)]);
    assert_eq!(distance(8), [(0, 1), (1, 2)], "two trips wait across a back edge, the last after the exit");
    let site = w.sites.iter().find(|s| s.pc == 4).unwrap();
    assert_eq!((site.instances, site.bindings.clone()), (3, vec![(8, 2), (0, 1)]));
}

#[test]
fn skips_waves_it_cannot_replay() {
    let program = lines(&[LOAD, "s_wait_loadcnt 0x0"]);
    let whole = wave(&program, &[0, 1], 5);
    let incomplete = Wave { complete: false, ..whole.clone() };
    let switched = Wave { contexts: 2, ..whole.clone() };
    let mut unstitched = whole.clone();
    unstitched.instructions[0].pc = 0;
    let w = analyze(Layout::Rdna4, &program, &[whole, incomplete, switched, unstitched]);
    assert_eq!((w.waves, w.skipped, w.instances), (1, 3, 1));
}

/// v1 (the ring-fill wait after the trip's matrix work) against v3 (the select pulling it up to
/// 0x74c): every main-loop global load is waited within its trip, so its prefetch distance is 0, and
/// v3's 0x74c waits on 0x6fc, the oldest of six loads.
#[test_case("svod/rdna4_gemm_v1", 0x914, 0x718)]
#[test_case("svod/rdna4_gemm_v1", 0xb98, 0x9a0)]
#[test_case("svod/rdna4_gemm_v3", 0x74c, 0x6fc)]
#[test_case("svod/rdna4_gemm_v3", 0xab0, 0xa64)]
fn gemm_ring_fill(fixture: &str, wait: u64, load: u64) {
    let s = stitched(fixture);
    let w = Waits::new(&s.waves(), &s.lines(), s.layout, s.base);
    assert_eq!((w.skipped, w.approx), (0, 0));
    let site = w.sites.iter().find(|x| x.pc == wait).unwrap();
    assert_eq!(site.bindings, [(load, site.instances)]);
    let b = w.bindings.iter().find(|b| b.pc == load).unwrap();
    assert_eq!(b.distance, [(0, b.count)]);
    assert!(b.latency.is_some_and(|l| l.min > 0));
}

/// Strix's main loop issues its last A load at 0x93c and waits on it after the back edge.
#[test]
fn strix_prefetches_across_the_back_edge() {
    let s = stitched("svod/strix_smoke");
    let w = Waits::new(&s.waves(), &s.lines(), s.layout, s.base);
    let b = w.bindings.iter().find(|b| b.pc == 0x93c).unwrap();
    assert_eq!((b.distance.clone(), b.waits.clone()), (vec![(1, b.count)], vec![(0x5fc, b.count)]));
}

/// Over every fixture: instances and stall add up between sites and bindings, and every binding is a
/// counted instruction the wait could see.
#[test_case("amd/navi4_setpc")]
#[test_case("amd/navi4_sdata")]
#[test_case("amd/navi4_fifo")]
#[test_case("amd/navi4_barriernop")]
#[test_case("amd/strix_barrier")]
#[test_case("gfx1100/gemm")]
#[test_case("gfx1200/gemm")]
#[test_case("svod/rdna4_gemm_v1")]
#[test_case("svod/rdna4_gemm_v3")]
#[test_case("svod/strix_smoke")]
fn fixtures_add_up(fixture: &str) {
    let s = stitched(fixture);
    let waves = s.waves();
    let w = Waits::new(&waves, &s.lines(), s.layout, s.base);
    assert_eq!(w.waves + w.skipped, waves.len());
    assert_eq!(w.sites.iter().map(|x| x.instances).sum::<usize>(), w.instances);
    assert_eq!(w.bindings.iter().map(|b| b.count).sum::<usize>() + w.unbound, w.instances);
    let (sites, bound) =
        (w.sites.iter().map(|x| x.stall).sum::<i64>(), w.bindings.iter().map(|b| b.stall).sum::<i64>());
    let always_bound: i64 = w.sites.iter().filter(|x| x.unbound == 0).map(|x| x.stall).sum();
    assert!(always_bound <= bound && bound <= sites, "{fixture}: {always_bound} {bound} {sites}");
    for b in &w.bindings {
        assert!(
            matches!(classify(s.text(b.pc + s.base).unwrap()).kind, Kind::Memory(_) | Kind::SendMsg),
            "{fixture} {b:?}"
        );
        assert!(b.stalled <= b.count && b.approx <= b.count);
        assert_eq!(b.waits.iter().map(|w| w.1).sum::<usize>(), b.count);
        assert_eq!(b.distance.iter().map(|d| d.1).sum::<usize>(), b.count);
        assert_eq!(b.by_outstanding.iter().map(|(_, l)| l.n).sum::<usize>(), b.stalled);
    }
    for site in &w.sites {
        assert!(matches!(classify(s.text(site.pc + s.base).unwrap()).kind, Kind::Wait(_)), "{fixture} {site:?}");
    }
}

proptest! {
    /// Any stream of counted instructions, waits and branches replays without panicking, and every
    /// wait is bound or unbound once.
    #[test]
    fn replays_any_stream(
        program in prop::collection::vec(prop::sample::select(vec![
            LOAD, STORE, DS_LOAD, DS_STORE, SMEM, FLAT, SAMPLE, "s_wait_loadcnt 0x1", "s_wait_dscnt 0x0",
            "s_wait_kmcnt 0x0", "s_wait_loadcnt_dscnt 0x101", "s_waitcnt vmcnt(1) lgkmcnt(2)", "s_wait_idle",
            "s_cbranch_scc1 65533", "v_add_f32_e32 v0, v1, v2",
        ]), 1..24),
        steps in prop::collection::vec(0usize..24, 1..64),
        rdna4: bool,
    ) {
        let program = lines(&program);
        let path: Vec<usize> = steps.iter().map(|s| s % program.len()).collect();
        let layout = if rdna4 { Layout::Rdna4 } else { Layout::Rdna3 };
        let w = analyze(layout, &program, &[wave(&program, &path, 3)]);
        prop_assert_eq!(w.bindings.iter().map(|b| b.count).sum::<usize>() + w.unbound, w.instances);
        prop_assert!(w.bindings.iter().all(|b| b.latency.is_none_or(|l| l.min > 0)));
    }
}
