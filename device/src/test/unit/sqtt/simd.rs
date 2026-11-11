use proptest::prelude::*;
use test_case::test_case;

use super::fixture::stitched;
use crate::sqtt::Unit;
use crate::sqtt::hidden::{Pipes, matrix_pcs};
use crate::sqtt::simd::{PipeBusy, Residency, Simd};
use crate::sqtt::wave::{Cause, Occupancy};

fn event(time: i64, simd: u8, wave: u8, start: bool, cause: Cause) -> Occupancy {
    Occupancy { time, unit: Unit { sa: 0, wgp: 1, simd, wave }, start, pc: 0x100, dispatch: 1, cause }
}

/// Four waves at most, from 30 to 60: the orphan AMD inserts at 0 and the context save and restore
/// at 40 and 45 leave the count alone.
#[test]
fn steady_window_skips_orphans_and_context_switches() {
    use Cause::*;
    let events = [
        event(0, 0, 0, true, Orphan),
        event(10, 0, 1, true, Wave),
        event(10, 1, 0, true, Wave),
        event(20, 1, 1, true, Wave),
        event(30, 0, 2, true, Wave),
        event(40, 0, 1, false, Context),
        event(45, 0, 1, true, Context),
        event(50, 0, 0, false, Wave),
        event(60, 0, 1, false, Wave),
        event(70, 1, 0, false, Wave),
        event(80, 1, 1, false, Wave),
        event(90, 0, 2, false, Wave),
    ];
    let r = Residency::new(&events).unwrap();
    assert_eq!((r.peak, r.robust_peak, r.window, r.wave_cycles), (4, 4, (30, 60), 4 * 30));
    let simd = |simd| Simd { sa: 0, wgp: 1, simd, wave_cycles: 2 * 30, cycles: 30, peak: 2 };
    assert_eq!(r.simds, [simd(0), simd(1)]);
    assert_eq!((r.mean(), r.simd(0, 1, 1).unwrap().mean()), (4.0, 2.0));
    assert!(Residency::new(&events[..1]).is_none(), "an orphan alone is no wave");
}

/// Four waves through `[0, 100)` and a fifth for two cycles: the spike sets the peak, the window
/// spans the four.
#[test]
fn robust_peak_ignores_a_spike() {
    let mut events: Vec<Occupancy> = (0..4).map(|w| event(0, w, 0, true, Cause::Wave)).collect();
    events.extend([event(50, 0, 1, true, Cause::Wave), event(52, 0, 1, false, Cause::Wave)]);
    events.extend((0..4).map(|w| event(100, w, 0, false, Cause::Wave)));
    let r = Residency::new(&events).unwrap();
    assert_eq!((r.peak, r.robust_peak, r.window, r.wave_cycles), (5, 4, (0, 100), 4 * 100 + 2));
}

/// svod's GEMMs: v1 and v3 hold 4 waves on each of an SE's 30 busy SIMDs through the window; the
/// Strix GEMM 4 on average (a few SIMDs drain early, 3.9) and 8 at a workgroup's handover.
#[test_case("svod/rdna4_gemm_v1", 120, 30, 4)]
#[test_case("svod/rdna4_gemm_v3", 120, 30, 4)]
#[test_case("svod/strix_smoke", 184, 40, 8)]
fn fixture_residency(name: &str, max_peak: u32, simds: usize, simd_peak: u32) {
    let s = stitched(name);
    for trace in &s.traces {
        let r = Residency::new(&trace.occupancy).unwrap();
        assert!(r.peak <= max_peak && r.simds.len() == simds, "{r:?}");
        let means: Vec<f64> = r.simds.iter().map(Simd::mean).collect();
        assert!(r.simds.iter().all(|x| (x.mean() - 4.0).abs() < 0.1 && x.peak <= simd_peak), "{name}: {means:?}");
        let h = trace.header;
        if !trace.waves.is_empty() {
            assert!(r.simd(h.dsa, h.dwgp, h.dsimd).is_some(), "the traced SIMD held waves");
        }
    }
}

/// The traced SIMD's pipes over the window: v1's matrix pipe busy about three quarters of it; on
/// gfx11 (Strix) a WMMA traces as one cycle, so the matrix pipe looks nearly idle.
#[test_case("svod/rdna4_gemm_v1", 0.70..0.80)]
#[test_case("svod/strix_smoke", 0.0..0.05)]
fn pipe_busy_over_the_window(name: &str, matrix: std::ops::Range<f64>) {
    let s = stitched(name);
    let wmma = matrix_pcs(&s.lines());
    for trace in &s.traces {
        let r = Residency::new(&trace.occupancy).unwrap();
        let b = PipeBusy::of(&Pipes::new(&trace.waves, &trace.other_simd, &wmma), r.window);
        assert_eq!(b.cycles, r.window.1 - r.window.0);
        assert!([b.matrix, b.valu, b.vmem, b.scalar].iter().all(|&c| (0..=b.any).contains(&c)) && b.any <= b.cycles);
        assert!(matrix.contains(&b.share(b.matrix)), "{name}: {b:?}");
    }
}

proptest! {
    /// Whatever the events, the SIMDs' waves add up to the SE's, no mean passes its peak and the
    /// window lies within the events.
    #[test]
    fn residency_adds_up(raw in prop::collection::vec((0i64..200, 0u8..4, 0u8..4, any::<bool>()), 1..80)) {
        let events: Vec<Occupancy> = raw.iter().map(|&(t, simd, wave, start)| event(t, simd, wave, start, Cause::Wave)).collect();
        // `None`: no wave stayed resident for a cycle.
        let Some(r) = Residency::new(&events) else { return Ok(()) };
        let (lo, hi) = (raw.iter().map(|e| e.0).min().unwrap(), raw.iter().map(|e| e.0).max().unwrap());
        prop_assert!(lo <= r.window.0 && r.window.0 < r.window.1 && r.window.1 <= hi);
        prop_assert_eq!(r.simds.iter().map(|s| s.wave_cycles).sum::<i64>(), r.wave_cycles);
        prop_assert!(r.mean() <= f64::from(r.peak) && r.peak > 0 && (1..=r.peak).contains(&r.robust_peak));
        prop_assert!(r.simds.iter().all(|s| s.mean() <= f64::from(s.peak) && s.peak <= r.peak));
    }
}
