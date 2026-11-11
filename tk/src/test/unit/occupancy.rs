//! The planner's resident blocks against the device's CU limiter
//! (`svod_device::amd::occupancy::limits`), which the SQTT analysis reports beside the waves a trace
//! saw. They differ where the planner leaves out the 24-register VGPR granule, and where its
//! register count by layout falls short of what the compiler allocates.

use proptest::prelude::*;
use svod_device::amd::occupancy::{Cu, limits};
use svod_device::profile::KernelResources;
use svod_dtype::{AmdArch, GpuArch};
use test_case::test_case;

use crate::atoms::{Device, Target};
use crate::kernels::gemm::Epilogue;
use crate::ops::config::{GemmProblem, Planner, gemm_cfg};

fn planner(card: &str) -> (Planner, Cu) {
    let target = Target::card(card).expect("a listed card");
    let GpuArch::Amd(arch) = target.arch else { panic!("{card}: not AMD") };
    let planner = Planner::new(Device { target, costs: crate::test::unit::gfx1201_costs() }).expect("an SM count");
    (planner, Cu::rdna(arch).expect("RDNA"))
}

fn resources(vgprs: u32, lds_bytes: usize) -> KernelResources {
    let lds_bytes = lds_bytes as u32;
    KernelResources {
        vgprs: Some(vgprs),
        agprs: None,
        sgprs: None,
        lds_bytes,
        scratch_bytes: None,
        wave_size: 32,
        occupancy: None,
    }
}

/// The device's workgroups a CU for `regs` registers a lane, `smem` LDS bytes and `waves` waves.
fn device(cu: &Cu, regs: u32, smem: usize, waves: usize) -> usize {
    limits(&resources(regs, smem), waves as u32 * 32, cu).expect("wave32").workgroups_per_cu as usize
}

/// On a GEMM sweep's claims the two agree once the planner's registers round up to the granule:
/// every difference is the granule's, and it makes some.
#[test_case("rx9070xt", 2)]
#[test_case("strix-halo", 2)]
fn resident_blocks_agree_up_to_the_vgpr_granule(card: &str, granule_differences: usize) {
    let (planner, cu) = planner(card);
    let p = GemmProblem::new(1, 1500, 1280, 5120, &Epilogue::DEFAULT);
    let tiles =
        [[256, 128, 32], [128, 128, 32], [128, 128, 64], [64, 64, 32], [128, 64, 32], [256, 256, 32], [64, 128, 64]];
    let (mut claims, mut differ) = (0, 0);
    for tile in tiles {
        for warps in [[4, 2], [2, 2], [2, 4], [4, 4], [2, 1]] {
            for stages in [1, 2] {
                let Ok(c) = planner.claim_of(&p, &gemm_cfg(tile, stages, warps, true)) else { continue };
                claims += 1;
                let granular = c.regs.next_multiple_of(cu.vgpr_granule);
                let ours = device(&cu, c.regs, c.smem, c.waves);
                assert_eq!(planner.resident(c.smem, granular, c.waves), ours, "{tile:?} {warps:?} s{stages}: {c:?}");
                differ += usize::from(planner.resident(c.smem, c.regs, c.waves) != ours);
            }
        }
    }
    assert!(claims >= 40, "{claims} claims");
    assert_eq!(differ, granule_differences, "configs the planner holds more of than the CU does");
}

/// The Strix GEMM of the `svod/strix_smoke` capture: the planner counts 176 registers a lane and
/// holds two blocks a CU; the compiler allocated 208, which fit one, and the trace saw one (4 of
/// its 8 waves a SIMD, `svod_runtime::sqtt`'s occupancy test).
#[test]
fn strix_gemm_registers_by_layout_fall_short_of_the_compilers() {
    let (planner, cu) = planner("strix-halo");
    let p = GemmProblem::new(1, 1500, 1280, 5120, &Epilogue::DEFAULT);
    let c = planner.claim_of(&p, &gemm_cfg([128, 128, 32], 2, [4, 2], true)).expect("admitted");
    assert_eq!((c.regs, c.smem, c.waves, planner.resident(c.smem, c.regs, c.waves)), (176, 32768, 8, 2));
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../device/src/test/fixtures/sqtt/svod/strix_smoke/kernel.co");
    let elf = std::fs::read(path).expect("the fixture's code object");
    let compiled = svod_device::amd::program::kernel_resources(&elf, "gemm", AmdArch::Gfx1151.gfx_major()).unwrap();
    assert_eq!((compiled.vgprs, compiled.lds_bytes), (Some(208), 32768));
    assert_eq!(limits(&compiled, 256, &cu).map(|l| l.workgroups_per_cu), Some(1));
}

proptest! {
    /// On whole granules of registers and LDS the two count the same blocks.
    #[test]
    fn resident_blocks_agree_on_whole_granules(
        card in prop::sample::select(vec!["rx9070xt", "strix-halo"]),
        granules in 1u32..=16,
        lds in 1usize..=128,
        waves in 1usize..=16,
    ) {
        let (planner, cu) = planner(card);
        let (regs, smem) = (granules * cu.vgpr_granule, lds * cu.lds_granule as usize);
        prop_assert_eq!(planner.resident(smem, regs, waves), device(&cu, regs, smem, waves));
    }
}
