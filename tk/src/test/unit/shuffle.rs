//! Cross-lane primitives ([`crate::Group::shuffle`]/`shuffle_xor`/`compare_exchange`)
//! — the `ds_bpermute` (AMD) / `shfl.sync` (CUDA) foundation for sorting networks,
//! arg-reduce, and scan. Graph-shape checks run GPU-free; the HW tests are
//! `#[ignore]` (the tile all-reduce needs a fragment layout — AMD; the scalar
//! shuffles run on any supported GPU).

use svod_dtype::{AmdArch, CudaArch, DType, GpuArch};
use svod_ir::Op;

use crate::arch::FragRole;
use crate::tiles::{RT_16X16, TileLayout};
use crate::{ArchCaps, Kernel, MoveIdx, SwapDir};
use svod_ir::ops;

const ROW: TileLayout = TileLayout::Row;

/// `shuffle_xor` lowers to a cross-lane gather (an `Op::Custom` — `ds_bpermute` on
/// AMD, `shfl.sync` on CUDA) with no LDS scratch and no barrier — on wave64
/// (gfx942), wave32 (gfx1151), and warp32 (sm_86), i.e. the lane math is arch-blind
/// (`ArchCaps::wave_size`).
#[test]
fn test_shuffle_xor_graph_shape() {
    let build = |caps: ArchCaps, block: i64| {
        let ker = Kernel::new("shuf", [1, 1, 1], block, vec![], caps);
        let warp = ker.warp();
        let src = warp.zero(ker.rt((16, 16), DType::Float32, ROW, RT_16X16));
        let dst = ker.rt((16, 16), DType::Float32, ROW, RT_16X16);
        warp.shuffle_xor(dst, &src, 16).uop().toposort()
    };
    let sm_86 = ArchCaps::for_arch(GpuArch::Cuda(CudaArch::from_compute_capability(8, 6)));
    for (caps, block) in [(ArchCaps::GFX942, 64), (ArchCaps::for_amd(AmdArch::Gfx1151), 32), (sm_86, 32)] {
        let topo = build(caps, block);
        assert!(
            topo.iter().any(|u| matches!(u.op(), Op::Custom(..))),
            "{:?}: shuffle_xor emits a cross-lane Op::Custom",
            caps.arch
        );
        assert!(
            !topo
                .iter()
                .any(|u| matches!(u.op(), Op::Buffer(ops::Buffer { arg, .. }) if arg.addrspace == Some(svod_ir::AddrSpace::Local))),
            "{:?}: no LDS scratch",
            caps.arch
        );
        assert!(!topo.iter().any(|u| matches!(u.op(), Op::Barrier(..))), "{:?}: no barrier", caps.arch);
    }
}

/// `compare_exchange` lowers to a `ds_bpermute` gather plus an ALU min/max select
/// (a `Ternary` `where`), with no LDS and no barrier.
#[test]
fn test_compare_exchange_graph_shape() {
    let ker = Kernel::new("ce", [1, 1, 1], 64, vec![], ArchCaps::GFX942);
    let warp = ker.warp();
    let src = warp.zero(ker.rt((16, 16), DType::Float32, ROW, RT_16X16));
    let dst = ker.rt((16, 16), DType::Float32, ROW, RT_16X16);
    let topo = warp.compare_exchange(dst, &src, 1, SwapDir::ByLaneBit(2)).uop().toposort();
    assert!(topo.iter().any(|u| matches!(u.op(), Op::Custom(..))), "ds_bpermute gather present");
    assert!(topo.iter().any(|u| matches!(u.op(), Op::Ternary(..))), "min/max select (where) present");
    assert!(
        !topo
            .iter()
            .any(|u| matches!(u.op(), Op::Buffer(ops::Buffer { arg, .. }) if arg.addrspace == Some(svod_ir::AddrSpace::Local))),
        "no LDS scratch"
    );
    assert!(!topo.iter().any(|u| matches!(u.op(), Op::Barrier(..))), "no barrier");
}

/// `SVOD_DEVICE={AMD,CUDA}:0 cargo test -p svod-tk --lib shuffle::test_scalar_shuffles_gpu -- --ignored`.
///
/// End-to-end on any supported GPU, fragment-free (plain per-lane stores): the
/// three scalar cross-lane primitives the single-query attention kernel is built
/// from. Per lane `L` of a `w`-lane wave, seeded with `x = L`:
/// - `wave_reduce_scalar(add)` → `Σ L = w(w−1)/2` in every lane (the XOR
///   butterfly — `shfl.sync.bfly` on CUDA);
/// - `subgroup_reduce_scalar(8, add)` → the 8-lane subgroup sum `8·(L/8)·8 + 28`;
/// - `broadcast_scalar(5)` → `5` in every lane (an indexed gather —
///   `shfl.sync.idx` on CUDA).
#[test]
#[ignore]
fn test_scalar_shuffles_gpu() {
    use svod_tensor::Tensor;

    use crate::index::{index_off, load_at};

    let dev = Tensor::rand(&[16, 16]).expect("probe").device();
    let Some(arch) = crate::target::resolve_arch(&dev) else {
        eprintln!("skip test_scalar_shuffles_gpu: no GPU device");
        return;
    };
    let w = ArchCaps::for_arch(arch).wave_size as i64;

    let seed: Vec<f32> = (0..w).map(|l| l as f32).collect();
    let seed_t = Tensor::from_slice(&seed);
    let mut out = Tensor::empty(&[3 * w as usize], DType::Float32);
    crate::run_kernel("scalar_shuffles", [1, 1, 1], w, &mut [&mut out], &[&seed_t], |ker| {
        let warp = ker.warp();
        let o = ker.gl(&[3 * w as usize], DType::Float32);
        let x_gl = ker.gl(&[w as usize], DType::Float32);
        let lane = ker.laneid();
        let x = load_at(x_gl.uop(), x_gl.shape(), &[crate::index::Idx::from(&lane)]);
        let results = [
            warp.wave_reduce_scalar(x.clone(), |a, b| a.add(b)),
            warp.subgroup_reduce_scalar(x.clone(), 8, |a, b| a.add(b)),
            warp.broadcast_scalar(&x, 5),
        ];
        let stores = results
            .into_iter()
            .enumerate()
            .map(|(i, v)| index_off(o.uop(), lane.add(&crate::index::cidx(i as i64 * w))).store(v))
            .collect();
        ker.push_store(svod_ir::UOp::group(stores), o.uop().clone());
        ker.finish(1)
    })
    .expect("scalar shuffle launch");

    let got = out.as_vec::<f32>().expect("read out");
    let wave_sum = (w * (w - 1) / 2) as f32;
    for l in 0..w as usize {
        let subgroup_sum = (8 * (l / 8) * 8 + 28) as f32;
        assert_eq!(got[l], wave_sum, "lane {l}: wave sum");
        assert_eq!(got[w as usize + l], subgroup_sum, "lane {l}: subgroup-8 sum");
        assert_eq!(got[2 * w as usize + l], 5.0, "lane {l}: broadcast from lane 5");
    }
}

/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib shuffle::test_shuffle_xor_allreduce_amd -- --ignored`.
///
/// End-to-end (gfx942): a butterfly all-reduce — `x += shuffle_xor(x, mask)` for
/// `mask ∈ {1,2,…,32}` — sums each tile element across all 64 lanes. Seeded with
/// `1.0` everywhere, every element must become `wave_size = 64`. This validates the
/// `shuffle_xor` transport + lane math end-to-end and is **layout-independent** (the
/// sum of 1s over the wave is `wave_size` regardless of the lane↔element map).
#[test]
#[ignore]
fn test_shuffle_xor_allreduce_amd() {
    use svod_tensor::Tensor;

    // Arch-aware: derive the wave width (64 on CDNA, 32 on RDNA) so the butterfly
    // mask sequence, launch block, and expected sum all match the device.
    let Some(caps) = super::fragment_device() else {
        eprintln!("skip test_shuffle_xor_allreduce_amd: no device with tk fragment layouts");
        return;
    };
    let w = caps.wave_size as i64;
    let masks: Vec<i64> = (0..).map(|i| 1i64 << i).take_while(|&m| m < w).collect();

    let mut out = Tensor::empty(&[1, 1, 16, 16], DType::Float32);
    crate::run_kernel("allreduce", [1, 1, 1], w, &mut [&mut out], &[], |ker| {
        let warp = ker.warp();
        let o = ker.gl(&[1, 1, 16, 16], DType::Float32);
        // The arch-correct f32 16×16 fragment: RT_16X16 (ept=4) on CDNA wave64,
        // RT_16X16_W32_ACC (ept=8, even/odd interleave) on RDNA wave32 — so the store
        // covers all 256 elements, not just the wave64 half.
        let frag = ker.frag(FragRole::Accumulator);
        let mut x = warp.ones(ker.rt((16, 16), DType::Float32, ROW, frag));
        for &mask in &masks {
            let tmp = warp.shuffle_xor(ker.rt((16, 16), DType::Float32, ROW, frag), &x, mask);
            x = warp.add(x, &tmp);
        }
        let _ = warp.store(o, x, MoveIdx::block((0, 0, 0, 0), 2));
        ker.finish(1)
    })
    .expect("allreduce launch");

    let got = out.as_vec::<f32>().expect("read out");
    assert_eq!(got.len(), 256, "16x16 tile");
    let expected = w as f32;
    let bad = got.iter().filter(|&&v| (v - expected).abs() > 1e-3).count();
    assert_eq!(bad, 0, "every element must be the {w}-lane sum of 1.0 = {expected}; got e.g. {:?}", &got[..8]);
}
