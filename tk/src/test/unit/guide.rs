//! Mirror of the "Writing a Kernel" documentation guide
//! (`website/docs/tile-kernels/first-kernel.md`): the simplest end-to-end hand
//! kernel — load two `16×16` f32 tiles, add them, store the result.
//!
//! `build_tile_add` is the exact kernel body shown in the guide; both the
//! always-on graph-shape check and the `#[ignore]` AMD execution test drive it,
//! so the doc and the tests cannot drift. The kernel mints `Op::Special` lane
//! indices (it is a real GPU kernel), so numeric execution is gated on AMD
//! hardware like every other tk kernel; the GPU-free shape check guards the
//! documented builder code on every `cargo test`.

use std::sync::Arc;

use svod_dtype::{DType, DeviceSpec};
use svod_ir::{BinaryOp, Op, UOp};

use crate::arch::FragRole;
use crate::tiles::TileLayout;
use crate::{ArchCaps, Kernel, MoveIdx};
use svod_ir::ops;

const ROW: TileLayout = TileLayout::Row;

/// The guide's kernel body, verbatim. `ker` is already bound to its buffers in
/// `gl()` declaration order — outputs first, then inputs — so the `gl` calls
/// below mint `out, a, b` in exactly the order `run_kernel` was handed them.
fn build_tile_add(ker: &Kernel) -> Arc<UOp> {
    let warp = ker.warp();

    // Globals: a typed view over each flat buffer. Order matches the launch:
    // output, then the two inputs.
    let o = ker.gl(&[1, 1, 16, 16], DType::Float32);
    let ga = ker.gl(&[1, 1, 16, 16], DType::Float32);
    let gb = ker.gl(&[1, 1, 16, 16], DType::Float32);

    // Ask for the fragment by ROLE, not a hardcoded constant — `caps.frag`
    // resolves it to the arch-correct `16×16` f32 accumulator (wave64 or wave32).
    let frag = ker.frag(FragRole::Accumulator);

    // global -> register, straight into the fragment layout (axis 2 splits the
    // row stride of the `[1, 1, N, N]` view).
    let ra = warp.load(ker.rt((16, 16), DType::Float32, ROW, frag), ga, MoveIdx::block((0, 0, 0, 0), 2));
    let rb = warp.load(ker.rt((16, 16), DType::Float32, ROW, frag), gb, MoveIdx::block((0, 0, 0, 0), 2));

    // The one compute op: elementwise add (note `add` takes `a` by value, `b` by ref).
    let rc = warp.add(ra, &rb);

    // register -> global, then close the kernel around its single store.
    let _ = warp.store(o, rc, MoveIdx::block((0, 0, 0, 0), 2));
    ker.finish(1)
}

/// Three flat `[N×N]` f32 BUFFER UOps (`out, a, b`) for GPU-free graph-shape builds.
fn dummy_buffers(n: usize) -> Vec<Arc<UOp>> {
    let sz = n * n;
    vec![
        UOp::new_buffer(DeviceSpec::Cpu, sz, DType::Float32),
        UOp::new_buffer(DeviceSpec::Cpu, sz, DType::Float32),
        UOp::new_buffer(DeviceSpec::Cpu, sz, DType::Float32),
    ]
}

/// GPU-free shape check (runs on every `cargo test`): the documented builder code
/// compiles and lowers to exactly what the guide claims — a real GPU kernel
/// (`Op::Special` lane index) that is *only* load / elementwise-add / store, with
/// **no** matrix-core op (`Wmma`) and **no** shared-memory staging (`DefineLocal`).
#[test]
fn test_tile_add_graph_shape() {
    let ker = Kernel::new("tile_add", [1, 1, 1], 64, dummy_buffers(16), ArchCaps::GFX942);
    let topo = build_tile_add(&ker).toposort();

    assert!(
        topo.iter().any(|u| matches!(u.op(), Op::Special(..))),
        "a hand kernel mints a lane Special — it is a GPU kernel"
    );
    assert!(topo.iter().any(|u| matches!(u.op(), Op::Binary(BinaryOp::Add, ..))), "the elementwise add is present");
    assert!(topo.iter().any(|u| matches!(u.op(), Op::Store(..))), "the register→global store is present");
    assert!(!topo.iter().any(|u| matches!(u.op(), Op::Wmma(..))), "no matrix core: this is a plain elementwise kernel");
    assert!(
        !topo
            .iter()
            .any(|u| matches!(u.op(), Op::Buffer(ops::Buffer { arg, .. }) if arg.addrspace == Some(svod_ir::AddrSpace::Local))),
        "no LDS: the round-trip is register-only"
    );
}

/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib guide::test_tile_add_amd -- --ignored`.
///
/// The same kernel body on real AMD hardware. Inputs `a[i] = i`, `b[i] = 2i`, so
/// the fragment round-trip (load → add → store) must reproduce `out[i] = 3i`. The
/// wave width (64 on CDNA, 32 on RDNA) sets the launch block; `caps.frag` inside
/// `build_tile_add` selects the matching fragment, so one body runs on both.
#[test]
#[ignore]
fn test_tile_add_amd() {
    use svod_tensor::Tensor;

    let Some(caps) = super::fragment_device() else {
        eprintln!("skip test_tile_add_amd: no device with tk fragment layouts");
        return;
    };
    let w = caps.wave_size as i64;

    let a: Vec<f32> = (0..256).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..256).map(|i| (2 * i) as f32).collect();
    let ta = Tensor::from_slice(&a);
    let tb = Tensor::from_slice(&b);
    let mut out = Tensor::empty(&[1, 1, 16, 16], DType::Float32);

    crate::run_kernel("tile_add", [1, 1, 1], w, &mut [&mut out], &[&ta, &tb], build_tile_add)
        .expect("tile_add launch (amd)");

    let got = out.as_vec::<f32>().expect("read out");
    let expected: Vec<f32> = (0..256).map(|i| (3 * i) as f32).collect();
    let bad = got.iter().zip(&expected).filter(|(g, e)| (**g - **e).abs() > 1e-3).count();
    assert_eq!(bad, 0, "out[i] must be a[i] + b[i] = 3i; got e.g. {:?}", &got[..8]);
}
