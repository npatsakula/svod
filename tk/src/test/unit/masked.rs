//! Masked boundary GLOBAL↔REG load/store ([`crate::MoveIdx::masked`]): a tile
//! straddling a ragged `N`/`D` edge reads `0.0` (load) / drops the write (store)
//! via validity-encoded index operands and a `WHERE` fill, instead of touching
//! out-of-bounds memory.
//! The gate is elided at build time when the dims are tile-aligned.
//!
//! Graph-shape checks run GPU-free; the round-trip is `#[ignore]` (arch-aware,
//! gfx942 wave64 / gfx1151 wave32 — the fragment is selected by role).

use std::sync::Arc;

use svod_dtype::{DType, DeviceSpec};
use svod_ir::{Op, TernaryOp, UOp};

use crate::arch::FragRole;
use crate::tiles::{RT_16X16, TileLayout};
use crate::{ArchCaps, Kernel, MoveIdx};
use svod_ir::ops;

const ROW: TileLayout = TileLayout::Row;

/// A masked GLOBAL→REG load over a ragged tensor emits an `INDEX` containing a
/// `WHERE(gate, index, INVALID)` and a `WHERE` fill; an unmasked load does not; and a masked load
/// over a tile-aligned tensor elides the gate (both axes divide evenly).
#[test]
fn test_masked_load_graph_shape() {
    let build = |shape: [usize; 4], masked: bool| {
        let bufs = vec![UOp::new_buffer(DeviceSpec::Cpu, shape.iter().product::<usize>(), DType::Float32)];
        let ker = Kernel::new("mload", [1, 1, 1], 64, bufs, ArchCaps::GFX942);
        let warp = ker.warp();
        let g = ker.gl(&shape, DType::Float32);
        let rt = ker.rt((32, 32), DType::Float32, ROW, RT_16X16);
        let mi = MoveIdx::block((0, 0, 0, 0), 2);
        let mi = if masked { mi.masked() } else { mi };
        warp.load(rt, g, mi).uop().toposort()
    };
    let has_valid_index = |t: &[Arc<UOp>]| {
        t.iter().any(|u| {
            matches!(
                u.op(),
                Op::Index(ops::Index { indices, .. })
                    if indices.iter().any(|idx| matches!(idx.op(), Op::Ternary(TernaryOp::Where, _, _, invalid) if UOp::is_invalid_marker(invalid)))
            )
        })
    };
    let has_fill = |t: &[Arc<UOp>]| {
        t.iter().any(|u| matches!(u.op(), Op::Ternary(TernaryOp::Where, _, _, alt) if !UOp::is_invalid_marker(alt)))
    };

    // Ragged N=17 and D=20 (a 32×32 tile straddles both) + masked → gated load.
    let ragged = build([1, 1, 17, 20], true);
    assert!(has_valid_index(&ragged), "masked ragged load: validity-encoded INDEX");
    assert!(has_fill(&ragged), "masked ragged load: WHERE with alternate fill");

    // Same ragged shape, unmasked → no gate (caller opted out).
    assert!(!has_valid_index(&build([1, 1, 17, 20], false)), "unmasked load: no validity mask");

    // Masked but tile-aligned (32×32) → gate elided at build time.
    assert!(!has_valid_index(&build([1, 1, 32, 32], true)), "masked aligned load: mask elided");
}

/// A masked REG→GLOBAL store over a ragged tensor validity-encodes its index.
#[test]
fn test_masked_store_graph_shape() {
    let bufs = vec![UOp::new_buffer(DeviceSpec::Cpu, 17 * 20, DType::Float32)];
    let ker = Kernel::new("mstore", [1, 1, 1], 64, bufs, ArchCaps::GFX942);
    let warp = ker.warp();
    let g = ker.gl(&[1, 1, 17, 20], DType::Float32);
    let rt = warp.zero(ker.rt((32, 32), DType::Float32, ROW, RT_16X16));
    let topo = warp.store(g, rt, MoveIdx::block((0, 0, 0, 0), 2).masked()).uop().toposort();
    assert!(
        topo.iter().any(|u| {
            matches!(
                u.op(),
                Op::Index(ops::Index { indices, .. })
                    if indices.iter().any(|idx| matches!(idx.op(), Op::Ternary(TernaryOp::Where, _, _, invalid) if UOp::is_invalid_marker(invalid)))
            )
        }),
        "masked ragged store: validity-encoded INDEX"
    );
}

/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib masked::test_masked_load_roundtrip_amd -- --ignored --nocapture`.
///
/// A 32×32 tile masked-loaded from a ragged `[17, 20]` tensor, then stored
/// (unmasked) to a 32×32 output. Load and store share the role-selected fragment,
/// so its layout cancels and `out[r][c] == in[r][c]` for the in-bounds prefix and
/// `0.0` for the out-of-bounds tail — proving the boundary gate + `alt` fill
/// without ever reading past the `[17, 20]` allocation.
#[test]
#[ignore]
fn test_masked_load_roundtrip_amd() {
    use svod_tensor::Tensor;

    let Some(caps) = super::fragment_device() else {
        eprintln!("skip test_masked_load_roundtrip_amd: no device with tk fragment layouts");
        return;
    };
    let w = caps.wave_size as i64;
    let (rows, cols) = (17usize, 20usize);

    // Ragged input: unique positive values, so the 0.0 out-of-bounds fill is
    // unambiguous.
    let data: Vec<f32> = (0..rows * cols).map(|i| (i + 1) as f32).collect();
    let mut a = Tensor::from_slice(&data).try_reshape([1usize, 1, rows, cols]).expect("reshape a");
    a.realize().expect("realize a");
    let mut out = Tensor::empty(&[1, 1, 32, 32], DType::Float32);

    crate::run_kernel("masked_roundtrip", [1, 1, 1], w, &mut [&mut out], &[&a], |ker| {
        let warp = ker.warp();
        let frag = ker.frag(FragRole::Accumulator);
        let o = ker.gl(&[1, 1, 32, 32], DType::Float32);
        let ain = ker.gl(&[1, 1, rows, cols], DType::Float32);
        // Masked load of a 32×32 tile straddling the [17, 20] edge → OOB lanes 0.0.
        let tile =
            warp.load(ker.rt((32, 32), DType::Float32, ROW, frag), ain, MoveIdx::block((0, 0, 0, 0), 2).masked());
        let _ = warp.store(o, tile, MoveIdx::block((0, 0, 0, 0), 2));
        ker.finish(1)
    })
    .expect("masked_roundtrip launch");

    let got = out.as_vec::<f32>().expect("read out");
    let mut bad = 0;
    for r in 0..32usize {
        for c in 0..32usize {
            let want = if r < rows && c < cols { data[r * cols + c] } else { 0.0 };
            if (got[r * 32 + c] - want).abs() > 1e-6 {
                if bad < 8 {
                    eprintln!("out[{r}][{c}] = {} want {want}", got[r * 32 + c]);
                }
                bad += 1;
            }
        }
    }
    assert_eq!(bad, 0, "{bad}/1024 elements wrong on {:?}", caps.arch);
    println!("masked round-trip: 1024/1024 correct on {:?} ([17,20] in-bounds, 0.0 OOB)", caps.arch);
}
