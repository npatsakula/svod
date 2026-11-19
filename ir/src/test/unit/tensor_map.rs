use std::sync::Arc;

use proptest::prelude::*;
use svod_dtype::{DType, DeviceSpec, ScalarDType};
use test_case::test_case;

use crate::{CanonicalGraph, Error, ProgramInfo, TensorMap, TensorMapSwizzle, UOp};

use ScalarDType::*;
use TensorMapSwizzle as Swizzle;

/// The driver's encoding rules, checked when the template is built.
#[test_case(Float32, [64, 32], [16, 8], Swizzle::None => matches Ok(_); "f32 box")]
#[test_case(Float16, [64, 32], [64, 8], Swizzle::B128 => matches Ok(_); "f16 row filling the 128-byte swizzle")]
#[test_case(BFloat16, [1 << 32, 1], [256, 256], Swizzle::None => matches Ok(_); "largest extent and box")]
#[test_case(UInt8, [16, 1], [16, 1], Swizzle::B32 => matches Ok(_); "byte row within the 32-byte swizzle")]
#[test_case(Int8, [64, 32], [16, 8], Swizzle::None => matches Err(Error::TensorMapDType { dtype: Int8 }); "no TMA type for i8")]
#[test_case(Bool, [64, 32], [16, 8], Swizzle::None => matches Err(Error::TensorMapDType { .. }); "no TMA type for bool")]
#[test_case(FP8E4M3, [64, 32], [16, 8], Swizzle::None => matches Err(Error::TensorMapDType { .. }); "no TMA type for fp8")]
#[test_case(Float32, [0, 32], [16, 8], Swizzle::None => matches Err(Error::TensorMapExtent { .. }); "empty row")]
#[test_case(Float32, [64, 0], [16, 8], Swizzle::None => matches Err(Error::TensorMapExtent { .. }); "no rows")]
#[test_case(UInt8, [(1 << 32) + 16, 1], [16, 1], Swizzle::None => matches Err(Error::TensorMapExtent { .. }); "row past 2^32 elements")]
#[test_case(Float32, [6, 32], [4, 8], Swizzle::None => matches Err(Error::TensorMapRowStride { bytes: 24 }); "row stride off the 16-byte grid")]
#[test_case(Float64, [1 << 32, 1 << 8], [16, 8], Swizzle::None => matches Ok(_); "the widest row stays below 2^40 bytes")]
#[test_case(Float32, [64, 32], [0, 8], Swizzle::None => matches Err(Error::TensorMapTile { .. }); "empty box row")]
#[test_case(Float32, [64, 32], [16, 257], Swizzle::None => matches Err(Error::TensorMapTile { .. }); "box past 256 rows")]
#[test_case(Float32, [64, 32], [2, 8], Swizzle::None => matches Err(Error::TensorMapTileRow { bytes: 8 }); "box row off the 16-byte grid")]
#[test_case(Float16, [64, 32], [32, 8], Swizzle::B32 => matches Err(Error::TensorMapSwizzleSpan { bytes: 64, span: 32 }); "box row wider than its swizzle")]
#[test_case(Float32, [64, 32], [64, 8], Swizzle::B128 => matches Err(Error::TensorMapSwizzleSpan { bytes: 256, span: 128 }); "f32 row wider than 128 bytes")]
fn template_validation(
    dtype: ScalarDType,
    extent: [u64; 2],
    tile: [u32; 2],
    swizzle: TensorMapSwizzle,
) -> crate::Result<TensorMap> {
    TensorMap::new(0, dtype, extent, tile, swizzle)
}

fn f32_map(buffer: usize) -> TensorMap {
    TensorMap::new(buffer, Float32, [64, 32], [16, 8], Swizzle::None).unwrap()
}

/// A tensor-map PARAM is typed by its element dtype, has no address space and
/// is not a scalar.
#[test]
fn tensor_map_param_is_neither_storage_nor_scalar() {
    let param = UOp::tensor_map_param(3, f32_map(1));
    assert_eq!(param.dtype(), DType::Float32);
    let crate::Op::Param(crate::ops::Param { arg, .. }) = param.op() else { panic!("PARAM") };
    assert_eq!((arg.slot, arg.addrspace, arg.tensor_map.as_ref()), (3, None, Some(&f32_map(1))));
    assert!(!arg.is_scalar());
    assert!(crate::ParamArg::variable("n".into(), DType::Int32, 0, 4).is_scalar());
}

/// A kernel reads a mapped buffer through its map: the map's buffer is an
/// input, the map itself neither a global nor a var.
#[test]
fn program_info_reads_a_buffer_through_its_map() {
    let map = UOp::tensor_map_param(1, f32_map(0));
    let src = UOp::param(0, 64 * 32, DType::Float32, None);
    let read = UOp::custom(smallvec::smallvec![map, src], "{0} {1}".into(), DType::Float32);
    let dst = UOp::index().buffer(UOp::param(2, 1, DType::Float32, None)).indices(vec![UOp::index_const(0)]).call();
    let sink = UOp::sink(vec![dst.unwrap().store(read)]);
    let info = ProgramInfo::from_sink(&sink, DeviceSpec::Cuda { device_id: 0 });
    assert_eq!((info.globals, info.ins, info.outs), (vec![0, 2], vec![0], vec![2]));
    assert!(info.vars.is_empty());
}

fn canonical_json(root: &Arc<UOp>) -> String {
    CanonicalGraph::from_root("kernel", root).unwrap().to_pretty_json().unwrap()
}

/// The template is part of the canonical identity; other PARAMs keep their
/// form.
#[test]
fn canonical_form_carries_the_template() {
    let json = canonical_json(&UOp::tensor_map_param(1, f32_map(0)));
    assert!(json.contains("\"tensor_map\""), "{json}");
    assert!(!canonical_json(&UOp::param(0, 4, DType::Float32, None)).contains("tensor_map"));
    let tile = TensorMap::new(0, Float32, [64, 32], [32, 8], Swizzle::None).unwrap();
    assert_ne!(json, canonical_json(&UOp::tensor_map_param(1, tile.clone())));
    let encode = |map: TensorMap| {
        let mut bytes = Vec::new();
        CanonicalGraph::from_root("kernel", &UOp::tensor_map_param(1, map)).unwrap().encode_into(&mut bytes).unwrap();
        bytes
    };
    assert_ne!(encode(f32_map(0)), encode(tile), "the binary identity sees the template too");
}

proptest! {
    /// Every accepted template keeps the driver's invariants.
    #[test]
    fn accepted_templates_satisfy_the_driver(
        dtype in prop::sample::select(vec![UInt8, UInt16, Float16, BFloat16, Float32, Int32, Float64, Int64]),
        extent in [1u64..=1 << 12, 1u64..=1 << 12],
        tile in [1u32..=300, 1u32..=300],
        swizzle in prop::sample::select(vec![Swizzle::None, Swizzle::B32, Swizzle::B64, Swizzle::B128]),
    ) {
        if let Ok(map) = TensorMap::new(0, dtype, extent, tile, swizzle) {
            let element = dtype.bytes() as u64;
            prop_assert_eq!(map.row_bytes() % 16, 0);
            prop_assert!(tile.iter().all(|&dim| (1..=256).contains(&dim)));
            prop_assert_eq!(u64::from(tile[0]) * element % 16, 0);
            prop_assert!(swizzle.span().is_none_or(|span| u64::from(tile[0]) * element <= span));
        }
    }
}
