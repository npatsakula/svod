//! ABI-retention preconditions for `AmdProgram::load`: the descriptor list is
//! kept verbatim (interleaved storage/scalar order), and malformed descriptors
//! are rejected before any ELF is touched.

use crate::amd::program::retain_program_abi;
use crate::device::{AbiParamDescriptor, AbiParamKind};
use svod_dtype::{AddrSpace, DType};

fn storage(slot: usize, name: Option<String>) -> AbiParamDescriptor {
    AbiParamDescriptor { slot, kind: AbiParamKind::Storage(AddrSpace::Global), dtype: DType::Float32, name }
}

fn scalar(slot: usize, dtype: DType, name: Option<String>) -> AbiParamDescriptor {
    AbiParamDescriptor { slot, kind: AbiParamKind::Scalar, dtype, name }
}

#[test]
fn loaded_program_retains_interleaved_abi_exactly() {
    let abi = vec![
        storage(0, None),
        scalar(1, DType::Int32, Some("low".into())),
        storage(2, None),
        scalar(3, DType::Int32, Some("high".into())),
    ];
    let (retained, buf_count, var_count) = retain_program_abi(&abi).unwrap();
    assert_eq!(retained, abi);
    assert_eq!((buf_count, var_count), (2, 2));
}

#[test_case::test_case(vec![storage(1, None), storage(0, None)]; "storage slots out of order")]
#[test_case::test_case(vec![storage(0, None), storage(0, None)]; "duplicate slot")]
#[test_case::test_case(vec![storage(usize::MAX, None)]; "slot out of range")]
#[test_case::test_case(vec![scalar(0, DType::Float32, Some("n".into()))]; "non-integer scalar")]
#[test_case::test_case(vec![scalar(0, DType::Int32, None)]; "unnamed scalar")]
#[test_case::test_case(vec![storage(0, Some("not_storage".into()))]; "named storage")]
#[test_case::test_case(vec![AbiParamDescriptor { dtype: DType::Void, ..storage(0, None) }]; "void storage dtype")]
fn loaded_program_rejects_malformed_descriptors_before_elf_loading(abi: Vec<AbiParamDescriptor>) {
    let err = retain_program_abi(&abi).expect_err("the ABI precondition must reject malformed input");
    assert!(
        matches!(&err, crate::Error::ProgramAbiMismatch { .. } | crate::Error::UnassignedProgramParam { .. }),
        "{err:?}"
    );
}

/// AMD kernels take no tensor map: one is rejected by slot before any ELF is
/// touched, never counted as a scalar.
#[test]
fn loaded_program_rejects_a_tensor_map() {
    let abi = vec![storage(0, None), crate::test::unit::hcq::tensor_map(1, 0)];
    let err = retain_program_abi(&abi).expect_err("AMD cannot pass a tensor map");
    assert!(matches!(err, crate::Error::UnsupportedTensorMapParam { backend: "AMD", slot: 1 }), "{err:?}");
}
