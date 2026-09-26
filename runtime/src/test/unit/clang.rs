use super::*;

fn storage_abi(count: usize) -> Vec<svod_device::device::AbiParamDescriptor> {
    (0..count)
        .map(|slot| svod_device::device::AbiParamDescriptor {
            slot,
            kind: svod_device::device::AbiParamKind::Storage(svod_dtype::AddrSpace::Global),
            dtype: svod_dtype::DType::Float32,
            name: None,
        })
        .collect()
}

#[test]
fn test_clang_kernel_noop() {
    let src = "void test_kernel(void) { }\n";
    let kernel = ClangKernel::compile_with_abi(src, "test_kernel", vec![], &storage_abi(0)).unwrap();
    assert_eq!(kernel.name(), "test_kernel");
    unsafe {
        kernel.execute_with_vals(&[], &[]).unwrap();
    }
}

#[test]
fn test_clang_kernel_add() {
    let src = r#"
void add_kernel(float* restrict a, float* restrict b, float* restrict out) {
    out[0] = a[0] + b[0];
}
"#;
    let kernel = ClangKernel::compile_with_abi(src, "add_kernel", vec![], &storage_abi(3)).unwrap();

    let mut a = [1.0f32];
    let mut b = [2.0f32];
    let mut out = [0.0f32];

    let buffers = vec![a.as_mut_ptr() as *mut u8, b.as_mut_ptr() as *mut u8, out.as_mut_ptr() as *mut u8];

    unsafe {
        kernel.execute_with_vals(&buffers, &[]).unwrap();
    }

    assert_eq!(out[0], 3.0);
}

#[test]
fn cpu_object_validation_checks_header_and_entry_symbol() {
    let toolchain = ClangToolchain::discover(None).unwrap();
    let flags = c_object_flags();
    let object = compile_c_object(&toolchain, "void cached_kernel(void) {}\n", &flags).unwrap();
    validate_c_object(&object, "cached_kernel").unwrap();
    assert!(validate_c_object(&object, "other_kernel").is_err());

    let mut wrong_machine = object;
    wrong_machine[18..20].copy_from_slice(&0xffffu16.to_le_bytes());
    assert!(validate_c_object(&wrong_machine, "cached_kernel").is_err());
}

/// `-march=native` resolves against the running CPU, so its `-###` probe must
/// never be shared between hosts; an explicit arch must be.
#[test_case::test_case("-march=native" => false; "host-resolved arch")]
#[test_case::test_case("-mcpu=native" => false; "host-resolved cpu")]
#[test_case::test_case("-march=x86-64" => true; "explicit arch")]
fn probe_key_is_shared_between_hosts(flag: &str) -> bool {
    let flags = vec![flag.to_string()];
    let executable = [7u8; 32];
    probe_key(&executable, &flags, Some(&[1u8; 32])) == probe_key(&executable, &flags, Some(&[2u8; 32]))
}

#[test]
fn unfingerprintable_host_disables_probe_sharing_for_native_flags() {
    let executable = [7u8; 32];
    assert!(probe_key(&executable, &["-march=native".into()], None).is_none());
    assert!(probe_key(&executable, &["-march=x86-64".into()], None).is_some());
}

/// The LLVM major is read off the `clang version` line whoever packaged it; a
/// banner without one has no version to check.
#[test_case::test_case("clang version 22.1.8\nTarget: x86_64-pc-linux-gnu\n" => Some(22); "upstream")]
#[test_case::test_case("Ubuntu clang version 18.1.3 (1ubuntu1)\n" => Some(18); "distro prefix")]
#[test_case::test_case("Homebrew clang version 22.1.0\n" => Some(22); "homebrew")]
#[test_case::test_case("Apple clang version 17.0.0 (clang-1700.0.13.5)\n" => Some(17); "apple")]
#[test_case::test_case("AMD clang version 19.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.4.0)\n" => Some(19); "rocm")]
#[test_case::test_case("gcc (GCC) 15.2.1\n" => None; "not clang")]
fn clang_major_version_reads_the_banner(banner: &str) -> Option<u32> {
    crate::clang::clang_major_version(banner)
}
