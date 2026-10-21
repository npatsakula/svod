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

/// A dump is named by its tag, module and IR: two programs of one kernel
/// name (configs of one tile kernel) get two files, one IR keeps its name
/// across runs, and its IR and assembly share a stem.
#[test]
fn dumps_of_one_kernel_name_do_not_collide() {
    let ir = |body: &str| format!("; ModuleID = 'gemm'\ndefine void @gemm() {{\n{body}\n}}\n");
    let (a, b) = (ir("  ret void"), ir("  br label %x\nx:\n  ret void"));
    let name = dump_name("gfx942", &a, "ll");
    assert!(name.starts_with("gfx942_gemm_") && name.ends_with(".ll"), "{name}");
    assert_eq!(name.len(), "gfx942_gemm_".len() + 8 + ".ll".len(), "{name}");
    assert_eq!(name, dump_name("gfx942", &a, "ll"));
    assert_ne!(name, dump_name("gfx942", &b, "ll"));
    assert_eq!(name.trim_end_matches(".ll"), dump_name("gfx942", &a, "s").trim_end_matches(".s"));
}

#[test_case::test_case("; ModuleID = 'r_4_4'\n", "sm_86_r_4_4_"; "a graph kernel")]
#[test_case::test_case("; ModuleID = 'a.b c'\n", "sm_86_a_b_c_"; "unsafe characters")]
#[test_case::test_case("define void @k() {\n}\n", "sm_86_unknown_"; "no module id")]
fn a_dump_is_named_by_its_module(ir: &str, stem: &str) {
    assert!(dump_name("sm_86", ir, "ll").starts_with(stem));
}

/// A compiler process past the thread's deadline is killed and its compile
/// fails at once; one that exits first returns its output whole, however much
/// it writes; outside the deadline's scope nothing is killed.
#[test]
fn a_compile_past_its_deadline_is_killed() {
    use std::process::Command;
    use std::time::{Duration, Instant};
    let sleeper = || {
        let mut c = Command::new("sh");
        c.args(["-c", "sleep 5; echo late"]);
        c
    };
    let start = Instant::now();
    let killed = with_compile_deadline(start + Duration::from_millis(100), || run_piped(sleeper(), b"", "sleep"));
    assert!(killed.is_err_and(|e| e.to_string().contains("deadline")));
    assert!(start.elapsed() < Duration::from_secs(2), "killed at the deadline, not after the sleep");
    let input = vec![b'x'; 1 << 20];
    let deadline = Instant::now() + Duration::from_secs(30);
    let echoed = with_compile_deadline(deadline, || run_piped(Command::new("cat"), &input, "cat"));
    assert_eq!(echoed.unwrap(), input);
    let mut quick = Command::new("sh");
    quick.args(["-c", "sleep 0.2; echo done"]);
    assert_eq!(run_piped(quick, b"", "sleep").unwrap(), b"done\n");
}
