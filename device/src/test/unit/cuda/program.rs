use std::sync::Arc;
use std::time::Instant;

use test_case::test_case;

use super::{
    KERNELS_PTX, cuda_alloc_or_skip, cuda_device_or_skip, device_ptr, download, load, scalar, scale_abi, storage,
    upload, vadd_abi,
};
use crate::cuda::program::{extra_array, ptx_entry_params};
use crate::cuda::{CudaProgram, Launch, is_cubin, validate_cubin};
use crate::device::{AbiParamDescriptor, Program};
use crate::hcq::ClikeKernargLayout;

/// The kernarg blob a launch hands the driver: pointers at 8-byte and scalars
/// at 4-byte alignment in slot order, exactly PTX's `.param` layout.
#[test_case(vec![storage(0), storage(1), storage(2)], &[0x1000, 0x2000, 0x3000], &[], 24; "buffers only")]
#[test_case(vec![storage(0), storage(1), scalar(2, "n")], &[0x1000, 0x2000], &[7], 20; "trailing scalar")]
#[test_case(vec![scalar(0, "a"), storage(1)], &[0xABCD], &[-1], 16; "scalar then pointer pads")]
#[test_case(vec![scalar(0, "a"), scalar(1, "b"), storage(2)], &[0x10], &[1, 2], 16; "two scalars pack")]
fn kernarg_blob_follows_ptx_param_layout(abi: Vec<AbiParamDescriptor>, buffers: &[u64], vals: &[i64], size: usize) {
    let layout = ClikeKernargLayout::from_abi(&abi);
    assert_eq!(layout.packed_size(), size);
    let mut blob = vec![0u8; size];
    assert_eq!(layout.pack(&mut blob, buffers, vals).unwrap(), size);
    let (mut cursor, mut buffer, mut val) = (0usize, 0usize, 0usize);
    for param in &abi {
        if param.is_storage() {
            cursor = cursor.next_multiple_of(8);
            assert_eq!(u64::from_le_bytes(blob[cursor..cursor + 8].try_into().unwrap()), buffers[buffer]);
            buffer += 1;
            cursor += 8;
        } else {
            cursor = cursor.next_multiple_of(4);
            assert_eq!(i32::from_le_bytes(blob[cursor..cursor + 4].try_into().unwrap()), vals[val] as i32);
            val += 1;
            cursor += 4;
        }
    }
    let mut blob_size = blob.len();
    let extra = extra_array(&mut blob, &mut blob_size);
    assert_eq!(extra[0] as usize, 1);
    assert_eq!(extra[1], blob.as_mut_ptr().cast());
    assert_eq!(extra[2] as usize, 2);
    assert_eq!(unsafe { *extra[3].cast::<usize>() }, size);
    assert!(extra[4].is_null());
}

#[test]
fn vector_add_executes_on_the_gpu() {
    let Some(alloc) = cuda_alloc_or_skip() else { return };
    let program = load(&alloc.dev, "vadd", &vadd_abi());
    assert_eq!(program.name(), "vadd");
    assert_eq!((program.layout().globals, program.layout().vars), (3, 0));
    const N: usize = 1024;
    let a: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..N).map(|i| (2 * i) as f32).collect();
    let (out, a_buf, b_buf) = (upload(&alloc, &vec![0.0; N]), upload(&alloc, &a), upload(&alloc, &b));
    unsafe {
        program.execute(
            &[device_ptr(&out), device_ptr(&a_buf), device_ptr(&b_buf)],
            &[],
            Some([N / 32, 1, 1]),
            Some([32, 1, 1]),
            true,
        )
    }
    .expect("dispatch");
    let expected: Vec<f32> = (0..N).map(|i| (3 * i) as f32).collect();
    assert_eq!(download(&alloc, &out, N), expected);
    // Sub-buffer views bind through their device offset.
    let shifted = unsafe { device_ptr(&out).add(64 * 4) };
    unsafe {
        program.execute(
            &[shifted, device_ptr(&a_buf), device_ptr(&b_buf)],
            &[],
            Some([1, 1, 1]),
            Some([64, 1, 1]),
            true,
        )
    }
    .unwrap();
    let result = download(&alloc, &out, N);
    assert_eq!(&result[64..128], &expected[..64]);
    assert_eq!(&result[..64], &expected[..64]);
}

#[test]
fn scalar_argument_is_packed_as_i32() {
    let Some(alloc) = cuda_alloc_or_skip() else { return };
    let program = load(&alloc.dev, "scale", &scale_abi());
    let a: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let (out, a_buf) = (upload(&alloc, &vec![0.0; 64]), upload(&alloc, &a));
    unsafe { program.execute(&[device_ptr(&out), device_ptr(&a_buf)], &[5], Some([2, 1, 1]), Some([32, 1, 1]), true) }
        .unwrap();
    assert_eq!(download(&alloc, &out, 64), a.iter().map(|x| x * 5.0).collect::<Vec<_>>());
    unsafe { program.execute(&[device_ptr(&out), device_ptr(&a_buf)], &[-3], Some([2, 1, 1]), Some([32, 1, 1]), true) }
        .unwrap();
    assert_eq!(download(&alloc, &out, 64), a.iter().map(|x| x * -3.0).collect::<Vec<_>>());
    let error = unsafe { program.execute(&[device_ptr(&out)], &[5], Some([2, 1, 1]), Some([32, 1, 1]), true) }
        .expect_err("arity is checked");
    assert!(matches!(error, crate::Error::ProgramAbiMismatch { .. }), "{error:?}");
    let error = unsafe { program.execute(&[device_ptr(&out), device_ptr(&a_buf)], &[], None, None, true) }
        .expect_err("scalar arity is checked");
    assert!(matches!(error, crate::Error::ProgramAbiMismatch { .. }), "{error:?}");
}

/// `global_size` is the grid in blocks, `local_size` the block in threads.
#[test]
fn launch_dims_are_blocks_and_threads() {
    let Some(dev) = cuda_device_or_skip() else { return };
    let program = load(&dev, "vadd", &vadd_abi());
    assert_eq!(program.launch_dims(None, None).unwrap(), Launch { grid: [1, 1, 1], block: [1, 1, 1] });
    assert_eq!(
        program.launch_dims(Some([7, 3, 2]), Some([32, 4, 1])).unwrap(),
        Launch { grid: [7, 3, 2], block: [32, 4, 1] }
    );
    let error = program.launch_dims(Some([1, 1, 1]), Some([2048, 1, 1])).expect_err("block too large");
    let message = format!("{error}");
    for needle in ["maxThreadsPerBlock", "numRegs", "sharedSizeBytes", "localSizeBytes"] {
        assert!(message.contains(needle), "{message}");
    }
    assert!(program.launch_dims(Some([0, 1, 1]), None).is_err());
    assert!(program.launch_dims(Some([1 << 40, 1, 1]), None).is_err());
}

/// Two `wait=false` dispatches chained through a buffer, then a copyout: the
/// copy drains the device so the second kernel's result is visible.
#[test]
fn async_dispatches_are_drained_by_copyout() {
    let Some(alloc) = cuda_alloc_or_skip() else { return };
    let program = load(&alloc.dev, "vadd", &vadd_abi());
    const N: usize = 1 << 16;
    let ones = vec![1.0f32; N];
    let (a, b, mid, out) =
        (upload(&alloc, &ones), upload(&alloc, &ones), upload(&alloc, &vec![0.0; N]), upload(&alloc, &vec![0.0; N]));
    for _ in 0..8 {
        unsafe {
            program
                .execute(
                    &[device_ptr(&mid), device_ptr(&a), device_ptr(&b)],
                    &[],
                    Some([N / 32, 1, 1]),
                    Some([32, 1, 1]),
                    false,
                )
                .unwrap();
            program
                .execute(
                    &[device_ptr(&out), device_ptr(&mid), device_ptr(&b)],
                    &[],
                    Some([N / 32, 1, 1]),
                    Some([32, 1, 1]),
                    false,
                )
                .unwrap();
        }
    }
    assert!(download(&alloc, &out, N).iter().all(|value| *value == 3.0));
}

#[test]
fn shared_memory_kernel_runs_and_reports_lds() {
    let Some(alloc) = cuda_alloc_or_skip() else { return };
    let program = load(&alloc.dev, "tile", &vadd_abi()[..1]);
    let out = upload(&alloc, &vec![0.0; 256]);
    unsafe { program.execute(&[device_ptr(&out)], &[], Some([2, 1, 1]), Some([128, 1, 1]), true) }.unwrap();
    let expected: Vec<f32> = (0..256).map(|i| (127 - i % 128) as f32).collect();
    assert_eq!(download(&alloc, &out, 256), expected);
    assert_eq!(program.resource_usage().unwrap().lds_bytes, 4096);
}

#[test]
fn jit_errors_carry_the_driver_log() {
    let Some(dev) = cuda_device_or_skip() else { return };
    let broken = KERNELS_PTX.replace("add.f32 \t%f3, %f1, %f2;", "add.f32 \t%f3, %f1, %nope;");
    let error = CudaProgram::load_ptx(Arc::clone(&*dev), broken.as_bytes(), "vadd", &vadd_abi()).expect_err("bad PTX");
    let crate::Error::CudaJit { kernel, cause, log } = &error else { panic!("{error:?}") };
    assert_eq!(kernel, "vadd");
    assert!(cause.contains("CUDA_ERROR"), "{cause}");
    assert!(log.contains("nope") || log.contains("ptxas") || log.contains("error"), "{log}");
    let error = CudaProgram::load_ptx(Arc::clone(&*dev), KERNELS_PTX.as_bytes(), "not_there", &vadd_abi())
        .expect_err("missing entry point");
    assert!(format!("{error}").contains("not_there"), "{error}");
    let error = CudaProgram::load_ptx(Arc::clone(&*dev), &[], "vadd", &vadd_abi()).expect_err("empty image");
    assert!(matches!(error, crate::Error::CudaJit { .. }), "{error:?}");
    // ABI validation happens before any driver call.
    let error = CudaProgram::load_ptx(Arc::clone(&*dev), KERNELS_PTX.as_bytes(), "vadd", &[storage(1), storage(0)])
        .expect_err("unsorted slots");
    assert!(matches!(error, crate::Error::ProgramAbiMismatch { .. }), "{error:?}");
}

/// The entry's `.param` list is checked against the ABI before the JIT: a
/// kernarg blob laid out for fewer or narrower parameters than the kernel
/// declares would otherwise be read past its end on the GPU.
#[test]
fn load_rejects_an_abi_that_disagrees_with_the_ptx_entry() {
    let Some(dev) = cuda_device_or_skip() else { return };
    let short = CudaProgram::load_ptx(Arc::clone(&*dev), KERNELS_PTX.as_bytes(), "vadd", &vadd_abi()[..2])
        .expect_err("two ABI slots for a three-parameter entry");
    assert!(matches!(short, crate::Error::ProgramAbiMismatch { .. }), "{short:?}");
    assert!(format!("{short}").contains("3 parameters"), "{short}");
    let widened = CudaProgram::load_ptx(Arc::clone(&*dev), KERNELS_PTX.as_bytes(), "scale", &vadd_abi())
        .expect_err("a 32-bit scalar described as an 8-byte buffer");
    assert!(matches!(widened, crate::Error::ProgramAbiMismatch { .. }), "{widened:?}");
    assert!(format!("{widened}").contains("parameter 2"), "{widened}");
    assert!(CudaProgram::load_ptx(Arc::clone(&*dev), KERNELS_PTX.as_bytes(), "scale", &scale_abi()).is_ok());
}

/// `(name, byte width)` of every `.param` of the named entry, across
/// `.visible`/`.weak` prefixes, multi-line lists and name prefixes.
#[test_case(KERNELS_PTX, "vadd" => Some(vec![("vadd_param_0".into(), Some(8)), ("vadd_param_1".into(), Some(8)), ("vadd_param_2".into(), Some(8))]); "visible entry, multi-line")]
#[test_case(KERNELS_PTX, "scale" => Some(vec![("scale_param_0".into(), Some(8)), ("scale_param_1".into(), Some(8)), ("scale_param_2".into(), Some(4))]); "trailing u32 scalar")]
#[test_case(KERNELS_PTX, "tile" => Some(vec![("tile_param_0".into(), Some(8))]); "one parameter")]
#[test_case(KERNELS_PTX, "va" => None; "entry name is a prefix")]
#[test_case(KERNELS_PTX, "vadd_" => None; "entry name is longer")]
#[test_case(".entry k()\n{ ret; }", "k" => Some(vec![]); "bare entry without parameters")]
#[test_case(".weak .entry k(.param .u64 .ptr .align 1 k_param_0, .param .s32 k_param_1)\n{ ret; }", "k" => Some(vec![("k_param_0".into(), Some(8)), ("k_param_1".into(), Some(4))]); "weak entry, one line, pointer attributes")]
#[test_case(".visible .entry k(\n\t.param .align 8 .b8 k_param_0[16],\n\t.param .f16 k_param_1\n)\n{ ret; }", "k" => Some(vec![("k_param_0".into(), None), ("k_param_1".into(), Some(2))]); "byte array has no scalar width")]
#[test_case(".func f(.param .u64 f_param_0)\n{ ret; }", "f" => None; "func is not an entry")]
#[test_case(".visible .entry k\n(\n.param .u64 k_param_0\n)\n{ ret; }", "k" => Some(vec![("k_param_0".into(), Some(8))]); "parenthesis on the next line")]
fn ptx_entry_params_are_parsed_from_the_text(ptx: &str, entry: &str) -> Option<Vec<(String, Option<usize>)>> {
    ptx_entry_params(ptx, entry)
}

#[test]
fn timed_execution_reports_gpu_duration() {
    let Some(alloc) = cuda_alloc_or_skip() else { return };
    let program = load(&alloc.dev, "vadd", &vadd_abi());
    const N: usize = 1 << 20;
    let (out, a, b) = (upload(&alloc, &vec![0.0; N]), upload(&alloc, &vec![1.0; N]), upload(&alloc, &vec![2.0; N]));
    let wall = Instant::now();
    let gpu = unsafe {
        program.execute_timed(
            &[device_ptr(&out), device_ptr(&a), device_ptr(&b)],
            &[],
            Some([N / 256, 1, 1]),
            Some([256, 1, 1]),
        )
    }
    .unwrap()
    .expect("CUDA stamps every timed launch");
    let wall = wall.elapsed();
    assert!(gpu.as_nanos() > 0 && gpu <= wall, "gpu {gpu:?} wall {wall:?}");
    assert_eq!(download(&alloc, &out, N)[N - 1], 3.0);
}

/// Static function attributes plus the driver's occupancy for the latest
/// launch's block size.
#[test]
fn resource_usage_reports_function_attributes() {
    let Some(alloc) = cuda_alloc_or_skip() else { return };
    let program = load(&alloc.dev, "vadd", &vadd_abi());
    let resources = program.resource_usage().expect("CUDA reports attributes");
    assert!(resources.vgprs.is_some_and(|regs| regs > 0), "{resources:?}");
    assert_eq!((resources.sgprs, resources.lds_bytes, resources.scratch_bytes), (None, 0, Some(0)));
    assert_eq!(resources.wave_size, 32);
    let occupancy = resources.occupancy.expect("occupancy query");
    assert!(occupancy > 0.0 && occupancy <= 1.0, "{occupancy}");
    let (out, a, b) = (upload(&alloc, &[0.0; 32]), upload(&alloc, &[0.0; 32]), upload(&alloc, &[0.0; 32]));
    unsafe {
        program.execute(
            &[device_ptr(&out), device_ptr(&a), device_ptr(&b)],
            &[],
            Some([1, 1, 1]),
            Some([32, 1, 1]),
            true,
        )
    }
    .unwrap();
    let after = program.resource_usage().unwrap().occupancy.unwrap();
    let limits = alloc.dev.limits();
    let blocks = program.max_active_blocks_per_sm(32).unwrap();
    assert!((after - (blocks * 32) as f32 / limits.max_threads_per_sm as f32).abs() < 1e-6, "{after} vs {blocks}");
}

#[test]
fn device_reports_limits_and_arch() {
    let Some(dev) = cuda_device_or_skip() else { return };
    let limits = dev.limits();
    assert!(limits.sm_count > 0 && limits.max_threads_per_block >= 1024 && limits.warp_size == 32, "{limits:?}");
    assert!(limits.shared_per_block >= 48 << 10 && limits.max_threads_per_sm >= 1024);
    assert!(dev.arch().major >= 5, "{}", dev.arch());
    assert!(!dev.name().is_empty());
    let (free, total) = dev.memory_info().unwrap();
    assert!(0 < free && free <= total);
    assert!(Arc::ptr_eq(&dev, &crate::cuda::CudaDevice::open(0).unwrap()), "device cache");
    assert!(crate::cuda::has_devices());
    assert!(!dev.is_poisoned());
}

/// A 64-byte ELF header with no section table: enough for the header checks
/// of `validate_cubin`, never enough to define an entry.
fn elf_header(class: u8, data: u8, machine: u16) -> Vec<u8> {
    let mut header = vec![0u8; 64];
    header[..4].copy_from_slice(&object::elf::ELFMAG);
    header[4] = class;
    header[5] = data;
    header[6] = object::elf::EV_CURRENT;
    header[16..18].copy_from_slice(&object::elf::ET_EXEC.to_le_bytes());
    header[18..20].copy_from_slice(&machine.to_le_bytes());
    header[20..24].copy_from_slice(&1u32.to_le_bytes());
    header[52..54].copy_from_slice(&64u16.to_le_bytes());
    header[58..60].copy_from_slice(&64u16.to_le_bytes());
    header
}

/// Header-level rejections need no `ptxas`; a genuine cubin is exercised by
/// the runtime's compiler tests.
#[test_case(b"garbage".to_vec(), "k", "not an ELF64"; "not an elf")]
#[test_case(elf_header(object::elf::ELFCLASS64, object::elf::ELFDATA2LSB, object::elf::EM_CUDA)[..40].to_vec(), "k", "not an ELF64"; "truncated header")]
#[test_case(elf_header(object::elf::ELFCLASS32, object::elf::ELFDATA2LSB, object::elf::EM_CUDA), "k", "not an ELF64"; "elf32")]
#[test_case(elf_header(object::elf::ELFCLASS64, object::elf::ELFDATA2MSB, object::elf::EM_CUDA), "k", "little-endian"; "big endian")]
#[test_case(elf_header(object::elf::ELFCLASS64, object::elf::ELFDATA2LSB, object::elf::EM_X86_64), "k", "e_machine is 62"; "wrong machine")]
#[test_case(elf_header(object::elf::ELFCLASS64, object::elf::ELFDATA2LSB, object::elf::EM_CUDA), "k", "no entry \"k\""; "missing entry")]
fn validate_cubin_rejects_bad_images(image: Vec<u8>, entry: &str, cause: &str) {
    let error = validate_cubin(&image, entry).expect_err("rejected");
    let crate::Error::CudaJit { kernel, cause: got, .. } = &error else { panic!("{error:?}") };
    assert_eq!(kernel, entry);
    assert!(got.contains(cause), "{got}");
}

#[test]
fn cubin_is_told_from_ptx_by_the_elf_magic() {
    assert!(is_cubin(&elf_header(object::elf::ELFCLASS64, object::elf::ELFDATA2LSB, object::elf::EM_CUDA)));
    assert!(is_cubin(b"\x7fELF"));
    assert!(!is_cubin(KERNELS_PTX.as_bytes()));
    assert!(!is_cubin(b""));
    assert!(!is_cubin(b"\x7fEL"));
}

/// `CompiledSpec` bytes that look like a cubin go through the cubin
/// validator, so a bad image is rejected without a driver call.
#[test]
fn load_dispatches_on_the_image_format() {
    let Some(dev) = cuda_device_or_skip() else { return };
    let image = elf_header(object::elf::ELFCLASS64, object::elf::ELFDATA2LSB, object::elf::EM_CUDA);
    let error = CudaProgram::load_cubin(Arc::clone(&*dev), &image, "vadd", &vadd_abi()).expect_err("no entry");
    assert!(format!("{error}").contains("no entry"), "{error}");
    let error = CudaProgram::load_ptx(Arc::clone(&*dev), &image, "vadd", &vadd_abi()).expect_err("NUL bytes");
    assert!(format!("{error}").contains("NUL"), "{error}");
}
