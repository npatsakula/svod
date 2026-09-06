//! CUDA device integration tests: factory registration, compile → load
//! through the runtime glue, compiler identity stability and graph replay.
//! Hardware tests self-skip on hosts without a CUDA device.

use svod_device::Allocator;
use svod_device::allocator::RawBuffer;
use svod_device::cuda::{CudaAllocator, CudaDevice, has_devices};
use svod_device::device::{AbiParamDescriptor, AbiParamKind, GraphKernel, ProgramSpec};
use svod_device::registry::{DeviceRegistry, resolve_cuda_arch};
use svod_dtype::{AddrSpace, CudaArch, DType, DeviceSpec};
use svod_ir::UOp;

use crate::clang::ClangToolchain;
use crate::cuda::compile::{Ptxas, validate_ptx};
use crate::devices::cuda::{create_cuda_codegen, create_cuda_device, create_cuda_program, cuda_compiler_identity};
use crate::object_cache::ObjectCacheKey;

/// `data0[i] = data1[i] + data2[i]` for one block of 32 threads per grid slot.
const VADD_IR: &str = r#"; ModuleID = 'vadd'
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()

define ptx_kernel void @vadd(ptr noalias %data0, ptr noalias %data1, ptr noalias %data2) {
entry:
  %g = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %l = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %gs = mul i32 %g, 32
  %i = add i32 %gs, %l
  %i64 = zext i32 %i to i64
  %p1 = getelementptr float, ptr %data1, i64 %i64
  %p2 = getelementptr float, ptr %data2, i64 %i64
  %p0 = getelementptr float, ptr %data0, i64 %i64
  %a = load float, ptr %p1
  %b = load float, ptr %p2
  %s = fadd float %a, %b
  store float %s, ptr %p0
  ret void
}
"#;

fn storage(slot: usize) -> AbiParamDescriptor {
    AbiParamDescriptor { slot, kind: AbiParamKind::Storage(AddrSpace::Global), dtype: DType::Float32, name: None }
}

fn vadd_spec() -> ProgramSpec {
    let mut spec =
        ProgramSpec::new("vadd".into(), VADD_IR.into(), DeviceSpec::Cuda { device_id: 0 }, UOp::sink(vec![]));
    spec.abi = vec![storage(0), storage(1), storage(2)];
    spec.buf_count = 3;
    spec
}

fn arch_or_skip() -> Option<CudaArch> {
    if let Ok(arch) = resolve_cuda_arch(0) {
        return Some(arch);
    }
    eprintln!("skipping CUDA runtime test: no CUDA device on this host");
    None
}

/// The `CUDA` factory exists exactly when a device does; `CUDA:0` resolves
/// through the process-global registry to a CUDA device.
#[test]
fn factory_registered_iff_device_present() {
    let spec = DeviceSpec::Cuda { device_id: 0 };
    match crate::DEVICE_FACTORIES.device(&spec, svod_device::registry::registry()) {
        Ok(device) => {
            assert!(has_devices());
            assert_eq!(device.device, spec);
            assert_eq!(device.base_device_key(), "CUDA");
            assert!(device.graph.is_some(), "CUDA devices replay static plans as graphs");
        }
        Err(error) => {
            assert!(!has_devices(), "factory missing although a device exists: {error}");
            assert!(matches!(error, crate::Error::UnsupportedDevice { .. }), "{error:?}");
        }
    }
}

/// The compiler emits a cubin when `ptxas` is installed and PTX text
/// otherwise; either loads through the runtime factory.
#[test]
fn compiler_emits_a_loadable_kernel_for_the_device_arch() {
    let Some(arch) = arch_or_skip() else { return };
    let device = create_cuda_device(&DeviceRegistry::default(), 0, arch).unwrap();
    let compiled = device.compiler.compile(&vadd_spec()).expect("compile NVPTX IR");
    assert!(compiled.src.is_none() && !compiled.bytes.is_empty());
    let ptxas = Ptxas::discover(None);
    assert_eq!(svod_device::cuda::is_cubin(&compiled.bytes), ptxas.is_some());
    match ptxas {
        Some(_) => svod_device::cuda::validate_cubin(&compiled.bytes, "vadd").unwrap(),
        None => validate_ptx(&compiled.bytes, arch, "vadd").unwrap(),
    }
    let dev = CudaDevice::open(0).unwrap();
    assert_eq!(dev.arch(), arch);
    let program = create_cuda_program(&dev, &compiled).expect("load compiled kernel");
    assert_eq!(program.name(), "vadd");

    let mut broken = vadd_spec();
    broken.src = "define ptx_kernel void @vadd() { nonsense }".into();
    assert!(device.compiler.compile(&broken).is_err(), "clang diagnostics must fail the compile stage");
}

#[test]
fn compiler_identity_is_stable_and_backend_specific() {
    let Some(arch) = arch_or_skip() else { return };
    let (_, first) = create_cuda_codegen(0, arch).unwrap();
    let (_, second) = create_cuda_codegen(0, arch).unwrap();
    assert_eq!(first.cache_key(), second.cache_key());
    assert!(first.cache_key().starts_with("nvptx-clang:"), "{}", first.cache_key());
    let (_, cpu) = crate::devices::cpu::create_cpu_codegen(crate::devices::cpu::CpuBackend::Llvm).unwrap();
    assert_ne!(first.cache_key(), cpu.cache_key());
    // Another compute capability is another `-march`, hence another cache slot.
    let (_, other) = create_cuda_codegen(0, CudaArch { major: 7, minor: 5 }).unwrap();
    assert_ne!(first.cache_key(), other.cache_key());
    assert_eq!(first.compile(&vadd_spec()).unwrap().bytes, second.compile(&vadd_spec()).unwrap().bytes);
}

/// The PTX and cubin object formats never share a cache entry, and the
/// cubin identity records both tools.
#[test]
fn identity_differs_between_ptx_and_cubin_formats() {
    let clang = match ClangToolchain::discover(None) {
        Ok(clang) => clang,
        Err(error) => {
            eprintln!("skipping: {error}");
            return;
        }
    };
    let arch = CudaArch { major: 8, minor: 6 };
    let ptxas = Ptxas::fake("ptxas:test");
    let ptx = cuda_compiler_identity(arch, &clang, None);
    let cubin = cuda_compiler_identity(arch, &clang, Some(&ptxas));
    assert_eq!((ptx.object_format.as_str(), cubin.object_format.as_str()), ("ptx-text-v1", "cubin-v1"));
    assert_ne!(ptx.cache_key(), cubin.cache_key());
    assert_eq!(ptx.toolchain, clang.identity());
    assert!(cubin.toolchain.starts_with(clang.identity()) && cubin.toolchain.ends_with("ptxas:test"));
    assert!(cubin.flags.contains(&"-arch=sm_86".to_string()) && !ptx.flags.contains(&"-arch=sm_86".to_string()));
    assert_eq!(cuda_compiler_identity(arch, &clang, Some(&ptxas)), cubin, "the identity is a pure function");
    let ptx_key = ObjectCacheKey::new(b"src", ptx).digest();
    assert_ne!(ptx_key, ObjectCacheKey::new(b"src", cubin).digest());
}

/// A compiled cubin round-trips through the runtime factory and runs; an
/// ABI that disagrees with the entry is caught before `ptxas` ever runs.
#[test]
fn cubin_from_ptxas_loads_and_runs() {
    let Some(arch) = arch_or_skip() else { return };
    if Ptxas::discover(None).is_none() {
        eprintln!("skipping: no ptxas on this host");
        return;
    }
    let device = create_cuda_device(&DeviceRegistry::default(), 0, arch).unwrap();
    let compiled = device.compiler.compile(&vadd_spec()).unwrap();
    assert!(svod_device::cuda::is_cubin(&compiled.bytes));
    let dev = CudaDevice::open(0).unwrap();
    let program = create_cuda_program(&dev, &compiled).unwrap();
    assert_eq!(run_vadd(program.as_ref()), (0..64).map(|i| 1000.0 + 2.0 * i as f32).collect::<Vec<_>>());
    let mut narrow = vadd_spec();
    narrow.abi.truncate(2);
    narrow.buf_count = 2;
    let error = device.compiler.compile(&narrow).expect_err("two ABI slots for a three-parameter entry");
    assert!(format!("{error}").contains("3 parameters"), "{error}");
}

/// `data0 = data1 + data2` over 64 floats with `data1[i] = i`,
/// `data2[i] = 1000 + i`.
fn run_vadd(program: &dyn svod_device::device::Program) -> Vec<f32> {
    let alloc = CudaAllocator::new(0).unwrap();
    let spec = svod_device::BufferSpec::default();
    const N: usize = 64;
    let buffers: Vec<_> = (0..3).map(|_| alloc._alloc(N * 4, &spec, true).unwrap()).collect();
    let a: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..N).map(|i| 1000.0 + i as f32).collect();
    alloc._copyin(&buffers[1], 0, bytemuck_cast(&a)).unwrap();
    alloc._copyin(&buffers[2], 0, bytemuck_cast(&b)).unwrap();
    let pointers: Vec<*mut u8> = buffers
        .iter()
        .map(|buffer| match buffer {
            RawBuffer::Cuda { device_ptr, .. } => *device_ptr as *mut u8,
            other => unreachable!("{other:?}"),
        })
        .collect();
    // SAFETY: the pointers are live device allocations sized for 64 floats.
    unsafe { program.execute(&pointers, &[], Some([2, 1, 1]), Some([32, 1, 1]), true) }.unwrap();
    let mut out = vec![0u8; N * 4];
    alloc._copyout(&mut out, &buffers[0], 0).unwrap();
    for buffer in buffers {
        alloc._free(buffer, &spec);
    }
    out.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
}

#[test]
fn runtime_factory_rejects_empty_bytes_and_bad_abi() {
    let Some(arch) = arch_or_skip() else { return };
    let dev = CudaDevice::open(0).unwrap();
    let device = create_cuda_device(&DeviceRegistry::default(), 0, arch).unwrap();
    let mut compiled = device.compiler.compile(&vadd_spec()).unwrap();
    compiled.bytes.clear();
    let Err(error) = create_cuda_program(&dev, &compiled) else { panic!("empty bytes must be rejected") };
    assert!(format!("{error}").contains("empty kernel image"), "{error}");

    let mut compiled = device.compiler.compile(&vadd_spec()).unwrap();
    compiled.buf_count = 2;
    let Err(error) = create_cuda_program(&dev, &compiled) else { panic!("ABI projection mismatch must be rejected") };
    assert!(matches!(error, svod_device::Error::ProgramAbiMismatch { .. }), "{error:?}");
}

#[test]
fn cuda_spec_round_trips_through_registry_parse() {
    use svod_device::registry::DeviceSpecExt;
    assert_eq!(DeviceSpec::parse("CUDA:1").unwrap(), DeviceSpec::Cuda { device_id: 1 });
    assert_eq!(DeviceSpec::Cuda { device_id: 0 }.base_type(), "CUDA");
}

/// A static chain of loaded kernels is captured by the device's graph
/// factory, and replaying it computes the same result as an eager launch.
#[test]
fn device_installs_graphs_that_replay_eager_results() {
    let Some(arch) = arch_or_skip() else { return };
    let device = create_cuda_device(&DeviceRegistry::default(), 0, arch).unwrap();
    let factory = device.graph.clone().expect("CUDA devices graph static plans");
    let compiled = device.compiler.compile(&vadd_spec()).unwrap();
    let dev = CudaDevice::open(0).unwrap();
    let program = create_cuda_program(&dev, &compiled).unwrap();
    let alloc = CudaAllocator::new(0).unwrap();
    let spec = svod_device::BufferSpec::default();
    const N: usize = 64;
    let buffers: Vec<_> = (0..3).map(|_| alloc._alloc(N * 4, &spec, true).unwrap()).collect();
    let a: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..N).map(|i| 1000.0 + i as f32).collect();
    alloc._copyin(&buffers[1], 0, bytemuck_cast(&a)).unwrap();
    alloc._copyin(&buffers[2], 0, bytemuck_cast(&b)).unwrap();
    let pointers: Vec<*mut u8> = buffers
        .iter()
        .map(|buffer| match buffer {
            RawBuffer::Cuda { device_ptr, .. } => *device_ptr as *mut u8,
            other => unreachable!("{other:?}"),
        })
        .collect();
    let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
    let read_back = |alloc: &CudaAllocator| {
        let mut out = vec![0u8; N * 4];
        alloc._copyout(&mut out, &buffers[0], 0).unwrap();
        out.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect::<Vec<_>>()
    };

    // SAFETY: the pointers are live device allocations sized for 64 floats.
    unsafe { program.execute(&pointers, &[], Some([2, 1, 1]), Some([32, 1, 1]), true) }.unwrap();
    assert_eq!(read_back(&alloc), expected, "eager launch");

    alloc._copyin(&buffers[0], 0, &vec![0u8; N * 4]).unwrap();
    let kernel = GraphKernel {
        program: program.as_ref(),
        buffers: pointers,
        vals: vec![],
        global_size: Some([2, 1, 1]),
        local_size: Some([32, 1, 1]),
        deps: vec![],
    };
    let graph = factory(&[kernel]).unwrap().expect("static CUDA chain is graphable");
    graph.replay(&[], &[]).unwrap();
    dev.synchronize().unwrap();
    assert_eq!(read_back(&alloc), expected, "graph replay");
    for buffer in buffers {
        alloc._free(buffer, &spec);
    }
}

fn bytemuck_cast(values: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding and any bit pattern is a valid byte.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values)) }
}
