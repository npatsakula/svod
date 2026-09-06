use std::collections::HashSet;
use std::mem::{offset_of, size_of};

use test_case::test_case;

use crate::cuda::sys::{
    CU_LAUNCH_PARAM_BUFFER_POINTER, CU_LAUNCH_PARAM_BUFFER_SIZE, CU_LAUNCH_PARAM_END, CUdeviceptr, CUresult, CUstream,
    CudaKernelNodeParams, SYMBOLS, api,
};

/// `CUDA_KERNEL_NODE_PARAMS_v2` as `cuda.h` lays it out on LP64.
#[test]
fn kernel_node_params_matches_cuda_h() {
    assert_eq!(size_of::<CudaKernelNodeParams>(), 72);
    assert_eq!(offset_of!(CudaKernelNodeParams, func), 0);
    assert_eq!(offset_of!(CudaKernelNodeParams, grid_dim_x), 8);
    assert_eq!(offset_of!(CudaKernelNodeParams, block_dim_z), 28);
    assert_eq!(offset_of!(CudaKernelNodeParams, shared_mem_bytes), 32);
    assert_eq!(offset_of!(CudaKernelNodeParams, kernel_params), 40);
    assert_eq!(offset_of!(CudaKernelNodeParams, extra), 48);
    assert_eq!(offset_of!(CudaKernelNodeParams, kern), 56);
    assert_eq!(offset_of!(CudaKernelNodeParams, ctx), 64);
    assert_eq!(size_of::<CUdeviceptr>(), 8);
    assert_eq!(size_of::<CUstream>(), size_of::<*mut ()>());
    assert_eq!(size_of::<CUresult>(), 4);
    assert_eq!((CU_LAUNCH_PARAM_END as usize, CU_LAUNCH_PARAM_BUFFER_POINTER as usize), (0, 1));
    assert_eq!(CU_LAUNCH_PARAM_BUFFER_SIZE as usize, 2);
}

/// The C compiler's view of the same layouts, when the toolkit headers are
/// installed (`/opt/cuda`, `/usr/local/cuda`); the Rust asserts above stand
/// alone elsewhere.
#[test]
fn kernel_node_params_matches_the_c_compiler() {
    let Some(include) = ["/opt/cuda/include", "/usr/local/cuda/include"]
        .into_iter()
        .find(|dir| std::path::Path::new(dir).join("cuda.h").exists())
    else {
        eprintln!("skipping C layout probe: no cuda.h");
        return;
    };
    let dir = std::env::temp_dir().join(format!("svod-cuda-layout-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("layout.c");
    std::fs::write(
        &source,
        "#include <cuda.h>\n#include <stdio.h>\n#include <stddef.h>\nint main(void){\n\
         printf(\"%zu %zu %zu %zu %zu %d %d %d %d %d %d %d\\n\", sizeof(CUDA_KERNEL_NODE_PARAMS_v2),\n\
         offsetof(CUDA_KERNEL_NODE_PARAMS_v2, kernelParams), offsetof(CUDA_KERNEL_NODE_PARAMS_v2, extra),\n\
         offsetof(CUDA_KERNEL_NODE_PARAMS_v2, kern), offsetof(CUDA_KERNEL_NODE_PARAMS_v2, ctx),\n\
         (int)(size_t)CU_LAUNCH_PARAM_BUFFER_POINTER, (int)(size_t)CU_LAUNCH_PARAM_BUFFER_SIZE,\n\
         CU_STREAM_NON_BLOCKING, CU_EVENT_DISABLE_TIMING, CU_MEMHOSTALLOC_DEVICEMAP,\n\
         CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, CU_FUNC_ATTRIBUTE_NUM_REGS);\n\
         return 0;}\n",
    )
    .unwrap();
    let binary = dir.join("layout");
    let compiled =
        std::process::Command::new("cc").arg(format!("-I{include}")).arg(&source).arg("-o").arg(&binary).status();
    if !compiled.is_ok_and(|status| status.success()) {
        eprintln!("skipping C layout probe: cc unavailable");
        return;
    }
    let output = std::process::Command::new(&binary).output().unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let text = String::from_utf8_lossy(&output.stdout);
    let values: Vec<usize> = text.split_whitespace().map(|v| v.parse().unwrap()).collect();
    assert_eq!(values, [72, 40, 48, 56, 64, 1, 2, 1, 2, 2, 75, 4], "{text}");
}

#[test]
fn symbol_table_is_unique_and_versioned() {
    let rust: HashSet<_> = SYMBOLS.iter().map(|(name, _)| *name).collect();
    let dlsym: HashSet<_> = SYMBOLS.iter().map(|(_, symbol)| *symbol).collect();
    assert_eq!((rust.len(), dlsym.len()), (SYMBOLS.len(), SYMBOLS.len()));
    assert!(dlsym.iter().all(|symbol| symbol.starts_with("cu")));
    // The plain names that `cuda.h` remaps must never be resolved.
    for legacy in [
        "cuMemAlloc",
        "cuMemFree",
        "cuMemGetInfo",
        "cuMemcpyHtoDAsync",
        "cuMemcpyDtoHAsync",
        "cuMemcpyDtoDAsync",
        "cuStreamDestroy",
        "cuEventDestroy",
        "cuGraphInstantiate",
        "cuGraphAddKernelNode",
        "cuGraphExecKernelNodeSetParams",
        "cuDevicePrimaryCtxRelease",
        "cuMemHostGetDevicePointer",
    ] {
        assert!(!dlsym.contains(legacy), "{legacy} must be bound by its versioned name");
    }
}

#[test_case("mem_alloc", "cuMemAlloc_v2")]
#[test_case("mem_free", "cuMemFree_v2")]
#[test_case("memcpy_htod_async", "cuMemcpyHtoDAsync_v2")]
#[test_case("event_elapsed_time", "cuEventElapsedTime")]
#[test_case("graph_instantiate_with_flags", "cuGraphInstantiateWithFlags")]
#[test_case("graph_add_kernel_node", "cuGraphAddKernelNode_v2")]
#[test_case("graph_exec_kernel_node_set_params", "cuGraphExecKernelNodeSetParams_v2")]
#[test_case("device_primary_ctx_release", "cuDevicePrimaryCtxRelease_v2")]
#[test_case("init", "cuInit")]
#[test_case("launch_kernel", "cuLaunchKernel")]
fn rust_names_map_to_exact_exports(rust: &str, symbol: &str) {
    assert!(SYMBOLS.contains(&(rust, symbol)), "{rust} -> {symbol}");
}

#[test_case(CUresult::SUCCESS, false; "success")]
#[test_case(CUresult::NOT_READY, false; "not ready")]
#[test_case(CUresult::INVALID_VALUE, false; "invalid value")]
#[test_case(CUresult::OUT_OF_MEMORY, false; "oom")]
#[test_case(CUresult::LAUNCH_OUT_OF_RESOURCES, false; "launch out of resources")]
#[test_case(CUresult::ILLEGAL_ADDRESS, true; "illegal address")]
#[test_case(CUresult::LAUNCH_FAILED, true; "launch failed")]
#[test_case(CUresult::LAUNCH_TIMEOUT, true; "launch timeout")]
#[test_case(CUresult::UNKNOWN, true; "unknown")]
fn sticky_errors_are_the_context_killing_ones(result: CUresult, sticky: bool) {
    assert_eq!(result.is_sticky(), sticky);
}

/// Without a driver the binding is unavailable with a typed error naming the
/// library and `has_devices` is false; with one, `check` speaks the driver's
/// own error vocabulary.
#[test]
fn driver_availability_is_consistent() {
    match api() {
        Ok(api) => {
            assert!(CUresult::SUCCESS.check("noop").is_ok());
            let error = CUresult::INVALID_VALUE.check("cuProbe").expect_err("nonzero code");
            let crate::Error::CudaDriver { call, code, name, message } = &error else { panic!("{error:?}") };
            assert_eq!((*call, *code, name.as_str()), ("cuProbe", 1, "CUDA_ERROR_INVALID_VALUE"));
            assert!(message.contains("invalid"), "{message}");
            assert!(format!("{error}").contains("CUDA cuProbe failed: CUDA_ERROR_INVALID_VALUE (1)"), "{error}");
            let (name, message) = CUresult(123_456).describe();
            assert_eq!((name.as_str(), message.as_str()), ("CUDA_ERROR_UNKNOWN", "unrecognized error code"));
            let (major, _) = api.driver_version().unwrap();
            assert!(major >= 11, "driver {major}");
        }
        Err(error) => {
            assert!(matches!(error, crate::Error::DeviceUnavailable { .. }), "{error:?}");
            assert!(format!("{error}").contains("cannot load"), "{error}");
            assert!(!crate::cuda::has_devices());
            let (name, message) = CUresult::INVALID_VALUE.describe();
            assert_eq!((name.as_str(), message.as_str()), ("CUDA_ERROR_UNKNOWN", "driver not loaded"));
        }
    }
}

/// Every declared export resolves against the real driver, and the binding
/// is consistent with `has_devices`.
#[test]
fn every_symbol_resolves_on_the_driver() {
    let Ok(api) = api() else { return };
    // SAFETY: the driver library is already resident.
    let library = unsafe { libloading::Library::new("libcuda.so.1") }.unwrap();
    for (rust, symbol) in SYMBOLS {
        // SAFETY: only the address is taken.
        let resolved = unsafe { library.get::<*const ()>(symbol.as_bytes()) };
        assert!(resolved.is_ok(), "{rust} -> {symbol} missing: {:?}", resolved.err());
    }
    assert_eq!(crate::cuda::has_devices(), api.init().and_then(|()| api.device_count()).unwrap_or(0) > 0);
}
