//! Hand-written CUDA driver API bindings, resolved from `libcuda.so.1` at
//! runtime with `libloading`.
//!
//! Nothing links against the CUDA toolkit: the module compiles on every host
//! and [`api`] reports [`crate::Error::DeviceUnavailable`] where the driver is
//! absent. Every entry point is resolved by its *versioned* export name (the
//! name `cuda.h` remaps to), so `cuMemAlloc` binds `cuMemAlloc_v2`, and the
//! plain `cuGraphInstantiate` (a legacy five-argument ABI) is never touched.

use std::ffi::{CStr, c_char, c_int, c_void};
use std::ptr::null_mut;
use std::sync::OnceLock;

use libloading::Library;

use crate::{Error, Result};

pub type CUdevice = c_int;
pub type CUdeviceptr = u64;

macro_rules! handles {
    ($($name:ident),* $(,)?) => {$(
        /// Opaque driver handle.
        #[repr(transparent)]
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub struct $name(pub *mut c_void);

        impl $name {
            pub const NULL: Self = Self(null_mut());
        }

        // SAFETY: driver handles are process-wide identifiers that the driver
        // documents as usable from any thread; they carry no thread affinity.
        unsafe impl Send for $name {}
        unsafe impl Sync for $name {}
    )*};
}

handles!(CUcontext, CUmodule, CUfunction, CUkernel, CUstream, CUevent, CUgraph, CUgraphNode, CUgraphExec);

/// `CUresult`: an enum in C, an integer newtype here so unknown codes from a
/// newer driver still round-trip through [`CUresult::check`].
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CUresult(pub c_int);

impl CUresult {
    pub const SUCCESS: Self = Self(0);
    pub const INVALID_VALUE: Self = Self(1);
    pub const OUT_OF_MEMORY: Self = Self(2);
    pub const DEINITIALIZED: Self = Self(4);
    pub const ECC_UNCORRECTABLE: Self = Self(214);
    pub const NOT_READY: Self = Self(600);
    pub const ILLEGAL_ADDRESS: Self = Self(700);
    pub const LAUNCH_OUT_OF_RESOURCES: Self = Self(701);
    pub const LAUNCH_TIMEOUT: Self = Self(702);
    pub const ASSERT: Self = Self(710);
    pub const HARDWARE_STACK_ERROR: Self = Self(714);
    pub const ILLEGAL_INSTRUCTION: Self = Self(715);
    pub const MISALIGNED_ADDRESS: Self = Self(716);
    pub const INVALID_ADDRESS_SPACE: Self = Self(717);
    pub const INVALID_PC: Self = Self(718);
    pub const LAUNCH_FAILED: Self = Self(719);
    pub const UNKNOWN: Self = Self(999);

    /// `Ok(())` for success, otherwise [`Error::CudaDriver`] naming `call`
    /// and carrying the driver's own name and description of the code.
    pub fn check(self, call: &'static str) -> Result<()> {
        if self == Self::SUCCESS {
            return Ok(());
        }
        let (name, message) = self.describe();
        Err(Error::CudaDriver { call, code: self.0, name, message })
    }

    /// Errors after which the context is unusable (the driver documents them
    /// as "sticky": every later call in the context fails the same way).
    pub fn is_sticky(self) -> bool {
        matches!(
            self,
            Self::DEINITIALIZED
                | Self::ECC_UNCORRECTABLE
                | Self::ILLEGAL_ADDRESS
                | Self::LAUNCH_TIMEOUT
                | Self::ASSERT
                | Self::HARDWARE_STACK_ERROR
                | Self::ILLEGAL_INSTRUCTION
                | Self::MISALIGNED_ADDRESS
                | Self::INVALID_ADDRESS_SPACE
                | Self::INVALID_PC
                | Self::LAUNCH_FAILED
                | Self::UNKNOWN
        )
    }

    /// `(cuGetErrorName, cuGetErrorString)`; placeholders when the driver
    /// itself does not know the code.
    pub fn describe(self) -> (String, String) {
        let Ok(api) = api() else { return ("CUDA_ERROR_UNKNOWN".into(), "driver not loaded".into()) };
        let lookup = |query: unsafe extern "C" fn(CUresult, *mut *const c_char) -> CUresult, fallback: &str| {
            let mut text: *const c_char = std::ptr::null();
            // SAFETY: the driver stores a pointer to a static string, or leaves it null.
            if unsafe { query(self, &mut text) } != Self::SUCCESS || text.is_null() {
                return fallback.to_string();
            }
            // SAFETY: non-null result of a successful lookup is a NUL-terminated static string.
            unsafe { CStr::from_ptr(text) }.to_string_lossy().into_owned()
        };
        (lookup(api.get_error_name, "CUDA_ERROR_UNKNOWN"), lookup(api.get_error_string, "unrecognized error code"))
    }
}

/// `CUDA_KERNEL_NODE_PARAMS_v2`, the kernel-node payload `cuda.h` maps
/// `CUDA_KERNEL_NODE_PARAMS` to. `kern`/`ctx` stay null: `func` selects the
/// kernel and the current context runs it.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CudaKernelNodeParams {
    pub func: CUfunction,
    pub grid_dim_x: u32,
    pub grid_dim_y: u32,
    pub grid_dim_z: u32,
    pub block_dim_x: u32,
    pub block_dim_y: u32,
    pub block_dim_z: u32,
    pub shared_mem_bytes: u32,
    pub kernel_params: *mut *mut c_void,
    pub extra: *mut *mut c_void,
    pub kern: CUkernel,
    pub ctx: CUcontext,
}

const _: () = assert!(std::mem::size_of::<CudaKernelNodeParams>() == 72);
const _: () = assert!(std::mem::align_of::<CudaKernelNodeParams>() == 8);
const _: () = assert!(std::mem::offset_of!(CudaKernelNodeParams, kernel_params) == 40);
const _: () = assert!(std::mem::offset_of!(CudaKernelNodeParams, extra) == 48);
const _: () = assert!(std::mem::offset_of!(CudaKernelNodeParams, kern) == 56);
const _: () = assert!(std::mem::offset_of!(CudaKernelNodeParams, ctx) == 64);

/// `cuLaunchKernel` / kernel-node `extra` sentinels.
pub const CU_LAUNCH_PARAM_END: *mut c_void = null_mut();
pub const CU_LAUNCH_PARAM_BUFFER_POINTER: *mut c_void = std::ptr::without_provenance_mut(0x01);
pub const CU_LAUNCH_PARAM_BUFFER_SIZE: *mut c_void = std::ptr::without_provenance_mut(0x02);

pub const CU_STREAM_NON_BLOCKING: u32 = 0x1;
pub const CU_EVENT_DEFAULT: u32 = 0x0;
pub const CU_EVENT_DISABLE_TIMING: u32 = 0x2;
pub const CU_MEMHOSTALLOC_PORTABLE: u32 = 0x01;
pub const CU_MEMHOSTALLOC_DEVICEMAP: u32 = 0x02;
pub const CU_MEM_ATTACH_GLOBAL: u32 = 0x1;

/// `CUdevice_attribute` ids.
pub mod attribute {
    pub const MAX_THREADS_PER_BLOCK: i32 = 1;
    pub const MAX_SHARED_MEMORY_PER_BLOCK: i32 = 8;
    pub const WARP_SIZE: i32 = 10;
    pub const MULTIPROCESSOR_COUNT: i32 = 16;
    pub const MAX_THREADS_PER_MULTIPROCESSOR: i32 = 39;
    pub const COMPUTE_CAPABILITY_MAJOR: i32 = 75;
    pub const COMPUTE_CAPABILITY_MINOR: i32 = 76;
    pub const MANAGED_MEMORY: i32 = 83;
    pub const CONCURRENT_MANAGED_ACCESS: i32 = 89;
}

/// `CUfunction_attribute` ids.
pub mod func_attribute {
    pub const MAX_THREADS_PER_BLOCK: i32 = 0;
    pub const SHARED_SIZE_BYTES: i32 = 1;
    pub const LOCAL_SIZE_BYTES: i32 = 3;
    pub const NUM_REGS: i32 = 4;
}

/// `CUjit_option` ids.
pub mod jit_option {
    pub const INFO_LOG_BUFFER: i32 = 3;
    pub const INFO_LOG_BUFFER_SIZE_BYTES: i32 = 4;
    pub const ERROR_LOG_BUFFER: i32 = 5;
    pub const ERROR_LOG_BUFFER_SIZE_BYTES: i32 = 6;
}

/// Declares the bound entry points: the Rust field name, the exact export
/// resolved with `dlsym`, and the C prototype (every driver call returns
/// `CUresult`).
macro_rules! cuda_api {
    ($($field:ident = $symbol:literal: fn($($arg:ty),* $(,)?);)*) => {
        /// The loaded driver.
        pub struct Api {
            $(pub $field: unsafe extern "C" fn($($arg),*) -> CUresult,)*
            // Declared last so the function pointers never outlive the library.
            _library: Library,
        }

        impl Api {
            fn bind(library: Library) -> Result<Self> {
                Ok(Self { $($field: sym(&library, $symbol)?,)* _library: library })
            }
        }

        /// `(Rust name, dlsym symbol)` of every bound entry point.
        pub const SYMBOLS: &[(&str, &str)] = &[$((stringify!($field), $symbol)),*];
    };
}

cuda_api! {
    init = "cuInit": fn(u32);
    driver_get_version = "cuDriverGetVersion": fn(*mut c_int);
    get_error_name = "cuGetErrorName": fn(CUresult, *mut *const c_char);
    get_error_string = "cuGetErrorString": fn(CUresult, *mut *const c_char);
    device_get_count = "cuDeviceGetCount": fn(*mut c_int);
    device_get = "cuDeviceGet": fn(*mut CUdevice, c_int);
    device_get_name = "cuDeviceGetName": fn(*mut c_char, c_int, CUdevice);
    device_get_attribute = "cuDeviceGetAttribute": fn(*mut c_int, c_int, CUdevice);
    device_primary_ctx_retain = "cuDevicePrimaryCtxRetain": fn(*mut CUcontext, CUdevice);
    device_primary_ctx_release = "cuDevicePrimaryCtxRelease_v2": fn(CUdevice);
    ctx_set_current = "cuCtxSetCurrent": fn(CUcontext);
    ctx_synchronize = "cuCtxSynchronize": fn();
    mem_get_info = "cuMemGetInfo_v2": fn(*mut usize, *mut usize);
    mem_alloc = "cuMemAlloc_v2": fn(*mut CUdeviceptr, usize);
    mem_alloc_managed = "cuMemAllocManaged": fn(*mut CUdeviceptr, usize, u32);
    mem_free = "cuMemFree_v2": fn(CUdeviceptr);
    mem_host_alloc = "cuMemHostAlloc": fn(*mut *mut c_void, usize, u32);
    mem_host_get_device_pointer = "cuMemHostGetDevicePointer_v2": fn(*mut CUdeviceptr, *mut c_void, u32);
    mem_free_host = "cuMemFreeHost": fn(*mut c_void);
    memcpy_htod_async = "cuMemcpyHtoDAsync_v2": fn(CUdeviceptr, *const c_void, usize, CUstream);
    memcpy_dtoh_async = "cuMemcpyDtoHAsync_v2": fn(*mut c_void, CUdeviceptr, usize, CUstream);
    memcpy_dtod_async = "cuMemcpyDtoDAsync_v2": fn(CUdeviceptr, CUdeviceptr, usize, CUstream);
    memcpy_dtoh = "cuMemcpyDtoH_v2": fn(*mut c_void, CUdeviceptr, usize);
    memset_d8_async = "cuMemsetD8Async": fn(CUdeviceptr, u8, usize, CUstream);
    module_load_data_ex = "cuModuleLoadDataEx": fn(*mut CUmodule, *const c_void, u32, *mut c_int, *mut *mut c_void);
    module_unload = "cuModuleUnload": fn(CUmodule);
    module_get_function = "cuModuleGetFunction": fn(*mut CUfunction, CUmodule, *const c_char);
    func_get_attribute = "cuFuncGetAttribute": fn(*mut c_int, c_int, CUfunction);
    occupancy_max_active_blocks_per_multiprocessor =
        "cuOccupancyMaxActiveBlocksPerMultiprocessor": fn(*mut c_int, CUfunction, c_int, usize);
    launch_kernel = "cuLaunchKernel": fn(
        CUfunction, u32, u32, u32, u32, u32, u32, u32, CUstream, *mut *mut c_void, *mut *mut c_void
    );
    stream_create = "cuStreamCreate": fn(*mut CUstream, u32);
    stream_destroy = "cuStreamDestroy_v2": fn(CUstream);
    stream_synchronize = "cuStreamSynchronize": fn(CUstream);
    stream_wait_event = "cuStreamWaitEvent": fn(CUstream, CUevent, u32);
    event_create = "cuEventCreate": fn(*mut CUevent, u32);
    event_destroy = "cuEventDestroy_v2": fn(CUevent);
    event_record = "cuEventRecord": fn(CUevent, CUstream);
    event_synchronize = "cuEventSynchronize": fn(CUevent);
    event_query = "cuEventQuery": fn(CUevent);
    // `_v2` (CUDA 12.8) shares the prototype and would alone raise the
    // driver floor from the 12.0 graph entry points to R570.
    event_elapsed_time = "cuEventElapsedTime": fn(*mut f32, CUevent, CUevent);
    graph_create = "cuGraphCreate": fn(*mut CUgraph, u32);
    graph_destroy = "cuGraphDestroy": fn(CUgraph);
    graph_add_kernel_node =
        "cuGraphAddKernelNode_v2": fn(*mut CUgraphNode, CUgraph, *const CUgraphNode, usize, *const CudaKernelNodeParams);
    graph_add_event_record_node =
        "cuGraphAddEventRecordNode": fn(*mut CUgraphNode, CUgraph, *const CUgraphNode, usize, CUevent);
    graph_instantiate_with_flags = "cuGraphInstantiateWithFlags": fn(*mut CUgraphExec, CUgraph, u64);
    graph_exec_kernel_node_set_params =
        "cuGraphExecKernelNodeSetParams_v2": fn(CUgraphExec, CUgraphNode, *const CudaKernelNodeParams);
    graph_exec_event_record_node_set_event =
        "cuGraphExecEventRecordNodeSetEvent": fn(CUgraphExec, CUgraphNode, CUevent);
    graph_exec_destroy = "cuGraphExecDestroy": fn(CUgraphExec);
    graph_launch = "cuGraphLaunch": fn(CUgraphExec, CUstream);
}

// SAFETY: immutable after construction; the driver's entry points are
// documented thread-safe and the library lives for the rest of the process.
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

const LIBCUDA: &str = "libcuda.so.1";

fn unavailable(reason: String) -> Error {
    Error::DeviceUnavailable { reason }
}

fn sym<T: Copy>(library: &Library, name: &str) -> Result<T> {
    // SAFETY: `T` is declared from the symbol's C prototype at the call site.
    let symbol = unsafe { library.get::<T>(name.as_bytes()) }
        .map_err(|error| unavailable(format!("{LIBCUDA} has no symbol {name}: {error}")))?;
    Ok(*symbol)
}

impl Api {
    fn load() -> Result<Self> {
        // SAFETY: the driver's initializers are safe to run from any thread.
        let library =
            unsafe { Library::new(LIBCUDA) }.map_err(|error| unavailable(format!("cannot load {LIBCUDA}: {error}")))?;
        Self::bind(library)
    }

    /// `cuInit(0)`; idempotent, so every entry that may run first calls it.
    pub fn init(&self) -> Result<()> {
        // SAFETY: plain call with the only documented flag value.
        unsafe { (self.init)(0) }.check("cuInit")
    }

    pub fn device_count(&self) -> Result<usize> {
        let mut count: c_int = 0;
        // SAFETY: out-pointer to a live integer.
        unsafe { (self.device_get_count)(&mut count) }.check("cuDeviceGetCount")?;
        Ok(usize::try_from(count).unwrap_or(0))
    }

    /// `(major, minor)` of `cuDriverGetVersion`'s `1000 * major + 10 * minor`.
    pub fn driver_version(&self) -> Result<(u32, u32)> {
        let mut version: c_int = 0;
        // SAFETY: out-pointer to a live integer.
        unsafe { (self.driver_get_version)(&mut version) }.check("cuDriverGetVersion")?;
        let version = u32::try_from(version).unwrap_or(0);
        Ok((version / 1000, (version % 1000) / 10))
    }
}

static API: OnceLock<std::result::Result<Api, String>> = OnceLock::new();

/// The process-wide driver binding, or why CUDA is unavailable (no driver
/// library, or one missing an entry point this backend needs).
pub fn api() -> Result<&'static Api> {
    API.get_or_init(|| Api::load().map_err(|error| error.to_string()))
        .as_ref()
        .map_err(|reason| unavailable(reason.clone()))
}
