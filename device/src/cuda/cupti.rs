//! Hand-written CUPTI range-profiler bindings, resolved from `libcupti.so.13`
//! at runtime with `libloading`.
//!
//! CUPTI is optional in the sense `ptxas` is: absent or unusable, [`api`]
//! reports `None` and the backend reports no hardware counters — it never makes
//! CUDA itself unavailable. Disable it with `SVOD_CUDA_CUPTI=0`.
//!
//! CUDA 13 folded the PerfWorks host layer into CUPTI, so `libcupti.so.13`
//! exports the whole sequence and this is the only library we bind. CUPTI itself
//! `dlopen`s `libnvperf_host.so`, which therefore has to be resolvable by the
//! loader; when it is not, host initialization fails with `NOT_INITIALIZED`.
//!
//! Counter collection is admin-gated by default: with
//! `NVreg_RestrictProfilingToAdminUsers=1` (the driver default) everything up to
//! and including config-image construction succeeds and only the counter
//! availability image and `cuptiRangeProfilerStart` fail with
//! `INSUFFICIENT_PRIVILEGES`, so neither `Enable` nor `SetConfig` is a usable
//! capability probe — [`available`] uses the availability image.

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::ptr::{null, null_mut};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use libloading::Library;

use crate::profile::{CounterSet, CudaCounter, PmcCounter};

use super::sys::CUcontext;

/// `CUptiResult`: an integer newtype so codes from a newer CUPTI still
/// round-trip through [`CUptiResult::describe`].
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CUptiResult(pub c_int);

impl CUptiResult {
    pub const SUCCESS: Self = Self(0);
    /// Among other causes, a params `struct_size` this CUPTI has never seen.
    pub const INVALID_PARAMETER: Self = Self(1);
    /// CUPTI could not load `libnvperf_host.so`.
    pub const NOT_INITIALIZED: Self = Self(15);
    pub const INVALID_METRIC_NAME: Self = Self(17);
    pub const NOT_SUPPORTED: Self = Self(27);
    /// Profiling is restricted to admin users; see the module docs.
    pub const INSUFFICIENT_PRIVILEGES: Self = Self(35);

    fn ok(self) -> bool {
        self == Self::SUCCESS
    }

    /// `Ok(())`, or the CUPTI message for this code prefixed with `call`.
    fn check(self, call: &'static str) -> Result<(), String> {
        if self.ok() {
            return Ok(());
        }
        Err(format!("{call}: {}", self.describe()))
    }

    /// CUPTI's own text for this code, or the bare number when unavailable.
    pub fn describe(self) -> String {
        let mut text: *const c_char = null();
        let named = api()
            .filter(|api| {
                // SAFETY: out-pointer to a live pointer; CUPTI owns the string.
                unsafe { (api.get_result_string)(self, &mut text) }.ok() && !text.is_null()
            })
            // SAFETY: CUPTI returned a live NUL-terminated static string.
            .map(|_| unsafe { CStr::from_ptr(text) }.to_string_lossy().into_owned());
        named.unwrap_or_else(|| format!("CUptiResult({})", self.0))
    }
}

macro_rules! handles {
    ($($name:ident),* $(,)?) => {$(
        /// Opaque CUPTI handle.
        #[repr(transparent)]
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub struct $name(pub *mut c_void);

        impl $name {
            pub const NULL: Self = Self(null_mut());
        }
    )*};
}

handles!(RangeProfilerObject, HostObject);

// ── parameter structs ───────────────────────────────────────────────────────
//
// Every CUPTI entry point takes one `#[repr(C)]` params struct whose leading
// `struct_size` selects the ABI. That size is
// `offsetof(last_field) + sizeof(last_field)`, which is *not* `size_of::<T>()`
// wherever the struct ends in trailing padding — three of these do. The
// `STRUCT_SIZE` constants below are the values to send; the asserts pin both
// them and the Rust layout against the C headers.

/// `cuptiProfilerInitialize` / `cuptiProfilerDeInitialize`.
#[repr(C)]
pub struct InitializeParams {
    pub struct_size: usize,
    pub priv_: *mut c_void,
}
const INITIALIZE_SIZE: usize = 16;

/// `cuptiDeviceGetChipName`.
#[repr(C)]
pub struct GetChipNameParams {
    pub struct_size: usize,
    pub priv_: *mut c_void,
    pub device_index: usize,
    pub chip_name: *const c_char,
}
const GET_CHIP_NAME_SIZE: usize = 32;

/// `cuptiProfilerGetCounterAvailability`. Trailing padding: send 41, not 48.
#[repr(C)]
pub struct GetCounterAvailabilityParams {
    pub struct_size: usize,
    pub priv_: *mut c_void,
    pub ctx: CUcontext,
    pub image_size: usize,
    pub image: *mut u8,
    pub allow_device_level_counters: u8,
}
/// CUDA 13.3 added `allow_device_level_counters`; 13.0 ends at `image`.
const GET_COUNTER_AVAILABILITY_SIZES: [usize; 2] = [41, 40];

/// `cuptiProfilerHostInitialize`.
#[repr(C)]
pub struct HostInitializeParams {
    pub struct_size: usize,
    pub priv_: *mut c_void,
    /// `CUPTI_PROFILER_TYPE_RANGE_PROFILER`.
    pub profiler_type: u32,
    pub chip_name: *const c_char,
    pub counter_availability_image: *const u8,
    pub host_object: HostObject,
    /// PM sampling only; NULL for the range profiler.
    pub single_pass_metric_set_name: *const c_char,
}
/// CUDA 13.3 added `single_pass_metric_set_name`; 13.0 ends at `host_object`.
const HOST_INITIALIZE_SIZES: [usize; 2] = [56, 48];
const PROFILER_TYPE_RANGE_PROFILER: u32 = 0;

/// `cuptiProfilerHostDeinitialize`.
#[repr(C)]
pub struct HostDeinitializeParams {
    pub struct_size: usize,
    pub priv_: *mut c_void,
    pub host_object: HostObject,
}
const HOST_DEINITIALIZE_SIZE: usize = 24;

/// `cuptiProfilerHostConfigAddMetrics`.
#[repr(C)]
pub struct ConfigAddMetricsParams {
    pub struct_size: usize,
    pub priv_: *mut c_void,
    pub host_object: HostObject,
    pub metric_names: *const *const c_char,
    pub num_metrics: usize,
}
const CONFIG_ADD_METRICS_SIZE: usize = 40;

/// `cuptiProfilerHostGetConfigImageSize`.
#[repr(C)]
pub struct GetConfigImageSizeParams {
    pub struct_size: usize,
    pub priv_: *mut c_void,
    pub host_object: HostObject,
    pub config_image_size: usize,
}
const GET_CONFIG_IMAGE_SIZE_SIZE: usize = 32;

/// `cuptiProfilerHostGetConfigImage`.
#[repr(C)]
pub struct GetConfigImageParams {
    pub struct_size: usize,
    pub priv_: *mut c_void,
    pub host_object: HostObject,
    pub config_image_size: usize,
    pub config_image: *mut u8,
}
const GET_CONFIG_IMAGE_SIZE: usize = 40;

/// `cuptiProfilerHostEvaluateToGpuValues`.
#[repr(C)]
pub struct EvaluateToGpuValuesParams {
    pub struct_size: usize,
    pub priv_: *mut c_void,
    pub host_object: HostObject,
    pub counter_data_image: *const u8,
    pub counter_data_image_size: usize,
    pub range_index: usize,
    pub metric_names: *const *const c_char,
    pub num_metrics: usize,
    pub metric_values: *mut f64,
}
const EVALUATE_TO_GPU_VALUES_SIZE: usize = 72;

/// `cuptiRangeProfilerEnable`.
#[repr(C)]
pub struct RangeProfilerEnableParams {
    pub struct_size: usize,
    pub priv_: *mut c_void,
    pub ctx: CUcontext,
    pub object: RangeProfilerObject,
}
const RANGE_PROFILER_ENABLE_SIZE: usize = 32;

/// `cuptiRangeProfilerDisable` / `Start`.
#[repr(C)]
pub struct RangeProfilerObjectParams {
    pub struct_size: usize,
    pub priv_: *mut c_void,
    pub object: RangeProfilerObject,
}
const RANGE_PROFILER_OBJECT_SIZE: usize = 24;

/// `cuptiRangeProfilerGetCounterDataSize`.
#[repr(C)]
pub struct GetCounterDataSizeParams {
    pub struct_size: usize,
    pub priv_: *mut c_void,
    pub object: RangeProfilerObject,
    pub metric_names: *const *const c_char,
    pub num_metrics: usize,
    pub max_num_of_ranges: usize,
    pub max_num_range_tree_nodes: u32,
    pub counter_data_size: usize,
}
const GET_COUNTER_DATA_SIZE_SIZE: usize = 64;

/// `cuptiRangeProfilerCounterDataImageInitialize`.
#[repr(C)]
pub struct CounterDataImageInitializeParams {
    pub struct_size: usize,
    pub priv_: *mut c_void,
    pub object: RangeProfilerObject,
    pub counter_data_size: usize,
    pub counter_data: *mut u8,
}
const COUNTER_DATA_IMAGE_INITIALIZE_SIZE: usize = 40;

/// `cuptiRangeProfilerSetConfig`. Trailing padding: send 90, not 96. Note
/// `num_nesting_levels` precedes `min_nesting_level`, reversed from the legacy
/// `cuptiProfilerSetConfig` struct.
#[repr(C)]
pub struct SetConfigParams {
    pub struct_size: usize,
    pub priv_: *mut c_void,
    pub object: RangeProfilerObject,
    pub config_size: usize,
    pub config: *const u8,
    pub counter_data_image_size: usize,
    pub counter_data_image: *mut u8,
    /// `CUPTI_AutoRange`.
    pub range: u32,
    /// `CUPTI_KernelReplay`.
    pub replay_mode: u32,
    pub max_ranges_per_pass: usize,
    pub num_nesting_levels: u16,
    pub min_nesting_level: u16,
    pub pass_index: usize,
    pub target_nesting_level: u16,
}
const SET_CONFIG_SIZE: usize = 90;
const RANGE_AUTO: u32 = 1;
const REPLAY_KERNEL: u32 = 2;

/// `cuptiRangeProfilerStop`. Trailing padding: send 41, not 48.
#[repr(C)]
pub struct StopParams {
    pub struct_size: usize,
    pub priv_: *mut c_void,
    pub object: RangeProfilerObject,
    pub pass_index: usize,
    pub target_nesting_level: usize,
    pub all_passes_submitted: u8,
}
const STOP_SIZE: usize = 41;

/// `cuptiRangeProfilerDecodeData`.
#[repr(C)]
pub struct DecodeDataParams {
    pub struct_size: usize,
    pub priv_: *mut c_void,
    pub object: RangeProfilerObject,
    pub num_ranges_dropped: usize,
}
const DECODE_DATA_SIZE: usize = 32;

/// `cuptiRangeProfilerGetCounterDataInfo`.
#[repr(C)]
pub struct GetCounterDataInfoParams {
    pub struct_size: usize,
    pub priv_: *mut c_void,
    pub counter_data_image: *const u8,
    pub counter_data_image_size: usize,
    pub num_total_ranges: usize,
}
const GET_COUNTER_DATA_INFO_SIZE: usize = 40;

/// Rust layout must reproduce the C offsets exactly, and each `STRUCT_SIZE` is
/// the C `offsetof(last) + sizeof(last)` — equal to `size_of` except where the
/// struct ends in padding. CUPTI accepts exactly the sizes a struct has ever
/// had (never a larger one), so the two structs that grew since CUDA 13.0
/// carry both sizes and [`abi_ladder`] steps back on an older library; the
/// 13.0 layouts are prefixes of these, so the same Rust struct serves both.
const _: () = {
    assert!(size_of::<InitializeParams>() == INITIALIZE_SIZE);
    assert!(size_of::<GetChipNameParams>() == GET_CHIP_NAME_SIZE);
    assert!(size_of::<GetCounterAvailabilityParams>() == 48);
    assert!(GET_COUNTER_AVAILABILITY_SIZES[0] == 41 && GET_COUNTER_AVAILABILITY_SIZES[1] == 40);
    assert!(size_of::<HostInitializeParams>() == HOST_INITIALIZE_SIZES[0] && HOST_INITIALIZE_SIZES[1] == 48);
    assert!(size_of::<HostDeinitializeParams>() == HOST_DEINITIALIZE_SIZE);
    assert!(size_of::<ConfigAddMetricsParams>() == CONFIG_ADD_METRICS_SIZE);
    assert!(size_of::<GetConfigImageSizeParams>() == GET_CONFIG_IMAGE_SIZE_SIZE);
    assert!(size_of::<GetConfigImageParams>() == GET_CONFIG_IMAGE_SIZE);
    assert!(size_of::<EvaluateToGpuValuesParams>() == EVALUATE_TO_GPU_VALUES_SIZE);
    assert!(size_of::<RangeProfilerEnableParams>() == RANGE_PROFILER_ENABLE_SIZE);
    assert!(size_of::<RangeProfilerObjectParams>() == RANGE_PROFILER_OBJECT_SIZE);
    assert!(size_of::<GetCounterDataSizeParams>() == GET_COUNTER_DATA_SIZE_SIZE);
    assert!(size_of::<CounterDataImageInitializeParams>() == COUNTER_DATA_IMAGE_INITIALIZE_SIZE);
    assert!(size_of::<SetConfigParams>() == 96 && SET_CONFIG_SIZE == 90);
    assert!(size_of::<StopParams>() == 48 && STOP_SIZE == 41);
    assert!(size_of::<DecodeDataParams>() == DECODE_DATA_SIZE);
    assert!(size_of::<GetCounterDataInfoParams>() == GET_COUNTER_DATA_INFO_SIZE);
};

dl_api! {
    "The loaded CUPTI.", LIBCUPTI, CUptiResult, String, std::convert::identity;
    get_result_string = "cuptiGetResultString": fn(CUptiResult, *mut *const c_char);
    profiler_initialize = "cuptiProfilerInitialize": fn(*mut InitializeParams);
    device_get_chip_name = "cuptiDeviceGetChipName": fn(*mut GetChipNameParams);
    get_counter_availability = "cuptiProfilerGetCounterAvailability": fn(*mut GetCounterAvailabilityParams);
    host_initialize = "cuptiProfilerHostInitialize": fn(*mut HostInitializeParams);
    host_deinitialize = "cuptiProfilerHostDeinitialize": fn(*mut HostDeinitializeParams);
    host_config_add_metrics = "cuptiProfilerHostConfigAddMetrics": fn(*mut ConfigAddMetricsParams);
    host_get_config_image_size = "cuptiProfilerHostGetConfigImageSize": fn(*mut GetConfigImageSizeParams);
    host_get_config_image = "cuptiProfilerHostGetConfigImage": fn(*mut GetConfigImageParams);
    host_evaluate_to_gpu_values = "cuptiProfilerHostEvaluateToGpuValues": fn(*mut EvaluateToGpuValuesParams);
    range_profiler_enable = "cuptiRangeProfilerEnable": fn(*mut RangeProfilerEnableParams);
    range_profiler_disable = "cuptiRangeProfilerDisable": fn(*mut RangeProfilerObjectParams);
    range_profiler_set_config = "cuptiRangeProfilerSetConfig": fn(*mut SetConfigParams);
    range_profiler_start = "cuptiRangeProfilerStart": fn(*mut RangeProfilerObjectParams);
    range_profiler_stop = "cuptiRangeProfilerStop": fn(*mut StopParams);
    range_profiler_decode_data = "cuptiRangeProfilerDecodeData": fn(*mut DecodeDataParams);
    range_profiler_get_counter_data_size =
        "cuptiRangeProfilerGetCounterDataSize": fn(*mut GetCounterDataSizeParams);
    range_profiler_counter_data_image_initialize =
        "cuptiRangeProfilerCounterDataImageInitialize": fn(*mut CounterDataImageInitializeParams);
    range_profiler_get_counter_data_info =
        "cuptiRangeProfilerGetCounterDataInfo": fn(*mut GetCounterDataInfoParams);
}

// SAFETY: immutable after construction; CUPTI's entry points are documented
// thread-safe and the library lives for the rest of the process.
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

const LIBCUPTI: &str = "libcupti.so.13";

/// Toolkit locations searched after the loader's own path.
const FALLBACK_DIRS: &[&str] = &["/opt/cuda/lib64", "/usr/local/cuda/extras/CUPTI/lib64"];

/// `libcupti.so.13` on the loader path, then the toolkit fallbacks, then
/// `$CUDA_PATH`.
fn candidates() -> impl Iterator<Item = PathBuf> {
    let fallbacks = FALLBACK_DIRS.iter().map(|dir| Path::new(dir).join(LIBCUPTI));
    let cuda_path = std::env::var_os("CUDA_PATH").into_iter().flat_map(|root| {
        let root = PathBuf::from(root);
        [root.join("lib64").join(LIBCUPTI), root.join("extras/CUPTI/lib64").join(LIBCUPTI)]
    });
    std::iter::once(PathBuf::from(LIBCUPTI)).chain(fallbacks).chain(cuda_path)
}

impl Api {
    fn load() -> Result<Self, String> {
        let mut last = format!("cannot load {LIBCUPTI}");
        for candidate in candidates() {
            // SAFETY: CUPTI's initializers are safe to run from any thread.
            match unsafe { Library::new(&candidate) } {
                Ok(library) => return Self::bind(library),
                Err(error) => last = format!("{}: {}", candidate.display(), crate::error::describe(&error)),
            }
        }
        Err(last)
    }
}

static API: OnceLock<Option<Api>> = OnceLock::new();

/// The process-wide CUPTI binding, or `None` when CUPTI is absent, unusable, or
/// disabled with `SVOD_CUDA_CUPTI=0`. Never fails the backend: callers degrade
/// to timing-only profiling.
pub fn api() -> Option<&'static Api> {
    API.get_or_init(|| {
        if std::env::var("SVOD_CUDA_CUPTI").as_deref() == Ok("0") {
            return None;
        }
        match Api::load() {
            Ok(api) => Some(api),
            Err(error) => {
                tracing::debug!(%error, "CUPTI unavailable; CUDA reports no hardware counters");
                None
            }
        }
    })
    .as_ref()
}

/// Zeroed params with `struct_size` set, which is how CUPTI selects the ABI.
/// Padding must be zero too, so this zeroes the whole struct rather than
/// assigning field by field.
fn params<T>(struct_size: usize) -> T {
    // SAFETY: every params struct is a C aggregate of integers and raw pointers,
    // for which all-zero is a valid value; `struct_size` is written immediately.
    let mut p: T = unsafe { std::mem::zeroed() };
    // SAFETY: `struct_size` is the first field of every params struct.
    unsafe { (&raw mut p).cast::<usize>().write(struct_size) };
    p
}

/// Call `call` with the newest ABI size of a params struct that grew across
/// CUPTI releases, stepping back to the older size when the library rejects it
/// with `INVALID_PARAMETER` (a CUDA 13.0 CUPTI knows only its own sizes). The
/// accepted size is remembered in `chosen` so later calls try it alone; any
/// other outcome ends the ladder, since it is the call's real answer.
pub(crate) fn abi_ladder(
    sizes: &[usize],
    chosen: &AtomicUsize,
    name: &'static str,
    mut call: impl FnMut(usize) -> CUptiResult,
) -> Result<(), String> {
    let known = [chosen.load(Ordering::Relaxed)];
    let candidates = if known[0] != 0 { &known[..] } else { sizes };
    let mut result = CUptiResult::INVALID_PARAMETER;
    for &size in candidates {
        result = call(size);
        if result != CUptiResult::INVALID_PARAMETER {
            chosen.store(size, Ordering::Relaxed);
            break;
        }
    }
    result.check(name)
}

static GET_COUNTER_AVAILABILITY_ABI: AtomicUsize = AtomicUsize::new(0);
static HOST_INITIALIZE_ABI: AtomicUsize = AtomicUsize::new(0);

/// `cuptiProfilerInitialize`, once per process. Idempotent.
fn profiler_initialize(api: &Api) -> Result<(), String> {
    static INIT: OnceLock<Result<(), String>> = OnceLock::new();
    INIT.get_or_init(|| {
        let mut p: InitializeParams = params(INITIALIZE_SIZE);
        // SAFETY: live params struct with its ABI size set.
        unsafe { (api.profiler_initialize)(&mut p) }.check("cuptiProfilerInitialize")
    })
    .clone()
}

/// The counter availability image for `ctx`, or the CUPTI error. This is the
/// privilege probe: unprivileged it fails with `INSUFFICIENT_PRIVILEGES` while
/// `Enable` and `SetConfig` still succeed.
fn counter_availability(api: &Api, ctx: CUcontext) -> Result<Vec<u8>, String> {
    let mut p: GetCounterAvailabilityParams = params(0);
    abi_ladder(
        &GET_COUNTER_AVAILABILITY_SIZES,
        &GET_COUNTER_AVAILABILITY_ABI,
        "cuptiProfilerGetCounterAvailability",
        |size| {
            p = params(size);
            p.ctx = ctx;
            // SAFETY: image NULL, so CUPTI only reports the size it needs.
            unsafe { (api.get_counter_availability)(&mut p) }
        },
    )?;
    let mut image = vec![0u8; p.image_size];
    p.image = image.as_mut_ptr();
    // SAFETY: `image` is live and `image_size` bytes long.
    unsafe { (api.get_counter_availability)(&mut p) }.check("cuptiProfilerGetCounterAvailability")?;
    Ok(image)
}

/// Whether hardware counters can actually be collected on `ctx`: CUPTI is
/// loadable and profiling is not restricted to admin users.
pub fn available(ctx: CUcontext) -> bool {
    let Some(api) = api() else { return false };
    let probe = profiler_initialize(api).and_then(|()| counter_availability(api, ctx));
    match probe {
        Ok(_) => true,
        Err(error) => {
            tracing::debug!(%error, "CUDA hardware counters unavailable");
            false
        }
    }
}

/// A range-profiling session bound to one CUDA context.
///
/// Enabling is per-context and reused across dispatches; each capture is one
/// `SetConfig`/`Start`/`Stop`/`Decode` cycle over a single launch, in
/// `CUPTI_AutoRange` with `CUPTI_KernelReplay` so CUPTI opens the range per
/// kernel and replays internally for multi-pass configs. The five counters we
/// collect schedule in one pass on every supported chip.
pub struct Session {
    object: RangeProfilerObject,
    host: HostObject,
    counters: Vec<CudaCounter>,
    /// Backing storage for `metric_ptrs`; CUPTI borrows these names on every
    /// call, so they must outlive the pointer array.
    _metrics: Vec<CString>,
    metric_ptrs: Vec<*const c_char>,
    /// The counter-availability image the host object was initialized with.
    /// CUPTI documents it as an input of that call only, but keeps no copy it
    /// promises to own, so the session holds it for the host object's life.
    _availability: Option<Vec<u8>>,
    config: Vec<u8>,
    counter_data: Vec<u8>,
}

// SAFETY: the CUPTI objects are owned by this session and only used behind
// `&mut self`; the raw pointers point into `metrics`, which outlives them.
unsafe impl Send for Session {}

impl Drop for Session {
    fn drop(&mut self) {
        let Some(api) = api() else { return };
        if self.object != RangeProfilerObject::NULL {
            let mut p: RangeProfilerObjectParams = params(RANGE_PROFILER_OBJECT_SIZE);
            p.object = self.object;
            // SAFETY: live params carrying an object this session owns.
            let _ = unsafe { (api.range_profiler_disable)(&mut p) };
        }
        if self.host != HostObject::NULL {
            let mut p: HostDeinitializeParams = params(HOST_DEINITIALIZE_SIZE);
            p.host_object = self.host;
            // SAFETY: live params carrying a host object this session owns.
            let _ = unsafe { (api.host_deinitialize)(&mut p) };
        }
    }
}

impl Session {
    /// Open a session collecting `counters` on `ctx`, where `device_index` is
    /// the CUDA ordinal the context belongs to.
    pub fn new(ctx: CUcontext, device_index: usize, counters: &[CudaCounter]) -> Result<Self, String> {
        let api = api().ok_or("CUPTI is not loaded")?;
        profiler_initialize(api)?;

        let metrics: Vec<CString> =
            counters.iter().map(|c| CString::new(c.metric()).expect("metric names carry no NUL")).collect();
        let metric_ptrs: Vec<*const c_char> = metrics.iter().map(|m| m.as_ptr()).collect();

        let mut chip: GetChipNameParams = params(GET_CHIP_NAME_SIZE);
        chip.device_index = device_index;
        // SAFETY: live params; CUPTI writes a pointer to a static chip name.
        unsafe { (api.device_get_chip_name)(&mut chip) }.check("cuptiDeviceGetChipName")?;

        // The image is only needed for chips newer than this CUPTI. Passing NULL
        // is supported for known chips and keeps the host path working when the
        // privileged fetch is refused.
        let availability = counter_availability(api, ctx).ok();

        let mut host: HostInitializeParams = params(0);
        abi_ladder(&HOST_INITIALIZE_SIZES, &HOST_INITIALIZE_ABI, "cuptiProfilerHostInitialize", |size| {
            host = params(size);
            host.profiler_type = PROFILER_TYPE_RANGE_PROFILER;
            host.chip_name = chip.chip_name;
            host.counter_availability_image = availability.as_ref().map_or(null(), |i| i.as_ptr());
            // SAFETY: live params; the chip name is static and the image is kept
            // for the host object's life.
            unsafe { (api.host_initialize)(&mut host) }
        })?;

        // From here a failure must still release the host object, so the session
        // is built first and every fallible step runs against it.
        let mut session = Self {
            object: RangeProfilerObject::NULL,
            host: host.host_object,
            counters: counters.to_vec(),
            _metrics: metrics,
            metric_ptrs,
            _availability: availability,
            config: Vec::new(),
            counter_data: Vec::new(),
        };
        session.build_config(api)?;
        session.enable(api, ctx)?;
        Ok(session)
    }

    /// Validate and schedule the metrics, producing the config image.
    fn build_config(&mut self, api: &Api) -> Result<(), String> {
        let mut add: ConfigAddMetricsParams = params(CONFIG_ADD_METRICS_SIZE);
        add.host_object = self.host;
        add.metric_names = self.metric_ptrs.as_ptr();
        add.num_metrics = self.metric_ptrs.len();
        // SAFETY: live params; the name array outlives the call.
        unsafe { (api.host_config_add_metrics)(&mut add) }.check("cuptiProfilerHostConfigAddMetrics")?;

        let mut size: GetConfigImageSizeParams = params(GET_CONFIG_IMAGE_SIZE_SIZE);
        size.host_object = self.host;
        // SAFETY: live params carrying this session's host object.
        unsafe { (api.host_get_config_image_size)(&mut size) }.check("cuptiProfilerHostGetConfigImageSize")?;

        // CUPTI documents config images as 8-byte aligned; a `Vec<u8>` is only
        // byte-aligned, so allocate as `u64` and reinterpret.
        let mut aligned: Vec<u64> = vec![0; size.config_image_size.div_ceil(8)];
        let mut image: GetConfigImageParams = params(GET_CONFIG_IMAGE_SIZE);
        image.host_object = self.host;
        image.config_image_size = size.config_image_size;
        image.config_image = aligned.as_mut_ptr().cast();
        // SAFETY: `aligned` holds at least `config_image_size` bytes.
        unsafe { (api.host_get_config_image)(&mut image) }.check("cuptiProfilerHostGetConfigImage")?;

        // SAFETY: `aligned` is a live allocation of `len * 8` initialized bytes.
        self.config =
            unsafe { std::slice::from_raw_parts(aligned.as_ptr().cast::<u8>(), size.config_image_size) }.to_vec();
        Ok(())
    }

    /// Enable range profiling on the context and size the counter-data image.
    fn enable(&mut self, api: &Api, ctx: CUcontext) -> Result<(), String> {
        let mut en: RangeProfilerEnableParams = params(RANGE_PROFILER_ENABLE_SIZE);
        en.ctx = ctx;
        // SAFETY: live params; CUPTI writes the object out.
        unsafe { (api.range_profiler_enable)(&mut en) }.check("cuptiRangeProfilerEnable")?;
        self.object = en.object;

        let mut size: GetCounterDataSizeParams = params(GET_COUNTER_DATA_SIZE_SIZE);
        size.object = self.object;
        size.metric_names = self.metric_ptrs.as_ptr();
        size.num_metrics = self.metric_ptrs.len();
        size.max_num_of_ranges = 1;
        size.max_num_range_tree_nodes = 1;
        // SAFETY: live params; the name array outlives the call.
        unsafe { (api.range_profiler_get_counter_data_size)(&mut size) }
            .check("cuptiRangeProfilerGetCounterDataSize")?;
        self.counter_data = vec![0; size.counter_data_size];
        Ok(())
    }

    /// Arm the hardware for the next launch on this context.
    ///
    /// The counter-data image must be re-initialized per capture, and the
    /// caller must run exactly one launch and synchronize before [`stop`](Self::stop).
    pub fn start(&mut self) -> Result<(), String> {
        let api = api().ok_or("CUPTI is not loaded")?;
        self.counter_data.fill(0);

        let mut init: CounterDataImageInitializeParams = params(COUNTER_DATA_IMAGE_INITIALIZE_SIZE);
        init.object = self.object;
        init.counter_data_size = self.counter_data.len();
        init.counter_data = self.counter_data.as_mut_ptr();
        // SAFETY: live params over this session's zeroed counter-data image.
        unsafe { (api.range_profiler_counter_data_image_initialize)(&mut init) }
            .check("cuptiRangeProfilerCounterDataImageInitialize")?;

        let mut cfg: SetConfigParams = params(SET_CONFIG_SIZE);
        cfg.object = self.object;
        cfg.config_size = self.config.len();
        cfg.config = self.config.as_ptr();
        cfg.counter_data_image_size = self.counter_data.len();
        cfg.counter_data_image = self.counter_data.as_mut_ptr();
        cfg.range = RANGE_AUTO;
        cfg.replay_mode = REPLAY_KERNEL;
        cfg.max_ranges_per_pass = 1;
        cfg.num_nesting_levels = 1;
        cfg.min_nesting_level = 1;
        cfg.target_nesting_level = 1;
        // SAFETY: the config and counter-data images stay live and at a stable
        // address for as long as the session holds them.
        unsafe { (api.range_profiler_set_config)(&mut cfg) }.check("cuptiRangeProfilerSetConfig")?;

        let mut start: RangeProfilerObjectParams = params(RANGE_PROFILER_OBJECT_SIZE);
        start.object = self.object;
        // SAFETY: live params carrying this session's object.
        unsafe { (api.range_profiler_start)(&mut start) }.check("cuptiRangeProfilerStart")
    }

    /// Disarm and decode. The launch must already have completed.
    ///
    /// Kernel replay covers multi-pass configs inside CUPTI, so a config that
    /// still reports unsubmitted passes cannot be served by a single launch and
    /// is reported rather than silently returning partial counters.
    pub fn stop(&mut self) -> Result<CounterSet, String> {
        let api = api().ok_or("CUPTI is not loaded")?;

        let mut stop: StopParams = params(STOP_SIZE);
        stop.object = self.object;
        // SAFETY: live params carrying this session's object.
        unsafe { (api.range_profiler_stop)(&mut stop) }.check("cuptiRangeProfilerStop")?;
        if stop.all_passes_submitted == 0 {
            return Err("cuptiRangeProfilerStop: metric set needs more than one pass".into());
        }

        let mut decode: DecodeDataParams = params(DECODE_DATA_SIZE);
        decode.object = self.object;
        // SAFETY: live params; CUPTI writes into the counter-data image.
        unsafe { (api.range_profiler_decode_data)(&mut decode) }.check("cuptiRangeProfilerDecodeData")?;
        if decode.num_ranges_dropped != 0 {
            return Err(format!("cuptiRangeProfilerDecodeData dropped {} range(s)", decode.num_ranges_dropped));
        }

        let mut info: GetCounterDataInfoParams = params(GET_COUNTER_DATA_INFO_SIZE);
        info.counter_data_image = self.counter_data.as_ptr();
        info.counter_data_image_size = self.counter_data.len();
        // SAFETY: live params over this session's counter-data image.
        unsafe { (api.range_profiler_get_counter_data_info)(&mut info) }
            .check("cuptiRangeProfilerGetCounterDataInfo")?;
        if info.num_total_ranges == 0 {
            return Ok(CounterSet::default());
        }

        let mut values = vec![0.0f64; self.counters.len()];
        let mut eval: EvaluateToGpuValuesParams = params(EVALUATE_TO_GPU_VALUES_SIZE);
        eval.host_object = self.host;
        eval.counter_data_image = self.counter_data.as_ptr();
        eval.counter_data_image_size = self.counter_data.len();
        eval.metric_names = self.metric_ptrs.as_ptr();
        eval.num_metrics = self.metric_ptrs.len();
        eval.metric_values = values.as_mut_ptr();
        // SAFETY: `values` holds one f64 per requested metric.
        unsafe { (api.host_evaluate_to_gpu_values)(&mut eval) }.check("cuptiProfilerHostEvaluateToGpuValues")?;

        // Values come back positionally, matching the order metrics were passed.
        // Counters are non-negative integers reported as doubles.
        let values = self
            .counters
            .iter()
            .zip(&values)
            .map(|(&c, &v)| (PmcCounter::Cuda(c), if v.is_finite() && v > 0.0 { v as u64 } else { 0 }))
            .collect();
        Ok(CounterSet { values })
    }
}
