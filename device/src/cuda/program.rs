//! A kernel module — a `ptxas` cubin loaded as is, or PTX text JIT-compiled
//! by the driver — dispatched with `cuLaunchKernel`; kernel arguments travel
//! as one packed blob in the `extra` array, laid out by [`ClikeKernargLayout`]
//! (8-byte device pointers, 4-byte `i32` scalars, ascending PARAM slot order
//! — PTX's natural `.param` layout).

use std::ffi::{CString, c_char, c_int, c_void};
use std::ptr::null_mut;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use object::elf::{ELFDATA2LSB, EM_CUDA, FileHeader64};
use object::read::elf::{ElfFile64, FileHeader};
use object::{LittleEndian, Object, ObjectSymbol, SymbolKind};

use super::device::{CudaDevice, CudaEvent};
use super::sys::{
    CU_LAUNCH_PARAM_BUFFER_POINTER, CU_LAUNCH_PARAM_BUFFER_SIZE, CU_LAUNCH_PARAM_END, CUfunction, CUmodule, CUresult,
    CUstream, func_attribute, jit_option,
};
use crate::device::{AbiParamDescriptor, CompiledSpec, Program};
use crate::hcq::ClikeKernargLayout;
use crate::profile::KernelResources;
use crate::{Error, Result};

const JIT_LOG_BYTES: usize = 16 << 10;

/// A loaded module, shared by the program and any graph that captured it so
/// the kernel outlives the program that loaded it.
pub(crate) struct CudaModule {
    dev: Arc<CudaDevice>,
    raw: CUmodule,
}

impl Drop for CudaModule {
    fn drop(&mut self) {
        if self.dev.enter().is_ok() {
            // SAFETY: the module this value loaded.
            unsafe { (self.dev.api().module_unload)(self.raw) };
        }
    }
}

/// Grid (blocks) and block (threads) of one launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Launch {
    pub grid: [u32; 3],
    pub block: [u32; 3],
}

pub struct CudaProgram {
    dev: Arc<CudaDevice>,
    module: Arc<CudaModule>,
    function: CUfunction,
    name: String,
    layout: ClikeKernargLayout,
    max_threads_per_block: u32,
    num_regs: u32,
    shared_bytes: u32,
    local_bytes: u32,
    /// Block size of the latest launch, the occupancy query's input.
    last_block: AtomicU32,
}

impl std::fmt::Debug for CudaProgram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaProgram")
            .field("name", &self.name)
            .field("buf_count", &self.layout.globals)
            .field("var_count", &self.layout.vars)
            .field("num_regs", &self.num_regs)
            .field("shared_bytes", &self.shared_bytes)
            .finish_non_exhaustive()
    }
}

impl CudaProgram {
    /// Load `spec.bytes` — a cubin (ELF) or PTX text — and bind entry point
    /// `spec.name`.
    pub fn load(dev: Arc<CudaDevice>, spec: &CompiledSpec) -> Result<Self> {
        if is_cubin(&spec.bytes) {
            Self::load_cubin(dev, &spec.bytes, &spec.name, &spec.abi)
        } else {
            Self::load_ptx(dev, &spec.bytes, &spec.name, &spec.abi)
        }
    }

    /// JIT PTX text; the entry's `.param` list is checked against `abi` first.
    pub fn load_ptx(dev: Arc<CudaDevice>, ptx: &[u8], name: &str, abi: &[AbiParamDescriptor]) -> Result<Self> {
        let layout = validate_abi(abi)?;
        let jit_error = |cause: &str| Error::CudaJit { kernel: name.into(), cause: cause.into(), log: String::new() };
        if ptx.is_empty() {
            return Err(jit_error("empty PTX image"));
        }
        // The driver parses PTX as a NUL-terminated string.
        let image = CString::new(ptx).map_err(|_| jit_error("PTX text contains an interior NUL byte"))?;
        if let Ok(text) = std::str::from_utf8(ptx) {
            check_ptx_entry_abi(text, name, abi)?;
        }
        Self::load_image(dev, image.as_bytes_with_nul(), name, layout)
    }

    /// Load a cubin produced by `ptxas` for this device's architecture; the
    /// image is validated by [`validate_cubin`] before it reaches the driver.
    pub fn load_cubin(dev: Arc<CudaDevice>, cubin: &[u8], name: &str, abi: &[AbiParamDescriptor]) -> Result<Self> {
        let layout = validate_abi(abi)?;
        validate_cubin(cubin, name)?;
        Self::load_image(dev, cubin, name, layout)
    }

    fn load_image(dev: Arc<CudaDevice>, image: &[u8], name: &str, layout: ClikeKernargLayout) -> Result<Self> {
        let entry = CString::new(name).map_err(|_| Error::CudaJit {
            kernel: name.into(),
            cause: "kernel name contains a NUL byte".into(),
            log: String::new(),
        })?;
        let api = dev.enter()?;
        let mut error_log = vec![0 as c_char; JIT_LOG_BYTES];
        let mut info_log = vec![0 as c_char; JIT_LOG_BYTES];
        let mut options = [
            jit_option::ERROR_LOG_BUFFER,
            jit_option::ERROR_LOG_BUFFER_SIZE_BYTES,
            jit_option::INFO_LOG_BUFFER,
            jit_option::INFO_LOG_BUFFER_SIZE_BYTES,
        ];
        // Size options are passed by value in the pointer slot, per `cuda.h`.
        let mut values: [*mut c_void; 4] = [
            error_log.as_mut_ptr().cast(),
            JIT_LOG_BYTES as *mut c_void,
            info_log.as_mut_ptr().cast(),
            JIT_LOG_BYTES as *mut c_void,
        ];
        let mut raw = CUmodule::NULL;
        // SAFETY: the option arrays are the same length; the log buffers
        // outlive the call; the image is a complete cubin or NUL-terminated
        // PTX, which is how the driver tells them apart.
        let result = unsafe {
            (api.module_load_data_ex)(
                &mut raw,
                image.as_ptr().cast(),
                options.len() as u32,
                options.as_mut_ptr(),
                values.as_mut_ptr(),
            )
        };
        let log_text = |log: &[c_char]| -> String {
            let bytes: Vec<u8> = log.iter().take_while(|byte| **byte != 0).map(|byte| *byte as u8).collect();
            String::from_utf8_lossy(&bytes).trim().to_string()
        };
        if result != CUresult::SUCCESS {
            let (code, message) = result.describe();
            return Err(Error::CudaJit {
                kernel: name.into(),
                cause: format!("{code}: {message}"),
                log: log_text(&error_log),
            });
        }
        let module = Arc::new(CudaModule { dev: Arc::clone(&dev), raw });
        let info = log_text(&info_log);
        if !info.is_empty() {
            tracing::debug!(kernel = name, info, "module load log");
        }

        let mut function = CUfunction::NULL;
        // SAFETY: a live module and a NUL-terminated entry name.
        unsafe { (api.module_get_function)(&mut function, raw, entry.as_ptr()) }.check("cuModuleGetFunction").map_err(
            |error| Error::CudaJit {
                kernel: name.into(),
                cause: format!("no entry point: {error}"),
                log: String::new(),
            },
        )?;
        let attribute = |id: i32| -> Result<u32> {
            let mut value: c_int = 0;
            // SAFETY: out-pointer to a live integer.
            unsafe { (api.func_get_attribute)(&mut value, id, function) }.check("cuFuncGetAttribute")?;
            Ok(u32::try_from(value).unwrap_or(0))
        };
        Ok(Self {
            max_threads_per_block: attribute(func_attribute::MAX_THREADS_PER_BLOCK)?,
            num_regs: attribute(func_attribute::NUM_REGS)?,
            shared_bytes: attribute(func_attribute::SHARED_SIZE_BYTES)?,
            local_bytes: attribute(func_attribute::LOCAL_SIZE_BYTES)?,
            dev,
            module,
            function,
            name: name.to_string(),
            layout,
            last_block: AtomicU32::new(0),
        })
    }

    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.dev
    }

    pub(crate) fn module(&self) -> &Arc<CudaModule> {
        &self.module
    }

    pub(crate) fn function(&self) -> CUfunction {
        self.function
    }

    pub(crate) fn layout(&self) -> &ClikeKernargLayout {
        &self.layout
    }

    /// `global_size` is the grid in blocks and `local_size` the block in
    /// threads (the work-group convention AMD and Metal use); `None` is one.
    pub fn launch_dims(&self, global_size: Option<[usize; 3]>, local_size: Option<[usize; 3]>) -> Result<Launch> {
        let dims = |what: &str, size: [usize; 3]| -> Result<[u32; 3]> {
            let mut out = [1u32; 3];
            for (dst, dim) in out.iter_mut().zip(size) {
                *dst = u32::try_from(dim).ok().filter(|dim| *dim > 0).ok_or_else(|| Error::Runtime {
                    message: format!("CUDA kernel '{}' {what} {size:?} is not a positive 32-bit size", self.name),
                })?;
            }
            Ok(out)
        };
        let grid = dims("grid", global_size.unwrap_or([1, 1, 1]))?;
        let block = dims("block", local_size.unwrap_or([1, 1, 1]))?;
        let threads = block.iter().try_fold(1u32, |acc, dim| acc.checked_mul(*dim)).unwrap_or(u32::MAX);
        if threads > self.max_threads_per_block {
            return Err(Error::Runtime {
                message: format!(
                    "CUDA kernel '{}' block {block:?} ({threads} threads) exceeds its maxThreadsPerBlock {} \
                     (numRegs {}, sharedSizeBytes {}, localSizeBytes {})",
                    self.name, self.max_threads_per_block, self.num_regs, self.shared_bytes, self.local_bytes
                ),
            });
        }
        Ok(Launch { grid, block })
    }

    /// Pack `buffers`/`vals` into a fresh kernarg blob.
    pub(crate) fn pack(&self, buffers: &[u64], vals: &[i64]) -> Result<Vec<u8>> {
        let mut blob = vec![0u8; self.layout.packed_size()];
        self.layout.pack(&mut blob, buffers, vals).map_err(|error| match error {
            Error::ProgramAbiMismatch { reason } => {
                Error::ProgramAbiMismatch { reason: format!("kernel {}: {reason}", self.name) }
            }
            other => other,
        })?;
        Ok(blob)
    }

    /// Launch on `stream`; asynchronous.
    ///
    /// # Safety
    ///
    /// Same contract as [`Program::execute`].
    pub(crate) unsafe fn launch(
        &self,
        stream: CUstream,
        buffers: &[*mut u8],
        vals: &[i64],
        global_size: Option<[usize; 3]>,
        local_size: Option<[usize; 3]>,
    ) -> Result<()> {
        let addresses: smallvec::SmallVec<[u64; 8]> = buffers.iter().map(|pointer| *pointer as u64).collect();
        let mut blob = self.pack(&addresses, vals)?;
        let launch = self.launch_dims(global_size, local_size)?;
        let api = self.dev.enter()?;
        let mut size = blob.len();
        let mut extra = extra_array(&mut blob, &mut size);
        let [gx, gy, gz] = launch.grid;
        let [bx, by, bz] = launch.block;
        // SAFETY: `extra` follows the sentinel protocol and points into live
        // locals; the caller guarantees the buffer addresses.
        let result = unsafe {
            (api.launch_kernel)(self.function, gx, gy, gz, bx, by, bz, 0, stream, null_mut(), extra.as_mut_ptr())
        };
        self.dev.check(result, "cuLaunchKernel")?;
        self.last_block.store(bx * by * bz, Ordering::Relaxed);
        Ok(())
    }

    /// Resident blocks per SM for `block_threads`-sized blocks
    /// (`cuOccupancyMaxActiveBlocksPerMultiprocessor`).
    pub fn max_active_blocks_per_sm(&self, block_threads: u32) -> Result<u32> {
        let api = self.dev.enter()?;
        let mut blocks: c_int = 0;
        // SAFETY: out-pointer to a live integer; the function is live.
        unsafe {
            (api.occupancy_max_active_blocks_per_multiprocessor)(&mut blocks, self.function, block_threads as c_int, 0)
        }
        .check("cuOccupancyMaxActiveBlocksPerMultiprocessor")?;
        Ok(u32::try_from(blocks).unwrap_or(0))
    }
}

/// The kernarg layout of `abi`, once the descriptors pass the shared checks.
fn validate_abi(abi: &[AbiParamDescriptor]) -> Result<ClikeKernargLayout> {
    let layout = ClikeKernargLayout::from_abi(abi);
    let var_names: Vec<String> =
        abi.iter().filter(|param| !param.is_storage()).map(|param| param.name.clone().unwrap_or_default()).collect();
    crate::device::validate_abi_descriptors(abi, layout.globals, &var_names)?;
    Ok(layout)
}

/// Does `image` start like an ELF (a cubin) rather than PTX text?
pub fn is_cubin(image: &[u8]) -> bool {
    image.starts_with(&object::elf::ELFMAG)
}

/// Check a cubin before it reaches the driver, on cached and fresh bytes
/// alike: a little-endian ELF64 for `EM_CUDA` that defines `entry` as code.
/// The compute capability is not readable here (the `e_flags` encoding
/// changes between toolkits); the compiler keys its cache on it and the
/// driver rejects a mismatch at load.
pub fn validate_cubin(cubin: &[u8], entry: &str) -> Result<()> {
    let reject = |cause: String| Err(Error::CudaJit { kernel: entry.into(), cause, log: String::new() });
    let header = match FileHeader64::<LittleEndian>::parse(cubin) {
        Ok(header) => header,
        Err(error) => return reject(format!("cubin is not an ELF64 image: {error}")),
    };
    if header.e_ident().data != ELFDATA2LSB {
        return reject("cubin is not little-endian".into());
    }
    let machine = header.e_machine(LittleEndian);
    if machine != EM_CUDA {
        return reject(format!("cubin e_machine is {machine}, not EM_CUDA ({EM_CUDA})"));
    }
    let file = match ElfFile64::<LittleEndian>::parse(cubin) {
        Ok(file) => file,
        Err(error) => return reject(format!("cubin sections are unreadable: {error}")),
    };
    let defined = file
        .symbols()
        .any(|symbol| symbol.name() == Ok(entry) && symbol.is_definition() && symbol.kind() == SymbolKind::Text);
    if !defined {
        return reject(format!("cubin has no entry {entry:?}"));
    }
    Ok(())
}

/// [`check_entry_params`] on the entry of `ptx`, when the text declares one:
/// the guard the PTX loader applies, exposed so a compiler that assembles
/// PTX to a cubin (whose parameter list is not readable) can apply it first.
pub fn check_ptx_entry_abi(ptx: &str, entry: &str, abi: &[AbiParamDescriptor]) -> Result<()> {
    match ptx_entry_params(ptx, entry) {
        Some(params) => check_entry_params(entry, &params, abi),
        None => Ok(()),
    }
}

/// The kernarg blob is laid out from `abi` (8-byte buffers, 4-byte scalars),
/// so the entry must declare exactly those parameters at those widths; the
/// driver checks neither and a short or misaligned blob is read past its end
/// on the GPU.
fn check_entry_params(name: &str, params: &[(String, Option<usize>)], abi: &[AbiParamDescriptor]) -> Result<()> {
    let mismatch = |reason: String| Err(Error::ProgramAbiMismatch { reason: format!("kernel {name}: {reason}") });
    if params.len() != abi.len() {
        return mismatch(format!("PTX entry declares {} parameters, the ABI describes {}", params.len(), abi.len()));
    }
    for (index, ((param, width), descriptor)) in params.iter().zip(abi).enumerate() {
        let expected = if descriptor.is_storage() { 8 } else { 4 };
        if width.is_some_and(|width| width != expected) {
            return mismatch(format!(
                "PTX parameter {index} ({param}) is {} bytes wide, the ABI slot {} needs {expected}",
                width.unwrap_or_default(),
                descriptor.slot
            ));
        }
    }
    Ok(())
}

/// `(name, byte width)` of every `.param` of PTX entry `entry` (`None` width
/// for aggregates such as `.b8 x[16]`), or `None` when the module declares
/// no such entry. Accepts `.visible`/`.weak` prefixes and multi-line lists.
pub(crate) fn ptx_entry_params(ptx: &str, entry: &str) -> Option<Vec<(String, Option<usize>)>> {
    let list = ptx.match_indices(".entry ").find_map(|(at, directive)| {
        let after = ptx[at + directive.len()..].trim_start().strip_prefix(entry)?.trim_start().strip_prefix('(')?;
        after.split_once(')').map(|(list, _)| list)
    })?;
    let width = |ty: &str| -> Option<usize> {
        let bits = ty.strip_prefix('.')?.trim_start_matches(['u', 's', 'b', 'f']);
        (ty.len() == bits.len() + 2).then(|| bits.parse::<usize>().ok().map(|bits| bits / 8)).flatten()
    };
    let params = list
        .split(',')
        .map(str::split_whitespace)
        .filter_map(|mut tokens| {
            tokens.next().filter(|token| *token == ".param")?;
            let tokens: Vec<&str> = tokens.collect();
            let name = tokens.last()?;
            let (name, aggregate) = name.split_once('[').map_or((*name, false), |(name, _)| (name, true));
            let scalar = tokens.iter().find_map(|token| width(token)).filter(|_| !aggregate);
            Some((name.to_string(), scalar))
        })
        .collect();
    Some(params)
}

/// The `cuLaunchKernel` / kernel-node `extra` array for a packed blob.
pub(crate) fn extra_array(blob: &mut [u8], size: &mut usize) -> [*mut c_void; 5] {
    [
        CU_LAUNCH_PARAM_BUFFER_POINTER,
        blob.as_mut_ptr().cast(),
        CU_LAUNCH_PARAM_BUFFER_SIZE,
        (size as *mut usize).cast(),
        CU_LAUNCH_PARAM_END,
    ]
}

impl Program for CudaProgram {
    unsafe fn execute(
        &self,
        buffers: &[*mut u8],
        vals: &[i64],
        global_size: Option<[usize; 3]>,
        local_size: Option<[usize; 3]>,
        wait: bool,
    ) -> Result<()> {
        let lane = self.dev.dispatch_lane();
        self.dev.order_launch(lane)?;
        // Fire-and-forget dispatches have no owner to publish a token: the
        // lane stays flagged and every scoped host wait drains it.
        lane.mark_unpublished();
        // SAFETY: forwarded contract.
        unsafe { self.launch(lane.raw(), buffers, vals, global_size, local_size) }?;
        if wait { self.dev.synchronize_lane(lane) } else { Ok(()) }
    }

    unsafe fn execute_timed(
        &self,
        buffers: &[*mut u8],
        vals: &[i64],
        global_size: Option<[usize; 3]>,
        local_size: Option<[usize; 3]>,
    ) -> Result<Option<std::time::Duration>> {
        let lane = self.dev.dispatch_lane();
        self.dev.order_launch(lane)?;
        let stream = lane.raw();
        let start = CudaEvent::new(Arc::clone(&self.dev), true)?;
        let end = CudaEvent::new(Arc::clone(&self.dev), true)?;
        start.record(stream)?;
        // SAFETY: forwarded contract.
        unsafe { self.launch(stream, buffers, vals, global_size, local_size) }?;
        end.record(stream)?;
        end.wait(0)?;
        let ms = end.elapsed_ms_since(start.raw())?;
        Ok(Some(std::time::Duration::from_secs_f64(f64::from(ms.max(0.0)) * 1e-3)))
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn new_exec_context(&self) -> Result<Option<Box<dyn crate::device::PlanContext>>> {
        Ok(Some(Box::new(super::sync::CudaPlanCtx::new(Arc::clone(&self.dev))?)))
    }

    /// Static function attributes; occupancy is the driver's resident-block
    /// count for the latest launch's block size (the kernel's maximum before
    /// any launch) over the SM's thread capacity.
    fn resource_usage(&self) -> Option<KernelResources> {
        let limits = self.dev.limits();
        let block = match self.last_block.load(Ordering::Relaxed) {
            0 => self.max_threads_per_block,
            block => block,
        };
        let occupancy = self
            .max_active_blocks_per_sm(block)
            .ok()
            .filter(|_| limits.max_threads_per_sm > 0)
            .map(|blocks| ((blocks * block) as f32 / limits.max_threads_per_sm as f32).min(1.0));
        Some(KernelResources {
            vgprs: Some(self.num_regs),
            sgprs: None,
            lds_bytes: self.shared_bytes,
            scratch_bytes: Some(self.local_bytes),
            wave_size: limits.warp_size,
            occupancy,
        })
    }
}
