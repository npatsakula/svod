//! A compute pipeline loaded from a metallib (or MSL source in fallback mode)
//! and dispatched one command buffer per launch; see [`super::graph`] for the
//! batched replay path.

use std::ffi::c_void;
use std::ptr::null_mut;
use std::sync::Arc;

use super::compile::{new_library_from_source, validate_metallib};
use super::device::MetalDevice;
use super::objc::{
    AutoreleasePool, Id, MTL_PIPELINE_OPTION_NONE, MTLSize, NSUInteger, ObjcBool, ObjcId, ns_error_message, ns_string,
};
use crate::device::{AbiParamDescriptor, Program};
use crate::profile::KernelResources;
use crate::sync::DispatchTimestamps;
use crate::{Error, Result};

/// `[[buffer(n)]]` indices are positional in MSL, so the ABI slot is the bind
/// index; Metal exposes 31 of them.
pub(crate) const MAX_BUFFER_BINDINGS: usize = 31;
/// `maxThreadsPerThreadgroup` of every Apple and Mac2 GPU; a pipeline reports
/// less only when its register demand forces it.
const DEVICE_MAX_THREADS_PER_THREADGROUP: usize = 1024;

/// One resolved kernel argument, ready to bind.
enum Arg {
    Buffer(Id, NSUInteger),
    Scalar(i32),
}

pub struct MetalProgram {
    dev: Arc<MetalDevice>,
    name: String,
    label: ObjcId,
    _library: ObjcId,
    _function: ObjcId,
    pipeline: ObjcId,
    abi: Vec<AbiParamDescriptor>,
    buf_count: usize,
    var_count: usize,
    max_total_threads: usize,
    thread_execution_width: usize,
    static_threadgroup_memory: usize,
}

impl std::fmt::Debug for MetalProgram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalProgram")
            .field("name", &self.name)
            .field("buf_count", &self.buf_count)
            .field("var_count", &self.var_count)
            .field("max_total_threads", &self.max_total_threads)
            .finish_non_exhaustive()
    }
}

impl MetalProgram {
    /// `msl-kernel-abi-v1` binds arguments **positionally**: the MSL signature is
    /// emitted by walking the ABI list in order (`CRenderer`'s param loop) and
    /// [`Self::execute`] binds `args.iter().enumerate()`, so the binding index is the
    /// parameter's POSITION, never its PARAM slot. The slot only spells the
    /// parameter's name (`data{slot}`).
    ///
    /// So slots may have **gaps** — the scheduler leaves them whenever it eliminates
    /// a param, and `[0, 1, 2, 4, 6, ...]` binds exactly as well as `[0, 1, 2, 3]`.
    /// What they may not do is descend: the order the ABI list is walked is the order
    /// the caller stacks its buffers, so a list that is not ascending would pair a
    /// buffer with the wrong parameter.
    pub fn load(dev: Arc<MetalDevice>, bytes: &[u8], name: &str, abi: &[AbiParamDescriptor]) -> Result<Self> {
        let ascending = abi.windows(2).all(|pair| pair[0].slot < pair[1].slot);
        if !ascending || abi.len() > MAX_BUFFER_BINDINGS {
            return Err(Error::ProgramAbiMismatch {
                reason: format!(
                    "msl-kernel-abi-v1 binds positionally: slots must ascend and number at most \
                     {MAX_BUFFER_BINDINGS}, got {:?}",
                    abi.iter().map(|param| param.slot).collect::<Vec<_>>()
                ),
            });
        }
        validate_metallib(bytes, name)?;

        let objc = dev.objc();
        let _pool = AutoreleasePool::push(objc);
        let sels = &objc.sels;
        let runtime = |message: String| Error::Runtime { message };

        let library = if bytes.starts_with(b"MTLB") {
            // SAFETY: `dispatch_data_create` copies the bytes (no destructor); `newLibraryWithData:error:` returns +1 or nil.
            unsafe {
                let data = ObjcId::adopt((objc.dispatch_data_create)(
                    bytes.as_ptr().cast(),
                    bytes.len(),
                    null_mut(),
                    null_mut(),
                ))
                .ok_or_else(|| runtime("dispatch_data_create returned nil".into()))?;
                let mut error: Id = null_mut();
                let library = objc.send2::<Id, *mut Id, Id>(
                    dev.mtl(),
                    sels.new_library_with_data_error,
                    data.as_raw(),
                    &mut error,
                );
                ObjcId::adopt(library).ok_or_else(|| {
                    runtime(format!(
                        "newLibraryWithData failed for kernel {name:?}: {}",
                        ns_error_message(objc, error).unwrap_or_default()
                    ))
                })?
            }
        } else {
            let source =
                std::str::from_utf8(bytes).map_err(|_| runtime("Metal fallback payload is not UTF-8 MSL".into()))?;
            new_library_from_source(&dev, source)?
        };

        let label = ns_string(objc, name)?;
        // SAFETY: `newFunctionWithName:` takes an NSString and returns +1 or nil.
        let function = unsafe {
            ObjcId::adopt(objc.send1::<Id, Id>(library.as_raw(), sels.new_function_with_name, label.as_raw()))
        }
        .ok_or_else(|| runtime(format!("Metal library has no kernel entry point {name:?}")))?;

        let descriptor = objc.new_object(objc.classes.compute_pipeline_descriptor)?;
        // SAFETY: setters with an object / BOOL argument.
        unsafe {
            objc.send1::<Id, ()>(descriptor.as_raw(), sels.set_compute_function, function.as_raw());
            objc.send1::<ObjcBool, ()>(descriptor.as_raw(), sels.set_support_indirect_command_buffers, 1);
        }
        let mut error: Id = null_mut();
        // SAFETY: (descriptor, options, reflection out-pointer, error out-pointer) → +1 pipeline or nil.
        let pipeline = unsafe {
            ObjcId::adopt(objc.send4::<Id, NSUInteger, *mut Id, *mut Id, Id>(
                dev.mtl(),
                sels.new_compute_pipeline_state,
                descriptor.as_raw(),
                MTL_PIPELINE_OPTION_NONE,
                null_mut(),
                &mut error,
            ))
        }
        .ok_or_else(|| {
            // SAFETY: autoreleased NSError or nil.
            runtime(format!(
                "compute pipeline creation failed for kernel {name:?}: {}",
                unsafe { ns_error_message(objc, error) }.unwrap_or_default()
            ))
        })?;
        // SAFETY: three NSUInteger accessors.
        let (max_total_threads, thread_execution_width, static_threadgroup_memory) = unsafe {
            (
                objc.send0::<NSUInteger>(pipeline.as_raw(), sels.max_total_threads_per_threadgroup) as usize,
                objc.send0::<NSUInteger>(pipeline.as_raw(), sels.thread_execution_width) as usize,
                objc.send0::<NSUInteger>(pipeline.as_raw(), sels.static_threadgroup_memory_length) as usize,
            )
        };

        Ok(Self {
            dev,
            name: name.to_string(),
            label,
            _library: library,
            _function: function,
            pipeline,
            abi: abi.to_vec(),
            buf_count: abi.iter().filter(|param| param.is_storage()).count(),
            var_count: abi.iter().filter(|param| !param.is_storage()).count(),
            max_total_threads,
            thread_execution_width,
            static_threadgroup_memory,
        })
    }

    pub fn max_total_threads_per_threadgroup(&self) -> usize {
        self.max_total_threads
    }

    pub(crate) fn pipeline(&self) -> Id {
        self.pipeline.as_raw()
    }

    pub(crate) fn buf_count(&self) -> usize {
        self.buf_count
    }

    pub(crate) fn var_count(&self) -> usize {
        self.var_count
    }

    /// `(threadgroups, threads per group)` for a launch. `global_size` is
    /// already the threadgroup count and `local_size` the threads per group;
    /// direct-id kernels run one thread per group.
    pub(crate) fn launch_sizes(
        &self,
        global_size: Option<[usize; 3]>,
        local_size: Option<[usize; 3]>,
    ) -> Result<(MTLSize, MTLSize)> {
        let groups = global_size.unwrap_or([1, 1, 1]);
        let threads = local_size.unwrap_or([1, 1, 1]);
        let thread_count = threads.iter().try_fold(1usize, |acc, dim| acc.checked_mul(*dim)).unwrap_or(usize::MAX);
        if thread_count > self.max_total_threads {
            return Err(Error::Runtime {
                message: format!(
                    "Metal kernel '{}' local size {threads:?} ({thread_count} threads) exceeds maxTotalThreadsPerThreadgroup {} \
                     (threadExecutionWidth {}, staticThreadgroupMemoryLength {} bytes)",
                    self.name, self.max_total_threads, self.thread_execution_width, self.static_threadgroup_memory
                ),
            });
        }
        Ok((MTLSize::from(groups), MTLSize::from(threads)))
    }

    /// Encode one dispatch into its own command buffer and commit it.
    fn submit(
        &self,
        buffers: &[*mut u8],
        vals: &[i64],
        global_size: Option<[usize; 3]>,
        local_size: Option<[usize; 3]>,
    ) -> Result<ObjcId> {
        if buffers.len() != self.buf_count || vals.len() != self.var_count {
            return Err(Error::ProgramAbiMismatch {
                reason: format!(
                    "kernel {} expects {} buffers and {} scalars, got {} and {}",
                    self.name,
                    self.buf_count,
                    self.var_count,
                    buffers.len(),
                    vals.len()
                ),
            });
        }
        let (groups, threads) = self.launch_sizes(global_size, local_size)?;

        // Resolve every argument before any Metal object exists: a command
        // encoder released without `endEncoding` is a fatal Metal assertion,
        // so the encoding loop below must be infallible.
        let mut args = Vec::with_capacity(self.abi.len());
        let (mut buffer_index, mut val_index) = (0usize, 0usize);
        for param in &self.abi {
            if param.is_storage() {
                let (buffer, offset) = self.dev.resolve(buffers[buffer_index])?;
                buffer_index += 1;
                args.push(Arg::Buffer(buffer, offset));
            } else {
                let value = i32::try_from(vals[val_index]).map_err(|_| Error::Runtime {
                    message: format!("Metal scalar argument {val_index} value {} does not fit i32", vals[val_index]),
                })?;
                val_index += 1;
                args.push(Arg::Scalar(value));
            }
        }

        let objc = self.dev.objc();
        let _pool = AutoreleasePool::push(objc);
        let sels = &objc.sels;
        let (command_buffer, encoder) = self.dev.begin_compute()?;
        let enc = encoder.as_raw();

        // SAFETY: each selector is sent with its documented argument types.
        unsafe {
            objc.send1::<Id, ()>(enc, sels.set_compute_pipeline_state, self.pipeline.as_raw());
            for (index, arg) in args.iter().enumerate() {
                let index = index as NSUInteger;
                match arg {
                    Arg::Buffer(buffer, offset) => {
                        objc.send3::<Id, NSUInteger, NSUInteger, ()>(
                            enc,
                            sels.set_buffer_offset_at_index,
                            *buffer,
                            *offset,
                            index,
                        );
                    }
                    Arg::Scalar(value) => objc.send3::<*const c_void, NSUInteger, NSUInteger, ()>(
                        enc,
                        sels.set_bytes_length_at_index,
                        (&raw const *value).cast(),
                        std::mem::size_of::<i32>() as NSUInteger,
                        index,
                    ),
                }
            }
            objc.send2::<MTLSize, MTLSize, ()>(enc, sels.dispatch_threadgroups, groups, threads);
            objc.send0::<()>(enc, sels.end_encoding);
            objc.send1::<Id, ()>(command_buffer.as_raw(), sels.set_label, self.label.as_raw());
            objc.send0::<()>(command_buffer.as_raw(), sels.commit);
        }
        Ok(command_buffer)
    }

    fn wait(&self, command_buffer: &ObjcId) -> Result<()> {
        self.dev.wait_command_buffer(command_buffer.as_raw(), &format!("Metal kernel '{}'", self.name))
    }
}

impl Program for MetalProgram {
    unsafe fn execute(
        &self,
        buffers: &[*mut u8],
        vals: &[i64],
        global_size: Option<[usize; 3]>,
        local_size: Option<[usize; 3]>,
        wait: bool,
    ) -> Result<()> {
        let command_buffer = self.submit(buffers, vals, global_size, local_size)?;
        if wait {
            self.wait(&command_buffer)
        } else {
            self.dev.push_in_flight(command_buffer);
            Ok(())
        }
    }

    /// The run is warm: the command buffer's own GPU timestamps bracket it, and
    /// nothing flushes the caches the queue's earlier work left.
    unsafe fn execute_timed(
        &self,
        buffers: &[*mut u8],
        vals: &[i64],
        global_size: Option<[usize; 3]>,
        local_size: Option<[usize; 3]>,
    ) -> Result<Option<std::time::Duration>> {
        let command_buffer = self.submit(buffers, vals, global_size, local_size)?;
        self.wait(&command_buffer)?;
        let _pool = AutoreleasePool::push(self.dev.objc());
        let timestamps = MetalDispatchTimestamps::read(&self.dev, command_buffer.as_raw());
        Ok(timestamps.timestamps_ns().map(|(start, end)| std::time::Duration::from_nanos(end - start)))
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    /// The pipeline's static limits; Metal exposes no register counts, so the
    /// threadgroup cap stands in for register-limited occupancy.
    fn resource_usage(&self) -> Option<KernelResources> {
        Some(KernelResources {
            vgprs: None,
            sgprs: None,
            lds_bytes: self.static_threadgroup_memory as u32,
            scratch_bytes: None,
            wave_size: self.thread_execution_width as u32,
            occupancy: Some(self.max_total_threads as f32 / DEVICE_MAX_THREADS_PER_THREADGROUP as f32),
        })
    }
}

/// A completed command buffer's `GPUStartTime`/`GPUEndTime`, converted from
/// seconds on the GPU clock.
#[derive(Debug, Clone, Copy)]
pub struct MetalDispatchTimestamps {
    start_ns: u64,
    end_ns: u64,
}

impl MetalDispatchTimestamps {
    /// Read the stamps of a command buffer that has completed.
    pub(crate) fn read(dev: &MetalDevice, command_buffer: Id) -> Self {
        let (start, end) = dev.gpu_times(command_buffer);
        Self { start_ns: (start * 1e9) as u64, end_ns: (end * 1e9) as u64 }
    }

    /// Stamps already converted to nanoseconds (Metal 4 counter heap).
    pub(crate) fn from_ns(start_ns: u64, end_ns: u64) -> Self {
        Self { start_ns, end_ns }
    }
}

impl DispatchTimestamps for MetalDispatchTimestamps {
    fn timestamps_ns(&self) -> Option<(u64, u64)> {
        // Zero stamps mean the driver did not record them (e.g. an empty encoder).
        (self.end_ns >= self.start_ns && self.start_ns > 0).then_some((self.start_ns, self.end_ns))
    }
}
