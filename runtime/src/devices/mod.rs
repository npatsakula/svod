//! Device implementations for different backends.

pub mod amd;
pub mod cpu;
pub mod cuda;
pub mod metal;

pub use amd::{create_amd_codegen, create_amd_device};

/// The [`ProgramSpec`] of a rendered kernel: source, ABI and var names carried
/// over, `buf_count` from its buffer arguments.
pub(crate) fn program_spec(
    rendered: &svod_codegen::RenderedKernel,
    device: &svod_dtype::DeviceSpec,
    ast: &std::sync::Arc<svod_ir::UOp>,
) -> svod_device::device::ProgramSpec {
    let mut spec = svod_device::device::ProgramSpec::new(
        rendered.name.clone(),
        rendered.code.clone(),
        device.clone(),
        ast.clone(),
    );
    spec.set_var_names(rendered.var_names.clone());
    spec.abi = rendered.abi.clone();
    if spec.buf_count == 0 {
        spec.buf_count = rendered.buffer_args.len();
    }
    spec
}
pub use cpu::{
    CpuBackend, cpu_device_with_backend, create_cpu_codegen, create_cpu_device, create_cpu_device_with_backend,
    ensure_thread_pool,
};
pub use cuda::{create_cuda_codegen, create_cuda_device, create_cuda_program};
pub use metal::{create_metal_codegen, create_metal_device, create_metal_program};
