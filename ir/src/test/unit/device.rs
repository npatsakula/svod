use crate::RendererDevice;

/// Pinned for every variant, because the benchmark scratch stream is *host*
/// memory: a GPU renderer that opts in pays for it and evicts nothing the timed
/// kernel reads. This predicate used to be `has_hardware_cache_invalidate`,
/// whose sense was the opposite and which named a primitive the workspace never
/// had — every CUDA and AMD beam search ran the stream for nothing.
#[test]
fn only_the_cpu_is_evicted_by_a_host_scratch_stream() {
    for device in [
        RendererDevice::CudaSm75,
        RendererDevice::CudaSm80,
        RendererDevice::CudaSm89,
        RendererDevice::Metal,
        RendererDevice::AmdRdna3,
        RendererDevice::AmdRdna4,
        RendererDevice::AmdCdna3,
        RendererDevice::AmdCdna4,
        RendererDevice::IntelXe,
        RendererDevice::WebGpu,
    ] {
        assert!(!device.benchmark_evicts_via_host_stream(), "{device} runs kernels out of device memory");
    }
    assert!(RendererDevice::Cpu.benchmark_evicts_via_host_stream());
}
