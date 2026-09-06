use crate::DeviceSpec;

#[test]
fn test_registry_cpu() {
    let allocator = crate::cpu().unwrap();
    assert_eq!(allocator.name(), "CPU");
}

#[test]
fn test_max_buffers_cpu() {
    let spec = DeviceSpec::Cpu;
    assert_eq!(spec.max_buffers(), None, "CPU should have no buffer limit");
}

#[test]
fn test_max_buffers_cuda() {
    let spec = DeviceSpec::Cuda { device_id: 0 };
    assert_eq!(spec.max_buffers(), Some(128));
}

#[test]
fn test_max_buffers_metal() {
    let spec = DeviceSpec::Metal { device_id: 0 };
    assert_eq!(spec.max_buffers(), Some(31), "Metal should have 31 buffer limit");
}

#[test]
fn test_max_buffers_webgpu() {
    let spec = DeviceSpec::WebGpu;
    assert_eq!(spec.max_buffers(), Some(8), "WebGPU should have 8 buffer limit");
}
