use std::collections::HashMap;
use std::sync::Arc;

use once_cell::sync::Lazy;
use parking_lot::RwLock;

pub use svod_dtype::DeviceSpec;

use crate::allocator::{Allocator, CpuAllocator, LruAllocator};
use crate::error::{DeviceUnavailableSnafu, InvalidDeviceSnafu, Result};
use snafu::OptionExt;

/// Extension trait for DeviceSpec to add parsing functionality.
///
/// Lives in the device crate because parsing reports this crate's error type.
pub trait DeviceSpecExt {
    /// Parse a device string into a DeviceSpec.
    ///
    /// Examples:
    /// - "CPU" -> DeviceSpec::Cpu
    /// - "CUDA:0" -> DeviceSpec::Cuda { device_id: 0 }
    /// - "cuda" -> DeviceSpec::Cuda { device_id: 0 } (default to device 0)
    fn parse(s: &str) -> Result<DeviceSpec>;
}

impl DeviceSpecExt for DeviceSpec {
    fn parse(s: &str) -> Result<Self> {
        // DISK: preserve path case (don't uppercase)
        if s.len() >= 5 && s[..5].eq_ignore_ascii_case("DISK:") {
            return Ok(DeviceSpec::Disk { path: std::path::PathBuf::from(&s[5..]) });
        }

        let s = s.to_uppercase();
        let (kind, id) = match s.split_once(':') {
            Some((kind, id)) => (kind, Some(id)),
            None => (s.as_str(), None),
        };
        // `NAME` alone selects device 0; `NAME:N` selects device N. The arch
        // of a GPU device is intentionally not part of the spec: it's a
        // hardware property of the opened device, and encoding it here would
        // give one physical device two identities.
        let device_id =
            || -> Result<usize> { id.map_or(Ok(0), |id| id.parse().ok().context(InvalidDeviceSnafu { device: &s })) };

        match kind {
            "CPU" => Ok(DeviceSpec::Cpu),
            "CUDA" | "GPU" => Ok(DeviceSpec::Cuda { device_id: device_id()? }),
            "METAL" => Ok(DeviceSpec::Metal { device_id: device_id()? }),
            "WEBGPU" => Ok(DeviceSpec::WebGpu),
            "AMD" | "HIP" => Ok(DeviceSpec::Amd { device_id: device_id()? }),
            _ => InvalidDeviceSnafu { device: s }.fail(),
        }
    }
}

#[derive(Default)]
pub struct DeviceRegistry {
    devices: RwLock<HashMap<DeviceSpec, Arc<dyn Allocator>>>,
}

impl DeviceRegistry {
    /// Get or create a device allocator.
    pub fn get(&self, spec: &DeviceSpec) -> Result<Arc<dyn Allocator>> {
        // Fast path: read lock
        {
            let devices = self.devices.read();
            if let Some(allocator) = devices.get(spec) {
                return Ok(Arc::clone(allocator));
            }
        }

        // Slow path: write lock to create
        let mut devices = self.devices.write();

        // Double-check after acquiring write lock
        if let Some(allocator) = devices.get(spec) {
            return Ok(Arc::clone(allocator));
        }

        // Create new allocator
        let allocator = self.create_allocator(spec)?;
        devices.insert(spec.clone(), Arc::clone(&allocator));
        Ok(allocator)
    }

    /// Get a device by parsing a device string.
    pub fn get_device(&self, device: &str) -> Result<Arc<dyn Allocator>> {
        let spec = <DeviceSpec as DeviceSpecExt>::parse(device)?;
        self.get(&spec)
    }

    fn create_allocator(&self, spec: &DeviceSpec) -> Result<Arc<dyn Allocator>> {
        // DISK: no LRU caching — DiskAllocator is used directly, not wrapped in LruAllocator.
        if let DeviceSpec::Disk { path } = spec {
            return Ok(Arc::new(crate::allocator::DiskAllocator::new(path.clone())));
        }

        let base: Box<dyn Allocator> = match spec {
            DeviceSpec::Cpu => Box::new(CpuAllocator),
            DeviceSpec::Cuda { device_id } => Box::new(crate::cuda::CudaAllocator::new(*device_id)?),
            DeviceSpec::Amd { device_id, .. } => Box::new(crate::amd::AmdAllocator::new(*device_id)?),
            DeviceSpec::Metal { device_id } => Box::new(crate::metal::MetalAllocator::new(*device_id)?),
            DeviceSpec::WebGpu => {
                return DeviceUnavailableSnafu { reason: "WebGPU allocator is not yet implemented" }.fail();
            }
            DeviceSpec::Disk { .. } => unreachable!(),
        };

        // Wrap with LRU cache (already thread-safe via Mutex)
        let lru = LruAllocator::new(base);

        Ok(Arc::new(lru))
    }
}

/// Global device registry instance.
static REGISTRY: Lazy<DeviceRegistry> = Lazy::new(DeviceRegistry::default);

/// Get the global device registry.
pub fn registry() -> &'static DeviceRegistry {
    &REGISTRY
}

/// Convenience function to get a device allocator by string.
pub fn get_device(device: &str) -> Result<Arc<dyn Allocator>> {
    registry().get_device(device)
}

/// Convenience function to get CPU allocator.
pub fn cpu() -> Result<Arc<dyn Allocator>> {
    registry().get(&DeviceSpec::Cpu)
}

/// Read the gfx arch of AMD device `device_id` from KFD topology. Used by
/// `DeviceSpec::parse("AMD:N")` so the resulting spec encodes the real arch
/// (not a hard-coded default that would break the kernel cache).
pub fn resolve_amd_arch_from_topology(device_id: usize) -> Result<svod_dtype::AmdArch> {
    let nodes = crate::amd::topology::enumerate();
    let node = nodes.get(device_id).ok_or_else(|| crate::error::Error::NoAmdGpu {
        reason: format!("device_id {device_id} out of range; {} GPU node(s) present", nodes.len()),
    })?;
    svod_dtype::AmdArch::from_gfx_target_version(node.gfx_target_version).ok_or_else(|| {
        crate::error::Error::DeviceUnavailable {
            reason: format!("unsupported gfx_target_version {} on AMD device {device_id}", node.gfx_target_version),
        }
    })
}

/// The compute capability of CUDA device `device_id`, read from the opened
/// (cached) device; the CUDA counterpart of [`resolve_amd_arch_from_topology`].
pub fn resolve_cuda_arch(device_id: usize) -> Result<svod_dtype::CudaArch> {
    Ok(crate::cuda::CudaDevice::open(device_id)?.arch())
}
