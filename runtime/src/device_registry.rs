//! Device factory registry for runtime device creation and caching.
//!
//! This module provides a registry for full Device objects (renderer + compiler + runtime + allocator).
//! It's separate from `svod_device::registry::DeviceRegistry` (which only manages allocators)
//! to avoid circular dependencies between `device` and `runtime` crates.

use std::collections::HashMap;
use std::sync::Arc;

use once_cell::sync::Lazy;
use parking_lot::RwLock;
use svod_device::Result as DeviceResult;
use svod_device::device::Device;
use svod_device::registry::DeviceRegistry;
use svod_dtype::DeviceSpec;

use crate::error::{Result, UnsupportedDeviceSnafu};

/// Factory function that creates a Device for a given DeviceSpec.
///
/// The factory receives both the device specification and the allocator registry,
/// allowing it to obtain the correct allocator for the device.
///
/// Returns `DeviceResult<Device>` (from svod_device) since device creation
/// errors come from the device crate.
pub type DeviceFactory = Arc<dyn Fn(&DeviceSpec, &DeviceRegistry) -> DeviceResult<Device> + Send + Sync>;

/// Registry for full Device objects with caching and factory registration.
///
/// # Thread Safety
///
/// This registry uses `parking_lot::RwLock` for efficient concurrent access:
/// - Multiple readers can access cached devices simultaneously
/// - Writers acquire exclusive lock only when creating new devices
/// - Double-checked locking pattern prevents redundant device creation
///
/// # Example
///
/// ```ignore
/// // Get a device (creates if not cached)
/// let alloc_registry = svod_device::registry::registry();
/// let device = DEVICE_FACTORIES.device(&DeviceSpec::Cpu, alloc_registry)?;
///
/// // Register a custom factory
/// DEVICE_FACTORIES.register_factory("CUSTOM", Arc::new(|spec, reg| {
///     // Create custom device...
/// }));
/// ```
/// Per-spec construction slot: the mutex serializes same-spec construction
/// outside the map locks; `None` means not yet (or unsuccessfully) built.
type DeviceSlot = Arc<parking_lot::Mutex<Option<Arc<Device>>>>;

pub struct DeviceFactoryRegistry {
    /// Cached device instances (DeviceSpec -> Device)
    devices: RwLock<HashMap<DeviceSpec, DeviceSlot>>,
    /// Registered factories (device type string -> factory function)
    factories: RwLock<HashMap<String, DeviceFactory>>,
}

impl DeviceFactoryRegistry {
    /// Create a new registry with built-in device factories registered.
    pub fn new() -> Self {
        let registry = Self { devices: RwLock::new(HashMap::new()), factories: RwLock::new(HashMap::new()) };

        // Register built-in CPU factory
        registry
            .register_factory("CPU", Arc::new(|_spec, alloc_reg| crate::devices::cpu::create_cpu_device(alloc_reg)));

        // AMD factory (KFD-direct). Always compiled; registered as an execution
        // provider only when a supported GPU is actually present, so a host with
        // no /dev/kfd (or no AMD hardware) cleanly has no "AMD" device type. The
        // closure constructs the device end-to-end, including the RuntimeFactory
        // that produces `AmdProgram` from a `CompiledSpec`.
        if svod_device::amd::has_devices() {
            registry.register_factory(
                "AMD",
                Arc::new(|spec, alloc_reg| {
                    use svod_ir::DeviceSpec;
                    let device_id = match spec {
                        DeviceSpec::Amd { device_id } => *device_id,
                        _ => {
                            return Err(svod_device::Error::DeviceUnavailable {
                                reason: format!("AMD factory called with non-AMD spec: {spec:?}"),
                            });
                        }
                    };
                    // Resolve arch from KFD topology — it lives on the opened
                    // device, not the spec.
                    let arch = svod_device::registry::resolve_amd_arch_from_topology(device_id)?;
                    crate::devices::amd::create_amd_device(alloc_reg, device_id, arch)
                }),
            );
        }

        // Metal factory. Same contract as AMD: always compiled, registered only
        // when the Apple frameworks load and a default GPU exists, so Linux
        // (dlopen fails) cleanly has no "METAL" device type.
        if svod_device::metal::has_devices() {
            registry.register_factory(
                "METAL",
                Arc::new(|spec, alloc_reg| {
                    let svod_ir::DeviceSpec::Metal { device_id } = spec else {
                        return Err(svod_device::Error::DeviceUnavailable {
                            reason: format!("Metal factory called with non-Metal spec: {spec:?}"),
                        });
                    };
                    crate::devices::metal::create_metal_device(alloc_reg, *device_id)
                }),
            );
        }

        // CUDA factory. Same contract again: always compiled, registered only
        // when `libcuda.so.1` loads and reports a device. The compute
        // capability lives on the opened device, not the spec.
        if svod_device::cuda::has_devices() {
            registry.register_factory(
                "CUDA",
                Arc::new(|spec, alloc_reg| {
                    let svod_ir::DeviceSpec::Cuda { device_id } = spec else {
                        return Err(svod_device::Error::DeviceUnavailable {
                            reason: format!("CUDA factory called with non-CUDA spec: {spec:?}"),
                        });
                    };
                    let arch = svod_device::registry::resolve_cuda_arch(*device_id)?;
                    crate::devices::cuda::create_cuda_device(alloc_reg, *device_id, arch)
                }),
            );
        }

        registry
    }

    /// Register a device factory for a device type.
    ///
    /// The device type string is case-insensitive (converted to uppercase).
    /// This allows plugins or extensions to register new device types at runtime.
    ///
    /// # Arguments
    ///
    /// * `device_type` - Device type identifier (e.g., "CPU", "CUDA", "METAL")
    /// * `factory` - Factory function that creates Device instances
    pub fn register_factory(&self, device_type: &str, factory: DeviceFactory) {
        self.factories.write().insert(device_type.to_uppercase(), factory);
    }

    /// Get or create a Device for the given specification.
    ///
    /// Construction (KFD open, toolchain probe) runs OUTSIDE the map locks,
    /// serialized per-spec on a slot mutex: concurrent first-touches of
    /// different devices construct in parallel, same-spec racers construct
    /// exactly once, and a failed construction leaves the slot empty so a
    /// later call retries.
    ///
    /// # Arguments
    ///
    /// * `spec` - Device specification (e.g., `DeviceSpec::Cpu`)
    /// * `alloc_registry` - Allocator registry for obtaining device allocators
    ///
    /// # Returns
    ///
    /// Arc-wrapped Device for the specification, or error if device type unsupported.
    pub fn device(&self, spec: &DeviceSpec, alloc_registry: &DeviceRegistry) -> Result<Arc<Device>> {
        let slot = {
            let devices = self.devices.read();
            devices.get(spec).cloned()
        };
        let slot = match slot {
            Some(slot) => slot,
            None => {
                let mut devices = self.devices.write();
                devices.entry(spec.clone()).or_default().clone()
            }
        };

        let mut cell = slot.lock();
        if let Some(device) = cell.as_ref() {
            return Ok(Arc::clone(device));
        }

        let device_type = spec.base_type();
        let factory = self
            .factories
            .read()
            .get(device_type)
            .cloned()
            .ok_or_else(|| UnsupportedDeviceSnafu { device: device_type.to_string() }.build())?;

        // Construct under the per-spec slot lock only; an error leaves the
        // slot empty so the next caller retries.
        let device = Arc::new(factory(spec, alloc_registry)?);
        *cell = Some(Arc::clone(&device));
        Ok(device)
    }
}

impl Default for DeviceFactoryRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Global device factory registry.
///
/// This static instance is lazily initialized on first access,
/// with built-in device factories automatically registered.
///
/// # Example
///
/// ```ignore
/// let device = svod_runtime::DEVICE_FACTORIES
///     .device(&DeviceSpec::Cpu, svod_device::registry::registry())?;
/// ```
pub static DEVICE_FACTORIES: Lazy<DeviceFactoryRegistry> = Lazy::new(DeviceFactoryRegistry::new);
