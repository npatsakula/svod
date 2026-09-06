//! Thread-local default `DeviceSpec` for tensor construction and rangeify.
//!
//! Lives in `svod-dtype` (the lowest dep) so any crate — `svod-tensor`
//! constructors, `svod-schedule` rangeify intermediates, etc. — can read the
//! same default without forming a dependency cycle.
//!
//! Precedence:
//!   1. `with_default_device(spec, fn)` scope (thread-local).
//!   2. `set_default_device(spec)` thread-local override.
//!   3. `SVOD_DEVICE` env var (parsed once at first access).
//!   4. [`platform_default`]: `METAL:0` on macOS, `CPU` elsewhere.
//!
//! Note: parsing is intentionally restricted to `NAME[:N]` forms ("CPU",
//! "AMD", "METAL", "CUDA" and their aliases). Devices that need richer specs
//! (DISK paths) go through `svod_device::DeviceSpecExt`.

use std::cell::RefCell;

use once_cell::sync::OnceCell;

use crate::DeviceSpec;

thread_local! {
    static THREAD_DEFAULT: RefCell<Option<DeviceSpec>> = const { RefCell::new(None) };
}

static PROCESS_DEFAULT: OnceCell<DeviceSpec> = OnceCell::new();

/// Set the current thread's default device.
pub fn set_default_device(spec: DeviceSpec) {
    THREAD_DEFAULT.with(|t| *t.borrow_mut() = Some(spec));
}

/// Clear the thread-local override; subsequent calls fall back to env / platform default.
pub fn clear_default_device() {
    THREAD_DEFAULT.with(|t| *t.borrow_mut() = None);
}

/// Resolve the active default device.
pub fn default_device() -> DeviceSpec {
    if let Some(d) = THREAD_DEFAULT.with(|t| t.borrow().clone()) {
        return d;
    }
    PROCESS_DEFAULT
        .get_or_init(|| {
            std::env::var("SVOD_DEVICE").ok().and_then(|s| parse_simple(s.trim())).unwrap_or_else(platform_default)
        })
        .clone()
}

/// The device used when nothing selects one: the Apple GPU on macOS (every
/// Mac has a Metal device; tinygrad defaults the same way), the CPU elsewhere.
/// `SVOD_DEVICE=CPU` opts out.
pub fn platform_default() -> DeviceSpec {
    if cfg!(target_os = "macos") { DeviceSpec::Metal { device_id: 0 } } else { DeviceSpec::Cpu }
}

/// Scoped override: runs `f` with `spec` as the default device, restoring
/// the previous value when the scope ends — including if `f` panics. The
/// restore is RAII so a caught unwind can't leak the override onto the thread.
pub fn with_default_device<R>(spec: DeviceSpec, f: impl FnOnce() -> R) -> R {
    let _guard = DefaultDeviceGuard { prev: THREAD_DEFAULT.with(|t| t.borrow().clone()) };
    set_default_device(spec);
    f()
}

/// Restores the captured thread-local default on drop.
struct DefaultDeviceGuard {
    prev: Option<DeviceSpec>,
}

impl Drop for DefaultDeviceGuard {
    fn drop(&mut self) {
        THREAD_DEFAULT.with(|t| *t.borrow_mut() = self.prev.take());
    }
}

/// Spawn a thread that inherits the CALLER's effective default device.
///
/// `default_device()` is thread-local: a bare `std::thread::spawn` silently
/// falls back to the process default (env var / CPU) even when the spawning
/// thread has an override, so tensors built on the worker land on the wrong
/// device. This captures the caller's effective default and installs it as a
/// scoped override on the new thread.
pub fn spawn_with_default_device<F, T>(f: F) -> std::thread::JoinHandle<T>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    let device = default_device();
    std::thread::spawn(move || with_default_device(device, f))
}

/// Minimal `DeviceSpec` parser for the `SVOD_DEVICE` env var. We intentionally
/// do NOT pull `svod_device::DeviceSpecExt` here — that crate depends on
/// `svod-dtype`, not the other way around. Supports `CPU`, `AMD[:N]` / `HIP`,
/// `METAL[:N]` and `CUDA[:N]` / `NV` / `GPU`; the GPU arch is a property of
/// the opened device, never of the spec.
fn parse_simple(s: &str) -> Option<DeviceSpec> {
    let upper = s.to_uppercase();
    let parts: Vec<&str> = upper.split(':').collect();
    let device_id = || -> Option<usize> { if parts.len() > 1 { parts[1].parse().ok() } else { Some(0) } };
    match parts[0] {
        "CPU" => Some(DeviceSpec::Cpu),
        "AMD" | "HIP" => Some(DeviceSpec::Amd { device_id: device_id()? }),
        "METAL" => Some(DeviceSpec::Metal { device_id: device_id()? }),
        "CUDA" | "NV" | "GPU" => Some(DeviceSpec::Cuda { device_id: device_id()? }),
        _ => None,
    }
}

#[cfg(test)]
#[path = "test/unit/default_device.rs"]
mod tests;
