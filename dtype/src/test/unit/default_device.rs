use super::*;

// Tests mutate the thread-local default; clear it on entry/exit so state can't
// leak between tests that happen to share a cargo worker thread. We assert
// against explicitly-set values rather than the post-clear fallback, since the
// process-wide env (`SVOD_DEVICE`) is cached in a `OnceCell` and would make a
// hard-coded `Cpu` expectation depend on how the suite was launched.

#[test]
fn parse_simple_variants() {
    assert_eq!(parse_simple("CPU"), Some(DeviceSpec::Cpu));
    assert_eq!(parse_simple("cpu"), Some(DeviceSpec::Cpu));
    assert_eq!(parse_simple("AMD"), Some(DeviceSpec::Amd { device_id: 0 }));
    assert_eq!(parse_simple("amd:2"), Some(DeviceSpec::Amd { device_id: 2 }));
    assert_eq!(parse_simple("HIP"), Some(DeviceSpec::Amd { device_id: 0 }));
    assert_eq!(parse_simple("HIP:1"), Some(DeviceSpec::Amd { device_id: 1 }));
    assert_eq!(parse_simple("METAL"), Some(DeviceSpec::Metal { device_id: 0 }));
    assert_eq!(parse_simple("metal:1"), Some(DeviceSpec::Metal { device_id: 1 }));
    assert_eq!(parse_simple("METAL:x"), None);
    assert_eq!(parse_simple("cuda"), Some(DeviceSpec::Cuda { device_id: 0 }));
    assert_eq!(parse_simple("CUDA:1"), Some(DeviceSpec::Cuda { device_id: 1 }));
    assert_eq!(parse_simple("nv:2"), Some(DeviceSpec::Cuda { device_id: 2 }));
    assert_eq!(parse_simple("GPU"), Some(DeviceSpec::Cuda { device_id: 0 }));
    assert_eq!(parse_simple("CUDA:x"), None);
    assert_eq!(parse_simple("webgpu"), None);
    assert_eq!(parse_simple(""), None);
    assert_eq!(parse_simple("AMD:notanum"), None);
}

#[test]
fn thread_override_takes_precedence() {
    set_default_device(DeviceSpec::Amd { device_id: 3 });
    assert_eq!(default_device(), DeviceSpec::Amd { device_id: 3 });
    set_default_device(DeviceSpec::Cpu);
    assert_eq!(default_device(), DeviceSpec::Cpu);
    clear_default_device();
}

#[test]
fn with_default_device_restores_on_normal_return() {
    set_default_device(DeviceSpec::Cpu);
    let inside = with_default_device(DeviceSpec::Amd { device_id: 1 }, default_device);
    assert_eq!(inside, DeviceSpec::Amd { device_id: 1 });
    assert_eq!(default_device(), DeviceSpec::Cpu, "default restored after scope");
    clear_default_device();
}

#[test]
fn with_default_device_restores_on_panic() {
    let before = DeviceSpec::Cpu;
    set_default_device(before.clone());
    let result = std::panic::catch_unwind(|| {
        with_default_device(DeviceSpec::Amd { device_id: 7 }, || {
            assert_eq!(default_device(), DeviceSpec::Amd { device_id: 7 });
            panic!("scoped closure blew up");
        })
    });
    assert!(result.is_err(), "the closure must have panicked");
    assert_eq!(default_device(), before, "a panic in the scope must not leak the override onto the thread");
    clear_default_device();
}

#[test]
fn nested_with_default_device_restores_each_layer() {
    set_default_device(DeviceSpec::Cpu);
    with_default_device(DeviceSpec::Amd { device_id: 0 }, || {
        assert_eq!(default_device(), DeviceSpec::Amd { device_id: 0 });
        with_default_device(DeviceSpec::Amd { device_id: 1 }, || {
            assert_eq!(default_device(), DeviceSpec::Amd { device_id: 1 });
        });
        assert_eq!(default_device(), DeviceSpec::Amd { device_id: 0 }, "inner scope restored");
    });
    assert_eq!(default_device(), DeviceSpec::Cpu, "outer scope restored");
    clear_default_device();
}

#[test]
fn spawn_with_default_device_propagates_callers_device() {
    let handle = with_default_device(DeviceSpec::Amd { device_id: 3 }, || spawn_with_default_device(default_device));
    assert_eq!(handle.join().unwrap(), DeviceSpec::Amd { device_id: 3 });
}

#[test]
fn platform_default_is_metal_only_on_macos() {
    let expected = if cfg!(target_os = "macos") { DeviceSpec::Metal { device_id: 0 } } else { DeviceSpec::Cpu };
    assert_eq!(platform_default(), expected);
}
