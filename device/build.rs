//! Build script: bindgen for the vendored AMD FFI/ABI headers.
//!
//! Two generated modules, both from headers vendored under `include/` so the
//! build needs no ROCm/KFD install — and no system kernel headers either:
//! `kfd_ioctl.h`'s `<linux/ioctl.h>`, `<linux/types.h>` and (vestigial)
//! `<drm/drm.h>` are vendored under `include/linux` and `include/drm`. This
//! runs on every platform (the AMD provider is always compiled; KFD hardware
//! is detected at runtime), so a header bump or ABI drift is caught by any
//! host's `cargo check`, not only on the Linux box.

use std::env;
use std::path::PathBuf;

fn main() {
    let out_dir: PathBuf = env::var("OUT_DIR").expect("OUT_DIR set by cargo").into();
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let include = manifest.join("include");
    let inc_arg = format!("-I{}", include.to_string_lossy());

    println!("cargo:rerun-if-changed=include/kfd_ioctl.h");
    println!("cargo:rerun-if-changed=include/linux");
    println!("cargo:rerun-if-changed=include/drm");
    let kfd = bindgen::Builder::default()
        .header(include.join("kfd_ioctl.h").to_string_lossy().into_owned())
        // Resolve the vendored <linux/ioctl.h>, <linux/types.h> and <drm/drm.h>
        // shims instead of the system kernel headers — hermetic on all hosts.
        .clang_arg(&inc_arg)
        .allowlist_type("kfd_ioctl_.*_args")
        .allowlist_type("kfd_process_device_apertures")
        .allowlist_type("kfd_event_data")
        .allowlist_type("kfd_hsa_signal_event_data")
        .allowlist_type("kfd_hsa_memory_exception_data")
        .allowlist_type("kfd_hsa_hw_exception_data")
        .allowlist_type("kfd_memory_exception_failure")
        .allowlist_type("kfd_mmio_remap")
        .allowlist_type("__u\\d+")
        .allowlist_type("__s\\d+")
        .allowlist_var("KFD_IOC_.*")
        .allowlist_var("KFD_MAX_QUEUE_PERCENTAGE")
        .allowlist_var("KFD_RUNTIME_ENABLE_MODE_ENABLE_MASK")
        .allowlist_var("AMDKFD_IOC_.*")
        .derive_default(true)
        .layout_tests(false)
        .generate_comments(false)
        .generate()
        .expect("bindgen kfd_ioctl.h");
    kfd.write_to_file(out_dir.join("kfd_sys.rs")).expect("write kfd_sys.rs");

    // HSA / AQL ABI. Tight allowlist over the vendored ROCm headers: the
    // four structs the AQL path lays into GART / the ring, plus the bit
    // enums that drive packet headers and queue/kernel-code properties.
    // Recursive allowlisting pulls their transitive field types
    // (`hsa_signal_t`, `hsa_queue_t`, …); no functions are requested, so
    // none of the libhsa-runtime API surface is emitted. `layout_tests`
    // stays ON here — these are the layout-critical descriptors (256-byte
    // `amd_queue_t`, 64-byte AQL packet) where a wrong offset silently
    // corrupts the GART, so the generated size/offset asserts are exactly
    // the regression guard we want.
    println!("cargo:rerun-if-changed=include/amd_hsa_wrapper.h");
    println!("cargo:rerun-if-changed=include/hsa");
    let hsa = bindgen::Builder::default()
        .header(include.join("amd_hsa_wrapper.h").to_string_lossy().into_owned())
        .clang_arg(&inc_arg)
        .allowlist_type("hsa_kernel_dispatch_packet_t")
        .allowlist_type("hsa_signal_t")
        .allowlist_type("hsa_queue_t")
        .allowlist_type("amd_queue_t")
        .allowlist_type("hsa_packet_header_t")
        .allowlist_type("hsa_packet_type_t")
        .allowlist_type("hsa_fence_scope_t")
        .allowlist_type("amd_queue_properties_t")
        .allowlist_type("amd_kernel_code_properties_t")
        .allowlist_type("amd_signal_t")
        .allowlist_type("amd_signal_kind_t")
        .default_enum_style(bindgen::EnumVariation::Consts)
        .derive_default(true)
        .generate_comments(false)
        .generate()
        .expect("bindgen amd_hsa_wrapper.h");
    hsa.write_to_file(out_dir.join("hsa_sys.rs")).expect("write hsa_sys.rs");
}
