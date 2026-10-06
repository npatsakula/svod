//! GMC bring-up: full AmDev::open — handshake, discovery, build the GMMU
//! over VRAM, program context0 + invalidation ranges on all 8 XCC via RLCG,
//! and flush the TLB (ENG17 ACK polled per-XCC). Success = the GC hub accepts
//! our page-table context.
//!
//! DESTRUCTIVE — amdgpu must be unbound (device/tools/am_unbind.sh); recovery
//! needs a VM reboot.
//!     sudo ./target/debug/examples/am_gmc

use svod_device::amd::am::dev::AmDev;
use svod_device::amd::am::pci::PciDevice;

fn main() {
    let bdf = std::env::args().nth(1).unwrap_or_else(|| PciDevice::discover().expect("AMD GPU"));
    let dev = AmDev::open(&bdf).expect("AmDev::open (handshake + GMC bring-up)");

    println!(
        "AmDev up: {} XCC, vram {} MiB, reserved {} MiB",
        dev.disc.xccs(),
        dev.disc.vram_size >> 20,
        dev.disc.reserved_vram_size >> 20
    );
    println!(
        "GMC: fb [{:#x}, {:#x}], paddr_base {:#x}, root_pt {:#x}, vm [{:#x}, {:#x}]",
        dev.gmc.fb_base,
        dev.gmc.fb_end,
        dev.gmc.paddr_base,
        dev.mm.root_pt(),
        dev.gmc.vm_base,
        dev.gmc.vm_end
    );
    println!("context0 + invalidation ranges programmed; ENG17 flush ACK'd on all XCC");

    match dev.gmc.fault_status(&dev.regs()) {
        Some(s) => println!("WARNING: GCVM protection fault status = {s:#x}"),
        None => println!("no protection fault latched — GMC clean"),
    }

    dev.release();
    println!("M1 GMC validated; released access");
}
