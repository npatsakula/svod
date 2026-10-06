//! Userspace AM driver internals: page tables and allocators for gfx9/gfx11,
//! hardware bring-up for the gfx9.4.x SR-IOV VF.
//!
//! **Status: experimental, not reachable from [`crate::amd::AmdDevice`].** The
//! [`mm`](crate::amd::am::mm) submodule (TLSF sub-allocators + the GMMU
//! page-table / PTE encoding) is pure logic with no MMIO, so the address-space
//! math is unit-tested without a GPU. [`AmDev`](crate::amd::am::dev::AmDev)
//! brings up a gfx9.4.x (CDNA3) SR-IOV virtual function: GIM mailbox
//! handshake, IP discovery, GMC page-table contexts, plus a minimal MEC compute
//! queue (`ip::gfx`) and an SDMA ring (`ip::sdma`). It needs amdgpu unbound and
//! root. No
//! [`crate::amd::iface::AmdIface`] implementor ties it to the runtime yet, and
//! `SVOD_AMD_BACKEND` accepts only `kfd`. The module compiles unconditionally
//! on Unix (no extra deps), so it is always type-checked and tested.
//!
//! Arch parametrization is data-driven: register tables are selected by the
//! `ip_ver` tuples read from IP discovery (`regs::select`), with per-arch
//! deltas as small branches inside shared modules. The gfx12 PTE encoding is
//! still unimplemented.

pub mod dev;
pub mod discovery;
pub mod ip;
pub mod mailbox;
pub mod mm;
pub mod pci;
pub mod regaccess;
pub mod regs;
pub mod rlcg;
