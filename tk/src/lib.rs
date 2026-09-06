//! `svod-tk` — a ThunderKittens-style tile DSL for GPUs. It serves two
//! audiences, and the public API below is grouped to match.
//!
//! **1. Use the built-in kernels** through the [`Tensor`](svod_tensor::Tensor)
//! interface. [`matmul`] and [`flash_attention`] / [`flash_attention_with`] take
//! and return *lazy* tensors (`custom_kernel` / `Op::Call` graph nodes), so they
//! compose into a model graph and realize through the normal `prepare()` path like
//! any other op — no kernel knowledge required:
//!
//! ```no_run
//! use svod_tensor::Tensor;
//! let a = Tensor::randn(&[256, 256]).unwrap();
//! let b = Tensor::randn(&[256, 256]).unwrap();
//! if let Some(mut c) = svod_tk::matmul(&a, &b).unwrap() { // `None` if the device can't run it
//!     c.prepare().unwrap();
//! }
//! ```
//!
//! **2. Author and debug your own kernel** with the tile DSL: build a [`Kernel`]
//! out of [`Group`] tile ops, [`Loop`]s, and register/shared [tiles](tile), then
//! either wrap its SINK as a lazy graph node ([`graph_launch`], production wiring)
//! or dispatch it directly against concrete buffers for isolation/debug
//! ([`run_kernel`] / [`compile_kernel`] / [`CompiledLaunch`]). The built-in
//! [`matmul`](kernels::matmul) is the worked reference kernel.
//!
//! It is a thin eager builder, not a backend: tiles wrap UOp buffers and emit the
//! same lowered-kernel IR (`Range` + `index().store(..).end(..)`) the normal
//! renderer consumes. Port of tinygrad's `extra/thunder/tiny/tk`.
//!
//! # Supported targets
//! - **gfx942** (CDNA3) — wave64, MFMA.
//! - **gfx1151** (RDNA3.5) — wave32, WMMA.
//! - **CUDA sm_80+** — warp32, `mma.sync.m16n8k16` (a 16×16 tile as two m16n8
//!   halves, [`layout::LaneMap::MmaSync`]); [`matmul`], [`flash_attention`] and the
//!   shuffle-only [`single_query_attention`].
//!
//! Each kernel declares the arches it is built for as an [`ArchSet`]. Inputs are
//! bf16/f16, accumulation is f32, the WMMA/MFMA K-edge is 16; the per-arch
//! fragment shapes resolve through [`ArchCaps`]. The layout-table calibration is
//! pinned to gfx942 wave64 ([`WARP_THREADS`]); the live lane count flows through
//! [`ArchCaps::wave_size`](arch::ArchCaps::wave_size). On a target outside a
//! kernel's set its launcher declines (`Ok(None)`) so the caller falls back to
//! the tensor scheduler.

pub mod arch;
pub mod asm;
pub mod fingerprint;
pub mod grid;
pub mod group;
pub mod index;
pub mod kernel;
pub mod kernels;
pub mod launch;
pub mod layout;
pub mod loop_scope;
pub mod math;
pub mod ops;
pub mod scaffold;
pub mod sched;
pub mod swizzle;
pub mod target;
pub mod tile;
pub mod tiles;

/// Threads per warp/wave the **register-tile fragment-layout tables**
/// ([`tiles`] strides, [`group`]'s per-lane WMMA upcast counts) are calibrated
/// for — gfx942 wave64. The *runtime* lane count flows through
/// [`ArchCaps::wave_size`](arch::ArchCaps::wave_size); this constant is only the
/// layout-table calibration, pinned to the canonical arch by the assert below.
pub const WARP_THREADS: usize = 64;
const _: () = assert!(WARP_THREADS == ArchCaps::GFX942.wave_size);

// ── Use the built-in kernels (Tensor in → Tensor out) ───────────────────────
pub use kernels::fa::{
    FLASH_ATTENTION_SEQUENCE_MULTIPLE, FaOpts, flash_attention, flash_attention_supported, flash_attention_with,
};
pub use kernels::kmeans::{kmeans_assign, kmeans_update};
pub use kernels::knn::knn;
pub use kernels::matmul::matmul;
pub use kernels::sq_attention::{SqAttentionOpts, single_query_attention, single_query_attention_packed};
pub use launch::{Error as LaunchError, Result as LaunchResult};
pub use target::ArchSet;

// ── Author your own kernel (the tile DSL) ───────────────────────────────────
pub use arch::ArchCaps;
pub use group::{ArgDir, Group, LoadInto, MoveIdx, StoreInto, SwapDir};
pub use index::IntoIdxs;
pub use kernel::Kernel;
pub use launch::{graph_launch, graph_launch_multi, launch_custom}; // wrap a hand kernel as a lazy Tensor graph node / kernel entry
pub use layout::{LaneMap, ReduceTree};
pub use loop_scope::Loop;
pub use scaffold::GlSpec;
pub use swizzle::Swizzle;
pub use tile::{AfterDep, AfterDeps, GL, RT, RV, RegTile, ST};
pub use tiles::{
    BaseShape, RT_16X16, RT_16X16_MMA, RT_16X32, RT_32X16, RT_32X32, RTBaseShape, ST_16X16, ST_16X16_MMA,
    ST_16X16_SWIZZLED, ST_16X32, ST_32X16, ST_32X32, STBaseShape, TileLayout, VecLayout,
};

// ── Debug your kernel (direct dispatch against concrete buffers) ─────────────
pub use fingerprint::{KernelFingerprint, kernel_fingerprint};
pub use launch::{CompiledLaunch, compile, compile_kernel, launch, run_kernel};

#[cfg(test)]
mod test;
