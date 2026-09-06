//! Kernel-scaffold helpers — the declarative preamble every builder re-typed by
//! hand: typed GL binding, role-based tile shortcuts, grid-index accessors, and
//! divisibility checks.
//!
//! Each helper is a thin, **allocation-order-preserving** forwarder over the
//! [`Kernel`]/[`crate::ArchCaps`] primitives, so a kernel migrated to the scaffold
//! emits the *identical* UOp graph (same `Param`/`DefineReg`/`DefineLocal` slot ids
//! → same content hash) — verified by the golden fingerprints
//! ([`crate::kernel_fingerprint`]). The point is to make the load-bearing invariants
//! safe by construction instead of by comment: the ABI slot order ([`Kernel::bind_abi`]
//! binds outputs-then-inputs by parameter structure) and the role→fragment
//! resolution (the tile shortcuts resolve [`crate::arch::FragRole`] via `caps`, so a
//! kernel never names a physical `RT_16X16`-family constant).

use std::sync::Arc;

use svod_dtype::DType;
use svod_ir::UOp;

use crate::arch::FragRole;
use crate::kernel::Kernel;
use crate::tile::{GL, RT, RV, ST};
use crate::tiles::{RTBaseShape, STBaseShape, TileLayout, VecLayout};

/// A global-buffer binding spec for [`Kernel::bind_abi`] (logical shape + element
/// dtype). The concrete buffer's dtype governs; `dtype` carries the author's intent.
#[derive(Clone, Debug)]
pub struct GlSpec {
    shape: Vec<usize>,
    dtype: DType,
}

impl GlSpec {
    /// A GLOBAL-buffer spec (logical `shape` + element `dtype`) for ABI binding
    /// via [`Kernel::bind_abi`].
    pub fn new(shape: &[usize], dtype: DType) -> Self {
        Self { shape: shape.to_vec(), dtype }
    }
}

impl Kernel {
    /// Bind `outputs` then `inputs` as GL tiles, **in that order** — so the ABI slot
    /// order (the kernel's buffer / `Param` order) is fixed by the parameter
    /// structure rather than by the order of free-standing `gl` calls + a comment.
    /// Calls [`Kernel::gl`] in slice order, so the `Param` slots are identical to the
    /// hand-written sequence. A conditional/optional buffer binds with a plain
    /// [`Kernel::gl`] *after* this call (trailing-only — never interleaved).
    pub fn bind_abi(&self, outputs: &[GlSpec], inputs: &[GlSpec]) -> (Vec<GL>, Vec<GL>) {
        let outs = outputs.iter().map(|s| self.gl(&s.shape, s.dtype.clone())).collect();
        let ins = inputs.iter().map(|s| self.gl(&s.shape, s.dtype.clone())).collect();
        (outs, ins)
    }

    /// The grid block index on axis 0, named for readability (`block_idx[0]`).
    pub fn grid_x(&self) -> Arc<UOp> {
        self.block_idx[0].clone()
    }
    /// The grid block index on axis 1 (`block_idx[1]`).
    pub fn grid_y(&self) -> Arc<UOp> {
        self.block_idx[1].clone()
    }
    /// The grid block index on axis 2 (`block_idx[2]`).
    pub fn grid_z(&self) -> Arc<UOp> {
        self.block_idx[2].clone()
    }

    /// The arch's physical fragment for `role` ([`crate::ArchCaps::frag`]) for a
    /// kernel that requires a matrix-core layout.
    ///
    /// # Panics
    /// Panics when tk defines no matrix-core fragment layouts for the arch (Metal,
    /// pre-Ampere CUDA) — an authoring error: such kernels must gate on an
    /// [`crate::ArchSet`] that excludes those arches.
    pub fn frag(&self, role: FragRole) -> RTBaseShape {
        self.caps.frag(role).unwrap_or_else(|| self.no_layout(&format!("{role:?} fragment")))
    }
    fn no_layout<T>(&self, what: &str) -> T {
        panic!(
            "{}: tk defines no {what} layout for this arch (matrix-core kernels need AMD or CUDA sm_80+)",
            self.caps.arch.target_name()
        )
    }

    /// An f32 accumulator register tile ([`FragRole::Accumulator`]), arch-resolved.
    ///
    /// # Panics
    /// Panics when the arch has no fragment layouts (see [`Kernel::frag`]).
    pub fn acc(&self, dims: (usize, usize), layout: TileLayout) -> RT<'_> {
        self.rt(dims, DType::Float32, layout, self.frag(FragRole::Accumulator))
    }
    /// An f32 **transposed** accumulator ([`FragRole::AccumulatorT`]) — the layout for
    /// an N-major store (e.g. the FA output `O[q,d]` from the `[d,q]` PV accumulator).
    ///
    /// # Panics
    /// Panics when the arch has no fragment layouts (see [`Kernel::frag`]).
    pub fn acc_t(&self, dims: (usize, usize), layout: TileLayout) -> RT<'_> {
        self.rt(dims, DType::Float32, layout, self.frag(FragRole::AccumulatorT))
    }
    /// A WMMA input-operand register tile ([`FragRole::Operand`]) of dtype `dt`.
    ///
    /// # Panics
    /// Panics when the arch has no fragment layouts (see [`Kernel::frag`]).
    pub fn operand(&self, dims: (usize, usize), dt: DType, layout: TileLayout) -> RT<'_> {
        self.rt(dims, dt, layout, self.frag(FragRole::Operand))
    }
    /// An f32 ortho register-vector — the softmax/reduce accumulator vectors, sized
    /// by the accumulator fragment's per-lane slots ([`FragRole::Accumulator`]).
    ///
    /// # Panics
    /// Panics when the arch has no fragment layouts (see [`Kernel::frag`]).
    pub fn acc_vec(&self, length: usize) -> RV<'_> {
        self.rv(length, DType::Float32, VecLayout::Ortho, self.frag(FragRole::Accumulator))
    }
    /// A shared (LDS) tile with the arch's canonical strip ([`crate::ArchCaps::shared_default`]).
    ///
    /// # Panics
    /// Panics when the arch has no fragment layouts (see [`Kernel::frag`]).
    pub fn shared(&self, dims: (usize, usize), dt: DType, layout: TileLayout) -> ST {
        self.st(dims, dt, layout, self.shared_strip(false))
    }
    /// A 2×-size double-buffered shared tile with the canonical strip.
    ///
    /// # Panics
    /// Panics when the arch has no fragment layouts (see [`Kernel::frag`]).
    pub fn shared_db(&self, dims: (usize, usize), dt: DType, layout: TileLayout) -> ST {
        self.st_db(dims, dt, layout, self.shared_strip(false))
    }
    /// A shared (LDS) tile with the XOR-swizzled strip ([`crate::ArchCaps::shared_swizzled`]),
    /// for kernels that swizzle to avoid LDS bank conflicts (the matmul A/B strips).
    ///
    /// # Panics
    /// Panics when the arch has no fragment layouts (see [`Kernel::frag`]).
    pub fn shared_sw(&self, dims: (usize, usize), dt: DType, layout: TileLayout) -> ST {
        self.st(dims, dt, layout, self.shared_strip(true))
    }
    fn shared_strip(&self, swizzled: bool) -> STBaseShape {
        let strip = if swizzled { self.caps.shared_swizzled() } else { self.caps.shared_default() };
        strip.unwrap_or_else(|| self.no_layout("shared strip"))
    }

    /// Build-time divisibility check with a uniform message (emits no UOps, so it is
    /// invisible to the graph fingerprint).
    ///
    /// # Panics
    /// Panics if `value` is not divisible by `by` (its whole job — the message
    /// names `what`). `by == 0` is a divide-by-zero.
    pub fn assert_divisible(value: usize, by: usize, what: &str) {
        assert_eq!(value % by, 0, "{what}: {value} must be a multiple of {by}");
    }
}
