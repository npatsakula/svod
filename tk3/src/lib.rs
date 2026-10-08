//! Tiled Language v3. Design: `tk3_design.md` at the repository root.
//!
//! A kernel is a *structured* tile program (statements in program order, effects
//! declared, sync derived), layouts are F2 linear maps carried as data, vendor
//! instructions are atoms that bring their own layouts, and the lowering emits a
//! pre-linearized instruction list so no toposort ever decides the order.

pub mod atoms;
pub mod build;
pub mod index;
pub mod interp;
pub mod ir;
pub mod kernels;
pub mod launch;
pub mod layout;
pub mod layouts;
pub mod lower;
pub mod ops;
pub mod schedule;
pub mod tune;

#[cfg(test)]
mod test;
