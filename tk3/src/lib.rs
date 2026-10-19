//! Tiled Language v3; the guide is the website's Tile Kernels pages
//! (<https://svod.vpermilp.online/docs/tile-kernels/overview>).
//!
//! A kernel is a *structured* tile program (statements in program order, effects
//! declared, sync derived), layouts are F2 linear maps carried as data, vendor
//! instructions are atoms that bring their own layouts, and the lowering emits a
//! pre-linearized instruction list so no toposort ever decides the order.
//!
//! No constant stands for a device. Each one's doc opens with what it is
//! instead: a *model* constant shapes the cost model the planner ranks with
//! and holds for every device or is wrong; a *search bound* says how far a
//! search goes; a *measurement* constant says how a probe or the tuner
//! times; a *numeric* constant is a floating-point margin. A constant that
//! names a fact of the hardware, of an instruction encoding, of a file
//! format or of a published formula carries no kind.

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
pub mod probe;
pub mod schedule;
pub mod tune;

#[cfg(test)]
mod test;
