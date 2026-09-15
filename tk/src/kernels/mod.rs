//! Kernel implementations authored on top of the tile DSL: the tiled [`gemm`]
//! core (the square bf16→f32 `matmul` and the NT linear-layer `gemm_nt`) and the
//! [`fa`] flash-attention forward (single-warp, multi-wave, double-buffered).
//!
//! The DSL tooling lives in the crate-root modules ([`kernel`](crate::kernel),
//! [`group`](crate::group), [`tile`](crate::tile), …); this module is the place
//! for concrete kernels built from those primitives.

pub mod fa;
pub mod gemm;
pub mod kmeans;
pub mod knn;
pub mod norm;
pub mod sq_attention;
