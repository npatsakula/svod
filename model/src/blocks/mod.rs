//! Shared ResNet-family building blocks.
//!
//! Originally part of `model::resnet`, factored out so sibling models
//! (`model::wespeaker`) can compose the same primitives without depending on
//! resnet's wrapping types. All conv layers are bias-less; biases live only in
//! the BN affine parameters. The blocks hold [`svod_tensor::nn`] layers, so a
//! PyTorch state dict loads key for key — `running_var` and all.

mod basic_block;
mod batchnorm;
mod bottleneck;
mod conv;
pub mod error;
pub mod remap;
mod stage;

pub use basic_block::{BasicBlock, BlockKind};
pub use batchnorm::{BN_EPS, batchnorm2d, batchnorm2d_with_eps};
pub use bottleneck::Bottleneck;
pub use conv::{conv2d, conv2d_grouped};
pub use error::{Error, Result};
pub use stage::{Block, ResidualStage};
