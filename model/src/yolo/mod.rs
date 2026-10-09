//! YOLO v26 — multi-task vision models.
//!
//! Each task is a separate type sharing common backbone/neck infrastructure:
//!
//! - [`Yolo26Detect`] — object detection (end2end, reg_max=1, NMS-free)
//! - [`Yolo26DetectP2`] — detection with extra P2/4 scale
//! - [`Yolo26DetectP6`] — detection with extra P6/64 scale
//! - [`Yolo26Classify`] — image classification
//! - [`Yolo26Segment`] — instance segmentation
//! - [`Yolo26Obb`] — oriented bounding box detection
//! - [`Yolo26Pose`] — pose estimation
//! - [`Yolo26Depth`] — monocular depth estimation
//! - [`Yolo26SemSeg`] — semantic segmentation
//!
//! ## Recipe
//!
//! ```no_run
//! use svod_model::yolo::{Yolo26Detect, YoloConfig, YoloScale, Yolo26DetectJit, postprocess_raw};
//! use svod_model::jit::InputSpec;
//!
//! let model = Yolo26Detect::from_hub("ultralytics/yolo26n",
//!     YoloConfig::new(YoloScale::Nano, 80))?;
//!
//! let mut jit = Yolo26DetectJit::new(model);
//! jit.prepare(InputSpec::f32(&[1, 3, 640, 640]))?;
//! // copy NCHW image into jit.images_mut()?, then:
//! jit.execute_bound(1)?;
//! let shape = jit.predictions_shape()?;
//! let preds = jit.predictions_to_vec::<f32>()?;
//! let detections = postprocess_raw(&preds, &shape, 80, 300)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod backbone;
mod blocks;
mod classify;
mod config;
mod depth;
mod detect;
mod error;
pub(crate) mod head;
mod jit;
pub(crate) mod loader;
mod neck;
mod obb;
mod pose;
mod segment;
mod semseg;

pub use backbone::{YoloBackbone, YoloBackboneCls, YoloBackboneP6, scaled_channels};
pub use blocks::{
    Attention, C2PSA, C2f, C3k, C3k2, C3k2Inner, PSABlock, Sppf, YoloBottleneck, YoloConv, conv2d_bias, deconv2d_2x,
};
pub use classify::{ClassifyHead, Yolo26Classify};
pub use config::{YoloConfig, YoloScale, make_depth, make_divisible, scale_channels};
pub use depth::{DepthHead, Yolo26Depth};
pub use detect::{Detect, Yolo26Detect, Yolo26DetectP2, Yolo26DetectP6};
pub use error::{Error, Result};
pub use head::{BoxBranch, ClsBranch, Detection, postprocess, postprocess_raw};
pub use jit::{Yolo26ClassifyJit, Yolo26DepthJit, Yolo26DetectJit, Yolo26ObbJit, Yolo26PoseJit, Yolo26SemSegJit};
pub use neck::{YoloNeck, YoloNeckP2, YoloNeckP6};
pub use obb::{AngleBranch, OBB26, Yolo26Obb};
pub use pose::{Pose26, PoseFeatBranch, Yolo26Pose};
pub use segment::{MaskBranch, Proto26, Segment26, Yolo26Segment};
pub use semseg::{SemSegClassifier, Yolo26SemSeg};
