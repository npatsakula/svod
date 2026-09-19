use svod_dtype::DType;
use svod_tensor::nn::BatchNorm2d;

/// Default PyTorch BatchNorm epsilon, which the timm and WeSpeaker checkpoints
/// we target keep. Ultralytics is the exception: `initialize_weights` rewrites
/// every BatchNorm's eps to 1e-3, so YOLO passes its own.
pub const BN_EPS: f64 = 1e-5;

/// Identity-initialized inference batch norm over the channel axis, keyed with
/// PyTorch's `weight` / `bias` / `running_mean` / `running_var` names.
pub fn batchnorm2d(channels: usize) -> BatchNorm2d {
    batchnorm2d_with_eps(channels, BN_EPS)
}

/// [`batchnorm2d`] for checkpoints that do not use PyTorch's default epsilon.
pub fn batchnorm2d_with_eps(channels: usize, eps: f64) -> BatchNorm2d {
    BatchNorm2d::with_dims(channels, eps, DType::Float32)
}
