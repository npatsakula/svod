use svod_dtype::DType;
use svod_tensor::Tensor;

/// YOLO model scale variants. Each maps to Ultralytics'
/// `[depth_multiple, width_multiple, max_channels]` scaling table.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum YoloScale {
    Nano,
    Small,
    Medium,
    Large,
    XLarge,
}

impl YoloScale {
    /// Returns `(depth, width, max_channels)` from the yolo26 scaling table.
    pub fn scaling(self) -> (f64, f64, usize) {
        match self {
            YoloScale::Nano => (0.50, 0.25, 1024),
            YoloScale::Small => (0.50, 0.50, 1024),
            YoloScale::Medium => (0.50, 1.00, 512),
            YoloScale::Large => (1.00, 1.00, 512),
            YoloScale::XLarge => (1.00, 1.50, 512),
        }
    }

    pub fn depth(self) -> f64 {
        self.scaling().0
    }

    pub fn width(self) -> f64 {
        self.scaling().1
    }

    pub fn max_channels(self) -> usize {
        self.scaling().2
    }

    /// Ultralytics' `parse_model` rewrites every `C3k2`'s `c3k` argument to
    /// `true` for the M/L/X scales, so the shallow blocks the YAML marks
    /// `False` still nest a `C3k` there instead of a plain bottleneck.
    pub fn forces_c3k(self) -> bool {
        matches!(self, YoloScale::Medium | YoloScale::Large | YoloScale::XLarge)
    }
}

/// `ceil(value / divisor) * divisor` — Ultralytics `make_divisible`.
pub fn make_divisible(value: usize, divisor: usize) -> usize {
    ((value as f64 / divisor as f64).ceil() as usize) * divisor
}

/// Scale a YAML channel spec through the width/max_channels table entry.
pub fn scale_channels(c: usize, scale: YoloScale) -> usize {
    let scaled = (std::cmp::min(c, scale.max_channels()) as f64 * scale.width()) as usize;
    make_divisible(scaled, 8)
}

/// `max(round(n * depth), 1)` — Ultralytics depth gain.
pub fn make_depth(n: usize, scale: YoloScale) -> usize {
    let scaled = (n as f64 * scale.depth()).round() as usize;
    scaled.max(1)
}

#[derive(Clone, Debug)]
pub struct YoloConfig {
    pub scale: YoloScale,
    pub nc: usize,
    pub reg_max: usize,
    pub max_batch_size: usize,
    /// Dtype the backbone and neck compute in; see [`Self::with_compute_dtype`].
    /// The heads always decode in f32 regardless (`head::in_head_dtype`).
    pub compute_dtype: DType,
}

impl YoloConfig {
    pub fn new(scale: YoloScale, nc: usize) -> Self {
        Self { scale, nc, reg_max: 1, max_batch_size: 1, compute_dtype: DType::Float32 }
    }

    pub fn with_max_batch_size(mut self, max_batch_size: usize) -> Self {
        self.max_batch_size = max_batch_size;
        self
    }

    /// Run the backbone and neck at `compute_dtype`, accumulating reductions in
    /// f32 (`Tensor::sum_acc_dtype` promotes f16/bf16 sums on its own, so a conv
    /// is `CAST_f16(REDUCE_f32(Add, CAST_f32(MUL(f16, f16))))` — the shape the
    /// tensor-core matcher wants).
    ///
    /// Weights must move with the activations: `least_upper_dtype(f16, f32)` is
    /// f32, so a checkpoint left at f32 would promote the very first conv back
    /// and silently revert the whole graph — no error, no speedup. The loaders
    /// therefore cast the state dict to match (`loader::cast_weights`).
    pub fn with_compute_dtype(mut self, compute_dtype: DType) -> Self {
        self.compute_dtype = compute_dtype;
        self
    }

    /// Cast an input image into the compute dtype. A no-op when they already
    /// match (`UOp::cast` returns the same node), so the f32 default adds
    /// nothing to the graph.
    pub fn cast_input(&self, images: &Tensor) -> Tensor {
        images.cast(self.compute_dtype.clone())
    }
}

// ---------------------------------------------------------------------------
// Detect head strides
// ---------------------------------------------------------------------------

/// Strides for the three detection levels (P3/8, P4/16, P5/32).
pub const DETECT_STRIDES: [usize; 3] = [8, 16, 32];

/// Strides for P2 variant (P2/4, P3/8, P4/16, P5/32).
pub const P2_STRIDES: [usize; 4] = [4, 8, 16, 32];

/// Strides for P6 variant (P3/8, P4/16, P5/32, P6/64).
pub const P6_STRIDES: [usize; 4] = [8, 16, 32, 64];
