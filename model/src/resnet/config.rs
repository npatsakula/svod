use crate::blocks::BlockKind;

/// Canonical ResNet depths. The depth selects both the block type and the
/// per-stage block count schedule used by the original paper.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ResNetDepth {
    R18,
    R34,
    R50,
    R101,
    R152,
}

impl ResNetDepth {
    /// Per-stage block count `[stage1, stage2, stage3, stage4]`.
    pub fn layers(self) -> [usize; 4] {
        match self {
            ResNetDepth::R18 => [2, 2, 2, 2],
            ResNetDepth::R34 => [3, 4, 6, 3],
            ResNetDepth::R50 => [3, 4, 6, 3],
            ResNetDepth::R101 => [3, 4, 23, 3],
            ResNetDepth::R152 => [3, 8, 36, 3],
        }
    }

    pub fn block(self) -> BlockKind {
        match self {
            ResNetDepth::R18 | ResNetDepth::R34 => BlockKind::Basic,
            ResNetDepth::R50 | ResNetDepth::R101 | ResNetDepth::R152 => BlockKind::Bottleneck,
        }
    }

    pub fn expansion(self) -> usize {
        self.block().expansion()
    }
}

/// Which forward tail the model executes; fixed at construction time
/// ([`ResNetConfig::new`], [`ResNet::from_hub`](super::ResNet::from_hub)).
#[derive(Copy, Clone, Debug)]
pub enum OutputMode {
    /// Add the FC head; forward returns logits `[B, num_classes]`. The FC
    /// `weight` / `bias` tensors are loaded from `fc.weight` / `fc.bias`.
    Classification { num_classes: usize },
    /// Stop after stage 4; forward returns the final feature map
    /// `[B, 512 * expansion, H/32, W/32]`. The FC weights are not loaded.
    Features,
}

#[derive(Clone, Debug)]
pub struct ResNetConfig {
    pub depth: ResNetDepth,
    pub output: OutputMode,
    /// Upper bound on the symbolic `b` variable exposed by the JIT wrapper.
    /// The prepared plan's image buffer is allocated to `max_batch_size`; the
    /// per-call `execute_bound(actual)` shrinks the batch dim to the live
    /// size.
    pub max_batch_size: usize,
}

impl ResNetConfig {
    pub fn new(depth: ResNetDepth, output: OutputMode) -> Self {
        Self { depth, output, max_batch_size: 1 }
    }

    pub fn with_max_batch_size(mut self, max_batch_size: usize) -> Self {
        self.max_batch_size = max_batch_size;
        self
    }
}
