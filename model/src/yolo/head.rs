//! Shared detection-head infrastructure: branches, anchor generation, box
//! decoding, and postprocessing.

use svod_ir::SInt;
use svod_tensor::Tensor;
use svod_tensor::nn::{Conv2d, Layer, Module};

use super::blocks::conv::{YoloConv, conv2d_bias};
use super::error::Result;
use crate::state::scoped;

/// The dtype every head decodes in, whatever the backbone computed in.
///
/// A box branch emits ltrb as *distances in stride units*, so [`dist2bbox`] sees
/// values up to ~80 — where f16's ulp is 0.0625, an order of magnitude past the
/// 0.05 px the parity tests allow, and the rounding would land *before* the
/// `× stride` multiply that would otherwise hide it. [`super::detect::Detect`]'s
/// branches run their convs at the backbone's dtype and have each final
/// `Conv2d` accumulate and emit at this width instead ([`BoxBranch`],
/// [`ClsBranch`]); the other heads still bring their features here at the
/// boundary ([`in_head_dtypes`]).
pub(crate) const HEAD_DTYPE: svod_dtype::DType = svod_dtype::DType::Float32;

/// Bring a backbone's feature maps back to [`HEAD_DTYPE`] at the head boundary,
/// for the heads whose branches have not been given a full-width accumulator.
///
/// A no-op when the features are f32 already: `UOp::cast` returns the same node.
pub(crate) fn in_head_dtype(feat: &Tensor) -> Tensor {
    feat.cast(HEAD_DTYPE)
}

/// [`in_head_dtype`] across a head's feature pyramid.
pub(crate) fn in_head_dtypes(feats: &[Tensor]) -> Vec<Tensor> {
    feats.iter().map(in_head_dtype).collect()
}

/// Generate anchor points and stride tensor from feature map sizes.
///
/// Returns `(anchors [2, A], strides [1, A])` as constant f32 tensors.
/// Anchor point `(x, y)` = `(col + 0.5, row + 0.5)` in grid coordinates.
pub(crate) fn make_anchors(feat_sizes: &[(usize, usize)], strides: &[usize]) -> Result<(Tensor, Tensor)> {
    let total: usize = feat_sizes.iter().map(|&(h, w)| h * w).sum();
    // Laid out as the `[2, A]` the decoders broadcast against: every x, then
    // every y — Ultralytics stacks `[A, 2]` and transposes to the same thing.
    let mut xs = Vec::with_capacity(2 * total);
    let mut ys = Vec::with_capacity(total);
    let mut stride_vec = Vec::with_capacity(total);
    for (&(h, w), &s) in feat_sizes.iter().zip(strides) {
        for y in 0..h {
            for x in 0..w {
                xs.push((x as f32) + 0.5);
                ys.push((y as f32) + 0.5);
                stride_vec.push(s as f32);
            }
        }
    }
    xs.append(&mut ys);
    Ok((Tensor::from_slice(&xs).try_reshape([2, total])?, Tensor::from_slice(&stride_vec).try_reshape([1, total])?))
}

/// Convert distance (lt, rb) predictions to xyxy boxes, then scale by strides.
///
/// `boxes [B, 4, A]`, `anchors [2, A]`, `strides [1, A]` → `[B, 4, A]`.
pub(crate) fn dist2bbox(boxes: &Tensor, anchors: &Tensor, strides: &Tensor, num_anchors: usize) -> Result<Tensor> {
    let parts = boxes.split(&[2, 2], 1)?;
    let lt = &parts[0];
    let rb = &parts[1];

    let anchors_3d = anchors.try_reshape([SInt::from(1isize), SInt::from(2isize), SInt::from(num_anchors as isize)])?;

    let x1y1 = anchors_3d.try_sub(lt)?;
    let x2y2 = anchors_3d.try_add(rb)?;
    let bbox = Tensor::cat(&[&x1y1, &x2y2], 1)?;

    let strides_3d = strides.try_reshape([SInt::from(1isize), SInt::from(1isize), SInt::from(num_anchors as isize)])?;
    Ok(bbox.try_mul(&strides_3d)?)
}

/// Box-regression branch: `Conv(k3) → Conv(k3) → Conv2d(k1, bias)`.
/// Outputs `4 * reg_max` channels at [`HEAD_DTYPE`] whatever the features came
/// in as: the first conv runs at the feature dtype, and from the second on the
/// accumulators emit full width, so nothing between it and the distances
/// rounds through f16 (the measured cost of doing so is 0.07 px; the checkpoint
/// weights are f16-exact, so half-width operands lose nothing).
///
/// State-dict keys: `0.{conv,bn}.*`, `1.{conv,bn}.*`, `2.weight`, `2.bias`.
#[derive(Clone, Module)]
pub struct BoxBranch {
    #[module(key = "0")]
    pub conv0: YoloConv,
    #[module(key = "1")]
    pub conv1: YoloConv,
    #[module(key = "2")]
    pub conv2: Conv2d,
}

impl BoxBranch {
    pub fn empty(in_ch: usize, hidden: usize, reg_max: usize) -> Self {
        Self {
            conv0: YoloConv::empty(in_ch, hidden, 3, 1, true),
            conv1: YoloConv::empty(hidden, hidden, 3, 1, true).with_acc_dtype(HEAD_DTYPE),
            conv2: conv2d_bias(hidden, 4 * reg_max, 1, 1).with_acc_dtype(HEAD_DTYPE),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = scoped("0", || self.conv0.forward(x))?;
        let x = scoped("1", || self.conv1.forward(&x))?;
        Ok(scoped("2", || self.conv2.forward(&x))?)
    }
}

/// Classification branch (non-legacy): `(DWConv→Conv) × 2 → Conv2d(bias)`.
/// The logits come out at [`HEAD_DTYPE`] the way the boxes do.
///
/// State-dict keys: `0.0.*`, `0.1.*`, `1.0.*`, `1.1.*`, `2.weight`, `2.bias`.
#[derive(Clone, Module)]
pub struct ClsBranch {
    #[module(key = "0.0")]
    pub dw0: YoloConv,
    #[module(key = "0.1")]
    pub conv0: YoloConv,
    #[module(key = "1.0")]
    pub dw1: YoloConv,
    #[module(key = "1.1")]
    pub conv1: YoloConv,
    #[module(key = "2")]
    pub conv2: Conv2d,
}

impl ClsBranch {
    pub fn empty(in_ch: usize, hidden: usize, nc: usize) -> Self {
        Self {
            dw0: YoloConv::empty_dw(in_ch, in_ch, 3, 1, true),
            conv0: YoloConv::empty(in_ch, hidden, 1, 1, true),
            dw1: YoloConv::empty_dw(hidden, hidden, 3, 1, true),
            conv1: YoloConv::empty(hidden, hidden, 1, 1, true),
            conv2: conv2d_bias(hidden, nc, 1, 1).with_acc_dtype(HEAD_DTYPE),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = scoped("0.0", || self.dw0.forward(x))?;
        let x = scoped("0.1", || self.conv0.forward(&x))?;
        let x = scoped("1.0", || self.dw1.forward(&x))?;
        // The logits conv ends its own kernel: fused into the 1x1 before it, the
        // two reduces nest and the pair runs without a tensor core.
        let x = scoped("1.1", || self.conv1.forward(&x))?.contiguous();
        Ok(scoped("2", || self.conv2.forward(&x))?)
    }
}

// ---------------------------------------------------------------------------
// Postprocess — top-k selection (outside JIT graph)
// ---------------------------------------------------------------------------

/// A single detection: `(x1, y1, x2, y2, confidence, class_id)`.
pub type Detection = [f32; 6];

/// Top-k postprocess on raw `[B, 4+nc, A]` f32 data.
///
/// Implements the two-stage selection from Ultralytics:
/// 1. Pre-filter anchors by max class score (top-k).
/// 2. Global top-k across all `(anchor, class)` pairs within those anchors.
pub fn postprocess_raw(data: &[f32], shape: &[usize], nc: usize, max_det: usize) -> Result<Vec<Vec<Detection>>> {
    let batch = shape[0];
    let out_ch = shape[1];
    let anchors = shape[2];

    let stride_b = out_ch * anchors;
    let stride_c = anchors;

    let k = max_det.min(anchors);
    let mut results = Vec::with_capacity(batch);

    for b in 0..batch {
        // Stage 1: top-k anchors by max class score.
        let mut max_scores: Vec<(f32, usize)> = (0..anchors)
            .map(|a| {
                let mut best = 0.0f32;
                for c in 0..nc {
                    let s = data[b * stride_b + (4 + c) * stride_c + a];
                    if s > best {
                        best = s;
                    }
                }
                (best, a)
            })
            .collect();
        max_scores.sort_by(|a, b2| b2.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        max_scores.truncate(k);

        // Stage 2: top-k (anchor, class) within the pre-filtered anchors.
        let mut candidates: Vec<(f32, usize, usize)> = Vec::with_capacity(k * nc);
        for &(_score, a) in &max_scores {
            for c in 0..nc {
                let s = data[b * stride_b + (4 + c) * stride_c + a];
                candidates.push((s, a, c));
            }
        }
        candidates.sort_by(|a, b2| b2.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(k);

        let dets: Vec<Detection> = candidates
            .iter()
            .map(|&(score, a, c)| {
                let base = b * stride_b;
                [
                    data[base + a],
                    data[base + stride_c + a],
                    data[base + 2 * stride_c + a],
                    data[base + 3 * stride_c + a],
                    score,
                    c as f32,
                ]
            })
            .collect();
        results.push(dets);
    }

    Ok(results)
}

/// Top-k postprocess on a realized `[B, 4+nc, A]` tensor. Returns one
/// `Vec<Detection>` per image, sorted by confidence descending.
pub fn postprocess(preds: &Tensor, nc: usize, max_det: usize) -> Result<Vec<Vec<Detection>>> {
    postprocess_raw(&preds.to_vec::<f32>()?, &preds.dims()?, nc, max_det)
}
