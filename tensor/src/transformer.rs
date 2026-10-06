//! Transformer building blocks: embedding, attention, rotary position embeddings.

use crate::Tensor;
use bon::bon;
use snafu::{OptionExt, ensure};
use std::borrow::Cow;
use svod_dtype::DType;
use svod_ir::{ConstValue, SInt};

use crate::error::{
    DivisibilitySnafu, FloatDTypeRequiredSnafu, NdimExactSnafu, NdimMinimumSnafu, ParamRangeSnafu,
    SymbolicShapeUnsupportedSnafu,
};

type Result<T> = crate::Result<T>;

/// Denominator floor for [`Tensor::masked_mean`], so an all-masked row divides
/// by ~0 instead of producing `NaN`. Matches sentence-transformers' `clamp(…,
/// min=1e-9)`.
const MASKED_MEAN_EPS: f64 = 1e-9;

impl Tensor {
    /// Embedding lookup: `self` is the weight table `[vocab_size, embed_dim]`.
    /// Returns `self[indices]` with shape `[*indices.shape, embed_dim]`.
    ///
    /// Mirrors tinygrad's `_embedding_fwd`: a one-hot mask
    /// (`indices.unsqueeze(-1) == arange(vocab)`) selects rows via
    /// `where(weight, 0).sum(vocab)`. It operates on the index's natural shape
    /// — no flatten-to-`[-1]` — so a symbolic dim on `indices` (e.g. a JIT
    /// batch bound to a `Variable`) passes through. Only the vocab axis
    /// (weight dim 0) must be concrete.
    #[track_caller]
    pub fn embedding(&self, indices: &Tensor) -> Result<Tensor> {
        origin_call!("embedding");
        let weight_shape = self.shape()?;
        let vocab_size =
            weight_shape[0].as_const().context(SymbolicShapeUnsupportedSnafu { operation: "embedding" })?;

        // one-hot mask: [*idx_shape, vocab] bool — True where the vocab row
        // matches the index value.
        let vocab_arange = Tensor::arange(0, Some(vocab_size as i64), None)?.cast(indices.uop().dtype());
        let mask = indices.try_unsqueeze(-1)?.try_eq(&vocab_arange)?;

        // Reshape weight to [1... (per idx dim), vocab, embed] so its trailing
        // (vocab, embed) rows broadcast under the mask in the where_. Binary
        // ops broadcast aligning trailing dims, so no explicit broadcast_to.
        let idx_ndim = indices.shape()?.len();
        let mut leading_ones: Vec<isize> = (0..idx_ndim).map(|_| 1).collect();
        leading_ones.push(vocab_size as isize);
        leading_ones.push(-1); // embed_dim inferred from weight.
        let weight_bc = self.try_reshape(&leading_ones)?;

        // Select + collapse the vocab axis. mask.unsqueeze(-1) broadcasts over
        // embed; summing the vocab axis (now -2) leaves [*idx_shape, embed].
        weight_bc
            .where_(
                &mask.try_unsqueeze(-1)?,
                Tensor::new(self.uop().const_like(ConstValue::zero(self.uop().dtype().base()))),
            )?
            .sum_with()
            .axes(-2isize)
            .dtype(self.uop().dtype())
            .call()
    }

    /// Apply rotary position embedding rotation.
    /// `self`: `[..., rot_dim]` tensor to rotate.
    /// `cos`, `sin`: broadcastable to `self`'s shape `[..., rot_dim/2]`.
    /// If interleaved: pairs are (even, odd) indices.
    /// If not interleaved: pairs are (first_half, second_half).
    #[track_caller]
    pub fn apply_rotary_emb(&self, cos: &Tensor, sin: &Tensor, interleaved: bool) -> Result<Tensor> {
        origin_call!("apply_rotary_emb");
        let shape = self.shape()?;
        let last_dim = shape
            .last()
            .context(NdimMinimumSnafu { op: "apply_rotary_emb", min: 1usize, actual: 0usize })?
            .as_const()
            .context(SymbolicShapeUnsupportedSnafu { operation: "apply_rotary_emb" })?;
        let half = last_dim / 2;

        let (x1, x2) = if interleaved {
            let mut rs: Vec<isize> = shape
                .iter()
                .take(shape.len() - 1)
                .map(|d| {
                    Ok(d.as_const().context(SymbolicShapeUnsupportedSnafu { operation: "apply_rotary_emb" })? as isize)
                })
                .collect::<Result<_>>()?;
            rs.push(half as isize);
            rs.push(2);
            let r = self.try_reshape(&rs)?;
            let p = r.split(&[1, 1], -1)?;
            (p[0].try_squeeze(Some(-1))?, p[1].try_squeeze(Some(-1))?)
        } else {
            let p = self.split(&[half, half], -1)?;
            (p[0].clone(), p[1].clone())
        };

        let real = x1.try_mul(cos)?.try_sub(&x2.try_mul(sin)?)?;
        let imag = x1.try_mul(sin)?.try_add(&x2.try_mul(cos)?)?;

        if interleaved {
            let stacked = Tensor::stack(&[&real, &imag], -1)?;
            let mut fs: Vec<isize> = shape
                .iter()
                .map(|d| {
                    Ok(d.as_const().context(SymbolicShapeUnsupportedSnafu { operation: "apply_rotary_emb" })? as isize)
                })
                .collect::<Result<_>>()?;
            // Last dim already correct from original shape
            let _ = fs.last_mut().map(|d| *d = last_dim as isize);
            stacked.try_reshape(&fs)
        } else {
            Tensor::cat(&[&real, &imag], -1)
        }
    }

    /// Rotary `(cos, sin)` tables for [`apply_rotary_emb`](Self::apply_rotary_emb),
    /// each shaped `[1, 1, seq_len, head_dim / 2]` — the half-width the
    /// non-interleaved (GPT-NeoX) rotation consumes, broadcasting over q/k of
    /// shape `[B, H, seq_len, head_dim]`.
    ///
    /// `inv_freq[i] = theta ^ (-2i / head_dim)`, `angle[s, i] = s * inv_freq[i]`:
    /// two `arange`s and an outer product, all in the graph — no host loop over
    /// `seq_len × head_dim`. Angles are built in float32 (bf16/f16 mantissas are
    /// too short for the `position × frequency` products) and cast to `dtype`
    /// last.
    #[track_caller]
    pub fn rope_table(theta: f64, seq_len: usize, head_dim: usize, dtype: DType) -> Result<(Tensor, Tensor)> {
        origin_call!("rope_table");
        ensure!(
            head_dim >= 2 && head_dim.is_multiple_of(2),
            ParamRangeSnafu {
                op: "rope_table",
                param: "head_dim",
                value: head_dim.to_string(),
                constraint: "an even number ≥ 2"
            }
        );
        ensure!(
            seq_len > 0,
            ParamRangeSnafu { op: "rope_table", param: "seq_len", value: seq_len.to_string(), constraint: "≥ 1" }
        );

        // angles: [seq_len, 1] × [head_dim/2] → [seq_len, head_dim/2].
        let angles = Tensor::arange_f64(0.0, seq_len as f64, 1.0, DType::Float32)?
            .try_unsqueeze(-1)?
            .try_mul(&Self::rope_inv_freq(theta, head_dim)?)?;
        let table = |t: Tensor| -> Result<Tensor> { Ok(t.try_unsqueeze(0)?.try_unsqueeze(0)?.cast(dtype.clone())) };
        Ok((table(angles.cos()?)?, table(angles.sin()?)?))
    }

    /// The per-pair angular frequencies of [`Self::rope_table`], `[head_dim/2]`
    /// f32: `theta^(-2i/head_dim)`. A position's table row is
    /// `cos/sin(position · inv_freq)` in f32, so rotating a token at an
    /// arbitrary position from these matches the table bit for bit.
    #[track_caller]
    pub fn rope_inv_freq(theta: f64, head_dim: usize) -> Result<Tensor> {
        origin_call!("rope_inv_freq");
        let exponent =
            Tensor::arange_f64(0.0, (head_dim / 2) as f64, 1.0, DType::Float32)?.try_mul(-2.0 / head_dim as f64)?;
        Tensor::const_(theta, DType::Float32).try_pow(&exponent)
    }

    /// `[B, L, H * D] → [B, H, L, D]`: split the feature axis into `n_heads`
    /// heads, then move the head axis ahead of the sequence axis. `B` and `L`
    /// may be symbolic; only the feature axis must be concrete.
    #[track_caller]
    pub fn split_heads(&self, n_heads: usize) -> Result<Tensor> {
        origin_call!("split_heads");
        let shape = self.shape()?;
        ensure!(shape.len() == 3, NdimExactSnafu { op: "split_heads", expected: 3usize, actual: shape.len() });
        ensure!(
            n_heads > 0,
            ParamRangeSnafu { op: "split_heads", param: "n_heads", value: n_heads.to_string(), constraint: "≥ 1" }
        );
        let features = self.dim_const(2)?;
        ensure!(
            features.is_multiple_of(n_heads),
            DivisibilitySnafu {
                op: "split_heads",
                lhs_name: "features",
                lhs: features,
                rhs_name: "n_heads",
                rhs: n_heads
            }
        );
        self.try_reshape([shape[0].clone(), shape[1].clone(), SInt::Const(n_heads), SInt::Const(features / n_heads)])?
            .try_permute(&[0, 2, 1, 3])
    }

    /// `[B, H, L, D] → [B, L, H * D]`, the inverse of
    /// [`split_heads`](Self::split_heads). `B` and `L` may be symbolic.
    #[track_caller]
    pub fn merge_heads(&self) -> Result<Tensor> {
        origin_call!("merge_heads");
        let shape = self.shape()?;
        ensure!(shape.len() == 4, NdimExactSnafu { op: "merge_heads", expected: 4usize, actual: shape.len() });
        let features = self.dim_const(1)? * self.dim_const(3)?;
        self.try_permute(&[0, 2, 1, 3])?.try_reshape([shape[0].clone(), shape[2].clone(), SInt::Const(features)])
    }

    /// Additive causal mask `[1, 1, len, len]`: `0` on and below the diagonal,
    /// `-inf` strictly above it. Added to attention scores (or handed to
    /// [`scaled_dot_product_attention`](Self::scaled_dot_product_attention) as
    /// the *float* `attn_mask`) it forbids attending to future positions.
    #[track_caller]
    pub fn causal_mask(len: usize, dtype: DType) -> Result<Tensor> {
        origin_call!("causal_mask");
        ensure!(dtype.is_float(), FloatDTypeRequiredSnafu { op: "causal_mask", arg: "dtype", dtype: dtype.clone() });
        let q_idx = Tensor::arange(0, Some(len as i64), None)?.try_unsqueeze(-1)?; // [L, 1]
        let k_idx = Tensor::arange(0, Some(len as i64), None)?; // [L]
        let upper = q_idx.try_lt(&k_idx)?; // True above the diagonal
        let neg_inf = Tensor::const_(ConstValue::Float(f64::NEG_INFINITY), dtype.clone());
        let zero = Tensor::const_(ConstValue::zero(dtype.base()), dtype);
        neg_inf.where_(&upper, zero)?.try_unsqueeze(0)?.try_unsqueeze(0)
    }

    /// Boolean validity mask `[B, max_len]` from per-row lengths `[B]` (any
    /// integer dtype): **`true` = valid**, i.e. `position < lengths[b]`.
    ///
    /// This is the polarity
    /// [`key_padding_mask`](Self::scaled_dot_product_attention) wants; invert it
    /// (`logical_not`) for `attn_mask`, which masks OUT on `true`.
    #[track_caller]
    pub fn sequence_mask(lengths: &Tensor, max_len: usize) -> Result<Tensor> {
        origin_call!("sequence_mask");
        let ndim = lengths.ndim()?;
        ensure!(ndim == 1, NdimExactSnafu { op: "sequence_mask", expected: 1usize, actual: ndim });
        let positions = Tensor::arange(0, Some(max_len as i64), None)?.cast(lengths.dtype());
        positions.try_lt(&lengths.try_unsqueeze(-1)?)
    }

    /// Repeat every slice along `dim` `repeats` times *consecutively*
    /// (`[a, b] → [a, a, b, b]`), like `torch.repeat_interleave`.
    ///
    /// reshape → expand → reshape, so it stays a view op and every other axis —
    /// symbolic ones included — passes through untouched. `dim` itself may be
    /// symbolic too.
    #[track_caller]
    pub fn repeat_interleave(&self, repeats: usize, dim: isize) -> Result<Tensor> {
        origin_call!("repeat_interleave");
        ensure!(
            repeats > 0,
            ParamRangeSnafu {
                op: "repeat_interleave",
                param: "repeats",
                value: repeats.to_string(),
                constraint: "≥ 1"
            }
        );
        let shape = self.shape()?;
        let axis = Self::normalize_axis(dim, shape.len())?;
        if repeats == 1 {
            return Ok(self.clone());
        }

        let dims: Vec<SInt> = shape.iter().cloned().collect();
        let mut split = dims.clone();
        split.insert(axis + 1, SInt::Const(1));
        let mut expanded = split.clone();
        expanded[axis + 1] = SInt::Const(repeats);
        let mut merged = dims;
        merged[axis] = &shape[axis] * &SInt::Const(repeats);
        self.try_reshape(split)?.try_expand(expanded)?.try_reshape(merged)
    }

    /// Mean over `axis` counting only the positions `mask` marks valid
    /// (`true`/non-zero), dropping `axis` from the result.
    ///
    /// `mask` covers `self`'s *leading* axes — `[B, L]` against a `[B, L, D]`
    /// input — and is unsqueezed over the trailing ones. The denominator is
    /// floored at `MASKED_MEAN_EPS`, so an all-masked row yields `0` rather
    /// than `NaN`. Sums promote to the accumulator dtype (float32 for
    /// f16/bf16 inputs), which is also the result dtype.
    #[track_caller]
    pub fn masked_mean(&self, mask: &Tensor, axis: isize) -> Result<Tensor> {
        origin_call!("masked_mean");
        let dtype = self.dtype();
        ensure!(dtype.is_float(), FloatDTypeRequiredSnafu { op: "masked_mean", arg: "self", dtype: dtype.clone() });

        let (ndim, mask_ndim) = (self.ndim()?, mask.ndim()?);
        let mut weights = mask.cast(dtype);
        for _ in mask_ndim..ndim {
            weights = weights.try_unsqueeze(-1)?;
        }

        let sum = self.try_mul(&weights)?.sum_with().axes(axis).keepdim(true).call()?;
        let count = weights.sum_with().axes(axis).keepdim(true).call()?.maximum(MASKED_MEAN_EPS)?;
        sum.try_div(&count)?.try_squeeze(Some(axis))
    }

    /// Take position `index` along `axis` and drop that axis
    /// (`[B, L, D] → [B, D]`) — CLS pooling is `take_index(1, 0)`, last-token
    /// pooling `take_index(1, -1)`.
    ///
    /// A negative `index` counts from the end and therefore needs a concrete
    /// axis size; a non-negative one works on a symbolic axis as well.
    #[track_caller]
    pub fn take_index(&self, axis: isize, index: isize) -> Result<Tensor> {
        origin_call!("take_index");
        let shape = self.shape()?;
        let ax = Self::normalize_axis(axis, shape.len())?;
        let position = if index < 0 {
            let size = shape[ax]
                .as_const()
                .context(SymbolicShapeUnsupportedSnafu { operation: "take_index with a negative index" })?;
            let resolved = size as isize + index;
            ensure!(
                resolved >= 0,
                ParamRangeSnafu {
                    op: "take_index",
                    param: "index",
                    value: index.to_string(),
                    constraint: "within the axis"
                }
            );
            resolved as usize
        } else {
            index as usize
        };

        let mut ranges: Vec<Option<(SInt, SInt)>> = vec![None; shape.len()];
        ranges[ax] = Some((SInt::Const(position), SInt::Const(position + 1)));
        self.try_shrink(ranges)?.try_squeeze(Some(ax as isize))
    }
}

#[bon]
impl Tensor {
    /// Scaled dot-product attention.
    /// `self` (Q): `[B, H, Sq, D]`, `key` (K): `[B, Hkv, Sk, D]`, `value` (V): `[B, Hkv, Sk, Dv]`.
    /// Returns `[B, H, Sq, Dv]`.
    ///
    /// # Mask polarity — read this before passing a mask
    ///
    /// The two mask arguments use **opposite** conventions, mirroring PyTorch:
    ///
    /// - `attn_mask`, when boolean: **`true` = masked OUT**, `false` = attend.
    ///   When floating-point it is an *additive* bias on the scores (`-inf`
    ///   forbids), added after the boolean masks.
    /// - `key_padding_mask`, `[B, Sk]`: **`true` = valid**, `false` = padding —
    ///   the polarity [`Tensor::sequence_mask`] produces. It is inverted
    ///   internally, broadcast to `[B, 1, 1, Sk]`, and intersected with
    ///   `attn_mask`, the causal mask and the window band.
    ///
    /// `window = Some((left, right))` restricts each query `q` to keys in
    /// `[q - left, q + right]` (sliding-window / banded attention, as in
    /// ModernBERT's local layers). `None` = full (global) attention.
    ///
    /// `enable_gqa` allows K/V to carry fewer heads than Q (grouped-query /
    /// multi-query attention): each KV head is repeated `H / Hkv` times with
    /// [`repeat_interleave`](Tensor::repeat_interleave), which requires
    /// `H % Hkv == 0`. Without it, Q and K/V must have matching head counts.
    #[builder]
    #[track_caller]
    pub fn scaled_dot_product_attention(
        &self,
        key: &Tensor,
        value: &Tensor,
        attn_mask: Option<&Tensor>,
        key_padding_mask: Option<&Tensor>,
        scale: Option<f64>,
        #[builder(default)] is_causal: bool,
        #[builder(default)] enable_gqa: bool,
        window: Option<(usize, usize)>,
        softcap: Option<f64>,
    ) -> Result<Tensor> {
        origin_call!("scaled_dot_product_attention");
        let q_dtype = self.uop().dtype();
        ensure!(
            q_dtype.is_float(),
            FloatDTypeRequiredSnafu { op: "scaled_dot_product_attention", arg: "query", dtype: q_dtype.clone() }
        );
        let k_dtype = key.uop().dtype();
        ensure!(
            k_dtype.is_float(),
            FloatDTypeRequiredSnafu { op: "scaled_dot_product_attention", arg: "key", dtype: k_dtype.clone() }
        );
        let v_dtype = value.uop().dtype();
        ensure!(
            v_dtype.is_float(),
            FloatDTypeRequiredSnafu { op: "scaled_dot_product_attention", arg: "value", dtype: v_dtype.clone() }
        );

        // Grouped-query attention: K/V carry `Hkv ≤ H` heads, each serving
        // `H / Hkv` query heads. Repeating them interleaved lines the head axes
        // up so the rest of the kernel is unchanged.
        let (gqa_key, gqa_value) = if enable_gqa {
            let (q_heads, kv_heads) = (self.dim_const(-3)?, key.dim_const(-3)?);
            ensure!(
                kv_heads > 0 && q_heads.is_multiple_of(kv_heads),
                DivisibilitySnafu {
                    op: "scaled_dot_product_attention",
                    lhs_name: "query heads",
                    lhs: q_heads,
                    rhs_name: "key/value heads",
                    rhs: kv_heads
                }
            );
            let repeats = q_heads / kv_heads;
            (Cow::Owned(key.repeat_interleave(repeats, -3)?), Cow::Owned(value.repeat_interleave(repeats, -3)?))
        } else {
            (Cow::Borrowed(key), Cow::Borrowed(value))
        };
        let (key, value): (&Tensor, &Tensor) = (&gqa_key, &gqa_value);

        let q_shape = self.shape()?;
        let k_shape = key.shape()?;
        let head_dim = q_shape[q_shape.len() - 1]
            .as_const()
            .context(SymbolicShapeUnsupportedSnafu { operation: "scaled_dot_product_attention" })?;
        let scale_val = scale.unwrap_or(1.0 / (head_dim as f64).sqrt());

        // Scores are formed, masked and softmaxed in the sum accumulator dtype
        // (float32 for fp16/bf16/fp8): the fallback softmax otherwise runs in
        // float16, where the additive mask constant is only -65504.
        let scores_dtype = Tensor::sum_acc_dtype(&q_dtype);

        // Q @ K^T
        let kt = key.try_transpose(-1, -2)?;
        let mut scores = self.matmul_with().other(&kt).dtype(scores_dtype.clone()).call()?;

        // Scale
        let scale_t = Tensor::const_(scale_val, scores_dtype.clone());
        scores = scores.try_mul(&scale_t)?;

        // Softcap the raw scaled scores, before any mask: capping afterwards
        // squashes a masked `dtype::min` to `-cap`, leaving it softmax weight.
        if let Some(cap) = softcap
            && cap > 0.0
        {
            let cap_t = Tensor::const_(cap, scores_dtype.clone());
            scores = scores.try_div(&cap_t)?.tanh()?.try_mul(&cap_t)?;
        }

        // Build a boolean "keep" mask that ANDs together the causal constraint,
        // the optional sliding-window band, and the user-supplied `attn_mask`.
        // True = attend, False = masked out. The mask is applied additively
        // (mask_out → -large) before softmax, and the weights are also zeroed
        // post-softmax to guarantee exact-zero out-of-band columns even when a
        // full row is masked (softmax-of-all-equal → uniform, not zero).
        let q_len = q_shape[q_shape.len() - 2]
            .as_const()
            .context(SymbolicShapeUnsupportedSnafu { operation: "scaled_dot_product_attention" })?;
        let k_len = k_shape[k_shape.len() - 2]
            .as_const()
            .context(SymbolicShapeUnsupportedSnafu { operation: "scaled_dot_product_attention" })?;

        let mut keep_mask: Option<Tensor> = None;

        // Causal: keep k ≤ q.
        if is_causal {
            let q_idx = Tensor::arange(0, Some(q_len as i64), None)?.try_unsqueeze(-1)?; // (Q, 1)
            let k_idx = Tensor::arange(0, Some(k_len as i64), None)?; // (K,)
            let causal = k_idx.try_le(&q_idx)?; // k <= q
            keep_mask = Some(causal);
        }

        // Sliding-window band: keep q - left ≤ k ≤ q + right.
        if let Some((left, right)) = window {
            let q_idx = Tensor::arange(0, Some(q_len as i64), None)?.try_unsqueeze(-1)?; // (Q, 1)
            let k_idx = Tensor::arange(0, Some(k_len as i64), None)?; // (K,)
            let lo = Tensor::const_(ConstValue::Int(left as i64), DType::Int32);
            let hi = Tensor::const_(ConstValue::Int(right as i64), DType::Int32);
            // q - left <= k  AND  k <= q + right
            let lower = q_idx.try_sub(&lo)?.try_le(&k_idx)?;
            let upper = k_idx.try_le(&q_idx.try_add(&hi)?)?;
            let band = lower.try_bitand(&upper)?;
            keep_mask = Some(match keep_mask {
                Some(prev) => prev.try_bitand(&band)?,
                None => band,
            });
        }

        // User-supplied attention mask. Bool: True = mask OUT (False = keep) —
        // invert before ANDing. Float: additive, applied separately below.
        let mut float_additive_mask: Option<&Tensor> = None;
        if let Some(mask) = attn_mask {
            if mask.uop().dtype() == DType::Bool {
                let keep = mask.logical_not()?; // True = keep
                keep_mask = Some(match keep_mask {
                    Some(prev) => prev.try_bitand(&keep)?,
                    None => keep,
                });
            } else {
                float_additive_mask = Some(mask);
            }
        }

        // Key padding mask: `[B, Sk]`, True = valid (the inverse of
        // `attn_mask`'s polarity). Broadcast over heads and queries.
        if let Some(padding) = key_padding_mask {
            let ndim = padding.ndim()?;
            ensure!(
                ndim == 2,
                NdimExactSnafu { op: "scaled_dot_product_attention key_padding_mask", expected: 2usize, actual: ndim }
            );
            let keep = padding.cast(DType::Bool).try_unsqueeze(1)?.try_unsqueeze(1)?; // [B, 1, 1, Sk]
            keep_mask = Some(match keep_mask {
                Some(prev) => prev.try_bitand(&keep)?,
                None => keep,
            });
        }

        // Apply the boolean keep mask additively (out-of-band → -large).
        if let Some(keep) = keep_mask.as_ref() {
            let neg_large = Tensor::const_(ConstValue::min(scores_dtype.base()), scores_dtype.clone());
            scores = scores.where_(keep, &neg_large)?;
        }
        // Apply a float additive mask (e.g. a pre-computed -inf padding mask).
        if let Some(additive) = float_additive_mask {
            scores = scores.try_add(additive)?;
        }

        // Softmax + output. Re-zero out-of-band weights so a fully-masked row
        // (whose softmax would otherwise be uniform over the masked keys)
        // produces exact zeros rather than `1/k_len` leakage.
        let mut attn_weights = scores.softmax(-1isize)?;
        if let Some(keep) = keep_mask.as_ref() {
            let zero = Tensor::const_(ConstValue::zero(scores_dtype.base()), scores_dtype.clone());
            let masked_out = keep.logical_not()?;
            attn_weights = zero.where_(&masked_out, &attn_weights)?;
        }
        // Back to the query dtype for `@ V`, as tinygrad does.
        if scores_dtype != q_dtype {
            attn_weights = attn_weights.cast(q_dtype);
        }
        attn_weights.matmul(value)
    }
}
