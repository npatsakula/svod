//! Resize operations (ONNX Resize operator building block).

use bon::bon;
use svod_dtype::DType;
use svod_ir::{ConstValue, SInt};

use super::{AspectRatioPolicy, CoordinateTransformMode, NearestMode, ResizeMode};
use crate::Tensor;
use crate::error::{KindResult, NdimMinimumSnafu, ParamRangeSnafu};

type Result<T> = crate::Result<T>;

#[bon]
impl Tensor {
    /// [`upsample`](Tensor::upsample) with an explicit coordinate transform.
    ///
    /// The default (`half_pixel`) reproduces "repeat every pixel" nearest
    /// upsampling; `align_corners` is what torch's bilinear
    /// `interpolate(..., align_corners=True)` uses.
    #[builder]
    #[track_caller]
    pub fn upsample_with(
        &self,
        scale: &[usize],
        #[builder(default)] mode: ResizeMode,
        #[builder(default)] coordinate_transformation_mode: CoordinateTransformMode,
    ) -> Result<Tensor> {
        origin_call!("upsample");
        let ndim = self.ndim()?;
        snafu::ensure!(ndim >= scale.len(), NdimMinimumSnafu { op: "upsample", min: scale.len(), actual: ndim });

        // Leading (batch / channel) axes get scale 1.0, so `resize` treats them
        // as inactive and carries symbolic extents through untouched.
        let mut scales = vec![1.0f64; ndim];
        for (i, &s) in scale.iter().enumerate() {
            snafu::ensure!(
                s > 0,
                ParamRangeSnafu { op: "upsample", param: "scale", value: s.to_string(), constraint: "> 0" }
            );
            scales[ndim - scale.len() + i] = s as f64;
        }

        self.resize().scales(&scales).mode(mode).coordinate_transformation_mode(coordinate_transformation_mode).call()
    }

    /// Resize a tensor using interpolation (ONNX Resize operator).
    ///
    /// Supports nearest, linear, and cubic interpolation modes with various
    /// coordinate transformation modes. Either `scales` or `sizes` must be
    /// provided to specify the target dimensions.
    ///
    /// # Examples
    ///
    /// Nearest-mode 2x upscale via `scales`:
    ///
    /// ```
    /// # use svod_tensor::Tensor;
    /// # use ndarray::Array4;
    /// let x = Tensor::from_ndarray(&Array4::from_elem((1, 1, 2, 2), 1.0f32));
    /// let y = x.resize().scales(&[1.0, 1.0, 2.0, 2.0]).call().unwrap();
    /// y.realize().unwrap();
    /// let shape: Vec<usize> = y.shape().unwrap().iter().map(|d| d.as_const().unwrap()).collect();
    /// assert_eq!(shape, vec![1, 1, 4, 4]);
    /// assert!(y.as_vec::<f32>().unwrap().iter().all(|&v| (v - 1.0).abs() < 1e-5));
    /// ```
    ///
    /// Resize to explicit output `sizes`:
    ///
    /// ```
    /// # use svod_tensor::Tensor;
    /// # use ndarray::Array4;
    /// let x = Tensor::from_ndarray(&Array4::from_elem((1, 1, 2, 2), 1.0f32));
    /// let y = x.resize().sizes(&[1, 1, 6, 6]).call().unwrap();
    /// y.realize().unwrap();
    /// let shape: Vec<usize> = y.shape().unwrap().iter().map(|d| d.as_const().unwrap()).collect();
    /// assert_eq!(shape, vec![1, 1, 6, 6]);
    /// assert!(y.as_vec::<f32>().unwrap().iter().all(|&v| (v - 1.0).abs() < 1e-5));
    /// ```
    ///
    /// Linear interpolation mode:
    ///
    /// ```
    /// # use svod_tensor::Tensor;
    /// # use svod_tensor::nn::ResizeMode;
    /// # use ndarray::Array4;
    /// let x = Tensor::from_ndarray(&Array4::from_elem((1, 1, 2, 2), 1.0f32));
    /// let y = x.resize()
    ///     .scales(&[1.0, 1.0, 2.0, 2.0])
    ///     .mode(ResizeMode::Linear)
    ///     .call()
    ///     .unwrap();
    /// y.realize().unwrap();
    /// let shape: Vec<usize> = y.shape().unwrap().iter().map(|d| d.as_const().unwrap()).collect();
    /// assert_eq!(shape, vec![1, 1, 4, 4]);
    /// assert!(y.as_vec::<f32>().unwrap().iter().all(|&v| (v - 1.0).abs() < 1e-5));
    /// ```
    // Tinygrad onnx.py:789-890
    #[builder]
    #[allow(clippy::too_many_arguments)]
    #[track_caller]
    pub fn resize(
        &self,
        scales: Option<&[f64]>,
        sizes: Option<&[usize]>,
        #[builder(default)] mode: ResizeMode,
        #[builder(default)] coordinate_transformation_mode: CoordinateTransformMode,
        #[builder(default)] nearest_mode: NearestMode,
        #[builder(default = -0.75)] cubic_coeff_a: f64,
        #[builder(default = false)] exclude_outside: bool,
        #[builder(default = false)] antialias: bool,
        #[builder(default)] keep_aspect_ratio_policy: AspectRatioPolicy,
        axes: Option<&[usize]>,
        roi: Option<&[f64]>,
        #[builder(default = 0.0)] extrapolation_value: f64,
    ) -> Result<Tensor> {
        origin_call!("resize");
        let ndim = self.ndim()?;
        let _shape = self.shape()?;

        let axes: Vec<usize> = axes.map(|a| a.to_vec()).unwrap_or_else(|| (0..ndim).collect());

        // Permute: put target axes last
        let non_axes: Vec<usize> = (0..ndim).filter(|d| !axes.contains(d)).collect();
        let perm: Vec<isize> = non_axes.iter().chain(axes.iter()).map(|&d| d as isize).collect();
        let inv_perm = argsort_usize(&perm.iter().map(|&p| p as usize).collect::<Vec<_>>());
        let inv_perm_i: Vec<isize> = inv_perm.iter().map(|&i| i as isize).collect();

        let mut x = if perm.iter().enumerate().all(|(i, &p)| p == i as isize) {
            self.clone()
        } else {
            self.try_permute(&perm)?
        };

        // Input spatial dimensions (last len(axes) dims of permuted x).
        // Only the spatial (axes) dims must be concrete; the prefix (batch,
        // channels) may be symbolic and is carried through as SInt.
        let x_shape = x.shape()?;
        let n_axes = axes.len();

        // Filter scales/sizes to spatial dims only
        let scales_trimmed: Option<Vec<f64>> = scales.map(|s| s[s.len().saturating_sub(n_axes)..].to_vec());
        let sizes_trimmed: Option<Vec<usize>> = sizes.map(|s| s[s.len().saturating_sub(n_axes)..].to_vec());

        // Identify active axes: axes where the scale != 1.0 (for scales) or
        // where the output size differs from the input (for sizes). Inactive
        // axes (e.g. symbolic batch with scale=1.0) are skipped entirely and
        // carried through as-is, matching tinygrad's interpolate which only
        // loops over trailing spatial dims.
        let active_idx: Vec<usize> = match &scales_trimmed {
            Some(sc) => (0..n_axes).filter(|&i| sc[i] != 1.0).collect(),
            None => match &sizes_trimmed {
                Some(sz) => {
                    // For sizes, we need to compare against input shape.
                    // Only dims that are concrete and differ are active.
                    (0..n_axes)
                        .filter(|&i| {
                            let in_sz = x_shape[ndim - n_axes + i].as_const();
                            in_sz.is_none_or(|s| s != sz[i])
                        })
                        .collect()
                }
                None => (0..n_axes).collect(),
            },
        };

        if active_idx.is_empty() {
            return if perm.iter().enumerate().any(|(i, &p)| p != i as isize) {
                x.try_permute(&inv_perm_i)
            } else {
                Ok(x)
            };
        }

        // Only extract/validate the active spatial dims (must be concrete).
        let input_shape: Vec<usize> = active_idx
            .iter()
            .map(|&i| {
                x_shape[ndim - n_axes + i].as_const().ok_or_else(|| crate::error::ErrorKind::SymbolicShapeUnsupported {
                    operation: "resize on a symbolic spatial dim".to_string(),
                })
            })
            .collect::<KindResult<_>>()?;

        let scales_active: Option<Vec<f64>> =
            scales_trimmed.as_ref().map(|sc| active_idx.iter().map(|&i| sc[i]).collect());
        let sizes_active: Option<Vec<usize>> =
            sizes_trimmed.as_ref().map(|sz| active_idx.iter().map(|&i| sz[i]).collect());

        // Compute output sizes and scales
        let (output_sizes, final_scales) = if let Some(mut sz) = sizes_active {
            if keep_aspect_ratio_policy == AspectRatioPolicy::NotLarger
                || keep_aspect_ratio_policy == AspectRatioPolicy::NotSmaller
            {
                let scale_fn: fn(f64, f64) -> f64 =
                    if keep_aspect_ratio_policy == AspectRatioPolicy::NotLarger { f64::min } else { f64::max };
                let mut scale = f64::NAN;
                for (s, &inp) in sz.iter().zip(&input_shape) {
                    let s_val = *s as f64 / inp as f64;
                    if scale.is_nan() {
                        scale = s_val;
                    } else {
                        scale = scale_fn(scale, s_val);
                    }
                }
                sz = input_shape.iter().map(|&sh| (scale * sh as f64 + 0.5) as usize).collect();
                let sc = vec![scale; sz.len()];
                (sz, sc)
            } else {
                let sc: Vec<f64> = sz.iter().zip(&input_shape).map(|(&s, &sh)| s as f64 / sh as f64).collect();
                (sz, sc)
            }
        } else if let Some(sc) = scales_active {
            let sz: Vec<usize> = sc.iter().zip(&input_shape).map(|(&s, &sh)| (s * sh as f64) as usize).collect();
            (sz, sc)
        } else {
            return Err(crate::error::ErrorKind::IrConstruction {
                details: "resize: either scales or sizes must be provided".into(),
            }
            .into());
        };

        // Early exit if no resize needed
        if output_sizes.iter().zip(&input_shape).all(|(&o, &i)| o == i) {
            return if perm.iter().enumerate().any(|(i, &p)| p != i as isize) {
                x.try_permute(&inv_perm_i)
            } else {
                Ok(x)
            };
        }

        // Nearest at an integer scale is "repeat each element" wherever the
        // coordinate transform and the rounding compose to `floor(o / s)`, and
        // `repeat_interleave` is that as a view. The general path below would
        // spend a one-hot masked sum-reduce per axis instead — the neck's two
        // upsamples are 101 µs of a 4.87 ms YOLO26x frame on gfx1201, and
        // 222 µs on an RTX 3060, for an operation that moves nothing.
        //
        // Only the two combinations that are exactly `floor(o / s)` are taken.
        // `half_pixel` gives `ceil((o + 0.5)/s - 1)`, and `(r + 0.5)/s - 1` is
        // in `(-1, 0)` for every `r < s`, so it is `o / s` exactly; `asymmetric`
        // with `floor` is `floor(o / s)` by construction. The four other pairings
        // are not — `asymmetric` with `round_prefer_floor` reads the next input
        // element once `r / s > 0.5` — and they keep the gather.
        if mode == ResizeMode::Nearest
            && matches!(
                (coordinate_transformation_mode, nearest_mode),
                (CoordinateTransformMode::HalfPixel, NearestMode::RoundPreferFloor)
                    | (CoordinateTransformMode::Asymmetric, NearestMode::Floor)
            )
            && let Some(repeats) = input_shape
                .iter()
                .zip(&output_sizes)
                .map(|(&inp, &out)| (inp > 0 && out >= inp && out.is_multiple_of(inp)).then_some(out / inp))
                .collect::<Option<Vec<_>>>()
        {
            for (i, repeat) in repeats.into_iter().enumerate() {
                x = x.repeat_interleave(repeat, (ndim - n_axes + active_idx[i]) as isize)?;
            }
            return if perm.iter().enumerate().any(|(i, &p)| p != i as isize) {
                x.try_permute(&inv_perm_i)
            } else {
                Ok(x)
            };
        }

        let n_spatial = active_idx.len();

        // Extract per-spatial-dim ROI (start, end) pairs
        let roi_pairs: Vec<(f64, f64)> = if let Some(roi) = roi {
            let half = roi.len() / 2;
            let starts = &roi[half - n_spatial..half];
            let ends = &roi[roi.len() - n_spatial..];
            starts.iter().zip(ends).map(|(&s, &e)| (s, e)).collect()
        } else {
            vec![(0.0, 1.0); n_spatial]
        };

        // Build coordinate transforms for each spatial dim
        let dtype = x.uop().dtype();
        let indexes: Vec<Tensor> = input_shape
            .iter()
            .zip(&output_sizes)
            .zip(&final_scales)
            .zip(&roi_pairs)
            .map(|(((&inp_sz, &out_sz), &scale), &(roi_start, roi_end))| {
                apply_coordinate_transform(
                    inp_sz,
                    out_sz,
                    scale,
                    coordinate_transformation_mode,
                    &dtype,
                    roi_start,
                    roi_end,
                )
            })
            .collect::<Result<_>>()?;

        // Clip for nearest/linear modes (skip for tf_crop_and_resize — uses extrapolation instead)
        let is_tf_crop = coordinate_transformation_mode == CoordinateTransformMode::TfCropAndResize;
        let indexes: Vec<Tensor> = if !is_tf_crop && matches!(mode, ResizeMode::Nearest | ResizeMode::Linear) {
            indexes
                .into_iter()
                .zip(&input_shape)
                .map(|(idx, &sz)| {
                    let zero = Tensor::const_(ConstValue::Float(0.0), dtype.clone());
                    let max_val = Tensor::const_(ConstValue::Float((sz - 1) as f64), dtype.clone());
                    idx.clamp().min(&zero).max(&max_val).call()
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            indexes
        };

        // For tf_crop_and_resize, build a validity mask from unclipped indexes
        let validity_mask: Option<Vec<Tensor>> = if is_tf_crop {
            Some(
                indexes
                    .iter()
                    .zip(&input_shape)
                    .map(|(idx, &sz)| {
                        let zero = Tensor::const_(ConstValue::Float(0.0), dtype.clone());
                        let max_val = Tensor::const_(ConstValue::Float((sz - 1) as f64), dtype.clone());
                        idx.try_ge(&zero)?.try_bitand(&idx.try_le(&max_val)?)
                    })
                    .collect::<Result<Vec<_>>>()?,
            )
        } else {
            None
        };

        // For tf_crop_and_resize, clip indexes before gather (to avoid OOB)
        let indexes: Vec<Tensor> = if is_tf_crop {
            indexes
                .into_iter()
                .zip(&input_shape)
                .map(|(idx, &sz)| {
                    let zero = Tensor::const_(ConstValue::Float(0.0), dtype.clone());
                    let max_val = Tensor::const_(ConstValue::Float((sz - 1) as f64), dtype.clone());
                    idx.clamp().min(&zero).max(&max_val).call()
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            indexes
        };

        if mode == ResizeMode::Nearest {
            let int_indexes: Vec<Tensor> = indexes
                .into_iter()
                .map(|idx| {
                    let rounded = match nearest_mode {
                        NearestMode::RoundPreferFloor => idx.try_sub(0.5f64)?.ceil(),
                        NearestMode::RoundPreferCeil => idx.try_add(0.5f64)?.floor(),
                        NearestMode::Floor => idx.floor(),
                        NearestMode::Ceil => idx.ceil(),
                    };
                    Ok(rounded.cast(DType::Int32))
                })
                .collect::<Result<Vec<_>>>()?;

            // Sequential gather per spatial dim
            for (i, idx) in int_indexes.iter().enumerate() {
                let dim = (ndim - n_axes + active_idx[i]) as isize;
                let cur_shape = x.shape()?;
                let out_sz = output_sizes[i];

                let mut idx_shape = vec![1isize; ndim];
                idx_shape[ndim - n_axes + active_idx[i]] = out_sz as isize;
                let idx_reshaped = idx.try_reshape(&idx_shape)?;

                let mut expand_shape: Vec<SInt> = cur_shape.to_vec();
                expand_shape[ndim - n_axes + active_idx[i]] = SInt::from(out_sz as i32);
                let idx_expanded = idx_reshaped.try_expand(&expand_shape)?;

                x = x.gather(dim, &idx_expanded)?;
            }
        } else if mode == ResizeMode::Linear {
            let mut expand: Vec<SInt> = x_shape.to_vec();
            for (i, &out_sz) in output_sizes.iter().enumerate() {
                let dim_pos = ndim - n_axes + active_idx[i];
                let scale = final_scales[i];
                let input_sz = input_shape[i];
                let index = &indexes[i];

                let mut reshape = vec![1isize; ndim];
                reshape[dim_pos] = out_sz as isize;
                expand[dim_pos] = SInt::from(out_sz as i32);

                if antialias && scale < 1.0 {
                    x = interpolate_antialias_linear(&x, index, dim_pos, input_sz, scale, &reshape, &expand, &dtype)?;
                } else {
                    let low = index.floor().cast(DType::Int32).try_reshape(&reshape)?.try_expand(&expand)?;
                    let high = index.ceil().cast(DType::Int32).try_reshape(&reshape)?.try_expand(&expand)?;
                    let perc = index.try_sub(index.floor())?.try_reshape(&reshape)?.try_expand(&expand)?;

                    let dim_i = dim_pos as isize;
                    let gathered_low = x.gather(dim_i, &low)?;
                    let gathered_high = x.gather(dim_i, &high)?;
                    x = gathered_low.lerp(&gathered_high, &perc)?;
                }
            }
        } else if mode == ResizeMode::Cubic {
            let a = cubic_coeff_a;
            let mut expand: Vec<SInt> = x_shape.to_vec();
            for (i, &out_sz) in output_sizes.iter().enumerate() {
                let dim_pos = ndim - n_axes + active_idx[i];
                let scale = final_scales[i];
                let input_sz = input_shape[i];
                let index = &indexes[i];

                let mut reshape = vec![1isize; ndim];
                reshape[dim_pos] = out_sz as isize;
                expand[dim_pos] = SInt::from(out_sz as i32);

                if antialias && scale < 1.0 {
                    x = interpolate_antialias_cubic(&x, index, dim_pos, input_sz, scale, a, &reshape, &expand, &dtype)?;
                } else {
                    let p = index.floor().cast(DType::Int32);
                    let ratio = index.try_sub(index.floor())?;

                    let one = Tensor::const_(ConstValue::Int(1), DType::Int32);
                    let two = Tensor::const_(ConstValue::Int(2), DType::Int32);
                    let idx0 = p.try_sub(&one)?;
                    let idx1 = p.clone();
                    let idx2 = p.try_add(&one)?;
                    let idx3 = p.try_add(&two)?;

                    let r1 = ratio.try_add(Tensor::const_(1.0f64, dtype.clone()))?;
                    let c0 = poly_n(&r1, &[a, -5.0 * a, 8.0 * a, -4.0 * a], &dtype)?;
                    let c1 = poly_n(&ratio, &[a + 2.0, -(a + 3.0), 0.0, 1.0], &dtype)?;
                    let r_neg1 = Tensor::const_(1.0f64, dtype.clone()).try_sub(&ratio)?;
                    let c2 = poly_n(&r_neg1, &[a + 2.0, -(a + 3.0), 0.0, 1.0], &dtype)?;
                    let r_neg2 = Tensor::const_(2.0f64, dtype.clone()).try_sub(&ratio)?;
                    let c3 = poly_n(&r_neg2, &[a, -5.0 * a, 8.0 * a, -4.0 * a], &dtype)?;

                    let (mut c0, mut c1, mut c2, mut c3) = (c0, c1, c2, c3);
                    if exclude_outside {
                        let max_idx = Tensor::const_(ConstValue::Int(input_sz as i64), DType::Int32);
                        let zero_i = Tensor::const_(ConstValue::Int(0), DType::Int32);
                        let zero_f = Tensor::const_(0.0f64, dtype.clone());
                        let valid0 = idx0.try_ge(&zero_i)?.try_mul(&idx0.try_lt(&max_idx)?)?;
                        let valid1 = idx1.try_ge(&zero_i)?.try_mul(&idx1.try_lt(&max_idx)?)?;
                        let valid2 = idx2.try_ge(&zero_i)?.try_mul(&idx2.try_lt(&max_idx)?)?;
                        let valid3 = idx3.try_ge(&zero_i)?.try_mul(&idx3.try_lt(&max_idx)?)?;
                        c0 = c0.where_(&valid0, &zero_f)?;
                        c1 = c1.where_(&valid1, &zero_f)?;
                        c2 = c2.where_(&valid2, &zero_f)?;
                        c3 = c3.where_(&valid3, &zero_f)?;
                        let total = c0.try_add(&c1)?.try_add(&c2)?.try_add(&c3)?;
                        let eps = Tensor::const_(1e-9f64, dtype.clone());
                        let total_safe = total.try_add(&eps)?;
                        c0 = c0.try_div(&total_safe)?;
                        c1 = c1.try_div(&total_safe)?;
                        c2 = c2.try_div(&total_safe)?;
                        c3 = c3.try_div(&total_safe)?;
                    }

                    let max_val = Tensor::const_(ConstValue::Int((input_sz - 1) as i64), DType::Int32);
                    let zero_i = Tensor::const_(ConstValue::Int(0), DType::Int32);
                    let clip = |t: &Tensor| -> Result<Tensor> {
                        t.clamp().min(&zero_i).max(&max_val).call()?.try_reshape(&reshape)?.try_expand(&expand)
                    };
                    let ei0 = clip(&idx0)?;
                    let ei1 = clip(&idx1)?;
                    let ei2 = clip(&idx2)?;
                    let ei3 = clip(&idx3)?;

                    let ec = |c: Tensor| -> Result<Tensor> { c.try_reshape(&reshape)?.try_expand(&expand) };
                    let ec0 = ec(c0)?;
                    let ec1 = ec(c1)?;
                    let ec2 = ec(c2)?;
                    let ec3 = ec(c3)?;

                    let dim_i = dim_pos as isize;
                    let v0 = x.gather(dim_i, &ei0)?.try_mul(&ec0)?;
                    let v1 = x.gather(dim_i, &ei1)?.try_mul(&ec1)?;
                    let v2 = x.gather(dim_i, &ei2)?.try_mul(&ec2)?;
                    let v3 = x.gather(dim_i, &ei3)?.try_mul(&ec3)?;
                    x = v0.try_add(&v1)?.try_add(&v2)?.try_add(&v3)?;
                }
            }
        }

        // Apply extrapolation for tf_crop_and_resize: out-of-bounds → extrapolation_value
        if let Some(masks) = validity_mask {
            let extrap = Tensor::const_(ConstValue::Float(extrapolation_value), dtype.clone());
            let expand_shape: Vec<SInt> = x.shape()?.to_vec();

            // Each mask_i is 1D [out_sz_i]; reshape to [1,..,out_sz_i,..,1] and broadcast
            let mut combined: Option<Tensor> = None;
            for (i, mask) in masks.into_iter().enumerate() {
                let mut shape = vec![1isize; ndim];
                shape[ndim - n_axes + active_idx[i]] = output_sizes[i] as isize;
                let broad = mask.try_reshape(&shape)?.try_expand(&expand_shape)?;
                combined = Some(match combined {
                    Some(c) => c.try_bitand(&broad)?,
                    None => broad,
                });
            }
            if let Some(valid) = combined {
                x = x.where_(&valid, &extrap)?;
            }
        }

        // Permute back
        if perm.iter().enumerate().any(|(i, &p)| p != i as isize) { x.try_permute(&inv_perm_i) } else { Ok(x) }
    }
}

impl Tensor {
    /// Upsample the trailing spatial axes by integer factors (NCHW / NCL).
    ///
    /// Shortcut for [`resize`](Tensor::resize) with half-pixel coordinates,
    /// which for an integer factor is exactly "repeat each element" in
    /// [`ResizeMode::Nearest`]. Leading batch/channel axes may be symbolic.
    /// Use [`upsample_with`](Tensor::upsample_with) to pick another coordinate
    /// transform (e.g. `align_corners` for torch-style bilinear).
    ///
    /// # Examples
    ///
    /// ```
    /// # use svod_tensor::Tensor;
    /// # use svod_tensor::nn::ResizeMode;
    /// # use ndarray::Array4;
    /// let x = Tensor::from_ndarray(&Array4::from_elem((1, 2, 3, 4), 1.0f32));
    /// let y = x.upsample(&[2, 2], ResizeMode::Nearest).unwrap();
    /// assert_eq!(y.dims().unwrap(), vec![1, 2, 6, 8]);
    /// ```
    #[track_caller]
    pub fn upsample(&self, scale: &[usize], mode: ResizeMode) -> Result<Tensor> {
        self.upsample_with().scale(scale).mode(mode).call()
    }
}

/// Coordinate transform for resize operations.
///
/// Computes in f64 to avoid precision loss from IR constant folding
/// (which uses mixed f64/f32 arithmetic), then casts to target dtype.
fn apply_coordinate_transform(
    input_sz: usize,
    output_sz: usize,
    scale: f64,
    mode: CoordinateTransformMode,
    dtype: &DType,
    roi_start: f64,
    roi_end: f64,
) -> Result<Tensor> {
    let f64_dt = DType::Float64;
    let index = Tensor::arange(0, Some(output_sz as i64), None)?.cast(f64_dt.clone());
    let result = match mode {
        CoordinateTransformMode::HalfPixel => {
            let half = Tensor::const_(0.5f64, f64_dt.clone());
            index.try_add(&half)?.try_div(Tensor::const_(scale, f64_dt))?.try_sub(&half)?
        }
        CoordinateTransformMode::AlignCorners => {
            // ONNX reference uses float output_width = scale * input_sz, not integer output_sz.
            // This matters when scale * input_sz is non-integer (e.g. 0.8 * 4 = 3.2 vs int 3).
            let output_width = scale * input_sz as f64;
            if output_width == 1.0 {
                Tensor::const_(0.0f64, f64_dt)
            } else {
                let ratio = (input_sz as f64 - 1.0) / (output_width - 1.0);
                index.try_mul(Tensor::const_(ratio, f64_dt))?
            }
        }
        CoordinateTransformMode::Asymmetric => index.try_div(Tensor::const_(scale, f64_dt))?,
        CoordinateTransformMode::PytorchHalfPixel => {
            let output_width = scale * input_sz as f64;
            if output_width == 1.0 {
                Tensor::const_(0.0f64, f64_dt)
            } else {
                let half = Tensor::const_(0.5f64, f64_dt.clone());
                index.try_add(&half)?.try_div(Tensor::const_(scale, f64_dt))?.try_sub(&half)?
            }
        }
        CoordinateTransformMode::HalfPixelSymmetric => {
            let output_dim_scaled = input_sz as f64 * scale;
            let offset = (input_sz as f64 / 2.0) * (1.0 - output_sz as f64 / output_dim_scaled);
            let half = Tensor::const_(0.5f64, f64_dt.clone());
            let off_t = Tensor::const_(offset, f64_dt.clone());
            off_t.try_add(&index.try_add(&half)?.try_div(Tensor::const_(scale, f64_dt))?)?.try_sub(&half)?
        }
        CoordinateTransformMode::TfCropAndResize => {
            let len = (input_sz as f64) - 1.0;
            let output_width = scale * input_sz as f64;
            if output_width == 1.0 {
                Tensor::const_((roi_end - roi_start) * len / 2.0 + roi_start * len, f64_dt)
            } else {
                let stride = (roi_end - roi_start) * len / (output_width - 1.0);
                let offset = roi_start * len;
                index.try_mul(Tensor::const_(stride, f64_dt.clone()))?.try_add(Tensor::const_(offset, f64_dt))?
            }
        }
    };
    Ok(result.cast(dtype.clone()))
}

/// Horner's method for polynomial evaluation.
fn poly_n(x: &Tensor, coeffs: &[f64], dtype: &DType) -> Result<Tensor> {
    coeffs.iter().try_fold(Tensor::const_(0.0f64, dtype.clone()), |acc, &c| {
        acc.try_mul(x)?.try_add(Tensor::const_(c, dtype.clone()))
    })
}

/// Antialias cubic interpolation for one spatial dimension.
/// When downsampling (scale < 1), widens the kernel by 1/scale to prevent aliasing.
/// ONNX ref: _cubic_coeffs_antialias in op_resize.py
#[allow(clippy::too_many_arguments)]
fn interpolate_antialias_cubic(
    x: &Tensor,
    index: &Tensor,
    dim_pos: usize,
    input_sz: usize,
    scale: f64,
    a: f64,
    reshape: &[isize],
    expand_i: &[SInt],
    dtype: &DType,
) -> Result<Tensor> {
    let i_start = (-2.0_f64 / scale).floor() as i32 + 1;
    let i_end = 2 - i_start;
    let n_taps = (i_end - i_start) as usize;

    let floored = index.floor();
    let p = floored.cast(DType::Int32);
    let ratio = index.try_sub(&floored)?;

    let one = Tensor::const_(1.0f64, dtype.clone());
    let two = Tensor::const_(2.0f64, dtype.clone());
    let zero_f = Tensor::const_(0.0f64, dtype.clone());

    let mut coeffs = Vec::with_capacity(n_taps);
    for tap in i_start..i_end {
        let arg = ratio
            .try_mul(Tensor::const_(-scale, dtype.clone()))?
            .try_add(Tensor::const_(scale * tap as f64, dtype.clone()))?;
        let abs_arg = arg.abs();
        let c_inner = poly_n(&abs_arg, &[a + 2.0, -(a + 3.0), 0.0, 1.0], dtype)?;
        let c_outer = poly_n(&abs_arg, &[a, -5.0 * a, 8.0 * a, -4.0 * a], dtype)?;
        let mask_outer = abs_arg.try_lt(&two)?;
        let c = c_outer.where_(&mask_outer, &zero_f)?;
        let mask_inner = abs_arg.try_le(&one)?;
        let c = c_inner.where_(&mask_inner, &c)?;
        coeffs.push(c);
    }

    normalize_and_gather(x, coeffs, &p, i_start, dim_pos, input_sz, reshape, expand_i, dtype)
}

/// Antialias linear interpolation for one spatial dimension.
/// ONNX ref: _linear_coeffs_antialias in op_resize.py
#[allow(clippy::too_many_arguments)]
fn interpolate_antialias_linear(
    x: &Tensor,
    index: &Tensor,
    dim_pos: usize,
    input_sz: usize,
    scale: f64,
    reshape: &[isize],
    expand_i: &[SInt],
    dtype: &DType,
) -> Result<Tensor> {
    let start = (-1.0_f64 / scale).floor() as i32 + 1;
    let footprint = (2 - 2 * start) as usize;

    let floored = index.floor();
    let p = floored.cast(DType::Int32);
    let ratio = index.try_sub(&floored)?;

    let one = Tensor::const_(1.0f64, dtype.clone());
    let zero_f = Tensor::const_(0.0f64, dtype.clone());

    let mut coeffs = Vec::with_capacity(footprint);
    for j in 0..footprint {
        let tap = start + j as i32;
        let arg = ratio
            .try_mul(Tensor::const_(-scale, dtype.clone()))?
            .try_add(Tensor::const_(scale * tap as f64, dtype.clone()))?;
        let abs_arg = arg.abs();
        let c = one.try_sub(&abs_arg)?;
        let c = c.clamp().min(&zero_f).max(&one).call()?;
        coeffs.push(c);
    }

    normalize_and_gather(x, coeffs, &p, start, dim_pos, input_sz, reshape, expand_i, dtype)
}

/// Normalize coefficients to sum to 1, then gather and accumulate weighted values.
/// Shared by antialias cubic and linear interpolation.
#[allow(clippy::too_many_arguments)]
fn normalize_and_gather(
    x: &Tensor,
    mut coeffs: Vec<Tensor>,
    p: &Tensor,
    tap_start: i32,
    dim_pos: usize,
    input_sz: usize,
    reshape: &[isize],
    expand_i: &[SInt],
    dtype: &DType,
) -> Result<Tensor> {
    let mut total = coeffs[0].clone();
    for c in &coeffs[1..] {
        total = total.try_add(c)?;
    }
    let eps = Tensor::const_(1e-9f64, dtype.clone());
    let total_safe = total.try_add(&eps)?;
    for c in &mut coeffs {
        *c = c.try_div(&total_safe)?;
    }

    let max_val = Tensor::const_(ConstValue::Int((input_sz - 1) as i64), DType::Int32);
    let zero_i = Tensor::const_(ConstValue::Int(0), DType::Int32);
    let dim_i = dim_pos as isize;

    let mut result: Option<Tensor> = None;
    for (j, c) in coeffs.into_iter().enumerate() {
        let tap = tap_start + j as i32;
        let idx = p.try_add(Tensor::const_(ConstValue::Int(tap as i64), DType::Int32))?;
        let idx_clipped = idx.clamp().min(&zero_i).max(&max_val).call()?.try_reshape(reshape)?.try_expand(expand_i)?;
        let c_expanded = c.try_reshape(reshape)?.try_expand(expand_i)?;
        let val = x.gather(dim_i, &idx_clipped)?.try_mul(&c_expanded)?;
        result = Some(match result {
            Some(acc) => acc.try_add(&val)?,
            None => val,
        });
    }
    Ok(result.unwrap())
}

fn argsort_usize(slice: &[usize]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..slice.len()).collect();
    indices.sort_by_key(|&i| slice[i]);
    indices
}
