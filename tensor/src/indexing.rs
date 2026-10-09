//! Indexing operations for Tensors.

use snafu::{OptionExt, ResultExt};
use strum::{Display, EnumString};

use super::*;
use crate::error::{NdimMinimumSnafu, ShapeMismatchSnafu, SymbolicShapeUnsupportedSnafu, TypeMismatchSnafu};

/// Reduction mode for scatter operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumString, Display)]
pub enum ScatterReduction {
    #[strum(serialize = "sum")]
    Sum,
    #[strum(serialize = "prod")]
    Prod,
    #[strum(serialize = "amax")]
    Amax,
    #[strum(serialize = "amin")]
    Amin,
}

impl Tensor {
    /// Gather values along an axis specified by `dim`, using `index` for element selection.
    #[track_caller]
    pub fn gather(&self, dim: isize, index: &Tensor) -> Result<Self> {
        origin_call!("gather");
        let self_shape = self.shape()?;
        let index_shape = index.shape()?;
        let ndim = self_shape.len();
        let dim = Self::normalize_axis(dim, ndim)?;

        snafu::ensure!(
            index_shape.len() == ndim,
            ShapeMismatchSnafu {
                context: "gather",
                expected: format!("{ndim}D"),
                actual: format!("{}D index", index_shape.len())
            }
        );

        // Compatibility holds where both dims are concrete; symbolic dims
        // (e.g. a JIT batch bound to a `BoundVariable`) pass through untouched.
        let compatible = self_shape.iter().zip(index_shape.iter()).enumerate().all(|(d, (s, i))| {
            if d == dim {
                return true;
            }
            match (s.as_const(), i.as_const()) {
                (Some(s), Some(i)) => s >= i,
                // A symbolic non-gather dim must be the *identical* extent on
                // both sides — gather doesn't broadcast, and the shrink below
                // would be ill-defined otherwise.
                _ => s == i,
            }
        });
        snafu::ensure!(
            compatible,
            ShapeMismatchSnafu {
                context: "gather",
                expected: "self[d] >= index[d] for d != dim".to_string(),
                actual: format!("self={self_shape:?}, index={index_shape:?}")
            }
        );

        // Shrink `self` to `index`'s size on every non-gather axis; keep the
        // gather axis full (`None`). Non-gather dims pass through as `SInt`, so
        // a symbolic batch survives. Port of tinygrad `Tensor.gather`.
        let shrink: Vec<_> = (0..ndim)
            .map(|d| if d == dim { None } else { Some((svod_ir::SInt::Const(0), index_shape[d].clone())) })
            .collect();
        let x = self.try_shrink(shrink)?.try_unsqueeze(-1)?.try_transpose(-1, dim as isize)?;

        // The gather axis is materialized as an arange, so it must be concrete.
        let dim_size = self_shape[dim].as_const().ok_or_else(|| crate::error::ErrorKind::SymbolicShapeUnsupported {
            operation: "gather along a symbolic dim".to_string(),
        })?;
        let arange = Tensor::index_arange(dim_size, &index.uop().dtype())?;
        let mask = index.try_unsqueeze(-1)?.try_eq(&arange)?;

        x.where_(&mask, Self::new(x.uop().const_like(0)))?.sum_with().axes(-1).dtype(self.uop().dtype()).call()
    }

    /// The positions `0..n` an index of `dtype` is compared against. Built in Int64
    /// and cast to `dtype` only when every position fits it: the comparison then
    /// happens in the index's own dtype, where the gated-load collapse can read the
    /// index without narrowing it. A position `dtype` cannot hold stays Int64 and
    /// the comparison widens the index instead of wrapping positions onto it.
    pub(crate) fn index_arange(n: usize, dtype: &DType) -> Result<Tensor> {
        let int64 = DType::Int64;
        let arange = Tensor::arange_with_dtype()
            .start(UOp::const_(int64.clone(), ConstValue::Int(0)))
            .stop(UOp::const_(int64.clone(), ConstValue::Int(n as i64)))
            .dtype(int64)
            .call()?;
        let fits = dtype.is_int() && n.saturating_sub(1) as f64 <= dtype.max_value();
        Ok(if fits { arange.cast(dtype.clone()) } else { arange })
    }

    /// Select elements along `dim` using a 1D index tensor.
    ///
    /// For input shape `[A, B, C]` with `dim=1` and index shape `[K]`,
    /// returns shape `[A, K, C]`.
    #[track_caller]
    pub fn index_select(&self, dim: isize, index: &Tensor) -> Result<Self> {
        origin_call!("index_select");
        let self_shape = self.shape()?;
        let ndim = self_shape.len();
        let dim = Self::normalize_axis(dim, ndim)?;
        // Reshape 1D index [K] → [1, ..., K, ..., 1] matching input ndim.
        let idx_len = index.dim_const(0)?;
        let mut idx_shape = vec![1isize; ndim];
        idx_shape[dim] = idx_len as isize;
        let idx_nd = index.try_reshape(&idx_shape)?;

        // Expand to `self`'s shape with K at the `dim` position. Built from
        // `self_shape` as `SInt`, so a symbolic dim (e.g. a JIT batch) passes
        // through untouched instead of forcing the whole shape concrete.
        let mut expand_shape: Vec<_> = self_shape.iter().cloned().collect();
        expand_shape[dim] = svod_ir::SInt::Const(idx_len);
        let idx_expanded = idx_nd.try_expand(expand_shape)?;

        self.gather(dim as isize, &idx_expanded)
    }

    /// One-hot encoding: self == arange(num_classes) broadcast along dim.
    /// Returns a boolean tensor with True at the class positions.
    #[track_caller]
    pub fn one_hot_along_dim(&self, num_classes: usize, dim: isize) -> Result<Tensor> {
        origin_call!("one_hot_along_dim");
        let ndim = self.ndim()?;
        let norm_dim = Self::normalize_axis(dim, ndim)?;
        let offset = ndim - norm_dim - 1;
        let arange = Tensor::index_arange(num_classes, &self.uop().dtype())?;
        let mut ar_shape = vec![1isize; 1 + offset];
        ar_shape[0] = num_classes as isize;
        self.try_eq(&arange.try_reshape(&ar_shape)?)
    }

    /// Normalize negative indices: `indices[i] = indices[i] < 0 ? indices[i] + dim_size : indices[i]`
    #[track_caller]
    pub fn normalize_negative_indices(&self, dim_size: i64) -> Result<Tensor> {
        origin_call!("normalize_negative_indices");
        let zero = Tensor::const_(ConstValue::Int(0), self.uop().dtype());
        let dim_t = Tensor::const_(ConstValue::Int(dim_size), self.uop().dtype());
        let neg_mask = self.try_lt(&zero)?;
        self.try_add(&dim_t)?.where_(&neg_mask, self)
    }

    // =========================================================================
    // Scatter Operations (Tinygrad tensor.py:2641-2728)
    // =========================================================================

    /// Internal: prepare src and mask for scatter operations.
    ///
    /// Validates shapes, shrinks src to index.shape, then:
    ///  - src: unsqueeze(-1), expand(self.shape[dim]), transpose(-1, dim)
    ///  - mask: one_hot_along_dim(self.shape[dim]), transpose(-1, dim)
    ///
    /// Both are padded to self.shape on non-dim axes.
    fn _pre_scatter(&self, dim: isize, index: &Tensor, src: &Tensor) -> Result<(Tensor, Tensor)> {
        let self_shape = self.shape()?;
        let index_shape = index.shape()?;
        let src_shape = src.shape()?;
        let ndim = self_shape.len();
        let dim = Self::normalize_axis(dim, ndim)?;

        let self_dims = svod_ir::shape::to_vec_usize(&self_shape).context(UOpSnafu)?;
        let index_dims = svod_ir::shape::to_vec_usize(&index_shape).context(UOpSnafu)?;
        let src_dims = svod_ir::shape::to_vec_usize(&src_shape).context(UOpSnafu)?;

        snafu::ensure!(
            index_shape.len() == ndim && src_shape.len() == ndim,
            ShapeMismatchSnafu {
                context: "scatter",
                expected: format!("{ndim}D"),
                actual: format!("index={}D, src={}D", index_shape.len(), src_shape.len())
            }
        );
        snafu::ensure!(
            self_dims
                .iter()
                .zip(&index_dims)
                .zip(&src_dims)
                .enumerate()
                .all(|(d, ((s, i), sr))| { (d == dim || s >= i) && sr >= i }),
            ShapeMismatchSnafu {
                context: "scatter",
                expected: "valid scatter shape constraints".to_string(),
                actual: format!("self={self_dims:?}, index={index_dims:?}, src={src_dims:?}")
            }
        );

        // Shrink src to index shape
        let shrink_ranges: Vec<(isize, isize)> = index_dims.iter().map(|&d| (0, d as isize)).collect();
        let src = src.try_shrink(&shrink_ranges)?;

        // src: unsqueeze(-1) → expand(... self.shape[dim]) → transpose(-1, dim)
        let mut expand_shape: Vec<isize> = index_dims.iter().map(|&d| d as isize).collect();
        expand_shape.push(self_dims[dim] as isize);
        let src = src.try_unsqueeze(-1)?.try_expand(&expand_shape)?.try_transpose(-1, dim as isize)?;

        // mask: one_hot_along_dim(self.shape[dim]) → transpose(-1, dim)
        let mask = index.try_unsqueeze(-1)?.one_hot_along_dim(self_dims[dim], -1)?.try_transpose(-1, dim as isize)?;

        // Pad both to self.shape on non-dim axes
        let src_cur = src.shape()?;
        let src_cur_dims = svod_ir::shape::to_vec_usize(&src_cur).context(UOpSnafu)?;
        let padding: Vec<(isize, isize)> =
            (0..ndim).map(|d| (0, (self_dims[d] as isize - src_cur_dims[d] as isize).max(0))).collect();
        let needs_pad = padding.iter().any(|&(_, e)| e > 0);
        let src = if needs_pad { src.try_pad(&padding)? } else { src };
        let mask = if needs_pad { mask.try_pad(&padding)? } else { mask };

        Ok((src, mask))
    }

    /// Scatter values along dim using index positions.
    ///
    /// For each position in index, places the corresponding src value into self at
    /// the specified index along dim. When multiple indices map to the same position,
    /// the last value wins (matching PyTorch/Tinygrad semantics).
    #[track_caller]
    pub fn scatter(&self, dim: isize, index: &Tensor, src: &Tensor) -> Result<Tensor> {
        origin_call!("scatter");
        let (src_p, mask_p) = self._pre_scatter(dim, index, src)?;
        masked_setitem(self, &src_p, &mask_p, &[-1])
    }

    /// Scatter with reduction. Applies reduce (sum/prod/amax/amin) at scatter positions.
    #[track_caller]
    pub fn scatter_reduce(
        &self,
        dim: isize,
        index: &Tensor,
        src: &Tensor,
        reduce: ScatterReduction,
        include_self: bool,
    ) -> Result<Tensor> {
        origin_call!("scatter_reduce");
        let (src_p, mask_p) = self._pre_scatter(dim, index, src)?;
        let dtype = src_p.uop().dtype();
        let inv_mask = |a: &Tensor, b: &Tensor| -> Result<Tensor> {
            let no_hit = mask_p.any(-1isize)?.logical_not()?;
            a.where_(&no_hit, b)
        };
        let self_or = |identity_val: ConstValue| -> Result<Tensor> {
            if include_self { Ok(self.clone()) } else { inv_mask(self, &Tensor::const_(identity_val, dtype.clone())) }
        };

        match reduce {
            ScatterReduction::Sum => {
                let zero = Tensor::const_(ConstValue::Int(0), dtype.clone());
                // Scatter keeps `self`'s dtype: no accumulator promotion here.
                let reduced = src_p.where_(&mask_p, &zero)?.sum_with().axes(-1isize).promote(false).call()?;
                reduced.try_add(&self_or(ConstValue::Int(0))?)
            }
            ScatterReduction::Prod => {
                let one = Tensor::const_(ConstValue::Int(1), dtype.clone());
                let reduced = src_p.where_(&mask_p, &one)?.prod_with().axes(-1isize).call()?;
                reduced.try_mul(&self_or(ConstValue::Int(1))?)
            }
            ScatterReduction::Amax => {
                let min_val =
                    if dtype.is_float() { ConstValue::Float(f64::NEG_INFINITY) } else { ConstValue::Int(i64::MIN) };
                let fill = Tensor::const_(min_val, dtype.clone());
                let reduced = src_p.where_(&mask_p, &fill)?.max(-1isize)?;
                reduced.maximum(&self_or(min_val)?)
            }
            ScatterReduction::Amin => {
                let max_val =
                    if dtype.is_float() { ConstValue::Float(f64::INFINITY) } else { ConstValue::Int(i64::MAX) };
                let fill = Tensor::const_(max_val, dtype.clone());
                let reduced = src_p.where_(&mask_p, &fill)?.min(-1isize)?;
                reduced.minimum(&self_or(max_val)?)
            }
        }
    }

    // =========================================================================
    // Masked Select (Tinygrad tensor.py:1528-1547)
    // =========================================================================

    /// Select elements where mask is true, returning a flat tensor.
    ///
    /// Requires `realize()` internally (data-dependent output size).
    #[track_caller]
    pub fn masked_select(&self, mask: &Tensor) -> Result<Tensor> {
        origin_call!("masked_select");
        if mask.uop().dtype() != DType::Bool {
            return TypeMismatchSnafu { expected: DType::Bool, actual: mask.uop().dtype() }.fail().map_err(Into::into);
        }
        let x = self.flatten()?;
        let mask_flat = mask.broadcast_to(&self.shape()?)?.flatten()?;
        let mask_cumsum = mask_flat.cast(DType::Int64).cumsum(0)?;
        // Realize to get output size (data-dependent shape)
        let n = mask_flat.numel()?;
        if n == 0 {
            return Ok(Tensor::empty_zero(self.uop().dtype()).to(self.device()));
        }
        let count_t = mask_cumsum.try_shrink([((n - 1) as isize, n as isize)])?;
        count_t.realize()?;
        let count_t = count_t.as_ndarray::<i64>()?;
        let count = count_t[[0]] as usize;
        if count == 0 {
            return Ok(Tensor::empty_zero(self.uop().dtype()).to(self.device()));
        }

        // Build gather indices: zeros.scatter(0, cumsum, 1).cumsum
        let zeros = Tensor::full(&[count], ConstValue::Int(0), DType::Int64);
        let ones = Tensor::full(&[n], ConstValue::Int(1), DType::Int64);
        let idxs = zeros.scatter_reduce(0, &mask_cumsum, &ones, ScatterReduction::Sum, false)?.cumsum(0)?;
        x.gather(0, &idxs)
    }

    /// Select elements along an axis where `condition` is true.
    ///
    /// If `axis` is None, the input is flattened first and selection is along axis 0.
    /// The condition is a 1D boolean/integer tensor; nonzero values select.
    #[track_caller]
    pub fn compress(&self, condition: &[bool], axis: Option<isize>) -> Result<Tensor> {
        origin_call!("compress");
        let x = if axis.is_none() { self.flatten()? } else { self.clone() };
        let axis = axis.unwrap_or(0);
        let indices: Vec<i64> = condition.iter().enumerate().filter(|(_, v)| **v).map(|(i, _)| i as i64).collect();
        let idx = Tensor::from_slice(&indices);
        x.index_select(axis, &idx)
    }

    // =========================================================================
    // Sort (Bitonic) (Tinygrad tensor.py:2730-2779)
    // =========================================================================

    /// Bitonic sort along a dimension. Returns (sorted_values, indices).
    #[track_caller]
    pub fn sort(&self, dim: isize, descending: bool) -> Result<(Tensor, Tensor)> {
        origin_call!("sort");
        let shape = self.shape()?;
        let ndim = shape.len();
        let dim = Self::normalize_axis(dim, ndim)?;
        let orig_len = shape[dim]
            .as_const()
            .ok_or_else(|| crate::error::ErrorKind::SymbolicShapeUnsupported { operation: "sort".into() })?;

        if orig_len <= 1 {
            let idx = Tensor::full(&self.dims()?, ConstValue::Int(0), svod_dtype::DType::Int32);
            return Ok((self.clone(), idx));
        }

        let n_stages = (orig_len as u64 - 1).ilog2() as usize + 1;
        let padded_len = 1usize << n_stages;

        // Pad to a power of 2 with a sentinel that loses to every real element:
        // ±inf for floats, the dtype's own extreme otherwise (an `i64`-derived
        // sentinel is wrong for `u64` above 2^63 and for `bool`).
        let dtype = self.uop().dtype();
        let mut padding = vec![(0isize, 0isize); ndim];
        padding[dim] = (0, (padded_len - orig_len) as isize);
        let mut x = if dtype.is_float() {
            let sentinel = if descending { f64::NEG_INFINITY } else { f64::INFINITY };
            self.try_pad_value(&padding, sentinel)?
        } else {
            let scalar = dtype.scalar().expect("sort requires a scalar dtype");
            let sentinel = if descending { ConstValue::min(scalar) } else { ConstValue::max(scalar) };
            self.pad_const(&padding, sentinel)?
        };

        // Unflatten dim into n_stages binary dimensions
        let unflatten_sizes: Vec<isize> = vec![2; n_stages];
        x = x.unflatten(dim as isize, &unflatten_sizes)?;

        // Bitonic sort network
        for stage in 1..=n_stages {
            if stage != n_stages {
                // Crossover: flip for green boxes
                let crossover_dim = (dim + n_stages - stage - 1) as isize;
                let halves = x.split(&[1, 1], crossover_dim)?;
                let (blue, green) = (&halves[0], &halves[1]);
                let flip_dims: Vec<isize> = (1..=(stage + (ndim - dim))).map(|i| -(i as isize)).collect();
                x = Tensor::cat(&[blue, &green.flip(&flip_dims)?], crossover_dim)?.contiguous();
            }

            for substage in (0..stage).rev() {
                let partner_dim = (dim + n_stages - substage - 1) as isize;
                let parts = x.split(&[1, 1], partner_dim)?;
                let (x_top, x_bottom) = (&parts[0], &parts[1]);
                let x_larger = x_top.maximum(x_bottom)?;
                let x_smaller = x_top.minimum(x_bottom)?;
                x = if descending {
                    Tensor::cat(&[&x_larger, &x_smaller], partner_dim)?
                } else {
                    Tensor::cat(&[&x_smaller, &x_larger], partner_dim)?
                }
                .contiguous();
            }

            if stage != n_stages {
                // Undo crossover
                let crossover_dim = (dim + n_stages - stage - 1) as isize;
                let halves = x.split(&[1, 1], crossover_dim)?;
                let (blue, flipped_green) = (&halves[0], &halves[1]);
                let flip_dims: Vec<isize> = (1..=(stage + (ndim - dim))).map(|i| -(i as isize)).collect();
                x = Tensor::cat(&[blue, &flipped_green.flip(&flip_dims)?], crossover_dim)?;
            }
        }

        // Flatten back and shrink to original size
        let flatten_end = dim + n_stages - 1;
        // Flatten dims [dim..dim+n_stages] back to one
        let cur_shape = x.shape()?;
        let cur_dims = svod_ir::shape::to_vec_usize(&cur_shape).context(UOpSnafu)?;
        let mut flat_shape: Vec<isize> = Vec::new();
        for (i, &d) in cur_dims.iter().enumerate() {
            if i == dim {
                flat_shape.push(padded_len as isize);
            } else if i > dim && i <= flatten_end {
                continue;
            } else {
                flat_shape.push(d as isize);
            }
        }
        x = x.try_reshape(&flat_shape)?;

        // Shrink to original size
        let x_shape = x.shape()?;
        let x_dims = svod_ir::shape::to_vec_usize(&x_shape).context(UOpSnafu)?;
        let shrink_ranges: Vec<(isize, isize)> =
            x_dims.iter().enumerate().map(|(d, &s)| (0, if d == dim { orig_len } else { s } as isize)).collect();
        x = x.try_shrink(&shrink_ranges)?;

        // Compute indices via count-matching (matches Tinygrad's approach)
        // Create 2D tril mask first (tril operates on last 2 dims), then reshape
        // to broadcast shape [1, ..., orig_len, orig_len, 1, ..., 1]
        // Tinygrad: Tensor.ones(orig_len, orig_len).tril().reshape((None, None) + (1,)*(ndim-dim-1))
        let tril_2d = Tensor::full(&[orig_len, orig_len], true, svod_dtype::DType::Bool).tril(0)?;
        let mut tril_reshape: Vec<isize> = vec![1; ndim + 1];
        tril_reshape[dim] = orig_len as isize;
        tril_reshape[dim + 1] = orig_len as isize;
        let tril_mask = tril_2d.try_reshape(&tril_reshape)?;

        // Count occurrences of each value up to current position
        let compute_counts = |t: &Tensor| -> Result<Tensor> {
            let eq = t.try_unsqueeze(dim as isize)?.try_eq(&t.try_unsqueeze((dim + 1) as isize)?)?;
            eq.try_bitand(&tril_mask)?.sum((dim + 1) as isize)
        };

        let count_orig = compute_counts(self)?;
        let count_sorted = compute_counts(&x)?;

        // Match: original[unsqueeze(dim+1)] == sorted[unsqueeze(dim)] && counts match
        let val_match = self.try_unsqueeze((dim + 1) as isize)?.try_eq(&x.try_unsqueeze(dim as isize)?)?;
        let cnt_match =
            count_orig.try_unsqueeze((dim + 1) as isize)?.try_eq(&count_sorted.try_unsqueeze(dim as isize)?)?;
        let cond = val_match.try_bitand(&cnt_match)?;

        // Build index arange and compute weighted sum
        let mut idx_shape = vec![1isize; ndim + 1];
        idx_shape[dim] = orig_len as isize;
        let idx = (cond
            .cast(svod_dtype::DType::Int32)
            .try_mul(&Tensor::arange(0, Some(orig_len as i64), None)?.try_reshape(&idx_shape)?)?)
        .sum(dim as isize)?;

        Ok((x, idx))
    }

    // =========================================================================
    // TopK (Tinygrad tensor.py:2792-2812)
    // =========================================================================

    /// Top-k elements along a dimension. Returns (values, indices).
    #[track_caller]
    pub fn topk(&self, k: usize, dim: isize, largest: bool) -> Result<(Tensor, Tensor)> {
        origin_call!("topk");
        let shape = self.shape()?;
        let ndim = shape.len();
        let norm_dim = Self::normalize_axis(dim, ndim)?;
        let (x, idx) = self.sort(dim, largest)?;
        // Shrink to first k along dim
        let x_shape = x.shape()?;
        let x_dims = svod_ir::shape::to_vec_usize(&x_shape).context(UOpSnafu)?;
        let shrink: Vec<(isize, isize)> =
            x_dims.iter().enumerate().map(|(d, &s)| (0, if d == norm_dim { k } else { s } as isize)).collect();
        Ok((x.try_shrink(&shrink)?, idx.try_shrink(&shrink)?))
    }

    // =========================================================================
    // NonZero (Tinygrad tensor.py:1549-1573)
    // =========================================================================

    /// Indices of non-zero elements. Returns [num_nonzero, ndim] tensor.
    #[track_caller]
    pub fn nonzero(&self) -> Result<Tensor> {
        origin_call!("nonzero");
        let shape = self.shape()?;
        let ndim = shape.len();
        let dims = svod_ir::shape::to_vec_usize(&shape).context(UOpSnafu)?;
        let numel: usize = dims.iter().product();

        let mask = self.try_ne(Tensor::const_(ConstValue::Int(0), self.uop().dtype()))?.flatten()?;

        if numel == 0 {
            return Tensor::empty_zero(DType::Int32).to(self.device()).try_reshape([0, ndim as isize]);
        }
        let index_dtype = DType::Int64;
        let flat_indices = Tensor::arange_with_dtype()
            .start(UOp::const_(index_dtype.clone(), ConstValue::Int(0)))
            .stop(UOp::const_(index_dtype.clone(), ConstValue::Int(numel as i64)))
            .dtype(index_dtype.clone())
            .call()?;
        let selected = flat_indices.masked_select(&mask)?;
        if ndim == 0 {
            return Tensor::empty_zero(DType::Int32).to(self.device()).try_reshape([selected.numel()? as isize, 0]);
        }
        let coords = (0..ndim)
            .map(|axis| {
                let stride = dims[axis + 1..].iter().product::<usize>();
                let mut coord = if stride == 1 { selected.clone() } else { selected.try_div(stride as i64)? };
                // Axis 0 needs no modulo (the division already bounds it); every interior
                // axis does, singleton dims included.
                if axis != 0 {
                    coord = coord.try_mod(dims[axis] as i64)?;
                }
                Ok(coord.cast(DType::Int32))
            })
            .collect::<Result<Vec<_>>>()?;
        Tensor::stack(&coords.iter().collect::<Vec<_>>(), -1)
    }

    /// Reverse the first `sequence_lens[i]` elements along `time_axis` for each
    /// batch element `i` along `batch_axis`, leaving the rest unchanged.
    #[track_caller]
    pub fn reverse_sequence(&self, sequence_lens: &Tensor, time_axis: isize, batch_axis: isize) -> Result<Self> {
        origin_call!("reverse_sequence");
        let dims = svod_ir::shape::to_vec_usize(&self.shape()?).context(UOpSnafu)?;
        let ndim = dims.len();
        let time_axis = Self::normalize_axis(time_axis, ndim)?;
        let batch_axis = Self::normalize_axis(batch_axis, ndim)?;
        let time_len = dims[time_axis];

        // Transpose so time_axis→0, batch_axis→1
        let mut perm: Vec<usize> = (0..ndim).collect();
        perm.swap(0, time_axis);
        let batch_pos = if batch_axis == 0 {
            time_axis
        } else if batch_axis == time_axis {
            0
        } else {
            batch_axis
        };
        perm.swap(1, batch_pos);
        let perm_i: Vec<isize> = perm.iter().map(|&p| p as isize).collect();
        let work = self.try_permute(&perm_i)?;
        let work_dims = svod_ir::shape::to_vec_usize(&work.shape()?).context(UOpSnafu)?;

        // t = arange(T) as [T, 1], seq_lens as [1, B]
        let idx_dt = sequence_lens.uop().dtype();
        let t = Tensor::arange(0, Some(time_len as i64), None)?.cast(idx_dt.clone()).try_unsqueeze(1)?;
        let sl = sequence_lens.try_unsqueeze(0)?;

        // reversed_t = seq_lens - 1 - t; idx = where(t < seq_lens, reversed_t, t)
        let one = Tensor::const_(ConstValue::Int(1), idx_dt);
        let reversed_t = sl.try_sub(&one)?.try_sub(&t)?;
        let mask = t.try_lt(&sl)?;
        let idx = reversed_t.where_(&mask, &t)?;

        // Expand indices to match work shape [T, B, ...] and gather along axis 0
        let expand_shape: Vec<isize> = work_dims.iter().map(|&d| d as isize).collect();
        let idx = idx.try_reshape(&expand_shape[..2])?.try_expand(&expand_shape)?;
        let result = work.gather(0, &idx)?;

        // Inverse permutation to restore original axis order
        let mut inv_perm = vec![0usize; ndim];
        for (i, &p) in perm.iter().enumerate() {
            inv_perm[p] = i;
        }
        let inv_perm_i: Vec<isize> = inv_perm.iter().map(|&p| p as isize).collect();
        result.try_permute(&inv_perm_i)
    }

    // =========================================================================
    // N-dimensional Gather/Scatter (from ONNX GatherND/ScatterND/TensorScatter)
    // =========================================================================

    /// Gather values using N-dimensional indices.
    #[track_caller]
    pub fn gather_nd(&self, indices: &Tensor, batch_dims: usize) -> Result<Tensor> {
        origin_call!("gather_nd");
        let x_shape = self.shape()?;
        let x_dims = svod_ir::shape::to_vec_usize(&x_shape).context(UOpSnafu)?;
        let idx_shape = indices.shape()?;
        let idx_dims = svod_ir::shape::to_vec_usize(&idx_shape).context(UOpSnafu)?;
        let last_idx_dim =
            *idx_dims.last().context(NdimMinimumSnafu { op: "gather_nd", min: 1usize, actual: idx_dims.len() })?;

        if batch_dims == 0 {
            let strides: Vec<i64> =
                (0..last_idx_dim).map(|k| x_dims[k + 1..last_idx_dim].iter().product::<usize>() as i64).collect();
            let inner: usize = x_dims[last_idx_dim..].iter().product();
            let outer = x_dims[..last_idx_dim].iter().product::<usize>();

            let mut flat_idx = Tensor::const_(ConstValue::Int(0), DType::Int64);
            for (k, stride) in strides.iter().enumerate() {
                let mut ranges: Vec<(isize, isize)> = idx_dims.iter().map(|&s| (0, s as isize)).collect();
                ranges[idx_dims.len() - 1] = (k as isize, k as isize + 1);
                let idx_k = indices.try_shrink(&ranges)?.try_squeeze(Some(-1))?;
                let stride_t = Tensor::const_(ConstValue::Int(*stride), DType::Int64);
                flat_idx = flat_idx.try_add(&idx_k.cast(DType::Int64).try_mul(&stride_t)?)?;
            }

            let x_flat = self.try_reshape([outer as isize, inner as isize])?;
            let gather_outer: Vec<isize> = idx_dims[..idx_dims.len() - 1].iter().map(|&d| d as isize).collect();
            let num_gathers: usize = gather_outer.iter().map(|&d| d as usize).product();

            let flat_idx_2d = flat_idx
                .try_reshape([num_gathers as isize, 1])?
                .try_expand([num_gathers as isize, inner as isize])?
                .cast(DType::Int32);
            let result = x_flat.gather(0, &flat_idx_2d)?;

            let mut out_shape = gather_outer;
            for &d in &x_dims[last_idx_dim..] {
                out_shape.push(d as isize);
            }
            result.try_reshape(&out_shape)
        } else {
            let batch_size: usize = x_dims[..batch_dims].iter().product();
            let inner_x: Vec<usize> = x_dims[batch_dims..].to_vec();
            let inner_idx: Vec<usize> = idx_dims[batch_dims..].to_vec();

            let x_flat = self.try_reshape(
                std::iter::once(batch_size as isize).chain(inner_x.iter().map(|&d| d as isize)).collect::<Vec<_>>(),
            )?;
            let idx_flat = indices.try_reshape(
                std::iter::once(batch_size as isize).chain(inner_idx.iter().map(|&d| d as isize)).collect::<Vec<_>>(),
            )?;

            let last_inner = *inner_idx.last().context(NdimMinimumSnafu {
                op: "gather_nd",
                min: 1usize,
                actual: inner_idx.len(),
            })?;
            let strides: Vec<i64> =
                (0..last_inner).map(|k| inner_x[k + 1..last_inner].iter().product::<usize>() as i64).collect();

            let mut flat_idx = Tensor::const_(ConstValue::Int(0), DType::Int64);
            let idx_flat_shape = idx_flat.shape()?;
            let idx_flat_dims = svod_ir::shape::to_vec_usize(&idx_flat_shape).context(UOpSnafu)?;
            for (k, stride) in strides.iter().enumerate() {
                let mut ranges: Vec<(isize, isize)> = idx_flat_dims.iter().map(|&s| (0, s as isize)).collect();
                ranges[idx_flat_dims.len() - 1] = (k as isize, k as isize + 1);
                let idx_k = idx_flat.try_shrink(&ranges)?.try_squeeze(Some(-1))?;
                let stride_t = Tensor::const_(ConstValue::Int(*stride), DType::Int64);
                flat_idx = flat_idx.try_add(&idx_k.cast(DType::Int64).try_mul(&stride_t)?)?;
            }

            let batch_stride = inner_x[..last_inner].iter().product::<usize>();
            let batch_offset_arr =
                Tensor::arange(0, Some(batch_size as i64), None)?.try_mul(Tensor::from_slice([batch_stride as i64]))?;
            let gather_inner = idx_flat_dims[1..idx_flat_dims.len() - 1].iter().product::<usize>();
            flat_idx = flat_idx.try_reshape([batch_size as isize, gather_inner as isize])?;
            let batch_offset = batch_offset_arr
                .try_reshape([batch_size as isize, 1])?
                .try_expand([batch_size as isize, gather_inner as isize])?;
            flat_idx = flat_idx.try_add(&batch_offset)?;

            let remaining: usize = inner_x[last_inner..].iter().product();
            let x_2d = x_flat.try_reshape([(batch_size * batch_stride) as isize, remaining as isize])?;
            let fi = flat_idx
                .try_reshape([(batch_size * gather_inner) as isize, 1])?
                .try_expand([(batch_size * gather_inner) as isize, remaining as isize])?
                .cast(DType::Int32);
            let result = x_2d.gather(0, &fi)?;

            let mut out_shape: Vec<isize> = x_dims[..batch_dims].iter().map(|&d| d as isize).collect();
            out_shape.extend(inner_idx[..inner_idx.len() - 1].iter().map(|&d| d as isize));
            out_shape.extend(inner_x[last_inner..].iter().map(|&d| d as isize));
            result.try_reshape(&out_shape)
        }
    }

    /// Scatter updates into a tensor using N-dimensional indices.
    ///
    /// `reduction` of `None` overwrites; `Some(..)` folds duplicate indices with
    /// that [`ScatterReduction`].
    #[track_caller]
    pub fn scatter_nd(
        &self,
        indices: &Tensor,
        updates: &Tensor,
        reduction: impl Into<Option<ScatterReduction>>,
    ) -> Result<Tensor> {
        origin_call!("scatter_nd");
        let x_shape = self.shape()?;
        let x_dims = svod_ir::shape::to_vec_usize(&x_shape).context(UOpSnafu)?;
        let idx_shape = indices.shape()?;
        snafu::ensure!(!idx_shape.is_empty(), NdimMinimumSnafu { op: "scatter_nd", min: 1usize, actual: 0usize });
        let last_idx_dim = idx_shape[idx_shape.len() - 1]
            .as_const()
            .context(SymbolicShapeUnsupportedSnafu { operation: "scatter_nd" })?;
        let strides: Vec<i64> =
            (0..last_idx_dim).map(|k| x_dims[k + 1..last_idx_dim].iter().product::<usize>() as i64).collect();
        let x_numel: usize = x_dims.iter().product();
        let inner: usize = x_dims[last_idx_dim..].iter().product();
        let outer = x_numel / inner;
        let x_flat = self.try_reshape([outer as isize, inner as isize])?;
        let idx_splits: Vec<Tensor> = (0..last_idx_dim)
            .map(|k| {
                let mut ranges: Vec<(isize, isize)> =
                    idx_shape.iter().map(|s| (0, s.as_const().unwrap() as isize)).collect();
                ranges[idx_shape.len() - 1] = (k as isize, k as isize + 1);
                let slice = indices.try_shrink(&ranges)?;
                slice.try_squeeze(Some(-1))
            })
            .collect::<Result<_>>()?;
        let mut flat_idx = Tensor::const_(ConstValue::Int(0), DType::Int64);
        for (k, idx_k) in idx_splits.iter().enumerate() {
            let stride_t = Tensor::const_(ConstValue::Int(strides[k]), DType::Int64);
            flat_idx = flat_idx.try_add(&idx_k.cast(DType::Int64).try_mul(&stride_t)?)?;
        }
        let upd_shape = updates.shape()?;
        let upd_outer: usize = upd_shape[..upd_shape.len() - (x_dims.len() - last_idx_dim)]
            .iter()
            .map(|s| s.as_const().unwrap())
            .product();
        let upd_flat = updates.try_reshape([upd_outer as isize, inner as isize])?;
        let flat_idx =
            flat_idx.try_reshape([upd_outer as isize, 1])?.try_expand([upd_outer as isize, inner as isize])?;
        let flat_idx_i32 = flat_idx.cast(DType::Int32);
        let result = match reduction.into() {
            None => x_flat.scatter(0, &flat_idx_i32, &upd_flat)?,
            Some(reduce) => x_flat.scatter_reduce(0, &flat_idx_i32, &upd_flat, reduce, true)?,
        };
        let out_shape: Vec<isize> = x_dims.iter().map(|&d| d as isize).collect();
        result.try_reshape(&out_shape)
    }

    /// Batch-aware tensor scatter with write index offsets.
    #[track_caller]
    pub fn tensor_scatter(
        &self,
        update: &Tensor,
        write_indices: Option<&Tensor>,
        mode: &str,
        axis: isize,
    ) -> Result<Tensor> {
        origin_call!("tensor_scatter");
        let data_shape = self.shape()?;
        let ndim = data_shape.len();
        let axis = Self::normalize_axis(axis, ndim)?;
        let data_dims = svod_ir::shape::to_vec_usize(&data_shape).context(UOpSnafu)?;
        let update_dims = svod_ir::shape::to_vec_usize(&update.shape()?).context(UOpSnafu)?;

        let batch_size = data_dims[0];
        let max_seq = data_dims[axis];
        let seq_len = update_dims[axis];

        let b_total: usize = data_dims[..axis].iter().product();
        let features: usize = data_dims[axis + 1..].iter().product();

        let write_idx = if let Some(wi) = write_indices {
            wi.cast(DType::Int32)
        } else {
            Tensor::full(&[batch_size], ConstValue::Int(0), DType::Int32)
        };

        let wi_flat = if axis > 1 {
            let mut wi_reshape: Vec<isize> = vec![batch_size as isize];
            wi_reshape.extend(std::iter::repeat_n(1, axis - 1));
            let wi_expand: Vec<isize> = data_dims[..axis].iter().map(|&d| d as isize).collect();
            write_idx.try_reshape(&wi_reshape)?.try_expand(&wi_expand)?.try_reshape([b_total as isize])?
        } else {
            write_idx
        };

        let data_flat = self.try_reshape([(b_total * max_seq) as isize, features as isize])?;
        let updates_flat = update.try_reshape([(b_total * seq_len) as isize, features as isize])?;

        let batch_offset = Tensor::arange(0, Some(b_total as i64), None)?
            .cast(DType::Int32)
            .try_mul(Tensor::const_(ConstValue::Int(max_seq as i64), DType::Int32))?
            .try_reshape([b_total as isize, 1])?;

        let wi_2d = wi_flat.try_reshape([b_total as isize, 1])?;
        let seq_arange =
            Tensor::arange(0, Some(seq_len as i64), None)?.cast(DType::Int32).try_reshape([1, seq_len as isize])?;
        let mut row_idx = wi_2d.try_add(&seq_arange)?;

        if mode == "circular" {
            let max_seq_t = Tensor::const_(ConstValue::Int(max_seq as i64), DType::Int32);
            row_idx = row_idx.try_mod(&max_seq_t)?;
        }

        let flat_idx = batch_offset
            .try_add(&row_idx)?
            .try_reshape([(b_total * seq_len) as isize, 1])?
            .try_expand([(b_total * seq_len) as isize, features as isize])?;

        let result = data_flat.scatter(0, &flat_idx, &updates_flat)?;

        let out_shape: Vec<isize> = data_dims.iter().map(|&d| d as isize).collect();
        result.try_reshape(&out_shape)
    }
}

/// Reduce repeated indices so the last value wins, then apply mask.
///
/// Tinygrad's `_masked_setitem`: for each axis, split mask/values into slices,
/// fold with OR on mask and last-writer-wins on values, squeeze, then
/// `mask.where(values, target)`.
fn masked_setitem(target: &Tensor, values: &Tensor, mask: &Tensor, axes: &[isize]) -> Result<Tensor> {
    let mut mask = mask.clone();
    let mut values = values.clone();

    // Phase 1: reduce repeated indices — last value wins
    for &dim in axes.iter().rev() {
        let shape = mask.shape()?;
        let ndim = shape.len();
        let norm_dim = Tensor::normalize_axis(dim, ndim)?;
        let dim_size = shape[norm_dim].as_const().unwrap();
        let ones = vec![1usize; dim_size];
        let mask_slices = mask.split(&ones, dim)?;
        let val_slices = values.split(&ones, dim)?;
        let (mut acc_mask, mut acc_vals) = (mask_slices[0].clone(), val_slices[0].clone());
        for (m, v) in mask_slices[1..].iter().zip(&val_slices[1..]) {
            // last-writer-wins: where m is true take v, otherwise keep acc
            acc_vals = v.where_(m, &acc_vals)?;
            acc_mask = acc_mask.try_bitor(m)?;
        }
        mask = acc_mask;
        values = acc_vals;
    }

    // Phase 2: squeeze reduced axes
    for &dim in axes.iter().rev() {
        mask = mask.try_squeeze(Some(dim))?;
        values = values.try_squeeze(Some(dim))?;
    }

    // Phase 3: select from values where mask is true, else target
    values.where_(&mask, target)
}
