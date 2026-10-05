use bon::bon;
use snafu::ResultExt;
use std::sync::Arc;

use svod_device::{Buffer, registry};
use svod_dtype::DType;
use svod_dtype::ext::HasDType;
use svod_ir::{DeviceSpec, SInt, UOp, shape::Shape};

use crate::Tensor;
use crate::error::*;
use crate::tensor_registry;
use svod_dtype::default_device::default_device;

#[bon]
impl Tensor {
    /// Create tensor from slice on the active default device (CPU unless
    /// overridden via `set_default_device` or the `SVOD_DEVICE` env var).
    ///
    /// # Examples
    /// ```
    /// # use svod_tensor::Tensor;
    /// let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
    /// ```
    pub fn from_slice<T: HasDType, C: AsRef<[T]>>(source: C) -> Self {
        let source = source.as_ref();
        Self::from_bytes_shaped(
            unsafe { std::slice::from_raw_parts(source.as_ptr() as *const u8, source.len() * T::DTYPE.bytes()) },
            &[source.len()],
            T::DTYPE,
            default_device(),
        )
    }

    /// Create tensor from slice with explicit device specification using builder pattern.
    #[builder]
    pub fn from_slice_with<T: HasDType, C: AsRef<[T]>>(
        source: C,
        #[builder(default = default_device())] device: DeviceSpec,
    ) -> Self {
        let source = source.as_ref();
        Self::from_bytes_shaped(
            unsafe { std::slice::from_raw_parts(source.as_ptr() as *const u8, source.len() * T::DTYPE.bytes()) },
            &[source.len()],
            T::DTYPE,
            device,
        )
    }
}

impl Tensor {
    /// Core: create a tensor from raw bytes with a known shape.
    ///
    /// Builds the buffer UOp with the target shape directly — no reshape,
    /// so the returned tensor retains its buffer for zero-copy `array_view`.
    /// Routes to whichever allocator the registry returns for `device`; for
    /// AMD this means data is mmapped through the host-visible VRAM aperture
    /// and the GPU sees the buffer directly.
    fn from_bytes_shaped(bytes: &[u8], shape: &[usize], dtype: DType, device: DeviceSpec) -> Self {
        Self::from_bytes_shaped_spec(bytes, shape, dtype, device, Default::default())
    }

    /// `from_bytes_shaped` with an explicit
    /// [`svod_device::BufferSpec`]. `cpu_access: false` keeps the buffer
    /// device-local (no host mapping): the init bytes and any later host
    /// access stage through the backend's copy engine (`copyin`/`copyout`),
    /// and device→device `copy_from` stays on-device. For state buffers the
    /// host shouldn't observe.
    pub fn from_bytes_shaped_spec(
        bytes: &[u8],
        shape: &[usize],
        dtype: DType,
        device: DeviceSpec,
        spec: svod_device::BufferSpec,
    ) -> Self {
        let numel: usize = shape.iter().product();
        let ir_shape = Shape::from_iter(shape.iter().map(|&d| SInt::Const(d)));

        let buffer_uop = UOp::new_buffer(device.clone(), numel, dtype.clone());
        let buffer_uop_id = buffer_uop.id;

        let allocator = registry::registry().get(&device).unwrap_or_else(|e| {
            panic!(
                "Failed to get allocator for {device:?}: {e}\n\
                 Hint: set SVOD_DEVICE=CPU (or unset it) to fall back to the CPU backend."
            )
        });

        let mut buffer = Buffer::new(allocator, dtype.clone(), shape.to_vec(), spec);
        buffer.copyin(bytes).expect("Buffer write always successful");

        let buffer_arc = Arc::new(buffer);
        let uop = buffer_uop.try_reshape(&ir_shape).expect("shape matches element count");

        let entry = tensor_registry::register_tensor_with_buffer(uop, buffer_arc.clone(), buffer_uop_id);
        Self::with_entry(entry)
    }

    /// Weight-loading constructor: weights with the same checkpoint
    /// provenance share ONE immutable device storage across model instances
    /// (see [`crate::weight_cache`]). The tensor still gets its own BUFFER
    /// UOp identity; sharing is at the storage level, so the planner and
    /// `replicate` treat it like any other pre-allocated read-only input.
    pub fn from_shared_weight(key: crate::weight_cache::WeightKey, bytes: &[u8]) -> Result<Self> {
        let numel: usize = key.shape.iter().product();
        let expected = numel * key.dtype.bytes();
        if bytes.len() != expected {
            return Err(ErrorKind::IrConstruction {
                details: format!("from_shared_weight: data length {} != expected {expected}", bytes.len()),
            }
            .into());
        }
        let ir_shape = Shape::from_iter(key.shape.iter().map(|&d| SInt::Const(d)));
        let (device, dtype) = (key.device.clone(), key.dtype.clone());
        let buffer_arc = crate::weight_cache::shared_weight_buffer(key, bytes);
        let buffer_uop = UOp::new_buffer(device, numel, dtype);
        let buffer_uop_id = buffer_uop.id;
        let uop = buffer_uop.try_reshape(&ir_shape).expect("shape matches element count");
        let entry = tensor_registry::register_tensor_with_buffer(uop, buffer_arc.clone(), buffer_uop_id);
        Ok(Self::with_entry(entry))
    }

    /// Create tensor from raw bytes with explicit dtype and shape.
    ///
    /// The bytes are interpreted as little-endian values of the given dtype.
    /// Length must equal `product(shape) * dtype.bytes()`.
    /// Used for types without a native Rust representation (Float16, BFloat16, FP8).
    pub fn from_raw_bytes(data: &[u8], shape: &[usize], dtype: DType) -> Result<Self> {
        let numel: usize = shape.iter().product();
        let expected_bytes = numel * dtype.bytes();
        if data.len() != expected_bytes {
            return Err(ErrorKind::IrConstruction {
                details: format!(
                    "from_raw_bytes: data length {} != expected {} ({} elements * {} bytes)",
                    data.len(),
                    expected_bytes,
                    numel,
                    dtype.bytes()
                ),
            }
            .into());
        }
        Ok(Self::from_bytes_shaped(data, shape, dtype, default_device()))
    }

    /// Create tensor from an ndarray (owned `Array` or `ArrayView`).
    ///
    /// When the array is already C-contiguous, uses the backing slice directly
    /// (no intermediate allocation). Falls back to `.iter().cloned().collect()`
    /// for Fortran-order or non-contiguous layouts.
    ///
    /// # Examples
    /// ```
    /// # use svod_tensor::Tensor;
    /// # use ndarray::array;
    /// let t = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    /// let view = t.array_view::<f32>().unwrap();
    /// assert_eq!(view[[1, 2]], 6.0);
    /// ```
    pub fn from_ndarray<T, S, D>(array: &ndarray::ArrayBase<S, D>) -> Self
    where
        T: HasDType + Clone,
        S: ndarray::Data<Elem = T>,
        D: ndarray::Dimension,
    {
        let shape: Vec<usize> = array.shape().to_vec();
        if array.is_empty() {
            let t = Self::empty_zero(T::DTYPE);
            if shape.len() <= 1 {
                return t;
            }
            let isize_shape: Vec<isize> = shape.iter().map(|&d| d as isize).collect();
            return t.try_reshape(&isize_shape).expect("empty reshape matches");
        }
        // Fast path: C-contiguous — use backing slice directly, no intermediate Vec
        if let Some(slice) = array.as_slice() {
            let bytes =
                unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * T::DTYPE.bytes()) };
            Self::from_bytes_shaped(bytes, &shape, T::DTYPE, default_device())
        } else {
            // Slow path: Fortran-order or non-contiguous — collect in logical order
            let data: Vec<T> = array.iter().cloned().collect();
            let bytes =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * T::DTYPE.bytes()) };
            Self::from_bytes_shaped(bytes, &shape, T::DTYPE, default_device())
        }
    }

    /// Get a reference to the underlying buffer.
    ///
    /// Returns `None` for lazy tensors that haven't been realized yet.
    /// Returns `Some(buffer)` for input tensors and realized tensors.
    pub fn buffer(&self) -> Option<Buffer> {
        // Check the entry first, then the global registry by base UOp ID.
        if let Some(buf) = self.entry.buffer() {
            return Some((**buf).clone());
        }
        crate::tensor_registry::get_buffer_arc(self.uop().base().id).map(|arc| (*arc).clone())
    }

    /// Read realized tensor data as an ndarray.
    ///
    /// The tensor must have a buffer (from `from_slice`, `realize()`, etc.).
    /// Returns error if the tensor has not been realized.
    ///
    /// # Examples
    /// ```
    /// # use svod_tensor::Tensor;
    /// let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
    /// let result = t.as_ndarray::<f32>().unwrap();
    /// assert_eq!(result.shape(), &[3]);
    /// ```
    pub fn as_ndarray<T: HasDType + Default + Clone>(&self) -> Result<ndarray::ArrayD<T>> {
        use ndarray::{ArrayD, IxDyn};

        let uop = self.uop();
        let shape = uop.shape().context(UOpSnafu)?.ok_or(ErrorKind::NoShape)?;

        // Refuse symbolic shapes — matches Tinygrad: assert all_int(self.shape)
        snafu::ensure!(shape.iter().all(SInt::is_const), SymbolicShapeSnafu);
        let dims: Vec<usize> = shape.iter().filter_map(SInt::as_const).collect();

        let data = if dims.contains(&0) { vec![] } else { self.as_vec::<T>()? };
        ArrayD::from_shape_vec(IxDyn(&dims), data).context(NdarrayShapeSnafu).map_err(Into::into)
    }

    /// The tensor whose buffer holds exactly this tensor's logical elements,
    /// contiguously and in row-major order.
    ///
    /// A view (shrink, permute, pad, …) carries no buffer of its own and
    /// resolves to its *base's* buffer, which holds the base's elements — read
    /// directly it yields the wrong count in the wrong order. Anything that is
    /// not a buffer identity is therefore materialized through a contiguous
    /// copy. A tensor with nothing realized upstream is returned untouched so
    /// the caller still reports [`ErrorKind::NoBuffer`] rather than realizing a
    /// lazy graph behind the user's back.
    fn materialized(&self) -> Result<Self> {
        if self.uop().has_buffer_identity() || self.buffer().is_none() {
            return Ok(self.clone());
        }
        let copy = self.contiguous();
        copy.realize()?;
        Ok(copy)
    }

    /// Read realized tensor data as a flat `Vec<T>`.
    ///
    /// The tensor must have a buffer (from `from_slice`, `realize()`, etc.).
    /// Returns error if the tensor has not been realized.
    ///
    /// # Examples
    /// ```
    /// # use svod_tensor::Tensor;
    /// let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
    /// let v = t.as_vec::<f32>().unwrap();
    /// assert_eq!(v, vec![1.0, 2.0, 3.0]);
    /// ```
    pub fn as_vec<T: HasDType + Default + Clone>(&self) -> Result<Vec<T>> {
        if let Ok(Some(shape)) = self.uop().shape() {
            // Refuse symbolic shapes — matches Tinygrad: assert all_int(self.shape)
            snafu::ensure!(shape.iter().all(SInt::is_const), SymbolicShapeSnafu);
            if shape.iter().any(|dim| dim.as_const() == Some(0)) {
                return Ok(vec![]);
            }
        }

        let source = self.materialized()?;
        let buffer = source.buffer().ok_or(ErrorKind::NoBuffer)?;

        if buffer.dtype() != T::DTYPE {
            return TypeMismatchSnafu { expected: T::DTYPE, actual: buffer.dtype() }.fail().map_err(Into::into);
        }

        // The logical element count — the backing allocation may be larger
        // (a `SLICE` alias, or a buffer sized to a symbolic dim's `vmax`).
        let count = source.numel().unwrap_or(buffer.size() / T::DTYPE.bytes());
        let mut data = vec![T::default(); count];
        buffer
            .copyout_prefix(unsafe {
                std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, count * T::DTYPE.bytes())
            })
            .context(DeviceSnafu)?;

        Ok(data)
    }

    /// Realize the graph if nothing behind this tensor has run yet.
    ///
    /// A tensor that already resolves to a buffer — realized, or a view of a
    /// realized base — is left alone, so a read never recompiles.
    fn realized_for_read(&self) -> Result<()> {
        if self.buffer().is_none() { self.realize() } else { Ok(()) }
    }

    /// Read this tensor as a flat `Vec<T>`, realizing it first if needed.
    ///
    /// The auto-realizing counterpart of [`as_vec`](Self::as_vec): use that one
    /// where triggering compilation would be wrong.
    ///
    /// # Examples
    /// ```
    /// # use svod_tensor::Tensor;
    /// let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
    /// assert_eq!((&a + &a).unwrap().to_vec::<f32>().unwrap(), vec![2.0, 4.0, 6.0]);
    /// ```
    pub fn to_vec<T: HasDType + Default + Clone>(&self) -> Result<Vec<T>> {
        self.realized_for_read()?;
        self.as_vec()
    }

    /// Read this tensor as an ndarray, realizing it first if needed.
    ///
    /// The auto-realizing counterpart of [`as_ndarray`](Self::as_ndarray).
    ///
    /// # Examples
    /// ```
    /// # use svod_tensor::Tensor;
    /// # use ndarray::array;
    /// let a = Tensor::from_ndarray(&array![[1.0f32, 2.0], [3.0, 4.0]]);
    /// assert_eq!(a.matmul(&a).unwrap().to_ndarray::<f32>().unwrap(), array![[7.0f32, 10.0], [15.0, 22.0]].into_dyn());
    /// ```
    pub fn to_ndarray<T: HasDType + Default + Clone>(&self) -> Result<ndarray::ArrayD<T>> {
        self.realized_for_read()?;
        self.as_ndarray()
    }

    /// The single element of a one-element tensor, realizing it first if needed.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::ShapeMismatch`] unless the tensor holds exactly one element.
    ///
    /// # Examples
    /// ```
    /// # use svod_tensor::Tensor;
    /// let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
    /// assert_eq!(a.sum(()).unwrap().item::<f32>().unwrap(), 6.0);
    /// ```
    pub fn item<T: HasDType + Default + Clone>(&self) -> Result<T> {
        let numel = self.numel()?;
        snafu::ensure!(
            numel == 1,
            ShapeMismatchSnafu { context: "item", expected: "1 element", actual: format!("{numel} elements") }
        );
        Ok(self.to_vec::<T>()?.pop().expect("a one-element tensor reads back exactly one element"))
    }

    /// Typed immutable view into the buffer, shaped by the tensor's logical shape.
    ///
    /// Uses the tensor's concrete shape for multidimensional indexing.
    /// Falls back to the buffer's flat shape for symbolic tensors.
    ///
    /// # Examples
    /// ```
    /// # use svod_tensor::Tensor;
    /// # use ndarray::array;
    /// let t = Tensor::from_ndarray(&array![[1.0f32, 2.0], [3.0, 4.0]]);
    /// let view = t.array_view::<f32>().unwrap();
    /// assert_eq!(view[[0, 1]], 2.0);
    /// ```
    /// A view (shrink, permute, …) borrows its base's buffer, whose contents are
    /// neither the right elements nor the right order, and a zero-copy borrow
    /// cannot outlive a materialized copy — so only a buffer identity is
    /// viewable; use [`as_ndarray`](Self::as_ndarray) for anything else.
    pub fn array_view<T: HasDType>(&self) -> Result<ndarray::ArrayViewD<'_, T>> {
        let flat = self.storage()?.as_array::<T>().context(DeviceSnafu)?;
        // Reshape to tensor's logical shape if concrete
        if let Ok(shape) = self.shape() {
            let dims: Vec<usize> = shape.iter().filter_map(|d| d.as_const()).collect();
            if dims.len() == shape.len() {
                return flat
                    .into_shape_with_order(ndarray::IxDyn(&dims))
                    .context(NdarrayShapeSnafu)
                    .map_err(Into::into);
            }
        }
        Ok(flat)
    }

    /// Storage behind a buffer identity: this tensor's own, or the one a plan
    /// bound to the identity when a placeholder's dependents were scheduled.
    fn storage(&self) -> Result<&Arc<Buffer>> {
        snafu::ensure!(self.uop().has_buffer_identity(), NoBufferSnafu);
        self.ensure_buffer();
        self.entry.buffer().ok_or_else(|| ErrorKind::NoBuffer.into())
    }

    /// Typed mutable view into the buffer, shaped by the tensor's logical shape.
    ///
    /// # Examples
    /// ```
    /// # use svod_tensor::Tensor;
    /// # use ndarray::array;
    /// let t = Tensor::from_ndarray(&array![[0.0f32, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    /// t.array_view_mut::<f32>().unwrap()[[1, 2]] = 42.0;
    /// assert_eq!(t.array_view::<f32>().unwrap()[[1, 2]], 42.0);
    /// ```
    /// Writable counterpart of [`array_view`](Self::array_view), with the same
    /// buffer-identity requirement — writing through a view would land the
    /// values at the wrong offsets in the base buffer.
    pub fn array_view_mut<T: HasDType>(&self) -> Result<ndarray::ArrayViewMutD<'_, T>> {
        let flat = self.storage()?.as_array_mut::<T>().context(DeviceSnafu)?;
        if let Ok(shape) = self.shape() {
            let dims: Vec<usize> = shape.iter().filter_map(|d| d.as_const()).collect();
            if dims.len() == shape.len() {
                return flat
                    .into_shape_with_order(ndarray::IxDyn(&dims))
                    .context(NdarrayShapeSnafu)
                    .map_err(Into::into);
            }
        }
        Ok(flat)
    }
}
