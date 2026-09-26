use std::collections::HashMap;
use std::path::Path;

use snafu::{ResultExt, Snafu};
use svod_dtype::DType;
use svod_ir::origin::{self, OriginScope};
use svod_tensor::Tensor;

pub use svod_tensor::nn::{StateDict, get_tensor, prefixed};

#[derive(Debug, Snafu)]
#[snafu(visibility(pub(crate)))]
pub enum Error {
    #[snafu(display("failed to read file: {source}"))]
    Io { source: std::io::Error },
    #[snafu(display("failed to deserialize safetensors"))]
    Safetensors { source: safetensors::SafeTensorError },
    #[snafu(display("unsupported dtype in safetensors: {dtype}"))]
    UnsupportedDtype { dtype: String },
    #[snafu(display("missing key in state dict: {key}"))]
    MissingKey { key: String },
    #[snafu(display("{source}"), context(false))]
    Tensor {
        #[snafu(source(from(svod_tensor::error::Error, Box::new)))]
        source: Box<svod_tensor::error::Error>,
    },
}

type Result<T> = std::result::Result<T, Error>;

pub fn load_safetensors(path: &Path) -> Result<StateDict> {
    // Canonical path = the provenance key: loading the same checkpoint into
    // N model instances shares each weight's immutable device storage
    // instead of duplicating VRAM and upload bandwidth per instance.
    let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let data = std::fs::read(path).context(IoSnafu)?;
    let tensors = safetensors::SafeTensors::deserialize(&data).context(SafetensorsSnafu)?;
    let mut sd = StateDict::new();
    for (name, view) in tensors.tensors() {
        let key = svod_tensor::weight_cache::WeightKey {
            path: canonical.clone(),
            name: name.to_string(),
            dtype: convert_dtype(view.dtype())?,
            shape: view.shape().to_vec(),
            device: svod_dtype::default_device::default_device(),
        };
        let tensor = Tensor::from_shared_weight(key, view.data())?;
        sd.insert(name.to_string(), tensor);
    }
    Ok(sd)
}

/// Load safetensors weights from a HuggingFace checkpoint directory.
///
/// Handles both single-file (`model.safetensors`) and multi-shard
/// (`model-00001-of-0000N.safetensors` + `model.safetensors.index.json`)
/// checkpoints. For single-file, loads `model.safetensors` directly. For
/// multi-shard, parses the index JSON to find all shard files, loads each,
/// and merges into one [`StateDict`].
pub fn load_safetensors_dir(dir: &Path) -> Result<StateDict> {
    let single = dir.join("model.safetensors");
    if single.exists() {
        return load_safetensors(&single);
    }

    let index_path = dir.join("model.safetensors.index.json");
    let index_data = std::fs::read_to_string(&index_path).context(IoSnafu)?;
    let index: SafetensorsIndex =
        serde_json::from_str(&index_data).map_err(|e| Error::Io { source: std::io::Error::other(e.to_string()) })?;

    let mut sd = StateDict::new();
    for shard_file in index.unique_shards() {
        let shard_path = dir.join(&shard_file);
        let shard_sd = load_safetensors(&shard_path)?;
        sd.extend(shard_sd);
    }
    Ok(sd)
}

#[derive(serde::Deserialize)]
pub(crate) struct SafetensorsIndex {
    #[serde(rename = "weight_map")]
    weight_map: HashMap<String, String>,
}

impl SafetensorsIndex {
    pub(crate) fn unique_shards(&self) -> Vec<String> {
        let mut shards: Vec<String> = self.weight_map.values().cloned().collect();
        shards.sort();
        shards.dedup();
        shards
    }
}

fn convert_dtype(dt: safetensors::Dtype) -> Result<DType> {
    use safetensors::Dtype as ST;
    match dt {
        ST::F32 => Ok(DType::Float32),
        ST::F16 => Ok(DType::Float16),
        ST::BF16 => Ok(DType::BFloat16),
        ST::F8_E4M3 => Ok(DType::FP8E4M3),
        ST::F8_E5M2 => Ok(DType::FP8E5M2),
        ST::F64 => Ok(DType::Float64),
        ST::I32 => Ok(DType::Int32),
        ST::I64 => Ok(DType::Int64),
        ST::I16 => Ok(DType::Int16),
        ST::I8 => Ok(DType::Int8),
        ST::U8 => Ok(DType::UInt8),
        ST::BOOL => Ok(DType::Bool),
        other => Err(Error::UnsupportedDtype { dtype: format!("{other:?}") }),
    }
}

/// Cast every tensor in a state dict to `dtype`, leaving any tensor that cannot
/// be cast (e.g. an int embedding key) at its original dtype. Shared by the
/// loaders so weight casting stays in one place.
///
/// The casts are realized here, once: left lazy, every forward re-reads the
/// checkpoint-dtype weight and casts it inside each kernel that reads it.
pub fn cast_all(sd: &StateDict, dtype: DType) -> Result<StateDict> {
    let out: StateDict = sd
        .iter()
        .map(|(k, v)| {
            let t = if v.dtype() == dtype { v.clone() } else { v.cast(dtype.clone()).contiguous() };
            (k.clone(), t)
        })
        .collect();
    Tensor::realize_batch(out.iter().filter(|(k, t)| sd[*k].dtype() != t.dtype()).map(|(_, t)| t))?;
    Ok(out)
}

/// Build a child module's graph under its own origin scope, named by the same
/// segment [`prefixed`] uses for its weights. A module is named by its parent —
/// the only place that knows which slot it fills (`ffn1` vs `ffn2`) — so a
/// rendered origin path is a state-dict key prefix by construction, for a model
/// loaded from a checkpoint and for one built with random weights alike.
///
/// Costs one atomic load while capture is off: the segment is only interned
/// inside [`OriginScope`].
#[inline]
pub fn scoped<T>(name: &str, f: impl FnOnce() -> T) -> T {
    let _origin = OriginScope::module(name);
    f()
}

/// [`scoped`] for one element of a repeated module (`layers.3`, `blocks.11`).
/// The segment is only formatted while capture is on.
#[inline]
pub fn scoped_index<T>(name: &str, index: usize, f: impl FnOnce() -> T) -> T {
    if origin::enabled() { scoped(&format!("{name}.{index}"), f) } else { f() }
}

/// [`scoped_index`] as a guard, for a loop body that cannot be a closure.
/// A no-op scope while capture is off, so the segment is never formatted.
#[inline]
pub fn scope_index(name: &str, index: usize) -> OriginScope {
    if origin::enabled() { OriginScope::module(format!("{name}.{index}")) } else { origin::install(origin::current()) }
}
