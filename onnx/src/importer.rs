//! ONNX model importer - converts ONNX protobuf to Svod Tensors.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use prost::Message;
use snafu::ResultExt;
use svod_dtype::DType;
use svod_ir::SInt;
use svod_ir::origin::OriginScope;
use svod_tensor::{Tensor, Variable};

use crate::error::{EmptyModelSnafu, IoSnafu, MissingInputSnafu, ProtobufDecodeSnafu, Result};
use crate::parser::onnx::{GraphProto, ModelProto, NodeProto, ValueInfoProto};
use crate::registry::{OpRegistry, OpSetId, convert_onnx_dtype, onnx_opset_version, tensor_from_proto_ext};

/// Dimension value - either static (known size) or dynamic (named, e.g., batch dim).
#[derive(Debug, Clone, PartialEq)]
pub enum DimValue {
    /// Static dimension with known size
    Static(usize),
    /// Dynamic dimension with symbolic name (e.g., "batch_size")
    Dynamic(String),
}

/// Input specification extracted from ONNX ValueInfoProto.
#[derive(Debug, Clone)]
pub struct InputSpec {
    /// Shape dimensions (can be static or dynamic)
    pub shape: Vec<DimValue>,
    /// Data type
    pub dtype: DType,
    /// Whether this input is optional
    pub optional: bool,
}

impl InputSpec {
    /// Create a new input spec.
    pub fn new(shape: Vec<DimValue>, dtype: DType, optional: bool) -> Self {
        Self { shape, dtype, optional }
    }

    /// Check if all dimensions are static (no dynamic dims).
    pub fn is_static(&self) -> bool {
        self.shape.iter().all(|d| matches!(d, DimValue::Static(_)))
    }

    /// Get static shape if all dimensions are known.
    pub fn static_shape(&self) -> Option<Vec<usize>> {
        self.shape
            .iter()
            .map(|d| match d {
                DimValue::Static(s) => Some(*s),
                DimValue::Dynamic(_) => None,
            })
            .collect()
    }
}

/// A prepared subgraph for control-flow operators (If, Loop, Scan).
#[allow(dead_code)]
pub(crate) struct SubGraph {
    pub initializers: HashMap<String, Tensor>,
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
    pub nodes: Vec<NodeProto>,
    /// Nested subgraph attributes within this subgraph's nodes.
    pub subgraphs: HashMap<(usize, String), SubGraph>,
}

/// Prepared ONNX graph - structure extracted, ready for execution.
pub struct OnnxGraph {
    /// Input specifications (name -> spec)
    pub inputs: HashMap<String, InputSpec>,
    /// Output names in order
    pub outputs: Vec<String>,
    /// Initializers/weights (name -> tensor)
    pub initializers: HashMap<String, Tensor>,
    /// Nodes in topological execution order
    pub nodes: Vec<NodeProto>,
    /// Opset versions
    pub opsets: Vec<OpSetId>,
    /// Pre-parsed subgraph attributes: (node_index, attr_name) -> SubGraph
    pub(crate) subgraphs: HashMap<(usize, String), SubGraph>,
    /// Variables auto-extracted from dynamic dimensions (`dim_param`).
    ///
    /// Key: dim_param name (e.g., "batch"), Value: Variable with bounds `[1, default_max_dim]`.
    /// Populated during `prepare()` by scanning input specs for `DimValue::Dynamic` entries.
    pub variables: HashMap<String, Variable>,
}

impl OnnxGraph {
    /// Get the list of input names.
    pub fn input_names(&self) -> Vec<&str> {
        self.inputs.keys().map(|s| s.as_str()).collect()
    }

    /// Get the list of output names.
    pub fn output_names(&self) -> &[String] {
        &self.outputs
    }

    /// Check if an input is optional.
    pub fn is_input_optional(&self, name: &str) -> bool {
        self.inputs.get(name).map(|s| s.optional).unwrap_or(false)
    }

    /// Get all dynamic dimension names found in the graph.
    pub fn dynamic_dims(&self) -> Vec<&str> {
        self.variables.keys().map(|s| s.as_str()).collect()
    }

    /// Check if the graph has any dynamic dimensions.
    pub fn has_dynamic_dims(&self) -> bool {
        !self.variables.is_empty()
    }
}

/// Imported ONNX model — normal Svod types, ready to use.
///
/// Contains the lazy computation graph as input/output Tensors and any
/// dynamic dimension Variables extracted from the ONNX graph.
///
/// Inputs the caller does not supply are uninitialized buffer-backed
/// placeholders (`Tensor::empty_dynamic`, dynamic dims sized to their max).
/// To run on real data, pass the input tensors to
/// [`OnnxImporter::import_model_with_inputs`].
///
/// # Example
///
/// ```ignore
/// use prost::Message;
/// use svod_onnx::parser::ModelProto;
///
/// let proto = ModelProto::decode(std::fs::read("model.onnx")?.as_slice())?;
/// let x = Tensor::from_slice(load_my_data()).try_reshape([1, 3, 224, 224])?;
/// let model = OnnxImporter::new().import_model_with_inputs(proto, HashMap::from([("x".into(), x)]), &[])?;
///
/// // Outputs are lazy — realize to execute
/// let result = model.outputs["prob"].realize()?;
/// ```
pub struct OnnxModel {
    /// Model inputs: name → caller-supplied Tensor, or an uninitialized
    /// placeholder with the model's shape/dtype.
    pub inputs: HashMap<String, Tensor>,
    /// Model outputs: name → lazy Tensor (realize to execute).
    pub outputs: HashMap<String, Tensor>,
    /// Dynamic dimension variables extracted from `dim_param` annotations.
    /// Empty for static models. Bind to new values for re-execution.
    pub variables: HashMap<String, Variable>,
}

/// ONNX model importer.
///
/// Converts ONNX models to Svod Tensors via a single `import()` call that
/// returns an [`OnnxModel`] with inputs, outputs, and auto-extracted variables.
pub struct OnnxImporter {
    registry: OpRegistry,
    model_dir: Option<std::path::PathBuf>,
    /// Default max bound for auto-extracted dynamic dimension Variables.
    ///
    /// When `prepare()` encounters `dim_param` names in ONNX input shapes,
    /// it creates `Variable::new(name, 1, default_max_dim)`. Default: 32767.
    pub default_max_dim: i64,
}

impl OnnxImporter {
    /// Create a new ONNX importer.
    pub fn new() -> Self {
        Self { registry: OpRegistry::new(), model_dir: None, default_max_dim: i16::MAX as i64 }
    }

    /// Import an ONNX model from a file path.
    ///
    /// Dynamic dimensions are bound via `dim_bindings`; unmentioned dynamic
    /// dims use their max value for buffer allocation.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let model = OnnxImporter::new().import("model.onnx", &[("batch", 4)])?;
    /// let result = model.outputs["output"].realize()?;
    /// ```
    pub fn import(&mut self, path: impl AsRef<Path>, dim_bindings: &[(&str, i64)]) -> Result<OnnxModel> {
        let path = path.as_ref();
        self.model_dir = path.parent().map(|p| p.to_path_buf());
        let file = File::open(path).context(IoSnafu)?;
        let mut reader = BufReader::new(file);
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).context(IoSnafu)?;
        // Decode from Bytes (not &[u8]) so raw_data fields are zero-copy sub-slices.
        // This enables computing byte offsets for DISK-backed weight loading.
        let file_bytes = prost::bytes::Bytes::from(bytes);
        let base_ptr = file_bytes.as_ptr();
        let model = ModelProto::decode(file_bytes).context(ProtobufDecodeSnafu)?;
        self.import_model_with_disk(model, dim_bindings, Some(path), base_ptr)
    }

    /// Import from a parsed `ModelProto` (no DISK-backed loading).
    pub fn import_model(&self, model: ModelProto, dim_bindings: &[(&str, i64)]) -> Result<OnnxModel> {
        self.import_model_with_disk(model, dim_bindings, None, std::ptr::null())
    }

    /// Import from a parsed `ModelProto` with optional DISK-backed weight loading.
    /// When `model_path` is Some, raw_data initializers become lazy DISK views.
    fn import_model_with_disk(
        &self,
        model: ModelProto,
        dim_bindings: &[(&str, i64)],
        model_path: Option<&Path>,
        base_ptr: *const u8,
    ) -> Result<OnnxModel> {
        let bindings: HashMap<String, i64> = dim_bindings.iter().map(|&(k, v)| (k.to_string(), v)).collect();
        let graph = self.prepare_with_disk(model, model_path, base_ptr)?;
        let inputs = resolve_symbolic_shapes(&graph.inputs, &graph.variables, &bindings)?;
        let outputs = self.trace_graph(&graph, inputs.clone())?;
        Ok(OnnxModel { inputs, outputs, variables: graph.variables })
    }

    /// Import with pre-built input tensors that override auto-resolved empty ones.
    ///
    /// Used for node conformance tests where inputs carry concrete data
    /// that ops need at trace time (shape parameters, indices, etc.).
    pub fn import_model_with_inputs(
        &self,
        model: ModelProto,
        inputs: HashMap<String, Tensor>,
        dim_bindings: &[(&str, i64)],
    ) -> Result<OnnxModel> {
        let bindings: HashMap<String, i64> = dim_bindings.iter().map(|&(k, v)| (k.to_string(), v)).collect();
        let graph = self.prepare(model)?;
        let unresolved: HashMap<String, InputSpec> = graph
            .inputs
            .iter()
            .filter(|(name, _)| !inputs.contains_key(*name))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let resolved = resolve_symbolic_shapes(&unresolved, &graph.variables, &bindings)?;
        let mut all_inputs = resolved;
        all_inputs.extend(inputs);
        let input_map = all_inputs.clone();
        let outputs = self.trace_graph(&graph, all_inputs)?;
        Ok(OnnxModel { inputs: input_map, outputs, variables: graph.variables })
    }

    /// Extract graph structure from ONNX model (internal — used by `import_model`).
    pub(crate) fn prepare(&self, model: ModelProto) -> Result<OnnxGraph> {
        self.prepare_with_disk(model, None, std::ptr::null())
    }

    fn prepare_with_disk(
        &self,
        model: ModelProto,
        model_path: Option<&Path>,
        base_ptr: *const u8,
    ) -> Result<OnnxGraph> {
        let proto_graph = model.graph.ok_or_else(|| EmptyModelSnafu.build())?;

        // Collect opsets
        let opsets: Vec<OpSetId> =
            model.opset_import.iter().map(|op| OpSetId { domain: op.domain.clone(), version: op.version }).collect();

        // Build initializer map (weights/constants).
        // When model_path is available: create a DISK-backed tensor from the file, then each
        // raw_data initializer becomes a lazy SHRINK → BITCAST → RESHAPE → COPY(CPU) chain.
        // All weights share ONE DISK buffer (matching Tinygrad's approach).
        // When model_path is None: fall back to eager loading via tensor_from_proto_ext.
        let mut initializers: HashMap<String, Tensor> = HashMap::new();
        let initializer_names: std::collections::HashSet<String> =
            proto_graph.initializer.iter().map(|i| i.name.clone()).collect();

        // Weights are built outside any node, so they get their own root scope.
        let initializer_scope = OriginScope::label("initializer");

        // Create DISK tensor if we have a file path
        let disk_tensor = model_path.map(|p| Tensor::from_path(p).expect("DISK tensor creation"));

        for init in &proto_graph.initializer {
            if init.name.is_empty() {
                continue;
            }
            if init.data_location == 1 {
                let tensor = tensor_from_proto_ext(init, self.model_dir.as_deref())?;
                initializers.insert(init.name.clone(), const_fold_scalar(tensor));
                continue;
            }

            // DISK path: raw_data is a Bytes sub-slice of the file buffer.
            // Compute byte offset via pointer arithmetic, create lazy view chain.
            let dtype = convert_onnx_dtype(init.data_type)?;
            let can_use_disk =
                disk_tensor.is_some() && !init.raw_data.is_empty() && !base_ptr.is_null() && dtype.is_float(); // INT64 shape tensors go eager (no cross-size bitcast in codegen yet)

            if can_use_disk {
                let dims: Vec<usize> = init.dims.iter().map(|&d| d as usize).collect();
                let numel: usize = dims.iter().product();

                // Scalars: eager path for const-folding (const_fold_scalar needs buffer)
                if numel <= 1 {
                    let tensor = tensor_from_proto_ext(init, self.model_dir.as_deref())?;
                    initializers.insert(init.name.clone(), const_fold_scalar(tensor));
                    continue;
                }

                let byte_offset = init.raw_data.as_ptr() as usize - base_ptr as usize;
                let byte_len = init.raw_data.len();
                let disk_uop = disk_tensor.as_ref().unwrap().uop();

                // SHRINK(uint8 byte range) → BITCAST(target dtype) → RESHAPE(dims) → COPY(DEFAULT)
                // Matches Tinygrad: data.bitcast(true_dtype).reshape(shape).to(Device.DEFAULT)
                // Previously hard-coded `DeviceSpec::Cpu` — that forced every
                // ONNX initializer onto CPU regardless of `SVOD_DEVICE`, so
                // any non-CPU runtime got cross-device kernels (inputs on
                // e.g. AMD, weights on CPU) and rangeify failed with
                // "normal kernel buffers must be on the same device".
                let view = disk_uop
                    .try_shrink(&[(SInt::from(byte_offset), SInt::from(byte_offset + byte_len))])
                    .map_err(|e| crate::error::Error::IrConstruction {
                        details: format!("DISK shrink '{}': {e}", init.name),
                    })?;
                let bitcasted = view.bitcast(dtype);
                let ir_dims: smallvec::SmallVec<[SInt; 4]> = dims.iter().map(|&d| SInt::from(d)).collect();
                let reshaped = bitcasted.try_reshape(&ir_dims).map_err(|e| crate::error::Error::IrConstruction {
                    details: format!("DISK reshape '{}': {e}", init.name),
                })?;
                let copied = reshaped.copy_to_device(svod_dtype::default_device::default_device());
                let tensor = Tensor::from_lazy(copied);
                initializers.insert(init.name.clone(), tensor);
            } else {
                let tensor = tensor_from_proto_ext(init, self.model_dir.as_deref())?;
                initializers.insert(init.name.clone(), const_fold_scalar(tensor));
            }
        }
        drop(initializer_scope);

        // Extract input specs (excluding initializers)
        let mut inputs: HashMap<String, InputSpec> = HashMap::new();
        for input in &proto_graph.input {
            if input.name.is_empty() {
                continue;
            }
            // Skip if this input is an initializer
            if initializer_names.contains(&input.name) {
                continue;
            }
            // Extract input spec from ValueInfoProto
            if let Some(spec) = self.extract_input_spec(input)? {
                inputs.insert(input.name.clone(), spec);
            }
        }

        // Collect output names
        let outputs: Vec<String> =
            proto_graph.output.iter().filter(|o| !o.name.is_empty()).map(|o| o.name.clone()).collect();

        // Collect nodes
        let nodes = proto_graph.node;

        // Pre-parse subgraph attributes for control-flow ops (If, Loop, Scan)
        let mut subgraphs: HashMap<(usize, String), SubGraph> = HashMap::new();
        for (idx, node) in nodes.iter().enumerate() {
            for attr in &node.attribute {
                if let Some(ref graph) = attr.g {
                    let sub = self.prepare_subgraph(graph)?;
                    subgraphs.insert((idx, attr.name.clone()), sub);
                }
            }
        }

        // Auto-extract Variables from dynamic dimensions (dim_param names)
        let mut variables: HashMap<String, Variable> = HashMap::new();
        for spec in inputs.values() {
            for dim in &spec.shape {
                if let DimValue::Dynamic(name) = dim
                    && !name.is_empty()
                {
                    variables.entry(name.clone()).or_insert_with(|| Variable::new(name, 1, self.default_max_dim));
                }
            }
        }

        Ok(OnnxGraph { inputs, outputs, initializers, nodes, opsets, subgraphs, variables })
    }

    /// Extract InputSpec from ValueInfoProto.
    fn extract_input_spec(&self, input: &ValueInfoProto) -> Result<Option<InputSpec>> {
        // ValueInfoProto has a `type` field (TypeProto) - use r#type for keyword
        // TypeProto has a oneof `value` containing tensor_type, sequence_type, etc.
        use crate::parser::onnx::tensor_shape_proto::dimension::Value as DimValueProto;
        use crate::parser::onnx::type_proto::Value;

        let type_proto = match input.r#type.as_ref() {
            Some(t) => t,
            None => return Ok(None),
        };

        // Access the oneof value
        let tensor_type = match &type_proto.value {
            Some(Value::TensorType(tt)) => tt,
            _ => return Ok(None), // Not a tensor type
        };

        // Extract dtype
        let dtype = convert_onnx_dtype(tensor_type.elem_type)?;

        // Extract shape from tensor_type.shape
        let shape: Vec<DimValue> = tensor_type
            .shape
            .as_ref()
            .map(|s| {
                s.dim
                    .iter()
                    .map(|d| match &d.value {
                        Some(DimValueProto::DimValue(v)) => {
                            if *v > 0 {
                                DimValue::Static(*v as usize)
                            } else {
                                DimValue::Dynamic(String::new())
                            }
                        }
                        Some(DimValueProto::DimParam(name)) => DimValue::Dynamic(name.clone()),
                        None => DimValue::Dynamic(String::new()),
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(Some(InputSpec::new(shape, dtype, false)))
    }

    /// Build the lazy UOp graph by walking ONNX nodes with initializers + inputs.
    fn trace_graph(&self, graph: &OnnxGraph, inputs: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
        // Merge initializers into values
        let mut values: HashMap<String, Tensor> = graph.initializers.clone();

        // Add user inputs
        for (name, tensor) in inputs {
            values.insert(name, tensor);
        }

        // Validate required inputs are present
        for (name, spec) in &graph.inputs {
            if !values.contains_key(name) && !spec.optional {
                return MissingInputSnafu { node: "graph_input".to_string(), input: name.clone() }.fail();
            }
        }

        // Track names of constant tensors (initializers + Constant node outputs).
        // Used for Gather fast path: when indices are known constants, use shrink+cat
        // instead of full index_select (which creates arange + one-hot + reduce kernels).
        // Matches Tinygrad's `self.const_names` (onnx.py:394).
        let mut const_names: HashSet<String> = graph.initializers.keys().cloned().collect();
        for node in &graph.nodes {
            if node.op_type == "Constant" {
                for out in &node.output {
                    const_names.insert(out.clone());
                }
            }
        }

        // Resolve the default (ai.onnx) opset version
        let opset_version = onnx_opset_version(&graph.opsets, "");

        // Process nodes in order (ONNX guarantees topological order)
        for (node_index, node) in graph.nodes.iter().enumerate() {
            let node_opset = if node.domain.is_empty() || node.domain == "ai.onnx" {
                opset_version
            } else {
                onnx_opset_version(&graph.opsets, &node.domain)
            };

            let _span = tracing::debug_span!("onnx_node", idx = node_index, op = %node.op_type).entered();
            let _origin = node_origin(node_index, node, node_opset);

            // Gather fast path: when indices are from initializers/Constant nodes,
            // use shrink+cat instead of index_select. Creates zero kernels vs 2-5
            // kernels per Gather. Port of Tinygrad onnx.py:468-471,1148-1158.
            if node.op_type == "Gather"
                && node.input.len() > 1
                && const_names.contains(&node.input[1])
                && let Some(result) = self.try_gather_fast_path(node, &values)?
            {
                for (i, output_name) in node.output.iter().enumerate() {
                    if i == 0 && !output_name.is_empty() {
                        values.insert(output_name.clone(), result.clone());
                    }
                }
                continue;
            }

            if node.op_type == "If" {
                self.process_if_node(node_index, node, &mut values, node_opset, &graph.subgraphs)?;
            } else {
                self.process_node(node, &mut values, node_opset)?;
            }

            // Shape op outputs are always const (shape of concrete tensor is deterministic)
            if node.op_type == "Shape" {
                for out in &node.output {
                    if !out.is_empty() {
                        const_names.insert(out.clone());
                    }
                }
            }

            // At trace level: realize each output and dump first values.
            // Intrusive (breaks fusion) — use for numerical bisection only.
            // RUST_LOG=svod_onnx::importer=trace to enable.
            #[allow(clippy::result_large_err)]
            if tracing::enabled!(tracing::Level::TRACE) {
                for out_name in &node.output {
                    if let Some(tensor) = values.get_mut(out_name) {
                        match tensor.realize().and_then(|()| tensor.as_vec::<f32>()) {
                            Ok(data) => {
                                let first5: Vec<f32> = data.iter().take(5).copied().collect();
                                let shape = tensor.shape().unwrap_or_default();
                                tracing::trace!(%out_name, ?shape, ?first5, "node output");
                            }
                            _ => tracing::trace!(%out_name, "node output (non-f32)"),
                        }
                    }
                }
            }
        }

        // Collect outputs by name
        let outputs: HashMap<String, Tensor> =
            graph.outputs.iter().filter_map(|name| values.get(name).cloned().map(|t| (name.clone(), t))).collect();

        Ok(outputs)
    }

    /// Prepare a subgraph from a GraphProto embedded in a node attribute.
    /// Recursively extracts nested subgraph attributes (e.g., If inside If).
    fn prepare_subgraph(&self, graph: &GraphProto) -> Result<SubGraph> {
        let initializer_scope = OriginScope::label("initializer");
        let mut initializers: HashMap<String, Tensor> = HashMap::new();
        for init in &graph.initializer {
            if !init.name.is_empty() {
                let tensor = tensor_from_proto_ext(init, self.model_dir.as_deref())?;
                initializers.insert(init.name.clone(), tensor);
            }
        }
        drop(initializer_scope);

        let input_names: Vec<String> = graph
            .input
            .iter()
            .filter(|i| !i.name.is_empty() && !initializers.contains_key(&i.name))
            .map(|i| i.name.clone())
            .collect();

        let output_names: Vec<String> =
            graph.output.iter().filter(|o| !o.name.is_empty()).map(|o| o.name.clone()).collect();

        let nodes = graph.node.clone();

        // Recursively extract nested subgraph attributes
        let mut subgraphs: HashMap<(usize, String), SubGraph> = HashMap::new();
        for (idx, node) in nodes.iter().enumerate() {
            for attr in &node.attribute {
                if let Some(ref nested_graph) = attr.g {
                    let sub = self.prepare_subgraph(nested_graph)?;
                    subgraphs.insert((idx, attr.name.clone()), sub);
                }
            }
        }

        Ok(SubGraph { initializers, input_names, output_names, nodes, subgraphs })
    }

    /// Execute a pre-parsed subgraph with access to parent scope values.
    fn execute_subgraph(
        &self,
        subgraph: &SubGraph,
        branch: &str,
        parent_values: &HashMap<String, Tensor>,
        opset_version: i64,
    ) -> Result<Vec<Tensor>> {
        // Both branches of an `If` end up in the merged graph, so the attribute name
        // is what separates them under the enclosing node's origin.
        let _origin = OriginScope::label(branch);

        // Start with subgraph's own initializers
        let mut values: HashMap<String, Tensor> = subgraph.initializers.clone();

        // Merge parent scope (subgraph initializers take precedence)
        for (k, v) in parent_values {
            values.entry(k.clone()).or_insert_with(|| v.clone());
        }

        // Execute subgraph nodes (subgraphs inherit the parent's opset).
        // Control-flow ops (If) are intercepted here, matching trace_graph.
        for (node_index, node) in subgraph.nodes.iter().enumerate() {
            let _origin = node_origin(node_index, node, opset_version);
            if node.op_type == "If" {
                self.process_if_node(node_index, node, &mut values, opset_version, &subgraph.subgraphs)?;
            } else {
                self.process_node(node, &mut values, opset_version)?;
            }
        }

        // Collect declared outputs in order
        subgraph
            .output_names
            .iter()
            .map(|name| {
                values.get(name).cloned().ok_or_else(|| crate::Error::IrConstruction {
                    details: format!("subgraph output '{name}' not found in values"),
                })
            })
            .collect()
    }

    /// Process an ONNX If node: execute both branches and merge with `where_()`.
    ///
    /// Both branches are always executed so the resulting graph is data-dependent
    /// on the condition rather than eagerly resolved. This enables the
    /// trace-once / run-many pattern where the condition changes at runtime.
    ///
    /// If the branches produce incompatible shapes or dtypes, the node errors.
    /// Models with legitimately incompatible branches (e.g., AffineGrid expanded)
    /// are already skipped via `SKIP_CONTAINS: ["_expanded"]`.
    fn process_if_node(
        &self,
        node_index: usize,
        node: &NodeProto,
        values: &mut HashMap<String, Tensor>,
        opset_version: i64,
        subgraphs: &HashMap<(usize, String), SubGraph>,
    ) -> Result<()> {
        let condition = values
            .get(&node.input[0])
            .ok_or_else(|| crate::Error::MissingInput { node: node.name.clone(), input: node.input[0].clone() })?
            .clone();

        let then_branch = subgraphs
            .get(&(node_index, "then_branch".to_string()))
            .ok_or_else(|| crate::Error::IrConstruction { details: "If node missing then_branch attribute".into() })?;
        let else_branch = subgraphs
            .get(&(node_index, "else_branch".to_string()))
            .ok_or_else(|| crate::Error::IrConstruction { details: "If node missing else_branch attribute".into() })?;

        let then_outputs = self.execute_subgraph(then_branch, "then_branch", values, opset_version)?;
        let else_outputs = self.execute_subgraph(else_branch, "else_branch", values, opset_version)?;

        for (i, output_name) in node.output.iter().enumerate() {
            if output_name.is_empty() {
                continue;
            }
            let then_out = then_outputs.get(i).ok_or_else(|| crate::Error::IrConstruction {
                details: format!("If node output {i}: then_branch missing output"),
            })?;
            let else_out = else_outputs.get(i).ok_or_else(|| crate::Error::IrConstruction {
                details: format!("If node output {i}: else_branch missing output"),
            })?;

            let (then_shape, else_shape) = (then_out.shape()?, else_out.shape()?);
            if then_shape != else_shape || then_out.dtype() != else_out.dtype() {
                return Err(crate::Error::IrConstruction {
                    details: format!(
                        "If node output {i}: incompatible branches: then={then_shape:?}/{:?}, else={else_shape:?}/{:?}",
                        then_out.dtype(),
                        else_out.dtype(),
                    ),
                });
            }

            let merged = then_out.where_(&condition, else_out)?;
            values.insert(output_name.clone(), merged);
        }

        Ok(())
    }

    /// Gather fast path: when indices are from initializers/Constant nodes, use
    /// shrink+cat instead of index_select (arange + one-hot + reduce).
    /// Port of Tinygrad onnx.py:468-471,1148-1158.
    fn try_gather_fast_path(&self, node: &NodeProto, values: &HashMap<String, Tensor>) -> Result<Option<Tensor>> {
        use crate::registry::attr::{Attrs, tensor_to_i64_vec};

        let data = match values.get(&node.input[0]) {
            Some(t) => t.clone(),
            None => return Ok(None),
        };
        let idx_tensor = match values.get(&node.input[1]) {
            Some(t) => t,
            None => return Ok(None),
        };

        let mut attrs = Attrs::new(node);
        let axis = attrs.int("axis", 0) as isize;

        let indices = match tensor_to_i64_vec(idx_tensor) {
            Ok(v) => v,
            Err(_) => return Ok(None), // fall back to normal path
        };
        // Symbolic index shape falls back to the normal path.
        let Ok(idx_shape) = idx_tensor.dims() else { return Ok(None) };

        let result = crate::registry::gather_const_fast_path(&data, &indices, &idx_shape, axis)?;
        Ok(Some(result))
    }

    /// Process a single ONNX node.
    fn process_node(&self, node: &NodeProto, values: &mut HashMap<String, Tensor>, opset_version: i64) -> Result<()> {
        let op_type = &node.op_type;
        let domain = &node.domain;
        let node_name = if node.name.is_empty() { "unnamed" } else { &node.name };

        // Collect input Tensors, preserving positional indices for optional inputs.
        // Empty input names in ONNX mean "optional, not provided" — we represent
        // these as None to keep correct positional indexing for operators like Clip.
        let mut inputs: Vec<Option<Tensor>> = Vec::new();
        for input_name in &node.input {
            if input_name.is_empty() {
                inputs.push(None);
            } else {
                match values.get(input_name) {
                    Some(tensor) => inputs.push(Some(tensor.clone())),
                    None => {
                        return Err(crate::Error::MissingInput {
                            node: node_name.to_string(),
                            input: input_name.clone(),
                        });
                    }
                }
            }
        }

        // Dispatch to operator registry - may return multiple outputs
        let outputs = self.registry.dispatch_multi(op_type, domain, &inputs, node, opset_version)?;

        // Register outputs by name
        for (i, output_name) in node.output.iter().enumerate() {
            if let Some(output_tensor) = outputs.get(i) {
                values.insert(output_name.clone(), output_tensor.clone());
            }
        }

        Ok(())
    }
}

impl Default for OnnxImporter {
    fn default() -> Self {
        Self::new()
    }
}

/// Origin scope for one ONNX node: the index into its own `graph.node` plus the
/// identity a profile row needs. Costs one atomic load while capture is off.
fn node_origin(index: usize, node: &NodeProto, opset: i64) -> OriginScope {
    OriginScope::onnx(
        index as u32,
        (!node.name.is_empty()).then_some(node.name.as_str()),
        &node.op_type,
        &node.domain,
        opset,
    )
}

/// Const-fold a scalar tensor (shape `()`) into a CONST UOp.
///
/// When a tensor has exactly 1 element and a buffer, extract the scalar value
/// and return `Tensor::const_()` instead. This makes the value inlineable into
/// compute kernels without a separate buffer load.
/// Matches Tinygrad onnx.py:254-256: `data = Tensor(data.item(), dtype=to_dtype)`.
fn const_fold_scalar(tensor: Tensor) -> Tensor {
    // Only fold true scalars with empty shape
    let shape = match tensor.shape() {
        Ok(s) => s,
        Err(_) => return tensor,
    };
    if !shape.is_empty() {
        return tensor;
    }
    let buf = match tensor.buffer() {
        Some(b) => b,
        None => return tensor,
    };
    let dtype = tensor.dtype();
    let bytes_needed = dtype.bytes();
    let mut raw = vec![0u8; bytes_needed];
    if buf.copyout(&mut raw).is_err() {
        return tensor;
    }
    match crate::registry::proto::extract_scalar_const(&raw, &dtype) {
        Ok(cv) => Tensor::const_(cv, dtype),
        Err(_) => tensor,
    }
}

/// Resolve input shapes using Variables for dynamic dimensions.
///
/// Static dims become `SInt::Const`. Dynamic dims are resolved via the
/// auto-extracted `variables` map: bound variables use their concrete value,
/// unbound variables use the `Variable` directly (allocates to max).
fn resolve_symbolic_shapes(
    specs: &HashMap<String, InputSpec>,
    variables: &HashMap<String, Variable>,
    bindings: &HashMap<String, i64>,
) -> Result<HashMap<String, Tensor>> {
    let _origin = OriginScope::label("input");
    specs
        .iter()
        .filter(|(_, spec)| !spec.optional)
        .map(|(name, spec)| {
            let shape: Vec<SInt> = spec
                .shape
                .iter()
                .map(|d| match d {
                    DimValue::Static(s) => Ok(SInt::from(*s)),
                    DimValue::Dynamic(dim_name) if dim_name.is_empty() => {
                        // Unnamed dynamic dim — treat as 1 (common for unknown batch)
                        Ok(SInt::from(1usize))
                    }
                    DimValue::Dynamic(dim_name) => {
                        let var = variables.get(dim_name).ok_or_else(|| crate::Error::IrConstruction {
                            details: format!("no Variable for dynamic dim '{dim_name}' in input '{name}'"),
                        })?;
                        if let Some(&val) = bindings.get(dim_name) {
                            // Concrete binding: use fixed value (no symbolic Variable)
                            let _ = var.bind(val).map_err(|e| crate::Error::IrConstruction {
                                details: format!("binding '{dim_name}' = {val} out of range: {e}"),
                            })?; // validate bounds
                            Ok(SInt::from(val as usize))
                        } else {
                            // Unbound: use variable directly (allocates to max)
                            Ok(var.as_sint())
                        }
                    }
                })
                .collect::<Result<_>>()?;
            let tensor = Tensor::empty_dynamic(&shape, spec.dtype.clone());
            Ok((name.clone(), tensor))
        })
        .collect()
}
