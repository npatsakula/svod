use crate::importer::{DimValue, InputSpec, OnnxImporter};
use crate::parser::onnx::type_proto::{self, Tensor as TensorTypeProto};
use crate::parser::onnx::{GraphProto, ModelProto, NodeProto, TensorProto, ValueInfoProto, tensor_proto};
use crate::registry::OpRegistry;
use crate::test::helpers::*;
use ndarray::{Array2, Array3, Array4, array};

// =========================================================================
// Non-codegen tests (no realize/to_vec needed)
// =========================================================================

#[test]
fn test_importer_creation() {
    let _importer = OnnxImporter::new();
}

#[test]
fn test_prepare_minimal_model() {
    let importer = OnnxImporter::new();
    let model = make_minimal_model();
    let result = importer.import_model(model, &[]).unwrap();

    // Minimal model: all inputs are initializers, so no user inputs
    assert!(result.inputs.is_empty());
    assert_eq!(result.outputs.len(), 1);
    assert!(result.outputs.contains_key("output"));
}

#[test]
fn test_import_model_minimal() {
    let importer = OnnxImporter::new();
    let model = make_minimal_model();
    let outputs = importer.import_model(model, &[]).unwrap().outputs;

    assert_eq!(outputs.len(), 1);
    assert!(outputs.contains_key("output"));
}

#[test]
fn test_multi_output_model() {
    let importer = OnnxImporter::new();
    let model = make_multi_output_model();
    let outputs = importer.import_model(model, &[]).unwrap().outputs;

    assert_eq!(outputs.len(), 2);
    assert!(outputs.contains_key("out1"));
    assert!(outputs.contains_key("out2"));
}

#[test]
fn test_import_empty_model() {
    let importer = OnnxImporter::new();
    let model = ModelProto::default();
    let result = importer.import_model(model, &[]);
    assert!(result.is_err());
}

#[test]
fn test_import_with_add() {
    let importer = OnnxImporter::new();

    let mut model = ModelProto::default();
    let mut graph = GraphProto::default();
    graph.name = "add_test".to_string();

    for (name, values) in [("a", vec![1.0f32, 2.0, 3.0]), ("b", vec![4.0f32, 5.0, 6.0])] {
        let mut input = ValueInfoProto::default();
        input.name = name.to_string();
        graph.input.push(input);

        let mut init = TensorProto::default();
        init.name = name.to_string();
        init.data_type = tensor_proto::DataType::Float as i32;
        init.dims = vec![3];
        init.raw_data = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        graph.initializer.push(init);
    }

    let mut output = ValueInfoProto::default();
    output.name = "output".to_string();
    graph.output.push(output);

    let mut node = NodeProto::default();
    node.op_type = "Add".to_string();
    node.input.push("a".to_string());
    node.input.push("b".to_string());
    node.output.push("output".to_string());
    graph.node.push(node);

    model.graph = Some(graph);

    let outputs = importer.import_model(model, &[]).unwrap().outputs;
    assert_eq!(outputs.len(), 1);
    assert!(outputs.contains_key("output"));
}

#[test]
fn test_import_with_matmul() {
    let importer = OnnxImporter::new();

    let mut model = ModelProto::default();
    let mut graph = GraphProto::default();
    graph.name = "matmul_test".to_string();

    for (name, values) in [("a", vec![1.0f32, 2.0, 3.0, 4.0]), ("b", vec![5.0f32, 6.0, 7.0, 8.0])] {
        let mut input = ValueInfoProto::default();
        input.name = name.to_string();
        graph.input.push(input);

        let mut init = TensorProto::default();
        init.name = name.to_string();
        init.data_type = tensor_proto::DataType::Float as i32;
        init.dims = vec![2, 2];
        init.raw_data = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        graph.initializer.push(init);
    }

    let mut output = ValueInfoProto::default();
    output.name = "output".to_string();
    graph.output.push(output);

    let mut node = NodeProto::default();
    node.op_type = "MatMul".to_string();
    node.input.push("a".to_string());
    node.input.push("b".to_string());
    node.output.push("output".to_string());
    graph.node.push(node);

    model.graph = Some(graph);

    let outputs = importer.import_model(model, &[]).unwrap().outputs;
    assert_eq!(outputs.len(), 1);
    assert!(outputs.contains_key("output"));
}

#[test]
fn test_input_spec_static_shape() {
    let spec = InputSpec::new(
        vec![DimValue::Static(2), DimValue::Static(3)],
        DType::Scalar(svod_dtype::ScalarDType::Float32),
        false,
    );

    assert!(spec.is_static());
    assert_eq!(spec.static_shape(), Some(vec![2, 3]));
}

#[test]
fn test_input_spec_dynamic_shape() {
    let spec = InputSpec::new(
        vec![DimValue::Dynamic("batch".to_string()), DimValue::Static(3)],
        DType::Scalar(svod_dtype::ScalarDType::Float32),
        false,
    );

    assert!(!spec.is_static());
    assert_eq!(spec.static_shape(), None);
}

#[test]
fn test_if_shape_mismatch_errors() {
    let mut model = ModelProto::default();
    let mut graph = GraphProto::default();
    graph.name = "if_mismatch_test".to_string();

    let mut cond_input = ValueInfoProto::default();
    cond_input.name = "condition".to_string();
    graph.input.push(cond_input);
    let mut cond_init = TensorProto::default();
    cond_init.name = "condition".to_string();
    cond_init.data_type = tensor_proto::DataType::Bool as i32;
    cond_init.dims = vec![];
    cond_init.raw_data = prost::bytes::Bytes::from(vec![1u8]);
    graph.initializer.push(cond_init);

    let then_const = {
        let mut n = NodeProto::default();
        n.op_type = "Constant".to_string();
        n.output.push("then_out".to_string());
        n.attribute.push(make_attr_tensor(
            "value",
            make_tensor_proto(
                [1.0f32, 2.0, 3.0].iter().flat_map(|v| v.to_le_bytes()).collect(),
                vec![3],
                tensor_proto::DataType::Float as i32,
            ),
        ));
        n
    };
    let then_graph = make_graph(vec![then_const], vec![], vec!["then_out"], vec![]);

    let else_const = {
        let mut n = NodeProto::default();
        n.op_type = "Constant".to_string();
        n.output.push("else_out".to_string());
        n.attribute.push(make_attr_tensor(
            "value",
            make_tensor_proto(
                [10.0f32, 20.0].iter().flat_map(|v| v.to_le_bytes()).collect(),
                vec![2],
                tensor_proto::DataType::Float as i32,
            ),
        ));
        n
    };
    let else_graph = make_graph(vec![else_const], vec![], vec!["else_out"], vec![]);

    let mut if_node = NodeProto::default();
    if_node.op_type = "If".to_string();
    if_node.input.push("condition".to_string());
    if_node.output.push("output".to_string());
    if_node.attribute.push(make_attr_graph("then_branch", then_graph));
    if_node.attribute.push(make_attr_graph("else_branch", else_graph));
    graph.node.push(if_node);

    let mut output = ValueInfoProto::default();
    output.name = "output".to_string();
    graph.output.push(output);

    model.graph = Some(graph);

    let importer = OnnxImporter::new();
    let result = importer.import_model(model, &[]);
    assert!(result.is_err(), "expected error for incompatible If branches");
    let msg = result.err().unwrap().to_string();
    assert!(msg.contains("incompatible branches"), "expected incompatible branches error, got: {msg}");
}

// =========================================================================
// Trace API tests
// =========================================================================

/// `y = x * [2, 2, 2]` with `x` a runtime input: a non-identity graph, so the
/// output's trace captures the placeholder rather than aliasing it.
fn make_mul_by_two_model() -> ModelProto {
    let mut g = GraphProto::default();
    g.name = "runtime_input_mul".to_string();
    g.input.push(make_typed_input("x", tensor_proto::DataType::Float as i32, &[3]));
    let mut two = TensorProto::default();
    two.name = "two".to_string();
    two.data_type = tensor_proto::DataType::Float as i32;
    two.dims = vec![3];
    two.float_data = vec![2.0, 2.0, 2.0];
    g.initializer.push(two);
    g.output.push(ValueInfoProto { name: "y".to_string(), ..Default::default() });
    g.node.push(NodeProto {
        op_type: "Mul".to_string(),
        input: vec!["x".to_string(), "two".to_string()],
        output: vec!["y".to_string()],
        ..Default::default()
    });
    ModelProto { graph: Some(g), ..Default::default() }
}

fn make_typed_input(name: &str, dtype: i32, dims: &[i64]) -> ValueInfoProto {
    use crate::parser::onnx::{TensorShapeProto, TypeProto, tensor_shape_proto};
    let shape = TensorShapeProto {
        dim: dims
            .iter()
            .map(|&d| tensor_shape_proto::Dimension {
                denotation: String::new(),
                value: if d > 0 { Some(tensor_shape_proto::dimension::Value::DimValue(d)) } else { None },
            })
            .collect(),
    };
    ValueInfoProto {
        name: name.to_string(),
        r#type: Some(TypeProto {
            denotation: String::new(),
            value: Some(type_proto::Value::TensorType(TensorTypeProto { elem_type: dtype, shape: Some(shape) })),
        }),
        ..Default::default()
    }
}

#[test]
fn test_trace_static_shapes() {
    let mut model = ModelProto::default();
    let mut graph = GraphProto::default();
    graph.name = "trace_test".to_string();

    graph.input.push(make_typed_input("input", tensor_proto::DataType::Float as i32, &[2, 3]));

    let mut output = ValueInfoProto::default();
    output.name = "output".to_string();
    graph.output.push(output);

    let mut node = NodeProto::default();
    node.op_type = "Identity".to_string();
    node.input.push("input".to_string());
    node.output.push("output".to_string());
    graph.node.push(node);

    model.graph = Some(graph);

    let importer = OnnxImporter::new();
    let result = importer.import_model(model, &[]).unwrap();

    assert!(result.inputs.contains_key("input"));
    assert!(result.outputs.contains_key("output"));

    let input_shape = result.inputs["input"].dims().unwrap();
    assert_eq!(input_shape, vec![2, 3]);
}

#[test]
fn test_import_with_dynamic_dims() {
    use crate::parser::onnx::{TensorShapeProto, TypeProto, tensor_shape_proto};

    let mut model = ModelProto::default();
    let mut graph = GraphProto::default();
    graph.name = "trace_dim_test".to_string();

    let shape = TensorShapeProto {
        dim: vec![
            tensor_shape_proto::Dimension {
                denotation: String::new(),
                value: Some(tensor_shape_proto::dimension::Value::DimParam("batch".to_string())),
            },
            tensor_shape_proto::Dimension {
                denotation: String::new(),
                value: Some(tensor_shape_proto::dimension::Value::DimValue(3)),
            },
        ],
    };
    graph.input.push(ValueInfoProto {
        name: "input".to_string(),
        r#type: Some(TypeProto {
            denotation: String::new(),
            value: Some(type_proto::Value::TensorType(TensorTypeProto {
                elem_type: tensor_proto::DataType::Float as i32,
                shape: Some(shape),
            })),
        }),
        ..Default::default()
    });

    let mut output = ValueInfoProto::default();
    output.name = "output".to_string();
    graph.output.push(output);

    let mut node = NodeProto::default();
    node.op_type = "Identity".to_string();
    node.input.push("input".to_string());
    node.output.push("output".to_string());
    graph.node.push(node);

    model.graph = Some(graph);

    let importer = OnnxImporter::new();
    let result = importer.import_model(model, &[("batch", 4)]).unwrap();
    assert!(result.outputs.contains_key("output"));
    assert!(result.variables.contains_key("batch"));
}

// =========================================================================
// Variable auto-extraction tests
// =========================================================================

#[test]
fn test_prepare_extracts_variables() {
    use crate::parser::onnx::{TensorShapeProto, TypeProto, tensor_shape_proto};

    let mut model = ModelProto::default();
    let mut graph = GraphProto::default();
    graph.name = "var_test".to_string();

    // Input with dim_param "batch" and static dim 3
    let shape = TensorShapeProto {
        dim: vec![
            tensor_shape_proto::Dimension {
                denotation: String::new(),
                value: Some(tensor_shape_proto::dimension::Value::DimParam("batch".to_string())),
            },
            tensor_shape_proto::Dimension {
                denotation: String::new(),
                value: Some(tensor_shape_proto::dimension::Value::DimValue(3)),
            },
        ],
    };
    graph.input.push(ValueInfoProto {
        name: "x".to_string(),
        r#type: Some(TypeProto {
            denotation: String::new(),
            value: Some(type_proto::Value::TensorType(TensorTypeProto {
                elem_type: tensor_proto::DataType::Float as i32,
                shape: Some(shape),
            })),
        }),
        ..Default::default()
    });

    let mut output = ValueInfoProto::default();
    output.name = "y".to_string();
    graph.output.push(output);

    model.graph = Some(graph);

    let importer = OnnxImporter::new();
    let result = importer.import_model(model, &[]).unwrap();

    // Variables should be auto-extracted
    assert!(!result.variables.is_empty());
    assert!(result.variables.contains_key("batch"));

    let var = result.variables.get("batch").unwrap();
    assert_eq!(var.name(), "batch");
    assert_eq!(var.bounds(), (1, i16::MAX as i64));
}

#[test]
fn test_prepare_no_variables_for_static_model() {
    let mut model = ModelProto::default();
    let mut graph = GraphProto::default();
    graph.name = "static_test".to_string();

    graph.input.push(make_typed_input("x", tensor_proto::DataType::Float as i32, &[2, 3]));

    let mut output = ValueInfoProto::default();
    output.name = "y".to_string();
    graph.output.push(output);

    model.graph = Some(graph);

    let importer = OnnxImporter::new();
    let result = importer.import_model(model, &[]).unwrap();

    assert!(result.variables.is_empty());
}

#[test]
fn test_prepare_multiple_dynamic_dims() {
    use crate::parser::onnx::{TensorShapeProto, TypeProto, tensor_shape_proto};

    let mut model = ModelProto::default();
    let mut graph = GraphProto::default();
    graph.name = "multi_var_test".to_string();

    // Input with two different dim_params
    let shape = TensorShapeProto {
        dim: vec![
            tensor_shape_proto::Dimension {
                denotation: String::new(),
                value: Some(tensor_shape_proto::dimension::Value::DimParam("batch".to_string())),
            },
            tensor_shape_proto::Dimension {
                denotation: String::new(),
                value: Some(tensor_shape_proto::dimension::Value::DimParam("seq_len".to_string())),
            },
            tensor_shape_proto::Dimension {
                denotation: String::new(),
                value: Some(tensor_shape_proto::dimension::Value::DimValue(768)),
            },
        ],
    };
    graph.input.push(ValueInfoProto {
        name: "x".to_string(),
        r#type: Some(TypeProto {
            denotation: String::new(),
            value: Some(type_proto::Value::TensorType(TensorTypeProto {
                elem_type: tensor_proto::DataType::Float as i32,
                shape: Some(shape),
            })),
        }),
        ..Default::default()
    });

    let mut output = ValueInfoProto::default();
    output.name = "y".to_string();
    graph.output.push(output);

    model.graph = Some(graph);

    let importer = OnnxImporter::new();
    let result = importer.import_model(model, &[]).unwrap();

    let mut dims: Vec<&str> = result.variables.keys().map(|s| s.as_str()).collect();
    dims.sort();
    assert_eq!(dims, vec!["batch", "seq_len"]);
}

#[test]
fn test_default_max_dim_configurable() {
    use crate::parser::onnx::{TensorShapeProto, TypeProto, tensor_shape_proto};

    let mut model = ModelProto::default();
    let mut graph = GraphProto::default();
    graph.name = "config_test".to_string();

    let shape = TensorShapeProto {
        dim: vec![tensor_shape_proto::Dimension {
            denotation: String::new(),
            value: Some(tensor_shape_proto::dimension::Value::DimParam("batch".to_string())),
        }],
    };
    graph.input.push(ValueInfoProto {
        name: "x".to_string(),
        r#type: Some(TypeProto {
            denotation: String::new(),
            value: Some(type_proto::Value::TensorType(TensorTypeProto {
                elem_type: tensor_proto::DataType::Float as i32,
                shape: Some(shape),
            })),
        }),
        ..Default::default()
    });

    let mut output = ValueInfoProto::default();
    output.name = "y".to_string();
    graph.output.push(output);

    model.graph = Some(graph);

    let mut importer = OnnxImporter::new();
    importer.default_max_dim = 256;
    let result = importer.import_model(model, &[]).unwrap();

    let var = result.variables.get("batch").unwrap();
    assert_eq!(var.bounds(), (1, 256));
}

#[test]
fn test_shared_dim_param_across_inputs() {
    use crate::parser::onnx::{TensorShapeProto, TypeProto, tensor_shape_proto};

    let mut model = ModelProto::default();
    let mut graph = GraphProto::default();
    graph.name = "shared_dim_test".to_string();

    // Two inputs sharing the same "batch" dim_param
    for name in ["x", "y"] {
        let shape = TensorShapeProto {
            dim: vec![
                tensor_shape_proto::Dimension {
                    denotation: String::new(),
                    value: Some(tensor_shape_proto::dimension::Value::DimParam("batch".to_string())),
                },
                tensor_shape_proto::Dimension {
                    denotation: String::new(),
                    value: Some(tensor_shape_proto::dimension::Value::DimValue(4)),
                },
            ],
        };
        graph.input.push(ValueInfoProto {
            name: name.to_string(),
            r#type: Some(TypeProto {
                denotation: String::new(),
                value: Some(type_proto::Value::TensorType(TensorTypeProto {
                    elem_type: tensor_proto::DataType::Float as i32,
                    shape: Some(shape),
                })),
            }),
            ..Default::default()
        });
    }

    let mut output = ValueInfoProto::default();
    output.name = "out".to_string();
    graph.output.push(output);

    model.graph = Some(graph);

    let importer = OnnxImporter::new();
    let result = importer.import_model(model, &[]).unwrap();

    // Should only have ONE "batch" variable, not duplicated
    assert_eq!(result.variables.len(), 1);
    assert!(result.variables.contains_key("batch"));
}

// =========================================================================
// OnnxModel (single import() call) tests
// =========================================================================

#[test]
fn test_import_model_static() {
    let mut model = ModelProto::default();
    let mut graph = GraphProto::default();
    graph.name = "import_test".to_string();

    graph.input.push(make_typed_input("x", tensor_proto::DataType::Float as i32, &[2, 3]));

    let mut output = ValueInfoProto::default();
    output.name = "y".to_string();
    graph.output.push(output);

    let mut node = NodeProto::default();
    node.op_type = "Identity".to_string();
    node.input.push("x".to_string());
    node.output.push("y".to_string());
    graph.node.push(node);

    model.graph = Some(graph);

    let importer = OnnxImporter::new();
    let onnx_model = importer.import_model(model, &[]).unwrap();

    // Inputs and outputs are normal Tensors
    assert!(onnx_model.inputs.contains_key("x"));
    assert!(onnx_model.outputs.contains_key("y"));

    // Static model has no variables
    assert!(onnx_model.variables.is_empty());

    // Input has correct shape
    let shape = onnx_model.inputs["x"].dims().unwrap();
    assert_eq!(shape, vec![2, 3]);
}

#[test]
fn test_import_model_with_dynamic_dims() {
    use crate::parser::onnx::{TensorShapeProto, TypeProto, tensor_shape_proto};

    let mut model = ModelProto::default();
    let mut graph = GraphProto::default();
    graph.name = "dynamic_import_test".to_string();

    let shape = TensorShapeProto {
        dim: vec![
            tensor_shape_proto::Dimension {
                denotation: String::new(),
                value: Some(tensor_shape_proto::dimension::Value::DimParam("batch".to_string())),
            },
            tensor_shape_proto::Dimension {
                denotation: String::new(),
                value: Some(tensor_shape_proto::dimension::Value::DimValue(3)),
            },
        ],
    };
    graph.input.push(ValueInfoProto {
        name: "x".to_string(),
        r#type: Some(TypeProto {
            denotation: String::new(),
            value: Some(type_proto::Value::TensorType(TensorTypeProto {
                elem_type: tensor_proto::DataType::Float as i32,
                shape: Some(shape),
            })),
        }),
        ..Default::default()
    });

    let mut output = ValueInfoProto::default();
    output.name = "y".to_string();
    graph.output.push(output);

    let mut node = NodeProto::default();
    node.op_type = "Identity".to_string();
    node.input.push("x".to_string());
    node.output.push("y".to_string());
    graph.node.push(node);

    model.graph = Some(graph);

    let importer = OnnxImporter::new();
    let onnx_model = importer.import_model(model, &[("batch", 4)]).unwrap();

    // Variables extracted
    assert!(onnx_model.variables.contains_key("batch"));
    assert_eq!(onnx_model.variables["batch"].bounds(), (1, i16::MAX as i64));

    // Inputs and outputs present
    assert!(onnx_model.inputs.contains_key("x"));
    assert!(onnx_model.outputs.contains_key("y"));
}

// =========================================================================
// Shape-only operator tests
// =========================================================================

#[test]
fn test_skip_layer_norm_no_optionals() {
    let registry = OpRegistry::new();
    let x = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0]]);
    let skip = Tensor::from_ndarray(&array![[4.0f32, 5.0, 6.0]]);
    let gamma = Tensor::from_slice([1.0f32, 1.0, 1.0]);
    let mut node = NodeProto::default();
    node.attribute.push(make_attr_float("epsilon", 1e-5));
    let inputs = vec![Some(x), Some(skip), Some(gamma)];
    let result = registry.dispatch_multi("SkipLayerNormalization", "com.microsoft", &inputs, &node, i64::MAX).unwrap();
    assert_eq!(result.len(), 4);
    let dims = result[0].dims().unwrap();
    assert_eq!(dims, [1, 3]);
}

#[test]
fn test_embed_layer_norm() {
    let registry = OpRegistry::new();
    let input_ids = Tensor::from_ndarray(&array![[0i32, 1, 2]]);
    let word_emb_data: Vec<f32> = (0..12).map(|v| v as f32).collect();
    let word_emb = Tensor::from_ndarray(&Array2::from_shape_vec((3, 4), word_emb_data).unwrap());
    let pos_emb = Tensor::from_ndarray(&Array2::from_elem((3, 4), 0.1f32));
    let gamma = Tensor::from_slice([1.0f32; 4]);
    let beta = Tensor::from_slice([0.0f32; 4]);
    let mut node = NodeProto::default();
    node.attribute.push(make_attr_float("epsilon", 1e-5));
    let inputs = vec![Some(input_ids), None, Some(word_emb), Some(pos_emb), None, Some(gamma), Some(beta)];
    let result = registry.dispatch_multi("EmbedLayerNormalization", "com.microsoft", &inputs, &node, i64::MAX).unwrap();
    assert_eq!(result.len(), 3);
    let dims = result[0].dims().unwrap();
    assert_eq!(dims, [1, 3, 4]);
    let sum_dims = result[2].dims().unwrap();
    assert_eq!(sum_dims, [1, 3, 4]);
}

#[test]
fn test_attention_contrib_basic() {
    let registry = OpRegistry::new();
    let x = Tensor::from_ndarray(&Array3::from_elem((1, 2, 4), 1.0f32));
    let mut w_data = vec![0.0f32; 48];
    for i in 0..4 {
        w_data[i * 12 + i] = 1.0;
        w_data[i * 12 + 4 + i] = 1.0;
        w_data[i * 12 + 8 + i] = 1.0;
    }
    let weights = Tensor::from_ndarray(&Array2::from_shape_vec((4, 12), w_data).unwrap());
    let bias = Tensor::from_slice([0.0f32; 12]);

    let mut node = NodeProto::default();
    node.attribute.push(make_attr_int("num_heads", 2));

    let inputs = vec![Some(x), Some(weights), Some(bias)];
    let result = registry.dispatch_multi("Attention", "com.microsoft", &inputs, &node, i64::MAX).unwrap();
    assert!(!result.is_empty());
    let dims = result[0].dims().unwrap();
    assert_eq!(dims, [1, 2, 4]);
}

#[test]
fn test_attention_contrib_causal() {
    let registry = OpRegistry::new();
    let x = Tensor::from_ndarray(&array![[[1.0f32, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]]]);
    let mut w_data = vec![0.1f32; 24];
    for i in 0..2 {
        w_data[i * 6 + i] = 1.0;
        w_data[i * 6 + 2 + i] = 1.0;
        w_data[i * 6 + 4 + i] = 1.0;
    }
    let weights = Tensor::from_ndarray(&Array2::from_shape_vec((4, 6), w_data).unwrap());
    let bias = Tensor::from_slice([0.0f32; 6]);

    let mut node = NodeProto::default();
    node.attribute.push(make_attr_int("num_heads", 1));
    node.attribute.push(make_attr_int("unidirectional", 1));
    node.attribute.push(make_attr_ints("qkv_hidden_sizes", &[2, 2, 2]));

    let inputs = vec![Some(x), Some(weights), Some(bias)];
    let result = registry.dispatch_multi("Attention", "com.microsoft", &inputs, &node, i64::MAX).unwrap();
    let dims = result[0].dims().unwrap();
    assert_eq!(dims, [1, 2, 2]);
}

#[test]
fn test_attention_onnx_basic() {
    let registry = OpRegistry::new();
    let q = Tensor::from_ndarray(&Array4::from_elem((1, 2, 2, 2), 1.0f32));
    let k = q.clone();
    let v = Tensor::from_ndarray(&array![[[[1.0f32, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]]]);

    let node = NodeProto::default();
    let inputs = vec![Some(q), Some(k), Some(v)];
    let result = registry.dispatch_multi("Attention", "", &inputs, &node, i64::MAX).unwrap();
    assert_eq!(result.len(), 4);
    let dims = result[0].dims().unwrap();
    assert_eq!(dims, [1, 2, 2, 2]);
}

#[test]
fn test_attention_domain_dispatch() {
    let registry = OpRegistry::new();

    let q = Tensor::from_ndarray(&Array4::from_elem((1, 1, 2, 2), 1.0f32));
    let k = q.clone();
    let v = q.clone();
    let node = NodeProto::default();
    let inputs_onnx = vec![Some(q), Some(k), Some(v)];
    let result_onnx = registry.dispatch_multi("Attention", "", &inputs_onnx, &node, i64::MAX).unwrap();
    assert_eq!(result_onnx.len(), 4);

    let x = Tensor::from_ndarray(&Array3::from_elem((1, 2, 2), 1.0f32));
    let w = Tensor::from_ndarray(&Array2::from_elem((2, 6), 0.1f32));
    let b = Tensor::from_slice([0.0f32; 6]);
    let mut node_contrib = NodeProto::default();
    node_contrib.attribute.push(make_attr_int("num_heads", 1));
    node_contrib.attribute.push(make_attr_ints("qkv_hidden_sizes", &[2, 2, 2]));
    let inputs_contrib = vec![Some(x), Some(w), Some(b)];
    let result_contrib =
        registry.dispatch_multi("Attention", "com.microsoft", &inputs_contrib, &node_contrib, i64::MAX).unwrap();
    assert_eq!(result_contrib.len(), 2);
}

#[test]
fn test_depth_to_space_dcr() {
    let registry = OpRegistry::new();
    let x = Tensor::from_ndarray(&Array4::from_shape_vec((1, 8, 1, 1), (0..8).map(|v| v as f32).collect()).unwrap());
    let mut node = NodeProto::default();
    node.attribute.push(make_attr_int("blocksize", 2));
    let result = registry.dispatch("DepthToSpace", "", &[x], &node).unwrap();
    let shape = result.dims().unwrap();
    assert_eq!(shape, vec![1, 2, 2, 2]);
}

#[test]
fn test_depth_to_space_crd() {
    let registry = OpRegistry::new();
    let x = Tensor::from_ndarray(&Array4::from_shape_vec((1, 8, 1, 1), (0..8).map(|v| v as f32).collect()).unwrap());
    let mut node = NodeProto::default();
    node.attribute.push(make_attr_int("blocksize", 2));
    node.attribute.push(make_attr_string("mode", "CRD"));
    let result = registry.dispatch("DepthToSpace", "", &[x], &node).unwrap();
    let shape = result.dims().unwrap();
    assert_eq!(shape, vec![1, 2, 2, 2]);
}

#[test]
fn test_space_to_depth() {
    let registry = OpRegistry::new();
    let x = Tensor::from_ndarray(&Array4::from_shape_vec((1, 1, 4, 4), (0..16).map(|v| v as f32).collect()).unwrap());
    let mut node = NodeProto::default();
    node.attribute.push(make_attr_int("blocksize", 2));
    let result = registry.dispatch("SpaceToDepth", "", &[x], &node).unwrap();
    let shape = result.dims().unwrap();
    assert_eq!(shape, vec![1, 4, 2, 2]);
}

#[test]
fn test_affine_grid() {
    let registry = OpRegistry::new();
    let theta = Tensor::from_ndarray(&array![[[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0]]]);
    let size = Tensor::from_slice([1i64, 1, 2, 2]);
    let mut node = NodeProto::default();
    node.attribute.push(make_attr_int("align_corners", 0));
    let inputs = vec![Some(theta), Some(size)];
    let result = registry.dispatch_multi("AffineGrid", "", &inputs, &node, i64::MAX).unwrap();
    let shape = result[0].dims().unwrap();
    assert_eq!(shape, vec![1, 2, 2, 2]);
}

#[test]
fn test_batch_norm_training() {
    let registry = OpRegistry::new();
    let x = Tensor::from_ndarray(&array![[[[1.0f32]], [[2.0]]], [[[3.0]], [[4.0]]]]);
    let scale = Tensor::from_slice([1.0f32, 1.0]);
    let bias = Tensor::from_slice([0.0f32, 0.0]);
    let mean = Tensor::from_slice([0.0f32, 0.0]);
    let var = Tensor::from_slice([1.0f32, 1.0]);
    let mut node = NodeProto::default();
    node.attribute.push(make_attr_int("training_mode", 1));
    let inputs = vec![Some(x), Some(scale), Some(bias), Some(mean), Some(var)];
    let result = registry.dispatch_multi("BatchNormalization", "", &inputs, &node, i64::MAX).unwrap();
    assert_eq!(result.len(), 3);
    let dims = result[0].dims().unwrap();
    assert_eq!(dims, [2, 2, 1, 1]);
}

// =========================================================================
// Helper functions
// =========================================================================

/// Build a minimal model with an If node.
///
/// Graph: condition (bool scalar initializer) -> If -> output
/// The then_branch adds `x + 10.0`, the else_branch adds `x + 20.0`.
pub(super) fn make_if_model(cond_value: bool, x_values: &[f32]) -> ModelProto {
    let mut model = ModelProto::default();
    let mut graph = GraphProto::default();
    graph.name = "if_test".to_string();

    let mut cond_input = ValueInfoProto::default();
    cond_input.name = "condition".to_string();
    graph.input.push(cond_input);

    let mut cond_init = TensorProto::default();
    cond_init.name = "condition".to_string();
    cond_init.data_type = tensor_proto::DataType::Bool as i32;
    cond_init.dims = vec![];
    cond_init.raw_data = prost::bytes::Bytes::from(vec![cond_value as u8]);
    graph.initializer.push(cond_init);

    let mut x_input = ValueInfoProto::default();
    x_input.name = "x".to_string();
    graph.input.push(x_input);

    let mut x_init = TensorProto::default();
    x_init.name = "x".to_string();
    x_init.data_type = tensor_proto::DataType::Float as i32;
    x_init.dims = vec![x_values.len() as i64];
    x_init.raw_data = x_values.iter().flat_map(|v| v.to_le_bytes()).collect();
    graph.initializer.push(x_init);

    let then_add_const = {
        let mut n = NodeProto::default();
        n.op_type = "Constant".to_string();
        n.output.push("ten".to_string());
        n.attribute.push(make_attr_tensor(
            "value",
            make_tensor_proto(10.0f32.to_le_bytes().to_vec(), vec![1], tensor_proto::DataType::Float as i32),
        ));
        n
    };
    let then_add = {
        let mut n = NodeProto::default();
        n.op_type = "Add".to_string();
        n.input.push("x".to_string());
        n.input.push("ten".to_string());
        n.output.push("then_out".to_string());
        n
    };
    let then_graph = make_graph(vec![then_add_const, then_add], vec!["x"], vec!["then_out"], vec![]);

    let else_add_const = {
        let mut n = NodeProto::default();
        n.op_type = "Constant".to_string();
        n.output.push("twenty".to_string());
        n.attribute.push(make_attr_tensor(
            "value",
            make_tensor_proto(20.0f32.to_le_bytes().to_vec(), vec![1], tensor_proto::DataType::Float as i32),
        ));
        n
    };
    let else_add = {
        let mut n = NodeProto::default();
        n.op_type = "Add".to_string();
        n.input.push("x".to_string());
        n.input.push("twenty".to_string());
        n.output.push("else_out".to_string());
        n
    };
    let else_graph = make_graph(vec![else_add_const, else_add], vec!["x"], vec!["else_out"], vec![]);

    let mut if_node = NodeProto::default();
    if_node.op_type = "If".to_string();
    if_node.input.push("condition".to_string());
    if_node.output.push("output".to_string());
    if_node.attribute.push(make_attr_graph("then_branch", then_graph));
    if_node.attribute.push(make_attr_graph("else_branch", else_graph));
    graph.node.push(if_node);

    let mut output = ValueInfoProto::default();
    output.name = "output".to_string();
    graph.output.push(output);

    model.graph = Some(graph);
    model
}

// =========================================================================
// Codegen-required tests (realize/to_vec)
// =========================================================================

svod_tensor::codegen_tests! {
    fn test_if_true_condition(config) {
        let importer = OnnxImporter::new();
        let model = make_if_model(true, &[1.0, 2.0, 3.0]);
        let outputs = importer.import_model(model, &[]).unwrap().outputs;

        let result = outputs.get("output").unwrap();
        let r = result.clone();
        r.realize_with(&config).unwrap();
        let vals = r.as_vec::<f32>().unwrap();
        assert_eq!(vals, vec![11.0, 12.0, 13.0]); // x + 10
    }

    fn test_if_false_condition(config) {
        let importer = OnnxImporter::new();
        let model = make_if_model(false, &[1.0, 2.0, 3.0]);
        let outputs = importer.import_model(model, &[]).unwrap().outputs;

        let result = outputs.get("output").unwrap();
        let r = result.clone();
        r.realize_with(&config).unwrap();
        let vals = r.as_vec::<f32>().unwrap();
        assert_eq!(vals, vec![21.0, 22.0, 23.0]); // x + 20
    }

    fn test_if_where_path(config) {
        let importer = OnnxImporter::new();
        let model = make_if_model(true, &[5.0, 6.0]);
        let outputs = importer.import_model(model, &[]).unwrap().outputs;

        let result = outputs.get("output").unwrap();
        let r = result.clone();
        r.realize_with(&config).unwrap();
        let vals = r.as_vec::<f32>().unwrap();
        assert_eq!(vals, vec![15.0, 16.0]);
    }

    fn test_if_with_parent_scope(config) {
        let mut model = ModelProto::default();
        let mut graph = GraphProto::default();
        graph.name = "if_parent_scope_test".to_string();

        let mut cond_input = ValueInfoProto::default();
        cond_input.name = "condition".to_string();
        graph.input.push(cond_input);
        let mut cond_init = TensorProto::default();
        cond_init.name = "condition".to_string();
        cond_init.data_type = tensor_proto::DataType::Bool as i32;
        cond_init.dims = vec![];
        cond_init.raw_data = prost::bytes::Bytes::from(vec![1u8]);
        graph.initializer.push(cond_init);

        let mut pv_input = ValueInfoProto::default();
        pv_input.name = "parent_val".to_string();
        graph.input.push(pv_input);
        let mut pv_init = TensorProto::default();
        pv_init.name = "parent_val".to_string();
        pv_init.data_type = tensor_proto::DataType::Float as i32;
        pv_init.dims = vec![3];
        pv_init.raw_data = [100.0f32, 200.0, 300.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        graph.initializer.push(pv_init);

        let then_id = {
            let mut n = NodeProto::default();
            n.op_type = "Identity".to_string();
            n.input.push("parent_val".to_string());
            n.output.push("then_out".to_string());
            n
        };
        let then_graph = make_graph(vec![then_id], vec![], vec!["then_out"], vec![]);

        let else_const = {
            let mut n = NodeProto::default();
            n.op_type = "Constant".to_string();
            n.output.push("else_out".to_string());
            n.attribute.push(make_attr_tensor(
                "value",
                make_tensor_proto(
                    [0.0f32, 0.0, 0.0].iter().flat_map(|v| v.to_le_bytes()).collect(),
                    vec![3],
                    tensor_proto::DataType::Float as i32,
                ),
            ));
            n
        };
        let else_graph = make_graph(vec![else_const], vec![], vec!["else_out"], vec![]);

        let mut if_node = NodeProto::default();
        if_node.op_type = "If".to_string();
        if_node.input.push("condition".to_string());
        if_node.output.push("output".to_string());
        if_node.attribute.push(make_attr_graph("then_branch", then_graph));
        if_node.attribute.push(make_attr_graph("else_branch", else_graph));
        graph.node.push(if_node);

        let mut output = ValueInfoProto::default();
        output.name = "output".to_string();
        graph.output.push(output);

        model.graph = Some(graph);

        let importer = OnnxImporter::new();
        let outputs = importer.import_model(model, &[]).unwrap().outputs;

        let result = outputs.get("output").unwrap();
        let r = result.clone();
        r.realize_with(&config).unwrap();
        let vals = r.as_vec::<f32>().unwrap();
        assert_eq!(vals, vec![100.0, 200.0, 300.0]);
    }

    fn test_if_nested(config) {
        let mut model = ModelProto::default();
        let mut graph = GraphProto::default();
        graph.name = "nested_if_test".to_string();

        for (name, val) in [("outer_cond", true), ("inner_cond", false)] {
            let mut inp = ValueInfoProto::default();
            inp.name = name.to_string();
            graph.input.push(inp);
            let mut init = TensorProto::default();
            init.name = name.to_string();
            init.data_type = tensor_proto::DataType::Bool as i32;
            init.dims = vec![];
            init.raw_data = prost::bytes::Bytes::from(vec![val as u8]);
            graph.initializer.push(init);
        }

        let mut x_input = ValueInfoProto::default();
        x_input.name = "x".to_string();
        graph.input.push(x_input);
        let mut x_init = TensorProto::default();
        x_init.name = "x".to_string();
        x_init.data_type = tensor_proto::DataType::Float as i32;
        x_init.dims = vec![2];
        x_init.raw_data = [10.0f32, 20.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        graph.initializer.push(x_init);

        let inner_then = {
            let c = {
                let mut n = NodeProto::default();
                n.op_type = "Constant".to_string();
                n.output.push("one".to_string());
                n.attribute.push(make_attr_tensor(
                    "value",
                    make_tensor_proto(1.0f32.to_le_bytes().to_vec(), vec![1], tensor_proto::DataType::Float as i32),
                ));
                n
            };
            let add = {
                let mut n = NodeProto::default();
                n.op_type = "Add".to_string();
                n.input.push("x".to_string());
                n.input.push("one".to_string());
                n.output.push("inner_then_out".to_string());
                n
            };
            make_graph(vec![c, add], vec!["x"], vec!["inner_then_out"], vec![])
        };

        let inner_else = {
            let c = {
                let mut n = NodeProto::default();
                n.op_type = "Constant".to_string();
                n.output.push("two".to_string());
                n.attribute.push(make_attr_tensor(
                    "value",
                    make_tensor_proto(2.0f32.to_le_bytes().to_vec(), vec![1], tensor_proto::DataType::Float as i32),
                ));
                n
            };
            let add = {
                let mut n = NodeProto::default();
                n.op_type = "Add".to_string();
                n.input.push("x".to_string());
                n.input.push("two".to_string());
                n.output.push("inner_else_out".to_string());
                n
            };
            make_graph(vec![c, add], vec!["x"], vec!["inner_else_out"], vec![])
        };

        let inner_if = {
            let mut n = NodeProto::default();
            n.op_type = "If".to_string();
            n.input.push("inner_cond".to_string());
            n.output.push("outer_then_out".to_string());
            n.attribute.push(make_attr_graph("then_branch", inner_then));
            n.attribute.push(make_attr_graph("else_branch", inner_else));
            n
        };
        let outer_then = make_graph(vec![inner_if], vec![], vec!["outer_then_out"], vec![]);

        let outer_else = {
            let c = {
                let mut n = NodeProto::default();
                n.op_type = "Constant".to_string();
                n.output.push("outer_else_out".to_string());
                n.attribute.push(make_attr_tensor(
                    "value",
                    make_tensor_proto(
                        [0.0f32, 0.0].iter().flat_map(|v| v.to_le_bytes()).collect(),
                        vec![2],
                        tensor_proto::DataType::Float as i32,
                    ),
                ));
                n
            };
            make_graph(vec![c], vec![], vec!["outer_else_out"], vec![])
        };

        let mut outer_if = NodeProto::default();
        outer_if.op_type = "If".to_string();
        outer_if.input.push("outer_cond".to_string());
        outer_if.output.push("output".to_string());
        outer_if.attribute.push(make_attr_graph("then_branch", outer_then));
        outer_if.attribute.push(make_attr_graph("else_branch", outer_else));
        graph.node.push(outer_if);

        let mut output = ValueInfoProto::default();
        output.name = "output".to_string();
        graph.output.push(output);

        model.graph = Some(graph);

        let importer = OnnxImporter::new();
        let outputs = importer.import_model(model, &[]).unwrap().outputs;

        let result = outputs.get("output").unwrap();
        let r = result.clone();
        r.realize_with(&config).unwrap();
        let vals = r.as_vec::<f32>().unwrap();
        // outer_cond=true -> outer then_branch executes
        // inner_cond=false -> inner else_branch executes (x + 2)
        // x = [10, 20], so result = [12, 22]
        assert_eq!(vals, vec![12.0, 22.0]);
    }

    fn test_rms_norm(config) {
        let registry = OpRegistry::new();
        let x = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0, 4.0]]);
        let scale = Tensor::from_slice([1.0f32, 1.0, 1.0, 1.0]);
        let mut node = NodeProto::default();
        node.attribute.push(make_attr_int("axis", -1));
        node.attribute.push(make_attr_float("epsilon", 1e-5));
        let inputs = vec![Some(x), Some(scale)];
        let result = registry.dispatch_multi("RMSNormalization", "", &inputs, &node, i64::MAX).unwrap();
        let r = result[0].clone().contiguous();
        r.realize_with(&config).unwrap();
        let dims = r.dims().unwrap();
        assert_eq!(dims, [1, 4]);
        let view = r.array_view::<f32>().unwrap();
        let rms_inv = 1.0 / (7.5f32 + 1e-5).sqrt();
        for i in 0..4 {
            let expected = (i + 1) as f32 * rms_inv;
            assert!((view[[0, i]] - expected).abs() < 1e-4);
        }
    }

    fn test_rms_norm_with_scale(config) {
        let registry = OpRegistry::new();
        let x = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0, 4.0]]);
        let scale = Tensor::from_slice([2.0f32, 0.5, 1.0, 3.0]);
        let mut node = NodeProto::default();
        node.attribute.push(make_attr_float("epsilon", 1e-5));
        let inputs = vec![Some(x), Some(scale)];
        let result = registry.dispatch_multi("RMSNormalization", "", &inputs, &node, i64::MAX).unwrap();
        let r = result[0].clone().contiguous();
        r.realize_with(&config).unwrap();
        let view = r.array_view::<f32>().unwrap();
        let rms_inv = 1.0 / (7.5f32 + 1e-5).sqrt();
        let scales = [2.0, 0.5, 1.0, 3.0];
        for i in 0..4 {
            let expected = (i + 1) as f32 * rms_inv * scales[i];
            assert!((view[[0, i]] - expected).abs() < 1e-4);
        }
    }

    fn test_skip_layer_norm(config) {
        let registry = OpRegistry::new();
        let x = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0]]);
        let skip = Tensor::from_ndarray(&array![[0.1f32, 0.2, 0.3]]);
        let gamma = Tensor::from_slice([1.0f32, 1.0, 1.0]);
        let beta = Tensor::from_slice([0.0f32, 0.0, 0.0]);
        let mut node = NodeProto::default();
        node.attribute.push(make_attr_float("epsilon", 1e-5));
        let inputs = vec![Some(x), Some(skip), Some(gamma), Some(beta)];
        let result = registry.dispatch_multi("SkipLayerNormalization", "com.microsoft", &inputs, &node, i64::MAX).unwrap();
        assert_eq!(result.len(), 4);
        let dims = result[0].dims().unwrap();
        assert_eq!(dims, [1, 3]);
        let r0 = result[0].clone();
        r0.realize_with(&config).unwrap();
        let vals = r0.as_vec::<f32>().unwrap();
        let mean: f32 = vals.iter().sum::<f32>() / 3.0;
        assert!(mean.abs() < 1e-4, "layernorm mean should be ~0, got {mean}");
        let r3 = result[3].clone().contiguous();
        r3.realize_with(&config).unwrap();
        let x_sum = r3.array_view::<f32>().unwrap();
        assert!((x_sum[[0, 0]] - 1.1).abs() < 1e-4);
    }

    fn test_rotary_embedding_split(config) {
        let registry = OpRegistry::new();
        let x_data: Vec<f32> = (1..=8).map(|v| v as f32).collect();
        let x = Tensor::from_ndarray(&Array4::from_shape_vec((1, 1, 2, 4), x_data).unwrap());
        let cos_cache = Tensor::from_ndarray(&Array2::from_elem((2, 2), 1.0f32));
        let sin_cache = Tensor::from_ndarray(&Array2::from_elem((2, 2), 0.0f32));
        let pos_ids = Tensor::from_ndarray(&array![[0i32, 1]]);

        let mut node = NodeProto::default();
        node.attribute.push(make_attr_int("interleaved", 0));
        node.attribute.push(make_attr_int("num_heads", 1));

        let inputs = vec![Some(x), Some(pos_ids), Some(cos_cache), Some(sin_cache)];
        let result = registry.dispatch_multi("RotaryEmbedding", "com.microsoft", &inputs, &node, i64::MAX).unwrap();
        assert_eq!(result.len(), 1);
        let dims = result[0].dims().unwrap();
        assert_eq!(dims, [1, 1, 2, 4]);
        let r = result[0].clone();
        r.realize_with(&config).unwrap();
        let flat = r.as_vec::<f32>().unwrap();
        for (i, val) in flat.iter().enumerate() {
            assert!((val - (i + 1) as f32).abs() < 1e-4, "rotary identity: got {val}, expected {}", i + 1);
        }
    }

    fn test_attention_onnx_causal(config) {
        let registry = OpRegistry::new();
        let q = Tensor::from_ndarray(&array![[[[1.0f32, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]]]]);
        let k = q.clone();
        let v = Tensor::from_ndarray(&array![[[[1.0f32, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]]]);

        let mut node = NodeProto::default();
        node.attribute.push(make_attr_int("is_causal", 1));
        let inputs = vec![Some(q), Some(k), Some(v)];
        let result = registry.dispatch_multi("Attention", "", &inputs, &node, i64::MAX).unwrap();
        let r = result[0].clone().contiguous();
        r.realize_with(&config).unwrap();
        let dims = r.dims().unwrap();
        assert_eq!(dims, [1, 1, 3, 4]);
        let view = r.array_view::<f32>().unwrap();
        assert!((view[[0, 0, 0, 0]] - 1.0).abs() < 1e-4);
        assert!((view[[0, 0, 0, 1]] - 0.0).abs() < 1e-4);
    }

    fn test_lp_norm_l1(config) {
        let registry = OpRegistry::new();
        let x = Tensor::from_ndarray(&array![[1.0f32, -2.0, 3.0]]);
        let mut node = NodeProto::default();
        node.attribute.push(make_attr_int("axis", 1));
        node.attribute.push(make_attr_int("p", 1));
        let result = registry.dispatch("LpNormalization", "", &[x], &node).unwrap();
        let r = result.contiguous();
        r.realize_with(&config).unwrap();
        let view = r.array_view::<f32>().unwrap();
        assert!((view[[0, 0]] - 1.0 / 6.0).abs() < 1e-5);
        assert!((view[[0, 1]] - (-2.0 / 6.0)).abs() < 1e-5);
        assert!((view[[0, 2]] - 3.0 / 6.0).abs() < 1e-5);
    }

    fn test_lp_norm_l2(config) {
        let registry = OpRegistry::new();
        let x = Tensor::from_ndarray(&array![[3.0f32, 4.0]]);
        let mut node = NodeProto::default();
        node.attribute.push(make_attr_int("axis", 1));
        node.attribute.push(make_attr_int("p", 2));
        let result = registry.dispatch("LpNormalization", "", &[x], &node).unwrap();
        let r = result.contiguous();
        r.realize_with(&config).unwrap();
        let view = r.array_view::<f32>().unwrap();
        assert!((view[[0, 0]] - 0.6).abs() < 1e-5);
        assert!((view[[0, 1]] - 0.8).abs() < 1e-5);
    }

    fn test_mean_variance_norm(config) {
        let registry = OpRegistry::new();
        let x = Tensor::from_ndarray(&array![[[[3.0f32]], [[5.0]]]]);
        let node = NodeProto::default();
        let result = registry.dispatch("MeanVarianceNormalization", "", &[x], &node).unwrap();
        let shape = result.dims().unwrap();
        assert_eq!(shape, vec![1, 2, 1, 1]);
        let r = result.contiguous();
        r.realize_with(&config).unwrap();
        let view = r.array_view::<f32>().unwrap();
        assert!(view[[0, 0, 0, 0]].abs() < 1e-3);
        assert!(view[[0, 1, 0, 0]].abs() < 1e-3);
    }

    fn test_lrn(config) {
        let registry = OpRegistry::new();
        let x = Tensor::from_ndarray(&array![[[[1.0f32]], [[2.0]], [[3.0]]]]);
        let mut node = NodeProto::default();
        node.attribute.push(make_attr_int("size", 3));
        node.attribute.push(make_attr_float("alpha", 0.0001));
        node.attribute.push(make_attr_float("beta", 0.75));
        node.attribute.push(make_attr_float("bias", 1.0));
        let result = registry.dispatch("LRN", "", &[x], &node).unwrap();
        let shape = result.dims().unwrap();
        assert_eq!(shape, vec![1, 3, 1, 1]);
        result.realize_with(&config).unwrap();
        let vals = result.as_vec::<f32>().unwrap();
        for val in &vals {
            assert!(val.is_finite(), "LRN produced non-finite value: {val}");
        }
    }

    fn test_nll_loss(config) {
        let registry = OpRegistry::new();
        let log_probs = Tensor::from_ndarray(&array![
            [-0.5f32, -1.0, -2.0],
            [-0.3, -1.5, -0.8],
        ]);
        let target = Tensor::from_slice([0i64, 2]);
        let node = NodeProto::default();
        let inputs = vec![Some(log_probs), Some(target)];
        let result = registry.dispatch_multi("NegativeLogLikelihoodLoss", "", &inputs, &node, i64::MAX).unwrap();
        assert_eq!(result.len(), 1);
        let r = result[0].clone();
        r.realize_with(&config).unwrap();
        let val = r.as_vec::<f32>().unwrap()[0];
        assert!((val - 0.65).abs() < 1e-4, "NLL loss got {val}");
    }

    fn test_softmax_ce_loss(config) {
        let registry = OpRegistry::new();
        let logits = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [1.0, 2.0, 3.0]]);
        let target = Tensor::from_slice([0i64, 2]);
        let node = NodeProto::default();
        let inputs = vec![Some(logits), Some(target)];
        let result = registry.dispatch_multi("SoftmaxCrossEntropyLoss", "", &inputs, &node, i64::MAX).unwrap();
        assert_eq!(result.len(), 2);
        let r = result[0].clone();
        r.realize_with(&config).unwrap();
        let val = r.as_vec::<f32>().unwrap()[0];
        assert!(val > 0.0, "CE loss should be positive, got {val}");
        let lp_shape = result[1].dims().unwrap();
        assert_eq!(lp_shape, vec![2, 3]);
    }

    fn test_affine_grid_aligned(config) {
        let registry = OpRegistry::new();
        let theta = Tensor::from_ndarray(&array![[[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0]]]);
        let size = Tensor::from_slice([1i64, 1, 3, 3]);
        let mut node = NodeProto::default();
        node.attribute.push(make_attr_int("align_corners", 1));
        let inputs = vec![Some(theta), Some(size)];
        let result = registry.dispatch_multi("AffineGrid", "", &inputs, &node, i64::MAX).unwrap();
        let shape = result[0].dims().unwrap();
        assert_eq!(shape, vec![1, 3, 3, 2]);
        let r = result[0].clone().contiguous();
        r.realize_with(&config).unwrap();
        let view = r.array_view::<f32>().unwrap();
        assert!((view[[0, 0, 0, 0]] - (-1.0)).abs() < 1e-4, "x corner got {}", view[[0, 0, 0, 0]]);
        assert!((view[[0, 0, 0, 1]] - (-1.0)).abs() < 1e-4, "y corner got {}", view[[0, 0, 0, 1]]);
    }

    fn test_dropout_v7_inference(config) {
        let registry = OpRegistry::new();
        let x = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let node = NodeProto::default();
        let inputs = vec![Some(x)];
        let result = registry.dispatch_multi("Dropout", "", &inputs, &node, 13).unwrap();
        assert_eq!(result.len(), 2);
        let r = result[0].clone();
        r.realize_with(&config).unwrap();
        assert_eq!(r.as_vec::<f32>().unwrap(), [1.0, 2.0, 3.0]);
    }

    fn test_optional_has_element_present(config) {
        let registry = OpRegistry::new();
        let x = Tensor::from_slice([1.0f32]);
        let inputs = vec![Some(x)];
        let node = NodeProto::default();
        let result = registry.dispatch_multi("OptionalHasElement", "", &inputs, &node, i64::MAX).unwrap();
        let r = result[0].clone();
        r.realize_with(&config).unwrap();
        assert!(r.as_vec::<bool>().unwrap()[0]);
    }

    fn test_optional_has_element_absent(config) {
        let registry = OpRegistry::new();
        let inputs: Vec<Option<Tensor>> = vec![None];
        let node = NodeProto::default();
        let result = registry.dispatch_multi("OptionalHasElement", "", &inputs, &node, i64::MAX).unwrap();
        let r = result[0].clone();
        r.realize_with(&config).unwrap();
        assert!(!r.as_vec::<bool>().unwrap()[0]);
    }

    fn test_optional_get_element(config) {
        let registry = OpRegistry::new();
        let x = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let inputs = vec![Some(x)];
        let node = NodeProto::default();
        let result = registry.dispatch_multi("OptionalGetElement", "", &inputs, &node, i64::MAX).unwrap();
        let r = result[0].clone();
        r.realize_with(&config).unwrap();
        let vals = r.as_vec::<f32>().unwrap();
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
    }
}

// =========================================================================
// Doc example tests — mirrors code blocks from onnx.md and onnx/README.md
// =========================================================================

svod_tensor::codegen_tests! {
    /// onnx.md "Simple: All-Initializer Models" — destructure + realize_batch
    fn test_doc_all_initializer_realize_batch(config) {
        use crate::importer::OnnxModel;

        let importer = OnnxImporter::new();
        let OnnxModel { outputs, .. } = importer.import_model(make_minimal_model(), &[]).unwrap();

        let outs: Vec<&Tensor> = outputs.values().collect();
        Tensor::realize_batch_with(outs, &config).unwrap();

        // Minimal model: Identity([1,2,3]) = [1,2,3]
        let view = outputs["output"].array_view::<f32>().unwrap();
        assert_eq!(view.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
    }

    /// onnx.md "Placeholder inputs" — inputs.remove + assign + realize the
    /// input + realize_batch. `y = x * 2` captured the placeholder before the
    /// assign; the realized write lands in that identity, so it sees the data.
    fn test_doc_runtime_input_assign_realize_batch(config) {
        use crate::importer::OnnxModel;

        let OnnxModel { mut inputs, outputs, .. } =
            OnnxImporter::new().import_model(make_mul_by_two_model(), &[]).unwrap();
        let input = inputs.remove("x").unwrap();
        let placeholder = input.uop().base().id;

        input.assign(&Tensor::from_slice([10.0f32, 20.0, 30.0]));
        input.realize_with(&config).unwrap();
        assert_eq!(input.uop().base().id, placeholder, "an in-place assign keeps the placeholder identity");
        assert_eq!(input.as_vec::<f32>().unwrap(), [10.0, 20.0, 30.0]);

        let outs: Vec<&Tensor> = outputs.values().collect();
        Tensor::realize_batch_with(outs, &config).unwrap();
        let view = outputs["y"].array_view::<f32>().unwrap();
        assert_eq!(view.as_slice().unwrap(), &[20.0, 40.0, 60.0]);
    }

    /// onnx.md "Replay" — prepare_batch over the placeholder, then write the
    /// input through `array_view_mut` before every execute.
    fn test_doc_runtime_input_prepare_replay(config) {
        use crate::importer::OnnxModel;

        let OnnxModel { inputs, outputs, .. } =
            OnnxImporter::new().import_model(make_mul_by_two_model(), &[]).unwrap();
        let plan = Tensor::prepare_batch_with(outputs.values(), &config).unwrap();

        let input = &inputs["x"];
        for (frame, expected) in [([1.0f32, 2.0, 3.0], [2.0f32, 4.0, 6.0]), ([4.0, 5.0, 6.0], [8.0, 10.0, 12.0])] {
            input.array_view_mut::<f32>().unwrap().as_slice_mut().unwrap().copy_from_slice(&frame);
            plan.execute().unwrap();
            assert_eq!(outputs["y"].as_vec::<f32>().unwrap(), expected);
        }
    }

    /// Replay through assign: every realized assign stores into the storage the
    /// prepared plan reads, so each execute sees the latest value.
    fn test_doc_runtime_input_assign_replay(config) {
        use crate::importer::OnnxModel;

        let OnnxModel { inputs, outputs, .. } =
            OnnxImporter::new().import_model(make_mul_by_two_model(), &[]).unwrap();
        let plan = Tensor::prepare_batch_with(outputs.values(), &config).unwrap();

        let input = &inputs["x"];
        for (frame, expected) in [([1.0f32, 2.0, 3.0], [2.0f32, 4.0, 6.0]), ([4.0, 5.0, 6.0], [8.0, 10.0, 12.0])] {
            input.assign(&Tensor::from_slice(frame));
            input.realize_with(&config).unwrap();
            plan.execute().unwrap();
            assert_eq!(outputs["y"].as_vec::<f32>().unwrap(), expected);
        }
    }
}
