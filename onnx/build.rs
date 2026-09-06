use std::path::Path;

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let proto_dir = Path::new(&manifest_dir).join("proto");
    let mut config = prost_build::Config::new();
    // Decode raw_data as bytes::Bytes (zero-copy sub-slice of the input buffer).
    // This enables computing byte offsets into the original file for DISK-backed weight loading.
    config.bytes([".onnx.TensorProto.raw_data"]);
    config.compile_protos(&[proto_dir.join("onnx.proto")], &[&proto_dir]).unwrap();
    generate_node_tests();
    generate_light_tests();
}

/// Generate a Clang, LLVM, and availability-gated AMD and CUDA test module for each case.
fn write_backend_test(code: &mut String, fn_name: &str, ignored: bool, helper_call: &str) {
    let attr = if ignored { "#[ignore]\n        " } else { "" };
    code.push_str(&format!(
        "\
mod {fn_name} {{
    use super::*;

    #[test]
    {attr}fn clang() {{
        ::svod_schedule::testing::setup_test_tracing();
        let config = svod_tensor::PrepareConfig::for_cpu_backend(svod_tensor::CpuBackend::Clang);
        {helper_call}
    }}

    #[test]
    {attr}fn llvm() {{
        ::svod_schedule::testing::setup_test_tracing();
        let config = svod_tensor::PrepareConfig::for_cpu_backend(svod_tensor::CpuBackend::Llvm);
        {helper_call}
    }}

    #[test]
    {attr}fn amd() {{
        ::svod_schedule::testing::setup_test_tracing();
        let Some(config) = svod_tensor::PrepareConfig::for_amd_if_available() else {{
            eprintln!(\"AMD ONNX variant skipped: no active supported AMD device\");
            return;
        }};
        {helper_call}
    }}

    #[test]
    {attr}fn cuda() {{
        ::svod_schedule::testing::setup_test_tracing();
        let Some(config) = svod_tensor::PrepareConfig::for_cuda_if_available() else {{
            eprintln!(\"CUDA ONNX variant skipped: no active CUDA device\");
            return;
        }};
        {helper_call}
    }}
}}

"
    ));
}

/// Resolve the ONNX backend test data root.
/// Prefers `ONNX_TEST_DATA` env var (set by Nix), falls back to the git submodule.
fn onnx_test_data_dir() -> Option<std::path::PathBuf> {
    println!("cargo:rerun-if-env-changed=ONNX_TEST_DATA");
    if let Ok(dir) = std::env::var("ONNX_TEST_DATA") {
        let p = Path::new(&dir).to_path_buf();
        if p.exists() {
            return Some(p);
        }
    }
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let p = Path::new(&manifest_dir).join("../submodules/onnx/onnx/backend/test/data");
    if p.exists() { Some(p) } else { None }
}

fn generate_node_tests() {
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let out_path = Path::new(&out_dir).join("onnx_node_tests.rs");

    let Some(test_data) = onnx_test_data_dir() else {
        std::fs::write(&out_path, "// ONNX test data not found\n").unwrap();
        return;
    };
    let node_dir = test_data.join("node");
    if !node_dir.exists() {
        std::fs::write(&out_path, "// ONNX node test data not found\n").unwrap();
        return;
    }
    let node_dir_str = node_dir.display();

    let mut test_names: Vec<String> = std::fs::read_dir(&node_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|ft| ft.is_dir()).unwrap_or(false))
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    test_names.sort();

    let mut code = String::new();
    for name in &test_names {
        let ignored = should_skip(name);
        let helper_call = format!("run_onnx_node_test(\"{node_dir_str}/{name}\", &config);");
        write_backend_test(&mut code, name, ignored, &helper_call);
    }

    std::fs::write(&out_path, code).unwrap();
}

fn generate_light_tests() {
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let out_path = Path::new(&out_dir).join("onnx_light_tests.rs");

    let Some(test_data) = onnx_test_data_dir() else {
        std::fs::write(&out_path, "// ONNX test data not found\n").unwrap();
        return;
    };
    let light_dir = test_data.join("light");
    if !light_dir.exists() {
        std::fs::write(&out_path, "// ONNX light test data not found\n").unwrap();
        return;
    }
    let light_dir_str = light_dir.display();

    let mut models: Vec<String> = std::fs::read_dir(&light_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            name.strip_suffix(".onnx").map(String::from)
        })
        .collect();
    models.sort();

    const SKIP_LIGHT: &[&str] = &[];

    let mut code = String::new();
    for name in &models {
        let ignored = SKIP_LIGHT.contains(&name.as_str());
        let helper_call = format!(
            "run_onnx_light_test(\
             \"{light_dir_str}/{name}.onnx\", \
             \"{light_dir_str}/{name}_output_0.pb\", &config);"
        );
        write_backend_test(&mut code, name, ignored, &helper_call);
    }

    std::fs::write(&out_path, code).unwrap();
}

fn should_skip(name: &str) -> bool {
    const SKIP_PREFIXES: &[&str] = &[
        // ML domain ops
        "test_ai_onnx_ml_",
        // String ops (unsupported string dtype)
        "test_string_",
        "test_strnormalizer_",
        "test_tfidfvectorizer_",
        "test_equal_string",
        "test_regex_full_match_",
        // Sequence ops
        "test_sequence_",
        "test_split_to_sequence_",
        // Control flow iteration (we support If, not Loop/Scan)
        "test_loop",
        "test_scan",
        // Training ops (ai.onnx.preview.training domain)
        "test_adagrad",
        "test_adam",
        "test_momentum",
        "test_nesterov_momentum",
        "test_training_dropout",
        // Image decoding (unsupported ImageDecoder op)
        "test_image_decoder_",
        // Quantization
        "test_quantize",
        "test_dequantize",
        "test_dynamicquantize",
        // Deformable convolution
        "test_basic_deform_conv",
        "test_deform_conv",
        // NMS
        "test_nonmaxsuppression_",
        // ROI align
        "test_roialign_",
        // Unique values
        "test_unique_",
        // Signal processing
        "test_dft",
        "test_stft",
        "test_melweight",
        // Window functions
        "test_hannwindow",
        "test_hammingwindow",
        "test_blackmanwindow",
        // Random (non-deterministic)
        "test_bernoulli",
        // Optional type ops
        "test_optional_",
        // Exotic dtype casts
        "test_cast_e8m0_",
        "test_cast_no_saturate_",
        "test_castlike_no_saturate_",
    ];

    const SKIP_EXACT: &[&str] = &[
        // The int8 expectations come from onnx's `op_qlinear_matmul.py`, which
        // casts without clipping; the spec, QLinearConv's reference, and ONNX
        // Runtime saturate, and so does svod.
        "test_qlinearmatmul_2D_int8_float16",
        "test_qlinearmatmul_2D_int8_float32",
        "test_qlinearmatmul_3D_int8_float16",
        "test_qlinearmatmul_3D_int8_float32",
        "test_batchnorm_example_training_mode",
        "test_batchnorm_epsilon_training_mode",
        "test_dropout_random_old",
        "test_constantofshape_int_shape_zero",
        // If variants using sequence/optional types in subgraphs
        "test_if_seq",
        "test_if_opt",
        // Identity variants using optional/sequence container types
        "test_identity_opt",
        "test_identity_sequence",
        // Expanded subgraphs using If with incompatible branch shapes
        "test_affine_grid_2d_expanded",
        "test_affine_grid_2d_align_corners_expanded",
        "test_affine_grid_3d_expanded",
        "test_affine_grid_3d_align_corners_expanded",
        // Expanded subgraphs using Loop (unsupported)
        "test_range_float_type_positive_delta_expanded",
        "test_range_int32_type_negative_delta_expanded",
    ];

    const SKIP_CONTAINS: &[&str] = &["INT4", "UINT4", "INT2", "UINT2", "FLOAT4E2M1", "FLOAT8E8M0", "COMPLEX", "FNUZ"];

    SKIP_PREFIXES.iter().any(|p| name.starts_with(p))
        || SKIP_EXACT.contains(&name)
        || SKIP_CONTAINS.iter().any(|c| name.contains(c))
}
