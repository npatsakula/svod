# svod-onnx

![ONNX coverage](assets/coverage.svg)

ONNX model frontend for Svod. Parses `.onnx` files and builds lazy Svod
tensor graphs that can be compiled once and executed repeatedly.

## Quick Start

Build the input tensors yourself and hand them to the importer: the graph is
traced on them, so the buffers you hold are the ones the kernels read.

```rust
use std::collections::HashMap;

use prost::Message;
use svod_onnx::parser::onnx::ModelProto;
use svod_onnx::{OnnxImporter, OnnxModel};
use svod_tensor::Tensor;

let proto = ModelProto::decode(std::fs::read("model.onnx")?.as_slice())?;
let input = Tensor::from_ndarray(&first_batch); // same shape and dtype as the graph input

let OnnxModel { outputs, variables, .. } = OnnxImporter::new().import_model_with_inputs(
    proto,
    HashMap::from([("input".to_string(), input.clone())]),
    &[("batch_size", 1), ("sequence_length", 512)], // bind dim_param dimensions
)?;
```

`variables` holds one `Variable` per named `dim_param`. `import(path, ..)`
memory-maps weights lazily from the file instead of decoding them up front.

## Compile Once, Run Many

```rust
let plan = Tensor::prepare_batch(outputs.values())?; // schedule + compile, once
plan.execute()?;

for batch in batches {
    input.array_view_mut::<f32>()?.as_slice_mut().unwrap().copy_from_slice(&batch);
    plan.execute()?; // replay: no tracing, no compilation
}
```

See the [ONNX guide](https://svod.vpermilp.online/docs/onnx) for entry points,
dynamic dimensions and the supported contrib ops.

## Control Flow — If via Where

ONNX `If` nodes execute both branches and merge results with
`Tensor::where_()`. The condition selects elements lazily at runtime,
enabling the compile-once / run-many pattern for models with data-dependent
branching (e.g., Silero VAD).

Both branches must produce outputs with identical shapes and dtypes.
Models with incompatible branches (e.g., expanded AffineGrid) are
rejected at import time.

## Operator Support

See [PARITY.md](PARITY.md) for the full operator support table with per-operator
test results from the ONNX backend conformance suite.

To regenerate (runs tests automatically, nightly toolchain required):

```bash
uv run --with='onnx' python onnx/scripts/parity.py
```
