# svod-ir

The single intermediate representation of the Svod ML compiler. A program is a
DAG of hash-consed `UOp` nodes (`Arc<UOp>`: an `Op`, a `DType` and sources), so
identical subgraphs are the same pointer from tensor graph to linearized kernel.
The crate holds the `Op` enum, the typed constructors (`try_add`, `try_mul`,
`try_cmplt`, `try_reshape`, `try_reduce_axis`, `range`, `index`, `load`,
`store`, `wmma`, ...), symbolic integers (`SInt`), the pattern matcher and
`graph_rewrite` engine that every pass is built on, and origin tracking that
attributes nodes and kernels to module paths, call sites and ONNX nodes.

## Example

```rust
use svod_ir::{ConstValue, UOp};
use svod_dtype::DType;

let a = UOp::const_(DType::Float32, ConstValue::Float(1.0));
let b = UOp::const_(DType::Float32, ConstValue::Float(2.0));

// Constructors are fallible: mismatched dtypes or shapes return an error.
let sum = a.try_add(&b)?;
println!("{}", sum.tree());
```

## Origin tracking

Capture is off by default; `SVOD_ORIGIN=1` or `origin::capture_for_thread(true)`
turns it on. Every node built inside an `OriginScope` records the innermost scope
(constants, buffers and index arithmetic stay origin-free), and the profiler rolls
kernel time up along that path.

```rust
use svod_ir::origin::{self, OriginScope};

let _capture = origin::capture_for_thread(true);
let _encoder = OriginScope::module("encoder");
let _layer = OriginScope::module("layers.3");
let product = a.try_mul(&b)?;
assert_eq!(origin::path(product.origin().unwrap()), "encoder.layers.3");
```

## Documentation

- IR design: <https://svod.vpermilp.online/docs/architecture/ir-design>
- Op reference: <https://svod.vpermilp.online/docs/architecture/op-bestiary>
- Pattern engine: <https://svod.vpermilp.online/docs/architecture/optimizations/pattern-system>
- Kernel origins: <https://svod.vpermilp.online/docs/architecture/kernel-origins>
