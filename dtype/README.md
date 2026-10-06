# svod-dtype

The type system of the Svod ML compiler, and the lowest crate in its dependency
graph. `DType` covers scalars (`Bool`, `Int8`–`Int64`, `UInt8`–`UInt64`, the FP8
formats `FP8E4M3`/`FP8E5M2` and their `FNUZ` variants, `Float16`, `BFloat16`,
`Float32`, `Float64`, `Index`, `Void`, and the weak literal types), vectors,
pointers tagged with an address space (`Global`, `Local`, `Reg`) and image types,
together with Tinygrad's promotion lattice (`DType::least_upper_dtype`) and
safe-cast rules. It also defines `DeviceSpec`, the GPU arch enums (`AmdArch`,
`CudaArch`, `MetalFamily`) and the process default device
(`default_device::default_device`: `SVOD_DEVICE`, else `METAL:0` on macOS and
`CPU` elsewhere).

## Example

```rust
use svod_dtype::{AddrSpace, DType};

let vec4 = DType::Float32.vec(4).expect("a scalar vectorizes");
let ptr = DType::Float32.ptr(None, AddrSpace::Global).expect("a scalar has a pointer type");
assert_eq!(vec4.bytes(), 16);
assert_eq!(ptr.base(), DType::Float32.base());

// Mixed operands promote along the lattice.
assert_eq!(DType::least_upper_dtype(&[DType::Int8, DType::Float16]), Some(DType::Float16));
```

## Features

| Feature | Default | Effect |
|---------|---------|--------|
| `serde` | yes | `Serialize`/`Deserialize` for the public types |
| `proptest` | no | `Arbitrary` derives and strategies for property tests |

Documentation: <https://svod.vpermilp.online/docs/architecture/ir-design>
