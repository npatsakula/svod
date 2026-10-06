# svod-tensor

Lazy tensor API: operations build a UOp graph and run nothing; `realize()`
schedules, compiles and executes it; `prepare()` compiles it once into a plan
you replay.

```rust
use svod_tensor::Tensor;

let a = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);
let b = Tensor::from_slice([10.0f32, 20.0, 30.0, 40.0]);

let scaled = ((&a + &b)? * 0.1)?;  // nothing runs yet
scaled.realize()?;                 // schedule, compile, execute
assert_eq!(scaled.as_vec::<f32>()?.len(), 4);
```

## Compile once, run many

```rust
let input = Tensor::from_slice(vec![0.0f32; 1024]);
let energy = input.try_mul(&input)?.mean(())?;

let plan = energy.prepare()?;      // schedule + compile, once
for frame in frames {
    input.array_view_mut::<f32>()?.as_slice_mut().unwrap().copy_from_slice(frame);
    plan.execute()?;               // replay: no tracing, no compilation
    println!("{}", energy.item::<f32>()?);
}
```

`Tensor::prepare_batch([&a, &b])` compiles several outputs into one plan.
`SVOD_THREADS` (default: host parallelism) bounds both kernel compilation and
CPU kernel execution.

## Reading data

| Method | Realizes? | Returns |
|---|---|---|
| `as_vec`, `as_ndarray` | never | owned copy |
| `to_vec`, `to_ndarray`, `item` | on demand | owned copy / element |
| `array_view`, `array_view_mut` | never | borrowed `ndarray` view, zero-copy |

`Tensor::from_slice` and `Tensor::from_ndarray` copy the host data into a new
buffer once.

See the [Tensor API guide](https://svod.vpermilp.online/docs/examples) for
shapes, modules, checkpoints, devices, recurrent layers and spectrograms.
