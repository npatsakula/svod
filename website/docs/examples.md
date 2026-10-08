---
sidebar_label: Tensor API
---

# Tensor API by Example

`svod-tensor` is the lazy tensor layer every model in Svod is written on. This
page walks through it in the order you will use it: build a graph, run it, read
the result, then compile a graph once and replay it. For the shipped models see
[Running models](./models); for `.onnx` files see [ONNX inference](./onnx).

```toml
[dependencies]
svod-tensor = "0.2"
svod-dtype  = "0.2"   # DType
ndarray     = "0.17"            # array!, views
```

The crates are published on crates.io; to track `main` use
`git = "https://github.com/npatsakula/svod"` instead of a version.

**The one rule:** operations build a graph and run nothing. `realize()` compiles
and executes the graph; `prepare()` compiles it into a plan you execute as many
times as you like.

---

## Hello tensor

```rust
use svod_tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);
    let b = Tensor::from_slice([10.0f32, 20.0, 30.0, 40.0]);

    let sum = (&a + &b)?;          // nothing runs yet
    let scaled = (&sum * 0.1)?;    // a scalar is a valid operand

    scaled.realize()?;             // schedule, compile, execute
    println!("{:?}", scaled.as_vec::<f32>()?);   // [1.1, 2.2, 3.3, 4.4]
    Ok(())
}
```

- `Tensor::from_slice` takes anything `AsRef<[T]>` — an array, a `Vec`, a
  slice — and copies it into a buffer on the default device.
- The binary operators return `Result<Tensor>`: a shape or dtype mismatch is a
  recoverable error, hence the `?`. Both sides may be `&Tensor` or an owned
  `Tensor`; the right-hand side may also be a scalar, which is materialized in
  the tensor's dtype. A scalar on the left needs an explicit type:
  `2.0f32 * &a`. Unary `-&a` cannot fail and returns a plain `Tensor`.
- `realize(&self)` schedules the whole graph behind the tensor, fuses what it
  can, compiles the kernels and runs them. A realized tensor stays behind a
  shared borrow.

Reading data back:

| Method | Realizes? | Returns |
|---|---|---|
| `as_vec::<T>()`, `as_ndarray::<T>()` | never — `NoBuffer` error if unrealized | owned copy |
| `to_vec::<T>()`, `to_ndarray::<T>()` | on demand | owned copy |
| `item::<T>()` | on demand | the single element |
| `array_view::<T>()`, `array_view_mut::<T>()` | never | borrowed `ndarray` view, zero-copy |

So the shortest form is `(&a + &b)?.to_vec::<f32>()?`. Use the `as_*` family
where a hidden realize would be a bug. The views need a host-mappable buffer
(CPU, or a GPU buffer with a host mapping) and a tensor that owns its buffer
rather than a view of one.

---

## Shapes and broadcasting

```rust
use ndarray::array;
use svod_tensor::Tensor;

fn shapes() -> Result<(), Box<dyn std::error::Error>> {
    let data = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    println!("{:?}", data.dims()?);                       // [6]

    let matrix = data.try_reshape(&[2, 3])?;              // [[1, 2, 3], [4, 5, 6]]
    let same = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    assert_eq!(matrix.dims()?, same.dims()?);

    let transposed = matrix.try_transpose(0, 1)?;         // [3, 2]

    // [3, 2] + [1, 2] -> [3, 2]: the row vector is added to every row
    let bias = Tensor::from_ndarray(&array![[100.0f32, 200.0]]);
    let biased = (&transposed + &bias)?;
    println!("{:?}", biased.to_ndarray::<f32>()?);
    // [[101, 204],
    //  [102, 205],
    //  [103, 206]]
    Ok(())
}
```

| Operation | Effect |
|---|---|
| `try_reshape(&[2, 3])` | New shape, same element count |
| `try_reshape(&[-1, 3])` | `-1` infers that axis from the total |
| `try_transpose(0, 1)` | Swap two axes |
| `try_permute(&[0, 2, 1])` | Reorder all axes |
| `try_squeeze(Some(dim))` / `try_squeeze(None)` | Drop one size-1 axis, or all of them |
| `try_unsqueeze(dim)` | Insert a size-1 axis |
| `try_expand(&[3, 2])` | Broadcast a size-1 axis without copying |
| `try_shrink([(0, 2), (1, 3)])` | Slice a range per axis |
| `try_pad(&[(1, 1), (0, 0)])` | Zero-pad per axis |
| `Tensor::cat(&[&a, &b], dim)` / `Tensor::stack(..)` | Concatenate / stack |

Negative axes count from the end everywhere. `Tensor::from_ndarray` copies the
array once; a non-contiguous array goes through an intermediate `Vec`.

**Inspecting a shape.** `dims()` returns `Vec<usize>` and fails if any axis is
symbolic; `dim(axis)` returns an `SInt` (constant or symbolic); `dim_const(axis)`
returns `usize` or `NonConstDim`; `shape()` is the whole `Shape`. `dtype()` and
`device()` are infallible. `Tensor` implements `Debug` and prints only
metadata — `Tensor { shape: [4], dtype: Scalar(Float32), device: Cpu, realized: false }` —
never the data, which would force a device read.

**Broadcasting** follows NumPy: shapes align from the right and every axis
must match or be 1.

```text
[3, 2] + [1, 2] -> [3, 2]
[3, 2] + [2]    -> [3, 2]   (implicit [1, 2])
[3, 2] + [3]    -> error    ("cannot broadcast shapes", reported as ErrorKind::UOp)
```

---

## Matrix multiply

```rust
use ndarray::array;
use svod_tensor::Tensor;

fn matmul() -> Result<(), Box<dyn std::error::Error>> {
    // 4 samples with 3 features each
    let input = Tensor::from_ndarray(&array![
        [1.0f32, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
    ]);
    // 3 features -> 2 outputs
    let weights = Tensor::from_ndarray(&array![[0.1f32, 0.2], [0.3, 0.4], [0.5, 0.6]]);

    let output = input.dot(&weights)?;                   // [4, 3] @ [3, 2] -> [4, 2]
    println!("{:?}", output.to_ndarray::<f32>()?);
    Ok(())
}
```

`dot` (alias `matmul`) contracts the last axis of the left operand with the
second-to-last of the right one; leading batch axes broadcast.

| Left | Right | Result |
|---|---|---|
| `[M, K]` | `[K, N]` | `[M, N]` |
| `[K]` | `[K, N]` | `[N]` |
| `[M, K]` | `[K]` | `[M]` |
| `[B, M, K]` | `[K, N]` or `[B, K, N]` | `[B, M, N]` |

A `K` mismatch is `DotShapeMismatch`. `matmul_with().other(&w).dtype(DType::Float32).call()`
chooses the accumulator dtype, which matters for fp16/bf16 inputs.

---

## A small classifier

`nn::Linear` computes `x @ W.T + b` with PyTorch's `[out, in]` weight layout.
`sequential` chains anything implementing `Layer`:

```rust
use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::{Layer, Linear, Relu};

fn classify() -> Result<(), Box<dyn std::error::Error>> {
    // 784 (28x28 pixels) -> 128 -> 10 classes; with_dims draws Kaiming-uniform weights
    let fc1 = Linear::with_dims(784, 128, true, DType::Float32);
    let fc2 = Linear::with_dims(128, 10, true, DType::Float32);

    let pixels: Vec<f32> = (0..784).map(|i| i as f32 / 784.0).collect();
    let image = Tensor::from_slice(pixels).try_reshape(&[1, 784])?;   // batch of 1

    let logits = image.sequential(&[&fc1, &Relu, &fc2])?;
    let probs = logits.softmax(-1)?;
    let prediction = logits.argmax(-1)?;                // Int32 indices

    // Two results sharing the logits: one schedule, one run
    Tensor::realize_batch([&probs, &prediction])?;
    println!("{:?}", probs.as_ndarray::<f32>()?);
    println!("{:?}", prediction.as_vec::<i32>()?);
    Ok(())
}
```

```rust
pub trait Layer {
    fn forward(&self, x: &Tensor) -> Result<Tensor>;
}
```

`realize_batch` takes an iterator of `&Tensor`; the shared subgraph (the logits)
is computed once. `Relu` is a zero-sized `Layer`; the same activations exist as
tensor methods (`relu`, `sigmoid`, `silu`, `gelu`, `softmax`, `log_softmax`),
as do the reductions `sum`, `mean`, `max` and `argmax`, all taking an axis or
`()` for "all axes".

---

## Modules and checkpoints

A layer struct owns its parameters plus the hyper-parameters its forward
needs. `#[derive(Module)]` turns the fields into a flat `StateDict`
(`HashMap<String, Tensor>`) keyed exactly as PyTorch names them, so a
checkpoint loads without a hand-written mapping:

```rust
use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::{LayerNorm, Module, StateDict};

#[derive(Clone, Module)]
struct Block {
    intermediate: usize,            // primitives are skipped automatically
    #[module(skip)]                 // a non-primitive that carries no weights
    dtype: DType,
    norm: LayerNorm,                // child module: "norm.weight", "norm.bias"
    #[module(key = "Wi.weight")]    // checkpoint name, dots allowed
    wi: Tensor,
    #[module(key = "Wo.weight")]
    wo: Tensor,
    #[module(optional)]             // written when Some, absent-tolerant on load
    out_bias: Option<Tensor>,
}

fn load(checkpoint: &StateDict) -> Result<Block, Box<dyn std::error::Error>> {
    let mut block = Block {
        intermediate: 3072,
        dtype: DType::Float32,
        norm: LayerNorm::with_dims(768, true, 1e-5, DType::Float32),
        wi: Tensor::zeros(&[3072, 768], DType::Float32),
        wo: Tensor::zeros(&[768, 3072], DType::Float32),
        out_bias: None,
    };
    // Reads "layers.0.norm.weight", "layers.0.Wi.weight", ...
    block.load_state_dict(checkpoint, "layers.0")?;
    // ...and writes them back out under any prefix
    let _round_trip: StateDict = block.state_dict("layers.0");
    Ok(block)
}
```

| Attribute | Effect |
|---|---|
| `#[module(key = "Wi.weight")]` | Replace the field-name key segment (may contain dots and digits) |
| `#[module(key = "")]` | Flatten: the field's keys use the parent prefix unchanged |
| `#[module(skip)]` | Ignore a non-primitive field (config, dtype, mode) |
| `#[module(optional)]` | Required on `Option<Tensor>`: saved when `Some`, load tolerates an absent key |
| `#[module(optional = "self.has_bias")]` | The key is required when the predicate holds and skipped otherwise |

Children compose through blanket impls: `Vec<M>` and `[M; N]` key their
elements `0.`, `1.`, …; `Option<M>`, `Box<M>` and `(A, B)` delegate the same
way, and enums derive too. The forward pass stays out of `Module`: it lives in
`Layer::forward` when the signature allows and in inherent methods otherwise.

The built-in layers implement both traits, with `new` for loaded tensors and
`with_dims` for a fresh initialization (Kaiming-uniform weights and zero
biases; identity affine for the normalizations):

| Layer | `with_dims` | State-dict keys |
|---|---|---|
| `Linear` | `(in, out, bias, dtype)` | `weight`, `bias` (when present) |
| `Conv1d` | `(in_c, out_c, kernel, bias, dtype)` | `weight`, `bias` |
| `Conv2d` / `ConvTranspose2d` | `(in_c, out_c, (kh, kw), bias, dtype)` | `weight`, `bias` |
| `BatchNorm2d` | `(channels, eps, dtype)` | `weight`, `bias`, `running_mean`, `running_var` |
| `LayerNorm` | `(size, bias, eps, dtype)` | `weight`, `bias` (when present) |
| `RmsNorm` | `(size, eps, dtype)` | `weight` |
| `Embedding` | `(vocab_size, embed_dim, dtype)` | `weight` |

Hyper-parameters are builder-style methods on the struct —
`Conv1d::new(w, bias).with_stride(2).with_padding((1, 1)).with_groups(4)`,
`LayerNorm::with_dims(..).with_axis(-2)`. Pooling, group norm and dropout are
tensor methods (`max_pool2d`, `avg_pool2d`, `group_norm`, `dropout`), not
structs.

Checkpoints come from `svod-model`, which reads safetensors as stored
(f32, f16, bf16, fp8, …) and casts only on request:

```rust
use std::path::Path;
use svod_model::state::{cast_all, load_safetensors, load_safetensors_dir};
use svod_tensor::nn::Module;

let sd = load_safetensors(Path::new("model.safetensors"))?;      // one file
let sd = load_safetensors_dir(Path::new("checkpoint/"))?;        // or the shards in model.safetensors.index.json
let sd = cast_all(&sd, DType::Float16);
block.load_state_dict(&sd, "layers.0")?;
```

The hub loaders (`ResNet::from_hub`, `GigaAm::from_hub_with_revision`, …) are
this pattern plus a Hugging Face download; see [Running models](./models).

---

## Compile once, run many

`realize()` schedules and compiles every time it is called. A model that runs
the same graph on new data — every inference server — should compile it once
with `prepare()` and replay the plan:

```rust
use svod_tensor::Tensor;

fn stream(frames: &[Vec<f32>]) -> Result<(), Box<dyn std::error::Error>> {
    let input = Tensor::from_slice(vec![0.0f32; 1024]);    // owns a host-mappable buffer
    let energy = input.try_mul(&input)?.mean(())?;

    let plan = energy.prepare()?;                          // schedule + compile, once
    for frame in frames {
        input.array_view_mut::<f32>()?.as_slice_mut().unwrap().copy_from_slice(frame);
        plan.execute()?;                                   // replay: no tracing, no compilation
        println!("{}", energy.item::<f32>()?);
    }
    Ok(())
}
```

`prepare()` wires `energy` to the plan's output buffer, so it already reports
`realized: true` before the first `execute()`; read it only after one.
`Tensor::prepare_batch([&a, &b])` compiles several outputs into one plan, and
`prepare_with(&PrepareConfig)` takes an explicit configuration
(`PrepareConfig::from_env()` is what `prepare()` uses; `PrepareConfig::device_local()`
keeps the outputs on the device with no host mapping).

Two things a replayed plan cannot do: change the shape, and change a value
that was folded into the graph at build time. For a variable batch or sequence
length the model layer offers `jit_wrapper!`, which declares symbolic bounds,
allocates the input buffers and rebinds the variables per call — see
[JIT graphs](./architecture/jit-graphs). For an ONNX model, `dim_bindings` on
import does the same job.

---

## Devices

Tensors are created on the default device: `SVOD_DEVICE` if set, otherwise
`METAL:0` on macOS and `CPU` elsewhere. The spelling is `NAME[:index]`,
case-insensitive:

| `SVOD_DEVICE` | Backend |
|---|---|
| `CPU` | LLVM IR compiled in-process (`SVOD_CPU_BACKEND=clang` selects the C backend) |
| `CUDA:0` (alias `GPU`) | NVIDIA, `libcuda.so.1` loaded at runtime |
| `AMD:0` (alias `HIP`) | AMD, direct KFD queues |
| `METAL:0` | Apple GPU |

```rust
use svod_dtype::DeviceSpec;
use svod_tensor::{Tensor, set_default_device, with_default_device};

let on_gpu = cpu_tensor.to(DeviceSpec::Cuda { device_id: 0 });   // lazy COPY node
set_default_device(DeviceSpec::Cuda { device_id: 0 });            // this thread, from now on
with_default_device(DeviceSpec::Cpu, || Tensor::zeros(&[4], DType::Float32));  // scoped
```

Compiled kernels are cached on disk (`~/.cache/svod/objects`, or
`$SVOD_OBJECT_CACHE_DIR`; `SVOD_OBJECT_CACHE=0` disables it), so the second
process start skips compilation. `SVOD_THREADS` bounds the compile and CPU
execution thread pool, `BEAM=N` turns on
[kernel search](./architecture/optimizations/kernel-search), and `SVOD_NOOPT=1`
disables the optimizer for bisection.

---

## Under the hood

The graph behind a tensor is a tree of `UOp`s:

```rust
let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
let b = Tensor::from_slice([4.0f32, 5.0, 6.0]);
let c = (&a + &b)?;
println!("{}", c.uop().tree());
```

```text
[8592] Add : Scalar(Float32) shape=[Const(3)]
├── [8590] BUFFER(slot=34, addrspace=Some(Global)) : Scalar(Float32) shape=[Const(3)]
│   └── [882] CONST(Int(3)) : Scalar(WeakInt) shape=[]
└── [8591] BUFFER(slot=35, addrspace=Some(Global)) : Scalar(Float32) shape=[Const(3)]
    └── [882] → (see above)
```

This is the pre-schedule graph: `BUFFER` nodes stand for the two inputs and
`Add` for the operation; loads, stores and ranges only appear once the
scheduler has turned it into kernels. To see those, prepare a plan and print
its kernels:

```rust
let plan = c.prepare()?;
for kernel in plan.kernels() {
    println!("{}\n{}", kernel.entry_point, kernel.code);   // one fused kernel, LLVM IR on the CPU
}
```

`SVOD_DUMP_LLVM_IR=<dir>` writes every kernel's IR to `<dir>/<name>.ll` without
touching the code, and `SVOD_DUMP_LINEAR=<dir>` dumps the linearized UOp
program. The [execution pipeline](./architecture/pipeline) page follows one
kernel through every stage.

---

## Recurrent layers

`rnn()`, `gru()` and `lstm()` are builders on `Tensor`. They accept either the
PyTorch weight names (`weight_ih`, `weight_hh`, `bias_ih`, `bias_hh`, `h0`,
`c0`) or the ONNX ones (`w`, `r` — spelled `r_weights` on `gru()` — `bias`,
`initial_h`, `initial_c`), and reorder the gate blocks for you:

```rust
use ndarray::Array3;
use svod_tensor::Tensor;

// seq=2, batch=1, input=3, hidden=4
let x = Tensor::from_ndarray(&Array3::from_elem((2, 1, 3), 0.1f32));
let w = Tensor::from_ndarray(&Array3::from_elem((1, 12, 3), 0.1f32));
let r = Tensor::from_ndarray(&Array3::from_elem((1, 12, 4), 0.1f32));

let out = x.gru().w(&w).r_weights(&r).hidden_size(4).call()?;
// ONNX-shaped: y [seq, num_directions, batch, hidden], y_h [num_directions, batch, hidden]
// PyTorch-shaped: output [seq, batch, D*hidden], h_n [num_directions, batch, hidden]
assert_eq!(out.y.dims()?, vec![2, 1, 1, 4]);
assert_eq!(out.output.dims()?, vec![2, 1, 4]);
```

`layout` picks `RnnLayout::SeqFirst` (`[seq, batch, input]`, the default) or
`BatchFirst`; `direction` takes `RnnDirection::{Forward, Backward, Bidirectional}`, and a bidirectional pass concatenates the two directions on
the feature axis. The GRU's `linear_before_reset` defaults to PyTorch's
placement with PyTorch weights and to ONNX's with ONNX weights. `LstmOutput`
adds `y_c` / `c_n` for the cell state.

The time axis must be concrete, but the batch axis may be symbolic. For a
hand-rolled loop — a decoder stepping one token at a time — use the cells
directly: `RnnCell`/`GruCell` expose `step(&x, &h) -> Result<Tensor>`,
`LstmCell` exposes `step(&x, &h, &c) -> Result<(Tensor, Tensor)>`, and
`RnnStack::new(cells)` steps a whole stack at once.

---

## Spectrograms

`stft()` is one `conv1d` against a windowed DFT kernel, so the whole transform
stays in the graph (and the batch axis may stay symbolic). The result is
`[B, F, T, 2]` — or `[F, T, 2]` for an unbatched `[L]` signal — with
`(real, imag)` on the trailing axis, matching
`torch.stft(..., return_complex=false)`:

```rust
use svod_tensor::Tensor;
use svod_tensor::nn::Window;

let x = Tensor::from_slice(vec![0.25f32; 64]);
let spec = x.stft().n_fft(16).hop(4).window(Window::Hann).call()?;
assert_eq!(spec.dims()?, vec![9, 17, 2]);   // [F, T, (re, im)]

let mag = spec.magnitude(0.0)?;             // sqrt(re² + im² + eps)
let signal = spec.istft().n_fft(16).hop(4).window(Window::Hann).length(64).call()?;
```

Defaults follow torch: `hop = n_fft / 4`, `win_length = n_fft`, a periodic Hann
window, `center` (reflect padding), `onesided`, no normalization — and `istft`
must be given the same ones. `Window` is `Hann`, `Hamming`, `Povey`,
`Rectangular` or `Custom(tensor)`, and `Tensor::window(&Window::Hann, n, periodic, dtype)`
materializes one. Alongside `magnitude`, the trailing-2 axis has `power`,
`complex_abs`, `complex_mul` and `Tensor::complex_from_polar(&mag, &phase)`.

A mel front-end is the same graph with a filterbank contraction and a log on
the end. `mel_spectrogram()` takes the `stft` framing parameters plus the mel
ones and returns `[B, n_mels, T]` (`[n_mels, T]` unbatched):

```rust
use svod_tensor::Tensor;
use svod_tensor::nn::{MelLog, MelNorm, MelScale};

let x = Tensor::from_slice(vec![0.25f32; 16000]);
let mel = x
    .mel_spectrogram()
    .sample_rate(16000)
    .n_fft(400)
    .hop(160)
    .n_mels(80)
    .mel_scale(MelScale::Slaney)
    .norm(MelNorm::Slaney)
    .log(MelLog::Whisper)
    .call()?;
assert_eq!(mel.dims()?, vec![80, 101]);
```

The defaults are torchaudio's `MelSpectrogram` (HTK scale, no normalization,
`power = 2`, `f_min = 0`, `f_max = sample_rate / 2`, no log); `MelScale::Slaney`
with `MelNorm::Slaney` is `librosa.filters.mel`, the filterbank behind Whisper.
`MelLog::Ln { min, max }` is `ln(clamp(x))` and `MelLog::Whisper` the
`log10` / floor-at-`max - 8` / `(x + 4) / 4` tail of `log_mel_spectrogram`;
`mel_log` applies either on its own, `preemphasis` and `remove_dc` cover the
Kaldi-style front-ends, and `filterbank(&t)` substitutes a precomputed
`[n_mels, F]` table (`Tensor::mel_filterbank(...)` builds one).

---

## Errors

Every fallible tensor method returns `svod_tensor::error::Result<T>`, whose
error is a pointer-sized `Error(Box<ErrorKind>)`; match on the cause through
`err.kind()` (or `into_kind()` to take it by value). Downstream crates convert
it with snafu's `context(false)`, so a model's own error enum absorbs it with a
plain `?` — no `.context(TensorSnafu)` at every call site.

Not everything is fallible. `cast`, `neg`, `abs`, `floor`, `ceil`, `round`,
`trunc`, `square`, `sign` and the `Tensor::full` / `zeros` / `ones`
constructors cannot fail and return a plain `Tensor`; `-&a` is likewise plain,
while the binary operators return `Result<Tensor>`.

---

## Summary

| Task | Code |
|---|---|
| Create tensor | `Tensor::from_slice([1.0f32, 2.0])`, `Tensor::from_ndarray(&arr)` |
| Arithmetic | `(&a + &b)?`, `(&a * 2.0)?`, `(2.0f32 * &a)?`, `-&a` |
| Reshape | `t.try_reshape(&[2, 3])?` |
| Transpose | `t.try_transpose(0, 1)?` |
| Matrix multiply | `a.dot(&b)?` |
| Inspect | `t.dims()?`, `t.dim_const(-1)?`, `t.dtype()` |
| Linear layer | `Linear::with_dims(in, out, bias, dtype)` |
| Chain layers | `x.sequential(&[&fc1, &Relu, &fc2])?` |
| Activation | `t.relu()?`, `t.softmax(-1)?` |
| Load weights | `model.load_state_dict(&sd, "")?` |
| Spectrogram | `x.stft().n_fft(512).hop(160).call()?` |
| Mel spectrogram | `x.mel_spectrogram().sample_rate(16000).n_fft(400).n_mels(80).call()?` |
| Recurrent layer | `x.lstm().weight_ih(&w).weight_hh(&r).hidden_size(h).call()?` |
| Execute | `t.realize()?` |
| Batch realize | `Tensor::realize_batch([&a, &b])?` |
| Compile once | `let plan = t.prepare()?; plan.execute()?` |
| Extract data | `t.to_vec::<f32>()?`, `t.to_ndarray::<f32>()?`, `t.item::<f32>()?` |
| Pick a device | `SVOD_DEVICE=CUDA:0`, `t.to(DeviceSpec::Cuda { device_id: 0 })` |

**Next steps:**

- [Running models](./models) — the shipped speech, text and vision models
- [ONNX inference](./onnx) — import a `.onnx` file into the same graph
- [JIT graphs](./architecture/jit-graphs) — `jit_wrapper!`, symbolic batches and on-device state
- [Execution pipeline](./architecture/pipeline) — how a graph becomes kernels
