---
sidebar_label: 张量 API
---

# 张量 API 示例

`svod-tensor` 是 Svod 中每个模型所依赖的惰性张量层。本页按你实际使用的顺序讲解：
构建计算图，运行它，读取结果，然后把一个图编译一次、重放多次。随附的模型见
[运行模型](./models)；`.onnx` 文件见 [ONNX 推理](./onnx)。

```toml
[dependencies]
svod-tensor = "0.1"
svod-dtype  = "0.1"   # DType
ndarray     = "0.17"            # array!, views
```

这些 crate 已发布到 crates.io；要跟踪 `main` 分支，用
`git = "https://github.com/npatsakula/svod"` 代替版本号。

**唯一的规则：**操作只构建图，不执行任何东西。`realize()` 编译并执行这个图；
`prepare()` 把它编译成一个计划，你可以执行任意多次。

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

- `Tensor::from_slice` 接受任何 `AsRef<[T]>`——数组、`Vec`、切片——并把它复制到
  默认设备上的一个缓冲区。
- 二元运算符返回 `Result<Tensor>`：形状或 dtype 不匹配是可恢复的错误，因此要写 `?`。
  两侧都可以是 `&Tensor` 或拥有所有权的 `Tensor`；右侧还可以是标量，它会以张量的
  dtype 实例化。左侧的标量需要显式类型：`2.0f32 * &a`。一元 `-&a` 不会失败，返回
  普通的 `Tensor`。
- `realize(&self)` 调度张量背后的整个图，尽可能融合，编译内核并运行。已执行的
  张量可以继续放在共享借用之后。

读回数据：

| 方法 | 是否触发 realize？ | 返回 |
|---|---|---|
| `as_vec::<T>()`、`as_ndarray::<T>()` | 从不——未执行时返回 `NoBuffer` 错误 | 拥有所有权的副本 |
| `to_vec::<T>()`、`to_ndarray::<T>()` | 按需 | 拥有所有权的副本 |
| `item::<T>()` | 按需 | 唯一的那个元素 |
| `array_view::<T>()`、`array_view_mut::<T>()` | 从不 | 借用的 `ndarray` 视图，零拷贝 |

因此最短的写法是 `(&a + &b)?.to_vec::<f32>()?`。在隐式 realize 会构成 bug 的地方
使用 `as_*` 系列。视图要求缓冲区可映射到宿主（CPU，或带宿主映射的 GPU 缓冲区），
并且张量要拥有自己的缓冲区而不是某个缓冲区的视图。

---

## 形状与广播

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

| 操作 | 效果 |
|---|---|
| `try_reshape(&[2, 3])` | 新形状，元素总数不变 |
| `try_reshape(&[-1, 3])` | `-1` 由总数推断该轴 |
| `try_transpose(0, 1)` | 交换两个轴 |
| `try_permute(&[0, 2, 1])` | 重新排列所有轴 |
| `try_squeeze(Some(dim))` / `try_squeeze(None)` | 去掉一个大小为 1 的轴，或全部去掉 |
| `try_unsqueeze(dim)` | 插入一个大小为 1 的轴 |
| `try_expand(&[3, 2])` | 广播大小为 1 的轴，不复制数据 |
| `try_shrink([(0, 2), (1, 3)])` | 按轴切片一个范围 |
| `try_pad(&[(1, 1), (0, 0)])` | 按轴补零 |
| `Tensor::cat(&[&a, &b], dim)` / `Tensor::stack(..)` | 拼接 / 堆叠 |

负数轴在所有地方都从末尾计数。`Tensor::from_ndarray` 把数组复制一次；非连续数组会
经过一个中间 `Vec`。

**查看形状。**`dims()` 返回 `Vec<usize>`，任一轴是符号维度时失败；`dim(axis)` 返回
`SInt`（常量或符号）；`dim_const(axis)` 返回 `usize`，否则 `NonConstDim`；`shape()`
是完整的 `Shape`。`dtype()` 和 `device()` 不会失败。`Tensor` 实现了 `Debug`，只打印
元数据——`Tensor { shape: [4], dtype: Scalar(Float32), device: Cpu, realized: false }`——
从不打印数据，否则会强制从设备读取。

**广播**遵循 NumPy 的规则：形状从右对齐，每个轴要么相等要么为 1。

```text
[3, 2] + [1, 2] -> [3, 2]
[3, 2] + [2]    -> [3, 2]   (implicit [1, 2])
[3, 2] + [3]    -> error    ("cannot broadcast shapes", reported as ErrorKind::UOp)
```

---

## 矩阵乘法

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

`dot`（别名 `matmul`）把左操作数的最后一个轴与右操作数的倒数第二个轴缩并；前导的
批次轴会广播。

| 左 | 右 | 结果 |
|---|---|---|
| `[M, K]` | `[K, N]` | `[M, N]` |
| `[K]` | `[K, N]` | `[N]` |
| `[M, K]` | `[K]` | `[M]` |
| `[B, M, K]` | `[K, N]` 或 `[B, K, N]` | `[B, M, N]` |

`K` 不匹配时报 `DotShapeMismatch`。`matmul_with().other(&w).dtype(DType::Float32).call()`
选择累加器的 dtype，这对 fp16/bf16 输入很重要。

---

## 一个小分类器

`nn::Linear` 计算 `x @ W.T + b`，权重布局采用 PyTorch 的 `[out, in]`。`sequential`
把任何实现了 `Layer` 的东西串起来：

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

`realize_batch` 接受一个 `&Tensor` 迭代器；共享的子图（这里是 logits）只计算一次。
`Relu` 是一个零大小的 `Layer`；同样的激活函数也作为张量方法存在（`relu`、`sigmoid`、
`silu`、`gelu`、`softmax`、`log_softmax`），归约 `sum`、`mean`、`max` 和 `argmax` 也是，
它们都接受一个轴，或用 `()` 表示"所有轴"。

---

## 模块与检查点

一个层结构体拥有它的参数，以及前向计算需要的超参数。`#[derive(Module)]` 把这些字段
变成一个扁平的 `StateDict`（`HashMap<String, Tensor>`），键名与 PyTorch 完全一致，
因此检查点无需手写映射即可加载：

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

| 属性 | 效果 |
|---|---|
| `#[module(key = "Wi.weight")]` | 替换字段名对应的键段（可以包含点和数字） |
| `#[module(key = "")]` | 扁平化：该字段的键直接使用父前缀 |
| `#[module(skip)]` | 忽略一个非基本类型字段（配置、dtype、模式） |
| `#[module(optional)]` | `Option<Tensor>` 必须加：为 `Some` 时保存，加载时容忍缺失的键 |
| `#[module(optional = "self.has_bias")]` | 谓词为真时该键必需，否则跳过 |

子模块通过泛型实现组合：`Vec<M>` 和 `[M; N]` 以 `0.`、`1.`、…为元素编号；`Option<M>`、
`Box<M>` 和 `(A, B)` 以同样方式委托，枚举也可以派生。前向计算不属于 `Module`：
签名允许时放在 `Layer::forward` 中，否则放在固有方法中。

内置层同时实现了这两个 trait，`new` 用于已加载的张量，`with_dims` 用于全新初始化
（Kaiming 均匀分布的权重和零偏置；归一化层为恒等仿射）：

| 层 | `with_dims` | state-dict 键 |
|---|---|---|
| `Linear` | `(in, out, bias, dtype)` | `weight`、`bias`（存在时） |
| `Conv1d` | `(in_c, out_c, kernel, bias, dtype)` | `weight`、`bias` |
| `Conv2d` / `ConvTranspose2d` | `(in_c, out_c, (kh, kw), bias, dtype)` | `weight`、`bias` |
| `BatchNorm2d` | `(channels, eps, dtype)` | `weight`、`bias`、`running_mean`、`running_var` |
| `LayerNorm` | `(size, bias, eps, dtype)` | `weight`、`bias`（存在时） |
| `RmsNorm` | `(size, eps, dtype)` | `weight` |
| `Embedding` | `(vocab_size, embed_dim, dtype)` | `weight` |

超参数是结构体上的构建器风格方法——
`Conv1d::new(w, bias).with_stride(2).with_padding((1, 1)).with_groups(4)`、
`LayerNorm::with_dims(..).with_axis(-2)`。池化、group norm 和 dropout 是张量方法
（`max_pool2d`、`avg_pool2d`、`group_norm`、`dropout`），不是结构体。

检查点来自 `svod-model`，它按存储格式读取 safetensors（f32、f16、bf16、fp8……），
只在要求时才转换：

```rust
use std::path::Path;
use svod_model::state::{cast_all, load_safetensors, load_safetensors_dir};
use svod_tensor::nn::Module;

let sd = load_safetensors(Path::new("model.safetensors"))?;      // one file
let sd = load_safetensors_dir(Path::new("checkpoint/"))?;        // or the shards in model.safetensors.index.json
let sd = cast_all(&sd, DType::Float16);
block.load_state_dict(&sd, "layers.0")?;
```

Hub 加载器（`ResNet::from_hub`、`GigaAm::from_hub_with_revision`……）就是这个模式加上
一次 Hugging Face 下载；见[运行模型](./models)。

---

## 编译一次，运行多次

`realize()` 每次调用都会调度和编译。在新数据上反复运行同一个图的模型——每个推理
服务都是如此——应当用 `prepare()` 编译一次，然后重放计划：

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

`prepare()` 把 `energy` 接到计划的输出缓冲区上，因此在第一次 `execute()` 之前它就已经
报告 `realized: true`；只在执行过一次之后再读取它。`Tensor::prepare_batch([&a, &b])`
把多个输出编译进一个计划，`prepare_with(&PrepareConfig)` 接受显式配置
（`prepare()` 使用的是 `PrepareConfig::from_env()`；`PrepareConfig::device_local()`
让输出留在设备上，不做宿主映射）。

重放的计划做不到两件事：改变形状，以及改变在构建时已折叠进图里的值。对于可变的
批大小或序列长度，模型层提供 `jit_wrapper!`，它声明符号边界、分配输入缓冲区并在每次
调用时重新绑定变量——见 [JIT 图](./architecture/jit-graphs)。对于 ONNX 模型，导入时
的 `dim_bindings` 做同样的事。

---

## 设备

张量在默认设备上创建：若设置了 `SVOD_DEVICE` 则用它，否则 macOS 上是 `METAL:0`，
其他平台是 `CPU`。拼写形式为 `NAME[:index]`，不区分大小写：

| `SVOD_DEVICE` | 后端 |
|---|---|
| `CPU` | 进程内编译的 LLVM IR（`SVOD_CPU_BACKEND=clang` 选择 C 后端） |
| `CUDA:0`（别名 `GPU`） | NVIDIA，运行时加载 `libcuda.so.1` |
| `AMD:0`（别名 `HIP`） | AMD，直接使用 KFD 队列 |
| `METAL:0` | Apple GPU |

```rust
use svod_dtype::DeviceSpec;
use svod_tensor::{Tensor, set_default_device, with_default_device};

let on_gpu = cpu_tensor.to(DeviceSpec::Cuda { device_id: 0 });   // lazy COPY node
set_default_device(DeviceSpec::Cuda { device_id: 0 });            // this thread, from now on
with_default_device(DeviceSpec::Cpu, || Tensor::zeros(&[4], DType::Float32));  // scoped
```

编译好的内核缓存在磁盘上（`~/.cache/svod/objects`，或 `$SVOD_OBJECT_CACHE_DIR`；
`SVOD_OBJECT_CACHE=0` 关闭缓存），因此进程第二次启动时跳过编译。`SVOD_THREADS`
限制编译和 CPU 执行的线程池，`BEAM=N` 开启
[内核搜索](./architecture/optimizations/kernel-search)，`SVOD_NOOPT=1` 关闭优化器以便
二分排查。

---

## 深入内部

张量背后的图是一棵 `UOp` 树：

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

这是调度之前的图：`BUFFER` 节点代表两个输入，`Add` 代表运算；load、store 和 range
只在调度器把它变成内核之后才出现。要看到那些，先准备一个计划并打印它的内核：

```rust
let plan = c.prepare()?;
for kernel in plan.kernels() {
    println!("{}\n{}", kernel.entry_point, kernel.code);   // one fused kernel, LLVM IR on the CPU
}
```

`SVOD_DUMP_LLVM_IR=<dir>` 把每个内核的 IR 写到 `<dir>/<name>.ll`，无需改动代码；
`SVOD_DUMP_LINEAR=<dir>` 转储线性化后的 UOp 程序。[执行流水线](./architecture/pipeline)
页面跟踪一个内核走过的每个阶段。

---

## 循环层

`rnn()`、`gru()` 和 `lstm()` 是 `Tensor` 上的构建器。它们既接受 PyTorch 的权重名
（`weight_ih`、`weight_hh`、`bias_ih`、`bias_hh`、`h0`、`c0`），也接受 ONNX 的
（`w`、`r`——在 `gru()` 上写作 `r_weights`——`bias`、`initial_h`、`initial_c`），
并替你重排门的分块：

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

`layout` 选择 `RnnLayout::SeqFirst`（`[seq, batch, input]`，默认）或 `BatchFirst`；
`direction` 接受 `RnnDirection::{Forward, Backward, Bidirectional}`，双向时两个方向
在特征轴上拼接。GRU 的 `linear_before_reset` 在使用 PyTorch 权重时默认取 PyTorch 的
位置，使用 ONNX 权重时取 ONNX 的。`LstmOutput` 额外带有细胞状态 `y_c` / `c_n`。

时间轴必须是具体值，批次轴可以是符号维度。手写循环——例如逐 token 步进的解码器——
直接使用 cell：`RnnCell`/`GruCell` 提供 `step(&x, &h) -> Result<Tensor>`，`LstmCell`
提供 `step(&x, &h, &c) -> Result<(Tensor, Tensor)>`，`RnnStack::new(cells)` 一次步进
整个栈。

---

## 频谱图

`stft()` 是对加窗 DFT 核做的一次 `conv1d`，因此整个变换都留在图里（批次轴也可以保持
符号化）。结果是 `[B, F, T, 2]`——未加批次的 `[L]` 信号则是 `[F, T, 2]`——末轴为
`(real, imag)`，与 `torch.stft(..., return_complex=false)` 一致：

```rust
use svod_tensor::Tensor;
use svod_tensor::nn::Window;

let x = Tensor::from_slice(vec![0.25f32; 64]);
let spec = x.stft().n_fft(16).hop(4).window(Window::Hann).call()?;
assert_eq!(spec.dims()?, vec![9, 17, 2]);   // [F, T, (re, im)]

let mag = spec.magnitude(0.0)?;             // sqrt(re² + im² + eps)
let signal = spec.istft().n_fft(16).hop(4).window(Window::Hann).length(64).call()?;
```

默认值沿用 torch：`hop = n_fft / 4`、`win_length = n_fft`、周期性 Hann 窗、`center`
（反射填充）、`onesided`、不归一化——`istft` 必须给相同的参数。`Window` 是 `Hann`、
`Hamming`、`Povey`、`Rectangular` 或 `Custom(tensor)`，
`Tensor::window(&Window::Hann, n, periodic, dtype)` 实例化一个窗。除了 `magnitude`，
末轴为 2 的张量还有 `power`、`complex_abs`、`complex_mul` 和
`Tensor::complex_from_polar(&mag, &phase)`。

梅尔前端是同一个图，末尾加上滤波器组缩并和一个对数。`mel_spectrogram()` 接受
`stft` 的分帧参数加上梅尔参数，返回 `[B, n_mels, T]`（未加批次时为 `[n_mels, T]`）：

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

默认值是 torchaudio 的 `MelSpectrogram`（HTK 刻度、不归一化、`power = 2`、`f_min = 0`、
`f_max = sample_rate / 2`、无对数）；`MelScale::Slaney` 配合 `MelNorm::Slaney` 就是
`librosa.filters.mel`，即 Whisper 背后的滤波器组。`MelLog::Ln { min, max }` 是
`ln(clamp(x))`，`MelLog::Whisper` 是 `log_mel_spectrogram` 末尾的
`log10` / 下限 `max - 8` / `(x + 4) / 4`；`mel_log` 可单独应用任一种，`preemphasis` 和
`remove_dc` 覆盖 Kaldi 风格的前端，`filterbank(&t)` 替换为预先计算的 `[n_mels, F]` 表
（`Tensor::mel_filterbank(...)` 可以生成一个）。

---

## 错误

每个可能失败的张量方法都返回 `svod_tensor::error::Result<T>`，其错误是指针大小的
`Error(Box<ErrorKind>)`；通过 `err.kind()`（或 `into_kind()` 取得所有权）匹配原因。
下游 crate 用 snafu 的 `context(false)` 转换它，因此模型自己的错误枚举用一个普通的 `?`
就能吸收——不必在每个调用点写 `.context(TensorSnafu)`。

并非一切都可能失败。`cast`、`neg`、`abs`、`floor`、`ceil`、`round`、`trunc`、`square`、
`sign` 以及 `Tensor::full` / `zeros` / `ones` 构造函数不会失败，返回普通的 `Tensor`；
`-&a` 同样如此，而二元运算符返回 `Result<Tensor>`。

---

## 小结

| 任务 | 代码 |
|---|---|
| 创建张量 | `Tensor::from_slice([1.0f32, 2.0])`、`Tensor::from_ndarray(&arr)` |
| 算术 | `(&a + &b)?`、`(&a * 2.0)?`、`(2.0f32 * &a)?`、`-&a` |
| 变形 | `t.try_reshape(&[2, 3])?` |
| 转置 | `t.try_transpose(0, 1)?` |
| 矩阵乘法 | `a.dot(&b)?` |
| 查看 | `t.dims()?`、`t.dim_const(-1)?`、`t.dtype()` |
| 线性层 | `Linear::with_dims(in, out, bias, dtype)` |
| 串联层 | `x.sequential(&[&fc1, &Relu, &fc2])?` |
| 激活 | `t.relu()?`、`t.softmax(-1)?` |
| 加载权重 | `model.load_state_dict(&sd, "")?` |
| 频谱图 | `x.stft().n_fft(512).hop(160).call()?` |
| 梅尔频谱图 | `x.mel_spectrogram().sample_rate(16000).n_fft(400).n_mels(80).call()?` |
| 循环层 | `x.lstm().weight_ih(&w).weight_hh(&r).hidden_size(h).call()?` |
| 执行 | `t.realize()?` |
| 批量执行 | `Tensor::realize_batch([&a, &b])?` |
| 编译一次 | `let plan = t.prepare()?; plan.execute()?` |
| 提取数据 | `t.to_vec::<f32>()?`、`t.to_ndarray::<f32>()?`、`t.item::<f32>()?` |
| 选择设备 | `SVOD_DEVICE=CUDA:0`、`t.to(DeviceSpec::Cuda { device_id: 0 })` |

**下一步：**

- [运行模型](./models)——随附的语音、文本和视觉模型
- [ONNX 推理](./onnx)——把 `.onnx` 文件导入同一个图
- [JIT 图](./architecture/jit-graphs)——`jit_wrapper!`、符号批次和设备上的状态
- [执行流水线](./architecture/pipeline)——图如何变成内核
