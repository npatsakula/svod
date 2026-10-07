---
sidebar_label: टेंसर API
---

# उदाहरणों के साथ टेंसर API

`svod-tensor` वह लेज़ी टेंसर परत है जिस पर Svod का हर मॉडल लिखा गया है। यह
पेज उसे उसी क्रम में दिखाता है जिसमें आप उसका उपयोग करेंगे: ग्राफ़ बनाना, उसे चलाना,
परिणाम पढ़ना, और फिर ग्राफ़ को एक बार कंपाइल करके बार-बार दोहराना। साथ आने वाले मॉडलों के लिए
[मॉडल चलाना](./models) देखें; `.onnx` फ़ाइलों के लिए [ONNX इन्फ़रेंस](./onnx) देखें।

```toml
[dependencies]
svod-tensor = "0.2"
svod-dtype  = "0.2"   # DType
ndarray     = "0.17"            # array!, views
```

क्रेट crates.io पर प्रकाशित हैं; `main` को ट्रैक करने के लिए वर्ज़न की जगह
`git = "https://github.com/npatsakula/svod"` लिखें।

**एकमात्र नियम:** ऑपरेशन ग्राफ़ बनाते हैं और कुछ भी नहीं चलाते। realize (`realize()`) ग्राफ़ को
कंपाइल करके चलाता है; `prepare()` उसे एक प्लान में कंपाइल करता है जिसे आप जितनी बार
चाहें चला सकते हैं।

---

## पहला टेंसर

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

- `Tensor::from_slice` कोई भी `AsRef<[T]>` लेता है — array, `Vec`,
  slice — और उसे डिफ़ॉल्ट डिवाइस पर एक बफ़र में कॉपी करता है।
- बाइनरी ऑपरेटर `Result<Tensor>` लौटाते हैं: shape या dtype का मेल न खाना एक
  रिकवर करने योग्य त्रुटि है, इसलिए `?`। दोनों पक्ष `&Tensor` या owned
  `Tensor` हो सकते हैं; दायाँ पक्ष scalar भी हो सकता है, जो टेंसर के dtype में
  materialize होता है। बाईं ओर के scalar को स्पष्ट टाइप चाहिए:
  `2.0f32 * &a`। यूनरी `-&a` विफल नहीं हो सकता और सीधा `Tensor` लौटाता है।
- `realize(&self)` टेंसर के पीछे के पूरे ग्राफ़ को शेड्यूल करता है, जो फ़्यूज़ हो सकता है उसे
  फ़्यूज़ करता है, कर्नेल कंपाइल करता है और उन्हें चलाता है। realize हुआ टेंसर shared borrow
  के पीछे ही रहता है।

डेटा वापस पढ़ना:

| मेथड | realize करता है? | लौटाता है |
|---|---|---|
| `as_vec::<T>()`, `as_ndarray::<T>()` | कभी नहीं — realize न होने पर `NoBuffer` त्रुटि | owned कॉपी |
| `to_vec::<T>()`, `to_ndarray::<T>()` | ज़रूरत पड़ने पर | owned कॉपी |
| `item::<T>()` | ज़रूरत पड़ने पर | एकमात्र एलिमेंट |
| `array_view::<T>()`, `array_view_mut::<T>()` | कभी नहीं | उधार लिया `ndarray` view, zero-copy |

इसलिए सबसे छोटा रूप `(&a + &b)?.to_vec::<f32>()?` है। जहाँ छिपा हुआ realize एक बग होगा,
वहाँ `as_*` परिवार का उपयोग करें। views को host-mappable बफ़र चाहिए
(CPU, या host mapping वाला GPU बफ़र) और ऐसा टेंसर जो अपने बफ़र का मालिक हो,
न कि किसी बफ़र का view।

---

## Shapes और ब्रॉडकास्टिंग

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

| ऑपरेशन | प्रभाव |
|---|---|
| `try_reshape(&[2, 3])` | नया shape, एलिमेंट की संख्या वही |
| `try_reshape(&[-1, 3])` | `-1` उस axis को कुल संख्या से निकालता है |
| `try_transpose(0, 1)` | दो axes की अदला-बदली |
| `try_permute(&[0, 2, 1])` | सभी axes का क्रम बदलना |
| `try_squeeze(Some(dim))` / `try_squeeze(None)` | एक size-1 axis हटाना, या सभी |
| `try_unsqueeze(dim)` | एक size-1 axis जोड़ना |
| `try_expand(&[3, 2])` | size-1 axis को बिना कॉपी के ब्रॉडकास्ट करना |
| `try_shrink([(0, 2), (1, 3)])` | हर axis पर एक range काटना |
| `try_pad(&[(1, 1), (0, 0)])` | हर axis पर शून्य से पैडिंग |
| `Tensor::cat(&[&a, &b], dim)` / `Tensor::stack(..)` | जोड़ना / स्टैक करना |

ऋणात्मक axes हर जगह अंत से गिने जाते हैं। `Tensor::from_ndarray` array को
एक बार कॉपी करता है; non-contiguous array एक मध्यवर्ती `Vec` से होकर जाता है।

**Shape की जाँच।** `dims()` `Vec<usize>` लौटाता है और कोई भी axis
symbolic हो तो विफल होता है; `dim(axis)` एक `SInt` (स्थिर या symbolic) लौटाता है; `dim_const(axis)`
`usize` या `NonConstDim` लौटाता है; `shape()` पूरा `Shape` है। `dtype()` और
`device()` विफल नहीं होते। `Tensor` `Debug` लागू करता है और केवल
मेटाडेटा प्रिंट करता है — `Tensor { shape: [4], dtype: Scalar(Float32), device: Cpu, realized: false }` —
डेटा कभी नहीं, क्योंकि उसके लिए डिवाइस से पढ़ना ज़रूरी होगा।

**ब्रॉडकास्टिंग** NumPy का पालन करती है: shapes दाईं ओर से संरेखित होते हैं और हर axis
का मेल खाना या 1 होना ज़रूरी है।

```text
[3, 2] + [1, 2] -> [3, 2]
[3, 2] + [2]    -> [3, 2]   (implicit [1, 2])
[3, 2] + [3]    -> error    ("cannot broadcast shapes", reported as ErrorKind::UOp)
```

---

## मैट्रिक्स गुणन

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

`dot` (उपनाम `matmul`) बाएँ ऑपरेंड के अंतिम axis को दाएँ ऑपरेंड के
अंत से दूसरे axis के साथ contract करता है; आगे के batch axes ब्रॉडकास्ट होते हैं।

| बायाँ | दायाँ | परिणाम |
|---|---|---|
| `[M, K]` | `[K, N]` | `[M, N]` |
| `[K]` | `[K, N]` | `[N]` |
| `[M, K]` | `[K]` | `[M]` |
| `[B, M, K]` | `[K, N]` या `[B, K, N]` | `[B, M, N]` |

`K` का मेल न खाना `DotShapeMismatch` है। `matmul_with().other(&w).dtype(DType::Float32).call()`
accumulator का dtype चुनता है, जो fp16/bf16 इनपुट के लिए मायने रखता है।

---

## एक छोटा क्लासिफ़ायर

`nn::Linear` PyTorch के `[out, in]` weight लेआउट के साथ `x @ W.T + b` की गणना करता है।
`sequential` `Layer` लागू करने वाली किसी भी चीज़ को श्रृंखला में जोड़ता है:

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

`realize_batch` `&Tensor` का iterator लेता है; साझा सबग्राफ़ (logits)
एक बार गणना होता है। `Relu` एक zero-sized `Layer` है; यही activations
टेंसर मेथड के रूप में भी मौजूद हैं (`relu`, `sigmoid`, `silu`, `gelu`, `softmax`, `log_softmax`),
और इसी तरह reductions `sum`, `mean`, `max` और `argmax` भी, जो सभी एक axis या
"सभी axes" के लिए `()` लेते हैं।

---

## मॉड्यूल और चेकपॉइंट

एक layer struct अपने पैरामीटर और वे hyper-parameters रखता है जिनकी उसके forward को
ज़रूरत है। `#[derive(Module)]` फ़ील्ड्स को एक सपाट `StateDict`
(`HashMap<String, Tensor>`) में बदलता है, जिसकी keys ठीक वैसी होती हैं जैसे PyTorch उन्हें नाम देता है, ताकि
चेकपॉइंट हाथ से लिखी mapping के बिना लोड हो:

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

| एट्रिब्यूट | प्रभाव |
|---|---|
| `#[module(key = "Wi.weight")]` | फ़ील्ड-नाम वाले key खंड को बदलता है (इसमें बिंदु और अंक हो सकते हैं) |
| `#[module(key = "")]` | सपाट करना: फ़ील्ड की keys पैरेंट prefix को बिना बदले उपयोग करती हैं |
| `#[module(skip)]` | non-primitive फ़ील्ड (config, dtype, mode) को अनदेखा करना |
| `#[module(optional)]` | `Option<Tensor>` पर अनिवार्य: `Some` होने पर सहेजा जाता है, लोड के समय key का न होना स्वीकार्य है |
| `#[module(optional = "self.has_bias")]` | predicate सत्य होने पर key अनिवार्य है, अन्यथा छोड़ दी जाती है |

चाइल्ड मॉड्यूल blanket impls के ज़रिए जुड़ते हैं: `Vec<M>` और `[M; N]` अपने
एलिमेंट्स को `0.`, `1.`, … keys देते हैं; `Option<M>`, `Box<M>` और `(A, B)` इसी तरह
delegate करते हैं, और enums भी derive होते हैं। forward pass `Module` से बाहर रहता है: वह
signature अनुमति दे तो `Layer::forward` में और अन्यथा inherent methods में रहता है।

बिल्ट-इन layers दोनों traits लागू करती हैं, लोड किए गए टेंसरों के लिए `new` और
नई initialization के लिए `with_dims` के साथ (Kaiming-uniform weights और शून्य
biases; normalizations के लिए identity affine):

| Layer | `with_dims` | State-dict keys |
|---|---|---|
| `Linear` | `(in, out, bias, dtype)` | `weight`, `bias` (मौजूद होने पर) |
| `Conv1d` | `(in_c, out_c, kernel, bias, dtype)` | `weight`, `bias` |
| `Conv2d` / `ConvTranspose2d` | `(in_c, out_c, (kh, kw), bias, dtype)` | `weight`, `bias` |
| `BatchNorm2d` | `(channels, eps, dtype)` | `weight`, `bias`, `running_mean`, `running_var` |
| `LayerNorm` | `(size, bias, eps, dtype)` | `weight`, `bias` (मौजूद होने पर) |
| `RmsNorm` | `(size, eps, dtype)` | `weight` |
| `Embedding` | `(vocab_size, embed_dim, dtype)` | `weight` |

Hyper-parameters struct पर builder-शैली के मेथड हैं —
`Conv1d::new(w, bias).with_stride(2).with_padding((1, 1)).with_groups(4)`,
`LayerNorm::with_dims(..).with_axis(-2)`। Pooling, group norm और dropout
टेंसर मेथड हैं (`max_pool2d`, `avg_pool2d`, `group_norm`, `dropout`),
structs नहीं।

चेकपॉइंट `svod-model` से आते हैं, जो safetensors को उसी रूप में पढ़ता है जैसे वे संग्रहीत हैं
(f32, f16, bf16, fp8, …) और केवल अनुरोध पर cast करता है:

```rust
use std::path::Path;
use svod_model::state::{cast_all, load_safetensors, load_safetensors_dir};
use svod_tensor::nn::Module;

let sd = load_safetensors(Path::new("model.safetensors"))?;      // one file
let sd = load_safetensors_dir(Path::new("checkpoint/"))?;        // or the shards in model.safetensors.index.json
let sd = cast_all(&sd, DType::Float16);
block.load_state_dict(&sd, "layers.0")?;
```

हब लोडर (`ResNet::from_hub`, `GigaAm::from_hub_with_revision`, …) यही
पैटर्न हैं, साथ में Hugging Face से डाउनलोड; [मॉडल चलाना](./models) देखें।

---

## एक बार कंपाइल, कई बार रन

`realize()` हर बार बुलाए जाने पर शेड्यूल और कंपाइल करता है। जो मॉडल नए डेटा पर
वही ग्राफ़ चलाता है — हर इन्फ़रेंस सर्वर — उसे `prepare()` से एक बार कंपाइल करना चाहिए
और प्लान को दोहराना चाहिए:

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

`prepare()` `energy` को प्लान के output बफ़र से जोड़ता है, इसलिए वह पहले
`execute()` से पहले ही `realized: true` दिखाता है; उसे कम से कम एक रन के बाद ही पढ़ें।
`Tensor::prepare_batch([&a, &b])` कई outputs को एक प्लान में कंपाइल करता है, और
`prepare_with(&PrepareConfig)` एक स्पष्ट कॉन्फ़िगरेशन लेता है
(`prepare()` `PrepareConfig::from_env()` का उपयोग करता है; `PrepareConfig::device_local()`
outputs को बिना host mapping के डिवाइस पर रखता है)।

दोहराया गया प्लान दो काम नहीं कर सकता: shape बदलना, और ऐसा मान बदलना
जो बिल्ड के समय ग्राफ़ में fold हो गया था। परिवर्तनशील batch या sequence
length के लिए मॉडल परत `jit_wrapper!` देती है, जो symbolic सीमाएँ घोषित करता है,
input बफ़र आवंटित करता है और हर कॉल पर variables फिर से बाँधता है — देखें
[JIT ग्राफ़](./architecture/jit-graphs)। ONNX मॉडल के लिए import के समय `dim_bindings`
यही काम करता है।

---

## डिवाइस

टेंसर डिफ़ॉल्ट डिवाइस पर बनाए जाते हैं: सेट हो तो `SVOD_DEVICE`, अन्यथा
macOS पर `METAL:0` और बाकी जगह `CPU`। लिखने का रूप `NAME[:index]` है,
case-insensitive:

| `SVOD_DEVICE` | बैकएंड |
|---|---|
| `CPU` | प्रोसेस के भीतर कंपाइल किया गया LLVM IR (`SVOD_CPU_BACKEND=clang` C बैकएंड चुनता है) |
| `CUDA:0` (उपनाम `GPU`) | NVIDIA, `libcuda.so.1` रनटाइम पर लोड होता है |
| `AMD:0` (उपनाम `HIP`) | AMD, सीधी KFD queues |
| `METAL:0` | Apple GPU |

```rust
use svod_dtype::DeviceSpec;
use svod_tensor::{Tensor, set_default_device, with_default_device};

let on_gpu = cpu_tensor.to(DeviceSpec::Cuda { device_id: 0 });   // lazy COPY node
set_default_device(DeviceSpec::Cuda { device_id: 0 });            // this thread, from now on
with_default_device(DeviceSpec::Cpu, || Tensor::zeros(&[4], DType::Float32));  // scoped
```

कंपाइल किए गए कर्नेल डिस्क पर कैश होते हैं (`~/.cache/svod/objects`, या
`$SVOD_OBJECT_CACHE_DIR`; `SVOD_OBJECT_CACHE=0` इसे बंद करता है), इसलिए प्रोसेस की दूसरी
शुरुआत कंपाइलेशन छोड़ देती है। `SVOD_THREADS` कंपाइल और CPU
execution thread pool को सीमित करता है, `BEAM=N`
[कर्नेल सर्च](./architecture/optimizations/kernel-search) चालू करता है, और `SVOD_NOOPT=1`
bisection के लिए ऑप्टिमाइज़र बंद करता है।

---

## अंदर क्या होता है

टेंसर के पीछे का ग्राफ़ `UOp`s का एक ट्री है:

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

यह शेड्यूल से पहले का ग्राफ़ है: `BUFFER` नोड दो इनपुट को दर्शाते हैं और
`Add` ऑपरेशन को; loads, stores और ranges तभी दिखते हैं जब
शेड्यूलर इसे कर्नेलों में बदल चुका हो। उन्हें देखने के लिए एक प्लान prepare करें और
उसके कर्नेल प्रिंट करें:

```rust
let plan = c.prepare()?;
for kernel in plan.kernels() {
    println!("{}\n{}", kernel.entry_point, kernel.code);   // one fused kernel, LLVM IR on the CPU
}
```

`SVOD_DUMP_LLVM_IR=<dir>` कोड को छुए बिना हर कर्नेल का IR `<dir>/<name>.ll` में
लिखता है, और `SVOD_DUMP_LINEAR=<dir>` linearized UOp
प्रोग्राम डंप करता है। [एक्ज़िक्यूशन पाइपलाइन](./architecture/pipeline) पेज एक
कर्नेल को हर चरण से होकर दिखाता है।

---

## रिकरेंट लेयर

`rnn()`, `gru()` और `lstm()` `Tensor` पर builders हैं। वे या तो
PyTorch weight नाम (`weight_ih`, `weight_hh`, `bias_ih`, `bias_hh`, `h0`,
`c0`) या ONNX नाम (`w`, `r` — `gru()` पर इसे `r_weights` लिखा जाता है — `bias`,
`initial_h`, `initial_c`) स्वीकार करते हैं, और gate blocks का क्रम आपके लिए बदल देते हैं:

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

`layout` `RnnLayout::SeqFirst` (`[seq, batch, input]`, डिफ़ॉल्ट) या
`BatchFirst` चुनता है; `direction` `RnnDirection::{Forward, Backward, Bidirectional}` लेता है, और bidirectional pass दोनों दिशाओं को
feature axis पर जोड़ता है। GRU का `linear_before_reset` PyTorch weights के साथ डिफ़ॉल्ट रूप से PyTorch की
placement और ONNX weights के साथ ONNX की placement लेता है। `LstmOutput`
cell state के लिए `y_c` / `c_n` जोड़ता है।

समय axis ठोस होना चाहिए, लेकिन batch axis symbolic हो सकता है।
हाथ से लिखे लूप के लिए — एक बार में एक टोकन आगे बढ़ने वाला decoder — cells का
सीधे उपयोग करें: `RnnCell`/`GruCell` `step(&x, &h) -> Result<Tensor>` देते हैं,
`LstmCell` `step(&x, &h, &c) -> Result<(Tensor, Tensor)>` देता है, और
`RnnStack::new(cells)` पूरे stack को एक साथ एक कदम आगे बढ़ाता है।

---

## स्पेक्ट्रोग्राम

`stft()` एक windowed DFT कर्नेल के विरुद्ध एक `conv1d` है, इसलिए पूरा transform
ग्राफ़ में रहता है (और batch axis symbolic रह सकता है)। परिणाम
`[B, F, T, 2]` है — या unbatched `[L]` सिग्नल के लिए `[F, T, 2]` — जिसमें
अंतिम axis पर `(real, imag)` होता है, जो
`torch.stft(..., return_complex=false)` से मेल खाता है:

```rust
use svod_tensor::Tensor;
use svod_tensor::nn::Window;

let x = Tensor::from_slice(vec![0.25f32; 64]);
let spec = x.stft().n_fft(16).hop(4).window(Window::Hann).call()?;
assert_eq!(spec.dims()?, vec![9, 17, 2]);   // [F, T, (re, im)]

let mag = spec.magnitude(0.0)?;             // sqrt(re² + im² + eps)
let signal = spec.istft().n_fft(16).hop(4).window(Window::Hann).length(64).call()?;
```

डिफ़ॉल्ट torch का पालन करते हैं: `hop = n_fft / 4`, `win_length = n_fft`, periodic Hann
window, `center` (reflect padding), `onesided`, कोई normalization नहीं — और `istft`
को यही मान दिए जाने चाहिए। `Window` `Hann`, `Hamming`, `Povey`,
`Rectangular` या `Custom(tensor)` है, और `Tensor::window(&Window::Hann, n, periodic, dtype)`
एक window materialize करता है। `magnitude` के साथ, अंतिम आकार-2 axis के लिए `power`,
`complex_abs`, `complex_mul` और `Tensor::complex_from_polar(&mag, &phase)` उपलब्ध हैं।

mel front-end यही ग्राफ़ है, अंत में एक filterbank contraction और एक log के साथ।
`mel_spectrogram()` `stft` framing पैरामीटर और mel
पैरामीटर लेता है और `[B, n_mels, T]` (unbatched में `[n_mels, T]`) लौटाता है:

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

डिफ़ॉल्ट torchaudio के `MelSpectrogram` हैं (HTK scale, कोई normalization नहीं,
`power = 2`, `f_min = 0`, `f_max = sample_rate / 2`, कोई log नहीं); `MelScale::Slaney`
के साथ `MelNorm::Slaney` `librosa.filters.mel` है, Whisper के पीछे का filterbank।
`MelLog::Ln { min, max }` `ln(clamp(x))` है और `MelLog::Whisper`
`log_mel_spectrogram` का `log10` / `max - 8` पर floor / `(x + 4) / 4` वाला अंतिम हिस्सा;
`mel_log` इनमें से किसी को भी अलग से लागू करता है, `preemphasis` और `remove_dc`
Kaldi-शैली के front-ends को कवर करते हैं, और `filterbank(&t)` पहले से गणना की गई
`[n_mels, F]` तालिका लगाता है (`Tensor::mel_filterbank(...)` एक तालिका बनाता है)।

---

## त्रुटियाँ

हर विफल हो सकने वाला टेंसर मेथड `svod_tensor::error::Result<T>` लौटाता है, जिसकी
त्रुटि एक pointer-sized `Error(Box<ErrorKind>)` है; कारण पर match
`err.kind()` से करें (या मान के रूप में लेने के लिए `into_kind()`)। आगे के क्रेट इसे
snafu के `context(false)` से convert करते हैं, ताकि मॉडल का अपना error enum इसे
सादे `?` से समाहित कर ले — हर कॉल साइट पर `.context(TensorSnafu)` नहीं।

हर चीज़ विफल नहीं हो सकती। `cast`, `neg`, `abs`, `floor`, `ceil`, `round`,
`trunc`, `square`, `sign` और `Tensor::full` / `zeros` / `ones`
constructors विफल नहीं हो सकते और सीधा `Tensor` लौटाते हैं; `-&a` भी इसी तरह सीधा है,
जबकि बाइनरी ऑपरेटर `Result<Tensor>` लौटाते हैं।

---

## सारांश

| कार्य | कोड |
|---|---|
| टेंसर बनाना | `Tensor::from_slice([1.0f32, 2.0])`, `Tensor::from_ndarray(&arr)` |
| अंकगणित | `(&a + &b)?`, `(&a * 2.0)?`, `(2.0f32 * &a)?`, `-&a` |
| Reshape | `t.try_reshape(&[2, 3])?` |
| Transpose | `t.try_transpose(0, 1)?` |
| मैट्रिक्स गुणन | `a.dot(&b)?` |
| जाँच | `t.dims()?`, `t.dim_const(-1)?`, `t.dtype()` |
| Linear layer | `Linear::with_dims(in, out, bias, dtype)` |
| layers को जोड़ना | `x.sequential(&[&fc1, &Relu, &fc2])?` |
| Activation | `t.relu()?`, `t.softmax(-1)?` |
| weights लोड करना | `model.load_state_dict(&sd, "")?` |
| स्पेक्ट्रोग्राम | `x.stft().n_fft(512).hop(160).call()?` |
| Mel स्पेक्ट्रोग्राम | `x.mel_spectrogram().sample_rate(16000).n_fft(400).n_mels(80).call()?` |
| रिकरेंट लेयर | `x.lstm().weight_ih(&w).weight_hh(&r).hidden_size(h).call()?` |
| चलाना | `t.realize()?` |
| बैच realize | `Tensor::realize_batch([&a, &b])?` |
| एक बार कंपाइल | `let plan = t.prepare()?; plan.execute()?` |
| डेटा निकालना | `t.to_vec::<f32>()?`, `t.to_ndarray::<f32>()?`, `t.item::<f32>()?` |
| डिवाइस चुनना | `SVOD_DEVICE=CUDA:0`, `t.to(DeviceSpec::Cuda { device_id: 0 })` |

**अगले कदम:**

- [मॉडल चलाना](./models) — साथ आने वाले स्पीच, टेक्स्ट और विज़न मॉडल
- [ONNX इन्फ़रेंस](./onnx) — `.onnx` फ़ाइल को उसी ग्राफ़ में इम्पोर्ट करना
- [JIT ग्राफ़](./architecture/jit-graphs) — `jit_wrapper!`, symbolic batches और डिवाइस पर state
- [एक्ज़िक्यूशन पाइपलाइन](./architecture/pipeline) — ग्राफ़ कर्नेलों में कैसे बदलता है
