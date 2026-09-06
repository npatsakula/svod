---
sidebar_label: परिचय
---

# Svod

> **अल्फ़ा सॉफ़्टवेयर।** कोर फ़ंक्शनैलिटी टेस्टेड है, लेकिन APIs अस्थिर हैं और बिना नोटिस के बदल सकते हैं।

Svod एक Rust-आधारित ML कम्पाइलर है जो [Tinygrad](https://github.com/tinygrad/tinygrad) से प्रेरित है। इसमें UOp-आधारित IR के साथ lazy tensor इवैल्यूएशन, पैटर्न-ड्रिवन ऑप्टिमाइज़ेशन, और मल्टी-बैकएंड कोड जनरेशन शामिल है।

## मुख्य विशेषताएँ

| विशेषता | विवरण |
|---------|-------|
| **डिक्लेरेटिव ऑप्टिमाइज़ेशन** | Z3-सत्यापित शुद्धता के साथ ग्राफ़ रीराइट्स के लिए `patterns!` DSL |
| **Lazy इवैल्यूएशन** | Tensor कम्प्यूटेशन ग्राफ़ बनाते हैं, कम्पाइल केवल `realize()` पर होता है |
| **Origin ट्रैकिंग** | scopes हर कर्नेल को उसके module path, call site या ONNX node से जोड़ते हैं |
| **80+ IR ऑपरेशन** | अरिथमेटिक, मेमोरी, कंट्रोल फ़्लो, WMMA tensor cores |
| **20+ ऑप्टिमाइज़ेशन** | कॉन्स्टेंट फ़ोल्डिंग, tensor cores, वेक्टराइज़ेशन, लूप अनरोलिंग |

आर्किटेक्चर की डिटेल्स के लिए [डॉक्यूमेंटेशन साइट](https://npatsakula.github.io/svod/) देखें।

## वर्कस्पेस

| Crate | विवरण |
|-------|-------|
| [dtype](https://github.com/npatsakula/svod/tree/main/dtype/) | टाइप सिस्टम: scalars, vectors, pointers, images |
| [macros](https://github.com/npatsakula/svod/tree/main/macros/) | प्रोसीज़रल मैक्रोज़ (`patterns!` DSL) |
| [ir](https://github.com/npatsakula/svod/tree/main/ir/) | UOp ग्राफ़ IR: 80+ ops, symbolic integers, origins |
| [device](https://github.com/npatsakula/svod/tree/main/device/) | बफ़र मैनेजमेंट: lazy alloc, zero-copy views, LRU कैशिंग |
| [schedule](https://github.com/npatsakula/svod/tree/main/schedule/) | ऑप्टिमाइज़ेशन इंजन: 20+ पासेज़, RANGEIFY, Z3 वेरिफ़िकेशन |
| [codegen](https://github.com/npatsakula/svod/tree/main/codegen/) | कोड जनरेशन: Clang (डिफ़ॉल्ट), LLVM JIT |
| [runtime](https://github.com/npatsakula/svod/tree/main/runtime/) | JIT कम्पाइलेशन और कर्नेल एक्ज़ीक्यूशन |
| [tensor](https://github.com/npatsakula/svod/tree/main/tensor/) | हाई-लेवल lazy tensor API |
| [onnx](https://github.com/npatsakula/svod/tree/main/onnx/) | ONNX मॉडल इम्पोर्टर |
| [arch](https://github.com/npatsakula/svod/tree/main/arch/) | इन्फ़रेंस प्रिमिटिव्स |
| [model](https://github.com/npatsakula/svod/tree/main/model/) | प्री-ट्रेन्ड मॉडल्स |

## त्वरित उदाहरण

```rust
use svod_tensor::Tensor;
use ndarray::array;

// Zero-copy from ndarray (C-contiguous fast path)
let a = Tensor::from_ndarray(&array![[1.0f32, 2.0], [3.0, 4.0]]);
let b = Tensor::from_ndarray(&array![[5.0f32, 6.0], [7.0, 8.0]]);

// Lazy — nothing executes yet
let mut c = &a + &b;
c.realize()?;

// Zero-copy view into the result
let view = c.array_view::<f32>()?;
assert_eq!(view, array![[6.0, 8.0], [10.0, 12.0]].into_dyn());
```

## Pattern DSL उदाहरण

```rust
use svod_schedule::patterns;

let optimizer = patterns! {
    // Identity folding (commutative)
    Add[x, @zero] => x,
    Mul[x, @one] => x,

    // Constant folding
    for op in binary [Add, Mul, Sub] {
        op(a @const(av), _b @const(bv))
          => eval_binary_op(op, av, bv).map(|r| UOp::const_(a.dtype(), r)),
    },
};
```

## डेवलपमेंट

### एनवायरनमेंट सेटअप

#### Nix

इस प्रोजेक्ट में सभी डिपेंडेंसीज़ और कम्पाइलर्स के साथ एक प्री-डिफ़ाइंड Nix डेवलपमेंट एनवायरनमेंट है। वही इंफ़्रास्ट्रक्चर CI/CD के लिए इस्तेमाल होता है, इसलिए यह डेवलप और टेस्ट करने का प्रिफ़र्ड तरीका है।

```bash
nix develop # Open development shell
nix flake check # Run CI tests
nix fmt # Format source files
```

#### बेयर मेटल

| डिपेंडेंसी | वर्शन | ज़रूरी | विवरण |
|------------|--------|--------|-------|
| Rust | 1.85+ | हाँ | Edition 2024 |
| LLVM | 21.x | हाँ | CPU कोड जनरेशन बैकएंड |
| Clang | - | हाँ | LLVM बिल्ड्स के लिए C कम्पाइलर |
| pkgconf | - | हाँ | बिल्ड कॉन्फ़िगरेशन टूल |
| protobuf | - | हाँ | ONNX proto कम्पाइलेशन |
| zlib | >=1.3 | हाँ | कम्प्रेशन लाइब्रेरी |
| libffi | >=3.4 | हाँ | फ़ॉरेन फ़ंक्शन इंटरफ़ेस |
| libxml2 | >=2.13 | हाँ | XML पार्सिंग |
| Z3 | >=4.15 | नहीं | ऑप्टिमाइज़ेशन वेरिफ़िकेशन के लिए SMT सॉल्वर |
| NVIDIA ड्राइवर | CUDA >=12.8 (R570) | नहीं | CUDA बैकएंड के लिए `libcuda.so.1`, रनटाइम पर लोड होता है; किसी toolkit की ज़रूरत नहीं |
| Clang NVPTX / AMDGPU टारगेट्स | - | नहीं | GPU कर्नेल कम्पाइलेशन (`clang --print-targets`) |

## टेस्ट

```bash
cargo test
cargo test --features z3,proptest  # With Z3 verification and PB generated tests
```
