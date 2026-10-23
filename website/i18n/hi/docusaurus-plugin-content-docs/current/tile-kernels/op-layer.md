---
sidebar_label: Op Layer
---

# Op Layer {#op-layer}

`svod_tk::ops` वह है जिसे मॉडल बुलाते हैं। हर op एक lazy `Tensor` लौटाता है। जब device, dtypes और
shapes किसी कर्नेल में फ़िट होते हैं, तो tk कर्नेल गणना करता है। वरना op बराबर का ग्राफ़ ख़ुद बनाता
है। मॉडल कभी नहीं जाँचता कि कर्नेल लागू होता है या नहीं, और कभी tile तक pad नहीं करता।

```rust
pub fn linear(x: &Tensor, w: &Tensor, opts: Linear) -> Result<Tensor>;
pub fn conv2d(x: &Tensor, w: &Tensor, opts: Conv) -> Result<Tensor>;
pub fn attention(q: &Tensor, k: &Tensor, v: &Tensor, opts: Attn) -> Result<Tensor>;
pub fn heads(qkv: &Tensor, opts: Qkv) -> Result<(Tensor, Tensor, Tensor)>;
pub fn layer_norm(x: &Tensor, w: &Tensor, b: Option<&Tensor>, eps: f64) -> Result<Tensor>;
pub fn add_layer_norm(x: &Tensor, residual: &Tensor, w: &Tensor, b: Option<&Tensor>, eps: f64)
    -> Result<(Tensor, Tensor)>;
pub fn rms_norm(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor>;
pub fn add_rms_norm(x: &Tensor, residual: &Tensor, w: &Tensor, eps: f64) -> Result<(Tensor, Tensor)>;
pub fn supported(device: &DeviceSpec) -> bool;
```

| Op | Shapes | Options |
|---|---|---|
| `attention` | `q [B, T, H, D]`, `k`/`v [B, Tk, H_kv, D]` → `[B, T, H, D]` | `Attn { causal, keys, window, chunk, seg_start, cache, splits, scale, bias }` |
| `linear` | `x [lead..., K]`, `w [N, K]` → `[lead..., N]` | `Linear { bias, act, gated, residual, scale }`; gated `w` `[2N, K]` है |
| `conv2d` | `x [B, H, W, Cin]`, `w [Cout, kh, kw, Cin / groups]` → `[B, Ho, Wo, Cout]` | `Conv { stride, pad, dilation, groups, bias, act, residual, scale, out_dtype }`; 1×1 `linear` है |
| `heads` | `qkv [B, T, (H + 2·H_kv)·D]` → `q`, `k`, `v` | `Qkv { heads, kv_heads, head_dim, q_norm, k_norm, eps, rope }` |
| `layer_norm`, `rms_norm` | `x [..., D]`, `w`/`b [D]` | `add_*` `x` जैसा `residual` लेते हैं और `(x + residual, norm)` लौटाते हैं |

`Linear::scale` residual जोड़ने से पहले activated मान को गुणा करता है:
`scale·act(x·wᵀ + bias) + residual`, इसलिए Conformer का half-step `x + 0.5·ffn(x)` एक ही GEMM है।

`Attn::keys` `KeyMask::None`, `KeyMask::Lens(&lens)` (`[B]` वैध key गिनतियाँ) या
`KeyMask::Bool(&mask)` (`[B, Tk]`, जहाँ attend हो वहाँ true) है। `Attn::cache`
`Cache { head_start, kv_heads, row_map, appended, app_lens }` लेता है, और `appended` के लिए
`KeyMask::Lens` ज़रूरी है। `Attn::chunk` block attention है (`Chunk { size, left, right }`)।
`Attn::scale` का default `1/√D` है। `Attn::bias` (`[B, H, T, Tk]` या `[1, H, T, Tk]`, stream dtype
में) masks से पहले scaled scores में जोड़ा जाता है, जैसे WavLM का relative position bias।
`Qkv::rope` `[1, T, 1, D/2]` (position के हिसाब से) या `[B, T, 1, D/2]` (token के हिसाब से) का
`(cos, sin)` है। Masks का वर्णन [attention कर्नेल](./kernel-library#flash-attention) के साथ है।

## कर्नेल या ग्राफ़ {#kernel-or-graph}

फ़ैसला `ops::shape` में एक pure function है, `fn(planner, dtypes, extents, …) -> Plan<Cfg>`, इसलिए
इसे बिना GPU के host पर टेस्ट किया जाता है (`test/unit/ops_plan.rs`)। `Plan::Kernel(configs)` में
ranked configs होते हैं; [tuner](./tuning) उनमें से उन पहले configs को मापता है जो बिना spill के
compile होते हैं (या, अगर हर एक spill करे, तो जो पहले compile हों: spills किसी op को कभी ग्राफ़ पर नहीं
भेजते)। `Plan::Graph(Fallback)` बताता है कि ग्राफ़ क्यों चलता है, और planner log उसका नाम लेता है
(`RUST_LOG=svod_tk::plan=debug`)।

`ops::chosen()` उस op का फ़ैसला एक value के रूप में लौटाता है जो इस thread ने आख़िरी चलाया:
`Chosen::Kernel { op, family, cfg }` — kernel का नाम (`family`) और उसकी pick का Debug text
(`cfg`, जो एक nearest pick का path बताता है: score matrix, queue, splits) — या
`Chosen::Graph { op, why }`। `k_nearest_reranked` का record उसकी search का है: जीतने वाली
rows का dtype उसके config के साथ।

| `Fallback` | कब |
|---|---|
| `Target` | Tensor default device पर नहीं है, device के पास कोई tk kernels नहीं हैं (`config::has_kernels`: sm_80 से नीचे का CUDA, Metal, CPU), या उसका target कोई SM count नहीं बताता |
| `Dtype` | कोई operand f32 है, या सभी operands matrix core वाला एक ही 16-bit type साझा नहीं करते (`nearest` और `k_nearest` i8 rows भी लेते हैं, हर एक 1 से scaled)। `conv2d`: input के type या f32 के अलावा कोई output type, या ऐसा residual जो उस type में न हो |
| `Symbolic` | Bound leading dim के अलावा कोई dim symbolic है (`linear` के लिए symbolic `N` भी) |
| `Shape` | `linear`: `N` 8 का गुणज नहीं या कोई rows नहीं। `conv2d`: `groups > 1`, `Cin` 16 का गुणज नहीं, `Cout` 8 का गुणज नहीं, या ख़ाली output। `attention`: कोई config `D` पर फ़िट नहीं होता (sm_86 पर 32, 48, 64, 128 और 256 plan होते हैं और 40 नहीं) या कोई ख़ाली dim। `heads`: `D` 16..=256 में दो की घात नहीं। Norms: `D` 256..=2048 में दो की घात नहीं |
| `Config` | कोई tile config फ़िट नहीं होता। `linear` के लिए, `K` 16 का गुणज नहीं है |

जब appended rows हर config के key block से चौड़ी हों, तब भी `attention` ग्राफ़ लेता है।

:::note[f32 ग्राफ़ पर क्यों रहता है]
कर्नेल 16-bit operands लेते हैं (और nearest-neighbour search के लिए 8-bit integers)। किसी f32 मॉडल
को नीचे cast करने से गति के बदले precision के लगभग तीन दशमलव अंक चले जाते, इसलिए op layer यह फ़ैसला
मॉडल के dtype पर छोड़ती है। `k_nearest_reranked` वह अपवाद है जो f32 numbers रखता है: यह rows को
8 bits या f16 में search करता है (`Quantized`, `Rerank`), हर row के `k + margin` सबसे छोटे रखता है
(default 16), और उन्हें दी गई rows से f32 में फिर से score करता है। जिस neighbour को search
`k + margin` के बाद rank करता है, वह छूट जाता है।
:::

## Batch variables {#batch-variables}

किसी operand का dim 0 symbolic हो सकता है जब वह किसी runtime variable से bound हो। वह variable या तो
JIT का `batch_var` है या min और max वाला `DefineVar`/bounded `Param`। तब कर्नेल grid z में live गिनती
पर launch होता है, और उसके buffers अधिकतम आकार के होते हैं। Outputs capacity पर allocate होते हैं और
symbolic dim उनके shape में बना रहता है (`Tensor::empty_dynamic_spec`)। इसलिए अगला कर्नेल realized
buffer को ही bind करता है, और उसे छोटा करने के लिए कोई copy नहीं बनती। कोई भी दूसरा symbolic dim
`Fallback::Symbolic` है।

## Errors {#errors}

`Err` उसी के लिए आरक्षित है जिसे graph op भी ठुकराता, साथ में कर्नेल build की विफलता:

| `ops::Error` | मतलब |
|---|---|
| `Shape { op, operand, got, expected }` | किसी operand का shape op में फ़िट नहीं होता |
| `Dtype { op, operand, got, want }` | `w` (या `k`, `v`, norm weights, rope tables) का dtype input से अलग है |
| `Heads { op, heads, kv_heads }` | `heads` `kv_heads` का गुणज नहीं है |
| `Graph { op, source }` | Graph fallback बनाना विफल रहा |
| `Launch { op, source }` | कर्नेल को lower या bind करना विफल रहा |

## एक मॉडल में {#in-a-model}

Nemotron-3-Diarization का self-attention, `model/src/nemotron_diar/model.rs` से: एक stacked QKV
projection, RoPE के साथ fused prologue, key lengths के तहत attention, और epilogue में residual के साथ
output projection।

```rust
fn forward(&self, x: &Tensor, rope: &(Tensor, Tensor), key_lens: &Tensor, residual: &Tensor) -> Result<Tensor> {
    let (b, s, d) = (x.dim(0)?, x.dim(1)?, x.dim_const(2)?);
    let qkv = ops::linear(x, &self.qkv_weight, ops::Linear::default())?;
    let (cos, sin) = rope;
    let split = Qkv {
        heads: self.num_heads,
        kv_heads: self.num_heads,
        head_dim: d / self.num_heads,
        q_norm: None,
        k_norm: None,
        eps: 0.0,
        rope: Some((cos, sin)),
    };
    let (q, k, v) = ops::heads(&qkv, split)?;
    let opts = Attn { keys: KeyMask::Lens(key_lens), ..Attn::default() };
    let out = ops::attention(&q, &k, &v, opts)?;
    project(&self.o_proj, &out.try_reshape([b, s, SInt::Const(d)])?, Act::None, Some(residual))
}
```

`project` layer के bias, एक activation और एक वैकल्पिक residual के साथ `ops::linear` है। वही फ़ाइल
अपने LayerNorms `ops::layer_norm` से चलाती है।
