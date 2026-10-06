---
sidebar_label: कर्नेल लाइब्रेरी
---

# कर्नेल library

USE चेहरा: `svod-tk` के साथ आने वाला हर कर्नेल, जिसे सादे tensors के साथ और tiles की किसी जानकारी के बिना
call किया जा सकता है। हर एक lazy `Tensor` (एक `Op::Call` node) लौटाता है जो model graph में compose होता है और
सामान्य `prepare()` path से realize होता है, और हर एक [IR में authoring](./lowering) वाले तीन-तरफ़ा contract
का पालन करता है:

| नतीजा | अर्थ |
|---|---|
| `Ok(Some(out))` | कर्नेल चला |
| `Ok(None)` | यह लागू नहीं होता: device कर्नेल के `ArchSet` से बाहर है, उसका LLVM backend मौजूद नहीं है, या shape tile नहीं होता — जान-बूझकर fallback करें |
| `Err(LaunchError)` | request ग़लत बना है (dtype, rank, कोई symbolic dim, कोई divisibility नियम) — caller का bug |

जब तक अलग से न कहा जाए, operands bf16 या f16 हैं; accumulation f32 में होता है।

---

## Targets

हर कर्नेल अपना `ArchSet` (`tk/src/target.rs`) ख़ुद declare करता है: AMD की एक explicit सूची, साथ में एक
खुला CUDA capability floor और एक Apple GPU-family floor।

| कर्नेल | gfx942 (CDNA3) | gfx1151 (RDNA3.5) | gfx1200 / gfx1201 (RDNA4) | CUDA sm_80+ | Metal Apple7+ |
|---|---|---|---|---|---|
| `flash_attention` / `_with` / `_tuned` | हाँ | हाँ | हाँ | हाँ | हाँ |
| `matmul` (square) | हाँ | हाँ | हाँ | हाँ | हाँ |
| `gemm_nt` / `_with` / `_with_epilogue` | — | हाँ | हाँ | हाँ | — |
| `rms_norm` / `add_rms_norm` | — | हाँ | हाँ | हाँ | — |
| `single_query_attention` / `_packed` | हाँ | हाँ | हाँ | हाँ | — |
| `knn` | हाँ | हाँ | हाँ | — | — |
| `kmeans_assign` | हाँ | हाँ | हाँ | — | — |

ये constants `tk/src/kernels/` में `FA_SUPPORTED_ARCHS`, `MATMUL_SUPPORTED_ARCHS`, `GEMM_NT_SUPPORTED_ARCHS`,
`NORM_SUPPORTED_ARCHS`, `SQ_ATTENTION_SUPPORTED_ARCHS`, `KNN_SUPPORTED_ARCHS` और `KMEANS_SUPPORTED_ARCHS`
हैं। कोई family किसी कर्नेल में validation से और अपनी tile table ख़ुद measure करके जुड़ती है — `gemm_nt` और
norms सिर्फ़ wave32 पर हैं क्योंकि किसी ने उनके लिए wave64 table measure नहीं की, इसलिए नहीं कि body वहाँ
चल नहीं सकता। `flash_attention_supported(&device)` सिर्फ़ arch gate का जवाब देता है, उन callers के लिए जो
launch से पहले sequence length को pad या bucket करते हैं।

---

## Flash attention

```rust
pub fn flash_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> LaunchResult<Option<Tensor>>
pub fn flash_attention_with(q, k, v, opts: FaOpts) -> LaunchResult<Option<Tensor>>
pub fn flash_attention_tuned(q, k, v, opts, policy: impl Fn(&DeviceSpec, GpuArch) -> FaPolicy + Copy) -> ..

pub struct FaOpts<'a> {
    pub causal: bool,                      // default true
    pub key_lens: Option<&'a Tensor>,      // [B] i32 valid-key counts: keys >= key_lens[b] are masked
    pub seg_start: Option<&'a Tensor>,     // [B, N] i32: query q of batch b sees no key before seg_start[b, q]
}
```

`q` `[B, N, H, D]` है, `k`/`v` `[B, N, H_kv, D]` हैं (GQA: `H % H_kv == 0`), output operand dtype में
`[B, N, H, D]` है। Sequence-major, head-major नहीं — model किसी projection को बिना transpose के सीधे इसमें
reshape कर देता है।

- `Ok(None)`: arch set से बाहर; `N` `q_blk · 8` का multiple नहीं (per-warp Q tile गुणा workgroup की आठ waves;
  `FLASH_ATTENTION_SEQUENCE_MULTIPLE` baseline का `128` है); KV length जो `N` से अलग हो (cross-attention
  implement नहीं है); ऐसा head dim जिसकी double-buffered K/V tiles device की shared memory से ज़्यादा हों।
- `Err`: dtype `{bf16, f16}` से बाहर, या `q` और `k`/`v` के बीच अलग; `D % 16 != 0`; `H % H_kv != 0`;
  `[B, N, H_kv, D]` के अलावा कोई `k`/`v` shape।

`key_lens` सिर्फ़ keys को mask करता है — padded query rows फिर भी compute होती हैं और caller उन्हें फेंक देता
है। `key_lens[b] == 0` को `1` पर clamp किया जाता है ताकि row finite रहे। `seg_start` कई sequences को एक row में
pack करता है: हर entry `0..=q` के भीतर होनी चाहिए और कम से कम एक visible key छोड़नी चाहिए।
[Flash Attention](./flash-attention) worked example है; per-warp tile पहले इस्तेमाल पर measure होती है
([Autotuning](./tuning))।

---

## GEMM

```rust
pub fn matmul(a: &Tensor, b: &Tensor) -> LaunchResult<Option<Tensor>>              // [n, n] · [n, n] → f32
pub fn gemm_nt(x: &Tensor, w: &Tensor) -> LaunchResult<Option<Tensor>>             // [lead..., K] · [N, K]ᵀ → [lead..., N]
pub fn gemm_nt_with(x, w, cfg: impl Fn(usize, usize, usize) -> Option<GemmCfg> + Copy) -> ..
pub fn gemm_nt_with_epilogue(x, w, epilogue: Epilogue<&Tensor>) -> LaunchResult<Option<Tensor>>

pub enum Epilogue<T> {
    Plain,               // y = x·wᵀ
    Add(T),              // y = x·wᵀ + residual, residual [lead..., N] in the operand dtype
    SwiGlu { pair: usize }, // y = silu(gate)·up off a fused [2I, K] gate/up weight; y is [lead..., N/2]
}
pub fn swiglu_pair_width(spec: &DeviceSpec) -> Option<usize>
```

`matmul` square reference कर्नेल है: कोई भी float dtype अंदर (bf16 में cast), f32 बाहर, हर arch पर। यह DSL
का performance canary है, production GEMM नहीं।

`gemm_nt` production linear layer है। `x` किसी भी rank ≥ 2 का `[lead..., K]` है (एक `[B, L, K]` activation
बिना reshape या copy के bind होता है), `w` `[N, K]` है जैसे weight store होता है, और `y` operand dtype में
`[lead..., N]` है — f32 accumulators registers में ही narrow किए जाते हैं, इसलिए memory से होकर f32 का कोई
round trip नहीं होता। `M = ∏lead` और `N` 64 के multiples होने चाहिए और `K` 32-wide strip का multiple, कम से
कम दो strips के साथ; वरना `Ok(None)`, और caller 128 तक pad करता है या `Tensor::linear` इस्तेमाल करता है।

Epilogues ही इस कर्नेल के होने की वजह हैं: GEMM के बाद graph जो pass चुकाता, उसे ये इसके store में fold कर
देते हैं। `Add` residual को store के अपने offset पर पढ़ता है और output dtype में जोड़ता है, ठीक वैसे जैसे
graph का `try_add` round करता है। `SwiGlu` को fused weight की rows `pair` rows के बारी-बारी gate/up blocks में
चाहिए — `swiglu_pair_width(&device)` device की tile table का `reg_n / 2` है, या `None` जब tiles आपस में
असहमत हों (तब caller अलग SwiGLU pass रखता है)। Model weight को उस क्रम में एक बार load करता है, क्योंकि
launch के समय `M` tile चुनता है और हर candidate tile को वही arrangement पढ़ना होता है।

Tile `GemmPolicy` (`tk/src/kernels/gemm.rs`) से आती है: एक per-family table — `CUDA_TILES`, `RDNA_TILES`,
`RDNA4_TILES` — जो हर shape और epilogue के लिए पहले इस्तेमाल पर measure होती है ([Autotuning](./tuning)), या
tuning बंद होने पर static `GemmPolicy::cfg` वाला चुनाव। `gemm_nt_with` chooser caller से लेता है (benches इसी
तरह उसे sweep करते हैं)।

---

## RMS norm

```rust
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> LaunchResult<Option<Tensor>>
pub fn add_rms_norm(x, residual: &Tensor, weight, eps) -> LaunchResult<Option<(Tensor, Tensor)>>   // (h, y)
pub fn select_norm_cfg(rows: usize, d: usize, lanes: usize) -> Option<NormCfg>
```

`x` `[rows..., D]` है, `weight` `[D]`, दोनों एक ही 16-bit dtype में। हर row के लिए एक wave, row registers
में रहती है, sum of squares एक butterfly shuffle से पूरा होता है — कोई LDS नहीं, कोई barrier नहीं, कोई
`RANGE` नहीं। Numerics हर op पर graph का अनुसरण करते हैं:
`y = dtype((f32(x) · rsqrt(Σx²/D + eps)) · f32(w))`, अंत में एक rounding; सिर्फ़ summation का क्रम अलग है।
`Ok(None)` जब `D` wave का multiple न हो या प्रति lane 64 elements से ज़्यादा हो (wave32 पर `2048`)।

`add_rms_norm` `(h, y)` लौटाता है, जहाँ `h = x + residual` वैसे ही round होता है जैसे graph का add, और
`y = rms_norm(h)`, ताकि एक pre-norm decoder layer अपनी residual stream एक ही बार लिखे और पढ़े। जब पिछली
projection residual को `Epilogue::Add` से ले चुकी हो, तब two-pass `rms_norm` काफ़ी है;
`model/src/qwen3/decoder_layer.rs` हर layer के लिए इनमें से चुनता है।

---

## Single-query attention

```rust
pub fn single_query_attention(q, k, v, opts: SqAttentionOpts<'_>) -> LaunchResult<Option<Tensor>>
pub fn single_query_attention_packed(q, k, v, head_offset: usize, opts) -> LaunchResult<Option<Tensor>>

pub struct SqAttentionOpts<'a> {
    pub key_lens: Option<&'a Tensor>,                 // [B] i32, entries in 0..=N
    pub include_last: bool,                           // also score key N-1 (Whisper's self-cache slot)
    pub appended: Option<(&'a Tensor, &'a Tensor)>,   // the step's own [B, 1, H, D] K/V, scored after the prefix
    pub split: Option<usize>,                         // K/V chunks; None = the device's SqPolicy, tuned on first use
    pub cache_map: Option<&'a Tensor>,                // [B] i32: which K/V row each query row reads
}
```

Decode-step कर्नेल: `q` f32 में `[B, 1, H, D]` है, `k`/`v` f32, f16 या bf16 में `[B, N, H_total, D]` हैं
(या हर row को एक cache से serve करने के लिए `[1, N, H_total, D]`), output f32 में `[B, 1, H, D]`।
एक wave एक `(batch, head)` की मालिक होती है; `Q` registers में रहता है जबकि K/V `N` पर stream होते हैं; dot
products XOR-shuffle all-reduces हैं और softmax एक one-pass online update है। कोई LDS नहीं और कोई matrix core
नहीं, इसीलिए इसका `ArchSet` सबसे चौड़ी AMD सूची और साथ में CUDA है।

`_packed` एक packed cache के heads `head_offset..head_offset + H` को उसे slice किए बिना चुनता है। लंबा
unmasked attention K/V को contiguous chunks में बाँटता है, हर chunk एक wave, और दूसरे pass में उनकी softmax
states को merge करता है; `SqPolicy` split का आकार device के resident-wave budget से तय करता है और पहले
इस्तेमाल पर सबसे नज़दीकी divisors को measure करता है।

---

## k-NN और k-means

```rust
pub fn knn(x: &Tensor, c: &Tensor, k: usize) -> LaunchResult<Option<(Tensor, Tensor)>>           // (dists [N, k] f32, idxs [N, k] i32)
pub fn kmeans_assign(x: &Tensor, c: &Tensor) -> LaunchResult<Option<(Tensor, Tensor)>>         // (cluster_ids [N] i32, best_dist [N] f32)
pub fn kmeans_update(x, cluster_ids, old_centroids) -> LaunchResult<(Tensor, Tensor)>           // (new_centroids [K, D], shift [K])
```

दोनों corpus (centroids) को matrix core से stream करते हैं और x²-free score `‖c‖² − 2⟨x, c⟩` से एक चालू
top-K (argmin) रखते हैं, इसलिए `[N, M]` distance matrix कभी बनता ही नहीं; host side bf16 में cast करता है,
`D` (और `N`) को WMMA edge तक pad करता है, और exact f32 distances के लिए `‖x‖²` फिर से जोड़ता है। `knn` `k`
को `1..=16` में लेता है। `kmeans_update` एक शुद्ध graph op है — हर cluster पर `scatter_reduce`, empty-cluster
fixup, हर cluster का shift — क्योंकि sort/scatter pattern tile नहीं होता; Lloyd loop caller का है। सिर्फ़ AMD।

---

## Model में कर्नेल का इस्तेमाल

Policy — कौन-सा कर्नेल, कौन-सा fallback — model की होती है, कर्नेल की कभी नहीं। `model/src/qwen3/` में
Qwen3 decoder reference integration है:

```rust
// model/src/qwen3/linear.rs
pub(crate) fn linear(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    if fusable(x, w)
        && let Some(y) = svod_tk::gemm_nt(x, w).context(TkSnafu)?
    {
        return Ok(y);
    }
    Ok(x.contiguous().linear().weight(w).call()?)
}
```

```rust
// model/src/qwen3/attention.rs
if matches!(q.dtype().base(), ScalarDType::Float16 | ScalarDType::BFloat16)
    && let Some(out) =
        svod_tk::flash_attention_with(q, k, v, svod_tk::FaOpts { causal: true, key_lens: None, seg_start })
            .context(TkSnafu)?
{
    return Ok(out);
}
// else: permute to head-major and run scaled_dot_product_attention
```

नक़ल करने लायक़ तीन आदतें:

- **पहले `fusable` पर gate करें।** `Err` का मतलब है ग़लत बनी request, और caller की तरफ़ एक 16-bit जाँच एक
  वैध f32 path को bug के रूप में report होने से बचाती है।
- **Error को bridge करें।** `LaunchError` model के error enum में box होता है
  (`#[snafu(source(from(svod_tk::LaunchError, Box::new)))]`), इसलिए एक failed build कर्नेल के context के
  साथ एक model error होता है।
- **कर्नेल के लिए shape upstream में बनाएँ।** `embed.rs` padded sequence lengths को
  `FLASH_ATTENTION_SEQUENCE_MULTIPLE` तक bucket करता है, और `feed_forward.rs` load के समय gate/up weight को
  `swiglu_pair_width` से interleave करता है, ताकि कर्नेल decline करने के बजाय लागू हों।

जो fusion किसी model का memory layout जानता है, वह model के बगल में रहता है: `model/src/qwen3/tk/mod.rs` एक
QKV-norm-RoPE prologue है, जो norm कर्नेल की row vocabulary पर लिखा गया है, `NORM_SUPPORTED_ARCHS` से gated
है, और तीन outputs के साथ `graph_launch_multi` से launch होता है। वही `launch_custom` policy लागू होती है —
यह हर लिहाज़ से एक tk कर्नेल है, सिवाय इसके कि यह कहाँ रहता है।
