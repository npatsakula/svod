---
sidebar_label: कर्नेल लाइब्रेरी
---

# कर्नेल लाइब्रेरी {#kernel-library}

`svod_tk::kernels` का हर कर्नेल spec से `Program` तक का एक function है, जो 16-bit element type
(`BF16` या `F16`) पर generic है; `gemm` और `nearest` 8-bit integers (`I8`) भी लेते हैं, जो i32 में
exact रूप से accumulate होते हैं: `gemm` f32 store करता है, और `nearest` हर row के products को rows
के `x_scale` और `c_scale` से scale करता है।

जहाँ कर्नेल का config planner चुनता है (`gemm`, `conv`, `attention`, `nearest`), वहाँ spec
(`GemmSpec`, `ConvSpec`, `AttnSpec`, `NearestSpec`) shapes और options का एक सादा विवरण है, और कर्नेल
उसे अपने config के तहत admitted रूप में लेता है: `fn k<T>(spec: &Admitted<Spec>) -> Program`।
`spec.admitted(&planner, cfg)` तब एक लौटाता है जब planner ठीक उसी spec के लिए `cfg` admit करे (config
target पर spec की बताई problem के लिए feasible, उसकी shape पर claimed), वरना मना करने का कारण;
`unsafe` `Admitted::assume(spec, cfg)` बिना admission के config लेता है और कुछ नहीं जाँचता। Admitted
spec केवल पढ़ा जा सकता है: वह spec में deref होता है, और `cfg()` config लौटाता है। बाक़ी कर्नेल
(`heads`, `norm`, `combine`, `split_merge` और nearest search के select, merge और rescore) अपना spec
जैसा है वैसा लेते हैं, config उसके भीतर।

`GemmSpec`, `ConvSpec`, `AttnSpec`, `HeadsSpec` और `NormSpec` में एक `batch: Batch` होता है:
`Batch::Static(n)` `n` batches launch करता है, और `Batch::var(name, min, max)` (एक `ir::Var`) नाम से
bind हुए runtime variable की live गिनती launch करता है, buffers `max` के आकार के होते हैं। `GemmCfg`
और `FaCfg` में एक `lowering(&device)` है (एक `atoms::Device`: target और उसकी costs); बाक़ी कर्नेल
`Lowering::new(&device)` से lower होते हैं। मॉडल इन कर्नेलों तक [op layer](./op-layer) के ज़रिए
पहुँचते हैं, जो config चुनती है।

| कर्नेल | Spec → program | कर्नेल का नाम | Op |
|---|---|---|---|
| GEMM + epilogue | `GemmSpec` → `gemm::gemm` (tail split होने पर `TailSpec` → `gemm::gemm_tail`) | `gemm` (+ `split_merge`) | `ops::linear` |
| Implicit-GEMM convolution | `ConvSpec` → `conv::conv` (+ split होने पर `gemm::split_merge`) | `conv`, `split_merge` | `ops::conv2d` |
| Flash attention forward | `AttnSpec` → `attention::attention` | `flash_attention` | `ops::attention` |
| Split merge | `CombineSpec` → `attention::combine` | `combine_splits` | splits के साथ `ops::attention` |
| Attention prologue | `HeadsSpec` → `heads::heads` | `heads` | `ops::heads` |
| LayerNorm / RMSNorm | `NormSpec` → `rows::norm` | `layer_norm` / `rms_norm` | `ops::{layer_norm, rms_norm, add_*}` |
| Nearest-neighbour search | `NearestSpec` → `nearest::nearest` | `nearest` (+ `nearest_select`, `nearest_merge`, `nearest_rescore`) | `ops::{nearest, k_nearest, k_nearest_reranked}` |

## GEMM {#gemm}

`c = scale·act(a·bᵀ + bias) + residual`, जहाँ `a [batch·m, k]` और `b [n, k]`। Matrix product के
बाद का सब कुछ f32 accumulator पर चलता है, और नतीजा store पर एक बार round होता है।

| `Epilogue` field | असर |
|---|---|
| `bias` | accumulator में जोड़ी गई `[n]` row (gated होने पर `[2n]`) |
| `act` | `Act::None`, `Act::Gelu` (erf approximation, error अधिकतम 1.5e-7), `Act::Silu` |
| `gated` | `b` `[2n, k]` है, gate rows ऊपर, up rows नीचे; output `act(gate)·up` (SwiGLU, GeGLU) |
| `scale` | residual से पहले activated value को गुणा करता है |
| `residual` | `[batch·m, n]`, सबसे आख़िर में जोड़ा जाता है |
| `out_f32` | f32 output और residual; accumulator बिना round किए store होता है |
| `rope` | `Rope { heads, head_dim, seq }`: `head_dim` columns वाले पहले `heads` heads `[seq, head_dim/2]` cos और sin tables से घुमाए जाते हैं (output row `m` position `m % seq` पर), उनके बाद के columns जैसे हैं वैसे store होते हैं; केवल bias के साथ, उस tile पर जिसके warps पूरे heads रखते हैं (`GemmCfg::turns`) |

`GemmCfg { tile: [bm, bn, bk], warps: [rows, cols], core, schedule, order, tail }`। `m` से आगे की
rows और `n` से आगे के columns bounded views हैं, इसलिए `m` और `n` मुक्त हैं। `k` का `bk` का गुणज
होना ज़रूरी है। `core: Core::Group` products को `wgmma` (sm_90) पर issue करता है;
`schedule: Schedule::ping_pong(k)` दोनों warp rows को एक slot पर ping-pong में चलाता है (CDNA)।
Blocks output tiles पर `order` tile rows (`gemm::TileOrder`, default 8, `TileOrder::DEFAULT`; एक
row यानी row-major, grid की सभी rows यानी column-major) के समूहों में चलते हैं, ताकि एक साथ resident
blocks वही B tiles पढ़ें। Planner हर config को default पर plan करता है; कोई config दूसरी group height
भी रख सकता है (उसके code का `g` field)।

Rotation हर element के partner half को, जो आधा head आगे है, column `flip` के ज़रिए पढ़ता है। जब
किसी warp के columns पूरे heads रखते हैं, तो एक lane दोनों halves रखती है, और `flip` एक register
renaming है। Tables एक periodic view से पढ़ी जाती हैं, इसलिए row का हर head वही `[head_dim/2]` row
पढ़ता है। Ranking में rotation मुफ़्त नहीं है: उसकी operations और tables के reads epilogue की
operations के साथ claim होते हैं, और जो tile पूरे heads नहीं रखता वह refuse होता है।

`tail` (`gemm::TailSplit`, अधिकतम `config::TAIL_SPLITS` हिस्से) अधूरी आख़िरी wave की tiles को K के
साथ split करता है: पूरी waves data-parallel चलती हैं, tail f32 partials लिखती है और `split_merge`
epilogue लगाता है। Planner हर split की कीमत पहले के बाद हर launch पर एक launch cost
(`config::LAUNCH`) के साथ लगाता है, और उस split को refuse करता है जिसके partials device memory के
आठवें हिस्से से बड़े हों (`ShapeRejected::Partials`); tuner आगे रहने वाले split को उसकी unsplit
pipeline के साथ time करता है। Batched, gated, rotating और convolution GEMMs कोई split नहीं लेते।

कोई tile table नहीं है। `config::Planner::gemm_ranked` tiles, `bk`, warp grids, ring depths और
unrolling गिनता है, वही रखता है जिसे lowering target पर चलाती है (atoms warp grid को tile करते हैं,
fills हर thread पर पूरे 16-byte chunks में बँटते हैं, ring shared memory में समाती है) और बाक़ी को
device की मापी गई `Costs` से SM time के nanoseconds में rank करता है। Ranking का हर device input मापा
गया है; उसके model constants (`BAND`, `OVERLAP_WAVES`, `LAUNCH` और हर family के operation weights,
`ops::config` में हर एक `Model:` चिह्नित) हर device के लिए सही रहने के लिए हैं, और fill bytes मापी
गई cache ladder से priced होते हैं। GEMM का output उसके type के bytes पर, residual और bias के reads
के साथ priced होता है। Ranking क्रम देती है, और tuner उन पहली 56 tiles (`config::SHORTLIST`) को
मापता है जो बिना spill के compile होती हैं ([Tuning](./tuning))।

## Convolution {#convolution}

`conv::conv` channels-last `x [b, h, w, cin]` और `w [cout, kh, kw, cin]` पर एक implicit GEMM है:
A image का row-mapped view है (`conv::im2col_view`), इसलिए padding rows शून्य पढ़ी जाती हैं और कुछ भी
materialize नहीं होता। `K = kh·kw·cin` एक pipeline के रूप में चलता है, हर step एक tap के भीतर (`bk`
`cin` को भाग देता है)। एक ही static batch के भूखे grid पर `ConvCfg::split` (`ConvSpec` का `split`)
K को grid z पर बाँटता है और `gemm::split_merge` epilogue लागू करता है। `ops::conv2d` 1×1 को
`linear` के रूप में लेता है, और `groups ≠ 1`, 16 पर न बैठने वाले `cin` और 8 पर न बैठने वाले `cout`
को ग्राफ़ पर रखता है।

## Flash attention {#flash-attention}

Sequence-major `q [batch, t, heads, d]` और `k`, `v [batch, tk, kv_heads, d]` पर
`o = softmax(q·kᵀ·scale + bias)·v`। Q registers में रहता है और K/V एक shared ring से stream होते
हैं। Online-softmax state `(m, l, o)` `exp2` के सहारे f32 में carry होती है।

| Feature | कैसे |
|---|---|
| GQA | Query head `h` KV head `h / (heads / kv_heads)` पढ़ता है |
| Cross attention | `tk ≠ t` views का गुण है |
| कोई भी `t`, `tk` | Padding की जगह bounds: query tile और आख़िरी key block bounded views हैं, और `tk` से आगे की keys masked हैं |
| Head dims | कोई भी `d` जिसे दोनों products tile करें और जिसकी fills हर thread पर पूरे chunks में बँटें (`FaCfg::fits`); मॉडल 48, 64 और 128 इस्तेमाल करते हैं |
| `AttnMask::causal` | Diagonal से आगे के key blocks trip count के ज़रिए छोड़ दिए जाते हैं |
| `AttnMask::window` | हर query के आसपास `(left, right)`; बाहर के blocks छोड़ दिए जाते हैं |
| `AttnMask::chunk` | Block attention: query अपना chunk देखती है, उससे पहले `left` chunks (bound न हो तो सभी) और बाद में `right` |
| `AttnMask::key_lens` | `[batch]` i32 वैध key counts; length से आगे के blocks छोड़ दिए जाते हैं |
| `AttnMask::key_mask` | `[batch, tk]` i32 दिखने वाली keys (row stride 8 तक ऊपर round), हर block में पढ़ी जाती है |
| `AttnMask::seg_start` | packed rows के `[batch, t]` i32 segment starts, `t` के साथ non-decreasing |
| `AttnMask::bias` | stream type में `[batch or 1, heads, t, tk]` additive bias (row stride 8 तक ऊपर round), masks से पहले scaled scores में जोड़ा जाता है |

सिर्फ़ वे key blocks predicate की गणना करते हैं जिन्हें कोई causal, window, chunk, length या segment
किनारा काटता है; पूरी तरह भीतर के blocks बिना mask के चलते हैं। Key mask और bias हर block में पढ़े जाते
हैं। जो query कोई key नहीं देखती वह शून्य देती है, जैसा ग्राफ़ लिखता है।

**Cache mode** (`AttnSpec::cache = Some(CacheLayout { .. })`) K और V को एक cache
`[rows, tk, heads_total, d]` से पढ़ता है जिसमें कई layers के heads होते हैं:

| `CacheLayout` field | असर |
|---|---|
| `rows`, `heads_total` | Cache का shape |
| `row_map` | एक `[batch]` i32 parameter: हर batch lane कौन सी cache row पढ़ती है |
| `appended` | `[batch, appended, kv_heads, d]` keys और values की rows जो cached prefix के बाद score होती हैं (वे tokens जिन्हें decoder step ने अभी project किया), अधिकतम एक key block; 0 पर कुछ नहीं जुड़ता |
| `app_lens` | दिखने वाली appended rows का एक `[batch]` i32 parameter |

इस attention के heads में से पहला एक `[1]` i32 `head_start` parameter है, इसलिए हर layer एक ही
program चलाती है।

**Packed mode** (`AttnSpec::packed`) `q`, `k` और `v` को एक `[batch, t, heads + 2·kv_heads, d]`
parameter से, जो एक fused projection है, head offsets 0, `heads` और `heads + kv_heads` पर पढ़ता है,
बिना cache के। Kernel `flash_attention_packed` है, जिसे अलग-अलग operands वाले से अलग tune किया जाता
है; एक ही config पर दोनों वही bits निकालते हैं।

**Key splits** (`FaCfg::splits > 1`) कुछ लंबी rows, जैसे decoder step, को ज़्यादा blocks देते हैं।
हर split f32 partials `o_part`, `m_part` और `l_part` लिखता है, और `combine` उन्हें मिलाता है।
`Attn::splits = Some(n)` गिनती तय करता है; `None` होने पर `Planner::splits` 1, 2, हर SM पर दो blocks
के आसपास की गिनतियाँ और हर key block पर एक split प्रस्तावित करता है, और tuner हर split candidate को
उसके merge के साथ मापता है।

`FaCfg { bq, bkv, schedule, splits, forms, key_warps, warp_rows, core }`:

| Field | असर |
|---|---|
| `warp_rows` | हर warp की query rows: 16 (एक atom row) या 32 (हर K और V fragment दो atom rows को खिलाता है); 32-row रूप एक variant है जिसे tuner उसके 16-row tile के साथ मापता है |
| `key_warps` | एक-row-warp block (`bq = 16`, `t ≤ 16` के लिए पेश) अपने key blocks को 1, 2 या 4 warps पर बाँटता है, हर एक के अपने ring slots और softmax state के साथ, जो shared memory के ज़रिए मिलाए जाते हैं |
| `core` | `Core::Wide` दोनों products CDNA के 32×32×8 MFMA पर चलाता है; `Core::Group` `wgmma` (sm_90) पर |
| `forms` | नीचे के softmax forms |

Candidates GEMM वाली उसी lattice और cost model से आते हैं (`Planner::attention_ranked`)। जहाँ fill
registers से होकर जाता है, वहाँ value tile column-major store होता है ताकि उसके fragments 16-byte
loads के रूप में gather हों; RDNA4 पर fill ख़ुद एक transposing load (`global_load_tr_b128`) है।

Softmax नया running maximum तभी अपनाता है जब वह log2 units में 8 से ज़्यादा बढ़े (FlashAttention-4
का lazy rescale)। पाँच `Forms` tuner knobs हैं, plan में सभी बंद; tuner हर एक को वहाँ चालू करता है
जहाँ वह तेज़ मापता है (एक forms pass, फिर उन forms के तहत tiles, फिर विजेता पर forms दोबारा):

| `Forms` field | असर |
|---|---|
| `truncate_p` | bf16 probabilities round होने के बजाय truncate होती हैं, और row sum वही जोड़ता है जो रखा है |
| `sum_after` | Row sum rescale से पहले के बजाय P·V product के बाद आता है |
| `skip_rescale` | Output का rescale एक `Scale` है, जिसे हर वह wave छोड़ देती है जिसकी सभी rows ने अपना maximum बनाए रखा |
| `raw_max` | Running maximum raw scores की units में रहता है: हर score के बजाय हर row पर एक गुणा |
| `bias_direct` | हर key block का bias tile ring slot के बजाय सीधे global memory से scores के layout में पढ़ा जाता है |

## Attention prologue {#attention-prologue}

`heads` एक fused projection `qkv [batch, t, (heads + 2·kv_heads)·d]` को sequence-major
`q [batch, t, heads, d]` और `k`, `v [batch, t, kv_heads, d]` में बाँटता है। वैकल्पिक रूप से यह `q`
और `k` को head पर `[d]` weights के साथ RMS-normalize करता है, फिर उन पर rotary embedding लागू करता
है। Rotation head के दो हिस्सों को जोड़ी बनाता है, `[t, d/2]` की `(cos, sin)` tables इस्तेमाल करते
हुए, जो साझा या हर batch की अलग होती हैं (`Rope { per_batch }`)। एक block एक head slot की `br` rows
संभालता है। `d` का 16..=256 में दो की घात होना ज़रूरी है, और candidates `br ∈ {4, 8, 16}` हैं।

## Norms {#norms}

`rows::norm` हर row पर एक warp और f32 में एक fused reduce-then-map इस्तेमाल करता है:

| `Norm` | सूत्र |
|---|---|
| `Layer` | `(x − mean)·rsqrt(var + eps)·w + b` |
| `Rms` | `x·rsqrt(mean(x²) + eps)·w` |

`residual: true` के साथ यह element type में round किए गए `x + residual` को normalize करता है और वह
sum भी लिखता है, जो transformer layer की pre-norm residual stream है। `d` का 256..=2048 में 256 का
गुणज होना ज़रूरी है, और candidates हर block में `br ∈ {4, 8, 16}` rows हैं।

## Nearest-neighbour search {#nearest-neighbour-search}

`nearest::nearest` `x [n, d]` की rows को candidates `c [m, d]` के विरुद्ध एक `Metric` (`SquaredL2`,
`ShiftedL2`, `L2`, `NegDot`) से score करता है और हर row के `k` सबसे छोटे scores, बढ़ते क्रम में,
उनके indices के साथ रखता है। `NearestSpec { n, m, d, dtype, metric, floors, groups, k, live }`
अपना element type बताता है (`dtype`: एक 16-bit float या 8-bit integers), और कर्नेल केवल उसी type पर
चलता है। `NearestCfg::list` (`nearest::Way`) बताता है कि row के `k` सबसे छोटे scores कहाँ रखे जाते
हैं:

| `Way` | List |
|---|---|
| `Registers` | सभी `k` एक register list में |
| `Queued(Pow2)` | उतने scores की एक register queue (2 से, `k` से कम), जो outputs में रखी बाक़ी list में merge होती है |
| `Matrix(SelectCfg)` | Scores एक matrix के रूप में store होते हैं, जिसकी rows `nearest_select` list करता है |

`Pow2` दो की एक घात रखता है, इसलिए queue, और list की lane queues (`LaneQueue`) की width और depth,
बनावट से ही दो की घातें हैं। Split होने पर (`NearestCfg::splits > 1`) एक query block के blocks हर एक
candidates का एक हिस्सा fold करते हैं और `nearest_merge` हिस्सों की lists को merge करता है।
`k_nearest_reranked` 8-bit या 16-bit rows पर search करता है और रखे गए candidates को
`nearest_rescore` से f32 में दोबारा score करता है।
