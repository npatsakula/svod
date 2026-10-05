---
sidebar_label: Builder API
---

# Builder API

[एक कर्नेल लिखना](./first-kernel) में बस चंद calls इस्तेमाल हुए थे। यह page `svod-tk` का पूरा AUTHOR
चेहरा है: हर वह type और method जिससे कर्नेल body लिखी जाती है, इस हिसाब से grouped कि वह क्या करता है।
जब तक कोई module path न दिया गया हो, यहाँ की हर चीज़ `tk/src/lib.rs` से re-export होती है।

एक कर्नेल body एक closure `FnOnce(&Kernel) -> Arc<UOp>` है। यह globals bind करती है, tiles allocate करती है,
एक `Group` के ज़रिए tile ops emit करती है, और `ker.finish(n)` लौटाती है।

---

## `Kernel` — context

```rust
// tk/src/kernel.rs
pub fn new(name: impl Into<String>, grid: [i64; 3], block: i64, buffers: Vec<Arc<UOp>>, caps: ArchCaps) -> Kernel
```

इसे आप शायद ही कभी ख़ुद construct करते हैं: `run_kernel`, `compile_kernel`, `graph_launch` और
`graph_launch_multi` इसे आपके लिए launch buffers से bound करके बना देते हैं। जिन fields को आप पढ़ते हैं:

| Field / method | मतलब |
|---|---|
| `ker.caps` | target का `ArchCaps`: `arch` और `wave_size` (CDNA पर 64, RDNA, CUDA और Metal पर 32) |
| `ker.grid_x()` / `grid_y()` / `grid_z()` | `Special` UOps के रूप में `blockIdx.{x,y,z}` (render सिर्फ़ तब होते हैं जब इस्तेमाल हों) |
| `ker.thread_idx` | `threadIdx.x` |
| `ker.warpid()` / `ker.laneid()` | `threadIdx / wave_size` और `threadIdx % wave_size` |
| `ker.frag(role)` | इस arch पर किसी `FragRole` का physical `RTBaseShape` — जिस arch में matrix-core layouts नहीं हैं, वहाँ panic करता है |

`block` waves की पूरी संख्या होना चाहिए (`Kernel::new` में `debug_assert`); launch block आम तौर पर
`warps * caps.wave_size` होता है।

### Globals bind करना

```rust
// tk/src/scaffold.rs
pub fn bind_abi(&self, outputs: &[GlSpec], inputs: &[GlSpec]) -> (Vec<GL>, Vec<GL>)
pub fn gl(&self, shape: &[usize], dtype: DType) -> GL                // tk/src/tile.rs
```

`bind_abi` slice order में `gl` ही है: पहले outputs, फिर inputs, उसी buffer order से मेल खाते हुए जो launcher
को सौंपा गया था। bound buffer का dtype ही मान्य होता है; debug build assert करता है कि declared dtype की byte
width वही हो। कोई optional buffer (FA का `key_lens`) `bind_abi` के बाद एक trailing `gl` से bind होता है, बीच
में कभी नहीं।

```rust
let (outs, ins) = ker.bind_abi(
    &[GlSpec::new(&[1, 1, m, n], DType::BFloat16)],
    &[GlSpec::new(&[1, 1, m, k], DType::BFloat16), GlSpec::new(&[1, 1, n, k], DType::BFloat16)],
);
```

### Tiles allocate करना

Shape descriptors (`tk/src/tiles.rs`) विशुद्ध data हैं; wrappers (`tk/src/tile.rs`) एक buffer bind करते हैं।
raw constructors एक explicit base shape लेते हैं; scaffold shortcuts इसे `ker.caps` के ज़रिए role से resolve
करते हैं, और tree के कर्नेल इन्हीं को इस्तेमाल करते हैं।

| Raw | Shortcut | क्या allocate करता है |
|---|---|---|
| `ker.rt(dims, dtype, layout, base)` | `ker.acc(dims, layout)` | `Accumulator` fragment में f32 `RT` |
| | `ker.acc_t(dims, layout)` | `AccumulatorT` में f32 `RT` (transposed accumulator का N-major store) |
| | `ker.operand(dims, dt, layout)` | A-operand fragment में 16-bit `RT` |
| | `ker.operand_b(dims, dt, layout)` | B-operand fragment (`operand` से सिर्फ़ Metal पर अलग) |
| `ker.rv(length, dtype, VecLayout::Ortho, base)` | `ker.acc_vec(length)` | f32 `RV`, `length / frag_rows` tiles × `LaneMap::slots()` |
| `ker.st(dims, dtype, layout, base)` | `ker.shared(dims, dt, layout)` | arch की plain strip में LDS `ST` |
| | `ker.shared_sw(dims, dt, layout)` | XOR-swizzled strip में LDS `ST` |
| `ker.st_db(..)` / `ker.st_stages(.., stages)` | `ker.shared_db(..)` / `ker.shared_sw_stages(.., stages)` | software pipeline के लिए वही tile, एक `stages`× buffer पर |

`dims` elements में `(rows, cols)` है और दोनों axes पर base fragment का गुणक होना चाहिए (एक `assert`)।
`TileLayout::{Row, Col}` बताता है कि एक lane के registers किस axis के साथ चलते हैं; reductions और global hops
इसे पढ़ते हैं।

एक `RT` का logical shape `[height, width, ept]` है (fragment grid, फिर प्रति lane elements); एक `ST` का
`[height, width, frag_rows, frag_cols]`। `ST::subtile(dims, (row_blk, col_blk))` किसी shared tile में एक wave
के band का zero-copy view है; `ST::with_base_offset(off)` एक pipeline stage चुनता है
(`parity * st.half_elems()`)।

### Ordering

Tiles immutable handles हैं। हर op destination tile को उसके emit किए store पर एक `After` edge के साथ
**दोबारा wrap** करके लौटाता है, ताकि अगला read उसके बाद order हो। जो चीज़ dataflow express नहीं करता, उसके
लिए हाथ से पिरोए जाने वाले दो edges हैं:

- `t.after(deps)` — `t` के अगले read को `deps` के बाद order करना (एक tile, एक range, एक barrier, या इनका
  एक array या tuple; `tk/src/tile.rs` में `AfterDeps`)।
- `st.after(deps)` — इसका `ST` वाला रूप।

---

## `Group` — compute वाली शब्दावली

```rust
ker.warp()             // 1 wave
ker.group(n)           // 1×n waves, for collaborative GLOBAL→LDS fills
ker.group_2d(r, c)     // an r×c wave grid; group_threads = r·c·wave_size
```

`g.warp_row()` / `g.warp_col()` grid में wave के coordinates हैं; `g.warpid_in_group()` उसका flat index।
Register ops per-lane हैं और किसी भी group पर wave-safe हैं; single-wave ops (`map_position`, `col_reduce` पर
reductions, shuffles) `warps == 1` assert करते हैं — इन्हें multi-wave कर्नेल में भी `ker.warp()` पर ही कॉल करें।

### Movement

```rust
// tk/src/group/movement.rs
pub fn load<Dst, Src: LoadInto<'k, Dst>>(&self, dst: Dst, src: Src, ix: MoveIdx) -> Src::Output
pub fn store<Dst, Src: StoreInto<'k, Dst>>(&self, dst: Dst, src: Src, ix: MoveIdx) -> Src::Output
```

जायज़ address-space जोड़े trait impls हैं, इसलिए कोई नाजायज़ जोड़ा (`RT ← RT`) compile error बन जाता है:

| Call | जोड़ा | क्या emit करता है |
|---|---|---|
| `g.load(st, gl, ix)` | `ST ← GL` | सारे group threads पर coalesced fill + एक workgroup barrier |
| `g.load(rt, st, ix)` | `RT ← ST` | `LaneMap` के ज़रिए per-lane fragment gather (CUDA पर हर 16-bit fragment के लिए एक `ldmatrix.x4`) |
| `g.load(rt, gl, ix)` | `RT ← GL` | सीधा global gather, बीच में LDS पर कोई रुकावट नहीं |
| `g.store(st, rt, ix)` | `ST ← RT` | LDS में fragment scatter |
| `g.store(gl, rt, ix)` | `GL ← RT` | global में fragment scatter |

`MoveIdx` indices को role से नाम देता है: `MoveIdx::block(idxs, axis)` global में tile का coordinate है (हर
global dim के लिए एक entry; `axis` वह dim है जिसके stride पर एक tile row फैली है), `MoveIdx::frag(idxs)` एक
register-side fragment offset, `MoveIdx::at(block, frag, axis)` दोनों, और `MoveIdx::default()` कुछ नहीं (एक
subtile अपना band पहले से साथ रखता है)। `.masked()` एक `GLOBAL ↔ REG` hop को tensor के extent के मुक़ाबले gate
करता है: एक ragged edge `0.0` पढ़ता है और write छोड़ देता है।

Pipeline primitives एक fill को उसके synchronization से अलग करते हैं:

| Primitive | Arch | इस्तेमाल |
|---|---|---|
| `fill_local_nobar` / `fill_local_vec_nobar` | सभी | बिना trailing barrier का fill; fence caller लगाता है |
| `stage_global_to_reg(st, gl, idxs, axis)` → `commit_regs_to_local(&[(st, stage), ..])` | सभी (AMD path) | global loads अभी registers में, `ds_write` बाद में LDS में, ताकि loads मौजूदा block के MMAs के नीचे in flight रहें |
| `cp_async_fill(st, gl, idxs, axis)` (`cp_async_fill_applies` से gated) | CUDA sm_80+ | सीधे LDS में 16-byte `cp.async`; `cp_async_wait(n, ..)` + `.barrier(..)` से retire करें |
| `war_fence2(a, b, extra)` | सभी | एक cross-wave barrier जिसे दोनों gathers consume करते हैं, और जो prefetch commits को deps के रूप में साथ रखता है |
| `store_local_fenced(st, rt, ix, deps)` | सभी | एक `RT → ST` scatter और उसके बाद एक barrier (RDNA3 softmax relayout) |
| `store_global_with(gl, rt, ix, f)` | सभी | एक global store जिसकी value `f(v, offset)` है — fused epilogues |

### Matrix multiply

```rust
// tk/src/group/mma.rs — C += A·B over every output fragment, reducing along K
pub fn mma_ab  (&self, c: RT<'k>, a: &RT<'k>, b: &RT<'k>) -> RT<'k>   // a[h, k] · b[k, w]
pub fn mma_abt (&self, c: RT<'k>, a: &RT<'k>, b: &RT<'k>) -> RT<'k>   // a[h, k] · b[w, k]ᵀ
pub fn mma_atb (&self, c: RT<'k>, a: &RT<'k>, b: &RT<'k>) -> RT<'k>   // a[k, h]ᵀ · b[k, w]
pub fn mma_atbt(&self, c: RT<'k>, a: &RT<'k>, b: &RT<'k>) -> RT<'k>
```

`a`/`b` operand fragments में bf16 या f16 होते हैं, `c` accumulator fragment में f32 (वरना panic)। AMD पर हर
16×16×16 step के लिए एक `Op::Wmma`, CUDA पर दो `m16n8k16`, Apple पर हर fragment के लिए एक 8×8×8
`simdgroup_matrix` op; descriptor scheduler की `TensorCore` table से आता है, इसलिए hand कर्नेल और BEAM का `TC`
action एक ही source साझा करते हैं।

### Reductions और shuffles

```rust
// tk/src/group/reduce.rs
pub fn row_reduce<F>(&self, vec: RV<'k>, src: &RT<'k>, op: F, init: f64) -> RV<'k>
pub fn col_reduce<F>(&self, vec: RV<'k>, src: &RT<'k>, op: F, init: f64) -> RV<'k>
pub fn arg_reduce(&self, val: RV<'k>, idx: RV<'k>, src: &RT<'k>, dir: ArgDir) -> (RV<'k>, RV<'k>)
```

एक reduce पहले lane-local elements को fold करता है, फिर fragment के `ReduceTree` से lanes के आर-पार पूरा
करता है — AMD पर एक `ds_bpermute` sibling gather, CUDA और Metal पर एक `shfl.bfly` butterfly। `op` कोई भी
associative combiner है (`|a, b| a.max(b)`, `|a, b| a.add(b)`); नतीजा `vec` में fold होता है, इसलिए `vec` में
running value पहले से होनी चाहिए।

Scalar wave primitives (`tk/src/group/shuffle.rs`): `wave_reduce_scalar(value, op)`,
`subgroup_reduce_scalar(value, width, op)`, `broadcast_scalar(value, lane)`, और tile वाले रूप `shuffle_xor`,
`shuffle_down`, `shuffle_up`, `compare_exchange` (bitonic stages)। इनमें से कोई भी LDS को नहीं छूता।

### Elementwise

| Call | मतलब |
|---|---|
| `g.zero(rt)` / `g.ones(rt)` / `g.neg_inf(rt)`; `zero_rv` / `clear_rv(rv, v)` | constant fills |
| `g.copy(dst, &src)` | element copy, dtype mismatch पर cast के साथ |
| `g.transpose(dst, &src)` | fragment grid के `height` और `width` की अदला-बदली |
| `g.map(t, \|x, idx\| ..)` | हर element पर एक UOp expression लागू करना |
| `g.map_position(rt, row_blk, col_blk, \|x, idx, row, col\| ..)` | वही, element के global `(row, col)` के साथ, जो `LaneMap` से पढ़ा जाता है |
| `g.mask_where(rt, row_blk, col_blk, fill, \|row, col\| pred)` | `where(pred, fill, x)` — causal और padding masks |
| `g.add/sub/mul/div/maximum(a, &b)`, `*_scalar(a, s)`, `*_rv(rt, &rv)`, `g.exp2(t)` | tile math (`tk/src/math.rs`) |

Operator sugar (`tk/src/ops.rs`) उन्हीं calls तक route करता है, इसलिए body गणित जैसी पढ़ी जाती है:
`tk/src/kernels/fa.rs` में online-softmax update यह है

```rust
let scale_vec = (max_vec_last - &max_vec).exp2();
o_reg = o_reg * &scale_vec;
norm_vec = norm_vec * &scale_vec;
let att = (att - &max_vec).exp2();
```

`T op &T` same-shape है, `RT op &RV` vector को tile के layout axis के साथ broadcast करता है, `T op f64` एक
scalar है।

---

## Loops

```rust
// tk/src/loop_scope.rs
let lp = ker.loop_static(trips);          // a tracked RANGE with a constant trip count
let lp = ker.loop_dynamic(bound_uop);     // a runtime trip count (FA's causal block-skip)
lp.index()                                 // the counter, for addressing
lp.reinit(t)                               // t.after(range): re-run a per-trip init inside the loop
lp.close()                                 // end the last store around the range; returns the END
lp.close_carry(t)                          // close and rebind one carried tile to its post-loop value
lp.close_barrier(commits)                  // close with a workgroup fence folded into the END
```

दो नियम, जिन्हें भूलना नामुमकिन बनाने के लिए ही loop scope मौजूद है:

- हर iteration वाला re-init (body के ऊपर `g.zero(acc)`) loop counter पर निर्भर होना चाहिए, वरना linearizer
  उसे loop के ऊपर hoist कर देता है और accumulator बासी state ढोता रहता है। `g.zero(lp.reinit(acc))` लिखें।
- एक `RANGE` ठीक एक `END` स्वीकार करता है। एक loop में कई accumulators हों, तो बाक़ियों को उस एक closing
  store में chain करें (GEMM हर accumulator का A input पिछले accumulator के MMA से होकर पिरोता है), फिर हर
  final value को `acc.after(&ended)` के रूप में पढ़ें।

`Kernel::range` / `range_uop` / `endrange` / `endrange_to` / `endrange_barrier_to` वे raw रूप हैं जिन्हें
`Loop` wrap करता है; वे हूबहू वही graph emit करते हैं।

---

## Finish और launch

```rust
pub fn finish(&self, stores: usize) -> Arc<UOp>          // tk/src/kernel.rs
```

`finish(n)` आख़िरी `n` terminal stores निकालता है — हर output global के लिए एक — हर एक के इर्द-गिर्द कोई भी
अब तक खुली tracked range बंद करता है, और उन्हें
`KernelInfo { opts_to_apply: Some(vec![]), name: Some(name) }` के साथ sink करता है। जो कर्नेल `finish` के
वक़्त कोई range खुली छोड़ता है, उसमें एक ही store होना चाहिए। Stores stack तक movement ops के ज़रिए पहुँचते
हैं, या explicitly `ker.push_store(store, buf)` से (straight-line norm कर्नेल अपने vector stores को इसी
तरह group करता है)।

```rust
// tk/src/launch.rs
pub fn graph_launch(name, grid, block, out: Tensor, ins: &[&Tensor], caps: ArchCaps, build) -> Result<Tensor>
pub fn graph_launch_multi(name, grid, block, outs: Vec<Tensor>, ins, caps, build) -> Result<Vec<Tensor>>
pub fn launch_custom<T>(device, archs: ArchSet, validate, applies, build) -> Result<Option<T>>
pub fn run_kernel(name, grid, block, outs: &mut [&mut Tensor], ins: &[&Tensor], build) -> Result<()>
pub fn compile_kernel(name, grid, block, outs, ins, build) -> Result<CompiledLaunch>
```

`graph_launch` `SINK` को एक `Op::Call` node के रूप में wrap करता है और एक lazy tensor लौटाता है; `out`
`Tensor::empty(shape, dtype)` है, और body जिन placeholders को देखती है वे `[out, ins...]` हैं — यानी `bind_abi`
वाला order। `launch_custom` वह तीन-तरफ़ा policy है जिसे हर library कर्नेल मानता है
([IR में authoring](./lowering)): कर्नेल के `ArchSet` के मुक़ाबले `resolve_supported_arch` (उसके बाहर
`Ok(None)`), `validate(arch)` (ग़लत request पर `Err`), `applies(arch)` (shape tile न हो तो `Ok(None)`), और
फिर `build(arch)`।

```rust
// tk/src/kernels/norm.rs — the shape of every graph-native entry
crate::launch_custom(
    &x.device(),
    NORM_SUPPORTED_ARCHS,
    move |_arch| check_norm_operands("rms-norm", &check.0, &check.1, &check.2, check.3),
    move |arch| select_norm_cfg(rows, d, crate::ArchCaps::for_arch(arch).wave_size).is_some(),
    move |arch| {
        let caps = crate::ArchCaps::for_arch(arch);
        let cfg = select_norm_cfg(rows, d, caps.wave_size).expect("checked by the fit predicate");
        let (grid, block) = launch_dims(rows, cfg.rows_per_block, caps.wave_size);
        let out = Tensor::empty(&xd, dtype.clone());
        let dt = dtype.clone();
        crate::graph_launch("rms_norm", grid, block, out, &[x, weight], caps, move |ker| {
            build_row_norm(ker, rows, d, dt, eps, cfg, false);
            ker.finish(1)
        })
    },
)
```

`run_kernel` / `compile_kernel` direct-dispatch वाला DEBUG चेहरा हैं; देखें [डीबगिंग](./debugging)।

---

## Tiles के नीचे

कुछ कर्नेल को tiles नहीं, addresses चाहिए। `tk/src/index.rs` वह flat-addressing layer है जिस पर हर tile op
बना है, और यह public है: `Idx` (`Const(i64)` या `Uop`), `flat_index(buf,
shape, idxs)`, `load_at`, `load_off`, `load_off_vec(buf, off, w)` / `store_off_vec` (प्रति lane एक `w`-wide
access, जिसे renderer एक 128-bit instruction में fold कर देता है), और gated रूप `load_off_gated` /
`index_off_gated`। RMS-norm कर्नेल और `model/src/qwen3/tk/mod.rs` में Qwen3 QKV-norm-RoPE prologue पूरी तरह
इसी level पर लिखे गए हैं — प्रति row एक wave, न कोई `RANGE`, न LDS — और norm की row वाली शब्दावली (`plan`,
`vload`, `vpick`, `vstore`, `inv_rms`, `scale_by`) दोबारा इस्तेमाल करते हैं।

`tk/src/asm.rs` AMD machine-scheduler controls को एक dependency पर पिरोए गए typed `Op::Custom` nodes के रूप में
expose करता है: `s_setprio(prio, dep)`, `s_waitcnt_lgkmcnt(n, dep)`, `sched_barrier(mask, dep)`,
`iglp_opt(mode, dep)`। GEMM gfx12 पर `sched_barrier(0, ..)` इस्तेमाल करता है, जहाँ
`ArchCaps::needs_pipeline_commit_fence()` बताता है कि वरना backend scheduler pipeline के LDS commit को trip के
MMAs के ऊपर hoist कर देगा।

`tk/src/grid.rs::l2_swizzle(wgid, num_wgs, grid_m, grid_n)` एक flattened workgroup id को `(pid_m, pid_n)` पर
map करता है ताकि साथ-साथ scheduled workgroups एक XCD का L2 साझा करें (HipKittens का chiplet transform);
`GemmCfg::l2_swizzle` इसे चालू करता है।
