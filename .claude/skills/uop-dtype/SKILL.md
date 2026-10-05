---
name: uop-dtype
description: Verified constructor and accessor signatures for `svod_ir::UOp`, `Op`, `DType`/`ScalarDType`/`AddrSpace`/`DeviceSpec`, plus the Tensor-level entry points. Use when building IR by hand (tests, passes, custom kernels), fixing a "no method" or type-mismatch error on UOp/DType, or checking which constructor is fallible, which takes `&Arc<Self>` vs an owned `Arc`, and what an Op's fields are named.
---

# UOp and DType reference

Source: `ir/src/uop/constructors/{data,compute,control,memory,shape,reduce,hardware,graph}.rs`, `ir/src/uop/core.rs`,
`ir/src/op.rs`, `ir/src/types.rs`, `dtype/src/{lib,cast}.rs`. Op semantics: `website/docs/architecture/op-bestiary.md`,
`ir-design.md`. Import `svod_ir::prelude::*` (`UOp, Op, DType, DeviceSpec, SInt, IndexSpec, IntoUOp, ...`).

Conventions: `try_*` returns `svod_ir::Result<Arc<UOp>>`; the un-prefixed twin panics on a type error and is for rewrite
bodies after validation. `self: &Arc<Self>` methods are called on an `Arc<UOp>`; associated fns take owned `Arc`s.
Children live in `SmallVec<[Arc<UOp>; 4]>` — build them with `smallvec![..]`. `UOp` is hash-consed: equal structure ⇒
same `Arc` ⇒ `Arc::ptr_eq`.

## Accessors (`core.rs`, `helpers.rs`)

`op() -> &Op`, `dtype() -> DType`, `id: u64`, `src_ops() -> OpMask`, `shape() -> Result<Option<&Shape>>`, `vmin()/vmax() -> &ConstValue`,
`toposort() -> Vec<Arc<UOp>>` (`toposort_filtered`, `_call_aware`), `node_count()`, `tree()/tree_full() -> String`, `ranges()`,
`in_scope_ranges()`, `get_consumer_map()`, `backward_slice()`, `with_sources(Vec)`, `replace(Option<DType>, Option<Vec>)`,
`substitute(&HashMap<UOpKey, Arc<UOp>>)` (+ `_walk`, `_preserve_calls`, `_gated`), `base()`, `buf_uop()`, `unwrap_after()`,
`unwrap_cast()`, `ptrdtype()`, `addrspace()`, `device_spec()`, `buffer_size()`, `get_idx()/get_valid()` (split
`WHERE(valid, idx, Invalid)`), `UOp::invalid_marker()`, `UOp::is_invalid_marker(&u)`, `const_factor()`, `divides(v)`,
`split_uop(BinaryOp)`, `pop_const(BinaryOp)`, `tag()/rtag()`, `origin()/rorigin()`.

## Constructors

### Constants and storage (`data.rs`)
| Signature | Notes |
|-----------|-------|
| `const_(dtype, ConstValue) -> Arc` / `try_const_` | `ConstValue::{Int(i64), UInt(u64), Float(f64), Bool, Invalid}` |
| `native_const<T: HasDType + IntoUOp>(v)` | dtype from the Rust type (`1i32`, `0.5f32`) |
| `index_const(i64)` | `WeakInt` constant (weak until it meets a concrete dtype, like Tinygrad `UOp.const`) |
| `u.const_like<T: IntoUOp>(v)` / `u.vconst_like(v)` | same dtype as `u` (vector: broadcast lanes) |
| `vconst(Vec<ConstValue>, scalar_dtype) -> Arc` / `try_vconst` | VCONST; dtype is `scalar.vec(len)` |
| `new_buffer(DeviceSpec, size, dtype)` | fresh slot, `AddrSpace::Global`; `buffer(slot, size, dtype, addrspace, Option<DeviceSpec>)` for Local/Reg (device must be `None`) |
| `param(slot, size, dtype, Option<DeviceSpec>)`, `param_with_shape(slot, &Shape, dtype, dev)`, `scalar_param(slot, name, dtype, min, max)` | PARAM = positional buffer; what kernels see after the cut |
| `placeholder(&Shape, dtype, slot, addrspace, dev) -> Result` | PARAM/BUFFER reshaped to `shape` (Tinygrad `UOp.placeholder`) |
| `u.contiguous_slice(size, offset_elems, dtype)` | SLICE view |
| `u.cast(dtype)`, `u.bitcast(dtype)`, `noop()`, `buffer_id(Option<usize>)`, `lunique(..)` | |

### Arithmetic (`compute.rs`), all `self: &Arc<Self>, rhs: &Arc<Self>` unless noted
- Fallible binary: `try_add try_mul try_sub try_div(FloorDiv) try_mod(FloorMod) try_cdiv try_cmod try_max try_pow`; bitwise `try_and_op try_or_op try_xor_op`; shifts `try_shl_op try_shr_op`; comparisons `try_cmplt try_cmple try_cmpeq try_cmpne try_cmpgt try_cmpge` (→ `Bool`, vector-aware). Promotion via `promote_and_cast`; shapes must match exactly (no broadcasting at IR level); constant zero divisor is an error.
- Scalar rhs: `UOp::try_add_scalar(lhs: Arc, v: impl IntoUOp)`, `try_sub_scalar`, `try_mul_scalar`, `try_mod_scalar`; or `1.into_uop(dtype)`.
- Panicking: `add sub mul floor_div mod_ cdiv cmod max and_ or_ xor shl shr lt le gt ge eq ne`, `alu(BinaryOp, lhs, rhs)`.
- Ternary: `UOp::try_where(cond, t, f)`, `UOp::try_mulacc(a, b, c)`, `UOp::threefry(lhs, rhs) -> Result`.
- Unary infallible methods: `neg abs square sign not`; rounding are associated fns on owned values: `UOp::trunc(x)`, `floor`, `ceil`, `round`.
- Float-only `Result`: `try_sqrt try_rsqrt try_exp try_exp2 try_log try_log2 try_sin try_cos try_tan`, `erf()`, `UOp::try_reciprocal(&x)`.

### Ranges and control (`control.rs`)
| Signature | Notes |
|-----------|-------|
| `range_axis(end: Arc, AxisId, AxisType)` / `range_axis_dtype(.., dtype)` | general RANGE |
| `range(end: Arc, axis_id: usize)`, `range_const(end: i64, axis_id: usize)` | `AxisType::Loop`, `AxisId::Renumbered(id)` |
| `u.end(smallvec![ranges])` | END closes ranges |
| `if_(cond, smallvec![body])`, `endif(if_op)`, `u.barrier(smallvec![deps])` | |
| `var(name, dtype, min, max)`, `define_var(name: String, min, max)` (Index), `variable(name, min, max, dtype)`, `u.bind(value)` | symbolic vars |
| `special(end, name: String)`, `special_dtype(..)` | GPU id (`gidx0`, ...) |

`AxisType::{Device, Global, Warp, Local, Weak, Loop, GroupReduce, Reduce, Upcast, Unroll, Thread, Placeholder}`;
`AxisId::{Unrenumbered(usize), Renumbered(usize), UnrenumberedPath(..), RenumberedPath(..)}`; `ReduceOp::{Add, Mul, Max, Min}`.

### Memory (`memory.rs`, `bon` builders)
```rust
let idx = UOp::index().buffer(buf).indices(vec![i, j]).call()?;      // Result; .dtype(..) optional; indices must be int dtype
let gated = UOp::index().buffer(buf).indices(vec![i.valid(cond)]).call()?;   // valid = WHERE(cond, i, Invalid)
let v = UOp::load().index(idx).call();                                // infallible; dtype = index dtype
let g = UOp::load().index(idx).alt(zero).gate(cond).call();           // alt and gate go together
let st = idx.store(value);  let st = idx.store_gated(value, gate);    // STORE, dtype Void
buf.index_axes(vec![2])  // constant-position INDEX (several → STACK index)
UOp::slice(buffer, Vec<IndexSpec>)?;  u.getaddr(Option<DeviceSpec>);  u.copy_to_device(dev) / u.copy(dev)
UOp::stage(compute, Vec<ranges>, BufferizeOpts) / stage_global(compute, ranges) / stage_local(compute, ranges)   // STAGE
```
`BufferizeOpts { device: Option<DeviceSpec>, local_axis: Option<AxisId>, addrspace: AddrSpace, removable: bool }` (`BufferizeOpts::local()`).

### Shapes (`shape.rs`), all `self: &Arc<Self>`
`try_reshape(&Shape)`, `try_expand(&Shape)`, `try_permute(Vec<usize>)`, `try_pad(&[(SInt, SInt)])`, `try_shrink(&[(SInt, SInt)])`,
`try_flip(Vec<bool>)`; `UOp::stack(smallvec![..])` (STACK: shaped lane value, the only "vector" op); `UOp::multi(src, axis)`.
`Shape = SmallVec<[SInt; _]>` (`Shape::from_iter(dims.map(SInt::Const))`); shapes are UOps inside `Reshape/Expand/Pad/Shrink` (`shape_to_uop`).

### Reductions (`reduce.rs`)
`u.try_reduce_axis(ReduceOp, Vec<usize>)` (tensor-level REDUCE_AXIS; returns `u` if every axis is 1),
`u.reduce(smallvec![ranges], ReduceOp)` (kernel-level REDUCE over RANGEs), `reduce_with_num_axes(..)`, `UOp::allreduce(src, DeviceSpec, ReduceOp)`.

### Kernel / program level (`hardware.rs`, `graph.rs`)
| Signature | Notes |
|-----------|-------|
| `sink(Vec)`, `sink_with_info(Vec, KernelInfo)`, `group(Vec)` | `SINK[KERNEL]` is a sink with `KernelInfo` |
| `body.call(smallvec![args], CallInfo)`, `body.function(args, info)` / `try_function`, `tuple(..)`, `u.gettuple(i)` / `try_gettuple` | a kernel is `CALL(SINK[KERNEL], args)` wrapped in `AFTER` |
| `u.after(smallvec![deps])` | AFTER: `u` ordered after `deps` |
| `program(..)`, `linear(smallvec![ops])`, `source(String)`, `binary(Vec<u8>)`, `ins(srcs, dtype, InsArg)` | codegen stages |
| `wmma(a, b, c, WmmaMetadata)`, `u.broadcast(n)` / `try_broadcast`, `mstack(..)`, `u.mselect(i)` | |
| `u.detach()`, `u.contiguous()`, `contiguous_with_opts(hints)`, `contiguous_backward()`, `precast()` | frontend markers |
| `custom(deps, code, dtype)`, `customi(..)`, `custom_function(kind, attrs)`, `custom_kernel(srcs, fxn, info)`, `placeholder_like(src, slot, addrspace)` | custom code |

Removed/renamed — do not look for: `define_global/local/reg` (→ `param`/`buffer` with `AddrSpace`), `bufferize*` (→ `stage*`),
`vectorize/gep/unroll/contract` (→ `stack`, `index_axes`, expander `RangeMap`), `view` (→ `contiguous_slice`), `device()`,
`assign` (→ `store` on an INDEX of the target + `after`), `range_outer_const`.

## Op enum (`ir/src/op.rs`, `#[op_enum]` gives `svod_ir::ops::<Variant>` structs)

`Const(ConstValueHash) Unique LUnique Noop Sink{sources,info} Group{sources} Unary(UnaryOp, a) Binary(BinaryOp, a, b)
Ternary(TernaryOp, a, b, c) Cast{src,dtype} BitCast{src,dtype} MSelect{buffer,device_index} Special{end,name}
Param{shape,arg} Buffer{shape,arg} Slice{buffer,offset,size} Stage{compute,ranges,opts} Index{buffer,indices}
GetAddr{src,device} Copy{src,device} MStack{buffers} Reshape{src,new_shape} Permute{src,axes} Expand{src,new_shape}
Pad{src,begin_pads,end_pads} Shrink{src,offsets,sizes} Flip{src,axes} Multi{src,axis} ReduceAxis{src,reduce_op,axes}
Reduce{src,ranges,reduce_op,num_axes} AllReduce{src,device,reduce_op} If{condition,body} EndIf{if_op}
Range{end,axis_id,axis_type,deps} End{computation,ranges} Barrier{src,deps} Stack{sources} VConst{values}
DefineVar{name,min_val,max_val} Bind{var,value} Wmma{a,b,c,metadata} Call{body,args,info} Function{body,args,info}
Tuple{src} GetTuple{src,index} Program{sink,info,linear,source,binary} Linear{ops} Source{code,identity}
ProgramBinary{bytes,identity} Detach{src} Contiguous{src,opts} ContiguousBackward{src} After{passthrough,deps}
Precast{src} Custom{deps,code} CustomFunction{kind,attrs} CustomI{deps,code} Load{index,alt,gate} Store{index,value,gate}
Ins{sources,arg}`

Match with `Op::Load(ops::Load { index, gate: Some(g), .. })`; `op.children()` gives the child list; `OpKey::from_op(op)`
(`svod_ir::op::pattern_derived`) is the dispatch key used by `patterns!` (see `/patterns`).

## DType (`svod_dtype`)

```rust
pub enum DType { Scalar(ScalarDType), Vector { scalar: ScalarDType, count: usize },
                 Ptr { base: Box<DType>, addrspace: AddrSpace, size: Option<usize>, vcount: usize },
                 Image { kind: ImageKind, shape: Vec<usize> } }
pub enum ScalarDType { Bool, WeakInt, Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64, WeakFloat,
                       FP8E4M3, FP8E4M3FNUZ, FP8E5M2, FP8E5M2FNUZ, Float16, BFloat16, Float32, Float64, Void, Index }
pub enum AddrSpace { Global, Local, Reg }
pub enum DeviceSpec { Cpu, Cuda { device_id }, Amd { device_id }, Metal { device_id }, WebGpu, Disk { path } }
```
Associated consts: `DType::Float32`, `DType::Int32`, `DType::Bool`, `DType::Index`, `DType::WeakInt`, `DType::WeakFloat`, `DType::Void`, ...
(`DEFAULT_INT = Int32`, `DEFAULT_FLOAT = Float32`). `WeakInt`/`WeakFloat` are untyped literals: `is_weak()`, `strong_dtype()`,
`weak_dtype()`; index arithmetic stays `WeakInt` until `pm_lower_index_dtype` (`17-pm_lower_index_dtype`) commits it to
`Int32`/`Int64`.

| Method | Returns |
|--------|---------|
| `vec(count)` | `Option<DType>` (`None` for non-scalar); `ScalarDType::vec(count)` is infallible |
| `ptr(size: Option<usize>, AddrSpace)` | `Option<DType>` |
| `scalar()` | `Option<ScalarDType>`; `base()` → `ScalarDType` through Vector/Ptr; `scalar_dtype()` → `DType` |
| `count()` / `vcount()`, `is_vector()`, `is_image()` | lane count |
| `bytes()`, `is_bool/is_int/is_float/is_signed/is_unsigned/is_fp8/is_weak()` | |
| `min_value()/max_value()`, `c_style()`, `with_base(ScalarDType)`, `with_ptr_base(DType)` | |
| `DType::least_upper_dtype(&[DType]) -> Option<DType>` (`cast.rs`) | promotion |
| `HasDType` (Rust type → dtype), `IntoUOp` (`v.into_uop(dtype)`) | traits behind `native_const`/`const_like` |

`DeviceSpec::canonicalize()` gives `"CPU"`, `"CUDA:0"`; the default device is `svod_dtype::default_device::default_device()`
(`SVOD_DEVICE` or platform default), scoped with `with_default_device(spec, || ..)`.

## Validation and errors

`promote_and_cast`, `check_bitwise_dtype`, `check_division_by_zero`, `validate_binary_shapes`, `validate_permutation`,
`validate_reduce_axes`, `validate_flip_axes` (`constructors/mod.rs`) are `pub(crate)` — outside `svod_ir` rely on the `try_*`
result. Errors are `svod_ir::Error` (snafu): `DTypeMismatch`, `IndexTypeMismatch`, `InvalidDTypeForUnaryOp`,
`WhereConditionNotBool`, `SymbolicShapeUnsupported`, ...; add context with `.context(MySnafu)`.

## Tensor layer (`tensor/src`)

```rust
let c = (&a + &b)?;                                     // std::ops on &Tensor → Result<Tensor> (+ - * / % & | ^ << >>; scalar lhs too)
let r = a.try_reshape([2, 3])?;  let t = a.try_transpose(0, 1)?;
let s = a.sum(())?;  let s1 = a.sum(1)?;  let m = a.max_with().axes(0).keepdim(true).call()?;
let y = a.matmul(&b)?;  let d = a.dot(&b)?;             // matmul_with(&b, Some(dtype))
y.realize()?;  y.realize_with(&PrepareConfig::for_cpu_backend(CpuBackend::Llvm))?;
let plan = y.prepare()?;                                // ExecutionPlan without executing
let v: Vec<f32> = y.to_vec()?;  let x: f32 = y.item()?;  let g = y.uop();   // Arc<UOp> from the registry
```

## Checking what you built

```rust
println!("{}", u.tree());                     // [id] OP : dtype shape=[..] with → (see above) back-refs
assert_eq!(u.dtype(), DType::Float32);
let shape = u.shape()?;                       // Option<&Shape>; Err for unshapeable graphs
svod_schedule::spec::type_verify(&u, &spec)?; // the boundary checks the pipeline runs (SVOD_SPEC)
```
Common errors: `IndexTypeMismatch` (an index that is not an int dtype — use `index_const` or an int-typed UOp), `DTypeMismatch`
on `load().dtype(..)` (must equal the INDEX dtype), `assert_eq!(alt.is_some(), gate.is_some())` on LOAD, `BUFFER dtype is
the stored element dtype, not a pointer` (pass `Float32`, not `Float32.ptr(..)`), shape mismatch on binary ops (broadcast
at the Tensor layer, not in IR).
