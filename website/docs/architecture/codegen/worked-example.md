---
sidebar_label: Worked example
---

# Worked Example: a Row Sum on the CPU

Every tree on this page is real output. The program:

```rust
let data = Array2::from_shape_fn((8, 64), |(r, c)| (r * 64 + c) as f32);
let x = Tensor::from_ndarray(&data);
let y = x.sum(1)?;
y.realize()?;
```

captured with `SVOD_PER_STAGE_UOPS=1 SVOD_DUMP_STAGE=` (every post-opt stage) and `RUST_LOG=svod_schedule::rangeify::transforms=debug,svod_schedule::optimizer=debug` under the JSON subscriber for the earlier passes, on the default CPU backend (LLVM, in-process). Node ids are allocation order and will differ between runs; the structure will not.

## Tensor graph

`sum(1)` is a tensor-form `REDUCE` over a `PERMUTE`d `RESHAPE` of the flat 512-element buffer, wrapped in `CONTIGUOUS` because the result is an output:

```text
[16] SINK : Scalar(Void)
└── [15] CONTIGUOUS : Scalar(Float32) shape=[Const(8)]
    └── [14] REDUCE(Add, num_axes=1, ranges=[]) : Scalar(Float32) shape=[Const(8)]
        └── [13] PERMUTE(axes=[1, 0]) : Scalar(Float32) shape=[Const(64), Const(8)]
            └── [12] RESHAPE : Scalar(Float32) shape=[Const(8), Const(64)]
                ├── [11] PARAM(slot=0) : Scalar(Float32) shape=[Const(512)]
                │   └── [0] CONST(Int(512)) : Scalar(WeakInt) shape=[]
                └── [4] STACK(len=2) : Scalar(WeakInt) shape=[Const(2)]
                    ├── [2] CONST(Int(8)) : Scalar(WeakInt) shape=[]
                    └── [3] CONST(Int(64)) : Scalar(WeakInt) shape=[]
```

The `STACK` is the reshape's shape payload — shapes are UOps too.

## After rangeify

Range assignment gives the output a `Weak` range `U0` (8) and the reduction a `Reduce` range `U1` (64); the movement ops collapse into the index `U0 * 64 + U1` (see the [rangeify page](./rangeify.md) for the tree). The kernel cut turns the `STAGE` into `STORE`/`END`, numbers the buffers as `PARAM`s and renumbers the ranges. The kernel body that enters `apply_pre_optimization`:

```text
[97] SINK[KERNEL] : Scalar(Void)
└── [96] END : Scalar(Void) shape=[]
    ├── [95] STORE : Scalar(Void) shape=[]
    │   ├── [94] INDEX : Scalar(Float32) shape=[]
    │   │   ├── [84] PARAM(slot=0) : Scalar(Float32) shape=[Const(8)]
    │   │   │   └── [2] CONST(Int(8)) : Scalar(WeakInt) shape=[]
    │   │   └── [89] RANGE(R1, Weak) : Scalar(WeakInt) shape=[]
    │   │       └── [2] → (see above)
    │   └── [93] REDUCE(Add, num_axes=0, ranges=[88]) : Scalar(Float32) shape=[]
    │       ├── [92] INDEX : Scalar(Float32) shape=[]
    │       │   ├── [87] PARAM(slot=1) : Scalar(Float32) shape=[Const(512)]
    │       │   │   └── [0] CONST(Int(512)) : Scalar(WeakInt) shape=[]
    │       │   └── [91] Add : Scalar(WeakInt) shape=[]
    │       │       ├── [90] Mul : Scalar(WeakInt) shape=[]
    │       │       │   ├── [89] → (see above)
    │       │       │   └── [3] CONST(Int(64)) : Scalar(WeakInt) shape=[]
    │       │       └── [88] RANGE(R0, Reduce) : Scalar(WeakInt) shape=[]
    │       │           └── [3] → (see above)
    │       └── [88] → (see above)
    └── [89] → (see above)
```

Slot 0 is the output (the `STAGE`'s buffer was mapped first), slot 1 the input. The five pre-optimization steps leave this graph untouched: no movement op, no collapsible reduce, no modulo to split, nothing to merge.

## After the optimizer (`00-initial`)

The CPU renderer has no locals, so `convert_loop_to_global` leaves `R1` as `Weak`. `hand_coded_optimizations` skips tensor cores, image upcasts, the matvec path and grouped reductions; `apply_unroll` sees a 64-wide reduce (above 32) and applies `UNROLL(0, 4)`; the kernel is already unrolled, so `apply_default_upcast` does nothing; 512 elements are far below the 131072-per-thread threshold, so no `THREAD`. The kernel is named `r_8_16_4` (reduce; extents 8, 16, 4):

```text
[135] SINK[KERNEL] : Scalar(Void)
└── [131] END : Scalar(Void) shape=[]
    ├── [130] STORE : Scalar(Void) shape=[]
    │   ├── [94] INDEX : Scalar(Float32) shape=[]
    │   │   ├── [84] PARAM(slot=0) : Scalar(Float32) shape=[Const(8)]
    │   │   │   └── [2] CONST(Int(8)) : Scalar(WeakInt) shape=[]
    │   │   └── [89] RANGE(R1, Weak) : Scalar(WeakInt) shape=[]
    │   │       └── [2] → (see above)
    │   └── [128] REDUCE(Add, num_axes=0, ranges=[118, 117]) : Scalar(Float32) shape=[]
    │       ├── [122] INDEX : Scalar(Float32) shape=[]
    │       │   ├── [87] PARAM(slot=1) : Scalar(Float32) shape=[Const(512)]
    │       │   │   └── [0] CONST(Int(512)) : Scalar(WeakInt) shape=[]
    │       │   └── [121] Add : Scalar(WeakInt) shape=[]
    │       │       ├── [90] Mul : Scalar(WeakInt) shape=[]
    │       │       │   ├── [89] → (see above)
    │       │       │   └── [3] CONST(Int(64)) : Scalar(WeakInt) shape=[]
    │       │       └── [120] Add : Scalar(WeakInt) shape=[]
    │       │           ├── [119] Mul : Scalar(WeakInt) shape=[]
    │       │           │   ├── [118] RANGE(R0, Reduce) : Scalar(WeakInt) shape=[]
    │       │           │   │   └── [115] CONST(Int(16)) : Scalar(WeakInt) shape=[]
    │       │           │   └── [116] CONST(Int(4)) : Scalar(WeakInt) shape=[]
    │       │           └── [117] RANGE(R2, Unroll) : Scalar(WeakInt) shape=[]
    │       │               └── [116] → (see above)
    │       ├── [118] → (see above)
    │       └── [117] → (see above)
    └── [89] → (see above)
```

`apply_opt` split the 64-wide `R0` into `R0 * 4 + R2` with `R0: Reduce(16)` and `R2: Unroll(4)`, and `pm_flatten_range` listed both on the `REDUCE`. Node count 20.

## `08-post_opt_sym`

Only `commutative_canonicalization` fires: the index becomes `(R0*4 + R2) + R1*64` (the tuplize order of the operands). Still 20 nodes.

## `09-pre_expand`

`R2` is replaced by `RESHAPE(STACK(0,1,2,3), [4])`, every consumer becomes shaped, and `expand_reduce` moves the lane axis into `num_axes`:

```text
[157] SINK[KERNEL] : Scalar(Void)
└── [155] END : Scalar(Void) shape=[]
    ├── [154] STORE : Scalar(Void) shape=[]
    │   ├── [94] INDEX : Scalar(Float32) shape=[]
    │   │   ├── [84] PARAM(slot=0) : Scalar(Float32) shape=[Const(8)]
    │   │   │   └── [2] CONST(Int(8)) : Scalar(WeakInt) shape=[]
    │   │   └── [89] RANGE(R1, Weak) : Scalar(WeakInt) shape=[]
    │   │       └── [2] → (see above)
    │   └── [152] RESHAPE : Scalar(Float32) shape=[Const(1)]
    │       ├── [151] REDUCE(Add, num_axes=1, ranges=[118]) : Scalar(Float32) shape=[]
    │       │   ├── [149] INDEX : Scalar(Float32) shape=[Const(4)]
    │       │   │   ├── [87] PARAM(slot=1) : Scalar(Float32) shape=[Const(512)]
    │       │   │   │   └── [0] CONST(Int(512)) : Scalar(WeakInt) shape=[]
    │       │   │   └── [148] Add : Scalar(WeakInt) shape=[Const(4)]
    │       │   │       ├── [147] Add : Scalar(WeakInt) shape=[Const(4)]
    │       │   │       │   ├── [119] Mul : Scalar(WeakInt) shape=[]
    │       │   │       │   │   ├── [118] RANGE(R0, Reduce) : Scalar(WeakInt) shape=[]
    │       │   │       │   │   │   └── [115] CONST(Int(16)) : Scalar(WeakInt) shape=[]
    │       │   │       │   │   └── [116] CONST(Int(4)) : Scalar(WeakInt) shape=[]
    │       │   │       │   └── [146] STACK(len=4) : Scalar(WeakInt) shape=[Const(4)]
    │       │   │       │       ├── [29] CONST(Int(0)) : Scalar(WeakInt) shape=[]
    │       │   │       │       ├── [28] CONST(Int(1)) : Scalar(WeakInt) shape=[]
    │       │   │       │       ├── [144] CONST(Int(2)) : Scalar(WeakInt) shape=[]
    │       │   │       │       └── [145] CONST(Int(3)) : Scalar(WeakInt) shape=[]
    │       │   │       └── [90] Mul : Scalar(WeakInt) shape=[]
    │       │   │           ├── [89] → (see above)
    │       │   │           └── [3] CONST(Int(64)) : Scalar(WeakInt) shape=[]
    │       │   └── [118] → (see above)
    │       └── [28] → (see above)
    └── [89] → (see above)
```

The `RESHAPE` to `[1]` is the placeholder `expand_reduce` leaves for the reduced lane axis; the devectorizer removes it.

## `10-pm_reduce`

`reduce_to_acc` builds the accumulator. `horizontal_reduce` folds the four lanes first (`((a0 + a1) + a2) + a3`, each lane an `INDEX` into the shaped index expression), then the loop over `R0` accumulates into a register buffer:

```text
[193] SINK[KERNEL] : Scalar(Void)
└── [192] END : Scalar(Void) shape=[]
    ├── [191] STORE : Scalar(Void) shape=[]
    │   ├── [94] INDEX : Scalar(Float32) shape=[]          ← PARAM(slot=0)[R1]
    │   └── [189] AFTER : Scalar(Float32) shape=[Const(1)]
    │       ├── [165] BUFFER(slot=0, addrspace=Some(Reg)) : Scalar(Float32) shape=[Const(1)]
    │       │   └── [28] CONST(Int(1)) : Scalar(WeakInt) shape=[]
    │       └── [188] END : Scalar(Void) shape=[Const(1)]
    │           ├── [187] STORE : Scalar(Void) shape=[Const(1)]
    │           │   ├── [165] → (see above)
    │           │   └── [186] Add : Scalar(Float32) shape=[Const(1)]
    │           │       ├── [169] AFTER : Scalar(Float32) shape=[Const(1)]
    │           │       │   ├── [165] → (see above)
    │           │       │   ├── [168] STORE : Scalar(Void) shape=[Const(1)]
    │           │       │   │   ├── [167] AFTER : Scalar(Float32) shape=[Const(1)]
    │           │       │   │   │   ├── [165] → (see above)
    │           │       │   │   │   └── [89] → (see above)          ← init inside the R1 loop
    │           │       │   │   └── [166] CONST(Float(0.0)) : Scalar(Float32) shape=[]
    │           │       │   └── [118] RANGE(R0, Reduce) : Scalar(WeakInt) shape=[]
    │           │       │       └── [115] CONST(Int(16)) : Scalar(WeakInt) shape=[]
    │           │       └── [185] Add : Scalar(Float32) shape=[]
    │           │           ├── [182] Add : Scalar(Float32) shape=[]
    │           │           │   ├── [179] Add : Scalar(Float32) shape=[]
    │           │           │   │   ├── [176] INDEX : Scalar(Float32) shape=[]
    │           │           │   │   │   ├── [87] PARAM(slot=1) : Scalar(Float32) shape=[Const(512)]
    │           │           │   │   │   └── [175] INDEX : Scalar(WeakInt) shape=[]
    │           │           │   │   │       ├── [148] Add : Scalar(WeakInt) shape=[Const(4)]   ← shaped index
    │           │           │   │   │       └── [29] CONST(Int(0))                             ← lane 0
    │           │           │   │   └── [178] INDEX ... lane 1
    │           │           │   └── [181] INDEX ... lane 2
    │           │           └── [184] INDEX ... lane 3
    │           └── [118] → (see above)
    └── [89] → (see above)
```

(Trimmed: the four lanes are identical except for the constant lane index.) Note `AFTER(acc, [R1])` on the init store: `input_ranges` places the zeroing inside the row loop. Stages `11` and `12` change nothing — no local stage, no GPU ranges.

## `13-pm_add_loads` and `14-devectorize`

`pm_expand_broadcast` makes the scalar terms of the shaped index explicit (`EXPAND(RESHAPE(R0*4, [1]), [4])`, same for `R1*64`) and `pm_add_loads` wraps the register reads and the four input lanes in `LOAD` (55 nodes). `devectorize` then scalarizes everything: the shaped index collapses into four scalar `Add`s and the per-lane `STORE`s are grouped. After `15-early_symbolic` the lanes read as `LOAD(INDEX(PARAM(1), (R0*4 + R1*64) + k))`:

```text
[269] LOAD : Scalar(Float32) shape=[]
└── [268] INDEX : Scalar(Float32) shape=[]
    ├── [87] PARAM(slot=1) : Scalar(Float32) shape=[Const(512)]
    └── [253] Add : Scalar(WeakInt) shape=[]
        ├── [119] Mul : Scalar(WeakInt) shape=[]      ← R0 * 4
        └── [90] Mul : Scalar(WeakInt) shape=[]       ← R1 * 64
[305] LOAD : Scalar(Float32) shape=[]
└── [304] INDEX : Scalar(Float32) shape=[]
    ├── [87] → (see above)
    └── [303] Add : Scalar(WeakInt) shape=[]
        ├── [253] → (see above)
        └── [28] CONST(Int(1))
```

`sym` put the index into the `base + const` form the next stage needs (49 nodes).

## `16-memory_coalescing`

The four loads share base `R0*4 + R1*64`, offsets 0..3, and the base is divisible by 4, so they become one 4-wide access; the lanes are `INDEX(load, k)`:

```text
[341] Add : Scalar(Float32) shape=[]
├── [340] Add : Scalar(Float32) shape=[]
│   ├── [339] Add : Scalar(Float32) shape=[]
│   │   ├── [335] INDEX : Scalar(Float32) shape=[]
│   │   │   ├── [334] LOAD : Scalar(Float32) shape=[Const(4)]
│   │   │   │   └── [333] SHRINK : Scalar(Float32) shape=[Const(4)]
│   │   │   │       ├── [87] PARAM(slot=1) : Scalar(Float32) shape=[Const(512)]
│   │   │   │       ├── [253] Add : Scalar(WeakInt) shape=[]          ← R0*4 + R1*64
│   │   │   │       └── [116] CONST(Int(4))                            ← width
│   │   │   └── [29] CONST(Int(0))
│   │   └── [336] INDEX ... [334], CONST(1)
│   └── [337] INDEX ... [334], CONST(2)
└── [338] INDEX ... [334], CONST(3)
```

44 nodes. `17-bottom_up_ew_image` and `16-extra_symbolic` are no-ops here.

## `17-pm_lower_index_dtype` and `18-final_symbolic`

Every `WeakInt` commits to `Int32` — the ranges, the constants, the `PARAM` sizes:

```text
[386] RANGE(R1, Weak) : Scalar(Int32) shape=[]
└── [351] CONST(Int(8)) : Scalar(Int32) shape=[]
[377] RANGE(R0, Reduce) : Scalar(Int32) shape=[]
└── [373] CONST(Int(16)) : Scalar(Int32) shape=[]
[391] Add : Scalar(Int32) shape=[]
├── [390] Mul : Scalar(Int32) shape=[]
│   ├── [377] → (see above)
│   └── [366] CONST(Int(4)) : Scalar(Int32) shape=[]
└── [389] Mul : Scalar(Int32) shape=[]
    ├── [386] → (see above)
    └── [381] CONST(Int(64)) : Scalar(Int32) shape=[]
```

`18-final_symbolic`, `19-cast_float_alu`, `19b` and `19c` change nothing: no transcendental, no emulated dtype.

## `19d-late_decompositions` to `20-final_rewrite`

The late rewrites turn `R1 * 64` into `R1 << 6` (`pm_mul_to_shl`), `R0 * 4` into `R0 << 2`, and `(R0 << 2) + (R1 << 6)` into an integer `MulAcc` (`pm_shl_add_to_mulacc`). Gate movement has nothing to move (no `Invalid` anywhere), and the final rewrite's `pm_split_ends` has nothing to split (every `END` already closes one range). The final graph, 43 nodes:

```text
[455] SINK[KERNEL] : Scalar(Void)
└── [454] END : Scalar(Void) shape=[]
    ├── [453] STORE : Scalar(Void) shape=[]
    │   ├── [428] INDEX : Scalar(Float32) shape=[]
    │   │   ├── [353] PARAM(slot=0) : Scalar(Float32) shape=[Const(8)]
    │   │   │   └── [351] CONST(Int(8)) : Scalar(Int32) shape=[]
    │   │   └── [386] RANGE(R1, Weak) : Scalar(Int32) shape=[]
    │   │       └── [351] → (see above)
    │   └── [452] LOAD : Scalar(Float32) shape=[]
    │       └── [451] INDEX : Scalar(Float32) shape=[]
    │           ├── [450] AFTER : Scalar(Float32) shape=[Const(1)]
    │           │   ├── [356] BUFFER(slot=0, addrspace=Some(Reg)) : Scalar(Float32) shape=[Const(1)]
    │           │   │   └── [354] CONST(Int(1)) : Scalar(Int32) shape=[]
    │           │   └── [449] END : Scalar(Void) shape=[]
    │           │       ├── [448] STORE : Scalar(Void) shape=[]
    │           │       │   ├── [363] INDEX : Scalar(Float32) shape=[]
    │           │       │   │   ├── [356] → (see above)
    │           │       │   │   └── [361] CONST(Int(0)) : Scalar(Int32) shape=[]
    │           │       │   └── [447] Add : Scalar(Float32) shape=[]
    │           │       │       ├── [418] LOAD : Scalar(Float32) shape=[]
    │           │       │       │   └── [417] INDEX : Scalar(Float32) shape=[]
    │           │       │       │       ├── [415] AFTER : Scalar(Float32) shape=[Const(1)]
    │           │       │       │       │   ├── [356] → (see above)
    │           │       │       │       │   ├── [413] STORE : Scalar(Void) shape=[]
    │           │       │       │       │   │   ├── [412] INDEX : Scalar(Float32) shape=[]
    │           │       │       │       │   │   │   ├── [410] AFTER : Scalar(Float32) shape=[Const(1)]
    │           │       │       │       │   │   │   │   ├── [356] → (see above)
    │           │       │       │       │   │   │   │   └── [386] → (see above)
    │           │       │       │       │   │   │   └── [361] → (see above)
    │           │       │       │       │   │   └── [166] CONST(Float(0.0)) : Scalar(Float32) shape=[]
    │           │       │       │       │   └── [377] RANGE(R0, Reduce) : Scalar(Int32) shape=[]
    │           │       │       │       │       └── [373] CONST(Int(16)) : Scalar(Int32) shape=[]
    │           │       │       │       └── [361] → (see above)
    │           │       │       └── [446] Add : Scalar(Float32) shape=[]
    │           │       │           ├── [445] Add : Scalar(Float32) shape=[]
    │           │       │           │   ├── [444] Add : Scalar(Float32) shape=[]
    │           │       │           │   │   ├── [443] INDEX : Scalar(Float32) shape=[]
    │           │       │           │   │   │   ├── [439] LOAD : Scalar(Float32) shape=[Const(4)]
    │           │       │           │   │   │   │   └── [438] SHRINK : Scalar(Float32) shape=[Const(4)]
    │           │       │           │   │   │   │       ├── [359] PARAM(slot=1) : Scalar(Float32) shape=[Const(512)]
    │           │       │           │   │   │   │       │   └── [357] CONST(Int(512)) : Scalar(Int32) shape=[]
    │           │       │           │   │   │   │       ├── [437] MulAcc : Scalar(Int32) shape=[]
    │           │       │           │   │   │   │       │   ├── [377] → (see above)
    │           │       │           │   │   │   │       │   ├── [366] CONST(Int(4)) : Scalar(Int32) shape=[]
    │           │       │           │   │   │   │       │   └── [435] Shl : Scalar(Int32) shape=[]
    │           │       │           │   │   │   │       │       ├── [386] → (see above)
    │           │       │           │   │   │   │       │       └── [434] CONST(Int(6)) : Scalar(Int32) shape=[]
    │           │       │           │   │   │   │       └── [366] → (see above)
    │           │       │           │   │   │   └── [361] → (see above)
    │           │       │           │   │   └── [442] INDEX : Scalar(Float32) shape=[]
    │           │       │           │   │       ├── [439] → (see above)
    │           │       │           │   │       └── [354] → (see above)
    │           │       │           │   └── [441] INDEX : Scalar(Float32) shape=[]
    │           │       │           │       ├── [439] → (see above)
    │           │       │           │       └── [399] CONST(Int(2)) : Scalar(Int32) shape=[]
    │           │       │           └── [440] INDEX : Scalar(Float32) shape=[]
    │           │       │               ├── [439] → (see above)
    │           │       │               └── [395] CONST(Int(3)) : Scalar(Int32) shape=[]
    │           │       └── [377] → (see above)
    │           └── [361] → (see above)
    └── [386] → (see above)
```

Reading it bottom-up: `BUFFER(Reg)` is the accumulator; `STORE([412], 0.0)` zeroes it after `AFTER(acc, R1)`, i.e. once per row; the loop body `STORE([363], LOAD(acc) + lanes)` is closed by `END(.., R0)`; the final `LOAD` reads the accumulator after that `END` and is stored to the output at `R1`; the outer `END` closes `R1`.

## Linearize and render

`linearize` emits 43 instructions in `(run_count, priority, slot, tuplize)` order: the two `PARAM`s (each preceded by its size constant), the register `BUFFER` and its `INDEX` first (`run_count` 1, priorities −20/−18), then `RANGE(R1)`, the zero store, `RANGE(R0)`, the body, `END(R0)`, the output store, `END(R1)`, `SINK`. The CPU renderer turns that list into:

```llvm
define void @r_8_16_4(ptr noalias align 32 %data0, ptr noalias align 32 %data1) #0 {
entry:
  %reg0 = alloca [1 x float]
  %v1 = getelementptr inbounds float, ptr %reg0, i32 0
  br label %loop_entry_1
loop_entry_1:
  br label %loop_latch_1
loop_latch_1:
  %r1 = phi i32 [ 0, %loop_entry_1 ], [ %r1phi, %loop_footer_1 ]
  %r1phi = add i32 %r1, 1
  %r1cmp = icmp ult i32 %r1, 8
  br i1 %r1cmp, label %loop_body_1, label %loop_exit_1
loop_body_1:
  %v3 = getelementptr inbounds float, ptr %reg0, i32 0
  %v4 = shl i32 %r1, 6
  store float 0x0000000000000000, ptr %v3
  br label %loop_entry_0
loop_entry_0:
  br label %loop_latch_0
loop_latch_0:
  %r0 = phi i32 [ 0, %loop_entry_0 ], [ %r0phi, %loop_footer_0 ]
  %r0phi = add i32 %r0, 1
  %r0cmp = icmp ult i32 %r0, 16
  br i1 %r0cmp, label %loop_body_0, label %loop_exit_0
loop_body_0:
  %v7 = getelementptr inbounds float, ptr %reg0, i32 0
  %v8 = load float, ptr %v7
  %v9.mul = mul i32 %r0, 4
  %v9 = add i32 %v9.mul, %v4
  %v10 = getelementptr inbounds float, ptr %data1, i32 %v9
  %v11 = load <4 x float>, ptr %v10
  %v12 = extractelement <4 x float> %v11, i32 0
  %v13 = extractelement <4 x float> %v11, i32 1
  %v14 = extractelement <4 x float> %v11, i32 2
  %v15 = extractelement <4 x float> %v11, i32 3
  %v16 = fadd nsz arcp contract afn float %v12, %v13
  %v17 = fadd nsz arcp contract afn float %v16, %v14
  %v18 = fadd nsz arcp contract afn float %v17, %v15
  %v19 = fadd nsz arcp contract afn float %v8, %v18
  store float %v19, ptr %v1
  br label %loop_footer_0
loop_footer_0:
  br label %loop_latch_0
loop_exit_0:
  %v23 = getelementptr inbounds float, ptr %reg0, i32 0
  %v24 = load float, ptr %v23
  %v25 = getelementptr inbounds float, ptr %data0, i32 %r1
  store float %v24, ptr %v25
  br label %loop_footer_1
loop_footer_1:
  br label %loop_latch_1
loop_exit_1:
  ret void
}
```

The `SHRINK` of width 4 became `load <4 x float>`, the integer `MulAcc` became `mul` + `add`, the register buffer an `alloca`; LLVM's own optimizer later keeps the accumulator in a register. The result is `[2016, 6112, 10208, 14304, 18400, 22496, 26592, 30688]`.

## Reading a dump

| Symptom | Stages to look at first |
|---------|-------------------------|
| wrong values | `08` (symbolic), `09` (expansion), `10` (accumulator init/identity), `19d` (decompositions) |
| wrong loop count or missing loop | pre-opt split/simplify ranges, `12` (gpudims), `10` (`END` merging) |
| scalar loads where vector loads were expected | `15`/`16`: the index must be `base + const` with a divisible base, same buffer, same validity, no gate |
| `WeakInt` in the final graph | `17-pm_lower_index_dtype` (`SVOD_SPEC` catches it at `18`) |
| `Invalid` in the final graph | `19e` gate movement, `20` `pm_remove_invalid` (debug assertion) |
| a backend rejects an op | `19b`/`19d` capability table (`supported_ops`) |

`node_count` per stage is the cheapest signal: a stage that doubles the count on a small kernel is the one to dump.
