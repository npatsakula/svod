---
sidebar_label: 完整示例
---

# 完整示例：CPU 上的行求和

本页的每一棵树都是真实输出。程序如下：

```rust
let data = Array2::from_shape_fn((8, 64), |(r, c)| (r * 64 + c) as f32);
let x = Tensor::from_ndarray(&data);
let y = x.sum(1)?;
y.realize()?;
```

在默认 CPU 后端（LLVM，进程内）上，用 `SVOD_PER_STAGE_UOPS=1 SVOD_DUMP_STAGE=`（每个 post-opt 阶段）以及 JSON subscriber 下的 `RUST_LOG=svod_schedule::rangeify::transforms=debug,svod_schedule::optimizer=debug`（更早的 pass）捕获。节点 id 是分配顺序，每次运行会有所不同；结构则不会变。

## 张量图

`sum(1)` 是一个张量形式的 `REDUCE`，作用在对 512 元素扁平缓冲区做 `RESHAPE` 再 `PERMUTE` 的结果上；由于结果是输出，它被包在 `CONTIGUOUS` 中：

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

`STACK` 是 reshape 的形状载荷 —— 形状本身也是 UOp。

## Rangeify 之后

range 分配为输出赋予一个 `Weak` range `U0`（8），为归约赋予一个 `Reduce` range `U1`（64）；movement 算子坍缩为索引 `U0 * 64 + U1`（树见 [Rangeify 页面](./rangeify.md)）。内核切分把 `STAGE` 变成 `STORE`/`END`，把缓冲区编号为 `PARAM`，并重新编号 range。进入 `apply_pre_optimization` 的内核体：

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

slot 0 是输出（`STAGE` 的缓冲区先被映射），slot 1 是输入。五个预优化步骤不会改动这张图：没有 movement 算子，没有可折叠的 reduce，没有可拆分的取模，没有可合并的内容。

## 优化器之后（`00-initial`）

CPU renderer 没有局部维度，因此 `convert_loop_to_global` 让 `R1` 保持为 `Weak`。`hand_coded_optimizations` 跳过 tensor core、图像 upcast、matvec 路径和分组归约；`apply_unroll` 看到 64 宽的 reduce（超过 32），于是应用 `UNROLL(0, 4)`；内核已被展开，因此 `apply_default_upcast` 什么也不做；512 个元素远低于每线程 131072 的阈值，因此没有 `THREAD`。内核被命名为 `r_8_16_4`（reduce；extent 为 8、16、4）：

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

`apply_opt` 把 64 宽的 `R0` 拆成 `R0 * 4 + R2`，其中 `R0: Reduce(16)`、`R2: Unroll(4)`，`pm_flatten_range` 把两者都列在 `REDUCE` 上。节点数 20。

## `08-post_opt_sym`

只有 `commutative_canonicalization` 生效：索引变为 `(R0*4 + R2) + R1*64`（操作数的 tuplize 顺序）。仍为 20 个节点。

## `09-pre_expand`

`R2` 被替换为 `RESHAPE(STACK(0,1,2,3), [4])`，其每个消费者都变成带形状的，`expand_reduce` 把 lane 轴移入 `num_axes`：

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

reshape 为 `[1]` 的 `RESHAPE` 是 `expand_reduce` 为已归约 lane 轴留下的占位；devectorizer 会移除它。

## `10-pm_reduce`

`reduce_to_acc` 构建累加器。`horizontal_reduce` 先折叠四个 lane（`((a0 + a1) + a2) + a3`，每个 lane 是对带形状索引表达式的一次 `INDEX`），然后在 `R0` 上的循环累加到一个寄存器缓冲区中：

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

（已裁剪：四个 lane 除常量 lane 索引外完全相同。）注意初始化 store 上的 `AFTER(acc, [R1])`：`input_ranges` 把清零放在行循环内部。阶段 `11` 和 `12` 不做任何改动 —— 没有 local stage，也没有 GPU range。

## `13-pm_add_loads` 与 `14-devectorize`

`pm_expand_broadcast` 把带形状索引中的标量项显式化（`EXPAND(RESHAPE(R0*4, [1]), [4])`，`R1*64` 同理），`pm_add_loads` 把寄存器读取和四个输入 lane 包进 `LOAD`（55 个节点）。随后 `devectorize` 把一切标量化：带形状的索引坍缩为四个标量 `Add`，逐 lane 的 `STORE` 被分组。`15-early_symbolic` 之后，各 lane 读作 `LOAD(INDEX(PARAM(1), (R0*4 + R1*64) + k))`：

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

`sym` 把索引整理成下一阶段需要的 `base + const` 形式（49 个节点）。

## `16-memory_coalescing`

四个 load 共享 base `R0*4 + R1*64`，偏移为 0..3，且 base 能被 4 整除，因此它们变成一次 4 宽访问；各 lane 为 `INDEX(load, k)`：

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

44 个节点。`17-bottom_up_ew_image` 和 `16-extra_symbolic` 在这里是空操作。

## `17-pm_lower_index_dtype` 与 `18-final_symbolic`

每个 `WeakInt` 都被确定为 `Int32` —— range、常量、`PARAM` 的大小：

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

`18-final_symbolic`、`19-cast_float_alu`、`19b` 和 `19c` 不做任何改动：没有超越函数，也没有需要模拟的 dtype。

## 从 `19d-late_decompositions` 到 `20-final_rewrite`

后期重写把 `R1 * 64` 变成 `R1 << 6`（`pm_mul_to_shl`），把 `R0 * 4` 变成 `R0 << 2`，再把 `(R0 << 2) + (R1 << 6)` 变成整数 `MulAcc`（`pm_shl_add_to_mulacc`）。gate 移动无事可做（任何地方都没有 `Invalid`），最终重写中的 `pm_split_ends` 也无可拆分（每个 `END` 已经只关闭一个 range）。最终的图，43 个节点：

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

自底向上阅读：`BUFFER(Reg)` 是累加器；`STORE([412], 0.0)` 在 `AFTER(acc, R1)` 之后将其清零，即每行一次；循环体 `STORE([363], LOAD(acc) + lanes)` 由 `END(.., R0)` 关闭；最后的 `LOAD` 在该 `END` 之后读取累加器，并在 `R1` 处存入输出；外层 `END` 关闭 `R1`。

## 线性化与渲染

`linearize` 按 `(run_count, priority, slot, tuplize)` 顺序输出 43 条指令：首先是两个 `PARAM`（各自前面是其大小常量）、寄存器 `BUFFER` 及其 `INDEX`（`run_count` 为 1，优先级 −20/−18），然后是 `RANGE(R1)`、清零 store、`RANGE(R0)`、循环体、`END(R0)`、输出 store、`END(R1)`、`SINK`。CPU renderer 把该列表变成：

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

宽度为 4 的 `SHRINK` 变成了 `load <4 x float>`，整数 `MulAcc` 变成 `mul` + `add`，寄存器缓冲区变成 `alloca`；LLVM 自身的优化器随后会把累加器保留在寄存器中。结果为 `[2016, 6112, 10208, 14304, 18400, 22496, 26592, 30688]`。

## 阅读 dump

| 症状 | 优先查看的阶段 |
|---------|-------------------------|
| 数值错误 | `08`（符号化简）、`09`（展开）、`10`（累加器初始化/单位元）、`19d`（分解） |
| 循环次数错误或缺少循环 | 预优化中的拆分/化简 range、`12`（gpudims）、`10`（`END` 合并） |
| 预期向量 load 却得到标量 load | `15`/`16`：索引必须是 `base + const` 形式，base 可整除，同一缓冲区，相同有效性，没有 gate |
| 最终图中出现 `WeakInt` | `17-pm_lower_index_dtype`（`SVOD_SPEC` 会在 `18` 捕获） |
| 最终图中出现 `Invalid` | `19e` gate 移动、`20` `pm_remove_invalid`（调试断言） |
| 后端拒绝某个算子 | `19b`/`19d` 能力表（`supported_ops`） |

每个阶段的 `node_count` 是最廉价的信号：在小内核上让节点数翻倍的阶段就是该 dump 的那个。
