---
sidebar_label: Разобранный пример
---

# Разобранный пример: сумма по строкам на CPU

Каждое дерево на этой странице — реальный вывод. Программа:

```rust
let data = Array2::from_shape_fn((8, 64), |(r, c)| (r * 64 + c) as f32);
let x = Tensor::from_ndarray(&data);
let y = x.sum(1)?;
y.realize()?;
```

снята с `SVOD_PER_STAGE_UOPS=1 SVOD_DUMP_STAGE=` (каждая post-opt стадия) и `RUST_LOG=svod_schedule::rangeify::transforms=debug,svod_schedule::optimizer=debug` под JSON-подписчиком для более ранних проходов, на CPU-бэкенде по умолчанию (LLVM, в процессе). Id узлов — это порядок аллокации, и между запусками они будут отличаться; структура — нет.

## Тензорный граф

`sum(1)` — это `REDUCE` в тензорной форме над `PERMUTE` от `RESHAPE` плоского буфера из 512 элементов, обёрнутый в `CONTIGUOUS`, потому что результат является выходом:

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

`STACK` — это полезная нагрузка формы для reshape: формы тоже являются UOp.

## После rangeify

Назначение диапазонов даёт выходу диапазон `Weak` `U0` (8), а редукции — диапазон `Reduce` `U1` (64); операции перемещения схлопываются в индекс `U0 * 64 + U1` (дерево — на [странице rangeify](./rangeify.md)). Разрез на ядра превращает `STAGE` в `STORE`/`END`, нумерует буферы как `PARAM` и перенумеровывает диапазоны. Тело ядра, которое попадает в `apply_pre_optimization`:

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

Слот 0 — выход (буфер `STAGE` был отображён первым), слот 1 — вход. Пять шагов предоптимизации оставляют этот граф нетронутым: нет операций перемещения, нет схлопываемых редукций, нет остатков для разбиения, нечего сливать.

## После оптимизатора (`00-initial`)

У CPU-рендерера нет локальных измерений, поэтому `convert_loop_to_global` оставляет `R1` как `Weak`. `hand_coded_optimizations` пропускает тензорные ядра, upcast изображений, путь matvec и групповые редукции; `apply_unroll` видит редукцию шириной 64 (больше 32) и применяет `UNROLL(0, 4)`; ядро уже развёрнуто, поэтому `apply_default_upcast` ничего не делает; 512 элементов намного меньше порога 131072 на поток, поэтому `THREAD` нет. Ядро называется `r_8_16_4` (reduce; протяжённости 8, 16, 4):

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

`apply_opt` разбил `R0` шириной 64 на `R0 * 4 + R2` с `R0: Reduce(16)` и `R2: Unroll(4)`, а `pm_flatten_range` перечислил оба на `REDUCE`. Число узлов — 20.

## `08-post_opt_sym`

Срабатывает только `commutative_canonicalization`: индекс становится `(R0*4 + R2) + R1*64` (порядок операндов по tuplize). По-прежнему 20 узлов.

## `09-pre_expand`

`R2` заменяется на `RESHAPE(STACK(0,1,2,3), [4])`, каждый потребитель получает форму, а `expand_reduce` переносит ось линий в `num_axes`:

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

`RESHAPE` в `[1]` — заглушка, которую `expand_reduce` оставляет для редуцированной оси линий; девекторизатор её убирает.

## `10-pm_reduce`

`reduce_to_acc` строит аккумулятор. `horizontal_reduce` сначала сворачивает четыре линии (`((a0 + a1) + a2) + a3`, каждая линия — `INDEX` в индексное выражение с формой), затем цикл по `R0` накапливает результат в регистровом буфере:

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

(Сокращено: четыре линии идентичны, кроме константного индекса линии.) Обратите внимание на `AFTER(acc, [R1])` у инициализирующей записи: `input_ranges` помещает обнуление внутрь цикла по строкам. Стадии `11` и `12` ничего не меняют — нет локальной стадии, нет GPU-диапазонов.

## `13-pm_add_loads` и `14-devectorize`

`pm_expand_broadcast` делает скалярные слагаемые индекса с формой явными (`EXPAND(RESHAPE(R0*4, [1]), [4])`, то же для `R1*64`), а `pm_add_loads` оборачивает в `LOAD` чтения регистра и четыре входные линии (55 узлов). Затем `devectorize` скаляризует всё: индекс с формой схлопывается в четыре скалярных `Add`, а записи по линиям группируются. После `15-early_symbolic` линии читаются как `LOAD(INDEX(PARAM(1), (R0*4 + R1*64) + k))`:

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

`sym` привёл индекс к форме `base + const`, нужной следующей стадии (49 узлов).

## `16-memory_coalescing`

Четыре загрузки имеют общую базу `R0*4 + R1*64`, смещения 0..3, и база делится на 4, поэтому они становятся одним доступом шириной 4; линии — это `INDEX(load, k)`:

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

44 узла. `17-bottom_up_ew_image` и `16-extra_symbolic` здесь ничего не делают.

## `17-pm_lower_index_dtype` и `18-final_symbolic`

Каждый `WeakInt` фиксируется в `Int32` — диапазоны, константы, размеры `PARAM`:

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

`18-final_symbolic`, `19-cast_float_alu`, `19b` и `19c` ничего не меняют: нет трансцендентных функций, нет эмулируемых dtype.

## От `19d-late_decompositions` до `20-final_rewrite`

Поздние перезаписи превращают `R1 * 64` в `R1 << 6` (`pm_mul_to_shl`), `R0 * 4` в `R0 << 2`, а `(R0 << 2) + (R1 << 6)` — в целочисленный `MulAcc` (`pm_shl_add_to_mulacc`). Перемещению гейтов нечего переносить (нигде нет `Invalid`), а `pm_split_ends` финальной перезаписи нечего разбивать (каждый `END` уже закрывает один диапазон). Итоговый граф, 43 узла:

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

Чтение снизу вверх: `BUFFER(Reg)` — аккумулятор; `STORE([412], 0.0)` обнуляет его после `AFTER(acc, R1)`, то есть один раз на строку; тело цикла `STORE([363], LOAD(acc) + lanes)` закрывается `END(.., R0)`; финальный `LOAD` читает аккумулятор после этого `END` и записывается в выход по `R1`; внешний `END` закрывает `R1`.

## Линеаризация и рендеринг

`linearize` выдаёт 43 инструкции в порядке `(run_count, priority, slot, tuplize)`: два `PARAM` (каждому предшествует константа его размера), регистровый `BUFFER` и его `INDEX` первыми (`run_count` 1, приоритеты −20/−18), затем `RANGE(R1)`, обнуляющая запись, `RANGE(R0)`, тело, `END(R0)`, запись в выход, `END(R1)`, `SINK`. CPU-рендерер превращает этот список в:

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

`SHRINK` ширины 4 стал `load <4 x float>`, целочисленный `MulAcc` — `mul` + `add`, регистровый буфер — `alloca`; собственный оптимизатор LLVM затем держит аккумулятор в регистре. Результат — `[2016, 6112, 10208, 14304, 18400, 22496, 26592, 30688]`.

## Чтение дампа

| Симптом | Какие стадии смотреть первыми |
|---------|-------------------------|
| неверные значения | `08` (символьное упрощение), `09` (развёртывание), `10` (инициализация/нейтральный элемент аккумулятора), `19d` (декомпозиции) |
| неверное число итераций или пропавший цикл | pre-opt разбиение/упрощение диапазонов, `12` (gpudims), `10` (слияние `END`) |
| скалярные загрузки там, где ожидались векторные | `15`/`16`: индекс должен быть `base + const` с делящейся базой, тот же буфер, та же валидность, без гейта |
| `WeakInt` в итоговом графе | `17-pm_lower_index_dtype` (`SVOD_SPEC` ловит это на `18`) |
| `Invalid` в итоговом графе | перемещение гейтов на `19e`, `pm_remove_invalid` на `20` (отладочная проверка) |
| бэкенд отвергает операцию | таблица возможностей на `19b`/`19d` (`supported_ops`) |

`node_count` по стадиям — самый дешёвый сигнал: стадию, которая удваивает число узлов на маленьком ядре, и нужно дампить.
