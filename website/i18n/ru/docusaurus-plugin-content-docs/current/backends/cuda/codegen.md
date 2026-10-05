---
sidebar_label: Кодген
---

# Кодген: таргет NVPTX

CUDA-бэкенд переиспользует текстовый LLVM-рендерер с третьим таргетом —
`LlvmTarget::Nvptx(CudaArch)` (`LlvmTextRenderer::nvptx(arch)`). Как и
AMD-эмиттер, `codegen/src/llvm/nvptx/` надстраивается над CPU-эмиттером: он
перехватывает операции, обобщённую LLVM-форму которых NVPTX-бэкенд не может
выбрать (`Special`, `Barrier`, LOCAL-буферы, `Log2`, `Wmma`, fp8-касты), и
пропускает всё остальное (ALU, INDEX, LOAD, STORE, CAST, RANGE) без изменений.
Таблица опускания проверена на clang 22 и `ptxas` 13.3 при `sm_86`.

---

## Форма модуля

```llvm
; ModuleID = 'r_64_32'
source_filename = "r_64_32"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()

define ptx_kernel void @r_64_32(ptr noalias align 32 %data0, ptr noalias align 32 %data1) #0 {
entry:
  ...
  ret void
}

attributes #0 = { nounwind "no-builtins" "no-trapping-math"="true" "nvvm.maxntid"="32" }
```

- Одного `ptx_kernel` достаточно, чтобы получить `.visible .entry`; никакие
  `!nvvm.annotations` не нужны.
- `target datalayout` — это дефолт clang 22 для `nvptx64`; clang молча
  перезаписывает несовпадение, так что строка существует ради инструментов,
  которые читают модуль отдельно (`opt`, `llvm-as`, дампы IR).
- `"nvvm.maxntid"` — это PTX-`.maxntid`, **launch bound**, по одной границе на
  ось: локальные размеры ядра выводятся как `nx[, ny[, nz]]` (`"16,8"` даёт
  `.maxntid 16, 8`; замыкающие оси, равные 1, отбрасываются), так что `ptxas`
  считает бюджет регистров на поток относительно него, а не относительно
  худшего случая в 1024 потока. Граница — это `vmax` каждой протяжённости, так
  что символьная протяжённость с целочисленной верхней границей всё равно её
  получает; протяжённость без целочисленной границы снимает атрибут, и
  действует аппаратный максимум. Более старый LLVM игнорирует строковый
  атрибут и просто теряет подсказку.

| Понятие | AMD | NVPTX |
|---|---|---|
| триплет | `amdgcn-amd-amdhsa` | `nvptx64-nvidia-cuda` |
| ABI ядра | `amdgpu_kernel`, дескриптор `.kd` | `ptx_kernel` |
| идентификаторы работы | `llvm.amdgcn.workgroup.id.*` / `workitem.id.*` | `llvm.nvvm.read.ptx.sreg.ctaid.{x,y,z}` / `tid.{x,y,z}` |
| барьер | `fence syncscope("workgroup")` + `s.barrier` | `fence syncscope("block") release; llvm.nvvm.barrier0; fence syncscope("block") acquire` (`bar.sync 0`) |
| адресные пространства | global 1, LDS 3, private 5 | shared 3, global 1 в кастах `cp.async` / `ldmatrix`; параметры ядра — generic-указатели, а REG-буферы остаются обычной `alloca` |
| разделяемая память | глобалы модуля в `addrspace(3)` | то же |
| launch bound | `"amdgpu-flat-work-group-size"` | `"nvvm.maxntid"` |

NVPTX называет скоуп рабочей группы `"block"`; `syncscope("workgroup")`
отвергается. `@llvm.nvvm.barrier0` — то написание, которое любой релиз LLVM
опускает в `bar.sync 0` (более новые автоматически его апгрейдят).

---

## Быстрая математика и деление

На GPU-таргетах рендерер урезает ` nsz arcp contract afn ` до ` contract `:
NVPTX опускает `fdiv ... arcp afn` в `rcp.approx.f32`, тогда как один только
`contract` сохраняет точный `div.rn.f32`. CUDA-фронтенд tinygrad тоже
компилирует с точным делением.

---

## Трансцендентные функции

У NVPTX **нет опускания** для обобщённых интринсиков
`@llvm.{exp,log,sin,cos,pow}` (выбор инструкций падает), а `@llvm.erf` он выдаёт
как внешний вызов, который падает только внутри `ptxas`. Поэтому обёртка
CUDA-рендерера (`runtime/src/devices/cuda.rs`) убирает `Exp`, `Log`, `Log2`,
`Sin`, `Cos`, `Tan`, `Erf`, `Pow`, `Max` и `Threefry` из своих `supported_ops`,
а планировщик декомпозирует их до рендеринга. Её `decompositor` — это
`nvptx_decomposition_patterns()`: набор AMD (полиномиальные
`exp`/`log`/`cos`/`tan`/`pow` поверх нативных `exp2`/`log2`, округление bf16 в
целочисленной области) плюс раскрытия f64 `Exp2`/`Log2`, поскольку NVPTX
опускает `@llvm.exp2` только для f16/f32. `Sin` и `Log2` для f32/f16 идут через
общие трансцендентные паттерны, выбираемые по `supported_ops`; `Erf`, `Max` и
`Threefry` переписываются собственными проходами оптимизатора (полином, select,
полное перемешивание `threefry2x32`). `Max`, `Pow` и `Threefry` убраны у каждого
GPU-рендерера, а не по выбору NVPTX.

Что остаётся нативным: `@llvm.exp2.f32` выбирает `ex2.approx.f32`, `@llvm.sqrt`
выбирает `sqrt.rn`, `fma`/`floor`/`rint`/`maxnum` опускаются напрямую.

`Log2` — намеренный случай. У `@llvm.log2.f32` нет опускания в NVPTX («no
libcall available for flog2»); аппаратный путь — это `@llvm.nvvm.lg2.approx.f` →
`lg2.approx.f32`, относительное приближение 2^-22.6, ошибка в 1 ulp там, где
AMD-шный `v_log_f32` и libm точны. Рендерер сохраняет это опускание
(`render_log2`: только скалярный f32, f16 расширяется вокруг него, векторы
разбиваются по лейнам) для явного использования, но `Log2` исключён из
`supported_ops`, так что обычные графы идут полиномиальным путём `xlog2` и
укладываются в общие тестовые допуски.

Недекомпозированная трансцендентная функция, всё же дошедшая до рендерера, — это
рассинхрон списка возможностей; она падает во время рендеринга с ошибкой
`InvalidGraph`, а не называет интринсик, который LLVM молча превратил бы во
внешний вызов.

В PTX булевы значения — это регистры-предикаты, но в памяти это байты, поэтому
`extra_matcher` для NVPTX — это `bool_storage_patterns` CPU-рендерера
(`ptx_matcher` из tinygrad). Профиль оптимизатора записывает
`nvptx-decomposition-v1` и `llvm-nvptx-extra-v1` в свой ключ кэша, так что эти
решения никогда не сталкиваются с ядрами другого бэкенда.

---

## Tensor cores: `mma.sync`

`Wmma` опускается в один интринсик `@llvm.nvvm.mma.*`, выбираемый функцией
`resolve_mma(arch, in_dtype, acc_dtype, (N, M, K))`. Каждый CUDA-профиль имеет
форму `m16n8kK`; PTX ISA фиксирует минимальную capability для каждой строки:

| Входы → аккумулятор | K | Суффикс интринсика | Мин. |
|---|---|---|---|
| f16 → f32 / f16 | 8 | `m16n8k8.row.col.f32.f32` / `.f16.f16` | `sm_75` |
| f16 → f32 / f16 | 16 | `m16n8k16.row.col.f32.f32` / `.f16.f16` | `sm_80` |
| bf16 → f32 | 16 | `m16n8k16.row.col.bf16` | `sm_80` |
| tf32 (сырые биты f32) → f32 | 8 | `m16n8k8.row.col.tf32` | `sm_80` |
| int8 → int32 | 32 | `m16n8k32.row.col.satfinite.s8` | `sm_80` |
| e4m3 / e5m2 → f32 | 32 | `m16n8k32.row.col.f32.e4m3.e4m3.f32` / `.e5m2...` | `sm_89` |

Любой другой кортеж, как и арка ниже минимума, возвращает `None`, а вызывающий
поднимает `InvalidGraph`, чтобы оптимизатор декомпозировал операцию выше по
цепочке. Фрагменты следуют разбиению регистров PTX (A — 16×K, B — K×8, C/D —
16×8, всё это по 32 лейнам в 32-битных регистрах): операнды и аккумуляторы
f16 передаются парами `<2 x half>`, операнды bf16 / tf32 / int8 / fp8 и
аккумуляторы i32 — словами `i32`, аккумуляторы f32 — как `float`; агрегатный результат пересобирается в естественный вектор
WMMA. Соответствующие строки `declare` синтезируются из типов операндов каждого
места вызова (`wmma_declaration_from_call`) — тот же механизм, что и у
интринсиков AMD WMMA/MFMA.

Набор дополняют ещё два семейства типизированных узлов `CUSTOM`. Варп-билдеры —
это `shfl` с четырьмя режимами: `shfl_bfly(value, lane_mask)`
(`llvm.nvvm.shfl.sync.bfly.i32`, шаг «бабочки» варп-редукции), `shfl_idx`,
`shfl_up` и `shfl_down`, — плюс `globaltimer()`
(`llvm.nvvm.read.ptx.sreg.globaltimer`, наносекундные часы GPU).

Билдеры разделяемой памяти (`codegen/src/llvm/nvptx/smem.rs`) — это то, что
позволяет tile-ядру гонять данные через `.shared` так же, как на AMD:
`ldmatrix` (sm_75+) загружает фрагмент матрицы сразу в регистровой раскладке,
которую ждёт `mma.sync`, а `cp_async` / `cp_async_16` (sm_80+) копируют
global → shared, минуя регистры; фиксация и ожидание — через
`cp_async_commit`, `cp_async_wait` и `cp_async_wait_all`. `CpAsyncCache` — это
политика кэша этого копирования: `.cg` (только L2, 16 байт) или `.ca` (L1 и L2,
4, 8 или 16). Каждый билдер несёт собственный `declare`; `dedup_declares`
оставляет первый на каждое имя функции, так что ядро, полное `cp.async`,
объявляет интринсик один раз.

---

## Компиляция в PTX

`compile_ir_to_ptx` (`runtime/src/cuda/compile.rs`) прогоняет IR через хостовый
clang, со stdin на stdout, ровно так же, как пути AMD и CPU:

```text
clang -x ir -S -O3 --target=nvptx64-nvidia-cuda -march=sm_86 --cuda-feature=+ptx78 -Wno-override-module - -o -
```

Кэшированная проверка `clang --print-targets` превращает clang без NVPTX в
аккуратную ошибку `JitCompilation`. `SVOD_DUMP_NVPTX_IR=<dir>` записывает туда
IR каждого ядра как `sm_XY_<module>.ll`.

Прежде чем любой PTX дойдёт до драйвера — свежий или из кэша объектов —
`validate_ptx` проверяет, что в нём есть `.version`, `.target`, равный `sm_XY`
устройства, `.entry <kernel>(` и **нет `.extern .func`**. Последнее важно:
опечатка в имени `llvm.nvvm.*` не является ошибкой компиляции, LLVM молча
выдаёт её как внешний вызов, и иначе она всплыла бы лишь как сбой
`cuModuleLoadDataEx`.

Когда `ptxas` установлен, PTX проверяется здесь же, на этапе компиляции, —
включая его список `.param` относительно ABI, которого cubin уже не несёт, — и в
кэш попадает именно собранный cubin; тогда попадание проходит через
`validate_cubin` (little-endian ELF64 для `EM_CUDA`, определяющий точку входа
как код).

Версия PTX ISA фиксируется по архитектуре, а не оставляется clang, чей вариант
по умолчанию зависит от найденного CUDA toolkit (clang 22 с CUDA 13:
`.version 8.8`, для которого нужен драйвер CUDA 12.9; без toolkit — версия,
слишком старая для любых тензорных ядер). `ptx_isa`
(`codegen/src/llvm/nvptx/mod.rs`) отображает вычислительную способность на
самую старую ISA, которая знает этот чип: `+ptx78` вплоть до sm_88, `+ptx84` от
sm_89 и по всем 9.x (fp8-формы `mma.sync` существуют с 8.4), `+ptx86` на sm_100
— sm_102, `+ptx87` на sm_120 и `+ptx88` на всех остальных чипах 10.x и новее
(sm_103, sm_110, sm_121, ...). Более старую clang отвергает:
`PTX version 8.4 does not support target 'sm_120'. Minimum required PTX version is 8.7`.
