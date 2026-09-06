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
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()

define ptx_kernel void @r_64_32(ptr addrspace(1) %data0, ptr addrspace(1) %data1) #0 {
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
- `"nvvm.maxntid"` — это PTX-`.maxntid`, **launch bound**, выставленный в
  наибольший локальный размер ядра: `ptxas` считает бюджет регистров на поток
  относительно него, а не относительно худшего случая в 1024 потока. Более
  старый LLVM игнорирует строковый атрибут и просто теряет подсказку.

| Понятие | AMD | NVPTX |
|---|---|---|
| триплет | `amdgcn-amd-amdhsa` | `nvptx64-nvidia-cuda` |
| ABI ядра | `amdgpu_kernel`, дескриптор `.kd` | `ptx_kernel` |
| идентификаторы работы | `llvm.amdgcn.workgroup.id.*` / `workitem.id.*` | `llvm.nvvm.read.ptx.sreg.ctaid.{x,y,z}` / `tid.{x,y,z}` |
| барьер | `fence syncscope("workgroup")` + `s.barrier` | `fence syncscope("block") release; llvm.nvvm.barrier0; fence syncscope("block") acquire` (`bar.sync 0`) |
| адресные пространства | global 1, LDS 3, private 5 | global 1, shared 3; REG-буферы остаются обычной generic-`alloca` |
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
как внешний вызов, который падает только внутри `ptxas`. Поэтому рендерер
убирает `Exp`, `Log`, `Log2`, `Sin`, `Cos`, `Tan`, `Erf`, `Pow`, `Max` и
`Threefry` из своих `supported_ops`, а планировщик декомпозирует их через
`nvptx_decomposition_patterns()`: набор AMD (полиномиальные `exp`/`log`/триг
поверх нативных `exp2`/`log2`, округление bf16 в целочисленной области) плюс
раскрытия f64 `Exp2`/`Log2`, поскольку NVPTX опускает `@llvm.exp2` только для
f16/f32.

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
16×8, всё это по 32 лейнам в 32-битных регистрах): операнды f16 передаются
парами `<2 x half>`, bf16 / tf32 / int8 / fp8 — словами `i32`, аккумуляторы
f32 — как `float`; агрегатный результат пересобирается в естественный вектор
WMMA. Соответствующие строки `declare` синтезируются из типов операндов каждого
места вызова (`wmma_declaration_from_call`) — тот же механизм, что и у
интринсиков AMD WMMA/MFMA.

Набор дополняют два билдера уровня варпа для типизированных узлов `CUSTOM`:
`shfl_bfly(value, lane_mask)` (`llvm.nvvm.shfl.sync.bfly.i32`, шаг «бабочки»
варп-редукции) и `globaltimer()`
(`llvm.nvvm.read.ptx.sreg.globaltimer`, наносекундные часы GPU).

---

## Компиляция в PTX

`compile_ir_to_ptx` (`runtime/src/cuda/compile.rs`) прогоняет IR через хостовый
clang, со stdin на stdout, ровно так же, как пути AMD и CPU:

```text
clang -x ir -S -O3 --target=nvptx64-nvidia-cuda -march=sm_86 -Wno-override-module - -o -
```

Кэшированная проверка `has_nvptx_target()` (`clang --print-targets`) превращает
clang без NVPTX в аккуратную ошибку `JitCompilation`.
`SVOD_DUMP_NVPTX_IR=<dir>` записывает туда `.ll` каждого ядра.

Прежде чем любой PTX дойдёт до драйвера — свежий или из кэша объектов —
`validate_ptx` проверяет, что в нём есть `.version`, `.target`, равный `sm_XY`
устройства, `.entry <kernel>(` и **нет `.extern .func`**. Последнее важно:
опечатка в имени `llvm.nvvm.*` не является ошибкой компиляции, LLVM молча
выдаёт её как внешний вызов, и иначе она всплыла бы лишь как сбой
`cuModuleLoadDataEx`.

Версия PTX ISA зафиксирована на 7.8 (`--cuda-feature=+ptx78`), а не та, которую
выдал бы релиз clang (clang 22: `.version 8.8`, для которого нужен драйвер
CUDA 12.8 / R570); каждая форма `mma.sync`, которую выбирает рендерер, есть в 7.8.
