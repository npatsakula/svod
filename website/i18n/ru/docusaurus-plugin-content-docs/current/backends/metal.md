---
sidebar_label: Metal
---

# Metal-бэкенд

Svod работает на GPU Apple через Metal. Бэкенд написан напрямую поверх
рантайма Objective-C: `libobjc`, `Metal.framework` и приватный
`MTLCompiler.framework` `dlopen`-ятся во время выполнения, и каждый вызов — это
`objc_msgSend` с вручную объявленной C-сигнатурой (`device/src/metal/objc.rs`).
Нет ни крейта `objc2` или `metal`, ни cargo-фичи, ни гейта `cfg(target_os)`:
модуль компилируется и проходит проверку типов на любом хосте, а машина с
Linux просто проваливает `dlopen` и никогда не регистрирует устройство. Ядра
рендерятся в Metal Shading Language диалектом Metal C-рендерера
(`codegen/src/c/metal.rs`) и компилируются в metallib в процессе.

Код находится в `device/src/metal/` (устройство, аллокатор, компиляция,
программа, граф, профилировщик Metal 4), `runtime/src/devices/metal.rs`
(фабрика устройства) и `codegen/src/c/metal.rs` (диалект).

---

## Статус

Бэкенд появился в сентябре 2026 года и опробован на одном семействе железа:
**Apple9** (класс M3/M4) под macOS 26, где зелёные тестовый набор тензоров,
набор ONNX (`METAL:0` и `CPU` проходят одни и те же 4577 случаев) и
аппаратные тесты `tk`, а flash attention и GEMM работают через
`simdgroup_matrix`. Пути, написанные для старых систем (публичный запасной путь
компиляции `newLibraryWithSource:`, обход для indirect command buffers до
Apple9, языковые стандарты `metal3.x` / `metal2.0`), реализованы, но не
проверены на таком железе. Аппаратные тесты сами себя пропускают, когда
устройства Metal нет, поэтому CI на Linux запускает только тесты на стороне
хоста.

Поддерживается только системное устройство по умолчанию (`METAL:0`); перечисление
через `MTLCopyAllDevices` — задача на будущее (`device/src/metal/device.rs`).

---

## Выбор устройства

`METAL[:N]` — единственное написание (псевдонимы в стиле `HIP` есть для AMD и
CUDA, для Metal их нет). На macOS это ещё и **платформенное устройство по
умолчанию**: `default_device()` разрешается в `METAL:0`, когда устройство ничем
не выбрано, поэтому `SVOD_DEVICE=CPU` — это способ вернуть Mac на CPU-бэкенд
(`dtype/src/default_device.rs`).

```bash
SVOD_DEVICE=METAL:0 cargo run --release -p svod-model --example gigaam_infer -- ./audio.wav
```

`svod_device::metal::has_devices()` загружает рантайм Objective-C и вызывает
`MTLCreateSystemDefaultDevice`; реестр устройств рантайма регистрирует фабрику
`"METAL"`, только если это удалось. Открытие пишет в лог одну строку `info` с
именем устройства и семейством GPU (`RUST_LOG=svod_device=info`).

Семейство определяется через `supportsFamily:` от Apple12 вниз до Apple1, затем
Mac2, и хранится как `MetalFamily { Unknown, Mac2, Apple(n) }`. Это `gpu_arch`
рендерера; оно входит в ключ кэша объектов и выбирает профиль оптимизатора
(`OptimizerRenderer::for_metal_family`): тензорным ядрам `simdgroup_matrix`
нужен Apple7 или новее.

---

## Codegen: диалект MSL

`CRenderer::metal()` — это C-рендерер CPU с `CDialect::Metal`; вывод Clang от
его существования не меняется. Ядро рендерится как

```c
#include <metal_stdlib>
using namespace metal;

kernel void r_64_32(device float* data0, device float* data1, constant int& data2,
                    uint3 gid [[threadgroup_position_in_grid]],
                    uint3 lid [[thread_position_in_threadgroup]]) {
  threadgroup __attribute__((aligned(16))) float local0[32];
  ...
}
```

| Понятие | Диалект Clang | Диалект Metal |
|---|---|---|
| параметр-буфер | `float* restrict data0` | `device float* data0` |
| скалярный параметр | `const int data2` | `constant int& data2` |
| идентификаторы запуска | переменная `core_id` | `gid.xyz` (`gidx*` / `idx*`), `lid.xyz` (`lidx*`), добавляются после списка PARAM |
| локальный буфер | массив на стеке | `threadgroup __attribute__((aligned(16))) T localN[size]` |
| барьер | нет | `threadgroup_barrier(mem_flags::mem_threadgroup)` |
| адресные пространства | нет | `device` / `threadgroup` / `thread` в приведениях указателей |
| 16-битные float | `_Float16` | `half`, `bfloat` |
| bitcast | union / memcpy | `as_type<T>()` |

Атрибутов `[[buffer(n)]]` нет: Metal связывает аргументы **позиционно**, так что
индекс привязки параметра — это его позиция в сигнатуре, и загрузчик это
отражает (см. ниже). Осей сетки не больше трёх; планировщик сворачивает
дополнительные глобальные оси (`global_max` в профиле оптимизатора Metal).

**Типы.** Float64, все форматы fp8 и векторы шире 4 отвергаются при рендеринге
(`reject_unsupported_metal_dtypes`, `codegen/src/c/types.rs`); планировщик
заранее понижает внутренние f64 до f32. Арифметика bf16 повышается через
`float`, а сужение до bf16 использует набор паттернов целочисленного округления
к ближайшему чётному.

**Математика.** `sqrt`, `exp2` и `log2` нативные; `sin` рендерится как
`precise::sin`; `exp`, `log`, `cos`, `tan` и `erf` (в MSL нет `erf`)
декомпозируются общими `amd_decomposition_patterns()` поверх нативных
`exp2`/`log2`, как на AMD. `extra_matcher` рендерера — тот же, что у CPU
(`cpu_extra_matcher()`). Fast math выключен везде (`-fno-fast-math` или
`MTLMathModeSafe` на публичном пути), чтобы держались общие допуски тестов.

**Тензорные ядра.** `Wmma` понижается в хелпер для каждой формы поверх
`simdgroup_<T>8x8` и `simdgroup_multiply_accumulate`: одна форма, 8×8×8 на 32
потоках по два элемента на полосу, для f32→f32, f16→f32, f16→f16, bf16→f32 и
bf16→bf16 (`METAL_888` в профиле оптимизатора). `tk` добавляет билдеры
`simd_shuffle`, `simd_shuffle_xor` и `simdgroup_barrier`
(`codegen/src/c/metal.rs`), из которых и собраны ядра flash attention и GEMM
для Apple.

---

## Путь компиляции

`compile_msl` (`device/src/metal/compile.rs`) отправляет исходник в приватный
`MTLCodeGenService` Apple — тот же путь, что использует tinygrad, — и получает
metallib (magic `MTLB`, трейлер `ENDT`) через вручную собранный
callback-блок Objective-C, с таймаутом 60 с и по одному запросу за раз. Флаги:

```text
-fno-fast-math -std=<std> --driver-mode=metal -x metal -fno-caret-diagnostics
-fmodules-cache-path=<cache>/metal-modules
```

где `<std>` следует мажорной версии macOS (`metal4.0` на 26+, `metal3.1` на
14–25, `metal3.0` на 13, `macos-metal2.0` раньше), а кэш модулей сокращает
разбор `metal_stdlib` примерно с 250 мс примерно до 8 мс.

`MTLCompiler.framework` загружает собственную libLLVM с `RTLD_GLOBAL`, которая не
может сосуществовать с libLLVM CPU-бэкенда в процессе, поэтому оба
соревнуются за один слот (`claim_inprocess_llvm`). Проигравший эту гонку или
система без приватного фреймворка идёт по `compile_msl_public`: одна компиляция
`newLibraryWithSource:options:error:`, чтобы всплыли диагностики, после чего
полезной нагрузкой становится **сам исходник MSL**, и загрузчик программы
компилирует его снова при загрузке. Обе нагрузки делят одну запись кэша
объектов:

```text
backend:             metal
target_architecture: Apple9/air64
toolchain:           macos=26.0
flags:               -fno-fast-math -std=metal4.0 --driver-mode=metal -x metal -fno-caret-diagnostics
abi:                 msl-kernel-abi-v1
object_format:       metallib-or-msl-v1
```

Транспорт намеренно не входит в идентичность: воркер BEAM, выигравший слот
libLLVM, и его родитель, проигравший его, должны сходиться в ключе.

---

## Программы и запуски

`MetalProgram::load` принимает любую из нагрузок (`newLibraryWithData:` для
metallib, `newLibraryWithSource:` для MSL), связывает функцию через
`newFunctionWithName:` и строит пайплайн с
`setSupportIndirectCommandBuffers:YES`, читая `maxTotalThreadsPerThreadgroup`,
`threadExecutionWidth` и `staticThreadgroupMemoryLength`.

Аргументы связываются по позиции: буферы — через `setBuffer:offset:atIndex:`
(указатель хоста разрешается в свою пару `(MTLBuffer, offset)` через
`PointerRegistry` устройства — `BTreeMap` с ключом по базе буфера), скаляры —
через `setBytes` как 4-байтные `i32`; значение вне `i32` — ошибка времени
выполнения. Слоты ABI должны возрастать и могут иметь пропуски, до 31 привязки
(`MAX_BUFFER_BINDINGS`). `global_size` — это число threadgroup, а `local_size` —
число потоков в группе; они передаются через
`dispatchThreadgroups:threadsPerThreadgroup:`; ядро без локальных осей
выполняет один поток на группу, а группа больше
`maxTotalThreadsPerThreadgroup` отвергается.

Каждая диспетчеризация — это один command buffer, помеченный именем ядра, в
единственной очереди глубиной 1024. При `wait = false` он попадает в список
`in_flight` устройства; `MetalDevice::synchronize` ждёт каждую запись через
`waitUntilCompleted` и поднимает первый `NSError`. Доступ хоста к буферу
сначала дренирует устройство. `execute_timed` читает `GPUStartTime` /
`GPUEndTime` command buffer'а, и именно по ним BEAM ранжирует кандидатов.

---

## Память

Каждое выделение — это один `MTLBuffer` в `MTLResourceStorageModeShared`:
Apple silicon имеет унифицированную память, поэтому флаги `BufferSpec`
игнорируются, а `copyin` / `copyout` / `_transfer` — это `memcpy` / `memmove`
хоста над `contents` после `synchronize()`. Нет ни private-, ни
managed-буферов и нет blit-энкодеров. Освобождение сначала дренирует
устройство; если дренирование не удалось, выделение утекает, а не
освобождается из-под работающего ядра.

---

## Графы

`MetalGraph::capture` записывает цепочку ядер в один `MTLIndirectCommandBuffer`
из команд `ConcurrentDispatch`, каждая с `setBarrier`, так что порядок захвата
сохраняется. Воспроизведение — это один command buffer:
`useResources:count:usage:` на связанных буферах, затем
`executeCommandsInBuffer:withRange:`. `replay` ждёт предыдущее воспроизведение
и перепривязывает только слоты, у которых сменился буфер. Захват отказывается
(`Ok(None)`, вместо этого — диспетчеризация по вызовам) для пустой цепочки,
программы, не являющейся `MetalProgram`, устройства, в имени которого есть
«virtual» (паравиртуализированные GPU в CI ломают ICB), смещения больше 32 бит
или **любого скалярного аргумента** — цепочка с символьными формами не
превращается в граф. Ниже Apple9 применяется обход tinygrad `FIX_METAL_ICB`
(одна пустая диспетчеризация на пайплайн).

---

## Профилирование

| Уровень | На Metal | Источник |
|---|---|---|
| 1 — время устройства | да | `GPUStartTime` / `GPUEndTime` на command buffer; внутри графа — counter heap Metal 4 (macOS 26+) или по одному command buffer на ядро |
| 2 — roofline | да | не зависит от бэкенда |
| 3 — статические ресурсы | частично | `lds_bytes` из `staticThreadgroupMemoryLength`, `wave_size` из `threadExecutionWidth`, `occupancy` как `maxTotalThreadsPerThreadgroup / 1024`; числа регистров нет |
| 4 — аппаратные счётчики | нет | |

`Mtl4Profiler` (`device/src/metal/mtl4.rs`) существует только для
профилированного воспроизведения графа: он выполняет каждую косвенную команду
отдельно в command buffer Metal 4 между двумя точными временными метками, с
residency set над связанными буферами и ожиданием shared event. Энкодер MTL4
молча пропускает первое выполнение косвенной команды, чей пайплайн ему не
передали, поэтому все пайплайны сначала устанавливаются на энкодер.

---

## Ограничения

- одно устройство (`METAL:0`), одна очередь команд, только shared-хранилище;
- скаляры — `i32`; нет f64, нет fp8, нет векторов шире 4;
- одна форма тензорных ядер (`simdgroup` 8×8×8);
- графы исключают цепочки со скалярными аргументами;
- нет аппаратных счётчиков, нет числа регистров, а тайминг каждой
  диспетчеризации вне графа — это метка всего command buffer;
- быстрый путь — недокументированный `MTLCodeGenService` Apple; публичный API —
  запасной путь.

Специфичных для Metal переменных окружения нет. Действуют общие:
`SVOD_DEVICE`, `SVOD_OBJECT_CACHE` / `SVOD_OBJECT_CACHE_DIR`, `XDG_CACHE_HOME`
для кэша модулей и `RUST_LOG=svod_device=debug` для сообщений о захвате графа
и отказах.

---

## Тесты

```bash
cargo test -p svod-device metal          # host tests everywhere; hardware tests self-skip
cargo test -p svod-codegen metal         # MSL golden tests
SVOD_DEVICE=METAL:0 cargo test -p svod-tensor   # codegen_tests! `metal` variants
SVOD_DEVICE=METAL:0 cargo test -p svod-onnx
```
