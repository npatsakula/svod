---
sidebar_label: Обзор
---

# Бэкенды

Бэкенд — это всё, что лежит ниже отрендеренного ядра: рендерер, превращающий
UOp IR в исходный код, компилятор, превращающий исходник в объект, загрузчик,
превращающий объект в вызываемую `Program`, аллокатор и, опционально, граф.
Svod поставляет четыре бэкенда, все в одном бинарнике; какие из них существуют
на конкретном хосте, решается во время выполнения.

| Устройство | Железо | Рендерер | Путь компиляции | Воспроизведение графа | Статус |
|---|---|---|---|---|---|
| [`CPU`](./cpu.md) | x86_64, aarch64, riscv64, loongarch64, ppc64le | Текст LLVM IR (по умолчанию) или C | libLLVM в процессе, иначе `clang -c`; [ELF-загрузчик в памяти](./jit-loader.md) | нет (синхронные вызовы) | в эксплуатации |
| [`AMD:N`](./amd/overview.md) | CDNA3, RDNA3, RDNA3.5, RDNA4 (Linux, KFD) | Текст LLVM IR, таргет AMDGPU | `clang` с таргетом `amdgcn` → ELF code object, загружаемый в VRAM | командный поток AQL (PM4 по запросу) | в эксплуатации |
| [`CUDA:N`](./cuda/overview.md) | NVIDIA, драйвер CUDA 12.0+ | Текст LLVM IR, таргет NVPTX | `clang` с таргетом NVPTX → PTX, `ptxas`, если установлен, иначе JIT драйвера | CUDA-графы | в эксплуатации |
| [`METAL:N`](./metal.md) | GPU Apple | C, диалект Metal | Apple `MTLCodeGenService` в процессе → metallib | indirect command buffers | проверен на Apple9 / macOS 26 |

Скомпилированные объекты всех бэкендов проходят через один дисковый кэш
объектов с ключом из исходника и `CompilerIdentity` конкретного бэкенда
([страница CPU](./cpu.md)).

---

## Выбор устройства

`SVOD_DEVICE` задаёт устройство по умолчанию для тензоров и ядер. Значение
разбирается без учёта регистра как `NAME[:N]` (`dtype/src/default_device.rs`):

| Значение | Устройство |
|---|---|
| `CPU` | `DeviceSpec::Cpu` |
| `AMD[:N]`, `HIP[:N]` | `DeviceSpec::Amd { device_id }` — N-й GPU-узел топологии KFD |
| `CUDA[:N]`, `GPU[:N]` | `DeviceSpec::Cuda { device_id }` |
| `METAL[:N]` | `DeviceSpec::Metal { device_id }` (существует только `0`) |

`NAME` без номера — это устройство 0. `NV` отвергается намеренно — это имя
зарезервировано для будущего userspace-драйвера NVIDIA. Если устройство ничем
не выбрано, действует платформенное значение по умолчанию: **`METAL:0` на
macOS, `CPU` везде остальном**. Полный порядок приоритета: область
`with_default_device`, затем thread-local `set_default_device`, затем
`SVOD_DEVICE` (читается один раз за процесс), затем платформенное значение по
умолчанию. Архитектура GPU никогда не входит в спецификацию: это свойство
открытого устройства, поэтому у одного физического GPU одна идентичность, а
кэш ядер ключуется тем, что сообщает устройство.

`DeviceSpecExt::parse` в `svod-device` принимает те же написания, а также
`DISK:<path>` (read-only устройство над файлом, отображённым в память, которое
не умеет запускать ядра) и `WEBGPU`, у которого пока нет аллокатора и который
завершается ошибкой `DeviceUnavailable`.

---

## Регистрация, определяемая во время выполнения

Каждый бэкенд компилируется на каждом хосте — нет cargo-фичи ни для AMD, ни для
CUDA, ни для Metal (модули AMD, работающие с ядром ОС, и привязки Objective-C и
драйверов для Metal/CUDA либо находятся под `cfg(unix)`, либо являются обычным
Rust поверх `libloading`, поэтому `cargo check` на Linux или macOS проверяет
типы во всех них). Будет ли бэкенд *доступен*, решается при первом обращении к
реестру фабрик устройств (`runtime/src/device_registry.rs`):

```rust
registry.register_factory("CPU", ...);                        // always
if svod_device::amd::has_devices()   { registry.register_factory("AMD", ...); }
if svod_device::metal::has_devices() { registry.register_factory("METAL", ...); }
if svod_device::cuda::has_devices()  { registry.register_factory("CUDA", ...); }
```

Каждая проба не имеет побочных эффектов и мемоизируется: AMD читает топологию
KFD из sysfs и проверяет, есть ли узел поддерживаемой архитектуры; Metal
`dlopen`-ит фреймворки Apple и запрашивает системное устройство по умолчанию;
CUDA загружает `libcuda.so.1`, привязывает все используемые точки входа,
вызывает `cuInit` и считает устройства. У хоста без соответствующего железа
такого типа устройства просто нет, и запрос его завершается ошибкой
`UnsupportedDevice`. Смысл компиляции всего везде в том, что изменение общих
трейтов `Program` / `PlanContext` / `Graph` ломает сборку на любой машине
разработчика, а не только на той, где есть GPU.

Реестр кэширует один `Device` на `DeviceSpec` (`DEVICE_FACTORIES`);
конструирование — открытие KFD, проба тулчейна — выполняется вне блокировок
карты, сериализованно для каждой спецификации, а неудачное конструирование
оставляет слот пустым для повторной попытки. Аллокаторы живут в отдельном
реестре в `svod-device` (`registry::registry()`), где каждый вычислительный
аллокатор обёрнут в `LruAllocator`, который пулит освобождённые буферы по
размеру и спецификации.

---

## Что реализует бэкенд

`Device` (`device/src/device.rs`) состоит из пяти частей:

```rust
pub struct Device {
    pub device: DeviceSpec,
    pub allocator: Arc<dyn Allocator>,
    pub compilers: Vec<CompilerPair>,     // (Arc<dyn Renderer>, Arc<dyn Compiler>)
    pub renderer: Arc<dyn Renderer>,
    pub compiler: Arc<dyn Compiler>,
    pub runtime: RuntimeFactory,          // Fn(&CompiledSpec) -> Result<Box<dyn Program>>
    pub graph: Option<GraphFactory>,      // Fn(&[GraphKernel]) -> Result<Option<Box<dyn Graph>>>
}
```

| Трейт | Обязательные | Роль |
|---|---|---|
| `Renderer` | `render`, `device`, `supported_ops` | граф UOp → `ProgramSpec` (исходник, точка входа, ABI, размеры запуска). `gpu_arch` выбирает профиль оптимизатора; `decompositor` и `extra_matcher` понижают то, что таргет не умеет выбрать |
| `Compiler` | `compile`, `cache_key` | `ProgramSpec` → байты `CompiledSpec`; `cache_key` — это `CompilerIdentity`, ключующий кэш объектов |
| `RuntimeFactory` | — | загружает `CompiledSpec` в `Program`; `Device::new` оборачивает её так, чтобы идентичность стадии каждой спецификации сначала проверялась |
| `Program` | `execute`, `name` | один запуск ядра; `execute_timed` (длительность по часам GPU для BEAM), `new_exec_context`, `resource_usage` и `as_any` опциональны |
| `PlanContext` | `dispatch`, `synchronize` | состояние плана, создаваемое `Program::new_exec_context`: полосы, токены завершения, временные метки, счётчики (`set_pmc`), нативное связанное воспроизведение (`replay_linked_plan`) |
| `Allocator` | `_alloc`, `name`, `device_spec` | `_copyin` / `_copyout` / `_transfer` / `_free` / `synchronize` / `supports_device_local` опциональны и по умолчанию имеют семантику памяти хоста |
| `Graph` | `replay` | захваченная цепочка ядер, воспроизводимая одной отправкой; `completion_token`, `replay_profiled` опциональны |
| `CompletionToken`, `TimelineSignal`, `DispatchTimestamps` | | дескрипторы синхронизации и профилирования, которые потребляет исполнитель (`device/src/sync.rs`) |

Соглашение о запуске общее для всех GPU-бэкендов: `global_size` — это сетка в
рабочих группах, `local_size` — рабочая группа в потоках; CPU использует
`global_size[0]` как разбиение по `core_id`. Аргументы ядра — это слоты `PARAM`
ABI по порядку — сначала указатели, затем скаляры `i32`, — и каждый загрузчик
упаковывает их одинаково (`ClikeKernargLayout` на AMD и CUDA, позиционные
`setBuffer`/`setBytes` на Metal, CIF libffi на CPU).

### Добавление бэкенда

Четыре фабрики в `runtime/src/devices/` служат шаблоном. Каждая
`create_*_device` делает одни и те же пять вещей:

1. получает аллокатор из реестра для своего `DeviceSpec`;
2. строит обёртку рендерера вокруг точки входа codegen
   (`LlvmTextRenderer::amd(arch)`, `LlvmTextRenderer::nvptx(arch)`,
   `CRenderer::metal()` или рендереры CPU), объявляя `supported_ops`,
   паттерны декомпозиции и `gpu_arch`;
3. строит компилятор с `CompilerIdentity` и `ObjectCache`, выдающий байты,
   которые загрузчик может проверить (`validate_amd_object`, `validate_ptx` /
   `validate_cubin`, `validate_metallib`, ELF-проверки на CPU);
4. устанавливает `RuntimeFactory`, загружающую эти байты в `Program` бэкенда;
5. опционально вызывает `with_graph(...)` для захвата/воспроизведения.

Затем фабрика регистрируется под строкой своего типа устройства, с проверкой
пробой `has_devices()`, а планировщику нужен профиль оптимизатора для нового
таргета (`OptimizerRenderer::for_*`: размер волны, формы тензорных ядер,
лимиты разделяемой памяти и local). `create_*_codegen` существует отдельно у
каждого бэкенда, чтобы воркеры BEAM могли рендерить и компилировать, не
открывая устройство.
