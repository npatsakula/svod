---
sidebar_label: Отладка
---

# Отладка

У бэкенда три поверхности отказов: хостовый тулчейн (clang, кэш объектов), JIT
драйвера (`cuModuleLoadDataEx`) и устройство во время выполнения (ошибки
драйвера, сбои памяти). Эта страница перечисляет рычаги и то, как читать каждую
ошибку.

---

## Переменные окружения

| Переменная | По умолчанию | Эффект |
|---|---|---|
| `SVOD_DEVICE` | `CPU` | `CUDA:N` (алиасы `NV`, `GPU`) выбирает тензорное устройство по умолчанию |
| `SVOD_DUMP_NVPTX_IR` | не задана | Каталог, куда попадает NVPTX LLVM IR каждого ядра в виде `sm_XY_<kernel>.ll` |
| `SVOD_OBJECT_CACHE` | включён | `0` отключает дисковый кэш PTX |
| `SVOD_OBJECT_CACHE_DIR` | `$XDG_CACHE_HOME` / `~/.cache` | Переносит кэш в другое место |
| `SVOD_PROFILE_ITERS`, `SVOD_ORIGIN`, `SVOD_ORIGIN_DEPTH` | | Рычаги профайлера, см. [Профилирование](./profiling.md) |
| `RUST_LOG` | не задана | `svod_device=debug` показывает строку открытия устройства, информационные логи PTX JIT, захват графа и откаты переигрывания; `svod_runtime=debug` показывает каждый вызов clang |

CUDA-специфичного дампа диспетчеризации нет; лог JIT драйвера и `tracing`
покрывают то, что на AMD делает `SVOD_DEBUG_DISPATCH`.

---

## «Нет CUDA-устройства» на хосте, где оно есть

`has_devices()` возвращает `false`, когда библиотека не загружается, отсутствует
привязанный символ, падает `cuInit` или счётчик равен нулю. `CudaDevice::open`
сообщает, что именно:

```text
device unavailable: cannot load libcuda.so.1: ...        # драйвера нет в путях загрузчика
device unavailable: libcuda.so.1 has no symbol cu...     # драйвер слишком стар для привязанной точки входа
no CUDA GPU available: CUDA cuInit failed: ...           # драйвер загружен, пригодного устройства нет
```

Проверьте `ldconfig -p | grep libcuda`, `nvidia-smi` и то, что процесс может
открыть `/dev/nvidia*`.

---

## Чтение сбоя JIT

PTX, который драйвер отвергает, всплывает как `Error::CudaJit`, чей вывод — это
причина, за которой следует лог ошибок драйвера:

```text
CUDA JIT of kernel "r_64_32" failed: CUDA_ERROR_INVALID_PTX (218): a PTX JIT compilation failed
ptxas application ptx input, line 27; error   : ...
```

`CUDA_ERROR_UNSUPPORTED_PTX_VERSION` означает, что драйвер старше, чем PTX ISA
модуля (зафиксирована на 7.8, CUDA 11.8 / R520), см.
[требования](./overview.md). Информационный лог (предупреждения, спиллы
регистров) логируется на уровне `debug` под `svod_device`.

Две ошибки приходят от собственного валидатора Svod ещё до того, как драйвер
что-либо увидит:

```text
PTX references an unresolved function: .extern .func ...   # имя LLVM-интринсика, которое NVPTX-
                                                            # бэкенд не распознал
cached PTX targets sm_80, not sm_86                          # испорченная или чужая запись кэша
```

Первая — важная ловушка: опечатка в интринсике `llvm.nvvm.*` не является ошибкой
clang, она становится внешним вызовом. Исправление — в
`codegen/src/llvm/nvptx/` или в таблице объявлений интринсиков в
`codegen/src/llvm/text/mod.rs`.

---

## Офлайн-проверки с тулкитом

Во время выполнения CUDA-тулкит не нужен нигде, но если он установлен, его
инструменты работают с выгруженным IR:

```bash
SVOD_DUMP_NVPTX_IR=/tmp/nvptx SVOD_DEVICE=CUDA:0 cargo test -p svod-tensor -- some_test

# Воспроизвести ту же компиляцию, затем ассемблировать через ptxas, чтобы увидеть настоящую диагностику
clang -x ir -S -O3 --target=nvptx64-nvidia-cuda -march=sm_86 -Wno-override-module \
      /tmp/nvptx/sm_86_r_64_32.ll -o r_64_32.ptx
ptxas -arch=sm_86 -v r_64_32.ptx -o r_64_32.cubin   # -v печатает регистры, разделяемую память, спиллы
nvdisasm r_64_32.cubin | less                         # это SASS
```

`ptxas -v` — самый быстрый способ увидеть, почему `maxThreadsPerBlock` вышел
меньше, чем хочет запуск: давление на регистры относительно launch bound
`.maxntid`. Соответствующая ошибка времени выполнения называет цифры:

```text
CUDA kernel 'r_64_32' block [512, 1, 1] (512 threads) exceeds its maxThreadsPerBlock 256
  (numRegs 96, sharedSizeBytes 4096, localSizeBytes 0)
```

---

## Ошибки драйвера во время выполнения

Каждый вызов драйвера проверяется и сообщается с собственным именем и текстом
драйвера:

```text
CUDA cuStreamSynchronize failed: CUDA_ERROR_ILLEGAL_ADDRESS (700): an illegal memory access was encountered
```

Сбой ядра асинхронен: `cuLaunchKernel` завершается успешно, а ошибка прилетает
на следующем синхронизирующем вызове (`cuStreamSynchronize`,
`cuEventSynchronize`, `cuCtxSynchronize`, копия на хост). Коды, которые драйвер
документирует как **липкие** (sticky) (`ILLEGAL_ADDRESS`, `LAUNCH_FAILED`,
`ILLEGAL_INSTRUCTION`, `MISALIGNED_ADDRESS`, `ECC_UNCORRECTABLE`, ...),
отравляют устройство: каждый последующий вызов сразу падает с записанным
сообщением, а освобождения помещают свои выделения в карантин вместо того, чтобы
отдавать память, которую зависшее ядро ещё может трогать.

В отличие от AMD-бэкенда, здесь нет реестра VA для классификации сбойного
адреса; драйвер его не раскрывает. Чтобы локализовать сбой, запустите тот же
бинарник под санитайзером из тулкита — он работает с программами на driver API
и с PTX, загруженным через JIT:

```bash
SVOD_DEVICE=CUDA:0 compute-sanitizer --tool memcheck \
  target/release/examples/gigaam_infer ./audio.wav
```

Если результат неверный, а не произошёл сбой, стоит посмотреть на откат
переигрывания графа: `RUST_LOG=svod_device=debug` печатает `CUDA graph replay
with re-aliased buffers; using the capture-order chain`, когда алиасинг буферов
при переигрывании отличается от захвата, а
`SVOD_DEVICE=CUDA:0 cargo test -p svod-tensor` прогоняет варианты `cuda` из
`codegen_tests!` — те же утверждения на уровне тензоров, которые проходят
CPU-бэкенды, по одному ядру за раз.

:::tip Отладчик пайплайна
Для проблем на стороне компилятора (какие UOp-ы породили какой IR) команда
`/svod-debug` документирует таргеты трассировки фронтенда → кодгена;
`SVOD_DUMP_NVPTX_IR` — это CUDA-член того же семейства рядом с
`SVOD_DUMP_AMD_IR` и `SVOD_DUMP_LLVM_IR`.
:::
